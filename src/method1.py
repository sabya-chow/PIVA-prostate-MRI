"""
Supervised Solution for the IVIM Quantitative Analysis Case
Authors: Batuhan Gundogdu
April, 2024
"""
from utils import read_data, rRMSE_per_case
import numpy as np
import torch
from model import PIAEncoder
from torch import optim


number_of_epochs = 500
file_dir = '../data/'
fname_gt = '_IVIMParam.npy'
fname_gtDWI = '_gtDWIs.npy'
fname_tissue = '_TissueType.npy'
fname_noisyDWIk = '_NoisyDWIk.npy'
file_Resultdir = '../results/supervised/'

model = PIAEncoder()

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

params = model.parameters()
lr = 3e-4
optimizer = optim.Adam(params, lr=lr)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fcn = torch.nn.MSELoss()
model = model.train()

b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

counter = 0
for ep in range(number_of_epochs):
    for image_number in range(950):
        # Using the first 950 samples for training
        params = read_data(file_dir, fname_gt, image_number + 1)
        tissue = read_data(file_dir, fname_tissue, image_number + 1)
        coordBody = np.argwhere(tissue != 1)
        k = read_data(file_dir, fname_noisyDWIk, image_number + 1)
        noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))

        for pix in range(coordBody.shape[0]):
            ix, jx = coordBody[pix, 0], coordBody[pix, 1]
            f_true, D_true, D_star_true = torch.from_numpy(params[ix, jx, :]).to(device).float()
            D_true = D_true.unsqueeze(0) * 1000
            D_star_true = D_star_true.unsqueeze(0) * 1000
            f_true = f_true.unsqueeze(0)
            signal = noisy[ix, jx, :] / noisy[ix, jx, 0]
            signal = torch.from_numpy(signal).to(device).float()
            f, D, D_star = model.module.encode(signal) if isinstance(model, torch.nn.DataParallel) else model.encode(signal)
            loss = loss_fcn(D, D_true) + loss_fcn(f, f_true) + loss_fcn(D_star, D_star_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tumor, non_tumor = 0, 0
        ctr = 0
        print('Testing')
        for image_number in range(950, 1000):
            params = read_data(file_dir, fname_gt, image_number + 1)
            tissue = read_data(file_dir, fname_tissue, image_number + 1)
            k = read_data(file_dir, fname_noisyDWIk, image_number + 1)
            noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))
            samples = noisy.reshape(-1, 8)
            samples = torch.from_numpy(samples / samples[:, 0, np.newaxis]).to(device).float()
            f, D, D_star = model.module.encode(samples) if isinstance(model, torch.nn.DataParallel) else model.encode(samples)
            f = f.detach().cpu().numpy().reshape(200, 200)
            D = D.detach().cpu().numpy().reshape(200, 200) / 1000
            D_star = D_star.detach().cpu().numpy().reshape(200, 200) / 1000
            t, nt = rRMSE_per_case(f, D, D_star, params[:, :, 0], params[:, :, 1], params[:, :, 2], tissue)
            tumor += t
            non_tumor += nt
            ctr += 1
        counter += 1
        print(counter, tumor / ctr, non_tumor / ctr)
        PATH = f'../checkpoints/pia_model_{tumor/ctr:.2f}_{counter}.pt'
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), PATH)
        else:
            torch.save(model.state_dict(), PATH)