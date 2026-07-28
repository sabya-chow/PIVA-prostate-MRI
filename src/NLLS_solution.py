"""
NLLS Baseline: Parallelized voxel-wise bi-exponential fitting.
Evaluates non-linear least squares against ground truth IVIM parameters.
"""
import numpy as np
from utils import fit_biExponential_model, read_data, rRMSE_per_case
from concurrent.futures import ProcessPoolExecutor
import os

file_dir = '../data/'
fname_gt = '_IVIMParam.npy'
fname_gtDWI = '_gtDWIs.npy'
fname_tissue = '_TissueType.npy'
fname_noisyDWIk = '_NoisyDWIk.npy'
file_Resultdir = '../results/nlls/'

if not os.path.isdir(file_Resultdir):
    os.makedirs(file_Resultdir)

num_cases = 1000
rRMSE_case = np.empty([num_cases])
rRMSE_t_case = np.empty([num_cases])
b = np.array([0, 5, 50, 100, 200, 500, 800, 1000])


def fit_one_case(image_number):
    k = read_data(file_dir, fname_noisyDWIk, image_number + 1)
    noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))
    return fit_biExponential_model(noisy, b)


if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(fit_one_case, range(num_cases)))

    for image_number in range(num_cases):
        np.save(f'{file_Resultdir}/{(image_number + 1):04d}.npy', results[image_number])
        params = read_data(file_dir, fname_gt, image_number + 1)
        tissue = read_data(file_dir, fname_tissue, image_number + 1)
        rRMSE_case[image_number], rRMSE_t_case[image_number] = rRMSE_per_case(
            results[image_number][:, :, 0],
            results[image_number][:, :, 1],
            results[image_number][:, :, 2],
            params[:, :, 0],
            params[:, :, 1],
            params[:, :, 2],
            tissue
        )
        print(f'RMSE ALL = {rRMSE_case[image_number]}\nRMSE tumor = {rRMSE_t_case[image_number]}')

    rRMSE_final_1 = np.average(rRMSE_case)
    rRMSE_final_tumor_1 = np.average(rRMSE_t_case)
    print('-----')
    print(f'Total RMSE all {rRMSE_final_1}\nRMSE tumor {rRMSE_final_tumor_1}')