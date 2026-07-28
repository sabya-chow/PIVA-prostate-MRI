"""
PIA Model for IVIM

Self-supervised physics-informed autoencoder for IVIM parameter estimation.
Uses an MLP encoder with physics-based decoding via the bi-exponential model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class PIA(nn.Module):
    def __init__(self,
                 number_of_signals=8,
                 f_mean=0.2,
                 Dt_mean=0.0011,
                 Dstar_mean=0.025,
                 f_delta=0.15,
                 Dt_delta=0.0004,
                 Dstar_delta=0.015,
                 b_values=[0, 5, 50, 100, 200, 500, 800, 1000],
                 hidden_dims: List = None,
                 predictor_depth=1,
                 device='cpu'):
        super(PIA, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.number_of_signals = number_of_signals
        self.f_mean = torch.tensor(f_mean, device=device)
        self.Dt_mean = torch.tensor(Dt_mean, device=device)
        self.Dstar_mean = torch.tensor(Dstar_mean, device=device)
        self.f_delta = torch.tensor(f_delta, device=device)
        self.Dt_delta = torch.tensor(Dt_delta, device=device)
        self.Dstar_delta = torch.tensor(Dstar_delta, device=device)
        self.b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        modules = []
        # Build The Encoder
        in_channels = number_of_signals
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules).to(device)

        # f predictor
        f_predictor = []
        for _ in range(predictor_depth):
            f_predictor.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU()))
        f_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.f_predictor = nn.Sequential(*f_predictor).to(device)

        # Dt predictor
        Dt_predictor = []
        for _ in range(predictor_depth):
            Dt_predictor.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU()))
        Dt_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.Dt_predictor = nn.Sequential(*Dt_predictor).to(device)

        # D* predictor
        star_predictor = []
        for _ in range(predictor_depth):
            star_predictor.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU()))
        star_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.Dstar_predictor = nn.Sequential(*star_predictor).to(device)

    def encode(self, x):
        result = self.encoder(x)

        f_var = self.f_delta * torch.tanh(self.f_predictor(result))
        f = self.f_mean + f_var

        Dt_var = self.Dt_delta * torch.tanh(self.Dt_predictor(result))
        Dt = self.Dt_mean + Dt_var

        Dstar_var = self.Dstar_delta * torch.tanh(self.Dstar_predictor(result))
        Dstar = self.Dstar_mean + Dstar_var

        return f, Dt, Dstar

    def decode(self, f, Dt, Dstar):
        f = f.view(-1, 1)
        Dt = Dt.view(-1, 1)
        Dstar = Dstar.view(-1, 1)

        b = self.b_values.view(1, -1)  # s/mm^2 native units

        S = (1 - f) * torch.exp(-b * Dt) + f * torch.exp(-b * Dstar)
        return S.to(self.device)

    def forward(self, x):
        f, Dt, Dstar = self.encode(x)
        S_pred = self.decode(f, Dt, Dstar)
        return S_pred, x, f, Dt, Dstar

    def loss_function(self, pred_signal, true_signal, weights=None):
        if weights is not None:
            loss = torch.mean(weights * (pred_signal - true_signal) ** 2)
        else:
            loss = F.mse_loss(pred_signal, true_signal)
        return loss
