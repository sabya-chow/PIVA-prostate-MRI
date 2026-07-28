"""
PIAEncoder: Supervised MLP encoder for IVIM parameter prediction.
Used as the supervised baseline (Model-1) for comparison.
"""
from torch import nn
import torch


class PIAEncoder(nn.Module):

    def __init__(self, b_values=[0, 5, 50, 100, 200, 500, 800, 1000],
                 hidden_dims=[32, 64, 128, 256, 512],
                 predictor_depth=2):
        super(PIAEncoder, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.number_of_signals = len(b_values)
        self.sigmoid = nn.Sigmoid()
        modules = []
        in_channels = self.number_of_signals
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=h_dim),
                nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules).to(self.device)

        D_predictor = []
        for _ in range(predictor_depth):
            D_predictor.append(nn.Sequential(
                nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                nn.LeakyReLU()))
        D_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.D_predictor = nn.Sequential(*D_predictor).to(self.device)

        D_star_predictor = []
        for _ in range(predictor_depth):
            D_star_predictor.append(nn.Sequential(
                nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                nn.LeakyReLU()))
        D_star_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.D_star_predictor = nn.Sequential(*D_star_predictor).to(self.device)

        f_predictor = []
        for _ in range(predictor_depth):
            f_predictor.append(nn.Sequential(
                nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                nn.LeakyReLU()))
        f_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.f_predictor = nn.Sequential(*f_predictor).to(self.device)

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder
        :return: (Tensor) Tuple of (f, D, D_star) predictions
        """
        result = self.encoder(x)
        D = self.D_predictor(result)
        D_star = 5 + self.D_star_predictor(result)
        f = self.sigmoid(self.f_predictor(result))
        return f, D, D_star
