"""
Reviewer-requested experiments (NOT the original recipe).

GenericPIVA covers the 2x2 ablation {diagonal, Cholesky} x {softmax v (3 latent dims), ILR v (2 latent dims)}:
  diag + softmax  == PIVA_Diagonal          (original)
  chol + ILR      == PIVA_CholeskyILR       (original)
  diag + ILR      new (experiment B)
  chol + softmax  new (experiment B)
Encoder, heads, bounds, clamps are the same as the original classes.

train_piva_gauss: experiment A. Loss per voxel = sum_k (recon_k - target_k)^2 / (2 s^2) + KL,
averaged over the batch (a proper Gaussian NLL with fixed SD s in signal units, beta = 1).
This equals (16 / (2 s^2)) * [MSE + (2 s^2 / 16) * KL], i.e. the original loss with
beta = 2 s^2 / 16, rescaled by a constant. Adam is nearly invariant to that constant, but
grad-norm clipping at 1.0 and the L2 weight decay are not, so this is not bit-equivalent
to setting beta in the original loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import piva_core as pc


class GenericPIVA(nn.Module):
    def __init__(self, cov="diag", vmap="softmax", n_signals=16, hidden_dims=None, predictor_depth=3):
        super().__init__()
        self.cov, self.vmap = cov, vmap
        self.latent_dim = 8 if vmap == "ilr" else 9
        hidden_dims = hidden_dims or [64, 128, 256, 512, 512]
        self.encoder = pc._mlp(n_signals, hidden_dims)
        self.fc_mu = pc._head(hidden_dims[-1], predictor_depth, self.latent_dim)
        k = self.latent_dim
        if cov == "diag":
            self.fc_logvar = pc._head(hidden_dims[-1], predictor_depth, k)
        else:
            self.fc_L_raw = pc._head(hidden_dims[-1], predictor_depth, k * (k + 1) // 2)
            tril_idx = torch.tril_indices(k, k)
            self.register_buffer("tril_row", tril_idx[0])
            self.register_buffer("tril_col", tril_idx[1])
            self.register_buffer("diag_mask", tril_idx[0] == tril_idx[1])
        self.register_buffer("D_mean", torch.tensor(pc.D_MEAN_3))
        self.register_buffer("T2_mean", torch.tensor(pc.T2_MEAN_3))
        self.register_buffer("D_delta", torch.tensor(pc.D_DELTA_3))
        self.register_buffer("T2_delta", torch.tensor(pc.T2_DELTA_3))
        self.register_buffer("Psi", pc.PSI_ILR.clone())

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        if self.cov == "diag":
            return mu, self.fc_logvar(h).clamp(-8.0, 4.0)
        L_raw = self.fc_L_raw(h)
        k = self.latent_dim
        L = torch.zeros(x.shape[0], k, k, dtype=x.dtype)
        vals = L_raw.clone()
        vals[:, self.diag_mask] = torch.exp(L_raw[:, self.diag_mask].clamp(-4.0, 2.0))
        L[:, self.tril_row, self.tril_col] = vals
        return mu, L

    def reparameterize(self, mu, second, sample=True):
        if not sample:
            return mu
        eps = torch.randn_like(mu)
        if self.cov == "diag":
            return mu + eps * torch.exp(0.5 * second)
        return mu + torch.bmm(second, eps.unsqueeze(-1)).squeeze(-1)

    def params_from_z(self, z):
        D = torch.clamp(self.D_mean + self.D_delta * torch.tanh(z[:, 0:3]), min=0.01, max=10.0)
        T2 = torch.clamp(self.T2_mean + self.T2_delta * torch.tanh(z[:, 3:6]), min=1.0, max=2000.0)
        if self.vmap == "ilr":
            v = F.softmax(z[:, 6:8] @ self.Psi.T, dim=1)
        else:
            v = F.softmax(z[:, 6:9], dim=1)
        return D, T2, v

    def forward(self, x, sample=True):
        mu, second = self.encode(x)
        D, T2, v = self.params_from_z(self.reparameterize(mu, second, sample))
        return pc.physics_decode(D, T2, v), D, T2, v, mu, second

    def kl(self, mu, second):
        return pc.kl_divergence_diag(mu, second) if self.cov == "diag" else pc.kl_divergence_full(mu, second)


def rician_nll(x, nu, sd):
    """-log Rice(x | nu, sd), summed over signals. log I0(z) = log(i0e(z)) + z for stability."""
    s2 = sd ** 2
    x = x.clamp_min(1e-6)
    z = x * nu / s2
    return (-torch.log(x / s2) + (x ** 2 + nu ** 2) / (2 * s2) - (torch.log(torch.special.i0e(z)) + z)).sum(dim=1)


def training_sigmas(n_total):
    """Per-voxel noise SD (unit scale) used by pc.build_training_set().

    augment_clean_signals() draws sigma ~ U(CLINICAL_NOISE_SIGMA_RANGE) as the first call on
    default_rng(1000 + rep), rep = 0, 1, so replaying that draw recovers each voxel's SD exactly.
    """
    n = n_total // 2
    lo, hi = pc.CLINICAL_NOISE_SIGMA_RANGE
    return np.concatenate([np.random.default_rng(1000 + rep).uniform(lo, hi, n).astype(np.float32)
                           for rep in range(2)])


def train_piva_generic(model, loader, n_epochs=200, lr=5e-4, beta=0.1, lik_sd=None, target="clean",
                       lik="gauss", log=print, label="PIVA"):
    """
    lik_sd=None : original loss MSE(recon, target) + beta*KL (beta ignored otherwise).
    lik_sd=s    : Gaussian NLL sum_k (recon-target)^2/(2 s^2) + KL.
    target      : 'clean' (original) or 'noisy' (reconstruct the observed input).
    lik         : 'gauss' or 'rician' (Rician NLL of the noisy target given recon, SD lik_sd; needs target='noisy').
    lik_sd='pervoxel' : Gaussian NLL with each voxel's own simulated noise SD. The loader's third
                  tensor must then hold the unit-scale per-voxel sigma instead of y (y is unused here).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        mse_l, kl_l = [], []
        for xb, clean_b, third in loader:
            tgt = clean_b if target == "clean" else xb
            opt.zero_grad()
            recon, D, T2, v, mu, second = model(xb, sample=True)
            kl = model.kl(mu, second)
            if lik_sd == "pervoxel":
                sd = third.view(-1, 1) * pc.SIGNAL_SCALE
                sq = (recon - tgt) ** 2
                sig_mse = sq.mean()
                loss = (sq / (2.0 * sd ** 2)).sum(dim=1).mean() + kl
            elif lik_sd is None:
                sig_mse = F.mse_loss(recon, tgt)
                loss = sig_mse + beta * kl
            elif lik == "rician":
                sig_mse = F.mse_loss(recon, tgt)
                loss = rician_nll(tgt, recon, lik_sd).mean() + kl
            else:
                sq = (recon - tgt) ** 2
                sig_mse = sq.mean()
                loss = (sq.sum(dim=1) / (2.0 * lik_sd ** 2)).mean() + kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            mse_l.append(sig_mse.item())
            kl_l.append(kl.item())
        hist.append({"epoch": epoch, "signal_mse": float(np.mean(mse_l)), "kl": float(np.mean(kl_l))})
        if epoch == 1 or epoch % 20 == 0 or epoch == n_epochs:
            log(f"{label} | epoch {epoch:03d} | signal_MSE={hist[-1]['signal_mse']:.2f} | KL={hist[-1]['kl']:.3f}")
    return hist
