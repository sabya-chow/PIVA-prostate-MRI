"""
piva_core.py: headless copy of the PIVA repo code used for the ISMRM 2027 audit rerun.

Source: PIVA-prostate-MRI @ 3933f7e
  - NB1 = notebooks/PIVA_comparative.ipynb
  - NB2 = notebooks/PIVA_comparative_full_covariance_v2_0.ipynb

Everything in sections 1-4 is copied verbatim from the notebooks (comments trimmed,
plotting removed). The only changes are:
  * seeding helpers (set_all_seeds, a seeded torch.Generator for the DataLoader)
  * the decoder is a module-level function instead of `pia_encoder.decode` (the
    notebook PIVA classes call the decode method of a global PIA instance; decode
    uses no learnable parameters, so this is numerically identical)
  * device is fixed to CPU and batch size fixed to 2048 (the value NB1/NB2 use on CUDA,
    which is how the repo outputs were produced; on CPU the notebooks silently fall back
    to 512)
Section 5 onward is new evaluation code.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cpu")
BATCH_SIZE = 2048

# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants and data generator (NB1 cell [2] + [6] == NB2 cell [1] + [3])
# ─────────────────────────────────────────────────────────────────────────────
B_VALS = [0, 150, 1000, 1500]
TE_VALS = [0, 13, 93, 143]
N_SIGNALS = len(B_VALS) * len(TE_VALS)
b_grid = np.array([[b] * len(TE_VALS) for b in B_VALS], dtype=float).ravel()
te_grid = np.array(TE_VALS * len(B_VALS), dtype=float)
b_TE_flat = np.vstack([b_grid, te_grid])

COMP_NAMES = ["epithelium", "stroma", "lumen"]
N_COMP = 3
SIGNAL_SCALE = 1000.0
CLINICAL_NOISE_SIGMA_RANGE = (0.01, 0.08)

D_MEAN_3 = np.array([0.5, 1.2, 2.85], dtype=np.float32)
T2_MEAN_3 = np.array([45.0, 70.0, 750.0], dtype=np.float32)
D_DELTA_3 = np.array([0.2, 0.5, 0.15], dtype=np.float32)
T2_DELTA_3 = np.array([25.0, 30.0, 250.0], dtype=np.float32)
D_LOW_3 = D_MEAN_3 - D_DELTA_3
D_HIGH_3 = D_MEAN_3 + D_DELTA_3
T2_LOW_3 = T2_MEAN_3 - T2_DELTA_3
T2_HIGH_3 = T2_MEAN_3 + T2_DELTA_3
D_VOXEL_SD_3 = np.array([0.04, 0.12, 0.03], dtype=np.float32)
T2_VOXEL_SD_3 = np.array([6.0, 8.0, 70.0], dtype=np.float32)

PARAM_NAMES = ["D_ep", "D_st", "D_lum", "T2_ep", "T2_st", "T2_lum", "v_ep", "v_st", "v_lum"]
FULL_RANGE = np.concatenate([2 * D_DELTA_3, 2 * T2_DELTA_3, np.ones(3, dtype=np.float32)])
PRIOR_CENTRE = np.concatenate([D_MEAN_3, T2_MEAN_3, np.full(3, 1 / 3, dtype=np.float32)])
RANGE_LO = np.concatenate([D_LOW_3, T2_LOW_3, np.zeros(3, dtype=np.float32)])
RANGE_HI = np.concatenate([D_HIGH_3, T2_HIGH_3, np.ones(3, dtype=np.float32)])


def pia3_signal_np(D, T2, v, scale=SIGNAL_SCALE):
    D = np.asarray(D, dtype=np.float32)
    T2 = np.asarray(T2, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    b = b_grid[None, :, None].astype(np.float32)
    te = te_grid[None, :, None].astype(np.float32)
    comp_signal = v[:, None, :] * np.exp(-b / 1000.0 * D[:, None, :]) * np.exp(-te / T2[:, None, :])
    return scale * comp_signal.sum(axis=2).astype(np.float32)


def generate_patient_clean_cohort(n_patients=80, voxels_per_patient=512, seed=0):
    rng = np.random.default_rng(seed)
    D_all, T2_all, v_all, patient_ids = [], [], [], []
    for pid in range(n_patients):
        n = voxels_per_patient
        patient_mix = rng.dirichlet(alpha=np.array([1.6, 1.8, 1.6]))
        v_alpha = 1.0 + 18.0 * patient_mix
        v = rng.dirichlet(v_alpha, size=n).astype(np.float32)
        d_shift = rng.normal(0, [0.025, 0.060, 0.020]).astype(np.float32)
        t2_shift = rng.normal(0, [4.0, 6.0, 55.0]).astype(np.float32)
        patient_D_mean = np.clip(D_MEAN_3 + d_shift, D_LOW_3, D_HIGH_3).astype(np.float32)
        patient_T2_mean = np.clip(T2_MEAN_3 + t2_shift, T2_LOW_3, T2_HIGH_3).astype(np.float32)
        D = rng.normal(patient_D_mean, D_VOXEL_SD_3, size=(n, N_COMP)).astype(np.float32)
        D = np.clip(D, D_LOW_3, D_HIGH_3).astype(np.float32)
        T2 = rng.normal(patient_T2_mean, T2_VOXEL_SD_3, size=(n, N_COMP)).astype(np.float32)
        T2 = np.clip(T2, T2_LOW_3, T2_HIGH_3).astype(np.float32)
        D_all.append(D)
        T2_all.append(T2)
        v_all.append(v)
        patient_ids.append(np.full(n, pid, dtype=np.int32))
    D = np.vstack(D_all).astype(np.float32)
    T2 = np.vstack(T2_all).astype(np.float32)
    v = np.vstack(v_all).astype(np.float32)
    patient_ids = np.concatenate(patient_ids)
    S_clean = pia3_signal_np(D, T2, v)
    y_params = np.concatenate([D, T2, v], axis=1).astype(np.float32)
    return S_clean.astype(np.float32), D, T2, v, y_params, patient_ids


def augment_clean_signals(S_clean, noise_sigma_range=CLINICAL_NOISE_SIGMA_RANGE, seed=0):
    rng = np.random.default_rng(seed)
    S_unit = S_clean.astype(np.float32) / SIGNAL_SCALE
    n = S_unit.shape[0]
    sigma = rng.uniform(noise_sigma_range[0], noise_sigma_range[1], n).astype(np.float32)
    e_real = rng.normal(0, 1, S_unit.shape).astype(np.float32) * sigma[:, None]
    e_imag = rng.normal(0, 1, S_unit.shape).astype(np.float32) * sigma[:, None]
    X_noisy = SIGNAL_SCALE * np.sqrt((S_unit + e_real) ** 2 + e_imag ** 2)
    return X_noisy.astype(np.float32)


def build_training_set():
    """NB1 cell [6]: 80 patients x 512 voxels (seed 0), bootstrap to 120 (seed 123), 2 noise draws."""
    S_clean_base, _, _, _, y_base, pid_base = generate_patient_clean_cohort(80, 512, seed=0)
    rng_boot = np.random.default_rng(seed=123)
    boot_patient_ids = rng_boot.choice(np.arange(80), size=120, replace=True)
    boot_indices = np.concatenate([np.where(pid_base == pid)[0] for pid in boot_patient_ids])
    S_boot, y_boot = S_clean_base[boot_indices], y_base[boot_indices]
    X_parts, S_parts, y_parts = [], [], []
    for rep in range(2):
        X_parts.append(augment_clean_signals(S_boot, CLINICAL_NOISE_SIGMA_RANGE, seed=1000 + rep))
        S_parts.append(S_boot)
        y_parts.append(y_boot)
    return (np.vstack(X_parts).astype(np.float32), np.vstack(S_parts).astype(np.float32),
            np.vstack(y_parts).astype(np.float32))


def add_rician_noise(S_clean_unit, sigma, seed):
    rng = np.random.default_rng(seed)
    e_real = rng.normal(0, sigma, S_clean_unit.shape).astype(np.float32)
    e_imag = rng.normal(0, sigma, S_clean_unit.shape).astype(np.float32)
    return SIGNAL_SCALE * np.sqrt((S_clean_unit + e_real) ** 2 + e_imag ** 2).astype(np.float32)


SIGMA_TEST_LEVELS = [0.01, 0.03, 0.05, 0.08]
COHORTS = {
    # name: (n_patients, voxels_per_patient, cohort seed, noise seed base)
    "orig240": (24, 10, 999, 7000),      # NB1 cell [10]: noise seed 7000 + index into SIGMA_TEST_LEVELS
    "large2000": (200, 10, 2027, 17000),  # new: larger cohort from the same generator
    # separate calibration cohorts for experiment C (never used as test data)
    "calib240": (24, 10, 31337, 27000),
    "calib2000": (200, 10, 4242, 37000),
}


def build_test_cohort(name):
    n_pat, vpp, seed, noise_base = COHORTS[name]
    S_clean, D, T2, v, y, pids = generate_patient_clean_cohort(n_pat, vpp, seed=seed)
    S_unit = S_clean / SIGNAL_SCALE
    X = {s: add_rician_noise(S_unit, s, seed=noise_base + i) for i, s in enumerate(SIGMA_TEST_LEVELS)}
    return {"y": y, "pids": pids, "X": X}


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLLS (NB1 cell [10])
# ─────────────────────────────────────────────────────────────────────────────
def three_compartment_signal_flat(b_TE, D_ep, D_st, D_lu, T2_ep, T2_st, T2_lu, V_ep, V_st):
    b, TE = b_TE
    V_lu = 1.0 - V_ep - V_st
    S_ep = V_ep * np.exp(-b / 1000.0 * D_ep) * np.exp(-TE / T2_ep)
    S_st = V_st * np.exp(-b / 1000.0 * D_st) * np.exp(-TE / T2_st)
    S_lu = V_lu * np.exp(-b / 1000.0 * D_lu) * np.exp(-TE / T2_lu)
    return SIGNAL_SCALE * (S_ep + S_st + S_lu)


NLLS_P0 = [0.55, 1.3, 2.8, 50, 70, 750, 0.3, 0.4]
NLLS_LO = [0.3, 0.7, 2.7, 20, 40, 500, 0.0, 0.0]
NLLS_HI = [0.7, 1.7, 3.0, 70, 100, 1000, 1.0, 1.0]


def nlls_fit_one(signal):
    """Returns (9 params, valid, at_bound_count). Same logic as nlls_fit_3comp, one voxel."""
    try:
        coeffs, _ = curve_fit(three_compartment_signal_flat, b_TE_flat, signal, p0=NLLS_P0,
                              bounds=(NLLS_LO, NLLS_HI), method="trf", maxfev=5000)
        valid = True
    except Exception:
        coeffs = np.array(NLLS_P0, dtype=float)
        valid = False
    out = np.empty(9, dtype=np.float32)
    out[0:6] = coeffs[0:6]
    out[6:8] = coeffs[6:8]
    out[8] = 1.0 - coeffs[6] - coeffs[7]
    tol = 1e-6
    at_bound = int(np.sum((np.abs(coeffs - NLLS_LO) < tol * (np.abs(NLLS_LO) + 1))
                          | (np.abs(coeffs - NLLS_HI) < tol * (np.abs(NLLS_HI) + 1))))
    return out, valid, at_bound


# ─────────────────────────────────────────────────────────────────────────────
# 3. Models (NB1 cell [14], [22]; NB2 cell [10], [11])
# ─────────────────────────────────────────────────────────────────────────────
def physics_decode(D, T2, v):
    """PIA.decode from NB1 cell [14] (loop over b, TE), as a free function."""
    signal = torch.zeros((D.shape[0], N_SIGNALS), device=D.device)
    D, T2, v = D.T, T2.T, v.T
    ctr = 0
    for b in B_VALS:
        for TE in TE_VALS:
            S_ep = v[0] * torch.exp(-b / 1000.0 * D[0]) * torch.exp(-TE / T2[0])
            S_st = v[1] * torch.exp(-b / 1000.0 * D[1]) * torch.exp(-TE / T2[1])
            S_lu = v[2] * torch.exp(-b / 1000.0 * D[2]) * torch.exp(-TE / T2[2])
            signal[:, ctr] = S_ep + S_st + S_lu
            ctr += 1
    return SIGNAL_SCALE * signal


def _mlp(n_in, hidden_dims):
    modules, in_ch = [], n_in
    for h in hidden_dims:
        modules.append(nn.Sequential(nn.Linear(in_ch, h), nn.LeakyReLU()))
        in_ch = h
    return nn.Sequential(*modules)


def _head(width, depth, out_dim):
    layers = [nn.Sequential(nn.Linear(width, width), nn.LeakyReLU()) for _ in range(depth)]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class PIA(nn.Module):
    def __init__(self, number_of_signals=16, hidden_dims=None, predictor_depth=3):
        super().__init__()
        hidden_dims = hidden_dims or [64, 128, 256, 512, 512]
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("D_mean", torch.tensor(D_MEAN_3))
        self.register_buffer("T2_mean", torch.tensor(T2_MEAN_3))
        self.register_buffer("D_delta", torch.tensor(D_DELTA_3))
        self.register_buffer("T2_delta", torch.tensor(T2_DELTA_3))
        self.encoder = _mlp(number_of_signals, hidden_dims)
        self.D_predictor = _head(hidden_dims[-1], predictor_depth, 3)
        self.T2_predictor = _head(hidden_dims[-1], predictor_depth, 3)
        self.v_predictor = _head(hidden_dims[-1], predictor_depth, 3)

    def encode(self, x):
        h = self.encoder(x)
        D = self.D_mean + self.D_delta * torch.tanh(self.D_predictor(h))
        T2 = self.T2_mean + self.T2_delta * torch.tanh(self.T2_predictor(h))
        v = self.softmax(self.v_predictor(h))
        return D, T2, v

    def forward(self, x):
        D, T2, v = self.encode(x)
        return physics_decode(D, T2, v), D, T2, v


class PIVA_Diagonal(nn.Module):
    """NB2 cell [10]. Identical in code to NB1 cell [22] PIVA(constrained=True) = PIVA-Tight."""

    def __init__(self, n_signals=16, latent_dim=9, hidden_dims=None, predictor_depth=3):
        super().__init__()
        hidden_dims = hidden_dims or [64, 128, 256, 512, 512]
        self.encoder = _mlp(n_signals, hidden_dims)
        self.fc_mu = _head(hidden_dims[-1], predictor_depth, latent_dim)
        self.fc_logvar = _head(hidden_dims[-1], predictor_depth, latent_dim)
        self.register_buffer("D_mean", torch.tensor(D_MEAN_3))
        self.register_buffer("T2_mean", torch.tensor(T2_MEAN_3))
        self.register_buffer("D_delta", torch.tensor(D_DELTA_3))
        self.register_buffer("T2_delta", torch.tensor(T2_DELTA_3))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-8.0, 4.0)

    def reparameterize(self, mu, logvar, sample=True):
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def params_from_z(self, z):
        D_raw, T2_raw, v_raw = z[:, 0:3], z[:, 3:6], z[:, 6:9]
        D = torch.clamp(self.D_mean + self.D_delta * torch.tanh(D_raw), min=0.01, max=10.0)
        T2 = torch.clamp(self.T2_mean + self.T2_delta * torch.tanh(T2_raw), min=1.0, max=2000.0)
        return D, T2, F.softmax(v_raw, dim=1)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        D, T2, v = self.params_from_z(self.reparameterize(mu, logvar, sample))
        return physics_decode(D, T2, v), D, T2, v, mu, logvar


def kl_divergence_diag(mu, logvar):
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


PSI_ILR = torch.tensor([
    [1 / np.sqrt(2), 1 / np.sqrt(6)],
    [-1 / np.sqrt(2), 1 / np.sqrt(6)],
    [0.0, -2 / np.sqrt(6)],
], dtype=torch.float32)


class PIVA_CholeskyILR(nn.Module):
    """NB2 cell [11]."""

    def __init__(self, n_signals=16, hidden_dims=None, predictor_depth=3):
        super().__init__()
        self.latent_dim = 8
        hidden_dims = hidden_dims or [64, 128, 256, 512, 512]
        self.encoder = _mlp(n_signals, hidden_dims)
        n_tril = self.latent_dim * (self.latent_dim + 1) // 2
        self.fc_mu = _head(hidden_dims[-1], predictor_depth, self.latent_dim)
        self.fc_L_raw = _head(hidden_dims[-1], predictor_depth, n_tril)
        tril_idx = torch.tril_indices(self.latent_dim, self.latent_dim)
        self.register_buffer("tril_row", tril_idx[0])
        self.register_buffer("tril_col", tril_idx[1])
        self.register_buffer("diag_mask", tril_idx[0] == tril_idx[1])
        self.register_buffer("D_mean", torch.tensor(D_MEAN_3))
        self.register_buffer("T2_mean", torch.tensor(T2_MEAN_3))
        self.register_buffer("D_delta", torch.tensor(D_DELTA_3))
        self.register_buffer("T2_delta", torch.tensor(T2_DELTA_3))
        self.register_buffer("Psi", PSI_ILR.clone())

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        L_raw = self.fc_L_raw(h)
        L = torch.zeros(x.shape[0], self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype)
        vals = L_raw.clone()
        vals[:, self.diag_mask] = torch.exp(L_raw[:, self.diag_mask].clamp(-4.0, 2.0))
        L[:, self.tril_row, self.tril_col] = vals
        return mu, L

    def reparameterize(self, mu, L, sample=True):
        if not sample:
            return mu
        return mu + torch.bmm(L, torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)

    def params_from_z(self, z):
        D_raw, T2_raw, v_raw2 = z[:, 0:3], z[:, 3:6], z[:, 6:8]
        D = torch.clamp(self.D_mean + self.D_delta * torch.tanh(D_raw), min=0.01, max=10.0)
        T2 = torch.clamp(self.T2_mean + self.T2_delta * torch.tanh(T2_raw), min=1.0, max=2000.0)
        return D, T2, F.softmax(v_raw2 @ self.Psi.T, dim=1)

    def forward(self, x, sample=True):
        mu, L = self.encode(x)
        D, T2, v = self.params_from_z(self.reparameterize(mu, L, sample))
        return physics_decode(D, T2, v), D, T2, v, mu, L


def kl_divergence_full(mu, L):
    k = mu.shape[1]
    trace_term = (L ** 2).sum(dim=(1, 2))
    mu_term = (mu ** 2).sum(dim=1)
    logdet_term = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
    return (0.5 * (trace_term + mu_term - k - logdet_term)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training (NB1 cell [14] for PIA; NB2 cell [10] train_piva for both PIVAs)
# ─────────────────────────────────────────────────────────────────────────────
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_loader(X_train, S_clean_tr, y_train, seed):
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(S_clean_tr), torch.from_numpy(y_train))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0, generator=g)


def train_pia(model, loader, n_epochs=200, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    hist = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for xb, clean_b, _ in loader:
            opt.zero_grad()
            recon, _, _, _ = model(xb)
            loss = F.mse_loss(recon, clean_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        hist.append({"epoch": epoch, "signal_mse": float(np.mean(losses))})
        if epoch == 1 or epoch % 20 == 0 or epoch == n_epochs:
            log(f"PIA | epoch {epoch:03d} | signal_MSE={hist[-1]['signal_mse']:.2f}")
    return hist


def train_piva(model, loader, kl_fn, n_epochs=200, lr=5e-4, beta=0.1, log=print, label="PIVA"):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        mse_l, kl_l = [], []
        for xb, clean_b, _ in loader:
            opt.zero_grad()
            recon, D, T2, v, mu, second = model(xb, sample=True)
            sig_mse = F.mse_loss(recon, clean_b)
            kl = kl_fn(mu, second)
            loss = sig_mse + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            mse_l.append(sig_mse.item())
            kl_l.append(kl.item())
        hist.append({"epoch": epoch, "signal_mse": float(np.mean(mse_l)), "kl": float(np.mean(kl_l))})
        if epoch == 1 or epoch % 20 == 0 or epoch == n_epochs:
            log(f"{label} | epoch {epoch:03d} | signal_MSE={hist[-1]['signal_mse']:.2f} | KL={hist[-1]['kl']:.3f}")
    return hist


@torch.no_grad()
def predict_pia(model, X):
    model.eval()
    _, D, T2, v = model(torch.as_tensor(X, dtype=torch.float32))
    return np.concatenate([D.numpy(), T2.numpy(), v.numpy()], axis=1)


@torch.no_grad()
def mc_predict_piva(model, X, n_mc=25, seed=0):
    """NB2 cell [15] mc_predict_piva, with a fixed torch seed for the MC draws."""
    torch.manual_seed(seed)
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32)
    samples = []
    for _ in range(n_mc):
        _, D, T2, v, _, _ = model(x, sample=True)
        samples.append(np.concatenate([D.numpy(), T2.numpy(), v.numpy()], axis=1))
    s = np.stack(samples)
    return s.mean(0), s.std(0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation (new)
# ─────────────────────────────────────────────────────────────────────────────
def r2_1d(t, p):
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def per_param_metrics(y_true, y_pred, mask=None):
    """Per-parameter R2, MAE, bias, Pearson r, plus repo-style pooled-over-compartments R2/MAE."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    out = {}
    for j, name in enumerate(PARAM_NAMES):
        t, p = y_true[:, j], y_pred[:, j]
        out[f"{name}_R2"] = r2_1d(t, p)
        out[f"{name}_MAE"] = float(np.mean(np.abs(t - p)))
        out[f"{name}_bias"] = float(np.mean(p - t))
        out[f"{name}_r"] = float(pearsonr(t, p)[0]) if np.std(p) > 0 else float("nan")
    for blk, sl in [("D", slice(0, 3)), ("T2", slice(3, 6)), ("v", slice(6, 9))]:
        t, p = y_true[:, sl].ravel(), y_pred[:, sl].ravel()
        out[f"pooled_{blk}_R2"] = r2_1d(t, p)
        out[f"pooled_{blk}_MAE"] = float(np.mean(np.abs(t - p)))
    return out


def coverage(y_true, mean, sd, tau=1.0):
    lo, hi = mean - 1.96 * tau * sd, mean + 1.96 * tau * sd
    return ((y_true >= lo) & (y_true <= hi)).mean(axis=0)


def tau_fit(y_true, mean, sd, grid, target=0.95):
    """NB2 cell [18] calibrate_model inner loop, per parameter: tau minimising |cov - target|."""
    taus = np.empty(y_true.shape[1])
    for j in range(y_true.shape[1]):
        covs = np.array([coverage(y_true[:, j:j + 1], mean[:, j:j + 1], sd[:, j:j + 1], t)[0] for t in grid])
        taus[j] = grid[int(np.argmin(np.abs(covs - target)))]
    return taus


def calibrate(y_true, mean, sd, pids, split="voxel", grid=None, seed=0):
    """
    split='voxel' + grid=[0.5,25] reproduces NB2 cell [18] exactly (random half of voxels, seed 0).
    split='patient' holds out whole patients. Returns per-parameter tau, test coverage, width % range.
    """
    if grid is None:
        grid = np.linspace(0.5, 25.0, 500)
    N = y_true.shape[0]
    rng = np.random.default_rng(seed)
    if split == "voxel":
        perm = rng.permutation(N)
        cal_idx, test_idx = perm[: N // 2], perm[N // 2:]
    else:
        up = np.unique(pids)
        rng.shuffle(up)
        cal_p = set(up[: len(up) // 2].tolist())
        is_cal = np.array([p in cal_p for p in pids])
        cal_idx, test_idx = np.where(is_cal)[0], np.where(~is_cal)[0]
    taus = tau_fit(y_true[cal_idx], mean[cal_idx], sd[cal_idx], grid)
    test_cov = np.array([coverage(y_true[test_idx, j:j + 1], mean[test_idx, j:j + 1], sd[test_idx, j:j + 1], taus[j])[0]
                         for j in range(9)])
    width = 2 * 1.96 * taus * sd[test_idx].mean(axis=0)
    return {"tau": taus, "cal_cov": test_cov, "cal_width_pct": 100 * width / FULL_RANGE,
            "tau_at_grid_max": taus >= grid[-2] - 1e-9, "n_cal": len(cal_idx), "n_test": len(test_idx)}


def uq_metrics(y_true, mean, sd):
    out = {}
    cov = coverage(y_true, mean, sd)
    err = np.abs(y_true - mean)
    for j, name in enumerate(PARAM_NAMES):
        out[f"{name}_rawcov"] = float(cov[j])
        out[f"{name}_rawwidth_pct"] = float(100 * 2 * 1.96 * sd[:, j].mean() / FULL_RANGE[j])
        out[f"{name}_sd_mean"] = float(sd[:, j].mean())
        out[f"{name}_err_sd_pearson"] = float(pearsonr(sd[:, j], err[:, j])[0])
        out[f"{name}_err_sd_spearman"] = float(spearmanr(sd[:, j], err[:, j])[0])
        out[f"{name}_rmse_over_sd"] = float(np.sqrt(np.mean(err[:, j] ** 2)) / np.sqrt(np.mean(sd[:, j] ** 2)))
    # repo-style (NB1 cell [26]/[40] calibration_r): voxel-mean sd vs voxel-mean |err| per block
    for blk, sl in [("D", slice(0, 3)), ("T2", slice(3, 6)), ("v", slice(6, 9))]:
        out[f"repo_calr_{blk}"] = float(pearsonr(sd[:, sl].mean(1), err[:, sl].mean(1))[0])
        out[f"pooled_cov_{blk}"] = float(cov[sl].mean())
    return out
