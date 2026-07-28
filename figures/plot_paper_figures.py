"""
Generate publication figures for the paper (all B&W):
  1. Qualitative parameter maps: Ground Truth vs NLLS vs CNN-PIA vs CNN-PIA + Refiner
  2. Refiner ablation: Before/after refiner
  3. Example input data: 8 b-value DWI images + tissue mask
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
CKPT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'checkpoints')

sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'src'))
from utils import fit_biExponential_model

PATIENT = 61  # representative test patient


# ── U-Net Refiner (reconstructed from checkpoint) ──

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.leaky_relu(x + residual)


class UNetRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResBlock(3, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.bottleneck = ResBlock(256, 256)
        self.dec3 = ResBlock(512, 128)
        self.dec2 = ResBlock(256, 64)
        self.dec1 = ResBlock(128, 64)
        self.final_conv = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([F.interpolate(b, e3.shape[2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1], dim=1))
        return self.final_conv(d1)


# ── Load data ──

print(f"Loading patient {PATIENT:04d}...", flush=True)

gt_params = np.load(os.path.join(DATA_DIR, f'{PATIENT:04d}_IVIMParam.npy'))
gt_dwi = np.load(os.path.join(DATA_DIR, f'{PATIENT:04d}_gtDWIs.npy'))
noisy_k = np.load(os.path.join(DATA_DIR, f'{PATIENT:04d}_NoisyDWIk.npy'))
tissue = np.load(os.path.join(DATA_DIR, f'{PATIENT:04d}_TissueType.npy'))
cnn_est = np.load(os.path.join(DATA_DIR, f'{PATIENT:04d}_NoisyEstimateCNN.npy'))
mlp_est = np.load(os.path.join(DATA_DIR, f'{PATIENT:04d}_NoisyEstimate.npy'))

noisy_img = np.abs(np.fft.ifft2(noisy_k, axes=(0, 1), norm='ortho'))
body_mask = tissue != 1

gt_f, gt_Dt, gt_Ds = gt_params[:, :, 0], gt_params[:, :, 1], gt_params[:, :, 2]
cnn_f, cnn_Dt, cnn_Ds = cnn_est[:, :, 0], cnn_est[:, :, 1], cnn_est[:, :, 2]

b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])


# ── Run NLLS on this patient ──

print("Running NLLS fitting...", flush=True)
nlls_result = fit_biExponential_model(noisy_img, b_values)
nlls_f, nlls_Dt, nlls_Ds = nlls_result[:, :, 0], nlls_result[:, :, 1], nlls_result[:, :, 2]


# ── Run refiner on CNN-PIA estimates with proper norm/denorm ──

print("Running CNN-PIA + Refiner...", flush=True)
refiner = UNetRefiner()
ckpt = torch.load(os.path.join(CKPT_DIR, 'refiner_cnn_e2e_best.pt'), map_location='cpu', weights_only=False)
refiner.load_state_dict(ckpt['model_state_dict'])
refiner.eval()

# Min-max normalization as expected by the UNet refiner
f_norm = (cnn_f - 0.05) / 0.3
Dt_norm = (cnn_Dt - 0.0007) / 0.0008
Ds_norm = (cnn_Ds - 0.005) / 0.05

cnn_norm = np.stack([f_norm, Dt_norm, Ds_norm], axis=0)
cnn_input = torch.tensor(cnn_norm, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    refined_norm = refiner(cnn_input).squeeze(0).numpy()

# Denormalize output maps
ref_f = np.clip(refined_norm[0] * 0.3 + 0.05, 0.0, 0.4)
ref_Dt = np.clip(refined_norm[1] * 0.0008 + 0.0007, 0.0001, 0.002)
ref_Ds = np.clip(refined_norm[2] * 0.05 + 0.005, 0.001, 0.06)


# ── Masking helper ──

def mask_param(param, mask):
    out = np.copy(param)
    out[~mask] = np.nan
    return out


# ── Figure style ──

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
})


# ════════════════════════════════════════════════════════════════
# FIGURE 1: Qualitative Comparison (4 methods × 3 parameters)
# ════════════════════════════════════════════════════════════════

print("Generating qualitative comparison figure...", flush=True)

param_names = ['$f$', '$D_t$', '$D^*$']
param_ranges = [(0, 0.3), (0, 0.0015), (0, 0.05)]

methods_data = {
    'Ground Truth': (gt_f, gt_Dt, gt_Ds),
    'NLLS': (nlls_f, nlls_Dt, nlls_Ds),
    'CNN-PIA': (cnn_f, cnn_Dt, cnn_Ds),
    'CNN-PIA\n+ Refiner': (ref_f, ref_Dt, ref_Ds),
}

fig, axes = plt.subplots(4, 3, figsize=(7, 8.5))

for row_idx, (method_name, (pf, pDt, pDs)) in enumerate(methods_data.items()):
    params = [pf, pDt, pDs]
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        im = ax.imshow(mask_param(params[col_idx], body_mask),
                       cmap='gray', vmin=param_ranges[col_idx][0],
                       vmax=param_ranges[col_idx][1], interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(param_names[col_idx], fontsize=12, fontweight='bold', pad=6)
        if col_idx == 0:
            ax.set_ylabel(method_name, fontsize=10, fontweight='bold', rotation=90,
                          labelpad=10, va='center')

plt.suptitle('Parameter Map Comparison', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(SCRIPT_DIR, 'qualitative_comparison.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════
# FIGURE 2: Refiner Ablation (before vs after, with error maps)
# ════════════════════════════════════════════════════════════════

print("Generating refiner ablation figure...", flush=True)

fig, axes = plt.subplots(3, 4, figsize=(9, 7))

col_titles = ['Ground Truth', 'CNN-PIA', 'CNN-PIA + Refiner', '|Error| Reduction']
params_gt = [gt_f, gt_Dt, gt_Ds]
params_before = [cnn_f, cnn_Dt, cnn_Ds]
params_after = [ref_f, ref_Dt, ref_Ds]

for row_idx in range(3):
    vmin, vmax = param_ranges[row_idx]

    # Ground truth
    axes[row_idx, 0].imshow(mask_param(params_gt[row_idx], body_mask),
                             cmap='gray', vmin=vmin, vmax=vmax)
    # Before refiner
    axes[row_idx, 1].imshow(mask_param(params_before[row_idx], body_mask),
                             cmap='gray', vmin=vmin, vmax=vmax)
    # After refiner
    axes[row_idx, 2].imshow(mask_param(params_after[row_idx], body_mask),
                             cmap='gray', vmin=vmin, vmax=vmax)

    # Error reduction: |before - gt| - |after - gt|
    err_before = np.abs(params_before[row_idx] - params_gt[row_idx])
    err_after = np.abs(params_after[row_idx] - params_gt[row_idx])
    err_diff = mask_param(err_before - err_after, body_mask)
    e_max = np.nanmax(np.abs(err_diff)) * 0.7 if np.nanmax(np.abs(err_diff)) > 0 else 1.0
    axes[row_idx, 3].imshow(err_diff, cmap='gray', vmin=-e_max, vmax=e_max)

    axes[row_idx, 0].set_ylabel(param_names[row_idx], fontsize=11, fontweight='bold')

    for col_idx in range(4):
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=10, fontweight='bold', pad=6)

plt.suptitle('Refiner Ablation', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(SCRIPT_DIR, 'refiner_ablation.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════
# FIGURE 3: Example Input Data (8 b-value images + tissue mask)
# ════════════════════════════════════════════════════════════════

print("Generating example input figure...", flush=True)

fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))

b_labels = [f'b = {b}' for b in b_values]

for i in range(8):
    row, col = divmod(i, 5)
    ax = axes[row][col]
    ax.imshow(noisy_img[:, :, i], cmap='gray', interpolation='nearest')
    ax.set_title(b_labels[i], fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# Tissue mask
ax = axes[1][3]
ax.imshow(tissue, cmap='gray', interpolation='nearest')
ax.set_title('Tissue Mask', fontsize=9, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])

# Tumor overlay
ax = axes[1][4]
tumor_mask = (tissue == 8).astype(float)
ax.imshow(noisy_img[:, :, 0], cmap='gray', interpolation='nearest')
ax.contour(tumor_mask, levels=[0.5], colors='white', linewidths=1.2)
ax.set_title('Tumor ROI', fontsize=9, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])

plt.suptitle(f'Example Input: Patient {PATIENT:04d} (DW-MRI, 8 b-values)',
             fontsize=12, fontweight='bold', y=1.0)
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, 'example_input_data.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out}")

print("\nAll figures generated successfully.")
