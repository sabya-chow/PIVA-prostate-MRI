# PIVA — Physics-Informed Variational Autoencoder for Prostate MRI Parameter Estimation

**Capstone Project | MS Applied Data Science | University of Chicago**  
**Authors:** Sabyasachi Chowdhury · sabyachow@gmail.com · Aaron Garay · aaron.m.garay@gmail.com

---

## Abstract

This project develops and evaluates **PIVA** (Physics-Informed Variational Autoencoder) for quantitative parameter estimation from 3-compartment prostate MRI signals. The core clinical problem: Non-Linear Least Squares (NLLS) fitting of the multi-compartment model collapses at clinical SNR (~20:1), producing diffusivity and T2 maps that are statistically indistinguishable from noise. ~168,000 men per year in the US receive false-positive PSA tests and undergo unnecessary biopsies partly because these quantitative maps are unreliable.

PIVA addresses this by embedding the known MRI physics equation inside a variational autoencoder, replacing voxel-by-voxel curve fitting with a learned probabilistic inverse map that returns a **posterior distribution** over the 9 tissue parameters rather than a point estimate.

**Latest update:** the repository now includes a focused v2.0 notebook that tests whether replacing the diagonal PIVA posterior with a Cholesky full-covariance posterior, plus an identifiable ILR parameterization for volume fractions, improves uncertainty calibration at the clinical reference noise level σ = 0.05. The full-covariance posterior direction is explicitly credited to Finkelstein et al.'s PS-VAE paper on multiparameter uncertainty mapping in quantitative molecular MRI.

---

## The 3-Compartment Signal Model

The forward physics model is:

$$S(b, TE) = S_0 \sum_{c \in \{\text{epi, str, lum}\}} v_c \cdot \exp\!\left(-b \cdot D_c - \frac{TE}{T2_c}\right)$$

where for each compartment $c$:
- $D_c$ — apparent diffusion coefficient (μm²/ms); range: epithelium 0.3–0.7, stroma 0.7–1.7, lumen 2.7–3.0
- $T2_c$ — transverse relaxation time (ms); range: epi 20–70, str 40–100, lumen 500–1000
- $v_c$ — volume fraction, $\sum_c v_c = 1$

Acquisitions: **4 b-values** × **4 echo times** = **16 signal measurements per voxel**.

---

## Repository Structure

```
PIVA-prostate-MRI/
│
├── notebooks/
│   ├── PIVA_comparative.ipynb                         ← Main experiment notebook (all models, all results)
│   └── PIVA_comparative_full_covariance_v2_0.ipynb    ← Focused v2.0 Cholesky/ILR posterior upgrade
│
├── docs/
│   ├── ARCHITECTURE.md            ← Full system architecture, cell-by-cell walkthrough
│   ├── PIVA_technical_article.md  ← Technical write-up (methodology + results)
│   └── math_explainer.md          ← Mathematical derivation of the ELBO and physics decoder
│
├── assets/
│   └── mri_1.png                  ← MRI signal diagram
│
├── results/
│   └── RESULTS.md                 ← Structured metrics tables across all models and noise levels
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methods

The broad comparative notebook evaluates three model families on an identical held-out synthetic cohort (240 voxels = 24 patients × 10 voxels):

### 1. NLLS Baseline
`scipy.curve_fit` applied independently per voxel. No shared information across voxels. Serves as the clinical baseline.

### 2. PIA — Physics-Informed Autoencoder (Deterministic)
```
x (16-signal) → Encoder (MLP) → [D, T2, v] → Physics Decoder (fixed) → x̂ (16-signal)
```
The decoder is the exact physics equation — not learned. The encoder is trained to minimize signal reconstruction MSE. At inference, it produces a single point estimate per voxel.

### 3. PIVA — Physics-Informed Variational Autoencoder (Probabilistic)
```
x → Encoder → [μ_z, σ_z] → z ~ N(μ_z, σ_z²) → [D, T2, v] → Physics Decoder → x̂
```
Training objective is the ELBO:
$$\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{signal reconstruction}} - \beta \cdot \underbrace{D_{\mathrm{KL}}[q(z|x) \| p(z)]}_{\text{prior regularization}}$$

Three prior configurations are tested:
- **PIVA-Tight** — Gaussian prior with half-ranges matching physiological constraints (D_DELTA_3, T2_DELTA_3)
- **PIVA-Wide** — Gaussian prior with 2× wider half-ranges
- **PIVA-Free** — Standard N(0,I) prior, unconstrained

Posterior inference: **25 Monte Carlo forward passes** → posterior mean and σ per parameter.

### 4. PIVA v2.0 — Full-Covariance Cholesky Posterior + ILR Volume Latent
The v2.0 notebook isolates one architectural change: diagonal Gaussian posterior versus full-covariance posterior. This direction owes direct credit to Finkelstein et al. (2026), whose PS-VAE for quantitative molecular MRI uses a physics simulator and a full covariance posterior to capture inter-parameter correlations in latent biophysical space. This repository adapts that idea to the 3-compartment prostate MRI setting.

The experiment keeps the same physics decoder, physiological window, β=0.1 ELBO training recipe, and held-out cohort, but replaces:

```math
q(z|x) = N(\mu, diag(\sigma^2))
```

with:

```math
q(z|x) = N(\mu, L L^T)
```

where $L$ is a learned lower-triangular Cholesky factor. The volume-fraction latent is also changed from a redundant 3-dimensional softmax input to a 2-dimensional ILR representation, removing the null direction where $softmax(v_{raw} + c*\hat1) = softmax(v_{raw})$.

---

## Key Results (σ = 0.05, clinical SNR ≈ 20:1)

| Model | D R² | T2 R² | v R² | D MAE | T2 MAE (ms) | v MAE |
|---|---|---|---|---|---|---|
| NLLS | 0.923 | 0.810 | 0.238 | 0.225 | 91.8 | 0.123 |
| **PIA** | **0.983** | **0.968** | **0.572** | **0.093** | **32.6** | **0.093** |
| PIVA-Tight | 0.952 | 0.788 | 0.254 | 0.173 | 94.9 | 0.128 |
| PIVA-Wide | 0.819 | 0.797 | 0.111 | 0.342 | 85.9 | 0.139 |
| PIVA-Free | −1.483 | −0.658 | −2.231 | 1.345 | 303.9 | 0.285 |

Full results across all noise levels and the β-sweep: see [`results/RESULTS.md`](results/RESULTS.md).

### Latest v2.0 Result: Diagonal vs Full-Covariance Posterior (σ = 0.05)

| Model | D R² | T2 R² | v R² | D MAE | T2 MAE (ms) | v MAE |
|---|---:|---:|---:|---:|---:|---:|
| PIVA-Diagonal | 0.987 | 0.966 | 0.577 | 0.080 | 33.525 | 0.097 |
| **PIVA-CholeskyILR** | **0.988** | 0.957 | 0.568 | **0.077** | 37.295 | 0.097 |

The full-covariance Cholesky/ILR model improves raw coverage for all nine compartment parameters. After per-compartment post-hoc calibration, coverage improves for seven of nine parameters, but some intervals remain clinically wide. The result is positive but not magic: full covariance helps posterior geometry, while calibration still needs a separate correction step.

### Uncertainty Calibration (PIVA-Tight, σ = 0.05)

| Metric | D | T2 | v |
|---|---|---|---|
| 95% CI empirical coverage | 0.156 | 0.085 | 0.053 |
| Target | 0.950 | 0.950 | 0.950 |
| Calibration Pearson r (σ vs \|error\|) | −0.121 | −0.040 | 0.267 |
| Mean CV (voxel uncertainty score) | 0.0071 | — | — |

The posterior is systematically overconfident. Coverage does not approach the nominal 95% target even at β=3.0 (maximum tested). The calibration Pearson r is negative for D and T2, indicating posterior variance is anti-correlated with actual prediction error.

### β-Sweep Summary (PIVA-Tight, D parameter)

| β | D R² | D 95% CI cov | D cal r |
|---|---|---|---|
| 0.01 | **0.9879** | 0.086 | −0.276 |
| 0.10 | 0.9874 | 0.151 | −0.121 |
| 0.30 | 0.9871 | 0.210 | −0.056 |
| 1.00 | 0.9843 | 0.307 | −0.174 |
| 3.00 | 0.9386 | 0.349 | −0.074 |

There is a monotonic accuracy–coverage trade-off with β. No single β achieves both high R² and well-calibrated uncertainty.

---

## How to Run

### Environment Setup

```bash
git clone https://github.com/sabya-chow/PIVA-prostate-MRI.git
cd PIVA-prostate-MRI
pip install -r requirements.txt
```

### Run the Main Notebook

```bash
jupyter notebook notebooks/PIVA_comparative.ipynb
```

All data is generated synthetically within the notebook — no external dataset is required. The notebook is fully self-contained and runs end-to-end from data generation through the β-sweep experiment.

**Estimated runtime:** ~10–15 min on CPU (200 epochs × 3 PIVA variants + β-sweep); ~3–5 min on GPU.

### Run the v2.0 Full-Covariance Notebook

```bash
jupyter notebook notebooks/PIVA_comparative_full_covariance_v2_0.ipynb
```

This notebook is narrower: it compares PIVA-Diagonal against PIVA-CholeskyILR at σ = 0.05 and includes raw plus post-hoc calibrated coverage diagnostics.

### Cell Execution Order

The notebook is linear. Run all cells top-to-bottom. Key checkpoints:

| Cell range | Content |
|---|---|
| [0]–[2] | Imports and constants |
| [3]–[7] | Clinical motivation + data generation |
| [8]–[11] | NLLS baseline evaluation |
| [12]–[19] | PIA architecture, training, evaluation |
| [20]–[26] | PIVA architecture (3 variants), training, evaluation, uncertainty diagnostics |
| [27]–[30] | Final comparison table + forest plot |
| [40] | β-sweep experiment (self-contained, reruns PIVA-Tight only) |

---

## Open Questions / Next Steps

1. **Calibration fix** — apply conformal prediction post-hoc on a calibration split to achieve guaranteed empirical coverage without retraining.
2. **Learned prior** — replace N(0,I) with a VampPrior or moment-matched Gaussian estimated from training data to fix prior mismatch (likely the dominant cause of miscalibration).
3. **Posterior variance compression** — the fixed physics decoder creates a bottleneck where posterior σ in latent space does not faithfully propagate to physical parameter uncertainty. Investigate a learnable decoder head.
4. **Real MRI data** — validate on a held-out real patient cohort (e.g., ProstateX) once synthetic results are stable.

---

## References

- Panagiotaki et al. (2014) — 3-compartment prostate MRI model
- Kingma & Welling (2014) — Variational Autoencoder (VAE)
- Higgins et al. (2017) — β-VAE: understanding disentangled representations
- Raissi et al. (2019) — Physics-Informed Neural Networks
- Finkelstein, A., Moneta, R., Zohar, O., Rivlin, M., Zaiss, M., Friedmann Morvinski, D., & Perlman, O. (2026) — Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE), arXiv:2602.03317. https://arxiv.org/abs/2602.03317
