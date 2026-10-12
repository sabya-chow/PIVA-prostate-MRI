# PIVA audit rerun: per-compartment metrics, seeds, uncertainty

Rerun of the PIVA repo code (PIVA-prostate-MRI @ 3933f7e, read-only) for the ISMRM 2027 abstract. All numbers below come from scripts in this folder. Full tables for every cohort and noise level are in `results/tables.md`; raw numbers are in `results/*.csv`.

## Headline

1. **D and T2 are not learned by any model.** Per-compartment R² is negative for all six D and T2 parameters, for every model (NLLS, PIA, PIVA-Diagonal, PIVA-CholeskyILR), at every noise level, on both test cohorts. A constant predictor at the prior centre has lower MAE than every trained model on all six. The repo's pooled R² of 0.98 to 0.99 for D is produced entirely by between-compartment differences.
2. **Volume fractions are learned.** v_lum R² about 0.89, v_ep 0.49 to 0.61, v_st 0.34 to 0.48 at σ = 0.05 (large cohort). On the large cohort PIA is as good as or better than both PIVA variants on all three.
3. **The information limit agrees.** A Bayesian Cramér-Rao calculation (experiment E) shows the 16-signal protocol at σ = 0.05 shrinks the population prior SD by only 1 to 13% for D and T2, but by 34 to 70% for v. There is little to learn about D and T2 from these data.
4. **Raw PIVA posteriors are severely overconfident** (coverage 0.05 to 0.33 Diagonal, 0.13 to 0.62 Cholesky). Full covariance raises raw coverage on 9/9 parameters; the ablation shows this comes from the Cholesky covariance, not the ILR map.
5. **After post-hoc calibration both variants reach about 0.95**, but for D and T2 the calibrated intervals are wider than a fixed, data-independent interval with the same coverage. Only the v intervals are informative.
6. **The noise-matched likelihood (experiment A) removes most of the overconfidence without any post-hoc step**, but for D and T2 it does so by collapsing to the latent prior (raw widths 95 to 120% of range, wider than the population prior). For v at σ = 0.05 it gives raw coverage 0.86 to 0.91 at widths matching the information limit (3 seeds, round 2).
7. **Round 2: a per-voxel noise SD makes v interval widths track the information limit across SNR**, where a fixed SD-50 gives flat widths. Coverage is still not nominal: per-voxel drops to 0.64 at σ = 0.01, SD-50 to 0.81 at σ = 0.08, and both under-cover low-epithelium voxels by 10 to 15 points.

## Setup

**Code.** `piva_core.py` holds the data generator, NLLS, PIA, PIVA-Diagonal (NB2 cell [10]) and PIVA-CholeskyILR (NB2 cell [11]) copied from the notebooks. Changes from the notebooks, all non-modelling:

- Seeds: `random`, `numpy` and `torch` seeded per run (seeds 0, 1, 2); DataLoader shuffle uses a seeded `torch.Generator`; MC sampling at evaluation uses a fixed torch seed (1234). The training data are the repo's fixed-seed data (generator seed 0, bootstrap 123, noise 1000/1001), identical across training seeds, as in the repo.
- Batch size fixed at 2048 (what the notebooks use on CUDA, which produced the repo outputs; on CPU the notebooks silently switch to 512).
- The decoder is a module-level function instead of `pia_encoder.decode`; numerically identical (no learnable parameters).
- **PIVA-Tight (NB1 cell [22]) and PIVA-Diagonal (NB2 cell [10]) are the same code**: same architecture, clamps, loss, β = 0.1, no warm-up, lr 5e-4, 200 epochs, 25 MC samples. They are reported once, as PIVA-Diagonal (verified by diffing the two notebook cells).
- Training on CPU with flush-to-zero for subnormal floats in some runs (see Runtime). An A/B rerun of diag seed 0 gave bit-identical losses through epoch 60 with FTZ on and off.

**Test cohorts.** Same generator (`generate_patient_clean_cohort`), Rician noise at fixed σ ∈ {0.01, 0.03, 0.05, 0.08}:

| Cohort | Patients x voxels | Cohort seed | Noise seeds | Use |
|---|---|---|---|---|
| orig240 | 24 x 10 = 240 | 999 | 7000 + index of σ (σ = 0.05 → 7002, as NB1; NB2 used 7000) | original held-out test set |
| large2000 | 200 x 10 = 2000 | 2027 | 17000 + index | larger test set |
| calib240 / calib2000 | 24 x 10 / 200 x 10 | 31337 / 4242 | 27000 / 37000 + index | experiment C calibration only, never tested on |

**Baselines.** "Const prior centre" predicts (0.5, 1.2, 2.85; 45, 70, 750; 1/3, 1/3, 1/3) for every voxel. "Const training mean" predicts the per-parameter mean of `y_train`. Neither looks at the signal; per-compartment R² for both is about 0 by construction.

**Metrics.** Per-parameter R² = 1 − SS_res/SS_tot within one parameter; MAE; bias. "Pooled" = the repo's metric (all three compartments flattened together). Coverage = fraction of voxels with truth inside mean ± 1.96 × posterior SD × τ. Width = 3.92 × mean posterior SD × τ, as % of the physiological range (2 × half-width for D, T2; 1 for v). Error-vs-σ correlation = per-parameter Spearman ρ between posterior SD and |error| across voxels (the repo-style voxel-averaged Pearson is also reported).

**Calibration variants.**
- *v2.0 replica*: exact NB2 cell [18] procedure: random voxel-level half of the test cohort, τ grid [0.5, 25] (500 points), evaluated on the other half.
- *Separate cohort* (experiment C, used for the figures): τ per parameter fit on calib240/calib2000 (grid [0.1, 200], 4000 points), then applied to the whole test cohort; 95% CI from a patient-level bootstrap (1000 resamples of patients).
- A patient-level within-cohort split is also in `results/calibration_long.csv` (`pat_*` columns).

## Task 1 to 3: per-compartment accuracy at σ = 0.05

Mean ± SD over 3 training seeds. NLLS and baselines are deterministic (1 run).

**large2000 (2000 voxels), per-parameter R²**

| Model | n | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|---|
| Const prior centre | 1 | -0.000 | -0.000 | -0.002 | -0.001 | -0.000 | -0.004 | -0.009 | -0.063 | -0.027 |
| Const training mean | 1 | -0.005 | -0.000 | -0.011 | -0.027 | -0.007 | -0.006 | -0.004 | -0.015 | -0.004 |
| NLLS | 1 | -15.35 | -8.41 | -16.32 | -9.52 | -5.92 | -7.40 | 0.279 | -0.252 | 0.750 |
| PIA | 3 | -1.41 ± 0.22 | -1.30 ± 0.13 | -2.08 ± 0.34 | -0.53 ± 0.10 | -0.31 ± 0.02 | -0.53 ± 0.06 | 0.612 ± 0.017 | 0.484 ± 0.015 | 0.899 ± 0.005 |
| PIVA-Diagonal | 3 | -1.38 ± 0.07 | -1.03 ± 0.02 | -1.41 ± 0.31 | -0.39 ± 0.08 | -0.62 ± 0.14 | -0.34 ± 0.16 | 0.488 ± 0.030 | 0.341 ± 0.032 | 0.891 ± 0.004 |
| PIVA-CholeskyILR | 3 | -1.28 ± 0.06 | -0.86 ± 0.12 | -1.25 ± 0.11 | -0.35 ± 0.13 | -0.67 ± 0.01 | -0.91 ± 0.11 | 0.582 ± 0.017 | 0.397 ± 0.016 | 0.886 ± 0.006 |

**large2000, per-parameter MAE** (D in µm²/ms, T2 in ms, v as fraction)

| Model | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|
| Const prior centre | 0.037 | 0.104 | 0.029 | 5.54 | 8.05 | 70.8 | 0.156 | 0.169 | 0.167 |
| Const training mean | 0.037 | 0.104 | 0.029 | 5.61 | 8.08 | 70.8 | 0.155 | 0.167 | 0.162 |
| NLLS | 0.174 | 0.351 | 0.144 | 20.88 | 23.59 | 239.7 | 0.125 | 0.178 | 0.077 |
| PIA | 0.057 ± 0.003 | 0.159 ± 0.006 | 0.052 ± 0.003 | 6.90 ± 0.24 | 9.19 ± 0.10 | 88.5 ± 1.6 | 0.093 ± 0.002 | 0.116 ± 0.001 | 0.050 ± 0.001 |
| PIVA-Diagonal | 0.057 ± 0.001 | 0.146 ± 0.001 | 0.046 ± 0.003 | 6.51 ± 0.19 | 10.14 ± 0.48 | 82.8 ± 4.9 | 0.106 ± 0.003 | 0.130 ± 0.003 | 0.052 ± 0.001 |
| PIVA-CholeskyILR | 0.056 ± 0.001 | 0.140 ± 0.004 | 0.044 ± 0.001 | 6.38 ± 0.30 | 10.25 ± 0.06 | 99.5 ± 2.4 | 0.097 ± 0.002 | 0.126 ± 0.002 | 0.053 ± 0.002 |

**orig240 (240 voxels), per-parameter R²**

| Model | n | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|---|
| Const prior centre | 1 | -0.033 | -0.001 | -0.035 | -0.017 | -0.007 | -0.002 | -0.070 | -0.008 | -0.019 |
| NLLS | 1 | -15.33 | -7.55 | -16.73 | -9.29 | -6.61 | -6.86 | 0.161 | -0.211 | 0.720 |
| PIA | 3 | -1.56 ± 0.24 | -0.96 ± 0.20 | -1.69 ± 0.31 | -0.35 ± 0.05 | -0.50 ± 0.08 | -0.48 ± 0.07 | 0.456 ± 0.028 | 0.368 ± 0.039 | 0.888 ± 0.003 |
| PIVA-Diagonal | 3 | -1.04 ± 0.22 | -0.39 ± 0.11 | -1.10 ± 0.28 | -0.22 ± 0.03 | -0.96 ± 0.19 | -0.31 ± 0.14 | 0.472 ± 0.034 | 0.352 ± 0.030 | 0.873 ± 0.006 |
| PIVA-CholeskyILR | 3 | -1.10 ± 0.04 | -0.29 ± 0.04 | -1.16 ± 0.20 | -0.23 ± 0.06 | -1.11 ± 0.10 | -0.84 ± 0.15 | 0.517 ± 0.017 | 0.374 ± 0.013 | 0.867 ± 0.004 |

Bias tables (all models shrink toward, or overshoot, the prior centre; T2_lum bias +42 ms Diagonal, +51 ms Cholesky, −31 ms PIA on large2000) are in `results/tables.md`.

**Repo-style pooled metrics, σ = 0.05** (for comparison with the repo)

| Model | Cohort | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE |
|---|---|---|---|---|---|---|---|
| Const prior centre | orig240 | 0.992 | 0.976 | 0.000 | 0.059 | 28.73 | 0.160 |
| NLLS | orig240 | 0.924 | 0.811 | 0.238 | 0.224 | 91.66 | 0.123 |
| PIA | orig240 | 0.984 ± 0.002 | 0.964 ± 0.002 | 0.586 ± 0.021 | 0.088 ± 0.005 | 34.23 ± 0.58 | 0.092 ± 0.002 |
| PIVA-Diagonal | orig240 | 0.988 ± 0.001 | 0.968 ± 0.003 | 0.579 ± 0.017 | 0.076 ± 0.002 | 32.77 ± 1.94 | 0.095 ± 0.001 |
| PIVA-CholeskyILR | orig240 | 0.989 ± 0.000 | 0.956 ± 0.003 | 0.598 ± 0.010 | 0.075 ± 0.001 | 38.51 ± 1.52 | 0.094 ± 0.001 |
| Const prior centre | large2000 | 0.993 | 0.975 | 0.000 | 0.056 | 28.13 | 0.164 |
| NLLS | large2000 | 0.926 | 0.791 | 0.277 | 0.223 | 94.72 | 0.127 |
| PIA | large2000 | 0.984 ± 0.001 | 0.962 ± 0.001 | 0.675 ± 0.009 | 0.090 ± 0.004 | 34.85 ± 0.53 | 0.086 ± 0.001 |
| PIVA-Diagonal | large2000 | 0.985 ± 0.000 | 0.967 ± 0.004 | 0.587 ± 0.018 | 0.083 ± 0.001 | 33.16 ± 1.80 | 0.096 ± 0.002 |
| PIVA-CholeskyILR | large2000 | 0.987 ± 0.001 | 0.952 ± 0.003 | 0.632 ± 0.012 | 0.080 ± 0.002 | 38.70 ± 0.81 | 0.092 ± 0.002 |

The constant predictor beats every model on pooled D and T2 R² and MAE, on both cohorts.

Figure: `figures/fig1_per_parameter_accuracy.png` (per-parameter R² and MAE relative to the constant predictor, large2000, σ = 0.05).

## Task 6: other noise levels

Full per-parameter R², MAE and pooled tables for σ = 0.01, 0.03, 0.08 on both cohorts are in `results/tables.md`. Summary on large2000:

- D and T2 per-parameter R² stay negative for every trained model at every σ. The only exception is T2_ep at σ = 0.01 (PIA 0.14, Diagonal 0.15, Cholesky 0.21).
- v degrades smoothly with σ (figure `figures/supplementary/fig3_vfraction_r2_vs_sigma.png`). At σ = 0.01: v_ep 0.70 / 0.61 / 0.72, v_st 0.70 / 0.61 / 0.69, v_lum 0.978 / 0.978 / 0.972 (PIA / Diagonal / Cholesky). At σ = 0.08: v_ep 0.53 / 0.40 / 0.50, v_st 0.33 / 0.18 / 0.25, v_lum 0.82 / 0.81 / 0.81.
- PIA ≥ Cholesky ≥ Diagonal on v_st at every σ; Cholesky ≈ PIA on v_ep; all three tie on v_lum.

## Task 4: uncertainty (σ = 0.05, large2000, 3 seeds)

| Param | Model | raw cov | raw width % | Spearman ρ(SD, \|err\|) | v2.0 replica τ | v2.0 cal cov | v2.0 width % | sep-cohort τ | sep-cohort cal cov [patient 95% CI] | sep-cohort width % |
|---|---|---|---|---|---|---|---|---|---|---|
| D_ep | Diagonal | 0.069 ± 0.004 | 3.2 | -0.28 ± 0.03 | 24.9 (2/3 at grid max) | 0.922 | 78.6 | 26.5 | 0.940 [0.93, 0.95] | 84.0 |
| D_ep | Cholesky | 0.286 ± 0.008 | 13.0 | -0.21 ± 0.03 | 6.3 | 0.943 | 81.2 | 6.2 | 0.945 [0.93, 0.96] | 80.3 |
| D_st | Diagonal | 0.075 ± 0.002 | 3.2 | -0.28 ± 0.00 | 25.0 (3/3 at grid max) | 0.920 | 78.7 | 29.2 | 0.947 [0.93, 0.96] | 91.9 |
| D_st | Cholesky | 0.189 ± 0.022 | 7.6 | -0.18 ± 0.01 | 11.0 | 0.944 | 83.7 | 11.5 | 0.954 [0.94, 0.97] | 87.5 |
| D_lum | Diagonal | 0.334 ± 0.075 | 18.4 | 0.06 ± 0.01 | 5.5 | 0.953 | 100.9 | 5.3 | 0.946 [0.93, 0.96] | 96.5 |
| D_lum | Cholesky | 0.620 ± 0.015 | 33.9 | -0.17 ± 0.02 | 2.3 | 0.960 | 78.3 | 2.4 | 0.961 [0.95, 0.97] | 80.3 |
| T2_ep | Diagonal | 0.085 ± 0.004 | 3.3 | -0.07 ± 0.01 | 21.2 | 0.945 | 70.5 | 20.6 | 0.943 [0.93, 0.95] | 68.6 |
| T2_ep | Cholesky | 0.177 ± 0.012 | 6.7 | -0.04 ± 0.04 | 10.7 | 0.949 | 70.7 | 10.5 | 0.947 [0.94, 0.96] | 70.1 |
| T2_st | Diagonal | 0.060 ± 0.007 | 3.1 | -0.19 ± 0.04 | 24.9 (2/3 at grid max) | 0.894 | 78.4 | 31.1 | 0.939 [0.92, 0.96] | 97.7 |
| T2_st | Cholesky | 0.146 ± 0.005 | 8.0 | 0.09 ± 0.04 | 12.2 | 0.953 | 96.9 | 10.8 | 0.928 [0.91, 0.94] | 86.1 |
| T2_lum | Diagonal | 0.180 ± 0.037 | 9.5 | -0.07 ± 0.05 | 16.6 | 0.961 | 152.2 | 15.4 | 0.949 [0.93, 0.97] | 141.8 |
| T2_lum | Cholesky | 0.487 ± 0.024 | 32.2 | -0.29 ± 0.02 | 4.5 | 0.964 | 145.5 | 3.8 | 0.935 [0.91, 0.96] | 121.6 |
| v_ep | Diagonal | 0.050 ± 0.002 | 1.7 | 0.29 ± 0.02 | 25.0 (2/3 at grid max) | 0.901 | 42.8 | 28.8 | 0.947 [0.93, 0.96] | 49.1 |
| v_ep | Cholesky | 0.134 ± 0.013 | 4.4 | 0.24 ± 0.02 | 11.6 | 0.963 | 51.7 | 11.1 | 0.945 [0.93, 0.96] | 48.9 |
| v_st | Diagonal | 0.050 ± 0.005 | 2.0 | 0.09 ± 0.02 | 25.0 (2/3 at grid max) | 0.869 | 50.0 | 30.6 | 0.933 [0.91, 0.95] | 61.1 |
| v_st | Cholesky | 0.133 ± 0.013 | 5.5 | 0.12 ± 0.01 | 12.0 | 0.963 | 65.6 | 10.9 | 0.933 [0.92, 0.95] | 59.3 |
| v_lum | Diagonal | 0.086 ± 0.003 | 1.5 | 0.06 ± 0.01 | 21.3 | 0.955 | 32.8 | 19.4 | 0.937 [0.92, 0.95] | 30.1 |
| v_lum | Cholesky | 0.129 ± 0.019 | 2.2 | 0.07 ± 0.04 | 13.6 | 0.948 | 30.1 | 12.9 | 0.937 [0.92, 0.95] | 28.6 |

Calibrated columns are seed means (seed SDs in `results/tables.md`; the bracketed CI is the mean of the per-seed patient-bootstrap bounds). The orig240 version of this table is in `results/tables.md`; its patient-bootstrap CIs are 3 to 4 times wider (e.g. T2_st Cholesky 0.92 [0.86, 0.97]).

**Reference intervals at σ = 0.05** (experiments D and E):

| Param | Fixed interval coverage (large2000) | Fixed interval width % | Information-limit width % (BCRB) | Best calibrated model width % (sep. cohort) |
|---|---|---|---|---|
| D_ep | 0.951 | 45.1 | 43.0 | 80.3 (Chol) |
| D_st | 0.965 | 54.3 | 48.5 | 87.5 (Chol) |
| D_lum | 0.936 | 45.2 | 44.5 | 80.3 (Chol) |
| T2_ep | 0.938 | 54.1 | 48.3 | 68.6 (Diag) |
| T2_st | 0.958 | 67.8 | 58.4 | 86.1 (Chol) |
| T2_lum | 0.948 | 68.9 | 68.0 | 121.6 (Chol) |
| v_ep | 0.934 | 67.2 | 36.3 | 48.9 (Chol) |
| v_st | 0.930 | 69.6 | 48.1 | 59.3 (Chol) |
| v_lum | 0.915 | 67.4 | 20.3 | 28.6 (Chol) |

Repo-style block summaries (large2000, σ = 0.05): calibration Pearson r (voxel-mean SD vs voxel-mean |error|) D −0.17 / −0.24, T2 −0.10 / −0.38, v 0.04 / 0.16 (Diagonal / Cholesky); pooled raw coverage D 0.16 / 0.37, T2 0.11 / 0.27, v 0.06 / 0.13.

Reading:

- Raw coverage: Cholesky > Diagonal on 9/9 parameters on both cohorts (seed SDs overlap only for v_lum on orig240). Both are far below 0.95.
- Error-vs-σ: posterior SD tracks error weakly and only for v_ep (ρ ≈ 0.24 to 0.29) and v_st (0.09 to 0.12). For D_ep, D_st and T2_lum (Cholesky) the correlation is negative: voxels with larger posterior SD have smaller errors.
- The v2.0 replica still hits the τ = 25 ceiling for Diagonal (D_ep, D_st, T2_st, v_ep, v_st in 2 or 3 of 3 seeds), so its 0.87 to 0.92 calibrated coverages for Diagonal are grid artefacts. With a wide grid and a separate calibration cohort, Diagonal and Cholesky both reach 0.93 to 0.96 on every parameter and are statistically indistinguishable (overlapping patient-bootstrap CIs on all 9).
- Calibrated D and T2 intervals are 69 to 142% of the physiological range and wider than the fixed data-independent interval (45 to 69%) for all six parameters, at the same coverage. They carry no voxel-specific information. Calibrated v intervals (29 to 61%) are narrower than the fixed interval (67 to 70%), but still 1.2 to 1.5 times the information limit.

Figure: `figures/fig2_coverage_width.png`.

## Task 5: NLLS fit diagnostics

| Cohort | σ | n | failed fits (masked by `valid`) | ≥ 1 of 8 params at a box bound | v_lum < 0 |
|---|---|---|---|---|---|
| orig240 | 0.01 / 0.03 / 0.05 / 0.08 | 240 | 0 / 0 / 0 / 0 | 238 / 240 / 240 / 240 | 4 / 4 / 19 / 20 |
| large2000 | 0.01 / 0.03 / 0.05 / 0.08 | 2000 | 2 / 1 / 1 / 0 | 1981 / 1996 / 1999 / 2000 | 34 / 143 / 218 / 302 |

The `valid` mask removes almost nothing (0 voxels on the original test set). The real NLLS problem is that essentially every fit ends with at least one parameter on its box bound, and up to 15% of fits violate the simplex (v_lum < 0). The NLLS baseline is a single-start bounded fit at the prior centre, as in the repo.

## Differences from repo-reported numbers

- **Live notebook outputs reproduce.** NLLS on orig240 matches NB1 cell [10] to 3 decimals in R² (σ = 0.05: 0.924 / 0.811 / 0.238 vs 0.923 / 0.810 / 0.238; T2 MAE 91.66 vs 91.81). The residual is scipy version. The constant-predictor numbers match the research brief exactly.
- **Trained-model pooled metrics fall inside the seed spread of the repo's single runs.** σ = 0.05, orig240: PIA 0.984 / 0.964 / 0.586 (repo NB1 0.983 / 0.955 / 0.580; NB2 0.984 / 0.960 / 0.593); PIVA-Diagonal 0.988 / 0.968 / 0.579 (NB1 Tight 0.988 / 0.962 / 0.555; NB2 Diagonal 0.987 / 0.966 / 0.577); Cholesky 0.989 / 0.956 / 0.598 (NB2 0.988 / 0.957 / 0.568).
- **The README / RESULTS.md "Final comparison" table does not reproduce.** PIVA-Tight 0.952 / 0.788 / 0.254 with T2 MAE 94.9 is far outside our 3-seed range (0.988 / 0.968 / 0.579, T2 MAE 32.8). The stale table should not be cited.
- **Raw coverage is close to NB2.** Diagonal orig240 0.07 to 0.30 (NB2 0.02 to 0.35); Cholesky 0.12 to 0.66 (NB2 0.09 to 0.63). The NB1 PIVA-Tight block coverage (D 0.156, T2 0.085, v 0.053) matches our pooled Diagonal values (0.16 / 0.11 / 0.06).
- **NB2's calibrated coverage of 0.83 to 0.98** came from a 120-voxel half of the test set with a τ grid that Diagonal saturates. With a separate calibration cohort and a wider grid, both models reach 0.93 to 0.96 and the "Cholesky improved calibrated coverage on 7/9" verdict disappears.
- **Seed-to-seed variation** in v R² (SD 0.01 to 0.04) is the same size as the PIVA-vs-PIA differences the repo reports. Single-run comparisons in the repo are not interpretable at that level.
- Repo docs say the β warm-up is 100 epochs; the main PIVA code has none. We used no warm-up (the code).

## Reviewer-requested experiments

All runs below **change the original recipe** and were trained with **one seed (0)** unless noted. Same data, test cohorts, MC count, and evaluation as the core audit. Labels: `X:<covariance>+<v map>|<loss>|<reconstruction target>`.

### A. Likelihood scale (with A2: noisy target, Rician)

The repo loss is `F.mse_loss(recon, clean) + 0.1·KL` on signals scaled ×1000. Read as a Gaussian ELBO, that is a likelihood SD of sqrt(16·0.1/2) ≈ 0.89 signal units against a simulated noise SD of 50 at σ = 0.05, so the likelihood is about 56 times too sharp. We trained with a **proper Gaussian NLL with fixed SD s** (loss per voxel = Σ_k (recon_k − target_k)² / 2s² + KL, β = 1), s ∈ {15.49, 28.28, 50}, equivalent to β ∈ {30, 100, 312.5} in the repo loss up to a constant factor. That constant interacts with the grad-norm clip (1.0) and the L2 weight decay, so it is not bit-equivalent to changing β. A2 variants reconstruct the **noisy input** instead of the clean signal, with either the Gaussian NLL (s = 50) or a **Rician NLL** (σ = 50 signal units, log I0 via `i0e`). Note the training noise varies per voxel (σ ~ U(0.01, 0.08), i.e. SD 10 to 80 signal units), so a fixed s = 50 is matched to the evaluation σ, not to each training voxel.

large2000, σ = 0.05, no post-hoc calibration:

| Model | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|
| *Raw coverage* | | | | | | | | | |
| Cholesky, original (3 seeds) | 0.29 | 0.19 | 0.62 | 0.18 | 0.15 | 0.49 | 0.13 | 0.13 | 0.13 |
| X:Chol+ILR, Gauss s = 15.5 | 0.92 | 0.94 | 1.00 | 0.86 | 0.95 | 1.00 | 0.58 | 0.54 | 0.60 |
| X:Chol+ILR, Gauss s = 28.3 | 0.99 | 0.98 | 1.00 | 0.96 | 0.99 | 1.00 | 0.75 | 0.72 | 0.76 |
| X:Chol+ILR, Gauss s = 50, clean | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.88 | 0.85 | 0.90 |
| X:Chol+ILR, Gauss s = 50, noisy | 0.99 | 0.99 | 1.00 | 0.96 | 0.97 | 1.00 | 0.92 | 0.90 | 0.93 |
| X:Chol+ILR, Rician s = 50, noisy | 0.99 | 0.98 | 1.00 | 0.97 | 0.99 | 1.00 | 0.90 | 0.88 | 0.92 |
| X:Diag+softmax, Gauss s = 50, clean | 0.86 | 0.96 | 1.00 | 0.97 | 0.93 | 1.00 | 0.71 | 0.75 | 0.90 |
| X:Diag+softmax, Rician s = 50, noisy | 0.93 | 0.93 | 1.00 | 0.94 | 0.89 | 1.00 | 0.75 | 0.75 | 0.88 |
| *Raw width, % of range* | | | | | | | | | |
| Cholesky, original (3 seeds) | 13 | 8 | 34 | 7 | 8 | 32 | 4 | 6 | 2 |
| X:Chol+ILR, Gauss s = 50, clean | 112 | 103 | 120 | 96 | 108 | 119 | 38 | 45 | 21 |
| X:Chol+ILR, Gauss s = 50, noisy | 109 | 97 | 120 | 93 | 102 | 118 | 39 | 47 | 21 |
| X:Chol+ILR, Rician s = 50, noisy | 110 | 100 | 120 | 93 | 106 | 119 | 36 | 44 | 21 |
| Information limit (BCRB) | 43 | 49 | 45 | 48 | 58 | 68 | 36 | 48 | 20 |
| *Per-parameter R²* | | | | | | | | | |
| Cholesky, original (3 seeds) | -1.28 | -0.86 | -1.25 | -0.35 | -0.67 | -0.91 | 0.58 | 0.40 | 0.89 |
| X:Chol+ILR, Gauss s = 50, clean | -0.57 | -0.31 | -0.23 | -0.11 | -0.10 | -0.14 | 0.61 | 0.42 | 0.90 |
| X:Chol+ILR, Gauss s = 50, noisy | -0.67 | -0.48 | -0.29 | -0.85 | -0.31 | -0.17 | 0.69 | 0.53 | 0.91 |
| X:Chol+ILR, Rician s = 50, noisy | -0.69 | -0.52 | -0.26 | -0.42 | -0.11 | -0.15 | 0.69 | 0.51 | 0.91 |

Spearman ρ(SD, |error|) with s = 50 Cholesky: D and T2 between −0.16 and 0.02 (no signal), v_ep 0.16 to 0.19, v_st 0.12 to 0.15. Final KL drops from 21.5 (original) to 2.7 to 3.0 (s = 50). Full tables, orig240, and the Diagonal series are in `results/tables.md`. Figure: `figures/fig4_likelihood_scale.png`.

**Verdict, stated plainly.** The noise-matched likelihood does fix the raw overconfidence. It does not produce useful D or T2 uncertainty. For all six D and T2 parameters the s = 50 posterior is about as wide as the whole physiological box (93 to 120% of range; a uniform distribution over the box is 113%), 1.7 to 2.7 times the information limit and wider than the population prior. Final KL of about 3 nats across 8 latent dimensions means the D and T2 latents have collapsed to the N(0, 1) prior pushed through tanh. That is a correct statement that the data say little about D and T2, but it uses the VAE prior instead of the population prior, so it is wider than it needs to be. For v the result is good: raw coverage 0.85 to 0.93 with widths of 36 to 47% (v_ep, v_st) and 21% (v_lum), matching the information limit (36 / 48 / 20%), with no post-hoc step. Accuracy on v also improves slightly (v_st R² 0.42 to 0.53 vs 0.40). These are single-seed results; they need the 3-seed treatment before the abstract leans on them.

**Clean vs noisy target.** Under additive Gaussian noise, E_noise‖x − f(θ)‖² = ‖S_clean − f(θ)‖² + const, so the clean-target loss equals the noisy-target ELBO in expectation and only lowers gradient variance. Rician noise breaks this at low SNR (the Rician floor biases the noisy target upward). Empirically the target matters little: noisy-target and Rician runs give coverage and widths within a few points of the clean-target run, with slightly better v coverage (0.90 to 0.93 vs 0.85 to 0.90) and slightly better v R². Gaussian vs Rician NLL on the noisy target is a wash in this experiment.

Diagonal at matched likelihood is worse than Cholesky: lower v coverage (0.71 to 0.90), and at s = 15.5 and 28.3 the Diagonal runs lose v_ep and v_st entirely (R² −0.28 to 0.01) while D_ep R² drops to −9.3 and −5.3.

### B. Ablation: covariance vs ILR (original loss, β = 0.1)

large2000, σ = 0.05, raw coverage (Diag+softmax and Chol+ILR are the 3-seed originals):

| Model | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|
| Diag + softmax (original Diagonal) | 0.07 | 0.08 | 0.33 | 0.09 | 0.06 | 0.18 | 0.05 | 0.05 | 0.09 |
| X:Diag + ILR | 0.07 | 0.08 | 0.49 | 0.08 | 0.06 | 0.16 | 0.05 | 0.05 | 0.09 |
| X:Chol + softmax | 0.32 | 0.17 | 0.73 | 0.21 | 0.14 | 0.56 | 0.12 | 0.11 | 0.12 |
| Chol + ILR (original Cholesky) | 0.29 | 0.19 | 0.62 | 0.18 | 0.15 | 0.49 | 0.13 | 0.13 | 0.13 |

The whole raw-coverage gain comes from the full-covariance posterior. The ILR map alone changes nothing (Diag+ILR ≈ Diag+softmax), and removing it from the Cholesky model changes nothing either (Chol+softmax ≈ Chol+ILR). Accuracy follows the same pattern (v_ep R² 0.48 Diag+ILR, 0.53 Chol+softmax; full table in `results/tables.md`). The repo's "98.7% of v variance in the softmax null direction" motivation for ILR has no measurable consequence here.

### C. Calibration hygiene

Done in the core uncertainty table (columns "sep-cohort"): τ fit on calib2000 (seed 4242) or calib240 (seed 31337), applied to the whole test cohort, patient-level bootstrap CIs. On large2000 all 18 (model, parameter) combinations land in 0.928 to 0.961, with CIs of about ±0.01 to 0.02. On orig240 the CIs are ±0.03 to 0.06, so the 240-voxel test set cannot distinguish the two models' calibrated coverage. The v2.0 within-test voxel-split procedure, with its τ ≤ 25 ceiling, under-covers Diagonal by up to 8 points (v_st 0.869) for purely mechanical reasons.

### D. Data-independent fixed interval

The central 95% of the training-set parameter distribution, used as the same interval for every voxel, covers 0.92 to 0.97 of the test truths at 45 to 70% of the range (reference table above). Every calibrated PIVA interval for D and T2 is wider than this at the same coverage. For v the calibrated PIVA intervals are narrower (29 to 61% vs 67 to 70%), so v is the only place where PIVA's uncertainty beats a trivial baseline.

### E. Information limit (review of `support_scripts/fisher2.py`)

`fisher2.py` is correct for what it computes: a Bayesian CRB (Fisher plus a Gaussian prior) at a single point (prior centre, v = 1/3), Gaussian noise, a diagonal prior using the test-cohort marginal SDs, and 8 parameters (no v_lum). Rerun output at σ = 0.05: posterior/prior SD ratio 0.90 to 1.00 for D and T2, 0.51 (v_ep) and 0.58 (v_st), matching its claim of "0 to 10% for D and T2, about 50% for v". Approximations: (1) single-point Fisher instead of the expected Fisher the van Trees bound requires, (2) Gaussian instead of Rician Fisher information (Rician info at per-channel SNR 0.5 is 20% of the Gaussian value, and several channels sit near that), (3) diagonal prior, which ignores the strong v_ep / v_st anti-correlation.

`fisher_crb.py` reproduces the fisher2 numbers to 0.1 percentage point, then removes all three approximations: Fisher averaged over the 2000 large-cohort voxels, Rician per-channel information (computed by quadrature), full training-set prior covariance, and v_lum by the delta method. Conclusions are unchanged: BCRB/prior SD ratio 0.87 to 0.99 for D and T2 and 0.30 to 0.66 for v (table above; all σ levels in `results/crb_widths.csv`). Without the prior (pure CRB), the 95% widths are 129 to 1198% of range for D and T2: the protocol alone cannot localise them inside the box. Caveat: the BCRB is a Gaussian approximation and the real prior is bounded and clipped, so treat the widths as a yardstick, not a hard floor.

## Round-2 follow-up: per-voxel likelihood and 3-seed SD-50

**What changed (recipe changes, labelled as such).** Two Cholesky+ILR and two Diagonal+softmax configurations, 3 seeds each (0, 1, 2), clean reconstruction target, Gaussian NLL, β = 1:

- **SD-50**: fixed likelihood SD of 50 signal units (experiment A). Seeds 1 and 2 are new; seed 0 is the round-1 run.
- **Per-voxel SD**: each training voxel's own simulated noise SD (σ × 1000, σ ~ U(0.01, 0.08)). The orchestrator's `extras_models.training_sigmas` recovers it by replaying the first `default_rng(1000 + rep)` draw of `augment_clean_signals`. I checked that this matches the generator code (σ is that function's first draw). The network never receives σ: at test time it has to infer the noise level from the signal.

Evaluation: the 10 new checkpoints were evaluated with `evaluate.py` through a symlink folder (`runs_r2/`, 10 processes, 2.3 min). The new rows were appended to `results/metrics_long.csv` and `results/calibration_long.csv` (pre-merge copies: `*.bak_r1`). `round2.py` computes everything below from the saved posterior means and SDs in `preds/`. Full tables are in `results/round2_tables.md`; raw numbers are in `results/round2_long.csv`. Everything is on large2000, mean ± SD over 3 seeds, with no post-hoc calibration. Box-clipped width = mean over voxels of the 95% interval intersected with the physiological box, as % of the range. For v the clipping changes nothing (posterior means stay inside [0, 1] with room). For D and T2 it caps the width near 100%.

### (a) Volume-fraction coverage and width across σ

Cholesky+ILR (`figures/fig5_vfraction_sigma_sweep.png`):

| σ | Model | v_ep cov | v_st cov | v_lum cov | v_ep width % | v_st width % | v_lum width % |
|---|---|---|---|---|---|---|---|
| 0.01 | SD-50 | 0.942 ± 0.004 | 0.935 ± 0.005 | 0.981 ± 0.001 | 37.6 | 44.7 | 21.7 |
| 0.01 | per-voxel | 0.640 ± 0.047 | 0.666 ± 0.050 | 0.907 ± 0.015 | 19.6 ± 0.8 | 22.8 ± 0.8 | 10.8 ± 0.4 |
| 0.01 | BCRB limit | | | | 13.4 | 16.3 | 6.6 |
| 0.03 | SD-50 | 0.913 ± 0.002 | 0.897 ± 0.006 | 0.961 ± 0.002 | 37.6 | 44.8 | 21.8 |
| 0.03 | per-voxel | 0.812 ± 0.018 | 0.792 ± 0.018 | 0.896 ± 0.011 | 29.3 ± 0.8 | 34.0 ± 0.8 | 16.4 ± 0.4 |
| 0.03 | BCRB limit | | | | 27.4 | 36.1 | 14.6 |
| 0.05 | SD-50 | 0.880 ± 0.005 | 0.855 ± 0.005 | 0.912 ± 0.013 | 37.8 | 44.9 | 21.8 |
| 0.05 | per-voxel | 0.876 ± 0.004 | 0.842 ± 0.002 | 0.907 ± 0.004 | 37.2 ± 0.4 | 43.9 ± 0.2 | 21.4 ± 0.2 |
| 0.05 | BCRB limit | | | | 36.3 | 48.1 | 20.3 |
| 0.08 | SD-50 | 0.835 ± 0.007 | 0.805 ± 0.009 | 0.817 ± 0.013 | 37.9 | 44.8 | 21.8 |
| 0.08 | per-voxel | 0.898 ± 0.005 | 0.876 ± 0.001 | 0.886 ± 0.007 | 43.3 ± 0.2 | 51.5 ± 0.1 | 25.5 ± 0.3 |
| 0.08 | BCRB limit | | | | 45.2 | 58.4 | 27.2 |

(SD-50 width seed SDs are 0.0 to 0.4.)

**Answer to the reviewers' question.** Yes, per-voxel-SD widths track the information limit across σ, and SD-50 widths do not. SD-50 widths are flat (37.7 / 44.8 / 21.8% at every σ), so its coverage falls from 0.94 to 0.98 at σ = 0.01 to 0.81 to 0.84 at σ = 0.08. Per-voxel widths grow with σ and sit within 1 to 7 points of the BCRB from σ = 0.03 to 0.08, and they keep coverage at 0.88 to 0.90 at σ = 0.08. **But the per-voxel model fails at high SNR.** At σ = 0.01 its v_ep and v_st coverage is only 0.64 and 0.67, even though its widths are still 1.4 to 1.6 times the BCRB. That means its point errors at σ = 0.01 are larger than the bound allows: v_ep R² is 0.67 ± 0.04 against 0.74 ± 0.02 for SD-50 at the same σ. My guess is that σ = 0.01 sits at the lower edge of the training noise range, so the network sees few training voxels that clean. I have not tested this. Neither likelihood reaches 0.95 on v at every σ, and neither reaches it at σ = 0.05.

The Diagonal per-voxel model is much worse at high SNR: at σ = 0.01, v_ep coverage is 0.08 at 8.9% width, and v_ep R² is −0.10 ± 0.05. Per-voxel scaling only helps with the full-covariance posterior.

### (b) SD-50 headline numbers, 3 seeds (supersedes the single-seed numbers in experiment A)

Cholesky+ILR SD-50, σ = 0.05, large2000:

| | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|
| raw coverage | 0.998 | 0.997 | 1.000 | 0.996 | 0.993 | 0.997 | 0.880 ± 0.005 | 0.855 ± 0.005 | 0.912 ± 0.013 |
| raw width % (unclipped) | 111.6 | 102.9 | 120.2 | 96.2 | 108.5 | 119.0 | 37.8 | 45.1 | 21.8 |
| box-clipped width % | 95.3 | 91.3 | 98.8 | 88.0 | 94.9 | 98.6 | 37.8 | 44.9 | 21.8 |
| BCRB width % | 43.0 | 48.5 | 44.5 | 48.3 | 58.4 | 68.0 | 36.3 | 48.1 | 20.3 |
| R² | −0.58 ± 0.00 | −0.34 ± 0.03 | −0.23 ± 0.00 | −0.10 ± 0.01 | −0.11 ± 0.01 | −0.14 ± 0.00 | 0.627 ± 0.018 | 0.431 ± 0.014 | 0.904 ± 0.002 |
| Spearman ρ(SD, \|err\|) | −0.03 | −0.10 | −0.03 | 0.03 | −0.01 | −0.05 | (see round2_long.csv) | | |

The single-seed round-1 values hold (seed SDs ≤ 0.013 on v coverage). Diagonal+softmax SD-50 is clearly worse on v: coverage 0.71 / 0.75 / 0.91 and R² 0.46 / 0.30 / 0.91.

### (c) Per-compartment accuracy, σ = 0.05

| Model | D_ep | D_st | D_lum | T2_ep | T2_st | T2_lum | v_ep | v_st | v_lum |
|---|---|---|---|---|---|---|---|---|---|
| Chol+ILR SD-50 | −0.575 ± 0.003 | −0.339 ± 0.033 | −0.225 ± 0.003 | −0.103 ± 0.005 | −0.113 ± 0.011 | −0.137 ± 0.001 | 0.627 ± 0.018 | 0.431 ± 0.014 | 0.904 ± 0.002 |
| Chol+ILR per-voxel | −0.593 ± 0.014 | −0.313 ± 0.017 | −0.224 ± 0.001 | −0.130 ± 0.014 | −0.111 ± 0.017 | −0.133 ± 0.000 | 0.628 ± 0.006 | 0.429 ± 0.006 | 0.904 ± 0.001 |
| Cholesky original | −1.276 ± 0.058 | −0.861 ± 0.120 | −1.251 ± 0.108 | −0.349 ± 0.129 | −0.668 ± 0.006 | −0.914 ± 0.107 | 0.582 ± 0.017 | 0.397 ± 0.016 | 0.886 ± 0.006 |
| PIA (for reference) | −1.41 ± 0.22 | −1.30 ± 0.13 | −2.08 ± 0.34 | −0.53 ± 0.10 | −0.31 ± 0.02 | −0.53 ± 0.06 | 0.612 ± 0.017 | 0.484 ± 0.015 | 0.899 ± 0.005 |

At σ = 0.05 the two likelihoods give the same accuracy. Both improve on the original Cholesky for every parameter, mostly because D and T2 predictions move closer to the prior centre. D and T2 R² are still negative, and MAE is still above the constant predictor (e.g. D_ep 0.046 vs 0.037, T2_lum 75.2 vs 70.8 ms). On v they tie PIA on v_ep and v_lum and trail it on v_st (0.43 vs 0.48). v R² across σ (round2_tables.md): per-voxel is 0.07 lower than SD-50 on v_ep at σ = 0.01 and 0.02 higher at σ = 0.08.

### (d) D and T2: still reverting to the prior

Yes, under both likelihoods. At σ = 0.05, Cholesky D/T2 unclipped widths are 95 to 120% of the range (a uniform over the box gives 113%). The SD of the posterior means across voxels is 0.36 to 0.76 of the true SD, meaning the predictions are shrunk toward the centre. Spearman ρ(SD, |error|) ranges from −0.10 to 0.03. The per-voxel model narrows its D/T2 intervals at σ = 0.01 (T2_ep 50%, D_ep 72% clipped). Even there, T2_ep (50% vs 54%) is the only one narrower than the fixed population interval, and coverage drops to about 0.89 to 0.90 for D_ep and T2_ep. In any form, no D/T2 interval here carries useful voxel-level information.

### (e) Coverage stratified by epithelium fraction (clinical check)

Raw v coverage split at the cohort median of true v_ep, Cholesky+ILR:

| σ | Model | v_ep: low / high v_ep | v_st: low / high | v_lum: low / high | v_ep width %: low / high |
|---|---|---|---|---|---|
| 0.05 | SD-50 | 0.82 / 0.94 | 0.79 / 0.92 | 0.92 / 0.91 | 31.9 / 43.7 |
| 0.05 | per-voxel | 0.81 / 0.94 | 0.77 / 0.91 | 0.91 / 0.91 | 31.6 / 42.8 |
| 0.08 | SD-50 | 0.76 / 0.91 | 0.73 / 0.88 | 0.81 / 0.82 | (round2_tables.md) |
| 0.08 | per-voxel | 0.85 / 0.95 | 0.81 / 0.94 | 0.88 / 0.89 | |

(Seed SDs ≤ 0.02.) Both models under-cover the epithelium and stroma fractions in low-epithelium voxels by 10 to 15 points, and their intervals are *narrower* there (32% vs 43%). This matches shrinkage toward the prior mean: low-v_ep truths sit below where the posterior mean is pulled, and the posterior does not widen to cover them. High-epithelium voxels, the more cancer-like tissue in HM-MRI, are close to nominal (0.91 to 0.95). D/T2 coverage is ≥ 0.98 in both strata for Cholesky (trivially, given box-wide intervals). The per-voxel model narrows the gap at σ = 0.08 (low stratum 0.85 vs 0.76) but does not close it.

## What the abstract can and cannot claim

Updated after round 2.

**Can claim (3 seeds, 2000-voxel test set):**
- With this 4b x 4TE protocol at SNR 20, the three volume fractions are recoverable (R² about 0.9 lumen, 0.5 to 0.6 epithelium, 0.35 to 0.5 stroma), and physics-informed networks beat NLLS on them.
- Compartment D and T2 cannot be identified from these data. Per-compartment R² is below zero for every method and every likelihood, and a Bayesian CRB shows the data shrink the population prior by at most 13%. This is the most important result of the audit, and it concerns the protocol, not the network.
- The original PIVA posterior is severely overconfident (raw coverage 5 to 62%). The cause is a loss that implies a likelihood about 56 times sharper than the simulated noise. A full-covariance posterior helps (all 9 parameters); ILR does not.
- With a noise-matched Gaussian likelihood and a Cholesky posterior, raw volume-fraction intervals reach 0.86 to 0.91 coverage at σ = 0.05, at widths equal to the information limit (3 seeds, no post-hoc step).
- With each voxel's own noise SD in the likelihood, interval width follows the information limit as SNR changes (within 1 to 7 points of the BCRB for σ = 0.03 to 0.08). A fixed SD gives flat widths that over-cover at high SNR and under-cover at low SNR. Figure 5 shows this directly.

**Cannot claim:**
- Calibrated, nominal 95% coverage from any likelihood variant without post-hoc scaling. The best raw v coverage across σ is 0.81 to 0.98 (SD-50) or 0.64 to 0.91 (per-voxel). The per-voxel model fails at σ = 0.01 (v_ep 0.64) despite intervals wider than the bound.
- Uniform calibration across tissue types. Both noise-matched models under-cover v_ep and v_st by 10 to 15 points in low-epithelium voxels, with narrower intervals there.
- Any pooled D or T2 R² or MAE as evidence of accuracy (a constant gets 0.98 to 0.99).
- That PIVA is more accurate than PIA. The best PIVA variants tie PIA on v_ep and v_lum and trail it on v_st. Nobody learns D or T2.
- Useful voxel-wise uncertainty for D or T2 under any variant. The noise-matched models fix coverage there only by returning a box-wide interval (95 to 120% of range) uncorrelated with error.
- That Cholesky-ILR improves *calibrated* coverage (indistinguishable after proper calibration), or that ILR matters.
- That per-voxel SD is uniformly better than SD-50. It wins at σ ≥ 0.05 and loses badly at σ ≤ 0.03.
- Anything about real tissue: the data are synthetic, generated by the same forward model the decoder uses.
- The README "Final comparison" numbers.

## Commands, runtime, hardware

Hardware: MacBook Pro, Intel Core i9-9880H (8 cores / 16 threads, 2.3 GHz), 64 GB RAM, CPU only. Software: conda env `pvae`, Python with torch 2.5.1 (OpenMP build, no MKL), numpy 2.3.1, scipy 1.16.0, scikit-learn 1.7.1.

```zsh
cd ismrm2027/analysis
# training (one checkpoint per line of the queue files; 8 jobs x 1 thread, highest throughput on this CPU)
./run_queue.sh      # queue 1, queue.txt: pia x3, chol x3 completed (diag x3 stalled, extras never launched)
./run_queue2.sh     # queue 2, queue2.txt: diag x3 + 12 reviewer experiments
#   single job example: python train_one.py --model chol --seed 0
#   experiment example: python train_one.py --model generic --cov chol --vmap ilr --lik_sd 50 --lik rician --target noisy --seed 0
OMP_NUM_THREADS=1 NLLS_PROCS=2 python run_nlls.py orig240 large2000   # NLLS, ~6 min
python fisher_crb.py                                                   # experiment E, ~30 s
OMP_NUM_THREADS=1 EVAL_PROCS=8 python evaluate.py                      # all checkpoints, ~3.5 min
python report.py                                                       # results/tables.md + figures
# round 2 (queue3.txt, trained by the orchestrator with the same xargs runner)
OMP_NUM_THREADS=1 EVAL_PROCS=10 EVAL_RUNS=runs_r2 EVAL_OUT=results_r2 python evaluate.py   # 10 new checkpoints, 2.3 min; rows merged into results/*.csv
python round2.py                                                       # results/round2_*.{csv,md} + fig5
```

Runtime: each 200-epoch training run took 114 to 191 min of wall time on one thread with 8 runs in parallel (per-run times in `results/tables.md`, Training summary). Queue 1 started 2026-10-09 13:54; queue 2 ran 2026-10-10 02:00 to 07:57. Total training wall time about 10 h including the stalled attempt.

Run notes:
- Queue 1's three Diagonal runs stalled after epoch 60 from subnormal-float arithmetic (subnormal sgemm is about 225 times slower on this CPU) and were killed. `train_one.py` now calls `torch.set_flush_denormal(True)` (disable with `PIVA_NO_FTZ=1`). **Queue-2 runs (diag x3 and all 12 generic runs) used FTZ; queue-1 runs (pia x3, chol x3) did not.** An A/B rerun of diag seed 0 gave bit-identical losses through epoch 60 with and without FTZ. `evaluate.py` also sets FTZ (inference only).
- Queue 1's 12 extra jobs never launched because BSD `xargs -I` limits each substituted command to 255 bytes; `run_queue2.sh` uses `xargs -S 2048`.
- Ignore `runs/stalled_queue1/`, `abtest_*/`, `smoke_q2/`.

## Files

- `piva_core.py`: repo code (generator, NLLS, PIA, PIVA-Diagonal, PIVA-CholeskyILR, training) plus seeding and evaluation helpers.
- `extras_models.py`: `GenericPIVA` (2x2 covariance x v-map ablation), Gaussian/Rician NLL training for experiments A, A2, B.
- `train_one.py`, `run_queue.sh`, `run_queue2.sh`, `queue.txt`, `queue2.txt`: training.
- `run_nlls.py`, `evaluate.py`, `fisher_crb.py`, `report.py`: evaluation and reporting.
- `runs/`: 21 checkpoints with training histories (`*.json`) and logs.
- `preds/`: NLLS predictions and PIVA posterior means/SDs per cohort and σ.
- `results/metrics_long.csv` (every metric, model, seed, cohort, σ), `results/summary_mean_sd.csv` (seed mean/SD), `results/calibration_long.csv` (per-parameter calibration, all three procedures, bootstrap CIs), `results/crb_widths.csv`, `results/tables.md` (all tables).
- `figures/fig1_per_parameter_accuracy.png`, `fig2_coverage_width.png`, `fig4_likelihood_scale.png`, `fig5_vfraction_sigma_sweep.png` (4 main figures; suggested for the abstract: fig1, fig2 and fig5). `figures/supplementary/fig3_vfraction_r2_vs_sigma.png` is supplementary.
- Round 2: `queue3.txt`, `round2.py`, `runs_r2/` (symlinks to the 10 new checkpoints), `results_r2/` (raw evaluate output for them), `results/round2_long.csv`, `results/round2_tables.md`.
- `repo_snapshot/`: copies of the two notebooks used.
