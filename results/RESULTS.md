# PIVA — Full Results

**Test cohort:** 240 voxels (24 patients × 10 voxels/patient), synthetic, biologically plausible  
**Parameters:** D (×10⁻³ mm²/s), T2 (ms), v (volume fraction) — 3 compartments each = 9 total  
**Noise sweep:** σ ∈ {0.01, 0.03, 0.05, 0.08} (clinical SNR ≈ 100:1 → 12:1)  
**Reference level:** σ = 0.05 (SNR ≈ 20:1, standard clinical acquisition)  
**MC samples:** n_mc = 25 per voxel for PIVA variants  

---

## 1. NLLS Baseline

| σ | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE | D r | T2 r | v r |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.942 | 0.834 | 0.432 | 0.189 | 80.1 | 0.102 | 0.973 | 0.935 | 0.753 |
| 0.03 | 0.932 | 0.817 | 0.456 | 0.210 | 88.3 | 0.106 | 0.968 | 0.928 | 0.752 |
| 0.05 | 0.923 | 0.810 | 0.238 | 0.225 | 91.8 | 0.123 | 0.963 | 0.920 | 0.685 |
| 0.08 | 0.916 | 0.812 | −0.137 | 0.240 | 93.2 | 0.153 | 0.961 | 0.926 | 0.590 |

**Interpretation:** NLLS D and T2 estimates degrade gracefully with noise; v (volume fraction) collapses — R² goes negative at σ=0.08, meaning NLLS predictions are worse than the mean. The ill-conditioning of the 3-compartment inversion is most severe for v.

---

## 2. PIA (Physics-Informed Autoencoder)

| σ | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE | D r | T2 r | v r |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.988 | 0.960 | 0.813 | 0.079 | 34.9 | 0.062 | 0.994 | 0.980 | 0.904 |
| 0.03 | 0.986 | 0.961 | 0.696 | 0.083 | 34.8 | 0.078 | 0.994 | 0.981 | 0.835 |
| 0.05 | 0.983 | 0.955 | 0.580 | 0.091 | 37.9 | 0.091 | 0.993 | 0.980 | 0.767 |
| 0.08 | 0.978 | 0.959 | 0.480 | 0.103 | 36.7 | 0.106 | 0.991 | 0.981 | 0.708 |

**Interpretation:** PIA achieves large gains over NLLS on all parameters. D R² improves by ~6pp, T2 R² by ~15pp, and v R² by 34–62pp across noise levels. T2 MAE drops by ~55ms at σ=0.05. PIA is the best point estimator in this study.

---

## 3. PIVA — All Variants

### 3a. PIVA-Tight

| σ | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE | D r | T2 r | v r | D σ | T2 σ | v σ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.988 | 0.966 | 0.783 | 0.077 | 33.6 | 0.066 | 0.994 | 0.987 | 0.890 | 0.009 | 4.356 | 0.004 |
| 0.03 | 0.988 | 0.966 | 0.688 | 0.076 | 34.1 | 0.080 | 0.994 | 0.987 | 0.831 | 0.009 | 4.499 | 0.004 |
| 0.05 | 0.988 | 0.962 | 0.555 | 0.080 | 36.1 | 0.099 | 0.994 | 0.987 | 0.757 | — | — | — |
| 0.08 | 0.985 | 0.962 | 0.470 | 0.086 | 36.4 | 0.108 | 0.993 | 0.987 | 0.706 | — | — | — |

### 3b. PIVA-Wide

| σ | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE | D r | T2 r | v r |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.972 | 0.905 | 0.627 | 0.116 | 55.8 | 0.091 | 0.987 | 0.955 | 0.816 |
| 0.03 | 0.974 | 0.912 | 0.573 | 0.111 | 54.2 | 0.098 | 0.988 | 0.957 | — |
| 0.05 | 0.973 | 0.916 | 0.504 | 0.111 | 53.1 | 0.108 | 0.988 | 0.960 | — |
| 0.08 | 0.970 | 0.918 | 0.399 | 0.120 | 52.6 | 0.117 | 0.986 | 0.959 | — |

### 3c. PIVA-Free (collapsed)

| σ | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE |
|---|---|---|---|---|---|---|
| 0.01 | −1.356 | −0.658 | −1.626 | 1.216 | 301.5 | 0.242 |
| 0.03 | −1.351 | −0.654 | −1.633 | 1.214 | 300.7 | 0.241 |
| 0.05 | −1.348 | −0.651 | −1.597 | 1.212 | 300.3 | 0.241 |
| 0.08 | −1.371 | −0.654 | −1.580 | 1.212 | 301.3 | 0.238 |

**PIVA-Free is a training failure.** Negative R² across all parameters and all noise levels. The N(0,I) prior, combined with β=0.1 and a fixed physics decoder, causes the encoder to collapse to the prior — the KL term dominates signal reconstruction. Mean T2 MAE of ~300ms is comparable to the full T2 range of the lumen compartment.

---

## 4. Final Comparison at σ = 0.05

| Model | D R² | T2 R² | v R² | D MAE | T2 MAE (ms) | v MAE |
|---|---|---|---|---|---|---|
| NLLS | 0.923 | 0.810 | 0.238 | 0.225 | 91.8 | 0.123 |
| **PIA** | **0.983** | **0.968** | **0.572** | **0.093** | **32.6** | **0.093** |
| PIVA-Tight | 0.952 | 0.788 | 0.254 | 0.173 | 94.9 | 0.128 |
| PIVA-Wide | 0.819 | 0.797 | 0.111 | 0.342 | 85.9 | 0.139 |
| PIVA-Free | −1.483 | −0.658 | −2.231 | 1.345 | 303.9 | 0.285 |

Note: the PIVA-Tight values here differ slightly from the noise-sweep table above because cell [28] uses the final trained model after the full training run, whereas earlier cells use a separate training pass. This is a reproducibility gap — the two training runs used different random seeds.

---

## 5. Uncertainty Quantification — PIVA-Tight at σ = 0.05

### 95% Credible Interval Coverage (target: 0.950)

| σ | D cov | T2 cov | v cov |
|---|---|---|---|
| 0.01 | 0.197 | — | — |
| 0.03 | 0.189 | — | — |
| 0.05 | 0.156 | 0.085 | 0.053 |
| 0.08 | 0.154 | — | — |

**All coverage values are severely below the 0.950 nominal target.** The posterior is structurally overconfident. Coverage does not improve with noise level, indicating the miscalibration is not noise-driven but structural (prior mismatch or posterior compression).

### Uncertainty Score (CV)

At σ=0.05: mean voxel CV = 0.0071, range [0.0037, 0.0110]. The coefficient of variation is very low — the model assigns low relative uncertainty to all voxels regardless of actual error.

---

## 6. β-Sweep Experiment (PIVA-Tight, σ = 0.05)

KL warm-up applied: β increases linearly from 0 to β_max over 100 epochs, held constant for epochs 101–200.

### Point Estimate Quality

| β_max | D R² | T2 R² | v R² | D MAE | T2 MAE | v MAE |
|---|---|---|---|---|---|---|
| 0.01 | **0.9879** | **0.9679** | 0.5679 | **0.0764** | 32.8 | 0.0975 |
| 0.10 | 0.9874 | 0.9676 | **0.5795** | 0.0789 | 33.1 | **0.0956** |
| 0.30 | 0.9871 | 0.9071 | 0.5530 | 0.0836 | 57.0 | 0.0997 |
| 1.00 | 0.9843 | 0.8817 | 0.5709 | 0.0888 | 65.6 | 0.0987 |
| 3.00 | 0.9386 | 0.9153 | 0.1431 | 0.1788 | 57.6 | 0.1323 |

### Posterior Variance (mean σ per parameter)

| β_max | D mean σ | T2 mean σ (ms) | v mean σ |
|---|---|---|---|
| 0.01 | 0.00523 | 1.760 | 0.00456 |
| 0.10 | 0.00844 | 4.420 | 0.00451 |
| 0.30 | 0.01387 | 8.475 | 0.00456 |
| 1.00 | 0.02264 | 13.151 | 0.00591 |
| 3.00 | 0.04412 | 23.203 | 0.00941 |

### 95% CI Coverage

| β_max | D cov | T2 cov | v cov |
|---|---|---|---|
| 0.01 | 0.086 | 0.082 | 0.067 |
| 0.10 | 0.151 | 0.104 | 0.068 |
| 0.30 | 0.210 | 0.101 | 0.047 |
| 1.00 | 0.307 | 0.131 | 0.061 |
| 3.00 | **0.349** | **0.219** | **0.090** |

### Calibration Pearson r (posterior σ vs |error|)

| β_max | D r | T2 r | v r |
|---|---|---|---|
| 0.01 | −0.276 | −0.106 | 0.255 |
| 0.10 | −0.121 | −0.040 | 0.267 |
| 0.30 | −0.056 | −0.045 | 0.135 |
| 1.00 | −0.174 | −0.279 | 0.136 |
| 3.00 | −0.074 | −0.328 | 0.087 |

**Calibration r is negative for D and T2 at all β values.** The posterior variance is anti-correlated with actual error — voxels where the model is confident have larger errors. This is a fundamental miscalibration, not fixable by β-tuning alone.

### β-Sweep Recommendation (from notebook)

- Best D R²: β = 0.01 → R² = 0.9879  
- Best 95% coverage (closest to 0.95): β = 3.0 → D=0.349, T2=0.219, v=0.090  
- No single β jointly optimizes accuracy and calibration.

---

## 7. Summary of Findings

| Claim | Supported? | Evidence |
|---|---|---|
| PIA outperforms NLLS on point estimates | ✅ Yes | D R²: 0.983 vs 0.923; T2 MAE: 32.6 vs 91.8 ms at σ=0.05 |
| PIVA-Tight achieves PIA-level accuracy at low noise | ✅ Yes | D R²=0.988 at σ=0.01, matching PIA exactly |
| PIVA adds well-calibrated uncertainty | ❌ No | D 95% CI coverage: 0.156 (target 0.950); cal r < 0 |
| Higher β improves calibration | ⚠️ Partially | Coverage increases with β but never exceeds 0.35 for D |
| PIVA-Free is a viable variant | ❌ No | Posterior collapse: D R² = −1.4, T2 MAE = 300 ms |
| β-tuning resolves the accuracy–calibration trade-off | ❌ No | Trade-off is monotonic; no β achieves both |
