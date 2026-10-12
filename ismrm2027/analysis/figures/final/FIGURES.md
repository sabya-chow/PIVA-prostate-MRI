# Final figures: ISMRM 2027 abstract (v5 spec, checklist B1 to B6, A6)

Everything here is made by `analysis/final_figures.py`. It does inference only, on existing checkpoints and on the posterior predictions saved in `analysis/preds/`. The raw numbers behind every value below are in `final_numbers.json`, in this folder.

Shared conventions:
- No suptitles anywhere.
- Parameter labels are rendered as D_ep, D_st, D_lum, T2,ep, T2,st, T2,lum, v_ep, v_st, v_lum (with subscripts).
- Widths are box-clipped and given as "% of prior box".
- One font family (DejaVu Sans) across all figures.
- The test cohort is 2000 voxels from 200 unseen patients at σ = 0.05, unless a figure says otherwise.

| File | Pixels | Size | Checklist |
|---|---|---|---|
| `fig1_schematic.png` | 1800 × 1200 | 212 KB | B4 |
| `fig2_accuracy.png` | 2250 × 1560 | 119 KB | B1 |
| `fig3_calibration_original_loss.png` | 2250 × 1740 | 153 KB | B2 |
| `fig4_sigma_sweep.png` | 2250 × 1470 | 206 KB | B3 |
| `fig5_phantom.png` | 3750 × 1620 | 316 KB | B5 |
| `preview_3x2.png` | 1800 × 1200 (3:2) | 185 KB | B6 |

---

## Fig. 1: schematic (`fig1_schematic.png`)

**What it shows.** The PIVA pipeline, drawn to the v5 spec:
- 16-signal grid → encoder trapezoid → μ (8-D) and Σ = LLᵀ (Cholesky) or diag(σ²) → z = μ + Lε.
- Three output heads: tanh → D (3), tanh → T2 (3), log-ratio → softmax → v (3).
- The fixed signal equation, in green and marked "physics, no weights", then the output grid Ŝ.
- A loss bracket with "loss = −log p(x | θ) + KL[q ‖ N(0, I)]", plus three rows: original (red, "implied noise SD 0.9"), matched (green, "50, or per voxel"), and "true noise SD 10–80 (signal units)".
- A prior-box table: D 0.5 ± 0.2, 1.2 ± 0.5, 2.85 ± 0.15 µm²/ms; T2 45 ± 25, 70 ± 30, 750 ± 250 ms.

There are no results in this figure. All text is 12.5 to 16 pt at 200 dpi, and everything except two small sub-labels is at least 14 pt.

**Deviations from the spec.**
- The prior-box table sits bottom-left, not in the right margin. At 14 pt there was no room beside the loss rows without overlap.
- "μ (8-D)" follows the spec. It is exact for CholeskyILR; the Diagonal latent is 9-D. The caption can say "8-D (CholeskyILR)" if the writer wants it exact.

---

## Fig. 2: per-compartment accuracy (`fig2_accuracy.png`)

**What it shows.**
- Top panel: per-parameter R² (NLLS bars for D and T2 clipped at −2.5, as noted in the panel).
- Bottom panel: MAE divided by the MAE of the constant prior-centre predictor (dashed line at 1).
- Bars: constant (prior centre), constant (training mean), NLLS, PIA, PIVA-Diagonal, PIVA-CholeskyILR.
- Mean ± SD over 3 seeds (NLLS and the constants are deterministic).

**Numbers a caption may cite** (from `RESULTS_rerun.md`, Task 1–3):
- Every D and T2 R² is negative for every method.
- MAE ratio > 1 for all six D/T2 parameters for every method: networks 1.15 to 1.8, NLLS 2.9 to 5.0.
- Fraction R²: v_lum 0.886 to 0.899 (networks), 0.750 (NLLS); v_ep 0.49 to 0.61 (networks), 0.28 (NLLS); v_st 0.34 to 0.48 (networks), −0.25 (NLLS).
- Pooled constant scores R² 0.993 (D) and 0.975 (T2); this pooled number is not plotted.
- The figure does **not** show a σ trend. Per checklist A5, drop "falls smoothly with σ" from the caption.

---

## Fig. 3: original-loss calibration (`fig3_calibration_original_loss.png`)

**What it shows.**
- Top panel: 95% coverage for PIVA-Diagonal (orange) and PIVA-CholeskyILR (green), raw (light) and after σ-scaling (dark). The σ-scaling uses one τ per parameter, fit on the separate calibration cohort.
- Bottom panel: box-clipped width as % of the prior box, with the BCRB width (black bar) and the fixed interval (×).
- Error bars on **all** bars are patient-bootstrap 95% CIs: 1000 resamples of the 200 patients, applied to the seed-averaged per-voxel indicator.

**Numbers a caption may cite** (point estimates, 3 seeds pooled):
- Raw coverage:
  - Diagonal 0.05 to 0.33 (D_lum highest).
  - CholeskyILR 0.13 to 0.62 (D_lum 0.62, T2_lum 0.49, v 0.13 each).
- Raw width:
  - Diagonal 1.5 to 18%.
  - CholeskyILR 2.2 to 34%.
- After σ-scaling, coverage:
  - Diagonal 0.933 to 0.949.
  - CholeskyILR 0.928 to 0.961.
- After σ-scaling, clipped width:
  - D and T2: 68 to 90% (Diagonal 68.1 to 88.4; CholeskyILR 69.2 to 89.8).
  - v: Diagonal 48 / 61 / 30%; CholeskyILR 49 / 59 / 29% (v_ep / v_st / v_lum).
- Fixed interval width:
  - D and T2: 45 to 69%, at coverage 0.94 to 0.97.
  - v: 67 to 70%.
- BCRB width:
  - D and T2: 43 to 68%.
  - v: 36 / 48 / 20%.
- Plain reading: every scaled D/T2 interval is wider than the fixed interval. Only the fraction intervals are narrower than it.

---

## Fig. 4: σ sweep, fractions (`fig4_sigma_sweep.png`)

**What it shows.**
- Layout: columns v_ep, v_st, v_lum; top row raw 95% coverage (dotted line at 0.95); bottom row box-clipped width as % of the prior box.
- x-axis: test σ = 0.01, 0.03, 0.05, 0.08.
- Series: CholeskyILR with a fixed likelihood SD of 50 (orange squares) and with per-voxel likelihood SD (blue circles).
- The BCRB is dashed, labelled "lower bound on error", per the physics reviewer.
- Mean ± SD over 3 seeds. No post-hoc scaling.

**Numbers a caption may cite** (v_ep / v_st / v_lum):

| σ | SD-50 coverage | per-voxel coverage | SD-50 width % | per-voxel width % | BCRB % |
|---|---|---|---|---|---|
| 0.01 | 0.942 / 0.935 / 0.981 | 0.640 / 0.666 / 0.907 | 37.6 / 44.7 / 21.7 | 19.6 / 22.8 / 10.8 | 13.4 / 16.3 / 6.6 |
| 0.03 | 0.913 / 0.897 / 0.961 | 0.812 / 0.792 / 0.896 | 37.6 / 44.8 / 21.8 | 29.3 / 34.0 / 16.4 | 27.4 / 36.1 / 14.6 |
| 0.05 | 0.880 / 0.855 / 0.912 | 0.876 / 0.842 / 0.907 | 37.8 / 44.9 / 21.8 | 37.2 / 43.9 / 21.4 | 36.3 / 48.1 / 20.3 |
| 0.08 | 0.835 / 0.805 / 0.817 | 0.898 / 0.876 / 0.886 | 37.9 / 44.8 / 21.8 | 43.3 / 51.5 / 25.5 | 45.2 / 58.4 / 27.2 |

Caption guidance (A1):
- SD-50 does **not** over-cover at high SNR on v_ep or v_st (0.942 and 0.935 at σ = 0.01). Only v_lum over-covers (0.981).
- Accurate wording: "flat widths, about 2.7–3.3 times the limit at σ = 0.01, that under-cover at σ = 0.08".

---

## Fig. 5: synthetic phantom (`fig5_phantom.png`) and what it actually shows (B5)

**Construction.**
- Grid: 64 × 64.
- Tissue: a background benign-like region (target mean v = 0.232 / 0.45 / 0.318) with a central disc of radius 12 px that is cancer-like (target mean v = 0.488 / 0.40 / 0.112).
- Voxel v ~ Dirichlet(1 + 18m), with m solved so the Dirichlet mean equals the target. This is the generator's own voxel model. The epithelium targets are the Chatterjee 2018 values; the stroma/lumen split is my design choice, not from the literature.
- D and T2: one generator "patient" (patient shift plus voxel spread, clipped to the box), identical in both regions.
- Noise: Rician, σ = 0.05.
- Seeds: phantom seed 2027, noise seed 2028.
- MC: 25 samples, seed 1234.
- Rendered: seed-0 checkpoints, `chol_s0` (original loss) and `gen-chol-ilr-gauss50-clean_s0` (matched SD 50).
- The numbers below are given for seed 0 and as the mean over seeds 0 to 2, which were run on the same phantom.
- Row 1 (v_ep): (a) truth, (b) mean under the original loss, (c) its 95% half-width, (d) mean under matched SD 50, (e) its half-width.
- Row 2 (T2,st): panels (f) to (j) in the same order.
- Colour bars: v 0 to 0.7; half-width 0 to 0.3 (v); T2,st 40 to 100 ms; half-width 0 to 40 ms.

**Phantom truth.**
- v_ep: background 0.233, disc 0.483, contrast 0.250 (n = 3648 / 448 voxels).
- T2,st: 68.6 ms mean, SD 7.9; 68.2 ms in the disc vs 68.6 ms in the background (no disc in truth).

**What the rendered phantom shows** (seed 0; 3-seed mean in brackets):

| | original loss | matched SD 50 | per-voxel SD (not drawn) |
|---|---|---|---|
| v_ep background mean (truth 0.233) | 0.286 [0.294] | 0.325 [0.321] | 0.327 [0.324] |
| v_ep disc mean (truth 0.483) | 0.439 [0.449] | 0.557 [0.551] | 0.565 [0.555] |
| contrast (truth 0.250) | 0.153 [0.156] = 62% of truth | 0.232 [0.230] = 92% of truth | 0.238 [0.231] |
| v_ep 95% half-width, background / disc | 0.020 / 0.029 | 0.184 / 0.246 | 0.180 / 0.239 |
| v_ep coverage, background / disc | 0.17 / 0.24 [0.16 / 0.24] | 0.91 / 0.97 [0.92 / 0.98] | 0.89 / 0.93 |
| disc intervals excluding benign 0.232 | 99.8% | 75% [73%] | 77% [74%] |
| background intervals excluding cancer 0.488 | 99% | 47% [49%] | 49% |
| T2,st 95% half-width (box half-width 30 ms) | 2.3 ms | 33.0 ms | 32.8 ms |
| T2,st coverage | 0.16 | 1.00 | 1.00 |
| T2,st mean, disc − background (truth −0.4 ms) | −11.4 ms [−11.1] | −3.9 ms [−3.9] | |

**Honest reading for the caption (A7).**
- **Original loss.** It compresses the disc: contrast is 62% of truth (0.15 vs 0.25). Background is biased up and disc biased down. Half-widths are near zero (about 0.02 to 0.03), so 99.8% of disc intervals "confidently" exclude the benign value while covering only 24% of the truth.
- **Matched loss, contrast.** In this phantom it restores most of the contrast (92%), but it shifts **both** regions upward: background +0.09, disc +0.07. That differs from the cohort numbers in the text (0.29 / 0.47, i.e. benign up, cancer still slightly low). The disc overshoot is phantom-specific: one patient, and my stroma/lumen split. So call it "**largely** restores contrast while biasing both regions upward", not "restores". "Partly" is still defensible, and the benign-side upward bias (clinical review C3) is clearly visible in panel (d).
- **Matched loss, intervals.** Fraction intervals are wide (half-width 0.18 to 0.25) and cover 0.91 to 0.97. 75% of disc intervals exclude the benign value.
- **T2,st (bottom row).** Truth has no disc. The original loss nonetheless paints a disc into T2,st (−11 ms), with near-zero half-widths, meaning confident and wrong: composition leaks into T2. The matched loss mostly removes the artefact (−4 ms) and reports half-widths of about 33 ms against a 30 ms box half-width. That is the reversion to the prior box, and it is visible as a uniform panel (j).

---

## Preview figure (`preview_3x2.png`, B6)

- 3:2, 1800 × 1200 px, 185 KB.
- Smallest text: 21 pt colour-bar ticks, at a 9 × 6 in canvas and 200 dpi.
- Layout: 2 × 3 grid, epithelium fraction only.
  - Top row: truth / original / matched estimate.
  - Bottom row: a 21 pt text cell, original 95% half-width, matched 95% half-width.
  - Two colour bars (0 to 0.7 and 0 to 0.3).

**Why this panel set.** It is the most striking single message in the abstract: the original network looks plausible but compressed, with essentially zero uncertainty, while the matched network shows the lesion with visibly wider uncertainty. It reads at thumbnail size without axes or numbers.

I departed from the spec's "row 1 panels a, b, c, e". Including the matched *mean* (d) shows contrast as well as uncertainty, and a 2 × 3 grid of square images fits 3:2 better than 1 × 4.

---

## A6: fraction coverage by epithelium stratum

Setup:
- Model: CholeskyILR, σ = 0.05, raw intervals (no scaling).
- Seeds: 3, values are the mean (seed SD ≤ 0.017).
- Cohort: large test cohort (2000 voxels).
- Strata use the **true** v_ep.
- Benign and cancer-like windows are ±0.05 around the Chatterjee values (0.232, 0.488). Quartiles of true v_ep are 0.172 / 0.287 / 0.434.

| Stratum (true v_ep) | n | SD-50: v_ep / v_st / v_lum cov | per-voxel: v_ep / v_st / v_lum cov | v_ep bias (SD-50 / per-voxel / original) |
|---|---|---|---|---|
| < 0.18 | 539 | **0.709** / 0.714 / 0.911 | **0.707** / 0.685 / 0.900 | +0.119 / +0.120 / +0.101 |
| near benign, 0.18–0.28 | 432 | **0.944** / 0.876 / 0.922 | **0.933** / 0.868 / 0.916 | +0.059 / +0.060 / +0.035 |
| near cancer-like, 0.44–0.54 | 213 | **0.955** / 0.941 / 0.897 | **0.942** / 0.928 / 0.890 | −0.022 / −0.018 / −0.092 |
| Q1 (< 0.172) | 500 | 0.695 / 0.701 / 0.913 | 0.691 / 0.671 / 0.902 | +0.121 / +0.122 / +0.103 |
| Q2 (0.172–0.287) | 500 | 0.941 / 0.881 / 0.919 | 0.932 / 0.870 / 0.914 | |
| Q3 (0.287–0.434) | 500 | 0.942 / 0.913 / 0.931 | 0.937 / 0.903 / 0.934 | |
| Q4 (≥ 0.434) | 500 | 0.944 / 0.925 / 0.884 | 0.943 / 0.922 / 0.877 | −0.056 / −0.052 / −0.145 |
| all | 2000 | 0.880 / 0.855 / 0.912 | 0.876 / 0.842 / 0.907 | |

**Suggested wording:**

> "epithelium coverage was 0.94–0.96 near the benign and cancer values (SD 50; 0.93–0.94 per-voxel) but 0.71 below v_ep 0.18 (n = 539)"

This confirms the clinical reviewer's numbers (539, 0.71, 0.95). Both likelihoods behave alike.

The under-coverage below 0.18 comes from shrinkage: bias is +0.12 there, and intervals are narrower. It is not a benign-tissue failure. Benign-like voxels are still biased upward by +0.06, against +0.035 under the original loss.
