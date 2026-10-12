# Provenance for abstract_v6.md (submission version)

Source abbreviations:
- `R` = `analysis/RESULTS_rerun.md`; `R2` = its "Round-2 follow-up" section.
- `T2` = `analysis/results/round2_tables.md`.
- `FIG` = `analysis/figures/final/FIGURES.md` (raw values in `final_numbers.json`).
- `C4`, `C5`, `C6` = `drafts/checks_v4_output.txt`, `checks_v5_output.txt`, `checks_v6_output.txt`. All three scripts are read-only on `analysis/`.

Claims unchanged from v5 keep the sources in `provenance_v5.md` (and v4/v3 before it). This file covers new or changed claims only.

## Text

| Claim (v6) | Value | Source |
|---|---|---|
| Impact: compartmental D and T2 "were not" recoverable | all D/T2 R² < 0 for every method and likelihood; BCRB shrinkage < 13% | R Task 1–3; T2 (c); C4 §4 |
| Impact: noise-matched likelihood "with full covariance helped without guaranteeing coverage" | CholeskyILR SD-50 0.86–0.91 at σ = 0.05; neither likelihood ≥ 0.95 across σ | T2 (a); R2 "Cannot claim" bullet 1 |
| Synopsis: "covered 86–91% at SNR 20, but not across SNR" | 0.855–0.912 at σ = 0.05; 0.805–0.981 across σ | T2 (a) |
| Methods: retraining "of both posteriors (same noise-free target)" | Diag+softmax and Chol+ILR, SD-50 and per-voxel, clean target | R2 opening paragraph; `analysis/queue3.txt` |
| Fraction R² across networks: about 0.9, 0.46–0.63, 0.30–0.48 | min Diag+softmax SD-50 0.455 / 0.304; max Chol per-voxel 0.628, PIA v_st 0.484; v_lum 0.886–0.908 | R Task 1–3; T2 (c) |
| "PIA best or tied" | v_st PIA 0.484 highest; v_ep 0.612 vs 0.628 ± 0.006; v_lum 0.899 vs 0.908 | T2 (c); R Task 1–3 |
| "(R² ≤ 0.24)" | 1 − 0.871² | C4 §4 |
| "within 0.05 of the benign and cancer epithelium values" | windows 0.18–0.28 (n = 432), 0.44–0.54 (n = 213) | FIG A6; C5 |
| Matched Diagonal coverage "(Diagonal: 0.71–0.91)" at σ = 0.05 | SD-50 0.709 / 0.746 / 0.907; per-voxel 0.709 / 0.740 / 0.900 | T2 (a) |
| "at σ = 0.01 point errors stayed about three times the limit" | per-voxel CholeskyILR RMSE / BCRB-implied SD = 3.16 (v_ep), 2.94 (v_st); 1.65 (v_lum) | C6 (`checks_v6.py`; BCRB SD = BCRB 95% width / 3.92) |
| "Epithelium coverage was 0.93–0.96 near the benign and cancer values but 0.71 below 0.18, under both likelihoods" | SD-50 0.944 / 0.955; per-voxel 0.933 / 0.942; below 0.18 (n = 539): 0.709 / 0.707 | FIG A6 table; coordinator message |
| "SD-50 intervals excluded the benign value in 49% of cancer-like voxels" | 0.485 | C5 |
| Discussion: "every estimator did worse than a constant" | MAE ratio > 1 on all six D/T2 for every method | FIG Fig. 2 |

## Figure captions

| Fig. | File | Facts in caption | Source |
|---|---|---|---|
| 1 | `fig1_schematic.png` | 16 signals, 4 b × 4 ΔTE; latent 9-D diagonal / 8-D Cholesky; prior box; implied SD 0.9 vs true 10–80; matched 50 or per voxel | FIG Fig. 1; rendered image (box reads "μ (8–9-D)"); R §A |
| 2 | `fig2_accuracy.png` | NLLS clipped at −2.5; MAE ratio networks 1.15–1.8, NLLS 2.9–5.0; D/T2 all worse than the constant | FIG Fig. 2 |
| 3 | `fig3_calibration_original_loss.png` | raw coverage 0.05–0.62; scaled 0.93–0.96 (0.928–0.961); D/T2 widths 68–90% vs fixed 45–69%; patient-bootstrap CIs | FIG Fig. 3 |
| 4 | `fig4_sigma_sweep.png` | SD-50 widths 2.7–3.3 × limit at σ = 0.01 (37.6/13.4 = 2.81, 44.7/16.3 = 2.74, 21.7/6.6 = 3.29); under-cover at σ = 0.08 (0.805–0.835); per-voxel follows the limit σ ≥ 0.03 and under-covers at σ = 0.01; dashed line "lower bound on error" | FIG Fig. 4 table and A1 guidance; rendered legend |
| 5 | `fig5_phantom.png` | disc truth 0.483, background 0.233; contrast kept 62% (original) and 92% (matched); upward shift +0.09 / +0.07 (0.325 − 0.233, 0.557 − 0.483); matched half-widths 0.184 / 0.246; false T2,st disc −11.4 ms (truth −0.4) | FIG Fig. 5 table (seed 0); coordinator message |
| Preview | `preview_3x2.png` | 1800 × 1200, 3:2, 185 KB, smallest text 21 pt | FIG Preview |

## Counts

`drafts/count_v6.py abstract_v6.md`, cross-checked with `wc -w` after splitting numeric ranges: body 728 normal / 746 worst case; Impact 36 / 36; Synopsis 94 / 96; captions 400–474 characters; title 106 characters.
