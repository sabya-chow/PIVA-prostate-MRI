# ISMRM 2027 Standard Abstract: submission version (v6)

Submission version after the final review round. It fixes checklist items A1 to A8 (`SUBMISSION_CHECKLIST.md`; map at the end of this file). The only remaining placeholders are the `[CONFIRM]` form fields, which the authors resolve (checklist section C). Numbers come from `analysis/RESULTS_rerun.md`, `analysis/results/*`, `analysis/figures/final/FIGURES.md` and `final_numbers.json`, plus the writer's check scripts `drafts/checks_v4.py`, `checks_v5.py` and `checks_v6.py` (all read-only on `analysis/`). Sources: `provenance_v6.md`.

---

## Title

What prostate HM-MRI data can and cannot determine: a simulation audit of physics-informed VAE uncertainty

## Authors and affiliations (ECHO form fields only; not shown to reviewers)

1. Aaron Garay (first author): Data Science Institute, University of Chicago, Chicago, IL, USA [CONFIRM: exact unit name]
2. Sabyasachi Chowdhury: Data Science Institute, University of Chicago, Chicago, IL, USA [CONFIRM: exact unit name]
3. Batuhan Gundogdu: Department of Radiology, University of Chicago, Chicago, IL, USA [CONFIRM]

## Category and keywords

- Abstract type: Standard
- Primary: 703 Diffusion > Diffusion Modeling
- Secondary: 311 Analysis Methods > Generative Models
- General keyword: Precision & Accuracy
- Additional keywords: Prostate; Relaxation; Simulations; Tissue Characterization
- Deep Dive into New Dimensions: do not opt in

## Impact (max 40 words)

In simulated prostate HM-MRI at SNR 20, tissue fractions were recoverable but compartmental diffusivity and T2 were not. A variational network's intervals were far too narrow; a noise-matched likelihood with full covariance helped without guaranteeing coverage.

## Synopsis (max 100 words, four parts)

**Motivation:** Epithelium fraction from hybrid multidimensional MRI can separate prostate cancer from benign tissue; a variational neural network could add per-voxel uncertainty.

**Goal(s):** Test, in simulation, what the data determine and whether that uncertainty holds.

**Approach:** We trained physics-informed networks on simulated signals, scored each compartment parameter, and compared 95% intervals with the information limit.

**Results:** No method estimated compartmental diffusivity or T2 better than a fixed population average. Initially, 95% intervals contained the truth in 5–62% of voxels; with a noise-matched loss, full-covariance tissue-fraction intervals covered 86–91% at SNR 20, but not across SNR.

---

## Introduction

Hybrid multidimensional MRI (HM-MRI) fits a three-compartment (epithelium, stroma, lumen) diffusion-relaxation model to a b-value × echo-time grid<sup>1,2</sup>; epithelium fraction separates cancer from benign tissue<sup>2</sup> and agrees with histology<sup>3</sup>. A physics-informed autoencoder (PIA) fits it far faster than nonlinear least squares (NLLS)<sup>4</sup> but returns point estimates. Variational HM-MRI has been prototyped<sup>5</sup>, and full-covariance or flow-based posteriors have been validated for CEST/MT and microstructure<sup>6,7</sup>, but variational posteriors tend to be too narrow<sup>8</sup>. In simulation, we test what the data determine, whether a three-compartment physics-informed variational autoencoder (PIVA) is calibrated, and why not.

## Methods

**Simulation.** $S=1000\sum_c v_c e^{-bD_c}e^{-\Delta TE/T2_c}$ was sampled at four b-values (0–1500 s/mm²) × four ΔTE (0–143 ms after minimum TE), as in ref. 4. With relative TE and known S<sub>0</sub>, v<sub>c</sub> are signal fractions at minimum TE, and fitting is easier than in vivo. Patient and voxel parameters were drawn hierarchically inside a prior box (Fig. 1). Rician noise: training σ ~ U(0.01, 0.08); test σ = 0.05 unless stated (SNR 20 at b = ΔTE = 0; 0.35 at b = 1500, longest TE). Test: 2000 voxels, 200 unseen patients.

**Models.** NLLS: single-start bounded least squares (1999/2000 fits ended on a bound). PIA: reimplementation of ref. 4, shortened training. PIVA-Diagonal: the PIA encoder with a diagonal Gaussian latent (Fig. 1); loss = squared error to the noise-free signal + 0.1·KL[q‖N(0, I)], a Gaussian likelihood with SD 0.9 signal units against noise SD 10–80. PIVA-CholeskyILR adds a Cholesky-factored full-covariance posterior<sup>6,9</sup> and log-ratio coordinates for v<sup>10</sup>. Calibrated-decoder retraining<sup>11</sup> of both posteriors (same noise-free target) set the likelihood SD to 50 (the test noise) or to each training voxel's own noise SD; the network never receives σ. Three initialisation seeds per model.

**Evaluation.** Per-compartment R² and MAE against a constant prior-centre predictor; coverage of mean ± 1.96 SD; width clipped to the prior box. Post hoc, one scale factor per parameter (σ-scaling<sup>12</sup>) was fit on a separate cohort. Widths were compared with a fixed interval (central 95% of the training distribution) and the information limit (Bayesian Cramér-Rao bound, BCRB; Rician Fisher information, population prior).

## Results

**Accuracy (Fig. 2).** Per compartment, every D and T2 R² was negative and the constant had the lowest MAE; pooled over compartments it scored R² 0.993 (D) and 0.975 (T2), above every model. By the BCRB, the data shrink the population prior SD by under 13% for D and T2 (R² ≤ 0.24), and without the prior the CRB 95% width is 1.3–12 times the box. Fractions were estimated, with PIA best or tied: networks reached R² about 0.9 (lumen), 0.46–0.63 (epithelium) and 0.30–0.48 (stroma), against NLLS 0.75, 0.28 and −0.25.

**Original loss (Fig. 3).** Raw coverage was 0.05–0.33 (Diagonal) and 0.13–0.62 (CholeskyILR). For voxels within 0.05 of the benign and cancer epithelium values (0.23, 0.49: volume fractions<sup>2</sup>, treated as signal fractions), CholeskyILR estimated 0.26 and 0.40, halving the contrast (shrinkage<sup>13</sup>), and covered 15% of cancer-like truths. σ-scaling reached 0.93–0.96 coverage, but D and T2 intervals (68–90% of the box) were wider than the fixed interval (45–69%).

**Noise-matched likelihoods (Fig. 4).** With SD 50, CholeskyILR fraction intervals covered 0.86–0.91 (Diagonal: 0.71–0.91) at σ = 0.05 without post-hoc scaling, at widths near the information limit, and partly restored contrast (0.29 and 0.47); D and T2 intervals spanned 88–99% of the box (posterior collapse<sup>14</sup>). Fixed-SD widths ignored σ: coverage fell from 0.94–0.98 (σ = 0.01) to 0.81–0.84 (σ = 0.08). Per-voxel SDs kept widths within 7 points of the limit for σ ≥ 0.03, but at σ = 0.01 point errors stayed about three times the limit, so epithelium and stroma coverage fell to 0.64–0.67. Epithelium coverage was 0.93–0.96 near the benign and cancer values but 0.71 below 0.18, under both likelihoods. SD-50 intervals excluded the benign value in 49% of cancer-like voxels.

## Discussion

The original posterior was overconfident mainly because its implied noise SD was 56 times too small, a known VAE pitfall<sup>11</sup>. Under this protocol at SNR 20, compartmental D and T2 are practically non-identifiable, and every estimator did worse than a constant, which pooled metrics hide. Limitation: data simulated with the decoder's own forward model (a best case). ROI-level coverage can be tested in vivo against registered histology fractions<sup>3</sup>.

## Conclusion

At SNR 20 on this protocol, HM-MRI fitters should report tissue fractions, not compartmental D or T2. Variational fraction intervals need a noise-matched likelihood and coverage checks across SNR and composition.

---

## Figures

All files are in `analysis/figures/final/` (built by `analysis/final_figures.py`; contents and citable numbers in `FIGURES.md`). Upload in this order. Captions were written from `FIGURES.md` and the rendered images.

**Figure 1.** `fig1_schematic.png` (1800 × 1200)

Caption: PIVA. An encoder maps 16 signals (4 b-values × 4 ΔTE) to a Gaussian posterior over latent z, diagonal (9-D) or Cholesky-factored (8-D). D and T2 are tanh-bounded to the prior box (table, centre ± half-width); fractions come from a softmax, via log-ratio coordinates for CholeskyILR. The decoder is the fixed signal equation. The original loss implies a noise SD of 0.9 signal units against a true 10–80; the matched loss uses 50 or each voxel's own noise SD.

**Figure 2.** `fig2_accuracy.png` (2250 × 1560)

Caption: Per-compartment accuracy, 2000 test voxels, σ = 0.05, mean ± SD over three seeds. Top: R² (NLLS D and T2 bars clipped at −2.5). Bottom: MAE relative to a constant predictor at the prior centre (dashed line); grey bars are this constant and the training mean. On all six D and T2 parameters every method is worse than the constant (networks 1.15–1.8×, NLLS 2.9–5.0×); only the fractions are estimated.

**Figure 3.** `fig3_calibration_original_loss.png` (2250 × 1740)

Caption: Original loss. Top: 95% coverage, raw (light) and after σ-scaling fit on a separate cohort (dark). Bottom: width as % of the prior box, clipped, with the information limit (BCRB, bar) and a fixed interval that ignores the signal (×). Error bars: patient-bootstrap 95% CIs. Raw coverage is 0.05–0.62. After scaling it is 0.93–0.96, but every D and T2 interval (68–90%) is wider than the fixed interval (45–69%); only fraction intervals are narrower.

**Figure 4.** `fig4_sigma_sweep.png` (2250 × 1470)

Caption: Fraction posteriors against test noise σ (SNR 100 to 12.5), CholeskyILR, no post-hoc scaling, mean ± SD over three seeds. Top: raw 95% coverage (dotted, 0.95). Bottom: width as % of the prior box; dashed, information limit (BCRB), a lower bound on error rather than a target width. A fixed likelihood SD of 50 gives flat widths, 2.7–3.3 times the limit at σ = 0.01, that under-cover at σ = 0.08. A per-voxel SD follows the limit for σ ≥ 0.03 but under-covers at σ = 0.01.

**Figure 5.** `fig5_phantom.png` (3750 × 1620)

Caption: Synthetic phantom (σ = 0.05): a cancer-like disc (epithelium signal fraction 0.48) in benign-like tissue (0.23); stromal T2 has no disc. Top: epithelium truth, means and 95% half-widths. The original loss keeps 62% of the contrast with near-zero half-widths. The matched loss (SD 50) keeps 92% but biases both regions upward (+0.09, +0.07), with half-widths of 0.18–0.25. Bottom: the original loss paints a false −11 ms stromal T2 disc; matched intervals span the prior box.

**Preview figure (ECHO, required).** `preview_3x2.png` (1800 × 1200, 3:2, 185 KB, smallest text 21 pt). Epithelium fraction only: truth, original and matched estimates (top row); original and matched 95% half-widths (bottom row). No caption is allowed.

---

## References

1. Wang S, Peng Y, Medved M, et al. Hybrid multidimensional T2 and diffusion-weighted MRI for prostate cancer detection. J Magn Reson Imaging. 2014;39(4):781-788. doi:10.1002/jmri.24212.
2. Chatterjee A, Bourne RM, Wang S, et al. Diagnosis of prostate cancer with noninvasive estimation of prostate tissue composition by using hybrid multidimensional MR imaging: a feasibility study. Radiology. 2018;287(3):864-873. doi:10.1148/radiol.2018171130.
3. Chatterjee A, Mercado C, Bourne RM, et al. Validation of prostate tissue composition by using hybrid multidimensional MRI: correlation with histologic findings. Radiology. 2022;302(2):368-377. doi:10.1148/radiol.2021204459.
4. Gundogdu B, Chatterjee A, Medved M, Bagci U, Karczmar GS, Oto A. Physics-informed autoencoder for prostate tissue microstructure profiling with hybrid multidimensional MRI. Radiol Artif Intell. 2025;7(2):e240167. doi:10.1148/ryai.240167.
5. Gundogdu B. PIvA: physics-informed variational autoencoder for mono-exponential HM-MRI. Unpublished source code; 2024.
6. Finkelstein A, Moneta R, Zohar O, Rivlin M, Zaiss M, Friedmann Morvinski D, Perlman O. Multiparameter uncertainty mapping in quantitative molecular MRI using a physics-structured variational autoencoder (PS-VAE). IEEE Trans Med Imaging. 2026. doi:10.1109/TMI.2026.3740952.
7. Jallais M, Palombo M. Introducing µGUIDE for quantitative imaging via generalized uncertainty-driven inference using deep learning. eLife. 2024;13:RP101069. doi:10.7554/eLife.101069.
8. Blei DM, Kucukelbir A, McAuliffe JD. Variational inference: a review for statisticians. J Am Stat Assoc. 2017;112(518):859-877. doi:10.1080/01621459.2017.1285773.
9. Kucukelbir A, Tran D, Ranganath R, Gelman A, Blei DM. Automatic differentiation variational inference. J Mach Learn Res. 2017;18(14):1-45.
10. Egozcue JJ, Pawlowsky-Glahn V, Mateu-Figueras G, Barceló-Vidal C. Isometric logratio transformations for compositional data analysis. Math Geol. 2003;35(3):279-300. doi:10.1023/A:1023818214614.
11. Rybkin O, Daniilidis K, Levine S. Simple and effective VAE training with calibrated decoders. Proc Mach Learn Res (ICML). 2021;139:9179-9189.
12. Laves M-H, Ihler S, Fast JF, Kahrs LA, Ortmaier T. Well-calibrated regression uncertainty in medical imaging with deep learning. Proc Mach Learn Res (MIDL). 2020;121:393-412.
13. Gyori NG, Palombo M, Clark CA, Zhang H, Alexander DC. Training data distribution significantly impacts the estimation of tissue microstructure with machine learning. Magn Reson Med. 2022;87(2):932-947. doi:10.1002/mrm.29014.
14. Lucas J, Tucker G, Grosse R, Norouzi M. Don't blame the ELBO! A linear VAE perspective on posterior collapse. Adv Neural Inf Process Syst (NeurIPS). 2019. arXiv:1911.02469.

## Acknowledgements (not shown to reviewers)

This work extends the physics-informed autoencoder for HM-MRI developed by B. Gundogdu and colleagues at the Grossman Center, University of Chicago [CONFIRM: formal centre name], and carries out the in-silico aim of the Center's 2024 proposal to develop and evaluate a physics-informed variational autoencoder (PIvA) for HM-MRI. B. Gundogdu advised the project. Code: https://github.com/sabya-chow/PIVA-prostate-MRI. Ref. 5 code: https://github.com/batuhan-gundogdu/PIA. [CONFIRM: funding/grant numbers; whether to acknowledge A. Chatterjee, G. Karczmar and A. Oto here (they are not authors).]

---

## Notes for authors (not part of submission)

- **Before submitting:** resolve every `[CONFIRM]` (checklist C1) and paste the body into ECHO to read its own word count, including the display equation (C3; our worst case is 746/750). The author list is fixed at Garay, Chowdhury and Gundogdu (C2, decided 2026-10-11).
- If ECHO's count comes out higher than ours, these clauses carry no claim in the Impact, Synopsis or Conclusion and can be cut first: "(same noise-free target)" (4 words), "(R² ≤ 0.24)" (3 words), or the parenthesis "(SNR 20 at b = ΔTE = 0; 0.35 at b = 1500, longest TE)", shortened to "(SNR 20 at b = ΔTE = 0)" (6 words).
- **Poster, not abstract:**
  - The benign-side upward bias under the matched loss (+0.035 → +0.059; Fig. 5 shows it).
  - Low SD–error rank correlation (ρ ≈ 0.16 for v_ep).
  - The single-seed ablation crediting full covariance rather than ILR.
  - The Diagonal per-voxel collapse at σ = 0.01.
- PS-VAE (ref. 6): mark "Early access" in the reference if IEEE Xplore lists it that way.

## Counts (computed by `drafts/count_v6.py`)

| Item | Count | Limit | OK |
|---|---|---|---|
| Title (characters) | 106 | 125 | yes |
| Impact (words) | 36 | 40 | yes |
| Impact, worst case: ranges = 2 words | 36 | 40 | yes |
| Synopsis incl. labels (words) | 94 | 100 | yes |
| Synopsis, worst case: ranges = 2 words | 96 | 100 | yes |
|   Body: Introduction (words) | 91 |  |  |
|   Body: Methods (words) | 252 |  |  |
|   Body: Results (words) | 281 |  |  |
|   Body: Discussion (words) | 68 |  |  |
|   Body: Conclusion (words) | 31 |  |  |
| Body total, excl. headings (words) | 723 | 750 | yes |
| Body total, incl. 5 section headings (words) | 728 | 750 | yes |
| Body incl. headings, worst case: ranges = 2 words | 746 | 750 | yes |
|   numeric ranges in body | 18 |  |  |
| Body total with markers left in (words) | 723 |  |  |
|   of which symbol-only tokens (=, ×, ±, ~, ...) | 25 |  |  |
| Figures | 5 | 5 | yes |
|   Fig. 1 caption (characters) | 458 | 500 | yes |
|   Fig. 2 caption (characters) | 400 | 500 | yes |
|   Fig. 3 caption (characters) | 448 | 500 | yes |
|   Fig. 4 caption (characters) | 471 | 500 | yes |
|   Fig. 5 caption (characters) | 474 | 500 | yes |

Em dashes in file: 0

Body = Introduction through Conclusion, `<sup>/<sub>` and markdown emphasis stripped. Worst case = whitespace words + one per numeric range (digit, en dash or hyphen, digit). Independently cross-checked with `wc -w` after splitting each range at the dash: body 746, Impact 36, Synopsis 96. Title 106 characters.

## Checklist A1 to A8: changed text

| Item | Where | Exact change (v5 → v6) |
|---|---|---|
| A1 | Fig. 4 caption | "gives flat widths that over-cover at high SNR and under-cover at low SNR" → "gives flat widths, 2.7–3.3 times the limit at σ = 0.01, that under-cover at σ = 0.08". Also adds "dashed, information limit (BCRB), a lower bound on error rather than a target width". |
| A2 | Results ¶3 | "but epithelium and stroma coverage dropped to 0.64–0.67 at σ = 0.01 despite widths above the limit" → "but at σ = 0.01 point errors stayed about three times the limit, so epithelium and stroma coverage fell to 0.64–0.67" (RMSE/BCRB-SD 2.9–3.2, `checks_v6_output.txt`) |
| A3 | Methods | "Calibrated-decoder retraining<sup>11</sup> set the likelihood SD…" → "Calibrated-decoder retraining<sup>11</sup> of both posteriors (same noise-free target) set the likelihood SD…" |
| A3 | Results ¶3 | "CholeskyILR fraction intervals covered 0.86–0.91 at σ = 0.05" → "CholeskyILR fraction intervals covered 0.86–0.91 (Diagonal: 0.71–0.91) at σ = 0.05"; fraction R² ranges widened to include the matched Diagonal runs ("0.46–0.63", "0.30–0.48") |
| A3, A4 | Impact | "A variational network's intervals were far too narrow until its training likelihood matched the noise." → "A variational network's intervals were far too narrow; a noise-matched likelihood with full covariance helped without guaranteeing coverage." Sentence 1 → "tissue fractions were recoverable but compartmental diffusivity and T2 were not" |
| A5 | Fig. 2 caption | "Fraction R² falls smoothly with σ, and NLLS is lowest at every level." deleted; replaced with the MAE ratios the figure shows ("networks 1.15–1.8×, NLLS 2.9–5.0×") |
| A6 | Results ¶2–3 | Stratum defined: "For voxels within 0.05 of the benign and cancer epithelium values". "and both under-covered low-epithelium voxels by 10–15 points" → "Epithelium coverage was 0.93–0.96 near the benign and cancer values but 0.71 below 0.18, under both likelihoods." (FIGURES.md A6 table, CholeskyILR, σ = 0.05, 3 seeds; n = 539 below 0.18) |
| A7 | Fig. 5 caption | Rewritten from the rendered phantom: "The original loss keeps 62% of the contrast with near-zero half-widths. The matched loss (SD 50) keeps 92% but biases both regions upward (+0.09, +0.07)…"; adds the false −11 ms stromal T2 disc. No "restores". |
| A8 | Counts | Recounted (table above): body 728 normal / 746 worst case ≤ 750; Impact 36 ≤ 40; Synopsis 94 / 96 ≤ 100; captions 400–474 ≤ 500 characters; title 106 ≤ 125. |

Optional polish adopted (within budget):
- Synopsis: "can separate"; "Initially, 95% intervals"; "at SNR 20, but not across SNR"; "and a fixed interval" dropped from Approach.
- Results ¶2: "treated as signal fractions"; the ablation parenthesis is cut.
- Results ¶3: "SD-50 intervals excluded…".
- Discussion: "and every estimator did worse than a constant, which pooled metrics hide"; "Limitation: data simulated with…".
- Methods: "(same noise-free target)".

Not adopted, for words: benign-bias clause, "point errors exceeded it by about 20%", "(unit KL weight)", "and, if not, why", SD–error ρ clause. These are listed in the author notes as poster items.


| A3 (Synopsis) | Synopsis Results | Orchestrator edit after the final pass: "tissue-fraction intervals covered 86–91%" → "full-covariance tissue-fraction intervals covered 86–91%", so the Synopsis matches the scoping in Impact, Methods and Results (Diagonal coverage at σ = 0.05 is 0.71–0.91; checked against metrics_long.csv). |
