# ISMRM 2027 audit of PIVA

This folder holds the work behind the ISMRM 2027 abstract *"What prostate HM-MRI data can and cannot determine: a simulation audit of physics-informed VAE uncertainty"* (Garay, Chowdhury, Gundogdu). It re-runs the models from `notebooks/` under a stricter protocol, adds experiments that test why PIVA's uncertainty fails, and contains the current abstract draft (v6, not yet submitted; deadline 28 Oct 2026).

**The findings change what the rest of this repo claims.** Read the summary below before citing numbers from the root `README.md` or `results/RESULTS.md`.

## Summary of findings

All results are in simulation, with the 4 b-value × 4 echo-time protocol of the PIA paper, at SNR 20 (σ = 0.05) unless stated. The test set is 2000 voxels from 200 unseen patients (`large2000`), with 3 training seeds per model.

1. **Compartmental D and T2 are not recoverable from these data.** When scored per compartment, every method (NLLS, PIA, PIVA-Diagonal, PIVA-CholeskyILR) has negative R² on all six D and T2 parameters, and does worse than a constant that predicts the prior centre for every voxel. The repo's pooled R² of 0.98–0.99 comes from the 10× differences between compartment means. The constant scores 0.993 (D) and 0.975 (T2) on that metric.
2. **The cause is the protocol, not the network.** A Bayesian Cramér-Rao bound (the *information limit*) shows that the 16 signals narrow the population prior SD of D and T2 by only 1–13%, against 34–70% for the volume fractions. Without the prior, the 95% bound for D and T2 is 1.3–12 times wider than the whole prior box.
3. **Volume fractions are recoverable, and PIA is best or tied.** R² is about 0.9 for lumen, 0.46–0.63 for epithelium and 0.30–0.48 for stroma. NLLS reaches 0.75, 0.28 and −0.25. PIA beating NLLS is consistent with the PIA paper.
4. **The original PIVA intervals are severely overconfident.** Their 95% intervals contain the truth in 5–62% of voxels. The loss (MSE + 0.1·KL on ×1000 signals) implies a likelihood SD of about 0.9 signal units, while the simulated noise SD is 10–80. Post-hoc σ-scaling reaches 0.93–0.96 coverage, but the scaled D and T2 intervals are then wider than a fixed interval that ignores the data.
5. **A noise-matched likelihood partly fixes the fractions.** With full covariance (CholeskyILR) and a likelihood SD of 50, fraction coverage is 0.86–0.91 at σ = 0.05, at widths near the information limit, with no post-hoc step. It does not hold across noise levels: coverage falls to 0.81–0.84 at σ = 0.08. With a per-voxel noise SD, widths follow the limit for σ ≥ 0.03, but at σ = 0.01 point errors are about 3× the bound and coverage drops to 0.64–0.67. D and T2 intervals widen to roughly the whole prior box.
6. **The full covariance does the work, not the ILR map.** In the ablation, Cholesky + softmax ≈ Cholesky + ILR, and Diagonal + ILR ≈ Diagonal + softmax.
7. **Shrinkage toward the prior costs clinical contrast.** For voxels near the benign (0.23) and cancer (0.49) epithelium values, the original CholeskyILR estimates 0.26 and 0.40, halving the contrast. Epithelium coverage is 0.93–0.96 near those values but 0.71 for epithelium below 0.18.

Bottom line: on this protocol at SNR 20, HM-MRI fitters should report tissue fractions, not compartmental D or T2. Variational fraction intervals need a noise-matched likelihood and coverage checks across SNR and composition.

## Experiments and what each one tests

Run tags below are checkpoint and prediction names: `gen-<cov>-<vmap>-<lik><SD>-<target>_s<seed>`. `gaussPV` means per-voxel SD. Queue files list the exact `train_one.py` arguments.

### Core audit: the original recipes, re-scored

| # | Question it answers | Runs | Main output |
|---|---|---|---|
| 1 | Do the models learn each parameter, or just which compartment it belongs to? Per-compartment R², MAE and bias against two constant baselines that never look at the signal. | `pia`, `diag`, `chol` × seeds 0–2 (`queue.txt`, `queue2.txt`); NLLS (`run_nlls.py`) | `RESULTS_rerun.md` Tasks 1–3; `results/tables.md`; Fig. 2 |
| 2 | Does the picture change with SNR? | Same checkpoints at σ = 0.01, 0.03, 0.05, 0.08 | `RESULTS_rerun.md` Task 6 |
| 3 | Are PIVA's 95% intervals calibrated? Raw coverage and width, then per-parameter σ-scaling. | `diag`, `chol` × 3 seeds | Task 4; Fig. 3 |
| 4 | How well does the NLLS baseline actually fit? | `run_nlls.py` | Task 5. Almost every fit (1999/2000) ends with a parameter on a box bound, and up to 15% violate the simplex. |
| 5 | Do the repo's reported numbers reproduce? | All of the above | "Differences from repo-reported numbers". The live notebook outputs reproduce. The README/RESULTS.md "Final comparison" table does not. |

### Follow-up experiments: changes to the recipe

| ID | Question it answers | Runs | Answer |
|---|---|---|---|
| A | Is the overconfidence caused by the likelihood scale? Retrains with a proper Gaussian NLL at fixed SD s ∈ {15.49, 28.28, 50}. | `gen-*-gauss{15.49,28.28,50}-clean_s0` | Yes. Coverage rises with s. At s = 50, fraction widths match the information limit and D/T2 intervals collapse to the prior box. |
| A2 | Does reconstructing the noisy input, or using a Rician likelihood, change that? | `gen-*-gauss50-noisy_s0`, `gen-*-rician50-noisy_s0` | Barely: within a few points of the clean-target run. |
| B | Which part of CholeskyILR helps: the full covariance or the ILR map? 2×2 ablation on the original loss. | `gen-diag-ilr-*`, `gen-chol-softmax-*` | The covariance does. ILR has no measurable effect. |
| C | Was the v2.0 notebook's calibration fair? τ is fit on a separate calibration cohort, with patient-level bootstrap CIs. | Evaluation only (`calib240`, `calib2000` cohorts) | Both models reach 0.93–0.96. The earlier "Cholesky calibrates better on 7/9" result was an artefact of a within-test split and a capped τ grid. |
| D | Is PIVA's uncertainty better than a data-independent interval (the central 95% of the training distribution)? | Evaluation only | Only for the fractions. For D and T2, every calibrated PIVA interval is wider than the fixed one. |
| E | What is the best any estimator could do? Expected Rician Fisher information plus the training-set prior covariance (Bayesian CRB). | `fisher_crb.py` | The data barely inform D and T2 (see finding 2). `results/crb_widths.csv` |
| PV | If the likelihood uses each training voxel's own noise SD, do interval widths follow SNR? The network never receives σ. | `gen-{diag-softmax,chol-ilr}-gaussPV-clean_s{0,1,2}` (`queue3.txt`) | Widths track the limit for σ ≥ 0.03, but coverage drops to 0.64–0.67 at σ = 0.01. |
| SD50-3seed | Is the s = 50 headline robust to the training seed? | `gen-*-gauss50-clean_s{1,2}` (`queue3.txt`) | Yes. Seed SD ≤ 0.013 on fraction coverage. |
| Strata | Is calibration uniform across tissue composition? Coverage stratified by true epithelium fraction. | `round2.py`, `final_figures.py` (A6 numbers) | No. It is 0.93–0.96 near the benign and cancer values and 0.71 below 0.18. |
| Phantom | Does the posterior keep the contrast of a cancer-like lesion, and does it invent structure? Synthetic disc (epithelium 0.48) in benign background (0.23). | `final_figures.py` (Fig. 5, seed 0) | The original loss keeps 62% of the contrast and paints a false −11 ms stromal-T2 disc. The matched loss keeps 92% but biases both regions upward. |

## Layout

```
ismrm2027/
  README.md                this file
  SUBMISSION_CHECKLIST.md  what remains before submission (author-only items)
  drafts/
    abstract_v6.md         current abstract (submission text, figure captions, references, counts)
    abstract_v6.pdf        rendered abstract with figures (internal, unblinded)
    provenance_v6.md       source file for every number in the abstract
    checks_v4/5/6.py       scripts that recompute cited numbers from analysis/ (+ their outputs)
    count_v6.py            ISMRM word and character counter (worst-case rule)
  render/build_pdf.py      rebuilds abstract_v6.pdf (needs pandoc and xelatex)
  analysis/
    RESULTS_rerun.md       full write-up of every experiment, with tables
    piva_core.py           repo notebook code (generator, NLLS, PIA, PIVA) + seeding/evaluation helpers
    extras_models.py       GenericPIVA (2×2 ablation), Gaussian/Rician NLL, per-voxel noise SD
    train_one.py           trains one model/seed; queue*.txt + run_queue*.sh run them in parallel
    run_nlls.py, evaluate.py, fisher_crb.py, report.py, round2.py, final_figures.py
    results/, results_r2/  every metric (long CSVs), summary tables, CRB widths
    preds/                 posterior means/SDs and NLLS fits per model, cohort and σ
    runs/                  training history (*.json) and log per checkpoint
    figures/final/         the five abstract figures, preview image, FIGURES.md, final_numbers.json
    figures/               earlier audit figures (fig1, 2, 4, 5) and supplementary
```

`abstract_v6.md`, `figures/final/FIGURES.md` and `SUBMISSION_CHECKLIST.md` are newer than `RESULTS_rerun.md`. Where the wording differs, follow them. For example, the round-3 review removed the claim that fixed-SD widths "over-cover at high SNR", which `RESULTS_rerun.md` still makes.

## Reproducing

Original environment: Python with torch 2.5.1 (CPU, OpenMP build), numpy 2.3.1, scipy 1.16.0, scikit-learn 1.7.1, pandas, matplotlib. Training was CPU only (Intel i9-9880H). Each 200-epoch run took about 2–3 h on one thread, with 8–10 runs in parallel.

```bash
cd ismrm2027/analysis
export OMP_NUM_THREADS=1 PYTHON=python          # PYTHON = interpreter used by run_queue*.sh
./run_queue.sh; ./run_queue2.sh; ./run_queue3.sh # train everything (≈10 h wall time); or one run:
python train_one.py --model generic --cov chol --vmap ilr --lik_sd 50 --target clean --seed 0
NLLS_PROCS=2 python run_nlls.py orig240 large2000
python fisher_crb.py                             # experiment E, ~30 s
EVAL_PROCS=8 python evaluate.py                  # writes preds/ and results/*_long.csv
python report.py                                 # results/tables.md + audit figures
python round2.py                                 # per-voxel vs SD-50 tables and figure
python final_figures.py                          # abstract figures + final_numbers.json
python ../drafts/checks_v6.py                    # re-derive the numbers cited in v6
```

The `queue3.txt` runs were evaluated with `EVAL_RUNS=runs_r2 EVAL_OUT=results_r2` and merged into `results/`. `runs_r2/` was a folder of symlinks to those checkpoints.

## What is not in this branch

- **Model checkpoints (`runs/*.pt`, 31 files, about 250 MB).** They are excluded by the repo's `.gitignore`. Without them you can still rerun `report.py`, `round2.py`, `fisher_crb.py` and the `checks_v*.py` scripts from the included `preds/` and `results/`. `evaluate.py` and the phantom in `final_figures.py` need the checkpoints: retrain them, or ask Aaron for a copy.
- Run logs, watchdog and smoke-test folders, and the flush-to-zero A/B test. These are operational only and are summarised in `RESULTS_rerun.md` ("Run notes").
- Internal material from the drafting process: earlier drafts, simulated reviews, agent prompts and the research brief.

## How this work was produced

The analysis and drafting were done with an AI-assisted research loop (Claude Code). Each round had research and re-analysis agents, a writer, and four simulated reviewers (MR physics, ML/uncertainty, clinical, program committee). There were three rounds. Every number in the abstract is traceable to a file in this folder through `drafts/provenance_v6.md`, and the `checks_v*.py` scripts recompute them. The reviewer scores were internal estimates, not real ISMRM reviews. The authors are responsible for the final content.
