# ISMRM 2027 submission checklist (PIVA abstract)

Deadline: **28 Oct 2026** (ECHO). This list merges the "Submission readiness" sections of the four round-3 reviews (`reviews/r3_*.md`). Reviewers who raised each item are in brackets: P = physics, M = ML, C = clinical, K = committee.

Round 3 scores: Physics Q75/I58 (power pitch), ML Q77/I60, Clinical Q75/I52, Committee Q78/I62 (digital poster; power pitch once the figures exist). Mean Q76.3 / I58.0, up from 47.8 / 39.0 in round 1.

## A. Blocking text fixes (agents: final writer pass, v6)

- [x] A1. Fig. 4 caption: delete "over-cover at high SNR". False for v_ep and v_st (0.942, 0.935 at σ 0.01). [P M K]
- [x] A2. Results ¶3: replace "despite widths above the limit". The BCRB is a lower bound on error, not a target width; at σ 0.01 point errors are about 3x the bound. [P]
- [x] A3. Scope the noise-matched result to the full-covariance (CholeskyILR) posterior in Methods, Results and Impact. The Diagonal matched runs give 0.71–0.91 at σ 0.05 and collapse to 0.08 at σ 0.01. [M C]
- [x] A4. Impact: drop "until its training likelihood matched the noise", which implies a fix that is only partial. [K]
- [x] A5. Fig. 2 caption: delete or correct "R² falls smoothly with σ… NLLS lowest at every level". NLLS v_ep goes 0.16 → 0.34. [M K]
- [x] A6. Low-epithelium clause: define the stratum; give it as a gap relative to high-epithelium voxels, and say where the failure is (v_ep < 0.18, about 0.71; about 0.95 near benign and cancer values). Recompute from `analysis/` for the model the text describes. [M C K]
- [x] A7. Fig. 5 caption: write it from the rendered phantom; "partly restores contrast", not "restores". [M C K]
- [x] A8. Recount after edits: body worst case (ranges = 2 words) ≤ 750, Impact ≤ 40, Synopsis ≤ 100. [all]

Section A done 2026-10-10 in `drafts/abstract_v6.md`. The orchestrator verified: no flagged phrase remains in the submission text; all 6 figure files exist; counts (worst case) title 106/125, Impact 36/40, Synopsis 97/100, body 746/750, captions ≤ 474/500; pooled baseline R² (0.993/0.975) and Diagonal SD-50 coverage (0.71–0.91) re-checked against `metrics_long.csv`; Synopsis scoped to full covariance (orchestrator edit).

## B. Blocking figure work (agents: re-analyst)

- [x] B1. Regenerate Fig. 2: no "large2000" suptitle. [C K]
- [x] B2. Regenerate Fig. 3: box-clipped widths, y-axis "% of prior box", patient-bootstrap CIs on both bar types, no suptitle. [P M C K]
- [x] B3. Regenerate Fig. 4: no suptitle, y-axis "% of prior box", relabel "v_lu" as "v_lum". [P M K]
- [x] B4. Draw Fig. 1 (schematic) to the spec in `drafts/abstract_v5.md`. [P M C K]
- [x] B5. Run and draw the Fig. 5 phantom. [P M C K]
- [x] B6. Export the preview figure: 3:2, ≤ 1 MB, no small text. **Mandatory in ECHO.** [P M C K]

Section B done 2026-10-10: `analysis/figures/final/` (see `FIGURES.md`); the orchestrator viewed the preview and Fig. 1 and fixed two Fig. 1 label overflows.

## C. Authors only (cannot be delegated)

- [ ] C1. Resolve every `[CONFIRM]`: affiliations (exact unit names), funding and grant numbers, the formal Grossman Center name.
- [x] C2. Authorship decided 2026-10-11: the authors are Garay, Chowdhury and Gundogdu only. A. Chatterjee, G. Karczmar and A. Oto are not authors. Still open: whether to name them in the Acknowledgements (folded into C1). [P C K]
- [ ] C3. Paste the body into ECHO and confirm its own word count, including the display equation (worst case under our count: 746/750; `drafts/abstract_v6.md` 'Notes for authors' lists clauses to cut if ECHO counts higher). [C K]
- [ ] C4. Gundogdu approves the abstract. He knows about the drafting process and will review the abstract before submission; tick this box once he signs off.
- [ ] C5. Pick the categories (primary/secondary in the draft) and upload the figures.

## Verified by the orchestrator

- Ref. 6 (PS-VAE) DOI 10.1109/TMI.2026.3740952 resolves via doi.org to IEEE Xplore. Crossref confirms the title, *IEEE Transactions on Medical Imaging* 2026, and Finkelstein, Moneta, Zohar, Rivlin et al. (The committee's web search found only the arXiv preprint.) Mark "Early access" if IEEE lists it that way.

## Optional polish (from reviewers, if words allow)

See the "Optional polish" lists at the end of each `reviews/r3_*.md`. The most useful:
- Disclose that the matched retrain still uses the noise-free target. [P]
- Note the benign-side bias under the matched loss (+0.035 → +0.059), or put it on the poster. [C]
- "point error, not width, limited coverage". [P M]
