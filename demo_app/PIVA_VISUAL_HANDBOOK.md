# PIVA Visual Results Handbook

## Purpose

This handbook explains how to present the PIVA Streamlit demonstration in a short video. The dashboard converts 16 MRI measurements per voxel into estimates of prostate tissue diffusion, T2 relaxation, and tissue composition, while also showing uncertainty and physics-based signal reconstruction.

The current patients and reference cohort are synthetic. The dashboard is an academic demonstration, not a diagnostic system.

## The visual language used throughout the app

- **Dark diamond:** the patient's PIVA point estimate.
- **Blue line:** the PIVA 95% credible interval—the range containing the central 95% of posterior estimates.
- **Green band:** the central 80% of the model-matched synthetic reference cohort, from its 10th to 90th percentile.
- **Red or amber:** a value in an outer part of the synthetic reference distribution or a result requiring careful interpretation.
- **Percentile:** the percentage of synthetic-reference values below the patient's estimate. It is not the probability of cancer.

An estimate outside the green band means it is unusual relative to this synthetic cohort. It does not by itself mean disease. A wide blue interval means the model is less certain about the estimate.

---

## 1. Selected patient card

### What it shows

The prepared case, age, PSA, PSA density, and a short description. Each case contains ten synthetic voxels, with 16 MRI measurements per voxel.

### What to say

> “We begin with a prepared synthetic patient. PIVA receives ten voxels, and each voxel contains signals measured at four diffusion weightings and four echo times, giving 16 measurements.”

### Important caution

Age and PSA provide presentation context only. They are not inputs to the current PIVA neural network or concern score.

---

## 2. Quantitative overview cards

### Concern score

The score combines three posterior events:

- epithelial fraction above the synthetic-reference upper boundary;
- lumen fraction below the synthetic-reference lower boundary;
- epithelial diffusion below the synthetic-reference lower boundary.

The weights are 40%, 30%, and 30%, respectively.

### What to say

> “The concern score summarizes three model-derived tissue deviations on a 0-to-100 scale. It is a transparent demonstration score, not a cancer probability.”

### Quantitative confidence

This summarizes the widths of all nine PIVA credible intervals relative to the corresponding synthetic-reference spans.

> “Confidence tells us how precisely the model identified the tissue factors. It describes uncertainty in the estimates, not diagnostic confidence.”

### Signal reconstruction

This reports how closely the physical tissue estimates reconstruct the observed mean MRI signal.

> “A high reconstruction score means the inferred tissue parameters reproduce the measured signal well under the three-compartment physics equation.”

### Voxels analyzed

This confirms the region size and number of measurements used.

---

## 3. Quantitative-position gauge

### How to read it

- **0–34:** within-range category
- **35–64:** indeterminate category
- **65–100:** elevated category

The colored zones are demonstration categories chosen for communicating the composite score.

### What to say

> “The gauge provides a quick summary before we inspect the individual measurements. We should never interpret it alone—the remaining visuals show what drove the score and how uncertain those drivers are.”

### Do not say

Do not call the gauge a cancer risk, malignancy probability, PI-RADS score, or clinical threshold.

---

## 4. What stands out most

### What it shows

The three parameters whose percentiles are farthest from the synthetic-reference median. A result can stand out because it is unusually high or unusually low.

### What to say

> “These are the three largest patient-to-reference deviations. They help us move from one overall score to the specific tissue properties responsible for the result.”

A 97th-percentile epithelial fraction, for example, means the estimate is higher than approximately 97% of the synthetic-reference estimates. It does not mean a 97% probability of cancer.

---

## 5. Estimated tissue-composition visual

### What it measures

PIVA estimates the relative fractions of:

- epithelium;
- stroma;
- lumen.

The three point estimates sum to 100%.

### How to read it

For each row:

1. Locate the green synthetic-reference band.
2. Locate the dark patient diamond.
3. Inspect the blue 95% credible interval.
4. Ask whether the point estimate and interval overlap the reference band.

### Example interpretation

If the epithelial diamond lies beyond the right edge of the green band, the estimated epithelial fraction is high relative to the synthetic cohort. If its entire blue interval is also outside the band, that deviation is more consistently supported by the posterior samples. If the blue interval crosses the band, the direction is less certain.

### What to say

> “This is the most physically intuitive PIVA result. The diamond is the estimated tissue fraction, the blue line is model uncertainty, and the green area is the expected region in our synthetic reference population.”

---

## 6. Quantitative tissue-factor plots

PIVA estimates three parameter families for epithelium, stroma, and lumen.

### Diffusion

Diffusion describes how freely water moves in each compartment. Lower diffusion indicates more restricted water motion. The expected ordering is restricted epithelial diffusion, intermediate stromal diffusion, and relatively free luminal diffusion.

> “The diffusion panel asks how freely water moves within each inferred tissue compartment.”

### T2 relaxation

T2 describes how quickly transverse MRI signal decays. The lumen normally has a much longer modelled T2 than epithelial or stromal tissue.

> “The T2 panel separates compartments by their relaxation behavior. Luminal fluid retains transverse signal much longer than the denser cellular compartments in this model.”

### How to read both panels

The diamond, blue credible interval, and green reference band use the same meaning as in the composition chart. Each family has its own physical scale so unlike units are not visually mixed.

### Important caution

The visible reference intervals are derived from the capstone notebook's physiological priors and the trained model's outputs. They are not universal clinical normal ranges.

---

## 7. Exact quantitative-values table

### What it shows

For all nine tissue factors, the table gives:

- point estimate;
- lower and upper limits of the PIVA 95% credible interval;
- synthetic-reference limits;
- reference percentile;
- below, within, or above-reference label;
- physical unit.

### What to say

> “The charts support rapid interpretation, while this table preserves the exact numerical estimates and intervals for technical review.”

Use the table to quote exact values only after explaining the visual comparison.

---

## 8. Reference-percentile chart

### How to read it

- 50th percentile is the synthetic-reference median.
- The green 10th–90th percentile area contains the central 80% of reference estimates.
- The outer shaded regions contain unusually low or high values relative to this reference.
- Every dot locates one patient parameter on the same percentile scale, allowing parameters with different units to be compared.

### What to say

> “Percentiles solve the different-units problem. Diffusion, T2, and fractions cannot share one physical axis, but each can be placed on a common reference-percentile axis.”

### Do not say

Do not describe the central 80% as a medical normal range or values outside it as automatically abnormal.

---

## 9. Exact reference-comparison table

This table shows the patient estimate beside the synthetic-reference 10th percentile, median, and 90th percentile.

> “This is the numeric version of the percentile plot and makes the synthetic comparison completely transparent.”

The reference-provenance expander explains that the displayed ranges come from 5,000 synthetic voxels processed by the same trained model. It also gives published values for scientific context, but those published means and standard deviations are not used as clinical cutoffs.

---

## 10. Voxel-variation heatmap

### What it shows

Rows are the nine tissue parameters, columns are the ten analyzed voxels, and each cell is that voxel's percentile in the synthetic-reference distribution.

### How to read it

- Values near 50 indicate reference-central estimates.
- Values near 0 are comparatively low.
- Values near 100 are comparatively high.
- A similar color across a row suggests a spatially consistent regional pattern.
- A mixture of colors suggests within-region heterogeneity.

### What to say

> “The patient-level summary is an average. This heatmap checks whether that average represents all ten voxels or hides a heterogeneous region.”

---

## 11. Probability dashboard

### What it shows

Posterior evidence for the three events used in the concern score:

- high epithelial fraction;
- low lumen fraction;
- restricted epithelial diffusion.

### How to read it

Each bar is the proportion of posterior samples crossing the corresponding synthetic-reference boundary, with conservative regularization to avoid displaying absolute 0% or 100% from a finite Monte Carlo sample.

### What to say

> “This panel exposes the evidence behind the overall score. Instead of only reporting a point estimate, PIVA asks how often plausible posterior solutions cross each reference boundary.”

### Important caution

These are posterior probabilities of parameter-threshold events under the model. They are not probabilities of disease.

---

## 12. Estimate-precision chart

### What it shows

For each parameter, the PIVA 95% credible-interval width is divided by the synthetic-reference 10th–90th percentile width.

### How to read it

- A small percentage indicates a relatively precise estimate.
- 100% means the PIVA interval is as wide as the visible reference span.
- Above 100% means the estimate is weakly identified relative to that reference span.

### What to say

> “This visual prevents false precision. A parameter may look unusual, but if its credible interval is very wide, we should interpret that deviation cautiously.”

---

## 13. Observed-signal versus PIVA-reconstruction plot

### What it shows

The solid series is the patient's observed mean MRI signal over the 16 b-value and echo-time combinations. The dashed series is the signal reconstructed from PIVA's estimated diffusion, T2, and tissue fractions.

### How to read it

Good alignment means the inferred physical parameters explain the measured signal under the fixed MRI equation. Large or systematic gaps would indicate model mismatch or poor reconstruction.

### What to say

> “This closes the physics-informed circuit. PIVA does not stop at predicting hidden factors; it inserts them into the known MRI signal equation and checks whether they reproduce the measurements.”

### Important caution

A good fit is necessary but not sufficient for biological correctness. Different parameter combinations can sometimes reconstruct similar signals.

---

## 14. How-it-was-calculated tab

This tab states the concern-score equation and explains the Full-ILR Cholesky posterior.

### Short explanation for the video

> “The encoder predicts a multivariate latent distribution rather than one deterministic answer. The Cholesky representation allows correlations between uncertain tissue factors. Posterior samples are transformed into physically constrained diffusion, T2, and composition values, and the decoder reconstructs the MRI signal.”

---

## Suggested 5–7 minute video sequence

1. **Problem and input — 30 seconds**

   Explain the ten voxels and 16 measurements per voxel.

2. **PIVA idea — 45 seconds**

   Explain posterior sampling, physical tissue factors, and signal reconstruction.

3. **Overview and gauge — 45 seconds**

   State clearly that the concern score is not a cancer probability.

4. **Composition and tissue factors — 90 seconds**

   Introduce the diamond, blue interval, and green synthetic-reference band.

5. **Percentiles and voxel heterogeneity — 60 seconds**

   Explain relative position and regional consistency.

6. **Posterior evidence and precision — 60 seconds**

   Show how uncertainty qualifies the result.

7. **Physics reconstruction — 45 seconds**

   Close the loop from measurements to tissue factors and back to reconstructed measurements.

8. **Limitations — 30 seconds**

   State that patients, thresholds, and displayed reference ranges are synthetic and require clinical-cohort validation.

## Recommended closing statement

> “The contribution of PIVA is not simply one score. It is an interpretable chain from multidimensional MRI measurements to compartment-specific tissue estimates, correlated uncertainty, comparison with a clearly identified synthetic reference cohort, and verification through the MRI physics model. This demonstration establishes the reporting workflow; clinical validation and population-specific reference intervals are the next steps.”

## Claims to avoid throughout the video

- “PIVA diagnoses prostate cancer.”
- “The concern score is the probability of cancer.”
- “The green band is a validated clinical normal range.”
- “A high reconstruction score proves the tissue estimates are biologically correct.”
- “The 95% credible interval means there is a 95% frequentist probability that one fixed true value lies inside it.”
- “The synthetic prepared cases are real patients.”
