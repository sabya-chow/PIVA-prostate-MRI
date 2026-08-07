# PIVA Quantitative Patient Report Demo

This local Streamlit application turns ten synthetic patient voxels from the capstone acquisition model into a patient-friendly quantitative report.

The augmented dashboard includes a 5,000-voxel model-matched synthetic reference cohort, patient percentiles, central reference bands, key score drivers, regional voxel variation, posterior precision charts, signal reconstruction, and a visual downloadable report.

## Run

```bash
cd "/Users/sabyasachi/macbook storages/ds_project/MSADS_Uchicago/capstone/capstone_work/implementation of PIVA"
source .venv/bin/activate
streamlit run demo_app/streamlit_app.py
```

The repository already contains the exported demonstration checkpoint. To regenerate it:

```bash
.venv/bin/python demo_app/train_demo_model.py
```

## Presentation path

1. Select **Elevated quantitative pattern**.
2. Click **Run PIVA analysis**.
3. Explain the four overview cards.
4. Point to the three largest percentile deviations and tissue composition.
5. Open **Where you stand** to show reference bands and voxel variation.
6. Open **Inside the MRI signal** to show posterior probabilities and signal reconstruction.
7. Download the detailed patient report.

## Scope

All patients and reference thresholds are synthetic. The concern score is an academic demonstration score, not a cancer probability, PI-RADS score, diagnosis, or clinical recommendation.
