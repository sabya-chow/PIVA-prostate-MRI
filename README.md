# Physics-Informed Deep Learning for Noise-Robust IVIM Parameter Estimation in Breast DW-MRI

## Members
- **Ken Lew** (Research Lead — Breast DW-MRI IVIM Integration)
- **Sabya Chow** (Sabyasachi Chowdhury — PIVA Prostate MRI Group Lead / Collaborator)

---

## Abstract

Breast cancer remains the most prevalent malignancy among women worldwide, with early and accurate non-invasive characterization being critical for clinical management. Diffusion-Weighted Magnetic Resonance Imaging (DW-MRI) paired with the Intravoxel Incoherent Motion (IVIM) biexponential model allows quantitative decoupling of microvascular perfusion ($f$), true tissue diffusion ($D_t$), and capillary pseudo-diffusion ($D^*$) without contrast dye injections (e.g., gadolinium). However, conventional voxel-wise non-linear least squares (NLLS) optimization suffers from extreme noise sensitivity, lack of spatial continuity, and prohibitive CPU computational latency (~187 seconds per 2D slice).

To overcome these challenges, we introduce a **Two-Stage Physics-Informed Artificial Intelligence Pipeline** tailored for breast DW-MRI:
1. **Stage 1 (Physics-Informed Autoencoder - PIA)**: Employs a self-supervised neural encoder paired with an exact bi-exponential IVIM physics decoder ($S(b) = (1-f)e^{-b D_t} + f e^{-b D^*}$) and bounded activation functions ($\text{mean} \pm \Delta \cdot \tanh(\cdot)$), guaranteeing parameter estimates stay within biophysical bounds without requiring ground-truth labels during initial training. We benchmark both multi-layer perceptron (MLP-PIA) and convolutional (CNN-PIA) encoders.
2. **Stage 2 (Spatial U-Net Refiner)**: A multi-scale residual U-Net (3.49M parameters) that operates across all Stage 1 output parameter maps simultaneously, exploiting spatial coherence to eliminate salt-and-pepper noise artifacts and applying tumor-weighted supervised loss to preserve sharp malignant lesion boundaries.

Evaluated on 400 held-out test patient datasets derived from the VICTRE (Virtual Imaging Clinical Trials for Regulatory Evaluation) digital breast phantom platform across 12 noise levels ($\text{SNR} = 4$ to $\text{SNR} = 100$):
- **Computational Speed**: The proposed pipeline achieves an inference latency of **29 milliseconds per slice on GPU** compared to **187.3 seconds for NLLS fitting**—a **6,431× speedup**, enabling real-time quantitative mapping directly on MRI console workstations.
- **Noise Stability**: Under severe noise conditions ($\text{SNR} = 10$), our CNN-PIA + Refiner maintains a total relative root mean square error ($\text{rRMSE}$) of **0.28**, whereas NLLS fitting diverges catastrophically with an $\text{rRMSE}$ exceeding **9.4** (a **33.5× improvement** in estimation stability).

---

## Repository Structure

```
├── README.md                Main repository documentation (Title, Members, Abstract, Install & Run)
├── DESIGN.md                LLM-oriented architectural summary (8 core design points)
├── requirements.txt         Python package dependencies
│
├── presentation/            Capstone presentations & flowchart diagrams
│   ├── PIA_IVIM_Presentation.pptx   13-slide Capstone presentation deck
│   └── flowcharts/                  System & network architecture flowcharts
│       ├── pipeline_overview.jpg
│       ├── mlp_pia_architecture.jpg
│       ├── cnn_pia_architecture.jpg
│       └── unet_refiner_architecture.jpg
│
├── src/                     Source code modules
│   ├── PIA.py               Self-supervised Physics-Informed Autoencoder (Stage 1 MLP & CNN)
│   ├── model.py             Supervised baseline encoder models
│   ├── utils.py             Biexponential math, data loading, rRMSE metrics, NLLS fitting
│   ├── NLLS_solution.py     Voxel-wise NLLS baseline evaluation script
│   └── method1.py           Supervised baseline training script
│
├── data/                    VICTRE digital breast phantom dataset (400 cases, 0001–0400)
│   └── XXXX_*.npy           Per-patient DWIs, k-space noise, tissue masks, & ground truth
│
├── checkpoints/             Pre-trained model weights (.pt files)
│   ├── pia_baseline.pt      MLP-PIA (Stage 1)
│   ├── pia_cnn_best.pt      CNN-PIA (Stage 1)
│   ├── refiner_cnn_best.pt  CNN-PIA + Refiner (Stage 2)
│   ├── refiner_cnn_e2e_best.pt  CNN-PIA + Refiner (End-to-End fine-tuned)
│   └── refiner_mlp_e2e_best.pt  MLP-PIA + Refiner (End-to-End fine-tuned)
│
├── figures/                 Publication scripts & generated figures
│   ├── benchmark_inference.py   CPU vs. GPU inference speed benchmarking
│   ├── plot_noise_clean.py      rRMSE vs. noise level line plots
│   ├── plot_tables.py           Paper summary tables (timing & error metrics)
│   ├── plot_paper_figures.py    Qualitative maps, ablation & input data figures
│   └── *.png / *.jpg            Generated figures & paper diagrams
│
└── results/                 Quantitative evaluation output files (.json)
    ├── noise_evaluation_results_e2e.json   rRMSE across noise levels
    └── inference_timing.json               CPU & GPU timing statistics
```

---

## Bi-exponential IVIM Signal Physics

The observed DW-MRI signal decay as a function of diffusion gradient factor $b$ is given by:

$$\frac{S(b)}{S(0)} = (1 - f) \cdot \exp(-b \cdot D_t) + f \cdot \exp(-b \cdot D^*)$$

| Parameter | Symbol | Biophysical Meaning | Typical Range | Units |
|-----------|--------|---------------------|---------------|-------|
| **Perfusion Fraction** | $f$ | Micro-vascular blood volume fraction | $0.05 - 0.35$ | dimensionless |
| **Tissue Diffusivity** | $D_t$ | True extravascular water diffusion | $0.0007 - 0.0015$ | $\text{mm}^2/\text{s}$ |
| **Pseudo-Diffusivity** | $D^*$ | Capillary blood flow microcirculation | $0.005 - 0.060$ | $\text{mm}^2/\text{s}$ |

Acquisition $b$-values: $[0, 5, 50, 100, 200, 500, 800, 1000]\text{ s/mm}^2$.

---

## How to Install and Run

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/sabya-chow/PIVA-prostate-MRI.git
cd PIVA-prostate-MRI

# Switch to the breast cancer branch
git checkout pia_breast

# Install required packages
pip install -r requirements.txt
```

### 2. Running Baseline NLLS Fitting

To run the voxel-wise Non-Linear Least Squares (NLLS) optimization baseline across test patient data:

```bash
cd src
python NLLS_solution.py
```

### 3. Benchmarking Inference Speeds (CPU & GPU)

To reproduce CPU vs. GPU latency benchmarks across NLLS and deep learning models:

```bash
cd figures
python benchmark_inference.py
```

Output results will be updated in `results/inference_timing.json` and saved as `figures/inference_time_table.png`.

### 4. Generating Quantitative Figures & Tables

To generate publication-ready performance plots, qualitative parameter maps, and ablation figures:

```bash
cd figures

# Generate rRMSE noise-level curves
python plot_noise_clean.py

# Generate qualitative comparison maps and ablation figures
python plot_paper_figures.py

# Generate paper summary tables
python plot_tables.py
```

---

## Citation & References

- Based on the PIA-IVIM framework by Batuhan Gundogdu, University of Chicago Radiology.
- Upstream reference: [batuhan-gundogdu/PIA_IVIM](https://github.com/batuhan-gundogdu/PIA_IVIM)
