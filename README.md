# PIA-IVIM: Physics-Informed AI for Intravoxel Incoherent Motion MRI

Two-stage deep learning pipeline for quantitative IVIM parameter estimation from diffusion-weighted breast MRI, benchmarked against non-linear least squares (NLLS) fitting.

Based on the AAPM IVIM Challenge using VICTRE digital breast phantoms.

## Repository Structure

```
├── src/                     Source code
│   ├── PIA.py               Self-supervised PIA model (Stage 1 MLP autoencoder)
│   ├── model.py             Supervised PIAEncoder baseline (Model-1)
│   ├── utils.py             Metrics (rRMSE), data loading, NLLS fitting, batch generation
│   ├── NLLS_solution.py     NLLS baseline evaluation script
│   └── method1.py           Supervised baseline training script
│
├── data/                    VICTRE phantom dataset (not tracked in git)
│   └── XXXX_*.npy           Per-patient files (see Data Format below)
│
├── checkpoints/             Trained model weights (not tracked in git)
│   ├── pia_baseline.pt      MLP-PIA (Stage 1)
│   ├── pia_cnn_best.pt      CNN-PIA (Stage 1)
│   ├── refiner_cnn_best.pt  CNN-PIA + U-Net Refiner (Stage 2)
│   ├── refiner_cnn_e2e_best.pt   CNN-PIA + Refiner, end-to-end fine-tuned
│   └── refiner_mlp_e2e_best.pt   MLP-PIA + Refiner, end-to-end fine-tuned
│
├── figures/                 Publication figures & scripts (Black & White)
│   ├── benchmark_inference.py   Inference timing benchmark (CPU & GPU)
│   ├── plot_noise_clean.py      rRMSE vs noise level line plots
│   ├── plot_tables.py           Publication tables (inference time & rRMSE)
│   ├── plot_paper_figures.py    Qualitative maps, ablation & input figures
│   └── *.png / *.jpg            Generated figures and architecture diagrams
│
├── results/                 Evaluation data
│   ├── noise_evaluation_results_e2e.json   rRMSE across 12 noise levels
│   └── inference_timing.json               CPU/GPU timing measurements
│
└── paper/                   Manuscript & presentation source
    ├── PIA_IVIM_1_Introduction.tex / .docx
    ├── PIA_IVIM_2_Methodology.tex / .docx
    └── PIA_IVIM_Presentation.pptx          13-slide PowerPoint presentation
```

## Data Format

Each patient case (400 cases, indexed 0001–0400) has 6 files in `data/`:

| File | Shape | Description |
|------|-------|-------------|
| `XXXX_IVIMParam.npy` | `200×200×3` | Ground truth parameters (f, Dt, D*) |
| `XXXX_gtDWIs.npy` | `200×200×8` | Clean ground truth DWI signals |
| `XXXX_NoisyDWIk.npy` | `200×200×8` | Complex k-space data with noise |
| `XXXX_NoisyEstimate.npy` | `200×200×3` | MLP-PIA parameter estimates |
| `XXXX_NoisyEstimateCNN.npy` | `200×200×3` | CNN-PIA parameter estimates |
| `XXXX_TissueType.npy` | `200×200` | Tissue segmentation (1=air, 8=tumor) |

**b-values**: [0, 5, 50, 100, 200, 500, 800, 1000] s/mm²

## IVIM Model

The bi-exponential IVIM signal model:

$$S(b) = (1-f) \cdot e^{-b \cdot D_t} + f \cdot e^{-b \cdot D^*}$$

| Parameter | Symbol | Units | Typical Range |
|-----------|--------|-------|---------------|
| Perfusion fraction | f | — | 0.05–0.35 |
| Tissue diffusivity | Dt | mm²/s | 0.0007–0.0015 |
| Pseudo-diffusivity | D* | mm²/s | 0.005–0.06 |

## Dependencies

- Python 3.8+
- PyTorch (with CUDA support for GPU inference)
- NumPy
- SciPy
- matplotlib
- python-pptx
- tqdm

## Usage

All models and evaluations are run from their respective directories:

```bash
# Run NLLS baseline
cd src
python NLLS_solution.py

# Benchmark inference speeds (CPU & GPU)
cd ../figures
python benchmark_inference.py

# Generate publication tables and figures
python plot_tables.py
python plot_noise_clean.py
python plot_paper_figures.py
```

## Citation

Based on the PIA-IVIM framework by Batuhan Gundogdu, University of Chicago Radiology.
Upstream: [batuhan-gundogdu/PIA_IVIM](https://github.com/batuhan-gundogdu/PIA_IVIM)
