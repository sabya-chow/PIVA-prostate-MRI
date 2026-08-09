# DESIGN

This repository implements a Physics-Informed Variational Autoencoder (PIVA) for estimating prostate MRI parameters.

- Purpose: Provide an end-to-end implementation and experiments for using a physics-informed variational autoencoder to estimate quantitative MRI parameters from prostate MRI data.
- Inputs: Raw or preprocessed prostate MRI images (as used in the notebooks); data preprocessing steps are demonstrated in the notebooks.
- Outputs: Estimated quantitative parameter maps, reconstructed images, training logs, and evaluation metrics (e.g., parameter error, reconstruction quality).
- Model: Variational Autoencoder architecture augmented with a physics-based forward model (MRI signal model) integrated into training to constrain reconstructions.
- Training: Notebooks show training loops, loss design (reconstruction + KL + physics/data-fidelity terms), hyperparameters, and saving/loading model checkpoints.
- Physics integration: The forward MRI signal model is embedded in the loss to enforce physically plausible parameter estimates during inference and training.
- Evaluation: Notebooks include evaluation routines and visualization of parameter maps, reconstructions, and comparisons to ground truth or baseline methods.
- Reproducibility & usage: The repo is notebook-driven — run the notebooks in sequence. Ensure required Python packages and GPU support (if available) as noted in the notebooks.

Notes for an LLM helping finalize the repo:
- Expect the code to be primarily in Jupyter notebooks; point out opportunities to extract reusable modules (data, model, training, metrics) into .py files.
- Suggest clear dependency specification (requirements.txt or environment.yml) and a README with step-by-step run instructions.
- Recommend breaking long notebooks into smaller, documented scripts for testing and CI.
