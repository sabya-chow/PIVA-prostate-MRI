"""Core physics, probabilistic inference, scoring, and reporting for the PIVA demo."""

# Import dataclass so each prepared patient has an explicit, readable data contract.
from dataclasses import dataclass

# Import HTML escaping so patient-supplied identifiers cannot break the generated report.
from html import escape

# Import Path so checkpoint paths work reliably on macOS and other platforms.
from pathlib import Path

# Import NumPy for synthetic MRI generation, posterior summaries, and probability scores.
import numpy as np

# Import PyTorch for the Full-ILR physics-informed variational model.
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the quantitative acquisition used throughout the capstone notebook.
B_VALUES = np.array([0, 150, 1000, 1500], dtype=np.float32)
TE_VALUES = np.array([0, 13, 93, 143], dtype=np.float32)
B_GRID = np.repeat(B_VALUES, len(TE_VALUES))
TE_GRID = np.tile(TE_VALUES, len(B_VALUES))
SIGNAL_SCALE = 1000.0

# Preserve the notebook's physiological parameterization for the three tissue compartments.
COMPARTMENTS = ("Epithelium", "Stroma", "Lumen")
D_MEAN = np.array([0.50, 1.20, 2.85], dtype=np.float32)
D_DELTA = np.array([0.20, 0.50, 0.15], dtype=np.float32)
T2_MEAN = np.array([45.0, 70.0, 750.0], dtype=np.float32)
T2_DELTA = np.array([25.0, 30.0, 250.0], dtype=np.float32)

# Define the transparent synthetic-reference thresholds used by the demonstration score.
REFERENCE_THRESHOLDS = {
    "v_ep_high": 0.42,
    "v_lumen_low": 0.20,
    "d_ep_low": 0.48,
}

# Define readable names and units once so the dashboard and downloaded report stay consistent.
PARAMETER_NAMES = (
    "Epithelium diffusion", "Stroma diffusion", "Lumen diffusion",
    "Epithelium T2", "Stroma T2", "Lumen T2",
    "Epithelium fraction", "Stroma fraction", "Lumen fraction",
)
PARAMETER_SHORT_NAMES = ("D ep", "D stroma", "D lumen", "T2 ep", "T2 stroma", "T2 lumen", "v ep", "v stroma", "v lumen")


# Format patient percentiles with correct English ordinal suffixes such as 1st, 2nd, and 13th.
def format_percentile(value: float) -> str:
    rounded = int(np.clip(np.rint(value), 0, 100))
    suffix = "th" if 10 <= rounded % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix} percentile"


@dataclass(frozen=True)
class DemoPatient:
    """One prepared synthetic patient containing ten quantitative MRI voxels."""

    patient_id: str
    label: str
    age: int
    psa: float
    psa_density: float
    noise_sigma: float
    signals: np.ndarray
    description: str


# Implement the exact three-compartment physics decoder used in the PIVA notebooks.
def physics_signal_numpy(D: np.ndarray, T2: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Convert every input to a two-dimensional float array so batch dimensions are consistent.
    D = np.atleast_2d(np.asarray(D, dtype=np.float32))
    T2 = np.atleast_2d(np.asarray(T2, dtype=np.float32))
    v = np.atleast_2d(np.asarray(v, dtype=np.float32))

    # Broadcast the 16 acquisition pairs against the three tissue compartments.
    diffusion = np.exp(-B_GRID[None, :, None] * D[:, None, :] / 1000.0)
    relaxation = np.exp(-TE_GRID[None, :, None] / T2[:, None, :])

    # Weight compartment signals by tissue fractions and return signals on the notebook's 1000 scale.
    return (SIGNAL_SCALE * (v[:, None, :] * diffusion * relaxation).sum(axis=2)).astype(np.float32)


# Generate a reproducible Rician magnitude measurement from a clean MRI signal.
def add_rician_noise(clean_signals: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    # Use an isolated random generator so loading a patient never changes global randomness.
    rng = np.random.default_rng(seed)

    # Convert to unit-scale signal because sigma is defined relative to S0 in the notebook.
    unit_signal = np.asarray(clean_signals, dtype=np.float32) / SIGNAL_SCALE

    # Simulate independent real and imaginary Gaussian measurement channels.
    real_noise = rng.normal(0.0, sigma, unit_signal.shape)
    imaginary_noise = rng.normal(0.0, sigma, unit_signal.shape)

    # Apply the MRI magnitude operation, which produces the Rician noise floor.
    return (SIGNAL_SCALE * np.sqrt((unit_signal + real_noise) ** 2 + imaginary_noise**2)).astype(np.float32)


# Create three fixed notebook-style patients for a dependable two-minute presentation path.
def build_demo_patients() -> dict[str, DemoPatient]:
    # Define patient-level tissue patterns; voxel jitter makes each case a small region, not one copied voxel.
    profiles = {
        "Within expected range": {
            "D": [0.56, 1.27, 2.87], "T2": [50, 74, 780], "v": [0.27, 0.34, 0.39],
            "age": 58, "psa": 4.7, "psad": 0.10, "sigma": 0.03,
            "description": "Balanced tissue composition with relatively stable diffusion and relaxation estimates.",
        },
        "Indeterminate pattern": {
            "D": [0.38, 1.12, 2.79], "T2": [34, 64, 650], "v": [0.53, 0.33, 0.14],
            "age": 64, "psa": 7.2, "psad": 0.16, "sigma": 0.05,
            "description": "Mixed quantitative findings with moderate uncertainty at the clinical reference noise level.",
        },
        "Elevated quantitative pattern": {
            "D": [0.31, 0.92, 2.73], "T2": [23, 48, 540], "v": [0.70, 0.23, 0.07],
            "age": 67, "psa": 9.1, "psad": 0.22, "sigma": 0.05,
            "description": "A moderately elevated regional pattern selected to demonstrate multiple quantitative deviations without absolute evidence.",
        },
    }

    # Use fixed patient seeds so the same case always produces the same presentation output.
    patients: dict[str, DemoPatient] = {}
    for index, (label, profile) in enumerate(profiles.items(), start=1):
        rng = np.random.default_rng(700 + index)
        voxel_count = 10

        # Add physiologically small within-patient variation to each compartment parameter.
        D = np.clip(
            rng.normal(profile["D"], [0.018, 0.045, 0.012], size=(voxel_count, 3)),
            D_MEAN - D_DELTA,
            D_MEAN + D_DELTA,
        )
        T2 = np.clip(
            rng.normal(profile["T2"], [3.0, 5.0, 35.0], size=(voxel_count, 3)),
            T2_MEAN - T2_DELTA,
            T2_MEAN + T2_DELTA,
        )

        # Draw tissue fractions around the profile while preserving nonnegativity and a sum of one.
        v = rng.dirichlet(np.asarray(profile["v"]) * 90.0, size=voxel_count)

        # Decode clean signals and add the same Rician noise process used in the notebook.
        clean = physics_signal_numpy(D, T2, v)
        noisy = add_rician_noise(clean, float(profile["sigma"]), seed=900 + index)

        # Package display metadata and measurements into the immutable patient record.
        patients[label] = DemoPatient(
            patient_id=f"PIVA-DEMO-{index:03d}",
            label=label,
            age=int(profile["age"]),
            psa=float(profile["psa"]),
            psa_density=float(profile["psad"]),
            noise_sigma=float(profile["sigma"]),
            signals=noisy,
            description=str(profile["description"]),
        )

    # Blend the elevated region with the indeterminate region to represent realistic partial-volume mixing.
    # A prostate region commonly contains more than one tissue pattern, and this prevents a deliberately
    # pure synthetic phenotype from producing artificial-looking 0th or 100th percentile observations.
    indeterminate_patient = patients["Indeterminate pattern"]
    elevated_patient = patients["Elevated quantitative pattern"]
    mixed_elevated_signals = 0.85 * indeterminate_patient.signals + 0.15 * elevated_patient.signals
    patients["Elevated quantitative pattern"] = DemoPatient(
        patient_id=elevated_patient.patient_id,
        label=elevated_patient.label,
        age=elevated_patient.age,
        psa=elevated_patient.psa,
        psa_density=elevated_patient.psa_density,
        noise_sigma=elevated_patient.noise_sigma,
        signals=mixed_elevated_signals.astype(np.float32),
        description=elevated_patient.description,
    )

    # Return label-keyed records so Streamlit can populate its patient selector directly.
    return patients


# Define a compact, deployable version of the notebook's Cholesky full-covariance + ILR architecture.
class PIVAFullILR(nn.Module):
    """Physics-informed variational network with an 8D full-covariance latent posterior."""

    def __init__(self, hidden_dims: tuple[int, ...] = (64, 128, 128)):
        super().__init__()
        self.latent_dim = 8
        self.hidden_dims = tuple(hidden_dims)

        # Build the shared encoder that converts 16 measured signals into latent features.
        layers: list[nn.Module] = []
        input_dim = 16
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)])
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Predict the latent mean and all 36 entries of an 8x8 lower-triangular Cholesky factor.
        self.mean_head = nn.Linear(input_dim, self.latent_dim)
        self.cholesky_head = nn.Linear(input_dim, self.latent_dim * (self.latent_dim + 1) // 2)

        # Store lower-triangle indices so every covariance is constructed consistently.
        triangle = torch.tril_indices(self.latent_dim, self.latent_dim)
        self.register_buffer("triangle_rows", triangle[0])
        self.register_buffer("triangle_cols", triangle[1])
        self.register_buffer("diagonal_mask", triangle[0] == triangle[1])

        # Store physiological constants and the orthonormal two-coordinate ILR basis as model buffers.
        self.register_buffer("D_mean", torch.tensor(D_MEAN))
        self.register_buffer("D_delta", torch.tensor(D_DELTA))
        self.register_buffer("T2_mean", torch.tensor(T2_MEAN))
        self.register_buffer("T2_delta", torch.tensor(T2_DELTA))
        self.register_buffer(
            "psi",
            torch.tensor(
                [[1 / np.sqrt(2), 1 / np.sqrt(6)], [-1 / np.sqrt(2), 1 / np.sqrt(6)], [0.0, -2 / np.sqrt(6)]],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("b_grid", torch.tensor(B_GRID))
        self.register_buffer("te_grid", torch.tensor(TE_GRID))

    # Convert each measured signal vector into a Gaussian mean and Cholesky factor.
    def encode(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(signals / SIGNAL_SCALE)
        mean = self.mean_head(features)
        raw_values = self.cholesky_head(features)

        # Fill a lower-triangular matrix and force positive diagonal elements using softplus.
        batch_size = signals.shape[0]
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=signals.device)
        values = raw_values.clone()
        values[:, self.diagonal_mask] = F.softplus(raw_values[:, self.diagonal_mask]) + 0.015
        L[:, self.triangle_rows, self.triangle_cols] = values
        return mean, L

    # Transform an unconstrained latent sample into physical D, T2, and volume fractions.
    def parameters_from_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.D_mean + self.D_delta * torch.tanh(latent[:, 0:3])
        T2 = self.T2_mean + self.T2_delta * torch.tanh(latent[:, 3:6])
        centered_log_ratio = latent[:, 6:8] @ self.psi.T
        v = F.softmax(centered_log_ratio, dim=1)
        return D, T2, v

    # Reconstruct the 16 measurements through the fixed prostate MRI physics equation.
    def decode(self, D: torch.Tensor, T2: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        diffusion = torch.exp(-self.b_grid[None, :, None] * D[:, None, :] / 1000.0)
        relaxation = torch.exp(-self.te_grid[None, :, None] / T2[:, None, :])
        return SIGNAL_SCALE * (v[:, None, :] * diffusion * relaxation).sum(dim=2)

    # Return a deterministic or stochastic parameter realization and its reconstructed signal.
    def forward(self, signals: torch.Tensor, sample: bool = True):
        mean, L = self.encode(signals)
        if sample:
            epsilon = torch.randn_like(mean)
            latent = mean + torch.bmm(L, epsilon.unsqueeze(-1)).squeeze(-1)
        else:
            latent = mean
        D, T2, v = self.parameters_from_latent(latent)
        return self.decode(D, T2, v), D, T2, v, mean, L


# Load the exported checkpoint and verify that its architecture metadata is compatible.
def load_model(checkpoint_path: str | Path) -> tuple[PIVAFullILR, dict]:
    # Load onto CPU because the Streamlit demonstration is deliberately hardware-independent.
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    hidden_dims = tuple(checkpoint.get("hidden_dims", (64, 128, 128)))
    model = PIVAFullILR(hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# Build a model-matched synthetic reference cohort for ranges and percentile positioning.
def build_reference_profile(model: PIVAFullILR, n_voxels: int = 5000, seed: int = 314) -> dict:
    # Generate a broad but balanced synthetic reference population from the notebook's physical priors.
    rng = np.random.default_rng(seed)
    D = np.clip(rng.normal(D_MEAN, D_DELTA / 3.0, size=(n_voxels, 3)), D_MEAN - D_DELTA, D_MEAN + D_DELTA)
    T2 = np.clip(rng.normal(T2_MEAN, T2_DELTA / 3.0, size=(n_voxels, 3)), T2_MEAN - T2_DELTA, T2_MEAN + T2_DELTA)
    v = rng.dirichlet([2.2, 2.4, 2.0], size=n_voxels)

    # Pass reference signals through the same trained model used for the patient report.
    clean = physics_signal_numpy(D, T2, v)
    noisy = add_rician_noise(clean, sigma=0.05, seed=seed + 1)
    with torch.no_grad():
        signals = torch.tensor(noisy, dtype=torch.float32)
        _, D_hat, T2_hat, v_hat, _, _ = model(signals, sample=False)
    estimates = np.concatenate([D_hat.numpy(), T2_hat.numpy(), v_hat.numpy()], axis=1)

    # Use the central 80% as the visible reference band and retain all samples for percentile calculation.
    return {
        "samples": estimates,
        "lower": np.percentile(estimates, 10, axis=0),
        "median": np.percentile(estimates, 50, axis=0),
        "upper": np.percentile(estimates, 90, axis=0),
        "n": int(n_voxels),
        "label": "Synthetic model-matched reference cohort",
    }


# Run Monte Carlo inference and derive calibrated intervals, probabilities, and patient-level summaries.
def analyze_patient(model: PIVAFullILR, patient: DemoPatient, n_mc: int = 300, reference_profile: dict | None = None) -> dict:
    # Fix the inference seed so the live presentation produces stable values across reruns.
    torch.manual_seed(2026)
    signals = torch.tensor(patient.signals, dtype=torch.float32)

    # Draw posterior samples in one vectorized batch for fast CPU inference.
    with torch.no_grad():
        mean, L = model.encode(signals)
        voxel_count = signals.shape[0]
        epsilon = torch.randn(n_mc, voxel_count, model.latent_dim)
        latent = mean.unsqueeze(0) + torch.einsum("bij,mbj->mbi", L, epsilon)
        flat_latent = latent.reshape(-1, model.latent_dim)
        D_flat, T2_flat, v_flat = model.parameters_from_latent(flat_latent)
        D_samples = D_flat.reshape(n_mc, voxel_count, 3).numpy()
        T2_samples = T2_flat.reshape(n_mc, voxel_count, 3).numpy()
        v_samples = v_flat.reshape(n_mc, voxel_count, 3).numpy()

    # Pool samples across voxels to summarize the analyzed patient region.
    patient_D = D_samples.mean(axis=1)
    patient_T2 = T2_samples.mean(axis=1)
    patient_v = v_samples.mean(axis=1)
    all_samples = np.concatenate([patient_D, patient_T2, patient_v], axis=1)

    # Compute posterior means and empirical 95% intervals without assuming normality after transformation.
    estimates = all_samples.mean(axis=0)
    lower = np.percentile(all_samples, 2.5, axis=0)
    upper = np.percentile(all_samples, 97.5, axis=0)

    # Decode the posterior-mean physical parameters to show how well PIVA explains the observed signal.
    reconstructed = physics_signal_numpy(estimates[0:3], estimates[3:6], estimates[6:9])[0]
    observed_mean = patient.signals.mean(axis=0)
    normalized_rmse = float(np.sqrt(np.mean((observed_mean - reconstructed) ** 2)) / max(observed_mean.max(), 1.0))
    reconstruction_score = float(np.clip(100.0 * (1.0 - normalized_rmse), 0.0, 100.0))

    # Use model-matched reference boundaries when available so displayed ranges and scoring agree.
    threshold_v_ep = float(reference_profile["upper"][6]) if reference_profile is not None else REFERENCE_THRESHOLDS["v_ep_high"]
    threshold_v_lumen = float(reference_profile["lower"][8]) if reference_profile is not None else REFERENCE_THRESHOLDS["v_lumen_low"]
    threshold_D_ep = float(reference_profile["lower"][0]) if reference_profile is not None else REFERENCE_THRESHOLDS["d_ep_low"]

    # Count how often posterior samples cross each reference boundary.
    raw_probabilities = {
        "epithelium_high": float((patient_v[:, 0] > threshold_v_ep).mean()),
        "lumen_low": float((patient_v[:, 2] < threshold_v_lumen).mean()),
        "diffusion_restricted": float((patient_D[:, 0] < threshold_D_ep).mean()),
    }

    # Regularize finite Monte Carlo proportions with a symmetric Beta prior to avoid false 0%/100% certainty.
    # The prior contributes 45 virtual successes and 45 virtual failures to 300 posterior draws,
    # limiting displayed evidence to a cautious range while preserving ordering between patients.
    prior_count = 45.0 * (n_mc / 300.0)
    probabilities = {
        name: float((probability * n_mc + prior_count) / (n_mc + 2.0 * prior_count))
        for name, probability in raw_probabilities.items()
    }

    # Combine the three posterior probabilities using the explicitly documented demo weights.
    concern_score = 100.0 * (
        0.40 * probabilities["epithelium_high"]
        + 0.30 * probabilities["lumen_low"]
        + 0.30 * probabilities["diffusion_restricted"]
    )
    concern_category = "Within expected range" if concern_score < 35 else "Indeterminate" if concern_score < 65 else "Elevated"

    # Position every patient estimate inside the same model-matched synthetic reference distribution.
    if reference_profile is not None:
        reference_samples = np.asarray(reference_profile["samples"])
        percentiles = np.array([(reference_samples[:, index] <= estimates[index]).mean() * 100.0 for index in range(9)])
        reference_lower = np.asarray(reference_profile["lower"])
        reference_median = np.asarray(reference_profile["median"])
        reference_upper = np.asarray(reference_profile["upper"])
    else:
        percentiles = np.full(9, 50.0)
        reference_lower = np.concatenate([D_MEAN - D_DELTA / 2, T2_MEAN - T2_DELTA / 2, [0.15, 0.15, 0.15]])
        reference_median = np.concatenate([D_MEAN, T2_MEAN, [1 / 3, 1 / 3, 1 / 3]])
        reference_upper = np.concatenate([D_MEAN + D_DELTA / 2, T2_MEAN + T2_DELTA / 2, [0.55, 0.55, 0.55]])

    # Label whether each mean estimate falls below, inside, or above the central reference band.
    range_status = np.where(estimates < reference_lower, "Below reference", np.where(estimates > reference_upper, "Above reference", "Within reference"))

    # Base the overall confidence score on the same reference-span denominator used by the precision chart.
    interval_width = upper - lower
    reference_width = np.maximum(reference_upper - reference_lower, 1e-6)
    mean_relative_uncertainty = float(np.mean(interval_width / reference_width))
    confidence_score = float(np.clip(100.0 - 50.0 * mean_relative_uncertainty, 0.0, 100.0))
    confidence_label = "High" if confidence_score >= 80 else "Moderate" if confidence_score >= 60 else "Limited"

    # Identify the largest patient-reference deviations for the plain-language key-driver section.
    distance_from_center = np.abs(percentiles - 50.0)
    driver_indices = np.argsort(distance_from_center)[::-1][:3]

    # Preserve voxel-level posterior means so the patient can see regional heterogeneity rather than one opaque average.
    voxel_estimates = np.concatenate([D_samples.mean(axis=0), T2_samples.mean(axis=0), v_samples.mean(axis=0)], axis=1)

    # Return both patient-level summaries and raw arrays required by the dashboard charts.
    return {
        "estimates": estimates,
        "lower": lower,
        "upper": upper,
        "samples": all_samples,
        "probabilities": probabilities,
        "raw_probabilities": raw_probabilities,
        "concern_score": float(concern_score),
        "concern_category": concern_category,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "reconstruction_score": reconstruction_score,
        "observed_signal": observed_mean,
        "reconstructed_signal": reconstructed,
        "reference_lower": reference_lower,
        "reference_median": reference_median,
        "reference_upper": reference_upper,
        "reference_n": int(reference_profile["n"]) if reference_profile is not None else 0,
        "percentiles": percentiles,
        "range_status": range_status,
        "driver_indices": driver_indices,
        "voxel_estimates": voxel_estimates,
        "score_thresholds": {"v_ep": threshold_v_ep, "v_lumen": threshold_v_lumen, "D_ep": threshold_D_ep},
    }


# Build a standalone, patient-readable HTML report that can be downloaded from Streamlit.
def build_html_report(patient: DemoPatient, result: dict) -> str:
    # Pull frequently used result arrays into short local names for readable report construction.
    estimates = result["estimates"]
    lower = result["lower"]
    upper = result["upper"]

    # Select measured, restrained language based on the transparent quantitative concern category.
    category = result["concern_category"]
    summary = {
        "Within expected range": "The analyzed region is broadly consistent with the synthetic reference pattern.",
        "Indeterminate": "The analyzed region contains mixed quantitative findings and should be interpreted with its uncertainty.",
        "Elevated": "Several quantitative tissue estimates differ from the synthetic reference pattern.",
    }[category]

    # Build one table row for each of the nine physical estimates.
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        is_fraction = index >= 6
        value = f"{estimates[index] * 100:.1f}%" if is_fraction else f"{estimates[index]:.2f}"
        interval = f"{lower[index] * 100:.1f}% to {upper[index] * 100:.1f}%" if is_fraction else f"{lower[index]:.2f} to {upper[index]:.2f}"
        percentile = float(result["percentiles"][index])
        status = str(result["range_status"][index])
        status_class = "flag" if status != "Within reference" else "normal"
        rows.append(f"<tr><td>{name}</td><td>{value}</td><td>{interval}</td><td>{format_percentile(percentile)}</td><td class='{status_class}'>{status}</td></tr>")

    # Build visual percentile rulers for the three largest drivers of the quantitative result.
    driver_bars = []
    for index in result["driver_indices"]:
        percentile = float(result["percentiles"][index])
        driver_bars.append(
            f"<div class='driver'><b>{PARAMETER_NAMES[index]}</b><span>{format_percentile(percentile)}</span>"
            f"<div class='track'><div class='normal-zone'></div><div class='marker' style='left:calc({percentile:.1f}% - 6px)'></div></div>"
            "<div class='axis'><span>Lower than reference</span><span>Typical range</span><span>Higher than reference</span></div></div>"
        )

    # Return a complete HTML document with embedded styling so the download has no external dependencies.
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PIVA report</title>
<style>
body{{font-family:Arial,sans-serif;color:#17243a;max-width:900px;margin:38px auto;line-height:1.55}}
h1{{color:#102a43}} .badge{{display:inline-block;padding:8px 14px;border-radius:20px;background:#e6f0ff;color:#174ea6;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}} .card{{border:1px solid #d7e1ea;border-radius:12px;padding:16px}}
.value{{font-size:28px;font-weight:700}} table{{width:100%;border-collapse:collapse;margin:18px 0}} th,td{{padding:10px;border-bottom:1px solid #d7e1ea;text-align:left}}
.note{{background:#fff7e6;border-left:4px solid #d99000;padding:14px;margin-top:24px}} footer{{margin-top:34px;color:#60758a;font-size:12px}}
.driver{{margin:18px 0}} .driver span{{float:right;color:#48657b}} .track{{position:relative;height:14px;background:#edf1f4;border-radius:8px;margin-top:8px;clear:both}}
.normal-zone{{position:absolute;left:10%;width:80%;height:14px;background:#bde5d8;border-radius:8px}} .marker{{position:absolute;top:-4px;width:12px;height:22px;background:#173f5f;border-radius:4px}}
.axis{{display:flex;justify-content:space-between;font-size:10px;color:#718496;margin-top:3px}} .flag{{color:#a64b00;font-weight:700}} .normal{{color:#237a57;font-weight:700}}
</style></head><body>
<h1>PIVA Quantitative Prostate MRI Report</h1>
<p><span class="badge">Synthetic capstone demonstration</span></p>
<p><strong>Patient:</strong> {escape(patient.patient_id)} &nbsp; <strong>Age:</strong> {patient.age} &nbsp; <strong>PSA:</strong> {patient.psa:.1f} ng/mL</p>
<h2>Summary</h2><p>{summary} PIVA classified the demonstration pattern as <strong>{category.lower()}</strong>, with {result['confidence_label'].lower()} quantitative confidence.</p>
<div class="grid"><div class="card"><div>Concern score</div><div class="value">{result['concern_score']:.0f}/100</div></div>
<div class="card"><div>Signal reconstruction</div><div class="value">{result['reconstruction_score']:.0f}%</div></div>
<div class="card"><div>Quantitative confidence</div><div class="value">{result['confidence_label']}</div></div></div>
<h2>Where this patient stands</h2><p>The markers compare this patient's estimated values with {result['reference_n']:,} model-matched synthetic reference voxels. The green area represents the central 80% reference band.</p>{''.join(driver_bars)}
<h2>What influenced the score</h2><ul>
<li>{result['probabilities']['epithelium_high']:.0%} regularized posterior evidence that epithelial fraction exceeds the synthetic reference threshold.</li>
<li>{result['probabilities']['lumen_low']:.0%} regularized posterior evidence that lumen fraction is below the synthetic reference threshold.</li>
<li>{result['probabilities']['diffusion_restricted']:.0%} regularized posterior evidence of comparatively restricted epithelial diffusion.</li></ul>
<h2>Quantitative tissue estimates</h2><table><thead><tr><th>Measurement</th><th>Estimate</th><th>95% posterior interval</th><th>Percentile</th><th>Reference comparison</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Understanding the measurements</h2><p><strong>Diffusion (D)</strong> describes how freely water moves through tissue. <strong>T2</strong> describes how quickly the MRI signal decays. <strong>Volume fractions (v)</strong> estimate the relative tissue composition. Posterior intervals show how much the model's estimate changes across plausible latent explanations of the measured signal.</p>
<div class="note"><strong>Important:</strong> This report uses synthetic patient data and synthetic-reference thresholds for an academic product demonstration. Evidence values are conservatively regularized to avoid false 0% or 100% certainty. The concern score is not a cancer probability, PI-RADS score, diagnosis, or clinical recommendation.</div>
<footer>Model: PIVA Full-ILR / Cholesky full covariance · Acquisition: 4 b-values × 4 echo times · MSADS Capstone Demonstration</footer>
</body></html>"""
