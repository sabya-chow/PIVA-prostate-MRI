"""Train and export a compact Full-ILR PIVA checkpoint for the local capstone demo."""

# Import Path so the checkpoint is always written beside this script.
from pathlib import Path

# Import NumPy for synthetic cohort generation.
import numpy as np

# Import PyTorch training tools.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import the shared model and physics utilities used by the Streamlit application.
from piva_core import D_DELTA, D_MEAN, SIGNAL_SCALE, T2_DELTA, T2_MEAN, PIVAFullILR, add_rician_noise, physics_signal_numpy


# Fix random seeds so the exported demonstration checkpoint is reproducible.
np.random.seed(42)
torch.manual_seed(42)


# Generate a notebook-style synthetic cohort with valid physical parameters and tissue fractions.
def generate_training_data(n_samples: int = 30000, seed: int = 42):
    # Use the same bounded Gaussian parameter families and Dirichlet compositions as the notebook.
    rng = np.random.default_rng(seed)
    D = np.clip(rng.normal(D_MEAN, D_DELTA / 3.0, size=(n_samples, 3)), D_MEAN - D_DELTA, D_MEAN + D_DELTA)
    T2 = np.clip(rng.normal(T2_MEAN, T2_DELTA / 3.0, size=(n_samples, 3)), T2_MEAN - T2_DELTA, T2_MEAN + T2_DELTA)
    v = rng.dirichlet([2.2, 2.4, 2.0], size=n_samples)

    # Decode noise-free signals and apply clinically plausible Rician noise between sigma 0.01 and 0.08.
    clean = physics_signal_numpy(D, T2, v)
    sigma = rng.uniform(0.01, 0.08, n_samples)
    noisy_parts = [add_rician_noise(clean[index:index + 1], float(sigma[index]), seed + index) for index in range(n_samples)]
    noisy = np.vstack(noisy_parts)

    # Concatenate the nine physical targets for a stable simulation-supervised demo checkpoint.
    targets = np.concatenate([D, T2, v], axis=1).astype(np.float32)
    return noisy.astype(np.float32), clean.astype(np.float32), targets


# Train the full-covariance variational model while retaining the fixed physics reconstruction objective.
def main():
    # Generate the complete synthetic training matrix once.
    noisy, clean, targets = generate_training_data()
    dataset = TensorDataset(torch.tensor(noisy), torch.tensor(clean), torch.tensor(targets))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Use Apple GPU acceleration when available, otherwise remain fully CPU-compatible.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PIVAFullILR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-6)

    # Normalize target scales so D, T2, and volume fractions contribute comparably.
    target_scale = torch.tensor([*D_DELTA, *T2_DELTA, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    # Train long enough for a stable presentation checkpoint without reproducing the notebook's full experiment runtime.
    epochs = 70
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for noisy_batch, clean_batch, target_batch in loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            target_batch = target_batch.to(device)

            # Run one stochastic Full-ILR posterior sample through the fixed physics decoder.
            reconstruction, D, T2, v, mean, L = model(noisy_batch, sample=True)
            prediction = torch.cat([D, T2, v], dim=1)

            # Combine physics reconstruction, simulation-supervised parameter accuracy, and a light Gaussian KL penalty.
            reconstruction_loss = F.mse_loss(reconstruction / SIGNAL_SCALE, clean_batch / SIGNAL_SCALE)
            parameter_loss = F.mse_loss((prediction - target_batch) / target_scale, torch.zeros_like(target_batch))
            diagonal = torch.diagonal(L, dim1=1, dim2=2)
            kl = 0.5 * ((L**2).sum((1, 2)) + (mean**2).sum(1) - model.latent_dim - 2.0 * torch.log(diagonal).sum(1)).mean()
            loss = 4.0 * reconstruction_loss + parameter_loss + 0.002 * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        # Print compact progress every ten epochs for a readable one-time export log.
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | loss={np.mean(losses):.5f}")

    # Save architecture metadata and a clear provenance label with the deployable weights.
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "piva_full_ilr_demo.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "hidden_dims": model.hidden_dims,
            "model_name": "PIVA Full-ILR / Cholesky Full Covariance",
            "training_scope": "Capstone demonstration checkpoint trained on notebook-style synthetic signals",
            "seed": 42,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


# Run training only when the export script is called directly.
if __name__ == "__main__":
    main()
