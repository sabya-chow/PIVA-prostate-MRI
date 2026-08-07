"""Fast correctness checks for the PIVA capstone demonstration."""

# Import NumPy for numeric assertions.
import numpy as np

# Import PyTorch to validate the full-covariance model output contract.
import torch

# Import the application components under test.
from piva_core import PIVAFullILR, build_demo_patients, physics_signal_numpy


# Confirm the physical decoder returns one finite 16-measurement signal per voxel.
def test_physics_decoder_shape():
    signal = physics_signal_numpy([[0.5, 1.2, 2.85]], [[45, 70, 750]], [[0.3, 0.4, 0.3]])
    assert signal.shape == (1, 16)
    assert np.isfinite(signal).all()
    np.testing.assert_allclose(signal[0, 0], 1000.0, atol=1e-4)


# Confirm every prepared patient follows the promised ten-voxel by sixteen-measurement contract.
def test_demo_patient_shapes():
    for patient in build_demo_patients().values():
        assert patient.signals.shape == (10, 16)
        assert np.isfinite(patient.signals).all()


# Confirm Full-ILR inference preserves physiological constraints and a valid Cholesky diagonal.
def test_model_constraints():
    model = PIVAFullILR()
    signals = torch.tensor(next(iter(build_demo_patients().values())).signals)
    reconstruction, D, T2, v, _, L = model(signals, sample=False)
    assert reconstruction.shape == (10, 16)
    assert torch.all(D > 0)
    assert torch.all(T2 > 0)
    assert torch.allclose(v.sum(dim=1), torch.ones(10), atol=1e-5)
    assert torch.all(torch.diagonal(L, dim1=1, dim2=2) > 0)
