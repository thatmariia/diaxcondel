"""Setting the model's distance from criticality directly."""

import numpy as np
import pytest

from diaxcondel.experiment import ExperimentSpec, build_model
from diaxcondel.model.stability import scale_to_spectral_radius, spectral_radius_from_phi


def _phi(seed: int = 0, p: int = 12, n: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.05, size=(p, n, n))


@pytest.mark.parametrize("target", [0.6, 0.9, 0.99, 1.05])
def test_scaling_hits_the_requested_radius(target):
    scaled, factor = scale_to_spectral_radius(_phi(), target)
    assert spectral_radius_from_phi(scaled) == pytest.approx(target, rel=1e-3)
    assert factor > 0


def test_scaling_moves_every_connection_by_the_same_factor():
    phi = _phi()
    scaled, factor = scale_to_spectral_radius(phi, 0.9)
    # Relative connectivity and the shape of every delay distribution are
    # untouched, so only the operating point changes.
    assert np.allclose(scaled, phi * factor)


def test_the_spec_carries_the_operating_point_into_the_model():
    spec = (
        ExperimentSpec(seed=1)
        .with_simulation(sample_rate_hz=200, sim_seconds=2.0, burnin_seconds=0.5, n_lags=60)
        .model_copy(update={"target_spectral_radius": 0.85})
    )
    built = build_model(spec)
    assert built.spectral_radius() == pytest.approx(0.85, rel=1e-3)
    # The rescaling changes effective coupling, so it is reported rather than
    # applied silently.
    assert any("spectral radius" in note for note in built.notes)


def test_the_operating_point_sharpens_the_spectrum():
    base = ExperimentSpec(seed=1).with_simulation(sample_rate_hz=200, sim_seconds=2.0, burnin_seconds=0.5, n_lags=60)
    sharpness = []
    for target in (0.7, 0.95):
        built = build_model(base.model_copy(update={"target_spectral_radius": target}))
        freqs = np.linspace(1.0, 40.0, 300)
        transfer = built.dynamics.transfer(freqs, built.params.dt_s)
        power = np.abs(np.diagonal(transfer, axis1=1, axis2=2)) ** 2
        sharpness.append(float(power.max() / np.median(power)))
    assert sharpness[1] > sharpness[0]
