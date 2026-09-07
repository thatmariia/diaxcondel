"""Tests for the two stationarity routes: dense eigenvalues and the winding number."""

import numpy as np
import pytest

from diaxcondel.model.build import build_lag_tensor
from diaxcondel.model.stability import (
    _companion_from_phi,
    count_roots_inside,
    is_stationary_from_phi,
    shrink_to_stationary,
    spectral_radius,
    spectral_radius_from_phi,
)


def _random_phi(n, p, scale, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(p, n, n)) * (scale / np.sqrt(n * p))


@pytest.mark.parametrize(("n", "p", "scale"), [(3, 20, 0.4), (4, 30, 1.5), (2, 50, 3.0)])
def test_winding_decision_matches_dense_eigenvalues(n, p, scale):
    phi = _random_phi(n, p, scale)
    dense_rho = spectral_radius(_companion_from_phi(phi))
    winding_says_stable = count_roots_inside(phi) == 0
    assert winding_says_stable == (dense_rho < 1.0)


@pytest.mark.parametrize(("n", "p", "scale"), [(3, 20, 0.4), (4, 30, 1.5), (5, 100, 0.5)])
def test_estimated_radius_matches_the_exact_one(n, p, scale):
    phi = _random_phi(n, p, scale)
    dense_rho = spectral_radius(_companion_from_phi(phi))
    estimate = spectral_radius_from_phi(phi, dense_limit=0)
    assert estimate == pytest.approx(dense_rho, rel=5e-3)


def test_estimated_radius_is_reproducible_and_scales():
    phi = _random_phi(4, 40, 1.0)
    first = spectral_radius_from_phi(phi, dense_limit=0)
    second = spectral_radius_from_phi(phi, dense_limit=0)
    assert first == second
    # Scaling every lag by s^p multiplies the eigenvalues by s.
    scaled = phi * (0.5 ** np.arange(1, phi.shape[0] + 1))[:, None, None]
    assert spectral_radius_from_phi(scaled, dense_limit=0) == pytest.approx(0.5 * first, rel=0.02)


def test_both_routes_agree_on_the_reference_model(distances, weights, kernel, sim_params):
    phi = build_lag_tensor(distances, weights, kernel, sim_params)
    for scale in (0.2, 0.5, 0.9, 1.3):
        scaled = phi * scale
        dense = spectral_radius(_companion_from_phi(scaled)) < 1.0
        assert is_stationary_from_phi(scaled, dense_limit=0) == dense


def test_zero_tensor_has_no_roots():
    phi = np.zeros((5, 2, 2))
    assert count_roots_inside(phi) == 0
    assert spectral_radius_from_phi(phi, dense_limit=0) == 0.0
    assert is_stationary_from_phi(phi)


def test_shrink_returns_a_stationary_tensor():
    phi = _random_phi(3, 40, 6.0)
    assert not is_stationary_from_phi(phi)
    shrunk, n_iter = shrink_to_stationary(phi, factor=0.8)
    assert n_iter > 0
    assert is_stationary_from_phi(shrunk)
    np.testing.assert_allclose(shrunk, phi * 0.8**n_iter)
