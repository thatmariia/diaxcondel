"""Tests for local (receiver-side) delay kernels and their effect on a model."""

import numpy as np
import pytest

from diaxcondel.catalog import build as build_component
from diaxcondel.kernels.local import GammaLocalDelay, NoLocalDelay, local_delay_mass
from diaxcondel.model.build import build_lag_tensor
from diaxcondel.model.params import SimulationParams


@pytest.fixture
def params():
    return SimulationParams(sample_rate_hz=1000, sim_seconds=1.0, burnin_seconds=0.5, n_lags=200)


@pytest.fixture
def sources():
    distances = build_component("distances", "steeghs_2025")
    weights = build_component("connectivity", "steeghs_2025", regions=distances.regions, distances=distances)
    kernel = build_component("kernel", "hursh", diameter=build_component("diameter", "gev"))
    return distances, weights, kernel


def test_no_local_delay_puts_all_mass_at_zero_offset():
    mass = NoLocalDelay().mass(1e-3, 20)
    assert mass[0] == 1.0
    assert mass[1:].sum() == 0.0
    np.testing.assert_array_equal(local_delay_mass(None, 1e-3, 20), mass)
    assert NoLocalDelay().parameters() == {}


def test_gamma_local_delay_has_the_requested_mean_and_spread():
    kernel = GammaLocalDelay(mean_ms=8.0, shape=4.0)
    mass = kernel.mass(1e-3, 200)
    lags_ms = np.arange(200)
    assert mass.sum() == pytest.approx(1.0)
    assert float(np.dot(lags_ms, mass)) == pytest.approx(8.0, abs=0.5)
    spread = np.sqrt(np.dot((lags_ms - 8.0) ** 2, mass))
    assert spread == pytest.approx(8.0 / 2.0, abs=0.6)  # mean / sqrt(shape)
    assert kernel.parameters() == {"mean_ms": 8.0, "shape": 4.0}


def test_local_delay_postpones_arrivals_without_changing_total_coupling(sources, params):
    distances, weights, kernel = sources
    plain = build_lag_tensor(distances, weights, kernel, params)
    delayed = build_lag_tensor(distances, weights, kernel, params, local_kernel=GammaLocalDelay(mean_ms=10.0))

    lags_ms = np.arange(1, params.n_lags + 1)
    peak_plain = lags_ms[np.argmax(plain[:, 2, 1])]
    peak_delayed = lags_ms[np.argmax(delayed[:, 2, 1])]
    assert peak_delayed - peak_plain == pytest.approx(10.0, abs=3.0)
    assert delayed.sum() == pytest.approx(plain.sum(), rel=1e-3)


def test_self_excitation_lands_on_the_diagonal(sources, params):
    distances, weights, kernel = sources
    phi = build_lag_tensor(
        distances,
        weights,
        kernel,
        params,
        local_kernel=GammaLocalDelay(mean_ms=4.0),
        self_weight=0.25,
    )
    diagonal = np.diagonal(phi, axis1=1, axis2=2)
    np.testing.assert_allclose(diagonal.sum(axis=0), [0.25] * 5, rtol=1e-6)
    lags_ms = np.arange(1, params.n_lags + 1)
    assert lags_ms[np.argmax(phi[:, 0, 0])] == pytest.approx(4.0, abs=2.0)
