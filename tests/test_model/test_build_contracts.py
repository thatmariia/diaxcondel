"""Tests for model construction contracts."""

import numpy as np

from diaxcondel.connectome.atlases.synthetic import PowerLawDistances
from diaxcondel.connectome.distance import ManualDistances, RelayedDistances
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.connectome.weights import ManualConnectivity
from diaxcondel.kernels.diameter import DiameterDelayKernel, TwoLegDelayKernel
from diaxcondel.kernels.gev import GEVDiameter
from diaxcondel.model.build import build_lag_tensor, build_linear_var
from diaxcondel.model.params import SimulationParams


def test_build_linear_var_accepts_stationarity_controls(distances, weights, kernel, sim_params):
    model = build_linear_var(
        distances,
        weights,
        kernel,
        sim_params,
        max_shrink_iter=5,
        shrink_factor=0.8,
    )
    assert model.phi.shape == (sim_params.n_lags, 5, 5)


def test_power_law_distances_are_deterministic_after_first_call():
    provider = PowerLawDistances(n_regions=8, beta=-1.0, d_min=1.0, d_max=10.0, rng=123)
    first = provider.matrix()
    second = provider.matrix()
    assert first is second
    np.testing.assert_array_equal(first, second)


def test_relayed_direct_connection_uses_single_leg_kernel():
    regions = RegionSet.from_codes(["A", "R", "B"])
    base = ManualDistances(
        regions,
        np.array(
            [
                [0.0, 2.0, 5.0],
                [2.0, 0.0, 3.0],
                [5.0, 3.0, 0.0],
            ],
        ),
    )
    relayed = RelayedDistances(base, relay_code="R")
    weights = ManualConnectivity(
        regions,
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
        ),
    )
    diameter = GEVDiameter(mu=0.25, sigma=0.09, xi=-0.25)
    kernel = TwoLegDelayKernel(diameter, n_samples=200, rng=123)
    params = SimulationParams(sample_rate_hz=1000, sim_seconds=1.0, burnin_seconds=0.0, n_lags=20)

    phi = build_lag_tensor(relayed, weights, kernel, params)
    direct = DiameterDelayKernel(diameter).pdf(np.arange(1, params.n_lags + 1) * params.dt_s, 2.0) * params.dt_s
    np.testing.assert_allclose(phi[:, 0, 1], direct)
