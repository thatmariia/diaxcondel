"""Shared pytest fixtures and options."""

import numpy as np
import pytest

from diaxcondel.connectome.atlases.manual import (
    steeghs_2025_distances,
    steeghs_2025_weights,
)
from diaxcondel.kernels.diameter import DiameterDelayKernel
from diaxcondel.kernels.gev import GEVDiameter
from diaxcondel.model.params import SimulationParams


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that download real atlas data (needs internet and the atlas extra)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "network: downloads real atlas data; needs --run-network")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(20260505)


@pytest.fixture
def sim_params():
    return SimulationParams(
        sample_rate_hz=1000,
        sim_seconds=10.0,
        burnin_seconds=1.0,
        n_lags=300,
    )


@pytest.fixture
def kernel():
    return DiameterDelayKernel(diameter=GEVDiameter(mu=0.25, sigma=0.09, xi=-0.25))


@pytest.fixture
def distances():
    return steeghs_2025_distances()


@pytest.fixture
def weights():
    return steeghs_2025_weights()
