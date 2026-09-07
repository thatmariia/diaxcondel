"""Tests for parameter sweeps."""

import numpy as np
import pytest

from diaxcondel.experiment import (
    ExperimentSpec,
    get_parameter,
    parameter_paths,
    set_parameter,
    sweep,
)


@pytest.fixture
def small_spec():
    return ExperimentSpec(seed=3).with_simulation(sample_rate_hz=100, sim_seconds=2.0, burnin_seconds=0.5, n_lags=30)


def test_parameter_paths_cover_components_and_settings(small_spec):
    paths = parameter_paths(small_spec)
    assert "kernel.speed_factor" in paths
    assert "connectivity.scale" in paths
    assert "simulation.sample_rate_hz" in paths
    assert "self_weight" in paths
    # Catalog metadata travels with component parameters.
    assert paths["kernel.speed_factor"].unit == "m/s/um"
    assert paths["self_weight"] is None


def test_get_and_set_round_trip(small_spec):
    assert get_parameter(small_spec, "kernel.speed_factor") == 6.0
    updated = set_parameter(small_spec, "kernel.speed_factor", 8.0)
    assert get_parameter(updated, "kernel.speed_factor") == 8.0
    assert get_parameter(small_spec, "kernel.speed_factor") == 6.0, "the original must not change"

    assert get_parameter(set_parameter(small_spec, "simulation.sim_seconds", 3.0), "simulation.sim_seconds") == 3.0
    assert get_parameter(set_parameter(small_spec, "n_trials", 4), "n_trials") == 4


def test_sweep_reports_one_point_per_value(small_spec):
    seen: list[int] = []
    result = sweep(
        small_spec,
        {"kernel.speed_factor": [4.0, 8.0]},
        progress=lambda index, total, values: seen.append(index),
    )
    assert seen == [0, 1]
    assert len(result.points) == 2
    assert [point.values["kernel.speed_factor"] for point in result.points] == [4.0, 8.0]

    rows = result.summary_table()
    assert set(rows[0]) == {"kernel.speed_factor", "spectral radius", "peak (Hz)", "1/f slope", "variance"}
    region_rows = result.region_table()
    assert len(region_rows) == 2 * 5
    assert {row["region"] for row in region_rows} == {"FL", "PL", "OL", "TL", "T"}


def test_faster_conduction_raises_the_peak_frequency(small_spec):
    result = sweep(small_spec, {"kernel.speed_factor": [4.0, 9.0]})
    slow, fast = result.points
    assert fast.mean_peak_hz > slow.mean_peak_hz


def test_sweep_grid_covers_every_combination(small_spec):
    result = sweep(small_spec, {"kernel.speed_factor": [5.0, 7.0], "connectivity.scale": [0.5, 1.0]})
    assert len(result.points) == 4
    assert len(result.axes) == 2
    combinations = {tuple(point.values.values()) for point in result.points}
    assert combinations == {(5.0, 0.5), (5.0, 1.0), (7.0, 0.5), (7.0, 1.0)}


def test_sweep_points_expose_aggregate_measures(small_spec):
    result = sweep(small_spec, {"connectivity.scale": [0.4]})
    point = result.points[0]
    assert np.isfinite(point.mean_peak_hz)
    assert np.isfinite(point.mean_slope)
    assert point.mean_variance > 0
    assert 0.0 < point.spectral_radius < 1.05
