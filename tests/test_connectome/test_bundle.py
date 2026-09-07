"""Tests for the Connectome bundle."""

import pickle

import numpy as np
import pytest

from diaxcondel.connectome.bundle import Connectome, region_metadata
from diaxcondel.connectome.distance import RelayedDistances
from diaxcondel.connectome.regions import Region, RegionSet

DISTANCES = np.array(
    [
        [0.0, 2.0, 5.0],
        [2.0, 0.0, 3.0],
        [5.0, 3.0, 0.0],
    ]
)
WEIGHTS = np.array(
    [
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.25],
        [0.5, 0.25, 0.0],
    ]
)


def _bundle() -> Connectome:
    regions = RegionSet.from_codes(["A", "R", "B"])
    return Connectome.from_matrices(regions, DISTANCES, WEIGHTS, name="test")


def test_describe_reports_ranges_and_density():
    bundle = _bundle()
    summary = bundle.describe()
    assert summary["n_regions"] == 3
    assert summary["density"] == 1.0
    assert summary["distance_cm_min"] == 2.0
    assert summary["distance_cm_max"] == 5.0
    assert summary["weight_max"] == 1.0


def test_density_ignores_zero_weights():
    regions = RegionSet.from_codes(["A", "B", "C"])
    weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    bundle = Connectome.from_matrices(regions, DISTANCES, weights)
    assert bundle.density == pytest.approx(2 / 6)


def test_select_reorders_and_subsets():
    subset = _bundle().select(["B", "A"])
    assert subset.codes == ("B", "A")
    np.testing.assert_allclose(subset.distance_matrix(), np.array([[0.0, 5.0], [5.0, 0.0]]))


def test_relayed_routes_through_relay_region():
    relayed = _bundle().relayed("R")
    assert isinstance(relayed.distances, RelayedDistances)
    matrix = relayed.distance_matrix()
    assert matrix[0, 2] == pytest.approx(DISTANCES[0, 1] + DISTANCES[1, 2])
    assert matrix[0, 1] == pytest.approx(DISTANCES[0, 1])
    assert relayed.metadata["relay_code"] == "R"


def test_rescale_weights_scales_and_validates():
    scaled = _bundle().rescale_weights(0.5)
    np.testing.assert_allclose(scaled.weight_matrix(), WEIGHTS * 0.5)
    assert _bundle().rescale_weights(1.0) is not None


def test_bundle_and_regions_survive_pickling():
    bundle = Connectome.from_matrices(
        RegionSet((Region(code="A", name="a", metadata={"x": 1}), Region(code="B"))),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        metadata={"source": "test"},
    )
    restored = pickle.loads(pickle.dumps(bundle))
    assert restored.codes == ("A", "B")
    assert dict(restored.metadata) == {"source": "test"}
    assert region_metadata(restored.regions[0], "x") == 1
