"""Tests for the network layout helpers."""

import numpy as np
import pytest

from diaxcondel.viz.network import complete_distances, mds_layout, stress_layout


def _triangle() -> np.ndarray:
    """A 3-4-5 triangle: an exact planar geometry to recover."""
    return np.array(
        [
            [0.0, 3.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, 5.0, 0.0],
        ]
    )


def test_layout_recovers_an_exactly_planar_geometry():
    coordinates = mds_layout(_triangle())
    drawn = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    np.testing.assert_allclose(drawn, _triangle(), atol=1e-6)


def test_missing_pairs_are_filled_in_by_shortest_paths():
    distances = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
        ]
    )
    completed, connected = complete_distances(distances)
    assert connected
    assert completed[0, 2] == pytest.approx(5.0), "the only route from 0 to 2 runs through 1"
    np.testing.assert_allclose(np.diag(completed), 0.0)


def test_disconnected_networks_still_produce_a_layout():
    distances = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    completed, connected = complete_distances(distances)
    assert not connected
    assert np.all(np.isfinite(completed))
    coordinates, distortion = stress_layout(distances)
    assert coordinates.shape == (4, 2)
    assert np.all(np.isfinite(coordinates))
    assert distortion >= 0.0


def test_minimum_separation_spreads_crowded_regions_and_reports_the_cost():
    distances = np.array(
        [
            [0.0, 0.2, 8.0],
            [0.2, 0.0, 8.0],
            [8.0, 8.0, 0.0],
        ]
    )
    faithful, faithful_distortion = stress_layout(distances, min_separation=0.0)
    spread, spread_distortion = stress_layout(distances, min_separation=3.0)

    def separation(coordinates: np.ndarray) -> float:
        return float(np.linalg.norm(coordinates[0] - coordinates[1]))

    assert separation(faithful) == pytest.approx(0.2, abs=0.1)
    assert separation(spread) > 2.0
    assert spread_distortion > faithful_distortion, "spreading regions costs faithfulness"


def test_layout_is_deterministic():
    first, _ = stress_layout(_triangle(), min_separation=1.0)
    second, _ = stress_layout(_triangle(), min_separation=1.0)
    np.testing.assert_allclose(first, second)
