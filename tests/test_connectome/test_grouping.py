"""Tests for region grouping and matrix aggregation."""

import numpy as np
import pytest

from diaxcondel.connectome.bundle import Connectome
from diaxcondel.connectome.grouping import (
    aggregate_matrix,
    ancestor_labels,
    group_connectome,
    metadata_labels,
)
from diaxcondel.connectome.regions import Region, RegionSet


def test_aggregate_matrix_sum_and_mean():
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    groups = [0, 0, 1, 1]
    summed = aggregate_matrix(matrix, groups, 2, how="sum")
    assert summed[0, 0] == matrix[:2, :2].sum()
    averaged = aggregate_matrix(matrix, groups, 2, how="mean")
    assert averaged[0, 1] == pytest.approx(matrix[:2, 2:].mean())


def test_aggregate_matrix_ignores_zeros_as_missing():
    matrix = np.array(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 6.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
        ]
    )
    groups = [0, 0, 1, 1]
    naive = aggregate_matrix(matrix, groups, 2, how="mean")
    masked = aggregate_matrix(matrix, groups, 2, how="mean", ignore_zeros=True)
    assert naive[0, 1] == pytest.approx(10.0 / 4)
    assert masked[0, 1] == pytest.approx(5.0)


def test_aggregate_matrix_weighted_mean_follows_strong_pairs():
    matrix = np.array(
        [
            [0.0, 0.0, 2.0, 10.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ]
    )
    weights = np.array(
        [
            [0.0, 0.0, 9.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    groups = [0, 0, 1, 1]
    weighted = aggregate_matrix(matrix, groups, 2, how="mean", ignore_zeros=True, pair_weights=weights)
    assert weighted[0, 1] == pytest.approx((2 * 9 + 10 * 1) / 10)


def _four_region_bundle() -> Connectome:
    regions = RegionSet(
        (
            Region(
                code="a1",
                metadata={
                    "ancestors": ("brain", "left", "front"),
                    "hemisphere": "left",
                },
            ),
            Region(
                code="a2",
                metadata={
                    "ancestors": ("brain", "left", "front"),
                    "hemisphere": "left",
                },
            ),
            Region(
                code="b1",
                metadata={
                    "ancestors": ("brain", "right", "back"),
                    "hemisphere": "right",
                },
            ),
            Region(
                code="b2",
                metadata={"ancestors": ("brain", "right"), "hemisphere": "right"},
            ),
        )
    )
    distances = np.array(
        [
            [0.0, 1.0, 6.0, 8.0],
            [1.0, 0.0, 0.0, 4.0],
            [6.0, 0.0, 0.0, 2.0],
            [8.0, 4.0, 2.0, 0.0],
        ]
    )
    weights = np.array(
        [
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    return Connectome.from_matrices(regions, distances, weights, name="fake")


def test_group_connectome_merges_regions_and_records_members():
    bundle = _four_region_bundle()
    grouped = group_connectome(bundle, {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}, weight_how="sum")
    assert grouped.codes == ("A", "B")
    assert grouped.regions[0].metadata["members"] == ("a1", "a2")
    # Within-group connections are dropped; the model has no self-loops.
    assert grouped.distance_matrix()[0, 0] == 0.0
    assert grouped.weight_matrix()[0, 0] == 0.0
    # Weights sum over the four cross pairs (one of which is zero).
    assert grouped.weight_matrix()[0, 1] == pytest.approx(3.0 + 1.0 + 0.0 + 1.0)
    # Distances are a weight-weighted mean over connected member pairs only.
    expected = (6.0 * 3.0 + 8.0 * 1.0 + 4.0 * 1.0) / (3.0 + 1.0 + 1.0)
    assert grouped.distance_matrix()[0, 1] == pytest.approx(expected)


def test_group_connectome_min_rule_and_sequence_labels():
    bundle = _four_region_bundle()
    grouped = group_connectome(bundle, ["A", "A", "B", "B"], distance_how="min")
    assert grouped.distance_matrix()[0, 1] == pytest.approx(4.0)


def test_grouping_drops_group_pairs_without_measured_distance():
    regions = RegionSet.from_codes(["x", "y"])
    bundle = Connectome.from_matrices(regions, np.zeros((2, 2)), np.zeros((2, 2)))
    grouped = group_connectome(bundle, ["G", "G"])
    assert grouped.n_regions == 1
    assert grouped.weight_matrix().sum() == 0.0


def test_ancestor_labels_selects_by_depth_and_clamps():
    bundle = _four_region_bundle()
    labels = ancestor_labels(bundle.regions, level=2)
    assert labels == {"a1": "front", "a2": "front", "b1": "back", "b2": "right"}
    assert set(ancestor_labels(bundle.regions, level=0).values()) == {"brain"}
    split = ancestor_labels(bundle.regions, level=1, split_hemispheres=True)
    assert split["a1"] == "left left"


def test_labels_fall_back_to_region_code_without_metadata():
    regions = RegionSet.from_codes(["p", "q"])
    assert ancestor_labels(regions, level=2) == {"p": "p", "q": "q"}
    assert metadata_labels(regions, "hemisphere") == {"p": "p", "q": "q"}


def test_metadata_labels_group_by_field():
    bundle = _four_region_bundle()
    assert metadata_labels(bundle.regions, "hemisphere") == {
        "a1": "left",
        "a2": "left",
        "b1": "right",
        "b2": "right",
    }
