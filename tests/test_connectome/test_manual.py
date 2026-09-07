"""Tests for hand-entered connectomes."""

import numpy as np

from diaxcondel.connectome.manual import (
    format_matrix,
    manual_connectivity,
    manual_distances,
    parse_codes,
    parse_matrix,
)
from diaxcondel.connectome.regions import RegionSet


def test_parse_matrix_reads_json_and_plain_rows():
    expected = np.array([[0.0, 5.0], [5.0, 0.0]])
    np.testing.assert_allclose(parse_matrix("[[0, 5], [5, 0]]"), expected)
    np.testing.assert_allclose(parse_matrix("0 5\n5 0"), expected)
    np.testing.assert_allclose(parse_matrix("0, 5; 5, 0"), expected)
    np.testing.assert_allclose(parse_matrix("", n=3), np.zeros((3, 3)))


def test_format_matrix_round_trips():
    matrix = np.array([[0.0, 1.2345], [1.2345, 0.0]])
    np.testing.assert_allclose(parse_matrix(format_matrix(matrix)), matrix, atol=1e-4)


def test_parse_codes_generates_and_validates():
    assert parse_codes("FL, PL, OL") == ("FL", "PL", "OL")
    assert parse_codes("", n=3) == ("R1", "R2", "R3")


def test_manual_distances_symmetrise_and_clear_the_diagonal():
    provider = manual_distances("[[3, 8], [0, 2]]", "A, B")
    matrix = provider.matrix()
    np.testing.assert_allclose(matrix, np.array([[0.0, 4.0], [4.0, 0.0]]))
    assert provider.regions.codes == ("A", "B")
    assert provider.metadata["source"] == "manual"


def test_manual_connectivity_matches_the_given_regions():
    regions = RegionSet.from_codes(["A", "B", "C"])
    provider = manual_connectivity(regions, "[[0, 1, 0], [0.5, 0, 0], [0, 0, 0]]")
    matrix = provider.matrix()
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 0.5, "weights are directed unless symmetrised"


def test_manual_connectivity_can_symmetrise_and_clips_negatives():
    regions = RegionSet.from_codes(["A", "B"])
    provider = manual_connectivity(regions, "[[0, 1], [0, 0]]", symmetrize=True)
    np.testing.assert_allclose(provider.matrix(), np.array([[0.0, 0.5], [0.5, 0.0]]))
    clipped = manual_connectivity(regions, np.array([[0.0, -1.0], [-1.0, 0.0]]))
    assert clipped.matrix().min() == 0.0
