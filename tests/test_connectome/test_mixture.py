"""
Tests for connection-length mixtures.

Merging regions turns one pathway into many, of different lengths. These
tests pin down what the model does with that: the mixture keeps the real
lengths, the delay kernel is evaluated at each of them, and the fibre counts
weight how much each contributes — without touching conduction speed.
"""

import numpy as np
import pytest

from diaxcondel.catalog import build as build_component
from diaxcondel.connectome.bundle import Connectome
from diaxcondel.connectome.distance import LengthMixture
from diaxcondel.connectome.grouping import build_length_mixture, group_connectome
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.build import build_lag_tensor
from diaxcondel.model.params import SimulationParams


def _four_region_bundle() -> Connectome:
    """Two groups of two regions; the long member carries most of the fibres."""
    regions = RegionSet.from_codes(["a1", "a2", "b1", "b2"])
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 16.0],
            [1.0, 0.0, 6.0, 0.0],
            [4.0, 6.0, 0.0, 1.0],
            [16.0, 0.0, 1.0, 0.0],
        ]
    )
    weights = np.array(
        [
            [0.0, 1.0, 1.0, 20.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [20.0, 0.0, 1.0, 0.0],
        ]
    )
    return Connectome.from_matrices(regions, distances, weights, name="fine")


def test_mixture_collects_cross_group_members_only():
    bundle = _four_region_bundle()
    mixture = build_length_mixture(bundle.distance_matrix(), bundle.weight_matrix(), [0, 0, 1, 1], 2)
    assert len(mixture) == 3  # (a1,b1), (a1,b2), (a2,b1); (a2,b2) has no connection
    assert set(np.round(mixture.lengths_cm, 6)) == {4.0, 16.0, 6.0}
    assert set(zip(mixture.rows.tolist(), mixture.cols.tolist(), strict=True)) == {(0, 1)}


def test_effective_length_follows_the_fibre_counts():
    bundle = _four_region_bundle()
    grouped = group_connectome(bundle, ["A", "A", "B", "B"])
    mixture = grouped.distances.mixture
    assert mixture is not None

    geometric = grouped.distance_matrix()[0, 1]
    effective = mixture.effective_lengths(2)[0, 1]
    assert geometric == pytest.approx((4.0 + 16.0 + 6.0) / 3)
    # The 16 cm member carries twenty times the fibres, so it dominates.
    assert effective == pytest.approx((4.0 * 1 + 16.0 * 20 + 6.0 * 1) / 22)
    assert effective > geometric


def test_mixture_survives_selection_and_pickling():
    bundle = _four_region_bundle()
    grouped = group_connectome(bundle, ["A", "A", "B", "B"])
    import pickle

    restored = pickle.loads(pickle.dumps(grouped))
    assert restored.distances.mixture is not None
    assert len(restored.distances.mixture) == 3

    subset = grouped.select(["A"])
    assert subset.distances.mixture is not None
    assert len(subset.distances.mixture) == 0, "no cross-group members survive a single-group subset"


def test_mixed_delays_differ_from_a_single_representative_length():
    bundle = _four_region_bundle()
    grouped = group_connectome(bundle, ["A", "A", "B", "B"])
    kernel = build_component("kernel", "hursh", diameter=build_component("diameter", "gev"))
    params = SimulationParams(sample_rate_hz=1000, sim_seconds=1.0, burnin_seconds=0.5, n_lags=400)

    mixed = build_lag_tensor(grouped.distances, grouped.weights, kernel, params)
    single = build_lag_tensor(grouped.distances, grouped.weights, kernel, params, use_length_mixture=False)

    lags_ms = np.arange(1, params.n_lags + 1)

    def mean_delay(phi: np.ndarray) -> float:
        column = phi[:, 0, 1]
        return float(np.dot(lags_ms, column) / column.sum())

    # The fibre-heavy 16 cm member pushes the mixed delay well beyond the one
    # implied by the mean length.
    assert mean_delay(mixed) > 1.5 * mean_delay(single)
    # Total coupling is a property of the weights, not of how delays are spread —
    # up to the mass that the longest members push past the lag horizon.
    assert mixed[:, 0, 1].sum() <= single[:, 0, 1].sum()
    assert mixed[:, 0, 1].sum() == pytest.approx(single[:, 0, 1].sum(), rel=0.02)


def test_uniform_member_weighting_recovers_the_plain_average():
    bundle = _four_region_bundle()
    grouped = group_connectome(bundle, ["A", "A", "B", "B"])
    mixture = grouped.distances.mixture
    assert mixture is not None
    uniform = LengthMixture(
        rows=mixture.rows,
        cols=mixture.cols,
        lengths_cm=mixture.lengths_cm,
        weights=np.ones_like(mixture.weights),
    )
    assert uniform.effective_lengths(2)[0, 1] == pytest.approx(grouped.distance_matrix()[0, 1])
