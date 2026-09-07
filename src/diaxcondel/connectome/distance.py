"""
Distance providers.

A :class:`DistanceProvider` exposes an ``(N, N)`` matrix of inter-region
distances (in centimetres) over a fixed :class:`RegionSet`. The protocol is
deliberately minimal so that anatomical, manual, synthetic, and composed
sources are interchangeable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from diaxcondel._typing import FloatArray, IntArray

from ._meta import dataclass_state, freeze_metadata, restore_dataclass_state
from .regions import RegionSet


@runtime_checkable
class DistanceProvider(Protocol):
    """
    Protocol for any source of inter-region distances.

    Implementations must be deterministic: repeated calls to :meth:`matrix`
    on an unchanged provider must return the same array (numerically
    identical, not merely equal-up-to-noise). Caching is allowed.
    """

    @property
    def regions(self) -> RegionSet:
        """Return the region set whose order indexes the matrix."""
        ...

    def matrix(self) -> FloatArray:
        """Return the ``(N, N)`` distance matrix in centimetres.

        Returns
        -------
        FloatArray of shape (N, N)
            Symmetric matrix of distances. Diagonal entries are zero.
        """
        ...


@dataclass(frozen=True, slots=True)
class LengthMixture:
    """
    The individual connection lengths behind a merged connection.

    When several fine-grained regions are merged into one node, the pathway
    between two merged nodes is not a single fibre bundle of one length: it
    is a *collection* of bundles with different lengths, each carrying a
    different number of fibres. Collapsing that to a mean length before
    computing delays throws away exactly the spread that delays are made of.

    Keeping the members lets the delay kernel be evaluated per member length
    and mixed afterwards, weighted by how many fibres each member carries.
    That is a mixture of delay distributions — it does not alter conduction
    speed, axon diameter, or any other physical quantity; it only says how
    much of the signal travels each distance.

    Parameters
    ----------
    rows, cols : IntArray of shape (K,)
        Indices of the merged region pair each member belongs to. Only the
        upper triangle is stored; the mixture is symmetric.
    lengths_cm : FloatArray of shape (K,)
        Member connection lengths in centimetres; must be positive.
    weights : FloatArray of shape (K,)
        Non-negative mixture weights, typically streamline counts.
    """

    rows: IntArray
    cols: IntArray
    lengths_cm: FloatArray
    weights: FloatArray

    def __post_init__(self) -> None:  # noqa: D105
        rows = np.asarray(self.rows, dtype=int)
        cols = np.asarray(self.cols, dtype=int)
        lengths = np.asarray(self.lengths_cm, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if not (rows.shape == cols.shape == lengths.shape == weights.shape) or rows.ndim != 1:
            raise ValueError("mixture arrays must be one-dimensional and the same length")
        if lengths.size and np.any(lengths <= 0):
            raise ValueError("member lengths must be positive")
        if weights.size and np.any(weights < 0):
            raise ValueError("mixture weights must be non-negative")
        for name, array in (("rows", rows), ("cols", cols), ("lengths_cm", lengths), ("weights", weights)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def __len__(self) -> int:  # noqa: D105
        return int(self.rows.size)

    def select(self, keep: Sequence[int]) -> LengthMixture:
        """
        Return the mixture restricted to a subset of regions, re-indexed.

        Parameters
        ----------
        keep : sequence of int
            Region indices to keep, in the desired output order.

        Returns
        -------
        LengthMixture
            Members whose both endpoints survive, with updated indices.
        """
        keep = list(keep)
        position = {old: new for new, old in enumerate(keep)}
        mask = np.array(
            [row in position and col in position for row, col in zip(self.rows, self.cols, strict=True)],
            dtype=bool,
        )
        if not mask.any():
            return LengthMixture(np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
        return LengthMixture(
            rows=np.array([position[int(r)] for r in self.rows[mask]], dtype=int),
            cols=np.array([position[int(c)] for c in self.cols[mask]], dtype=int),
            lengths_cm=self.lengths_cm[mask],
            weights=self.weights[mask],
        )

    def effective_lengths(self, n_regions: int) -> FloatArray:
        """
        Return the weighted mean member length per merged pair.

        Parameters
        ----------
        n_regions : int
            Number of merged regions.

        Returns
        -------
        FloatArray of shape (N, N)
            The length whose delay matches the mixture's mean delay, which
            is generally *not* the plain geometric mean: a pathway's delay is
            set by where its fibres actually are, and the members carrying
            the most fibres can be much longer (or shorter) than average.
            Zero where a pair has no members.
        """
        total = np.zeros((n_regions, n_regions))
        norm = np.zeros((n_regions, n_regions))
        np.add.at(total, (self.rows, self.cols), self.lengths_cm * self.weights)
        np.add.at(norm, (self.rows, self.cols), self.weights)
        out = np.zeros_like(total)
        np.divide(total, norm, out=out, where=norm > 0)
        return out + out.T

    def describe(self) -> dict[str, float]:
        """Return member count and the spread of member lengths."""
        if not len(self):
            return {}
        return {
            "members": float(len(self)),
            "shortest_cm": float(self.lengths_cm.min()),
            "longest_cm": float(self.lengths_cm.max()),
        }


@dataclass(frozen=True, slots=True)
class ManualDistances:
    """Distance provider backed by an explicit numpy array.

    Every materialised distance source ends up here — hand-entered numbers,
    atlas streamline lengths, synthetic draws — which is why it also carries
    the provenance of those numbers.

    Parameters
    ----------
    regions : RegionSet
        Region ordering corresponding to the matrix axes.
    matrix_cm : FloatArray of shape (N, N)
        Symmetric, non-negative distance matrix. Diagonal must be zero.
    name : str
        Short identifier for figures and reports.
    metadata : Mapping[str, Any], optional
        Free-form provenance (atlas query, seed, aggregation rule, ...).
    mixture : LengthMixture, optional
        For merged regions, the individual member lengths behind each pair.
        Model builders use it to mix delay distributions instead of using a
        single representative length; see :class:`LengthMixture`.

    Raises
    ------
    ValueError
        If the matrix is malformed.
    """

    regions: RegionSet
    matrix_cm: FloatArray
    name: str = "distances"
    metadata: Mapping[str, Any] | None = None
    mixture: LengthMixture | None = None

    def __post_init__(self) -> None:  # noqa: D105
        n = len(self.regions)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        m = np.array(self.matrix_cm, dtype=float, copy=True)
        if m.shape != (n, n):
            raise ValueError(f"matrix shape {m.shape} != ({n}, {n})")
        if not np.all(np.isfinite(m)):
            raise ValueError("distance matrix must contain only finite values")
        if not np.allclose(m, m.T):
            raise ValueError("distance matrix must be symmetric")
        if np.any(m < 0):
            raise ValueError("distances must be non-negative")
        if not np.allclose(np.diag(m), 0.0):
            raise ValueError("distance diagonal must be zero")
        m.setflags(write=False)
        object.__setattr__(self, "matrix_cm", m)

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state (the read-only metadata view becomes a dict)."""
        return dataclass_state(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, re-freezing the metadata mapping."""
        restore_dataclass_state(self, state)

    def matrix(self) -> FloatArray:  # noqa: D102
        return self.matrix_cm


@dataclass(frozen=True, slots=True)
class RelayedDistances:
    """Distances composed via a relay region (e.g. cortico-thalamo-cortical).

    For any pair ``(u, w)`` with ``u != relay`` and ``w != relay``, the
    relayed distance is ``d(u, relay) + d(relay, w)``. Pairs involving the
    relay itself use the direct distance. The resulting matrix is symmetric
    and has zero diagonal.

    Parameters
    ----------
    base : DistanceProvider
        Underlying distance source.
    relay_code : str
        Code of the region acting as the relay station.

    Notes
    -----
    The kernel module must be told that this provider returns a *summed*
    distance (cf. :class:`diaxcondel.kernels.diameter.TwoLegDelayKernel`),
    because the delay distribution for two-leg distances is the
    convolution of two inverse-GEVs, not a single inverse-GEV.
    """

    base: DistanceProvider
    relay_code: str

    def __post_init__(self) -> None:  # noqa: D105
        # Validate that the relay exists in the base region set.
        _ = self.base.regions.index(self.relay_code)

    @property
    def regions(self) -> RegionSet:  # noqa: D102
        return self.base.regions

    def matrix(self) -> FloatArray:  # noqa: D102
        d = self.base.matrix().copy()
        r = self.base.regions.index(self.relay_code)
        # Replace every (i, j) with (i, r) + (r, j) for i, j != r.
        n = d.shape[0]
        out = np.zeros_like(d)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if i == r or j == r:
                    out[i, j] = d[i, j]
                else:
                    out[i, j] = d[i, r] + d[r, j]
        return out
