"""
Connectome bundles.

A :class:`Connectome` ties together the three things that must agree for a
model to be well-posed: a :class:`~diaxcondel.connectome.regions.RegionSet`,
a :class:`~diaxcondel.connectome.distance.DistanceProvider`, and a
:class:`~diaxcondel.connectome.weights.ConnectivityProvider`. Bundling them
makes region-set consistency a construction-time guarantee instead of an
error surfaced deep inside :func:`~diaxcondel.model.build.build_lag_tensor`.

Bundles are the unit that the component catalog and the playground hand
around: every ``connectome`` catalog entry returns one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from diaxcondel._typing import FloatArray

from ._meta import dataclass_state, freeze_metadata, metadata_of, restore_dataclass_state
from .distance import DistanceProvider, ManualDistances, RelayedDistances
from .regions import Region, RegionSet
from .weights import ConnectivityProvider, ManualConnectivity


@dataclass(frozen=True, slots=True)
class Connectome:
    """
    A region set with matching distance and connectivity providers.

    Parameters
    ----------
    regions : RegionSet
        Region ordering shared by both providers.
    distances : DistanceProvider
        Inter-region distances in centimetres.
    weights : ConnectivityProvider
        Dimensionless coupling weights ``gamma``; entry ``[i, j]`` is the
        weight from source ``j`` onto target ``i``.
    name : str
        Short identifier used in figures and provenance records.
    metadata : Mapping[str, object], optional
        Free-form provenance (atlas version, cohort, aggregation rule, ...).
        Never consumed by the model; carried through for reporting.

    Raises
    ------
    ValueError
        If either provider indexes a different region set.
    """

    regions: RegionSet
    distances: DistanceProvider
    weights: ConnectivityProvider
    name: str = "connectome"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        codes = self.regions.codes
        if self.distances.regions.codes != codes:
            raise ValueError("distance provider indexes a different RegionSet than the bundle")
        if self.weights.regions.codes != codes:
            raise ValueError("weight provider indexes a different RegionSet than the bundle")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state; the read-only metadata view becomes a dict."""
        return dataclass_state(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, re-freezing the metadata mapping."""
        restore_dataclass_state(self, state)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_sources(
        cls,
        distances: DistanceProvider,
        weights: ConnectivityProvider,
        *,
        name: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """
        Bundle a distance source and a connectivity source that were chosen separately.

        Parameters
        ----------
        distances : DistanceProvider
            Distances (cm); its region set defines the model's regions.
        weights : ConnectivityProvider
            Coupling weights over the same regions.
        name : str
            Bundle name; defaults to ``"<distances> + <weights>"``.
        metadata : Mapping[str, Any], optional
            Extra provenance. The metadata of both sources is merged in
            under ``"distance_source"`` and ``"weight_source"``.

        Returns
        -------
        Connectome
            Validated bundle.

        Raises
        ------
        ValueError
            If the two sources index different region sets — with the codes
            that differ, because mixing an atlas connectome with hand-tuned
            weights for other regions is the easiest mistake to make here.
        """
        distance_codes = distances.regions.codes
        weight_codes = weights.regions.codes
        if distance_codes != weight_codes:
            only_distance = [code for code in distance_codes if code not in set(weight_codes)]
            only_weights = [code for code in weight_codes if code not in set(distance_codes)]
            raise ValueError(
                "the distance source and the connectivity source describe different regions "
                f"({len(distance_codes)} vs {len(weight_codes)}); "
                f"only in distances: {only_distance[:5]}; only in connectivity: {only_weights[:5]}"
            )
        distance_name = getattr(distances, "name", "") or type(distances).__name__
        weight_name = getattr(weights, "name", "") or type(weights).__name__
        return cls(
            regions=distances.regions,
            distances=distances,
            weights=weights,
            name=name or f"{distance_name} + {weight_name}",
            metadata={
                "distance_source": dict(metadata_of(distances)),
                "weight_source": dict(metadata_of(weights)),
                **dict(metadata or {}),
            },
        )

    @classmethod
    def from_matrices(
        cls,
        regions: RegionSet,
        distances_cm: FloatArray,
        weights: FloatArray,
        *,
        name: str = "connectome",
        metadata: Mapping[str, object] | None = None,
    ) -> Self:
        """
        Build a bundle from plain matrices.

        Parameters
        ----------
        regions : RegionSet
            Region ordering for both matrices.
        distances_cm : FloatArray of shape (N, N)
            Symmetric distances in centimetres, zero diagonal.
        weights : FloatArray of shape (N, N)
            Non-negative weights, zero diagonal.
        name : str
            Bundle name.
        metadata : Mapping[str, object], optional
            Provenance metadata.

        Returns
        -------
        Connectome
            Validated bundle backed by
            :class:`~diaxcondel.connectome.distance.ManualDistances` and
            :class:`~diaxcondel.connectome.weights.ManualConnectivity`.
        """
        return cls(
            regions=regions,
            distances=ManualDistances(regions=regions, matrix_cm=distances_cm),
            weights=ManualConnectivity(regions=regions, weights=weights),
            name=name,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_regions(self) -> int:
        """Number of regions in the bundle."""
        return len(self.regions)

    @property
    def codes(self) -> tuple[str, ...]:
        """Region codes, in matrix order."""
        return self.regions.codes

    def distance_matrix(self) -> FloatArray:
        """Return the ``(N, N)`` distance matrix in centimetres."""
        return self.distances.matrix()

    def weight_matrix(self) -> FloatArray:
        """Return the ``(N, N)`` connectivity matrix."""
        return self.weights.matrix()

    @property
    def density(self) -> float:
        """
        Fraction of off-diagonal pairs carrying a non-zero weight.

        Returns
        -------
        float
            Value in ``[0, 1]``. ``1.0`` for a fully connected network.
        """
        n = self.n_regions
        if n < 2:
            return 0.0
        offdiag = ~np.eye(n, dtype=bool)
        return float(np.count_nonzero(self.weight_matrix()[offdiag]) / np.count_nonzero(offdiag))

    def describe(self) -> dict[str, object]:
        """
        Return a small JSON-serialisable summary for reports and dashboards.

        Returns
        -------
        dict
            Region count, connection density, distance range (cm), weight
            range, and any provenance metadata.
        """
        d = self.distance_matrix()
        w = self.weight_matrix()
        offdiag = ~np.eye(self.n_regions, dtype=bool) if self.n_regions else np.zeros((0, 0), dtype=bool)
        connected = offdiag & (w > 0)
        return {
            "name": self.name,
            "n_regions": self.n_regions,
            "density": self.density,
            "distance_cm_min": float(d[connected].min()) if np.any(connected) else 0.0,
            "distance_cm_max": float(d[connected].max()) if np.any(connected) else 0.0,
            "weight_max": float(w.max()) if w.size else 0.0,
            "metadata": dict(self.metadata or {}),
        }

    # ------------------------------------------------------------------
    # Derived bundles
    # ------------------------------------------------------------------

    def select(self, codes: Sequence[str], *, name: str | None = None) -> Connectome:
        """
        Return a sub-connectome restricted to ``codes``.

        Parameters
        ----------
        codes : sequence of str
            Region codes to keep, in the desired output order.
        name : str, optional
            Name for the derived bundle. Defaults to ``"<name>[subset]"``.

        Returns
        -------
        Connectome
            Bundle over the selected regions only.

        Raises
        ------
        KeyError
            If a requested code is not part of this bundle.
        ValueError
            If ``codes`` is empty or contains duplicates.
        """
        codes = list(codes)
        if not codes:
            raise ValueError("codes must be non-empty")
        if len(set(codes)) != len(codes):
            raise ValueError("codes must not contain duplicates")
        idx = np.array([self.regions.index(code) for code in codes], dtype=int)
        regions = RegionSet(tuple(self.regions[int(i)] for i in idx))
        metadata = dict(self.metadata or {})
        metadata["derived_from"] = self.name
        metadata["selection"] = list(codes)
        bundle_name = name or f"{self.name}[subset]"
        mixture = getattr(self.distances, "mixture", None)
        distances = ManualDistances(
            regions=regions,
            matrix_cm=self.distance_matrix()[np.ix_(idx, idx)],
            name=getattr(self.distances, "name", "distances"),
            metadata=dict(source_metadata(self.distances)),
            mixture=mixture.select([int(i) for i in idx]) if mixture is not None else None,
        )
        weights = ManualConnectivity(
            regions=regions,
            weights=self.weight_matrix()[np.ix_(idx, idx)],
            name=getattr(self.weights, "name", "connectivity"),
            metadata=dict(source_metadata(self.weights)),
        )
        return Connectome(
            regions=regions,
            distances=distances,
            weights=weights,
            name=bundle_name,
            metadata=metadata,
        )

    def relayed(self, relay_code: str) -> Connectome:
        """
        Return a bundle whose distances route through a relay region.

        Parameters
        ----------
        relay_code : str
            Region code acting as the relay station (e.g. the thalamus).

        Returns
        -------
        Connectome
            Bundle with :class:`~diaxcondel.connectome.distance.RelayedDistances`.
            Building a model from it requires a two-leg delay kernel; the
            builders in :mod:`diaxcondel.experiment` handle that automatically.
        """
        return Connectome(
            regions=self.regions,
            distances=RelayedDistances(base=self.distances, relay_code=relay_code),
            weights=self.weights,
            name=f"{self.name}[relay={relay_code}]",
            metadata={**dict(self.metadata or {}), "relay_code": relay_code},
        )

    def rescale_weights(self, factor: float, *, name: str | None = None) -> Connectome:
        """
        Return a bundle with all weights multiplied by ``factor``.

        Parameters
        ----------
        factor : float
            Positive multiplicative factor applied to ``gamma``. Values above
            one increase effective coupling and may trigger the stationarity
            shrink in :func:`~diaxcondel.model.build.build_linear_var`.
        name : str, optional
            Name for the derived bundle.

        Returns
        -------
        Connectome
            Bundle with rescaled weights and unchanged distances.
        """
        if not np.isfinite(factor) or factor < 0:
            raise ValueError("factor must be finite and non-negative")
        if factor == 1.0:
            return self
        metadata = {**dict(self.metadata or {}), "weight_scale": factor}
        return Connectome(
            regions=self.regions,
            distances=self.distances,
            weights=ManualConnectivity(regions=self.regions, weights=self.weight_matrix() * factor),
            name=name or self.name,
            metadata=metadata,
        )


def source_metadata(source: object) -> Mapping[str, Any]:
    """
    Return the provenance metadata of a distance or connectivity source.

    Parameters
    ----------
    source : object
        Any provider. Wrappers such as
        :class:`~diaxcondel.connectome.distance.RelayedDistances` forward to
        the provider they wrap.

    Returns
    -------
    Mapping[str, Any]
        Metadata mapping; empty when the source records none.
    """
    return metadata_of(source)


def region_metadata(region: Region, key: str, default: object = None) -> object:
    """
    Read one metadata entry from a region, tolerating regions without metadata.

    Parameters
    ----------
    region : Region
        The region to inspect.
    key : str
        Metadata key.
    default : object
        Value returned when the region carries no metadata or lacks the key.

    Returns
    -------
    object
        The metadata value, or ``default``.
    """
    if region.metadata is None:
        return default
    return region.metadata.get(key, default)
