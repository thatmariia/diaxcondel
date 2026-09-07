"""
siibra-backed atlas connectomes.

Pulls region-to-region **streamline lengths** (converted to centimetres and
used as conduction distances) and connectivity **weights** (streamline
counts, resting-state functional connectivity, or a uniform placeholder)
from the EBRAINS/siibra connectivity datasets.

Distances and weights are separate functions on purpose: a model may take its
geometry from one source and its coupling from another. What ties them
together is the :class:`AtlasQuery`: the parcellation, cohort, subject, and
the filtering and merging applied to the region list. :func:`siibra_weights`
replays the query recorded by :func:`siibra_distances`, so the two always
describe the same regions in the same order.

This module is part of the optional ``atlas`` extra; importing it without
``siibra`` installed raises :class:`ImportError` immediately. Import it
lazily (inside a function) if your code must run without the extra.

Data model
----------
For a parcellation (e.g. Julich-Brain v3.1) siibra exposes one *compound
feature* per cohort and modality, holding one matrix per subject. We select a
subject or aggregate across subjects (mean, median, or trimmed mean),
symmetrise, convert millimetres to centimetres, and zero the diagonal. The
VAR model carries no self-connections through white matter.

A zero entry in a tractography matrix means "no streamlines found", not "zero
distance". Weights are therefore masked to the support of the length matrix,
and the pruning that follows is recorded in the provider metadata.

Cost and caching
----------------
The first query downloads and aggregates a few hundred matrices (tens of
seconds). Results are cached on disk by :mod:`diaxcondel._cache`, keyed by
the full query, so repeated calls (including from the playground) are
instant. Clear it with :func:`diaxcondel._cache.clear_cache`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np
from scipy import stats

if find_spec("siibra") is None:  # pragma: no cover - exercised only without the extra
    raise ImportError("siibra is required for atlas loaders. Install with `poetry install --with atlas`.")

from diaxcondel._cache import load_or_compute
from diaxcondel._typing import FloatArray

from ..bundle import Connectome, source_metadata
from ..distance import DistanceProvider, LengthMixture, ManualDistances
from ..grouping import (
    aggregate_matrix,
    ancestor_labels,
    build_length_mixture,
    node_labels,
    select_codes,
)
from ..regions import Region, RegionSet
from ..weights import ManualConnectivity

ConnectivityKind = Literal["streamline_lengths", "streamline_counts", "functional"]
WeightSource = Literal["streamline_counts", "functional", "uniform"]
SubjectAggregation = Literal["mean", "median", "trimmed_mean"]
WeightNormalization = Literal["none", "max", "mean"]
MemberWeighting = Literal["streamline_counts", "uniform"]

DEFAULT_PARCELLATION = "julich 3.1"
DEFAULT_COHORT = "HCP"

#: Parcellations known to carry Domhof et al. (2022) connectivity data.
#: Any siibra parcellation specification is accepted by the loaders; this
#: tuple only seeds menus and documentation.
KNOWN_PARCELLATIONS: tuple[str, ...] = ("julich 3.1", "julich 3.0.3", "julich 2.9")

_FEATURE_CLASS_NAMES: dict[str, str] = {
    "streamline_lengths": "StreamlineLengths",
    "streamline_counts": "StreamlineCounts",
    "functional": "FunctionalConnectivity",
}

_MM_PER_CM = 10.0

#: Metadata key under which a distance provider records the query that made it.
QUERY_KEY = "atlas_query"


# ---------------------------------------------------------------------------
# Query description
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AtlasQuery:
    """
    Everything that determines *which regions* an atlas model has.

    Two providers built from equal queries describe the same regions in the
    same order, which is what lets distances and weights be chosen
    independently and still fit together.

    Attributes
    ----------
    parcellation : str
        siibra parcellation specification, e.g. ``"julich 3.1"``.
    cohort : str
        Subject cohort of the connectivity dataset (``"HCP"``, ``"1000BRAINS"``).
    subject : str
        Single subject id (e.g. ``"000"``), or empty to aggregate the cohort.
    subject_aggregation : str
        How subject matrices are combined when ``subject`` is empty.
    trim_fraction : float
        Tail fraction removed per subject for ``"trimmed_mean"``.
    nodes : tuple of str
        The regions to model, named individually: ``("frontal lobe",
        "parietal lobe", "thalamus")``. Each becomes one node unless it
        carries its own depth (``"frontal lobe:1"`` splits it one level
        further). Anything outside the selection is dropped. When this is
        set it defines the model's regions, and ``merge_level`` is unused.
    merge_level : int or None
        Depth of the atlas hierarchy at which *every* region is merged, used
        only when ``nodes`` is empty; ``None`` keeps the native
        parcellation.
    split_hemispheres : bool
        Keep left and right groups separate when merging.
    keep_parts : tuple of str
        Keep only regions belonging to these parts of the brain (e.g.
        ``("cerebral cortex", "thalamus")``); empty keeps everything.
    drop_parts : tuple of str
        Drop regions belonging to these parts, applied after ``keep_parts``.
    member_weighting : str
        How much each member connection of a merged pathway contributes to
        its delay distribution: in proportion to how many streamlines it
        carries, or equally.
    drop_isolated : bool
        Drop regions left without a single measured connection.
    """

    parcellation: str = DEFAULT_PARCELLATION
    cohort: str = DEFAULT_COHORT
    subject: str = ""
    subject_aggregation: SubjectAggregation = "mean"
    trim_fraction: float = 0.1
    nodes: tuple[str, ...] = ()
    merge_level: int | None = 3
    split_hemispheres: bool = False
    keep_parts: tuple[str, ...] = ()
    drop_parts: tuple[str, ...] = ()
    member_weighting: MemberWeighting = "streamline_counts"
    drop_isolated: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return the query as a plain JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: Any) -> AtlasQuery:
        """
        Recover the query recorded in a provider's metadata.

        Parameters
        ----------
        metadata : Mapping
            Metadata of a provider built by :func:`siibra_distances`.

        Returns
        -------
        AtlasQuery
            The query that produced it.

        Raises
        ------
        ValueError
            If the metadata carries no atlas query, for example when the
            distances are synthetic or hand-entered, which cannot be paired
            with atlas connectivity.
        """
        stored = dict(metadata or {}).get(QUERY_KEY)
        if stored:
            tuple_fields = {"keep_parts", "drop_parts", "nodes"}
            stored = {
                key: tuple(value) if key in tuple_fields and value is not None else value
                for key, value in dict(stored).items()
            }
        if not stored:
            raise ValueError(
                "these regions do not come from a siibra atlas, so atlas connectivity cannot be "
                "matched to them; use the atlas distance source, or a connectivity rule that works "
                "for any regions (uniform, distance decay, manual)"
            )
        return cls(**dict(stored))


# ---------------------------------------------------------------------------
# siibra access
# ---------------------------------------------------------------------------


def _feature_class(kind: ConnectivityKind) -> type:
    """Return the siibra feature class for a connectivity kind."""
    from siibra.features import connectivity as siibra_connectivity

    try:
        return getattr(siibra_connectivity, _FEATURE_CLASS_NAMES[kind])
    except KeyError as exc:
        raise ValueError(f"unknown connectivity kind {kind!r}; expected one of {list(_FEATURE_CLASS_NAMES)}") from exc


def _select_feature(parcellation: str, kind: ConnectivityKind, cohort: str, paradigm: str) -> Any:
    """
    Return the siibra compound feature matching a parcellation/cohort query.

    Raises
    ------
    LookupError
        If the parcellation carries no such feature, or the cohort/paradigm
        filter leaves nothing. The message lists what *is* available.
    """
    import siibra

    parcellation_obj = siibra.parcellations.get(parcellation)
    features = list(siibra.features.get(parcellation_obj, _feature_class(kind)))
    if not features:
        raise LookupError(
            f"no {kind!r} connectivity available for parcellation {parcellation!r}; "
            f"parcellations known to carry connectivity: {list(KNOWN_PARCELLATIONS)}"
        )

    if cohort:
        matching = [f for f in features if str(getattr(f, "cohort", "")).upper() == cohort.upper()]
        if not matching:
            cohorts = sorted({str(getattr(f, "cohort", "")) for f in features})
            raise LookupError(f"cohort {cohort!r} not available for {parcellation!r}; available cohorts: {cohorts}")
        features = matching

    if paradigm:
        matching = [f for f in features if paradigm.lower() in str(getattr(f, "paradigm", "")).lower()]
        if not matching:
            paradigms = sorted({str(getattr(f, "paradigm", "")) for f in features})
            raise LookupError(f"paradigm {paradigm!r} not available for {parcellation!r}; available: {paradigms}")
        features = matching

    return features[0]


def list_cohorts(parcellation: str = DEFAULT_PARCELLATION, kind: ConnectivityKind = "streamline_lengths") -> list[str]:
    """
    List cohorts with connectivity data for a parcellation.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification, e.g. ``"julich 3.1"``.
    kind : {"streamline_lengths", "streamline_counts", "functional"}
        Connectivity modality to query.

    Returns
    -------
    list of str
        Cohort names (e.g. ``["HCP", "1000BRAINS"]``); empty if none.
    """
    import siibra

    parcellation_obj = siibra.parcellations.get(parcellation)
    features = siibra.features.get(parcellation_obj, _feature_class(kind))
    return sorted({str(getattr(f, "cohort", "")) for f in features if getattr(f, "cohort", None)})


def list_paradigms(parcellation: str = DEFAULT_PARCELLATION, cohort: str = DEFAULT_COHORT) -> list[str]:
    """
    List functional-connectivity paradigms available for a parcellation/cohort.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification.
    cohort : str
        Cohort name.

    Returns
    -------
    list of str
        Paradigm labels (e.g. resting-state runs); empty if none.
    """
    import siibra

    parcellation_obj = siibra.parcellations.get(parcellation)
    features = siibra.features.get(parcellation_obj, _feature_class("functional"))
    return sorted(
        {
            str(getattr(f, "paradigm", ""))
            for f in features
            if getattr(f, "paradigm", None) and str(getattr(f, "cohort", "")).upper() == cohort.upper()
        }
    )


def list_subjects(parcellation: str = DEFAULT_PARCELLATION, cohort: str = DEFAULT_COHORT) -> list[str]:
    """
    List the subject ids available for a parcellation and cohort.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification.
    cohort : str
        Cohort name.

    Returns
    -------
    list of str
        Subject ids, e.g. ``["000", "001", ...]``. Pick one to model a single
        participant instead of the cohort aggregate.
    """
    feature = _select_feature(parcellation, "streamline_lengths", cohort, "")
    return [str(index) for index in getattr(feature, "indices", [])]


@lru_cache(maxsize=8)
def parcellation_info(parcellation: str = DEFAULT_PARCELLATION) -> dict[str, Any]:
    """
    Return descriptive metadata and documentation links for a parcellation.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification.

    Returns
    -------
    dict
        ``name``, ``description``, ``modality``, ``urls`` (DOIs and dataset
        pages), ``publications`` (citation strings), and ``n_regions``.
        Fields the atlas does not provide come back empty.
    """
    import siibra

    obj = siibra.parcellations.get(parcellation)
    publications = []
    for entry in getattr(obj, "publications", []) or []:
        citation = entry.get("citation") if isinstance(entry, dict) else str(entry)
        if citation:
            publications.append(str(citation).strip())
    return {
        "name": str(getattr(obj, "name", parcellation)),
        "description": str(getattr(obj, "description", "") or "").strip(),
        "modality": str(getattr(obj, "modality", "") or ""),
        "urls": [str(url) for url in (getattr(obj, "urls", []) or [])],
        "publications": publications,
        "n_regions": len(list(obj)),
    }


def _element_frames(feature: Any) -> list[Any]:
    """Return the per-subject feature elements of a (possibly compound) feature."""
    elements = getattr(feature, "elements", None)
    if elements:
        return list(elements)
    return [feature]


def _subject_frames(feature: Any, subject: str) -> list[Any]:
    """Return the elements for one subject, or all of them when unspecified."""
    elements = _element_frames(feature)
    if not subject:
        return elements
    indices = [str(index) for index in getattr(feature, "indices", [])]
    for index, element in zip(indices, elements, strict=False):
        if index == subject or str(getattr(element, "subject", "")) == subject:
            return [element]
    raise LookupError(f"subject {subject!r} not in this cohort; first available: {indices[:5]} (of {len(indices)})")


def _frame_matrix(frame: Any, reference_names: list[str]) -> FloatArray:
    """Extract a matrix from a siibra data frame, aligned to ``reference_names``."""
    names = [str(r) for r in frame.index]
    if names != reference_names:
        lookup = {str(r): r for r in frame.index}
        missing = [name for name in reference_names if name not in lookup]
        if missing:
            raise ValueError(f"subject matrix is missing {len(missing)} regions, e.g. {missing[:3]}")
        order = [lookup[name] for name in reference_names]
        frame = frame.reindex(index=order, columns=order)
    return np.asarray(frame.to_numpy(), dtype=float)


def _aggregate_subjects(
    matrices: list[FloatArray],
    aggregation: SubjectAggregation,
    trim_fraction: float,
) -> tuple[FloatArray, int]:
    """
    Combine per-subject matrices into one, tolerating missing entries.

    Released connectomes occasionally carry a handful of non-finite cells for
    individual subjects. Those cells are excluded from the aggregate rather
    than poisoning it; cells missing for *every* subject become zero, which
    this model reads as "no connection".

    Returns
    -------
    matrix : FloatArray of shape (N, N)
        Aggregated matrix, guaranteed finite.
    n_nonfinite : int
        Number of non-finite input cells that were excluded.
    """
    if aggregation == "mean" or len(matrices) == 1:
        total = np.zeros_like(matrices[0], dtype=float)
        count = np.zeros_like(matrices[0], dtype=float)
        n_nonfinite = 0
        for matrix in matrices:
            finite = np.isfinite(matrix)
            n_nonfinite += int(np.count_nonzero(~finite))
            total += np.where(finite, matrix, 0.0)
            count += finite
        out = np.zeros_like(total)
        np.divide(total, count, out=out, where=count > 0)
        return out, n_nonfinite

    stacked = np.stack(matrices, axis=0)
    n_nonfinite = int(np.count_nonzero(~np.isfinite(stacked)))
    masked = np.ma.masked_invalid(stacked)
    if aggregation == "median":
        aggregated = np.ma.median(masked, axis=0)
    elif aggregation == "trimmed_mean":
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must lie in [0, 0.5)")
        aggregated = stats.mstats.trimmed_mean(masked, limits=(trim_fraction, trim_fraction), axis=0)
    else:
        raise ValueError(f"unknown subject aggregation {aggregation!r}")
    return np.asarray(np.ma.filled(aggregated, 0.0), dtype=float), n_nonfinite


def load_connectivity_matrix(
    parcellation: str = DEFAULT_PARCELLATION,
    kind: ConnectivityKind = "streamline_lengths",
    *,
    cohort: str = DEFAULT_COHORT,
    subject: str = "",
    subject_aggregation: SubjectAggregation = "mean",
    trim_fraction: float = 0.1,
    paradigm: str = "",
) -> tuple[FloatArray, dict[str, Any]]:
    """
    Fetch one connectivity matrix from siibra, with disk caching.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification, e.g. ``"julich 3.1"``.
    kind : {"streamline_lengths", "streamline_counts", "functional"}
        Connectivity modality.
    cohort : str
        Subject cohort (``"HCP"``, ``"1000BRAINS"``, ...). Empty string
        accepts whatever the parcellation offers first.
    subject : str
        Single subject id (see :func:`list_subjects`); empty aggregates the
        whole cohort.
    subject_aggregation : {"mean", "median", "trimmed_mean"}
        How per-subject matrices are combined. ``"median"`` and
        ``"trimmed_mean"`` hold every subject matrix in memory at once.
    trim_fraction : float
        Fraction trimmed from each tail when
        ``subject_aggregation="trimmed_mean"``; must lie in ``[0, 0.5)``.
    paradigm : str
        Case-insensitive substring selecting a functional-connectivity
        paradigm (see :func:`list_paradigms`). Ignored for structural
        modalities.

    Returns
    -------
    matrix : FloatArray of shape (N, N)
        Symmetric matrix with zero diagonal, in the parcellation's native
        region order. Streamline lengths are converted to **centimetres**;
        other modalities keep their native units.
    metadata : dict
        Provenance: region names, ancestor chains, hemispheres, subject
        count, feature name, units, and the measured asymmetry of the raw
        matrix.
    """
    payload = {
        "source": "siibra",
        "parcellation": parcellation,
        "kind": kind,
        "cohort": cohort,
        "subject": subject,
        "subject_aggregation": subject_aggregation if not subject else "single",
        "trim_fraction": trim_fraction if (subject_aggregation == "trimmed_mean" and not subject) else None,
        "paradigm": paradigm,
        "schema": 2,
    }

    def compute() -> tuple[dict[str, FloatArray], dict[str, Any]]:
        import siibra

        feature = _select_feature(parcellation, kind, cohort, paradigm)
        elements = _subject_frames(feature, subject)
        first_frame = elements[0].data
        reference_names = [str(r) for r in first_frame.index]
        matrices = [_frame_matrix(first_frame, reference_names)]
        matrices.extend(_frame_matrix(element.data, reference_names) for element in elements[1:])

        raw, n_nonfinite = _aggregate_subjects(matrices, subject_aggregation, trim_fraction)
        scale = np.abs(raw).max()
        asymmetry = float(np.abs(raw - raw.T).max() / scale) if scale > 0 else 0.0
        matrix = 0.5 * (raw + raw.T)
        np.fill_diagonal(matrix, 0.0)
        if kind == "streamline_lengths":
            matrix = matrix / _MM_PER_CM

        regions = list(first_frame.index)
        meta: dict[str, Any] = {
            **payload,
            "region_names": reference_names,
            "region_ancestors": [[a.name for a in getattr(r, "ancestors", [])] for r in regions],
            "region_keys": [str(getattr(r, "key", "")) for r in regions],
            "n_subjects": len(elements),
            "n_nonfinite_cells": n_nonfinite,
            "feature_name": str(getattr(feature, "name", "")),
            "units": "cm" if kind == "streamline_lengths" else "arbitrary",
            "source_units": "mm" if kind == "streamline_lengths" else "arbitrary",
            "relative_asymmetry": asymmetry,
            "siibra_version": getattr(siibra, "__version__", "unknown"),
        }
        return {"matrix": matrix}, meta

    entry = load_or_compute("siibra", payload, compute)
    return entry.arrays["matrix"], dict(entry.metadata)


# ---------------------------------------------------------------------------
# Region construction
# ---------------------------------------------------------------------------


def _hemisphere(name: str) -> str:
    """Return ``"left"``/``"right"`` when the region name carries a side."""
    lowered = name.lower()
    if lowered.endswith(" left"):
        return "left"
    if lowered.endswith(" right"):
        return "right"
    return ""


def _region_code(name: str) -> str:
    """Return a compact, plot-friendly code for a siibra region name."""
    hemisphere = _hemisphere(name)
    if hemisphere == "left":
        return f"{name[: -len(' left')]} L"
    if hemisphere == "right":
        return f"{name[: -len(' right')]} R"
    return name


def region_set_from_metadata(metadata: dict[str, Any]) -> RegionSet:
    """
    Build a :class:`RegionSet` from the metadata of a cached siibra query.

    Parameters
    ----------
    metadata : dict
        Metadata returned by :func:`load_connectivity_matrix`.

    Returns
    -------
    RegionSet
        Regions in matrix order. Each region carries its full atlas name,
        ancestor chain (root first), hemisphere, and siibra key as metadata,
        which is what the merging and filtering helpers in
        :mod:`diaxcondel.connectome.grouping` consume.
    """
    names: list[str] = list(metadata["region_names"])
    ancestors: list[list[str]] = list(metadata.get("region_ancestors") or [[] for _ in names])
    keys: list[str] = list(metadata.get("region_keys") or ["" for _ in names])

    seen: dict[str, int] = {}
    regions: list[Region] = []
    for name, chain, key in zip(names, ancestors, keys, strict=True):
        code = _region_code(name)
        count = seen.get(code, 0)
        seen[code] = count + 1
        if count:  # pragma: no cover - siibra region names are unique in practice
            code = f"{code}#{count + 1}"
        regions.append(
            Region(
                code=code,
                name=name,
                metadata={
                    "ancestors": tuple(chain),
                    "hemisphere": _hemisphere(name),
                    "siibra_key": key,
                },
            )
        )
    return RegionSet(tuple(regions))


# ---------------------------------------------------------------------------
# The shared region pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Regions and one matrix, after filtering, merging, masking, and pruning."""

    regions: RegionSet
    matrix: FloatArray
    lengths_cm: FloatArray
    native_n_regions: int
    metadata: dict[str, Any]
    mixture: LengthMixture | None = None


def _group_index(labels: list[str]) -> tuple[list[str], list[int]]:
    """Map labels to group indices, preserving first-appearance order."""
    order: list[str] = []
    index_of: dict[str, int] = {}
    for label in labels:
        if label not in index_of:
            index_of[label] = len(order)
            order.append(label)
    return order, [index_of[label] for label in labels]


def _shared_ancestors(regions: RegionSet, member_codes: list[str]) -> tuple[str, ...]:
    """Return the ancestor chain shared by every member of a group."""
    chains = [tuple(str(a) for a in ((regions[code].metadata or {}).get("ancestors", ()))) for code in member_codes]
    if not chains:  # pragma: no cover - groups always have members
        return ()
    shared: list[str] = []
    for level, name in enumerate(chains[0]):
        if all(len(chain) > level and chain[level] == name for chain in chains):
            shared.append(name)
        else:
            break
    return tuple(shared)


@lru_cache(maxsize=16)
def _prepare(query: AtlasQuery, kind: ConnectivityKind, paradigm: str) -> _Prepared:
    """
    Fetch one feature and apply the query's region pipeline.

    The pipeline (filter, merge, mask, prune) is identical for every
    modality, which is what guarantees that distances and weights built from
    equal queries line up region for region.
    """
    lengths, lengths_meta = load_connectivity_matrix(
        query.parcellation,
        "streamline_lengths",
        cohort=query.cohort,
        subject=query.subject,
        subject_aggregation=query.subject_aggregation,
        trim_fraction=query.trim_fraction,
    )
    regions = region_set_from_metadata(lengths_meta)
    native_n_regions = len(regions)

    if kind == "streamline_lengths":
        matrix, feature_meta = np.array(lengths, dtype=float), dict(lengths_meta)
    else:
        loaded, feature_meta = load_connectivity_matrix(
            query.parcellation,
            kind,
            cohort=query.cohort,
            subject=query.subject,
            subject_aggregation=query.subject_aggregation,
            trim_fraction=query.trim_fraction,
            paradigm=paradigm,
        )
        if loaded.shape != lengths.shape:
            raise ValueError(f"{kind} matrix shape {loaded.shape} != length matrix shape {lengths.shape}")
        matrix = np.array(loaded, dtype=float)
    lengths = np.array(lengths, dtype=float)
    mixture: LengthMixture | None = None
    dropped_indices: list[int] | None = None

    counts: FloatArray | None = None
    groups_regions = bool(query.nodes) or query.merge_level is not None
    if groups_regions and query.member_weighting == "streamline_counts":
        loaded_counts, _ = load_connectivity_matrix(
            query.parcellation,
            "streamline_counts",
            cohort=query.cohort,
            subject=query.subject,
            subject_aggregation=query.subject_aggregation,
            trim_fraction=query.trim_fraction,
        )
        counts = np.array(loaded_counts, dtype=float)

    def _restrict(keep: FloatArray) -> None:
        """Keep only the given region indices, in every array in play."""
        nonlocal regions, matrix, lengths, counts
        regions = RegionSet(tuple(regions[int(i)] for i in keep))
        matrix = matrix[np.ix_(keep, keep)]
        lengths = lengths[np.ix_(keep, keep)]
        if counts is not None:
            counts = counts[np.ix_(keep, keep)]

    # 1. Anatomical filtering.
    if query.keep_parts or query.drop_parts:
        keep_codes = select_codes(regions, keep_parts=query.keep_parts, drop_parts=query.drop_parts)
        if not keep_codes:
            raise ValueError(
                f"no regions of {query.parcellation!r} are left after keeping "
                f"{list(query.keep_parts)} and dropping {list(query.drop_parts)}"
            )
        _restrict(np.array([regions.index(code) for code in keep_codes], dtype=int))

    # 2. Grouping: either the named selection, or one level for everything.
    label_list: list[str] | None = None
    if query.nodes:
        labels, unmatched = node_labels(regions, query.nodes, split_hemispheres=query.split_hemispheres)
        if unmatched:
            warnings.warn(
                f"no region of {query.parcellation!r} matches {list(unmatched)}; "
                "check the spelling against browse_regions()",
                stacklevel=3,
            )
        if not labels:
            raise ValueError(
                f"none of the regions {list(query.nodes)} were found in {query.parcellation!r}; "
                "use browse_regions() to see what the atlas offers"
            )
        _restrict(np.array([regions.index(code) for code in regions.codes if code in labels], dtype=int))
        label_list = [labels[code] for code in regions.codes]
    elif query.merge_level is not None:
        labels = ancestor_labels(regions, query.merge_level, split_hemispheres=query.split_hemispheres)
        label_list = [labels[code] for code in regions.codes]

    if label_list is not None:
        group_names, group_index = _group_index(label_list)
        n_groups = len(group_names)
        # The representative length is plain geometry: the mean over member
        # connections that exist. How much each member contributes to the
        # *delay distribution* is recorded separately, in the mixture below.
        grouped_lengths = aggregate_matrix(lengths, group_index, n_groups, how="mean", ignore_zeros=True)
        mixture = build_length_mixture(
            lengths,
            counts if counts is not None else np.ones_like(lengths),
            group_index,
            n_groups,
        )
        if kind == "streamline_lengths":
            matrix = grouped_lengths
        elif kind == "streamline_counts":
            matrix = aggregate_matrix(matrix, group_index, n_groups, how="sum")
        else:  # functional connectivity averages over member pairs
            matrix = aggregate_matrix(matrix, group_index, n_groups, how="mean")

        members: dict[str, list[str]] = {group: [] for group in group_names}
        for code, label in zip(regions.codes, label_list, strict=True):
            members[label].append(code)
        regions = RegionSet(
            tuple(
                Region(
                    code=group,
                    name=group,
                    metadata={
                        "members": tuple(members[group]),
                        "n_members": len(members[group]),
                        "ancestors": _shared_ancestors(regions, members[group]),
                    },
                )
                for group in group_names
            )
        )
        lengths = grouped_lengths

    # 3. Symmetrise, clear the diagonal, and mask weights to measured connections.
    matrix = 0.5 * (matrix + matrix.T)
    lengths = 0.5 * (lengths + lengths.T)
    np.fill_diagonal(matrix, 0.0)
    np.fill_diagonal(lengths, 0.0)
    if kind != "streamline_lengths":
        matrix = np.where(lengths > 0, matrix, 0.0)

    # 4. Drop regions the tractography never reaches.
    if query.drop_isolated:
        connected = (lengths > 0).any(axis=1)
        if not bool(np.any(connected)):
            raise ValueError(
                f"the selected regions of {query.parcellation!r} have no connections between them "
                f"({len(regions)} node(s) after grouping); select more regions, or split one of them "
                "further, so that the model has pathways to carry signal"
            )
        if not bool(np.all(connected)):
            keep = np.flatnonzero(connected)
            dropped = [code for code, flag in zip(regions.codes, connected, strict=True) if not flag]
            regions = RegionSet(tuple(regions[int(i)] for i in keep))
            matrix = matrix[np.ix_(keep, keep)]
            lengths = lengths[np.ix_(keep, keep)]
            dropped_indices = [int(i) for i in keep]
            feature_meta = {**feature_meta, "dropped_isolated_regions": dropped}

    matrix.setflags(write=False)
    lengths.setflags(write=False)
    if mixture is not None and dropped_indices is not None:
        mixture = mixture.select(dropped_indices)
    return _Prepared(
        regions=regions,
        matrix=matrix,
        lengths_cm=lengths,
        native_n_regions=native_n_regions,
        metadata=feature_meta,
        mixture=mixture,
    )


# ---------------------------------------------------------------------------
# Public sources
# ---------------------------------------------------------------------------


def siibra_distances(
    parcellation: str = DEFAULT_PARCELLATION,
    *,
    cohort: str = DEFAULT_COHORT,
    subject: str = "",
    subject_aggregation: SubjectAggregation = "mean",
    trim_fraction: float = 0.1,
    nodes: Sequence[str] = (),
    merge_level: int | None = 3,
    split_hemispheres: bool = False,
    keep_parts: Sequence[str] = (),
    drop_parts: Sequence[str] = (),
    member_weighting: MemberWeighting = "streamline_counts",
    drop_isolated: bool = True,
) -> ManualDistances:
    """
    Load streamline-length distances (cm) from a siibra parcellation.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification, e.g. ``"julich 3.1"``.
    cohort : str
        Subject cohort of the connectivity dataset (see :func:`list_cohorts`).
    subject : str
        Single subject id (see :func:`list_subjects`); empty aggregates the
        cohort.
    subject_aggregation : {"mean", "median", "trimmed_mean"}
        How subject matrices are combined when no subject is chosen.
    trim_fraction : float
        Tail fraction trimmed per subject for ``"trimmed_mean"``.
    nodes : sequence of str
        The regions to model, named one by one from anywhere in the atlas
        tree: ``("frontal lobe", "parietal lobe", "thalamus")`` gives three
        nodes and drops everything else. Add a depth to split one of them
        further: ``"frontal lobe:1"`` makes a node per frontal sub-region.
        Use :func:`diaxcondel.connectome.grouping.browse_regions` to see the
        names on offer. Setting this replaces ``merge_level``.
    merge_level : int or None
        Depth at which *all* regions are merged, used when ``nodes`` is
        empty; ``None`` keeps the native parcellation, which is usually too
        large to simulate. Use
        :func:`diaxcondel.connectome.grouping.hierarchy_levels` to see what
        each level yields for a given atlas.
    split_hemispheres : bool
        Keep left and right groups separate when merging.
    keep_parts, drop_parts : sequence of str
        Which parts of the brain to model, named by any entry in a region's
        ancestor chain, so ``keep_parts=("cerebral cortex",)`` gives a
        cortex-only model. See
        :func:`diaxcondel.connectome.grouping.atlas_parts` for the names an
        atlas offers.
    member_weighting : {"streamline_counts", "uniform"}
        When regions are merged, how much each member connection contributes
        to the merged pathway's delay distribution: in proportion to the
        streamlines it carries, or equally. The representative *length* is
        always the plain mean over member connections.
    drop_isolated : bool
        Drop regions the tractography leaves without any connection.

    Returns
    -------
    ManualDistances
        Distances in centimetres, with the atlas query recorded in
        ``metadata["atlas_query"]`` so that matching connectivity can be
        loaded for exactly these regions.
    """
    query = AtlasQuery(
        parcellation=parcellation,
        cohort=cohort,
        subject=subject,
        subject_aggregation=subject_aggregation,
        trim_fraction=trim_fraction,
        nodes=tuple(nodes),
        merge_level=merge_level,
        split_hemispheres=split_hemispheres,
        keep_parts=tuple(keep_parts),
        drop_parts=tuple(drop_parts),
        member_weighting=member_weighting,
        drop_isolated=drop_isolated,
    )
    prepared = _prepare(query, "streamline_lengths", "")
    distances = np.array(prepared.matrix, dtype=float, copy=True)

    connected = distances > 0
    if np.any(connected):
        median_cm = float(np.median(distances[connected]))
        if not 1.0 <= median_cm <= 30.0:  # pragma: no cover - guards a unit regression upstream
            warnings.warn(
                f"median connection length is {median_cm:.2f} cm, outside the plausible range for a human brain; "
                "check the atlas units",
                stacklevel=2,
            )

    metadata = {
        QUERY_KEY: query.as_dict(),
        "source": "siibra",
        "measure": "streamline length",
        "units": "cm",
        "parcellation": parcellation,
        "cohort": cohort,
        "subject": subject or f"{prepared.metadata.get('n_subjects', 0)}-subject {subject_aggregation}",
        "n_subjects": prepared.metadata.get("n_subjects"),
        "feature": prepared.metadata.get("feature_name"),
        "native_n_regions": prepared.native_n_regions,
        "dropped_isolated_regions": prepared.metadata.get("dropped_isolated_regions", []),
        "relative_asymmetry": prepared.metadata.get("relative_asymmetry"),
        "siibra_version": prepared.metadata.get("siibra_version"),
    }
    if nodes:
        metadata["nodes"] = list(nodes)
    if prepared.mixture is not None:
        metadata["mixture_members"] = len(prepared.mixture)
        metadata["member_weighting"] = member_weighting
    if nodes:
        suffix = f" ({len(prepared.regions)} chosen regions)"
    elif merge_level is not None:
        suffix = f" @L{merge_level}"
    else:
        suffix = " (native)"
    name = f"{parcellation}{suffix}"
    return ManualDistances(
        regions=prepared.regions,
        matrix_cm=distances,
        name=name,
        metadata=metadata,
        mixture=prepared.mixture,
    )


def _normalize_weights(weights: FloatArray, normalization: WeightNormalization) -> FloatArray:
    """Scale weights so that a chosen statistic of the off-diagonal equals one."""
    n = weights.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    active = weights[offdiag]
    active = active[active > 0]
    if normalization == "none" or active.size == 0:
        return weights
    if normalization == "max":
        return weights / float(active.max())
    if normalization == "mean":
        return weights / float(active.mean())
    raise ValueError(f"unknown weight normalization {normalization!r}")


def siibra_weights(
    distances: DistanceProvider,
    *,
    source: WeightSource = "streamline_counts",
    paradigm: str = "",
    fc_exponent: float = 1.5,
    normalization: WeightNormalization = "max",
    scale: float = 1.0,
    threshold: float = 0.0,
) -> ManualConnectivity:
    """
    Load atlas connectivity weights for regions that came from the same atlas.

    Parameters
    ----------
    distances : DistanceProvider
        Distances built by :func:`siibra_distances`; the query it recorded
        determines which regions, subject, and merge level to reproduce.
    source : {"streamline_counts", "functional", "uniform"}
        Where the coupling comes from. ``"streamline_counts"`` uses
        tractography counts; ``"functional"`` applies the power-law mapping
        ``gamma = nu * b ** chi`` to resting-state functional connectivity
        ``b``; ``"uniform"`` gives every measured connection the same weight,
        isolating the effect of the distance distribution.
    paradigm : str
        Functional-connectivity paradigm substring (see :func:`list_paradigms`).
    fc_exponent : float
        Power-law exponent ``chi`` applied to functional connectivity.
    normalization : {"max", "mean", "none"}
        Rescales weights so the largest (or mean) measured connection equals
        one, which makes ``scale`` the single interpretable coupling knob.
        ``"none"`` keeps raw streamline counts.
    scale : float
        Overall coupling multiplier applied after normalisation (the ``nu``
        of the power-law mapping).
    threshold : float
        Drop connections whose normalised weight falls below this value.

    Returns
    -------
    ManualConnectivity
        Weights over exactly the regions of ``distances``.

    Raises
    ------
    ValueError
        If ``distances`` did not come from a siibra atlas, or the atlas
        regions no longer match the recorded query.
    """
    query = AtlasQuery.from_metadata(source_metadata(distances))
    regions = distances.regions

    if source == "uniform":
        weights = (np.asarray(distances.matrix(), dtype=float) > 0).astype(float)
        feature_name = "uniform over measured connections"
    else:
        kind: ConnectivityKind = "streamline_counts" if source == "streamline_counts" else "functional"
        prepared = _prepare(query, kind, paradigm)
        if prepared.regions.codes != regions.codes:  # pragma: no cover - guards a query mismatch
            raise ValueError(
                "atlas connectivity does not line up with these regions; rebuild the distances and "
                "the connectivity from the same atlas query"
            )
        weights = np.array(prepared.matrix, dtype=float, copy=True)
        feature_name = str(prepared.metadata.get("feature_name", kind))
        if source == "functional":
            weights = np.power(np.clip(weights, 0.0, None), fc_exponent)

    weights = np.clip(weights, 0.0, None)
    weights = _normalize_weights(weights, normalization)
    if threshold > 0:
        weights = np.where(weights >= threshold, weights, 0.0)
    weights = weights * scale
    np.fill_diagonal(weights, 0.0)

    metadata = {
        "source": "siibra",
        "measure": source,
        "feature": feature_name,
        "parcellation": query.parcellation,
        "cohort": query.cohort,
        "normalization": normalization,
        "scale": scale,
        "threshold": threshold,
        "fc_exponent": fc_exponent if source == "functional" else None,
        "paradigm": paradigm or None,
    }
    return ManualConnectivity(
        regions=regions,
        weights=weights,
        name=f"{source} ({query.parcellation})",
        metadata=metadata,
    )


def siibra_connectome(
    parcellation: str = DEFAULT_PARCELLATION,
    *,
    cohort: str = DEFAULT_COHORT,
    weights_from: WeightSource = "streamline_counts",
    subject: str = "",
    subject_aggregation: SubjectAggregation = "mean",
    trim_fraction: float = 0.1,
    paradigm: str = "",
    nodes: Sequence[str] = (),
    merge_level: int | None = 3,
    split_hemispheres: bool = False,
    keep_parts: Sequence[str] = (),
    drop_parts: Sequence[str] = (),
    fc_exponent: float = 1.5,
    weight_normalization: WeightNormalization = "max",
    weight_scale: float = 1.0,
    weight_threshold: float = 0.0,
    drop_isolated: bool = True,
) -> Connectome:
    """
    Build a complete connectome bundle from siibra in one call.

    A convenience wrapper over :func:`siibra_distances` and
    :func:`siibra_weights` for scripts that want both from the same atlas;
    those two functions are the composable route.

    Parameters
    ----------
    parcellation, cohort, subject, subject_aggregation, trim_fraction
        Atlas query; see :func:`siibra_distances`.
    nodes, merge_level, split_hemispheres, keep_parts, drop_parts, drop_isolated
        Region pipeline; see :func:`siibra_distances`.
    weights_from, paradigm, fc_exponent, weight_normalization, weight_scale, weight_threshold
        Connectivity options; see :func:`siibra_weights`.

    Returns
    -------
    Connectome
        Bundle whose distances are streamline lengths in centimetres and
        whose weights come from ``weights_from``.
    """
    distances = siibra_distances(
        parcellation,
        cohort=cohort,
        subject=subject,
        subject_aggregation=subject_aggregation,
        trim_fraction=trim_fraction,
        merge_level=merge_level,
        nodes=nodes,
        split_hemispheres=split_hemispheres,
        keep_parts=keep_parts,
        drop_parts=drop_parts,
        drop_isolated=drop_isolated,
    )
    weights = siibra_weights(
        distances,
        source=weights_from,
        paradigm=paradigm,
        fc_exponent=fc_exponent,
        normalization=weight_normalization,
        scale=weight_scale,
        threshold=weight_threshold,
    )
    return Connectome.from_sources(distances, weights, name=f"siibra:{parcellation}")


__all__ = [
    "DEFAULT_COHORT",
    "DEFAULT_PARCELLATION",
    "KNOWN_PARCELLATIONS",
    "QUERY_KEY",
    "AtlasQuery",
    "list_cohorts",
    "list_paradigms",
    "list_subjects",
    "load_connectivity_matrix",
    "parcellation_info",
    "region_set_from_metadata",
    "siibra_connectome",
    "siibra_distances",
    "siibra_weights",
]
