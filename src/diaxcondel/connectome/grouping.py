"""
Region grouping and sub-selection for connectome bundles.

Atlas-derived connectomes are far finer than the macroscopic VAR model needs:
Julich-Brain v3.0.3 yields 314 connectivity regions, and a VAR(300) over 314
nodes is neither tractable nor identifiable. Grouping coarsens a bundle by
merging regions into lobes, into hemispheres, or into any user-supplied
partition, while keeping distances and weights physically meaningful.

There are two ways to say which regions a model has.
:func:`ancestor_labels` cuts the whole tree at one depth, which is quick but
blunt. :func:`node_labels` instead takes a list of regions chosen by name,
from any level, each kept whole or split further by its own depth. It is the only
way to ask for, say, two lobes and the entire thalamus.
:func:`browse_regions` lists what is on offer, one level at a time or by
name, so a level becomes a way of *looking* at the atlas rather than of
choosing from it.

Aggregation rules
-----------------
**Weights** are summed (``"sum"``, appropriate for streamline counts, where
the group-to-group total is the sum of member-pair counts) or averaged
(``"mean"``, appropriate for correlation-like measures).

**Distances** are averaged across member pairs (``"mean"``) or reduced to the
shortest connected member (``"min"``). Zero entries are treated as *absent
connections* rather than zero-length ones and are excluded from the average:
a distance matrix from tractography is sparse, and averaging its zeros in
would collapse group distances towards zero.

The averaged length is only a summary. Merging also records the individual
member lengths as a
:class:`~diaxcondel.connectome.distance.LengthMixture`, so that model
builders can evaluate the delay kernel at each real length and mix the
results, weighted by how many fibres carry each one, instead of pretending
the pathway has a single length. Nothing about conduction speed or axon
calibre is altered by that weighting: it says how much of the signal travels
each distance, which is what a merged pathway actually is.

Within-group connections are discarded (the diagonal is zeroed): the model
has no self-connections, and local recurrent dynamics are absorbed into the
noise term (see :mod:`diaxcondel.model.var.linear`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from diaxcondel._typing import FloatArray

from .bundle import Connectome, region_metadata
from .distance import LengthMixture, ManualDistances
from .regions import Region, RegionSet
from .weights import ManualConnectivity

DistanceAggregation = Literal["mean", "min"]
WeightAggregation = Literal["sum", "mean"]


def _indicator(group_index: Sequence[int], n_groups: int) -> FloatArray:
    """Return the ``(G, N)`` 0/1 group-membership matrix."""
    idx = np.asarray(group_index, dtype=int)
    ind = np.zeros((n_groups, idx.size), dtype=float)
    ind[idx, np.arange(idx.size)] = 1.0
    return ind


def aggregate_matrix(
    matrix: FloatArray,
    group_index: Sequence[int],
    n_groups: int,
    *,
    how: Literal["sum", "mean"] = "mean",
    ignore_zeros: bool = False,
    pair_weights: FloatArray | None = None,
) -> FloatArray:
    """
    Aggregate a square region-by-region matrix over groups.

    Parameters
    ----------
    matrix : FloatArray of shape (N, N)
        Matrix to aggregate.
    group_index : sequence of int
        ``group_index[i]`` is the group of region ``i``; values in
        ``[0, n_groups)``.
    n_groups : int
        Number of output groups.
    how : {"sum", "mean"}
        ``"sum"`` adds member-pair entries; ``"mean"`` averages them.
    ignore_zeros : bool
        When ``True`` (and ``how="mean"``), zero entries are excluded from
        both numerator and denominator, so they read as *missing* rather than
        as a measured zero.
    pair_weights : FloatArray of shape (N, N), optional
        Non-negative weights for a weighted mean over member pairs. Ignored
        when ``how="sum"``. Pairs whose weights sum to zero fall back to the
        unweighted mean over the same pairs.

    Returns
    -------
    FloatArray of shape (G, G)
        Aggregated matrix. Group pairs with no contributing member pair are
        zero.

    Raises
    ------
    ValueError
        If shapes disagree or ``group_index`` is out of range.
    """
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    if m.ndim != 2 or m.shape[1] != n:
        raise ValueError(f"matrix must be square; got shape {m.shape}")
    idx = np.asarray(group_index, dtype=int)
    if idx.shape != (n,):
        raise ValueError(f"group_index must have length {n}; got {idx.shape}")
    if n_groups < 1 or idx.min(initial=0) < 0 or (n and idx.max(initial=0) >= n_groups):
        raise ValueError("group_index values must lie in [0, n_groups)")

    ind = _indicator(idx, n_groups)
    if how == "sum":
        return ind @ m @ ind.T

    mask = np.ones_like(m) if not ignore_zeros else (m != 0).astype(float)
    if pair_weights is not None:
        w = np.asarray(pair_weights, dtype=float)
        if w.shape != m.shape:
            raise ValueError(f"pair_weights shape {w.shape} != matrix shape {m.shape}")
        if np.any(w < 0):
            raise ValueError("pair_weights must be non-negative")
        weighted_mask = mask * w
        num = ind @ (m * weighted_mask) @ ind.T
        den = ind @ weighted_mask @ ind.T
        # Fall back to the unweighted mean where the weights vanish.
        num_u = ind @ (m * mask) @ ind.T
        den_u = ind @ mask @ ind.T
        out = np.zeros_like(num)
        np.divide(num, den, out=out, where=den > 0)
        fallback = (den <= 0) & (den_u > 0)
        np.divide(num_u, den_u, out=out, where=fallback)
        return out

    num = ind @ (m * mask) @ ind.T
    den = ind @ mask @ ind.T
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _labels_to_index(labels: Sequence[str]) -> tuple[list[str], list[int]]:
    """Map labels to group indices, preserving first-appearance order."""
    order: list[str] = []
    index_of: dict[str, int] = {}
    for label in labels:
        if label not in index_of:
            index_of[label] = len(order)
            order.append(label)
    return order, [index_of[label] for label in labels]


def group_connectome(
    connectome: Connectome,
    labels: Mapping[str, str] | Sequence[str],
    *,
    distance_how: DistanceAggregation = "mean",
    weight_how: WeightAggregation = "sum",
    name: str | None = None,
    keep_mixture: bool = True,
) -> Connectome:
    """
    Merge regions of a connectome according to a partition.

    Parameters
    ----------
    connectome : Connectome
        Bundle to coarsen.
    labels : Mapping[str, str] or sequence of str
        Either ``{region_code: group_label}`` (regions missing from the
        mapping keep their own code as label) or a sequence of labels
        parallel to ``connectome.regions``.
    distance_how : {"mean", "min"}
        How member-pair distances combine into the group's representative
        length: the mean over connected member pairs, or the shortest one.
    weight_how : {"sum", "mean"}
        How member-pair weights combine.
    name : str, optional
        Name for the derived bundle.
    keep_mixture : bool
        Record the individual member lengths (and the weights carrying them)
        so that delays can be mixed rather than averaged.

    Returns
    -------
    Connectome
        Coarsened bundle whose regions carry ``members`` metadata listing the
        source region codes.

    Raises
    ------
    ValueError
        If ``labels`` has the wrong length, or an aggregation rule is unknown.
    """
    codes = connectome.codes
    if isinstance(labels, Mapping):
        label_list = [str(labels.get(code, code)) for code in codes]
    else:
        label_list = [str(label) for label in labels]
        if len(label_list) != len(codes):
            raise ValueError(f"labels must have length {len(codes)}; got {len(label_list)}")

    group_codes, group_index = _labels_to_index(label_list)
    n_groups = len(group_codes)

    d = connectome.distance_matrix()
    w = connectome.weight_matrix()

    if distance_how == "mean":
        grouped_d = aggregate_matrix(d, group_index, n_groups, how="mean", ignore_zeros=True)
    elif distance_how == "min":
        grouped_d = _aggregate_min(d, group_index, n_groups)
    else:
        raise ValueError(f"unknown distance_how {distance_how!r}")

    if weight_how not in {"sum", "mean"}:
        raise ValueError(f"unknown weight_how {weight_how!r}")
    grouped_w = aggregate_matrix(w, group_index, n_groups, how=weight_how, ignore_zeros=weight_how == "mean")

    # Symmetrise (aggregation of a symmetric matrix stays symmetric up to
    # floating-point noise) and drop within-group connections.
    grouped_d = 0.5 * (grouped_d + grouped_d.T)
    np.fill_diagonal(grouped_d, 0.0)
    np.fill_diagonal(grouped_w, 0.0)
    # A group pair without a measured distance cannot carry a delay.
    grouped_w = np.where(grouped_d > 0, grouped_w, 0.0)

    members: dict[str, list[str]] = {code: [] for code in group_codes}
    for code, label in zip(codes, label_list, strict=True):
        members[label].append(code)

    regions = RegionSet(
        tuple(
            Region(
                code=group_code,
                name=group_code,
                metadata={
                    "members": tuple(members[group_code]),
                    "n_members": len(members[group_code]),
                },
            )
            for group_code in group_codes
        )
    )
    metadata = {
        **dict(connectome.metadata or {}),
        "derived_from": connectome.name,
        "grouping": {
            "distance_how": distance_how,
            "weight_how": weight_how,
            "n_groups": n_groups,
        },
    }
    bundle_name = name or f"{connectome.name}[{n_groups} groups]"
    mixture = build_length_mixture(d, w, group_index, n_groups) if keep_mixture else None
    return Connectome(
        regions=regions,
        distances=ManualDistances(
            regions=regions,
            matrix_cm=grouped_d,
            name=bundle_name,
            metadata=metadata,
            mixture=mixture,
        ),
        weights=ManualConnectivity(
            regions=regions,
            weights=grouped_w,
            name=getattr(connectome.weights, "name", "connectivity"),
            metadata=metadata,
        ),
        name=bundle_name,
        metadata=metadata,
    )


def build_length_mixture(
    distances_cm: FloatArray,
    pair_weights: FloatArray,
    group_index: Sequence[int],
    n_groups: int,
) -> LengthMixture:
    """
    Collect the member lengths behind every merged connection.

    Parameters
    ----------
    distances_cm : FloatArray of shape (N, N)
        Fine-grained distances; zero means no measured connection.
    pair_weights : FloatArray of shape (N, N)
        How much each member connection carries: streamline counts, or any
        non-negative strength. All-zero weights fall back to counting every
        member equally.
    group_index : sequence of int
        Group each fine-grained region belongs to.
    n_groups : int
        Number of merged regions.

    Returns
    -------
    LengthMixture
        Upper-triangle members with positive length, indexed by merged pair.
        Members inside a group are dropped: the model has no self-connections
        through white matter.
    """
    d = np.asarray(distances_cm, dtype=float)
    w = np.asarray(pair_weights, dtype=float)
    idx = np.asarray(group_index, dtype=int)
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    upper = np.triu(np.ones_like(d, dtype=bool), k=1)
    members = upper & (d > 0) & (idx[:, None] != idx[None, :])
    i, j = np.nonzero(members)
    rows = idx[i].copy()
    cols = idx[j].copy()
    swap = rows > cols
    rows[swap], cols[swap] = cols[swap], rows[swap]
    weights = w[i, j]
    if weights.size and not np.any(weights > 0):
        weights = np.ones_like(weights)
    return LengthMixture(rows=rows, cols=cols, lengths_cm=d[i, j], weights=weights)


def _aggregate_min(matrix: FloatArray, group_index: Sequence[int], n_groups: int) -> FloatArray:
    """Return the minimum positive entry per group pair (zero when none)."""
    m = np.asarray(matrix, dtype=float)
    idx = np.asarray(group_index, dtype=int)
    out = np.zeros((n_groups, n_groups))
    masked = np.where(m > 0, m, np.inf)
    for g in range(n_groups):
        rows = masked[idx == g]
        for h in range(n_groups):
            block = rows[:, idx == h]
            value = block.min() if block.size else np.inf
            out[g, h] = 0.0 if not np.isfinite(value) else float(value)
    return out


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """
    One model node, named after a region of the atlas tree.

    Attributes
    ----------
    name : str
        Atlas region name at any level: ``"frontal lobe"``, ``"thalamus"``,
        ``"Area hOc1 (V1)"``. Matched case-insensitively; a name the atlas
        actually has matches only itself, and free text that names no region
        falls back to the highest region whose name contains it.
    depth : int
        How far *below* that region to split it. ``0`` (the default) keeps it
        whole, as a single node; ``1`` makes one node per immediate child,
        and so on down to the atlas's own leaves.
    """

    name: str
    depth: int = 0

    def as_text(self) -> str:
        """Return the ``"name"`` or ``"name:depth"`` form of this node."""
        return self.name if self.depth == 0 else f"{self.name}:{self.depth}"


def parse_nodes(specs: Sequence[str | NodeSpec] | Mapping[str, int] | None) -> tuple[NodeSpec, ...]:
    """
    Read a node selection written as names, ``"name:depth"``, or a mapping.

    Parameters
    ----------
    specs : sequence of str or NodeSpec, Mapping[str, int], or None
        ``["frontal lobe", "thalamus"]`` selects two whole regions;
        ``["frontal lobe:1", "thalamus"]`` splits the frontal lobe one level
        further while the thalamus stays a single node. A mapping from name
        to depth is equivalent.

    Returns
    -------
    tuple of NodeSpec
        Parsed selection, in the order given.

    Raises
    ------
    ValueError
        If a depth is not a non-negative integer, or a name is empty.
    """
    if not specs:
        return ()
    if isinstance(specs, Mapping):
        items: list[NodeSpec] = [NodeSpec(str(name).strip(), int(depth)) for name, depth in specs.items()]
    else:
        items = []
        for entry in specs:
            if isinstance(entry, NodeSpec):
                items.append(entry)
                continue
            text = str(entry).strip()
            name, separator, depth_text = text.rpartition(":")
            if not separator:
                items.append(NodeSpec(text))
                continue
            try:
                depth = int(depth_text)
            except ValueError:
                # A colon inside a region name ("Area 4p: left") is not a depth.
                items.append(NodeSpec(text))
                continue
            items.append(NodeSpec(name.strip(), depth))
    for spec in items:
        if not spec.name:
            raise ValueError("a node selection cannot contain an empty region name")
        if spec.depth < 0:
            raise ValueError(f"node {spec.as_text()!r} must use a non-negative depth")
    return tuple(items)


def _region_chain(region: Region) -> tuple[str, ...]:
    """Return the region's ancestor names followed by its own name."""
    ancestors = region_metadata(region, "ancestors") or ()
    return (*(str(a) for a in ancestors), region.name or region.code)


def _match_position(chain: Sequence[str], name: str, *, exact: bool) -> int | None:
    """
    Return the chain position ``name`` selects, or ``None``.

    An exact name resolves to its deepest occurrence, so a repeated name
    means the more specific one. Free text resolves to its *shallowest*
    match instead: searching for "occipital" asks for the occipital lobe,
    not for every area whose name mentions it.
    """
    wanted = name.lower()
    if exact:
        positions = [i for i, entry in enumerate(chain) if entry.lower() == wanted]
        return positions[-1] if positions else None
    positions = [i for i, entry in enumerate(chain) if wanted in entry.lower()]
    return positions[0] if positions else None


def node_labels(
    regions: RegionSet,
    nodes: Sequence[str | NodeSpec] | Mapping[str, int],
    *,
    split_hemispheres: bool = False,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """
    Label regions by an explicit selection of atlas regions.

    This is region selection by name rather than by level: each entry of
    ``nodes`` names a region anywhere in the atlas tree and becomes one model
    node, unless it carries a depth that splits it further. Regions under no
    selected node are left unlabelled, which drops them from the model, so
    "the frontal and parietal lobes plus the whole thalamus" is exactly
    ``("frontal lobe", "parietal lobe", "thalamus")``, with no single
    hierarchy level able to express it.

    Where several entries match one region, the most specific wins: the one
    matching deepest in that region's ancestor chain.

    Parameters
    ----------
    regions : RegionSet
        Regions carrying ancestor metadata (see :func:`ancestor_labels`).
    nodes : sequence of str or NodeSpec, or Mapping[str, int]
        The selection; see :func:`parse_nodes` for the accepted forms.
    split_hemispheres : bool
        Append each region's hemisphere to its label, keeping left and right
        nodes separate.

    Returns
    -------
    labels : dict of str to str
        Region code to node label, covering only the selected regions.
    unmatched : tuple of str
        Entries of ``nodes`` that matched no region at all, so a caller can
        report a typo rather than silently modelling fewer nodes.

    Raises
    ------
    ValueError
        If ``nodes`` is empty or malformed.
    """
    specs = parse_nodes(nodes)
    if not specs:
        raise ValueError("no regions selected; name at least one atlas region")

    chains = {region.code: _region_chain(region) for region in regions.regions}
    # A name that exists in the atlas is matched exactly, so choosing the
    # thalamus does not also drag in the subthalamus. Free text that names no
    # region falls back to a substring, which is what makes searching work.
    exact_only = _exact_names(regions, specs)

    labels: dict[str, str] = {}
    used: set[str] = set()
    for region in regions.regions:
        chain = chains[region.code]
        best: tuple[int, int, NodeSpec] | None = None
        for spec in specs:
            position = _match_position(chain, spec.name, exact=exact_only[spec.name])
            if position is None:
                continue
            candidate = (position, len(spec.name), spec)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            continue
        position, _, spec = best
        used.add(spec.name)
        index = min(position + spec.depth, len(chain) - 1)
        label = region.code if index == len(chain) - 1 else chain[index]
        if split_hemispheres:
            hemisphere = str(region_metadata(region, "hemisphere", "") or "")
            if hemisphere:
                label = f"{label} {hemisphere}"
        labels[region.code] = label
    unmatched = tuple(spec.name for spec in specs if spec.name not in used)
    return labels, unmatched


def _exact_names(regions: RegionSet, specs: Sequence[NodeSpec]) -> dict[str, bool]:
    """Say, per spec, whether the atlas has a region of exactly that name."""
    known = {entry.lower() for region in regions.regions for entry in _region_chain(region)}
    return {spec.name: spec.name.lower() in known for spec in specs}


def selection_coverage(
    regions: RegionSet,
    nodes: Sequence[str | NodeSpec] | Mapping[str, int],
) -> dict[str, tuple[str, ...]]:
    """
    Return the atlas regions each entry of a selection covers.

    Parameters
    ----------
    regions : RegionSet
        Regions carrying ancestor metadata.
    nodes : sequence of str or NodeSpec, or Mapping[str, int]
        The selection; see :func:`parse_nodes`.

    Returns
    -------
    dict of str to tuple of str
        Region codes matched by each entry, judged on its own, unlike
        :func:`node_labels`, where entries compete and the most specific one
        wins.
    """
    specs = parse_nodes(nodes)
    exact_only = _exact_names(regions, specs)
    coverage: dict[str, list[str]] = {spec.name: [] for spec in specs}
    for region in regions.regions:
        chain = _region_chain(region)
        for spec in specs:
            if _match_position(chain, spec.name, exact=exact_only[spec.name]) is not None:
                coverage[spec.name].append(region.code)
    return {name: tuple(codes) for name, codes in coverage.items()}


def describe_overlaps(
    regions: RegionSet,
    nodes: Sequence[str | NodeSpec] | Mapping[str, int],
) -> tuple[str, ...]:
    """
    Report selections that lay claim to the same atlas regions.

    Overlap is legal, since :func:`node_labels` gives a shared region to
    whichever entry matches deepest in its ancestor chain, but it is rarely what
    someone means. Asking for the cerebral cortex *and* the frontal lobe gives
    a frontal node and a cortex node holding everything else, not two nodes of
    equal standing.

    Parameters
    ----------
    regions : RegionSet
        Regions carrying ancestor metadata.
    nodes : sequence of str or NodeSpec, or Mapping[str, int]
        The selection to check.

    Returns
    -------
    tuple of str
        One sentence per overlapping pair, naming what happens. Empty when
        the selection is disjoint.
    """
    specs = parse_nodes(nodes)
    coverage = selection_coverage(regions, specs)
    messages: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, first in enumerate(specs):
        for second in specs[index + 1 :]:
            if first.name == second.name:
                messages.append(f"{first.name!r} is selected twice.")
                continue
            pair = tuple(sorted((first.name, second.name)))
            if pair in seen:
                continue
            shared = set(coverage[first.name]) & set(coverage[second.name])
            if not shared:
                continue
            seen.add(pair)
            inner, outer = first, second
            if len(coverage[first.name]) > len(coverage[second.name]):
                inner, outer = second, first
            plural = "region" if len(shared) == 1 else "regions"
            messages.append(
                f"{outer.name!r} contains {inner.name!r} ({len(shared)} {plural} in common); "
                f"those go to {inner.name!r}, and {outer.name!r} keeps the rest."
            )
    return tuple(messages)


@dataclass(frozen=True, slots=True)
class RegionChoice:
    """
    One region of the atlas tree, offered as a candidate model node.

    Attributes
    ----------
    name : str
        The region's name, as the atlas writes it.
    level : int
        Its depth below the atlas root, so choices can be listed one level at
        a time.
    n_members : int
        How many of the parcellation's finest regions sit underneath it:
        the size of the node it would become.
    """

    name: str
    level: int
    n_members: int


def browse_regions(
    regions: RegionSet,
    *,
    level: int | None = None,
    contains: str = "",
    include_leaves: bool = True,
) -> tuple[RegionChoice, ...]:
    """
    List the atlas regions that can be selected as model nodes.

    Levels are a way of *looking* at the tree, not of choosing from it: pick
    a level to see a manageable list, or search by name to find a region
    wherever it sits. What comes back is the menu :func:`node_labels` reads.

    Parameters
    ----------
    regions : RegionSet
        Regions carrying ancestor metadata.
    level : int, optional
        Only list regions at this depth below the atlas root. ``None`` lists
        every level.
    contains : str
        Only list regions whose name contains this text (case-insensitive).
    include_leaves : bool
        Include the parcellation's own regions, not just their ancestors.

    Returns
    -------
    tuple of RegionChoice
        Choices ordered by level, then by first appearance in the atlas.
    """
    counts: dict[tuple[int, str], int] = {}
    order: list[tuple[int, str]] = []
    for region in regions.regions:
        chain = _region_chain(region)
        last = len(chain) - 1
        for position, name in enumerate(chain):
            if position == last and not include_leaves:
                continue
            key = (position, name)
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
    wanted = contains.strip().lower()
    return tuple(
        RegionChoice(name=name, level=position, n_members=counts[(position, name)])
        for position, name in sorted(order, key=lambda key: (key[0], order.index(key)))
        if (level is None or position == level) and (not wanted or wanted in name.lower())
    )


def ancestor_labels(
    regions: RegionSet,
    level: int,
    *,
    split_hemispheres: bool = False,
) -> dict[str, str]:
    """
    Derive group labels from the atlas region hierarchy.

    Regions loaded from siibra carry their ancestor chain in
    ``Region.metadata["ancestors"]``, ordered from the atlas root down to the
    region's immediate parent. This helper picks the ancestor at a fixed
    depth, which is what turns a fine parcellation into a lobe-level or
    lobe-and-hemisphere-level model.

    One depth for the whole brain is a blunt instrument: see
    :func:`node_labels` to choose regions individually instead, each with its
    own depth.

    Parameters
    ----------
    regions : RegionSet
        Regions to label. Regions without ancestor metadata keep their own
        code as label (i.e. they are not merged).
    level : int
        Default depth below the atlas root. ``0`` puts everything in one
        group; for Julich-Brain, ``3`` corresponds to lobes and larger values
        give progressively finer groups. Levels deeper than a region's own
        chain clamp to that region's parent.
    split_hemispheres : bool
        When ``True``, append the region's hemisphere
        (``Region.metadata["hemisphere"]``) to the label, keeping left and
        right groups separate.

    Returns
    -------
    dict of str to str
        Mapping from region code to group label, ready for
        :func:`group_connectome`.

    Raises
    ------
    ValueError
        If ``level`` is negative.
    """
    if level < 0:
        raise ValueError("level must be non-negative")

    labels: dict[str, str] = {}
    for region in regions.regions:
        ancestors = region_metadata(region, "ancestors")
        if not ancestors:
            labels[region.code] = region.code
            continue
        chain = [str(a) for a in ancestors]
        label = chain[min(level, len(chain) - 1)]
        if split_hemispheres:
            hemisphere = str(region_metadata(region, "hemisphere", "") or "")
            if hemisphere:
                label = f"{label} {hemisphere}"
        labels[region.code] = label
    return labels


@dataclass(frozen=True, slots=True)
class HierarchyLevel:
    """
    What one merge level of an atlas hierarchy produces.

    Attributes
    ----------
    level : int
        Depth below the atlas root.
    n_groups : int
        Number of regions the model would have at this level.
    examples : tuple of str
        The first few group names, so the level can be recognised at a glance.
    """

    level: int
    n_groups: int
    examples: tuple[str, ...]


def hierarchy_levels(
    regions: RegionSet,
    *,
    max_level: int = 8,
    split_hemispheres: bool = False,
    n_examples: int = 4,
) -> tuple[HierarchyLevel, ...]:
    """
    Describe what every merge level of an atlas hierarchy would produce.

    Merge levels are only meaningful relative to the atlas they come from:
    "level 3" means lobes in one parcellation and something else in another.
    This function answers the question directly, by counting the groups each
    level produces and naming a few of them.

    Parameters
    ----------
    regions : RegionSet
        Regions carrying ancestor metadata (see :func:`ancestor_labels`).
    max_level : int
        Deepest level to report.
    split_hemispheres : bool
        Report levels as they would be with hemispheres kept separate.
    n_examples : int
        How many group names to include per level.

    Returns
    -------
    tuple of HierarchyLevel
        One entry per level, ending as soon as a level resolves every region
        separately (deeper levels would be identical).
    """
    out: list[HierarchyLevel] = []
    for level in range(max(0, max_level) + 1):
        labels = ancestor_labels(regions, level, split_hemispheres=split_hemispheres)
        names, _ = _labels_to_index([labels[code] for code in regions.codes])
        out.append(HierarchyLevel(level=level, n_groups=len(names), examples=tuple(names[:n_examples])))
        if len(names) >= len(regions):
            break
    return tuple(out)


def select_codes(
    regions: RegionSet,
    *,
    keep_parts: Sequence[str] = (),
    drop_parts: Sequence[str] = (),
    name_contains: str = "",
) -> tuple[str, ...]:
    """
    Return the codes of regions belonging to chosen parts of the brain.

    Atlases cover more than a given model needs: a cortical model has no use
    for cerebellar nuclei. Parts are named by any entry in a region's
    ancestor chain (``"cerebral cortex"``, ``"thalamus"``, ``"frontal
    lobe"``), matched case-insensitively on substrings.

    Parameters
    ----------
    regions : RegionSet
        Regions to filter.
    keep_parts : sequence of str
        Keep regions belonging to *any* of these parts. Empty keeps
        everything.
    drop_parts : sequence of str
        Drop regions belonging to any of these parts, applied after
        ``keep_parts``.
    name_contains : str
        Keep only regions whose name or code contains this text.

    Returns
    -------
    tuple of str
        Matching region codes, in the original order.
    """
    keep_lower = [part.lower() for part in keep_parts if part]
    drop_lower = [part.lower() for part in drop_parts if part]
    kept: list[str] = []
    for region in regions.regions:
        chain = " ".join(str(a).lower() for a in (region_metadata(region, "ancestors") or ()))
        haystack = f"{chain} {region.name.lower()} {region.code.lower()}"
        if keep_lower and not any(part in chain for part in keep_lower):
            continue
        if drop_lower and any(part in chain for part in drop_lower):
            continue
        if name_contains and name_contains.lower() not in haystack:
            continue
        kept.append(region.code)
    return tuple(kept)


def atlas_parts(regions: RegionSet, level: int = 2) -> tuple[str, ...]:
    """
    List the names of the atlas branches at one level of the region tree.

    Parameters
    ----------
    regions : RegionSet
        Regions carrying ancestor metadata.
    level : int
        Depth below the atlas root; level 2 is typically the coarse tissue
        division (cortex, subcortical nuclei, thalamus, cerebellum).

    Returns
    -------
    tuple of str
        Branch names, in first-appearance order: the choices worth offering
        when asking which parts of the brain to model.
    """
    labels = ancestor_labels(regions, level)
    names, _ = _labels_to_index([labels[code] for code in regions.codes])
    return tuple(names)


def metadata_labels(regions: RegionSet, key: str) -> dict[str, str]:
    """
    Derive group labels from a single region-metadata field.

    Parameters
    ----------
    regions : RegionSet
        Regions to label.
    key : str
        Metadata key to read (e.g. ``"hemisphere"`` or ``"parent"``).

    Returns
    -------
    dict of str to str
        Mapping from region code to group label. Regions lacking the key
        keep their own code.
    """
    labels: dict[str, str] = {}
    for region in regions.regions:
        value = region_metadata(region, key)
        labels[region.code] = str(value) if value else region.code
    return labels
