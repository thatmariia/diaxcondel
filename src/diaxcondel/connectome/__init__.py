"""Connectome region, distance, and connectivity providers."""

from __future__ import annotations

from .bundle import Connectome, region_metadata, source_metadata
from .distance import DistanceProvider, ManualDistances, RelayedDistances
from .grouping import (
    HierarchyLevel,
    NodeSpec,
    RegionChoice,
    aggregate_matrix,
    ancestor_labels,
    browse_regions,
    describe_overlaps,
    group_connectome,
    hierarchy_levels,
    metadata_labels,
    node_labels,
    parse_nodes,
    select_codes,
    selection_coverage,
)
from .manual import (
    format_matrix,
    manual_connectivity,
    manual_distances,
    parse_codes,
    parse_matrix,
)
from .regions import Region, RegionSet
from .weights import ConnectivityProvider, FCConnectivity, ManualConnectivity

__all__ = [
    "Connectome",
    "ConnectivityProvider",
    "DistanceProvider",
    "FCConnectivity",
    "HierarchyLevel",
    "ManualConnectivity",
    "ManualDistances",
    "NodeSpec",
    "Region",
    "RegionChoice",
    "RegionSet",
    "RelayedDistances",
    "aggregate_matrix",
    "ancestor_labels",
    "browse_regions",
    "describe_overlaps",
    "format_matrix",
    "group_connectome",
    "hierarchy_levels",
    "manual_connectivity",
    "manual_distances",
    "metadata_labels",
    "node_labels",
    "parse_codes",
    "parse_matrix",
    "parse_nodes",
    "region_metadata",
    "select_codes",
    "selection_coverage",
    "source_metadata",
]
