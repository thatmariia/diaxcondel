"""
Visualisation helpers (optional ``viz`` extra).
"""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
except ImportError as e:
    raise ImportError("matplotlib is required for diaxcondel.viz; install with `poetry install --with viz`.") from e

from .connectivity import plot_connectivity_matrix, plot_lag_kernels
from .network import complete_distances, mds_layout
from .power import plot_coherence, plot_psd
from .simulation import plot_signal, plot_signal_overlay
from .style import PALETTE, region_colors, styled, use_style

__all__ = [
    "PALETTE",
    "complete_distances",
    "mds_layout",
    "plot_coherence",
    "plot_connectivity_matrix",
    "plot_lag_kernels",
    "plot_psd",
    "plot_signal",
    "plot_signal_overlay",
    "region_colors",
    "styled",
    "use_style",
]
