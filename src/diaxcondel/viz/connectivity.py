"""
Visualisations of connectivity and lag-kernel structure.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet

from ._utils import resolve_axes


def plot_connectivity_matrix(
    matrix: FloatArray,
    regions: RegionSet,
    *,
    log: bool = False,
    title: str | None = None,
    cmap: str = "viridis",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot a region-by-region matrix as a labelled heatmap.

    Parameters
    ----------
    matrix : FloatArray of shape (N, N)
        Square matrix to plot. Convention: ``matrix[i, j]`` is the
        post-pre weight from region ``j`` onto region ``i``.
    regions : RegionSet
        Region labels for the axes (must have ``N`` entries).
    log : bool
        Use a logarithmic colour scale. Falls back to linear if any entry
        is non-positive.
    title : str, optional
        Plot title.
    cmap : str
        matplotlib colormap name.
    ax : Axes, optional
        Existing axes to draw on. A new figure is created if ``None``.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    n = len(regions)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} != ({n}, {n})")

    fig, ax = resolve_axes(ax, figsize=(6, 5))

    norm = None
    if log and np.all(matrix > 0):
        norm = LogNorm(vmin=matrix.min(), vmax=matrix.max())

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(regions.codes, rotation=45, ha="right")
    ax.set_yticklabels(regions.codes)
    ax.set_xlabel("source (pre)")
    ax.set_ylabel("target (post)")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_lag_kernels(
    phi: FloatArray,
    regions: RegionSet,
    *,
    dt_s: float,
    pairs: list[tuple[str, str]] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot lag kernels ``Phi[:, i, j]`` over delay (ms) for selected pairs.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag-coefficient tensor as built by
        :func:`diaxcondel.model.build.build_lag_tensor`.
    regions : RegionSet
        Region labels.
    dt_s : float
        Sampling period. Used to convert lag index to milliseconds.
    pairs : list of (str, str), optional
        Source-target code pairs to plot. ``None`` plots all upper-triangular
        pairs with non-zero kernel.
    title : str, optional
        Plot title.
    ax : Axes, optional
        Existing axes to draw on.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    p, n, _ = phi.shape
    if phi.shape[2] != n:
        raise ValueError(f"phi has inconsistent N axes: {phi.shape}")
    if len(regions) != n:
        raise ValueError(f"regions has {len(regions)} entries but phi has N={n}")

    fig, ax = resolve_axes(ax, figsize=(8, 4))
    delays_ms = np.arange(1, p + 1) * dt_s * 1000.0

    selected = pairs or _auto_select_pairs(phi, regions)
    for src, dst in selected:
        i = regions.index(dst)
        j = regions.index(src)
        y = phi[:, i, j]
        if not np.any(y):
            continue
        ax.plot(delays_ms, y, label=f"{src} → {dst}")

    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Phi[p, dst, src]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def _auto_select_pairs(phi: FloatArray, regions: RegionSet) -> list[tuple[str, str]]:
    """Pick upper-triangular pairs with any non-zero kernel entry."""
    n = phi.shape[1]
    out: list[tuple[str, str]] = []
    for j in range(n):
        for i in range(j + 1, n):
            if np.any(phi[:, i, j] != 0.0):
                out.append((regions.codes[j], regions.codes[i]))
    return out
