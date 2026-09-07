"""
Visualisations of power spectral density and coherence.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet

from ._utils import resolve_axes


def plot_psd(
    freqs_hz: FloatArray,
    psd: FloatArray,
    regions: RegionSet,
    *,
    xlim: tuple[float, float] = (0.0, 30.0),
    log_y: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot per-channel PSD with the canonical EEG band-of-interest x-range.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid.
    psd : FloatArray of shape (F, N) or (F, N, N)
        Power spectra. If the array has a 3rd axis (e.g. output of
        :func:`diaxcondel.spectral.analytical.analytical_psd`), the diagonal
        is extracted and plotted.
    regions : RegionSet
        Region labels (length ``N``).
    xlim : tuple of float
        x-axis limits in Hz.
    log_y : bool
        Use a logarithmic y-axis.
    title : str, optional
        Plot title.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    if psd.ndim == 3:
        psd = np.diagonal(psd, axis1=1, axis2=2)

    f, n = psd.shape
    if f != freqs_hz.size:
        raise ValueError(f"psd has {f} frequencies but freqs_hz has {freqs_hz.size}")
    if n != len(regions):
        raise ValueError(f"psd has N={n} channels but regions has {len(regions)}")

    fig, ax = resolve_axes(ax, figsize=(8, 4))

    for i in range(n):
        ax.plot(freqs_hz, psd[:, i], label=regions.codes[i])

    ax.set_xlim(*xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    if log_y:
        ax.set_yscale("log")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_coherence(
    freqs_hz: FloatArray,
    coherence: FloatArray,
    regions: RegionSet,
    *,
    pairs: list[tuple[str, str]] | None = None,
    xlim: tuple[float, float] = (0.0, 30.0),
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot magnitude-squared coherence per region pair.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid.
    coherence : FloatArray of shape (F, N, N)
        Per-frequency coherence matrix; see
        :func:`diaxcondel.spectral.coherence.pairwise_coherence`.
    regions : RegionSet
        Region labels.
    pairs : list of (str, str), optional
        Region-code pairs to plot. ``None`` plots all upper-triangular pairs.
    xlim : tuple of float
        x-axis range.
    title : str, optional
        Plot title.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    n = len(regions)
    if coherence.shape != (freqs_hz.size, n, n):
        raise ValueError(f"coherence shape {coherence.shape} != ({freqs_hz.size}, {n}, {n})")

    selected = pairs or [(regions.codes[i], regions.codes[j]) for i in range(n) for j in range(i + 1, n)]

    fig, ax = resolve_axes(ax, figsize=(8, 4))
    for src, dst in selected:
        i = regions.index(src)
        j = regions.index(dst)
        ax.plot(freqs_hz, coherence[:, i, j], label=f"{src}–{dst}")

    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax
