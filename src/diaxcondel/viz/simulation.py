"""
Visualisations of time-domain simulation output.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet

from ._utils import resolve_axes


def plot_signal(
    signal: FloatArray,
    regions: RegionSet,
    *,
    sample_rate_hz: float,
    t_window_s: tuple[float, float] | None = None,
    offset: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot multi-channel time-domain signal as offset traces.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Multi-channel signal.
    regions : RegionSet
        Region labels (length ``N``).
    sample_rate_hz : float
        Sampling rate, used to construct the time axis.
    t_window_s : tuple of float, optional
        ``(t_start, t_end)`` window in seconds. ``None`` plots everything;
        ``(0, 2)`` plots the first two seconds.
    offset : float, optional
        Vertical offset between channels. ``None`` uses 4 standard
        deviations of the first channel.
    title : str, optional
        Plot title.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    t_total, n = signal.shape
    if n != len(regions):
        raise ValueError(f"signal has N={n} channels but regions has {len(regions)}")

    t = np.arange(t_total) / sample_rate_hz
    if t_window_s is not None:
        mask = (t >= t_window_s[0]) & (t <= t_window_s[1])
        t = t[mask]
        sig = signal[mask]
    else:
        sig = signal

    if offset is None:
        offset = 4.0 * float(np.std(sig[:, 0])) if sig.size else 1.0

    fig, ax = resolve_axes(ax, figsize=(10, 1 + 0.6 * n))
    for i in range(n):
        ax.plot(t, sig[:, i] + i * offset, linewidth=0.8)

    ax.set_yticks([i * offset for i in range(n)])
    ax.set_yticklabels(regions.codes)
    ax.set_xlabel("Time (s)")
    ax.grid(visible=True, axis="x", alpha=0.3)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_signal_overlay(
    signal: FloatArray,
    regions: RegionSet,
    *,
    sample_rate_hz: float,
    t_window_s: tuple[float, float] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot all channels overlaid on the same y-axis (no vertical offset).

    Useful for short windows where channel amplitudes are comparable.

    Parameters
    ----------
    signal, regions, sample_rate_hz, t_window_s, title, ax
        See :func:`plot_signal`.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    t_total, n = signal.shape
    if n != len(regions):
        raise ValueError(f"signal has N={n} channels but regions has {len(regions)}")

    t = np.arange(t_total) / sample_rate_hz
    if t_window_s is not None:
        mask = (t >= t_window_s[0]) & (t <= t_window_s[1])
        t = t[mask]
        sig = signal[mask]
    else:
        sig = signal

    fig, ax = resolve_axes(ax, figsize=(10, 4))
    for i in range(n):
        ax.plot(t, sig[:, i], label=regions.codes[i], linewidth=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax
