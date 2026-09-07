"""
Figures for the playground.

Every figure is drawn inside the package's shared style, so the dashboard,
notebooks, and any figure exported from here look like one piece of work.
Where a view can be dominated by noise — traces, spectra, coherence — the
function takes a smoothing argument, applied for display only and always
reported in the caption next to the plot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from diaxcondel._typing import FloatArray  # noqa: E402
from diaxcondel.analysis.smoothing import smooth_over_frequency, smooth_over_time  # noqa: E402
from diaxcondel.connectome.bundle import Connectome  # noqa: E402
from diaxcondel.connectome.regions import RegionSet  # noqa: E402
from diaxcondel.experiment import RunResult, Spectra  # noqa: E402
from diaxcondel.viz.network import stress_layout  # noqa: E402
from diaxcondel.viz.style import (  # noqa: E402
    ACCENT,
    ANNOTATION,
    DISTANCE_MAP,
    SEQUENTIAL,
    region_colors,
    region_styles,
    styled,
)

_MAX_TICK_LABELS = 32
_MAX_LEGEND_ENTRIES = 12


def _matrix_axes(ax: plt.Axes, regions: RegionSet) -> None:
    """Label matrix axes with region codes, thinning them when crowded."""
    n = len(regions)
    ax.grid(visible=False)
    ax.set_ylabel("target")
    if n <= _MAX_TICK_LABELS:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(regions.codes, rotation=90, fontsize=7)
        ax.set_yticklabels(regions.codes, fontsize=7)
        ax.set_xlabel("source")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"source ({n} regions)")


def _legend(ax: plt.Axes, n_entries: int) -> None:
    """Add a legend only when it would stay readable."""
    if 0 < n_entries <= _MAX_LEGEND_ENTRIES:
        ax.legend(fontsize=7, ncol=2, loc="upper right")


def connectome_figure(connectome: Connectome) -> Figure:
    """
    Plot the distance and weight matrices side by side.

    Parameters
    ----------
    connectome : Connectome
        Bundle to display.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels: conduction distance and coupling weight.
    """
    with styled():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
        distances = connectome.distance_matrix()
        weights = connectome.weight_matrix()

        first = axes[0].imshow(
            np.where(distances > 0, distances, np.nan),
            cmap=DISTANCE_MAP,
            interpolation="nearest",
        )
        axes[0].set_title("Conduction distance")
        fig.colorbar(first, ax=axes[0], fraction=0.046, pad=0.03, label="cm")

        positive = weights[weights > 0]
        ratio = float(positive.max() / positive.min()) if positive.size else 1.0
        norm = (
            LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
            if ratio > 100
            else Normalize(0.0, float(weights.max()) if weights.size else 1.0)
        )
        second = axes[1].imshow(
            np.where(weights > 0, weights, np.nan),
            cmap=SEQUENTIAL,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].set_title("Coupling weight" + (" (log scale)" if ratio > 100 else ""))
        fig.colorbar(second, ax=axes[1], fraction=0.046, pad=0.03, label="weight")

        for ax in axes:
            _matrix_axes(ax, connectome.regions)
        return fig


def network_figure(
    connectome: Connectome,
    *,
    max_labels: int = 30,
    separation_fraction: float = 0.0,
) -> tuple[Figure, float]:
    """
    Draw the network laid out so that drawn separations follow conduction distance.

    Parameters
    ----------
    connectome : Connectome
        Bundle to display.
    max_labels : int
        Label regions only when there are at most this many.
    separation_fraction : float
        Minimum separation between nodes, as a fraction of the median
        connection length. Zero keeps the layout faithful to the distances;
        raising it spreads crowded regions apart at the cost of distorting
        them, which the returned distortion reports.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Nodes placed by stress majorisation, with edges shaded by coupling
        strength.
    distortion : float
        Median relative difference between drawn and true separations.
    """
    distances = connectome.distance_matrix()
    weights = connectome.weight_matrix()
    measured = distances[distances > 0]
    floor = separation_fraction * float(np.median(measured)) if measured.size else 0.0
    coordinates, distortion = stress_layout(distances, min_separation=floor)

    with styled():
        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        n = connectome.n_regions
        strongest = float(weights.max()) if weights.size else 1.0
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if weights[i, j] > 0]
        # Draw weak connections first so strong ones stay visible on top.
        pairs.sort(key=lambda pair: weights[pair])
        for i, j in pairs:
            strength = weights[i, j] / strongest if strongest > 0 else 0.0
            ax.plot(
                [coordinates[i, 0], coordinates[j, 0]],
                [coordinates[i, 1], coordinates[j, 1]],
                color=ANNOTATION,
                alpha=float(np.clip(0.08 + 0.5 * strength, 0.05, 0.65)),
                linewidth=float(np.clip(0.4 + 2.2 * strength, 0.4, 2.8)),
                zorder=1,
            )

        degree = (weights > 0).sum(axis=1)
        sizes = 50 + 200 * (degree / degree.max() if degree.max() else 1)
        colours = region_colors(n) if n <= 8 else [ACCENT] * n
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=sizes,
            c=colours,
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )

        # Labels are pushed away from the centre of mass so that the crowded
        # middle of a layout stays readable.
        if n <= max_labels:
            centre = coordinates.mean(axis=0)
            spread = float(np.linalg.norm(coordinates - centre, axis=1).max()) or 1.0
            for index, code in enumerate(connectome.codes):
                direction = coordinates[index] - centre
                norm = float(np.linalg.norm(direction)) or 1.0
                offset = 11 * direction / norm
                ax.annotate(
                    code if len(code) <= 26 else f"{code[:24]}…",
                    coordinates[index],
                    fontsize=6.5,
                    ha="left" if direction[0] >= 0 else "right",
                    va="bottom" if direction[1] >= 0 else "top",
                    xytext=tuple(offset),
                    textcoords="offset points",
                    color="#222222",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.8},
                    zorder=3,
                )
            ax.margins(0.18 if spread > 0 else 0.1)
        else:
            ax.set_title(f"{n} regions (too many to label)")

        ax.set_aspect("equal")
        ax.set_xlabel("layout axis (cm)")
        ax.set_ylabel("layout axis (cm)")
        ax.grid(visible=True, alpha=0.25)
        return fig, distortion


def distance_histogram_figure(connectome: Connectome, *, bins: int = 30) -> Figure:
    """
    Plot the distribution of connected-pair distances.

    Parameters
    ----------
    connectome : Connectome
        Bundle to display.
    bins : int
        Maximum number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
        Histogram of connection lengths, with the median marked.
    """
    distances = connectome.distance_matrix()
    weights = connectome.weight_matrix()
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1) & (weights > 0) & (distances > 0)
    values = distances[mask]

    with styled():
        fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
        if values.size:
            ax.hist(
                values,
                bins=min(bins, max(5, values.size // 2)),
                color=ACCENT,
                alpha=0.85,
                edgecolor="white",
            )
            median = float(np.median(values))
            ax.axvline(median, color=ANNOTATION, linestyle="--", linewidth=1.0)
            ax.annotate(
                f"median {median:.1f} cm",
                (median, ax.get_ylim()[1]),
                xytext=(5, -11),
                textcoords="offset points",
                fontsize=7,
                color=ANNOTATION,
            )
        ax.set_xlabel("Connection length (cm)")
        ax.set_ylabel("Connections")
        return fig


def lag_kernel_figure(
    phi: FloatArray,
    regions: RegionSet,
    dt_s: float,
    pairs: Sequence[tuple[str, str]],
) -> Figure:
    """
    Plot delay-weighted lag kernels for selected region pairs.

    Parameters
    ----------
    phi : FloatArray of shape (P, N, N)
        Lag tensor.
    regions : RegionSet
        Region labels.
    dt_s : float
        Sample period.
    pairs : sequence of (str, str)
        ``(target, source)`` code pairs to draw.

    Returns
    -------
    matplotlib.figure.Figure
        One curve per pair, against delay in milliseconds.
    """
    lags_ms = 1000.0 * np.arange(1, phi.shape[0] + 1) * dt_s
    styles = region_styles(max(1, len(pairs)))
    with styled():
        fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
        for index, (target, source) in enumerate(pairs):
            i, j = regions.index(target), regions.index(source)
            colour, dash = styles[index]
            ax.plot(lags_ms, phi[:, i, j], label=f"{source} → {target}", color=colour, linestyle=dash)
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Lag weight")
        _legend(ax, len(pairs))
        return fig


def delay_scatter_figure(lengths_cm: FloatArray, peak_delay_ms: FloatArray, coupling: FloatArray) -> Figure:
    """
    Plot each connection's peak delay against its length.

    Parameters
    ----------
    lengths_cm : FloatArray of shape (N, N)
        Connection lengths.
    peak_delay_ms : FloatArray of shape (N, N)
        Delay at which each connection's lag weights peak.
    coupling : FloatArray of shape (N, N)
        Total weight per connection, used for marker size.

    Returns
    -------
    matplotlib.figure.Figure
        Scatter of delay against length, with the implied frequency on a
        second axis.
    """
    mask = np.isfinite(peak_delay_ms) & (coupling > 0) & (lengths_cm > 0)
    lengths = lengths_cm[mask]
    delays = peak_delay_ms[mask]
    strengths = coupling[mask]
    largest = float(strengths.max()) if strengths.size else 1.0
    sizes = 10 + 70 * (strengths / largest if largest > 0 else 1.0)

    with styled():
        fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
        ax.scatter(lengths, delays, s=sizes, alpha=0.55, color=ACCENT, edgecolor="none")
        ax.set_xlabel("Connection length (cm)")
        ax.set_ylabel("Peak delay (ms)")

        if delays.size:
            right = ax.twinx()
            right.set_ylim(ax.get_ylim())
            ticks = np.linspace(float(np.nanmin(delays)), float(np.nanmax(delays)), 5)
            right.set_yticks(ticks)
            right.set_yticklabels([f"{1000.0 / (2.0 * value):.0f}" for value in ticks])
            right.set_ylabel("Implied frequency (Hz)")
            right.grid(visible=False)
        return fig


def signal_figure(
    run: RunResult,
    window_s: tuple[float, float],
    *,
    smoothing_ms: float = 0.0,
    trial: int = 0,
) -> Figure:
    """
    Plot simulated time series as offset traces.

    Parameters
    ----------
    run : RunResult
        A completed run.
    window_s : tuple of float
        Time window to display, in seconds.
    smoothing_ms : float
        Moving-average window applied for display only; ``0`` shows the raw
        simulation.
    trial : int
        Which trial to show.

    Returns
    -------
    matplotlib.figure.Figure
        Offset traces, one per region.
    """
    sample_rate_hz = float(run.params.sample_rate_hz)
    signal = np.asarray(run.trials.trials[trial], dtype=float)
    if smoothing_ms > 0:
        signal = smooth_over_time(signal, sample_rate_hz, smoothing_ms)

    time_s = np.arange(signal.shape[0]) / sample_rate_hz
    mask = (time_s >= window_s[0]) & (time_s <= window_s[1])
    time_s, shown = time_s[mask], signal[mask]
    n = shown.shape[1]
    offset = 4.0 * float(np.std(shown)) if shown.size else 1.0
    colors = region_colors(n)

    with styled():
        fig, ax = plt.subplots(figsize=(10.0, 1.4 + 0.42 * n), constrained_layout=True)
        for index in range(n):
            ax.plot(time_s, shown[:, index] + index * offset, color=colors[index], linewidth=0.9)
        ax.set_yticks([index * offset for index in range(n)])
        ax.set_yticklabels(run.regions.codes, fontsize=7)
        ax.set_xlabel("Time (s)")
        ax.grid(visible=True, axis="x")
        return fig


def spectra_figure(
    spectra: Spectra,
    regions: RegionSet,
    *,
    log_y: bool = True,
    log_x: bool = False,
    fmax_hz: float = 45.0,
    smoothing_hz: float = 0.0,
    highlight: Sequence[str] = (),
) -> Figure:
    """
    Plot the simulated spectrum and, when available, the closed-form one.

    Parameters
    ----------
    spectra : Spectra
        Spectra from :func:`diaxcondel.experiment.compute_spectra`.
    regions : RegionSet
        Region labels.
    log_y : bool
        Use a logarithmic power axis.
    log_x : bool
        Use a logarithmic frequency axis, which is how a broadband slope is
        usually judged.
    fmax_hz : float
        Upper frequency limit.
    smoothing_hz : float
        Moving-average bandwidth applied to the simulated spectrum for
        display only.
    highlight : sequence of str
        Region codes to draw in colour; the rest are drawn in light grey.
        Empty highlights every region, which stays readable up to about two
        dozen curves.

    Returns
    -------
    matplotlib.figure.Figure
        One panel for the simulated spectrum, one for the closed-form
        spectrum when it exists.
    """
    simulated = np.asarray(spectra.welch, dtype=float)
    if smoothing_hz > 0:
        simulated = smooth_over_frequency(spectra.freqs_hz, simulated, smoothing_hz)
    has_analytic = spectra.has_analytic
    chosen = [code for code in (highlight or regions.codes) if code in regions.codes]
    styles = dict(zip(chosen, region_styles(len(chosen)), strict=True))

    with styled():
        fig, axes = plt.subplots(
            1,
            2 if has_analytic else 1,
            figsize=(11.0 if has_analytic else 6.6, 3.8),
            constrained_layout=True,
            squeeze=False,
            # One power axis for both panels: the two spectra are on the same
            # scale, and sharing the axis is what makes that checkable.
            sharey=True,
        )
        panels = axes[0]
        _draw_spectrum(panels[0], spectra.freqs_hz, simulated, regions, styles)
        panels[0].set_title(f"Simulated ({spectra.estimator})")

        if has_analytic and spectra.analytic is not None and spectra.analytic_freqs_hz is not None:
            _draw_spectrum(panels[1], spectra.analytic_freqs_hz, spectra.analytic, regions, styles)
            panels[1].set_title("Closed form")

        for ax in panels:
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            if log_y:
                ax.set_yscale("log")
                _readable_log_ticks(ax)
            if log_x:
                ax.set_xscale("log")
                lowest = float(spectra.freqs_hz[1]) if spectra.freqs_hz.size > 1 else 1.0
                ax.set_xlim(max(0.5, lowest), fmax_hz)
            else:
                ax.set_xlim(0, fmax_hz)
        _legend(panels[0], len(styles))
        return fig


def _readable_log_ticks(ax: plt.Axes) -> None:
    """Use plain numbers on a log axis that spans less than a decade."""
    low, high = ax.get_ylim()
    if low > 0 and high / low < 10:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())


def _draw_spectrum(
    ax: plt.Axes,
    freqs_hz: FloatArray,
    power: FloatArray,
    regions: RegionSet,
    styles: Mapping[str, tuple[str, str]],
) -> None:
    """Draw one spectrum panel, greying out the regions that are not highlighted."""
    for index, code in enumerate(regions.codes):
        style = styles.get(code)
        if style is None:
            ax.plot(freqs_hz, power[:, index], color="#C9C9C9", linewidth=0.7, zorder=1)
    for index, code in enumerate(regions.codes):
        style = styles.get(code)
        if style is not None:
            colour, dash = style
            ax.plot(freqs_hz, power[:, index], label=code, color=colour, linestyle=dash, linewidth=1.2, zorder=2)


def coherence_figure(
    freqs_hz: FloatArray,
    coherence: FloatArray,
    regions: RegionSet,
    *,
    fmax_hz: float,
    smoothing_hz: float = 0.0,
    max_pairs: int = 20,
) -> Figure:
    """
    Plot pairwise magnitude-squared coherence.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid.
    coherence : FloatArray of shape (F, N, N)
        Coherence matrix per frequency.
    regions : RegionSet
        Region labels.
    fmax_hz : float
        Upper frequency limit.
    smoothing_hz : float
        Moving-average bandwidth applied for display only.
    max_pairs : int
        Draw at most this many region pairs, choosing the most coherent.

    Returns
    -------
    matplotlib.figure.Figure
        One curve per region pair.
    """
    values = np.asarray(coherence, dtype=float)
    if smoothing_hz > 0:
        flat = values.reshape(values.shape[0], -1)
        values = smooth_over_frequency(freqs_hz, flat, smoothing_hz).reshape(values.shape)

    n = len(regions)
    band = freqs_hz <= fmax_hz
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda pair: float(values[band, pair[0], pair[1]].mean()), reverse=True)
    shown = pairs[:max_pairs]
    styles = region_styles(max(1, len(shown)))

    with styled():
        fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
        for index, (i, j) in enumerate(shown):
            colour, dash = styles[index]
            ax.plot(
                freqs_hz,
                values[:, i, j],
                label=f"{regions.codes[i]}–{regions.codes[j]}",
                color=colour,
                linestyle=dash,
                linewidth=1.0,
            )
        ax.set_xlim(0, fmax_hz)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Coherence")
        _legend(ax, len(shown))
        if len(pairs) > len(shown):
            ax.set_title(f"{len(shown)} most coherent of {len(pairs)} pairs")
        return fig


def default_event_window(
    onsets_s: Sequence[float], record_end_s: float, *, before: float = 0.5, after: float = 1.5
) -> tuple[float, float]:
    """
    Return the time window an event-related average should be read in.

    Parameters
    ----------
    onsets_s : sequence of float
        Event onsets, in seconds after the start of the analysis window.
    record_end_s : float
        End of the recording, so the window cannot run past the data.
    before : float
        Seconds of baseline to keep before the first onset.
    after : float
        Seconds to keep after the last onset.

    Returns
    -------
    tuple of float
        Start and end of the window. With no events it is the whole record.
    """
    if not onsets_s:
        return (0.0, record_end_s)
    start = max(0.0, min(onsets_s) - before)
    end = min(record_end_s, max(onsets_s) + after)
    return (start, end if end > start else record_end_s)


def erp_figure(
    run: RunResult,
    onsets_s: Sequence[float],
    *,
    smoothing_ms: float = 0.0,
    window_s: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the trial-averaged response with event onsets marked.

    Parameters
    ----------
    run : RunResult
        A completed multi-trial run.
    onsets_s : sequence of float
        Event onsets, in seconds after the start of the analysis window.
    smoothing_ms : float
        Moving-average window applied for display only.
    window_s : tuple of float, optional
        Absolute time window to draw. ``None`` shows a second either side of
        the events, which is where an event-related response lives; the rest
        of a long recording only compresses it out of view.

    Returns
    -------
    matplotlib.figure.Figure
        Trial average per region, with a shaded standard error.
    """
    time_s = run.time_s
    average = np.asarray(run.average, dtype=float)
    sem = np.asarray(run.trials.sem, dtype=float)
    if smoothing_ms > 0:
        sample_rate_hz = float(run.params.sample_rate_hz)
        average = smooth_over_time(average, sample_rate_hz, smoothing_ms)
        sem = smooth_over_time(sem, sample_rate_hz, smoothing_ms)
    window = window_s if window_s is not None else default_event_window(onsets_s, float(time_s[-1]))
    shown = (time_s >= window[0]) & (time_s <= window[1])
    time_s, average, sem = time_s[shown], average[shown], sem[shown]
    styles = region_styles(average.shape[1])

    with styled():
        fig, ax = plt.subplots(figsize=(9.6, 4.0), constrained_layout=True)
        for index, code in enumerate(run.regions.codes):
            colour, dash = styles[index]
            ax.plot(time_s, average[:, index], label=code, color=colour, linestyle=dash, linewidth=1.2)
            ax.fill_between(
                time_s,
                average[:, index] - sem[:, index],
                average[:, index] + sem[:, index],
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
        for onset in onsets_s:
            if window[0] <= onset <= window[1]:
                ax.axvline(onset, color=ANNOTATION, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xlim(*window)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial-averaged activity")
        _legend(ax, len(run.regions))
        return fig


def drive_figure(waveforms: Mapping[str, FloatArray], sample_rate_hz: float, onsets_s: Mapping[str, float]) -> Figure:
    """
    Plot the stimulus waveforms of the configured events.

    Parameters
    ----------
    waveforms : Mapping[str, FloatArray]
        Event name to rendered waveform.
    sample_rate_hz : float
        Sampling rate the waveforms were rendered at.
    onsets_s : Mapping[str, float]
        Event name to onset time.

    Returns
    -------
    matplotlib.figure.Figure
        Each waveform drawn at its onset time.
    """
    styles = region_styles(max(1, len(waveforms)))
    with styled():
        fig, ax = plt.subplots(figsize=(7.6, 2.6), constrained_layout=True)
        for index, (name, waveform) in enumerate(waveforms.items()):
            onset = onsets_s.get(name, 0.0)
            time_s = onset + np.arange(waveform.size) / sample_rate_hz
            colour, dash = styles[index]
            ax.plot(time_s, waveform, label=name, color=colour, linestyle=dash, linewidth=1.2)
        ax.axhline(0.0, color=ANNOTATION, linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stimulus")
        _legend(ax, len(waveforms))
        return fig


def sweep_figure(
    values: Sequence[float],
    metric: Sequence[float],
    *,
    x_label: str,
    y_label: str,
    reference: float | None = None,
) -> Figure:
    """
    Plot one summary measure against a swept parameter.

    Parameters
    ----------
    values : sequence of float
        Parameter values, in order.
    metric : sequence of float
        Measured value at each point.
    x_label, y_label : str
        Axis labels.
    reference : float, optional
        Horizontal reference line, e.g. the stability boundary at one.

    Returns
    -------
    matplotlib.figure.Figure
        Line plot with markers at the simulated points.
    """
    with styled():
        fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
        ax.plot(list(values), list(metric), marker="o", markersize=5, color=ACCENT)
        if reference is not None:
            ax.axhline(reference, color=ANNOTATION, linestyle="--", linewidth=0.9)
            ax.annotate(
                "stability boundary",
                (list(values)[0], reference),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=ANNOTATION,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return fig
