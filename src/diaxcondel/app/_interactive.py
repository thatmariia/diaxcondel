"""
Interactive versions of the playground's figures.

The static figures in :mod:`diaxcondel.app._figures` are what you export into
a paper. These are what you explore with: the same numbers, the same palette,
drawn with Plotly so that a curve can be hovered for its exact value, a band
zoomed into, and a region switched off by clicking its legend entry. On a
model with thirty regions that is the difference between a readable figure and
a thicket.

Every function here returns a ``plotly.graph_objects.Figure``. The module
imports Plotly at call time, so the package still works without it — the page
falls back to the static figures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from diaxcondel._typing import FloatArray
from diaxcondel.analysis.smoothing import smooth_over_frequency, smooth_over_time
from diaxcondel.connectome.bundle import Connectome
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.experiment import RunResult, Spectra
from diaxcondel.viz.style import DISTANCE_MAP, PALETTE, SEQUENTIAL, region_styles

#: Dash patterns matching the line styles of the static figures.
_DASH = {"-": "solid", "--": "dash", ":": "dot"}

#: Grey used for curves that are present but not highlighted.
_MUTED = "#C9C9C9"

#: Font and hover settings shared by every figure here. The legend sits below
#: the axes so it never lands on a panel title, and the margins leave room for
#: it without squeezing the data.
_LAYOUT: dict[str, Any] = {
    "hovermode": "closest",
    "font": {"size": 12},
    "legend": {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.18,
        "xanchor": "left",
        "x": 0,
        "font": {"size": 11},
    },
}


def _style(
    figure: go.Figure,
    *,
    height: int,
    x_title: str = "",
    y_title: str = "",
    has_legend: bool = False,
) -> go.Figure:
    """Apply the shared layout to a figure and return it."""
    figure.update_layout(
        height=height + (60 if has_legend else 0),
        margin={"l": 70, "r": 30, "t": 46, "b": 110 if has_legend else 56},
        **_LAYOUT,
    )
    if x_title:
        figure.update_xaxes(title_text=x_title)
    if y_title:
        figure.update_yaxes(title_text=y_title)
    return figure


def _traces_for(regions: RegionSet, highlight: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Return colour and dash per highlighted region code."""
    chosen = [code for code in (highlight or regions.codes) if code in regions.codes]
    return dict(zip(chosen, region_styles(len(chosen)), strict=True))


def signal_figure(
    run: RunResult,
    window_s: tuple[float, float],
    *,
    smoothing_ms: float = 0.0,
    trial: int = 0,
) -> go.Figure:
    """
    Draw simulated time series as offset traces.

    Parameters
    ----------
    run : RunResult
        A completed run.
    window_s : tuple of float
        Time window to display, in seconds.
    smoothing_ms : float
        Moving-average window applied for display only.
    trial : int
        Which trial to show.

    Returns
    -------
    plotly.graph_objects.Figure
        One trace per region, stacked, with the region's own value on hover
        rather than the offset one.
    """
    sample_rate_hz = float(run.params.sample_rate_hz)
    signal = np.asarray(run.trials.trials[trial], dtype=float)
    if smoothing_ms > 0:
        signal = smooth_over_time(signal, sample_rate_hz, smoothing_ms)

    time_s = np.arange(signal.shape[0]) / sample_rate_hz
    mask = (time_s >= window_s[0]) & (time_s <= window_s[1])
    time_s, shown = time_s[mask], signal[mask]
    n_regions = shown.shape[1]
    offset = 4.0 * float(np.std(shown)) if shown.size else 1.0
    styles = region_styles(n_regions)

    figure = go.Figure()
    for index, code in enumerate(run.regions.codes):
        colour, _ = styles[index]
        figure.add_trace(
            go.Scatter(
                x=time_s,
                y=shown[:, index] + index * offset,
                name=code,
                mode="lines",
                line={"color": colour, "width": 1.1},
                customdata=shown[:, index],
                hovertemplate=f"<b>{code}</b><br>%{{x:.3f}} s<br>%{{customdata:.3g}}<extra></extra>",
            )
        )
    figure.update_yaxes(
        tickmode="array",
        tickvals=[index * offset for index in range(n_regions)],
        ticktext=list(run.regions.codes),
    )
    figure.update_layout(showlegend=False)
    return _style(figure, height=max(280, 60 + 34 * n_regions), x_title="Time (s)")


def spectra_figure(
    spectra: Spectra,
    regions: RegionSet,
    *,
    log_y: bool = True,
    log_x: bool = False,
    fmax_hz: float = 45.0,
    smoothing_hz: float = 0.0,
    highlight: Sequence[str] = (),
) -> go.Figure:
    """
    Draw the simulated spectrum and, where it exists, the closed-form one.

    Parameters
    ----------
    spectra : Spectra
        Spectra from :func:`diaxcondel.experiment.compute_spectra`.
    regions : RegionSet
        Region labels.
    log_y, log_x : bool
        Logarithmic power and frequency axes.
    fmax_hz : float
        Upper frequency limit.
    smoothing_hz : float
        Display-only smoothing bandwidth for the simulated spectrum.
    highlight : sequence of str
        Region codes drawn in colour; the rest are drawn in grey and can be
        brought back by clicking the legend.

    Returns
    -------
    plotly.graph_objects.Figure
        Both spectra on one frequency axis, sharing a legend so that
        switching a region off switches it off in both.
    """
    from plotly.subplots import make_subplots

    simulated = np.asarray(spectra.welch, dtype=float)
    if smoothing_hz > 0:
        simulated = smooth_over_frequency(spectra.freqs_hz, simulated, smoothing_hz)
    styles = _traces_for(regions, highlight)
    has_analytic = spectra.has_analytic and spectra.analytic is not None

    titles = (f"Simulated ({spectra.estimator})", "Closed form")
    # One power axis for both panels: the two spectra are on the same scale,
    # and sharing the axis is what makes that checkable at a glance.
    figure = make_subplots(
        rows=1,
        cols=2 if has_analytic else 1,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=titles if has_analytic else titles[:1],
    )

    panels: list[tuple[int, FloatArray, FloatArray]] = [(1, spectra.freqs_hz, simulated)]
    if has_analytic and spectra.analytic_freqs_hz is not None and spectra.analytic is not None:
        panels.append((2, spectra.analytic_freqs_hz, spectra.analytic))

    for column, freqs, power in panels:
        for index, code in enumerate(regions.codes):
            style = styles.get(code)
            colour, dash = style if style else (_MUTED, "-")
            figure.add_trace(
                go.Scatter(
                    x=freqs,
                    y=power[:, index],
                    name=code,
                    legendgroup=code,
                    showlegend=column == 1,
                    mode="lines",
                    line={"color": colour, "width": 1.6 if style else 0.9, "dash": _DASH[dash]},
                    hovertemplate=f"<b>{code}</b><br>%{{x:.2f}} Hz<br>%{{y:.3g}}<extra></extra>",
                ),
                row=1,
                col=column,
            )

    lowest = float(spectra.freqs_hz[1]) if spectra.freqs_hz.size > 1 else 1.0
    figure.update_xaxes(
        title_text="Frequency (Hz)",
        type="log" if log_x else "linear",
        range=[np.log10(max(0.5, lowest)), np.log10(fmax_hz)] if log_x else [0, fmax_hz],
    )
    figure.update_yaxes(title_text="Power", type="log" if log_y else "linear", col=1)
    return _style(figure, height=400, has_legend=True)


def matrix_figure(
    matrix: FloatArray,
    regions: RegionSet,
    *,
    title: str,
    units: str = "",
    colorscale: str | None = None,
    mask_zeros: bool = True,
) -> go.Figure:
    """
    Draw a region-by-region matrix as a heat map.

    Parameters
    ----------
    matrix : FloatArray of shape (N, N)
        Values to draw. Entry ``[i, j]`` is read as input from ``j`` to ``i``.
    regions : RegionSet
        Row and column labels.
    title : str
        Panel title.
    units : str
        Unit shown in the hover box.
    colorscale : str, optional
        Plotly colour scale; defaults to the package's sequential map.
    mask_zeros : bool
        Leave zero cells blank rather than colouring them. A zero off the
        diagonal means *no connection*, not a small one, and colouring it
        would put absent pairs at one end of the scale.

    Returns
    -------
    plotly.graph_objects.Figure
        Heat map whose hover names both regions, which is what makes a large
        matrix readable at all.
    """
    codes = list(regions.codes)
    suffix = f" {units}" if units else ""
    values = np.asarray(matrix, dtype=float)
    if mask_zeros:
        values = np.where(values > 0, values, np.nan)
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=codes,
            y=codes,
            colorscale=colorscale or SEQUENTIAL,
            hovertemplate=f"from %{{x}}<br>to %{{y}}<br>%{{z:.3g}}{suffix}<extra></extra>",
            colorbar={"title": {"text": units or ""}, "thickness": 12},
        )
    )
    figure.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    figure.update_layout(title={"text": title, "x": 0.0, "font": {"size": 13}})
    return _style(figure, height=420, x_title="source region", y_title="target region")


def connectome_figure(connectome: Connectome) -> go.Figure:
    """
    Draw the distance and weight matrices side by side.

    Parameters
    ----------
    connectome : Connectome
        The bundle to display.

    Returns
    -------
    plotly.graph_objects.Figure
        Two heat maps: connection lengths, and the coupling over them.
    """
    from plotly.subplots import make_subplots

    codes = list(connectome.codes)
    distances = connectome.distance_matrix()
    weights = connectome.weight_matrix()
    # An absent connection is a blank cell, not the extreme of a colour scale.
    lengths = np.where(distances > 0, distances, np.nan)
    coupling = np.where(weights > 0, weights, np.nan)

    positive = weights[weights > 0]
    log_weights = bool(positive.size) and float(positive.max() / positive.min()) > 100
    if log_weights:
        coupling = np.log10(coupling)

    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,  # room for the first panel's colour bar
        subplot_titles=(
            "Connection length (cm)",
            "Coupling weight" + (" (log scale)" if log_weights else ""),
        ),
    )
    figure.add_trace(
        go.Heatmap(
            z=lengths,
            x=codes,
            y=codes,
            colorscale=DISTANCE_MAP,
            colorbar={"title": {"text": "cm"}, "x": 0.40, "thickness": 12},
            hovertemplate="from %{x}<br>to %{y}<br>%{z:.2f} cm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=coupling,
            x=codes,
            y=codes,
            customdata=weights,
            colorscale=SEQUENTIAL,
            colorbar={"title": {"text": "log₁₀" if log_weights else "weight"}, "thickness": 12},
            hovertemplate="from %{x}<br>to %{y}<br>weight %{customdata:.3g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(autorange="reversed")
    return _style(figure, height=440)


def network_figure(
    coords: FloatArray,
    connectome: Connectome,
    *,
    max_edges: int = 400,
) -> go.Figure:
    """
    Draw the network laid out so that drawn distance follows conduction distance.

    Parameters
    ----------
    coords : FloatArray of shape (N, 2)
        Node positions from :func:`diaxcondel.viz.network.stress_layout`.
    connectome : Connectome
        Regions, distances, and weights.
    max_edges : int
        Draw only the strongest connections beyond this many, so a dense
        atlas model stays legible.

    Returns
    -------
    plotly.graph_objects.Figure
        Nodes sized by how many connections they carry, edges shaded by
        coupling strength, everything named on hover.
    """
    coords = np.asarray(coords, dtype=float)
    weights = connectome.weight_matrix()
    distances = connectome.distance_matrix()
    codes = list(connectome.codes)

    rows, cols = np.triu_indices(len(codes), k=1)
    strengths = weights[rows, cols]
    keep = strengths > 0
    rows, cols, strengths = rows[keep], cols[keep], strengths[keep]
    if rows.size > max_edges:
        strongest = np.argsort(strengths)[-max_edges:]
        rows, cols, strengths = rows[strongest], cols[strongest], strengths[strongest]

    figure = go.Figure()
    if rows.size:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for i, j in zip(rows, cols, strict=True):
            edge_x += [coords[i, 0], coords[j, 0], None]
            edge_y += [coords[i, 1], coords[j, 1], None]
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "rgba(120,120,120,0.35)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # A second, invisible trace carries the hover text for each edge.
        figure.add_trace(
            go.Scatter(
                x=(coords[rows, 0] + coords[cols, 0]) / 2,
                y=(coords[rows, 1] + coords[cols, 1]) / 2,
                mode="markers",
                marker={"size": 10, "color": "rgba(0,0,0,0)"},
                text=[
                    f"{codes[i]} — {codes[j]}<br>{distances[i, j]:.1f} cm<br>weight {weights[i, j]:.3g}"
                    for i, j in zip(rows, cols, strict=True)
                ],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    degree = np.count_nonzero(weights > 0, axis=1)
    figure.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=codes,
            textposition="top center",
            textfont={"size": 10},
            marker={
                "size": 12 + 14 * degree / max(1, degree.max()),
                "color": PALETTE[0],
                "line": {"color": "white", "width": 1.5},
            },
            hovertemplate="<b>%{text}</b><br>%{marker.size:.0f} connections<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _style(figure, height=520)


def delay_distribution_figure(
    lags_ms: FloatArray,
    weights: FloatArray,
    labels: Sequence[str],
    *,
    max_curves: int = 12,
) -> go.Figure:
    """
    Draw the delay distribution each connection contributes.

    Parameters
    ----------
    lags_ms : FloatArray of shape (P,)
        Lag axis in milliseconds.
    weights : FloatArray of shape (P, K)
        Lag weights per connection.
    labels : sequence of str
        One label per connection.
    max_curves : int
        Draw at most this many, choosing the strongest.

    Returns
    -------
    plotly.graph_objects.Figure
        One curve per connection, hoverable for the weight at a given delay.
    """
    weights = np.asarray(weights, dtype=float)
    totals = weights.sum(axis=0)
    order = np.argsort(totals)[::-1][:max_curves]
    styles = region_styles(len(order))

    figure = go.Figure()
    for position, index in enumerate(order):
        colour, dash = styles[position]
        figure.add_trace(
            go.Scatter(
                x=lags_ms,
                y=weights[:, index],
                name=labels[index],
                mode="lines",
                line={"color": colour, "width": 1.5, "dash": _DASH[dash]},
                hovertemplate=f"<b>{labels[index]}</b><br>%{{x:.1f}} ms<br>%{{y:.3g}}<extra></extra>",
            )
        )
    return _style(figure, height=340, x_title="Delay (ms)", y_title="Lag weight", has_legend=True)


def scatter_figure(
    x: FloatArray,
    y: FloatArray,
    *,
    labels: Sequence[str],
    x_title: str,
    y_title: str,
    colour: FloatArray | None = None,
    colour_title: str = "",
) -> go.Figure:
    """
    Draw a labelled scatter, e.g. connection length against the delay it produces.

    Parameters
    ----------
    x, y : FloatArray
        Point coordinates.
    labels : sequence of str
        Name of each point, shown on hover.
    x_title, y_title : str
        Axis labels.
    colour : FloatArray, optional
        Value mapped to marker colour.
    colour_title : str
        Label for the colour bar.

    Returns
    -------
    plotly.graph_objects.Figure
        Scatter with per-point hover text.
    """
    marker: dict[str, Any] = {"size": 7, "color": PALETTE[0], "opacity": 0.8}
    if colour is not None:
        marker = {
            "size": 7,
            "color": np.asarray(colour, dtype=float),
            "colorscale": SEQUENTIAL,
            "colorbar": {"title": {"text": colour_title}, "thickness": 12},
            "opacity": 0.85,
        }
    figure = go.Figure(
        go.Scatter(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            mode="markers",
            marker=marker,
            text=list(labels),
            hovertemplate="<b>%{text}</b><br>%{x:.3g}<br>%{y:.3g}<extra></extra>",
        )
    )
    return _style(figure, height=380, x_title=x_title, y_title=y_title)


def coherence_figure(
    freqs_hz: FloatArray,
    coherence: FloatArray,
    regions: RegionSet,
    *,
    fmax_hz: float = 45.0,
    smoothing_hz: float = 0.0,
    max_pairs: int = 12,
) -> go.Figure:
    """
    Draw magnitude-squared coherence for the most strongly coupled pairs.

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
        Display-only smoothing bandwidth.
    max_pairs : int
        How many pairs to draw, chosen by mean coherence.

    Returns
    -------
    plotly.graph_objects.Figure
        One curve per pair, on a fixed 0-1 axis.
    """
    codes = list(regions.codes)
    rows, cols = np.triu_indices(len(codes), k=1)
    curves = coherence[:, rows, cols]
    if smoothing_hz > 0:
        curves = smooth_over_frequency(freqs_hz, curves, smoothing_hz)
    order = np.argsort(curves.mean(axis=0))[::-1][:max_pairs]
    styles = region_styles(len(order))

    figure = go.Figure()
    for position, index in enumerate(order):
        colour, dash = styles[position]
        label = f"{codes[rows[index]]} — {codes[cols[index]]}"
        figure.add_trace(
            go.Scatter(
                x=freqs_hz,
                y=curves[:, index],
                name=label,
                mode="lines",
                line={"color": colour, "width": 1.4, "dash": _DASH[dash]},
                hovertemplate=f"<b>{label}</b><br>%{{x:.2f}} Hz<br>coherence %{{y:.3f}}<extra></extra>",
            )
        )
    figure.update_xaxes(range=[0, fmax_hz])
    figure.update_yaxes(range=[0, 1])
    return _style(figure, height=380, x_title="Frequency (Hz)", y_title="Coherence", has_legend=True)


def erp_figure(
    time_s: FloatArray,
    average: FloatArray,
    regions: RegionSet,
    onsets_s: Sequence[float],
    *,
    sem: FloatArray | None = None,
    highlight: Sequence[str] = (),
    window_s: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Draw the trial-averaged response, with event onsets marked.

    Parameters
    ----------
    time_s : FloatArray of shape (T,)
        Time axis of the analysis window.
    average : FloatArray of shape (T, N)
        Trial-averaged signal.
    regions : RegionSet
        Region labels.
    onsets_s : sequence of float
        Event onsets to mark.
    sem : FloatArray of shape (T, N), optional
        Standard error across trials, drawn as a band.
    highlight : sequence of str
        Region codes drawn in colour.
    window_s : tuple of float, optional
        Absolute time window to draw; ``None`` shows the whole record.

    Returns
    -------
    plotly.graph_objects.Figure
        Overlaid responses with a shaded error band where available.
    """
    if window_s is not None:
        shown = (time_s >= window_s[0]) & (time_s <= window_s[1])
        time_s, average = time_s[shown], average[shown]
        sem = sem[shown] if sem is not None else None
    styles = _traces_for(regions, highlight)
    figure = go.Figure()
    for index, code in enumerate(regions.codes):
        style = styles.get(code)
        if style is None:
            continue
        colour, dash = style
        if sem is not None and np.all(np.isfinite(sem[:, index])):
            band = np.concatenate([average[:, index] + sem[:, index], (average[:, index] - sem[:, index])[::-1]])
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([time_s, time_s[::-1]]),
                    y=band,
                    fill="toself",
                    fillcolor=_translucent(colour),
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=code,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=time_s,
                y=average[:, index],
                name=code,
                legendgroup=code,
                mode="lines",
                line={"color": colour, "width": 1.6, "dash": _DASH[dash]},
                hovertemplate=f"<b>{code}</b><br>%{{x:.3f}} s<br>%{{y:.3g}}<extra></extra>",
            )
        )
    for onset in onsets_s:
        if window_s is None or window_s[0] <= onset <= window_s[1]:
            figure.add_vline(x=float(onset), line={"color": "#444444", "width": 1, "dash": "dot"})
    return _style(figure, height=380, x_title="Time (s)", y_title="Trial-averaged activity", has_legend=True)


def sweep_figure(
    values: Sequence[Any],
    curves: Mapping[str, Sequence[float]],
    *,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """
    Draw how a measure moves as one parameter is swept.

    Parameters
    ----------
    values : sequence
        The swept values, in order.
    curves : Mapping[str, sequence of float]
        One named curve per measure or region.
    x_title, y_title : str
        Axis labels.

    Returns
    -------
    plotly.graph_objects.Figure
        Markers joined by lines, so both the trend and the sampled points are
        visible.
    """
    styles = region_styles(len(curves))
    figure = go.Figure()
    for position, (name, series) in enumerate(curves.items()):
        colour, dash = styles[position]
        figure.add_trace(
            go.Scatter(
                x=list(values),
                y=list(series),
                name=name,
                mode="lines+markers",
                line={"color": colour, "width": 1.6, "dash": _DASH[dash]},
                marker={"size": 7},
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.4g}}<extra></extra>",
            )
        )
    return _style(figure, height=380, x_title=x_title, y_title=y_title, has_legend=True)


def histogram_figure(
    values: FloatArray,
    *,
    x_title: str,
    bins: int = 30,
    reference: float | None = None,
    reference_label: str = "",
) -> go.Figure:
    """
    Draw a histogram, optionally with a reference line.

    Parameters
    ----------
    values : FloatArray
        Values to bin.
    x_title : str
        Axis label.
    bins : int
        Number of bins.
    reference : float, optional
        Value to mark with a vertical line, e.g. the median.
    reference_label : str
        Annotation for that line.

    Returns
    -------
    plotly.graph_objects.Figure
        Histogram with the count on hover.
    """
    figure = go.Figure(
        go.Histogram(
            x=np.asarray(values, dtype=float),
            nbinsx=bins,
            marker={"color": PALETTE[0], "line": {"color": "white", "width": 0.5}},
            hovertemplate=f"{x_title} %{{x}}<br>%{{y}} connections<extra></extra>",
        )
    )
    if reference is not None:
        figure.add_vline(
            x=float(reference),
            line={"color": "#444444", "width": 1.5, "dash": "dash"},
            annotation_text=reference_label,
            annotation_position="top right",
        )
    return _style(figure, height=320, x_title=x_title, y_title="Connections")


def _translucent(colour: str, alpha: float = 0.18) -> str:
    """Return a hex colour as an ``rgba`` string with the given opacity."""
    value = colour.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
