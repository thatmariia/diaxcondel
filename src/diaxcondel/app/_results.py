"""
The results half of the playground: what the model did.

Each view states how its numbers were produced (which estimator, which
window, how many trials), so a figure taken from here can be read, and
reproduced, without guessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from diaxcondel import catalog
from diaxcondel.catalog.param import ParamField
from diaxcondel.connectome.bundle import Connectome, region_metadata, source_metadata
from diaxcondel.erp.analysis import baseline_correct, peak_amplitude, peak_latency
from diaxcondel.experiment import (
    ALPHA_BAND_HZ,
    SLOPE_BAND_HZ,
    BuiltModel,
    ExperimentSpec,
    delay_statistics,
    parameter_paths,
    region_summaries,
    spec_to_python,
)

from . import _docs as docs, _figures as figures
from ._charts import INTERACTIVE, show

if INTERACTIVE:  # pragma: no cover - exercised only when plotly is installed
    from . import _interactive as live
from ._state import (
    cached_coherence,
    cached_model,
    cached_run,
    cached_spectra,
    cached_sweep,
    sweep_values,
)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def network_tab(spec_json: str, model: BuiltModel) -> None:
    """
    Show the connectome, its length distribution, and the delays it produces.

    Parameters
    ----------
    spec_json : str
        Serialised specification (for cache lookups).
    model : BuiltModel
        The built model.
    """
    connectome: Connectome = model.connectome
    summary = connectome.describe()
    weights = connectome.weight_matrix()
    distances = connectome.distance_matrix()
    n_connections = int(np.count_nonzero(np.triu(weights > 0, k=1)))

    columns = st.columns(4)
    columns[0].metric("regions", summary["n_regions"])
    columns[1].metric("connections", n_connections, help="Region pairs with a non-zero weight.")
    columns[2].metric(
        "length range",
        f"{summary['distance_cm_min']:.1f}–{summary['distance_cm_max']:.1f} cm",
        help="Shortest and longest connection that carries weight.",
    )
    columns[3].metric("strongest weight", f"{summary['weight_max']:.2f}")

    views = st.tabs(["Layout", "Matrices"])
    with views[0]:
        spread = st.slider(
            "spread crowded regions",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Pushes regions apart so labels stay readable, at the cost of drawing their "
            "separations less faithfully. Zero is the honest layout; the caption reports how much "
            "the picture departs from the real distances.",
        )
        figure, distortion = figures.network_figure(connectome, separation_fraction=spread)
        show(
            (lambda: live.network_figure(_layout_coordinates(connectome, spread), connectome)) if INTERACTIVE else None,
            lambda: figure,
            key="dx.chart.network",
        )
        st.caption(docs.network_note(summary["n_regions"], distortion))
    with views[1]:
        show(
            (lambda: live.connectome_figure(connectome)) if INTERACTIVE else None,
            lambda: figures.connectome_figure(connectome),
            key="dx.chart.matrices",
        )
        st.caption(docs.connectome_note(summary["n_regions"], float(summary["density"])))

    mixture = getattr(connectome.distances, "mixture", None)
    if mixture is not None and len(mixture):
        effective = mixture.effective_lengths(connectome.n_regions)
        shown = (effective > 0) & (distances > 0)
        ratio = float(np.median(effective[shown] / distances[shown])) if np.any(shown) else 1.0
        st.info(docs.mixture_note(len(mixture), ratio))

    stats = delay_statistics(model)
    left, right = st.columns(2, gap="large")
    with left:
        connected = np.triu(distances, k=1) > 0
        lengths = distances[np.triu(np.ones_like(distances, dtype=bool), k=1) & (distances > 0)]
        show(
            (
                (
                    lambda: live.histogram_figure(
                        lengths,
                        x_title="Connection length (cm)",
                        reference=float(np.median(lengths)) if lengths.size else None,
                        reference_label="median",
                    )
                )
                if INTERACTIVE
                else None
            ),
            lambda: figures.distance_histogram_figure(connectome),
            key="dx.chart.lengths",
        )
        st.caption(docs.length_histogram_note(n_connections))
    with right:
        pairs = (distances > 0) & (stats.coupling > 0)
        codes = list(connectome.codes)
        rows_i, cols_i = np.nonzero(np.triu(pairs, k=1))
        show(
            (
                (
                    lambda: live.scatter_figure(
                        distances[rows_i, cols_i],
                        stats.peak_delay_ms[rows_i, cols_i],
                        labels=[f"{codes[i]} — {codes[j]}" for i, j in zip(rows_i, cols_i, strict=True)],
                        x_title="Connection length (cm)",
                        y_title="Most likely delay (ms)",
                        colour=stats.coupling[rows_i, cols_i],
                        colour_title="coupling",
                    )
                )
                if INTERACTIVE and rows_i.size
                else None
            ),
            lambda: figures.delay_scatter_figure(distances, stats.peak_delay_ms, stats.coupling),
            key="dx.chart.delay_scatter",
        )
        st.caption(docs.delay_distribution_note())
    del connected

    described = stats.describe()
    if described:
        metrics = st.columns(4)
        metrics[0].metric("mean delay", f"{described['mean delay (ms)']:.0f} ms")
        metrics[1].metric(
            "delay range",
            f"{described['shortest delay (ms)']:.0f}–{described['longest delay (ms)']:.0f} ms",
        )
        metrics[2].metric(
            "implied frequencies",
            f"{described['lowest implied frequency (Hz)']:.1f}–{described['highest implied frequency (Hz)']:.1f} Hz",
            help="Frequencies the round-trip delays favour, f ≈ 1/(2τ).",
        )
        metrics[3].metric("total coupling", f"{described['total coupling']:.2f}")

    with st.expander("Lag kernels — the delays the simulation actually uses"):
        codes = model.regions.codes
        target = st.selectbox("arriving at", options=codes, key="dx.lag.target")
        strongest = np.argsort(stats.coupling[codes.index(target)])[::-1]
        default_sources = [codes[int(i)] for i in strongest[:3] if codes[int(i)] != target]
        sources = st.multiselect(
            "coming from",
            options=[code for code in codes if code != target],
            default=default_sources,
            key="dx.lag.sources",
        )
        if sources:
            pairs_shown = [(target, source) for source in sources]
            show(
                (lambda: _lag_kernel_chart(model, pairs_shown)) if INTERACTIVE else None,
                lambda: figures.lag_kernel_figure(model.phi, model.regions, model.params.dt_s, pairs_shown),
                key="dx.chart.lag_kernels",
            )
            st.caption(docs.lag_kernel_note(1000 * model.params.dt_s, model.n_lags))
        else:
            st.info("Pick at least one source region.")

    with st.expander("Regions"):
        rows: list[dict[str, Any]] = []
        for index, region in enumerate(connectome.regions.regions):
            members = region_metadata(region, "members", ())
            connected = weights[index] > 0
            rows.append(
                {
                    "code": region.code,
                    "name": region.name,
                    "merged from": len(members) if members else 1,
                    "connections": int(np.count_nonzero(connected)),
                    "mean length (cm)": (float(distances[index][connected].mean()) if np.any(connected) else 0.0),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

    with st.expander("Where these numbers came from"):
        st.json(
            {
                "distances": {k: str(v) for k, v in dict(source_metadata(connectome.distances)).items()},
                "connectivity": {k: str(v) for k, v in dict(source_metadata(connectome.weights)).items()},
            }
        )


def _layout_coordinates(connectome: Connectome, separation_fraction: float) -> Any:
    """Return node positions for the interactive network view."""
    from diaxcondel.viz.network import stress_layout

    coords, _ = stress_layout(connectome.distance_matrix(), min_separation=separation_fraction)
    return coords


def _lag_kernel_chart(model: BuiltModel, pairs: list[tuple[str, str]]) -> Any:
    """Return the interactive lag-kernel figure for the chosen connections."""
    codes = list(model.regions.codes)
    phi = model.phi
    lags_ms = np.arange(1, phi.shape[0] + 1) * 1000.0 * model.params.dt_s
    weights = np.column_stack([phi[:, codes.index(target), codes.index(source)] for target, source in pairs])
    labels = [f"{source} → {target}" for target, source in pairs]
    return live.delay_distribution_figure(lags_ms, weights, labels, max_curves=len(labels))


# ---------------------------------------------------------------------------
# Signals and spectra
# ---------------------------------------------------------------------------


def signals_tab(spec_json: str, spec: ExperimentSpec) -> None:
    """
    Show simulated time series.

    Parameters
    ----------
    spec_json : str
        Serialised specification.
    spec : ExperimentSpec
        The specification itself.
    """
    run = cached_run(spec_json)
    controls = st.columns([3, 2, 1])
    window = controls[0].slider(
        "time window (s)",
        min_value=0.0,
        max_value=float(spec.simulation.sim_seconds),
        value=(0.0, min(2.0, float(spec.simulation.sim_seconds))),
        step=0.1,
    )
    smoothing_ms = controls[1].slider(
        "smoothing (ms)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        help="Moving average applied to the traces for display only. It suppresses everything "
        "faster than roughly 1000 / window Hz, so use it to see slow structure, not to measure it.",
    )
    trial = int(
        controls[2].number_input(
            "trial",
            min_value=1,
            max_value=int(run.trials.n_trials),
            value=1,
            help="Which repetition to show; trials differ only in their noise.",
        )
    )
    show(
        (lambda: live.signal_figure(run, window, smoothing_ms=smoothing_ms, trial=trial - 1)) if INTERACTIVE else None,
        lambda: figures.signal_figure(run, window, smoothing_ms=smoothing_ms, trial=trial - 1),
        key="dx.chart.signals",
    )
    st.caption(
        docs.signal_note(spec.simulation.sample_rate_hz, window, trial, run.trials.n_trials)
        + docs.smoothing_suffix_time(smoothing_ms)
    )


def spectra_tab(spec_json: str, spec: ExperimentSpec) -> None:
    """
    Show spectra, per-region features, and coherence.

    Parameters
    ----------
    spec_json : str
        Serialised specification.
    spec : ExperimentSpec
        The specification itself.
    """
    run = cached_run(spec_json)
    controls = st.columns(4)
    estimator = controls[0].radio(
        "estimator",
        options=["welch", "multitaper"],
        horizontal=True,
        help="Welch averages the spectra of overlapping segments. The multitaper estimate averages "
        "over orthogonal tapers of the whole record instead, which is smoother at the same "
        "frequency resolution.",
    )
    segment_seconds = controls[1].slider(
        "segment length (s)",
        min_value=0.5,
        max_value=max(1.0, float(spec.simulation.sim_seconds) / 2),
        value=min(2.0, max(1.0, float(spec.simulation.sim_seconds) / 4)),
        step=0.5,
        help="Welch only: longer segments resolve frequencies more finely (about 1/segment) but average fewer of them.",
        disabled=estimator != "welch",
    )
    fmax_hz = controls[2].slider("show up to (Hz)", min_value=10.0, max_value=100.0, value=45.0, step=5.0)
    smoothing_hz = controls[3].slider(
        "smoothing (Hz)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.25,
        help="Moving average across frequency, for display only. Keep it well below the width of "
        "any peak being judged, or the peak is flattened.",
    )
    axes = st.columns(2)
    log_y = axes[0].checkbox("log power axis", value=True, help="Makes broadband structure easier to judge.")
    log_x = axes[1].checkbox(
        "log frequency axis",
        value=False,
        help="On log–log axes a broadband background falls along a straight line.",
    )

    spectra = cached_spectra(spec_json, segment_seconds, fmax_hz, estimator)
    codes = list(run.regions.codes)
    highlight: list[str] = []
    if len(codes) > 8:
        strongest = _strongest_regions(spectra, run.regions, limit=6)
        highlight = st.multiselect(
            "highlight regions",
            options=codes,
            default=strongest,
            help="With this many regions, curves are easier to read when a few are picked out. "
            "The rest are still drawn, in grey.",
        )
    for note in spectra.notes:
        st.info(note)
    spectra_args = {
        "log_y": log_y,
        "log_x": log_x,
        "fmax_hz": fmax_hz,
        "smoothing_hz": smoothing_hz,
        "highlight": highlight,
    }
    show(
        (lambda: live.spectra_figure(spectra, run.regions, **spectra_args)) if INTERACTIVE else None,
        lambda: figures.spectra_figure(spectra, run.regions, **spectra_args),
        key="dx.chart.spectra",
    )
    st.caption(
        docs.spectra_note(estimator, segment_seconds, 0.5, run.trials.n_trials, spectra.has_analytic)
        + docs.smoothing_suffix_frequency(smoothing_hz)
    )

    st.subheader("Per-region summary")
    summaries = region_summaries(run, spectra)
    methods = {item.fit_method for item in summaries}
    st.dataframe(
        [
            {
                "region": item.code,
                "peak (Hz)": round(item.peak_hz, 2),
                "power at peak": float(f"{item.peak_power:.4g}"),
                "1/f exponent": round(item.exponent, 3),
                "slope (log–log)": round(item.slope, 3),
                "fit R²": round(item.r_squared, 3),
                "variance": float(f"{item.variance:.4g}"),
            }
            for item in summaries
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        docs.summary_table_note(ALPHA_BAND_HZ, SLOPE_BAND_HZ, spectra.has_analytic)
        + docs.aperiodic_method_note(sorted(methods))
    )

    if st.checkbox("show coherence between regions", value=False):
        coherence_smoothing = st.slider(
            "coherence smoothing (Hz)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="Coherence estimated from one trial is noisy; this averages across neighbouring "
            "frequencies for display only.",
        )
        freqs, coherence = cached_coherence(spec_json, segment_seconds)
        coherence_args = {"fmax_hz": fmax_hz, "smoothing_hz": coherence_smoothing}
        show(
            (lambda: live.coherence_figure(freqs, coherence, run.regions, **coherence_args)) if INTERACTIVE else None,
            lambda: figures.coherence_figure(freqs, coherence, run.regions, **coherence_args),
            key="dx.chart.coherence",
        )
        st.caption(docs.coherence_note(segment_seconds) + docs.smoothing_suffix_frequency(coherence_smoothing))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def event_results(spec_json: str, spec: ExperimentSpec) -> None:
    """
    Show the stimulus waveforms and the trial-averaged response.

    Parameters
    ----------
    spec_json : str
        Serialised specification.
    spec : ExperimentSpec
        The specification itself.
    """
    if not spec.events:
        return
    run = cached_run(spec_json)
    model = cached_model(spec_json)

    waveforms: dict[str, np.ndarray] = {}
    onsets: dict[str, float] = {}
    for event in spec.events:
        if event.drive is None:
            continue
        drive = event.drive.build("drive", regions=model.regions)
        waveforms[event.name] = np.asarray(drive.render(float(spec.simulation.sample_rate_hz)))
        onsets[event.name] = event.onset_seconds
    if waveforms:
        show(
            (lambda: _drive_chart(waveforms, float(spec.simulation.sample_rate_hz), onsets)) if INTERACTIVE else None,
            lambda: figures.drive_figure(waveforms, float(spec.simulation.sample_rate_hz), onsets),
            key="dx.chart.drives",
        )
        st.caption("The stimulus waveforms as rendered at the current sample rate, drawn at their onsets.")

    if run.trials.n_trials < 5:
        st.warning("With so few trials the average is still mostly noise. Raise the trial count in the sidebar.")

    first_onset = min(event.onset_seconds for event in spec.events)
    sample_rate_hz = float(spec.simulation.sample_rate_hz)
    average = run.average
    baseline_end = first_onset if first_onset > 0 else None
    if baseline_end:
        average = baseline_correct(average, (0.0, baseline_end), sample_rate_hz)

    event_onsets = [event.onset_seconds for event in spec.events]
    record_end = float(spec.simulation.sim_seconds)
    default_window = figures.default_event_window(event_onsets, record_end)
    controls = st.columns(2)
    erp_window = controls[0].slider(
        "window (s)",
        min_value=0.0,
        max_value=record_end,
        value=default_window,
        step=0.05,
        help="An event-related response lives in the second or so around the onset; drawing the "
        "whole recording only compresses it out of view.",
    )
    erp_smoothing = controls[1].slider(
        "smoothing (ms)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        help="Moving average applied to the averaged response, for display only.",
    )
    show(
        (lambda: _erp_chart(run, average, event_onsets, erp_smoothing, erp_window)) if INTERACTIVE else None,
        lambda: figures.erp_figure(run, event_onsets, smoothing_ms=erp_smoothing, window_s=erp_window),
        key="dx.chart.erp",
    )
    st.caption(docs.erp_note(run.trials.n_trials, baseline_end) + docs.smoothing_suffix_time(erp_smoothing))

    search_window = (first_onset, float(spec.simulation.sim_seconds))
    amplitudes = peak_amplitude(average, search_window, sample_rate_hz)
    latencies = peak_latency(average, search_window, sample_rate_hz)
    st.dataframe(
        [
            {
                "region": code,
                "largest response": float(f"{amplitudes[index]:.4g}"),
                "at (s)": round(float(latencies[index]), 3),
                "after onset (s)": round(float(latencies[index]) - first_onset, 3),
            }
            for index, code in enumerate(run.regions.codes)
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Largest absolute deflection of the trial average after the first onset, and when it occurred. "
        "Latency after onset is the network's response time, set by the delays along the path from the "
        "stimulated region."
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

_METRICS: dict[str, tuple[str, str]] = {
    "peak (Hz)": ("peak frequency", "Peak frequency (Hz)"),
    "1/f slope": ("spectral slope", "Slope"),
    "variance": ("signal variance", "Variance"),
    "spectral radius": ("spectral radius", "Spectral radius"),
}


def sweep_tab(spec_json: str, spec: ExperimentSpec) -> None:
    """
    Run one parameter over a range and plot what changes.

    Parameters
    ----------
    spec_json : str
        Serialised specification.
    spec : ExperimentSpec
        The specification itself.
    """
    st.caption(
        "Vary one setting, hold everything else fixed, and watch a summary measure move. The random "
        "seed stays fixed too, so differences come from the parameter rather than from the noise."
    )
    paths = parameter_paths(spec)
    names = sorted(paths)
    default = "kernel.speed_factor" if "kernel.speed_factor" in names else names[0]

    top = st.columns([2, 1, 1, 1])
    path = top[0].selectbox("parameter", options=names, index=names.index(default), key="dx.sweep.path")
    field: ParamField | None = paths[path]
    current = float(np.atleast_1d(np.asarray(_current_value(spec, path), dtype=float))[0])
    low = float(field.minimum) if field and field.minimum is not None else max(0.0, current * 0.5)
    high = float(field.maximum) if field and field.maximum is not None else max(current * 2.0, current + 1.0)

    start = top[1].number_input("from", value=float(max(low, current * 0.5)), key="dx.sweep.start")
    stop = top[2].number_input("to", value=float(min(high, max(current * 1.5, current + 1))), key="dx.sweep.stop")
    count = int(top[3].number_input("steps", min_value=2, max_value=25, value=7, key="dx.sweep.count"))

    metric_name = st.selectbox("measure", options=list(_METRICS), key="dx.sweep.metric")
    integer = bool(field and field.kind == "int") or path in {"n_trials", "simulation.sample_rate_hz"}
    values = tuple(sweep_values(float(start), float(stop), count, integer=integer))

    estimate = cached_model(spec_json).cost().total_seconds * len(values)
    st.caption(f"{len(values)} runs, roughly {estimate:.0f} s in total.")
    if not st.button("Run the sweep", type="primary"):
        return

    result = cached_sweep(spec_json, path, values)
    rows = result.summary_table()
    metric_values = [row[metric_name] for row in rows]
    swept = [row[path] for row in rows]
    show(
        (
            (
                lambda: live.sweep_figure(
                    swept,
                    {_METRICS[metric_name][0]: metric_values},
                    x_title=result.axes[0].label,
                    y_title=_METRICS[metric_name][1],
                )
            )
            if INTERACTIVE
            else None
        ),
        lambda: figures.sweep_figure(
            swept,
            metric_values,
            x_label=result.axes[0].label,
            y_label=_METRICS[metric_name][1],
            reference=1.0 if metric_name == "spectral radius" else None,
        ),
        key="dx.chart.sweep",
    )
    st.caption(docs.sweep_note(path, len(values), _METRICS[metric_name][0]))
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button(
        "Download the sweep (CSV)",
        data=_to_csv(result.region_table()),
        file_name="diaxcondel_sweep.csv",
        mime="text/csv",
    )


def _drive_chart(waveforms: dict[str, np.ndarray], sample_rate_hz: float, onsets: dict[str, float]) -> Any:
    """Return the interactive stimulus-waveform figure."""
    import plotly.graph_objects as go

    from diaxcondel.viz.style import region_styles

    styles = region_styles(len(waveforms))
    figure = go.Figure()
    for position, (name, wave) in enumerate(waveforms.items()):
        colour, _ = styles[position]
        time_s = onsets[name] + np.arange(wave.size) / sample_rate_hz
        figure.add_trace(
            go.Scatter(
                x=time_s,
                y=wave,
                name=name,
                mode="lines",
                line={"color": colour, "width": 1.6},
                hovertemplate=f"<b>{name}</b><br>%{{x:.3f}} s<br>%{{y:.3g}}<extra></extra>",
            )
        )
    figure.update_layout(height=280, margin={"l": 60, "r": 20, "t": 30, "b": 45})
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Stimulus amplitude")
    return figure


def _erp_chart(
    run: Any,
    average: np.ndarray,
    onsets_s: list[float],
    smoothing_ms: float,
    window_s: tuple[float, float],
) -> Any:
    """Return the interactive trial-average figure."""
    from diaxcondel.analysis.smoothing import smooth_over_time

    sample_rate_hz = float(run.params.sample_rate_hz)
    shown = smooth_over_time(average, sample_rate_hz, smoothing_ms) if smoothing_ms > 0 else average
    sem = np.asarray(run.trials.sem, dtype=float) if run.trials.n_trials > 1 else None
    return live.erp_figure(run.time_s, shown, run.regions, onsets_s, sem=sem, window_s=window_s)


def _strongest_regions(spectra: Any, regions: Any, *, limit: int) -> list[str]:
    """Pick the regions with the most power, as a sensible default highlight."""
    power = np.asarray(spectra.welch, dtype=float)
    totals = power.sum(axis=0)
    order = np.argsort(totals)[::-1][:limit]
    return [regions.codes[int(index)] for index in sorted(order)]


def _current_value(spec: ExperimentSpec, path: str) -> Any:
    """Read a parameter, falling back to 1.0 when it cannot be read."""
    from diaxcondel.experiment import get_parameter

    try:
        return get_parameter(spec, path)
    except KeyError:  # pragma: no cover - paths come from parameter_paths
        return 1.0


def _to_csv(rows: list[dict[str, Any]]) -> str:
    """Render rows as CSV text."""
    if not rows:
        return ""
    header = list(rows[0])
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in header))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guide and reproduce
# ---------------------------------------------------------------------------


def guide_tab() -> None:
    """Explain the model, the components, and the vocabulary."""
    st.markdown(docs.MODEL_OVERVIEW)
    st.markdown(docs.BUILD_PIPELINE)

    st.subheader("The pieces you can change")
    st.caption(
        "Every option below is listed straight from the component catalog, so this page always "
        "matches what the package can actually build."
    )
    for slot, text in docs.SLOT_HELP.items():
        with st.expander(f"{slot.replace('_', ' ')} — {text}"):
            for entry in catalog.options(slot):
                st.markdown(f"**{entry.label}** — {entry.summary}")
                settings = ", ".join(f"`{field.name}`" for field in entry.fields) or "no settings"
                source = f" · {entry.reference}" if entry.reference else ""
                st.caption(f"{settings}{source}")

    st.subheader("Terms used here")
    st.dataframe(
        [{"term": term, "meaning": meaning} for term, meaning in docs.GLOSSARY],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Sources")
    for citation, url in docs.REFERENCES:
        st.markdown(f"- {citation} [link]({url})")


def reproduce_tab(spec: ExperimentSpec) -> None:
    """
    Offer the specification and a runnable script for the current view.

    Parameters
    ----------
    spec : ExperimentSpec
        The specification behind everything on the page.
    """
    st.caption(
        "Everything on this page comes from the specification below. Save it next to your results, "
        "and the run can be repeated exactly, since the seed is part of it."
    )
    columns = st.columns(2)
    columns[0].download_button(
        "Download setup (JSON)",
        data=spec.to_json(),
        file_name="diaxcondel_spec.json",
        mime="application/json",
        width="stretch",
    )
    columns[1].download_button(
        "Download script (Python)",
        data=spec_to_python(spec),
        file_name="reproduce_diaxcondel.py",
        mime="text/x-python",
        width="stretch",
    )
    st.code(spec_to_python(spec), language="python")
    with st.expander("The specification itself"):
        st.code(spec.to_json(), language="json")
    st.caption("From the command line: `diaxcondel run spec.json --output run.npz`.")
