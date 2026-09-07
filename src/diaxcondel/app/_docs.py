"""
Explanatory text for the playground.

Every figure in the dashboard says how it was made, every component says what
it does to the model, and every term that could be jargon is defined once in
the glossary. Keeping that text here, instead of scattering it through the
page code, makes it reviewable as prose and keeps the page itself readable.
"""

from __future__ import annotations

from collections.abc import Mapping

from diaxcondel.catalog.registry import SLOT_INFO

MODEL_OVERVIEW = """
Each region carries one number over time: how strongly it is being driven by
the rest of the brain. A region's activity is the sum of what every other
region sent it, delayed by the time the signal needed to travel, plus ongoing
random input:

$$H_n(t) = \\sum_{m \\neq n} \\sum_p \\gamma_{nm} w_{nm}(p) F(H_m(t - p \\Delta t)) + \\epsilon_n(t)$$

* $\\gamma_{nm}$: how strongly region $m$ drives region $n$ (**connectivity**).
* $w_{nm}(p)$: how much of that drive arrives after $p$ samples (**delay kernel**).
  It follows from the connection's length, the spread of **axon calibres** in the
  bundle, and the speed those calibres imply: $v = k \\phi$, so an axon of diameter
  $\\phi$ delivers the signal after $d / (k \\phi)$.
* $F$: the **transfer**. Identity keeps the model linear, a saturating
  sigmoid bounds it.
* $\\epsilon_n(t)$: the ongoing **noise** standing in for everything local.

There is no oscillator anywhere in that equation. Rhythms appear because a
signal that leaves a region comes back after a round trip of twice the
transmission delay, and frequencies near $1/(2\\tau)$ reinforce themselves.
Which rhythms a network produces is therefore a question about its geometry:
how far apart its regions are, and how fast its axons conduct.
""".strip()

BUILD_PIPELINE = """
**How the model is assembled.** The distance source fixes the regions and the
length of every connection. The delay kernel turns each length into a
distribution over delays, which is discretised onto the simulation's lag
grid. Multiplying by the connectivity weight gives the lag tensor $\\Phi$,
one number per (lag, target, source). Local delay, if any, is convolved into
every connection; self-excitation, if any, is added to the diagonal. The
result is checked for stability, simulated with the chosen noise, and
analysed.
""".strip()

#: What each catalog slot does to the model, in the dashboard's own words.
#: What each slot changes about the model, read straight from the catalog so
#: the dashboard and the command line cannot describe a slot differently.
SLOT_HELP: Mapping[str, str] = {slot: info.detail for slot, info in SLOT_INFO.items()}

#: Terms the dashboard uses that deserve a one-line definition.
GLOSSARY: tuple[tuple[str, str], ...] = (
    (
        "Lag tensor (Φ)",
        "One coefficient per (lag, target region, source region): how much of a source's past "
        "activity reaches a target after that many samples. Everything the model does is in here.",
    ),
    (
        "Spectral radius (ρ)",
        "How much the network amplifies its own echo. Below 1 activity decays between round trips "
        "and the model is stable; at 1 it is critical, with the sharpest spectral peaks; above 1 a "
        "linear model diverges, and only a saturating transfer keeps it bounded.",
    ),
    (
        "Operating point",
        "The spectral radius you ask the model to sit at. Setting it scales every coupling by one "
        "common factor, leaving relative connectivity and all delays untouched, so peak sharpness "
        "can be varied on its own. The closer to one, the sharper the peaks.",
    ),
    (
        "Shared input",
        "The fraction of each region's random input that is common to all of them. Regions driven "
        "by the same signal are coherent without being connected, so it is the control for asking "
        "how much of a coherence pattern the connections explain.",
    ),
    (
        "Stationarity shrink",
        "If the coupling is too strong, the builder repeatedly scales the whole lag tensor by 0.9 "
        "until the linear system is stable, and reports how much it shrank.",
    ),
    (
        "Burn-in",
        "The start of each simulation, discarded before analysis, so the measured signal does not "
        "include the transient from starting at zero. Should exceed the longest delay.",
    ),
    (
        "Delay coverage / lags",
        "The longest delay the model can represent. Delays beyond it are truncated, so it must "
        "comfortably exceed the longest connection's travel time.",
    ),
    (
        "Welch spectrum",
        "The power spectrum estimated from the simulated signal: cut it into overlapping windowed "
        "segments, take each segment's spectrum, average them. Noisy but assumption-free.",
    ),
    (
        "Closed-form spectrum",
        "The spectrum computed directly from the model's coefficients rather than from a "
        "simulation. Exact and noise-free, but only available for a linear model driven by white noise.",
    ),
    (
        "Coherence",
        "How consistently two regions keep the same phase relationship at a given frequency, from 0 to 1.",
    ),
    (
        "Aperiodic slope",
        "The straight-line slope of the spectrum on log-power against log-frequency axes, "
        "describing the broadband background under any peaks.",
    ),
    (
        "Streamline length / count",
        "Tractography reconstructs fibre bundles between regions: their average length is used as "
        "the conduction distance, and how many were found is used as connection strength.",
    ),
    (
        "Merge level",
        "How far down an atlas's own region tree the model keeps detail. Low levels give few large "
        "regions, high levels many small ones.",
    ),
    (
        "Merged pathway",
        "When atlas regions are merged, the connection between two merged regions stands for many "
        "finer ones of different lengths. The model keeps those lengths and mixes their delays, "
        "weighted by how many fibres each carries, so a merged pathway's delay follows where its "
        "fibres actually are, not the average of its parts.",
    ),
    (
        "Multitaper spectrum",
        "A spectrum estimated by averaging over several orthogonal tapers of the whole recording, "
        "rather than over segments. Smoother than Welch at the same frequency resolution.",
    ),
    (
        "Spectral parameterisation",
        "Fitting a spectrum as a smooth background plus a set of peaks, so the background exponent "
        "is not distorted by the peaks sitting on it. Used here when the optional `specparam` "
        "package is installed.",
    ),
    (
        "Display smoothing",
        "A moving average applied to a figure only: over time for traces, over frequency for "
        "spectra. It never changes the model, the fits, or the reported numbers.",
    ),
    (
        "Relay region",
        "A region every other connection is routed through, so a signal's delay is the sum of two "
        "legs instead of one direct hop.",
    ),
)

#: Published sources behind the model and its data.
REFERENCES: tuple[tuple[str, str], ...] = (
    (
        "Steeghs-Turchina, Srinivasan, Nunez & Nunez (2025). Slow wave dynamics of scalp EEG can be "
        "explained by simple statistical models of long-range connections. NeuroImage 321:121418.",
        "https://doi.org/10.1016/j.neuroimage.2025.121418",
    ),
    (
        "Nunez (1974). The brain wave equation: a model for the EEG. Mathematical Biosciences 21:279–297.",
        "https://doi.org/10.1016/0025-5564(74)90020-0",
    ),
    (
        "Domhof, Jung, Eickhoff & Popovych (2022). Parcellation-based structural and resting-state "
        "functional brain connectomes of a healthy cohort.",
        "https://doi.org/10.25493/NVS8-XS5",
    ),
    (
        "Hursh (1939). Conduction velocity and diameter of nerve fibers. American Journal of Physiology 127:131–139.",
        "https://doi.org/10.1152/ajplegacy.1939.127.1.131",
    ),
    (
        "Sepehrband et al. (2016). Parametric probability distribution functions for axon diameters "
        "of corpus callosum. Frontiers in Neuroanatomy 10:59.",
        "https://doi.org/10.3389/fnana.2016.00059",
    ),
)


# ---------------------------------------------------------------------------
# Figure notes: how each plot was made, with the settings actually used
# ---------------------------------------------------------------------------


def connectome_note(n_regions: int, density: float) -> str:
    """Describe the connectome matrices figure."""
    return (
        f"Both matrices as built: {n_regions} regions, {density:.0%} of possible pairs connected. "
        "Row *i*, column *j* is the connection from region *j* to region *i*; blank cells are pairs "
        "with no connection. Distances are the conduction lengths the delay kernel uses; weights are "
        "the coupling strengths, after any normalisation and scaling."
    )


def length_histogram_note(n_connections: int) -> str:
    """Describe the connection-length histogram."""
    return (
        f"Distribution of the {n_connections} connection lengths that carry non-zero weight. "
        "Each length becomes a delay, so this is the raw material the network turns into "
        "frequencies: short connections act fast, long ones slowly."
    )


def lag_kernel_note(dt_ms: float, n_lags: int) -> str:
    """Describe the lag-kernel figure."""
    return (
        f"One curve per connection: the lag tensor Φ along the delay axis, sampled every "
        f"{dt_ms:.1f} ms out to {n_lags} lags. The curve's area is the connection's total coupling "
        "and its peak is the delay at which most of the signal arrives. Local delay, if any, has "
        "already been convolved in."
    )


def delay_distribution_note() -> str:
    """Describe the delay-versus-length scatter."""
    return (
        "Every connection's peak arrival time against its length, with the frequency each implies "
        "through the round-trip relation f ≈ 1/(2τ). A network whose lengths cluster produces "
        "delays that cluster, and so favours a narrow band of frequencies."
    )


def signal_note(sample_rate_hz: int, window: tuple[float, float], trial: int, n_trials: int) -> str:
    """Describe the time-series figure."""
    return (
        f"Simulated activity for each region, {window[0]:.1f}–{window[1]:.1f} s of trial {trial} of "
        f"{n_trials}, sampled at {sample_rate_hz} Hz. Traces are offset vertically to separate them; "
        "burn-in has already been discarded."
    )


def spectra_note(estimator: str, segment_seconds: float, overlap: float, n_trials: int, has_analytic: bool) -> str:
    """Describe the spectra figure."""
    if estimator == "multitaper":
        method = "averaged over Slepian tapers of the whole record"
    else:
        method = f"from {segment_seconds:g} s Hann-windowed segments with {overlap:.0%} overlap"
    text = f"Left: power spectrum estimated from the simulation, {method}" + (
        f", then averaged over {n_trials} trials." if n_trials > 1 else "."
    )
    if has_analytic:
        text += (
            " Right: the same spectrum computed directly from the model coefficients, "
            "S(f) = T(f) Σ T(f)*, with T(f) = (I − Σₚ Φₚ e^(−2πifpΔt))⁻¹ and Σ the noise covariance. "
            "It is noise-free, and the two panels are on the same scale, so they should agree."
        )
    return text


def summary_table_note(alpha_band: tuple[float, float], slope_band: tuple[float, float], from_analytic: bool) -> str:
    """Describe the per-region summary table."""
    source = "the closed-form spectrum" if from_analytic else "the simulated spectrum"
    return (
        f"Computed from {source}. **Peak** is the strongest frequency between "
        f"{alpha_band[0]:g} and {alpha_band[1]:g} Hz. **1/f exponent** and **slope** both describe "
        f"the broadband background over {slope_band[0]:g}–{slope_band[1]:g} Hz: the slope is a plain "
        "straight-line fit of log power against log frequency, the exponent its counterpart with the "
        "sign flipped. **Variance** is the variance of the region's time series. "
    )


def aperiodic_method_note(methods: list[str]) -> str:
    """Say which route produced the aperiodic exponent."""
    if "specparam" in methods:
        return (
            "The exponent comes from spectral parameterisation, which fits peaks and background "
            "together, so it is not simply the negated slope. A slope is pulled by any peak "
            "sitting on the background, the exponent is not."
        )
    return (
        "The exponent is the negated straight-line slope. Install the optional `specparam` package "
        "to separate peaks from the background before estimating it."
    )


def network_note(n_regions: int, distortion: float) -> str:
    """Describe the network layout figure."""
    return (
        f"The {n_regions} regions placed so that drawn separations follow conduction distance "
        "(stress majorisation of the distance matrix, with unmeasured pairs filled in by their "
        "shortest path through the network). Axes carry no anatomical meaning, and drawn "
        f"separations differ from the real distances by {distortion:.0%} at the median, partly "
        "because brain-wide fibre lengths do not fit exactly in a plane, partly because crowded "
        "regions are pushed apart to keep labels readable. Line darkness and thickness follow "
        "coupling strength; node size follows the number of connections."
    )


def mixture_note(n_members: int, ratio: float) -> str:
    """Explain what merged connections do to the delays."""
    comparison = "about the same as" if 0.95 <= ratio <= 1.05 else ("longer than" if ratio > 1 else "shorter than")
    return (
        f"These regions are merged, so each connection above stands for many finer ones, "
        f"{n_members:,} in total. Each finer connection keeps its own length and produces its own "
        f"delay distribution; those are mixed, weighted by "
        f"how many fibres carry each. The fibre-weighted length is typically {ratio:.2f}× the plain "
        f"geometric mean shown in the matrix, i.e. {comparison} it, which is why the delays below "
        "need not match what the distance matrix alone would suggest."
    )


def smoothing_suffix_time(window_ms: float) -> str:
    """Describe time-domain display smoothing, if any."""
    if window_ms <= 0:
        return " No smoothing: this is the raw simulation."
    return (
        f" Smoothed for display with a {window_ms:.0f} ms moving average, which removes structure "
        f"above roughly {1000 / window_ms:.0f} Hz."
    )


def smoothing_suffix_frequency(bandwidth_hz: float) -> str:
    """Describe frequency-domain display smoothing, if any."""
    if bandwidth_hz <= 0:
        return " No smoothing: this is the raw estimate."
    return f" Smoothed for display with a {bandwidth_hz:g} Hz moving average across frequency."


def coherence_note(segment_seconds: float) -> str:
    """Describe the coherence figure."""
    return (
        f"Magnitude-squared coherence between region pairs, from {segment_seconds:g} s segments of "
        "the first trial. It measures how consistently two regions hold a phase relationship at "
        "each frequency, from 0 (unrelated) to 1 (perfectly locked)."
    )


def erp_note(n_trials: int, baseline_end: float | None) -> str:
    """Describe the event-related figure."""
    baseline = f" Each trace is baseline-corrected against 0–{baseline_end:.2f} s." if baseline_end else ""
    return (
        f"Average across {n_trials} trials, so activity that is not time-locked to the event averages "
        f"away and what remains is the network's response.{baseline} Shading is the standard error "
        "across trials; dashed lines mark event onsets."
    )


def sweep_note(path: str, n_points: int, metric: str) -> str:
    """Describe the sweep figure."""
    return (
        f"The model was rebuilt and simulated {n_points} times, varying **{path}** and holding "
        f"everything else fixed (including the random seed, so differences come from the parameter "
        f"rather than from noise). The line shows {metric} at each value."
    )


REGION_PICKER = (
    "Each row picks regions from the atlas and says how far to split them: **split by 0** keeps a "
    "region whole, **1** makes a node per immediate sub-region. Rows can name regions from "
    "different levels, so a lobe and a whole nucleus can stand side by side. Anything not named, "
    "or not inside something named, is left out of the model. The level filter only shortens the "
    "list beside it."
)


def hierarchy_note(parcellation: str) -> str:
    """Describe the atlas merge-level table."""
    return (
        f"What each level of {parcellation} holds: how many regions sit there, and the first few "
        "of their names. The atlas defines this tree, so a level means different things in "
        "different atlases; this table is read from the atlas itself."
    )
