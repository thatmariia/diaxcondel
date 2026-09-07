"""
Ready-made experiment specifications.

Presets are starting points, not policy: each one is an ordinary
:class:`~diaxcondel.experiment.spec.ExperimentSpec` that can be inspected,
edited, and saved. They exist so that a new user (or a new dashboard session)
begins from something that runs and means something, rather than from an
empty form.
"""

from __future__ import annotations

from collections.abc import Callable

from .spec import ComponentRef, EventRef, ExperimentSpec


def reference_resting() -> ExperimentSpec:
    """
    Resting state in the five-region model of the original paper.

    Lobe-scale distances with the paper's tuned coupling, white noise, and no
    stimulus — the configuration whose occipital alpha peak comes from
    transmission delays alone.

    Returns
    -------
    ExperimentSpec
        Reference resting-state experiment.
    """
    return ExperimentSpec(label="reference resting state")


def thalamic_relay() -> ExperimentSpec:
    """
    Relay every cortico-cortical path of the five-region model via thalamus.

    Delays become the sum of two legs, which the builder handles by switching
    to the matching two-leg kernel.

    Returns
    -------
    ExperimentSpec
        Relayed variant, shortened because the relay kernel is Monte-Carlo.
    """
    return ExperimentSpec(relay_code="T", label="thalamic relay").with_simulation(sim_seconds=5.0)


def local_recurrence() -> ExperimentSpec:
    """
    Five-region model with postsynaptic delay and recurrent self-excitation.

    Adds the two ingredients the original model leaves out: a few
    milliseconds of processing inside each region, and each region exciting
    itself through that delay.

    Returns
    -------
    ExperimentSpec
        Reference model with local delay and self-excitation.
    """
    return ExperimentSpec(
        local_delay=ComponentRef(name="gamma", params={"mean_ms": 3.0, "shape": 4.0}),
        self_weight=0.2,
        label="local recurrence",
    )


def atlas_lobes() -> ExperimentSpec:
    """
    Lobe-scale model built from the Julich-Brain atlas.

    Streamline lengths give the distances and streamline counts the coupling,
    with atlas regions merged at the lobe level. Sampled at 500 Hz with 150
    lags, which still covers 300 ms of delay at half the cost.

    Returns
    -------
    ExperimentSpec
        Atlas-derived experiment (requires the ``atlas`` extra).
    """
    spec = ExperimentSpec(
        distances=ComponentRef(name="siibra", params={"parcellation": "julich 3.1", "merge_level": 3}),
        connectivity=ComponentRef(name="siibra", params={"source": "streamline_counts"}),
        label="atlas lobes",
    )
    return spec.with_simulation(sample_rate_hz=500, n_lags=150, sim_seconds=20.0, burnin_seconds=1.0)


def atlas_cortical_gyri() -> ExperimentSpec:
    """
    Cortex-only model at gyrus scale, from the Julich-Brain atlas.

    Keeps the cortical branch of the atlas at a finer merge level, so
    connection lengths span a much wider range than the lobe-scale model —
    the natural next step when asking how spatial scale changes the spectrum.

    Returns
    -------
    ExperimentSpec
        Roughly seventy-region cortical experiment (requires the ``atlas``
        extra).
    """
    spec = ExperimentSpec(
        distances=ComponentRef(
            name="siibra",
            params={"parcellation": "julich 3.1", "merge_level": 4, "keep_parts": ("cerebral cortex",)},
        ),
        connectivity=ComponentRef(name="siibra", params={"source": "streamline_counts", "threshold": 0.02}),
        label="atlas cortical gyri",
    )
    return spec.with_simulation(sample_rate_hz=250, n_lags=75, sim_seconds=20.0, burnin_seconds=1.0)


def wide_length_range() -> ExperimentSpec:
    """
    Synthetic network with connection lengths from half a centimetre to 25 cm.

    Log-uniform lengths (equal weight per decade of length), uniform coupling
    tuned close to the stability boundary, and a saturating transfer so the
    network stays bounded there.

    Returns
    -------
    ExperimentSpec
        Twenty-node synthetic experiment.
    """
    spec = ExperimentSpec(
        distances=ComponentRef(
            name="power_law",
            params={"n_regions": 20, "beta": -1.0, "d_min_cm": 0.5, "d_max_cm": 25.0},
        ),
        connectivity=ComponentRef(name="uniform", params={"coupling": 0.98}),
        transfer=ComponentRef(name="centred_sigmoid", params={"slope": 4.0}),
        enforce_stationarity=False,
        label="wide length range",
    )
    return spec.with_simulation(sample_rate_hz=500, n_lags=200, sim_seconds=30.0, burnin_seconds=2.0)


def cortex_with_hub() -> ExperimentSpec:
    """
    Synthetic cortex with a wide range of lengths, plus one subcortical hub.

    Cortico-cortical connections span many lengths, while every region also
    connects to a single hub at a well-defined distance — a way to combine
    broadly spread and sharply timed delays in one network.

    Returns
    -------
    ExperimentSpec
        Synthetic experiment with a hub node called ``HUB``.
    """
    spec = ExperimentSpec(
        distances=ComponentRef(
            name="power_law_with_hub",
            params={"n_regions": 16, "beta": -1.0, "d_min_cm": 0.5, "d_max_cm": 25.0, "hub_distance_cm": 6.0},
        ),
        connectivity=ComponentRef(name="uniform", params={"coupling": 0.9}),
        label="cortex with hub",
    )
    return spec.with_simulation(sample_rate_hz=500, n_lags=200, sim_seconds=20.0, burnin_seconds=2.0)


def erp_occipital_pulse() -> ExperimentSpec:
    """
    Event-related run: a brief pulse into the occipital lobe, averaged over trials.

    The response shape comes from the network's own delays rather than from
    the stimulus waveform.

    Returns
    -------
    ExperimentSpec
        Twenty-trial event-related experiment.
    """
    event = EventRef(
        onset_seconds=0.5,
        drive=ComponentRef(name="gaussian_pulse", params={"targets": ("OL",), "amplitude": 20.0}),
        name="visual pulse",
    )
    spec = ExperimentSpec(events=(event,), n_trials=20, label="occipital ERP")
    return spec.with_simulation(sim_seconds=2.0, burnin_seconds=1.0)


def entrainment() -> ExperimentSpec:
    """
    Rhythmic drive at 10 Hz into one region, to see how far it spreads.

    Drives the network at a fixed frequency and lets the delays decide which
    regions follow it.

    Returns
    -------
    ExperimentSpec
        Ten-trial entrainment experiment.
    """
    event = EventRef(
        onset_seconds=0.5,
        drive=ComponentRef(
            name="sine_burst",
            params={"targets": ("OL",), "frequency_hz": 10.0, "amplitude": 10.0, "duration_seconds": 2.0},
        ),
        name="10 Hz drive",
    )
    spec = ExperimentSpec(events=(event,), n_trials=10, label="entrainment")
    return spec.with_simulation(sim_seconds=4.0, burnin_seconds=1.0)


def criticality_sweep() -> ExperimentSpec:
    """
    Five-region model placed just below the stability boundary.

    Peak sharpness is governed by how close the network runs to instability.
    Fixing ``target_spectral_radius`` makes that a variable of its own: every
    coupling is scaled by one factor, so relative connectivity and every delay
    stay as they are. Sweep it — ``sweep(spec, {"target_spectral_radius":
    [0.85, 0.93, 0.97, 0.995]})`` — and only the operating point changes.

    Returns
    -------
    ExperimentSpec
        Reference model held at a spectral radius of 0.97.
    """
    spec = ExperimentSpec(target_spectral_radius=0.97, label="near-critical")
    return spec.with_simulation(sim_seconds=30.0, burnin_seconds=2.0)


def saturating_supercritical() -> ExperimentSpec:
    """
    Model driven past the linear boundary, held bounded by a saturating transfer.

    Above a spectral radius of one a linear model diverges. A saturating
    transfer bounds activity instead, which is the only way to reach the
    sharpest resonances. There is no closed-form spectrum here, so the
    spectrum comes from the simulation alone.

    Returns
    -------
    ExperimentSpec
        Supercritical model with a centred-sigmoid transfer.
    """
    spec = ExperimentSpec(
        transfer=ComponentRef(name="centred_sigmoid"),
        target_spectral_radius=1.05,
        enforce_stationarity=False,
        label="supercritical, saturating",
    )
    return spec.with_simulation(sim_seconds=30.0, burnin_seconds=2.0)


def shared_input_control() -> ExperimentSpec:
    """
    Uncoupled regions driven by partly shared input.

    Coherence between regions is usually read as evidence of a connection.
    Shared input produces it without one. With the coupling switched off,
    whatever coherence remains is the baseline that any coupled model has to
    beat before its own coherence means anything.

    Returns
    -------
    ExperimentSpec
        Five regions, no coupling, 40% shared input.
    """
    spec = ExperimentSpec(
        connectivity=ComponentRef(name="uniform", params={"coupling": 0.0}),
        noise=ComponentRef(name="correlated", params={"correlation": 0.4}),
        label="shared input only",
    )
    return spec.with_simulation(sim_seconds=30.0, burnin_seconds=2.0)


def frequency_sweep() -> ExperimentSpec:
    """
    Chirp into the occipital lobe, tracing the network's frequency response.

    A sine burst measures the response at one frequency; a sweep measures it
    at all of them in a single trial. Where the response grows, the drive met
    a delay-driven resonance.

    Returns
    -------
    ExperimentSpec
        Ten-trial run with a 2-40 Hz sweep.
    """
    event = EventRef(
        onset_seconds=0.5,
        drive=ComponentRef(
            name="chirp",
            params={
                "targets": ("OL",),
                "start_hz": 2.0,
                "end_hz": 40.0,
                "amplitude": 10.0,
                "duration_seconds": 8.0,
            },
        ),
        name="2-40 Hz sweep",
    )
    spec = ExperimentSpec(events=(event,), n_trials=10, label="frequency sweep")
    return spec.with_simulation(sim_seconds=10.0, burnin_seconds=1.0)


#: Named starting points, in the order a newcomer should meet them.
PRESETS: dict[str, Callable[[], ExperimentSpec]] = {
    "reference_resting": reference_resting,
    "thalamic_relay": thalamic_relay,
    "local_recurrence": local_recurrence,
    "atlas_lobes": atlas_lobes,
    "atlas_cortical_gyri": atlas_cortical_gyri,
    "wide_length_range": wide_length_range,
    "cortex_with_hub": cortex_with_hub,
    "erp_occipital_pulse": erp_occipital_pulse,
    "entrainment": entrainment,
    "frequency_sweep": frequency_sweep,
    "criticality_sweep": criticality_sweep,
    "saturating_supercritical": saturating_supercritical,
    "shared_input_control": shared_input_control,
}


def preset_names() -> tuple[str, ...]:
    """Return the available preset names."""
    return tuple(PRESETS)


def get_preset(name: str) -> ExperimentSpec:
    """
    Return a preset specification by name.

    Parameters
    ----------
    name : str
        One of :func:`preset_names`.

    Returns
    -------
    ExperimentSpec
        A fresh copy of the preset.

    Raises
    ------
    KeyError
        If the name is unknown; the message lists the valid ones.
    """
    try:
        return PRESETS[name]()
    except KeyError as exc:
        raise KeyError(f"unknown preset {name!r}; available: {list(PRESETS)}") from exc


def preset_summary(name: str) -> str:
    """
    Return the one-line description of a preset.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    str
        First line of the preset function's docstring.
    """
    doc = (PRESETS[name].__doc__ or "").strip().splitlines()
    return doc[0].strip() if doc else ""
