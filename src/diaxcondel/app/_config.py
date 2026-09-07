"""
The configuration half of the playground: choosing what to model.

Two rules shape this page. First, distances and connectivity are chosen
separately, because they can come from different places: an atlas geometry
with a theoretical coupling rule, say, or hand-entered numbers from a paper.
Second, only the settings that the current choice actually uses are shown,
with the rest folded away.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st

from diaxcondel import catalog
from diaxcondel.catalog import SLOT_DESCRIPTIONS
from diaxcondel.connectome.grouping import NodeSpec, RegionChoice, describe_overlaps, parse_nodes
from diaxcondel.experiment import ComponentRef, EventRef, ExperimentSpec
from diaxcondel.experiment.presets import get_preset, preset_names, preset_summary
from diaxcondel.model.params import SimulationParams

from ._docs import REGION_PICKER, SLOT_HELP, hierarchy_note
from ._state import (
    cached_distances,
    cached_hierarchy,
    cached_native_regions,
    cached_parcellation_info,
    cached_region_choices,
)
from ._widgets import render_slot

WIDGET_PREFIX = "dx"
SEED_KEY = "dx_seed_spec"
EVENTS_KEY = "dx_events"
REGIONS_KEY = "dx_region_rows"

#: Depth the whole brain is split to when no selection has been made yet.
DEFAULT_MERGE_DEPTH = 3

#: Column widths of one row of the region-selection table.
_ROW_LAYOUT = [1.3, 6, 1.2, 0.7]
_SAMPLE_RATES = [100, 200, 250, 500, 1000, 2000]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def seed_spec() -> ExperimentSpec:
    """
    Return the specification whose values seed the controls.

    Returns
    -------
    ExperimentSpec
        The last loaded preset or uploaded spec, or the reference model.
    """
    stored = st.session_state.get(SEED_KEY)
    if isinstance(stored, str):
        return ExperimentSpec.model_validate_json(stored)
    return get_preset("reference_resting")


def apply_spec(spec: ExperimentSpec) -> None:
    """
    Reset every control to a given specification and rerun the page.

    Parameters
    ----------
    spec : ExperimentSpec
        Specification to load.
    """
    for key in [key for key in st.session_state if key.startswith(f"{WIDGET_PREFIX}.")]:
        del st.session_state[key]
    st.session_state[SEED_KEY] = spec.to_json()
    st.session_state[EVENTS_KEY] = [event.model_dump(mode="json") for event in spec.events]
    st.rerun()


def _stored_events(spec: ExperimentSpec) -> list[dict[str, Any]]:
    """Return the editable event list, seeded from the spec on first use."""
    if EVENTS_KEY not in st.session_state:
        st.session_state[EVENTS_KEY] = [event.model_dump(mode="json") for event in spec.events]
    return list(st.session_state[EVENTS_KEY])


# ---------------------------------------------------------------------------
# Sidebar: presets and simulation settings
# ---------------------------------------------------------------------------


def _simulation_controls(defaults: SimulationParams) -> SimulationParams:
    """Render the timing controls and return validated simulation settings."""
    sample_rate_hz = st.select_slider(
        "sample rate (Hz)",
        options=_SAMPLE_RATES,
        value=defaults.sample_rate_hz if defaults.sample_rate_hz in _SAMPLE_RATES else 1000,
        key=f"{WIDGET_PREFIX}.sim.sample_rate_hz",
        help="How finely time is resolved. It caps the highest frequency you can see (half the rate) "
        "and costs one lag per sample of delay coverage.",
    )
    sim_seconds = st.slider(
        "duration (s)",
        min_value=1.0,
        max_value=120.0,
        value=float(defaults.sim_seconds),
        step=1.0,
        key=f"{WIDGET_PREFIX}.sim.sim_seconds",
        help="How much signal to analyse. Longer runs give smoother spectra and finer frequency "
        "resolution (roughly 1 / segment length).",
    )
    burnin_seconds = st.slider(
        "burn-in (s)",
        min_value=0.0,
        max_value=5.0,
        value=float(defaults.burnin_seconds),
        step=0.5,
        key=f"{WIDGET_PREFIX}.sim.burnin_seconds",
        help="Discarded start of the run, so the analysed signal excludes the transient from "
        "starting at zero. Keep it longer than the longest delay.",
    )
    delay_ms = st.slider(
        "delay coverage (ms)",
        min_value=20,
        max_value=600,
        value=int(round(1000 * defaults.n_lags / defaults.sample_rate_hz)),
        step=10,
        key=f"{WIDGET_PREFIX}.sim.delay_ms",
        help="Longest delay the model can represent. Anything slower is cut off, so this must "
        "exceed the travel time of the longest connection.",
    )
    n_lags = max(1, int(round(delay_ms * sample_rate_hz / 1000)))
    st.caption(f"{n_lags} lags · frequencies up to {sample_rate_hz // 2} Hz")
    return SimulationParams(
        sample_rate_hz=sample_rate_hz,
        sim_seconds=sim_seconds,
        burnin_seconds=burnin_seconds,
        n_lags=n_lags,
    )


def sidebar(seed: ExperimentSpec) -> dict[str, Any]:
    """
    Render the sidebar: presets, timing, trials, and the random seed.

    Parameters
    ----------
    seed : ExperimentSpec
        Specification seeding the control values.

    Returns
    -------
    dict
        ``simulation``, ``n_trials``, ``seed`` and ``enforce_stationarity``.
    """
    with st.sidebar:
        st.title("diaxcondel")
        st.caption("")

        with st.expander("Start from an example", expanded=False):
            choice = st.selectbox(
                "example",
                options=preset_names(),
                format_func=lambda name: name.replace("_", " "),
                key=f"{WIDGET_PREFIX}.preset",
            )
            st.caption(preset_summary(choice))
            if st.button("Load it", width="stretch"):
                apply_spec(get_preset(choice))
            uploaded = st.file_uploader("or open a saved setup", type="json", key=f"{WIDGET_PREFIX}.upload")
            if uploaded is not None and st.button("Open file", width="stretch"):
                apply_spec(ExperimentSpec.model_validate_json(uploaded.getvalue().decode("utf-8")))

        st.header("Simulation")
        simulation = _simulation_controls(seed.simulation)
        n_trials = st.number_input(
            "trials",
            min_value=1,
            max_value=500,
            value=int(seed.n_trials),
            key=f"{WIDGET_PREFIX}.n_trials",
            help="Independent repetitions with different noise. More trials mean a cleaner "
            "event-related average and smoother spectra.",
        )
        run_seed = st.number_input(
            "random seed",
            value=int(seed.seed),
            key=f"{WIDGET_PREFIX}.seed",
            help="Fixes every random draw, so the same setup always gives the same result.",
        )
        enforce = st.checkbox(
            "keep the model stable",
            value=seed.enforce_stationarity,
            key=f"{WIDGET_PREFIX}.enforce",
            help="Scales the coupling down until activity cannot grow without bound. Switch it off "
            "only with a saturating transfer, which keeps the model bounded by itself.",
        )
    return {
        "simulation": simulation,
        "n_trials": int(n_trials),
        "seed": int(run_seed),
        "enforce_stationarity": enforce,
    }


# ---------------------------------------------------------------------------
# Model tab
# ---------------------------------------------------------------------------


def _operating_point_control(seed: ExperimentSpec) -> float | None:
    """Let the model be placed at a chosen distance from criticality."""
    st.markdown("**How close the network runs to instability**")
    st.caption(
        "The spectral radius is what decides how sharp the resonances are. Fixing it scales every "
        "coupling by one common factor, leaving relative connectivity and every delay as they are, "
        "so it can be varied on its own, or swept. Leave it off to use the coupling the components "
        "specify."
    )
    columns = st.columns([1, 3])
    enabled = columns[0].checkbox(
        "set it directly",
        value=seed.target_spectral_radius is not None,
        key=f"{WIDGET_PREFIX}.use_target_radius",
    )
    if not enabled:
        return None
    value = columns[1].slider(
        "spectral radius",
        min_value=0.50,
        max_value=1.20,
        value=float(seed.target_spectral_radius or 0.95),
        step=0.01,
        key=f"{WIDGET_PREFIX}.target_radius",
        help="One is the stability boundary: below it the model settles, at it the peaks are "
        "sharpest, above it only a saturating transfer keeps activity bounded. A model whose "
        "coupling is spread over many lags cannot be scaled far below one; the page says so if "
        "the value is out of reach.",
    )
    if value >= 1.0:
        st.caption(
            "Above the boundary: switch off *keep the model stable* in the sidebar and choose a "
            "saturating transfer, or the coupling will simply be scaled back down."
        )
    return float(value)


def _atlas_panel(ref: ComponentRef) -> None:
    """Show atlas documentation links and what each merge level yields."""
    parcellation = str(ref.params.get("parcellation", ""))
    if not parcellation:
        return
    try:
        info = cached_parcellation_info(parcellation)
    except Exception as exc:  # noqa: BLE001 - the atlas service may be unreachable
        st.caption(f"Atlas details unavailable: {type(exc).__name__}: {exc}")
        return

    with st.expander(f"About {info['name']}", expanded=False):
        if info.get("description"):
            st.write(info["description"][:800] + ("…" if len(info["description"]) > 800 else ""))
        st.caption(f"{info['n_regions']} regions in the atlas · modality: {info.get('modality') or 'unknown'}")
        for url in info.get("urls", []):
            st.markdown(f"- [{url}]({url})")
        for citation in info.get("publications", [])[:3]:
            st.caption(citation[:300])

    with st.expander("What each level of this atlas contains", expanded=False):
        try:
            levels = cached_hierarchy(ref.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim
            st.caption(f"Could not read the region tree: {type(exc).__name__}: {exc}")
            return
        st.caption(hierarchy_note(parcellation))
        st.dataframe(
            [
                {
                    "level": level.level,
                    "regions in the model": level.n_groups,
                    "for example": ", ".join(level.examples),
                }
                for level in levels
            ],
            width="stretch",
            hide_index=True,
        )


def _unsuitable_connectivity(distances_ref: ComponentRef, region_codes: tuple[str, ...]) -> dict[str, str]:
    """Name the connectivity rules that cannot describe the current regions."""
    reasons: dict[str, str] = {}
    if distances_ref.name != "siibra":
        reasons["siibra"] = (
            "Atlas connectivity is looked up for atlas regions, so it needs the atlas distance "
            "source. Uniform, distance-decay, random and manual coupling work for any regions."
        )
    if region_codes and region_codes != ("FL", "PL", "OL", "TL", "T"):
        reasons["steeghs_2025"] = (
            "These are the published weights for the five lobes FL, PL, OL, TL and T, so they only "
            "apply to that region set."
        )
    return reasons


def model_tab(seed: ExperimentSpec) -> dict[str, Any]:
    """
    Render the model configuration, one concern per sub-tab.

    Parameters
    ----------
    seed : ExperimentSpec
        Specification seeding the control values.

    Returns
    -------
    dict
        Component references plus ``relay_code``, ``self_weight``,
        ``include_regions`` and ``use_length_mixture``; the built distance
        source under ``distances_provider`` (or the error text under
        ``error``).
    """
    tabs = st.tabs(["Regions & distances", "Connections", "Delays", "Local processing", "Input & output"])

    # --- regions and distances --------------------------------------------
    with tabs[0]:
        st.caption(SLOT_HELP["distances"])
        distances_ref = render_slot(
            "distances",
            seed.distances,
            key_prefix=f"{WIDGET_PREFIX}.distances",
            label="where the geometry comes from",
            columns=2,
        )
        if distances_ref.name == "siibra":
            distances_ref = _region_picker(distances_ref, seed.distances)
            _atlas_panel(distances_ref)

        provider = None
        error: str | None = None
        try:
            provider = cached_distances(distances_ref.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            error = f"{type(exc).__name__}: {exc}"
            st.error(f"Could not load these distances. {error}")

        all_codes = tuple(provider.regions.codes) if provider is not None else ()
        include, relay = _region_controls(seed, all_codes, provider)

    region_codes = tuple(include) if include else all_codes

    # --- connections -------------------------------------------------------
    with tabs[1]:
        st.caption(SLOT_HELP["connectivity"])
        connectivity_ref = render_slot(
            "connectivity",
            seed.connectivity,
            key_prefix=f"{WIDGET_PREFIX}.connectivity",
            label="where the coupling comes from",
            region_codes=region_codes,
            columns=2,
            unsuitable=_unsuitable_connectivity(distances_ref, all_codes),
        )
        st.divider()
        target_radius = _operating_point_control(seed)

    # --- delays ------------------------------------------------------------
    with tabs[2]:
        st.markdown("**How long a signal takes to travel**")
        st.caption(SLOT_HELP["kernel"])
        left, right = st.columns(2, gap="large")
        with left:
            kernel_ref = render_slot(
                "kernel",
                seed.kernel,
                key_prefix=f"{WIDGET_PREFIX}.kernel",
                label="what sets conduction speed",
            )
        needs_diameter = "diameter" in catalog.get("kernel", kernel_ref.name).context
        with right:
            if needs_diameter:
                st.caption(SLOT_HELP["diameter"])
                diameter_ref = render_slot(
                    "diameter",
                    seed.diameter,
                    key_prefix=f"{WIDGET_PREFIX}.diameter",
                    label="how thick the axons are",
                )
            else:
                diameter_ref = seed.diameter
                st.caption(
                    "This kernel sets conduction speed directly, so the axon-calibre distribution "
                    "is not used. Switch to the speed-from-calibre kernel to choose one."
                )

        merged = provider is not None and getattr(provider, "mixture", None) is not None
        use_mixture = seed.use_length_mixture
        if merged:
            st.divider()
            use_mixture = st.checkbox(
                "spread delays over every connection inside a merged pathway",
                value=bool(seed.use_length_mixture),
                key=f"{WIDGET_PREFIX}.use_mixture",
                help="Merging regions turns one pathway into many finer connections of different "
                "lengths. With this on, each of those lengths gets its own delay distribution and "
                "they are combined in proportion to the fibres each carries. With it off, a single "
                "representative length stands for the whole pathway, which is quicker to build and "
                "gives a narrower spread of delays.",
            )
            st.caption(f"{len(provider.mixture):,} finer connections sit behind the merged pathways of this model.")

    # --- local processing --------------------------------------------------
    with tabs[3]:
        st.markdown("**What happens after a signal arrives**")
        st.caption(SLOT_HELP["local_delay"])
        local_ref = render_slot(
            "local_delay",
            seed.local_delay,
            key_prefix=f"{WIDGET_PREFIX}.local_delay",
            label="postsynaptic processing time",
            columns=2,
        )
        self_weight = st.slider(
            "self-excitation",
            min_value=0.0,
            max_value=1.0,
            value=float(seed.self_weight),
            step=0.05,
            key=f"{WIDGET_PREFIX}.self_weight",
            help="How strongly each region drives itself, through the local delay above. The "
            "original model has none, folding local recurrence into the noise instead, so "
            "raising this is an extension, not a setting of the published model.",
        )
        if self_weight > 0 and local_ref.name == "none":
            st.warning("Self-excitation needs a local delay: pick one above, or set it back to zero.")

    # --- input and output --------------------------------------------------
    with tabs[4]:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Ongoing input**")
            st.caption(SLOT_HELP["noise"])
            noise_ref = render_slot(
                "noise",
                seed.noise,
                key_prefix=f"{WIDGET_PREFIX}.noise",
                label="what keeps the network active",
                columns=2,
            )
        with right:
            st.markdown("**Output of a region**")
            st.caption(SLOT_HELP["transfer"])
            transfer_ref = render_slot(
                "transfer",
                seed.transfer,
                key_prefix=f"{WIDGET_PREFIX}.transfer",
                label="how activity is passed on",
                columns=2,
            )

    return {
        "distances": distances_ref,
        "connectivity": connectivity_ref,
        "diameter": diameter_ref,
        "kernel": kernel_ref,
        "local_delay": local_ref,
        "noise": noise_ref,
        "transfer": transfer_ref,
        "relay_code": relay,
        "self_weight": float(self_weight),
        "target_spectral_radius": target_radius,
        "include_regions": tuple(include),
        "use_length_mixture": bool(use_mixture),
        "distances_provider": provider,
        "error": error,
    }


def _default_rows(choices: Sequence[RegionChoice], nodes: Sequence[str]) -> list[dict[str, Any]]:
    """Seed the selection table from a specification, or from the whole brain."""
    # Rows always start with the level filter open: it narrows the list beside
    # it, and a row that opens already narrowed cannot reach the rest of the
    # atlas without the reader first noticing the filter.
    specs = parse_nodes(nodes)
    if specs:
        rows: list[dict[str, Any]] = []
        for spec in specs:
            # One row per depth, so entries split the same way stay together.
            match = next((row for row in rows if row["depth"] == spec.depth), None)
            if match is None:
                rows.append({"level": None, "names": [spec.name], "depth": spec.depth})
            else:
                match["names"].append(spec.name)
        return rows
    root = next((choice.name for choice in choices if choice.level == 0), None)
    if root is None:  # pragma: no cover - every atlas tree has a root
        return []
    return [{"level": None, "names": [root], "depth": DEFAULT_MERGE_DEPTH}]


def _region_rows(choices: Sequence[RegionChoice], nodes: Sequence[str]) -> list[dict[str, Any]]:
    """Return the selection table, seeding it on first use."""
    stored = st.session_state.get(REGIONS_KEY)
    if stored is None:
        stored = _default_rows(choices, nodes)
        st.session_state[REGIONS_KEY] = stored
    return stored


def _region_picker(ref: ComponentRef, seed: ComponentRef) -> ComponentRef:
    """
    Choose the model's regions as a table of selections.

    Each row picks one or more regions from the atlas and says how far to
    split them: depth 0 keeps a region whole, depth 1 makes a node per
    immediate child. Rows can name regions from different levels, which is
    what lets a lobe and a whole nucleus stand side by side.
    """
    try:
        choices = cached_region_choices(ref.model_dump_json())
    except Exception:  # noqa: BLE001 - the atlas may be unreachable
        return ref
    if not choices:
        return ref

    rows = _region_rows(choices, seed.params.get("nodes") or ())
    levels = sorted({choice.level for choice in choices})
    label_of = {
        choice.name: f"{choice.name}  ·  level {choice.level}, {choice.n_members} regions" for choice in choices
    }

    st.markdown("**Which regions the model has**")
    st.caption(REGION_PICKER)

    header = st.columns(_ROW_LAYOUT, gap="small")
    header[0].caption("level")
    header[1].caption("regions")
    header[2].caption("split by")
    header[3].caption("")

    nodes: list[str] = []
    keep: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        edited = _selection_row(index, row, choices, levels, label_of)
        if edited is None:
            continue
        keep.append(edited)
        nodes.extend(NodeSpec(name, edited["depth"]).as_text() for name in edited["names"])

    add, clear = st.columns([1, 1])
    if add.button("Add a selection", width="stretch", key=f"{WIDGET_PREFIX}.row_add"):
        keep.append({"level": None, "names": [], "depth": 0})
        st.session_state[REGIONS_KEY] = keep
        st.rerun()
    if clear.button("Reset to the whole brain", width="stretch", key=f"{WIDGET_PREFIX}.row_reset"):
        st.session_state[REGIONS_KEY] = _default_rows(choices, ())
        st.rerun()

    if keep != rows:
        st.session_state[REGIONS_KEY] = keep
        if len(keep) != len(rows):
            st.rerun()

    if not nodes:
        st.info("No regions selected yet. Add a selection above, or reset to the whole brain.")
        return ref

    try:
        overlaps = describe_overlaps(cached_native_regions(ref.model_dump_json()), nodes)
    except Exception:  # noqa: BLE001 - the atlas may be unreachable
        overlaps = ()
    for message in overlaps:
        st.warning(message)

    if tuple(nodes) == tuple(ref.params.get("nodes") or ()):
        return ref
    return ComponentRef(name=ref.name, params={**ref.params, "nodes": tuple(nodes)})


def _selection_row(
    index: int,
    row: dict[str, Any],
    choices: Sequence[RegionChoice],
    levels: Sequence[int],
    label_of: Mapping[str, str],
) -> dict[str, Any] | None:
    """Render one row of the selection table; ``None`` when it is removed."""
    columns = st.columns(_ROW_LAYOUT, gap="small")
    options: list[int | None] = [None, *levels]
    stored_level = row.get("level")
    with columns[0]:
        level = st.selectbox(
            "level",
            options=options,
            index=options.index(stored_level) if stored_level in options else 0,
            format_func=lambda value: "any" if value is None else str(value),
            key=f"{WIDGET_PREFIX}.row_level.{index}",
            label_visibility="collapsed",
            help="Narrows the region list beside it. It does not change the model.",
        )

    shown = [choice.name for choice in choices if level is None or choice.level == level]
    # A region already chosen stays selectable whatever the filter shows.
    offered = list(dict.fromkeys([*shown, *row.get("names", [])]))
    with columns[1]:
        names = st.multiselect(
            "regions",
            options=offered,
            default=[name for name in row.get("names", []) if name in offered],
            format_func=lambda name: label_of.get(name, name),
            key=f"{WIDGET_PREFIX}.row_names.{index}",
            placeholder="type a region name",
            label_visibility="collapsed",
        )
    with columns[2]:
        depth = int(
            st.number_input(
                "split by",
                min_value=0,
                max_value=6,
                value=int(row.get("depth", 0)),
                step=1,
                key=f"{WIDGET_PREFIX}.row_depth.{index}",
                label_visibility="collapsed",
                help="0 keeps each region whole; 1 makes a node per immediate sub-region.",
            )
        )
    with columns[3]:
        if st.button("✕", key=f"{WIDGET_PREFIX}.row_drop.{index}", help="Remove this selection"):
            return None
    return {"level": level, "names": list(names), "depth": depth}


def _region_controls(
    seed: ExperimentSpec,
    all_codes: tuple[str, ...],
    provider: object | None,
) -> tuple[tuple[str, ...], str | None]:
    """Render the region subset and relay controls, and return their values."""
    st.divider()
    st.markdown("**Which regions to model, and how signals are routed**")
    left, right = st.columns(2, gap="large")

    with left:
        default = [code for code in seed.include_regions if code in all_codes]
        include = tuple(
            st.multiselect(
                "regions to keep",
                options=list(all_codes),
                default=default,
                key=f"{WIDGET_PREFIX}.include_regions",
                placeholder=f"all {len(all_codes)} regions",
                help="Leave empty to model every region the source provides. Picking a subset keeps "
                "those regions and the connections between them, and drops the rest. Useful for "
                "isolating a system without changing where its numbers came from.",
            )
        )
        if include and len(include) < 2:
            st.warning("Keep at least two regions, or leave the selection empty.")
            include = ()

    with right:
        region_codes = include or all_codes
        options = ["(direct)", *region_codes]
        current = seed.relay_code if seed.relay_code in region_codes else "(direct)"
        relay = st.selectbox(
            "route connections through",
            options=options,
            index=options.index(current),
            key=f"{WIDGET_PREFIX}.relay",
            help="Direct means every connection goes straight from source to target. Choosing a "
            "region (the thalamus, say) makes every other connection travel to it and onwards, "
            "so its delay is the sum of two legs. The matching two-leg delay kernel is selected "
            "automatically.",
        )
        if provider is not None and include:
            st.caption(f"{len(region_codes)} of {len(all_codes)} regions in the model.")

    return include, (None if relay == "(direct)" else relay)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _event_editor(index: int, stored: dict[str, Any], region_codes: tuple[str, ...]) -> EventRef | None:
    """Render one event and return it, or ``None`` if it was removed."""
    prefix = f"{WIDGET_PREFIX}.event{index}"
    current = EventRef.model_validate(stored)
    header = f"{current.name} — {current.onset_seconds:.2f} s"
    with st.expander(header, expanded=True):
        top, remove = st.columns([4, 1])
        with top:
            name = st.text_input("name", value=current.name, key=f"{prefix}.name")
            onset = st.slider(
                "starts at (s)",
                min_value=0.0,
                max_value=20.0,
                value=float(current.onset_seconds),
                step=0.05,
                key=f"{prefix}.onset",
                help="Time from the start of the analysed window. Leave room before it for a "
                "baseline, and after it for the response.",
            )
        with remove:
            st.write("")
            if st.button("Remove", key=f"{prefix}.remove", width="stretch"):
                return None

        drive_ref = None
        if current.drive is not None:
            st.markdown("**Stimulus**")
            drive_ref = render_slot(
                "drive",
                current.drive,
                key_prefix=f"{prefix}.drive",
                label="waveform",
                region_codes=region_codes,
                columns=2,
                show_reference=False,
            )

        modulation_ref = None
        use_modulation = st.checkbox(
            "also change coupling during this event",
            value=current.modulation is not None,
            key=f"{prefix}.use_mod",
            help="Temporarily scales connection strengths. On its own this changes variability "
            "rather than the average response; combined with a stimulus it changes the response size.",
        )
        if use_modulation:
            modulation_ref = render_slot(
                "modulation",
                current.modulation,
                key_prefix=f"{prefix}.mod",
                label="modulation",
                region_codes=region_codes,
                columns=2,
                show_reference=False,
            )

        if drive_ref is None and modulation_ref is None:
            st.info("This event does nothing yet. Give it a stimulus or a coupling change.")
            return None
        return EventRef(onset_seconds=onset, drive=drive_ref, modulation=modulation_ref, name=name)


def events_editor(seed: ExperimentSpec, region_codes: tuple[str, ...]) -> tuple[EventRef, ...]:
    """
    Render the event list, with controls to add and remove events.

    Parameters
    ----------
    seed : ExperimentSpec
        Specification seeding the list on first use.
    region_codes : tuple of str
        Region codes a stimulus can target.

    Returns
    -------
    tuple of EventRef
        The events as configured.
    """
    from diaxcondel.catalog import options

    stored = _stored_events(seed)

    st.caption(
        "Events add a stimulus, a temporary change of coupling, or both, at a chosen time in every "
        "trial. Without events the model just sits at rest."
    )
    picker, adder, clearer = st.columns([3, 1, 1])
    drive_entries = options("drive")
    with picker:
        kind = st.selectbox(
            "stimulus type to add",
            options=[entry.name for entry in drive_entries],
            format_func=lambda name: next(e.label for e in drive_entries if e.name == name),
            key=f"{WIDGET_PREFIX}.new_event_kind",
        )
    with adder:
        st.write("")
        if st.button("Add event", width="stretch"):
            stored.append(
                EventRef(
                    onset_seconds=0.5,
                    drive=ComponentRef(name=kind),
                    name=f"event {len(stored) + 1}",
                ).model_dump(mode="json")
            )
            st.session_state[EVENTS_KEY] = stored
            st.rerun()
    with clearer:
        st.write("")
        if st.button("Clear all", width="stretch", disabled=not stored):
            st.session_state[EVENTS_KEY] = []
            st.rerun()

    events: list[EventRef] = []
    kept: list[dict[str, Any]] = []
    for index, event in enumerate(stored):
        edited = _event_editor(index, event, region_codes)
        if edited is None:
            continue
        events.append(edited)
        kept.append(edited.model_dump(mode="json"))

    if kept != stored:
        st.session_state[EVENTS_KEY] = kept
        if len(kept) != len(stored):
            st.rerun()

    return tuple(events)


def describe_slot(slot: str) -> str:
    """
    Return the one-line description of a catalog slot.

    Parameters
    ----------
    slot : str
        Slot name.

    Returns
    -------
    str
        Description for headings and tooltips.
    """
    return SLOT_DESCRIPTIONS.get(slot, "")  # type: ignore[arg-type]
