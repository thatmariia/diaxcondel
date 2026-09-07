"""
The diaxcondel playground: build a model, run it, look at what it does.

Run it with ``diaxcondel playground`` (or ``python -m diaxcondel.app``).

The page is a front end for the package, not a second implementation of it:
the controls come from the component catalog, and everything shown derives
from one :class:`~diaxcondel.experiment.spec.ExperimentSpec` that can be
downloaded and re-run offline. Adding a component to the catalog makes it
appear here, with its documentation, without touching this file.
"""

from __future__ import annotations

import streamlit as st

# Absolute imports: Streamlit executes this file as a top-level script, so it
# has no package context for relative ones.
from diaxcondel.app import _results as results
from diaxcondel.app._config import events_editor, model_tab, seed_spec, sidebar
from diaxcondel.app._state import cached_model, cached_spectral_radius
from diaxcondel.experiment import ExperimentSpec, estimate_cost


def _headline(spec: ExperimentSpec, n_regions: int) -> bool:
    """Show model size and cost; return ``False`` when the page should stop."""
    cost = estimate_cost(
        n_regions,
        spec.simulation,
        n_trials=spec.n_trials,
        enforce_stationarity=spec.enforce_stationarity,
    )
    header = st.columns(4)
    header[0].metric("regions", n_regions)
    header[1].metric(
        "delay coverage",
        f"{1000 * spec.simulation.n_lags * spec.simulation.dt_s:.0f} ms",
        help=f"{spec.simulation.n_lags} lags at {spec.simulation.sample_rate_hz} Hz.",
    )
    header[2].metric(
        "memory for delays",
        f"{cost.phi_megabytes:.1f} MiB",
        help="Size of the lag tensor: one number per lag, target and source.",
    )
    header[3].metric(
        "estimated run",
        f"{cost.total_seconds:.1f} s",
        help="Rough estimate for building and simulating this configuration.",
    )

    for note in cost.notes:
        st.warning(note)
    if cost.is_heavy and not st.checkbox("This is a heavy configuration. Run it anyway", value=False):
        st.info("Reduce the regions, the duration, or the sample rate, or tick the box to go ahead.")
        return False
    return True


def _stability_note(spec_json: str) -> None:
    """Report the spectral radius and what it means for this model."""
    try:
        radius = cached_spectral_radius(spec_json)
    except Exception:  # noqa: BLE001 - a diagnostic must never break the page
        return
    if radius >= 1.0:
        st.warning(
            f"Spectral radius {radius:.3f}, at or beyond the stability boundary. A linear model "
            "diverges here; a saturating transfer keeps it bounded."
        )
    elif radius > 0.97:
        st.info(f"Spectral radius {radius:.3f}, close to the boundary, so spectral peaks are sharp.")
    else:
        st.caption(f"Spectral radius {radius:.3f}: activity decays between round trips, well inside the stable range.")


def main() -> None:
    """Render the playground page."""
    st.set_page_config(page_title="diaxcondel playground", page_icon="🧠", layout="wide")

    seed = seed_spec()
    settings = sidebar(seed)

    st.title("Playground")
    st.caption(
        "Choose where the geometry and the coupling come from, decide how signals travel, and see what "
        "the network does. Load an example from the sidebar for a working model to take apart."
    )

    tabs = st.tabs(["Model", "Network", "Signals", "Spectra", "Events", "Sweep", "Guide", "Reproduce"])

    with tabs[0]:
        configuration = model_tab(seed)

    provider = configuration.pop("distances_provider", None)
    error = configuration.pop("error", None)
    all_codes = tuple(provider.regions.codes) if provider is not None else ()
    region_codes = configuration["include_regions"] or all_codes

    with tabs[4]:
        events = events_editor(seed, region_codes)

    if provider is None:
        with tabs[1]:
            st.error(f"Fix the distance source before anything else can run. {error}")
        return

    try:
        spec = ExperimentSpec(
            distances=configuration["distances"],
            connectivity=configuration["connectivity"],
            diameter=configuration["diameter"],
            kernel=configuration["kernel"],
            local_delay=configuration["local_delay"],
            noise=configuration["noise"],
            transfer=configuration["transfer"],
            simulation=settings["simulation"],
            relay_code=configuration["relay_code"],
            self_weight=configuration["self_weight"],
            target_spectral_radius=configuration["target_spectral_radius"],
            include_regions=configuration["include_regions"],
            use_length_mixture=configuration["use_length_mixture"],
            events=events,
            n_trials=settings["n_trials"],
            seed=settings["seed"],
            enforce_stationarity=settings["enforce_stationarity"],
            label=seed.label,
        )
    except ValueError as exc:
        with tabs[0]:
            st.error(f"This combination does not describe a model yet. {exc}")
        return

    spec_json = spec.model_dump_json()

    with tabs[0]:
        st.divider()
        if not _headline(spec, len(region_codes)):
            return
        try:
            model = cached_model(spec_json)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            st.error(f"Could not build this model. {type(exc).__name__}: {exc}")
            return
        for note in model.notes:
            st.warning(note)
        _stability_note(spec_json)

    with tabs[1]:
        results.network_tab(spec_json, model)
    with tabs[2]:
        results.signals_tab(spec_json, spec)
    with tabs[3]:
        results.spectra_tab(spec_json, spec)
    with tabs[4]:
        if spec.events:
            st.divider()
            results.event_results(spec_json, spec)
    with tabs[5]:
        results.sweep_tab(spec_json, spec)
    with tabs[6]:
        results.guide_tab()
    with tabs[7]:
        results.reproduce_tab(spec)


if __name__ == "__main__":
    main()
