"""
Cached model building and simulation for the dashboard.

Streamlit reruns the whole script on every interaction, so everything
expensive is memoised on the specification that produced it: change the plot
window and nothing is recomputed; change a kernel parameter and only what
depends on it is.

Caching uses ``st.cache_resource`` (which stores objects as they are) rather
than ``st.cache_data`` (which copies them through pickle), because these
results are large, read-only, and shared safely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import streamlit as st

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.distance import DistanceProvider
from diaxcondel.connectome.grouping import HierarchyLevel, RegionChoice, browse_regions, hierarchy_levels
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.experiment import (
    BuiltModel,
    ComponentRef,
    ExperimentSpec,
    RunResult,
    Spectra,
    SweepResult,
    build_model,
    compute_coherence,
    compute_spectra,
    run_experiment,
    sweep,
)

_MAX_ENTRIES = 8


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Loading distances…")
def cached_distances(ref_json: str) -> DistanceProvider:
    """
    Build the distance source described by a serialised component reference.

    Parameters
    ----------
    ref_json : str
        JSON of a :class:`~diaxcondel.experiment.spec.ComponentRef`.

    Returns
    -------
    DistanceProvider
        The built source. Atlas downloads inside are additionally cached on
        disk, so a cold start pays for them only once per machine.
    """
    return ComponentRef.model_validate_json(ref_json).build("distances")


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Reading the atlas region tree…")
def cached_hierarchy(ref_json: str) -> tuple[HierarchyLevel, ...]:
    """
    Return what each merge level of an atlas distance source would produce.

    Parameters
    ----------
    ref_json : str
        JSON of a distance-source reference; its parcellation and filters are
        reused, with merging switched off.

    Returns
    -------
    tuple of HierarchyLevel
        One entry per level.
    """
    ref = ComponentRef.model_validate_json(ref_json)
    native = ComponentRef(name=ref.name, params={**ref.params, "merge_level": None, "nodes": ()})
    provider = native.build("distances")
    split = bool(ref.params.get("split_hemispheres", False))
    return hierarchy_levels(provider.regions, max_level=8, split_hemispheres=split)


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Reading the atlas region tree…")
def cached_native_regions(ref_json: str) -> RegionSet:
    """
    Return an atlas's regions before any merging or filtering.

    Parameters
    ----------
    ref_json : str
        JSON of a distance-source reference; only its parcellation matters.

    Returns
    -------
    RegionSet
        Every region of the parcellation, carrying its ancestor chain.
    """
    ref = ComponentRef.model_validate_json(ref_json)
    native = ComponentRef(
        name=ref.name,
        params={**ref.params, "merge_level": None, "nodes": (), "keep_parts": (), "drop_isolated": False},
    )
    return native.build("distances").regions


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Reading the atlas region tree…")
def cached_region_choices(ref_json: str) -> tuple[RegionChoice, ...]:
    """
    Return every region of an atlas that could be picked as a model node.

    Parameters
    ----------
    ref_json : str
        JSON of a distance-source reference.

    Returns
    -------
    tuple of RegionChoice
        Name, level, and member count for every region at every level.
    """
    return browse_regions(cached_native_regions(ref_json))


@st.cache_resource(max_entries=4, show_spinner="Looking up the atlas…")
def cached_parcellation_info(parcellation: str) -> dict[str, Any]:
    """
    Return descriptive metadata and documentation links for a parcellation.

    Parameters
    ----------
    parcellation : str
        siibra parcellation specification.

    Returns
    -------
    dict
        Name, description, links, publications, and region count.
    """
    from diaxcondel.connectome.atlases.siibra_atlas import parcellation_info

    return parcellation_info(parcellation)


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Building model…")
def cached_model(spec_json: str) -> BuiltModel:
    """
    Build the model for a serialised specification.

    Parameters
    ----------
    spec_json : str
        JSON of an :class:`~diaxcondel.experiment.spec.ExperimentSpec`.

    Returns
    -------
    BuiltModel
        Dynamics plus provenance.
    """
    return build_model(ExperimentSpec.model_validate_json(spec_json))


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Checking stability…")
def cached_spectral_radius(spec_json: str) -> float:
    """
    Return the spectral radius of a built model.

    Parameters
    ----------
    spec_json : str
        JSON of an experiment specification.

    Returns
    -------
    float
        Largest absolute eigenvalue of the companion matrix.
    """
    return cached_model(spec_json).spectral_radius()


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Simulating…")
def cached_run(spec_json: str) -> RunResult:
    """
    Simulate a serialised specification, reusing the cached model.

    Parameters
    ----------
    spec_json : str
        JSON of an experiment specification.

    Returns
    -------
    RunResult
        Trials and the model that produced them.
    """
    spec = ExperimentSpec.model_validate_json(spec_json)
    return run_experiment(spec, built=cached_model(spec_json))


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Estimating spectra…")
def cached_spectra(spec_json: str, segment_seconds: float, fmax_hz: float, estimator: str = "welch") -> Spectra:
    """
    Compute spectra for a serialised specification.

    Parameters
    ----------
    spec_json : str
        JSON of an experiment specification.
    segment_seconds : float
        Welch segment length.
    fmax_hz : float
        Upper frequency of the closed-form curve.
    estimator : {"welch", "multitaper"}
        How to estimate the spectrum from the simulated signal.

    Returns
    -------
    Spectra
        Empirical and, where valid, closed-form spectra.
    """
    return compute_spectra(
        cached_run(spec_json),
        estimator=estimator,
        segment_seconds=segment_seconds,
        fmax_hz=fmax_hz,
    )


@st.cache_resource(max_entries=_MAX_ENTRIES, show_spinner="Computing coherence…")
def cached_coherence(spec_json: str, segment_seconds: float) -> tuple[FloatArray, FloatArray]:
    """
    Compute pairwise coherence for a serialised specification.

    Parameters
    ----------
    spec_json : str
        JSON of an experiment specification.
    segment_seconds : float
        Welch segment length.

    Returns
    -------
    freqs_hz : FloatArray
    coherence : FloatArray
    """
    return compute_coherence(cached_run(spec_json), segment_seconds=segment_seconds)


@st.cache_resource(max_entries=4, show_spinner=False)
def cached_sweep(spec_json: str, path: str, values: tuple[float, ...]) -> SweepResult:
    """
    Run a one-parameter sweep, showing progress while it runs.

    Parameters
    ----------
    spec_json : str
        JSON of the base experiment specification.
    path : str
        Parameter path to vary, e.g. ``"kernel.speed_factor"``.
    values : tuple of float
        Values to run.

    Returns
    -------
    SweepResult
        One point per value.
    """
    spec = ExperimentSpec.model_validate_json(spec_json)
    bar = st.progress(0.0, text=f"Sweeping {path}…")

    def report(index: int, total: int, point: Mapping[str, Any]) -> None:
        shown = ", ".join(f"{key} = {value:g}" for key, value in point.items())
        bar.progress(index / total, text=f"{shown} — run {index + 1} of {total}")

    try:
        return sweep(spec, {path: list(values)}, progress=report)
    finally:
        bar.empty()


def sweep_values(start: float, stop: float, count: int, *, integer: bool = False) -> Sequence[float]:
    """
    Build an evenly spaced list of sweep values.

    Parameters
    ----------
    start, stop : float
        Range ends, inclusive.
    count : int
        Number of values.
    integer : bool
        Round to whole numbers and drop duplicates.

    Returns
    -------
    sequence of float
        The values to sweep.
    """
    values = np.linspace(start, stop, max(2, count))
    if integer:
        return sorted({int(round(value)) for value in values})
    return [float(value) for value in values]
