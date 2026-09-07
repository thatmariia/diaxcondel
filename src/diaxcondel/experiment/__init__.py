"""
Declarative experiments: describe a model, build it, run it, report it.

This layer is the bridge between the component catalog and the pipeline. An
:class:`ExperimentSpec` names one catalog entry per component; ``build_model``
assembles it; ``run_experiment`` simulates it; the report helpers turn the
result into the spectra and summaries that the dashboard and the command line
both display.

>>> from diaxcondel.experiment import ExperimentSpec, run_experiment, compute_spectra
>>> spec = ExperimentSpec().with_simulation(sim_seconds=4.0)
>>> result = run_experiment(spec)                      # doctest: +SKIP
>>> spectra = compute_spectra(result)                  # doctest: +SKIP

Because the spec is plain JSON, an experiment can be saved next to its
results, re-run later, or handed to a colleague.
"""

from __future__ import annotations

from .build import BuiltModel, ModelCost, build_model, estimate_cost
from .report import (
    ALPHA_BAND_HZ,
    SLOPE_BAND_HZ,
    DelayStatistics,
    RegionSummary,
    Spectra,
    compute_coherence,
    compute_spectra,
    delay_statistics,
    region_summaries,
    spec_to_python,
)
from .run import RunResult, run_experiment
from .spec import ComponentRef, EventRef, ExperimentSpec, load_spec, save_spec
from .sweep import (
    SweepAxis,
    SweepPoint,
    SweepResult,
    get_parameter,
    parameter_paths,
    set_parameter,
    sweep,
)

__all__ = [
    "ALPHA_BAND_HZ",
    "SLOPE_BAND_HZ",
    "BuiltModel",
    "ComponentRef",
    "DelayStatistics",
    "EventRef",
    "ExperimentSpec",
    "ModelCost",
    "RegionSummary",
    "RunResult",
    "Spectra",
    "SweepAxis",
    "SweepPoint",
    "SweepResult",
    "build_model",
    "compute_coherence",
    "compute_spectra",
    "delay_statistics",
    "estimate_cost",
    "get_parameter",
    "load_spec",
    "parameter_paths",
    "region_summaries",
    "run_experiment",
    "save_spec",
    "set_parameter",
    "spec_to_python",
    "sweep",
]
