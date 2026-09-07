"""
Running a specification.

:func:`run_experiment` builds the model (unless one is supplied), resolves any
stimulus events against its regions, and simulates every trial. Resting-state
and event-driven runs return the same container, so downstream analysis does
not branch on which kind of experiment was run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.erp.event import ERPEvent
from diaxcondel.model.params import SimulationParams
from diaxcondel.simulate.trials import TrialResult, simulate_trials

from .build import BuiltModel, build_model
from .spec import EventRef, ExperimentSpec


@dataclass(frozen=True, slots=True)
class RunResult:
    """
    Simulated signals together with the model that produced them.

    Attributes
    ----------
    built : BuiltModel
        The model, connectome, and construction notes.
    trials : TrialResult
        Per-trial signals of shape ``(n_trials, T, N)``.
    notes : tuple of str
        Warnings captured while simulating (added to the build notes).
    """

    built: BuiltModel
    trials: TrialResult
    notes: tuple[str, ...] = ()

    @property
    def spec(self) -> ExperimentSpec:
        """The specification behind this run."""
        return self.built.spec

    @property
    def regions(self) -> RegionSet:
        """Region set of the simulated signals."""
        return self.built.regions

    @property
    def params(self) -> SimulationParams:
        """Simulation settings used."""
        return self.built.params

    @property
    def signal(self) -> FloatArray:
        """First trial, shape ``(T, N)`` — the resting-state view."""
        return self.trials.trials[0]

    @property
    def average(self) -> FloatArray:
        """Trial-average signal, shape ``(T, N)``."""
        return np.asarray(self.trials.average)

    @property
    def time_s(self) -> FloatArray:
        """Time axis of the post-burn-in window, in seconds."""
        return np.asarray(self.trials.time_s)

    @property
    def all_notes(self) -> tuple[str, ...]:
        """Build and run notes, in order."""
        return (*self.built.notes, *self.notes)


def _build_events(spec: ExperimentSpec, regions: RegionSet) -> list[ERPEvent]:
    """Resolve event references against the model's regions."""
    events: list[ERPEvent] = []
    for ref in spec.events:
        events.append(_build_event(ref, regions))
    return events


def _build_event(ref: EventRef, regions: RegionSet) -> ERPEvent:
    """Build one :class:`ERPEvent` from its specification."""
    drive = ref.drive.build("drive", regions=regions) if ref.drive else None
    modulation = ref.modulation.build("modulation", regions=regions) if ref.modulation else None
    return ERPEvent(
        onset_seconds=ref.onset_seconds,
        drive=drive,
        modulation=modulation,
        name=ref.name,
    )


def run_experiment(
    spec: ExperimentSpec,
    *,
    built: BuiltModel | None = None,
    n_jobs: int = 1,
) -> RunResult:
    """
    Build (if needed) and simulate the experiment described by a spec.

    Parameters
    ----------
    spec : ExperimentSpec
        Experiment description.
    built : BuiltModel, optional
        Pre-built model for this spec, to avoid rebuilding when only the
        simulation changes. It is the caller's responsibility that it matches.
    n_jobs : int
        Worker processes for multi-trial runs; ``1`` runs serially.

    Returns
    -------
    RunResult
        Trials, model, and any warnings raised along the way.
    """
    model = built if built is not None else build_model(spec)
    noise_fn = spec.noise.build("noise")
    events = _build_events(spec, model.regions)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trials = simulate_trials(
            model.dynamics,
            model.params,
            noise_fn,
            n_trials=spec.n_trials,
            events=events or None,
            regions=model.regions,
            rng=spec.seed,
            n_jobs=n_jobs,
        )
    notes = tuple(dict.fromkeys(str(entry.message) for entry in caught))

    return RunResult(built=model, trials=trials, notes=notes)
