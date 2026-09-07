"""
ERP extension for diaxcondel.

Provides connectivity modulations, multi-trial simulation, and analysis
helpers for ERP-style experiments on the linear VAR model.

Conceptual separation
---------------------
**Drive** — additive input
    Injects a deterministic waveform into one or more regions each trial.
    This is what creates a signed mean ERP when trials are averaged.
    Drive classes live in :mod:`diaxcondel.simulate.stimulus`.

**ConnectivityModulation** — routing/gain change
    Temporarily scales the effective coupling in the lag tensor ``Phi``.
    In the original linear zero-mean model, modulation *alone* does not
    produce a signed ERP average — it changes variance, power, and coherence.
    Pair with a drive to get a signed mean ERP with modulated amplitude or
    latency.

Quick-start
-----------
>>> from diaxcondel.simulate import DriveTarget, GaussianPulseDrive, TrialResult
>>> from diaxcondel.erp import (
...     StepModulation, ERPEvent, EventJitter,
...     simulate_trials, baseline_correct, average_trials,
... )
"""

from __future__ import annotations

from diaxcondel.simulate.stimulus import ArrayDrive, Drive, DriveTarget, GaussianPulseDrive, SquarePulseDrive
from diaxcondel.simulate.trials import TrialResult, simulate_trials

from .analysis import (
    average_trials,
    baseline_correct,
    epoch_around_event,
    peak_amplitude,
    peak_latency,
    sem_trials,
)
from .event import ERPEvent, EventJitter, resolve_event
from .modulation import InstabilityPolicy, StepModulation, compose_modulations, region_gain_factor

__all__ = [
    # drives (re-exported from simulate for convenience)
    "Drive",
    "DriveTarget",
    "ArrayDrive",
    "GaussianPulseDrive",
    "SquarePulseDrive",
    # result (re-exported from simulate for convenience)
    "TrialResult",
    # modulation
    "InstabilityPolicy",
    "StepModulation",
    "compose_modulations",
    "region_gain_factor",
    # events
    "ERPEvent",
    "EventJitter",
    "resolve_event",
    # simulation
    "simulate_trials",
    # analysis
    "average_trials",
    "baseline_correct",
    "epoch_around_event",
    "peak_amplitude",
    "peak_latency",
    "sem_trials",
]
