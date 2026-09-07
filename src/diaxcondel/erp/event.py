"""
ERP event definition and per-trial resolution.

An :class:`ERPEvent` specifies a stimulus in *experiment time* (relative to
the post-burn-in simulation window).  It carries:

- An optional additive :class:`~diaxcondel.simulate.stimulus.Drive` that injects a
  waveform into one or more regions.
- An optional :class:`~diaxcondel.erp.modulation.StepModulation` that
  temporarily changes effective coupling.

At least one of ``drive`` or ``modulation`` must be present.

Per-trial jitter
----------------
Each :class:`ERPEvent` can carry :class:`EventJitter` that describes
trial-to-trial variability in timing, amplitude, and modulation strength.
Before a trial is simulated, :func:`resolve_event` draws independent
per-trial realisations from the jitter distributions.  The base event
specification is never mutated; the resolved, jitter-applied copies are only
used during simulation.

Timing convention
-----------------
All ``onset_seconds`` values are relative to the *post-burn-in* window.
``build_phi_schedule`` and the drive-rendering functions add the burn-in
offset internally, so user-facing code can think in experiment time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from diaxcondel._rng import RNGLike, as_generator
from diaxcondel.simulate.stimulus import ArrayDrive, Drive, GaussianPulseDrive, SquarePulseDrive

from .modulation import StepModulation


@dataclass(frozen=True, slots=True)
class EventJitter:
    """
    Per-trial variability for an :class:`ERPEvent`.

    All values are specified as standard deviations of a zero-mean Gaussian
    unless noted.  Negative realised values are clamped where they would be
    physically meaningless (e.g. duration cannot be negative).

    Parameters
    ----------
    onset_std_seconds : float
        Jitter in onset time.  Default ``0.0`` (no jitter).
    amplitude_std : float
        Additive Gaussian jitter applied to the drive amplitude as a
        multiplicative factor: ``effective_amplitude = base_amplitude * (1 +
        N(0, amplitude_std))``.  Default ``0.0``.
    duration_std_seconds : float
        Jitter in drive/modulation duration.  Default ``0.0``.
    modulation_strength_std : float
        Jitter applied to the modulation factor matrix as an additive
        perturbation: ``effective_factor = base_factor + N(0, strength_std)``.
        Default ``0.0``.
    """

    onset_std_seconds: float = 0.0
    amplitude_std: float = 0.0
    duration_std_seconds: float = 0.0
    modulation_strength_std: float = 0.0

    def __post_init__(self) -> None:
        for name, val in [
            ("onset_std_seconds", self.onset_std_seconds),
            ("amplitude_std", self.amplitude_std),
            ("duration_std_seconds", self.duration_std_seconds),
            ("modulation_strength_std", self.modulation_strength_std),
        ]:
            if val < 0:
                raise ValueError(f"EventJitter.{name} must be non-negative; got {val}")


@dataclass(frozen=True, slots=True)
class ERPEvent:
    """
    A single stimulus event for ERP simulation.

    Parameters
    ----------
    onset_seconds : float
        Onset time relative to the post-burn-in trial window.  Must be
        non-negative.
    drive : Drive or None
        Additive waveform drive.  If ``None``, no signal is injected.
    modulation : StepModulation or None
        Connectivity modulation active from ``onset_seconds`` for
        ``modulation.duration_seconds``.  If ``None``, no modulation is
        applied.
    name : str
        Human-readable label for logging and plotting.  Default ``"event"``.
    jitter : EventJitter or None
        Per-trial variability specification.  ``None`` means no jitter.
    metadata : dict or None
        Free-form metadata for downstream analysis.  Not used by the
        simulator.

    Notes
    -----
    At least one of ``drive`` or ``modulation`` must be provided.  Providing
    neither is a programming error and raises :class:`ValueError`.

    For modulation-only events in the original linear model: the signed
    trial-average will remain near zero because zero-mean noise through a
    linear system has zero expectation regardless of coupling strength.  These
    events affect variance, power, and coherence, not the signed ERP mean.
    """

    onset_seconds: float
    drive: Drive | None
    modulation: StepModulation | None
    name: str = "event"
    jitter: EventJitter | None = None
    metadata: dict | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.onset_seconds < 0:
            raise ValueError(f"ERPEvent.onset_seconds must be non-negative; got {self.onset_seconds}")
        if self.drive is None and self.modulation is None:
            raise ValueError(f"ERPEvent '{self.name}': at least one of 'drive' or 'modulation' must be provided")


@dataclass(frozen=True, slots=True)
class _ResolvedEvent:
    """Internal: a jitter-applied event copy used during one trial."""

    onset_seconds: float
    drive: Drive | None
    modulation: StepModulation | None
    name: str


def resolve_event(event: ERPEvent, rng: RNGLike = None) -> _ResolvedEvent:
    """
    Apply per-trial jitter and return a resolved event.

    Parameters
    ----------
    event : ERPEvent
        Template event with optional jitter specification.
    rng : RNGLike, optional
        Random generator for this trial.

    Returns
    -------
    _ResolvedEvent
        Jitter-applied copy.  If ``event.jitter`` is ``None`` (or all
        standard deviations are zero), the returned object wraps the
        original drive/modulation without copying.
    """
    jitter = event.jitter
    if jitter is None or (
        jitter.onset_std_seconds == 0.0
        and jitter.amplitude_std == 0.0
        and jitter.duration_std_seconds == 0.0
        and jitter.modulation_strength_std == 0.0
    ):
        return _ResolvedEvent(
            onset_seconds=event.onset_seconds,
            drive=event.drive,
            modulation=event.modulation,
            name=event.name,
        )

    gen = as_generator(rng)

    # --- onset jitter ---
    onset = event.onset_seconds
    if jitter.onset_std_seconds > 0:
        onset = max(0.0, onset + gen.normal(0.0, jitter.onset_std_seconds))

    # --- drive jitter ---
    drive = event.drive
    if drive is not None and (jitter.amplitude_std > 0 or jitter.duration_std_seconds > 0):
        drive = _apply_drive_jitter(drive, jitter, gen)

    # --- modulation jitter ---
    modulation = event.modulation
    if modulation is not None and (jitter.modulation_strength_std > 0 or jitter.duration_std_seconds > 0):
        modulation = _apply_modulation_jitter(modulation, jitter, gen)

    return _ResolvedEvent(
        onset_seconds=onset,
        drive=drive,
        modulation=modulation,
        name=event.name,
    )


def _apply_drive_jitter(drive: Drive, jitter: EventJitter, gen: np.random.Generator) -> Drive:
    """Return a drive with amplitude/duration jitter applied."""
    amp_scale = 1.0
    if jitter.amplitude_std > 0:
        amp_scale = 1.0 + gen.normal(0.0, jitter.amplitude_std)

    dur_delta = 0.0
    if jitter.duration_std_seconds > 0:
        dur_delta = gen.normal(0.0, jitter.duration_std_seconds)

    if isinstance(drive, GaussianPulseDrive):
        new_dur = max(drive.width_seconds * 2, drive.duration_seconds + dur_delta)
        return GaussianPulseDrive(
            duration_seconds=new_dur,
            peak_seconds=min(drive.peak_seconds, new_dur),
            width_seconds=drive.width_seconds,
            amplitude=drive.amplitude * amp_scale,
            targets=drive.targets,
        )
    if isinstance(drive, SquarePulseDrive):
        new_dur = max(1e-6, drive.duration_seconds + dur_delta)
        return SquarePulseDrive(
            duration_seconds=new_dur,
            amplitude=drive.amplitude * amp_scale,
            targets=drive.targets,
        )
    if isinstance(drive, ArrayDrive):
        # For ArrayDrive, only scale amplitude; duration is implicit in array length.
        if amp_scale != 1.0:
            return ArrayDrive(waveform=drive.waveform * amp_scale, targets=drive.targets)
        return drive
    # Unknown Drive type: return unchanged
    return drive


def _apply_modulation_jitter(mod: StepModulation, jitter: EventJitter, gen: np.random.Generator) -> StepModulation:
    """Return a StepModulation with strength/duration jitter applied."""
    factor = mod.factor
    dur = mod.duration_seconds

    if jitter.modulation_strength_std > 0:
        noise = gen.normal(0.0, jitter.modulation_strength_std, size=factor.shape)
        factor = factor + noise

    if jitter.duration_std_seconds > 0:
        dur = max(1e-6, dur + gen.normal(0.0, jitter.duration_std_seconds))

    return StepModulation(
        factor=factor,
        duration_seconds=dur,
        instability_policy=mod.instability_policy,
    )
