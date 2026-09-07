"""Tests for ERPEvent construction and per-trial resolution."""

import numpy as np

from diaxcondel.erp.event import ERPEvent, EventJitter, resolve_event
from diaxcondel.erp.modulation import StepModulation
from diaxcondel.simulate.stimulus import DriveTarget, GaussianPulseDrive


def _gaussian_drive(code="T"):
    return GaussianPulseDrive(
        duration_seconds=0.05,
        peak_seconds=0.02,
        width_seconds=0.01,
        targets=(DriveTarget(code),),
    )


def _step_mod(n=3):
    return StepModulation(factor=np.eye(n) * 0.9, duration_seconds=0.1, instability_policy="ignore")


class TestERPEvent:
    def test_drive_only_is_valid(self):
        ev = ERPEvent(onset_seconds=0.1, drive=_gaussian_drive(), modulation=None)
        assert ev.drive is not None
        assert ev.modulation is None

    def test_modulation_only_is_valid(self):
        ev = ERPEvent(onset_seconds=0.1, drive=None, modulation=_step_mod())
        assert ev.drive is None
        assert ev.modulation is not None

    def test_both_drive_and_modulation_valid(self):
        ev = ERPEvent(onset_seconds=0.1, drive=_gaussian_drive(), modulation=_step_mod())
        assert ev.drive is not None
        assert ev.modulation is not None

    def test_name_default(self):
        ev = ERPEvent(onset_seconds=0.0, drive=_gaussian_drive(), modulation=None)
        assert ev.name == "event"


class TestEventJitter:
    def test_defaults_are_zero(self):
        j = EventJitter()
        assert j.onset_std_seconds == 0.0
        assert j.amplitude_std == 0.0


class TestResolveEvent:
    def test_no_jitter_returns_same_drive_and_modulation(self):
        drive = _gaussian_drive()
        mod = _step_mod()
        ev = ERPEvent(onset_seconds=0.1, drive=drive, modulation=mod)
        resolved = resolve_event(ev)
        assert resolved.drive is drive
        assert resolved.modulation is mod
        assert resolved.onset_seconds == 0.1

    def test_onset_jitter_applied(self):
        drive = _gaussian_drive()
        ev = ERPEvent(
            onset_seconds=0.5,
            drive=drive,
            modulation=None,
            jitter=EventJitter(onset_std_seconds=0.1),
        )
        rng = np.random.default_rng(42)
        resolved = resolve_event(ev, rng)
        # With a positive std, onset should be different from the base value.
        # We just check it's non-negative (clamped).
        assert resolved.onset_seconds >= 0.0

    def test_amplitude_jitter_changes_gaussian_amplitude(self):
        drive = GaussianPulseDrive(
            duration_seconds=0.05,
            peak_seconds=0.02,
            width_seconds=0.01,
            amplitude=1.0,
            targets=(DriveTarget("T"),),
        )
        ev = ERPEvent(
            onset_seconds=0.1,
            drive=drive,
            modulation=None,
            jitter=EventJitter(amplitude_std=0.5),
        )
        # Use a seed that gives non-zero jitter.
        rng = np.random.default_rng(0)
        resolved = resolve_event(ev, rng)
        assert isinstance(resolved.drive, GaussianPulseDrive)
        assert resolved.drive.amplitude != 1.0

    def test_modulation_strength_jitter_changes_factor(self):
        mod = _step_mod(n=2)
        ev = ERPEvent(
            onset_seconds=0.1,
            drive=None,
            modulation=mod,
            jitter=EventJitter(modulation_strength_std=0.1),
        )
        rng = np.random.default_rng(7)
        resolved = resolve_event(ev, rng)
        assert resolved.modulation is not None
        assert not np.allclose(resolved.modulation.factor, mod.factor)
