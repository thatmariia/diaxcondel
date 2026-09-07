"""
Drive entries: deterministic stimulus waveforms injected into regions.

Drive factories take the model's ``regions`` as build context, so the region
codes a user picks are validated against the connectome actually in use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated

import numpy as np

from diaxcondel.connectome.regions import RegionSet
from diaxcondel.simulate.stimulus import (
    ArrayDrive,
    ChirpDrive,
    Drive,
    DriveTarget,
    GaussianPulseDrive,
    SineBurstDrive,
    SquarePulseDrive,
)

from ..param import Param
from ..registry import register


def _resolve_targets(regions: RegionSet, targets: Sequence[str], weight: float) -> tuple[DriveTarget, ...]:
    """Turn region codes into drive targets, defaulting to every region."""
    codes = list(targets) if targets else list(regions.codes)
    for code in codes:
        regions.index(code)  # raises KeyError with the offending code
    return tuple(DriveTarget(code=code, weight=weight) for code in codes)


@register("drive", "gaussian_pulse", label="Gaussian pulse", tags=("erp",))
def gaussian_pulse(
    regions: RegionSet,
    targets: Annotated[tuple[str, ...], Param(widget="regions")] = (),
    amplitude: Annotated[float, Param(minimum=-50.0, maximum=50.0, step=0.5)] = 5.0,
    width_seconds: Annotated[float, Param(unit="s", minimum=0.001, maximum=0.5, step=0.001)] = 0.01,
    duration_seconds: Annotated[float, Param(unit="s", minimum=0.005, maximum=2.0, step=0.005)] = 0.06,
    peak_seconds: Annotated[float, Param(unit="s", minimum=0.0, maximum=2.0, step=0.005)] = 0.03,
    target_weight: Annotated[float, Param(minimum=-10.0, maximum=10.0, step=0.1, advanced=True)] = 1.0,
) -> Drive:
    """
    Brief bell-shaped pulse added to the input of the target regions.

    A short pulse is the informative stimulus: the shape of the response is
    then produced by the network's own delays rather than copied from the
    input.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    targets : tuple of str
        Which regions receive the pulse. Empty means all of them.
    amplitude : float
        Height of the pulse relative to the ongoing noise (whose strength is
        1 by default), so 5 is a clearly visible stimulus.
    width_seconds : float
        Half-width of the pulse. Short pulses excite a wide band of
        frequencies; long ones excite only slow components.
    duration_seconds : float
        How long the waveform is rendered for; it should comfortably contain
        the pulse.
    peak_seconds : float
        Where the peak sits inside that window.
    target_weight : float
        Extra per-region multiplier, e.g. ``-1`` to invert the input.

    Returns
    -------
    Drive
        Renderable drive.
    """
    return GaussianPulseDrive(
        duration_seconds=duration_seconds,
        peak_seconds=min(peak_seconds, duration_seconds),
        width_seconds=width_seconds,
        amplitude=amplitude,
        targets=_resolve_targets(regions, targets, target_weight),
    )


@register("drive", "square_pulse", label="Square pulse", tags=("erp",))
def square_pulse(
    regions: RegionSet,
    targets: Annotated[tuple[str, ...], Param(widget="regions")] = (),
    amplitude: Annotated[float, Param(minimum=-50.0, maximum=50.0, step=0.5)] = 5.0,
    duration_seconds: Annotated[float, Param(unit="s", minimum=0.005, maximum=2.0, step=0.005)] = 0.05,
    target_weight: Annotated[float, Param(minimum=-10.0, maximum=10.0, step=0.1, advanced=True)] = 1.0,
) -> Drive:
    """
    Step of constant input held for a fixed time.

    A sustained drive rather than a transient: useful for asking how the
    network settles into and out of a driven state.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    targets : tuple of str
        Which regions receive the step. Empty means all of them.
    amplitude : float
        Level held during the step, relative to the noise strength.
    duration_seconds : float
        How long the step lasts.
    target_weight : float
        Extra per-region multiplier.

    Returns
    -------
    Drive
        Renderable drive.
    """
    return SquarePulseDrive(
        duration_seconds=duration_seconds,
        amplitude=amplitude,
        targets=_resolve_targets(regions, targets, target_weight),
    )


@register("drive", "sine_burst", label="Sine burst", tags=("erp", "entrainment"))
def sine_burst(
    regions: RegionSet,
    targets: Annotated[tuple[str, ...], Param(widget="regions")] = (),
    frequency_hz: Annotated[float, Param(unit="Hz", minimum=0.5, maximum=100.0, step=0.5)] = 10.0,
    amplitude: Annotated[float, Param(minimum=-50.0, maximum=50.0, step=0.5)] = 5.0,
    duration_seconds: Annotated[float, Param(unit="s", minimum=0.05, maximum=10.0, step=0.05)] = 1.0,
    ramp_seconds: Annotated[float, Param(unit="s", minimum=0.0, maximum=2.0, step=0.05)] = 0.1,
    phase_deg: Annotated[float, Param(unit="deg", minimum=0.0, maximum=360.0, step=15.0, advanced=True)] = 0.0,
    target_weight: Annotated[float, Param(minimum=-10.0, maximum=10.0, step=0.1, advanced=True)] = 1.0,
) -> Drive:
    """
    Rhythmic input at a chosen frequency, with a smooth on- and offset.

    Drives the network at one frequency to see how strongly it follows,
    which is the natural probe for a model whose resonances come from delays:
    the response is largest when the drive matches a resonance.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    targets : tuple of str
        Which regions receive the drive. Empty means all of them.
    frequency_hz : float
        Frequency of the input.
    amplitude : float
        Peak height, relative to the noise strength.
    duration_seconds : float
        Length of the burst.
    ramp_seconds : float
        Rise and fall time of the envelope. Ramping avoids the broadband
        click that an abrupt start would inject.
    phase_deg : float
        Starting phase, in degrees.
    target_weight : float
        Extra per-region multiplier, e.g. to drive two regions in antiphase.

    Returns
    -------
    Drive
        Renderable drive.
    """
    return SineBurstDrive(
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
        amplitude=amplitude,
        ramp_seconds=ramp_seconds,
        phase_deg=phase_deg,
        targets=_resolve_targets(regions, targets, target_weight),
    )


@register("drive", "chirp", label="Frequency sweep (chirp)", tags=("erp", "entrainment"))
def chirp(
    regions: RegionSet,
    targets: Annotated[tuple[str, ...], Param(widget="regions")] = (),
    start_hz: Annotated[float, Param(unit="Hz", minimum=0.5, maximum=100.0, step=0.5)] = 2.0,
    end_hz: Annotated[float, Param(unit="Hz", minimum=0.5, maximum=100.0, step=0.5)] = 40.0,
    amplitude: Annotated[float, Param(minimum=-50.0, maximum=50.0, step=0.5)] = 5.0,
    duration_seconds: Annotated[float, Param(unit="s", minimum=0.2, maximum=60.0, step=0.1)] = 5.0,
    ramp_seconds: Annotated[float, Param(unit="s", minimum=0.0, maximum=2.0, step=0.05)] = 0.1,
    target_weight: Annotated[float, Param(minimum=-10.0, maximum=10.0, step=0.1, advanced=True)] = 1.0,
) -> Drive:
    """
    Input whose frequency sweeps a range, tracing the frequency response.

    A sine burst answers "does the network follow 10 Hz?"; a sweep answers
    "which frequencies does it follow?" in a single trial. The response grows
    where the drive meets a delay-driven resonance, so the envelope of the
    output is a direct read-out of where the network resonates.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    targets : tuple of str
        Which regions receive the drive. Empty means all of them.
    start_hz, end_hz : float
        First and last frequency of the sweep. Both must stay below half the
        sample rate.
    amplitude : float
        Peak height, relative to the noise strength.
    duration_seconds : float
        Length of the sweep. The frequency rises linearly across it, so a
        longer sweep dwells longer at each frequency and resolves narrow
        resonances better.
    ramp_seconds : float
        Rise and fall time of the envelope, which keeps the start and end
        from injecting a broadband click of their own.
    target_weight : float
        Extra per-region multiplier.

    Returns
    -------
    Drive
        Renderable drive.
    """
    return ChirpDrive(
        start_hz=start_hz,
        end_hz=end_hz,
        duration_seconds=duration_seconds,
        amplitude=amplitude,
        ramp_seconds=ramp_seconds,
        targets=_resolve_targets(regions, targets, target_weight),
    )


@register("drive", "custom_waveform", label="Custom waveform", tags=("manual",))
def custom_waveform(
    regions: RegionSet,
    targets: Annotated[tuple[str, ...], Param(widget="regions")] = (),
    samples: Annotated[str, Param(widget="vector")] = "[0, 1, 2, 1, 0]",
    amplitude: Annotated[float, Param(minimum=-50.0, maximum=50.0, step=0.5)] = 5.0,
    target_weight: Annotated[float, Param(minimum=-10.0, maximum=10.0, step=0.1, advanced=True)] = 1.0,
) -> Drive:
    """
    Any waveform, given sample by sample.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    targets : tuple of str
        Which regions receive the waveform. Empty means all of them.
    samples : str
        The waveform as a list of numbers, e.g. ``"[0, 1, 2, 1, 0]"``. The
        samples are taken to be at the simulation's sample rate, so a
        50-sample waveform lasts 50 ms at 1000 Hz. It is scaled to peak at
        ``amplitude``.
    amplitude : float
        Peak height of the rendered waveform, relative to the noise strength.
    target_weight : float
        Extra per-region multiplier.

    Returns
    -------
    Drive
        Renderable drive.
    """
    text = (samples or "").strip()
    try:
        values = json.loads(text) if text.startswith("[") else [float(v) for v in text.replace(",", " ").split()]
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not read the waveform samples: {exc}") from exc
    waveform = np.asarray(values, dtype=float).ravel()
    if waveform.size == 0:
        raise ValueError("the waveform needs at least one sample")
    peak = float(np.abs(waveform).max())
    if peak > 0:
        waveform = waveform * (amplitude / peak)
    return ArrayDrive(waveform=waveform, targets=_resolve_targets(regions, targets, target_weight))
