"""
Additive drive sources and stimulus-array construction.

Drive classes render deterministic waveforms that are injected additively into
one or more target regions.  :func:`build_drive_array` converts a sequence of
``(onset_seconds, Drive)`` pairs into the ``(T_total, N)`` stimulus array
accepted by :func:`~diaxcondel.simulate.engine.simulate`.

Drive targets specify which regions receive the waveform and with what weight.
Duplicate region codes are rejected at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.params import SimulationParams


@dataclass(frozen=True, slots=True)
class DriveTarget:
    """
    Region-specific loading for an additive drive.

    Parameters
    ----------
    code : str
        Region code that receives the drive.
    weight : float
        Multiplicative scaling applied to the rendered waveform for this
        region.  Default ``1.0``.
    """

    code: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("DriveTarget.code must be non-empty")
        if not np.isfinite(self.weight):
            raise ValueError("DriveTarget.weight must be finite")


@runtime_checkable
class Drive(Protocol):
    """
    Protocol for additive drive waveform sources.

    Any object implementing :meth:`render` and carrying :attr:`targets` is a
    valid drive.  No inheritance is required.
    """

    @property
    def targets(self) -> tuple[DriveTarget, ...]:
        """Ordered, duplicate-free region targets."""
        ...

    def render(self, sample_rate_hz: float) -> FloatArray:
        """
        Render a 1-D waveform at the given sample rate.

        Parameters
        ----------
        sample_rate_hz : float
            Sampling rate in Hertz.

        Returns
        -------
        FloatArray of shape (K,)
            Waveform samples.  ``K`` may depend on ``sample_rate_hz``.
        """
        ...


def _validate_targets(targets: tuple[DriveTarget, ...]) -> None:
    codes = [t.code for t in targets]
    if len(set(codes)) != len(codes):
        raise ValueError(f"duplicate drive target codes: {sorted({c for c in codes if codes.count(c) > 1})}")
    if not targets:
        raise ValueError("targets must be non-empty")


@dataclass(frozen=True, slots=True)
class ArrayDrive:
    """
    User-supplied waveform projected to target regions.

    The waveform array is treated as pre-sampled; ``render`` returns it
    directly regardless of ``sample_rate_hz``.  Use this when you have a
    waveform already digitised at the correct rate.

    Parameters
    ----------
    waveform : array-like of shape (K,)
        Pre-sampled waveform.  Must be finite and 1-D.
    targets : tuple of DriveTarget
        Region targets.  Duplicate codes are rejected.
    """

    waveform: FloatArray
    targets: tuple[DriveTarget, ...]

    def __post_init__(self) -> None:
        wf = np.asarray(self.waveform, dtype=float)
        if wf.ndim != 1 or wf.size == 0:
            raise ValueError("ArrayDrive.waveform must be a non-empty 1-D array")
        if not np.all(np.isfinite(wf)):
            raise ValueError("ArrayDrive.waveform must contain only finite values")
        object.__setattr__(self, "waveform", wf)
        _validate_targets(self.targets)

    def render(self, sample_rate_hz: float) -> FloatArray:  # noqa: ARG002
        """Return the stored waveform (``sample_rate_hz`` is ignored)."""
        return self.waveform


@dataclass(frozen=True, slots=True)
class GaussianPulseDrive:
    """
    Short Gaussian pulse drive.

    Generates a Gaussian envelope of the form

        A * exp(-0.5 * ((t - peak) / width)^2)

    Brief pulses are scientifically preferred because they let the *network*
    produce the ERP morphology through its impulse response rather than
    encoding the full waveform in the input.

    Parameters
    ----------
    duration_seconds : float
        Waveform duration.  Must be positive.
    peak_seconds : float
        Time of peak within the waveform.  Must lie in
        ``[0, duration_seconds]``.
    width_seconds : float
        Gaussian standard deviation.  Must be positive.
    amplitude : float
        Peak amplitude.  Default ``1.0``.
    targets : tuple of DriveTarget
        Region targets.  Duplicate codes are rejected.
    """

    duration_seconds: float
    peak_seconds: float
    width_seconds: float
    targets: tuple[DriveTarget, ...]
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.width_seconds <= 0:
            raise ValueError("width_seconds must be positive")
        if not (0 <= self.peak_seconds <= self.duration_seconds):
            raise ValueError("peak_seconds must lie within [0, duration_seconds]")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        _validate_targets(self.targets)

    def render(self, sample_rate_hz: float) -> FloatArray:
        """Render a Gaussian pulse sampled at ``sample_rate_hz``."""
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        n = max(1, int(round(self.duration_seconds * sample_rate_hz)))
        t = np.arange(n, dtype=float) / sample_rate_hz
        return self.amplitude * np.exp(-0.5 * ((t - self.peak_seconds) / self.width_seconds) ** 2)


@dataclass(frozen=True, slots=True)
class SquarePulseDrive:
    """
    Constant-amplitude square pulse drive.

    Parameters
    ----------
    duration_seconds : float
        Pulse duration.  Must be positive.
    amplitude : float
        Constant amplitude over the pulse.  Default ``1.0``.
    targets : tuple of DriveTarget
        Region targets.  Duplicate codes are rejected.
    """

    duration_seconds: float
    targets: tuple[DriveTarget, ...]
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        _validate_targets(self.targets)

    def render(self, sample_rate_hz: float) -> FloatArray:
        """Render a square pulse sampled at ``sample_rate_hz``."""
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        n = max(1, int(round(self.duration_seconds * sample_rate_hz)))
        return np.full(n, self.amplitude, dtype=float)


@dataclass(frozen=True, slots=True)
class SineBurstDrive:
    """
    Rhythmic drive at a fixed frequency, with a smooth on- and offset.

    Driving a network at one frequency measures how strongly it follows —
    the natural probe for a model whose resonances come from delays. The
    envelope ramps up and down so that switching the drive on does not itself
    inject a broadband transient.

    Parameters
    ----------
    frequency_hz : float
        Frequency of the drive. Must be positive.
    duration_seconds : float
        Length of the burst. Must be positive.
    amplitude : float
        Peak amplitude before the envelope is applied.
    ramp_seconds : float
        Rise and fall time of the envelope; ``0`` starts and stops abruptly.
    phase_deg : float
        Starting phase in degrees.
    targets : tuple of DriveTarget
        Region targets. Duplicate codes are rejected.
    """

    frequency_hz: float
    duration_seconds: float
    targets: tuple[DriveTarget, ...]
    amplitude: float = 1.0
    ramp_seconds: float = 0.0
    phase_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.ramp_seconds < 0:
            raise ValueError("ramp_seconds must be non-negative")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        _validate_targets(self.targets)

    def render(self, sample_rate_hz: float) -> FloatArray:
        """Render the burst sampled at ``sample_rate_hz``."""
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.frequency_hz > 0.5 * sample_rate_hz:
            raise ValueError(
                f"frequency_hz ({self.frequency_hz}) exceeds the Nyquist frequency of {0.5 * sample_rate_hz} Hz"
            )
        n = max(1, int(round(self.duration_seconds * sample_rate_hz)))
        t = np.arange(n, dtype=float) / sample_rate_hz
        wave = self.amplitude * np.sin(2 * np.pi * self.frequency_hz * t + np.deg2rad(self.phase_deg))
        if self.ramp_seconds > 0:
            rise = np.clip(t / self.ramp_seconds, 0.0, 1.0)
            fall = np.clip((self.duration_seconds - t) / self.ramp_seconds, 0.0, 1.0)
            wave = wave * rise * fall
        return wave


@dataclass(frozen=True, slots=True)
class ChirpDrive:
    """
    Input whose frequency sweeps a range, for reading resonances off directly.

    A sine burst measures the response at one frequency and has to be
    repeated to find a resonance; a chirp sweeps the whole range in one trial
    and the response envelope traces the network's frequency response. Where
    the output is largest, the drive matched a delay-driven resonance.

    Parameters
    ----------
    start_hz, end_hz : float
        First and last frequency of the sweep. Both positive; they may be
        given in either order.
    duration_seconds : float
        Length of the sweep. Must be positive.
    amplitude : float
        Peak amplitude before the envelope is applied.
    ramp_seconds : float
        Rise and fall time of the envelope; ``0`` starts and stops abruptly.
    targets : tuple of DriveTarget
        Region targets. Duplicate codes are rejected.

    Notes
    -----
    The sweep is linear in frequency, so every frequency gets the same dwell
    time and the response envelope can be read against a linear axis. Phase
    is integrated rather than assumed, which keeps the waveform continuous.
    """

    start_hz: float
    end_hz: float
    duration_seconds: float
    targets: tuple[DriveTarget, ...]
    amplitude: float = 1.0
    ramp_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.start_hz <= 0 or self.end_hz <= 0:
            raise ValueError("start_hz and end_hz must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.ramp_seconds < 0:
            raise ValueError("ramp_seconds must be non-negative")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        _validate_targets(self.targets)

    def render(self, sample_rate_hz: float) -> FloatArray:
        """Render the sweep sampled at ``sample_rate_hz``."""
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        nyquist = 0.5 * sample_rate_hz
        top = max(self.start_hz, self.end_hz)
        if top > nyquist:
            raise ValueError(f"the sweep reaches {top} Hz, above the Nyquist frequency of {nyquist} Hz")
        n = max(1, int(round(self.duration_seconds * sample_rate_hz)))
        t = np.arange(n, dtype=float) / sample_rate_hz
        rate = (self.end_hz - self.start_hz) / self.duration_seconds
        phase = 2 * np.pi * (self.start_hz * t + 0.5 * rate * t**2)
        wave = self.amplitude * np.sin(phase)
        if self.ramp_seconds > 0:
            rise = np.clip(t / self.ramp_seconds, 0.0, 1.0)
            fall = np.clip((self.duration_seconds - t) / self.ramp_seconds, 0.0, 1.0)
            wave = wave * rise * fall
        return wave


def build_drive_array(
    drives_with_onsets: list[tuple[float, Drive]],
    regions: RegionSet,
    params: SimulationParams,
) -> FloatArray:
    """
    Render ``(onset_seconds, Drive)`` pairs into a ``(T_total, N)`` stimulus array.

    Onsets are relative to the post-burn-in window; the burn-in offset is added
    internally.  Overlapping drives are summed.

    Parameters
    ----------
    drives_with_onsets : list of (onset_seconds, Drive)
        Each tuple pairs an onset time (relative to the post-burn-in window)
        with a :class:`Drive` instance to render.
    regions : RegionSet
        Region ordering for stimulus columns.
    params : SimulationParams
        Simulation timing parameters.

    Returns
    -------
    FloatArray of shape (n_burnin + n_sim, N)
        Deterministic external input, ready to pass as ``stim`` to
        :func:`~diaxcondel.simulate.engine.simulate`.
    """
    n_total = params.n_burnin_samples + params.n_sim_samples
    n = len(regions)
    stim = np.zeros((n_total, n), dtype=float)
    origin = params.n_burnin_samples

    for onset_seconds, drive in drives_with_onsets:
        waveform = drive.render(float(params.sample_rate_hz))
        start = origin + int(round(onset_seconds * params.sample_rate_hz))
        if start >= n_total:
            continue
        end = min(n_total, start + waveform.size)
        wf = waveform[: end - start]
        for target in drive.targets:
            idx = regions.index(target.code)
            stim[start:end, idx] += target.weight * wf

    return stim
