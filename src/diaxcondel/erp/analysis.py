"""
ERP analysis helpers.

These functions operate on raw time-domain signals (``FloatArray`` of shape
``(T, N)`` or ``(n_trials, T, N)``) without knowledge of the VAR model.
They are intentionally small and composable so they can be used both on
:class:`~diaxcondel.simulate.trials.TrialResult` data and on arbitrary signals.

Units
-----
- Time is always in **seconds** when exposed to the user.
- Samples are computed as ``int(round(seconds * sample_rate_hz))``.
- ``sample_rate_hz`` is always passed explicitly; there is no hidden state.
"""

from __future__ import annotations

import numpy as np

from diaxcondel._typing import FloatArray


def baseline_correct(
    signal: FloatArray,
    baseline_window_seconds: tuple[float, float],
    sample_rate_hz: float,
) -> FloatArray:
    """
    Subtract the mean over a baseline window from each channel.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Time-domain signal.
    baseline_window_seconds : (start, end)
        Inclusive-start, exclusive-end window in seconds from the beginning
        of ``signal``.  The mean over this window is subtracted from every
        sample.
    sample_rate_hz : float
        Sampling rate in Hertz.

    Returns
    -------
    FloatArray of shape (T, N)
        Baseline-corrected signal.

    Raises
    ------
    ValueError
        If the window extends beyond the signal or is empty.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (T, N); got shape {signal.shape}")
    t_total = signal.shape[0]
    start_s, end_s = baseline_window_seconds
    if start_s >= end_s:
        raise ValueError(f"baseline window start ({start_s}) must be < end ({end_s})")
    start_idx = int(round(start_s * sample_rate_hz))
    end_idx = int(round(end_s * sample_rate_hz))
    if start_idx < 0 or end_idx > t_total or start_idx >= end_idx:
        raise ValueError(f"baseline window [{start_idx}, {end_idx}) is out of bounds for signal length {t_total}")
    baseline_mean = signal[start_idx:end_idx].mean(axis=0, keepdims=True)
    return signal - baseline_mean


def average_trials(trials: FloatArray) -> FloatArray:
    """
    Compute the trial average.

    Parameters
    ----------
    trials : FloatArray of shape (n_trials, T, N)
        Per-trial signals.

    Returns
    -------
    FloatArray of shape (T, N)
        Mean across trials (axis 0).
    """
    if trials.ndim != 3:
        raise ValueError(f"trials must be 3-D (n_trials, T, N); got shape {trials.shape}")
    return trials.mean(axis=0)


def sem_trials(trials: FloatArray) -> FloatArray:
    """
    Compute the standard error of the mean across trials.

    Parameters
    ----------
    trials : FloatArray of shape (n_trials, T, N)
        Per-trial signals.

    Returns
    -------
    FloatArray of shape (T, N)
        SEM across trials.
    """
    if trials.ndim != 3:
        raise ValueError(f"trials must be 3-D (n_trials, T, N); got shape {trials.shape}")
    n = trials.shape[0]
    if n < 2:
        return np.full(trials.shape[1:], np.nan)
    return trials.std(axis=0, ddof=1) / np.sqrt(n)


def epoch_around_event(
    signal: FloatArray,
    event_sample: int,
    pre_samples: int,
    post_samples: int,
) -> FloatArray:
    """
    Extract an epoch centred around an event sample.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Continuous time-domain signal.
    event_sample : int
        Sample index of the event (e.g. stimulus onset).
    pre_samples : int
        Number of samples before the event to include.  The epoch starts at
        ``event_sample - pre_samples``.
    post_samples : int
        Number of samples after the event to include.  The epoch ends at
        (exclusive) ``event_sample + post_samples``.

    Returns
    -------
    FloatArray of shape (pre_samples + post_samples, N)
        Extracted epoch.

    Raises
    ------
    ValueError
        If the epoch would extend outside ``signal``.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (T, N); got shape {signal.shape}")
    t_total = signal.shape[0]
    start = event_sample - pre_samples
    end = event_sample + post_samples
    if start < 0 or end > t_total:
        raise ValueError(f"epoch [{start}, {end}) extends outside signal of length {t_total}")
    return signal[start:end]


def peak_amplitude(
    signal: FloatArray,
    search_window_seconds: tuple[float, float],
    sample_rate_hz: float,
) -> FloatArray:
    """
    Return the peak (max absolute) amplitude in a time window.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Time-domain signal.
    search_window_seconds : (start, end)
        Time window in seconds over which to search.
    sample_rate_hz : float
        Sampling rate in Hertz.

    Returns
    -------
    FloatArray of shape (N,)
        Peak absolute amplitude for each region.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (T, N); got shape {signal.shape}")
    t_total = signal.shape[0]
    start_s, end_s = search_window_seconds
    start_idx = int(round(start_s * sample_rate_hz))
    end_idx = int(round(end_s * sample_rate_hz))
    start_idx = max(0, min(start_idx, t_total))
    end_idx = max(0, min(end_idx, t_total))
    if start_idx >= end_idx:
        raise ValueError(f"search window is empty after rounding: [{start_idx}, {end_idx})")
    return np.max(np.abs(signal[start_idx:end_idx]), axis=0)


def peak_latency(
    signal: FloatArray,
    search_window_seconds: tuple[float, float],
    sample_rate_hz: float,
) -> FloatArray:
    """
    Return the latency of the peak absolute amplitude in a time window.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Time-domain signal.
    search_window_seconds : (start, end)
        Time window in seconds over which to search.
    sample_rate_hz : float
        Sampling rate in Hertz.

    Returns
    -------
    FloatArray of shape (N,)
        Latency in seconds (relative to the start of ``signal``) for each
        region.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (T, N); got shape {signal.shape}")
    t_total = signal.shape[0]
    start_s, end_s = search_window_seconds
    start_idx = int(round(start_s * sample_rate_hz))
    end_idx = int(round(end_s * sample_rate_hz))
    start_idx = max(0, min(start_idx, t_total))
    end_idx = max(0, min(end_idx, t_total))
    if start_idx >= end_idx:
        raise ValueError(f"search window is empty after rounding: [{start_idx}, {end_idx})")
    window = signal[start_idx:end_idx]
    peak_indices = np.argmax(np.abs(window), axis=0)
    return (start_idx + peak_indices) / sample_rate_hz
