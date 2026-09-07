"""
Display smoothing for noisy estimates.

Simulated signals and the spectra estimated from them are noisy: that noise
is part of the model's output, not an artefact to be hidden. Smoothing here
is therefore a *display* operation — it never feeds back into a model, a fit,
or a reported statistic, and every function says how much it averaged.

Two kinds are provided: a moving average over time (for traces) and a moving
average over frequency (for spectra and coherence), both centred so nothing
is shifted.
"""

from __future__ import annotations

import numpy as np

from diaxcondel._typing import FloatArray


def moving_average(values: FloatArray, window: int, *, axis: int = 0) -> FloatArray:
    """
    Average values over a centred sliding window.

    Parameters
    ----------
    values : FloatArray
        Data to smooth.
    window : int
        Number of samples in the window. Values below two return the input
        unchanged; even values are rounded up to keep the window centred.
    axis : int
        Axis to smooth along.

    Returns
    -------
    FloatArray
        Smoothed data, the same shape as the input. Edges are handled by
        reflecting the data, so the ends are not pulled towards zero.
    """
    values = np.asarray(values, dtype=float)
    window = int(window)
    if window < 2:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window

    def smooth_1d(column: FloatArray) -> FloatArray:
        padded = np.pad(column, pad, mode="reflect")
        return np.convolve(padded, kernel, mode="valid")

    return np.apply_along_axis(smooth_1d, axis, values)


def smooth_over_time(signal: FloatArray, sample_rate_hz: float, window_ms: float) -> FloatArray:
    """
    Smooth a time series with a moving average of a given duration.

    Parameters
    ----------
    signal : FloatArray of shape (T,) or (T, N)
        Time-domain signal.
    sample_rate_hz : float
        Sampling rate.
    window_ms : float
        Window length in milliseconds. ``0`` returns the signal unchanged.

    Returns
    -------
    FloatArray
        Smoothed signal, same shape as the input.

    Notes
    -----
    A moving average is a low-pass filter whose first null sits at
    ``1000 / window_ms`` Hz, so a 20 ms window removes structure above about
    50 Hz. Use it to see slow structure, not to make claims about it.
    """
    if window_ms <= 0 or sample_rate_hz <= 0:
        return np.asarray(signal, dtype=float)
    window = int(round(window_ms * 1e-3 * sample_rate_hz))
    return moving_average(np.asarray(signal, dtype=float), window, axis=0)


def smooth_over_frequency(freqs_hz: FloatArray, values: FloatArray, bandwidth_hz: float) -> FloatArray:
    """
    Smooth a spectrum with a moving average of a given bandwidth.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid; assumed evenly spaced.
    values : FloatArray of shape (F,) or (F, ...)
        Spectral values (power, coherence, ...).
    bandwidth_hz : float
        Width of the averaging window in Hertz. ``0`` returns the input
        unchanged.

    Returns
    -------
    FloatArray
        Smoothed values, same shape as the input.

    Notes
    -----
    Averaging neighbouring frequency bins trades resolution for stability:
    a peak narrower than ``bandwidth_hz`` is flattened, so keep the window
    well below the width of any feature being judged.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    values = np.asarray(values, dtype=float)
    if bandwidth_hz <= 0 or freqs_hz.size < 2:
        return values
    spacing = float(np.median(np.diff(freqs_hz)))
    if spacing <= 0:
        return values
    window = int(round(bandwidth_hz / spacing))
    return moving_average(values, window, axis=0)
