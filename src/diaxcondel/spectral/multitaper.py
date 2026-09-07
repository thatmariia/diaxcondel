"""
Multitaper power spectral density.

Welch's method controls the variance of a spectral estimate by cutting the
signal into segments; the multitaper method does it by tapering the *same*
data with several orthogonal windows and averaging the results. For a fixed
amount of data it gives a smoother estimate at the same frequency
resolution, which matters when a simulation is short or a spectral feature
is narrow.

The tapers are discrete prolate spheroidal sequences (Slepian sequences)
from :mod:`scipy.signal.windows`, so the numerics come from a mature
implementation rather than this package.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import dpss

from diaxcondel._typing import FloatArray


def multitaper_psd(
    signal: FloatArray,
    sample_rate_hz: float,
    *,
    time_bandwidth: float = 4.0,
    n_tapers: int | None = None,
    detrend: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """
    Estimate the power spectral density with Slepian tapers.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Multivariate signal.
    sample_rate_hz : float
        Sampling rate.
    time_bandwidth : float
        Time-bandwidth product ``NW``. It sets the spectral smoothing:
        the estimate is smoothed over roughly ``2 * NW / T`` Hz, so larger
        values are smoother and blur narrow peaks.
    n_tapers : int, optional
        Number of tapers to average. ``None`` uses ``2 * NW - 1``, the
        largest number whose tapers remain well concentrated.
    detrend : bool
        Remove the mean of each channel before estimating.

    Returns
    -------
    freqs : FloatArray of shape (F,)
        One-sided frequency grid.
    psd : FloatArray of shape (F, N)
        Power spectral density per channel, in units of signal squared per
        Hertz — the same convention as
        :func:`~diaxcondel.spectral.empirical.welch_psd`.

    Raises
    ------
    ValueError
        If the input is malformed or the settings leave no usable taper.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError(f"signal must have shape (T, N); got {signal.shape}")
    n_samples, n_channels = signal.shape
    if n_samples < 4 or n_channels < 1:
        raise ValueError("signal must have at least four samples and one channel")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if time_bandwidth < 1.0:
        raise ValueError("time_bandwidth must be at least 1")

    if n_tapers is None:
        n_tapers = max(1, int(2 * time_bandwidth) - 1)
    if n_tapers < 1:
        raise ValueError("n_tapers must be at least 1")

    data = signal - signal.mean(axis=0, keepdims=True) if detrend else signal
    tapers = np.asarray(dpss(n_samples, time_bandwidth, n_tapers), dtype=float)

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    psd = np.zeros((freqs.size, n_channels))
    for taper in tapers:
        spectrum = np.fft.rfft(data * taper[:, None], axis=0)
        psd += np.abs(spectrum) ** 2
    psd /= n_tapers * sample_rate_hz

    # One-sided scaling: fold the negative frequencies onto the positive
    # ones. DC is never doubled; the final bin is the Nyquist frequency only
    # for an even-length record, and for an odd one it is an ordinary
    # positive frequency that must be.
    if freqs.size > 1:
        last = -1 if n_samples % 2 == 0 else None
        psd[1:last] *= 2.0
    return freqs, psd
