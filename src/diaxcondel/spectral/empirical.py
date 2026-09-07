"""
Empirical PSD estimation from time series.

Welch-style segmentation followed by FFT, with explicit windowing and
sample-rate-correct normalisation.
"""

from __future__ import annotations

from scipy import signal as sig_

from diaxcondel._typing import FloatArray


def welch_psd(
    signal: FloatArray,
    sample_rate_hz: float,
    *,
    segment_seconds: float = 1.0,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[FloatArray, FloatArray]:
    """
    Welch PSD per channel.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Multivariate signal.
    sample_rate_hz : float
        Sampling rate.
    segment_seconds : float
        Window length.
    overlap : float
        Fractional overlap between consecutive segments, in ``[0, 1)``.
    window : str
        Window name; passed to :func:`scipy.signal.welch`.

    Returns
    -------
    freqs : FloatArray of shape (F,)
    psd : FloatArray of shape (F, N)
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must have shape (T, N); got {signal.shape}")
    if signal.shape[0] < 2 or signal.shape[1] < 1:
        raise ValueError("signal must have at least two samples and one channel")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1)")

    nperseg = int(round(segment_seconds * sample_rate_hz))
    if nperseg < 2:
        raise ValueError("segment_seconds is too short for the sample rate")
    nperseg = min(nperseg, signal.shape[0])
    noverlap = int(round(overlap * nperseg))
    freqs, psd = sig_.welch(
        signal,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        axis=0,
        scaling="density",
    )
    return freqs, psd
