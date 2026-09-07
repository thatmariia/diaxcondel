"""Pairwise coherence."""

from __future__ import annotations

import numpy as np
from scipy import signal as sig_

from diaxcondel._typing import FloatArray


def pairwise_coherence(
    signal: FloatArray,
    sample_rate_hz: float,
    *,
    segment_seconds: float = 1.0,
    overlap: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """
    Magnitude-squared coherence between every pair of channels.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Multivariate signal.
    sample_rate_hz : float
        Sampling rate.
    segment_seconds, overlap
        Welch parameters.

    Returns
    -------
    freqs : FloatArray of shape (F,)
    coh : FloatArray of shape (F, N, N)
        Symmetric coherence matrix per frequency. Diagonal is 1.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must have shape (T, N); got {signal.shape}")
    if signal.shape[0] < 2:
        raise ValueError("signal must have at least two samples")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1)")
    n = signal.shape[1]
    if n == 0:
        raise ValueError("signal has zero channels")
    nperseg = int(round(segment_seconds * sample_rate_hz))
    if nperseg < 2:
        raise ValueError("segment_seconds is too short for the sample rate")
    nperseg = min(nperseg, signal.shape[0])
    noverlap = int(round(overlap * nperseg))

    # Run i=j=0 once to discover the frequency grid and allocate.
    freqs, _ = sig_.coherence(
        signal[:, 0],
        signal[:, 0],
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    coh = np.ones((freqs.size, n, n))  # diagonal is 1 by construction
    for i in range(n):
        for j in range(i + 1, n):
            _, c = sig_.coherence(
                signal[:, i],
                signal[:, j],
                fs=sample_rate_hz,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            coh[:, i, j] = c
            coh[:, j, i] = c
    return freqs, coh  # type: ignore[return-value]
