"""Peak summaries for one-dimensional spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diaxcondel._typing import FloatArray


@dataclass(frozen=True, slots=True)
class BandPeak:
    """
    Maximum-power frequency inside a band.

    Attributes
    ----------
    frequency_hz : float
        Frequency where power is maximal.
    power : float
        Power at ``frequency_hz``.
    index : int
        Index into the original arrays.
    """

    frequency_hz: float
    power: float
    index: int


def find_band_peak(
    freqs_hz: FloatArray,
    power: FloatArray,
    *,
    fmin_hz: float,
    fmax_hz: float,
) -> BandPeak:
    """
    Return the maximum-power point within a frequency band.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid in Hertz.
    power : FloatArray of shape (F,)
        Power values aligned to ``freqs_hz``.
    fmin_hz, fmax_hz : float
        Inclusive search band.

    Returns
    -------
    BandPeak
        Frequency, power, and original index of the band maximum.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    power = np.asarray(power, dtype=float)
    if freqs_hz.ndim != 1 or power.ndim != 1:
        raise ValueError("freqs_hz and power must be one-dimensional")
    if freqs_hz.shape != power.shape:
        raise ValueError(f"freqs_hz shape {freqs_hz.shape} != power shape {power.shape}")
    if fmin_hz > fmax_hz:
        raise ValueError("fmin_hz must be <= fmax_hz")
    if not np.all(np.isfinite(freqs_hz)) or not np.all(np.isfinite(power)):
        raise ValueError("freqs_hz and power must contain only finite values")

    mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz)
    if not np.any(mask):
        raise ValueError("no frequency bins fall within the requested band")
    band_indices = np.flatnonzero(mask)
    local = int(np.argmax(power[mask]))
    index = int(band_indices[local])
    return BandPeak(frequency_hz=float(freqs_hz[index]), power=float(power[index]), index=index)
