"""Tests for spectral summary helpers."""

import numpy as np

from diaxcondel.analysis.aperiodic import fit_loglog_slope
from diaxcondel.analysis.peaks import find_band_peak


def test_fit_loglog_slope_recovers_power_law():
    freqs = np.linspace(1.0, 40.0, 200)
    power = 3.0 * freqs**-1.5

    fit = fit_loglog_slope(freqs, power, fmin_hz=2.0, fmax_hz=30.0)

    assert np.isclose(fit.slope, -1.5)
    assert fit.r_squared > 0.999


def test_find_band_peak_returns_original_index():
    freqs = np.array([4.0, 8.0, 10.0, 12.0, 20.0])
    power = np.array([1.0, 2.0, 5.0, 3.0, 1.0])

    peak = find_band_peak(freqs, power, fmin_hz=8.0, fmax_hz=13.0)

    assert peak.frequency_hz == 10.0
    assert peak.power == 5.0
    assert peak.index == 2
