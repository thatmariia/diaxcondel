"""Tests for display smoothing and aperiodic fitting."""

import numpy as np
import pytest

from diaxcondel.analysis import (
    fit_aperiodic,
    moving_average,
    smooth_over_frequency,
    smooth_over_time,
)
from diaxcondel.analysis.aperiodic import specparam_available


def test_moving_average_preserves_shape_and_mean():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((200, 3))
    smoothed = moving_average(values, 11)
    assert smoothed.shape == values.shape
    assert smoothed.mean() == pytest.approx(values.mean(), abs=0.05)
    assert smoothed.std() < values.std(), "smoothing must reduce variability"
    np.testing.assert_array_equal(moving_average(values, 1), values)


def test_moving_average_leaves_a_constant_untouched():
    values = np.full((50, 2), 3.0)
    np.testing.assert_allclose(moving_average(values, 9), values)


def test_time_smoothing_suppresses_fast_structure():
    fs = 1000.0
    t = np.arange(2000) / fs
    slow = np.sin(2 * np.pi * 5 * t)
    fast = np.sin(2 * np.pi * 200 * t)
    signal = (slow + fast)[:, None]
    smoothed = smooth_over_time(signal, fs, window_ms=20.0)[:, 0]
    # The 200 Hz component is far above the 50 Hz cut-off of a 20 ms window.
    assert np.std(smoothed - slow) < 0.2
    np.testing.assert_array_equal(smooth_over_time(signal, fs, 0.0), signal)


def test_frequency_smoothing_widens_a_narrow_peak():
    freqs = np.linspace(0, 50, 501)
    spectrum = np.exp(-0.5 * ((freqs - 10) / 0.2) ** 2)
    smoothed = smooth_over_frequency(freqs, spectrum, bandwidth_hz=2.0)
    assert smoothed.max() < spectrum.max()
    assert np.trapezoid(smoothed, freqs) == pytest.approx(np.trapezoid(spectrum, freqs), rel=0.1)
    np.testing.assert_array_equal(smooth_over_frequency(freqs, spectrum, 0.0), spectrum)


def _synthetic_spectrum(exponent: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.linspace(1.0, 45.0, 400)
    power = 10.0 * freqs**-exponent + 3.0 * np.exp(-0.5 * ((freqs - 10.0) / 1.5) ** 2)
    return freqs, power


def test_loglog_fit_reports_a_slope_and_its_quality():
    freqs, power = _synthetic_spectrum(1.4)
    fit = fit_aperiodic(freqs, power, fmin_hz=1.0, fmax_hz=45.0, method="loglog")
    assert fit.method == "loglog"
    assert fit.peaks == ()
    # The peak pulls the straight-line fit away from the true exponent.
    assert fit.exponent == pytest.approx(1.4, abs=0.4)
    assert 0.0 <= fit.r_squared <= 1.0


@pytest.mark.skipif(not specparam_available(), reason="needs the optional specparam package")
def test_spectral_parameterisation_separates_the_peak_from_the_background():
    freqs, power = _synthetic_spectrum(1.4)
    fitted = fit_aperiodic(freqs, power, fmin_hz=1.0, fmax_hz=45.0)
    naive = fit_aperiodic(freqs, power, fmin_hz=1.0, fmax_hz=45.0, method="loglog")
    assert fitted.method == "specparam"
    assert fitted.exponent == pytest.approx(1.4, abs=0.1)
    assert abs(fitted.exponent - 1.4) < abs(naive.exponent - 1.4)
    assert any(abs(peak[0] - 10.0) < 1.5 for peak in fitted.peaks)


def test_a_spectrum_with_no_structure_falls_back_to_the_straight_line():
    """A model the fit reproduces exactly leaves nothing for a goodness-of-fit.

    An uncoupled network's spectrum is its input spectrum, which the
    parameterisation matches perfectly; its own R-squared then divides by a
    zero-variance residual. The result must be a usable number from the
    straight-line route, not a NaN in the summary table.
    """
    freqs = np.linspace(1.0, 45.0, 200)
    flat = np.full_like(freqs, 2.0)
    fitted = fit_aperiodic(freqs, flat, fmin_hz=2.0, fmax_hz=40.0)
    assert np.isfinite(fitted.exponent)
    assert np.isfinite(fitted.r_squared)
    assert fitted.exponent == pytest.approx(0.0, abs=1e-6)
