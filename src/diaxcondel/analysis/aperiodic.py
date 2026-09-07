"""
Aperiodic spectral summaries.

Two routes are offered. The straight-line fit to log power against log
frequency (:func:`fit_loglog_slope`) is transparent and dependency-free, but
it is biased by any oscillatory peak sitting on the background.
:func:`fit_aperiodic` prefers the *specparam* implementation of spectral
parameterisation (Donoghue et al., 2020), which separates the aperiodic
background from periodic peaks before reporting an exponent, and falls back
to the straight-line fit when that package is not installed. Both report
which route produced the number.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib.util import find_spec

import numpy as np

from diaxcondel._typing import FloatArray


@dataclass(frozen=True, slots=True)
class LogLogSlope:
    """
    Linear fit to log10 power as a function of log10 frequency.

    Attributes
    ----------
    slope : float
        Fitted log-log slope. A 1/f-like spectrum has a negative value.
    intercept : float
        Fitted intercept in log10 units.
    r_squared : float
        Coefficient of determination for the fit.
    """

    slope: float
    intercept: float
    r_squared: float


def fit_loglog_slope(
    freqs_hz: FloatArray,
    power: FloatArray,
    *,
    fmin_hz: float,
    fmax_hz: float,
) -> LogLogSlope:
    """
    Fit the aperiodic spectral fall-off over a frequency band.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid in Hertz.
    power : FloatArray of shape (F,)
        Non-negative power values aligned to ``freqs_hz``.
    fmin_hz, fmax_hz : float
        Inclusive frequency range used for the fit. Frequencies must be
        strictly positive because the fit is done in log space.

    Returns
    -------
    LogLogSlope
        Slope, intercept, and R-squared of the log-log least-squares fit.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    power = np.asarray(power, dtype=float)
    if freqs_hz.ndim != 1 or power.ndim != 1:
        raise ValueError("freqs_hz and power must be one-dimensional")
    if freqs_hz.shape != power.shape:
        raise ValueError(f"freqs_hz shape {freqs_hz.shape} != power shape {power.shape}")
    if not (0 < fmin_hz < fmax_hz):
        raise ValueError("require 0 < fmin_hz < fmax_hz")
    if not np.all(np.isfinite(freqs_hz)) or not np.all(np.isfinite(power)):
        raise ValueError("freqs_hz and power must contain only finite values")

    mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz) & (freqs_hz > 0) & (power > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two positive frequency/power points are required")

    x = np.log10(freqs_hz[mask])
    y = np.log10(power[mask])
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return LogLogSlope(slope=float(slope), intercept=float(intercept), r_squared=float(r_squared))


@dataclass(frozen=True, slots=True)
class AperiodicFit:
    """
    The aperiodic background of a spectrum, and any peaks found on it.

    Attributes
    ----------
    exponent : float
        Aperiodic exponent: power falls as ``1 / f ** exponent``. Reported
        as a positive number for a falling spectrum, matching the
        convention of the spectral-parameterisation literature (and so
        opposite in sign to :attr:`LogLogSlope.slope`).
    offset : float
        Log-power intercept of the background.
    r_squared : float
        Goodness of fit of the model that produced the exponent.
    method : {"specparam", "loglog"}
        Which route produced the numbers: full spectral parameterisation, or
        a straight-line fit in log-log space.
    peaks : tuple of tuple
        ``(centre_hz, height, bandwidth_hz)`` per detected peak; empty for
        the straight-line route, which does not model peaks.
    """

    exponent: float
    offset: float
    r_squared: float
    method: str
    peaks: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)


def specparam_available() -> bool:
    """Return ``True`` when the optional spectral-parameterisation package is installed."""
    return find_spec("specparam") is not None


def fit_aperiodic(
    freqs_hz: FloatArray,
    power: FloatArray,
    *,
    fmin_hz: float,
    fmax_hz: float,
    method: str = "auto",
    max_peaks: int = 6,
) -> AperiodicFit:
    """
    Estimate the aperiodic background of a spectrum.

    Parameters
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid in Hertz.
    power : FloatArray of shape (F,)
        Non-negative power values aligned to ``freqs_hz``.
    fmin_hz, fmax_hz : float
        Inclusive frequency range used for the fit. Both must be positive.
    method : {"auto", "specparam", "loglog"}
        Which route to use. ``"auto"`` prefers spectral parameterisation when
        the package is installed, and otherwise fits a straight line.
    max_peaks : int
        Maximum number of periodic peaks the spectral-parameterisation route
        may fit alongside the background.

    Returns
    -------
    AperiodicFit
        Exponent, offset, goodness of fit, the route taken, and any peaks.

    Raises
    ------
    ValueError
        If the inputs are malformed, or ``method="specparam"`` is requested
        without the package installed.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    power = np.asarray(power, dtype=float)
    if method not in {"auto", "specparam", "loglog"}:
        raise ValueError(f"unknown method {method!r}; expected 'auto', 'specparam' or 'loglog'")
    if method == "specparam" and not specparam_available():
        raise ValueError("spectral parameterisation needs the optional `specparam` package")

    if method in {"auto", "specparam"} and specparam_available():
        fitted = _fit_with_specparam(freqs_hz, power, fmin_hz, fmax_hz, max_peaks)
        if fitted is not None:
            return fitted
        if method == "specparam":
            raise ValueError("spectral parameterisation failed on this spectrum")

    line = fit_loglog_slope(freqs_hz, power, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
    return AperiodicFit(
        exponent=-line.slope,
        offset=line.intercept,
        r_squared=line.r_squared,
        method="loglog",
        peaks=(),
    )


def _fit_with_specparam(
    freqs_hz: FloatArray,
    power: FloatArray,
    fmin_hz: float,
    fmax_hz: float,
    max_peaks: int,
) -> AperiodicFit | None:
    """Fit with the spectral-parameterisation package, or return ``None`` if it cannot."""
    try:  # pragma: no cover - exercised only with the optional package installed
        from specparam import SpectralModel

        mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz) & (freqs_hz > 0) & (power > 0)
        if np.count_nonzero(mask) < 5:
            return None
        model = SpectralModel(max_n_peaks=max_peaks, verbose=False)
        with warnings.catch_warnings():
            # A spectrum the model reproduces exactly — an uncoupled network,
            # say — leaves a zero-variance residual, and the package's own
            # goodness-of-fit divides by it. The check below turns that into a
            # fallback rather than a NaN in the results table.
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(freqs_hz[mask], power[mask])
        aperiodic = np.atleast_1d(np.asarray(model.get_params("aperiodic"), dtype=float))
        if aperiodic.size < 2:
            return None
        peaks_raw = np.atleast_2d(np.asarray(model.get_params("peak"), dtype=float))
        peaks = tuple(
            (float(row[0]), float(row[1]), float(row[2]))
            for row in peaks_raw
            if row.size >= 3 and np.all(np.isfinite(row[:3]))
        )
        r_squared = _specparam_r_squared(model)
        if not np.isfinite(aperiodic[-1]) or not np.isfinite(r_squared):
            return None
        return AperiodicFit(
            exponent=float(aperiodic[-1]),
            offset=float(aperiodic[0]),
            r_squared=r_squared,
            method="specparam",
            peaks=peaks,
        )
    except Exception:  # noqa: BLE001 - any failure falls back to the straight-line fit
        return None


def _specparam_r_squared(model: object) -> float:
    """Read the fit's R-squared from a spectral model, across package versions."""
    scores = getattr(getattr(model, "results", None), "metrics", None)
    results = getattr(scores, "results", None)
    if isinstance(results, dict):
        for key in ("gof_rsquared", "r_squared"):
            if key in results:
                return float(results[key])
    direct = getattr(model, "r_squared_", None)  # pragma: no cover - older versions
    return float(direct) if direct is not None else float("nan")
