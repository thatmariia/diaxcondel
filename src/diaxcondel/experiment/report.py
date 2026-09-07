"""
Standard summaries of a run: spectra, per-region features, reproducible code.

These helpers exist so that the dashboard, notebooks, and the command line
all report the *same* numbers, computed the same way — in particular the
scaling that lets the analytical and empirical spectra be plotted on one
axis.

Matching the two spectra
------------------------
The noise generators draw per-sample variance ``std**2 * fs`` so that
integrated power per unit time does not depend on the sample rate. For a VAR
driven by that noise, the one-sided power spectral density is
``2 * std**2 * |T(f)|**2`` — which is what :func:`compute_spectra` passes as
the noise covariance, so the analytical curve lands on top of the Welch
estimate instead of a decade away from it.

The analytical spectrum exists only for a linear model driven by white noise.
With a saturating transfer or a coloured input, it is omitted and the reason
is recorded in :attr:`Spectra.notes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diaxcondel._typing import FloatArray
from diaxcondel.analysis.aperiodic import fit_aperiodic, fit_loglog_slope
from diaxcondel.analysis.peaks import find_band_peak
from diaxcondel.spectral.analytical import analytical_psd
from diaxcondel.spectral.coherence import pairwise_coherence
from diaxcondel.spectral.empirical import welch_psd
from diaxcondel.spectral.multitaper import multitaper_psd

from .build import BuiltModel
from .run import RunResult
from .spec import ExperimentSpec

#: Canonical alpha band (Hz) used for the peak summary.
ALPHA_BAND_HZ = (8.0, 13.0)

#: Default band (Hz) for the aperiodic log-log slope fit.
SLOPE_BAND_HZ = (2.0, 40.0)


@dataclass(frozen=True, slots=True)
class Spectra:
    """
    Empirical and (where available) analytical spectra of a run.

    Attributes
    ----------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid of the simulated estimate.
    welch : FloatArray of shape (F, N)
        Trial-averaged PSD per region, estimated from the simulation.
    estimator : str
        Which estimator produced it: ``"welch"`` or ``"multitaper"``.
    analytic_freqs_hz : FloatArray of shape (G,) or None
        Frequency grid of the closed-form spectrum.
    analytic : FloatArray of shape (G, N) or None
        Closed-form PSD per region, on the same scale as ``welch``.
    notes : tuple of str
        Why the analytical spectrum is missing, when it is.
    """

    freqs_hz: FloatArray
    welch: FloatArray
    analytic_freqs_hz: FloatArray | None = None
    analytic: FloatArray | None = None
    estimator: str = "welch"
    notes: tuple[str, ...] = ()

    @property
    def has_analytic(self) -> bool:
        """Whether a closed-form spectrum was computed."""
        return self.analytic is not None


@dataclass(frozen=True, slots=True)
class RegionSummary:
    """
    Per-region spectral features.

    Attributes
    ----------
    code : str
        Region code.
    peak_hz : float
        Frequency of maximum power inside the alpha band.
    peak_power : float
        Power at that frequency.
    slope : float
        Straight-line slope of log power against log frequency over the
        fitting band. Negative for a falling spectrum.
    r_squared : float
        Goodness of that straight-line fit.
    variance : float
        Time-domain variance of the first trial.
    exponent : float
        Aperiodic exponent: power falls as ``1 / f ** exponent``. When the
        optional spectral-parameterisation package is installed this comes
        from a fit that separates peaks from the background, and so is not
        simply the negated slope.
    fit_method : str
        Which route produced ``exponent``: ``"specparam"`` or ``"loglog"``.
    """

    code: str
    peak_hz: float
    peak_power: float
    slope: float
    r_squared: float
    variance: float
    exponent: float = float("nan")
    fit_method: str = "loglog"


@dataclass(frozen=True, slots=True)
class DelayStatistics:
    """
    The delays a built model actually applies, per connection.

    Attributes
    ----------
    lags_ms : FloatArray of shape (P,)
        Lag axis in milliseconds.
    coupling : FloatArray of shape (N, N)
        Total weight of each connection, summed over lags.
    mean_delay_ms : FloatArray of shape (N, N)
        Weighted mean delay of each connection; ``nan`` where there is no
        connection.
    peak_delay_ms : FloatArray of shape (N, N)
        Lag at which each connection's weight is largest; ``nan`` where there
        is no connection.
    """

    lags_ms: FloatArray
    coupling: FloatArray
    mean_delay_ms: FloatArray
    peak_delay_ms: FloatArray

    @property
    def connected(self) -> FloatArray:
        """Boolean mask of pairs carrying a connection."""
        return self.coupling > 0

    @property
    def resonance_hz(self) -> FloatArray:
        """
        Frequency implied by each connection's peak delay.

        A delayed loop between two regions returns to its source after twice
        the transmission delay, so it reinforces frequencies near
        ``1 / (2 * peak delay)`` — the delay-driven resonance the model is
        built on.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return 1000.0 / (2.0 * self.peak_delay_ms)

    def describe(self) -> dict[str, float]:
        """
        Return a compact summary of the delay structure.

        Returns
        -------
        dict
            Mean, shortest and longest connection delay (ms), the frequency
            band those delays imply (Hz), and the total coupling.
        """
        mask = self.connected
        if not np.any(mask):
            return {}
        delays = self.mean_delay_ms[mask]
        peaks = self.peak_delay_ms[mask]
        return {
            "mean delay (ms)": float(np.nanmean(delays)),
            "shortest delay (ms)": float(np.nanmin(delays)),
            "longest delay (ms)": float(np.nanmax(delays)),
            "lowest implied frequency (Hz)": float(1000.0 / (2.0 * np.nanmax(peaks))),
            "highest implied frequency (Hz)": float(1000.0 / (2.0 * np.nanmin(peaks))),
            "total coupling": float(self.coupling.sum()),
        }


def delay_statistics(built: BuiltModel) -> DelayStatistics:
    """
    Summarise the delays in a built model's lag tensor.

    Parameters
    ----------
    built : BuiltModel
        A model built from a specification.

    Returns
    -------
    DelayStatistics
        Per-connection coupling, mean delay, peak delay, and the frequencies
        those delays imply.
    """
    phi = built.phi
    dt_ms = 1000.0 * built.params.dt_s
    lags_ms = np.arange(1, phi.shape[0] + 1) * dt_ms
    coupling = phi.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_delay = np.einsum("p,pij->ij", lags_ms, phi) / np.where(coupling > 0, coupling, np.nan)
    peak_delay = np.where(coupling > 0, lags_ms[np.argmax(phi, axis=0)], np.nan)
    return DelayStatistics(
        lags_ms=lags_ms,
        coupling=coupling,
        mean_delay_ms=mean_delay,
        peak_delay_ms=peak_delay,
    )


def _white_noise_covariance(spec: ExperimentSpec, n_regions: int) -> FloatArray | None:
    """
    Return the driving-noise covariance when the input spectrum is flat.

    The closed-form spectrum needs the input to be white — flat in frequency
    — but it does not need it to be independent across regions. Shared white
    input has a known covariance, so a correlated model keeps its analytical
    curve. Anything with a sloped input spectrum returns ``None``.

    Parameters
    ----------
    spec : ExperimentSpec
        The specification, whose noise entry names the input.
    n_regions : int
        Model dimension.

    Returns
    -------
    FloatArray of shape (N, N) or None
        Covariance on the one-sided density scale, or ``None`` when the
        input is not white.

    Notes
    -----
    The noise generators draw per-sample variance ``std**2 * fs``, whose
    one-sided density is ``2 * std**2`` — hence the factor of two.
    """
    name = spec.noise.name
    params = spec.noise.params
    std = float(params.get("std", 1.0))
    if name == "white":
        return 2.0 * std**2 * np.eye(n_regions)
    if name == "correlated" and float(params.get("exponent", 0.0)) == 0.0:
        correlation = float(params.get("correlation", 0.0))
        shared = correlation * np.ones((n_regions, n_regions))
        independent = (1.0 - correlation) * np.eye(n_regions)
        return 2.0 * std**2 * (independent + shared)
    if name == "colored" and float(params.get("exponent", 1.0)) == 0.0:
        return 2.0 * std**2 * np.eye(n_regions)
    return None


def compute_spectra(
    run: RunResult,
    *,
    estimator: str = "welch",
    segment_seconds: float = 2.0,
    overlap: float = 0.5,
    time_bandwidth: float = 4.0,
    fmin_hz: float = 1.0,
    fmax_hz: float = 45.0,
    n_analytic_points: int = 400,
) -> Spectra:
    """
    Compute the trial-averaged Welch PSD and, when valid, the analytical one.

    Parameters
    ----------
    run : RunResult
        A completed run.
    estimator : {"welch", "multitaper"}
        How to estimate the spectrum from the simulated signal. Welch
        averages over segments; the multitaper estimate averages over
        orthogonal tapers of the whole record, which is smoother at the same
        frequency resolution.
    segment_seconds : float
        Welch segment length, in seconds.
    overlap : float
        Fractional overlap between Welch segments.
    time_bandwidth : float
        Time-bandwidth product for the multitaper estimate.
    fmin_hz, fmax_hz : float
        Frequency range of the analytical curve.
    n_analytic_points : int
        Number of points on the analytical frequency grid.

    Returns
    -------
    Spectra
        Spectra on a common scale, plus notes about anything omitted.
    """
    params = run.params
    segment_seconds = min(segment_seconds, params.sim_seconds)

    def estimate(trial: FloatArray) -> tuple[FloatArray, FloatArray]:
        if estimator == "multitaper":
            return multitaper_psd(trial, params.sample_rate_hz, time_bandwidth=time_bandwidth)
        if estimator != "welch":
            raise ValueError(f"unknown estimator {estimator!r}; expected 'welch' or 'multitaper'")
        return welch_psd(trial, params.sample_rate_hz, segment_seconds=segment_seconds, overlap=overlap)

    freqs, psd = estimate(run.trials.trials[0])
    accumulated = np.asarray(psd, dtype=float)
    for trial in run.trials.trials[1:]:
        _, extra = estimate(trial)
        accumulated = accumulated + np.asarray(extra, dtype=float)
    welch = accumulated / run.trials.n_trials

    notes: list[str] = []
    analytic_freqs: FloatArray | None = None
    analytic: FloatArray | None = None

    noise_cov = _white_noise_covariance(run.spec, run.built.n_regions)
    if not run.built.is_linear:
        notes.append("no closed-form spectrum for a non-linear transfer; showing the simulated spectrum only")
    elif noise_cov is None:
        notes.append(
            f"the closed-form spectrum needs a flat input spectrum, but the noise is "
            f"{run.spec.noise.name!r}; showing the simulated spectrum only"
        )
    else:
        analytic_freqs = np.linspace(max(fmin_hz, 1e-3), fmax_hz, n_analytic_points)
        full = analytical_psd(run.built.dynamics, analytic_freqs, params.dt_s, noise_cov=noise_cov)
        analytic = np.diagonal(full, axis1=1, axis2=2).copy()

    return Spectra(
        freqs_hz=freqs,
        welch=welch,
        analytic_freqs_hz=analytic_freqs,
        analytic=analytic,
        estimator=estimator,
        notes=tuple(notes),
    )


def region_summaries(
    run: RunResult,
    spectra: Spectra,
    *,
    alpha_band_hz: tuple[float, float] = ALPHA_BAND_HZ,
    slope_band_hz: tuple[float, float] = SLOPE_BAND_HZ,
    aperiodic_method: str = "auto",
) -> tuple[RegionSummary, ...]:
    """
    Summarise each region's spectrum: band peak, aperiodic slope, variance.

    Parameters
    ----------
    run : RunResult
        The run the spectra came from.
    spectra : Spectra
        Spectra computed by :func:`compute_spectra`. The analytical curve is
        used when available (it is noise-free), otherwise the Welch estimate.
    alpha_band_hz : tuple of float
        Band searched for the spectral peak.
    slope_band_hz : tuple of float
        Band used for the aperiodic fit.
    aperiodic_method : {"auto", "specparam", "loglog"}
        How to estimate the aperiodic exponent; ``"auto"`` uses spectral
        parameterisation when that optional package is installed.

    Returns
    -------
    tuple of RegionSummary
        One entry per region, in model order.
    """
    if spectra.has_analytic and spectra.analytic_freqs_hz is not None and spectra.analytic is not None:
        freqs = np.asarray(spectra.analytic_freqs_hz, dtype=float)
        power = np.asarray(spectra.analytic, dtype=float)
    else:
        freqs = np.asarray(spectra.freqs_hz, dtype=float)
        power = np.asarray(spectra.welch, dtype=float)

    signal = run.signal
    out: list[RegionSummary] = []
    for index, code in enumerate(run.regions.codes):
        column = power[:, index]
        peak = find_band_peak(freqs, column, fmin_hz=alpha_band_hz[0], fmax_hz=alpha_band_hz[1])
        aperiodic = fit_aperiodic(
            freqs,
            column,
            fmin_hz=slope_band_hz[0],
            fmax_hz=slope_band_hz[1],
            method=aperiodic_method,
        )
        fit = fit_loglog_slope(freqs, column, fmin_hz=slope_band_hz[0], fmax_hz=slope_band_hz[1])
        out.append(
            RegionSummary(
                code=code,
                peak_hz=peak.frequency_hz,
                peak_power=peak.power,
                slope=fit.slope,
                r_squared=fit.r_squared,
                variance=float(np.var(signal[:, index])),
                exponent=aperiodic.exponent,
                fit_method=aperiodic.method,
            )
        )
    return tuple(out)


def compute_coherence(
    run: RunResult,
    *,
    segment_seconds: float = 2.0,
    overlap: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """
    Magnitude-squared coherence between regions, from the first trial.

    Parameters
    ----------
    run : RunResult
        A completed run.
    segment_seconds : float
        Welch segment length, in seconds.
    overlap : float
        Fractional overlap between segments.

    Returns
    -------
    freqs_hz : FloatArray of shape (F,)
        Frequency grid.
    coherence : FloatArray of shape (F, N, N)
        Symmetric coherence matrix per frequency.
    """
    return pairwise_coherence(
        run.signal,
        run.params.sample_rate_hz,
        segment_seconds=min(segment_seconds, run.params.sim_seconds),
        overlap=overlap,
    )


def spec_to_python(spec: ExperimentSpec) -> str:
    """
    Render a runnable script that reproduces an experiment exactly.

    Parameters
    ----------
    spec : ExperimentSpec
        The specification to reproduce.

    Returns
    -------
    str
        A short Python script embedding the spec as JSON. It reproduces the
        run bit for bit, because every stochastic step derives from
        ``spec.seed``.
    """
    return (
        '"""Reproduce a diaxcondel experiment."""\n\n'
        "from diaxcondel.experiment import ExperimentSpec, compute_spectra, run_experiment\n\n"
        'spec = ExperimentSpec.model_validate_json(\n    """\n'
        f"{spec.to_json()}\n"
        '    """\n)\n\n'
        "result = run_experiment(spec)\n"
        "spectra = compute_spectra(result)\n"
        "print(result.signal.shape, spectra.freqs_hz.shape)\n"
    )
