"""Spectral estimators: scale, conventions, and agreement with the closed form."""

import numpy as np
import pytest

from diaxcondel.model.params import SimulationParams
from diaxcondel.model.var import LinearVAR
from diaxcondel.simulate import get_noise_fn, simulate
from diaxcondel.spectral.analytical import analytical_psd
from diaxcondel.spectral.coherence import pairwise_coherence
from diaxcondel.spectral.empirical import welch_psd
from diaxcondel.spectral.multitaper import multitaper_psd

SAMPLE_RATE_HZ = 200.0


@pytest.mark.parametrize("n_samples", [1024, 1025])
@pytest.mark.parametrize("estimator", ["welch", "multitaper"])
def test_integrated_power_equals_variance(n_samples, estimator):
    """Both estimators return a one-sided density: its integral is the variance.

    An odd-length record has no Nyquist bin, so its final bin is an ordinary
    positive frequency and has to be folded like the rest.
    """
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(n_samples, 2))
    if estimator == "welch":
        freqs, psd = welch_psd(signal, SAMPLE_RATE_HZ, segment_seconds=n_samples / SAMPLE_RATE_HZ)
    else:
        freqs, psd = multitaper_psd(signal, SAMPLE_RATE_HZ)
    for channel in range(signal.shape[1]):
        integrated = float(np.trapezoid(psd[:, channel], freqs))
        assert integrated == pytest.approx(signal[:, channel].var(), rel=0.15)


def test_the_two_estimators_agree_on_a_white_signal():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(8192, 1))
    _, welch = welch_psd(signal, SAMPLE_RATE_HZ, segment_seconds=4.0)
    _, taper = multitaper_psd(signal, SAMPLE_RATE_HZ)
    assert np.median(welch) == pytest.approx(np.median(taper), rel=0.15)


def test_the_closed_form_matches_the_simulated_spectrum():
    rng = np.random.default_rng(5)
    phi = rng.normal(0.0, 0.02, size=(20, 3, 3))
    model = LinearVAR(phi=phi)
    params = SimulationParams(sample_rate_hz=500, sim_seconds=300.0, burnin_seconds=2.0, n_lags=20)
    signal = simulate(model, params, get_noise_fn("white"), rng=3).signal

    freqs, empirical = welch_psd(signal, params.sample_rate_hz, segment_seconds=4.0)
    band = (freqs > 2.0) & (freqs < 100.0)
    # Noise of per-sample variance std**2 * fs has one-sided density 2 * std**2.
    closed_form = analytical_psd(model, freqs[band], params.dt_s, noise_cov=2.0 * np.eye(3))
    ratio = empirical[band] / np.diagonal(closed_form, axis1=1, axis2=2)
    assert float(np.median(ratio)) == pytest.approx(1.0, abs=0.05)


def test_coherence_is_bounded_and_symmetric():
    rng = np.random.default_rng(7)
    shared = rng.normal(size=(4096, 1))
    signal = np.hstack([shared + 0.1 * rng.normal(size=(4096, 1)), rng.normal(size=(4096, 1))])
    freqs, coherence = pairwise_coherence(signal, SAMPLE_RATE_HZ, segment_seconds=2.0)

    assert coherence.shape == (freqs.size, 2, 2)
    assert np.all(coherence >= 0.0) and np.all(coherence <= 1.0 + 1e-9)
    assert np.allclose(coherence, np.transpose(coherence, (0, 2, 1)))
    assert np.allclose(np.diagonal(coherence, axis1=1, axis2=2), 1.0)


def test_a_shared_input_raises_coherence_above_an_independent_one():
    rng = np.random.default_rng(9)
    independent = rng.normal(size=(4096, 2))
    common = rng.normal(size=(4096, 1))
    shared = 0.5 * independent + 0.5 * np.hstack([common, common])
    _, low = pairwise_coherence(independent, SAMPLE_RATE_HZ, segment_seconds=2.0)
    _, high = pairwise_coherence(shared, SAMPLE_RATE_HZ, segment_seconds=2.0)
    assert float(high[:, 0, 1].mean()) > float(low[:, 0, 1].mean())
