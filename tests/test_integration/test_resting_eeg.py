"""Reproduce qualitative features of Steeghs-Turchina et al. (2025)."""

import numpy as np

from diaxcondel.model.build import build_linear_var
from diaxcondel.simulate.engine import simulate
from diaxcondel.simulate.noise import get_noise_fn
from diaxcondel.spectral.analytical import analytical_psd
from diaxcondel.spectral.empirical import welch_psd


def test_alpha_peak_in_occipital(distances, weights, kernel, sim_params, rng):
    model = build_linear_var(distances, weights, kernel, sim_params)
    result = simulate(model, sim_params, get_noise_fn("white"), rng=rng)
    freqs, psd = welch_psd(result.signal, sim_params.sample_rate_hz)
    assert freqs.ndim == 1
    assert psd.shape[1] == 5
    assert np.all(np.isfinite(psd))

    freqs = np.linspace(1, 50, 200)
    psd = analytical_psd(model, freqs, sim_params.dt_s)
    # Region 2 is OL.
    alpha_band = (freqs >= 8) & (freqs <= 13)
    control_band = (freqs >= 14) & (freqs <= 20)
    assert psd[alpha_band, 2, 2].mean() > psd[control_band, 2, 2].mean()
