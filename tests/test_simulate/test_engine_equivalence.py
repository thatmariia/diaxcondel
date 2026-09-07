"""The sliding-window engine must reproduce the defining VAR recursion."""

import numpy as np

from diaxcondel.model.params import SimulationParams
from diaxcondel.model.var import LinearVAR
from diaxcondel.simulate.engine import simulate
from diaxcondel.simulate.noise import white_noise


def _reference(phi, noise, stim):
    """Direct transcription of h(t) = sum_p Phi_p h(t-p) + noise + stim."""
    n_total, n = noise.shape
    p = phi.shape[0]
    out = np.zeros((n_total, n))
    for t in range(n_total):
        acc = noise[t] + stim[t]
        for lag in range(1, p + 1):
            if t - lag >= 0:
                acc = acc + phi[lag - 1] @ out[t - lag]
        out[t] = acc
    return out


def test_engine_matches_the_defining_recursion():
    rng = np.random.default_rng(4)
    n, p = 3, 7
    phi = rng.normal(size=(p, n, n)) * 0.02
    params = SimulationParams(sample_rate_hz=100, sim_seconds=0.5, burnin_seconds=0.1, n_lags=p)
    model = LinearVAR(phi=phi)

    noise_seed = 11
    result = simulate(model, params, white_noise, rng=noise_seed)

    n_total = params.n_burnin_samples + params.n_sim_samples
    noise = white_noise(n, n_total, float(params.sample_rate_hz), np.random.default_rng(noise_seed))
    expected = _reference(phi, noise, np.zeros_like(noise))[params.n_burnin_samples :]
    np.testing.assert_allclose(result.signal, expected, rtol=1e-10, atol=1e-12)


def test_engine_matches_the_recursion_across_the_window_rebase():
    """The history window is re-based every P samples; results must not jump."""
    rng = np.random.default_rng(5)
    n, p = 2, 4
    phi = rng.normal(size=(p, n, n)) * 0.05
    # 30 samples spans several window re-bases for P = 4.
    params = SimulationParams(sample_rate_hz=60, sim_seconds=0.5, burnin_seconds=0.0, n_lags=p)
    model = LinearVAR(phi=phi)
    result = simulate(model, params, white_noise, rng=3)

    noise = white_noise(n, params.n_sim_samples, float(params.sample_rate_hz), np.random.default_rng(3))
    expected = _reference(phi, noise, np.zeros_like(noise))
    np.testing.assert_allclose(result.signal, expected, rtol=1e-10, atol=1e-12)


def test_companion_matrix_is_consistent_with_the_lag_tensor():
    rng = np.random.default_rng(6)
    n, p = 3, 5
    phi = rng.normal(size=(p, n, n)) * 0.1
    model = LinearVAR(phi=phi)
    companion = model.companion_matrix()
    history = rng.normal(size=(p, n))
    state = history.ravel()
    np.testing.assert_allclose((companion @ state)[:n], np.einsum("pij,pj->i", phi, history))
