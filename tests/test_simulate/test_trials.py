"""Tests for TrialResult and simulate_trials() in fixed-input (no-events) mode."""

import numpy as np

from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.params import SimulationParams
from diaxcondel.model.var import LinearVAR
from diaxcondel.simulate import simulate_trials
from diaxcondel.simulate.noise import white_noise
from diaxcondel.simulate.trials import TrialResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phi(n=2, p=3):
    phi = np.zeros((p, n, n))
    phi[0] = np.eye(n) * 0.1
    return phi


def _params(sim_seconds=0.2, burnin_seconds=0.05, rate=50, n_lags=3):
    return SimulationParams(
        sample_rate_hz=rate,
        sim_seconds=sim_seconds,
        burnin_seconds=burnin_seconds,
        n_lags=n_lags,
    )


def _zero_noise(n_nodes, n_samples, sample_rate_hz, rng):
    return np.zeros((n_samples, n_nodes))


# ---------------------------------------------------------------------------
# TrialResult construction
# ---------------------------------------------------------------------------


class TestTrialResult:
    def test_average_is_the_mean_over_trials(self):
        trials = np.random.default_rng(0).standard_normal((5, 8, 3))
        result = TrialResult(trials=trials, params=_params())
        np.testing.assert_allclose(result.average, trials.mean(axis=0))

    def test_sem_single_trial_is_nan(self):
        result = TrialResult(trials=np.ones((1, 8, 2)), params=_params())
        assert np.all(np.isnan(result.sem))


# ---------------------------------------------------------------------------
# simulate_trials — fixed-input mode (no events)
# ---------------------------------------------------------------------------


class TestSimulateTrialsFixedInput:
    def test_reproducible_with_integer_seed(self):
        dynamics = LinearVAR(phi=_phi())
        params = _params()
        r1 = simulate_trials(dynamics, params, white_noise, n_trials=3, rng=42)
        r2 = simulate_trials(dynamics, params, white_noise, n_trials=3, rng=42)
        np.testing.assert_array_equal(r1.trials, r2.trials)

    def test_different_seeds_differ(self):
        dynamics = LinearVAR(phi=_phi())
        params = _params()
        r1 = simulate_trials(dynamics, params, white_noise, n_trials=2, rng=1)
        r2 = simulate_trials(dynamics, params, white_noise, n_trials=2, rng=2)
        assert not np.allclose(r1.trials, r2.trials)

    def test_trials_are_independent(self):
        """Each trial must have a different noise realisation."""
        dynamics = LinearVAR(phi=np.zeros((3, 2, 2)))
        params = _params()
        result = simulate_trials(dynamics, params, white_noise, n_trials=4, rng=7)
        for i in range(result.n_trials):
            for j in range(i + 1, result.n_trials):
                assert not np.allclose(result.trials[i], result.trials[j])

    def test_single_trial_equals_simulate(self):
        """Fixed-input mode with n_trials=1 should match engine.simulate()."""
        from diaxcondel.simulate.engine import simulate

        dynamics = LinearVAR(phi=_phi())
        params = _params()
        # Manually replicate the seed-derivation path used by simulate_trials.
        # Two fresh sequences from the same entropy: one for simulate_trials, one
        # to compute the expected noise generator independently.
        (child,) = np.random.SeedSequence(99).spawn(1)
        (noise_seed,) = child.spawn(1)

        result_trials = simulate_trials(dynamics, params, white_noise, n_trials=1, rng=99)
        result_single = simulate(dynamics, params, white_noise, rng=np.random.default_rng(noise_seed))
        np.testing.assert_array_equal(result_trials.trials[0], result_single.signal)

    def test_zero_noise_with_stim_is_deterministic(self):
        """Same stim across trials produces identical signals with zero noise."""
        from diaxcondel.simulate.stimulus import DriveTarget, SquarePulseDrive, build_drive_array

        dynamics = LinearVAR(phi=_phi())
        params = _params()
        regions = RegionSet.from_codes(["A", "B"])
        drive = SquarePulseDrive(duration_seconds=0.05, amplitude=1.0, targets=(DriveTarget("A"),))
        stim = build_drive_array([(0.0, drive)], regions, params)

        result = simulate_trials(dynamics, params, _zero_noise, n_trials=3, stim=stim, regions=regions, rng=0)
        # With zero noise, all trials must be identical.
        np.testing.assert_array_equal(result.trials[0], result.trials[1])
        np.testing.assert_array_equal(result.trials[0], result.trials[2])
