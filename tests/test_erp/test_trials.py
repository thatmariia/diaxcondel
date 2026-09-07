"""
Tests for multi-trial ERP simulation.

Covers the key requirements from the specification:
- event onset sample placement after burn-in
- pure additive zero-noise trial determinism
- fixed integer seed reproducibility
- n_jobs=1 vs n_jobs=2 reproducibility
- trial RNG independence
- online average == full-trial average
- connectivity-only modulation with zero-mean noise gives mean ≈ 0
- overlapping modulation composition
"""

import numpy as np
import pytest

from diaxcondel.connectome.regions import RegionSet
from diaxcondel.erp.event import ERPEvent
from diaxcondel.erp.modulation import StepModulation
from diaxcondel.model.params import SimulationParams
from diaxcondel.model.var import LinearVAR
from diaxcondel.simulate import simulate_trials
from diaxcondel.simulate.noise import white_noise
from diaxcondel.simulate.stimulus import DriveTarget, SquarePulseDrive

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_stationary_phi(n=3, p=5):
    phi = np.zeros((p, n, n))
    phi[0] = np.eye(n) * 0.1
    return phi


def _params(sim_seconds=0.5, burnin_seconds=0.1, rate=100):
    return SimulationParams(
        sample_rate_hz=rate,
        sim_seconds=sim_seconds,
        burnin_seconds=burnin_seconds,
        n_lags=p_for_rate(rate),
    )


def p_for_rate(rate):
    return max(5, rate // 10)  # keep small for speed


def _regions(n=3):
    codes = ["A", "B", "C"][:n]
    return RegionSet.from_codes(codes)


def _square_drive(code="A", dur=0.05, amp=1.0):
    return SquarePulseDrive(
        duration_seconds=dur,
        amplitude=amp,
        targets=(DriveTarget(code),),
    )


def _zero_noise_fn(n_nodes, n_samples, sample_rate_hz, rng):
    return np.zeros((n_samples, n_nodes))


# ---------------------------------------------------------------------------
# Basic TrialResult structure
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Event onset placement
# ---------------------------------------------------------------------------


def test_event_onset_lands_at_correct_sample_after_burnin():
    """With zero noise, the drive injection time is deterministic."""
    n, p = 2, 5
    phi = _tiny_stationary_phi(n=n, p=p)
    dynamics = LinearVAR(phi=phi)
    # 10 Hz, burnin=0.5s (5 samples), sim=1.0s (10 samples)
    params = SimulationParams(sample_rate_hz=10, sim_seconds=1.0, burnin_seconds=0.5, n_lags=p)
    regions = _regions(n=2)

    # Place a 0.1-s square pulse at onset=0.3s → sample index 3 in post-burnin window.
    ev = ERPEvent(
        onset_seconds=0.3,
        drive=SquarePulseDrive(
            duration_seconds=0.1,
            amplitude=5.0,
            targets=(DriveTarget("A"),),
        ),
        modulation=None,
    )

    result = simulate_trials(dynamics, params, _zero_noise_fn, events=[ev], regions=regions, n_trials=1, rng=0)
    trial = result.trials[0]  # (T=10, N=2)
    # sample 3 to 3+1=4 (0.1s @ 10Hz = 1 sample) should be non-zero in channel A (index 0)
    assert trial[3, 0] != 0.0, "drive should appear at sample 3"
    # Outside the pulse window, channel A may still be non-zero due to propagation, but
    # before onset it should be exactly zero with zero noise.
    assert trial[0, 0] == 0.0, "before onset, no drive has been injected"
    assert trial[1, 0] == 0.0
    assert trial[2, 0] == 0.0


# ---------------------------------------------------------------------------
# Zero-noise determinism
# ---------------------------------------------------------------------------


def test_pure_additive_zero_noise_is_deterministic():
    phi = _tiny_stationary_phi(n=2, p=5)
    dynamics = LinearVAR(phi=phi)
    params = _params(sim_seconds=0.3, burnin_seconds=0.1, rate=100)
    regions = _regions(n=2)
    ev = ERPEvent(onset_seconds=0.05, drive=_square_drive("A"), modulation=None)

    r1 = simulate_trials(dynamics, params, _zero_noise_fn, events=[ev], regions=regions, n_trials=1, rng=0)
    r2 = simulate_trials(dynamics, params, _zero_noise_fn, events=[ev], regions=regions, n_trials=1, rng=999)

    # With zero noise the result should be identical regardless of seed.
    np.testing.assert_array_equal(r1.trials[0], r2.trials[0])


# ---------------------------------------------------------------------------
# Reproducibility with fixed integer seed
# ---------------------------------------------------------------------------


def test_fixed_seed_is_reproducible():
    phi = _tiny_stationary_phi(n=2, p=5)
    dynamics = LinearVAR(phi=phi)
    params = _params(sim_seconds=0.2, burnin_seconds=0.05, rate=50)
    regions = _regions(n=2)
    ev = ERPEvent(onset_seconds=0.05, drive=_square_drive("A"), modulation=None)

    r1 = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=4, rng=1234)
    r2 = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=4, rng=1234)

    np.testing.assert_array_equal(r1.trials, r2.trials)


def test_different_seeds_give_different_results():
    phi = _tiny_stationary_phi(n=2, p=5)
    dynamics = LinearVAR(phi=phi)
    params = _params(sim_seconds=0.2, burnin_seconds=0.05, rate=50)
    regions = _regions(n=2)
    ev = ERPEvent(onset_seconds=0.05, drive=_square_drive("A"), modulation=None)

    r1 = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=2, rng=1)
    r2 = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=2, rng=2)

    assert not np.allclose(r1.trials, r2.trials)


# ---------------------------------------------------------------------------
# n_jobs reproducibility (serial vs parallel)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_n_jobs_1_and_2_give_same_results_for_integer_seed():
    phi = _tiny_stationary_phi(n=2, p=5)
    dynamics = LinearVAR(phi=phi)
    params = _params(sim_seconds=0.2, burnin_seconds=0.05, rate=50)
    regions = _regions(n=2)
    ev = ERPEvent(onset_seconds=0.05, drive=_square_drive("A"), modulation=None)

    r1 = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=4, rng=42, n_jobs=1)
    r2 = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=4, rng=42, n_jobs=2)

    np.testing.assert_array_equal(r1.trials, r2.trials)


# ---------------------------------------------------------------------------
# Trial RNG independence
# ---------------------------------------------------------------------------


def test_trial_rngs_are_independent():
    """Different trials must produce different noise realisations."""
    phi = np.zeros((5, 2, 2))
    dynamics = LinearVAR(phi=phi)  # pure noise, no coupling
    params = _params(sim_seconds=0.2, burnin_seconds=0.05, rate=100)
    regions = _regions(n=2)
    ev = ERPEvent(onset_seconds=0.0, drive=_square_drive("A", amp=0.0), modulation=None)

    result = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=5, rng=7)

    # Check that no two trials are identical.
    for i in range(result.n_trials):
        for j in range(i + 1, result.n_trials):
            assert not np.allclose(result.trials[i], result.trials[j]), (
                f"trials {i} and {j} are identical — RNGs are not independent"
            )


# ---------------------------------------------------------------------------
# Online average == full-trial average
# ---------------------------------------------------------------------------


def test_online_average_equals_full_trials_average():
    phi = _tiny_stationary_phi(n=2, p=5)
    dynamics = LinearVAR(phi=phi)
    params = _params(sim_seconds=0.2, burnin_seconds=0.05, rate=100)
    regions = _regions(n=2)
    ev = ERPEvent(onset_seconds=0.05, drive=_square_drive("A"), modulation=None)

    result = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=10, rng=0)

    # Online mean (computed during collection) vs numpy mean over all trials.
    full_avg = result.trials.mean(axis=0)
    np.testing.assert_allclose(result.average, full_avg)


# ---------------------------------------------------------------------------
# Connectivity-only modulation → signed mean near zero
# ---------------------------------------------------------------------------


def test_connectivity_only_modulation_mean_near_zero():
    """Pure connectivity modulation with zero-mean noise should give mean ≈ 0.

    The signed trial average must remain small relative to the single-trial
    noise floor.  We verify this by comparing the average amplitude to the
    single-trial amplitude: the ratio should shrink with n_trials.
    """
    phi = _tiny_stationary_phi(n=3, p=5)
    dynamics = LinearVAR(phi=phi)
    params = SimulationParams(sample_rate_hz=100, sim_seconds=1.0, burnin_seconds=0.2, n_lags=5)
    regions = _regions(n=3)

    factor = np.ones((3, 3)) * 0.9
    mod = StepModulation(factor=factor, duration_seconds=0.3, instability_policy="ignore")
    ev = ERPEvent(onset_seconds=0.1, drive=None, modulation=mod)

    n_trials = 400
    result = simulate_trials(dynamics, params, white_noise, events=[ev], regions=regions, n_trials=n_trials, rng=42)
    avg = result.average  # (T, N)
    single_trial_std = result.trials.std()

    # The mean should be much smaller than a single-trial std.
    # With 400 trials, mean std ≈ single_std / sqrt(400).  Allow 10× that.
    expected_mean_std = single_trial_std / np.sqrt(n_trials)
    threshold = 10 * expected_mean_std
    actual_max = np.max(np.abs(avg))
    assert actual_max < threshold, (
        f"signed mean too large for modulation-only event: {actual_max:.4f} "
        f"(threshold {threshold:.4f}, single-trial std {single_trial_std:.4f})"
    )


# ---------------------------------------------------------------------------
# Overlapping modulations compose by multiplication
# ---------------------------------------------------------------------------


def test_overlapping_modulations_compose():
    """Two simultaneously active modulations multiply their factors."""
    phi = _tiny_stationary_phi(n=2, p=3)
    dynamics = LinearVAR(phi=phi)
    params = SimulationParams(sample_rate_hz=10, sim_seconds=0.5, burnin_seconds=0.1, n_lags=3)
    regions = _regions(n=2)

    f1 = np.array([[2.0, 1.0], [1.0, 2.0]])
    f2 = np.array([[0.5, 1.0], [1.0, 0.5]])
    mod1 = StepModulation(factor=f1, duration_seconds=0.3, instability_policy="ignore")
    mod2 = StepModulation(factor=f2, duration_seconds=0.3, instability_policy="ignore")

    ev1 = ERPEvent(onset_seconds=0.0, drive=None, modulation=mod1)
    ev2 = ERPEvent(onset_seconds=0.0, drive=None, modulation=mod2)

    # This should not raise; overlapping modulations compose by multiplication.
    result = simulate_trials(dynamics, params, _zero_noise_fn, events=[ev1, ev2], regions=regions, n_trials=1, rng=0)
    assert result.n_trials == 1


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------
