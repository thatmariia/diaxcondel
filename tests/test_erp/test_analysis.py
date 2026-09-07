"""Tests for ERP analysis helpers."""

import numpy as np

from diaxcondel.erp.analysis import (
    average_trials,
    baseline_correct,
    epoch_around_event,
    peak_amplitude,
    peak_latency,
    sem_trials,
)


def _signal(t=100, n=3, seed=0):
    return np.random.default_rng(seed).standard_normal((t, n))


def _trials(n_trials=5, t=100, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_trials, t, n))


class TestBaselineCorrect:
    def test_mean_is_zero_in_window(self):
        sig = _signal()
        corrected = baseline_correct(sig, (0.0, 0.2), sample_rate_hz=100.0)
        np.testing.assert_allclose(corrected[:20].mean(axis=0), 0.0, atol=1e-12)


class TestAverageTrials:
    def test_matches_numpy(self):
        trials = _trials()
        np.testing.assert_allclose(average_trials(trials), trials.mean(axis=0))


class TestSemTrials:
    def test_single_trial_returns_nan(self):
        trials = _trials(n_trials=1)
        sem = sem_trials(trials)
        assert np.all(np.isnan(sem))

    def test_value_matches_formula(self):
        trials = _trials(n_trials=10)
        expected = trials.std(axis=0, ddof=1) / np.sqrt(10)
        np.testing.assert_allclose(sem_trials(trials), expected)


class TestEpochAroundEvent:
    def test_content(self):
        sig = _signal(t=200)
        epoch = epoch_around_event(sig, event_sample=50, pre_samples=10, post_samples=30)
        np.testing.assert_array_equal(epoch, sig[40:80])


class TestPeakAmplitude:
    def test_value(self):
        sig = np.zeros((100, 2))
        sig[20, 0] = 5.0
        sig[30, 1] = -3.0
        amp = peak_amplitude(sig, (0.0, 1.0), sample_rate_hz=100.0)
        assert np.isclose(amp[0], 5.0)
        assert np.isclose(amp[1], 3.0)


class TestPeakLatency:
    def test_value(self):
        sig = np.zeros((100, 2))
        sig[35, 0] = 10.0  # peak at sample 35 → 0.35 s
        sig[60, 1] = -8.0  # peak at sample 60 → 0.60 s
        lat = peak_latency(sig, (0.0, 1.0), sample_rate_hz=100.0)
        assert np.isclose(lat[0], 0.35)
        assert np.isclose(lat[1], 0.60)
