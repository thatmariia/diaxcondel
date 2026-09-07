"""Tests for connectivity modulation."""

import warnings

import numpy as np

from diaxcondel.erp.modulation import StepModulation, compose_modulations


def _make_phi(p=5, n=3):
    """Tiny stationary Phi for testing."""
    phi = np.zeros((p, n, n))
    phi[0] = np.eye(n) * 0.1
    return phi


class TestStepModulation:
    def test_apply_scales_phi(self):
        phi = _make_phi(p=2, n=2)
        factor = np.array([[2.0, 1.0], [1.0, 0.5]])
        mod = StepModulation(factor=factor, duration_seconds=0.1, instability_policy="ignore")
        phi_mod = mod.apply(phi)
        assert phi_mod.shape == phi.shape
        np.testing.assert_allclose(phi_mod, phi * factor[np.newaxis, :, :])

    def test_instability_warns_by_default(self):
        # A very large factor should make the system non-stationary.
        phi = _make_phi(p=2, n=2)
        # Use a factor that guarantees instability.
        big_factor = np.full((2, 2), 100.0)
        mod = StepModulation(factor=big_factor, duration_seconds=0.1)  # default "warn"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod.apply(phi)
        assert any("unstable" in str(warning.message).lower() for warning in w)

    def test_stable_modulation_no_warning(self):
        phi = _make_phi(p=2, n=2)
        factor = np.eye(2) * 0.5  # shrinks coupling → more stable
        mod = StepModulation(factor=factor, duration_seconds=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod.apply(phi)
        assert not any("unstable" in str(warning.message).lower() for warning in w)


class TestComposeModulations:
    def test_empty_list_returns_base(self):
        phi = _make_phi()
        result = compose_modulations([], phi)
        assert result is phi

    def test_single_modulation_equals_apply(self):
        phi = _make_phi(p=2, n=2)
        factor = np.array([[1.5, 1.0], [1.0, 1.5]])
        mod = StepModulation(factor=factor, duration_seconds=0.1, instability_policy="ignore")
        composed = compose_modulations([mod], phi)
        direct = mod.apply(phi)
        np.testing.assert_allclose(composed, direct)

    def test_two_modulations_multiply(self):
        phi = _make_phi(p=2, n=2)
        f1 = np.array([[2.0, 1.0], [1.0, 2.0]])
        f2 = np.array([[0.5, 1.0], [1.0, 0.5]])
        m1 = StepModulation(factor=f1, duration_seconds=0.1, instability_policy="ignore")
        m2 = StepModulation(factor=f2, duration_seconds=0.1, instability_policy="ignore")
        composed = compose_modulations([m1, m2], phi)
        expected = phi * (f1 * f2)[np.newaxis, :, :]
        np.testing.assert_allclose(composed, expected)
