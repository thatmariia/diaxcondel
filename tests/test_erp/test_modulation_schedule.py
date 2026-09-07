"""The connectivity-modulation schedule: correctness and cost."""

import time

import numpy as np

from diaxcondel.erp.modulation import StepModulation, build_phi_schedule


def _phi(p: int = 20, n: int = 3) -> np.ndarray:
    return np.full((p, n, n), 0.01)


def test_the_schedule_is_piecewise_constant_over_a_window():
    phi = _phi()
    mod = StepModulation(factor=np.full((3, 3), 2.0), duration_seconds=0.1, instability_policy="ignore")
    schedule = build_phi_schedule(phi, [(0.05, mod)], 10, 200, 100.0)

    # Onset is 0.05 s after the burn-in ends, and it lasts 0.1 s: samples 15-24.
    modulated = [index for index, tensor in enumerate(schedule) if tensor is not phi]
    assert modulated == list(range(15, 25))
    # Every sample in the window shares one array, so the simulation engine can
    # reuse its factorised form and the memory cost is one tensor, not one per
    # sample.
    assert len({id(tensor) for tensor in schedule}) == 2
    assert np.allclose(schedule[15], phi * 2.0)


def test_overlapping_modulations_compose_where_they_overlap():
    phi = _phi()
    first = StepModulation(factor=np.full((3, 3), 2.0), duration_seconds=0.1, instability_policy="ignore")
    second = StepModulation(factor=np.full((3, 3), 3.0), duration_seconds=0.1, instability_policy="ignore")
    schedule = build_phi_schedule(phi, [(0.05, first), (0.10, second)], 10, 200, 100.0)

    values = np.array([tensor[0, 0, 1] for tensor in schedule]) / phi[0, 0, 1]
    assert sorted(set(np.round(values, 6))) == [1.0, 2.0, 3.0, 6.0]
    assert len({id(tensor) for tensor in schedule}) == 4  # base plus three stretches


def test_a_long_window_does_not_cost_a_tensor_per_sample():
    # A second of modulation at 1 kHz is a thousand samples; the window costs
    # one stability check and one tensor however many samples it spans.
    phi = np.zeros((120, 8, 8))
    mod = StepModulation(factor=np.full((8, 8), 1.1), duration_seconds=1.0)
    start = time.perf_counter()
    schedule = build_phi_schedule(phi, [(0.1, mod)], 500, 2000, 1000.0)
    assert time.perf_counter() - start < 10.0
    assert len({id(tensor) for tensor in schedule}) == 2


def test_windows_outside_the_trial_are_ignored():
    phi = _phi()
    mod = StepModulation(factor=np.full((3, 3), 2.0), duration_seconds=0.1, instability_policy="ignore")
    schedule = build_phi_schedule(phi, [(99.0, mod)], 10, 50, 100.0)
    assert all(tensor is phi for tensor in schedule)
