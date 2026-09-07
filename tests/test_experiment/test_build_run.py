"""Tests for building and running specifications."""

import numpy as np
import pytest

from diaxcondel.experiment import (
    ComponentRef,
    EventRef,
    ExperimentSpec,
    build_model,
    estimate_cost,
    run_experiment,
)
from diaxcondel.kernels.diameter import TwoLegDelayKernel
from diaxcondel.model.var import LinearVAR, NonlinearVAR


@pytest.fixture
def small_spec():
    return ExperimentSpec(
        distances=ComponentRef(name="steeghs_2025"),
        connectivity=ComponentRef(name="steeghs_2025"),
        seed=7,
    ).with_simulation(sample_rate_hz=200, sim_seconds=1.0, burnin_seconds=0.5, n_lags=60)


def test_build_model_reports_shrinkage_as_a_note(small_spec):
    built = build_model(small_spec)
    assert isinstance(built.dynamics, LinearVAR)
    assert built.n_regions == 5
    assert built.phi.shape == (60, 5, 5)
    assert any("stationarity" in note for note in built.notes)
    assert built.is_linear


def test_relay_selects_the_two_leg_kernel_automatically(small_spec):
    relayed = small_spec.model_copy(update={"relay_code": "T"})
    built = build_model(relayed)
    assert isinstance(built.kernel, TwoLegDelayKernel)
    # Relayed distances sum both legs, so cortico-cortical delays grow.
    direct = build_model(small_spec)
    assert built.connectome.distance_matrix()[0, 1] > direct.connectome.distance_matrix()[0, 1]


def test_nonlinear_transfer_gives_a_nonlinear_model(small_spec):
    spec = small_spec.with_component("transfer", "centred_sigmoid", slope=4.0)
    built = build_model(spec)
    assert isinstance(built.dynamics, NonlinearVAR)
    assert not built.is_linear


def test_disabling_stationarity_is_flagged(small_spec):
    built = build_model(small_spec.model_copy(update={"enforce_stationarity": False}))
    assert any("diverges" in note for note in built.notes)


def test_run_is_reproducible_and_seed_dependent(small_spec):
    first = run_experiment(small_spec)
    second = run_experiment(small_spec)
    np.testing.assert_array_equal(first.signal, second.signal)

    other = run_experiment(small_spec.model_copy(update={"seed": 8}))
    assert not np.allclose(first.signal, other.signal)

    assert first.signal.shape == (200, 5)
    assert first.regions.codes == ("FL", "PL", "OL", "TL", "T")
    assert first.time_s[0] == 0.0


def test_multi_trial_run_and_event_drive(small_spec):
    spec = small_spec.model_copy(
        update={
            "n_trials": 4,
            "events": (
                EventRef(
                    onset_seconds=0.2,
                    drive=ComponentRef(
                        name="gaussian_pulse",
                        params={"targets": ("OL",), "amplitude": 50.0},
                    ),
                    name="pulse",
                ),
            ),
        }
    )
    result = run_experiment(spec)
    assert result.trials.trials.shape == (4, 200, 5)
    average = result.average
    onset_index = int(0.2 * spec.simulation.sample_rate_hz)
    occipital = result.regions.index("OL")
    pre = np.abs(average[:onset_index, occipital]).mean()
    post = np.abs(average[onset_index : onset_index + 40, occipital]).mean()
    assert post > pre, "the drive should raise the trial-averaged response after onset"


def test_reusing_a_built_model_skips_rebuilding(small_spec):
    built = build_model(small_spec)
    result = run_experiment(small_spec, built=built)
    assert result.built is built
    assert result.all_notes[: len(built.notes)] == built.notes


def test_local_delay_and_self_excitation_change_the_lag_tensor(small_spec):
    plain = build_model(small_spec)
    delayed = build_model(small_spec.with_component("local_delay", "gamma", mean_ms=6.0))
    lags_ms = 1000 * small_spec.simulation.dt_s * np.arange(1, small_spec.simulation.n_lags + 1)

    def peak(phi, i, j):
        return lags_ms[int(np.argmax(phi[:, i, j]))]

    assert peak(delayed.phi, 2, 1) > peak(plain.phi, 2, 1), "a local delay must postpone arrivals"
    assert np.allclose(np.diagonal(plain.phi, axis1=1, axis2=2), 0.0)

    recurrent = build_model(
        small_spec.with_component("local_delay", "gamma", mean_ms=3.0).model_copy(update={"self_weight": 0.3})
    )
    assert np.diagonal(recurrent.phi, axis1=1, axis2=2).sum() > 0


def test_spectral_radius_is_reported(small_spec):
    built = build_model(small_spec)
    radius = built.spectral_radius()
    assert 0.0 < radius < 1.05
    weaker = build_model(small_spec.with_component("connectivity", "steeghs_2025", scale=0.2))
    assert weaker.spectral_radius() < radius


def test_cost_estimate_scales_with_size(small_spec):
    small = estimate_cost(5, small_spec.simulation)
    bigger = estimate_cost(50, small_spec.simulation)
    assert bigger.phi_bytes == 100 * small.phi_bytes
    assert bigger.simulate_seconds > small.simulate_seconds
    assert not small.is_heavy

    heavy = estimate_cost(400, small_spec.with_simulation(sim_seconds=60.0).simulation, n_trials=20)
    assert heavy.is_heavy
    assert heavy.notes
    assert estimate_cost(5, small_spec.simulation, enforce_stationarity=False).build_seconds == 0.0
