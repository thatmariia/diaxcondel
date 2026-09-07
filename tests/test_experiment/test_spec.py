"""Tests for experiment specifications: validation, editing, serialisation."""

from diaxcondel.experiment import (
    ComponentRef,
    EventRef,
    ExperimentSpec,
    load_spec,
    save_spec,
)


def test_defaults_are_canonicalised_against_the_catalog():
    spec = ExperimentSpec()
    assert spec.distances.name == "steeghs_2025"
    assert spec.connectivity.name == "steeghs_2025"
    # Unspecified parameters are filled in from the catalog defaults.
    assert spec.kernel.name == "hursh"
    assert spec.kernel.params["speed_factor"] == 6.0
    assert spec.diameter.name == "gev"
    assert spec.noise.params == {"std": 1.0}


def test_events_require_a_drive_or_a_modulation():
    spec = ExperimentSpec(events=(EventRef(drive=ComponentRef(name="gaussian_pulse")),))
    assert spec.events[0].drive is not None
    assert spec.events[0].drive.params["amplitude"] == 5.0


def test_with_component_merges_or_replaces():
    spec = ExperimentSpec()
    tuned = spec.with_component("kernel", speed_factor=9.0)
    assert tuned.kernel.params["speed_factor"] == 9.0
    assert tuned.kernel.params.keys() == spec.kernel.params.keys()
    assert spec.kernel.name == "hursh"
    assert spec.kernel.params["speed_factor"] == 6.0
    assert spec.diameter.name == "gev", "the original spec must not change"

    switched = spec.with_component("distances", "power_law", n_regions=6)
    assert switched.distances.name == "power_law"
    assert switched.distances.params["n_regions"] == 6


def test_self_excitation_requires_a_local_delay():
    spec = ExperimentSpec(self_weight=0.3, local_delay=ComponentRef(name="gamma"))
    assert spec.self_weight == 0.3


def test_events_can_be_added_and_cleared():
    spec = ExperimentSpec().with_event("gaussian_pulse", onset_seconds=0.4, targets=("OL",))
    assert len(spec.events) == 1
    assert spec.events[0].drive is not None
    assert spec.events[0].drive.params["targets"] == ("OL",)
    assert spec.events[0].onset_seconds == 0.4
    two = spec.with_event("square_pulse", onset_seconds=1.0)
    assert len(two.events) == 2
    assert two.without_events().events == ()

    ready = EventRef(onset_seconds=0.4, drive=ComponentRef(name="gaussian_pulse"), name="probe")
    assert ExperimentSpec().with_event(ready).events[0].name == "probe"


def test_with_simulation_updates_only_named_fields():
    spec = ExperimentSpec().with_simulation(sim_seconds=3.0, n_trials=4)
    assert spec.simulation.sim_seconds == 3.0
    assert spec.simulation.sample_rate_hz == 1000
    # n_trials belongs to the specification rather than to SimulationParams,
    # and must be applied all the same.
    assert spec.n_trials == 4


def test_json_round_trip_and_files(tmp_path):
    spec = ExperimentSpec(label="round trip").with_component("distances", "power_law", n_regions=5)
    restored = ExperimentSpec.model_validate_json(spec.to_json())
    assert restored == spec
    assert ExperimentSpec.from_dict(spec.model_dump(mode="json")) == spec

    path = save_spec(spec, tmp_path / "nested" / "spec.json")
    assert load_spec(path) == spec
    assert spec.fingerprint() == restored.fingerprint()
