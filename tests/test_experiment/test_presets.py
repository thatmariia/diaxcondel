"""Every preset must validate, and the offline ones must build and run."""

import pytest

from diaxcondel.experiment import build_model, run_experiment
from diaxcondel.experiment.presets import (
    PRESETS,
    get_preset,
    preset_names,
    preset_summary,
)

ATLAS = {"atlas_lobes", "atlas_cortical_gyri"}
OFFLINE = [name for name in preset_names() if name not in ATLAS]


def test_all_presets_validate_and_describe_themselves():
    assert set(preset_names()) == set(PRESETS)
    for name in preset_names():
        spec = get_preset(name)
        assert spec.label
        assert preset_summary(name)
        # Re-validating a serialised preset must be a no-op.
        assert type(spec).model_validate_json(spec.to_json()) == spec


@pytest.mark.parametrize("name", OFFLINE)
def test_offline_presets_run(name):
    spec = get_preset(name).with_simulation(sample_rate_hz=100, sim_seconds=1.0, burnin_seconds=0.5, n_lags=30)
    if spec.distances.name in {"power_law", "power_law_with_hub"}:
        spec = spec.with_component("distances", spec.distances.name, n_regions=5)
    result = run_experiment(spec)
    assert result.signal.shape[0] == 100
    assert result.built.n_regions >= 5


@pytest.mark.parametrize("name", sorted(ATLAS))
def test_atlas_presets_request_the_atlas_entries(name):
    spec = get_preset(name)
    assert spec.distances.name == "siibra"
    assert spec.connectivity.name == "siibra"
    assert spec.distances.params["merge_level"] in (3, 4)
    # It must be describable without touching the network.
    assert "siibra" in spec.to_json()


def test_thalamic_relay_preset_builds_with_a_two_leg_kernel():
    from diaxcondel.kernels.diameter import TwoLegDelayKernel

    spec = get_preset("thalamic_relay").with_simulation(
        sample_rate_hz=100, sim_seconds=1.0, burnin_seconds=0.5, n_lags=30
    )
    built = build_model(spec)
    assert isinstance(built.kernel, TwoLegDelayKernel)
