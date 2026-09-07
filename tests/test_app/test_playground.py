"""Smoke tests for the Streamlit playground."""

import pytest

pytest.importorskip("streamlit", reason="the playground needs the app extra")

from streamlit.testing.v1 import AppTest  # noqa: E402

from diaxcondel.app import PLAYGROUND_SCRIPT  # noqa: E402
from diaxcondel.experiment import ComponentRef, EventRef, ExperimentSpec  # noqa: E402


def _small_spec() -> ExperimentSpec:
    return ExperimentSpec(seed=2).with_simulation(sample_rate_hz=100, sim_seconds=2.0, burnin_seconds=0.5, n_lags=30)


@pytest.fixture
def app():
    at = AppTest.from_file(str(PLAYGROUND_SCRIPT), default_timeout=300)
    at.session_state["dx_seed_spec"] = _small_spec().to_json()
    return at


def test_playground_renders_every_tab(app):
    app.run()
    assert not app.exception
    assert len(app.tabs) >= 8
    assert app.title[0].value == "Playground"
    labels = [metric.label for metric in app.metric]
    assert {"regions", "delay coverage", "memory for delays", "estimated run"} <= set(labels)

    # Views behind a toggle have to render too.
    for checkbox in app.checkbox:
        if "coherence" in checkbox.label:
            checkbox.set_value(True).run()
    assert not app.exception


def test_every_model_slot_has_its_own_chooser(app):
    app.run()
    keys = {
        "dx.distances.name",
        "dx.connectivity.name",
        "dx.kernel.name",
        "dx.diameter.name",
        "dx.local_delay.name",
        "dx.noise.name",
        "dx.transfer.name",
    }
    for key in keys:
        assert app.selectbox(key=key) is not None


def test_the_calibre_chooser_appears_only_when_the_kernel_uses_it(app):
    app.run()
    assert any(box.label == "how thick the axons are" for box in app.selectbox)
    app.selectbox(key="dx.kernel.name").set_value("fixed_speed").run()
    assert not app.exception
    assert not any(box.label == "how thick the axons are" for box in app.selectbox)
    assert any("calibre" in caption.value for caption in app.caption)


def test_switching_the_distance_source_updates_the_model(app):
    app.run()
    app.selectbox(key="dx.distances.name").set_value("power_law").run()
    # The five-lobe weights do not describe those regions, and say so.
    assert any("five regions" in error.value for error in app.error)
    app.selectbox(key="dx.connectivity.name").set_value("uniform").run()
    assert not app.exception
    regions = next(metric for metric in app.metric if metric.label == "regions")
    assert regions.value == "12"


def test_manual_entry_renders_an_editable_table(app):
    app.run()
    app.selectbox(key="dx.distances.name").set_value("manual").run()
    assert not app.exception
    regions = next(metric for metric in app.metric if metric.label == "regions")
    assert regions.value == "2"


def test_events_can_be_added_from_the_events_tab(app):
    app.run()
    add = next(button for button in app.button if button.label == "Add event")
    add.click().run()
    assert not app.exception
    assert app.session_state["dx_events"]


def test_stationarity_shrinkage_is_surfaced_to_the_user(app):
    app.run()
    assert any("stationarity" in warning.value for warning in app.warning)


def test_event_results_render_for_a_seeded_event(app):
    spec = _small_spec().model_copy(
        update={
            "n_trials": 5,
            "events": (
                EventRef(
                    onset_seconds=0.5,
                    drive=ComponentRef(name="gaussian_pulse", params={"targets": ("OL",), "amplitude": 20.0}),
                    name="pulse",
                ),
            ),
        }
    )
    app.session_state["dx_seed_spec"] = spec.to_json()
    app.run()
    assert not app.exception
    # The ERP table adds a third dataframe (connectome, spectra, ERP peaks).
    assert len(app.dataframe) >= 3


def test_reproduce_tab_offers_the_spec_and_a_script(app):
    app.run()
    labels = [button.label for button in app.download_button]
    assert "Download setup (JSON)" in labels
    assert "Download script (Python)" in labels


def _atlas_app() -> AppTest:
    spec = (
        _small_spec()
        .with_component("distances", "siibra", parcellation="julich 3.1")
        .with_component("connectivity", "siibra", source="streamline_counts")
    )
    app = AppTest.from_file(str(PLAYGROUND_SCRIPT), default_timeout=1500)
    app.session_state["dx_seed_spec"] = spec.to_json()
    app.run()
    return app


@pytest.mark.network
def test_regions_from_different_levels_can_be_picked_together():
    """The thalamus and a lobe sit at different levels; a model may want both."""
    app = _atlas_app()
    # Out of the box the table holds one row: the whole brain, split to lobes.
    assert len(app.multiselect(key="dx.row_names.0").value) == 1
    assert len(app.multiselect(key="dx.include_regions").options) > 10

    app.multiselect(key="dx.row_names.0").set_value(["parietal lobe", "thalamus"]).run()
    app.number_input(key="dx.row_depth.0").set_value(0).run()
    assert not app.exception
    assert set(app.multiselect(key="dx.include_regions").options) == {"parietal lobe", "thalamus"}

    # A second row can split one of them further while the first stays whole.
    app.button(key="dx.row_add").click().run()
    app.multiselect(key="dx.row_names.1").set_value(["occipital lobe"]).run()
    app.number_input(key="dx.row_depth.1").set_value(1).run()
    regions = set(app.multiselect(key="dx.include_regions").options)
    assert {"parietal lobe", "thalamus"} <= regions
    assert "occipital lobe" not in regions


@pytest.mark.network
def test_overlapping_selections_warn_rather_than_fail():
    app = _atlas_app()
    app.multiselect(key="dx.row_names.0").set_value(["cerebral cortex", "occipital lobe"]).run()
    app.number_input(key="dx.row_depth.0").set_value(0).run()

    assert not app.exception  # overlap is legal, so the model still builds
    warnings = [element.value for element in app.warning]
    assert any("occipital lobe" in text and "cerebral cortex" in text for text in warnings)
    # The more specific selection keeps the regions the two share.
    assert "occipital lobe" in set(app.multiselect(key="dx.include_regions").options)


def test_the_operating_point_can_be_set_from_the_page(app):
    app.run()
    app.checkbox(key="dx.use_target_radius").set_value(True).run()
    app.slider(key="dx.target_radius").set_value(0.90).run()
    assert not app.exception
    # The rescaling changes effective coupling, so the page has to say so.
    notes = [element.value for element in app.info] + [element.value for element in app.warning]
    assert any("spectral radius of 0.9" in note for note in notes)
