"""The playground draws interactively when it can, and statically otherwise."""

import numpy as np
import pytest

pytest.importorskip("streamlit", reason="the playground needs the app extra")

from streamlit.testing.v1 import AppTest  # noqa: E402

from diaxcondel.app import PLAYGROUND_SCRIPT  # noqa: E402
from diaxcondel.app._charts import INTERACTIVE  # noqa: E402
from diaxcondel.experiment import ExperimentSpec  # noqa: E402


def _spec() -> ExperimentSpec:
    return (
        ExperimentSpec(seed=2)
        .with_simulation(sample_rate_hz=200, sim_seconds=3.0, burnin_seconds=1.0, n_lags=50, n_trials=4)
        .with_event("gaussian_pulse", onset_seconds=0.4)
    )


def _run_app() -> AppTest:
    app = AppTest.from_file(str(PLAYGROUND_SCRIPT), default_timeout=600)
    app.session_state["dx_seed_spec"] = _spec().to_json()
    app.run()
    return app


@pytest.mark.skipif(not INTERACTIVE, reason="plotly is not installed")
def test_figures_are_interactive_when_plotly_is_available():
    app = _run_app()
    assert len(app.get("plotly_chart")) >= 5


def test_the_static_backend_still_works_without_plotly(monkeypatch):
    """The app extra can be installed without plotly; the page must still draw."""
    from diaxcondel.app import _charts

    calls: list[str] = []
    monkeypatch.setattr(_charts, "INTERACTIVE", False)
    monkeypatch.setattr(_charts.st, "pyplot", lambda *a, **k: calls.append("static"))
    _charts.show(lambda: pytest.fail("interactive backend used"), lambda: object())
    assert calls == ["static"]


@pytest.mark.skipif(not INTERACTIVE, reason="plotly is not installed")
def test_interactive_figures_carry_region_names_for_hover():
    from diaxcondel.app import _interactive as live
    from diaxcondel.experiment import compute_spectra, run_experiment

    run = run_experiment(_spec())
    spectra = compute_spectra(run)

    figure = live.spectra_figure(spectra, run.regions)
    named = {trace.name for trace in figure.data}
    assert set(run.regions.codes) <= named
    assert all("Hz" in trace.hovertemplate for trace in figure.data)

    lengths = run.built.connectome.distance_matrix()
    matrix = live.matrix_figure(lengths, run.regions, title="Length", units="cm")
    assert list(matrix.data[0].x) == list(run.regions.codes)
    # Pairs that are connected keep their length; pairs that are not are left
    # blank rather than being drawn at one end of the colour scale.
    connected = lengths > 0
    drawn = np.asarray(matrix.data[0].z, dtype=float)
    assert np.allclose(drawn[connected], lengths[connected])
    assert np.all(np.isnan(drawn[~connected]))
