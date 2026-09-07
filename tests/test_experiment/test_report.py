"""Tests for the standard summaries, including the spectral scale convention."""

import numpy as np
import pytest

from diaxcondel.experiment import (
    ExperimentSpec,
    compute_coherence,
    compute_spectra,
    region_summaries,
    run_experiment,
    spec_to_python,
)


@pytest.fixture(scope="module")
def resting_run():
    spec = ExperimentSpec(seed=3).with_simulation(sample_rate_hz=200, sim_seconds=20.0, burnin_seconds=1.0, n_lags=60)
    return run_experiment(spec)


def test_analytical_and_simulated_spectra_share_a_scale(resting_run):
    spectra = compute_spectra(resting_run, segment_seconds=4.0, fmax_hz=60.0)
    assert spectra.has_analytic
    assert spectra.notes == ()

    # Compare the two curves on a common frequency grid, in log space: the
    # scaling convention in compute_spectra is what makes this hold.
    band = (spectra.freqs_hz >= 2.0) & (spectra.freqs_hz <= 50.0)
    interpolated = np.array(
        [
            np.interp(
                spectra.freqs_hz[band],
                spectra.analytic_freqs_hz,
                spectra.analytic[:, i],
            )
            for i in range(resting_run.built.n_regions)
        ]
    ).T
    ratio = np.log10(spectra.welch[band] / interpolated)
    assert abs(float(np.mean(ratio))) < 0.15, "spectra differ by more than 40% on average"


def test_spectra_are_skipped_for_coloured_noise_and_nonlinear_models():
    base = ExperimentSpec(seed=4).with_simulation(sample_rate_hz=100, sim_seconds=2.0, burnin_seconds=0.5, n_lags=30)
    pink = compute_spectra(run_experiment(base.with_component("noise", "pink")))
    assert not pink.has_analytic
    assert any("flat input spectrum" in note for note in pink.notes)

    nonlinear = compute_spectra(run_experiment(base.with_component("transfer", "centred_sigmoid")))
    assert not nonlinear.has_analytic
    assert any("non-linear" in note for note in nonlinear.notes)


def test_shared_input_keeps_its_closed_form_spectrum():
    # Correlated input is still white, so the closed form applies: only the
    # covariance changes, not the flatness the derivation needs.
    base = ExperimentSpec(seed=4).with_simulation(sample_rate_hz=100, sim_seconds=2.0, burnin_seconds=0.5, n_lags=30)
    shared = compute_spectra(run_experiment(base.with_component("noise", "correlated", correlation=0.6)))
    assert shared.has_analytic
    independent = compute_spectra(run_experiment(base.with_component("noise", "correlated", correlation=0.0)))
    assert independent.analytic is not None and shared.analytic is not None
    # Sharing the input raises power, because the regions now add coherently.
    assert shared.analytic.mean() > independent.analytic.mean()

    sloped = compute_spectra(run_experiment(base.with_component("noise", "correlated", exponent=1.0)))
    assert not sloped.has_analytic


def test_region_summaries_cover_every_region(resting_run):
    spectra = compute_spectra(resting_run, segment_seconds=4.0)
    summaries = region_summaries(resting_run, spectra)
    assert [item.code for item in summaries] == list(resting_run.regions.codes)
    for item in summaries:
        assert 8.0 <= item.peak_hz <= 13.0
        assert item.variance > 0
        assert np.isfinite(item.slope)


def test_occipital_alpha_dominates_in_the_reference_model(resting_run):
    spectra = compute_spectra(resting_run, segment_seconds=4.0)
    summaries = {item.code: item for item in region_summaries(resting_run, spectra)}
    cortical = ["FL", "PL", "OL", "TL"]
    assert max(cortical, key=lambda code: summaries[code].peak_power) == "OL"


def test_coherence_shape_and_diagonal(resting_run):
    freqs, coherence = compute_coherence(resting_run, segment_seconds=2.0)
    n = resting_run.built.n_regions
    assert coherence.shape == (freqs.size, n, n)
    np.testing.assert_allclose(np.diagonal(coherence, axis1=1, axis2=2), 1.0)


def test_multi_trial_spectra_average_over_trials():
    spec = ExperimentSpec(n_trials=3, seed=5).with_simulation(
        sample_rate_hz=100, sim_seconds=2.0, burnin_seconds=0.5, n_lags=30
    )
    spectra = compute_spectra(run_experiment(spec), segment_seconds=1.0)
    assert np.all(np.isfinite(spectra.welch))
    assert spectra.welch.shape[1] == 5


def test_generated_script_is_valid_python_and_carries_the_spec():
    spec = ExperimentSpec(label="snippet").with_component("distances", "power_law", n_regions=4)
    spec = spec.with_component("connectivity", "uniform")
    code = spec_to_python(spec)
    compile(code, "<generated>", "exec")
    assert "power_law" in code
    assert "run_experiment" in code
