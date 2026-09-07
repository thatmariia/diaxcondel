"""Tests for the command-line interface."""

import json

import numpy as np

from diaxcondel.cli import main
from diaxcondel.experiment import ExperimentSpec, load_spec, save_spec


def test_catalog_json_lists_slots_and_parameters(capsys):
    assert main(["catalog", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {
        "distances",
        "connectivity",
        "diameter",
        "kernel",
        "local_delay",
        "noise",
        "transfer",
        "drive",
        "modulation",
    } == set(payload)
    kernels = {entry["name"]: entry for entry in payload["kernel"]}
    assert kernels["hursh"]["params"]["speed_factor"]["default"] == 6.0
    assert kernels["hursh"]["params"]["speed_factor"]["unit"] == "m/s/um"
    calibres = {entry["name"]: entry for entry in payload["diameter"]}
    assert calibres["gev"]["params"]["mu"]["unit"] == "um"


def test_catalog_text_can_be_restricted_to_one_slot(capsys):
    assert main(["catalog", "noise"]) == 0
    out = capsys.readouterr().out
    assert "white" in out and "pink" in out
    assert "hursh" not in out


def test_presets_listing_and_export(tmp_path, capsys):
    assert main(["presets"]) == 0
    assert "reference_resting" in capsys.readouterr().out

    destination = tmp_path / "preset.json"
    assert main(["presets", "erp_occipital_pulse", "-o", str(destination)]) == 0
    spec = load_spec(destination)
    assert spec.n_trials == 20


def test_run_prints_a_summary_and_writes_signals(tmp_path, capsys):
    spec = ExperimentSpec(label="cli test").with_simulation(
        sample_rate_hz=100, sim_seconds=2.0, burnin_seconds=0.5, n_lags=30
    )
    spec_path = save_spec(spec, tmp_path / "spec.json")
    output = tmp_path / "out" / "run.npz"

    assert (
        main(
            [
                "run",
                str(spec_path),
                "-o",
                str(output),
                "--trials",
                "2",
                "--seconds",
                "1.0",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "cli test" in out
    assert "alpha peak" in out

    with np.load(output) as data:
        assert data["trials"].shape == (2, 100, 5)
        assert list(data["region_codes"]) == ["FL", "PL", "OL", "TL", "T"]
        assert ExperimentSpec.model_validate_json(str(data["spec_json"])).label == "cli test"


def test_sweep_prints_a_table_and_writes_csv(tmp_path, capsys):
    spec = ExperimentSpec().with_simulation(sample_rate_hz=100, sim_seconds=1.0, burnin_seconds=0.5, n_lags=30)
    spec_path = save_spec(spec, tmp_path / "spec.json")
    output = tmp_path / "sweep.csv"

    code = main(
        [
            "sweep",
            str(spec_path),
            "kernel.speed_factor",
            "--from",
            "4",
            "--to",
            "8",
            "--steps",
            "3",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "kernel.speed_factor" in out
    assert "spectral radius" in out
    rows = output.read_text().splitlines()
    assert rows[0].startswith("kernel.speed_factor")
    assert len(rows) == 1 + 3 * 5  # three sweep points, five regions each


def test_sweep_rejects_unknown_parameters(tmp_path, capsys):
    assert main(["sweep", "nonsense.path", "--from", "1", "--to", "2"]) == 2
    assert "unknown parameter" in capsys.readouterr().err


def test_cache_command_reports_the_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DIAXCONDEL_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["cache"]) == 0
    assert str(tmp_path / "cache") in capsys.readouterr().out
    assert main(["cache", "--clear"]) == 0
    assert "removed 0 cached file(s)" in capsys.readouterr().out
