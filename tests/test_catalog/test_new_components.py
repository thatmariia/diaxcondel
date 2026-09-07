"""The components added for model exploration: what each one guarantees."""

import numpy as np
import pytest

from diaxcondel import catalog
from diaxcondel.connectome.regions import RegionSet


def test_shared_input_has_the_correlation_it_promises():
    for requested in (0.0, 0.4, 1.0):
        noise_fn = catalog.build("noise", "correlated", {"std": 1.0, "correlation": requested})
        sample = noise_fn(5, 20000, 200.0, 1)
        realised = np.corrcoef(sample.T)[~np.eye(5, dtype=bool)]
        assert float(realised.mean()) == pytest.approx(requested, abs=0.05)


def test_sharing_the_input_leaves_its_strength_unchanged():
    independent = catalog.build("noise", "correlated", {"correlation": 0.0})(4, 20000, 200.0, 2)
    shared = catalog.build("noise", "correlated", {"correlation": 0.8})(4, 20000, 200.0, 2)
    assert float(shared.var()) == pytest.approx(float(independent.var()), rel=0.05)


def test_the_saturating_transfers_bound_activity_and_the_rectifier_does_not():
    values = np.linspace(-20.0, 20.0, 101)
    sigmoid = catalog.build("transfer", "centred_sigmoid")
    hard = catalog.build("transfer", "hard_saturation", {"limit": 2.0})
    rectified = catalog.build("transfer", "threshold_linear")

    assert np.all(np.abs(sigmoid(values)) <= 0.5)
    assert np.all(np.abs(hard(values)) <= 2.0)
    # Below the limit a hard saturation is exactly linear, which is what makes
    # it the control for the sigmoid's smooth compression.
    small = np.linspace(-1.0, 1.0, 21)
    assert np.allclose(hard(small), small)
    assert np.all(rectified(values) >= 0.0)
    assert rectified(values).max() == pytest.approx(20.0)


def test_the_biexponential_delay_rises_then_falls():
    kernel = catalog.build("local_delay", "biexponential", {"rise_ms": 1.0, "decay_ms": 6.0})
    mass = kernel.mass(1e-3, 200)
    lags_ms = np.arange(200)

    assert mass.sum() == pytest.approx(1.0)
    assert np.all(mass >= 0.0)
    assert mass[0] == pytest.approx(0.0, abs=1e-12)  # nothing arrives instantly
    peak = int(np.argmax(mass))
    assert 0 < peak < 10  # rises over the shorter constant
    assert float((lags_ms * mass).sum()) == pytest.approx(7.0, abs=0.5)  # rise + decay


def test_the_chirp_sweeps_from_its_first_frequency_to_its_last():
    regions = RegionSet.from_codes(["A", "B"])
    drive = catalog.build(
        "drive",
        "chirp",
        {"start_hz": 4.0, "end_hz": 32.0, "duration_seconds": 8.0, "ramp_seconds": 0.0},
        regions=regions,
    )
    wave = drive.render(500.0)

    # Count zero crossings in the first and last second: the rate is the
    # instantaneous frequency, and it must rise across the sweep.
    def crossings(segment: np.ndarray) -> int:
        return int(np.count_nonzero(np.diff(np.signbit(segment))))

    assert crossings(wave[:500]) < crossings(wave[-500:])
    # Over the whole sweep the mean frequency is the midpoint of the range.
    assert crossings(wave) == pytest.approx(2 * 0.5 * (4.0 + 32.0) * 8.0, rel=0.05)


def test_the_lattice_places_regions_on_a_grid():
    distances = catalog.build("distances", "lattice", {"rows": 3, "columns": 4, "spacing_cm": 2.0})
    matrix = distances.matrix()

    assert len(distances.regions) == 12
    assert matrix.shape == (12, 12)
    assert np.allclose(np.diagonal(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)
    # Neighbours sit one spacing apart; opposite corners span the diagonal.
    assert matrix[0, 1] == pytest.approx(2.0)
    assert matrix[0, 11] == pytest.approx(2.0 * np.hypot(2, 3))


def test_matrices_can_be_loaded_from_files(tmp_path):
    lengths_mm = np.array([[0.0, 80.0, 120.0], [80.0, 0.0, 90.0], [120.0, 90.0, 0.0]])
    matrix_file = tmp_path / "lengths.csv"
    np.savetxt(matrix_file, lengths_mm, delimiter=",")
    codes_file = tmp_path / "codes.txt"
    codes_file.write_text("FL\nPL\nOL\n")

    distances = catalog.build(
        "distances", "from_file", {"path": str(matrix_file), "codes_path": str(codes_file), "scale_to_cm": 0.1}
    )
    assert distances.regions.codes == ("FL", "PL", "OL")
    assert distances.matrix()[0, 2] == pytest.approx(12.0)

    weight_file = tmp_path / "weights.csv"
    np.savetxt(weight_file, np.full((3, 3), 2.0), delimiter=",")
    weights = catalog.build(
        "connectivity",
        "from_file",
        {"path": str(weight_file), "normalization": "max", "scale": 0.5},
        regions=distances.regions,
    )
    assert np.allclose(np.diagonal(weights.matrix()), 0.0)
    assert weights.matrix()[0, 1] == pytest.approx(0.5)
