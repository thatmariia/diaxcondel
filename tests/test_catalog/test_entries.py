"""
Tests for what the built-in entries actually produce.

The registry tests check that entries are well formed; these check that the
numbers they return mean what their documentation says.
"""

import numpy as np
import pytest

from diaxcondel import catalog
from diaxcondel.kernels.diameter import positive_diameters


def _kernel(name: str = "hursh", params: dict | None = None, diameter: str = "gev"):
    """Build a delay kernel, supplying the calibre distribution when the kernel needs one."""
    entry = catalog.get("kernel", name)
    context = {"diameter": catalog.build("diameter", diameter)} if "diameter" in entry.context else {}
    return catalog.build("kernel", name, params or {}, **context)


def _sources(distance_name: str = "steeghs_2025", **params):
    distances = catalog.build("distances", distance_name, params)
    return distances, distances.regions


# ---------------------------------------------------------------------------
# Distance sources
# ---------------------------------------------------------------------------


def test_five_lobe_distances_scale_the_whole_geometry():
    base = catalog.build("distances", "steeghs_2025")
    stretched = catalog.build("distances", "steeghs_2025", {"scale": 2.0})
    np.testing.assert_allclose(stretched.matrix(), base.matrix() * 2.0)
    assert base.regions.codes == ("FL", "PL", "OL", "TL", "T")


def test_power_law_distances_respect_their_bounds_and_seed():
    provider = catalog.build("distances", "power_law", {"n_regions": 8, "d_min_cm": 2.0, "d_max_cm": 9.0})
    matrix = provider.matrix()
    offdiag = matrix[~np.eye(8, dtype=bool)]
    assert offdiag.min() >= 2.0
    assert offdiag.max() <= 9.0
    again = catalog.build("distances", "power_law", {"n_regions": 8, "d_min_cm": 2.0, "d_max_cm": 9.0})
    np.testing.assert_array_equal(matrix, again.matrix())


def test_hub_source_connects_every_region_to_the_hub():
    provider = catalog.build(
        "distances",
        "power_law_with_hub",
        {"n_regions": 6, "hub_distance_cm": 7.0, "hub_spread_cm": 0.0, "hub_code": "T"},
    )
    assert provider.regions.codes[-1] == "T"
    matrix = provider.matrix()
    hub_row = matrix[-1, :-1]
    np.testing.assert_allclose(hub_row, 7.0)
    assert matrix.shape == (7, 7)


def test_ring_distances_are_evenly_spaced():
    provider = catalog.build("distances", "ring", {"n_regions": 8, "circumference_cm": 80.0})
    matrix = provider.matrix()
    assert matrix[0, 1] == pytest.approx(10.0)
    assert matrix[0, 4] == pytest.approx(40.0), "opposite points are half the circumference apart"
    assert matrix[0, 7] == pytest.approx(10.0), "the ring wraps around"


# ---------------------------------------------------------------------------
# Connectivity rules
# ---------------------------------------------------------------------------


def test_uniform_coupling_fixes_the_total_input_per_region():
    distances, regions = _sources()
    weights = catalog.build("connectivity", "uniform", {"coupling": 0.8}, regions=regions, distances=distances)
    totals = weights.matrix().sum(axis=1)
    np.testing.assert_allclose(totals, 0.8)


def test_distance_decay_prefers_short_connections():
    distances, regions = _sources()
    matrix = distances.matrix()
    weights = catalog.build(
        "connectivity",
        "distance_decay",
        {"form": "exponential", "length_constant_cm": 4.0},
        regions=regions,
        distances=distances,
    ).matrix()
    pairs = [(i, j) for i in range(len(regions)) for j in range(len(regions)) if i != j]
    shortest = min(pairs, key=lambda pair: matrix[pair])
    longest = max(pairs, key=lambda pair: matrix[pair])
    assert weights[shortest] > weights[longest]

    power = catalog.build(
        "connectivity",
        "distance_decay",
        {"form": "power", "exponent": 0.0},
        regions=regions,
        distances=distances,
    ).matrix()
    assert np.allclose(power[power > 0], power[power > 0][0]), "a zero exponent is uniform coupling"


def test_distance_decay_can_cut_long_connections():
    distances, regions = _sources()
    weights = catalog.build(
        "connectivity",
        "distance_decay",
        {"max_distance_cm": 7.0},
        regions=regions,
        distances=distances,
    ).matrix()
    assert weights[distances.matrix() > 7.0].max(initial=0.0) == 0.0


def test_random_sparse_is_sparse_symmetric_and_seeded():
    distances, regions = _sources("power_law", n_regions=20)
    first = catalog.build(
        "connectivity",
        "random_sparse",
        {"density": 0.3, "seed": 5},
        regions=regions,
        distances=distances,
    ).matrix()
    second = catalog.build(
        "connectivity",
        "random_sparse",
        {"density": 0.3, "seed": 5},
        regions=regions,
        distances=distances,
    ).matrix()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, first.T)
    density = np.count_nonzero(np.triu(first, k=1)) / (20 * 19 / 2)
    assert 0.15 < density < 0.45


# ---------------------------------------------------------------------------
# Kernels, noise, drives
# ---------------------------------------------------------------------------


def test_fixed_speed_kernel_peaks_at_length_over_speed():
    kernel = _kernel("fixed_speed", {"speed_m_s": 5.0, "jitter_ms": 0.5})
    delays = np.arange(1, 201) * 1e-3
    density = kernel.pdf(delays, distance_cm=10.0)
    peak_ms = 1000 * delays[int(np.argmax(density))]
    assert peak_ms == pytest.approx(20.0, abs=1.0)  # 0.1 m / 5 m/s = 20 ms


def test_local_delay_kernels_return_normalised_mass():
    none = catalog.build("local_delay", "none")
    mass = none.mass(1e-3, 10)
    assert mass[0] == 1.0 and mass[1:].sum() == 0.0

    gamma = catalog.build("local_delay", "gamma", {"mean_ms": 5.0, "shape": 4.0})
    mass = gamma.mass(1e-3, 100)
    assert mass.sum() == pytest.approx(1.0)
    mean_ms = float(np.dot(np.arange(100), mass))
    assert mean_ms == pytest.approx(5.0, abs=0.5)


def test_colored_noise_exponent_sets_the_input_slope():
    from diaxcondel.spectral import welch_psd

    noise_fn = catalog.build("noise", "colored", {"exponent": 2.0})
    signal = noise_fn(2, 20000, 500.0, 11)
    freqs, psd = welch_psd(signal, 500.0, segment_seconds=4.0)
    band = (freqs >= 2) & (freqs <= 100)
    slope = np.polyfit(np.log10(freqs[band]), np.log10(psd[band, 0]), 1)[0]
    assert slope == pytest.approx(-2.0, abs=0.2)


def test_sine_burst_drive_has_the_requested_frequency_and_length():
    distances, regions = _sources()
    drive = catalog.build(
        "drive",
        "sine_burst",
        {"targets": ("OL",), "frequency_hz": 12.0, "duration_seconds": 1.0, "ramp_seconds": 0.0},
        regions=regions,
    )
    waveform = drive.render(1000.0)
    assert waveform.size == 1000
    spectrum = np.abs(np.fft.rfft(waveform))
    peak_hz = float(np.fft.rfftfreq(waveform.size, d=1e-3)[int(np.argmax(spectrum))])
    assert peak_hz == pytest.approx(12.0, abs=1.0)
    assert [target.code for target in drive.targets] == ["OL"]


def test_custom_waveform_drive_is_scaled_to_the_amplitude():
    distances, regions = _sources()
    drive = catalog.build(
        "drive",
        "custom_waveform",
        {"samples": "[0, 1, 2, 1, 0]", "amplitude": 6.0},
        regions=regions,
    )
    np.testing.assert_allclose(drive.render(1000.0), [0.0, 3.0, 6.0, 3.0, 0.0])


def test_step_modulation_targets_incoming_connections():
    distances, regions = _sources()
    modulation = catalog.build(
        "modulation",
        "step_gain",
        {"targets": ("OL",), "gain": 2.0, "direction": "incoming"},
        regions=regions,
    )
    factor = modulation.factor
    occipital = regions.index("OL")
    np.testing.assert_allclose(factor[occipital, :], 2.0)
    others = [i for i in range(len(regions)) if i != occipital]
    np.testing.assert_allclose(factor[np.ix_(others, others)], 1.0)


# ---------------------------------------------------------------------------
# Delay kernel families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hursh", "gamma_delay", "fixed_speed"])
def test_every_kernel_is_a_normalised_delay_distribution(name):
    kernel = _kernel(name)
    delays = np.arange(1, 601) * 1e-3
    density = np.asarray(kernel.pdf(delays, 8.0), dtype=float)
    assert np.all(density >= 0)
    assert np.all(np.isfinite(density))
    mass = float(density.sum() * 1e-3)
    assert 0.9 <= mass <= 1.01, f"{name} lost too much probability mass within 600 ms"
    batched = np.asarray(kernel.pdf_batched(delays, np.array([8.0, 12.0])), dtype=float)
    assert batched.shape == (2, delays.size)
    np.testing.assert_allclose(batched[0], density, rtol=1e-10)


@pytest.mark.parametrize("diameter", ["gev", "gamma", "lognormal", "rayleigh"])
def test_calibre_distributions_all_scale_delay_with_distance(diameter):
    kernel = _kernel("hursh", diameter=diameter)
    delays = np.arange(1, 801) * 1e-3

    def peak_ms(distance_cm: float) -> float:
        density = np.asarray(kernel.pdf(delays, distance_cm), dtype=float)
        return 1000 * float(delays[int(np.argmax(density))])

    near, far = peak_ms(6.0), peak_ms(12.0)
    assert far == pytest.approx(2 * near, rel=0.05), "delay must be proportional to length"


def test_faster_axons_shorten_delays():
    delays = np.arange(1, 801) * 1e-3

    def peak_ms(speed: float) -> float:
        kernel = _kernel("hursh", {"speed_factor": speed})
        density = np.asarray(kernel.pdf(delays, 8.0), dtype=float)
        return 1000 * float(delays[int(np.argmax(density))])

    assert peak_ms(9.0) < peak_ms(6.0) < peak_ms(3.0)


def test_gamma_delay_shape_controls_the_spread():
    delays = np.arange(1, 801) * 1e-3

    def spread_ms(shape: float) -> float:
        kernel = _kernel("gamma_delay", {"shape": shape})
        density = np.asarray(kernel.pdf(delays, 8.0), dtype=float)
        density = density / density.sum()
        mean = float(np.dot(delays, density))
        return 1000 * float(np.sqrt(np.dot((delays - mean) ** 2, density)))

    assert spread_ms(16.0) < spread_ms(2.0)


# ---------------------------------------------------------------------------
# Calibres and speed are separate choices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gev", "gamma", "lognormal", "rayleigh"])
def test_diameter_distributions_are_positive_and_normalised(name):
    distribution = catalog.build("diameter", name)
    grid = np.linspace(1e-4, 4.0, 4000)
    density = np.asarray(distribution.pdf(grid), dtype=float)
    assert np.all(density >= 0)
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=0.02)
    draws = distribution.sample(2000, np.random.default_rng(0))
    assert draws.shape == (2000,)
    # Fitted calibre distributions can put a little mass below zero; the
    # model only ever uses the positive part.
    assert (draws > 0).mean() > 0.95
    physical = positive_diameters(distribution, 500, np.random.default_rng(0))
    assert physical.shape == (500,)
    assert np.all(physical > 0)


def test_the_speed_factor_lives_with_the_conduction_relation_not_the_calibres():
    entry = catalog.get("kernel", "hursh")
    assert "diameter" in entry.context, "the kernel receives the calibre distribution"
    assert "speed_factor" in entry.defaults()
    for name in ("gev", "gamma", "lognormal", "rayleigh"):
        assert "speed_factor" not in catalog.get("diameter", name).defaults()


def test_the_same_calibres_give_different_delays_at_different_speeds():
    delays = np.arange(1, 801) * 1e-3
    slow = _kernel("hursh", {"speed_factor": 3.0})
    fast = _kernel("hursh", {"speed_factor": 9.0})
    peak = lambda kernel: float(delays[int(np.argmax(kernel.pdf(delays, 8.0)))])  # noqa: E731
    assert peak(slow) == pytest.approx(3 * peak(fast), rel=0.05)


def test_relayed_kernels_keep_the_calibres_and_speed_of_their_direct_form():
    kernel = _kernel("hursh", {"speed_factor": 7.0}, diameter="lognormal")
    relay = kernel.relay_kernel(n_samples=4000, rng=3)
    assert relay.parameters() == kernel.parameters()

    delays = np.arange(1, 2001) * 1e-3
    two_leg = np.asarray(relay.pdf_two_leg(delays, 4.0, 5.0), dtype=float)
    single = np.asarray(relay.pdf(delays, 9.0), dtype=float)
    assert np.all(two_leg >= 0)

    def moments(density: np.ndarray) -> tuple[float, float]:
        mass = density.sum()
        mean = float(np.dot(delays, density) / mass)
        spread = float(np.sqrt(np.dot((delays - mean) ** 2, density) / mass))
        return mean, spread

    two_leg_mean, two_leg_spread = moments(two_leg)
    single_mean, single_spread = moments(single)
    # Travel time adds, so two legs of 4 and 5 cm take as long on average as
    # one 9 cm hop — but each leg draws its own axons, so the extremes cancel
    # and the arrival is more tightly timed.
    assert two_leg_mean == pytest.approx(single_mean, rel=0.1)
    assert two_leg_spread < single_spread
