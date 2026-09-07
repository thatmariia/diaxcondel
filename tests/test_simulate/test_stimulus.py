"""Tests for Drive classes and stimulus-array construction."""

import numpy as np

from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.params import SimulationParams
from diaxcondel.simulate.stimulus import (
    ArrayDrive,
    DriveTarget,
    GaussianPulseDrive,
    SquarePulseDrive,
    build_drive_array,
)


class TestDriveTarget:
    def test_default_weight(self):
        t = DriveTarget("A")
        assert t.weight == 1.0


class TestArrayDrive:
    def test_render_returns_stored_waveform(self):
        wf = np.array([0.5, 1.0, 0.5])
        drive = ArrayDrive(waveform=wf, targets=(DriveTarget("A"),))
        np.testing.assert_array_equal(drive.render(1000.0), wf)

    def test_render_ignores_sample_rate(self):
        wf = np.array([1.0, 2.0])
        drive = ArrayDrive(waveform=wf, targets=(DriveTarget("A"),))
        np.testing.assert_array_equal(drive.render(500.0), wf)


class TestGaussianPulseDrive:
    def test_render_shape(self):
        drive = GaussianPulseDrive(
            duration_seconds=0.1,
            peak_seconds=0.05,
            width_seconds=0.01,
            targets=(DriveTarget("T"),),
        )
        out = drive.render(1000.0)
        assert out.shape == (100,)

    def test_render_peak_near_amplitude(self):
        drive = GaussianPulseDrive(
            duration_seconds=0.1,
            peak_seconds=0.05,
            width_seconds=0.01,
            amplitude=2.0,
            targets=(DriveTarget("T"),),
        )
        out = drive.render(1000.0)
        assert np.isclose(out.max(), 2.0, atol=1e-3)


class TestSquarePulseDrive:
    def test_render_shape_and_value(self):
        drive = SquarePulseDrive(
            duration_seconds=0.05,
            amplitude=3.0,
            targets=(DriveTarget("FL"),),
        )
        out = drive.render(1000.0)
        assert out.shape == (50,)
        assert np.all(out == 3.0)


class TestBuildDriveArray:
    def test_offsets_drive_after_burnin(self):
        regions = RegionSet.from_codes(["A", "B"])
        params = SimulationParams(sample_rate_hz=10, sim_seconds=1.0, burnin_seconds=0.5, n_lags=2)
        drive = SquarePulseDrive(
            duration_seconds=0.2,
            amplitude=3.0,
            targets=(DriveTarget("B", weight=2.0),),
        )
        # onset=0.2 s → sample 0.5*10 + 0.2*10 = 5 + 2 = 7
        stim = build_drive_array([(0.2, drive)], regions, params)

        assert stim.shape == (15, 2)
        np.testing.assert_array_equal(stim[:7], np.zeros((7, 2)))
        # 0.2 s at 10 Hz = 2 samples; weight 2.0 × amplitude 3.0 = 6.0
        np.testing.assert_array_equal(stim[7:9, 1], np.array([6.0, 6.0]))

    def test_overlapping_drives_are_summed(self):
        regions = RegionSet.from_codes(["A"])
        params = SimulationParams(sample_rate_hz=10, sim_seconds=1.0, burnin_seconds=0.0, n_lags=2)
        d1 = SquarePulseDrive(duration_seconds=0.5, amplitude=1.0, targets=(DriveTarget("A"),))
        d2 = SquarePulseDrive(duration_seconds=0.5, amplitude=2.0, targets=(DriveTarget("A"),))
        stim = build_drive_array([(0.0, d1), (0.0, d2)], regions, params)
        np.testing.assert_array_equal(stim[:5, 0], np.full(5, 3.0))

    def test_drive_beyond_signal_is_ignored(self):
        regions = RegionSet.from_codes(["A"])
        params = SimulationParams(sample_rate_hz=10, sim_seconds=0.5, burnin_seconds=0.0, n_lags=2)
        drive = SquarePulseDrive(duration_seconds=0.1, amplitude=1.0, targets=(DriveTarget("A"),))
        # onset at 1.0 s is beyond the 0.5 s simulation
        stim = build_drive_array([(1.0, drive)], regions, params)
        np.testing.assert_array_equal(stim, np.zeros((5, 1)))

    def test_empty_list_gives_zeros(self):
        regions = RegionSet.from_codes(["A", "B"])
        params = SimulationParams(sample_rate_hz=10, sim_seconds=1.0, burnin_seconds=0.1, n_lags=2)
        stim = build_drive_array([], regions, params)
        assert stim.shape == (11, 2)
        np.testing.assert_array_equal(stim, np.zeros((11, 2)))
