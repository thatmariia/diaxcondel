"""Simulation engines, noise generators, and stimulus helpers."""

from __future__ import annotations

from .engine import SimulationResult, simulate
from .noise import colored_noise, get_noise_fn, pink_noise, white_noise
from .stimulus import (
    ArrayDrive,
    Drive,
    DriveTarget,
    GaussianPulseDrive,
    SineBurstDrive,
    SquarePulseDrive,
    build_drive_array,
)
from .trials import TrialResult, simulate_trials

__all__ = [
    "ArrayDrive",
    "Drive",
    "DriveTarget",
    "GaussianPulseDrive",
    "SimulationResult",
    "SineBurstDrive",
    "SquarePulseDrive",
    "TrialResult",
    "build_drive_array",
    "colored_noise",
    "get_noise_fn",
    "pink_noise",
    "simulate",
    "simulate_trials",
    "white_noise",
]
