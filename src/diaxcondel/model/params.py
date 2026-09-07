"""
User-facing simulation configuration.

These types are pydantic models so that user inputs are validated up front
with clear error messages. They are converted to plain numpy/scalar values
before entering any hot path.
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel, Field, model_validator


class SimulationParams(BaseModel):
    """
    Discrete-time simulation parameters.

    Parameters
    ----------
    sample_rate_hz : int
        Sampling rate in Hertz. Must be positive.
    sim_seconds : float
        Simulation duration *after* burn-in. Must be positive.
    burnin_seconds : float
        Burn-in duration discarded before analysis. Must be non-negative.
    n_lags : int
        Number of lag steps in the VAR model (``P`` in the paper). Must be
        such that ``n_lags / sample_rate_hz`` covers the longest expected
        transmission delay; default 300 at 1 kHz covers up to 300 ms.
    """

    model_config = {"frozen": True}

    sample_rate_hz: int = Field(gt=0)
    sim_seconds: float = Field(gt=0)
    burnin_seconds: float = Field(ge=0)
    n_lags: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_lag_coverage(self) -> SimulationParams:
        coverage_ms = 1000 * self.n_lags / self.sample_rate_hz
        if coverage_ms < 50:
            warnings.warn(
                f"n_lags={self.n_lags} at {self.sample_rate_hz} Hz covers only "
                f"{coverage_ms:.1f} ms of delay, which is shorter than typical "
                "cortico-cortical transmission times (~30-150 ms).",
                stacklevel=2,
            )
        return self

    @property
    def dt_s(self) -> float:
        """Sample period in seconds."""
        return 1.0 / self.sample_rate_hz

    @property
    def n_burnin_samples(self) -> int:
        """Number of burn-in samples (rounded to nearest int)."""
        return int(round(self.burnin_seconds * self.sample_rate_hz))

    @property
    def n_sim_samples(self) -> int:
        """Number of post-burn-in simulation samples."""
        return int(round(self.sim_seconds * self.sample_rate_hz))
