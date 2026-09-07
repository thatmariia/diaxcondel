"""
Dynamics protocol.

A :class:`Dynamics` is a discrete-time dynamical system on ``N`` nodes that
can be (a) advanced in time, given history, noise, and stimulus, and
(b) optionally evaluated for its closed-form transfer function in the
frequency domain.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from diaxcondel._typing import ComplexArray, FloatArray


@runtime_checkable
class Dynamics(Protocol):
    """Protocol for a macroscopic dynamical model on a region set."""

    @property
    def n_nodes(self) -> int:
        """Number of nodes (regions)."""
        ...

    @property
    def n_lags(self) -> int:
        """Maximum lag depth ``P`` in samples."""
        ...

    def step(
        self,
        history: FloatArray,
        noise: FloatArray,
        stim: FloatArray,
    ) -> FloatArray:
        """
        Advance the system by one sample.

        Parameters
        ----------
        history : FloatArray of shape (P, N)
            State at the previous ``P`` time steps, with ``history[0]`` the
            most recent (i.e. ``t - 1``).
        noise : FloatArray of shape (N,)
            Stochastic input at this step.
        stim : FloatArray of shape (N,)
            Deterministic external input at this step (zero for resting-state).

        Returns
        -------
        FloatArray of shape (N,)
            Next state.
        """
        ...

    def transfer(self, freqs_hz: FloatArray, dt_s: float) -> ComplexArray | None:
        """
        Closed-form transfer function evaluated at frequencies.

        Parameters
        ----------
        freqs_hz : FloatArray of shape (F,)
            Frequencies (in Hertz) at which to evaluate.
        dt_s : float
            Sampling period.

        Returns
        -------
        ComplexArray of shape (F, N, N) or None
            Transfer matrix at each frequency, or ``None`` if no closed form
            is available (e.g. non-linear dynamics). Callers fall back to
            simulation-based PSD estimation in that case.
        """
        ...
