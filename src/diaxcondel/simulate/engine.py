"""
Vectorized simulation engine.

The core kernel runs a VAR(P) system forward one sample at a time:

    h(t) = [Phi_1 ... Phi_P] @ [h(t-1); ...; h(t-P)] + noise(t) + stim(t)

The stacked history is held in a sliding window over a ``(2P, N)`` buffer, so
each step costs a single ``(N, NP)`` by ``(NP,)`` matrix-vector product and the
window is re-based only once every ``P`` samples.

This is mathematically the top block-row of the companion-form update
``x[t] = A @ x[t - 1] + b[t]``, but it never forms the ``(NP, NP)`` companion
matrix: that costs ``P`` times more arithmetic per step and ``P`` times more
memory (2.6 GB at 60 regions and 300 lags). The companion matrix remains
available on :class:`~diaxcondel.model.var.linear.LinearVAR` for stability
analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from diaxcondel._rng import RNGLike, as_generator
from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.dynamics import Dynamics
from diaxcondel.model.params import SimulationParams
from diaxcondel.model.transfer import Identity
from diaxcondel.model.var import LinearVAR, NonlinearVAR

from .noise import NoiseFn


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Output of a single-trial simulation.

    Parameters
    ----------
    signal : FloatArray of shape (T, N)
        Time-domain signal after burn-in removal.
    regions : RegionSet
        Region labels corresponding to columns of ``signal``. ``None`` only
        if the dynamics did not carry region information.
    params : SimulationParams
        The parameters used to produce this result.
    """

    signal: FloatArray
    params: SimulationParams
    regions: RegionSet | None = None

    @property
    def sample_rate_hz(self) -> int:  # noqa: D102
        return self.params.sample_rate_hz

    @property
    def time_s(self) -> FloatArray:  # noqa: D102
        return np.arange(self.signal.shape[0]) / self.params.sample_rate_hz


def simulate(
    dynamics: Dynamics,
    params: SimulationParams,
    noise_fn: NoiseFn,
    *,
    rng: RNGLike = None,
    stim: FloatArray | None = None,
    regions: RegionSet | None = None,
    phi_schedule: list[FloatArray] | None = None,
) -> SimulationResult:
    """
    Simulate a single trial of the dynamics.

    Parameters
    ----------
    dynamics : Dynamics
        The model.
    params : SimulationParams
        Simulation parameters.
    noise_fn : NoiseFn
        Noise generator; see :mod:`diaxcondel.simulate.noise`.
    rng : RNGLike, optional
        Seed or generator. ``None`` uses a fresh OS-seeded generator.
    stim : FloatArray of shape (T_total, N), optional
        Stimulus signal sampled at ``sample_rate_hz`` for the full trial
        (burn-in + simulation). If ``None``, no stimulus is applied.
    regions : RegionSet, optional
        Region labels corresponding to columns of the simulated signal.
        If ``None``, the result will have no region information.
    phi_schedule : list of FloatArray of shape (P, N, N), optional
        Per-sample lag tensor schedule of length ``n_burnin + n_sim``.
        When provided, the VAR coefficients change over time (e.g. for
        connectivity modulations).  Requires :class:`LinearVAR` dynamics.
        If ``None``, the dynamics object supplies a constant Phi.

    Returns
    -------
    SimulationResult
        Simulated signal with burn-in removed.

    Raises
    ------
    ValueError
        If ``phi_schedule`` is provided with non-:class:`LinearVAR` dynamics,
        or if its length does not equal ``n_burnin + n_sim``.

    Notes
    -----
    For :class:`LinearVAR` the engine uses a vectorized state-space form;
    for any other :class:`Dynamics` it falls back to a Python-level loop
    calling :meth:`Dynamics.step`.
    """
    if params.n_burnin_samples < dynamics.n_lags:
        warnings.warn(
            f"burnin_samples ({params.n_burnin_samples}) must be >= n_lags "
            f"({dynamics.n_lags}) to fully absorb the VAR start-up transient",
            stacklevel=2,
        )

    gen = as_generator(rng)
    n = dynamics.n_nodes
    p = dynamics.n_lags
    n_total = params.n_burnin_samples + params.n_sim_samples

    noise = noise_fn(n, n_total, float(params.sample_rate_hz), gen)
    if stim is None:
        stim = np.zeros_like(noise)
    elif stim.shape != noise.shape:
        raise ValueError(f"stim shape {stim.shape} != noise shape {noise.shape}")

    if phi_schedule is not None:
        if not isinstance(dynamics, LinearVAR):
            raise ValueError(f"phi_schedule requires LinearVAR dynamics; got {type(dynamics).__name__}")
        if len(phi_schedule) != n_total:
            raise ValueError(f"phi_schedule length {len(phi_schedule)} != n_total {n_total}")
        signal = _simulate_linear_scheduled(phi_schedule, noise, stim, p, n)
    elif isinstance(dynamics, LinearVAR) or (
        isinstance(dynamics, NonlinearVAR) and isinstance(dynamics.transfer_fn, Identity)
    ):
        # The companion-form fast path requires LinearVAR's interface;
        # wrap a NonlinearVAR-with-Identity into a LinearVAR for the simulation.
        if isinstance(dynamics, NonlinearVAR):
            signal = _simulate_linear(LinearVAR(phi=dynamics.phi), noise, stim, p)
        else:
            signal = _simulate_linear(dynamics, noise, stim, p)
    else:
        signal = _simulate_generic(dynamics, noise, stim, p)

    signal = signal[params.n_burnin_samples :, :]
    return SimulationResult(signal=signal, params=params, regions=regions)


def _simulate_linear(model: LinearVAR, noise: FloatArray, stim: FloatArray, p: int) -> FloatArray:
    """Sliding-window state-space simulation for a linear VAR."""
    n_total, n = noise.shape
    top = _top_block(model.phi, p, n)  # (N, NP)
    # window[w : w + p] holds [h(t-1), h(t-2), ..., h(t-P)] contiguously, so
    # its ravel() is a free view; new samples are written just before the
    # window start, which is re-based to the middle every P steps.
    window = np.zeros((2 * p, n))
    w = p
    out = np.empty((n_total, n))
    for t in range(n_total):
        h = top @ window[w : w + p].ravel() + noise[t] + stim[t]
        out[t] = h
        if w == 0:
            window[p:] = window[:p]
            w = p
        w -= 1
        window[w] = h
    return out


def _simulate_generic(model: Dynamics, noise: FloatArray, stim: FloatArray, p: int) -> FloatArray:
    """Run the generic per-step driver for arbitrary :class:`Dynamics`."""
    n_total, n = noise.shape
    history = np.zeros((p, n))
    out = np.zeros((n_total, n))
    for t in range(n_total):
        out[t] = model.step(history, noise[t], stim[t])
        history = np.roll(history, 1, axis=0)
        history[0] = out[t]
    return out


def _simulate_linear_scheduled(
    phi_schedule: list[FloatArray],
    noise: FloatArray,
    stim: FloatArray,
    p: int,
    n: int,
) -> FloatArray:
    """
    Step-by-step simulation when Phi changes over time.

    When a connectivity modulation is active the companion matrix changes
    each sample; we cannot use the pre-computed constant-A shortcut and must
    recompute the top block whenever Phi changes.  For efficiency we detect
    runs of identical Phi (same object) and skip recomputes during plateaus.
    """
    n_total = noise.shape[0]
    out = np.empty((n_total, n))
    # Same sliding window as the constant-Phi path; see _simulate_linear.
    window = np.zeros((2 * p, n))
    w = p

    # Cache the most-recently-used Phi → top block to avoid redundant recomputes.
    _cached_phi: FloatArray = phi_schedule[0]
    _cached_top: FloatArray = _top_block(_cached_phi, p, n)

    for t in range(n_total):
        phi_t = phi_schedule[t]
        if phi_t is not _cached_phi:
            _cached_top = _top_block(phi_t, p, n)
            _cached_phi = phi_t
        # h(t) = sum_p Phi[p] @ h(t-p) + noise + stim
        #      = top_block @ [h(t-1); ...; h(t-P)] + noise + stim
        h = _cached_top @ window[w : w + p].ravel() + noise[t] + stim[t]
        out[t] = h
        if w == 0:
            window[p:] = window[:p]
            w = p
        w -= 1
        window[w] = h

    return out


def _top_block(phi: FloatArray, p: int, n: int) -> FloatArray:
    """Extract the (N, NP) top-block from a Phi tensor."""
    top = np.empty((n, n * p))
    for k in range(p):
        top[:, k * n : (k + 1) * n] = phi[k]
    return top
