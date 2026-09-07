"""
Multi-trial simulation engine and result container.

:class:`TrialResult` holds the per-trial signals from any multi-trial
simulation together with the metadata needed for downstream analysis.

:func:`simulate_trials` runs ``n_trials`` independent replications of a VAR
simulation.  Two usage modes are supported:

**Event-driven** (``events`` provided)
    Each trial builds its own stimulus array and connectivity-modulation
    schedule from the resolved :class:`~diaxcondel.erp.event.ERPEvent` list.
    Per-trial jitter, if present in the events, is applied independently for
    every trial.  ``stim`` and ``phi_schedule`` kwargs must not be supplied in
    this mode.

**Fixed-input** (``events=None``)
    Every trial sees the same ``stim`` and/or ``phi_schedule`` (or neither);
    only the noise differs across trials.

Trials are independent: each gets its own random number stream derived from a
:class:`numpy.random.SeedSequence`, so results are reproducible and the same
regardless of ``n_jobs`` when an integer seed is supplied.

Parallelism
-----------
- ``n_jobs=1`` runs trials serially (useful for debugging).
- ``n_jobs > 1`` uses :mod:`concurrent.futures.ProcessPoolExecutor`.  All
  worker inputs must be picklable (``numpy`` arrays, dataclasses, integers).
- Exact reproducibility across ``n_jobs`` values **requires an integer seed**.
  If the caller passes a :class:`numpy.random.Generator`, derived child
  streams may differ between serial and parallel execution due to Python
  object identity semantics.

Memory
------
All ``n_trials`` trial arrays are kept in memory.  For large experiments
(many trials × long signals × many regions) the caller should monitor RSS and
consider post-processing in chunks.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from diaxcondel._rng import RNGLike
from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.dynamics import Dynamics
from diaxcondel.model.params import SimulationParams
from diaxcondel.model.var import LinearVAR
from diaxcondel.simulate.engine import simulate
from diaxcondel.simulate.noise import NoiseFn
from diaxcondel.simulate.stimulus import Drive, build_drive_array

if TYPE_CHECKING:
    from diaxcondel.erp.event import ERPEvent


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialResult:
    """
    Output of a multi-trial simulation.

    Parameters
    ----------
    trials : FloatArray of shape (n_trials, T, N)
        Per-trial post-burn-in signals.  Axis 0 = trials, axis 1 = time,
        axis 2 = regions.
    params : SimulationParams
        Simulation parameters used to generate the trials.
    regions : RegionSet or None
        Region labels corresponding to columns of each trial signal.
        ``None`` when no region information was provided to the simulator.

    Notes
    -----
    All trials are stored; downstream code can choose to discard them after
    computing the average if memory is a concern.
    """

    trials: FloatArray
    params: SimulationParams
    regions: RegionSet | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.trials)
        if t.ndim != 3:
            raise ValueError(f"trials must be 3-D (n_trials, T, N); got shape {t.shape}")
        object.__setattr__(self, "trials", t)

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return int(self.trials.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of time samples per trial."""
        return int(self.trials.shape[1])

    @property
    def n_regions(self) -> int:
        """Number of regions."""
        return int(self.trials.shape[2])

    @property
    def sample_rate_hz(self) -> int:
        """Sampling rate in Hertz."""
        return self.params.sample_rate_hz

    @property
    def time_s(self) -> FloatArray:
        """Time axis in seconds (post-burn-in, length = n_samples)."""
        return np.arange(self.n_samples) / self.params.sample_rate_hz

    @property
    def average(self) -> FloatArray:
        """
        Trial-average signal of shape (T, N).

        For the linear zero-mean model with pure connectivity modulation and
        no additive drive, the expectation of this quantity is near zero.
        """
        return self.trials.mean(axis=0)

    @property
    def sem(self) -> FloatArray:
        """Standard error of the mean across trials, shape (T, N)."""
        if self.n_trials < 2:
            return np.full((self.n_samples, self.n_regions), np.nan)
        return self.trials.std(axis=0, ddof=1) / np.sqrt(self.n_trials)

    @property
    def variance(self) -> FloatArray:
        """Per-trial variance estimate across trials, shape (T, N)."""
        if self.n_trials < 2:
            return np.full((self.n_samples, self.n_regions), np.nan)
        return self.trials.var(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_trials(
    dynamics: Dynamics,
    params: SimulationParams,
    noise_fn: NoiseFn,
    *,
    n_trials: int = 1,
    events: Sequence[ERPEvent] | None = None,
    regions: RegionSet | None = None,
    stim: FloatArray | None = None,
    phi_schedule: list[FloatArray] | None = None,
    rng: RNGLike = None,
    n_jobs: int = 1,
) -> TrialResult:
    """
    Simulate many independent trials.

    Parameters
    ----------
    dynamics : Dynamics
        VAR dynamics model (linear or nonlinear).  For event-driven mode with
        connectivity modulations a :class:`~diaxcondel.model.var.LinearVAR`
        is required.
    params : SimulationParams
        Simulation timing parameters.
    noise_fn : NoiseFn
        Noise generator; see :mod:`diaxcondel.simulate.noise`.
    n_trials : int
        Number of independent trials.  Must be positive.
    events : sequence of ERPEvent or None
        When provided, each trial builds per-trial stimulus and connectivity
        modulation from these events (with per-trial jitter applied
        independently).  Mutually exclusive with ``stim`` and
        ``phi_schedule``.
    regions : RegionSet or None
        Region labels.  Required when ``events`` is provided (needed to
        project drive waveforms onto region columns).  Optional otherwise.
    stim : FloatArray of shape (T_total, N) or None
        Constant stimulus shared across all trials.  Ignored when ``events``
        is provided.
    phi_schedule : list of FloatArray or None
        Constant per-sample Phi schedule shared across all trials.  Ignored
        when ``events`` is provided.
    rng : int, SeedSequence, Generator, or None
        Randomness source.  For cross-``n_jobs`` reproducibility pass an
        ``int`` or :class:`numpy.random.SeedSequence`.  ``None`` uses a fresh
        OS-seeded sequence.
    n_jobs : int
        Number of worker processes.  ``1`` runs serially; ``>1`` spawns
        worker processes via :class:`concurrent.futures.ProcessPoolExecutor`.

    Returns
    -------
    TrialResult
        All trial signals, shape ``(n_trials, T, N)``, with metadata.

    Raises
    ------
    ValueError
        If ``events`` is provided together with ``stim`` or ``phi_schedule``.
        If ``events`` is provided but ``regions`` is ``None``.
        If ``n_trials < 1`` or ``n_jobs < 1``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1; got {n_trials}")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1; got {n_jobs}")

    if events is not None:
        if stim is not None or phi_schedule is not None:
            raise ValueError("stim and phi_schedule must not be supplied when events is provided")
        if regions is None:
            raise ValueError("regions must be provided when events is provided")

    seed_seq = _to_seed_sequence(rng)
    child_sequences = seed_seq.spawn(n_trials)

    if events is not None:
        # Event-driven mode: per-trial stimulus and modulation schedule.
        _dynamics = dynamics  # must be LinearVAR if modulations are present
        _events = list(events)
        _regions: RegionSet = regions  # type: ignore[assignment]  # checked above

        if n_jobs == 1:
            trial_arrays = [
                _simulate_one_erp_trial(_dynamics, params, noise_fn, _events, _regions, child_sequences[i])
                for i in range(n_trials)
            ]
        else:
            args = [
                _ERPTrialArgs(
                    phi=dynamics.phi if isinstance(dynamics, LinearVAR) else None,  # type: ignore[union-attr]
                    dynamics=dynamics if not isinstance(dynamics, LinearVAR) else None,
                    params=params,
                    noise_fn=noise_fn,
                    events=_events,
                    region_codes=_regions.codes,
                    seed_seq=child_sequences[i],
                )
                for i in range(n_trials)
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
                trial_arrays = list(pool.map(_erp_worker, args))
    else:
        # Fixed-input mode: same stim/phi_schedule for every trial, only noise differs.
        if n_jobs == 1:
            trial_arrays = [
                _simulate_one_fixed_trial(dynamics, params, noise_fn, stim, phi_schedule, regions, child_sequences[i])
                for i in range(n_trials)
            ]
        else:
            args_fixed = [
                _FixedTrialArgs(
                    phi=dynamics.phi if isinstance(dynamics, LinearVAR) else None,  # type: ignore[union-attr]
                    dynamics=dynamics if not isinstance(dynamics, LinearVAR) else None,
                    params=params,
                    noise_fn=noise_fn,
                    stim=stim,
                    phi_schedule=phi_schedule,
                    region_codes=regions.codes if regions is not None else None,
                    seed_seq=child_sequences[i],
                )
                for i in range(n_trials)
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
                trial_arrays = list(pool.map(_fixed_worker, args_fixed))

    return TrialResult(
        trials=np.stack(trial_arrays, axis=0),
        params=params,
        regions=regions,
    )


# ---------------------------------------------------------------------------
# Internal helpers — ERP (event-driven) path
# ---------------------------------------------------------------------------


@dataclass
class _ERPTrialArgs:
    """Picklable argument bundle for one ERP worker trial."""

    phi: FloatArray | None
    dynamics: Dynamics | None
    params: SimulationParams
    noise_fn: NoiseFn
    events: list[ERPEvent]
    region_codes: tuple[str, ...]
    seed_seq: np.random.SeedSequence

    def __post_init__(self) -> None:
        if self.phi is None and self.dynamics is None:
            raise ValueError("_ERPTrialArgs: exactly one of phi or dynamics must be provided")


def _erp_worker(args: _ERPTrialArgs) -> FloatArray:
    """Entry point for each ERP worker process."""
    from diaxcondel.connectome.regions import RegionSet

    dynamics: Dynamics
    if args.phi is not None:
        dynamics = LinearVAR(phi=args.phi)
    else:
        assert args.dynamics is not None
        dynamics = args.dynamics
    regions = RegionSet.from_codes(list(args.region_codes))
    return _simulate_one_erp_trial(dynamics, args.params, args.noise_fn, args.events, regions, args.seed_seq)


def _simulate_one_erp_trial(
    dynamics: Dynamics,
    params: SimulationParams,
    noise_fn: NoiseFn,
    events: Sequence[ERPEvent],
    regions: RegionSet,
    seed_seq: np.random.SeedSequence,
) -> FloatArray:
    """Run a single ERP trial and return the post-burn-in signal (T, N)."""
    from diaxcondel.erp.event import resolve_event
    from diaxcondel.erp.modulation import build_phi_schedule

    jitter_gen, noise_gen = (np.random.default_rng(s) for s in seed_seq.spawn(2))

    resolved = [resolve_event(ev, jitter_gen) for ev in events]

    modulations_with_onset = [(r.onset_seconds, r.modulation) for r in resolved if r.modulation is not None]
    trial_phi_schedule = None
    if modulations_with_onset:
        trial_phi_schedule = build_phi_schedule(
            phi_base=dynamics.phi,  # type: ignore[union-attr]
            modulations_with_onset=modulations_with_onset,
            params_n_burnin_samples=params.n_burnin_samples,
            params_n_sim_samples=params.n_sim_samples,
            sample_rate_hz=float(params.sample_rate_hz),
        )

    drives_with_onsets: list[tuple[float, Drive]] = [
        (r.onset_seconds, r.drive) for r in resolved if r.drive is not None
    ]
    trial_stim = build_drive_array(drives_with_onsets, regions, params) if drives_with_onsets else None

    result = simulate(
        dynamics,
        params,
        noise_fn,
        rng=noise_gen,
        stim=trial_stim,
        regions=regions,
        phi_schedule=trial_phi_schedule,
    )
    return result.signal


# ---------------------------------------------------------------------------
# Internal helpers — fixed-input path
# ---------------------------------------------------------------------------


@dataclass
class _FixedTrialArgs:
    """Picklable argument bundle for one fixed-input worker trial."""

    phi: FloatArray | None
    dynamics: Dynamics | None
    params: SimulationParams
    noise_fn: NoiseFn
    stim: FloatArray | None
    phi_schedule: list[FloatArray] | None
    region_codes: tuple[str, ...] | None
    seed_seq: np.random.SeedSequence

    def __post_init__(self) -> None:
        if self.phi is None and self.dynamics is None:
            raise ValueError("_FixedTrialArgs: exactly one of phi or dynamics must be provided")


def _fixed_worker(args: _FixedTrialArgs) -> FloatArray:
    """Entry point for each fixed-input worker process."""
    from diaxcondel.connectome.regions import RegionSet

    dynamics: Dynamics
    if args.phi is not None:
        dynamics = LinearVAR(phi=args.phi)
    else:
        assert args.dynamics is not None
        dynamics = args.dynamics
    regions = RegionSet.from_codes(list(args.region_codes)) if args.region_codes is not None else None
    return _simulate_one_fixed_trial(
        dynamics, args.params, args.noise_fn, args.stim, args.phi_schedule, regions, args.seed_seq
    )


def _simulate_one_fixed_trial(
    dynamics: Dynamics,
    params: SimulationParams,
    noise_fn: NoiseFn,
    stim: FloatArray | None,
    phi_schedule: list[FloatArray] | None,
    regions: RegionSet | None,
    seed_seq: np.random.SeedSequence,
) -> FloatArray:
    """Run a single fixed-input trial and return the post-burn-in signal (T, N)."""
    (noise_seed,) = seed_seq.spawn(1)
    noise_gen = np.random.default_rng(noise_seed)
    result = simulate(
        dynamics,
        params,
        noise_fn,
        rng=noise_gen,
        stim=stim,
        regions=regions,
        phi_schedule=phi_schedule,
    )
    return result.signal


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def _to_seed_sequence(rng: RNGLike) -> np.random.SeedSequence:
    """Convert any RNG-like to a SeedSequence for spawning."""
    if isinstance(rng, np.random.SeedSequence):
        return rng
    if isinstance(rng, int):
        return np.random.SeedSequence(rng)
    if isinstance(rng, np.random.Generator):
        state = rng.bit_generator.state
        raw = int(state["state"]["state"]) % (2**128)
        return np.random.SeedSequence(raw)
    return np.random.SeedSequence()
