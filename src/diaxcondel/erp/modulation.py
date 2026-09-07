"""
Connectivity modulation for ERP simulation.

A :class:`ConnectivityModulation` temporarily changes the lag-coefficient
tensor ``Phi`` during a trial.  This models stimulus-related changes in
effective coupling or routing without rebuilding the full lag tensor from
scratch.

The modulation operates on the already-constructed ``Phi`` tensor (shape
``(P, N, N)``) via element-wise multiplication:

    Phi_modulated[p, i, j] = Phi_base[p, i, j] * factor[i, j]

where ``factor`` is a time-varying ``(N, N)`` matrix.

Scientific note
---------------
In the original linear zero-mean model, connectivity modulation *alone* does
not produce a signed time-locked ERP mean.  Zero-mean symmetric noise times
a modulated (but still linear) system still has zero expectation.  Modulation
affects variance, power, coherence, and oscillatory ringing — not the signed
average.  To obtain a signed ERP, pair connectivity modulation with an
additive :class:`~diaxcondel.simulate.stimulus.Drive`.

Instability policy
------------------
A connectivity modulation can transiently push the effective Phi outside the
stationarity region.  Three policies are available:

``"warn"`` (default)
    Issue a :class:`UserWarning` and continue.  Suitable for exploratory ERP
    work where brief transient near-instability is acceptable.
``"raise"``
    Raise a :class:`ValueError` immediately.
``"ignore"``
    Proceed silently.

Composition of overlapping modulations
---------------------------------------
When multiple modulations are active at the same time step, their factors are
multiplied element-wise.  Additive factor semantics are not supported; if you
need them, compose the factors yourself before constructing the modulation.
If a composition is ambiguous (e.g. conflicting policies), the strictest
policy wins (``"raise"`` > ``"warn"`` > ``"ignore"``).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from diaxcondel._typing import FloatArray
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.model.stability import is_stationary_from_phi

InstabilityPolicy = Literal["warn", "raise", "ignore"]
_POLICY_ORDER: dict[str, int] = {"raise": 2, "warn": 1, "ignore": 0}


def _check_instability(phi_mod: FloatArray, policy: InstabilityPolicy) -> None:
    """Check stationarity of a modulated Phi and apply policy."""
    if policy == "ignore":
        return
    if not is_stationary_from_phi(phi_mod):
        msg = "Connectivity modulation produces a transiently unstable VAR system (spectral radius >= 1)"
        if policy == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)


@dataclass(frozen=True, slots=True)
class StepModulation:
    """
    Multiplicative step-envelope modulation of the Phi tensor.

    During the interval ``[onset_seconds, onset_seconds + duration_seconds)``
    the lag tensor is scaled element-wise:

        Phi_t[p, i, j] = Phi_base[p, i, j] * factor[i, j]

    Outside this window the base Phi is used unchanged.

    Parameters
    ----------
    factor : FloatArray of shape (N, N)
        Per-edge multiplicative factor.  ``factor[i, j]`` scales the coupling
        from region ``j`` to region ``i``.  Diagonal entries are allowed but
        should normally be ``1.0`` because the base model has no
        self-connections.
    duration_seconds : float
        Modulation duration.  Must be positive.
    instability_policy : {"warn", "raise", "ignore"}
        What to do if the modulated system is transiently non-stationary.
        Default ``"warn"``.

    Notes
    -----
    A global gain change is achieved by setting
    ``factor = gain * np.ones((N, N))``.
    """

    factor: FloatArray
    duration_seconds: float
    instability_policy: InstabilityPolicy = "warn"

    def __post_init__(self) -> None:
        f = np.asarray(self.factor, dtype=float)
        if f.ndim != 2 or f.shape[0] != f.shape[1]:
            raise ValueError(f"factor must be a square (N, N) matrix; got shape {f.shape}")
        if not np.all(np.isfinite(f)):
            raise ValueError("factor must contain only finite values")
        if f.shape[0] == 0:
            raise ValueError("factor must be at least 1×1")
        object.__setattr__(self, "factor", f)
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.instability_policy not in _POLICY_ORDER:
            valid = list(_POLICY_ORDER)
            raise ValueError(f"instability_policy must be one of {valid}; got {self.instability_policy!r}")

    def apply(self, phi_base: FloatArray) -> FloatArray:
        """
        Return a modulated copy of ``phi_base``.

        Parameters
        ----------
        phi_base : FloatArray of shape (P, N, N)
            Base lag tensor.

        Returns
        -------
        FloatArray of shape (P, N, N)
            Modulated tensor.  The base tensor is not mutated.
        """
        n = self.factor.shape[0]
        if phi_base.shape[1] != n or phi_base.shape[2] != n:
            raise ValueError(f"factor shape ({n}, {n}) is incompatible with phi shape {phi_base.shape}")
        phi_mod = phi_base * self.factor[np.newaxis, :, :]
        _check_instability(phi_mod, self.instability_policy)
        return phi_mod


def region_gain_factor(
    regions: RegionSet,
    targets: Sequence[str] = (),
    *,
    gain: float = 1.5,
    direction: Literal["incoming", "outgoing", "both"] = "incoming",
) -> FloatArray:
    """
    Build a per-edge gain matrix that scales the connections of some regions.

    Parameters
    ----------
    regions : RegionSet
        Region ordering of the model.
    targets : sequence of str
        Region codes whose connections are scaled. Empty means *every*
        region, i.e. a global gain change.
    gain : float
        Multiplicative factor applied to the selected edges. ``1.0`` leaves
        coupling unchanged, values above one strengthen it.
    direction : {"incoming", "outgoing", "both"}
        Which edges of a target region are scaled: connections arriving at it
        (its row), leaving it (its column), or both.

    Returns
    -------
    FloatArray of shape (N, N)
        Factor matrix suitable for :class:`StepModulation`.

    Raises
    ------
    KeyError
        If a target code is not part of ``regions``.
    ValueError
        If ``gain`` is not finite and non-negative.
    """
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and non-negative")
    n = len(regions)
    factor = np.ones((n, n), dtype=float)
    indices = [regions.index(code) for code in targets] if targets else list(range(n))
    for i in indices:
        if direction in ("incoming", "both"):
            factor[i, :] = gain
        if direction in ("outgoing", "both"):
            factor[:, i] = gain
    return factor


def compose_modulations(
    modulations: list[StepModulation],
    phi_base: FloatArray,
) -> FloatArray:
    """
    Compose multiple active modulations by multiplying their factors.

    Parameters
    ----------
    modulations : list of StepModulation
        All modulations active at the current time step.
    phi_base : FloatArray of shape (P, N, N)
        Base lag tensor before any modulation.

    Returns
    -------
    FloatArray of shape (P, N, N)
        Phi with all factors multiplied in.

    Notes
    -----
    The strictest instability policy among all active modulations is applied
    to the *composed* result.
    """
    if not modulations:
        return phi_base
    combined_factor = np.ones_like(modulations[0].factor)
    strictest_policy: InstabilityPolicy = "ignore"
    for mod in modulations:
        combined_factor = combined_factor * mod.factor
        if _POLICY_ORDER[mod.instability_policy] > _POLICY_ORDER[strictest_policy]:
            strictest_policy = mod.instability_policy
    phi_mod = phi_base * combined_factor[np.newaxis, :, :]
    _check_instability(phi_mod, strictest_policy)
    return phi_mod


def build_phi_schedule(
    phi_base: FloatArray,
    modulations_with_onset: list[tuple[float, StepModulation]],
    params_n_burnin_samples: int,
    params_n_sim_samples: int,
    sample_rate_hz: float,
) -> list[FloatArray]:
    """
    Pre-compute a per-sample Phi schedule for the full trial.

    Returns a list of length ``n_burnin + n_sim`` where each entry is either
    ``phi_base`` (no modulation active) or a composed modulated tensor.  Uses
    object identity so that unchanged samples share the same array.

    Parameters
    ----------
    phi_base : FloatArray of shape (P, N, N)
        Base lag tensor.
    modulations_with_onset : list of (onset_seconds, StepModulation)
        Each tuple gives the trial-relative onset (post-burn-in seconds) and
        the modulation to apply.
    params_n_burnin_samples, params_n_sim_samples : int
        Burn-in and simulation sample counts.
    sample_rate_hz : float
        Sampling rate.

    Returns
    -------
    list of FloatArray of shape (P, N, N)
        Per-sample Phi tensors.  Length = ``n_burnin + n_sim``.
    """
    n_total = params_n_burnin_samples + params_n_sim_samples
    schedule: list[FloatArray] = [phi_base] * n_total
    origin = params_n_burnin_samples

    windows: list[tuple[int, int, StepModulation]] = []
    for onset_s, mod in modulations_with_onset:
        start = origin + int(round(onset_s * sample_rate_hz))
        end = start + int(round(mod.duration_seconds * sample_rate_hz))
        start = max(0, min(start, n_total))
        end = max(0, min(end, n_total))
        if end > start:
            windows.append((start, end, mod))
    if not windows:
        return schedule

    # The schedule is piecewise constant: it can only change where a window
    # opens or closes. Build one tensor per stretch between those boundaries
    # and share it across the stretch, so a modulated interval costs one
    # array and one stability check rather than one of each per sample.
    boundaries = sorted({0, n_total, *(edge for start, end, _ in windows for edge in (start, end))})
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        active = [mod for start, end, mod in windows if start <= left < end]
        if not active:
            continue
        phi_t = active[0].apply(phi_base) if len(active) == 1 else compose_modulations(active, phi_base)
        for t in range(left, right):
            schedule[t] = phi_t
    return schedule
