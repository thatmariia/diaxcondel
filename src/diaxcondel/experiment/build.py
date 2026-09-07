"""
Turning a specification into a runnable model.

:func:`build_model` resolves every catalog reference in an
:class:`~diaxcondel.experiment.spec.ExperimentSpec`, assembles the lag tensor,
and returns the dynamics together with the connectome it came from. Two things
happen automatically here, because getting them wrong is easy and silent:

* a relayed connectome selects the two-leg delay kernel matching the chosen
  direct kernel, so the delay distribution stays consistent;
* warnings raised during construction (most importantly the stationarity
  shrink, which changes effective coupling) are captured and returned as
  :attr:`BuiltModel.notes` instead of scrolling past.

:func:`estimate_cost` gives a rough idea of memory and runtime *before*
anything is simulated, which is what stops a 300-region atlas model from
quietly locking up a dashboard.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from diaxcondel import catalog
from diaxcondel._typing import FloatArray
from diaxcondel.connectome.bundle import Connectome
from diaxcondel.connectome.regions import RegionSet
from diaxcondel.kernels.base import DelayKernel
from diaxcondel.kernels.local import LocalDelayKernel, NoLocalDelay
from diaxcondel.model.build import build_var
from diaxcondel.model.dynamics import Dynamics
from diaxcondel.model.params import SimulationParams
from diaxcondel.model.stability import DENSE_EIGENVALUE_LIMIT, spectral_radius_from_phi
from diaxcondel.model.var import LinearVAR

from .spec import ExperimentSpec

# Rough machine constants for the cost estimate. They are deliberately
# order-of-magnitude: the point is to separate "instant" from "go for coffee".
_SIM_FLOPS_PER_SECOND = 4.0e9
_SIM_STEP_OVERHEAD_S = 3.0e-6
_DENSE_EIG_SECONDS_PER_CUBE = 5.0e-10
_CONTOUR_SECONDS_PER_FLOP = 2.7e-9
_CONTOUR_POINTS = 2048
# The stationarity check runs once per shrink iteration; a few are typical.
_TYPICAL_STATIONARITY_CHECKS = 4

#: Above this estimated runtime (seconds) a model counts as heavy.
HEAVY_SECONDS = 30.0

#: Above this lag-tensor size (bytes) a model counts as heavy.
HEAVY_PHI_BYTES = 512 * 1024**2


@dataclass(frozen=True, slots=True)
class ModelCost:
    """
    Rough resource estimate for building and simulating a model.

    Attributes
    ----------
    n_regions, n_lags : int
        Model dimensions.
    n_samples : int
        Samples per trial, including burn-in.
    n_trials : int
        Number of trials.
    phi_bytes : int
        Memory for the lag tensor (the simulation needs about twice this,
        because the engine keeps a flattened copy).
    build_seconds, simulate_seconds : float
        Estimated stationarity-check and simulation times.
    notes : tuple of str
        Human-readable warnings about the estimate.
    """

    n_regions: int
    n_lags: int
    n_samples: int
    n_trials: int
    phi_bytes: int
    build_seconds: float
    simulate_seconds: float
    notes: tuple[str, ...] = ()

    @property
    def total_seconds(self) -> float:
        """Estimated total time for building plus simulating."""
        return self.build_seconds + self.simulate_seconds

    @property
    def phi_megabytes(self) -> float:
        """Lag-tensor size in mebibytes."""
        return self.phi_bytes / 1024**2

    @property
    def is_heavy(self) -> bool:
        """Whether this configuration warrants a confirmation before running."""
        return self.total_seconds > HEAVY_SECONDS or self.phi_bytes > HEAVY_PHI_BYTES


def estimate_cost(
    n_regions: int,
    params: SimulationParams,
    *,
    n_trials: int = 1,
    enforce_stationarity: bool = True,
) -> ModelCost:
    """
    Estimate memory and runtime for a model of a given size.

    Parameters
    ----------
    n_regions : int
        Number of regions ``N``.
    params : SimulationParams
        Simulation settings; ``n_lags`` and the sample counts drive the cost.
    n_trials : int
        Number of trials to simulate.
    enforce_stationarity : bool
        Whether the stationarity check runs (it dominates build time, and is
        repeated once per shrink iteration).

    Returns
    -------
    ModelCost
        Rough estimate, accurate to roughly a factor of two. Use it to warn,
        not to schedule.
    """
    n = int(n_regions)
    p = int(params.n_lags)
    n_samples = params.n_burnin_samples + params.n_sim_samples
    phi_bytes = 8 * n * n * p

    step_flops = 2.0 * n * n * p
    simulate_seconds = n_trials * n_samples * (step_flops / _SIM_FLOPS_PER_SECOND + _SIM_STEP_OVERHEAD_S)

    if not enforce_stationarity:
        build_seconds = 0.0
    elif n * p <= DENSE_EIGENVALUE_LIMIT:
        build_seconds = _TYPICAL_STATIONARITY_CHECKS * _DENSE_EIG_SECONDS_PER_CUBE * (n * p) ** 3
    else:
        build_seconds = _TYPICAL_STATIONARITY_CHECKS * _CONTOUR_SECONDS_PER_FLOP * _CONTOUR_POINTS * (p * n**2 + n**3)

    notes: list[str] = []
    if phi_bytes > HEAVY_PHI_BYTES:
        notes.append(f"the lag tensor alone needs {phi_bytes / 1024**2:.0f} MiB; consider fewer regions or lags")
    if simulate_seconds > HEAVY_SECONDS:
        notes.append(
            f"simulation is estimated at {simulate_seconds:.0f} s; "
            "lower the sample rate, shorten the run, or coarsen the parcellation"
        )
    if build_seconds > HEAVY_SECONDS / 3:
        notes.append(
            f"the stationarity check is estimated at {build_seconds:.0f} s; "
            "it runs once per configuration and is then cached"
        )
    return ModelCost(
        n_regions=n,
        n_lags=p,
        n_samples=n_samples,
        n_trials=n_trials,
        phi_bytes=phi_bytes,
        build_seconds=build_seconds,
        simulate_seconds=simulate_seconds,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class BuiltModel:
    """
    A model assembled from a specification, with its provenance.

    Attributes
    ----------
    spec : ExperimentSpec
        The specification this model was built from.
    connectome : Connectome
        Regions, distances, and weights actually used (already relayed, if
        the spec asked for a relay).
    kernel : DelayKernel
        Delay kernel actually used (the two-leg variant for relayed models).
    local_kernel : LocalDelayKernel or None
        Receiver-side delay convolved into the lag tensor, if any.
    dynamics : Dynamics
        The VAR system: :class:`~diaxcondel.model.var.linear.LinearVAR` for a
        linear transfer, :class:`~diaxcondel.model.var.nonlinear.NonlinearVAR`
        otherwise.
    params : SimulationParams
        Simulation settings.
    notes : tuple of str
        Warnings captured during construction, e.g. stationarity shrinkage.
    """

    spec: ExperimentSpec
    connectome: Connectome
    kernel: DelayKernel
    dynamics: Dynamics
    params: SimulationParams
    local_kernel: LocalDelayKernel | None = None
    notes: tuple[str, ...] = ()

    @property
    def regions(self) -> RegionSet:
        """Region set of the model."""
        return self.connectome.regions

    @property
    def n_regions(self) -> int:
        """Number of regions."""
        return self.connectome.n_regions

    @property
    def n_lags(self) -> int:
        """Lag depth ``P``."""
        return int(self.dynamics.n_lags)

    @property
    def phi(self) -> FloatArray:
        """
        Lag-coefficient tensor of the built dynamics.

        Raises
        ------
        AttributeError
            If the dynamics do not expose a lag tensor. Every model this
            package builds does; a custom :class:`Dynamics` need not.
        """
        phi = getattr(self.dynamics, "phi", None)
        if phi is None:
            raise AttributeError(f"{type(self.dynamics).__name__} exposes no lag tensor")
        return np.asarray(phi)

    @property
    def is_linear(self) -> bool:
        """Whether a closed-form spectrum is available."""
        return self.dynamics.transfer(np.array([1.0]), self.params.dt_s) is not None

    def spectral_radius(self) -> float:
        """
        Return the spectral radius of the linearised system.

        Values below one mean the linear model settles; approaching one
        sharpens every resonance; above one the linear model diverges and
        only a saturating transfer keeps it bounded.

        Returns
        -------
        float
            Largest absolute eigenvalue of the companion matrix.
        """
        return spectral_radius_from_phi(self.phi)

    def cost(self) -> ModelCost:
        """Return the cost estimate for simulating this model."""
        return estimate_cost(
            self.n_regions,
            self.params,
            n_trials=self.spec.n_trials,
            enforce_stationarity=self.spec.enforce_stationarity,
        )


def build_model(spec: ExperimentSpec) -> BuiltModel:
    """
    Build the dynamics described by a specification.

    Parameters
    ----------
    spec : ExperimentSpec
        Validated experiment description.

    Returns
    -------
    BuiltModel
        Dynamics, connectome, kernel, and any construction warnings.

    Raises
    ------
    ValueError
        If the distance and connectivity sources describe different regions,
        ``spec.relay_code`` names a region that does not exist, or the chosen
        kernel has no two-leg variant for relayed paths.
    ImportError
        If a chosen entry needs an optional dependency that is not installed.
    """
    distances = spec.distances.build("distances")
    weights = spec.connectivity.build("connectivity", regions=distances.regions, distances=distances)
    connectome = Connectome.from_sources(distances, weights)

    if spec.include_regions:
        unknown = [code for code in spec.include_regions if code not in connectome.codes]
        if unknown:
            raise ValueError(
                f"include_regions names regions this connectome does not have: {unknown[:5]}; "
                f"available codes start with {list(connectome.codes)[:5]}"
            )
        connectome = connectome.select(spec.include_regions)

    kernel_entry = catalog.get("kernel", spec.kernel.name)
    kernel_context: dict[str, Any] = {}
    if "diameter" in kernel_entry.context:
        kernel_context["diameter"] = spec.diameter.build("diameter")
    kernel: DelayKernel = spec.kernel.build("kernel", **kernel_context)

    if spec.relay_code:
        if spec.relay_code not in connectome.codes:
            raise ValueError(
                f"relay_code {spec.relay_code!r} is not a region of connectome {connectome.name!r}; "
                f"available codes: {list(connectome.codes)}"
            )
        relay_variant = getattr(kernel, "relay_kernel", None)
        if relay_variant is None:
            raise ValueError(
                f"kernel {spec.kernel.name!r} has no two-leg variant, so it cannot be used with a relay region"
            )
        connectome = connectome.relayed(spec.relay_code)
        kernel = relay_variant()

    transfer_fn = spec.transfer.build("transfer")
    local_kernel = spec.local_delay.build("local_delay")
    if isinstance(local_kernel, NoLocalDelay):
        local_kernel = None  # no local delay at all, rather than a zero-width one

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dynamics = build_var(
            connectome.distances,
            connectome.weights,
            kernel,
            spec.simulation,
            transfer_fn=transfer_fn,
            local_kernel=local_kernel,
            self_weight=spec.self_weight,
            use_length_mixture=spec.use_length_mixture,
            target_spectral_radius=spec.target_spectral_radius,
            enforce_stationarity=spec.enforce_stationarity,
        )
    notes = tuple(str(entry.message) for entry in caught)

    if not spec.enforce_stationarity and isinstance(dynamics, LinearVAR):
        notes = (
            *notes,
            "stationarity was not enforced; a linear model above the stability boundary diverges",
        )

    return BuiltModel(
        spec=spec,
        connectome=connectome,
        kernel=kernel,
        local_kernel=local_kernel,
        dynamics=dynamics,
        params=spec.simulation,
        notes=notes,
    )
