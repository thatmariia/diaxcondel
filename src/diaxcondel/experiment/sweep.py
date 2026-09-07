"""
Parameter sweeps.

A single simulation answers "what does this model do"; a sweep answers "what
does this parameter do", which is the question most of the science asks. This
module runs a specification over a grid of parameter values and returns tidy
rows ready for a table or a plot.

Parameters are addressed by path — ``"kernel.speed_factor"``,
``"connectivity.coupling"``, ``"simulation.sample_rate_hz"``,
``"self_weight"`` — the same strings the playground offers in its menus,
because both sides read them from the catalog:

>>> from diaxcondel.experiment import ExperimentSpec, sweep
>>> spec = ExperimentSpec()
>>> result = sweep(spec, {"kernel.speed_factor": [4.0, 6.0, 8.0]})   # doctest: +SKIP
>>> result.summary_table()                                          # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from diaxcondel import catalog
from diaxcondel.catalog.param import ParamField
from diaxcondel.model.params import SimulationParams

from .build import build_model
from .report import RegionSummary, compute_spectra, region_summaries
from .run import run_experiment
from .spec import ExperimentSpec

#: Spec fields that can be swept directly, without a slot prefix.
SCALAR_FIELDS: tuple[str, ...] = ("self_weight", "target_spectral_radius", "n_trials", "seed")


@dataclass(frozen=True, slots=True)
class SweepAxis:
    """
    One swept parameter and the values it takes.

    Attributes
    ----------
    path : str
        Parameter path, e.g. ``"kernel.speed_factor"``.
    values : tuple
        Values to run, in order.
    field : ParamField or None
        Catalog metadata for the parameter (units, bounds), when it has any.
    """

    path: str
    values: tuple[Any, ...]
    field: ParamField | None = None

    @property
    def label(self) -> str:
        """Return a display label including the unit, when known."""
        if self.field is not None and self.field.unit:
            return f"{self.path} ({self.field.unit})"
        return self.path


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """
    One combination of parameter values and what the model did there.

    Attributes
    ----------
    values : dict
        Parameter path to value for this point.
    spectral_radius : float
        Spectral radius of the linearised system; one is the stability
        boundary.
    summaries : tuple of RegionSummary
        Per-region spectral features.
    notes : tuple of str
        Warnings raised while building or running this point.
    """

    values: dict[str, Any]
    spectral_radius: float
    summaries: tuple[RegionSummary, ...]
    notes: tuple[str, ...] = ()

    @property
    def mean_peak_hz(self) -> float:
        """Mean band-peak frequency across regions."""
        return float(np.mean([item.peak_hz for item in self.summaries])) if self.summaries else float("nan")

    @property
    def mean_slope(self) -> float:
        """Mean aperiodic log-log slope across regions."""
        return float(np.mean([item.slope for item in self.summaries])) if self.summaries else float("nan")

    @property
    def mean_variance(self) -> float:
        """Mean time-domain variance across regions."""
        return float(np.mean([item.variance for item in self.summaries])) if self.summaries else float("nan")


@dataclass(frozen=True, slots=True)
class SweepResult:
    """
    Everything a sweep produced.

    Attributes
    ----------
    spec : ExperimentSpec
        The specification the sweep started from.
    axes : tuple of SweepAxis
        The swept parameters.
    points : tuple of SweepPoint
        One entry per grid combination, in run order.
    """

    spec: ExperimentSpec
    axes: tuple[SweepAxis, ...]
    points: tuple[SweepPoint, ...] = field(default_factory=tuple)

    def summary_table(self) -> list[dict[str, Any]]:
        """
        Return one row per grid point, averaged over regions.

        Returns
        -------
        list of dict
            Swept values plus ``spectral radius``, ``peak (Hz)``,
            ``1/f slope`` and ``variance``.
        """
        rows: list[dict[str, Any]] = []
        for point in self.points:
            rows.append(
                {
                    **point.values,
                    "spectral radius": point.spectral_radius,
                    "peak (Hz)": point.mean_peak_hz,
                    "1/f slope": point.mean_slope,
                    "variance": point.mean_variance,
                }
            )
        return rows

    def region_table(self) -> list[dict[str, Any]]:
        """
        Return one row per grid point and region.

        Returns
        -------
        list of dict
            Swept values plus the per-region features.
        """
        rows: list[dict[str, Any]] = []
        for point in self.points:
            for item in point.summaries:
                rows.append(
                    {
                        **point.values,
                        "region": item.code,
                        "peak (Hz)": item.peak_hz,
                        "peak power": item.peak_power,
                        "1/f slope": item.slope,
                        "variance": item.variance,
                    }
                )
        return rows


# ---------------------------------------------------------------------------
# Parameter addressing
# ---------------------------------------------------------------------------


def parameter_paths(spec: ExperimentSpec) -> dict[str, ParamField | None]:
    """
    List every parameter of a specification that a sweep can vary.

    Parameters
    ----------
    spec : ExperimentSpec
        Specification whose components define what is available.

    Returns
    -------
    dict
        Parameter path to its catalog field metadata (``None`` for the plain
        spec fields and simulation settings, which have no catalog entry).
    """
    paths: dict[str, ParamField | None] = {}
    for slot in spec.MODEL_SLOTS:
        ref = spec.component(slot)  # type: ignore[arg-type]
        entry = catalog.get(slot, ref.name)  # type: ignore[arg-type]
        for entry_field in entry.fields:
            if entry_field.kind in {"int", "float"}:
                paths[f"{slot}.{entry_field.name}"] = entry_field
    for name in SCALAR_FIELDS:
        paths[name] = None
    for name in ("sample_rate_hz", "sim_seconds", "burnin_seconds", "n_lags"):
        paths[f"simulation.{name}"] = None
    return paths


def get_parameter(spec: ExperimentSpec, path: str) -> Any:
    """
    Read one parameter of a specification by path.

    Parameters
    ----------
    spec : ExperimentSpec
        Specification to read.
    path : str
        ``"<slot>.<param>"``, ``"simulation.<field>"``, or a spec field name.

    Returns
    -------
    object
        The current value.

    Raises
    ------
    KeyError
        If the path does not exist.
    """
    if "." not in path:
        if path not in SCALAR_FIELDS:
            raise KeyError(f"unknown parameter {path!r}; try one of {sorted(parameter_paths(spec))}")
        return getattr(spec, path)
    head, _, tail = path.partition(".")
    if head == "simulation":
        return getattr(spec.simulation, tail)
    ref = spec.component(head)  # type: ignore[arg-type]
    if tail not in ref.params:
        raise KeyError(f"{head!r} has no parameter {tail!r}; available: {sorted(ref.params)}")
    return ref.params[tail]


def set_parameter(spec: ExperimentSpec, path: str, value: Any) -> ExperimentSpec:
    """
    Return a copy of a specification with one parameter changed.

    Parameters
    ----------
    spec : ExperimentSpec
        Specification to update.
    path : str
        ``"<slot>.<param>"``, ``"simulation.<field>"``, or a spec field name.
    value : object
        New value; validated by the catalog or by the spec model.

    Returns
    -------
    ExperimentSpec
        Updated copy.

    Raises
    ------
    KeyError
        If the path does not exist.
    """
    if "." not in path:
        if path not in SCALAR_FIELDS:
            raise KeyError(f"unknown parameter {path!r}; try one of {sorted(parameter_paths(spec))}")
        return spec.model_copy(update={path: value}, deep=True)
    head, _, tail = path.partition(".")
    if head == "simulation":
        merged = {**spec.simulation.model_dump(), tail: value}
        return spec.model_copy(update={"simulation": SimulationParams(**merged)}, deep=True)
    ref = spec.component(head)  # type: ignore[arg-type]
    if tail not in ref.params:
        raise KeyError(f"{head!r} has no parameter {tail!r}; available: {sorted(ref.params)}")
    return spec.with_component(head, ref.name, **{tail: value})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Running a sweep
# ---------------------------------------------------------------------------


def sweep(
    spec: ExperimentSpec,
    axes: Mapping[str, Sequence[Any]],
    *,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    segment_seconds: float = 2.0,
    alpha_band_hz: tuple[float, float] | None = None,
) -> SweepResult:
    """
    Run a specification over a grid of parameter values.

    Parameters
    ----------
    spec : ExperimentSpec
        Base specification; unswept parameters keep their values.
    axes : Mapping[str, Sequence]
        Parameter path to the values it should take. Several paths give the
        full grid of their combinations.
    progress : callable, optional
        Called as ``progress(index, total, values)`` before each point, for
        progress bars.
    segment_seconds : float
        Welch segment length used when summarising each point.
    alpha_band_hz : tuple of float, optional
        Band searched for the spectral peak; defaults to the alpha band.

    Returns
    -------
    SweepResult
        One point per grid combination, with per-region features.

    Raises
    ------
    KeyError
        If a path does not exist in the specification.
    ValueError
        If ``axes`` is empty or an axis has no values.
    """
    if not axes:
        raise ValueError("a sweep needs at least one parameter axis")

    available = parameter_paths(spec)
    swept: list[SweepAxis] = []
    for path, values in axes.items():
        if path not in available:
            raise KeyError(f"unknown parameter {path!r}; available: {sorted(available)}")
        listed = tuple(values)
        if not listed:
            raise ValueError(f"axis {path!r} has no values")
        swept.append(SweepAxis(path=path, values=listed, field=available[path]))

    combinations = list(product(*[axis.values for axis in swept]))
    points: list[SweepPoint] = []
    for index, combination in enumerate(combinations):
        values = {axis.path: value for axis, value in zip(swept, combination, strict=True)}
        if progress is not None:
            progress(index, len(combinations), values)
        point_spec = spec
        for path, value in values.items():
            point_spec = set_parameter(point_spec, path, value)
        built = build_model(point_spec)
        run = run_experiment(point_spec, built=built)
        spectra = compute_spectra(run, segment_seconds=segment_seconds)
        summaries = (
            region_summaries(run, spectra, alpha_band_hz=alpha_band_hz)
            if alpha_band_hz is not None
            else region_summaries(run, spectra)
        )
        points.append(
            SweepPoint(
                values=values,
                spectral_radius=built.spectral_radius(),
                summaries=summaries,
                notes=run.all_notes,
            )
        )

    return SweepResult(spec=spec, axes=tuple(swept), points=tuple(points))
