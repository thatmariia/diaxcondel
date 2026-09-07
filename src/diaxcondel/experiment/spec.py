"""
Declarative experiment specifications.

An :class:`ExperimentSpec` names one catalog entry per model component,
together with its parameters and the simulation settings. It is a plain
pydantic model, so it validates on construction, round-trips through JSON,
and can be stored next to results as an exact record of what was run.

The spec is the single description that the builders, the command line, and
the playground all consume:

>>> from diaxcondel.experiment import ExperimentSpec, run_experiment
>>> spec = ExperimentSpec()
>>> spec = spec.with_component("kernel", "hursh", speed_factor=8.0)
>>> result = run_experiment(spec)          # doctest: +SKIP

Component names and parameters are checked against the catalog at validation
time, so a typo fails immediately with the list of valid options rather than
halfway through a simulation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diaxcondel import catalog
from diaxcondel.catalog.registry import Slot
from diaxcondel.model.params import SimulationParams


class ComponentRef(BaseModel):
    """
    A reference to one catalog entry plus its parameter values.

    Parameters
    ----------
    name : str
        Entry name within its slot (e.g. ``"hursh"``).
    params : dict
        Parameter overrides; unspecified parameters use the entry defaults.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def validated(self, slot: Slot) -> ComponentRef:
        """
        Return a copy whose parameters are checked and completed for ``slot``.

        Parameters
        ----------
        slot : str
            Catalog slot this reference belongs to.

        Returns
        -------
        ComponentRef
            Reference with canonical, fully specified parameters.

        Raises
        ------
        ValueError
            If the slot has no entry with this name, or a parameter is
            unknown or invalid. Raised as ``ValueError`` (not ``KeyError``)
            so that pydantic reports it as a validation error rather than
            letting it escape.
        """
        try:
            entry = catalog.get(slot, self.name)
        except KeyError as exc:
            raise ValueError(str(exc.args[0]) if exc.args else str(exc)) from exc
        return ComponentRef(name=self.name, params=entry.coerce(self.params))

    def build(self, slot: Slot, **context: Any) -> Any:
        """
        Instantiate the referenced component.

        Parameters
        ----------
        slot : str
            Catalog slot.
        **context
            Build-time context (e.g. ``regions``).

        Returns
        -------
        object
            The constructed component.
        """
        return catalog.get(slot, self.name).build(self.params, **context)


class EventRef(BaseModel):
    """
    One stimulus event: a drive, a connectivity modulation, or both.

    Parameters
    ----------
    onset_seconds : float
        Onset relative to the start of the post-burn-in window.
    drive : ComponentRef, optional
        Additive stimulus waveform.
    modulation : ComponentRef, optional
        Time-limited coupling change.
    name : str
        Label used in plots and reports.
    """

    model_config = ConfigDict(extra="forbid")

    onset_seconds: float = Field(default=0.2, ge=0.0)
    drive: ComponentRef | None = None
    modulation: ComponentRef | None = None
    name: str = "event"

    @model_validator(mode="after")
    def _check_content(self) -> EventRef:
        if self.drive is None and self.modulation is None:
            raise ValueError(f"event {self.name!r}: provide a drive, a modulation, or both")
        return self


def _default_simulation() -> SimulationParams:
    """Return the reference simulation settings (1 kHz, 10 s, 300 lags)."""
    return SimulationParams(sample_rate_hz=1000, sim_seconds=10.0, burnin_seconds=1.0, n_lags=300)


class ExperimentSpec(BaseModel):
    """
    A complete, reproducible description of one model and one simulation.

    Parameters
    ----------
    distances : ComponentRef
        The regions and how far apart they are.
    connectivity : ComponentRef
        How strongly regions drive one another, over those same regions.
    diameter : ComponentRef
        Distribution of axon calibres within a bundle. Used by delay kernels
        that derive conduction speed from calibre, ignored by the others.
    kernel : ComponentRef
        Delay kernel turning connection length into transmission delays.
    local_delay : ComponentRef
        Receiver-side delay convolved into every incoming connection.
    noise : ComponentRef
        Ongoing stochastic drive.
    transfer : ComponentRef
        Element-wise output transfer; ``"identity"`` keeps the model linear.
    simulation : SimulationParams
        Sampling rate, duration, burn-in, and lag depth.
    include_regions : tuple of str
        Keep only these regions, by code, whatever the distance source is.
        Empty keeps all of them. Applied before the relay and before any
        stimulus targets are resolved.
    relay_code : str, optional
        Route every other connection through this region (e.g. the
        thalamus). The two-leg delay kernel is selected automatically.
    self_weight : float
        Recurrent self-excitation added to the diagonal of the weight
        matrix. Requires a local delay, since the feedback must take time.
    use_length_mixture : bool
        For merged regions, evaluate the delay kernel at every member
        connection's length and mix the results instead of using one
        representative length. Closer to the anatomy, and slower to build.
    target_spectral_radius : float, optional
        Scale every coupling by one factor so the model sits at this distance
        from criticality, leaving relative connectivity and all delays
        untouched. Sweeping it varies peak sharpness on its own. ``None``
        keeps the coupling the components specify.
    events : tuple of EventRef
        Stimulus events. Empty means resting state.
    n_trials : int
        Number of independent trials to simulate.
    seed : int
        Master seed; every stochastic step derives from it.
    enforce_stationarity : bool
        Shrink the coupling until the linearised system is stable. A
        saturating transfer keeps the model bounded without it.
    label : str
        Free-form label carried into reports.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    distances: ComponentRef = Field(default_factory=lambda: ComponentRef(name="steeghs_2025"))
    connectivity: ComponentRef = Field(default_factory=lambda: ComponentRef(name="steeghs_2025"))
    diameter: ComponentRef = Field(default_factory=lambda: ComponentRef(name="gev"))
    kernel: ComponentRef = Field(default_factory=lambda: ComponentRef(name="hursh"))
    local_delay: ComponentRef = Field(default_factory=lambda: ComponentRef(name="none"))
    noise: ComponentRef = Field(default_factory=lambda: ComponentRef(name="white"))
    transfer: ComponentRef = Field(default_factory=lambda: ComponentRef(name="identity"))
    simulation: SimulationParams = Field(default_factory=_default_simulation)
    include_regions: tuple[str, ...] = ()
    relay_code: str | None = None
    self_weight: float = Field(default=0.0, ge=0.0)
    use_length_mixture: bool = True
    target_spectral_radius: float | None = Field(default=None, gt=0.0)
    events: tuple[EventRef, ...] = ()
    n_trials: int = Field(default=1, ge=1)
    seed: int = 20260505
    enforce_stationarity: bool = True
    label: str = ""

    #: Slots configured once per model (the rest are configured per event).
    MODEL_SLOTS: ClassVar[tuple[str, ...]] = (
        "distances",
        "connectivity",
        "diameter",
        "kernel",
        "local_delay",
        "noise",
        "transfer",
    )

    @model_validator(mode="after")
    def _validate_components(self) -> ExperimentSpec:
        """Check every reference against the catalog and canonicalise params."""
        for slot in self.MODEL_SLOTS:
            object.__setattr__(self, slot, getattr(self, slot).validated(slot))
        if self.self_weight > 0 and self.local_delay.name == "none":
            raise ValueError(
                "self_weight needs a local delay: recurrent excitation cannot arrive instantaneously. "
                "Choose a local delay kernel, or set self_weight to 0."
            )
        events = tuple(
            EventRef(
                onset_seconds=event.onset_seconds,
                drive=event.drive.validated("drive") if event.drive else None,
                modulation=(event.modulation.validated("modulation") if event.modulation else None),
                name=event.name,
            )
            for event in self.events
        )
        object.__setattr__(self, "events", events)
        return self

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def component(self, slot: Slot) -> ComponentRef:
        """
        Return the reference filling a component slot.

        Parameters
        ----------
        slot : str
            One of :attr:`MODEL_SLOTS`.

        Returns
        -------
        ComponentRef
            The stored reference.

        Raises
        ------
        ValueError
            For slots that are not single-valued (``"drive"``,
            ``"modulation"``); read :attr:`events` instead.
        """
        if slot in self.MODEL_SLOTS:
            return getattr(self, slot)
        raise ValueError(f"{slot!r} is configured per event; see ExperimentSpec.events")

    def with_component(self, slot: Slot, name: str | None = None, /, **params: Any) -> ExperimentSpec:
        """
        Return a copy with one component replaced or reconfigured.

        Parameters
        ----------
        slot : str
            Slot to change; one of :attr:`MODEL_SLOTS`.
        name : str, optional
            Entry name. ``None`` keeps the current entry and only updates
            parameters.
        **params
            Parameter overrides, merged into the existing ones when the entry
            is unchanged.

        Returns
        -------
        ExperimentSpec
            Updated copy; the original is left untouched.
        """
        current = self.component(slot)
        if name is None or name == current.name:
            merged = {**current.params, **params}
            ref = ComponentRef(name=current.name, params=merged)
        else:
            ref = ComponentRef(name=name, params=dict(params))
        return self.model_copy(update={slot: ref.validated(slot)}, deep=True)

    def with_simulation(self, **params: Any) -> ExperimentSpec:
        """
        Return a copy with updated simulation settings.

        Parameters
        ----------
        **params
            Fields of :class:`~diaxcondel.model.params.SimulationParams`
            (``sample_rate_hz``, ``sim_seconds``, ``burnin_seconds``,
            ``n_lags``), or ``n_trials``, which belongs to the specification
            rather than to one simulation.

        Returns
        -------
        ExperimentSpec
            Updated copy.

        Raises
        ------
        ValueError
            If a keyword names no simulation setting. Silently ignoring one
            would mean running something other than what was asked for.
        """
        fields = set(SimulationParams.model_fields)
        unknown = [key for key in params if key not in fields and key != "n_trials"]
        if unknown:
            raise ValueError(f"unknown simulation setting(s) {unknown}; expected any of {sorted(fields)} or 'n_trials'")
        update: dict[str, Any] = {}
        if "n_trials" in params:
            update["n_trials"] = int(params.pop("n_trials"))
        merged = {**self.simulation.model_dump(), **params}
        update["simulation"] = SimulationParams(**merged)
        return self.model_copy(update=update, deep=True)

    def with_event(
        self,
        drive: str | ComponentRef | EventRef | None = None,
        *,
        onset_seconds: float = 0.2,
        modulation: str | ComponentRef | None = None,
        name: str = "",
        **params: Any,
    ) -> ExperimentSpec:
        """
        Return a copy with one more stimulus event.

        Parameters
        ----------
        drive : str, ComponentRef or EventRef, optional
            Drive entry name (with its options given as keyword arguments), a
            ready drive reference, or a complete :class:`EventRef`, in which
            case the other arguments are unused.
        onset_seconds : float
            When the event starts, relative to the analysis window.
        modulation : str or ComponentRef, optional
            Coupling modulation accompanying the drive.
        name : str
            Label for plots; defaults to ``"event <n>"``.
        **params
            Options for ``drive`` when it is given as a name.

        Returns
        -------
        ExperimentSpec
            Updated copy.
        """
        if isinstance(drive, EventRef):
            return self.model_copy(update={"events": (*self.events, drive)}, deep=True)
        drive_ref = ComponentRef(name=drive, params=dict(params)) if isinstance(drive, str) else drive
        modulation_ref = ComponentRef(name=modulation) if isinstance(modulation, str) else modulation
        event = EventRef(
            onset_seconds=onset_seconds,
            drive=drive_ref,
            modulation=modulation_ref,
            name=name or f"event {len(self.events) + 1}",
        )
        return self.model_copy(update={"events": (*self.events, event)}, deep=True)

    def with_regions(self, codes: Sequence[str]) -> ExperimentSpec:
        """
        Return a copy restricted to a subset of regions.

        Parameters
        ----------
        codes : sequence of str
            Region codes to keep; empty keeps all of them.

        Returns
        -------
        ExperimentSpec
            Updated copy.
        """
        return self.model_copy(update={"include_regions": tuple(codes)}, deep=True)

    def without_events(self) -> ExperimentSpec:
        """Return a copy with every stimulus event removed (resting state)."""
        return self.model_copy(update={"events": ()}, deep=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        Build a spec from a plain mapping (e.g. parsed JSON).

        Parameters
        ----------
        data : Mapping[str, Any]
            Spec fields.

        Returns
        -------
        ExperimentSpec
            Validated spec.
        """
        return cls.model_validate(dict(data))

    def to_json(self, *, indent: int | None = 2) -> str:
        """
        Serialise to JSON.

        Parameters
        ----------
        indent : int, optional
            Indentation for readability; ``None`` gives a compact string
            suitable as a cache key.

        Returns
        -------
        str
            JSON document describing the whole experiment.
        """
        return json.dumps(self.model_dump(mode="json"), indent=indent, sort_keys=True)

    def fingerprint(self) -> str:
        """Return a compact canonical JSON string, stable enough to key a cache."""
        return self.to_json(indent=None)


def save_spec(spec: ExperimentSpec, path: str | Path) -> Path:
    """
    Write a spec to a JSON file.

    Parameters
    ----------
    spec : ExperimentSpec
        Spec to store.
    path : str or Path
        Destination file.

    Returns
    -------
    Path
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.to_json())
    return destination


def load_spec(path: str | Path) -> ExperimentSpec:
    """
    Read a spec from a JSON file.

    Parameters
    ----------
    path : str or Path
        Source file.

    Returns
    -------
    ExperimentSpec
        Validated spec.
    """
    return ExperimentSpec.model_validate_json(Path(path).read_text())
