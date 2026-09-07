"""
The component registry.

Every interchangeable piece of a model (connectome, delay kernel, noise
process, output transfer, stimulus drive, connectivity modulation) is
registered here under a *slot*. Anything that can enumerate slots and their
entries (the dashboard, the CLI, reports, tests) then works for components
that did not exist when it was written.

Registering an entry::

    @register("kernel", "hursh", label="Speed proportional to axon calibre")
    def hursh_kernel(speed_factor: float = 6.0) -> DelayKernel:
        # The docstring summary becomes the entry description, and the
        # documented parameters become its options.
        ...

and using it::

    from diaxcondel import catalog

    catalog.options("kernel")                     # every registered kernel
    catalog.build("kernel", "hursh", speed_factor=7.0)  # instantiate one
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Literal, TypeVar

from .param import ParamField, coerce_params, infer_fields

Slot = Literal[
    "distances",
    "connectivity",
    "diameter",
    "kernel",
    "local_delay",
    "noise",
    "transfer",
    "drive",
    "modulation",
]

#: Registered slots, in the order a model is assembled.
SLOTS: tuple[Slot, ...] = (
    "distances",
    "connectivity",
    "diameter",
    "kernel",
    "local_delay",
    "noise",
    "transfer",
    "drive",
    "modulation",
)


@dataclass(frozen=True, slots=True)
class SlotInfo:
    """
    What one slot of the model is, and what choosing it changes.

    Attributes
    ----------
    summary : str
        One line naming what the slot provides. Used wherever space is
        short: command-line listings, menu headers.
    detail : str
        What the choice does to the model, for a reader deciding between
        entries. Used as the explanatory text above a chooser.
    """

    summary: str
    detail: str


#: What each slot contributes to a model. The single source for every listing
#: and every dashboard caption, so the two cannot drift apart.
SLOT_INFO: dict[Slot, SlotInfo] = {
    "distances": SlotInfo(
        "The regions and how far apart they are (cm), which sets the delays.",
        "Fixes the regions and how far apart they are. Distance sets delay, and delay is what "
        "creates rhythms, so this choice shapes the spectrum more than any other.",
    ),
    "connectivity": SlotInfo(
        "How strongly each region drives each other region.",
        "Fixes how strongly each region drives each other one. Stronger coupling means sharper "
        "spectral peaks, up to the point where the network becomes unstable.",
    ),
    "diameter": SlotInfo(
        "How thick the axons of a bundle are; thicker axons conduct faster.",
        "How thick the axons of a bundle are. Thicker axons conduct faster, so the spread of "
        "calibres is what turns one connection into a spread of arrival times.",
    ),
    "kernel": SlotInfo(
        "Turns a connection's length into a distribution of transmission delays.",
        "Turns a connection's length into a distribution of arrival times. Faster conduction "
        "shortens delays and raises the frequencies the network favours.",
    ),
    "local_delay": SlotInfo(
        "Extra delay inside the receiving region (postsynaptic processing).",
        "Extra delay inside the receiving region: synaptic rise and decay, crossing layers, "
        "short fibres folded into the region. It shifts every arrival a little later.",
    ),
    "noise": SlotInfo(
        "The ongoing random input that keeps the network active.",
        "The ongoing random input that keeps the network active. Its spectrum is the baseline "
        "the network shapes, and how much of it regions share sets how much coherence they "
        "start with.",
    ),
    "transfer": SlotInfo(
        "How a region's activity is passed on: linear, or saturating.",
        "How a region passes activity on. Linear is the published model; a saturating transfer "
        "bounds activity, which lets the network run at or beyond the linear stability limit.",
    ),
    "drive": SlotInfo(
        "A stimulus waveform added to chosen regions during an event.",
        "A stimulus waveform added to chosen regions when an event fires.",
    ),
    "modulation": SlotInfo(
        "A temporary change of coupling strength during an event.",
        "A temporary change of coupling strength while an event lasts.",
    ),
}

#: One-line summary per slot, derived from :data:`SLOT_INFO`.
SLOT_DESCRIPTIONS: dict[Slot, str] = {slot: info.summary for slot, info in SLOT_INFO.items()}


def slot_info(slot: Slot) -> SlotInfo:
    """
    Return what a slot provides and what choosing it changes.

    Parameters
    ----------
    slot : str
        Slot name.

    Returns
    -------
    SlotInfo
        Summary and detail text.

    Raises
    ------
    KeyError
        If the slot is not registered.
    """
    if slot not in SLOT_INFO:
        raise KeyError(f"unknown slot {slot!r}; valid slots are {list(SLOTS)}")
    return SLOT_INFO[slot]


#: Extras that provide optional dependencies, for install hints.
_EXTRA_FOR_MODULE: dict[str, str] = {"siibra": "atlas", "matplotlib": "viz", "streamlit": "app"}

_REGISTRY: dict[str, dict[str, ComponentSpec]] = {slot: {} for slot in SLOTS}

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """
    A registered, introspectable model component.

    Attributes
    ----------
    slot : str
        Which part of the model this entry fills (see :data:`SLOTS`).
    name : str
        Unique identifier within the slot, used in specs and URLs.
    label : str
        Human-readable name for menus and figures.
    summary : str
        One-line description (defaults to the factory's docstring summary).
    factory : callable
        The function that builds the component.
    fields : tuple of ParamField
        User-facing options, derived from the factory signature.
    context : tuple of str
        Names of build-time arguments supplied by the caller rather than the
        user (e.g. ``regions``).
    requires : tuple of str
        Import names this entry needs (e.g. ``("siibra",)``).
    tags : tuple of str
        Free-form labels for filtering (``"reference"``, ``"synthetic"``, ...).
    reference : str
        Literature citation for the values or method.
    """

    slot: Slot
    name: str
    label: str
    summary: str
    factory: Callable[..., Any]
    fields: tuple[ParamField, ...] = ()
    context: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    reference: str = ""

    @property
    def key(self) -> str:
        """Return the ``"slot:name"`` identifier."""
        return f"{self.slot}:{self.name}"

    def defaults(self) -> dict[str, Any]:
        """Return the default value of every option."""
        return {f.name: f.default for f in self.fields}

    def coerce(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """
        Validate a parameter mapping against this entry, filling in defaults.

        Parameters
        ----------
        params : Mapping[str, Any], optional
            Partial parameter mapping.

        Returns
        -------
        dict
            Complete, validated parameters.
        """
        return coerce_params(self.fields, params or {}, where=self.key)

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        """Return the import names required by this entry that are not installed."""
        return tuple(module for module in self.requires if find_spec(module) is None)

    @property
    def available(self) -> bool:
        """Return ``True`` when every optional dependency of this entry is installed."""
        return not self.missing_requirements

    @property
    def install_hint(self) -> str:
        """Return an install command for the missing dependencies, or an empty string."""
        missing = self.missing_requirements
        if not missing:
            return ""
        extras = sorted({_EXTRA_FOR_MODULE.get(module, module) for module in missing})
        return f"poetry install --with {','.join(extras)}"

    def build(self, params: Mapping[str, Any] | None = None, **context: Any) -> Any:
        """
        Instantiate the component.

        Parameters
        ----------
        params : Mapping[str, Any], optional
            User-facing options; missing keys fall back to defaults.
        **context
            Build-time arguments named in :attr:`context` (e.g. ``regions``).

        Returns
        -------
        object
            Whatever the factory returns.

        Raises
        ------
        ImportError
            If an optional dependency of this entry is missing.
        TypeError
            If required context arguments are missing or unexpected ones are
            supplied.
        ValueError
            If a parameter is unknown or invalid.
        """
        if not self.available:
            raise ImportError(
                f"{self.key} requires {list(self.missing_requirements)}; install with `{self.install_hint}`"
            )
        missing = [name for name in self.context if name not in context]
        if missing:
            raise TypeError(f"{self.key}: missing build context {missing}")
        unexpected = sorted(set(context) - set(self.context))
        if unexpected:
            raise TypeError(f"{self.key}: unexpected build context {unexpected}")
        return self.factory(**context, **self.coerce(params))


def register(
    slot: Slot,
    name: str,
    *,
    label: str,
    summary: str = "",
    requires: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    reference: str = "",
) -> Callable[[F], F]:
    """
    Register a factory function as a catalog entry.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    name : str
        Identifier, unique within the slot. Stable: it is written into saved
        experiment specs.
    label : str
        Human-readable name for menus.
    summary : str, optional
        One-line description. Defaults to the first line of the factory
        docstring.
    requires : tuple of str
        Import names needed by the entry (e.g. ``("siibra",)``). Entries with
        missing dependencies stay listed but refuse to build, with an install
        hint.
    tags : tuple of str
        Free-form labels for filtering.
    reference : str
        Literature citation for the defaults or method.

    Returns
    -------
    callable
        Decorator returning the factory unchanged, so it stays directly
        callable and testable.

    Raises
    ------
    ValueError
        If the slot is unknown or the name already taken.
    """
    if slot not in _REGISTRY:
        raise ValueError(f"unknown slot {slot!r}; expected one of {list(SLOTS)}")

    def decorator(factory: F) -> F:
        if name in _REGISTRY[slot]:
            raise ValueError(f"{slot}:{name} is already registered")
        fields, context = infer_fields(factory)
        doc = (factory.__doc__ or "").strip().splitlines()
        spec = ComponentSpec(
            slot=slot,
            name=name,
            label=label,
            summary=summary or (doc[0].strip() if doc else ""),
            factory=factory,
            fields=fields,
            context=context,
            requires=requires,
            tags=tags,
            reference=reference,
        )
        _REGISTRY[slot][name] = spec
        return factory

    return decorator


def options(slot: Slot, *, available_only: bool = False) -> tuple[ComponentSpec, ...]:
    """
    List the entries registered for a slot, in registration order.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    available_only : bool
        Skip entries whose optional dependencies are missing.

    Returns
    -------
    tuple of ComponentSpec
        Matching entries.
    """
    if slot not in _REGISTRY:
        raise ValueError(f"unknown slot {slot!r}; expected one of {list(SLOTS)}")
    entries = tuple(_REGISTRY[slot].values())
    return tuple(e for e in entries if e.available) if available_only else entries


def get(slot: Slot, name: str) -> ComponentSpec:
    """
    Look up one entry.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    name : str
        Entry name.

    Returns
    -------
    ComponentSpec
        The registered entry.

    Raises
    ------
    KeyError
        If the slot has no such entry; the message lists the valid names.
    """
    if slot not in _REGISTRY:
        raise ValueError(f"unknown slot {slot!r}; expected one of {list(SLOTS)}")
    try:
        return _REGISTRY[slot][name]
    except KeyError as exc:
        raise KeyError(f"no {slot} entry named {name!r}; available: {sorted(_REGISTRY[slot])}") from exc


def build(slot: Slot, name: str, params: Mapping[str, Any] | None = None, **context: Any) -> Any:
    """
    Look up an entry and instantiate it.

    Parameters
    ----------
    slot : str
        One of :data:`SLOTS`.
    name : str
        Entry name.
    params : Mapping[str, Any], optional
        User-facing options.
    **context
        Build-time context arguments.

    Returns
    -------
    object
        The constructed component.
    """
    return get(slot, name).build(params, **context)


def iter_entries() -> Iterator[ComponentSpec]:
    """Iterate over every registered entry, slot by slot."""
    for slot in SLOTS:
        yield from _REGISTRY[slot].values()
