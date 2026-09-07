"""
Parameter descriptions for catalog entries.

A catalog entry is an ordinary factory function. Everything the rest of the
package needs to know about its options — types, defaults, ranges, units,
descriptions — is *derived* from that function: its signature, its type
annotations, and its NumPy docstring. Nothing is written twice, so a factory
and its dashboard controls cannot drift apart.

Extra facts that Python types cannot express (a unit, a plausible range, a
preferred widget) travel in :class:`Param` annotations::

    def my_kernel(
        speed_factor: Annotated[float, Param(unit="m/s/um", minimum=1.0, maximum=12.0)] = 6.0,
    ) -> DelayKernel:
        ...

Each option's description comes from the entry in the factory's own NumPy
docstring, so documentation and dashboard tooltips are the same text.

Parameters *without* a default are treated as **context**: they are supplied
by the builder (for example the region set a drive must target), never by the
user, and are not exposed as options.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from ._docstring import parse_parameter_docs

FieldKind = Literal["bool", "int", "float", "str", "choice", "str_list"]
WidgetHint = Literal[
    "auto",
    "slider",
    "number",
    "select",
    "multiselect",
    "text",
    "regions",
    "matrix",
    "vector",
    "hidden",
]


@dataclass(frozen=True, slots=True)
class Param:
    """
    Annotation metadata for a catalog-entry parameter.

    Parameters
    ----------
    unit : str, optional
        Physical unit, shown next to the control and in reports (``"cm"``,
        ``"Hz"``, ``"um"``, ...).
    minimum, maximum : float, optional
        Inclusive bounds. Supplying both makes the dashboard render a slider.
    step : float, optional
        Control increment.
    choices : sequence, optional
        Allowed values for string or numeric parameters (``Literal`` types
        provide these automatically).
    open_choices : bool
        When ``True``, ``choices`` are suggestions rather than a closed set:
        other values validate, and the dashboard offers free text.
    widget : str
        Preferred control: ``"regions"`` marks a list of region codes, which
        the dashboard renders against the model's current regions;
        ``"matrix"`` and ``"vector"`` mark text holding a numeric matrix or
        waveform, which the dashboard renders as an editable table; and
        ``"hidden"`` marks an option the page renders with a control of its
        own, so the generic form leaves it alone.
    description : str, optional
        Overrides the description parsed from the factory docstring.
    advanced : bool
        Hide behind an "advanced" expander in the dashboard.
    """

    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: Sequence[Any] | None = None
    open_choices: bool = False
    widget: WidgetHint = "auto"
    description: str | None = None
    advanced: bool = False


@dataclass(frozen=True, slots=True)
class ParamField:
    """
    A resolved, user-facing option of a catalog entry.

    Attributes
    ----------
    name : str
        Keyword-argument name of the factory parameter.
    kind : {"bool", "int", "float", "str", "choice", "str_list"}
        Value type, inferred from the annotation.
    default : object
        Default value taken from the factory signature.
    optional : bool
        Whether ``None`` is a valid value (annotation ``X | None``).
    description : str
        Human-readable description, from the docstring or :class:`Param`.
    unit : str or None
        Physical unit, if any.
    minimum, maximum, step : float or None
        Numeric bounds and increment.
    choices : tuple or None
        Allowed values, if constrained.
    open_choices : bool
        Whether values outside ``choices`` are accepted.
    widget : str
        Preferred dashboard control.
    advanced : bool
        Whether the option is secondary.
    """

    name: str
    kind: FieldKind
    default: Any
    optional: bool = False
    description: str = ""
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None
    open_choices: bool = False
    widget: WidgetHint = "auto"
    advanced: bool = False

    @property
    def label(self) -> str:
        """Return a display label: the name in words, with its unit."""
        words = self.name.replace("_", " ")
        return f"{words} ({self.unit})" if self.unit else words

    def coerce(self, value: Any) -> Any:
        """
        Validate and convert a user-supplied value for this field.

        Parameters
        ----------
        value : object
            Raw value (e.g. from JSON, a dashboard control, or a script).

        Returns
        -------
        object
            Value converted to the field's type.

        Raises
        ------
        ValueError
            If the value is out of range, not among the allowed choices, or
            not convertible to the field's type.
        """
        if value is None:
            if self.optional:
                return None
            raise ValueError(f"{self.name!r} does not accept None")

        if self.kind == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
                raise ValueError(f"{self.name!r} expects a boolean; got {value!r}")
            return bool(value)

        if self.kind in {"int", "float"}:
            try:
                number = int(value) if self.kind == "int" else float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.name!r} expects a {self.kind}; got {value!r}") from exc
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"{self.name!r} must be >= {self.minimum}; got {number}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"{self.name!r} must be <= {self.maximum}; got {number}")
            if self.choices is not None and not self.open_choices and number not in self.choices:
                raise ValueError(f"{self.name!r} must be one of {list(self.choices)}; got {number}")
            return number

        if self.kind in {"str", "choice"}:
            text = str(value)
            if self.choices is not None and not self.open_choices and text not in {str(c) for c in self.choices}:
                raise ValueError(f"{self.name!r} must be one of {list(self.choices)}; got {text!r}")
            return text

        if self.kind == "str_list":
            if isinstance(value, str):
                raise ValueError(f"{self.name!r} expects a list of strings, not a single string")
            items = tuple(str(v) for v in value)
            if self.choices is not None and not self.open_choices:
                allowed = {str(c) for c in self.choices}
                unknown = [item for item in items if item not in allowed]
                if unknown:
                    raise ValueError(f"{self.name!r} contains unknown values {unknown}")
            return items

        raise ValueError(f"unsupported field kind {self.kind!r}")  # pragma: no cover - guarded by inference


def _unwrap(annotation: Any) -> tuple[Any, tuple[Param, ...], bool]:
    """Split an annotation into (base type, Param metadata, optional flag)."""
    metadata: tuple[Param, ...] = ()
    optional = False

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        metadata = tuple(m for m in args[1:] if isinstance(m, Param))

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        optional = len(members) != len(get_args(annotation))
        if len(members) != 1:
            raise TypeError(f"unions of several types are not supported in catalog entries: {annotation!r}")
        inner, inner_metadata, _ = _unwrap(members[0])
        return inner, metadata + inner_metadata, optional

    return annotation, metadata, optional


def _kind_and_choices(annotation: Any) -> tuple[FieldKind, tuple[Any, ...] | None]:
    """Map a base annotation onto a field kind and (for literals) its choices."""
    if get_origin(annotation) is Literal:
        return "choice", get_args(annotation)
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    if annotation is str:
        return "str", None
    origin = get_origin(annotation)
    if origin in (tuple, list, Sequence, set, frozenset):
        args = [a for a in get_args(annotation) if a is not Ellipsis]
        if all(a is str for a in args) and args:
            return "str_list", None
    raise TypeError(
        f"unsupported catalog parameter type {annotation!r}; "
        "use bool, int, float, str, Literal[...], tuple[str, ...], or an optional of those"
    )


def infer_fields(
    factory: Callable[..., Any],
) -> tuple[tuple[ParamField, ...], tuple[str, ...]]:
    """
    Derive the option fields and context parameters of a factory function.

    Parameters
    ----------
    factory : callable
        Catalog entry factory. Every parameter must be annotated; every
        parameter with a default becomes a user-facing option, and every
        parameter without one becomes a context parameter supplied by the
        builder.

    Returns
    -------
    fields : tuple of ParamField
        User-facing options, in signature order.
    context : tuple of str
        Names of parameters the caller must supply at build time.

    Raises
    ------
    TypeError
        If a parameter lacks an annotation, uses an unsupported type, or the
        factory takes ``*args``/``**kwargs``.
    """
    signature = inspect.signature(factory)
    hints = get_type_hints(factory, include_extras=True)
    docs = parse_parameter_docs(factory.__doc__)

    fields: list[ParamField] = []
    context: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"catalog factory {factory.__name__!r} must not use *args/**kwargs")
        if name not in hints:
            raise TypeError(f"catalog factory {factory.__name__!r} parameter {name!r} needs a type annotation")
        if parameter.default is inspect.Parameter.empty:
            context.append(name)
            continue

        base, metadata, optional = _unwrap(hints[name])
        kind, literal_choices = _kind_and_choices(base)
        meta = metadata[0] if metadata else Param()
        choices = tuple(meta.choices) if meta.choices is not None else literal_choices
        widget = meta.widget
        if widget == "auto" and kind == "str_list":
            widget = "multiselect"
        fields.append(
            ParamField(
                name=name,
                kind=kind,
                default=parameter.default,
                optional=optional,
                description=meta.description or docs.get(name, ""),
                unit=meta.unit,
                minimum=meta.minimum,
                maximum=meta.maximum,
                step=meta.step,
                choices=choices,
                open_choices=meta.open_choices,
                widget=widget,
                advanced=meta.advanced,
            )
        )

    return tuple(fields), tuple(context)


def coerce_params(fields: Sequence[ParamField], params: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """
    Validate a parameter mapping against fields, filling in defaults.

    Parameters
    ----------
    fields : sequence of ParamField
        Fields to validate against.
    params : Mapping[str, Any]
        User-supplied values; may be partial.
    where : str
        Identifier used in error messages (e.g. ``"kernel:hursh"``).

    Returns
    -------
    dict
        Complete, validated parameter mapping.

    Raises
    ------
    ValueError
        If a key is unknown or a value invalid.
    """
    by_name = {f.name: f for f in fields}
    unknown = sorted(set(params) - set(by_name))
    if unknown:
        raise ValueError(f"{where}: unknown parameter(s) {unknown}; available: {sorted(by_name)}")
    out: dict[str, Any] = {}
    for name, field_spec in by_name.items():
        value = params[name] if name in params else field_spec.default
        try:
            out[name] = field_spec.coerce(value)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    return out
