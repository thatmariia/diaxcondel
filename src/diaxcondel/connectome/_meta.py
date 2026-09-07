"""
Provenance metadata shared by connectome objects.

Regions, distance providers, connectivity providers, and bundles all carry an
optional free-form ``metadata`` mapping recording where their numbers came
from — atlas version, cohort, aggregation rule, random seed. The mapping is
frozen on construction so that a provider cannot change under a model that
was built from it, and thawed for pickling (a ``mappingproxy`` cannot be
pickled).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import Any

_METADATA_FIELD = "metadata"


def freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """
    Return a read-only copy of a metadata mapping.

    Parameters
    ----------
    metadata : Mapping[str, Any] or None
        Mapping to freeze.

    Returns
    -------
    Mapping[str, Any] or None
        A ``mappingproxy`` over a private copy, or ``None``.
    """
    if metadata is None:
        return None
    if isinstance(metadata, MappingProxyType):
        return metadata
    return MappingProxyType(dict(metadata))


def dataclass_state(obj: Any) -> dict[str, Any]:
    """
    Return picklable state for a frozen dataclass with a metadata field.

    Parameters
    ----------
    obj : object
        Frozen dataclass instance.

    Returns
    -------
    dict
        Field values, with the read-only metadata view converted to a dict.
    """
    state: dict[str, Any] = {}
    for spec in fields(obj):
        value = getattr(obj, spec.name)
        if spec.name == _METADATA_FIELD and value is not None:
            value = dict(value)
        state[spec.name] = value
    return state


def restore_dataclass_state(obj: Any, state: Mapping[str, Any]) -> None:
    """
    Restore a frozen dataclass from :func:`dataclass_state`.

    Parameters
    ----------
    obj : object
        Instance being unpickled.
    state : Mapping[str, Any]
        State produced by :func:`dataclass_state`.
    """
    for name, value in state.items():
        if name == _METADATA_FIELD:
            value = freeze_metadata(value)
        object.__setattr__(obj, name, value)


def metadata_of(obj: Any, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """
    Read the metadata of any connectome object, following relay wrappers.

    Parameters
    ----------
    obj : object
        Region, provider, or bundle. Objects wrapping another provider (e.g.
        :class:`~diaxcondel.connectome.distance.RelayedDistances`) forward to
        the object they wrap.
    default : Mapping[str, Any], optional
        Returned when no metadata is present.

    Returns
    -------
    Mapping[str, Any]
        The metadata mapping, or ``default`` (or an empty mapping).
    """
    metadata = getattr(obj, _METADATA_FIELD, None)
    if metadata is None and hasattr(obj, "base"):
        return metadata_of(obj.base, default)
    if metadata is None:
        return {} if default is None else default
    return metadata
