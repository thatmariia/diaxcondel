"""
Parameter packing and unpacking for inference.

A :class:`ParameterSpec` describes a flat parameter vector by listing
``(name, shape, transform)`` entries. It provides ``pack`` and ``unpack``
so that scientific code works with named dicts while inference code works
with a flat array.

Transforms support unconstrained sampling: a parameter constrained to be
positive can be specified with :class:`LogTransform`, and the inference
backend then samples in the unconstrained log-space, with the spec
handling the bijection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from diaxcondel._typing import FloatArray


class Transform(Protocol):
    """Bijection between an unconstrained real and a constrained value."""

    def forward(self, x: FloatArray) -> FloatArray:
        """Map unconstrained -> constrained."""
        ...

    def inverse(self, y: FloatArray) -> FloatArray:
        """Map constrained -> unconstrained."""
        ...


@dataclass(frozen=True, slots=True)
class IdentityTransform:
    """Identity bijection."""

    def forward(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return x

    def inverse(self, y: FloatArray) -> FloatArray:  # noqa: D102
        return y


@dataclass(frozen=True, slots=True)
class LogTransform:
    """Log bijection: unconstrained <-> positive reals."""

    def forward(self, x: FloatArray) -> FloatArray:  # noqa: D102
        return np.exp(x)

    def inverse(self, y: FloatArray) -> FloatArray:  # noqa: D102
        if np.any(y <= 0):
            raise ValueError("LogTransform.inverse requires positive constrained values")
        return np.log(y)


@dataclass(frozen=True, slots=True)
class ParameterEntry:
    """One entry in a parameter spec.

    Parameters
    ----------
    name : str
        Unique parameter name.
    shape : tuple of int
        Shape of the parameter as a numpy array. Use ``()`` for scalars.
    transform : Transform
        Bijection from unconstrained to constrained space.
    """

    name: str
    shape: tuple[int, ...]
    transform: Transform = IdentityTransform()

    @property
    def size(self) -> int:
        """Number of scalar entries occupied by this parameter."""
        if any(s < 0 for s in self.shape):
            raise ValueError("parameter shapes must be non-negative")
        out = 1
        for s in self.shape:
            out *= s
        return out


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Flat-vector view over a named parameter dictionary.

    Parameters
    ----------
    entries : tuple of ParameterEntry
        Parameter entries in the order they should appear in the flat vector.
    """

    entries: tuple[ParameterEntry, ...]

    def __post_init__(self) -> None:  # noqa: D105
        names = [entry.name for entry in self.entries]
        if len(set(names)) != len(names):
            duplicates = {name for name in names if names.count(name) > 1}
            raise ValueError(f"duplicate parameter names: {sorted(duplicates)}")
        for entry in self.entries:
            if not entry.name:
                raise ValueError("parameter names must be non-empty")
            if any(dim < 0 for dim in entry.shape):
                raise ValueError(f"parameter {entry.name!r} has a negative shape dimension")

    @property
    def size(self) -> int:
        """Total length of the flat unconstrained vector."""
        return sum(e.size for e in self.entries)

    def pack(self, values: Mapping[str, FloatArray | float]) -> FloatArray:
        """
        Convert a dict of constrained values to an unconstrained vector.

        Parameters
        ----------
        values : Mapping[str, ndarray | float]
            Constrained values keyed by name. Every entry in :attr:`entries`
            must be present.

        Returns
        -------
        FloatArray of shape (size,)
            Concatenated unconstrained vector.
        """
        chunks: list[FloatArray] = []
        for e in self.entries:
            if e.name not in values:
                raise KeyError(f"missing parameter {e.name!r}")
            v = np.asarray(values[e.name], dtype=float).reshape(e.shape)
            chunks.append(e.transform.inverse(v).ravel())
        if not chunks:
            return np.empty(0, dtype=float)
        return np.concatenate(chunks)

    def unpack(self, theta: FloatArray) -> dict[str, FloatArray]:
        """
        Convert an unconstrained vector to a dict of constrained values.

        Parameters
        ----------
        theta : FloatArray of shape (size,)
            Unconstrained vector as produced by :meth:`pack` (or a sampler).

        Returns
        -------
        dict[str, FloatArray]
            Constrained values keyed by parameter name.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.size,):
            raise ValueError(f"theta shape {theta.shape} != ({self.size},)")
        out: dict[str, FloatArray] = {}
        offset = 0
        for e in self.entries:
            chunk = theta[offset : offset + e.size].reshape(e.shape)
            out[e.name] = e.transform.forward(chunk)
            offset += e.size
        return out
