"""
Connectivity-weight providers.

A :class:`ConnectivityProvider` exposes an ``(N, N)`` matrix of dimensionless
inter-region coupling strengths over a fixed :class:`RegionSet`. By
convention entry ``[i, j]`` is the weight from region ``j`` onto region
``i`` (post-synaptic in row, pre-synaptic in column) — matching the VAR
form ``h_i(t) = sum_j gamma[i, j] * h_j(t - tau)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from diaxcondel._typing import FloatArray

from ._meta import dataclass_state, freeze_metadata, restore_dataclass_state
from .regions import RegionSet


@runtime_checkable
class ConnectivityProvider(Protocol):
    """Protocol for any source of inter-region connectivity weights."""

    @property
    def regions(self) -> RegionSet:
        """Return the region set whose order indexes the matrix."""
        ...

    def matrix(self) -> FloatArray:
        """
        Return the ``(N, N)`` weight matrix.

        Returns
        -------
        FloatArray of shape (N, N)
            Non-negative weights. Diagonal is conventionally zero
            (self-connections absorbed into the noise term in
            :class:`diaxcondel.model.linear_var.LinearVAR`).
        """
        ...


@dataclass(frozen=True, slots=True)
class ManualConnectivity:
    """
    Connectivity provider backed by an explicit numpy array.

    Parameters
    ----------
    regions : RegionSet
        Region ordering.
    weights : FloatArray of shape (N, N)
        Non-negative weight matrix with a zero diagonal. Recurrent
        self-excitation is not a property of the connectome here: it is added
        at build time together with the local delay it requires (see
        :func:`diaxcondel.model.build.build_lag_tensor`).
    name : str
        Short identifier for figures and reports.
    metadata : Mapping[str, Any], optional
        Free-form provenance (source dataset, normalisation, threshold, ...).
    """

    regions: RegionSet
    weights: FloatArray
    name: str = "connectivity"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        n = len(self.regions)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        weights = np.array(self.weights, dtype=float, copy=True)
        if weights.shape != (n, n):
            raise ValueError(f"weights shape {weights.shape} != ({n}, {n})")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if not np.allclose(np.diag(weights), 0.0):
            raise ValueError("weight diagonal must be zero (self-loops not modelled here)")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state (the read-only metadata view becomes a dict)."""
        return dataclass_state(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, re-freezing the metadata mapping."""
        restore_dataclass_state(self, state)

    def matrix(self) -> FloatArray:  # noqa: D102
        return self.weights


@dataclass(frozen=True, slots=True)
class FCConnectivity:
    """
    Connectivity weights derived from a functional connectivity matrix.

    Implements the power-law mapping:
    ``gamma(u, w) = nu * b(u, w) ** chi``,
    where ``b`` is the (non-negative) functional connectivity matrix.

    Parameters
    ----------
    regions : RegionSet
        Region ordering.
    fc : FloatArray of shape (N, N)
        Functional connectivity.
    nu : float
        Overall scale factor.
    chi : float
        Power-law exponent.
    """

    regions: RegionSet
    fc: FloatArray
    nu: float = 1.0
    chi: float = 1.5

    def __post_init__(self) -> None:  # noqa: D105
        n = len(self.regions)
        fc = np.array(self.fc, dtype=float, copy=True)
        if fc.shape != (n, n):
            raise ValueError(f"fc shape {fc.shape} != ({n}, {n})")
        if not np.all(np.isfinite(fc)):
            raise ValueError("fc must contain only finite values")
        if self.nu <= 0:
            raise ValueError("nu must be positive")
        if self.chi <= 0:
            raise ValueError("chi must be positive")
        fc.setflags(write=False)
        object.__setattr__(self, "fc", fc)

    def matrix(self) -> FloatArray:  # noqa: D102
        b = np.clip(self.fc, 0.0, None)
        w = self.nu * np.power(b, self.chi)
        np.fill_diagonal(w, 0.0)
        return w
