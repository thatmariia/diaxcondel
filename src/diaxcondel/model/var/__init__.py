"""
VAR-family dynamics implementations.

This subpackage holds concrete :class:`diaxcondel.model.dynamics.Dynamics`
implementations. Add new variants here (e.g. PSP-convolved, mesoscopically
coupled) so that the top-level :mod:`diaxcondel.model` namespace stays
focused on abstract concepts.
"""

from __future__ import annotations

from .linear import LinearVAR
from .nonlinear import NonlinearVAR

__all__ = ["LinearVAR", "NonlinearVAR"]
