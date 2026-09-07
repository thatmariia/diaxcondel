"""
Built-in catalog entries.

Importing this package registers every built-in component. New entry modules
must be imported here; that single line is all it takes for an option to
appear in the catalog, the command line, and the playground.
"""

from __future__ import annotations

from . import (
    connectivity,
    diameters,
    distances,
    drives,
    kernels,
    local_delay,
    modulations,
    noise,
    transfers,
)

__all__ = [
    "connectivity",
    "diameters",
    "distances",
    "drives",
    "kernels",
    "local_delay",
    "modulations",
    "noise",
    "transfers",
]
