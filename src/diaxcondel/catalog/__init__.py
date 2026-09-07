"""
The component catalog: every interchangeable model piece, with its options.

The catalog answers three questions for any part of the model — what options
exist, what each one takes, and how to build it — without the caller knowing
which components exist:

>>> from diaxcondel import catalog
>>> [entry.name for entry in catalog.options("kernel")]
['hursh', 'gamma_delay', 'fixed_speed']
>>> calibres = catalog.build("diameter", "gev")
>>> kernel = catalog.build("kernel", "hursh", {"speed_factor": 6.0}, diameter=calibres)

Entries describe themselves (see :mod:`diaxcondel.catalog.param`), so the
dashboard, the CLI, and saved experiment specs all stay in step with the
package automatically.
"""

from __future__ import annotations

from . import entries as _entries  # noqa: F401  (import registers built-in entries)
from .param import Param, ParamField
from .registry import (
    SLOT_DESCRIPTIONS,
    SLOT_INFO,
    SLOTS,
    ComponentSpec,
    Slot,
    SlotInfo,
    build,
    get,
    iter_entries,
    options,
    register,
    slot_info,
)

__all__ = [
    "SLOTS",
    "SLOT_DESCRIPTIONS",
    "SLOT_INFO",
    "SlotInfo",
    "slot_info",
    "ComponentSpec",
    "Param",
    "ParamField",
    "Slot",
    "build",
    "get",
    "iter_entries",
    "options",
    "register",
]
