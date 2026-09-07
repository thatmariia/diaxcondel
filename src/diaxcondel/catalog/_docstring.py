"""
Minimal NumPy-docstring parameter extraction.

Catalog entries describe their options in the docstring they must write
anyway (the linter enforces one). Reading parameter descriptions from there
keeps a single source of truth: there is no second place to update when a
parameter changes meaning.
"""

from __future__ import annotations

import inspect
import re

_SECTION_UNDERLINE = re.compile(r"^-{3,}\s*$")
_PARAM_HEADER = re.compile(r"^(?P<names>[*\w][\w*, ]*?)\s*(?::\s*(?P<type>.+?))?\s*$")


def parse_parameter_docs(doc: str | None) -> dict[str, str]:
    """
    Extract ``{parameter_name: description}`` from a NumPy-style docstring.

    Parameters
    ----------
    doc : str or None
        Raw docstring.

    Returns
    -------
    dict of str to str
        One entry per documented parameter, with the description collapsed
        into a single whitespace-normalised line. Parameters documented as a
        comma-separated group (``fmin_hz, fmax_hz : float``) share the same
        description. Returns an empty dict when no ``Parameters`` section is
        present.
    """
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()

    start: int | None = None
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "Parameters" and _SECTION_UNDERLINE.match(lines[i + 1]):
            start = i + 2
            break
    if start is None:
        return {}

    out: dict[str, str] = {}
    current: list[str] = []
    names: list[str] = []

    def flush() -> None:
        if names:
            description = " ".join(" ".join(current).split())
            for name in names:
                out[name] = description

    for i in range(start, len(lines)):
        line = lines[i]
        stripped = line.strip()
        # A new section header ends the parameter block.
        if i + 1 < len(lines) and stripped and _SECTION_UNDERLINE.match(lines[i + 1]):
            break
        if not stripped:
            current.append("")
            continue
        if not line[:1].isspace():
            flush()
            match = _PARAM_HEADER.match(stripped)
            names = []
            current = []
            if match:
                names = [n.strip().lstrip("*") for n in match.group("names").split(",") if n.strip()]
            continue
        current.append(stripped)

    flush()
    return out
