"""Macroscopic VAR modelling of EEG/MEG from white-matter delays."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diaxcondel")
except PackageNotFoundError:  # pragma: no cover - source tree without installation metadata
    __version__ = "0.0.0+local"
