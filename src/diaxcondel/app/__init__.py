"""
The interactive playground (optional ``app`` extra).

``diaxcondel playground`` starts a Streamlit server on
:mod:`diaxcondel.app.playground`. The page is a front end for the package: it
builds an :class:`~diaxcondel.experiment.spec.ExperimentSpec` from its
controls and calls the same functions a script would, so anything explored
there can be exported and re-run offline.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

#: Path of the Streamlit script that renders the playground.
PLAYGROUND_SCRIPT = Path(__file__).with_name("playground.py")


def launch(
    *,
    port: int | None = None,
    headless: bool = False,
    extra_args: Sequence[str] = (),
) -> int:
    """
    Start the playground in a Streamlit server.

    Parameters
    ----------
    port : int, optional
        Port to serve on. ``None`` lets Streamlit choose (8501 by default).
    headless : bool
        Do not open a browser window; useful on remote machines.
    extra_args : sequence of str
        Additional arguments passed through to ``streamlit run``.

    Returns
    -------
    int
        Exit status of the Streamlit process.

    Raises
    ------
    ImportError
        If the ``app`` extra is not installed.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("the playground needs streamlit; install it with `poetry install --with app,viz`") from exc

    command = [sys.executable, "-m", "streamlit", "run", str(PLAYGROUND_SCRIPT)]
    if port is not None:
        command += ["--server.port", str(port)]
    if headless:
        command += ["--server.headless", "true"]
    command += list(extra_args)
    return subprocess.call(command)


__all__ = ["PLAYGROUND_SCRIPT", "launch"]
