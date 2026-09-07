"""
Choosing how a figure is drawn.

The playground draws every figure interactively when Plotly is available —
hover for the exact value, drag to zoom into a band, click a legend entry to
drop a region — and falls back to the static matplotlib figures otherwise, so
the page works with only the base app extra installed.

Interactive figures are for exploring. The static ones are what
:mod:`diaxcondel.viz` produces for a paper, and the *Reproduce* tab hands
back the code that makes them.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import streamlit as st

#: Whether the interactive backend is installed.
INTERACTIVE: bool = find_spec("plotly") is not None


def show(
    interactive: Callable[[], Any] | None,
    static: Callable[[], Any],
    *,
    key: str | None = None,
) -> None:
    """
    Render a figure with whichever backend is available.

    Parameters
    ----------
    interactive : callable, optional
        Builds the Plotly figure. Only called when Plotly is installed.
        ``None`` means this figure has no interactive form.
    static : callable
        Builds the matplotlib figure, used as the fallback.
    key : str, optional
        Streamlit element key, needed when several charts share a page.
    """
    if INTERACTIVE and interactive is not None:
        st.plotly_chart(interactive(), width="stretch", theme="streamlit", key=key)
        return
    st.pyplot(static(), width="stretch")
