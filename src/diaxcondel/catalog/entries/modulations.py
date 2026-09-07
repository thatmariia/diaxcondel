"""
Modulation entries: time-limited changes of effective coupling.

In the linear zero-mean model a modulation alone does not create a signed
ERP average. It changes variance, power, and coherence. Pair it with a
drive to modulate ERP amplitude or latency.
"""

from __future__ import annotations

from typing import Annotated, Literal

from diaxcondel.connectome.regions import RegionSet
from diaxcondel.erp.modulation import StepModulation, region_gain_factor

from ..param import Param
from ..registry import register


@register("modulation", "step_gain", label="Step gain change", tags=("erp",))
def step_gain(
    regions: RegionSet,
    targets: Annotated[tuple[str, ...], Param(widget="regions")] = (),
    gain: Annotated[float, Param(minimum=0.0, maximum=3.0, step=0.05)] = 1.5,
    duration_seconds: Annotated[float, Param(unit="s", minimum=0.005, maximum=2.0, step=0.005)] = 0.2,
    direction: Literal["incoming", "outgoing", "both"] = "incoming",
    instability_policy: Literal["warn", "raise", "ignore"] = "warn",
) -> StepModulation:
    """
    Scale the coupling of chosen regions for a fixed window.

    Parameters
    ----------
    regions : RegionSet
        Model regions (supplied by the builder).
    targets : tuple of str
        Region codes whose edges are scaled. Empty means a global gain
        change across the whole network.
    gain : float
        Multiplicative factor on the selected edges; ``1.0`` is no change.
    duration_seconds : float
        How long the modulation stays active after its onset.
    direction : {"incoming", "outgoing", "both"}
        Whether to scale connections arriving at, leaving, or touching the
        target regions.
    instability_policy : {"warn", "raise", "ignore"}
        What to do if the modulated system is transiently non-stationary.

    Returns
    -------
    StepModulation
        Modulation ready to attach to an event.
    """
    factor = region_gain_factor(regions, targets, gain=gain, direction=direction)
    return StepModulation(
        factor=factor,
        duration_seconds=duration_seconds,
        instability_policy=instability_policy,
    )
