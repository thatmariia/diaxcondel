"""
The reference delay kernel, under its historical names.

Steeghs-Turchina et al. (2025) build the transmission-delay distribution in
two steps: axon calibres within a bundle follow a generalised extreme value
distribution, and conduction speed grows linearly with calibre (Hursh, 1939),
``v = k * phi``. The delay over a length ``d`` is then ``d / (100 k phi)``,
whose density follows by change of variables.

That construction is now general — any calibre distribution goes through the
same steps, see :mod:`diaxcondel.kernels.diameter` — so these names are kept
as the GEV special case, which is what the paper uses:

* ``InvGEVKernel`` is :class:`~diaxcondel.kernels.diameter.DiameterDelayKernel`
* ``InvGEVSumKernel`` is :class:`~diaxcondel.kernels.diameter.TwoLegDelayKernel`

Both take a :class:`~diaxcondel.kernels.gev.GEVDiameter` and a speed factor,
exactly as before.
"""

from __future__ import annotations

from .diameter import DiameterDelayKernel, TwoLegDelayKernel

#: The paper's delay kernel: GEV axon calibres through the Hursh speed relation.
InvGEVKernel = DiameterDelayKernel

#: The relayed (two-leg) counterpart of :data:`InvGEVKernel`.
InvGEVSumKernel = TwoLegDelayKernel

__all__ = ["InvGEVKernel", "InvGEVSumKernel"]
