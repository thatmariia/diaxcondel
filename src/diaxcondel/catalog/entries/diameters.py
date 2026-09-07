"""
Axon-diameter distributions: how thick the fibres of a bundle are.

A bundle's axons are not all the same thickness, and thicker axons conduct
faster. The delay kernel of Steeghs-Turchina et al. (2025) is built from
exactly those two facts: a distribution over calibres, and a speed that grows
with calibre. They are separate choices here for the same reason they are
separate in the paper. The calibre distribution is an anatomical
measurement, while the speed constant that turns calibre into velocity is a
physiological one.

Which family fits histological data best is itself an open question;
Sepehrband et al. (2016) fit several to corpus-callosum measurements, and all
of them are available here.
"""

from __future__ import annotations

from typing import Annotated

from diaxcondel.kernels.diameter import (
    DiameterDistribution,
    GammaDiameter,
    LognormalDiameter,
    RayleighDiameter,
)
from diaxcondel.kernels.gev import GEVDiameter

from ..param import Param
from ..registry import register


@register(
    "diameter",
    "gev",
    label="Generalised extreme value (reference)",
    tags=("reference",),
    reference="Sepehrband et al. (2016); values used by Steeghs-Turchina et al. (2025)",
)
def gev_diameter(
    mu: Annotated[float, Param(unit="um", minimum=0.05, maximum=2.0, step=0.01)] = 0.25,
    sigma: Annotated[float, Param(unit="um", minimum=0.01, maximum=1.0, step=0.01)] = 0.09,
    xi: Annotated[float, Param(minimum=-1.0, maximum=1.0, step=0.01)] = -0.25,
) -> DiameterDistribution:
    """
    Axon calibres following a generalised extreme value distribution.

    The distribution used in the original paper. Its negative shape parameter
    puts a hard ceiling on calibre, and therefore a hard floor on delay: no
    axon conducts arbitrarily fast.

    Parameters
    ----------
    mu : float
        Location of the distribution, in micrometres, close to the most
        common calibre. Raising it makes fibres thicker and delays shorter.
    sigma : float
        Scale, in micrometres. Wider spread of calibres means a wider spread
        of delays, and flatter spectral peaks.
    xi : float
        Shape. Negative values bound calibre from above (the case fitted to
        histology); zero gives an unbounded exponential-like tail; positive
        values give a heavy tail of very thick, very fast axons.

    Returns
    -------
    DiameterDistribution
        Calibre distribution, ready for a conduction relation.
    """
    return GEVDiameter(mu=mu, sigma=sigma, xi=xi)


@register(
    "diameter",
    "gamma",
    label="Gamma",
    tags=("alternative",),
    reference="Sepehrband et al. (2016), diameter distribution fits",
)
def gamma_diameter(
    mean_um: Annotated[float, Param(unit="um", minimum=0.05, maximum=3.0, step=0.05)] = 0.3,
    shape: Annotated[float, Param(minimum=1.0, maximum=20.0, step=0.5)] = 4.0,
) -> DiameterDistribution:
    """
    Axon calibres following a gamma distribution.

    Right-skewed with a light tail and no ceiling, so unlike the reference
    distribution it allows a few arbitrarily thick (and therefore very fast)
    axons, which softens the sharp early edge of the delay distribution.

    Parameters
    ----------
    mean_um : float
        Mean calibre in micrometres.
    shape : float
        How tightly calibres cluster: the spread is ``mean_um / sqrt(shape)``,
        so large values approach a single calibre.

    Returns
    -------
    DiameterDistribution
        Calibre distribution.
    """
    return GammaDiameter(mean_um=mean_um, shape=shape)


@register(
    "diameter",
    "lognormal",
    label="Log-normal",
    tags=("alternative",),
    reference="Sepehrband et al. (2016), diameter distribution fits",
)
def lognormal_diameter(
    median_um: Annotated[float, Param(unit="um", minimum=0.05, maximum=3.0, step=0.05)] = 0.3,
    sigma_log: Annotated[float, Param(minimum=0.05, maximum=2.0, step=0.05)] = 0.5,
) -> DiameterDistribution:
    """
    Axon calibres following a log-normal distribution.

    The heaviest-tailed option: most fibres are thin and slow while a few are
    very thick and fast, which gives a long slow tail of delays behind a
    sharp fast edge.

    Parameters
    ----------
    median_um : float
        Median calibre in micrometres.
    sigma_log : float
        Spread of the logarithm of calibre. Near 0.1 the bundle is almost of
        one thickness; near 1 its calibres span an order of magnitude, and
        its delays with them.

    Returns
    -------
    DiameterDistribution
        Calibre distribution.
    """
    return LognormalDiameter(median_um=median_um, sigma_log=sigma_log)


@register(
    "diameter",
    "rayleigh",
    label="Rayleigh",
    tags=("alternative",),
    reference="Sepehrband et al. (2016), diameter distribution fits",
)
def rayleigh_diameter(
    scale_um: Annotated[float, Param(unit="um", minimum=0.05, maximum=3.0, step=0.05)] = 0.25,
) -> DiameterDistribution:
    """
    Axon calibres following a Rayleigh distribution.

    A one-parameter family: only the typical calibre can be tuned, its shape
    is fixed. The most constrained of the alternatives, and so the strictest
    test of whether the shape of the calibre distribution matters.

    Parameters
    ----------
    scale_um : float
        Rayleigh scale in micrometres, which is also the most common calibre.

    Returns
    -------
    DiameterDistribution
        Calibre distribution.
    """
    return RayleighDiameter(scale_um=scale_um)
