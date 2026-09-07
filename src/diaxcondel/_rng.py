"""
Canonical RNG handling.

Every function in diaxcondel that consumes randomness takes an
``rng: RNGLike`` argument and resolves it through :func:`as_generator`.
This forbids hidden global state and makes every stochastic computation
exactly reproducible.
"""

from __future__ import annotations

import numpy as np

type RNGLike = np.random.Generator | int | np.random.SeedSequence | None


def as_generator(rng: RNGLike) -> np.random.Generator:
    """Coerce any seed-like input into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    rng : Generator, int, SeedSequence, or None
        The randomness source. ``None`` requests a fresh, OS-seeded generator.

    Returns
    -------
    numpy.random.Generator
        A fresh or wrapped generator. If ``rng`` is already a Generator it is
        returned unchanged (no copy), so callers that want isolation must call
        :meth:`Generator.spawn` themselves.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
