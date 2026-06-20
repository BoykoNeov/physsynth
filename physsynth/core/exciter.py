"""Excitations for the string: initial-condition shapes (pluck) and velocity strikes.

These return arrays sampled on the resonator's grid ``x``; feed them to
:meth:`IdealString.set_state`. Kept dependency-free (NumPy only) so they live in the headless core.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["triangular_pluck", "raised_cosine"]


def triangular_pluck(
    x: NDArray[np.float64],
    L: float,
    position: float,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Triangular plucked-string initial displacement (zero at both ends).

    A tent peaked at ``position`` (in metres, ``0 < position < L``): linear rise from the left end
    to the pluck point, linear fall to the right end. This excites the full harmonic series, so it
    is the right shape for the modal/partial-detection test.
    """
    if not (0.0 < position < L):
        raise ValueError(f"pluck position must satisfy 0 < position < L (L={L}), got {position}.")
    xp = position
    left = x <= xp
    u0 = np.empty_like(x)
    u0[left] = amplitude * x[left] / xp
    u0[~left] = amplitude * (L - x[~left]) / (L - xp)
    return u0


def raised_cosine(
    x: NDArray[np.float64],
    L: float,
    center: float,
    width: float,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Smooth (C^1) raised-cosine displacement hump, zero outside ``[center-width, center+width]``.

    Band-limited compared with the triangular pluck (no corner -> no slowly-converging high
    partials), which makes it the cleaner excitation for a grid-convergence study.
    """
    if width <= 0:
        raise ValueError("width must be > 0.")
    u0 = np.zeros_like(x)
    d = np.abs(x - center)
    inside = d < width
    u0[inside] = amplitude * 0.5 * (1.0 + np.cos(np.pi * d[inside] / width))
    # Clamp the ends so a hump near a boundary cannot leave a non-zero node at a fixed end.
    u0[0] = 0.0
    u0[-1] = 0.0
    return u0
