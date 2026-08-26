"""Excitations for the string: initial-condition shapes (pluck) and velocity strikes.

These return arrays sampled on the resonator's grid ``x``; feed them to
:meth:`IdealString.set_state`. Kept dependency-free (NumPy only) so they live in the headless core.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

__all__ = ["triangular_pluck", "raised_cosine", "raised_cosine_2d"]


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


def raised_cosine_2d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    center: tuple[float, float],
    width: float,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Smooth (C^1) radial raised-cosine hump on a 2D grid, zero outside radius ``width``.

    The 2D analogue of :func:`raised_cosine`: a band-limited bump centred at ``center = (xc, yc)``,
    falling smoothly to zero at radial distance ``width``. Returns a field of shape ``X.shape``;
    feed it (or its masked selection) to :meth:`physsynth.core.membrane.Membrane.set_state`. Excites
    a broad band of modes without the slowly-converging high partials of a cornered shape.
    """
    if width <= 0:
        raise ValueError("width must be > 0.")
    xc, yc = center
    d = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    field = np.zeros_like(X)
    inside = d < width
    field[inside] = amplitude * 0.5 * (1.0 + np.cos(np.pi * d[inside] / width))
    return field


# --- the Rust swap (docs/dev/rust-migration-plan.md, Phase 2) -----------------------------------
#
# Same shape as `operators.py`'s: the ``_py`` aliases above the switch are the reference
# implementations and are what every parity check reaches for; the public names below are bound to
# whichever implementation this process is meant to exercise.
#
# One divergence worth naming rather than discovering: ``triangular_pluck`` builds its result with
# ``np.empty_like(x)``, so on a float32 grid the Python version returns float32 while the Rust one
# always returns float64. Nothing in this repo samples a resonator grid at single precision -- every
# grid comes from ``np.linspace`` -- but the dtype is not preserved, and that is a difference a
# future caller could notice.
#
# Off by default. The Python implementations are still the reference oracle.
triangular_pluck_py = triangular_pluck
raised_cosine_py = raised_cosine
raised_cosine_2d_py = raised_cosine_2d

_USE_RUST = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")

if _USE_RUST:  # pragma: no cover - exercised by the dedicated CI job, not the default gate
    import physsynth_rs as _rs

    def triangular_pluck(x, L, position, amplitude=1.0):  # type: ignore[misc]  # noqa: F811
        return _rs.triangular_pluck(x, L, position, amplitude)

    def raised_cosine(x, L, center, width, amplitude=1.0):  # type: ignore[misc]  # noqa: F811
        return _rs.raised_cosine(x, L, center, width, amplitude)

    def raised_cosine_2d(X, Y, center, width, amplitude=1.0):  # type: ignore[misc]  # noqa: F811
        return _rs.raised_cosine_2d(X, Y, center, width, amplitude)
