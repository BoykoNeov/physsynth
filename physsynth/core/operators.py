"""Finite-difference operators (HANDOFF Appendix A) and the discrete inner product.

All spatial operators act on a 1-D grid array ``u`` of shape ``(N + 1,)`` sampling
``u(l*h)`` for ``l = 0 .. N``. The discrete inner product ``<f, g> = h * sum_l f[l] g[l]``
(and its norm) is the bookkeeping device behind every energy proof in this project, so it lives
here next to the operators it pairs with.

Headless: NumPy only. No I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "delta_x_forward",
    "delta_x_backward",
    "delta_xx",
    "inner",
    "norm2",
]


def delta_x_forward(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Forward spatial difference ``delta_x+ u[l] = (u[l+1] - u[l]) / h``.

    Returns the ``N`` inter-node strains (length ``len(u) - 1``); element ``l`` lives on the
    segment between nodes ``l`` and ``l+1``.
    """
    return (u[1:] - u[:-1]) / h


def delta_x_backward(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Backward spatial difference ``delta_x- u[l] = (u[l] - u[l-1]) / h`` (length ``len(u) - 1``).

    Numerically equal array to :func:`delta_x_forward`; provided for notational symmetry.
    """
    return (u[1:] - u[:-1]) / h


def delta_xx(u: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """Second spatial difference at interior nodes.

    ``delta_xx u[l] = (u[l+1] - 2 u[l] + u[l-1]) / h^2`` for ``l = 1 .. N-1``.
    Returns the ``N - 1`` interior values (length ``len(u) - 2``); boundary handling is the
    caller's responsibility.
    """
    return (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (h * h)


def inner(f: NDArray[np.float64], g: NDArray[np.float64], h: float) -> float:
    """Discrete inner product ``<f, g> = h * sum_l f[l] g[l]``."""
    return float(h * np.dot(f, g))


def norm2(f: NDArray[np.float64], h: float) -> float:
    """Squared discrete norm ``||f||^2 = <f, f> = h * sum_l f[l]^2`` (>= 0)."""
    return float(h * np.dot(f, f))
