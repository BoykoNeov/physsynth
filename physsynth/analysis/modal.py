"""Analytic modal oracles for the ideal string.

These are the closed-form references the FDTD output is validated against (HANDOFF §5, §6):

- the continuous harmonic series ``f_n = n * c / (2L)`` (fixed–fixed string),
- the spatial mode shapes ``sin(n pi x / L)``,
- the *exact* oscillation frequency of the discrete explicit scheme (the numerical-dispersion
  oracle), which lets convergence be predicted in closed form rather than only measured.

Pure NumPy. No dependency on the core (oracles are independent of the implementation under test).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "harmonic_frequencies",
    "mode_shape",
    "discrete_mode_frequency",
    "cents",
]


def harmonic_frequencies(c: float, L: float, n_partials: int) -> NDArray[np.float64]:
    """Continuous fixed–fixed harmonic series ``f_n = n c / (2L)`` for ``n = 1 .. n_partials``."""
    n = np.arange(1, n_partials + 1)
    return n * c / (2.0 * L)


def mode_shape(x: NDArray[np.float64], L: float, m: int) -> NDArray[np.float64]:
    """The ``m``-th fixed–fixed spatial mode ``sin(m pi x / L)`` sampled on grid ``x``.

    This is an exact eigenvector of the discrete second-difference operator, so initializing the
    string with it yields a clean single-frequency oscillation — ideal for a convergence study.
    """
    return np.sin(m * np.pi * x / L)


def discrete_mode_frequency(c: float, L: float, N: int, lam: float, m: int) -> float:
    """Exact oscillation frequency (Hz) of mode ``m`` for the explicit scheme.

    Solving the scheme's amplification relation gives ``sin(omega k / 2) = lambda sin(m pi / 2N)``,
    hence

        f_m^discrete = (1 / (pi k)) * arcsin( lambda * sin(m pi / (2N)) ),    k = lambda L / (c N).

    At ``lambda = 1`` this collapses to the exact ``m c / (2L)``; for ``lambda < 1`` it lies below
    it (numerical dispersion), with the gap shrinking as O(h^2). Provided as an independent oracle
    for the convergence test.
    """
    k = lam * L / (c * N)  # = 1/fs, from lambda = c k / h and h = L/N
    return float(np.arcsin(lam * np.sin(m * np.pi / (2 * N))) / (np.pi * k))


def cents(f: float | NDArray[np.float64], f_ref: float | NDArray[np.float64]):
    """Pitch error in cents: ``1200 * log2(f / f_ref)``. Scalar or elementwise on arrays."""
    return 1200.0 * np.log2(np.asarray(f, dtype=float) / np.asarray(f_ref, dtype=float))
