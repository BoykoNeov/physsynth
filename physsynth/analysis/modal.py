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
    "inharmonicity_B",
    "stiff_harmonic_frequencies",
    "discrete_stiff_mode_frequency",
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


# -- stiff string (model #2): stretched partials ----------------------------------------------


def inharmonicity_B(c: float, L: float, kappa: float) -> float:
    """Inharmonicity coefficient ``B = pi^2 kappa^2 / (c^2 L^2)`` (simply-supported stiff string).

    The single audible stiffness parameter; for a piano ``B ~ 1e-4 .. 1e-3``. Scales as
    ``kappa^2`` -- the relationship the B-vs-kappa sweep test pins down quantitatively.
    """
    return float((np.pi ** 2) * kappa ** 2 / (c ** 2 * L ** 2))


def stiff_harmonic_frequencies(
    c: float, L: float, kappa: float, n_partials: int
) -> NDArray[np.float64]:
    """Continuum stretched partials ``f_n = n f0 sqrt(1 + B n^2)``, ``f0 = c/(2L)``.

    The simply-supported stiff-string oracle (HANDOFF §5, Bilbao Ch. 7): bending stiffness pushes
    every partial sharp, including the fundamental (``f_1 = f0 sqrt(1 + B)``, *not* ``f0``). This is
    the "theory" the FDTD partials are measured against.
    """
    n = np.arange(1, n_partials + 1)
    f0 = c / (2.0 * L)
    B = inharmonicity_B(c, L, kappa)
    return n * f0 * np.sqrt(1.0 + B * n * n)


def discrete_stiff_mode_frequency(
    c: float, L: float, N: int, kappa: float, k: float, m: int, theta: float
) -> float:
    """Exact oscillation frequency (Hz) of mode ``m`` for the implicit theta-scheme stiff string.

    Inserting ``u^n = z^n sin(m pi x / L)`` (an exact discrete eigenvector) with the spatial
    eigenvalue ``p^2 = (4/h^2) sin^2(m pi / 2N)`` (so ``delta_xx -> -p^2`` and ``delta_xxxx ->
    +p^4``) and ``Q = c^2 p^2 + kappa^2 p^4`` gives

        sin^2(omega k / 2) = s = Q k^2 / (4 + 4 theta Q k^2),
        f_m = arcsin(sqrt(s)) / (pi k).

    As ``h, k -> 0`` this tends to the continuum ``f_n = n f0 sqrt(1 + B n^2)`` for any ``theta``.
    **The frequency depends on ``theta``** (the temporal scheme), so callers must pass the
    resonator's own ``theta``. At ``kappa = 0`` it is the implicit wave scheme's oracle (which is
    *not* the explicit one -- not exact even at ``lambda = 1``).
    """
    h = L / N
    p2 = (4.0 / (h * h)) * np.sin(m * np.pi / (2 * N)) ** 2
    Q = c * c * p2 + kappa * kappa * p2 * p2
    s = Q * k * k / (4.0 + 4.0 * theta * Q * k * k)
    return float(np.arcsin(np.sqrt(s)) / (np.pi * k))


def cents(f: float | NDArray[np.float64], f_ref: float | NDArray[np.float64]):
    """Pitch error in cents: ``1200 * log2(f / f_ref)``. Scalar or elementwise on arrays."""
    return 1200.0 * np.log2(np.asarray(f, dtype=float) / np.asarray(f_ref, dtype=float))
