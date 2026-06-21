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
from scipy import special

__all__ = [
    "harmonic_frequencies",
    "mode_shape",
    "discrete_mode_frequency",
    "inharmonicity_B",
    "stiff_harmonic_frequencies",
    "discrete_stiff_mode_frequency",
    "cents",
    # 2D membrane (model #4)
    "rectangular_membrane_freqs",
    "rectangular_mode_field",
    "rectangular_discrete_eigenvalues",
    "circular_membrane_freqs",
    "discrete_membrane_eigenfrequency",
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


# -- 2D membrane (model #4) --------------------------------------------------------------------


def rectangular_membrane_freqs(
    c: float, Lx: float, Ly: float, modes: list[tuple[int, int]]
) -> NDArray[np.float64]:
    """Continuum rectangular-membrane frequencies ``f_{mn} = (c/2) sqrt((m/Lx)² + (n/Ly)²)``.

    ``modes`` is a list of ``(m, n)`` mode indices (both >= 1). Fixed-rim modes are
    ``sin(mπx/Lx) sin(nπy/Ly)``. This is the exact, O(h²)-clean oracle for the harness unit-test
    geometry (the rectangle), used before the staircase error enters on the circle.
    """
    mn = np.asarray(modes, dtype=float)
    return 0.5 * c * np.sqrt((mn[:, 0] / Lx) ** 2 + (mn[:, 1] / Ly) ** 2)


def rectangular_mode_field(
    X: NDArray[np.float64], Y: NDArray[np.float64], Lx: float, Ly: float, m: int, n: int
) -> NDArray[np.float64]:
    """The ``(m, n)`` fixed-rim mode ``sin(mπx/Lx) sin(nπy/Ly)`` sampled on grid ``(X, Y)``.

    An *exact* discrete eigenvector of the 5-point Laplacian (tensor product of the 1D
    ``sin(mπl/N)`` eigenvector), so a single-mode initial condition stays one clean tone — the 2D
    analogue of :func:`mode_shape`.
    """
    return np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)


def rectangular_discrete_eigenvalues(
    h: float, Nx: int, Ny: int, modes: list[tuple[int, int]]
) -> NDArray[np.float64]:
    """Closed-form eigenvalues ``Λ_{mn}`` of ``-Δ_h`` on a rectangle (``-L`` is SPD, ``Λ > 0``).

    ``Λ_{mn} = (4/h²)[sin²(mπ/(2Nx)) + sin²(nπ/(2Ny))]`` for ``m = 1..Nx-1``, ``n = 1..Ny-1`` — the
    2D tensor product of the 1D second-difference spectrum (see
    :func:`physsynth.core.operators.second_difference_matrix`). The assembled masked Laplacian must
    reproduce these to machine precision, which is what proves the operator is wired correctly
    before any continuum/Bessel comparison.
    """
    mn = np.asarray(modes, dtype=float)
    sx = np.sin(mn[:, 0] * np.pi / (2 * Nx)) ** 2
    sy = np.sin(mn[:, 1] * np.pi / (2 * Ny)) ** 2
    return (4.0 / (h * h)) * (sx + sy)


def circular_membrane_freqs(
    c: float, a: float, n_modes: int, m_max: int = 12, n_max: int = 12
) -> list[tuple[int, int, float, int]]:
    """Lowest ``n_modes`` circular-membrane frequencies, each tagged ``(m, n, freq, degeneracy)``.

    ``f_{mn} = c · j_{m,n} / (2π a)`` where ``j_{m,n}`` is the n-th positive zero of the Bessel
    function ``J_m`` (mode shape ``J_m(j_{m,n} r/a)·{cos,sin}(mθ)``). ``m = 0`` modes are
    non-degenerate (``degeneracy = 1``); ``m >= 1`` come as a cos/sin pair (``degeneracy = 2``).
    Returned sorted by frequency. ``m_max, n_max`` bound the search grid of zeros (raise if
    ``n_modes`` is large). The drumhead oracle (HANDOFF §5 row 4).
    """
    entries: list[tuple[int, int, float, int]] = []
    for m in range(0, m_max + 1):
        zeros = special.jn_zeros(m, n_max)  # first n_max positive zeros of J_m
        deg = 1 if m == 0 else 2
        for n_i, z in enumerate(zeros, start=1):
            entries.append((m, n_i, c * z / (2.0 * np.pi * a), deg))
    entries.sort(key=lambda e: e[2])
    return entries[:n_modes]


def discrete_membrane_eigenfrequency(
    Lambda: float | NDArray[np.float64], c: float, k: float
) -> NDArray[np.float64]:
    """Discrete temporal frequency (Hz) of an eigenmode with Laplacian eigenvalue ``Λ`` (of ``-L``).

    Inserting ``u^n = z^n φ`` with ``L φ = -Λ φ`` into the explicit scheme gives
    ``cos(ω k) = 1 − c²k²Λ/2``, hence ``f = arccos(1 − c²k²Λ/2) / (2π k)``. As ``k → 0`` this tends
    to the continuum ``f = c sqrt(Λ_cont)/(2π)``. The temporal companion to the spatial eigenvalue
    test: spectrum (``Λ`` of the masked Laplacian) → measurable frequency.
    """
    Lambda = np.asarray(Lambda, dtype=float)
    arg = 1.0 - 0.5 * (c * k) ** 2 * Lambda
    return np.arccos(np.clip(arg, -1.0, 1.0)) / (2.0 * np.pi * k)
