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
from scipy.optimize import brentq

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
    # 2D plate (model #5)
    "rectangular_plate_freqs",
    "discrete_plate_eigenfrequency",
    # 1D free-free beam (model #5b-pre)
    "free_free_beam_betaL",
    "free_free_beam_freqs",
    "discrete_beam_eigenfrequency",
    # 2D free-edge (FFFF) plate (model #5b)
    "free_plate_ffff_square_lambdas",
    "free_plate_freq_from_lambda",
    # 1D acoustic bore (wind leg)
    "bore_resonance_frequencies",
    "discrete_bore_eigenfrequency",
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


# -- 2D plate (model #5): simply-supported rectangle ------------------------------------------


def rectangular_plate_freqs(
    kappa: float, Lx: float, Ly: float, modes: list[tuple[int, int]]
) -> NDArray[np.float64]:
    """Continuum simply-supported rectangular-plate frequencies (HANDOFF §5 row 5).

    With stiffness ``kappa = sqrt(D/rho_s)`` (D = flexural rigidity, rho_s = areal density), the
    Navier modes are ``sin(mπx/Lx) sin(nπy/Ly)`` with ``ω_{mn} = kappa·γ_{mn}`` where
    ``γ_{mn} = (mπ/Lx)² + (nπ/Ly)²`` is the Laplacian eigenvalue magnitude, hence

        f_{mn} = kappa·γ_{mn} / (2π) = (π/2)·kappa·[(m/Lx)² + (n/Ly)²].

    Pure 4th-power in the wavenumbers (no tension term — the plate is bending-only). ``modes`` is a
    list of ``(m, n)`` indices (both >= 1). Poisson's ratio drops out for simply-supported edges, so
    ``kappa`` is the single stiffness parameter.
    """
    mn = np.asarray(modes, dtype=float)
    gamma = (mn[:, 0] * np.pi / Lx) ** 2 + (mn[:, 1] * np.pi / Ly) ** 2
    return kappa * gamma / (2.0 * np.pi)


def discrete_plate_eigenfrequency(
    Lambda_lap: float | NDArray[np.float64], kappa: float, k: float, theta: float
) -> NDArray[np.float64]:
    """Discrete temporal frequency (Hz) of a plate eigenmode, implicit theta-scheme.

    ``Lambda_lap`` is the **Laplacian** eigenvalue magnitude ``Λ`` (of ``-L``); the *biharmonic*
    eigenvalue is ``Λ²``, so the plate's modal stiffness is ``Q = kappa²·Λ²`` (4th power — easy to
    under-square or double-square, so it is pinned here). Inserting ``u^n = z^n φ`` with
    ``B φ = Λ² φ`` into ``δ_tt u = -kappa² B (θ u^{n+1} + (1-2θ) u^n + θ u^{n-1})`` gives

        sin²(ω k / 2) = s = Q k² / (4 + 4 θ Q k²),   f = arcsin(sqrt(s)) / (π k).

    As ``k → 0`` this tends to the continuum ``f = kappa·Λ/(2π)``. Unconditionally stable for
    ``θ >= 1/4`` (``s <= 1``); the frequency depends on ``θ`` (the temporal scheme), so callers must
    pass the resonator's own ``theta``.
    """
    Lambda_lap = np.asarray(Lambda_lap, dtype=float)
    Q = kappa * kappa * Lambda_lap * Lambda_lap
    s = Q * k * k / (4.0 + 4.0 * theta * Q * k * k)
    return np.arcsin(np.sqrt(s)) / (np.pi * k)


# -- 1D free-free Euler-Bernoulli beam (model #5b-pre): the free-edge plate de-risk ------------


def free_free_beam_betaL(n_modes: int) -> NDArray[np.float64]:
    """First ``n_modes`` positive roots ``β_n L`` of the free-free frequency equation.

    The elastic modes of a free-free Euler–Bernoulli beam satisfy ``cos(βL)·cosh(βL) = 1``. The
    roots are found by ``brentq`` on the overflow-safe rearrangement ``cos(x) − sech(x) = 0``
    (``sech`` underflows harmlessly for large ``x``); root ``i`` lives in ``(iπ, (i+1)π)`` and tends
    to ``(2i+1)π/2`` from above. The first few are ``4.730041, 7.853205, 10.995608, 14.137165, …``.

    The double root at ``x = 0`` (the two rigid-body modes ``u ≡ 1`` and ``u ≡ x``, ``ω = 0``) is
    **not** returned — those are the operator's nullspace, checked separately.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    f = lambda x: np.cos(x) - 1.0 / np.cosh(x)  # noqa: E731
    return np.array([brentq(f, i * np.pi, (i + 1) * np.pi) for i in range(1, n_modes + 1)])


def free_free_beam_freqs(kappa: float, L: float, n_modes: int) -> NDArray[np.float64]:
    """Closed-form free-free Euler–Bernoulli bending frequencies (the Part-0 oracle).

    With stiffness ``kappa = sqrt(E I / (rho A))`` (same convention as the stiff string), the
    bending PDE ``u_tt = -kappa² u_xxxx`` on a *free-free* beam has ``ω_n = kappa·β_n²`` where
    ``β_n L`` are the roots of ``cos(βL)·cosh(βL) = 1`` (:func:`free_free_beam_betaL`), hence

        f_n = kappa · (β_n L)² / (2π L²).

    This is a *genuine closed form* (unlike the 2D free plate), which is exactly why the beam is
    built first: it gives a tight oracle for the free-edge stencil before the 2D corners and Poisson
    term. Returns the lowest ``n_modes`` elastic frequencies (the two ``ω = 0`` rigid-body modes
    excluded).
    """
    bl = free_free_beam_betaL(n_modes)
    return kappa * bl * bl / (2.0 * np.pi * L * L)


# -- 2D free-edge (FFFF) plate (model #5b): the curved-Chladni plate, no closed form -------------


def free_plate_ffff_square_lambdas() -> NDArray[np.float64]:
    """Tabulated dimensionless frequencies of the **completely free (FFFF) square** plate, ν = 0.3.

    The free plate has **no closed-form** modal oracle (unlike the simply-supported plate or the
    free-free beam), so this is the *absolute* percent-level anchor for the free-edge plate. The
    values are the dimensionless frequency parameters

        λ = ω a² √(ρ_s / D) = ω a² / κ        (a = square side length, κ = √(D/ρ_s)),

    so the physical frequency is ``f = λ κ / (2π a²)`` (:func:`free_plate_freq_from_lambda`).

    Returned are the **lowest 5 elastic modes** (the 3 rigid-body modes ``{1, x, y}`` are excluded):

        λ = [13.468, 19.596, 24.270, 34.801, 34.801]

    The **fundamental is the saddle/twist** (diagonal nodal lines, λ₁ ≈ 13.47) — *not* a drum-like
    bulge; a bulge-shaped lowest mode signals a wrong (ν-dropped) operator. Modes 4 and 5 are a
    degenerate pair (the square's symmetry), so the comparison must be by **sorted eigenvalue**, not
    per-mode label.

    Source (exact digits, not remembered): Y. Narita, "Natural Frequencies of Isotropic Rectangular
    Plates in Improved Accuracy", *EPI International Journal of Engineering* **5**(1), 2022,
    pp. 26–36 (DOI 10.25042/epi-ije.022022.05), Table 1, F-F-F-F, a/b = 1, 12×12 Ritz. These are
    converged upper-bound Ritz values that *improve on* Leissa's classic 1969 monograph
    (NASA SP-160), whose corresponding parameters (≈13.49, 19.79, 24.43, 35.02) are slightly higher.
    """
    return np.array([13.468, 19.596, 24.270, 34.801, 34.801])


def free_plate_freq_from_lambda(
    lam: float | NDArray[np.float64], kappa: float, a: float
) -> NDArray[np.float64]:
    """Convert a dimensionless free-plate frequency parameter ``λ`` to a physical frequency (Hz).

    With ``λ = ω a² √(ρ_s/D) = ω a²/κ`` (``κ = √(D/ρ_s)``, ``a`` = square side length), the
    frequency is

        f = ω / (2π) = λ κ / (2π a²).

    No areal-density or thickness factor to fumble — ``κ`` is the single stiffness parameter (as for
    the simply-supported plate). Pairs with :func:`free_plate_ffff_square_lambdas`.
    """
    lam = np.asarray(lam, dtype=float)
    return lam * kappa / (2.0 * np.pi * a * a)


# -- 1D acoustic bore (wind leg): the air column of a clarinet / flute ------------------------


def bore_resonance_frequencies(
    c0: float, L: float, n_partials: int, boundary: str = "closed-open"
) -> NDArray[np.float64]:
    """Continuum acoustic-tube resonances (HANDOFF §12.A, the clarinet oracle).

    A tube of length ``L`` in air of sound speed ``c0`` resonates at

    - ``"closed-open"`` -> ``f_n = (2n - 1) c0 / (4 L)``  — the **odd** harmonics only (a quarter-
      wave resonator: pressure antinode at the rigid wall, node at the open end). This is the
      clarinet signature and the model-specific correctness test for the bore.
    - ``"open-open"`` or ``"closed-closed"`` -> ``f_n = n c0 / (2 L)`` — the **full** harmonic
      series (half-wave resonator; symmetric ends, matching boundary condition at both).

    Returns the lowest ``n_partials`` resonances (Hz), ascending. The exact air constant ``c0`` is
    the analogue of the string's wave speed here; unlike string tension it is a property of the
    medium, not tunable.
    """
    n = np.arange(1, n_partials + 1)
    if boundary == "closed-open":
        return (2 * n - 1) * c0 / (4.0 * L)
    if boundary in ("open-open", "closed-closed"):
        return n * c0 / (2.0 * L)
    raise ValueError(
        f"boundary must be 'closed-open', 'open-open', or 'closed-closed', got {boundary!r}."
    )


def discrete_bore_eigenfrequency(
    omega2: float | NDArray[np.float64], k: float
) -> NDArray[np.float64]:
    """Discrete temporal frequency (Hz) of a bore eigenmode from its angular-frequency-squared.

    The staggered p/U leapfrog eliminates the velocity to ``C δ_tt p = -k² L p`` (``L = Gᵀ M⁻¹ G``,
    ``C`` the compliance mass), so an eigenmode ``L φ = ω² C φ`` obeys the same simple-harmonic
    leapfrog as every other second-order scheme here:

        sin(Ω k / 2) = (k/2) ω = (k/2) sqrt(omega2),    f = arcsin((k/2) sqrt(omega2)) / (π k).

    ``omega2`` is the **generalized eigenvalue** of ``(L, C)`` restricted to the free (non-open)
    pressure nodes — already ``ω²`` (no extra wave-speed factor: ``c0`` is baked into ``L`` and
    ``C``). As ``k → 0`` this tends to the continuum ``f = sqrt(omega2)/(2π)``, whose values are the
    resonances of :func:`bore_resonance_frequencies`. Stable for all valid ``λ`` (the argument stays
    ``<= 1`` when ``λ <= 1``).
    """
    omega2 = np.asarray(omega2, dtype=float)
    arg = 0.5 * k * np.sqrt(np.clip(omega2, 0.0, None))
    return np.arcsin(np.clip(arg, -1.0, 1.0)) / (np.pi * k)


def discrete_beam_eigenfrequency(
    mu: float | NDArray[np.float64], kappa: float, k: float, theta: float
) -> NDArray[np.float64]:
    """Discrete temporal frequency (Hz) of a beam eigenmode, implicit theta-scheme.

    ``mu`` is the **4th-power spatial eigenvalue** ``ω²/κ²`` — the generalized eigenvalue of
    ``K φ = mu W φ`` for the energy-first operator
    (:func:`physsynth.core.operators.free_beam_stiffness`), ``mu → β⁴`` in the continuum. The modal
    stiffness is ``Q = kappa²·mu = ω²``. Inserting ``u^n = z^n φ`` into
    ``W δ_tt u = -kappa² K (θ u^{n+1} + (1-2θ) u^n + θ u^{n-1})`` gives the same
    relation as the plate (with ``Q`` from the beam operator rather than ``kappa²Λ²``):

        sin²(ω k / 2) = s = Q k² / (4 + 4 θ Q k²),    f = arcsin(sqrt(s)) / (π k).

    As ``k → 0`` this tends to the spatial ``f = kappa·sqrt(mu)/(2π)``. Unconditionally stable for
    ``θ >= 1/4``; depends on ``θ``, so callers pass the resonator's own ``theta``.
    """
    mu = np.asarray(mu, dtype=float)
    Q = kappa * kappa * mu
    s = Q * k * k / (4.0 + 4.0 * theta * Q * k * k)
    return np.arcsin(np.sqrt(s)) / (np.pi * k)
