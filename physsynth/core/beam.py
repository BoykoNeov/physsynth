"""Free-free Euler–Bernoulli beam — implicit theta-scheme FDTD (energy-first, free edges).

The 1D rehearsal of the free-edge Chladni plate (HANDOFF §5 row 5;
``docs/dev/plate-free-edge-plan.md``
Part 0). It isolates the *free-boundary* flexural stencil and the energy-first operator symmetry
**without** the 2D corners or Poisson's ratio, and — unlike every prior resonator — has a genuine
**closed-form** modal oracle (``cos(βL)·cosh(βL) = 1``) to validate that stencil against.

PDE — **bending only, free-free** (transverse displacement ``u(x, t)`` on ``[0, L]``, both ends
free):

    u_tt = -kappa² u_xxxx - 2 sigma u_t,    kappa² = E I / (rho A)   (same kappa as the stiff
    string)

The free-free natural BCs are zero bending moment ``u_xx = 0`` and zero shear ``u_xxx = 0`` at both
ends. Because every node — including the two ends — is an unknown, the simply-supported trick of
squaring a *Dirichlet* Laplacian (:class:`~physsynth.core.string_stiff.StiffString`,
:class:`~physsynth.core.plate.Plate`) does **not** apply: there is no clamped rim and no exact
``sin`` eigenvector. Instead the operator is built **from the energy** (Bilbao's free bar): a
symmetric stiffness ``K`` and a diagonal trapezoidal mass ``W`` from
:func:`physsynth.core.operators.free_beam_stiffness`, with

    ∫(u_xx)² dx ≈ uᵀ K u,    K = h · D2ᵀ D2  (nullspace exactly {1, x}),
    W = diag(h/2, h, …, h, h/2).

The natural BCs and the rigid-body nullspace ``{1, x}`` fall out *by construction*; the free-end
closure is supplied by the ``h/2`` mass cells, not by hand-coded stencil rows (see the operator).

Treating ``∇⁴`` explicitly forces a brutal ``kappa² k² / h⁴`` CFL (``mu = kappa k / h² <= 1/4``), so
the spatial operator is time-averaged with a theta weight (as for the stiff string / plate):

    W δ_tt u = -kappa² K (theta u^{n+1} + (1-2 theta) u^n + theta u^{n-1}) - 2 sigma W δ_t. u

unconditionally stable for ``theta >= 1/4`` (no CFL limit). Rearranged, each step is one sparse SPD
solve with the constant matrix

    A = (1 + sigma k) W + theta k² kappa² K        (SPD because W is, even though K is only PSD).

``A`` is factored once with ``scipy.sparse.linalg.splu`` and back-substituted each step (mirrors the
plate; the plate's splu path is itself de-risked here).

**Energy** (theta-dependent; the W/K generalization of the stiff string / plate form):

    E^n = rho [ 1/2 (δ_t- u)ᵀ W (δ_t- u)
                + kappa²·( (theta/2)(uⁿᵀ K uⁿ + uⁿ⁻¹ᵀ K uⁿ⁻¹) + (1/2 - theta) uⁿᵀ K uⁿ⁻¹ ) ]

evaluated through the **same** ``K, W`` as the update, so ``E^{n+1} = E^n`` is an exact algebraic
identity (machine precision lossless; monotone decreasing at ``e^{-2 sigma t}`` lossy). The ``h``
quadrature weights live inside ``K`` and ``W`` (no extra scalar ``h`` as in the SS string).

> **Damping caveat (same as the plate, broad).** The theta-time-average makes frequency-independent
> loss effectively frequency-*dependent*: mode decays at ``2 sigma (1 - theta Q k²)`` with
> ``Q = kappa²·mu`` (4th-power across the whole spectrum), so the under-damping bites mid-spectrum.
> Passivity still holds unconditionally; only the *rate* above low modes is wrong. Cure = a later
> frequency-dependent-loss model.

Headless: NumPy + SciPy (sparse LU). No I/O, no plotting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import splu

from .operators import free_beam_stiffness

Boundary = Literal["free"]

# theta below 1/4 is only conditionally stable; theta in (0, 1] keeps A SPD (genuinely implicit).
# Default a hair above 1/4 (accuracy-first) -- the project-wide theta, inherited from the stiff
# string.
THETA_DEFAULT = 0.28


class FreeBeam:
    """A discretized free-free Euler–Bernoulli beam (implicit theta-scheme, energy-first
    operator).

    Parameters
    ----------
    L, rho : float
        Length (m) and linear density ``rho A`` (kg/m). ``rho`` scales the energy (Joules) but not
        the frequencies.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``. **No CFL limit** (unconditionally stable for
        ``theta >= 1/4``) -- coarse grids / large timesteps the explicit beam could not run are
        fine.
    N : int
        Number of spatial segments; the grid has ``N + 1`` nodes, spacing ``h = L/N``. **Every node
        is an unknown** (both ends free), unlike the clamped/supported strings.
    kappa : float
        Stiffness coefficient ``kappa = sqrt(E I / (rho A))`` (units m²/s). Larger ``kappa`` ->
        higher modes. Must be > 0 (``kappa = 0`` => ``u_tt = 0``, degenerate).
    sigma : float
        Frequency-independent loss (>= 0) for the ``-2 sigma u_t`` term. ``0`` -> lossless. See the
        broad damping caveat in the module docstring.
    theta : float
        Time-averaging weight in ``(0, 1]``. ``>= 1/4`` is unconditionally stable; smaller theta is
        more accurate (less numerical dispersion) but only conditionally stable. Default a hair
        above ``1/4``.
    boundary : {"free"}
        Free-free edges -- the iconic Chladni boundary, and the one with a closed-form 1D oracle.

    Raises
    ------
    ValueError
        Non-physical parameters (non-positive ``L``/``rho``/``fs``/``kappa``, negative ``sigma``,
        ``N < 4``), ``theta`` outside ``(0, 1]``, or an unsupported boundary.
    """

    def __init__(
        self,
        *,
        L: float,
        rho: float,
        fs: float,
        N: int,
        kappa: float,
        sigma: float = 0.0,
        theta: float = THETA_DEFAULT,
        boundary: Boundary = "free",
    ) -> None:
        if min(L, rho, fs) <= 0:
            raise ValueError("L, rho, fs must all be positive.")
        if kappa <= 0:
            raise ValueError("kappa (stiffness) must be positive.")
        if N < 4:
            raise ValueError("N must be >= 4 (need a few interior nodes for the free-free modes).")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if boundary != "free":
            raise ValueError(f"boundary must be 'free', got {boundary!r}.")

        self.L = float(L)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.kappa = float(kappa)
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.boundary: Boundary = boundary

        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)
        # Beam "Courant" number mu = kappa k / h²: the EXPLICIT-scheme stability parameter (explicit
        # needs mu <= 1/4). Reported only -- the implicit theta-scheme has no limit.
        self.mu = self.kappa * self.k / (self.h * self.h)

        # Energy-first operator: symmetric PSD stiffness K (nullspace {1, x}) + diagonal trapezoidal
        # mass W. The natural free BCs and the free-end closure are baked in (see
        # free_beam_stiffness).
        self.K, self.W = free_beam_stiffness(self.N, self.h)
        self.w: NDArray[np.float64] = self.W.diagonal()  # lumped mass weights (h, h/2 at the ends)

        # A = (1 + sigma k) W + theta k² kappa² K  (SPD; W is SPD even though K is only PSD).
        sk = self.sigma * self.k
        coeff = self.theta * self.k * self.k * self.kappa * self.kappa
        A = (1.0 + sk) * self.W + coeff * self.K
        self._lu = splu(A.tocsc())

        self.u: NDArray[np.float64] = np.zeros(self.N + 1)
        self.u_prev: NDArray[np.float64] = np.zeros(self.N + 1)
        self.n: int = 0  # completed steps

    # -- initial conditions -------------------------------------------------------------

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacement (and optional velocity).

        Uses the consistent second-order start ``u^{-1} = u^0 - k v^0 + 1/2 k² a^0`` with the
        acceleration ``a^0 = -kappa² W⁻¹ K u^0`` (from ``W u_tt = -kappa² K u``; ``W`` diagonal, so
        ``W⁻¹`` is a per-node divide), so a single eigenmode oscillates as a clean discrete cosine
        and zero initial velocity is exact to second order. There are **no clamped nodes**
        (free-free), so nothing is zeroed at the ends.
        """
        u0 = np.asarray(u0, dtype=float).copy()
        if u0.shape != (self.N + 1,):
            raise ValueError(f"u0 must have shape {(self.N + 1,)}, got {u0.shape}.")
        v0_arr = np.broadcast_to(np.asarray(v0, dtype=float), (self.N + 1,)).copy()

        accel = -(self.kappa * self.kappa) * (self.K @ u0) / self.w  # a^0 = -kappa² W⁻¹ K u^0
        self.u = u0
        self.u_prev = u0 - self.k * v0_arr + 0.5 * self.k * self.k * accel
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep via the prefactored sparse SPD solve (rolls the history)."""
        sk = self.sigma * self.k
        k2 = self.k * self.k
        kappa2 = self.kappa * self.kappa
        Lop_u = -kappa2 * (self.K @ self.u)  # -kappa² K u^n   (= W a^n)
        Lop_prev = -kappa2 * (self.K @ self.u_prev)  # -kappa² K u^{n-1}
        rhs = (
            self.W @ (2.0 * self.u - self.u_prev)
            + (1.0 - 2.0 * self.theta) * k2 * Lop_u
            + self.theta * k2 * Lop_prev
            + sk * (self.W @ self.u_prev)
        )
        u_next = self._lu.solve(rhs)
        self.u_prev = self.u
        self.u = u_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current displacement field ``u^n`` (a copy, safe to mutate/store for plotting)."""
        return self.u.copy()

    def energy(self) -> float:
        """Discrete energy ``E^n`` (Joules) for the implicit theta-scheme (bending-only
        potential).

        ``E^n = rho [ 1/2 (δ_t- u)ᵀ W (δ_t- u) + (theta/2)(P_nn + P_pp) + (1/2 - theta) P_np ]``
        with ``P(f,g) = kappa² (K f)·g >= 0`` (K positive-semidefinite). The ``h`` quadrature
        weights live inside ``W`` and ``K``. Lossless -> conserved to machine precision; lossy ->
        monotone
        decreasing.
        """
        dt_u = (self.u - self.u_prev) / self.k  # δ_t- u^n
        kinetic = 0.5 * float(dt_u @ (self.w * dt_u))  # 1/2 (δ_t- u)ᵀ W (δ_t- u)

        p_nn = self._P(self.u, self.u)
        p_pp = self._P(self.u_prev, self.u_prev)
        p_np = self._P(self.u, self.u_prev)
        potential = 0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np
        return self.rho * (kinetic + potential)

    def displacement_at(self, index: int) -> float:
        """Displacement at grid node ``index`` -- a pickup for spectral analysis."""
        return float(self.u[index])

    # -- internals ----------------------------------------------------------------------

    def _P(self, f: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """Potential bilinear form ``P(f,g) = kappa² (K f)·g`` (full-grid vectors).

        Uses the *same* matrix ``K`` as the update, so the energy identity is exact. ``K`` is
        positive-semidefinite (nullspace ``{1, x}``), hence ``P(f,f) >= 0``.
        """
        return self.kappa * self.kappa * float((self.K @ f) @ g)
