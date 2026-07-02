"""Kirchhoff plate (simply-supported **or** free rectangle) — implicit theta-scheme FDTD.

Two boundaries share one resonator (``boundary=``):

- ``"supported"`` — the simply-supported (Navier) rectangle of HANDOFF §5 model #5 (this docstring),
  built from the squared Dirichlet Laplacian ``B = L²`` (see ``docs/dev/plate-plan.md``).
- ``"free"`` — the **completely free** rectangle of model #5b (``docs/dev/plate-free-edge-plan.md``
  Part 1), the iconic *curved-Chladni* plate. Every node is a free unknown, so ``B = L²`` does not
  apply; the symmetric stiffness ``K`` and the diagonal lumped mass ``W`` are assembled **from the
  strain energy** by :func:`physsynth.core.operators2d.free_plate_stiffness` (the 2D generalisation
  of the free beam), Poisson's ratio ``nu`` re-enters, and the update is the **W-weighted** scheme
  (``W δ_tt u = -kappa² K(…)``, the beam's verbatim with 2D ``K, W``) not the ``I``-based one
  below. The free branch has no closed-form modal oracle — it is validated by the rigid-body
  nullspace, O(h²) self-convergence, and Leissa's FFFF-square frequency parameters.

The simply-supported plate (the rest of this docstring) is the
composition of the two prior 2D/4th-order advances, with no fundamentally new machinery:

- the membrane's **masked Laplacian** ``L``
  (:func:`physsynth.core.operators2d.laplacian_from_mask`),
- the stiff string's **implicit theta-scheme** + **biharmonic-by-squaring** (1D ``(δ_xx)²`` -> 2D
  ``∇⁴ = (∇²)² = L @ L``).

PDE — **bending only, no membrane tension** (transverse displacement ``u(x, y, t)``):

    u_tt = -kappa² ∇⁴u - 2 sigma u_t,    kappa² = D / rho_s
    (D = flexural rigidity, kg-free m⁴/s²)

The biharmonic ``B = L @ L`` is built from the *Dirichlet* Laplacian, so it bakes in **both**
simply-supported (Navier) conditions ``u = 0`` and ``∇²u = 0`` automatically and keeps
``sin(mπx/Lx) sin(nπy/Ly)`` an *exact* discrete eigenvector (eigenvalue ``Λ_{mn}²``). There is
**no** ``c²∇²`` wave term — the modal law
``f_{mn} = (π/2)√(D/rho_s)[(m/Lx)² + (n/Ly)²]`` is pure 4th-power.

Treating ``∇⁴`` explicitly forces a brutal ``kappa² k² / h⁴`` CFL
(``μ = kappa k / h² <= 1/4``), so
the whole spatial operator ``𝓛 = -kappa² B`` is time-averaged with a theta weight:

    δ_tt u = 𝓛 (theta u^{n+1} + (1-2 theta) u^n + theta u^{n-1}) - 2 sigma δ_t. u

which is **unconditionally stable for theta >= 1/4** (no CFL limit — coarse grids / large
timesteps
the explicit plate could not run are admissible). Rearranged, each step is one sparse SPD solve with
the constant matrix

    A = (1 + sigma k) I - theta k² 𝓛 = (1 + sigma k) I + theta k² kappa² B.

``B = L²`` is a 13-point stencil (bandwidth ~2 Nx), so — unlike the 1D pentadiagonal case — there
is
no useful banded structure and scipy has no sparse Cholesky; ``A`` (SPD) is factored once with
``scipy.sparse.linalg.splu`` and back-substituted each step.

**Energy** (theta-dependent; reduces to the cross-time form at theta = 1/4) — bending-only
potential:

    E^n = rho_s [ 1/2 ||δ_t- u^n||²
                  + (theta/2)(P_nn + P_pp) + (1/2 - theta) P_np ]

with ``P(f,g) = <-𝓛 f, g> = kappa² <B f, g> >= 0`` (B positive-definite). ``P`` is evaluated
through
the *same* matrix ``B`` used in the update, so conservation ``E^{n+1} = E^n`` is an exact algebraic
identity (machine precision lossless; monotone decreasing at ``e^{-2 sigma t}`` lossy).

> **Damping caveat (broader than the stiff string).** The theta-time-average makes
> frequency-independent loss effectively frequency-*dependent*: mode ``m`` decays at
> ``2 sigma (1 - theta Q k²)`` with ``Q = kappa² Λ²``. Because ``Q`` is 4th-power across the
> *whole*
> spectrum (no gentle ``c²p²`` term as in the stiff string), the under-damping bites mid-spectrum,
> not only the top partials. Passivity still holds unconditionally; the *rate*, not the sign, is
> wrong above low modes. Cure = frequency-dependent loss (a later model). See the plan.

Headless: NumPy + SciPy (sparse LU). No I/O, no plotting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from .operators2d import (
    AiryStressSolver,
    VonKarmanBracket,
    embed,
    free_plate_stiffness,
    laplacian_from_mask,
    rectangle_mask,
)

Boundary = Literal["supported", "free"]

# theta below 1/4 is only conditionally stable; theta in (0, 1] keeps A SPD (genuinely implicit).
# Default a hair above 1/4 (accuracy-first per the plan) so the energy has a small positivity margin
# while staying near the minimal-dispersion theta = 1/4 -- inherited from the stiff string.
THETA_DEFAULT = 0.28


class Plate:
    """A discretized Kirchhoff plate resonator (implicit theta-scheme, simply-supported rectangle).

    Parameters
    ----------
    Lx, Ly : float
        Rectangle side lengths (m). ``Ly`` is snapped to an integer number of cells so cells stay
        square; the snapped value is stored back on :attr:`Ly`.
    kappa : float
        Stiffness coefficient ``kappa = sqrt(D / rho_s)`` (units m²/s), the single bending
        parameter
        (Poisson's ratio drops out for simply-supported edges). Larger ``kappa`` -> higher modes.
    rho : float
        Areal density ``rho_s`` (kg/m²); scales the energy (Joules) but not the frequencies.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``. **No CFL limit** (unconditionally stable for
        ``theta >= 1/4``) -- coarse grids / large timesteps the explicit plate could not run are
        fine.
    N : int
        Number of spatial segments along x; spacing ``h = Lx/N`` (square cells). The bounding-box
        edge nodes are the clamped Dirichlet rim; the interior nodes are the unknowns.
    sigma : float
        Frequency-independent loss (>= 0) for the ``-2 sigma u_t`` term. ``0`` -> lossless. See the
        broad damping caveat in the module docstring.
    theta : float
        Time-averaging weight in ``(0, 1]``. ``>= 1/4`` is unconditionally stable; smaller theta is
        more accurate (less numerical dispersion) but only conditionally stable. Default a hair
        above
        ``1/4``.
    boundary : {"supported", "free"}
        ``"supported"`` = simply-supported (Navier) edges (the closed-form-oracle case).
        ``"free"`` = completely free edges -- the iconic curved-Chladni plate (model #5b), assembled
        energy-first; ``nu`` re-enters and the W-weighted update is used.
    nu : float
        Poisson's ratio, in ``(-1, 1/2)``. **Only used for** ``boundary="free"`` (it drops out of
        the simply-supported modal law). Default ``0.3`` (matches the Leissa FFFF tables).

    Raises
    ------
    ValueError
        Non-physical parameters (negative kappa/rho/loss, non-positive Lx/Ly/fs, ``N < 2``),
        ``theta`` outside ``(0, 1]``, ``nu`` outside ``(-1, 1/2)``, or an unsupported boundary.
    """

    def __init__(
        self,
        *,
        Lx: float,
        Ly: float,
        kappa: float,
        rho: float,
        fs: float,
        N: int,
        sigma: float = 0.0,
        theta: float = THETA_DEFAULT,
        boundary: Boundary = "supported",
        nu: float = 0.3,
    ) -> None:
        if min(Lx, Ly, fs) <= 0:
            raise ValueError("Lx, Ly, fs must all be positive.")
        if kappa <= 0:
            raise ValueError("kappa (stiffness) must be positive.")
        if rho <= 0:
            raise ValueError("rho (areal density) must be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu (Poisson's ratio) must be in (-1, 1/2), got {nu}.")
        if boundary not in ("supported", "free"):
            raise ValueError(f"boundary must be 'supported' or 'free', got {boundary!r}.")

        self.kappa = float(kappa)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.nu = float(nu)
        self.boundary: Boundary = boundary

        self.k = 1.0 / self.fs
        self.h = float(Lx) / self.N
        Ny = max(int(round(float(Ly) / self.h)), 1)
        self.Lx = float(Lx)
        self.Ly = Ny * self.h  # snapped so cells are square
        xs = np.linspace(0.0, self.Lx, self.N + 1)
        ys = np.linspace(0.0, self.Ly, Ny + 1)
        self.X, self.Y = np.meshgrid(xs, ys)

        # Plate "Courant" number mu = kappa k / h^2: the explicit-scheme stability parameter
        # (explicit needs mu <= 1/4). Reported only -- the implicit scheme has no limit.
        self.mu = self.kappa * self.k / (self.h * self.h)

        sk = self.sigma * self.k
        coeff = self.theta * self.k * self.k * self.kappa * self.kappa

        if self.boundary == "supported":
            self.mask = rectangle_mask(self.N, Ny)
            # Masked Dirichlet Laplacian L (symmetric, negative-definite) and biharmonic B = L^2
            # (symmetric, positive-definite); B carries the simply-supported conditions for free.
            self.L, self.index_map = laplacian_from_mask(self.mask, self.h)
            self.B = (self.L @ self.L).tocsr()
            self.n_live = self.B.shape[0]
            if self.n_live < 1:
                raise ValueError("the plate has no interior (live) nodes; refine the grid.")
            # A = (1 + sigma k) I + theta k^2 kappa^2 B (SPD, 13-point, constant -> factor once).
            A = (1.0 + sk) * sparse.identity(self.n_live, format="csc") + coeff * self.B
        else:  # free: energy-first stiffness K + diagonal lumped mass W (W-weighted update)
            self.mask = np.ones((Ny + 1, self.N + 1), dtype=bool)  # every node is a free unknown
            self.K, self.W, self.index_map = free_plate_stiffness(self.N, Ny, self.h, self.nu)
            self.w: NDArray[np.float64] = self.W.diagonal()  # lumped area mass (h², h²/2, h²/4)
            self.n_live = self.K.shape[0]
            # A = (1 + sigma k) W + theta k^2 kappa^2 K (SPD because W is, though K is only PSD).
            A = (1.0 + sk) * self.W + coeff * self.K

        self._lu = splu(A.tocsc())

        self.u: NDArray[np.float64] = np.zeros(self.n_live)
        self.u_prev: NDArray[np.float64] = np.zeros(self.n_live)
        self.n: int = 0  # completed steps

    # -- initial conditions -------------------------------------------------------------

    def to_live(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """Select the live-node values from a full 2D ``field`` (shape ``mask.shape``)."""
        field = np.asarray(field, dtype=float)
        if field.shape != self.mask.shape:
            raise ValueError(f"field must have shape {self.mask.shape}, got {field.shape}.")
        return field[self.mask]

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacement (and optional velocity).

        ``u0`` may be a full 2D field (shape ``mask.shape``) or a flat live-node vector
        (length ``n_live``). Uses the consistent second-order start
        ``u^{-1} = u^0 - k v^0 + 1/2 k² a^0`` with the acceleration ``a^0 = 𝓛 u^0`` so a single
        eigenmode oscillates as a clean discrete cosine and zero initial velocity is exact to second
        order. For ``"supported"`` ``a^0 = -kappa² B u^0`` (dead rim nodes stay clamped); for
        ``"free"`` ``a^0 = -kappa² W⁻¹ K u^0`` (``W`` diagonal, so ``W⁻¹`` is a per-node divide; no
        clamped nodes -- every node is free).
        """
        u0 = np.asarray(u0, dtype=float)
        if u0.shape == self.mask.shape:
            u0 = u0[self.mask]
        elif u0.shape != (self.n_live,):
            raise ValueError(
                f"u0 must have shape {self.mask.shape} (full field) or {(self.n_live,)} (live), "
                f"got {u0.shape}."
            )
        u0 = u0.copy()

        if np.isscalar(v0) or np.asarray(v0).shape == ():
            v0_live = np.full(self.n_live, float(v0))
        else:
            v0 = np.asarray(v0, dtype=float)
            v0_live = v0[self.mask] if v0.shape == self.mask.shape else v0

        half_k2_kappa2 = 0.5 * self.k * self.k * self.kappa * self.kappa
        if self.boundary == "supported":
            accel_term = half_k2_kappa2 * (self.B @ u0)  # -1/2 k² a^0 = +1/2 k² kappa² B u^0
        else:
            accel_term = half_k2_kappa2 * (self.K @ u0) / self.w  # +1/2 k² kappa² W⁻¹ K u^0
        self.u = u0
        self.u_prev = u0 - self.k * v0_live - accel_term
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep via the prefactored sparse SPD solve (rolls the history).

        ``"supported"`` uses the ``I``-based update (the SS mass factors out); ``"free"`` uses the
        **W-weighted** update ``W δ_tt u = -kappa² K(…)`` (the free beam's scheme in 2D), so ``W``
        multiplies the inertial terms on the RHS.
        """
        sk = self.sigma * self.k
        k2 = self.k * self.k
        kappa2 = self.kappa * self.kappa
        if self.boundary == "supported":
            Lop_u = -kappa2 * (self.B @ self.u)  # 𝓛 u^n
            Lop_prev = -kappa2 * (self.B @ self.u_prev)  # 𝓛 u^{n-1}
            rhs = (
                2.0 * self.u
                + (1.0 - 2.0 * self.theta) * k2 * Lop_u
                - self.u_prev
                + self.theta * k2 * Lop_prev
                + sk * self.u_prev
            )
        else:  # free: W-weighted inertial terms (W u_tt = -kappa² K u)
            Lop_u = -kappa2 * (self.K @ self.u)  # = W a^n
            Lop_prev = -kappa2 * (self.K @ self.u_prev)
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
        """Current displacement field ``u^n`` as a full 2D array (dead nodes are 0)."""
        return embed(self.u, self.index_map)

    def energy(self) -> float:
        """Discrete energy ``E^n`` (Joules) for the implicit theta-scheme (bending-only potential).

        ``E^n = rho_s [ 1/2 ||δ_t- u||²_W + (theta/2)(P_nn + P_pp) + (1/2 - theta) P_np ]`` with
        ``P(f,g) = <-𝓛 f, g> >= 0``. For ``"supported"`` the lumped mass is the scalar ``h²`` and
        ``P = kappa² h² <B f, g>``; for ``"free"`` the mass is the diagonal ``W`` (so the kinetic
        norm is W-weighted) and ``P = kappa² <K f, g>`` (``h²`` weights live inside ``K``/``W``).
        Both use the *same* matrix as the update, so lossless -> conserved to machine precision;
        lossy -> monotone decreasing.
        """
        dt_u = (self.u - self.u_prev) / self.k  # delta_t- u^n
        if self.boundary == "supported":
            kinetic = 0.5 * (self.h * self.h) * float(np.dot(dt_u, dt_u))
        else:
            kinetic = 0.5 * float(np.dot(dt_u, self.w * dt_u))  # 1/2 (δ_t- u)ᵀ W (δ_t- u)

        p_nn = self._P(self.u, self.u)
        p_pp = self._P(self.u_prev, self.u_prev)
        p_np = self._P(self.u, self.u_prev)
        potential = 0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np
        return self.rho * (kinetic + potential)

    def displacement_at(self, index: int) -> float:
        """Displacement at flat live-node ``index`` -- a pickup for spectral analysis."""
        return float(self.u[index])

    def pickup_index_at(self, x: float, y: float) -> int:
        """Flat live-node index nearest the physical point ``(x, y)`` (for placing a pickup)."""
        live = self.index_map >= 0
        xs = self.X[live]
        ys = self.Y[live]
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        return int(np.argmin(d2))

    # -- internals ----------------------------------------------------------------------

    def _P(self, f: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """Potential bilinear form ``P(f,g) = <-𝓛 f, g> >= 0`` (live vectors).

        ``"supported"``: ``kappa² h² (B f)·g`` (``B`` positive-definite). ``"free"``:
        ``kappa² (K f)·g`` (``K`` positive-semidefinite, the ``h²`` weights baked in). Each uses the
        *same* matrix as its update, so the energy identity is exact and ``P(f,f) >= 0``.
        """
        if self.boundary == "supported":
            return self.kappa * self.kappa * self.h * self.h * float(np.dot(self.B @ f, g))
        return self.kappa * self.kappa * float(np.dot(self.K @ f, g))


class VKPlate:
    """Von Kármán **nonlinear** plate — simply-supported rectangle (HANDOFF §5 model #6, Part 3).

    The first resonator with genuine nonlinear coupling and **no analytic modal oracle**: the
    transverse displacement ``w`` is stretched into an in-plane stress state carried by the Airy
    stress function ``F``, which in turn stiffens ``w`` (amplitude-dependent pitch — the *hardening*
    pitch glide). Two coupled fields, one elliptic ``F``-solve per iteration, energy conservation
    replacing the closed-form spectrum as *the* correctness test. See
    ``docs/dev/von-karman-plate-plan.md`` (Part 3) and ``[[von-karman-plate-state]]``.

    **The Föppl–von Kármán system (simply-supported, ``w = 0, Δw = 0``):**

        ρ_s w_tt = -D ∇⁴w + l(w, F) - 2 ρ_s σ w_t          (transverse)
        ∇⁴F      = -(E e / 2) l(w, w)                       (in-plane, elliptic — solved for F)

    with the Monge–Ampère bracket ``l`` (:class:`VonKarmanBracket`, Part 1), the clamped ``F``-solve
    ``B_F`` (:class:`AiryStressSolver`, Part 2, in-plane BC ``F = 0, F,n = 0``), ρ_s = ρ e the areal
    density, ``D = E e³ / (12(1-ν²))`` the flexural rigidity, ``κ = √(D/ρ_s)`` the bending speed and
    ``Y_mem = E e`` the membrane coefficient. The onset of nonlinearity is at ``w ≈ e`` (thickness),
    so ``e`` is a *physically meaningful* input (the linear models used ``κ`` alone): the material
    surface is ``(E, e, ν, ρ)`` and ``κ, D, Y_mem, ρ_s`` are derived.

    **The conservative scheme (exactly energy-conserving to machine precision).** The bending is
    model #5's implicit θ-scheme verbatim; the membrane coupling is time-averaged so the discrete
    energy telescopes. With ``μ_{t·}g = (g^{n+1} + g^{n-1})/2`` and ``F^m`` solved from ``w^m``,

        ρ_s δ_tt w = -D B(θw^{n+1}+(1-2θ)w^n+θw^{n-1}) - 2ρ_s σ δ_{t·}w + l(μ_{t·}w, μ_{t·}F)

    The coupling ``⟨l(μ_{t·}w, μ_{t·}F), w^{n+1}-w^{n-1}⟩`` telescopes **exactly** into
    ``-(H_mem^{n+1} - H_mem^{n-1})`` — a discrete replay of the continuous cancellation
    ``⟨l(w,w_t), F⟩ = ⟨l(w,F), w_t⟩`` — **iff** the bracket is triple self-adjoint (the Part-1 gate)
    and the fields vanish on the rim (which ``μ`` and difference of rim-vanishing fields preserve).
    ``H_mem^m = (1/(2Y)) (F^m)ᵀ B_F F^m`` (:meth:`AiryStressSolver.laplacian_norm_sq`); the
    ``Wa``-vs-``h²`` weighting mismatch between ``B_F`` and the bracket's inner product is harmless
    because ``F = 0`` on the rim, where they differ.

    **Linearly-implicit? No — a fixed-point iteration.** Exact conservation needs
    ``F^{n+1} = F(w^{n+1})`` (quadratic in the unknown ``w^{n+1}``), so the step is genuinely
    implicit. It is solved by Picard iteration on the *prefactored* operators (predictor
    ``w^{n+1}_{(0)} = 2 w^n - w^{n-1}``; each sweep = one ``B_F``-solve for ``F^{n+1}`` then one
    ``A``-solve for ``w^{n+1}``, both back-substitutions), converging on ``‖Δw‖/‖w‖ ≤ couple_tol``.
    Because exact conservation holds only at the fixed point,
    ``couple_tol`` is tied to the drift target (default ``1e-13``); a self-certifying test confirms
    the energy drift falls with ``couple_tol``. The cross-time ``ψ`` (SAV-style) variant that would
    make the step linear is left as a future optimisation (it trades the prefactored ``A`` for a
    dense per-step operator — no real saving, and its conservation is unverified).

    **Reported energy (odd-even-safe).** Kinetic + bending telescope at half-steps ``n±1/2``; the
    membrane at integer steps. Reporting the raw sum gives a two-step (odd/even) invariant that can
    show a spurious energy oscillation. :meth:`energy` therefore averages the membrane to the half
    step: ``E^{n+1/2} = E_lin^{n+1/2} + ½(H_mem(F^{n+1})+H_mem(F^n))`` (one-step conserved).
    ``F`` is cached at both stored levels (:attr:`F`, :attr:`F_prev`).

    Parameters
    ----------
    Lx, Ly : float
        Rectangle side lengths (m). ``Ly`` is snapped to an integer number of square cells; the
        snapped value is stored back on :attr:`Ly`.
    E : float
        Young's modulus (Pa).
    e : float
        Plate thickness (m) — the amplitude scale of the nonlinearity (onset at ``w ≈ e``).
    nu : float
        Poisson's ratio, in ``(-1, 1/2)``. Enters ``D`` (hence ``κ``); the SS bending law is
        otherwise ν-independent. Retained for the free-edge follow-on (Part 2).
    rho : float
        Volumetric density (kg/m³), stored as :attr:`rho_v`; areal density ``ρ_s = ρ e`` on
        :attr:`rho_s`. (Distinct from :class:`Plate`, whose ``rho`` *is* the areal density.)
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``. **Oversample around the nonlinearity** (HANDOFF §8)
        — the quadratic/cubic coupling folds HF energy down; render high.
    N : int
        Segments along x; spacing ``h = Lx/N`` (square cells). Bounding-box edge nodes are the
        clamped Dirichlet rim; interior nodes are the unknowns.
    sigma : float
        Frequency-independent loss (>= 0). ``0`` -> lossless (the energy-drift regime).
    theta : float
        θ-scheme weight in ``(0, 1]``; ``>= 1/4`` unconditionally stable for the *linear* part.
        Stability of the coupled scheme is not inherited — :meth:`energy` non-negativity is checked.
    nonlinear : bool
        ``True`` (default) runs the coupled von Kármán scheme. ``False`` disables the coupling
        entirely (no ``F``-solve, single ``A``-solve) and reproduces model #5's simply-supported
        plate **bit-for-bit** (the regression path).
    couple_tol : float
        Picard relative-increment tolerance ``‖Δw‖/‖w‖``. Tied to the drift target (default 1e-13).
    couple_max_iter : int
        Safety cap on Picard sweeps per step (default 50). In the gate regime ``w ≳ e`` a handful
        suffice; the strong-cascade regime ``w ≫ e`` may not converge (qualitative, not a gate).

    Raises
    ------
    ValueError
        Non-physical parameters, ``theta`` outside ``(0, 1]``, ``nu`` outside ``(-1, 1/2)``, or a
        grid with no interior nodes.
    """

    def __init__(
        self,
        *,
        Lx: float,
        Ly: float,
        E: float,
        e: float,
        nu: float,
        rho: float,
        fs: float,
        N: int,
        sigma: float = 0.0,
        theta: float = THETA_DEFAULT,
        nonlinear: bool = True,
        couple_tol: float = 1e-13,
        couple_max_iter: int = 50,
    ) -> None:
        if min(Lx, Ly, fs) <= 0:
            raise ValueError("Lx, Ly, fs must all be positive.")
        if E <= 0:
            raise ValueError("E (Young's modulus) must be positive.")
        if e <= 0:
            raise ValueError("e (thickness) must be positive.")
        if rho <= 0:
            raise ValueError("rho (density) must be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu (Poisson's ratio) must be in (-1, 1/2), got {nu}.")
        if couple_tol <= 0:
            raise ValueError("couple_tol must be positive.")
        if couple_max_iter < 1:
            raise ValueError("couple_max_iter must be >= 1.")

        self.E = float(E)
        self.e = float(e)
        self.nu = float(nu)
        self.rho_v = float(rho)  # volumetric density (kg/m^3); cf. Plate.rho = areal density
        self.rho_s = self.rho_v * self.e  # areal density
        self.D = self.E * self.e**3 / (12.0 * (1.0 - self.nu**2))  # flexural rigidity
        self.kappa = float(np.sqrt(self.D / self.rho_s))  # bending speed sqrt(D/rho_s)
        self.Y_mem = self.E * self.e  # membrane coefficient E*e
        self.fs = float(fs)
        self.N = int(N)
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.nonlinear = bool(nonlinear)
        self.couple_tol = float(couple_tol)
        self.couple_max_iter = int(couple_max_iter)

        self.k = 1.0 / self.fs
        self.h = float(Lx) / self.N
        Ny = max(int(round(float(Ly) / self.h)), 1)
        self.Ny = Ny
        self.Lx = float(Lx)
        self.Ly = Ny * self.h  # snapped so cells are square
        xs = np.linspace(0.0, self.Lx, self.N + 1)
        ys = np.linspace(0.0, self.Ly, Ny + 1)
        self.X, self.Y = np.meshgrid(xs, ys)

        # Plate "Courant" number mu = kappa k / h^2 (explicit-scheme parameter; reported only).
        self.mu = self.kappa * self.k / (self.h * self.h)

        # Simply-supported bending operators (model #5 verbatim): B = L^2,
        # A = (1+sk) I + theta k^2 kappa^2 B.
        self.mask = rectangle_mask(self.N, Ny)
        self.L, self.index_map = laplacian_from_mask(self.mask, self.h)
        self.B = (self.L @ self.L).tocsr()
        self.n_live = self.B.shape[0]
        if self.n_live < 1:
            raise ValueError("the plate has no interior (live) nodes; refine the grid.")
        sk = self.sigma * self.k
        coeff = self.theta * self.k * self.k * self.kappa * self.kappa
        A = (1.0 + sk) * sparse.identity(self.n_live, format="csc") + coeff * self.B
        self._lu = splu(A.tocsc())

        # Nonlinear pieces: the shared bracket (F-source and coupling) and the clamped Airy solve.
        self.bracket = VonKarmanBracket(self.N, Ny, self.h)
        self.airy = AiryStressSolver(self.N, Ny, self.h)
        self._mask_flat = self.mask.ravel()
        self.n_nodes = self.mask.size

        self.u: NDArray[np.float64] = np.zeros(self.n_live)  # w^n (live/interior nodes)
        self.u_prev: NDArray[np.float64] = np.zeros(self.n_live)  # w^{n-1}
        self.F: NDArray[np.float64] = np.zeros(self.n_nodes)  # F(w^n), full grid (rim 0)
        self.F_prev: NDArray[np.float64] = np.zeros(self.n_nodes)  # F(w^{n-1}), full grid
        self.n: int = 0  # completed steps
        self.n_iters: int = 0  # Picard sweeps used by the last step (diagnostic)
        self.converged: bool = True  # did the last step's Picard iteration reach couple_tol?
        self.last_residual: float = 0.0  # its final ‖Δw‖/‖w‖ (watch this in the cascade regime)

    # -- grid <-> full-grid seam --------------------------------------------------------

    def _to_full(self, u_live: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scatter a live-node vector to a full-grid vector (rim held at 0) for bracket/Airy."""
        return embed(u_live, self.index_map).ravel()

    def _to_live(self, full_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Restrict a full-grid vector to the live (interior) nodes (C-order, per index_map)."""
        return full_vec[self._mask_flat]

    def _airy_F(self, w_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``∇⁴F = -(Y/2) l(w, w)`` for the stress function from a full-grid ``w`` (rim 0)."""
        return self.airy.solve(-0.5 * self.Y_mem * self.bracket(w_full, w_full))

    # -- initial conditions -------------------------------------------------------------

    def to_live(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """Select the live-node values from a full 2D ``field`` (shape ``mask.shape``)."""
        field = np.asarray(field, dtype=float)
        if field.shape != self.mask.shape:
            raise ValueError(f"field must have shape {self.mask.shape}, got {field.shape}.")
        return field[self.mask]

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacement (and optional velocity), a consistent 2nd-order start.

        ``u0`` may be a full 2D field (shape ``mask.shape``) or a flat live vector (length
        ``n_live``). Uses ``w^{-1} = w^0 - k v^0 + ½ k² a^0`` with the **full** acceleration
        ``a^0 = -κ² B w^0 + (1/ρ_s) l(w^0, F^0)`` (the coupling included, so a struck plate starts
        cleanly even at large amplitude); for ``nonlinear=False`` the coupling term is dropped and
        the start is model #5's exactly. Also seeds the cached ``F(w^0), F(w^{-1})``.
        """
        u0 = np.asarray(u0, dtype=float)
        if u0.shape == self.mask.shape:
            u0 = u0[self.mask]
        elif u0.shape != (self.n_live,):
            raise ValueError(
                f"u0 must have shape {self.mask.shape} (full field) or {(self.n_live,)} (live), "
                f"got {u0.shape}."
            )
        u0 = u0.copy()

        if np.isscalar(v0) or np.asarray(v0).shape == ():
            v0_live = np.full(self.n_live, float(v0))
        else:
            v0 = np.asarray(v0, dtype=float)
            v0_live = v0[self.mask] if v0.shape == self.mask.shape else v0

        # Bending part of w^{-1}, computed with model #5's exact arithmetic so the nonlinear=False
        # path is bit-identical to Plate.  accel_term = -1/2 k^2 a0_bending = 1/2 k^2 kappa^2 B w0.
        accel_term = (0.5 * self.k * self.k * self.kappa * self.kappa) * (self.B @ u0)
        self.u = u0
        self.u_prev = u0 - self.k * v0_live - accel_term
        if self.nonlinear:
            u0_full = self._to_full(u0)
            F0 = self._airy_F(u0_full)
            coupling0 = self._to_live(self.bracket(u0_full, F0)) / self.rho_s
            self.u_prev += (0.5 * self.k * self.k) * coupling0  # + 1/2 k^2 coupling/rho_s
            self.F = F0
            self.F_prev = self._airy_F(self._to_full(self.u_prev))
        else:
            self.F = np.zeros(self.n_nodes)
            self.F_prev = np.zeros(self.n_nodes)
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def _linear_rhs(self) -> NDArray[np.float64]:
        """Model #5's simply-supported θ-scheme right-hand side (bending + inertia + loss)."""
        sk = self.sigma * self.k
        k2 = self.k * self.k
        kappa2 = self.kappa * self.kappa
        lop_u = -kappa2 * (self.B @ self.u)  # 𝓛 w^n
        lop_prev = -kappa2 * (self.B @ self.u_prev)  # 𝓛 w^{n-1}
        return (
            2.0 * self.u
            + (1.0 - 2.0 * self.theta) * k2 * lop_u
            - self.u_prev
            + self.theta * k2 * lop_prev
            + sk * self.u_prev
        )

    def step(self) -> None:
        """Advance one timestep.

        ``nonlinear=False``: one prefactored SS solve (model #5). ``nonlinear=True``: Picard iterate
        the conservative coupled scheme — predictor ``2 w^n - w^{n-1}``, then sweeps of one
        ``B_F``-solve (``F^{n+1}`` from ``w^{n+1}_{(j)}``) and one ``A``-solve (``w^{n+1}_{(j+1)}``
        with the ``μ``-averaged coupling ``+ k² l(μ_{t·}w, μ_{t·}F)/ρ_s`` on the RHS), until
        ``‖Δw‖/‖w‖ <= couple_tol``. Rolls the history and the cached ``F``.
        """
        rhs_lin = self._linear_rhs()
        if not self.nonlinear:
            w_next = self._lu.solve(rhs_lin)
            self.u_prev, self.u = self.u, w_next
            self.n += 1
            self.n_iters = 1
            self.converged = True
            self.last_residual = 0.0
            return

        k2 = self.k * self.k
        w_prev_full = self._to_full(self.u_prev)  # w^{n-1}
        F_prev_full = self.F_prev  # F^{n-1} (cached)
        w_j = 2.0 * self.u - self.u_prev  # predictor w^{n+1}_(0)
        f_new_full = self.F  # fallback (unused once the loop runs)
        self.n_iters = 0
        self.converged = False
        for sweep in range(1, self.couple_max_iter + 1):
            self.n_iters = sweep
            w_j_full = self._to_full(w_j)
            f_new_full = self._airy_F(w_j_full)  # F^{n+1}_(j)
            w_avg = 0.5 * (w_j_full + w_prev_full)  # μ_{t·}w
            f_avg = 0.5 * (f_new_full + F_prev_full)  # μ_{t·}F
            coupling = self._to_live(self.bracket(w_avg, f_avg))
            rhs = rhs_lin + (k2 / self.rho_s) * coupling
            w_next = self._lu.solve(rhs)
            incr = float(np.linalg.norm(w_next - w_j))
            scale = float(np.linalg.norm(w_next))
            w_j = w_next
            self.last_residual = incr / max(scale, 1e-30)
            if self.last_residual <= self.couple_tol:
                self.converged = True
                break
        # Roll: F cache always tracks (u, u_prev).  Old F = F^n -> F_prev; f_new = F^{n+1} -> F.
        self.F_prev = self.F
        self.F = f_new_full
        self.u_prev = self.u
        self.u = w_j
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current displacement field ``w^n`` as a full 2D array (rim nodes are 0)."""
        return embed(self.u, self.index_map)

    @property
    def stress_field(self) -> NDArray[np.float64]:
        """Current Airy stress function ``F(w^n)`` as a full 2D array (rim 0)."""
        return self.F.reshape(self.mask.shape)

    def _membrane_energy(self, f_full: NDArray[np.float64]) -> float:
        """Membrane potential ``(1/2Y) Fᵀ B_F F = (1/2Y)‖∇²F‖²`` (>= 0) for a full-grid ``F``."""
        return self.airy.laplacian_norm_sq(f_full) / (2.0 * self.Y_mem)

    def membrane_energy(self) -> float:
        """Half-step membrane energy ``½(H_mem(F^{n+1})+H_mem(F^n))`` (>= 0; 0 if linear)."""
        if not self.nonlinear:
            return 0.0
        return 0.5 * (self._membrane_energy(self.F) + self._membrane_energy(self.F_prev))

    def linear_energy(self) -> float:
        """Kinetic + bending energy ``E_lin^{n+1/2}`` (Joules) — model #5's θ-scheme energy."""
        dt_u = (self.u - self.u_prev) / self.k  # δ_{t-} w^n
        kinetic = 0.5 * (self.h * self.h) * float(np.dot(dt_u, dt_u))
        p_nn = self._P(self.u, self.u)
        p_pp = self._P(self.u_prev, self.u_prev)
        p_np = self._P(self.u, self.u_prev)
        potential = 0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np
        return self.rho_s * (kinetic + potential)

    def energy(self) -> float:
        """Total discrete energy ``𝓔^{n+1/2}`` (Joules): ``E_lin + ½(H_mem(F^{n+1}) + H_mem(F^n))``.

        Lossless (``sigma = 0``) and converged Picard -> conserved to machine precision; lossy ->
        monotone non-increasing. The membrane term is averaged to the half step to avoid a spurious
        odd-even oscillation (see the class docstring).
        """
        return self.linear_energy() + self.membrane_energy()

    def displacement_at(self, index: int) -> float:
        """Displacement at flat live-node ``index`` -- a pickup for spectral analysis."""
        return float(self.u[index])

    def pickup_index_at(self, x: float, y: float) -> int:
        """Flat live-node index nearest the physical point ``(x, y)`` (for placing a pickup)."""
        live = self.index_map >= 0
        xs = self.X[live]
        ys = self.Y[live]
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        return int(np.argmin(d2))

    # -- internals ----------------------------------------------------------------------

    def _P(self, f: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """Bending potential form ``P(f,g) = κ² h² (B f)·g >= 0`` (live vectors; model #5's)."""
        return self.kappa * self.kappa * self.h * self.h * float(np.dot(self.B @ f, g))
