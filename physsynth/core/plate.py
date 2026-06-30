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

from .operators2d import embed, free_plate_stiffness, laplacian_from_mask, rectangle_mask

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
