"""Acoustic bore — the 1D air column of a wind instrument (clarinet, flute, ...).

The first **acoustic** resonator of the ``exciter -> resonator -> body`` abstraction and the opening
of the wind/brass breadth leg (HANDOFF §12.A). It models the pressure/flow wave in a tube of
(possibly varying) cross-section ``S(x)`` — Webster's horn equation — with an energy-conserving
staggered finite-difference scheme, the acoustic analogue of the ideal string.

**State on a staggered grid (a Yee cell in 1D).** Acoustic pressure ``p`` lives at the ``N + 1``
integer nodes ``x_l = l h``; volume velocity ``U = S v`` (cross-section times particle velocity)
lives at the ``N`` half-nodes ``x_{l+1/2}`` *between* them, and half a timestep offset in time. The
two first-order conservation laws (lossless) are

    dp/dt = -(rho0 c0^2 / S) dU/dx ,        (continuity / mass)
    dU/dt = -(S / rho0)      dp/dx ,        (momentum / Euler)

discretized as the leapfrog

    U_{l+1/2}^{n+1/2} = U_{l+1/2}^{n-1/2} - (k S_{l+1/2} / (rho0 h)) (p_{l+1}^n - p_l^n),
    p_l^{n+1}         = p_l^n - (k rho0 c0^2 / (w_l S_l)) (U_{l+1/2}^{n+1/2} - U_{l-1/2}^{n+1/2}).

Pressure updates from the *divergence* of the velocity; velocity updates from the *gradient* of the
pressure — the same difference matrix ``G`` and its transpose. That adjoint pairing is exactly what
makes the discrete energy conserve (below). ``lambda = c0 k / h <= 1`` (CFL); ``lambda = 1`` is the
dispersionless sweet spot (the closed-open tube then reflects exactly).

**Energy-first boundaries (the free-beam lesson, again).** With pressure DOFs carrying the
**trapezoidal** node weight ``w_l`` (``h`` interior, ``h/2`` at a wall), the discrete energy

    E^n = 1/2 sum_l (w_l S_l / (rho0 c0^2)) (p_l^n)^2
        + 1/2 sum_j (h rho0 / S_{j})     U_j^{n+1/2} U_j^{n-1/2}

telescopes to **zero** change per step — algebraically, independent of the weights — because
continuity uses ``G^T`` against the same ``G`` momentum uses. The *velocity* term is the
**cross-time product** ``U^{n+1/2} U^{n-1/2}`` of the staggered variable (never the same-time
square): the same "do not collapse the two time levels" trick as the string's potential term, and
the reason a lossless run conserves to machine precision. A **rigid/closed** wall (``U = 0`` just
outside) needs *no* ghost stencil — it is the ``h/2`` half-cell at that node, which the trapezoidal
weight supplies, mirroring the free beam's end masses. An **open** (pressure-release) end pins
``p = 0`` (Dirichlet); it carries no energy and its flux ``p U = 0``, so an ideal open end is
**lossless** (perfect, sign-flipped reflection — it radiates nothing yet; the passively-lossy bell
impedance is the next batch).

**The clarinet signature.** A **closed-open** cylinder resonates at the **odd** harmonics
``f_n = (2n - 1) c0 / (4 L)`` (quarter-wave: pressure antinode at the reed wall, node at the open
end) — the model-specific correctness oracle here, the way stick-slip was for the bow and Chladni
for the plate. An **open-open** tube gives the full series ``f_n = n c0 / (2 L)``.

**Loss / passivity.** An optional viscous term ``-2 sigma U`` (a **frequency-independent** drag on
the air column) damps only the velocity; the continuum balance
``dE/dt = -[p U]_boundary - 2 sigma int (rho0/S) U^2 <= 0`` is monotone, so ``sigma > 0`` makes the
discrete energy decrease monotonically (passivity) and a single mode decays at ``2 sigma``.
``sigma = 0`` is bit-for-bit the lossless scheme. This is a **placeholder** loss for the passivity
test — the same role the ideal string's flat ``sigma`` played before model #3 — **not** the
physical viscothermal wall loss (which scales as ``sqrt(omega)``, Zwikker–Kosten); realistic
frequency-dependent tube losses, and the passively-lossy radiating bell, come later.

**Webster from day one.** ``S(x)`` is carried as node/segment arrays throughout, so a cylinder
(constant ``S``) is just the constant case; a cone or flare (a bell, a saxophone) is a *different
area profile*, not a different solver.

Headless: NumPy only (SciPy sparse for the modal-oracle operator).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

__all__ = ["Bore", "RHO0_AIR", "C0_AIR"]

# Ambient air (matches physsynth.core.radiation so the bore and its radiation load agree).
RHO0_AIR = 1.2041  # kg/m^3
C0_AIR = 343.0     # m/s

End = Literal["closed", "open"]
# Per-end boundary: one value applies to both ends; a (left, right) tuple sets them independently.
# A clarinet is ("closed", "open") — rigid mouthpiece wall, open bell.
BoundarySpec = End | tuple[End, End]

_LAMBDA_TOL = 1e-12


class Bore:
    """A discretized acoustic tube (Webster horn equation) — staggered p/U leapfrog.

    Implements the engine's ``Resonator`` protocol (``step``/``energy``/``state``/``k``/
    ``displacement_at``), where ``state`` and the pickup are the **pressure** field.

    Parameters
    ----------
    L : float
        Tube length (m).
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``.
    N : int
        Number of segments; the grid has ``N + 1`` pressure nodes, spacing ``h = L/N``, and ``N``
        interleaved volume-velocity half-nodes.
    radius : float
        Bore radius (m) for a cylinder; the cross-section is ``S = pi radius^2``. Constant here (the
        cylinder case); a cone/flare is the same solver with a non-constant area profile.
    boundary : {"closed", "open"} or (left, right) tuple
        Per-end termination. ``"closed"`` is a rigid wall (``U = 0``, a pressure antinode);
        ``"open"`` is pressure-release (``p = 0``, a pressure node). Default ``("closed", "open")``
        — the clarinet (odd-harmonic) configuration.
    sigma : float
        Viscous loss coefficient (>= 0) for the ``-2 sigma U`` drag on the air column. ``0`` ->
        lossless (energy conserved); ``> 0`` -> energy decreases monotonically (passivity).
    rho0, c0 : float
        Air density (kg/m^3) and sound speed (m/s). Default ambient air.

    Raises
    ------
    ValueError
        Non-physical parameters, an unknown boundary token, or CFL ``lambda = c0 k / h > 1``.
    """

    def __init__(
        self,
        *,
        L: float,
        fs: float,
        N: int,
        radius: float = 0.008,
        boundary: BoundarySpec = ("closed", "open"),
        sigma: float = 0.0,
        rho0: float = RHO0_AIR,
        c0: float = C0_AIR,
    ) -> None:
        if min(L, fs, radius, rho0, c0) <= 0:
            raise ValueError("L, fs, radius, rho0, c0 must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
        bc_left, bc_right = (boundary, boundary) if isinstance(boundary, str) else boundary
        if bc_left not in ("closed", "open") or bc_right not in ("closed", "open"):
            raise ValueError(
                f"each boundary end must be 'closed' or 'open', got {boundary!r}."
            )

        self.L = float(L)
        self.fs = float(fs)
        self.N = int(N)
        self.radius = float(radius)
        self.boundary: BoundarySpec = boundary
        self._bc_left: End = bc_left
        self._bc_right: End = bc_right
        self.sigma = float(sigma)
        self.rho0 = float(rho0)
        self.c0 = float(c0)

        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.lam = self.c0 * self.k / self.h
        if self.lam > 1.0 + _LAMBDA_TOL:
            raise ValueError(
                f"CFL violated: lambda = c0*k/h = {self.lam:.6f} > 1. "
                "Reduce fs, refine the grid (increase N), or shorten the tube."
            )

        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)  # pressure nodes
        self.x_u: NDArray[np.float64] = 0.5 * (self.x[1:] + self.x[:-1])    # velocity half-nodes

        # Cross-section (Webster). Constant cylinder now; the arrays are the seam for S(x) later.
        area = np.pi * self.radius * self.radius
        self.S_node: NDArray[np.float64] = np.full(self.N + 1, area)  # S at pressure nodes
        self.S_seg: NDArray[np.float64] = np.full(self.N, area)       # S at velocity half-nodes

        # Trapezoidal node weight: h interior, h/2 at each end (the half-cell that closes a rigid
        # wall without a hand-coded ghost stencil — the free-beam end-mass lesson).
        self._w: NDArray[np.float64] = np.full(self.N + 1, self.h)
        self._w[0] = self._w[-1] = 0.5 * self.h

        # Diagonal capacitance C_l (pressure) and inductance M_j (velocity); update prefactors.
        self._C: NDArray[np.float64] = self._w * self.S_node / (self.rho0 * self.c0**2)
        self._M: NDArray[np.float64] = self.h * self.rho0 / self.S_seg
        self._p_pref: NDArray[np.float64] = self.k / self._C   # p += -p_pref * div(U)
        self._u_pref: NDArray[np.float64] = self.k / self._M   # U += -u_pref * grad(p)

        # Open ends are Dirichlet p = 0; mask them out of the pressure DOFs.
        self._open_left = self._bc_left == "open"
        self._open_right = self._bc_right == "open"

        self.p: NDArray[np.float64] = np.zeros(self.N + 1)   # p^n
        self.U: NDArray[np.float64] = np.zeros(self.N)       # U^{n+1/2}
        self.U_prev: NDArray[np.float64] = np.zeros(self.N)  # U^{n-1/2}
        self.n: int = 0

        # Pressure-evolution operator L = G^T M^{-1} G and mass C, for the modal oracle. The scheme
        # eliminates U to C d_tt p = -k^2 L p; the generalized eigenvalues of (L, C) restricted to
        # the free (non-open) nodes are omega^2 (see analysis/modal + tests/helpers).
        self.Lop, self.Cmat, self.dof = self._build_pressure_operator()

    # -- initial conditions -------------------------------------------------------------

    def set_state(
        self,
        p0: NDArray[np.float64],
        u0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial pressure field ``p^0`` (and optional half-node volume velocity).

        Starts from rest by default (``U^{-1/2} = 0``): the first half-node velocity ``U^{1/2}`` is
        taken as one consistent momentum half-step from ``p^0``, so a single-mode pressure IC
        oscillates cleanly and the initial energy is exactly the acoustic potential
        ``1/2 sum C_l (p^0_l)^2``. Pass ``u0`` (length ``N``) to seed a nonzero ``U^{-1/2}``.
        """
        p0 = np.asarray(p0, dtype=float).copy()
        if p0.shape != (self.N + 1,):
            raise ValueError(f"p0 must have shape {(self.N + 1,)}, got {p0.shape}.")
        self._apply_open_ends(p0)
        u_prev = np.broadcast_to(np.asarray(u0, dtype=float), (self.N,)).astype(float).copy()

        self.p = p0
        self.U_prev = u_prev
        self.U = self._momentum(p0, u_prev)  # U^{1/2} from U^{-1/2} and p^0
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def _momentum(
        self, p: NDArray[np.float64], u_prev: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """One momentum half-step ``U^{+1/2} = [(1-sk) U^{-1/2} - u_pref * (grad p)] / (1+sk)``."""
        sk = self.sigma * self.k
        grad_p = p[1:] - p[:-1]  # (Gp)_j = p_{l+1} - p_l, length N
        return ((1.0 - sk) * u_prev - self._u_pref * grad_p) / (1.0 + sk)

    def _divergence(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Discrete divergence ``(G^T U)`` per node: ``U_{l+1/2} - U_{l-1/2}``, zero wall ghosts.

        Node 0 sees only the segment to its right (``U_0``), node N only the one to its left
        (``-U_{N-1}``) — the rigid-wall closure, no ghost velocity needed.
        """
        div = np.zeros(self.N + 1)
        div[:-1] += u   # +U_{l+1/2} at node l (l = 0 .. N-1)
        div[1:] -= u    # -U_{l-1/2} at node l (l = 1 .. N)
        return div

    def step(self) -> None:
        """Advance one timestep: pressure from the current velocity, then velocity from it."""
        # p^{n+1} = p^n - p_pref * div(U^{n+1/2})
        p_next = self.p - self._p_pref * self._divergence(self.U)
        self._apply_open_ends(p_next)
        # U^{n+3/2} from p^{n+1} and U^{n+1/2}
        u_next = self._momentum(p_next, self.U)

        self.U_prev = self.U
        self.U = u_next
        self.p = p_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current pressure field ``p^n`` (a copy, safe to store for plotting)."""
        return self.p.copy()

    def energy(self) -> float:
        """Discrete acoustic energy ``E^n`` (Joules): compliance ``p^2`` plus the **cross-time**
        inductive ``U^{n+1/2} U^{n-1/2}`` term. Conserved to machine precision when ``sigma = 0``;
        monotonically decreasing when ``sigma > 0`` (passivity)."""
        compliance = 0.5 * float(np.dot(self._C, self.p * self.p))
        inductive = 0.5 * float(np.dot(self._M, self.U * self.U_prev))
        return compliance + inductive

    def displacement_at(self, index: int) -> float:
        """Pressure at node ``index`` — a microphone pickup for spectral analysis."""
        return float(self.p[index])

    def pressure_at(self, index: int) -> float:
        """Alias for :meth:`displacement_at` reading in the natural acoustic quantity (pressure)."""
        return float(self.p[index])

    # -- internals ----------------------------------------------------------------------

    def _apply_open_ends(self, p: NDArray[np.float64]) -> None:
        if self._open_left:
            p[0] = 0.0
        if self._open_right:
            p[-1] = 0.0

    def _build_pressure_operator(
        self,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int64]]:
        """Assemble ``L = G^T M^{-1} G`` (pressure stiffness) and the mass ``C`` as sparse matrices,
        plus the indices of the **free** (non-open) pressure nodes. The generalized eigenproblem
        ``L phi = Lambda C phi`` on the free nodes yields ``Lambda = omega^2`` (continuum), which
        the modal oracle maps through the leapfrog dispersion to the discrete frequency."""
        n_seg = self.N
        # G: (N) x (N+1), row j has -1 at col j and +1 at col j+1.
        rows = np.repeat(np.arange(n_seg), 2)
        cols = np.empty(2 * n_seg, dtype=np.int64)
        cols[0::2] = np.arange(n_seg)
        cols[1::2] = np.arange(n_seg) + 1
        data = np.tile(np.array([-1.0, 1.0]), n_seg)
        G = sparse.coo_matrix((data, (rows, cols)), shape=(n_seg, self.N + 1)).tocsr()
        Minv = sparse.diags(1.0 / self._M)
        Lop = (G.T @ Minv @ G).tocsr()
        Cmat = sparse.diags(self._C).tocsr()
        free = np.ones(self.N + 1, dtype=bool)
        if self._open_left:
            free[0] = False
        if self._open_right:
            free[-1] = False
        dof = np.nonzero(free)[0].astype(np.int64)
        return Lop, Cmat, dof
