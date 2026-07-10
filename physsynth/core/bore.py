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
frequency-dependent tube losses come later.

**The radiating bell (batch 2) — how sound leaves.** An ideal open end reflects perfectly and
radiates nothing. A real bell/open end presents a **radiation impedance**: the resistive part ``R``
(acoustic, Pa·s/m³) relates the end pressure to the volume velocity leaving the tube,
``p_end = R U_out``. That single resistor is exactly the
:class:`~physsynth.core.radiation.RadiatedBody` pattern — a rank-1, unconditionally-passive
dashpot — moved onto a boundary node. The terminating
node stays a live **half-cell** DOF (like a rigid wall), drained by ``U_out`` taken at the
**centered** end pressure ``p_bar = (p^{n+1} + p^n)/2`` so ``U_out = p_bar / R`` (implicit, the VK/
bow/radiation-load lesson). Because the coupling is one scalar the implicit solve collapses to a
**1×1** equation for the end node,

    p_end^{n+1} = (a p_rigid - b p_end^n) / (a + b),   a = C_end / k,  b = 1 / (2 R),

where ``p_rigid`` is the force-free rigid-wall step. The discrete energy then telescopes to
``E^{n+1} - E^n = -k p_bar U_out = -k R U_out^2 <= 0``, and that shed power is booked into
:attr:`radiated_energy` so ``E_bore + radiated_energy`` is conserved to machine precision (the
passivity money test — the acoustic dual of the ``sigma`` channel and of ``RadiatedBody``). Since
``a + b > 0`` for any ``R > 0`` the solve is never singular: the load is **unconditionally
passive**, no stability guard beyond the interior CFL. ``R`` interpolates the two ideal ends —
``R -> infinity`` recovers the rigid **closed** wall (no radiation), ``R -> 0`` the pressure-release
**open** end — with a real, physical, energy-shedding bell in between. The independent oracle (the
batch-2 analogue of the odd-harmonic signature) is the pressure **reflection coefficient**
``r = (R - Z0) / (R + Z0)`` (``Z0 = rho0 c0 / S`` the tube's characteristic acoustic impedance): a
pulse hitting the end sheds the fraction ``1 - r^2 = 4 R Z0 / (R + Z0)^2`` of its energy to the far
field, and a **matched** load ``R = Z0`` (``r = 0``) is anechoic — it absorbs everything. ``R`` is
constant across frequency here (evaluate a representative value from
:func:`~physsynth.core.radiation.monopole_radiation_resistance` /
:func:`~physsynth.core.radiation.piston_radiation_resistance` at the fundamental); the frequency-
dependent radiation *reactance* (the end correction, ``propto sqrt(omega)``) is a later batch, the
same placeholder philosophy as ``sigma``.

**Overdamped end node (a cosmetic Nyquist artifact, by design).** At a realistic bell ``R << Z0``
the drain ``b = 1/(2R)`` dwarfs the node compliance ``a = C_end/k``, so the update is
``p_end^{n+1} ~ -p_end^n`` — the *raw* end-node pressure carries a marginal Nyquist ripple. It
**cancels exactly** in ``U_out propto (p_end^{n+1} + p_end^n)``, so every validated quantity — the
radiated energy, the far-field read-out ``pressure() = dU_out/dt``, and any **interior** pickup — is
clean. Read the far field from ``U_out``/:meth:`pressure`, never the raw terminating node, and place
spectral pickups in the interior (node 1), as the batch-1 tests do.

**Webster from day one.** ``S(x)`` is carried as node/segment arrays throughout, so a cylinder
(constant ``S``) is just the constant case; a cone or flare (a bell, a saxophone) is a *different
area profile*, not a different solver.

Headless: NumPy only (SciPy sparse for the modal-oracle operator).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

__all__ = ["Bore", "RHO0_AIR", "C0_AIR"]

# An in-place pressure-field corrector applied inside step() between the pressure and momentum
# sub-steps: the seam through which an implicit boundary exciter (the reed) injects into a half-cell
# node. It receives the freshly-updated p^{n+1} and mutates it. See Bore.step / ReedBore.
SourceHook = Callable[[NDArray[np.float64]], None]

# Ambient air (matches physsynth.core.radiation so the bore and its radiation load agree).
RHO0_AIR = 1.2041  # kg/m^3
C0_AIR = 343.0     # m/s

End = Literal["closed", "open", "radiating"]
# Per-end boundary: one value applies to both ends; a (left, right) tuple sets them independently.
# A clarinet is ("closed", "open") — rigid mouthpiece wall, open bell; ("closed", "radiating") gives
# it a passively-lossy, sound-shedding bell (batch 2).
BoundarySpec = End | tuple[End, End]

_ENDS = ("closed", "open", "radiating")

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
    boundary : {"closed", "open", "radiating"} or (left, right) tuple
        Per-end termination. ``"closed"`` is a rigid wall (``U = 0``, a pressure antinode);
        ``"open"`` is pressure-release (``p = 0``, a pressure node); ``"radiating"`` is a
        passively-lossy bell (resistance ``R_bell``, sheds sound to the far field — batch 2), which
        needs ``R_bell > 0``. Default ``("closed", "open")`` — the ideal (lossless) clarinet.
    sigma : float
        Viscous loss coefficient (>= 0) for the ``-2 sigma U`` drag on the air column. ``0`` ->
        lossless (energy conserved); ``> 0`` -> energy decreases monotonically (passivity).
    R_bell : float
        Radiation resistance ``R`` (acoustic, Pa·s/m³) of any ``"radiating"`` end — the rank-1
        passive dashpot that lets sound leave. ``R -> infinity`` recovers a rigid wall, ``R -> 0`` a
        pressure-release open end; a matched ``R = Z0 = rho0 c0 / S`` is anechoic. Must be ``> 0``
        when an end radiates (``0`` when no end does). A representative value comes from
        :func:`~physsynth.core.radiation.piston_radiation_resistance` at the fundamental.
    rho0, c0 : float
        Air density (kg/m^3) and sound speed (m/s). Default ambient air.

    Raises
    ------
    ValueError
        Non-physical parameters, an unknown boundary token, a ``"radiating"`` end without
        ``R_bell > 0``, or CFL ``lambda = c0 k / h > 1``.
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
        R_bell: float = 0.0,
        rho0: float = RHO0_AIR,
        c0: float = C0_AIR,
    ) -> None:
        if min(L, fs, radius, rho0, c0) <= 0:
            raise ValueError("L, fs, radius, rho0, c0 must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
        if R_bell < 0:
            raise ValueError("R_bell (radiation resistance) must be >= 0.")
        bc_left, bc_right = (boundary, boundary) if isinstance(boundary, str) else boundary
        if bc_left not in _ENDS or bc_right not in _ENDS:
            raise ValueError(
                f"each boundary end must be one of {_ENDS}, got {boundary!r}."
            )
        radiating = bc_left == "radiating" or bc_right == "radiating"
        if radiating and R_bell <= 0.0:
            raise ValueError(
                "a 'radiating' end needs R_bell > 0 (the bell's radiation resistance). "
                "Use 'open' for the ideal lossless pressure-release end (R -> 0)."
            )

        self.L = float(L)
        self.fs = float(fs)
        self.N = int(N)
        self.radius = float(radius)
        self.boundary: BoundarySpec = boundary
        self._bc_left: End = bc_left
        self._bc_right: End = bc_right
        self.sigma = float(sigma)
        self.R_bell = float(R_bell)
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

        # Radiating ends: a live half-cell node drained by the resistor R_bell (batch 2). The node
        # stays a free DOF (like a closed wall); the rank-1 solve uses a = C_end/k and b = 1/(2R).
        self._rad_left = self._bc_left == "radiating"
        self._rad_right = self._bc_right == "radiating"
        self._a_left = self._C[0] / self.k    # node compliance rate C_end/k (= 1/p_pref), each end
        self._a_right = self._C[-1] / self.k
        # Characteristic acoustic impedance Z0 = rho0 c0 / S (constant cylinder); sets the
        # reflection coefficient r = (R - Z0)/(R + Z0) and the matched (anechoic) load R = Z0.
        self.Z0 = self.rho0 * self.c0 / area

        # Energy shed to the far field through radiating ends: integral of P_rad = R U_out^2 dt.
        # Makes E_bore + radiated_energy conserved (the passivity identity), mirroring RadiatedBody.
        self.radiated_energy = 0.0
        self._U_out = 0.0       # total outgoing volume velocity U_out^{n+1/2} (far-field readout)
        self._U_out_prev = 0.0  # U_out^{n-1/2}, for the volume-acceleration read-out pressure()

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
        self.radiated_energy = 0.0           # reset the far-field energy channel for a fresh run
        self._U_out = 0.0
        self._U_out_prev = 0.0
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

    def step(self, source: SourceHook | None = None) -> None:
        """Advance one timestep: pressure from the current velocity, then velocity from it.

        ``source`` is an optional callback ``source(p_next)`` invoked on the freshly-updated
        pressure field **after** the open-end pin and **before** the momentum sub-step — the one
        place a boundary-node injection (an implicit exciter such as the reed) can correct
        ``p_next`` in time for the velocity ``U^{n+3/2}`` to see it (the same ordering the radiating
        bell obeys). It modifies ``p_next`` in place; the bore stays agnostic to what drives it.
        ``None``
        (the default) is bit-for-bit the un-driven bore — how :func:`simulate` and every batch-1/2
        test call it. A reed feeds volume flow into node 0's half-cell here (see
        :class:`~physsynth.core.reed.ReedBore`)."""
        # p^{n+1} = p^n - p_pref * div(U^{n+1/2})
        p_next = self.p - self._p_pref * self._divergence(self.U)
        self._apply_open_ends(p_next)
        # An external exciter (the reed) injects into a boundary half-cell here — before the
        # momentum step, so U^{n+3/2} sees the corrected node pressure (same as the radiating bell).
        if source is not None:
            source(p_next)
        # Drain any radiating end (rank-1 scalar dashpot) *before* the momentum step, so that the
        # velocity U^{n+3/2} sees the corrected end pressure — why this must live inside step().
        self._apply_radiating_ends(p_next)
        # U^{n+3/2} from p^{n+1} and U^{n+1/2}
        u_next = self._momentum(p_next, self.U)

        self.U_prev = self.U
        self.U = u_next
        self.p = p_next
        self.n += 1

    def _apply_radiating_ends(self, p_next: NDArray[np.float64]) -> None:
        """Correct the end node(s) for the passive radiation load and book the shed energy.

        Each radiating node's un-pinned value in ``p_next`` is already the **rigid** (force-free)
        half-cell step ``p_rigid`` (the ``_divergence`` closure). The centered resistor
        ``U_out = (p^{n+1} + p^n) / (2R)`` turns that into the 1×1 solve
        ``p^{n+1} = (a p_rigid - b p^n) / (a + b)`` (``a = C_end/k``, ``b = 1/2R``). One formula
        serves either end — the outgoing direction differs but the dissipated ``k R U_out^2`` does
        not — so a left-end sign error can only show up as energy drift (the both-ends test).
        """
        if self.R_bell <= 0.0:
            return
        b = 0.5 / self.R_bell
        u_out_total = 0.0
        if self._rad_left:
            u_out_total += self._radiate_node(p_next, 0, self._a_left, b)
        if self._rad_right:
            u_out_total += self._radiate_node(p_next, -1, self._a_right, b)
        self._U_out_prev = self._U_out
        self._U_out = u_out_total

    def _radiate_node(self, p_next: NDArray[np.float64], idx: int, a: float, b: float) -> float:
        """Rank-1 dashpot at end node ``idx``: correct ``p_next[idx]`` in place, return ``U_out``.

        See :meth:`_apply_radiating_ends` for the derivation of the 1×1 solve."""
        p_old = float(self.p[idx])
        p_rigid = float(p_next[idx])                 # force-free rigid half-cell step
        p_new = (a * p_rigid - b * p_old) / (a + b)  # 1×1 implicit solve (a + b > 0 always)
        p_next[idx] = p_new
        u_out = b * (p_new + p_old)                  # = p_bar / R (Nyquist part cancels in the sum)
        self.radiated_energy += self.k * self.R_bell * u_out * u_out  # P_rad dt = k R U_out^2 >= 0
        return u_out

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current pressure field ``p^n`` (a copy, safe to store for plotting)."""
        return self.p.copy()

    def acoustic_energy(self) -> float:
        """Energy **stored in the air column** (Joules): compliance ``p^2`` plus the **cross-time**
        inductive ``U^{n+1/2} U^{n-1/2}`` term — the field energy alone, excluding what has already
        radiated away. This is the quantity that *decreases* as a radiating bell sheds sound."""
        compliance = 0.5 * float(np.dot(self._C, self.p * self.p))
        inductive = 0.5 * float(np.dot(self._M, self.U * self.U_prev))
        return compliance + inductive

    def energy(self) -> float:
        """Total conserved energy ``E_bore + radiated_energy`` (Joules).

        The air-column field energy (:meth:`acoustic_energy`) **plus** everything shed to the far
        field through radiating ends (:attr:`radiated_energy`). Conserved to machine precision when
        ``sigma = 0`` (any ``R_bell``) — the radiation channel exactly captures what the bell sheds,
        just like :class:`~physsynth.core.radiation.RadiatedBody`; monotonically decreasing if the
        air column is itself viscous (``sigma > 0``). Assert conservation on this total, not
        :meth:`acoustic_energy` alone (which falls as the bell radiates). With no radiating end
        ``radiated_energy`` stays ``0`` and this is bit-for-bit the field energy."""
        return self.acoustic_energy() + self.radiated_energy

    def displacement_at(self, index: int) -> float:
        """Pressure at node ``index`` — a microphone pickup for spectral analysis."""
        return float(self.p[index])

    def pressure_at(self, index: int) -> float:
        """Alias for :meth:`displacement_at` reading in the natural acoustic quantity (pressure)."""
        return float(self.p[index])

    def pressure(self) -> float:
        """Far-field monopole read-out: the bell's net volume **acceleration** ``dU_out/dt``
        (m³/s²).

        The batch-1 :class:`~physsynth.core.radiation.AirRadiation` reads exactly this — hand it the
        bore (``AirRadiation.radiate(bore)``) to turn the radiating end into a listener's pressure.
        Computed from the *outgoing* volume velocity ``U_out`` (whose Nyquist part has cancelled),
        so it is clean even though the raw terminating node carries the cosmetic ripple. ``0`` when
        no end radiates."""
        return (self._U_out - self._U_out_prev) / self.k

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
