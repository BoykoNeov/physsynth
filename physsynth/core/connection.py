"""Energy-conserving bridge connection: string terminus <-> modal body (the coupling node of the
``exciter -> resonator -> body/radiation`` abstraction, HANDOFF §3.2).

A string clamped at the nut (left, fixed) and **free at the right end** is loaded there by a linear
spring of stiffness ``K`` connecting the string-end displacement ``u_b = u[N]`` to the body's
driving-point displacement ``w_b = sum_i phi_i q_i``. The spring stores ``E_conn = 1/2 K eta^2``
with stretch ``eta = u_b - w_b``; it pulls the string end with ``-F`` and the body with ``+F``,
where ``F = K eta`` (Newton's third law).

**Why explicit is exact here (not a compromise).** The string, the body, and this spring together
form *one linear leapfrog* ``x^{n+1} = 2x^n - x^{n-1} - k^2 M^{-1} S x^n`` (``S`` symmetric, the
spring contributing the rank-1 block ``K a a^T`` with ``a = [.. +1 (end) .. -phi_i (modes) ..]``).
A linear leapfrog conserves the cross-time energy

    H^n = 1/2 (delta_t- x)^T M (delta_t- x)  +  1/2 (x^n)^T S x^{n-1}

*exactly* for any stable ``k``. That ``H`` splits cleanly into ``E_string + E_body + E_conn`` with
``E_conn = 1/2 K eta^n eta^{n-1}`` (the spring block of ``1/2 (x^n)^T S x^{n-1}``) — so the
three-way energy decomposition is a rigorous identity, not a seam. No implicit solve is needed (that
was for the *nonlinear* von Kármán coupling); a linear spring is conserved explicitly.

**Stability (exact, not a heuristic).** Being explicit, the connection has a CFL. The isolated
end-mass + body 2-DOF estimate ``K (beta_s + beta_b) < 4`` (``beta_s = 2 k^2 / (rho h)`` the string
end-node inverse mass from its ``h/2`` half-cell weight, ``beta_b = k^2 sum_i phi_i^2 / m_i``) is
kept only as a diagnostic — it is *necessary but not sufficient*, because the spring is a rank-1
perturbation that lifts the top coupled eigenvalue above both parts' isolated maxima (empirically
the system can blow up at half that bound). The **actual** guard assembles the coupled leapfrog
operator ``A`` (``x^{n+1} = 2x^n - x^{n-1} - k^2 A x^n``) — matrix-free, applying the *same* step
operators so it cannot disagree with the scheme — and requires ``k^2 lambda_max(A) < 4``, the exact
stability bound. The string's own Nyquist mode already sits near this ceiling at ``lambda = 1``, so
coupling has the most headroom below it; run the string at ``lambda < 1``.

Headless: NumPy only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from .body import ModalBody
from .plate import Plate
from .string_ideal import IdealString

_CFL_TOL = 1e-12


class StringBodyBridge:
    """A string terminated on a :class:`ModalBody` through a linear bridge spring.

    Implements the engine's ``Resonator`` protocol (``step``/``energy``/``state``/``k``), with
    :meth:`energy` returning the **total** ``E_string + E_body + E_conn`` — the quantity a lossless
    run must conserve. ``E_string`` alone is *not* conserved once coupled (it exchanges power with
    the body); assert on the total.

    Parameters
    ----------
    string : IdealString
        The string. Its right end must be ``"free"`` (the bridge terminus); the left is typically
        ``"fixed"`` (the nut). Build it with ``boundary=("fixed", "free")``.
    body : ModalBody
        The soundboard/body. Its driving point is where the string attaches.
    K : float
        Bridge spring stiffness (N/m). ``K = 0`` decouples the two (they evolve independently —
        used for the bit-identity check). Larger ``K`` -> stiffer bridge (``K -> inf`` approaches a
        rigid displacement-continuity terminus).

    Raises
    ------
    ValueError
        If the two timesteps differ, the string's right end is not free, ``K < 0``, or the
        connection CFL ``K (beta_s + beta_b) < 4`` is violated.
    """

    def __init__(self, *, string: IdealString, body: ModalBody, K: float) -> None:
        if not np.isclose(string.k, body.k, rtol=0, atol=1e-15):
            raise ValueError(
                f"string and body must share a timestep (got k={string.k:.3e} vs {body.k:.3e}); "
                "build them at the same fs."
            )
        if string._bc_right != "free":
            raise ValueError(
                "the string's right end must be 'free' to attach a body bridge "
                "(build it with boundary=('fixed', 'free'))."
            )
        if K < 0:
            raise ValueError("bridge stiffness K must be >= 0.")

        self.string = string
        self.body = body
        self.K = float(K)
        self.k = string.k

        # beta_s: inverse of the string end-node mass rho*(h/2). beta_b: body driving-point inverse
        # modal mass sum. Their sum sets the connection's effective (squared) frequency. Kept as a
        # cheap diagnostic (the isolated 2-DOF bound K*(beta_s+beta_b) < 4) -- NOT the real guard.
        self.beta_s = 2.0 * self.k * self.k / (string.rho * string.h)
        self.beta_b = self.k * self.k * float(np.sum(body.phi * body.phi / body.m))
        self.cfl_2dof = self.K * (self.beta_s + self.beta_b)

        # Exact guard: k^2 * lambda_max of the coupled leapfrog operator A must be < 4.
        self.spectral_radius = self._max_leapfrog_eigenvalue()
        if self.k * self.k * self.spectral_radius >= 4.0 - _CFL_TOL:
            raise ValueError(
                f"connection unstable: k^2 * lambda_max(A) = "
                f"{self.k * self.k * self.spectral_radius:.6f} >= 4. "
                "Reduce K, raise fs, or increase the body/string end mass."
            )

        self.n = 0

    # -- stability (assembled once at construction, off the hot loop) --------------------

    def _apply_A(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the coupled leapfrog operator ``A`` (``x^{n+1} = 2x^n - x^{n-1} - k^2 A x^n``).

        ``x`` stacks the string DOFs (nodes ``1..N`` — the fixed left node 0 is excluded) then the
        ``M`` modal coordinates. Built entirely from the same stencils :meth:`step` uses, so the
        stability spectrum is guaranteed consistent with the actual integrator.
        """
        s, b = self.string, self.body
        N = s.N
        u = np.zeros(N + 1)
        u[1:] = x[:N]  # node 0 stays clamped
        q = x[N:]

        # String internal: A_str u = -(c^2/h^2) * second_difference (fixed-left/free-right stencil).
        out_u = -(s.c * s.c / (s.h * s.h)) * s._second_diff(u)[1:]
        # Body internal: A_body q = omega^2 q.
        out_q = b.omega * b.omega * q

        # Spring (rank-1) at the terminus: eta = u_N - sum phi_i q_i.
        eta = u[-1] - float(np.dot(b.phi, q))
        out_u[-1] += 2.0 * self.K * eta / (s.rho * s.h)   # (A x)_N += 2 K eta / (rho h)
        out_q += -self.K * b.phi * eta / b.m              # (A x)_i += -K phi_i eta / m_i
        return np.concatenate([out_u, out_q])

    def _max_leapfrog_eigenvalue(self) -> float:
        """Largest eigenvalue of ``A`` (real, >= 0). ``A`` is small (``N + M``); assemble it densely
        by applying :meth:`_apply_A` to each basis vector and take ``max Re(eig)``."""
        n = self.string.N + self.body.M
        A = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            A[:, j] = self._apply_A(e)
            e[j] = 0.0
        return float(np.max(np.linalg.eigvals(A).real))

    # -- helpers ------------------------------------------------------------------------

    def _stretch(self, *, prev: bool = False) -> float:
        """Spring stretch ``eta = u_b - w_b`` now (``prev=True`` -> the previous step)."""
        if prev:
            u_b = self.string.u_prev[-1]
            w_b = float(np.dot(self.body.phi, self.body.q_prev))
        else:
            u_b = self.string.u[-1]
            w_b = self.body.bridge_displacement()
        return float(u_b - w_b)

    def connection_force(self) -> float:
        """Current bridge force ``F = K eta^n`` (explicit; N)."""
        return self.K * self._stretch()

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep. Explicit spring ``F = K eta^n`` drives both parts at time ``n``."""
        F = self.connection_force()
        # String: free-end leapfrog, then subtract the reaction impulse at the terminus node.
        self.string.step()
        self.string.u[-1] -= self.beta_s * F
        # Body: the same F enters each mode as phi_i F (ModalBody.step does the phi_i/m_i scaling).
        self.body.step(force=F)
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E_string + E_body + E_conn`` (Joules).

        ``E_conn = 1/2 K eta^n eta^{n-1}`` (cross-time, matching every other potential in the
        project). Conserved to machine precision for a lossless run; monotonically decreasing when
        the string or any body mode is lossy.
        """
        e_conn = 0.5 * self.K * self._stretch() * self._stretch(prev=True)
        return self.string.energy() + self.body.energy() + e_conn

    @property
    def state(self) -> NDArray[np.float64]:
        """The string displacement field (the visible resonator, for animation snapshots)."""
        return self.string.state

    def displacement_at(self, index: int) -> float:
        """String pickup at node ``index`` (for spectral analysis of the driven string)."""
        return self.string.displacement_at(index)

    def pressure(self) -> float:
        """Radiated pressure from the body, ``sum_i a_i q_i''`` (monopole ∝ volume acceleration)."""
        return self.body.pressure()


class StringPlateBridge:
    """A string terminated on a **grid** :class:`Plate` (the body) through a linear bridge spring.

    Steps 4-5 of the body/radiation node: instead of the lumped :class:`ModalBody`, the radiating
    body is a distributed Kirchhoff plate — the **simply-supported** rectangle (model #5, Step 4) or
    the **free-edge** FFFF plate (model #5b, the suspended cymbal/gong, Step 5). One class covers
    both: the plate has no modal coordinates, so the string couples to it at a single
    **driving-point** node ``dp`` via a point force — the *driving-point adapter* the modal bridge
    did not need — and :class:`Plate` internally branches its ``step``/``energy``/``pressure`` on
    ``boundary``, so only the stability guard below is boundary-specific. Everything else mirrors
    :class:`StringBodyBridge`: a fixed-left / free-right string, free end loaded by a linear spring
    ``F = K eta^n`` with stretch ``eta = u_end - w_dp`` (``w_dp`` the plate displacement at ``dp``),
    Newton's third law splitting ``-F`` to the string end and ``+F`` to the plate.

    **Why the coupling is still exactly energy-conserving.** The spring is explicit
    (``F = K eta^n``, at time ``n``). The string is an explicit leapfrog, so its reaction is applied
    post-step as the impulse ``u_end -= beta_s F`` (exact for a linear scheme). The plate is the
    *implicit* θ-scheme, so its ``+F`` source must enter the RHS **before** the solve as
    ``k² F / (rho_s h²)`` at ``dp`` (:meth:`Plate.step` ``f_ext``); the θ-averaging touches only the
    elastic term, so a time-``n`` source contributes ``k F delta_{t.} w_dp`` to the plate energy
    regardless of θ. The three per-step energy increments
    ``E_str += -k F delta_{t.} u_end``, ``E_plate += +k F delta_{t.} w_dp`` and
    ``E_conn += +k F delta_{t.} eta`` sum to zero because ``eta = u_end - w_dp`` — the same
    telescoping the modal bridge proves, one distributed body up.
    ``E_conn = 1/2 K eta^n eta^{n-1}``.

    **Stability (exact, Sherman–Morrison — cleaner than the modal case).** The coupled leapfrog is
    energy-stable iff the conserved quadratic form ``G0 - (k²/4) K a a^T`` (``a = e_end - e_dp``,
    the spring's rank-1 block) is positive-definite. ``G0`` is block-diagonal (string + plate) and
    each block has the *same* general shape ``M + (θ - 1/4) k² S`` (mass ``M``, stiffness ``S``):

    * String block (explicit leapfrog, ``θ = 0``): ``M_str - (k²/4) S_str``, PD for ``lambda < 1``.
    * **Supported** plate block: ``rho_s h² [I + (θ - 1/4) k² kappa² B]``.
    * **Free** plate block: ``rho_s [W + (θ - 1/4) k² kappa² K]`` — ``W`` (the lumped-area diagonal
      mass) already carries the ``h²`` weight, so there is *no extra* ``h²`` prefactor.

    Every plate block is PD for ``θ >= 1/4`` — that is *why* the implicit plate is unconditionally
    stable, and for the free edge the diagonal ``W`` is exactly what pins the ``{1, x, y}``
    rigid-body nullspace of ``K``, so the plate contributes only headroom. So the *only* thing that
    can destabilise ``G0`` is the rank-1 spring, and a rank-1 negative perturbation stays PD iff
    (Sherman–Morrison)

        (k²/4) K [ (G0_str^{-1})_{end,end} + (G0_plate^{-1})_{dp,dp} ]  <  1 .

    That is the *exact* stability bound (not the 2-DOF estimate that is a footgun for the modal
    bridge), obtained from two small linear solves at construction. The margin is stored on
    :attr:`stability_margin`; the string is run at ``lambda < 1`` (its Nyquist mode sits marginally
    at ``lambda = 1``, and the spring pushes it unstable).

    **No rigid drift on the free plate.** The single-point spring only couples to (and lifts) plate
    modes with ``w_dp != 0``. The rigid-body modes that survive lie in ``span{1, x, y}`` with
    ``v(dp) = 0`` (the one constraint leaves 2 of the 3), so the driving-point force is *orthogonal*
    to them: starting from rest they are never excited — no piston/tilt drift, and the total energy
    drifts to the same ``~1e-13`` as the supported bridge. The piston/tilt combinations with
    ``w_dp != 0`` become genuine bounded oscillations on the spring (the plate bouncing on the
    bridge).

    Parameters
    ----------
    string : IdealString
        The string; its right end must be ``"free"`` (build with ``boundary=("fixed", "free")``).
    plate : Plate
        The body — either ``boundary="supported"`` (model #5, a rectangular soundboard) or
        ``boundary="free"`` (model #5b, a suspended cymbal/gong). The guard and the force injection
        pick the matching mass/stiffness forms automatically.
    K : float
        Bridge spring stiffness (N/m). ``K = 0`` decouples the two (the bit-identity check).
    drive_index : int, optional
        Live-node index on the plate where the string attaches. Defaults to the node nearest the
        plate centroid offset toward a corner (an off-symmetry point, so it is not a node of the
        low modes). Use :meth:`Plate.pickup_index_at` to choose one explicitly.

    Raises
    ------
    ValueError
        If the two timesteps differ, the string's right end is not free, the plate boundary is not
        ``"supported"``/``"free"``, ``K < 0``, ``drive_index`` is out of range, or the exact
        stability margin is ``>= 1``.
    """

    def __init__(
        self,
        *,
        string: IdealString,
        plate: Plate,
        K: float,
        drive_index: int | None = None,
    ) -> None:
        if not np.isclose(string.k, plate.k, rtol=0, atol=1e-15):
            raise ValueError(
                f"string and plate must share a timestep (got k={string.k:.3e} vs {plate.k:.3e}); "
                "build them at the same fs."
            )
        if string._bc_right != "free":
            raise ValueError(
                "the string's right end must be 'free' to attach a plate bridge "
                "(build it with boundary=('fixed', 'free'))."
            )
        if plate.boundary not in ("supported", "free"):  # Plate validates this, but be explicit
            raise ValueError(
                f"the plate body must be 'supported' or 'free', got {plate.boundary!r}."
            )
        if string.lam >= 1.0 - _CFL_TOL:
            raise ValueError(
                "the string must run at lambda < 1: its Nyquist mode is marginal at lambda = 1 and "
                "the bridge spring pushes it unstable (the guard's string block G0_str is singular "
                "there). Rebuild the string at lambda < 1 (0.9 is a good default)."
            )
        if K < 0:
            raise ValueError("bridge stiffness K must be >= 0.")

        self.string = string
        self.plate = plate
        self.K = float(K)
        self.k = string.k

        if drive_index is None:
            # A corner-offset point: off every low mode's symmetry axis, so the string can excite
            # the whole low spectrum (the plate centre is a node of the antisymmetric modes).
            drive_index = plate.pickup_index_at(0.3 * plate.Lx, 0.4 * plate.Ly)
        if not (0 <= int(drive_index) < plate.n_live):
            raise ValueError(f"drive_index {drive_index} out of range [0, {plate.n_live}).")
        self.drive_index = int(drive_index)

        # String end-node inverse mass (its h/2 half-cell): u_end -= beta_s F applies the reaction.
        self.beta_s = 2.0 * self.k * self.k / (string.rho * string.h)
        # Preallocated live-node force vector (only the drive node is ever nonzero).
        self._f_ext = np.zeros(plate.n_live)

        # Exact Sherman–Morrison guard.
        self.stability_margin = self._stability_margin()
        if self.stability_margin >= 1.0 - _CFL_TOL:
            raise ValueError(
                f"connection unstable: stability margin = {self.stability_margin:.6f} >= 1. "
                "Reduce K, raise fs, or increase the string/plate node mass."
            )

        self.n = 0

    # -- stability (assembled once at construction, off the hot loop) --------------------

    def _stability_margin(self) -> float:
        """Exact coupled stability margin ``(k²/4) K [ (G0_str^{-1})_end + (G0_plate^{-1})_dp ]``.

        Stable iff ``< 1`` (the Sherman–Morrison PD condition for ``G0 - (k²/4) K a a^T``). ``G0``
        is block-diagonal (string + plate) and PD by construction, so only the two diagonal entries
        of its inverse at the coupled nodes enter — two small linear solves, no eigenproblem.
        """
        s, p = self.string, self.plate
        quarter_k2 = 0.25 * self.k * self.k

        # String block: G0_str = M_str - (k²/4) S_str, unknowns = nodes 1..N (node 0 clamped).
        # M_str = diag(rho h, ..., rho h, rho h/2) (free end half-cell); S_str = (T/h) D^T D with
        # D^T D = tridiag(-1, [2,...,2,1], -1) (fixed-left / free-right strain energy).
        N = s.N
        m_diag = np.full(N, s.rho * s.h)
        m_diag[-1] = 0.5 * s.rho * s.h
        main = np.full(N, 2.0)
        main[-1] = 1.0
        dtd = sparse.diags([-np.ones(N - 1), main, -np.ones(N - 1)], [-1, 0, 1], format="csc")
        s_str = (s.T / s.h) * dtd
        g_str = sparse.diags(m_diag, format="csc") - quarter_k2 * s_str
        e_end = np.zeros(N)
        e_end[-1] = 1.0
        g_str_inv_end = float(spsolve(g_str, e_end)[-1])

        # Plate block: general form G0_plate = M + (θ - 1/4) k² S (PD for θ >= 1/4).
        #   "supported": M = rho_s h² I, S = rho_s kappa² h² B -> rho_s h² [I + (θ-1/4)k²kappa² B]
        #   "free":      M = rho_s W,     S = rho_s kappa² K     -> rho_s [W + (θ-1/4)k²kappa² K]
        # W already carries the h² area weight, so the free block has NO extra h² prefactor. Both
        # are PD for θ >= 1/4: the supported identity and the free diagonal W each anchor the
        # semidefinite bending operator -- for "free" the diagonal W is exactly what pins the
        # {1, x, y} rigid-body nullspace of K, so the guard stays a clean two-solve computation.
        coeff = (p.theta - 0.25) * self.k * self.k * p.kappa * p.kappa
        if p.boundary == "supported":
            g_plate = (p.rho * p.h * p.h) * (
                sparse.identity(p.n_live, format="csc") + coeff * p.B
            )
        else:  # free: W-weighted mass + PSD stiffness K (h² baked into both W and K)
            g_plate = p.rho * (p.W + coeff * p.K)
        e_dp = np.zeros(p.n_live)
        e_dp[self.drive_index] = 1.0
        g_plate_inv_dp = float(splu(g_plate.tocsc()).solve(e_dp)[self.drive_index])

        return quarter_k2 * self.K * (g_str_inv_end + g_plate_inv_dp)

    # -- helpers ------------------------------------------------------------------------

    def _stretch(self, *, prev: bool = False) -> float:
        """Spring stretch ``eta = u_end - w_dp`` now (``prev=True`` -> the previous step)."""
        if prev:
            return float(self.string.u_prev[-1] - self.plate.u_prev[self.drive_index])
        return float(self.string.u[-1] - self.plate.u[self.drive_index])

    def connection_force(self) -> float:
        """Current bridge force ``F = K eta^n`` (explicit; N)."""
        return self.K * self._stretch()

    def driving_point_displacement(self) -> float:
        """Plate displacement ``w_dp`` at the driving point (the body's motion the string feels)."""
        return float(self.plate.u[self.drive_index])

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep. Explicit spring ``F = K eta^n`` drives both parts at time ``n``."""
        F = self.connection_force()
        # String: free-end leapfrog, then subtract the reaction impulse at the terminus node.
        self.string.step()
        self.string.u[-1] -= self.beta_s * F
        # Plate: inject +F at the driving point (into the implicit RHS, before the solve).
        self._f_ext[self.drive_index] = F
        self.plate.step(f_ext=self._f_ext)
        self._f_ext[self.drive_index] = 0.0
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E_string + E_plate + E_conn`` (Joules).

        ``E_conn = 1/2 K eta^n eta^{n-1}`` (cross-time). Conserved to machine precision for a
        lossless run; monotonically decreasing when the string or plate is lossy. ``E_string`` alone
        is *not* conserved once coupled — assert on the total.
        """
        e_conn = 0.5 * self.K * self._stretch() * self._stretch(prev=True)
        return self.string.energy() + self.plate.energy() + e_conn

    @property
    def state(self) -> NDArray[np.float64]:
        """The string displacement field (the excited resonator, for animation snapshots)."""
        return self.string.state

    def displacement_at(self, index: int) -> float:
        """String pickup at node ``index`` (for spectral analysis of the driven string)."""
        return self.string.displacement_at(index)

    def pressure(self) -> float:
        """Radiated pressure from the plate (monopole ∝ volume acceleration).

        Delegates to :meth:`Plate.pressure`, whose area weight is ``h²`` for a supported plate and
        the lumped-cell ``W_ii`` for a free plate; either way it reads the *actual* acceleration, so
        it carries the driving-point coupling force.
        """
        return self.plate.pressure()


class SympatheticStrings:
    """Several strings sharing **one** bridge point on a common :class:`ModalBody` — the substrate
    for **sympathetic resonance** and **unison coupling** (piano unisons, sitar/tanpura sympathetic
    strings, the coupled-string family of HANDOFF §12.B).

    Each string ``j`` is clamped at the nut and **free at the bridge end**, where a linear spring
    ``K_j`` connects its end displacement ``u_{end,j}`` to the *shared* body driving-point
    displacement ``w_b = sum_i phi_i q_i``. Because every string attaches at the **same** point, the
    body feels the scalar sum of the bridge forces

        F = sum_j F_j ,      F_j = K_j eta_j ,      eta_j = u_{end,j} - w_b ,

    entering each body mode as ``phi_i F`` — so :class:`ModalBody` is used exactly as-is
    (``body.step(force=sum_j F_j)``, the scalar path). The strings never touch directly; they talk
    to each other *only* through the common ``w_b``. That single shared coordinate is enough to
    produce both signatures of the family: **unison beating** between near-identical strings and
    **sympathetic transfer**, where plucking one string rings up another tuned to one of its
    partials. (Distinct per-string bridge points — each string sampling a different ``phi^{(j)}`` —
    are the natural follow-on; on a lumped modal bank they would be invented spatial inputs, so the
    shared point is built first, and it is what makes the antisymmetric energy oracle exact.)

    **Why this is still one energy-conserving linear leapfrog.** Every spring is explicit
    (``F_j = K_j eta_j^n``); the whole system (all strings + body + all springs) is again *one*
    linear leapfrog conserving the cross-time energy ``H = E_body + sum_j E_{string,j} +
    sum_j E_{conn,j}`` with ``E_{conn,j} = 1/2 K_j eta_j^n eta_j^{n-1}``. The per-step increments
    telescope: string ``j`` contributes ``-k F_j delta_{t.} u_{end,j}``, its connection
    ``+k F_j delta_{t.} eta_j`` and the body ``+k (sum_j F_j) delta_{t.} w_b``; summing over ``j``
    and using ``eta_j = u_{end,j} - w_b`` cancels everything (the modal-bridge proof, one string per
    spring). ``E_{string,j}`` alone is *not* conserved once coupled — **assert on the total**.

    **The discriminating oracle (beyond energy).** Energy conservation and passivity follow from
    the linear-leapfrog structure and would pass even with a flipped coupling sign or a mis-summed
    force, so they are necessary but *not* discriminating. The sharp, machine-precision test is the
    **antisymmetric normal mode**: two identical strings with equal springs and the body at rest,
    started ``u_B = -u_A``. By symmetry ``F_A + F_B = -2 K w_b``, so a body that starts at
    ``w_b = 0`` with zero velocity feels zero force and ``w_b ≡ 0`` (and ``E_body ≡ 0``) *forever*,
    while ``u_B ≡ -u_A`` to machine precision. A wrong coupling sign or summation moves the bridge
    and fails this immediately. Its contrast is the **symmetric** start (``u_B = +u_A``): the bridge
    swings and energy flows into the body.

    **Stability (exact, dense).** Being explicit, the coupling has a CFL. As for the single-string
    bridge the *cheap* per-part 2-DOF bound is a footgun (a rank-1 spring lifts the top coupled
    eigenvalue above both isolated maxima), so the guard assembles the full coupled leapfrog
    operator ``A`` (``x^{n+1} = 2x^n - x^{n-1} - k^2 A x^n``) over the stacked state (all string
    DOFs then the ``M`` modal coordinates) — matrix-free, from the *same* stencils :meth:`step`
    uses — and requires ``k^2 lambda_max(A) < 4``. ``A`` is small (``sum_j N_j + M``); one dense
    ``eigvals`` at construction is cheap and is the *trusted* guard (no hand-rolled Woodbury for the
    rank-``J`` spring). Each string is run at ``lambda < 1`` (its Nyquist mode is marginal at
    ``lambda = 1`` and a bridge spring pushes it unstable).

    Parameters
    ----------
    strings : list[IdealString]
        The strings (at least one). Each must have its right end ``"free"`` (build with
        ``boundary=("fixed", "free")``) and run at ``lambda < 1``. They may differ in every physical
        parameter; they must only share the timestep ``k`` with the body.
    body : ModalBody
        The shared soundboard/body. Its single driving point is the common bridge where every string
        attaches (its ``phi`` samples the body modes there).
    Ks : array_like
        Per-string bridge stiffnesses ``K_j`` (N/m), one per string. ``K_j = 0`` decouples string
        ``j``; all-zero reproduces the uncoupled parts bit-for-bit.

    Raises
    ------
    ValueError
        If ``strings`` is empty, any timestep differs from the body's, any string's right end is not
        free or runs at ``lambda >= 1``, ``len(Ks) != len(strings)``, any ``K_j < 0``, or the exact
        coupled CFL ``k^2 lambda_max(A) < 4`` is violated.
    """

    def __init__(
        self,
        *,
        strings: list[IdealString],
        body: ModalBody,
        Ks: NDArray[np.float64] | list[float],
    ) -> None:
        strings = list(strings)
        if len(strings) < 1:
            raise ValueError("need at least one string.")
        K_arr = np.atleast_1d(np.asarray(Ks, dtype=float))
        if K_arr.shape != (len(strings),):
            raise ValueError(
                f"Ks must have one stiffness per string (got {K_arr.shape} for "
                f"{len(strings)} strings)."
            )
        if np.any(K_arr < 0.0):
            raise ValueError("every bridge stiffness K must be >= 0.")
        for j, s in enumerate(strings):
            if not np.isclose(s.k, body.k, rtol=0, atol=1e-15):
                raise ValueError(
                    f"string {j} and the body must share a timestep (got k={s.k:.3e} vs "
                    f"{body.k:.3e}); build them at the same fs."
                )
            if s._bc_right != "free":
                raise ValueError(
                    f"string {j}'s right end must be 'free' to attach to the bridge "
                    "(build it with boundary=('fixed', 'free'))."
                )
            if s.lam >= 1.0 - _CFL_TOL:
                raise ValueError(
                    f"string {j} must run at lambda < 1: its Nyquist mode is marginal at "
                    "lambda = 1 and the bridge spring pushes it unstable. Rebuild at lambda < 1."
                )

        self.strings = strings
        self.body = body
        self.K = K_arr
        self.k = body.k
        self.J = len(strings)

        # Per-string end-node inverse mass (its rho_j (h_j/2) half-cell): u_end,j -= beta_s,j F_j.
        self.beta_s = np.array(
            [2.0 * self.k * self.k / (s.rho * s.h) for s in strings]
        )

        # Exact guard: k^2 * lambda_max of the stacked coupled leapfrog operator A must be < 4.
        self._offsets = np.cumsum([0] + [s.N for s in strings])  # string j -> [off_j, off_j+N_j)
        self.spectral_radius = self._max_leapfrog_eigenvalue()
        if self.k * self.k * self.spectral_radius >= 4.0 - _CFL_TOL:
            raise ValueError(
                f"connection unstable: k^2 * lambda_max(A) = "
                f"{self.k * self.k * self.spectral_radius:.6f} >= 4. "
                "Reduce the bridge stiffnesses, raise fs, or increase the body/string end mass."
            )

        self.n = 0

    # -- stability (assembled once at construction, off the hot loop) --------------------

    def _apply_A(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the stacked coupled leapfrog operator ``A`` (``x^{n+1}=2x^n-x^{n-1}-k^2 A x^n``).

        ``x`` stacks each string's DOFs (nodes ``1..N_j``, the clamped left node 0 excluded) in
        order, then the ``M`` modal coordinates. Built from the same stencils :meth:`step` uses, so
        the stability spectrum is guaranteed consistent with the integrator. All strings couple to
        the *shared* bridge displacement ``w_b = sum_i phi_i q_i``.
        """
        b = self.body
        M = b.M
        q = x[-M:] if M else x[0:0]
        w_b = float(np.dot(b.phi, q)) if M else 0.0

        out = np.empty_like(x)
        body_force = 0.0  # sum_j F_j = sum_j K_j eta_j, injected into the modal block
        for j, s in enumerate(self.strings):
            off = self._offsets[j]
            xj = x[off : off + s.N]
            u = np.zeros(s.N + 1)
            u[1:] = xj  # node 0 stays clamped
            out_u = -(s.c * s.c / (s.h * s.h)) * s._second_diff(u)[1:]
            eta = u[-1] - w_b
            out_u[-1] += 2.0 * self.K[j] * eta / (s.rho * s.h)  # (A x)_{end,j} += 2 K eta/(rho h)
            out[off : off + s.N] = out_u
            body_force += self.K[j] * eta

        if M:
            # Body internal A_body q = omega^2 q, plus the shared spring force -phi_i F / m_i.
            out[-M:] = b.omega * b.omega * q - b.phi * body_force / b.m
        return out

    def _max_leapfrog_eigenvalue(self) -> float:
        """Largest eigenvalue of the stacked ``A`` (real, >= 0). ``A`` is small
        (``sum_j N_j + M``); assemble it densely by applying :meth:`_apply_A` to each basis vector
        and take ``max Re(eig)``."""
        n = int(self._offsets[-1]) + self.body.M
        A = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            A[:, j] = self._apply_A(e)
            e[j] = 0.0
        return float(np.max(np.linalg.eigvals(A).real))

    # -- helpers ------------------------------------------------------------------------

    def _bridge_displacement(self, *, prev: bool = False) -> float:
        """Shared bridge displacement ``w_b = sum_i phi_i q_i`` now (``prev`` -> previous step)."""
        if prev:
            return float(np.dot(self.body.phi, self.body.q_prev))
        return self.body.bridge_displacement()

    def _stretch(self, j: int, *, prev: bool = False) -> float:
        """Stretch ``eta_j = u_{end,j} - w_b`` of string ``j``'s spring (``prev`` -> prev step)."""
        w_b = self._bridge_displacement(prev=prev)
        u_end = self.strings[j].u_prev[-1] if prev else self.strings[j].u[-1]
        return float(u_end - w_b)

    def connection_forces(self) -> NDArray[np.float64]:
        """Per-string bridge forces ``F_j = K_j eta_j^n`` (explicit; N)."""
        w_b = self._bridge_displacement()
        return np.array(
            [self.K[j] * (self.strings[j].u[-1] - w_b) for j in range(self.J)]
        )

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep. Each explicit spring ``F_j = K_j eta_j^n`` drives its string and
        the shared body (which feels the scalar sum ``sum_j F_j``) at time ``n``."""
        forces = self.connection_forces()
        for j, s in enumerate(self.strings):
            s.step()
            s.u[-1] -= self.beta_s[j] * forces[j]
        self.body.step(force=float(np.sum(forces)))
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``sum_j E_{string,j} + E_body + sum_j E_{conn,j}`` (Joules).

        ``E_{conn,j} = 1/2 K_j eta_j^n eta_j^{n-1}`` (cross-time). Conserved to machine precision
        for a lossless run; monotonically decreasing when any string or body mode is lossy. No
        single string's energy is conserved once coupled — assert on the total.
        """
        e = self.body.energy()
        for j, s in enumerate(self.strings):
            e += s.energy()
            e += 0.5 * self.K[j] * self._stretch(j) * self._stretch(j, prev=True)
        return e

    def string_energy(self, j: int) -> float:
        """Energy of string ``j`` alone (for the sympathetic-transfer / two-stage-decay traces)."""
        return self.strings[j].energy()

    @property
    def state(self) -> NDArray[np.float64]:
        """The first string's displacement field (a representative resonator for snapshots)."""
        return self.strings[0].state

    def displacement_at(self, index: int, *, string: int = 0) -> float:
        """Pickup at node ``index`` on string ``string`` (for spectral analysis)."""
        return self.strings[string].displacement_at(index)

    def pressure(self) -> float:
        """Radiated pressure from the shared body, ``sum_i a_i q_i''`` (∝ volume acceleration)."""
        return self.body.pressure()
