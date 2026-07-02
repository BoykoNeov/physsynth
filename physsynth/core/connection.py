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

    Step 4 of the body/radiation node: instead of the lumped :class:`ModalBody`, the radiating body
    is the distributed simply-supported Kirchhoff plate (model #5). The plate has no modal
    coordinates, so the string couples to it at a single **driving-point** node ``dp`` via a point
    force — the *driving-point adapter* the modal bridge did not need. Everything else mirrors
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
    energy-stable iff the conserved quadratic form is positive-definite, i.e.
    ``G = M_aug - (k²/4) S`` is SPD, where ``S`` is the total stiffness (string + plate bending +
    the spring's rank-1 block ``K a a^T``, ``a = e_end - e_dp``) and ``M_aug`` absorbs the plate's
    implicit θ-augmentation (``M_aug`` on the plate block is ``rho_s h² (I + θ k² kappa² B)``). The
    **plate block of ``G`` is always PD**: ``M_aug - (k²/4) S`` reduces there to
    ``rho_s h² [I + (θ - 1/4) k² kappa² B] ⪰ 0`` for ``θ >= 1/4`` — that is *why* the implicit plate
    is unconditionally stable, and it means the plate contributes only headroom. The string block is
    PD for ``lambda < 1``. So the *only* thing that can destabilise ``G`` is the rank-1 spring, and
    a rank-1 negative perturbation ``G_0 - (k²/4) K a a^T`` stays PD iff (Sherman–Morrison)

        (k²/4) K [ (G0_str^{-1})_{end,end} + (G0_plate^{-1})_{dp,dp} ]  <  1 .

    That is the *exact* stability bound (not the 2-DOF estimate that is a footgun for the modal
    bridge), obtained from two small linear solves at construction. The margin is stored on
    :attr:`stability_margin`; the string is run at ``lambda < 1`` (its Nyquist mode sits marginally
    at ``lambda = 1``, and the spring pushes it unstable).

    Parameters
    ----------
    string : IdealString
        The string; its right end must be ``"free"`` (build with ``boundary=("fixed", "free")``).
    plate : Plate
        The body. Must be ``boundary="supported"`` (the driving-point mass ``rho_s h²`` and the
        biharmonic ``B`` are the simply-supported forms used by the guard and the force injection).
    K : float
        Bridge spring stiffness (N/m). ``K = 0`` decouples the two (the bit-identity check).
    drive_index : int, optional
        Live-node index on the plate where the string attaches. Defaults to the node nearest the
        plate centroid offset toward a corner (an off-symmetry point, so it is not a node of the
        low modes). Use :meth:`Plate.pickup_index_at` to choose one explicitly.

    Raises
    ------
    ValueError
        If the two timesteps differ, the string's right end is not free, the plate is not
        ``"supported"``, ``K < 0``, ``drive_index`` is out of range, or the exact stability margin
        is ``>= 1``.
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
        if plate.boundary != "supported":
            raise ValueError(
                "the plate body must be boundary='supported' (the driving-point mass and "
                "biharmonic used by the coupling are the simply-supported forms)."
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

        # Plate block: G0_plate = rho_s h² [I + (θ - 1/4) k² kappa² B] (PD for θ >= 1/4).
        coeff = (p.theta - 0.25) * self.k * self.k * p.kappa * p.kappa
        g_plate = (p.rho * p.h * p.h) * (
            sparse.identity(p.n_live, format="csc") + coeff * p.B
        )
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
        """Radiated pressure from the plate, ``h² sum u_ij''`` (monopole ∝ volume acceleration)."""
        return self.plate.pressure()
