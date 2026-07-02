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

from .body import ModalBody
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
