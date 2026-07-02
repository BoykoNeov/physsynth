"""Bowed string — the first continuous **nonlinear exciter** of the
``exciter -> resonator -> body`` abstraction (HANDOFF §3.2, method-ladder breadth after model #6).

A bow drawn across a :class:`~physsynth.core.string_damped.DampedStiffString` at a point ``x_b``
with surface speed ``v_B`` applies a friction force set by the **relative** velocity between the
string and the bow,

    v_rel = (I delta_t. u) - v_B ,      f_B = -Phi(v_rel) ,

where ``I`` reads the string velocity at the bow node and ``Phi`` is the (smooth, single-hump)
friction characteristic below. This is the classic **stick-slip** mechanism: near ``v_rel = 0`` the
bow drags the string (sticking, injecting energy at rate ``f_B v_B``); past the peak the friction
falls with speed (slipping, the string flies back). The result is self-sustained **Helmholtz
motion** — a travelling corner and a sawtooth velocity at the bow point.

**Why this is a von-Kármán-shaped problem, not a pluck.** The friction force is *nonlinear* and,
crucially, **implicit**: ``Phi`` is evaluated at the *centered* velocity
``delta_t. u = (u^{n+1} - u^{n-1}) / 2k``, which itself depends on ``f_B``. Freezing ``Phi`` at the
known ``u^n`` velocity (an explicit bow) drifts the energy at ``O(k^2)`` and looks non-passive —
the exact trap model #6 taught us. So ``v_rel`` is solved *implicitly* each step.

**The coupling reduces to one scalar equation (rank-1).** The string update is linear in ``f_B``
except through the single bow node, so

    u^{n+1} = u^{n+1}_free + (k^2 / (rho h)) f_B a ,     a = A^{-1} e_i ,

with ``A`` the string's (constant, SPD, banded) update matrix and ``a`` its one-step **driving-point
admittance** (precomputed once via :meth:`DampedStiffString.apply_Ainv`). Reading the bow node then
gives an affine relation ``v_rel = v_free + g f_B`` with the scalar
``g = k a_i / (2 rho h)``, so ``f_B = -Phi(v_rel)`` becomes the scalar root problem

    v_rel = v_free - g Phi(v_rel) ,

solved by a safeguarded Newton iteration (seeded from the previous step's ``v_rel`` —
*continuation*, which follows the physical branch through the multivalued Helmholtz regime). One
banded solve per step (the force-free advance) plus a scalar solve; the force is then applied as the
exact rank-1 correction above. This is *bit-identical* to solving ``A u^{n+1} = rhs_0 + force`` but
needs only the precomputed ``a``.

**Validation is an energy *balance*, not conservation.** The bow is an *active* element (that is how
a note sustains), so ``E`` is not conserved. Instead the discrete work–energy identity holds
exactly (the ``delta_t.``-telescoping every model uses, plus the ``J = (1/h) I^T`` spread/read
duality):

    E^{n+1} - E^n = k f_B (delta_t. u)_i - (string loss) = k * bow_power - loss .

The force ``f_B`` is applied *exactly*, so we report ``bow_power`` from the **true** post-correction
velocity ``v_rel = v_free + g f_B`` — so the balance is machine-precision regardless of the Newton
residual. Newton convergence is a *separate* guarantee that the applied ``f_B`` obeys the friction
law ``f_B = -Phi(v_rel)``. Lossless (``sigma0 = sigma1 = 0``): ``E^n - E^0`` equals the
accumulated bow work to ``~1e-13``. The stick-slip signature (velocity alternating stick ``~v_B`` /
slip, once per period, slip fraction ``~ beta``) is the exciter-specific correctness test.

Headless: NumPy only (delegates the banded solve to the string).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .string_damped import DampedStiffString

__all__ = ["BowedString", "friction_smooth", "friction_smooth_deriv"]


def friction_smooth(v_rel: float, force: float, sharpness: float) -> float:
    """Smooth single-hump friction characteristic ``Phi(v_rel)`` (Newtons).

    ``Phi(v) = force * sqrt(2a) * v * exp(-a v^2 + 1/2)`` with ``a = sharpness``. This is the widely
    used differentiable friction curve (Smith/Serafin; Bilbao): odd, zero at ``v = 0`` with maximum
    slope ``force*sqrt(2a)*e^{1/2}`` there, peaking at ``|Phi| = force`` at ``v = +/- 1/sqrt(2a)``,
    then decaying — the **negative-slope** region past the peak is what makes stick-slip (and the
    multivalued Helmholtz regime) possible. Differentiable everywhere, so Newton behaves; a single
    unique root exists below the Helmholtz threshold and the *continuation*-seeded solve tracks the
    physical branch above it.
    """
    a = sharpness
    return force * math.sqrt(2.0 * a) * v_rel * math.exp(-a * v_rel * v_rel + 0.5)


def friction_smooth_deriv(v_rel: float, force: float, sharpness: float) -> float:
    """Derivative ``Phi'(v_rel)`` of :func:`friction_smooth` (N·s/m)."""
    a = sharpness
    return (
        force * math.sqrt(2.0 * a) * math.exp(-a * v_rel * v_rel + 0.5)
        * (1.0 - 2.0 * a * v_rel * v_rel)
    )


class BowedString:
    """A :class:`DampedStiffString` driven by a bow at a point (nonlinear friction exciter).

    Implements the engine's ``Resonator`` protocol (``step``/``energy``/``state``/``k``/
    ``displacement_at``), delegating the field to the string. The bow stores *no* energy (friction
    is memoryless), so :meth:`energy` is exactly the string's energy; correctness is the *balance*
    of that energy against the accumulated bow work (:attr:`bow_work`).

    Parameters
    ----------
    string : DampedStiffString
        The resonator. Use ``kappa = 0`` (a flexible, fixed-end damped string) to isolate the bow
        physics for the first-pass validation; ``kappa > 0`` (stiff) also works — the admittance
        ``a = A^{-1} e_i`` carries whatever operator ``A`` encodes. A little loss (``sigma0 > 0``)
        lets a bowed note reach a steady Helmholtz regime instead of growing without bound.
    bow_position : float
        Bow contact point in metres, ``0 < bow_position < L``. Snapped to the nearest **interior**
        grid node (sub-grid interpolation is a later refinement).
    v_bow : float
        Bow surface speed (m/s). Sets the Helmholtz amplitude (``~ proportional`` to ``v_bow``).
    force : float
        Bow normal force parameter (N) — the peak of the friction curve ``|Phi|_max = force``.
        Larger ``force`` -> deeper into the multivalued Helmholtz regime (see
        :attr:`helmholtz_number`).
    sharpness : float
        Friction-curve sharpness ``a`` (s^2/m^2): the peak sits at ``v_rel = 1/sqrt(2a)``. Larger
        ``a`` -> a narrower capture range around ``v_rel = 0`` (crisper stick-slip).
    newton_tol : float
        Convergence tolerance on the scalar friction residual (velocity units). Default ``1e-13``.
    newton_maxiter : int
        Maximum safeguarded-Newton iterations per step before giving up (raises). Default ``60``.

    Raises
    ------
    ValueError
        Non-physical parameters, ``bow_position`` not strictly interior, or the bow node landing on
        a boundary.
    RuntimeError
        Only if the friction residual has *no* real root in the guaranteed bracket — impossible for
        the smooth curve (``r(v) = v - v_free + g Phi(v)`` runs from ``-inf`` to ``+inf`` with
        ``Phi`` bounded), so this never fires in practice; a loud backstop like the model-#6 gate.
    """

    def __init__(
        self,
        *,
        string: DampedStiffString,
        bow_position: float,
        v_bow: float,
        force: float,
        sharpness: float = 100.0,
        newton_tol: float = 1e-13,
        newton_maxiter: int = 60,
    ) -> None:
        if force < 0:
            raise ValueError("bow force must be >= 0.")
        if sharpness <= 0:
            raise ValueError("sharpness (a) must be > 0.")
        if newton_maxiter < 1:
            raise ValueError("newton_maxiter must be >= 1.")
        if not (0.0 < bow_position < string.L):
            raise ValueError(
                f"bow_position must satisfy 0 < x < L (L={string.L}), got {bow_position}."
            )

        self.string = string
        self.k = string.k
        self.v_bow = float(v_bow)
        self.force = float(force)
        self.sharpness = float(sharpness)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)

        # Snap the bow to the nearest interior node (node 0 and N are clamped boundaries).
        node = int(round(bow_position / string.h))
        node = min(max(node, 1), string.N - 1)
        self.node = node
        self.x_bow = float(string.x[node])
        self.L = string.L
        # Fractional bow position from the left end; the Helmholtz slip fraction of the period is
        # ~ min(beta, 1 - beta) (verified empirically in the tests). Bookkeeping only, not physics.
        self.beta = self.x_bow / string.L

        # One-step driving-point admittance a = A^{-1} e_i (interior indexing) and the scalar
        # g = k a_i / (2 rho h) that turns the implicit friction into a scalar root problem.
        e_local = np.zeros(string.N - 1)
        e_local[node - 1] = 1.0  # interior node `node` <-> local index node-1
        self._a_vec = string.apply_Ainv(e_local)              # length N-1
        self._a_full = np.zeros(string.N + 1)
        self._a_full[1:-1] = self._a_vec
        a_i = float(self._a_vec[node - 1])
        self._g = self.k * a_i / (2.0 * string.rho * string.h)
        # Force-injection prefactor for the rank-1 correction: u += force_pref * f_B * a_full.
        self._force_pref = self.k * self.k / (string.rho * string.h)

        # Diagnostic: the Helmholtz number g * max|Phi'| = g * force * sqrt(2a) * e^{1/2}. Below 1
        # the scalar equation is single-valued ("surface sound"); above 1 it is multivalued — the
        # regime of real, sustained bowing. NOT a stability limit (friction is bounded; the scheme
        # is stable and the balance exact for any root), so it is reported, never asserted.
        self.helmholtz_number = (
            self._g * self.force * math.sqrt(2.0 * self.sharpness) * math.exp(0.5)
        )

        # Per-step observables (updated by step()); cumulative bow work for the balance test.
        self.v_rel = 0.0
        self.bow_force = 0.0
        self.bow_power = 0.0
        self.bow_work = 0.0
        self.fallbacks = 0  # count of steps that needed the bracketed root fallback (slip events)
        self.n = 0

    # -- friction ------------------------------------------------------------------------

    def _friction(self, v_rel: float) -> float:
        return friction_smooth(v_rel, self.force, self.sharpness)

    def _friction_deriv(self, v_rel: float) -> float:
        return friction_smooth_deriv(v_rel, self.force, self.sharpness)

    def _residual(self, v_rel: float, v_free: float) -> float:
        return v_rel - v_free + self._g * self._friction(v_rel)

    def _solve_v_rel(self, v_free: float) -> tuple[float, bool]:
        """Solve ``r(v) = v - v_free + g Phi(v) = 0`` for the relative velocity.

        Newton seeded from the previous ``v_rel`` (*continuation*) handles the common case in a few
        iterations and keeps the operating point on the physically continuous branch. In the
        multivalued Helmholtz regime a **slip event** makes the current branch's root vanish, so
        Newton is only accepted while it strictly reduces ``|r|``; if it stalls, a guaranteed
        bracketed fallback (:meth:`_bracketed_root`) finds *all* roots and takes the one nearest the
        seed — which, when the stick root has disappeared, is exactly the slip jump. Both give a
        machine-precision root, so ``f_B = -Phi(v_rel)`` is applied exactly either way. Returns
        ``(v_rel, used_fallback)``.
        """
        v = self.v_rel  # continuation seed
        r = self._residual(v, v_free)
        for _ in range(self.newton_maxiter):
            if abs(r) <= self.newton_tol:
                return v, False
            rp = 1.0 + self._g * self._friction_deriv(v)
            if abs(rp) < 1e-30:
                break  # flat spot -> hand off to the robust bracket
            v_new = v - r / rp
            r_new = self._residual(v_new, v_free)
            if not (abs(r_new) < abs(r)):
                break  # no progress (branch vanished) -> robust bracket
            v, r = v_new, r_new
        if abs(r) <= self.newton_tol:
            return v, False
        return self._bracketed_root(v_free), True

    def _bracketed_root(self, v_free: float) -> float:
        """Every root of ``r(v)`` lies in ``|v - v_free| <= g |Phi|_max = g*force``; scan that band
        for sign changes, ``brentq`` each bracket, and return the root nearest the continuation seed
        (the physically-correct branch pick: at a slip the vanished stick root leaves the slip root
        as the nearest). ``r`` is continuous and changes sign, so a root always exists."""
        g, force, a = self._g, self.force, self.sharpness
        span = g * force + 6.0 / math.sqrt(2.0 * a)  # cover all roots + the curve's hump width
        vs = np.linspace(v_free - span, v_free + span, 512)
        rs = vs - v_free + g * (force * math.sqrt(2.0 * a)) * vs * np.exp(-a * vs * vs + 0.5)
        sign_change = np.where(rs[:-1] * rs[1:] < 0.0)[0]
        roots = [
            brentq(self._residual, vs[j], vs[j + 1], args=(v_free,), xtol=1e-15, rtol=8.9e-16)
            for j in sign_change
        ]
        if not roots:
            raise RuntimeError(
                f"bow friction residual has no root in the bracket at step {self.n} "
                "(should be impossible for the bounded smooth friction curve)."
            )
        roots_arr = np.asarray(roots)
        return float(roots_arr[int(np.argmin(np.abs(roots_arr - self.v_rel)))])

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one step: force-free string advance, scalar friction solve, rank-1 correction."""
        i = self.node
        u_prev_i = self.string.u_prev[i]  # u^{n-1}_i (before step() rolls history)

        # Force-free advance (commits u^{n+1}_free, rolls u_prev <- u^n).
        self.string.step()

        # Relative velocity the string would have with no bow force this step (centered).
        v_free = (self.string.u[i] - u_prev_i) / (2.0 * self.k) - self.v_bow
        v_rel, used_fallback = self._solve_v_rel(v_free)
        f_B = -self._friction(v_rel)

        # Apply the force exactly via the precomputed rank-1 admittance.
        self.string.u += self._force_pref * f_B * self._a_full

        # Report the TRUE post-correction velocity so the energy balance is exact irrespective of
        # the Newton residual: v_rel_true = v_free + g f_B (== the Newton root once converged).
        v_true = v_free + self._g * f_B
        self.v_rel = v_true
        self.bow_force = f_B
        self.bow_power = f_B * (v_true + self.v_bow)  # power to the string = f_B * (I d_t. u)
        self.bow_work += self.k * self.bow_power
        self.fallbacks += int(used_fallback)
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Discrete string energy ``E^n`` (Joules). The bow stores none; assert the *balance*
        ``E^n - E^0 == bow_work - loss`` (exact when lossless), not conservation."""
        return self.string.energy()

    @property
    def state(self) -> NDArray[np.float64]:
        """The string displacement field (the vibrating resonator, for animation snapshots)."""
        return self.string.state

    def displacement_at(self, index: int) -> float:
        """String pickup at grid node ``index`` (for spectral analysis of the bowed tone)."""
        return self.string.displacement_at(index)

    def bow_velocity(self) -> float:
        """String transverse velocity at the bow node for the last step (centered ``delta_t. u``,
        m/s) — exactly ``v_rel + v_bow``, the value the friction solve used.

        In the sustained Helmholtz regime this is the classic stick-slip sawtooth: it sits near
        ``v_bow`` during stick and swings away during the once-per-period slip.
        """
        return self.v_rel + self.v_bow
