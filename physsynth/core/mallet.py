"""Mallet–membrane collision — the first **contact/collision** model (HANDOFF §12.B; model #7).

A soft mallet (a lumped mass ``M``) strikes a :class:`~physsynth.core.membrane.Membrane` at a point,
the drum/timpani gesture. Unlike the bow (a *memoryless* nonlinear exciter whose correctness is an
energy *balance*), the mallet is a **mass in one-sided nonlinear contact**: it stores kinetic energy
``½M v_H²`` and — through the felt — **potential** energy ``φ(η)``. So the money test flips back to
strict **conservation**:

    H^n = E_membrane^n  +  ½ M (δ_t- z_H^n)²  +  ½ ( φ(η^n) + φ(η^{n-1}) )  =  const

(lossless, elastic felt), and monotone **decreasing** once the membrane loses energy or the felt is
hysteretic. See ``docs/dev/hammer-collision-plan.md``.

**Geometry / sign convention.** ``z_H`` is the mallet position and ``u_S`` the head surface at the
contact node, in a common ``+z`` (up) frame; the mallet comes from above. The **penetration** is

    η = u_S − z_H        (η > 0  ⟺  the mallet tip has dipped below the head surface, i.e. contact)

The felt is a one-sided nonlinear spring with **contact potential** and (repulsive, ``≥ 0``) force

    φ(η) = (K / (α+1)) [η]₊^(α+1) ,      φ'(η) = K [η]₊^α ,      [η]₊ = max(η, 0)

pushing the mallet up (``M z_H'' = +φ'``) and the head down (``ρ u_tt = elastic − φ'/h²``).

**Energy-conserving force = the discrete gradient (Chatziioannou–van Walstijn).** Evaluating
``φ'`` at a single point drifts the energy at ``O(k²)`` (the trap model #6 taught us). Instead the
force is the *discrete gradient* of the potential,

    f = ( φ(η^{n+1}) − φ(η^{n-1}) ) / ( η^{n+1} − η^{n-1} )                    [DG]
      → φ'( ½(η^{n+1}+η^{n-1}) )              when |η^{n+1} − η^{n-1}| < η_tol   (removable 0/0),

which makes the contact power telescope *exactly*: ``f · δ_t·η = δ_t· φ``. The ``η_tol`` branch is
mandatory — ``[DG]`` is a genuine ``0/0`` in the quiet/stick regions and NaNs without it.

**Hunt–Crossley/Stulov hysteresis (passivity, not conservation).** An optional velocity-dependent,
penetration-gated damping ``f_hyst = λ_h ⟦η⟧₊^α · δ_t·η`` (``⟦η⟧ = ½(η^{n+1}+η^{n-1})``) models the
lossy felt. It is **not** potential-derived; dissipation ``f_hyst·δ_t·η = λ_h⟦η⟧₊^α(δ_t·η)² ≥ 0``
is sign-definite, so it only ever *removes* energy. ``λ_h = 0`` recovers the conservative scheme.

**The coupling reduces to one scalar equation (the bow shape).** Both DOFs are linear in ``f``
except through the contact, so with the membrane's **local** admittance (it is *explicit*, so a
node force touches only that node next step) and the mallet's,

    η^{n+1} = η_free^{n+1} − g f(η^{n+1}) ,   g = g_s + g_H ,
    g_s = k² / (ρ h² (1+σk))   (membrane node) ,   g_H = k² / M   (mallet) .

The residual ``r(η) = η − η_free + g f(η)`` is **monotone increasing** (convex potential ⇒ ``f``
non-decreasing), so there is a unique root — a safeguarded Newton seeded by continuation, with a
guaranteed bracketed fallback, converges cleanly (no multivalued branch, unlike the bow). ``f`` is
applied *exactly* (local membrane correction + closed-form mallet update), so the reported energy is
machine-precision regardless of the Newton residual.

Headless: NumPy + SciPy (delegates the field solve to the membrane). No I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .membrane import Membrane

__all__ = [
    "MalletMembrane",
    "MalletWall",
    "contact_potential",
    "contact_force_elastic",
    "contact_stiffness",
    "contact_force_dg",
    "contact_force_total",
    "solve_contact",
]


# -- contact-force primitives (vector-ready: scalars are the size-1 case) ------------------------

def contact_potential(eta, K: float, alpha: float):
    """One-sided felt potential ``φ(η) = K/(α+1) · [η]₊^(α+1)`` (Joules). Zero for ``η ≤ 0``."""
    ep = np.maximum(eta, 0.0)
    return K / (alpha + 1.0) * ep ** (alpha + 1.0)


def contact_force_elastic(eta, K: float, alpha: float):
    """Elastic contact force ``φ'(η) = K [η]₊^α`` (Newtons, ``≥ 0``). Zero for ``η ≤ 0``."""
    ep = np.maximum(eta, 0.0)
    return K * ep ** alpha


def contact_stiffness(eta, K: float, alpha: float):
    """Contact stiffness ``φ''(η) = K α [η]₊^(α-1)`` (N/m). Zero for ``η ≤ 0``.

    ``np.where`` guards the ``α = 1`` edge case, where ``[η]₊^0`` would otherwise be ``1`` (even at
    ``η = 0``) and leak a nonzero stiffness into the no-contact region.
    """
    ep = np.maximum(eta, 0.0)
    return np.where(ep > 0.0, K * alpha * ep ** (alpha - 1.0), 0.0)


def contact_force_dg(eta_next: float, eta_prev: float, K: float, alpha: float, tol: float) -> float:
    """Energy-conserving **discrete-gradient** contact force ``[DG]`` (Newtons).

    ``(φ(η⁺) − φ(η⁻)) / (η⁺ − η⁻)`` with the removable ``0/0`` handled by the midpoint-derivative
    Taylor branch ``φ'(½(η⁺+η⁻))`` when the denominator is below ``tol`` (stick / grazing).
    """
    da = eta_next - eta_prev
    if abs(da) < tol:
        return float(contact_force_elastic(0.5 * (eta_next + eta_prev), K, alpha))
    return float(
        (contact_potential(eta_next, K, alpha) - contact_potential(eta_prev, K, alpha)) / da
    )


def _contact_force_dg_deriv(
    eta_next: float, eta_prev: float, K: float, alpha: float, tol: float
) -> float:
    """``∂/∂η⁺`` of :func:`contact_force_dg` (N/m) — for the Newton iteration."""
    da = eta_next - eta_prev
    if abs(da) < tol:
        return float(0.5 * contact_stiffness(0.5 * (eta_next + eta_prev), K, alpha))
    fe = float(contact_force_elastic(eta_next, K, alpha))  # φ'(η⁺)
    phi_next = float(contact_potential(eta_next, K, alpha))
    phi_prev = float(contact_potential(eta_prev, K, alpha))
    return float((fe * da - (phi_next - phi_prev)) / (da * da))


def _contact_force_hyst(
    eta_next: float, eta_prev: float, alpha: float, lam_h: float, k: float
) -> float:
    """Hunt–Crossley/Stulov hysteretic force ``λ_h ⟦η⟧₊^α · δ_t·η`` (Newtons); 0 if ``λ_h = 0``."""
    if lam_h == 0.0:
        return 0.0
    mid = max(0.5 * (eta_next + eta_prev), 0.0)
    w = mid ** alpha if mid > 0.0 else 0.0
    return float(lam_h * w * (eta_next - eta_prev) / (2.0 * k))


def _contact_force_hyst_deriv(
    eta_next: float, eta_prev: float, alpha: float, lam_h: float, k: float
) -> float:
    """``∂/∂η⁺`` of :func:`_contact_force_hyst` (N/m)."""
    if lam_h == 0.0:
        return 0.0
    mid = max(0.5 * (eta_next + eta_prev), 0.0)
    if mid <= 0.0:
        return 0.0
    w = mid ** alpha
    wp = 0.5 * alpha * mid ** (alpha - 1.0)  # ∂w/∂η⁺
    return float(lam_h / (2.0 * k) * (wp * (eta_next - eta_prev) + w))


def contact_force_total(
    eta_next: float, eta_prev: float, K: float, alpha: float, lam_h: float, k: float, tol: float
) -> float:
    """Total felt force = elastic discrete gradient + hysteretic damping (Newtons)."""
    return (
        contact_force_dg(eta_next, eta_prev, K, alpha, tol)
        + _contact_force_hyst(eta_next, eta_prev, alpha, lam_h, k)
    )


def _contact_force_total_deriv(
    eta_next: float, eta_prev: float, K: float, alpha: float, lam_h: float, k: float, tol: float
) -> float:
    return (
        _contact_force_dg_deriv(eta_next, eta_prev, K, alpha, tol)
        + _contact_force_hyst_deriv(eta_next, eta_prev, alpha, lam_h, k)
    )


def solve_contact(
    eta_free: float,
    eta_prev: float,
    g: float,
    K: float,
    alpha: float,
    lam_h: float,
    k: float,
    *,
    tol: float,
    seed: float,
    newton_tol: float = 1e-14,
    maxiter: int = 60,
) -> tuple[float, float, bool]:
    """Solve the scalar contact equation ``η = η_free − g · f(η)`` for the penetration ``η^{n+1}``.

    ``f`` is the total felt force (:func:`contact_force_total`). The residual is monotone increasing
    in ``η`` (convex potential + non-negative hysteresis weight), so a safeguarded Newton from
    the previous step's penetration (*continuation*) converges fast; a guaranteed bracketed
    fallback (scan + ``brentq``, root nearest the seed) covers any stall. Returns ``(η, f,
    used_fallback)`` with ``f`` the applied force, so the caller injects it exactly.
    """
    def resid(eta: float) -> tuple[float, float]:
        f = contact_force_total(eta, eta_prev, K, alpha, lam_h, k, tol)
        return eta - eta_free + g * f, f

    eta = seed
    r, f = resid(eta)
    for _ in range(maxiter):
        if abs(r) <= newton_tol:
            return eta, f, False
        rp = 1.0 + g * _contact_force_total_deriv(eta, eta_prev, K, alpha, lam_h, k, tol)
        if abs(rp) < 1e-30:
            break
        eta_new = eta - r / rp
        r_new, f_new = resid(eta_new)
        if not (abs(r_new) < abs(r)):
            break  # no progress -> hand off to the robust bracket
        eta, r, f = eta_new, r_new, f_new
    if abs(r) <= newton_tol:
        return eta, f, False

    # Bracketed fallback: scan a band around eta_free for a sign change, brentq, pick nearest seed.
    f_free = contact_force_total(eta_free, eta_prev, K, alpha, lam_h, k, tol)
    span = abs(g * f_free) + abs(eta_free - eta_prev) + 1e-12
    for _ in range(6):
        vs = np.linspace(eta_free - span, eta_free + span, 1025)
        rs = np.array([resid(v)[0] for v in vs])
        idx = np.where(rs[:-1] * rs[1:] < 0.0)[0]
        if len(idx):
            roots = [brentq(lambda e: resid(e)[0], vs[j], vs[j + 1], xtol=1e-15, rtol=8.9e-16)
                     for j in idx]
            roots_arr = np.asarray(roots)
            eta_b = float(roots_arr[int(np.argmin(np.abs(roots_arr - seed)))])
            return eta_b, resid(eta_b)[1], True
        span *= 10.0
    raise RuntimeError(
        "contact residual has no root in the bracket (should be impossible for the monotone "
        "convex-potential force)."
    )


class MalletMembrane:
    """A :class:`Membrane` struck by a lumped-mass mallet through a nonlinear felt.

    Implements the engine's ``Resonator`` protocol (``step``/``energy``/``state``/``k``/
    ``displacement_at``), delegating the field to the membrane. The mallet stores kinetic *and*
    (through the felt) potential energy, so :meth:`energy` returns the **total** conserved quantity
    ``H`` (membrane + mallet KE + averaged contact PE); correctness is its *conservation* (lossless,
    ``λ_h = 0``) or *passivity* (``σ > 0`` or ``λ_h > 0``).

    Parameters
    ----------
    membrane : Membrane
        The resonator (the drumhead). Use ``sigma = 0`` for the lossless conservation money test.
    mass : float
        Mallet mass ``M`` (kg).
    stiffness : float
        Felt stiffness ``K`` (N/m^α). The contact frequency ``√(K/M)`` sets the temporal resolution
        the strike needs; a warning fires if it is under-resolved at the membrane's ``fs``.
    alpha : float
        Felt exponent ``α ≥ 1`` (``1`` = linear felt, the closed-form-oracle case; ``≈ 2–3`` = real
        felt). Any ``α ≥ 1`` conserves energy.
    hysteresis : float
        Hunt–Crossley damping ``λ_h ≥ 0`` (N·s/m^(α+1)). ``0`` -> lossless elastic felt (conserves);
        ``> 0`` -> a lossy felt (passive, energy decreases).
    strike_x, strike_y : float
        Contact point (m); snapped to the nearest **live** membrane node.
    strike_velocity : float
        Mallet impact speed toward the head (m/s, ``> 0`` = into the head).
    gap : float
        Initial mallet–surface separation (m, ``≥ 0``). ``0`` -> contact begins immediately.
    eta_tol : float
        Denominator threshold below which the discrete gradient uses its Taylor branch. Default
        ``1e-12``.
    newton_tol, newton_maxiter : float, int
        Scalar-solve tolerance / iteration cap (defaults ``1e-14`` / ``60``).

    Raises
    ------
    ValueError
        Non-physical parameters or a strike point with no live node.
    """

    def __init__(
        self,
        *,
        membrane: Membrane,
        mass: float,
        stiffness: float,
        alpha: float = 2.3,
        hysteresis: float = 0.0,
        strike_x: float,
        strike_y: float,
        strike_velocity: float,
        gap: float = 0.0,
        eta_tol: float = 1e-12,
        newton_tol: float = 1e-14,
        newton_maxiter: int = 60,
    ) -> None:
        if mass <= 0.0:
            raise ValueError("mallet mass must be > 0.")
        if stiffness <= 0.0:
            raise ValueError("felt stiffness K must be > 0.")
        if alpha < 1.0:
            raise ValueError("felt exponent alpha must be >= 1.")
        if hysteresis < 0.0:
            raise ValueError("hysteresis lambda_h must be >= 0.")
        if gap < 0.0:
            raise ValueError("initial gap must be >= 0.")

        self.membrane = membrane
        self.k = membrane.k
        self.M = float(mass)
        self.K = float(stiffness)
        self.alpha = float(alpha)
        self.lam_h = float(hysteresis)
        self.eta_tol = float(eta_tol)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)

        # Snap the strike to the nearest live node; record its physical location.
        self.node = membrane.pickup_index_at(strike_x, strike_y)
        live = membrane.index_map >= 0
        self.x_strike = float(membrane.X[live][self.node])
        self.y_strike = float(membrane.Y[live][self.node])

        # Driving-point admittances. The membrane is explicit, so a node force hits only that node
        # next step -> the admittance is the bare local nodal mass (no A^-1 solve). The (1+sigma k)
        # carries the loss factor the force-free step already applied.
        sk = membrane.sigma * membrane.k
        self._g_s = membrane.k ** 2 / (membrane.rho * membrane.h ** 2 * (1.0 + sk))
        self._g_h = membrane.k ** 2 / self.M
        self._g = self._g_s + self._g_h

        # Mallet state: start `gap` above the (at-rest) head, moving in at `strike_velocity`. The
        # pre-contact flight is force-free, so u_H^{-1} = u_H^0 - k*(dz_H/dt) is exact (accel 0).
        self.z_H = float(gap)
        self.z_H_prev = float(gap) + self.k * float(strike_velocity)  # dz_H/dt|_0 = -strike_vel
        self.strike_velocity = float(strike_velocity)

        # Contact resolution guard: the felt half-period pi*sqrt(M/K) should span several steps.
        self.contact_frequency = float(np.sqrt(self.K / self.M) / (2.0 * np.pi))  # Hz (alpha=1 ref)
        steps_per_contact = np.pi * np.sqrt(self.M / self.K) / self.k
        if steps_per_contact < 8.0:
            import warnings
            warnings.warn(
                f"stiff contact under-resolved: ~{steps_per_contact:.1f} steps per half-period "
                f"(want >= 8). Raise fs or lower K/increase M to avoid aliasing the strike.",
                stacklevel=2,
            )

        # Per-step observables + continuation seed.
        self.penetration = self.membrane.u[self.node] - self.z_H  # eta^0
        self.contact_force = 0.0
        self.in_contact = self.penetration > 0.0
        self.fallbacks = 0
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one step: force-free advance, scalar contact solve, exact force inject."""
        i = self.node
        mem = self.membrane
        zH_n, zH_nm1 = self.z_H, self.z_H_prev
        eta_prev = mem.u_prev[i] - zH_nm1  # eta^{n-1}

        # Force-free advance: membrane (commits u_free^{n+1}, rolls u_prev <- u^n) + mallet flight.
        mem.step()
        u_free = mem.u[i]
        zH_free = 2.0 * zH_n - zH_nm1
        eta_free = u_free - zH_free  # eta_free^{n+1}

        # Scalar contact solve (continuation-seeded), then apply the force exactly.
        eta_next, f, used_fb = solve_contact(
            eta_free, eta_prev, self._g, self.K, self.alpha, self.lam_h, self.k,
            tol=self.eta_tol, seed=self.penetration,
            newton_tol=self.newton_tol, maxiter=self.newton_maxiter,
        )
        mem.u[i] = u_free - self._g_s * f     # head pushed by -f (local, exact)
        self.z_H_prev = zH_n
        self.z_H = zH_free + self._g_h * f     # mallet pushed by +f

        self.penetration = eta_next
        self.contact_force = f
        self.in_contact = eta_next > 0.0
        self.fallbacks += int(used_fb)
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``H^n`` (Joules): membrane + mallet KE + averaged contact PE.

        Lossless + elastic (``σ = 0, λ_h = 0``) -> conserved to machine precision; ``σ > 0`` or
        ``λ_h > 0`` -> monotone decreasing (passive). The contact PE is the **two-time-average**
        ``½(φ(η^n)+φ(η^{n-1}))`` — the form that telescopes with the discrete-gradient force.
        """
        i = self.node
        mem = self.membrane
        eta_n = mem.u[i] - self.z_H
        eta_nm1 = mem.u_prev[i] - self.z_H_prev
        ke = 0.5 * self.M * ((self.z_H - self.z_H_prev) / self.k) ** 2
        pe = 0.5 * (
            float(contact_potential(eta_n, self.K, self.alpha))
            + float(contact_potential(eta_nm1, self.K, self.alpha))
        )
        return mem.energy() + ke + pe

    @property
    def state(self) -> NDArray[np.float64]:
        """The membrane displacement field (full 2D array, for animation snapshots)."""
        return self.membrane.state

    def displacement_at(self, index: int) -> float:
        """Membrane pickup at flat live-node ``index`` (for spectral analysis of the tone)."""
        return self.membrane.displacement_at(index)

    def mallet_velocity(self) -> float:
        """Mallet velocity ``δ_t- z_H`` (m/s): negative moving into the head, positive after it
        rebounds."""
        return (self.z_H - self.z_H_prev) / self.k


class MalletWall:
    """A lumped mass in one-sided contact with a **fixed rigid wall** — the standalone rig.

    This is the collision scheme with the resonator removed (``g_s = 0``, ``u_S`` fixed), so the
    closed-form oracle lives here: a mass on a *fixed* linear spring (``α = 1``, ``λ_h = 0``) is a
    half-period of ``ω = √(K/M)``: contact lasts ``π√(M/K)`` with **exact velocity reversal**
    (coefficient of restitution 1). It reuses :func:`solve_contact` verbatim, so it de-risks the
    discrete-gradient scheme in isolation before any coupling — the analog of unit-testing the VK
    bracket before the time loop. With ``λ_h > 0`` the felt is lossy: energy decreases monotonically
    and the restitution drops below 1.

    Penetration convention matches :class:`MalletMembrane` with the wall as the surface:
    ``η = wall_position − z_H`` (``η > 0`` once the mallet tip passes the wall).
    """

    def __init__(
        self,
        *,
        mass: float,
        stiffness: float,
        fs: float,
        alpha: float = 1.0,
        hysteresis: float = 0.0,
        wall_position: float = 0.0,
        strike_velocity: float,
        gap: float = 0.0,
        eta_tol: float = 1e-12,
        newton_tol: float = 1e-14,
        newton_maxiter: int = 60,
    ) -> None:
        if mass <= 0.0:
            raise ValueError("mallet mass must be > 0.")
        if stiffness <= 0.0:
            raise ValueError("felt stiffness K must be > 0.")
        if alpha < 1.0:
            raise ValueError("felt exponent alpha must be >= 1.")
        if hysteresis < 0.0:
            raise ValueError("hysteresis lambda_h must be >= 0.")
        if gap < 0.0:
            raise ValueError("initial gap must be >= 0.")

        self.M = float(mass)
        self.K = float(stiffness)
        self.alpha = float(alpha)
        self.lam_h = float(hysteresis)
        self.wall = float(wall_position)
        self.k = 1.0 / float(fs)
        self.eta_tol = float(eta_tol)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)

        self._g = self.k ** 2 / self.M  # only the mallet admittance; the wall is rigid (g_s = 0)
        # Start `gap` above the wall, moving in at `strike_velocity` (force-free -> exact u^-1).
        self.z_H = self.wall + float(gap)
        self.z_H_prev = self.z_H + self.k * float(strike_velocity)
        self.strike_velocity = float(strike_velocity)

        self.penetration = self.wall - self.z_H
        self.contact_force = 0.0
        self.in_contact = self.penetration > 0.0
        self.fallbacks = 0
        self.n = 0

    def step(self) -> None:
        """Advance one step: force-free mallet flight, scalar contact solve, exact force inject."""
        z_n, z_nm1 = self.z_H, self.z_H_prev
        eta_prev = self.wall - z_nm1
        z_free = 2.0 * z_n - z_nm1
        eta_free = self.wall - z_free
        eta_next, f, used_fb = solve_contact(
            eta_free, eta_prev, self._g, self.K, self.alpha, self.lam_h, self.k,
            tol=self.eta_tol, seed=self.penetration,
            newton_tol=self.newton_tol, maxiter=self.newton_maxiter,
        )
        self.z_H_prev = z_n
        self.z_H = z_free + self._g * f  # mallet pushed by +f
        self.penetration = eta_next
        self.contact_force = f
        self.in_contact = eta_next > 0.0
        self.fallbacks += int(used_fb)
        self.n += 1

    def energy(self) -> float:
        """Total energy ``½M(δ_t- z_H)² + ½(φ(η^n)+φ(η^{n-1}))`` (J). Conserved (``λ_h = 0``) or
        monotone decreasing (``λ_h > 0``)."""
        eta_n = self.wall - self.z_H
        eta_nm1 = self.wall - self.z_H_prev
        ke = 0.5 * self.M * ((self.z_H - self.z_H_prev) / self.k) ** 2
        pe = 0.5 * (
            float(contact_potential(eta_n, self.K, self.alpha))
            + float(contact_potential(eta_nm1, self.K, self.alpha))
        )
        return ke + pe

    def velocity(self) -> float:
        """Mallet velocity ``δ_t- z_H`` (m/s): ``−strike_velocity`` inbound, ``+`` after rebound."""
        return (self.z_H - self.z_H_prev) / self.k
