"""Contact & collision primitives + the distributed-barrier string (models #7's home, #8).

This module is the shared home of the **energy-conserving contact scheme** first built for the
mallet (model #7, ``core/mallet.py``) and now reused, promoted from scalar to **vector**, for the
distributed barrier (model #8): a **string vibrating against a one-sided nonlinear barrier** —
string–fret buzz, the tanpura/sitar *jawari* bridge, prepared-piano rattle, a snare snap. See
``docs/dev/collision-barrier-plan.md``.

**The contact potential (a one-sided nonlinear spring).** With penetration ``η`` (``> 0`` in
contact), the felt/barrier stores a potential *density*

    φ(η) = (K/(α+1)) [η]₊^(α+1) ,   φ'(η) = K [η]₊^α ,   [η]₊ = max(η, 0) ,   K > 0, α ≥ 1 .

**Energy-conserving force = the discrete gradient (Chatziioannou–van Walstijn).** Evaluating ``φ'``
at a point drifts the energy at ``O(k²)``. The discrete gradient

    f = (φ(η⁺) − φ(η⁻)) / (η⁺ − η⁻)   → φ'(½(η⁺+η⁻)) when |η⁺−η⁻| < tol   (removable 0/0)

makes the contact power telescope *exactly* (``f·δ_t·η = δ_t·φ``). The Taylor branch is mandatory —
``[DG]`` is a genuine ``0/0`` in the quiet/stick regions and NaNs without it.

**Scalar vs vector coupling.** The mallet's host (the membrane) is *explicit*, so a node force hits
only that node next step → the contact solve is one **scalar** equation (:func:`solve_contact`). The
barrier's host is the *implicit* θ-scheme :class:`~physsynth.core.string_damped.DampedStiffString`,
so a force at contact node ``j`` propagates through ``A⁻¹`` to **every** node. With contact along a
set ``𝒞`` of nodes the unknown is the whole penetration **vector**

    η_𝒞 = η_free,𝒞 − G · F(η_𝒞) ,   G_ij = (k²/ρ)(A⁻¹)_ij   (the string admittance block on 𝒞),

solved by :func:`solve_contact_vector` (damped Newton + Armijo). Well-posedness is *provable*:
``J = I + G·diag(F')`` with ``G`` SPD and ``diag(F') ⪰ 0`` has every eigenvalue ``≥ 1`` everywhere,
so the root is **unique** and Newton converges globally — no branch-picking (unlike the bow).

**Force is a density here (N/m), not a point force.** ``force_pref = k²/ρ`` (NOT ``k²/(ρh)``):
``φ'(η)`` is already a density because the barrier energy ``V = h·Σⱼ φ(ηⱼ)`` makes ``φ`` an energy
density. Update contribution ``(k²/ρ)·A⁻¹F``; energy ``h·Σⱼ φ(ηⱼ)``.

Headless: NumPy + SciPy (delegates the field solve to the string). No I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

from .string_damped import DampedStiffString

__all__ = [
    "BarrierString",
    "contact_potential",
    "contact_force_elastic",
    "contact_stiffness",
    "contact_force_dg",
    "contact_force_total",
    "solve_contact",
    "solve_contact_vector",
]


# -- contact-force primitives (scalar; vector solve reuses these component-wise) -----------------

def contact_potential(eta, K: float, alpha: float):
    """One-sided felt/barrier potential ``φ(η) = K/(α+1) · [η]₊^(α+1)``. Zero for ``η ≤ 0``.

    Vectorized (``np.maximum``): a NumPy array in gives an array out, elementwise.
    """
    ep = np.maximum(eta, 0.0)
    return K / (alpha + 1.0) * ep ** (alpha + 1.0)


def contact_force_elastic(eta, K: float, alpha: float):
    """Elastic contact force ``φ'(η) = K [η]₊^α`` (``≥ 0``). Zero for ``η ≤ 0``. Vectorized."""
    ep = np.maximum(eta, 0.0)
    return K * ep ** alpha


def contact_stiffness(eta, K: float, alpha: float):
    """Contact stiffness ``φ''(η) = K α [η]₊^(α-1)``. Zero for ``η ≤ 0``. Vectorized.

    ``np.where`` guards the ``α = 1`` edge case, where ``[η]₊^0`` would otherwise be ``1`` (even at
    ``η = 0``) and leak a nonzero stiffness into the no-contact region.
    """
    ep = np.maximum(eta, 0.0)
    return np.where(ep > 0.0, K * alpha * ep ** (alpha - 1.0), 0.0)


def contact_force_dg(eta_next: float, eta_prev: float, K: float, alpha: float, tol: float) -> float:
    """Energy-conserving **discrete-gradient** contact force ``[DG]`` (scalar).

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
    """Hunt–Crossley/Stulov hysteretic force ``λ_h ⟦η⟧₊^α · δ_t·η`` (scalar); 0 if ``λ_h = 0``."""
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
    """Total felt/barrier force = elastic discrete gradient + hysteretic damping (scalar)."""
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
    """Solve the **scalar** contact equation ``η = η_free − g · f(η)`` for ``η^{n+1}``.

    ``f`` is the total felt force (:func:`contact_force_total`). The residual is monotone increasing
    in ``η`` (convex potential + non-negative hysteresis weight), so a safeguarded Newton from
    the previous step's penetration (*continuation*) converges fast; a guaranteed bracketed
    fallback (scan + ``brentq``, root nearest the seed) covers any stall. Returns ``(η, f,
    used_fallback)`` with ``f`` the applied force, so the caller injects it exactly. Used by the
    mallet (explicit host) and by the single-active-node collapse of :class:`BarrierString`.
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


# -- vectorized force / derivative (the separable potential -> component-wise scalar [DG]) -------

def _force_total_vec(eta_next, eta_prev, K, alpha, lam_h, k, tol):
    """Elementwise total force on a vector of penetrations — numerically identical to calling
    :func:`contact_force_total` per component (the potential ``V = h·Σⱼ φ(ηⱼ)`` is separable, so
    each node's force is the same scalar discrete gradient applied independently)."""
    da = eta_next - eta_prev
    small = np.abs(da) < tol
    safe = np.where(small, 1.0, da)
    dg = np.where(
        small,
        contact_force_elastic(0.5 * (eta_next + eta_prev), K, alpha),
        (contact_potential(eta_next, K, alpha) - contact_potential(eta_prev, K, alpha)) / safe,
    )
    if lam_h != 0.0:
        mid = np.maximum(0.5 * (eta_next + eta_prev), 0.0)
        w = np.where(mid > 0.0, mid ** alpha, 0.0)
        dg = dg + lam_h * w * da / (2.0 * k)
    return dg


def _deriv_total_vec(eta_next, eta_prev, K, alpha, lam_h, k, tol):
    """Elementwise ``∂F/∂η⁺`` on a vector — the diagonal of the vector Jacobian's force block."""
    da = eta_next - eta_prev
    small = np.abs(da) < tol
    safe = np.where(small, 1.0, da)
    fe = contact_force_elastic(eta_next, K, alpha)
    dphi = contact_potential(eta_next, K, alpha) - contact_potential(eta_prev, K, alpha)
    d = np.where(
        small,
        0.5 * contact_stiffness(0.5 * (eta_next + eta_prev), K, alpha),
        (fe * da - dphi) / (safe * safe),
    )
    if lam_h != 0.0:
        mid = np.maximum(0.5 * (eta_next + eta_prev), 0.0)
        pos = mid > 0.0
        w = np.where(pos, mid ** alpha, 0.0)
        wp = np.where(pos, 0.5 * alpha * mid ** (alpha - 1.0), 0.0)
        d = d + lam_h / (2.0 * k) * (wp * da + w)
    return d


def solve_contact_vector(
    eta_free: NDArray[np.float64],
    eta_prev: NDArray[np.float64],
    G: NDArray[np.float64],
    K: float,
    alpha: float,
    lam_h: float,
    k: float,
    *,
    tol: float,
    seed: NDArray[np.float64],
    newton_tol: float = 1e-13,
    maxiter: int = 60,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Solve the **vector** contact system ``η = η_free − G·F(η)`` over the contact nodes ``𝒞``.

    ``G`` is the dense ``|𝒞|×|𝒞|`` symmetric-PD string admittance block ``(k²/ρ)(A⁻¹)_𝒞``. ``F`` is
    the per-node total force density (:func:`_force_total_vec`). Damped Newton with an Armijo
    line-search on ``½‖r‖²``; the Jacobian ``J = I + G·diag(F')`` has ``λ_min ≥ 1`` everywhere, so
    the root is unique and Newton converges globally — no branch selection, no ``brentq`` fallback
    (the scalar-only bracket does not transfer to a vector). Inactive nodes (``η < 0``) self-select
    out: ``F = F' = 0`` there ⇒ that row of ``J`` is the identity. Returns ``(η, F, iters)``.
    """
    m = eta_free.shape[0]
    eye = np.eye(m)

    def residual(eta):
        return eta - eta_free + G @ _force_total_vec(eta, eta_prev, K, alpha, lam_h, k, tol)

    eta = np.array(seed, dtype=float)
    r = residual(eta)
    for it in range(maxiter):
        if np.max(np.abs(r)) <= newton_tol:
            f = _force_total_vec(eta, eta_prev, K, alpha, lam_h, k, tol)
            return eta, f, it
        fp = _deriv_total_vec(eta, eta_prev, K, alpha, lam_h, k, tol)
        jac = eye + G * fp[np.newaxis, :]  # G @ diag(F'): scale column j of G by F'_j
        # scipy LU rather than np.linalg.solve: the latter hits a catastrophic threaded-BLAS
        # slowdown on some platforms above ~100x100 (0.05 ms -> 250 ms); scipy stays ~1 ms.
        delta = lu_solve(lu_factor(jac), -r)
        # Armijo backtracking on 0.5||r||^2 (guards only the [η]₊ semismooth kink).
        f0 = 0.5 * float(r @ r)
        t = 1.0
        for _ls in range(40):
            r_try = residual(eta + t * delta)
            if 0.5 * float(r_try @ r_try) < (1.0 - 1e-4 * t) * f0:
                break
            t *= 0.5
        eta = eta + t * delta
        r = residual(eta)
    # Reached the iteration cap without converging. The applied force is exact only *at* the root
    # (energy conservation depends on it — see the drift-vs-newton_tol test), so a stall would
    # silently corrupt the energy balance; warn rather than fail so a long render still finishes.
    if np.max(np.abs(r)) > newton_tol:
        import warnings
        warnings.warn(
            f"vector contact solve did not converge in {maxiter} iterations "
            f"(residual {np.max(np.abs(r)):.2e} > {newton_tol:.1e}); energy may drift. Raise "
            f"newton_maxiter or oversample the contact.",
            stacklevel=2,
        )
    f = _force_total_vec(eta, eta_prev, K, alpha, lam_h, k, tol)
    return eta, f, maxiter


class BarrierString:
    """A :class:`DampedStiffString` vibrating against a one-sided distributed barrier ``b(x)``.

    Implements the engine's ``Resonator`` protocol (``step``/``energy``/``state``/``k``/
    ``displacement_at``), delegating the field to the string. The barrier stores potential energy
    ``h·Σⱼ φ(ηⱼ)``, so :meth:`energy` returns the **total** conserved quantity ``E_string +
    barrier PE``; correctness is its *conservation* (lossless, ``σ = λ_h = 0``) or *passivity*
    (``σ > 0`` or ``λ_h > 0``).

    The barrier is a fixed profile ``b(x)`` below the string; penetration ``η(x) = b(x) − u(x)`` is
    ``> 0`` where the string has swung *into* the barrier. Contact is a one-sided nonlinear spring
    (:func:`contact_potential`). Because the string is *implicit*, the per-step contact force is the
    solution of the coupled **vector** system :func:`solve_contact_vector` through the string's
    driving-point admittance block on the contact nodes.

    Parameters
    ----------
    string : DampedStiffString
        The resonator. Use ``sigma0 = sigma1 = 0`` for the lossless conservation money test.
    barrier : float or array of shape ``(N+1,)``
        Barrier height ``b`` at each grid node. A scalar is a flat rail at that height under the
        whole string. Nodes whose barrier is non-finite (``±inf``/``nan``) are excluded from the
        contact support (use this for a point fret: one finite node, the rest ``-inf``).
    stiffness : float
        Contact stiffness ``K`` (N/m^α).
    alpha : float
        Contact exponent ``α ≥ 1`` (``1`` = linear, the closed-form-oracle case; ``≈ 1.5–2`` real).
    hysteresis : float
        Hunt–Crossley damping ``λ_h ≥ 0``. ``0`` -> lossless elastic barrier (conserves); ``> 0``
        -> lossy (passive).
    eta_tol : float
        Discrete-gradient Taylor-branch threshold. Default ``1e-12``.
    newton_tol, newton_maxiter : float, int
        Vector-solve tolerance / iteration cap (defaults ``1e-13`` / ``60``).

    Raises
    ------
    ValueError
        Non-physical parameters, a bad barrier shape, or an empty contact support.
    """

    def __init__(
        self,
        *,
        string: DampedStiffString,
        barrier,
        stiffness: float,
        alpha: float = 1.5,
        hysteresis: float = 0.0,
        eta_tol: float = 1e-12,
        newton_tol: float = 1e-13,
        newton_maxiter: int = 60,
    ) -> None:
        if stiffness <= 0.0:
            raise ValueError("contact stiffness K must be > 0.")
        if alpha < 1.0:
            raise ValueError("contact exponent alpha must be >= 1.")
        if hysteresis < 0.0:
            raise ValueError("hysteresis lambda_h must be >= 0.")

        self.string = string
        self.k = string.k
        self.K = float(stiffness)
        self.alpha = float(alpha)
        self.lam_h = float(hysteresis)
        self.eta_tol = float(eta_tol)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)

        N = string.N
        b_full = np.broadcast_to(np.asarray(barrier, dtype=float), (N + 1,)).astype(float)
        if b_full.shape != (N + 1,):
            raise ValueError(f"barrier must be scalar or shape {(N + 1,)}, got {b_full.shape}.")
        # Support = interior nodes (1..N-1) where the barrier is defined (finite).
        interior = np.arange(1, N)
        finite = np.isfinite(b_full[interior])
        self._support = interior[finite]  # grid node indices
        if self._support.size == 0:
            raise ValueError("barrier has no finite interior node -> empty contact support.")
        self._b = b_full[self._support]  # barrier heights on the support (length m)
        self._int_idx = self._support - 1  # interior-array indices (0..N-2) of the support nodes

        # Driving-point admittance block G = (k^2/rho) (A^-1)_support, built once (A is fixed).
        # force_pref = k^2/rho (force DENSITY: phi' is N/m since barrier energy is h*sum phi).
        self._force_pref = string.k ** 2 / string.rho
        m = self._support.size
        cols = []
        for nd in self._support:
            e = np.zeros(N - 1)
            e[nd - 1] = 1.0
            cols.append(string.apply_Ainv(e))  # A^-1 e_nd, length N-1 (full interior column)
        # Columns as a matrix ``(N-1, m)`` so the rank-m correction is one matmul, not a loop:
        # ``u_int += force_pref * (cols_mat @ F)`` == ``force_pref * sum_j F_j a_j``.
        self._cols_mat = np.array(cols).T
        # Admittance block G = force_pref * (A^-1)_{support,support} (rows picked at the support).
        self._G = self._force_pref * self._cols_mat[self._int_idx, :]

        # Per-step observables + continuation seed (penetration on the support at eta^0).
        self.penetration = self._b - string.u[self._support]  # eta^0 (length m)
        self.contact_force = np.zeros(m)
        self.newton_iters = 0
        self.n = 0

    # -- initial conditions -------------------------------------------------------------

    def set_state(self, u0, v0=0.0) -> None:
        """Set the string's initial displacement/velocity (delegates to the string), then refresh
        the continuation seed. Clamps and the consistent second-order start are the string's."""
        self.string.set_state(u0, v0)
        self.penetration = self._b - self.string.u[self._support]
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one step: force-free string advance, vector contact solve, exact force inject."""
        s = self.string
        eta_prev = self._b - s.u_prev[self._support]  # eta^{n-1} (before the roll)

        # Force-free advance: string commits u_free^{n+1}, rolls u_prev <- u^n.
        s.step()
        u_free = s.u[self._support]
        eta_free = self._b - u_free

        # Vector contact solve, then apply the rank-m force correction exactly.
        eta_next, f, iters = solve_contact_vector(
            eta_free, eta_prev, self._G, self.K, self.alpha, self.lam_h, self.k,
            tol=self.eta_tol, seed=self.penetration,
            newton_tol=self.newton_tol, maxiter=self.newton_maxiter,
        )
        s.u[1:-1] = s.u[1:-1] + self._force_pref * (self._cols_mat @ f)

        self.penetration = eta_next
        self.contact_force = f
        self.newton_iters = iters
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E^n`` (Joules): string energy + averaged barrier PE.

        Lossless + elastic (``σ = 0, λ_h = 0``) -> conserved to machine precision; ``σ > 0`` or
        ``λ_h > 0`` -> monotone decreasing (passive). The barrier PE is the **two-time-average**
        ``h·Σⱼ ½(φ(ηⱼ^n)+φ(ηⱼ^{n-1}))`` — the form that telescopes with the discrete-gradient force.
        """
        s = self.string
        eta_n = self._b - s.u[self._support]
        eta_nm1 = self._b - s.u_prev[self._support]
        pe = 0.5 * s.h * float(
            np.sum(contact_potential(eta_n, self.K, self.alpha))
            + np.sum(contact_potential(eta_nm1, self.K, self.alpha))
        )
        return s.energy() + pe

    @property
    def state(self) -> NDArray[np.float64]:
        """The string displacement field ``u^n`` (a copy, for animation snapshots)."""
        return self.string.state

    def displacement_at(self, index: int) -> float:
        """String pickup at grid node ``index`` (for spectral analysis of the tone)."""
        return self.string.displacement_at(index)

    def contact_mask(self) -> NDArray[np.bool_]:
        """Boolean mask over the support: which nodes are currently in contact (``η > 0``)."""
        return self.penetration > 0.0
