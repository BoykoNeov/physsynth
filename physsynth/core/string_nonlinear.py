"""Tension-modulated string — Kirchhoff-Carrier nonlinearity (model #9).

The **string family's nonlinearity**, and the direct analog of model #6's von Karman plate: a
geometric (not material) nonlinearity with a **quartic** potential, validated energy-first. Models
#1-3 are strictly linear -- pluck twice as hard, get exactly twice the amplitude at exactly the same
pitch. Real strings don't: displacing a string *stretches* it, which raises its tension, which
raises its pitch. Hit one hard and the note starts sharp and glides down as it decays.

PDE (model #3 plus a state-dependent tension):

    rho u_tt = T(t) u_xx - rho kappa^2 u_xxxx - 2 rho sigma0 u_t + 2 rho sigma1 u_txx
    T(t) = T0 + (EA/2L) I,        I = integral_0^L u_x^2 dx        ("the stretch")

``I >= 0`` always, so tension only ever **rises**: transverse motion cannot un-stretch a string.
Hardening, never softening. The nonlinearity is cubic in ``u`` (``I`` quadratic, times ``u_xx``), so
it vanishes as ``u -> 0``, recovering model #3 -- and ``EA = 0`` recovers it **bit-for-bit**.

**Scheme -- the split (why EA=0 is bit-identical).** Exact conservation of the nonlinear potential
requires the tension term at ``theta = 1/2`` (``mu_t. u = (u^{n+1} + u^{n-1})/2``), but model #3
averages its operator at ``theta ~ 0.28``. Moving the *whole* tension term to ``mu_t`` would make
``EA=0`` a theta=1/2 wave term against model #3's theta=0.28 -- **not** bit-identical, losing the
family's regression anchor. So split ``T_eff = T0 + dT`` and average only the nonlinear **excess**
at theta=1/2. The two pieces telescope independently, so this costs nothing:

    delta_tt u = L.(theta u^{n+1} + (1-2theta) u^n + theta u^{n-1})   <- model #3, verbatim
                 + (dT/rho) D2 (u^{n+1} + u^{n-1})/2                  <- NEW, theta=1/2 (mu_t)
                 - 2 sigma0 delta_t. u + 2 sigma1 delta_t.(delta_xx u)

    dT = (EA/4L) (I^{n+1} + I^{n-1})     [ = (EA/2L) * mean stretch ]

**The tension is the discrete gradient -- and it is a plain midpoint.** The energy-conserving choice
is ``T_eff = 2 (V(I+) - V(I-))/(I+ - I-)`` with ``V(I) = (T0/2) I + (EA/8L) I^2``. Because ``V`` is
**quadratic in I**, that collapses *exactly* to ``2 V'(Ibar)`` at the mean stretch -- no limit, no
``0/0``, no Taylor branch. (Contrast ``collision.py``'s ``[DG]``, whose 0/0 machinery exists for the
mallet's *power-law* potential. Do not import it here.)

**Solve.** ``dT`` is one scalar, so the step is a **scalar root-find** (the bow's shape, not model
#6's vector Picard). With ``beta = k^2 dT / (2 rho) >= 0``:

    (A0 - beta D2) u^{n+1} = rhs0 + beta D2 u^{n-1}

``A0`` is model #3's SPD pentadiagonal matrix and ``rhs0`` its RHS, both verbatim; ``-D2`` is SPD
and ``beta >= 0``, so ``A`` stays SPD and ``cholesky_banded`` carries over. ``A`` depends on ``dT``,
so the prefactorization is lost -- one banded refactor per residual evaluation (O(n), offline;
the human took this over the faster SAV/quadratisation route, which would conserve a *modified*
numerical energy rather than the physical one -- see the plan doc).

**Energy.**

    E^n = [ model #3's E^n ]  +  (EA/16L) ( (I^n)^2 + (I^{n-1})^2 )

The nonlinear term is the **two-time half-average** -- model #6's odd/even lesson (a single-level
``V_nl(I^n)`` is a 2-step invariant and oscillates spuriously). Here it is *derived*: by SBP
``h <D2(u+ + u-), u+ - u-> = -(I+ - I-)``, so the nonlinear power is exactly
``-delta_t.[(EA/8L) I^2]``, which ``delta_t+`` of the term above cancels identically. Lossless ->
conserved to machine precision; lossy -> monotone decreasing (both losses are model #3's,
dissipative by SBP, and neither enters E).

**Boundary: simply supported** -- and that is what buys the oracle. ``sin(m pi x / L)`` stays an
exact discrete eigenvector of ``D2`` *and* the biharmonic ``(delta_xx)^2``, while ``I`` depends on
the state only through the amplitude squared. So ``A(beta) s = (lambda0 + beta p^2) s`` for **any**
tension, and a single-mode state maps to a single-mode state: the scheme collapses onto a 1-DOF
Duffing oscillator with a closed-form elliptic frequency (:mod:`physsynth.analysis.duffing`) -- a
closed-form nonlinear oracle, which model #6 never had.

**But single-mode motion is *dynamically unstable* above a pump threshold** (measured, and NOT a bug
-- see the plan doc). The collapse above is exact **in exact arithmetic**; in floating point,
roundoff seeds the other modes, and as the tension pumps at twice the mode frequency those grow:
sit in parametric (Mathieu) resonance tongues and grow **exponentially** once ``dT/T0`` exceeds ~3.
Energy stays conserved to ~1e-13 throughout -- the motion redistributes across modes rather than
blowing up, which is exactly how you tell physics from a numerical artifact (it is also invariant
under grid/timestep refinement: same onset time, same unstable modes). Consequences:

- Purity is **structural per step** and holds at any amplitude -- test it over a short run.
- Purity **persists indefinitely only below threshold**; above it the mode audibly disintegrates
  into its neighbours. That modal energy exchange is a *feature* -- a linear string cannot do it.
- The Duffing frequency oracle is only meaningful while the motion is still single-mode.

This is the **planar** modal-exchange instability. It is *not* the out-of-plane whirling instability
of real strings, which needs two transverse polarizations -- this model has one.

Headless: NumPy + SciPy (banded Cholesky, brentq). No I/O, no plotting.
See ``docs/dev/tension-modulated-string-plan.md``.
"""

from __future__ import annotations

import warnings
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.optimize import brentq

from .operators import biharmonic_matrix, second_difference_matrix
from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]

TENSION_TOL_DEFAULT = 1e-13
"""Relative tolerance of the scalar tension solve (as a fraction of the bracket width)."""

MAX_BRACKET_EXPANSIONS = 40
"""Cap on doubling the tension bracket. Reaching it means the monotonicity argument broke."""


class StringCoefficients(NamedTuple):
    """A physically consistent coefficient set for a plain (unwound) cylindrical string."""

    rho: float
    """Linear density (kg/m) = rho_v * pi r^2."""
    kappa: float
    """Stiffness sqrt(E I_area / rho) (m^2/s), I_area = pi r^4 / 4."""
    EA: float
    """Axial stiffness (N) = E pi r^2."""
    c: float
    """Transverse wave speed sqrt(T/rho) (m/s)."""
    c_long: float
    """Longitudinal wave speed sqrt(E/rho_v) (m/s)."""
    EA_over_T: float
    """The governing nonlinearity ratio EA/T0 = (c_long/c)^2 -- radius-independent."""


def string_coefficients_from_material(
    *, E: float, radius: float, rho_v: float, T: float
) -> StringCoefficients:
    """Derive a **consistent** ``(rho, kappa, EA)`` from real material + geometry.

    A *modeling oracle*, not a constraint (cf. radiation's ``R_a`` helpers). The core deliberately
    takes **effective coefficients** ``(T, rho, kappa, EA)`` that are mutually unconstrained -- so a
    string can be given steel's bending stiffness and rubber's axial stiffness. That is a *feature*
    (HANDOFF section 12.J, hyperreal instruments: physics beyond real materials), and it is also the
    honest surface for **wound** strings (a steel core under a bronze overwind has no single ``E``,
    ``radius``, or ``rho_v``; the literature characterizes it by exactly these effective
    coefficients). This helper *offers* realism; it never imposes it.

    Parameters
    ----------
    E : Young's modulus (Pa). radius : string radius (m). rho_v : volumetric density (kg/m^3).
    T : rest tension (N) -- sets the transverse wave speed, hence the ratio below.

    Notes
    -----
    The nonlinearity's governing ratio is **radius-independent**::

        EA/T0 = E pi r^2 / (rho_v pi r^2 c^2) = E / (rho_v c^2) = (c_long / c)^2

    The radius cancels exactly: hardening is set by the ratio of **longitudinal to transverse wave
    speed**. Steel (``c_long ~ 5000 m/s``) at musical ``c ~ 200-400 m/s`` gives ``EA/T0 ~ 150-600``.
    That single number -- not the radius -- predicts whether a string will audibly glide.
    """
    if min(E, radius, rho_v, T) <= 0:
        raise ValueError("E, radius, rho_v, T must all be positive.")
    area = np.pi * radius**2
    second_moment = np.pi * radius**4 / 4.0
    rho = rho_v * area
    c = float(np.sqrt(T / rho))
    return StringCoefficients(
        rho=float(rho),
        kappa=float(np.sqrt(E * second_moment / rho)),
        EA=float(E * area),
        c=c,
        c_long=float(np.sqrt(E / rho_v)),
        EA_over_T=float(E * area / T),
    )


class TensionModulatedString:
    """A Kirchhoff-Carrier tension-modulated stiff string (model #9).

    Model #3's interface and parameters verbatim, plus ``EA``. ``EA = 0`` reduces **bit-for-bit** to
    :class:`~physsynth.core.string_damped.DampedStiffString`.

    Parameters
    ----------
    L, T, rho : float
        Length (m), **rest** tension (N), linear density (kg/m). ``c = sqrt(T/rho)``.
    fs : float
        Sample rate (Hz); ``k = 1/fs``. No CFL limit (unconditional for ``theta >= 1/4``), though
        the nonlinearity wants oversampling on its own account (HANDOFF section 8).
    N : int
        Spatial segments; ``N + 1`` nodes, ``h = L/N``, ends clamped, ``N - 1`` unknowns.
    kappa : float
        Stiffness ``sqrt(E I / rho)`` (m^2/s). ``0`` -> flexible.
    EA : float
        **Axial stiffness (N)** -- the nonlinearity. ``0`` -> model #3, bit-for-bit. The governing
        ratio is ``EA/T`` (see :func:`string_coefficients_from_material`); real strings sit at
        ``EA/T ~ 150-600``.
    sigma0, sigma1 : float
        Frequency-independent / -dependent loss (>= 0), model #3's.
    theta : float
        Time-averaging weight of the **linear** operator, ``(0, 1]``; ``>= 1/4`` unconditionally
        stable. The nonlinear term always uses ``theta = 1/2`` regardless (conservation needs it).
    boundary : {"supported"}
        Simply supported -- the boundary the closed-form Duffing oracle needs.
    tension_tol : float
        Relative tolerance of the scalar tension root-find. Energy drift is proportional to it
        (the machine-precision self-certification, absent a closed form for general motion).

    Raises
    ------
    ValueError
        On non-physical parameters (negative tension/density/stiffness/losses/``EA``, ``N < 2``,
        ``theta`` outside ``(0, 1]``, unsupported boundary).
    """

    def __init__(
        self,
        *,
        L: float,
        T: float,
        rho: float,
        fs: float,
        N: int,
        kappa: float = 0.0,
        EA: float = 0.0,
        sigma0: float = 0.0,
        sigma1: float = 0.0,
        theta: float = THETA_DEFAULT,
        boundary: Boundary = "supported",
        tension_tol: float = TENSION_TOL_DEFAULT,
    ) -> None:
        if min(L, T, rho, fs) <= 0:
            raise ValueError("L, T, rho, fs must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if kappa < 0:
            raise ValueError("kappa (stiffness) must be >= 0.")
        if EA < 0:
            raise ValueError("EA (axial stiffness) must be >= 0.")
        if sigma0 < 0:
            raise ValueError("sigma0 (frequency-independent loss) must be >= 0.")
        if sigma1 < 0:
            raise ValueError("sigma1 (frequency-dependent loss) must be >= 0.")
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if tension_tol <= 0:
            raise ValueError("tension_tol must be > 0.")
        if boundary != "supported":
            raise ValueError(f"boundary must be 'supported', got {boundary!r}.")

        self.L = float(L)
        self.T = float(T)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.kappa = float(kappa)
        self.EA = float(EA)
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.theta = float(theta)
        self.boundary: Boundary = boundary
        self.tension_tol = float(tension_tol)

        self.c = float(np.sqrt(T / rho))
        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.lam = self.c * self.k / self.h  # reported only; no CFL limit (unconditional)
        self.B = float((np.pi**2) * self.kappa**2 / (self.c**2 * self.L**2))
        self.EA_over_T = self.EA / self.T

        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)

        # Conservative linear operator L = c^2 delta_xx - kappa^2 delta_xxxx (model #3, verbatim).
        self._D2 = second_difference_matrix(self.N, self.h)
        self._L = (self.c**2) * self._D2
        if self.kappa != 0.0:
            self._L = self._L - (self.kappa**2) * biharmonic_matrix(self.N, self.h)
        self._L = self._L.tocsr()

        # A0 = (1 + sigma0 k) I - theta k^2 L - sigma1 k D2  (model #3's matrix, verbatim).
        # A(beta) = A0 - beta D2 gains the nonlinear tension; both SPD pentadiagonal.
        s0k = self.sigma0 * self.k
        n_int = self.N - 1
        ident = sparse.identity(n_int, format="csr")
        A0 = (1.0 + s0k) * ident - (self.theta * self.k**2) * self._L
        if self.sigma1 != 0.0:
            A0 = A0 - (self.sigma1 * self.k) * self._D2
        self._ab0 = self._banded(A0.tocsr())
        self._ab_D2 = self._banded(self._D2.tocsr())
        self._chol0 = cholesky_banded(self._ab0, lower=False)  # the dT = 0 factor (model #3's)

        self.u: NDArray[np.float64] = np.zeros(self.N + 1)
        self.u_prev: NDArray[np.float64] = np.zeros(self.N + 1)
        self.n: int = 0  # number of completed steps

        # Solver telemetry (the self-certifying gates -- see the plan doc).
        self.delta_tension: float = 0.0
        """Nonlinear tension excess ``dT`` (N) applied by the most recent :meth:`step`."""
        self.converged: bool = True
        """Whether the most recent tension solve converged."""
        self.bracket_expansions: int = 0
        """Cumulative bracket doublings -- **diagnostic, not an error**. ``I^{n+1}`` is not monotone
        in ``dT``, so the initial bracket guess misses on a small fraction of steps and doubling
        (which provably terminates) takes over. See :meth:`_solve_tension`."""
        self.n_not_converged: int = 0
        """Cumulative steps whose tension solve failed. Never render such a run as physics."""

    # -- initial conditions -------------------------------------------------------------

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacement (and optional velocity).

        Lossless consistent second-order start ``u^{-1} = u^0 - k v^0 + (k^2/2) u_tt^0``, with
        ``u_tt^0`` including the **nonlinear** tension at ``t = 0`` (``dT_0 = (EA/2L) I(u^0)``) so a
        single eigenmode starts as a clean discrete Duffing cosine. ``EA = 0`` skips that term ->
        model #3's start exactly. Ends are clamped. Under damping the start is only *consistent*
        (not exact), as in model #3.
        """
        u0 = np.asarray(u0, dtype=float).copy()
        if u0.shape != (self.N + 1,):
            raise ValueError(f"u0 must have shape {(self.N + 1,)}, got {u0.shape}.")
        v0_arr = np.broadcast_to(np.asarray(v0, dtype=float), (self.N + 1,)).copy()

        u0[0] = u0[-1] = 0.0
        accel = self._apply_L(u0)  # full-grid L u0 (0 at the clamped nodes)
        if self.EA != 0.0:
            dT0 = (self.EA / (2.0 * self.L)) * self._stretch(u0)
            accel = accel + (dT0 / self.rho) * self._apply_D2(u0)
        u_prev = u0 - self.k * v0_arr + 0.5 * self.k**2 * accel
        u_prev[0] = u_prev[-1] = 0.0

        self.u = u0
        self.u_prev = u_prev
        self.n = 0
        self.delta_tension = 0.0
        self.converged = True

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep (rolls the history).

        ``EA = 0``: one banded back-substitution against the prefactored model-#3 matrix -- the
        identical code path, hence bit-identical results. Otherwise: a scalar ``brentq`` root-find
        for the tension excess ``dT`` (see :meth:`_solve_tension` for why a bracket always exists),
        each residual costing one banded refactor + solve.
        """
        un = self.u[1:-1]
        up = self.u_prev[1:-1]
        Lu = self._L @ un
        Lu_prev = self._L @ up
        # Model #3's RHS, expression-for-expression (float addition is not associative -- this
        # ordering is what makes EA=0 bit-identical, not merely equal to tolerance).
        rhs0 = (
            2.0 * un
            + (1.0 - 2.0 * self.theta) * self.k**2 * Lu
            - up
            + self.theta * self.k**2 * Lu_prev
            + self.sigma0 * self.k * up
        )
        if self.sigma1 != 0.0:
            rhs0 = rhs0 - (self.sigma1 * self.k) * (self._D2 @ up)

        if self.EA == 0.0:
            u_next_int = cho_solve_banded((self._chol0, False), rhs0)
            self.delta_tension = 0.0
            self.converged = True
        else:
            u_next_int = self._solve_tension(rhs0, up)

        u_next = np.zeros(self.N + 1)
        u_next[1:-1] = u_next_int
        self.u_prev = self.u
        self.u = u_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current displacement field ``u^n`` (a copy, safe to mutate/store for plotting)."""
        return self.u.copy()

    @property
    def stretch(self) -> float:
        """Current stretch ``I^n = h ||delta_x+ u^n||^2`` (m) -- what modulates the tension."""
        return self._stretch(self.u)

    @property
    def tension(self) -> float:
        """Current total tension ``T0 + (EA/2L) I^n`` (N). Always ``>= T0`` (hardening only)."""
        return self.T + (self.EA / (2.0 * self.L)) * self.stretch

    def energy(self) -> float:
        """Discrete energy ``E^n`` (Joules) = model #3's energy + the nonlinear stretch term.

        The nonlinear term is the **two-time half-average** ``(EA/16L)((I^n)^2 + (I^{n-1})^2)``,
        which is what telescopes against the scheme's nonlinear power (see the module docstring).
        Lossless -> conserved to machine precision; lossy -> monotone decreasing.
        """
        return self._linear_energy() + self.nonlinear_energy()

    def nonlinear_energy(self) -> float:
        """The stretch (membrane) part of ``E^n`` alone (J) -- ``0`` iff ``EA = 0``.

        Report its **fraction** of :meth:`energy` in every energy test: a nonlinearity bug *hides*
        at small amplitude, where the test merely re-runs the linear scheme (model #6's lesson).
        """
        if self.EA == 0.0:
            return 0.0
        i_n = self._stretch(self.u)
        i_p = self._stretch(self.u_prev)
        return float((self.EA / (16.0 * self.L)) * (i_n**2 + i_p**2))

    def displacement_at(self, index: int) -> float:
        """Displacement at grid node ``index`` -- a pickup for spectral analysis."""
        return float(self.u[index])

    def apply_Ainv(self, rhs_int: NDArray[np.float64]) -> NDArray[np.float64]:
        """Not available on this model -- **the update matrix is time-varying**.

        Model #3 exposes ``A^{-1}``'s action so a coupled element (e.g.
        :class:`~physsynth.core.bow.BowedString`) can precompute a *constant* driving-point
        admittance ``a = A^{-1} e_i``. Here ``A = A0 - beta D2`` moves with the tension every step,
        so any such precompute would be silently wrong. Coupling an exciter to a tension-modulated
        string needs a joint solve (the tension and the contact/friction force resolved together) --
        deliberately out of scope; see the plan doc.
        """
        raise NotImplementedError(
            "TensionModulatedString has a time-varying update matrix (A depends on the tension), "
            "so a constant driving-point admittance A^-1 e_i does not exist. Coupling an exciter "
            "here requires a joint solve -- see docs/dev/tension-modulated-string-plan.md."
        )

    # -- internals ----------------------------------------------------------------------

    def _solve_tension(self, rhs0: NDArray[np.float64], up: NDArray[np.float64]) -> NDArray:
        """Scalar root-find for the nonlinear tension excess ``dT``; returns ``u^{n+1}`` interior.

        Solves ``resid(dT) = dT - (EA/4L)(I^{n+1}(dT) + I^{n-1}) = 0``.

        **A bracket always exists and doubling always finds it.** ``resid(0) <= 0``, because
        stretches are non-negative. And ``resid -> +infinity`` as ``dT -> infinity``: the update
        ``(A0 - beta D2) u+ = rhs0 + beta D2 u-`` tends to ``u+ -> -u-`` as ``beta -> infinity``, so
        ``I^{n+1} -> I^{n-1}`` -- **bounded** -- and the linear ``dT`` outruns it. So the doubling
        loop below provably terminates; it is the normal mechanism, not an anomaly.

        **``I^{n+1}`` is NOT monotone in ``dT``** (it dips, then climbs back to ``I^{n-1}``). So the
        tempting bracket ``[0, (EA/4L)(I^{n+1}(0) + I^{n-1})]`` is *not* guaranteed: it fails
        whenever ``I^{n+1}(0) < I^{n-1}``, i.e. while the string is winding back up (~2 % of steps,
        measured). Seeding with ``max(I^{n+1}(0), I^{n-1})`` fixed every observed case; the doubling
        is what makes it safe.

        **Uniqueness is deliberately not gated** (the :class:`~physsynth.core.bow.BowedString`
        precedent): *any* root conserves energy exactly, since the telescoping needs only a
        self-consistent ``dT = (EA/4L)(I+ + I-)``. Non-uniqueness would be a branch-selection
        question, not a correctness one. (Empirically ``resid`` is strictly increasing -- a single
        sign change over a dense sweep -- at any usable timestep.)

        The unknown is normalized to ``s = dT/dT_hi in [0, 1]`` so ``tension_tol`` is a unit-free
        relative bar.
        """
        i_prev = self._stretch(self.u_prev)
        coeff = self.EA / (4.0 * self.L)
        d2_up = self._D2 @ up

        cache: dict[float, NDArray[np.float64]] = {}

        def u_next_for(dT: float) -> NDArray[np.float64]:
            if dT not in cache:
                beta = self.k**2 * dT / (2.0 * self.rho)
                chol = cholesky_banded(self._ab0 - beta * self._ab_D2, lower=False)
                cache[dT] = cho_solve_banded((chol, False), rhs0 + beta * d2_up)
            return cache[dT]

        def resid(dT: float) -> float:
            return dT - coeff * (self._stretch_int(u_next_for(dT)) + i_prev)

        i_free = self._stretch_int(u_next_for(0.0))
        dT_hi = coeff * (max(i_free, i_prev) + i_prev)
        if dT_hi <= 0.0:  # string exactly at rest -> no stretch, no modulation
            self.delta_tension = 0.0
            self.converged = True
            return u_next_for(0.0)

        expansions = 0
        while resid(dT_hi) < 0.0 and expansions < MAX_BRACKET_EXPANSIONS:
            dT_hi *= 2.0
            expansions += 1
        self.bracket_expansions += expansions
        if resid(dT_hi) < 0.0:
            self.converged = False
            self.n_not_converged += 1
            warnings.warn(
                f"Tension solve failed to bracket a root at step {self.n} after "
                f"{MAX_BRACKET_EXPANSIONS} doublings. Do not treat this run as physics.",
                RuntimeWarning,
                stacklevel=3,
            )
            self.delta_tension = dT_hi
            return u_next_for(dT_hi)

        # Normalized unknown s = dT/dT_hi in [0, 1]: tension_tol is then relative, unit-free.
        root_s = brentq(
            lambda s: resid(s * dT_hi), 0.0, 1.0, xtol=self.tension_tol, rtol=8.9e-16
        )
        self.delta_tension = root_s * dT_hi
        self.converged = True
        return u_next_for(self.delta_tension)

    def _linear_energy(self) -> float:
        """Model #3's energy form, unchanged: kinetic + theta-weighted linear potential."""
        un = self.u[1:-1]
        up = self.u_prev[1:-1]
        dt_u = (un - up) / self.k  # delta_t- u^n on the interior (boundary velocity is 0)
        kinetic = 0.5 * self.h * float(np.dot(dt_u, dt_u))

        p_nn = self._P(un, un)
        p_pp = self._P(up, up)
        p_np = self._P(un, up)
        potential = 0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np
        return self.rho * (kinetic + potential)

    def _stretch(self, u_full: NDArray[np.float64]) -> float:
        """Stretch ``I = h ||delta_x+ u||^2`` (m) on the **full** grid (ends included).

        Note the ``h``: ``I = h sum ((u_{j+1} - u_j)/h)^2 = sum (du)^2 / h``. Dropping it is the
        model-#8 "force *density*" trap -- it looks exactly like a mis-scaled ``EA`` and passes
        every qualitative test.
        """
        du = np.diff(u_full)
        return float(np.dot(du, du) / self.h)

    def _stretch_int(self, u_int: NDArray[np.float64]) -> float:
        """Stretch of an interior-only vector (the clamped ends contribute their own slopes)."""
        du = np.diff(u_int)
        return float((np.dot(du, du) + u_int[0] ** 2 + u_int[-1] ** 2) / self.h)

    def _P(self, f: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """Potential bilinear form ``P(f,g) = <-L f, g> = h * (-L f) . g`` (interior vectors)."""
        return -self.h * float(np.dot(self._L @ f, g))

    def _apply_L(self, u_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """``L u`` returned on the full grid (zeros at the clamped boundary nodes)."""
        out = np.zeros_like(u_full)
        out[1:-1] = self._L @ u_full[1:-1]
        return out

    def _apply_D2(self, u_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """``D2 u`` returned on the full grid (zeros at the clamped boundary nodes)."""
        out = np.zeros_like(u_full)
        out[1:-1] = self._D2 @ u_full[1:-1]
        return out

    @staticmethod
    def _banded(M: sparse.csr_matrix) -> NDArray[np.float64]:
        """Upper-banded storage (2 superdiagonals) of a symmetric pentadiagonal matrix."""
        n = M.shape[0]
        ab = np.zeros((3, n))
        ab[2, :] = M.diagonal(0)
        ab[1, 1:] = M.diagonal(1)
        ab[0, 2:] = M.diagonal(2)
        return ab
