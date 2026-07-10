"""Single-reed mouthpiece — the wind leg's continuous **nonlinear exciter** (wind batch 3), the
acoustic dual of the bowed string (:mod:`physsynth.core.bow`).

This closes the ``exciter -> resonator`` loop for wind instruments: a **dynamic reed** valve driven
by a steady mouth pressure ``p_m`` self-oscillates against the air column of a :class:`Bore`,
turning a constant breath into a sustained tone — the clarinet. It is the mirror of the bow (a
constant bow speed self-oscillates the string into Helmholtz motion), and it reuses every lesson the
bow and the von-Kármán plate taught: an *implicit*, centered nonlinearity solved to a **scalar**
each step, an energy **balance** (not conservation — the mouth is active), and machine-precision
telescoping as the money test.

**The physical model (Bilbao §9; Chatziioannou & van Walstijn; Dalmont/Kergomard).** The reed is a
damped harmonic oscillator whose tip displacement ``y`` opens/closes a channel of rest height
``H0``; the instantaneous opening is ``H = H0 + y``. It is **inward-striking**: a positive pressure
drop across the reed ``dp = p_m - p0`` (mouth minus mouthpiece) pushes the reed *closed*, so

    mu (y'' + g y' + wr^2 y) = -dp ,          H = H0 + y ,          (reed, per unit area)

with ``mu`` the reed's areal mass, ``wr`` its resonance and ``g = wr / Q`` its damping. Statically
the reed shuts (``H -> 0``, ``y = -H0``) at the **closing pressure** ``p_closing = mu wr^2 H0`` —
the natural pressure scale (the control is ``gamma = p_m / p_closing``). Air jets through the
open channel by Bernoulli, and the moving reed sweeps its own volume, so the total flow **into the
bore** at the mouthpiece node is

    U = U_B - Sr y' ,     U_B = w H^+ sign(dp) sqrt(2 |dp| / rho) ,   H^+ = max(H, 0) .

``U_B`` is the nonlinear jet (a passive resistor: ``dp U_B >= 0`` always); ``-Sr y'`` is the reed's
volume sweep (``Sr`` the effective reed area — the **same** area that the pressure acts on, which is
exactly what makes the coupling reactive/lossless below). Blowing hard enough that the descending
branch of ``U_B(dp)`` overcomes the losses makes the whole thing oscillate — the clarinet.

**Why the reed-sweep flow is not optional.** Include ``-Sr y'`` and the reed-force work and the bore
sweep-flow work are the *same* reactive term with opposite sign — they cancel exactly, leaving a
clean, sign-definite energy budget. Drop it and that term floats free and the balance is lost. The
sweep is what makes the coupling energy-consistent, so ``Sr`` must be the one area shared by the
pressure force and the sweep.

**Discrete scheme — the bow pattern, made two-field.** The reed is a centered leapfrog (``y''`` and
the ``g y'`` damping centered, the spring ``wr^2 y`` explicit at ``n`` — the string's cross-time
potential), and the bore is its staggered p/U leapfrog. They couple only through the mouthpiece
half-cell node 0. The nonlinearity is evaluated at the **centered** pressure drop
``dp_bar = p_m - (p0^{n+1} + p0^n)/2`` and the **centered** reed velocity
``y' = (y^{n+1} - y^{n-1}) / 2k`` — the *same* two quantities appear in the reed force and the bore
injection, so the reactive coupling telescopes to machine precision (the von-Kármán/bow centering
lesson; an off-center pressure leaks O(k) energy). The reed's linear response is affine in
``dp_bar``, so eliminating ``y^{n+1}`` collapses the whole implicit step to **one scalar equation**
for ``dp_bar`` (the bow's rank-1 move):

    D dp_bar = C0const + p_pref0 U_B(dp_bar) ,

with ``U_B`` the only nonlinear term. Following the advisor's simplification, the **opening**
``H^+`` is frozen explicit at ``n`` (it only scales the passive conductance, never the reactive
coupling), so the scalar residual carries a single ``sqrt`` cusp and no clamp kink — it is
strictly **monotone** in ``dp_bar`` (unique root), solved by a continuation-seeded Newton with a
guaranteed bracketing fallback (the ``sqrt`` gives infinite slope at ``dp = 0``, so the fallback
fires more than the bow's did).

**The node-0 half-cell injection.** The flow enters node 0's continuity ``C0 dp0/dt = ... + U``, so
the pressure gain per step is ``(k / C0) U`` with ``C0`` the **half-cell** capacitance (``h/2`` node
weight — the trapezoidal wall weight the bore already carries). Getting that ``C0`` right is the
node-0 gotcha: the wrong capacitance rescales the injected flow and the energy balance drifts. The
reed rides on the bore's ``"closed"`` end (a live half-cell DOF, like a rigid wall), and the
injection is applied **inside** ``Bore.step`` between the pressure and momentum sub-steps (the
``source`` hook) so the velocity ``U^{n+3/2}`` sees the corrected node pressure — the same ordering
the radiating bell obeys.

**Validation is an energy *balance* (the exciter money test).** The reed *stores* energy (it is a
mass-spring, unlike the bow's memoryless friction), so the conserved quantity is
``E = E_bore + E_reed``, and the exact discrete identity is

    E^n - E^0 = mouth_work - jet_loss - reed_damp_work   (+ any bore loss/radiation),

every dissipation channel sign-definite: ``jet_loss = sum k dp_bar U_B >= 0`` (Bernoulli),
``reed_damp_work = sum k mu Sr g y'^2 >= 0`` (reed damping), ``mouth_work = sum k p_m U`` (the
breath, the *active* input). Unlike the bow (memoryless, so its balance is residual-*independent* —
force applied and booked from one number), the reed's two-field coupling ties the bore node and the
booked flow together only *at* the root: the per-step error is ``k p_bar R / p_pref0``, **linear in
the scalar residual** ``R``. So the balance both **requires and verifies** a converged solve each
step; it is machine precision because ``newton_tol ~ 1e-10`` keeps ``R`` tiny (loosen it and the
balance degrades in proportion). Lossless bore -> the identity holds to ~1e-14. **But balance passes
on a dead (wrong-sign) reed that rings down**, so the independent oracle is the *signature*:
self-sustained oscillation above a blowing threshold, locked near the bore's ``c/4L`` with **odd**
harmonics (the clarinet), decaying below threshold.

Headless: NumPy + SciPy (delegates the air column to :class:`Bore`, the scalar solve to ``brentq``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .bore import Bore

__all__ = ["ReedBore", "bernoulli_flow"]


def bernoulli_flow(dp: float, opening: float, width: float, rho: float) -> float:
    """Quasi-static Bernoulli volume flow ``U_B`` (m^3/s) through the reed channel.

    ``U_B = width * opening * sign(dp) * sqrt(2 |dp| / rho)`` — the jet velocity ``sqrt(2|dp|/rho)``
    (Bernoulli, incompressible) times the channel cross-section ``width * opening``, signed by the
    pressure drop ``dp = p_mouth - p_mouthpiece``. ``opening`` is the *clamped* height ``H^+ =
    max(H0 + y, 0)`` (a shut reed passes no air). The characteristic nonlinearity of the reed valve;
    a passive resistor because ``dp * U_B = width * opening * |dp| sqrt(2|dp|/rho) >= 0`` always.
    """
    if opening <= 0.0:
        return 0.0
    return width * opening * math.copysign(math.sqrt(2.0 * abs(dp) / rho), dp)


class ReedBore:
    """A :class:`Bore` blown through a dynamic single reed (nonlinear self-oscillating exciter).

    Implements the engine's ``Resonator`` protocol (``step``/``energy``/``state``/``k``/
    ``displacement_at``), delegating the air column to the wrapped bore. The natural pickup is the
    **mouthpiece pressure** :meth:`mouthpiece_pressure` (``p0``) — the mouth-side of the reed, whose
    waveform goes square-ish as the note speaks; :meth:`displacement_at` reads any interior node,
    and :meth:`pressure` is the bell's far-field read-out (if the bore has a radiating end).

    Parameters
    ----------
    bore : Bore
        The air column. Its **left** end must be ``"closed"`` (the mouthpiece is a live half-cell
        DOF the reed drives); the far end is typically ``"open"`` (ideal) or ``"radiating"`` (a
        sounding bell). A little loss (a radiating bell or ``sigma > 0``) lets the note settle into
        a steady regime instead of growing without bound.
    p_mouth : float
        Steady mouth (blowing) pressure ``p_m`` (Pa) — the control input. Oscillation needs it above
        a threshold fraction of the reed's closing pressure (``gamma = p_m / p_closing``, the
        clarinet speaks around ``gamma ~ 1/3``); mutate the attribute between steps for an attack.
    f_reed : float
        Reed resonance frequency (Hz), ``wr = 2 pi f_reed``. A clarinet reed sits ~2-3 kHz, well
        above the bore fundamental (the bore, not the reed, sets the pitch — an *inward-striking*
        reed). Must satisfy ``wr k < 2`` (trivially true at the bore's oversampled rate).
    q_reed : float
        Reed quality factor; the damping rate is ``g = wr / q_reed``. Clarinet reeds are heavily
        lip-damped (``q ~ 3-5``).
    mu : float
        Reed areal mass ``mu`` (kg/m^2). With ``f_reed`` and ``H0`` it fixes the closing pressure
        ``p_closing = mu wr^2 H0`` — the pressure scale of the instrument.
    Sr : float
        Effective reed area (m^2): the area the mouthpiece pressure acts on **and** the area the
        reed sweeps as it moves. One shared area — the energy identity requires it (see above).
    width : float
        Effective channel width ``w`` (m) for the Bernoulli jet ``U_B = w H^+ ...``.
    H0 : float
        Reed rest opening (m): the channel height at ``y = 0`` (no blowing). The reed beats shut
        when ``H0 + y <= 0`` (a hard flow clamp; the true contact *force* is deferred, HANDOFF §12).
    newton_tol : float
        Convergence tolerance on the scalar pressure-drop residual (Pa). Default ``1e-10``. The
        energy balance is exact only up to this residual (per-step error ``k p_bar R / p_pref0``),
        so keep it tight — loosening it degrades the balance in proportion.
    newton_maxiter : int
        Max safeguarded-Newton iterations per step before the bracketed fallback. Default ``60``.

    Raises
    ------
    ValueError
        Non-physical parameters, a non-``"closed"`` left end, or a reed CFL ``wr k >= 2``.
    """

    def __init__(
        self,
        *,
        bore: Bore,
        p_mouth: float,
        f_reed: float = 2500.0,
        q_reed: float = 4.0,
        mu: float = 0.03,
        Sr: float = 1.5e-4,
        width: float = 1.5e-2,
        H0: float = 4.0e-4,
        newton_tol: float = 1e-10,
        newton_maxiter: int = 60,
    ) -> None:
        if min(f_reed, q_reed, mu, Sr, width, H0) <= 0.0:
            raise ValueError("f_reed, q_reed, mu, Sr, width, H0 must all be positive.")
        if newton_maxiter < 1:
            raise ValueError("newton_maxiter must be >= 1.")
        if bore._bc_left != "closed":
            raise ValueError(
                "the reed rides on the bore's LEFT end, which must be 'closed' (a live half-cell "
                f"mouthpiece DOF), got {bore._bc_left!r}. Use boundary=('closed', <far end>)."
            )

        self.bore = bore
        self.k = bore.k
        self.p_mouth = float(p_mouth)
        self.f_reed = float(f_reed)
        self.q_reed = float(q_reed)
        self.mu = float(mu)
        self.Sr = float(Sr)
        self.width = float(width)
        self.H0 = float(H0)
        self.rho = bore.rho0
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)

        self.wr = 2.0 * math.pi * self.f_reed
        self.g = self.wr / self.q_reed
        if self.wr * self.k >= 2.0:
            raise ValueError(
                f"reed CFL violated: wr*k = {self.wr * self.k:.3f} >= 2 (reed too stiff for the "
                "timestep). Raise the sample rate (finer bore grid / larger lam) or lower f_reed."
            )
        self.Mr = self.mu * self.Sr                       # lumped reed mass (kg)
        self.p_closing = self.mu * self.wr * self.wr * self.H0  # static closing pressure (Pa)

        # Node-0 half-cell capacitance C0 = (h/2) S / (rho c^2) and the injection prefactor k/C0.
        # Built from the bore's PUBLIC geometry so the half-cell weight (the node-0 gotcha) is
        # explicit and we never reach into the bore's private update arrays.
        c0_cap = (0.5 * bore.h) * bore.S_node[0] / (bore.rho0 * bore.c0 * bore.c0)
        self._p_pref0 = self.k / c0_cap        # p0 += p_pref0 * U_inject  (== bore's _p_pref[0])

        # Reed leapfrog coefficients. y^{n+1} = y_hist + c_reed * dp_bar, from
        #   (1 + gk/2) y^{n+1} = (2 - wr^2 k^2) y^n + (gk/2 - 1) y^{n-1} - (Sr k^2/Mr)(p_m - p_bar)
        # (spring explicit at n; damping/inertia centered). Sr/Mr = 1/mu.
        gk = self.g * self.k
        self._den = 1.0 + 0.5 * gk                        # coeff of y^{n+1}
        self._cy_n = (2.0 - (self.wr * self.k) ** 2) / self._den       # of y^n
        self._cy_prev = (0.5 * gk - 1.0) / self._den                   # of y^{n-1}
        self._c_reed = (self.k * self.k / self.mu) / self._den         # of dp_bar in y^{n+1}

        # Reactive stiffening D of the scalar equation (>= 2): the reed inertia+sweep response fed
        # back through the node. D dp_bar = C_const + p_pref0 U_B(dp_bar). Constant across the run.
        self._D = 2.0 + self._p_pref0 * self.Sr * self._c_reed / (2.0 * self.k)

        # Reed state (starts at rest: y = y' = 0, channel fully open at H0).
        self.y = 0.0
        self.y_prev = 0.0

        # Per-step observables + cumulative energy channels (the balance identity).
        self.dp = 0.0            # centered pressure drop dp_bar last step
        self.reed_velocity = 0.0  # centered y' last step
        self.flow = 0.0          # total volume flow into the bore U = U_B - Sr y'
        self.jet_flow = 0.0      # Bernoulli U_B alone
        self.mouth_work = 0.0    # sum k p_m U      (active breath input)
        self.jet_loss = 0.0      # sum k dp_bar U_B (Bernoulli dissipation, >= 0)
        self.reed_damp_work = 0.0  # sum k Mr g y'^2 (reed damping, >= 0)
        self.fallbacks = 0
        self.n = 0

    # -- flow / opening ------------------------------------------------------------------

    def reed_opening(self) -> float:
        """Current clamped channel opening ``H^+ = max(H0 + y, 0)`` (m). ``0`` when the reed beats
        shut. This is the value frozen explicit in the Bernoulli conductance each step."""
        return max(self.H0 + self.y, 0.0)

    # -- the scalar coupling solve -------------------------------------------------------

    def _residual(self, dp: float, opening: float, c_const: float) -> float:
        """``R(dp) = D (p_m - dp) - c_const - p_pref0 U_B(dp)`` (Pa). Strictly decreasing in ``dp``
        (unique root): ``-D dp`` falls and the passive jet ``-p_pref0 U_B`` falls too."""
        u_b = bernoulli_flow(dp, opening, self.width, self.rho)
        return self._D * (self.p_mouth - dp) - c_const - self._p_pref0 * u_b

    def _solve_dp(self, opening: float, c_const: float) -> tuple[float, bool]:
        """Solve ``R(dp) = 0`` for the centered pressure drop. Continuation-seeded Newton (from last
        step's ``dp``) with a guaranteed bracketing fallback. ``U_B ~ sign(dp) sqrt|dp|`` has
        infinite slope at ``dp = 0``, so Newton can stall near the origin — accept a Newton step
        only while it strictly shrinks ``|R|``, else hand to :meth:`_bracketed_root`. Both give a
        machine-precision root, so the applied flow (and the energy balance) is exact either way.
        Returns ``(dp, used_fallback)``."""
        sq = math.sqrt(2.0 / self.rho)
        dp = self.dp  # continuation seed
        r = self._residual(dp, opening, c_const)
        for _ in range(self.newton_maxiter):
            if abs(r) <= self.newton_tol:
                return dp, False
            # R'(dp) = -D - p_pref0 * w * opening * sqrt(2/rho) / (2 sqrt|dp|)
            if opening > 0.0 and abs(dp) > 1e-30:
                slope = self.width * opening * sq / (2.0 * math.sqrt(abs(dp)))
                rp = -self._D - self._p_pref0 * slope
            else:
                rp = -self._D
            dp_new = dp - r / rp
            r_new = self._residual(dp_new, opening, c_const)
            if not (abs(r_new) < abs(r)):
                break  # stalled (the sqrt cusp) -> robust bracket
            dp, r = dp_new, r_new
        if abs(r) <= self.newton_tol:
            return dp, False
        return self._bracketed_root(opening, c_const), True

    def _bracketed_root(self, opening: float, c_const: float) -> float:
        """Bracket the (unique, monotone) root of ``R(dp)`` by expanding a window around ``p_mouth``
        until ``R`` changes sign, then ``brentq``. ``R(-inf) = +inf`` and ``R(+inf) = -inf`` (the
        linear ``-D dp`` dominates the ``sqrt`` jet), so a sign-changing bracket always exists."""
        span = max(abs(self.p_mouth), self.p_closing, 1.0)
        lo, hi = self.p_mouth - span, self.p_mouth + span
        r_lo = self._residual(lo, opening, c_const)
        r_hi = self._residual(hi, opening, c_const)
        for _ in range(60):
            if r_lo > 0.0 >= r_hi:
                break
            span *= 2.0
            lo, hi = self.p_mouth - span, self.p_mouth + span
            r_lo = self._residual(lo, opening, c_const)
            r_hi = self._residual(hi, opening, c_const)
        else:
            raise RuntimeError(
                f"reed pressure-drop residual failed to bracket at step {self.n} "
                "(should be impossible for the monotone residual)."
            )
        return float(
            brentq(self._residual, lo, hi, args=(opening, c_const), xtol=1e-13, rtol=8.9e-16)
        )

    # -- time stepping ------------------------------------------------------------------

    def _inject(self, p_next: NDArray[np.float64]) -> None:
        """The ``Bore.step`` source hook: correct the mouthpiece node ``p_next[0]`` for the reed
        flow and stash the post-solve quantities for the energy bookkeeping in :meth:`step`.

        ``p_next[0]`` arrives as the bore's force-free (rigid-wall) half-cell step ``p_rigid``. With
        the reed's affine response ``y^{n+1} = y_hist + c_reed p_bar`` eliminated, the node balance
        is the scalar ``D p_bar = C_const + p_pref0 U_B(dp_bar)`` in ``p_bar = (p0^{n+1}+p0^n)/2``.
        Solve it, write ``p0^{n+1} = 2 p_bar - p0^n``, and record ``dp_bar``, ``y^{n+1}`` and the
        flow for :meth:`step` to commit and book (all from post-solve values)."""
        p_old = float(self.bore.p[0])   # p0^n (bore not yet committed)
        p_rigid = float(p_next[0])      # force-free rigid half-cell step
        opening = self.reed_opening()   # H^+ frozen explicit at n (advisor simplification)

        # y_hist = the dp-independent part of y^{n+1}; yd_hist = its part of y' = (y+ - y-)/2k
        y_hist = self._cy_n * self.y + self._cy_prev * self.y_prev - self._c_reed * self.p_mouth
        yd_hist = (y_hist - self.y_prev) / (2.0 * self.k)
        # C_const collects everything known: node balance minus the reed-sweep history term.
        c_const = p_rigid + p_old - self._p_pref0 * self.Sr * yd_hist

        dp, used_fallback = self._solve_dp(opening, c_const)
        p_bar = self.p_mouth - dp
        p_next[0] = 2.0 * p_bar - p_old

        # Post-solve reed velocity, flows (exact -> the energy balance is machine precision).
        y_new = y_hist + self._c_reed * p_bar
        y_dot = (y_new - self.y_prev) / (2.0 * self.k)
        u_b = bernoulli_flow(dp, opening, self.width, self.rho)
        u_total = u_b - self.Sr * y_dot

        # stash for commit/bookkeeping in step()
        self._y_new = y_new
        self.dp = dp
        self.reed_velocity = y_dot
        self.jet_flow = u_b
        self.flow = u_total
        self.fallbacks += int(used_fallback)

    def step(self) -> None:
        """Advance one step: the bore's leapfrog with the reed injecting at node 0 (implicit scalar
        solve in the ``source`` hook), then commit the reed state and book the energy channels."""
        self.bore.step(source=self._inject)

        # Commit the reed leapfrog: (y_prev, y) <- (y^n, y^{n+1}).
        self.y_prev = self.y
        self.y = self._y_new

        # Energy channels from post-solve values; the balance E - E0 = mouth - jet - reed_damp is
        # exact up to k*p_bar*R/p_pref0 (linear in the scalar residual R, ~1e-10 by newton_tol).
        k = self.k
        self.mouth_work += k * self.p_mouth * self.flow            # active breath input
        self.jet_loss += k * self.dp * self.jet_flow               # Bernoulli dissipation >= 0
        self.reed_damp_work += k * self.Mr * self.g * self.reed_velocity ** 2  # reed damping >= 0
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def reed_energy(self) -> float:
        """Stored reed mechanical energy ``E_reed`` (Joules): kinetic ``1/2 Mr (delta_t- y)^2`` plus
        the **cross-time** potential ``1/2 Mr wr^2 y^n y^{n-1}`` (matching the explicit spring — the
        string's potential trick). Positive-definite while ``wr k < 2``."""
        y_dot_back = (self.y - self.y_prev) / self.k
        return 0.5 * self.Mr * y_dot_back * y_dot_back + 0.5 * self.Mr * self.wr * self.wr * (
            self.y * self.y_prev
        )

    def energy(self) -> float:
        """Total stored energy ``E_bore + E_reed`` (Joules) — the quantity the balance identity
        tracks. **Not** conserved (the mouth is active): assert
        ``E^n - E^0 == mouth_work - jet_loss - reed_damp_work`` (exact for a lossless bore), not
        conservation. With a lossy/radiating bore add its shed energy to the right-hand side."""
        return self.bore.energy() + self.reed_energy()

    @property
    def state(self) -> NDArray[np.float64]:
        """The bore pressure field (the vibrating air column, for animation snapshots)."""
        return self.bore.state

    def displacement_at(self, index: int) -> float:
        """Bore pressure at node ``index`` — an interior microphone for spectral analysis."""
        return self.bore.displacement_at(index)

    def mouthpiece_pressure(self) -> float:
        """Pressure at the mouthpiece node ``p0`` (Pa) — the mouth-side reed pickup, the natural
        'playing' signal (goes square-ish once the note speaks)."""
        return float(self.bore.p[0])

    def pressure(self) -> float:
        """Far-field read-out of the bore's radiating bell (``dU_out/dt``); ``0`` with no radiating
        end. Feed to :class:`~physsynth.core.radiation.AirRadiation` for a listener's pressure."""
        return self.bore.pressure()

    @property
    def gamma(self) -> float:
        """Dimensionless blowing pressure ``gamma = p_mouth / p_closing`` — the clarinet control
        parameter. The reed beats shut statically at ``gamma = 1``; the note speaks around the
        small-oscillation threshold ``gamma ~ 1/3`` (Dalmont/Kergomard)."""
        return self.p_mouth / self.p_closing
