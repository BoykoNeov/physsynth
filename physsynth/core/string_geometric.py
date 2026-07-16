"""Geometrically-exact string — two polarizations + longitudinal motion (model #10).

The string family's **exact** geometry, and the model that discharges the two claims model #9
(:class:`~physsynth.core.string_nonlinear.TensionModulatedString`) explicitly refuses to make:
**piano phantom partials** and **out-of-plane whirling**. Both refusals have the same root cause --
model #9 has *one field*. Kirchhoff-Carrier collapses the tension to a **spatial scalar**
``T(t) = T0 + (EA/2L) I``, which is precisely what makes it (a) blind to longitudinal dynamics and
(b) planar. This model keeps the tension a **local field** and carries all three components::

    r(x, t) = ( x + v(x,t),  u(x,t),  w(x,t) )     v = longitudinal, u/w = the two polarizations

with ``x`` the coordinate in the **rest (already tensioned)** configuration.

**The potential is exact, not a truncation.** With ``q = (u_x, w_x, v_x)`` the local strain and

    Lambda(q) = |dr/dx| = sqrt( (1 + v_x)^2 + u_x^2 + w_x^2 )     the local STRETCH RATIO

the stored energy density is (see the plan doc for the pre-stress derivation)::

    V(q) = (T0/2)(u_x^2 + w_x^2) + (EA/2) v_x^2
           + a * [ (u_x^2 + w_x^2)/2 + 1 + v_x - Lambda ]              a = EA - T0
           \\____ LINEAR: model #3's theta-scheme ____/  \\__ NONLINEAR EXCESS: DG, theta=1/2 __/

The **nonlinearity coefficient is** ``a = EA - T0``, so ``EA = T0`` is **exactly linear** -- three
decoupled waves, and the ``u`` polarization is then **bit-for-bit**
:class:`~physsynth.core.string_damped.DampedStiffString`. That is this model's regression anchor,
the direct analog of model #9's ``EA = 0``.

Expanding the excess for small slopes (``r^2 = u_x^2 + w_x^2``) shows **the two claims as two
terms**::

    V_nl = a * [ r^2 v_x / 2  +  r^4 / 8  + ... ]
                \\_________/     \\______/
                 PHANTOM         model #9's KC quartic, recovered LOCALLY (u_x^4) rather
                 PARTIALS        than spatially averaged (I^2/L). The KC<->GE distinction.

``r^2 v_x/2`` is quadratic in transverse times linear in longitudinal: two transverse partials at
``f_i``, ``f_j`` drive the **longitudinal** field at ``f_i +- f_j``. That term *is* phantom
partials, and it is **structurally absent** from model #9, which has no ``v``.

**Parameterisation.** ``c = sqrt(T0/rho)`` for **both** polarizations (the tension is isotropic by
construction); ``c_long = sqrt(EA/rho)``. The governing ratio is ``EA/T0 = (c_long/c)^2`` --
*literally* model #9's ratio, so :func:`~physsynth.core.string_nonlinear.\
string_coefficients_from_material` carries over. **But mind the identification
``EA_#9 <-> (EA - T0)_#10``**: cross-model KC checks show a ~0.2-0.7 % offset (``T0/EA ~ 1/150 ..
1/600``). That is the identification, not a discrepancy -- do not chase it.

**The discrete gradient -- closed form, and NO 0/0 branch.** ``V`` is not quadratic (unlike model
#9), so a genuine discrete gradient is required: ``<gradbar V, q+ - q-> = V(q+) - V(q-)`` exactly.
It has a closed form, via two exact facts:

1. ``g = Lambda^2`` is **quadratic** in ``q``, so the midpoint rule is exact:
   ``<grad g(qbar), q+ - q-> = g+ - g-``.
2. The square-root difference quotient **rationalizes**:
   ``(sqrt(g+) - sqrt(g-)) / (g+ - g-) = 1 / (Lambda+ + Lambda-)`` -- exactly, no limit, no Taylor.

Chaining them collapses the whole thing to one substitution:

    **The exact DG is the CONTINUUM gradient at the midpoint strain ``qbar``, with the single
    replacement ``Lambda(qbar) -> Lambdabar = (Lambda+ + Lambda-)/2``.  mean(Lambda), NOT
    Lambda(mean).**

    gradbar V_nl = a * ( chi * ubar_x,  chi * wbar_x,  chi * (1 + vbar_x) - vbar_x )
    chi = 1 - 1/Lambdabar = (Lambdabar - 1)/Lambdabar          the mean strain-ratio

``chi = 0`` at rest, ``> 0`` stretched (hardening), ``< 0`` slack. **``mean(Lambda)`` vs
``Lambda(mean)`` is the single highest-risk line in this model**: the naive midpoint's DG error
*shrinks with amplitude* (measured 1.4e-2 -> 1.3e-5 -> 1.8e-8 as strain goes 0.1 -> 0.01 -> 0.001),
so it looks *nearly right* in every qualitative test -- right glide, right spectrum -- and fails
**only** the energy gate.

Contrast the family: models #7/#8 need ``collision.py``'s ``[DG]`` 0/0 Taylor branch (power-law
potential, genuinely 0/0 in the quiet region); model #9 needs no DG at all (quadratic ``V`` ->
midpoint collapse). Model #10 sits between: a real DG, but **exactly regular** -- the denominator is
``Lambda+ + Lambda- ~ 2`` for any physical configuration (``Lambda`` is a *stretch ratio*;
``Lambda -> 0`` means an element crushed to zero length). **Do not import ``[DG]``** -- there is no
0/0 here to protect against.

**The split (why ``EA = T0`` stays bit-identical).** Model #9's trick, verbatim. The excess needs
theta=1/2 to telescope, but model #3 averages its operator at theta ~ 0.28. Split, and average only
the **excess** at theta=1/2 -- the pieces telescope independently, so it is free::

    rho delta_tt u = L_u . (theta u+ + (1-2theta) u^n + theta u-)          <- model #3, VERBATIM
                     - 2 rho sigma0 delta_t. u + 2 rho sigma1 delta_t.(delta_xx u)
                     + delta_x-[ gradbar V_nl ]_u                          <- NEW, theta=1/2 (DG)
    rho delta_tt w = (same, with kappa_w)
    rho delta_tt v = EA delta_xx(theta v+ + (1-2theta) v^n + theta v-)
                     - 2 rho sigma0_long delta_t. v + 2 rho sigma1_long delta_t.(delta_xx v)
                     + delta_x-[ gradbar V_nl ]_v                          <- no kappa: no bending

``L_u = c^2 delta_xx - kappa_u^2 delta_xxxx``, ``L_w = c^2 delta_xx - kappa_w^2 delta_xxxx``.
**The detuning lives there and nowhere else**: ``c`` is shared and ``gradbar V_nl`` sees the
polarizations only through ``r^2 = u_x^2 + w_x^2``, so ``kappa_u != kappa_w`` splits **only the
linear operator** and the geometric nonlinearity stays exactly isotropic.

**Energy.**

    E^n = [ model #3's theta-form, per field, with each field's own L ]
          + (1/2) ( Vnl(q^n) + Vnl(q^{n-1}) )

The nonlinear term is the **two-time half-average** -- model #6/#9's odd/even lesson, here
*derived*: by the SBP adjoint pair ``<delta_x- F, y> = -<F, delta_x+ y>``, the DG identity makes the
nonlinear power exactly ``-delta_t+[ (Vnl^n + Vnl^{n-1})/2 ]``.

**The energy floor is 0, and that is less obvious than it looks.** Measured against the natural
(zero-tension) length ``Lambda0 = a/EA``, the potential is exactly::

    V(q) = (EA/2)(Lambda - Lambda0)^2 - T0^2/(2 EA) - T0 v_x

The ``-T0 v_x`` null Lagrangian telescopes to ``T0 (v_N - v_0) = 0`` at fixed ends and drops from
the sum, but the pre-stress density ``T0^2/(2 EA)`` survives on every cell -- which tempts the
conclusion ``E >= -L T0^2/(2 EA)`` (a string relaxed everywhere). **That state is inadmissible**:
relaxing every element needs ``v_x = -T0/EA`` throughout, i.e. ``v_N - v_0 != 0``, and both ends
are clamped. Imposing the constraint, ``Lambda >= 1 + v_x`` gives ``mean(Lambda) >= 1``, and Jensen
makes the square term dominate the pre-stress term exactly::

    =>  E >= 0,   with equality iff Lambda == 1 everywhere and the string is at rest.

``-L T0^2/(2 EA)`` is the floor for a **free** string -- relevant only if a free end is ever added.
See :attr:`GeometricString.energy_floor`. Structural only at theta=1/2 (the split's two averaging
rules recombine into ``h sum V`` only there); at theta ~ 0.28 it is an empirical bar. If it is ever
hit, **distinguish bug from physics via ``min Lambda``**: a slack element (negative :attr:`tension`
-> buckling) is *real physics* -- measured, the scheme conserves energy to 1e-12 straight through
local slackness -- not a scheme artifact.

**Resolve the longitudinal field: ``lam_long = c_long k / h`` governs, and nothing enforces it.**
The scheme is unconditionally stable, so there is no CFL to violate and no error to raise -- but
``c_long/c = sqrt(EA/T0)`` is ~22 at realistic ``EA/T0``, so the familiar ``lam = 0.5`` silently
means ``lam_long = 11``. Measured: ``lam_long <= 2`` conserves to ~1e-12 on every hard case tried;
by ``lam_long >= 4`` the Newton solve stops converging and drift explodes to 1e+3 .. 1e+5. *Stable
is not accurate.* Oversample (CLAUDE.md's rule for nonlinearities) and check :attr:`lam_long`.

**Solve.** Vector Newton on ``3(N-1)`` unknowns ``[u+; w+; v+]``. ``gradbar V_nl`` is **local per
cell**, so the Jacobian ``J = A0 - (k^2/rho) delta_x- D_cell delta_x+`` has ``D_cell``
block-diagonal (3x3 per cell) and is **sparse**. ``splu``, not ``cholesky_banded``: the DG Jacobian
is **not symmetric** (a discrete gradient is not the gradient of anything -- ``D_cell`` pairs a
midpoint ``mbar`` against a plus-level ``n+``). Damped Newton + Armijo; ``newton_tol`` is exposed
and **drift is proportional to it** -- the self-certification, absent a closed form for general
motion. **Uniqueness is deliberately not gated** (bow / model #9 precedent): *any* root conserves
energy exactly, since the telescoping needs only the DG identity.

**Bending is added linearly** (Euler-Bernoulli, per polarization), *not* geometrically-exact. This
is honestly a "stiff geometrically-exact string", not a Timoshenko/Cosserat rod.

Headless: NumPy + SciPy (sparse LU, banded Cholesky). No I/O, no plotting.
See ``docs/dev/geometrically-exact-string-plan.md``.
"""

from __future__ import annotations

import warnings
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse.linalg import splu

from .operators import biharmonic_matrix, second_difference_matrix
from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]

NEWTON_TOL_DEFAULT = 1e-15
"""**Relative** tolerance on the max-norm of the update residual (see :meth:`GeometricString.\
_solve_newton`).

Unit-free: the bar is ``newton_tol * max|Y_seed|``, following model #9's normalized tension solve
("so ``tension_tol`` is a unit-free relative bar"). An **absolute** bar here is a trap -- the
residual scales with the displacement, so a fixed ``1e-14`` is ``1e-11`` relative on a 1 mm string
and lands the drift at ``2e-10``, *over* the ``1e-10`` gate, while looking like a tight number.
Measured: drift is proportional to this over five decades.
"""

NEWTON_MAXITER_DEFAULT = 60
"""Cap on damped-Newton iterations per step. Reaching it warns; it never silently renders."""

LAM_LONG_WARN = 1.0
"""Warn when ``lam_long = c_long k / h`` exceeds this. **The one guard with no CFL behind it.**

Every other explicit scheme in this package rejects ``lam > 1`` because the scheme is *unstable*
above it. This one is an **accuracy** bar on an unconditionally stable scheme, which is exactly what
makes it dangerous: nothing throws, nothing violates, and the model quietly returns nonsense.
Measured, over plucked and mode-3 ICs at amplitudes up to 1e-2::

    lam_long <= 2    drift ~1e-12 .. 1e-13   conserves, every case tried
    lam_long  = 4    drift 1e-13 .. 2e+3     case-dependent
    lam_long >= 8    drift 1e+3 .. 1e+5      Newton stops converging; blow-up

The bar sits at 1 rather than the measured-safe 2 to mirror this project's "tune toward lambda = 1"
rule, and to keep 4x of margin on a *sharp* cliff. It **warns rather than rejects**:
``lam_long = 2`` demonstrably conserves, so a hard bar would forbid working configurations -- and
the regime above is worth being able to *study*, just not to trust. Because
``c_long/c = sqrt(EA/T0) ~ 22`` at realistic stiffness, the familiar transverse ``lam = 0.5`` lands
at ``lam_long ~ 11``: this warning will fire
on the parameters a reader of models #1-#9 would reach for first. That is its whole purpose.

**Not raised at ``EA == T0``** (``a == 0``), regardless of ``lam_long``. There the three fields
decouple and the model *is* :class:`~physsynth.core.string_damped.DampedStiffString` three times
over -- which does not warn about its own ``lambda`` either. The exemption is load-bearing rather
than tidy: the ``EA = T0`` bit-identity anchor sits at ``lam_long == 1.0`` **exactly**, flush
against this bar, so without it a float wobble would fire a spurious warning on this model's
single most important regression test the day CI turns warnings into errors.

**The bar is one-sided, and mind the flip.** It reads ``lam_long`` because the *longitudinal* field
is the fast one -- but only while ``EA > T0``. Below the anchor ``c_long < c``, the **transverse**
field becomes the fast one and plain ``lam`` governs instead; this bar then says nothing useful, and
a caller who resolves ``lam_long`` on a softening string is under-resolving the wave that actually
sets the timestep (measured: ``EA/T0 = 1/200`` at ``lam_long = 0.5`` means ``lam = 7`` and the
Newton Jacobian goes singular, while the same string at ``lam = 0.5`` conserves to 1e-13).
Softening is a deliberate opt-in (see ``allow_softening``), so this is documented rather than
branched on."""


class GeometricState(NamedTuple):
    """A snapshot of the three displacement fields (copies, safe to mutate/store)."""

    u: NDArray[np.float64]
    """Transverse polarization 1 (m) -- the one that reduces to model #3 at ``EA = T0``."""
    w: NDArray[np.float64]
    """Transverse polarization 2 (m) -- the out-of-plane direction."""
    v: NDArray[np.float64]
    """Longitudinal displacement (m) -- where phantom partials live."""


class GeometricString:
    """A geometrically-exact stiff string: two polarizations + longitudinal motion (model #10).

    ``EA = T`` reduces the ``u`` polarization **bit-for-bit** to
    :class:`~physsynth.core.string_damped.DampedStiffString` (the nonlinearity coefficient is
    ``EA - T0``, so it vanishes identically).

    Parameters
    ----------
    L, T, rho : float
        Length (m), **rest** tension (N), linear density (kg/m). ``c = sqrt(T/rho)`` for *both*
        polarizations -- the tension is isotropic by construction.
    fs : float
        Sample rate (Hz); ``k = 1/fs``. No CFL limit (unconditional for ``theta >= 1/4``), though
        the nonlinearity wants oversampling on its own account (HANDOFF section 8), and the
        **longitudinal** field wants it much more (see :attr:`lam_long`).
    N : int
        Spatial segments; ``N + 1`` nodes, ``h = L/N``, ends clamped in **all three** components
        (``u = w = v = 0``), ``N - 1`` unknowns per field.
    EA : float
        **Axial stiffness (N)** -- required, and the defining parameter. ``EA = T`` -> exactly
        linear (the regression anchor). The governing ratio is ``EA/T = (c_long/c)^2``; real
        strings sit at ``EA/T ~ 150-600``. See
        :func:`~physsynth.core.string_nonlinear.string_coefficients_from_material` -- but mind
        ``EA_#9 <-> (EA - T0)_#10``.
    kappa : float
        Bending stiffness of the ``u`` polarization, ``sqrt(E I / rho)`` (m^2/s). ``0`` -> flexible.
    kappa_w : float or None
        Bending stiffness of the ``w`` polarization. ``None`` (default) -> **same as** ``kappa``:
        a circular cross-section, and the degenerate string. Setting ``kappa_w != kappa`` models a
        **non-circular cross-section** and breaks the ``u``/``w`` degeneracy -- which is what
        **whirling requires**: an isotropic string provably cannot whirl (see
        :meth:`is_degenerate`). The knob touches only the *linear* operator, so the geometric
        nonlinearity stays exactly isotropic.
    sigma0, sigma1 : float
        Frequency-independent / -dependent transverse loss (>= 0), model #3's, applied to **both**
        polarizations.
    sigma0_long, sigma1_long : float or None
        The same two losses for the **longitudinal** field. ``None`` (default) -> inherit the
        transverse values, so the constructor makes no silent physics claim. **Real strings damp
        longitudinal motion far less than transverse** -- that is a setting to *opt into* (pass
        ``sigma0_long=0.0``), not a default.
    theta : float
        Time-averaging weight of the **linear** operator, ``(0, 1]``; ``>= 1/4`` unconditionally
        stable. The nonlinear excess always uses ``theta = 1/2`` regardless -- that is not a knob,
        it is what makes the DG telescope. At ``theta = 1/2`` the :attr:`energy_floor` becomes
        structural.
    boundary : {"supported"}
        Simply supported transversely (``u = u_xx = 0``) and Dirichlet longitudinally (``v = 0``).
    newton_tol : float
        **Relative** max-norm tolerance of the vector Newton solve (the bar is
        ``newton_tol * max|Y_seed|`` -- model #9's unit-free idiom). **Energy drift is proportional
        to it** -- the self-certification, absent a closed form for general motion. Do not make it
        absolute: the residual scales with displacement, so an absolute bar loosens silently as the
        string quietens.
    newton_maxiter : int
        Iteration cap; exceeding it warns (never silently renders).
    allow_softening : bool
        Permit ``EA < T``. **Off by default -- but the line it draws is materials, not stability.**

        ``EA < T`` makes ``Lambda0 = (EA - T0)/EA`` negative, and ``Lambda0`` is *both* the
        minimizer of ``V`` *and* the natural-length ratio of an element (see :attr:`energy_floor`
        for why it is the same number). A negative natural length is not a thing a real material
        has: the string would be pre-stretched from an unstretched state shorter than nothing.

        **What it is not is unstable, and an earlier version of this docstring said otherwise.**
        The claim was that ``EA_n = EA - T0 < 0`` leaves the potential unbounded below. It does
        not: the identity in :attr:`energy_floor` holds for **either sign** of ``a``, ``EA > 0`` is
        separately enforced, so ``(EA/2)(Lambda - Lambda0)^2 >= 0`` regardless -- and the same
        Jensen step gives ``E >= 0`` at ``a < 0`` too. A softening string cannot even go slack
        (``tension = EA Lambda - a = EA Lambda + |a| > 0`` for every ``Lambda > 0``, where a
        *hardening* one genuinely can). Measured, plucked, at ``EA/T0`` from ``1/2`` down to
        ``1/200``: drift ~1e-13, ``E`` positive, ``min(tension) ~ T0``. So :attr:`energy_floor` and
        the drift gate are **not** void with the hatch open -- both still hold, and the tests
        assert it.

        The hatch therefore reads the way [[unphysical-params-are-a-feature]] asks, from the other
        side: the surface ``(T, rho, kappa, EA)`` stays mutually unconstrained, and the *unphysical*
        corner is opt-in rather than forbidden. **Resolution flips here**: below ``EA = T0`` the
        longitudinal wave is the slow one, so resolve ``lam``, not :attr:`lam_long` (see
        :data:`LAM_LONG_WARN`).

    Raises
    ------
    ValueError
        On non-physical parameters (non-positive ``L``/``T``/``rho``/``fs``/``EA``, negative
        stiffness or losses, ``N < 2``, ``theta`` outside ``(0, 1]``, unsupported boundary), or on
        ``EA < T`` without ``allow_softening``.
    """

    def __init__(
        self,
        *,
        L: float,
        T: float,
        rho: float,
        fs: float,
        N: int,
        EA: float,
        kappa: float = 0.0,
        kappa_w: float | None = None,
        sigma0: float = 0.0,
        sigma1: float = 0.0,
        sigma0_long: float | None = None,
        sigma1_long: float | None = None,
        theta: float = THETA_DEFAULT,
        boundary: Boundary = "supported",
        newton_tol: float = NEWTON_TOL_DEFAULT,
        newton_maxiter: int = NEWTON_MAXITER_DEFAULT,
        allow_softening: bool = False,
    ) -> None:
        if min(L, T, rho, fs) <= 0:
            raise ValueError("L, T, rho, fs must all be positive.")
        if EA <= 0:
            raise ValueError("EA (axial stiffness) must be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if kappa < 0:
            raise ValueError("kappa (stiffness) must be >= 0.")
        if kappa_w is not None and kappa_w < 0:
            raise ValueError("kappa_w (stiffness) must be >= 0.")
        if sigma0 < 0 or sigma1 < 0:
            raise ValueError("sigma0, sigma1 (losses) must be >= 0.")
        if (sigma0_long is not None and sigma0_long < 0) or (
            sigma1_long is not None and sigma1_long < 0
        ):
            raise ValueError("sigma0_long, sigma1_long (losses) must be >= 0.")
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if newton_tol <= 0:
            raise ValueError("newton_tol must be > 0.")
        if newton_maxiter < 1:
            raise ValueError("newton_maxiter must be >= 1.")
        if boundary != "supported":
            raise ValueError(f"boundary must be 'supported', got {boundary!r}.")
        if EA < T and not allow_softening:
            raise ValueError(
                f"EA ({EA}) < T ({T}) makes the natural (unstretched) length ratio Lambda0 = "
                f"(EA - T0)/EA = {(EA - T) / EA:.4g} NEGATIVE, i.e. a SOFTENING string no real "
                f"material can be: at rest every element is already stretched from a natural "
                f"length below zero. The model stays well-posed there (energy still conserves, "
                f"E >= 0 still holds, and the string cannot go slack -- tension = EA*Lambda + "
                f"|EA - T0| > "
                f"0 always), so this is hyperreality, not blow-up. Pass allow_softening=True to "
                f"build it -- and mind that below EA = T0 the LONGITUDINAL wave is the slow one, "
                f"so resolve the transverse lam, not lam_long."
            )

        self.L = float(L)
        self.T = float(T)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.EA = float(EA)
        self.kappa = float(kappa)
        self.kappa_u = self.kappa
        """Alias of :attr:`kappa` -- the ``u`` polarization's bending stiffness."""
        self.kappa_w = self.kappa if kappa_w is None else float(kappa_w)
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.sigma0_long = self.sigma0 if sigma0_long is None else float(sigma0_long)
        self.sigma1_long = self.sigma1 if sigma1_long is None else float(sigma1_long)
        self.theta = float(theta)
        self.boundary: Boundary = boundary
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)
        self.allow_softening = bool(allow_softening)

        self.c = float(np.sqrt(T / rho))
        self.c_long = float(np.sqrt(EA / rho))
        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.lam = self.c * self.k / self.h  # reported only; no CFL limit (unconditional)
        self.lam_long = self.c_long * self.k / self.h
        self.B = float((np.pi**2) * self.kappa**2 / (self.c**2 * self.L**2))
        self.EA_over_T = self.EA / self.T

        # a = EA - T0 is THE nonlinearity coefficient. a == 0.0 exactly -> the linear code path,
        # which is what earns the bit-for-bit model #3 anchor.
        self._a = self.EA - self.T

        # The one guard with no CFL behind it. Warn, do not reject: the scheme really is
        # unconditionally stable, and lam_long = 2 conserves to 1e-12 -- a hard bar would forbid
        # configurations that demonstrably work. See LAM_LONG_WARN.
        #
        # Skipped entirely at a == 0: the model is then literally DampedStiffString x3 (the fields
        # do not couple, and nothing rides on the longitudinal field's accuracy that model #3 does
        # not already own), and model #3 does not warn about its own lambda. This is not cosmetic --
        # the EA = T0 anchor lands at lam_long == 1.0 exactly, flush against the bar, so without
        # this branch a float wobble would fire a spurious warning on the single most important
        # regression test in the model the day CI turns warnings into errors.
        if self._a != 0.0 and self.lam_long > LAM_LONG_WARN:
            warnings.warn(
                f"lam_long = {self.lam_long:.2f} > {LAM_LONG_WARN}: the longitudinal field "
                f"advances {self.lam_long:.1f} cells per timestep and is under-resolved in time. "
                f"The scheme is unconditionally STABLE here, so no CFL is violated and nothing "
                f"else will warn -- but stable is not accurate: past lam_long ~ 4 the Newton solve "
                f"stops converging and energy drift explodes (measured 1e+3 .. 1e+5). "
                f"c_long/c = sqrt(EA/T) = "
                f"{np.sqrt(self.EA_over_T):.1f}, so a transverse lam of "
                f"{self.lam:.3g} buys this. Raise fs (or lower EA) until lam_long <= 1.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)

        n_int = self.N - 1
        self._D2 = second_difference_matrix(self.N, self.h)

        # Per-field conservative linear operators. The detuning kappa_u != kappa_w lives HERE and
        # nowhere else -- c is shared, and the nonlinearity sees only r^2 = u_x^2 + w_x^2.
        self._L_u = self._wave_operator(self.kappa_u)
        self._L_w = self._wave_operator(self.kappa_w)
        self._L_v = ((self.c_long**2) * self._D2).tocsr()  # no bending: longitudinal has none

        # A_f = (1 + sigma0 k) I - theta k^2 L_f - sigma1 k D2 -- model #3's matrix, per field.
        # NOTE: unlike model #9, A does NOT move with the state: the nonlinearity is a force on the
        # RHS, not a term in the matrix. The banded Cholesky factors below are therefore valid for
        # the whole run, and serve as both the EA=T fast path and the Newton seed.
        self._A_u = self._update_matrix(self._L_u, self.sigma0, self.sigma1)
        self._A_w = self._update_matrix(self._L_w, self.sigma0, self.sigma1)
        self._A_v = self._update_matrix(self._L_v, self.sigma0_long, self.sigma1_long)
        self._chol_u = cholesky_banded(self._banded(self._A_u), lower=False)
        self._chol_w = cholesky_banded(self._banded(self._A_w), lower=False)
        self._chol_v = cholesky_banded(self._banded(self._A_v), lower=False)
        self._A3 = sparse.block_diag([self._A_u, self._A_w, self._A_v], format="csr")

        # The SBP adjoint pair on the CELL <-> NODE staggering. Gp maps the N-1 interior nodes to
        # the N inter-node strains (the clamped ends contribute their own slopes); Gm = -Gp^T is
        # its adjoint, mapping cell forces back to interior nodes. Gm @ Gp == D2 exactly.
        inv_h = 1.0 / self.h
        self._Gp = sparse.diags(
            [np.full(n_int, inv_h), np.full(n_int, -inv_h)],
            offsets=[0, -1],
            shape=(self.N, n_int),
            format="csr",
        )
        self._Gm = (-self._Gp.T).tocsr()
        self._Gp3 = sparse.block_diag([self._Gp] * 3, format="csr")
        self._Gm3 = sparse.block_diag([self._Gm] * 3, format="csr")

        self.u: NDArray[np.float64] = np.zeros(self.N + 1)
        self.w: NDArray[np.float64] = np.zeros(self.N + 1)
        self.v: NDArray[np.float64] = np.zeros(self.N + 1)
        self.u_prev: NDArray[np.float64] = np.zeros(self.N + 1)
        self.w_prev: NDArray[np.float64] = np.zeros(self.N + 1)
        self.v_prev: NDArray[np.float64] = np.zeros(self.N + 1)
        self.n: int = 0  # number of completed steps

        # Solver telemetry (the self-certifying gates -- see the plan doc).
        self.converged: bool = True
        """Whether the most recent Newton solve converged."""
        self.newton_iters: int = 0
        """Iterations taken by the most recent :meth:`step`."""
        self.total_newton_iters: int = 0
        """Cumulative Newton iterations -- the cost telemetry."""
        self.n_not_converged: int = 0
        """Cumulative steps whose Newton solve stalled. Never render such a run as physics."""

    # -- construction helpers -----------------------------------------------------------

    def _wave_operator(self, kappa: float) -> sparse.csr_matrix:
        """``L = c^2 delta_xx - kappa^2 delta_xxxx`` (model #3's operator, per polarization)."""
        op = (self.c**2) * self._D2
        if kappa != 0.0:
            op = op - (kappa**2) * biharmonic_matrix(self.N, self.h)
        return op.tocsr()

    def _update_matrix(
        self, op: sparse.csr_matrix, sigma0: float, sigma1: float
    ) -> sparse.csr_matrix:
        """``A = (1 + sigma0 k) I - theta k^2 L - sigma1 k D2`` (model #3's, verbatim)."""
        ident = sparse.identity(self.N - 1, format="csr")
        A = (1.0 + sigma0 * self.k) * ident - (self.theta * self.k**2) * op
        if sigma1 != 0.0:
            A = A - (sigma1 * self.k) * self._D2
        return A.tocsr()

    # -- initial conditions -------------------------------------------------------------

    def set_state(
        self,
        u0: NDArray[np.float64] | float = 0.0,
        w0: NDArray[np.float64] | float = 0.0,
        v0: NDArray[np.float64] | float = 0.0,
        *,
        u_dot: NDArray[np.float64] | float = 0.0,
        w_dot: NDArray[np.float64] | float = 0.0,
        v_dot: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacements (and optional velocities).

        .. warning::
           **Name clash with models #1-3/#9.** There, ``set_state(u0, v0)``'s ``v0`` is the initial
           **velocity**. Here ``v`` is the **longitudinal displacement field**, so ``v0`` is a
           *displacement* and the velocities are the keyword-only ``u_dot``/``w_dot``/``v_dot``.
           Porting a call from model #3 without renaming silently injects a longitudinal
           displacement where a transverse velocity was meant.

        Uses the lossless consistent second-order start ``y^{-1} = y^0 - k ydot^0 + (k^2/2) ytt^0``,
        with ``ytt^0`` including the **nonlinear** force at ``t = 0`` (the *continuum* gradient at
        ``q^0`` -- at a single time level there is no DG) so a single eigenmode starts as a clean
        discrete Duffing cosine. ``EA = T`` skips that term -> model #3's start exactly. All three
        components are clamped at both ends. Under damping the start is only *consistent* (not
        exact), as in model #3.
        """
        u0 = self._as_field(u0, "u0")
        w0 = self._as_field(w0, "w0")
        v0 = self._as_field(v0, "v0")
        dots = [self._as_field(d, name) for d, name in ((u_dot, "u_dot"), (w_dot, "w_dot"),
                                                        (v_dot, "v_dot"))]
        for f in (u0, w0, v0, *dots):
            f[0] = f[-1] = 0.0

        accel = [
            self._apply_full(self._L_u, u0),
            self._apply_full(self._L_w, w0),
            self._apply_full(self._L_v, v0),
        ]
        if self._a != 0.0:
            q0 = self._strain(u0, w0, v0)
            force = self._dg_force(q0, q0)  # DG with q+ == q- IS the continuum gradient
            for i in range(3):
                accel[i][1:-1] += (self._Gm @ force[i]) / self.rho

        prev = [
            f - self.k * d + 0.5 * self.k**2 * a
            for f, d, a in zip((u0, w0, v0), dots, accel, strict=True)
        ]
        for f in prev:
            f[0] = f[-1] = 0.0

        self.u, self.w, self.v = u0, w0, v0
        self.u_prev, self.w_prev, self.v_prev = prev
        self.n = 0
        self.converged = True
        self.newton_iters = 0

    def _as_field(self, value: NDArray[np.float64] | float, name: str) -> NDArray[np.float64]:
        """Broadcast/validate an IC argument to a fresh full-grid ``(N+1,)`` array."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.N + 1, float(arr))
        if arr.shape != (self.N + 1,):
            raise ValueError(f"{name} must have shape {(self.N + 1,)}, got {arr.shape}.")
        return arr.copy()

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep (rolls the history).

        ``EA = T``: three independent banded back-substitutions against the prefactored model-#3
        matrices -- the identical code path per field, hence bit-identical results. Otherwise: a
        damped Newton solve on the coupled ``3(N-1)`` system (see :meth:`_solve_newton`).
        """
        rhs = [
            self._rhs0(self.u, self.u_prev, self._L_u, self.sigma0, self.sigma1),
            self._rhs0(self.w, self.w_prev, self._L_w, self.sigma0, self.sigma1),
            self._rhs0(self.v, self.v_prev, self._L_v, self.sigma0_long, self.sigma1_long),
        ]

        if self._a == 0.0:
            nxt = [
                cho_solve_banded((self._chol_u, False), rhs[0]),
                cho_solve_banded((self._chol_w, False), rhs[1]),
                cho_solve_banded((self._chol_v, False), rhs[2]),
            ]
            self.converged = True
            self.newton_iters = 0
        else:
            nxt = self._solve_newton(rhs)

        rolled = []
        for interior in nxt:
            full = np.zeros(self.N + 1)
            full[1:-1] = interior
            rolled.append(full)
        self.u_prev, self.w_prev, self.v_prev = self.u, self.w, self.v
        self.u, self.w, self.v = rolled
        self.n += 1

    def _rhs0(
        self,
        fn: NDArray[np.float64],
        fp: NDArray[np.float64],
        op: sparse.csr_matrix,
        sigma0: float,
        sigma1: float,
    ) -> NDArray[np.float64]:
        """Model #3's RHS, **expression-for-expression**.

        Float addition is not associative -- this ordering is what makes ``EA = T`` bit-identical to
        :class:`~physsynth.core.string_damped.DampedStiffString`, not merely equal to tolerance.
        """
        un = fn[1:-1]
        up = fp[1:-1]
        Lu = op @ un
        Lu_prev = op @ up
        rhs = (
            2.0 * un
            + (1.0 - 2.0 * self.theta) * self.k**2 * Lu
            - up
            + self.theta * self.k**2 * Lu_prev
            + sigma0 * self.k * up
        )
        if sigma1 != 0.0:
            rhs = rhs - (sigma1 * self.k) * (self._D2 @ up)
        return rhs

    def _solve_newton(self, rhs: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
        """Damped Newton + Armijo on the coupled ``3(N-1)`` system; returns the three interiors.

        Residual ``F(Y) = A3 Y - rhs0 - (k^2/rho) delta_x-[ gradbar V_nl(delta_x+ Y, q^{n-1}) ]``.
        Note the DG depends on ``y^{n+1}`` and ``y^{n-1}`` only -- never on ``y^n``; that is the
        discrete-gradient structure, and it is what telescopes.

        Seeded with the **linear** (force-free) solve against the prefactored factors -- ``A3`` is
        constant here, unlike model #9's, so that seed is free and is the exact answer whenever the
        nonlinearity is weak.

        **The convergence bar is RELATIVE** (``newton_tol * max|Y_seed|``), model #9's unit-free
        idiom. This is not fussiness: the residual scales with the displacement, so an absolute bar
        silently loosens as the string gets quieter, and it is the **energy drift's floor** --
        measured drift tracks it proportionally over five decades (``1e-12 -> 8e-9``,
        ``1e-16 -> 2e-12``), because the DG force is exact only *at* the root. A run that decays
        under damping keeps its relative bar automatically; at exact rest the residual is exactly
        ``0.0`` (``gradbar V_nl(0, 0) == 0`` identically) and the seed is the answer.

        **Uniqueness is deliberately not gated** (the :class:`~physsynth.core.bow.BowedString` and
        model #9 precedent): *any* root conserves energy exactly, since the telescoping needs only
        the DG identity ``<gradbar V, q+ - q-> = V(q+) - V(q-)``. Non-uniqueness would be a
        branch-selection question, not a correctness one.
        """
        rhs3 = np.concatenate(rhs)
        q_minus = self._strain(self.u_prev, self.w_prev, self.v_prev)
        force_pref = self.k**2 / self.rho

        def residual(Y: NDArray[np.float64]) -> NDArray[np.float64]:
            q_plus = (self._Gp3 @ Y).reshape(3, self.N)
            f = self._dg_force(q_plus, q_minus).ravel()
            return self._A3 @ Y - rhs3 - force_pref * (self._Gm3 @ f)

        Y = np.concatenate(
            [
                cho_solve_banded((self._chol_u, False), rhs[0]),
                cho_solve_banded((self._chol_w, False), rhs[1]),
                cho_solve_banded((self._chol_v, False), rhs[2]),
            ]
        )
        tol_abs = self.newton_tol * float(np.max(np.abs(Y)))
        r = residual(Y)
        iters = self.newton_maxiter
        for it in range(self.newton_maxiter):
            if np.max(np.abs(r)) <= tol_abs:
                iters = it
                break
            q_plus = (self._Gp3 @ Y).reshape(3, self.N)
            jac = self._A3 - force_pref * (self._Gm3 @ self._dg_jacobian(q_plus, q_minus)
                                           @ self._Gp3)
            delta = splu(jac.tocsc()).solve(-r)
            # Armijo backtracking on 0.5||r||^2. The DG is smooth (no kink, unlike model #8's
            # [eta]+), so full steps are the norm; this only guards the far-from-root transient.
            f0 = 0.5 * float(r @ r)
            t = 1.0
            for _ls in range(40):
                r_try = residual(Y + t * delta)
                if 0.5 * float(r_try @ r_try) < (1.0 - 1e-4 * t) * f0:
                    break
                t *= 0.5
            Y = Y + t * delta
            r = residual(Y)

        if np.max(np.abs(r)) > tol_abs:
            self.converged = False
            self.n_not_converged += 1
            warnings.warn(
                f"Geometric string Newton solve did not converge at step {self.n} in "
                f"{self.newton_maxiter} iterations (residual {np.max(np.abs(r)):.2e} > "
                f"{tol_abs:.1e}); energy may drift. The DG force is exact only *at* the root. "
                f"Raise newton_maxiter or oversample. Do not treat this run as physics.",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            self.converged = True
        self.newton_iters = iters
        self.total_newton_iters += iters
        return [Y[: self.N - 1], Y[self.N - 1 : 2 * (self.N - 1)], Y[2 * (self.N - 1) :]]

    # -- the discrete gradient ----------------------------------------------------------

    def _strain(
        self, u: NDArray[np.float64], w: NDArray[np.float64], v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Cell strains ``q = (u_x, w_x, v_x)`` as ``(3, N)`` from **full-grid** fields."""
        return np.stack((np.diff(u), np.diff(w), np.diff(v))) / self.h

    @staticmethod
    def _stretch_ratio(q: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Lambda = sqrt((1 + v_x)^2 + u_x^2 + w_x^2)`` per cell -- the local stretch ratio."""
        return np.sqrt((1.0 + q[2]) ** 2 + q[0] ** 2 + q[1] ** 2)

    @staticmethod
    def _stretch_terms(
        q: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        """``(Lambda, Lambda-1, Lambda-(1+v_x), r^2, Lambda+1+v_x)``, free of cancellation.

        **Why this exists (a real defect, measured).** Both ``Lambda - 1`` and ``Lambda - (1+v_x)``
        are ``O(strain)`` or ``O(strain^2)`` quantities assembled from ``O(1)`` ones, so evaluating
        them literally destroys relative accuracy exactly where the model is used: musical strings
        run at strain ``~1e-4 .. 1e-3``. Measured on the raw forms, the DG identity's relative error
        degraded ``4e-16 -> 7e-8`` as strain went ``0.5 -> 1e-3`` -- i.e. **worse the more realistic
        the string**, and within one digit of the energy gate.

        The cure is the *same* rationalization the discrete gradient itself rests on::

            Lambda - 1       = (v_x (2 + v_x) + r^2) / (Lambda + 1)
            Lambda - (1+v_x) = r^2 / (Lambda + 1 + v_x)

        The first is **unconditionally safe** (``Lambda + 1 >= 1`` always). The second degenerates
        (0/0) only for an **inverted** element (``1 + v_x < 0`` with ``r = 0``, i.e. compressed
        through zero length) -- and that is precisely the region where the plain subtraction has no
        cancellation at all (there ``Lambda ~ -(1+v_x)``, so the difference is ``O(1)``). The two
        forms' bad regions are **complementary**, so the branch below is exact everywhere, not a
        tolerance. This is *not* model #7/#8's ``[DG]`` 0/0 Taylor branch: nothing here is
        genuinely 0/0 in the physical region.
        """
        r2 = q[0] ** 2 + q[1] ** 2
        vx = q[2]
        lam = np.sqrt((1.0 + vx) ** 2 + r2)
        lam_m1 = (vx * (2.0 + vx) + r2) / (lam + 1.0)  # = (Lambda^2 - 1)/(Lambda + 1)
        denom = lam + 1.0 + vx
        safe = denom > 1.0  # physical elements sit at denom ~ 2; only inversion goes near 0
        d = np.where(safe, r2 / np.where(safe, denom, 1.0), lam - 1.0 - vx)
        return lam, lam_m1, d, r2, denom

    def _dg_force(
        self, q_plus: NDArray[np.float64], q_minus: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The exact discrete gradient ``gradbar V_nl`` per cell, ``(3, N)``.

        ``<gradbar V_nl, q+ - q-> = V_nl(q+) - V_nl(q-)`` **exactly**. It is the continuum gradient
        at ``qbar`` with the single replacement ``Lambda(qbar) -> Lambdabar = mean(Lambda)``.
        Passing ``q_plus is q_minus`` therefore yields the plain **continuum** gradient -- which is
        exactly how :meth:`set_state` uses it.

        **``mean(Lambda)``, NOT ``Lambda(mean)``.** The naive midpoint's error shrinks with
        amplitude, so it passes every qualitative test and fails only the energy gate.

        Written in the cancellation-free variables of :meth:`_stretch_terms`, using two exact
        rearrangements of the same expression::

            chi = (Lambdabar - 1)/Lambdabar  with  Lambdabar - 1     = mean(Lambda^m - 1)
            [.]_v = (Lambdabar - 1 - vbar_x)/Lambdabar   with the numerator = mean(Lambda^m - 1 -
                    v_x^m)   -- both means are exact because Lambdabar and vbar_x are themselves
                    means, so the per-level stable forms carry straight through.
        """
        lam_p, e_p, d_p, _, _ = self._stretch_terms(q_plus)
        lam_m, e_m, d_m, _, _ = self._stretch_terms(q_minus)
        lam_bar = 0.5 * (lam_p + lam_m)
        q_bar = 0.5 * (q_plus + q_minus)
        chi = 0.5 * (e_p + e_m) / lam_bar  # mean strain-ratio: 0 at rest, >0 stretched, <0 slack
        out = np.empty_like(q_bar)
        out[0] = self._a * chi * q_bar[0]
        out[1] = self._a * chi * q_bar[1]
        out[2] = self._a * (0.5 * (d_p + d_m)) / lam_bar  # = (Lambdabar - 1 - vbar_x)/Lambdabar
        return out

    def _dg_jacobian(
        self, q_plus: NDArray[np.float64], q_minus: NDArray[np.float64]
    ) -> sparse.csr_matrix:
        """``d(gradbar V_nl)/d q+`` as a ``3N x 3N`` sparse matrix of diagonal blocks.

        Per cell this is ``a [ (chi/2) I3 - (1/2) e_v e_v^T + (1/(2 Lambdabar^2)) mbar (n+)^T ]``
        with ``mbar = (ubar_x, wbar_x, 1 + vbar_x)`` and ``n+ = (u_x+, w_x+, 1 + v_x+)/Lambda+``.

        **It is not symmetric** -- ``mbar`` is a midpoint quantity while ``n+`` is a plus-level one.
        A discrete gradient is not the gradient of anything, which is exactly why the Newton solve
        uses ``splu`` and not ``cholesky_banded``.
        """
        lam_p, e_p, _, _, _ = self._stretch_terms(q_plus)
        lam_m, e_m, _, _, _ = self._stretch_terms(q_minus)
        lam_bar = 0.5 * (lam_p + lam_m)
        q_bar = 0.5 * (q_plus + q_minus)
        chi = 0.5 * (e_p + e_m) / lam_bar
        n_p = np.stack((q_plus[0], q_plus[1], 1.0 + q_plus[2])) / lam_p
        m_bar = np.stack((q_bar[0], q_bar[1], 1.0 + q_bar[2]))
        coef = 0.5 / lam_bar**2

        blocks: list[list[sparse.spmatrix]] = []
        for a_i in range(3):
            row: list[sparse.spmatrix] = []
            for b_i in range(3):
                d = coef * m_bar[a_i] * n_p[b_i]
                if a_i == b_i:
                    d = d + 0.5 * chi
                if a_i == 2 and b_i == 2:
                    d = d - 0.5
                row.append(sparse.diags(self._a * d, format="csr"))
            blocks.append(row)
        return sparse.bmat(blocks, format="csr")

    def _nl_density(self, q: NDArray[np.float64]) -> float:
        """``h sum_c V_nl(q_c)`` (J) -- the nonlinear **excess** only, not the whole potential.

        ``V_nl(q) = a [ (u_x^2 + w_x^2)/2 + 1 + v_x - Lambda ]``, and ``V_nl(0) = 0`` exactly (at
        rest ``Lambda = 1``). The linear part lives in the theta-form, per the split.

        Evaluated through the cancellation-free rearrangement (see :meth:`_stretch_terms`)::

            V_nl = a r^2 ((Lambda - 1) + v_x) / (2 (Lambda + 1 + v_x))

        which is exact and keeps full **relative** accuracy on a quantity that is ``O(strain^3)``
        -- the literal expression is a difference of ``O(1)`` terms and was the accuracy limiter on
        the energy gate at musical amplitudes. Leading order this is ``a r^2 v_x / 2``: the phantom
        term, visible directly in the formula.
        """
        if self._a == 0.0:
            return 0.0
        _, lam_m1, d, r2, denom = self._stretch_terms(q)
        safe = denom > 1.0
        dens = np.where(
            safe,
            r2 * (lam_m1 + q[2]) / (2.0 * np.where(safe, denom, 1.0)),
            0.5 * r2 - d,  # inverted element: no cancellation here (d is O(1))
        )
        return float(self._a * self.h * np.sum(dens))

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> GeometricState:
        """The three displacement fields (copies, safe to mutate/store for plotting)."""
        return GeometricState(self.u.copy(), self.w.copy(), self.v.copy())

    @property
    def is_degenerate(self) -> bool:
        """Whether the two polarizations are exactly degenerate (``kappa_u == kappa_w``).

        **A degenerate string provably cannot whirl.** ``w -> -w`` is a reflection symmetry, so a
        planar initial condition stays *bit-exactly* planar forever; and an out-of-plane
        perturbation is only **marginal**, never exponential -- ``dw = u(t)`` is an exact solution
        of the out-of-plane variational equation (it is the rotation generator), so one Floquet
        multiplier is ``+1``, and the conserved Wronskian forces the other to ``+1`` too. The second
        solution grows only *secularly* (linear in ``t``), proportional to injected angular
        momentum. Exponential whirling (Gough 1984, threshold ``eps A^2 ~ dw0^2``) needs
        ``kappa_u != kappa_w``. There is likewise **no universal precession rate**: rotational
        symmetry forces ``omega_w = omega_u`` exactly, at any amplitude.
        """
        return self.kappa_u == self.kappa_w

    @property
    def stretch_ratio(self) -> NDArray[np.float64]:
        """Per-cell stretch ratio ``Lambda^n`` (length ``N``, dimensionless). ``1`` at rest.

        Unlike model #9's scalar :attr:`~physsynth.core.string_nonlinear.\
TensionModulatedString.stretch`, this is a **field** -- which is the whole point of the model.
        """
        return self._stretch_ratio(self._strain(self.u, self.w, self.v))

    @property
    def tension(self) -> NDArray[np.float64]:
        """Per-cell axial tension ``T(Lambda) = EA Lambda - (EA - T0)`` (N) -- a **field**.

        Exactly ``T0`` at rest (``Lambda = 1``), and **linear in ``Lambda``**. Goes **negative**
        (compression -> buckling) when ``Lambda < (EA - T0)/EA = 1/(1 + eps0)``, i.e. when an
        element is squeezed below its natural length. Unlike model #9 -- where ``I >= 0`` forces the
        tension to only ever *rise* -- a geometrically-exact string genuinely can go slack, and that
        is real physics, not a bug. Cross-check it against :attr:`energy_floor` before suspecting
        the scheme.
        """
        return self.EA * self.stretch_ratio - self._a

    @property
    def energy_floor(self) -> float:
        """Lower bound on :meth:`energy` (J): **zero**, attained only at rest.

        Kept as a property rather than inlined as ``0.0`` because the number is not the obvious one
        and the derivation is worth pinning -- it went wrong here in *both* directions before it
        went right. Measure the potential against the natural (zero-tension) length
        ``Lambda0 = a/EA``::

            V(q) = (EA/2)(Lambda - Lambda0)^2 - T0^2/(2 EA) - T0 v_x        (exact, any sign of a)

        The ``-T0 v_x`` term is a **null Lagrangian**: it sums to ``T0 (v_N - v_0) = 0`` at fixed
        ends. The pre-stress density ``T0^2/(2 EA)`` does *not* vanish -- it survives on every cell,
        which is what tempts the conclusion ``E >= -L T0^2/(2 EA)``, the energy of a string relaxed
        everywhere. **That state is inadmissible here**: relaxing every element needs
        ``v_x = -T0/EA`` throughout, hence ``v_N - v_0 = -L T0/EA != 0`` -- the string would have to
        shorten, and both ends are clamped.

        Imposing the constraint instead: ``Lambda >= 1 + v_x`` always (the root only *adds*
        transverse length), so ``mean(Lambda) >= 1 + mean(v_x) = 1``. Jensen then gives
        ``mean((Lambda - Lambda0)^2) >= (mean(Lambda) - Lambda0)^2 >= (1 - Lambda0)^2``, whose
        weight ``(EA/2) L (1 - Lambda0)^2`` is *exactly* the pre-stress term. They cancel::

            E >= 0,    equality iff Lambda == 1 everywhere and the string is at rest.

        So ``-L T0^2/(2 EA)`` is the floor for a **free** string, where the null Lagrangian no
        longer telescopes -- worth remembering if a free end is ever added. Note this bounds the
        *continuum* potential; the discrete ``E`` pairs a cross-time theta-form against a same-time
        nonlinear half-average, so at ``theta ~ 0.28`` the bar is empirical, structural only at
        ``theta = 1/2``.
        """
        return 0.0

    def energy(self) -> float:
        """Discrete energy ``E^n`` (J) = the per-field theta-form + the nonlinear half-average.

        Lossless -> conserved to machine precision; lossy -> monotone decreasing (all six loss
        terms are dissipative by SBP, and none enters ``E``). Bounded below by
        :attr:`energy_floor` -- which is zero, but for a reason worth reading.
        """
        return self._linear_energy() + self.nonlinear_energy()

    def nonlinear_energy(self) -> float:
        """The nonlinear excess part of ``E^n`` alone (J) -- ``0`` iff ``EA = T``.

        The **two-time half-average** ``(Vnl(q^n) + Vnl(q^{n-1}))/2`` -- model #6/#9's odd/even
        lesson, here *derived* rather than certified after the fact (a single-level ``Vnl(q^n)``
        is a 2-step invariant and oscillates spuriously).

        **Sign-indefinite** (the ``r^2 v_x/2`` term), unlike model #9's, so report ``abs()`` of its
        fraction of :meth:`energy` in energy tests: a nonlinearity bug *hides* at small amplitude,
        where the test merely re-runs the linear scheme (model #6's lesson).
        """
        if self._a == 0.0:
            return 0.0
        q_n = self._strain(self.u, self.w, self.v)
        q_p = self._strain(self.u_prev, self.w_prev, self.v_prev)
        return 0.5 * (self._nl_density(q_n) + self._nl_density(q_p))

    def longitudinal_energy(self) -> float:
        """The ``v`` field's kinetic + linear-potential energy alone (J).

        The channel phantom partials live in. Report its fraction in any test that claims to
        exercise longitudinal physics.
        """
        return self.rho * (
            self._kinetic(self.v, self.v_prev) + self._potential(self.v, self.v_prev, self._L_v)
        )

    def displacement_at(self, index: int) -> float:
        """Transverse ``u`` displacement at grid node ``index`` -- the pickup, model #3's signature.

        For the other components read :attr:`w` / :attr:`v` (or :attr:`state`) directly; for what a
        piano bridge actually feels longitudinally, read :attr:`tension`.
        """
        return float(self.u[index])

    def apply_Ainv(self, rhs_int: NDArray[np.float64]) -> NDArray[np.float64]:
        """Not available on this model -- **the one-step response is state-dependent**.

        Model #3 exposes ``A^{-1}``'s action so a coupled element (e.g.
        :class:`~physsynth.core.bow.BowedString`) can precompute a *constant* driving-point
        admittance ``a = A^{-1} e_i``.

        The reason it fails here is **not** model #9's. There, ``A`` itself moves with the tension
        every step. Here ``A3`` is genuinely **constant** -- the nonlinearity is a force on the RHS,
        not a term in the matrix. But the string's one-step response to an injected force is still
        *not* ``A3^{-1}``: the DG force couples all three fields implicitly at ``n+1``, so the true
        admittance is the inverse of the **Newton Jacobian**, which is state-dependent (and not
        symmetric). A precomputed ``A3^{-1} e_i`` would be silently wrong by exactly the
        nonlinearity one came here for. Coupling an exciter needs a joint solve -- deliberately out
        of scope; see the plan doc.
        """
        raise NotImplementedError(
            "GeometricString's one-step admittance is state-dependent: A3 is constant, but the "
            "implicit discrete-gradient force makes the true response the inverse of the Newton "
            "Jacobian, not of A3. Coupling an exciter here requires a joint solve -- see "
            "docs/dev/geometrically-exact-string-plan.md."
        )

    # -- internals ----------------------------------------------------------------------

    def _linear_energy(self) -> float:
        """Model #3's energy form, per field, each with its **own** operator."""
        kinetic = (
            self._kinetic(self.u, self.u_prev)
            + self._kinetic(self.w, self.w_prev)
            + self._kinetic(self.v, self.v_prev)
        )
        potential = (
            self._potential(self.u, self.u_prev, self._L_u)
            + self._potential(self.w, self.w_prev, self._L_w)
            + self._potential(self.v, self.v_prev, self._L_v)
        )
        return self.rho * (kinetic + potential)

    def _kinetic(self, fn: NDArray[np.float64], fp: NDArray[np.float64]) -> float:
        """``(1/2) ||delta_t- f^n||^2`` on the interior (the clamped ends have zero velocity)."""
        dt_f = (fn[1:-1] - fp[1:-1]) / self.k
        return 0.5 * self.h * float(np.dot(dt_f, dt_f))

    def _potential(
        self, fn: NDArray[np.float64], fp: NDArray[np.float64], op: sparse.csr_matrix
    ) -> float:
        """Model #3's theta-weighted cross-time linear potential for one field."""
        un = fn[1:-1]
        up = fp[1:-1]
        p_nn = self._P(op, un, un)
        p_pp = self._P(op, up, up)
        p_np = self._P(op, un, up)
        return 0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np

    def _P(
        self, op: sparse.csr_matrix, f: NDArray[np.float64], g: NDArray[np.float64]
    ) -> float:
        """Potential bilinear form ``P(f,g) = <-L f, g> = h (-L f) . g`` (interior vectors)."""
        return -self.h * float(np.dot(op @ f, g))

    def _apply_full(
        self, op: sparse.csr_matrix, f_full: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``L f`` returned on the full grid (zeros at the clamped boundary nodes)."""
        out = np.zeros_like(f_full)
        out[1:-1] = op @ f_full[1:-1]
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
