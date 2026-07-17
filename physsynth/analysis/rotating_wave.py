"""Rotating-wave relative equilibrium of the geometrically-exact string (model #10, Tier B).

**The one exact frequency oracle model #10 has**, and the only escape from its Tier C "measure the
residual, do not promise cents" taxonomy. Where model #9's tension is a spatial *scalar* -- which is
what buys it an exact elliptic Duffing oracle -- model #10 promotes the tension to a **field**, and
every closed form degrades to a limit. Every one but this one.

**The ansatz.** A helix spinning rigidly about the string axis::

    u(x,t) = phi(x) cos(Omega t),    w(x,t) = phi(x) sin(Omega t),    v(x,t) = psi(x)

solves the **full** nonlinear PDE exactly -- no truncation, no small-slope assumption -- because the
transverse amplitude ``r^2 = u_x^2 + w_x^2 = phi'(x)^2 (cos^2 + sin^2) = phi'(x)^2`` is
**time-independent**. The potential sees the two polarizations *only* through ``r^2``, so the whole
nonlinearity goes static: the local stretch ``Lambda``, the tension field, and the longitudinal
forcing are all frozen. The string is bent into a fixed helical shape and spun.

That is also **exactly why** :attr:`~physsynth.core.string_geometric.GeometricString.\
longitudinal_energy` has nothing to do: a circular polarization does not pump the longitudinal field
at all, where a planar one pumps it at ``2 Omega`` (phantom partials). This module is what sharpens
that claim -- Tier A/3's discriminator -- from *orders of magnitude apart* to **bit-zero
longitudinal kinetic energy**.

**What it is an oracle FOR, stated honestly.** Two different claims, and blurring them is the trap:

1. **Scheme-consistency (exact, ~1e-15).** Seeded from a converged BVP the *discrete* scheme rotates
   rigidly forever. This certifies that the discrete-gradient scheme admits an exact discrete
   relative equilibrium and that this solver finds it. It is **not** a claim about physics -- the
   scheme is being checked against its own equations.
2. **Physics: the mechanism behind Tier C/8 (a limit).** The Kirchhoff-Carrier prediction
   ``Omega = sqrt(omega0^2 + eps R^2)`` (:func:`~physsynth.analysis.duffing.kc_mode_coefficients`,
   with ``EA_#9 <-> a = EA - T0``, and ``eps R^2`` unhalved because a circular mode's stretch is
   **2x** a planar mode's time-average) assumes ``phi`` is a *sine*. It is not. A rigid helix is
   stretched **non-uniformly** -- most near the nodes, where ``phi'`` is largest -- so ``T(x)`` is
   non-uniform and the true ``phi`` is a deformed sine. :attr:`RotatingWave.shape_residual` measures
   that deformation, and it *is* the reason the KC frequency is wrong. The residual gets a
   **mechanism**, not just an amplitude scaling.

**The equations.** With ``q = (u_x, w_x, v_x)``, ``a = EA - T0``, and the nonlinear excess
``V_nl = a[r^2/2 + 1 + v_x - Lambda]``, substituting the ansatz into the continuum PDE gives::

    -rho Omega^2 phi = ( T(x) phi' )' ,     T(x) = T0 + a chi(x),   chi = 1 - 1/Lambda
    d/dx [ dV/dv_x ] = 0    <=>    EA psi' + a (1 - (1 + psi')/Lambda) = const

the second being the plan's ``dV/dv_x = const``: the longitudinal field is in **static equilibrium**
against a frozen transverse load. On the grid the discrete divergence ``Gm`` has exactly the
constants in its kernel, so the discrete second equation reproduces the continuum statement *as a
statement*, not merely to O(h^2).

**The two time-discrete factors -- the crux of this module.** The oracle would be worthless at 1e-15
if it solved the continuum-time BVP, because the scheme does not. Substituting the *discrete*
rotating wave ``u_j^n = phi_j cos(Omega n k)`` into the theta-scheme gives two factors, and a naive
port drops the second:

- ``delta_tt -> -s`` with ``s = (4/k^2) sin^2(Omega k / 2)`` -- and the theta-average
  ``theta u+ + (1-2theta) u^n + theta u-`` contributes ``(1 - theta k^2 s)``. Familiar from
  :func:`~physsynth.analysis.modal.discrete_stiff_mode_frequency`.
- The discrete gradient pairs ``q^{n+1}`` against ``q^{n-1}`` -- it spans **2k**, not ``k`` -- so
  its midpoint ``qbar = (q+ + q-)/2`` picks up ``cos(Omega k)`` on the *transverse* strains:
  ``ubar_x = phi' cos(Omega n k) cos(Omega k)``. Hence the DG force carries a factor
  ``cos(Omega k) = 1 - k^2 s / 2``.

Both factors are **linear in ``s``**, which is why ``s`` (not ``Omega``) is the natural unknown. The
longitudinal row carries **neither**: ``psi`` is static, so its theta-average is ``psi`` for any
``theta``, and its DG component is time-independent. Pass ``time_discrete=False`` to drop both and
get the semi-discrete (continuous-time) BVP -- the difference between the two is the scheme's
temporal dispersion, which is what must be kept *out* of the Tier C/8 comparison.

**The Jacobian is not symmetric, and that is not a bug.** The reduced (phi, psi) system looks
variational -- its cell blocks are the Hessian of ``V_nl`` on the **planar** strain slice
``(p, 0, z)`` (see :func:`planar_hessian_cells`), which is symmetric. But ``cos(Omega k)``
multiplies the transverse row *only*, so ``dF_phi/dpsi = cos(Omega k) dF_psi/dphi``: symmetric
only as ``k -> 0``. Solved with ``splu``, exactly as the core solves its own non-symmetric DG
Jacobian.

**Degeneracy required.** The rotating wave needs ``kappa_u == kappa_w``: the ``u`` and ``w`` rows
must be the *same* equation for one ``phi`` to serve both. On a non-degenerate string
(:attr:`~physsynth.core.string_geometric.GeometricString.is_degenerate` false -- the configuration
that *whirls*) no rigid rotation exists. This module takes a single ``kappa`` for that reason.

Pure NumPy/SciPy; independent of the core (an oracle must not depend on the code it validates), so
the operators are rebuilt here rather than imported. ``theta`` is required rather than defaulted --
the frequency depends on it, so the caller must pass the resonator's own, as
:func:`~physsynth.analysis.modal.discrete_stiff_mode_frequency` also demands.

See ``docs/dev/geometrically-exact-string-plan.md`` (Tier B, decision #5).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

__all__ = [
    "RotatingWave",
    "solve_rotating_wave",
    "rotating_wave_history",
    "planar_hessian_cells",
    "kc_circular_frequency",
]

CONTINUATION_STEPS_DEFAULT = 8
"""Amplitude-continuation steps from the linear mode up to the requested amplitude.

The ``R -> 0`` seed (a sine at the theta-scheme's linear frequency) is exact only in the limit, and
the mode shape deforms as ``R`` grows. Stepping the amplitude and reusing each solution as the next
seed is what keeps Newton in its basin at amplitudes where the deformation is large.
"""

NEWTON_TOL_DEFAULT = 1e-14
"""**Relative** convergence bar on the Newton *step* (see :func:`solve_rotating_wave`).

Relative, and on the step rather than the residual, because the residual's three blocks carry
*different units* (transverse force density, longitudinal force density, and a displacement), so a
single max-norm bar on it would be a unit-mixing accident. The step is compared against the
solution's own scale, which is unit-free per block and is what Newton's quadratic convergence makes
a proxy for the error.
"""

NEWTON_MAXITER_DEFAULT = 50
"""Cap on Newton iterations per continuation step. Reaching it warns; it never silently renders."""


class RotatingWave(NamedTuple):
    """A converged rotating-wave relative equilibrium: the helix, its stretch, its frequency."""

    phi: NDArray[np.float64]
    """Transverse profile on the **full grid** ``(N+1,)`` (m), zero at both clamped ends.

    The helix is ``u = phi cos(Omega t)``, ``w = phi sin(Omega t)``. A *deformed* sine -- see
    :attr:`shape_residual`.
    """
    psi: NDArray[np.float64]
    """Longitudinal profile on the **full grid** ``(N+1,)`` (m), zero at both clamped ends.

    **Static, and nonzero.** The helix holds a fixed longitudinal stretch against the frozen
    transverse load. Do not test the rotating wave by asserting ``v == 0``; assert that ``v`` does
    not *move* (zero longitudinal kinetic energy).
    """
    Omega: float
    """Angular frequency (rad/s) of the rotation."""
    frequency: float
    """``Omega / (2 pi)`` (Hz)."""
    s: float
    """The scheme's ``s = (4/k^2) sin^2(Omega k / 2)`` (rad^2/s^2) -- the unknown solved for.

    Equals ``Omega^2`` exactly when ``time_discrete`` is false. Both time-discrete factors are
    linear in this variable, which is why it, and not ``Omega``, is the natural unknown.
    """
    amplitude: float
    """The imposed modal amplitude ``R`` (m) -- ``phi``'s projection onto ``sin(m pi x / L)``."""
    mode: int
    """The mode number ``m`` continued up from."""
    stretch_ratio: NDArray[np.float64]
    """Per-cell ``Lambda`` ``(N,)`` -- the frozen stretch field. ``1`` at rest."""
    tension: NDArray[np.float64]
    """Per-cell axial tension ``EA Lambda - a`` (N) ``(N,)`` -- **non-uniform**, which is the point.

    A rigid helix is stretched most near the nodes, where ``phi'`` is largest. That non-uniformity
    is exactly what a Kirchhoff-Carrier tension (a single scalar) cannot represent, and it is the
    mechanism behind :attr:`shape_residual` and Tier C/8's breakage.
    """
    shape_residual: float
    """``max|phi - R sin(m pi x / L)| / max|phi|`` -- the deviation from a sine.

    **The physics payoff of this module.** ``-> 0`` with amplitude; grows as the helix stretches
    non-uniformly. This is *why* :func:`kc_circular_frequency` is wrong, and reporting it turns Tier
    C/8's frequency residual from a bare number into a mechanism.
    """
    iterations: int
    """Total Newton iterations summed over the continuation."""
    converged: bool
    """Whether every continuation step converged. Never render a false one as physics."""
    time_discrete: bool
    """Whether the scheme's ``(1 - theta k^2 s)`` and ``cos(Omega k)`` factors were included."""


def _operators(N: int, h: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """``(D2, Gp, Gm)``: the Dirichlet second difference and the SBP node<->cell adjoint pair.

    Rebuilt here rather than imported from ``core.operators`` -- an oracle must not depend on the
    code it validates, and a silently divergent discretisation is exactly the kind of error the
    seeded-rotation test exists to catch. ``Gm @ Gp == D2`` exactly, and ``Gm``'s kernel is the
    constants, which is what makes the discrete ``dV/dv_x = const`` statement exact.
    """
    n_int = N - 1
    main = np.full(n_int, -2.0 / (h * h))
    off = np.full(n_int - 1, 1.0 / (h * h))
    d2 = sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    inv_h = 1.0 / h
    gp = sparse.diags(
        [np.full(n_int, inv_h), np.full(n_int, -inv_h)],
        offsets=[0, -1],
        shape=(N, n_int),
        format="csr",
    )
    return d2, gp.tocsr(), (-gp.T).tocsr()


def _stretch(p: NDArray[np.float64], z: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    """``(Lambda, chi, d)`` per cell from planar strains ``p = phi'``, ``z = psi'``.

    ``chi = 1 - 1/Lambda`` and ``d = Lambda - 1 - z``, both through the **cancellation-free**
    rearrangements the core uses (``core.string_geometric._stretch_terms``)::

        Lambda - 1     = (z (2 + z) + p^2) / (Lambda + 1)
        Lambda - 1 - z = p^2 / (Lambda + 1 + z)

    This is not tidiness and not duplicated code for its own sake: both are ``O(strain)``
    quantities assembled from ``O(1)`` ones, and musical strings run at strain ``1e-4 .. 1e-3``.
    Evaluating them literally would converge this BVP to a root of a *slightly different* equation
    than the one the core steps, and the seeded helix would then shed that difference into the
    longitudinal field -- the 1e-15 gate would fail for a reason unrelated to the physics.
    """
    lam = np.sqrt((1.0 + z) ** 2 + p**2)
    lam_m1 = (z * (2.0 + z) + p**2) / (lam + 1.0)
    chi = lam_m1 / lam
    denom = lam + 1.0 + z
    safe = denom > 1.0  # physical cells sit at denom ~ 2; only an inverted element nears 0
    d = np.where(safe, p**2 / np.where(safe, denom, 1.0), lam - 1.0 - z)
    return lam, chi, d


def planar_hessian_cells(
    p: NDArray[np.float64], z: NDArray[np.float64], a: float
) -> tuple[NDArray[np.float64], ...]:
    """``(H_pp, H_pz, H_zz)``: the Hessian of ``V_nl`` on the **planar** strain slice, per cell.

    The rotating wave's linearisation. Although the helix is fully three-dimensional, its reduced
    ``(phi, psi)`` system linearises on the *planar* slice ``q = (p, 0, z)``: ``V_nl`` sees the
    polarizations only through ``r^2 = p^2 (cos^2 + sin^2) = p^2``, which is the same value the
    planar strain ``(p, 0, z)`` produces. So the 2x2 reduced block is the full 3x3 Hessian with
    the ``w`` row and column struck out::

        H = a [ chi I3 + m m^T / Lambda^3 - e3 e3^T ] ,   m = (u_x, w_x, 1 + v_x)

    Three entries survive, and the third simplifies exactly -- ``(1+z)^2 - Lambda^2 = -p^2``::

        H_pp = a (chi + p^2 / Lambda^3)
        H_pz = a p (1 + z) / Lambda^3
        H_zz = a ((1+z)^2 / Lambda^3 - 1/Lambda) = **-a p^2 / Lambda^3**

    That last identity is worth keeping: written literally it is a difference of ``O(1)`` terms
    (catastrophic at musical strain), and written as ``-a p^2 / Lambda^3`` it has no cancellation at
    all. ``H_zz`` is *negative* for a hardening string -- the nonlinear excess softens the
    longitudinal direction -- but the full longitudinal diagonal adds ``EA`` and stays positive.

    Exposed publicly because it cross-checks against the core: ``H/2`` is exactly
    ``GeometricString._dg_jacobian(q, q)``, whose factor of ``1/2`` is ``d(qbar)/d(q+)``.
    """
    lam, chi, _ = _stretch(p, z)
    inv_lam3 = 1.0 / lam**3
    return (
        a * (chi + p**2 * inv_lam3),
        a * p * (1.0 + z) * inv_lam3,
        -a * p**2 * inv_lam3,
    )


def kc_circular_frequency(*, omega0_sq: float, eps: float, amplitude: float) -> float:
    """Kirchhoff-Carrier's ``Omega = sqrt(omega0^2 + eps R^2)`` for a **circular** mode (rad/s).

    A **limit** oracle (Tier C/8), not an exact one -- it is here to be *measured against*
    :func:`solve_rotating_wave`, and it breaks by mode-shape deformation
    (:attr:`RotatingWave.shape_residual`).

    Take ``(omega0_sq, eps)`` from :func:`~physsynth.analysis.duffing.kc_mode_coefficients` with
    ``EA = a = EA_#10 - T0`` (mind the identification) and the **discrete**
    ``p2 = spatial_eigenvalue_p2(...)``, so the only thing left to explain is the shape.

    **Note the missing half.** For a *planar* mode the stretch ``I ~ cos^2(Omega t)`` averages to
    ``R^2 p^2 L / 4``; a circular mode's is ``R^2 p^2 L / 2`` -- **exactly twice**, and constant
    rather than pulsing, because ``cos^2 + sin^2 = 1``. So the tension shift is the full ``eps R^2``
    with no factor of ``1/2``, and this is *not* the same expression as the planar Duffing
    frequency (:func:`~physsynth.analysis.duffing.duffing_frequency`) evaluated at the same ``R``.
    """
    val = omega0_sq + eps * amplitude**2
    if val <= 0:
        raise ValueError(f"omega0^2 + eps R^2 must be positive, got {val}.")
    return float(np.sqrt(val))


def rotating_wave_history(
    wave: RotatingWave, *, fs: float
) -> tuple[NDArray[np.float64], ...]:
    """The **exact** two-level history ``(u0, w0, v0, u_prev, w_prev, v_prev)`` seeding the helix.

    Assign these straight onto the resonator's ``u/w/v`` and ``u_prev/w_prev/v_prev`` --
    **do not route them through**
    :meth:`~physsynth.core.string_geometric.GeometricString.set_state`. That method builds
    ``y^{-1}`` from a second-order Taylor start, which is *consistent* but not *exact*: it would
    seed an ``O(k^3)`` history error, the helix would shed it into the longitudinal field, and
    the rigid rotation would hold to ``O(k^3)`` instead of roundoff. The whole 1e-15 claim lives
    in this function.

    The exact history is just the ansatz evaluated at ``n = 0`` and ``n = -1``::

        u^0 = phi,              w^0 = 0,                v^0 = psi
        u^-1 = phi cos(Omega k), w^-1 = -phi sin(Omega k), v^-1 = psi

    ``fs`` must be the resonator's own (``Omega`` was solved for at that ``k``, if
    ``time_discrete``).
    """
    if fs <= 0:
        raise ValueError("fs must be positive.")
    k = 1.0 / fs
    cos_k = float(np.cos(wave.Omega * k))
    sin_k = float(np.sin(wave.Omega * k))
    zero = np.zeros_like(wave.phi)
    return (
        wave.phi.copy(),
        zero.copy(),
        wave.psi.copy(),
        wave.phi * cos_k,
        -wave.phi * sin_k,
        wave.psi.copy(),
    )


def solve_rotating_wave(
    *,
    L: float,
    T: float,
    rho: float,
    EA: float,
    fs: float,
    N: int,
    theta: float,
    amplitude: float,
    mode: int = 1,
    kappa: float = 0.0,
    time_discrete: bool = True,
    continuation_steps: int = CONTINUATION_STEPS_DEFAULT,
    tol: float = NEWTON_TOL_DEFAULT,
    maxiter: int = NEWTON_MAXITER_DEFAULT,
) -> RotatingWave:
    """Solve the rotating-wave BVP for ``(phi, psi, Omega)`` by amplitude continuation.

    Parameters mirror :class:`~physsynth.core.string_geometric.GeometricString`'s, and must match
    the resonator the result will seed -- ``theta``, ``fs`` and ``N`` included, since the frequency
    depends on all three.

    Parameters
    ----------
    L, T, rho, EA, fs, N, theta, kappa :
        The string, exactly as the resonator was built. ``kappa`` applies to **both** polarizations:
        a rotating wave exists only on a degenerate (``kappa_u == kappa_w``) string.
    amplitude :
        The modal amplitude ``R`` (m): ``phi``'s projection onto ``sin(m pi x / L)``, normalised so
        that ``R`` means what it means in
        :func:`~physsynth.analysis.duffing.kc_mode_coefficients`. Chosen over ``max|phi|`` precisely
        so the Tier C/8 comparison against :func:`kc_circular_frequency` is amplitude-consistent.
    mode :
        Mode number ``m`` to continue up from (the linear sine the helix deforms away from).
    time_discrete :
        Include the scheme's ``(1 - theta k^2 s)`` and ``cos(Omega k)`` factors (default). Required
        for the seeded helix to rotate rigidly to machine precision. Pass ``False`` for the
        semi-discrete BVP, whose ``Omega`` carries no temporal dispersion -- the honest comparison
        for Tier C/8.
    continuation_steps, tol, maxiter :
        Continuation and Newton controls. ``tol`` is a **relative** bar on the Newton step.

    Returns
    -------
    RotatingWave
        The converged helix. Check :attr:`RotatingWave.converged`.

    Raises
    ------
    ValueError
        On non-physical parameters, ``EA < T`` (softening: outside this oracle's scope), a ``mode``
        outside ``1 .. N-1``, or a rotation past Nyquist (``Omega k > pi``, where the discrete
        ``s`` no longer maps back to a frequency).
    """
    if min(L, T, rho, fs, EA) <= 0:
        raise ValueError("L, T, rho, fs, EA must all be positive.")
    if N < 2:
        raise ValueError("N must be >= 2.")
    if not 1 <= mode <= N - 1:
        raise ValueError(f"mode must be in 1 .. {N - 1}, got {mode}.")
    if kappa < 0:
        raise ValueError("kappa must be >= 0.")
    if not (0.0 < theta <= 1.0):
        raise ValueError(f"theta must be in (0, 1], got {theta}.")
    if EA < T:
        raise ValueError(
            f"EA ({EA}) < T ({T}) is a softening string, outside this oracle's scope: the "
            "continuation seed is the hardening branch. The resonator itself permits it via "
            "allow_softening."
        )
    if continuation_steps < 1:
        raise ValueError("continuation_steps must be >= 1.")
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")

    h = L / N
    k = 1.0 / fs
    c_sq = T / rho
    a = EA - T
    d2, gp, gm = _operators(N, h)
    op_u = (c_sq * d2 - (kappa**2) * (d2 @ d2)).tocsr() if kappa else (c_sq * d2).tocsr()
    op_v = ((EA / rho) * d2).tocsr()
    ident = sparse.identity(N - 1, format="csr")

    x_int = np.linspace(0.0, L, N + 1)[1:-1]
    shape = np.sin(mode * np.pi * x_int / L)
    # h * sum(sin^2) == L/2 exactly on the interior grid -- the identity that makes this projection
    # the same amplitude convention as duffing.kc_mode_stretch.
    proj = (2.0 / L) * h * shape

    # Linear seed: the theta-scheme's own discrete dispersion. Algebraically identical to
    # modal.discrete_stiff_mode_frequency's s = Q k^2 / (4 + 4 theta Q k^2), which is the R -> 0
    # gate on this whole module.
    p2 = (4.0 / (h * h)) * np.sin(mode * np.pi / (2 * N)) ** 2
    q_lin = c_sq * p2 + (kappa**2) * p2**2
    s = q_lin / (1.0 + theta * k * k * q_lin) if time_discrete else q_lin

    phi = np.zeros(N - 1)
    psi = np.zeros(N - 1)
    total_iters = 0
    converged = True

    for step in range(1, continuation_steps + 1):
        target = amplitude * step / continuation_steps
        if step == 1:
            phi = target * shape  # the linear mode: exact as R -> 0
        else:
            # Extrapolate the previous solution: phi scales linearly in R, psi quadratically (it is
            # driven by the r^2 v_x / 2 term). Cheap, and it keeps Newton's first step small where
            # the shape deformation is strongest.
            ratio = step / (step - 1)
            phi = phi * ratio
            psi = psi * ratio**2
        phi, psi, s, iters, ok = _newton(
            phi, psi, s, target,
            op_u=op_u, op_v=op_v, gp=gp, gm=gm, ident=ident, proj=proj,
            rho=rho, a=a, k=k, theta=theta,
            time_discrete=time_discrete, tol=tol, maxiter=maxiter,
        )
        total_iters += iters
        if not ok:
            converged = False
            warnings.warn(
                f"Rotating-wave BVP did not converge at continuation step {step}/"
                f"{continuation_steps} (amplitude {target:.3e} m) in {maxiter} Newton iterations. "
                f"The helix is not a relative equilibrium; seeding a resonator with it will NOT "
                f"rotate rigidly. Raise continuation_steps, or lower the amplitude.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

    if time_discrete:
        arg = 0.5 * k * np.sqrt(max(s, 0.0))
        if arg >= 1.0:
            raise ValueError(
                f"The converged rotation is at or past Nyquist (s = {s:.4g} gives "
                f"sin(Omega k / 2) = {arg:.4g} >= 1): the discrete frequency does not map back to "
                f"an Omega. Raise fs."
            )
        omega = float(2.0 * np.arcsin(arg) / k)
    else:
        omega = float(np.sqrt(max(s, 0.0)))

    lam, _, _ = _stretch(gp @ phi, gp @ psi)
    phi_full = np.zeros(N + 1)
    psi_full = np.zeros(N + 1)
    phi_full[1:-1] = phi
    psi_full[1:-1] = psi
    peak = float(np.max(np.abs(phi)))
    residual = float(np.max(np.abs(phi - amplitude * shape)) / peak) if peak > 0 else 0.0

    return RotatingWave(
        phi=phi_full,
        psi=psi_full,
        Omega=omega,
        frequency=omega / (2.0 * np.pi),
        s=float(s),
        amplitude=float(amplitude),
        mode=int(mode),
        stretch_ratio=lam,
        tension=EA * lam - a,
        shape_residual=residual,
        iterations=total_iters,
        converged=converged,
        time_discrete=bool(time_discrete),
    )


def _residual(
    phi: NDArray[np.float64],
    psi: NDArray[np.float64],
    s: float,
    target: float,
    *,
    op_u, op_v, gp, gm, proj, rho, a, k, theta, time_discrete,
) -> NDArray[np.float64]:
    """The BVP residual ``[F_phi; F_psi; F_norm]``.

    ``F_phi = rho (1 - theta k^2 s) L_u phi + cos(Omega k) Gm[a chi phi'] + rho s phi``
    ``F_psi = rho L_v psi + Gm[a (Lambda - 1 - psi')/Lambda]``      (no time factors: psi is static)
    ``F_norm = <phi, sin_m> (2/L) h - R``
    """
    p = gp @ phi
    z = gp @ psi
    lam, chi, d = _stretch(p, z)
    fac_lin = 1.0 - theta * k * k * s if time_discrete else 1.0
    fac_dg = 1.0 - 0.5 * k * k * s if time_discrete else 1.0
    f_phi = rho * fac_lin * (op_u @ phi) + fac_dg * (gm @ (a * chi * p)) + rho * s * phi
    f_psi = rho * (op_v @ psi) + gm @ (a * d / lam)
    return np.concatenate([f_phi, f_psi, [float(proj @ phi) - target]])


def _jacobian(
    phi: NDArray[np.float64],
    psi: NDArray[np.float64],
    s: float,
    *,
    op_u, op_v, gp, gm, ident, proj, rho, a, k, theta, time_discrete,
) -> sparse.csc_matrix:
    """The exact (non-symmetric) Jacobian of :func:`_residual`.

    Non-symmetric because ``cos(Omega k)`` multiplies the transverse row only, so
    ``dF_phi/dpsi = cos(Omega k) dF_psi/dphi``. The cell blocks are
    :func:`planar_hessian_cells`; the ``d/ds`` column is where a dropped factor hides, which is why
    the tests finite-difference-check this against :func:`_residual`.
    """
    p = gp @ phi
    z = gp @ psi
    chi_pp, chi_pz, chi_zz = planar_hessian_cells(p, z, a)
    _, chi, _ = _stretch(p, z)
    fac_lin = 1.0 - theta * k * k * s if time_discrete else 1.0
    fac_dg = 1.0 - 0.5 * k * k * s if time_discrete else 1.0

    j_pp = rho * fac_lin * op_u + fac_dg * (gm @ sparse.diags(chi_pp) @ gp) + rho * s * ident
    j_pz = fac_dg * (gm @ sparse.diags(chi_pz) @ gp)
    j_zp = gm @ sparse.diags(chi_pz) @ gp
    j_zz = rho * op_v + gm @ sparse.diags(chi_zz) @ gp

    if time_discrete:
        d_phi_ds = (
            -rho * theta * k * k * (op_u @ phi) - 0.5 * k * k * (gm @ (a * chi * p)) + rho * phi
        )
    else:
        d_phi_ds = rho * phi
    zero_col = np.zeros((len(psi), 1))
    return sparse.bmat(
        [
            [j_pp, j_pz, d_phi_ds.reshape(-1, 1)],
            [j_zp, j_zz, zero_col],
            [sparse.csr_matrix(proj.reshape(1, -1)), None, None],
        ],
        format="csc",
    )


def _newton(
    phi: NDArray[np.float64],
    psi: NDArray[np.float64],
    s: float,
    target: float,
    *,
    op_u, op_v, gp, gm, ident, proj, rho, a, k, theta, time_discrete, tol, maxiter,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int, bool]:
    """Newton on ``[phi; psi; s]``; returns ``(phi, psi, s, iterations, converged)``.

    The convergence bar is on the **relative step**: the residual's blocks carry different units
    (two force densities and a displacement), so a max-norm bar on it would mix them. Displacement
    steps are measured against ``max|phi|`` -- ``psi`` included, since it can legitimately be ~0 at
    small amplitude and would otherwise divide by nothing.
    """
    n_int = len(phi)
    for it in range(1, maxiter + 1):
        r = _residual(
            phi, psi, s, target,
            op_u=op_u, op_v=op_v, gp=gp, gm=gm, proj=proj,
            rho=rho, a=a, k=k, theta=theta, time_discrete=time_discrete,
        )
        jac = _jacobian(
            phi, psi, s,
            op_u=op_u, op_v=op_v, gp=gp, gm=gm, ident=ident, proj=proj,
            rho=rho, a=a, k=k, theta=theta, time_discrete=time_discrete,
        )
        delta = splu(jac).solve(-r)
        phi = phi + delta[:n_int]
        psi = psi + delta[n_int : 2 * n_int]
        s = s + float(delta[-1])
        scale = max(float(np.max(np.abs(phi))), np.finfo(float).tiny)
        err = max(
            float(np.max(np.abs(delta[: 2 * n_int]))) / scale,
            abs(float(delta[-1])) / max(abs(s), np.finfo(float).tiny),
        )
        if err <= tol:
            return phi, psi, s, it, True
    return phi, psi, s, maxiter, False
