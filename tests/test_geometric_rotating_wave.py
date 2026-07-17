"""Model #10, Tier B — the rotating-wave relative equilibrium (test 16).

The helix: ``u = phi cos(Omega t)``, ``w = phi sin(Omega t)``, ``v = psi``. Because ``V`` sees the
polarizations only through ``r^2 = u_x^2 + w_x^2 = phi'^2 (cos^2 + sin^2)``, a circular polarization
freezes the whole nonlinearity — the stretch, the tension field and the longitudinal forcing all go
*static*. The string is bent into a fixed helical shape and spun, forever.

This closes model #10's oracle taxonomy. It is the model's **only** exact frequency oracle, and it
buys two things the rest of the batch could not:

- **Tier A/3 sharpens from "orders apart" to "bit-zero."** Batch 2 could only say a circular
  polarization radiates ~1e5x less longitudinal energy than a planar one. Seeded from a converged
  BVP the longitudinal field does not move **at all**: ``long_kin/E ~ 1e-26``, against ``6e-3``
  planar. Twenty-three orders, not five.
- **Tier C/8 gets a mechanism, not just a scaling.** Batch 2 could only report that
  ``Omega = sqrt(omega0^2 + eps R^2)`` breaks. Here the converged ``phi`` **is** the non-sine mode
  shape that explains it, and the frequency error tracks the shape deformation ~1:1.

**Two claims, and blurring them is the trap** (the module docstring says this; the tests are
partitioned by it):

1. Scheme-consistency — the seeded helix rotates rigidly to machine precision. Exact, but it
   certifies the scheme against *its own* equations, not against physics.
2. Physics — the KC limit oracle's error, and its mechanism. That one is a *limit*, and is measured
   rather than promised.
"""

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    geometric_mode_ic,
    geometric_rotating_wave,
    longitudinal_kinetic_energy,
    make_geometric_string,
    seed_rotating_wave,
)

from physsynth.analysis import rotating_wave as rw
from physsynth.analysis.damping import spatial_eigenvalue_p2
from physsynth.analysis.duffing import kc_mode_coefficients
from physsynth.analysis.modal import discrete_stiff_mode_frequency
from physsynth.analysis.rotating_wave import (
    kc_circular_frequency,
    planar_hessian_cells,
    rotating_wave_history,
    solve_rotating_wave,
)

AMP = 5e-3
"""Test amplitude (m). Large enough that the shape deformation is ~1e-5 (a nonlinearity bug cannot
hide), small enough that the continuation stays in Newton's basin at the default 8 steps."""


def _string(**kw):
    """A degenerate, flexible, lossless string — the cleanest configuration for the helix.

    ``kappa = 0`` because bending is irrelevant to the relative equilibrium and only adds a term to
    read past; ``lam_long`` is the helper's default 0.5, so any failure here is unambiguously the
    BVP and not the longitudinal field being under-resolved (the trap of this model).
    """
    kw.setdefault("N", 32)
    kw.setdefault("EA", EA_DEFAULT)
    kw.setdefault("kappa", 0.0)
    return make_geometric_string(**kw)


def _spin(s, n_steps):
    """Step ``n_steps``, returning ``(max long_kin, max |r - r0|/max r0, max |v - v0|)``."""
    r0 = np.hypot(s.u, s.w)
    v0 = s.v.copy()
    long_kin = 0.0
    r_dev = 0.0
    v_dev = 0.0
    for _ in range(n_steps):
        s.step()
        long_kin = max(long_kin, longitudinal_kinetic_energy(s))
        r_dev = max(r_dev, float(np.max(np.abs(np.hypot(s.u, s.w) - r0))))
        v_dev = max(v_dev, float(np.max(np.abs(s.v - v0))))
    return long_kin, r_dev / float(np.max(np.abs(r0))), v_dev


# -- claim 1: scheme-consistency (exact) ------------------------------------------------


def test_rotating_wave_at_zero_amplitude_is_the_linear_modal_oracle():
    """``R -> 0`` reproduces :func:`discrete_stiff_mode_frequency` — the free gate on the module.

    The whole BVP collapses at zero amplitude: ``chi -> 0``, the nonlinear force vanishes, and the
    transverse row becomes the theta-scheme's own discrete dispersion
    ``s = Q / (1 + theta k^2 Q)``. That is *algebraically identical* to the model #2/#3 oracle's
    ``s = Q k^2 / (4 + 4 theta Q k^2)``, which was written years of models ago and knows nothing
    about helices.

    So this pins the two time-discrete factors — the ``(1 - theta k^2 s)`` one, at least — against
    independent code. A module that dropped it would still converge, to a subtly wrong frequency,
    and every downstream test would inherit the error. Checked across modes *and* stiffness, since
    ``kappa`` enters ``Q``.
    """
    for kappa in (0.0, 2.0):
        s = _string(kappa=kappa)
        for m in (1, 2, 5):
            wave = geometric_rotating_wave(s, 1e-9, m)
            want = discrete_stiff_mode_frequency(s.c, s.L, s.N, kappa, s.k, m, s.theta)
            assert wave.converged
            assert abs(wave.frequency - want) / want < 1e-12


def test_seeded_helix_rotates_rigidly_and_the_longitudinal_field_never_moves():
    """**The money test.** One full revolution: ``long_kin/E ~ 1e-26``, and the radius is constant.

    This is Tier A/3's sharpening. Batch 2's phantom tests could only say a circular polarization
    pumps the longitudinal field *far less* than a planar one; from a converged relative equilibrium
    it does not pump it **at all**.

    **Note what is asserted and what is not.** ``psi`` is emphatically *nonzero* — the helix holds a
    static longitudinal stretch against its own frozen transverse load — so ``v == 0`` is the wrong
    test and would assert the physics away. What is bit-zero is the longitudinal *motion*.

    The residual ``long_kin`` is not scheme error at all; it is accumulated roundoff over ~2800
    steps, and the negative controls below show every ingredient buying orders of magnitude.
    """
    s = _string()
    wave = geometric_rotating_wave(s, AMP)
    assert wave.converged
    seed_rotating_wave(s, wave)
    E0 = s.energy()

    long_kin, r_dev, _ = _spin(s, int(s.fs / wave.frequency))  # one full revolution

    assert long_kin / E0 < 1e-20  # measured 1.3e-26; set_state seeding gives 1e-16
    assert r_dev < 1e-9  # the helix is rigid: |(u,w)| is constant in time
    # ...and the static stretch it holds is real, not a rounding artifact.
    assert np.max(np.abs(wave.psi)) > 1e-7
    assert np.max(np.abs(s.v)) > 1e-7


def test_rotating_wave_conserves_energy():
    """The standing gate (CLAUDE.md): lossless drift < 1e-10, with the nonlinear fraction reported.

    Cheap here, but not redundant: the helix is a *steady* state, and a steady state is exactly
    where a conservation bug can hide behind nothing happening. The nonlinear-fraction assertion is
    model #6/#9's lesson — a nonlinearity bug hides at small amplitude, where the test merely
    re-runs the linear scheme.
    """
    s = _string()
    wave = geometric_rotating_wave(s, AMP)
    seed_rotating_wave(s, wave)
    E0 = s.energy()
    for _ in range(400):
        s.step()
    assert abs(s.energy() - E0) / E0 < 1e-10
    assert abs(s.nonlinear_energy()) / E0 > 1e-6  # the nonlinearity is actually engaged


def test_set_state_seeding_costs_ten_orders():
    """**The trap, pinned.** The same helix through ``set_state`` is ten orders worse.

    ``set_state``'s ``y^{-1}`` is a second-order Taylor start: *consistent*, but not *exact*. The
    helix sheds that ``O(k^3)`` history error straight into the longitudinal field. Measured at the
    same amplitude: exact history ``~1e-26``, ``set_state`` ``~1e-16``.

    Worth a test rather than a comment because ``set_state`` is the obvious thing to reach for, the
    resulting number (1e-16) still *looks* like machine precision, and the run still conserves
    energy and still rotates — it just quietly stops being a relative equilibrium. This is why
    :func:`~helpers.seed_rotating_wave` exists.
    """
    wave = geometric_rotating_wave(_string(), AMP)

    exact = _string()
    seed_rotating_wave(exact, wave)
    E_exact = exact.energy()
    lk_exact, _, _ = _spin(exact, 400)

    taylor = _string()
    zero = np.zeros(taylor.N + 1)
    taylor.set_state(wave.phi, zero, wave.psi, w_dot=wave.Omega * wave.phi)
    E_taylor = taylor.energy()
    lk_taylor, _, _ = _spin(taylor, 400)

    assert lk_taylor / E_taylor > 1e6 * (lk_exact / E_exact)


def test_a_sine_is_not_a_relative_equilibrium():
    """The BVP earns its keep: the linear mode shape at the KC frequency is **not** the helix.

    The negative control that proves the solve does something. Substituting the obvious guess — a
    sine at the Kirchhoff-Carrier circular frequency, with the BVP's own ``psi`` handed to it for
    free — still pumps the longitudinal field orders harder than the converged shape does. What is
    left is *exactly* the mode-shape deformation, and it is the same deformation
    :attr:`RotatingWave.shape_residual` reports and Tier C/8 pays for in frequency.
    """
    s_ref = _string()
    wave = geometric_rotating_wave(s_ref, AMP)
    p2 = spatial_eigenvalue_p2(s_ref.N, s_ref.h, 1)
    omega0_sq, eps = kc_mode_coefficients(
        c=s_ref.c, kappa=0.0, EA=s_ref.EA - s_ref.T, rho=s_ref.rho, p2=p2, L=s_ref.L
    )
    omega_kc = kc_circular_frequency(omega0_sq=omega0_sq, eps=eps, amplitude=AMP)

    exact = _string()
    seed_rotating_wave(exact, wave)
    E_exact = exact.energy()
    lk_exact, _, _ = _spin(exact, 400)

    sine = _string()
    shape = geometric_mode_ic(sine.N, 1, AMP)
    sine.u, sine.w, sine.v = shape.copy(), np.zeros_like(shape), wave.psi.copy()
    sine.u_prev = shape * np.cos(omega_kc * sine.k)
    sine.w_prev = -shape * np.sin(omega_kc * sine.k)
    sine.v_prev = wave.psi.copy()
    E_sine = sine.energy()
    lk_sine, _, _ = _spin(sine, 400)

    assert lk_sine / E_sine > 1e6 * (lk_exact / E_exact)


def test_circular_is_bit_zero_where_planar_is_percent_level():
    """Tier A/3, at full strength: the discriminator is **twenty-three orders**, not five.

    Same string, same amplitude, same mode — only the polarization differs. This is the claim model
    #9 structurally could not make (it has no ``v``), stated as sharply as it can be stated.
    """
    circ = _string()
    wave = geometric_rotating_wave(circ, AMP)
    seed_rotating_wave(circ, wave)
    E_circ = circ.energy()
    lk_circ, _, _ = _spin(circ, 400)

    planar = _string()
    planar.set_state(geometric_mode_ic(planar.N, 1, AMP))
    E_planar = planar.energy()
    lk_planar, _, _ = _spin(planar, 400)

    assert lk_circ / E_circ < 1e-20
    assert lk_planar / E_planar > 1e-4
    assert (lk_planar / E_planar) / (lk_circ / E_circ) > 1e15


def test_the_static_stretch_is_the_ingredient_batch_2_never_varied():
    """**Batch 2 predicted this batch would be "mostly solving for ``Omega``, not ``phi``." It had
    three unknowns and only ever varied two.**

    ``test_the_circular_residual_is_ellipticity_not_a_defect_of_the_scheme`` bounded ``phi``'s share
    of the circular residual against ``Omega``'s and concluded ``Omega`` was the lever. That
    conclusion is sound *on its own terms* — but every circular IC in batch 2 was
    ``set_state(shape, 0.0, w_dot=Omega*shape)``, i.e. ``v^0 = 0``. The helix's **static
    longitudinal stretch ``psi`` was pinned at zero in every run**, so it could not appear as a
    variable, and it is the biggest lever of the three.

    Isolated here (each ingredient wrong, the other two at their BVP values, ``long_kin/E`` after
    400 steps)::

        nothing wrong (full BVP)  1.3e-26
        psi = 0                   3.7e-03      <- 23 orders
        phi = sine                3.3e-11      <- 15 orders
        Omega = KC                1.6e-15      <- 11 orders

    The reason is not subtle once seen: a helix started with ``v = 0`` is not near its equilibrium
    at all — it is released into the static stretch it *should already be holding* and rings about
    it. That is a relaxation transient, not a phantom pump, which is exactly why batch 2's
    2f1-band metric could not see it and why its ``Omega``-vs-``phi`` bound was not wrong, just
    blind to a third axis.

    **What is deliberately NOT asserted here:** the ``phi`` vs ``Omega`` ordering. It *inverts*
    between the two metrics (on batch 2's 2f1 bridge-pump metric ``Omega`` costs ~69x more than
    ``phi``; on total longitudinal motion ``phi`` costs ~2e4x more than ``Omega``), and the obvious
    explanation — that shape error is broadband while ellipticity is a 2f1 pump — was **measured
    and refuted**: both put ~51-57 % of their longitudinal energy at 2f1. Both perturbations are
    also the same relative size (~1.2e-5). So the inversion is real, reproducible, and currently
    unexplained; it is recorded in the plan doc rather than asserted here.
    """
    ref = _string()
    wave = geometric_rotating_wave(ref, AMP)

    def seeded(psi):
        s = _string()
        s.u, s.w, s.v = wave.phi.copy(), np.zeros(s.N + 1), psi.copy()
        s.u_prev = wave.phi * np.cos(wave.Omega * s.k)
        s.w_prev = -wave.phi * np.sin(wave.Omega * s.k)
        s.v_prev = psi.copy()
        return s

    full = seeded(wave.psi)
    E_full = full.energy()
    lk_full, _, _ = _spin(full, 400)

    no_stretch = seeded(np.zeros(ref.N + 1))
    E_none = no_stretch.energy()
    lk_none, _, _ = _spin(no_stretch, 400)

    # psi is worth ~23 orders: everything else identical, only the static stretch removed.
    assert lk_none / E_none > 1e15 * (lk_full / E_full)
    assert lk_full / E_full < 1e-20


def test_helix_does_not_survive_on_a_non_degenerate_string():
    """No rigid rotation without ``kappa_u == kappa_w`` — the tie to batch 2's whirling.

    The helix needs one ``phi`` to serve *both* polarizations, so the ``u`` and ``w`` rows must be
    the same equation. Detune them and the relative equilibrium ceases to exist: the same seed
    pumps the longitudinal field ~14 orders harder.

    The two claims are the same fact from opposite sides. A degenerate string **cannot whirl** (a
    planar IC stays bit-exactly planar) but **can** rotate rigidly; a detuned one **can** whirl
    (batch 2's Mathieu tongue) but **cannot** rotate rigidly. The knob that unlocks one closes the
    other.
    """
    ref = _string()
    wave = geometric_rotating_wave(ref, AMP)

    degenerate = _string()
    assert degenerate.is_degenerate
    seed_rotating_wave(degenerate, wave)
    E_deg = degenerate.energy()
    lk_deg, _, _ = _spin(degenerate, 400)

    detuned = _string(kappa=0.0, kappa_w=0.3)
    assert not detuned.is_degenerate
    seed_rotating_wave(detuned, wave)
    E_det = detuned.energy()
    lk_det, _, _ = _spin(detuned, 400)

    assert lk_det / E_det > 1e6 * (lk_deg / E_deg)


# -- the solver's own internals ---------------------------------------------------------


def test_jacobian_matches_finite_differences():
    """The BVP Jacobian against central differences — including the ``d/ds`` column.

    Newton converges to a root of the *residual*, so a wrong Jacobian shows up as slow convergence
    rather than a wrong answer — which means it can be wrong for a long time without anything
    failing. The ``d/ds`` column is the one to watch: it carries the derivative of **both** time
    factors, and dropping either leaves a column that is merely *nearly* right.
    """
    s = _string()
    h, k = s.h, s.k
    d2, gp, gm = rw._operators(s.N, h)
    from scipy import sparse

    ident = sparse.identity(s.N - 1, format="csr")
    a = s.EA - s.T
    x_int = np.linspace(0.0, s.L, s.N + 1)[1:-1]
    shape = np.sin(np.pi * x_int / s.L)
    kw = dict(
        op_u=((s.T / s.rho) * d2).tocsr(), op_v=((s.EA / s.rho) * d2).tocsr(), gp=gp, gm=gm,
        proj=(2.0 / s.L) * h * shape, rho=s.rho, a=a, k=k, theta=s.theta,
    )
    rng = np.random.default_rng(0)
    phi = AMP * shape + 1e-4 * rng.standard_normal(s.N - 1)
    psi = 1e-5 * rng.standard_normal(s.N - 1)
    s_var = (s.c * np.pi) ** 2
    n = s.N - 1

    for time_discrete in (True, False):
        jac = rw._jacobian(phi, psi, s_var, ident=ident, time_discrete=time_discrete,
                           **kw).toarray()
        fd = np.zeros_like(jac)
        for j in range(2 * n + 1):
            args = []
            for sign in (+1, -1):
                ph, ps, sv = phi.copy(), psi.copy(), s_var
                if j < n:
                    eps = 1e-7 * max(abs(phi[j]), 1e-6)
                    ph[j] += sign * eps
                elif j < 2 * n:
                    eps = 1e-7 * max(abs(psi[j - n]), 1e-9)
                    ps[j - n] += sign * eps
                else:
                    eps = 1e-7 * s_var
                    sv = s_var + sign * eps
                args.append(rw._residual(ph, ps, sv, AMP, time_discrete=time_discrete, **kw))
            fd[:, j] = (args[0] - args[1]) / (2.0 * eps)
        assert np.max(np.abs(jac - fd)) / np.max(np.abs(fd)) < 1e-6


def test_jacobian_asymmetry_is_exactly_the_dg_time_factor():
    """``dF_phi/dpsi == cos(Omega k) dF_psi/dphi`` — the structural signature of the 2k-wide DG.

    The reduced system *looks* variational: its cell blocks are the Hessian of ``V_nl`` on the
    planar strain slice, which is symmetric. It is not, and the asymmetry is not roundoff — it is
    the discrete gradient spanning ``q^{n+1}`` to ``q^{n-1}``, which puts a ``cos(Omega k)`` on the
    transverse row and nothing on the longitudinal one.

    Assuming the symmetry (and reaching for a Cholesky, as the family's *linear* models do) would
    stall Newton against a Jacobian that is wrong by one part in 2e5 at these settings — small
    enough to look like a conditioning problem and not a bug. So: assert the exact relation, assert
    the plain symmetry **fails**, and assert it is restored as ``k -> 0``.
    """
    s = _string()
    from scipy import sparse

    d2, gp, gm = rw._operators(s.N, s.h)
    x_int = np.linspace(0.0, s.L, s.N + 1)[1:-1]
    shape = np.sin(np.pi * x_int / s.L)
    kw = dict(
        op_u=((s.T / s.rho) * d2).tocsr(), op_v=((s.EA / s.rho) * d2).tocsr(), gp=gp, gm=gm,
        ident=sparse.identity(s.N - 1, format="csr"), proj=(2.0 / s.L) * s.h * shape,
        rho=s.rho, a=s.EA - s.T, k=s.k, theta=s.theta,
    )
    phi = AMP * shape
    psi = np.zeros(s.N - 1)
    s_var = (s.c * np.pi) ** 2
    n = s.N - 1

    jac = rw._jacobian(phi, psi, s_var, time_discrete=True, **kw).toarray()
    j_pz, j_zp = jac[:n, n : 2 * n], jac[n : 2 * n, :n]
    cos_k = 1.0 - 0.5 * s.k**2 * s_var
    scale = np.max(np.abs(j_pz))
    assert np.max(np.abs(j_pz - cos_k * j_zp)) / scale < 1e-14  # the exact relation
    assert np.max(np.abs(j_pz - j_zp)) / scale > 1e-9  # ...and it is NOT symmetric

    jac0 = rw._jacobian(phi, psi, s_var, time_discrete=False, **kw).toarray()
    j_pz0, j_zp0 = jac0[:n, n : 2 * n], jac0[n : 2 * n, :n]
    assert np.max(np.abs(j_pz0 - j_zp0)) / np.max(np.abs(j_pz0)) < 1e-14  # symmetric at k -> 0


def test_planar_hessian_matches_the_core_discrete_gradient_jacobian():
    """``planar_hessian_cells == 2 * GeometricString._dg_jacobian(q, q)`` — a cross-check both ways.

    Two independent derivations of the same Hessian meeting: the oracle's (assembled in closed form
    on the planar slice) and the core's (assembled for its Newton solve). The factor of 2 is
    ``d(qbar)/d(q+) = 1/2`` — evaluating the core's DG Jacobian at ``q+ == q-`` collapses it to half
    the continuum Hessian.

    **The (v,v) block is the loose one, and the core is the inaccurate side.** This oracle uses the
    exact identity ``(1+z)^2 - Lambda^2 = -p^2``, giving ``H_zz = -a p^2 / Lambda^3`` with no
    cancellation at all; the core assembles ``a(chi - 1) + ...``, which cancels two ``O(1)`` terms
    at musical strain (measured: 7e-11 relative at strain 1e-3, 7e-15 at strain 0.1 — *worse the
    more realistic the string*, the same pathology :meth:`_stretch_terms` was written to cure).
    Harmless where it lives: a Newton Jacobian only steers the iteration, and the **residual**
    defines the root. Recorded rather than fixed — but if the core's DG Jacobian is ever reused
    somewhere the accuracy matters, this is the note.
    """
    s = _string()
    a = s.EA - s.T
    rng = np.random.default_rng(1)
    for strain in (1e-3, 1e-2, 0.1):
        p = strain * rng.standard_normal(s.N)
        z = 0.1 * strain * rng.standard_normal(s.N)
        core = s._dg_jacobian(np.stack((p, np.zeros(s.N), z)), np.stack((p, np.zeros(s.N), z)))
        core = core.toarray()
        n = s.N
        h_pp, h_pz, h_zz = planar_hessian_cells(p, z, a)
        assert np.max(np.abs(h_pp - 2 * np.diag(core[:n, :n]))) / np.max(np.abs(h_pp)) < 1e-12
        assert (
            np.max(np.abs(h_pz - 2 * np.diag(core[:n, 2 * n :]))) / np.max(np.abs(h_pz)) < 1e-12
        )
        assert (
            np.max(np.abs(h_pz - 2 * np.diag(core[2 * n :, :n]))) / np.max(np.abs(h_pz)) < 1e-12
        )
        # The core's own cancellation sets this bar, not the identity.
        assert (
            np.max(np.abs(h_zz - 2 * np.diag(core[2 * n :, 2 * n :]))) / np.max(np.abs(h_zz))
            < 1e-8
        )


# -- claim 2: physics — the Tier C/8 limit and its mechanism ----------------------------


def test_kc_circular_frequency_error_is_the_mode_shape_deformation():
    """**Tier C/8, with a mechanism.** The KC error tracks ``shape_residual`` ~1:1, universally.

    ``Omega = sqrt(omega0^2 + eps R^2)`` assumes ``phi`` is a sine. It is not: a rigid helix is
    stretched **non-uniformly** — most near the nodes, where ``phi'`` is largest — so ``T(x)`` is
    non-uniform and the true ``phi`` is a deformed sine. That is the whole content of the breakage,
    and it was gotten backwards once at plan time.

    Measured, the frequency error and the shape deformation are not merely both ``O(R^2)`` — their
    **ratio is ~4/3 and does not depend on ``EA/T`` or on the mode** (1.31–1.33 across ``EA/T`` =
    50/100/400 and modes 1/2). The universality is the real claim: it says the deformation is a
    single geometric fact about spinning a helix, not a parameter-dependent accident.

    The ratio is asserted as a **band**, not as ``4/3``: the constant is measured, not derived, and
    this project does not pin an error constant it obtained from a fit (the plan's own instruction
    for Tier C — "measure the residual's amplitude scaling empirically; do not pin the error order
    from a Taylor argument").

    Run at ``time_discrete=False`` **on purpose**: the fully-discrete ``Omega`` also carries the
    theta-scheme's temporal dispersion, which has nothing to do with mode-shape deformation and
    would swamp it. The discrete ``p2`` goes into the KC coefficients for the same reason. What is
    left over is the shape, alone.
    """
    for EA_over_T, mode in ((50.0, 1), (500.0, 1), (500.0, 2)):
        s = _string(EA=EA_over_T * 200.0)
        p2 = spatial_eigenvalue_p2(s.N, s.h, mode)
        omega0_sq, eps = kc_mode_coefficients(
            c=s.c, kappa=0.0, EA=s.EA - s.T, rho=s.rho, p2=p2, L=s.L
        )
        wave = geometric_rotating_wave(s, 1e-3, mode, time_discrete=False)
        kc = kc_circular_frequency(omega0_sq=omega0_sq, eps=eps, amplitude=1e-3)
        rel = (wave.Omega - kc) / kc
        # Positive, and physically so: tension is highest where phi' is largest (at the nodes), so
        # the Rayleigh quotient beats the uniform-tension estimate. A negative sign here would mean
        # the deformation was understood backwards -- which is exactly what happened once.
        assert rel > 0
        assert 1.2 < rel / wave.shape_residual < 1.45
        # ...and KC is exact in the limit it is a limit of.
        tiny = geometric_rotating_wave(s, 1e-6, mode, time_discrete=False)
        kc_tiny = kc_circular_frequency(omega0_sq=omega0_sq, eps=eps, amplitude=1e-6)
        assert abs(tiny.Omega - kc_tiny) / kc_tiny < 1e-9


def test_shape_residual_scales_as_amplitude_squared():
    """``shape_residual ~ R^2`` — the deformation's amplitude law, measured not Taylor-asserted.

    Exactly ``4.000x`` per doubling over four amplitudes. Which also says the deformation is *not*
    an artifact of the continuation path: an under-converged Newton would not land on a clean power
    law.
    """
    s = _string()
    residuals = []
    for amp in (5e-4, 1e-3, 2e-3, 4e-3):
        wave = geometric_rotating_wave(s, amp, 1, time_discrete=False)
        assert wave.converged
        residuals.append(wave.shape_residual)
    for lo, hi in zip(residuals[:-1], residuals[1:], strict=True):
        assert abs(hi / lo - 4.0) < 0.05


def test_the_tension_field_is_non_uniform_and_that_is_the_mechanism():
    """The helix's tension varies along ``x`` — what a Kirchhoff-Carrier scalar cannot represent.

    The mechanism behind the two tests above, read directly off the solution rather than inferred
    from a frequency. Model #9 collapses the tension to **one number**; here it is a field, the
    field is genuinely non-uniform, and the non-uniformity grows with amplitude. That difference is
    the entire reason model #10 exists.

    Tension is maximal at the **ends** (where ``phi'`` is largest for a mode-1 helix) and minimal at
    the centre (where ``phi'`` vanishes) — so it is *not* peaked where the displacement is.

    **How non-uniform, honestly: barely.** At ``R = 4e-3`` the helix raises the tension from 200 N
    to ~203.94 N, and the *spread along x* is ~0.016 N — **0.4 % of the rise it sits on**. That
    smallness is not a weakness of the test, it is the explanation of the whole Tier C taxonomy: KC
    replaces this nearly-flat field with its average, which is why KC is a *good* oracle (error
    ~1e-6 at musical amplitude) and still a *wrong* one. Model #10 exists for the 0.4 %.
    """
    s = _string()
    spreads = []
    for amp in (1e-3, 2e-3):
        wave = geometric_rotating_wave(s, amp)
        tension = wave.tension
        assert np.all(tension > 0)  # a hardening helix never goes slack
        assert tension[0] > tension[len(tension) // 2]  # peaked at the ends, not the middle
        spreads.append((tension.max() - tension.min()) / s.T)
    assert spreads[0] > 1e-6
    # O(R^2), with a visible O(R^4) correction: the ratio measures 4.004 / 4.015 / 4.059 / 4.232
    # across R = 5e-4 .. 8e-3, so the bar is set where the leading order is still clean.
    assert abs(spreads[1] / spreads[0] - 4.0) < 0.05
    # ...and every cell is stretched: Lambda >= 1 (the energy_floor argument's premise).
    assert np.all(wave.stretch_ratio >= 1.0)


def test_time_discrete_and_semi_discrete_differ_by_temporal_dispersion():
    """The two BVP flavours differ by the theta-scheme's ``O(k^2)`` temporal dispersion, only.

    Which is what licenses using ``time_discrete=False`` for the physics comparison and
    ``time_discrete=True`` for the seeding. If the gap did not shrink at ``O(k^2)``, one of the two
    factors would be wrong rather than merely time-discrete.
    """
    gaps = []
    for lam_long in (0.5, 0.25):
        s = _string(lam_long=lam_long)
        discrete = geometric_rotating_wave(s, AMP, 1, time_discrete=True)
        semi = geometric_rotating_wave(s, AMP, 1, time_discrete=False)
        gaps.append(abs(discrete.Omega - semi.Omega) / semi.Omega)
    assert gaps[0] > 0.0
    assert abs(gaps[0] / gaps[1] - 4.0) < 0.3  # halving k quarters the gap: O(k^2)


# -- guards -----------------------------------------------------------------------------


def test_softening_string_is_rejected():
    """``EA < T`` is outside this oracle's scope — the continuation seed is the hardening branch.

    The *resonator* permits softening (deliberately: ``allow_softening``, and the string provably
    conserves there). The oracle does not, and the honest reason is scope rather than physics.
    """
    s = _string()
    with pytest.raises(ValueError, match="softening"):
        solve_rotating_wave(
            L=s.L, T=s.T, rho=s.rho, EA=0.5 * s.T, fs=s.fs, N=s.N, theta=s.theta, amplitude=1e-3
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(L=-1.0), "positive"),
        (dict(rho=0.0), "positive"),
        (dict(N=1), "N must be"),
        (dict(mode=0), "mode must be"),
        (dict(mode=99), "mode must be"),
        (dict(kappa=-1.0), "kappa"),
        (dict(theta=0.0), "theta"),
        (dict(theta=1.5), "theta"),
        (dict(continuation_steps=0), "continuation_steps"),
        (dict(maxiter=0), "maxiter"),
        (dict(tol=0.0), "tol"),
    ],
)
def test_bad_parameters_are_rejected(kwargs, match):
    """Non-physical parameters raise rather than returning a plausible-looking helix."""
    s = _string()
    base = dict(
        L=s.L, T=s.T, rho=s.rho, EA=s.EA, fs=s.fs, N=s.N, theta=s.theta, amplitude=1e-3, mode=1
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        solve_rotating_wave(**base)


def test_history_helper_requires_a_positive_rate():
    """``rotating_wave_history`` needs the resonator's own ``fs`` — a wrong one is a wrong helix."""
    wave = geometric_rotating_wave(_string(), 1e-3)
    with pytest.raises(ValueError, match="fs"):
        rotating_wave_history(wave, fs=0.0)


def test_non_convergence_warns_and_is_reported():
    """A stalled continuation **warns** and reports ``converged=False`` — it never silently renders.

    Model #6's honesty rule. Forced here by starving the continuation (one step, straight to a large
    amplitude, one iteration), which is a synthetic failure — but the reporting path is the point,
    and a caller must be able to tell a converged helix from a shape that merely came back.
    """
    s = _string()
    with pytest.warns(RuntimeWarning, match="did not converge"):
        wave = solve_rotating_wave(
            L=s.L, T=s.T, rho=s.rho, EA=s.EA, fs=s.fs, N=s.N, theta=s.theta,
            amplitude=0.05, mode=1, continuation_steps=1, maxiter=1,
        )
    assert not wave.converged
