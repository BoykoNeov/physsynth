"""Model #10 (geometrically-exact string) — the structural gates.

The batch-1 money tests: the ``EA = T`` regression anchor, the discrete-gradient identity that
*makes* the scheme conservative, energy drift (with the newton-tol self-certification), the energy
floor and why it is 0 rather than the tempting negative bound, passivity, the linear limit, and the
guards.

Everything here is about the *scheme*. The payoff claims model #9 refuses to make (phantom partials,
whirling) are batch 2 — but they can only be trusted if this file is green, because they are oracles
read off a resonator these tests certify conservative.
"""

import warnings

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    GEO_NEWTON_TOL,
    KAPPA_DEFAULT,
    L_DEFAULT,
    RHO_DEFAULT,
    T_DEFAULT,
    geometric_mode_ic,
    geometric_pluck_ic,
    make_geometric_string,
    wave_speed,
)

from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_geometric import GeometricString

DRIFT_GATE = 1e-10
"""CLAUDE.md's bar: a lossless run must drift < 1e-10 relative."""


# -- the discrete gradient: the identity the whole scheme rests on ----------------------


def test_discrete_gradient_telescopes_exactly():
    """``h Σ_cells <gradbar V_nl, q+ − q−> == V_nl(q+) − V_nl(q−)`` — to machine precision.

    **This is the scheme.** Energy conservation is a *corollary* of this identity plus SBP; if it
    holds, the lossless drift is roundoff, and if it does not, no amount of tuning will save the
    drift. Testing it directly (rather than only through ``energy()``) is what separates "the model
    conserves" from "the model conserves *because of the thing we claim*".

    Checked across five decades of strain because that is where the danger is: the naive
    ``Λ(mean)`` midpoint is tested against below, and its error *shrinks* with amplitude.
    """
    s = make_geometric_string(N=32)
    rng = np.random.default_rng(11)
    for scale in (0.5, 0.1, 1e-2, 1e-3, 1e-4):
        q_plus = rng.normal(size=(3, s.N)) * scale
        q_minus = rng.normal(size=(3, s.N)) * scale
        lhs = s.h * float(np.sum(s._dg_force(q_plus, q_minus) * (q_plus - q_minus)))
        rhs = s._nl_density(q_plus) - s._nl_density(q_minus)
        assert abs(lhs - rhs) <= 1e-12 * abs(rhs), f"DG identity failed at strain {scale}"


def test_naive_midpoint_gradient_is_not_a_discrete_gradient():
    """The negative control for the trap: ``Λ(mean)`` — the tempting midpoint — **fails** the
    identity by 5–1000 %, at every strain scale.

    Worth a test because the failure is *invisible* dynamically: the naive form has the right limit,
    the right glide, the right spectrum, and its DG error shrinks with amplitude. It fails only the
    energy gate. Pinning it here says out loud that ``mean(Λ)`` vs ``Λ(mean)`` is a real distinction
    and not a stylistic one.
    """
    s = make_geometric_string(N=32)
    rng = np.random.default_rng(11)
    worst = 0.0
    for scale in (0.5, 0.1, 1e-2, 1e-3):
        q_plus = rng.normal(size=(3, s.N)) * scale
        q_minus = rng.normal(size=(3, s.N)) * scale
        q_bar = 0.5 * (q_plus + q_minus)
        chi = 1.0 - 1.0 / s._stretch_ratio(q_bar)  # Λ(mean): the WRONG one
        naive = np.stack(
            [s._a * chi * q_bar[0], s._a * chi * q_bar[1],
             s._a * (chi * (1.0 + q_bar[2]) - q_bar[2])]
        )
        lhs = s.h * float(np.sum(naive * (q_plus - q_minus)))
        rhs = s._nl_density(q_plus) - s._nl_density(q_minus)
        worst = max(worst, abs(lhs - rhs) / abs(rhs))
    assert worst > 1e-2, "the naive Λ(mean) midpoint should MISS the DG identity by percent-level"


def test_dg_at_equal_states_is_the_continuum_gradient():
    """``gradbar V_nl(q, q)`` is the plain continuum gradient — which is why
    :meth:`GeometricString.set_state` can reuse it for the consistent start."""
    s = make_geometric_string(N=32)
    rng = np.random.default_rng(3)
    q = rng.normal(size=(3, s.N)) * 0.05
    lam = s._stretch_ratio(q)
    chi = 1.0 - 1.0 / lam
    expected = np.stack(
        [s._a * chi * q[0], s._a * chi * q[1], s._a * (chi * (1.0 + q[2]) - q[2])]
    )
    assert np.allclose(s._dg_force(q, q), expected, rtol=1e-11, atol=0.0)


def test_sbp_adjoint_pair():
    """``δ_x− = −(δ_x+)^T`` and ``δ_x− δ_x+ == D2`` exactly.

    The summation-by-parts pair is what turns the DG identity into an energy statement (CLAUDE.md:
    "when E^n drifts in a lossless run, suspect the boundary handling first"). Both hold *exactly*,
    so assert equality, not tolerance.
    """
    from physsynth.core.operators import second_difference_matrix

    s = make_geometric_string(N=16)
    assert np.array_equal(s._Gm.toarray(), -s._Gp.toarray().T)
    d2 = second_difference_matrix(s.N, s.h).toarray()
    assert np.allclose((s._Gm @ s._Gp).toarray(), d2, rtol=0.0, atol=1e-9 * np.abs(d2).max())


def test_dg_jacobian_matches_finite_differences_and_is_not_symmetric():
    """The Newton Jacobian is right, **and it is not symmetric** — which is why ``splu``.

    A discrete gradient is not the gradient of anything: ``D_cell`` pairs a midpoint ``mbar``
    against a plus-level ``n+``. Asserting the asymmetry (not just tolerating it) pins the reason
    ``cholesky_banded`` — which every other string model in the family uses — is wrong here.
    """
    s = make_geometric_string(N=12)
    rng = np.random.default_rng(5)
    q_plus = rng.normal(size=(3, s.N)) * 0.05
    q_minus = rng.normal(size=(3, s.N)) * 0.05
    jac = s._dg_jacobian(q_plus, q_minus).toarray()

    eps = 1e-6
    jac_fd = np.zeros_like(jac)
    for i in range(3 * s.N):
        d = np.zeros(3 * s.N)
        d[i] = eps
        f_p = s._dg_force((q_plus.ravel() + d).reshape(3, s.N), q_minus).ravel()
        f_m = s._dg_force((q_plus.ravel() - d).reshape(3, s.N), q_minus).ravel()
        jac_fd[:, i] = (f_p - f_m) / (2 * eps)
    assert np.max(np.abs(jac - jac_fd)) < 1e-6 * np.max(np.abs(jac_fd))
    assert np.max(np.abs(jac - jac.T)) > 0.1 * np.max(np.abs(jac)), "expected a NON-symmetric DG"


# -- the regression anchor -------------------------------------------------------------


@pytest.mark.parametrize(
    "kappa,sigma0,sigma1",
    [(0.0, 0.0, 0.0), (KAPPA_DEFAULT, 0.0, 0.0), (KAPPA_DEFAULT, 2.0, 5e-3)],
)
def test_EA_equals_T_is_bit_identical_to_damped_string(kappa, sigma0, sigma1):
    """``EA = T`` reduces the ``u`` polarization to model #3 **bit-for-bit** — the free anchor.

    The nonlinearity coefficient is exactly ``a = EA − T0``, so ``EA = T`` makes ``gradbar V_nl``
    vanish *identically* and the guarded branch skips it. Bit-identical, not merely close: the
    linear path is the same expressions in the same order (float addition is not associative). This
    is the direct analog of model #9's ``EA = 0``, and it is **why the θ-split is worth keeping** —
    moving the whole operator to θ=1/2 would forfeit it.
    """
    N, fs = 64, 12800.0
    kw = dict(L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=fs, N=N, kappa=kappa,
              sigma0=sigma0, sigma1=sigma1)
    ref = DampedStiffString(**kw)
    geo = GeometricString(EA=T_DEFAULT, **kw)

    ic = geometric_pluck_ic(N, amp=1e-3)
    ref.set_state(ic)
    geo.set_state(ic)
    assert np.array_equal(ref.u_prev, geo.u_prev), "the consistent start must match bit-for-bit"

    for _ in range(300):
        ref.step()
        geo.step()
    assert np.array_equal(ref.u, geo.u)
    assert ref.energy() == geo.energy()
    assert geo.nonlinear_energy() == 0.0
    # ...and the two spare fields never woke up.
    assert np.max(np.abs(geo.w)) == 0.0
    assert np.max(np.abs(geo.v)) == 0.0


def test_EA_equals_T_makes_all_three_fields_linear_and_decoupled():
    """At ``EA = T`` the ``w`` and ``v`` fields are independent linear waves too — the model is
    exactly three decoupled strings, so cross-talk of any kind is a bug."""
    N, fs = 64, 12800.0
    geo = GeometricString(L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=fs, N=N, EA=T_DEFAULT)
    geo.set_state(geometric_mode_ic(N, 1, 1e-3), geometric_mode_ic(N, 2, 1e-3),
                  geometric_mode_ic(N, 3, 1e-3))
    e0 = geo.energy()
    for _ in range(400):
        geo.step()
    assert abs(geo.energy() - e0) / abs(e0) < 1e-12
    assert geo.nonlinear_energy() == 0.0


# -- energy: drift, the floor, passivity ------------------------------------------------


def test_lossless_drift_single_mode_with_nonlinear_fraction_reported():
    """Lossless drift < 1e-10 **with the nonlinear fraction asserted non-trivial**.

    The second half is the point (model #6/#9's lesson): a nonlinearity bug *hides* at small
    amplitude, where the test silently re-runs the linear scheme and passes. Assert that the
    nonlinear channel actually carried energy.
    """
    s = make_geometric_string(N=64, EA=EA_DEFAULT)
    s.set_state(geometric_mode_ic(s.N, 1, 4e-3))
    e0 = s.energy()
    peak_nl = 0.0
    peak_long = 0.0
    for _ in range(600):
        s.step()
        peak_nl = max(peak_nl, abs(s.nonlinear_energy()) / abs(e0))
        peak_long = max(peak_long, s.longitudinal_energy() / abs(e0))
    assert s.n_not_converged == 0
    assert abs(s.energy() - e0) / abs(e0) < DRIFT_GATE
    assert peak_nl > 1e-3, f"nonlinearity barely engaged ({peak_nl:.2e}) — test is secretly linear"
    assert peak_long > 1e-3, f"longitudinal field barely moved ({peak_long:.2e})"


def test_lossless_drift_from_a_plucked_broadband_ic():
    """Drift < 1e-10 from a **plucked** (broadband) IC — the general-case test.

    A single-mode IC is secretly a scalar Duffing test: the strain stays nearly uniform, so it never
    exercises the **local** tension field that is the entire difference between model #10 and model
    #9's spatial scalar. The pluck's corner makes the stretch vary cell-to-cell. If only one of
    these two tests could exist, it would be this one.
    """
    s = make_geometric_string(N=64, EA=EA_DEFAULT)
    s.set_state(geometric_pluck_ic(s.N, amp=3e-3))
    e0 = s.energy()
    for _ in range(800):
        s.step()
    assert s.n_not_converged == 0
    assert abs(s.energy() - e0) / abs(e0) < DRIFT_GATE
    # The strain really is non-uniform — i.e. this IC tests what it claims to.
    spread = np.ptp(s.stretch_ratio)
    assert spread > 1e-6, f"stretch field is uniform ({spread:.2e}) — KC would have sufficed"


def test_drift_is_controlled_by_the_newton_tolerance():
    """**The self-certification.** Absent a closed form for general motion, this is how we know the
    drift belongs to the *solve* and not to the scheme: tighten the solve, and the drift follows.

    The DG force is exact only *at* the root, so an inexact root injects energy in proportion to the
    residual — and the tolerance must be **relative**, since an absolute bar loosens silently as the
    string quietens.

    **What this is not: proportional.** Newton converges quadratically at a resolved ``lam_long``,
    so the residual leaps ``1e-4 -> 1e-11 -> 1e-18`` and *every* tolerance between those rungs
    exits at the identical root. Measured at N=32, mode-1 amp 8e-3, 400 steps::

        tol 1e-4 .. 1e-3   0 iters/step (the linear seed is accepted)   drift 4.6e-2
        tol 1e-5 .. 1e-13  1 iter/step                                  drift 3.99e-11 (all equal)
        tol 1e-15          1.8 iters/step                               drift 1.1e-11

    So drift is a step function of the *iteration count*, not a line in ``tol``. An earlier version
    of this test asserted proportionality and appeared to measure it over five decades — but that
    was read in the ``lam_long ~ 11`` regime, where Newton grinds against its round-off floor and
    never converges. The proportionality was an artifact of the broken parameterisation; the
    control is real. Assert the control.
    """
    drifts = []
    for tol in (1e-4, GEO_NEWTON_TOL):
        s = make_geometric_string(N=32, EA=EA_DEFAULT, newton_tol=tol)
        s.set_state(geometric_mode_ic(s.N, 1, 8e-3))
        e0 = s.energy()
        for _ in range(400):
            s.step()
        drifts.append(abs(s.energy() - e0) / abs(e0))
    assert drifts[0] > 1e6 * drifts[1], (
        f"loosening the solve must loosen the drift, got {drifts[0]:.2e} vs {drifts[1]:.2e}"
    )
    assert drifts[1] < DRIFT_GATE


def test_energy_floor_is_zero_and_respected():
    """``E >= 0``, and the string really does approach it — the floor is not a slack gate.

    The naive read of the potential says ``E >= −L T0²/(2 EA)``: the ``−T0 v_x`` null Lagrangian
    telescopes to ``T0(v_N − v_0) = 0`` at fixed ends, but the pre-stress density ``T0²/(2EA)``
    survives on every cell. That bound is *true but not tight*, and it is worth being explicit that
    it is not what this model obeys — see :attr:`GeometricString.energy_floor`. Clamped ends forbid
    the relaxed state that would attain it, and the constrained bound is exactly zero.
    """
    s = make_geometric_string(N=32, EA=EA_DEFAULT)
    assert s.energy_floor == 0.0
    loose = -L_DEFAULT * T_DEFAULT**2 / (2.0 * EA_DEFAULT)
    assert s.energy_floor > loose, "the free-string floor is the loose one; clamped is tighter"

    s.set_state(geometric_pluck_ic(s.N, amp=3e-3))
    for _ in range(400):
        s.step()
        assert s.energy() >= s.energy_floor
    # ...and the gate has teeth: a rest string sits *at* the floor, not comfortably above it.
    at_rest = make_geometric_string(N=32, EA=EA_DEFAULT)
    at_rest.set_state(0.0)
    assert at_rest.energy() == 0.0


def test_the_relaxed_state_is_inadmissible_with_clamped_ends():
    """**Why the floor is 0 and not ``−L T0²/(2 EA)``.** The relaxed state cannot be reached.

    ``−L T0²/(2 EA)`` is the energy of a string whose every element has shrunk to its natural
    length. That needs ``v_x = −T0/EA`` *throughout*, i.e. ``v(L) − v(0) = −L T0/EA ≠ 0`` — the
    string would have to get shorter, and both ends are clamped. It is the floor for a **free**
    string, and would matter if a free end were ever added.
    """
    s = make_geometric_string(N=32, EA=EA_DEFAULT)
    relaxed = (EA_DEFAULT - T_DEFAULT) / EA_DEFAULT  # Λ with zero tension
    q = np.zeros((3, s.N))
    q[2] = relaxed - 1.0  # a uniform contraction to the natural length
    # The tension really does vanish there — the state is physical, just not reachable.
    assert s.EA * relaxed - s._a == pytest.approx(0.0, abs=1e-9)
    # It is inadmissible: a uniform v_x cannot integrate to v(L) - v(0) = 0.
    assert np.sum(q[2]) * s.h == pytest.approx(-L_DEFAULT * T_DEFAULT / EA_DEFAULT, rel=1e-12)
    # And the excess vanishes there (Λ = 1 + v_x exactly), so it is NOT the floor of E.
    assert s._nl_density(q) == pytest.approx(0.0, abs=1e-12)


def test_passivity_is_monotone_with_all_six_losses():
    """With σ > 0 the energy is monotone non-increasing — unconditionally.

    None of the six loss terms enters ``E``; each is dissipative by SBP. So passivity is automatic
    and needs no separate proof, exactly as in model #3 — but it is asserted because it is the
    cheapest possible check that no loss term got a sign wrong.
    """
    s = make_geometric_string(
        N=64, EA=EA_DEFAULT, sigma0=2.0, sigma1=5e-3, sigma0_long=1.0, sigma1_long=2e-3
    )
    s.set_state(geometric_pluck_ic(s.N, amp=3e-3))
    energies = [s.energy()]
    for _ in range(600):
        s.step()
        energies.append(s.energy())
    diffs = np.diff(np.array(energies))
    assert np.all(diffs <= 0.0), f"energy increased by {diffs.max():.3e}"
    assert energies[-1] < energies[0]


def test_longitudinal_loss_defaults_to_the_transverse_values():
    """``sigma*_long=None`` inherits — the constructor makes **no silent physics claim**.

    Real strings damp longitudinal motion far less than transverse, but that is a setting to opt
    into (``sigma0_long=0.0``), not a default to smuggle in.
    """
    s = make_geometric_string(sigma0=2.0, sigma1=5e-3)
    assert s.sigma0_long == 2.0
    assert s.sigma1_long == 5e-3
    opted_in = make_geometric_string(sigma0=2.0, sigma1=5e-3, sigma0_long=0.0, sigma1_long=0.0)
    assert opted_in.sigma0_long == 0.0
    assert opted_in.sigma1_long == 0.0


# -- the linear limit ------------------------------------------------------------------


def test_small_amplitude_recovers_three_linear_waves():
    """Tier A/5: as ``u → 0`` the model is three decoupled linear waves — transverse at ``n c/(2L)``
    (**both** polarizations, isotropic) and longitudinal at ``n c_long/(2L)``.

    Note the timestep is set from ``lam_long``, not ``lam``: the longitudinal field runs ~22x
    faster, and while the implicit scheme is *stable* there, it is not *accurate*. Measuring a
    longitudinal mode frequency at transverse Courant would report the θ-scheme's temporal
    dispersion, not the physics.
    """
    from physsynth.analysis import spectrum

    N = 64
    s = make_geometric_string(N=N, lam_long=0.7, kappa=0.0, EA=EA_DEFAULT)
    amp = 1e-9  # deep in the linear limit: the nonlinearity is cubic, so it vanishes here
    s.set_state(geometric_mode_ic(N, 1, amp), geometric_mode_ic(N, 2, amp),
                geometric_mode_ic(N, 3, amp))

    n_steps = 60_000
    probe_u, probe_w, probe_v = [], [], []
    for _ in range(n_steps):
        s.step()
        probe_u.append(s.u[N // 3])
        probe_w.append(s.w[N // 3])
        probe_v.append(s.v[N // 3])

    c = wave_speed()
    c_long = float(np.sqrt(EA_DEFAULT / RHO_DEFAULT))
    for probe, f_exact, name in (
        (probe_u, 1 * c / (2 * L_DEFAULT), "u mode 1"),
        (probe_w, 2 * c / (2 * L_DEFAULT), "w mode 2"),
        (probe_v, 3 * c_long / (2 * L_DEFAULT), "v mode 3"),
    ):
        got = spectrum.measure_partials_near(np.array(probe), s.fs, np.array([f_exact]))[0]
        assert got == pytest.approx(f_exact, rel=2e-3), f"{name}: {got:.2f} vs {f_exact:.2f}"


def test_both_polarizations_share_one_wave_speed():
    """The tension is **isotropic by construction** — ``c`` is shared, and only ``kappa`` can
    distinguish the polarizations. A tension-anisotropy knob would have broken geometric exactness;
    this pins that it does not exist."""
    s = make_geometric_string(kappa=1.0, kappa_w=3.0)
    assert s.c == wave_speed()
    assert s.kappa_u == 1.0
    assert s.kappa_w == 3.0
    assert not s.is_degenerate
    assert make_geometric_string(kappa=1.0).is_degenerate


# -- guards ----------------------------------------------------------------------------


def test_softening_EA_is_rejected_by_default_and_permitted_on_request():
    """``EA < T`` ⟹ ``EA_n < 0``: a potential **unbounded below**. Blow-up, not hyperreality.

    This is the one parameter in the family with a real floor, so the default protects — but the
    hatch exists, because where the hyperreal line sits is the human's call, not the constructor's.
    Above the floor the effective-coefficient surface still stands: ``(T, rho, kappa, EA)`` remain
    mutually unconstrained.
    """
    kw = dict(L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=12800.0, N=32)
    with pytest.raises(ValueError, match="SOFTENING"):
        GeometricString(EA=T_DEFAULT * 0.5, **kw)
    soft = GeometricString(EA=T_DEFAULT * 0.5, allow_softening=True, **kw)
    assert soft.allow_softening
    assert soft._a < 0.0
    # EA = T is the anchor and must remain constructible.
    assert GeometricString(EA=T_DEFAULT, **kw)._a == 0.0


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(EA=0.0), "EA"),
        (dict(EA=-1.0), "EA"),
        (dict(EA=1e5, N=1), "N must be"),
        (dict(EA=1e5, kappa=-1.0), "kappa"),
        (dict(EA=1e5, kappa_w=-1.0), "kappa_w"),
        (dict(EA=1e5, sigma0=-1.0), "sigma0"),
        (dict(EA=1e5, sigma0_long=-1.0), "sigma0_long"),
        (dict(EA=1e5, theta=0.0), "theta"),
        (dict(EA=1e5, theta=1.5), "theta"),
        (dict(EA=1e5, newton_tol=0.0), "newton_tol"),
        (dict(EA=1e5, boundary="clamped"), "boundary"),
        (dict(EA=1e5, T=-1.0), "must all be positive"),
    ],
)
def test_construction_rejects_nonphysical_parameters(kw, match):
    base = dict(L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=12800.0, N=32)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        GeometricString(**base)


def test_apply_Ainv_raises_for_a_reason_that_is_not_model_9s():
    """``apply_Ainv`` is unavailable — but **not** for model #9's reason, and the message says so.

    Model #9's ``A`` moves with the tension. Here ``A3`` is genuinely *constant*: the nonlinearity
    is a force on the RHS, not a term in the matrix. The one-step response is still state-dependent,
    because the implicit DG force couples all three fields at ``n+1`` — so the true admittance is
    the inverse of the (non-symmetric) Newton Jacobian. A precomputed ``A3⁻¹ e_i`` would be wrong by
    exactly the nonlinearity one came here for: silently, and only under coupling.
    """
    s = make_geometric_string(N=16)
    with pytest.raises(NotImplementedError, match="Newton Jacobian"):
        s.apply_Ainv(np.zeros(s.N - 1))


def test_non_convergence_warns_and_is_counted():
    """A stalled solve **warns and is counted** — it never silently renders as physics.

    Driven far past any sane amplitude the Newton solve cannot find the root; the DG force is then
    not exact, so energy conservation is void. The contract is model #6's honesty rule: warn, don't
    render.
    """
    with pytest.warns(RuntimeWarning, match="under-resolved"):  # lam=2 ⟹ lam_long ≈ 45
        s = make_geometric_string(N=32, lam=2.0, EA=EA_DEFAULT, newton_maxiter=3)
    s.set_state(geometric_mode_ic(s.N, 1, 0.35))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        for _ in range(30):
            s.step()
    assert s.n_not_converged > 0
    assert not s.converged


def test_set_state_velocity_arguments_are_keyword_only():
    """The ``v0`` name clash with models #1–3/#9 is real, so the velocities are keyword-only.

    There, ``set_state(u0, v0)``'s ``v0`` is a *velocity*. Here ``v`` is the longitudinal
    **displacement field**. A positional port from model #3 would silently inject a longitudinal
    displacement where a transverse velocity was meant — so make it impossible to write by accident.
    """
    import inspect

    sig = inspect.signature(GeometricString.set_state)
    for name in ("u_dot", "w_dot", "v_dot"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("u0", "w0", "v0"):
        assert sig.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_reported_courant_numbers_expose_the_longitudinal_tax():
    """``lam_long = sqrt(EA/T) · lam`` is reported: ~22x ``lam`` at realistic ``EA/T``.

    The implicit scheme is unconditionally stable there — *stable is not accurate*. This is
    CLAUDE.md's "tune toward λ=1" applied to the field that cannot have it for free.
    """
    with pytest.warns(RuntimeWarning, match="under-resolved"):  # lam=1 already ⟹ lam_long ≈ 22
        s = make_geometric_string(N=64, lam=1.0, EA=EA_DEFAULT)
    assert s.lam == pytest.approx(1.0)
    assert s.lam_long == pytest.approx(np.sqrt(EA_DEFAULT / T_DEFAULT), rel=1e-12)
    assert s.lam_long > 20.0
    assert s.c_long == pytest.approx(np.sqrt(EA_DEFAULT / RHO_DEFAULT))
    assert s.EA_over_T == pytest.approx(EA_DEFAULT / T_DEFAULT)


def test_under_resolved_longitudinal_field_warns_because_nothing_else_will():
    """**The guard with no CFL behind it** — and the batch-1 lesson that earned it.

    Every other explicit scheme in this package *rejects* ``lam > 1`` because it is unstable there.
    This scheme is unconditionally stable, so ``lam_long = 11`` violates nothing, raises nothing —
    and silently returns garbage (measured drift 1e+3..1e+5, Newton never converging). The trap is
    that ``c_long/c = sqrt(EA/T) ≈ 22``, so the ``lam = 0.5`` a reader of models #1–#9 would reach
    for *first* lands at ``lam_long ≈ 11``. Warn, don't reject: ``lam_long = 2`` conserves to 1e-12,
    so a hard bar would forbid configurations that demonstrably work.
    """
    from physsynth.core.string_geometric import LAM_LONG_WARN

    # The familiar transverse Courant is exactly the trap.
    with pytest.warns(RuntimeWarning, match=r"under-resolved|lam_long"):
        bad = make_geometric_string(N=32, lam=0.5, EA=EA_DEFAULT)
    assert bad.lam_long > 4.0, "lam=0.5 should land deep in the failing regime — that's the point"

    # ...and a resolved build is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        good = make_geometric_string(N=32, lam_long=0.5, EA=EA_DEFAULT)
    assert good.lam_long == pytest.approx(0.5, rel=1e-12)
    assert good.lam_long <= LAM_LONG_WARN


def test_tension_is_a_field_not_a_scalar():
    """The tension varies **along** the string — the one-line statement of what model #10 adds.

    Model #9's Kirchhoff–Carrier tension is a spatial scalar ``T(t)``; that collapse is exactly what
    makes it blind to longitudinal dynamics and planar. Here ``T(x) = EA·Λ(x) − (EA − T0)``, exactly
    ``T0`` at rest and linear in the local stretch.
    """
    s = make_geometric_string(N=64, EA=EA_DEFAULT, newton_tol=GEO_NEWTON_TOL)
    assert np.allclose(s.tension, T_DEFAULT, rtol=1e-12)  # at rest: uniform T0
    assert np.allclose(s.stretch_ratio, 1.0, rtol=1e-12)

    s.set_state(geometric_pluck_ic(s.N, amp=4e-3))
    for _ in range(300):
        s.step()
    assert np.ptp(s.tension) > 1e-3 * T_DEFAULT, "tension should vary along a plucked string"
    assert len(s.tension) == s.N
