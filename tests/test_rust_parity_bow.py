"""Rust vs NumPy/SciPy for the bowed string — Phase 3's last model.

``docs/dev/rust-migration-plan.md`` §20. The bow is a *continuous nonlinear exciter*: every step
solves a scalar friction root by safeguarded Newton, with a scanned bracket plus ``brentq`` behind
it for the slip events where the current branch's root vanishes. That makes this file's job
narrower than most parity files' and more specific:

**Three things are compared that no other file in the repo compares.**

* the **Newton work per step**, step for step. §19.11 asked for it: a last bit that reaches an
  iteration count is a control-flow difference rather than a rounding difference, and the tension
  string's batch found exactly that (a different count on 1,400 of 5,000 steps). The Rust side
  reports it through ``step_reporting``, which exists only for this; the Python side is
  instrumented here by patching ``_residual`` and muting the bracket. Both count *residual
  evaluations in the Newton phase, seed included* — see the core module for why that, and not
  accepted steps;
* the **fallback branch**, step for step rather than in total. Two runs can reach the same total
  with the bracket firing on different steps;
* the two **spellings of the friction residual**. ``_residual`` multiplies by ``g`` last;
  ``_bracketed_root`` hoists ``g * force * sqrt(2a)`` into one scalar for the array. They are
  different doubles, they must stay different, and the assertion below is the Python half of the
  pin in ``crates/physsynth-core/tests/bow.rs``.

**The qualifier every trajectory assertion here carries.** The two implementations agree to the
BIT — field, telemetry, ``energy()`` and Newton work alike — only when both sides use the same
banded solver, which is what :func:`shared_solver` (and ``PHYSSYNTH_RS=1``) arranges. Without it
SciPy calls OpenBLAS's blocked ``DTBSV`` while the Rust string runs the reference ``DTBSV``
transcribed (§15.3), so the strings differ in the last bit from step one for a reason this batch
did not introduce.

**And the way that difference then behaves is this batch's finding.** It does not grow. Measured
2026-08-27 over 20,000 steps and all three fixtures below, the field gap sits at 1e-14 of the run's
peak amplitude from step 500 onward and never exceeds 6.7e-14 — flat, not a trend. A bowed string
is driven onto a *stable limit cycle*, so a perturbation is squeezed back onto it rather than
amplified: the opposite of the barrier's chaotic separation (§16.5), and a fifth regime next to it,
the mallet's transient (§17.5), the linear string's random walk (§18.6) and the tension string's
amplitude threshold (§19.5). The bars below come from that measurement, not from a default.

**The normaliser is part of the claim.** §14.2 established that a decaying trajectory must be
normalised by amplitude rather than pointwise. The bow needs one more word: by the *running peak*
amplitude, not the instantaneous one. Helmholtz motion beats, so the instantaneous maximum passes
through near-nodes, and dividing by it turns a flat 1e-14 into apparent 3e-12 spikes that are the
denominator rather than the trajectory.

Green with the flag and without it, per §16.4's convention — the assertions differ between the two
modes, the file does not.
"""

import contextlib
import math

import numpy as np
import pytest

from physsynth.core import string_damped
from physsynth.core.bow import BowedStringPy, friction_smooth_deriv_py, friction_smooth_py
from physsynth.core.string_damped import DampedStiffStringPy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13  # the plan's §4 agreement target for a short run
# The whole-run ceiling for an independent-solver comparison, from the measurement in the module
# docstring: 6.7e-14 worst over 20,000 steps across all three fixtures, so this is that number with
# an order of magnitude of headroom. It is a ceiling on a NON-GROWING quantity, which is the point —
# a bar this loose would be meaningless against an exponential and is not against a flat line.
LIMIT_CYCLE_TOL = 1e-12
DRIFT_TOL = 1e-10  # CLAUDE.md's acceptance bar, which neither implementation may cross
BALANCE_TOL = 1e-11  # the bow's own money-test bar

L_DEF, T_DEF, RHO_DEF = 1.0, 200.0, 0.005


def rs_cholesky(ab, lower=False):
    return physsynth_rs.cholesky_banded_upper(np.ascontiguousarray(ab, dtype=float))


def rs_cho_solve(cb_and_lower, b):
    cb, _lower = cb_and_lower
    return physsynth_rs.cho_solve_banded_upper(
        np.ascontiguousarray(cb, dtype=float), np.ascontiguousarray(b, dtype=float)
    )


@contextlib.contextmanager
def shared_solver():
    """Put the Python string on the Rust banded solver for the duration of the block.

    ``string_damped`` captures ``cho_solve_banded`` at import (the hazard ``test_stability.py``'s
    guard watches), so patching the captured name is the only way to hold the solver constant while
    the *model* varies. Under ``PHYSSYNTH_RS=1`` this is already the state of the world and the
    patch is a no-op. Same helper as ``test_rust_parity_strings.py``'s, one model narrower.
    """
    saved = (string_damped.cholesky_banded, string_damped.cho_solve_banded)
    string_damped.cholesky_banded = rs_cholesky
    string_damped.cho_solve_banded = rs_cho_solve
    try:
        yield
    finally:
        string_damped.cholesky_banded, string_damped.cho_solve_banded = saved


def string_kw(N=100, lam=0.9, kappa=0.0, sigma0=0.5, sigma1=0.05, theta=0.28):
    c = math.sqrt(T_DEF / RHO_DEF)
    return dict(
        L=L_DEF,
        T=T_DEF,
        rho=RHO_DEF,
        fs=c * N / (L_DEF * lam),
        N=N,
        kappa=kappa,
        sigma0=sigma0,
        sigma1=sigma1,
        theta=theta,
    )


# The two regimes the bow's own suite establishes, and they are not interchangeable here:
# `WEAK` has `helmholtz_number < 1`, a single-valued friction equation and therefore NO fallback;
# `STRONG` is deep in the multivalued regime, where a slip makes the current branch's root vanish
# and the bracket fires. A parity file that only ran one of them would be comparing one of the two
# code paths (§16.4's fixture question, this model's version of it).
WEAK = dict(bow_position=0.2, v_bow=0.1, force=0.02, sharpness=60.0)
BASE = dict(bow_position=0.2, v_bow=0.1, force=1.0, sharpness=100.0)
STRONG = dict(bow_position=0.2, v_bow=0.1, force=4.0, sharpness=120.0)


def _pair(bow_kw, **skw):
    py = BowedStringPy(string=DampedStiffStringPy(**string_kw(**skw)), **bow_kw)
    rs = physsynth_rs.BowedString(
        string=physsynth_rs.DampedStiffString(**string_kw(**skw)), **bow_kw
    )
    return py, rs


def _count_newton_evals(bow):
    """Instrument a Python bow to count residual evaluations inside its Newton phase.

    The bracket's own ``brentq`` calls ``self._residual`` too, so it is muted for the duration —
    otherwise a fallback step would report the bracket's work as Newton's and the comparison would
    be against a different quantity on the two sides. Returns the list the wrapper appends to.
    """
    counts = []
    orig_solve, orig_residual, orig_bracket = (
        bow._solve_v_rel,
        bow._residual,
        bow._bracketed_root,
    )
    live = {"n": 0, "counting": False}

    def residual(v, v_free):
        if live["counting"]:
            live["n"] += 1
        return orig_residual(v, v_free)

    def bracket(v_free):
        live["counting"] = False
        try:
            return orig_bracket(v_free)
        finally:
            live["counting"] = True

    def solve(v_free):
        live["n"], live["counting"] = 0, True
        try:
            return orig_solve(v_free)
        finally:
            live["counting"] = False
            counts.append(live["n"])

    bow._residual, bow._bracketed_root, bow._solve_v_rel = residual, bracket, solve
    return counts


def _run(py, rs, steps):
    """Step both; return the worst relative field gap, both eval counts, both fallback flags.

    The gap is normalised by the **running peak** amplitude, not the instantaneous one — see the
    module docstring. Helmholtz motion beats, and an instantaneous normaliser reports the beat's
    nodes as divergence spikes two orders of magnitude above the real number.
    """
    py_evals = _count_newton_evals(py)
    rs_evals, py_fallback, rs_fallback = [], [], []
    worst, peak = 0.0, 1e-300
    for _ in range(steps):
        before = py.fallbacks
        py.step()
        py_fallback.append(py.fallbacks != before)
        evals, used = rs.step_reporting()
        rs_evals.append(evals)
        rs_fallback.append(used)
        peak = max(peak, float(np.max(np.abs(py.state))))
        worst = max(worst, float(np.max(np.abs(py.state - rs.state))) / peak)
    return worst, py_evals, rs_evals, py_fallback, rs_fallback


# -- construction ------------------------------------------------------------------------------


@pytest.mark.parametrize("bow_kw", [WEAK, BASE, STRONG])
def test_the_derived_scalars_are_identical(bow_kw):
    py, rs = _pair(bow_kw)
    for name in (
        "k",
        "v_bow",
        "force",
        "sharpness",
        "newton_tol",
        "newton_maxiter",
        "node",
        "x_bow",
        "L",
        "beta",
        "helmholtz_number",
        "_g",
        "_force_pref",
    ):
        assert getattr(py, name) == getattr(rs, name), name


def test_the_admittance_agrees_to_the_group_a_target():
    # `a = A^{-1} e_i` is a banded solve, so without a shared solver it is LAPACK against the
    # transcription (§15.3) and cannot be exact. It is also the ONLY solve in this model that
    # differs at construction rather than per step, which is why the bow escapes §15.4's window.
    py, rs = _pair(BASE)
    scale = np.max(np.abs(py._a_full))
    assert np.max(np.abs(py._a_full - rs._a_full)) / scale < GROUP_A_TOL
    with shared_solver():
        py, rs = _pair(BASE)
    np.testing.assert_array_equal(py._a_full, rs._a_full)
    np.testing.assert_array_equal(py._a_vec, rs._a_vec)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(bow_position=0.13, v_bow=0.1, force=-1.0), "force"),
        (dict(bow_position=0.13, v_bow=0.1, force=1.0, sharpness=0.0), "sharpness"),
        (dict(bow_position=1.5, v_bow=0.1, force=1.0), "bow_position"),
        (dict(bow_position=0.13, v_bow=0.1, force=1.0, newton_maxiter=0), "newton_maxiter"),
    ],
)
def test_both_reject_the_same_construction_with_the_same_text(kwargs, message):
    with pytest.raises(ValueError, match=message) as py_err:
        BowedStringPy(string=DampedStiffStringPy(**string_kw()), **kwargs)
    with pytest.raises(ValueError, match=message) as rs_err:
        physsynth_rs.BowedString(
            string=physsynth_rs.DampedStiffString(**string_kw()), **kwargs
        )
    assert str(py_err.value) == str(rs_err.value)


def test_the_rust_bow_refuses_a_python_string():
    # The reed's rule (§12.8), for the reed's reason: a silent fallback would be a Rust bow
    # reporting Rust while bowing a Python string, which is the green-and-meaningless run the whole
    # swap guard exists to prevent.
    with pytest.raises(TypeError, match="DampedStiffString"):
        physsynth_rs.BowedString(string=DampedStiffStringPy(**string_kw()), **BASE)


# -- the friction characteristic -----------------------------------------------------------------


@pytest.mark.parametrize("force", [0.0, 0.02, 1.0, 4.0])
@pytest.mark.parametrize("sharpness", [40.0, 100.0, 400.0])
def test_the_friction_curve_is_bit_identical(force, sharpness):
    # Pure scalar arithmetic — no reduction, no solve, so IEEE-754 fixes the answer exactly once
    # the operation order matches. `math.exp` and Rust's `f64::exp` are the same C library call.
    for i in range(-200, 201):
        v = i * 0.001
        assert friction_smooth_py(v, force, sharpness) == physsynth_rs.friction_smooth(
            v, force, sharpness
        ), v
        assert friction_smooth_deriv_py(
            v, force, sharpness
        ) == physsynth_rs.friction_smooth_deriv(v, force, sharpness), v


def test_the_two_residual_spellings_are_not_the_same_double():
    """The Python half of the pin in `crates/physsynth-core/tests/bow.rs`.

    `_residual` multiplies by `g` last; `_bracketed_root` hoists `g * force * sqrt(2a)` into one
    scalar so NumPy can apply it to the whole scan array at once. Those are different doubles, and
    the scan's values decide which brackets exist — so a maintainer merging them would change a
    branch at a slip event while every physics bar stayed green. This test fails if they are ever
    made to agree, on either side.

    The bar is "any", not a fraction, on purpose: measured 2026-08-27 they disagree in 4,158 of
    20,000 samples at this fixture and in 568 at the weakest one the suite builds, so *how often* is
    set by where the bow is being played. *That* they can differ is the port's business and is what
    is asserted.
    """
    bow = BowedStringPy(string=DampedStiffStringPy(**string_kw()), **STRONG)
    g, force, a = bow._g, bow.force, bow.sharpness
    vs = np.linspace(-0.25, 0.25, 20001)
    v_free = 0.113
    hoisted = vs - v_free + g * (force * math.sqrt(2.0 * a)) * vs * np.exp(-a * vs * vs + 0.5)
    direct = np.array([bow._residual(float(v), v_free) for v in vs])
    assert np.any(hoisted != direct), (
        "the hoisted and direct residual spellings agreed everywhere -- one of them has been "
        "edited into the other, and the bracket scan no longer reproduces NumPy's"
    )


# -- the trajectory ------------------------------------------------------------------------------


@pytest.mark.parametrize("bow_kw", [WEAK, BASE, STRONG])
def test_the_trajectory_is_bit_identical_under_a_shared_solver(bow_kw):
    with shared_solver():
        py, rs = _pair(bow_kw)
        worst, py_evals, rs_evals, py_fb, rs_fb = _run(py, rs, 4000)
    assert worst == 0.0, f"the field diverged by {worst:.2e} under a shared solver"
    np.testing.assert_array_equal(py.state, rs.state)
    assert py.v_rel == rs.v_rel
    assert py.bow_force == rs.bow_force
    assert py.bow_power == rs.bow_power
    assert py.bow_work == rs.bow_work
    assert py.energy() == rs.energy()
    assert py.n == rs.n == 4000
    assert py_evals == rs_evals, "the Newton work per step differs"
    assert py_fb == rs_fb, "the fallback branch was taken on different steps"
    assert py.fallbacks == rs.fallbacks


@pytest.mark.parametrize("bow_kw", [BASE, STRONG])
def test_the_branch_decisions_survive_independent_solvers(bow_kw):
    # The sharper form of the question §19.11 left. On the tension string a last bit in a reduction
    # changed the root-find's iteration count on 1,400 of 5,000 steps. Here the perturbation the
    # two solvers introduce is ~1e-14 RELATIVE while the Newton tolerance is 1e-13 ABSOLUTE in
    # velocity units of order 0.1, so it almost never straddles the test: measured 2026-08-27, the
    # eval counts differ on at most 1 step in 20,000 and the fallback branch on none. The bar is
    # written as a fraction rather than as zero because "almost never" is the honest claim.
    py, rs = _pair(bow_kw)
    _worst, py_evals, rs_evals, py_fb, rs_fb = _run(py, rs, 4000)
    assert py_fb == rs_fb, "the fallback branch was taken on different steps"
    differing = sum(1 for a, b in zip(py_evals, rs_evals, strict=True) if a != b)
    assert differing <= len(py_evals) // 1000, (
        f"the Newton work differed on {differing} of {len(py_evals)} steps -- more than a last bit "
        "straddling the tolerance"
    )


@pytest.mark.parametrize("bow_kw", [WEAK, BASE, STRONG])
def test_the_field_stays_bounded_over_a_long_run_with_independent_solvers(bow_kw):
    # The finding, asserted rather than described: a bowed string is driven onto a stable limit
    # cycle, so the solver difference is squeezed rather than amplified. Measured over 20,000 steps
    # the gap is FLAT at ~1e-14 of the run's peak amplitude -- the opposite shape from the barrier's
    # exponential separation, and the reason a ceiling is a meaningful assertion here.
    py, rs = _pair(bow_kw)
    worst, *_ = _run(py, rs, 8000)
    assert worst < LIMIT_CYCLE_TOL, f"the field diverged by {worst:.2e} over 8,000 steps"


# -- the physics bars, asserted on the Rust side -------------------------------------------------


@pytest.mark.parametrize("force,sharpness", [(1.0, 60.0), (2.0, 100.0), (0.5, 40.0)])
def test_the_rust_bow_balances_energy(force, sharpness):
    # The acceptance contract, which holds whatever the comparison above says. Lossless: every
    # joule of stored energy is accounted for by the bow's work.
    rs = physsynth_rs.BowedString(
        string=physsynth_rs.DampedStiffString(**string_kw(sigma0=0.0, sigma1=0.0)),
        bow_position=0.2,
        v_bow=0.1,
        force=force,
        sharpness=sharpness,
    )
    e0 = rs.energy()
    worst = 0.0
    for _ in range(4000):
        rs.step()
        e, w = rs.energy(), rs.bow_work
        worst = max(worst, abs((e - e0) - w) / (abs(e) + abs(w) + 1e-30))
    assert worst < BALANCE_TOL, f"energy-balance error {worst:.2e}"


def test_the_rust_bow_is_passive_under_loss():
    rs = physsynth_rs.BowedString(
        string=physsynth_rs.DampedStiffString(**string_kw()), **BASE
    )
    e0 = rs.energy()
    dissipation = [0.0]
    for _ in range(4000):
        rs.step()
        dissipation.append(rs.bow_work - (rs.energy() - e0))
    assert min(np.diff(dissipation)) >= -1e-9 * (abs(rs.bow_work) + 1.0)
    assert dissipation[-1] >= -DRIFT_TOL
