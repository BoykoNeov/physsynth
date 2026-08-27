"""Rust vs NumPy/SciPy for the contact leg — and the reason bit-identity survives in one half.

``docs/dev/rust-migration-plan.md`` section 16. This batch ports the contact primitives and both
contact solves. The two halves land in different places and the file is organised around that:

* the **scalar** solve, which the mallet uses, contains no reduction anywhere. It comes out
  bit-identical, brentq fallback included, and is asserted that way;
* the **vector** solve, which the barrier uses, is driven by ``G @ F(eta)`` -- a dense BLAS matvec
  whose result feeds back into the next Newton iterate. That is section 14.2's construction, so
  bit-identity is off the table above one contact node and the target is Group A.

Three things here are worth reading before the assertions, because each was measured and each
would otherwise look like an arbitrary choice:

1. **The single-node case is the cause-separator.** At ``m = 1`` the matvec is one multiply and
   the LU a scalar divide, so the two implementations *must* agree to the bit. If they ever stop,
   the transcription is wrong and no story about BLAS applies to it.

2. **A "bit-identical" reading on a soft contact proves nothing about the solver.** Measured
   2026-08-27: the divergence tracks how far the Newton Jacobian ``I + G diag(F')`` is from the
   identity, not the number of contact nodes. At the default barrier fixture the condition number
   is 1.004 and the LU is effectively solving ``I d = -r``, so its differences from LAPACK never
   reach the answer -- 79 nodes in contact and the string stays bit-identical for 2,000 steps.
   Raise ``K`` to 1e8 (condition number 1.14) and it separates at step 250. This is section 14.3's
   blindness finding again, one level up: the fixture the suite uses most is in the blind spot, so
   the parity test has to bring its own stiff case.

3. **Which of these claims can be *exact* is decided by the machine, not by the port.** NumPy
   does not call the platform C library for the transcendentals -- it carries its own vectorised
   routines, dispatched at import by CPU feature -- while the Rust port calls libm. So
   ``x ** 1.5`` on an array is two implementations agreeing, and whether they do is a property of
   the processor GitHub happens to hand the job. Measured 2026-08-27, two CI runs of *identical*
   code: on one, every claim below held; on the next, fifteen of them failed by one or two ulp,
   and the failures correlate perfectly with the ufunc's shortcut ladder -- every exponent NumPy
   spells as ``sqrt``/``x*x``/``x``/``1``/``1/x`` agreed, every exponent it hands to its own
   ``pow`` did not.

   So the exact claims here are pinned to ``alpha = 1.0``, which is the **only** exponent that
   puts all three of the primitives' exponents (``alpha - 1``, ``alpha``, ``alpha + 1``) on that
   ladder simultaneously -- both sides then perform literally the same multiply, and agreement is
   a proof rather than a coincidence. It is still a one-sided contact, so the contact-set
   detection, the Newton solve, the dense LU and the discrete gradient's 0/0 branch are all
   exercised. The shipped physical exponent, ``alpha = 1.5``, keeps its coverage through the
   Group A tolerance tests, which is where a machine-dependent last bit belongs.

   Two things this does **not** buy, both of which cost a measurement to find out. The dense
   matvec ``G @ F`` is still OpenBLAS against a hand-rolled loop, so the exact claims survive on
   a reduction that is measured to agree rather than proved to; if a runner ever breaks *that*,
   the finding is one level below this one and the fix is the same shape. And the exponent
   cannot be swapped freely: at ``m = 79`` moving to ``alpha = 1.0`` **loses** item 2's blind
   spot rather than keeping it, because the tangent stiffness stops vanishing at grazing contact.
   The numbers are under that test.

4. **The vector solve's Group A window is short, per fixture, and it closes for a dynamical
   reason rather than an arithmetic one.** A string buzzing against a one-sided barrier is
   chaotic, so the two trajectories do not drift apart -- they separate. Measured 2026-08-27, the
   step at which each fixture first exceeds 1e-13 of amplitude: the single- and two-node frets and
   the lossy rail, never within 6,000; the default rail, never within 6,000 (it reaches 8.2e-14);
   the ``alpha = 1`` rail at step **1,175**; the stiff rail at step **1,584**. Past that the
   growth is exponential rather than linear -- the stiff lossless fixture reads 1.2e-13 at 5,000
   steps, 3.4e-12 at 10,000 and **1.1e-7 at 20,000**, where every earlier batch's divergence grew
   like the run length. So the tolerance assertions below run for 500 steps, roughly a third of
   the tightest measured window, and past a few thousand the honest statement is not a tolerance
   at all but the physics bars.

5. **The model shell ported later than the solve it drives, and section 4 had to be rewritten
   rather than reused.** Batch 2 left ``BarrierString`` in Python and measured the vector solve
   *through* it, by pinning ``collision.solve_contact_vector``. Once the class itself swaps, that
   pin reaches nothing under ``PHYSSYNTH_RS=1`` -- the Rust model never looks the name up -- so
   every test in section 4 would have been comparing Rust against Rust and passing for the wrong
   reason. Verified, not assumed: with the flag set, a pin that raises on call is simply never
   called. Section 4 therefore builds ``BarrierStringPy`` explicitly, and section 5 is the new
   comparison of the *shell*.

6. **The shell's own matvec differs at two contact nodes, and the state provably cannot see
   it.** Injecting the force is a second dense matvec -- ``u[1:-1] += force_pref * (cols_mat @ f)``
   -- that nothing compared across the languages until the model ported. Measured 2026-08-27 over
   2,000 steps: identical at ``m = 1`` (0 rows, the sum is one product), and at ``m = 2`` it
   differs in **1,291 of 158,000 rows** while the trajectory stays bit-identical.

   That is not luck, and the distinction matters because an exact assertion resting on luck is one
   that expires without warning. A **two**-term sum can only be reordered into a different double
   if its two terms *cancel* -- and where they cancel the correction is tiny: at every one of
   those 1,291 rows it is at most ``9.3e-13`` of ``u``, so one of its ulps is worth about
   ``1e-12`` of one of ``u``'s and cannot survive the addition. The error of a two-term reduction
   is correlated with its own smallness.

   The control is the same code at 79 terms, where that correlation is gone: the matvec differs on
   14,746 rows, the correction there is an ordinary size (median ``1.2e-4`` of ``u``), and about
   one difference in ``1/1.2e-4`` crosses a rounding boundary -- 7 reach the state over 2,000 steps
   and 30 over 6,000. So the exactness at two nodes is a statement about the *length of the sum*,
   and both halves are asserted below, because the exact claim is only honest next to the fixture
   that shows where it stops being available.
"""

import contextlib
import re
import warnings

import numpy as np
import pytest
from helpers import (
    BARRIER_ALPHA_DEFAULT,
    BARRIER_HEIGHT_DEFAULT,
    BARRIER_K_DEFAULT,
    L_DEFAULT,
    RHO_DEFAULT,
    T_DEFAULT,
    THETA_DEFAULT,
    make_barrier_string,
    make_damped_string,
    make_mallet,
    make_mallet_wall,
    wave_speed,
)
from scipy.linalg import lu_factor as sp_lu_factor
from scipy.linalg import lu_solve as sp_lu_solve

from physsynth.core import collision as C
from physsynth.core import mallet as M
from physsynth.core import string_damped
from physsynth.core.string_damped import DampedStiffStringPy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13     # the plan's short-run agreement target
DRIFT_TOL = 1e-10       # CLAUDE.md's energy bar, which neither implementation may cross
SHORT_RUN = 2000        # long enough for a bit-identity claim to mean something
GROUP_A_RUN = 500       # a third of the tightest measured window -- see item 4 in the header

# The exponents NumPy's float64 `power` ufunc loop shortcuts, and which one each primitive uses.
# On the ladder both sides perform the same arithmetic; off it, both call a `pow` and the last bit
# belongs to whichever library each language reached. See item 3 in the header.
LADDER = (-1.0, 0.0, 0.5, 1.0, 2.0)
EXPONENT = {
    "contact_potential": lambda a: a + 1.0,
    "contact_force_elastic": lambda a: a,
    "contact_stiffness": lambda a: a - 1.0,
}
# A *detection* threshold, not a measurement. The one machine that has been read separated
# by 4.0e-16 relative, i.e. between 1.8 and 3.6 ulp depending on where the value sits in
# its binade -- and that quantity has just been shown to vary with the CPU, so a bar set
# near it would be decided by the runner in exactly the way section 21.6 warns about.
# Nothing is bought by tightening it: a transcription error is an O(1) relative
# difference, which is ~1e16 ulp. The printed count is what tracks drift.
ULP_BAR = 64


def _worst_ulp(a, b):
    """Worst elementwise separation between two float arrays, in units in the last place."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(a), np.abs(b))
    gap = np.where(a == b, 0.0, np.abs(a - b) / np.spacing(np.maximum(scale, np.finfo(float).tiny)))
    return float(np.max(gap))

K_DEFAULT = 1.0e6
K_STIFF = 1.0e8         # far enough from the identity Jacobian for the LU to reach the answer
TOL = 1e-12
KSTEP = 1.0 / 48000.0


def _penetrations(n, seed=20260827):
    """Penetrations spanning free flight, grazing and deep contact, at realistic magnitudes.

    Deliberately not powers of two and not a linspace: section 14.3's finding is that a fixture
    built from exact binary values cannot see a rounding difference at all, because the products
    never round.
    """
    rng = np.random.default_rng(seed)
    mag = rng.choice([1e-6, 1e-4, 1e-2, 1.0], n)
    return rng.standard_normal(n) * mag


def rs_solve_contact_vector(eta_free, eta_prev, G, K, alpha, lam_h, k, *, tol, seed,
                            newton_tol=1e-13, maxiter=60):
    eta, f, iters, residual, converged = physsynth_rs.solve_contact_vector(
        np.ascontiguousarray(eta_free, dtype=float),
        np.ascontiguousarray(eta_prev, dtype=float),
        np.ascontiguousarray(G, dtype=float),
        K, alpha, lam_h, k, tol=tol,
        seed=np.ascontiguousarray(seed, dtype=float),
        newton_tol=newton_tol, maxiter=maxiter,
    )
    if not converged:  # pragma: no cover - the fixtures here all converge
        warnings.warn("vector contact solve did not converge", stacklevel=2)
    return eta, f, iters


@contextlib.contextmanager
def _vector_solve(fn):
    """Pin the vector contact solve to ``fn`` for the duration of the block.

    Both sides are pinned explicitly, and that is not tidiness. This file must give the same answer
    whether or not ``PHYSSYNTH_RS`` is set -- and with the flag set the module-level name is
    *already* the Rust one, so a comparison that only swapped the Rust side would be comparing Rust
    against Rust and passing for the wrong reason. That is the same empty-assertion shape the CI
    job's bare ``import physsynth_rs`` exists to catch one level up.

    What is deliberately NOT pinned is the string's banded Cholesky. Under the flag both runs use
    the Rust one (section 15), which is consistent and therefore harmless; what would confound the
    reading is one run using each.
    """
    saved = C.solve_contact_vector
    C.solve_contact_vector = fn
    try:
        yield
    finally:
        C.solve_contact_vector = saved


def python_vector_solve():
    return _vector_solve(C.solve_contact_vector_py)


def rust_vector_solve():
    return _vector_solve(rs_solve_contact_vector)


@contextlib.contextmanager
def _scalar_solve(fn):
    """Pin the scalar contact solve, in ``collision`` and in ``mallet``'s re-export. Both sides are
    pinned, for the reason :func:`_vector_solve` gives."""
    saved_c, saved_m = C.solve_contact, M.solve_contact
    C.solve_contact = fn
    M.solve_contact = fn
    try:
        yield
    finally:
        C.solve_contact, M.solve_contact = saved_c, saved_m


def python_scalar_solve():
    return _scalar_solve(C.solve_contact_py)


def rust_scalar_solve():
    return _scalar_solve(physsynth_rs.solve_contact)


# -- 1. the primitives: bit-identical on both power paths ----------------------------------------

PRIMITIVES = ["contact_potential", "contact_force_elastic", "contact_stiffness"]


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0, 2.3, 3.0])
@pytest.mark.parametrize("name", PRIMITIVES)
def test_primitives_agree_on_the_array_path(name, alpha):
    """Exact where NumPy shortcuts the exponent, within a few ulp where it calls its own ``pow``.

    The split is not a hedge. On the ladder the two sides run the *same* multiply, so equality is
    a proof and a failure is a transcription bug. Off it, equality would be a claim about which
    processor ran the job -- see item 3 in the header for the two CI runs that established this --
    so the assertion becomes an ulp bound and the measured separation is printed.
    """
    eta = _penetrations(20000)
    py = np.asarray(getattr(C, name + "_py")(eta, K_DEFAULT, alpha), dtype=float)
    rs = np.asarray(getattr(physsynth_rs, name)(eta, K_DEFAULT, alpha), dtype=float)
    e = EXPONENT[name](alpha)
    if e in LADDER:
        np.testing.assert_array_equal(rs, py)
        return
    ulp = _worst_ulp(rs, py)
    n = int(np.sum(rs != py))
    print(f"{name} at exponent {e}: {n} of 20000 differ, worst {ulp:.1f} ulp on this math library")
    assert ulp <= ULP_BAR, f"{name} at exponent {e} separated by {ulp:.1f} ulp"


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0, 2.3, 3.0])
@pytest.mark.parametrize("name", PRIMITIVES)
def test_primitives_are_bit_identical_on_the_scalar_path(name, alpha):
    # A float in, not an array: NumPy takes a different code path and so must the port. The two
    # are not interchangeable -- see the next test, which is what makes this one non-redundant.
    py_fn, rs_fn = getattr(C, name + "_py"), getattr(physsynth_rs, name)
    for eta in _penetrations(2000):
        x = float(eta)
        assert float(rs_fn(x, K_DEFAULT, alpha)) == float(py_fn(x, K_DEFAULT, alpha))


@pytest.mark.parametrize("alpha,exponent", [(1.0, "alpha + 1 = 2"), (1.5, "alpha - 1 = 0.5")])
def test_the_two_power_paths_really_do_differ_in_python(alpha, exponent):
    """The finding the port had to be built around, asserted against NumPy rather than assumed.

    NumPy's float64 ``power`` ufunc loop shortcuts the exponents -1, 0, 0.5, 1 and 2; its scalar
    path calls the C library's ``pow``. If this ever stops being true, ``PowPath`` in the Rust
    module has become a distinction without a difference and the two spellings should collapse --
    so this test is here to *notice* that, not because the project wants the discrepancy.
    """
    eta = np.abs(_penetrations(200000, seed=3))
    arr = np.asarray(C.contact_stiffness_py(eta, K_DEFAULT, alpha), dtype=float)
    sca = np.array([float(C.contact_stiffness_py(float(x), K_DEFAULT, alpha)) for x in eta])
    arr_pot = np.asarray(C.contact_potential_py(eta, K_DEFAULT, alpha), dtype=float)
    sca_pot = np.array([float(C.contact_potential_py(float(x), K_DEFAULT, alpha)) for x in eta])
    differ = int(np.sum(arr != sca)) + int(np.sum(arr_pot != sca_pot))
    assert differ > 0, (
        f"NumPy's array and scalar power paths agreed everywhere at {exponent}; the Rust side "
        "carries two spellings on the premise that they do not"
    )


@pytest.mark.parametrize("lam_h", [0.0, 2.0e4])
def test_the_vector_force_and_derivative_are_bit_identical(lam_h):
    """``alpha = 1.0``: every exponent on the ladder, so this is exact -- the cause-separator.

    The fixture is deliberately adversarial -- ``eta_next - eta_prev`` runs right down to the
    ``tol`` that picks the discrete gradient's Taylor branch -- because that is where a
    transcription error in the branch condition would show. It can only be asked here, at the
    exponent where both sides do the same arithmetic; see the next test for why.
    """
    en = _penetrations(20000)
    ep = en + _penetrations(20000, seed=99) * 1e-2
    np.testing.assert_array_equal(
        physsynth_rs.force_total_vec(en, ep, K_DEFAULT, 1.0, lam_h, KSTEP, TOL),
        C.force_total_vec_py(en, ep, K_DEFAULT, 1.0, lam_h, KSTEP, TOL),
    )
    np.testing.assert_array_equal(
        physsynth_rs.deriv_total_vec(en, ep, K_DEFAULT, 1.0, lam_h, KSTEP, TOL),
        C.deriv_total_vec_py(en, ep, K_DEFAULT, 1.0, lam_h, KSTEP, TOL),
    )


@pytest.mark.parametrize("alpha", [1.5, 2.3])
@pytest.mark.parametrize("lam_h", [0.0, 2.0e4])
def test_the_vector_force_and_derivative_agree_where_the_expression_is_conditioned(alpha, lam_h):
    """Off the ladder, and therefore a bound rather than an equality -- but the bound has to be
    read against the *expression*, not against the port.

    The discrete gradient divides by ``da = eta_next - eta_prev``, and its derivative divides by
    ``da^2`` after a cancellation of the same order. So a last-bit difference in a ``pow`` is
    amplified by roughly ``1/da^2``, and near the ``tol`` cutoff that is unbounded: measured
    2026-08-27, injecting a one-ulp nudge into the powers at the rate CI observed moves the
    derivative by **15% of its own scale** on the fixture above, and the force by 1.3e-7. Those
    are properties of the formula that both implementations share, not a disagreement between
    them, and no elementwise tolerance can tell the two apart there.

    So this fixture keeps ``|da|`` away from ``tol``, where the same injection reads 6.1e-14 on
    the force and 6.3e-10 on the derivative -- the ``1/da^2`` scaling is clean, a decade of floor
    buys two decades of agreement. The near-``tol`` regime is covered exactly by the test above.
    A transcription error is an O(1) relative difference and is caught by either bar.
    """
    en = _penetrations(20000)
    gap = _penetrations(20000, seed=99) * 1e-2
    gap = np.where(np.abs(gap) < 1e-3, np.copysign(1e-3, np.where(gap >= 0.0, 1.0, -1.0)), gap)
    ep = en + gap
    for fn, bar in (("force_total_vec", 1e-12), ("deriv_total_vec", 1e-8)):
        py = getattr(C, fn + "_py")(en, ep, K_DEFAULT, alpha, lam_h, KSTEP, TOL)
        rs = getattr(physsynth_rs, fn)(en, ep, K_DEFAULT, alpha, lam_h, KSTEP, TOL)
        worst = float(np.max(np.abs(rs - py)) / np.max(np.abs(py)))
        print(f"{fn} at alpha={alpha}, lam_h={lam_h}: worst {worst:.2e} of scale")
        assert worst <= bar, f"{fn} at alpha={alpha} separated by {worst:.2e} of its scale"


# -- 2. the scalar solve: bit-identical, through a whole model ------------------------------------

@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.3])
@pytest.mark.parametrize("lam_h", [0.0, 1.0e3])
def test_the_scalar_solve_is_bit_identical(alpha, lam_h):
    seeds = _penetrations(3000)
    frees = _penetrations(3000, seed=5)
    for s, fr in zip(seeds, frees, strict=True):
        args = (float(fr), float(fr) - 3e-5, 1e-8, K_DEFAULT, alpha, lam_h, KSTEP)
        kw = dict(tol=TOL, seed=float(s), newton_tol=1e-14, maxiter=60)
        py = C.solve_contact_py(*args, **kw)
        rs = physsynth_rs.solve_contact(*args, **kw)
        assert rs[0] == py[0] and rs[1] == py[1] and rs[2] == py[2]


@pytest.mark.parametrize("mk,kw", [
    (make_mallet, {}),
    (make_mallet, {"alpha": 1.0}),
    (make_mallet, {"hysteresis": 1.0e3}),
    (make_mallet_wall, {}),
    (make_mallet_wall, {"alpha": 2.3}),
])
def test_the_mallet_trajectory_is_bit_identical(mk, kw):
    """The scalar solve driven through its real client, which is where the bracket fallback fires.

    Measured 2026-08-27: the fallback fires once per 3,000 steps in the flagship configuration and
    eight times at ``alpha = 1``. So this run, unlike a sweep over synthetic inputs, actually
    exercises the transcribed ``linspace`` scan and ``brentq`` -- and it asserts the *count* as
    well as the trajectory, because taking the fallback a different number of times would be a
    branch difference that the penetrations might still absorb.
    """
    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = mk(**kw)
            eta, force, fb = [], [], []
            for _ in range(SHORT_RUN):
                m.step()
                eta.append(float(m.penetration))
                force.append(float(m.contact_force))
                fb.append(m.fallbacks)
        return np.array(eta), np.array(force), np.array(fb)

    with python_scalar_solve():
        py = run()
    with rust_scalar_solve():
        rs = run()
    np.testing.assert_array_equal(rs[2], py[2])  # the fallback fired at the same steps
    np.testing.assert_array_equal(rs[0], py[0])
    np.testing.assert_array_equal(rs[1], py[1])


# -- 3. the dense LU, on its own ------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2, 5, 20, 79])
def test_the_dense_lu_picks_the_same_pivots_as_lapack(m):
    """The pivot sequence is a *discrete* choice, and it must match even though the arithmetic
    cannot. A different pivot is a different elimination, which would separate the trajectories by
    far more than rounding does -- so this is the one property of the LU held to equality."""
    rng = np.random.default_rng(11 + m)
    A = np.ascontiguousarray(np.eye(m) + rng.standard_normal((m, m)) * 0.3)
    _, piv_py = sp_lu_factor(A)
    _, piv_rs, info = physsynth_rs.lu_factor(A)
    assert info == 0
    np.testing.assert_array_equal(piv_rs, piv_py)


@pytest.mark.parametrize("m", [1, 2, 5, 20, 79])
def test_the_dense_solve_agrees_with_lapack_to_group_a(m):
    rng = np.random.default_rng(29 + m)
    A = np.ascontiguousarray(np.eye(m) + rng.standard_normal((m, m)) * 0.3)
    b = rng.standard_normal(m)
    lu_py, piv_py = sp_lu_factor(A)
    lu_rs, piv_rs, _ = physsynth_rs.lu_factor(A)
    x_py = sp_lu_solve((lu_py, piv_py), b)
    x_rs = physsynth_rs.lu_solve(lu_rs, piv_rs.astype(np.int64), b)
    scale = max(float(np.max(np.abs(x_py))), 1e-300)
    assert np.max(np.abs(x_rs - x_py)) <= GROUP_A_TOL * scale


def test_an_identity_system_is_solved_exactly():
    """The state the barrier spends most of its life in: no node in contact, so the Jacobian is
    exactly ``I``. Both sides must return the right-hand side unchanged, or a barriered string
    stops being bit-for-bit a bare one."""
    m = 8
    A = np.ascontiguousarray(np.eye(m))
    b = np.linspace(-0.7, 1.3, m)
    lu_rs, piv_rs, _ = physsynth_rs.lu_factor(A)
    np.testing.assert_array_equal(physsynth_rs.lu_solve(lu_rs, piv_rs.astype(np.int64), b), b)


# -- 4. the vector solve, through the barrier string -----------------------------------------------

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

    Section 5 needs it and section 4 does not: comparing the two *shells* means holding everything
    under them constant, and the banded back-substitution is the one piece a Python string and a
    Rust string would otherwise do differently (§15.3 -- OpenBLAS's blocked ``DTBSV`` against the
    reference one transcribed). ``string_damped`` captures the name at import, which is the hazard
    ``test_stability.py``'s guard watches, so patching the captured binding is the only way to hold
    it. Under ``PHYSSYNTH_RS=1`` this is already the state of the world and the patch is a no-op.
    Same helper as ``test_rust_parity_bow.py``'s.
    """
    saved = (string_damped.cholesky_banded, string_damped.cho_solve_banded)
    string_damped.cholesky_banded = rs_cholesky
    string_damped.cho_solve_banded = rs_cho_solve
    try:
        yield
    finally:
        string_damped.cholesky_banded, string_damped.cho_solve_banded = saved


def _string_kw(*, N=80, lam=0.9, kappa=0.0, sigma0=0.0, sigma1=0.0, theta=THETA_DEFAULT):
    c = wave_speed(T_DEFAULT, RHO_DEFAULT)
    return dict(L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=c * N / (L_DEFAULT * lam), N=N,
                kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta)


def _build(barrier_cls, string_cls, *, N=80, lam=0.9, K=BARRIER_K_DEFAULT,
           alpha=BARRIER_ALPHA_DEFAULT, barrier=BARRIER_HEIGHT_DEFAULT, hysteresis=0.0,
           kappa=0.0, sigma0=0.0, sigma1=0.0, theta=THETA_DEFAULT, newton_tol=1e-13,
           newton_maxiter=60):
    """``helpers.make_barrier_string``, with the two implementations spelled out.

    The helper picks both classes off the swapped module names, which is right for every other
    file and wrong for this one -- see item 5 in the module docstring. The defaults here track the
    helper's, and ``test_the_local_builder_is_the_shipped_fixture`` is what keeps them tracking.
    """
    s = string_cls(**_string_kw(N=N, lam=lam, kappa=kappa, sigma0=sigma0, sigma1=sigma1,
                                theta=theta))
    return barrier_cls(string=s, barrier=barrier, stiffness=K, alpha=alpha,
                       hysteresis=hysteresis, newton_tol=newton_tol,
                       newton_maxiter=newton_maxiter)


def _drive(bar, steps):
    """Set the standard half-sine start and step, recording field, energy and Newton work."""
    x = np.linspace(0.0, 1.0, bar.string.N + 1)
    bar.set_state(5.0e-3 * np.sin(np.pi * x))
    u = np.empty((steps, bar.string.N + 1))
    energy = np.empty(steps)
    iters = np.empty(steps, dtype=int)
    for n in range(steps):
        bar.step()
        u[n] = bar.string.u
        energy[n] = bar.energy()
        iters[n] = bar.newton_iters
    return u, energy, iters


def _barrier_run(steps, **kw):
    """Section 4's rig: the **Python** shell, on whichever string the ambient flag selects.

    Pinning the shell is what keeps the ``solve_contact_vector`` swap below meaningful -- with
    ``PHYSSYNTH_RS`` set, ``collision.BarrierString`` is the Rust class and never looks that name
    up at all. The string is deliberately left ambient: both runs then use the same one, which is
    consistent and therefore harmless, whereas one run per implementation would confound the
    reading.
    """
    return _drive(_build(C.BarrierStringPy, string_damped.DampedStiffString, **kw), steps)


def _point_fret(N=80, node=None):
    b = np.full(N + 1, -np.inf)
    b[node if node is not None else N // 3] = -2.0e-4
    return b


def _two_frets(N=80):
    b = np.full(N + 1, -np.inf)
    b[27], b[54] = -2.0e-4, -2.0e-4
    return b


CASES = {
    "point fret (m=1)": {"N": 80, "barrier": _point_fret(), "lam": 0.4, "K": K_DEFAULT,
                         "alpha": 1.5},
    "two frets (m=2)": {"N": 80, "barrier": _two_frets(), "lam": 0.4, "K": K_DEFAULT,
                        "alpha": 1.5},
    "flat rail (m=79)": {"K": K_DEFAULT, "alpha": 1.5, "lam": 0.4},
    "flat rail alpha=1": {"K": K_DEFAULT, "alpha": 1.0, "lam": 0.4},
    "flat rail stiff": {"K": K_STIFF, "alpha": 1.5, "lam": 0.4},
    "flat rail lossy": {"K": K_DEFAULT, "alpha": 1.5, "lam": 0.4, "hysteresis": 2.0e4},
}


def test_a_single_contact_node_is_bit_identical():
    """The cause-separator. One node means ``G`` is 1x1, so the matvec is a single multiply and the
    LU a scalar divide -- neither can round differently.

    ``alpha = 1.0``, and that is load-bearing rather than incidental: it is the one exponent at
    which "everything else in the solve is shared" is *true*. At the fixture's own 1.5 the two
    sides reach two different ``pow`` implementations and a failure here would mean the runner
    changed, not that the transcription did -- which is exactly how this test read on 2026-08-27.
    See item 3 in the header. The shipped 1.5 keeps its Group A comparison below.
    """
    kw = dict(CASES["point fret (m=1)"], alpha=1.0)
    with python_vector_solve():
        py = _barrier_run(SHORT_RUN, **kw)
    with rust_vector_solve():
        rs = _barrier_run(SHORT_RUN, **kw)
    np.testing.assert_array_equal(rs[0], py[0])
    np.testing.assert_array_equal(rs[1], py[1])


def test_two_contact_nodes_are_bit_identical_too():
    """``m = 2`` adds a real two-term sum to the matvec and a real pivot choice to the LU, and is
    still short enough that neither can reorder. It separates "the transcription is right" from
    "the reduction is too short to disagree". At ``alpha = 1.0``, for the reason above."""
    kw = dict(CASES["two frets (m=2)"], alpha=1.0)
    with python_vector_solve():
        py = _barrier_run(SHORT_RUN, **kw)
    with rust_vector_solve():
        rs = _barrier_run(SHORT_RUN, **kw)
    np.testing.assert_array_equal(rs[0], py[0])


@pytest.mark.parametrize("label", list(CASES))
def test_the_vector_solve_agrees_to_group_a_over_a_short_run(label):
    with python_vector_solve():
        py = _barrier_run(GROUP_A_RUN, **CASES[label])
    with rust_vector_solve():
        rs = _barrier_run(GROUP_A_RUN, **CASES[label])
    amp = float(np.max(np.abs(py[0])))
    assert np.max(np.abs(rs[0] - py[0])) <= GROUP_A_TOL * amp, (
        f"{label}: displacement fields separated by more than the Group A target"
    )
    # The iteration count is a branch, not a value: an extra Newton step or an extra Armijo
    # backtrack would be an O(1) trajectory change that the tolerance above might still absorb.
    # Section 15.9 flagged that nothing in the repo compares these; this does.
    np.testing.assert_array_equal(rs[2], py[2])


@pytest.mark.parametrize("label", list(CASES))
def test_the_energy_bars_hold_on_both_implementations(label):
    """What survives past the Group A window: the physics. Both sides are held to CLAUDE.md's bar
    on the same run, so a regression in either shows up as a bar failure rather than as a
    comparison that quietly got looser."""
    kw = dict(CASES[label], sigma0=0.0, sigma1=0.0, hysteresis=0.0)
    with python_vector_solve():
        py_e = _barrier_run(SHORT_RUN, **kw)[1]
    with rust_vector_solve():
        rs_e = _barrier_run(SHORT_RUN, **kw)[1]
    for name, e in (("SciPy", py_e), ("Rust", rs_e)):
        drift = abs(e[-1] - e[0]) / abs(e[0])
        assert drift < DRIFT_TOL, f"{label} on {name}: lossless energy drifted {drift:.2e}"


def _live_jacobian(**kw):
    """Run the barrier a while and hand back a Jacobian of the shape this fixture builds, taken at
    its deepest penetration and evaluated at **coincident** penetrations.

    Not a synthetic matrix: the point below is about the matrices *this model* builds, and their
    distance from the identity is set by the fixture's contact stiffness.

    Coincident (``eta_next == eta_prev``) is a deliberate simplification and not the exact matrix
    the solve factors -- that one uses the previous timestep's penetration, so it takes the
    quotient branch of the derivative rather than the Taylor branch. The two agree in the limit and
    the conclusion below is about an order of magnitude, not a digit. It matters only in that the
    thresholds are calibrated against *this* construction: swapping in a real ``eta_prev`` would
    move them for a reason that has nothing to do with the port.
    """
    bar = make_barrier_string(**kw)
    x = np.linspace(0.0, 1.0, bar.string.N + 1)
    bar.set_state(5.0e-3 * np.sin(np.pi * x))
    best, best_pen = None, -np.inf
    for _ in range(GROUP_A_RUN):
        bar.step()
        pen = float(np.max(bar.penetration))
        if pen > best_pen:
            fp = C.deriv_total_vec_py(bar.penetration, bar.penetration, bar.K, bar.alpha,
                                      bar.lam_h, bar.k, bar.eta_tol)
            m = bar._G.shape[0]
            best = np.ascontiguousarray(np.eye(m) + bar._G * fp[np.newaxis, :])
            best_pen = pen
    return best


def test_a_soft_contact_hides_the_solver_and_that_is_why_the_stiff_case_is_here():
    """The blind spot, asserted rather than described.

    At the default fixture the Newton Jacobian is within 1e-2 of the identity, so the LU is
    effectively solving ``I d = -r``, its disagreement with LAPACK never reaches the answer, and
    the run comes out bit-identical with 79 nodes in contact. That reading is not evidence the
    solvers match. It is evidence the solver was barely used -- and the second half of this test
    is what says so, by putting the *same* run's own Jacobian in front of both implementations and
    showing they disagree on it.

    Stated that way the claim is deterministic. An earlier version asserted that the stiff
    fixture's *trajectory* separated by a non-zero amount, which is a measurement rather than a
    contract: it depends on the ambient state of ``PHYSSYNTH_RS``, because the flag changes the
    string's banded solve and hence the admittance block the contact solve is handed.

    The soft run keeps ``alpha = 1.5`` and is held to Group A rather than to the bit, and both
    halves of that are forced. It cannot be exact, because off the ladder the last bit belongs to
    the machine (item 3 in the header). And it cannot move to ``alpha = 1.0`` to buy exactness,
    because **the blind spot is a property of the exponent**: measured 2026-08-27 on this
    fixture's own Jacobian, ``cond(J)`` is 1.0032 at ``alpha = 1.5``, 1.0625 at ``alpha = 1.0``
    and 1.0001 at ``alpha = 2.3``. A *higher* exponent hides the solver *better*, because the
    tangent stiffness ``K a eta^(a-1)`` vanishes at grazing contact for ``a > 1`` and is the flat
    constant ``K`` at ``a = 1``. So the linear law is the one case where this rail is not soft in
    the sense that matters, the LU reaches the answer, and the run separates within 2,000 steps --
    which is item 2 arriving from the opposite direction and is why the run below is 1.5.
    """
    soft_kw = CASES["flat rail (m=79)"]
    with python_vector_solve():
        soft = _barrier_run(GROUP_A_RUN, **soft_kw)
    with rust_vector_solve():
        soft_rs = _barrier_run(GROUP_A_RUN, **soft_kw)
    amp = float(np.max(np.abs(soft[0])))
    worst = float(np.max(np.abs(soft_rs[0] - soft[0]))) / amp
    print(f"79 nodes in contact, soft: the two separated by {worst:.2e} of amplitude")
    assert worst <= GROUP_A_TOL, f"the soft rail separated by {worst:.2e} of amplitude"

    jac = _live_jacobian(**CASES["flat rail (m=79)"])
    rhs = np.linspace(-1.0e-5, 3.0e-5, jac.shape[0])
    d_py = sp_lu_solve(sp_lu_factor(jac), rhs)
    lu_rs, piv_rs, _ = physsynth_rs.lu_factor(jac)
    d_rs = physsynth_rs.lu_solve(lu_rs, piv_rs.astype(np.int64), rhs)
    assert not np.array_equal(d_py, d_rs), (
        "the two dense solves agreed to the bit on a real 79x79 Newton Jacobian, which would make "
        "the bit-identical trajectory above uninformative in the opposite direction -- either the "
        "LU stopped being the one under test, or SciPy started using this one"
    )
    assert np.max(np.abs(d_rs - d_py)) <= GROUP_A_TOL * float(np.max(np.abs(d_py)))


@pytest.mark.parametrize("label", ["flat rail (m=79)", "flat rail stiff"])
def test_the_jacobian_distance_from_the_identity_is_what_decides(label):
    """The measurement behind the blind spot, kept as a test so it cannot silently stop being true.

    The default fixture's Jacobian is within 1e-2 of the identity; the stiff one's is an order of
    magnitude further away. That distance -- not the number of contact nodes, which is *larger* in
    the soft case -- is what decides whether the dense solve reaches the answer.
    """
    jac = _live_jacobian(**CASES[label])
    off = float(np.max(np.abs(jac - np.eye(jac.shape[0]))))
    if label == "flat rail (m=79)":
        assert off < 1.0e-2, f"the soft fixture stopped being near-identity: {off:.3e}"
    else:
        assert off > 5.0e-2, f"the stiff fixture stopped exercising the solve: {off:.3e}"


def test_an_out_of_reach_barrier_is_bit_identical_to_a_bare_string_on_rust_too():
    """The one bit-identity anchor ``collision`` owns is blind to everything this batch added.

    With the barrier at -100 the contact derivative is exactly zero, so the Jacobian is the
    identity and the LU factors ``I``. That anchor is green under *any* correct dense LU -- which
    is worth writing down, because it is the test a reader would otherwise assume covers the
    solver. It is reproduced here on the Rust side so the anchor is at least known to hold there.
    """
    N, steps, amp = 80, 500, 5.0e-3
    x = np.linspace(0.0, 1.0, N + 1)
    phi = amp * np.sin(np.pi * x)
    bare = make_damped_string(N=N, lam=0.9, kappa=0.0, sigma0=0.3)
    bare.set_state(phi.copy())

    with rust_vector_solve():
        bar = make_barrier_string(N=N, lam=0.9, kappa=0.0, sigma0=0.3, barrier=-100.0)
        bar.set_state(phi.copy())
        for _ in range(steps):
            bare.step()
            bar.step()
        assert np.all(bar.contact_force == 0.0)
        np.testing.assert_array_equal(bar.string.u, bare.u)


# -- 5. the model shell: BarrierString itself ------------------------------------------------------
#
# Everything above compares a piece of the barrier with the rest of it held in Python. This section
# compares the whole model: `BarrierStringPy` on a Python string against the Rust
# `physsynth_rs.BarrierString` on a Rust one, with the banded solver shared so that the only
# thing varying is the shell. What the
# shell does is construction (broadcast the profile, pick the support, solve `m` admittance columns,
# form `k**2/rho`), the two penetration gathers, the force injection, and the barrier's potential
# energy. Three of those five are new arithmetic; see items 5 and 6 in the module docstring.

def _pow_multiply_witness():
    """The first sample rate where `k ** 2 / rho` and `k * k / rho` differ, or None.

    Searched rather than hardcoded, and that is item 3's lesson applied one function along: which
    arguments the C library's `pow` rounds differently from a multiply is a property of the runner,
    so pinning a value measured here would be a bar decided by the machine (plan section 21.6).
    On the machine this was written on, 86 of 200,000 sample rates in this range separate the two
    powers -- but the predicate here is the WHOLE expression, division included, because the first
    witness a `k ** 2 != k * k` sweep returned had its difference absorbed by the `/ rho` and left
    the negative control below comparing a value against itself. The Rust twin of this function
    carries a third scar the Python one cannot have: written with a literal exponent it found
    nothing at all under `--release`, because LLVM folds `powf(x, 2.0)` into `x * x` and the
    search's own predicate became a tautology (plan section 17.2, third occurrence).
    """
    fs = 20000.0
    for _ in range(400000):
        k = 1.0 / fs
        if k ** 2 / RHO_DEFAULT != k * k / RHO_DEFAULT:
            return fs
        fs += 0.7
    return None


def _pair(**kw):
    """The two shells on the same fixture, each on its own implementation's string."""
    py = _build(C.BarrierStringPy, DampedStiffStringPy, **kw)
    rs = _build(physsynth_rs.BarrierString, physsynth_rs.DampedStiffString, **kw)
    return py, rs


def test_the_local_builder_is_the_shipped_fixture():
    """`_build` duplicates `helpers.make_barrier_string`, and this is what keeps the copy honest.

    The duplication is forced -- the helper reads both classes off the swapped module names, which
    is exactly what this file must not do -- but a fixture that drifts from the shipped one is a
    parity test measuring something nobody runs. Bit-for-bit on the derived scalars and on 200
    steps of the field, so a changed default in `helpers` fails here rather than silently.
    """
    ship = make_barrier_string()
    mine = _build(type(ship), type(ship.string))
    for name in ("K", "alpha", "lam_h", "k", "eta_tol", "newton_tol", "newton_maxiter"):
        assert getattr(mine, name) == getattr(ship, name), f"{name} drifted from helpers'"
    np.testing.assert_array_equal(mine._b, ship._b)
    np.testing.assert_array_equal(mine._support, ship._support)
    np.testing.assert_array_equal(mine._G, ship._G)
    np.testing.assert_array_equal(_drive(mine, 200)[0], _drive(ship, 200)[0])


# -- 5a. construction ------------------------------------------------------------------------------

@pytest.mark.parametrize("label", list(CASES))
def test_the_admittance_block_is_bit_identical(label):
    """`G` and the correction columns are `m` banded solves and one scalar multiply each.

    Under a shared solver they must match to the bit -- construction happens before any contact,
    so nothing here has passed through the vector solve yet. This is the cause-separator for
    everything in 5b: a trajectory that separates on a fixture whose `G` already differs would be
    telling you about the string, not about the shell.
    """
    with shared_solver():
        py, rs = _pair(**CASES[label])
    np.testing.assert_array_equal(rs._G, py._G)
    np.testing.assert_array_equal(rs._cols_mat, py._cols_mat)
    np.testing.assert_array_equal(rs._b, py._b)
    np.testing.assert_array_equal(rs._support, py._support)
    np.testing.assert_array_equal(rs._int_idx, py._int_idx)
    assert rs._force_pref == py._force_pref


def test_the_force_prefactor_is_a_pow_not_a_multiply():
    """`force_pref = string.k ** 2 / string.rho` is `float.__pow__`, i.e. the C library's `pow`.

    Not `k * k`, which is a different double for a small fraction of sample rates -- and `bow.py`
    writes `self.k * self.k` at the same spot, so the two models' ports have to spell their
    prefactor differently to stay faithful to their own originals. Every shipped fixture's `fs`
    happens to be a value where the two agree, which is what makes this test necessary rather than
    redundant: it goes and finds an `fs` where they do not.
    """
    fs = _pow_multiply_witness()
    if fs is None:  # pragma: no cover - not seen on any machine yet
        pytest.skip("this machine's pow never separates k**2 from k*k in 20-300 kHz")
    k = 1.0 / fs
    rs = physsynth_rs.BarrierString(
        string=physsynth_rs.DampedStiffString(
            L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=fs, N=80, theta=THETA_DEFAULT,
        ),
        barrier=BARRIER_HEIGHT_DEFAULT, stiffness=BARRIER_K_DEFAULT, alpha=1.0,
    )
    assert rs._force_pref == k ** 2 / RHO_DEFAULT
    assert rs._force_pref != k * k / RHO_DEFAULT


@pytest.mark.parametrize("kwargs, message", [
    ({"stiffness": 0.0}, "contact stiffness K must be > 0."),
    ({"stiffness": -1.0}, "contact stiffness K must be > 0."),
    ({"alpha": 0.5}, "contact exponent alpha must be >= 1."),
    ({"hysteresis": -1.0}, "hysteresis lambda_h must be >= 0."),
])
def test_both_reject_the_same_construction_with_the_same_text(kwargs, message):
    good = dict(barrier=BARRIER_HEIGHT_DEFAULT, stiffness=BARRIER_K_DEFAULT, alpha=1.5,
                hysteresis=0.0)
    for cls, string_cls in ((C.BarrierStringPy, DampedStiffStringPy),
                            (physsynth_rs.BarrierString, physsynth_rs.DampedStiffString)):
        with pytest.raises(ValueError, match=re.escape(message)):
            cls(string=string_cls(**_string_kw()), **{**good, **kwargs})


def test_both_refuse_a_barrier_with_no_finite_interior_node():
    """A profile that is `-inf` everywhere has an empty support, and the original refuses it at
    construction rather than handing an empty system to the solve."""
    b = np.full(81, -np.inf)
    for cls, string_cls in ((C.BarrierStringPy, DampedStiffStringPy),
                            (physsynth_rs.BarrierString, physsynth_rs.DampedStiffString)):
        with pytest.raises(ValueError, match="empty contact support"):
            cls(string=string_cls(**_string_kw()), barrier=b, stiffness=BARRIER_K_DEFAULT)


def test_the_endpoints_are_outside_the_support_on_both_sides():
    """A finite barrier at node 0 or node N is ignored: the support is the *interior*, because the
    two end nodes are clamped and cannot move into anything. Worth pinning because it is the one
    place the port had to reproduce a `range` rather than an expression."""
    b = np.full(81, -np.inf)
    b[0], b[80], b[40] = -1.0e-4, -1.0e-4, -1.0e-4
    with shared_solver():
        py = C.BarrierStringPy(string=DampedStiffStringPy(**_string_kw()), barrier=b,
                               stiffness=BARRIER_K_DEFAULT)
        rs = physsynth_rs.BarrierString(
            string=physsynth_rs.DampedStiffString(**_string_kw()), barrier=b,
            stiffness=BARRIER_K_DEFAULT,
        )
    np.testing.assert_array_equal(rs._support, py._support)
    assert list(py._support) == [40]


def test_the_rust_barrier_refuses_a_python_string():
    """The reed's rule (plan section 12.8) for the reed's reason: a Rust model reporting Rust while
    driving a Python resonator is the green-and-meaningless run the swap guard exists to prevent.
    """
    with pytest.raises(TypeError, match="needs a Rust DampedStiffString"):
        physsynth_rs.BarrierString(
            string=DampedStiffStringPy(**_string_kw()), barrier=BARRIER_HEIGHT_DEFAULT,
            stiffness=BARRIER_K_DEFAULT,
        )


def test_a_wrong_shaped_barrier_is_refused_the_same_way():
    """The original never reaches its own shape check -- `np.broadcast_to` fails first -- so the
    message a caller sees is NumPy's, and the port quotes it rather than improving on it."""
    b = np.zeros(5)
    for cls, string_cls in ((C.BarrierStringPy, DampedStiffStringPy),
                            (physsynth_rs.BarrierString, physsynth_rs.DampedStiffString)):
        with pytest.raises(ValueError, match="could not be broadcast"):
            cls(string=string_cls(**_string_kw()), barrier=b, stiffness=BARRIER_K_DEFAULT)


# -- 5b. the trajectory ----------------------------------------------------------------------------

def test_the_shell_is_bit_identical_at_one_contact_node():
    """`m = 1`: the solve's matvec is one multiply and so is the injection's. Nothing in either
    implementation can round differently, so this must be exact -- and if it ever is not, the
    transcription is wrong and no story about BLAS applies. `alpha = 1.0` for item 3's reason."""
    kw = dict(CASES["point fret (m=1)"], alpha=1.0)
    with shared_solver():
        py, rs = _pair(**kw)
        py_run, rs_run = _drive(py, SHORT_RUN), _drive(rs, SHORT_RUN)
    np.testing.assert_array_equal(rs_run[0], py_run[0])
    np.testing.assert_array_equal(rs_run[1], py_run[1])
    np.testing.assert_array_equal(rs_run[2], py_run[2])


def test_the_shell_is_bit_identical_at_two_contact_nodes():
    """`m = 2`, and this one is exact for a reason that is NOT "the arithmetic agrees".

    The force-injection matvec genuinely differs here -- the test below measures it -- and the
    state cannot see it, because the correction is a small enough fraction of `u` that one of its
    ulps falls off the end of the addition. Both facts are asserted, in that order, so nobody
    later reads this pass as evidence the two matvecs are the same computation.
    """
    kw = dict(CASES["two frets (m=2)"], alpha=1.0)
    with shared_solver():
        py, rs = _pair(**kw)
        py_run, rs_run = _drive(py, SHORT_RUN), _drive(rs, SHORT_RUN)
    np.testing.assert_array_equal(rs_run[0], py_run[0])
    np.testing.assert_array_equal(rs_run[1], py_run[1])


def _injection_spellings(bar, steps):
    """Step `bar`, comparing `cols_mat @ f` against a left-to-right row sum at every step.

    Returns `(rows, differ, differ_scaled, differ_added, ratio_where_differing)` -- the last being
    every `|force_pref * (cols_mat @ f)| / |u|` at a row where the two spellings disagree. That
    ratio is the whole argument: it is how many of `u`'s last bits one of the correction's last
    bits is worth.
    """
    x = np.linspace(0.0, 1.0, bar.string.N + 1)
    bar.set_state(5.0e-3 * np.sin(np.pi * x))
    m = bar.penetration.size
    rows = differ = differ_scaled = differ_added = 0
    ratios = []
    for _ in range(steps):
        bar.step()
        f, A, u_int = bar.contact_force, bar._cols_mat, bar.string.u[1:-1]
        blas = A @ f
        ltr = np.zeros(A.shape[0])
        for j in range(m):
            ltr = ltr + A[:, j] * f[j]
        sb, sl = bar._force_pref * blas, bar._force_pref * ltr
        live = np.abs(u_int) > 0.0
        d = (blas != ltr) & live
        rows += int(live.sum())
        differ += int(d.sum())
        differ_scaled += int(np.count_nonzero((sb != sl) & live))
        differ_added += int(np.count_nonzero(((u_int + sb) != (u_int + sl)) & live))
        if d.any():
            ratios.append(np.abs(sb[d]) / np.abs(u_int[d]))
    return rows, differ, differ_scaled, differ_added, (
        np.concatenate(ratios) if ratios else np.zeros(0)
    )


def test_two_terms_cannot_disagree_without_cancelling_and_so_cannot_reach_the_state():
    """The mechanism behind the exact two-node claim -- and it is a mechanism, not a coincidence.

    A two-term sum is reordered by BLAS and by a `for` loop into the same value unless the two
    terms **cancel**; and where they cancel the sum is tiny, so the correction is tiny, so its
    last bit is far below the field's. Measured 2026-08-27 over 2,000 steps at this fixture: the
    matvec disagrees on 1,291 rows, and at every one of them the correction is at most **9.3e-13**
    of `u` -- i.e. one of its ulps is about `1e-12` of one of `u`'s. Nothing that small can survive
    an addition, and none of them does.

    Stated this way the claim is falsifiable in the right place. It is not "the two matvecs agree"
    (they do not) and not "the difference happened to vanish" (it cannot); it is that a two-term
    reduction's error is *correlated with its own smallness*, which is a property of the length 2.
    The companion test below is the control: at 79 terms that correlation is gone.
    """
    kw = dict(CASES["two frets (m=2)"], alpha=1.0)
    with shared_solver():
        rows, differ, scaled, added, ratio = _injection_spellings(
            _build(C.BarrierStringPy, DampedStiffStringPy, **kw), SHORT_RUN
        )
    if differ == 0:  # pragma: no cover - not seen on any machine yet
        pytest.skip(
            "this machine's BLAS sums a two-element matvec exactly left to right, so there is no "
            "difference here to be absorbed -- the exact claim above is then trivially safe and "
            "this test has nothing to demonstrate. Which kernel OpenBLAS picks is a property of "
            "the CPU (section 14.2), so this is a skip rather than a failure"
        )
    print(f"m=2 injection: {differ}/{rows} rows differ, {scaled} survive the scaling, "
          f"{added} survive the add; |correction|/|u| where they differ <= {ratio.max():.2e}")
    assert ratio.max() < 1.0e-9, (
        f"where the matvec differs, the correction reached {ratio.max():.2e} of the field -- the "
        "cancellation argument has stopped holding and the exact two-node claim above now rests "
        "on nothing"
    )
    assert added == 0, (
        f"{added} of the injection's differences reached `u`, so the exact two-node claim above "
        "is no longer true; downgrade it to Group A with the count printed rather than 'fixing' "
        "the port, which is faithful"
    )


def test_at_seventy_nine_terms_the_cancellation_argument_does_not_apply():
    """The control, and the reason the test above is worth writing down.

    A 79-term reduction can be reordered into a different double without the result being small,
    so where it differs the correction is an ordinary size -- median `1.2e-4` of `u`, measured
    2026-08-27 -- and roughly one such difference in `1/1.2e-4` crosses a rounding boundary and
    reaches the state. It does: 7 of 14,746 over 2,000 steps, 30 of 44,653 over 6,000, which is
    what that ratio predicts to within a factor of two. So the exact claim at two nodes is about
    the *length of the sum*, and this is where the same code stops being able to make it.

    What is asserted is the *ratio*, not that count -- see the comment below.
    """
    with shared_solver():
        rows, differ, scaled, added, ratio = _injection_spellings(
            _build(C.BarrierStringPy, DampedStiffStringPy, **CASES["flat rail (m=79)"]), SHORT_RUN
        )
    if differ == 0:  # pragma: no cover - not seen on any machine yet
        pytest.skip("this machine's BLAS matvec agrees with a left-to-right sum at m = 79")
    median = float(np.median(ratio))
    print(f"m=79 injection: {differ}/{rows} rows differ, {scaled} survive the scaling, "
          f"{added} survive the add; median |correction|/|u| where they differ = {median:.2e}")
    # The BAR is the eight-order-of-magnitude contrast in the ratio, which is set by the trajectory
    # and the coupling. The COUNT of differences that reach `u` is deliberately printed rather than
    # asserted: it is a handful of rounding-boundary crossings out of tens of thousands of chances,
    # so a bar on it would be a bar on a small integer that the runner's BLAS kernel helps decide --
    # section 21.6's failure. The number above is what tracks drift, the same convention `ULP_BAR`
    # follows at the top of this file.
    assert median > 1.0e-6, (
        f"the 79-node correction shrank to {median:.2e} of the field where the matvec differs, "
        "which would make it the two-node case and leave that test's claim unsupported by a "
        "contrast"
    )


@pytest.mark.parametrize("label", list(CASES))
def test_the_shell_agrees_to_group_a_over_a_short_run(label):
    """At 79 nodes the shell's matvec DOES reach the state -- and the regime does not change,
    because the solve's own matvec was already spending the bit-identity there (item 4).

    Measured 2026-08-27, the shell's contribution alone at 500 steps is at most 1.9e-14 of peak
    against this 1e-13 bar, and it first crosses the bar between steps 1,597 and 3,076 depending
    on fixture -- later than the solve's own window of 1,175-1,584. So the run length below stays
    the one item 4 chose, and it stays chosen by the solve.
    """
    with shared_solver():
        py, rs = _pair(**CASES[label])
        py_run, rs_run = _drive(py, GROUP_A_RUN), _drive(rs, GROUP_A_RUN)
    amp = float(np.max(np.abs(py_run[0])))
    worst = float(np.max(np.abs(rs_run[0] - py_run[0]))) / amp
    print(f"{label}: the two shells separated by {worst:.2e} of amplitude over {GROUP_A_RUN} steps")
    assert worst <= GROUP_A_TOL, f"{label}: the shells separated by more than the Group A target"
    # The iteration count is a branch, not a value -- section 16.6's convention, applied to the
    # model rather than to the solve.
    np.testing.assert_array_equal(rs_run[2], py_run[2])


@pytest.mark.parametrize("label", list(CASES))
def test_the_energy_bars_hold_on_the_rust_model(label):
    """What survives past the Group A window: the physics, on the model as shipped.

    Distinct from the identically-named test in section 4, which held the Python shell to the bar
    on two different solves. This one holds the Rust *model* to it -- so a shell that got the
    two-time-averaged barrier potential subtly wrong shows up here even though every trajectory
    comparison above would still be green.
    """
    kw = dict(CASES[label], sigma0=0.0, sigma1=0.0, hysteresis=0.0)
    with shared_solver():
        e = _drive(_build(physsynth_rs.BarrierString, physsynth_rs.DampedStiffString, **kw),
                   SHORT_RUN)[1]
    drift = abs(e[-1] - e[0]) / abs(e[0])
    assert drift < DRIFT_TOL, f"{label} on the Rust model: lossless energy drifted {drift:.2e}"


def test_the_rust_model_is_passive_under_loss():
    """Hysteresis and string damping both on: the total must decrease monotonically."""
    kw = dict(CASES["flat rail lossy"], sigma0=0.5, sigma1=0.05)
    with shared_solver():
        e = _drive(_build(physsynth_rs.BarrierString, physsynth_rs.DampedStiffString, **kw),
                   1000)[1]
    rise = float(np.max(np.diff(e)))
    assert rise <= 1.0e-12 * abs(e[0]), f"energy rose by {rise:.2e} on a lossy run"


# -- 5c. the interface -----------------------------------------------------------------------------

def test_the_settable_underscored_attributes_round_trip():
    """`_G`, `_force_pref`, `_b` and `penetration` are written by tests and by the viewer.

    Section 12.2's rule -- a leading underscore is not a statement about the interface -- with the
    stronger reading this model forces: three of these four are not merely read but *assigned*,
    and one of them (`_G`) is how `test_collision_modal.py` builds its negative control. A getter
    would have looked complete and left that test unable to run.
    """
    with shared_solver():
        py, rs = _pair(**dict(CASES["two frets (m=2)"], alpha=1.0))
    for bar in (py, rs):
        bar._G = bar._G * 2.0
        bar._force_pref = bar._force_pref * 2.0
        bar._b = np.full_like(bar._b, -1.0e-3)
        bar.penetration = bar._b - bar.string.u[bar._support]
    np.testing.assert_array_equal(rs._G, py._G)
    assert rs._force_pref == py._force_pref
    np.testing.assert_array_equal(rs._b, py._b)
    np.testing.assert_array_equal(rs.penetration, py.penetration)
    # Group A rather than the bit, and the reason is the finding two tests up: doubling the
    # coupling and raising the rail doubles the correction and deepens the contact, which is
    # precisely the regime where the injection's cancelled-and-therefore-tiny correction stops
    # being tiny. Measured here at 3.7e-10 relative -- so this rig can check that both sides
    # *honour* the writes, and cannot check them to the bit.
    py_run, rs_run = _drive(py, 200), _drive(rs, 200)
    amp = float(np.max(np.abs(py_run[0])))
    assert np.max(np.abs(rs_run[0] - py_run[0])) <= 1.0e-9 * amp


def test_the_doubled_coupling_moves_the_trajectory_on_both_sides():
    """The negative control's mechanism, checked here rather than only in the physics suite: the
    two writes above are not decoration, they change what the model computes."""
    with shared_solver():
        base = _drive(_build(physsynth_rs.BarrierString, physsynth_rs.DampedStiffString,
                             **CASES["flat rail (m=79)"]), 200)[0]
        bad = _build(physsynth_rs.BarrierString, physsynth_rs.DampedStiffString,
                     **CASES["flat rail (m=79)"])
        bad._G = bad._G * 2.0
        bad._force_pref = bad._force_pref * 2.0
        moved = _drive(bad, 200)[0]
    assert not np.array_equal(base, moved), "writing `_G` and `_force_pref` changed nothing"


def test_the_observables_and_the_delegated_reads_match():
    """`contact_mask`, `state`, `displacement_at`, `n` and the two per-step arrays."""
    with shared_solver():
        py, rs = _pair(**dict(CASES["two frets (m=2)"], alpha=1.0))
        x = np.linspace(0.0, 1.0, py.string.N + 1)
        for bar in (py, rs):
            bar.set_state(5.0e-3 * np.sin(np.pi * x))
        for i in range(300):
            py.step()
            rs.step()
            np.testing.assert_array_equal(rs.contact_mask(), py.contact_mask(), err_msg=f"{i}")
            assert rs.n == py.n
    np.testing.assert_array_equal(rs.penetration, py.penetration)
    np.testing.assert_array_equal(rs.contact_force, py.contact_force)
    np.testing.assert_array_equal(rs.state, py.state)
    assert rs.displacement_at(40) == py.displacement_at(40)
    assert rs.newton_iters == py.newton_iters


def test_set_state_resets_the_step_count_but_not_the_last_force():
    """The original's `set_state` refreshes the seed and zeroes `n`, and leaves `contact_force` and
    `newton_iters` alone. Faithful rather than tidy: a fixture that reads a force before its first
    step must read the same stale value on both sides."""
    with shared_solver():
        py, rs = _pair(**dict(CASES["two frets (m=2)"], alpha=1.0))
        x = np.linspace(0.0, 1.0, py.string.N + 1)
        for bar in (py, rs):
            bar.set_state(5.0e-3 * np.sin(np.pi * x))
            for _ in range(50):
                bar.step()
        stale_py, stale_rs = py.contact_force.copy(), rs.contact_force.copy()
        for bar in (py, rs):
            bar.set_state(5.0e-3 * np.sin(np.pi * x))
    assert py.n == rs.n == 0
    np.testing.assert_array_equal(rs.contact_force, stale_rs)
    np.testing.assert_array_equal(stale_rs, stale_py)
    np.testing.assert_array_equal(rs.penetration, py.penetration)


def test_the_string_is_the_object_that_was_passed_in():
    """`.string` hands back the caller's object, not a copy -- `web/serialize.py` reads
    `bar.string.u` for every animation frame and `tests/` call `bar.string.set_state`."""
    s = physsynth_rs.DampedStiffString(**_string_kw())
    bar = physsynth_rs.BarrierString(string=s, barrier=BARRIER_HEIGHT_DEFAULT,
                                     stiffness=BARRIER_K_DEFAULT)
    assert bar.string is s
    s.set_state(np.zeros(s.N + 1) + 0.0)
    bar.step()
    assert bar.string is s


def test_a_stalled_solve_warns_with_the_original_text():
    """Capped at one Newton iteration a deep contact cannot converge, so the warning fires.

    The message is byte-for-byte the original's -- including the two `%` formats, which had to be
    reproduced by hand because Rust's `{:e}` is not Python's `{:.2e}`. What cannot match is
    `stacklevel`: the original raises it from inside `solve_contact_vector` with `stacklevel=2`,
    naming `BarrierString.step`, and that frame does not exist once the model is Rust. The Rust
    side names the Python code that called `step()` instead, which is the nearest true statement
    about who to blame. Nothing in the repo asserts the attribution, which is exactly why the
    choice is written down here rather than left to be discovered.
    """
    kw = dict(CASES["flat rail stiff"], alpha=1.0, newton_maxiter=1)
    with shared_solver():
        py, rs = _pair(**kw)
        messages = {}
        for name, bar in (("py", py), ("rs", rs)):
            x = np.linspace(0.0, 1.0, bar.string.N + 1)
            bar.set_state(5.0e-3 * np.sin(np.pi * x))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(200):
                    bar.step()
            messages[name] = [str(w.message) for w in caught
                              if "did not converge" in str(w.message)]
    assert messages["py"], "the Python model stopped stalling -- the fixture no longer bites"
    assert messages["rs"], "the Rust model stalled without warning"
    assert messages["rs"][0].startswith("vector contact solve did not converge in 1 iterations")
    assert messages["rs"][0].endswith(
        "energy may drift. Raise newton_maxiter or oversample the contact."
    )
