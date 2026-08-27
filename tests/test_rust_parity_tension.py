"""Rust vs NumPy/SciPy for the tension-modulated string — model #9.

``docs/dev/rust-migration-plan.md`` §19. The third model out of the four-string chain, and the
first one in the project whose **update matrix moves every step**: the banded factorization sits
inside a scalar root-find rather than at construction, so ``banded`` and ``root`` meet here for the
first time.

**What that changes about parity, and it is the reason this file is shaped differently from
``test_rust_parity_strings.py``.** For models #2 and #3 an implementation difference of one bit
stayed one bit. Here the stretch feeds ``brentq``'s residual, so a last-bit disagreement changes
the **iterate sequence** — an integer, not a tolerance. Measured on the Python side before the port
was written: moving ``_stretch`` from BLAS ``ddot`` to a left-to-right sum changed the per-step
residual-evaluation count on **1,400 of 5,000 steps**. That is why ``portable.dot`` now covers the
stretch too (unconditionally, on the default path) and why every trajectory assertion below can be
``array_equal`` rather than a tolerance.

**Two qualifiers the assertions cannot be stated without.**

* *The shared solver.* As in §18, the exact rows hold only when both sides run the **same** banded
  Cholesky, which ``PHYSSYNTH_RS=1`` arranges and :func:`shared_solver` arranges otherwise. Without
  it SciPy calls OpenBLAS's blocked ``DPBTRF``/``DTBSV`` and Rust runs the reference transcription
  (§15.3). Here that gap is amplified rather than merely carried: it perturbs the residual, so it
  moves the *root*, not only the solve.
* *The fixture.* §16.4's rule, live again. A gentle string is bit-identical **even with the port
  wrong**: measured before the port was written, two implementations differing only in the stretch
  reduction still agree exactly at 20,000 steps at amplitude 1e-4, because ``dT`` is so small that
  the root-find never separates. So every trajectory fixture below is at an amplitude where
  ``brentq`` actually does work, and both halves of that claim are assertions rather than comments:
  :func:`test_the_gentle_fixture_is_not_a_test` shows the blindness (at 1e-6 over 400 steps, the
  cheap form of the same measurement) and
  :func:`test_the_root_find_sees_the_reduction_long_before_the_state_does` shows the chosen
  amplitude is not blind.

**And one detector that is sharper than the trajectory.** ``delta_tension`` caught the batch's one
real porting error — a mis-associated ``_stretch_int`` — while the state stayed bit-identical
through it, because ``beta = k^2 dT / (2 rho)`` is ~1e-9 in every realistic fixture and a last bit
of ``dT`` never reaches a band entry. A parity file that compared only ``u`` would have passed. So
the telemetry is compared **exactly**, and deliberately first.

Green with the flag and without it, per §16.4's convention — the assertions differ between the two
modes, the file does not.
"""

import contextlib
import math

import numpy as np
import pytest

from physsynth.core import banded, string_nonlinear
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_damped import DampedStiffStringPy
from physsynth.core.string_nonlinear import (
    MAX_BRACKET_EXPANSIONS,
    TENSION_TOL_DEFAULT,
    TensionModulatedStringPy,
)
from physsynth.core.string_stiff import THETA_DEFAULT

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13  # the plan's §4 agreement target for a short run
DRIFT_TOL = 1e-10  # CLAUDE.md's acceptance bar, which neither implementation may cross

L_DEF, T_DEF, RHO_DEF = 1.0, 200.0, 0.005
KAPPA_DEF = 1.5
EA_DEF = 8.0 * T_DEF  # EA/T0 = 8: enough hardening that the root-find does real work


def rs_cholesky(ab, lower=False):
    return physsynth_rs.cholesky_banded_upper(np.ascontiguousarray(ab, dtype=float))


def rs_cho_solve(cb_and_lower, b):
    cb, _lower = cb_and_lower
    return physsynth_rs.cho_solve_banded_upper(
        np.ascontiguousarray(cb, dtype=float), np.ascontiguousarray(b, dtype=float)
    )


@contextlib.contextmanager
def shared_solver():
    """Put the Python model on the Rust banded solver for the duration of the block.

    ``string_nonlinear`` captures ``cholesky_banded`` and ``cho_solve_banded`` at import (the
    hazard ``test_stability.py``'s guard watches), so patching the captured names is the only way
    to hold the solver constant while the *model* varies. Under ``PHYSSYNTH_RS=1`` this is already
    the state of the world and the patch is a no-op.

    It matters more here than it did for models #2 and #3: those called the solver once per step,
    so a solver gap was a per-step perturbation. This model calls it ~7 times per step *inside a
    residual*, so a solver gap moves the root the residual is looking for.
    """
    saved = (string_nonlinear.cholesky_banded, string_nonlinear.cho_solve_banded)
    string_nonlinear.cholesky_banded = rs_cholesky
    string_nonlinear.cho_solve_banded = rs_cho_solve
    try:
        yield
    finally:
        string_nonlinear.cholesky_banded, string_nonlinear.cho_solve_banded = saved


def kw(N=64, kappa=KAPPA_DEF, EA=EA_DEF, sigma0=0.0, sigma1=0.0, fs=44100.0,
       theta=THETA_DEFAULT, tension_tol=TENSION_TOL_DEFAULT):
    return dict(
        L=L_DEF, T=T_DEF, rho=RHO_DEF, fs=fs, N=N, kappa=kappa, EA=EA,
        sigma0=sigma0, sigma1=sigma1, theta=theta, tension_tol=tension_tol,
    )


def pair(k):
    """The Python reference and the Rust implementation of the same string."""
    return TensionModulatedStringPy(**k), physsynth_rs.TensionModulatedString(**k)


def mode(s, m=1, amplitude=1e-2):
    return amplitude * np.sin(m * np.pi * s.x / s.L)


def pluck(s, amplitude=1e-2):
    return triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=amplitude)


TELEMETRY = ("delta_tension", "converged", "bracket_expansions", "n_not_converged")


def telemetry(s):
    return tuple(getattr(s, name) for name in TELEMETRY)


# =====================================================================================
# Structure — always exact, flag or no flag
# =====================================================================================


@pytest.mark.parametrize(
    "k",
    [
        kw(),
        kw(N=8, kappa=0.0),
        kw(N=201, kappa=0.4, sigma0=1.2, sigma1=6e-5),
        kw(EA=0.0),
        kw(theta=1.0, EA=40.0 * T_DEF),
        kw(fs=96000.0, tension_tol=1e-9),
    ],
)
def test_parameters_and_grid_are_identical(k):
    a, b = pair(k)
    for name in (
        "L", "T", "rho", "fs", "N", "kappa", "EA", "sigma0", "sigma1", "theta",
        "tension_tol", "c", "h", "k", "lam", "B", "EA_over_T", "boundary",
    ):
        assert getattr(a, name) == getattr(b, name), name
    assert np.array_equal(a.x, b.x)
    assert a.x.dtype == b.x.dtype


@pytest.mark.parametrize("k", [kw(), kw(N=8, kappa=0.0), kw(N=201, kappa=0.4), kw(EA=0.0)])
def test_the_operators_match_to_the_index(k):
    """``_L`` and ``_D2`` down to ``data``, ``indices``, ``indptr`` and ``nnz``.

    ``indices`` is in the list on purpose: §18 found that a CSR matvec sums each row in *stored*
    order, and ``L @ u`` is on this model's update path exactly as it is on model #2's.
    """
    a, b = pair(k)
    for name in ("_L", "_D2"):
        pa, pb = getattr(a, name), getattr(b, name)
        assert pa.shape == pb.shape
        assert pa.nnz == pb.nnz, name
        assert np.array_equal(pa.indptr, pb.indptr), name
        assert np.array_equal(pa.indices, pb.indices), name
        assert np.array_equal(pa.data, pb.data), name
        assert pa.has_sorted_indices and pb.has_sorted_indices, f"{name} must be canonical"


@pytest.mark.parametrize("k", [kw(), kw(N=8, kappa=0.0), kw(N=201, kappa=0.4, sigma0=1.0)])
def test_the_bands_are_identical_and_the_factor_is_the_solvers(k):
    """``_ab0`` and ``_ab_D2`` are pure assembly and match exactly, always.

    ``_chol0`` is whichever Cholesky ran, so it is exact only under a shared solver and held to
    Group A otherwise — §15.3 measured the transcription agreeing with LAPACK on 82 of 120 of this
    family's matrices.
    """
    a, b = pair(k)
    assert np.array_equal(a._ab0, b._ab0)
    assert np.array_equal(a._ab_D2, b._ab_D2)
    with shared_solver():
        c, _ = pair(k)
    assert np.array_equal(c._chol0, b._chol0), "a shared solver must give the same factor"
    assert np.allclose(a._chol0, b._chol0, rtol=0, atol=1e-14)


def test_the_second_band_of_D2_is_empty():
    """``D2`` is tridiagonal, so its second superdiagonal is zeros — the row exists because ``A``
    is pentadiagonal and the two bands are subtracted elementwise."""
    a, b = pair(kw())
    assert np.array_equal(a._ab_D2[0], np.zeros(a.N - 1))
    assert np.array_equal(b._ab_D2[0], np.zeros(a.N - 1))


# =====================================================================================
# Construction rejections — the message text, and the order the checks run in
# =====================================================================================


@pytest.mark.parametrize(
    "bad, message",
    [
        (dict(T=-1.0), "L, T, rho, fs must all be positive."),
        (dict(N=1), "N must be >= 2 (need at least one interior node)."),
        (dict(kappa=-1.0), "kappa (stiffness) must be >= 0."),
        (dict(EA=-1.0), "EA (axial stiffness) must be >= 0."),
        (dict(sigma0=-1.0), "sigma0 (frequency-independent loss) must be >= 0."),
        (dict(sigma1=-1.0), "sigma1 (frequency-dependent loss) must be >= 0."),
        (dict(theta=1.5), "theta must be in (0, 1], got 1.5."),
        (dict(theta=0.0), "theta must be in (0, 1], got 0.0."),
        (dict(tension_tol=0.0), "tension_tol must be > 0."),
    ],
)
def test_rejections_carry_the_same_message(bad, message):
    k = kw()
    k.update(bad)
    with pytest.raises(ValueError) as py_err:
        TensionModulatedStringPy(**k)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.TensionModulatedString(**k)
    assert str(py_err.value) == message
    assert str(rs_err.value) == message


@pytest.mark.parametrize("boundary", ["fixed", "free", "Supported", 7])
def test_the_boundary_rejection_quotes_the_value_with_repr(boundary):
    k = kw()
    k["boundary"] = boundary
    with pytest.raises(ValueError) as py_err:
        TensionModulatedStringPy(**k)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.TensionModulatedString(**k)
    expected = f"boundary must be 'supported', got {boundary!r}."
    assert str(py_err.value) == str(rs_err.value) == expected


def test_the_check_order_is_pythons():
    """A doubly-invalid parameter set must report the *same* one on both sides."""
    k = kw()
    k.update(kappa=-1.0, EA=-1.0, sigma0=-1.0)
    with pytest.raises(ValueError) as py_err:
        TensionModulatedStringPy(**k)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.TensionModulatedString(**k)
    assert str(py_err.value) == str(rs_err.value) == "kappa (stiffness) must be >= 0."


# =====================================================================================
# The consistent start — which carries the nonlinear tension at t = 0
# =====================================================================================


@pytest.mark.parametrize("k", [kw(), kw(EA=0.0), kw(N=8, kappa=0.0), kw(EA=40.0 * T_DEF)])
@pytest.mark.parametrize("v0", [None, 0.0, 2.5, "array"])
def test_set_state_including_u_minus_one(k, v0):
    """``u^{-1}`` includes ``dT_0 = (EA/2L) I(u^0)`` — and at ``EA = 0`` that term is *skipped*,
    not added as a zero, which is what makes the start model #3's rather than close to it."""
    a, b = pair(k)
    u0 = pluck(a)
    v = np.linspace(-1.0, 1.0, a.N + 1) if v0 == "array" else v0
    args = () if v is None else (v,)
    a.set_state(u0.copy(), *args)
    b.set_state(u0.copy(), *args)
    assert np.array_equal(a.u, b.u)
    assert np.array_equal(a.u_prev, b.u_prev)
    assert a.n == b.n == 0
    assert telemetry(a) == telemetry(b)


def test_the_start_ends_clamped_and_u0_is_not_mutated_by_either():
    a, b = pair(kw())
    u0 = pluck(a)
    keep = u0.copy()
    a.set_state(u0)
    assert np.array_equal(u0, keep), "the Python model must not mutate the caller's array"
    u0 = pluck(b)
    keep = u0.copy()
    b.set_state(u0)
    assert np.array_equal(u0, keep), "nor may the Rust one"
    for s in (a, b):
        assert s.u[0] == s.u[-1] == 0.0
        assert s.u_prev[0] == s.u_prev[-1] == 0.0


@pytest.mark.parametrize("shape", [(3,), (200,)])
def test_set_state_rejects_a_wrong_shape_the_same_way(shape):
    a, b = pair(kw())
    for s in (a, b):
        with pytest.raises(ValueError) as err:
            s.set_state(np.zeros(shape))
        assert f"got {shape}" in str(err.value)


# =====================================================================================
# Trajectories — exact under a shared solver, which is the whole claim
# =====================================================================================


TRAJECTORY_FIXTURES = [
    pytest.param(kw(), 1e-2, id="flagship"),
    pytest.param(kw(kappa=0.0), 1e-2, id="flexible"),
    pytest.param(kw(EA=40.0 * T_DEF), 6e-3, id="hard"),
    pytest.param(kw(sigma0=1.5, sigma1=8e-5), 1e-2, id="lossy"),
    pytest.param(kw(N=201, kappa=0.4), 8e-3, id="fine-grid"),
    pytest.param(kw(N=8, kappa=0.0), 2e-2, id="tiny-grid"),
    pytest.param(kw(theta=1.0), 1e-2, id="theta-one"),
    pytest.param(kw(tension_tol=1e-9), 1e-2, id="loose-tol"),
]


@pytest.mark.parametrize("k, amplitude", TRAJECTORY_FIXTURES)
def test_the_trajectory_and_its_telemetry_are_bit_identical(k, amplitude):
    """State, history, energy **and** all four telemetry attributes, step for step.

    The telemetry is asserted first on purpose: ``delta_tension`` is the sharper detector. A
    mis-associated ``_stretch_int`` moves it while leaving ``u`` untouched, because
    ``beta = k^2 dT / (2 rho)`` is ~1e-9 here and a last bit of ``dT`` never reaches a band entry.
    """
    with shared_solver():
        a, b = pair(k)
        u0 = pluck(a, amplitude)
        a.set_state(u0.copy())
        b.set_state(u0.copy())
        for n in range(600):
            a.step()
            b.step()
            assert telemetry(a) == telemetry(b), f"telemetry diverged at step {n}"
            assert np.array_equal(a.u, b.u), f"state diverged at step {n}"
            assert np.array_equal(a.u_prev, b.u_prev), f"history diverged at step {n}"
            assert a.energy() == b.energy(), f"energy diverged at step {n}"
            assert a.nonlinear_energy() == b.nonlinear_energy()
        assert a.n == b.n == 600


def test_the_long_run_stays_bit_identical():
    """20,000 steps. §18.6 catalogued three agreement regimes; this model is a *recurring*
    nonlinearity that is not chaotic below the parametric threshold, so it stays exact — the
    trajectory never separates because there is nothing to separate."""
    with shared_solver():
        a, b = pair(kw())
        u0 = pluck(a, 1e-2)
        a.set_state(u0.copy())
        b.set_state(u0.copy())
        for _ in range(20_000):
            a.step()
            b.step()
        assert np.array_equal(a.u, b.u)
        assert a.energy() == b.energy()
        assert telemetry(a) == telemetry(b)


@pytest.mark.parametrize("k, amplitude", TRAJECTORY_FIXTURES[:4])
def test_without_a_shared_solver_the_gap_stays_inside_group_a(k, amplitude):
    """The other half of the separation: hold the *model* fixed and let the solver differ.

    This is §15.3's gap, and here it is amplified rather than carried — it perturbs the residual,
    so it moves the root ``brentq`` converges on rather than only the solve. The bar is the plan's
    Group A target over a short run, normalised by amplitude (§14.4: a decaying trajectory must
    never be compared pointwise).
    """
    a, b = pair(k)
    u0 = pluck(a, amplitude)
    a.set_state(u0.copy())
    b.set_state(u0.copy())
    scale = float(np.abs(u0).max())
    for _ in range(200):
        a.step()
        b.step()
    assert np.abs(a.u - b.u).max() / scale < GROUP_A_TOL
    assert abs(a.energy() - b.energy()) / abs(a.energy()) < GROUP_A_TOL
    assert a.converged and b.converged


@contextlib.contextmanager
def _stretch_on_np_dot():
    """Put the Python model's stretch back on ``np.dot`` — the spelling §18.2 left it in.

    This is the *pre-port* implementation, kept alive as a probe: it is the thing the Rust side
    cannot reproduce, and the two tests below use it to show what a fixture can and cannot see.
    """
    cls = TensionModulatedStringPy
    saved = (cls._stretch, cls._stretch_int)

    def _stretch(self, u_full):
        du = np.diff(u_full)
        return float(np.dot(du, du) / self.h)

    def _stretch_int(self, u_int):
        du = np.diff(u_int)
        return float((np.dot(du, du) + u_int[0] ** 2 + u_int[-1] ** 2) / self.h)

    cls._stretch, cls._stretch_int = _stretch, _stretch_int
    try:
        yield
    finally:
        cls._stretch, cls._stretch_int = saved


def _run_python(k, u0, steps, np_dot):
    ctx = _stretch_on_np_dot() if np_dot else contextlib.nullcontext()
    with ctx:
        s = TensionModulatedStringPy(**k)
        s.set_state(u0.copy())
        evaluations = 0
        base = s._stretch_int
        def counted(v, _b=base):
            nonlocal evaluations
            evaluations += 1
            return _b(v)
        s._stretch_int = counted
        for _ in range(steps):
            s.step()
        return s.u.copy(), s.delta_tension, evaluations


def test_the_gentle_fixture_is_not_a_test():
    """§16.4's rule, stated as an assertion rather than trusted as a comment.

    At small amplitude the tension excess is so small that ``brentq`` converges the same way
    whatever the stretch reduction is — so a gentle fixture is **blind to the very thing this
    batch had to change**, and would be green with the port's reduction wrong. Which is exactly
    why every trajectory fixture above is at an amplitude where the root-find does real work.
    """
    k = kw()
    s = TensionModulatedStringPy(**k)
    gentle = mode(s, 1, 1e-6)
    a, _, _ = _run_python(k, gentle, 400, np_dot=False)
    b, _, _ = _run_python(k, gentle, 400, np_dot=True)
    assert np.array_equal(a, b), (
        "the gentle fixture stopped being blind — remeasure §19.2 before trusting this file's "
        "fixture choice; it is not the port that changed"
    )


def test_the_root_find_sees_the_reduction_long_before_the_state_does():
    """The other half of the fixture claim, and the sharpest measurement in this file.

    At the flagship amplitude the two spellings of the stretch take a **different number of
    residual evaluations** within the first few hundred steps — an integer, not a tolerance. And
    yet the *state* stays bit-identical for **1,882 steps** (measured 2026-08-27), because
    ``brentq`` mostly converges onto the same double from a different path, and where it does not
    the one-ulp difference in ``dT`` is swallowed by ``beta = k^2 dT / (2 rho) ~ 1e-9`` before it
    reaches a band entry.

    So the trajectory is a **hopelessly weak detector** for this class of error, and the telemetry
    is the sharp one. That is the reason every trajectory test above compares the telemetry first,
    and the reason a Group A tolerance on ``u`` would have been no test at all.

    **The window itself depends on the banded solver**, which is why this test pins one: 1,882
    steps with SciPy's LAPACK and 210 with the Rust transcription, on the same fixture. So the
    assertion below is at 100 steps — comfortably inside the shorter of the two — rather than at a
    number that happens to hold in whichever mode the suite is running.
    """
    k = kw()
    s = TensionModulatedStringPy(**k)
    u0 = pluck(s, 1e-2)
    with shared_solver():
        short_a, _, evals_a = _run_python(k, u0, 100, np_dot=False)
        short_b, _, evals_b = _run_python(k, u0, 100, np_dot=True)
        long_a, _, _ = _run_python(k, u0, 3000, np_dot=False)
        long_b, _, _ = _run_python(k, u0, 3000, np_dot=True)
    assert evals_a != evals_b, (
        "the two stretch spellings took the same number of brentq evaluations — the fixture no "
        "longer exercises the root-find, so remeasure before trusting this file's fixture choice"
    )
    assert np.array_equal(short_a, short_b), (
        "the state separated within 100 steps — the claim that the trajectory is the weaker "
        "detector needs remeasuring; it is not the port that changed"
    )
    assert not np.array_equal(long_a, long_b), "the state must separate eventually"


# =====================================================================================
# The chain anchor: EA = 0 is model #3, across the language boundary
# =====================================================================================


@pytest.mark.parametrize("sigma0, sigma1", [(0.0, 0.0), (2.0, 0.0), (0.0, 1e-4), (1.5, 6e-5)])
def test_ea_zero_is_a_damped_string_bit_for_bit(sigma0, sigma1):
    """The anchor §15.2 found, now with a **Rust** model #9 against a **Python** model #3.

    That is the direction that matters: the Python damped string is still the reference oracle, and
    this is the assertion that would have failed had ``portable.py`` not existed.
    """
    with shared_solver(), _damped_on_the_shared_solver():
        k = kw(EA=0.0, sigma0=sigma0, sigma1=sigma1)
        rs = physsynth_rs.TensionModulatedString(**k)
        py = DampedStiffStringPy(
            L=k["L"], T=k["T"], rho=k["rho"], fs=k["fs"], N=k["N"], kappa=k["kappa"],
            sigma0=sigma0, sigma1=sigma1, theta=k["theta"],
        )
        u0 = pluck(rs, 1e-2)
        rs.set_state(u0.copy())
        py.set_state(u0.copy())
        assert np.array_equal(rs.u_prev, py.u_prev), "the consistent start must be model #3's"
        for n in range(600):
            rs.step()
            py.step()
            assert np.array_equal(rs.u, py.u), f"diverged at step {n}"
            assert rs.energy() == py.energy(), f"energy diverged at step {n}"
            assert rs.nonlinear_energy() == 0.0
            assert rs.delta_tension == 0.0


@contextlib.contextmanager
def _damped_on_the_shared_solver():
    from physsynth.core import string_damped

    saved = (string_damped.cholesky_banded, string_damped.cho_solve_banded)
    string_damped.cholesky_banded = rs_cholesky
    string_damped.cho_solve_banded = rs_cho_solve
    try:
        yield
    finally:
        string_damped.cholesky_banded, string_damped.cho_solve_banded = saved


def test_ea_zero_uses_the_prefactored_matrix_on_both_sides():
    """``EA = 0`` must be the *identical code path*, not a tension solve that happens to return
    zero — which is what makes the anchor exact rather than merely close. The observable is that
    the root-find's telemetry never moves."""
    for s in pair(kw(EA=0.0)):
        s.set_state(pluck(s, 1e-2))
        for _ in range(50):
            s.step()
        assert s.delta_tension == 0.0
        assert s.bracket_expansions == 0
        assert s.n_not_converged == 0
        assert s.converged


# =====================================================================================
# The surface `cargo test` cannot see
# =====================================================================================


def test_apply_Ainv_refuses_identically():
    """The one method the rest of the family implements and this model refuses, because ``A``
    moves with the tension. Three coupled models call it on whatever string they are handed, so it
    has to be a clean ``NotImplementedError`` with the same text — not a panic."""
    a, b = pair(kw())
    messages = []
    for s in (a, b):
        with pytest.raises(NotImplementedError, match="time-varying") as err:
            s.apply_Ainv(np.zeros(s.N - 1))
        messages.append(str(err.value))
    assert messages[0] == messages[1]


def test_the_telemetry_attributes_are_settable_on_both():
    """All four are plain public attributes on the Python class, so a diagnostic script may zero a
    counter mid-run. A read-only property on the Rust side would be a silent interface change."""
    for s in pair(kw()):
        s.bracket_expansions = 5
        s.n_not_converged = 2
        s.converged = False
        s.delta_tension = 1.25
        assert (s.bracket_expansions, s.n_not_converged, s.converged, s.delta_tension) == (
            5, 2, False, 1.25,
        )


def test_state_stretch_and_tension_agree():
    a, b = pair(kw())
    u0 = pluck(a, 1e-2)
    with shared_solver():
        a.set_state(u0.copy())
        b.set_state(u0.copy())
        for _ in range(120):
            a.step()
            b.step()
            assert a.stretch == b.stretch
            assert a.tension == b.tension
            assert a.tension >= a.T, "hardening only — the tension may never fall below T0"
    assert np.array_equal(a.state, b.state)
    assert a.state is not a.u and b.state is not b.u, "`state` is a copy on both sides"


@pytest.mark.parametrize("index", [0, 1, 7, -1, -2])
def test_displacement_at_matches_including_negative_indices(index):
    a, b = pair(kw())
    a.set_state(pluck(a, 1e-2))
    b.set_state(pluck(b, 1e-2))
    assert a.displacement_at(index) == b.displacement_at(index)


@pytest.mark.parametrize("index", [200, -200])
def test_displacement_at_refuses_out_of_range_the_same_way(index):
    for s in pair(kw()):
        with pytest.raises(IndexError):
            s.displacement_at(index)


def test_the_state_arrays_are_writable_and_rebindable():
    """The buffer contract (§9.3): ``u`` and ``u_prev`` are Python-owned arrays a client may write
    through or replace. ``collision`` and the bridges do exactly this on the other strings."""
    for s in pair(kw()):
        s.set_state(pluck(s, 1e-2))
        s.u[3] += 1e-9
        assert s.u[3] != 0.0
        fresh = np.zeros(s.N + 1)
        s.u_prev = fresh
        assert np.array_equal(s.u_prev, fresh)
        s.n = 17
        assert s.n == 17
    # The binding validates the shape on assignment; the Python model is a plain attribute and
    # does not. That asymmetry is deliberate — the Rust side has to reject what it cannot hold —
    # so it is asserted on the Rust object alone rather than papered over.
    rs = physsynth_rs.TensionModulatedString(**kw())
    with pytest.raises(ValueError):
        rs.u = np.zeros(3)


# =====================================================================================
# The physics bars, asserted on the Rust side directly
# =====================================================================================


@pytest.mark.parametrize("amplitude", [1e-3, 1e-2, 3e-2])
def test_rust_conserves_energy_in_a_lossless_run(amplitude):
    """CLAUDE.md's acceptance contract. This is the bar that must hold whatever the comparison
    says, and the nonlinear term has to carry real weight or the test merely re-runs model #3."""
    s = physsynth_rs.TensionModulatedString(**kw(EA=20.0 * T_DEF))
    s.set_state(mode(s, 1, amplitude))
    e0 = s.energy()
    worst = 0.0
    nl_fraction = 0.0
    for _ in range(2000):
        s.step()
        worst = max(worst, abs(s.energy() - e0) / e0)
        nl_fraction = max(nl_fraction, s.nonlinear_energy() / s.energy())
    assert worst < DRIFT_TOL, f"drift {worst:.3e}"
    if amplitude >= 1e-2:
        assert nl_fraction > 1e-3, f"only {nl_fraction:.3e} of E is nonlinear — a linear fixture"


@pytest.mark.parametrize("sigma0, sigma1", [(2.0, 0.0), (0.0, 1e-4), (1.0, 5e-5)])
def test_rust_is_passive_under_either_loss(sigma0, sigma1):
    s = physsynth_rs.TensionModulatedString(**kw(sigma0=sigma0, sigma1=sigma1))
    s.set_state(pluck(s, 1e-2))
    prev = s.energy()
    for _ in range(600):
        s.step()
        e = s.energy()
        assert e <= prev * (1 + 1e-12), f"energy rose: {prev:.6e} -> {e:.6e}"
        prev = e


def test_rust_hardens_the_pitch_with_amplitude():
    """The model's whole reason for existing: pluck harder, start sharper. Measured from zero
    crossings of the modal projection, because the peak moves far enough that a window anchored on
    the linear frequency simply misses it."""
    periods = []
    for amplitude in (1e-3, 6e-3, 1.2e-2):
        s = physsynth_rs.TensionModulatedString(**kw(kappa=0.0, EA=8.0 * T_DEF))
        shape = mode(s, 1, 1.0)
        s.set_state(mode(s, 1, amplitude))
        periods.append(_first_crossing(s, shape))
    assert periods[0] > periods[1] > periods[2], f"pitch did not rise with amplitude: {periods}"


def _first_crossing(s, shape):
    denom = float(np.dot(shape, shape))
    prev = float(np.dot(s.state, shape)) / denom
    for n in range(1, 200_000):
        s.step()
        cur = float(np.dot(s.state, shape)) / denom
        if prev > 0.0 >= cur:
            return (n - 1 + prev / (prev - cur)) * s.k
        prev = cur
    raise AssertionError("no zero crossing in 200,000 steps")


def test_the_bracket_cap_and_tolerance_constants_agree():
    """The two constants the solve is parameterised by. A silent drift in either would change
    every trajectory in the model without moving a single physics bar."""
    assert TENSION_TOL_DEFAULT == 1e-13
    assert MAX_BRACKET_EXPANSIONS == 40
    a, b = pair(dict(L=1.0, T=200.0, rho=0.005, fs=44100.0, N=32))
    assert a.tension_tol == b.tension_tol == TENSION_TOL_DEFAULT
    assert a.EA == b.EA == 0.0, "EA defaults to 0 — model #3, bit-for-bit — on both sides"
    assert a.theta == b.theta == THETA_DEFAULT


def test_a_string_at_rest_stays_at_rest_on_both():
    """``dT_hi <= 0`` is a real branch, and it is the one a fixture reaches by doing nothing."""
    for s in pair(kw()):
        for _ in range(30):
            s.step()
            assert np.array_equal(s.u, np.zeros(s.N + 1))
            assert s.delta_tension == 0.0
            assert s.converged


def test_the_banded_module_is_the_one_the_model_captured():
    """The captured-binding hazard, scoped to this file: if ``string_nonlinear`` holds a different
    ``cholesky_banded`` than ``banded`` exposes, every exact assertion above is comparing two
    solvers rather than two models — and would fail with a message about physics."""
    assert string_nonlinear.cholesky_banded is banded.cholesky_banded
    assert string_nonlinear.cho_solve_banded is banded.cho_solve_banded


def test_the_stretch_reduction_is_the_portable_one_on_both_sides():
    """The edit this batch rests on. ``_stretch`` must be a left-to-right sum — which is what a
    Rust ``for`` loop is — and *not* ``np.dot``, whose BLAS kernel no portable implementation
    reproduces. Asserted against an explicit loop, so the claim is *which* order.
    """
    rng = np.random.default_rng(20260827)
    s = TensionModulatedStringPy(**kw())
    differ_from_blas = 0
    for _ in range(200):
        u = rng.standard_normal(s.N + 1)
        u[0] = u[-1] = 0.0
        du = np.diff(u)
        loop = 0.0
        for v in du:
            loop += float(v) * float(v)
        assert s._stretch(u) == loop / s.h
        if float(np.dot(du, du)) != loop:
            differ_from_blas += 1
    assert differ_from_blas > 0, (
        "np.dot agreed with a left-to-right sum on every vector — the reason `portable.dot` exists "
        "no longer reproduces on this machine, so remeasure before trusting §19.2"
    )


def test_stretch_int_sums_left_to_right():
    """``(dot + u_0^2) + u_last^2``. Grouping the two end terms first is a different number, and it
    was the batch's one real porting error — invisible to the trajectory (§19.4)."""
    s = TensionModulatedStringPy(**kw())
    u = np.array([-15.143835037313956, 3.9498186274953, -6.7056582368787945])
    du = np.diff(u)
    d = float(du[0] * du[0] + du[1] * du[1])
    a = math.pow(float(u[0]), 2.0)
    b = math.pow(float(u[2]), 2.0)
    assert (d + a) + b != d + (a + b), "the witness no longer separates the two groupings"
    assert s._stretch_int(u) == ((d + a) + b) / s.h
