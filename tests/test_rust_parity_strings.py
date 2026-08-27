"""Rust vs NumPy/SciPy for the two theta-scheme strings — models #2 and #3.

``docs/dev/rust-migration-plan.md`` §18. This is the first batch that ports a *model* out of the
four-string chain, and the thing that had to be true before it could is not in Rust at all: two
evaluation orders had to move to a spelling both languages can express, and both live in
``physsynth/core/portable.py``. The tests below check that they landed on both sides, and then that
the models built on them agree.

**The qualifier that shapes every assertion here.** The two strings agree to the BIT — trajectory,
history and ``energy()`` alike — only when both sides use the *same banded solver*, which is what
``PHYSSYNTH_RS=1`` arranges. Without the flag SciPy calls OpenBLAS's blocked ``DTBSV`` and the Rust
side runs the reference ``DTBSV`` transcribed (§15.3), so the two differ in the last bit from the
first step for a reason this batch did not introduce and cannot remove. So:

* structure — parameters, ``x``, ``_L`` down to its ``indices`` and ``nnz`` — is compared
  **exactly**, always. The Cholesky *factor* is not in that list: it is whichever Cholesky ran, and
  §15.3 measured the transcription agreeing with LAPACK on 82 of 120 of this family's matrices, so
  it is exact under a shared solver and held to Group A otherwise;
* trajectories are compared **exactly** under a shared solver, and to the plan's Group A target
  otherwise. Measured 2026-08-27, that gap grows like a random walk rather than exponentially:
  1.7e-14 of amplitude at 100 steps and 1.6e-13 at 20,000, which is a third regime next to the
  chaotic barrier's exponential separation (§16.5) and the mallet's transient non-separation
  (§17.5);
* the physics bars are asserted on the Rust side directly, because those are the acceptance
  contract and they must hold whatever the comparison says.

Green with the flag and without it, per §16.4's convention — the assertions differ between the two
modes, the file does not.
"""

import contextlib

import numpy as np
import pytest

from physsynth.core import portable, string_damped, string_stiff
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_damped import DampedStiffStringPy
from physsynth.core.string_stiff import THETA_DEFAULT, StiffStringPy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13  # the plan's §4 agreement target for a short run
DRIFT_TOL = 1e-10  # CLAUDE.md's acceptance bar, which neither implementation may cross

L_DEF, T_DEF, RHO_DEF = 1.0, 200.0, 0.005
KAPPA_DEF = 1.5


def rs_cholesky(ab, lower=False):
    return physsynth_rs.cholesky_banded_upper(np.ascontiguousarray(ab, dtype=float))


def rs_cho_solve(cb_and_lower, b):
    cb, _lower = cb_and_lower
    return physsynth_rs.cho_solve_banded_upper(
        np.ascontiguousarray(cb, dtype=float), np.ascontiguousarray(b, dtype=float)
    )


@contextlib.contextmanager
def shared_solver():
    """Put the Python models on the Rust banded solver for the duration of the block.

    The models capture ``cho_solve_banded`` at import (the hazard ``test_stability.py``'s guard
    watches), so patching the captured name is the only way to hold the solver constant while the
    *model* varies — which is what makes a bit-identity claim about the port rather than about
    OpenBLAS. Under ``PHYSSYNTH_RS=1`` this is already the state of the world and the patch is a
    no-op; without the flag it is what the exact comparisons below need.
    """
    saved = [(m, m.cholesky_banded, m.cho_solve_banded) for m in (string_stiff, string_damped)]
    for m in (string_stiff, string_damped):
        m.cholesky_banded = rs_cholesky
        m.cho_solve_banded = rs_cho_solve
    try:
        yield
    finally:
        for m, chol, solve in saved:
            m.cholesky_banded = chol
            m.cho_solve_banded = solve


def stiff_kw(N=64, kappa=KAPPA_DEF, sigma=0.0, fs=44100.0, theta=THETA_DEFAULT):
    return dict(L=L_DEF, T=T_DEF, rho=RHO_DEF, fs=fs, N=N, kappa=kappa, sigma=sigma, theta=theta)


def damped_kw(N=64, kappa=KAPPA_DEF, sigma0=0.0, sigma1=0.0, fs=44100.0, theta=THETA_DEFAULT):
    return dict(
        L=L_DEF, T=T_DEF, rho=RHO_DEF, fs=fs, N=N, kappa=kappa,
        sigma0=sigma0, sigma1=sigma1, theta=theta,
    )


def pair(kw, damped=False):
    """The Python reference and the Rust implementation of the same string."""
    if damped:
        return DampedStiffStringPy(**kw), physsynth_rs.DampedStiffString(**kw)
    return StiffStringPy(**kw), physsynth_rs.StiffString(**kw)


def pluck(s, amplitude=1e-3):
    return triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=amplitude)


# =====================================================================================
# The two portable spellings — the thing that had to move before any model could
# =====================================================================================


@pytest.mark.parametrize("n", [1, 2, 7, 63, 99, 128, 257])
def test_portable_dot_is_a_left_to_right_sum(n):
    """``portable.dot`` must be the naive accumulation, which is what a Rust ``for`` loop is.

    Asserted against an explicit Python loop rather than against ``np.dot``: the claim is *which*
    order, not that it differs from BLAS's.
    """
    rng = np.random.default_rng(20260827 + n)
    for _ in range(200):
        a = rng.standard_normal(n) * rng.choice([1e-8, 1.0, 1e6])
        b = rng.standard_normal(n) * rng.choice([1e-8, 1.0, 1e6])
        acc = 0.0
        for i in range(n):
            acc += float(a[i]) * float(b[i])
        assert portable.dot(a, b) == acc


def test_portable_dot_actually_differs_from_np_dot():
    """The measurement that justifies the module existing — **reported, not asserted**.

    How *often* BLAS's reduction differs from a left-to-right one is a property of the kernel
    OpenBLAS picked for this CPU (§14.2), and on a machine whose kernel happened to accumulate
    left to right the count would be zero while nothing here was wrong — the four models would
    still be consistent with each other, which is the property that matters. So this prints and
    asserts nothing, deliberately. A test that *looks* like it asserts and does not is §17.2's
    failure mode, so the docstring says so rather than leaving the reader to check the body.
    """
    rng = np.random.default_rng(20260827)
    differ = sum(
        portable.dot(a, b) != float(np.dot(a, b))
        for a, b in (
            (rng.standard_normal(99) * 1e-3, rng.standard_normal(99) * 1e-3) for _ in range(2000)
        )
    )
    print(f"portable.dot differs from np.dot in {differ} of 2000 vectors at n = 99")


def test_the_operator_is_canonical_on_both_sides():
    """§18's other half: ascending column indices, so a CSR matvec sums in one known order.

    Checked on the *model's* operator rather than on a fabricated matrix, because the descending
    order arrives through ``biharmonic_matrix`` and only shows up once it has been subtracted.
    """
    for kw, damped in ((stiff_kw(), False), (damped_kw(sigma1=5e-3), True)):
        py, rs = pair(kw, damped)
        for name, m in (("python", py._L), ("rust", rs._L)):
            assert m.has_sorted_indices, f"{name}'s _L is not in canonical order"
        assert np.array_equal(py._L.indptr, rs._L.indptr)
        assert np.array_equal(py._L.indices, rs._L.indices)
        assert np.array_equal(py._L.data, rs._L.data)
        assert py._L.nnz == rs._L.nnz


def test_the_unsorted_operator_would_have_been_a_different_matvec():
    """The measurement that made the sort necessary, kept as a test so it cannot quietly stop
    being true. A descending-index copy of the same matrix multiplies a vector differently."""
    py, _ = pair(stiff_kw(), False)
    desc = py._L.copy()
    # Reverse each row's stored order: the same matrix, a different accumulation order.
    for i in range(desc.shape[0]):
        lo, hi = desc.indptr[i], desc.indptr[i + 1]
        desc.indices[lo:hi] = desc.indices[lo:hi][::-1]
        desc.data[lo:hi] = desc.data[lo:hi][::-1]
    desc.has_sorted_indices = False
    rng = np.random.default_rng(11)
    differ = sum(
        not np.array_equal(py._L @ v, desc @ v)
        for v in (rng.standard_normal(py.N - 1) * 1e-3 for _ in range(200))
    )
    # REPORTED, not asserted, for the reason above it is tempting to assert: a non-zero count
    # here is a claim about SciPy's `csr_matvec` honouring stored index order and about `@` not
    # sorting on the way in. Both hold for SciPy 1.16 and neither is promised, so requiring them
    # would turn a SciPy upgrade that made this edit unnecessary into a red gate for a non-bug —
    # which is §18.3's own argument pointed at this file. The port does not depend on the count:
    # `canonical` makes the order well-defined whatever the kernel does with it.
    print(f"L @ u differs between ascending and descending order in {differ} of 200 vectors")


# =====================================================================================
# Construction: parameters, operator, factor
# =====================================================================================


@pytest.mark.parametrize("N", [2, 3, 16, 64, 129])
@pytest.mark.parametrize("kappa", [0.0, 1e-3, KAPPA_DEF])
@pytest.mark.parametrize("sigma", [0.0, 3.0])
def test_stiff_construction_is_bit_identical(N, kappa, sigma):
    # The FACTOR is compared under a shared solver and the rest unconditionally, because the two
    # claims have different owners: `_L` is this port's arithmetic, `_chol` is whichever Cholesky
    # ran. SciPy's is LAPACK's `dpbtrf`, the Rust one is `DPBTF2` transcribed, and §15.3 measured
    # them agreeing exactly on 82 of 120 of this family's matrices -- so an unconditional
    # `array_equal` here would be asserting something about OpenBLAS.
    with shared_solver():
        py, rs = pair(stiff_kw(N=N, kappa=kappa, sigma=sigma), False)
        assert np.array_equal(py._chol, rs._chol), "the factor must match, not merely the matrix"
    py, rs = pair(stiff_kw(N=N, kappa=kappa, sigma=sigma), False)
    for field in ("L", "T", "rho", "fs", "N", "kappa", "sigma", "theta", "boundary",
                  "c", "h", "k", "lam", "B"):
        assert getattr(py, field) == getattr(rs, field), field
    assert np.array_equal(py.x, rs.x)
    assert np.array_equal(py._L.data, rs._L.data)
    assert np.array_equal(py._L.indices, rs._L.indices)
    assert np.array_equal(py._L.indptr, rs._L.indptr)
    assert py._chol.shape == rs._chol.shape
    band = py._chol != 0.0
    assert np.allclose(py._chol[band], rs._chol[band], rtol=GROUP_A_TOL, atol=0.0)


@pytest.mark.parametrize("N", [2, 3, 16, 64, 129])
@pytest.mark.parametrize("kappa", [0.0, KAPPA_DEF])
@pytest.mark.parametrize("sigma0,sigma1", [(0.0, 0.0), (2.0, 0.0), (0.0, 5e-3), (2.0, 5e-3)])
def test_damped_construction_is_bit_identical(N, kappa, sigma0, sigma1):
    py, rs = pair(damped_kw(N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1), True)
    for field in ("L", "T", "rho", "fs", "N", "kappa", "sigma0", "sigma1", "theta", "boundary",
                  "c", "h", "k", "lam", "B"):
        assert getattr(py, field) == getattr(rs, field), field
    assert np.array_equal(py.x, rs.x)
    assert np.array_equal(py._L.data, rs._L.data)
    assert np.array_equal(py._L.indices, rs._L.indices)
    assert np.array_equal(py._D2.data, rs._D2.data)
    assert np.array_equal(py._D2.indices, rs._D2.indices)
    band = py._chol != 0.0  # the factor: see the stiff string's note above
    assert np.allclose(py._chol[band], rs._chol[band], rtol=GROUP_A_TOL, atol=0.0)
    with shared_solver():
        py2, rs2 = pair(damped_kw(N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1), True)
        assert np.array_equal(py2._chol, rs2._chol)


def test_kappa_zero_is_structurally_tridiagonal_on_both_sides():
    """The guard is a branch, not a zero: ``kappa = 0`` skips the subtraction, so ``nnz`` differs.

    A Rust ``sub`` that kept explicit zeros would agree on every value here and disagree on the
    sparsity, which no trajectory test would see.
    """
    py0, rs0 = pair(stiff_kw(N=32, kappa=0.0), False)
    py1, rs1 = pair(stiff_kw(N=32, kappa=KAPPA_DEF), False)
    assert py0._L.nnz == rs0._L.nnz < py1._L.nnz == rs1._L.nnz


@pytest.mark.parametrize("theta", [0.25, 0.28, 0.5, 1.0])
def test_theta_is_carried_through_to_the_factor(theta):
    with shared_solver():
        py, rs = pair(stiff_kw(theta=theta), False)
        assert py.theta == rs.theta == theta
        assert np.array_equal(py._chol, rs._chol)


# =====================================================================================
# Construction rejections — the messages `test_stability.py` matches on
# =====================================================================================


@pytest.mark.parametrize(
    "bad,message",
    [
        (dict(L=0.0), "L, T, rho, fs must all be positive."),
        (dict(T=-1.0), "L, T, rho, fs must all be positive."),
        (dict(rho=0.0), "L, T, rho, fs must all be positive."),
        (dict(fs=-1.0), "L, T, rho, fs must all be positive."),
        (dict(N=1), "N must be >= 2 (need at least one interior node)."),
        (dict(N=-3), "N must be >= 2 (need at least one interior node)."),
        (dict(kappa=-1.0), "kappa (stiffness) must be >= 0."),
        (dict(sigma=-1.0), "sigma (loss) must be >= 0."),
        (dict(theta=0.0), "theta must be in (0, 1], got 0.0."),
        (dict(theta=1.5), "theta must be in (0, 1], got 1.5."),
        (dict(boundary="clamped"), "boundary must be 'supported', got 'clamped'."),
    ],
)
def test_stiff_rejections_match_verbatim(bad, message):
    kw = stiff_kw()
    kw.update(bad)
    for cls in (StiffStringPy, physsynth_rs.StiffString):
        with pytest.raises(ValueError) as exc:
            cls(**kw)
        assert str(exc.value) == message, cls


@pytest.mark.parametrize(
    "bad,message",
    [
        (dict(sigma0=-1.0), "sigma0 (frequency-independent loss) must be >= 0."),
        (dict(sigma1=-1.0), "sigma1 (frequency-dependent loss) must be >= 0."),
        (dict(kappa=-1.0), "kappa (stiffness) must be >= 0."),
        (dict(theta=2.0), "theta must be in (0, 1], got 2.0."),
    ],
)
def test_damped_rejections_match_verbatim(bad, message):
    kw = damped_kw()
    kw.update(bad)
    for cls in (DampedStiffStringPy, physsynth_rs.DampedStiffString):
        with pytest.raises(ValueError) as exc:
            cls(**kw)
        assert str(exc.value) == message, cls


def test_the_check_order_is_pythons():
    """A call wrong in two ways must report the same fault on both sides."""
    kw = stiff_kw()
    kw.update(T=-1.0, kappa=-1.0, theta=5.0)
    for cls in (StiffStringPy, physsynth_rs.StiffString):
        with pytest.raises(ValueError) as exc:
            cls(**kw)
        assert str(exc.value) == "L, T, rho, fs must all be positive."


# =====================================================================================
# set_state
# =====================================================================================


@pytest.mark.parametrize("kappa", [0.0, KAPPA_DEF])
@pytest.mark.parametrize("v0", [None, 0.0, 1.5, "array"])
def test_set_state_is_bit_identical(kappa, v0):
    py, rs = pair(stiff_kw(kappa=kappa), False)
    u0 = pluck(py)
    if v0 == "array":
        v0 = np.linspace(-1.0, 1.0, py.N + 1)
    args = (u0.copy(),) if v0 is None else (u0.copy(), v0)
    py.set_state(*args)
    rs.set_state(*args)
    assert np.array_equal(py.u, rs.u)
    assert np.array_equal(py.u_prev, rs.u_prev), "the consistent start must match to the bit"
    assert py.n == rs.n == 0


def test_set_state_clamps_a_non_zero_end_the_same_way():
    py, rs = pair(stiff_kw(), False)
    u0 = pluck(py)
    u0[0], u0[-1] = 7.0, -7.0
    py.set_state(u0.copy())
    rs.set_state(u0.copy())
    assert np.array_equal(py.u, rs.u)
    assert np.array_equal(py.u_prev, rs.u_prev)
    assert rs.u[0] == rs.u[-1] == 0.0


@pytest.mark.parametrize("shape", [(3,), (200,)])
def test_set_state_rejects_a_wrong_shape_the_same_way(shape):
    py, rs = pair(stiff_kw(N=64), False)
    bad = np.zeros(shape)
    for s in (py, rs):
        with pytest.raises(ValueError) as exc:
            s.set_state(bad)
        assert str(exc.value) == f"u0 must have shape ({65},), got ({shape[0]},)."


# =====================================================================================
# Stepping — the trajectory claim, in both modes
# =====================================================================================


STIFF_FIXTURES = [
    ("plain", stiff_kw(kappa=0.0, sigma=0.0)),
    ("stiff", stiff_kw(kappa=KAPPA_DEF, sigma=0.0)),
    ("lossy", stiff_kw(kappa=KAPPA_DEF, sigma=3.0)),
    ("coarse", stiff_kw(N=16, kappa=KAPPA_DEF, sigma=1.0)),
    ("supercritical", stiff_kw(N=200, kappa=KAPPA_DEF, sigma=0.0, fs=8000.0)),
    ("theta-one", stiff_kw(kappa=KAPPA_DEF, sigma=0.0, theta=1.0)),
]

DAMPED_FIXTURES = [
    ("plain", damped_kw(kappa=0.0)),
    ("stiff", damped_kw(kappa=KAPPA_DEF)),
    ("sigma0", damped_kw(kappa=KAPPA_DEF, sigma0=2.0)),
    ("sigma1", damped_kw(kappa=KAPPA_DEF, sigma1=5e-3)),
    ("both", damped_kw(kappa=KAPPA_DEF, sigma0=2.0, sigma1=5e-3)),
    ("coarse", damped_kw(N=16, kappa=KAPPA_DEF, sigma0=1.0, sigma1=1e-3)),
]


def _run_pair(py, rs, steps):
    """Step both, returning the first differing step (or ``None``) and the worst gaps."""
    u0 = pluck(py)
    amp = float(np.max(np.abs(u0)))
    py.set_state(u0.copy())
    rs.set_state(u0.copy())
    first = None
    worst_state = 0.0
    worst_energy = 0.0
    for n in range(steps):
        py.step()
        rs.step()
        if first is None and not np.array_equal(py.u, rs.u):
            first = n
        worst_state = max(worst_state, float(np.max(np.abs(py.u - rs.u))) / amp)
        e_py, e_rs = py.energy(), rs.energy()
        worst_energy = max(worst_energy, abs(e_py - e_rs) / max(abs(e_py), 1e-300))
    return first, worst_state, worst_energy


@pytest.mark.parametrize("name,kw", STIFF_FIXTURES, ids=[f[0] for f in STIFF_FIXTURES])
def test_stiff_trajectory_is_bit_identical_under_a_shared_solver(name, kw):
    """The batch's central claim. Exact, including ``energy()``, over 2,000 steps.

    ``energy()`` matching is the half that needed ``portable.dot``: it is three reductions plus a
    fourth, and under ``np.dot`` none of them would agree.
    """
    with shared_solver():
        py, rs = pair(kw, False)
        first, state_gap, energy_gap = _run_pair(py, rs, 2000)
    assert first is None, f"{name}: diverged at step {first}"
    assert state_gap == 0.0 and energy_gap == 0.0


@pytest.mark.parametrize("name,kw", DAMPED_FIXTURES, ids=[f[0] for f in DAMPED_FIXTURES])
def test_damped_trajectory_is_bit_identical_under_a_shared_solver(name, kw):
    with shared_solver():
        py, rs = pair(kw, True)
        first, state_gap, energy_gap = _run_pair(py, rs, 2000)
    assert first is None, f"{name}: diverged at step {first}"
    assert state_gap == 0.0 and energy_gap == 0.0


def test_bit_identity_survives_a_long_run():
    """A linear model has no mechanism to amplify a difference, so the window does not close —
    the same reason the mallet's did not (§17.5), reached without a transient."""
    with shared_solver():
        py, rs = pair(damped_kw(N=32, kappa=KAPPA_DEF, sigma0=1.0, sigma1=1e-3), True)
        first, state_gap, _ = _run_pair(py, rs, 20000)
    assert first is None and state_gap == 0.0


@pytest.mark.parametrize("name,kw", DAMPED_FIXTURES, ids=[f[0] for f in DAMPED_FIXTURES])
def test_trajectory_meets_group_a_against_scipys_own_solver(name, kw):
    """Without a shared solver the two differ, and this is the bar they are held to instead.

    The gap is §15.3's — OpenBLAS's blocked ``DTBSV`` against the reference one transcribed — not
    this port's, which the test above establishes by removing it.

    **100 steps, because that is the window §15.4 actually defines.** Group A is a run-length claim
    (§14.4), and §15.4 shortened it to ~100 steps specifically for a *fed-back solve*, which is what
    this is. The first version ran 500 and passed on Windows at 7.6e-14 — 76% of the bar, with the
    margin coming from the run being four times longer than the claim it tests. On the CI runner the
    same fixture read 1.01e-13 and went red. Nothing about the port differs between the two
    machines: the quantity being measured is one BLAS's banded solve against another's, so its exact
    value is a property of the runner (§21.1), and a bar this close to the measurement was going to
    be decided by whichever machine ran it. At 100 steps the worst fixture reads 3.0e-14 here, a
    third of the bar, and the *tolerance is unchanged* — which is the point. Measured 2026-08-27.
    """
    py, rs = DampedStiffStringPy(**kw), physsynth_rs.DampedStiffString(**kw)
    _first, state_gap, energy_gap = _run_pair(py, rs, 100)
    assert state_gap < GROUP_A_TOL, f"{name}: {state_gap:.2e} of amplitude"
    assert energy_gap < GROUP_A_TOL


# =====================================================================================
# The reduction anchor, across languages
# =====================================================================================


@pytest.mark.parametrize("kappa", [0.0, KAPPA_DEF])
@pytest.mark.parametrize("sigma", [0.0, 3.0])
def test_rust_sigma1_zero_is_the_rust_stiff_string(kappa, sigma):
    """The Python anchor, run between the two Rust classes — which is only a real comparison
    because they are two separate transcriptions rather than one superset."""
    ss = physsynth_rs.StiffString(**stiff_kw(N=100, kappa=kappa, sigma=sigma))
    ds = physsynth_rs.DampedStiffString(**damped_kw(N=100, kappa=kappa, sigma0=sigma, sigma1=0.0))
    u0 = pluck(ss)
    ss.set_state(u0.copy())
    ds.set_state(u0.copy())
    assert np.array_equal(ss.u_prev, ds.u_prev)
    for step in range(1500):
        ss.step()
        ds.step()
        assert np.array_equal(ss.u, ds.u), f"diverged at step {step}"
        assert ss.energy() == ds.energy()


@pytest.mark.parametrize("kappa", [0.0, KAPPA_DEF])
def test_a_rust_string_and_a_python_string_stay_anchored(kappa):
    """The cross-language form of the same anchor, which is what makes porting one model of the
    four safe: a Rust ``StiffString`` at ``sigma1 = 0`` must equal a Python ``DampedStiffString``.

    This is the assertion the whole ``portable.py`` detour exists to make possible.
    """
    with shared_solver():
        rs = physsynth_rs.StiffString(**stiff_kw(N=100, kappa=kappa, sigma=3.0))
        py = DampedStiffStringPy(**damped_kw(N=100, kappa=kappa, sigma0=3.0, sigma1=0.0))
        u0 = pluck(rs)
        rs.set_state(u0.copy())
        py.set_state(u0.copy())
        assert np.array_equal(rs.u_prev, py.u_prev)
        for step in range(600):
            rs.step()
            py.step()
            assert np.array_equal(rs.u, py.u), f"diverged at step {step}"
            assert rs.energy() == py.energy()


# =====================================================================================
# The physics bars, on the Rust side directly
# =====================================================================================


@pytest.mark.parametrize("kappa", [0.0, KAPPA_DEF])
def test_rust_lossless_run_conserves_energy(kappa):
    s = physsynth_rs.StiffString(**stiff_kw(N=100, kappa=kappa, sigma=0.0))
    s.set_state(pluck(s))
    e0 = s.energy()
    assert e0 > 0.0
    worst = max(abs(s.energy() - e0) / e0 for _ in range(2000) if (s.step() or True))
    assert worst < DRIFT_TOL, f"drift {worst:.2e}"


@pytest.mark.parametrize("sigma0,sigma1", [(2.0, 0.0), (0.0, 5e-3), (2.0, 5e-3)])
def test_rust_lossy_run_is_passive(sigma0, sigma1):
    s = physsynth_rs.DampedStiffString(
        **damped_kw(N=100, kappa=KAPPA_DEF, sigma0=sigma0, sigma1=sigma1)
    )
    s.set_state(pluck(s))
    e0 = prev = s.energy()
    for n in range(2000):
        s.step()
        e = s.energy()
        assert e - prev <= 1e-12 * e0, f"energy rose at step {n}"
        prev = e
    assert prev < e0


# =====================================================================================
# What a coupled model reaches for
# =====================================================================================


@pytest.mark.parametrize("sigma1", [0.0, 5e-3])
def test_apply_ainv_is_bit_identical_under_a_shared_solver(sigma1):
    """``bow``, ``collision.BarrierString`` and ``connection`` build an admittance out of this."""
    with shared_solver():
        py, rs = pair(damped_kw(N=48, kappa=KAPPA_DEF, sigma0=1.0, sigma1=sigma1), True)
        rng = np.random.default_rng(7)
        for _ in range(20):
            b = rng.standard_normal(py.N - 1) * 1e-3
            assert np.array_equal(py.apply_Ainv(b), rs.apply_Ainv(b))
        # The unit columns a coupled model actually asks for.
        for node in (1, 7, py.N - 1):
            e = np.zeros(py.N - 1)
            e[node - 1] = 1.0
            assert np.array_equal(py.apply_Ainv(e), rs.apply_Ainv(e))


def test_apply_ainv_rejects_a_wrong_length_the_same_way():
    py, rs = pair(damped_kw(N=48), True)
    bad = np.zeros(5)
    for s in (py, rs):
        with pytest.raises(ValueError) as exc:
            s.apply_Ainv(bad)
        assert str(exc.value) == "rhs_int must have shape (47,), got (5,)."


def test_the_state_arrays_are_writable_in_place():
    """``bow`` does ``string.u += ...`` and ``BarrierString`` does ``s.u[1:-1] = ...``.

    Both are writes *through* the attribute, and both only reach the model if the array handed
    back is the real one — §9.3's finding, which every coupled model in the tree depends on.
    """
    rs = physsynth_rs.DampedStiffString(**damped_kw(N=32))
    rs.set_state(pluck(rs))
    before = np.array(rs.u, copy=True)
    kick = np.zeros(rs.N + 1)
    kick[7] = 1e-4
    rs.u += kick
    assert rs.u[7] == before[7] + 1e-4, "an in-place write did not reach the model"
    rs.u[1:-1] = rs.u[1:-1] + 1e-5
    assert rs.u[7] == before[7] + 1e-4 + 1e-5
    # And the object identity a snapshot depends on: `u_prev` after a step IS the old `u` — the
    # same object, not a copy of it, which is what lets a caller hold a reference across a step
    # and still have a valid snapshot of the level it took (§9.3).
    held = rs.u
    rs.step()
    assert rs.u_prev is held


def test_state_and_displacement_at_match():
    py, rs = pair(damped_kw(N=48, sigma0=1.0), True)
    u0 = pluck(py)
    py.set_state(u0.copy())
    rs.set_state(u0.copy())
    with shared_solver():
        py2, rs2 = pair(damped_kw(N=48, sigma0=1.0), True)
        py2.set_state(u0.copy())
        rs2.set_state(u0.copy())
        for _ in range(50):
            py2.step()
            rs2.step()
        assert np.array_equal(py2.state, rs2.state)
        for idx in (0, 1, 17, -1, -2):
            assert py2.displacement_at(idx) == rs2.displacement_at(idx)
    # `state` is a copy on both sides — mutating it must not reach the model.
    snap = rs.state
    snap[3] = 1e9
    assert rs.u[3] != 1e9


def test_displacement_at_rejects_out_of_range():
    rs = physsynth_rs.StiffString(**stiff_kw(N=16))
    for idx in (17, 100, -18):
        with pytest.raises(IndexError):
            rs.displacement_at(idx)
