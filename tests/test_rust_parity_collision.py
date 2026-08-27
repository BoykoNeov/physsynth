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

3. **The vector solve's Group A window is short, per fixture, and it closes for a dynamical
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
"""

import contextlib
import warnings

import numpy as np
import pytest
from helpers import make_barrier_string, make_damped_string, make_mallet, make_mallet_wall
from scipy.linalg import lu_factor as sp_lu_factor
from scipy.linalg import lu_solve as sp_lu_solve

from physsynth.core import collision as C
from physsynth.core import mallet as M

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13     # the plan's short-run agreement target
DRIFT_TOL = 1e-10       # CLAUDE.md's energy bar, which neither implementation may cross
SHORT_RUN = 2000        # long enough for a bit-identity claim to mean something
GROUP_A_RUN = 500       # a third of the tightest measured window -- see item 3 in the header

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
def test_primitives_are_bit_identical_on_the_array_path(name, alpha):
    eta = _penetrations(20000)
    py = np.asarray(getattr(C, name + "_py")(eta, K_DEFAULT, alpha), dtype=float)
    rs = np.asarray(getattr(physsynth_rs, name)(eta, K_DEFAULT, alpha), dtype=float)
    np.testing.assert_array_equal(rs, py)


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


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.3])
@pytest.mark.parametrize("lam_h", [0.0, 2.0e4])
def test_the_vector_force_and_derivative_are_bit_identical(alpha, lam_h):
    en = _penetrations(20000)
    ep = en + _penetrations(20000, seed=99) * 1e-2
    np.testing.assert_array_equal(
        physsynth_rs.force_total_vec(en, ep, K_DEFAULT, alpha, lam_h, KSTEP, TOL),
        C.force_total_vec_py(en, ep, K_DEFAULT, alpha, lam_h, KSTEP, TOL),
    )
    np.testing.assert_array_equal(
        physsynth_rs.deriv_total_vec(en, ep, K_DEFAULT, alpha, lam_h, KSTEP, TOL),
        C.deriv_total_vec_py(en, ep, K_DEFAULT, alpha, lam_h, KSTEP, TOL),
    )


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

def _barrier_run(steps, **kw):
    bar = make_barrier_string(**kw)
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


def _point_fret(N=80, node=None):
    b = np.full(N + 1, -np.inf)
    b[node if node is not None else N // 3] = -2.0e-4
    return b


CASES = {
    "point fret (m=1)": {"N": 80, "barrier": _point_fret(), "lam": 0.4, "K": K_DEFAULT,
                         "alpha": 1.5},
    "flat rail (m=79)": {"K": K_DEFAULT, "alpha": 1.5, "lam": 0.4},
    "flat rail alpha=1": {"K": K_DEFAULT, "alpha": 1.0, "lam": 0.4},
    "flat rail stiff": {"K": K_STIFF, "alpha": 1.5, "lam": 0.4},
    "flat rail lossy": {"K": K_DEFAULT, "alpha": 1.5, "lam": 0.4, "hysteresis": 2.0e4},
}


def test_a_single_contact_node_is_bit_identical():
    """The cause-separator. One node means ``G`` is 1x1, so the matvec is a single multiply and the
    LU a scalar divide -- neither can round differently. Everything else in the solve is shared, so
    a failure here is a transcription bug and nothing else."""
    with python_vector_solve():
        py = _barrier_run(SHORT_RUN, **CASES["point fret (m=1)"])
    with rust_vector_solve():
        rs = _barrier_run(SHORT_RUN, **CASES["point fret (m=1)"])
    np.testing.assert_array_equal(rs[0], py[0])
    np.testing.assert_array_equal(rs[1], py[1])


def test_two_contact_nodes_are_bit_identical_too():
    """``m = 2`` adds a real two-term sum to the matvec and a real pivot choice to the LU, and is
    still short enough that neither can reorder. It separates "the transcription is right" from
    "the reduction is too short to disagree"."""
    kw = {"N": 80, "lam": 0.4, "K": K_DEFAULT, "alpha": 1.5}
    b = np.full(81, -np.inf)
    b[27], b[54] = -2.0e-4, -2.0e-4
    with python_vector_solve():
        py = _barrier_run(SHORT_RUN, barrier=b, **kw)
    with rust_vector_solve():
        rs = _barrier_run(SHORT_RUN, barrier=b, **kw)
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
    """
    with python_vector_solve():
        soft = _barrier_run(SHORT_RUN, **CASES["flat rail (m=79)"])
    with rust_vector_solve():
        soft_rs = _barrier_run(SHORT_RUN, **CASES["flat rail (m=79)"])
    np.testing.assert_array_equal(soft_rs[0], soft[0])

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
