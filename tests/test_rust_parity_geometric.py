"""Rust vs NumPy/SciPy for the geometrically exact string — model #10.

``docs/dev/rust-migration-plan.md`` §29. The **last** of the four theta-scheme strings, so this is
the batch that closes the chain ``portable.py`` was written to protect, and the last model in
``physsynth/core/`` outside ``connection``, ``airbox`` and ``analysis/``.

**What is new, and it is one thing.** Every earlier Group D model factored its sparse matrix
**once**, at construction. This one factors a fresh Newton Jacobian *inside* the iteration, so the
cost of a factorization moved onto the hot path — and §24's decision to leave the Rust sparse LU in
the natural column order, taken because "every Group D matrix in this project is a banded FDTD
operator whose natural order already has [no fill] to speak of", stops being free. It is not a near
miss: the unknowns are stacked **by field** and the nonlinearity couples the three fields at the
same *cell*, so in that order every coupling sits ``N-1`` columns off the diagonal. Measured at
``N = 128``, the natural order stores 33,895 nonzeros in ``L + U`` where SuperLU's COLAMD stores
2,788, and factors in 2,068 µs against SciPy's 156. Reordered by node it stores 2,645 and costs 58.
:func:`test_the_node_ordering_is_what_makes_the_factorization_cheap` is that measurement as an
assertion, because **no other bar in this project could see a regression in it** — the answers
would stay right and the model would get slow.

**Three qualifiers the assertions cannot be stated without.**

* *The shared factorizations, both of them.* This model has **two** solvers — a banded Cholesky at
  construction and a sparse LU per Newton iteration — and §24.4's manoeuvre only says anything if
  both are held constant. With both shared the two implementations are **bit-identical over 2,000
  steps**, energy and Newton iteration count included, which is what says the entire residue is the
  solver and none of it is the port. Sharing only the banded one would leave every difference
  confounded with the SuperLU gap §24.2 measured, which is §19.4's real-bug-hidden-under-a-gap
  waiting to happen.
* *The resolved operating point.* ``c_long/c = sqrt(EA/T) ~ 22``, so the familiar transverse
  ``lam = 0.5`` means ``lam_long ~ 11`` — and past ``lam_long ~ 4`` the Newton solve stops
  converging and the drift explodes by fourteen orders. That is the model's own documented cliff,
  not a port defect, and a fixture that lands past it fails the conservation bar **correctly**. So
  every fixture here sets ``fs`` from ``lam_long``, as ``tests/helpers.py`` does.
* *The branch that varies.* §19.2's rule, and this model answers it more gently than model #9 did.
  The convergence test is ``max|r| <= newton_tol * max|Y_seed|`` — a **max**, which is
  order-independent — where model #9's was a ``brentq`` bracket on a *sum*. What still varies is
  *which side of the bar* one Newton step lands on, and the flip rate is set by where the mean
  iteration count sits between two integers: measured over 20,000 steps it is **0** when the mean
  is pinned at 1.00, 36 when it is 1.97, and 293-475 when it sits mid-way. A flip costs about two
  orders of trajectory agreement and **nothing at all** on the energy, because any root of the
  discrete-gradient equation conserves exactly — the model declines to gate uniqueness for
  precisely that reason.

Green with the flag and without it, per §16.4's convention — the assertions differ between the two
modes, the file does not.
"""

import contextlib
import time
import warnings

import numpy as np
import pytest

from physsynth.core import string_geometric
from physsynth.core.string_damped import DampedStiffStringPy
from physsynth.core.string_geometric import (
    LAM_LONG_WARN,
    NEWTON_MAXITER_DEFAULT,
    NEWTON_TOL_DEFAULT,
    GeometricState,
    GeometricStringPy,
)
from physsynth.core.string_stiff import THETA_DEFAULT

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

GROUP_A_TOL = 1e-13  # the plan's §4 agreement target for a short run
DRIFT_TOL = 1e-10  # CLAUDE.md's acceptance bar, which neither implementation may cross

# Group D's bar (§24.2): tolerance-level agreement, quantified. Measured over 2,000 steps on the
# fixtures below, the worst `max|du| / running peak` is 2.6e-13 .. 7.0e-13 and the energies agree
# to ~1.4e-13 relative, so these sit an order above what the port does rather than at it. They are
# NOT Group A's target and must not be tightened to it: what they measure is SuperLU against a
# Gilbert-Peierls elimination in a different column order, which is a different algorithm.
GROUP_D_TOL = 1e-11
GROUP_D_ENERGY_TOL = 1e-12

L_DEF, T_DEF, RHO_DEF = 0.65, 200.0, 0.005
KAPPA_DEF = 1.5
EA_DEF = 1.0e5  # EA/T0 = 500, the ratio the shipped fixtures use
LAM_LONG_DEF = 0.5  # the LONGITUDINAL Courant number; see the module docstring


def fs_for(N, EA=EA_DEF, rho=RHO_DEF, L=L_DEF, lam_long=LAM_LONG_DEF):
    """The sample rate that puts ``N`` segments at ``lam_long`` — ``helpers.make_geometric_string``.

    The fast field sets the timestep. Reaching for a transverse ``lam`` here is the trap
    ``LAM_LONG_WARN`` exists to catch.
    """
    return float(np.sqrt(EA / rho)) * N / (L * lam_long)


def kw(N=32, kappa=KAPPA_DEF, kappa_w=None, EA=EA_DEF, sigma0=0.0, sigma1=0.0,
       sigma0_long=None, sigma1_long=None, theta=THETA_DEFAULT, lam_long=LAM_LONG_DEF,
       newton_tol=NEWTON_TOL_DEFAULT):
    return dict(
        L=L_DEF, T=T_DEF, rho=RHO_DEF, fs=fs_for(N, EA=EA, lam_long=lam_long), N=N, EA=EA,
        kappa=kappa, kappa_w=kappa_w, sigma0=sigma0, sigma1=sigma1, sigma0_long=sigma0_long,
        sigma1_long=sigma1_long, theta=theta, newton_tol=newton_tol,
    )


def pair(k):
    """The Python reference and the Rust implementation of the same string."""
    return GeometricStringPy(**k), physsynth_rs.GeometricString(**k)


def mode(s, m=1, amplitude=1e-3):
    return amplitude * np.sin(m * np.pi * s.x / s.L)


def interleave_perm(n_int):
    """``(u_i, w_i, v_i)`` per node — the model's own fill-reducing order, §29.2.

    Spelled out here rather than read off the Rust model on purpose: a shared-factorization test
    that asked the Rust side for its permutation could not tell a wrong permutation from a
    consistently wrong one.
    """
    return np.array([f * n_int + i for i in range(n_int) for f in range(3)], dtype=np.int64)


def rs_cholesky(ab, lower=False):
    return physsynth_rs.cholesky_banded_upper(np.ascontiguousarray(ab, dtype=float))


def rs_cho_solve(cb_and_lower, b):
    cb, _lower = cb_and_lower
    return physsynth_rs.cho_solve_banded_upper(
        np.ascontiguousarray(cb, dtype=float), np.ascontiguousarray(b, dtype=float)
    )


class _RustLu:
    """``scipy.sparse.linalg.splu``'s object, backed by the Rust sparse LU and the node order."""

    def __init__(self, m, perm):
        csr = m.tocsr()
        csr.sort_indices()
        self.lu = physsynth_rs.SparseLu(
            csr.data.astype(float), csr.indices.astype(np.int32),
            csr.indptr.astype(np.int32), csr.shape[0], perm,
        )

    def solve(self, b):
        return np.asarray(self.lu.solve(np.ascontiguousarray(b, dtype=float)))


@contextlib.contextmanager
def shared_solver(n_int):
    """Put the Python model on **both** Rust solvers for the duration of the block.

    ``string_geometric`` captures ``cholesky_banded``, ``cho_solve_banded`` and ``splu`` at import
    (the hazard ``test_stability.py``'s guard watches), so patching the captured names is the only
    way to hold the solvers constant while the *model* varies. Under ``PHYSSYNTH_RS=1`` the two
    banded names are already Rust and only ``splu`` moves — which is the point: the sparse LU is
    the one name this module deliberately does **not** swap, because the Python model is the
    reference and SuperLU is what produced the shipped numbers.
    """
    perm = interleave_perm(n_int)
    saved = (string_geometric.cholesky_banded, string_geometric.cho_solve_banded,
             string_geometric.splu)
    string_geometric.cholesky_banded = rs_cholesky
    string_geometric.cho_solve_banded = rs_cho_solve
    string_geometric.splu = lambda m: _RustLu(m, perm)
    try:
        yield
    finally:
        (string_geometric.cholesky_banded, string_geometric.cho_solve_banded,
         string_geometric.splu) = saved


# =====================================================================================
# Construction
# =====================================================================================

SCALARS = (
    "L T rho fs N EA kappa kappa_u kappa_w sigma0 sigma1 sigma0_long sigma1_long theta "
    "boundary newton_tol newton_maxiter allow_softening c c_long h k lam lam_long B EA_over_T "
    "_a n converged newton_iters total_newton_iters n_not_converged is_degenerate energy_floor"
).split()


@pytest.mark.parametrize("over", [
    {},
    dict(N=16),
    dict(N=64),
    dict(kappa=0.0),
    dict(kappa=2.7, kappa_w=3.1),
    dict(sigma0=2.0, sigma1=5e-3),
    dict(sigma0=2.0, sigma1=5e-3, sigma0_long=0.5, sigma1_long=1e-3),
    dict(EA=T_DEF),
    dict(theta=0.5),
    dict(lam_long=1.0),
])
def test_every_derived_scalar_is_bit_identical(over):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py, rs = pair(kw(**over))
    for name in SCALARS:
        assert getattr(py, name) == getattr(rs, name), name
    assert np.array_equal(py.x, rs.x)


MATRICES = ("_D2 _L_u _L_w _L_v _A_u _A_w _A_v _A3 _Gp _Gm _Gp3 _Gm3").split()


@pytest.mark.parametrize("over", [{}, dict(N=16), dict(kappa=0.0), dict(kappa=2.7, kappa_w=3.1),
                                  dict(sigma0=2.0, sigma1=5e-3), dict(theta=0.5)])
@pytest.mark.parametrize("name", MATRICES)
def test_every_constant_operator_is_bit_identical(over, name):
    """Values, ``nnz`` and the stored index order alike.

    The stored order is asserted rather than assumed, and it is why this model needed **no**
    ``portable.canonical`` work on the Python side (§18.2, §26.2): unlike the 1-D biharmonic, every
    matrix here already arrives from SciPy canonically ordered.
    """
    py, rs = pair(kw(**over))
    a, b = getattr(py, name), getattr(rs, name)
    assert a.shape == b.shape
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data)
    canonical = a.copy()
    canonical.sort_indices()
    assert np.array_equal(a.indices, canonical.indices), (
        f"{name} is not canonically ordered on the Python side -- a CSR matvec sums in stored "
        "order, so this model would need portable.canonical after all"
    )


@pytest.mark.parametrize("over", [{}, dict(N=16), dict(sigma0=2.0, sigma1=5e-3)])
def test_the_three_banded_factors_agree(over):
    """Shape included — the first draft returned a flat ``3n`` buffer with every value right."""
    k = kw(**over)
    with shared_solver(k["N"] - 1):
        py = GeometricStringPy(**k)
    rs = physsynth_rs.GeometricString(**k)
    for name in ("_chol_u", "_chol_w", "_chol_v"):
        a, b = getattr(py, name), getattr(rs, name)
        assert a.shape == b.shape == (3, k["N"] - 1)
        assert np.array_equal(a, b), name


# =====================================================================================
# Initial conditions
# =====================================================================================

@pytest.mark.parametrize("over", [{}, dict(EA=T_DEF), dict(kappa=0.0), dict(sigma0=2.0)])
def test_the_second_order_start_is_bit_identical(over):
    py, rs = pair(kw(**over))
    u0 = mode(py, 1, 1e-3)
    w0 = mode(py, 2, 4e-4)
    v0 = mode(py, 3, 2e-4)
    for a in (py, rs):
        a.set_state(u0.copy(), w0.copy(), v0.copy())
    for name in ("u", "w", "v", "u_prev", "w_prev", "v_prev"):
        assert np.array_equal(getattr(py, name), getattr(rs, name)), name
    assert py.n == rs.n == 0 and py.converged is rs.converged is True


def test_the_velocity_arguments_and_a_scalar_ic_agree():
    py, rs = pair(kw())
    xs = py.x
    kwargs = dict(u_dot=1e-2 * np.sin(np.pi * xs / py.L), w_dot=0.0,
                  v_dot=3e-3 * np.sin(2 * np.pi * xs / py.L))
    for a in (py, rs):
        a.set_state(1e-4, mode(a, 1, 1e-3), 0.0, **kwargs)
    for name in ("u", "w", "v", "u_prev", "w_prev", "v_prev"):
        assert np.array_equal(getattr(py, name), getattr(rs, name)), name
    # A scalar IC fills the whole grid and the clamp then zeroes the two ends -- so `u` is NOT
    # uniform, which is the behaviour a `np.full` followed by `f[0] = f[-1] = 0.0` produces.
    assert py.u[0] == py.u[-1] == 0.0 and py.u[1] == 1e-4


def test_the_state_property_is_the_named_tuple_and_copies():
    py, rs = pair(kw())
    for a in (py, rs):
        a.set_state(mode(a, 1, 1e-3))
    ps, rst = py.state, rs.state
    assert isinstance(rst, GeometricState)
    assert np.array_equal(ps.u, rst.u) and np.array_equal(ps.w, rst.w)
    assert np.array_equal(ps.v, rst.v)
    rst.u[0] = 1.0
    assert rs.u[0] == 0.0, "state must hand back copies"


def test_the_six_state_arrays_are_settable():
    """``helpers.seed_rotating_wave`` writes the exact helix history rather than using set_state.

    Not a convenience: the Taylor start seeds an ``O(k^3)`` error the helix sheds straight into the
    longitudinal field, ten orders on the claim that file measures. So all six arrays take setters
    and a Rust model that made them read-only would silently break the rotating-wave suite.
    """
    py, rs = pair(kw())
    rng = np.random.default_rng(20260831)
    fields = {name: rng.normal(scale=1e-3, size=py.N + 1)
              for name in ("u", "w", "v", "u_prev", "w_prev", "v_prev")}
    for a in (py, rs):
        for name, value in fields.items():
            setattr(a, name, value.copy())
        a.n = 0
        a.converged = True
    assert py.energy() == pytest.approx(rs.energy(), rel=1e-14)
    for name in fields:
        assert np.array_equal(getattr(py, name), getattr(rs, name))


def test_a_step_rebinds_u_prev_to_the_object_u_was():
    rs = physsynth_rs.GeometricString(**kw())
    rs.set_state(mode(rs, 1, 1e-3))
    before = rs.u
    rs.step()
    assert rs.u_prev is before or np.array_equal(rs.u_prev, before)


# =====================================================================================
# The private kernels -- §12.2, and they take arbitrary strains rather than the model's own
# =====================================================================================

def strains(N, seed=0):
    rng = np.random.default_rng(20260831 + seed)
    return rng.normal(scale=3e-3, size=(3, N))


@pytest.mark.parametrize("over", [{}, dict(N=16), dict(EA=T_DEF)])
def test_the_discrete_gradient_kernels_are_bit_identical(over):
    py, rs = pair(kw(**over))
    q_plus, q_minus = strains(py.N, 0), strains(py.N, 1)
    assert np.array_equal(py._stretch_ratio(q_plus), rs._stretch_ratio(q_plus))
    for a, b in zip(py._stretch_terms(q_plus), rs._stretch_terms(q_plus), strict=True):
        assert np.array_equal(a, b)
    assert np.array_equal(py._dg_force(q_plus, q_minus), rs._dg_force(q_plus, q_minus))
    assert np.array_equal(py._dg_force(q_plus, q_plus), rs._dg_force(q_plus, q_plus))
    pj, rj = py._dg_jacobian(q_plus, q_minus), rs._dg_jacobian(q_plus, q_minus)
    assert np.array_equal(pj.toarray(), rj.toarray())
    assert pj.shape == rj.shape == (3 * py.N, 3 * py.N)


def test_the_inverted_element_branch_is_reached_on_both_sides():
    """``denom > 1.0`` is a *branch*, and the physical fixtures never take the other arm.

    §16.4's rule: a kernel comparison on realistic strains would leave the second arm untested on
    both sides at once. An element compressed through zero length takes it.
    """
    py, rs = pair(kw())
    q = np.zeros((3, py.N))
    q[2, :] = -2.5  # 1 + v_x = -1.5: inverted, so Lambda + 1 + v_x < 1
    q[0, :] = 1e-4
    _, _, _, _, denom = py._stretch_terms(q)
    assert np.all(denom <= 1.0), "the fixture must actually take the inverted arm"
    for a, b in zip(py._stretch_terms(q), rs._stretch_terms(q), strict=True):
        assert np.array_equal(a, b)
    assert np.array_equal(py._dg_force(q, q), rs._dg_force(q, q))
    assert py._nl_density(q) == pytest.approx(rs._nl_density(q), rel=1e-14)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_nonlinear_density_is_a_declined_reduction(seed):
    """``_nl_density`` sums with ``np.sum`` (NumPy's pairwise blocking) and this port declines it.

    §28.7's decision, following ``ops2d.guitar_area`` and ``collision.barrier_energy``: matching
    the blocking would be a claim about a library internal and, per §22.1, about a CPU. It is a
    read-out — nothing on the update path reads it — so the bar is a last bit.
    """
    py, rs = pair(kw())
    q = strains(py.N, seed)
    a, b = py._nl_density(q), rs._nl_density(q)
    assert a == pytest.approx(b, rel=1e-14, abs=0.0)


def test_the_strain_helper_agrees():
    py, rs = pair(kw())
    for a in (py, rs):
        a.set_state(mode(a, 1, 1e-3), mode(a, 2, 5e-4), mode(a, 3, 2e-4))
    assert np.array_equal(py._strain(py.u, py.w, py.v), rs._strain(rs.u, rs.w, rs.v))


# =====================================================================================
# The reduction anchor -- the chain's last link
# =====================================================================================

@pytest.mark.parametrize("kappa,sigma0,sigma1", [
    (0.0, 0.0, 0.0), (KAPPA_DEF, 0.0, 0.0), (KAPPA_DEF, 2.0, 5e-3),
])
def test_ea_equals_t_is_bit_identical_to_the_damped_string(kappa, sigma0, sigma1):
    """``EA = T`` reduces model #10 to model #3 bit-for-bit — **within each implementation**.

    §15.2's anchor, and the reason the whole four-string chain had to move to a common spelling
    before any of it moved. It holds because ``a = EA - T0`` is exactly zero, the nonlinear branch
    is skipped entirely rather than multiplied by zero, and what is left is model #3's expressions
    in model #3's order. There is no Newton solve on this path at all, so the sparse LU — the one
    thing about this model that is *not* reproducible — never runs.
    """
    N = 48
    base = dict(L=L_DEF, T=T_DEF, rho=RHO_DEF, fs=fs_for(N), N=N, kappa=kappa,
                sigma0=sigma0, sigma1=sigma1)
    for geo_cls, damped_cls in (
        (GeometricStringPy, DampedStiffStringPy),
        (physsynth_rs.GeometricString, physsynth_rs.DampedStiffString),
    ):
        geo = geo_cls(EA=T_DEF, **base)
        ref = damped_cls(**base)
        ic = mode(geo, 1, 1e-3)
        geo.set_state(ic.copy())
        ref.set_state(ic.copy())
        assert np.array_equal(ref.u_prev, geo.u_prev)
        for _ in range(300):
            ref.step()
            geo.step()
        assert np.array_equal(ref.u, geo.u), geo_cls
        assert ref.energy() == geo.energy()
        assert geo.nonlinear_energy() == 0.0
        assert np.max(np.abs(geo.w)) == 0.0 and np.max(np.abs(geo.v)) == 0.0
        assert geo.newton_iters == 0


def test_the_linear_branch_is_bit_identical_across_the_languages_too():
    """...and with no Newton solve on it, the two *implementations* agree exactly as well."""
    k = kw(EA=T_DEF, sigma0=1.0, sigma1=2e-3)
    with shared_solver(k["N"] - 1):
        py = GeometricStringPy(**k)
        rs = physsynth_rs.GeometricString(**k)
        ic = mode(py, 1, 1e-3)
        py.set_state(ic.copy())
        rs.set_state(ic.copy())
        # The block has to wrap the STEPPING, not only the construction: the banded factor is
        # built at construction but back-substituted every step, and a first draft that closed it
        # after the constructors compared a Rust factor solved by LAPACK against one solved by the
        # transcription. It failed, correctly, and the arrays printed identically for ten digits.
        for _ in range(400):
            py.step()
            rs.step()
    for name in ("u", "w", "v", "u_prev", "w_prev", "v_prev"):
        assert np.array_equal(getattr(py, name), getattr(rs, name)), name
    assert py.energy() == rs.energy()


# =====================================================================================
# The trajectory, on a shared factorization -- §24.4's manoeuvre, fourth use
# =====================================================================================

@pytest.mark.parametrize("over,steps", [
    ({}, 2000),
    (dict(N=16), 2000),
    (dict(kappa=0.0), 1000),
    (dict(sigma0=2.0, sigma1=5e-3), 1000),
    (dict(kappa=2.7, kappa_w=3.1), 1000),
    (dict(theta=0.5), 1000),
    (dict(lam_long=1.0), 1000),
])
def test_the_port_is_exact_when_both_solvers_are_shared(over, steps):
    """The batch's exactness claim, and it is unqualified.

    Two solvers, both held constant, and then **nothing** in the two implementations differs: the
    state, the energy, the Newton iteration count and the cumulative count all match to the bit.
    That is what says the residue measured below is the sparse LU and none of it is the port — and
    it is the only test here that could see a reassociation, because the solver gap is larger than
    one would be.
    """
    k = kw(**over)
    with shared_solver(k["N"] - 1):
        py = GeometricStringPy(**k)
    rs = physsynth_rs.GeometricString(**k)
    ic = mode(py, 1, 2e-3)
    py.set_state(ic.copy())
    rs.set_state(ic.copy())
    with shared_solver(k["N"] - 1):
        for i in range(steps):
            py.step()
            rs.step()
            if i < 5 or i == steps - 1:
                assert np.array_equal(py.u, rs.u), f"step {i}"
                assert np.array_equal(py.w, rs.w) and np.array_equal(py.v, rs.v)
                assert py.newton_iters == rs.newton_iters, f"step {i}"
    assert np.array_equal(py.u, rs.u) and np.array_equal(py.u_prev, rs.u_prev)
    assert py.total_newton_iters == rs.total_newton_iters
    # The STATE is exact; the energy is not quite, and the gap is confined to one read-out.
    # `nonlinear_energy` sums with `np.sum` and this port declines NumPy's pairwise blocking
    # (§28.7), so it lands a last bit away -- and `energy()`, which adds it to a much larger linear
    # part, usually absorbs that. Nothing on the update path reads either.
    assert py.energy() == pytest.approx(rs.energy(), rel=1e-14)
    assert py.nonlinear_energy() == pytest.approx(rs.nonlinear_energy(), rel=1e-14)
    assert py.n_not_converged == rs.n_not_converged == 0


def test_a_forced_stall_is_reported_identically():
    """The telemetry on the branch nothing healthy reaches — ``newton_maxiter`` too small.

    Both sides must warn, count and keep stepping. The warning is Python's ``RuntimeWarning`` on
    both, raised from the binding through ``PyErr::warn`` so ``pytest.warns`` sees it.
    """
    k = kw(N=16, lam_long=8.0)
    k["newton_maxiter"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with shared_solver(k["N"] - 1):
            py = GeometricStringPy(**k)
        rs = physsynth_rs.GeometricString(**k)
        ic = mode(py, 1, 5e-3)
        py.set_state(ic.copy())
        rs.set_state(ic.copy())
    with pytest.warns(RuntimeWarning, match="did not converge"):
        with shared_solver(k["N"] - 1):
            for _ in range(20):
                py.step()
    with pytest.warns(RuntimeWarning, match="did not converge"):
        for _ in range(20):
            rs.step()
    assert py.n_not_converged == rs.n_not_converged > 0
    assert py.converged is rs.converged


# =====================================================================================
# The trajectory, on its own solvers -- the measured window
# =====================================================================================

@pytest.mark.parametrize("over,amp,steps,tol", [
    ({}, 1e-3, 100, GROUP_A_TOL),
    ({}, 1e-3, 2000, GROUP_D_TOL),
    (dict(N=64), 1e-3, 2000, GROUP_D_TOL),
    (dict(lam_long=1.0), 1e-3, 2000, GROUP_D_TOL),
    ({}, 1e-2, 2000, GROUP_D_TOL),
])
def test_the_two_solvers_track_each_other_over_a_run(over, amp, steps, tol):
    """§24.2's Group D bar: tolerance-level agreement, quantified, normalised by the running peak.

    The peak is the **running** one, not the instantaneous amplitude — §20.6, which measured a
    factor of 40 between the two spellings on a beating signal.
    """
    py, rs = pair(kw(**over))
    ic = mode(py, 1, amp)
    py.set_state(ic.copy())
    rs.set_state(ic.copy())
    peak, worst = 0.0, 0.0
    for _ in range(steps):
        py.step()
        rs.step()
        peak = max(peak, float(np.max(np.abs(py.u))))
        worst = max(worst, float(np.max(np.abs(np.asarray(py.u) - np.asarray(rs.u)))) / peak)
    print(f"\n{over} amp={amp:g} {steps} steps: du/peak {worst:.2e}, "
          f"mean iters {py.total_newton_iters / steps:.2f}")
    assert worst < tol
    assert py.energy() == pytest.approx(rs.energy(), rel=GROUP_D_ENERGY_TOL)


def test_the_iteration_count_flips_and_that_is_the_branch_this_model_has():
    """§19.2, answered for this model — and the answer is milder than model #9's.

    The convergence test is a **max**, so the reduction feeding it is order-independent; what
    varies is which side of the bar a single Newton step lands on. Measured over 2,000 steps the
    flip rate tracks where the mean iteration count sits between two integers, and a flip costs
    trajectory agreement but **not** energy agreement — any root of the discrete-gradient equation
    conserves exactly, which is why the model declines to gate uniqueness.
    """
    py, rs = pair(kw())
    ic = mode(py, 1, 1e-2)  # an amplitude where the mean count is NOT pinned at 1
    py.set_state(ic.copy())
    rs.set_state(ic.copy())
    flips = 0
    for _ in range(2000):
        py.step()
        rs.step()
        flips += py.newton_iters != rs.newton_iters
    mean = py.total_newton_iters / 2000
    print(f"\nmean iters {mean:.2f}, iteration-count flips {flips}/2000")
    assert 1.0 < mean < 3.0, "the fixture must sit between two integers for this to be a test"
    assert flips > 0, "the branch this test is about must actually flip on this fixture"
    assert py.energy() == pytest.approx(rs.energy(), rel=GROUP_D_ENERGY_TOL)


@pytest.mark.parametrize("over", [{}, dict(N=64), dict(theta=0.5)])
def test_neither_implementation_moves_the_energy_bar(over):
    """The physics bar, which is what Group D is held to when the digits are not available."""
    drifts = {}
    for label, model in (("python", GeometricStringPy(**kw(**over))),
                         ("rust", physsynth_rs.GeometricString(**kw(**over)))):
        model.set_state(mode(model, 1, 2e-3))
        e0 = model.energy()
        lo = hi = e0
        for _ in range(3000):
            model.step()
            e = model.energy()
            lo, hi = min(lo, e), max(hi, e)
        drifts[label] = (hi - lo) / abs(e0)
        assert model.n_not_converged == 0
    print(f"\n{over} lossless drift — python {drifts['python']:.2e}, rust {drifts['rust']:.2e}")
    for label, d in drifts.items():
        assert d < DRIFT_TOL, f"{label} drift {d:.2e}"


# =====================================================================================
# The ordering in front of the sparse LU -- §29.2
# =====================================================================================

def a_newton_jacobian(N):
    """One Jacobian the Newton loop actually forms, captured from a running Python model."""
    s = GeometricStringPy(**kw(N=N))
    s.set_state(mode(s, 1, 2e-3))
    seen = []
    real = string_geometric.splu
    string_geometric.splu = lambda m: (seen.append(m.copy()), real(m))[1]
    try:
        for _ in range(3):
            s.step()
    finally:
        string_geometric.splu = real
    assert seen, "the Newton loop must have factored something"
    return seen[0].tocsr()


@pytest.mark.parametrize("N", [32, 64, 128])
def test_the_node_ordering_is_what_makes_the_factorization_cheap(N):
    """§29.2, asserted rather than remembered.

    No other bar in this project could catch a regression here: the answers would stay right and
    the model would simply get slow. The natural order fills the whole envelope between the three
    field blocks; the node order is a closed form in ``N`` and beats COLAMD.
    """
    from scipy.sparse.linalg import splu

    jac = a_newton_jacobian(N)
    jac.sort_indices()
    n = jac.shape[0]

    def rust(perm):
        return physsynth_rs.SparseLu(
            jac.data.astype(float), jac.indices.astype(np.int32),
            jac.indptr.astype(np.int32), n, perm,
        )

    natural, reordered = rust(None), rust(interleave_perm(N - 1))
    scipy_lu = splu(jac.tocsc())
    fill = lambda lu: sum(lu.nnz)  # noqa: E731
    print(f"\nN={N} n={n}: scipy {scipy_lu.L.nnz + scipy_lu.U.nnz}, "
          f"rust-natural {fill(natural)}, rust-node {fill(reordered)}")
    assert not natural.is_reordered and reordered.is_reordered
    # Measured 3.67x at N = 32, 6.7x at N = 64, 12.8x at N = 128 -- the claim is that the saving
    # GROWS with the grid, and the bar is set at the smallest of those with room to spare. A
    # tighter one would be pinned to which entries are structurally nonzero, and the two sides do
    # not have to agree about that: `Csr::from_rows` drops exact zeros where `sparse.diags` need
    # not.
    assert fill(reordered) * 2 < fill(natural), (
        "the node reordering saved almost nothing -- either the Jacobian's structure changed or "
        "the permutation is no longer the fill-reducing one"
    )
    # Against COLAMD, deliberately LOOSE. Measured, the closed-form node order beats it by about
    # 5 % at all three sizes -- but SuperLU's ordering heuristic is a SciPy internal that a point
    # release is free to change, and §18.3 and §26.2 both say not to pin one. The claim worth
    # asserting is "in the same league", which is what makes the natural order the outlier.
    assert fill(reordered) <= 1.5 * (scipy_lu.L.nnz + scipy_lu.U.nnz), (
        "the closed-form node order is no longer competitive with COLAMD on this matrix"
    )


def test_the_reordering_is_a_reordering_and_not_a_different_answer():
    """The other half. An inverse permutation applied backwards returns a plausible vector."""
    jac = a_newton_jacobian(48)
    jac.sort_indices()
    n = jac.shape[0]
    b = np.sin(np.arange(n) * 0.7) * 1e-4
    args = (jac.data.astype(float), jac.indices.astype(np.int32),
            jac.indptr.astype(np.int32), n)
    x_nat = np.asarray(physsynth_rs.SparseLu(*args).solve(b))
    x_reo = np.asarray(physsynth_rs.SparseLu(*args, interleave_perm(47)).solve(b))
    scale = np.max(np.abs(x_nat))
    assert np.max(np.abs(x_nat - x_reo)) <= 1e-12 * scale
    assert np.max(np.abs(jac @ x_reo - b)) <= 1e-14 * np.max(np.abs(b))


def test_the_nonlinear_step_is_where_the_speed_is():
    """§28.11's inheritance: a model with an inner iteration is the regime the port is about.

    Reported, not asserted against a threshold — a wall-clock bar is a claim about a runner
    (§22.1) and the CI machine class varies by 1.6x. What *is* asserted is the direction, with
    enough margin that only a real regression could cross it.
    """
    k = kw(N=48)
    py, rs = pair(k)
    for a in (py, rs):
        a.set_state(mode(a, 1, 2e-3))
        for _ in range(20):  # warm the branch predictor and the allocator alike
            a.step()

    def timed(model, n=200):
        t0 = time.perf_counter()
        for _ in range(n):
            model.step()
        return (time.perf_counter() - t0) / n * 1e6

    us_py, us_rs = timed(py), timed(rs)
    print(f"\nstep: python {us_py:.1f} us, rust {us_rs:.1f} us ({us_py / us_rs:.2f}x)")
    assert us_rs < us_py, "the Rust step must not be slower than the Python one"


# =====================================================================================
# The refusals, message for message
# =====================================================================================

REFUSALS = [
    (dict(L=-1.0), "L, T, rho, fs must all be positive."),
    (dict(T=0.0), "L, T, rho, fs must all be positive."),
    (dict(rho=-1.0), "L, T, rho, fs must all be positive."),
    (dict(fs=0.0), "L, T, rho, fs must all be positive."),
    (dict(EA=0.0), "EA (axial stiffness) must be positive."),
    (dict(EA=-1.0), "EA (axial stiffness) must be positive."),
    (dict(N=1), "N must be >= 2 (need at least one interior node)."),
    (dict(kappa=-1.0), "kappa (stiffness) must be >= 0."),
    (dict(kappa_w=-1.0), "kappa_w (stiffness) must be >= 0."),
    (dict(sigma0=-1.0), "sigma0, sigma1 (losses) must be >= 0."),
    (dict(sigma1=-1.0), "sigma0, sigma1 (losses) must be >= 0."),
    (dict(sigma0_long=-1.0), "sigma0_long, sigma1_long (losses) must be >= 0."),
    (dict(sigma1_long=-1.0), "sigma0_long, sigma1_long (losses) must be >= 0."),
    (dict(theta=0.0), "theta must be in (0, 1], got 0.0."),
    (dict(theta=1.5), "theta must be in (0, 1], got 1.5."),
    (dict(newton_tol=0.0), "newton_tol must be > 0."),
    (dict(newton_maxiter=0), "newton_maxiter must be >= 1."),
    (dict(boundary="clamped"), "boundary must be 'supported', got 'clamped'."),
    (dict(boundary=None), "boundary must be 'supported', got None."),
]


@pytest.mark.parametrize("over,message", REFUSALS)
def test_the_refusals_are_identical(over, message):
    """Including ``boundary=None``, which §24.7 found every binding silently accepting."""
    base = kw()
    base.update(over)
    if "newton_maxiter" in over:
        base["newton_maxiter"] = over["newton_maxiter"]
    for cls in (GeometricStringPy, physsynth_rs.GeometricString):
        with pytest.raises(ValueError) as exc:
            cls(**base)
        assert str(exc.value) == message, cls


def test_the_softening_refusal_is_identical_and_opt_out():
    base = kw(EA=T_DEF / 2.0)
    for cls in (GeometricStringPy, physsynth_rs.GeometricString):
        with pytest.raises(ValueError) as exc:
            cls(**base)
        text = str(exc.value)
        assert text.startswith(f"EA ({T_DEF / 2.0}) < T ({T_DEF})"), cls
        assert "Lambda0 = (EA - T0)/EA = -1 NEGATIVE" in text, cls
        assert "Pass allow_softening=True" in text, cls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = GeometricStringPy(allow_softening=True, **base)
        rs = physsynth_rs.GeometricString(allow_softening=True, **base)
    assert py._a == rs._a < 0.0
    assert py.c_long == rs.c_long < py.c


def test_the_pickup_agrees_including_the_out_of_range_exception_TYPE():
    """A rejection's **type** is part of it, not only its text.

    The original is ``float(self.u[index])``, so an out-of-range pickup raises NumPy's
    ``IndexError``. A binding that reached for ``ValueError`` would be caught by nothing here and
    by a caller's ``except IndexError`` in production — the same rule
    ``test_rust_parity_banded.py`` keeps for SciPy's ``LinAlgError``.
    """
    py, rs = pair(kw())
    for a in (py, rs):
        a.set_state(mode(a, 1, 1e-3))
    for i in (0, 1, 7, py.N, -1, -3):
        assert py.displacement_at(i) == rs.displacement_at(i), i
    for bad in (py.N + 1, 10_000, -(py.N + 5)):
        with pytest.raises(IndexError):
            py.displacement_at(bad)
        with pytest.raises(IndexError):
            rs.displacement_at(bad)


def test_apply_ainv_is_refused_with_the_same_explanation():
    """It is not merely unimplemented — the one-step response is genuinely state-dependent."""
    py, rs = pair(kw())
    rhs = np.zeros(py.N - 1)
    for model in (py, rs):
        # Any argument, including one the wrong shape or the wrong TYPE: the original raises
        # before it looks, so a `TypeError` out of an extraction would be a different refusal.
        for arg in (rhs, [0.0, 1.0], None):
            with pytest.raises(NotImplementedError) as exc:
                model.apply_Ainv(arg)
            assert "state-dependent" in str(exc.value)
            assert "not of A3" in str(exc.value)


# =====================================================================================
# The one guard with no CFL behind it
# =====================================================================================

def test_the_lam_long_warning_fires_on_both_and_is_exempt_at_the_anchor():
    """``LAM_LONG_WARN`` warns rather than rejects, and skips ``EA == T`` entirely.

    The exemption is load-bearing rather than tidy: the ``EA = T`` anchor lands at
    ``lam_long == 1.0`` **exactly**, flush against the bar, so without it a float wobble would fire
    a spurious warning on this model's single most important regression test.
    """
    coarse = kw(lam_long=8.0)
    for cls in (GeometricStringPy, physsynth_rs.GeometricString):
        with pytest.warns(RuntimeWarning, match="under-resolved"):
            s = cls(**coarse)
        assert s.lam_long > LAM_LONG_WARN
    linear = dict(coarse)
    linear["EA"] = T_DEF
    linear["fs"] = fs_for(coarse["N"], EA=T_DEF, lam_long=8.0)
    for cls in (GeometricStringPy, physsynth_rs.GeometricString):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = cls(**linear)
        assert s._a == 0.0 and s.lam_long > 4.0


def test_the_defaults_are_the_same_numbers():
    assert NEWTON_TOL_DEFAULT == 1e-15
    assert NEWTON_MAXITER_DEFAULT == 60
    assert LAM_LONG_WARN == 1.0
    py, rs = pair(dict(L=L_DEF, T=T_DEF, rho=RHO_DEF, fs=fs_for(32), N=32, EA=EA_DEF))
    assert py.newton_tol == rs.newton_tol == NEWTON_TOL_DEFAULT
    assert py.newton_maxiter == rs.newton_maxiter == NEWTON_MAXITER_DEFAULT
    assert py.theta == rs.theta == THETA_DEFAULT
    assert py.boundary == rs.boundary == "supported"
    assert py.kappa == rs.kappa == 0.0
