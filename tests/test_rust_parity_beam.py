"""Rust vs NumPy/SciPy for the free-free beam — model #5b-pre, and the first of Group D.

``docs/dev/rust-migration-plan.md`` §24. This is the batch §4.1 designed: the smallest model that
factors a sparse matrix, ported so that the SuperLU hypothesis could be tested on 254 lines rather
than on the 3,925-line room. The hypothesis **fails**, and the shape of this file follows from how
it fails.

**What was measured, and what it retires.** §4.1 named three things a Rust port would have to
match to reproduce ``splu``'s digits, and the answers are: the column ordering is a **closed form
in n** and not a barrier; equilibration **never runs**, because SciPy calls ``gstrf`` and not the
``gssvx`` driver; and the pivot threshold is **a real obstacle after all**, though only above a
grid size the batch's first two fixtures did not reach — SuperLU takes the diagonal at ``N <= 48``
and starts swapping rows at ``N = 64``. All three are asserted below, the third one in the shape
its correction took, because that correction is §16.4's blind fixture arriving in the *measurement*
instead of in the model.

The thing that actually decides the hypothesis is a fourth, which §4.1 did not name: SuperLU is
**supernodal**. ``relax`` and ``panel_size`` visibly change the factors, and handed SuperLU's *own*
factors a longhand triangular solve still disagrees with ``lu.solve`` in ~20 % of entries at
~4e-16. That is a property of how SciPy *built* its copy — its blocking defaults, its patch level,
whether it linked an external BLAS — so it is recorded here and not asserted: §22.1's rule, that a
claim which is really about a machine does not belong in an equality.

**So the comparison has two halves, and only one of them is about the port.**

* With the solver held constant — the Python beam driven through the Rust factorization, i.e.
  ``test_rust_parity_strings.py``'s ``shared_solver()`` manoeuvre — the two beams are
  **bit-identical** in ``u``, ``u_prev`` and the history. That is the porting claim, and it is
  sharp: it is the only thing here that a mis-associated expression would break, because the
  solver gap below would otherwise swallow it (§19.4's finding, which is what that manoeuvre exists
  to prevent).
* With each side on its own solver, the beams separate, and **more than any earlier model**. ``K``'s
  nullspace is exactly ``{1, x}``, so along the rigid-body directions the beam is a *free particle*
  and a per-step difference is integrated twice. Measured at ``N = 32`` over 20,000 steps: the rigid
  component grows like ``t²`` — 3.7e-17, 8.8e-14, 6.9e-12, 3.3e-9 of amplitude at 1, 100, 1,000 and
  20,000 steps — while the elastic component saturates at ~1e-12 and the energy stays inside 7e-12
  relative. Damping *attenuates* the growth without removing it (3.3e-9 at ``sigma = 0`` against
  2.1e-10 at ``sigma = 100``), so this is a sixth agreement regime and the first one set by a
  **boundary condition** rather than by a nonlinearity, an amplitude or an attractor.

The consequence for every free-edge model in Phases 5 and 6, stated once: a parity bar reads the
rigid/elastic split or the energy. A bar on ``max|du|/amp`` will read as a failure that is not one.

Green with the flag and without it, per §16.4's convention — the assertions differ between the two
modes, the file does not.
"""

import contextlib

import numpy as np
import pytest
from scipy.sparse.linalg import splu

import physsynth.core.beam as beam_module
from physsynth.core.beam import THETA_DEFAULT, FreeBeamPy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

DRIFT_TOL = 1e-10  # CLAUDE.md's acceptance bar, which neither implementation may cross
ENERGY_GAP_TOL = 1e-10  # the two solvers' energies, measured at <= 7.2e-12 over four fixtures

L_DEF, RHO_DEF, FS_DEF = 1.0, 0.005, 48000.0
KAPPA_DEF, N_DEF = 20.0, 32


def rs_splu(a):
    """``scipy.sparse.linalg.splu``'s interface over the Rust sparse LU."""
    csr = a.tocsr()
    return physsynth_rs.SparseLu(
        np.ascontiguousarray(csr.data, dtype=float),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.indptr, dtype=np.int32),
        csr.shape[0],
    )


@contextlib.contextmanager
def shared_solver():
    """Put the Python beam on the Rust sparse LU for the duration of the block.

    ``beam.py`` captures ``splu`` at import, so patching the captured name is the only way to hold
    the solver constant while the *model* varies — which is what turns the comparison below into a
    claim about the port instead of a claim about SuperLU. Unlike the strings' version of this
    manoeuvre, it is never a no-op: the Rust beam factors internally and the Python one calls
    SciPy, under the flag as much as without it.
    """
    saved = beam_module.splu
    beam_module.splu = rs_splu
    try:
        yield
    finally:
        beam_module.splu = saved


def make_pair(*, N=N_DEF, kappa=KAPPA_DEF, sigma=0.0, theta=THETA_DEFAULT, shared):
    """A Python beam and a Rust beam with identical parameters, optionally sharing a solver."""
    kwargs = dict(L=L_DEF, rho=RHO_DEF, fs=FS_DEF, N=N, kappa=kappa, sigma=sigma, theta=theta)
    ctx = shared_solver() if shared else contextlib.nullcontext()
    with ctx:
        py = FreeBeamPy(**kwargs)
    return py, physsynth_rs.FreeBeam(**kwargs)


def bump(x, L=L_DEF, amplitude=1e-3):
    """A smooth full-field initial displacement — no node is clamped on a free-free beam."""
    return amplitude * np.sin(np.pi * x / L) ** 2


def rigid_elastic(beam, d):
    """Split a field ``d`` into its rigid-body ``{1, x}`` part and the rest, in ``W``'s inner
    product.

    The rigid space is exactly ``K``'s nullspace, so this is the split the scheme itself makes:
    the elastic part is restored by the stiffness every step and the rigid part is not.
    """
    v = np.column_stack([np.ones_like(beam.x), beam.x])
    gram = v.T @ (beam.w[:, None] * v)
    rigid = v @ np.linalg.solve(gram, v.T @ (beam.w * d))
    return rigid, d - rigid


# -- structure: exact, always ------------------------------------------------------------------


def test_scalar_parameters_are_identical():
    py, rs = make_pair(shared=False)
    for name in ("L", "rho", "fs", "N", "kappa", "sigma", "theta", "h", "k", "mu", "boundary"):
        assert getattr(py, name) == getattr(rs, name), f"{name} differs"


def test_grid_and_mass_weights_are_bit_identical():
    py, rs = make_pair(shared=False)
    assert np.array_equal(py.x, rs.x), "the grid differs"
    assert np.array_equal(py.w, rs.w), "the lumped mass weights differ"


@pytest.mark.parametrize("N", [4, 8, 32, 64])
def test_the_operator_pair_is_bit_identical_including_index_order(N):
    # `K` multiplies the state twice per step, so its stored column order is part of the
    # arithmetic, not a detail of the container -- §18.2's finding, asserted rather than assumed.
    py, rs = make_pair(N=N, shared=False)
    for name in ("K", "W"):
        p, r = getattr(py, name), getattr(rs, name)
        assert p.nnz == r.nnz, f"{name}: nnz {p.nnz} vs {r.nnz}"
        assert np.array_equal(p.indptr, r.indptr), f"{name}: indptr differs"
        assert np.array_equal(p.indices, r.indices), f"{name}: index ORDER differs"
        assert np.array_equal(p.data, r.data), f"{name}: data differs"


def test_the_initial_previous_field_is_bit_identical():
    # `u^{-1}` is built from `K @ u0` and a per-node divide -- no solve, so this is exact whatever
    # the solvers do, and it isolates `set_state` from everything downstream of it.
    py, rs = make_pair(shared=False)
    u0 = bump(py.x)
    py.set_state(u0)
    rs.set_state(u0)
    assert np.array_equal(py.u, rs.u)
    assert np.array_equal(py.u_prev, rs.u_prev)


@pytest.mark.parametrize("bad", ["clamped", "supported", 3, None])
def test_an_unsupported_boundary_is_refused_with_the_same_message(bad):
    kwargs = dict(L=L_DEF, rho=RHO_DEF, fs=FS_DEF, N=8, kappa=KAPPA_DEF)
    with pytest.raises(ValueError) as py_err:
        FreeBeamPy(**kwargs, boundary=bad)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.FreeBeam(**kwargs, boundary=bad)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(L=0.0), dict(rho=-1.0), dict(fs=0.0), dict(kappa=0.0), dict(kappa=-1.0),
        dict(N=3), dict(N=-2), dict(sigma=-1e-9), dict(theta=0.0), dict(theta=1.5),
    ],
)
def test_rejections_carry_the_same_message(kwargs):
    base = dict(L=L_DEF, rho=RHO_DEF, fs=FS_DEF, N=8, kappa=KAPPA_DEF)
    base.update(kwargs)
    with pytest.raises(ValueError) as py_err:
        FreeBeamPy(**base)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.FreeBeam(**base)
    assert str(py_err.value) == str(rs_err.value)


# -- the port, with the solver held constant: bit-identical ------------------------------------


@pytest.mark.parametrize(
    "N,kappa,sigma,theta",
    [
        (32, 20.0, 0.0, THETA_DEFAULT),
        (48, 20.0, 0.0, 0.5),
        (32, 2.0, 4.0, 0.25),
        (8, 50.0, 0.0, 1.0),
    ],
)
def test_one_solver_makes_the_two_beams_bit_identical(N, kappa, sigma, theta):
    # THE porting claim. Everything else in this file is a measurement of the solver gap; this is
    # the only assertion that fails if an expression was reassociated, and it is sharp precisely
    # because the gap is removed. 2,000 steps rather than 20,000 keeps the file fast -- the claim
    # is exactness, which does not decay with run length.
    py, rs = make_pair(N=N, kappa=kappa, sigma=sigma, theta=theta, shared=True)
    u0 = bump(py.x)
    py.set_state(u0)
    rs.set_state(u0)
    for n in range(2000):
        py.step()
        rs.step()
        if n % 500 == 0 or n == 1999:
            assert np.array_equal(py.u, rs.u), f"u diverged at step {n + 1}"
            assert np.array_equal(py.u_prev, rs.u_prev), f"u_prev diverged at step {n + 1}"
    assert py.n == rs.n == 2000


def test_energy_agrees_to_the_reduction_under_a_shared_solver():
    # `energy()` is a `np.dot` on the Python side and a left-to-right loop on the Rust one, so it
    # is the one observable that is NOT bit-identical even with the solver held constant -- §14.2.
    # Measured at ~3e-16 relative, which is where a BLAS reduction lives and nowhere else.
    py, rs = make_pair(shared=True)
    u0 = bump(py.x)
    py.set_state(u0)
    rs.set_state(u0)
    worst = 0.0
    for _ in range(500):
        py.step()
        rs.step()
        worst = max(worst, abs(py.energy() - rs.energy()) / abs(py.energy()))
    assert worst < 1e-14, f"the energies differ by {worst:.3e}, which is more than a reduction"


# -- the solver gap, each side on its own factorization ----------------------------------------


@pytest.mark.parametrize("sigma", [0.0, 20.0])
def test_the_divergence_lives_in_the_rigid_body_directions(sigma):
    # The batch's headline, as a bar. `K`'s nullspace is `{1, x}`; along it the beam is a free
    # particle, so the solver difference integrates twice and the rigid part grows like t^2 while
    # the elastic part saturates. Both bounds are ~4x above the measured values, which is this
    # project's usual headroom and not a fit.
    py, rs = make_pair(sigma=sigma, shared=False)
    u0 = bump(py.x)
    py.set_state(u0)
    rs.set_state(u0)
    peak = np.abs(py.u).max()
    for _ in range(5000):
        py.step()
        rs.step()
        peak = max(peak, np.abs(py.u).max())  # running peak: §20.6, never the instantaneous one
    rigid, elastic = rigid_elastic(py, py.u - rs.u)
    r, e = np.abs(rigid).max() / peak, np.abs(elastic).max() / peak
    assert r < 1e-9, f"the rigid divergence {r:.3e} is larger than measured"
    assert e < 1e-11, f"the elastic divergence {e:.3e} is larger than measured"
    assert e < r, f"the divergence is not rigid-dominated after 5,000 steps: {e:.3e} vs {r:.3e}"


def test_the_energies_stay_together_while_the_fields_separate():
    # The complement of the test above, and the reason the regime is survivable: the quantity the
    # acceptance contract is written on does NOT inherit the t^2 growth.
    py, rs = make_pair(shared=False)
    u0 = bump(py.x)
    py.set_state(u0)
    rs.set_state(u0)
    worst = 0.0
    for _ in range(5000):
        py.step()
        rs.step()
        worst = max(worst, abs(py.energy() - rs.energy()) / abs(py.energy()))
    assert worst < ENERGY_GAP_TOL, f"the energies diverged by {worst:.3e}"


# -- the physics bars, on the Rust side, because those are the acceptance contract --------------


def test_the_rust_beam_conserves_energy():
    rs = physsynth_rs.FreeBeam(L=L_DEF, rho=RHO_DEF, fs=FS_DEF, N=N_DEF, kappa=KAPPA_DEF)
    rs.set_state(bump(rs.x))
    e0 = rs.energy()
    worst = 0.0
    for _ in range(5000):
        rs.step()
        worst = max(worst, abs(rs.energy() - e0) / abs(e0))
    assert worst < DRIFT_TOL, f"lossless drift {worst:.3e} crosses the acceptance bar"


def test_the_rust_beam_is_passive_when_lossy():
    rs = physsynth_rs.FreeBeam(
        L=L_DEF, rho=RHO_DEF, fs=FS_DEF, N=N_DEF, kappa=KAPPA_DEF, sigma=20.0
    )
    rs.set_state(bump(rs.x))
    previous = rs.energy()
    for n in range(2000):
        rs.step()
        current = rs.energy()
        assert current <= previous * (1.0 + 1e-12), f"energy rose at step {n + 1}"
        previous = current


# -- what §4.1 assumed about SuperLU, and what is actually true ---------------------------------


def beam_matrix(N, sigma=0.0):
    """``A = (1 + sigma k) W + theta k^2 kappa^2 K`` — the matrix the beam factors."""
    b = FreeBeamPy(L=L_DEF, rho=RHO_DEF, fs=FS_DEF, N=N, kappa=KAPPA_DEF, sigma=sigma)
    return (1.0 + b.sigma * b.k) * b.W + b.theta * b.k * b.k * b.kappa * b.kappa * b.K


def colamd_order(n):
    """SuperLU's COLAMD order for this pentadiagonal family, as a closed form in ``n``."""
    p = np.arange(n)
    if n >= 6:
        p[n - 5 : n - 1] = [n - 3, n - 2, n - 5, n - 4]
    return p


@pytest.mark.parametrize("N", [4, 5, 8, 12, 16, 17, 32, 33, 64])
def test_the_reference_column_order_is_a_closed_form(N):
    # §4.1's first named obstacle, retired. COLAMD on this family is the identity except that the
    # two pairs at n-5, n-4 and n-3, n-2 are exchanged -- deterministic, and derivable without
    # reproducing COLAMD. Pinned because the *reason* the ordering was not reproduced in Rust is
    # that it buys nothing, and that reason is only honest while the order stays this simple.
    lu = splu(beam_matrix(N).tocsc())
    assert np.array_equal(lu.perm_c, colamd_order(N + 1))


@pytest.mark.parametrize("N", [8, 32, 64])
def test_the_reference_never_equilibrates(N):
    # §4.1's third named obstacle, retired: `Equil` is a no-op in BOTH directions, because SciPy
    # calls `gstrf` -- the factorization -- and not the `gssvx` driver that would have computed row
    # and column scalings first. So there is no hidden scaling to reproduce.
    a = beam_matrix(N).tocsc()
    base = splu(a)
    for kwargs in (dict(options=dict(Equil=True)), dict(options=dict(Equil=False))):
        other = splu(a, **kwargs)
        assert np.array_equal(other.L.toarray(), base.L.toarray()), f"{kwargs} changed L"
        assert np.array_equal(other.U.toarray(), base.U.toarray()), f"{kwargs} changed U"


def test_the_reference_pivot_threshold_is_not_moot_and_the_grid_decides():
    # §4.1's second named obstacle, and this one is REAL -- which the batch's first two fixtures
    # missed, because both were below the transition. The beam's matrix is diagonally largest on a
    # coarse grid and stops being so as the stiffness term outgrows the mass: SuperLU takes the
    # diagonal at N <= 48 and starts swapping rows at N = 64, filling U as it goes (600 entries
    # would be the band at N = 200; it stores 773). §16.4's blind fixture, arriving in the
    # measurement rather than in the model.
    for N in (8, 16, 32, 48):
        lu = splu(beam_matrix(N).tocsc())
        assert np.array_equal(lu.perm_r, lu.perm_c), f"N = {N} pivoted after all"
    for N in (64, 96, 128, 200):
        a = beam_matrix(N).tocsc()
        lu = splu(a)
        assert not np.array_equal(lu.perm_r, lu.perm_c), f"N = {N} no longer pivots"
        band = (a.nnz - (N + 1)) // 2
        assert lu.U.nnz > band + N + 1, f"N = {N}: the reference pivot no longer fills U"


@pytest.mark.parametrize("N", [8, 32, 64, 128, 200])
def test_the_rust_factorization_declines_the_pivot_the_reference_takes(N):
    # The consequence, stated rather than smoothed over: above the transition the two eliminations
    # make a DIFFERENT discrete decision, not merely a different last bit. The Rust side prefers
    # the diagonal (`sparse_lu.rs`'s DIAG_PIVOT_THRESH = 0.1), which is legal here for a reason
    # SuperLU has no way to know -- `A` is symmetric positive definite, so elimination without any
    # pivoting is unconditionally stable, and that is what makes Cholesky a valid algorithm for it.
    # The payoff is measurable: no fill at any size, against the reference's growing U.
    a = beam_matrix(N)
    lu = rs_splu(a)
    assert lu.is_natural, "the Rust factorization pivoted"
    below = (a.nnz - (N + 1)) // 2
    assert lu.nnz == (below, below + N + 1), f"the Rust factorization filled in: {lu.nnz}"


def test_the_divergence_grows_where_the_reference_starts_pivoting():
    # The price of that disagreement, measured rather than assumed. Below the transition the two
    # beams differ only by rounding; above it they also eliminate differently, and the rigid-body
    # divergence at 5,000 steps grows by more than an order of magnitude between N = 48 and N = 96.
    # Still nowhere near a physics bar -- the energies agree to ~1e-12 at both — but it is the
    # reason this file measures N-dependence at all.
    worst = {}
    for N in (48, 96):
        py, rs = make_pair(N=N, shared=False)
        u0 = bump(py.x)
        py.set_state(u0)
        rs.set_state(u0)
        peak = np.abs(py.u).max()
        for _ in range(5000):
            py.step()
            rs.step()
            peak = max(peak, np.abs(py.u).max())
        rigid, _ = rigid_elastic(py, py.u - rs.u)
        worst[N] = np.abs(rigid).max() / peak
        assert abs(py.energy() - rs.energy()) / abs(py.energy()) < ENERGY_GAP_TOL
    assert worst[96] > 5.0 * worst[48], f"the pivot disagreement stopped costing anything: {worst}"
    assert worst[96] < 1e-7, f"the divergence at N = 96 is larger than measured: {worst[96]:.3e}"


@pytest.mark.parametrize("N", [8, 32, 64])
def test_both_factorizations_invert_the_matrix_they_were_given(N):
    # The bar that does not care which library ran: whatever the digits, each solve must return an
    # x with A x = b. Held to Group A rather than to bit-identity, deliberately.
    a = beam_matrix(N)
    rng = np.random.default_rng(20260828)
    for _ in range(5):
        b = rng.standard_normal(N + 1)
        for solver in (splu(a.tocsc()), rs_splu(a)):
            x = solver.solve(b)
            residual = np.abs(a @ x - b).max() / np.abs(b).max()
            assert residual < 1e-13, f"{type(solver).__name__} residual {residual:.3e}"
