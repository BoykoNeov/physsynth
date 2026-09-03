"""Rust vs LAPACK for the banded Cholesky — the first swap in the migration that changes numbers.

``docs/dev/rust-migration-plan.md`` §15. Every earlier batch could open with "these two must agree
to the bit, and here is where they stop". This one cannot, and the reason belongs before the
assertions rather than after:

* the **factor** is transcribable and *nearly* reproduces LAPACK. It reproduces it exactly if the
  rank-1 update fuses its multiply-add — which is a property of the ``DSYR`` kernel OpenBLAS
  selects at run time, not of the algorithm. §14.2 refused that class of claim and this file keeps
  the refusal, so the factor is held to a tolerance and the *count* of exact agreements is
  measured rather than asserted;
* the **solve** is not transcribable at all. Per-element candidate elimination over {forward,
  reverse} x {plain, fused} x {divide, reciprocal} left disjoint candidate sets on a single
  15-element system, and one element admitted nothing. OpenBLAS's ``DTBSV`` is a blocked kernel.

So the interesting assertions here were not "Rust equals LAPACK". They were:

1. ~~the two solvers agree to the plan's Group A target over a short run, and the *physics* bars
   are untouched at any length — the hand-off §4 describes;~~ **retired with unit 1's deletion,
   2026-09-03** (plan §42). It compared one string built on LAPACK against the same string built on
   Rust, by patching the captured solver name in each model — and the four models no longer capture
   a solver name, because they no longer have a Python body to capture it in. There is one string
   now and it factors inside the crate. The measurement the test carried is kept below rather than
   lost, because its *shape* was the finding.
2. **the four models still agree with each other exactly.** That is the property this batch existed
   to protect, it is why the solver was ported before any of its callers, and it **survives the
   deletion unchanged** — see the last test in this file, which needed only its patch removed.

The **primitive** half of this file (Rust against `scipy.linalg`, its error messages and its shape
refusals) is untouched by any of this and is not going anywhere: SciPy is the comparand there, and
SciPy is not being deleted.
"""

import numpy as np
import pytest
from scipy.linalg import LinAlgError
from scipy.linalg import cho_solve_banded as sp_cho_solve
from scipy.linalg import cholesky_banded as sp_cholesky

from physsynth.core import string_damped, string_geometric, string_nonlinear, string_stiff
from physsynth.core.exciter import triangular_pluck

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target, and — per §14.4 — a SHORT-run one. The banded solve feeds
# straight back into the state, so the difference grows with the run rather than saturating.
GROUP_A_TOL = 1e-13

# `DRIFT_TOL` and `MODELS` were here. Both went with the two retired tests below: the first was
# CLAUDE.md's 1e-10 energy bar applied to LAPACK and Rust side by side, and the second was the
# tuple `rust_solver()` patched. Neither has a subject any more.


def rs_cholesky(ab, lower=False):
    return physsynth_rs.cholesky_banded_upper(np.ascontiguousarray(ab, dtype=float))


def rs_cho_solve(cb_and_lower, b):
    cb, _lower = cb_and_lower
    return physsynth_rs.cho_solve_banded_upper(
        np.ascontiguousarray(cb, dtype=float), np.ascontiguousarray(b, dtype=float)
    )


# `rust_solver()` stood here: a context manager that patched `cholesky_banded` /
# `cho_solve_banded` on all four models, because they captured those names at import and patching
# the captured name was the only honest way to build one model on each solver in a single process.
# It is gone with the bodies that did the capturing. `tests/test_stability.py`'s captured-binding
# guard is retired for the same reason and says so at more length: a Rust model does not capture a
# Python solver.


def model_band(n, kappa, sigma=0.0):
    """A pentadiagonal SPD band shaped like the theta-scheme's ``A``."""
    ab = np.zeros((3, n))
    ab[2, :] = 1.0 + sigma + 2.0 + 6.0 * kappa + 0.011 * np.arange(n)
    ab[1, 1:] = -1.0 - 4.0 * kappa
    ab[0, 2:] = kappa
    return ab


# =====================================================================================
# The primitive
# =====================================================================================


@pytest.mark.parametrize("n", [3, 7, 16, 33, 128, 257])
@pytest.mark.parametrize("kappa", [0.0, 0.3, 7.25])
def test_factor_agrees_with_lapack(n, kappa):
    ab = model_band(n, kappa)
    rs, sp = rs_cholesky(ab), sp_cholesky(ab, lower=False)
    assert rs.shape == sp.shape and rs.dtype == sp.dtype
    band = sp != 0.0
    assert np.allclose(rs[band], sp[band], rtol=GROUP_A_TOL, atol=0.0)
    # The corners outside the band are untouched by both, and a transcription that wrote into them
    # would still pass the comparison above.
    assert np.array_equal(rs[~band], sp[~band])


@pytest.mark.parametrize("n", [3, 7, 16, 33, 128, 257])
@pytest.mark.parametrize("kappa", [0.0, 0.3, 7.25])
def test_solve_agrees_with_lapack(n, kappa):
    ab = model_band(n, kappa)
    cb = sp_cholesky(ab, lower=False)  # the SAME factor both sides, so this measures the SOLVE
    rng = np.random.default_rng(20260827 + n)
    for _ in range(5):
        b = rng.standard_normal(n) * 1e-3
        rs = rs_cho_solve((cb, False), b)
        sp = sp_cho_solve((cb, False), b)
        assert np.max(np.abs(rs - sp)) <= GROUP_A_TOL * np.max(np.abs(sp))


def test_the_solve_actually_inverts_the_matrix():
    # Agreeing with LAPACK to 1e-13 is not the same as being right: a solver that returned a
    # slightly-scaled `b` would also agree to some tolerance on a well-conditioned band.
    n, kappa = 64, 2.0
    ab = model_band(n, kappa)
    cb = rs_cholesky(ab)
    a = np.diag(ab[2]) + np.diag(ab[1, 1:], 1) + np.diag(ab[1, 1:], -1)
    a = a + np.diag(ab[0, 2:], 2) + np.diag(ab[0, 2:], -2)
    x_true = np.sin(np.arange(n) * 0.31) * 1e-3
    x = rs_cho_solve((cb, False), a @ x_true)
    assert np.max(np.abs(x - x_true)) <= 1e-12 * np.max(np.abs(x_true))


def test_how_often_the_factor_is_exact_is_measured_not_asserted():
    """The fused multiply-add, priced rather than reproduced.

    With ``fma`` in the rank-1 update the transcription matched OpenBLAS on 120/120 of this
    family's matrices; without it, 82/120. The Rust side does not fuse (see the module header), so
    a good fraction of these come out exact anyway and the rest differ in the last bits. Asserting
    a *count* would pin the port to a kernel; asserting nothing would let a real regression hide.
    The bar below is the one that matters — nothing may be worse than Group A — and the count is
    printed so a change in it is visible in the log.
    """
    exact = total = 0
    worst = 0.0
    for n in (5, 12, 31, 64, 129):
        for kappa in (0.0, 0.05, 0.9, 4.0):
            for sigma in (0.0, 0.7):
                ab = model_band(n, kappa, sigma)
                rs, sp = rs_cholesky(ab), sp_cholesky(ab, lower=False)
                total += 1
                exact += int(np.array_equal(rs, sp))
                band = sp != 0.0
                worst = max(worst, np.max(np.abs(rs[band] - sp[band]) / np.abs(sp[band])))
    print(f"\nbanded factor exactly equal to LAPACK: {exact}/{total}, worst rel {worst:.2e}")
    assert worst <= GROUP_A_TOL, f"worst relative factor difference {worst:.2e}"


# =====================================================================================
# The surface: shapes, refusals, exception types
# =====================================================================================


def test_a_non_spd_band_is_refused_with_lapacks_own_message():
    n = 6
    ab = model_band(n, 0.0)
    ab[2, 3] = -1.0
    with pytest.raises(LinAlgError) as sp_err:
        sp_cholesky(ab, lower=False)
    with pytest.raises(physsynth_rs.NotPositiveDefinite) as rs_err:
        rs_cholesky(ab)
    assert str(rs_err.value) == str(sp_err.value)


def test_the_shim_keeps_the_exception_type_scipy_raises():
    # `LinAlgError` is not a `ValueError` subclass, so which type comes out is part of the
    # contract even though nothing in this repo catches it today.
    from physsynth.core import banded

    ab = model_band(5, 0.0)
    ab[2, 0] = -3.0
    with pytest.raises(LinAlgError):
        banded.cholesky_banded(ab, lower=False)


@pytest.mark.parametrize("bad", [np.zeros((3, 4, 2)), np.zeros((0, 5)), np.zeros((3, 0))])
def test_a_shape_that_is_not_a_band_is_refused(bad):
    with pytest.raises(ValueError):
        rs_cholesky(bad)


def test_a_right_hand_side_of_the_wrong_shape_is_refused():
    cb = rs_cholesky(model_band(8, 0.0))
    with pytest.raises(ValueError, match="not compatible"):
        rs_cho_solve((cb, False), np.ones(7))
    with pytest.raises(ValueError, match="1-D"):
        rs_cho_solve((cb, False), np.ones((8, 2)))


def test_lower_storage_is_refused_rather_than_silently_transposed():
    from physsynth.core import banded

    if banded.cholesky_banded is banded.cholesky_banded_py:
        pytest.skip("the shim's refusal only exists on the Rust path")
    with pytest.raises(NotImplementedError):
        banded.cholesky_banded(model_band(5, 0.0), lower=True)


# =====================================================================================
# The models: the anchors this batch exists to protect
# =====================================================================================
#
# TWO TESTS WERE RETIRED HERE on 2026-09-03 with unit 1's deletion (plan section 42), and both were
# two-sided comparisons that no longer have two sides:
#
#   * `test_the_two_solvers_track_each_other_over_a_run` measured how far a string on LAPACK and
#     the same string on Rust separate. Its measurement is the part worth keeping, because the
#     SHAPE was the finding -- roughly square-root growth, not saturation. On N = 128, kappa = 2.7,
#     sigma = 3, worst state difference as a fraction of the run's amplitude:
#
#         100 steps 1.1e-13 | 500 2.9e-13 | 1000 4.1e-13 | 2000 9.7e-13 | 5000 2.0e-12 |
#         20000 3.2e-12
#
#     So the plan's Group A target of ~1e-13 was a HUNDRED-step claim for a fed-back solve, not a
#     two-thousand-step one: a fed-back reduction held 1e-13 out to 2,000 steps and a fed-back
#     solve is an order of magnitude worse at the same length. Also recorded because it is easy to
#     get wrong: the difference must be normalised by the run's AMPLITUDE, never pointwise, since a
#     damped string decays by orders of magnitude.
#
#   * `test_neither_solver_moves_the_energy_bar` asserted that both solvers keep the lossless drift
#     under CLAUDE.md's 1e-10. The Rust half of that claim is asserted by every energy bar in the
#     suite -- `test_stiff_string.py`, `test_damped_string.py`, `test_tension_string.py`,
#     `test_geometric_energy.py` -- all of which now run on the Rust solver by construction. The
#     LAPACK half has no subject.
#
# What follows is the third test, which survives. It needed exactly one change: the `with
# rust_solver():` block came off, because building these four models IS building them on the Rust
# solver now.


def test_the_family_still_reduces_to_itself_exactly():
    """The batch's whole reason for existing, asserted directly.

    ``tests/test_damped_string.py``, ``tests/test_tension_string.py`` and
    ``tests/test_geometric_energy.py`` each carry one of these anchors and each would catch a
    break — but each in isolation. Here all three run in one place, so a change that reaches three
    of the four models and misses the fourth fails with a message that says which pair diverged.

    Renamed from ``..._on_the_rust_solver`` when unit 1 landed: there is no other solver to
    contrast with, so the qualifier was claiming a distinction that no longer exists.
    """
    N, fs = 64, 12800.0
    kw = dict(L=0.65, T=200.0, rho=0.005, fs=fs, N=N, kappa=2.7, sigma0=1.0, sigma1=0.0)
    stiff_kw = {k: v for k, v in kw.items() if k not in ("sigma0", "sigma1")}
    ds = string_damped.DampedStiffString(**kw)
    ss = string_stiff.StiffString(sigma=1.0, **stiff_kw)
    tn = string_nonlinear.TensionModulatedString(EA=0.0, **kw)
    gm = string_geometric.GeometricString(EA=kw["T"], **kw)

    u0 = triangular_pluck(ds.x, ds.L, 0.137 * ds.L, amplitude=1e-3)
    for m in (ds, ss, tn):
        m.set_state(u0.copy())
    gm.set_state(u0.copy())
    for _ in range(300):
        for m in (ds, ss, tn, gm):
            m.step()

    assert np.array_equal(ds.u, ss.u), "sigma1 = 0 no longer reduces to the stiff string"
    assert np.array_equal(ds.state, tn.state), "EA = 0 no longer reduces to the damped string"
    assert np.array_equal(ds.u, gm.u), "EA = T no longer reduces to the damped string"
