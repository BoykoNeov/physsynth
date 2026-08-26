"""Rust vs Python for the ideal string — the migration's comparison, not its acceptance.

``docs/dev/rust-migration-plan.md`` is explicit that the acceptance bar is the physics harness
(energy drift, modal frequencies, convergence order) and that cross-implementation agreement is a
**diagnostic**. This file is that diagnostic, and it exists because the physics bars alone are too
loose to catch a transcription slip: a wrong trapezoidal end weight, or a ``set_state`` that starts
half a step off, changes the numbers by ~1e-8 and still passes every bar in ``test_energy.py``.

Two different bars are asserted below, and the gap between them is the interesting part:

* **The state arrays must agree bit-for-bit.** The step is pure elementwise arithmetic — no
  reduction, no library call — so IEEE-754 fixes the answer exactly once the *operation order*
  matches. `physsynth-core`'s `step_into` is written in NumPy's evaluation order deliberately, and
  this is what checks that it stayed that way. Plan §2.1 says a port will not match bit-for-bit;
  that is right about the models that *solve* something and about any reduction, and wrong here.
* **The energy may differ at ~1e-15 relative.** ``energy()`` reduces, ``np.dot`` goes through BLAS,
  and BLAS accumulates in an order no portable loop reproduces. Held to the plan's Group A target
  of ~1e-13, which it clears by two orders of magnitude.

Also here: the **buffer-lifetime** tests. Python's ``step()`` rebinds ``self.u`` rather than
overwriting it, so a reference taken before a step is a valid snapshot afterwards, while a write
*through* ``.u`` reaches the string (which is how ``connection.py`` applies its bridge force). No
test in the ideal string's own suite can see either property — they all go through ``state``, which
copies — so a binding that got them wrong would be green everywhere and broken in the viewer.
"""

import numpy as np
import pytest

from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_ideal import IdealStringPy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)
IdealStringRs = physsynth_rs.IdealString

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = 200 m/s

# Every boundary spelling the two implementations support, including the mixed tuple form. The
# mixed form is worth naming: no test in the ideal string's own suite builds one (``make_string``
# is typed for a single ``Boundary``), yet ``web/serialize.py`` uses ("fixed", "free") at six sites
# and every string-to-body connection depends on it. Untested and load-bearing is the combination
# that a parity file is for.
BOUNDARIES = ["fixed", "free", ("fixed", "free"), ("free", "fixed")]

# The plan's Group A agreement target. The state clears it by being exact; the energy clears it by
# two orders of magnitude. It is the bar because it is the one the plan wrote down, not because
# anything here is close to it.
GROUP_A_TOL = 1e-13


def _pair(**kwargs):
    """Build the same string twice, once per implementation."""
    return IdealStringPy(**kwargs), IdealStringRs(**kwargs)


def _params(*, N=100, lam=1.0, boundary="fixed", sigma=0.0):
    c = np.sqrt(T_DEFAULT / RHO_DEFAULT)
    return {
        "L": L_DEFAULT,
        "T": T_DEFAULT,
        "rho": RHO_DEFAULT,
        "fs": c * N / (L_DEFAULT * lam),
        "N": N,
        "boundary": boundary,
        "sigma": sigma,
    }


# -- derived quantities ------------------------------------------------------------------------


@pytest.mark.parametrize("N", [3, 17, 100, 257])
@pytest.mark.parametrize("lam", [1.0, 0.9, 0.5, 0.137])
def test_derived_parameters_are_identical(N, lam):
    py, rs = _pair(**_params(N=N, lam=lam))
    for name in ("L", "T", "rho", "fs", "N", "sigma", "c", "h", "k", "lam"):
        assert getattr(py, name) == getattr(rs, name), name


@pytest.mark.parametrize("N", [3, 17, 100, 257])
def test_the_grid_is_identical(N):
    # `x` is what the analysis layer measures pickup positions against, and `np.linspace` does not
    # simply compute `i * h` — it overwrites the last sample with the endpoint. Reproducing that
    # is a one-line detail that nothing else in the suite would ever notice.
    py, rs = _pair(**_params(N=N))
    assert np.array_equal(py.x, rs.x)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_boundary_spec_round_trips(boundary):
    py, rs = _pair(**_params(boundary=boundary))
    assert py.boundary == rs.boundary
    assert (py._bc_left, py._bc_right) == (rs._bc_left, rs._bc_right)


# -- the trajectory ----------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("sigma", [0.0, 3.0])
@pytest.mark.parametrize("lam", [1.0, 0.731])
def test_the_state_is_bit_identical_over_a_run(boundary, sigma, lam):
    py, rs = _pair(**_params(lam=lam, boundary=boundary, sigma=sigma))
    u0 = triangular_pluck(py.x, py.L, 0.137 * py.L, amplitude=1e-3)
    py.set_state(u0)
    rs.set_state(u0)

    assert np.array_equal(py.u, rs.u), "set_state disagreed on u^0"
    assert np.array_equal(py.u_prev, rs.u_prev), "set_state disagreed on the Taylor start u^-1"

    for step in range(4000):
        py.step()
        rs.step()
        if step % 500 == 0 or step == 3999:
            assert np.array_equal(py.u, rs.u), f"diverged at step {step}"
    assert py.n == rs.n


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("sigma", [0.0, 3.0])
def test_the_energy_agrees_to_the_group_a_target(boundary, sigma):
    py, rs = _pair(**_params(boundary=boundary, sigma=sigma))
    u0 = triangular_pluck(py.x, py.L, 0.3 * py.L, amplitude=1e-3)
    py.set_state(u0)
    rs.set_state(u0)

    worst = 0.0
    for _ in range(2000):
        py.step()
        rs.step()
        e_py, e_rs = py.energy(), rs.energy()
        worst = max(worst, abs(e_py - e_rs) / abs(e_py))
    assert worst < GROUP_A_TOL, f"energy disagreed by {worst:.2e} (Group A target {GROUP_A_TOL:g})"


def test_a_nonzero_initial_velocity_agrees():
    # `set_state`'s second argument goes through `np.broadcast_to`, so it accepts a scalar and a
    # full array. Both paths reach a different branch of the binding, and neither is exercised by
    # the ideal string's own tests, which always pluck from rest.
    for v0 in (0.0, 0.25, np.linspace(-1.0, 1.0, 101)):
        py, rs = _pair(**_params())
        u0 = triangular_pluck(py.x, py.L, 0.4 * py.L, amplitude=1e-3)
        py.set_state(u0, v0)
        rs.set_state(u0, v0)
        assert np.array_equal(py.u_prev, rs.u_prev)
        for _ in range(200):
            py.step()
            rs.step()
        assert np.array_equal(py.u, rs.u)


def test_set_state_accepts_a_plain_list():
    py, rs = _pair(**_params(N=4))
    shape = [0.0, 1.0, 2.0, 1.0, 0.0]
    py.set_state(shape)
    rs.set_state(shape)
    assert np.array_equal(py.u, rs.u)


# -- the surface `connection.py` and the viewer reach for --------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_second_diff_is_identical(boundary):
    # A *private* method, exposed on purpose: `physsynth/core/connection.py` calls it, and
    # connection is a Phase 5 model — so a Python module is a client of the binding's private
    # names for the whole migration.
    py, rs = _pair(**_params(N=32, boundary=boundary))
    rng = np.random.default_rng(20260826)
    u = rng.standard_normal(33)
    assert np.array_equal(py._second_diff(u), rs._second_diff(u))


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_apply_boundary_is_identical(boundary):
    py, rs = _pair(**_params(N=8, boundary=boundary))
    a = np.arange(9, dtype=float)
    b = a.copy()
    py._apply_boundary(a)
    rs._apply_boundary(b)
    assert np.array_equal(a, b)


def test_displacement_at_agrees_including_negative_indices():
    py, rs = _pair(**_params(N=16))
    u0 = triangular_pluck(py.x, py.L, 0.3 * py.L, amplitude=1e-3)
    py.set_state(u0)
    rs.set_state(u0)
    for _ in range(50):
        py.step()
        rs.step()
    for index in (0, 1, 8, 16, -1, -3):
        assert py.displacement_at(index) == rs.displacement_at(index), index


def test_state_is_a_copy_not_a_view():
    for cls in (IdealStringPy, IdealStringRs):
        s = cls(**_params(N=8))
        s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
        snapshot = s.state
        snapshot[3] = 12345.0
        assert s.u[3] != 12345.0, f"{cls.__name__}: `state` handed out a live view"


# -- buffer lifetime: the property no physics test can see --------------------------------------


@pytest.mark.parametrize("cls", [IdealStringPy, IdealStringRs], ids=["python", "rust"])
def test_a_reference_to_u_held_across_a_step_is_a_snapshot(cls):
    # Python rebinds `self.u` in `step()`; it does not write into it. So a reference taken before
    # the step keeps showing that step's values. The Rust binding reproduces this by owning NumPy
    # arrays rather than Rust `Vec`s — a zero-copy view over a reallocated `Vec` would *look*
    # right here (measured: it still reads the old contents) while being a use-after-free.
    s = cls(**_params(N=16))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    held = s.u
    before = np.array(held, copy=True)

    s.step()

    assert np.array_equal(held, before), "the held reference changed under a step"
    assert not np.array_equal(s.u, before), "the string did not actually advance"
    assert s.u_prev is held, "u_prev after a step must be the very object u was"


@pytest.mark.parametrize("cls", [IdealStringPy, IdealStringRs], ids=["python", "rust"])
def test_an_in_place_write_through_u_reaches_the_string(cls):
    # This is exactly what a bridge connection does — `self.string.u[-1] -= beta_s * F` — and it
    # is the reason `.u` cannot be handed out as a copy.
    s = cls(**_params(N=16, boundary=("fixed", "free")))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    before = s.energy()

    s.u[-1] -= 1e-3

    assert s.energy() != before, "a write through `.u` did not reach the string"
    assert s.state[-1] == s.u[-1]
    s.step()  # and the modified state is what gets stepped, without complaint


@pytest.mark.parametrize("cls", [IdealStringPy, IdealStringRs], ids=["python", "rust"])
def test_the_state_arrays_can_be_replaced_wholesale(cls):
    # `tests/test_collision_modal.py` does this to a string it owns (`bar.string.u = uf.copy()`),
    # so assignment has to work, not just mutation.
    s = cls(**_params(N=8))
    fresh = np.linspace(0.0, 1.0, 9)
    s.u = fresh.copy()
    s.u_prev = fresh.copy()
    s.n = 0
    assert np.array_equal(s.u, fresh)
    s.step()
    assert s.n == 1


# -- construction-time rejections ---------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"rho": -1.0},
        {"T": 0.0},
        {"L": -2.0},
        {"sigma": -0.1},
        {"N": 1},
        {"boundary": "clamped"},
        {"boundary": ("fixed", "welded")},
    ],
)
def test_invalid_parameters_are_rejected_with_the_same_message(override):
    kwargs = _params()
    kwargs.update(override)
    with pytest.raises(ValueError) as py_err:
        IdealStringPy(**kwargs)
    with pytest.raises(ValueError) as rs_err:
        IdealStringRs(**kwargs)
    assert str(py_err.value) == str(rs_err.value)


def test_the_cfl_rejection_carries_the_same_message():
    c = np.sqrt(T_DEFAULT / RHO_DEFAULT)
    kwargs = _params()
    kwargs["fs"] = c * kwargs["N"] / (L_DEFAULT * 1.05)  # forces lambda = 1.05
    with pytest.raises(ValueError, match="CFL") as py_err:
        IdealStringPy(**kwargs)
    with pytest.raises(ValueError, match="CFL") as rs_err:
        IdealStringRs(**kwargs)
    assert str(py_err.value) == str(rs_err.value)


def test_courant_exactly_one_is_accepted_by_both():
    py, rs = _pair(**_params(lam=1.0))
    assert py.lam == rs.lam == pytest.approx(1.0, abs=1e-12)
