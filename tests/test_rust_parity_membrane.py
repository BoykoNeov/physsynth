"""Rust vs Python for the membrane — Phase 2's model comparison.

The acceptance contract is still the physics harness (``docs/dev/rust-migration-plan.md`` §2.1);
this file is the diagnostic. It is the sibling of ``test_rust_parity.py`` one model on, and almost
everything it asserts is inherited from that file. Three things are genuinely new, and they are the
reason this is a separate page rather than a few more cases there.

**1. The trajectory is bit-identical, and the update contains a matrix.** §2.1's rule is "does the
step contain a reduction", and a sparse mat-vec is a reduction — each output entry is a sum of up
to five products. It still comes out exact, for the same reason the assembled matrices did in
Phase 1: the summation order is *knowable*. SciPy's CSR mat-vec walks a row in stored order and so
does the Rust one, and both sides hold the same canonical CSR. This is worth asserting rather than
assuming, because it is the first time a matrix has appeared inside a timestep in this port.

**2. The buffer-lifetime properties, again, on a model that has a second consumer.** Phase 0
established that `.u` must be a Python-owned array — a reference taken before a step stays a valid
snapshot, and a write *through* `.u` reaches the model. `airbox._MembraneSurface.commit` assigns
straight into `.u` and `.u_prev` and bumps `.n`, so for the membrane this is not a hypothetical
about future callers; it is how a shipped model drives this one. Asserted against **both**
implementations, so the claim is about behaviour rather than about Rust.

**3. `L` is an attribute, not a function return, and it must be built once.**
`_MembraneSurface.rhs` evaluates ``m.L @ m.u`` every timestep. A getter that rebuilt a
``csr_matrix`` per access would assemble a sparse matrix inside the inner loop of the heaviest
tests in the suite — passing every physics bar while making the flagged run inexplicably slower.
That is invisible to a value comparison, so it is asserted by *identity*: the same call twice
returns the same object.

Two smaller things also live here because nothing else would catch them: the `Ly` snap uses
Python's round-half-to-**even** (Rust's own `f64::round` would build a different, entirely
healthy-looking membrane), and the rejection messages match verbatim.
"""

import numpy as np
import pytest

from physsynth.core.membrane import MembranePy

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# The plan's Group A agreement target. Only `energy()` is measured against it; the state is exact.
GROUP_A_TOL = 1e-13

T = 100.0
RHO = 0.26  # -> c ~= 19.6 m/s

CASES = [
    dict(domain="circle", radius=0.15, N=24),
    dict(domain="circle", radius=0.15, N=31),
    dict(domain="circle", radius=0.08, N=40),
    dict(domain="rectangle", Lx=0.4, Ly=0.3, N=20),
    dict(domain="rectangle", Lx=0.4, Ly=0.31, N=25),
    dict(domain="rectangle", Lx=0.25, Ly=0.25, N=16),
]


def _fs_for(case, lam):
    """A sample rate putting the Courant number at ``lam`` for this case's spacing."""
    c = np.sqrt(T / RHO)
    h = 2.0 * case["radius"] / case["N"] if case["domain"] == "circle" else case["Lx"] / case["N"]
    return c / (lam * h)


def _pair(case, *, lam=0.6, sigma=0.0):
    kw = dict(case, T=T, rho=RHO, fs=_fs_for(case, lam), sigma=sigma)
    return MembranePy(**kw), physsynth_rs.Membrane(**kw)


def _bump(m, center=(0.01, -0.005), width=0.05, amplitude=1e-3):
    """A smooth radial hump on the full grid — built here, not through `exciter`, so this file
    compares the *model* and not the excitation (which `test_rust_parity_ops2d.py` owns)."""
    d = np.sqrt((m.X - center[0]) ** 2 + (m.Y - center[1]) ** 2)
    field = np.zeros_like(m.X)
    inside = d < width
    field[inside] = amplitude * 0.5 * (1.0 + np.cos(np.pi * d[inside] / width))
    return field


# -- construction ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['domain']}-{c['N']}")
def test_every_derived_parameter_agrees_exactly(case):
    py, rs = _pair(case)
    for name in ("domain", "T", "rho", "fs", "N", "sigma", "c", "k", "h", "lam", "n_live"):
        assert getattr(py, name) == getattr(rs, name), name
    assert py.Lx == rs.Lx
    assert py.Ly == rs.Ly
    assert py.radius == rs.radius


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['domain']}-{c['N']}")
def test_the_grid_the_mask_and_the_index_map_are_identical(case):
    py, rs = _pair(case)
    assert np.array_equal(py.X, rs.X)
    assert np.array_equal(py.Y, rs.Y)
    assert np.array_equal(py.mask, rs.mask)
    assert rs.mask.dtype == np.bool_
    assert np.array_equal(py.index_map, rs.index_map)
    assert rs.index_map.dtype == np.int64


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['domain']}-{c['N']}")
def test_the_laplacian_is_bit_identical(case):
    py, rs = _pair(case)
    a = py.L.tocsr()
    a.sort_indices()
    b = rs.L.tocsr()
    b.sort_indices()
    assert a.shape == b.shape
    assert a.nnz == b.nnz
    assert np.array_equal(a.data, b.data)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.indptr, b.indptr)


def test_the_laplacian_is_built_once_not_per_access():
    # `airbox._MembraneSurface.rhs` does `m.L @ m.u` EVERY timestep. A getter that rebuilt the
    # matrix would be correct and would quietly put a sparse assembly in the inner loop of the
    # heaviest tests in the suite. Only identity catches that; every value check would pass.
    _, rs = _pair(CASES[0])
    assert rs.L is rs.L
    # And the same holds for the other three immutable grid objects.
    assert rs.X is rs.X
    assert rs.Y is rs.Y
    assert rs.mask is rs.mask
    assert rs.index_map is rs.index_map


# -- the trajectory --------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['domain']}-{c['N']}")
@pytest.mark.parametrize("sigma", [0.0, 4.0])
def test_the_state_stays_bit_identical_over_a_long_run(case, sigma):
    py, rs = _pair(case, sigma=sigma)
    u0 = _bump(py)
    py.set_state(u0)
    rs.set_state(u0)

    assert np.array_equal(py.u, rs.u), "set_state disagreed before a single step"
    assert np.array_equal(py.u_prev, rs.u_prev), "the second-order start disagreed"

    for i in range(2000):
        py.step()
        rs.step()
        if i % 250 == 0 or i == 1999:
            assert np.array_equal(py.u, rs.u), f"state diverged at step {i}"
            assert np.array_equal(py.u_prev, rs.u_prev), f"history diverged at step {i}"
    assert py.n == rs.n == 2000


@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: f"{c['domain']}-{c['N']}")
def test_the_energy_agrees_to_the_group_a_target(case):
    py, rs = _pair(case)
    u0 = _bump(py)
    py.set_state(u0)
    rs.set_state(u0)

    worst = 0.0
    for _ in range(500):
        py.step()
        rs.step()
        want = py.energy()
        got = rs.energy()
        worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
    assert worst <= GROUP_A_TOL, f"worst relative energy disagreement {worst:e}"


@pytest.mark.parametrize("case", CASES[:3], ids=lambda c: f"{c['domain']}-{c['N']}")
def test_a_velocity_start_is_bit_identical_too(case):
    # A scalar `v0` and a full-field `v0` take different branches in `set_state`; both have to land
    # on the same `u^{-1}`, which is where the second-order start could silently differ.
    py, rs = _pair(case)
    u0 = _bump(py)
    v_field = _bump(py, center=(-0.02, 0.01), width=0.04, amplitude=0.5)

    for v0 in (0.0, 0.25, v_field):
        py.set_state(u0, v0)
        rs.set_state(u0, v0)
        assert np.array_equal(py.u_prev, rs.u_prev)
        for _ in range(50):
            py.step()
            rs.step()
        assert np.array_equal(py.u, rs.u)


@pytest.mark.parametrize("case", CASES[:3], ids=lambda c: f"{c['domain']}-{c['N']}")
def test_a_live_node_vector_and_a_full_field_are_the_same_initial_condition(case):
    py, rs = _pair(case)
    full = _bump(py)
    live = full[py.mask]

    py.set_state(full)
    rs.set_state(live)
    assert np.array_equal(py.u, rs.u)
    assert np.array_equal(py.u_prev, rs.u_prev)


# -- the read-outs ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['domain']}-{c['N']}")
def test_state_embeds_identically_with_a_dead_rim(case):
    py, rs = _pair(case)
    u0 = _bump(py)
    py.set_state(u0)
    rs.set_state(u0)
    for _ in range(17):
        py.step()
        rs.step()
    assert np.array_equal(py.state, rs.state)
    assert rs.state.shape == py.mask.shape
    assert np.all(rs.state[~py.mask] == 0.0)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['domain']}-{c['N']}")
def test_to_live_and_the_pickup_agree(case):
    py, rs = _pair(case)
    field = _bump(py)
    assert np.array_equal(py.to_live(field), rs.to_live(field))
    for probe in ((0.0, 0.0), (0.03, -0.02), (-0.1, 0.05), (1.0, 1.0)):
        assert py.pickup_index_at(*probe) == rs.pickup_index_at(*probe)


def test_displacement_at_matches_including_a_negative_index():
    py, rs = _pair(CASES[0])
    u0 = _bump(py)
    py.set_state(u0)
    rs.set_state(u0)
    for _ in range(9):
        py.step()
        rs.step()
    for idx in (0, 3, py.n_live - 1, -1, -5):
        assert py.displacement_at(idx) == rs.displacement_at(idx)


# -- the buffer contract (plan §9.3), asserted against both implementations ------------------------


@pytest.mark.parametrize("make", [MembranePy, lambda **kw: physsynth_rs.Membrane(**kw)])
def test_a_reference_held_across_a_step_is_a_snapshot(make):
    case = CASES[0]
    kw = dict(case, T=T, rho=RHO, fs=_fs_for(case, 0.6))
    m = make(**kw)
    m.set_state(_bump(m))

    held = m.u
    before = held.copy()
    m.step()
    assert np.array_equal(held, before), (
        "the array handed out before a step changed underneath the caller -- `step` must rebind, "
        "not overwrite"
    )
    assert m.u is not held


@pytest.mark.parametrize("make", [MembranePy, lambda **kw: physsynth_rs.Membrane(**kw)])
def test_a_write_through_u_reaches_the_model(make):
    # This is how a coupled model applies a force: `airbox._MembraneSurface` and the mallet both
    # go through the live array. A binding that handed out a copy would lose the write silently.
    case = CASES[0]
    kw = dict(case, T=T, rho=RHO, fs=_fs_for(case, 0.6))
    m = make(**kw)
    m.set_state(_bump(m))

    m.u[0] += 1.0
    assert m.u[0] == pytest.approx(m.state[m.index_map == 0][0])
    e_after = m.energy()
    assert np.isfinite(e_after)


@pytest.mark.parametrize("make", [MembranePy, lambda **kw: physsynth_rs.Membrane(**kw)])
def test_the_previous_array_is_the_object_u_used_to_be(make):
    case = CASES[0]
    kw = dict(case, T=T, rho=RHO, fs=_fs_for(case, 0.6))
    m = make(**kw)
    m.set_state(_bump(m))

    was = m.u
    m.step()
    assert m.u_prev is was, "the history roll must hand the same object over, not a copy"


@pytest.mark.parametrize("make", [MembranePy, lambda **kw: physsynth_rs.Membrane(**kw)])
def test_assigning_the_state_directly_is_supported(make):
    # `airbox._MembraneSurface.commit` does exactly this: `m.u_prev = m.u; m.u = u_next; m.n += 1`.
    case = CASES[0]
    kw = dict(case, T=T, rho=RHO, fs=_fs_for(case, 0.6))
    m = make(**kw)
    m.set_state(_bump(m))

    u_next = np.ascontiguousarray(m.u * 0.5)
    m.u_prev = m.u
    m.u = u_next
    m.n += 1
    assert m.u is u_next
    assert m.n == 1
    m.step()
    assert np.isfinite(m.energy())


# -- rejections ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(domain="rectangle", T=0.0, rho=RHO, fs=1000.0, N=10, Lx=1.0, Ly=1.0),
        dict(domain="rectangle", T=T, rho=-1.0, fs=1000.0, N=10, Lx=1.0, Ly=1.0),
        dict(domain="rectangle", T=T, rho=RHO, fs=1000.0, N=1, Lx=1.0, Ly=1.0),
        dict(domain="rectangle", T=T, rho=RHO, fs=1000.0, N=10, Lx=1.0, Ly=1.0, sigma=-1.0),
        dict(domain="rectangle", T=T, rho=RHO, fs=1000.0, N=10, Lx=1.0),
        dict(domain="rectangle", T=T, rho=RHO, fs=1000.0, N=10, Lx=1.0, Ly=0.0),
        dict(domain="circle", T=T, rho=RHO, fs=1000.0, N=10),
        dict(domain="circle", T=T, rho=RHO, fs=1000.0, N=10, radius=-0.1),
        dict(domain="ellipse", T=T, rho=RHO, fs=1000.0, N=10, Lx=1.0, Ly=1.0),
        dict(domain="circle", T=T, rho=RHO, fs=500.0, N=10, radius=0.15),  # lambda > 1/sqrt(2)
    ],
)
def test_both_sides_reject_the_same_thing_with_the_same_words(kwargs):
    with pytest.raises(ValueError) as py_err:
        MembranePy(**kwargs)
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.Membrane(**kwargs)
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("bad_shape", [(3,), (2, 2), (5, 5, 5)])
def test_a_wrongly_shaped_initial_condition_is_rejected_the_same_way(bad_shape):
    py, rs = _pair(CASES[0])
    u0 = np.zeros(bad_shape)
    with pytest.raises(ValueError) as py_err:
        py.set_state(u0)
    with pytest.raises(ValueError) as rs_err:
        rs.set_state(u0)
    assert str(py_err.value) == str(rs_err.value)


def test_to_live_rejects_a_wrongly_shaped_field_the_same_way():
    py, rs = _pair(CASES[0])
    field = np.zeros((3, 4))
    with pytest.raises(ValueError) as py_err:
        py.to_live(field)
    with pytest.raises(ValueError) as rs_err:
        rs.to_live(field)
    assert str(py_err.value) == str(rs_err.value)


# -- the snap that no physics detector can see ----------------------------------------------------


@pytest.mark.parametrize(
    ("N", "Lx", "Ly", "cells"),
    [
        (2, 1.0, 1.25, 2),  # Ly/h == 2.5 -> ties-to-even rounds DOWN
        (2, 1.0, 1.75, 4),  # Ly/h == 3.5 -> ties-to-even rounds UP
        (4, 1.0, 0.625, 2),  # Ly/h == 2.5 again, on a finer grid
        (4, 1.0, 0.875, 4),  # Ly/h == 3.5
        (10, 1.0, 0.37, 4),  # an ordinary, non-tie case
    ],
)
def test_the_height_snap_uses_pythons_round_half_to_even(N, Lx, Ly, cells):
    # Rust's own `f64::round` is half-AWAY-from-zero, so it would give 3 and 4 where Python gives
    # 2 and 4. The resulting membrane would have a different height, a different mask and a
    # different spectrum -- and would conserve energy perfectly, pass passivity, and look right in
    # the viewer. Nothing else in this repo can see the difference, so it is pinned here.
    kw = dict(domain="rectangle", T=T, rho=RHO, fs=400.0, N=N, Lx=Lx, Ly=Ly)
    py = MembranePy(**kw)
    rs = physsynth_rs.Membrane(**kw)
    assert py.Ly == rs.Ly
    assert rs.Ly == pytest.approx(cells * (Lx / N))
    assert py.mask.shape == rs.mask.shape == (cells + 1, N + 1)
