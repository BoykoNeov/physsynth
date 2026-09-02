"""Rust parity for ``physsynth.core.airbox``'s **port tier** — plan §31.

``RoomPort``, ``SurfacePort``, ``InteriorSurfacePort`` and the module helper
``_free_pressure_nodes`` they share. The six ``RoomLoaded*`` / ``RoomSuspended*`` wrappers above
them stay Python, so this file has the same two jobs the room's had one batch earlier: the numbers
are right, and the seam holds.

**Everything here is exact, and that is a change of position rather than a change of luck.** §30.2
measured ``np.sum``'s blocking, found the eight-element cutoff, and declined to transcribe the
blocking above it — on the grounds that doing so would be a claim about a NumPy internal and, after
§22.1, about a CPU. The first half of that is kept and sharpened; the second half is retired by
measurement. NumPy's pairwise sum is one fixed algorithm, not a dispatched kernel, and transcribing
it reproduces ``np.sum`` bit for bit at every length tried — which matters here and did not matter
there, because this tier's reductions are on the **update path**: ``w = W / W.sum()`` is the share
of the volume velocity each node receives, ``R_room`` is what the coupled solve divides by, and
``free_pressure`` is the pressure the body is pushed by. §14.2's question — does the reduction reach
the next timestep? — is answered *yes* for all three, where the room's two energy books answered no.

The riskiest sentence in the batch is therefore "NumPy's blocking is the same on every machine", and
:func:`test_numpy_pairwise_blocking_is_an_algorithm_not_a_kernel` is the one place it is asserted.
If it is ever false on a runner, that test says so by name instead of every exact claim below going
red at once with no diagnosis (§22.1's eighteen-failure morning, pre-empted).

The other measured claim worth reading before the assertions is the **association**. SciPy computes
``T.T @ diags(R) @ T`` left-associated, so each term is ``(T_ki R_k) T_kj`` and not
``T_ki (R_k T_kj)`` — different doubles in about 30% of entries. The fixture that cannot see it is
``spreading="nearest"``, because there every stored entry in a row of ``T`` is the same uniform node
area and ``(x d) x`` is ``x (d x)`` identically. That fixture is one of the six goldens in
``tests/test_airbox_dipole.py``, so §16.4's blind-fixture finding has a fourth instance and this
time the blindness is provable rather than measured.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from physsynth.core.airbox import (
    C0_AIR,
    RHO0_AIR,
    AirBoxPy,
    InteriorSurfacePortPy,
    RoomPortPy,
    SurfacePortPy,
    free_pressure_nodes_py,
    impedance_from_zeta,
)

# NOT a bare `import physsynth_rs`: the default gate does not build the extension, so a module-scope
# import is a collection error there rather than a skip.
physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

H = 0.03
N_DEFAULT = (9, 7, 6)
CFL = 0.9

WALLS = {
    "rigid": "rigid",
    "one lossy face": {"x0": 2.0 * RHO0_AIR * C0_AIR},
    "two lossy faces": {"x0": 2.0 * RHO0_AIR * C0_AIR, "y1": 0.5 * RHO0_AIR * C0_AIR},
    "all lossy": 1.5 * RHO0_AIR * C0_AIR,
}


def _kwargs(*, N=N_DEFAULT, h=H, walls="rigid", **extra):
    kw = dict(
        L=tuple(n * h for n in N),
        h=h,
        fs=C0_AIR * np.sqrt(3.0) / (CFL * h),
        walls=walls,
        c0=C0_AIR,
    )
    kw.update(extra)
    return kw


def _rooms(**kw):
    """One Python room and one Rust room, built from identical arguments."""
    return AirBoxPy(**kw), physsynth_rs.AirBox(**kw)


def _ports(*, room_kw=None, **port_kw):
    """One Python port on a Python room and one Rust port on a Rust room."""
    py_room, rs_room = _rooms(**_kwargs(**(room_kw or {})))
    return (
        RoomPortPy(room=py_room, **port_kw),
        physsynth_rs.RoomPort(room=rs_room, **port_kw),
    )


def _grid(nx, ny, lx, ly):
    """A uniform rectangular surface: node coordinates and equal per-node areas."""
    xs = np.linspace(0.0, lx, nx)
    ys = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])
    area = (lx / (nx - 1)) * (ly / (ny - 1))
    return coords, np.full(coords.shape[0], area)


def _surfaces(*, room_kw=None, cls="wall", **port_kw):
    """One Python surface port and one Rust surface port, on their own rooms."""
    py_room, rs_room = _rooms(**_kwargs(**(room_kw or {})))
    if cls == "wall":
        return (
            SurfacePortPy(room=py_room, **port_kw),
            physsynth_rs.SurfacePort(room=rs_room, **port_kw),
        )
    return (
        InteriorSurfacePortPy(room=py_room, **port_kw),
        physsynth_rs.InteriorSurfacePort(room=rs_room, **port_kw),
    )


def _csr_equal(a, b):
    """Bit-identical as *stored*: same structure, same order, same data."""
    a, b = a.tocsr(), b.tocsr()
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


# The termination the closed loop below solves against, as a multiple of the port's OWN
# resistance. The port relation is `pbar = pbar_free + R q`, so `q = amp - pbar_free / (ALPHA R)`
# makes the loop a strict contraction for any ALPHA > 1 — passive, stable, and scale-free, which a
# fixed impedance is not: a distributed port injects over hundreds of nodes and a constant divisor
# would have the feedback gain grow with the node count (the first draft did, and overflowed).
Z_ALPHA = 4.0


def _drive(room, port, steps, q=1e-4):
    """Drive a room through its port for `steps` steps and hand back the field.

    The loop is **closed**: the injection is `amp - pbar_free / (ALPHA R)`, so the value
    :meth:`free_pressure` returns feeds the next injection and therefore the next field. That
    matters for what this file is allowed to claim. A drive that ignored the read and injected a
    prescribed function of the step index would produce an identical field even if
    ``free_pressure`` returned garbage — the comparison would then be of the *room*, which §30
    already made, rather than of the port. This is the smallest thing that makes the read
    load-bearing at every fixture, and it is the scalar Thevenin solve a wrapper actually does.
    """
    for n in range(steps):
        free = port.free_pressure()
        amp = q * np.sin(0.07 * n)
        if isinstance(free, tuple):           # an interior patch reads a pair of planes
            lo, hi = free
            # Its load is `2 T^T R T`, so the jump sees twice the per-face resistance.
            port.inject(amp - (np.asarray(hi) - np.asarray(lo)) / (2.0 * Z_ALPHA * port.R))
        elif np.ndim(free):                   # a wall patch reads a vector
            port.inject(amp - np.asarray(free) / (Z_ALPHA * np.asarray(port.R)))
        else:
            port.inject(amp - free / (Z_ALPHA * port.R_room))
        room.step()
    return np.array(room.p, copy=True)


# -- the claim every exact assertion below rests on ------------------------------------------------


def test_numpy_pairwise_blocking_is_an_algorithm_not_a_kernel():
    """The transcription reproduces ``np.sum`` exactly, at every length and both sides of both
    cut-offs — and this is the one test that says so.

    §30.2 established the shape of ``np.sum``'s blocking (a plain loop below eight, eight
    accumulators to 128, a recursive split above) and refused to transcribe it, calling that "a
    claim about a library internal, and after §22.1 a claim about the CPU as well". This batch takes
    the opposite view, and the distinction is worth stating because the two hazards look alike.
    §22.1's is that NumPy computes ``pow``, ``sin`` and ``exp`` with its own CPU-dispatched
    routines: two machines, two instruction selections, two last bits, invisible in the source of
    either language. A summation has no comparable freedom — the *order* is fixed by the blocking,
    and the unroll by eight exists precisely so a vector unit can be used without changing it.

    Measured here rather than argued: the lengths below straddle 8 and 128, include a length whose
    ragged tail is non-empty and one whose recursive split rounds down to a multiple of eight, and
    the negative control asserts that a plain left-to-right loop is *not* the same computation
    above the cutoff (without which this test could pass while asserting nothing — §23.5).
    """
    rng = np.random.default_rng(20260831)
    naive_differed = 0
    for n in (1, 2, 4, 7, 8, 9, 15, 16, 20, 29, 56, 128, 129, 200, 560, 4641):
        for _ in range(60):
            a = rng.random(n)
            assert physsynth_rs._pairwise_sum(a) == float(np.sum(a)), f"n = {n}"
            plain = 0.0
            for x in a:
                plain += float(x)
            if plain != float(np.sum(a)):
                naive_differed += 1
    assert naive_differed > 100, (
        f"only {naive_differed} of 960 vectors distinguished a left-to-right loop from np.sum, so "
        "this test cannot tell the transcription from the naive spelling"
    )


def test_below_eight_terms_the_two_spellings_are_the_same_computation():
    """§30.2's half that survives: under the cutoff, exactness is free and value-independent."""
    rng = np.random.default_rng(11)
    for n in range(1, 8):
        for _ in range(200):
            a = rng.random(n) * rng.choice([1.0, 1e-8, 1e8])
            plain = 0.0
            for x in a:
                plain += float(x)
            assert plain == float(np.sum(a)) == physsynth_rs._pairwise_sum(a)


# -- the shared helper -----------------------------------------------------------------------------


@pytest.mark.parametrize("walls", list(WALLS))
def test_free_pressure_nodes_agrees_with_the_python_helper(walls):
    """The module helper, Python against Rust, at wall, edge, corner and interior nodes.

    Both spellings stay live on purpose. If ``_free_pressure_nodes`` had simply been replaced, this
    comparison would run Rust against Rust under the flag and pass having compared nothing —
    §23.6's emptied section, which this migration has now reached through five different doors.
    """
    py_room, rs_room = _rooms(**_kwargs(walls=WALLS[walls]))
    for room in (py_room, rs_room):
        room.inject(1e-3, at=(4 * H, 3 * H, 3 * H))
        for _ in range(11):
            room.step()
    nodes = tuple(
        np.array(v, dtype=np.intp)
        for v in ([0, 0, 4, 9, 4], [0, 3, 0, 7, 3], [0, 3, 3, 6, 3])
    )
    want = free_pressure_nodes_py(py_room, nodes)
    got = physsynth_rs._free_pressure_nodes(rs_room, nodes)
    assert np.array_equal(want, got)


def test_the_local_read_still_reproduces_the_full_array_closure():
    """``_free_pressure_nodes``' own docstring claim, now Rust reading a Rust room.

    A local divergence read must equal ``_divergence()``-then-closure exactly, on low **and** high
    faces (the ``idx < N`` and ``idx > 0`` branches are different code paths and both are here).
    §30.3 asserted this with a Python helper over a Rust room; porting the helper turns it into a
    claim about two Rust code paths, so it is re-asserted rather than inherited.
    """
    room = physsynth_rs.AirBox(**_kwargs(walls=WALLS["two lossy faces"]))
    room.inject(1e-3, at=(4 * H, 3 * H, 3 * H))
    for _ in range(9):
        room.step()
    div = room._divergence()
    p_free = room.p - room.k * room.rho0 * room.c0**2 * div
    beta = room._beta
    p_free = (p_free - beta * room.p) / (1.0 + beta)
    full = 0.5 * (p_free + room.p)
    nodes = tuple(
        np.array(v, dtype=np.intp)
        for v in ([0, 9, 0, 5, 4], [0, 7, 3, 0, 3], [0, 6, 3, 3, 2])
    )
    assert np.array_equal(physsynth_rs._free_pressure_nodes(room, nodes), full[nodes])


# -- RoomPort --------------------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [None, 0.05, 0.09])
@pytest.mark.parametrize("walls", list(WALLS))
def test_room_port_construction_is_bit_identical(radius, walls):
    """Every construction product, at a point port and at two ball sizes.

    ``node_count`` spans the reduction cutoff on purpose: a point port sums one term and the two
    balls sum 19 and 81, so this is the same assertion below and above §30.2's eight.
    """
    py, rs = _ports(room_kw=dict(walls=WALLS[walls]), at=(4 * H, 3 * H, 3 * H), radius=radius)
    assert py.index == rs.index
    assert py.node_count == rs.node_count
    assert py.radius == rs.radius
    for axis in range(3):
        assert np.array_equal(py.nodes[axis], rs.nodes[axis])
    assert np.array_equal(py._flat, rs._flat)
    assert np.array_equal(py.w, rs.w)
    assert py.R_room == rs.R_room
    assert py.volume == rs.volume


@pytest.mark.parametrize("radius", [None, 0.09])
@pytest.mark.parametrize("walls", list(WALLS))
def test_a_driven_room_port_keeps_the_field_bit_identical(radius, walls):
    """2,000 steps of read-solve-inject, with the port's own weights feeding the injection."""
    py_room, rs_room = _rooms(**_kwargs(walls=WALLS[walls]))
    py = RoomPortPy(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=radius)
    rs = physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=radius)
    want = _drive(py_room, py, 2000)
    got = _drive(rs_room, rs, 2000)
    assert np.max(np.abs(want - got)) == 0.0


def test_the_free_pressure_read_is_bit_identical_step_for_step():
    """Not only at the end: the quantity the coupled solve actually consumes, every step."""
    py_room, rs_room = _rooms(**_kwargs(walls=WALLS["two lossy faces"]))
    py = RoomPortPy(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    rs = physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    for n in range(400):
        assert py.free_pressure() == rs.free_pressure(), f"step {n}"
        amp = 1e-4 * np.sin(0.07 * n)
        py.inject(amp)
        rs.inject(amp)
        py_room.step()
        rs_room.step()


@pytest.mark.parametrize("radius", [None, 0.09])
def test_the_rooms_port_book_is_exact_at_every_size(radius):
    """The one quantity in this batch that was **not** bit-identical, and it belonged to the room.

    ``AirBox.step`` books a port injection as ``np.sum(w * 0.5 * (p_next + p_old))`` over the port's
    nodes. At one node that is a sum of length one and exact on any spelling. At 123 it is above
    §30.2's eight-element cutoff, and ``airbox.rs`` — written one batch before ``reduce`` existed —
    booked it with a plain left-to-right loop, so above the cutoff this was a **tolerance**: the
    two books wandered in and out of agreement at a last bit (1.6e-16 relative under an open-loop
    drive, exactly 0.0 under the closed loop this file uses — a bound, never a difference).

    The parked tightening (plan §31.11) is taken as of 2026-09-02: the book goes through
    ``reduce::sum_by`` and both arms assert equality. Kept as two arms rather than one assertion so
    the *structural* case (below the cutoff, exact whatever the spelling) and the *bought* case
    (above it, exact because of the transcription) stay distinguishable in a failure.
    """
    py_room, rs_room = _rooms(**_kwargs(walls=WALLS["all lossy"]))
    py = RoomPortPy(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=radius)
    rs = physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=radius)
    want = _drive(py_room, py, 300)
    got = _drive(rs_room, rs, 300)
    assert np.max(np.abs(want - got)) == 0.0
    if py.node_count < 8:
        assert py_room.injected == rs_room.injected, "below the cutoff: structural, any spelling"
    else:
        assert py_room.injected == rs_room.injected, "above the cutoff: reduce::sum_by transcribes"


# -- the distributed tier ------------------------------------------------------------------------


SURFACE_ROOM = dict(N=(12, 11, 9), h=0.03)
SURFACE_CASES = {
    "bilinear": dict(face="z0", spreading="bilinear"),
    "nearest": dict(face="z0", spreading="nearest"),
    "offcentre": dict(face="z0", origin=(0.055, 0.065)),
    "high face": dict(face="y1"),
    "lossy mounting": dict(face="z0"),
}


def _case_grid(case):
    """`nearest` needs a finer surface than `bilinear` does, or the comb refusal fires.

    Each surface node feeds exactly one air node under `nearest`, so a surface at the air grid's
    own spacing leaves every other node unfed. That refusal is the whole point of the spelling, so
    the fixture is refined rather than the check relaxed.
    """
    return _grid(11, 9, 0.15, 0.12) if case == "nearest" else _grid(6, 5, 0.15, 0.12)


@pytest.mark.parametrize("case", list(SURFACE_CASES))
def test_surface_port_construction_is_bit_identical(case):
    """``T``, ``R`` and the load matrix as *stored* — structure, order and data.

    ``_csr_equal`` compares ``indptr`` and ``indices`` as well as ``data``, which is the half §26.2
    found is a separate question from the values. Here both answers come out the same way and for
    a reason worth recording: ``T`` is built by ``coo_matrix(...).tocsr()`` and the product ends in
    a CSC-to-CSR conversion, and both canonicalize — so unlike the plate's matrices this tier
    needed no ``portable.canonical`` at all.
    """
    kw = dict(SURFACE_CASES[case])
    room_kw = dict(SURFACE_ROOM)
    if case == "lossy mounting":
        room_kw["walls"] = {"z0": impedance_from_zeta(3.0)}
    coords, areas = _case_grid(case)
    py, rs = _surfaces(room_kw=room_kw, coords=coords, areas=areas, **kw)
    assert py.index == rs.index
    assert py.node_count == rs.node_count
    assert py.n_surface == rs.n_surface
    assert py.origin == rs.origin
    assert py.footprint_empty == rs.footprint_empty
    assert py.net_area == rs.net_area
    assert py.in_plane_axes == rs.in_plane_axes
    assert py.spreading == rs.spreading
    assert py._where == rs._where
    for axis in range(3):
        assert np.array_equal(py.nodes[axis], rs.nodes[axis])
    assert np.array_equal(py._flat, rs._flat)
    assert np.array_equal(py.R, rs.R)
    assert py.T.nnz == rs.T.nnz
    assert _csr_equal(py.T, rs.T)
    assert py.load_matrix.nnz == rs.load_matrix.nnz
    assert _csr_equal(py.load_matrix, rs.load_matrix)


def test_interior_surface_port_construction_is_bit_identical():
    """The dipole tier: two straddling node planes, one ``R``, and a doubled load."""
    coords, areas = _grid(6, 5, 0.15, 0.12)
    py, rs = _surfaces(
        room_kw=SURFACE_ROOM, cls="interior", plane="z", index=4, coords=coords, areas=areas
    )
    assert py.index == rs.index
    assert (py.node_count, py.face_count) == (rs.node_count, rs.face_count)
    assert py.blocked_area == rs.blocked_area
    # The same four the wall test asserts. `footprint_empty` earns its place twice over here: the
    # reference counts unfed nodes over the CONCATENATED node set and this port counts over the low
    # plane alone. The two agree because the spans are deduplicated either way, but that is an
    # equivalence nothing else in the suite states.
    assert py.footprint_empty == rs.footprint_empty
    assert py.net_area == rs.net_area
    assert py.origin == rs.origin
    assert py.n_surface == rs.n_surface
    for axis in range(3):
        assert np.array_equal(py.nodes[axis], rs.nodes[axis])
        assert np.array_equal(py.nodes_lo[axis], rs.nodes_lo[axis])
        assert np.array_equal(py.nodes_hi[axis], rs.nodes_hi[axis])
    for d in range(2):
        assert np.array_equal(py._in_plane[d], rs._in_plane[d])
    assert np.array_equal(py.R, rs.R)
    assert _csr_equal(py.T, rs.T)
    assert _csr_equal(py.load_matrix, rs.load_matrix)


def test_the_interior_port_cuts_the_same_faces():
    """The one registration that mutates the room, and the only one that can leave it half-built."""
    coords, areas = _grid(6, 5, 0.15, 0.12)
    py, rs = _surfaces(
        room_kw=SURFACE_ROOM, cls="interior", plane="z", index=4, coords=coords, areas=areas
    )
    assert py.room.cut_faces == rs.room.cut_faces
    assert np.array_equal(py.room._cut_mask[2], rs.room._cut_mask[2])
    for d in range(3):
        assert np.array_equal(py.room._cut_index[2][d], rs.room._cut_index[2][d])


@pytest.mark.parametrize("case", ["bilinear", "nearest"])
def test_a_driven_surface_port_keeps_the_field_bit_identical(case):
    """1,000 steps of read-solve-inject through the distributed tier."""
    coords, areas = _case_grid(case)
    py, rs = _surfaces(
        room_kw=SURFACE_ROOM, coords=coords, areas=areas, **SURFACE_CASES[case]
    )
    want = _drive(py.room, py, 1000)
    got = _drive(rs.room, rs, 1000)
    assert np.max(np.abs(want - got)) == 0.0


def test_a_driven_interior_port_keeps_the_field_bit_identical():
    """The same, through the ``-q``/``+q`` pair and across a cut."""
    coords, areas = _grid(6, 5, 0.15, 0.12)
    py, rs = _surfaces(
        room_kw=SURFACE_ROOM, cls="interior", plane="z", index=4, coords=coords, areas=areas
    )
    want = _drive(py.room, py, 1000)
    got = _drive(rs.room, rs, 1000)
    assert np.max(np.abs(want - got)) == 0.0
    lo_py, hi_py = py.free_pressure()
    lo_rs, hi_rs = rs.free_pressure()
    assert np.array_equal(lo_py, lo_rs) and np.array_equal(hi_py, hi_rs)


# -- the association, and the fixture that cannot see it -----------------------------------------


def _triple(T, R, side):
    """``T^T diag(R) T`` with the diagonal folded into the given factor, over ``k`` ascending."""
    T = T.tocsr()
    n = T.shape[1]
    cols = [[] for _ in range(n)]
    for k in range(T.shape[0]):
        for a in range(T.indptr[k], T.indptr[k + 1]):
            cols[T.indices[a]].append((k, T.data[a]))
    out = []
    for i in range(n):
        acc = {}
        for k, tki in cols[i]:
            for a in range(T.indptr[k], T.indptr[k + 1]):
                j = int(T.indices[a])
                term = (tki * R[k]) * T.data[a] if side == "left" else tki * (R[k] * T.data[a])
                acc[j] = acc.get(j, 0.0) + term
        out.append(acc)
    return out


def test_the_load_matrix_uses_scipys_association_and_the_other_one_is_visible():
    """``(T_ki R_k) T_kj``, not ``T_ki (R_k T_kj)`` — and the difference is not hypothetical.

    Python left-associates ``T.T @ diags(R) @ T``, so the diagonal folds into ``T.T``. §26.5 asked
    whether an association is observable by looking at whether the two outer factors share a
    mantissa; here they do not, and about 30% of entries distinguish the two spellings. The port
    copies SciPy's choice rather than making one.
    """
    coords, areas = _grid(6, 5, 0.15, 0.12)
    _, rs = _surfaces(room_kw=SURFACE_ROOM, coords=coords, areas=areas, face="z0")
    left = _triple(rs.T, rs.R, "left")
    right = _triple(rs.T, rs.R, "right")
    load = rs.load_matrix.tocsr()
    differed = total = 0
    for i, row in enumerate(left):
        for j, v in row.items():
            assert load[i, j] == v, f"entry ({i}, {j}) is not the left association"
            total += 1
            if right[i][j] != v:
                differed += 1
    assert differed > 0, f"the two associations agreed at all {total} entries"


def test_the_nearest_fixture_is_structurally_blind_to_the_association():
    """And it is one of the six goldens in ``tests/test_airbox_dipole.py``.

    With ``spreading="nearest"`` every surface node lands on one air node with weight exactly 1, so
    every stored entry in a row of ``T`` is the same uniform node area — and ``(x d) x`` is
    ``x (d x)`` **identically**, for every ``x`` and ``d``, because the two outer factors are the
    same number rather than merely commensurate. §16.4's blind fixture with a proof instead of a
    measurement.
    """
    coords, areas = _case_grid("nearest")
    _, rs = _surfaces(
        room_kw=SURFACE_ROOM, coords=coords, areas=areas, face="z0", spreading="nearest"
    )
    T = rs.T.tocsr()
    for k in range(T.shape[0]):
        row = T.data[T.indptr[k] : T.indptr[k + 1]]
        assert row.size == 0 or np.all(row == row[0]), "the mechanism does not hold on this fixture"
    left = _triple(rs.T, rs.R, "left")
    right = _triple(rs.T, rs.R, "right")
    assert all(left[i][j] == right[i][j] for i in range(len(left)) for j in left[i])


# -- the seam ------------------------------------------------------------------------------------


def test_a_port_accepts_either_room():
    """Duck typing is the interface, and it is not decorative.

    §29.1 found `connection` polymorphic over its collaborators' types; the port tier has the same
    shape one level down. A Rust port reads its room through Python attributes and Python methods,
    so it works against the Python room too — which is what lets the flag be set per module rather
    than per process, and what the *next* batch's wrappers will need.
    """
    py_room = AirBoxPy(**_kwargs())
    rs_room = physsynth_rs.AirBox(**_kwargs())
    crossed = physsynth_rs.RoomPort(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    plain = physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    assert np.array_equal(crossed.w, plain.w)
    assert crossed.R_room == plain.R_room
    assert py_room._ports == [crossed]


def test_a_ports_methods_can_be_replaced_on_the_instance():
    """``tests/test_airbox_dipole.py`` swaps ``free_pressure`` and ``inject`` for lambdas.

    A `#[pyclass]` without `dict` refuses that outright, and the failure would be an
    ``AttributeError`` deep inside a wrapper's step rather than anything a physics bar could see.
    Pinned directly because it is a property of the *binding* that no number can detect.
    """
    room = physsynth_rs.AirBox(**_kwargs())
    port = physsynth_rs.RoomPort(room=room, at=(4 * H, 3 * H, 3 * H), radius=None)
    free, inject = port.free_pressure, port.inject
    port.free_pressure = lambda: -free()
    port.inject = lambda q: inject(-q)
    room.inject(1e-3, at=(4 * H, 3 * H, 3 * H))
    room.step()
    assert port.free_pressure() == -free()
    port.inject(1e-4)
    assert room._pending_ports[0][2] == -1e-4


def test_the_four_written_attributes_are_settable_and_read_back():
    """``T``, ``load_matrix``, ``R`` and ``areas`` are replaced wholesale by five test files.

    §30.3's rule — *grep for assignment, not only for reference* — applied to this tier. ``areas``
    is the one that must be read back live, because zeroing it is how two tests switch a surface's
    radiating area off while leaving its node set in place.
    """
    coords, areas = _grid(6, 5, 0.15, 0.12)
    _, rs = _surfaces(room_kw=SURFACE_ROOM, coords=coords, areas=areas, face="z0")
    rs.T = sparse.csr_matrix(rs.T.shape)
    rs.load_matrix = sparse.csr_matrix(rs.load_matrix.shape)
    rs.R = 0.5 * np.array(rs.R)
    assert rs.T.nnz == 0 and rs.load_matrix.nnz == 0
    before = rs.net_area
    rs.areas = np.zeros_like(np.asarray(rs.areas))
    assert before > 0.0 and rs.net_area == 0.0
    rs._queued_at = 7
    assert rs._queued_at == 7


def test_the_derived_index_arrays_refuse_assignment():
    """A deliberate narrowing: `nodes` and `w` are cached alongside a Rust index vector, so a
    silent assignment would leave the two disagreeing. Nothing in the suite writes them, and an
    ``AttributeError`` is the loud direction if something ever starts.
    """
    room = physsynth_rs.AirBox(**_kwargs())
    port = physsynth_rs.RoomPort(room=room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    for name in ("nodes", "w", "R_room", "index", "_flat"):
        with pytest.raises(AttributeError):
            setattr(port, name, None)


def test_set_state_resets_every_registered_port():
    """``AirBox.set_state`` writes ``port._queued_at = -1`` on everything in ``room._ports``."""
    room = physsynth_rs.AirBox(**_kwargs())
    port = physsynth_rs.RoomPort(room=room, at=(4 * H, 3 * H, 3 * H), radius=None)
    port.inject(1e-4)
    assert port._queued_at == 0
    room.set_state(np.zeros_like(room.p))
    assert port._queued_at == -1


# -- refusals ------------------------------------------------------------------------------------


def test_an_omitted_spreading_and_an_explicit_none_are_different_arguments():
    """§24.7, which PyO3 collapses by default and which no parity file had ever thought to try.

    An omitted ``spreading`` builds the bilinear default; ``spreading=None`` is a value the
    reference rejects by name. A plain ``Option`` makes those the same call and silently accepts
    the second — and the first draft of this binding did, which eleven tests in
    ``test_airbox_surface.py`` caught because their fixture forwards a ``None`` default.
    """
    coords, areas = _grid(6, 5, 0.15, 0.12)
    room = physsynth_rs.AirBox(**_kwargs(**SURFACE_ROOM))
    ok = physsynth_rs.SurfacePort(room=room, face="z0", coords=coords, areas=areas)
    assert ok.spreading == "bilinear"
    with pytest.raises(ValueError, match="unknown spreading"):
        physsynth_rs.SurfacePort(
            room=physsynth_rs.AirBox(**_kwargs(**SURFACE_ROOM)),
            face="z0",
            coords=coords,
            areas=areas,
            spreading=None,
        )


@pytest.mark.parametrize(
    ("kw", "pattern"),
    [
        (dict(at=(4 * H, 3 * H, 3 * H), radius=0.0), "positive length"),
        (dict(at=(4 * H, 3 * H, 3 * H), radius=1e-4), "smaller than the grid"),
        (dict(at=(99.0, 3 * H, 3 * H), radius=None), "outside the room"),
    ],
)
def test_room_port_refusals_match(kw, pattern):
    """Both implementations refuse, and neither leaves the room holding a half-built port."""
    py_room, rs_room = _rooms(**_kwargs())
    with pytest.raises(ValueError, match=pattern):
        RoomPortPy(room=py_room, **kw)
    with pytest.raises(ValueError, match=pattern):
        physsynth_rs.RoomPort(room=rs_room, **kw)
    assert py_room._ports == [] and rs_room._ports == []


def test_an_overlapping_port_is_refused_by_both():
    py_room, rs_room = _rooms(**_kwargs())
    RoomPortPy(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=0.09)
    with pytest.raises(ValueError, match="shares node"):
        RoomPortPy(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=None)
    with pytest.raises(ValueError, match="shares node"):
        physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=None)


def test_a_port_on_an_open_face_is_refused_by_both():
    kw = _kwargs(walls={"z0": "open"})
    py_room, rs_room = AirBoxPy(**kw), physsynth_rs.AirBox(**kw)
    with pytest.raises(ValueError, match="open"):
        RoomPortPy(room=py_room, at=(4 * H, 3 * H, 0.0), radius=None)
    with pytest.raises(ValueError, match="open"):
        physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 0.0), radius=None)


def test_solving_twice_in_one_room_step_is_refused_by_both():
    py_room, rs_room = _rooms(**_kwargs())
    py = RoomPortPy(room=py_room, at=(4 * H, 3 * H, 3 * H), radius=None)
    rs = physsynth_rs.RoomPort(room=rs_room, at=(4 * H, 3 * H, 3 * H), radius=None)
    py.inject(1e-4)
    rs.inject(1e-4)
    with pytest.raises(RuntimeError, match="twice within one room step"):
        py.inject(1e-4)
    with pytest.raises(RuntimeError, match="twice within one room step"):
        rs.inject(1e-4)


def test_a_stencil_reaching_the_planes_rim_is_refused_by_both():
    coords, areas = _grid(6, 5, 0.35, 0.30)
    with pytest.raises(ValueError, match="rim"):
        _surfaces(room_kw=SURFACE_ROOM, coords=coords, areas=areas, face="z0")


def test_an_interior_index_out_of_range_is_refused_by_both():
    coords, areas = _grid(6, 5, 0.15, 0.12)
    with pytest.raises(ValueError, match="out of range"):
        _surfaces(
            room_kw=SURFACE_ROOM, cls="interior", plane="z", index=0, coords=coords, areas=areas
        )


def test_the_wrong_shaped_injection_is_refused_by_both():
    coords, areas = _grid(6, 5, 0.15, 0.12)
    py, rs = _surfaces(room_kw=SURFACE_ROOM, coords=coords, areas=areas, face="z0")
    with pytest.raises(ValueError, match="per-node"):
        py.inject(np.zeros(3))
    with pytest.raises(ValueError, match="per-node"):
        rs.inject(np.zeros(3))
    py2, rs2 = _surfaces(
        room_kw=SURFACE_ROOM, cls="interior", plane="z", index=4, coords=coords, areas=areas
    )
    with pytest.raises(ValueError, match="per-FACE"):
        py2.inject(np.zeros(3))
    with pytest.raises(ValueError, match="per-FACE"):
        rs2.inject(np.zeros(3))


def test_a_zero_area_surface_still_names_its_nodes():
    """SciPy keeps a stored ``0.0`` and ``Csr::from_rows`` drops one, so the port needs the other
    constructor.

    The reference is explicit that entries whose *geometric* weight is nonzero stay even when the
    node's **area** is zero, so a zero-area surface still names the nodes it covers and the
    ``T = 0`` reduction to a bare resonator stays exercisable. No fixture the suite builds has an
    explicit zero — §16.4's shape — so it is constructed here on purpose.
    """
    coords, areas = _grid(6, 5, 0.15, 0.12)
    areas = areas.copy()
    areas[: areas.size // 2] = 0.0
    py, rs = _surfaces(room_kw=SURFACE_ROOM, coords=coords, areas=areas, face="z0")
    assert py.T.nnz == rs.T.nnz
    assert int(np.sum(rs.T.data == 0.0)) > 0
    assert _csr_equal(py.T, rs.T)
    assert _csr_equal(py.load_matrix, rs.load_matrix)
