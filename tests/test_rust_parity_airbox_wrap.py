"""Rust parity for ``physsynth.core.airbox``'s **wrapper tier** — plan §32.

``RoomLoadedBody``, ``RoomLoadedPlate``, ``RoomSuspendedPlate``, ``RoomLoadedVKPlate``,
``RoomSuspendedVKPlate`` and the two seams (``_PlateSurface``, ``_VKPlateSurface``) the four plate
wrappers drive. This finishes ``airbox.py``'s three-batch port; the membrane wrappers stay Python.

**Everything here is exact, and §31.11's prediction that this would be a Group D batch is wrong.**
The prediction was reasonable — all six of the file's ``splu`` factorizations are in this tier — and
it is wrong for a reason that turns out to be structural rather than lucky: *the wrapper does not
own a factorization, it calls one*. ``splu`` is read as a module global at construction, exactly as
the reference reads it, so a Rust wrapper and a Python wrapper factor the same matrix with the same
routine and there is no solver difference to tolerance. What §24's measured-tolerance rule governs
is the module that *implements* a solve, not every module that appears downstream of one — the
group is a property of ownership, not of the file.

**Why the arithmetic is exact is the batch's finding and is worth reading before the assertions.**
The tier below stores ``T``, ``R`` and ``load_matrix`` as plain Python slots (§31), because ten
tests replace them — or the port's methods, or the factorization — on the instance. So a Rust
wrapper *cannot* cache any of them: it reads them live and computes through SciPy, exactly as the
reference does. Every sparse product, the assembly, the factorization and ``np.dot`` are therefore
the same call on the same object in both languages, and what is left for Rust is the control flow,
the guards, the ledgers and the elementwise arithmetic between those calls — which is exact
because elementwise ``+ - * /`` on doubles admits no reassociation. The price is that this tier
wins no time (measured, §32), and the alternative was ten tests that pass having asserted nothing.

The two searches that shape this file:

* ``_lu_loaded``, ``port.T``, ``port.load_matrix``, ``port.R`` and the port's own methods are
  **replaced by tests**, so :func:`test_a_zeroed_operator_reaches_the_rust_wrapper` and its
  neighbours assert that a Rust wrapper honours each substitution. Without them the port would be
  the seventh door onto §23.6's emptied comparison.
* A wrapper is a **drop-in for the model it holds**, so every name ``connection.py`` reads has to
  reach the plate through ``__getattr__``. A ``#[pyclass]`` getter is the opposite default from
  Python's ``__getattr__`` — it shadows permanently — so
  :func:`test_every_delegated_name_reaches_the_model` walks that list by name.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from helpers import (
    AIRBOX_DIPOLE_INDEX,
    AIRBOX_SURFACE_FS,
    AIRBOX_SURFACE_KAPPA,
    AIRBOX_SURFACE_PLATE_L,
    AIRBOX_SURFACE_RHO,
    PLATE_THETA_DEFAULT,
    make_air_vk_plate,
    make_surface_room,
    plate_bump,
    vk_strike,
)
from scipy import sparse
from scipy.sparse.linalg import splu as scipy_splu

from physsynth.core import airbox
from physsynth.core.airbox import (
    RoomLoadedBodyPy,
    RoomLoadedPlatePy,
    RoomLoadedVKPlatePy,
    RoomSuspendedPlatePy,
    RoomSuspendedVKPlatePy,
    SurfacePortPy,
    _PlateSurfacePy,
    _VKPlateSurfacePy,
)
from physsynth.core.body import ModalBody
from physsynth.core.plate import Plate

# NOT a bare `import physsynth_rs`: the default gate does not build the extension, so a module-scope
# import is a collection error there rather than a skip.
physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

BOUNDARIES = ("supported", "free")
N_PLATE = 8
STEPS = 60


# -- fixtures -------------------------------------------------------------------------------


def _plate(boundary="supported", *, sigma=0.0, N=N_PLATE):
    return Plate(
        Lx=AIRBOX_SURFACE_PLATE_L,
        Ly=AIRBOX_SURFACE_PLATE_L,
        kappa=AIRBOX_SURFACE_KAPPA,
        rho=AIRBOX_SURFACE_RHO,
        fs=AIRBOX_SURFACE_FS,
        N=N,
        sigma=sigma,
        theta=PLATE_THETA_DEFAULT,
        boundary=boundary,
    )


def _vk_plate(boundary="supported", *, nonlinear=True, sigma=0.0):
    return make_air_vk_plate(N=N_PLATE, boundary=boundary, nonlinear=nonlinear, sigma=sigma)


def _drive(plate, amplitude=3.0):
    """A one-node external force — the driving-point coupling a bridge would inject."""
    f = np.zeros(plate.n_live)
    f[plate.n_live // 3] = amplitude
    return f


@contextlib.contextmanager
def _rust_collaborators():
    """Point ``airbox``'s module globals at the Rust seams and ports for the duration.

    §28.4 in the mirror. A Rust wrapper reads ``_PlateSurface``, ``SurfacePort`` and ``splu`` as
    module globals, because that is what the reference does — so in the default gate, where the
    swap is off, a Rust wrapper built without this patch holds **Python** collaborators. That is a
    useful comparison (it isolates the wrapper's own arithmetic) and a misleading one to make by
    accident, which is why both are spelled out here rather than inherited.
    """
    names = {
        "_PlateSurface": physsynth_rs._PlateSurface,
        "_VKPlateSurface": physsynth_rs._VKPlateSurface,
        "RoomPort": physsynth_rs.RoomPort,
        "SurfacePort": physsynth_rs.SurfacePort,
        "InteriorSurfacePort": physsynth_rs.InteriorSurfacePort,
    }
    saved = {n: getattr(airbox, n) for n in names}
    for name, value in names.items():
        setattr(airbox, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(airbox, name, value)


def _make(cls, plate, room, tier, **kw):
    if tier == "baffled":
        return cls(plate=plate, room=room, face="z0", **kw)
    return cls(plate=plate, room=room, plane="z", index=AIRBOX_DIPOLE_INDEX, **kw)


PY_CLASSES = {
    ("linear", "baffled"): RoomLoadedPlatePy,
    ("linear", "suspended"): RoomSuspendedPlatePy,
    ("vk", "baffled"): RoomLoadedVKPlatePy,
    ("vk", "suspended"): RoomSuspendedVKPlatePy,
}
RS_CLASSES = {
    ("linear", "baffled"): physsynth_rs.RoomLoadedPlate,
    ("linear", "suspended"): physsynth_rs.RoomSuspendedPlate,
    ("vk", "baffled"): physsynth_rs.RoomLoadedVKPlate,
    ("vk", "suspended"): physsynth_rs.RoomSuspendedVKPlate,
}
LEDGER = {"baffled": "surface_pressure", "suspended": "pressure_jump"}


def _pair(kind="linear", tier="baffled", boundary="supported", *, nonlinear=True, **kw):
    """One Python wrapper and one Rust wrapper, over identical but **independent** rooms.

    Sharing one room would put two ports on the same nodes and be refused, so each side gets its
    own — identical by construction, because :func:`make_surface_room` is deterministic in its
    arguments and both sides take its defaults.
    """
    build = _plate if kind == "linear" else (
        lambda b: _vk_plate(b, nonlinear=nonlinear)
    )
    py_plate, rs_plate = build(boundary), build(boundary)
    py_inst = _make(PY_CLASSES[(kind, tier)], py_plate, make_surface_room(), tier, **kw)
    with _rust_collaborators():
        rs_inst = _make(RS_CLASSES[(kind, tier)], rs_plate, make_surface_room(), tier, **kw)
    return py_inst, rs_inst


def _run(py_inst, rs_inst, steps=STEPS, f_ext=None, tier="baffled"):
    """Step both sides in the contract's order and assert the state at every step."""
    for n in range(steps):
        py_inst.step(f_ext)
        py_inst.room.step()
        rs_inst.step(f_ext)
        rs_inst.room.step()
        assert np.array_equal(py_inst.plate.u, rs_inst.plate.u), f"u at step {n}"
    assert np.array_equal(py_inst.plate.u_prev, rs_inst.plate.u_prev)
    assert np.array_equal(py_inst.room.p, rs_inst.room.p)
    assert np.array_equal(
        py_inst.nodal_volume_velocity, rs_inst.nodal_volume_velocity
    )
    assert np.array_equal(
        getattr(py_inst, LEDGER[tier]), getattr(rs_inst, LEDGER[tier])
    )
    assert py_inst.radiated_energy == rs_inst.radiated_energy
    assert py_inst.volume_velocity == rs_inst.volume_velocity
    assert py_inst.energy() == rs_inst.energy()
    assert py_inst.n == rs_inst.n


def _csr_equal(a, b):
    a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
    a.sort_indices()
    b.sort_indices()
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


# -- 1. the seams ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("kind", ["linear", "vk"])
def test_the_seam_reports_the_same_surface(kind, boundary):
    """``surface()``, ``areas`` and ``denominator`` — the three numbers the port is built from."""
    plate = _plate(boundary) if kind == "linear" else _vk_plate(boundary)
    py_seam = (_PlateSurfacePy if kind == "linear" else _VKPlateSurfacePy)(plate)
    rs_seam = (
        physsynth_rs._PlateSurface if kind == "linear" else physsynth_rs._VKPlateSurface
    )(plate)
    py_coords, py_areas = py_seam.surface()
    rs_coords, rs_areas = rs_seam.surface()
    assert np.array_equal(py_coords, rs_coords)
    assert np.array_equal(py_areas, rs_areas)
    assert py_seam.denominator == rs_seam.denominator
    assert py_seam.k == rs_seam.k


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("kind", ["linear", "vk"])
def test_the_seams_unloaded_matrix_is_bit_identical(kind, boundary):
    """``a_bare()`` is the plate's own ``A`` reassembled, and the anchor that pins the
    reassembly compares it across model classes — so it has to agree to the stored byte, not to a
    tolerance (plan §15.2, and §28's finding that the claim reaches a client)."""
    plate = _plate(boundary) if kind == "linear" else _vk_plate(boundary)
    py_seam = (_PlateSurfacePy if kind == "linear" else _VKPlateSurfacePy)(plate)
    rs_seam = (
        physsynth_rs._PlateSurface if kind == "linear" else physsynth_rs._VKPlateSurface
    )(plate)
    assert _csr_equal(py_seam.a_bare(), rs_seam.a_bare())


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("kind", ["linear", "vk"])
def test_the_seams_right_hand_side_is_bit_identical(kind, boundary, forced):
    """The five-term theta-scheme RHS, as a Rust fold in the reference's own operand order.

    This is where a transcription slip would live and be nearly invisible: every term is O(1) next
    to its neighbours, so a mis-association shifts a last bit rather than breaking anything, and
    §19.4's lesson is that such a slip can leave the trajectory bit-identical for thousands of
    steps. Comparing the RHS itself is the sharp detector.
    """
    plate = _plate(boundary) if kind == "linear" else _vk_plate(boundary)
    plate.set_state(plate_bump(plate) if kind == "linear" else vk_strike(plate, 1e-4))
    plate.step()  # so u and u_prev differ -- an RHS read at rest hides an operand swap
    py_seam = (_PlateSurfacePy if kind == "linear" else _VKPlateSurfacePy)(plate)
    rs_seam = (
        physsynth_rs._PlateSurface if kind == "linear" else physsynth_rs._VKPlateSurface
    )(plate)
    f_ext = _drive(plate) if forced else None
    assert np.array_equal(py_seam.rhs(f_ext), rs_seam.rhs(f_ext))


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_linear_seam_commits_the_same_state(boundary):
    """``commit`` rolls two levels and refreshes ``_accel`` — the acceleration is the half a
    trajectory comparison would miss, because nothing reads it back into the scheme."""
    a, b = _plate(boundary), _plate(boundary)
    for p in (a, b):
        p.set_state(plate_bump(p))
    u_next = plate_bump(a, 5e-4)
    _PlateSurfacePy(a).commit(u_next)
    physsynth_rs._PlateSurface(b).commit(u_next)
    assert np.array_equal(a.u, b.u)
    assert np.array_equal(a.u_prev, b.u_prev)
    assert np.array_equal(a._accel, b._accel)
    assert a.n == b.n


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_von_karman_seam_solves_and_reports_the_same(boundary, nonlinear):
    """The Picard hook, including the three diagnostics it writes on the model.

    ``n_iters`` is the quantity §27.5 found the whole agreement regime turns on, and it is written
    by this hook rather than by the model's own step, so a port that got the loop right and the
    bookkeeping wrong would look identical in the state and lie in the read-out.
    """
    a, b = _vk_plate(boundary, nonlinear=nonlinear), _vk_plate(boundary, nonlinear=nonlinear)
    for p in (a, b):
        p.set_state(vk_strike(p))
    py_seam, rs_seam = _VKPlateSurfacePy(a), physsynth_rs._VKPlateSurface(b)
    lu_a = scipy_splu(sparse.csc_matrix(py_seam.a_bare()))
    lu_b = scipy_splu(sparse.csc_matrix(rs_seam.a_bare()))
    for _ in range(20):
        w_a, f_a = py_seam.solve(lu_a, py_seam.rhs(None))
        w_b, f_b = rs_seam.solve(lu_b, rs_seam.rhs(None))
        assert np.array_equal(w_a, w_b)
        assert np.array_equal(f_a, f_b)
        assert a.n_iters == b.n_iters
        assert a.converged == b.converged
        assert a.last_residual == b.last_residual
        py_seam.commit(w_a, f_a)
        rs_seam.commit(w_b, f_b)
    assert np.array_equal(a.u, b.u)
    assert np.array_equal(a.F, b.F)
    assert np.array_equal(a.F_prev, b.F_prev)


# -- 2. construction ------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("tier", ["baffled", "suspended"])
@pytest.mark.parametrize("kind", ["linear", "vk"])
def test_construction_is_bit_identical(kind, tier, boundary):
    """The loaded assembly and both of its cost reports.

    ``lu_nnz`` is the one that would catch a matrix built with the right values in the wrong
    sparsity pattern: ``eliminate_zeros`` is what makes a zero-area surface factor the plate's OWN
    matrix, and dropping it changes the fill without changing a single answer.
    """
    py_inst, rs_inst = _pair(kind, tier, boundary)
    assert py_inst._load_scale == rs_inst._load_scale
    assert py_inst._denominator == rs_inst._denominator
    assert py_inst.nnz_growth == rs_inst.nnz_growth
    assert py_inst.lu_nnz == rs_inst.lu_nnz
    assert py_inst.k == rs_inst.k
    assert py_inst.n == rs_inst.n == 0
    assert np.array_equal(py_inst.port.T.toarray(), rs_inst.port.T.toarray())
    assert _csr_equal(py_inst.port.load_matrix, rs_inst.port.load_matrix)


def test_the_wrapper_reads_splu_as_a_module_global():
    """Which is what makes §31.11's Group D prediction not bind, and is asserted rather than
    argued: the wrapper factors with whatever ``airbox.splu`` names at construction, so both
    languages use one routine and there is no solver difference to tolerance."""
    seen = []

    def spy(a):
        seen.append(a)
        return scipy_splu(a)

    plate, room = _plate(), make_surface_room()
    with _rust_collaborators():
        original = airbox.splu
        airbox.splu = spy
        try:
            inst = physsynth_rs.RoomLoadedPlate(plate=plate, room=room, face="z0")
        finally:
            airbox.splu = original
    assert len(seen) == 1, "the Rust wrapper did not go through the module-global splu"
    assert inst.lu_nnz > 0


def test_a_rust_wrapper_built_without_the_patch_holds_python_collaborators():
    """§28.4, spelled out rather than tripped over. The module globals decide, so a parity fixture
    that does not say which implementation it wants gets whichever the flag left behind."""
    plate, room = _plate(), make_surface_room()
    bare = physsynth_rs.RoomLoadedPlate(plate=plate, room=room, face="z0")
    assert type(bare._surface) is _PlateSurfacePy
    assert type(bare.port) is SurfacePortPy
    with _rust_collaborators():
        patched = physsynth_rs.RoomLoadedPlate(
            plate=_plate(), room=make_surface_room(), face="z0"
        )
    assert type(patched._surface) is physsynth_rs._PlateSurface
    assert type(patched.port) is physsynth_rs.SurfacePort


# -- 3. the trajectory ----------------------------------------------------------------------


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("tier", ["baffled", "suspended"])
def test_a_driven_linear_wrapper_is_bit_identical(tier, boundary, forced):
    py_inst, rs_inst = _pair("linear", tier, boundary)
    u0 = plate_bump(py_inst.plate)
    py_inst.set_state(u0)
    rs_inst.set_state(u0)
    _run(py_inst, rs_inst, f_ext=_drive(py_inst.plate) if forced else None, tier=tier)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("tier", ["baffled", "suspended"])
def test_a_driven_von_karman_wrapper_is_bit_identical(tier, boundary, nonlinear):
    """Exact, and §27.5 says not to expect that of a von Karman plate's *trajectory* — the two
    implementations here are not two discretizations but one, calling one Picard loop through one
    factorization, so what would separate them is a transcription difference and nothing else."""
    py_inst, rs_inst = _pair("vk", tier, boundary, nonlinear=nonlinear)
    u0 = vk_strike(py_inst.plate)
    py_inst.set_state(u0)
    rs_inst.set_state(u0)
    _run(py_inst, rs_inst, steps=40, f_ext=_drive(py_inst.plate, 1.0), tier=tier)
    assert py_inst.plate.n_iters == rs_inst.plate.n_iters
    assert py_inst.plate.last_residual == rs_inst.plate.last_residual


@pytest.mark.parametrize("tier", ["baffled", "suspended"])
def test_nonlinear_false_reduces_to_the_linear_wrapper_in_rust_too(tier):
    """The anchor that made this batch's shape (plan §15.2): ``RoomLoadedVKPlate(nonlinear=False)``
    must be ``array_equal`` to ``RoomLoadedPlate``. It is asserted in ``test_airbox_vk.py`` for
    whichever implementation the flag selected; here it is asserted for the Rust pair explicitly,
    because with the flag off that file compares Python with Python."""
    from helpers import vk_linear_twin

    vk = _vk_plate("supported", nonlinear=False)
    twin = vk_linear_twin(vk)
    with _rust_collaborators():
        a = _make(RS_CLASSES[("vk", tier)], vk, make_surface_room(), tier)
        b = _make(RS_CLASSES[("linear", tier)], twin, make_surface_room(), tier)
    u0 = vk_strike(vk, 1e-4)
    a.set_state(u0)
    b.set_state(u0)
    f_ext = _drive(vk, 1.0)
    for n in range(40):
        a.step(f_ext)
        a.room.step()
        b.step(f_ext)
        b.room.step()
        assert np.array_equal(vk.u, twin.u), f"step {n}"
    assert a.radiated_energy == b.radiated_energy
    assert a.energy() == b.energy()


def test_the_shared_factorization_changes_nothing():
    """§24.4's manoeuvre, fifth use — and here it is a *negative* control rather than a repair.

    Put both languages' wrappers on the crate's sparse LU instead of SuperLU and the trajectory is
    still bit-identical, because it already was: the wrapper never owned the difference.
    """

    class _Fill:
        __slots__ = ("nnz",)

        def __init__(self, nnz):
            self.nnz = nnz

    class _RustSuperLU:
        __slots__ = ("_lu", "L", "U")

        def __init__(self, a):
            m = sparse.csr_matrix(a, copy=True)
            m.sort_indices()
            self._lu = physsynth_rs.SparseLu(
                m.data, m.indices.astype(np.int32), m.indptr.astype(np.int32), m.shape[0]
            )
            l_nnz, u_nnz = self._lu.nnz
            self.L, self.U = _Fill(l_nnz), _Fill(u_nnz)

        def solve(self, b):
            return np.asarray(self._lu.solve(np.ascontiguousarray(b, dtype=float)))

    original = airbox.splu
    airbox.splu = _RustSuperLU
    try:
        py_inst, rs_inst = _pair("linear", "baffled", "supported")
        assert isinstance(py_inst._lu_loaded, _RustSuperLU)
        assert isinstance(rs_inst._lu_loaded, _RustSuperLU)
        u0 = plate_bump(py_inst.plate)
        py_inst.set_state(u0)
        rs_inst.set_state(u0)
        _run(py_inst, rs_inst, steps=40)
    finally:
        airbox.splu = original


# -- 4. the substitutions ten tests make ----------------------------------------------------


@pytest.mark.parametrize("tier", ["baffled", "suspended"])
def test_a_zeroed_operator_reaches_the_rust_wrapper(tier):
    """Switching the coupling off must reduce a loaded plate to a **bare** one, byte for byte.

    Three ``test_airbox_*`` files do exactly this, and they can only do it because the port stores
    ``T`` and ``load_matrix` as replaceable slots. A wrapper that cached either would keep loading
    the plate and the test would pass anyway — it asserts equality with the bare plate, which is
    what a cached-but-zeroed load would silently stop being.
    """
    plate, bare = _plate(), _plate()
    with _rust_collaborators():
        inst = _make(RS_CLASSES[("linear", tier)], plate, make_surface_room(), tier)
    inst.port.T = sparse.csr_matrix(inst.port.T.shape)
    inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
    a_bare = inst._surface.a_bare()
    # `airbox.splu` and not SciPy's: unit 5's deletion made `Plate` the Rust class on every path,
    # and the bare plate below factors with the crate's LU. Refactoring this one with SuperLU
    # would compare two solvers and fail at ~1e-16 for a reason that is not this test's subject
    # (plan §43, §24.2). `airbox.py`'s own `splu` is the crate's for exactly the same reason.
    inst._lu_loaded = airbox.splu(sparse.csc_matrix(a_bare))
    u0 = plate_bump(plate)
    inst.set_state(u0)
    bare.set_state(u0)
    for n in range(40):
        inst.step()
        inst.room.step()
        bare.step()
        assert np.array_equal(plate.u, bare.u), f"step {n}"
    assert inst.radiated_energy == 0.0
    assert np.array_equal(inst.room.p, np.zeros_like(inst.room.p))


def test_a_replaced_factorization_is_the_one_the_rust_wrapper_solves_with():
    """``inst._lu_loaded = splu(a)`` is how three tests halve a load without rebuilding a room."""
    plate = _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
    calls = []

    class _Spy:
        def __init__(self, lu):
            self._lu = lu
            self.L = lu.L
            self.U = lu.U

        def solve(self, b):
            calls.append(b)
            return self._lu.solve(b)

    inst._lu_loaded = _Spy(inst._lu_loaded)
    inst.set_state(plate_bump(plate))
    inst.step()
    assert len(calls) == 1, "the Rust wrapper solved with a factorization of its own"


def test_a_halved_load_matrix_reaches_the_rust_wrapper():
    """``test_airbox_dipole`` and ``test_airbox_membrane`` scale ``R`` and ``load_matrix`` together
    to halve the coupling; the wrapper must read both live, on the update path."""
    plate, twin = _plate(), _plate()
    with _rust_collaborators():
        scaled = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
        full = physsynth_rs.RoomLoadedPlate(
            plate=twin, room=make_surface_room(), face="z0"
        )
    scaled.port.R = 0.5 * scaled.port.R
    scaled.port.load_matrix = 0.5 * scaled.port.load_matrix
    a = scaled._surface.a_bare() + scaled._load_scale * scaled.port.load_matrix
    scaled._lu_loaded = airbox.splu(sparse.csc_matrix(a))
    u0 = plate_bump(plate)
    scaled.set_state(u0)
    full.set_state(u0)
    for _ in range(30):
        scaled.step()
        scaled.room.step()
        full.step()
        full.room.step()
    assert not np.array_equal(plate.u, twin.u), "halving the load changed nothing"
    assert abs(scaled.radiated_energy) < abs(full.radiated_energy)


def test_a_ports_methods_replaced_on_the_instance_reach_a_rust_wrapper():
    """``test_airbox_dipole`` inverts a sign convention by replacing ``free_pressure`` and
    ``inject`` on the port instance — no attribute search finds this door (§31.6), and a wrapper
    that called the port's Rust method directly would step straight past it."""
    plate = _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomSuspendedPlate(
            plate=plate, room=make_surface_room(), plane="z", index=AIRBOX_DIPOLE_INDEX
        )
    free, inject = inst.port.free_pressure, inst.port.inject
    seen = []
    inst.port.free_pressure = lambda: (seen.append("read"), tuple(reversed(free())))[1]
    inst.port.inject = lambda q: (seen.append("write"), inject(-q))[1]
    inst.set_state(plate_bump(plate))
    inst.step()
    assert seen == ["read", "write"]


# -- 5. delegation --------------------------------------------------------------------------

# Every name `connection.py` reads off a body or a plate collaborator. A `#[pyclass]` getter
# shadows delegation permanently, where Python's `__getattr__` only fires on a miss -- so this list
# is the contract, and the source it comes from says "NOTHING here may shadow a name that bridge
# reads".
PLATE_DELEGATED = (
    "n_live", "u_prev", "boundary", "Lx", "Ly", "X", "Y", "mask", "rho", "h",
    "theta", "kappa", "sigma", "state", "to_live", "pickup_index_at", "fs",
)
BODY_DELEGATED = (
    "phi", "m", "M", "q", "q_prev", "omega", "sigma", "a", "fs", "bridge_displacement",
)


@pytest.mark.parametrize("name", PLATE_DELEGATED)
def test_every_delegated_name_reaches_the_model(name):
    plate = _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
    mine, theirs = getattr(inst, name), getattr(plate, name)
    if isinstance(theirs, np.ndarray):
        assert np.array_equal(mine, theirs)
    elif callable(theirs):
        assert mine.__self__ is plate
    else:
        assert mine == theirs


@pytest.mark.parametrize("name", BODY_DELEGATED)
def test_every_delegated_body_name_reaches_the_model(name):
    body = _body()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedBody(
            body=body, room=make_surface_room(), at=_AT, radius=None
        )
    mine, theirs = getattr(inst, name), getattr(body, name)
    if isinstance(theirs, np.ndarray):
        assert np.array_equal(mine, theirs)
    else:
        assert mine == theirs


PAIRS = [("linear", "baffled"), ("linear", "suspended"), ("vk", "baffled"), ("vk", "suspended")]


@pytest.mark.parametrize(("kind", "tier"), PAIRS)
def test_the_wrappers_step_count_is_settable(kind, tier):
    """§33.2, asserted on the four classes the setter was *widened* to rather than the two that
    needed it.

    The membrane batch found that a ``#[getter]`` with no ``#[setter]`` is still a data descriptor,
    so a plain Python attribute becomes read-only the moment it is ported, and put the setter in
    the shared macro — which changed these four shipped classes. Nothing in that batch's own run
    could see the difference: a setter adds no name to ``dir()``, so
    :func:`test_the_wrappers_own_surface_matches_the_reference` passes identically either way.
    """
    py_inst, rs_inst = _pair(kind, tier)
    for inst in (py_inst, rs_inst):
        inst.n = 7
        inst.n += 3
    assert py_inst.n == rs_inst.n == 10


@pytest.mark.parametrize(("kind", "tier"), PAIRS)
def test_the_wrappers_own_surface_matches_the_reference(kind, tier):
    """DERIVED, not listed — and this is the guard §32.6 actually needs.

    The hand-written list above says what ``connection.py`` reads *today*; this one fires on the
    next getter added to any of these classes, whether or not anybody thought to list the name it
    shadows. §17.6 and §23.7 are the same lesson twice: a derive catches the paste a list forgets.
    Compared on INSTANCES rather than classes, because the reference sets these in ``__init__``
    where PyO3 makes them type descriptors — the class-level sets differ by construction and the
    instance-level sets are the contract.
    """
    py_inst, rs_inst = _pair(kind, tier)
    py_own = {n for n in dir(py_inst) if not n.startswith("__")}
    rs_own = {n for n in dir(rs_inst) if not n.startswith("__")}
    assert py_own == rs_own


@pytest.mark.parametrize(("kind", "tier"), PAIRS)
def test_no_model_name_is_unreachable_through_the_wrapper(kind, tier):
    """Every public name on the plate must still answer through the wrapper, and every one the
    wrapper does not deliberately own must answer *with the plate's own value*."""
    py_inst, rs_inst = _pair(kind, tier)
    own = {n for n in dir(rs_inst) if not n.startswith("__")}
    plate = rs_inst.plate
    for name in dir(plate):
        if name.startswith("__"):
            continue
        # `dir()` of a `#[pyclass]` lists every getter, including the branch-only ones that raise
        # `AttributeError` -- a supported plate has no `K`, `W` or `w`. The Python plate simply
        # never assigned them, so `dir()` did not list them and this loop never saw them. Ask the
        # plate rather than its class: the claim is about names that ANSWER.
        if not hasattr(plate, name):
            continue
        assert hasattr(rs_inst, name), f"{name} is unreachable through the wrapper"
        if name in own:
            continue  # a deliberate override -- the test above pins that set
        mine, theirs = getattr(rs_inst, name), getattr(plate, name)
        if mine is theirs:
            continue  # delegation hands back the model's own object, which is the usual case
        if callable(theirs):
            assert getattr(mine, "__self__", None) is plate, name
        elif isinstance(theirs, np.ndarray):
            assert np.array_equal(mine, theirs), name
        elif sparse.issparse(theirs):
            assert _csr_equal(mine, theirs), name
        elif theirs is not getattr(plate, name):
            # A `#[pyclass]` getter may build a fresh object per access: `Plate._lu` hands back a
            # new `SparseLu` wrapper every time, so `plate._lu is plate._lu` is already False and
            # there is no value here for the wrapper to agree with. The Python plate stored one
            # object and this branch never fired. Delegation is still asserted -- the `hasattr`
            # above is the claim that the name answers at all.
            continue
        else:
            assert mine == theirs, name


def test_the_body_wrappers_surface_matches_the_reference():
    py_inst = RoomLoadedBodyPy(body=_body(), room=make_surface_room(), at=_AT, radius=None)
    with _rust_collaborators():
        rs_inst = physsynth_rs.RoomLoadedBody(
            body=_body(), room=make_surface_room(), at=_AT, radius=None
        )
    py_own = {n for n in dir(py_inst) if not n.startswith("__")}
    rs_own = {n for n in dir(rs_inst) if not n.startswith("__")}
    assert py_own == rs_own


@pytest.mark.parametrize("kind", ["linear", "vk"])
def test_the_seams_surface_matches_the_reference(kind):
    plate = _plate() if kind == "linear" else _vk_plate()
    py_seam = (_PlateSurfacePy if kind == "linear" else _VKPlateSurfacePy)(plate)
    rs_seam = (
        physsynth_rs._PlateSurface if kind == "linear" else physsynth_rs._VKPlateSurface
    )(plate)
    py_own = {n for n in dir(py_seam) if not n.startswith("__")}
    rs_own = {n for n in dir(rs_seam) if not n.startswith("__")}
    assert py_own == rs_own


def test_the_bridge_calls_step_by_keyword():
    """``connection.py`` writes ``body.step(force=F)`` and ``plate.step(f_ext=...)``, so the
    keyword names are part of the interface and not an implementation detail of the signature."""
    plate = _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
        body_inst = physsynth_rs.RoomLoadedBody(
            body=_body(), room=make_surface_room(), at=_AT, radius=None
        )
    inst.set_state(plate_bump(plate))
    inst.step(f_ext=_drive(plate))
    body_inst.step(force=0.5)
    assert inst.n == 1
    assert body_inst.n == 1


def test_the_overrides_are_not_delegated():
    """``energy``, ``n``, ``k`` and ``u`` are the four that must NOT come from the plate — the
    first because the delegated number is the total without its coupling channel, and the others
    because the wrapper keeps its own step count while ``u`` must still be the plate's."""
    plate = _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
    inst.set_state(plate_bump(plate))
    for _ in range(5):
        inst.step()
        inst.room.step()
    assert inst.n == 5
    assert plate.n == 5
    assert inst.energy() == plate.energy() + inst.radiated_energy
    assert inst.energy() != plate.energy()
    assert np.array_equal(inst.u, plate.u)  # delegated, not shadowed


def test_the_von_karman_wrapper_has_no_pressure():
    """Model #6 has none, and the absence is load-bearing: a ``pressure()`` added here would give
    ``StringVKPlateBridge`` a compact monopole read-out measured at 3e-7 of the truth."""
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedVKPlate(
            plate=_vk_plate(), room=make_surface_room(), face="z0"
        )
    with pytest.raises(AttributeError):
        _ = inst.pressure


# -- 6. the body wrapper --------------------------------------------------------------------

_AT = (0.15, 0.15, 0.15)


def _body():
    return ModalBody(
        freqs=np.array([180.0, 430.0]),
        fs=AIRBOX_SURFACE_FS,
        masses=np.array([0.02, 0.03]),
        sigmas=0.0,
        phi=np.array([1.0, -0.7]),
        radiation=np.array([1.3e-4, 0.9e-4]),
    )


@pytest.mark.parametrize("radius", [None, 0.09])
def test_the_body_wrapper_is_bit_identical(radius):
    """The lumped tier: one division, no factorization, and a rank-1 correction whose two
    precomputes are the only arithmetic this class does at construction."""
    py_body, rs_body = _body(), _body()
    py_inst = RoomLoadedBodyPy(
        body=py_body, room=make_surface_room(), at=_AT, radius=radius
    )
    with _rust_collaborators():
        rs_inst = physsynth_rs.RoomLoadedBody(
            body=rs_body, room=make_surface_room(), at=_AT, radius=radius
        )
    assert py_inst._G == rs_inst._G
    assert np.array_equal(py_inst._corr, rs_inst._corr)

    q0 = np.array([2e-3, -5e-4])
    py_inst.set_state(q0)
    rs_inst.set_state(q0)
    for n in range(200):
        py_inst.step()
        py_inst.room.step()
        rs_inst.step()
        rs_inst.room.step()
        assert np.array_equal(py_body.q, rs_body.q), f"q at step {n}"
        assert np.array_equal(py_body._accel, rs_body._accel), f"accel at step {n}"
    assert np.array_equal(py_inst.room.p, rs_inst.room.p)
    assert py_inst.radiated_energy == rs_inst.radiated_energy
    assert py_inst.volume_velocity == rs_inst.volume_velocity
    assert py_inst.port_pressure == rs_inst.port_pressure
    assert py_inst.energy() == rs_inst.energy()
    assert py_inst.pressure() == rs_inst.pressure()


def test_the_body_wrapper_takes_a_bridge_force():
    """It is a drop-in body for ``StringBodyBridge``, which owns ``body.step(force)``."""
    py_body, rs_body = _body(), _body()
    py_inst = RoomLoadedBodyPy(body=py_body, room=make_surface_room(), at=_AT, radius=None)
    with _rust_collaborators():
        rs_inst = physsynth_rs.RoomLoadedBody(
            body=rs_body, room=make_surface_room(), at=_AT, radius=None
        )
    for _ in range(50):
        py_inst.step(0.7)
        py_inst.room.step()
        rs_inst.step(0.7)
        rs_inst.room.step()
    assert np.array_equal(py_body.q, rs_body.q)
    assert py_inst.radiated_energy == rs_inst.radiated_energy


# -- 7. refusals and argument shapes --------------------------------------------------------


def test_a_sample_rate_mismatch_is_refused_with_the_same_message():
    plate = Plate(
        Lx=AIRBOX_SURFACE_PLATE_L,
        Ly=AIRBOX_SURFACE_PLATE_L,
        kappa=AIRBOX_SURFACE_KAPPA,
        rho=AIRBOX_SURFACE_RHO,
        fs=AIRBOX_SURFACE_FS * 2.0,
        N=N_PLATE,
        boundary="supported",
    )
    room = make_surface_room()
    with pytest.raises(ValueError) as py_err:
        RoomLoadedPlatePy(plate=plate, room=room, face="z0")
    with _rust_collaborators(), pytest.raises(ValueError) as rs_err:
        physsynth_rs.RoomLoadedPlate(plate=plate, room=make_surface_room(), face="z0")
    assert str(py_err.value) == str(rs_err.value)


def test_a_body_rate_mismatch_names_the_body():
    body = ModalBody(
        freqs=np.array([180.0]),
        fs=AIRBOX_SURFACE_FS * 2.0,
        masses=np.array([0.02]),
        sigmas=0.0,
        phi=np.array([1.0]),
        radiation=np.array([1.3e-4]),
    )
    with pytest.raises(ValueError) as py_err:
        RoomLoadedBodyPy(body=body, room=make_surface_room(), at=_AT, radius=None)
    with _rust_collaborators(), pytest.raises(ValueError) as rs_err:
        physsynth_rs.RoomLoadedBody(
            body=body, room=make_surface_room(), at=_AT, radius=None
        )
    assert str(py_err.value) == str(rs_err.value)
    assert "body fs" in str(rs_err.value)


def test_an_omitted_spreading_and_an_explicit_none_are_different_arguments():
    """§24.7's arm order, inherited one tier up: the wrapper's default is ``"bilinear"``, so an
    omitted argument builds a bilinear port and an explicit ``None`` must reach the port's own
    refusal rather than being quietly replaced by the default."""
    with _rust_collaborators():
        ok = physsynth_rs.RoomLoadedPlate(
            plate=_plate(), room=make_surface_room(), face="z0"
        )
        assert ok.port.spreading == "bilinear"
        with pytest.raises(ValueError, match="unknown spreading"):
            physsynth_rs.RoomLoadedPlate(
                plate=_plate(), room=make_surface_room(), face="z0", spreading=None
            )


def test_an_explicit_none_velocity_does_whatever_the_model_does():
    """The wrapper must not add an argument convention of its own — and this test had one wrong.

    It was written as "``set_state(u0, None)`` is a different call and the model refuses it",
    §24.7's rule applied to ``v0``. It passed because the *Python* plate refused: its default was
    the float ``0.0``, so ``None`` reached ``np.asarray(None, float)`` and NumPy raised. Unit 5's
    deletion made ``_plate()`` build the Rust class and showed that the model does **not** refuse
    — ``velocity_arg`` maps an explicit ``None`` onto the same zero vector as an omitted argument,
    and so does every other model in the crate.

    That is a divergence from the deleted Python, and it is the kind ``plate.rs``'s header already
    lists and declines to hide: the Python refusal was NumPy's accident rather than a designed
    one, unlike ``boundary=None``, which the original rejected in its own words and where §24.7's
    ``Option<Option<_>>`` is therefore load-bearing. So the claim that is actually worth making
    here is the wrapper's, not the model's: whatever the model does with an explicit ``None``, the
    wrapper does the same thing, because a wrapper that quietly disagreed with the object it wraps
    would be a second convention for callers to learn.
    """
    plate, twin = _plate(), _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
    u0 = plate_bump(plate)
    twin.set_state(u0, None)
    inst.set_state(u0, None)
    assert np.array_equal(plate.state, twin.state), (
        "the wrapper's `set_state` disagrees with the model's about an explicit `None` velocity"
    )
    # ... and it is the same thing an omitted argument does, which is what makes it one convention
    # rather than two.
    third = _plate()
    third.set_state(u0)
    assert np.array_equal(twin.state, third.state)


@pytest.mark.parametrize("tier", ["baffled", "suspended"])
def test_reset_clears_the_ledgers_and_the_ports_pending_mark(tier):
    py_inst, rs_inst = _pair("linear", tier)
    u0 = plate_bump(py_inst.plate)
    py_inst.set_state(u0)
    rs_inst.set_state(u0)
    for _ in range(10):
        py_inst.step()
        py_inst.room.step()
        rs_inst.step()
        rs_inst.room.step()
    py_inst.reset()
    rs_inst.reset()
    assert py_inst.n == rs_inst.n == 0
    assert py_inst.radiated_energy == rs_inst.radiated_energy == 0.0
    assert py_inst.volume_velocity == rs_inst.volume_velocity == 0.0
    assert np.array_equal(py_inst.plate.u, rs_inst.plate.u)
    assert np.array_equal(
        py_inst.nodal_volume_velocity, rs_inst.nodal_volume_velocity
    )
    assert np.array_equal(
        getattr(py_inst, LEDGER[tier]), getattr(rs_inst, LEDGER[tier])
    )


def test_the_room_is_not_stepped_by_the_wrapper():
    """Deliberate, and the reason a scene can hold two instruments: forgetting the room's own step
    must raise from the port rather than silently freeze it."""
    plate = _plate()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedPlate(
            plate=plate, room=make_surface_room(), face="z0"
        )
    inst.set_state(plate_bump(plate))
    inst.step()
    with pytest.raises(RuntimeError):
        inst.step()  # the room has not advanced, so the port still holds a pending injection
