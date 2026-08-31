"""Rust parity for ``physsynth.core.airbox``'s **membrane tier** — plan §33.

``RoomLoadedMembrane``, ``RoomSuspendedMembrane`` and the seam they drive (``_MembraneSurface``).
This is the last slice of ``airbox.py``: §30 ported the room, §31 the ports, §32 the plate and body
wrappers, and this batch the pair that was left.

**No stepping arithmetic is added by this batch**, which is the fact that shapes the file.
``RoomLoadedMembrane.step`` and ``RoomLoadedPlate.step`` are the same eleven lines over a different
seam, so the Rust side reuses §32's ``Wrap`` unchanged and the two tiers differ only in which seam
``build`` is told to construct. What is genuinely new is therefore small and worth naming:

* **the seam**, whose ``a_bare`` is ``(1 + sigma k) I`` and whose ``commit`` is a two-level roll
  with no acceleration cache; and
* **the seam's ``f_ext`` term**, which is the one piece of arithmetic in either batch with nothing
  in the model to be bit-identical *to* — ``Membrane.step()`` takes no force at all. A parity run
  that only ever passes ``f_ext=None`` compares the shared half twice and never reaches it, and
  passes: §23.6's emptied comparison through an eighth door. Every trajectory test here is
  therefore parametrized over ``forced``, and the seam's own right-hand-side test drives it
  directly.

Two things this file pins that §32's does not, both found by asking §30.3's question — what does a
client *do* to the object, rather than what does it read:

* ``inst.n += 1``. ``test_airbox_membrane.py``'s lagged-velocity negative control drives the seam
  by hand and advances the wrapper's own step count. A ``#[getter]`` with no ``#[setter]`` is a
  data descriptor whose ``__set__`` raises, so porting a class silently turns a plain Python
  attribute into a read-only one — the mirror image of §32.6, where a getter silently *takes* a
  name from ``__getattr__``. Neither is visible to ``cargo test`` or to any physics bar.
* ``inst.membrane``. The wrapper's getter name, the label ``_require_same_rate`` puts in its
  message and the name ``__getattr__`` refuses to delegate are three spellings of one decision. Get
  the third wrong and the wrapper loses its own model: the miss falls through to a delegation the
  membrane itself cannot answer. :func:`test_the_wrapper_answers_for_its_own_model` is that one
  assertion.

And one absence: **there is no ``pressure()``** on either arm. Model #4 caches no acceleration, so
it has no monopole read-out, and the reference's mixin says so in as many words.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from helpers import (
    AIRBOX_MEMBRANE_FS,
    AIRBOX_MEMBRANE_INDEX,
    AIRBOX_MEMBRANE_L,
    AIRBOX_MEMBRANE_RHO,
    AIRBOX_MEMBRANE_T,
    make_air_membrane,
    make_membrane_room,
    membrane_bulge,
    membrane_bump,
)
from scipy import sparse
from scipy.sparse.linalg import splu as scipy_splu

from physsynth.core import airbox
from physsynth.core.airbox import (
    RoomLoadedMembranePy,
    RoomSuspendedMembranePy,
    _MembraneSurfacePy,
)
from physsynth.core.membrane import Membrane

# NOT a bare `import physsynth_rs`: the default gate does not build the extension, so a module-scope
# import is a collection error there rather than a skip.
physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

DOMAINS = ("rectangle", "circle")
TIERS = ("baffled", "suspended")
N_MEMBRANE = 12
STEPS = 60


# -- fixtures -------------------------------------------------------------------------------


def _membrane(domain="rectangle", *, sigma=0.0, N=N_MEMBRANE, **kw):
    return make_air_membrane(domain=domain, N=N, sigma=sigma, **kw)


def _drive(membrane, amplitude=2.0):
    """A one-node external force — the driving-point injection the ``f_ext`` path exists for."""
    f = np.zeros(membrane.n_live)
    f[membrane.n_live // 3] = amplitude
    return f


@contextlib.contextmanager
def _rust_collaborators():
    """Point ``airbox``'s module globals at the Rust seam and ports for the duration.

    §28.4 in the mirror, and §32's helper with the membrane seam added. A Rust wrapper reads
    ``_MembraneSurface``, ``SurfacePort`` and ``splu`` as module globals because that is what the
    reference does, so in the default gate a Rust wrapper built without this patch holds **Python**
    collaborators. Both comparisons are useful; neither should be made by accident.
    """
    names = {
        "_MembraneSurface": physsynth_rs._MembraneSurface,
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


PY_CLASSES = {"baffled": RoomLoadedMembranePy, "suspended": RoomSuspendedMembranePy}
RS_CLASSES = {
    "baffled": physsynth_rs.RoomLoadedMembrane,
    "suspended": physsynth_rs.RoomSuspendedMembrane,
}
LEDGER = {"baffled": "surface_pressure", "suspended": "pressure_jump"}


def _make(cls, membrane, room, tier, **kw):
    if tier == "baffled":
        return cls(membrane=membrane, room=room, face="z0", **kw)
    return cls(membrane=membrane, room=room, plane="z", index=AIRBOX_MEMBRANE_INDEX, **kw)


def _pair(tier="baffled", domain="rectangle", *, head=None, **kw):
    """One Python wrapper and one Rust wrapper, over identical but **independent** rooms.

    Sharing one room would put two ports on the same nodes and be refused, so each side gets its
    own — identical by construction, because :func:`make_membrane_room` is deterministic in its
    arguments and both sides take its defaults. ``head`` is forwarded to the membrane, ``kw`` to
    the wrapper.
    """
    head = dict(head or {})
    py_m, rs_m = _membrane(domain, **head), _membrane(domain, **head)
    py_inst = _make(PY_CLASSES[tier], py_m, make_membrane_room(), tier, **kw)
    with _rust_collaborators():
        rs_inst = _make(RS_CLASSES[tier], rs_m, make_membrane_room(), tier, **kw)
    return py_inst, rs_inst


def _seeded(tier="baffled", domain="rectangle", shape=membrane_bulge, **kw):
    py_inst, rs_inst = _pair(tier, domain, **kw)
    py_inst.set_state(shape(py_inst.membrane))
    rs_inst.set_state(shape(rs_inst.membrane))
    return py_inst, rs_inst


def _run(py_inst, rs_inst, steps=STEPS, f_ext=None, tier="baffled"):
    """Step both sides in the contract's order and assert the state at every step."""
    for n in range(steps):
        py_inst.step(f_ext)
        py_inst.room.step()
        rs_inst.step(f_ext)
        rs_inst.room.step()
        assert np.array_equal(py_inst.membrane.u, rs_inst.membrane.u), f"u at step {n}"
    assert np.array_equal(py_inst.membrane.u_prev, rs_inst.membrane.u_prev)
    assert np.array_equal(py_inst.room.p, rs_inst.room.p)
    assert np.array_equal(py_inst.nodal_volume_velocity, rs_inst.nodal_volume_velocity)
    assert np.array_equal(getattr(py_inst, LEDGER[tier]), getattr(rs_inst, LEDGER[tier]))
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


# -- 1. the seam ----------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_seam_reports_the_same_surface(domain):
    """``(coords, areas)`` — the live nodes only, in the model's own C-order over ``mask``."""
    m = _membrane(domain)
    py_coords, py_areas = _MembraneSurfacePy(m).surface()
    rs_coords, rs_areas = physsynth_rs._MembraneSurface(m).surface()
    assert np.array_equal(py_coords, rs_coords)
    assert np.array_equal(py_areas, rs_areas)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("sigma", [0.0, 3.0])
def test_the_seams_unloaded_matrix_is_bit_identical(domain, sigma):
    """``(1 + sigma k) I``, values **and** stored order — §26.2's two questions, asked of the
    emptiest matrix in the project."""
    m = _membrane(domain, sigma=sigma)
    assert _csr_equal(_MembraneSurfacePy(m).a_bare(), physsynth_rs._MembraneSurface(m).a_bare())


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("forced", [False, True])
def test_the_seams_right_hand_side_is_bit_identical(domain, forced):
    """``2 u - (1 - sigma k) u_prev + c^2 k^2 L u``, plus the ``f_ext`` term.

    ``forced=True`` is the arm that matters: the model has no ``f_ext`` path of its own, so this is
    the only comparison that reaches it (module docstring).
    """
    m = _membrane(domain, sigma=2.0)
    m.set_state(membrane_bulge(m))
    for _ in range(3):
        m.step()
    f_ext = _drive(m) if forced else None
    py_rhs = _MembraneSurfacePy(m).rhs(f_ext)
    rs_rhs = physsynth_rs._MembraneSurface(m).rhs(f_ext)
    assert np.array_equal(py_rhs, rs_rhs)
    if forced:
        # ...and the forced arm is genuinely a different vector, so the parametrization is not
        # two spellings of one comparison.
        assert not np.array_equal(py_rhs, _MembraneSurfacePy(m).rhs(None))


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_seam_commits_the_same_state(domain):
    """The two-level roll, and **no** ``_accel``: model #4 caches no acceleration, so a seam that
    grew one would be inventing a read-out the model does not have."""
    py_m, rs_m = _membrane(domain), _membrane(domain)
    for m in (py_m, rs_m):
        m.set_state(membrane_bump(m))
    u_next = np.linspace(-1e-4, 1e-4, py_m.n_live)
    _MembraneSurfacePy(py_m).commit(u_next.copy())
    physsynth_rs._MembraneSurface(rs_m).commit(u_next.copy())
    assert np.array_equal(py_m.u, rs_m.u)
    assert np.array_equal(py_m.u_prev, rs_m.u_prev)
    assert py_m.n == rs_m.n == 1
    assert not hasattr(rs_m, "_accel")


def test_the_seams_two_associations_are_pinned_by_a_witness():
    """The seam's two scalar folds, pinned by a **searched** witness rather than a chosen constant.

    §26.6: a spelling pin that asserts three hand-picked numbers can land in the agreeing majority
    and go red, or land there and assert nothing. Both folds here are left-to-right in the
    reference — ``(rho * h) * h`` and ``((c * c) * k) * k`` — and the tidier spellings a cleanup
    would reach for (``rho * (h * h)``, ``(c * k) * (c * k)``) are different doubles often enough
    that a walk of a few hundred neighbouring values finds a witness. If it does not, this test
    fails saying so, which is the only way to tell "no difference exists" from "I did not look"
    (§23.5).
    """
    h = AIRBOX_MEMBRANE_L / N_MEMBRANE
    rho = next(
        (r for r in _walk(AIRBOX_MEMBRANE_RHO, 500) if (r * h) * h != r * (h * h)), None
    )
    assert rho is not None, "no witness for the denominator's association -- search wider"
    m = _membrane(rho=rho)
    assert m.h == h
    seam = physsynth_rs._MembraneSurface(m)
    assert seam.denominator == (rho * h) * h
    assert seam.denominator != rho * (h * h)
    assert seam.denominator == _MembraneSurfacePy(m).denominator

    k = 1.0 / AIRBOX_MEMBRANE_FS

    def _folds(t):
        c = float(np.sqrt(t / AIRBOX_MEMBRANE_RHO))
        return ((c * c) * k) * k, (c * k) * (c * k)

    tension = next(
        (t for t in _walk(AIRBOX_MEMBRANE_T, 500) if _folds(t)[0] != _folds(t)[1]), None
    )
    assert tension is not None, "no witness for c^2 k^2's association -- search wider"
    m = _membrane(T=tension)
    assert m.k == k
    m.set_state(membrane_bulge(m))
    m.step()
    rhs = physsynth_rs._MembraneSurface(m).rhs(None)
    assert np.array_equal(rhs, _MembraneSurfacePy(m).rhs(None))
    # The alternative association, spelled the way a tidy-up would spell it. The scalars differ by
    # construction above; what this asserts is that the difference survives into the vector.
    sk = m.sigma * m.k
    alt = 2.0 * m.u - (1.0 - sk) * m.u_prev + _folds(tension)[1] * (m.L @ m.u)
    assert not np.array_equal(rhs, alt), "the associations agree here -- the pin asserts nothing"


def _walk(x, n):
    """``x`` and its ``n`` nearest neighbours above, as a search space for a spelling witness."""
    for _ in range(n):
        yield x
        x = float(np.nextafter(x, np.inf))


# -- 2. construction ------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("tier", TIERS)
def test_construction_is_bit_identical(tier, domain):
    """Every number the constructor derives, including the two fill reports it publishes."""
    py_inst, rs_inst = _pair(tier, domain)
    assert py_inst.k == rs_inst.k
    assert py_inst._denominator == rs_inst._denominator
    assert py_inst._load_scale == rs_inst._load_scale
    assert py_inst.nnz_growth == rs_inst.nnz_growth
    assert py_inst.lu_nnz == rs_inst.lu_nnz
    assert py_inst.n == rs_inst.n == 0
    assert py_inst.radiated_energy == rs_inst.radiated_energy == 0.0
    assert np.array_equal(py_inst.nodal_volume_velocity, rs_inst.nodal_volume_velocity)
    assert np.array_equal(getattr(py_inst, LEDGER[tier]), getattr(rs_inst, LEDGER[tier]))
    assert _csr_equal(py_inst.port.load_matrix, rs_inst.port.load_matrix)


def test_the_wrapper_reads_splu_as_a_module_global():
    """The factorization is whatever ``airbox.splu`` names at construction — §32's finding, which
    is why this tier is exact and not a Group D one: the wrapper *calls* a solver, it does not own
    one, and a solver group is a property of ownership."""
    calls = []
    saved = airbox.splu

    def counting_splu(a):
        calls.append(a.shape)
        return scipy_splu(a)

    airbox.splu = counting_splu
    try:
        with _rust_collaborators():
            physsynth_rs.RoomLoadedMembrane(
                membrane=_membrane(), room=make_membrane_room(), face="z0"
            )
    finally:
        airbox.splu = saved
    assert len(calls) == 1


def test_a_rust_wrapper_built_without_the_patch_holds_python_collaborators():
    """Without :func:`_rust_collaborators` the module globals are the reference's, and a Rust
    wrapper honours them — which is what makes the comparison above a real one rather than Rust
    against Rust (§23.6)."""
    inst = physsynth_rs.RoomLoadedMembrane(
        membrane=_membrane(), room=make_membrane_room(), face="z0"
    )
    assert type(inst._surface) is _MembraneSurfacePy
    assert type(inst.port) is airbox.SurfacePortPy


# -- 3. the trajectory ----------------------------------------------------------------------


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("tier", TIERS)
def test_a_driven_wrapper_is_bit_identical(tier, domain, forced):
    """Sixty coupled steps, state and every ledger, on both tiers and both head shapes.

    Exact for §32's reason: the sparse products, the assembly, the factorization and ``np.dot`` are
    the same calls on the same objects in both languages, and everything between them is
    elementwise, where ``+ - * /`` admits no reassociation.
    """
    py_inst, rs_inst = _seeded(tier, domain)
    f_ext = _drive(py_inst.membrane) if forced else None
    _run(py_inst, rs_inst, f_ext=f_ext, tier=tier)


@pytest.mark.parametrize("tier", TIERS)
def test_a_lossy_head_is_bit_identical(tier):
    """``sigma > 0`` puts the loss factor in ``a_bare`` and in the right-hand side at once."""
    py_inst, rs_inst = _seeded(tier, shape=membrane_bump, head={"sigma": 4.0})
    _run(py_inst, rs_inst, tier=tier)


def test_a_zero_area_head_reduces_to_the_bare_membrane_in_rust_too():
    """``T = 0`` must give **bit-identical** state to a bare :class:`Membrane`, not merely close.

    The reference's own falsifiable end (``test_zero_area_reduces_to_the_bare_membrane``), asked of
    the Rust wrapper: with the load structurally present and numerically zero, ``eliminate_zeros``
    must leave ``(1 + sigma k) I`` and the solve must reproduce the model's explicit division
    exactly.
    """
    m = _membrane(sigma=3.0)
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedMembrane(
            membrane=m, room=make_membrane_room(), face="z0"
        )
    inst.port.areas = np.zeros_like(inst.port.areas)
    inst.port.T = sparse.csr_matrix(inst.port.T.shape)
    inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
    a = ((1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")).tocsc()
    a.eliminate_zeros()
    inst._lu_loaded = airbox.splu(a)

    bare = _membrane(sigma=3.0)
    u0 = membrane_bump(bare)
    inst.set_state(u0)
    bare.set_state(u0)
    for _ in range(50):
        inst.step()
        inst.room.step()
        bare.step()
    assert np.array_equal(inst.membrane.u, bare.u)


# -- 4. the substitutions the reference's tests make ----------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_a_zeroed_operator_reaches_the_rust_wrapper(tier):
    """``inst.port.T = 0`` must be the matrix the wrapper actually multiplies by.

    A wrapper that cached ``T`` at construction would keep the real one and this test — and the
    reference's ``_unload`` rig built on it — would pass having compared a loaded head with itself
    (§23.6, §31, §32).
    """
    py_inst, rs_inst = _seeded(tier)
    for inst in (py_inst, rs_inst):
        inst.port.T = sparse.csr_matrix(inst.port.T.shape)
        inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
        m = inst.membrane
        a = ((1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")).tocsc()
        inst._lu_loaded = airbox.splu(a)
    _run(py_inst, rs_inst, steps=25, tier=tier)
    assert rs_inst.volume_velocity == 0.0
    assert rs_inst.radiated_energy == 0.0


def test_a_replaced_factorization_is_the_one_the_rust_wrapper_solves_with():
    """Three of the reference's tests refactor a rescaled load and hand the wrapper the result, so
    ``_lu_loaded`` is settable and is read live."""
    py_inst, rs_inst = _seeded()
    for inst in (py_inst, rs_inst):
        a = inst._surface.a_bare()
        a = (a + 0.5 * inst._load_scale * inst.port.load_matrix).tocsc()
        a.eliminate_zeros()
        inst._lu_loaded = airbox.splu(a)
    _run(py_inst, rs_inst, steps=25)


def test_a_halved_load_matrix_reaches_the_rust_wrapper():
    """``port.R`` and ``port.load_matrix`` replaced together — the reference's half-coupling
    control, which only means anything if both reach the step."""
    py_inst, rs_inst = _seeded()
    for inst in (py_inst, rs_inst):
        inst.port.R = 0.5 * inst.port.R
        inst.port.load_matrix = 0.5 * inst.port.load_matrix
    _run(py_inst, rs_inst, steps=25)


def test_the_wrappers_step_count_is_settable():
    """``inst.n += 1``, which the reference's lagged-velocity negative control does while driving
    the seam by hand.

    §30.3's question — what does a client *do* to the object — with the answer this batch found. A
    ``#[getter]`` with no ``#[setter]`` is a data descriptor whose ``__set__`` raises, so a plain
    Python attribute becomes read-only the instant it is ported, and nothing in ``cargo test`` or
    in any physics bar sees it.
    """
    py_inst, rs_inst = _seeded()
    for inst in (py_inst, rs_inst):
        inst.n += 1
        inst.radiated_energy += 1.5
    assert py_inst.n == rs_inst.n == 1
    assert py_inst.radiated_energy == rs_inst.radiated_energy == 1.5


def test_the_hand_driven_seam_loop_runs_against_a_rust_wrapper():
    """The reference's lagged-velocity control end to end: the wrapper's own ``step`` is bypassed
    and the seam, the port and the books are driven directly, on a Rust instance."""
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedMembrane(
            membrane=_membrane(), room=make_membrane_room(), face="z0"
        )
    m, port = inst.membrane, inst.port
    inst.set_state(membrane_bulge(m))
    lu = airbox.splu(((1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")).tocsc())
    for _ in range(20):
        port.require_ready()
        pbar_free = port.free_pressure()
        q = port.T @ ((m.u - m.u_prev) / m.k)
        pbar = pbar_free + port.R * q
        rhs = inst._surface.rhs(None) - m.k * m.k * (port.T.T @ pbar) / inst._denominator
        inst._surface.commit(lu.solve(rhs))
        port.inject(q)
        inst.radiated_energy += m.k * float(np.dot(pbar, q))
        inst.n += 1
        inst.room.step()
    assert inst.n == 20
    assert m.n == 20
    assert np.isfinite(m.u).all()


# -- 5. delegation --------------------------------------------------------------------------

# Every read accessor the reference's mixin promises to hand through. A `#[pyclass]` getter shadows
# delegation permanently, where Python's `__getattr__` only fires on a miss.
MEMBRANE_DELEGATED = (
    "u", "u_prev", "X", "Y", "mask", "L", "c", "h", "n_live", "domain", "sigma", "rho",
    "T", "fs", "lam", "state", "to_live", "pickup_index_at",
)


@pytest.mark.parametrize("name", MEMBRANE_DELEGATED)
def test_every_delegated_name_reaches_the_model(name):
    m = _membrane()
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedMembrane(
            membrane=m, room=make_membrane_room(), face="z0"
        )
    mine, theirs = getattr(inst, name), getattr(m, name)
    if isinstance(theirs, np.ndarray):
        assert np.array_equal(mine, theirs)
    elif sparse.issparse(theirs):
        assert _csr_equal(mine, theirs)
    elif callable(theirs):
        assert mine.__self__ is m
    else:
        assert mine == theirs


def test_the_wrapper_answers_for_its_own_model():
    """``inst.membrane`` is the head the wrapper holds — the batch's one structural hazard.

    The getter's name, ``_require_same_rate``'s label and the name ``__getattr__`` refuses to
    delegate are three spellings of one decision. Left as the plate tier's ``"plate"``, this class
    would expose ``.plate`` and ``inst.membrane`` would raise, because the miss falls through to a
    delegation the membrane itself cannot answer.
    """
    py_inst, rs_inst = _pair()
    assert rs_inst.membrane is not None
    assert type(rs_inst.membrane) is Membrane
    assert not hasattr(rs_inst, "plate")
    assert not hasattr(py_inst, "plate")
    # ...and the recursion guard: the model has no `.membrane` of its own to fall through to.
    with pytest.raises(AttributeError):
        _ = rs_inst.membrane.membrane


@pytest.mark.parametrize("tier", TIERS)
def test_the_wrappers_own_surface_matches_the_reference(tier):
    """DERIVED, not listed — §32.6's guard, which fires on the next getter added to either class
    whether or not anybody thought to list the name it shadows."""
    py_inst, rs_inst = _pair(tier)
    py_own = {n for n in dir(py_inst) if not n.startswith("__")}
    rs_own = {n for n in dir(rs_inst) if not n.startswith("__")}
    assert py_own == rs_own


@pytest.mark.parametrize("tier", TIERS)
def test_no_model_name_is_unreachable_through_the_wrapper(tier):
    """Every public name on the membrane must still answer through the wrapper, and every one the
    wrapper does not deliberately own must answer *with the membrane's own value*."""
    _, rs_inst = _pair(tier)
    own = {n for n in dir(rs_inst) if not n.startswith("__")}
    m = rs_inst.membrane
    for name in dir(m):
        if name.startswith("__"):
            continue
        assert hasattr(rs_inst, name), f"{name} is unreachable through the wrapper"
        if name in own:
            continue  # a deliberate override -- the test above pins that set
        mine, theirs = getattr(rs_inst, name), getattr(m, name)
        if mine is theirs:
            continue
        if callable(theirs):
            assert getattr(mine, "__self__", None) is m, name
        elif isinstance(theirs, np.ndarray):
            assert np.array_equal(mine, theirs), name
        elif sparse.issparse(theirs):
            assert _csr_equal(mine, theirs), name
        else:
            assert mine == theirs, name


def test_the_seams_surface_matches_the_reference():
    m = _membrane()
    py_own = {n for n in dir(_MembraneSurfacePy(m)) if not n.startswith("__")}
    rs_own = {n for n in dir(physsynth_rs._MembraneSurface(m)) if not n.startswith("__")}
    assert py_own == rs_own


@pytest.mark.parametrize("tier", TIERS)
def test_neither_membrane_wrapper_has_a_pressure(tier):
    """Model #4 caches no acceleration, so it has no monopole read-out and neither does a wrapper
    over it. The absence is the reference's, stated in the mixin's ``__getattr__`` comment."""
    _, rs_inst = _pair(tier)
    with pytest.raises(AttributeError):
        _ = rs_inst.pressure


def test_the_overrides_are_not_delegated():
    """``energy``, ``n`` and ``k`` must not come from the membrane — the first because the
    delegated number is the total *without* its coupling channel."""
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedMembrane(
            membrane=_membrane(), room=make_membrane_room(), face="z0"
        )
    m = inst.membrane
    inst.set_state(membrane_bulge(m))
    for _ in range(5):
        inst.step()
        inst.room.step()
    assert inst.n == 5
    assert m.n == 5
    assert inst.energy() == m.energy() + inst.radiated_energy
    assert inst.energy() != m.energy()
    assert np.array_equal(inst.u, m.u)  # delegated, not shadowed


# -- 6. refusals, argument shapes and the ledgers -------------------------------------------


def test_a_sample_rate_mismatch_names_the_membrane():
    """``_require_same_rate``'s label is the third spelling of the model's name, and the
    reference's tests match on the word ``membrane fs``."""
    room = make_membrane_room()
    slow = _membrane(fs=AIRBOX_MEMBRANE_FS / 2.0)
    with pytest.raises(ValueError, match="membrane fs"):
        physsynth_rs.RoomLoadedMembrane(membrane=slow, room=room, face="z0")
    with pytest.raises(ValueError, match="membrane fs") as py_err:
        RoomLoadedMembranePy(membrane=slow, room=room, face="z0")
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs.RoomSuspendedMembrane(
            membrane=slow, room=room, plane="z", index=AIRBOX_MEMBRANE_INDEX
        )
    assert str(py_err.value) == str(rs_err.value)


def test_an_omitted_spreading_and_an_explicit_none_are_different_arguments():
    """§31.7's arm order, which is §24.7's: PyO3 wraps the DEFAULT expression, so ``Some(None)`` is
    "argument omitted" and a bare ``None`` is what the caller wrote — and the port refuses the
    latter."""
    with _rust_collaborators():
        inst = physsynth_rs.RoomLoadedMembrane(
            membrane=_membrane(), room=make_membrane_room(), face="z0"
        )
        assert inst.port.spreading == "bilinear"
        with pytest.raises(ValueError):
            physsynth_rs.RoomLoadedMembrane(
                membrane=_membrane(),
                room=make_membrane_room(),
                face="z0",
                spreading=None,
            )


def test_an_explicit_none_velocity_is_not_the_default():
    """``set_state``'s ``v0`` defaults to ``0.0``, so an omitted argument and an explicit ``None``
    are not the same call (§24.7)."""
    py_inst, rs_inst = _pair()
    u0 = membrane_bulge(py_inst.membrane)
    py_err = rs_err = None
    try:
        py_inst.set_state(u0, None)
    except Exception as exc:  # noqa: BLE001 - the shape of the refusal is the assertion
        py_err = type(exc)
    try:
        rs_inst.set_state(u0, None)
    except Exception as exc:  # noqa: BLE001
        rs_err = type(exc)
    assert py_err is rs_err


@pytest.mark.parametrize("tier", TIERS)
def test_reset_clears_the_ledgers_and_the_ports_pending_mark(tier):
    py_inst, rs_inst = _seeded(tier)
    _run(py_inst, rs_inst, steps=10, tier=tier)
    for inst in (py_inst, rs_inst):
        inst.reset()
    assert rs_inst.n == py_inst.n == 0
    assert rs_inst.radiated_energy == py_inst.radiated_energy == 0.0
    assert rs_inst.volume_velocity == py_inst.volume_velocity == 0.0
    assert np.array_equal(py_inst.membrane.u, rs_inst.membrane.u)
    assert np.array_equal(py_inst.nodal_volume_velocity, rs_inst.nodal_volume_velocity)
    assert np.array_equal(getattr(py_inst, LEDGER[tier]), getattr(rs_inst, LEDGER[tier]))
    # The room may be stepped again immediately: `reset()` clears the port's pending mark.
    for inst in (py_inst, rs_inst):
        inst.step()
        inst.room.step()
    assert np.array_equal(py_inst.membrane.u, rs_inst.membrane.u)


@pytest.mark.parametrize("tier", TIERS)
def test_the_room_is_not_stepped_by_the_wrapper(tier):
    """The contract is one room, one step, after every port has solved — so a wrapper that stepped
    its own room would double-advance a shared one."""
    _, rs_inst = _seeded(tier)
    before = rs_inst.room.n
    rs_inst.step()
    assert rs_inst.room.n == before
# -- 7. the two module helpers that finish the file -----------------------------------------


@pytest.mark.parametrize("face", airbox.FACES)
def test_the_face_axis_table_is_the_same(face):
    """``_face_axes`` — pure integer arithmetic off a six-element table, so nothing here can round.

    It is in this batch because it was never *bound*, not because it was hard: both it and
    :func:`impedance_from_zeta` had Rust twins with native tests since section 31 and were simply
    never swapped, which left ``airbox.py`` one tier **and two functions** from finished rather
    than one tier.
    """
    assert tuple(airbox.face_axes_py(face)) == tuple(physsynth_rs._face_axes(face))


def test_the_face_refusal_is_word_for_word():
    """The one thing in ``_face_axes`` that can differ is the message, and a test matches on it."""
    with pytest.raises(ValueError) as py_err:
        airbox.face_axes_py("q0")
    with pytest.raises(ValueError) as rs_err:
        physsynth_rs._face_axes("q0")
    assert str(py_err.value) == str(rs_err.value)


@pytest.mark.parametrize("zeta", [0.0, 1.0, 4.0, 1e-7, 1e7, 0.3141592653589793])
def test_the_wall_impedance_helper_is_bit_identical(zeta):
    """``float(zeta) * rho0 * c0``, left-folded as Python folds it — ``(zeta rho0) c0``.

    Both the ambient default and an explicit pair, because the defaults are spelled in the binding
    rather than read off the module and a divergence there would be silent.
    """
    assert airbox.impedance_from_zeta_py(zeta) == physsynth_rs.impedance_from_zeta(zeta)
    assert airbox.impedance_from_zeta_py(
        zeta, rho0=1.19, c0=340.5
    ) == physsynth_rs.impedance_from_zeta(zeta, rho0=1.19, c0=340.5)
