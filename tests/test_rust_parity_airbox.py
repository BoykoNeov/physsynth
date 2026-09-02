"""Rust parity for ``physsynth.core.airbox.AirBox`` — the 3-D room (plan §30).

Only the class ported. The ports (:class:`RoomPort`, :class:`SurfacePort`,
:class:`InteriorSurfacePort`) and the six ``RoomLoaded*`` / ``RoomSuspended*`` wrappers above it
stay Python, so this file has two jobs rather than one:

* the usual one — the **field** is bit-identical, and it is, on every boundary type, with cuts,
  with injections and from an exact discrete mode;
* a new one — the **seam** holds. Everything above ``AirBox`` reaches into it through fourteen
  private names, six of which are containers a client *writes*, and one public one nobody's
  private-name grep would have found (``source_index``). Those are asserted here directly, because
  a wrapper that silently stopped seeing a room's ``_pending_ports`` would not fail loudly — it
  would inject nothing, and every energy ledger would stay perfectly green.

**What was not bit-identical, and is now.** Four reductions: ``acoustic_energy``'s volume and
three kinetic sums, ``step``'s per-face wall book, and the port injection's ``pbar``. All are
``np.sum``, which is pairwise-blocked above **eight** elements and a plain left-to-right loop below
it. §30 declined to reproduce that blocking as a claim about a NumPy internal, affordably, because
``dissipated`` and ``injected`` are pure bookkeeping that no timestep reads (§14.2's question,
answered "no"); §31 then found the blocking is one fixed algorithm rather than a kernel and
transcribed it (``crate::reduce``) for the ports, whose sums *do* reach the update. The parked
tightening (§31.11) is taken as of 2026-09-02: the room's books go through the same reduction, so
the energy ledgers are asserted **equal** below, not within ``1e-13`` — a sharper detector of a
mis-transcribed booking, at no run-time cost.

The eight-element cutoff is still what decides where exactness is *structural* (free, whatever the
spelling) and where it has to be bought with ``reduce``; it is measured in
:func:`test_the_reduction_cutoff_is_eight_and_the_room_is_always_past_it`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from physsynth.core.airbox import (
    C0_AIR,
    RHO0_AIR,
    AirBoxPy,
    _free_pressure_nodes,
)

# NOT a bare `import physsynth_rs`: the default gate does not build the extension, so a module-scope
# import is a collection error there rather than a skip (see `test_rust_parity_plate.py`).
physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

H = 0.03
N_DEFAULT = (9, 7, 6)
CFL = 0.9

# The four wall configurations the class distinguishes, plus a mixed one. `rigid` takes the
# `_has_walls = False` branch; every other entry takes the closure.
WALLS = {
    "rigid": "rigid",
    "one lossy face": {"x0": 2.0 * RHO0_AIR * C0_AIR},
    "two lossy faces": {"x0": 2.0 * RHO0_AIR * C0_AIR, "y1": 0.5 * RHO0_AIR * C0_AIR},
    "all lossy": 1.5 * RHO0_AIR * C0_AIR,
    "an open face": {"y1": "open", "z0": 0.0},
}


def _kwargs(*, N=N_DEFAULT, h=H, walls="rigid", c0=C0_AIR, **extra):
    """A room sized in cells, at a fixed fraction of the 3-D CFL ceiling."""
    kw = dict(
        L=tuple(n * h for n in N),
        h=h,
        fs=c0 * np.sqrt(3.0) / (CFL * h),
        walls=walls,
        c0=c0,
    )
    kw.update(extra)
    return kw


def _pair(**kw):
    """One Python room and one Rust room, built from identical arguments."""
    return AirBoxPy(**kw), physsynth_rs.AirBox(**kw)


def _seed(a, b, seed=0, amplitude=1.0):
    """Give both rooms the same broadband initial field — the structural-test IC."""
    p0 = amplitude * np.random.default_rng(seed).standard_normal(a.p.shape)
    a.set_state(p0)
    b.set_state(p0)
    return a, b


def _fields(box):
    return (box.p, box.ux, box.uy, box.uz, box.ux_prev, box.uy_prev, box.uz_prev)


def _identical(a, b):
    return all(np.array_equal(x, y) for x, y in zip(_fields(a), _fields(b), strict=True))


# -- construction ---------------------------------------------------------------------------------


@pytest.mark.parametrize("wall_name", list(WALLS))
def test_everything_built_at_construction_is_bit_identical(wall_name):
    """The grid, the weights, the wall closure and the source node, all exact.

    The weights are the interesting half: ``_W`` is a triple product of trapezoid vectors and
    ``_beta`` folds ``k rho0 c0^2`` into a per-node admittance, so this is the first place a
    reassociation would show. Nothing here is a reduction, so exactness is structural rather than
    lucky.
    """
    a, b = _pair(**_kwargs(walls=WALLS[wall_name]))
    assert a.N == b.N
    assert a.L == b.L and a.L_actual == b.L_actual
    assert a.lam == b.lam and a.k == b.k
    assert a.walls == b.walls
    assert a.source_index == b.source_index
    for x, y in zip(a._w, b._w, strict=True):
        assert np.array_equal(x, y)
    assert np.array_equal(a._W, b._W)
    assert np.array_equal(a._Wx, b._Wx)
    assert np.array_equal(a._Wy, b._Wy)
    assert np.array_equal(a._Wz, b._Wz)
    assert np.array_equal(a._beta, b._beta)
    assert np.array_equal(a._open, b._open)
    assert a._has_walls == b._has_walls


def test_the_cfl_ceiling_is_the_same_double_on_both_sides():
    """``1/sqrt(3)`` spelled as a literal would put a fixture at the ceiling on one side only.

    ``lambda = 1/sqrt(3)`` is deliberately *allowed* (it is the textbook CFL, and the grid-diagonal
    mode is exact there), so a room built exactly at it has to construct on both sides — and one
    ulp past it has to be refused on both.
    """
    at = _kwargs(N=N_DEFAULT, h=H)
    at["fs"] = C0_AIR * np.sqrt(3.0) / H  # cfl = 1.0 exactly
    a, b = _pair(**at)
    assert a.lam == b.lam
    past = dict(at)
    past["fs"] = np.nextafter(at["fs"], 0.0) * (1.0 - 1e-9)
    with pytest.raises(ValueError, match="CFL violated"):
        AirBoxPy(**past)
    with pytest.raises(ValueError, match="CFL violated"):
        physsynth_rs.AirBox(**past)


def test_the_grid_count_rounds_half_to_even_where_rust_would_round_away():
    """``N = int(round(L/h))`` is a **decision**, not a number (§25.2).

    Python's ``round`` is half-to-even and Rust's ``f64::round`` is half-away-from-zero, so at an
    exact tie the naive port builds a room one cell larger on one axis — which conserves energy
    perfectly, reports a plausible spectrum and is simply a different room. The tie is searched for
    rather than asserted as a constant, so the test can tell "the two agree" from "no tie was
    reached" (§26.6).
    """
    ties = [
        (n + 0.5) * 0.01
        for n in range(1, 40)
        if (n + 0.5) * 0.01 / 0.01 - math.floor((n + 0.5) * 0.01 / 0.01) == 0.5
    ]
    assert ties, "no exact tie in the search range; the test would assert nothing"
    # The two modes disagree at a tie only when the value below it is EVEN: half-to-even keeps it
    # there, half-away-from-zero goes up. So roughly half the ties are witnesses and the other half
    # would pass under either mode -- counting them separately is what stops this test from being
    # satisfied by the wrong ties.
    witnesses = [v for v in ties if math.floor(v / 0.01) % 2 == 0]
    assert witnesses, "no tie with an even floor; the rounding mode is not being exercised"
    caught = 0
    for length in ties:
        kw = _kwargs(N=(1, 1, 1), h=0.01)
        kw["L"] = (length, length, length)
        a, b = _pair(**kw)
        assert a.N == b.N
        if b.N[0] != math.floor(length / 0.01 + 0.5):
            caught += 1
    assert caught == len(witnesses), (
        f"{caught} of {len(ties)} ties differ from half-away-from-zero, expected "
        f"{len(witnesses)} -- either the port rounds the wrong way or the search moved"
    )


def test_the_compliance_gain_is_a_pow_and_not_a_multiply():
    """``self.c0 ** 2`` is CPython's ``float.__pow__``, i.e. libm's ``pow`` — not ``c0 * c0``.

    They disagree in 99 of 200,000 sound speeds drawn from the range this class accepts, and the
    quantity multiplies the divergence at **every** timestep, so a multiply would put a last bit on
    the state of every step of every run while conserving energy perfectly. The witness is searched
    for, and the search is asserted to have found one (§26.6, and §17.2 for why the Rust side needs
    an opaque call to keep the two spellings apart in ``--release``).
    """
    rng = np.random.default_rng(7)
    witness = None
    for _ in range(200_000):
        c = float(rng.uniform(200.0, 1500.0))
        if c**2 != c * c:
            witness = c
            break
    assert witness is not None, "no c0 where pow and multiply differ; the test asserts nothing"
    a, b = _pair(**_kwargs(walls={"x0": 2.0 * RHO0_AIR * witness}, c0=witness))
    assert np.array_equal(a._beta, b._beta)
    _seed(a, b)
    for _ in range(500):
        a.step()
        b.step()
    assert _identical(a, b), "the port took the multiply spelling of c0 ** 2"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(L=(0.3, 0.24), fs=1e5, h=H), "must be a .Lx, Ly, Lz. triple"),
        (dict(L=(0.3, 0.24, -1.0), fs=1e5, h=H), "must all be positive"),
        (dict(L=(0.3, 0.24, 0.18), fs=1e5, h=1.0), "coarser than the room"),
        (dict(L=(0.3, 0.24, 0.18), fs=1e3, h=H), "CFL violated"),
        (dict(L=(0.3, 0.24, 0.18), fs=1e5, h=H, walls={"q0": "rigid"}), "unknown face name"),
        (dict(L=(0.3, 0.24, 0.18), fs=1e5, h=H, walls="squishy"), "unknown token"),
        (dict(L=(0.3, 0.24, 0.18), fs=1e5, h=H, walls=-1.0), "must be >= 0"),
        (dict(L=(0.3, 0.24, 0.18), fs=1e5, h=H, source=(9.0, 0.1, 0.1)), "outside the room"),
    ],
)
def test_the_rejections_carry_the_same_message(kwargs, match):
    """Every ported rejection reproduces the original's text, because the suite matches on it."""
    with pytest.raises(ValueError, match=match):
        AirBoxPy(**kwargs)
    with pytest.raises(ValueError, match=match):
        physsynth_rs.AirBox(**kwargs)


# -- the field ------------------------------------------------------------------------------------


@pytest.mark.parametrize("wall_name", list(WALLS))
def test_the_field_is_bit_identical_over_two_thousand_steps(wall_name):
    """``p``, ``ux``, ``uy``, ``uz`` and both stored half-steps, to ``0.0``.

    Every update in this model is elementwise — a divergence, a gradient, a 1x1 wall solve — so
    there is no reduction anywhere on the update path and exactness is structural. This is the
    claim the whole port rests on, because everything above ``AirBox`` reads these arrays.
    """
    a, b = _seed(*_pair(**_kwargs(walls=WALLS[wall_name])))
    for n in range(2000):
        a.step()
        b.step()
        if n % 250 == 0 or n == 1999:
            assert _identical(a, b), f"step {n}"
    assert a.n == b.n == 2000


def test_a_driven_room_is_bit_identical_and_the_scalar_book_is_exact():
    """A soft point source adds no reduction: its work is ``k pbar q`` at one node.

    So unlike the wall book and the port book, ``injected`` is expected **equal**, not close — and
    that is §23.2's question ("how long is the sum?") answered for a sum of length one.
    """
    a, b = _seed(*_pair(**_kwargs()))
    rng = np.random.default_rng(3)
    for _ in range(1000):
        q = float(rng.standard_normal()) * 1e-3
        a.inject(q)
        b.inject(q)
        a.step()
        b.step()
    assert _identical(a, b)
    assert a.injected_energy() == b.injected_energy()
    assert abs(a.injected_energy()) > 1e-12, "the source did nothing; the claim would be vacuous"


def test_a_cut_room_is_bit_identical_and_the_cut_bookkeeping_matches():
    """The cut is three pieces of state and a fancy index, and all three must survive the port.

    ``add_cut`` with an extent takes the restricted branch, a second cut on another axis takes the
    additive one, and ``cut_faces`` reads the mask rather than the index — so a port that kept only
    the index would pass every energy bar and report the wrong blocked area.
    """
    a, b = _pair(**_kwargs())
    for box in (a, b):
        box.add_cut("z", 3)
        box.add_cut("x", 2, extent=((1, 5), (1, 4)))
    assert a.cut_faces == b.cut_faces
    for m_a, m_b in zip(a._cut_mask, b._cut_mask, strict=True):
        assert (m_a is None) == (m_b is None)
        if m_a is not None:
            assert np.array_equal(m_a, m_b)
    for i_a, i_b in zip(a._cut_index, b._cut_index, strict=True):
        assert (i_a is None) == (i_b is None)
        if i_a is not None:
            assert all(np.array_equal(x, y) for x, y in zip(i_a, i_b, strict=True))
    _seed(a, b)
    for _ in range(1000):
        a.step()
        b.step()
    assert _identical(a, b)
    assert np.max(np.abs(b.uz[:, :, 3])) == 0.0, "the cut face carries velocity"


def test_two_cuts_sharing_a_face_are_refused_the_same_way():
    """A port's cut and its ``-q``/``+q`` pair are two halves of one object.

    The refusal is a *set intersection* on flat face indices, and the message quotes the first
    shared face — which is ``np.unravel_index``'s tuple, not an int tuple, so the port builds it
    through NumPy rather than formatting it itself.
    """
    for cls in (AirBoxPy, physsynth_rs.AirBox):
        box = cls(**_kwargs())
        box._register_cut(object(), 2, 3, np.arange(4), np.arange(4))
        with pytest.raises(ValueError, match="shares face"):
            box._register_cut(object(), 2, 3, np.arange(4), np.arange(4))


def test_the_exact_discrete_mode_is_bit_identical_and_so_is_its_frequency():
    """``set_mode`` seeds an initial **condition**, so a last bit here is a different run.

    The shape is a tensor cosine, and §22.1 established that ``np.cos`` is a claim about the
    runner's CPU rather than about this code. So ``mode_shape`` takes the portable spelling
    (``math.cos``) on the Python side, where CPython and Rust meet at the platform libm — §22.3's
    manoeuvre a sixth time, and free at ~60 calls per room. Refusing it would have made this test
    pass on Windows and fail on a Linux runner, which is exactly the shape of §22.1's red run.
    """
    for mode in [(1, 0, 0), (0, 1, 0), (2, 1, 1), (3, 2, 2)]:
        a, b = _pair(**_kwargs())
        assert np.array_equal(a.mode_shape(*mode), b.mode_shape(*mode))
        f_a = a.set_mode(*mode)
        f_b = b.set_mode(*mode)
        assert f_a == f_b
        assert a.mode_frequency(*mode) == b.mode_frequency(*mode)
        assert a.continuum_mode_frequency(*mode) == b.continuum_mode_frequency(*mode)
        assert _identical(a, b)
        for _ in range(400):
            a.step()
            b.step()
        assert _identical(a, b), f"mode {mode} diverged"


# -- the reductions -------------------------------------------------------------------------------


def test_the_reduction_cutoff_is_eight_and_the_room_is_always_past_it():
    """``np.sum`` is a plain left-to-right loop **below eight elements** and pairwise at or above.

    That number is what turns "does this agree?" into a question that costs no measurement, the way
    §23.2 turned it into "how long is the sum?" for a two-term one. The consequence for this model
    is sharp and is the reason the energy books had to be *bought* with ``reduce`` rather than
    being exact for free:

    * the smallest room the class can build is one cell per axis, i.e. **eight** pressure nodes, so
      ``acoustic_energy``'s volume sum is *never* below the cutoff. There is no room for which it is
      structurally exact.
    * a wall face carries ``(N+1)(N+1) >= 4`` nodes, so a degenerate 1x1-cell face **is** below it.
    * a one-node port books its work as a sum of length one, which is exact — asserted in
      :func:`test_the_port_seam_holds`.
    """
    rng = np.random.default_rng(0)
    rates = {}
    for n in (4, 6, 7, 8, 16, 560):
        differ = 0
        for _ in range(400):
            x = rng.standard_normal(n) ** 2 * rng.uniform(0.5, 2.0, n)
            left = 0.0
            for v in x:
                left += float(v)
            differ += left != float(np.sum(x))
        rates[n] = differ
    assert rates[4] == rates[6] == rates[7] == 0, (
        "np.sum is supposed to be a plain loop below eight elements; if it is not, the "
        "structural half of this port's exactness claim needs re-deriving"
    )
    assert rates[8] > 0 and rates[560] > rates[16] > 0, (
        "no disagreement above the cutoff: this test found no witness rather than proving agreement"
    )
    # And the model's own smallest room is already at the boundary.
    tiny = physsynth_rs.AirBox(**_kwargs(N=(1, 1, 1), h=0.1))
    assert tiny.p.size == 8


@pytest.mark.parametrize("wall_name", ["one lossy face", "two lossy faces", "all lossy"])
def test_the_energy_books_are_bit_identical(wall_name):
    """The four ``np.sum`` reductions were the whole of the difference between the two rooms.

    Until 2026-09-02 this asserted ``<= 1e-13`` relative and its name said "and no better". The
    room now books ``acoustic_energy``, the per-face wall flux and the port injection through
    ``crate::reduce`` — NumPy's pairwise blocking, transcribed — so the three ledgers are asserted
    *equal*. The field is bit-identical (asserted above), so a gap here now means a booking was
    transcribed wrongly, not that a sum was ordered differently: the test became a sharper
    detector, which is what the parked tightening was for.
    """
    a, b = _seed(*_pair(**_kwargs(walls=WALLS[wall_name])))
    for _ in range(2000):
        a.step()
        b.step()
    assert _identical(a, b), "the field moved; the energy gap below would not be about reductions"
    for name in ("acoustic_energy", "dissipated_energy", "energy"):
        x = getattr(a, name)()
        y = getattr(b, name)()
        assert x == y, f"{name}: {x!r} vs {y!r}"
    assert a.dissipated_energy() > 0.0


def test_both_rooms_meet_the_projects_own_energy_bar():
    """The port has to pass the acceptance contract, not merely match the reference.

    ``energy()`` is ``acoustic + dissipated - injected`` and is flat for any walls and any source;
    the bar is the standing 1e-10, deliberately loose against the ~1e-15 observed.
    """
    for wall_name, walls in WALLS.items():
        for box in _seed(*_pair(**_kwargs(walls=walls))):
            e0 = box.energy()
            for _ in range(1000):
                box.step()
            drift = abs(box.energy() - e0) / abs(e0)
            assert drift < 1e-10, f"{wall_name}: {type(box).__module__} drifted {drift:e}"


# -- the seam -------------------------------------------------------------------------------------


def test_the_private_surface_is_present_and_settable():
    """Fifteen names cross the seam, and six of them are **written** from outside.

    A wrapper that silently stopped seeing a room's ``_pending_ports`` would inject nothing and
    every ledger would stay green, so the writable half is exercised rather than merely read. The
    three cut fields are cleared exactly as ``tests/test_airbox_dipole.py::_uncut`` clears them.
    """
    box = physsynth_rs.AirBox(**_kwargs())
    for name in (
        "_w",
        "_W",
        "_Wx",
        "_Wy",
        "_Wz",
        "_beta",
        "_open",
        "_has_walls",
        "_pending",
        "_pending_ports",
        "_ports",
        "_cut_mask",
        "_cut_index",
        "_cuts",
        "_divergence",
        "_register_cut",
        "_plane_axis",
    ):
        assert hasattr(box, name), name
    # `_cut_mask` and `_cut_index` must be DISTINCT objects: building both from one list made
    # `_register_cut` overwrite each with the other, and `cut_faces` then reported 375 instead of
    # 136 (plan §30). Nothing else in the suite can see that.
    assert box._cut_mask is not box._cut_index
    box.add_cut("z", 3)
    assert box.cut_faces > 0
    box._cut_mask = [None, None, None]
    box._cut_index = [None, None, None]
    box._cuts = []
    assert box.cut_faces == 0
    # `source_index` is the one PUBLIC name across the seam, and `test_airbox_freefield.py`
    # assigns to it. The private-name grep that found the other fourteen could not see it.
    box.source_index = (1, 1, 1)
    assert box.source_index == (1, 1, 1)


def test_the_port_seam_holds():
    """A hand-built port injection, appended straight onto ``_pending_ports``.

    This is verbatim what ``RoomPort.inject`` does, and it is the shape
    ``tests/test_airbox_dipole.py`` uses to build its phantom dipole. Two claims: the resulting
    field matches the reference bit for bit, and a **one-node** port's booked work is *exactly*
    equal, because its ``pbar`` is a sum of length one.
    """
    a, b = _seed(*_pair(**_kwargs()))
    node = (np.array([4]), np.array([3]), np.array([2]))
    rng = np.random.default_rng(11)
    for _ in range(500):
        q = float(rng.standard_normal())
        for box in (a, b):
            box._pending_ports.append((node, np.array([1.0]), q))
        a.step()
        b.step()
    assert _identical(a, b)
    assert a.injected_energy() == b.injected_energy(), "a one-node port book is a sum of length one"


def test_the_shared_free_pressure_helper_is_bit_identical_across_the_languages():
    """``_free_pressure_nodes`` is Python, unchanged, and now reads a **Rust** room's arrays.

    Its docstring's claim — that a local divergence read reproduces ``_divergence()``-then-closure
    exactly, at wall, edge and corner nodes alike — was a within-language claim while the room was
    Python on both sides of it. Porting the room turned it into a cross-language one, which is
    §28.2's anchor-to-an-unported-client seen from the other end: here the client re-derives what
    the ported code computes and the *reference* for the comparison is the ported side.
    """
    # `"an open face"` is deliberately NOT in this list, and adding it would fail by design: the
    # helper's own docstring says the identity holds for every wall type EXCEPT an open one, where
    # `step` pins `p = 0` and this read does not. Ports touching an open face are refused at
    # construction, so the identity holds by construction rather than by luck.
    for wall_name in ("rigid", "two lossy faces"):
        a, b = _seed(*_pair(**_kwargs(walls=WALLS[wall_name])))
        for _ in range(50):
            a.step()
            b.step()
        # Every node on a wall, an edge and a corner, plus the interior.
        nx, ny, nz = a.N
        picks = [
            (0, 0, 0),
            (nx, ny, nz),
            (0, ny // 2, nz // 2),
            (nx, 0, nz),
            (nx // 2, ny // 2, nz // 2),
        ]
        nodes = tuple(np.array([p[d] for p in picks]) for d in range(3))
        free_a = _free_pressure_nodes(a, nodes)
        free_b = _free_pressure_nodes(b, nodes)
        assert np.array_equal(free_a, free_b), wall_name
        # ... and the helper still agrees with the full-array route on the Rust room.
        p_old = b.p
        p_full = p_old - b.k * b.rho0 * b.c0**2 * b._divergence()
        if b._has_walls:
            p_full = (p_full - b._beta * p_old) / (1.0 + b._beta)
        want = 0.5 * (p_full[nodes] + p_old[nodes])
        assert np.array_equal(free_b, want), wall_name


def test_a_state_assignment_of_the_wrong_shape_is_refused_at_the_assignment():
    """A migration wants a wrong write to fail where it happens, not three models downstream."""
    box = physsynth_rs.AirBox(**_kwargs())
    with pytest.raises(ValueError, match="must have shape"):
        box.p = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="must have shape"):
        box.ux = np.zeros(box.uy.shape)
    good = np.ones(box.p.shape)
    box.p = good
    assert np.array_equal(box.p, good)


def test_the_read_outs_agree():
    """``node_index``, ``snapped`` and ``pressure_at`` — the geometry a listener goes through.

    ``node_index`` is the second discrete output in the file (§25.2): a wrong one silently
    relocates a microphone, and every energy bar passes.
    """
    a, b = _seed(*_pair(**_kwargs()))
    for _ in range(20):
        a.step()
        b.step()
    rng = np.random.default_rng(5)
    for _ in range(200):
        point = tuple(float(rng.uniform(0.0, v)) for v in a.L_actual)
        assert a.node_index(point) == b.node_index(point)
        assert a.snapped(point) == b.snapped(point)
        assert a.pressure_at(point) == b.pressure_at(point)
    assert np.array_equal(a.state, b.state)
    assert a.state is not a.p and b.state is not b.p, "`state` must be a copy"
