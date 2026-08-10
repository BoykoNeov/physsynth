"""The drumhead in the room: model #4 as a suspended and baffled surface (air-box batch 5).

Batches 3 and 4 gave the room a *plate* — mounted flush in a wall (a source) and hanging in the air
(an object). This hangs a **membrane** there instead, and the physics that changes is not a detail.

**A membrane has no coincidence frequency.** Kirchhoff bending gives ``c_b(omega) = sqrt(kappa
omega)``, unbounded, so every plate in this repo *crosses* ``c0`` at one frequency: poor radiator
below, good above. A membrane's wave speed is the constant ``c = sqrt(T/rho)`` with no ``omega`` in
it. So the surface is subsonic at **every** mode or supersonic at every mode, the control is the
single number ``c/c0`` that a player sets by tightening the head, and a real drumhead sits well
below it (``c/c0 ~ 0.31`` for Mylar) — which is why a head with no shell is quiet. That claim ships
**bracketed**, because the 5-point scheme's own dispersion manufactures a spurious coincidence at
about 2.2 nodes per wavelength for ``c/c0 = 1.05``; see :class:`RoomLoadedMembrane`.

**What this file inherits and does not re-derive.** The conserved total is *necessary and not
sufficient* (batch 3: a wrong ``R_j`` leaves it flatter than the correct run while the two ledgers
sit 18% apart), and the money test ``radiated == injected`` is not sufficient *either* (batch 4: it
is blind to a ``2`` that is wrong only inside the factorization). Both guards ship here as they
stand, and a green total is not read as a pass.

**What is new, and is therefore pinned by its own oracle:**

* ``Membrane.step()`` takes no ``f_ext``, so this batch's force path has nothing in the model to be
  bit-identical *to*. :func:`test_the_f_ext_term_is_pinned_twice` checks the coefficient exactly
  (one step from rest) and its sign and operator physically (static deflection).
* Model #4 was **explicit** — one matvec, no solve. The load's unknown is ``u^{n+1}``, so it goes
  into ``A`` and the membrane acquires a factorization it never had. That buys passivity by
  construction, and :func:`test_the_lagged_explicit_load_is_caught_only_by_the_total` is the
  measured negative control for the alternative — which turned up **a third detector**. Batch 3's
  blind spot was the conserved total, batch 4's was the money test; the lagged load is caught by
  the total (3.8e-2 of ``E0``) and is *invisible* to ``radiated == injected`` (1.6e-16), because
  that identity is a property of the port relation alone and cannot see which velocity produced
  the ``q``. Three batches, three different detectors: the lesson is not which test is the money
  test, it is that no single one of them is.
* The **round** head is the interesting one, and the port refused every disk until this batch (see
  ``test_airbox_surface.py``'s footprint section). Its two staircases — the clamped rim on the
  membrane grid, the footprint on the air grid — are why every disk assertion here is a ratio or a
  rate and never a magnitude.
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_MEMBRANE_FS,
    AIRBOX_MEMBRANE_INDEX,
    make_air_membrane,
    make_membrane_room,
    make_room_loaded_membrane,
    make_suspended_membrane,
    membrane_bulge,
    membrane_bump,
    surface_scene_energy,
)
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from physsynth.core.airbox import (
    RoomLoadedMembrane,
    RoomSuspendedMembrane,
    impedance_from_zeta,
)

DRIFT_TOL = 1e-12   # the scene total, relative -- necessary, not sufficient (see the module head)
LEDGER_TOL = 1e-12  # |radiated - injected| / |radiated| -- necessary, ALSO not sufficient
DOMAINS = ("rectangle", "circle")
TIERS = ("baffled", "suspended")
WALLS = {
    "rigid": "rigid",
    "all-lossy": impedance_from_zeta(4.0),
    "one-lossy-wall": {"z0": impedance_from_zeta(3.0)},
}


def _make(tier, **kw):
    maker = make_room_loaded_membrane if tier == "baffled" else make_suspended_membrane
    return maker(**kw)


def _run(inst, steps, f_ext=None):
    """Step instrument and room in the contract's order: the port solves, then one room step."""
    for _ in range(steps):
        inst.step(f_ext)
        inst.room.step()


def _seeded(tier, shape=membrane_bulge, **kw):
    inst = _make(tier, **kw)
    inst.set_state(shape(inst.membrane))
    return inst


def _unload(inst):
    """Zero the port's coupling in place — the bare membrane, still inside the wrapper.

    Not the same as building a bare :class:`Membrane`: this keeps every line of
    :meth:`RoomLoadedMembrane.step` in play and removes only the room, which is what makes it a
    usable rig for the ``f_ext`` oracle (the room has no DC response to compare against).
    """
    port = inst.port
    port.T = sparse.csr_matrix(port.T.shape)
    port.load_matrix = sparse.csr_matrix(port.load_matrix.shape)
    m = inst.membrane
    a = (1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")
    inst._lu_loaded = splu(a.tocsc())
    return inst


# -- the reduction: the load's zero is a clean zero ----------------------------------------


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_zero_area_reduces_to_the_bare_membrane(tier, domain):
    """``T = 0`` must give **bit-identical** state to a bare :class:`Membrane`, not merely close.

    This is the ``eliminate_zeros()`` path: with zero areas the load block is structurally present
    and numerically zero, so the factorization must come out as the membrane's own ``(1 + sigma k)
    I`` and the solve must reproduce the explicit division exactly. It is what makes the whole
    batch falsifiable at one end — every other assertion here is to a tolerance.
    """
    inst = _make(tier, domain=domain, N=16, sigma=3.0)
    inst.port.areas = np.zeros_like(inst.port.areas)
    inst.port.T = sparse.csr_matrix(inst.port.T.shape)
    inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
    m = inst.membrane
    a = ((1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")).tocsc()
    a.eliminate_zeros()
    inst._lu_loaded = splu(a)

    bare = make_air_membrane(domain=domain, N=16, sigma=3.0)
    u0 = membrane_bump(bare)
    inst.set_state(u0)
    bare.set_state(u0)
    for _ in range(50):
        inst.step()
        inst.room.step()
        bare.step()
    assert np.array_equal(inst.membrane.u, bare.u), "the zero load must be bit-identical, not close"
    assert np.array_equal(inst.membrane.u_prev, bare.u_prev)
    assert inst.radiated_energy == 0.0


# -- the money test, and the channel it runs through ---------------------------------------


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_ledgers_agree_and_the_channel_is_not_vacuous(tier, domain, wall_name):
    """``radiated_energy == room.injected``, reported **with the channel size**.

    A conservation test on a channel worth 1e-14 of the total passes with the coupling
    disconnected, so the channel is asserted too. Batch 3 named the free plate's piston as its
    non-vacuous configuration; **a membrane has no piston** — the rim is clamped, so there is no
    rigid-body nullspace at all — and the configuration that had to be found instead is the
    single-signed fundamental bulge. Measured 0.14 … 0.82 of ``E0`` across these cases, at the
    drum's *actual* subsonic operating point and needing neither a fast head nor a lossy room.
    """
    inst = _seeded(tier, domain=domain, N=16, walls=WALLS[wall_name])
    e0 = surface_scene_energy(inst)
    _run(inst, 300)
    gap = abs(inst.radiated_energy - inst.room.injected)
    assert gap <= LEDGER_TOL * abs(inst.radiated_energy)
    assert abs(inst.radiated_energy) > 1e-2 * e0, "the channel must be worth asserting on"


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_the_scene_total_is_flat(tier, domain):
    """Necessary and not sufficient — asserted, and *named* so (see the module head)."""
    inst = _seeded(tier, domain=domain, N=16, walls=WALLS["all-lossy"])
    e0 = surface_scene_energy(inst)
    worst = 0.0
    for _ in range(300):
        inst.step()
        inst.room.step()
        worst = max(worst, abs(surface_scene_energy(inst) - e0))
    assert worst <= DRIFT_TOL * abs(e0)


@pytest.mark.parametrize("tier", TIERS)
def test_the_channel_shows_the_acoustic_short_circuit(tier):
    """The fine spatial pattern radiates worse than the smooth one — in the channel itself.

    The same membrane, the same energy, two shapes: the single-signed bulge (large net volume
    velocity) against a narrow strike (fine pattern, cancelling on the scale of the acoustic
    wavelength). This is the short circuit measured *without* a mode decomposition, and it is the
    reason the bulge is the configuration the conservation tests use.
    """
    channels = {}
    for name, shape in (("bulge", membrane_bulge), ("bump", membrane_bump)):
        inst = _seeded(tier, shape=shape, N=16, walls=WALLS["all-lossy"])
        e0 = surface_scene_energy(inst)
        _run(inst, 300)
        channels[name] = abs(inst.radiated_energy) / e0
    assert channels["bulge"] > 2.0 * channels["bump"], channels


# -- the guard batch 4 had to invent, applied to the doubled membrane load -----------------


def _refactor(inst, scale):
    """Rebuild the factorization with the load block scaled by ``scale`` (1.0 = as built)."""
    m = inst.membrane
    a = (1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")
    a = (a + scale * inst._load_scale * inst.port.load_matrix).tocsc()
    a.eliminate_zeros()
    inst._lu_loaded = splu(a)


@pytest.mark.parametrize("fs", (AIRBOX_MEMBRANE_FS, 0.75 * AIRBOX_MEMBRANE_FS))
def test_the_coupled_residual_catches_both_wrong_2s(fs):
    """The achieved ``u^{n+1}`` back in the coupled PDE, with the **room's own** pressure jump.

    Batch 4's guard, inherited because it is the only one that catches both ways of getting the
    two-loaded-faces factor wrong: control **A** (``1x`` in the factorization only) is invisible to
    ``radiated == injected``, control **B** (``1x`` consistently) is invisible to the scene total.
    Two timesteps, because a wrong-but-consistent ``k``-dependent factor passes at one.
    """
    def residual(control):
        inst = make_suspended_membrane(
            N=12, sigma=2.0, walls=WALLS["one-lossy-wall"], fs=fs
        )
        if control == "A":        # 1x inside the factorization only
            _refactor(inst, 0.5)
        elif control == "B":      # 1x consistently: "I forgot the head has two faces"
            inst.port.R = 0.5 * inst.port.R
            inst.port.load_matrix = 0.5 * inst.port.load_matrix
            _refactor(inst, 1.0)
        m, room, port = inst.membrane, inst.room, inst.port
        inst.set_state(membrane_bump(m))
        _run(inst, 5)

        f_ext = 1e-3 * np.random.default_rng(0).standard_normal(m.n_live)
        u_n, u_nm1 = m.u.copy(), m.u_prev.copy()
        p_old = room.p.copy()
        inst.step(f_ext)
        room.step()
        u_np1 = m.u.copy()
        pbar = 0.5 * (room.p + p_old)
        jump = pbar[port.nodes_hi] - pbar[port.nodes_lo]

        k = m.k
        velocity = (u_np1 - u_nm1) / (2.0 * k)
        f_total = f_ext - (port.T.T @ jump)
        accel = u_np1 - 2.0 * u_n + u_nm1
        res = (
            accel
            - k * k * m.c * m.c * (m.L @ u_n)
            + 2.0 * m.sigma * k * k * velocity
            - k * k * f_total / (m.rho * m.h * m.h)
        )
        return float(np.max(np.abs(res)) / np.max(np.abs(accel)))

    assert residual("correct") <= 1e-11
    assert residual("A") > 1e-6, "a 1x factorization must not pass the residual"
    assert residual("B") > 1e-6, "a consistent 1x must not pass the residual either"


# -- passivity, and R read off the room rather than trusted --------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_a_lossy_head_in_a_lossy_room_is_monotone(tier):
    """Every channel dissipative -> the scene total decreases at every step, never rises."""
    inst = _seeded(tier, N=16, sigma=40.0, walls=WALLS["all-lossy"])
    previous = surface_scene_energy(inst)
    for _ in range(300):
        inst.step()
        inst.room.step()
        now = surface_scene_energy(inst)
        assert now <= previous + 1e-14 * abs(previous)
        previous = now


@pytest.mark.parametrize("tier", TIERS)
def test_R_is_the_rooms_own_differential_response(tier):
    """``R_j`` measured off the room per node, not trusted from the assembly line.

    Inject a unit volume velocity at one port node into an otherwise silent room and read the
    pressure the room itself produces. ``pbar = pbar_free + R q`` is the Thevenin relation the
    whole coupling rests on; this is the only test that checks the ``R`` in it against the object
    it claims to describe.
    """
    inst = _make(tier, N=12, walls=WALLS["one-lossy-wall"])
    port, room = inst.port, inst.room
    q = np.zeros(port.T.shape[0])
    q[len(q) // 2] = 1.0
    before = room.p.copy()
    port.inject(q)
    room.step()
    measured = (room.p + before) * 0.5
    if hasattr(port, "nodes_hi"):
        response = measured[port.nodes_hi] - measured[port.nodes_lo]
        expected = 2.0 * port.R * q
    else:
        response = measured[port.nodes]
        expected = port.R * q
    live = q > 0
    assert np.allclose(response[live], expected[live], rtol=1e-12, atol=0.0)


# -- the f_ext path: new arithmetic, so two independent oracles ----------------------------


def test_the_f_ext_term_is_pinned_twice():
    """``Membrane.step()`` takes no force, so this term has nothing to be bit-identical to.

    Batch 3 could copy :meth:`~physsynth.core.plate.Plate.step`'s own ``f_ext`` path line for line
    and a slip would show up against the model. Here it is *new* arithmetic, and the energy ledger
    would stay green with the coefficient wrong (it telescopes against whatever force was used).
    So it is pinned twice and neither is a conservation check:

    1. **The coefficient, exactly.** One step from rest with the load removed must give exactly
       ``k^2 f / (rho h^2) / (1 + sigma k)``. The load has to be removed and it is not a
       simplification: even from rest the first step's centered velocity
       ``(u^1 - u^{-1}) / 2k`` is nonzero, so the room loads the head immediately — measured, that
       alone moves the answer by 0.96%, which is exactly the size of thing an "approximately
       equal" version of this test would have waved through.
    2. **The sign and the operator, physically.** Held down by a constant force with the room
       removed, the head must settle on the static deflection ``u_ss = -L^-1 f / (T h^2)``, which
       is the continuum ``-T grad^2 u = f`` discretized — a different statement from (1), and the
       one that would catch a sign or a ``c^2`` in the wrong place.
    """
    inst = _unload(_make("baffled", N=12, sigma=5.0))
    m = inst.membrane
    f = 7.0 * np.cos(11.0 * m.X[m.mask]) * np.sin(5.0 * m.Y[m.mask])
    inst.set_state(np.zeros(m.n_live))
    assert np.all(inst.room.p == 0.0), "the exact step needs a silent room"
    inst.step(f)
    expected = m.k * m.k * f / (m.rho * m.h * m.h) / (1.0 + m.sigma * m.k)
    assert np.array_equal(m.u, expected), "the f_ext coefficient must be exact, not close"

    settled = _unload(_make("baffled", N=12, sigma=4000.0))
    m = settled.membrane
    f = 2.0 * np.ones(m.n_live)
    settled.set_state(np.zeros(m.n_live))
    _run(settled, 8000, f)
    static = -spsolve(m.L.tocsc(), f) / (m.T * m.h * m.h)
    assert np.allclose(m.u, static, rtol=2e-6, atol=0.0)


# -- the negative control, measured once ---------------------------------------------------


def test_the_lagged_explicit_load_is_caught_only_by_the_total():
    """Keeping model #4 explicit costs conservation — and **the money test does not notice**.

    Evaluating the load velocity at the *backward* difference ``(u^n - u^{n-1})/k`` leaves the
    membrane explicit (no factorization, one matvec, its old character intact) and looks entirely
    reasonable. What it loses is passivity by construction: the load force no longer pairs with the
    centered velocity the membrane's energy identity uses, so the head's own ledger stops
    telescoping. Measured over 300 steps: the scene total drifts **3.8e-2 of E0**, against 8.6e-15
    for the shipped scheme.

    **And ``radiated == injected`` stays at rounding — 1.6e-16 — through all of it.** That inverts
    batches 3 and 4, where the conserved total was the blind one and the cross-ledger identity was
    the money test. It is blind here for a reason worth stating: ``radiated == injected`` is a
    property of the *port relation* alone (the room receives exactly the ``q`` it was handed, at
    exactly the pressure ``pbar_free + R q`` it then has), so it cannot see which velocity produced
    that ``q``. Three batches, three different detectors — the lesson is not which test is the
    money test, it is that **no single one of them is**.

    ``spreading="nearest"`` is the precedent: ship exactly one measured negative control, and do
    not offer it as a configuration.
    """
    def measure(lagged):
        inst = make_room_loaded_membrane(N=16, walls=WALLS["all-lossy"])
        inst.set_state(membrane_bulge(inst.membrane))
        m, port = inst.membrane, inst.port
        lu = splu(((1.0 + m.sigma * m.k) * sparse.identity(m.n_live, format="csc")).tocsc())
        e0 = surface_scene_energy(inst)
        worst = 0.0
        for _ in range(300):
            if not lagged:
                inst.step()
            else:
                port.require_ready()
                pbar_free = port.free_pressure()
                v_lagged = (m.u - m.u_prev) / m.k      # BACKWARD, not centered: the whole change
                q = port.T @ v_lagged
                pbar = pbar_free + port.R * q
                rhs = inst._surface.rhs(None) - m.k * m.k * (port.T.T @ pbar) / inst._denominator
                inst._surface.commit(lu.solve(rhs))
                port.inject(q)
                inst.radiated_energy += m.k * float(np.dot(pbar, q))
                inst.n += 1
            inst.room.step()
            worst = max(worst, abs(surface_scene_energy(inst) - e0))
        gap = abs(inst.radiated_energy - inst.room.injected) / abs(inst.radiated_energy)
        return worst / abs(e0), gap, abs(inst.radiated_energy) / e0

    shipped_drift, shipped_gap, shipped_channel = measure(False)
    lagged_drift, lagged_gap, lagged_channel = measure(True)
    assert shipped_drift <= DRIFT_TOL and shipped_gap <= LEDGER_TOL
    assert lagged_drift > 1e-3, f"the lagged load must cost real energy, got {lagged_drift:.2e}"
    assert lagged_drift > 1e9 * shipped_drift
    # The point of the test: the cross-ledger identity is BLIND to it.
    assert lagged_gap <= LEDGER_TOL, f"expected the money test to stay green, got {lagged_gap:.2e}"
    # ... and not because nothing happened -- the channel is the same size either way.
    assert lagged_channel > 0.5 * shipped_channel


# -- geometry: the round head, and the two areas that are not the same number ---------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_cut_follows_the_port_and_the_two_areas_differ(domain):
    """The obstacle is the *air* footprint; the moving surface is the *live* nodes. Not equal.

    Batch 4 measured that a suspended surface's radiated magnitude tracks ``blocked_area`` rather
    than the air spacing, so an area quietly taken as ``pi R^2`` would give a plausible, wrong and
    green-ledgered result. Both numbers are asserted to exist, to disagree, and to bracket the
    nominal area from opposite sides — the clamped rim makes the mover *smaller* than nominal and
    the staircased footprint makes the obstacle *larger*.
    """
    inst = make_suspended_membrane(domain=domain, N=16)
    port, m = inst.port, inst.membrane
    assert port.face_count == inst.room.cut_faces
    nominal = np.pi * m.radius**2 if domain == "circle" else m.Lx * m.Ly
    assert np.isclose(port.net_area, m.n_live * m.h * m.h, rtol=1e-12)
    assert port.net_area < nominal < port.blocked_area
    assert port.blocked_area > 1.05 * port.net_area


def test_the_default_origin_centres_a_disk():
    """The first surface for which ``origin=None`` centres something round — re-measured, not
    assumed. A centred surface in a mirror-symmetric room keeps the reflection equivariance the
    spreading operator's symmetry argument rests on, so the load matrix must be invariant under
    the in-plane 180-degree rotation that maps the disk to itself."""
    inst = make_suspended_membrane(domain="circle", N=16)
    port = inst.port
    coords = port.coords
    centre = 0.5 * (coords.max(axis=0) + coords.min(axis=0))
    flipped = 2.0 * centre - coords
    order = np.lexsort((np.round(flipped[:, 1], 9), np.round(flipped[:, 0], 9)))
    inverse = np.lexsort((np.round(coords[:, 1], 9), np.round(coords[:, 0], 9)))
    perm = np.empty(len(order), dtype=int)
    perm[inverse] = order
    load = port.load_matrix.toarray()
    assert np.allclose(load[np.ix_(perm, perm)], load, atol=1e-18)


# -- refusals -------------------------------------------------------------------------------


def test_refuses_a_sample_rate_mismatch_and_names_the_membrane():
    room = make_membrane_room()
    membrane = make_air_membrane(fs=0.5 * AIRBOX_MEMBRANE_FS)
    with pytest.raises(ValueError, match="membrane fs"):
        RoomLoadedMembrane(membrane=membrane, room=room, face="z0")
    with pytest.raises(ValueError, match="membrane fs"):
        RoomSuspendedMembrane(
            membrane=membrane, room=room, plane="z", index=AIRBOX_MEMBRANE_INDEX
        )


def test_refuses_a_head_that_overruns_the_plane():
    """A footprint reaching the plane's own rim touches a second wall, so ``R_j`` stops being
    uniform across the patch — inherited from :class:`SurfacePort` unchanged."""
    with pytest.raises(ValueError, match="rim"):
        make_room_loaded_membrane(N=24, L=0.50)


def test_refuses_solving_twice_without_a_room_step():
    inst = make_room_loaded_membrane(N=12)
    inst.step()
    with pytest.raises(RuntimeError, match="twice within one room step"):
        inst.step()


def test_refuses_overlapping_ports():
    room = make_membrane_room()
    make_room_loaded_membrane(room=room, N=12)
    with pytest.raises(ValueError, match="shares node"):
        make_room_loaded_membrane(room=room, N=12)


def test_the_membrane_cfl_is_the_models_own_refusal():
    """The coupling adds no third stability condition: ``lambda_mem <= 1/sqrt(2)`` and the room's
    ``lambda_air <= 1/sqrt(3)`` remain the only two, and the first is refused by model #4 itself
    before a port is ever built."""
    with pytest.raises(ValueError, match="CFL"):
        make_air_membrane(N=32, T=3.0e5)


def test_there_is_no_pressure_readout_and_that_is_deliberate():
    """Model #4 has no ``pressure()``, and the wrapper does not invent one.

    ``Plate`` exposes the monopole ``sum_i area_i u_i''``; ``Membrane`` never has, and its
    two-level roll keeps no ``_accel`` to build one from. The batch-3 docstring used to claim all
    three grid models had it. For the room's own pressure, read the field.
    """
    inst = make_room_loaded_membrane(N=12)
    assert not hasattr(inst, "pressure")
    assert isinstance(inst.room.pressure_at((0.1, 0.1, 0.1)), float)


# -- the drop-in surface --------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_delegates_to_the_membrane_and_overrides_energy(tier):
    """A drop-in for a bare :class:`Membrane` on every read accessor — except ``energy()``, which
    must include the coupling channel or it is the number that looks fine and is not conserved."""
    inst = _seeded(tier, N=12, walls=WALLS["all-lossy"])
    _run(inst, 40)
    assert isinstance(inst, RoomLoadedMembrane | RoomSuspendedMembrane)
    assert inst.n_live == inst.membrane.n_live
    assert np.array_equal(inst.u, inst.membrane.u)
    assert inst.state.shape == inst.membrane.mask.shape
    assert inst.energy() == inst.membrane.energy() + inst.radiated_energy
    assert inst.energy() != inst.membrane.energy()


@pytest.mark.parametrize("tier", TIERS)
def test_reset_clears_the_ledger_but_not_the_geometry(tier):
    """The cut is geometry, not state: it must survive a reset, as it must for batch 4."""
    inst = _seeded(tier, N=12)
    _run(inst, 20)
    faces_before = inst.room.cut_faces
    inst.reset()
    assert inst.radiated_energy == 0.0
    assert inst.volume_velocity == 0.0
    assert np.all(inst.membrane.u == 0.0)
    assert inst.room.cut_faces == faces_before


def test_two_heads_share_one_room():
    """N instruments in one room, inherited unchanged: disjoint node sets, one room step."""
    room = make_membrane_room()
    a = make_suspended_membrane(room=room, N=12, index=3)
    b = make_suspended_membrane(room=room, N=12, index=6)
    a.set_state(membrane_bulge(a.membrane))
    b.set_state(membrane_bump(b.membrane))
    e0 = surface_scene_energy(a, b)
    worst = 0.0
    for _ in range(200):
        a.step()
        b.step()
        room.step()
        worst = max(worst, abs(surface_scene_energy(a, b) - e0))
    assert worst <= DRIFT_TOL * abs(e0)
    total = a.radiated_energy + b.radiated_energy
    assert abs(total - room.injected) <= LEDGER_TOL * abs(total)
