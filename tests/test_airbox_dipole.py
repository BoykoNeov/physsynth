"""The interior two-sided (dipole) plate: the plate stops being a source (air-box batch 4).

Batch 3 mounted a plate flush in a wall and let it radiate from every node. The wall did the rest —
it was the textbook infinite baffle, the plate's back face was unloaded, and the plate was, for all
the room could tell, a **source**: a patch of wall that moved. Batch 4 takes the wall away. The
plate hangs *in* the room, radiates from both faces, and is driven by the pressure **jump** across
it.

That is not batch 3 with a sign. A source adds sound to a room; an **object** also removes paths
through it, and does so whether or not it is moving. The headline test is that the difference does
not converge away: :func:`test_the_source_alone_converges_to_silence` drops the cut and keeps the
``-q``/``+q`` pair (the "phantom" — a legal, perfectly conservative dipole *source* with the plate's
own motion and no obstacle) and finds its decay **diverging** from the real plate's under air-grid
refinement, because a transparent doublet at separation ``h`` has moment proportional to ``h``.

**This file's methodological finding corrects batch 3's.** Batch 3 established that the conserved
total is blind to a wrong ``R_j`` and promoted ``radiated == injected`` to the money test. Batch 4
has a coefficient batch 3 did not — the **2** of the two loaded faces — and the money test is blind
to half of the ways to get it wrong. Measured in
:func:`test_the_coupled_residual_catches_both_wrong_2s`:

==============================================  ===========  ================  ================
error                                             residual        |rad-inj|        scene drift
==============================================  ===========  ================  ================
*(correct)*                                       1.4e-15          6.1e-16           1.6e-14
**A** — ``1x`` in the **factorization only**      **2.3e-04**      4.5e-16 blind     1.8e-02
**B** — ``1x`` **consistently** ("one face?")     **6.5e-04**      1.80x             2.4e-14 blind
==============================================  ===========  ================  ================

So **the money test alone is not sufficient either**. What catches both is putting the achieved
``u^{n+1}`` back into the coupled PDE with the force computed from the **room's own post-closure
pressure jump** — a number the port never touched — at **two** timesteps, because a wrong-but-
consistent ``k``-dependent factor passes at one. ``assert load_matrix == 2 * one_sided`` is not a
test; it re-checks arithmetic the same file just wrote.

Two further traps get their own tests:

* **The orientation flip is invisible to every energetic quantity.** ``2 T^T R T`` is
  sign-invariant, and a *consistent* flip (inject ``+q``/``-q`` and read the jump the other way
  round) leaves the load matrix, the solve, the pressure jump and ``radiated_energy`` all
  **bit-identical** while the room's field is exactly inverted and the plate is anti-driven by it.
  Its only detector is the sign of the room's own pressure on the **first** step.
* **``radiated_energy`` is not a radiation measure here.** Batch 3's channel was a one-way drain;
  this one is a **reservoir** — half its increments are negative — so neither it nor a decay time
  ships as a radiation figure. Radiation efficiency needs a prescribed-velocity rig, which is
  ``scripts/diagnose_airbox_dipole.py``'s job, not this file's.
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_SURFACE_N,
    make_room_loaded_plate,
    make_surface_room,
    make_suspended_plate,
    plate_bump,
    plate_mode_shape,
    surface_scene_energy,
)
from scipy import sparse
from scipy.sparse.linalg import splu

from physsynth.core.airbox import (
    PLANES,
    InteriorSurfacePort,
    RoomSuspendedPlate,
    impedance_from_zeta,
)
from physsynth.core.connection import StringPlateBridge
from physsynth.core.plate import Plate
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-12   # the scene total, relative -- necessary, not sufficient (see the module head)
LEDGER_TOL = 1e-12  # |radiated - injected| / |radiated| -- necessary, ALSO not sufficient
BOUNDARIES = ("supported", "free")
WALLS = {
    "rigid": "rigid",
    "all-lossy": impedance_from_zeta(4.0),
    "one-lossy-wall": {"z0": impedance_from_zeta(3.0)},
}


def _run(inst, steps, f_ext=None):
    """Step instrument and room in the contract's order: the port solves, then one room step."""
    for _ in range(steps):
        inst.step(f_ext)
        inst.room.step()


def _seeded(**kwargs):
    inst = make_suspended_plate(**kwargs)
    inst.set_state(plate_bump(inst.plate))
    return inst


def _piston(**kwargs):
    """A free plate given a uniform velocity — the loudest motion the geometry has.

    Batch 3's baffled version of this is a piston in an infinite baffle and the room takes ~all of
    it. Suspended, the *same* motion is a two-sided plate at low ``ka``, which is a far weaker
    radiator — so the channel is large but no longer everything, and that contrast is physics.
    """
    inst = make_suspended_plate(boundary="free", **kwargs)
    inst.set_state(np.zeros(inst.plate.n_live), 1e-3 * np.ones(inst.plate.n_live))
    return inst


def _refactor(inst, scale):
    """Rebuild the factorization with the load block scaled by ``scale`` (1.0 = as built)."""
    p = inst.plate
    sk = p.sigma * p.k
    coeff = p.theta * p.k * p.k * p.kappa * p.kappa
    if p.boundary == "supported":
        a = (1.0 + sk) * sparse.identity(p.n_live, format="csc") + coeff * p.B
    else:
        a = (1.0 + sk) * p.W + coeff * p.K
    a = (a + scale * inst._load_scale * inst.port.load_matrix).tocsc()
    a.eliminate_zeros()
    inst._lu_loaded = splu(a)


def _uncut(room):
    """Drop every cut, turning a suspended plate into the **phantom** — see the headline test.

    Asserts the room really is uncut afterwards: if the cut ever grows a fourth piece of state this
    helper stops uncutting, and the headline test would silently compare the dipole to itself and
    pass trivially.
    """
    room._cut_mask = [None, None, None]
    room._cut_index = [None, None, None]
    room._cuts = []
    assert room.cut_faces == 0
    return room


# -- the money test, and why it is not enough --------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_ledgers_agree(boundary, wall_name):
    """``radiated_energy == room.injected``, with the room booking from its **own** field.

    The port predicts its work from ``d_pbar = d_free + 2 R q`` summed over the faces; the room
    books ``k q . (pbar_hi - pbar_lo)`` from its own post-closure pressure on the two node planes,
    never from a number handed back. They agree only if every ``R_j`` is right *and* the
    ``-q``/``+q`` pair lands on the right planes — but not only if the **2** is right, which is what
    :func:`test_the_coupled_residual_catches_both_wrong_2s` exists for.
    """
    inst = _seeded(boundary=boundary, walls=WALLS[wall_name])
    e0 = inst.plate.energy()
    _run(inst, 300)
    gap = abs(inst.radiated_energy - inst.room.injected)
    assert gap <= LEDGER_TOL * abs(inst.radiated_energy)
    assert abs(inst.radiated_energy) > 1e-5 * e0  # ... and the channel is not vacuous


def test_the_piston_is_the_non_vacuous_channel():
    """A free plate's rigid-body piston puts ~46% of ``E0`` through the coupling, and it balances.

    Every conservation claim must report how big the channel is: a conservation test on a channel
    worth 1e-14 of the total passes with the coupling disconnected. A bump's channel here is
    ~2e-4 of ``E0`` (fine spatial patterns radiate badly — the acoustic short circuit doing its job,
    and a *dipole* short-circuits twice over), so this is the configuration that makes the assertion
    mean something. Note the contrast with batch 3: baffled, this same motion loses **99.7%** of
    ``E0`` to a lossy room; suspended it loses about half, and half of *that* comes back.
    """
    inst = _piston(walls=WALLS["all-lossy"])
    e0 = inst.plate.energy()
    _run(inst, 400)
    assert abs(inst.radiated_energy) > 0.2 * e0
    gap = abs(inst.radiated_energy - inst.room.injected)
    assert gap <= LEDGER_TOL * abs(inst.radiated_energy)


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("fs", [8000.0, 11000.0])
def test_the_coupled_residual_catches_both_wrong_2s(boundary, fs):
    """The batch's primary guard: the achieved ``u^{n+1}`` back in the coupled PDE, twice over.

    The force is rebuilt from the **room's own post-closure pressure jump** — a number the port
    never touched — so this is not a re-check of the port's own arithmetic. Run at two timesteps
    because a wrong-but-consistent ``k``-dependent factor passes at one, with ``sigma > 0``, a
    nonzero ``f_ext`` and a lossy wall so nothing is invisible. Both negative controls are here
    because each is blind to a *different* one of the two ledgers (see the module docstring).
    """
    def residual(control):
        inst = make_suspended_plate(
            boundary=boundary, walls=WALLS["one-lossy-wall"], N=6, sigma=2.0, fs=fs
        )
        if control == "A":        # 1x inside the factorization only
            _refactor(inst, 0.5)
        elif control == "B":      # 1x consistently: "I forgot the plate has two faces"
            inst.port.R = 0.5 * inst.port.R
            inst.port.load_matrix = 0.5 * inst.port.load_matrix
            _refactor(inst, 1.0)
        plate, room, port = inst.plate, inst.room, inst.port
        inst.set_state(plate_bump(plate))
        _run(inst, 5)

        f_ext = 1e-3 * np.random.default_rng(0).standard_normal(plate.n_live)
        u_n, u_nm1 = plate.u.copy(), plate.u_prev.copy()
        p_old = room.p.copy()
        inst.step(f_ext)
        room.step()
        u_np1 = plate.u.copy()
        pbar = 0.5 * (room.p + p_old)
        jump = pbar[port.nodes_hi] - pbar[port.nodes_lo]

        k, theta = plate.k, plate.theta
        average = theta * u_np1 + (1.0 - 2.0 * theta) * u_n + theta * u_nm1
        velocity = (u_np1 - u_nm1) / (2.0 * k)
        f_total = f_ext - (port.T.T @ jump)
        accel = u_np1 - 2.0 * u_n + u_nm1
        if boundary == "supported":
            mass, stiffness, weight = plate.rho * plate.h * plate.h, plate.B, 1.0
        else:
            mass, stiffness, weight = plate.rho, plate.K, plate.w
        res = (
            weight * accel
            + k * k * plate.kappa**2 * (stiffness @ average)
            + 2.0 * plate.sigma * k * k * weight * velocity
            - k * k * f_total / mass
        )
        return float(np.max(np.abs(res)) / np.max(np.abs(weight * accel)))

    assert residual("correct") <= 1e-11
    assert residual("A") > 1e-6, "a 1x factorization must not pass the residual"
    assert residual("B") > 1e-6, "a consistent 1x must not pass the residual either"


def test_each_ledger_is_blind_to_a_different_wrong_2():
    """The measurement the module docstring rests on, asserted rather than asserted-about.

    Control **A** (``1x`` in the factorization only) leaves ``|radiated - injected|`` at *rounding*:
    both sides use the same ``R`` and the same ``q``, and neither knows what matrix produced that
    ``q``. Control **B** (``1x`` consistently) leaves the scene total at rounding: each side's
    ledger telescopes against whatever pressure *it* used, and the sum of two internally-consistent
    identities is conserved even when the two disagree. If a future change made either detector
    sensitive to the other's error, this test fails and the framing above is wrong.
    """
    def books(control):
        inst = make_suspended_plate(walls=WALLS["one-lossy-wall"], N=6)
        if control == "A":
            _refactor(inst, 0.5)
        elif control == "B":
            inst.port.R = 0.5 * inst.port.R
            inst.port.load_matrix = 0.5 * inst.port.load_matrix
            _refactor(inst, 1.0)
        inst.set_state(plate_bump(inst.plate))
        e0 = surface_scene_energy(inst)
        worst = 0.0
        for _ in range(300):
            inst.step()
            inst.room.step()
            worst = max(worst, abs(surface_scene_energy(inst) - e0))
        gap = abs(inst.radiated_energy - inst.room.injected) / abs(inst.radiated_energy)
        return worst / abs(e0), gap

    good_drift, good_gap = books("correct")
    a_drift, a_gap = books("A")
    b_drift, b_gap = books("B")
    assert good_drift <= DRIFT_TOL and good_gap <= LEDGER_TOL
    assert a_gap <= LEDGER_TOL, "the money test is supposed to MISS a wrong factorization"
    assert a_drift > 1e-6, "... and the conserved total is supposed to catch it"
    assert b_drift <= DRIFT_TOL, "the conserved total is supposed to MISS a consistent 1x"
    assert b_gap > 0.5, f"... and the money test is supposed to catch it, got {b_gap:.2e}"


@pytest.mark.parametrize("walls", ["rigid", {"z0": impedance_from_zeta(3.0)}])
def test_R_j_is_the_same_on_both_planes_with_opposite_signs(walls):
    """``d pbar/dq`` measured off the room: ``-R_j`` low, ``+R_j`` high, off-diagonal *exactly* 0.

    That — the same ``R_j``, opposite signs, exactly zero elsewhere — is what the **2** in
    ``2 T^T R T`` *is*, measured rather than constructed. Comparing the coupled step's ``pbar``
    against ``pbar_free + R q`` would be a tautology, so instead: save the room, step it once with
    nothing injected, restore, step it again with a unit ``q`` on one face, and read the difference
    straight off the room's own post-closure field.

    The exact zeros are the stronger half. The room's instantaneous response over a node set is
    diagonal — propagation waits for the next momentum sub-step — and that stays true *across the
    cut*, one cell apart, which is what makes the two planes independent within a step.
    """
    inst = _seeded(walls=walls, N=6)
    room, port = inst.room, inst.port
    _run(inst, 7)  # a nontrivial field, so pbar_free is not trivially zero

    def snapshot():
        return {k: getattr(room, k).copy() for k in
                ("p", "ux", "uy", "uz", "ux_prev", "uy_prev", "uz_prev")}

    def restore(state):
        for k, v in state.items():
            setattr(room, k, v.copy())
        room._pending.clear()
        room._pending_ports.clear()
        port.reset()

    base = snapshot()
    both = tuple(
        np.concatenate((lo, hi))
        for lo, hi in zip(port.nodes_lo, port.nodes_hi, strict=True)
    )
    p_before = base["p"][both]
    room.step()
    pbar0 = 0.5 * (room.p[both] + p_before)

    amp, n = 1e-4, port.face_count
    measured = np.zeros((2 * n, n))
    for j in range(n):
        restore(base)
        q = np.zeros(n)
        q[j] = amp
        port.inject(q)
        room.step()
        measured[:, j] = (0.5 * (room.p[both] + p_before) - pbar0) / amp
    restore(base)

    assert np.allclose(np.diag(measured[:n]), -port.R, rtol=1e-12, atol=0.0)
    assert np.allclose(np.diag(measured[n:]), port.R, rtol=1e-12, atol=0.0)
    off = measured - np.vstack((np.diag(np.diag(measured[:n])), np.diag(np.diag(measured[n:]))))
    assert np.all(off == 0.0), f"max |off| = {np.abs(off).max():.3e}"


# -- conservation, and the volume identity ------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_scene_total_is_flat(boundary, wall_name):
    """``plate.energy() + radiated + room.energy()`` — necessary, and shown not sufficient."""
    inst = _seeded(boundary=boundary, walls=WALLS[wall_name])
    e0 = surface_scene_energy(inst)
    worst = 0.0
    for _ in range(300):
        inst.step()
        inst.room.step()
        worst = max(worst, abs(surface_scene_energy(inst) - e0))
    assert worst <= DRIFT_TOL * abs(e0)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_volume_is_conserved_exactly(boundary):
    """``sum_j q_j == sum_n area_n v_n``, with ``v`` recomputed from the **plate**.

    Batch 3's test unchanged, and it still earns its place: a *consistently* wrong ``q`` factor (a
    stray ``k``, or ``2k`` for ``k``) satisfies both energy ledgers happily, because they both use
    it. Which is why ``v`` is rebuilt here from ``plate.u``/``plate.u_prev``, never off the port.
    """
    inst = _seeded(boundary=boundary, walls=WALLS["all-lossy"])
    plate = inst.plate
    for _ in range(40):
        u_nm1 = plate.u_prev.copy()
        inst.step()
        inst.room.step()
        v = (plate.u - u_nm1) / (2.0 * plate.k)
        net = float(np.sum(inst.nodal_volume_velocity))
        want = float(np.sum(inst.port.areas * v))
        scale = float(np.sum(inst.port.areas * np.abs(v)))
        assert abs(net - want) <= 1e-13 * scale


def test_the_channel_is_a_reservoir_not_a_drain():
    """Half of ``radiated_energy``'s increments are **negative** — which is why it is not a measure.

    The same free-plate piston into the same lossy room: baffled, ``radiated_energy`` is essentially
    monotone (measured 1.2% negative increments — a drain); suspended, it is 50.2% (a reservoir).
    The dipole's channel is dominantly the **reactive near field**, which ``radiated_energy`` counts
    as though it had left. So a decay time and a radiated fraction can disagree about direction, and
    neither ships as a radiation figure — that needs a prescribed-velocity rig over whole cycles.
    """
    def negative_fraction(inst):
        prev, neg = 0.0, 0
        for _ in range(400):
            inst.step()
            inst.room.step()
            neg += inst.radiated_energy < prev
            prev = inst.radiated_energy
        return neg / 400

    suspended = negative_fraction(_piston(walls=WALLS["all-lossy"]))
    baffled = make_room_loaded_plate(boundary="free", walls=WALLS["all-lossy"])
    baffled.set_state(np.zeros(baffled.plate.n_live), 1e-3 * np.ones(baffled.plate.n_live))
    assert negative_fraction(baffled) < 0.1
    assert suspended > 0.4


# -- the sign convention -------------------------------------------------------------------


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_sign_is_readable_on_the_first_step(plane, boundary):
    """A plate moving along ``+plane`` compresses the air it moves toward — read at step **1**.

    Batch 3 recorded that a six-step read gave the *wrong* answer because the plate's own half
    period was five steps; here the flip lands at step 8 instead, which is the point — the number of
    steps is not the lesson, reading step 1 is.
    """
    index = 4 if plane == "z" else 5
    inst = make_suspended_plate(boundary=boundary, plane=plane, index=index)
    n_live = inst.plate.n_live
    v0 = 1e-3 * np.ones(n_live)
    if boundary == "supported":
        v0 = 1e-3 * plate_mode_shape(inst.plate, 1, 1)  # the rim is dead; a piston is not available
    inst.set_state(np.zeros(n_live), v0)
    inst.step()
    inst.room.step()
    port, p = inst.port, inst.room.p
    assert float(np.mean(p[port.nodes_hi])) > 0.0
    assert float(np.mean(p[port.nodes_lo])) < 0.0


def test_a_mirror_symmetric_scene_gives_pbar_lo_equal_to_minus_pbar_hi():
    """A free second oracle: with the cut on the room's own mirror plane the two sides are exact
    negatives of each other, to the last digit. ``N_z = 9`` and ``index = 4`` put the cut at
    ``4.5 h`` from both walls."""
    inst = _piston()
    assert inst.room.N[2] == 2 * inst.port.face_index + 1, "the scene must be symmetric in z"
    _run(inst, 1)
    lo, hi = inst.port.free_pressure()
    assert np.allclose(lo, -hi, rtol=1e-13, atol=0.0)


def test_a_sign_flip_is_invisible_to_every_energy_quantity():
    """Flip the convention consistently and **everything energetic is bit-identical**.

    ``2 T^T R T`` is sign-invariant, so a port that injects ``+q``/``-q`` *and* reads the jump the
    other way round solves the same system, develops the same pressure jump, and books the same
    ``radiated_energy`` — to the last bit — while the room's field is exactly inverted and the plate
    is anti-driven by its own reflections. Asserting the bit-identity of the **wrong** run is the
    point of this test: it is what makes "invisible to every energy quantity" a measured claim
    rather than a warning, and it is what leaves the first-step room pressure as the only detector.
    """
    def run(flip):
        inst = _piston(walls=WALLS["all-lossy"])
        if flip:
            port = inst.port
            free, inject = port.free_pressure, port.inject
            port.free_pressure = lambda: tuple(reversed(free()))
            port.inject = lambda q: inject(-q)
        _run(inst, 120)
        return inst

    good, bad = run(flip=False), run(flip=True)
    assert bad.radiated_energy == good.radiated_energy
    assert np.array_equal(bad.plate.u, good.plate.u)
    assert np.array_equal(bad.pressure_jump, good.pressure_jump)
    assert np.array_equal(bad.room.p, -good.room.p)  # ... and the ROOM is where it shows


# -- the headline ---------------------------------------------------------------------------


def test_the_source_alone_converges_to_silence():
    """Drop the cut and the coupling *diverges* from the real plate's under refinement.

    The phantom — the ``-q``/``+q`` pair with no obstacle — is a legal, perfectly conservative
    dipole **source** carrying the plate's own motion. Measured ``t50`` ratios phantom/dipole of
    **5.2** and **19.3** at 1x and 2x air-grid refinement (40.8 at 3x, in the diagnose script). The
    assertion is the *growth*, not a value.

    Be exact about what that proves. A doublet at separation ``h`` has moment proportional to ``h``
    by construction, so of course it vanishes under refinement; this is a precise **implementation**
    control, not a general claim about source-only tiers. It is exactly what batch 4 degrades to if
    the cut is omitted or clobbered, so the divergence is what proves **the cut is load-bearing and
    cannot be quietly dropped** — at 3x, omitting it is a 40x error that every ledger calls green.
    """
    def t50(refine, phantom, limit=4000):
        inst = make_suspended_plate(
            fs=8000.0 * refine,
            N_room=tuple(n * refine for n in AIRBOX_SURFACE_N),
            index=4 * refine,
            walls=impedance_from_zeta(1.0),
        )
        if phantom:
            _uncut(inst.room)
        inst.set_state(1e-3 * plate_mode_shape(inst.plate, 1, 1))
        e0 = inst.plate.energy()
        for step in range(1, limit + 1):
            inst.step()
            inst.room.step()
            if inst.plate.energy() <= 0.5 * e0:
                return step
        raise AssertionError(f"no t50 within {limit} steps")

    ratios = [t50(r, phantom=True) / t50(r, phantom=False) for r in (1, 2)]
    assert ratios[0] > 2.0, f"the phantom must already be much weaker: {ratios}"
    assert ratios[1] > 2.0 * ratios[0], f"and the gap must GROW with refinement: {ratios}"


def test_the_phantom_is_bit_identically_two_monopoles():
    """Injecting ``-q``/``+q`` with no cut is not an approximation of anything — it is exactly two
    :meth:`AirBox.inject` soft sources, to ``0.000e+00`` over 60 randomised injections.

    That matters because the phantom is the headline's control, and a control is only worth its
    bit-identity to the tier it stands in for — here, batch 1's monopole, twice.
    """
    port_room, ref_room = make_surface_room(), make_surface_room()
    lo = (np.array([4]), np.array([4]), np.array([4]))
    hi = (np.array([4]), np.array([4]), np.array([5]))
    rng = np.random.default_rng(3)
    for _ in range(60):
        q = rng.standard_normal()
        port_room._pending_ports.append((lo, np.array([-q]), 1.0))
        port_room._pending_ports.append((hi, np.array([q]), 1.0))
        port_room.step()
        ref_room.inject(-q, at=tuple(i * ref_room.h for i in (4, 4, 4)))
        ref_room.inject(q, at=tuple(i * ref_room.h for i in (4, 4, 5)))
        ref_room.step()
        assert np.array_equal(port_room.p, ref_room.p)


# -- reductions, the chain, and the guard ----------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_no_air_load_reproduces_the_bare_plate(boundary):
    """``T = 0`` reduces to :class:`Plate` **bit-for-bit** — even though the cut is still there.

    The family's reduction ledger entry, and a statement about what the two halves of this object
    are: the cut belongs to the **room** and the load to the **plate**, so a zero-area surface still
    blocks the room while the plate never notices it is in one. Bit-identity is claimable because
    the load's structural zeros are eliminated before factoring, so a zero load factors the plate's
    own matrix.
    """
    inst = make_suspended_plate(boundary=boundary, sigma=1.0)
    loaded, room = inst.plate, inst.room
    assert room.cut_faces > 0, "the cut must survive a zero-area surface, or this proves less"
    bare = Plate(
        Lx=loaded.Lx, Ly=loaded.Ly, kappa=loaded.kappa, rho=loaded.rho, fs=loaded.fs,
        N=loaded.N, sigma=loaded.sigma, theta=loaded.theta, boundary=boundary,
    )
    inst.port.T = sparse.csr_matrix(inst.port.T.shape)
    inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
    _refactor(inst, 1.0)

    force = 1e-3 * np.random.default_rng(1).standard_normal(loaded.n_live)
    bare.set_state(plate_bump(bare))
    inst.set_state(plate_bump(loaded))
    for _ in range(60):
        bare.step(force)
        inst.step(force)
        room.step()
    assert np.array_equal(loaded.u, bare.u)
    assert inst.radiated_energy == 0.0


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_string_bridge_plate_room_chain(boundary):
    """``string -> bridge -> suspended plate -> room``, no ``connection.py`` edit, guard **safe**.

    And this retires batch 3's own prediction. ``test_string_bridge_plate_room_chain`` there says
    the two-sided dipole plate, "whose face cut removes air mass", would make the load
    non-dissipative and fail loudly here. It does not, and the reasoning does not survive: the face
    cut removes air inertia from the **room's** ledger, where it never was part of ``G0``,
    while the load itself stays proportional to ``u^{n+1} - u^{n-1}`` — dissipative, merely doubled
    — so it enters ``A`` and never ``G0``. The margin comes out bit-identical, and it is the *same*
    margin batch 3 measured (0.2061806714931906 supported, 0.2061840079056186 free) because the
    guard never saw either load.
    """
    inst = make_suspended_plate(boundary=boundary, walls=WALLS["all-lossy"])
    loaded = inst.plate
    bare = Plate(
        Lx=loaded.Lx, Ly=loaded.Ly, kappa=loaded.kappa, rho=loaded.rho, fs=loaded.fs,
        N=loaded.N, boundary=boundary,
    )
    strings = [
        IdealString(L=0.6, T=60.0, rho=0.005, fs=loaded.fs, N=40, boundary=("fixed", "free"))
        for _ in range(2)
    ]
    bridge_bare = StringPlateBridge(string=strings[0], plate=bare, K=800.0)
    bridge = StringPlateBridge(string=strings[1], plate=inst, K=800.0)
    assert bridge.stability_margin == bridge_bare.stability_margin

    xs = np.linspace(0.0, strings[1].L, strings[1].N + 1)
    strings[1].set_state(1e-3 * np.sin(np.pi * xs / strings[1].L))
    e0 = bridge.energy() + inst.room.energy()
    worst = 0.0
    for _ in range(600):
        bridge.step()
        inst.room.step()
        worst = max(worst, abs(bridge.energy() + inst.room.energy() - e0))
    assert worst <= 1e-11 * abs(e0)
    assert abs(inst.radiated_energy) > 1e-4 * abs(e0)


def test_two_suspended_plates_share_one_room():
    """N instruments in one room, inherited — and now they must have disjoint **cuts** as well."""
    room = make_surface_room(N=(12, 11, 12))
    lower = make_suspended_plate(room=room, index=3)
    upper = make_suspended_plate(room=room, index=8)
    lower.set_state(plate_bump(lower.plate))
    upper.set_state(plate_bump(upper.plate))
    assert room.cut_faces == lower.port.face_count + upper.port.face_count
    e0 = surface_scene_energy(lower, upper)
    worst = 0.0
    for _ in range(200):
        lower.step()
        upper.step()
        room.step()
        worst = max(worst, abs(surface_scene_energy(lower, upper) - e0))
    assert worst <= DRIFT_TOL * abs(e0)
    total = lower.radiated_energy + upper.radiated_energy
    assert abs(total - room.injected) <= LEDGER_TOL * abs(total)


# -- the shared spreading refactor -----------------------------------------------------------

# Captured from SurfacePort BEFORE the batch-4 refactor moved `_spread` and the refusals into
# `_PatchPort`. Each entry is (sum, index-weighted sum) of a data array -- a checksum that a
# reordering would break -- plus the run's end state after 200 coupled steps.
SURFACE_GOLDEN = {
    "supported": dict(
        node_count=20, nnz_T=182, T=(0.06890625, 6.304921875),
        load=(11.019939263311961, 7912.316391057989),
        radiated=0.22569076780460062, p=-682.9102360410334,
    ),
    "free": dict(
        node_count=30, nnz_T=306, T=(0.09, 13.814999999999998),
        load=(14.294317013847094, 17867.896267308868),
        radiated=0.20582260950187845, p=-304.87757961997477,
    ),
    "lossy": dict(
        node_count=20, nnz_T=182, T=(0.06890625, 6.304921875),
        load=(9.393020405437131, 6744.18865110386),
        radiated=0.4800487590186858, p=271.37761865380895,
    ),
    "nearest": dict(
        node_count=12, nnz_T=49, T=(0.06890624999999999, 1.72265625),
        load=(15.89495519000725, 2034.5542643209278),
        radiated=0.8630548593255223, p=-3.4930675607618014,
    ),
    "offcentre": dict(
        node_count=20, nnz_T=196, T=(0.06890625, 6.549605168643696),
        load=(10.931278651909219, 6708.01901861878),
        radiated=0.3114054166233353, p=-772.3358792223344,
    ),
    "y1face": dict(
        node_count=30, nnz_T=306, T=(0.09, 13.815000000000001),
        load=(14.294317013847095, 17867.89626730887),
        radiated=0.15219518846696697, p=-254.30039916325904,
    ),
}
SURFACE_CASES = {
    "supported": dict(boundary="supported"),
    "free": dict(boundary="free"),
    "lossy": dict(boundary="supported", walls={"z0": impedance_from_zeta(3.0)}),
    "nearest": dict(boundary="supported", spreading="nearest"),
    "offcentre": dict(boundary="supported", origin=(0.11, 0.13)),
    "y1face": dict(boundary="free", face="y1"),
}


@pytest.mark.parametrize("case", list(SURFACE_CASES))
def test_surface_port_is_unchanged_across_the_shared_spreading_refactor(case):
    """Batch 3's :class:`SurfacePort` is unchanged by batch 4's refactor, which moved the spreading
    operator and four refusals into a shared base so the interior port could reuse them.

    **This was an ``==`` bit-identity test and is now a tolerance test, deliberately.** The goldens
    below were captured on Windows; on Linux ``load`` differs in the last ULP
    (``6708.019018618780`` vs ``…779``) and ``radiated`` by ~1.7e-14 relative after 200 coupled
    steps. Under ``==`` that is a *platform* assertion, not a refactor assertion, and it failed CI
    on every run while passing on the machine that wrote it — the worst possible split, because the
    green side is the side nobody ships from.

    **``p`` is toleranced against the FIELD's scale, not its own — because it is a cancelling sum.**
    ``sum(p)`` is ~1e2 out of ``sum|p|`` ~1e6, a cancellation factor of 650–360000 across these six
    cases, so its roundoff is set by the *terms* and reading it as a fraction of the *result*
    inflates it by exactly that factor. Measured: the same three Linux deviations are 8.0e-11,
    3.3e-10 and 1.9e-10 relative to ``sum(p)`` — and 3.8e-14, 9.1e-16 and 1.1e-13 relative to
    ``sum|p|``, i.e. plain last-ULP like everything else here. ``nearest`` is the worst offender
    under the wrong denominator and the *best* under the right one, which is the tell. A tolerance
    is only meaningful next to the scale the error is actually generated at.

    **Say what the tolerance no longer catches:** a pure summation *reordering* of the stepping loop
    can move ``radiated``/``p`` by less than 1e-11 and would now pass. What still carries that claim
    is the structure, asserted exactly: ``node_count`` and ``nnz_T`` are integers, and the
    index-weighted digest ``sum_i a_i * i`` moves by *order unity* under any permutation of the data
    — so a reordering of ``T`` or of the load matrix is still caught outright. The run-end values
    are regression detection now, not a bit-identity proof; the refactor they were written to guard
    has shipped, and this is what survives it honestly.
    """
    def digest(a):
        a = np.asarray(a, dtype=float).ravel()
        return float(a.sum()), float(np.dot(a, np.arange(1, a.size + 1)))

    want = SURFACE_GOLDEN[case]
    inst = make_room_loaded_plate(**SURFACE_CASES[case])
    port = inst.port
    # Exact: integers, and the permutation detector.
    assert (port.node_count, port.T.nnz) == (want["node_count"], want["nnz_T"])
    assert digest(port.T.data) == pytest.approx(want["T"], rel=1e-12)
    assert digest(port.load_matrix.data) == pytest.approx(want["load"], rel=1e-12)
    inst.set_state(plate_bump(inst.plate))
    for _ in range(200):
        inst.step()
        inst.room.step()
    # 200 steps of accumulation, so one decade looser than the construction-time digests.
    assert inst.radiated_energy == pytest.approx(want["radiated"], rel=1e-11)
    # ... and `p` against the field's own scale, since the sum cancels by up to 3.6e5 (see above).
    assert abs(float(inst.room.p.sum()) - want["p"]) <= 1e-11 * float(np.abs(inst.room.p).sum())


# -- refusals ---------------------------------------------------------------------------------


def test_refuses_a_sample_rate_mismatch():
    room = make_surface_room(fs=8000.0)
    plate = Plate(Lx=0.30, Ly=0.30, kappa=20.0, rho=0.5, fs=9000.0, N=8)
    with pytest.raises(ValueError, match="sample-rate mismatch"):
        RoomSuspendedPlate(plate=plate, room=room, plane="z", index=4)


@pytest.mark.parametrize("index", [0, 8])
def test_refuses_a_surface_whose_node_planes_are_not_both_interior(index):
    """A node plane on a wall carries half the node weight and the wall's admittance, so ``R_j``
    would differ between the two sides and the load would stop being ``2 T^T R T``."""
    with pytest.raises(ValueError, match="out of range"):
        make_suspended_plate(index=index)


def test_refuses_a_footprint_reaching_the_rim():
    with pytest.raises(ValueError, match="rim"):
        make_suspended_plate(L=0.85, N=16)


def test_refuses_a_footprint_outside_the_plane():
    with pytest.raises(ValueError, match="outside the plane"):
        make_suspended_plate(origin=(0.9, 0.4))


def test_refuses_a_surface_too_coarse_for_the_air_grid():
    with pytest.raises(ValueError, match="fed by no surface node"):
        make_suspended_plate(boundary="free", N=3, spreading="nearest")


def test_refuses_overlapping_node_sets():
    room = make_surface_room()
    make_suspended_plate(room=room, index=4)
    with pytest.raises(ValueError, match="shares node"):
        make_suspended_plate(room=room, index=5)


def test_refuses_a_patch_sharing_cut_faces_with_a_hand_placed_cut():
    """A genuinely new failure mode, and not one the node check covers.

    Two ports can have disjoint *node* sets and overlapping *cuts*, because the node sets live on
    different planes while the cuts can share one. Here the collision is with a hand-placed
    partition — the cut is additive, so nothing is clobbered; it is refused because a port's cut and
    its ``-q``/``+q`` pair are two halves of one object.
    """
    room = make_surface_room()
    room.add_cut("z", 4)
    with pytest.raises(ValueError, match="shares face"):
        make_suspended_plate(room=room, index=4)


def test_refuses_solving_twice_without_a_room_step():
    inst = make_suspended_plate()
    inst.step()
    with pytest.raises(RuntimeError, match="twice within one room step"):
        inst.step()


def test_refuses_an_unknown_plane_and_spreading():
    with pytest.raises(ValueError, match="unknown plane"):
        make_suspended_plate(plane="z0")
    with pytest.raises(ValueError, match="unknown spreading"):
        make_suspended_plate(spreading="cubic")


def test_refuses_a_wrongly_shaped_injection():
    """``q`` is per **face**, which is half the node count — the two planes share one ``q``."""
    inst = make_suspended_plate()
    with pytest.raises(ValueError, match="per-FACE"):
        inst.port.inject(np.zeros(inst.port.node_count))


def test_the_interior_port_can_never_touch_an_open_face():
    """No open-face refusal is needed, and that is provable rather than an omission: the rim and
    index refusals keep every patch node strictly interior on all three axes."""
    room = make_surface_room(walls={"z0": "open", "x1": "open"})
    inst = make_suspended_plate(room=room)
    for axis in range(3):
        assert np.all(inst.port.nodes[axis] > 0)
        assert np.all(inst.port.nodes[axis] < room.N[axis])
    assert isinstance(inst.port, InteriorSurfacePort)
