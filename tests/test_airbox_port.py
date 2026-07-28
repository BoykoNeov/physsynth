"""The two-way body<->room port: exact conservation, the Thevenin constant, and the refusals.

The structural tier for air-box batch 2. The claim under test, in one line: **the room load is exact
and passive — what the body loses, the room gains, to machine precision, for any wall, any port
position, any number of instruments.** Not "approximately conserves". Each port's
``radiated_energy`` *is* the room's ``injected``, seen from the other side of the same terminal, so
summing the two ledgers cancels the coupling term identically and the conserved statement

    sum_j inst_j.energy() + room.energy()

contains no coupling term at all. That is what makes a drift in it unambiguous evidence of a bug
rather than of accounting — and it works only because the room books its side from its **own**
post-closure pressure, never from a number the port hands back.

Two traps get their own tests here because nothing else would catch them:

* ``R_room`` must carry the wall-closure factor ``1/(1 + beta)``. ``AirBox.step`` injects *before*
  it closes the wall, so a port on a lossy wall is divided by ``1 + beta`` along with everything
  else. Without the factor an interior-port suite stays perfectly green and a wall-mounted port
  leaks ~2% per run (measured 1.9e-2 against 8.4e-15). ``test_R_room_is_what_the_room_does``
  measures it **differentially** — comparing the coupled step's ``pbar`` against
  ``pbar_free + R_room U`` would be a tautology, since that expression is how ``pbar`` was computed,
  and would pass for any ``R_room`` whatsoever.
* The port's local ``O(port)`` free-pressure read must replicate the full-array
  ``_divergence()``-then-closure exactly, *including* at wall, edge and corner nodes where a node
  sees only the faces it has. ``test_free_pressure_matches_full_array`` asserts bit-identity there.

And one refusal that the energy report is structurally blind to: a port on an ``open`` face is
perfectly conservative and completely silent (``injected`` and ``acoustic`` exactly zero, drift
8.6e-15) — the physics is exactly right and exactly useless, so it must raise at construction.
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_PORT_AT_DEFAULT,
    make_airbox,
    make_room_loaded_body,
    room_scene_energy,
)

from physsynth.core.airbox import AirBox, RoomLoadedBody, RoomPort, impedance_from_zeta
from physsynth.core.body import ModalBody
from physsynth.core.connection import StringBodyBridge
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-10  # acceptance criterion 1 -- the same bar as every other resonator
Z_MATCHED = impedance_from_zeta(1.0)

# Port sites in the default 0.5 x 0.4 x 0.3 m room at h = 5 cm: interior, on a face, on an edge,
# in a corner. Every one of them is a legitimate place to mount a loudspeaker, and each exercises a
# different number of summed wall admittances in beta.
INTERIOR = AIRBOX_PORT_AT_DEFAULT
ON_WALL = (0.0, 0.15, 0.15)
ON_EDGE = (0.0, 0.0, 0.15)
IN_CORNER = (0.0, 0.0, 0.0)


def _drift(instruments, steps=400):
    """Relative spread of the scene total over a run -- the primary bug detector."""
    room = instruments[0].room
    e0 = room_scene_energy(*instruments)
    lo = hi = e0
    for _ in range(steps):
        for inst in instruments:
            inst.step()
        room.step()
        e = room_scene_energy(*instruments)
        lo, hi = min(lo, e), max(hi, e)
    return (hi - lo) / abs(e0)


# -- 1-2. Exact conservation ---------------------------------------------------------------------
#    Rigid walls isolate the coupling channel entirely: the ONLY place energy can go is the room,
#    so a leak has nowhere to hide. The lossy cases then add the wall channel on top, and the ones
#    with the port ON a lossy wall are the ones that need the 1/(1+beta) factor in R_room.
@pytest.mark.parametrize("at", [INTERIOR, ON_WALL, ON_EDGE, IN_CORNER])
def test_conserved_rigid_room(at):
    assert _drift([make_room_loaded_body(at=at)]) < DRIFT_TOL


@pytest.mark.parametrize("at", [INTERIOR, ON_WALL, ON_EDGE, IN_CORNER])
def test_conserved_lossy_walls(at):
    """The port sits on a lossy wall in three of these four -- trap 6.1's home ground."""
    assert _drift([make_room_loaded_body(at=at, walls=Z_MATCHED)]) < DRIFT_TOL


@pytest.mark.parametrize("walls", ["rigid", Z_MATCHED, {"x0": Z_MATCHED, "y1": 100.0}])
def test_conserved_spread_port(walls):
    """A spread port is many nodes with differing W and beta -- the weighted sums, exercised."""
    inst = make_room_loaded_body(at=(0.15, 0.15, 0.15), radius=0.12, walls=walls)
    assert inst.port.node_count > 1
    assert _drift([inst]) < DRIFT_TOL


def test_conserved_lossy_body():
    """With sigma > 0 the body is no longer lossless, so the scene total DECREASES rather than
    staying flat -- passivity, the weaker statement that still has to hold."""
    inst = make_room_loaded_body(sigmas=40.0)
    room = inst.room
    e0 = prev = room_scene_energy(inst)
    for _ in range(400):
        inst.step()
        room.step()
        e = room_scene_energy(inst)
        assert e <= prev + 1e-14 * abs(e0)  # monotone to within a rounding ulp
        prev = e
    assert prev < 0.9 * e0  # and it genuinely decayed, rather than merely failing to grow


# -- The local free-pressure read ----------------------------------------------------------------
def test_free_pressure_matches_full_array():
    """``RoomPort.free_pressure`` is ``O(port)``; the room's own update is a full-array operation.

    They must agree **bit for bit**, and the interesting nodes are the ones where ``_divergence``
    gives a node only the faces it actually has (that absence *is* the rigid closure). An off-by-one
    in the local read would be a small, plausible, position-dependent error -- exactly the kind that
    survives an energy test, because the port and the room would still agree with each other.
    """
    for walls in ("rigid", Z_MATCHED):
        for at in (INTERIOR, ON_WALL, ON_EDGE, IN_CORNER):
            inst = make_room_loaded_body(at=at, walls=walls)
            room, port = inst.room, inst.port
            for _ in range(17):  # a field with structure, not a symmetric one
                inst.step()
                room.step()
            p_old = room.p
            p_full = p_old - room.k * room.rho0 * room.c0**2 * room._divergence()
            if room._has_walls:
                p_full = (p_full - room._beta * p_old) / (1.0 + room._beta)
            want = float(np.sum(port.w * 0.5 * (p_full[port.nodes] + p_old[port.nodes])))
            assert port.free_pressure() == want, f"{walls} at {at}"


def test_free_pressure_matches_full_array_spread():
    """Same, for a ball straddling a corner -- clipped, one-sided, every weight different."""
    inst = make_room_loaded_body(at=(0.0, 0.0, 0.0), radius=0.12, walls=Z_MATCHED)
    room, port = inst.room, inst.port
    assert 1 < port.node_count < room.p.size  # genuinely a clipped ball
    for _ in range(17):
        inst.step()
        room.step()
    p_old = room.p
    p_full = (p_old - room.k * room.rho0 * room.c0**2 * room._divergence())
    p_full = (p_full - room._beta * p_old) / (1.0 + room._beta)
    want = float(np.sum(port.w * 0.5 * (p_full[port.nodes] + p_old[port.nodes])))
    assert port.free_pressure() == want


# -- 3. R_room is what the room actually does ----------------------------------------------------
def _snapshot(room):
    return (
        room.p.copy(), room.ux.copy(), room.uy.copy(), room.uz.copy(),
        room.ux_prev.copy(), room.uy_prev.copy(), room.uz_prev.copy(),
        room.dissipated, room.injected, room.n,
    )


def _restore(room, snap, port):
    (room.p, room.ux, room.uy, room.uz,
     room.ux_prev, room.uy_prev, room.uz_prev,
     room.dissipated, room.injected, room.n) = (a.copy() if hasattr(a, "copy") else a
                                                for a in snap)
    room._pending.clear()
    room._pending_ports.clear()
    port.reset()


@pytest.mark.parametrize("at", [INTERIOR, ON_WALL, ON_EDGE, IN_CORNER])
@pytest.mark.parametrize("walls", ["rigid", Z_MATCHED])
def test_R_room_is_what_the_room_does(at, walls):
    """**Differential**, not definitional: step the room twice from an identical saved state, once
    with ``q = 0`` and once with ``q = U``, and read the incremental centered pressure per unit
    volume velocity straight off the room. Nothing in this test consults ``R_room`` except the final
    comparison, so it catches a wrong constant directly instead of waiting for a drift to build.
    """
    inst = make_room_loaded_body(at=at, walls=walls)
    room, port = inst.room, inst.port
    for _ in range(23):
        inst.step()
        room.step()
    snap = _snapshot(room)

    def pbar_after(q):
        _restore(room, snap, port)
        p_old = room.p.copy()
        port.inject(q)
        room.step()
        return float(np.sum(port.w * 0.5 * (room.p[port.nodes] + p_old[port.nodes])))

    u = 3.7e-4
    measured = (pbar_after(u) - pbar_after(0.0)) / u
    assert abs(measured - port.R_room) <= 1e-12 * port.R_room
    _restore(room, snap, port)


def test_R_room_wall_factor_is_not_free():
    """Pin the trap: on a lossy wall the naive ``k rho c^2 / (2 W)`` differs from the truth by
    exactly ``1 + beta``, so this is not a factor that "cancels anyway"."""
    inst = make_room_loaded_body(at=IN_CORNER, walls=Z_MATCHED)
    room, port = inst.room, inst.port
    naive = float(np.sum(port.w**2 * room.k * room.rho0 * room.c0**2 / (2.0 * room._W[port.nodes])))
    beta = float(room._beta[port.nodes][0])
    assert beta > 0.5  # a corner sums three admittances: the factor is a big one
    assert np.isclose(naive / port.R_room, 1.0 + beta, rtol=1e-13)


# -- 4. The reduction ----------------------------------------------------------------------------
def test_zero_radiation_is_bit_identical_to_bare_body():
    """``a = 0`` -> ``G = 0``, ``U = 0``, nothing injected: the family's reduction-ledger entry,
    asserted as **bit**-identity (``array_equal``), not closeness. This is the check that catches
    sign errors nothing else catches."""
    inst = make_room_loaded_body(radiation=0.0)
    bare = ModalBody(
        freqs=inst.body.freqs, fs=inst.body.fs, sigmas=inst.body.sigma,
        masses=inst.body.m, phi=inst.body.phi, radiation=0.0,
    )
    bare.set_state(inst.body.q.copy(), 0.0)
    bare.q_prev = inst.body.q_prev.copy()
    for n in range(200):
        inst.step(force=0.3 * np.sin(0.07 * n))
        inst.room.step()
        bare.step(force=0.3 * np.sin(0.07 * n))
        assert np.array_equal(inst.body.q, bare.q), f"step {n}"
        assert np.array_equal(inst.body._accel, bare._accel), f"accel at step {n}"
    assert inst.volume_velocity == 0.0
    assert inst.radiated_energy == 0.0
    assert inst.room.injected == 0.0


# -- 5. Unconditional passivity ------------------------------------------------------------------
def test_absurd_coupling_stays_passive():
    """No CFL to find, and the test's job is to prove that rather than trust it.

    Note which direction stresses the solve: ``R_room ~ k / W``, so a *coarse* grid makes ``W``
    large and ``R_room`` **small**. The stress case is a **corner node on a fine grid**, where
    ``W = h^3/8`` is eight times smaller than the interior weight and, with lossy walls, ``beta`` is
    largest too -- and radiation weights inflated by 10^3 on top.
    """
    inst = make_room_loaded_body(
        h=0.02, at=IN_CORNER, walls=Z_MATCHED, radiation=np.array([2.0, 1.3])
    )
    assert inst._G * inst.port.R_room > 1e3  # the solve is genuinely far from the decoupled limit
    peak = 0.0
    e0 = room_scene_energy(inst)
    lo = hi = e0
    for _ in range(300):
        inst.step()
        inst.room.step()
        e = room_scene_energy(inst)
        lo, hi = min(lo, e), max(hi, e)
        peak = max(peak, float(np.max(np.abs(inst.body.q))))
    assert (hi - lo) / abs(e0) < DRIFT_TOL
    assert np.isfinite(inst.room.p).all()
    # Bounded, not monotone: the room couples the two modes, so |q_1| may exceed its own initial
    # value while the scene total does not. A CFL failure here would be orders of magnitude, and
    # secular growth would show as a slow climb -- neither is a few percent.
    assert peak < 5.0 * 1e-3


def test_passivity_across_grids_and_walls():
    """The claim is "any wall, any grid", so sweep both rather than asserting it once."""
    for h in (0.05, 0.025):
        for walls in ("rigid", Z_MATCHED, 20.0, {"x0": 5.0, "z1": Z_MATCHED}):
            inst = make_room_loaded_body(h=h, at=ON_WALL, walls=walls, radius=0.1)
            assert _drift([inst], steps=150) < DRIFT_TOL, f"h={h} walls={walls}"


# -- 6. The refusals -----------------------------------------------------------------------------
def test_open_face_port_is_refused():
    room = make_airbox(walls={"x0": "open"})
    with pytest.raises(ValueError, match="open"):
        RoomPort(room=room, at=(0.0, 0.15, 0.15), radius=None)


def test_open_face_reached_by_a_BALL_is_refused():
    """A port whose *centre* is interior can still reach the face once the ball is laid down --
    so the check runs over the whole node set, never the centre alone."""
    room = make_airbox(walls={"x0": "open"})
    RoomPort(room=room, at=(0.3, 0.3, 0.3), radius=0.12)  # far enough: fine
    with pytest.raises(ValueError, match="open"):
        RoomPort(room=room, at=(0.1, 0.3, 0.3), radius=0.12)


def test_shared_node_is_refused():
    inst = make_room_loaded_body(at=(0.15, 0.15, 0.15))
    with pytest.raises(ValueError, match="shares node"):
        RoomPort(room=inst.room, at=(0.16, 0.16, 0.16), radius=None)  # snaps onto (3, 3, 3)


def test_overlapping_balls_are_refused():
    inst = make_room_loaded_body(at=(0.15, 0.15, 0.15), radius=0.1)
    with pytest.raises(ValueError, match="shares node"):
        RoomPort(room=inst.room, at=(0.3, 0.15, 0.15), radius=0.1)


def test_disjoint_ports_are_accepted():
    inst = make_room_loaded_body(at=(0.15, 0.15, 0.15), radius=0.07)
    second = RoomPort(room=inst.room, at=(0.35, 0.25, 0.15), radius=0.07)
    assert second.node_count > 1
    assert not np.intersect1d(inst.port._flat, second._flat).size


def test_port_outside_the_room_is_refused():
    room = make_airbox()
    with pytest.raises(ValueError, match="outside the room"):
        RoomPort(room=room, at=(5.0, 0.15, 0.15), radius=None)


def test_unresolvable_radius_is_refused():
    """A ball smaller than the grid IS a point port; say so rather than pretending otherwise."""
    room = make_airbox()
    with pytest.raises(ValueError, match="smaller than the grid"):
        RoomPort(room=room, at=(0.15, 0.15, 0.15), radius=0.01)
    with pytest.raises(ValueError, match="positive length"):
        RoomPort(room=room, at=(0.15, 0.15, 0.15), radius=0.0)


def test_forgotten_room_step_raises():
    """The port does not step the room; forgetting to is caught loudly, not silently."""
    inst = make_room_loaded_body()
    inst.step()
    with pytest.raises(RuntimeError, match="twice within one room step"):
        inst.step()
    inst.room.step()
    inst.step()  # and it recovers the moment the room advances


def test_forgotten_room_step_guard_is_per_port():
    """With two instruments in one room the second solves while the first's injection is queued --
    a global "is anything pending" guard would fire on every scene ever built."""
    a = make_room_loaded_body(at=(0.15, 0.15, 0.15))
    b = make_room_loaded_body(room=a.room, at=(0.35, 0.25, 0.15))
    for _ in range(10):
        a.step()
        b.step()  # must NOT raise
        a.room.step()


def test_sample_rate_mismatch_is_refused():
    room = make_airbox()
    body = ModalBody(freqs=[220.0], fs=room.fs * 1.5, masses=0.05, radiation=1e-3)
    with pytest.raises(ValueError, match="sample-rate mismatch"):
        RoomLoadedBody(body=body, room=room, at=INTERIOR, radius=None)


def test_set_state_and_reset_clear_the_coupling_ledger():
    """``__getattr__`` would delegate both to the bare body and leave a stale ledger behind."""
    inst = make_room_loaded_body()
    for _ in range(20):
        inst.step()
        inst.room.step()
    assert inst.radiated_energy != 0.0
    inst.set_state(np.array([1e-3, 0.0]))
    assert (inst.radiated_energy, inst.volume_velocity, inst.port_pressure, inst.n) == (
        0.0, 0.0, 0.0, 0,
    )
    inst.step()  # the pending mark went with it
    inst.room.step()
    inst.reset()
    assert inst.radiated_energy == 0.0
    assert np.array_equal(inst.body.q, np.zeros(2))


def test_energy_is_an_override_not_a_delegation():
    """If ``energy()`` were delegated it would return the bare modal energy -- the total *without*
    its coupling channel, which is the number that looks fine and is not conserved."""
    inst = make_room_loaded_body()
    for _ in range(50):
        inst.step()
        inst.room.step()
    assert inst.energy() != inst.body.energy()
    assert inst.energy() == inst.body.energy() + inst.radiated_energy


def test_room_set_state_unsticks_every_port():
    inst = make_room_loaded_body()
    inst.step()
    inst.room.set_state(np.zeros_like(inst.room.p))
    inst.step()  # must not raise: the room's fresh run owes nothing


# -- 7. Composition ------------------------------------------------------------------------------
def test_string_bridge_body_room_chain_conserves():
    """The full ``string -> bridge -> body -> room`` chain, with ``connection.py`` untouched.

    ``RoomLoadedBody`` is a drop-in for ``ModalBody``, so the bridge calls ``body.step(force=F)``
    without knowing a room exists. Note the caller's loop: the *bridge* owns the body's step, which
    is exactly why a port must not step its own room.
    """
    N, L, T, rho, K = 100, 1.0, 200.0, 0.005, 8000.0
    c = np.sqrt(T / rho)
    fs = c * N / (L * 0.9)
    string = IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    h = 343.0 * np.sqrt(3.0) / (0.9 * fs)
    room = AirBox(L=(10 * h, 8 * h, 6 * h), fs=fs, h=h, walls=Z_MATCHED)
    body = ModalBody(
        freqs=np.array([180.0, 291.0]), fs=fs, masses=0.02, phi=1.0,
        radiation=np.array([3e-3, 2e-3]),
    )
    inst = RoomLoadedBody(body=body, room=room, at=(3 * h, 3 * h, 3 * h), radius=None)
    bridge = StringBodyBridge(string=string, body=inst, K=K)
    string.set_state(triangular_pluck(string.x, string.L, 0.3 * string.L, amplitude=1e-3))

    def total():
        # bridge.energy() already carries inst.energy(), i.e. the body PLUS its radiated channel --
        # the drop-in property, visible in the ledger: adding radiated_energy again double-counts.
        return bridge.energy() + room.energy()

    e0 = total()
    lo = hi = e0
    for _ in range(600):
        bridge.step()
        room.step()
        e = total()
        lo, hi = min(lo, e), max(hi, e)
    assert (hi - lo) / abs(e0) < DRIFT_TOL
    assert inst.radiated_energy != 0.0  # the room was actually driven
