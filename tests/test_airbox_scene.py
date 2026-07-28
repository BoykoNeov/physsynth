"""What the room can do that no lumped load can: give energy back late, and let two bodies hear
each other.

The *physical* tier for air-box batch 2. The structural tests
(:mod:`tests.test_airbox_port`) prove the coupling is exact and passive; they would all still pass
if ``pbar_free`` were accidentally always zero, because the port would then degenerate into exactly
:class:`~physsynth.core.radiation.RadiatedBody` with ``R = R_room`` — passive, perfectly
conservative, and **silently reflection-free**. These tests are the ones that prove the free field
is read at all, which is why they are not garnish.

``RationalAirLoad`` is a causal one-port with no memory of geometry, so its impulse response is a
decaying exponential: it can shape *how much* the air takes and *when* in frequency, never *when in
time*. A room returns the body's own wave ``2d/c0`` later, from a direction. That is the claim.

**Both arrival oracles are discrete, and getting that wrong would fail them for reasons having
nothing to do with the coupling:**

* The reflection returns after ``2d + 1`` steps for a port ``d`` nodes from the wall — the ``+1``
  being the injection's own step. Asserted as **bit-identity** between two rooms differing *only* in
  ``Lz``: identical before, different immediately after. Both halves matter, and the body's energy
  is *not* monotone in the meantime (the near-field reactance hands energy back every cycle), so a
  monotonicity assertion here would simply be false.
* A second body first moves at the **Manhattan** distance in nodes, not ``r/c0``. The 7-point
  stencil spreads influence one node along one axis per step, so the numerical domain of dependence
  after ``m`` steps is the ``L1`` ball of radius ``m`` and an off-axis listener gets a
  machine-precision *precursor* before the physical wavefront. Measured: 6 nodes apart, arrival at
  step 6 against 11.5 for ``r/c0``.

The cross-tier test then says what a port *is*. At low ``ka`` the load is an added mass,
``Z ~ j omega M_a`` with ``M_a = rho0 / (4 pi a_eff)``, so ``a_eff = rho0 / (4 pi M_a)`` is one
number characterising the port independently of the room's modal wiggle. A **point** port's is
``~ h/3.1``, so it *halves* every time the grid is refined; a **spread** port's barely moves. That
contrast — a factor of twenty in grid sensitivity — is the assertion, and both halves matter.

The spread port's *absolute* size is a separate claim with a separate trap, and the trap is the
room: the same port reads 8.6% above the ball's ``5a/6`` in a room ten times its radius and 0.3%
above it in a room twenty times its radius. The excess is the **room's** reactance, not the port's,
so the closed form is asserted where the port is compact and the small room's excess is asserted
too, as the attribution.
"""

import numpy as np
import pytest
from helpers import make_room_loaded_body, room_scene_energy

from physsynth.core.airbox import C0_AIR, RHO0_AIR, AirBox, RoomPort, impedance_from_zeta

DRIFT_TOL = 1e-10
H = 0.05  # the scene grid: 5 cm nodes, so a travel time is a countable number of steps


def _history(nz, steps=40):
    """A room identical in every way except ``Lz``; return the body's modal history."""
    inst = make_room_loaded_body(L=(0.5, 0.4, nz * H), h=H, at=(0.15, 0.15, 0.15))
    hist = []
    for _ in range(steps):
        inst.step()
        inst.room.step()
        hist.append(inst.body.q.copy())
    return np.array(hist)


def _first_difference(a, b):
    differs = np.any(a != b, axis=1)
    return int(np.argmax(differs)) if differs.any() else None


# -- 8. The room gives energy back, and the delay is right ---------------------------------------
@pytest.mark.parametrize("nz, d", [(6, 3), (8, 5), (10, 7)])
def test_reflection_returns_at_the_round_trip_time(nz, d):
    """Bit-identity is the right instrument here, and it tests both directions at once: the
    identical prefix catches a coupling that arrives **early** (or a free pressure that is really
    just a local self-term), and the difference immediately afterwards catches one that never
    arrives at all.

    The reference room is deep enough (``Nz = 12``) that its own z1 reflection is still in transit
    throughout, so the *only* thing that can make the histories differ is the near wall. That
    margin is four steps -- the reference's own round trip lands at ``2*9 + 1 = 19`` against a
    largest asserted ``t`` of 15 -- so a deeper case added to the list above needs a deeper
    reference, not just a bigger number.
    """
    reference = _history(12)
    hist = _history(nz)
    t = _first_difference(hist, reference)
    # The port sits d nodes from the z1 wall; influence needs d steps out and d back, plus the
    # injection's own step. Measured 2d+1 at d = 3, 5 and 7 -- three geometries, one law.
    assert t == 2 * d + 1
    assert np.array_equal(hist[:t], reference[:t])  # bit-identical BEFORE, not merely close
    assert not np.array_equal(hist[t], reference[t])


def test_reflection_is_not_a_lumped_load():
    """The delay scales with the distance -- which is the whole point, and what a one-port cannot
    do at any order. A load with no memory of geometry would give the same first-difference step
    for every ``Lz``."""
    reference = _history(12)
    steps = {nz: _first_difference(_history(nz), reference) for nz in (6, 8, 10)}
    assert list(steps.values()) == [7, 11, 15]
    deltas = np.diff(list(steps.values()))
    assert (deltas == 4).all()  # 2 nodes further away costs exactly 4 steps, out and back


def test_the_room_hands_energy_back():
    """The channel itself runs **both ways**, and that is what separates this tier from the lumped
    one. ``RadiatedBody`` books ``k R U^2 >= 0``: monotone by construction, energy only ever leaves.
    A port books ``k pbar U`` with ``pbar`` carrying the room's returning field, so
    ``radiated_energy`` genuinely *decreases* on some steps -- the room pushing the body along
    rather than damping it.

    This is also why oracle 8 is written as bit-identity: with the sign of the increment free, no
    monotonicity statement about the body is available to assert on. Only the scene total is flat.
    """
    inst = make_room_loaded_body()
    room = inst.room
    e0 = room_scene_energy(inst)
    hist = []
    for _ in range(400):
        inst.step()
        room.step()
        hist.append(inst.radiated_energy)
        assert abs(room_scene_energy(inst) - e0) < DRIFT_TOL * abs(e0)
    increments = np.diff(hist)
    assert increments.max() > 0.0  # the body drives the room ...
    assert increments.min() < 0.0  # ... and the room drives the body back


# -- 9. Two instruments, one room ----------------------------------------------------------------
@pytest.mark.parametrize(
    "at_a, at_b",
    [((0.10, 0.20, 0.15), (0.40, 0.20, 0.15)), ((0.10, 0.10, 0.10), (0.35, 0.30, 0.25))],
)
def test_second_body_moves_only_when_the_sound_arrives(at_a, at_b):
    """B is at rest and stays **exactly** zero -- not small -- until A's disturbance can reach it.

    The oracle is the arrival index; the amplitude is a sanity check, not an assertion. And the
    index is **Manhattan**: an assertion written against ``r/c0`` would fail by up to 2x for a
    reason having nothing to do with the back-reaction.
    """
    a = make_room_loaded_body(at=at_a)
    b = make_room_loaded_body(room=a.room, at=at_b, q0=0.0)
    room = a.room
    manhattan = sum(abs(x - y) for x, y in zip(a.port.index, b.port.index, strict=True))
    euclid_steps = H * np.sqrt(
        sum((x - y) ** 2 for x, y in zip(a.port.index, b.port.index, strict=True))
    ) / C0_AIR * room.fs

    first = None
    e0 = room_scene_energy(a, b)
    for n in range(2 * manhattan + 10):
        a.step()
        b.step()
        room.step()
        if first is None and np.any(b.body.q != 0.0):
            first = n
        assert abs(room_scene_energy(a, b) - e0) < DRIFT_TOL * abs(e0)
    assert first == manhattan
    assert first < euclid_steps  # the grid's precursor, arriving before the physical wavefront


def test_disjoint_ports_are_exactly_independent():
    """The N-instrument scope and the overlap refusal are the same measurement, read in two
    directions: disjointness is exactly the condition that makes each port's scalar solve exact."""
    a = make_room_loaded_body(at=(0.15, 0.15, 0.15), q0=np.array([1e-3, 5e-4]))
    b = make_room_loaded_body(room=a.room, at=(0.35, 0.25, 0.15), q0=np.array([7e-4, 2e-4]))
    room = a.room
    e0 = room_scene_energy(a, b)
    lo = hi = e0
    for _ in range(400):
        a.step()
        b.step()
        room.step()
        e = room_scene_energy(a, b)
        lo, hi = min(lo, e), max(hi, e)
    assert (hi - lo) / abs(e0) < DRIFT_TOL


def test_port_solve_order_does_not_matter():
    """A consequence of the same fact, and the reason ``free_pressure`` may ignore queued
    injections: with disjoint ports the scene is **bit-identical** whichever order the ports
    solve in. Were they overlapping, B would see A and A would never see B."""
    def run(reverse):
        a = make_room_loaded_body(at=(0.15, 0.15, 0.15), q0=np.array([1e-3, 5e-4]))
        b = make_room_loaded_body(room=a.room, at=(0.35, 0.25, 0.15), q0=np.array([7e-4, 2e-4]))
        order = [b, a] if reverse else [a, b]
        for _ in range(120):
            for inst in order:
                inst.step()
            a.room.step()
        return a.body.q.copy(), b.body.q.copy(), a.room.p.copy()

    forward, backward = run(False), run(True)
    for x, y in zip(forward, backward, strict=True):
        assert np.array_equal(x, y)


# -- 10. Cross-tier: what a port IS --------------------------------------------------------------
#    A Gaussian volume-velocity pulse driven straight through the port's own two numbers, so this
#    exercises R_room as well as the geometry. Two rooms, on purpose: a RATIO survives a small cheap
#    room, a MAGNITUDE does not (the small room's own reactance is the bigger term -- measured).
_SWEEP_H = (0.027, 0.0135)   # one halving
_SWEEP_FREQS = np.array([50.0, 75.0, 100.0])
_BALL_RADIUS = 0.05
_CONTRAST_ROOM = 0.5         # cheap, and adequate for a ratio
_COMPACT_ROOM = 1.0          # a / L = 0.05: compact, which is what the closed form wants


def _equivalent_radius(side, h, radius, cfl=0.9):
    """Drive a port with a pulse, read ``Z = pbar/q``, and turn its reactance into a radius.

    ``M_a = Im Z / omega`` and ``a_eff = rho0 / (4 pi M_a)`` -- one number saying what the port
    *is*, as a sphere. Reactance rather than ``|Z|`` on purpose: ``|Z|`` carries the room's modal
    wiggle and the near-field mass does not.
    """
    fs = C0_AIR * np.sqrt(3.0) / (cfl * h)
    room = AirBox(L=(side,) * 3, fs=fs, h=h, walls=impedance_from_zeta(1.0))
    port = RoomPort(room=room, at=tuple(0.5 * v for v in room.L_actual), radius=radius)
    n_steps = int(round(0.04 * fs))
    t = np.arange(n_steps) / fs
    q = np.exp(-0.5 * ((t - 4.0e-3) / 6.0e-4) ** 2)

    pbar = np.empty(n_steps)
    for n in range(n_steps):
        pbar[n] = port.free_pressure() + port.R_room * q[n]
        port.inject(q[n])
        room.step()

    P, Q = np.fft.rfft(pbar), np.fft.rfft(q)
    f = np.fft.rfftfreq(n_steps, 1.0 / fs)
    ratio = P / Q
    Z = np.interp(_SWEEP_FREQS, f, ratio.real) + 1j * np.interp(_SWEEP_FREQS, f, ratio.imag)
    m_a = Z.imag / (2.0 * np.pi * _SWEEP_FREQS)
    return float(np.mean(RHO0_AIR / (4.0 * np.pi * m_a)))


def test_point_port_load_is_a_grid_quantity_and_the_spread_port_is_not():
    """The measured non-convergence, shipped as one. Both halves are the assertion.

    A point port behaves as a sphere of radius ``~ h/3.1``, so refining the grid **halves** its
    equivalent radius and doubles the added mass it hangs on the body: refinement makes the artifact
    *worse*. A fixed-radius ball does not move. The energy identity is exact either way -- that
    claim is structural and stands -- but the *magnitude* of a point port's load is a property of
    the grid, not of the physics, which is why ``radius`` has no default.
    """
    point = [_equivalent_radius(_CONTRAST_ROOM, h, None) for h in _SWEEP_H]
    spread = [_equivalent_radius(_CONTRAST_ROOM, h, _BALL_RADIUS) for h in _SWEEP_H]

    # A point port IS the grid: a_eff / h is the same constant on both, so a_eff halves with h.
    for a_eff, h in zip(point, _SWEEP_H, strict=True):
        assert 0.30 < a_eff / h < 0.34            # measured 0.324 and 0.320
    assert 0.45 < point[1] / point[0] < 0.55      # measured 0.493 and 0.496 (three grids)

    # A fixed-radius ball is not. THE CONTRAST IS THE ASSERTION -- a factor of 20 in grid
    # sensitivity -- and the residual few percent is left alone deliberately (see the magnitude
    # test below for why a tighter bound here would be measuring the room, not the port).
    assert 0.95 < spread[1] / spread[0] < 1.10    # measured 1.045 and 1.038 (three grids)
    assert spread[0] > 4.0 * point[0]             # the two tiers are nowhere near each other


def test_spread_port_matches_the_uniformly_injecting_ball():
    """The magnitude, and the room is the thing that has to be got out of the way first.

    A uniformly injecting **ball** is not a pulsating shell: its volume-averaged self-pressure
    carries the classic **6/5** factor (the same one as the mean potential of a uniformly charged
    sphere), so its equivalent shell radius is ``5a/6`` and an assertion against
    ``RationalAirLoad.from_sphere(a)`` would be wrong by design.

    The room matters more than the shape factor does. Measured at fixed ``h`` and a fixed port,
    ``a_eff / (5a/6)`` reads 1.086, 1.040, **1.003**, 0.977 for rooms of 0.5, 0.7, 1.0 and 1.4 m:
    the small room's own reactance, not the port's. So the closed form is asserted where the port is
    compact, and the small room's excess is asserted too -- as the attribution, not as an error.
    """
    ball = 5.0 * _BALL_RADIUS / 6.0
    compact = _equivalent_radius(_COMPACT_ROOM, 0.027, _BALL_RADIUS)
    small = _equivalent_radius(_CONTRAST_ROOM, 0.027, _BALL_RADIUS)
    assert abs(compact / ball - 1.0) < 0.05  # measured 1.0034 -- the 6/5 shape factor, confirmed
    assert small > 1.05 * compact            # and the small room reads high, by its own reactance


def test_the_reactance_is_not_a_dispersion_artifact():
    """``a_eff`` is a *static* near-field quantity, so it must not depend on the Courant number --
    and this is worth asserting rather than assuming, because 3-D has no dispersionless ``lambda``
    and the ceiling is where batch 1 measured the corner mode going defective. Measured across
    ``cfl`` = 0.5, 0.7, 0.9 and 0.998 the answer moves in the **fifth** significant figure."""
    slow = _equivalent_radius(_CONTRAST_ROOM, 0.027, _BALL_RADIUS, cfl=0.5)
    fast = _equivalent_radius(_CONTRAST_ROOM, 0.027, _BALL_RADIUS, cfl=0.9)
    assert abs(fast / slow - 1.0) < 1e-4  # measured 45.231 vs 45.233 mm
