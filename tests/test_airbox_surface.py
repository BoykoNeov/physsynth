"""The **distributed** area coupling: a surface radiating from every node (air-box batch 3).

The claim under test, in one line: **a surface radiates according to the SHAPE of its motion, not
only its net volume displacement** — so a mode with exactly zero net volume velocity, which every
one-port in this repo (``AirRadiation``, ``RadiatedBody``, ``RationalAirLoad``, ``RoomPort``) calls
exactly silent, is not. That is the acoustic short circuit, and it is *structural*: a one-port
couples through a single scalar and has no length scale on its surface, so no ``R(omega)``
reproduces it at any order.

**The conserved total is NOT the money test here, and this file is organised around that.** The
plate's energy identity telescopes to ``-k pbar . q`` for *whatever* pressure was used in the force,
and the room's identity is exact for *whatever* injection it received — so the scene total is the
sum of two separately-exact identities and stays flat even when the two disagree with each other.
Measured (:func:`test_conservation_is_blind_to_a_wrong_R`): dropping the ``1 + beta`` wall factor
from ``R_j`` on a lossy mounting wall leaves the total drifting **4.9e-15, which is smaller than the
correct run's own 2.0e-14** — green, and not even in the suspicious direction — while
``|radiated - injected|`` goes from **exactly 0.00e+00 to 18% of the channel**. So conservation
ships as *necessary and not sufficient*, and the tests that carry the weight are:

* :func:`test_ledgers_agree` — ``radiated_energy == room.injected``, two numbers computed from
  opposite sides of the same terminal, agreeing only if every ``R_j`` is exactly right;
* :func:`test_R_j_is_what_the_room_does` — ``R_j`` measured **differentially** off the room, one
  node at a time, together with the off-diagonal asserted ``== 0.0`` *exactly* (the room's
  instantaneous response over a node set is diagonal, which is what makes the whole coupling cheap);
* :func:`test_volume_is_conserved_exactly` — ``sum_j q_j == sum_n area_n v_n`` with ``v``
  recomputed **from the plate's own state**, never from a number the port stored, because a
  consistently-wrong ``q`` factor would satisfy both ledgers happily;
* :func:`test_coupled_scheme_residual` — the achieved ``u^{n+1}`` put back into the coupled PDE, at
  **two** timesteps (a wrong-but-consistent ``k``-dependent factor passes at one).

Two further traps get their own tests because nothing else would catch them:

* **A consistent sign flip in ``T``/``T^T`` is invisible to every energetic quantity.** ``T^T R T``
  is sign-invariant, so the load matrix, the solve and ``radiated_energy`` come out *bit-identical*
  while the room's field is inverted. :func:`test_sign_convention_is_uniform_over_faces` asserts the
  bit-identity of the **wrong** run: that is the point of it.
* **The surface must be centred**, and for two independent reasons — it is what makes the load
  matrix mirror-equivariant at all, and what lets the scene be symmetric about a mode's own
  antisymmetry plane. Both are measured here, and the second has *no tolerance band*.
"""

import numpy as np
import pytest
from helpers import (
    make_room_loaded_plate,
    make_surface_room,
    plate_bump,
    plate_mode_shape,
    surface_scene_energy,
)
from scipy import sparse
from scipy.sparse.linalg import splu

from physsynth.core.airbox import FACES, RoomLoadedPlate, SurfacePort, impedance_from_zeta
from physsynth.core.connection import StringPlateBridge
from physsynth.core.plate import Plate
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-12   # the scene total, relative -- necessary, not sufficient (see the module head)
LEDGER_TOL = 1e-12  # |radiated - injected| / |radiated| -- the money test
BOUNDARIES = ("supported", "free")
WALLS = {
    "rigid": "rigid",
    "all-lossy": impedance_from_zeta(4.0),
    "lossy-mounting-wall": {"z0": impedance_from_zeta(3.0)},
}


def _run(inst, steps, f_ext=None):
    """Step instrument and room in the contract's order: the port solves, then one room step."""
    for _ in range(steps):
        inst.step(f_ext)
        inst.room.step()


def _seeded(**kwargs):
    inst = make_room_loaded_plate(**kwargs)
    inst.set_state(plate_bump(inst.plate))
    return inst


# -- the money test ---------------------------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_ledgers_agree(boundary, wall_name):
    """``radiated_energy == room.injected`` — the test a wrong ``R_j`` cannot survive.

    The port predicts its work from ``pbar = pbar_free + R q``; the room books the same work from
    its **own** post-closure field, never from a number handed back. The two agree only if every
    ``R_j`` is exactly right, which is why this and not the conserved total is the money test.
    The lossy-mounting-wall case is what pins the ``(1 + beta)`` factor — the radiation leg left
    exactly that kind of denominator unpinned twice in its history.
    """
    inst = _seeded(boundary=boundary, walls=WALLS[wall_name])
    e0 = inst.energy()
    _run(inst, 300)
    gap = abs(inst.radiated_energy - inst.room.injected)
    assert gap <= LEDGER_TOL * abs(inst.radiated_energy)
    # ... and the channel is not vacuous: something actually went through it.
    assert abs(inst.radiated_energy) > 1e-4 * e0


def test_conservation_is_blind_to_a_wrong_R():
    """The conserved total stays green with an 18%-wrong coupling — so it is not sufficient.

    Drops the wall-closure divisor from ``R_j`` on a **lossy** mounting wall (where it is the only
    thing that matters) and reruns. This is the measurement the module docstring rests on, and it
    is asserted rather than asserted-about: the drift must stay at rounding *and* the ledger gap
    must become a real fraction. If a future change made the total sensitive to ``R``, this test
    fails and the framing above is wrong.
    """
    def drift_and_gap(naive):
        inst = make_room_loaded_plate(walls=WALLS["lossy-mounting-wall"])
        port, plate = inst.port, inst.plate
        if naive:
            beta = inst.room._beta[port.nodes]
            assert np.all(beta > 0.0), "the mounting wall must be lossy or there is nothing to drop"
            port.R = port.R * (1.0 + beta)
            port.load_matrix = (port.T.T @ sparse.diags(port.R) @ port.T).tocsr()
            a = sparse.identity(plate.n_live, format="csc") + (
                plate.theta * plate.k**2 * plate.kappa**2
            ) * plate.B
            inst._lu_loaded = splu((a + inst._load_scale * port.load_matrix).tocsc())
        inst.set_state(plate_bump(plate))
        e0 = surface_scene_energy(inst)
        worst = 0.0
        for _ in range(300):
            inst.step()
            inst.room.step()
            worst = max(worst, abs(surface_scene_energy(inst) - e0))
        gap = abs(inst.radiated_energy - inst.room.injected) / abs(inst.radiated_energy)
        return worst / abs(e0), gap

    good_drift, good_gap = drift_and_gap(naive=False)
    bad_drift, bad_gap = drift_and_gap(naive=True)
    assert good_gap <= LEDGER_TOL and good_drift <= DRIFT_TOL
    assert bad_drift <= DRIFT_TOL, "the point of this test is that the total does NOT notice"
    assert bad_gap > 0.05, f"the ledger gap must blow up, got {bad_gap:.2e}"


@pytest.mark.parametrize(
    "walls",
    [
        "rigid",
        {"z0": impedance_from_zeta(3.0)},                                   # lossy mounting wall
        {"z0": impedance_from_zeta(3.0), "x0": impedance_from_zeta(2.0)},   # + a lossy side wall
    ],
)
def test_R_j_is_what_the_room_does(walls):
    """``R_j`` measured **differentially** off the room, and the off-diagonal asserted *exactly* 0.

    Comparing the coupled step's ``pbar`` against ``pbar_free + R q`` would be a tautology — that
    expression is how ``pbar`` was computed, and it passes for any ``R`` whatsoever. So: save the
    room, step it once with nothing injected, restore, step it again with a unit injection at one
    node, and read the difference straight off the room's own post-closure field.

    The second assertion is the one that makes the *diagonal* load provable rather than plausible.
    Within a step an injection changes the pressure at its own node and nowhere else — propagation
    waits for the next momentum sub-step — so there is no cross-resistance at any separation,
    including between nodes one cell apart. Exactly zero, not small.
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
    p_before = base["p"][port.nodes]
    room.step()
    pbar0 = 0.5 * (room.p[port.nodes] + p_before)

    amp = 1e-4
    measured = np.zeros((port.node_count, port.node_count))
    for j in range(port.node_count):
        restore(base)
        q = np.zeros(port.node_count)
        q[j] = amp
        port.inject(q)
        room.step()
        measured[:, j] = (0.5 * (room.p[port.nodes] + p_before) - pbar0) / amp
    restore(base)

    assert np.allclose(np.diag(measured), port.R, rtol=1e-12, atol=0.0)
    off_diagonal = measured - np.diag(np.diag(measured))
    assert np.all(off_diagonal == 0.0), f"max |off| = {np.abs(off_diagonal).max():.3e}"


# -- the volume identity and the scheme itself -------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_volume_is_conserved_exactly(boundary):
    """``sum_j q_j == sum_n area_n v_n``, with ``v`` recomputed from the **plate**.

    The identity that makes the lumped monopole the low-frequency limit of the distributed port —
    and the only test a *consistently* wrong ``q`` factor (a stray ``k``, or ``2k`` for ``k``)
    cannot pass, since both energy ledgers would use the same wrong ``q`` and agree happily. Which
    is why ``v`` is rebuilt here from ``plate.u`` / ``plate.u_prev`` and never read off the port.

    Normalised by ``sum area |v|``: for a plate mode the net is a heavily cancelling sum, so a
    relative tolerance against the net would be measuring the cancellation, not the identity.
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


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_T_distributes_every_node_area(boundary):
    """``T^T 1 == areas``: every surface node's area is fully distributed, none created or lost."""
    inst = make_room_loaded_plate(boundary=boundary)
    distributed = np.asarray(inst.port.T.T @ np.ones(inst.port.node_count)).ravel()
    assert np.allclose(distributed, inst.port.areas, rtol=0.0, atol=1e-18)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_net_area_is_not_the_bounding_rectangle(boundary):
    """The radiating area is ``((N-1)/N)^2 Lx Ly`` supported, exactly ``Lx Ly`` free.

    A simply-supported plate's rim nodes are **dead** — they do not move, so they displace no
    volume. The shortfall is physics, not a defect, but it means comparing against a closed form
    for a piston of area ``Lx Ly`` is wrong by that factor at coarse ``N``. The free plate has no
    dead rim, and that contrast is the test.
    """
    inst = make_room_loaded_plate(boundary=boundary)
    plate = inst.plate
    factor = ((plate.N - 1) / plate.N) ** 2 if boundary == "supported" else 1.0
    assert inst.port.net_area == pytest.approx(factor * plate.Lx * plate.Ly, rel=1e-15)


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("fs", [8000.0, 11000.0])
def test_coupled_scheme_residual(boundary, fs):
    """Put the achieved ``u^{n+1}`` back into the coupled PDE and check the residual vanishes.

    Stronger than comparing against a hand-assembled copy of the same algebra: this writes the
    scheme out — ``delta_tt u = L(theta-average) - 2 sigma delta_t. u + f/rho_s`` with
    ``f = f_ext - T^T(pbar_free + R q)`` — and asks whether the solve satisfied it. Run at **two**
    timesteps, because a wrong-but-consistent ``k``-dependent factor passes at one; with
    ``sigma > 0``, a nonzero ``f_ext`` and a lossy mounting wall, so nothing is invisible. This is
    what pins the ``rho_s h^2`` versus ``rho_s`` branch difference between the two boundaries.
    """
    inst = _seeded(boundary=boundary, fs=fs, walls=WALLS["lossy-mounting-wall"], N=6, sigma=2.0)
    plate = inst.plate
    _run(inst, 5)

    f_ext = 1e-3 * np.random.default_rng(0).standard_normal(plate.n_live)
    u_n, u_nm1 = plate.u.copy(), plate.u_prev.copy()
    pbar_free = inst.port.free_pressure()
    inst.step(f_ext)
    u_np1 = plate.u.copy()

    k, theta = plate.k, plate.theta
    average = theta * u_np1 + (1.0 - 2.0 * theta) * u_n + theta * u_nm1
    velocity = (u_np1 - u_nm1) / (2.0 * k)
    q = inst.port.T @ velocity
    f_total = f_ext - (inst.port.T.T @ (pbar_free + inst.port.R * q))
    accel = u_np1 - 2.0 * u_n + u_nm1
    if boundary == "supported":
        mass, stiffness = plate.rho * plate.h * plate.h, plate.B
        weight = 1.0
    else:
        mass, stiffness = plate.rho, plate.K
        weight = plate.w
    residual = (
        weight * accel
        + k * k * plate.kappa**2 * (stiffness @ average)
        + 2.0 * plate.sigma * k * k * weight * velocity
        - k * k * f_total / mass
    )
    assert np.max(np.abs(residual)) <= 1e-11 * np.max(np.abs(weight * accel))


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_no_air_load_reproduces_the_bare_plate(boundary):
    """``T = 0`` reduces to :class:`Plate` **bit-for-bit** — the family's reduction ledger entry.

    (Alongside ``R = 0 -> `` bare body, ``M_a = inf -> RadiatedBody``, ``sigma_1 = 0 -> `` model #2,
    ``nonlinear = False -> `` #5.) There is no natural ``R = 0`` reduction here — ``R_j`` vanishes
    only on an open face, which is refused — so a disconnected ``T`` is the available one, and it
    is worth having because ``RoomLoadedPlate`` reassembles the plate's RHS and its ``A`` rather
    than reaching into ``plate.py``. Bit-identity is claimable because the load's structural zeros
    are eliminated before factoring, so a zero load factors the plate's **own** matrix.
    """
    inst = make_room_loaded_plate(boundary=boundary, sigma=1.0)
    loaded, room = inst.plate, inst.room
    bare = Plate(
        Lx=loaded.Lx, Ly=loaded.Ly, kappa=loaded.kappa, rho=loaded.rho, fs=loaded.fs,
        N=loaded.N, sigma=loaded.sigma, theta=loaded.theta, boundary=boundary,
    )
    inst.port.T = sparse.csr_matrix(inst.port.T.shape)
    inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
    sk = loaded.sigma * loaded.k
    coeff = loaded.theta * loaded.k**2 * loaded.kappa**2
    a = (
        (1.0 + sk) * sparse.identity(loaded.n_live, format="csc") + coeff * loaded.B
        if boundary == "supported"
        else (1.0 + sk) * loaded.W + coeff * loaded.K
    )
    inst._lu_loaded = splu(a.tocsc())

    force = 1e-3 * np.random.default_rng(1).standard_normal(loaded.n_live)
    bare.set_state(plate_bump(bare))
    inst.set_state(plate_bump(loaded))
    for _ in range(60):
        bare.step(force)
        inst.step(force)
        room.step()
    assert np.array_equal(loaded.u, bare.u)
    assert inst.radiated_energy == 0.0


# -- the port's read-out, and the sign convention ----------------------------------------


@pytest.mark.parametrize("walls", ["rigid", impedance_from_zeta(4.0)])
@pytest.mark.parametrize("face", FACES)
def test_free_pressure_matches_full_array(walls, face):
    """The ``O(patch)`` local read is **bit-identical** to the full-array closure, on all six faces.

    Batch 2 added the scalar version of this on review, because an off-by-one in the local read
    survives every energy test (port and room would still agree on a wrong value). Over a whole
    wall plane it is more exposed, not less — and the **high** faces exercise the
    ``where(idx < N, ...)`` branch that a low face never touches, which no ledger sees either.
    """
    inst = _seeded(walls=walls, face=face, N=6)
    room = inst.room
    _run(inst, 23)

    local = inst.port.free_pressure()
    p_next = room.p - room.k * room.rho0 * room.c0**2 * room._divergence()
    if room._has_walls:
        p_next = (p_next - room._beta * room.p) / (1.0 + room._beta)
        p_next[room._open] = 0.0
    full = (0.5 * (p_next + room.p))[inst.port.nodes]
    assert np.array_equal(local, full)
    assert np.max(np.abs(full)) > 1e-3, "the read must be exercised, not trivially zero"


@pytest.mark.parametrize("face", FACES)
def test_sign_convention_is_uniform_over_faces(face):
    """Positive displacement is along the **inward normal**, on every one of the six faces.

    Two assertions, and the first is the one no ledger sees: a port mounted on the wrong wall of a
    symmetric room is perfectly self-consistent, so the node indices are checked against the
    requested wall directly. Then the physics: a surface moving into the room **compresses** the air
    at its own face, read at ``n = 1`` — and the read-out time matters, because a six-step read gave
    the *wrong* answer in this batch's prototype when the plate's own half period was five steps.
    """
    inst = make_room_loaded_plate(face=face)
    axis = "xyz".index(face[0])
    wall_index = 0 if face[1] == "0" else inst.room.N[axis]
    assert np.all(inst.port.nodes[axis] == wall_index)

    inst.set_state(np.zeros(inst.plate.n_live), 1.0)  # uniformly into the room
    inst.step()
    inst.room.step()
    assert float(np.mean(inst.surface_pressure)) > 0.0


def test_a_sign_flip_is_invisible_to_every_energy_quantity():
    """The negative control, on a **high** face: flipping ``T`` is perfectly conservative and wrong.

    ``T^T R T`` is sign-invariant, so the load matrix, the solve and ``radiated_energy`` come out
    **bit-identical** while the room's field is inverted. Asserting the bit-identity of the wrong
    run *is* the point: it records that the "positive displacement along the global axis" convention
    — which needs an explicit per-face sign, and is wrong on three of the six faces — cannot be
    caught by any energy report. The local inward-normal convention disarms that instead of testing
    around it, which is why no inward normal appears in the code at all.
    """
    def run(flip):
        inst = make_room_loaded_plate(face="z1")
        if flip:
            inst.port.T = -inst.port.T
        inst.set_state(np.zeros(inst.plate.n_live), 1.0)
        for _ in range(4):
            inst.step()
            inst.room.step()
        return inst.radiated_energy, float(np.mean(inst.surface_pressure))

    good_energy, good_pressure = run(flip=False)
    bad_energy, bad_pressure = run(flip=True)
    assert bad_energy == good_energy, "a sign flip must be BIT-identical in the energy ledger"
    assert bad_pressure == pytest.approx(-good_pressure, rel=1e-12)


# -- conservation (necessary, not sufficient) --------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_scene_total_is_flat(boundary, wall_name):
    """``plate.energy() + radiated + room.energy()`` flat to machine precision.

    Necessary and not sufficient — see :func:`test_conservation_is_blind_to_a_wrong_R` — and it
    still ships, because it catches a genuinely broken scheme and it is the statement a reader
    wants. The channel size is asserted alongside so the test cannot pass on a disconnected
    coupling; note it is small here (~0.2% of ``E0``) because a narrow struck bump excites fine
    spatial patterns, which radiate badly. That *is* the acoustic short circuit, and
    :func:`test_free_plate_piston_is_fully_radiated` is the configuration where the channel is
    essentially the whole energy.
    """
    inst = _seeded(boundary=boundary, walls=WALLS[wall_name])
    e0 = surface_scene_energy(inst)
    worst = 0.0
    for _ in range(300):
        inst.step()
        inst.room.step()
        worst = max(worst, abs(surface_scene_energy(inst) - e0))
    assert worst <= DRIFT_TOL * abs(e0)
    assert abs(inst.radiated_energy) > 1e-4 * abs(e0)


def test_lossy_plate_scene_is_monotone():
    """With ``sigma_plate > 0`` the scene total is monotone non-increasing -- passivity."""
    inst = _seeded(walls=WALLS["all-lossy"], sigma=3.0)
    previous = e0 = surface_scene_energy(inst)
    for _ in range(300):
        inst.step()
        inst.room.step()
        current = surface_scene_energy(inst)
        assert current <= previous + 1e-14 * abs(e0)
        previous = current
    assert previous < e0


def test_free_plate_piston_is_fully_radiated():
    """The free plate's rigid-body nullspace is inert bare and **fully radiated** when baffled.

    Model #5b's stiffness nullspace is exactly ``{1, x, y}``: give the bare plate a uniform velocity
    and nothing resists it — it translates forever at *constant* energy, which is the negative
    control that makes the contrast mean something. Mount the same plate flush in a baffle and that
    identical motion **is a piston**, the most efficient radiator the geometry has, so a lossy room
    takes essentially all of it. No bare free plate and no lumped body-loss coefficient can do that.

    This is also the configuration where the coupling channel is ~100% of ``E0`` rather than the
    0.2% a struck bump gives, which is what makes the conservation assertion here non-vacuous.

    Deliberately **not** asserted: monotone decay in a *rigid* room. A closed box gives the piston's
    energy back — measured, the plate drops to 4.5% and climbs back — so only the total is monotone
    there, and asserting the drop as a decay would pass on the sampling instants and fail on the
    physics.
    """
    bare = Plate(Lx=0.30, Ly=0.30, kappa=20.0, rho=0.5, fs=8000.0, N=8, boundary="free")
    bare.set_state(np.zeros(bare.n_live), 1.0)
    e0_bare = bare.energy()
    for _ in range(400):
        bare.step()
    assert bare.energy() == pytest.approx(e0_bare, rel=1e-9)

    inst = make_room_loaded_plate(boundary="free", walls=WALLS["all-lossy"])
    inst.set_state(np.zeros(inst.plate.n_live), 1.0)
    e0 = surface_scene_energy(inst)
    worst = 0.0
    for _ in range(800):
        inst.step()
        inst.room.step()
        worst = max(worst, abs(surface_scene_energy(inst) - e0))
    assert inst.plate.energy() < 1e-3 * e0
    assert inst.radiated_energy == pytest.approx(e0, rel=1e-2)  # the channel IS the energy
    assert worst <= 1e-10 * abs(e0)


# -- the spreading operator ---------------------------------------------------------------


def _mirror_permutation(plate):
    """The plate's own ``x -> Lx - x`` as a permutation of its live nodes."""
    x, y = plate.X[plate.mask], plate.Y[plate.mask]
    source = np.lexsort((np.round(x, 12), np.round(y, 12)))
    target = np.lexsort((np.round(plate.Lx - x, 12), np.round(y, 12)))
    perm = np.empty(plate.n_live, dtype=int)
    perm[source] = target
    assert np.allclose(x[perm], plate.Lx - x) and np.allclose(y[perm], y)
    return perm


def _mirror_defect(spreading, shift):
    """Relative defect of ``T^T R T`` under the surface's mirror, at an in-plane offset of
    ``shift`` air cells from centred."""
    room = make_surface_room()
    plate = Plate(Lx=0.30, Ly=0.30, kappa=20.0, rho=0.5, fs=room.fs, N=16)
    origin = (
        0.5 * (room.N[0] * room.h - plate.Lx) + shift * room.h,
        0.5 * (room.N[1] * room.h - plate.Ly),
    )
    port = SurfacePort(
        room=room,
        face="z0",
        coords=np.column_stack((plate.X[plate.mask], plate.Y[plate.mask])),
        areas=np.full(plate.n_live, plate.h * plate.h),
        origin=origin,
        spreading=spreading,
    )
    perm = _mirror_permutation(plate)
    matrix = port.load_matrix.toarray()
    return np.linalg.norm(matrix[np.ix_(perm, perm)] - matrix) / np.linalg.norm(matrix)


def test_bilinear_equivariance_needs_centring():
    """Mirror-equivariance of the load holds **exactly when the surface is centred**, not near it.

    The batch plan expected bilinear's equivariance to be offset-*independent*. Measured here it is
    not: it holds when ``S = 2 (surface centre) / h_air`` is an **integer** (defect ~1e-15) and
    fails at 1.6e-01 … 3.8e-01 otherwise. The algebra agrees: the mirror sends a node to a cell
    fraction ``frac(S - t)``, which is the ``1 - f`` that reverses a bilinear weight pair only for
    integral ``S``. This is why :attr:`SurfacePort.origin` defaults to centred for **two**
    independent reasons rather than one: centring buys the load's equivariance as well as the
    scene's symmetry.
    """
    for shift in (0.0, 0.5):  # integral S
        assert _mirror_defect("bilinear", shift) < 1e-13
    for shift in (0.125, 0.25, 0.375, 0.625):
        assert _mirror_defect("bilinear", shift) > 1e-2


def test_nearest_node_equivariance_is_an_accident():
    """The negative control: nearest-node's symmetry rides on the rounding rule, not on geometry.

    Exact at an **even** ``S`` and broken at an **odd** one, because there the surface's own centre
    node lands on a rounding tie that round-half-to-even resolves the same way from both directions.
    Bilinear is exact at both. ``spreading="nearest"`` exists only for this.
    """
    assert _mirror_defect("nearest", 0.0) < 1e-13     # even S -- exact by accident
    assert _mirror_defect("nearest", 0.5) > 1e-2      # odd S -- the tie breaks it
    assert _mirror_defect("bilinear", 0.5) < 1e-13    # ... where bilinear is still exact


def _interior_area_spread(spreading, *, L, N):
    """Spread of the assigned area over air nodes strictly inside the footprint, in ``h_air^2``."""
    inst = make_room_loaded_plate(N=N, L=L, spreading=spreading)
    room, port = inst.room, inst.port
    assigned = np.asarray(port.T @ np.ones(inst.plate.n_live)).ravel()
    t0, t1 = port.in_plane_axes
    c = port._face_coords
    inside = (
        (port.nodes[t0] * room.h > c[:, 0].min() + room.h)
        & (port.nodes[t0] * room.h < c[:, 0].max() - room.h)
        & (port.nodes[t1] * room.h > c[:, 1].min() + room.h)
        & (port.nodes[t1] * room.h < c[:, 1].max() - room.h)
    )
    assert inside.sum() > 4, "need a real interior to measure"
    return (assigned[inside].max() - assigned[inside].min()) / (room.h * room.h)


def test_bilinear_assignment_is_exact_at_an_integral_grid_ratio():
    """``h_air^2`` per interior air node **exactly**, when ``h_air/h_surface`` is an integer.

    "Partition of unity" promises more than it delivers, and the batch plan overclaimed here (it
    reported exact flatness at every ratio). Poisson summation on the periodised hat gives Fourier
    coefficients ``sinc^2(pi k h_air/h_surface)``, whose ``k``-th term vanishes exactly when
    ``k h_air/h_surface`` is a nonzero integer — so *all* of them vanish only for an integral ratio.
    This test builds that ratio deliberately and pins the exactness; the next one measures the
    ripple that remains otherwise.
    """
    room = make_surface_room()
    for divisions in (2, 3, 4):
        N = int(round(0.45 / (room.h / divisions)))
        spread = _interior_area_spread("bilinear", L=N * room.h / divisions, N=N)
        assert spread < 1e-14, f"h_air/h_p = {divisions}: spread {spread:.3e}"


def test_bilinear_beats_nearest_node_at_every_refinement():
    """Off an integral ratio bilinear ripples — and nearest-node is 10x-100x worse and *diverges*.

    This is the argument that actually decides the spreading operator (the symmetry one does not
    survive — see above). Measured over ``N = 8, 16, 24, 32`` on a 0.60 m plate: bilinear gives
    0.082, 0.062, 0.051, 0.031 ``h_air^2`` — decreasing — while nearest-node gives 0.83, 1.03,
    0.64, 0.46, wandering with no convergence at all. A lumpy source at the grid scale, for
    nothing.
    """
    bilinear = [_interior_area_spread("bilinear", L=0.60, N=N) for N in (8, 16, 24, 32)]
    nearest = [_interior_area_spread("nearest", L=0.60, N=N) for N in (8, 16, 24, 32)]
    assert all(b < 0.1 * n for b, n in zip(bilinear, nearest, strict=True))
    assert bilinear[-1] < 0.5 * bilinear[0], "bilinear must improve with refinement"
    assert nearest[-1] > 0.4, "nearest-node does not converge, and that is the point"


def test_load_matrix_is_symmetric_and_the_cost_is_reported():
    """``T^T R T`` is symmetric to machine precision, and the factorization's real cost is exposed.

    The load matrix is left as the **raw** triple product, never symmetrised: its ~1e-16 asymmetry
    is the sparse product's summation order, and symmetrising would make this assertion vacuous.

    The plan expected the factorization not to thicken meaningfully. It does: the load couples every
    plate node sharing an air node, so measured LU fill is 1.55x, 3.50x and 5.29x a bare plate's at
    ``h_plate/h_air = 0.45, 0.23, 0.15``. Hence :attr:`RoomLoadedPlate.lu_nnz` — stored ``nnz`` is
    not what ``splu`` pays.
    """
    inst = make_room_loaded_plate(N=16)
    matrix = inst.port.load_matrix
    assert abs(matrix - matrix.T).max() <= 1e-14 * abs(matrix).max()

    plate = inst.plate
    bare = splu(
        (
            sparse.identity(plate.n_live, format="csc")
            + (plate.theta * plate.k**2 * plate.kappa**2) * plate.B
        ).tocsc()
    )
    assert inst.lu_nnz > bare.L.nnz + bare.U.nnz
    assert inst.nnz_growth > 1.0


# -- the headline -------------------------------------------------------------------------


def _peak_monopole(inst, steps=200):
    """Largest ``|sum_j q_j| / net_area`` over a run — all a one-port could ever couple through."""
    peak = 0.0
    for _ in range(steps):
        inst.step()
        inst.room.step()
        peak = max(peak, abs(inst.volume_velocity) / inst.port.net_area)
    return peak


@pytest.mark.parametrize("mode", [(2, 1), (1, 2)])
def test_an_even_mode_is_silent_to_every_one_port_and_still_radiates(mode):
    """**The batch's claim.** A mode with exactly zero net volume velocity radiates definitely.

    For ``boundary="supported"`` the scheme's modes are the *exact* discrete ``sin x sin``, and
    ``sum_{i=1}^{N-1} sin(m pi i/N) = 0`` identically for even ``m`` — so an even-index mode's net
    volume displacement is **zero, not small**, and ``AirRadiation``, ``RadiatedBody``,
    ``RationalAirLoad`` and ``RoomPort`` all report exact silence. The distributed port reports a
    definite nonzero power, because each patch of surface pushes on the air *locally* and the
    cancellation is only as complete as the acoustic wavelength's ability to bridge a ``+`` region
    and its ``-`` neighbour.

    The zero must hold for the **whole run**, not at ``t = 0``: it is the run-long assertion that
    would catch a load matrix that mixes the odd modes back in. And the radiated energy is bounded
    **below**, so the test cannot pass on a disconnected coupling.

    Deliberately not asserted: any *ranking* of the modes by radiated energy. A plate mode locks
    spatial fineness to frequency, so a finer mode completes more cycles in the same window and its
    count beats the per-cycle suppression — the finer mode radiates *more*, which is the opposite of
    the fineness law and belongs only to a prescribed-velocity rig where frequency is a knob.
    """
    room = make_surface_room(N=(12, 12, 9))  # x <-> y symmetric, or (2,1) and (1,2) see two rooms
    inst = make_room_loaded_plate(room=room, N=16)
    inst.set_state(1e-3 * plate_mode_shape(inst.plate, *mode))
    reference = make_room_loaded_plate(room=make_surface_room(N=(12, 12, 9)), N=16)
    reference.set_state(1e-3 * plate_mode_shape(reference.plate, 1, 1))

    assert _peak_monopole(inst) < 1e-13
    assert _peak_monopole(reference) > 1e-2  # the (1,1) control DOES have a monopole
    assert inst.radiated_energy > 0.05 * reference.radiated_energy


def test_the_silence_is_a_property_of_the_whole_scene():
    """The zero survives only in a scene that is mirror-symmetric about the mode's own plane.

    Equivariance of the load is necessary and **not** sufficient: the incoming ``T^T pbar_free`` is
    the *room's* field. Measured three ways, and there is **no tolerance band** — the leak is linear
    in the offset with no threshold, so "approximately centred" is not approximately silent.

    The two positive controls are not defects to hide. A room that is asymmetric about the plate
    re-excites the plate's *shape*, converting an acoustically silent mode into a radiating one at
    the percent level — and no ``R(omega)`` one-port can represent that at all, because a lumped
    port couples through a single scalar and has no shape for the room to push on.
    """
    def leak(walls, offset=0.0):
        room = make_surface_room(walls=walls)
        origin = None
        if offset:
            plate_L = 0.30
            origin = (
                0.5 * (room.N[0] * room.h - plate_L) + offset * room.h,
                0.5 * (room.N[1] * room.h - plate_L),
            )
        inst = make_room_loaded_plate(room=room, N=16, origin=origin)
        inst.set_state(1e-3 * plate_mode_shape(inst.plate, 2, 1))
        return _peak_monopole(inst)

    # symmetric scenes: silent to rounding
    assert leak("rigid") < 1e-13
    assert leak(impedance_from_zeta(4.0)) < 1e-13
    # asymmetric in y only -- the (2,1) mode is antisymmetric in X, so its zero is untouched
    assert leak({"y0": impedance_from_zeta(4.0)}) < 1e-13
    # asymmetric in the mode's OWN axis: the room drives the plate's odd modes
    assert leak({"x0": impedance_from_zeta(4.0)}) > 1e-3
    # and off-centre by a third of an air cell, in an otherwise perfect room
    assert leak("rigid", offset=1.0 / 3.0) > 1e-3


# -- the chain, and the refusals ----------------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_string_bridge_plate_room_chain(boundary):
    """``string -> bridge -> plate -> room``, with **no edit to** ``connection.py``, guard safe.

    The margin assertion is the load-bearing half. ``StringPlateBridge._stability_margin``
    reassembles the plate's ``G0`` block from scratch out of ``theta, rho, h, kappa, B / W, K`` —
    every one of which ``__getattr__`` delegation hands over happily — so the guard is computed
    against physics that is not happening, and the delegation would hide that perfectly. It is safe
    because ``G0 = M + (theta - 1/4) k^2 S`` is a statement about mass and theta-excess stiffness
    while the air load is **dissipative**: it enters ``A``, never ``G0``. Pinning the bit-identity
    here means a future change making the load non-dissipative fails loudly instead of silently
    mis-guarding.

    **This docstring used to name batch 4's two-sided dipole plate, "whose face cut removes air
    mass", as that future change. It is not one**, and the reasoning did not survive: the face cut
    removes air inertia from the *room's* ledger, where it was never part of the plate's ``G0``,
    while the load stays proportional to ``u^{n+1} - u^{n-1}`` and enters ``A``. Measured in
    ``tests/test_airbox_dipole.py::test_string_bridge_plate_room_chain``, a ``RoomSuspendedPlate``
    gives the **same** margin this test pins, to the last digit.
    """
    inst = make_room_loaded_plate(boundary=boundary, walls=WALLS["all-lossy"])
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


def test_refuses_a_sample_rate_mismatch():
    room = make_surface_room(fs=8000.0)
    plate = Plate(Lx=0.30, Ly=0.30, kappa=20.0, rho=0.5, fs=9000.0, N=8)
    with pytest.raises(ValueError, match="sample-rate mismatch"):
        RoomLoadedPlate(plate=plate, room=room, face="z0")


def test_refuses_a_footprint_reaching_the_face_rim():
    """A face-rim node touches a **second** wall, so ``R_j`` would stop being uniform.

    Clipping the stencil there would fold the outboard weight back onto the boundary node: volume
    still conserved, every ledger still green, and the source geometry quietly wrong — the same
    failure shape as the sign flip. ``AirBox.node_index`` already refuses to relocate an out-of-room
    point rather than snapping it, and this matches.
    """
    with pytest.raises(ValueError, match="rim"):
        make_room_loaded_plate(L=0.85, N=16)


def test_refuses_a_footprint_outside_the_face():
    with pytest.raises(ValueError, match="outside face"):
        make_room_loaded_plate(origin=(0.9, 0.4))


def test_refuses_a_surface_too_coarse_for_the_air_grid():
    """Unfed air nodes under the footprint make the acoustic source a comb at the grid scale.

    The condition is a **count of unfed nodes**, not an inequality on ``h_surface/h_air``: at ratio
    0.909, comfortably inside the naive inequality, nearest-node still leaves half the footprint
    unfed.
    """
    with pytest.raises(ValueError, match="fed by no surface node"):
        make_room_loaded_plate(boundary="free", N=3, spreading="nearest")


def test_refuses_a_surface_on_an_open_face():
    """Perfectly conservative, completely silent — and the energy report is blind to it."""
    with pytest.raises(ValueError, match="open"):
        make_room_loaded_plate(walls={"z0": "open"})


def test_refuses_overlapping_ports():
    """Two ports sharing a node are not independent within a step, so each solves against a
    pressure that never occurred. Disjointness is exactly what makes the cheap per-port solve
    *exact* — and it is what lets N instruments share one room."""
    room = make_surface_room()
    make_room_loaded_plate(room=room)
    with pytest.raises(ValueError, match="shares node"):
        make_room_loaded_plate(room=room)


def test_refuses_solving_twice_without_a_room_step():
    """A port does not step its room — the caller does, once, after every port has solved."""
    inst = make_room_loaded_plate()
    inst.step()
    with pytest.raises(RuntimeError, match="twice within one room step"):
        inst.step()


def test_refuses_unknown_face_and_spreading():
    with pytest.raises(ValueError, match="unknown face"):
        make_room_loaded_plate(face="q0")
    with pytest.raises(ValueError, match="unknown spreading"):
        make_room_loaded_plate(spreading="cubic")


def test_two_disjoint_surfaces_share_one_room():
    """N instruments in one room, inherited unchanged: disjoint node sets, one room step."""
    room = make_surface_room(N=(12, 11, 12))
    top = make_room_loaded_plate(room=room, face="z0")
    bottom = make_room_loaded_plate(room=room, face="z1")
    top.set_state(plate_bump(top.plate))
    bottom.set_state(plate_bump(bottom.plate))
    e0 = surface_scene_energy(top, bottom)
    worst = 0.0
    for _ in range(200):
        top.step()
        bottom.step()
        room.step()
        worst = max(worst, abs(surface_scene_energy(top, bottom) - e0))
    assert worst <= DRIFT_TOL * abs(e0)
    gap = abs(top.radiated_energy + bottom.radiated_energy - room.injected)
    assert gap <= LEDGER_TOL * abs(top.radiated_energy + bottom.radiated_energy)
