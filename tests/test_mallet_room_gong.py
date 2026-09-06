"""A mallet on a gong **in a room** — the nested solve meets the loaded operator.

``docs/dev/mallet-vk-room-plan.md``. Two models that had never met: the mallet's outer chord
re-solves the plate several times from one time-``n`` state, and the room wrapper's ``step()`` is a
once-per-step transaction that reads its port, injects once and books the radiated energy once.

**What this file is mostly about is that the room is stepped exactly once.** Driving the wrapper's
own ``step()`` per outer trial would inject ``n_outer`` times and book the energy ``n_outer`` times,
and the scene total that would normally catch such a thing is the very number a double injection
corrupts on *both* sides at once. So the bars here are structural rather than statistical: the
injection is counted, and the volume velocity the room receives is recomputed from the plate's own
committed buffers.

The other half is the frozen column. The chord's tangent ``_g_s`` comes off the drive-point
response of the operator the step actually inverts, and in a room that is ``A_loaded``, not ``A``.
It changes **no physics** — the two ``g_s`` terms cancel at the fixed point, so the committed force
and field are the same either way — and it is worth about five times the outer iterations, which is
why the exact ``n_outer == 1`` bar below is the one that discriminates.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    MALLET_ROOM_V0,
    make_air_vk_plate,
    make_mallet_room_gong,
    make_room_loaded_vk_plate,
    make_suspended_vk_plate,
)
from scipy.sparse.linalg import splu

from physsynth.core.mallet import MalletPlate, MalletVKPlate
from physsynth.core.plate import Plate

TIERS = ("baffled", "suspended")


def loaded_column(inst, node):
    """``(k^2 / force_denominator) A_loaded^-1 e_node`` — what the chord must have frozen."""
    plate = inst.plate
    e = np.zeros(plate.n_live)
    e[node] = 1.0
    return (plate.k * plate.k / plate.force_denominator) * inst._lu_loaded.solve(e)


def bare_column(inst, node):
    """The same column off the plate's **own** factorization — the one a bare gong freezes."""
    plate = inst.plate
    e = np.zeros(plate.n_live)
    e[node] = 1.0
    lu = splu(inst._surface.a_bare().tocsc())
    return (plate.k * plate.k / plate.force_denominator) * lu.solve(e)


# -- the composition holds together -----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_the_wrapper_is_what_mal_plate_returns(tier):
    """``mal.plate`` must be the object the caller passed, as it is for a bare gong.

    The room, the port and the radiated ledger all hang off the wrapper, so handing back the bare
    plate would silently take the whole coupling channel away from anyone who asked for it — and
    ``mal.energy()`` is one of the things that would then be quietly wrong.
    """
    make = make_room_loaded_vk_plate if tier == "baffled" else make_suspended_vk_plate
    inst = make()
    plate = inst.plate
    mal = MalletVKPlate(
        plate=inst, mass=0.05, stiffness=5e4, alpha=2.3,
        strike_x=0.3 * plate.Lx, strike_y=0.4 * plate.Ly, strike_velocity=MALLET_ROOM_V0,
    )
    assert mal.in_room
    assert mal.plate is inst              # the wrapper, not the plate inside it
    assert mal.plate.plate is plate       # and the gong is one delegation further in
    assert mal.plate.room is inst.room
    assert mal.plate.port is inst.port


def test_a_bare_gong_is_unchanged_by_the_widening():
    """The three-armed cast must leave the bare path exactly as it was."""
    plate = make_air_vk_plate()
    mal = MalletVKPlate(
        plate=plate, mass=0.05, stiffness=5e4, alpha=2.3,
        strike_x=0.3 * plate.Lx, strike_y=0.4 * plate.Ly, strike_velocity=MALLET_ROOM_V0,
    )
    assert not mal.in_room
    assert mal.plate is plate


def test_a_linear_plate_still_gets_the_message_that_names_MalletPlate():
    """The third arm of the cast is load-bearing: it is what sends a caller to the right model."""
    plate = Plate(Lx=0.3, Ly=0.3, kappa=20.0, rho=7.8, fs=8000.0, N=8)
    with pytest.raises(TypeError, match="MalletPlate"):
        MalletVKPlate(
            plate=plate, mass=0.05, stiffness=5e4, alpha=2.3,
            strike_x=0.09, strike_y=0.12, strike_velocity=6.0,
        )
    assert MalletPlate is not MalletVKPlate  # the message names a real alternative


# -- the frozen column ------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_the_chord_freezes_the_LOADED_column_not_the_plates_own(tier):
    """``_g_s`` must come off ``A_loaded``, the operator the coupled step actually inverts.

    Asserted to the **bit** against the column rebuilt here, and asserted to differ from the bare
    plate's. The difference is small — 2.9e-04 relative on this fixture — and that is exactly why
    it needs an exact test rather than a tolerance: it is far too small to show up in any
    trajectory comparison, and it is worth about five times the outer iterations
    (``docs/dev/mallet-vk-room-plan.md`` §2.2).
    """
    mal = make_mallet_room_gong(tier=tier)
    inst = mal.plate
    want = loaded_column(inst, mal.node)
    other = bare_column(inst, mal.node)

    assert np.array_equal(mal._influence, want)
    assert mal._g_s == want[mal.node]
    assert mal._g == mal._g_s + mal._g_h
    # The two columns are genuinely different, so the bit-equality above is a real choice.
    assert mal._g_s != other[mal.node]
    assert abs(other[mal.node] - want[mal.node]) / abs(want[mal.node]) > 1e-5


@pytest.mark.parametrize("tier", TIERS)
def test_nonlinear_false_in_a_room_exits_the_chord_at_one_iteration(tier):
    """The exact bar that discriminates the two columns — and the reason the loaded one ships.

    With ``nonlinear=False`` the plate is affine in the contact force, so
    ``w_node(f) = w_free,node - g_s_true f`` where ``g_s_true`` belongs to the operator being
    inverted. The chord then forms ``u_eff = w_node(f) + g_s f``, and **only** when ``g_s`` is
    ``g_s_true`` do the two terms cancel and the second contact solve receive the same arguments as
    the first. It exits at one iteration.

    Frozen on the bare column instead, the residual would contract by
    ``|g_s_bare - g_s_loaded| / g`` a pass — about 2.3e-04 here — and need roughly five iterations
    to reach ``outer_tol``. That arithmetic is asserted below from the two columns, so the claim
    "the loaded column is worth about five times the chord" is checked against this fixture rather
    than quoted from the plan.
    """
    mal = make_mallet_room_gong(tier=tier, nonlinear=False)
    inst = mal.plate
    landed = 0
    for _ in range(400):
        mal.step()
        inst.room.step()
        if mal.in_contact:
            landed += 1
            assert mal.n_outer == 1, "the affine chord must exit at one iteration"
            assert mal.outer_converged
            assert mal.outer_residual <= mal.outer_tol
    assert landed > 50, "the mallet has to actually land for this to mean anything"

    # What the wrong column would have cost, on this fixture's own numbers.
    rate = abs(bare_column(inst, mal.node)[mal.node] - mal._g_s) / mal._g
    assert 1e-5 < rate < 1e-3, rate
    assert 4 <= int(np.ceil(np.log(mal.outer_tol) / np.log(rate))) + 1 <= 6


def test_a_swapped_factorization_refreshes_the_frozen_column():
    """``_lu_loaded`` has a setter, and a column frozen on the old one would go stale in silence.

    Nothing would turn red: the chord converges to the same root whichever column it froze, so the
    only symptom is more outer iterations. The guard is an identity check on the factorization
    object, which costs a pointer comparison per step and re-derives the column exactly when it has
    really changed.

    Swapped here for the plate's **own** factorization — the room's load removed from the operator
    entirely — which is the largest change a caller can make through that setter.
    """
    mal = make_mallet_room_gong()
    inst = mal.plate
    before = mal._g_s
    assert before == loaded_column(inst, mal.node)[mal.node]

    inst._lu_loaded = splu(inst._surface.a_bare().tocsc())
    mal.step()

    assert mal._g_s == bare_column(inst, mal.node)[mal.node]
    assert mal._g_s != before
    assert mal._g == mal._g_s + mal._g_h


# -- the room is stepped exactly once ----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_a_mallet_that_never_lands_leaves_the_room_scene_bit_identical(tier):
    """Structural, not a floating-point coincidence — and it is the whole miss path in one line.

    A zero contact force short-circuits to the force-free advance and returns it **unmodified**, so
    the trial that produced it was assembled by ``prepare`` with ``f_ext=None`` — the very
    expression ``RoomLoadedVKPlate.step()`` uses. Driving the plate with a zero force *vector*
    instead would add ``+0.0`` into every right-hand-side entry, which is the identity for every
    double except ``-0.0``, and the seam's own ``Option`` short-circuit is what avoids it.

    The room is compared too, not just the plate: a mallet that never lands must leave the air
    exactly where the bare wrapper leaves it.
    """
    mal = make_mallet_room_gong(tier=tier, strike_velocity=-1.0, gap=0.01)
    make = make_room_loaded_vk_plate if tier == "baffled" else make_suspended_vk_plate
    ref = make()

    ic = 2.0 * mal.plate.plate.e * np.sin(np.linspace(0.0, 3.0, mal.plate.n_live))
    mal.plate.set_state(ic)
    ref.set_state(ic)

    for _ in range(120):
        mal.step()
        mal.plate.room.step()
        ref.step()
        ref.room.step()
        assert mal.contact_force == 0.0
        assert mal.n_outer == 0
        assert np.array_equal(mal.plate.plate.u, ref.plate.u)
        assert np.array_equal(mal.plate.room.p, ref.room.p)
        assert mal.plate.radiated_energy == ref.radiated_energy


@pytest.mark.parametrize("tier", TIERS)
def test_the_port_is_injected_once_per_step_however_long_the_chord_runs(tier):
    """The failure this batch was designed around, asserted directly.

    ``port.inject`` must be called once per step whatever ``n_outer`` is. A wrapper driven per
    trial would inject ``n_outer`` times, and the scene total cannot see it — the radiated ledger
    and the room's energy would both be wrong by the same amount and in the same direction.
    """
    mal = make_mallet_room_gong(tier=tier)
    inst = mal.plate
    port = inst.port
    calls = []
    real = port.inject
    port.inject = lambda q: (calls.append(np.asarray(q).copy()), real(q))[1]

    multi = 0
    for _ in range(200):
        before = len(calls)
        mal.step()
        inst.room.step()
        assert len(calls) - before == 1, "one injection per step, whatever the chord did"
        if mal.n_outer > 1:
            multi += 1
    assert multi > 20, "the chord has to iterate for this test to be about anything"


@pytest.mark.parametrize("tier", TIERS)
def test_the_room_is_driven_by_the_COMMITTED_field_not_by_a_trial(tier):
    """The one error the inject-once bar above cannot see.

    ``q = T (w^{n+1} - w^{n-1}) / 2k``. Hand the room half a *trial* iterate instead of the
    accepted one and the injection count is still one, the radiated ledger still advances by a
    plausible amount, and every energy bar still passes — the room simply receives a volume
    velocity the gong never had. So it is recomputed here from the plate's own buffers, taken
    around the step.
    """
    mal = make_mallet_room_gong(tier=tier)
    inst = mal.plate
    checked = 0
    for _ in range(200):
        u_prev = inst.plate.u.copy()          # w^{n-1} after the step is w^n now -- take it before
        w_nm1 = inst.plate.u_prev.copy()
        mal.step()
        w_np1 = inst.plate.u.copy()
        assert np.array_equal(u_prev, inst.plate.u_prev)   # the roll happened exactly once
        want = inst.port.T @ ((w_np1 - w_nm1) / (2.0 * inst.k))
        assert np.array_equal(inst.nodal_volume_velocity, want)
        inst.room.step()
        if mal.n_outer > 1:
            checked += 1
    assert checked > 20, "the chord has to iterate for this to be about the accepted iterate"


@pytest.mark.parametrize("tier", TIERS)
def test_the_rooms_load_terms_do_not_move_across_the_chord(tier):
    """§1.1's premise, asserted rather than assumed.

    The two room terms — the open-circuit pressure ``T^T pbar_free`` and the ``w^{n-1}`` carry —
    are functions of time-``n`` and time-``n-1`` state alone, which is what makes it legitimate to
    assemble them once and let the chord vary only its own force. If either moved with the trial,
    every iteration would be solving a different problem and the fixed point would mean nothing.

    Checked the way the injection is checked, by counting: ``prepare`` reads the port **once** per
    step, so ``free_pressure`` and ``require_ready`` are each called once however many times the
    chord went round. A later refactor that moved the port read inside the chord would still give
    the right answer on this fixture — the terms really are invariant — and would be caught here
    rather than by a number.
    """
    mal = make_mallet_room_gong(tier=tier)
    inst = mal.plate
    port = inst.port
    reads, readies = [], []
    real_free, real_ready = port.free_pressure, port.require_ready
    port.free_pressure = lambda: (reads.append(1), real_free())[1]
    port.require_ready = lambda: (readies.append(1), real_ready())[1]

    checked = 0
    for _ in range(200):
        n_free, n_ready = len(reads), len(readies)
        mal.step()
        assert len(reads) - n_free == 1, "the port is read once a step, not once a trial"
        assert len(readies) - n_ready == 1
        inst.room.step()
        if mal.n_outer > 1:
            checked += 1
    assert checked > 20, "the chord has to iterate for this test to be about anything"


# -- energy ------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_the_scene_total_is_conserved_through_the_strike(tier):
    """Lossless gong, rigid room: ``mal.energy() + room.energy()`` is flat across the contact.

    ``mal.energy()`` takes its plate term from the **wrapper**, so the radiated channel is inside
    it; the air itself is the second term. Necessary and not sufficient — this family's standing
    rule is that no single one of its three detectors is enough — and here with model #6's own
    caveat that the statement holds only at the Picard fixed point, which is why
    ``inner_converged`` is asserted beside it.
    """
    mal = make_mallet_room_gong(tier=tier)
    room = mal.plate.room
    e0 = mal.energy() + room.energy()
    lo = hi = e0
    landed = 0
    for _ in range(600):
        mal.step()
        room.step()
        assert mal.inner_converged
        landed += int(mal.in_contact)
        e = mal.energy() + room.energy()
        lo, hi = min(lo, e), max(hi, e)
    assert landed > 50
    assert (hi - lo) / abs(e0) < 1e-10, (hi - lo) / abs(e0)


def test_the_delegated_plate_energy_is_the_wrong_number_and_the_wrapper_knows_it():
    """A guard on the trap, not on the physics.

    ``mal.plate.plate.energy()`` is the gong's total **without** the channel it radiates through,
    and the difference is the radiated ledger. Anyone assembling a scene total out of it is short
    by exactly that, so the two are asserted to differ once the room has taken some energy.
    """
    mal = make_mallet_room_gong()
    room = mal.plate.room
    for _ in range(300):
        mal.step()
        room.step()
    bare = mal.plate.plate.energy()
    wrapped = mal.plate.energy()
    assert wrapped != bare
    assert wrapped == bare + mal.plate.radiated_energy
    assert abs(mal.plate.radiated_energy) > 0.0
