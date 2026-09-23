"""The gong in the room, driven by a string: ``string -> bridge -> room-loaded gong -> room``.

**What is left here, and why.** This file used to hold the whole of air-box batch 6 — model #6 as a
baffled and suspended surface. Plan §17 (``docs/dev/python-retirement-plan.md``) re-homed the two
von Kármán room wrappers into ``physsynth-core`` as ``RoomGrid<VkSeam>`` and moved their seventeen
tests to ``crates/physsynth-core/tests/airbox_vk.rs``: the ``nonlinear=False`` bit-identity against
the linear room-loaded plate, the zero-area reduction to the bare plate, the two ledgers, the
coupled residual of two fields, the ``couple_tol`` detector split, the six iteration read-outs,
``couple_method``, Newton past the Picard wall and the rate refusal. They went because their
referent — the wrapper — is ported.

What stays is the **three-way chain**, because its referent is :class:`StringVKPlateBridge`, which
is not ported yet (the retirement rule of plan §14.3: a test goes when its referent is the ported
class, and stays when it drives a caller that has not been). These nine run against the binding's
wrappers and move when the bridge tier does.

``docs/dev/string-vk-plate-room-plan.md`` is the design record for the chain: the bridge plan
deferred it on "a third fixed point (string spring, Picard, room load)"; there is ONE, and these
tests measure it rather than arguing it.
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_DIPOLE_INDEX,
    make_surface_room,
    make_vk_room_bare_twin,
    make_vk_room_chain,
    step_vk_room_chain,
    vk_linear_twin,
    vk_room_pluck,
    vk_room_rigid_share,
)

from physsynth.core.airbox import (
    RoomLoadedPlate,
    RoomSuspendedPlate,
    impedance_from_zeta,
)
from physsynth.core.connection import StringPlateBridge
from physsynth.core.string_ideal import IdealString

BOUNDARIES = ("supported", "free")
TIERS = ("baffled", "suspended")
DRIFT_TOL = 1e-12   # the scene total, relative -- necessary, not sufficient
LEDGER_TOL = 1e-12  # |radiated - injected| / |radiated| -- necessary, ALSO not sufficient
WALLS = {
    "rigid": "rigid",
    "all-lossy": impedance_from_zeta(4.0),
    "one-lossy-wall": {"z0": impedance_from_zeta(3.0)},
}


def _make_linear(tier, vk):
    """The batch-3/4 wrapper around :func:`vk_linear_twin`, in its **own** copy of the same room.

    Sharing one room would put two ports on the same nodes and be refused, so each side gets its
    own — identical by construction, because :func:`make_surface_room` is deterministic in its
    arguments and both sides take its defaults. The wrappers are built directly rather than through
    ``make_room_loaded_plate`` so the twin's ``nu`` and ``theta`` reach the plate that is compared,
    instead of being re-derived from that helper's own defaults.
    """
    room = make_surface_room()
    plate = vk_linear_twin(vk)
    if tier == "baffled":
        return RoomLoadedPlate(plate=plate, room=room, face="z0")
    return RoomSuspendedPlate(plate=plate, room=room, plane="z", index=AIRBOX_DIPOLE_INDEX)


# -- the three-way chain: string -> bridge -> room-loaded gong -> room ----------------------
#
# `docs/dev/string-vk-plate-room-plan.md`. The bridge plan deferred this on "a third fixed point
# (string spring, Picard, room load)"; there is ONE, and these tests measure it rather than
# arguing it. Nothing in `connection.py`, `airbox.py` or `plate.py` changed to make them pass.


def _bare_string(bridge):
    """A standalone copy of the chain's string, for the ``K = 0`` decoupling identity."""
    s = bridge.string
    return IdealString(
        L=s.L, T=s.T, rho=s.rho, fs=s.fs, N=s.N, boundary=("fixed", "free"), sigma=s.sigma
    )


def _linear_chain_twin(bridge, tier):
    """:class:`StringPlateBridge` on the **linear** room-loaded twin of this chain's plate.

    The regression target for ``nonlinear=False``: same string coefficients, same ``K``, same
    drive index **copied** rather than re-derived, and the plate through :func:`vk_linear_twin`
    (whose ``rho=vk.rho_s`` is the 1000x substitution the whole regression exists to police).
    """
    s = bridge.string
    twin_string = IdealString(
        L=s.L, T=s.T, rho=s.rho, fs=s.fs, N=s.N, boundary=("fixed", "free"), sigma=s.sigma
    )
    return StringPlateBridge(
        string=twin_string,
        plate=_make_linear(tier, bridge.plate.plate),
        K=bridge.K,
        drive_index=bridge.drive_index,
    )


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_chain_composes_and_the_guard_is_bit_identical(boundary, tier):
    """``string -> bridge -> room-loaded gong -> room``, with **no edit to** ``connection.py``.

    The margin assertion is the load-bearing half, exactly as in
    ``test_airbox_surface.py::test_string_bridge_plate_room_chain`` one model down.
    :meth:`StringVKPlateBridge._stability_margin` reassembles the plate's ``G0`` block from
    scratch out of ``theta, rho_s, h, kappa, B / W / K`` -- every one of which the wrapper's
    ``__getattr__`` hands over happily -- so the guard is computable against physics that is not
    happening, and the delegation would hide that perfectly. It is safe because
    ``G0 = M + (theta - 1/4) k^2 S`` is a statement about mass and theta-excess stiffness while
    the air load is **dissipative**: it enters ``A``, never ``G0``. Pinning the bit-identity means
    a future change making the load non-dissipative fails loudly instead of mis-guarding silently.

    Measured on this rig (``K = 800``): 2.0440593233341828e-01 supported and
    2.0440593249574418e-01 free, the same to the last digit loaded and bare, and on both tiers.
    (The plan quotes 7.665222462503e-01 / 7.665222468590e-01 for the same four combinations at
    ``K = 3000`` — the margin is linear in ``K``, and only the *identity* is the assertion.)
    """
    bridge = make_vk_room_chain(tier=tier, boundary=boundary, walls=WALLS["all-lossy"])
    bare = make_vk_room_bare_twin(bridge)
    assert bridge.stability_margin == bare.stability_margin

    room = bridge.plate.room
    vk_room_pluck(bridge)
    e0 = bridge.energy() + room.energy()
    worst = 0.0
    for _ in range(400):
        bridge.step()
        room.step()
        worst = max(worst, abs(bridge.energy() + room.energy() - e0))
        assert bridge.converged, "read convergence per step -- a green ledger needs a fixed point"
    assert worst <= DRIFT_TOL * abs(e0)
    assert abs(bridge.plate.radiated_energy) > 1e-7 * abs(e0), "the channel must not be vacuous"


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_money_test_holds_with_the_string_as_the_only_excitation(boundary, tier):
    """``radiated_energy == room.injected`` when the plate starts at rest and the string drives it.

    Not a restatement of the batch-6 case above: there the plate carried the whole initial energy,
    here it carries none and every joule it radiates arrived through the bridge. A coupling that
    leaked at the *spring* rather than at the port would still leave this identity intact -- which
    is the point of asserting the scene total beside it, and of
    :func:`test_a_wrong_string_reaction_is_seen_by_the_total_and_not_the_money_test`.
    """
    bridge = make_vk_room_chain(tier=tier, boundary=boundary, walls=WALLS["all-lossy"])
    room = bridge.plate.room
    vk_room_pluck(bridge)
    e0 = bridge.energy() + room.energy()
    step_vk_room_chain(bridge, 400)
    radiated = bridge.plate.radiated_energy
    assert abs(radiated - room.injected) <= LEDGER_TOL * abs(radiated)
    assert abs(radiated) > 1e-7 * abs(e0)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_room_adds_no_outer_iteration(boundary, tier):
    """The bridge plan predicted a **third fixed point**. There is one, and this measures it.

    ``F = K eta^n`` depends only on time-``n`` state, so it is sweep-invariant and enters the RHS
    *outside* the Picard loop; the room's two terms are sweep-invariant and go into ``rhs_fixed``;
    ``T^T R T`` folds into ``A`` once at construction. So the loaded chain must take the **same**
    number of sweeps as the bare one at the same pluck.

    Phrase the result as "the room adds no outer iteration", never as "the room does not affect
    convergence": batch 6 measured that *coarsening* the room breaks the plate's fixed point (72
    sweeps at 57.9 kHz, NaN at 33 kHz). The room changes the operator the loop contracts on; it
    does not wrap a loop around it.
    """
    loaded = make_vk_room_chain(tier=tier, boundary=boundary, walls=WALLS["all-lossy"])
    bare = make_vk_room_bare_twin(loaded)
    counts = []
    for bridge in (loaded, bare):
        vk_room_pluck(bridge, 5e-2)
        room = getattr(bridge.plate, "room", None)
        worst = 0
        for _ in range(300):
            bridge.step()
            if room is not None:
                room.step()
            worst = max(worst, bridge.n_iters)
            assert bridge.converged
        counts.append(worst)
    assert counts[0] == counts[1], f"loaded took {counts[0]} sweeps, bare {counts[1]}"
    assert counts[0] > 1, "the nonlinear path must actually iterate for this to mean anything"


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_zero_bridge_stiffness_decouples_the_chain(boundary, tier):
    """``K = 0``: the plate never leaves rest and the string is bit-identical to a bare one.

    The disconnection check every bridge in ``connection.py`` carries, one composition up. It is
    sharper here than a tolerance: with no spring there is no force, so ``plate.u`` must be
    **exactly** zero -- not small -- and the room must have nothing injected into it at all.
    """
    bridge = make_vk_room_chain(tier=tier, boundary=boundary, K=0.0)
    alone = _bare_string(bridge)
    room = bridge.plate.room
    vk_room_pluck(bridge)
    alone.set_state(bridge.string.u.copy())
    for _ in range(200):
        bridge.step()
        room.step()
        alone.step()
    assert np.array_equal(bridge.plate.plate.u, np.zeros(bridge.plate.plate.n_live))
    assert bridge.plate.radiated_energy == 0.0
    assert np.array_equal(bridge.string.u, alone.u)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_nonlinear_false_chain_is_the_linear_bridge_bit_identical(boundary, tier):
    """``nonlinear=False`` reduces the whole chain to :class:`StringPlateBridge` on batch 3/4.

    The batch's strongest single assertion, and it discriminates three separate things at once:
    the ``rho_v``/``rho_s`` substitution inside the guard (1000x, and every ledger stays green),
    the drive-index defaulting on both sides, and the bridge's ``f_ext`` reaching the seam's RHS
    with batch 3's arithmetic. Byte-exact, because :class:`~physsynth.core.plate.VKPlate` with the
    flag off is byte-exact against :class:`~physsynth.core.plate.Plate` and the room load is the
    same matrix either way.
    """
    bridge = make_vk_room_chain(tier=tier, boundary=boundary, nonlinear=False)
    twin = _linear_chain_twin(bridge, tier)
    assert bridge.stability_margin == twin.stability_margin
    assert bridge.drive_index == twin.drive_index

    vk_room_pluck(bridge)
    vk_room_pluck(twin)
    for _ in range(200):
        bridge.step()
        bridge.plate.room.step()
        twin.step()
        twin.plate.room.step()
    assert np.array_equal(bridge.string.u, twin.string.u)
    assert np.array_equal(bridge.plate.plate.u, twin.plate.plate.u)
    assert bridge.plate.radiated_energy == twin.plate.radiated_energy


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_lossy_chain_is_monotone(boundary, tier):
    """Lossy walls plus a lossy plate: the scene total decreases, step after step.

    ``sigma > 0`` on the plate and an absorbing room are two independent sinks, and the spring is
    neither -- ``E_conn`` is a cross-time term that can rise and fall. Passivity is a statement
    about the **total**, which is why this is asserted on the sum and not on any part of it.
    """
    bridge = make_vk_room_chain(
        tier=tier, boundary=boundary, walls=WALLS["all-lossy"], sigma=2.0
    )
    room = bridge.plate.room
    vk_room_pluck(bridge)
    prev = bridge.energy() + room.energy()
    for _ in range(400):
        bridge.step()
        room.step()
        now = bridge.energy() + room.energy()
        assert now <= prev + 1e-14 * abs(prev), f"energy rose: {prev:.6e} -> {now:.6e}"
        prev = now


@pytest.mark.parametrize("tier", TIERS)
def test_band_overlap_decides_the_rigid_share_not_the_pluck(tier):
    """**The batch's claim.** Whether a string can play the gong nonlinearly is set by band overlap.

    A point force on a *free* plate feeds the ``{1, x, y}`` nullspace, and rigid motion stretches
    nothing -- ``l(w, w) = 0``, so the von Karman coupling is asleep in exactly that fraction
    (``test_the_piston_is_the_free_plate_s_fat_channel`` names the same fact for a piston start).
    Hold everything fixed and move only the string's **length**, which moves its fundamental while
    holding the wave impedance ``sqrt(T rho)``: as ``f1`` falls below the plate's first flexural
    mode the plate stops flexing and starts merely bouncing on the bridge.

    Measured on this rig — whose plate's first free flexural mode is **35.57 Hz** (measured, not
    taken off a table) — over 400 steps, rigid share baffled / suspended::

        L (m)   f1 (Hz)   f1/f_el   rigid share
        0.6     91.3      2.566     4.3% / 4.2%
        1.2     45.6      1.283     20.8% / 19.5%
        2.4     22.8      0.642     77.9% / 72.0%
        4.8     11.4      0.321     95.4% / 95.0%

    and it is a **cross-rig** reproduction: the plan's 57.9 kHz rig — 7x the sample rate, a plate
    3x smaller — gives 95.5% at ``f1/f_el = 0.28``, against 95.4% at 0.321 here. Note what does
    and does not carry across: the **transition** reproduces quantitatively at the low-ratio end,
    while above it the two rigs agree only in order (4.3% at 2.566 here, 1.7% at 2.00 there). The
    transition is the claim; the residual floor is not, which is why this test asserts
    monotonicity and the two endpoints rather than any interior value.

    Two things are asserted **not** to explain it. The peak displacement barely moves (``w/e``
    stays inside a factor of 2 while the rigid share moves 20x): ``w/e`` is not an amplitude when
    the drive is a point force (plan section 0.2). And the plate's *energy* share is deliberately
    not asserted — it counts rigid bouncing as energy the plate received, which is exactly the
    confusion this test exists to separate, and it is non-monotone here for that reason.
    """
    lengths = (0.6, 2.4, 4.8)
    runs = []
    for length in lengths:
        bridge = make_vk_room_chain(tier=tier, boundary="free", string_L=length)
        vk_room_pluck(bridge)
        runs.append(vk_room_rigid_share(bridge, 400))
    shares = [r["rigid"] for r in runs]
    wes = [r["peak_we"] for r in runs]

    assert shares[0] < 0.10, f"a string above the plate's 1st mode should flex it: {shares[0]:.3f}"
    assert shares[-1] > 0.85, f"a string far below it should only bounce it: {shares[-1]:.3f}"
    assert shares == sorted(shares), f"the share must rise monotonically with L: {shares}"
    assert shares[-1] / shares[0] > 10.0
    assert max(wes) / min(wes) < 3.0, "peak w/e is NOT what moved -- that is the whole point"


@pytest.mark.parametrize("tier", TIERS)
def test_a_wrong_string_reaction_is_seen_by_the_total_and_not_the_money_test(tier):
    """The fourth insufficiency: this batch adds ``E_conn``, and only one detector sees it break.

    Four batches, four blind spots (batch 3: the conserved total cannot see a wrong ``R_j``;
    batch 4: the money test cannot see the two-faces ``2``; batch 5: it cannot see which velocity
    made the ``q``; batch 6: it is arithmetic on whatever ``w^{n+1}`` came out of the solve).
    Here the slip is the string's reaction impulse ``beta_s`` -- Newton's third law at the spring,
    which is upstream of the port entirely. Measured: the scene total goes 2.8e-15 -> **8.8e-02**
    while ``|radiated - injected|`` sits at 7.8e-15 either way, because the money test is a
    property of the port relation alone and the port never sees the string.
    """
    drifts, gaps = [], []
    for scale in (1.0, 2.0):
        bridge = make_vk_room_chain(tier=tier, boundary="free", walls=WALLS["all-lossy"])
        bridge.beta_s *= scale
        room = bridge.plate.room
        vk_room_pluck(bridge)
        e0 = bridge.energy() + room.energy()
        worst = 0.0
        for _ in range(400):
            bridge.step()
            room.step()
            worst = max(worst, abs(bridge.energy() + room.energy() - e0))
        drifts.append(worst / abs(e0))
        gaps.append(abs(bridge.plate.radiated_energy - room.injected)
                    / abs(bridge.plate.radiated_energy))

    assert drifts[0] <= DRIFT_TOL and gaps[0] <= LEDGER_TOL, "the correct chain must be green"
    assert drifts[1] > 1e-4, f"the scene total must CATCH a wrong reaction, got {drifts[1]:.2e}"
    assert gaps[1] <= LEDGER_TOL, f"...and the money test must not, got {gaps[1]:.2e}"


@pytest.mark.parametrize("tier", TIERS)
def test_every_detector_is_blind_to_a_drive_index_that_differs_between_two_runs(tier):
    """And the one **no** detector sees, because it is a difference between two valid runs.

    The batch's quantitative content -- departure from the chain's own linear self -- is computed
    from two separate runs, and both derive ``drive_index`` from
    ``pickup_index_at(0.3 Lx, 0.4 Ly)``. Let the two disagree and the departure figure moves
    **1.7x** while every ledger in the family stays at machine precision: each run is internally
    consistent, so there is nothing inconsistent to detect. The three detectors are jointly
    insufficient against a **comparison**, not only against a coefficient -- which is this batch's
    addition to the family rule, and why the plan requires ``drive_index`` to be passed explicitly
    wherever two chains are compared.
    """
    ref = make_vk_room_chain(tier=tier, boundary="free")
    other = ref.plate.pickup_index_at(0.35 * ref.plate.Lx, 0.45 * ref.plate.Ly)
    assert other != ref.drive_index

    def history(nonlinear, drive_index):
        bridge = make_vk_room_chain(
            tier=tier, boundary="free", nonlinear=nonlinear, drive_index=drive_index
        )
        room = bridge.plate.room
        vk_room_pluck(bridge, 5e-2)
        e0 = bridge.energy() + room.energy()
        hist = np.empty((300, bridge.plate.plate.n_live))
        worst = 0.0
        for n in range(300):
            bridge.step()
            room.step()
            hist[n] = bridge.plate.plate.u
            worst = max(worst, abs(bridge.energy() + room.energy() - e0))
        gap = abs(bridge.plate.radiated_energy - room.injected)
        return hist, worst / abs(e0), gap / abs(bridge.plate.radiated_energy)

    nl, drift_n, gap_n = history(True, ref.drive_index)
    matched, drift_a, gap_a = history(False, ref.drive_index)
    mismatched, drift_b, gap_b = history(False, other)

    for drift, gap in ((drift_n, gap_n), (drift_a, gap_a), (drift_b, gap_b)):
        assert drift <= DRIFT_TOL and gap <= LEDGER_TOL, "every run is individually green"

    honest = np.linalg.norm(nl - matched) / np.linalg.norm(matched)
    corrupt = np.linalg.norm(nl - mismatched) / np.linalg.norm(mismatched)
    assert honest > 1e-3, "the departure must be a live number for this to say anything"
    assert corrupt / honest > 1.3, f"the comparison moved only {corrupt / honest:.2f}x"
