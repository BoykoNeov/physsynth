"""The 3-D air box: energy conservation, wall passivity, source booking, and the reductions.

The structural tier (HANDOFF §6.1-§6.2) for the distributed air node. Three energy channels have to
close against each other exactly:

- **stored** — the cross-time acoustic energy in the room;
- **dissipated** — what the impedance walls absorb (monotone non-decreasing);
- **injected** — the work the soft source does on the room.

``E = stored + dissipated - injected`` is flat to machine precision for *any* wall configuration and
*any* source, which is the whole passivity claim: the walls can only take, the source can only give,
and neither can leak. The velocity term is the **cross-time product** ``u^{n+1/2} u^{n-1/2}`` —
collapsing it to a same-time square is the classic bug these tests exist to catch, and in 3-D it has
three chances to happen.

The other structural claim is the **reduction ledger** entry: one wall closure covers rigid, open
and impedance, with ``Z = inf`` recovering the rigid box *bit-identically* and ``Z = 0`` pinning the
face to exactly zero. That is what makes "three boundary types" one code path rather than three.
"""

import numpy as np
import pytest
from helpers import airbox_noise, gaussian_pulse, make_airbox

from physsynth.core.airbox import C0_AIR, FACES, RHO0_AIR, AirBox, impedance_from_zeta

DRIFT_TOL = 1e-10  # acceptance criterion 1 (the same bar as every other resonator)
Z_MATCHED = RHO0_AIR * C0_AIR  # the normal-incidence matched wall


def _drift(box, steps):
    """Relative spread of the conserved total over a run — the primary bug detector."""
    e0 = box.energy()
    lo = hi = e0
    for _ in range(steps):
        box.step()
        e = box.energy()
        lo, hi = min(lo, e), max(hi, e)
    return (hi - lo) / abs(e0)


# -- Conservation: the lossless rigid room ------------------------------------------------------
#    Several aspect ratios, because the tensor-trapezoid weights only cancel in the energy pairing
#    if every axis is weighted independently; a shared-weight slip survives a cube and dies here.
@pytest.mark.parametrize(
    "L", [(0.9, 0.7, 0.6), (1.2, 0.3, 0.5), (0.6, 0.6, 0.6), (1.5, 0.2, 0.2)]
)
def test_energy_conserved_rigid(L):
    box = airbox_noise(make_airbox(L=L))
    assert _drift(box, 400) < DRIFT_TOL


@pytest.mark.parametrize("cfl", [0.3, 0.6, 0.9, 0.999])
def test_energy_conserved_across_courant(cfl):
    """Conservation is an algebraic identity, not a special value of lambda. Stops just short of
    the ceiling on purpose: *at* ``lambda = 1/sqrt(3)`` the identity still holds but the field grows
    linearly (the corner mode is defective), so the relative drift loses digits to a growing field
    rather than to the scheme. That case is pinned in ``test_airbox_modal`` where it belongs."""
    box = airbox_noise(make_airbox(cfl=cfl))
    assert _drift(box, 400) < DRIFT_TOL


def test_energy_positive_and_no_source_or_loss_channel():
    """With rigid walls and no source, the conserved total *is* the stored energy, bit-for-bit."""
    box = airbox_noise(make_airbox())
    for _ in range(50):
        box.step()
        assert box.energy() > 0.0
    assert box.dissipated_energy() == 0.0
    assert box.injected_energy() == 0.0
    assert box.energy() == box.acoustic_energy()


# -- Passivity: impedance walls take, and only take ---------------------------------------------
@pytest.mark.parametrize("zeta", [0.25, 1.0, 4.0])
def test_impedance_wall_passive_and_booked(zeta):
    """``stored`` falls monotonically while ``stored + dissipated`` stays flat: the wall channel
    captures *exactly* what leaves. A sign error in the flux shows up here and nowhere else."""
    box = airbox_noise(make_airbox(walls={"x0": impedance_from_zeta(zeta)}))
    e0 = box.energy()
    prev = box.acoustic_energy()
    worst = 0.0
    for _ in range(600):
        box.step()
        worst = max(worst, abs(box.energy() - e0) / abs(e0))
        stored = box.acoustic_energy()
        assert stored <= prev + 1e-18, "stored energy rose: the wall put energy back in"
        prev = stored
    assert worst < DRIFT_TOL
    assert box.dissipated_energy() > 0.0


def test_every_face_absorbing_still_books():
    """All six walls lossy at once — the case where edge and corner nodes pay into two and three
    faces. Summed admittances in ``beta`` and summed flux in the book have to agree, or the total
    drifts exactly where the geometry is most awkward."""
    box = airbox_noise(make_airbox(walls=Z_MATCHED))
    assert _drift(box, 600) < DRIFT_TOL
    assert box.dissipated_energy() / box.energy() > 0.9, "a matched room should drain"


def test_open_face_is_lossless():
    """An ideal pressure-release face reflects perfectly and radiates nothing (the bore's lesson):
    its flux ``p u`` vanishes because ``p`` does, so the room is still conservative."""
    box = airbox_noise(make_airbox(walls={"y1": "open", "x0": "open"}))
    assert _drift(box, 400) < DRIFT_TOL
    assert box.dissipated_energy() == 0.0


# -- The source is an energy channel, not a free lunch ------------------------------------------
def test_source_booking_flat():
    """``stored + dissipated - injected`` is flat while a soft source drives an absorbing room —
    the three-channel statement, all live at once."""
    box = make_airbox(walls={"x1": Z_MATCHED})
    box.set_state(np.zeros(box.p.shape))
    q, _, _ = gaussian_pulse(box.fs, 900.0)
    worst = 0.0
    for n in range(500):
        box.inject(q(n * box.k))
        box.step()
        worst = max(worst, abs(box.energy()))
    injected = box.injected_energy()
    assert injected > 0.0, "the source did no work"
    assert worst / injected < DRIFT_TOL


def test_source_at_a_wall_node_books_correctly():
    """A source sitting *on* a wall shares its node with the boundary closure. The injection is
    folded into the pressure update before the 1x1 wall solve, so the booked ``pbar`` is the
    post-solve one; taking it pre-solve would leak here and only here."""
    box = make_airbox(walls={"x0": Z_MATCHED}, source=(0.0, 0.3, 0.3))
    box.set_state(np.zeros(box.p.shape))
    q, _, _ = gaussian_pulse(box.fs, 900.0)
    for n in range(400):
        box.inject(q(n * box.k))
        box.step()
    assert abs(box.energy()) / box.injected_energy() < DRIFT_TOL


def test_multiple_injections_accumulate():
    """Two sources in one step are two bookings, not one.

    Opposite signs at *different* nodes, so each sees its own ``pbar`` and the two contributions do
    **not** cancel — the denominator below is a real quantity (~8e-6 J), not a near-zero that would
    make the ratio meaningless. Booking only the last injection of a step would show up here."""
    box = make_airbox()
    box.set_state(np.zeros(box.p.shape))
    for _ in range(20):
        box.inject(1e-3, at=(0.2, 0.2, 0.2))
        box.inject(-1e-3, at=(0.7, 0.5, 0.4))
        box.step()
    assert box.injected_energy() > 1e-9, "the denominator collapsed; the test would be vacuous"
    assert abs(box.energy()) / abs(box.injected_energy()) < DRIFT_TOL


# -- The reduction ledger: one closure, three boundary types ------------------------------------
def test_rigid_token_is_bit_identical_to_infinite_impedance():
    """``walls="rigid"`` and ``Z = inf`` are the same wall, and the equality is exact, not close —
    the family's standing bar for a reduction (``R=0`` -> bare body, ``sigma_1=0`` -> model #2)."""
    a = airbox_noise(make_airbox(walls="rigid"))
    b = airbox_noise(make_airbox(walls=dict.fromkeys(FACES, np.inf)))
    for _ in range(200):
        a.step()
        b.step()
    assert np.array_equal(a.p, b.p)
    assert a.energy() == b.energy()


def test_open_token_pins_the_face_to_exactly_zero():
    box = airbox_noise(make_airbox(walls={"y1": "open", "z0": 0.0}))
    for _ in range(50):
        box.step()
        assert np.max(np.abs(box.p[:, -1, :])) == 0.0
        assert np.max(np.abs(box.p[:, :, 0])) == 0.0


def test_large_impedance_approaches_rigid():
    """The *numerical* limit, as opposed to the normalization above: a very stiff but finite wall
    goes through the full beta arithmetic and still tracks the rigid room. This is the one that
    would catch a wrong power of ``h`` in ``beta``."""
    stiff = airbox_noise(make_airbox(walls=1e12))
    rigid = airbox_noise(make_airbox(walls="rigid"))
    for _ in range(200):
        stiff.step()
        rigid.step()
    err = np.max(np.abs(stiff.p - rigid.p)) / np.max(np.abs(rigid.p))
    assert err < 1e-6


# -- Construction guards -------------------------------------------------------------------------
def test_cfl_rejected_above_the_3d_ceiling():
    with pytest.raises(ValueError, match="CFL"):
        make_airbox(cfl=1.001)


def test_cfl_accepted_exactly_at_the_ceiling():
    box = make_airbox(cfl=1.0)
    assert box.lam == pytest.approx(1.0 / np.sqrt(3.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"L": (1.0, 1.0)},
        {"L": (1.0, -1.0, 1.0)},
        {"h": 5.0},                       # coarser than the room: an axis with no cells
        {"walls": "squishy"},
        {"walls": {"x2": 1.0}},
        {"walls": {"x0": -1.0}},
    ],
)
def test_bad_construction_refused(kwargs):
    base = {"L": (0.9, 0.7, 0.6), "fs": 12000.0, "h": 0.1}
    with pytest.raises(ValueError):
        AirBox(**{**base, **kwargs})


def test_a_point_outside_the_room_is_refused_not_relocated():
    """Snapping *within* the room is the resolution; snapping a point from outside it onto a wall
    would be a silently wrong answer, so it raises. Half a cell past the last node still counts as
    inside — that is rounding, not relocation."""
    box = make_airbox()
    Lx, Ly, Lz = box.L_actual
    assert box.node_index((Lx + 0.4 * box.h, Ly, Lz)) == box.N  # rounds back onto the wall
    for outside in [(Lx + box.h, 0.3, 0.3), (-box.h, 0.3, 0.3), (0.3, 0.3, Lz + box.h)]:
        with pytest.raises(ValueError, match="outside the room"):
            box.pressure_at(outside)


def test_grid_snap_is_reported_not_hidden():
    """A room that does not divide by ``h`` is snapped, and says so — the juari precedent."""
    box = AirBox(L=(0.93, 0.71, 0.58), fs=12000.0, h=0.1)
    assert box.N == (9, 7, 6)
    assert box.L_actual == pytest.approx((0.9, 0.7, 0.6))
    assert box.L == (0.93, 0.71, 0.58)
    assert box.snapped((0.44, 0.0, 0.0)) == pytest.approx((0.4, 0.0, 0.0))
