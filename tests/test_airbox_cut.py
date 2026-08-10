"""The cut: a rigid, zero-thickness internal partition on a plane of faces (air-box batch 4).

The cut is batch 4's **only** new machinery. Prescribing a face velocity and injecting a ``-q``/
``+q`` pair on the two node planes that straddle it are the same arithmetic (``A_face * w_z = W``
identically), so the port, the injection weights and the ``injected`` ledger are all batch 3's —
what is genuinely new is the *obstacle* — and an obstacle is a set of faces where ``u`` is held
at zero.

Three things make it worth its own file:

* **It costs nothing to book.** A cut face's ``u`` is identically zero at every half-step, so its
  contribution to the kinetic sum is identically zero: the energy identity needs no new term and no
  exclusion list. The area-coupling plan anticipated "removing those faces from the kinetic-energy
  sum"; they remove themselves. :func:`test_a_cut_room_still_conserves` is that, measured.
* **It brings an exact modal oracle that is a genuinely new one.** A full cut splits the room into
  two independent rooms of length ``(m + 1/2) h`` and ``(N - m - 1/2) h`` — summing to ``N h``
  exactly, because the cut lies on a *face*, half a cell past the last node on each side. That end
  is **face-centered**, so the exact discrete eigenvector along the cut axis is ``cos(n pi i/(m +
  1/2))`` and **not** the room's own ``cos(n pi i/N)``. Batch-1-tier: machine precision, not a rate.
* **It must be additive, and the failure mode of a single-slot cut is silent.** With
  ``room._cut = ...`` a second plate's cut replaces the first's, the first keeps injecting its
  ``-q``/``+q`` pair and stops blocking — i.e. it degrades to a transparent doublet — while every
  ledger stays green. :func:`test_cuts_are_additive` pins the accumulation.

The oracle here exercises the cut **primitive**, never a port: :class:`InteriorSurfacePort`'s rim
refusal means a legal port can never span a full cross-section, so a *port* can never seal the room.
A hand-placed cut can, and the corollary — a legal interior plate always has a diffraction path
around it — is measured in :func:`test_a_full_cut_isolates_exactly`.
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_SURFACE_N,
    airbox_noise,
    make_airbox,
    make_cut_room,
    make_surface_room,
    sub_room_mode,
)

from physsynth.core.airbox import PLANES, impedance_from_zeta

DRIFT_TOL = 1e-12  # relative to the acoustic scale
FIELD_TOL = 1e-12  # the modal oracle, absolute against an amplitude of 1
WALLS = {"rigid": "rigid", "lossy": impedance_from_zeta(4.0)}


# -- the ledger --------------------------------------------------------------------------


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_a_cut_room_still_conserves(plane, wall_name):
    """Energy flat with a cut **and** a driven source — the identity needs no new term.

    Measured 6.5e-15 (rigid) and 8.0e-15 (lossy) of the acoustic scale over 400 steps. The source is
    there on purpose: a cut room with nothing driving it would leave the ``injected`` channel out of
    the statement, and it is the *sum* ``acoustic + dissipated - injected`` that is flat.
    """
    box = make_airbox(walls=WALLS[wall_name])
    box.add_cut(plane, 3)
    airbox_noise(box, amplitude=1.0)
    e0, scale = box.energy(), box.acoustic_energy()
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(400):
        box.inject(1e-4 * rng.standard_normal(), at=(0.2, 0.2, 0.1))
        box.step()
        worst = max(worst, abs(box.energy() - e0))
    assert worst <= DRIFT_TOL * scale


def test_a_cut_face_carries_no_velocity_at_any_half_step():
    """Both stored half-steps are zero on a cut face — including on a room already in motion.

    The zeroing lives in :meth:`AirBox._momentum`, the single place both ``step`` and ``set_state``
    produce velocities, so the consistent start cannot leave a live velocity behind either. Adding a
    cut to a *moving* room clears what is already there, which is the case a lazier implementation
    would miss.
    """
    box = airbox_noise(make_airbox())
    for _ in range(5):
        box.step()
    assert np.any(box.uz[:, :, 3] != 0.0), "the room must be moving there, or this proves nothing"
    box.add_cut("z", 3)
    for _ in range(5):
        assert np.all(box.uz[:, :, 3] == 0.0)
        assert np.all(box.uz_prev[:, :, 3] == 0.0)
        box.step()


# -- the oracle --------------------------------------------------------------------------


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("index", [1, 3, 4])
@pytest.mark.parametrize("side", ["lo", "hi"])
def test_the_sub_rooms_have_exact_half_offset_modes(plane, index, side):
    """``cos(n pi i / (m + 1/2))`` is exact to machine precision — a new oracle, not a restriction.

    The room's own tensor-cosine oracle (batch 1) has a **node-centered** mirror at each wall. The
    cut end is **face-centered**: the mirror plane sits between nodes, ghost condition
    ``p_{m+1} = p_m``, which moves the eigenvector's denominator by exactly half a cell. Getting
    that half wrong is the sort of error that looks like scheme inaccuracy, so it is asserted at
    machine precision (measured 1.0e-14 … 7.4e-14 field error, 3.3e-16 … 8.2e-16 drift) rather than
    as a rate.

    The other sub-room stays **exactly** zero throughout, which is the same statement as
    :func:`test_a_full_cut_isolates_exactly` seen from the initial condition's side.
    """
    box = make_airbox()
    axis = PLANES.index(plane)
    if index > box.N[axis] - 1:
        pytest.skip(f"the room has only {box.N[axis]} cells along {plane}")
    box.add_cut(plane, index)
    p0, u0, f = sub_room_mode(box, plane, index, side, 1, transverse=(1, 0))
    box.set_state(p0, u0)
    e0 = box.energy()

    other = [slice(None)] * 3
    other[axis] = slice(index + 1, None) if side == "lo" else slice(None, index + 1)
    worst = drift = 0.0
    for step in range(1, 201):
        box.step()
        want = np.cos(2.0 * np.pi * f * step * box.k) * p0
        worst = max(worst, float(np.max(np.abs(box.p - want))))
        drift = max(drift, abs(box.energy() - e0))
        assert np.all(box.p[tuple(other)] == 0.0), "the far sub-room must stay EXACTLY at rest"
    assert worst <= FIELD_TOL
    assert drift <= DRIFT_TOL * e0


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("index", [0, 2, 5])
def test_the_sub_room_lengths_sum_to_the_room(plane, index):
    """``(m + 1/2) h + (N - m - 1/2) h == N h`` — no cell is lost, and neither side is the room."""
    box = make_airbox()
    axis = PLANES.index(plane)
    if index > box.N[axis] - 1:
        pytest.skip(f"the room has only {box.N[axis]} cells along {plane}")
    lo = (index + 0.5) * box.h
    hi = (box.N[axis] - index - 0.5) * box.h
    assert lo + hi == pytest.approx(box.L_actual[axis], rel=1e-15)
    # ... and neither side is the room's own grid restricted: a half cell shifts BOTH ends, so
    # neither sub-room length is a whole number of cells.
    assert not float(lo / box.h).is_integer()
    assert not float(hi / box.h).is_integer()


# -- what a cut does to sound -------------------------------------------------------------


def test_a_full_cut_isolates_exactly():
    """A full cut passes **exactly** ``0.0``; a partial cut does not — so a port can never seal.

    The exact zero is the assertion that a cut is a *rigid* boundary and not a strong impedance. Its
    corollary is physical rather than a limitation to hide: a legal interior plate can only ever be
    a partial cut (:class:`InteriorSurfacePort` refuses a stencil reaching the room's rim), so it
    always has a diffraction path around it. Measured, against the same source and the same probe:
    full ``0.000e+00``, partial ``8.3e-01`` of the uncut room's own peak — the diffraction path
    around a half-width partition is barely an obstacle at all in a room this size.
    """
    def peak_across(*, cut, extent=None):
        box = make_airbox()
        if cut:
            box.add_cut("z", 3, extent)
        peak = 0.0
        for _ in range(220):
            box.inject(1e-3, at=(0.4, 0.3, 0.1))
            box.step()
            peak = max(peak, float(np.max(np.abs(box.p[:, :, 4:]))))
        return peak

    open_peak = peak_across(cut=False)
    assert peak_across(cut=True) == 0.0
    partial = peak_across(cut=True, extent=((0, 4), (0, 4)))
    assert partial > 0.01 * open_peak, f"partial/open = {partial / open_peak:.3e}"


def test_cuts_are_additive_so_a_second_one_cannot_un_block_the_first():
    """Two cuts coexist; the first still blocks. A single-slot cut would fail this silently.

    Measured with a single slot: the first plate degrades to the phantom (keeps injecting, stops
    blocking) while the scene stays perfectly green — at the refinement where the phantom is 40x
    weaker, a silent 40x error in one instrument. Hence the boolean union, and hence a port records
    its own face set.
    """
    box = make_airbox()
    box.add_cut("z", 3)
    faces_one = box.cut_faces
    box.add_cut("x", 2)
    assert box.cut_faces == faces_one + (box.N[1] + 1) * (box.N[2] + 1)
    # The source at node (4, 3, 1) sits in the sub-room {x >= 3, z <= 3}; both other quadrants
    # must stay at rest, and it is the FIRST cut's quadrant that a single-slot implementation
    # would have re-opened.
    for _ in range(220):
        box.inject(1e-3, at=(0.4, 0.3, 0.1))
        box.step()
    assert np.all(box.p[:, :, 4:] == 0.0), "the z cut must still block after the x cut was added"
    assert np.all(box.p[:3, :, :] == 0.0), "and the x cut must block too"


def test_overlapping_hand_placed_cuts_are_idempotent():
    """A boolean union: cutting the same faces twice is the same room, and is not refused.

    Only a **port**'s cut is exclusive (its cut and its ``-q``/``+q`` pair are two halves of one
    object). Two hand-placed partitions occupying the same faces are simply one partition.
    """
    box = make_airbox()
    box.add_cut("z", 3, ((0, 5), (0, 5)))
    once = box.cut_faces
    box.add_cut("z", 3, ((2, 4), (1, 3)))
    assert box.cut_faces == once


def test_cut_faces_counts_the_full_cross_section():
    box = make_surface_room(N=AIRBOX_SURFACE_N)
    assert box.cut_faces == 0
    box.add_cut("z", 4)
    assert box.cut_faces == (AIRBOX_SURFACE_N[0] + 1) * (AIRBOX_SURFACE_N[1] + 1)
    room = make_cut_room(plane="y", index=3)
    assert room.cut_faces == (AIRBOX_SURFACE_N[0] + 1) * (AIRBOX_SURFACE_N[2] + 1)


# -- refusals -----------------------------------------------------------------------------


def test_refuses_an_unknown_plane():
    with pytest.raises(ValueError, match="unknown plane"):
        make_airbox().add_cut("z0", 3)


@pytest.mark.parametrize("index", [-1, 100])
def test_refuses_a_cut_index_outside_the_faces(index):
    """The room's walls are at NODE planes 0 and N and are already rigid — not cut positions."""
    with pytest.raises(ValueError, match="out of range"):
        make_airbox().add_cut("z", index)


def test_refuses_a_malformed_extent():
    box = make_airbox()
    with pytest.raises(ValueError, match="inclusive node-index range"):
        box.add_cut("z", 3, ((0, 4), (3, 1)))
    with pytest.raises(ValueError, match="extent must be"):
        box.add_cut("z", 3, (4, 4))
