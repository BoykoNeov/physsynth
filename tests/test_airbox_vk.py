"""The gong in the room: model #6 as a baffled and suspended surface (air-box batch 6).

Batch 5 put the *membrane* in the air box; this puts the **nonlinear plate** there, which completes
§12H's model list — every resonator in ``physsynth/core/`` that can be a surface now can be one in
the room. What it buys is not a refinement of the radiation. Every other radiating thing in this
repo is linear in its excitation, so radiated fraction, directivity and dipole-over-baffled are
amplitude-**invariant** by construction; the von Kármán coupling is quadratic, so the *shape* of the
motion evolves during a single strike, and shape is exactly what :class:`SurfacePort` was built to
make audible. Batch 6's claim is therefore:

    **a loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not** —

which no ``R(omega)`` in ``radiation.py`` can state, because a scalar-per-frequency load has one
pattern per frequency and cannot change it mid-strike.

**This file is Commit A, whose entire claim is "zero new physics, here is the proof."** The batch
splits because :class:`~physsynth.core.plate.VKPlate` with ``nonlinear=False`` is already
bit-identical to :class:`~physsynth.core.plate.Plate`; therefore
:class:`~physsynth.core.airbox.RoomLoadedVKPlate` with the flag off must be bit-identical to
:class:`~physsynth.core.airbox.RoomLoadedPlate`, and likewise suspended. That single equality
discriminates the whole commit, because it is simultaneously a check on:

* **the ``rho_v`` / ``rho_s`` substitution** — model #6 has no ``rho``, and writing the volumetric
  density where the areal one belongs leaves the air load 1000x too weak at ``e = 1 mm`` while every
  ledger still telescopes against the pressure it used. **Nothing green turns red**; only this test
  turns red. It is the reason the commit exists on its own.
* **the RHS's operand order**, which must reproduce :meth:`RoomLoadedPlate.step`'s statement for
  statement — the seam's ``f_ext`` path included, which is the first time this seam's force path has
  had a byte-exact counterpart at all (batch 5's had none and was pinned by a static-deflection
  oracle instead).
* **the factorization's inputs**: same ``a_bare``, same ``_load_scale``, same load matrix, hence the
  same ``splu`` — asserted directly through ``nnz_growth`` and ``lu_nnz`` as well as through the
  trajectory.

``sigma > 0`` runs in every case beside a non-zero ``f_ext``, and that pairing is deliberate: the
``(1 + sigma k)`` factor in ``A`` and the ``sigma k * w^{n-1}`` term in the RHS are where a density
slip could hide *asymmetrically* between the matrix and the right-hand side, and a lossless run
never exercises them.

The guards, the two-parameter money test and the physics are Commits B and C
(``docs/dev/air-box-vk-plate-plan.md`` §7).
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_DIPOLE_INDEX,
    make_room_loaded_vk_plate,
    make_surface_room,
    make_suspended_vk_plate,
    vk_linear_twin,
    vk_strike,
)

from physsynth.core.airbox import RoomLoadedPlate, RoomSuspendedPlate

BOUNDARIES = ("supported", "free")
TIERS = ("baffled", "suspended")
SIGMAS = (0.0, 2.0)


def _make_vk(tier, **kw):
    maker = make_room_loaded_vk_plate if tier == "baffled" else make_suspended_vk_plate
    return maker(**kw)


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


def _drive(plate, peak=1.0):
    """A constant off-centre nodal force (N), large enough that ``f_ext`` is not decorative.

    Peak 1 N on this rig moves the plate about a thickness over the 50 steps the regressions run,
    i.e. the same scale as the initial strike — so a wrong coefficient on the force term shows up in
    the trajectory rather than in the last digits.
    """
    x, y = plate.X[plate.mask], plate.Y[plate.mask]
    width = 0.12 * plate.Lx
    return peak * np.exp(
        -(((x - 0.42 * plate.Lx) ** 2 + (y - 0.38 * plate.Ly) ** 2) / (width * width))
    )


# -- the split point: coupling off is batch 3 / batch 4, bit-for-bit -----------------------


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_nonlinear_false_is_the_linear_room_loaded_plate_bit_identical(boundary, tier, sigma):
    """``RoomLoadedVKPlate(nonlinear=False)`` must reproduce :class:`RoomLoadedPlate` **exactly**.

    Not "closely" — ``array_equal`` on both stored levels, on the coupling ledger and on the
    energy, for 50 steps with a non-zero ``f_ext`` throughout. See the module docstring for what
    each half of that equality guards; the short version is that the one substitution it makes
    (``rho -> rho_s``) is invisible to every energy report in the repo.
    """
    inst = _make_vk(tier, boundary=boundary, sigma=sigma, nonlinear=False)
    vk = inst.plate
    ref = _make_linear(tier, vk)

    u0 = vk_strike(vk, 1e-4)
    f_ext = _drive(vk)
    inst.set_state(u0)
    ref.set_state(u0)
    for _ in range(50):
        inst.step(f_ext)
        inst.room.step()
        ref.step(f_ext)
        ref.room.step()
        assert np.array_equal(vk.u, ref.plate.u), "the linear reduction must be bit-identical"
    assert np.array_equal(vk.u_prev, ref.plate.u_prev)
    assert inst.radiated_energy == ref.radiated_energy
    assert inst.energy() == ref.energy()
    assert inst.volume_velocity == ref.volume_velocity


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_loaded_factorization_matches_the_linear_one(boundary, tier):
    """Same ``a_bare``, same ``_load_scale``, same load matrix — so the same ``splu``.

    The trajectory test above would catch a wrong factorization too, but only as a difference in the
    last digits of a state vector. This says *where* it came from, and it is the assertion that a
    ``rho_v`` slip fails first: ``_load_scale = k / (2 rho_s)`` is one of the two places the areal
    density enters (the other is the ``f_ext`` and open-circuit divide), and a factor of 1000 there
    changes the stored pattern as well as the values.
    """
    inst = _make_vk(tier, boundary=boundary, nonlinear=False)
    ref = _make_linear(tier, inst.plate)
    assert inst._load_scale == ref._load_scale
    assert inst._denominator == ref._denominator
    assert inst.nnz_growth == ref.nnz_growth
    assert inst.lu_nnz == ref.lu_nnz


# -- the loop hook is wired, and it does not mutate the model ------------------------------


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_loaded_factorization_is_never_assigned_to_the_model(boundary, tier):
    """The loop hook takes ``lu`` as an argument (:meth:`_VKPlateSurface.solve`, rule 1).

    Assigning it to ``vk._lu`` would make bare-versus-loaded unobservable and would quietly turn
    every reduction test in this file into a tautology.
    """
    inst = _make_vk(tier, boundary=boundary)
    assert inst.plate._lu is not inst._lu_loaded


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_nonlinear_path_runs_inside_the_load(boundary, tier):
    """A smoke test, and named as one: the Picard loop is live and converging under the air load.

    It asserts no physics — Commit B owns the guards and Commit C the claim. What it does assert is
    that the flag is not decorative: at ``w = 2e`` the coupled run must *diverge from* the linear
    one (else the loop is not running), the iteration must converge on **every** step (never sampled
    at the end — a plate can sit at the sweep cap throughout a run and still converge on its last
    step), and both cached ``F`` levels must be non-zero, which is the roll that neither predecessor
    seam had.
    """
    inst = _make_vk(tier, boundary=boundary)
    linear = _make_vk(tier, boundary=boundary, nonlinear=False)
    u0 = vk_strike(inst.plate, 2.0 * inst.plate.e)
    inst.set_state(u0)
    linear.set_state(u0)
    for _ in range(20):
        inst.step()
        inst.room.step()
        linear.step()
        linear.room.step()
        assert inst.converged, f"the Picard loop hit the cap at step {inst.n}"
    assert np.all(np.isfinite(inst.plate.u))
    assert not np.array_equal(inst.plate.u, linear.plate.u), "the coupling did nothing"
    assert np.any(inst.plate.F != 0.0) and np.any(inst.plate.F_prev != 0.0)
