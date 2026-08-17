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

**The batch splits, and this file holds the first two commits.** Commit A's entire claim is "zero
new physics, here is the proof"; Commit B is the guards. The split is possible because
:class:`~physsynth.core.plate.VKPlate` with ``nonlinear=False`` is already
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

**Commit B's guards, and what they found.** This batch has a second error axis no previous one had:
model #6 conserves only at the **Picard fixed point**, so the iteration tolerance ``couple_tol`` is
an error source sitting alongside the air load. Measured on all three detectors at once
(:func:`test_couple_tol_moves_the_total_and_not_the_money_test`), they split cleanly — the scene
total and ``last_residual`` both track ``couple_tol`` almost exactly, while
``radiated == injected`` stays at rounding at *every* tolerance. So the money test is blind for a
**third distinct reason**: batch 4's was a ``2`` inside the factorization, batch 5's was which
velocity produced the ``q``, and here it is that the identity is arithmetic on whatever ``w^{n+1}``
came out of the solve — an under-converged one is ported self-consistently. The family's standing
rule (no single detector is sufficient) survives another batch, and the self-certifying half is
that **loaded drift falls with ``couple_tol`` at the same rate as unloaded**: the air adds no error
floor of its own.

The primary guard remains batch 4's coupled residual, now the residual of *two* fields — and
``F^{n-1}`` has to be captured before the step, because the commit rolls it away.

The physics is Commit C (``docs/dev/air-box-vk-plate-plan.md`` §7.6–§7.8).
"""

import numpy as np
import pytest
from helpers import (
    AIRBOX_DIPOLE_INDEX,
    make_air_vk_plate,
    make_room_loaded_vk_plate,
    make_surface_room,
    make_suspended_vk_plate,
    surface_scene_energy,
    vk_linear_twin,
    vk_strike,
)
from scipy import sparse
from scipy.sparse.linalg import splu

from physsynth.core.airbox import (
    RoomLoadedPlate,
    RoomLoadedVKPlate,
    RoomSuspendedPlate,
    RoomSuspendedVKPlate,
    impedance_from_zeta,
)

BOUNDARIES = ("supported", "free")
TIERS = ("baffled", "suspended")
SIGMAS = (0.0, 2.0)
DRIFT_TOL = 1e-12   # the scene total, relative -- necessary, not sufficient (see the module head)
LEDGER_TOL = 1e-12  # |radiated - injected| / |radiated| -- necessary, ALSO not sufficient
WALLS = {
    "rigid": "rigid",
    "all-lossy": impedance_from_zeta(4.0),
    "one-lossy-wall": {"z0": impedance_from_zeta(3.0)},
}


def _make_vk(tier, **kw):
    maker = make_room_loaded_vk_plate if tier == "baffled" else make_suspended_vk_plate
    return maker(**kw)


def _run(inst, steps, f_ext=None):
    """Step instrument and room in the contract's order: the port solves, then one room step."""
    for _ in range(steps):
        inst.step(f_ext)
        inst.room.step()


def _seeded(tier, config="strike", **kw):
    """A struck plate at ``w = 2e`` (the nonlinearity live), or batch 4's rigid-body velocity kick.

    ``w = 2e`` is chosen over the plan's headline ``3e`` for the guards deliberately: the Picard
    sweep count is a strong function of amplitude (19 here, 37 at ``3e``, and the cap — i.e. NaN —
    for a narrow strike), and a guard that costs four times as much says nothing more.
    """
    inst = _make_vk(tier, **kw)
    p = inst.plate
    if config == "piston":
        inst.set_state(np.zeros(p.n_live), 1e-3 * np.ones(p.n_live))
    else:
        inst.set_state(vk_strike(p, 2.0 * p.e))
    return inst


def _refactor(inst, scale):
    """Rebuild the factorization with the load block scaled by ``scale`` (1.0 = as built)."""
    a = (inst._surface.a_bare() + scale * inst._load_scale * inst.port.load_matrix).tocsc()
    a.eliminate_zeros()
    inst._lu_loaded = splu(a)


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


# -- the reduction: the load's zero is a clean zero, nonlinear path included ----------------


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_zero_area_reduces_to_the_bare_vk_plate(boundary, tier):
    """``T = 0`` must give **bit-identical** state to a bare :class:`VKPlate`, coupling and all.

    The reduction batch 5 made for the membrane, one model on and with a second history to match:
    both cached Airy levels must come out identical too, which is what pins :meth:`commit`'s new
    roll. It is also the only test in the file that exercises the Picard loop against an
    independent implementation of itself — the model's own — so a slip in the loop hook's predictor,
    coupling factor or convergence test shows up here as an inequality rather than as a plausible
    trajectory.
    """
    inst = _make_vk(tier, boundary=boundary, N=6, sigma=3.0)
    inst.port.areas = np.zeros_like(inst.port.areas)
    inst.port.T = sparse.csr_matrix(inst.port.T.shape)
    inst.port.load_matrix = sparse.csr_matrix(inst.port.load_matrix.shape)
    a = inst._surface.a_bare().tocsc()
    a.eliminate_zeros()
    inst._lu_loaded = splu(a)

    bare = make_air_vk_plate(boundary=boundary, N=6, sigma=3.0)
    u0 = vk_strike(bare, 2.0 * bare.e)
    inst.set_state(u0)
    bare.set_state(u0)
    for _ in range(50):
        inst.step()
        inst.room.step()
        bare.step()
    assert np.array_equal(inst.plate.u, bare.u), "the zero load must be bit-identical, not close"
    assert np.array_equal(inst.plate.u_prev, bare.u_prev)
    assert np.array_equal(inst.plate.F, bare.F), "the Airy roll must reduce too"
    assert np.array_equal(inst.plate.F_prev, bare.F_prev)
    assert inst.plate.n_iters == bare.n_iters
    assert inst.radiated_energy == 0.0


# -- the money test, the channel it runs through, and the conserved total -------------------


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("wall_name", list(WALLS))
def test_ledgers_agree_and_the_channel_is_not_vacuous(boundary, tier, wall_name):
    """``radiated_energy == room.injected``, reported **with the channel size**.

    A conservation test on a channel worth 1e-14 of the total passes with the coupling
    disconnected, so the channel is asserted beside the identity. At ``w = 2e`` — the nonlinearity
    live, 19 Picard sweeps at the peak — it is 3.7% and 2.9% of ``E0`` for the supported gong
    (baffled, suspended) and 0.25% and 0.13% for the free cymbal, which is the acoustic short
    circuit doing its job on a plate whose strike is a fine spatial pattern.
    :func:`test_the_piston_is_the_free_plate_s_fat_channel` is the configuration that makes the
    free arm non-vacuous by a wide margin.
    """
    inst = _seeded(tier, boundary=boundary, walls=WALLS[wall_name])
    e0 = surface_scene_energy(inst)
    _run(inst, 300)
    gap = abs(inst.radiated_energy - inst.room.injected)
    assert gap <= LEDGER_TOL * abs(inst.radiated_energy)
    assert abs(inst.radiated_energy) > 1e-3 * e0, "the channel must be worth asserting on"


@pytest.mark.parametrize("tier", TIERS)
def test_the_piston_is_the_free_plate_s_fat_channel(tier):
    """A free plate's rigid-body **velocity** piston, inherited from batch 4 and still the fat one.

    Measured 27.6% of ``E0`` baffled and 4.6% suspended, against 0.25%/0.13% for a strike — the
    contrast batch 4 named (a piston in a baffle is the most efficient radiator the geometry has; a
    two-sided plate at low ``ka`` short-circuits twice over). Note what it is *not*: a rigid
    translation carries no stretching, so ``l(w, w) = 0`` and the von Kármán coupling is asleep
    here. That is the point of running both configurations — this one asserts the ledger on a fat
    channel, and the strike above asserts it with the nonlinearity awake.
    """
    inst = _seeded(tier, config="piston", boundary="free", walls=WALLS["all-lossy"])
    e0 = surface_scene_energy(inst)
    _run(inst, 300)
    gap = abs(inst.radiated_energy - inst.room.injected)
    assert abs(inst.radiated_energy) > 0.04 * e0
    assert gap <= LEDGER_TOL * abs(inst.radiated_energy)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_scene_total_is_flat(boundary, tier):
    """Lossless plate, rigid room, nonlinearity live: the scene total is flat to 1.7e-13 of ``E0``.

    Necessary and not sufficient for the fourth batch running — and here the *reason* it is not
    sufficient has a second half that no previous batch had. Model #6 conserves only at the Picard
    fixed point, so this number is bounded below by ``couple_tol`` and not by the air load at all;
    :func:`test_couple_tol_moves_the_total_and_not_the_money_test` is what separates the two.
    """
    inst = _seeded(tier, boundary=boundary)
    e0 = surface_scene_energy(inst)
    worst = 0.0
    for _ in range(300):
        inst.step()
        inst.room.step()
        assert inst.converged, f"the Picard loop hit the cap at step {inst.n}"
        worst = max(worst, abs(surface_scene_energy(inst) - e0))
    assert worst <= DRIFT_TOL * abs(e0)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_lossy_scene_total_is_monotone(boundary, tier):
    """Lossy plate **and** lossy room: the scene total never rises. Measured rise exactly 0.0.

    Passivity of the whole scene, which is the property the load's placement inside ``A`` buys:
    the load is proportional to ``w^{n+1} - w^{n-1}``, so it is dissipative by construction rather
    than by an inequality that has to be checked each step.
    """
    inst = _seeded(tier, boundary=boundary, walls=WALLS["all-lossy"], sigma=1.0)
    prev = surface_scene_energy(inst)
    e0 = prev
    rise = 0.0
    for _ in range(300):
        inst.step()
        inst.room.step()
        cur = surface_scene_energy(inst)
        rise = max(rise, cur - prev)
        prev = cur
    assert rise <= 1e-14 * abs(e0), f"the scene total rose by {rise:.3e}"


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_energy_is_an_override_and_not_a_delegation(boundary, tier):
    """``inst.energy()`` must be the plate's **plus** the coupling channel, not ``__getattr__``'s.

    The failure this guards is silent and has caught batches 2–5: delegation hands back the bare
    resonator's energy, i.e. the total without its coupling channel — the number that looks fine
    and is not conserved.
    """
    inst = _seeded(tier, boundary=boundary, walls=WALLS["all-lossy"])
    _run(inst, 100)
    assert inst.radiated_energy != 0.0
    assert inst.energy() == inst.plate.energy() + inst.radiated_energy
    assert inst.energy() != inst.plate.energy()


# -- the primary guard: the achieved state back in the COUPLED PDE ---------------------------


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("fs", [8000.0, 11000.0])
def test_the_coupled_residual_at_two_timesteps(boundary, tier, fs):
    """Batch 4's guard with the von Kármán term added — and it is now the residual of *two* fields.

    The force is rebuilt from the **room's own post-closure pressure** (a number the port never
    touched) and the coupling from the **committed** ``(w^{n+1}, F^{n+1})``. Both halves of that
    sentence are load-bearing:

    * built from the last sweep's cached coupling instead, this reports the Picard *increment* and
      then passes or fails for a reason that has nothing to do with the air load (plan §6.2);
    * ``F^{n-1}`` must be captured **before** the step, because :meth:`_VKPlateSurface.commit`
      rolls it away — the ``mu``-average the scheme uses is ``(F^{n+1} + F^{n-1})/2``, not
      ``(F^{n+1} + F^n)/2``.

    Run at two timesteps because a wrong-but-consistent ``k``-dependent factor passes at one, with
    ``sigma > 0``, a non-zero ``f_ext`` and a lossy wall so nothing is invisible. Measured 1.4e-14
    …3.5e-14 correct, against 8.6e-2 with the coupling term dropped, 4.3e-2 with it halved and
    1.2e-2 with the air load halved — so the residual sees the nonlinear force and the air load
    separately, which no ledger in this file does.
    """
    def residual(control):
        inst = _make_vk(tier, boundary=boundary, walls=WALLS["one-lossy-wall"], N=6, sigma=2.0,
                        fs=fs)
        if control == "half-load":
            _refactor(inst, 0.5)
        plate, room, port = inst.plate, inst.room, inst.port
        inst.set_state(vk_strike(plate, 2.0 * plate.e))
        _run(inst, 5)

        f_ext = 1e-3 * np.random.default_rng(0).standard_normal(plate.n_live)
        u_n, u_nm1 = plate.u.copy(), plate.u_prev.copy()
        f_nm1 = plate.F_prev.copy()   # F^{n-1}: rolled away by commit(), so take it now
        p_old = room.p.copy()
        inst.step(f_ext)
        room.step()
        u_np1 = plate.u.copy()
        f_np1 = plate.F.copy()        # F^{n+1}, from the COMMITTED state -- never a cached sweep
        pbar = 0.5 * (room.p + p_old)
        load = (
            pbar[port.nodes_hi] - pbar[port.nodes_lo] if tier == "suspended" else pbar[port.nodes]
        )

        k, theta = plate.k, plate.theta
        average = theta * u_np1 + (1.0 - 2.0 * theta) * u_n + theta * u_nm1
        velocity = (u_np1 - u_nm1) / (2.0 * k)
        f_total = f_ext - (port.T.T @ load)
        accel = u_np1 - 2.0 * u_n + u_nm1
        w_avg = 0.5 * (plate._to_full(u_np1) + plate._to_full(u_nm1))
        f_avg = 0.5 * (f_np1 + f_nm1)
        coupling = plate._to_live(plate.bracket(w_avg, f_avg))
        if boundary == "supported":
            mass, stiffness, weight = plate.rho_s * plate.h * plate.h, plate.B, 1.0
            couple = k * k * coupling / plate.rho_s
        else:
            mass, stiffness, weight = plate.rho_s, plate.K, plate.wdiag
            couple = k * k * plate.h * plate.h * coupling / plate.rho_s
        scale = {"half-coupling": 0.5, "no-coupling": 0.0}.get(control, 1.0)
        res = (
            weight * accel
            + k * k * plate.kappa**2 * (stiffness @ average)
            + 2.0 * plate.sigma * k * k * weight * velocity
            - k * k * f_total / mass
            - scale * couple
        )
        return float(np.max(np.abs(res)) / np.max(np.abs(weight * accel)))

    assert residual("correct") <= 1e-11
    assert residual("no-coupling") > 1e-6, "the residual must see the von Karman force"
    assert residual("half-coupling") > 1e-6, "... and see its magnitude"
    assert residual("half-load") > 1e-6, "... and see the air load's, separately"


# -- the two-parameter money test: a second error axis, and which detector sees it -----------


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_couple_tol_moves_the_total_and_not_the_money_test(boundary, tier):
    """The batch's detector finding, measured on all three at once rather than predicted on two.

    Model #6 conserves only at the Picard fixed point, so ``couple_tol`` is an error source
    *alongside* the air load — an axis no previous batch had, and the three detectors split along
    it cleanly:

    ==========================  =============  =============  =============
    ``couple_tol``              ``1e-13``      ``1e-6``       ``1e-3``
    ==========================  =============  =============  =============
    scene-total drift / ``E0``  1.2e-13        5.1e-7         1.0e-3
    ``|radiated - injected|``   2.2e-15        8.7e-16        1.6e-15
    ``last_residual``           9.9e-14        9.9e-7         1.0e-3
    Picard sweeps               19             9              4
    ==========================  =============  =============  =============

    **The money test is blind, and for a third distinct reason.** Batch 4's blind spot was a ``2``
    inside the factorization; batch 5's was which velocity produced the ``q``; here it is that
    ``radiated == injected`` is arithmetic on whatever ``w^{n+1}`` came out of the solve —
    ``q = T(w^{n+1} - w^{n-1})/2k``, ``pbar = pbar_free + Rq``, inject ``q`` — so an
    under-converged ``w^{n+1}`` is ported *self-consistently* and the identity holds to rounding
    while the physics is wrong by a part in a thousand.

    **And the plan's more interesting outcome did not happen, which is worth recording.** It
    predicted the scene total might also be nearly blind, because ``VKPlate.energy()`` is built
    from ``(u, u_prev, F, F_prev)`` and the roll keeps those mutually consistent regardless of
    convergence. Measured, the total tracks ``couple_tol`` almost exactly — because the committed
    ``F^{n+1}`` is the Airy solve of the *previous* iterate while ``w^{n+1}`` is the current one,
    and the gap between them is precisely the increment the tolerance bounds.

    The self-certifying half, and the only part of this that is about the air at all:
    **the loaded drift falls with ``couple_tol`` at the same rate as the unloaded**, within 1.2x at
    every tolerance. The air load adds no new error floor.
    """
    drifts = []
    for tol in (1e-13, 1e-6, 1e-3):
        inst = _seeded(tier, boundary=boundary, couple_tol=tol)
        bare = make_air_vk_plate(boundary=boundary, couple_tol=tol)
        bare.set_state(vk_strike(bare, 2.0 * bare.e))
        e0 = surface_scene_energy(inst)
        b0 = bare.energy()
        worst = bare_worst = 0.0
        for _ in range(200):
            inst.step()
            inst.room.step()
            bare.step()
            worst = max(worst, abs(surface_scene_energy(inst) - e0))
            bare_worst = max(bare_worst, abs(bare.energy() - b0))
        gap = abs(inst.radiated_energy - inst.room.injected) / abs(inst.radiated_energy)
        assert gap <= LEDGER_TOL, f"the money test is supposed to MISS couple_tol = {tol:.0e}"
        assert worst <= 3.0 * bare_worst, "the air load must add no error floor of its own"
        assert inst.last_residual <= tol
        drifts.append(worst / abs(e0))

    tight, middle, loose = drifts
    assert tight <= DRIFT_TOL, f"a converged run must still be flat, got {tight:.2e}"
    assert middle > 1e-9, "... and the conserved total is supposed to CATCH an under-converged one"
    assert loose > 1e-5 and loose > middle > tight


# -- refusals -------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_refuses_a_sample_rate_mismatch(tier):
    """The port's Thevenin solve is one timestep shared by both — a refusal, not a resampling."""
    room = make_surface_room()
    plate = make_air_vk_plate(fs=2.0 * room.fs)
    with pytest.raises(ValueError, match="sample-rate mismatch"):
        if tier == "baffled":
            RoomLoadedVKPlate(plate=plate, room=room, face="z0")
        else:
            RoomSuspendedVKPlate(plate=plate, room=room, plane="z", index=AIRBOX_DIPOLE_INDEX)
