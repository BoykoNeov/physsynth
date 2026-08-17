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
    make_vk_room_bare_twin,
    make_vk_room_chain,
    step_vk_room_chain,
    surface_scene_energy,
    vk_linear_twin,
    vk_room_pluck,
    vk_room_rigid_share,
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
from physsynth.core.connection import StringPlateBridge
from physsynth.core.string_ideal import IdealString

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
        # Load-bearing at 1e-6 and 1e-3 only: at 1e-13 both sides sit at the rounding floor
        # (~1.2e-13 each), so the comparison passes there for reasons unrelated to the air load.
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

    Measured on this rig (plate's first free elastic mode ~36 Hz), 400 steps, rigid share
    baffled / suspended::

        L (m)   f1 (Hz)   rigid share
        0.6     91.3      4.3% / 4.2%
        1.2     45.6      20.8% / 19.5%
        2.4     22.8      77.9% / 72.0%
        4.8     11.4      95.4% / 95.0%

    and it is a **cross-rig** reproduction: the plan's own 57.9 kHz rig, a 100 mm plate and a
    different string give 95.5% at ``f1/f_elastic = 0.28`` against 4.5% at 1.00. Two rigs seven
    times apart in sample rate agree, which is what makes this a mechanism rather than a tuning.

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
