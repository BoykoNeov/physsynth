"""Validation for the mallet -> plate collision (model #7p) — the strike on an *implicit* body.

Model #7 struck a membrane, and the membrane made the coupling easy: it is explicit, so a nodal
force reaches only its own node by the next step and the driving-point admittance is the bare local
nodal mass. The plate solves ``A u^{n+1} = rhs``, so a force at one node reaches *every* node and
there is no local admittance to read. What stands in for it is a **precomputed influence column**,
``(k^2 / force_den) A^-1 e_node``, whose entry at the strike node is the admittance the scalar
contact equation closes on and whose whole length is how the force is spread back into the field.

That column is the only genuinely new arithmetic in this model — the drumhead's replacement, the
contact root-find, the discrete-gradient force and its removable ``0/0`` are all model #7's, already
proven. So the first test here is the **superposition identity** the column stands on, asserted at
every node rather than only at the struck one: a wrong column elsewhere would still satisfy a
drive-point check, still conserve energy (it would simply be a different self-consistent
trajectory), and still sound wrong.

Two results the batch did not go looking for and that a later reader needs:

* **The free branch's energy read-out is not the scheme's.** A point strike feeds the free plate's
  ``{1, x, y}`` rigid nullspace, the plate translates for ever, and the potential form
  ``kappa^2 (K u).u`` — in which the rigid part cancels *mathematically* — loses precision to that
  cancellation at a rate exactly quadratic in the rigid displacement. It is the **plate's**, not the
  mallet's: a bare free plate handed a uniform velocity does the same thing with no mallet in the
  room, which is what ``test_the_free_readout_error_is_quadratic_in_the_rigid_drift`` asserts. The
  supported branch, which has no rigid mode, conserves to ~3e-13 on the identical coupling.
* **Superposition is exact in exact arithmetic and tight-but-not-bitwise in this one**, because a
  sparse LU back-substitution is not a linear map over doubles. The one place it *is* exact is a
  miss, where the force is a hard zero.
"""

import numpy as np
import pytest
from helpers import make_free_plate, make_mallet_plate, make_plate, plate_bump

from physsynth.core.mallet import MalletPlate
from physsynth.core.plate import Plate, VKPlate

CONSERVE_TOL = 1e-10  # lossless, elastic, supported: relative drift of H; observed ~3e-13

# The free branch's bar, and it is a property of the READ-OUT rather than of the scheme -- see the
# module docstring and the quadratic-law test below. Measured 9.8e-10 over this window on the
# shipped rig; the factor of ten is headroom for a different BLAS, not slack for a leak.
#
# It is a claim about THIS RIG and deliberately not a general one, because the quantity it bounds is
# the rigid-to-elastic displacement ratio: anything that shrinks the elastic response at fixed
# strike energy makes it worse, and both a finer grid and a stiffer plate do (measured 1.2e-08 at
# `mu = 4, N = 48`, and 4.5e-08 at `kappa = 160`). That is why the assertion below is paired with a
# supported control on the *same* mallet -- the ratio between the branches is the transferable
# claim, and the constant is only this rig's value of it.
FREE_READOUT_TOL = 1e-8
FREE_STEPS = 2000


def _run(mal, steps):
    """Step ``mal`` for ``steps``, returning the per-step total-energy array."""
    e = np.empty(steps + 1)
    e[0] = mal.energy()
    for i in range(1, steps + 1):
        mal.step()
        e[i] = mal.energy()
    return e


def _drift(e):
    return float(np.max(np.abs(e - e[0])) / abs(e[0]))


# -- Criterion 0: the influence column, which is the only new arithmetic in the model -------------


@pytest.mark.parametrize("boundary", ["supported", "free"])
def test_the_influence_column_reproduces_a_forced_step_at_every_node(boundary):
    # Stepping force-free and then adding `influence * f` must be the same plate as stepping with
    # `f` in the right-hand side: `A (u_free + d) = rhs_0 + A d = rhs_0 + (k^2 f / den) e_node`.
    #
    # Asserted over the WHOLE field. A column right at the strike node and wrong elsewhere passes
    # every other test in this file, so this is the one that has to be exhaustive.
    mal = make_mallet_plate(boundary=boundary, N=16)
    node, influence = mal.node, mal._influence
    force = 7.5  # N -- an arbitrary nonzero

    # A generic non-rest state: a bump, advanced far enough that `u` and `u_prev` really differ.
    forced = make_mallet_plate(boundary=boundary, N=16).plate
    forced.set_state(plate_bump(forced))
    for _ in range(5):
        forced.step()
    split_u, split_prev = forced.u.copy(), forced.u_prev.copy()

    f_ext = np.zeros(forced.n_live)
    f_ext[node] = -force
    forced.step(f_ext=f_ext)

    split = make_mallet_plate(boundary=boundary, N=16).plate
    split.u, split.u_prev = split_u, split_prev
    split.step()
    corrected = split.u - influence * force

    scale = float(np.max(np.abs(forced.u)))
    assert scale > 0.0, "the reference field is identically zero"
    # Not `array_equal`: `solve(rhs) + c*solve(e)` and `solve(rhs + c*e)` round differently.
    worst = float(np.max(np.abs(forced.u - corrected))) / scale
    assert worst < 1e-14, f"{boundary}: the column disagrees with a forced step by {worst:.3e}"


@pytest.mark.parametrize("boundary", ["supported", "free"])
def test_one_newton_moves_the_strike_node_by_the_admittance(boundary):
    # `_g_s` is READ from the column rather than computed a second way, so its identity with the
    # column's own entry is structural and can be exact. What is measured is the physics: from
    # rest, a unit force at the strike node must move that node by `g_s` metres, one step later.
    mal = make_mallet_plate(boundary=boundary, N=16)
    assert mal._g_s == mal._influence[mal.node]
    assert mal._g_s > 0.0, "an SPD plate cannot have a negative driving-point admittance"
    assert mal._g == mal._g_s + mal._g_h

    plate = make_mallet_plate(boundary=boundary, N=16).plate
    f_ext = np.zeros(plate.n_live)
    f_ext[mal.node] = 1.0
    plate.step(f_ext=f_ext)
    assert plate.u[mal.node] == pytest.approx(mal._g_s, rel=1e-14)


def test_the_strike_snaps_to_a_node_and_says_where():
    # `pickup_index_at` counts LIVE nodes; `X`/`Y` are full-grid arrays. A confusion between the
    # two indexings reports a plausible-looking strike point and builds a wrong column, so the
    # reported point is checked against the plate's own answer for that live index.
    mal = make_mallet_plate(N=24, strike_x=0.3, strike_y=0.4)
    plate = mal.plate
    assert mal.node == plate.pickup_index_at(mal.x_strike, mal.y_strike)
    live_x, live_y = plate.X[plate.mask], plate.Y[plate.mask]
    assert mal.x_strike == live_x[mal.node]
    assert mal.y_strike == live_y[mal.node]
    h = plate.h
    assert abs(mal.x_strike - 0.3) <= h and abs(mal.y_strike - 0.4) <= h


def test_the_struck_field_is_oriented_the_way_the_coordinates_are():
    # `state` is the full 2-D field, and this project has been bitten before by reading such a
    # field with its axes swapped -- a transposed decode renders as a plausible picture (viewer
    # batch 18). `Plate.X` varies along axis 1 and `Y` along axis 0, so the strike must show up at
    # `state[row, col]` with `X[row, col] == x_strike`. A square plate cannot tell the difference
    # any other way.
    mal = make_mallet_plate(N=24, strike_x=0.3, strike_y=0.4)
    plate = mal.plate
    for _ in range(12):
        mal.step()
    row, col = np.unravel_index(int(np.argmax(np.abs(mal.state))), mal.state.shape)
    assert plate.X[row, col] == mal.x_strike, "the field's x axis is not X's"
    assert plate.Y[row, col] == mal.y_strike, "the field's y axis is not Y's"


# -- Criterion 1 (money test): lossless conservation on the supported branch ----------------------


@pytest.mark.parametrize("K,mass,v0,alpha", [
    (5.0e4, 0.02, 3.0, 2.3), (2.0e4, 0.05, 3.0, 1.0),
    (3.0e4, 0.03, 4.0, 3.0), (5.0e4, 0.02, 3.0, 1.5),
])
def test_supported_lossless_energy_conserved(K, mass, v0, alpha):
    # E_plate + mallet KE + averaged felt PE, flat to machine precision. Varying alpha closes the
    # same "force x coupling" gap the membrane model's parametrisation closes.
    mal = make_mallet_plate(K=K, mass=mass, alpha=alpha, strike_velocity=v0)
    drift = _drift(_run(mal, 4000))
    assert drift < CONSERVE_TOL, f"energy drift {drift:.2e} (K={K}, M={mass}, v0={v0}, a={alpha})"


def test_the_bar_is_not_fitted_to_one_rig():
    """Conservation at 16x the conditioning and 4x the node count, same felt, same sample rate.

    ``mu`` is the plate's own Courant number, and it sets the conditioning of ``A`` on its own:
    ``cond(A) ~ 1 + 64 theta mu^2``, independent of ``N``. Holding ``fs`` fixed needs ``N^2 ~ mu``,
    so ``mu = 4, N = 48`` is the shipped rig's sample rate and felt resolution (11520 Hz, 22.9
    steps) with sixteen times the conditioning and four times the nodes. Both bars survive it.

    Measured separately, and worth recording because the two effects look alike in one sweep:
    conditioning does **nothing** to this bar (holding ``N`` and scaling ``kappa`` with ``mu`` to
    keep ``fs``, the drift is 1.6e-13 to 2.3e-12 across a 256x range, non-monotone), while the
    **node count** does — 1.9e-13 at 256 live nodes to 1.7e-11 at 4489, because ``energy()`` is a
    reduction and a longer sum accumulates more rounding. Still six times inside the bar at the
    largest rig tried.
    """
    mal = make_mallet_plate(N=48, mu=4.0)
    assert mal.plate.n_live == 47 * 47
    assert mal.steps_per_contact >= 8.0
    drift = _drift(_run(mal, 2000))
    assert drift < CONSERVE_TOL, f"drift {drift:.2e} at 16x conditioning, 4x the nodes"


def test_strike_actually_couples():
    # Without this, conservation is satisfied by a mallet that sailed past: it conserves perfectly
    # because nothing happened. The plate must take a real share of the strike.
    mal = make_mallet_plate()
    share = 0.0
    for _ in range(4000):
        mal.step()
        share = max(share, mal.plate.energy() / mal.energy())
    assert share > 0.3, f"the plate took only {share:.2f} of the energy -- coupling too weak"


def test_the_felt_is_resolved_on_the_shipped_defaults():
    # The plate's sample rate comes out of its own Courant number (`fs = kappa / (mu h^2)`), not out
    # of anything the felt cares about -- so "is the contact resolved" is a question about the
    # helper's defaults and has to be asked of them directly rather than assumed.
    mal = make_mallet_plate()
    assert mal.steps_per_contact >= 8.0, (
        f"only {mal.steps_per_contact:.1f} steps through the felt half-period; the model warns "
        "below 8 and the conservation tests would be measuring aliasing"
    )


# -- Criterion 2: conservation is discrete-gradient-limited (drift proportional to solve tol) -----


def test_drift_scales_with_newton_tolerance():
    # Tightening the scalar contact solve tightens conservation, because the applied force
    # approaches the exact discrete gradient. This is what separates "the scheme conserves" from
    # "the solver happened to be good enough".
    drifts = []
    for tol in (1e-14, 1e-10, 1e-6):
        plate = make_plate(N=24, mu=1.0)
        mal = MalletPlate(
            plate=plate, mass=0.02, stiffness=5.0e4, alpha=2.3, hysteresis=0.0,
            strike_x=0.3, strike_y=0.4, strike_velocity=3.0, newton_tol=tol,
        )
        drifts.append(_drift(_run(mal, 2500)))
    assert drifts[0] < CONSERVE_TOL, f"tight-tol drift {drifts[0]:.2e} is not machine precision"
    assert drifts[-1] > drifts[0] * 100, (
        f"loosening the solve did not increase drift ({drifts}); conservation is not solve-limited"
    )


# -- Criterion 3: passivity survives the coupling --------------------------------------------------


@pytest.mark.parametrize("sigma,lam_h", [(2.0, 0.0), (0.0, 5.0e3), (1.0, 2.0e3)])
def test_loss_only_removes_energy(sigma, lam_h):
    mal = make_mallet_plate(sigma=sigma, hysteresis=lam_h)
    e = _run(mal, 3000)
    assert np.all(np.isfinite(e)), "non-finite energy"
    assert np.max(np.diff(e)) <= 1e-9 * e[0], f"loss added energy (sigma={sigma}, lam_h={lam_h})"
    assert e[-1] < 0.99 * e[0], "the run lost nothing measurable -- the bar is testing itself"


# -- Criterion 4: a mallet that never touches leaves the plate bit-for-bit unchanged ---------------


def test_missing_mallet_is_bit_identical_to_bare_plate():
    # The `K = 0` analog, and the one case where superposition is EXACT rather than tight: with
    # `f == 0.0` every increment is a signed zero and `x - (+-0.0) == x` for finite `x`. It is also
    # the guard that the whole-field injection never touches a node it should not.
    steps = 400
    bare = make_plate(N=24, mu=1.0, sigma=0.3)
    seed = plate_bump(bare)
    bare.set_state(seed)

    plate = make_plate(N=24, mu=1.0, sigma=0.3)
    plate.set_state(seed.copy())
    # A huge gap with the mallet moving away: it never reaches the plate inside the window.
    mal = MalletPlate(
        plate=plate, mass=0.02, stiffness=5.0e4, alpha=2.3,
        strike_x=0.3, strike_y=0.4, strike_velocity=-1.0, gap=10.0,
    )
    for _ in range(steps):
        bare.step()
        mal.step()
    assert mal.contact_force == 0.0 and not mal.in_contact
    np.testing.assert_array_equal(mal.plate.u, bare.u)
    np.testing.assert_array_equal(mal.plate._accel, bare._accel)


# -- The free branch, and whose drift it is -------------------------------------------------------


def test_the_free_readout_error_is_quadratic_in_the_rigid_drift():
    # NO MALLET IN THIS TEST. A free plate handed a uniform velocity carries net momentum and
    # translates for ever; the potential form `kappa^2 (K u).u` annihilates that rigid part
    # mathematically and cancels it only to `eps` numerically, leaving an absolute error that goes
    # like the SQUARE of the rigid displacement -- because the potential is a quadratic form.
    #
    # This is the attribution. What the struck free plate does below is what the plate does, and
    # what it does here it does with nothing striking it.
    errors, drifts = [], []
    for v_rigid in (0.0, 0.3, 1.0, 3.0):
        plate = make_free_plate(N=24, mu=1.0)
        bump = plate_bump(plate)
        plate.set_state(bump, np.full(plate.n_live, v_rigid))
        e0 = plate.energy()
        worst = 0.0
        for _ in range(2000):
            plate.step()
            worst = max(worst, abs(plate.energy() - e0))
        errors.append(worst)
        drifts.append(abs(float(np.dot(plate.w, plate.u) / np.sum(plate.w))))

    # With no net momentum the read-out is exact to machine precision -- so this is not a defect of
    # the free plate's energy, it is a defect of reading it through a large rigid displacement.
    assert errors[0] < 1e-14, f"a free plate at rest in the mean drifts by {errors[0]:.2e} J"
    # And with net momentum, error / drift^2 is a constant across a tenfold range of drift.
    ratios = [e / d**2 for e, d in zip(errors[1:], drifts[1:], strict=True)]
    assert max(ratios) / min(ratios) < 1.3, (
        f"the read-out error is not quadratic in the rigid drift: error/drift^2 = {ratios}"
    )


def test_free_strike_conserves_within_the_readouts_reach():
    # The struck cymbal. The bar is `FREE_READOUT_TOL` rather than `CONSERVE_TOL`, and the test
    # above says why: the plate translates, and the energy read-out -- not the scheme -- pays for
    # it quadratically. The supported plate, which has no rigid mode to translate along, is run
    # here on the identical mallet as the control, and it must be orders better.
    free = _drift(_run(make_mallet_plate(boundary="free"), FREE_STEPS))
    supported = _drift(_run(make_mallet_plate(boundary="supported"), FREE_STEPS))
    assert free < FREE_READOUT_TOL, f"free-plate strike drifted {free:.2e}"
    assert supported < CONSERVE_TOL, f"supported control drifted {supported:.2e}"
    assert supported < 1e-2 * free, (
        f"the two branches drift alike ({supported:.2e} vs {free:.2e}) -- then the free branch's "
        "excess is not the rigid mode and this file's explanation is wrong"
    )


def test_a_struck_free_plate_recoils():
    # Physics, asserted so a reader watching a struck cymbal drift off screen does not file it as a
    # bug. `K 1 = 0`, so projecting the step onto `1^T W` kills the stiffness term and leaves
    # `m^{n+1} = 2 m^n - m^{n-1}`: after the contact ends the mass-weighted mean is EXACTLY linear
    # in the step index.
    mal = make_mallet_plate(boundary="free")
    plate = mal.plate
    weight = plate.w / np.sum(plate.w)
    marks = []
    for step in range(1, 2001):
        mal.step()
        if step in (1000, 1500, 2000):
            marks.append(float(np.dot(weight, plate.u)))
    first, second = marks[1] - marks[0], marks[2] - marks[1]
    assert abs(first) > 1e-6, f"the plate took no net momentum: mean moved {first:.2e} m"
    assert abs(second - first) <= 1e-9 * abs(first), (
        f"the drift is not a constant velocity: {first:.9e} then {second:.9e}"
    )


@pytest.mark.parametrize("domain", ["circle", "guitar"])
def test_a_curved_free_plate_is_struck_too(domain):
    # The outline is orthogonal to everything this model does: the influence column is built from
    # whatever `A` the plate factored, and a staircased rim only changes which nodes are live.
    # Curved outlines are offered on the free branch only, so this inherits the read-out bar.
    mal = make_mallet_plate(boundary="free", domain=domain, N=20)
    drift = _drift(_run(mal, 1500))
    assert drift < FREE_READOUT_TOL, f"{domain} plate strike drifted {drift:.2e}"
    assert mal.plate.n_live < 21 * 21, "a curved outline should have pruned some nodes"


# -- What the model refuses, and what it is -------------------------------------------------------


def test_a_nonlinear_plate_is_refused_with_the_reason():
    # Not a missing cast. The von Karman step is nonlinear, so it is not affine in `f_ext` and the
    # influence column -- the whole basis of the scalar collapse -- does not exist. The message has
    # to say that, because the obvious reading of the refusal is that someone forgot a branch.
    vk = VKPlate(Lx=0.4, Ly=0.4, E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0, fs=48000.0, N=12)
    with pytest.raises(TypeError, match="nonlinear"):
        MalletPlate(
            plate=vk, mass=0.02, stiffness=5.0e4, strike_x=0.3, strike_y=0.4, strike_velocity=3.0,
        )


@pytest.mark.parametrize("kwargs,message", [
    (dict(mass=0.0), "mass must be > 0"),
    (dict(stiffness=0.0), "stiffness K must be > 0"),
    (dict(alpha=0.5), "alpha must be >= 1"),
    (dict(hysteresis=-1.0), "lambda_h must be >= 0"),
    (dict(gap=-1e-3), "gap must be >= 0"),
])
def test_the_scalar_checks_are_the_membrane_models(kwargs, message):
    base = dict(
        plate=make_plate(N=12, mu=1.0), mass=0.02, stiffness=5.0e4, alpha=2.3,
        strike_x=0.3, strike_y=0.4, strike_velocity=3.0,
    )
    with pytest.raises(ValueError, match=message):
        MalletPlate(**{**base, **kwargs})


def test_mallet_plate_is_a_resonator():
    # Duck-types the engine's Resonator protocol, as the membrane model does.
    # `mu = 0.5` rather than the default 1.0: a coarse plate at the default would drop the
    # sample rate below the felt's resolution and warn, and a warning nobody needs is noise.
    mal = make_mallet_plate(N=12, mu=0.5)
    assert isinstance(mal, MalletPlate)
    assert isinstance(mal.plate, Plate)
    assert hasattr(mal, "k") and callable(mal.step) and callable(mal.energy)
    assert mal.state.shape == mal.plate.mask.shape
    assert isinstance(mal.displacement_at(0), float)
    assert isinstance(mal.pressure(), float)


def test_aliasing_the_two_state_buffers_is_survivable():
    # `u` and `_accel` both have public setters that ADOPT the array they are handed, so a caller
    # can make them the same object. The injection writes both, and two simultaneous mutable
    # borrows of one NumPy buffer would be a panic across the FFI boundary rather than an
    # exception -- which is why the binding takes them one after the other.
    #
    # On THIS path the hazard turns out not to arise anyway, and the reason is worth recording
    # rather than assuming: `Plate.step()` rebinds both buffers to fresh arrays before the mallet
    # injects, so whatever the caller aliased is already gone. That makes the sequential borrow a
    # property of the binding rather than of this model, and it is asserted here because the next
    # caller of those two methods may not step first.
    plate = make_plate(N=12, mu=0.5)
    mal = MalletPlate(
        plate=plate, mass=0.02, stiffness=5.0e4, strike_x=0.3, strike_y=0.4, strike_velocity=3.0,
    )
    shared = np.zeros(plate.n_live)
    plate.u = shared
    plate._accel = shared
    assert plate.u is plate._accel
    mal.step()  # must not raise, and in particular must not raise pyo3_runtime.PanicException
    assert plate.u is not plate._accel, "step() is supposed to rebind both buffers"
    assert np.any(plate.u != 0.0)


def test_the_plate_handed_in_is_the_plate_held():
    # The binding holds the caller's object, not a copy -- every caller reads the struck field back
    # through the handle it passed.
    plate = make_plate(N=12, mu=0.5)
    mal = MalletPlate(
        plate=plate, mass=0.02, stiffness=5.0e4, strike_x=0.3, strike_y=0.4, strike_velocity=3.0,
    )
    assert mal.plate is plate
    mal.step()
    assert plate.n == 1 and np.any(plate.u != 0.0)
