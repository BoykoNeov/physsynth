"""Mallet on a gong — model #7g, the nested solve (``docs/dev/mallet-gong-plan.md``).

An outer iteration on the contact force wrapped around a full von Kármán plate solve per
evaluation. The algorithm, and everything that needs the plate's *pre-step* buffers in hand, is
asserted natively in ``crates/physsynth-core/tests/mallet_gong.rs``; this file is what only makes
sense from Python — the binding's surface and its two refusals, the reduction to model #7p through
the shipped classes, and the physics payoff.

**The physics payoff is one measurement.** Model #7p found that on a *linear* plate "the felt
exponent is the only source of dynamic timbre": at ``alpha = 1`` the mallet is a mass, the contact
is a linear spring, the one-sided switching is scale-invariant, and the response to a strike is
therefore exactly proportional to how hard you hit it. That is still true here — measured at
**exactly 0.0**, not merely small — and it is what makes the gong's number attributable: with the
felt contributing nothing by construction, the departure from a scaled response is the *plate's*,
and it is 2.1 times the loud strike's own peak.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import GONG_MATERIAL, gong_linear_twin, make_mallet_gong

from physsynth.analysis.spectrum import magnitude_spectrum
from physsynth.core.mallet import MalletPlate, MalletVKPlate
from physsynth.core.plate import Plate, VKPlate

THICKNESS = GONG_MATERIAL["e"]


def ring(gong, steps, pickup=None):
    """Step ``gong`` and return ``(energy range / |E0|, peak |w| / e, membrane share, signal)``."""
    e0 = gong.energy()
    lo = hi = e0
    peak = share = 0.0
    out = np.empty(steps)
    for i in range(steps):
        gong.step()
        e = gong.energy()
        lo, hi = min(lo, e), max(hi, e)
        peak = max(peak, np.abs(gong.plate.u).max())
        share = max(share, gong.plate.membrane_energy() / abs(e0))
        out[i] = gong.displacement_at(pickup) if pickup is not None else 0.0
    return (hi - lo) / abs(e0), peak / THICKNESS, share, out


# -- the surface -----------------------------------------------------------------------------------


def test_the_gong_holds_the_plate_object_it_was_handed():
    """``mal.plate`` **is** the object passed in, not a copy — every other coupled mallet's rule.

    The whole of the viewer and half of the test suite read the field, the mask and the energy back
    through this handle, so a copy would be a silently frozen animation.

    Asserted as a **read-back through the caller's own reference**, not as ``shares_memory``. A
    ``shares_memory`` check here would pass for the wrong reason and keep passing after the
    property broke: ``commit`` allocates a *fresh* array and rebinds, exactly as ``step()`` does, so
    the two ``.u`` accesses agree because both getters return the same stored handle at that
    moment — not because any buffer survived the step. A mallet that had quietly taken its own copy
    of the plate would still satisfy it.
    """
    plate = VKPlate(Lx=0.4, Ly=0.4, fs=48000.0, N=20, **GONG_MATERIAL)
    mal = MalletVKPlate(
        plate=plate, mass=0.05, stiffness=5.0e4, strike_x=0.12, strike_y=0.16,
        strike_velocity=6.0,
    )
    assert mal.plate is plate
    for _ in range(40):
        mal.step()
    assert plate.n == 40, "the caller's own reference did not see the steps"
    # The strike node, read two ways: off the caller's array, and through the mallet.
    assert plate.u[mal.node] == mal.displacement_at(mal.node)
    assert plate.u[mal.node] != 0.0, "a struck node that never moved proves nothing"


def test_the_gong_refuses_a_linear_plate_and_points_at_the_model_that_takes_one():
    """And the refusal has to say *why*, because the obvious reading is a missing branch.

    The mirror of ``MalletPlate``'s own refusal of a ``VKPlate``. Between them the two messages
    make the choice of model a property of the collaborator rather than of the caller's memory.
    """
    linear = Plate(Lx=0.4, Ly=0.4, kappa=1.5, rho=7.8, fs=48000.0, N=20)
    with pytest.raises(TypeError) as excinfo:
        MalletVKPlate(
            plate=linear, mass=0.05, stiffness=5.0e4, strike_x=0.12, strike_y=0.16,
            strike_velocity=6.0,
        )
    assert "MalletPlate" in str(excinfo.value)
    assert "influence column" in str(excinfo.value)

    nonlinear = VKPlate(Lx=0.4, Ly=0.4, fs=48000.0, N=20, **GONG_MATERIAL)
    with pytest.raises(TypeError) as excinfo:
        MalletPlate(
            plate=nonlinear, mass=0.05, stiffness=5.0e4, strike_x=0.12, strike_y=0.16,
            strike_velocity=6.0,
        )
    assert "not affine" in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(mass=0.0), "mass"),
        (dict(K=0.0), "stiffness"),
        (dict(alpha=0.5), "alpha"),
        (dict(hysteresis=-1.0), "hysteresis"),
        (dict(gap=-1.0), "gap"),
        (dict(outer_tol=0.0), "outer_tol"),
        (dict(outer_max_iter=0), "outer_max_iter"),
    ],
)
def test_the_five_shared_refusals_plus_the_outer_loops_own_two(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_mallet_gong(**kwargs)


def test_the_telemetry_is_zero_before_the_first_step_and_never_nan_after():
    """A nested solve has to report both loops, or the cost claim is unfalsifiable.

    ``outer_residual`` is the one that starts at ``nan``: there is no residual before a step, and
    reporting ``0.0`` would say the loop converged when it has not run.
    """
    gong = make_mallet_gong()
    assert gong.n_outer == 0
    assert gong.n_solves == 0
    assert np.isnan(gong.outer_residual)
    assert gong.outer_converged and gong.inner_converged and not gong.outer_stalled
    for _ in range(200):
        gong.step()
        assert not np.isnan(gong.outer_residual)
        assert gong.n_solves >= gong.inner_iters  # at least one back-substitution per iteration


# -- the anchor: with the coupling off this IS model #7p -------------------------------------------


def test_a_linear_gong_reduces_to_the_linear_mallet_plate():
    """``nonlinear=False`` makes the plate affine in the contact force, so ``Psi(f)`` is
    ``u_free - g_s f`` and the chord's effective free node is ``u_free`` **whatever the trial force
    was**. The outer loop's first residual is exactly zero and it exits at one iteration.

    That count is the structural half, and it is the half a sign error in ``f_ext = -f e_node``
    cannot survive: with the sign flipped, ``u_eff`` stays a function of ``f`` and the loop takes
    two or more. The agreement itself is ``1e-13``-relative and **not** bitwise, because this model
    puts the force in the right-hand side before the solve while ``MalletPlate`` adds the influence
    column after it — and a sparse LU back-substitution is not a linear map over doubles.
    """
    gong = make_mallet_gong(nonlinear=False)
    twin = gong_linear_twin(gong)
    assert gong._g_s == twin._g_s  # the same column, to the bit
    assert gong._g_h == twin._g_h
    assert gong.node == twin.node

    worst_field = worst_force = 0.0
    ever_bitwise = False
    for _ in range(400):
        gong.step()
        twin.step()
        assert gong.n_outer == 1
        scale = max(np.abs(twin.plate.u).max(), 1e-300)
        worst_field = max(worst_field, np.abs(gong.plate.u - twin.plate.u).max() / scale)
        worst_force = max(
            worst_force,
            abs(gong.contact_force - twin.contact_force) / max(abs(twin.contact_force), 1.0),
        )
        ever_bitwise |= np.array_equal(gong.plate.u, twin.plate.u)
    assert worst_field < 1e-13, worst_field  # measured 2.08e-14
    assert worst_force < 1e-13, worst_force
    assert not ever_bitwise, (
        "the two routes rounded identically on some step -- one of them stopped solving the way "
        "its documentation says it does"
    )


def test_a_mallet_that_never_lands_leaves_the_gong_bit_identical():
    """Structural, not a floating-point coincidence: a zero contact force short-circuits to the
    force-free advance and returns it *unmodified*. Driving the plate with a zero force **vector**
    would add ``+0.0`` into every right-hand-side entry instead — the identity for every double
    except ``-0.0``.
    """
    gong = make_mallet_gong(strike_velocity=-1.0, gap=0.01)
    bare = VKPlate(
        Lx=gong.plate.Lx, Ly=gong.plate.Ly, fs=gong.plate.fs, N=gong.plate.N, **GONG_MATERIAL
    )
    ic = 2.0 * THICKNESS * np.sin(np.linspace(0.0, 3.0, gong.plate.n_live))
    gong.plate.set_state(ic)
    bare.set_state(ic)
    for _ in range(150):
        gong.step()
        bare.step()
        assert gong.contact_force == 0.0
        assert gong.n_outer == 0
        assert np.array_equal(gong.plate.u, bare.u)


# -- energy ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("v0", [3.0, 6.0, 12.0])
def test_energy_is_conserved_through_the_nested_solve(v0):
    """The money test, and what is new in it.

    The plate is implicit and the contact force enters its right-hand side, so dotting the
    θ-scheme with ``w^{n+1} - w^{n-1}`` gives the same discrete power ``f · δ_t· u`` a force does
    on a membrane — the von Kármán coupling telescopes on its own and adds nothing to the contact's
    ledger. What is new is that the identity now holds only as far as the **outer** loop converges,
    which is why this bar is a function of ``outer_tol`` and not of ``newton_tol``.

    The membrane-share assertion is what stops this from silently re-testing the linear θ-scheme
    at a velocity that never left the linear regime.
    """
    drift, we, share, _ = ring(make_mallet_gong(strike_velocity=v0), 2000)
    assert drift < 1e-11, drift  # measured 4e-13 .. 3e-12, against a 1e-10 contract
    assert share > 0.01, f"membrane share {share:.4f} — this strike stayed linear"
    assert we > 1.0, f"peak w/e {we:.3f} — the plate never reached its own thickness"


@pytest.mark.parametrize("lossy", [dict(sigma=2.0), dict(hysteresis=5.0)])
def test_loss_and_hysteresis_each_make_the_total_monotone(lossy):
    gong = make_mallet_gong(**lossy)
    prev = gong.energy()
    worst = 0.0
    for _ in range(1200):
        gong.step()
        e = gong.energy()
        worst = max(worst, (e - prev) / abs(prev))
        prev = e
    assert worst < 1e-13, worst


def test_the_drift_answers_to_the_outer_tolerance():
    """Self-certification, in the shape ``test_vk_energy.py`` uses: a scheme with no closed form
    proves its conservation is the discrete gradient's and not the solver's luck by showing the
    drift falls when the solver is asked for more.

    Here the tolerance that has to move it is the **outer** one — this model's own. Measured, the
    fall stops around ``6e-13``, which is not a solver limit at all but the energy read-out's own
    reduction rounding over 361 live nodes plus 441 stress-function nodes; ``outer_tol = 1e-14``
    is where the two meet, and the shipped default is one decade looser because that is where no
    step stalls.
    """
    drifts = [ring(make_mallet_gong(outer_tol=t), 1200)[0] for t in (1e-8, 1e-10, 1e-12)]
    assert drifts[0] > drifts[1] > drifts[2], drifts
    assert drifts[0] / drifts[2] > 100.0, drifts


def test_the_free_branch_is_a_read_out_bar_with_the_supported_one_as_its_control():
    """A point strike feeds the free plate's ``{1, x, y}`` rigid nullspace and the cymbal recoils
    for ever after — physics, and model #7p attributed the resulting energy error to
    ``Plate.energy()``'s bending form rather than to the mallet. It carries over unchanged, and the
    von Kármán term adds nothing to it because the Monge–Ampère bracket annihilates the nullspace
    too.

    The supported branch on the identical mallet is the control that makes this an attribution.
    """
    free, _, _, _ = ring(make_mallet_gong(boundary="free"), 4000)
    supported, _, _, _ = ring(make_mallet_gong(boundary="supported"), 4000)
    assert supported < 1e-11, supported
    assert free < 1e-8, free
    assert free > 10.0 * supported, (free, supported)


# -- the payoff ---------------------------------------------------------------------------------


def _scaled_departure(nonlinear, alpha, ratio=4.0, steps=6000):
    """Departure from an exactly proportional response between a loud and a quiet strike.

    ``ratio`` is a power of two on purpose: scaling a double by 4 is exact, so nothing in the
    *arithmetic* of a degree-one system need round differently between the two runs. That is
    necessary and it is **not sufficient**, which is what the first version of this file got
    wrong — the caller names the two absolute scales that break it, and ``ratio=8`` is a power of
    two that reads 4.30e-13 on the machine where ``ratio=4`` reads 0.0.
    """
    pickup = None
    sig = []
    for v0 in (6.0, 6.0 / ratio):
        gong = make_mallet_gong(strike_velocity=v0, alpha=alpha, nonlinear=nonlinear)
        if pickup is None:
            pickup = gong.plate.pickup_index_at(0.63 * gong.plate.Lx, 0.21 * gong.plate.Ly)
        sig.append(ring(gong, steps, pickup=pickup)[3])
    loud, soft = sig
    return float(np.abs(loud - ratio * soft).max() / np.abs(loud).max())


def test_on_a_linear_plate_an_alpha_one_felt_scales_exactly_and_a_gong_does_not():
    """**The batch's headline.**

    At ``alpha = 1`` every part of the exciter's *physics* is homogeneous of degree one — the head
    is a mass, the felt a linear spring, and the one-sided switching is scale-invariant — so on a
    *linear* plate a four-times-harder strike gives a four-times-larger response and nothing else.

    **The solver is not homogeneous, and this assertion used to say it was.** It was written
    ``== 0.0``, which is what this one fixture reads on Windows, and it failed on every Linux CI
    runner at a byte-identical ``3.52e-13`` — nineteen runs, two job families, a dozen unrelated
    commits, the same digits, so a last-bits difference in that machine's arithmetic rather than
    anything wandering. The physics is degree one; the *numerics* carries two *absolute* scales
    that a four-times-larger trajectory meets at a different place — the discrete-gradient force's
    0/0 Taylor-branch threshold (``ContactParams::tol``) and the bracketed scalar root find's own
    exit. Neither is a defect: the Taylor branch is model #7's deliberate 0/0 guard, and making
    either scale relative would move numbers across the whole contact family. An exactly
    proportional response is a property of the continuous scheme, not of the code.

    That the zero was a **fixture** and not an identity is measurable here as well as on the
    runner. Holding everything else fixed, ``ratio=8`` and ``strike_velocity=1.5`` both read
    4.30e-13 on the same machine that reads 0.0 for this pair — two innocuous, exactly
    representable changes.

    So the bar is the project's tier-1 acceptance number rather than one chosen to clear what was
    observed (``HANDOFF.md`` §6.1, and ``CLAUDE.md`` says not to tighten it): 232× above the worst
    departure measured either way, and ten orders of magnitude below the gong's 2.12. **What the
    exact zero was protecting is untouched** — the gong's number is attributable to the plate
    because the two readings are separated, and on the runner that produced the nonzero one they
    are separated by twelve orders.

    Model #7p's finding was "the felt exponent is the only source of dynamic timbre". That is a
    statement about a linear resonator. On a gong the felt can be removed from the question
    entirely and the timbre still moves — by twice the loud strike's own peak.
    """
    control = _scaled_departure(nonlinear=False, alpha=1.0)
    assert control < 1e-10, control  # 0.0 here, 3.52e-13 on the Linux runner, 4.30e-13 worst seen
    gong = _scaled_departure(nonlinear=True, alpha=1.0)
    assert gong > 1.0, gong  # measured 2.12


def test_the_gongs_spectral_centroid_climbs_with_the_strike_and_a_linear_plates_does_not():
    """*How* it changes: energy cascades up the spectrum as the strike gets harder — the crash.

    Both sources of dynamic timbre are present and the measurement separates them by a factor of
    **450**: across a sixteenfold range of strike velocity the linear plate's centroid moves
    0.16% — that is the felt, model #7p's effect, and it is real — while the gong's moves 74%.

    So this detector is not blind to the felt, it is merely almost blind to it, and the reason is
    worth keeping: a power-weighted centroid on a plate is dominated by the lowest partials, whose
    comb a shorter contact pulse barely reshapes. The scaled-response test above weights the two
    quite differently (1.17 against 2.12, a factor of under two). Two detectors, two very different
    splits of the same pair of causes — which is the air-box family's standing rule about no single
    detector being sufficient, arriving on a contact model.
    """
    def centroid(sig):
        f, mag, _ = magnitude_spectrum(sig, 48000.0)
        w = mag[f > 5.0] ** 2
        return float((f[f > 5.0] * w).sum() / w.sum())

    out = {}
    for nonlinear in (False, True):
        row = []
        for v0 in (0.75, 3.0, 12.0):
            gong = make_mallet_gong(strike_velocity=v0, nonlinear=nonlinear)
            pickup = gong.plate.pickup_index_at(0.63 * gong.plate.Lx, 0.21 * gong.plate.Ly)
            row.append(centroid(ring(gong, 16384, pickup=pickup)[3]))
        out[nonlinear] = row

    felt_only = out[False]
    plate = out[True]
    felt_swing = max(felt_only) / min(felt_only) - 1.0
    plate_swing = plate[-1] / plate[0] - 1.0
    assert felt_swing < 0.01, felt_only  # measured 0.0016
    assert plate == sorted(plate), plate
    assert plate_swing > 0.4, plate  # measured 0.74
    assert plate_swing > 100.0 * felt_swing, (plate_swing, felt_swing)


# -- the instrument -------------------------------------------------------------------------------


def test_the_outer_tangent_is_closed_form_and_bounds_the_iteration_count():
    """The plan this batch closes says a gong has "no closed-form derivative for the outer
    iteration, a secant or a bracket instead". It has one.

    Differentiating the fixed point gives ``dw/df = -J^-1 influence`` for exactly the ``J`` the
    von Kármán Newton batch already asserted against a finite difference, so the true tangent is
    ``g_exact = [J^-1 influence]_node + g_h`` and the chord's contraction is bounded by
    ``|1 - g_exact/g|``. Reached through the binding here, which is what makes it usable as an
    instrument rather than a native curiosity — it needs the plate's *pre-step* buffers, so the
    caller has to supply them and the signature says so.
    """
    gong = make_mallet_gong(strike_velocity=6.0)
    best = (0.0, None)
    for _ in range(600):
        before = (gong.plate.u.copy(), gong.plate.u_prev.copy(), gong.plate.F_prev.copy())
        gong.step()
        if gong.contact_force > best[0]:
            best = (gong.contact_force, (before, gong.plate.u.copy()))
    force, ((u, u_prev, f_prev), w) = best
    g_exact, response, products = gong._drive_point_tangent(
        u=u, u_prev=u_prev, F_prev=f_prev, force=force, w=w
    )
    assert products > 0
    assert g_exact == response + gong._g_h
    bound = abs(1.0 - g_exact / gong._g)
    # Measured 5.1e-3 at this mass and strike -- better than two decades of contraction per outer
    # iteration.
    assert 1e-6 < bound < 5e-2, bound
    # And the loop does spend the handful of iterations that implies. Note the count is *smaller*
    # than a contraction from an O(1) start would give: the chord's first guess is the exact
    # linear model's answer, so the iteration begins already close and the bound only has to
    # close the nonlinear part of the gap.
    worst = 0
    for _ in range(400):
        gong.step()
        worst = max(worst, gong.n_outer)
    assert worst <= 3, worst
