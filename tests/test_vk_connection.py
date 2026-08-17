"""Validation for the bridge to a **von Kármán** plate body (`StringVKPlateBridge`).

A string terminated on model #6 — the last thing `HANDOFF.md` §12H left out of the family. The
spring, the explicit ``F = K eta^n`` and the three-way energy decomposition are
:class:`StringPlateBridge`'s; what is new is that the body's driving-point impedance is no longer a
property of the body alone but of the body *and how hard the string is playing it*.

The battery has four layers:

* **the regression** — ``nonlinear=False`` is bit-identical to the linear bridge on the plate's
  areal-density twin, which is what polices the ``rho_v``/``rho_s`` factor of 1000 that no energy
  ledger can see;
* **the ledgers** — lossless conservation, passivity, ``K = 0`` decoupling, and the string's own
  energy demonstrably not conserved;
* **the claim** — a linear body's response scales *bit-exactly* with the pluck, so the departure
  from linearity has a machine-precision zero to be measured against, and it is second order in the
  pluck amplitude;
* **the guard** — the exact Sherman-Morrison margin is still sufficient (the membrane energy is a
  sum of squared norms and cannot subtract), and the failure mode has migrated to non-convergence,
  which the guard is structurally blind to.

See ``docs/dev/string-vk-plate-bridge-plan.md``.
"""

import numpy as np
import pytest
from helpers import (
    VK_BRIDGE_SIDE,
    make_vk_bridge_linear_twin,
    make_vk_plate_bridge,
)

from physsynth.core.connection import StringVKPlateBridge
from physsynth.core.exciter import triangular_pluck
from physsynth.core.plate import VKPlate
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-10
BOUNDARIES = ["supported", "free"]


def _pluck(bridge, amplitude=3e-4):
    s = bridge.string
    s.set_state(triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=amplitude))
    return bridge


def _run(bridge, steps):
    """Step, returning (relative energy drift, min/max energies, worst sweeps, all-converged)."""
    e0 = bridge.energy()
    lo = hi = e0
    worst_iters = 0
    converged = True
    for _ in range(steps):
        bridge.step()
        e = bridge.energy()
        lo, hi = min(lo, e), max(hi, e)
        worst_iters = max(worst_iters, bridge.n_iters)
        converged &= bridge.converged
    return (hi - lo) / abs(e0), worst_iters, converged


# -- Layer 1: the regression, and the density substitution it polices ------------------------
@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("sigma_plate", [0.0, 2.0])
def test_nonlinear_false_is_the_linear_bridge_bit_identical(boundary, sigma_plate):
    """``nonlinear=False`` must reproduce :class:`StringPlateBridge` **exactly** — state, energy
    and stability margin.

    This is the test that catches ``rho_v`` for ``rho_s``. That substitution does not raise (both
    attributes exist), is a factor of 1000 at ``e = 1 mm``, and leaves every energy ledger green,
    because each side of the coupling telescopes against whatever force *it* used. Only a
    bit-for-bit comparison against an independently-built linear body sees it — in the margin
    immediately, and in the trajectory as soon as the force enters the RHS.
    """
    vk = make_vk_plate_bridge(boundary=boundary, nonlinear=False, sigma_plate=sigma_plate)
    lin = make_vk_bridge_linear_twin(vk)

    assert vk.stability_margin == lin.stability_margin  # exact, not approx: same two solves
    assert vk.drive_index == lin.drive_index

    _pluck(vk)
    _pluck(lin)
    for n in range(300):
        vk.step()
        lin.step()
        assert np.array_equal(vk.string.u, lin.string.u), f"string diverged at step {n}"
        assert np.array_equal(vk.plate.u, lin.plate.u), f"plate diverged at step {n}"
    assert vk.energy() == lin.energy()
    assert vk.connection_force() == lin.connection_force()


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_the_margin_ignores_the_nonlinearity_and_uses_the_areal_density(boundary):
    """The *nonlinear* bridge's margin equals the linear twin's, exactly — two statements at once.

    It is not implied by the regression above, which only compares ``nonlinear=False`` bridges: the
    guard is a statement about the *linear* blocks, so it must come out identical with the coupling
    switched on. And because the twin's plate is built with ``rho=vk.rho_s``, equality also pins the
    density: reading ``rho_v`` here would be a factor of ``1/e`` — 10^4 on this rig — and would
    still raise nothing, pass construction, and leave every energy ledger green.
    """
    nl = make_vk_plate_bridge(boundary=boundary)
    lin = make_vk_bridge_linear_twin(make_vk_plate_bridge(boundary=boundary, nonlinear=False))
    assert nl.plate.nonlinear  # the twin is a bare linear Plate, which has no such flag
    assert nl.stability_margin == lin.stability_margin
    assert nl.plate.rho_s == nl.plate.rho_v * nl.plate.e  # the two the trap confuses


# -- Layer 2: the ledgers --------------------------------------------------------------------
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_total_energy_conserved_lossless(boundary):
    """``E_string + E_plate + E_conn`` is conserved to machine precision at large amplitude.

    Asserted *with* the convergence flag: the total is arithmetic on whatever ``w^{n+1}`` came out
    of the solve, so an under-converged step is ported self-consistently and the drift alone cannot
    tell the difference (batch 6's third blind spot, inherited).
    """
    bridge = _pluck(make_vk_plate_bridge(boundary=boundary), amplitude=1e-3)
    drift, worst_iters, converged = _run(bridge, 2500)
    assert converged, "Picard failed to converge; the drift below is not a valid measurement"
    assert worst_iters > 1, "the nonlinearity never engaged (one sweep = the linear path)"
    assert drift < DRIFT_TOL, f"total drift {drift:.2e} ({boundary})"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_nonlinearity_is_genuinely_engaged(boundary):
    """The claim regime is real: the plate is driven past its own thickness by the string alone."""
    bridge = _pluck(make_vk_plate_bridge(boundary=boundary), amplitude=1e-3)
    peak = 0.0
    for _ in range(2500):
        bridge.step()
        peak = max(peak, float(np.max(np.abs(bridge.plate.u))))
    assert peak > bridge.plate.e, f"peak w = {peak:.2e} never reached e = {bridge.plate.e:.2e}"
    assert bridge.plate.membrane_energy() > 0.0


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("lossy", ["plate", "string"])
def test_passivity_with_loss(boundary, lossy):
    """With loss on either part the total decreases monotonically (never increases).

    The tolerance is ``1e-12 * E0`` — relative to the run's own initial energy, and lifted from
    :func:`test_plate_connection.test_passivity_with_plate_damping` rather than invented. An
    *absolute* epsilon here would have been a few ulps of this rig's ~8.5e-4 J, which is exactly the
    kind of bar CLAUDE.md §6.1 keeps headroom against: this suite is the acceptance contract for a
    native port under a different compiler and BLAS.
    """
    kw = {"sigma_plate": 3.0} if lossy == "plate" else {"sigma_string": 1.0}
    bridge = _pluck(make_vk_plate_bridge(boundary=boundary, **kw), amplitude=1e-3)
    e0 = e_prev = bridge.energy()
    for n in range(1200):
        bridge.step()
        e = bridge.energy()
        assert e - e_prev <= 1e-12 * e0, f"energy rose at step {n}: {e_prev:.12e} -> {e:.12e}"
        e_prev = e
    assert e_prev < e0, "the lossy run never lost anything"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_zero_stiffness_decouples_bit_identically(boundary):
    """``K = 0`` must leave both parts exactly as they would run alone."""
    bridge = _pluck(make_vk_plate_bridge(boundary=boundary, K=0.0), amplitude=1e-3)
    solo_string = IdealString(
        L=bridge.string.L, T=bridge.string.T, rho=bridge.string.rho, fs=bridge.string.fs,
        N=bridge.string.N, boundary=("fixed", "free"),
    )
    solo_string.set_state(triangular_pluck(solo_string.x, solo_string.L,
                                           0.137 * solo_string.L, amplitude=1e-3))
    for _ in range(200):
        bridge.step()
        solo_string.step()
        assert np.array_equal(bridge.string.u, solo_string.u)
    assert bridge.stability_margin == 0.0
    assert np.all(bridge.plate.u == 0.0)  # never driven


def test_string_energy_alone_is_not_conserved():
    """Sanity that the coupling carries real energy — the total is the thing to assert on."""
    bridge = _pluck(make_vk_plate_bridge(), amplitude=1e-3)
    e_str = [bridge.string.energy()]
    for _ in range(1500):
        bridge.step()
        e_str.append(bridge.string.energy())
    e_str = np.array(e_str)
    assert (e_str.max() - e_str.min()) / e_str[0] > 1e-3


# -- Layer 3: the claim ----------------------------------------------------------------------
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_a_linear_body_scales_bit_exactly_with_the_pluck(boundary):
    """The control, and it is exact rather than close.

    Doubling the pluck doubles every quantity in a linear leapfrog and an LU back-substitution,
    because scaling by 2 commutes with the +-, x and / of both. So the residual is ``0.0``, not
    1e-16 — a machine-precision zero for the nonlinear departure to be measured against, obtained
    without an oracle. If this ever fails, something on the path is not homogeneous.
    """
    a = _pluck(make_vk_plate_bridge(boundary=boundary, nonlinear=False), amplitude=5e-5)
    b = _pluck(make_vk_plate_bridge(boundary=boundary, nonlinear=False), amplitude=1e-4)
    for n in range(600):
        a.step()
        b.step()
        assert np.array_equal(2.0 * a.string.u, b.string.u), f"string not exact at step {n}"
        assert np.array_equal(2.0 * a.plate.u, b.plate.u), f"plate not exact at step {n}"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_departure_from_a_linear_body_is_second_order_in_the_pluck(boundary):
    """The headline: the string's trajectory leaves the linear-body one at **second order**.

    Distance is ``max_t ||u_nl - u_lin||_inf / max_t ||u_lin||_inf`` over a fixed window, against
    the *same* plate with ``nonlinear=False`` — so it is identically zero for a linear body (the
    test above) and everything it measures is the von Karman coupling.

    Only the small-amplitude ratios are asserted. The measure **saturates** near 0.81 once the two
    trajectories decorrelate in phase (two decorrelated waveforms differ by ~their own amplitude
    however mildly they decorrelated), so it is a detector with an order, not a magnitude — the
    plan's §0.1. Measured orders on this rig: 1.99 and 1.94 (supported), 1.97 and 1.87 (free).
    """
    amps = [1e-5, 2e-5, 5e-5]
    window = 500
    dists = []
    for amp in amps:
        lin = _pluck(make_vk_plate_bridge(boundary=boundary, nonlinear=False), amplitude=amp)
        nl = _pluck(make_vk_plate_bridge(boundary=boundary), amplitude=amp)
        num = den = 0.0
        for _ in range(window):
            lin.step()
            nl.step()
            num = max(num, float(np.max(np.abs(nl.string.u - lin.string.u))))
            den = max(den, float(np.max(np.abs(lin.string.u))))
        assert nl.converged
        dists.append(num / den)

    assert dists[0] > 0.0, "the nonlinearity produced no departure at all"
    for lo in range(len(amps) - 1):
        hi = lo + 1
        order = np.log(dists[hi] / dists[lo]) / np.log(amps[hi] / amps[lo])
        assert 1.6 < order < 2.2, (
            f"departure order {order:.2f} between {amps[lo]:.0e} and {amps[hi]:.0e} ({boundary})"
        )


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_linear_energy_share_is_amplitude_invariant_and_the_gong_s_is_not(boundary):
    """What the body takes is a *constant* for any linear body, and a function of dynamics here.

    Every energy scales as amplitude^2, so their ratio does not scale at all: the linear body's
    share is identical to ~machine precision across a 100x amplitude range, without an oracle. The
    von Karman body's is not. Only that split is asserted — the *size* and even the *sign* of the
    shift depend on where the drive point sits relative to the impedance match (measured both ways
    on different rigs), and the peak share is reached early enough that it reports the drive point
    rather than the whole plate. See the plan's §0.4.
    """
    def peak_share(bridge, steps=800):
        best = 0.0
        for _ in range(steps):
            bridge.step()
            total = bridge.string.energy() + bridge.plate.energy()
            if total > 0:
                best = max(best, bridge.plate.energy() / total)
        return best

    amps = [1e-5, 1e-4, 1e-3]
    lin = [peak_share(_pluck(make_vk_plate_bridge(boundary=boundary, nonlinear=False), a))
           for a in amps]
    nl = [peak_share(_pluck(make_vk_plate_bridge(boundary=boundary), a)) for a in amps]

    assert lin[1] == pytest.approx(lin[0], rel=1e-12)
    assert lin[2] == pytest.approx(lin[0], rel=1e-12)
    assert abs(nl[2] - nl[0]) / nl[0] > 1e-3, "the gong's share did not move with the pluck"


# -- Layer 4: the guard, and the failure it cannot see ---------------------------------------
def test_rigid_modes_are_immune_to_the_nonlinearity():
    """The free plate's ``{1, x, y}`` nullspace survives model #6 untouched.

    The Monge-Ampere bracket is built from second derivatives, so ``l(w, w) = 0`` identically for
    any ``w`` in span{1, x, y}: no stress function is generated, no coupling force appears, and
    :class:`StringPlateBridge`'s no-rigid-drift argument carries over word for word. Started in a
    pure tilt, the plate must simply *stay* tilted.
    """
    bridge = make_vk_plate_bridge(boundary="free", K=0.0)
    p = bridge.plate
    tilt = ((p.X - 0.5 * p.Lx) * 2.5e-2)[p.mask]
    p.set_state(tilt, v0=0.0)
    assert np.max(np.abs(p.F)) == 0.0  # F(w^0) is exactly zero, not merely small
    for _ in range(400):
        p.step()
    assert np.max(np.abs(p.F)) < 1e-18, "the nonlinearity woke up on a rigid mode"
    assert np.max(np.abs(p.u - tilt)) < 1e-12 * VK_BRIDGE_SIDE
    assert abs(p.energy()) < 1e-15


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_guard_rejects_an_overstiff_spring(boundary):
    """Construction fails above the exact margin, and the margin is linear in ``K``."""
    probe = make_vk_plate_bridge(boundary=boundary, K=1000.0)
    ceiling = 1000.0 / probe.stability_margin
    ok = make_vk_plate_bridge(boundary=boundary, K=0.95 * ceiling)
    assert ok.stability_margin == pytest.approx(0.95, rel=1e-9)
    with pytest.raises(ValueError, match="connection unstable"):
        make_vk_plate_bridge(boundary=boundary, K=1.05 * ceiling)


def test_the_linear_margin_survives_a_strongly_nonlinear_run():
    """At 95% of the exact ceiling the coupled system still conserves, at large amplitude.

    The reason it can: the conserved total is ``E_lin + H_mem + E_conn``, and
    ``H_mem = 1/2 (H(F^{n+1}) + H(F^n))`` is a sum of two squared norms with no cross-time term
    (unlike the theta-weighted bending potential, which has an indefinite one). It cannot subtract,
    so a positive-definite linear form stays coercive however hard the plate is driven.
    """
    probe = make_vk_plate_bridge(K=1000.0)
    ceiling = 1000.0 / probe.stability_margin
    bridge = _pluck(make_vk_plate_bridge(K=0.95 * ceiling), amplitude=1e-3)
    drift, _, converged = _run(bridge, 1500)
    assert converged
    assert drift < DRIFT_TOL, f"drift {drift:.2e} at 95% of the exact margin"


def test_the_failure_mode_migrates_to_non_convergence():
    """A configuration the guard **passes** can still fail — and the guard cannot see it.

    Exact conservation holds only *at* the Picard fixed point, so the guard (a statement about a
    quadratic form) is structurally blind to non-convergence (a statement about a fixed point).
    Driven hard enough, the plate hits its sweep cap while the margin stays comfortably below 1.
    This is why the bridge surfaces the convergence diagnostics at all, and why they must be read
    per step rather than at the end of a run.
    """
    probe = make_vk_plate_bridge(K=1000.0)
    ceiling = 1000.0 / probe.stability_margin
    bridge = _pluck(make_vk_plate_bridge(K=0.9 * ceiling, couple_max_iter=12), amplitude=3e-2)
    assert bridge.stability_margin < 1.0  # the guard passed it

    failed_at = None
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(400):
            bridge.step()
            if not bridge.converged:
                failed_at = n
                break
    assert failed_at is not None, "expected the sweep cap to be reached"
    assert bridge.n_iters == 12
    assert bridge.last_residual > bridge.plate.couple_tol


# -- construction validation -----------------------------------------------------------------
def test_construction_rejects_mismatched_and_malformed_inputs():
    good = make_vk_plate_bridge()
    s, p = good.string, good.plate

    other = IdealString(L=s.L, T=s.T, rho=s.rho, fs=2.0 * s.fs, N=s.N,
                        boundary=("fixed", "free"))
    with pytest.raises(ValueError, match="share a timestep"):
        StringVKPlateBridge(string=other, plate=p, K=1.0)

    clamped = IdealString(L=s.L, T=s.T, rho=s.rho, fs=s.fs, N=s.N, boundary=("fixed", "fixed"))
    with pytest.raises(ValueError, match="right end must be 'free'"):
        StringVKPlateBridge(string=clamped, plate=p, K=1.0)

    with pytest.raises(ValueError, match="must be >= 0"):
        StringVKPlateBridge(string=s, plate=p, K=-1.0)

    with pytest.raises(ValueError, match="out of range"):
        StringVKPlateBridge(string=s, plate=p, K=1.0, drive_index=p.n_live)


def test_bridge_exposes_no_pressure_readout():
    """Deliberate: batch 6 measured the compact monopole at 3e-7 of the truth for a gong, and
    moving the *wrong way* for a suspended cymbal. The radiation path is the room classes."""
    bridge = make_vk_plate_bridge()
    assert not hasattr(bridge, "pressure")
    assert not hasattr(VKPlate, "pressure")
