"""Validation for the energy-conserving bridge to a **free-edge (FFFF) grid plate** body (Step 5).

The string terminus drives a *suspended* free-edge Kirchhoff plate (model #5b -- a cymbal/gong) at a
single driving-point node through a linear spring. Step 5 generalizes :class:`StringPlateBridge` to
the free boundary; the only boundary-specific code is the Sherman-Morrison guard's plate block,
which becomes ``rho_s [W + (theta - 1/4) k^2 kappa^2 K]`` (``W`` the lumped-area diagonal mass, no
extra ``h^2``; ``K`` the free-edge stiffness). Everything else (force injection into the implicit
RHS, energy, pressure) is delegated to :class:`Plate`, which already branches on ``boundary``.

Same battery as the supported bridge (test_plate_connection.py): total energy conservation,
passivity under loss, a K=0 bit-identity to the uncoupled parts, real energy transfer, the radiation
read-out carrying the coupling force, and the exact stability guard with a *bracketed onset* (just
inside conserves; a guard-bypassed spring just past the ceiling diverges). Plus one free-specific
check: the plate's rigid-body content stays bounded -- the single-point spring cannot pump the
``w_dp = 0`` nullspace modes, so there is no rigid drift.
"""

import numpy as np
import pytest
from helpers import make_free_plate_bridge

from physsynth.core.connection import StringPlateBridge
from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.plate import Plate
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-10


def _pluck(bridge, amplitude=1e-3):
    s = bridge.string
    s.set_state(triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=amplitude))
    return bridge


def _ceiling_K(**kw) -> float:
    """The exact spring-stiffness ceiling: margin is linear in K, so scale a probe to margin = 1."""
    probe = make_free_plate_bridge(K=1000.0, **kw)
    return 1000.0 / probe.stability_margin


# -- Criterion 1: the TOTAL energy is conserved (string alone is not) -----------------------
@pytest.mark.parametrize("lam", [0.9, 0.7, 0.5])
def test_total_energy_conserved_across_lambda(lam):
    bridge = _pluck(make_free_plate_bridge(lam=lam))
    res = simulate(bridge, num_steps=int(1.5 * bridge.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"total drift {res.energy_drift:.2e} at lam={lam}"


@pytest.mark.parametrize("K", [500.0, 3000.0, 6000.0, 10000.0])
def test_total_energy_conserved_across_stiffness(K):
    bridge = _pluck(make_free_plate_bridge(lam=0.9, K=K))
    res = simulate(bridge, num_steps=int(1.2 * bridge.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"total drift {res.energy_drift:.2e} at K={K}"


def test_string_energy_alone_is_not_conserved():
    # Sanity that the coupling is doing something: the string exchanges real energy with the plate,
    # so E_string wanders even though the total is pinned.
    bridge = _pluck(make_free_plate_bridge(lam=0.9))
    s = bridge.string
    e_str = [s.energy()]
    for _ in range(int(0.3 * s.fs)):
        bridge.step()
        e_str.append(s.energy())
    e_str = np.array(e_str)
    spread = (e_str.max() - e_str.min()) / e_str[0]
    assert spread > 1e-3, f"string energy barely moved ({spread:.2e}); coupling not engaged"


# -- energy transfer: the free plate ends up carrying a substantial share -------------------
def test_energy_flows_string_to_plate():
    bridge = _pluck(make_free_plate_bridge(lam=0.9))
    s, p = bridge.string, bridge.plate
    max_plate_frac = 0.0
    for _ in range(int(0.5 * s.fs)):
        bridge.step()
        total = s.energy() + p.energy()
        max_plate_frac = max(max_plate_frac, p.energy() / total)
    assert max_plate_frac > 0.1, f"free plate only ever held {max_plate_frac:.1%} of the energy"


# -- Criterion (passivity): with loss anywhere, the total decreases monotonically -----------
def test_passivity_with_plate_damping():
    bridge = _pluck(make_free_plate_bridge(lam=0.9, sigma_plate=10.0))
    res = simulate(bridge, num_steps=int(1.5 * bridge.string.fs))
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


def test_passivity_with_string_damping():
    bridge = _pluck(make_free_plate_bridge(lam=0.9, sigma_string=5.0))
    res = simulate(bridge, num_steps=int(1.5 * bridge.string.fs))
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


# -- K = 0: the coupled system is bit-identical to the two uncoupled parts -------------------
def test_K0_bit_identical_to_uncoupled_parts():
    bridge = make_free_plate_bridge(K=0.0)
    s, p = bridge.string, bridge.plate
    s.set_state(triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=1e-3))
    # excite the free plate too, so its path is non-trivial (a smooth interior bump).
    u0 = 1e-3 * np.cos(np.pi * p.X / p.Lx) * np.cos(np.pi * p.Y / p.Ly)
    p.set_state(u0)

    s_ref = IdealString(
        L=s.L, T=s.T, rho=s.rho, fs=s.fs, N=s.N, boundary=("fixed", "free")
    )
    s_ref.set_state(triangular_pluck(s_ref.x, s_ref.L, 0.137 * s_ref.L, amplitude=1e-3))
    p_ref = Plate(
        Lx=p.Lx, Ly=p.Ly, kappa=p.kappa, rho=p.rho, fs=p.fs, N=p.N, boundary="free", nu=p.nu
    )
    p_ref.set_state(u0)

    for _ in range(400):
        bridge.step()
        s_ref.step()
        p_ref.step()
    assert np.array_equal(s.u, s_ref.u), "string not bit-identical to uncoupled at K=0"
    assert np.array_equal(p.u, p_ref.u), "free plate not bit-identical to uncoupled at K=0"


# -- no rigid drift: the single-point spring cannot excite the w_dp = 0 nullspace modes ------
def test_no_rigid_body_drift():
    # The free plate's {1, x, y} nullspace would allow unbounded rigid translation/tilt, but the
    # driving-point force is orthogonal to every zero-mode with v(dp) = 0, so starting from rest
    # those stay quiescent; the piston/tilt modes with v(dp) != 0 become bounded oscillators on the
    # spring. A rigid ramp (q ~ t) would grow the kinetic energy without bound -- already ruled out
    # by the drift bound -- but check the area-weighted mean displacement directly: it never ramps.
    bridge = _pluck(make_free_plate_bridge(lam=0.9, K=6000.0))
    p = bridge.plate
    area = float(np.sum(p.w))
    piston = []
    for n in range(int(1.5 * bridge.string.fs)):
        bridge.step()
        if n % 500 == 0:
            piston.append(float(np.dot(p.w, p.u)) / area)  # area-weighted mean (piston overlap)
    piston = np.array(piston)
    # Bounded and small (same order as the pluck amplitude), and the late half is no larger than the
    # early half -- a ramp would make the tail dominate.
    assert np.max(np.abs(piston)) < 1e-2, f"piston mode grew to {np.max(np.abs(piston)):.2e}"
    early = np.max(np.abs(piston[: len(piston) // 2]))
    late = np.max(np.abs(piston[len(piston) // 2 :]))
    assert late <= 3.0 * early + 1e-9, f"piston ramps: late {late:.2e} >> early {early:.2e}"


# -- radiation read-out captures the connection force (free-plate W-weighted injection) ------
def test_pressure_includes_coupling_term():
    # After a coupled step with F != 0, the plate's stored acceleration must carry the injected
    # driving force. For the free plate the injection is k^2 F / rho at the drive node (the W divide
    # is handled by the A-solve), so the forced-minus-unforced acceleration difference is exactly
    # A^-1 (k^2-free source s_F) -- the coupling term the naive -kappa^2 W^-1 K u read-out drops.
    bridge = _pluck(make_free_plate_bridge(K=6000.0))
    p = bridge.plate
    for _ in range(400):  # let the wave reach the terminus so the bridge force is engaged
        bridge.step()

    u, u_prev = p.u.copy(), p.u_prev.copy()
    F = bridge.connection_force()
    assert abs(F) > 0.0, "no bridge force to test against"

    p_free = Plate(
        Lx=p.Lx, Ly=p.Ly, kappa=p.kappa, rho=p.rho, fs=p.fs, N=p.N, boundary="free", nu=p.nu
    )
    p_free.u, p_free.u_prev = u.copy(), u_prev.copy()
    p_free.step()  # unforced reference from the identical state

    bridge.step()  # forced step

    s_f = np.zeros(p.n_live)
    s_f[bridge.drive_index] = F / p.rho  # free: rhs gains k^2 f_ext / rho (W divide is in A)
    expected_diff = p._lu.solve(s_f)  # accel_forced - accel_unforced = A^-1 s_F
    assert np.allclose(p._accel - p_free._accel, expected_diff, rtol=0, atol=1e-9), \
        "free-plate acceleration does not carry the injected coupling force"

    naive_pressure = float(np.dot(p.w, p_free._accel))  # what dropping the force gives
    assert abs(bridge.pressure() - naive_pressure) > 1e-9 * abs(naive_pressure), \
        "pressure() equals the un-forced read-out -- coupling term is missing"


# -- the exact Sherman-Morrison stability guard rejects an over-stiff spring -----------------
def test_unstable_stiffness_rejected():
    with pytest.raises(ValueError, match="unstable"):
        make_free_plate_bridge(lam=0.9, K=1e7)  # far past the guard bound


def test_margin_is_linear_in_stiffness():
    # The guard is a rank-1 (Sherman-Morrison) condition, so the margin is exactly linear in K.
    m1 = make_free_plate_bridge(K=1000.0).stability_margin
    m3 = make_free_plate_bridge(K=3000.0).stability_margin
    assert np.isclose(m3, 3.0 * m1, rtol=1e-9), f"margin not linear in K ({m3:.4e} vs {3*m1:.4e})"


def test_guard_holds_at_its_boundary():
    # A spring just inside the exact guard must still conserve energy to machine precision.
    K_in = 0.93 * _ceiling_K()
    bridge = _pluck(make_free_plate_bridge(K=K_in))
    assert bridge.stability_margin < 1.0
    res = simulate(bridge, num_steps=int(1.2 * bridge.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} near the guard boundary"


def test_just_over_the_ceiling_is_rejected():
    Kc = _ceiling_K()
    make_free_plate_bridge(K=0.95 * Kc)  # just inside: constructs fine
    with pytest.raises(ValueError, match="unstable"):
        make_free_plate_bridge(K=1.05 * Kc)  # just outside: rejected


def test_ceiling_is_the_true_instability_onset():
    # The free-plate guard block (rho_s [W + (theta-1/4) k^2 kappa^2 K]) must give the *actual*
    # physical onset, not merely a self-consistent formula. Bracket it: just inside conserves; a
    # guard-bypassed spring just past the ceiling (beta_s and f_ext are K-independent, so bumping .K
    # after construction is a clean bypass) actually diverges. This is what proves the free block is
    # the real boundary rather than a plausible paste of the supported one.
    Kc = _ceiling_K()
    inside = _pluck(make_free_plate_bridge(K=0.99 * Kc))
    res = simulate(inside, num_steps=int(1.0 * inside.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"0.99x should conserve, drift {res.energy_drift:.2e}"

    over = _pluck(make_free_plate_bridge(K=0.5 * Kc))
    over.K = 1.05 * Kc  # bypass the construction guard: just past the ceiling
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(0.3 * over.string.fs)):
            over.step()
    assert not np.isfinite(over.string.u).all(), "system did not diverge just past the ceiling"


def test_string_at_lambda_one_rejected():
    # lambda = 1 makes the guard's string block G0_str singular (the Nyquist trap); require < 1.
    with pytest.raises(ValueError, match="lambda < 1"):
        make_free_plate_bridge(lam=1.0)


# -- construction guards ---------------------------------------------------------------------
def test_mismatched_timestep_rejected():
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=22000.0, N=100, boundary=("fixed", "free"))
    p = Plate(Lx=1.0, Ly=1.0, kappa=20.0, rho=0.005, fs=24000.0, N=16, boundary="free")
    with pytest.raises(ValueError, match="timestep"):
        StringPlateBridge(string=s, plate=p, K=1000.0)


def test_right_end_must_be_free():
    fs = 22000.0
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=fs, N=100, boundary="fixed")  # both fixed
    p = Plate(Lx=1.0, Ly=1.0, kappa=20.0, rho=0.005, fs=fs, N=16, boundary="free")
    with pytest.raises(ValueError, match="free"):
        StringPlateBridge(string=s, plate=p, K=1000.0)


def test_drive_index_out_of_range_rejected():
    fs = 22000.0
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=fs, N=100, boundary=("fixed", "free"))
    p = Plate(Lx=1.0, Ly=1.0, kappa=20.0, rho=0.005, fs=fs, N=16, boundary="free")
    with pytest.raises(ValueError, match="drive_index"):
        StringPlateBridge(string=s, plate=p, K=1000.0, drive_index=p.n_live)
