"""Validation for the energy-conserving bridge connection (string terminus <-> modal body).

The connection is the core deliverable of the body/radiation node, so it faces the project's full
battery: total energy conserved to machine precision (E_string + E_body + E_conn — the string's own
energy is *not* conserved once coupled), passivity when either part is lossy, an exact bit-identity
to the uncoupled parts at K=0, a real energy-transfer check (the coupling does physical work), and
the exact leapfrog stability guard rejecting an over-stiff spring.
"""

import numpy as np
import pytest
from helpers import BODY_FREQS_DEFAULT, K_BRIDGE_DEFAULT, make_bridge

from physsynth.core.body import ModalBody
from physsynth.core.connection import StringBodyBridge
from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-10


def _pluck(bridge, amplitude=1e-3):
    s = bridge.string
    s.set_state(triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=amplitude))
    return bridge


# -- Criterion 1: the TOTAL energy is conserved (string alone is not) -----------------------
@pytest.mark.parametrize("lam", [0.9, 0.7, 0.5])
def test_total_energy_conserved_across_lambda(lam):
    bridge = _pluck(make_bridge(lam=lam))
    res = simulate(bridge, num_steps=int(2.0 * bridge.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"total drift {res.energy_drift:.2e} at lam={lam}"


@pytest.mark.parametrize("K", [500.0, 4000.0, 8000.0, 15000.0])
def test_total_energy_conserved_across_stiffness(K):
    bridge = _pluck(make_bridge(lam=0.9, K=K))
    res = simulate(bridge, num_steps=int(1.5 * bridge.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"total drift {res.energy_drift:.2e} at K={K}"


def test_string_energy_alone_is_not_conserved():
    # Sanity that the coupling is doing something: the string exchanges real energy with the body,
    # so E_string wanders even though the total is pinned.
    bridge = _pluck(make_bridge(lam=0.9))
    s = bridge.string
    e_str = [s.energy()]
    for _ in range(int(0.3 * s.fs)):
        bridge.step()
        e_str.append(s.energy())
    e_str = np.array(e_str)
    spread = (e_str.max() - e_str.min()) / e_str[0]
    assert spread > 1e-3, f"string energy barely moved ({spread:.2e}); coupling not engaged"


# -- energy transfer: the body ends up carrying a substantial share --------------------------
def test_energy_flows_string_to_body():
    bridge = _pluck(make_bridge(lam=0.9))
    s, b = bridge.string, bridge.body
    max_body_frac = 0.0
    for _ in range(int(0.5 * s.fs)):
        bridge.step()
        total = s.energy() + b.energy()
        max_body_frac = max(max_body_frac, b.energy() / total)
    assert max_body_frac > 0.1, f"body only ever held {max_body_frac:.1%} of the energy"


# -- Criterion (passivity): with loss anywhere, the total decreases monotonically ------------
def test_passivity_with_body_damping():
    bridge = _pluck(make_bridge(lam=0.9, sigma_body=10.0))
    res = simulate(bridge, num_steps=int(2.0 * bridge.string.fs))
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


def test_passivity_with_string_damping():
    bridge = _pluck(make_bridge(lam=0.9, sigma_string=5.0))
    res = simulate(bridge, num_steps=int(2.0 * bridge.string.fs))
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


# -- K = 0: the coupled system is bit-identical to the two uncoupled parts --------------------
def test_K0_bit_identical_to_uncoupled_parts():
    N, lam = 100, 0.9
    L, T, rho, c = 1.0, 200.0, 0.005, 200.0
    fs = c * N / (L * lam)
    q0 = 1e-3 * np.array([1.0, -0.5, 0.7, 0.3])  # excite the body too, so its path is non-trivial

    # coupled with K = 0
    s = IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs, masses=0.02)
    bridge = StringBodyBridge(string=s, body=b, K=0.0)
    s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
    b.set_state(q0)

    # standalone references
    s_ref = IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    b_ref = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs, masses=0.02)
    s_ref.set_state(triangular_pluck(s_ref.x, L, 0.137 * L, amplitude=1e-3))
    b_ref.set_state(q0)

    for _ in range(500):
        bridge.step()
        s_ref.step()
        b_ref.step()
    assert np.array_equal(s.u, s_ref.u), "string not bit-identical to uncoupled at K=0"
    assert np.array_equal(b.q, b_ref.q), "body not bit-identical to uncoupled at K=0"


# -- radiation read-out captures the connection force (not just the free response) -----------
def test_pressure_includes_coupling_term():
    # After a coupled step with F != 0, the stored modal acceleration must equal the FULL equation
    # of motion -omega^2 q + phi F / m -- i.e. it carries the bridge force. The naive reconstruction
    # -omega^2 q (indistinguishable on a standalone body) would drop that term; assert it differs.
    bridge = _pluck(make_bridge(lam=0.9, K=8000.0))
    b = bridge.body
    for _ in range(400):  # let the wave reach the terminus so the bridge force is engaged
        bridge.step()
    q_before = b.q.copy()
    F = bridge.connection_force()
    assert abs(F) > 0.0, "no bridge force to test against"

    bridge.step()  # b.q_prev is now q_before; b._accel is the just-taken second difference
    expected = -b.omega * b.omega * q_before + b.phi * F / b.m  # lossless EoM at that step
    assert np.allclose(b._accel, expected, rtol=0, atol=1e-9 * np.max(np.abs(expected))), \
        "modal acceleration does not match the forced EoM (connection term dropped?)"

    naive = float(np.dot(b.a, -b.omega * b.omega * q_before))  # what the reconstruction would give
    assert abs(b.pressure() - naive) > 1e-6 * abs(naive), \
        "pressure() equals the un-forced reconstruction -- coupling term is missing"


# -- the exact stability guard rejects an over-stiff spring ----------------------------------
def test_unstable_stiffness_rejected():
    with pytest.raises(ValueError, match="unstable"):
        make_bridge(lam=0.9, K=1e6)  # far past the leapfrog bound


def test_guard_holds_at_its_boundary():
    # A spring just inside the exact guard must still conserve energy to machine precision -- the
    # guard is not merely "doesn't NaN", it is the true energy-preserving bound.
    bridge = _pluck(make_bridge(lam=0.9, K=20000.0))  # close to the ~21.5k ceiling at lam=0.9
    assert bridge.k * bridge.k * bridge.spectral_radius < 4.0
    res = simulate(bridge, num_steps=int(1.5 * bridge.string.fs))
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} near the guard boundary"


# -- construction guards ---------------------------------------------------------------------
def test_mismatched_timestep_rejected():
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=22000.0, N=100, boundary=("fixed", "free"))
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=24000.0)  # different fs
    with pytest.raises(ValueError, match="timestep"):
        StringBodyBridge(string=s, body=b, K=K_BRIDGE_DEFAULT)


def test_right_end_must_be_free():
    fs = 22000.0
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=fs, N=100, boundary="fixed")  # both fixed
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs)
    with pytest.raises(ValueError, match="free"):
        StringBodyBridge(string=s, body=b, K=K_BRIDGE_DEFAULT)
