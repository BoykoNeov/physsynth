"""Validation for sympathetic / coupled strings (several strings sharing one bridge point on a
common modal body — piano unisons, sitar/tanpura sympathetics; HANDOFF §12.B).

Energy conservation and passivity follow automatically from the linear-leapfrog structure, so they
are necessary but **not discriminating** (a flipped coupling sign would still conserve energy). The
sharp tests here are the *structural* ones the energy report cannot see:

* **antisymmetric normal mode** — two identical strings started ``u_B = -u_A`` with the body at
  rest keep the bridge exactly still (``w_b ≡ 0``, ``E_body ≡ 0``) and stay ``u_B ≡ -u_A`` to
  machine precision; a wrong coupling sign/summation moves the bridge and fails at once;
* its **symmetric** contrast (``u_B = +u_A``) — the bridge swings and energy floods into the body;
* **sympathetic transfer** — plucking one string rings up a second one far more when it is tuned to
  a partial of the first than when it is detuned;

plus a bit-identity to :class:`StringBodyBridge` for a single string, the K=0 decoupling, and the
exact dense leapfrog stability guard.
"""

import numpy as np
import pytest
from helpers import (
    BODY_FREQS_DEFAULT,
    K_BRIDGE_DEFAULT,
    make_sympathetic,
)

from physsynth.core.body import ModalBody
from physsynth.core.connection import StringBodyBridge, SympatheticStrings
from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_ideal import IdealString

DRIFT_TOL = 1e-10


def _pluck(symp, j=0, amplitude=1e-3):
    """Pluck string ``j`` of a :class:`SympatheticStrings` (others left at rest)."""
    s = symp.strings[j]
    s.set_state(triangular_pluck(s.x, s.L, 0.137 * s.L, amplitude=amplitude))
    return symp


# -- Criterion 1: TOTAL energy conserved (no single string's energy is) ----------------------
@pytest.mark.parametrize("lam", [0.9, 0.7, 0.5])
def test_total_energy_conserved_across_lambda(lam):
    symp = _pluck(make_sympathetic(lam=lam))
    res = simulate(symp, num_steps=int(1.5 * symp.strings[0].fs))
    assert res.energy_drift < DRIFT_TOL, f"total drift {res.energy_drift:.2e} at lam={lam}"


@pytest.mark.parametrize("n_strings", [2, 3, 4])
def test_total_energy_conserved_across_count(n_strings):
    symp = _pluck(make_sympathetic(n_strings=n_strings, lam=0.9, K=6000.0))
    res = simulate(symp, num_steps=int(1.0 * symp.strings[0].fs))
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} for {n_strings} strings"


# -- passivity: with loss anywhere, the total decreases monotonically ------------------------
def test_passivity_with_body_damping():
    symp = _pluck(make_sympathetic(lam=0.9, sigma_body=10.0))
    res = simulate(symp, num_steps=int(1.5 * symp.strings[0].fs))
    e0 = res.energy[0]
    assert np.all(np.diff(res.energy) <= 1e-12 * e0)


def test_passivity_with_string_damping():
    symp = _pluck(make_sympathetic(lam=0.9, sigma_string=5.0))
    res = simulate(symp, num_steps=int(1.5 * symp.strings[0].fs))
    e0 = res.energy[0]
    assert np.all(np.diff(res.energy) <= 1e-12 * e0)


# -- one string reproduces StringBodyBridge bit-for-bit --------------------------------------
def test_single_string_bit_identical_to_string_body_bridge():
    N, lam = 100, 0.9
    L, T, rho, c = 1.0, 200.0, 0.005, 200.0
    fs = c * N / (L * lam)
    K = K_BRIDGE_DEFAULT
    q0 = 1e-3 * np.array([1.0, -0.5, 0.7, 0.3])

    def build(cls_wrap):
        s = IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
        b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs, masses=0.02)
        s.set_state(triangular_pluck(s.x, L, 0.137 * L, amplitude=1e-3))
        b.set_state(q0)
        return cls_wrap(s, b), s, b

    ref, s_ref, b_ref = build(lambda s, b: StringBodyBridge(string=s, body=b, K=K))
    symp, s_sy, b_sy = build(lambda s, b: SympatheticStrings(strings=[s], body=b, Ks=[K]))

    for _ in range(500):
        ref.step()
        symp.step()
    assert np.array_equal(s_sy.u, s_ref.u), "single-string coupling not bit-identical to bridge"
    assert np.array_equal(b_sy.q, b_ref.q), "body not bit-identical to StringBodyBridge"


# -- K = 0: coupled system is bit-identical to the uncoupled parts ---------------------------
def test_K0_bit_identical_to_uncoupled_parts():
    N, lam = 100, 0.9
    L, T, rho, c = 1.0, 200.0, 0.005, 200.0
    fs = c * N / (L * lam)
    q0 = 1e-3 * np.array([1.0, -0.5, 0.7, 0.3])

    s0 = IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    s1 = IdealString(L=L, T=0.5 * T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs, masses=0.02)
    symp = SympatheticStrings(strings=[s0, s1], body=b, Ks=[0.0, 0.0])
    s0.set_state(triangular_pluck(s0.x, L, 0.137 * L, amplitude=1e-3))
    s1.set_state(triangular_pluck(s1.x, L, 0.29 * L, amplitude=7e-4))
    b.set_state(q0)

    # standalone references
    r0 = IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    r1 = IdealString(L=L, T=0.5 * T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
    rb = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs, masses=0.02)
    r0.set_state(triangular_pluck(r0.x, L, 0.137 * L, amplitude=1e-3))
    r1.set_state(triangular_pluck(r1.x, L, 0.29 * L, amplitude=7e-4))
    rb.set_state(q0)

    for _ in range(400):
        symp.step()
        r0.step()
        r1.step()
        rb.step()
    assert np.array_equal(s0.u, r0.u) and np.array_equal(s1.u, r1.u), "strings coupled at K=0"
    assert np.array_equal(b.q, rb.q), "body coupled at K=0"


# -- THE discriminating oracle: antisymmetric mode leaves the bridge exactly still -----------
def test_antisymmetric_mode_keeps_bridge_still():
    # Two identical strings, equal springs, body at rest, started u_B = -u_A. By symmetry the total
    # bridge force F_A + F_B = -2 K w_b, so a bridge at w_b = 0 feels zero force and stays there
    # forever, while u_B stays exactly -u_A. Energy conservation can't see this; a flipped sign can.
    symp = make_sympathetic(n_strings=2, lam=0.9, K=8000.0)
    a, bstr = symp.strings
    pluck = triangular_pluck(a.x, a.L, 0.137 * a.L, amplitude=1e-3)
    a.set_state(pluck)
    bstr.set_state(-pluck)  # exact antisymmetric IC (linear Taylor start negates too)

    e0 = a.energy() + bstr.energy()
    max_wb = 0.0
    max_ebody = 0.0
    for _ in range(4000):
        symp.step()
        max_wb = max(max_wb, abs(symp._bridge_displacement()))
        max_ebody = max(max_ebody, symp.body.energy())
        assert np.allclose(bstr.u, -a.u, rtol=0, atol=1e-15), "u_B drifted from -u_A"
    assert max_wb < 1e-13, f"bridge moved (max|w_b| = {max_wb:.2e}); coupling sign/sum wrong"
    assert max_ebody < 1e-13 * e0, f"body gained energy ({max_ebody:.2e}) from a still bridge"


# -- contrast: the symmetric mode DOES drive the bridge --------------------------------------
def test_symmetric_mode_drives_bridge():
    symp = make_sympathetic(n_strings=2, lam=0.9, K=8000.0)
    a, bstr = symp.strings
    pluck = triangular_pluck(a.x, a.L, 0.137 * a.L, amplitude=1e-3)
    a.set_state(pluck)
    bstr.set_state(pluck)  # symmetric: both ends push the bridge the same way

    e0 = a.energy() + bstr.energy()
    max_ebody = 0.0
    for _ in range(int(0.5 * a.fs)):
        symp.step()
        max_ebody = max(max_ebody, symp.body.energy())
    assert max_ebody > 0.05 * e0, f"symmetric mode barely moved the body ({max_ebody / e0:.1%})"


# -- sympathetic transfer: a tuned neighbour rings up far more than a detuned one ------------
def _peak_neighbour_fraction(*, Ts, K=1500.0, lam=0.9, secs=1.5):
    """Pluck string 0; return the peak fraction of the *total* energy ever held by string 1.

    A softer bridge (``K = 1500``) makes the coupling frequency-selective — the resonant transfer
    that is the whole point of sympathetic strings — so a tuned neighbour drains most of the energy
    while a detuned one barely responds.
    """
    symp = _pluck(make_sympathetic(n_strings=2, lam=lam, K=K, Ts=Ts))
    peak = 0.0
    for _ in range(int(secs * symp.strings[0].fs)):
        symp.step()
        peak = max(peak, symp.string_energy(1) / symp.energy())
    return peak


def test_sympathetic_transfer_tuned_beats_detuned():
    tuned = _peak_neighbour_fraction(Ts=[200.0, 200.0])       # unison -> near-complete exchange
    detuned = _peak_neighbour_fraction(Ts=[200.0, 120.0])     # ~4 semitones flat -> stays quiet
    assert tuned > 0.5, f"tuned unison neighbour should drain most of the energy ({tuned:.1%})"
    assert detuned < 0.25, f"detuned neighbour rang too much ({detuned:.1%})"
    assert tuned > 3.0 * detuned, f"tuned {tuned:.1%} not >> detuned {detuned:.1%}"


# -- the exact dense stability guard rejects an over-stiff spring ----------------------------
def test_unstable_stiffness_rejected():
    with pytest.raises(ValueError, match="unstable"):
        make_sympathetic(lam=0.9, K=1e6)  # far past the leapfrog bound


def test_guard_holds_at_its_boundary():
    # Two strings share the bridge, so each contributes to the coupled operator; a stiffness just
    # inside the guard must still conserve energy to machine precision.
    symp = _pluck(make_sympathetic(n_strings=2, lam=0.9, K=10000.0))
    assert symp.k * symp.k * symp.spectral_radius < 4.0
    res = simulate(symp, num_steps=int(1.0 * symp.strings[0].fs))
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} near the guard boundary"


# -- construction guards ---------------------------------------------------------------------
def test_mismatched_timestep_rejected():
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=22000.0, N=100, boundary=("fixed", "free"))
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=24000.0)
    with pytest.raises(ValueError, match="timestep"):
        SympatheticStrings(strings=[s], body=b, Ks=[K_BRIDGE_DEFAULT])


def test_right_end_must_be_free():
    fs = 22000.0
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=fs, N=100, boundary="fixed")
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=fs)
    with pytest.raises(ValueError, match="free"):
        SympatheticStrings(strings=[s], body=b, Ks=[K_BRIDGE_DEFAULT])


def test_ks_length_must_match():
    with pytest.raises(ValueError, match="one stiffness per string"):
        make_sympathetic(n_strings=2, K=np.array([8000.0]))


def test_empty_strings_rejected():
    b = ModalBody(freqs=BODY_FREQS_DEFAULT, fs=22000.0)
    with pytest.raises(ValueError, match="at least one string"):
        SympatheticStrings(strings=[], body=b, Ks=[])
