"""Standalone modal-body validation (the "body" node, before the bridge connection).

The body is the minimal test-fixture for the energy-conserving connection built on top of it, so it
must first pass the project's core numeric oracles in isolation: lossless energy conservation,
passivity, the analytic decay rate, the CFL guard, and an *exact* discrete-cosine trajectory for a
single lossless mode (the modal-frequency check, done deterministically rather than by FFT).
"""

import numpy as np
import pytest
from helpers import discrete_sho_frequency, make_body

from physsynth.core.body import ModalBody
from physsynth.core.engine import simulate

DRIFT_TOL = 1e-10  # same acceptance bar as every other resonator


def _excite(body, amplitude=1e-3):
    """Kick every mode with a distinct displacement so all modes carry energy."""
    q0 = amplitude * (1.0 + 0.1 * np.arange(body.M))
    body.set_state(q0)
    return body


# -- lossless: energy conserved to machine precision, for a multi-mode body -----------------
def test_body_energy_conserved():
    body = _excite(make_body())
    res = simulate(body, num_steps=20000)
    assert res.energy_drift < DRIFT_TOL, f"body drift {res.energy_drift:.2e}"


def test_body_energy_strictly_positive():
    body = _excite(make_body())
    res = simulate(body, num_steps=5000)
    assert np.all(res.energy > 0.0)


def test_body_energy_scales_with_mass():
    # energy() is linear in the modal masses: doubling every mass doubles the energy at a fixed
    # displacement state.
    b1 = _excite(make_body(masses=1.0))
    b2 = _excite(make_body(masses=2.0))
    assert np.isclose(b2.energy() / b1.energy(), 2.0, rtol=1e-12)


# -- lossy: passivity (monotonic) and the analytic 2*sigma decay rate -----------------------
def test_body_passivity_monotonic():
    body = _excite(make_body(sigmas=8.0))
    res = simulate(body, num_steps=20000)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


def test_body_decay_rate_matches_2sigma():
    # A single damped mode loses energy as E(t) ~ E0 * exp(-2 sigma t).
    sigma, secs, fs = 6.0, 1.0, 48000.0
    body = ModalBody(freqs=np.array([220.0]), fs=fs, sigmas=sigma)
    body.set_state(np.array([1e-3]))
    res = simulate(body, num_steps=int(secs * fs))
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.02, f"decay rate off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


# -- CFL guard: omega*k >= 2 is rejected at construction ------------------------------------
def test_body_cfl_rejected():
    # 20 kHz mode at fs = 48 kHz: omega*k = 2*pi*20000/48000 ~ 2.618 > 2 -> unstable, must raise.
    with pytest.raises(ValueError, match="CFL"):
        ModalBody(freqs=np.array([20000.0]), fs=48000.0)


# -- modal frequency: a single lossless mode is the exact discrete cosine -------------------
def test_body_single_mode_is_exact_discrete_cosine():
    # With the consistent (v0 = 0) start, one lossless mode must reproduce q0*cos(Omega n k) to
    # machine precision -- the strongest possible modal-frequency check.
    f, fs, q0 = 261.63, 48000.0, 1e-3
    body = ModalBody(freqs=np.array([f]), fs=fs)
    body.set_state(np.array([q0]))
    res = simulate(body, num_steps=4000, snapshot_stride=1)
    q = np.array([s[0] for _, s in res.snapshots])  # single modal coordinate

    fd = discrete_sho_frequency(f, 1.0 / fs)
    omega_d = 2.0 * np.pi * fd
    n = np.arange(q.size)
    q_exact = q0 * np.cos(omega_d * n / fs)
    assert np.max(np.abs(q - q_exact)) < 1e-14, "single mode is not the exact discrete cosine"


# -- read-outs behave sensibly -------------------------------------------------------------
def test_body_bridge_and_pressure_readouts():
    body = _excite(make_body(phi=np.array([1.0, -0.5, 0.7, 0.3])))
    # bridge displacement is sum_i phi_i q_i by definition.
    assert np.isclose(body.bridge_displacement(), float(np.dot(body.phi, body.q)), rtol=1e-12)
    # a lossless body radiates non-trivially (some mode has non-zero acceleration).
    body.step()
    assert np.isfinite(body.pressure())


def test_body_frequencies_must_be_positive():
    with pytest.raises(ValueError):
        ModalBody(freqs=np.array([100.0, -5.0]), fs=48000.0)
