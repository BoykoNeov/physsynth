"""Validation for the far-field radiation node (:class:`physsynth.core.radiation.AirRadiation`).

Batch 1 is a pure, passive output transform (no back-reaction, so no energy channel yet), so its
correctness is asserted against the closed-form free-space **monopole** solution rather than an
energy identity:

  * amplitude/gain oracle: ``p_far = rho0 / (4 pi r) * Q''`` exactly, and for a prescribed
    sinusoidal volume velocity ``U0 sin(omega t)`` the amplitude is ``rho0 omega U0 / (4 pi r)``;
  * inverse-distance law ``p ∝ 1 / r``;
  * retardation is an exact integer-sample delay (amplitude preserved, wavefront in transit =
    silence for ``r / c0`` seconds);
  * linearity/superposition of the transform;
  * end-to-end: a real ModalBody and a full string->bridge->body chain radiate a finite, non-trivial
    pressure equal to the gain times the body's volume acceleration.
"""

import numpy as np
import pytest
from helpers import make_body, make_bridge, make_radiation

from physsynth.core.radiation import C0_AIR, RHO0_AIR, AirRadiation


# -- amplitude / gain oracle: p_far = rho0/(4 pi r) * Q'' exactly ----------------------------
def test_far_field_gain_is_exact():
    r, fs = 2.0, 48000.0
    rad = AirRadiation(fs=fs, distance=r, retarded=False)
    gain = RHO0_AIR / (4.0 * np.pi * r)
    assert rad.gain == pytest.approx(gain, rel=0, abs=0.0)
    qdd = np.array([0.0, 1.0, -3.5, 42.0, -1e-4, 7.0])
    out = np.array([rad.process(v) for v in qdd])
    assert np.allclose(out, gain * qdd, rtol=0, atol=1e-15)


# -- monopole oracle: prescribed volume velocity U0 sin(wt) -> |p| = rho0 w U0 /(4 pi r) -----
def test_monopole_amplitude_from_volume_velocity():
    r, fs, f = 1.5, 48000.0, 220.0
    U0 = 3e-4  # m^3/s volume-velocity amplitude
    rad = AirRadiation(fs=fs, distance=r, retarded=False)
    n = np.arange(int(4 * fs / f))  # a few periods
    t = n / fs
    omega = 2.0 * np.pi * f
    # U(t) = U0 sin(wt) -> volume acceleration Q'' = U' = U0 w cos(wt).
    qdd = U0 * omega * np.cos(omega * t)
    p = np.array([rad.process(v) for v in qdd])
    expected_amp = RHO0_AIR * omega * U0 / (4.0 * np.pi * r)
    # Skip the first sample; peak of a sampled cosine matches the analytic amplitude to grid.
    assert np.max(np.abs(p)) == pytest.approx(expected_amp, rel=1e-6)


# -- inverse-distance law: p ∝ 1/r ----------------------------------------------------------
def test_inverse_distance_law():
    fs = 48000.0
    qdd = 5.0
    p1 = AirRadiation(fs=fs, distance=1.0, retarded=False).process(qdd)
    p2 = AirRadiation(fs=fs, distance=2.0, retarded=False).process(qdd)
    p4 = AirRadiation(fs=fs, distance=4.0, retarded=False).process(qdd)
    assert p2 == pytest.approx(p1 / 2.0, rel=1e-12)
    assert p4 == pytest.approx(p1 / 4.0, rel=1e-12)


# -- retardation: exact integer-sample delay, amplitude preserved ---------------------------
def test_retardation_is_an_exact_sample_delay():
    r, fs = 3.43, 48000.0  # r/c0 = 0.01 s -> exactly 480 samples at 343 m/s
    rad = AirRadiation(fs=fs, distance=r)
    expected_delay = int(round(r / C0_AIR * fs))
    assert rad.latency_samples == expected_delay
    assert abs(rad.retardation_residual) <= 0.5
    # An impulse in emerges undistorted, delayed by exactly latency_samples.
    n_steps = expected_delay + 5
    out = np.array([rad.process(1.0 if i == 0 else 0.0) for i in range(n_steps)])
    assert np.count_nonzero(out) == 1
    peak = int(np.argmax(np.abs(out)))
    assert peak == expected_delay
    assert out[peak] == pytest.approx(rad.gain, rel=1e-12)  # amplitude preserved exactly


def test_wavefront_in_transit_is_silence():
    r, fs = 3.43, 48000.0
    rad = AirRadiation(fs=fs, distance=r)
    # Constant drive; the listener hears nothing until the wavefront arrives.
    out = [rad.process(1.0) for _ in range(rad.latency_samples)]
    assert np.allclose(out, 0.0)
    assert rad.process(1.0) == pytest.approx(rad.gain, rel=1e-12)  # first arrival


def test_retarded_false_has_no_delay():
    rad = AirRadiation(fs=48000.0, distance=5.0, retarded=False)
    assert rad.latency_samples == 0
    assert rad.retardation_residual == 0.0
    assert rad.process(2.0) == pytest.approx(rad.gain * 2.0, rel=1e-12)


# -- linearity / superposition of the transform ---------------------------------------------
def test_linearity():
    fs = 48000.0
    a = np.array([1.0, -2.0, 0.5, 3.0, -0.25])
    b = np.array([0.3, 0.3, -1.0, 2.0, 4.0])
    ra = AirRadiation(fs=fs, distance=1.0, retarded=False)
    rb = AirRadiation(fs=fs, distance=1.0, retarded=False)
    rab = AirRadiation(fs=fs, distance=1.0, retarded=False)
    out_a = np.array([ra.process(v) for v in a])
    out_b = np.array([rb.process(v) for v in b])
    out_ab = np.array([rab.process(v) for v in (a + b)])
    assert np.allclose(out_ab, out_a + out_b, rtol=0, atol=1e-15)


# -- end-to-end: a real ModalBody radiates ---------------------------------------------------
def test_radiates_a_real_modal_body():
    body = make_body()
    body.set_state(np.full(body.M, 1e-3))
    fs = 1.0 / body.k
    rad = make_radiation(fs=fs, retarded=False)
    # p_far each step must equal the gain times the body's volume acceleration (pressure()).
    peaks = []
    for _ in range(2000):
        body.step()
        p = rad.radiate(body)
        assert p == pytest.approx(rad.gain * body.pressure(), rel=1e-12)
        peaks.append(abs(p))
    assert max(peaks) > 0.0  # the body genuinely radiates


def test_full_chain_radiates_with_retardation():
    # string -> bridge -> modal body -> air: the full instrument chain producing radiated sound.
    bridge = make_bridge()
    from physsynth.core.exciter import triangular_pluck

    s = bridge.string
    bridge.string.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    fs = 1.0 / bridge.k
    rad = make_radiation(fs=fs)  # retarded default
    p = np.empty(4000)
    for i in range(4000):
        bridge.step()
        p[i] = rad.radiate(bridge)
    assert np.all(np.isfinite(p))
    # Silence until the wavefront arrives, then a non-trivial radiated signal.
    assert np.allclose(p[: rad.latency_samples], 0.0)
    assert np.max(np.abs(p[rad.latency_samples:])) > 0.0


def test_reset_clears_the_delay_line():
    rad = make_radiation(fs=48000.0)
    for _ in range(50):
        rad.process(1.0)
    rad.reset()
    assert rad.n == 0
    assert np.allclose(rad._buf, 0.0)
    # Post-reset, the first output is silence again (delay line empty).
    assert rad.process(1.0) == pytest.approx(0.0)


# -- construction validation ----------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"fs": 0.0},
        {"fs": -48000.0},
        {"fs": 48000.0, "distance": 0.0},
        {"fs": 48000.0, "distance": -1.0},
        {"fs": 48000.0, "rho0": 0.0},
        {"fs": 48000.0, "c0": 0.0},
    ],
)
def test_rejects_nonphysical_parameters(kwargs):
    with pytest.raises(ValueError):
        AirRadiation(**kwargs)
