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
from helpers import (
    make_body,
    make_bridge,
    make_radiated_body,
    make_radiation,
)

from physsynth.core.body import ModalBody
from physsynth.core.connection import StringBodyBridge
from physsynth.core.radiation import (
    C0_AIR,
    RHO0_AIR,
    AirRadiation,
    RadiatedBody,
    monopole_radiation_resistance,
    piston_radiation_resistance,
)


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


# =============================================================================================
# Batch 2 — the radiation LOAD (back-reaction): a passive rank-1 dashpot on the body, with the
# radiated energy tracked as an explicit channel. The money test is the energy identity
# E_body + integral P_rad = const (lossless body), not a spectral match to any piston.
# =============================================================================================


# -- closed-form resistance oracle: Rayleigh (ka -> 0) limits, mechanical/acoustic units -------
def test_monopole_resistance_is_the_free_space_value():
    # R_a = rho0 omega^2 / (4 pi c0), acoustic (per volume velocity) units.
    omega = 2.0 * np.pi * 200.0
    r_a = monopole_radiation_resistance(omega)
    assert r_a == pytest.approx(RHO0_AIR * omega**2 / (4.0 * np.pi * C0_AIR), rel=1e-14)


def test_piston_rayleigh_limit_is_twice_the_free_space_monopole():
    # As ka -> 0 the baffled piston (half-space, 2 pi) tends to exactly twice the free-space (4 pi)
    # monopole: R_a(ka->0) -> rho0 omega^2 / (2 pi c0).
    omega, a = 2.0 * np.pi * 5.0, 1e-3  # ka = omega a / c0 ~ 9e-5, deep in the Rayleigh regime
    r_piston = piston_radiation_resistance(omega, a)
    r_mono = monopole_radiation_resistance(omega)
    assert r_piston == pytest.approx(2.0 * r_mono, rel=1e-6)
    assert r_piston == pytest.approx(RHO0_AIR * omega**2 / (2.0 * np.pi * C0_AIR), rel=1e-6)


def test_piston_resistance_matches_bessel_formula_away_from_the_limit():
    from scipy.special import j1

    omega, a = 2.0 * np.pi * 2000.0, 0.05  # ka ~ 1.8, well past the Rayleigh limit
    ka = omega * a / C0_AIR
    expected = RHO0_AIR * C0_AIR / (np.pi * a * a) * (1.0 - j1(2.0 * ka) / ka)
    assert piston_radiation_resistance(omega, a) == pytest.approx(expected, rel=1e-12)


# -- money test: E_body + integral P_rad is conserved for a lossless body ----------------------
def test_energy_channel_is_conserved_for_a_lossless_body():
    loaded = make_radiated_body(sigmas=0.0, R=2000.0)  # lossless modes: radiation is the only sink
    loaded.set_state(np.array([1e-3, -8e-4, 6e-4, 4e-4]))
    e0 = loaded.energy()
    peak = 0.0
    for _ in range(4000):
        loaded.step()
        peak = max(peak, abs(loaded.energy() - e0) / e0)
    assert peak < 1e-10  # E_body + radiated_energy is conserved to machine precision


def test_body_energy_bleeds_entirely_into_the_radiated_channel():
    loaded = make_radiated_body(sigmas=0.0, R=3000.0)
    loaded.set_state(np.full(4, 1e-3))
    e_body0 = loaded.body.energy()
    for _ in range(6000):
        loaded.step()
    # All the energy the body lost is now in the radiated channel (lossless modes).
    assert loaded.body.energy() < 0.2 * e_body0            # the body has genuinely rung down
    assert loaded.radiated_energy == pytest.approx(e_body0 - loaded.body.energy(), rel=1e-10)


# -- passivity: body energy monotonically decreases, radiated energy monotonically increases ---
def test_radiation_load_is_passive():
    loaded = make_radiated_body(sigmas=0.0, R=2000.0)
    loaded.set_state(np.array([1e-3, 5e-4, -7e-4, 2e-4]))
    body_e, rad_e = [], []
    for _ in range(3000):
        loaded.step()
        body_e.append(loaded.body.energy())
        rad_e.append(loaded.radiated_energy)
    body_e, rad_e = np.asarray(body_e), np.asarray(rad_e)
    # A small cross-time ripple is allowed; assert monotone to a tiny fraction of the start energy.
    tol = 1e-12 * body_e[0]
    assert np.all(np.diff(body_e) <= tol)   # body sheds energy every step
    assert np.all(np.diff(rad_e) >= -tol)   # the far field only ever gains


# -- R = 0 is bit-identical to a bare ModalBody ------------------------------------------------
def test_R_zero_is_bit_identical_to_a_bare_body():
    kw = dict(freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=48000.0, masses=0.02)
    plain = ModalBody(**kw)
    loaded = RadiatedBody(body=ModalBody(**kw), R=0.0)
    q0 = np.array([1e-3, -5e-4, 3e-4, 8e-4])
    plain.set_state(q0)
    loaded.set_state(q0)
    for _ in range(500):
        plain.step()
        loaded.step()
        assert np.array_equal(loaded.body.q, plain.q)       # bit-for-bit, not just close
        assert np.array_equal(loaded.body.q_prev, plain.q_prev)
        assert loaded.pressure() == plain.pressure()
    assert loaded.radiated_energy == 0.0


# -- unconditionally passive: no CFL, no guard, stable at an absurd R --------------------------
def test_unconditionally_stable_at_enormous_R():
    loaded = make_radiated_body(sigmas=0.0, R=1e12)  # far beyond any physical value; no guard
    loaded.set_state(np.full(4, 1e-3))
    e0 = loaded.energy()
    for _ in range(2000):
        loaded.step()
        assert np.all(np.isfinite(loaded.body.q))           # never blows up
    assert abs(loaded.energy() - e0) / e0 < 1e-10           # total still conserved
    assert loaded.body.energy() < e0                        # (critically) over-damped, not growing


# -- the loaded body radiates through the air node, carrying the back-reaction -----------------
def test_loaded_body_radiates_through_the_air():
    loaded = make_radiated_body(sigmas=0.0, R=2000.0)
    loaded.set_state(np.full(4, 1e-3))
    rad = make_radiation(fs=1.0 / loaded.k, retarded=False)
    for _ in range(1000):
        loaded.step()
        # pressure() reflects the corrected (post-load) acceleration; the air reads it exactly.
        assert rad.radiate(loaded) == pytest.approx(rad.gain * loaded.pressure(), rel=1e-12)


# -- full chain: string -> bridge -> RADIATED body conserves E_str+E_body+E_conn+integral P_rad -
def test_full_chain_with_radiation_conserves_the_total():
    # A lossless string and lossless body modes; the ONLY sink is the radiation channel, so the
    # bridge's own energy() (string + loaded-body.energy() + E_conn) must be conserved.
    bridge = make_bridge(sigma_string=0.0, sigma_body=0.0, K=8000.0)
    loaded = RadiatedBody(body=bridge.body, R=1500.0)
    chain = StringBodyBridge(string=bridge.string, body=loaded, K=8000.0)
    from physsynth.core.exciter import triangular_pluck

    s = chain.string
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    e0 = chain.energy()
    peak = 0.0
    for _ in range(6000):
        chain.step()
        peak = max(peak, abs(chain.energy() - e0) / abs(e0))
    assert peak < 1e-9                          # the four-way energy identity holds
    assert loaded.radiated_energy > 0.0         # the body genuinely radiated
    # The radiated energy is a real fraction of the total: the chain has audibly rung down into air.
    assert loaded.radiated_energy > 0.05 * e0


# -- the (1 + sigma k) factor: a LOSSY body must match the exact dense coupled implicit solve ---
def test_lossy_body_matches_the_exact_dense_coupled_solve():
    # The ``(1 + sigma k)`` factor in G and the correction is INVISIBLE at sigma = 0 (1 + 0 = 1), so
    # every other test (all sigma = 0) leaves it unpinned. This is the discriminating check: a lossy
    # body's loaded step must equal the exact dense coupled implicit solve
    #   [diag(1 + sigma k) + (k R / 2) (a/m) a^T] q^{n+1} = free_rhs + (k R / 2)(a . q^{n-1})(a/m),
    # whose denominator carries the body's true ``1 + sigma k``. A wrong-but-consistent factor (drop
    # it from BOTH G and corr) passes self-consistency and monotonicity but diverges from this
    # reference at ~1e-11 (verified), so atol = 1e-13 catches it.
    loaded = make_radiated_body(sigmas=3.0, masses=0.02, R=1500.0)
    loaded.set_state(
        np.array([1e-3, -6e-4, 4e-4, 7e-4]), v0=np.array([0.2, -0.1, 0.05, 0.0])
    )
    b = loaded.body
    k, a, m, om = b.k, b.a, b.m, b.omega
    sk = b.sigma * k
    for _ in range(20):
        q_n, q_nm1 = b.q.copy(), b.q_prev.copy()
        free_rhs = 2.0 * q_n - (1.0 - sk) * q_nm1 - k * k * om * om * q_n  # body.step numerator
        mmat = np.diag(1.0 + sk) + 0.5 * k * loaded.R * np.outer(a / m, a)
        rhs = free_rhs + 0.5 * k * loaded.R * float(np.dot(a, q_nm1)) * (a / m)
        q1_ref = np.linalg.solve(mmat, rhs)
        loaded.step()
        assert np.allclose(b.q, q1_ref, rtol=0, atol=1e-13)


def test_radiated_body_rejects_negative_R():
    with pytest.raises(ValueError):
        RadiatedBody(body=make_body(), R=-1.0)
