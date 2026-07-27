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
    SPHERE_RADIUS_DEFAULT,
    make_body,
    make_bridge,
    make_radiated_body,
    make_radiation,
    make_reactive_body,
)

from physsynth.core.body import ModalBody
from physsynth.core.connection import StringBodyBridge
from physsynth.core.radiation import (
    C0_AIR,
    RHO0_AIR,
    AirRadiation,
    RadiatedBody,
    RationalAirLoad,
    ReactiveRadiatedBody,
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


# ============================================================================================
# Batch 3 — the frequency-dependent (rational) radiation impedance Z_a(omega).
#
# Batch 2 loaded the body with one constant R; this is the full first-order impedance, resistance
# in PARALLEL with the radiation mass M_a, which is the *exact* pulsating-sphere load (no filter
# approximation involved). The oracles come in three tiers:
#   * structural  — the energy identity now has a STORED term (the air can give back) and the two
#                   exact reductions (R = 0 -> bare body, M_a = inf -> batch 2), bit-for-bit;
#   * spectral    — a measured impedance sweep against the PRE-WARPED closed form (trapezoid is the
#                   bilinear transform, so the scheme realises Z_a at s = (2j/k)tan(omega k/2), not
#                   at j omega), plus both closed-form limits;
#   * physical    — per-mode decay rates matching alpha_i = a_i^2 Re Z_a(omega_i)/(2 m_i), with
#                   batch 2's constant R as the negative control (it cannot bend with frequency).
# ============================================================================================


def _drive_and_measure_impedance(load, omega, *, n_window, warmup=8000):
    """Drive a load standalone at prescribed volume velocity and return the measured ``p / U``.

    ``G = 0`` is the prescribed-velocity drive: a rigid piston whose ``U`` is imposed regardless of
    the load's reaction. ``omega`` must land exactly on a DFT bin of the ``n_window``-sample window
    so the ratio is exact; ``warmup`` lets the first-order transient (pole ``(1-b)/(1+b)``) die.
    """
    k = load.k
    n = 0
    for _ in range(warmup):
        load.step(np.cos(omega * n * k))
        n += 1
    u = np.empty(n_window)
    p = np.empty(n_window)
    for i in range(n_window):
        u[i] = np.cos(omega * n * k)
        p[i], _ = load.step(u[i])
        n += 1
    # Single-bin DFT at the drive frequency (the window holds an integer number of periods).
    phase = np.exp(-1j * omega * k * np.arange(n_window))
    return complex(np.sum(p * phase) / np.sum(u * phase))


# -- the impedance IS the pulsating sphere's, in closed form -----------------------------------
def test_from_sphere_impedance_is_the_closed_form_monopole():
    a, fs = 0.05, 48000.0
    load = RationalAirLoad.from_sphere(fs=fs, radius=a)
    S = 4.0 * np.pi * a * a
    assert load.R == pytest.approx(RHO0_AIR * C0_AIR / S, rel=1e-14)
    assert load.M_a == pytest.approx(RHO0_AIR / (4.0 * np.pi * a), rel=1e-14)
    assert load.tau == pytest.approx(a / C0_AIR, rel=1e-14)
    for f in (50.0, 400.0, 3000.0):
        omega = 2.0 * np.pi * f
        ka = omega * a / C0_AIR
        expected = (RHO0_AIR * C0_AIR / S) * (1j * ka) / (1.0 + 1j * ka)
        assert load.impedance(omega) == pytest.approx(expected, rel=1e-13)


# -- the two closed-form limits: batch 2's helper below, plane-wave saturation above ------------
def test_low_frequency_limit_is_the_batch2_monopole_resistance():
    # As ka -> 0 the rational impedance's REAL part becomes exactly rho0 omega^2 / (4 pi c0) — the
    # constant-R batch's own helper. The two batches meet here; that is the continuity check.
    a = 0.05
    load = RationalAirLoad.from_sphere(fs=48000.0, radius=a)
    for f in (5.0, 20.0, 80.0):
        omega = 2.0 * np.pi * f
        ka = omega * a / C0_AIR
        # The exact relation, not just the limit: Re Z_a = R_mono / (1 + (ka)^2). The batch-2
        # helper is the numerator, so the two agree to O((ka)^2) and the gap is knowable, not fuzzy.
        assert load.impedance(omega).real == pytest.approx(
            monopole_radiation_resistance(omega) / (1.0 + ka * ka), rel=1e-13
        )
        assert load.impedance(omega).real == pytest.approx(
            monopole_radiation_resistance(omega), rel=2.0 * ka * ka
        )
    # ...and the agreement improves as ka shrinks (it is a limit, not a coincidence).
    err = [
        abs(load.impedance(2.0 * np.pi * f).real / monopole_radiation_resistance(2.0 * np.pi * f)
            - 1.0)
        for f in (80.0, 20.0, 5.0)
    ]
    assert err[0] > err[1] > err[2]


def test_high_frequency_limit_saturates_at_the_plane_wave_resistance():
    a = 0.05
    load = RationalAirLoad.from_sphere(fs=48000.0, radius=a)
    plane_wave = RHO0_AIR * C0_AIR / (4.0 * np.pi * a * a)
    omega = 2.0 * np.pi * 1e6
    ka = omega * a / C0_AIR
    # Exactly, again: Re Z = R (ka)^2/(1 + (ka)^2), so the approach to the plane-wave value is
    # 1/(ka)^2 from below and the reactance dies as 1/(ka).
    assert load.impedance(omega).real == pytest.approx(plane_wave / (1.0 + 1.0 / (ka * ka)),
                                                       rel=1e-13)
    assert load.impedance(omega).real == pytest.approx(plane_wave, rel=2.0 / (ka * ka))
    assert load.impedance(omega).imag == pytest.approx(plane_wave / ka, rel=1e-5)


def test_constant_R_load_has_a_flat_impedance():
    load = RationalAirLoad(fs=48000.0, R=2000.0)  # M_a = inf: batch 2's approximation, stated
    for f in (10.0, 1000.0, 10000.0):
        assert load.impedance(2.0 * np.pi * f) == 2000.0 + 0j
        assert load.impedance_discrete(2.0 * np.pi * f) == 2000.0 + 0j


# -- MONEY (spectral): the measured sweep matches the PRE-WARPED closed form ---------------------
def test_impedance_sweep_matches_the_prewarped_closed_form():
    # The trapezoid on the inertance IS the bilinear transform, so the scheme realises Z_a at
    # s = (2j/k) tan(omega k / 2). Driven standalone at prescribed U (G = 0), the measured p/U must
    # match that pre-warped value to machine precision -- magnitude AND phase.
    fs, n_window = 48000.0, 4096
    load = RationalAirLoad.from_sphere(fs=fs, radius=0.05)
    for bin_index in (13, 137, 613, 1361):
        omega = 2.0 * np.pi * fs * bin_index / n_window
        load.reset()
        measured = _drive_and_measure_impedance(load, omega, n_window=n_window)
        assert measured == pytest.approx(load.impedance_discrete(omega), rel=1e-11)


def test_the_prewarping_gap_is_the_honest_discretisation_error():
    # The gap between the scheme's impedance and the physics is real and worth naming: it is the
    # bilinear frequency warp, O((omega k)^2), and it vanishes at second order as k -> 0. (Comparing
    # a measured sweep to the CONTINUOUS Z_a shows exactly this and looks like a scheme bug.)
    omega = 2.0 * np.pi * 2000.0
    gaps = []
    for fs in (48000.0, 96000.0, 192000.0):
        load = RationalAirLoad.from_sphere(fs=fs, radius=0.05)
        z_c = load.impedance(omega)
        gaps.append(abs(load.impedance_discrete(omega) - z_c) / abs(z_c))
    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    for r in ratios:
        assert 3.5 < r < 4.5          # second order: halving k quarters the gap
    # The measured number, pinned rather than hand-waved: 2 kHz at 48 kHz costs ~0.27%.
    assert 2.0e-3 < gaps[0] < 3.5e-3


# -- MONEY (structural): the three-way energy identity, with a STORED air term -------------------
@pytest.mark.parametrize("R,M_a", [(2000.0, 0.2), (2000.0, 0.02), (5e4, 1.0), (13146.0, 1.916)])
def test_reactive_energy_identity_is_conserved_for_a_lossless_body(R, M_a):
    # E_body + 1/2 M_a U_L^2 + integral R U_R^2 dt = const. The middle term is what is new: batch
    # 2's air could only take, this one also gives back, so a wrong reactance shows up as drift.
    loaded = make_reactive_body(sigmas=0.0, R=R, M_a=M_a)
    loaded.set_state(np.array([1e-3, -8e-4, 6e-4, 4e-4]))
    e0 = loaded.energy()
    peak = 0.0
    for _ in range(4000):
        loaded.step()
        peak = max(peak, abs(loaded.energy() - e0) / e0)
    assert peak < 1e-10
    assert loaded.radiated_energy > 0.0            # the far field genuinely received energy
    assert loaded.load.stored_energy() > 0.0       # and the radiation mass genuinely stores some


def test_the_air_stores_as_well_as_dissipates():
    # The distinguishing structural fact vs batch 2: the stored term is non-trivially two-way. It
    # must both rise and fall over a run (a purely dissipative channel can only rise).
    loaded = make_reactive_body(sigmas=0.0, R=2000.0, M_a=0.02)
    loaded.set_state(np.full(4, 1e-3))
    stored = []
    for _ in range(3000):
        loaded.step()
        stored.append(loaded.load.stored_energy())
    d = np.diff(np.asarray(stored))
    assert np.any(d > 0) and np.any(d < 0)         # the air hands energy back, repeatedly
    assert np.count_nonzero(d > 0) > 0.2 * d.size  # and not just as a rounding-level ripple


def test_reactive_load_is_passive():
    loaded = make_reactive_body(sigmas=0.0, R=2000.0, M_a=0.05)
    loaded.set_state(np.array([1e-3, 5e-4, -7e-4, 2e-4]))
    total, rad = [], []
    for _ in range(3000):
        loaded.step()
        total.append(loaded.body.energy() + loaded.load.stored_energy())
        rad.append(loaded.radiated_energy)
    total, rad = np.asarray(total), np.asarray(rad)
    tol = 1e-12 * total[0]
    assert np.all(np.diff(total) <= tol)   # body + stored air sheds energy every step
    assert np.all(np.diff(rad) >= -tol)    # the far field only ever gains


def test_unconditionally_stable_at_extreme_parameters():
    # 1 + R_eff G >= 1 for every R >= 0, M_a > 0, k: no CFL, no guard, nothing to tune.
    for R, M_a in ((1e12, 1e-9), (1e12, 1e9), (1e-9, 1e-9)):
        loaded = make_reactive_body(sigmas=0.0, R=R, M_a=M_a)
        loaded.set_state(np.full(4, 1e-3))
        e0 = loaded.energy()
        for _ in range(2000):
            loaded.step()
            assert np.all(np.isfinite(loaded.body.q))
        assert abs(loaded.energy() - e0) / e0 < 1e-10
        assert loaded.body.energy() <= e0 * (1.0 + 1e-10)


# -- the two exact reductions, bit-for-bit ------------------------------------------------------
def test_infinite_radiation_mass_is_bit_identical_to_the_constant_R_load():
    # M_a -> inf at fixed R IS batch 2 (the reduction a sphere radius cannot express, which is why
    # the API takes (R, M_a) and not a radius). In IEEE arithmetic k R / (2 inf) is 0.0 and p / inf
    # is 0.0, so R_eff = R exactly and the auxiliary state stays exactly zero -- provided the
    # operation order matches RadiatedBody.step term for term. Equality, not allclose.
    kw = dict(freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=48000.0, masses=0.02)
    R = 2000.0
    batch2 = RadiatedBody(body=ModalBody(**kw), R=R)
    batch3 = ReactiveRadiatedBody(
        body=ModalBody(**kw), load=RationalAirLoad(fs=kw["fs"], R=R, M_a=np.inf)
    )
    q0 = np.array([1e-3, -5e-4, 3e-4, 8e-4])
    batch2.set_state(q0)
    batch3.set_state(q0)
    for _ in range(500):
        batch2.step()
        batch3.step()
        assert np.array_equal(batch3.body.q, batch2.body.q)
        assert np.array_equal(batch3.body.q_prev, batch2.body.q_prev)
        assert batch3.pressure() == batch2.pressure()
        assert batch3.radiated_energy == batch2.radiated_energy
        assert batch3.energy() == batch2.energy()
    assert batch3.load.stored_energy() == 0.0      # inf * 0.0 would be NaN; it is special-cased


def test_R_zero_with_reactance_is_bit_identical_to_a_bare_body():
    kw = dict(freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=48000.0, masses=0.02)
    plain = ModalBody(**kw)
    loaded = ReactiveRadiatedBody(
        body=ModalBody(**kw), load=RationalAirLoad(fs=kw["fs"], R=0.0, M_a=0.2)
    )
    q0 = np.array([1e-3, -5e-4, 3e-4, 8e-4])
    plain.set_state(q0)
    loaded.set_state(q0)
    for _ in range(500):
        plain.step()
        loaded.step()
        assert np.array_equal(loaded.body.q, plain.q)
        assert loaded.pressure() == plain.pressure()
    assert loaded.energy() == plain.energy()       # no radiated energy, no stored energy


# -- MONEY (physical): the loaded mode -- reactance shifts the pitch, resistance damps it ---------
def _measure_single_mode(f0, *, load, weight, mass, fs, seconds=0.6):
    """Excite ONE mode of a loaded body and measure its frequency and its decay rate.

    Frequency from zero crossings; decay from a straight-line fit through the log of the
    *envelope peaks* (fitting the log of the modal energy instead biases the rate badly, because
    the loaded mode no longer oscillates at the bare ``omega`` that expression assumes)."""
    b = ModalBody(freqs=np.array([f0]), fs=fs, sigmas=0.0, masses=mass, radiation=weight)
    loaded = ReactiveRadiatedBody(body=b, load=load)
    loaded.set_state(np.array([1e-3]))
    steps = int(seconds * fs)
    q = np.empty(steps)
    for n in range(steps):
        loaded.step()
        q[n] = b.q[0]
    sign = np.signbit(q)
    crossings = np.flatnonzero(sign[:-1] != sign[1:])
    f_meas = 0.5 * fs * (crossings.size - 1) / (crossings[-1] - crossings[0])
    env = np.abs(q)
    peaks = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:])) + 1
    t = np.arange(steps) / fs
    alpha = -float(np.polyfit(t[peaks], np.log(env[peaks]), 1)[0])
    return f_meas, alpha


@pytest.mark.parametrize("f0", [110.0, 196.0, 261.0, 440.0])
def test_a_loaded_mode_matches_the_closed_form_in_both_parts_of_Z(f0):
    # THE physics claim of this batch, and it needs BOTH parts of the impedance:
    #   * Im Z_a is an added mass -- the air makes the body heavier and its pitch DROPS;
    #   * Re Z_a damps it, at the shifted frequency: alpha = a^2 Re Z_a / (2 m_eff).
    # A constant-R load has Im Z = 0 and so cannot produce the pitch drop at all (next test).
    a_w, mass, fs = 0.02, 0.02, 48000.0
    load = RationalAirLoad.from_sphere(fs=fs, radius=SPHERE_RADIUS_DEFAULT)
    f_meas, alpha = _measure_single_mode(f0, load=load, weight=a_w, mass=mass, fs=fs)
    w_pred, a_pred = load.loaded_mode(2.0 * np.pi * f0, weight=a_w, mass=mass)
    assert f_meas == pytest.approx(w_pred / (2.0 * np.pi), rel=1e-3)   # the pitch drop, measured
    assert alpha == pytest.approx(a_pred, rel=0.02)                    # the damping, measured
    assert f_meas < 0.995 * f0                                         # air loading really flattens
    assert alpha / (2.0 * np.pi * f_meas) < 0.02                       # ...and stays weakly loaded


def test_higher_partials_radiate_better_and_die_first():
    # The spectral shaping, end to end: sweep the modes and watch the decay rate climb with
    # frequency the way Re Z_a(omega) says it must.
    a_w, mass, fs = 0.02, 0.02, 48000.0
    load = RationalAirLoad.from_sphere(fs=fs, radius=SPHERE_RADIUS_DEFAULT)
    freqs = np.array([110.0, 220.0, 440.0, 880.0])
    alpha = np.array([
        _measure_single_mode(f, load=load, weight=a_w, mass=mass, fs=fs)[1] for f in freqs
    ])
    assert np.all(np.diff(alpha) > 0.0)              # monotone in frequency
    assert alpha[-1] / alpha[0] > 20.0               # and by a wide margin over three octaves
    predicted = np.array([
        load.loaded_mode(2.0 * np.pi * f, weight=a_w, mass=mass)[1] for f in freqs
    ])
    assert np.allclose(alpha, predicted, rtol=0.02)


def test_a_constant_R_load_cannot_bend_with_frequency():
    # The negative control, and the reason this batch exists. Batch 2's load, matched exactly at
    # the fundamental, damps EVERY mode at that same rate -- the frequency dependence is simply
    # absent -- and it cannot shift the pitch either, because a real R has no reactance.
    a_w, mass, fs = 0.02, 0.02, 48000.0
    sphere = RationalAirLoad.from_sphere(fs=fs, radius=SPHERE_RADIUS_DEFAULT)
    freqs = np.array([110.0, 220.0, 440.0, 880.0])
    r_const = sphere.impedance(2.0 * np.pi * freqs[0]).real
    flat = RationalAirLoad(fs=fs, R=r_const, M_a=np.inf)   # = RadiatedBody(R=r_const), bit for bit
    measured = [_measure_single_mode(f, load=flat, weight=a_w, mass=mass, fs=fs) for f in freqs]
    alpha = np.array([m[1] for m in measured])
    assert np.allclose(alpha, alpha[0], rtol=0.02)                 # flat: no spectral shaping
    for f, (f_meas, _) in zip(freqs, measured, strict=True):
        assert f_meas == pytest.approx(f, rel=2e-3)                # and no pitch drop: Im Z = 0
    top_true = sphere.loaded_mode(2.0 * np.pi * freqs[-1], weight=a_w, mass=mass)[1]
    assert alpha[-1] < 0.1 * top_true              # under-damps the top mode by more than 10x


# -- MONEY (calibration): the booked radiated power crosses a far-field sphere -------------------
def test_far_field_power_balances_the_booked_radiated_power():
    # The load books R U_R^2 as gone; a sphere of radius r around it must carry exactly that power.
    # With p_far = (a/r) p_load the balance is EXACT at every ka, because S |Z_a|^2/(rho0 c0) is
    # identically Re Z_a -- so this pins the far-field geometry constants (a/r, S, the 4 pi), which
    # no energy identity can see.
    fs, n_window, a, r = 48000.0, 4096, 0.05, 3.0
    load = RationalAirLoad.from_sphere(fs=fs, radius=a)
    omega = 2.0 * np.pi * fs * 137 / n_window
    k = load.k
    n = 0
    for _ in range(8000):                                    # warm past the transient
        load.step(np.cos(omega * n * k))
        n += 1
    e0, far = load.radiated_energy, 0.0
    for _ in range(n_window):
        load.step(np.cos(omega * n * k))
        p_far = load.far_field_pressure(r)
        far += 4.0 * np.pi * r * r * p_far * p_far / (RHO0_AIR * C0_AIR) * k
        n += 1
    assert far == pytest.approx(load.radiated_energy - e0, rel=1e-12)


def test_the_compact_read_out_overstates_the_far_field_of_a_finite_sphere():
    # The trap, made explicit. AirRadiation is the a -> 0 compact limit; a finite sphere's far field
    # carries an extra 1/(1 + j k a), so a power balance against batch 1 misses by 1 + (ka)^2 --
    # here a factor of ~5.8. It is not a bug in either node; they are different limits.
    fs, n_window, a, r = 48000.0, 4096, 0.15, 3.0
    load = RationalAirLoad.from_sphere(fs=fs, radius=a)
    air = AirRadiation(fs=fs, distance=r, retarded=False)
    omega = 2.0 * np.pi * fs * 68 / n_window                 # ~797 Hz -> ka ~ 2.2
    ka = omega * a / C0_AIR
    k = load.k
    n = 0
    for _ in range(8000):
        load.step(np.cos(omega * n * k))
        n += 1
    sphere_ms, compact_ms = 0.0, 0.0
    for _ in range(n_window):
        load.step(np.cos(omega * n * k))
        # Compact source: the same monopole read out as rho0 Q'' / (4 pi r), Q'' = dU/dt exactly.
        compact = air.process(-omega * np.sin(omega * n * k))
        sphere_ms += load.far_field_pressure(r) ** 2
        compact_ms += compact * compact
        n += 1
    assert compact_ms / sphere_ms == pytest.approx(1.0 + ka * ka, rel=0.02)


# -- the (1 + sigma k) factor AND R_eff's k-dependence, pinned against the exact dense solve ------
@pytest.mark.parametrize("fs", [48000.0, 32000.0])
def test_lossy_reactive_body_matches_the_exact_dense_coupled_solve(fs):
    # Batch 2's debrief catch, with a second unpinned factor. All the sigma = 0 tests leave the
    # (1 + sigma k) in G/corr free, and every single-k test leaves R_eff = R/(1 + kR/(2 M_a)) free
    # -- a wrong-but-consistent factor still conserves energy and still decays monotonically. The
    # discriminating reference is the exact dense coupled implicit solve
    #   [diag(1+sk) + (k R_eff/2)(a/m) a^T] q^{n+1}
    #        = free_rhs + (k R_eff/2)(a . q^{n-1})(a/m) + k^2 R_eff L^- (a/m),
    # run at TWO timesteps so R_eff's k-dependence has to be right at both.
    load = RationalAirLoad(fs=fs, R=1500.0, M_a=0.03)     # finite M_a: R_eff is ~35% below R
    loaded = ReactiveRadiatedBody(
        body=ModalBody(
            freqs=np.array([110.0, 196.0, 261.0, 440.0]), fs=fs, sigmas=3.0, masses=0.02
        ),
        load=load,
    )
    loaded.set_state(np.array([1e-3, -6e-4, 4e-4, 7e-4]), v0=np.array([0.2, -0.1, 0.05, 0.0]))
    b = loaded.body
    k, a, m, om = b.k, b.a, b.m, b.omega
    sk = b.sigma * k
    assert load.R_eff < 0.75 * load.R                      # the reactance really is in play
    for _ in range(20):
        q_n, q_nm1 = b.q.copy(), b.q_prev.copy()
        free_rhs = 2.0 * q_n - (1.0 - sk) * q_nm1 - k * k * om * om * q_n   # body.step numerator
        mmat = np.diag(1.0 + sk) + 0.5 * k * load.R_eff * np.outer(a / m, a)
        rhs = (
            free_rhs
            + 0.5 * k * load.R_eff * float(np.dot(a, q_nm1)) * (a / m)
            + k * k * load.R_eff * load.u_l * (a / m)
        )
        q1_ref = np.linalg.solve(mmat, rhs)
        loaded.step()
        assert np.allclose(b.q, q1_ref, rtol=0, atol=1e-13)


# -- full chain: string -> bridge -> REACTIVELY radiated body, with zero bridge edits -------------
def test_full_chain_with_reactive_radiation_conserves_the_total():
    bridge = make_bridge(sigma_string=0.0, sigma_body=0.0, K=8000.0)
    load = RationalAirLoad(fs=1.0 / bridge.body.k, R=1500.0, M_a=0.05)
    loaded = ReactiveRadiatedBody(body=bridge.body, load=load)
    chain = StringBodyBridge(string=bridge.string, body=loaded, K=8000.0)
    from physsynth.core.exciter import triangular_pluck

    s = chain.string
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    e0 = chain.energy()
    peak = 0.0
    for _ in range(6000):
        chain.step()
        peak = max(peak, abs(chain.energy() - e0) / abs(e0))
    assert peak < 1e-9                       # E_string + E_body + E_conn + E_air holds
    assert loaded.radiated_energy > 0.05 * e0


def test_the_discrete_impedance_converges_to_the_continuous_one():
    # The pre-warping is a property of k, not of the model: as k -> 0 the two coincide, so a load
    # built at a very high rate needs no distinction at audio frequencies.
    omega = 2.0 * np.pi * 1000.0
    fine = RationalAirLoad.from_sphere(fs=4.0e6, radius=SPHERE_RADIUS_DEFAULT)
    # (omega k)^2 / 12 at 1 kHz and 4 MHz is ~2e-7 -- the tolerance is the warp, not slop.
    assert fine.impedance_discrete(omega) == pytest.approx(fine.impedance(omega), rel=1e-6)


def test_a_constant_R_load_has_no_added_mass():
    # The reactive half of loaded_mode(), isolated: Im Z = 0 means no mass, so no pitch shift at
    # all, and the decay rate collapses to batch 2's a^2 R / (2 m).
    flat = RationalAirLoad(fs=48000.0, R=2000.0, M_a=np.inf)
    w0 = 2.0 * np.pi * 300.0
    w_eff, alpha = flat.loaded_mode(w0, weight=0.02, mass=0.02)
    assert w_eff == pytest.approx(w0, rel=1e-14)
    assert alpha == pytest.approx(0.02 * 0.02 * 2000.0 / (2.0 * 0.02), rel=1e-14)


@pytest.mark.parametrize("kwargs", [dict(weight=0.02, mass=0.0), dict(weight=0.02, mass=-1.0)])
def test_loaded_mode_rejects_a_nonphysical_mass(kwargs):
    load = RationalAirLoad.from_sphere(fs=48000.0, radius=SPHERE_RADIUS_DEFAULT)
    with pytest.raises(ValueError):
        load.loaded_mode(2.0 * np.pi * 100.0, **kwargs)


def test_loaded_mode_rejects_a_nonpositive_frequency():
    load = RationalAirLoad.from_sphere(fs=48000.0, radius=SPHERE_RADIUS_DEFAULT)
    with pytest.raises(ValueError):
        load.loaded_mode(0.0, weight=0.02, mass=0.02)


# -- construction validation --------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fs=0.0, R=1.0),
        dict(fs=-48000.0, R=1.0),
        dict(fs=48000.0, R=-1.0),
        dict(fs=48000.0, R=1.0, M_a=0.0),
        dict(fs=48000.0, R=1.0, M_a=-1.0),
        dict(fs=48000.0, R=1.0, M_a=np.nan),
        dict(fs=48000.0, R=1.0, rho0=0.0),
        dict(fs=48000.0, R=1.0, c0=-1.0),
    ],
)
def test_rational_load_rejects_nonphysical_parameters(kwargs):
    with pytest.raises(ValueError):
        RationalAirLoad(**kwargs)


def test_from_sphere_rejects_a_nonpositive_radius():
    with pytest.raises(ValueError):
        RationalAirLoad.from_sphere(fs=48000.0, radius=0.0)


def test_a_timestep_mismatch_is_rejected():
    with pytest.raises(ValueError):
        ReactiveRadiatedBody(
            body=make_body(fs=48000.0), load=RationalAirLoad(fs=44100.0, R=1000.0)
        )


def test_far_field_refuses_an_inconsistent_load_but_serves_a_sphere():
    # (R, M_a) stays permissive by design -- an arbitrary pair is a perfectly good effective load.
    # Only the read-out that genuinely needs the sphere interpretation refuses to guess a radius.
    arbitrary = RationalAirLoad(fs=48000.0, R=2000.0, M_a=0.2)
    assert arbitrary.sphere_radius is None
    with pytest.raises(ValueError):
        arbitrary.far_field_pressure(1.0)
    sphere = RationalAirLoad.from_sphere(fs=48000.0, radius=0.05)
    assert sphere.sphere_radius == pytest.approx(0.05, rel=1e-12)
    sphere.step(1e-3)
    assert sphere.far_field_pressure(2.0) == pytest.approx(
        0.05 / 2.0 * sphere.pressure_load, rel=1e-14
    )
    with pytest.raises(ValueError):
        sphere.far_field_pressure(0.0)


def test_reset_clears_the_auxiliary_state_and_the_channels():
    load = RationalAirLoad.from_sphere(fs=48000.0, radius=0.05)
    for n in range(200):
        load.step(np.cos(2.0 * np.pi * 300.0 * n * load.k))
    assert load.u_l != 0.0 and load.radiated_energy > 0.0
    load.reset()
    assert load.u_l == 0.0 and load.radiated_energy == 0.0 and load.energy() == 0.0
