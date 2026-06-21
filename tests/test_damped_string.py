"""Validation suite for the damped stiff string (model #3, frequency-dependent loss).

Builds on the stiff-string harness (model #2) and adds model #3's defining check: the per-mode
**decay rate** is right -- high partials die faster than low ones, the ordering real strings have
and that model #2's frequency-*independent* loss got backwards. The PDE is

    u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma0 u_t + 2 sigma1 u_txx,

so a mode's energy decays as ``exp(-2 sigma_eff t)``, ``sigma_eff = sigma0 + sigma1 (wavenum)^2``.
See ``docs/dev/damped-string-plan.md`` and HANDOFF section 5 row 3 ("measured decay rates per
partial") / section 6.

Test decomposition (the honest split):

* **Implementation / money test** -- a single mode's measured per-step energy factor lands on the
  scheme's own closed-form discrete oracle ``g_m`` to a rate-relative ~1e-4 (far tighter than
  model #2's 1% low-mode bar; here *every* mode is checkable because ``g_m`` accounts for the
  theta-scheme rate-suppression exactly).
* **Physics anchor** -- the discrete rate tends to the continuum ``2(sigma0 + sigma1 beta^2)`` as
  the grid refines (a pure computation); ``sigma1`` flips the high-vs-low decay ordering (the cure).
* **Reduction anchor** -- ``sigma1 = 0`` is bit-for-bit :class:`StiffString` (free regression).

Tolerances inherit Milestone 1's bar (drift < 1e-10), set looser than the observed clean residuals.
"""

import numpy as np
import pytest
from helpers import (
    KAPPA_DEFAULT,
    L_DEFAULT,
    make_damped_string,
    measure_mode_decay_factor,
    wave_speed,
)

from physsynth.analysis import damping, modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_stiff import THETA_DEFAULT, StiffString

DRIFT_TOL = 1e-10  # acceptance criterion 1 (inherited)


def _pluck_run(s, secs=1.0, pluck_frac=0.137, pickup_frac=0.241):
    s.set_state(triangular_pluck(s.x, s.L, pluck_frac * s.L, amplitude=1e-3))
    steps = int(secs * s.fs)
    return simulate(s, num_steps=steps, pickup_index=int(round(pickup_frac * s.N)))


# =====================================================================================
# Energy: lossless conservation, the StiffString reduction, and passivity
# =====================================================================================


def test_lossless_energy_conserved_quick():
    # sigma0 = sigma1 = 0 must conserve energy to the inherited bar (it is the same conservative
    # scheme as model #2). A gross energy-form regression trips here in the fast suite.
    s = make_damped_string(N=100, lam=1.0, kappa=KAPPA_DEFAULT, sigma0=0.0, sigma1=0.0)
    res = _pluck_run(s, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e}"
    assert np.all(res.energy > 0.0)


def test_sigma1_zero_reduces_to_stiff_string_bit_for_bit():
    # Reduction anchor: with sigma1 = 0 the matrix A and the step RHS are constructed identically to
    # StiffString (the sigma1 terms are guarded out, like kappa), so a damped (sigma0 > 0) run must
    # match StiffString's energy AND pickup trace to the bit. Frees a whole regression dimension.
    N, fs = 128, wave_speed() * 128 / L_DEFAULT  # lam = 1
    common = dict(L=L_DEFAULT, T=200.0, rho=0.005, fs=fs, N=N, kappa=KAPPA_DEFAULT)
    ds = DampedStiffString(**common, sigma0=3.0, sigma1=0.0)
    ss = StiffString(**common, sigma=3.0)
    u0 = triangular_pluck(ds.x, ds.L, 0.137 * ds.L, amplitude=1e-3)
    ds.set_state(u0)
    ss.set_state(u0)
    pickup = int(round(0.241 * N))
    rd = simulate(ds, num_steps=1500, pickup_index=pickup)
    rs = simulate(ss, num_steps=1500, pickup_index=pickup)
    assert np.array_equal(rd.energy, rs.energy), (
        f"energy diverged from StiffString: max|d| = {np.max(np.abs(rd.energy - rs.energy)):.2e}"
    )
    assert np.array_equal(rd.output, rs.output)


@pytest.mark.slow
def test_passivity_broadband():
    # With both losses, a broadband pluck's energy never increases (passivity is unconditional: the
    # discrete losses -2 sigma0 ||delta_t. u||^2 - 2 sigma1 ||delta_x+ delta_t. u||^2 <= 0).
    s = make_damped_string(N=128, lam=1.0, kappa=KAPPA_DEFAULT, sigma0=2.0, sigma1=1e-4)
    res = _pluck_run(s, secs=1.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max() / e0:.2e} E0"
    assert np.all(np.isfinite(res.energy))


@pytest.mark.slow
def test_passivity_sigma1_only():
    # Isolate the NEW term (sigma0 = 0, sigma1 > 0): the frequency-dependent loss alone is passive.
    # A clean bisection -- a passivity failure here points straight at the sigma1 discretization.
    s = make_damped_string(N=128, lam=1.0, kappa=KAPPA_DEFAULT, sigma0=0.0, sigma1=5e-4)
    res = _pluck_run(s, secs=1.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max() / e0:.2e} E0"


# =====================================================================================
# Per-mode decay rate -- the money test (HANDOFF section 5 row 3)
# =====================================================================================


@pytest.mark.slow
def test_per_mode_decay_matches_discrete_oracle():
    # The money test: a single mode's measured per-step energy factor g_m lands on the scheme's
    # closed-form discrete oracle to a rate-relative ~1e-4 (observed worst ~2e-5 at the lowest mode,
    # where the dynamic range is smallest). This is far tighter than model #2's 1% low-mode bar
    # because g_m accounts for the theta-scheme rate-suppression exactly -- so EVERY mode checks.
    c, L, N, kappa = wave_speed(), L_DEFAULT, 128, KAPPA_DEFAULT
    sigma0, sigma1 = 2.0, 1e-4
    s = make_damped_string(N=N, lam=1.0, kappa=kappa, sigma0=sigma0, sigma1=sigma1)
    k, theta = s.k, s.theta

    worst = 0.0
    for m in (1, 2, 4, 8, 16, 32):
        assert damping.discrete_damped_mode_is_underdamped(
            c, L, N, kappa, k, theta, sigma0, sigma1, m
        ), f"mode {m} is not underdamped (g_m oracle assumes a complex-conjugate pair)"
        g_meas = measure_mode_decay_factor(
            m, N=N, lam=1.0, kappa=kappa, sigma0=sigma0, sigma1=sigma1, steps=15000
        )
        g_or = damping.discrete_damped_mode_decay(c, L, N, kappa, k, theta, sigma0, sigma1, m)
        rate_rel = abs(np.log(g_meas) - np.log(g_or)) / abs(np.log(g_or))
        worst = max(worst, rate_rel)
        assert rate_rel < 5e-4, f"mode {m}: decay-rate off by {rate_rel:.2e}"
    # Keep a foot on the gas: the suite as a whole should stay well under the bar.
    assert worst < 5e-4


@pytest.mark.slow
def test_sigma1_makes_high_partials_die_faster():
    # The cure, made falsifiable. sigma1 > 0 flips the decay ordering vs the model #2 regime:
    #   * oracle: over a confirmed-rising range [1..16] the per-mode rate RISES with sigma1 > 0 but
    #     FALLS with sigma1 = 0 (the model #2 backwards artifact -- highs under-damp). NOT asserted
    #     across the whole spectrum: the rate turns over past ~m=32 (numerator ~p^2 vs the theta
    #     denominator ~p^4), so this checks only the in-band range.
    #   * simulation: measured rate at m=16 exceeds m=2 with sigma1 > 0 (ties the claim to a run).
    c, L, N, kappa = wave_speed(), L_DEFAULT, 128, KAPPA_DEFAULT
    s = make_damped_string(N=N, lam=1.0, kappa=kappa, sigma0=2.0, sigma1=1e-4)
    k, theta = s.k, s.theta
    modes = np.arange(1, 17)

    def rates(s0, s1):
        return np.array(
            [
                damping.discrete_damped_mode_rate(c, L, N, kappa, k, theta, s0, s1, int(m))
                for m in modes
            ]
        )

    # Controlled comparison: same sigma0, only sigma1 differs -- the slope flip is purely sigma1's.
    rate_with = rates(2.0, 1e-4)
    rate_without = rates(2.0, 0.0)
    assert np.all(np.diff(rate_with) > 0.0), f"sigma1>0 rate not rising in-band: {rate_with}"
    assert np.all(np.diff(rate_without) < 0.0), f"sigma1=0 not falling (model #2): {rate_without}"

    # Simulated spot-check: a high mode genuinely decays faster than a low one with sigma1 > 0.
    def meas(m):
        g = measure_mode_decay_factor(
            m, N=N, lam=1.0, kappa=kappa, sigma0=2.0, sigma1=1e-4, steps=15000
        )
        return -np.log(g) / k

    rate_lo, rate_hi = meas(2), meas(16)
    assert rate_hi > rate_lo * 1.02, f"high not faster: rate16={rate_hi:.3f} <= rate2={rate_lo:.3f}"


# =====================================================================================
# T60 -> (sigma0, sigma1) mapping: the deepened-physics payoff (continuum, loose round-trip)
# =====================================================================================


@pytest.mark.slow
def test_T60_mapping_roundtrip():
    # Specify two decay-time targets (T60 decreasing with frequency so sigma0, sigma1 >= 0), invert
    # to (sigma0, sigma1), simulate those two modes, and recover the target T60s within a few %. The
    # gap is the documented continuum approximation (the sim decays at the discrete rate, with the
    # theta-factor) -- a physics demo, NOT a machine-precision test (that is the per-mode g_m test).
    c, L, N, kappa = wave_speed(), L_DEFAULT, 256, KAPPA_DEFAULT
    s_ref = make_damped_string(N=N, lam=1.0, kappa=kappa)  # for k, theta and the mode frequencies
    k, theta = s_ref.k, s_ref.theta
    m1, m2 = 1, 20
    f1 = modal.discrete_stiff_mode_frequency(c, L, N, kappa, k, m1, theta)
    f2 = modal.discrete_stiff_mode_frequency(c, L, N, kappa, k, m2, theta)
    T60_1, T60_2 = 4.0, 0.8

    sigma0, sigma1 = damping.loss_coefficients_from_T60(c, L, kappa, f1, T60_1, f2, T60_2)
    assert sigma0 > 0 and sigma1 > 0

    for m, T60_target in ((m1, T60_1), (m2, T60_2)):
        g = measure_mode_decay_factor(
            m, N=N, lam=1.0, kappa=kappa, sigma0=sigma0, sigma1=sigma1, steps=20000
        )
        # amplitude 60 dB drop (x1e-3): g^(n/2) = 1e-3 -> n = 2 ln(1e-3)/ln(g); T60 = n k.
        T60_meas = 2.0 * np.log(1e-3) / np.log(g) * k
        rel = abs(T60_meas - T60_target) / T60_target
        assert rel < 0.04, f"mode {m}: T60 {T60_meas:.3f}s vs target {T60_target:.3f}s ({rel:.2%})"


# =====================================================================================
# Stability / construction guards
# =====================================================================================


@pytest.mark.slow
@pytest.mark.parametrize("lam", [4.0, 2.0, 1.0, 0.5])
def test_no_nan_unconditional(lam):
    # No CFL limit: large kappa + lam >= 1 + both losses must run stably to completion.
    s = make_damped_string(N=100, lam=lam, kappa=5.0, sigma0=3.0, sigma1=2e-4)
    res = _pluck_run(s, secs=0.5)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -1.0},
        {"T": 0.0},
        {"L": -2.0},
        {"kappa": -0.1},
        {"sigma0": -0.1},
        {"sigma1": -0.1},
        {"N": 1},
        {"theta": 0.0},
        {"theta": 1.5},
        {"boundary": "clamped"},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"L": 1.0, "T": 200.0, "rho": 0.005, "fs": 20000.0, "N": 100}
    base.update(kwargs)
    with pytest.raises(ValueError):
        DampedStiffString(**base)


def test_lambda_above_one_accepted():
    # Unconditionally stable: lam > 1 must NOT be rejected (unlike IdealString).
    s = make_damped_string(N=100, lam=2.5, kappa=KAPPA_DEFAULT, sigma0=1.0, sigma1=1e-4)
    assert s.lam == pytest.approx(2.5, rel=1e-9)


def test_partials_unmoved_by_light_damping():
    # Light damping shifts the oscillation frequency only at O(damping^2): a plucked lightly-damped
    # string's partials still land on the (undamped) stiff oracle. Confirms the loss terms did not
    # corrupt the conservative scheme's dispersion.
    c, L, N, kappa = wave_speed(), L_DEFAULT, 128, KAPPA_DEFAULT
    s = make_damped_string(N=N, lam=1.0, kappa=kappa, sigma0=0.5, sigma1=1e-5)
    res = _pluck_run(s, secs=2.0)
    oracle = np.array(
        [modal.discrete_stiff_mode_frequency(c, L, N, kappa, s.k, m, s.theta) for m in range(1, 9)]
    )
    detected = spectrum.measure_partials_near(res.output, res.fs, oracle)
    assert not np.any(np.isnan(detected)), "a partial was not detected"
    assert np.max(np.abs(modal.cents(detected, oracle))) < 1.0  # within a cent


# =====================================================================================
# Fast unit tests: the damping oracle (no simulation)
# =====================================================================================


def test_continuum_rate_is_two_sigma_eff():
    c, L = wave_speed(), L_DEFAULT
    sigma0, sigma1 = 1.5, 3e-4
    for m in (1, 3, 7):
        beta2 = (m * np.pi / L) ** 2
        expected = 2.0 * (sigma0 + sigma1 * beta2)
        got = damping.modal_loss_rate_continuum(c, L, KAPPA_DEFAULT, sigma0, sigma1, m)
        assert got == pytest.approx(expected, rel=1e-12)
    # sigma1 = 0 is frequency-independent (same rate every mode); sigma1 > 0 grows with mode.
    flat = [damping.modal_loss_rate_continuum(c, L, 0.0, 2.0, 0.0, m) for m in (1, 5, 10)]
    assert flat[0] == flat[1] == flat[2]
    rising = [damping.modal_loss_rate_continuum(c, L, 0.0, 2.0, 1e-3, m) for m in (1, 5, 10)]
    assert rising[0] < rising[1] < rising[2]


def test_discrete_rate_tends_to_continuum_on_refinement():
    # Pure computation: as N, fs grow the discrete per-mode rate -> 2(sigma0 + sigma1 beta^2).
    c, L, kappa, m = wave_speed(), L_DEFAULT, KAPPA_DEFAULT, 4
    sigma0, sigma1 = 2.0, 5e-4
    cont = damping.modal_loss_rate_continuum(c, L, kappa, sigma0, sigma1, m)
    errs = []
    for N in (256, 512, 1024):
        k = L / (c * N)  # lam = 1
        r = damping.discrete_damped_mode_rate(c, L, N, kappa, k, THETA_DEFAULT, sigma0, sigma1, m)
        errs.append(abs(r - cont))
    assert errs[0] > errs[1] > errs[2], f"not converging to continuum: {errs}"
    assert errs[-1] / cont < 1e-3  # within 0.1% on the finest grid


def test_decay_factor_lossless_is_unity():
    c, L, N, kappa = wave_speed(), L_DEFAULT, 128, KAPPA_DEFAULT
    k = L / (c * N)  # lam = 1
    for m in (1, 10, 50):
        g = damping.discrete_damped_mode_decay(c, L, N, kappa, k, THETA_DEFAULT, 0.0, 0.0, m)
        assert g == pytest.approx(1.0, abs=1e-15)
        rate = damping.discrete_damped_mode_rate(c, L, N, kappa, k, THETA_DEFAULT, 0.0, 0.0, m)
        assert rate == 0.0


def test_loss_coefficients_from_T60_pure_roundtrip():
    # Invert the mapping against itself (no sim): pick (sigma0, sigma1), compute the continuum T60s
    # the formula implies at two frequencies, invert, recover (sigma0, sigma1) to machine precision.
    c, L, kappa = wave_speed(), L_DEFAULT, KAPPA_DEFAULT
    sigma0, sigma1 = 1.3, 8e-4

    def t60_at(f):
        omega2 = (2.0 * np.pi * f) ** 2
        beta2 = (-(c**2) + np.sqrt(c**4 + 4.0 * kappa**2 * omega2)) / (2.0 * kappa**2)
        return damping.T60_SECONDS_PER_RATE / (sigma0 + sigma1 * beta2)

    f1, f2 = 120.0, 1800.0
    s0, s1 = damping.loss_coefficients_from_T60(c, L, kappa, f1, t60_at(f1), f2, t60_at(f2))
    assert s0 == pytest.approx(sigma0, rel=1e-10)
    assert s1 == pytest.approx(sigma1, rel=1e-10)


def test_loss_coefficients_from_T60_rejects_increasing_T60():
    # T60 increasing with frequency would imply a negative sigma1 (brightness growing over time) --
    # unphysical for passive loss; rejected.
    c, L, kappa = wave_speed(), L_DEFAULT, KAPPA_DEFAULT
    with pytest.raises(ValueError):
        damping.loss_coefficients_from_T60(c, L, kappa, 100.0, 1.0, 2000.0, 5.0)
    with pytest.raises(ValueError):
        damping.loss_coefficients_from_T60(c, L, kappa, 100.0, 1.0, 100.0, 2.0)  # equal frequencies
