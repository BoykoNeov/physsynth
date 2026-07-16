"""Tension-modulated string — model #9 (Kirchhoff–Carrier), the string family's nonlinearity.

Two properties, and the suite turns on keeping them apart (see the plan doc,
``docs/dev/tension-modulated-string-plan.md``):

- **Energy conservation is STRUCTURAL** — it holds at any amplitude, from any state, and even while
  the motion below is disintegrating. It is the headline correctness gate (model #6's culture), and
  it must be measured where the nonlinear term is a real fraction of ``H`` *and* from a **broadband
  (plucked)** start: a single-mode energy test is secretly a *scalar Duffing* test and never
  exercises the cross-mode coupling through ``I = ∫u_x²`` that is the whole point of the model.
- **Mode purity is DYNAMICAL.** ``A(β)s = (λ₀+βp²)s`` for any tension, so single-mode-in →
  single-mode-out *per step*, at any amplitude — a short run certifies that structurally. But over a
  long run it survives only **below** a parametric threshold (``ΔT/T₀ ≈ 3``): the tension pumps at
  ``2ω_m``, roundoff-seeded neighbours sit in Mathieu tongues, and above threshold they grow
  exponentially and the mode breaks up. That breakup is **physics** (energy-conserving,
  refinement-invariant) and gets its own signature test.

Unlike model #6 this model has a **closed-form nonlinear oracle** — the single-mode reduction is an
exact Duffing oscillator with an elliptic-integral frequency and ``cn`` waveform
(:mod:`physsynth.analysis.duffing`), unit-tested here before it is used.
"""

import numpy as np
import pytest
from helpers import (
    EA_DEFAULT,
    KAPPA_DEFAULT,
    L_DEFAULT,
    RHO_DEFAULT,
    T_DEFAULT,
    THETA_DEFAULT,
    make_tension_string,
    measure_tension_mode_frequency,
    mode_off_fraction,
    wave_speed,
)

from physsynth.analysis.damping import spatial_eigenvalue_p2
from physsynth.analysis.duffing import (
    duffing_displacement,
    duffing_elliptic_parameter,
    duffing_frequency,
    duffing_frequency_expansion,
    duffing_frequency_shift,
    kc_mode_coefficients,
    kc_mode_stretch,
)
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_nonlinear import (
    TensionModulatedString,
    string_coefficients_from_material,
)

PLUCK_POS = 0.3


def _mode(s, m):
    """The exact discrete eigenvector ``sin(m π x / L)`` on ``s``'s grid."""
    return np.sin(m * np.pi * np.arange(s.N + 1) / s.N)


def _pluck(s, amplitude):
    return triangular_pluck(s.x, s.L, PLUCK_POS * s.L, amplitude=amplitude)


def _run(s, steps):
    """Step ``steps`` times; return (max relative drift, max nonlinear fraction, max T/T0)."""
    e0 = s.energy()
    drift = frac = tmax = 0.0
    for _ in range(steps):
        s.step()
        drift = max(drift, abs(s.energy() - e0) / abs(e0))
        frac = max(frac, s.nonlinear_energy() / s.energy())
        tmax = max(tmax, s.tension / s.T)
    return drift, frac, tmax


# == the oracle itself, before anything is validated against it ====================================
def test_duffing_frequency_linear_limit_is_exact():
    """``eps = 0`` must return ``omega0`` exactly — ``m = 0`` and ``K(0) = π/2`` cancel."""
    for w0sq in (1.0, 1e4, 3.947e5):
        assert duffing_frequency(2.3, w0sq, 0.0) == pytest.approx(np.sqrt(w0sq), rel=1e-15)


def test_duffing_frequency_matches_expansion_at_small_amplitude():
    """Elliptic form vs the independent Lindstedt–Poincaré expansion ω₀(1 + 3εA²/8ω₀²).

    They must converge as A → 0 (agreement is evidence *both* are right), and visibly diverge once
    ``εA²/ω₀²`` is no longer small (which is why the expansion is not the test oracle)."""
    w0sq, eps = 3.947e5, 5.0e7
    assert duffing_frequency(1e-3, w0sq, eps) == pytest.approx(
        duffing_frequency_expansion(1e-3, w0sq, eps), rel=1e-7
    )
    # at large amplitude the expansion is simply wrong (~74 % off) — the elliptic form is exact
    exact, approx = duffing_frequency(0.3, w0sq, eps), duffing_frequency_expansion(0.3, w0sq, eps)
    assert abs(exact - approx) / exact > 0.5


def test_duffing_elliptic_parameter_never_reaches_the_singularity():
    """Hardening ⇒ ``m ∈ [0, ½]``, strictly away from ``K(m)``'s ``m→1`` pole: the frequency is
    well-conditioned at *any* amplitude, however large. (A softening spring would not be — it has a
    finite escape amplitude.) Analytically ``m < ½`` strictly; in floating point it *rounds to* ½ at
    extreme amplitude, which is harmless: ``K(½) ≈ 1.854``, nowhere near the pole."""
    w0sq, eps = 3.947e5, 5.0e7
    for A in (0.0, 1.0, 1e4, 1e8):
        assert 0.0 <= duffing_elliptic_parameter(A, w0sq, eps) <= 0.5
    assert duffing_elliptic_parameter(1.0, w0sq, eps) < 0.5  # strict at any sane amplitude
    assert np.isfinite(duffing_frequency(1e8, w0sq, eps))  # no overflow / no singularity


def test_duffing_displacement_starts_at_rest_and_degenerates_to_cosine():
    """``q(0) = A`` (cn(0)=1, a maximum ⇒ q'(0)=0), and ``eps = 0`` gives ``A cos(ω₀t)`` exactly."""
    w0sq = 3.947e5
    assert duffing_displacement(0.0, 0.07, w0sq, 5e7) == pytest.approx(0.07, rel=1e-14)
    t = np.linspace(0, 0.05, 200)
    np.testing.assert_allclose(
        duffing_displacement(t, 0.07, w0sq, 0.0), 0.07 * np.cos(np.sqrt(w0sq) * t), atol=1e-14
    )


def test_kc_mode_stretch_matches_the_discrete_grid_exactly():
    """``I = q² p² L/2`` must equal the grid's own ``h‖δ_x⁺u‖²`` — the identity that makes the
    tension a function of ``q²`` alone, hence the whole single-mode reduction."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    for m in (1, 3, 7):
        q, shape = 0.37, _mode(s, m)
        p2 = spatial_eigenvalue_p2(s.N, s.h, m)
        grid = float(np.sum(np.diff(q * shape) ** 2) / s.h)
        assert kc_mode_stretch(q, p2=p2, L=s.L) == pytest.approx(grid, rel=1e-12)


def test_kc_mode_coefficients_reduce_to_the_linear_string():
    """``EA = 0`` ⇒ ``eps = 0``; and ``ω₀² = c²p² + κ²p⁴`` is the linear stiff-string relation."""
    p2 = 9.87
    w0sq, eps = kc_mode_coefficients(c=200.0, kappa=2.0, EA=0.0, rho=0.005, p2=p2, L=1.0)
    assert eps == 0.0
    assert w0sq == pytest.approx(200.0**2 * p2 + 2.0**2 * p2**2)


def test_duffing_oracle_rejects_nonphysical_input():
    with pytest.raises(ValueError):
        kc_mode_coefficients(c=200.0, kappa=2.0, EA=-1.0, rho=0.005, p2=9.87, L=1.0)
    with pytest.raises(ValueError):
        duffing_frequency(0.1, -1.0, 1e7)


# == #1 regression anchor: EA = 0 is model #3, bit-for-bit =========================================
@pytest.mark.parametrize("sigma0,sigma1", [(0.0, 0.0), (1.0, 0.0), (1.0, 5e-3)])
def test_EA_zero_is_model3_bit_identical(sigma0, sigma1):
    """The whole point of the θ-split: only the *nonlinear excess* moves to θ=½, so the linear path
    is provably untouched. Bit-identical, not merely close — ``EA=0`` skips the branch entirely."""
    fs = wave_speed() * 100 / L_DEFAULT
    kw = dict(L=L_DEFAULT, T=T_DEFAULT, rho=RHO_DEFAULT, fs=fs, N=100, kappa=KAPPA_DEFAULT,
              sigma0=sigma0, sigma1=sigma1, theta=THETA_DEFAULT)
    a = TensionModulatedString(EA=0.0, **kw)
    b = DampedStiffString(**kw)
    ic = _pluck(a, 0.01)
    a.set_state(ic.copy())
    b.set_state(ic.copy())
    for _ in range(400):
        a.step()
        b.step()
        assert np.array_equal(a.state, b.state)
    assert a.energy() == b.energy()
    assert a.nonlinear_energy() == 0.0


# == #2/#3 the headline: lossless energy conservation ==============================================
def test_lossless_drift_single_mode_large_amplitude():
    """Drift < 1e-10 where the nonlinear term is a *large* share of ``H``.

    A nonlinearity bug hides at small amplitude — there the test merely re-runs the linear scheme
    (model #6's lesson), so the nonlinear fraction is asserted, not just reported."""
    s = make_tension_string(N=100, EA=3e5)
    s.set_state(0.05 * _mode(s, 1))
    drift, frac, tmax = _run(s, 2000)
    assert frac > 0.5, f"nonlinear energy only {frac:.1%} of H — not a real nonlinear test"
    assert tmax > 5.0, f"tension only reached {tmax:.2f}×T0"
    assert drift < 1e-10, f"drift {drift:.3e} at nonlinear fraction {frac:.1%}"


def test_lossless_drift_plucked_broadband():
    """**The general-case energy test.** A single-mode start collapses to a scalar Duffing
    oscillator, whose conservation is nearly trivial and which never exercises the cross-mode
    coupling through ``I = ∫u_x²`` — the whole point of the model. A triangular pluck's corner is
    broadband, so the nonlinearity mixes modes from step 1. This is the realistic excitation."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    s.set_state(_pluck(s, 0.04))
    drift, frac, tmax = _run(s, 1200)
    assert frac > 0.25, f"nonlinear energy only {frac:.1%} of H"
    assert tmax > 1.8, f"tension only reached {tmax:.2f}×T0"
    assert drift < 1e-10, f"drift {drift:.3e} from a broadband pluck, nonlinear fraction {frac:.1%}"


def test_drift_falls_with_tension_tol():
    """The self-certification: absent a closed form for general motion, the proof that the drift is
    the *tension solve's* residual — not a scheme bug — is that it tracks ``tension_tol``. A loose
    solve must drift visibly; a tight one must reach machine precision."""
    drifts = {}
    for tol in (1e-4, 1e-6, 1e-8, 1e-12):
        s = make_tension_string(N=100, EA=EA_DEFAULT, tension_tol=tol)
        s.set_state(_pluck(s, 0.01))
        drifts[tol], _, _ = _run(s, 600)
    assert drifts[1e-4] > 1e-7, "a 1e-4 tension solve should drift visibly — is tol even used?"
    assert drifts[1e-4] > drifts[1e-6] > drifts[1e-8]
    assert drifts[1e-12] < 1e-11


def test_energy_non_negative_at_large_amplitude():
    """``H^n ≥ 0`` is gated, never assumed: stability does **not** transfer automatically from the
    linear θ-scheme to the nonlinear one (model #6's lesson)."""
    s = make_tension_string(N=100, EA=3e5)
    s.set_state(_pluck(s, 0.03))
    for _ in range(800):
        s.step()
        assert s.energy() >= 0.0


def test_passivity_with_losses():
    """σ > 0 ⇒ monotone non-increasing. The losses are model #3's and never enter ``E``, only its
    rate — so passivity should be exact, not approximate."""
    s = make_tension_string(N=100, EA=EA_DEFAULT, sigma0=2.0, sigma1=5e-3)
    s.set_state(_pluck(s, 0.01))
    prev = s.energy()
    for _ in range(1200):
        s.step()
        e = s.energy()
        assert e <= prev + 1e-18
        prev = e


def test_tension_only_ever_rises():
    """``I ≥ 0`` ⇒ ``T ≥ T₀`` always: transverse motion cannot *un*-stretch a string. Hardening,
    never softening — the sign that would flip pitch glide the wrong way."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    s.set_state(_pluck(s, 0.02))
    for _ in range(400):
        s.step()
        assert s.tension >= s.T
        assert s.delta_tension >= 0.0
        assert s.stretch >= 0.0


# == #7 mode purity: STRUCTURAL, short-run, at any amplitude =======================================
@pytest.mark.parametrize("m,amp", [(1, 0.05), (3, 0.03), (7, 0.02)])
def test_mode_purity_is_structural_short_run(m, amp):
    """``A(β)s = (λ₀+βp²)s`` for **any** tension ⇒ single-mode-in, single-mode-out, per step.

    Deliberately run **short** (20 steps) and at amplitudes well *above* the parametric threshold:
    the property is structural and per-step, so it must hold regardless of the threshold, and a real
    per-step leak out of ``span(s)`` shows immediately. The long-run dynamics are a separate matter
    (see the two tests below). Normalized by ``‖u₀‖`` — never the instantaneous ``‖u‖``, which
    passes through 0 twice a period and would report a spurious 1.0."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    shape = _mode(s, m)
    s.set_state(amp * shape)
    scale = float(np.linalg.norm(amp * shape))
    for _ in range(20):
        s.step()
        assert mode_off_fraction(s.state, shape, scale) < 1e-12


def test_single_mode_stays_pure_below_the_parametric_threshold():
    """Below threshold (ΔT/T₀ ≲ 2.6) the single-mode motion persists indefinitely — the reduction
    to a Duffing oscillator is dynamically stable, which is what lets the frequency oracles work."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    shape = _mode(s, 3)
    s.set_state(0.01 * shape)  # ΔT/T₀ ≈ 2.1 — sub-threshold
    scale = float(np.linalg.norm(0.01 * shape))
    worst, tmax = 0.0, 0.0
    for _ in range(1500):
        s.step()
        worst = max(worst, mode_off_fraction(s.state, shape, scale))
        tmax = max(tmax, s.tension / s.T)
    assert tmax < 3.0, f"meant to be sub-threshold, but reached {tmax:.2f}×T0"
    assert worst < 1e-11, f"off-mode {worst:.3e} below threshold — purity should persist here"


def test_single_mode_breaks_up_above_threshold_while_energy_conserves():
    """**The instability, as a signature test.** Above threshold the tension (pumping at 2ω₃) drives
    roundoff-seeded neighbours through Mathieu resonance and the mode disintegrates — energy moving
    into m=4 and m=8. This is *physics*, not a blow-up, and the discriminator is that **energy is
    conserved to machine precision throughout**: a numerical instability grows the energy; a
    parametric one only redistributes it. A linear string would hold its mode forever, so this test
    also fails on any accidental loss of the nonlinearity."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    shape = _mode(s, 3)
    s.set_state(0.03 * shape)  # ΔT/T₀ ≈ 11 — well above threshold
    scale = float(np.linalg.norm(0.03 * shape))
    e0 = s.energy()
    worst, drift = 0.0, 0.0
    for _ in range(1500):
        s.step()
        worst = max(worst, mode_off_fraction(s.state, shape, scale))
        drift = max(drift, abs(s.energy() - e0) / e0)
    assert worst > 1e-3, f"expected parametric breakup above threshold, got off-mode {worst:.3e}"
    assert drift < 1e-10, f"breakup must CONSERVE energy (drift {drift:.3e}) — else it's a blow-up"
    # the energy lands in the neighbours the Mathieu tongues predict, not in a grid-scale mode
    u = s.state
    amps = {mm: abs(float(np.dot(u, _mode(s, mm)) / np.dot(_mode(s, mm), _mode(s, mm))))
            for mm in (2, 4, 8, 20, 40)}
    assert max(amps[4], amps[8]) > 100 * max(amps[20], amps[40]), (
        f"off-mode energy should sit in the low neighbours (4/8), not at grid scale: {amps}"
    )


# == #9 the closed-form nonlinear oracle ==========================================================
@pytest.mark.parametrize("amp", [0.01, 0.02, 0.03])
def test_amplitude_shift_matches_duffing(amp):
    """**Lead with this**, not an absolute frequency. A measured ω(A) carries the linear θ-scheme's
    temporal dispersion error, which would dominate the nonlinear shift being measured; ω(A→0)
    carries the *same* error, so the difference cancels it and isolates the nonlinear physics."""
    N, m = 100, 1
    p2 = spatial_eigenvalue_p2(N, L_DEFAULT / N, m)
    w0sq, eps = kc_mode_coefficients(
        c=wave_speed(), kappa=KAPPA_DEFAULT, EA=EA_DEFAULT, rho=RHO_DEFAULT, p2=p2, L=L_DEFAULT
    )

    def measure(a):
        s = make_tension_string(N=N, EA=EA_DEFAULT)
        shape = _mode(s, m)
        s.set_state(a * shape)
        return measure_tension_mode_frequency(s, shape)

    shift_meas = measure(amp) - measure(1e-5)
    shift_oracle = duffing_frequency_shift(amp, w0sq, eps) / (2 * np.pi)
    assert shift_oracle > 1.0, "the shift should be a real, audible number of Hz"
    assert shift_meas == pytest.approx(shift_oracle, rel=1e-2)


def test_frequency_converges_to_the_exact_duffing_solution():
    """Richardson: error of ``q(t*)`` against the **exact** ``cn`` waveform (continuum β²) at
    O(h²), refining h and k together. Sub-threshold and short-time — breakup (or accumulated
    nonlinear phase drift, model #6's lesson) would make a convergence rate meaningless."""
    m, amp = 1, 0.02
    beta2 = (m * np.pi / L_DEFAULT) ** 2
    w0sq, eps = kc_mode_coefficients(
        c=wave_speed(), kappa=KAPPA_DEFAULT, EA=EA_DEFAULT, rho=RHO_DEFAULT, p2=beta2, L=L_DEFAULT
    )
    t_star = 2.0 * (2 * np.pi / duffing_frequency(amp, w0sq, eps))  # two nonlinear periods
    errs, hs = [], []
    for N in (50, 100, 200, 400):
        s = make_tension_string(N=N, EA=EA_DEFAULT)
        shape = _mode(s, m)
        s.set_state(amp * shape)
        n_steps = int(round(t_star * s.fs))
        for _ in range(n_steps):
            s.step()
        q_sim = float(np.dot(s.state, shape) / np.dot(shape, shape))
        q_exact = float(duffing_displacement(n_steps * s.k, amp, w0sq, eps))
        errs.append(abs(q_sim - q_exact))
        hs.append(s.h)
    errs, hs = np.array(errs), np.array(hs)
    orders = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    # Measured 2.97 → 2.40 → 2.25: approaching 2 from above (the spatial O(h²) and temporal O(k²)
    # errors partly cancel at these resolutions). Assert the rate is real and second-order-ish.
    assert np.all(orders > 1.9), f"orders {orders} — not second order"
    assert orders[-1] < 3.2, f"orders {orders} — suspiciously fast, check the oracle"


@pytest.mark.parametrize("amp", [0.005, 0.02])
def test_absolute_frequency_matches_duffing_loosely(amp):
    """The absolute frequency *does* land on the oracle — but only to ~0.1 %, because it carries the
    linear θ-scheme dispersion error. Kept as a coarse sanity bar, and as the documented reason the
    amplitude-shift test above is the tight one."""
    N, m = 100, 1
    p2 = spatial_eigenvalue_p2(N, L_DEFAULT / N, m)
    w0sq, eps = kc_mode_coefficients(
        c=wave_speed(), kappa=KAPPA_DEFAULT, EA=EA_DEFAULT, rho=RHO_DEFAULT, p2=p2, L=L_DEFAULT
    )
    s = make_tension_string(N=N, EA=EA_DEFAULT)
    shape = _mode(s, m)
    s.set_state(amp * shape)
    f_meas = measure_tension_mode_frequency(s, shape)
    assert f_meas == pytest.approx(duffing_frequency(amp, w0sq, eps) / (2 * np.pi), rel=3e-3)


def test_pitch_rises_monotonically_with_amplitude_and_lands_on_the_linear_limit():
    """The payoff, and the sign check: hit it harder → it goes **sharp** (hardening, never
    softening). As A → 0 the frequency lands on model #3's linear fundamental."""
    N, m = 100, 1
    freqs = []
    for amp in (1e-5, 0.01, 0.02, 0.03, 0.05):
        s = make_tension_string(N=N, EA=EA_DEFAULT)
        shape = _mode(s, m)
        s.set_state(amp * shape)
        freqs.append(measure_tension_mode_frequency(s, shape))
    assert np.all(np.diff(freqs) > 0), f"pitch must rise with amplitude: {freqs}"
    assert freqs[-1] > 1.5 * freqs[0], f"expected a big glide by amp=0.05, got {freqs}"

    # A → 0 lands on the linear string (model #3), which the nonlinear model must contain
    lin = make_tension_string(N=N, EA=0.0)
    shape = _mode(lin, m)
    lin.set_state(1e-5 * shape)
    assert freqs[0] == pytest.approx(measure_tension_mode_frequency(lin, shape), rel=1e-6)


# == #11 guards ===================================================================================
def test_rejects_negative_EA():
    with pytest.raises(ValueError, match="EA"):
        make_tension_string(EA=-1.0)


def test_rejects_bad_tension_tol_and_boundary():
    with pytest.raises(ValueError, match="tension_tol"):
        make_tension_string(EA=EA_DEFAULT, tension_tol=0.0)
    with pytest.raises(ValueError, match="boundary"):
        TensionModulatedString(L=1.0, T=200.0, rho=0.005, fs=20000.0, N=50, boundary="fixed")


def test_apply_Ainv_refuses_because_A_is_time_varying():
    """Model #3 exposes a *constant* driving-point admittance ``A⁻¹eᵢ`` for coupled exciters. Here
    ``A = A₀ − βD2`` moves with the tension every step, so that precompute would be silently wrong —
    it must raise, not return a plausible number."""
    s = make_tension_string(EA=EA_DEFAULT)
    with pytest.raises(NotImplementedError, match="time-varying"):
        s.apply_Ainv(np.zeros(s.N - 1))


def test_solver_telemetry_is_exposed():
    """The self-certifying gates must be observable: a run that never fails to converge is the
    certificate (model #6's honesty rule — never render a non-converged run as physics)."""
    s = make_tension_string(N=100, EA=EA_DEFAULT)
    s.set_state(_pluck(s, 0.02))
    for _ in range(300):
        s.step()
    assert s.converged
    assert s.n_not_converged == 0
    assert s.delta_tension > 0.0


def test_string_at_rest_stays_at_rest():
    """Zero stretch ⇒ zero tension modulation ⇒ the degenerate bracket must not be entered."""
    s = make_tension_string(N=50, EA=EA_DEFAULT)
    s.set_state(np.zeros(s.N + 1))
    for _ in range(20):
        s.step()
    assert np.all(s.state == 0.0)
    assert s.delta_tension == 0.0
    assert s.energy() == 0.0


# == the material helper (a modeling oracle, not a constraint) =====================================
@pytest.mark.parametrize("radius", [2e-4, 5e-4, 1e-3])
def test_material_helper_ratio_is_radius_independent(radius):
    """``EA/T₀ = E πr² / (ρ_v πr² c²) = (c_long/c)²`` — the radius **cancels exactly**. Hardening is
    set by the ratio of longitudinal to transverse wave speed, which is why the core takes ``EA``
    rather than a three-number materials surface that over-parameterizes this physics."""
    co = string_coefficients_from_material(E=2.0e11, radius=radius, rho_v=7850.0, T=T_DEFAULT)
    assert co.EA_over_T == pytest.approx((co.c_long / co.c) ** 2, rel=1e-12)
    assert co.c_long == pytest.approx(np.sqrt(2.0e11 / 7850.0), rel=1e-12)


def test_material_helper_lands_in_the_real_steel_range():
    """A real steel string should come out at ``EA/T ~ 150–600`` — the sanity check that the helper
    is a modeling oracle and not just arithmetic."""
    co = string_coefficients_from_material(E=2.0e11, radius=2e-4, rho_v=7850.0, T=T_DEFAULT)
    assert 100 < co.EA_over_T < 700
    assert 4000 < co.c_long < 6000


def test_material_helper_feeds_the_core_consistently():
    """The helper's coefficients must build a string that actually runs and conserves energy — the
    realism path is exercised, not just offered."""
    co = string_coefficients_from_material(E=2.0e11, radius=5e-4, rho_v=7850.0, T=T_DEFAULT)
    s = TensionModulatedString(
        L=L_DEFAULT, T=T_DEFAULT, rho=co.rho, fs=48000.0, N=100, kappa=co.kappa, EA=co.EA
    )
    s.set_state(_pluck(s, 2e-3))
    drift, frac, _ = _run(s, 600)
    assert frac > 1e-3, "a real steel string at a hard pluck should show some nonlinearity"
    assert drift < 1e-10


def test_material_helper_rejects_nonphysical_input():
    with pytest.raises(ValueError):
        string_coefficients_from_material(E=-1.0, radius=5e-4, rho_v=7850.0, T=T_DEFAULT)
