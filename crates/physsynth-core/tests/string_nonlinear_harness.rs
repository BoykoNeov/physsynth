//! The tension-modulated string's validation harness — model #9, carried from
//! `tests/test_tension_string.py` (retirement plan §34) and run at the retired Python helper
//! `make_tension_string`'s parameters: `L = 1`, `T = 200`, `rho = 0.005` (so `c = 200 m/s`),
//! `kappa = 2`, `theta = 0.28`, `tension_tol = 1e-13`, `fs = c N / (L lam)` with `lam = 1`, and
//! `EA = 1e5` (`EA/T = 500`, a real steel string's ratio). The pluck is at `0.3 L`.
//!
//! **A separate file from `string_nonlinear.rs`**, for the reason `string_damped_harness.rs` is
//! one: that file runs another fixture (`fs = 44100`, `kappa = 1.5`) and holds
//! `the_stretch_squares_with_pow_and_not_a_multiply`, which must run unoptimised.
//!
//! Two properties, and the suite turns on keeping them apart
//! (`docs/dev/tension-modulated-string-plan.md`):
//!
//! - **Energy conservation is STRUCTURAL**: it holds at any amplitude, from any state, even while
//!   the motion below is disintegrating. It must be measured where the nonlinear term is a real
//!   fraction of `H` *and* from a **broadband (plucked)** start — a single-mode energy test is
//!   secretly a scalar Duffing test and never exercises the cross-mode coupling through
//!   `I = ∫u_x²`. Every native energy bar before this file started on one mode.
//! - **Mode purity is DYNAMICAL.** `A(β)s = (λ₀ + βp²)s` for any tension, so a single mode stays
//!   single per step at any amplitude; over a long run only **below** a parametric threshold
//!   (`ΔT/T₀ ≈ 3`). Above it the breakup is physics — energy-conserving — and has its own bar.
//!
//! The Duffing oracle's own bars went to `crates/physsynth-analysis/tests/oracles.rs`, except the
//! stretch identity, which needs this crate's grid. The material helper was ported with this batch
//! (`string_nonlinear::string_coefficients_from_material`), and NumPy's arithmetic for it is the
//! one outside referee here, recorded before the deletion and asserted to the bit.
//!
//! One Python test has **no analogue**: `test_apply_Ainv_refuses_because_A_is_time_varying`
//! asserted that a call raises. The native [`TensionModulatedString`] has no `apply_ainv` and
//! implements no trait that asks for one, so the wrong call does not compile (§16's "an absence
//! becomes a type").

use physsynth_analysis::damping::spatial_eigenvalue_p2;
use physsynth_analysis::duffing::{
    duffing_displacement, duffing_frequency, duffing_frequency_shift, kc_mode_coefficients,
    kc_mode_stretch,
};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_damped as damped;
use physsynth_core::string_nonlinear::{
    self as nl, string_coefficients_from_material, MaterialError, ParamError,
    TensionModulatedString,
};

/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, f0 = 100 Hz
/// `KAPPA_DEFAULT`.
const KAPPA: f64 = 2.0;
/// `THETA_DEFAULT`, written here rather than imported (§29.3).
const THETA: f64 = 0.28;
/// `TENSION_TOL_DEFAULT`, written here for the same reason.
const TOL: f64 = 1e-13;
/// `EA_DEFAULT`: `EA/T = 500`.
const EA: f64 = 1.0e5;
const PLUCK_POS: f64 = 0.3;

/// `wave_speed()` — 200 m/s.
fn c() -> f64 {
    (T / RHO).sqrt()
}

/// The retired `make_tension_string`'s parameters at `lam = 1`: `fs = c N / L`.
#[allow(clippy::too_many_arguments)]
fn params(
    n: i64,
    kappa: f64,
    ea: f64,
    sigma0: f64,
    sigma1: f64,
    theta: f64,
    tol: f64,
) -> nl::Params {
    let fs = c() * n as f64 / L;
    nl::Params::new(
        L, T, RHO, fs, n, kappa, ea, sigma0, sigma1, theta, tol, true,
    )
    .expect("valid parameters")
}

/// `make_tension_string(N=n, EA=ea)` with everything else at its default.
fn make(n: i64, ea: f64) -> TensionModulatedString {
    TensionModulatedString::new(params(n, KAPPA, ea, 0.0, 0.0, THETA, TOL))
}

/// The Python's `_mode`: `np.sin(m * np.pi * np.arange(N + 1) / N)`, spelled in its order.
fn mode(n: i64, m: i64) -> Vec<f64> {
    (0..=n)
        .map(|j| ((m as f64 * std::f64::consts::PI) * j as f64 / n as f64).sin())
        .collect()
}

fn scaled(v: &[f64], a: f64) -> Vec<f64> {
    v.iter().map(|x| a * x).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Start `s` from rest at displacement `u0` — `set_state(u0)`.
fn start(s: &mut TensionModulatedString, u0: &[f64]) {
    let v0 = vec![0.0; u0.len()];
    s.set_state(u0, &v0);
}

/// The Python's `_pluck`: a triangle of height `amplitude` at `0.3 L`, with the model's exciter.
fn pluck(s: &TensionModulatedString, amplitude: f64) -> Vec<f64> {
    triangular_pluck(&s.p.grid(), s.p.l, PLUCK_POS * s.p.l, amplitude).expect("an interior pluck")
}

fn step(s: &mut TensionModulatedString) {
    s.step().expect("the tension solve must not error");
}

/// The Python's `_run`: step `steps` times; return (max relative drift, max nonlinear fraction,
/// max `T/T0`).
fn run(s: &mut TensionModulatedString, steps: usize) -> (f64, f64, f64) {
    let e0 = s.energy();
    let (mut drift, mut frac, mut tmax) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..steps {
        step(s);
        let e = s.energy();
        // NaN-propagating, as `max` on a Python float is not but the bars below need (§29.3).
        let d = (e - e0).abs() / e0.abs();
        drift = if d.is_nan() { f64::NAN } else { drift.max(d) };
        frac = frac.max(s.nonlinear_energy() / e);
        tmax = tmax.max(s.tension() / s.p.t);
    }
    (drift, frac, tmax)
}

/// The modal projection `<u, shape> / <shape, shape>`.
fn project(u: &[f64], shape: &[f64]) -> f64 {
    dot(u, shape) / dot(shape, shape)
}

/// The retired `mode_off_fraction`: off-mode content of `u` over the **fixed** amplitude `scale`
/// (never the instantaneous `||u||`, which passes through 0 twice a period).
fn off_fraction(u: &[f64], shape: &[f64], scale: f64) -> f64 {
    let q = project(u, shape);
    let rest: Vec<f64> = u.iter().zip(shape).map(|(a, b)| a - q * b).collect();
    norm(&rest) / scale
}

/// The retired `measure_tension_mode_frequency`: the mode's **nonlinear** frequency (Hz) from ten
/// descending zero crossings of the modal projection, each linearly interpolated, as one over the
/// mean spacing. Zero crossings, not a spectrum: a window anchored on the linear frequency misses
/// a peak shifted tens of percent by hardening.
fn measure_frequency(s: &mut TensionModulatedString, shape: &[f64]) -> f64 {
    let denom = dot(shape, shape);
    let mut prev = dot(&s.u, shape) / denom;
    let mut times = Vec::new();
    for n in 1..=400_000usize {
        step(s);
        let cur = dot(&s.u, shape) / denom;
        if prev > 0.0 && 0.0 >= cur {
            times.push(((n - 1) as f64 + prev / (prev - cur)) * s.p.k);
            if times.len() >= 10 {
                break;
            }
        }
        prev = cur;
    }
    assert!(
        times.len() >= 2,
        "only {} crossings — can't measure",
        times.len()
    );
    let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    1.0 / (gaps.iter().sum::<f64>() / gaps.len() as f64)
}

/// A mode-1 string at amplitude `a` and its measured frequency, at `N = 100`.
fn measure_mode_one(a: f64, ea: f64) -> f64 {
    let mut s = make(100, ea);
    let shape = mode(100, 1);
    start(&mut s, &scaled(&shape, a));
    measure_frequency(&mut s, &shape)
}

/// `(ω₀², ε)` of mode 1 at `N = 100`, on the discrete `p²` — the scheme's own Duffing.
fn discrete_mode_one_duffing() -> (f64, f64) {
    let p2 = spatial_eigenvalue_p2(100, L / 100.0, 1);
    kc_mode_coefficients(c(), KAPPA, EA, RHO, p2, L).expect("physical coefficients")
}

// == the stretch identity the single-mode reduction stands on =====================================

#[test]
fn the_modal_stretch_is_the_grids_own_stretch_exactly() {
    // Carried from `test_kc_mode_stretch_matches_the_discrete_grid_exactly`. `I = q² p² L/2` must
    // equal the grid's `h‖δ_x⁺u‖²` — the identity that makes the tension a function of `q²` alone.
    // Checked against both a hand-written sum (the Python's `np.sum(np.diff(...)**2) / h`) and the
    // model's own `stretch`, which is the function the step actually calls.
    let s = make(100, EA);
    for m in [1i64, 3, 7] {
        let u = scaled(&mode(100, m), 0.37);
        let p2 = spatial_eigenvalue_p2(100, s.p.h, m);
        let by_hand: f64 = u.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>() / s.p.h;
        let modal = kc_mode_stretch(0.37, p2, s.p.l);
        let model = nl::stretch(&u, &s.p);
        println!("mode {m}: modal {modal:e}, by hand {by_hand:e}, model {model:e}");
        assert!(
            (modal / by_hand - 1.0).abs() < 1e-12,
            "mode {m}: {modal} vs {by_hand}"
        );
        assert!(
            (modal / model - 1.0).abs() < 1e-12,
            "mode {m}: {modal} vs the model's {model}"
        );
    }
}

// == the regression anchor: EA = 0 is model #3, bit for bit ======================================

#[test]
fn ea_zero_is_model_three_bit_for_bit_at_the_pythons_fixture() {
    // Carried from `test_EA_zero_is_model3_bit_identical`, at its three loss sets. The native
    // anchor in `string_nonlinear.rs` runs a different fixture with sigma1 at 8e-5; the Python's
    // 5e-3 is sixty times that, which is the loss the frequency-dependent band actually carries.
    for (sigma0, sigma1) in [(0.0, 0.0), (1.0, 0.0), (1.0, 5e-3)] {
        let fs = c() * 100.0 / L;
        let pd = damped::Params::new(L, T, RHO, fs, 100, KAPPA, sigma0, sigma1, THETA, true)
            .expect("valid parameters");
        let mut a =
            TensionModulatedString::new(params(100, KAPPA, 0.0, sigma0, sigma1, THETA, TOL));
        let mut b = damped::DampedStiffString::new(pd);
        let ic = pluck(&a, 0.01);
        start(&mut a, &ic);
        b.set_state(&ic, &vec![0.0; ic.len()]);
        for n in 0..400 {
            step(&mut a);
            b.step();
            assert_eq!(
                a.u, b.u,
                "state diverged at step {n}, sigma = ({sigma0}, {sigma1})"
            );
        }
        assert_eq!(a.energy(), b.energy());
        assert_eq!(a.nonlinear_energy(), 0.0);
    }
}

// == the headline: lossless energy conservation ==================================================

#[test]
fn a_single_mode_at_large_amplitude_conserves_energy_where_the_stretch_dominates() {
    // Carried from `test_lossless_drift_single_mode_large_amplitude`. A nonlinearity bug hides at
    // small amplitude, so the nonlinear fraction and the tension excursion are asserted in the
    // SAME run as the drift, not in a separate one.
    let mut s = make(100, 3e5);
    start(&mut s, &scaled(&mode(100, 1), 0.05));
    let (drift, frac, tmax) = run(&mut s, 2000);
    println!("single mode: drift {drift:e}, fraction {frac}, T/T0 {tmax}");
    assert!(
        frac > 0.5,
        "nonlinear energy only {frac} of H — not a real nonlinear test"
    );
    assert!(tmax > 5.0, "tension only reached {tmax} T0");
    assert!(
        drift < DRIFT_TOL,
        "drift {drift:e} at nonlinear fraction {frac}"
    );
}

#[test]
fn a_broadband_pluck_conserves_energy_through_the_cross_mode_coupling() {
    // Carried from `test_lossless_drift_plucked_broadband` — **the general-case energy test**. A
    // single-mode start collapses to a scalar Duffing oscillator and never exercises the coupling
    // through `I = ∫u_x²`; a triangle's corner is broadband, so the nonlinearity mixes modes from
    // step 1. Every native energy bar before this one started on a single mode.
    let mut s = make(100, EA);
    let ic = pluck(&s, 0.04);
    start(&mut s, &ic);
    let (drift, frac, tmax) = run(&mut s, 1200);
    println!("pluck: drift {drift:e}, fraction {frac}, T/T0 {tmax}");
    assert!(frac > 0.25, "nonlinear energy only {frac} of H");
    assert!(tmax > 1.8, "tension only reached {tmax} T0");
    assert!(
        drift < DRIFT_TOL,
        "drift {drift:e} from a broadband pluck, fraction {frac}"
    );
}

#[test]
fn a_broadband_pluck_conserves_energy_at_every_theta() {
    // Added by §34, the human's call (§33's sweep on this model). Every Python test and every native
    // bar built the string at the default theta = 0.28, where a theta hard-coded to 0.28 in the
    // step or in the energy is exact, and both were planted and seen by nothing in the workspace.
    // theta = 1/4 is the edge of the scheme's unconditional stability; 1/2 is far from the default.
    for theta in [0.25, 0.28, 0.5] {
        let mut s = TensionModulatedString::new(params(100, KAPPA, EA, 0.0, 0.0, theta, TOL));
        let ic = pluck(&s, 0.04);
        start(&mut s, &ic);
        let (drift, frac, _) = run(&mut s, 1200);
        println!("theta {theta}: drift {drift:e}, fraction {frac}");
        assert!(
            frac > 0.25,
            "theta {theta}: nonlinear energy only {frac} of H"
        );
        assert!(drift < DRIFT_TOL, "theta {theta}: drift {drift:e}");
    }
}

#[test]
fn the_drift_tracks_the_tension_solves_tolerance() {
    // Carried from `test_drift_falls_with_tension_tol` — the self-certification. Absent a closed
    // form for general motion, the proof that the drift is the tension solve's residual and not a
    // scheme bug is that it follows `tension_tol`: a loose solve drifts visibly, a tight one does
    // not. No native bar showed the tolerance is honoured at all.
    let tols = [1e-4, 1e-6, 1e-8, 1e-12];
    let drifts: Vec<f64> = tols
        .iter()
        .map(|&tol| {
            let mut s = TensionModulatedString::new(params(100, KAPPA, EA, 0.0, 0.0, THETA, tol));
            let ic = pluck(&s, 0.01);
            start(&mut s, &ic);
            run(&mut s, 600).0
        })
        .collect();
    println!("drift by tension_tol {tols:?}: {drifts:?}");
    assert!(
        drifts[0] > 1e-7,
        "a 1e-4 tension solve should drift visibly — is tol even used?"
    );
    assert!(drifts[0] > drifts[1] && drifts[1] > drifts[2], "{drifts:?}");
    assert!(drifts[3] < 1e-11, "{drifts:?}");
}

#[test]
fn the_energy_stays_non_negative_at_large_amplitude() {
    // Carried from `test_energy_non_negative_at_large_amplitude`: `H >= 0` is gated, never assumed
    // — stability does not transfer automatically from the linear theta-scheme to the nonlinear one.
    let mut s = make(100, 3e5);
    let ic = pluck(&s, 0.03);
    start(&mut s, &ic);
    for n in 0..800 {
        step(&mut s);
        let e = s.energy();
        assert!(e >= 0.0, "energy {e} at step {n}");
    }
}

#[test]
fn losses_make_the_energy_monotone_from_a_pluck() {
    // Carried from `test_passivity_with_losses`. The losses are model #3's and never enter `E`,
    // only its rate, so passivity is exact; the Python's `+ 1e-18` slack is kept as it was.
    let mut s = TensionModulatedString::new(params(100, KAPPA, EA, 2.0, 5e-3, THETA, TOL));
    let ic = pluck(&s, 0.01);
    start(&mut s, &ic);
    let mut prev = s.energy();
    let mut worst = f64::NEG_INFINITY;
    for n in 0..1200 {
        step(&mut s);
        let e = s.energy();
        assert!(
            e <= prev + 1e-18,
            "energy rose at step {n}: {prev:e} -> {e:e}"
        );
        worst = worst.max(e - prev);
        prev = e;
    }
    println!("passivity: worst one-step change {worst:e}");
}

#[test]
fn the_tension_only_ever_rises_after_a_pluck() {
    // Carried from `test_tension_only_ever_rises`: `I >= 0`, so `T >= T0` always — hardening,
    // never softening. Asserted on all three reported quantities, as the Python did.
    let mut s = make(100, EA);
    let ic = pluck(&s, 0.02);
    start(&mut s, &ic);
    for n in 0..400 {
        step(&mut s);
        assert!(
            s.tension() >= s.p.t,
            "tension {} below T0 at step {n}",
            s.tension()
        );
        assert!(s.delta_tension >= 0.0, "dT {} at step {n}", s.delta_tension);
        assert!(s.stretch() >= 0.0, "stretch {} at step {n}", s.stretch());
    }
}

// == mode purity: structural, then dynamical =====================================================

#[test]
fn a_single_mode_stays_single_per_step_at_any_amplitude() {
    // Carried from `test_mode_purity_is_structural_short_run` (three cases). Twenty steps at
    // amplitudes well above the parametric threshold: the property is per step, so a real leak out
    // of span(s) shows at once. Normalised by ||u0||, never the instantaneous ||u||.
    for (m, amp) in [(1i64, 0.05), (3, 0.03), (7, 0.02)] {
        let mut s = make(100, EA);
        let shape = mode(100, m);
        let u0 = scaled(&shape, amp);
        let scale = norm(&u0);
        start(&mut s, &u0);
        let mut worst = 0.0f64;
        for n in 0..20 {
            step(&mut s);
            let f = off_fraction(&s.u, &shape, scale);
            assert!(f < 1e-12, "mode {m}: off-mode {f:e} at step {n}");
            worst = worst.max(f);
        }
        println!("mode {m}: worst off-mode {worst:e}");
    }
}

#[test]
fn a_single_mode_stays_pure_below_the_parametric_threshold() {
    // Carried from `test_single_mode_stays_pure_below_the_parametric_threshold`: below threshold
    // the single-mode motion persists, which is what lets the frequency oracles below work.
    let mut s = make(100, EA);
    let shape = mode(100, 3);
    let u0 = scaled(&shape, 0.01);
    let scale = norm(&u0);
    start(&mut s, &u0);
    let (mut worst, mut tmax) = (0.0f64, 0.0f64);
    for _ in 0..1500 {
        step(&mut s);
        let f = off_fraction(&s.u, &shape, scale);
        worst = if f.is_nan() { f64::NAN } else { worst.max(f) };
        tmax = tmax.max(s.tension() / s.p.t);
    }
    println!("sub-threshold: worst off-mode {worst:e}, T/T0 {tmax}");
    assert!(
        tmax < 3.0,
        "meant to be sub-threshold, but reached {tmax} T0"
    );
    assert!(
        worst < 1e-11,
        "off-mode {worst:e} below threshold — purity should persist here"
    );
}

#[test]
fn above_the_threshold_the_mode_breaks_up_while_the_energy_is_conserved() {
    // Carried from `test_single_mode_breaks_up_above_threshold_while_energy_conserves` — the
    // instability as a signature. The tension pumps at 2 omega_3 and drives roundoff-seeded
    // neighbours through Mathieu resonance; energy moves into m = 4 and m = 8. A numerical
    // instability grows the energy, a parametric one only redistributes it, so the drift is the
    // discriminator. A linear string would hold its mode forever.
    let mut s = make(100, EA);
    let shape = mode(100, 3);
    let u0 = scaled(&shape, 0.03);
    let scale = norm(&u0);
    start(&mut s, &u0);
    let e0 = s.energy();
    let (mut worst, mut drift) = (0.0f64, 0.0f64);
    let mut crossed = None;
    for n in 0..1500 {
        step(&mut s);
        let f = off_fraction(&s.u, &shape, scale);
        worst = worst.max(f);
        if crossed.is_none() && f > 1e-3 {
            crossed = Some(n + 1);
        }
        let d = (s.energy() - e0).abs() / e0;
        drift = if d.is_nan() { f64::NAN } else { drift.max(d) };
    }
    let amp = |mm: i64| project(&s.u, &mode(100, mm)).abs();
    let amps: Vec<(i64, f64)> = [2i64, 4, 8, 20, 40]
        .iter()
        .map(|&mm| (mm, amp(mm)))
        .collect();
    println!("breakup: worst {worst:e} (crossed 1e-3 at step {crossed:?}), drift {drift:e}");
    println!("breakup: modal amplitudes {amps:?}");
    assert!(
        worst > 1e-3,
        "expected parametric breakup above threshold, got off-mode {worst:e}"
    );
    assert!(
        drift < DRIFT_TOL,
        "breakup must CONSERVE energy (drift {drift:e})"
    );
    assert!(
        amp(4).max(amp(8)) > 100.0 * amp(20).max(amp(40)),
        "off-mode energy should sit in the low neighbours (4/8), not at grid scale: {amps:?}"
    );
}

// == the closed-form nonlinear oracle ===========================================================

#[test]
fn the_amplitude_shift_lands_on_the_duffing_shift() {
    // Carried from `test_amplitude_shift_matches_duffing` (three cases) — the tight frequency bar.
    // A measured omega(A) carries the linear theta-scheme's temporal dispersion; omega(A -> 0)
    // carries the same error, so the difference isolates the nonlinear physics.
    let (w0sq, eps) = discrete_mode_one_duffing();
    let f_small = measure_mode_one(1e-5, EA);
    for amp in [0.01, 0.02, 0.03] {
        let shift = measure_mode_one(amp, EA) - f_small;
        let oracle =
            duffing_frequency_shift(amp, w0sq, eps).unwrap() / (2.0 * std::f64::consts::PI);
        println!("A = {amp}: measured shift {shift} Hz, Duffing {oracle} Hz");
        assert!(
            oracle > 1.0,
            "the shift should be a real, audible number of Hz"
        );
        assert!(
            (shift - oracle).abs() <= 1e-2 * oracle.abs(),
            "A = {amp}: {shift} vs {oracle}"
        );
    }
}

#[test]
fn the_waveform_converges_at_second_order_to_the_exact_duffing_solution() {
    // Carried from `test_frequency_converges_to_the_exact_duffing_solution`. Richardson: error of
    // q(t*) against the exact `cn` waveform (on the continuum beta²) after two nonlinear periods,
    // refining h and k together. The Python measured 2.97 -> 2.40 -> 2.25, approaching 2 from above.
    let amp = 0.02;
    let beta2 = (std::f64::consts::PI / L).powi(2);
    let (w0sq, eps) = kc_mode_coefficients(c(), KAPPA, EA, RHO, beta2, L).unwrap();
    let t_star = 2.0 * (2.0 * std::f64::consts::PI / duffing_frequency(amp, w0sq, eps).unwrap());
    let mut errs = Vec::new();
    let mut hs = Vec::new();
    for n in [50i64, 100, 200, 400] {
        let mut s = make(n, EA);
        let shape = mode(n, 1);
        start(&mut s, &scaled(&shape, amp));
        let steps = (t_star * s.p.fs).round() as usize;
        for _ in 0..steps {
            step(&mut s);
        }
        let q_sim = project(&s.u, &shape);
        let q_exact = duffing_displacement(&[steps as f64 * s.p.k], amp, w0sq, eps).unwrap()[0];
        errs.push((q_sim - q_exact).abs());
        hs.push(s.p.h);
    }
    let orders: Vec<f64> = (0..3)
        .map(|i| (errs[i] / errs[i + 1]).ln() / (hs[i] / hs[i + 1]).ln())
        .collect();
    println!("errors {errs:?}, orders {orders:?}");
    assert!(
        orders.iter().all(|&o| o > 1.9),
        "orders {orders:?} — not second order"
    );
    assert!(
        orders[2] < 3.2,
        "orders {orders:?} — suspiciously fast, check the oracle"
    );
}

#[test]
fn the_absolute_frequency_lands_on_the_duffing_frequency_loosely() {
    // Carried from `test_absolute_frequency_matches_duffing_loosely` (two cases): to ~0.1 %,
    // because the absolute frequency carries the linear scheme's dispersion — which is why the
    // shift bar above is the tight one.
    let (w0sq, eps) = discrete_mode_one_duffing();
    for amp in [0.005, 0.02] {
        let f = measure_mode_one(amp, EA);
        let oracle = duffing_frequency(amp, w0sq, eps).unwrap() / (2.0 * std::f64::consts::PI);
        println!("A = {amp}: measured {f} Hz, Duffing {oracle} Hz");
        assert!(
            (f - oracle).abs() <= 3e-3 * oracle,
            "A = {amp}: {f} vs {oracle}"
        );
    }
}

#[test]
fn the_pitch_rises_with_amplitude_and_lands_on_the_linear_string() {
    // Carried from `test_pitch_rises_monotonically_with_amplitude_and_lands_on_the_linear_limit` —
    // the sign check: hit it harder and it goes SHARP. As A -> 0 it lands on model #3's linear
    // fundamental, which the nonlinear model must contain.
    let freqs: Vec<f64> = [1e-5, 0.01, 0.02, 0.03, 0.05]
        .iter()
        .map(|&a| measure_mode_one(a, EA))
        .collect();
    println!("glide {freqs:?}");
    assert!(
        freqs.windows(2).all(|w| w[1] > w[0]),
        "pitch must rise with amplitude: {freqs:?}"
    );
    assert!(
        freqs[4] > 1.5 * freqs[0],
        "expected a big glide by A = 0.05, got {freqs:?}"
    );
    let linear = measure_mode_one(1e-5, 0.0);
    println!("linear limit {linear}");
    assert!(
        (freqs[0] - linear).abs() <= 1e-6 * linear,
        "{} vs {linear}",
        freqs[0]
    );
}

// == guards ======================================================================================

#[test]
fn the_pythons_three_refusals_are_the_right_variants() {
    // Carried from `test_rejects_negative_EA` and `test_rejects_bad_tension_tol_and_boundary`, at
    // the Python's values; `rejections_carry_the_python_messages` checks the prose at its own
    // fixture. `boundary="fixed"` is `boundary_ok = false`: the string parse retires with the
    // binding (§32.2).
    let fs = c() * 100.0 / L;
    let refuse = |ea: f64, tol: f64, fs: f64, n: i64, ok: bool| {
        nl::Params::new(L, T, RHO, fs, n, KAPPA, ea, 0.0, 0.0, THETA, tol, ok)
            .expect_err("must be refused")
    };
    assert_eq!(refuse(-1.0, TOL, fs, 100, true), ParamError::NegativeEa);
    assert_eq!(refuse(EA, 0.0, fs, 100, true), ParamError::BadTensionTol);
    let fixed = nl::Params::new(
        L, T, RHO, 20000.0, 50, 0.0, 0.0, 0.0, 0.0, THETA, TOL, false,
    )
    .expect_err("must be refused");
    assert_eq!(fixed, ParamError::BadBoundary);
}

#[test]
fn the_solver_telemetry_reports_a_clean_run() {
    // Carried from `test_solver_telemetry_is_exposed`: a run that never fails to converge is the
    // certificate, so the record must be observable.
    let mut s = make(100, EA);
    let ic = pluck(&s, 0.02);
    start(&mut s, &ic);
    for _ in 0..300 {
        step(&mut s);
    }
    println!(
        "telemetry: converged {}, failures {}, dT {}",
        s.converged, s.n_not_converged, s.delta_tension
    );
    assert!(s.converged);
    assert_eq!(s.n_not_converged, 0);
    assert!(s.delta_tension > 0.0);
}

#[test]
fn a_string_at_rest_stays_at_rest_with_no_energy() {
    // Carried from `test_string_at_rest_stays_at_rest`: zero stretch, zero modulation, and the
    // degenerate bracket is not entered. The native bar of the same name does not assert `E = 0`.
    let mut s = make(50, EA);
    start(&mut s, &[0.0; 51]);
    for _ in 0..20 {
        step(&mut s);
    }
    assert!(s.u.iter().all(|&v| v == 0.0));
    assert_eq!(s.delta_tension, 0.0);
    assert_eq!(s.energy(), 0.0);
}

// == the material helper (a modelling oracle, not a constraint) ==================================

/// `(rho, kappa, EA, c, c_long, EA/T)` from `string_coefficients_from_material(E=2e11,
/// radius=r, rho_v=7850, T=200)`, recorded from the Python implementation (NumPy arithmetic)
/// before §34 deleted its tests (`W:\temp\claude\tension-string\python_record.txt`).
const MATERIAL_RECORD: [(f64, [f64; 6]); 3] = [
    (
        2e-4,
        [
            0.000986460093227195,
            0.5047544651250688,
            25132.741228718343,
            450.27230698271404,
            5047.544651250688,
            125.66370614359171,
        ],
    ),
    (
        5e-4,
        [
            0.0061653755826699685,
            1.261886162812672,
            157079.63267948964,
            180.10892279308564,
            5047.544651250688,
            785.3981633974482,
        ],
    ),
    (
        1e-3,
        [
            0.024661502330679874,
            2.523772325625344,
            628318.5307179586,
            90.05446139654282,
            5047.544651250688,
            3141.592653589793,
        ],
    ),
];

#[test]
fn the_material_helper_reproduces_numpys_arithmetic_to_the_bit() {
    // The one outside referee in the file: the Python helper was NumPy arithmetic over four
    // scalars, so its six fields are an independent record of the formulas, not Rust through the
    // binding.
    for (radius, want) in MATERIAL_RECORD {
        let co = string_coefficients_from_material(2.0e11, radius, 7850.0, T).unwrap();
        let got = [co.rho, co.kappa, co.ea, co.c, co.c_long, co.ea_over_t];
        assert_eq!(got, want, "radius {radius}");
    }
}

#[test]
fn the_material_ratio_is_radius_independent() {
    // Carried from `test_material_helper_ratio_is_radius_independent` (three cases):
    // `EA/T0 = E pi r² / (rho_v pi r² c²) = (c_long/c)²` — the radius cancels exactly.
    for radius in [2e-4, 5e-4, 1e-3] {
        let co = string_coefficients_from_material(2.0e11, radius, 7850.0, T).unwrap();
        let ratio = (co.c_long / co.c).powi(2);
        assert!(
            (co.ea_over_t / ratio - 1.0).abs() < 1e-12,
            "r = {radius}: {} vs {ratio}",
            co.ea_over_t
        );
        let c_long = (2.0e11f64 / 7850.0).sqrt();
        assert!(
            (co.c_long / c_long - 1.0).abs() < 1e-12,
            "r = {radius}: {}",
            co.c_long
        );
    }
}

#[test]
fn the_material_helper_lands_in_the_real_steel_range() {
    // Carried from `test_material_helper_lands_in_the_real_steel_range`.
    let co = string_coefficients_from_material(2.0e11, 2e-4, 7850.0, T).unwrap();
    assert!(
        100.0 < co.ea_over_t && co.ea_over_t < 700.0,
        "EA/T {}",
        co.ea_over_t
    );
    assert!(
        4000.0 < co.c_long && co.c_long < 6000.0,
        "c_long {}",
        co.c_long
    );
}

#[test]
fn the_material_helper_builds_a_string_that_runs_and_conserves_energy() {
    // Carried from `test_material_helper_feeds_the_core_consistently`: the realism path is
    // exercised, not just offered.
    let co = string_coefficients_from_material(2.0e11, 5e-4, 7850.0, T).unwrap();
    let p = nl::Params::new(
        L, T, co.rho, 48000.0, 100, co.kappa, co.ea, 0.0, 0.0, THETA, TOL, true,
    )
    .expect("a real steel string constructs");
    let mut s = TensionModulatedString::new(p);
    let ic = pluck(&s, 2e-3);
    start(&mut s, &ic);
    let (drift, frac, _) = run(&mut s, 600);
    println!("steel: drift {drift:e}, fraction {frac}");
    assert!(
        frac > 1e-3,
        "a real steel string at a hard pluck should show some nonlinearity"
    );
    assert!(drift < DRIFT_TOL, "drift {drift:e}");
}

#[test]
fn the_material_helper_refuses_a_negative_modulus() {
    // Carried from `test_material_helper_rejects_nonphysical_input`; the variant and its prose.
    let err = string_coefficients_from_material(-1.0, 5e-4, 7850.0, T).unwrap_err();
    assert_eq!(err, MaterialError::NonPositive);
    assert_eq!(err.to_string(), "E, radius, rho_v, T must all be positive.");
}
