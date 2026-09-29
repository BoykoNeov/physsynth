//! The damped stiff string's validation harness — model #3, carried from
//! `tests/test_damped_string.py` (retirement plan §33) and run at the retired Python helper
//! `make_damped_string`'s parameters: `L = 1`, `T = 200`, `rho = 0.005` (so `c = 200 m/s`),
//! `kappa = 2`, `theta = 0.28`, and `fs = c N / (L lam)` so that the Courant number is exactly the
//! `lam` asked for. The pluck is at `0.137 L` and the pickup at `round(0.241 N)`, and the default
//! run is **1 s**, as the Python's `_pluck_run` had them.
//!
//! **A separate file from `string_stiff.rs`**, for the reason `string_stiff_harness.rs` is: that
//! file runs another fixture and holds `squaring_is_pow_not_multiply`, which must run unoptimised.
//! These are long trajectories, and whatever CI decides about this file's debug run must not take
//! that test with it.
//!
//! **One outside referee retires here, and it is replaced rather than dropped.** The per-mode decay
//! factor was measured by `np.polyfit` over the log-energy — a NumPy number. [`mode_decay_factor`]
//! is its transcription, and [`the_decay_fit_reproduces_numpys_polyfit`] certifies it against the
//! six factors the Python recorded before the deletion (§26.1's move). Every other number the
//! Python compared against was a closed form written in the test or already Rust through the
//! binding. The oracle-only claims went to `crates/physsynth-analysis/tests/oracles.rs`.

use physsynth_analysis::damping::{
    discrete_damped_mode_decay, discrete_damped_mode_is_underdamped, discrete_damped_mode_rate,
    loss_coefficients_from_t60,
};
use physsynth_analysis::modal::{cents, discrete_stiff_mode_frequency, mode_shape};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_damped::{DampedStiffString, ParamError, Params};
use physsynth_core::string_stiff;

/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, f0 = 100 Hz
/// `KAPPA_DEFAULT`: `B ~ 9.87e-4`, a piano-ish inharmonicity.
const KAPPA: f64 = 2.0;
/// `THETA_DEFAULT`, written here rather than imported (§29.3). `string_stiff_harness.rs` pins the
/// model's default to it, and model #3's `THETA` is that same constant re-exported.
const THETA: f64 = 0.28;

/// `wave_speed()` — 200 m/s.
fn c() -> f64 {
    (T / RHO).sqrt()
}

/// The retired `make_damped_string`: a string at Courant number `lam` via `fs = c N / (L lam)`.
fn make(n: i64, lam: f64, kappa: f64, sigma0: f64, sigma1: f64) -> DampedStiffString {
    let fs = c() * n as f64 / (L * lam);
    DampedStiffString::new(
        Params::new(L, T, RHO, fs, n, kappa, sigma0, sigma1, THETA, true)
            .expect("valid parameters"),
    )
}

/// The Python's `_pluck_run`: pluck at `0.137 L` with the model's own exciter, run `secs` through
/// the driver, read the pickup at `round(0.241 N)`.
fn pluck_run(s: &mut DampedStiffString, secs: f64) -> SimResult {
    let x = s.p.grid();
    let u0 = triangular_pluck(&x, s.p.l, 0.137 * s.p.l, 1e-3).expect("an interior pluck");
    s.set_state(&u0, &vec![0.0; x.len()]);
    let steps = (secs * s.p.fs) as usize;
    let pickup = (0.241 * s.p.n as f64).round() as usize;
    simulate(s, steps, Some(pickup), 0).expect("a lossless or passive run completes")
}

/// The retired `measure_mode_decay_factor`: the per-step **energy** decay factor `g_m` of a single
/// exact discrete eigenvector `sin(m pi x / L)` at amplitude 1e-3, from a least-squares line through
/// `ln E^n` over the interior window `[int(0.1 steps), int(0.7 steps)]` (skipping the start, where
/// the lossless Taylor `u^{-1}` is slightly inconsistent under damping), clipped to where the energy
/// is still above `1e-13 E^0`, and refusing fewer than eight points. Returns `exp(slope)`.
///
/// The slope is `np.polyfit(idx, log(e), 1)[0]`, written as the centred normal equation. It is not
/// NumPy's algorithm (an SVD least-squares on a scaled Vandermonde), so agreement is a tolerance,
/// certified in [`the_decay_fit_reproduces_numpys_polyfit`].
fn mode_decay_factor(m: i64, n: i64, kappa: f64, sigma0: f64, sigma1: f64, steps: usize) -> f64 {
    let mut s = make(n, 1.0, kappa, sigma0, sigma1);
    let x = s.p.grid();
    let u0: Vec<f64> = mode_shape(&x, L, m).iter().map(|v| v * 1e-3).collect();
    s.set_state(&u0, &vec![0.0; x.len()]);
    let e = simulate(&mut s, steps, None, 0)
        .expect("a passive run completes")
        .energy;

    let i0 = (0.1 * steps as f64) as usize;
    let i1 = (0.7 * steps as f64) as usize;
    let floor = e[0] * 1e-13;
    let idx: Vec<usize> = (i0..=i1).filter(|&i| e[i] > floor).collect();
    assert!(
        idx.len() >= 8,
        "decay window too short / energy hit the roundoff floor"
    );
    let xs: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
    let ys: Vec<f64> = idx.iter().map(|&i| e[i].ln()).collect();
    let len = xs.len() as f64;
    let xm = xs.iter().sum::<f64>() / len;
    let ym = ys.iter().sum::<f64>() / len;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (xi, yi) in xs.iter().zip(&ys) {
        sxy += (xi - xm) * (yi - ym);
        sxx += (xi - xm) * (xi - xm);
    }
    (sxy / sxx).exp()
}

/// `np.max`: NaN-propagating, unlike `fold(_, f64::max)` (§29.3).
fn nan_max(it: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for v in it {
        if v.is_nan() {
            return f64::NAN;
        }
        m = m.max(v);
    }
    m
}

/// The worst one-step energy rise over `E^0`, NaN-propagating.
fn worst_rise(energy: &[f64]) -> f64 {
    nan_max(energy.windows(2).map(|w| w[1] - w[0])) / energy[0]
}

// -- energy: conservation, the stiff-string reduction, passivity --------------------------------

#[test]
fn with_both_losses_zero_the_energy_is_conserved() {
    // Carried from `test_lossless_energy_conserved_quick`. sigma0 = sigma1 = 0 is model #2's
    // conservative scheme, so the inherited bar applies. No native bar ran `DampedStiffString`
    // lossless before this one: `string_stiff.rs`'s lossless bar runs `StiffString`.
    let mut s = make(100, 1.0, KAPPA, 0.0, 0.0);
    let res = pluck_run(&mut s, 1.0);
    let drift = res.energy_drift();
    println!("lossless drift: {drift:e}");
    assert!(drift < DRIFT_TOL, "drift {drift:e}");
    assert!(
        res.energy.iter().all(|&e| e > 0.0),
        "the energy went non-positive"
    );
}

#[test]
fn lossless_energy_is_conserved_and_positive_at_every_theta() {
    // NOT carried from Python: added in §33.4 (the human's call). Every Python test built the
    // damped string at the default theta, and so did every native bar, so a theta hard-coded to
    // 0.28 in the energy or in the step was exact everywhere it was run and seen by nothing in the
    // workspace. The stiff string's own sweep (`string_stiff_harness.rs`) runs a separate
    // transcription. Same sweep here: theta = 1/4 is the zero-positivity-margin case.
    for theta in [0.25, THETA, 0.5] {
        let fs = c() * 120.0 / L;
        let mut s = DampedStiffString::new(
            Params::new(L, T, RHO, fs, 120, KAPPA, 0.0, 0.0, theta, true)
                .expect("valid parameters"),
        );
        let res = pluck_run(&mut s, 2.0);
        let drift = res.energy_drift();
        println!("theta = {theta}: drift {drift:e}");
        assert!(drift < DRIFT_TOL, "theta = {theta}: drift {drift:e}");
        assert!(
            res.energy.iter().all(|&e| e > 0.0),
            "theta = {theta}: the energy went non-positive"
        );
    }
}

#[test]
fn sigma1_zero_is_the_stiff_string_bit_for_bit_at_the_pythons_fixture() {
    // Carried from `test_sigma1_zero_reduces_to_stiff_string_bit_for_bit`, at its own fixture
    // (N = 128, lam = 1, kappa = 2, sigma = 3, 1,500 steps through the driver): the energy trace
    // AND the pickup trace, exactly. `string_stiff.rs` asserts the same anchor state-by-state at
    // its own fixture; this one is the Python's.
    let n = 128i64;
    let fs = c() * n as f64 / L;
    let mut d = DampedStiffString::new(
        Params::new(L, T, RHO, fs, n, KAPPA, 3.0, 0.0, THETA, true).expect("valid"),
    );
    let mut s = string_stiff::StiffString::new(
        string_stiff::Params::new(L, T, RHO, fs, n, KAPPA, 3.0, THETA, true).expect("valid"),
    );
    let x = d.p.grid();
    let u0 = triangular_pluck(&x, L, 0.137 * L, 1e-3).expect("an interior pluck");
    d.set_state(&u0, &vec![0.0; x.len()]);
    s.set_state(&u0, &vec![0.0; x.len()]);
    let pickup = (0.241 * n as f64).round() as usize;
    let rd = simulate(&mut d, 1500, Some(pickup), 0).expect("runs");
    let rs = simulate(&mut s, 1500, Some(pickup), 0).expect("runs");
    assert_eq!(rd.energy, rs.energy, "energy diverged from StiffString");
    assert_eq!(rd.output, rs.output, "pickup diverged from StiffString");
}

#[test]
fn both_losses_together_are_passive_for_a_broadband_pluck() {
    // Carried from `test_passivity_broadband`: the discrete losses
    // `-2 sigma0 ||dt. u||^2 - 2 sigma1 ||dx+ dt. u||^2 <= 0`, so no step may gain energy.
    let mut s = make(128, 1.0, KAPPA, 2.0, 1e-4);
    let res = pluck_run(&mut s, 1.0);
    let rise = worst_rise(&res.energy);
    println!("worst step rise / E0: {rise:e}");
    assert!(rise <= 1e-12, "the energy rose by {rise:e} E0 in one step");
    assert!(res.energy.iter().all(|e| e.is_finite()));
    assert!(
        *res.energy.last().unwrap() < res.energy[0],
        "a lossy run must actually lose energy"
    );
}

#[test]
fn the_frequency_dependent_loss_alone_is_passive() {
    // Carried from `test_passivity_sigma1_only`: sigma0 = 0 isolates the new term, so a failure
    // here points straight at the sigma1 discretisation.
    let mut s = make(128, 1.0, KAPPA, 0.0, 5e-4);
    let res = pluck_run(&mut s, 1.0);
    let rise = worst_rise(&res.energy);
    println!("worst step rise / E0: {rise:e}");
    assert!(rise <= 1e-12, "the energy rose by {rise:e} E0 in one step");
    assert!(
        *res.energy.last().unwrap() < res.energy[0],
        "a lossy run must actually lose energy"
    );
}

// -- per-mode decay: the money test ---------------------------------------------------------------

/// The six decay factors the retired Python measured with `np.polyfit` at the money test's fixture
/// (`N = 128`, `sigma0 = 2`, `sigma1 = 1e-4`, 15,000 steps), recorded 2026-09-29 with the wheel
/// freshly reinstalled (`W:\temp\claude\damped-string\python_record.txt`).
const NUMPY_DECAY: [(i64, f64); 6] = [
    (1, 0.999843708155008),
    (2, 0.9998435615646393),
    (4, 0.9998429596308299),
    (8, 0.9998406661905631),
    (16, 0.999833163553775),
    (32, 0.9998250313299294),
];

#[test]
fn the_decay_fit_reproduces_numpys_polyfit() {
    // NOT a physics bar: the certificate for [`mode_decay_factor`], the transcription of the one
    // NumPy number this file retired. The energy it fits is Rust on both sides, so what is compared
    // is the least-squares slope alone — in log space, where the physics bars read it.
    let mut worst = 0.0f64;
    for (m, g_numpy) in NUMPY_DECAY {
        let g = mode_decay_factor(m, 128, KAPPA, 2.0, 1e-4, 15000);
        let rel = (g.ln() - g_numpy.ln()).abs() / g_numpy.ln().abs();
        println!("mode {m}: native {g:?}, numpy {g_numpy:?}, rate-relative {rel:e}");
        worst = worst.max(rel);
        assert!(rel < 1e-9, "mode {m}: fit off NumPy's by {rel:e}");
    }
    println!("worst rate-relative gap to polyfit: {worst:e}");
}

#[test]
fn every_modes_decay_lands_on_the_schemes_own_oracle() {
    // Carried from `test_per_mode_decay_matches_discrete_oracle`, the money test: a single mode's
    // measured per-step energy factor lands on the scheme's closed-form `g_m = c/a` to a
    // rate-relative 5e-4. The oracle accounts for the theta-scheme's rate suppression exactly, so
    // EVERY mode checks, out to 32 — the one where a continuum `beta^2` in place of the discrete `p^2`
    // shows. Each mode is first asserted underdamped, which the `g_m` oracle assumes.
    let (n, sigma0, sigma1) = (128i64, 2.0, 1e-4);
    let k = make(n, 1.0, KAPPA, sigma0, sigma1).p.k;
    let mut worst = 0.0f64;
    for m in [1i64, 2, 4, 8, 16, 32] {
        assert!(
            discrete_damped_mode_is_underdamped(c(), L, n, KAPPA, k, THETA, sigma0, sigma1, m),
            "mode {m} is not underdamped (the g_m oracle assumes a complex-conjugate pair)"
        );
        let g_meas = mode_decay_factor(m, n, KAPPA, sigma0, sigma1, 15000);
        let g_or = discrete_damped_mode_decay(c(), L, n, KAPPA, k, THETA, sigma0, sigma1, m);
        let rel = (g_meas.ln() - g_or.ln()).abs() / g_or.ln().abs();
        println!("mode {m}: decay rate off the oracle by {rel:e}");
        assert!(rel < 5e-4, "mode {m}: decay rate off by {rel:e}");
        worst = worst.max(rel);
    }
    println!("worst: {worst:e}");
}

#[test]
fn sigma1_makes_high_partials_die_faster() {
    // Carried from `test_sigma1_makes_high_partials_die_faster`: the cure, made falsifiable.
    //   * oracle: over [1..16] the per-mode rate RISES with sigma1 > 0 and FALLS with sigma1 = 0
    //     (model #2's backwards artifact), same sigma0, so the flip is sigma1's alone. Not across
    //     the whole spectrum: the rate turns over past ~m = 32 (numerator ~p^2 against the theta
    //     denominator ~p^4). The band is a DECAY-RATE band, not the pitch horizon —
    //     `docs/dev/resolution-horizon-plan.md` §7.1.
    //   * simulation: the measured rate at m = 16 exceeds m = 2 by more than 2%.
    let n = 128i64;
    let k = make(n, 1.0, KAPPA, 2.0, 1e-4).p.k;
    let rates = |s1: f64| -> Vec<f64> {
        (1..=16)
            .map(|m| discrete_damped_mode_rate(c(), L, n, KAPPA, k, THETA, 2.0, s1, m))
            .collect()
    };
    let with = rates(1e-4);
    let without = rates(0.0);
    assert!(
        with.windows(2).all(|w| w[1] - w[0] > 0.0),
        "sigma1 > 0 rate not rising in-band: {with:?}"
    );
    assert!(
        without.windows(2).all(|w| w[1] - w[0] < 0.0),
        "sigma1 = 0 not falling (model #2): {without:?}"
    );

    let measured = |m: i64| -mode_decay_factor(m, n, KAPPA, 2.0, 1e-4, 15000).ln() / k;
    let (lo, hi) = (measured(2), measured(16));
    println!(
        "measured rate: mode 2 {lo}, mode 16 {hi}, ratio {}",
        hi / lo
    );
    assert!(
        hi > lo * 1.02,
        "high not faster: rate16 = {hi} <= rate2 = {lo}"
    );
}

// -- the T60 mapping ------------------------------------------------------------------------------

#[test]
fn two_t60_targets_invert_to_losses_that_reproduce_them() {
    // Carried from `test_T60_mapping_roundtrip`: specify T60 = 4 s at mode 1 and 0.8 s at mode 20,
    // invert through the CONTINUUM dispersion to (sigma0, sigma1), simulate those two modes, and
    // recover the targets within 4%. The gap is the documented continuum approximation — a physics
    // demo, not a machine-precision test (that is the money test above). kappa = 2 runs the
    // mapping's stiff branch, which no other native bar reaches.
    let n = 256i64;
    let k = make(n, 1.0, KAPPA, 0.0, 0.0).p.k;
    let (m1, m2) = (1i64, 20i64);
    let f1 = discrete_stiff_mode_frequency(c(), L, n, KAPPA, k, m1, THETA);
    let f2 = discrete_stiff_mode_frequency(c(), L, n, KAPPA, k, m2, THETA);
    let (t60_1, t60_2) = (4.0, 0.8);
    let (sigma0, sigma1) =
        loss_coefficients_from_t60(c(), L, KAPPA, f1, t60_1, f2, t60_2).expect("a solvable pair");
    println!("sigma0 = {sigma0}, sigma1 = {sigma1}");
    assert!(sigma0 > 0.0 && sigma1 > 0.0);
    for (m, target) in [(m1, t60_1), (m2, t60_2)] {
        let g = mode_decay_factor(m, n, KAPPA, sigma0, sigma1, 20000);
        // Amplitude 60 dB drop (x1e-3): g^(n/2) = 1e-3 -> n = 2 ln(1e-3) / ln(g); T60 = n k.
        let t60 = 2.0 * (1e-3f64).ln() / g.ln() * k;
        let rel = (t60 - target).abs() / target;
        println!("mode {m}: T60 {t60} s vs target {target} s ({rel:e})");
        assert!(rel < 0.04, "mode {m}: T60 {t60} s vs target {target} s");
    }
}

// -- partials, stability, construction --------------------------------------------------------

#[test]
fn light_damping_leaves_every_partial_on_the_undamped_oracle() {
    // Carried from `test_partials_unmoved_by_light_damping`: damping shifts an oscillation
    // frequency only at O(damping^2), so a lightly damped pluck's partials still land on the
    // undamped stiff oracle, within a cent. Eight partials is detectability, not a horizon — the
    // reference is the scheme's own discrete oracle.
    let n = 128i64;
    let mut s = make(n, 1.0, KAPPA, 0.5, 1e-5);
    let res = pluck_run(&mut s, 2.0);
    let oracle: Vec<f64> = (1..=8)
        .map(|m| discrete_stiff_mode_frequency(c(), L, n, KAPPA, s.p.k, m, THETA))
        .collect();
    let out = res.output.expect("a pickup was requested");
    let detected = measure_partials_near(&out, res.fs, &oracle, None);
    assert!(
        detected.iter().all(|f| !f.is_nan()),
        "a partial was not detected: {detected:?}"
    );
    let err = nan_max(
        detected
            .iter()
            .zip(&oracle)
            .map(|(&d, &o)| cents(d, o).abs()),
    );
    println!("worst partial vs the undamped oracle: {err:e} cents");
    assert!(err < 1.0, "worst partial-vs-oracle error {err} cents");
}

#[test]
fn there_is_no_courant_limit_even_with_heavy_stiffness_and_both_losses() {
    // Carried from `test_no_nan_unconditional` (4 cases): kappa = 5, both losses, at
    // lam = 4, 2, 1 and 0.5, over 0.5 s — every run completes with pickup and energy finite.
    for lam in [4.0, 2.0, 1.0, 0.5] {
        let mut s = make(100, lam, 5.0, 3.0, 2e-4);
        let res = pluck_run(&mut s, 0.5);
        assert!(
            res.output
                .expect("a pickup was requested")
                .iter()
                .all(|v| v.is_finite()),
            "lam = {lam}: the pickup went non-finite"
        );
        assert!(
            res.energy.iter().all(|e| e.is_finite()),
            "lam = {lam}: the energy went non-finite"
        );
    }
}

#[test]
fn the_ten_invalid_parameter_sets_are_rejected() {
    // Carried from `test_invalid_parameters_rejected`, at its own base (the binding's defaults for
    // everything it did not name: kappa = sigma0 = sigma1 = 0, theta = THETA_DEFAULT). Each is
    // asserted to be rejected for the RIGHT reason, which `pytest.raises(ValueError)` did not ask.
    #[allow(clippy::too_many_arguments)]
    let base =
        |l: f64, t: f64, rho: f64, n: i64, kappa: f64, s0: f64, s1: f64, theta: f64, ok: bool| {
            Params::new(l, t, rho, 20000.0, n, kappa, s0, s1, theta, ok)
        };
    let (l, t, rho, n) = (1.0, 200.0, 0.005, 100i64);
    // The control: the base itself constructs.
    assert!(base(l, t, rho, n, 0.0, 0.0, 0.0, THETA, true).is_ok());
    let cases: [(&str, Result<Params, ParamError>, ParamError); 10] = [
        (
            "rho = -1",
            base(l, t, -1.0, n, 0.0, 0.0, 0.0, THETA, true),
            ParamError::NonPositive,
        ),
        (
            "T = 0",
            base(l, 0.0, rho, n, 0.0, 0.0, 0.0, THETA, true),
            ParamError::NonPositive,
        ),
        (
            "L = -2",
            base(-2.0, t, rho, n, 0.0, 0.0, 0.0, THETA, true),
            ParamError::NonPositive,
        ),
        (
            "kappa = -0.1",
            base(l, t, rho, n, -0.1, 0.0, 0.0, THETA, true),
            ParamError::NegativeKappa,
        ),
        (
            "sigma0 = -0.1",
            base(l, t, rho, n, 0.0, -0.1, 0.0, THETA, true),
            ParamError::NegativeSigma0,
        ),
        (
            "sigma1 = -0.1",
            base(l, t, rho, n, 0.0, 0.0, -0.1, THETA, true),
            ParamError::NegativeSigma1,
        ),
        (
            "N = 1",
            base(l, t, rho, 1, 0.0, 0.0, 0.0, THETA, true),
            ParamError::TooFewSegments,
        ),
        (
            "theta = 0",
            base(l, t, rho, n, 0.0, 0.0, 0.0, 0.0, true),
            ParamError::BadTheta(0.0),
        ),
        (
            "theta = 1.5",
            base(l, t, rho, n, 0.0, 0.0, 0.0, 1.5, true),
            ParamError::BadTheta(1.5),
        ),
        // `boundary="clamped"`: the STRING parse lives only in the binding and retires with it.
        // What the core owns is the refusal, which it takes as `boundary_ok = false`.
        (
            "boundary not supported",
            base(l, t, rho, n, 0.0, 0.0, 0.0, THETA, false),
            ParamError::BadBoundary,
        ),
    ];
    for (name, got, want) in cases {
        assert_eq!(got.expect_err(name), want, "{name}");
    }
}

#[test]
fn a_courant_number_above_one_is_accepted_and_reported() {
    // Carried from `test_lambda_above_one_accepted`: the implicit scheme must NOT reject lam > 1,
    // and the lam it reports is the one the fixture built.
    let s = make(100, 2.5, KAPPA, 1.0, 1e-4);
    assert!(
        (s.p.lam - 2.5).abs() <= 1e-9 * 2.5,
        "lam = {}, want 2.5",
        s.p.lam
    );
}

// -- the start-up --------------------------------------------------------------------------------

#[test]
fn the_start_up_is_the_lossless_taylor_step_exactly_under_both_losses() {
    // NOT carried from Python: planted in §33.4. Dropping the 1/2 in
    // `u^{-1} = u^0 - k v^0 + 1/2 k^2 L u^0` passed every physics bar in the workspace. It was seen
    // by eight tests, every one a comparison with a copy: the stiff twin anchors here and in
    // `string_stiff.rs`, the two strings that reduce to this one bit for bit (`string_geometric`,
    // `string_nonlinear`), the recorded-polyfit certificate, and three viewer freezes. §32's remedy, now with both losses: on an exact eigenmode
    // `L -> -Q` and `D2 -> -p^2`, so one step is scalar algebra. With `a = k^2 Q` and
    // `s = (sigma0 + sigma1 p^2) k`, the step is `(1 + theta a + s) u^1 =
    // (2 - (1 - 2 theta) a) u^0 - (1 + theta a - s) u^{-1}`, and the start is the LOSSLESS Taylor
    // one (neither loss enters it, `string_damped::initial_previous`), so
    //   at rest:     u^{-1} = (1 - a/2) u^0,  u^1 = R u^0,
    //                R = [2 - (1 - 2 theta) a - (1 + theta a - s)(1 - a/2)] / (1 + theta a + s);
    //   launched:    u^0 = 0, u^{-1} = -k v^0,  u^1 = (1 + theta a - s) / (1 + theta a + s) k v^0.
    // The defect moves R by `(1 + theta a - s) (a/2) / (1 + theta a + s)`, about `a/2`.
    let (m, sigma0, sigma1) = (3usize, 2.0, 1e-4);
    for lam in [1.0, 4.0] {
        let mut s = make(100, lam, KAPPA, sigma0, sigma1);
        let x = s.p.grid();
        let (h, k) = (s.p.h, s.p.k);
        let sin = (m as f64 * std::f64::consts::PI / (2.0 * s.p.n as f64)).sin();
        let p2 = 4.0 / (h * h) * sin * sin;
        let a = k * k * (c() * c() * p2 + KAPPA * KAPPA * p2 * p2);
        let sk = (sigma0 + sigma1 * p2) * k;
        let phi = mode_shape(&x, L, m as i64);

        let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
        s.set_state(&u0, &vec![0.0; x.len()]);
        s.step();
        let r = (2.0 - (1.0 - 2.0 * THETA) * a - (1.0 + THETA * a - sk) * (1.0 - a / 2.0))
            / (1.0 + THETA * a + sk);
        let worst = nan_max((1..x.len() - 1).map(|i| (s.u[i] - r * u0[i]).abs() / 1e-3));
        println!(
            "lam = {lam}: a = {a:e}, defect size ~{:e}, at rest off by {worst:e}",
            a / 2.0
        );
        assert!(worst < 1e-12, "lam = {lam}: u^1 off R u^0 by {worst:e}");

        s.set_state(&vec![0.0; x.len()], &phi);
        s.step();
        let g = (1.0 + THETA * a - sk) / (1.0 + THETA * a + sk) * k;
        let worst = nan_max((1..x.len() - 1).map(|i| (s.u[i] - g * phi[i]).abs() / k));
        println!("lam = {lam}: launched off by {worst:e} of k v0");
        assert!(worst < 1e-12, "lam = {lam}: launched u^1 off by {worst:e}");
    }
}
