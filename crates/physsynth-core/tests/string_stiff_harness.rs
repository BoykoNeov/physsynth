//! The stiff string's validation harness — model #2, carried from `tests/test_stiff_string.py`
//! (retirement plan §32) and run at the retired Python helper `make_stiff_string`'s parameters:
//! `L = 1`, `T = 200`, `rho = 0.005` (so `c = 200 m/s`), `kappa = 2`, `theta = 0.28`, and
//! `fs = c N / (L lam)` so that the Courant number is exactly the `lam` asked for. The pluck is at
//! `0.137 L` and the pickup at `round(0.241 N)`, as the Python's `_pluck_run` had them.
//!
//! **A separate file from `string_stiff.rs`**, which builds the string at `fs = 44100` and
//! `kappa = 1.5` and holds `squaring_is_pow_not_multiply` — a test that must run unoptimised too.
//! These are long trajectories; keeping them apart keeps that test's debug run whatever CI decides
//! about this file's.
//!
//! **No outside referee retires here**, as in §31. Every number the Python compared against was a
//! closed form written in the test (`exp(-2 sigma t)`, the stretched law's `B`, the one-cent and
//! 0.05-cent bars) or already ran through the Rust binding: the exciter, `simulate`, the spectrum
//! detector, every oracle. NumPy's only role was arithmetic on those results. The oracle-only
//! claims went to `crates/physsynth-analysis/tests/oracles.rs`, the operator ones to `ops.rs`.

use physsynth_analysis::dispersion::{phase_velocity, stiff_dispersion_frequencies};
use physsynth_analysis::modal::{
    cents, discrete_stiff_mode_frequency, inharmonicity_b, mode_shape, stiff_harmonic_frequencies,
};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_stiff::{ParamError, Params, StiffString};

/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, f0 = 100 Hz
/// `KAPPA_DEFAULT`: `B = pi^2 kappa^2 / (c^2 L^2) ~ 9.87e-4`, a piano-ish inharmonicity.
const KAPPA: f64 = 2.0;
/// `THETA_DEFAULT`, written here rather than imported (§29.3: a bar about a constant must not read
/// the constant from the model it is checking). `string_stiff.rs` pins the model's own value.
const THETA: f64 = 0.28;

/// `wave_speed()` — 200 m/s.
fn c() -> f64 {
    (T / RHO).sqrt()
}

/// The retired `make_stiff_string`: a string at Courant number `lam` via `fs = c N / (L lam)`.
fn make(n: i64, lam: f64, kappa: f64, sigma: f64, theta: f64) -> StiffString {
    let fs = c() * n as f64 / (L * lam);
    StiffString::new(
        Params::new(L, T, RHO, fs, n, kappa, sigma, theta, true).expect("valid parameters"),
    )
}

/// The Python's `_pluck_run`: pluck at `0.137 L` with the model's own exciter, run `secs` through
/// the driver, read the pickup at `round(0.241 N)`.
fn pluck_run(s: &mut StiffString, secs: f64) -> SimResult {
    let x = s.p.grid();
    let u0 = triangular_pluck(&x, s.p.l, 0.137 * s.p.l, 1e-3).expect("an interior pluck");
    s.set_state(&u0, &vec![0.0; x.len()]);
    let steps = (secs * s.p.fs) as usize;
    let pickup = (0.241 * s.p.n as f64).round() as usize;
    simulate(s, steps, Some(pickup), 0).expect("a lossless or passive run completes")
}

/// The pickup signal alone, stepped by hand — for runs whose energy nothing asserts, where the
/// driver's per-step energy would roughly double the cost. The same trajectory as [`pluck_run`]'s.
fn pluck_signal(s: &mut StiffString, secs: f64) -> Vec<f64> {
    let x = s.p.grid();
    let u0 = triangular_pluck(&x, s.p.l, 0.137 * s.p.l, 1e-3).expect("an interior pluck");
    s.set_state(&u0, &vec![0.0; x.len()]);
    let steps = (secs * s.p.fs) as usize;
    let pickup = (0.241 * s.p.n as f64).round() as usize;
    let mut out = Vec::with_capacity(steps + 1);
    out.push(s.u[pickup]);
    for _ in 0..steps {
        s.step();
        out.push(s.u[pickup]);
    }
    out
}

/// The retired `measure_stiff_mode_frequencies`: the frequency of each single mode, read off its
/// MODAL COORDINATE `q = <u, phi_m>` rather than a point pickup (a point can sit on a node of the
/// mode), with the peak search anchored at the scheme's own discrete oracle. `sin(m pi x / L)` is
/// an exact discrete eigenvector of the stiff operator too (the biharmonic block is `D2 @ D2`), so
/// `q` is a clean cosine at the stiff discrete frequency.
fn measure_stiff_mode_frequencies(modes: &[i64], n: i64, lam: f64, secs: f64) -> Vec<f64> {
    modes
        .iter()
        .map(|&m| {
            let mut s = make(n, lam, KAPPA, 0.0, THETA);
            let x = s.p.grid();
            let phi = mode_shape(&x, L, m);
            let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
            s.set_state(&u0, &vec![0.0; x.len()]);
            let dot = |u: &[f64]| u.iter().zip(&phi).map(|(a, b)| a * b).sum::<f64>();
            let steps = (secs * s.p.fs) as usize;
            let mut q = Vec::with_capacity(steps + 1);
            q.push(dot(&s.u));
            for _ in 0..steps {
                s.step();
                q.push(dot(&s.u));
            }
            let oracle = discrete_stiff_mode_frequency(c(), L, n, KAPPA, s.p.k, m, THETA);
            measure_partials_near(&q, s.p.fs, &[oracle], None)[0]
        })
        .collect()
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

// -- energy: conservation and passivity ----------------------------------------------------------

#[test]
fn lossless_energy_is_conserved_across_stiffness_and_courant_number() {
    // Carried from `test_energy_conserved_across_kappa_and_lambda` (16 cases), with
    // `test_energy_conserved_quick` (kappa = 2, lam = 1 over 1 s, energy positive) and
    // `test_no_nan_including_explicit_forbidden_lambda` (kappa = 5 at every lam over 0.5 s,
    // output and energy finite) as its subsets: every run here is 2 s, and all three claims are
    // asserted on every one of them. The identity is algebraic, so it must hold at every kappa and
    // every lam -- including lam = 2 and 4, which the explicit stiff scheme could not run at all.
    let mut worst = 0.0f64;
    for kappa in [0.0, 0.5, 2.0, 5.0] {
        for lam in [1.0, 0.5, 2.0, 4.0] {
            let mut s = make(100, lam, kappa, 0.0, THETA);
            let res = pluck_run(&mut s, 2.0);
            let drift = res.energy_drift();
            assert!(
                drift < DRIFT_TOL,
                "kappa = {kappa}, lam = {lam}: drift {drift:e}"
            );
            assert!(
                res.energy.iter().all(|&e| e > 0.0 && e.is_finite()),
                "kappa = {kappa}, lam = {lam}: energy not finite and positive at every step"
            );
            let out = res.output.expect("a pickup was requested");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "kappa = {kappa}, lam = {lam}: the pickup went non-finite"
            );
            worst = worst.max(drift);
        }
    }
    println!("worst lossless drift over the kappa x lam sweep: {worst:e}");
}

#[test]
fn lossless_energy_is_conserved_and_positive_at_every_theta() {
    // Carried from `test_energy_conserved_across_theta`. Conservation is exact for every
    // theta > 0; theta = 1/4 is the zero-positivity-margin case (the stabiliser term vanishes), so
    // the energy staying strictly positive there, for a real pluck, is a claim of its own.
    for theta in [0.25, THETA, 0.5] {
        let mut s = make(120, 1.0, KAPPA, 0.0, theta);
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
fn the_stiff_strings_own_loss_is_passive() {
    // Carried from `test_passivity_monotonic_decrease`. The model's OWN sigma: the native
    // passivity bar in `string_stiff.rs` runs on `DampedStiffString`, a separate transcription.
    let mut s = make(100, 1.0, KAPPA, 5.0, THETA);
    let res = pluck_run(&mut s, 2.0);
    let e0 = res.energy[0];
    let worst_rise = nan_max(res.energy.windows(2).map(|w| w[1] - w[0]));
    println!("worst step rise / E0: {:e}", worst_rise / e0);
    assert!(
        worst_rise <= 1e-12 * e0,
        "the energy rose by {worst_rise:e} in one step"
    );
    assert!(
        *res.energy.last().unwrap() < e0,
        "a lossy run must actually lose energy"
    );
}

#[test]
fn a_low_mode_decays_at_two_sigma() {
    // Carried from `test_decay_rate_matches_2sigma_low_mode`. The theta-scheme decays mode m at
    // `2 sigma (1 - theta Q k^2)`; for mode 1, `Q k^2 << 1`, so the rate is `2 sigma` to a fraction
    // of a percent even with stiffness. High modes under-damp hard -- the documented property
    // model #3 cures -- which is why this is a single-mode start and not a pluck.
    let (sigma, secs) = (4.0, 1.0);
    let mut s = make(100, 1.0, KAPPA, sigma, THETA);
    let x = s.p.grid();
    let u0: Vec<f64> = mode_shape(&x, L, 1).iter().map(|v| v * 1e-3).collect();
    s.set_state(&u0, &vec![0.0; x.len()]);
    let steps = (secs * s.p.fs) as usize;
    let res = simulate(&mut s, steps, None, 0).expect("a passive run completes");
    let measured = res.energy[steps] / res.energy[0];
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    println!("decay rate error: {rel:e}");
    assert!(
        rel < 0.01,
        "decay rate off by {rel:e} (got {measured:e}, want {expected:e})"
    );
}

// -- the start-up --------------------------------------------------------------------------------

#[test]
fn the_start_up_is_the_consistent_taylor_step_exactly() {
    // NOT carried from Python: planted in §32.4. Dropping the 1/2 in
    // `u^{-1} = u^0 - k v^0 + 1/2 k^2 L u^0` passed every bar this batch carried -- energy is
    // conserved from any start, and a frequency does not care about the phase -- and was seen only
    // by `sigma1_zero_is_the_stiff_string_exactly`, which compares two transcriptions rather than
    // either with the physics. §30.4's remedy, for this scheme: on an exact eigenmode, `L -> -lam`
    // and one step is scalar algebra. With `a = k^2 lam` and the string at rest,
    //   (1 + theta a) u^1 = [2 - (1 - 2 theta) a - (1 - a/2)(1 + theta a)] u^0
    // so `u^1 - u^{-1} = theta a^2 / (1 + theta a) u^0` -- second order in `a`, where the defect
    // makes it first order. And launched from rest position, `u^1 = -u^{-1} = k v^0`: the centred
    // velocity is `v^0` exactly.
    let m = 3usize;
    for lam in [1.0, 4.0] {
        let mut s = make(100, lam, KAPPA, 0.0, THETA);
        let x = s.p.grid();
        let (h, k) = (s.p.h, s.p.k);
        let sin = (m as f64 * std::f64::consts::PI / (2.0 * s.p.n as f64)).sin();
        let p2 = 4.0 / (h * h) * sin * sin;
        let a = k * k * (c() * c() * p2 + KAPPA * KAPPA * p2 * p2);
        let phi = mode_shape(&x, L, m as i64);

        let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
        s.set_state(&u0, &vec![0.0; x.len()]);
        let u_prev = s.u_prev.clone();
        s.step();
        let coeff = THETA * a * a / (1.0 + THETA * a);
        let worst = nan_max(
            (1..x.len() - 1).map(|i| ((s.u[i] - u_prev[i]) - coeff * u0[i]).abs() / (coeff * 1e-3)),
        );
        println!("lam = {lam}: a = {a:e}, at-rest identity off by {worst:e} of its size");
        assert!(
            worst < 1e-8,
            "lam = {lam}: u^1 - u^-1 off the identity by {worst:e}"
        );

        let v0: Vec<f64> = phi.clone();
        s.set_state(&vec![0.0; x.len()], &v0);
        let u_prev = s.u_prev.clone();
        s.step();
        let worst =
            nan_max((1..x.len() - 1).map(|i| ((s.u[i] - u_prev[i]) / (2.0 * k) - v0[i]).abs()));
        println!("lam = {lam}: centred launch velocity off by {worst:e}");
        assert!(
            worst < 1e-12,
            "lam = {lam}: centred launch velocity off by {worst:e}"
        );
    }
}

// -- the stretched partials ----------------------------------------------------------------------

#[test]
fn every_partial_of_a_pluck_lands_on_the_schemes_own_oracle() {
    // Carried from `test_partials_match_discrete_oracle`: the implementation test. The oracle is
    // the scheme's own discrete dispersion (it depends on theta), so the resolution horizon is
    // irrelevant here; eight is the detectability limit of a pluck at this grid.
    let mut s = make(128, 1.0, KAPPA, 0.0, THETA);
    let res = pluck_run(&mut s, 2.0);
    let oracle: Vec<f64> = (1..=8)
        .map(|m| discrete_stiff_mode_frequency(c(), L, 128, KAPPA, s.p.k, m, THETA))
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
    println!("worst partial vs its discrete oracle: {err:e} cents");
    assert!(err < 0.05, "worst partial-vs-oracle error {err} cents");
}

#[test]
fn the_fitted_inharmonicity_tracks_kappa_squared() {
    // Carried from `test_B_tracks_kappa_squared`, the money test: `B` fitted from SIMULATED
    // partials across a kappa sweep tracks `pi^2 kappa^2 / (c^2 L^2)` in sign and in scale.
    // Numerical dispersion pulls high partials flat, the opposite of bending, so the fit lands a
    // little UNDER the truth -- one-directional and understood, hence the asymmetric band.
    let f0 = c() / (2.0 * L);
    let kappas = [1.0, 2.0, 3.0, 4.0, 5.0];
    let n_partials = 10usize;
    let mut b_fit = Vec::new();
    let mut b_true = Vec::new();
    for &kappa in &kappas {
        let mut s = make(512, 1.0, kappa, 0.0, THETA);
        let out = pluck_signal(&mut s, 1.0);
        let oracle: Vec<f64> = (1..=n_partials as i64)
            .map(|m| discrete_stiff_mode_frequency(c(), L, 512, kappa, s.p.k, m, THETA))
            .collect();
        let det = measure_partials_near(&out, s.p.fs, &oracle, None);
        assert!(
            det.iter().all(|f| !f.is_nan()),
            "a partial undetected at kappa = {kappa}"
        );
        // (f_n / (n f0))^2 - 1 = B n^2  ->  least-squares slope through the origin.
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &f) in det.iter().enumerate() {
            let n = (i + 1) as f64;
            num += n * n * ((f / (n * f0)).powi(2) - 1.0);
            den += n.powi(4);
        }
        b_fit.push(num / den);
        b_true.push(inharmonicity_b(c(), L, kappa));
    }
    let ratio: Vec<f64> = b_fit.iter().zip(&b_true).map(|(f, t)| f / t).collect();
    println!("B_fit / B_true = {ratio:?}");

    // (a) Sign and absolute scale: within a few percent, biased low, never sharp of the truth.
    assert!(
        ratio.iter().all(|&r| r > 0.92 && r < 1.02),
        "B_fit / B_true = {ratio:?}"
    );
    // (b) Monotone: more stiffness, more inharmonicity.
    assert!(
        b_fit.windows(2).all(|w| w[1] > w[0]),
        "B_fit not increasing with kappa: {b_fit:?}"
    );
    // (c) B ~ kappa^2 where bending dominates the small flat bias (kappa >= 2).
    let coeff: Vec<f64> = (1..5).map(|i| b_fit[i] / kappas[i].powi(2)).collect();
    let spread =
        nan_max(coeff.iter().copied()) / coeff.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "B / kappa^2 spread {spread}, B(4) / B(2) = {}",
        b_fit[3] / b_fit[1]
    );
    assert!(spread < 1.03, "B / kappa^2 not constant: {coeff:?}");
    // Doubling kappa quadruples B -- `pytest.approx(4.0, rel=0.03)`.
    assert!(
        (b_fit[3] / b_fit[1] - 4.0).abs() <= 0.03 * 4.0,
        "kappa 2 -> 4 gave B ratio {}",
        b_fit[3] / b_fit[1]
    );
}

// -- convergence and dispersion ------------------------------------------------------------------

#[test]
fn a_stiff_mode_converges_at_second_order_onto_the_stretched_law() {
    // Carried from `test_second_order_convergence`. Unlike the ideal string at lam = 1, the stiff
    // string disperses at every lam, so the error against the continuum stretched law has signal
    // to refine: halving h shrinks it ~4x.
    let (lam, mode) = (0.9, 4i64);
    let f_cont = stiff_harmonic_frequencies(c(), L, KAPPA, mode as usize)[mode as usize - 1];
    let grids = [64i64, 128, 256];
    let errors: Vec<f64> = grids
        .iter()
        .map(|&n| (measure_stiff_mode_frequencies(&[mode], n, lam, 0.6)[0] - f_cont).abs())
        .collect();
    let orders: Vec<f64> = (0..2)
        .map(|i| {
            (errors[i] / errors[i + 1]).ln()
                / ((L / grids[i] as f64) / (L / grids[i + 1] as f64)).ln()
        })
        .collect();
    let mean = (orders[0] + orders[1]) / 2.0;
    println!("errors {errors:?} Hz, orders {orders:?}, mean {mean}");
    assert!(
        errors.windows(2).all(|w| w[1] < w[0]),
        "errors did not shrink: {errors:?}"
    );
    assert!(
        orders.iter().all(|&p| p > 1.7),
        "a refinement step fell below 2nd order: {orders:?}"
    );
    assert!(mean > 1.85 && mean < 2.15, "mean order {mean} not ~2");
}

#[test]
fn every_swept_mode_lands_on_the_stiff_oracle_and_the_phase_velocity_rises_above_c() {
    // Carried from `test_dispersion_matches_stiff_oracle_and_stiffens_high_partials`. Bending
    // stiffens high partials: the phase velocity RISES above c with mode number, the opposite of
    // the ideal string's numerical droop. lam = 0.8 keeps the curve cleanly monotone.
    let (n, lam) = (256i64, 0.8);
    let modes = [2i64, 4, 8, 16, 32, 48, 64];
    let measured = measure_stiff_mode_frequencies(&modes, n, lam, 0.4);
    let k = make(n, lam, KAPPA, 0.0, THETA).p.k;
    let oracle = stiff_dispersion_frequencies(c(), L, n, KAPPA, k, THETA, &modes);
    assert!(
        measured.iter().all(|f| !f.is_nan()),
        "a mode frequency was not detected: {measured:?}"
    );
    let rel: Vec<f64> = measured
        .iter()
        .zip(&oracle)
        .map(|(m, o)| (m - o).abs() / o)
        .collect();
    let worst = nan_max(rel.iter().copied());
    println!("measured vs stiff oracle, relative: {rel:?}");
    assert!(worst < 1e-4, "worst measured-vs-oracle {worst:e}");

    let vp: Vec<f64> = phase_velocity(&measured, L, &modes)
        .iter()
        .map(|v| v / c())
        .collect();
    println!("v_p / c: {vp:?}");
    assert!(
        vp.iter().all(|&v| v > 1.0),
        "phase velocity not above c (no stiffening): {vp:?}"
    );
    assert!(
        vp.windows(2).all(|w| w[1] > w[0]),
        "phase velocity not rising with mode: {vp:?}"
    );
}

// -- construction ---------------------------------------------------------------------------------

#[test]
fn the_nine_invalid_parameter_sets_are_rejected() {
    // Carried from `test_invalid_parameters_rejected`, at its own base and values. Each is asserted
    // to be rejected for the RIGHT reason, which `pytest.raises(ValueError)` did not ask.
    let base = |l: f64, t: f64, rho: f64, n: i64, kappa: f64, sigma: f64, theta: f64, ok: bool| {
        Params::new(l, t, rho, 20000.0, n, kappa, sigma, theta, ok)
    };
    let (l, t, rho, n) = (1.0, 200.0, 0.005, 100i64);
    // The control: the base itself constructs.
    assert!(base(l, t, rho, n, 0.0, 0.0, THETA, true).is_ok());
    let cases: [(&str, Result<Params, ParamError>, ParamError); 9] = [
        (
            "rho = -1",
            base(l, t, -1.0, n, 0.0, 0.0, THETA, true),
            ParamError::NonPositive,
        ),
        (
            "T = 0",
            base(l, 0.0, rho, n, 0.0, 0.0, THETA, true),
            ParamError::NonPositive,
        ),
        (
            "L = -2",
            base(-2.0, t, rho, n, 0.0, 0.0, THETA, true),
            ParamError::NonPositive,
        ),
        (
            "kappa = -0.1",
            base(l, t, rho, n, -0.1, 0.0, THETA, true),
            ParamError::NegativeKappa,
        ),
        (
            "sigma = -0.1",
            base(l, t, rho, n, 0.0, -0.1, THETA, true),
            ParamError::NegativeSigma,
        ),
        (
            "N = 1",
            base(l, t, rho, 1, 0.0, 0.0, THETA, true),
            ParamError::TooFewSegments,
        ),
        (
            "theta = 0",
            base(l, t, rho, n, 0.0, 0.0, 0.0, true),
            ParamError::BadTheta(0.0),
        ),
        (
            "theta = 1.5",
            base(l, t, rho, n, 0.0, 0.0, 1.5, true),
            ParamError::BadTheta(1.5),
        ),
        // `boundary="clamped"`: the STRING parse lives only in the binding and retires with it
        // (`crates/physsynth-py/src/string_stiff.rs::boundary_ok`). What the core owns is the
        // refusal of an unusable boundary, which it takes as `boundary_ok = false`.
        (
            "boundary not supported",
            base(l, t, rho, n, 0.0, 0.0, THETA, false),
            ParamError::BadBoundary,
        ),
    ];
    for (name, got, want) in cases {
        assert_eq!(got.expect_err(name), want, "{name}");
    }
}

#[test]
fn the_models_default_theta_is_the_one_this_harness_validates() {
    // The Python fixtures took `THETA_DEFAULT` from the model, so a changed default (say below the
    // 1/4 stability floor) ran every one of them at the new value. This file writes the value down
    // instead (§29.3), so the default needs this one line to stay under the same bars.
    assert_eq!(physsynth_core::string_stiff::THETA_DEFAULT, THETA);
}

#[test]
fn a_courant_number_above_one_is_accepted_and_reported() {
    // Carried from `test_lambda_above_one_accepted`: unlike `IdealString`, the implicit scheme
    // must NOT reject lam > 1, and the lam it reports is the one the fixture built.
    let s = make(100, 2.5, KAPPA, 0.0, THETA);
    assert!(
        (s.p.lam - 2.5).abs() <= 1e-9 * 2.5,
        "lam = {}, want 2.5",
        s.p.lam
    );
}
