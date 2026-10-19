//! Musical controls for the ideal string (model #1, at `lambda = 1`) and for the flexible string
//! the barrier pages build — `src/string_tuning.rs`, `docs/dev/string-mapping-plan.md` §2, §13.
//!
//! **Every pitch and ring time here is read from the simulation**, with the damped mapping's
//! well-conditioned single-mode fit (`tests/string_tuning.rs`). The `lambda = 1` relation and the
//! textbook mapping are written out again below, not imported.

use physsynth_core::string_damped::{DampedStiffString, Params as DampedParams};
use physsynth_core::string_ideal::{Boundary, IdealString, Params};
use physsynth_core::string_stiff::THETA_DEFAULT;
use physsynth_core::string_tuning::{
    ideal_mode_frequency, mode_frequency, tune_flexible_string, tune_ideal_string, IdealTuning,
    TuningError,
};
use std::f64::consts::PI;

/// Relative pitch tolerance — the damped mapping's `F_TOL`.
const F_TOL: f64 = 1e-8;
/// Relative T60 tolerance — the damped mapping's `T60_TOL`.
const T60_TOL: f64 = 1e-6;

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

/// `(f, T60)` of a decaying single-mode series — the damped mapping's well-conditioned fit.
fn fit(amp: &[f64], k: f64) -> (f64, f64) {
    let (mut s11, mut s12, mut s22, mut r1, mut r2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 2..amp.len() - 1 {
        let (x, v) = (amp[i], amp[i] - amp[i - 1]);
        let z = (amp[i + 1] - amp[i]) - v;
        s11 += x * x;
        s12 += x * v;
        s22 += v * v;
        r1 += x * z;
        r2 += v * z;
    }
    let det = s11 * s22 - s12 * s12;
    let g = (r1 * s22 - s12 * r2) / det;
    let d = (s11 * r2 - s12 * r1) / det;
    let r = (1.0 + d).sqrt();
    let one_minus_cos = (2.0 * d / (r + 1.0) - d - g) / (2.0 * r);
    let phi = 2.0 * (0.5 * one_minus_cos).sqrt().asin();
    let log10_r = 0.5 * d.ln_1p() / 10f64.ln();
    (phi / (2.0 * PI * k), -3.0 * k / log10_r)
}

fn shape(n: usize, m: usize) -> Vec<f64> {
    (0..=n)
        .map(|j| (m as f64 * PI * j as f64 / n as f64).sin())
        .collect()
}

fn ideal(l: f64, rho: f64, n: usize, tu: &IdealTuning) -> IdealString {
    let p = Params::new(
        l,
        tu.t,
        rho,
        tu.fs,
        n as i64,
        tu.sigma,
        Some((Boundary::Fixed, Boundary::Fixed)),
    )
    .expect("the mapped coefficients construct an ideal string");
    IdealString::new(p)
}

/// Start an ideal string in exactly mode `m`, step it, and read `(f, T60)` back.
fn play_ideal(s: &mut IdealString, m: usize, steps: usize) -> (f64, f64) {
    let n = s.params().n;
    let sh = shape(n, m);
    s.set_state(&sh, &vec![0.0; n + 1]);
    let proj = |u: &[f64]| u.iter().zip(&sh).map(|(a, b)| a * b).sum::<f64>();
    let mut amp = vec![proj(&s.u)];
    for _ in 0..steps {
        s.step();
        amp.push(proj(&s.u));
    }
    fit(&amp, 1.0 / s.params().fs)
}

/// Pitches from a sub-audio bass to a treble, grids from coarse to fine, ring times from a thud
/// to a long ring, and the string's `L` and `rho` off their harness defaults.
fn ideal_fixtures() -> Vec<(f64, f64, usize, f64, f64)> {
    let mut v = Vec::new();
    for &(l, rho) in &[(0.65, 0.0012), (1.3, 0.02)] {
        for &f0 in &[27.5, 196.0, 1000.0] {
            for &n in &[16usize, 100, 300] {
                for &t60 in &[0.1, 2.0, 30.0] {
                    v.push((l, rho, n, f0, t60));
                }
            }
        }
    }
    v
}

// -- the ideal string ---------------------------------------------------------------------------

#[test]
fn the_ideal_string_plays_its_pitch_and_every_partial_its_ring_time() {
    let (mut worst_f, mut worst_t) = (0.0f64, 0.0f64);
    for (l, rho, n, f0, t60) in ideal_fixtures() {
        let tu = tune_ideal_string(l, rho, n, f0, t60).unwrap();
        let mut s = ideal(l, rho, n, &tu);
        // lambda = c k / h = 1: the grid sits at the scheme's exact setting.
        let lam = tu.c / tu.fs * n as f64 / l;
        assert!((lam - 1.0).abs() < 1e-12, "lambda {lam}");
        assert!((s.params().lam - 1.0).abs() < 1e-12);
        let steps = ((0.05 * tu.fs) as usize).max(400);
        let (f1, _) = play_ideal(&mut s, 1, steps);
        let ef = (f1 / f0 - 1.0).abs();
        worst_f = worst_f.max(ef);
        assert!(ef < F_TOL, "f0 {f0} N {n} T60 {t60}: plays {f1}");
        // The ring time is the same for every partial: the fundamental and two above it.
        for m in [1, 3, n / 2] {
            let (_, t) = play_ideal(&mut s, m, steps);
            let et = (t / t60 - 1.0).abs();
            worst_t = worst_t.max(et);
            assert!(et < T60_TOL, "f0 {f0} N {n} T60 {t60} mode {m}: rings {t}");
        }
    }
    println!("ideal round trip: worst pitch {worst_f:.2e}, worst T60 {worst_t:.2e}");
}

#[test]
fn the_partials_above_the_fundamental_run_sharp_by_what_the_loss_gives() {
    // The read-out's frequency is the simulation's, mode by mode; lossless the partials are
    // exactly harmonic, and loss pushes them sharp (plan §13: physics, not the scheme).
    for &(f0, n, t60, lo, hi) in &[
        (27.5, 16, 0.1, 90.0, 140.0), // the probe: +98 to +126 cents
        (27.5, 64, 1.0, 1.0, 1.5),
        (110.0, 64, 0.1, 6.0, 9.0),
    ] {
        let tu = tune_ideal_string(1.0, 0.005, n, f0, t60).unwrap();
        let mut s = ideal(1.0, 0.005, n, &tu);
        for m in [2, n / 4] {
            let (f, _) = play_ideal(&mut s, m, ((0.05 * tu.fs) as usize).max(400));
            let reported = ideal_mode_frequency(n, tu.fs, tu.sigma, m).unwrap();
            assert!(
                (f / reported - 1.0).abs() < 1e-8,
                "mode {m}: {f} vs {reported}"
            );
            let c = cents(f, m as f64 * f0);
            assert!(
                c > lo && c < hi,
                "f0 {f0} N {n} T60 {t60} mode {m}: {c:+.3} cents"
            );
        }
        let lossless = tune_ideal_string(1.0, 0.005, n, f0, 1e9).unwrap();
        let f3 = ideal_mode_frequency(n, lossless.fs, lossless.sigma, 3).unwrap();
        assert!(cents(f3, 3.0 * f0).abs() < 1e-6, "lossless mode 3 at {f3}");
    }
}

#[test]
fn the_textbook_ideal_string_misses_a_short_ring() {
    // Textbook: c = 2 L f0 and sigma = 3 ln 10 / T60, run at lambda = 1. The loss pulls the
    // fundamental flat, which the corrected mapping folds in; at a thud on a coarse grid the miss
    // is a large fraction of a semitone.
    let (l, rho, n, f0, t60) = (1.0, 0.005, 16, 27.5, 0.1);
    let c = 2.0 * l * f0;
    let fs = c * n as f64 / l;
    let sigma = 3.0 * 10f64.ln() / t60;
    let book = IdealTuning {
        t: rho * c * c,
        sigma,
        fs,
        c,
    };
    let mut s = ideal(l, rho, n, &book);
    let (f1, _) = play_ideal(&mut s, 1, 2000);
    let miss = cents(f1, f0);
    let tu = tune_ideal_string(l, rho, n, f0, t60).unwrap();
    let (fc, _) = play_ideal(&mut ideal(l, rho, n, &tu), 1, 2000);
    println!(
        "27.5 Hz, N = 16, T60 = 0.1 s: textbook {miss:+.2} cents, corrected {:+.2e}",
        cents(fc, f0)
    );
    assert!(miss < -20.0, "textbook miss {miss}");
    assert!(cents(fc, f0).abs() < 1e-6);
}

#[test]
fn a_bad_ideal_request_is_refused_by_name() {
    assert_eq!(
        tune_ideal_string(0.0, 0.005, 64, 110.0, 2.0),
        Err(TuningError::NonPositive("L"))
    );
    assert_eq!(
        tune_ideal_string(1.0, f64::NAN, 64, 110.0, 2.0),
        Err(TuningError::NonPositive("rho"))
    );
    assert_eq!(
        tune_ideal_string(1.0, 0.005, 64, -1.0, 2.0),
        Err(TuningError::NonPositive("f0"))
    );
    assert_eq!(
        tune_ideal_string(1.0, 0.005, 64, 110.0, 0.0),
        Err(TuningError::NonPositive("t60"))
    );
    assert_eq!(
        tune_ideal_string(1.0, 0.005, 1, 110.0, 2.0),
        Err(TuningError::TooFewSegments)
    );
}

// -- the flexible string of the barrier pages ----------------------------------------------------

/// Start a damped string in exactly mode `m`, step it, and read `(f, T60)` back.
fn play_damped(s: &mut DampedStiffString, m: usize, steps: usize) -> (f64, f64) {
    let n = s.p.n;
    let sh = shape(n, m);
    s.set_state(&sh, &vec![0.0; n + 1]);
    let proj = |u: &[f64]| u.iter().zip(&sh).map(|(a, b)| a * b).sum::<f64>();
    let mut amp = vec![proj(&s.u)];
    for _ in 0..steps {
        s.step();
        amp.push(proj(&s.u));
    }
    fit(&amp, s.p.k)
}

#[test]
fn the_flexible_string_plays_its_pitch_and_ring_time_at_mode_one() {
    let mut worst = (0.0f64, 0.0f64);
    for &(l, rho) in &[(0.65, 0.0012), (1.3, 0.02)] {
        for &(fs, n) in &[(22050.0, 32usize), (44100.0, 64), (96000.0, 128)] {
            for &theta in &[0.25, THETA_DEFAULT, 0.5, 1.0] {
                for &(f0, t60) in &[(82.4, 4.0), (220.0, 0.3), (440.0, 12.0)] {
                    let tu = tune_flexible_string(l, rho, fs, n, theta, f0, t60).unwrap();
                    let p = DampedParams::new(
                        l, tu.t, rho, fs, n as i64, 0.0, tu.sigma0, 0.0, theta, true,
                    )
                    .unwrap();
                    let mut s = DampedStiffString::new(p);
                    assert!((s.p.lam / tu.lam - 1.0).abs() < 1e-12);
                    let steps = (0.06 * fs) as usize;
                    let (f, t) = play_damped(&mut s, 1, steps);
                    let (ef, et) = ((f / f0 - 1.0).abs(), (t / t60 - 1.0).abs());
                    worst = (worst.0.max(ef), worst.1.max(et));
                    assert!(
                        ef < F_TOL && et < T60_TOL,
                        "{f0} Hz {fs} {n} {theta}: {f} {t}"
                    );
                    // Off mode 1, the read-out's pitch is the simulation's.
                    let (f3, _) = play_damped(&mut s, 3, steps);
                    let r3 = mode_frequency(l, n, fs, theta, tu.c, 0.0, tu.sigma0, 0.0, 3).unwrap();
                    assert!((f3 / r3 - 1.0).abs() < 1e-8, "mode 3: {f3} vs {r3}");
                }
            }
        }
    }
    println!(
        "flexible round trip: worst pitch {:.2e}, worst T60 {:.2e}",
        worst.0, worst.1
    );
}

#[test]
fn a_bad_flexible_request_is_refused_by_name() {
    let run = |fs: f64, n: usize, theta: f64, f0: f64, t60: f64| {
        tune_flexible_string(1.0, 0.005, fs, n, theta, f0, t60)
    };
    assert_eq!(
        run(0.0, 64, 0.28, 110.0, 2.0),
        Err(TuningError::NonPositive("fs"))
    );
    assert_eq!(
        run(44100.0, 64, 0.0, 110.0, 2.0),
        Err(TuningError::BadTheta(0.0))
    );
    assert_eq!(
        run(44100.0, 1, 0.28, 110.0, 2.0),
        Err(TuningError::TooFewSegments)
    );
    // At theta = 1 the lossless pitch is capped at sin^2(phi/2) < 1/(4 theta), i.e. below fs / 6;
    // at theta = 1/4 the cap is the Nyquist frequency, so the same note is reachable there.
    assert_eq!(
        run(8000.0, 64, 1.0, 2000.0, 2.0),
        Err(TuningError::Unreachable { mode: 1 })
    );
    assert!(run(8000.0, 64, 0.25, 2000.0, 2.0).is_ok());
}
