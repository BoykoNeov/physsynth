//! Musical controls for the stiff string (model #2, one loss) — `src/string_tuning.rs`'s
//! `tune_stiff_string`, `docs/dev/string-mapping-plan.md` §3 and §14.
//!
//! **Every pitch and ring time here is read from the simulation** (a `StiffString` started in one
//! exact mode, the damped mapping's well-conditioned fit). The stretched law and the textbook
//! mapping are written out again below, not imported.

use physsynth_core::string_stiff::{Params, StiffString};
use physsynth_core::string_tuning::{stiff_mode_t60, tune_stiff_string, Tuning, TuningError};
use std::f64::consts::PI;

/// Relative pitch tolerance — the damped mapping's `F_TOL`.
const F_TOL: f64 = 1e-8;
/// Relative T60 tolerance — the damped mapping's `T60_TOL`.
const T60_TOL: f64 = 1e-6;

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

/// The stretched partial law the targets are written in.
fn law(f0: f64, b: f64, m: usize) -> f64 {
    let m = m as f64;
    m * f0 * ((1.0 + b * m * m) / (1.0 + b)).sqrt()
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

struct Fx {
    l: f64,
    rho: f64,
    fs: f64,
    n: usize,
    theta: f64,
    f0: f64,
    b: f64,
    t60: f64,
    f_ref: f64,
}

fn tune(fx: &Fx) -> Result<Tuning, TuningError> {
    tune_stiff_string(
        fx.l, fx.rho, fx.fs, fx.n, fx.theta, fx.f0, fx.b, fx.t60, fx.f_ref,
    )
}

fn build(fx: &Fx, t: f64, kappa: f64, sigma: f64) -> StiffString {
    let p = Params::new(
        fx.l,
        t,
        fx.rho,
        fx.fs,
        fx.n as i64,
        kappa,
        sigma,
        fx.theta,
        true,
    )
    .expect("the mapped coefficients construct a stiff string");
    StiffString::new(p)
}

/// Start in exactly mode `m`, step, and read `(f, T60)` back.
fn play(s: &mut StiffString, m: usize, steps: usize) -> (f64, f64) {
    let n = s.p.n;
    let sh: Vec<f64> = (0..=n)
        .map(|j| (m as f64 * PI * j as f64 / n as f64).sin())
        .collect();
    s.set_state(&sh, &vec![0.0; n + 1]);
    let proj = |u: &[f64]| u.iter().zip(&sh).map(|(a, b)| a * b).sum::<f64>();
    let mut amp = vec![proj(&s.u)];
    for _ in 0..steps {
        s.step();
        amp.push(proj(&s.u));
    }
    fit(&amp, s.p.k)
}

/// Each input off model #2's harness defaults (`L = 1`, `rho = 0.005`, `theta = 0.28`, `N = 128`)
/// somewhere; harmonic and stiff targets; a thud and a long ring.
fn fixtures() -> Vec<Fx> {
    let mut v = Vec::new();
    for &(l, rho) in &[(0.65, 0.0012), (1.3, 0.02)] {
        for &(fs, n) in &[(22050.0, 32usize), (48000.0, 128), (96000.0, 200)] {
            for &theta in &[0.25, 0.28, 0.5, 1.0] {
                for &(f0, b, t60, f_ref) in &[
                    (55.0, 1e-4, 8.0, 1000.0),
                    (220.0, 0.0, 0.3, 1500.0),
                    (440.0, 1e-2, 3.0, 3000.0),
                ] {
                    v.push(Fx {
                        l,
                        rho,
                        fs,
                        n,
                        theta,
                        f0,
                        b,
                        t60,
                        f_ref,
                    });
                }
            }
        }
    }
    v
}

#[test]
fn the_stiff_string_plays_its_pitch_ring_time_and_stretched_anchor() {
    let (mut wf, mut wt, mut wa, mut wat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut skipped = 0;
    for fx in fixtures() {
        let tu = match tune(&fx) {
            Ok(tu) => tu,
            // The scheme's ceiling at theta = 1 (below fs / 6): counted, and pinned below.
            Err(TuningError::Unreachable { .. }) => {
                skipped += 1;
                continue;
            }
            Err(e) => panic!("{e}"),
        };
        assert_eq!(tu.sigma1, 0.0);
        let mut s = build(&fx, tu.t, tu.kappa, tu.sigma0);
        let steps = (0.06 * fx.fs) as usize;
        let (f1, t1) = play(&mut s, 1, steps);
        let (fa, ta) = play(&mut s, tu.anchor, steps);
        let (ef, et) = ((f1 / fx.f0 - 1.0).abs(), (t1 / fx.t60 - 1.0).abs());
        let ea = (fa / law(fx.f0, fx.b, tu.anchor) - 1.0).abs();
        // The anchor's ring time is a read-out, and it is the simulation's.
        let reported = stiff_mode_t60(fx.l, fx.n, fx.fs, fx.theta, &tu, tu.anchor);
        let eat = (ta / reported - 1.0).abs();
        (wf, wt, wa, wat) = (wf.max(ef), wt.max(et), wa.max(ea), wat.max(eat));
        assert!(
            ef < F_TOL && et < T60_TOL && ea < F_TOL && eat < T60_TOL,
            "fs {} N {} theta {} f0 {} B {}: f1 {f1}, T60 {t1}, anchor {fa} ({ta} s vs {reported})",
            fx.fs,
            fx.n,
            fx.theta,
            fx.f0,
            fx.b
        );
    }
    println!(
        "stiff round trip: pitch {wf:.2e}, T60 {wt:.2e}, anchor pitch {wa:.2e}, anchor T60 \
         read-out {wat:.2e}; {skipped} of {} past the scheme's ceiling",
        fixtures().len()
    );
    assert!(skipped <= 6, "{skipped} fixtures unreachable");
}

#[test]
fn the_textbook_stiff_string_misses_the_anchor() {
    // Textbook: c = 2 L f0 / sqrt(1 + B), kappa = sqrt(B) c L / pi, sigma = 3 ln 10 / T60.
    let fx = Fx {
        l: 1.0,
        rho: 0.005,
        fs: 44100.0,
        n: 64,
        theta: 1.0,
        f0: 82.4,
        b: 1e-4,
        t60: 4.0,
        f_ref: 2000.0,
    };
    let c = 2.0 * fx.l * fx.f0 / (1.0 + fx.b).sqrt();
    let kappa = fx.b.sqrt() * c * fx.l / PI;
    let sigma = 3.0 * 10f64.ln() / fx.t60;
    let tu = tune(&fx).unwrap();
    let mut book = build(&fx, c * c * fx.rho, kappa, sigma);
    let mut ours = build(&fx, tu.t, tu.kappa, tu.sigma0);
    let target = law(fx.f0, fx.b, tu.anchor);
    let (fb, _) = play(&mut book, tu.anchor, 3000);
    let (fo, _) = play(&mut ours, tu.anchor, 3000);
    let miss = cents(fb, target);
    println!(
        "anchor {} at {target:.1} Hz: textbook {miss:+.1} cents, corrected {:+.2e}",
        tu.anchor,
        cents(fo, target)
    );
    assert!(miss < -20.0, "textbook miss {miss}");
    assert!(cents(fo, target).abs() < 1e-5);
}

#[test]
fn a_stiff_string_past_the_schemes_ceiling_or_badly_asked_is_refused() {
    let base = |theta: f64, fs: f64, f_ref: f64| Fx {
        l: 1.0,
        rho: 0.005,
        fs,
        n: 64,
        theta,
        f0: 110.0,
        b: 1e-4,
        t60: 3.0,
        f_ref,
    };
    // At theta = 1 the lossless pitch is capped below fs / 6 = 3,675 Hz at 22.05 kHz.
    let high = base(1.0, 22050.0, 4000.0);
    assert!(matches!(tune(&high), Err(TuningError::Unreachable { mode }) if mode > 1));
    // ...and the same anchor is reachable at theta = 1/4, whose cap is the Nyquist frequency.
    assert!(tune(&base(0.25, 22050.0, 4000.0)).is_ok());
    assert_eq!(
        tune(&base(0.28, 44100.0, 100.0)),
        Err(TuningError::RefNotAboveFundamental)
    );
    let mut few = base(0.28, 44100.0, 2000.0);
    few.n = 2;
    assert_eq!(tune(&few), Err(TuningError::TooFewSegments));
    let mut shrink = base(0.28, 44100.0, 2000.0);
    shrink.b = -1e-3;
    assert_eq!(tune(&shrink), Err(TuningError::ShrinkingPartials));
    let mut nan = base(0.28, 44100.0, 2000.0);
    nan.t60 = f64::NAN;
    assert_eq!(tune(&nan), Err(TuningError::NonPositive("t60")));
}
