//! Musical controls for model #3 (`src/string_tuning.rs`) — `docs/dev/parameter-mapping-plan.md`
//! §5.
//!
//! **Every bar here runs the simulation.** None of them inverts the mapping's own algebra: a
//! string is built from the mapped coefficients, started in one exact mode (or plucked), stepped,
//! and what it *plays* is compared with what was asked. The stretched partial law and the anchor
//! choice are written out again below rather than imported, so a wrong law or a wrong anchor in
//! the module cannot agree with itself here.
//!
//! **The single-mode read-out is the well-conditioned one** (plan §7). The mode's amplitude obeys
//! `a_{n+1} = alpha a_n + beta a_{n-1}` with `alpha = 2 r cos(phi)`, `beta = -r^2`. Fitting that
//! directly is ill-conditioned at a low pitch (`phi ~ 0.005`: the two regressors are nearly the
//! same vector) and read `2.6e-5` in T60 on the probe. Regressing the second difference on
//! `(a_n, a_n - a_{n-1})` and taking `log r` through `ln_1p` read `1.8e-8`; the tolerances below
//! sit ~50x and ~20x above that floor.

use physsynth_analysis::damping::loss_coefficients_from_t60;
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_damped::{DampedStiffString, Params};
use physsynth_core::string_tuning::{
    mode_frequency, partial_drift, tune_damped_string, Targets, Tuning, TuningError,
};
use std::f64::consts::PI;

/// Relative tolerance on a single-mode frequency read back from the simulation.
const F_TOL: f64 = 1e-8;
/// Relative tolerance on a single-mode T60 read back from the simulation.
const T60_TOL: f64 = 1e-6;

/// The stretched partial law, written out again (not imported — module header).
fn law(f0: f64, b: f64, m: usize) -> f64 {
    let m = m as f64;
    m * f0 * ((1.0 + b * m * m) / (1.0 + b)).sqrt()
}

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

#[derive(Clone, Copy, Debug)]
struct Fixture {
    l: f64,
    rho: f64,
    fs: f64,
    n: usize,
    theta: f64,
    t: Targets,
}

fn targets(f0: f64, b: f64, t60_low: f64, t60_high: f64, f_ref: f64) -> Targets {
    Targets {
        f0,
        inharmonicity: b,
        t60_low,
        t60_high,
        f_ref,
    }
}

fn tune(fx: &Fixture) -> Result<Tuning, TuningError> {
    tune_damped_string(fx.l, fx.rho, fx.fs, fx.n, fx.theta, &fx.t)
}

fn build(fx: &Fixture, t: f64, kappa: f64, sigma0: f64, sigma1: f64) -> DampedStiffString {
    let p = Params::new(
        fx.l,
        t,
        fx.rho,
        fx.fs,
        fx.n as i64,
        kappa,
        sigma0,
        sigma1,
        fx.theta,
        true,
    )
    .expect("the mapped coefficients construct a string");
    DampedStiffString::new(p)
}

/// Start `s` in exactly mode `m`, step it, and read `(f, T60)` back — module header.
fn play_mode(s: &mut DampedStiffString, m: usize, steps: usize) -> (f64, f64) {
    let n = s.p.n;
    let shape: Vec<f64> = (0..=n)
        .map(|j| (m as f64 * PI * j as f64 / n as f64).sin())
        .collect();
    s.set_state(&shape, &vec![0.0; n + 1]);
    let proj = |u: &[f64]| u.iter().zip(&shape).map(|(a, b)| a * b).sum::<f64>();
    let mut amp = vec![proj(&s.u)];
    for _ in 0..steps {
        s.step();
        amp.push(proj(&s.u));
    }
    // D2 = g a_n + d v_n with v_n = a_n - a_{n-1}: g = alpha + beta - 1, d = -(beta + 1).
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
    let k = s.p.k;
    let r = (1.0 + d).sqrt();
    // 1 - cos(phi) = (2r - alpha) / 2r, alpha = g + 2 + d, and 2r - 2 = 2d / (r + 1).
    let one_minus_cos = (2.0 * d / (r + 1.0) - d - g) / (2.0 * r);
    let phi = 2.0 * (0.5 * one_minus_cos).sqrt().asin();
    let log10_r = 0.5 * d.ln_1p() / 10f64.ln();
    (phi / (2.0 * PI * k), -3.0 * k / log10_r)
}

/// The fixtures every round-trip bar runs: each input off model #3's harness defaults (`L = 1`,
/// `rho = 0.005`, `theta = 0.28`, `N = 128`) somewhere, and the corners the probe found hardest.
fn fixtures() -> Vec<Fixture> {
    let mut v = Vec::new();
    for &(l, rho) in &[(0.65, 0.0012), (1.0, 0.005), (1.3, 0.02)] {
        for &(fs, n) in &[(44100.0, 64), (48000.0, 128), (96000.0, 256)] {
            for &theta in &[0.25, 0.28, 0.5, 1.0] {
                for &(f0, b, tl, th, f_ref) in &[
                    (82.4, 1e-4, 4.0, 1.0, 2000.0),
                    (220.0, 0.0, 1.0, 0.2, 1500.0),
                    (440.0, 1e-2, 6.0, 0.7, 3000.0),
                ] {
                    v.push(Fixture {
                        l,
                        rho,
                        fs,
                        n,
                        theta,
                        t: targets(f0, b, tl, th, f_ref),
                    });
                }
            }
        }
    }
    v
}

// -- the round trip ---------------------------------------------------------------------------

#[test]
fn both_anchors_play_their_targets_in_the_simulation() {
    let (mut worst_f, mut worst_t60) = (0.0f64, 0.0f64);
    for fx in fixtures() {
        let tu = tune(&fx).unwrap_or_else(|e| panic!("{fx:?}: {e}"));
        let mut s = build(&fx, tu.t, tu.kappa, tu.sigma0, tu.sigma1);
        let anchors = [
            (1, fx.t.f0, fx.t.t60_low),
            (
                tu.anchor,
                law(fx.t.f0, fx.t.inharmonicity, tu.anchor),
                fx.t.t60_high,
            ),
        ];
        for (m, f_target, t60_target) in anchors {
            let (f, t60) = play_mode(&mut s, m, 3000);
            let (ef, et) = ((f / f_target - 1.0).abs(), (t60 / t60_target - 1.0).abs());
            assert!(
                ef < F_TOL,
                "{fx:?} mode {m}: plays {f} Hz, asked {f_target} ({ef:e})"
            );
            assert!(
                et < T60_TOL,
                "{fx:?} mode {m}: T60 {t60} s, asked {t60_target} ({et:e})"
            );
            worst_f = worst_f.max(ef);
            worst_t60 = worst_t60.max(et);
        }
    }
    println!("worst relative miss: frequency {worst_f:e}, T60 {worst_t60:e}");
}

#[test]
fn the_anchor_is_the_partial_nearest_the_reference() {
    // Written against the law above, not the module's `anchor_mode`: a wrong anchor would still
    // play its own target, so the round trip alone cannot see it.
    for fx in fixtures() {
        let tu = tune(&fx).unwrap();
        let dist = |m: usize| (law(fx.t.f0, fx.t.inharmonicity, m) - fx.t.f_ref).abs();
        let nearest = (2..fx.n)
            .min_by(|&a, &b| dist(a).total_cmp(&dist(b)))
            .unwrap();
        assert_eq!(tu.anchor, nearest, "{fx:?}");
    }
}

#[test]
fn a_plucked_string_sounds_both_anchors() {
    // The broadband path: a pluck, a pickup, the analysis crate's peak detector. Hann-windowed
    // peaks with parabolic refinement read a 2 s tone to a fraction of a cent, so 0.5 cents is
    // the detector's tolerance, not the mapping's.
    let fx = Fixture {
        l: 0.65,
        rho: 0.0012,
        fs: 48000.0,
        n: 128,
        theta: 0.5,
        t: targets(146.8, 3e-4, 5.0, 1.5, 1200.0),
    };
    let tu = tune(&fx).unwrap();
    let mut s = build(&fx, tu.t, tu.kappa, tu.sigma0, tu.sigma1);
    let x: Vec<f64> = (0..=fx.n).map(|j| fx.l * j as f64 / fx.n as f64).collect();
    let u0 = triangular_pluck(&x, fx.l, 0.137 * fx.l, 1e-3).unwrap();
    s.set_state(&u0, &vec![0.0; fx.n + 1]);
    let pickup = (0.241 * fx.n as f64).round() as usize;
    let steps = (2.0 * fx.fs) as usize;
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        s.step();
        out.push(s.u[pickup]);
    }
    let want = [fx.t.f0, law(fx.t.f0, fx.t.inharmonicity, tu.anchor)];
    let got = measure_partials_near(&out, fx.fs, &want, Some(0.3 * fx.t.f0));
    for (g, w) in got.iter().zip(&want) {
        let e = cents(*g, *w);
        println!("partial {w} Hz: detected {g} Hz ({e:+.4} cents)");
        assert!(e.abs() < 0.5, "detected {g} Hz for {w} Hz ({e} cents)");
    }
}

// -- the control ------------------------------------------------------------------------------

#[test]
fn the_textbook_mapping_misses_the_anchor_the_corrected_one_hits() {
    // The control must differ from its subject (CLAUDE.md): this is the probe's worst textbook
    // case, where the continuum mapping's anchor miss is far above measurement.
    let fx = Fixture {
        l: 0.65,
        rho: 0.005,
        fs: 44100.0,
        n: 64,
        theta: 1.0,
        t: targets(82.4, 1e-4, 4.0, 1.0, 2000.0),
    };
    let b = fx.t.inharmonicity;
    let tu = tune(&fx).unwrap();
    let f_anchor = law(fx.t.f0, b, tu.anchor);

    let c = 2.0 * fx.l * fx.t.f0 / (1.0 + b).sqrt();
    let kappa = b.sqrt() * c * fx.l / PI;
    let (s0, s1) = loss_coefficients_from_t60(
        c,
        fx.l,
        kappa,
        fx.t.f0,
        fx.t.t60_low,
        f_anchor,
        fx.t.t60_high,
    )
    .unwrap();
    let mut textbook = build(&fx, c * c * fx.rho, kappa, s0, s1);
    let (f_tb, t60_tb) = play_mode(&mut textbook, tu.anchor, 3000);

    let mut corrected = build(&fx, tu.t, tu.kappa, tu.sigma0, tu.sigma1);
    let (f_co, t60_co) = play_mode(&mut corrected, tu.anchor, 3000);

    let (tb_cents, tb_t60) = (cents(f_tb, f_anchor), t60_tb / fx.t.t60_high - 1.0);
    println!(
        "anchor mode {}: textbook {tb_cents:+.2} cents, T60 {:+.2} %; corrected {:+.2e} cents, \
         T60 {:+.2e} %",
        tu.anchor,
        100.0 * tb_t60,
        cents(f_co, f_anchor),
        100.0 * (t60_co / fx.t.t60_high - 1.0)
    );
    assert!(
        tb_cents < -100.0,
        "textbook anchor only {tb_cents} cents off"
    );
    assert!(tb_t60 > 0.10, "textbook anchor T60 only {tb_t60} long");
    assert!((f_co / f_anchor - 1.0).abs() < F_TOL);
    assert!((t60_co / fx.t.t60_high - 1.0).abs() < T60_TOL);
}

#[test]
fn a_harmonic_target_still_needs_stiffness() {
    // Module header: the grid and the theta average flatten high modes, so B = 0 maps to a small
    // positive kappa, where the textbook would give exactly zero.
    let fx = Fixture {
        l: 1.0,
        rho: 0.005,
        fs: 44100.0,
        n: 64,
        theta: 0.28,
        t: targets(110.0, 0.0, 3.0, 1.0, 2000.0),
    };
    let tu = tune(&fx).unwrap();
    assert!(tu.kappa > 0.0, "kappa = {}", tu.kappa);
}

// -- the drift read-out -----------------------------------------------------------------------

#[test]
fn mode_frequency_is_what_the_simulation_plays_off_the_anchors() {
    // `partial_drift` is built on `mode_frequency`, so it is checked against the simulation at
    // modes the mapping does not pin.
    for fx in fixtures().into_iter().step_by(7) {
        let tu = tune(&fx).unwrap();
        let mut s = build(&fx, tu.t, tu.kappa, tu.sigma0, tu.sigma1);
        for m in [2, 3, tu.anchor + 1, fx.n / 2] {
            let want = mode_frequency(
                fx.l, fx.n, fx.fs, fx.theta, tu.c, tu.kappa, tu.sigma0, tu.sigma1, m,
            )
            .unwrap();
            let (f, _) = play_mode(&mut s, m, 3000);
            assert!(
                (f / want - 1.0).abs() < F_TOL,
                "{fx:?} mode {m}: {f} vs {want}"
            );
        }
    }
}

#[test]
fn a_low_stiff_string_on_a_coarse_grid_warns() {
    // Plan §7's table: a low, stiff note on N = 64 drifts ~24 cents between its anchors. Pinned
    // loosely so a change in the read-out is noticed, not to the digit.
    let fx = Fixture {
        l: 1.0,
        rho: 0.005,
        fs: 44100.0,
        n: 64,
        theta: 0.28,
        t: targets(82.4, 1e-2, 4.0, 1.0, 2000.0),
    };
    let tu = tune(&fx).unwrap();
    let d = partial_drift(fx.l, fx.n, fx.fs, fx.theta, &tu, &fx.t);
    println!("anchor {}, drift {d:?}", tu.anchor);
    assert!(d.warn(tu.anchor));
    assert!(d.horizon < tu.anchor);
    assert!(
        d.worst_between > 15.0 && d.worst_between < 35.0,
        "{}",
        d.worst_between
    );
    assert!(d.worst_mode > 1 && d.worst_mode < tu.anchor);
}

#[test]
fn a_fine_grid_stays_in_tune_between_its_anchors() {
    let fx = Fixture {
        l: 1.0,
        rho: 0.005,
        fs: 96000.0,
        n: 256,
        theta: 0.28,
        t: targets(440.0, 1e-4, 4.0, 1.0, 2000.0),
    };
    let tu = tune(&fx).unwrap();
    let d = partial_drift(fx.l, fx.n, fx.fs, fx.theta, &tu, &fx.t);
    println!("anchor {}, drift {d:?}", tu.anchor);
    assert!(!d.warn(tu.anchor));
    assert!(d.horizon >= tu.anchor);
    assert!(d.worst_between < 5.0);
}

// -- refusals ---------------------------------------------------------------------------------

fn base() -> Fixture {
    Fixture {
        l: 1.0,
        rho: 0.005,
        fs: 44100.0,
        n: 128,
        theta: 0.28,
        t: targets(220.0, 1e-4, 3.0, 0.8, 2000.0),
    }
}

#[test]
fn non_positive_inputs_are_refused_by_name() {
    type Spoil = fn(&mut Fixture);
    let cases: [(&str, Spoil); 8] = [
        ("L", |f| f.l = 0.0),
        ("rho", |f| f.rho = -1.0),
        ("fs", |f| f.fs = f64::NAN),
        ("f0", |f| f.t.f0 = 0.0),
        ("t60_low", |f| f.t.t60_low = -2.0),
        ("t60_high", |f| f.t.t60_high = f64::INFINITY),
        ("f_ref", |f| f.t.f_ref = 0.0),
        ("L", |f| f.l = f64::NAN),
    ];
    for (name, spoil) in cases {
        let mut fx = base();
        spoil(&mut fx);
        assert_eq!(tune(&fx), Err(TuningError::NonPositive(name)));
    }
}

#[test]
fn each_unrepresentable_target_gets_its_own_refusal() {
    let mut fx = base();
    fx.theta = 0.0;
    assert_eq!(tune(&fx), Err(TuningError::BadTheta(0.0)));

    let mut fx = base();
    fx.n = 2;
    assert_eq!(tune(&fx), Err(TuningError::TooFewSegments));

    let mut fx = base();
    fx.t.f_ref = fx.t.f0;
    assert_eq!(tune(&fx), Err(TuningError::RefNotAboveFundamental));

    let mut fx = base();
    fx.t.inharmonicity = -0.5;
    assert_eq!(tune(&fx), Err(TuningError::BadInharmonicity));

    // Ring time rising with frequency: no passive string.
    let mut fx = base();
    fx.t.t60_high = 2.0 * fx.t.t60_low;
    assert!(matches!(tune(&fx), Err(TuningError::RisingDecay { .. })));

    // Partials compressed below a flexible string's.
    let mut fx = base();
    fx.t.inharmonicity = -1e-3;
    assert_eq!(tune(&fx), Err(TuningError::ShrinkingPartials));

    // An anchor above what theta = 1 at this fs can produce at all (sin^2(phi/2) < 1/4).
    let mut fx = base();
    fx.theta = 1.0;
    fx.fs = 8000.0;
    fx.t.f_ref = 3500.0;
    assert!(
        matches!(tune(&fx), Err(TuningError::Unreachable { .. })),
        "{:?}",
        tune(&fx)
    );
}
