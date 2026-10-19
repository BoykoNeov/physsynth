//! Musical controls for model #9, the tension string (`src/string_tuning.rs`'s
//! `tune_tension_string`) — `docs/dev/string-mapping-plan.md` §5 and §9.
//!
//! **Every pitch and ring time here is read from the simulation.** A tension string is built from
//! the mapped coefficients and the gauge's `rho` and `EA`, started in one exact mode, stepped, and
//! what it plays is compared with what was asked. The read-out is the damped mapping's
//! well-conditioned single-mode fit (`tests/string_tuning.rs`, plan §7 of the damped batch).
//!
//! **The gauge's arithmetic is written out again with literal numbers** — `7800`, `190e9`, Eq. 19's
//! three coefficients — not imported from the module's table, because a bar about a constant must
//! not import the constant (CLAUDE.md).

use physsynth_core::string_nonlinear::{Params, TensionModulatedString, TENSION_TOL_DEFAULT};
use physsynth_core::string_tuning::{
    string_material, tune_tension_string, GaugeTuning, Modulus, StringMaterial, Targets,
    TuningError, STRING_MATERIALS,
};
use std::f64::consts::PI;

/// Relative pitch tolerance of the gentle round trip — the damped mapping's `F_TOL`.
const F_TOL: f64 = 1e-8;
/// Relative T60 tolerance of the gentle round trip — the damped mapping's `T60_TOL`.
const T60_TOL: f64 = 1e-6;
/// The gentle pluck: small enough that `EA` moves the pitch by far less than `F_TOL` (the probe
/// read ≤ 3e-8 cents at 1e-7 m, plan §9).
const GENTLE: f64 = 1e-8;

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

struct Fixture {
    material: &'static str,
    diameter: f64,
    l: f64,
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

fn tune(fx: &Fixture) -> Result<GaugeTuning, TuningError> {
    let m = string_material(fx.material).expect("a named material");
    tune_tension_string(m, fx.diameter, fx.l, fx.fs, fx.n, fx.theta, &fx.t)
}

fn build(fx: &Fixture, g: &GaugeTuning, lossy: bool) -> TensionModulatedString {
    let (s0, s1) = if lossy {
        (g.tuning.sigma0, g.tuning.sigma1)
    } else {
        (0.0, 0.0)
    };
    let p = Params::new(
        fx.l,
        g.tuning.t,
        g.rho,
        fx.fs,
        fx.n as i64,
        g.tuning.kappa,
        g.ea,
        s0,
        s1,
        fx.theta,
        TENSION_TOL_DEFAULT,
        true,
    )
    .expect("the mapped coefficients construct a tension string");
    TensionModulatedString::new(p)
}

fn shape(n: usize, m: usize) -> Vec<f64> {
    (0..=n)
        .map(|j| (m as f64 * PI * j as f64 / n as f64).sin())
        .collect()
}

/// Start `s` in exactly mode `m` at `amp`, step it, and return the mode's projected amplitude.
fn mode_series(s: &mut TensionModulatedString, m: usize, amp: f64, steps: usize) -> Vec<f64> {
    let sh = shape(s.p.n, m);
    let u0: Vec<f64> = sh.iter().map(|x| amp * x).collect();
    s.set_state(&u0, &vec![0.0; u0.len()]);
    let proj = |u: &[f64]| u.iter().zip(&sh).map(|(a, b)| a * b).sum::<f64>();
    let mut out = vec![proj(&s.u)];
    for _ in 0..steps {
        s.step().expect("the tension solve converges");
        out.push(proj(&s.u));
    }
    out
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

/// Mean frequency from upward zero crossings (linear interpolation), for an undamped series.
fn zero_cross_freq(sig: &[f64], fs: f64) -> f64 {
    let mut t = Vec::new();
    for i in 0..sig.len() - 1 {
        if sig[i] <= 0.0 && sig[i + 1] > 0.0 {
            t.push(i as f64 + sig[i] / (sig[i] - sig[i + 1]));
        }
    }
    assert!(t.len() >= 3, "too few crossings to read a pitch");
    (t.len() - 1) as f64 * fs / (t[t.len() - 1] - t[0])
}

/// Every material, each with an input off the defaults somewhere (`L`, `fs`, `N`, `theta`, the
/// targets), and gauges from a thin treble to a thick bass.
fn fixtures() -> Vec<Fixture> {
    let mut v = Vec::new();
    for &material in &["steel", "nylon", "gut"] {
        for &(diameter, l, f0, b) in &[
            (0.25e-3, 0.65, 330.0, 1e-5),
            (1.0e-3, 0.65, 110.0, 1e-3),
            (0.5e-3, 0.33, 440.0, 0.0),
        ] {
            for &(fs, n, theta) in &[(44100.0, 64, 0.28), (48000.0, 128, 0.5), (96000.0, 96, 1.0)] {
                v.push(Fixture {
                    material,
                    diameter,
                    l,
                    fs,
                    n,
                    theta,
                    t: targets(f0, b, 4.0, 0.9, 2500.0),
                });
            }
        }
    }
    v
}

// -- the round trip ---------------------------------------------------------------------------

#[test]
fn a_gentle_pluck_plays_both_anchors_on_every_material() {
    let (mut worst_f, mut worst_t60) = (0.0f64, 0.0f64);
    for fx in fixtures() {
        let g = tune(&fx).expect("each fixture is representable");
        let anchor = g.tuning.anchor;
        let law = |m: usize| {
            let mf = m as f64;
            mf * fx.t.f0
                * ((1.0 + fx.t.inharmonicity * mf * mf) / (1.0 + fx.t.inharmonicity)).sqrt()
        };
        for &(m, t60) in &[(1, fx.t.t60_low), (anchor, fx.t.t60_high)] {
            let mut s = build(&fx, &g, true);
            let steps = (0.06 * fx.fs) as usize;
            let (f, t) = fit(&mode_series(&mut s, m, GENTLE, steps), 1.0 / fx.fs);
            let (ef, et) = ((f / law(m) - 1.0).abs(), (t / t60 - 1.0).abs());
            worst_f = worst_f.max(ef);
            worst_t60 = worst_t60.max(et);
            assert!(
                ef < F_TOL && et < T60_TOL,
                "{} d={} mode {m}: f {f} vs {}, T60 {t} vs {t60}",
                fx.material,
                fx.diameter,
                law(m)
            );
        }
    }
    println!("gentle round trip: worst pitch {worst_f:.2e}, worst T60 {worst_t60:.2e}");
}

// -- the gauge --------------------------------------------------------------------------------

#[test]
fn steel_and_gut_take_their_density_and_modulus_from_the_gauge() {
    for &(name, density, e) in &[("steel", 7800.0, 190.0e9), ("gut", 1322.0, 6.0e9)] {
        let fx = Fixture {
            material: name,
            diameter: 0.6e-3,
            l: 0.65,
            fs: 44100.0,
            n: 100,
            theta: 0.28,
            t: targets(196.0, 1e-4, 3.0, 0.8, 2000.0),
        };
        let g = tune(&fx).unwrap();
        let area = PI * 0.3e-3 * 0.3e-3;
        let rel = |a: f64, b: f64| (a / b - 1.0).abs();
        assert!(rel(g.rho, density * area) < 1e-14, "{name}: rho {}", g.rho);
        assert!(rel(g.ea, e * area) < 1e-14, "{name}: EA {}", g.ea);
        assert!(rel(g.youngs, e) < 1e-15, "{name}: E {}", g.youngs);
        assert!(rel(g.stress, g.tuning.t / area) < 1e-14, "{name}: stress");
        // kappa_gauge = sqrt(E I / rho), I = pi r^4 / 4 — so kappa^2 = E r^2 / (4 density).
        let kappa2 = e * 0.3e-3 * 0.3e-3 / (4.0 * density);
        assert!(
            rel(g.gauge_kappa * g.gauge_kappa, kappa2) < 1e-13,
            "{name}: gauge kappa"
        );
        let b = PI * PI * kappa2 / (g.tuning.c * g.tuning.c * 0.65 * 0.65);
        assert!(rel(g.gauge_inharmonicity, b) < 1e-13, "{name}: gauge B");
        // The gauge's stiffness is a read-out, not imposed: the mapped kappa answers the target B.
        assert!(
            rel(g.tuning.kappa, g.gauge_kappa) > 1e-3,
            "{name}: kappa was imposed"
        );
    }
}

#[test]
fn nylon_stiffens_with_stress_by_the_cited_fit() {
    let fx = |f0: f64| Fixture {
        material: "nylon",
        diameter: 0.8e-3,
        l: 0.65,
        fs: 44100.0,
        n: 100,
        theta: 0.28,
        t: targets(f0, 1e-4, 3.0, 0.8, 2500.0),
    };
    let (lo, hi) = (tune(&fx(196.0)).unwrap(), tune(&fx(392.0)).unwrap());
    for g in [&lo, &hi] {
        // Eq. 19: E = 32.0 + 0.0353 sigma[MPa] - 0.0269 rho[kg/m^3] GPa, at rho = 1080.
        let e = (32.0 + 0.0353 * (g.stress / 1e6) - 0.0269 * 1080.0) * 1e9;
        assert!((g.youngs / e - 1.0).abs() < 1e-14, "E {} vs {e}", g.youngs);
        let area = PI * 0.4e-3 * 0.4e-3;
        assert!((g.rho / (1080.0 * area) - 1.0).abs() < 1e-14);
    }
    // The control differs from its subject: an octave up is four times the stress, and stiffer.
    assert!(
        (hi.stress / lo.stress - 4.0).abs() < 0.01,
        "stress ratio {}",
        hi.stress / lo.stress
    );
    assert!(
        hi.youngs > lo.youngs * 1.2,
        "E {} vs {}",
        hi.youngs,
        lo.youngs
    );
}

// -- the hard pluck: the gauge's EA is live, and the note settles onto the target -------------

/// The lossless start pitch at `amp`, against a 1e-6 m reference, in cents.
fn start_shift(fx: &Fixture, g: &GaugeTuning, amp: f64) -> f64 {
    let steps = (12.0 * fx.fs / fx.t.f0) as usize;
    let mut a = build(fx, g, false);
    let mut r = build(fx, g, false);
    let fa = zero_cross_freq(&mode_series(&mut a, 1, amp, steps), fx.fs);
    let fr = zero_cross_freq(&mode_series(&mut r, 1, 1e-6, steps), fx.fs);
    cents(fa, fr)
}

#[test]
fn a_hard_pluck_starts_sharp_by_what_the_gauge_gives() {
    let fx = |material| Fixture {
        material,
        diameter: 0.6e-3,
        l: 0.65,
        fs: 44100.0,
        n: 64,
        theta: 0.28,
        t: targets(196.0, 1e-4, 3.0, 0.8, 2000.0),
    };
    let (steel, gut) = (fx("steel"), fx("gut"));
    let (gs, gg) = (tune(&steel).unwrap(), tune(&gut).unwrap());
    let (cs, cg) = (start_shift(&steel, &gs, 3e-3), start_shift(&gut, &gg, 3e-3));
    println!("3 mm pluck at 196 Hz, 0.6 mm: steel +{cs:.2} cents, gut +{cg:.2} cents");
    // Hardening only ever raises the pitch, and steel's EA/T is the larger, so it rises more.
    assert!(cg > 0.1, "gut barely moved: {cg}");
    assert!(cs > 3.0 * cg, "steel {cs} vs gut {cg}");
}

#[test]
fn a_hard_pluck_settles_onto_the_target() {
    let fx = Fixture {
        material: "steel",
        diameter: 0.6e-3,
        l: 0.65,
        fs: 44100.0,
        n: 64,
        theta: 0.28,
        t: targets(196.0, 1e-4, 1.0, 0.3, 2000.0),
    };
    let g = tune(&fx).unwrap();
    let mut s = build(&fx, &g, true);
    let total = (1.5 * fx.fs) as usize;
    let series = mode_series(&mut s, 1, 3e-3, total);
    let win = (0.1 * fx.fs) as usize;
    let early = fit(&series[..win], 1.0 / fx.fs).0;
    let late = fit(&series[total - win..], 1.0 / fx.fs).0;
    let (ce, cl) = (cents(early, fx.t.f0), cents(late, fx.t.f0));
    println!("steel 3 mm pluck, T60 1 s: first 0.1 s {ce:+.3} cents, last 0.1 s {cl:+.5} cents");
    assert!(ce > 2.0, "the start is not sharp: {ce}");
    assert!(
        cl.abs() < 1e-3,
        "the note did not settle onto the target: {cl}"
    );
}

// -- the table and the refusals ---------------------------------------------------------------

#[test]
fn every_material_is_named_cited_and_found() {
    let names: Vec<&str> = STRING_MATERIALS.iter().map(|m| m.name).collect();
    assert_eq!(names, ["steel", "nylon", "gut"]);
    for m in STRING_MATERIALS {
        assert!(!m.source.is_empty(), "{} has no source", m.name);
        assert_eq!(string_material(m.name), Some(m));
    }
    assert_eq!(string_material("silk"), None);
}

#[test]
fn a_bad_gauge_or_material_is_refused_by_name() {
    let steel = string_material("steel").unwrap();
    let t = targets(196.0, 1e-4, 3.0, 0.8, 2000.0);
    let run = |m: &StringMaterial, d: f64| tune_tension_string(m, d, 0.65, 44100.0, 64, 0.28, &t);
    for d in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        assert_eq!(
            run(steel, d),
            Err(TuningError::NonPositive("diameter")),
            "d = {d}"
        );
    }
    let custom = |density, modulus| StringMaterial {
        name: "custom",
        density,
        modulus,
        source: "",
    };
    assert_eq!(
        run(&custom(0.0, Modulus::Fixed(1e9)), 1e-3),
        Err(TuningError::NonPositive("density"))
    );
    assert_eq!(
        run(&custom(1000.0, Modulus::Fixed(-1e9)), 1e-3),
        Err(TuningError::NonPositive("Young's modulus"))
    );
    // A fit that goes negative at this stress is refused, not used.
    let neg = Modulus::StressFit {
        a: -100.0,
        b: 0.0,
        c: 0.0,
    };
    assert_eq!(
        run(&custom(1000.0, neg), 1e-3),
        Err(TuningError::NonPositive("Young's modulus"))
    );
    // A target the damped string cannot meet is the damped mapping's refusal, passed through.
    let low_ref = targets(196.0, 1e-4, 3.0, 0.8, 150.0);
    assert_eq!(
        tune_tension_string(steel, 1e-3, 0.65, 44100.0, 64, 0.28, &low_ref),
        Err(TuningError::RefNotAboveFundamental)
    );
}
