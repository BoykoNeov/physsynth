//! The membrane's musical controls (`docs/dev/membrane-mapping-plan.md` §3–§5): heads, and the
//! mapping from pitch or tension and two ring times onto the simulated membrane.
//!
//! Every round trip runs the simulation and reads the pitch and ring time back from it — a run
//! started in one mode, fitted in the well-conditioned form — never the mapping's algebra run
//! backwards. The rectangle's modes are closed form here; the circle's come from the eigensolver,
//! and the read-back is what says the vector is a mode.

use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::membrane::{Domain, Membrane, ParamError, Params};
use physsynth_core::membrane_tuning::{
    head, lowest_modes, tune_membrane, Geometry, MembraneTuning, SetBy, Targets, TuningError,
    FILM_DENSITY, HEADS, MIL,
};
use std::f64::consts::{LN_10, PI};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cents(f: f64, g: f64) -> f64 {
    1200.0 * (f / g).log2()
}

fn geom(domain: Domain, size: f64, n: i64, fs: f64) -> Geometry {
    Geometry {
        domain,
        lx: size,
        ly: 0.73 * size,
        radius: size,
        n,
        fs,
    }
}

fn rho_of(name: &str) -> f64 {
    head(name).expect("a listed head").areal_density()
}

/// A unit eigenvector of the membrane's `−L` with eigenvalue `mu` — closed form on the rectangle,
/// from the solver on the circle.
fn mode_vector(p: &Params, mu: f64) -> Vec<f64> {
    match p.domain {
        Domain::Rectangle => {
            let (lx, ly, h) = (p.lx.unwrap(), p.ly.unwrap(), p.h);
            let (nx, ny) = ((lx / h).round() as usize, (ly / h).round() as usize);
            let s = |q: usize, l: f64| (q as f64 * PI * h / (2.0 * l)).sin().powi(2);
            let mut best = (1, 1, f64::INFINITY);
            for m in 1..nx {
                for n in 1..ny {
                    let d = (4.0 / (h * h) * (s(m, lx) + s(n, ly)) - mu).abs();
                    if d < best.2 {
                        best = (m, n, d);
                    }
                }
            }
            let (x, y) = (p.to_live(&p.x), p.to_live(&p.y));
            let mut v: Vec<f64> = x
                .iter()
                .zip(&y)
                .map(|(x, y)| {
                    (best.0 as f64 * PI * x / lx).sin() * (best.1 as f64 * PI * y / ly).sin()
                })
                .collect();
            let nn = dot(&v, &v).sqrt();
            v.iter_mut().for_each(|e| *e /= nn);
            v
        }
        Domain::Circle => {
            let pairs = eigsh_shift_invert(&p.l.scaled(-1.0), None, 0.999 * mu, 3).unwrap();
            let j = (0..pairs.values.len())
                .min_by(|&a, &b| {
                    (pairs.values[a] - mu)
                        .abs()
                        .total_cmp(&(pairs.values[b] - mu).abs())
                })
                .unwrap();
            assert!((pairs.values[j] - mu).abs() < 1e-9 * mu, "no mode at {mu}");
            pairs.vectors[j].clone()
        }
    }
}

/// Start the tuned membrane in mode `phi`, run, project, and fit `a_{n+1} = p a_n − q a_{n−1}` in
/// the second-difference form; return the run's (frequency, T60).
fn read_back(t: &MembraneTuning, phi: &[f64], steps: usize) -> (f64, f64) {
    let mut mem = Membrane::new(t.params.clone());
    mem.set_displacement(phi);
    let mut a = vec![dot(&mem.u_prev, phi), dot(&mem.u, phi)];
    for _ in 0..steps {
        mem.step();
        a.push(dot(&mem.u, phi));
    }
    let (mut saa, mut sad, mut sdd, mut sya, mut syd) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for n in 1..a.len() - 1 {
        let d = a[n] - a[n - 1];
        let y = (a[n + 1] - a[n]) - d;
        saa += a[n] * a[n];
        sad += a[n] * d;
        sdd += d * d;
        sya += y * a[n];
        syd += y * d;
    }
    let det = saa * sdd - sad * sad;
    let alpha = (sya * sdd - syd * sad) / det;
    let beta = (syd * saa - sya * sad) / det;
    let (q, p) = (1.0 + beta, alpha + 2.0 + beta);
    let k = t.params.k;
    let r = q.sqrt();
    let f = (p / (2.0 * r)).acos() / (2.0 * PI * k);
    (f, -6.0 * LN_10 * k / beta.ln_1p())
}

// -- heads ----------------------------------------------------------------------------------------

#[test]
fn every_head_is_film_density_times_its_plies_retyped() {
    // Retyped here from the constructions the presets cite and DuPont's 1.390 g/cm³ — not read
    // from the table's arithmetic.
    let expected = [
        ("single-ply 3 mil", 1390.0 * 3.0 * 25.4e-6),
        ("single-ply 7.5 mil", 1390.0 * 7.5 * 25.4e-6),
        ("single-ply 10 mil", 1390.0 * 10.0 * 25.4e-6),
        ("two-ply 7 + 7 mil", 1390.0 * 14.0 * 25.4e-6),
    ];
    assert_eq!(HEADS.len(), expected.len());
    for (name, rho) in expected {
        let got = rho_of(name);
        assert!(
            (got - rho).abs() < 1e-12 * rho,
            "{name}: {got} against {rho}"
        );
    }
    assert_eq!(FILM_DENSITY, 1390.0);
    assert_eq!(MIL, 25.4e-6);
    assert!(head("calfskin").is_none());
    let mut names: Vec<_> = HEADS.iter().map(|h| h.name).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), HEADS.len(), "head names must be unique");
    assert!(HEADS
        .iter()
        .all(|h| h.source.contains("Remo") && h.source.contains("Mylar")));
}

// -- the round trip, read from the running scheme -------------------------------------------------

#[test]
fn both_anchors_land_on_their_targets_in_the_simulation_both_shapes_both_ways() {
    let mut ran = 0;
    let (mut worst_c, mut worst_t) = (0.0f64, 0.0f64);
    for (domain, size) in [
        (Domain::Rectangle, 0.30),
        (Domain::Rectangle, 0.45),
        (Domain::Circle, 0.12),
        (Domain::Circle, 0.33),
    ] {
        for n in [30i64, 56] {
            for fs in [44_100.0, 96_000.0] {
                for (name, f0, t_low, t_high, mult) in [
                    ("single-ply 10 mil", 90.0, 3.0, 0.8, 4.2),
                    ("two-ply 7 + 7 mil", 210.0, 1.2, 0.25, 2.7),
                ] {
                    let g = geom(domain, size, n, fs);
                    let rho = rho_of(name);
                    let pitch = Targets {
                        set_by: SetBy::Pitch(f0),
                        t60_low: t_low,
                        t60_high: t_high,
                        f_ref: mult * f0,
                    };
                    let by_pitch = tune_membrane(&g, rho, &pitch).expect("a reachable drum");
                    // Set by the tension the pitch run worked out, scaled off it, so the read-out
                    // pitch is not the target.
                    let by_tension = tune_membrane(
                        &g,
                        rho,
                        &Targets {
                            set_by: SetBy::Tension(1.1 * by_pitch.tension),
                            ..pitch
                        },
                    )
                    .expect("a reachable drum");
                    for (t, f_target) in [(&by_pitch, f0), (&by_tension, by_tension.f0)] {
                        assert_eq!(t.params.sigma.to_bits(), t.sigma0.to_bits());
                        assert_eq!(t.params.sigma1.to_bits(), t.sigma1.to_bits());
                        assert_eq!(t.params.rho.to_bits(), rho.to_bits());
                        let (f1, t1) =
                            read_back(t, &mode_vector(&t.params, t.fundamental_mu), 3000);
                        let (_, tm) = read_back(t, &mode_vector(&t.params, t.anchor_mu), 3000);
                        worst_c = worst_c.max(cents(f1, f_target).abs());
                        worst_t = worst_t
                            .max(((t1 - t_low) / t_low).abs())
                            .max(((tm - t_high) / t_high).abs());
                        ran += 1;
                    }
                    let ratio = by_tension.f0 / by_pitch.f0;
                    assert!(
                        (ratio - 1.1f64.sqrt()).abs() < 1e-3,
                        "tension is not pitch²"
                    );
                }
            }
        }
    }
    assert_eq!(ran, 64);
    assert!(worst_c < 1e-6, "pitch off by {worst_c:.2e} cents");
    assert!(worst_t < 1e-7, "ring time off by {worst_t:.2e}");
}

#[test]
fn the_anchor_is_the_partial_nearest_f_ref() {
    for (domain, size) in [(Domain::Rectangle, 0.40), (Domain::Circle, 0.15)] {
        let g = geom(domain, size, 40, 44_100.0);
        for mult in [2.3, 3.9, 6.1] {
            let t = tune_membrane(
                &g,
                rho_of("single-ply 7.5 mil"),
                &Targets {
                    set_by: SetBy::Pitch(130.0),
                    t60_low: 2.0,
                    t60_high: 0.6,
                    f_ref: mult * 130.0,
                },
            )
            .expect("a reachable drum");
            let k = t.params.k;
            let c2 = t.params.c * t.params.c;
            let f = |mu: f64| (0.5 * (c2 * k * k * mu).sqrt()).asin() / (PI * k);
            let target = mult * 130.0;
            let nearest = lowest_modes(&t.params, 60)
                .unwrap()
                .into_iter()
                .skip(1)
                .map(f)
                .fold(f64::INFINITY, |b, x| {
                    if (x - target).abs() < (b - target).abs() {
                        x
                    } else {
                        b
                    }
                });
            assert!(
                (f(t.anchor_mu) - nearest).abs() < 1e-9 * nearest,
                "{domain:?} x{mult}: anchor {} Hz, nearest {nearest} Hz",
                f(t.anchor_mu)
            );
            assert!(t.anchor_margin > 0.0 && t.anchor_margin.is_finite());
        }
    }
}

// -- struck, and the textbook control -------------------------------------------------------------

/// A struck round head's fundamental through the analysis crate's detector.
fn struck_fundamental(p: &Params, guess: f64) -> f64 {
    let (x, y) = (p.to_live(&p.x), p.to_live(&p.y));
    let r = p.radius.unwrap();
    let u0: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(x, y)| {
            let d2 = (x - 0.2 * r).powi(2) + (y + 0.1 * r).powi(2);
            1e-3 * (-d2 / (0.3 * r).powi(2)).exp()
        })
        .collect();
    let pick = p.pickup_index_at(0.31 * r, 0.22 * r);
    let mut mem = Membrane::new(p.clone());
    mem.set_displacement(&u0);
    let mut sig = Vec::new();
    for _ in 0..(p.fs as usize) {
        mem.step();
        sig.push(mem.displacement_at(pick));
    }
    measure_partials_near(&sig, p.fs, &[guess], Some(0.2 * guess))[0]
}

#[test]
fn a_struck_round_head_sounds_f0_where_the_textbook_mapping_is_flat() {
    // The textbook mapping takes c from the continuum f₀₁ = j₀₁ c / (2πa); the staircased rim puts
    // the simulated drum well flat of it (plan §9: 10–25 cents at the viewer's grids). The mapping
    // lands on the target through the same detector.
    let (radius, n, fs, f0) = (0.12, 40i64, 44_100.0, 180.0);
    let g = geom(Domain::Circle, radius, n, fs);
    let rho = rho_of("single-ply 7.5 mil");
    let t = tune_membrane(
        &g,
        rho,
        &Targets {
            set_by: SetBy::Pitch(f0),
            t60_low: 3.0,
            t60_high: 0.7,
            f_ref: 700.0,
        },
    )
    .expect("a reachable drum");
    let mapped = struck_fundamental(&t.params, f0);

    let j01 = 2.404825557695773;
    let c = 2.0 * PI * radius * f0 / j01;
    let textbook = Params::new(
        Some(Domain::Circle),
        rho * c * c,
        rho,
        fs,
        n,
        None,
        None,
        Some(radius),
        t.sigma0,
    )
    .unwrap()
    .with_sigma1(t.sigma1)
    .unwrap();
    let flat = struck_fundamental(&textbook, f0);
    assert!(
        cents(mapped, f0).abs() < 0.05,
        "mapped {:.4} cents",
        cents(mapped, f0)
    );
    assert!(
        cents(flat, f0) < -5.0,
        "textbook only {:.3} cents off",
        cents(flat, f0)
    );
}

// -- the head's density is live -------------------------------------------------------------------

#[test]
fn the_head_density_changes_no_pitch_and_scales_every_energy() {
    // Two heads at the same targets: the same wave speed and losses, a tension and an energy in
    // the ratio of their densities — and each the membrane the raw constructor builds from them.
    let g = geom(Domain::Circle, 0.17, 36, 48_000.0);
    let targets = Targets {
        set_by: SetBy::Pitch(150.0),
        t60_low: 2.5,
        t60_high: 0.5,
        f_ref: 600.0,
    };
    let (light, heavy) = (rho_of("single-ply 3 mil"), rho_of("two-ply 7 + 7 mil"));
    let (a, b) = (
        tune_membrane(&g, light, &targets).unwrap(),
        tune_membrane(&g, heavy, &targets).unwrap(),
    );
    assert!((a.params.c - b.params.c).abs() < 1e-12 * a.params.c);
    assert_eq!(a.sigma0.to_bits(), b.sigma0.to_bits());
    assert_eq!(a.sigma1.to_bits(), b.sigma1.to_bits());
    assert!((b.tension / a.tension - heavy / light).abs() < 1e-12 * heavy / light);

    let u0 = vec![1e-4; a.params.n_live()];
    let energy = |p: &Params| {
        let mut m = Membrane::new(p.clone());
        m.set_displacement(&u0);
        m.step();
        m.energy()
    };
    let (ea, eb) = (energy(&a.params), energy(&b.params));
    assert!(
        ((eb / ea) - heavy / light).abs() < 1e-9 * heavy / light,
        "energy ratio {}",
        eb / ea
    );
    let raw = Params::new(
        Some(Domain::Circle),
        b.tension,
        heavy,
        g.fs,
        g.n,
        None,
        None,
        Some(g.radius),
        b.sigma0,
    )
    .unwrap()
    .with_sigma1(b.sigma1)
    .unwrap();
    assert_eq!(energy(&raw).to_bits(), eb.to_bits());
}

#[test]
fn set_by_pitch_then_by_its_tension_gives_the_pitch_back() {
    for domain in [Domain::Rectangle, Domain::Circle] {
        let g = geom(domain, 0.25, 40, 44_100.0);
        let rho = rho_of("single-ply 10 mil");
        let pitch = Targets {
            set_by: SetBy::Pitch(140.0),
            t60_low: 1.8,
            t60_high: 0.4,
            f_ref: 520.0,
        };
        let a = tune_membrane(&g, rho, &pitch).unwrap();
        let b = tune_membrane(
            &g,
            rho,
            &Targets {
                set_by: SetBy::Tension(a.tension),
                ..pitch
            },
        )
        .unwrap();
        assert!(cents(b.f0, 140.0).abs() < 1e-9, "{domain:?}: {} Hz", b.f0);
        assert!((b.sigma0 - a.sigma0).abs() <= 1e-12 * a.sigma0);
        assert!((b.sigma1 - a.sigma1).abs() <= 1e-12 * a.sigma1);
    }
}

// -- refusals -------------------------------------------------------------------------------------

#[test]
fn the_refusals() {
    let g = geom(Domain::Circle, 0.12, 40, 44_100.0);
    let rho = rho_of("single-ply 7.5 mil");
    let base = Targets {
        set_by: SetBy::Pitch(200.0),
        t60_low: 1.0,
        t60_high: 0.3,
        f_ref: 800.0,
    };
    let tune = |t: Targets| tune_membrane(&g, rho, &t);

    assert!(matches!(
        tune(Targets {
            f_ref: 150.0,
            ..base
        }),
        Err(TuningError::AnchorIsTheFundamental { .. })
    ));
    assert!(matches!(
        tune(Targets {
            f_ref: 30_000.0,
            ..base
        }),
        Err(TuningError::OutOfReach { what: "f_ref", .. })
    ));
    for t60_high in [5.0, 0.004] {
        match tune(Targets { t60_high, ..base }) {
            Err(TuningError::HighRingTimeOutOfRange {
                asked,
                shortest,
                longest,
            }) => {
                assert!(shortest < longest);
                assert!(asked < shortest || asked > longest);
                // Both ends are reachable: just inside each one tunes.
                for inside in [shortest * 1.01, longest * 0.99] {
                    assert!(tune(Targets {
                        t60_high: inside,
                        ..base
                    })
                    .is_ok());
                }
            }
            other => panic!("t60_high = {t60_high}: expected the range refusal, got {other:?}"),
        }
    }
    for (bad, what) in [
        (
            Targets {
                t60_low: 0.0,
                ..base
            },
            "t60_low",
        ),
        (
            Targets {
                set_by: SetBy::Pitch(-1.0),
                ..base
            },
            "f0",
        ),
        (
            Targets {
                set_by: SetBy::Tension(f64::NAN),
                ..base
            },
            "tension",
        ),
    ] {
        assert_eq!(tune(bad).err(), Some(TuningError::NonPositive(what)));
    }
    assert_eq!(
        tune_membrane(&g, 0.0, &base).err(),
        Some(TuningError::NonPositive("rho"))
    );

    // Too high a pitch for the grid: the plain CFL.
    match tune(Targets {
        set_by: SetBy::Pitch(4000.0),
        f_ref: 9000.0,
        ..base
    }) {
        Err(TuningError::Membrane(ParamError::CflViolated(_))) => {}
        other => panic!("expected the plain CFL refusal, got {other:?}"),
    }
    // Near the plain CFL ceiling, a short high ring time asks for more σ₁ than the explicit
    // scheme can hold: the loss-tightened bound.
    let h = 2.0 * 0.12 / 40.0;
    let c = 107.0;
    let near_ceiling = Geometry {
        fs: c / (0.70 * h),
        ..g
    };
    let t = tune_membrane(
        &near_ceiling,
        rho,
        &Targets {
            set_by: SetBy::Tension(rho * c * c),
            t60_low: 1.0,
            t60_high: 0.1,
            f_ref: 4.0 * 340.0,
        },
    );
    assert!(
        matches!(
            t,
            Err(TuningError::Membrane(ParamError::LossCflViolated(_)))
        ),
        "expected the loss CFL refusal, got {t:?}"
    );
}
