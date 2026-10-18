//! Musical controls for the plates (`docs/dev/plate-mapping-plan.md`): the materials half.

use physsynth_core::plate::{Boundary, Domain, MaterialError, Params, PlateSpec};
use physsynth_core::plate_tuning::{
    plate_material, preset, thickness_for_kappa, GrainAlong, Material, PRESETS,
};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn every_preset_builds_every_plate_with_its_grain_either_way() {
    for p in PRESETS {
        for along in [GrainAlong::X, GrainAlong::Y] {
            let m = plate_material(&p.material, along, 0.003).expect("a preset is admissible");
            for (boundary, domain) in [
                (Boundary::Supported, Domain::Rectangle),
                (Boundary::Free, Domain::Rectangle),
                (Boundary::Free, Domain::Guitar),
            ] {
                let mut s = PlateSpec {
                    lx: 0.4,
                    ly: 0.5,
                    fs: 44100.0,
                    n: 10,
                    boundary: Some(boundary),
                    domain: Some(domain),
                    ..PlateSpec::default()
                };
                m.apply(&mut s);
                Params::new(&s).unwrap_or_else(|e| {
                    panic!("{} {along:?} {boundary:?}/{domain:?}: {e}", p.name)
                });
            }
        }
    }
}

#[test]
fn the_handbook_footnotes_are_applied_to_sitka_spruce() {
    // Retyped from FPL-GTR-190 here, not read from the preset: bending modulus 10,800 MPa raised
    // 10 % for shear (Table 5-1's footnote), specific gravity 0.40 on volume at 12 % moisture, so
    // 448 kg/m³ with the water (Table 5-3a's footnote b).
    let Material::Wood {
        e_l,
        e_r,
        g_lr,
        nu_lr,
        rho,
    } = preset("sitka spruce").unwrap().material
    else {
        panic!("sitka spruce is a wood");
    };
    assert!(rel(e_l, 11.88e9) < 1e-15);
    assert!(rel(e_r, 0.078 * 11.88e9) < 1e-15);
    assert!(rel(g_lr, 0.064 * 11.88e9) < 1e-15);
    assert_eq!(nu_lr, 0.372);
    assert!(rel(rho, 448.0) < 1e-15);
}

#[test]
fn an_isotropic_plate_gets_the_textbook_stiffness_and_an_areal_density() {
    // κ = sqrt(D / ρ_s) with D = E t³ / 12(1 − ν²) and ρ_s = ρ t, written out here.
    for p in PRESETS {
        let Material::Isotropic { e, nu, rho } = p.material else {
            continue;
        };
        for t in [0.0005, 0.002, 0.011] {
            let m = plate_material(&p.material, GrainAlong::X, t).unwrap();
            let kappa = t * (e / (12.0 * rho * (1.0 - nu * nu))).sqrt();
            assert!(rel(m.kappa, kappa) < 1e-14, "{} t={t}", p.name);
            // the trap: a volume density here leaves every pitch right and every energy wrong
            assert!(
                rel(m.rho_s, rho * t) < 1e-15,
                "{} t={t}: rho_s {}",
                p.name,
                m.rho_s
            );
            assert_eq!(m.nu, Some(nu));
            assert_eq!((m.grain_x, m.grain_y), (1.0, 1.0));
            assert!(m.grain_cross.is_none() && m.grain_coupling.is_none());
        }
    }
}

#[test]
fn turning_a_woods_grain_swaps_its_two_bending_rigidities_and_nothing_else() {
    // Rigidities recovered from what the plate is handed: D_x = κ² ρ_s, the rest as ratios to it.
    // Grain along y must be grain along x seen from the other side — D_x and D_y swap, the cross,
    // coupling and torsion terms stay. An unrotated Poisson ratio breaks D_1 and H (or is refused).
    let rig = |p: &physsynth_core::plate_tuning::PlateMaterial| {
        let dx = p.kappa * p.kappa * p.rho_s;
        (
            dx,
            dx * p.grain_y,
            dx * p.grain_cross.unwrap(),
            dx * p.grain_coupling.unwrap(),
            dx * p.grain_torsion.unwrap(),
        )
    };
    let mut woods = 0;
    for p in PRESETS {
        if !matches!(p.material, Material::Wood { .. }) {
            continue;
        }
        woods += 1;
        let x = rig(&plate_material(&p.material, GrainAlong::X, 0.003).unwrap());
        let y = rig(&plate_material(&p.material, GrainAlong::Y, 0.003).unwrap());
        assert!(
            rel(y.0, x.1) < 1e-12 && rel(y.1, x.0) < 1e-12,
            "{}: D_x, D_y",
            p.name
        );
        assert!(rel(y.2, x.2) < 1e-12, "{}: H", p.name);
        assert!(rel(y.3, x.3) < 1e-12, "{}: D_1", p.name);
        assert!(rel(y.4, x.4) < 1e-12, "{}: D_xy", p.name);
        // the control: a wood is genuinely orthotropic, so the swap moved something
        assert!(
            rel(x.0, x.1) > 1.0,
            "{}: D_x and D_y are not different",
            p.name
        );
    }
    assert!(woods >= 5);
}

#[test]
fn the_thickness_comes_back_from_its_own_stiffness() {
    for p in PRESETS {
        for along in [GrainAlong::X, GrainAlong::Y] {
            for t in [0.0007, 0.0027, 0.019] {
                let kappa = plate_material(&p.material, along, t).unwrap().kappa;
                let back = thickness_for_kappa(&p.material, along, kappa).unwrap();
                assert!(rel(back, t) < 1e-14, "{} {along:?} t={t}: {back}", p.name);
            }
        }
    }
}

#[test]
fn what_no_plate_can_be_is_refused() {
    let steel = preset("steel").unwrap().material;
    for t in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        assert_eq!(
            plate_material(&steel, GrainAlong::X, t),
            Err(MaterialError::NonPositive),
            "t = {t}"
        );
    }
    for kappa in [0.0, -1.0, f64::NAN] {
        assert!(thickness_for_kappa(&steel, GrainAlong::X, kappa).is_err());
    }
    let rubbery = Material::Isotropic {
        e: 1e9,
        nu: 0.5,
        rho: 1000.0,
    };
    assert!(matches!(
        plate_material(&rubbery, GrainAlong::X, 0.001),
        Err(MaterialError::Inadmissible { .. })
    ));
    // A made-up wood whose Poisson pair breaks thermodynamics: ν_LR ν_RL = ν_LR² E_R/E_L ≥ 1.
    let impossible = Material::Wood {
        e_l: 1e10,
        e_r: 1e10,
        g_lr: 1e9,
        nu_lr: 1.2,
        rho: 500.0,
    };
    for along in [GrainAlong::X, GrainAlong::Y] {
        assert!(matches!(
            plate_material(&impossible, along, 0.001),
            Err(MaterialError::Inadmissible { .. })
        ));
    }
}

#[test]
fn every_preset_names_its_source_and_names_are_unique() {
    let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PRESETS.len());
    assert!(PRESETS.iter().all(|p| !p.source.is_empty()));
    assert!(preset("unobtainium").is_none());
}

// == the mapping ===================================================================================

use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::eig::generalized_eigen_diag;
use physsynth_core::plate::Plate;
use physsynth_core::plate_tuning::{tune_plate, PlateTuning, SetBy, Targets, TuningError};
use std::f64::consts::PI;

fn geometry(boundary: Boundary, domain: Domain, n: i64, fs: f64, theta: f64) -> PlateSpec {
    let (lx, ly) = if domain == Domain::Guitar {
        (0.4, 0.5)
    } else {
        (0.5, 0.37)
    };
    PlateSpec {
        lx,
        ly,
        fs,
        n,
        theta,
        boundary: Some(boundary),
        domain: Some(domain),
        ..PlateSpec::default()
    }
}

fn built(g: &PlateSpec, t: &PlateTuning) -> Plate {
    let mut s = g.clone();
    t.apply(&mut s);
    Plate::new(Params::new(&s).expect("a tuned plate builds"))
}

/// `(f, T60)` read back from a mode's projected amplitudes — the well-conditioned fit Part 0 found
/// (the damped string's bar): regress the second difference on `(a_n, a_n − a_{n−1})`, so
/// `D² = g a + d v` with `d = r² − 1`, and take `log r` through `ln_1p`.
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

/// Start `pl` in `shape`, step, and fit its projection under `weight` (the mass).
fn play(pl: &mut Plate, shape: &[f64], weight: &[f64], steps: usize) -> (f64, f64) {
    let ws: Vec<f64> = shape.iter().zip(weight).map(|(a, b)| a * b).collect();
    let proj = |u: &[f64]| u.iter().zip(&ws).map(|(a, b)| a * b).sum::<f64>();
    pl.set_state(shape, &vec![0.0; shape.len()]);
    let mut amp = vec![proj(&pl.u)];
    for _ in 0..steps {
        pl.step(None);
        amp.push(proj(&pl.u));
    }
    fit(&amp, pl.p.k)
}

/// The supported plate's Navier shape whose stiffness eigenvalue is `beta`, found here from the
/// closed form — the bar's own copy of it.
fn navier_shape(p: &Params, beta: f64) -> Vec<f64> {
    let (gx, gc, gy) = if p.grain_is_isotropic {
        (1.0, 1.0, 1.0)
    } else {
        (p.grain_x, p.grain_cross, p.grain_y)
    };
    let h2 = p.h * p.h;
    let mut best = (f64::INFINITY, 1usize, 1usize);
    for pm in 1..p.n {
        let d = 4.0 / h2 * (pm as f64 * PI / (2.0 * p.n as f64)).sin().powi(2);
        for qm in 1..p.ny {
            let e = 4.0 / h2 * (qm as f64 * PI / (2.0 * p.ny as f64)).sin().powi(2);
            let b = gx * d * d + 2.0 * gc * d * e + gy * e * e;
            if (b / beta - 1.0).abs() < best.0 {
                best = ((b / beta - 1.0).abs(), pm, qm);
            }
        }
    }
    assert!(best.0 < 1e-12, "no Navier mode at beta = {beta}");
    let flags = p.mask.flags();
    (0..flags.len())
        .filter(|&i| flags[i])
        .map(|i| {
            (best.1 as f64 * PI * p.x[i] / p.lx).sin() * (best.2 as f64 * PI * p.y[i] / p.ly).sin()
        })
        .collect()
}

/// Every Navier mode of the built plate as `(β, lossless frequency)`, from the closed form and the
/// lossless θ-scheme relation `sin²(πfk) = X / 4(1 + θX)`, `X = k²κ²β` — the bar's own copy.
fn navier_lossless(p: &Params) -> Vec<(f64, f64)> {
    let (gx, gc, gy) = if p.grain_is_isotropic {
        (1.0, 1.0, 1.0)
    } else {
        (p.grain_x, p.grain_cross, p.grain_y)
    };
    let h2 = p.h * p.h;
    let mut out = Vec::new();
    for pm in 1..p.n {
        let d = 4.0 / h2 * (pm as f64 * PI / (2.0 * p.n as f64)).sin().powi(2);
        for qm in 1..p.ny {
            let e = 4.0 / h2 * (qm as f64 * PI / (2.0 * p.ny as f64)).sin().powi(2);
            let b = gx * d * d + 2.0 * gc * d * e + gy * e * e;
            let x = p.k * p.k * p.kappa * p.kappa * b;
            let s = x / (4.0 * (1.0 + p.theta * x));
            if s < 1.0 {
                out.push((b, s.sqrt().asin() / (PI * p.k)));
            }
        }
    }
    out
}

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

#[test]
fn a_supported_plate_meets_its_targets_in_the_simulation_off_every_default() {
    // Both anchors read back from the running scheme, to rounding: the loss form shares the modes.
    let materials = [
        ("steel", GrainAlong::X),
        ("sitka spruce", GrainAlong::X),
        ("sitka spruce", GrainAlong::Y),
    ];
    let targets = [
        Targets {
            set_by: SetBy::Pitch(220.0),
            t60_low: 3.0,
            t60_high: 0.6,
            f_ref: 2000.0,
        },
        Targets {
            set_by: SetBy::Thickness(0.002),
            t60_low: 1.5,
            t60_high: 0.2,
            f_ref: 3000.0,
        },
    ];
    let (mut worst_c, mut worst_t, mut runs) = (0.0f64, 0.0f64, 0);
    for (name, along) in materials {
        let mat = preset(name).unwrap().material;
        for n in [12, 20] {
            for fs in [44100.0, 96000.0] {
                for theta in [0.25, 0.28, 0.5, 1.0] {
                    for tg in &targets {
                        let g = geometry(Boundary::Supported, Domain::Rectangle, n, fs, theta);
                        let tu = tune_plate(&g, &mat, along, tg).expect("reachable targets");
                        assert!(tu.exact);
                        let mut pl = built(&g, &tu);
                        let ones = vec![1.0; pl.p.n_live];
                        let s1 = navier_shape(&pl.p, tu.fundamental.beta);
                        let (f1, t1) = play(&mut pl, &s1, &ones, 1500);
                        // The anchor is the partial nearest f_ref: no other lands closer. (A
                        // mapping that anchored elsewhere would still meet its targets there.)
                        let modes = navier_lossless(&pl.p);
                        let (_, nearest) = modes
                            .iter()
                            .copied()
                            .min_by(|a, b| {
                                (a.1 - tg.f_ref).abs().total_cmp(&(b.1 - tg.f_ref).abs())
                            })
                            .unwrap();
                        let (_, mine) = modes
                            .iter()
                            .copied()
                            .find(|m| (m.0 / tu.anchor.beta - 1.0).abs() < 1e-12)
                            .unwrap();
                        assert!(
                            (mine - nearest).abs() <= 1e-9 * nearest,
                            "{name}: {nearest} Hz is nearer {} than the anchor ({mine} Hz)",
                            tg.f_ref
                        );
                        let sm = navier_shape(&pl.p, tu.anchor.beta);
                        let (fm, tm) = play(&mut pl, &sm, &ones, 1500);
                        if let SetBy::Pitch(f0) = tg.set_by {
                            worst_c = worst_c.max(cents(f1, f0).abs());
                        }
                        worst_c = worst_c.max(cents(f1, tu.f0).abs());
                        worst_c = worst_c.max(cents(fm, tu.anchor_frequency).abs());
                        worst_t = worst_t.max((t1 / tg.t60_low - 1.0).abs());
                        worst_t = worst_t.max((tm / tg.t60_high - 1.0).abs());
                        runs += 1;
                    }
                }
            }
        }
    }
    println!("supported, {runs} fixtures: worst {worst_c:.2e} cents, T60 {worst_t:.2e}");
    assert!(worst_c < 1e-6, "pitch {worst_c:.3e} cents");
    assert!(worst_t < 1e-6, "T60 {worst_t:.3e}");
}

#[test]
fn a_free_plate_meets_its_targets_within_the_coupling_part_0_measured() {
    // Not exact (the free loss form couples the modes); the mode starts in a (K, W) eigenvector
    // and its projection is fitted. Part 0 measured the scheme's own roots at <= 0.05 cents and
    // <= 0.3 % in T60; the projection adds the neighbours' leakage, so the bar is 0.1 cents, 1 %.
    let (mut worst_c, mut worst_t) = (0.0f64, 0.0f64);
    for (domain, n) in [(Domain::Rectangle, 10), (Domain::Guitar, 12)] {
        for (name, along) in [("steel", GrainAlong::X), ("sitka spruce", GrainAlong::Y)] {
            for theta in [0.28, 1.0] {
                let mat = preset(name).unwrap().material;
                let g = geometry(Boundary::Free, domain, n, 44100.0, theta);
                let tg = Targets {
                    set_by: SetBy::Pitch(180.0),
                    t60_low: 2.5,
                    t60_high: 1.2,
                    f_ref: 1500.0,
                };
                let tu = tune_plate(&g, &mat, along, &tg).expect("reachable targets");
                assert!(!tu.exact);
                let mut pl = built(&g, &tu);
                let nl = pl.p.n_live;
                let mut kd = vec![0.0; nl * nl];
                for i in 0..nl {
                    for j in 0..nl {
                        kd[i * nl + j] = pl.p.stiffness.get(i, j);
                    }
                }
                let (beta, vecs) = generalized_eigen_diag(&kd, &pl.p.w, nl).unwrap();
                let shape_at = |b: f64| {
                    let j = (0..nl)
                        .min_by(|&a, &c| (beta[a] - b).abs().total_cmp(&(beta[c] - b).abs()))
                        .unwrap();
                    assert!((beta[j] / b - 1.0).abs() < 1e-8, "no mode at beta = {b}");
                    (0..nl).map(|i| vecs[i * nl + j]).collect::<Vec<f64>>()
                };
                let w = pl.p.w.clone();
                let (f1, t1) = play(&mut pl, &shape_at(tu.fundamental.beta), &w, 1500);
                let (fm, tm) = play(&mut pl, &shape_at(tu.anchor.beta), &w, 1500);
                worst_c = worst_c.max(cents(f1, 180.0).abs());
                worst_c = worst_c.max(cents(fm, tu.anchor_frequency).abs());
                worst_t = worst_t.max((t1 / tg.t60_low - 1.0).abs());
                worst_t = worst_t.max((tm / tg.t60_high - 1.0).abs());
            }
        }
    }
    println!("free: worst {worst_c:.2e} cents, T60 {worst_t:.2e}");
    assert!(worst_c < 0.1, "pitch {worst_c:.3e} cents");
    assert!(worst_t < 0.01, "T60 {worst_t:.3e}");
}

#[test]
fn a_struck_supported_plate_sounds_both_anchors_where_the_mapping_says() {
    // A broadband strike and the analysis crate's partial detector on the output — the ear's view.
    let mat = preset("brass").unwrap().material;
    let g = geometry(Boundary::Supported, Domain::Rectangle, 16, 44100.0, 0.28);
    let tg = Targets {
        set_by: SetBy::Pitch(196.0),
        t60_low: 4.0,
        t60_high: 1.0,
        f_ref: 1200.0,
    };
    let tu = tune_plate(&g, &mat, GrainAlong::X, &tg).unwrap();
    let mut pl = built(&g, &tu);
    let p = pl.p.clone();
    let flags = p.mask.flags();
    let u0: Vec<f64> = (0..flags.len())
        .filter(|&i| flags[i])
        .map(|i| {
            let (dx, dy) = (p.x[i] - 0.31 * p.lx, p.y[i] - 0.27 * p.ly);
            1e-4 * (-(dx * dx + dy * dy) / (0.04 * p.lx * p.lx)).exp()
        })
        .collect();
    pl.set_state(&u0, &vec![0.0; u0.len()]);
    let pick = physsynth_core::plate::pickup_index_at(0.63 * p.lx, 0.71 * p.ly, &p);
    let mut out = Vec::with_capacity(88200);
    for _ in 0..88200 {
        pl.step(None);
        out.push(pl.u[pick]);
    }
    let got = measure_partials_near(&out, 44100.0, &[196.0, tu.anchor_frequency], Some(20.0));
    let (c1, cm) = (cents(got[0], 196.0), cents(got[1], tu.anchor_frequency));
    println!("struck: fundamental {c1:.4} cents, anchor {cm:.4} cents");
    // Two seconds of a decaying partial and a parabolic peak refinement.
    assert!(c1.abs() < 0.05 && cm.abs() < 0.05, "{c1} / {cm} cents");
}

#[test]
fn the_textbook_stiffness_misses_where_the_corrected_one_lands() {
    // The continuum fundamental f₁₁ = (π/2) κ (1/Lx² + 1/Ly²) gives the κ a textbook user would
    // pick; on the grid that plate is audibly off, and the bar asserts the miss exists.
    let mat = preset("steel").unwrap().material;
    let g = geometry(Boundary::Supported, Domain::Rectangle, 12, 44100.0, 0.28);
    let tg = Targets {
        set_by: SetBy::Pitch(262.0),
        t60_low: 3.0,
        t60_high: 0.8,
        f_ref: 2000.0,
    };
    let tu = tune_plate(&g, &mat, GrainAlong::X, &tg).unwrap();
    let mut pl = built(&g, &tu);
    let ones = vec![1.0; pl.p.n_live];
    let shape = navier_shape(&pl.p, tu.fundamental.beta);
    let (f_mapped, _) = play(&mut pl, &shape, &ones, 1500);

    let mut book = tu;
    let (lx, ly) = (pl.p.lx, pl.p.ly);
    book.material.kappa = 262.0 / (0.5 * PI * (1.0 / (lx * lx) + 1.0 / (ly * ly)));
    let mut pb = built(&g, &book);
    let (f_book, _) = play(&mut pb, &shape, &ones, 1500);
    let (miss, hit) = (cents(f_book, 262.0), cents(f_mapped, 262.0));
    println!("textbook {miss:.3} cents, corrected {hit:.2e} cents");
    assert!(hit.abs() < 1e-6);
    assert!(
        miss.abs() > 3.0,
        "the textbook plate is only {miss} cents off here"
    );
}

#[test]
fn a_thickness_set_by_pitch_plays_that_pitch_when_set_directly() {
    for (boundary, domain) in [
        (Boundary::Supported, Domain::Rectangle),
        (Boundary::Free, Domain::Guitar),
    ] {
        let mat = preset("western redcedar").unwrap().material;
        let g = geometry(boundary, domain, 14, 48000.0, 0.28);
        let by_pitch = Targets {
            set_by: SetBy::Pitch(150.0),
            t60_low: 1.2,
            t60_high: 0.3,
            f_ref: 1100.0,
        };
        let a = tune_plate(&g, &mat, GrainAlong::Y, &by_pitch).unwrap();
        let b = tune_plate(
            &g,
            &mat,
            GrainAlong::Y,
            &Targets {
                set_by: SetBy::Thickness(a.thickness),
                ..by_pitch
            },
        )
        .unwrap();
        assert!(
            (b.f0 / 150.0 - 1.0).abs() < 1e-10,
            "{boundary:?}: {} Hz",
            b.f0
        );
        assert!((b.sigma0 / a.sigma0 - 1.0).abs() < 1e-9);
        assert!((b.sigma1 / a.sigma1 - 1.0).abs() < 1e-9);
    }
}

#[test]
fn what_the_mapping_cannot_reach_is_refused_by_name() {
    let mat = preset("aluminium").unwrap().material;
    let g = geometry(Boundary::Supported, Domain::Rectangle, 12, 44100.0, 0.28);
    let ok = Targets {
        set_by: SetBy::Pitch(220.0),
        t60_low: 2.0,
        t60_high: 0.5,
        f_ref: 1800.0,
    };
    assert!(tune_plate(&g, &mat, GrainAlong::X, &ok).is_ok());
    let refuse = |t: Targets| tune_plate(&g, &mat, GrainAlong::X, &t).unwrap_err();
    assert!(matches!(
        refuse(Targets { f_ref: 150.0, ..ok }),
        TuningError::AnchorIsTheFundamental { .. }
    ));
    // Too long and too short both name the same range; its ends are the two one-loss plates.
    let range = |e: TuningError| match e {
        TuningError::HighRingTimeOutOfRange {
            shortest, longest, ..
        } => (shortest, longest),
        other => panic!("expected the ring-time range, got {other:?}"),
    };
    let (lo, hi) = range(refuse(Targets {
        t60_high: 3.0,
        ..ok
    }));
    assert_eq!(
        (lo, hi),
        range(refuse(Targets {
            t60_high: 0.01,
            ..ok
        }))
    );
    // The long end sits a little ABOVE t60_low: the scheme suppresses a high mode's decay rate
    // (hurdle §4), so equal loss rates ring longer up there.
    assert!(lo < 0.5 && 0.5 < hi, "[{lo}, {hi}]");
    let at = |t60_high: f64| tune_plate(&g, &mat, GrainAlong::X, &Targets { t60_high, ..ok });
    let short = at(lo * (1.0 + 1e-9)).expect("just inside the short end");
    let long = at(hi * (1.0 - 1e-9)).expect("just inside the long end");
    assert!(short.sigma0.abs() < 1e-6 * short.sigma1 * short.fundamental.lambda);
    assert!(long.sigma1 * long.fundamental.lambda < 1e-6 * long.sigma0);
    assert!(matches!(
        refuse(Targets {
            set_by: SetBy::Pitch(30_000.0),
            ..ok
        }),
        TuningError::OutOfReach { what: "f0", .. }
    ));
    assert!(matches!(
        refuse(Targets {
            f_ref: 25_000.0,
            ..ok
        }),
        TuningError::OutOfReach { what: "f_ref", .. }
    ));
    for bad in [0.0, -1.0, f64::NAN] {
        assert_eq!(
            refuse(Targets { t60_low: bad, ..ok }),
            TuningError::NonPositive("t60_low")
        );
        assert_eq!(
            refuse(Targets {
                set_by: SetBy::Thickness(bad),
                ..ok
            }),
            TuningError::NonPositive("thickness")
        );
    }
}
