//! Musical controls for the gong and the cymbal (`docs/dev/vk-mapping-plan.md` §3, bars §5.2–§5.8).
//!
//! [`tune_vk_plate`] is the plate mapping behind an adapter, so what is new here is the coupling:
//! every bar runs the **nonlinear** gong, never the mapping's algebra backwards. The targets are
//! met as a strike tends to zero (the bend grows as the strike's square, plan §8.2), so the
//! round trips strike at a ten-thousandth of a thickness with the coupling on, and the bars that
//! follow show what a harder strike does: it starts above the note and glides down onto it.

use physsynth_core::eig::generalized_eigen_diag;
use physsynth_core::plate::{Boundary, Params, VkParams, VkPlate, VkSpec};
use physsynth_core::plate_tuning::{
    preset, tune_vk_plate, Material, SetBy, Targets, TuningError, VkTuning,
};
use std::f64::consts::PI;

const METALS: [&str; 5] = ["steel", "brass", "bronze", "aluminium", "glass"];

/// A gong of outline `lx × ly` on `n` segments at `fs`; its material and losses come from a tuning.
fn geometry(boundary: Boundary, lx: f64, ly: f64, n: i64, fs: f64) -> VkSpec {
    VkSpec {
        lx,
        ly,
        fs,
        n,
        boundary: Some(boundary),
        ..VkSpec::default()
    }
}

fn metal(name: &str) -> Material {
    preset(name).expect("a preset").material
}

fn built(g: &VkSpec, t: &VkTuning) -> VkPlate {
    let mut s = g.clone();
    t.apply(&mut s);
    VkPlate::new(VkParams::new(&s).expect("a tuned gong builds"))
}

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

/// `(f, T60)` from a mode's projected amplitudes — the plate mapping's well-conditioned fit:
/// regress the second difference on `(a_n, a_n − a_{n−1})` and take `log r` through `ln_1p`.
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

/// The pitch (Hz) of a signal from its interpolated zero crossings about its mean.
fn zc_pitch(sig: &[f64], fs: f64) -> f64 {
    let m = sig.iter().sum::<f64>() / sig.len() as f64;
    let t: Vec<f64> = (0..sig.len() - 1)
        .filter(|&i| (sig[i] - m).is_sign_negative() != (sig[i + 1] - m).is_sign_negative())
        .map(|i| {
            let (a, b) = (sig[i] - m, sig[i + 1] - m);
            i as f64 + a / (a - b)
        })
        .collect();
    assert!(t.len() >= 3, "too few zero crossings to read a pitch");
    fs * (t.len() - 1) as f64 / (2.0 * (t[t.len() - 1] - t[0]))
}

/// Start `vk` in `shape` scaled to a peak of `amp`, at rest, and record its projection under
/// `weight` for `steps` steps. Asserts the coupling is on and engaged, so a gentle round trip is a
/// run of the gong and not of its linear half.
fn projection(vk: &mut VkPlate, shape: &[f64], weight: &[f64], amp: f64, steps: usize) -> Vec<f64> {
    assert!(vk.p.nonlinear, "the coupling is off");
    let peak = shape.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let u0: Vec<f64> = shape.iter().map(|x| amp * x / peak).collect();
    let ws: Vec<f64> = shape.iter().zip(weight).map(|(a, b)| a * b).collect();
    let proj = |u: &[f64]| u.iter().zip(&ws).map(|(a, b)| a * b).sum::<f64>();
    vk.set_state(&u0, &vec![0.0; u0.len()]).unwrap();
    let mut out = vec![proj(&vk.u)];
    let mut membrane = 0.0f64;
    for _ in 0..steps {
        vk.step(None).unwrap();
        assert!(vk.converged, "a step did not settle");
        membrane = membrane.max(vk.membrane_energy());
        out.push(proj(&vk.u));
    }
    assert!(membrane > 0.0, "the membrane never engaged");
    out
}

/// The supported gong's Navier mode whose stiffness eigenvalue is `beta`, from the closed form.
fn navier_shape(p: &Params, beta: f64) -> Vec<f64> {
    let h2 = p.h * p.h;
    let mut best = (f64::INFINITY, 1usize, 1usize);
    for pm in 1..p.n {
        let d = 4.0 / h2 * (pm as f64 * PI / (2.0 * p.n as f64)).sin().powi(2);
        for qm in 1..p.ny {
            let e = 4.0 / h2 * (qm as f64 * PI / (2.0 * p.ny as f64)).sin().powi(2);
            let b = (d + e) * (d + e);
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

/// The free cymbal's `(K, W)` eigenvector at `beta`, by a dense solve (small grids only).
fn free_shape(p: &Params, beta: f64) -> Vec<f64> {
    let nl = p.n_live;
    let mut kd = vec![0.0; nl * nl];
    for i in 0..nl {
        for j in 0..nl {
            kd[i * nl + j] = p.stiffness.get(i, j);
        }
    }
    let (b, vecs) = generalized_eigen_diag(&kd, &p.w, nl).unwrap();
    let j = (0..nl)
        .min_by(|&a, &c| (b[a] - beta).abs().total_cmp(&(b[c] - beta).abs()))
        .unwrap();
    assert!((b[j] / beta - 1.0).abs() < 1e-8, "no mode at beta = {beta}");
    (0..nl).map(|i| vecs[i * nl + j]).collect()
}

/// A ten-thousandth of a thickness. The miss the round trips read is the bend, which grows as the
/// strike's square (plan §8.2): measured at 2.6e-6 cents and 6.6e-9 in ring time here, and 100×
/// that at 1e-3 — the supported gong's mapping itself is exact to rounding.
const GENTLE: f64 = 1e-4;

// -- §5.2 the gentle-strike round trip, the coupling on -------------------------------------------

#[test]
fn a_gently_struck_gong_meets_its_targets_off_every_default() {
    // Every metal, both set-by modes, two grids and two sample rates, an oblong outline: mode 1
    // and the anchor read back from the running nonlinear gong: pitch to 1e-4 cents, ring time to
    // the linear plate's own 1e-6.
    let (mut worst_c, mut worst_t, mut runs) = (0.0f64, 0.0f64, 0);
    for name in METALS {
        for (n, fs) in [(12, 32_000.0), (20, 32_000.0), (12, 96_000.0)] {
            for set_by in [SetBy::Pitch(100.0), SetBy::Thickness(3.0e-3)] {
                let g = geometry(Boundary::Supported, 0.4, 0.31, n, fs);
                let tg = Targets {
                    set_by,
                    t60_low: 2.0,
                    t60_high: 0.5,
                    f_ref: 1500.0,
                };
                let tu = tune_vk_plate(&g, &metal(name), &tg).expect("reachable targets");
                assert!(tu.plate.exact);
                let mut vk = built(&g, &tu);
                let ones = vec![1.0; vk.p.lin.n_live];
                let e = vk.p.thickness;
                let k = vk.p.lin.k;
                for (beta, f_want, t_want) in [
                    (tu.plate.fundamental.beta, tu.plate.f0, tg.t60_low),
                    (tu.plate.anchor.beta, tu.plate.anchor_frequency, tg.t60_high),
                ] {
                    let shape = navier_shape(&vk.p.lin, beta);
                    let (f, t) = fit(&projection(&mut vk, &shape, &ones, GENTLE * e, 800), k);
                    worst_c = worst_c.max(cents(f, f_want).abs());
                    worst_t = worst_t.max((t / t_want - 1.0).abs());
                }
                if let SetBy::Pitch(f0) = set_by {
                    worst_c = worst_c.max(cents(tu.plate.f0, f0).abs());
                }
                runs += 1;
            }
        }
    }
    println!("supported, {runs} gongs: worst {worst_c:.2e} cents, T60 {worst_t:.2e}");
    assert!(worst_c < 1e-4, "pitch {worst_c:.3e} cents");
    assert!(worst_t < 1e-6, "T60 {worst_t:.3e}");
}

#[test]
fn a_gently_struck_cymbal_meets_its_targets_within_the_free_plates_bound() {
    // First-order exact (the free loss form couples the modes): the plate mapping's free bar is
    // 0.1 cents and 1 %; measured here at 2.2e-5 cents and 3.2e-7.
    let (mut worst_c, mut worst_t) = (0.0f64, 0.0f64);
    for name in ["bronze", "glass"] {
        for set_by in [SetBy::Pitch(100.0), SetBy::Thickness(2.0e-3)] {
            let g = geometry(Boundary::Free, 0.4, 0.31, 12, 44_100.0);
            let tg = Targets {
                set_by,
                t60_low: 2.5,
                t60_high: 1.2,
                f_ref: 1200.0,
            };
            let tu = tune_vk_plate(&g, &metal(name), &tg).expect("reachable targets");
            assert!(!tu.plate.exact);
            let mut vk = built(&g, &tu);
            let w = vk.p.lin.w.clone();
            let (e, k) = (vk.p.thickness, vk.p.lin.k);
            for (beta, f_want, t_want) in [
                (tu.plate.fundamental.beta, tu.plate.f0, tg.t60_low),
                (tu.plate.anchor.beta, tu.plate.anchor_frequency, tg.t60_high),
            ] {
                let shape = free_shape(&vk.p.lin, beta);
                let (f, t) = fit(&projection(&mut vk, &shape, &w, GENTLE * e, 1500), k);
                worst_c = worst_c.max(cents(f, f_want).abs());
                worst_t = worst_t.max((t / t_want - 1.0).abs());
            }
        }
    }
    println!("free: worst {worst_c:.2e} cents, T60 {worst_t:.2e}");
    assert!(worst_c < 0.1, "pitch {worst_c:.3e} cents");
    assert!(worst_t < 0.01, "T60 {worst_t:.3e}");
}

// -- §5.3, §5.4 a harder strike starts above the note and settles onto it -------------------------

#[test]
fn a_hard_strike_starts_above_the_note_and_settles_onto_it() {
    // The supported gong in mode 1 at two thicknesses: the first 40 ms read well above the target
    // (Part 0: +277 cents on a square), and with the loss on every later window is lower, the last
    // back on the note. The gentle round trip above is not passing because the coupling is off.
    let g = geometry(Boundary::Supported, 0.4, 0.31, 16, 32_000.0);
    let tg = Targets {
        set_by: SetBy::Pitch(200.0),
        t60_low: 0.6,
        t60_high: 0.3,
        f_ref: 2000.0,
    };
    let tu = tune_vk_plate(&g, &metal("steel"), &tg).expect("reachable targets");
    let mut vk = built(&g, &tu);
    let shape = navier_shape(&vk.p.lin, tu.plate.fundamental.beta);
    let ones = vec![1.0; shape.len()];
    let e = vk.p.thickness;
    let a = projection(&mut vk, &shape, &ones, 2.0 * e, 19_200);
    let win = 1280; // 40 ms
    let read: Vec<f64> = [0usize, 3, 7, 14]
        .iter()
        .map(|&w| cents(zc_pitch(&a[w * win..(w + 1) * win], 32_000.0), 200.0))
        .collect();
    println!("windows at 0, 120, 280, 560 ms: {read:+.1?} cents");
    assert!(
        read[0] > 100.0,
        "a 2 e strike starts only {:+.1} cents up",
        read[0]
    );
    assert!(
        read.windows(2).all(|w| w[1] < w[0]),
        "the note does not glide down: {read:?}"
    );
    assert!(
        read[3].abs() < 5.0,
        "it ends {:+.2} cents off the note",
        read[3]
    );
}

// -- §5.5 the bend depends on the outline and ν, not on the metal's stiffness or the size ---------

/// The first 50 ms of a 2 e strike in mode 1, read as cents above the tuned pitch.
fn bend(material: &Material, lx: f64) -> f64 {
    let g = geometry(Boundary::Supported, lx, 0.775 * lx, 16, 32_000.0);
    let tg = Targets {
        set_by: SetBy::Pitch(200.0),
        t60_low: 50.0,
        t60_high: 20.0,
        f_ref: 2000.0,
    };
    let tu = tune_vk_plate(&g, material, &tg).expect("reachable targets");
    let mut vk = built(&g, &tu);
    let shape = navier_shape(&vk.p.lin, tu.plate.fundamental.beta);
    let ones = vec![1.0; shape.len()];
    let e = vk.p.thickness;
    let a = projection(&mut vk, &shape, &ones, 2.0 * e, 1600);
    cents(zc_pitch(&a, 32_000.0), 200.0)
}

#[test]
fn a_strike_bends_every_metal_and_size_alike_except_through_nu() {
    // Scale w by e, x by L, t by 1/f0: the only material constant left is 12(1 − ν²) (plan §1).
    // So a 0.4 m steel gong and a 0.25 m one, and a brass gong given steel's ν, bend alike to
    // rounding; real brass (ν 0.331) does not.
    let Material::Isotropic { nu, .. } = metal("steel") else {
        unreachable!()
    };
    let Material::Isotropic { e, rho, .. } = metal("brass") else {
        unreachable!()
    };
    let brass_with_steels_nu = Material::Isotropic { e, nu, rho };
    let steel = bend(&metal("steel"), 0.4);
    let small = bend(&metal("steel"), 0.25);
    let alias = bend(&brass_with_steels_nu, 0.6);
    let brass = bend(&metal("brass"), 0.6);
    println!(
        "steel {steel:+.6} c, smaller {small:+.6}, brass at steel's ν {alias:+.6}, brass \
         {brass:+.6}"
    );
    assert!(
        steel > 100.0,
        "the strike did not bend the note ({steel:+.3} cents)"
    );
    assert!(
        (small - steel).abs() < 1e-6,
        "size moved the bend: {small} vs {steel}"
    );
    assert!(
        (alias - steel).abs() < 1e-6,
        "stiffness or density moved it: {alias} vs {steel}"
    );
    assert!(
        (brass - steel).abs() > 0.005 * steel,
        "ν did not move the bend: {brass} vs {steel}"
    );
}

// -- §5.7 the gong gets the material's volume density, and the thickness that sets its pitch ------

#[test]
fn the_gong_is_built_from_the_metal_itself() {
    // `apply` against a gong written out by hand from the preset's own E, ν, ρ and the mapped
    // thickness and losses: the same state after a strike, bit for bit — so `ρ` is the volume
    // density, `ν` the metal's, and each loss in its own slot. Then the thickness set by pitch,
    // fed back as the thickness, gives the pitch back.
    for b in [Boundary::Supported, Boundary::Free] {
        for name in METALS {
            let Material::Isotropic { e, nu, rho } = metal(name) else {
                unreachable!()
            };
            let g = geometry(b, 0.4, 0.31, 12, 32_000.0);
            let tg = Targets {
                set_by: SetBy::Pitch(150.0),
                t60_low: 2.0,
                t60_high: 0.5,
                f_ref: 1500.0,
            };
            let tu = tune_vk_plate(&g, &metal(name), &tg).expect("reachable targets");
            assert_eq!(
                tu.rho_v.to_bits(),
                rho.to_bits(),
                "{name}: ρ is not the metal's"
            );
            let by_hand = VkSpec {
                young: e,
                nu,
                rho,
                thickness: tu.plate.thickness,
                sigma: tu.plate.sigma0,
                sigma1: tu.plate.sigma1,
                ..g.clone()
            };
            let mut mine = built(&g, &tu);
            let mut theirs = VkPlate::new(VkParams::new(&by_hand).unwrap());
            // A smooth half-thickness bump: enough to load the membrane, far from §8.5's walls.
            let lin = &mine.p.lin;
            let full: Vec<f64> = lin
                .x
                .iter()
                .zip(&lin.y)
                .map(|(&x, &y)| {
                    let (dx, dy) = (x - 0.42 * lin.lx, y - 0.38 * lin.ly);
                    0.5 * tu.plate.thickness * (-(dx * dx + dy * dy) / (0.01 * lin.lx)).exp()
                })
                .collect();
            let shape = mine.p.to_live(&full);
            mine.set_state(&shape, &vec![0.0; shape.len()]).unwrap();
            theirs.set_state(&shape, &vec![0.0; shape.len()]).unwrap();
            for _ in 0..50 {
                mine.step(None).unwrap();
                theirs.step(None).unwrap();
            }
            assert_eq!(
                mine.energy().to_bits(),
                theirs.energy().to_bits(),
                "{b:?} {name}: the mapped gong is not the metal's"
            );
            let back = tune_vk_plate(
                &g,
                &metal(name),
                &Targets {
                    set_by: SetBy::Thickness(tu.plate.thickness),
                    ..tg
                },
            )
            .unwrap();
            assert!(
                cents(back.plate.f0, 150.0).abs() < 1e-9,
                "{b:?} {name}: set by its own thickness it plays {} Hz",
                back.plate.f0
            );
        }
    }
}

// -- §5.8 refusals --------------------------------------------------------------------------------

#[test]
fn what_no_gong_can_be_is_refused_by_name() {
    let g = geometry(Boundary::Supported, 0.4, 0.31, 12, 32_000.0);
    let ok = Targets {
        set_by: SetBy::Pitch(150.0),
        t60_low: 2.0,
        t60_high: 0.5,
        f_ref: 1500.0,
    };
    assert!(tune_vk_plate(&g, &metal("steel"), &ok).is_ok());
    assert_eq!(
        tune_vk_plate(&g, &metal("sitka spruce"), &ok),
        Err(TuningError::NotIsotropic)
    );
    let at_fundamental = Targets { f_ref: 150.0, ..ok };
    assert!(matches!(
        tune_vk_plate(&g, &metal("steel"), &at_fundamental),
        Err(TuningError::AnchorIsTheFundamental { .. })
    ));
    let too_long = Targets {
        t60_high: 5.0,
        ..ok
    };
    assert!(matches!(
        tune_vk_plate(&g, &metal("steel"), &too_long),
        Err(TuningError::HighRingTimeOutOfRange { .. })
    ));
    let too_high = Targets {
        set_by: SetBy::Pitch(20_000.0),
        ..ok
    };
    assert!(matches!(
        tune_vk_plate(&g, &metal("steel"), &too_high),
        Err(TuningError::OutOfReach { .. })
    ));
}
