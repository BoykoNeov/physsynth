//! Native bars for the bowed string (plan §20) — the exciter's acceptance contract, no interpreter.
//!
//! The Python suite's criteria are reproduced in the order they are argued there: the lossless
//! **energy balance** (the money test — the bow is active, so `E` is not conserved and the claim is
//! that every joule is accounted for by the bow's work), **passivity** under loss, `force = 0`
//! decoupling the bow **bit-for-bit** from the bare string, the friction curve's shape, and the
//! `helmholtz_number` diagnostic marking the single-valued / multivalued boundary — including that
//! the bracketed fallback fires above it and never below.
//!
//! **One test here has no Python counterpart, and it is why this file matters beyond re-running the
//! suite.** `residual` and `scan_residual` are two spellings of one expression, kept apart on
//! purpose (§20.2); a maintainer "tidying" them into one call would move the scan's values by a
//! last bit, change which brackets exist, and change a branch at a slip event — while every bar
//! above stayed green. `the_two_residual_spellings_are_not_the_same_double` is the pin against
//! that. Unlike the mallet's `** 2` pin it needs no `#[inline(never)]`: LLVM may not reassociate
//! floating-point multiplication without fast-math, which Rust does not enable, so the distinction
//! survives `--release` on its own. It is still run in both profiles, because that is what the
//! mallet's batch established and the cost is zero.

use physsynth_core::bow::{
    friction_smooth, friction_smooth_deriv, residual, scan_residual, BowedString, ParamError,
    Params,
};
use physsynth_core::string_damped::{self, DampedStiffString};

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s
const BALANCE_TOL: f64 = 1e-11; // the Python suite's bar; observed ~1e-15

/// The project's canonical bowed-string rig: `lam = 0.9`, flexible (`kappa = 0`), 100 segments.
fn string(n: i64, sigma0: f64, sigma1: f64) -> DampedStiffString {
    let c = (T / RHO).sqrt();
    let fs = c * (n as f64) / (L * 0.9);
    let p = string_damped::Params::new(L, T, RHO, fs, n, 0.0, sigma0, sigma1, 0.28, true)
        .expect("valid string");
    DampedStiffString::new(p)
}

fn bow(sigma0: f64, sigma1: f64, force: f64, sharpness: f64) -> BowedString {
    BowedString::new(
        string(100, sigma0, sigma1),
        0.2,
        0.1,
        force,
        sharpness,
        1e-13,
        60,
    )
    .expect("valid bow")
}

// -- criterion 1 (money test): the lossless energy balance ------------------------------------

#[test]
fn a_lossless_bowed_string_balances_energy_against_the_bow_work() {
    for (force, sharpness) in [(1.0, 60.0), (2.0, 100.0), (0.5, 40.0)] {
        let mut b = bow(0.0, 0.0, force, sharpness);
        let e0 = b.energy();
        let mut worst: f64 = 0.0;
        for _ in 0..4000 {
            b.step().expect("the friction root always exists");
            let e = b.energy();
            let w = b.s.bow_work;
            let scale = e.abs() + w.abs() + 1e-30;
            worst = worst.max(((e - e0) - w).abs() / scale);
        }
        assert!(
            worst < BALANCE_TOL,
            "energy-balance error {worst:.2e} (force={force}, a={sharpness})"
        );
    }
}

#[test]
fn the_bow_does_net_work_on_a_string_at_rest() {
    let mut b = bow(0.0, 0.0, 1.0, 100.0);
    for _ in 0..2000 {
        b.step().expect("root");
    }
    assert!(b.s.bow_work > 0.0, "bow did no net work");
    assert!(b.energy() > 1e-8, "string never gained energy from the bow");
}

// -- criterion 2: passivity survives the coupling ----------------------------------------------

#[test]
fn loss_only_ever_removes_energy() {
    for (sigma0, sigma1) in [(0.5, 0.05), (2.0, 0.0), (0.0, 0.1)] {
        let mut b = bow(sigma0, sigma1, 1.0, 100.0);
        let e0 = b.energy();
        let mut prev_dissipation = 0.0;
        let mut worst_backwards: f64 = 0.0;
        for _ in 0..6000 {
            b.step().expect("root");
            let dissipation = b.s.bow_work - (b.energy() - e0);
            worst_backwards = worst_backwards.min(dissipation - prev_dissipation);
            prev_dissipation = dissipation;
        }
        assert!(
            prev_dissipation >= -BALANCE_TOL,
            "loss added energy overall"
        );
        assert!(
            worst_backwards >= -1e-9 * (b.s.bow_work.abs() + 1.0),
            "a loss step added energy ({worst_backwards:.2e}) at sigma0={sigma0}"
        );
    }
}

// -- criterion 3: force = 0 decouples the bow bit-for-bit --------------------------------------

#[test]
fn a_force_free_bow_is_bit_identical_to_the_bare_string() {
    // The rank-1 correction is `u += (force_pref * 0) * a_full` — an add of a signed zero, never
    // short-circuited on either side. If one implementation ever skips the loop, this stops being
    // a comparison of two strings and starts being a comparison of one with itself.
    let mut bare = string(100, 0.5, 0.05);
    let mut bowed = BowedString::new(string(100, 0.5, 0.05), 0.2, 0.1, 0.0, 100.0, 1e-13, 60)
        .expect("valid bow");

    let x = bare.p.grid();
    let u0: Vec<f64> = x
        .iter()
        .map(|&xi| 1e-3 * (3.0 * std::f64::consts::PI * xi / L).sin())
        .collect();
    let v0 = vec![0.0; u0.len()];
    bare.set_state(&u0, &v0);
    bowed.string.set_state(&u0, &v0);

    for _ in 0..500 {
        bare.step();
        bowed.step().expect("root");
    }
    assert_eq!(bowed.s.bow_force, 0.0);
    assert_eq!(bowed.s.bow_work, 0.0);
    assert_eq!(bare.u, bowed.string.u, "a force-free bow moved the string");
}

// -- the friction characteristic ---------------------------------------------------------------

#[test]
fn the_friction_curve_has_the_documented_shape() {
    let (force, a) = (1.5, 80.0);
    assert_eq!(friction_smooth(0.0, force, a), 0.0);
    let v_peak = 1.0 / (2.0 * a).sqrt();
    assert!((friction_smooth(v_peak, force, a) - force).abs() < 1e-12);
    assert!((friction_smooth(-0.03, force, a) + friction_smooth(0.03, force, a)).abs() < 1e-18);

    let v = 0.05;
    let fd = (friction_smooth(v + 1e-7, force, a) - friction_smooth(v - 1e-7, force, a)) / 2e-7;
    let exact = friction_smooth_deriv(v, force, a);
    assert!((exact - fd).abs() / exact.abs() < 1e-5, "{exact} vs {fd}");
}

/// The pin. See the module header — this is the one assertion a "harmless" cleanup would break.
#[test]
fn the_two_residual_spellings_are_not_the_same_double() {
    let b = bow(0.5, 0.05, 4.0, 120.0);
    let mut differ = 0;
    // A deterministic sweep across the band the solve actually visits, rather than a witness value:
    // *how often* they differ is the platform's business, *that* they can differ is the port's.
    for i in 0..20_000 {
        let v = -0.25 + 5.0e-5 * (i as f64);
        let v_free = 0.113 - 3.0e-5 * (i as f64);
        if residual(v, v_free, &b.p) != scan_residual(v, v_free, &b.p) {
            differ += 1;
        }
    }
    assert!(
        differ > 0,
        "`residual` and `scan_residual` agreed everywhere -- the hoist has been optimised away or \
         edited together, and the bracket scan no longer reproduces NumPy's"
    );
}

// -- the helmholtz_number diagnostic, and the fallback it predicts ------------------------------

#[test]
fn the_helmholtz_number_is_the_documented_product() {
    let b = bow(0.5, 0.05, 1.0, 60.0);
    let expected = b.p.g * b.p.force * (2.0 * b.p.sharpness).sqrt() * 0.5_f64.exp();
    assert_eq!(b.p.helmholtz_number, expected);
}

#[test]
fn the_multivalued_regime_needs_the_bracketed_fallback_and_the_single_valued_one_does_not() {
    let mut strong = bow(0.5, 0.05, 4.0, 120.0);
    assert!(strong.p.helmholtz_number > 1.0);
    for _ in 0..4000 {
        strong.step().expect("the root search must never fail");
    }
    assert!(
        strong.s.fallbacks > 0,
        "expected slip-event fallbacks in the multivalued regime"
    );

    let mut weak = bow(0.5, 0.05, 0.02, 60.0);
    assert!(weak.p.helmholtz_number < 1.0);
    for _ in 0..3000 {
        weak.step().expect("root");
    }
    assert_eq!(
        weak.s.fallbacks, 0,
        "single-valued regime should need no root fallback"
    );
}

// -- no blow-up across the playable parameter space ---------------------------------------------

#[test]
fn nothing_blows_up_across_the_playable_space() {
    for force in [0.2, 1.0, 3.0, 8.0] {
        for v_bow in [0.02, 0.1, 0.4] {
            let mut b =
                BowedString::new(string(100, 0.5, 0.05), 0.2, v_bow, force, 100.0, 1e-13, 60)
                    .expect("valid bow");
            for _ in 0..3000 {
                b.step().expect("root");
            }
            assert!(
                b.string.u.iter().all(|x| x.is_finite()) && b.energy().is_finite(),
                "non-finite state (force={force}, v_bow={v_bow})"
            );
        }
    }
}

// -- construction --------------------------------------------------------------------------------

#[test]
fn construction_rejects_the_unphysical() {
    let s = string(100, 0.5, 0.05);
    let mk = |pos: f64, force: f64, sharp: f64, maxiter: i64| {
        Params::new(pos, 0.1, force, sharp, 1e-13, maxiter, s.p.l, s.p.h, s.p.n)
    };
    assert_eq!(mk(0.13, -1.0, 100.0, 60), Err(ParamError::NegativeForce));
    assert_eq!(
        mk(0.13, 1.0, 0.0, 60),
        Err(ParamError::NonPositiveSharpness)
    );
    assert_eq!(mk(0.13, 1.0, 100.0, 0), Err(ParamError::BadMaxIter));
    for pos in [-0.1, 0.0, 1.0, 1.5] {
        assert!(matches!(
            mk(pos, 1.0, 100.0, 60),
            Err(ParamError::BowPosition(_))
        ));
    }
    // The message quotes both numbers as Python's `str(float)` would.
    let msg = mk(1.5, 1.0, 100.0, 60).unwrap_err().to_string();
    assert_eq!(msg, "bow_position must satisfy 0 < x < L (L=1.0), got 1.5.");
}

#[test]
fn the_bow_node_is_snapped_into_the_interior() {
    // A bow requested very close to the nut still lands on an interior node.
    let b = BowedString::new(string(100, 0.5, 0.05), 0.004, 0.1, 1.0, 100.0, 1e-13, 60)
        .expect("valid bow");
    assert!(b.p.node >= 1 && b.p.node < b.string.p.n);
    assert!(b.x_bow > 0.0 && b.x_bow < b.string.p.l);
    // The admittance vanishes at both clamped ends, which is what makes the correction a no-op
    // there and the `force = 0` anchor above meaningful.
    assert_eq!(b.a_full[0], 0.0);
    assert_eq!(*b.a_full.last().unwrap(), 0.0);
}
