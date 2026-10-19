//! A lossy string on a bridge keeps the books: total energy plus the scheme's own dissipation is
//! constant, on every bridge a string can sit on.
//!
//! New with the string-mapping plan's slice 4 (`docs/dev/string-mapping-plan.md` §16), which puts
//! a ring time on the coupled strings. Before it, the bridges were only ever asked to *decrease*
//! with a lossy string (`connection.rs`, `connection_body.rs`, `connection_plate.rs`, each at
//! `sigma = 5` or less), and that held while the books missed by 1.5e-4 of `E0` at `sigma = 5` and
//! 7e-4 at `sigma = 69`: the spring's pull on the end node was not divided by the string's
//! `1 + sigma k` (`connection::end_correction`). A monotone bar cannot see a wrong loss, only a
//! gain; a ledger can.
//!
//! The dissipation is the scheme's, not the continuum's: per step, `2 k sigma rho sum_l w_l
//! (delta_t. u_l)^2` on the string (trapezoidal node weights, so the free end's half cell counts
//! half) and `2 k sigma_i m_i (delta_t. q_i)^2` per body mode, `delta_t.` the centred difference
//! `(x^{n+1} - x^{n-1}) / 2k`. The radiated body books its own channel inside `energy()`.

use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::{
    BridgeBody, StringBodyBridge, StringPlateBridge, SympatheticStrings,
};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::plate::{self, Boundary as PB, Plate, PlateSpec};
use physsynth_core::radiation::RadiatedBody;
use physsynth_core::string_ideal::{Boundary, IdealString, Params as StringParams};

/// The project's acceptance bar (`CLAUDE.md`), deliberately not tightened.
const BOOKS_TOL: f64 = 1e-10;

/// Steps per run: about 0.18 s of a 100 Hz string, a dozen bridge sloshes. Short enough for
/// CI's unoptimised lane.
const STEPS: usize = 4000;

/// The viewer pages' body: guitar-ish modes, each about the string's own mass.
const BODY_FREQS: [f64; 4] = [110.0, 196.0, 261.0, 440.0];

/// `fs` of a `T = 200`, `rho = 0.005` string on `N = 100` at `lambda = 0.9`.
fn fs() -> f64 {
    200.0 * 100.0 / 0.9
}

fn string(sigma: f64, plucked: bool) -> IdealString {
    let p = StringParams::new(
        1.0,
        200.0,
        0.005,
        fs(),
        100,
        sigma,
        Some((Boundary::Fixed, Boundary::Free)),
    )
    .expect("valid string");
    let mut s = IdealString::new(p.clone());
    if plucked {
        let u0 = triangular_pluck(&p.grid(), p.l, 0.3 * p.l, 1e-3).expect("inside");
        s.set_state(&u0, &vec![0.0; u0.len()]);
    }
    s
}

fn modal(freqs: &[f64], sigma: f64) -> ModalBody {
    let m = freqs.len();
    ModalBody::new(
        body::Params::new(
            freqs.to_vec(),
            fs(),
            vec![sigma; m],
            vec![0.02; m],
            vec![1.0; m],
            None,
        )
        .expect("valid body"),
    )
}

/// The string's dissipation over the step that took `before` (its `u^{n-1}`) to `s.u` (`u^{n+1}`).
fn string_work(before: &[f64], s: &IdealString) -> f64 {
    let p = s.params();
    let w = p.node_weights();
    let mut acc = 0.0;
    for l in 0..w.len() {
        let v = (s.u[l] - before[l]) / (2.0 * p.k);
        acc += w[l] * v * v;
    }
    2.0 * p.k * p.sigma * p.rho * acc
}

/// The modal body's dissipation over the step that took `before` (`q^{n-1}`) to `b.q()`.
fn body_work(before: &[f64], b: &ModalBody) -> f64 {
    let p = b.params();
    let mut acc = 0.0;
    for (i, (q, q_before)) in b.q().iter().zip(before).enumerate() {
        let v = (q - q_before) / (2.0 * p.k);
        acc += 2.0 * p.k * p.sigma[i] * p.m[i] * v * v;
    }
    acc
}

/// `max |E^n + D^n - E^0| / E^0` over the run, and `D` at the end over `E^0`.
fn books_body<B: BridgeBody<Room = ()>>(br: &mut StringBodyBridge<B>) -> (f64, f64)
where
    B::Error: std::fmt::Debug,
{
    let e0 = br.energy();
    let (mut d, mut worst) = (0.0, 0.0f64);
    for _ in 0..STEPS {
        let u_before = br.string().u_prev.clone();
        let q_before = br.body().modal().q_prev().to_vec();
        br.step(&mut ()).unwrap();
        d += string_work(&u_before, br.string());
        d += body_work(&q_before, br.body().modal());
        worst = worst.max((br.energy() + d - e0).abs());
    }
    (worst / e0, d / e0)
}

#[test]
fn a_lossy_string_on_the_modal_body_keeps_the_books() {
    for (sigma, sigma_body) in [(5.0, 0.0), (69.0, 0.0), (69.0, 20.0)] {
        let mut br =
            StringBodyBridge::new(string(sigma, true), modal(&BODY_FREQS, sigma_body), 8000.0)
                .expect("inside the guard");
        let (miss, lost) = books_body(&mut br);
        assert!(
            miss < BOOKS_TOL,
            "sigma {sigma}, body {sigma_body}: books miss by {miss:.2e} of E0"
        );
        // The ledger has something in it: the string's loss did real work.
        assert!(
            lost > 0.01,
            "sigma {sigma}: only {lost:.2e} of E0 dissipated"
        );
    }
}

#[test]
fn a_lossy_string_on_the_radiating_body_keeps_the_books() {
    // `energy()` already carries the radiated channel; the string's and the body's losses are
    // added here. R = 133 is the radiating page's default.
    let loaded = RadiatedBody::new(modal(&BODY_FREQS, 0.0), 133.0).expect("R >= 0");
    let mut br = StringBodyBridge::new(string(35.0, true), loaded, 8000.0).expect("inside");
    let (miss, lost) = books_body(&mut br);
    assert!(miss < BOOKS_TOL, "books miss by {miss:.2e} of E0");
    assert!(
        lost > 0.01,
        "only {lost:.2e} of E0 dissipated by the string"
    );
}

#[test]
fn two_strings_with_different_losses_each_keep_their_own_books() {
    // Different losses on purpose: each string's pull must be divided by ITS OWN 1 + sigma k.
    let strings = vec![string(69.0, true), string(5.0, true)];
    let mut sy = SympatheticStrings::new(
        strings,
        modal(&[137.0, 213.0, 330.0, 471.0, 620.0], 20.0),
        vec![8000.0, 6000.0],
    )
    .expect("inside the guard");
    let e0 = sy.energy();
    let (mut d, mut worst) = (0.0, 0.0f64);
    for _ in 0..STEPS {
        let before: Vec<Vec<f64>> = sy.strings().iter().map(|s| s.u_prev.clone()).collect();
        let q_before = sy.body().q_prev().to_vec();
        sy.step();
        for (j, s) in sy.strings().iter().enumerate() {
            d += string_work(&before[j], s);
        }
        d += body_work(&q_before, sy.body());
        worst = worst.max((sy.energy() + d - e0).abs());
    }
    assert!(
        worst < BOOKS_TOL * e0,
        "books miss by {:.2e} of E0",
        worst / e0
    );
    assert!(d > 0.01 * e0, "only {:.2e} of E0 dissipated", d / e0);
}

#[test]
fn a_lossy_string_on_a_lossless_plate_keeps_the_books() {
    for b in [PB::Supported, PB::Free] {
        let plate = Plate::new(
            plate::Params::new(&PlateSpec {
                lx: 1.0,
                ly: 1.0,
                kappa: 20.0,
                rho: 0.005,
                fs: fs(),
                n: 12,
                sigma: 0.0,
                boundary: Some(b),
                nu: if b == PB::Free { Some(0.3) } else { None },
                ..PlateSpec::default()
            })
            .expect("valid plate"),
        );
        let mut br =
            StringPlateBridge::new(string(69.0, true), plate, 3000.0, None).expect("inside");
        let e0 = br.energy();
        let (mut d, mut worst) = (0.0, 0.0f64);
        for _ in 0..STEPS {
            let before = br.string().u_prev.clone();
            br.step(&mut ()).unwrap();
            d += string_work(&before, br.string());
            worst = worst.max((br.energy() + d - e0).abs());
        }
        assert!(
            worst < BOOKS_TOL * e0,
            "{b:?}: books miss by {:.2e} of E0",
            worst / e0
        );
        assert!(d > 0.01 * e0, "{b:?}: only {:.2e} of E0 dissipated", d / e0);
    }
}
