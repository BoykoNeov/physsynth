//! Musical controls for the bore — `src/wind_tuning.rs`, `docs/dev/wind-mapping-plan.md` §2.
//!
//! **Every pitch and ring time here is read from the simulation**, two ways: the eigenvalues of
//! the one-step map built column by column from `Bore::step`, and the rendered pressure, which at
//! `lambda = 1` must repeat itself every round trip (`2N` steps) inverted and scaled by `|rho|`.
//! The mapping's formulas are not imported: the expectations are written from the targets.

use physsynth_core::bore::{Bore, End, Params};
use physsynth_core::eig::general_eigenvalues;
use physsynth_core::wind_tuning::{bore_partial, bore_t60, tune_bore, BoreTuning, WindTuningError};
use std::f64::consts::PI;

/// Relative tolerance on a ring time read off the map or the render.
const T60_TOL: f64 = 1e-8;
/// Pitch tolerance, cents.
const CENTS_TOL: f64 = 1e-6;

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

/// A closed–radiating bore at `lambda = 1` on the mapping, with `rho0` and the radius off the
/// page's (`R/Z0` is dimensionless, so neither may matter).
fn bore(tu: &BoreTuning, n: usize, c0: f64, ratio: f64) -> Bore {
    let (rho0, radius) = (1.18, 0.0123);
    let z0 = rho0 * c0 / (PI * radius * radius);
    let p = Params::new(
        tu.l,
        c0 * n as f64 / tu.l,
        n,
        radius,
        Some((End::Closed, End::Radiating)),
        0.0,
        ratio * z0,
        rho0,
        c0,
    )
    .expect("the mapped bore constructs");
    assert!((p.lam - 1.0).abs() < 1e-12, "the mapping is for lambda = 1");
    Bore::new(p)
}

/// `(f, T60)` of every oscillating mode of the bore's one-step map, lowest first.
fn map_modes(proto: &Bore) -> Vec<(f64, f64)> {
    let p = proto.params();
    let (n, k) = (p.n, p.k);
    let dim = 2 * n + 1;
    // Columns of (Phi T), where T is what set_state makes of a basis vector: p = e_i gives
    // U = A e_i, u0 = e_j gives U = e_j. Then Phi = (Phi T) T^-1, T^-1 = [[I, 0], [-A, I]].
    let (mut phit, mut a) = (vec![0.0; dim * dim], vec![0.0; n * (n + 1)]);
    for c in 0..dim {
        let mut b = Bore::new(p.clone());
        let (mut p0, mut u0) = (vec![0.0; n + 1], vec![0.0; n]);
        if c <= n {
            p0[c] = 1.0;
        } else {
            u0[c - n - 1] = 1.0;
        }
        b.set_state(&p0, &u0);
        if c <= n {
            for j in 0..n {
                a[j * (n + 1) + c] = b.u()[j];
            }
        }
        b.step(None);
        for r in 0..=n {
            phit[r * dim + c] = b.p()[r];
        }
        for r in 0..n {
            phit[(n + 1 + r) * dim + c] = b.u()[r];
        }
    }
    let mut phi = phit.clone();
    for r in 0..dim {
        for i in 0..=n {
            let mut s = phit[r * dim + i];
            for j in 0..n {
                s -= a[j * (n + 1) + i] * phit[r * dim + n + 1 + j];
            }
            phi[r * dim + i] = s;
        }
    }
    let mut modes: Vec<(f64, f64)> = general_eigenvalues(&phi, dim)
        .expect("the one-step map has eigenvalues")
        .into_iter()
        .filter(|e| e.1 > 1e-12)
        .map(|(re, im)| {
            let r = (re * re + im * im).sqrt();
            (im.atan2(re) / (2.0 * PI * k), -3.0 * k / r.log10())
        })
        .collect();
    modes.sort_by(|x, y| x.0.total_cmp(&y.0));
    modes
}

/// Pitches from the page's lowest to its highest, ring times from a thud to the longest the bell
/// range holds at that pitch, grids not a power of two, and the speed of sound off the page's.
fn fixtures() -> Vec<(f64, f64, usize, f64)> {
    let mut v = Vec::new();
    for &c0 in &[343.0, 331.3] {
        for &(f0, t60) in &[(43.5, 60.0), (146.83, 2.0), (146.83, 0.05), (600.0, 9.0)] {
            for &n in &[37usize, 100] {
                v.push((f0, t60, n, c0));
            }
        }
    }
    v
}

#[test]
fn every_partial_of_the_map_sounds_and_rings_on_target() {
    for (f0, t60, n, c0) in fixtures() {
        let tu = tune_bore(f0, Some(t60), c0).expect("in range");
        let ratio = tu.r_ratio.expect("a ring time was asked for");
        let modes = map_modes(&bore(&tu, n, c0, ratio));
        for (i, &(f, t)) in modes.iter().take(5).enumerate() {
            let want = (2 * i + 1) as f64 * f0;
            assert!(
                cents(f, want).abs() < CENTS_TOL,
                "f0 {f0}, T60 {t60}, N {n}, c0 {c0}: partial {} at {f} Hz, want {want}",
                i + 1
            );
            assert!(
                (t / t60 - 1.0).abs() < T60_TOL,
                "f0 {f0}, T60 {t60}, N {n}, c0 {c0}: partial {} rings {t} s, want {t60}",
                i + 1
            );
        }
    }
}

/// The render: one round trip later every node reads `-|rho|` times what it read, `|rho|` being
/// what the target ring time allows in `2N` steps.
#[test]
fn the_render_repeats_each_round_trip_scaled_by_the_target_ring_time() {
    for (f0, t60, n, c0) in fixtures().into_iter().chain([(300.0, 1.0, 250, 343.0)]) {
        let tu = tune_bore(f0, Some(t60), c0).expect("in range");
        let mut b = bore(&tu, n, c0, tu.r_ratio.expect("a ring time"));
        let x = b.params().grid();
        let bump: Vec<f64> = x
            .iter()
            .map(|&xi| (-((xi - 0.3 * tu.l) / (0.07 * tu.l)).powi(2)).exp())
            .collect();
        b.set_state(&bump, &vec![0.0; n]);
        let trip = 2 * n;
        // Each sample is the mean of two steps: the bell's marginal grid-Nyquist ripple (an
        // eigenvalue at `z ~ -1`, `R << Z0`; ~1.5e-9 of the bump here) does not repeat inverted,
        // and the mean cancels it while every partial keeps its round trip.
        let mut prev = b.p().to_vec();
        let mut rec = Vec::with_capacity(3 * trip);
        for _ in 0..3 * trip {
            b.step(None);
            rec.push(
                b.p()
                    .iter()
                    .zip(&prev)
                    .map(|(a, c)| 0.5 * (a + c))
                    .collect::<Vec<_>>(),
            );
            prev = b.p().to_vec();
        }
        let k = b.params().k;
        let want = 10f64.powf(-3.0 * trip as f64 * k / t60);
        let (mut num, mut den, mut worst, mut peak) = (0.0, 0.0, 0.0f64, 0.0f64);
        for i in trip..3 * trip {
            for (a, b) in rec[i].iter().zip(&rec[i - trip]) {
                num -= a * b;
                den += b * b;
                worst = worst.max((a + want * b).abs());
                peak = peak.max(b.abs());
            }
        }
        let read = num / den;
        let t60_read = -3.0 * trip as f64 * k / read.log10();
        assert!(
            (t60_read / t60 - 1.0).abs() < T60_TOL,
            "f0 {f0}, T60 {t60}, N {n}: the render rings {t60_read} s"
        );
        // Pitch: nothing but the odd series of fs / 4N = f0 survives this.
        assert!(
            worst < 1e-10 * peak,
            "f0 {f0}, T60 {t60}, N {n}: the round trip misses by {worst:e} of {peak:e}"
        );
        assert!((0.25 / (n as f64 * k) / f0 - 1.0).abs() < 1e-12);
    }
}

/// The other root, `R/Z0 -> Z0/R`, rings for the same time and is a closed end: no partial of it
/// is anywhere near the pitch asked for. The control: the mapped root has one there.
#[test]
fn the_other_bell_rings_as_long_but_plays_another_series() {
    let (f0, t60, n, c0) = (146.83, 0.3, 40, 343.0);
    let tu = tune_bore(f0, Some(t60), c0).expect("in range");
    let ratio = tu.r_ratio.expect("a ring time");
    assert!(ratio < 1.0, "the bell is the open-like root");
    assert!((bore_t60(tu.l, 1.0 / ratio, c0) / t60 - 1.0).abs() < 1e-12);
    let near = |modes: &[(f64, f64)]| {
        modes
            .iter()
            .map(|&(f, _)| cents(f, f0).abs())
            .fold(f64::INFINITY, f64::min)
    };
    let mapped = map_modes(&bore(&tu, n, c0, ratio));
    let other = map_modes(&bore(&tu, n, c0, 1.0 / ratio));
    assert!(near(&mapped) < CENTS_TOL);
    assert!(
        near(&other) > 600.0,
        "the other bell plays {:?}",
        &other[..3]
    );
}

#[test]
fn the_ring_time_read_out_inverts_the_mapping() {
    for (f0, t60, _, c0) in fixtures() {
        let tu = tune_bore(f0, Some(t60), c0).expect("in range");
        let ratio = tu.r_ratio.expect("a ring time");
        assert!((bore_t60(tu.l, ratio, c0) / t60 - 1.0).abs() < 1e-12);
        assert!((bore_partial(3, tu.l, c0) / (5.0 * f0) - 1.0).abs() < 1e-15);
    }
    assert_eq!(bore_t60(0.5, 0.0, 343.0), f64::INFINITY);
    assert_eq!(bore_t60(0.5, 1.0, 343.0), 0.0);
}

#[test]
fn the_open_end_maps_the_pitch_alone() {
    let tu = tune_bore(220.0, None, 343.0).expect("in range");
    assert_eq!(tu.r_ratio, None);
    assert!((tu.l - 343.0 / 880.0).abs() < 1e-15);
}

#[test]
fn a_non_positive_input_is_refused_by_name() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            tune_bore(bad, Some(1.0), 343.0),
            Err(WindTuningError::NonPositive("f0"))
        );
        assert_eq!(
            tune_bore(200.0, Some(1.0), bad),
            Err(WindTuningError::NonPositive("c0"))
        );
        assert_eq!(
            tune_bore(200.0, Some(bad), 343.0),
            Err(WindTuningError::NonPositive("t60"))
        );
    }
}
