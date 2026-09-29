//! The membrane's harness bars — model #4, at the Python suite's own parameters.
//!
//! Carried from `tests/test_membrane_{energy,modal,stability}.py` (retirement plan §29); the fourth
//! file, `test_membrane_dispersion.py`, touched no model and went to
//! `crates/physsynth-analysis/tests/modal.rs`. `membrane.rs` next door is the port's own floor at a
//! drumhead's `T` and `rho`; this file is the acceptance contract at `make_membrane`'s — `T = 200`,
//! `rho = 0.005` (so `c = 200 m/s`), a unit square, a disk of radius 0.5 and `fs = c / (lambda h)`.
//!
//! **One outside referee retires here, with its numbers.** Everything else the Python tests
//! touched — the exciter, `simulate`, the spectrum detector, every `modal.*` oracle — already ran
//! through the Rust binding, so it was Rust checking Rust. What was not: SciPy's eigensolver on the
//! masked Laplacian, and SciPy's Bessel zeros. Both were frozen on 2026-09-29 before the Python was
//! deleted (NumPy 2.4.6, SciPy 1.17.1, wheel reinstalled) into `tests/reference/membrane.json` —
//! LAPACK's dense `eigh` rather than ARPACK (§24.3's rule; here the two agree to 5e-16 of the
//! spectral bound, so the rule costs nothing), with a mask digest per grid so a fixture that built
//! a different disk fails loudly instead of comparing two different drums.
//!
//! **Energy is perpendicular to geometry**, as `membrane.rs` says: the staircased disk conserves
//! exactly as well as the square, so no bar in the first block can see a wrong rim. The rim is
//! seen only by the Bessel block, and only at the staircase's O(h).

use physsynth_analysis::modal::{
    cents, circular_membrane_freqs, discrete_membrane_eigenfrequency,
    rectangular_discrete_eigenvalues, rectangular_membrane_freqs,
};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::membrane::{Domain, Membrane, ParamError, Params};
use serde_json::Value;
use std::f64::consts::PI;

/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;
const T: f64 = 200.0;
const RHO: f64 = 0.005;
const RADIUS: f64 = 0.5;
/// The Python sweeps' top Courant number: four digits of 1/sqrt 2, so a hair BELOW the ceiling
/// on purpose — the ceiling itself is `the_ceiling_itself_is_accepted_and_reported`'s case.
#[allow(clippy::approx_constant)]
const NEAR_CEILING: f64 = 0.7071;

/// The 2-D ceiling, written HERE rather than read from `membrane::lambda_max`: the two CFL bars
/// below are about that constant, and reading it from the model would move the bar with the
/// defect (a planted `1.1 / sqrt 2` would pass both that way — `1.05 x` the moved ceiling is still
/// past it, and the moved ceiling is still accepted; §29.3).
fn lambda_max() -> f64 {
    1.0 / 2.0f64.sqrt()
}

/// `wave_speed()` — 200 m/s.
fn c() -> f64 {
    (T / RHO).sqrt()
}

fn reference() -> Value {
    serde_json::from_str(include_str!("reference/membrane.json")).expect("the frozen record")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

/// `make_membrane`: the sample rate is solved for so the Courant number is exactly `lam`.
fn make(domain: Domain, n: i64, lam: f64, sigma: f64, t: f64, rho: f64) -> Membrane {
    let c = (t / rho).sqrt();
    let h = match domain {
        Domain::Rectangle => 1.0 / n as f64,
        Domain::Circle => 2.0 * RADIUS / n as f64,
    };
    let fs = c / (lam * h);
    let params = match domain {
        Domain::Rectangle => Params::new(
            Some(domain),
            t,
            rho,
            fs,
            n,
            Some(1.0),
            Some(1.0),
            None,
            sigma,
        ),
        Domain::Circle => Params::new(Some(domain), t, rho, fs, n, None, None, Some(RADIUS), sigma),
    };
    Membrane::new(params.expect("valid membrane"))
}

fn membrane(domain: Domain, n: i64, lam: f64, sigma: f64) -> Membrane {
    make(domain, n, lam, sigma, T, RHO)
}

/// A raised-cosine bump on the live nodes — `raised_cosine_2d`, then `field[~mask] = 0`.
fn bump(m: &Membrane, center: (f64, f64), width: f64) -> Vec<f64> {
    let p = m.params();
    let full = raised_cosine_2d(&p.x, &p.y, center, width, 1e-3).expect("positive width");
    p.to_live(&full)
}

/// `_pluck`: a smooth off-centre bump, placed per domain.
fn pluck(m: &Membrane) -> Vec<f64> {
    let p = m.params();
    match p.domain {
        Domain::Rectangle => {
            let (lx, ly) = (p.lx.expect("Lx"), p.ly.expect("Ly"));
            bump(m, (0.4 * lx, 0.55 * ly), 0.25 * lx.min(ly))
        }
        Domain::Circle => bump(m, (0.2 * RADIUS, -0.15 * RADIUS), 0.55 * RADIUS),
    }
}

/// `_run`: pluck, then `simulate` for `int(secs * fs)` steps.
fn run(m: &mut Membrane, secs: f64) -> SimResult {
    let u0 = pluck(m);
    m.set_displacement(&u0);
    let steps = (secs * m.params().fs) as usize;
    simulate(m, steps, None, 0).expect("a membrane step cannot fail")
}

/// The `count` lowest eigenvalues of `-L`, ascending — `membrane_low_eigenfrequencies`' solve.
/// `-L` is positive definite with no nullspace, so the Python's shift of 0 carries unchanged.
fn lowest_lambda(m: &Membrane, count: usize) -> Vec<f64> {
    eigsh_shift_invert(&m.params().l.scaled(-1.0), None, 0.0, count)
        .expect("-L factors at the origin")
        .values
}

/// `(count, Σ (i+1), Σ (i+1)²)` over the live flat indices, row-major — the recorded digest.
fn digest(m: &Membrane) -> [u64; 3] {
    let mut d = [0u64; 3];
    for (i, _) in m
        .params()
        .mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
    {
        let k = i as u64 + 1;
        d[0] += 1;
        d[1] += k;
        d[2] += k * k;
    }
    d
}

/// Certify the native eigenvalues of `m` against LAPACK's, recorded under `key`.
///
/// The bar is in units of `eps · 8/h²` — `8/h²` bounds the 5-point Laplacian's spectrum from above
/// (Gershgorin) and sits within a hair of `lambda_max` — rather than a bare relative figure, and is
/// §25's 20 of them. Measured worst over all 30 recorded eigenvalues: 2.1 (the disk at N = 64,
/// seventh value), so the bar has ~10x headroom (retirement plan §29.2).
fn certify_against_lapack(m: &Membrane, record: &Value, count: usize) -> Vec<f64> {
    let p = m.params();
    let want_digest: Vec<u64> = record["mask_digest"]
        .as_array()
        .expect("a digest")
        .iter()
        .map(|v| v.as_u64().expect("an integer"))
        .collect();
    assert_eq!(
        digest(m).to_vec(),
        want_digest,
        "not the grid the record was taken on"
    );
    assert_eq!(
        p.n_live() as u64,
        record["n_live"].as_u64().expect("n_live")
    );
    assert_eq!(p.h, record["h"].as_f64().expect("h"));

    let lapack = floats(&record["lapack_lowest"]);
    let native = lowest_lambda(m, count);
    let unit = f64::EPSILON * 8.0 / (p.h * p.h);
    for (i, (&got, &want)) in native.iter().zip(lapack.iter()).enumerate() {
        let gap = (got - want).abs() / unit;
        assert!(
            gap < 20.0,
            "eigenvalue {i}: native {got} vs LAPACK {want}, {gap:.1} eps·8/h²"
        );
    }
    native
}

// -- energy: conservation and passivity ------------------------------------------------------

#[test]
fn lossless_energy_is_flat_and_positive_on_both_domains_at_every_courant_number() {
    // `test_energy_conserved` (both domains x three lambdas), and in the same runs
    // `test_circle_conserves_like_rectangle` (its two runs are two of these six) and
    // `test_energy_strictly_positive_when_lossless` (its circle at 0.6 over 0.5 s is a prefix of
    // this one's 1 s).
    for domain in [Domain::Rectangle, Domain::Circle] {
        for lam in [NEAR_CEILING, 0.6, 0.4] {
            let mut m = membrane(domain, 48, lam, 0.0);
            let res = run(&mut m, 1.0);
            let drift = res.energy_drift();
            assert!(
                drift < DRIFT_TOL,
                "{domain:?} drift {drift:e} at lam = {lam}"
            );
            assert!(
                res.energy.iter().all(|&e| e > 0.0),
                "{domain:?} energy not strictly positive at lam = {lam}"
            );
        }
    }
}

#[test]
fn loss_makes_the_energy_fall_at_every_step() {
    // `test_passivity_monotonic_decrease`, at its parameters and its slack of 1e-12 E^0.
    let mut m = membrane(Domain::Circle, 48, 0.6, 8.0);
    let res = run(&mut m, 1.0);
    let e0 = res.energy[0];
    let worst = res
        .energy
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(worst <= 1e-12 * e0, "max positive step {worst:e}");
}

#[test]
fn a_uniformly_damped_membrane_loses_energy_at_two_sigma() {
    // E(t) ~ E0 exp(-2 sigma t). The log-ratio's relative error is the figure, bar 2%.
    let (sigma, secs) = (6.0, 0.6);
    let mut m = membrane(Domain::Rectangle, 48, 0.6, sigma);
    let res = run(&mut m, secs);
    let measured = res.energy[res.energy.len() - 1] / res.energy[0];
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    assert!(
        rel < 0.02,
        "decay rate off by {:.3}% (got {measured:e}, want {expected:e})",
        100.0 * rel
    );
}

#[test]
fn energy_is_in_joules_and_linear_in_the_areal_density() {
    // Double rho and T together: c — hence the grid, the timestep and the field — is unchanged, so
    // the ratio isolates the rho prefactor. The only bar that can see that prefactor: every other
    // energy check is a ratio of the energy to itself.
    let m1 = make(Domain::Circle, 40, 0.6, 0.0, 200.0, 0.005);
    let m2 = make(Domain::Circle, 40, 0.6, 0.0, 400.0, 0.010);
    let mut e = Vec::new();
    for mut m in [m1, m2] {
        let u0 = pluck(&m);
        m.set_displacement(&u0);
        e.push(m.energy());
    }
    let ratio = e[1] / e[0];
    assert!(
        (ratio - 2.0).abs() <= 1e-12 * 2.0,
        "E(2 rho) / E(rho) = {ratio}"
    );
}

// -- modal: the rectangle's closed form, the disk's Bessel series ----------------------------

#[test]
fn the_square_eigenvalues_are_the_closed_form_and_lapacks() {
    // `test_rectangle_eigenvalues_match_closed_form`. The six lowest on the unit square include two
    // exact degenerate pairs, (2,1)/(1,2) and (3,1)/(1,3) — the case a single-start Krylov solve
    // could collapse to one copy, which the sorted comparison would catch.
    let n = 24;
    let m = membrane(Domain::Rectangle, n, 0.6, 0.0);
    let p = m.params();
    let ny = (p.ly.expect("Ly") / p.h).round() as i64;
    let modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)];
    let mut oracle = rectangular_discrete_eigenvalues(p.h, n, ny, &modes);
    oracle.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

    let native = certify_against_lapack(&m, &reference()["square_24"], modes.len());
    let rel = native
        .iter()
        .zip(oracle.iter())
        .map(|(g, w)| (g - w).abs() / w)
        .fold(0.0, f64::max);
    assert!(
        rel < 1e-10,
        "discrete eigenvalue mismatch {rel:e} (operator is mis-assembled)"
    );
}

#[test]
fn the_square_continuum_error_converges_at_second_order() {
    // `test_rectangle_continuum_convergence_order`: the scheme's exact discrete frequency against
    // the continuum's, over four grids. Second order: each halving quarters the error.
    let modes = [(1, 1), (2, 1)];
    let f_cont = rectangular_membrane_freqs(c(), 1.0, 1.0, &modes);
    let (mut hs, mut errs) = (Vec::new(), Vec::new());
    for n in [16, 32, 64, 128] {
        let m = membrane(Domain::Rectangle, n, 0.6, 0.0);
        let p = m.params();
        let ny = (p.ly.expect("Ly") / p.h).round() as i64;
        let lam = rectangular_discrete_eigenvalues(p.h, n, ny, &modes);
        let err = lam
            .iter()
            .zip(f_cont.iter())
            .map(|(&l, &fc)| (discrete_membrane_eigenfrequency(l, c(), p.k) - fc).abs())
            .fold(0.0, f64::max);
        hs.push(p.h);
        errs.push(err);
    }
    for i in 1..errs.len() {
        assert!(errs[i] < errs[i - 1], "errors not decreasing: {errs:?}");
    }
    let last = errs.len() - 1;
    let order = (errs[last - 1] / errs[last]).ln() / (hs[last - 1] / hs[last]).ln();
    assert!(
        order > 1.8,
        "continuum convergence order {order:.2} < 1.8 (expected ~2)"
    );
}

#[test]
fn the_disk_fundamental_converges_to_bessel_at_the_staircase_rate() {
    // `test_circle_bessel_convergence_rate`. Monotone, first-order-ish — neither stalled (p > 0.5)
    // nor the clean O(h²) of a fitted boundary (p < 1.5) — plus a loose absolute bound at the
    // finest grid, NOT the 1-D ~1-cent bar: the staircase caps the disk at O(h).
    let f01 = circular_membrane_freqs(c(), RADIUS, 1, 12, 12)[0].freq;
    let record = reference();
    let (mut hs, mut errs, mut cents_fine) = (Vec::new(), Vec::new(), 0.0);
    for n in [32i64, 64, 128] {
        let m = membrane(Domain::Circle, n, 0.6, 0.0);
        let lam = certify_against_lapack(&m, &record["disk"][n.to_string()], 8);
        let p = m.params();
        let f = discrete_membrane_eigenfrequency(lam[0], p.c, p.k);
        hs.push(p.h);
        errs.push((f - f01).abs());
        cents_fine = cents(f, f01).abs();
    }
    for i in 1..errs.len() {
        assert!(
            errs[i] < errs[i - 1],
            "Bessel error not decreasing: {errs:?}"
        );
    }
    for i in 1..errs.len() {
        let order = (errs[i - 1] / errs[i]).ln() / (hs[i - 1] / hs[i]).ln();
        assert!(
            order > 0.5 && order < 1.5,
            "staircase order {order:.3} out of band (0.5, 1.5); errors {errs:?}"
        );
    }
    assert!(
        cents_fine < 12.0,
        "fundamental off by {cents_fine:.2} cents at N = 128 (bound 12)"
    );
}

#[test]
fn the_disk_low_spectrum_tracks_the_sorted_bessel_series() {
    // `test_circle_low_spectrum_tracks_bessel`. The oracle is expanded by degeneracy (a cos/sin
    // pair is two modes) and sorted; SciPy's zeros certify it first. The zero is recovered from
    // the frequency, so the bar carries that round trip too: measured worst 1.9 eps, bar 8.
    let record = reference();
    let bessel = record["bessel_lowest"]
        .as_array()
        .expect("the recorded zeros");
    let modes = circular_membrane_freqs(c(), RADIUS, 8, 12, 12);
    for (got, want) in modes.iter().zip(bessel.iter()) {
        let z = got.freq * 2.0 * PI * RADIUS / c();
        let zero = want["zero"].as_f64().expect("a zero");
        assert!(
            (z - zero).abs() <= 8.0 * f64::EPSILON * zero,
            "j = {z} vs SciPy {zero}"
        );
        assert_eq!(u64::from(got.m), want["m"].as_u64().expect("m"));
        assert_eq!(got.n as u64, want["n"].as_u64().expect("n"));
        assert_eq!(
            u64::from(got.degeneracy),
            want["degeneracy"].as_u64().expect("degeneracy")
        );
    }
    let mut oracle: Vec<f64> = modes
        .iter()
        .flat_map(|md| std::iter::repeat_n(md.freq, md.degeneracy as usize))
        .collect();
    oracle.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    oracle.truncate(8);

    let m = membrane(Domain::Circle, 128, 0.6, 0.0);
    let p = m.params();
    let lam = certify_against_lapack(&m, &record["disk"]["128"], 8);
    // Every measured mode within the loose staircase band of its Bessel partner (in cents).
    // The 20-cent bound is NOT a pitch horizon and must not be rewritten as one: the dominant
    // error here is the staircased circular boundary, so this compares against the frequency of a
    // slightly DIFFERENT shape and a cents reading mixes two errors. The circle needs a
    // geometry-convergence study instead (docs/dev/resolution-horizon-plan.md section 5).
    let worst = lam
        .iter()
        .zip(oracle.iter())
        .map(|(&l, &fo)| cents(discrete_membrane_eigenfrequency(l, p.c, p.k), fo).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 20.0,
        "worst low-mode error {worst:.2} cents (bound 20)"
    );
}

#[test]
fn a_struck_disk_rings_at_its_discrete_fundamental() {
    // `test_circle_fft_peak_at_fundamental`: the end-to-end check that the time-stepper, not just
    // the operator, carries the eigenvalue. Off-centre bump and off-axis pickup, so m >= 1 rings
    // too and the detector has neighbours to be confused by.
    let mut m = membrane(Domain::Circle, 64, 0.6, 0.0);
    let (f_disc, fs, pickup) = {
        let p = m.params();
        let lam = lowest_lambda(&m, 1)[0];
        (
            discrete_membrane_eigenfrequency(lam, p.c, p.k),
            p.fs,
            p.pickup_index_at(0.3 * RADIUS, 0.2 * RADIUS),
        )
    };
    let u0 = bump(&m, (0.25 * RADIUS, -0.1 * RADIUS), 0.5 * RADIUS);
    m.set_displacement(&u0);
    let res = simulate(&mut m, (0.5 * fs) as usize, Some(pickup), 0).expect("steps");
    let output = res.output.expect("a pickup was requested");
    let found = measure_partials_near(&output, res.fs, &[f_disc], Some(20.0))[0];
    let off = cents(found, f_disc).abs();
    assert!(
        off < 5.0,
        "FFT fundamental off by {off:.2} cents (found {found:.2}, want {f_disc:.2})"
    );
}

// -- stability and construction --------------------------------------------------------------

#[test]
fn nothing_blows_up_anywhere_in_the_admissible_courant_range() {
    // `test_no_nan_across_valid_lambda`: both domains, lambda from the ceiling down to 0.1, a
    // bump at the origin (a corner of the square, the centre of the disk), pickup at node 0.
    for domain in [Domain::Rectangle, Domain::Circle] {
        for lam in [NEAR_CEILING, 0.65, 0.5, 0.3, 0.1] {
            let mut m = membrane(domain, 40, lam, 0.0);
            let width = 0.3 * m.params().radius.unwrap_or(1.0);
            let u0 = bump(&m, (0.0, 0.0), width);
            m.set_displacement(&u0);
            let steps = (0.3 * m.params().fs) as usize;
            let res = simulate(&mut m, steps, Some(0), 0).expect("steps");
            let output = res.output.expect("a pickup was requested");
            assert!(
                output.iter().all(|v| v.is_finite()),
                "{domain:?} output non-finite at lam = {lam}"
            );
            assert!(
                res.energy.iter().all(|v| v.is_finite()),
                "{domain:?} energy non-finite at lam = {lam}"
            );
        }
    }
}

#[test]
fn a_courant_number_just_past_the_ceiling_is_refused_at_construction() {
    // `test_lambda_above_cfl_rejected_at_construction`: 5% past 1/sqrt(2), not the 41% past it
    // that `membrane.rs` tries — the refusal has to hold near the edge, not only far from it.
    let (n, a) = (40, 0.5);
    let h = 2.0 * a / n as f64;
    let fs = c() / (1.05 * lambda_max() * h);
    let err = Params::new(
        Some(Domain::Circle),
        T,
        RHO,
        fs,
        n,
        None,
        None,
        Some(a),
        0.0,
    )
    .expect_err("lambda past the 2-D ceiling");
    assert!(matches!(err, ParamError::CflViolated(_)), "{err:?}");
    assert!(err.to_string().contains("CFL"), "{err}");
}

#[test]
fn the_ceiling_itself_is_accepted_and_reported() {
    // `test_lambda_at_cfl_ceiling_accepted`: the guard must not reject lambda = 1/sqrt(2) on
    // round-off, and the model must report the lambda it was asked for.
    let m = membrane(Domain::Rectangle, 40, lambda_max(), 0.0);
    let lam = m.params().lam;
    assert!((lam - lambda_max()).abs() <= 1e-9, "lam = {lam}");
}

#[test]
fn non_physical_or_missing_parameters_are_refused() {
    // `test_invalid_parameters_rejected`, its seven cases on its base disk.
    let base = |domain: Option<Domain>, t: f64, rho: f64, n: i64, r: Option<f64>, sigma: f64| {
        Params::new(domain, t, rho, 200_000.0, n, None, None, r, sigma)
    };
    let circle = Some(Domain::Circle);
    assert!(
        base(circle, T, RHO, 40, Some(0.5), 0.0).is_ok(),
        "the base disk is valid"
    );
    let cases: [(&str, Result<Params, ParamError>, ParamError); 7] = [
        (
            "rho = -1",
            base(circle, T, -1.0, 40, Some(0.5), 0.0),
            ParamError::NonPositive,
        ),
        (
            "T = 0",
            base(circle, 0.0, RHO, 40, Some(0.5), 0.0),
            ParamError::NonPositive,
        ),
        (
            "sigma = -0.1",
            base(circle, T, RHO, 40, Some(0.5), -0.1),
            ParamError::NegativeSigma,
        ),
        (
            "N = 1",
            base(circle, T, RHO, 1, Some(0.5), 0.0),
            ParamError::TooFewSegments,
        ),
        (
            "domain = triangle",
            base(Domain::parse("triangle"), T, RHO, 40, Some(0.5), 0.0),
            ParamError::BadDomain,
        ),
        (
            "radius = -0.5",
            base(circle, T, RHO, 40, Some(-0.5), 0.0),
            ParamError::NonPositiveRadius,
        ),
        (
            "radius = None",
            base(circle, T, RHO, 40, None, 0.0),
            ParamError::CircleNeedsRadius,
        ),
    ];
    for (name, got, want) in cases {
        assert_eq!(got.expect_err(name), want, "{name}");
    }
}

#[test]
fn a_rectangle_without_sides_says_rectangle() {
    // `test_rectangle_requires_sides`, matched on the message as the Python matched it.
    let err = Params::new(
        Some(Domain::Rectangle),
        T,
        RHO,
        200_000.0,
        40,
        None,
        None,
        None,
        0.0,
    )
    .expect_err("no sides");
    assert_eq!(err, ParamError::RectangleNeedsSides);
    assert!(err.to_string().contains("rectangle"), "{err}");
}
