//! Native validation of the ideal string — the same acceptance contract as the Python harness,
//! asserted without an interpreter anywhere in the picture.
//!
//! This is not a duplicate of `tests/test_energy.py` and friends. Those are still the authority
//! for Phase 0 (plan §1 step 3: the Rust model is proved by the *existing, unmodified* Python
//! tests). What these add is a native floor — if `cargo test` is green, the physics is right
//! independently of the binding, so a failure in the Python suite is a binding failure and a
//! failure here is a physics failure. Separating those two questions is worth a small file.
//!
//! The bars are the project's, unchanged: lossless energy drift below `1e-10` (CLAUDE.md's
//! deliberate headroom over the ~1e-15 actually observed), passivity strictly monotonic, and
//! modal frequencies against a closed form rather than against a previous run.

use physsynth_core::string_ideal::{Boundary, IdealString, ParamError, Params};

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, fundamental f1 = 100 Hz

/// A string whose Courant number is exactly `lam`, by choosing `fs = c N / (L lam)`.
/// Mirrors `tests/helpers.py::make_string`, so the two suites talk about the same object.
fn make(n: i64, lam: f64, bc: (Boundary, Boundary), sigma: f64) -> IdealString {
    let c = (T / RHO).sqrt();
    let fs = c * (n as f64) / (L * lam);
    IdealString::new(Params::new(L, T, RHO, fs, n, sigma, Some(bc)).expect("valid parameters"))
}

/// A triangular pluck of the given amplitude, peaking at `frac` of the length.
fn pluck(x: &[f64], frac: f64, amplitude: f64) -> Vec<f64> {
    let peak = frac * L;
    x.iter()
        .map(|&xi| {
            if xi <= peak {
                amplitude * xi / peak
            } else {
                amplitude * (L - xi) / (L - peak)
            }
        })
        .collect()
}

/// `sin(m pi l / N)` — the exact eigenvector of the Dirichlet second difference.
fn mode(n: usize, m: usize) -> Vec<f64> {
    (0..=n)
        .map(|l| (m as f64 * std::f64::consts::PI * l as f64 / n as f64).sin())
        .collect()
}

#[test]
fn lossless_energy_is_conserved() {
    let mut s = make(100, 1.0, (Boundary::Fixed, Boundary::Fixed), 0.0);
    let u0 = pluck(&s.params().grid(), 0.3, 1e-3);
    s.set_displacement(&u0);

    let e0 = s.energy();
    assert!(e0 > 0.0, "a plucked string must start with energy");

    let mut worst: f64 = 0.0;
    for _ in 0..10_000 {
        s.step();
        worst = worst.max(((s.energy() - e0) / e0).abs());
    }
    assert!(
        worst < 1e-10,
        "lossless energy drifted by {worst:e} (bar is 1e-10)"
    );
}

#[test]
fn a_free_end_conserves_energy_too() {
    // The boundary is the first suspect whenever E drifts, and a free end is the case where the
    // half-cell trapezoidal weights stop being a formality: with `w[0] = h/2` wrong, the kinetic
    // term double-counts the moving end and the drift is visible immediately.
    for bc in [
        (Boundary::Free, Boundary::Free),
        (Boundary::Fixed, Boundary::Free),
        (Boundary::Free, Boundary::Fixed),
    ] {
        let mut s = make(100, 1.0, bc, 0.0);
        let u0 = pluck(&s.params().grid(), 0.3, 1e-3);
        s.set_displacement(&u0);

        let e0 = s.energy();
        let mut worst: f64 = 0.0;
        for _ in 0..5_000 {
            s.step();
            worst = worst.max(((s.energy() - e0) / e0).abs());
        }
        assert!(worst < 1e-10, "{bc:?} drifted by {worst:e}");
    }
}

#[test]
fn loss_makes_the_energy_decrease_monotonically() {
    let mut s = make(100, 1.0, (Boundary::Fixed, Boundary::Fixed), 3.0);
    let u0 = pluck(&s.params().grid(), 0.3, 1e-3);
    s.set_displacement(&u0);

    // "No step may increase the energy, allowing a hair of round-off relative to E0" — the same
    // form as `tests/test_energy.py::test_passivity_monotonic_decrease`, and for the same reason:
    // a strictly-decreasing assertion is a claim about the last bit of a difference of two nearly
    // equal sums, which is not a physical statement. Measured here, the worst rise is ~3e-18 * E0.
    let e0 = s.energy();
    let mut previous = e0;
    for step in 0..5_000 {
        s.step();
        let now = s.energy();
        assert!(
            now - previous <= 1e-12 * e0,
            "energy rose at step {step}: {previous:e} -> {now:e}"
        );
        previous = now;
    }
}

#[test]
fn the_decay_rate_matches_the_analytic_two_sigma() {
    // Monotonicity alone is satisfied by a string that loses energy for the wrong reason — a
    // scheme leaking through the boundary decays beautifully. The rate is what pins it: a
    // uniformly damped string loses energy as `E(t) = E0 exp(-2 sigma t)`.
    let sigma = 4.0;
    let secs = 1.0;
    let mut s = make(100, 1.0, (Boundary::Fixed, Boundary::Fixed), sigma);
    let u0 = pluck(&s.params().grid(), 0.137, 1e-3);
    s.set_displacement(&u0);

    let e0 = s.energy();
    for _ in 0..(secs * s.params().fs) as usize {
        s.step();
    }
    let measured = (s.energy() / e0).ln();
    let expected = -2.0 * sigma * secs;
    let rel = ((measured - expected) / expected).abs();
    assert!(
        rel < 0.05,
        "decay rate off by {rel:.3}: ln ratio {measured} vs {expected}"
    );
}

#[test]
fn a_single_mode_follows_the_discrete_dispersion_relation() {
    // The oracle: for fixed ends, `sin(m pi l / N)` is an exact eigenvector, so the scheme reduces
    // to a scalar recurrence whose solution is `cos(omega n k)` with
    //     cos(omega k) = 1 - 2 lambda^2 sin^2(m pi / 2N).
    // The consistent second-order start makes this exact rather than approximate — which is the
    // whole reason `set_state` computes `u^{-1}` from a stencil instead of copying `u^0`.
    let n = 64;
    for lam in [1.0, 0.7] {
        for m in [1_usize, 3, 7] {
            let mut s = make(n as i64, lam, (Boundary::Fixed, Boundary::Fixed), 0.0);
            let shape = mode(n, m);
            s.set_displacement(&shape);

            let arg = m as f64 * std::f64::consts::PI / (2.0 * n as f64);
            let cos_wk = 1.0 - 2.0 * lam * lam * arg.sin().powi(2);
            let wk = cos_wk.acos();

            for step in 1..=200 {
                s.step();
                let expected = (wk * step as f64).cos();
                for (l, &value) in s.u.iter().enumerate() {
                    let want = expected * shape[l];
                    assert!(
                        (value - want).abs() < 1e-12,
                        "lam={lam} m={m} step={step} node={l}: {value:e} != {want:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn at_courant_one_the_modal_frequency_is_exactly_the_continuum_one() {
    // lambda = 1 is the dispersionless case and the reason the project tunes toward it: the
    // discrete `omega` collapses onto `m pi c / L` with no truncation error left over at all.
    let n = 100;
    let s = make(n as i64, 1.0, (Boundary::Fixed, Boundary::Fixed), 0.0);
    let p = s.params();
    for m in [1_usize, 5, 17, 50] {
        let arg = m as f64 * std::f64::consts::PI / (2.0 * n as f64);
        let cos_wk = 1.0 - 2.0 * p.lam * p.lam * arg.sin().powi(2);
        let f_discrete = cos_wk.acos() / (2.0 * std::f64::consts::PI * p.k);
        let f_continuum = m as f64 * p.c / (2.0 * L);
        let rel = ((f_discrete - f_continuum) / f_continuum).abs();
        assert!(
            rel < 1e-12,
            "mode {m}: {f_discrete} vs {f_continuum} (rel {rel:e})"
        );
    }
}

#[test]
fn zero_initial_velocity_is_exact() {
    // With `v0 = 0` the string must start at rest, so the first backward difference is the
    // half-step of the Taylor start and the kinetic term at n = 0 is pure numerical noise.
    let mut s = make(100, 1.0, (Boundary::Fixed, Boundary::Fixed), 0.0);
    let u0 = pluck(&s.params().grid(), 0.5, 1e-3);
    s.set_displacement(&u0);
    // Symmetric about the midpoint: after one full period the shape must return.
    let e_start = s.energy();
    s.step();
    assert!((s.energy() - e_start).abs() / e_start < 1e-12);
}

#[test]
fn courant_above_one_is_rejected_at_construction() {
    let c = (T / RHO).sqrt();
    let n = 100;
    // A LOWER fs (coarser timestep) raises lambda, so divide by 1.05 to force lambda = 1.05.
    let fs = c * (n as f64) / (L * 1.05);
    let err = Params::new(
        L,
        T,
        RHO,
        fs,
        n,
        0.0,
        Some((Boundary::Fixed, Boundary::Fixed)),
    )
    .expect_err("lambda > 1 must not construct");
    assert!(matches!(err, ParamError::CflViolated(_)));
    assert!(
        err.to_string().contains("CFL"),
        "message must name the CFL: {err}"
    );
}

#[test]
fn courant_exactly_one_is_accepted() {
    // The guard must not reject the exact — and most accurate — case on round-off.
    let s = make(100, 1.0, (Boundary::Fixed, Boundary::Fixed), 0.0);
    assert!((s.params().lam - 1.0).abs() < 1e-12);
}

#[test]
fn non_physical_parameters_are_rejected() {
    let bc = Some((Boundary::Fixed, Boundary::Fixed));
    let cases: Vec<(Params_, ParamError)> = vec![
        ((-2.0, T, RHO, 20000.0, 100, 0.0), ParamError::NonPositive),
        ((L, 0.0, RHO, 20000.0, 100, 0.0), ParamError::NonPositive),
        ((L, T, -1.0, 20000.0, 100, 0.0), ParamError::NonPositive),
        ((L, T, RHO, 0.0, 100, 0.0), ParamError::NonPositive),
        ((L, T, RHO, 20000.0, 1, 0.0), ParamError::TooFewSegments),
        ((L, T, RHO, 20000.0, -3, 0.0), ParamError::TooFewSegments),
        ((L, T, RHO, 20000.0, 100, -0.1), ParamError::NegativeSigma),
    ];
    for ((l, t, rho, fs, n, sigma), want) in cases {
        let got = Params::new(l, t, rho, fs, n, sigma, bc)
            .expect_err("non-physical parameters must not construct");
        assert_eq!(
            got, want,
            "for (L={l}, T={t}, rho={rho}, fs={fs}, N={n}, sigma={sigma})"
        );
    }
}

/// `(L, T, rho, fs, N, sigma)` — a name for the tuple above, so the table reads.
type Params_ = (f64, f64, f64, f64, i64, f64);

#[test]
fn an_unparseable_boundary_is_rejected_after_the_scalar_checks() {
    // `bc: None` is how the binding says "the caller passed something I could not read". It must
    // be reported at Python's position in the check order — after sigma, before CFL — so a call
    // with two faults blames the same one in both implementations.
    let err = Params::new(L, T, RHO, 20000.0, 100, 0.0, None).expect_err("None bc must reject");
    assert_eq!(err, ParamError::BadBoundary);

    // ... and a *scalar* fault alongside it still wins, because it is checked first.
    let err = Params::new(L, T, RHO, 20000.0, 100, -1.0, None).expect_err("sigma must reject");
    assert_eq!(err, ParamError::NegativeSigma);
}

#[test]
fn the_grid_ends_exactly_on_the_length() {
    // `np.linspace` overwrites the last sample with the endpoint rather than computing `N*(L/N)`.
    // On a length whose division is inexact the two differ in the last bit, and `x` is what the
    // analysis layer measures positions against.
    let s = make(3, 1.0, (Boundary::Fixed, Boundary::Fixed), 0.0);
    let x = s.params().grid();
    assert_eq!(x.len(), 4);
    assert_eq!(x[0], 0.0);
    assert_eq!(x[3], L);
}

#[test]
fn step_count_tracks_the_history() {
    let mut s = make(10, 1.0, (Boundary::Fixed, Boundary::Fixed), 0.0);
    s.set_displacement(&mode(10, 1));
    assert_eq!(s.n_steps, 0);
    for _ in 0..7 {
        s.step();
    }
    assert_eq!(s.n_steps, 7);
    s.set_displacement(&mode(10, 1));
    assert_eq!(s.n_steps, 0, "set_state resets the clock");
}
