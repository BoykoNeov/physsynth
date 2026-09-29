//! Native validation of the ideal string — model #1, the project's first acceptance contract.
//!
//! This file began as the port's native floor beside the Python harness. Since retirement plan
//! §31 it IS the harness: `tests/test_{energy,modal,convergence,dispersion}.py` and the string
//! half of `tests/test_stability.py` were carried here and deleted, and the oracle-only claims
//! they made went to `crates/physsynth-analysis/tests/oracles.rs`. The carried bars run at the
//! retired Python helper `make_string`'s parameters (`L = 1`, `T = 200`, `rho = 0.005`, so
//! `c = 200 m/s` and `f1 = 100 Hz`), pluck at `0.137 L` and read the pickup where the Python did.
//!
//! **No outside referee retires here.** Every number the Python tests compared against was a
//! closed form (`n c / 2L`, `exp(-2 sigma t)`, the discrete dispersion relation) or already ran
//! through the Rust binding (the exciter, `simulate`, the spectrum detector, every oracle), so
//! nothing had to be frozen before the deletion — unlike the plates, the membrane and the beam.
//!
//! The bars are the project's, unchanged: lossless energy drift below `1e-10` (CLAUDE.md's
//! deliberate headroom over the ~1e-15 actually observed), passivity strictly monotonic, and
//! modal frequencies against a closed form rather than against a previous run.

use physsynth_analysis::dispersion::{dispersion_frequencies, phase_velocity};
use physsynth_analysis::modal::{cents, discrete_mode_frequency, harmonic_frequencies, mode_shape};
use physsynth_analysis::spectrum::{detect_peaks, measure_partials_near};
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_ideal::{Boundary, IdealString, ParamError, Params};

/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;
/// Acceptance criterion 2: a partial within one cent of its oracle.
const ONE_CENT: f64 = 1.0;
const FIXED: (Boundary, Boundary) = (Boundary::Fixed, Boundary::Fixed);

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, fundamental f1 = 100 Hz

/// A string whose Courant number is exactly `lam`, by choosing `fs = c N / (L lam)`.
/// This was the Python harness's `make_string` (deleted with its last caller, §31), so the
/// carried bars build the same object the retired tests did.
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

/// `wave_speed()` — 200 m/s.
fn c() -> f64 {
    (T / RHO).sqrt()
}

/// `np.max`: NaN-propagating, unlike `fold(_, f64::max)`, which drops a NaN and so reports a
/// clean figure for a run that blew up (§29.3). Every worst-case figure in this file uses it.
fn nan_max(it: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for v in it {
        if v.is_nan() {
            return f64::NAN;
        }
        m = m.max(v);
    }
    m
}

/// The Python harness's `_run`: pluck at `frac` of the length with the model's own exciter, run
/// `secs` through the driver, read the pickup at `round(pickup_frac N)`.
fn run(s: &mut IdealString, frac: f64, secs: f64, pickup_frac: f64) -> SimResult {
    let p = s.params().clone();
    let u0 = triangular_pluck(&p.grid(), p.l, frac * p.l, 1e-3).expect("an interior pluck");
    s.set_displacement(&u0);
    let steps = (secs * p.fs) as usize;
    let pickup = (pickup_frac * p.n as f64).round() as usize;
    simulate(s, steps, Some(pickup), 0).expect("a lossless or passive run completes")
}

/// `measure_mode_frequencies`: the frequency of each single mode, read off its MODAL COORDINATE
/// `q = <u, phi_m>` rather than a point pickup — a point can sit on a node of the mode (mode N/2
/// vanishes at every even node), while the projection has full SNR for every mode and any N. The
/// peak search is anchored at the discrete oracle, as the Python's was.
fn measure_mode_frequencies(modes: &[i64], n: usize, lam: f64) -> Vec<f64> {
    modes
        .iter()
        .map(|&m| {
            let mut s = make(n as i64, lam, FIXED, 0.0);
            let p = s.params().clone();
            let phi = mode_shape(&p.grid(), L, m);
            let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
            s.set_displacement(&u0);
            let dot = |u: &[f64]| u.iter().zip(&phi).map(|(a, b)| a * b).sum::<f64>();
            let steps = (0.5 * p.fs) as usize;
            let mut q = Vec::with_capacity(steps + 1);
            q.push(dot(&s.u));
            for _ in 0..steps {
                s.step();
                q.push(dot(&s.u));
            }
            let oracle = discrete_mode_frequency(c(), L, n as i64, lam, m);
            measure_partials_near(&q, p.fs, &[oracle], None)[0]
        })
        .collect()
}

/// The driver's relative drift, NaN-propagating (`engine.rs` proves it), so a run that blew up
/// cannot pass the bar the way a `worst.max(..)` fold would let it.
fn drift_through_the_driver(s: &mut IdealString, u0: &[f64], steps: usize) -> f64 {
    s.set_displacement(u0);
    simulate(s, steps, None, 0)
        .expect("a lossless run completes")
        .energy_drift()
}

#[test]
fn lossless_energy_is_conserved() {
    let mut s = make(100, 1.0, FIXED, 0.0);
    let u0 = pluck(&s.params().grid(), 0.3, 1e-3);
    s.set_displacement(&u0);
    assert!(s.energy() > 0.0, "a plucked string must start with energy");

    let drift = drift_through_the_driver(&mut s, &u0, 10_000);
    assert!(
        drift < DRIFT_TOL,
        "lossless energy drifted by {drift:e} (bar is 1e-10)"
    );
}

#[test]
fn lossless_energy_is_conserved_and_positive_at_every_courant_number() {
    // Criterion 1 (`test_energy_conserved_across_lambda`): the conservation identity is algebraic,
    // not a lambda = 1 special case, so it must hold at every admissible Courant number — which is
    // what catches a stencil coefficient that is right only at lambda = 1 (lambda written for
    // lambda^2 is invisible there). The whole 2 s of every run is also asserted strictly positive
    // (`test_energy_strictly_positive_when_lossless` asked it of the first second at 0.9; this is
    // a superset). `e > 0.0` is false for a NaN, so a blow-up fails it too.
    for lam in [1.0, 0.99, 0.9, 0.7, 0.5] {
        let mut s = make(100, lam, FIXED, 0.0);
        let r = run(&mut s, 0.137, 2.0, 0.241);
        let drift = r.energy_drift();
        eprintln!(
            "lam={lam}: drift {drift:e} over {} steps",
            r.energy.len() - 1
        );
        assert!(drift < DRIFT_TOL, "drift {drift:e} at lambda={lam}");
        assert!(
            r.energy.iter().all(|&e| e > 0.0),
            "a lossless energy was not strictly positive at lambda={lam}"
        );
    }
}

#[test]
fn a_free_end_conserves_energy_too() {
    // The boundary is the first suspect whenever E drifts, and a free end is the case where the
    // half-cell trapezoidal weights stop being a formality: with `w[0] = h/2` wrong, the kinetic
    // term double-counts the moving end and the drift is visible immediately. Each combination
    // runs at lambda = 1 (the old native case, pluck at 0.3) and at `test_energy_conserved_free_
    // boundary`'s lambda = 0.9 with its pluck, both for the Python's 2 s — no run shorter than
    // the 5,000 steps it had before.
    for bc in [
        (Boundary::Free, Boundary::Free),
        (Boundary::Fixed, Boundary::Free),
        (Boundary::Free, Boundary::Fixed),
    ] {
        for (lam, frac) in [(1.0, 0.3), (0.9, 0.137)] {
            let mut s = make(100, lam, bc, 0.0);
            let u0 = pluck(&s.params().grid(), frac, 1e-3);
            let steps = ((2.0 * s.params().fs) as usize).max(5_000);
            let drift = drift_through_the_driver(&mut s, &u0, steps);
            eprintln!("{bc:?} lam={lam}: drift {drift:e}");
            assert!(
                drift < DRIFT_TOL,
                "{bc:?} at lambda={lam} drifted by {drift:e}"
            );
        }
    }
}

#[test]
fn loss_makes_the_energy_decrease_monotonically() {
    // "No step may increase the energy, allowing a hair of round-off relative to E0" — the form of
    // `test_passivity_monotonic_decrease`, and for its reason: a strictly-decreasing assertion is
    // a claim about the last bit of a difference of two nearly equal sums, which is not a
    // physical statement. Two runs: the old native one (sigma = 3, pluck 0.3, 5,000 steps) and the
    // Python's (sigma = 5, pluck 0.137, 2 s through the driver).
    for (sigma, frac, secs) in [(3.0, 0.3, 0.25), (5.0, 0.137, 2.0)] {
        let mut s = make(100, 1.0, FIXED, sigma);
        let r = run(&mut s, frac, secs, 0.241);
        let e0 = r.energy[0];
        let worst_rise = nan_max(r.energy.windows(2).map(|w| w[1] - w[0]));
        eprintln!("sigma={sigma}: worst rise {:e} E0", worst_rise / e0);
        assert!(
            worst_rise <= 1e-12 * e0,
            "sigma={sigma}: the energy rose by {worst_rise:e} (E0 = {e0:e})"
        );
    }
}

#[test]
fn the_decay_rate_matches_the_analytic_two_sigma() {
    // Monotonicity alone is satisfied by a string that loses energy for the wrong reason — a
    // scheme leaking through the boundary decays beautifully. The rate is what pins it: a
    // uniformly damped string loses energy as `E(t) = E0 exp(-2 sigma t)`. Compared in log space,
    // at `test_decay_rate_matches_2sigma`'s 2% (this file had 5% before the carry).
    let sigma = 4.0;
    let secs = 1.0;
    let mut s = make(100, 1.0, FIXED, sigma);
    let r = run(&mut s, 0.137, secs, 0.241);
    let measured = (r.energy[r.energy.len() - 1] / r.energy[0]).ln();
    let expected = -2.0 * sigma * secs;
    let rel = ((measured - expected) / expected).abs();
    eprintln!("decay: rel {rel:e}");
    assert!(
        rel < 0.02,
        "decay rate off by {rel:.3}: ln ratio {measured} vs {expected}"
    );
}

#[test]
fn the_energy_is_in_joules_and_scales_with_density() {
    // `test_energy_units_scale_with_density`: doubling rho AND T keeps c, and so the grid, the
    // sample rate and the displacement field, identical — doubling 0.005 and 200 is exact — which
    // isolates the rho prefactor. The Python's `np.isclose(rtol=1e-12)` carried NumPy's default
    // `atol = 1e-8` (§16's (d)); the bar here is EXACT, and that is a structural claim rather than
    // a measured one: every factor but rho is bit-identical between the two strings, and rho was
    // doubled, which is exact in binary floating point on any IEEE platform.
    let s1 = {
        let mut s = make_with(80, 0.9, RHO, T);
        let u0 = pluck(&s.params().grid(), 0.3, 1e-3);
        s.set_displacement(&u0);
        s
    };
    let s2 = {
        let mut s = make_with(80, 0.9, 2.0 * RHO, 2.0 * T);
        let u0 = pluck(&s.params().grid(), 0.3, 1e-3);
        s.set_displacement(&u0);
        s
    };
    assert_eq!(
        s1.params().fs,
        s2.params().fs,
        "the control: the two grids must be identical"
    );
    assert_eq!(
        s1.u, s2.u,
        "the control: the two displacement fields must be identical"
    );
    assert_eq!(s2.energy(), 2.0 * s1.energy(), "E is not linear in rho");
}

/// [`make`] at a non-default density and tension.
fn make_with(n: i64, lam: f64, rho: f64, t: f64) -> IdealString {
    let c = (t / rho).sqrt();
    let fs = c * (n as f64) / (L * lam);
    IdealString::new(Params::new(L, t, rho, fs, n, 0.0, Some(FIXED)).expect("valid parameters"))
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

    // `test_invalid_parameters_rejected`'s `boundary="clamped"` is a refusal about a VALUE, so it
    // has a native analogue (§14): the spelling is read by `Boundary::parse`, and anything it does
    // not know becomes the `None` above. The two it does know are the control.
    assert_eq!(
        Boundary::parse("clamped"),
        None,
        "a clamped end is not an ideal-string boundary"
    );
    assert_eq!(Boundary::parse("fixed"), Some(Boundary::Fixed));
    assert_eq!(Boundary::parse("free"), Some(Boundary::Free));
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

// -- carried from `tests/test_modal.py` (criterion 2) -----------------------------------------

#[test]
fn a_plucked_string_sounds_its_harmonic_series_within_a_cent() {
    // `test_partials_within_one_cent_at_lambda_one`: at lambda = 1 the scheme is dispersion-free,
    // so the partials sit on n c / 2L. Why ten and not a derived band: at lambda = 1 the leapfrog's
    // time error cancels the spatial droop exactly, so there is no resolution horizon to read; the
    // limiter is DETECTABILITY — a triangular pluck's partials fall off like 1/n^2, so past a
    // couple of dozen the peak is not reliably separable from the floor. That is a property of
    // the excitation and the FFT, not of the scheme (docs/dev/resolution-horizon-plan.md §2).
    let mut s = make(100, 1.0, FIXED, 0.0);
    let r = run(&mut s, 0.137, 2.0, 0.241);
    let analytic = harmonic_frequencies(c(), L, 10);
    let detected = measure_partials_near(r.output.as_ref().unwrap(), r.fs, &analytic, None);
    let worst = nan_max(
        detected
            .iter()
            .zip(&analytic)
            .map(|(&f, &a)| cents(f, a).abs()),
    );
    eprintln!("guided: worst {worst:e} cents");
    assert!(!worst.is_nan(), "a partial was not detected: {detected:?}");
    assert!(worst < ONE_CENT, "worst partial error {worst:.4} cents");
}

#[test]
fn the_blind_detector_finds_the_harmonic_series_on_its_own() {
    // `test_blind_detection_finds_harmonic_series`: with no prior knowledge, every strong peak must
    // lie on the grid n f1 — the guard against the guided search merely confirming its own
    // assumption. Which harmonics dominate depends on the pluck and pickup, so each peak must snap
    // to SOME integer harmonic within a cent, the six must be distinct, and the fundamental hit.
    let mut s = make(100, 1.0, FIXED, 0.0);
    let r = run(&mut s, 0.137, 2.0, 0.241);
    let f1 = c() / (2.0 * L);
    let peaks = detect_peaks(r.output.as_ref().unwrap(), r.fs, 6, 50.0, None);
    assert_eq!(peaks.len(), 6, "{peaks:?}");

    let nearest: Vec<f64> = peaks.iter().map(|&p| (p / f1).round()).collect();
    let worst = nan_max(
        peaks
            .iter()
            .zip(&nearest)
            .map(|(&p, &n)| cents(p, n * f1).abs()),
    );
    eprintln!("blind: peaks {peaks:?}, harmonics {nearest:?}, worst {worst:e} cents");
    assert!(
        worst < ONE_CENT,
        "a peak is off-grid by {worst} cents: {peaks:?}"
    );
    let mut distinct = nearest.clone();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        6,
        "peaks collapsed onto duplicate harmonics: {nearest:?}"
    );
    assert_eq!(
        nearest[0], 1.0,
        "the fundamental is not among the peaks: {nearest:?}"
    );
}

// -- carried from `tests/test_convergence.py` (criterion 3) -----------------------------------

#[test]
fn a_dispersive_mode_converges_at_second_order_onto_the_dispersion_oracle() {
    // At lambda = 1 the scheme is exact for any h, so refinement shows nothing; the rate is
    // measured at a fixed lambda = 0.9, where dispersion leaves an O(h^2) frequency error that must
    // shrink ~4x per halving. One spatial mode (m = 8, high enough that the error stays well above
    // the spectral floor across the refinement) makes the output a clean single tone.
    //
    // Both of the Python file's tests in one: `test_second_order_convergence_at_fixed_lambda` is
    // the refinement, and `test_detected_frequency_tracks_dispersion_oracle` is its N = 128 run
    // read against the closed-form oracle — confirming the error IS dispersion, not some other
    // O(h^2) defect that happens to converge at the right rate.
    let (lam, m) = (0.9, 8_i64);
    let f_cont = m as f64 * c() / (2.0 * L);
    let grids = [64_usize, 128, 256];
    let mut errors = Vec::new();
    for &n in &grids {
        let mut s = make(n as i64, lam, FIXED, 0.0);
        let p = s.params().clone();
        let u0: Vec<f64> = mode_shape(&p.grid(), L, m)
            .iter()
            .map(|v| v * 1e-3)
            .collect();
        s.set_displacement(&u0);
        let pickup = (0.413 * n as f64).round() as usize;
        let r = simulate(&mut s, (1.5 * p.fs) as usize, Some(pickup), 0).unwrap();
        let f_det = measure_partials_near(r.output.as_ref().unwrap(), r.fs, &[f_cont], None)[0];
        errors.push((f_det - f_cont).abs());

        if n == 128 {
            let f_oracle = discrete_mode_frequency(c(), L, n as i64, lam, m);
            assert!(
                f_oracle < f_cont,
                "dispersion must lower the frequency below lambda = 1"
            );
            let off = (f_det - f_oracle).abs();
            let bar = 0.01 * (f_cont - f_oracle) + 1e-3;
            eprintln!("N=128: |f_det - f_oracle| {off:e} against bar {bar:e}");
            assert!(
                off < bar,
                "detected {f_det} vs oracle {f_oracle} (bar {bar:e})"
            );
        }
    }
    // `convergence_orders`: log(e_i / e_{i+1}) / log(h_i / h_{i+1}), with h = L / N.
    let orders: Vec<f64> = (0..grids.len() - 1)
        .map(|i| (errors[i] / errors[i + 1]).ln() / (grids[i + 1] as f64 / grids[i] as f64).ln())
        .collect();
    let mean = orders.iter().sum::<f64>() / orders.len() as f64;
    eprintln!("errors {errors:?}, orders {orders:?}, mean {mean}");
    assert!(
        errors.windows(2).all(|w| w[1] < w[0]),
        "errors did not shrink: {errors:?}"
    );
    assert!(
        orders.iter().all(|&o| o > 1.7),
        "a refinement fell below 2nd order: {orders:?}"
    );
    assert!(1.85 < mean && mean < 2.15, "mean order {mean:.3} is not ~2");
}

// -- carried from `tests/test_dispersion.py` (HANDOFF §6 test 5) ------------------------------

/// A coarse grid with low modes well below Nyquist, swept to 0.75 N (the top decile is
/// measurement-limited).
const DISPERSION_N: usize = 128;
const DISPERSION_MODES: [i64; 9] = [2, 4, 8, 16, 32, 48, 64, 80, 96];

#[test]
fn every_swept_mode_lands_on_the_dispersion_oracle_and_droops_below_c() {
    // `test_dispersion_matches_oracle_below_lambda_one` — the implementation test: every swept
    // mode's measured frequency is on the closed-form discrete oracle at lambda = 0.8, a regime the
    // lambda = 1 bars never exercise and where a coefficient or indexing bug hides. The worst error
    // is the lowest mode, limited only by FFT resolution over the window.
    //
    // `test_phase_velocity_droops_with_mode_below_lambda_one`, on the same measurement — the
    // direction-of-physics anchor: dispersion slows high partials, so v_p / c < 1 and falls
    // strictly with mode number.
    let lam = 0.8;
    let measured = measure_mode_frequencies(&DISPERSION_MODES, DISPERSION_N, lam);
    let oracle = dispersion_frequencies(c(), L, DISPERSION_N as i64, lam, &DISPERSION_MODES);
    let worst = nan_max(
        measured
            .iter()
            .zip(&oracle)
            .map(|(&f, &o)| (f - o).abs() / o),
    );
    eprintln!("lambda 0.8: worst measured-vs-oracle {worst:e}");
    assert!(
        !worst.is_nan(),
        "a mode frequency was not detected: {measured:?}"
    );
    assert!(worst < 1e-4, "worst measured-vs-oracle error {worst:e}");

    let vp_over_c: Vec<f64> = phase_velocity(&measured, L, &DISPERSION_MODES)
        .iter()
        .map(|v| v / c())
        .collect();
    eprintln!("lambda 0.8: v_p / c {vp_over_c:?}");
    assert!(
        vp_over_c.iter().all(|&v| v < 1.0),
        "a phase velocity reached c: {vp_over_c:?}"
    );
    assert!(
        vp_over_c.windows(2).all(|w| w[1] < w[0]),
        "the phase velocity does not droop monotonically: {vp_over_c:?}"
    );
}

#[test]
fn at_courant_one_every_swept_mode_is_on_the_continuum() {
    // `test_dispersion_flat_at_lambda_one` — the physics anchor: at lambda = 1 the measured curve
    // sits on the CONTINUUM n c / 2L, not only on an oracle drawn from the same recurrence, and
    // the phase velocity is flat at c. The Python's `np.allclose(vp/c, 1, atol=1e-7)` kept
    // `rtol = 1e-5` (§16's (d)); asserted here as the bare 1e-7 it was written as.
    let measured = measure_mode_frequencies(&DISPERSION_MODES, DISPERSION_N, 1.0);
    let continuum: Vec<f64> = DISPERSION_MODES
        .iter()
        .map(|&m| m as f64 * c() / (2.0 * L))
        .collect();
    let worst = nan_max(
        measured
            .iter()
            .zip(&continuum)
            .map(|(&f, &o)| (f - o).abs() / o),
    );
    eprintln!("lambda 1: worst measured-vs-continuum {worst:e}");
    assert!(
        !worst.is_nan(),
        "a mode frequency was not detected: {measured:?}"
    );
    assert!(
        worst < 1e-7,
        "worst measured-vs-continuum error at lambda = 1: {worst:e}"
    );

    let vp = phase_velocity(&measured, L, &DISPERSION_MODES);
    let flat = nan_max(vp.iter().map(|v| (v / c() - 1.0).abs()));
    eprintln!("lambda 1: worst |v_p/c - 1| {flat:e}");
    assert!(
        flat < 1e-7,
        "the phase velocity is not flat at lambda = 1: {vp:?}"
    );
}

// -- carried from `tests/test_stability.py` (criterion 4) -------------------------------------

#[test]
fn no_admissible_courant_number_produces_a_nan() {
    // `test_no_nan_across_valid_lambda`: half a second at each lambda in (0, 1), down to 0.1 (a
    // ten-times-oversampled string, 100,000 steps), with the pickup and the energy both finite.
    for lam in [0.999, 0.95, 0.9, 0.75, 0.5, 0.3, 0.1] {
        let mut s = make(100, lam, FIXED, 0.0);
        let u0 = triangular_pluck(&s.params().grid(), L, 0.3 * L, 1e-3).unwrap();
        s.set_displacement(&u0);
        let steps = (0.5 * s.params().fs) as usize;
        let r = simulate(&mut s, steps, Some(50), 0).unwrap();
        assert!(
            r.output.as_ref().unwrap().iter().all(|v| v.is_finite()),
            "lambda={lam}"
        );
        assert!(r.energy.iter().all(|e| e.is_finite()), "lambda={lam}");
    }
}
