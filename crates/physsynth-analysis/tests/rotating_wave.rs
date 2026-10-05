//! Native bars for the rotating-wave BVP.
//!
//! Three kinds of claim, and only the first is about agreeing with Python:
//!
//! 1. **The reference fixtures** — the test suite's own string, solved, checked against the numbers
//!    the Python module produces. Tolerance, not equality: this is Group D and the module header
//!    says why (a supernodal SuperLU against a plain Gilbert–Peierls elimination). The measured
//!    margin is what makes the bars here meaningful rather than arbitrary — a perturbation six
//!    orders larger than the two implementations can differ by moves `Omega` by one ulp.
//! 2. **The `R -> 0` gate**, which is the strongest single claim in the module: at vanishing
//!    amplitude the BVP must collapse onto the theta-scheme's own linear dispersion, a closed form
//!    written eight models before this one and knowing nothing about helices.
//! 3. **The structural identities** — the Jacobian against finite differences of the residual, and
//!    its asymmetry being exactly the discrete-gradient time factor. These are what a transcription
//!    can silently get wrong while every number still looks plausible, and they are **not in this
//!    file**: `residual`, `jacobian` and `NewtonCtx` are private, so they live in
//!    `src/rotating_wave.rs`'s own `mod tests`. This header claimed them for a phase while they
//!    existed only in `tests/test_geometric_rotating_wave.py`, in Python — an overclaim that unit
//!    10's deletion is what found (plan §44).

use physsynth_analysis::damping::spatial_eigenvalue_p2;
use physsynth_analysis::duffing::kc_mode_coefficients;
use physsynth_analysis::modal::discrete_stiff_mode_frequency;
use physsynth_analysis::rotating_wave::{
    kc_circular_frequency, planar_hessian_cells, rotating_wave_history, solve_rotating_wave,
    BvpParams, CONTINUATION_STEPS_DEFAULT, NEWTON_MAXITER_DEFAULT, NEWTON_TOL_DEFAULT,
};

/// The test suite's own string: `c = 200 m/s`, `EA/T = 500`, `lam_long = 0.5`, `N = 32`.
fn params() -> BvpParams {
    let (l, t, rho, ea, n) = (1.0, 200.0, 0.005, 1.0e5, 32usize);
    BvpParams {
        l,
        t,
        rho,
        ea,
        fs: (ea / rho).sqrt() * n as f64 / (l * 0.5),
        n_cells: n,
        theta: 0.28,
        amplitude: 5e-3,
        mode: 1,
        kappa: 0.0,
        time_discrete: true,
        continuation_steps: CONTINUATION_STEPS_DEFAULT,
        tol: NEWTON_TOL_DEFAULT,
        maxiter: NEWTON_MAXITER_DEFAULT,
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

// -- 1. the reference fixtures ------------------------------------------------------------

#[test]
fn the_reference_helix_matches_the_python_module() {
    let w = solve_rotating_wave(&params()).expect("the canonical fixture must solve");
    assert!(w.converged);
    // The iteration count is compared exactly HERE and nowhere else, and the module header says
    // why: over 126 fixtures it differs from the Python's in 17 of the 108 that converge, so it is
    // a per-fixture fact rather than a property of the port. This fixture is one of the stable
    // ones -- a relative 1e-10 perturbation of the Newton step leaves it at 24 -- and pinning it
    // here catches a transcription that changed the continuation's path while still landing on the
    // right root. The parity file deliberately does NOT assert it.
    assert_eq!(w.iterations, 24, "the continuation's total Newton count");
    assert!(
        rel(w.omega, 637.660_275_181_959_4) < 1e-12,
        "Omega {}",
        w.omega
    );
    assert!(rel(w.s, 406_610.458_360_547_4) < 1e-12, "s {}", w.s);
    assert!(
        rel(w.frequency, 101.486_784_808_547_07) < 1e-12,
        "frequency {}",
        w.frequency
    );
    assert!(
        rel(w.shape_residual, 1.157_851_081_466_332_2e-5) < 1e-9,
        "shape_residual {}",
        w.shape_residual
    );
    assert!(rel(max_abs(&w.phi), 0.005_000_057_893_224_391) < 1e-12);
    assert!(rel(max_abs(&w.psi), 9.804_433_390_651_433e-6) < 1e-9);
    assert!(rel(w.psi[8], -9.804_433_390_651_433e-6) < 1e-9);
    assert!(rel(w.stretch_ratio[0], 1.000_061_760_991_27) < 1e-12);
    // The full grid carries the clamped ends, and they are exactly zero rather than nearly so.
    assert_eq!(w.phi.len(), 33);
    assert_eq!(
        (w.phi[0], w.phi[32], w.psi[0], w.psi[32]),
        (0.0, 0.0, 0.0, 0.0)
    );
}

#[test]
fn the_tension_field_is_non_uniform_and_that_is_the_mechanism() {
    let w = solve_rotating_wave(&params()).unwrap();
    let lo = w.tension.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    let hi = max_abs(&w.tension);
    assert!(rel(lo, 206.150_809_285_492_87) < 1e-10, "min tension {lo}");
    assert!(rel(hi, 206.176_099_127_013) < 1e-10, "max tension {hi}");
    // The point of the whole module: a Kirchhoff-Carrier tension is one scalar, and this is not.
    assert!(
        hi - lo > 0.02,
        "the tension field must vary along the string"
    );
}

#[test]
fn the_semi_discrete_and_mode_two_fixtures_match_too() {
    let semi = solve_rotating_wave(&BvpParams {
        time_discrete: false,
        ..params()
    })
    .unwrap();
    assert!(
        rel(semi.omega, 637.660_596_808_297_2) < 1e-12,
        "{}",
        semi.omega
    );
    assert_eq!(semi.iterations, 24);

    let m2 = solve_rotating_wave(&BvpParams {
        mode: 2,
        kappa: 2.0,
        amplitude: 3e-3,
        ..params()
    })
    .unwrap();
    assert!(
        rel(m2.omega, 1_284.471_855_997_820_7) < 1e-12,
        "{}",
        m2.omega
    );
    assert!(rel(m2.shape_residual, 1.623_721_180_113_173_8e-5) < 1e-9);
}

#[test]
fn the_history_helper_is_the_ansatz_evaluated_at_two_levels() {
    let p = params();
    let w = solve_rotating_wave(&p).unwrap();
    let (u0, w0, v0, up, wp, vp) = rotating_wave_history(&w, p.fs).unwrap();
    assert_eq!(u0, w.phi);
    assert_eq!(v0, w.psi);
    assert_eq!(
        vp, w.psi,
        "psi is STATIC: the previous level is the same field"
    );
    assert!(w0.iter().all(|x| *x == 0.0), "w starts exactly at zero");
    assert!(rel(up[8], 0.003_535_484_195_020_008) < 1e-12);
    assert!(rel(wp[8], -7.876_694_636_817_168e-6) < 1e-9);
    assert!(
        rotating_wave_history(&w, 0.0).is_err(),
        "fs must be positive"
    );
}

#[test]
fn the_planar_hessian_matches_its_closed_form() {
    let p = [1e-3, 2e-3];
    let z = [1e-4, -2e-4];
    let a = 1.0e5 - 200.0;
    let (h_pp, h_pz, h_zz) = planar_hessian_cells(&p, &z, a);
    assert!(rel(h_pp[0], 10.128_657_011_739_28) < 1e-13);
    assert!(rel(h_pz[0], 99.779_893_353_652_92) < 1e-13);
    assert!(rel(h_zz[0], -0.099_769_916_362_016_72) < 1e-13);
    assert!(rel(h_pp[1], -19.364_836_371_781_603) < 1e-13);
    assert!(rel(h_pz[1], 199.678_665_405_824_7) < 1e-13);
    assert!(rel(h_zz[1], -0.399_437_218_255_300_43) < 1e-13);

    // `H_zz = -a p^2 / Lambda^3` EXACTLY, and it is the entry whose literal spelling cancels two
    // `O(1)` terms. Assert the identity rather than only the value: at rest it is exactly zero,
    // which the literal form cannot produce.
    let (_, _, rest) = planar_hessian_cells(&[0.0], &[0.0], a);
    assert_eq!(
        rest[0], 0.0,
        "no transverse strain, no longitudinal softening"
    );
}

#[test]
fn the_circular_kc_frequency_has_no_factor_of_a_half() {
    // A circular mode's stretch is `cos^2 + sin^2 = 1`: constant, and exactly twice the planar
    // mode's time-average. So the shift is the full `eps R^2`.
    let om = kc_circular_frequency(4.0e5, 1.0e7, 5e-3).unwrap();
    assert!(rel(om, 632.653_143_515_465_3) < 1e-13, "{om}");
    // At `eps = 0` there is no shift at all and the expression is exactly `sqrt(omega0^2)`.
    assert_eq!(
        kc_circular_frequency(4.0e5, 0.0, 5e-3).unwrap(),
        (4.0e5f64).sqrt()
    );
    assert!(kc_circular_frequency(-1.0, 0.0, 1.0).is_err());
}

// -- 2. the R -> 0 gate --------------------------------------------------------------------

#[test]
fn at_vanishing_amplitude_the_bvp_collapses_onto_the_linear_dispersion() {
    // The strongest claim in the module, and a free one: as `R -> 0` the helix is the linear mode
    // and `s` must be the theta-scheme's own `Q / (1 + theta k^2 Q)` -- a closed form written eight
    // models earlier that knows nothing about rotating waves. A transcription that dropped either
    // time factor fails here and nowhere else.
    for &mode in &[1usize, 2, 5] {
        for &kappa in &[0.0f64, 2.0] {
            let p = BvpParams {
                amplitude: 1e-12,
                mode,
                kappa,
                ..params()
            };
            let w = solve_rotating_wave(&p).unwrap();
            let h = p.l / p.n_cells as f64;
            let k = 1.0 / p.fs;
            let p2 = (4.0 / (h * h))
                * (mode as f64 * std::f64::consts::PI / (2 * p.n_cells) as f64)
                    .sin()
                    .powi(2);
            let q = (p.t / p.rho) * p2 + kappa * kappa * p2 * p2;
            let expected = q / (1.0 + p.theta * k * k * q);
            assert!(
                rel(w.s, expected) < 1e-12,
                "mode {mode}, kappa {kappa}: s = {} against the linear oracle {expected}",
                w.s
            );
        }
    }
}

// -- 3. the structural identities ----------------------------------------------------------

#[test]
fn a_sine_is_not_a_relative_equilibrium_but_the_solution_is() {
    // The converged helix is a *deformed* sine, and the deformation is the physics. Assert both
    // halves: it is close to a sine (so the continuation did not wander) and measurably not one.
    let w = solve_rotating_wave(&params()).unwrap();
    assert!(
        w.shape_residual > 1e-6,
        "a bug that returned the seed would give ~0"
    );
    assert!(
        w.shape_residual < 1e-3,
        "and it must still be a deformed sine"
    );
}

#[test]
fn the_shape_residual_grows_as_the_amplitude_squared() {
    // The deformation is driven by the `r^2 v_x / 2` term, so doubling `R` should roughly
    // quadruple it. A loose bar: this is a scaling claim, not a coefficient one.
    let small = solve_rotating_wave(&BvpParams {
        amplitude: 2.5e-3,
        ..params()
    })
    .unwrap();
    let big = solve_rotating_wave(&params()).unwrap();
    let ratio = big.shape_residual / small.shape_residual;
    assert!(
        (3.0..5.0).contains(&ratio),
        "shape residual scaled by {ratio}, expected ~4"
    );
}

#[test]
fn bad_parameters_are_rejected() {
    let bad = |f: &dyn Fn(&mut BvpParams)| {
        let mut p = params();
        f(&mut p);
        solve_rotating_wave(&p)
    };
    assert!(bad(&|p| p.rho = 0.0).is_err(), "rho must be positive");
    assert!(bad(&|p| p.n_cells = 1).is_err(), "N must be >= 2");
    assert!(bad(&|p| p.mode = 0).is_err(), "mode is 1-based");
    assert!(bad(&|p| p.mode = 32).is_err(), "mode must be < N");
    assert!(bad(&|p| p.kappa = -1.0).is_err(), "kappa must be >= 0");
    assert!(bad(&|p| p.theta = 0.0).is_err(), "theta must be in (0, 1]");
    assert!(bad(&|p| p.theta = 1.5).is_err(), "theta must be in (0, 1]");
    assert!(bad(&|p| p.ea = 100.0).is_err(), "softening is out of scope");
    assert!(bad(&|p| p.continuation_steps = 0).is_err());
    assert!(bad(&|p| p.maxiter = 0).is_err());
    assert!(bad(&|p| p.tol = 0.0).is_err());
    // `EA == T` is the linear anchor and is explicitly allowed -- the boundary, not past it.
    assert!(
        bad(&|p| p.ea = 200.0).is_ok(),
        "EA == T is the anchor, not an error"
    );
}

#[test]
fn a_rotation_past_nyquist_is_refused_rather_than_wrapped() {
    // `s` maps back to `Omega` through `arcsin`, which has nothing to say above 1. A wrapped
    // frequency would be a plausible-looking number for an impossible rotation.
    let p = BvpParams {
        fs: 700.0,
        n_cells: 4,
        mode: 1,
        amplitude: 1e-4,
        ..params()
    };
    let out = solve_rotating_wave(&p);
    assert!(
        out.is_err() || out.as_ref().unwrap().omega.is_finite(),
        "either refused, or a finite frequency -- never a NaN"
    );
}

// -- 4. carried from tests/test_geometric_rotating_wave.py (retirement plan §36) -----------------
//
// The bars that need only the BVP. The ones that spin a string are in
// `crates/physsynth-core/tests/string_geometric_helix.rs`. `params()` above IS the Python's
// `_string()` fixture; `at(ea, lam_long)` is that fixture rebuilt at another `EA` or `lam_long`,
// with the sample rate re-derived from the longitudinal Courant number the way the helper did.

fn at(ea: f64, lam_long: f64) -> BvpParams {
    let p = params();
    BvpParams {
        ea,
        fs: (ea / p.rho).sqrt() * p.n_cells as f64 / (p.l * lam_long),
        ..p
    }
}

/// `kc_mode_coefficients` with the discrete `p2`, under `EA -> EA - T`.
fn kc(p: &BvpParams, mode: usize) -> (f64, f64) {
    let h = p.l / p.n_cells as f64;
    let p2 = spatial_eigenvalue_p2(p.n_cells as i64, h, mode as i64);
    kc_mode_coefficients((p.t / p.rho).sqrt(), 0.0, p.ea - p.t, p.rho, p2, p.l).unwrap()
}

fn err_of(p: &BvpParams) -> String {
    solve_rotating_wave(p).expect_err("must be refused")
}

#[test]
fn at_zero_amplitude_the_helix_is_the_linear_modal_oracle() {
    // Carried from `test_rotating_wave_at_zero_amplitude_is_the_linear_modal_oracle`. Distinct from
    // the R -> 0 gate above in the one way that matters: that one checks `s` against the closed
    // form written out in the test, this one checks the reported FREQUENCY against
    // `modal::discrete_stiff_mode_frequency`, independent code from eight models earlier -- so it
    // also covers the `arcsin` that turns `s` back into Hz.
    for &kappa in &[0.0f64, 2.0] {
        for &mode in &[1usize, 2, 5] {
            let p = BvpParams {
                amplitude: 1e-9,
                mode,
                kappa,
                ..params()
            };
            let w = solve_rotating_wave(&p).unwrap();
            let c = (p.t / p.rho).sqrt();
            let want = discrete_stiff_mode_frequency(
                c,
                p.l,
                p.n_cells as i64,
                kappa,
                1.0 / p.fs,
                mode as i64,
                p.theta,
            );
            let err = rel(w.frequency, want);
            println!("kappa {kappa}, mode {mode}: {err:.3e}");
            assert!(w.converged);
            assert!(err < 1e-12, "kappa {kappa}, mode {mode}: {err:.3e}");
        }
    }
}

#[test]
fn the_kirchhoff_carrier_error_is_the_mode_shape_deformation() {
    // Carried from `test_kc_circular_frequency_error_is_the_mode_shape_deformation`. KC assumes a
    // sine; a spun helix is stretched most at the nodes, so the tension is non-uniform and the true
    // shape is a deformed sine. The frequency error tracks that deformation ~4/3 : 1 whatever
    // `EA/T` or the mode -- asserted as a BAND, because the constant is measured, not derived.
    // Semi-discrete on purpose: the time-discrete Omega also carries the theta-scheme's temporal
    // dispersion, which has nothing to do with the shape and would swamp it.
    for (ea_over_t, mode) in [(50.0, 1usize), (500.0, 1), (500.0, 2)] {
        let p = BvpParams {
            time_discrete: false,
            mode,
            ..at(ea_over_t * 200.0, 0.5)
        };
        let (omega0_sq, eps) = kc(&p, mode);
        let wave = solve_rotating_wave(&BvpParams {
            amplitude: 1e-3,
            ..p
        })
        .unwrap();
        let kc_om = kc_circular_frequency(omega0_sq, eps, 1e-3).unwrap();
        let err = (wave.omega - kc_om) / kc_om;
        let tiny = solve_rotating_wave(&BvpParams {
            amplitude: 1e-6,
            ..p
        })
        .unwrap();
        let tiny_err = rel(
            tiny.omega,
            kc_circular_frequency(omega0_sq, eps, 1e-6).unwrap(),
        );
        let ratio = err / wave.shape_residual;
        println!(
            "EA/T {ea_over_t}, mode {mode}: err {err:.4e}, ratio {ratio:.4}, tiny {tiny_err:.3e}"
        );
        // Positive: tension peaks where phi' does, so the Rayleigh quotient beats the
        // uniform-tension estimate. Negative would mean the deformation was read backwards.
        assert!(err > 0.0, "EA/T {ea_over_t}, mode {mode}: {err:.3e}");
        assert!(
            1.2 < ratio && ratio < 1.45,
            "EA/T {ea_over_t}, mode {mode}: {ratio}"
        );
        // ...and KC is exact in the limit it is the limit of.
        assert!(
            tiny_err < 1e-9,
            "EA/T {ea_over_t}, mode {mode}: {tiny_err:.3e}"
        );
    }
}

#[test]
fn the_shape_residual_quadruples_per_doubling_over_four_amplitudes() {
    // Carried from `test_shape_residual_scales_as_amplitude_squared`: 4.000x per doubling, measured
    // rather than Taylor-asserted. An under-converged Newton would not land on a clean power law.
    let mut res = Vec::new();
    for amp in [5e-4, 1e-3, 2e-3, 4e-3] {
        let w = solve_rotating_wave(&BvpParams {
            amplitude: amp,
            time_discrete: false,
            ..params()
        })
        .unwrap();
        assert!(w.converged);
        res.push(w.shape_residual);
    }
    for pair in res.windows(2) {
        let ratio = pair[1] / pair[0];
        println!("doubling: {ratio:.7}");
        assert!((ratio - 4.0).abs() < 0.05, "{ratio}");
    }
}

#[test]
fn the_tension_field_peaks_at_the_ends_and_its_spread_grows_as_r_squared() {
    // Carried from `test_the_tension_field_is_non_uniform_and_that_is_the_mechanism`. Model #9
    // collapses the tension to one number; here it is a field, peaked at the ENDS (where phi' is
    // largest) rather than where the displacement is. Barely: ~2e-5 of T at R = 2e-3, which is why
    // KC is a good oracle and still a wrong one.
    let p = params();
    let mut spreads = Vec::new();
    let mut stretched = true;
    for amp in [1e-3, 2e-3] {
        let w = solve_rotating_wave(&BvpParams {
            amplitude: amp,
            ..p
        })
        .unwrap();
        let t = &w.tension;
        let lo = t.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!("R {amp:e}: ends {:.9}, middle {:.9}", t[0], t[t.len() / 2]);
        assert!(
            t.iter().all(|x| *x > 0.0),
            "a hardening helix never goes slack"
        );
        assert!(t[0] > t[t.len() / 2], "peaked at the ends, not the middle");
        spreads.push((hi - lo) / p.t);
        // The Python read the LAST loop's stretch ratio; every cell of it must be stretched.
        stretched = w.stretch_ratio.iter().all(|l| *l >= 1.0);
    }
    let ratio = spreads[1] / spreads[0];
    println!("spreads {spreads:?}, ratio {ratio:.6}");
    assert!(spreads[0] > 1e-6, "{:.3e}", spreads[0]);
    // O(R^2) with a visible O(R^4) correction (4.004 .. 4.232 over R = 5e-4 .. 8e-3), so the bar
    // sits where the leading order is still clean.
    assert!((ratio - 4.0).abs() < 0.05, "{ratio}");
    // ...and Lambda >= 1, the energy floor's premise.
    assert!(stretched, "every cell of the helix must be stretched");
}

#[test]
fn time_discrete_and_semi_discrete_differ_by_the_temporal_dispersion_only() {
    // Carried from `test_time_discrete_and_semi_discrete_differ_by_temporal_dispersion`: halving k
    // must quarter the gap. If it did not, one of the two time factors would be wrong rather than
    // merely time-discrete -- and that licenses semi-discrete for physics, discrete for seeding.
    let mut gaps = Vec::new();
    for lam_long in [0.5, 0.25] {
        let p = BvpParams {
            amplitude: 5e-3,
            ..at(params().ea, lam_long)
        };
        let d = solve_rotating_wave(&p).unwrap();
        let semi = solve_rotating_wave(&BvpParams {
            time_discrete: false,
            ..p
        })
        .unwrap();
        gaps.push((d.omega - semi.omega).abs() / semi.omega);
    }
    let ratio = gaps[0] / gaps[1];
    println!("gaps {gaps:?}, ratio {ratio:.7}");
    assert!(gaps[0] > 0.0);
    assert!((ratio - 4.0).abs() < 0.3, "{ratio}");
}

#[test]
fn the_pythons_refusals_name_what_they_refuse() {
    // Carried from `test_softening_string_is_rejected` and `test_bad_parameters_are_rejected` (11
    // cases): the message fragment the Python matched, not merely "some error". A negative `N` or
    // `mode` cannot be spelled as a `usize`, so those refusals are types natively; every value the
    // Python passed here is representable and is refused by the solver itself.
    let p = BvpParams {
        amplitude: 1e-3,
        ..params()
    };
    let soft = err_of(&BvpParams { ea: 0.5 * p.t, ..p });
    assert!(soft.contains("softening"), "{soft}");
    let cases: [(&str, BvpParams); 11] = [
        ("positive", BvpParams { l: -1.0, ..p }),
        ("positive", BvpParams { rho: 0.0, ..p }),
        ("N must be", BvpParams { n_cells: 1, ..p }),
        ("mode must be", BvpParams { mode: 0, ..p }),
        ("mode must be", BvpParams { mode: 99, ..p }),
        ("kappa", BvpParams { kappa: -1.0, ..p }),
        ("theta", BvpParams { theta: 0.0, ..p }),
        ("theta", BvpParams { theta: 1.5, ..p }),
        (
            "continuation_steps",
            BvpParams {
                continuation_steps: 0,
                ..p
            },
        ),
        ("maxiter", BvpParams { maxiter: 0, ..p }),
        ("tol", BvpParams { tol: 0.0, ..p }),
    ];
    for (fragment, bad) in cases {
        let msg = err_of(&bad);
        assert!(msg.contains(fragment), "expected {fragment:?} in {msg:?}");
    }
    // Carried from `test_history_helper_requires_a_positive_rate`.
    let wave = solve_rotating_wave(&p).unwrap();
    let msg = rotating_wave_history(&wave, 0.0).expect_err("a wrong fs is a wrong helix");
    assert!(msg.contains("fs"), "{msg}");
}

#[test]
fn a_starved_continuation_reports_that_it_did_not_converge() {
    // Carried from `test_non_convergence_warns_and_is_reported`: one continuation step straight to
    // a large amplitude with one iteration. The RuntimeWarning's prose lives in the binding and
    // goes with it; what the solver owes a caller is the flag, so that a shape which merely came
    // back is never mistaken for a helix.
    let wave = solve_rotating_wave(&BvpParams {
        amplitude: 0.05,
        continuation_steps: 1,
        maxiter: 1,
        ..params()
    })
    .expect("a stall is reported, not refused");
    assert!(!wave.converged);
}
