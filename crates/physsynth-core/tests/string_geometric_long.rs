//! The geometrically exact string's validation harness, second half — the long simulations,
//! carried from `tests/test_geometric_energy.py` and `tests/test_geometric_limits.py` (retirement
//! plan §35) at the same fixture as `string_geometric_harness.rs`, whose header describes it.
//!
//! Split out so that it can run **optimised only** in CI (the human's call, §35.7): together these
//! are most of the batch's cost unoptimised (one run is 60,000 Newton steps), and none of their
//! exact checks pins a spelling the release profile could fold differently — they are frequency,
//! convergence-rate and drift bars against tolerances.
//!
//! - The linear limit: three decoupled waves at vanishing amplitude.
//! - A softening string (`EA < T`) conserves, stays positive and cannot go slack.
//! - The under-resolved band, where the Newton solve stalls and the energy still conserves.
//! - Richardson self-convergence, the amplitude shift against model #9's Duffing oracle, and model
//!   #9 itself as the Kirchhoff–Carrier limit.

mod geometric_fixture;

use geometric_fixture::*;
use physsynth_analysis::damping::spatial_eigenvalue_p2;
use physsynth_analysis::duffing::{duffing_frequency_shift, kc_mode_coefficients};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::string_nonlinear as nl;
use std::f64::consts::PI;

#[test]
fn at_vanishing_amplitude_the_model_is_three_linear_waves() {
    // Carried from `test_small_amplitude_recovers_three_linear_waves`: transverse at `n c/(2L)` in
    // BOTH polarizations, longitudinal at `n c_long/(2L)`. The timestep is set from `lam_long`:
    // the longitudinal field runs ~22x faster, and at transverse Courant the measurement would
    // report the theta-scheme's temporal dispersion, not the physics.
    let n = 64;
    let mut s = Geo {
        lam_long: Some(0.7),
        kappa: 0.0,
        ..Geo::new(n)
    }
    .build();
    let amp = 1e-9; // the nonlinearity is cubic, so it vanishes here
    let (u0, w0, v0) = (
        mode_ic(&s, 1, amp),
        mode_ic(&s, 2, amp),
        mode_ic(&s, 3, amp),
    );
    start(&mut s, &u0, &w0, &v0);

    let probe = n as usize / 3;
    let (mut pu, mut pw, mut pv) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..60_000 {
        step(&mut s);
        pu.push(s.u[probe]);
        pw.push(s.w[probe]);
        pv.push(s.v[probe]);
    }
    let c_long = (EA / RHO).sqrt();
    for (signal, f_exact, name) in [
        (&pu, c() / (2.0 * L), "u mode 1"),
        (&pw, 2.0 * c() / (2.0 * L), "w mode 2"),
        (&pv, 3.0 * c_long / (2.0 * L), "v mode 3"),
    ] {
        let got = measure_partials_near(signal, s.p.fs, &[f_exact], None)[0];
        let err = (got - f_exact).abs() / f_exact;
        println!("{name}: {got:.6} vs {f_exact:.6} ({err:.3e})");
        assert!(err <= 2e-3, "{name}: {got:.2} vs {f_exact:.2}");
    }
}

#[test]
fn a_softening_string_conserves_stays_positive_and_cannot_go_slack() {
    // Carried from `test_softening_is_hyperreal_not_unstable` (three EAs). The potential is exact
    // for either sign of `a`, `E >= 0` survives by the same Jensen step, and `tension = EA Λ + |a|
    // > 0` for every `Λ > 0`. Below the anchor the TRANSVERSE wave is the fast one, so `lam` sets
    // the timestep.
    for ea in [T * 0.5, T * 0.1, T / 200.0] {
        let mut s = Geo {
            lam: Some(0.5),
            ea,
            allow_softening: true,
            ..Geo::new(32)
        }
        .build();
        assert!(
            s.p.a < 0.0 && s.p.c_long < s.p.c,
            "below the anchor transverse is fast"
        );
        let u0 = pluck_ic(&s, 2e-3);
        start_u(&mut s, &u0);
        let e0 = s.energy();
        let (mut lo, mut hi) = (e0, e0);
        let mut min_tension = s.tension().iter().copied().fold(f64::INFINITY, f64::min);
        for _ in 0..3000 {
            step(&mut s);
            let e = s.energy();
            lo = lo.min(e);
            hi = hi.max(e);
            for t in s.tension() {
                min_tension = min_tension.min(t);
            }
        }
        let drift = (hi - lo) / e0.abs();
        println!("EA {ea}: drift {drift:.3e}, min E {lo:.4e}, min tension {min_tension:.4}");
        assert!(
            drift < DRIFT_GATE,
            "softening drifts {drift:.2e} — V IS bounded below"
        );
        assert!(lo > 0.0, "E >= 0 holds at a < 0 too; got {lo:.3e}");
        assert!(
            min_tension > 0.0,
            "a softening string cannot go slack; got {min_tension:.3}"
        );
    }
}

#[test]
fn a_flat_energy_is_not_a_convergence_certificate_in_the_under_resolved_band() {
    // Carried from
    // `test_a_flat_energy_is_not_a_convergence_certificate_in_the_under_resolved_band`. Two
    // `lam_long` edges a factor of two apart: the convergence edge (~4) and the energy edge (5-10).
    // Between them the solve stalls and the energy still conserves, so anything asserting
    // convergence must read `n_not_converged`, not the energy.
    let trajectory = |lam_long: f64| {
        let mut s = Geo {
            lam_long: Some(lam_long),
            ..Geo::new(32)
        }
        .build();
        let u0 = mode_ic(&s, 3, 4e-3);
        start_u(&mut s, &u0);
        let e0 = s.energy();
        let steps = (0.004 * s.p.fs).round() as usize;
        step_n(&mut s, steps);
        (rel_drift(s.energy(), e0), s.n_not_converged)
    };
    let (drift_ok, stalled_ok) = trajectory(2.0);
    println!("lam_long 2: drift {drift_ok:.3e}, {stalled_ok} stalled");
    assert_eq!(stalled_ok, 0, "lam_long = 2 should converge on every step");
    assert!(drift_ok < DRIFT_GATE);

    let (drift_band, stalled_band) = trajectory(6.0);
    println!("lam_long 6: drift {drift_band:.3e}, {stalled_band} stalled");
    assert!(
        stalled_band > 0,
        "lam_long = 6 sits above the convergence edge — if nothing stalls the edge has moved; \
         re-run `cargo run --release -p physsynth-core --example geometric_lam_long`"
    );
    assert!(
        drift_band < DRIFT_GATE,
        "the energy gate passes while {stalled_band} steps stalled"
    );
}

// == the limits
// ====================================================================================

/// The Python's `_mode_frequency`: the **nonlinear** frequency (Hz) from six descending zero
/// crossings of a modal projection, each linearly interpolated, as one over the mean spacing.
/// `advance` steps the model and returns the new projection.
fn crossing_frequency(start: f64, k: f64, mut advance: impl FnMut() -> f64) -> f64 {
    let mut prev = start;
    let mut times = Vec::new();
    for n in 1..=2_000_000usize {
        let cur = advance();
        if prev > 0.0 && 0.0 >= cur {
            times.push(((n - 1) as f64 + prev / (prev - cur)) * k);
            if times.len() >= 6 {
                break;
            }
        }
        prev = cur;
    }
    assert!(
        times.len() >= 2,
        "too few zero crossings to measure a frequency"
    );
    let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    1.0 / (gaps.iter().sum::<f64>() / gaps.len() as f64)
}

/// A model #10 string started on mode 1 at amplitude `amp`, and its measured frequency.
fn geometric_mode_one_frequency(amp: f64) -> f64 {
    let mut s = Geo::new(32).build();
    let shape = mode_ic(&s, 1, 1.0);
    let u0: Vec<f64> = shape.iter().map(|x| amp * x).collect();
    start_u(&mut s, &u0);
    let denom = dot(&shape, &shape);
    let first = dot(&s.u, &shape) / denom;
    let k = s.p.k;
    crossing_frequency(first, k, || {
        step(&mut s);
        dot(&s.u, &shape) / denom
    })
}

#[test]
fn the_scheme_self_converges_at_second_order() {
    // Carried from `test_richardson_second_order_self_convergence`. SELF-convergence, because
    // Duffing is only a LIMIT for this model: its error plateaus at the phantom leakage and would
    // read as a decaying order. `h` and `k` refined together (`lam_long` fixed), so every grid
    // lands on the identical physical time; a smooth two-mode start, because a pluck's corner is
    // O(h).
    let (m, amp) = (1500usize, 3e-3);
    let state_after = |n: i64| {
        let mut s = Geo::new(n).build();
        let ic: Vec<f64> =
            s.p.grid()
                .iter()
                .map(|&x| amp * (((3.0 * PI) * x / L).sin() + 0.5 * ((4.0 * PI) * x / L).sin()))
                .collect();
        start_u(&mut s, &ic);
        step_n(&mut s, m * (n as usize / 32));
        assert!(s.converged && s.n_not_converged == 0);
        let nl_frac = (s.nonlinear_energy() / s.energy()).abs();
        (s.u[(0.25 * n as f64).round() as usize], nl_frac)
    };
    let runs: Vec<(f64, f64)> = [32, 64, 128].into_iter().map(state_after).collect();
    // The nonlinearity must actually be engaged, or this is a linear convergence test.
    for (n, (_, frac)) in [32, 64, 128].iter().zip(&runs) {
        println!("N={n}: nonlinear fraction {frac:.3e}");
        assert!(
            *frac > 1e-3,
            "N={n}: nonlinear fraction {frac:.2e} too small"
        );
    }
    let e1 = (runs[0].0 - runs[1].0).abs();
    let e2 = (runs[1].0 - runs[2].0).abs();
    let ratio = e1 / e2;
    println!("ratio {ratio:.6} (order {:.4})", ratio.log2());
    assert!(ratio > 3.4, "ratio {ratio:.2} — not second order");
    assert!(
        ratio < 5.2,
        "ratio {ratio:.2} — suspiciously fast, check the setup"
    );
}

#[test]
fn the_amplitude_shift_tracks_the_duffing_limit() {
    // Carried from `test_amplitude_shift_tracks_the_duffing_limit` (two amplitudes). The SHIFT, not
    // an absolute frequency: `omega(A→0)` carries the same temporal dispersion and cancels it. The
    // oracle is model #9's with `EA -> a = EA − T0`, and the bar is loose on purpose — a LIMIT
    // oracle, since model #10 leaks motion into the longitudinal field that KC has nowhere to put.
    let a = Geo::new(32).params().unwrap().a;
    let p2 = spatial_eigenvalue_p2(32, L / 32.0, 1);
    let (w0sq, eps) = kc_mode_coefficients(c(), KAPPA, a, RHO, p2, L).expect("physical");
    let f_linear = geometric_mode_one_frequency(1e-6);
    for amp in [0.004, 0.008] {
        let measured = geometric_mode_one_frequency(amp) - f_linear;
        let oracle = duffing_frequency_shift(amp, w0sq, eps).expect("a valid oracle") / (2.0 * PI);
        let err = (measured - oracle).abs() / oracle;
        println!("A={amp}: shift {measured:.6} Hz vs Duffing {oracle:.6} Hz ({err:.3e})");
        // Not vacuity theatre: a linear string's zero shift would pass a bare relative bar.
        assert!(
            oracle > 0.5,
            "the shift should be a real, audible number of Hz"
        );
        assert!(err <= 0.05, "A={amp}: {measured} vs {oracle}");
    }
}

/// A model #9 string at the identification `EA_#9 = (EA − T0)_#10`, started on mode 1 at `amp`
/// — `make_tension_string(N=32, EA=geo._a, lam=0.5)` — and its measured frequency.
fn tension_mode_one_frequency(amp: f64) -> f64 {
    let a = Geo::new(32).params().unwrap().a;
    let fs = c() * 32.0 / (L * 0.5);
    let kp = nl::Params::new(
        L,
        T,
        RHO,
        fs,
        32,
        KAPPA,
        a,
        0.0,
        0.0,
        THETA,
        TENSION_TOL,
        true,
    )
    .expect("the model #9 twin must construct");
    let shape: Vec<f64> = kp
        .grid()
        .iter()
        .map(|&x| ((1.0 * PI) * x / L).sin())
        .collect();
    let mut kc = nl::TensionModulatedString::new(kp);
    let u0: Vec<f64> = shape.iter().map(|x| amp * x).collect();
    kc.set_state(&u0, &vec![0.0; u0.len()]);
    let denom = dot(&shape, &shape);
    let first = dot(&kc.u, &shape) / denom;
    let k = kc.p.k;
    crossing_frequency(first, k, || {
        kc.step().expect("the tension solve must not error");
        dot(&kc.u, &shape) / denom
    })
}

#[test]
fn model_nine_is_the_kirchhoff_carrier_limit_of_model_ten() {
    // Carried from `test_model_9_is_the_kc_limit_of_model_10`: quasi-static longitudinal and small
    // slopes make model #9 an oracle for model #10's transverse limit, with `EA_#9 = (EA −
    // T0)_#10`.
    //
    // **Compared as pitch RISE, not raw pitch** (§35.4, the human's call). The Python compared the
    // two absolute frequencies within 2%, and at this amplitude the rise is 0.18% of the pitch, so
    // a LINEAR model #9 (its tension coefficient zeroed) sat 0.21% off and passed: the bar could
    // not see the nonlinearity it was written for. Each model's rise from its own quiet frequency
    // is compared instead, at the Python's 2%; a linear twin has no rise and fails outright.
    let amp = 2e-3;
    let rise_geo = geometric_mode_one_frequency(amp) - geometric_mode_one_frequency(1e-6);
    let rise_kc = tension_mode_one_frequency(amp) - tension_mode_one_frequency(1e-6);
    let err = (rise_geo - rise_kc).abs() / rise_kc;
    println!("rise: model #10 {rise_geo:.6} Hz, model #9 {rise_kc:.6} Hz ({err:.3e})");
    assert!(
        rise_kc > 0.1,
        "model #9 must actually harden here: {rise_kc} Hz"
    );
    assert!(
        err <= 0.02,
        "model #10's rise {rise_geo} Hz should sit on model #9's {rise_kc} Hz"
    );
}
