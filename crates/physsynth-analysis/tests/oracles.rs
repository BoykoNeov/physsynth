//! `damping`, `dispersion` and `duffing` — their own bars.
//!
//! Three small modules with one property each that a parity test could not establish: the decay
//! oracle must return *no decay* for a lossless string, the dispersion curve must be flat when the
//! scheme is exact, and the Duffing solution must actually solve the Duffing equation.

use physsynth_analysis::bessel::j1;
use physsynth_analysis::damping::{
    discrete_damped_mode_decay, discrete_damped_mode_is_underdamped, discrete_damped_mode_rate,
    loss_coefficients_from_t60, modal_loss_rate_continuum, spatial_eigenvalue_p2,
    t60_seconds_per_rate,
};
use physsynth_analysis::dispersion::{
    dispersion_frequencies, phase_velocity, stiff_dispersion_frequencies,
};
use physsynth_analysis::duffing::{
    duffing_displacement, duffing_elliptic_parameter, duffing_frequency,
    duffing_frequency_expansion, duffing_frequency_shift, kc_mode_coefficients, kc_mode_stretch,
};
use physsynth_analysis::horizon::pitch_horizon;
use physsynth_analysis::modal::{
    discrete_mode_frequency, discrete_stiff_mode_frequency, harmonic_frequencies, inharmonicity_b,
    stiff_harmonic_frequencies,
};
use physsynth_analysis::radiation::{
    piston_radiation_resistance, C0_AIR, PISTON_SERIES_CUTOFF_KA, RHO0_AIR,
};

// -- damping ---------------------------------------------------------------------------------

#[test]
fn a_lossless_string_does_not_decay_at_all() {
    // With both sigmas zero, `a` and `c` are the same expression and `g = c/a` must be exactly 1 —
    // not 1 within a tolerance. Anything else is a loss term that exists when it should not, which
    // is the single most consequential bug this oracle could have: it would make a *correct*
    // lossless simulation look like it was leaking energy.
    for &m in &[1i64, 3, 17] {
        let g = discrete_damped_mode_decay(200.0, 0.65, 400, 0.0, 1e-5, 0.5, 0.0, 0.0, m);
        assert_eq!(g, 1.0, "lossless decay factor for mode {m}");
        assert_eq!(
            discrete_damped_mode_rate(200.0, 0.65, 400, 0.0, 1e-5, 0.5, 0.0, 0.0, m),
            0.0,
            "a lossless mode has a zero rate, not a small one"
        );
    }
}

#[test]
fn the_discrete_decay_rate_converges_to_the_continuum_one_first_order_in_k() {
    // Measured: 7.0e-5 → 1.8e-5 → 4.4e-6 → 1.1e-6 as the grid doubles, so the ratio is 4 and the
    // convergence is second order in h (the eigenvalue's error), not first in k.
    let (c, l, kappa, theta, s0, s1, m) = (200.0, 0.65, 0.0, 0.5, 0.5, 1e-5, 3i64);
    let cont = modal_loss_rate_continuum(c, l, kappa, s0, s1, m);
    let mut prev: Option<f64> = None;
    for &n in &[200i64, 400, 800, 1600] {
        let k = 1.0 / (4.0 * n as f64 * c / l);
        let e = (discrete_damped_mode_rate(c, l, n, kappa, k, theta, s0, s1, m) / cont - 1.0).abs();
        assert!(e < 1e-4, "not near the continuum rate at N={n}");
        if let Some(p) = prev {
            assert!(
                (p / e - 4.0).abs() < 0.2,
                "the error fell by {} rather than 4 at N={n}",
                p / e
            );
        }
        prev = Some(e);
    }
}

#[test]
fn the_underdamped_predicate_separates_the_two_regimes() {
    let (c, l, n, k, theta) = (200.0, 0.65, 400i64, 1e-5, 0.5);
    assert!(
        discrete_damped_mode_is_underdamped(c, l, n, 0.0, k, theta, 0.5, 1e-5, 3),
        "a lightly damped mode oscillates"
    );
    assert!(
        !discrete_damped_mode_is_underdamped(c, l, n, 0.0, k, theta, 1e6, 0.0, 3),
        "a mode drowned in sigma0 does not"
    );
}

#[test]
fn the_t60_inversion_is_the_inverse_of_the_thing_it_inverts() {
    // Round-trip: pick sigmas, compute what T60 they imply at two frequencies through the same
    // continuum relation the solver uses, and require the solve to hand them back.
    let (c, l, kappa) = (200.0, 0.65, 0.0);
    let (s0, s1) = (1.2, 4e-3);
    let beta2 = |f: f64| (2.0 * std::f64::consts::PI * f).powi(2) / (c * c);
    let t60 = |f: f64| t60_seconds_per_rate() / (s0 + s1 * beta2(f));
    let (g0, g1) = loss_coefficients_from_t60(c, l, kappa, 100.0, t60(100.0), 1000.0, t60(1000.0))
        .expect("a decreasing T60 pair is solvable");
    assert!((g0 - s0).abs() < 1e-10, "sigma0 {g0} != {s0}");
    assert!((g1 - s1).abs() < 1e-13, "sigma1 {g1} != {s1}");
    // And the refusals: equal frequencies cannot separate the two, and a T60 rising with frequency
    // asks for negative loss, which no passive string can do.
    assert!(loss_coefficients_from_t60(c, l, kappa, 100.0, 4.0, 100.0, 1.0).is_err());
    assert!(loss_coefficients_from_t60(c, l, kappa, 100.0, 1.0, 1000.0, 4.0).is_err());
    assert!(loss_coefficients_from_t60(c, l, kappa, -1.0, 4.0, 1000.0, 1.0).is_err());
}

#[test]
fn the_spatial_eigenvalue_reaches_its_continuum_value() {
    // p² = (4/h²)sin²(mπ/2N) → (mπ/L)² as h → 0. It is the one number every damping formula here
    // is built on, so it gets its own bar.
    let l = 0.65;
    for &n in &[100i64, 1000, 10000] {
        let h = l / n as f64;
        let want = (3.0 * std::f64::consts::PI / l).powi(2);
        let got = spatial_eigenvalue_p2(n, h, 3);
        assert!((got / want - 1.0).abs() < 20.0 / (n * n) as f64);
    }
}

// The first five below are carried from `tests/test_damped_string.py` (retirement plan §33), at
// the damped-string harness's own numbers — the stiff string's c = 200 m/s, L = 1, kappa = 2 and
// theta = 0.28, declared with the stiff-string bars further down. The bars above make the same
// kinds of claim at L = 0.65, kappa = 0 and theta = 0.5, which is not the Python's case, and never
// reach the T60 mapping's stiff branch. The sixth was added in §33.4 and says so.

#[test]
fn the_continuum_loss_rate_is_two_sigma_eff_and_rises_only_with_sigma1() {
    // Carried from `test_continuum_rate_is_two_sigma_eff`. `2(sigma0 + sigma1 beta^2)` with
    // `beta = m pi / L`, written here; then frequency-independent (identical at every mode) when
    // sigma1 = 0, and rising with mode number when sigma1 > 0.
    let (c, l, kappa) = (STIFF_C, STIFF_L, STIFF_KAPPA);
    let (s0, s1) = (1.5, 3e-4);
    for m in [1i64, 3, 7] {
        let beta2 = (m as f64 * std::f64::consts::PI / l).powi(2);
        let want = 2.0 * (s0 + s1 * beta2);
        let got = modal_loss_rate_continuum(c, l, kappa, s0, s1, m);
        assert!(
            ((got - want) / want).abs() <= 1e-12,
            "mode {m}: {got} != {want}"
        );
    }
    let flat: Vec<f64> = [1i64, 5, 10]
        .iter()
        .map(|&m| modal_loss_rate_continuum(c, l, 0.0, 2.0, 0.0, m))
        .collect();
    assert!(flat[0] == flat[1] && flat[1] == flat[2], "{flat:?}");
    let rising: Vec<f64> = [1i64, 5, 10]
        .iter()
        .map(|&m| modal_loss_rate_continuum(c, l, 0.0, 2.0, 1e-3, m))
        .collect();
    assert!(rising[0] < rising[1] && rising[1] < rising[2], "{rising:?}");
}

#[test]
fn the_discrete_loss_rate_tends_to_the_continuum_one_at_the_harness_parameters() {
    // Carried from `test_discrete_rate_tends_to_continuum_on_refinement`: mode 4 at lam = 1
    // (`k = L / (c N)`), N = 256, 512, 1024 — the error falls monotonically and is inside 0.1% on
    // the finest grid.
    let (c, l, kappa, m) = (STIFF_C, STIFF_L, STIFF_KAPPA, 4i64);
    let (s0, s1) = (2.0, 5e-4);
    let cont = modal_loss_rate_continuum(c, l, kappa, s0, s1, m);
    let errs: Vec<f64> = [256i64, 512, 1024]
        .iter()
        .map(|&n| {
            let k = l / (c * n as f64);
            (discrete_damped_mode_rate(c, l, n, kappa, k, STIFF_THETA, s0, s1, m) - cont).abs()
        })
        .collect();
    println!("errors {errs:?}, finest relative {:e}", errs[2] / cont);
    assert!(
        errs[0] > errs[1] && errs[1] > errs[2],
        "not converging to the continuum: {errs:?}"
    );
    assert!(
        errs[2] / cont < 1e-3,
        "{:e} on the finest grid",
        errs[2] / cont
    );
}

#[test]
fn a_lossless_stiff_string_does_not_decay_at_the_harness_parameters() {
    // Carried from `test_decay_factor_lossless_is_unity`: N = 128, lam = 1, modes 1, 10 and 50.
    // The Python allowed `abs = 1e-15` on `g`; this asserts it EXACTLY, as the kappa = 0 bar above
    // does — with both sigmas zero, `a` and `c` are the same expression (`base +- 0 k`).
    let (c, l, n, kappa) = (STIFF_C, STIFF_L, 128i64, STIFF_KAPPA);
    let k = l / (c * n as f64);
    for m in [1i64, 10, 50] {
        let g = discrete_damped_mode_decay(c, l, n, kappa, k, STIFF_THETA, 0.0, 0.0, m);
        assert_eq!(g, 1.0, "mode {m}");
        let rate = discrete_damped_mode_rate(c, l, n, kappa, k, STIFF_THETA, 0.0, 0.0, m);
        assert_eq!(rate, 0.0, "mode {m}");
    }
}

#[test]
fn the_t60_inversion_round_trips_through_the_stiff_dispersion() {
    // Carried from `test_loss_coefficients_from_T60_pure_roundtrip`, at kappa = 2: the STIFF
    // branch of `beta^2(omega)`, which the kappa = 0 round trip above never takes. The forward map
    // is the Python's own `t60_at`, written here from the continuum dispersion
    // `kappa^2 beta^4 + c^2 beta^2 = omega^2` rather than read from the implementation, so a
    // wrong branch is not inverted by itself.
    let (c, l, kappa) = (STIFF_C, STIFF_L, STIFF_KAPPA);
    let (s0, s1) = (1.3, 8e-4);
    let t60_at = |f: f64| {
        let omega2 = (2.0 * std::f64::consts::PI * f).powi(2);
        let beta2 =
            (-(c * c) + (c.powi(4) + 4.0 * kappa * kappa * omega2).sqrt()) / (2.0 * kappa * kappa);
        t60_seconds_per_rate() / (s0 + s1 * beta2)
    };
    let (f1, f2) = (120.0, 1800.0);
    let (g0, g1) = loss_coefficients_from_t60(c, l, kappa, f1, t60_at(f1), f2, t60_at(f2))
        .expect("a decreasing T60 pair is solvable");
    println!("sigma0 {g0:?}, sigma1 {g1:?}");
    assert!(((g0 - s0) / s0).abs() <= 1e-10, "sigma0 {g0} != {s0}");
    assert!(((g1 - s1) / s1).abs() <= 1e-10, "sigma1 {g1} != {s1}");
}

#[test]
fn the_t60_inversion_refuses_a_rising_t60_and_a_single_frequency_for_their_own_reasons() {
    // Carried from `test_loss_coefficients_from_T60_rejects_increasing_T60`, at kappa = 2. The
    // Python asked only for a `ValueError`; the two refusals are told apart here by their prose
    // (not by the numbers the first one formats).
    let (c, l, kappa) = (STIFF_C, STIFF_L, STIFF_KAPPA);
    let rising = loss_coefficients_from_t60(c, l, kappa, 100.0, 1.0, 2000.0, 5.0)
        .expect_err("T60 rising with frequency asks for negative loss");
    assert!(rising.contains("negative loss"), "{rising}");
    let single = loss_coefficients_from_t60(c, l, kappa, 100.0, 1.0, 100.0, 2.0)
        .expect_err("one frequency cannot separate sigma0 from sigma1");
    assert!(single.contains("two distinct frequencies"), "{single}");
}

#[test]
fn the_t60_inversion_lands_on_the_pythons_recorded_answer() {
    // NOT carried from `test_damped_string.py`: added in §33.4 (the human's call). The two round
    // trips above spell the forward map with `t60_seconds_per_rate()` too, so a constant moved by
    // 1% cancels out of both, and the harness's simulated T60 bar is 4%; a 1% move was seen by
    // nothing native in the workspace. This is the row `tests/analysis_frozen_values.py` recorded
    // from the Python implementation before it was deleted (gap 0.0 at generation), at the frozen
    // case's own arguments, against that file's 1e-13 relative bar.
    let (g0, g1) = loss_coefficients_from_t60(200.0, 0.65, 0.7, 200.0, 6.0, 2000.0, 1.5)
        .expect("the frozen case is solvable");
    println!("sigma0 {g0:?}, sigma1 {g1:?}");
    for (got, want) in [(g0, 1.1147930129676977), (g1, 0.0009249908881590224)] {
        let rel = ((got - want) / want).abs();
        assert!(rel <= 1e-13, "{got:?} vs the Python's {want:?} ({rel:e})");
    }
}

// -- dispersion ------------------------------------------------------------------------------

#[test]
fn dispersion_delegates_exactly_and_flattens_at_lambda_one() {
    let (c, l, n) = (200.0, 0.65, 128i64);
    let modes: Vec<i64> = (1..=10).collect();
    let f = dispersion_frequencies(c, l, n, 0.9, &modes);
    for (i, &m) in modes.iter().enumerate() {
        assert_eq!(f[i], discrete_mode_frequency(c, l, n, 0.9, m));
    }
    // At λ = 1 the phase velocity is flat at c — a dispersionless scheme, which is the whole point
    // of tuning to λ = 1. Below it the curve droops monotonically with mode number.
    let v1 = phase_velocity(&dispersion_frequencies(c, l, n, 1.0, &modes), l, &modes);
    for &v in &v1 {
        assert!(
            (v / c - 1.0).abs() < 1e-12,
            "phase velocity {v} != c at lambda 1"
        );
    }
    let v9 = phase_velocity(&f, l, &modes);
    for i in 1..v9.len() {
        assert!(v9[i] < v9[i - 1], "the dispersion curve is not monotone");
    }
    assert!(v9[0] < c, "every mode must be slow below lambda = 1");
}

#[test]
fn the_string_harness_oracles_hold_at_its_own_parameters() {
    // Carried from `tests/test_modal.py::test_discrete_oracle_matches_continuous_at_lambda_one` and
    // `tests/test_dispersion.py`'s two helper tests (retirement plan §31), at the Python's own
    // numbers: `make_string`'s c = 200 m/s on a unit length. The bar above makes the same claims at
    // L = 0.65 and N = 128 with a 1e-9 ABSOLUTE bound; these are relative, at 1e-12. The Python's
    // `np.isclose` / `np.allclose` carried NumPy's default `atol = 1e-8` (§16's (d)), which at
    // 2,500 Hz was the looser of its two terms; the bound here is the `rtol` it wrote down.
    let (c, l) = (200.0, 1.0);
    let rel = |a: f64, b: f64| ((a - b) / b).abs();

    // The scalar oracle at lambda = 1 IS the continuum, mode by mode (N = 100).
    for m in [1_i64, 5, 10, 25] {
        let f = discrete_mode_frequency(c, l, 100, 1.0, m);
        let cont = m as f64 * c / (2.0 * l);
        assert!(rel(f, cont) < 1e-12, "mode {m}: {f} vs {cont}");
    }

    // `v_p = 2 L f / m` hands back c for every mode of the continuum series.
    let modes: Vec<i64> = (1..=20).collect();
    let f_cont: Vec<f64> = modes.iter().map(|&m| m as f64 * c / (2.0 * l)).collect();
    for (m, v) in modes.iter().zip(phase_velocity(&f_cont, l, &modes)) {
        assert!(rel(v, c) < 1e-12, "mode {m}: v_p {v} != c");
    }

    // The vectorised oracle is the scalar one (exactly — it is a map over it), equals the
    // continuum at lambda = 1, and lies below it at `LAMBDA_DISPERSIVE = 0.8` for every mode.
    let (n, lam) = (128_i64, 0.8);
    let modes = [1_i64, 5, 10, 25, 50];
    let vec = dispersion_frequencies(c, l, n, lam, &modes);
    let at_one = dispersion_frequencies(c, l, n, 1.0, &modes);
    for (i, &m) in modes.iter().enumerate() {
        let cont = m as f64 * c / (2.0 * l);
        assert_eq!(vec[i], discrete_mode_frequency(c, l, n, lam, m), "mode {m}");
        assert!(
            rel(at_one[i], cont) < 1e-12,
            "mode {m}: {} vs {cont}",
            at_one[i]
        );
        assert!(
            vec[i] < cont,
            "mode {m} is not lowered by dispersion: {}",
            vec[i]
        );
    }
}

#[test]
fn stiff_dispersion_delegates_exactly_too() {
    let (c, l, n, kappa, k, theta) = (200.0, 0.65, 128i64, 0.7, 1e-5, 0.5);
    let modes: Vec<i64> = (1..=6).collect();
    let f = stiff_dispersion_frequencies(c, l, n, kappa, k, theta, &modes);
    for (i, &m) in modes.iter().enumerate() {
        assert_eq!(
            f[i],
            discrete_stiff_mode_frequency(c, l, n, kappa, k, m, theta)
        );
    }
    // Stiffness raises the phase velocity with mode number, the opposite of pure numerical
    // dispersion -- which is why a stiff string's partials go sharp rather than flat.
    let v = phase_velocity(&f, l, &modes);
    for i in 1..v.len() {
        assert!(
            v[i] > v[i - 1],
            "stiffness did not raise the phase velocity"
        );
    }
}

// -- the stiff string's oracles, at its harness's parameters (retirement plan §32) -------------
//
// Carried from `tests/test_stiff_string.py`'s oracle-only tests: `c = 200 m/s`, `L = 1`,
// `kappa = 2` (`KAPPA_DEFAULT`), `theta = 0.28` (`THETA_DEFAULT`, written as a literal: this crate
// cannot see the core's constant, and `string_stiff_harness.rs` pins the two equal).

const STIFF_C: f64 = 200.0;
const STIFF_L: f64 = 1.0;
const STIFF_KAPPA: f64 = 2.0;
const STIFF_THETA: f64 = 0.28;

#[test]
fn the_stiff_oracle_sits_on_the_stretched_law_out_to_a_measured_horizon() {
    // Carried from `test_discrete_oracle_converges_to_continuum_stretched_law`, the physics anchor
    // (no simulation). On a fine grid the discrete oracle sits on `f_n = n f0 sqrt(1 + B n^2)`, and
    // the fundamental is itself stretched. The band is MEASURED by `pitch_horizon`; the old
    // hand-picked 10 survives as a FLOOR, so a wrong "fix" that shortened the band cannot pass on
    // fewer modes (resolution-horizon plan §6; 48 when measured on 2026-09-07).
    let (c, l, kappa) = (STIFF_C, STIFF_L, STIFF_KAPPA);
    let (n, fs) = (4000i64, 8.0e5);
    let k = 1.0 / fs;
    let window = 200usize;
    let floor = 10usize;
    let oracle: Vec<f64> = (1..=window as i64)
        .map(|m| discrete_stiff_mode_frequency(c, l, n, kappa, k, m, STIFF_THETA))
        .collect();
    let continuum = stiff_harmonic_frequencies(c, l, kappa, window);
    let (horizon, monotone) = pitch_horizon(&oracle, &continuum, 1.0).unwrap();
    println!("stiff pitch horizon: {horizon} (monotone {monotone})");
    assert!(
        monotone,
        "pitch error is not monotone in mode index, so the prefix count hides a mode"
    );
    assert!(
        horizon < window,
        "horizon {horizon} truncated by the window"
    );
    assert!(
        horizon >= floor,
        "only the first {horizon} partials are within a cent of the stretched law"
    );

    let f0 = c / (2.0 * l);
    let b = inharmonicity_b(c, l, kappa);
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    // The continuum stretches even the fundamental: f1 = f0 sqrt(1 + B), sharp of f0.
    assert!(rel(continuum[0], f0 * (1.0 + b).sqrt()) <= 1e-12);
    assert!(continuum[0] > f0);
    assert!(
        rel(oracle[0], continuum[0]) <= 1e-4,
        "{} vs {}",
        oracle[0],
        continuum[0]
    );
}

#[test]
fn the_inharmonicity_is_zero_scales_as_kappa_squared_and_stretches_every_partial() {
    // Carried from `test_inharmonicity_B_and_stretched_law`.
    let (c, l) = (STIFF_C, STIFF_L);
    assert_eq!(inharmonicity_b(c, l, 0.0), 0.0);
    // B ~ kappa^2 -- `pytest.approx`'s default `rel = 1e-6`.
    let (b1, b2) = (inharmonicity_b(c, l, 1.0), inharmonicity_b(c, l, 2.0));
    assert!((b2 - 4.0 * b1).abs() <= 1e-6 * 4.0 * b1, "{b2} vs 4 x {b1}");
    // kappa = 0 recovers the exact harmonic series.
    let harm = stiff_harmonic_frequencies(c, l, 0.0, 6);
    for (a, b) in harm.iter().zip(harmonic_frequencies(c, l, 6)) {
        assert!((a - b).abs() <= 1e-8 + 1e-5 * b.abs(), "{a} vs {b}");
    }
    // kappa > 0: every partial sharp of n f0, and increasingly so -- the stretch grows with n.
    let f0 = c / (2.0 * l);
    let stretch: Vec<f64> = stiff_harmonic_frequencies(c, l, STIFF_KAPPA, 6)
        .iter()
        .enumerate()
        .map(|(i, f)| f / ((i + 1) as f64 * f0))
        .collect();
    assert!(stretch.iter().all(|&s| s > 1.0), "{stretch:?}");
    assert!(stretch.windows(2).all(|w| w[1] > w[0]), "{stretch:?}");
}

#[test]
fn the_stiff_oracle_approaches_the_continuum_under_refinement() {
    // Carried from `test_discrete_stiff_oracle_tends_to_continuum_on_refinement`: mode 3 at
    // lam = 1 (`fs = c N / L`), N = 256, 512, 1024.
    let (c, l, kappa, m) = (STIFF_C, STIFF_L, STIFF_KAPPA, 3i64);
    let cont = stiff_harmonic_frequencies(c, l, kappa, m as usize)[m as usize - 1];
    let errs: Vec<f64> = [256i64, 512, 1024]
        .iter()
        .map(|&n| {
            let fs = c * n as f64 / l;
            (discrete_stiff_mode_frequency(c, l, n, kappa, 1.0 / fs, m, STIFF_THETA) - cont).abs()
        })
        .collect();
    assert!(
        errs[0] > errs[1] && errs[1] > errs[2],
        "not monotone: {errs:?}"
    );
}

#[test]
fn the_stiff_dispersion_is_the_scalar_oracle_and_rises_above_c() {
    // Carried from `test_stiff_dispersion_frequencies_match_scalar_oracle`, at its N = 128 and
    // lam = 1 (`k = L / (c N)`). Exactly equal -- the vector form is a map over the scalar one --
    // and every phase velocity above c, where the bar above asserts only that it rises.
    let (c, l, n, kappa) = (STIFF_C, STIFF_L, 128i64, STIFF_KAPPA);
    let k = l / (c * n as f64);
    let modes = [1i64, 5, 10, 25, 50];
    let vec = stiff_dispersion_frequencies(c, l, n, kappa, k, STIFF_THETA, &modes);
    for (i, &m) in modes.iter().enumerate() {
        assert_eq!(
            vec[i],
            discrete_stiff_mode_frequency(c, l, n, kappa, k, m, STIFF_THETA),
            "mode {m}"
        );
    }
    let vp = phase_velocity(&vec, l, &modes);
    assert!(vp.iter().all(|&v| v / c > 1.0), "{vp:?}");
}

#[test]
fn at_zero_stiffness_the_theta_scheme_is_its_own_closed_form_and_only_near_the_ideal_string() {
    // Carried from `test_kappa_zero_is_self_consistent_not_ideal_string`. At kappa = 0 the stiff
    // string is the implicit theta-scheme -- a DIFFERENT scheme from the explicit ideal string,
    // not exact even at lam = 1. (a) Its oracle is the closed form
    // `s = lam^2 sin^2 / (1 + 4 theta lam^2 sin^2)`, strictly below the explicit oracle for
    // theta > 0; (c) the two agree only loosely, in the low-mode limit.
    let (c, l, n) = (STIFF_C, STIFF_L, 128i64);
    // `make_stiff_string(N=128, lam=1.0)`: fs = c N / (L lam), k = 1 / fs.
    let k = 1.0 / (c * n as f64 / (l * 1.0));
    let theta = STIFF_THETA;
    for m in [1i64, 5, 20] {
        let f_stiff0 = discrete_stiff_mode_frequency(c, l, n, 0.0, k, m, theta);
        let f_ideal = discrete_mode_frequency(c, l, n, 1.0, m);
        assert!(
            f_stiff0 < f_ideal,
            "mode {m}: the implicit scheme must lie below"
        );
        let sin2 = (m as f64 * std::f64::consts::PI / (2 * n) as f64)
            .sin()
            .powi(2);
        let s_val = sin2 / (1.0 + 4.0 * theta * sin2);
        let closed = s_val.sqrt().asin() / (std::f64::consts::PI * k);
        assert!(
            ((f_stiff0 - closed) / closed).abs() <= 1e-12,
            "mode {m}: {f_stiff0} vs the closed form {closed}"
        );
    }
    for m in [1i64, 2, 4] {
        let f_stiff0 = discrete_stiff_mode_frequency(c, l, n, 0.0, k, m, theta);
        let f_ideal = discrete_mode_frequency(c, l, n, 1.0, m);
        let rel = (f_stiff0 - f_ideal).abs() / f_ideal;
        assert!(
            rel < 1e-2,
            "mode {m}: {rel:e} -- close, but NOT machine precision"
        );
    }
}

// -- duffing ---------------------------------------------------------------------------------

#[test]
fn a_linear_duffing_is_a_cosine_and_its_frequency_is_omega_zero() {
    let w0sq = (2.0 * std::f64::consts::PI * 220.0f64).powi(2);
    assert_eq!(duffing_elliptic_parameter(0.05, w0sq, 0.0).unwrap(), 0.0);
    let f = duffing_frequency(0.05, w0sq, 0.0).unwrap();
    assert!(
        (f / w0sq.sqrt() - 1.0).abs() < 1e-15,
        "eps = 0 must give back omega0: {f} vs {}",
        w0sq.sqrt()
    );
    assert!(duffing_frequency_shift(0.05, w0sq, 0.0).unwrap().abs() < 1e-9);
    // And the waveform is A cos(omega0 t) exactly, because cn(u, 0) is cos(u) exactly.
    let ts: Vec<f64> = (0..50).map(|i| i as f64 * 1e-4).collect();
    let q = duffing_displacement(&ts, 0.07, w0sq, 0.0).unwrap();
    for (i, &t) in ts.iter().enumerate() {
        assert_eq!(q[i], 0.07 * (w0sq.sqrt() * t).cos());
    }
}

#[test]
fn a_stiffening_duffing_goes_sharp_and_the_first_order_expansion_agrees_at_small_amplitude() {
    let w0sq = (2.0 * std::f64::consts::PI * 220.0f64).powi(2);
    let eps = 5e7;
    let mut prev = w0sq.sqrt();
    for &a in &[0.001, 0.01, 0.03, 0.07] {
        let w = duffing_frequency(a, w0sq, eps).unwrap();
        assert!(w > prev, "a hardening spring must go sharp with amplitude");
        prev = w;
    }
    // The Lindstedt–Poincaré expansion keeps the `3εA²/8ω₀²` term and drops the next one, so its
    // *relative* error is O(A⁴): dividing the amplitude by √10 must divide the error by 100.
    // Measured 99.80. That is a much sharper statement than "they are close" — an exact form that
    // was really another approximation would converge at some other rate — and it is the reason
    // `duffing_frequency_expansion`'s docstring can call itself a cross-check rather than an oracle.
    let mut prev_e: Option<f64> = None;
    for &a in &[0.01, 0.003_162_277_660_168_38, 0.001] {
        let exact = duffing_frequency(a, w0sq, eps).unwrap();
        let approx = duffing_frequency_expansion(a, w0sq, eps).unwrap();
        let e = (approx / exact - 1.0).abs();
        if let Some(p) = prev_e {
            assert!(
                (p / e - 100.0).abs() < 5.0,
                "the expansion error fell by {} rather than 100 per sqrt-decade",
                p / e
            );
        }
        prev_e = Some(e);
    }
}

#[test]
fn the_duffing_waveform_starts_at_rest_and_repeats_at_its_own_period() {
    let w0sq = (2.0 * std::f64::consts::PI * 220.0f64).powi(2);
    let (a, eps) = (0.07, 5e7);
    assert_eq!(duffing_displacement(&[0.0], a, w0sq, eps).unwrap()[0], a);
    // q(t) = A cn(Ωt, m) has period 2π/ω where ω is `duffing_frequency` — which ties the waveform
    // and the frequency together, two functions that share only the elliptic parameter.
    let period = 2.0 * std::f64::consts::PI / duffing_frequency(a, w0sq, eps).unwrap();
    let q = duffing_displacement(&[period, 2.0 * period, 0.5 * period], a, w0sq, eps).unwrap();
    assert!(
        (q[0] - a).abs() < 1e-12,
        "one period does not return to A: {}",
        q[0]
    );
    assert!(
        (q[1] - a).abs() < 1e-12,
        "two periods do not either: {}",
        q[1]
    );
    assert!(
        (q[2] + a).abs() < 1e-12,
        "half a period is not -A: {}",
        q[2]
    );
}

#[test]
fn the_kirchhoff_carrier_coefficients_are_the_linear_and_cubic_halves() {
    let (c, kappa, ea, rho, p2, l) = (200.0, 0.0, 1.2e4, 6.3e-3, 400.0, 0.65);
    let (w0sq, eps) = kc_mode_coefficients(c, kappa, ea, rho, p2, l).unwrap();
    assert_eq!(w0sq, c * c * p2, "kappa = 0 leaves only the wave term");
    assert_eq!(eps, (ea / (4.0 * rho)) * p2 * p2);
    // EA = 0 kills the nonlinearity entirely: the string stops modulating its own tension.
    let (_, none) = kc_mode_coefficients(c, kappa, 0.0, rho, p2, l).unwrap();
    assert_eq!(none, 0.0);
    assert!(kc_mode_coefficients(c, kappa, -1.0, rho, p2, l).is_err());
    assert!(kc_mode_coefficients(c, kappa, ea, 0.0, p2, l).is_err());
    assert!(kc_mode_coefficients(c, kappa, ea, rho, -1.0, l).is_err());
    // The stretch is quadratic in amplitude, which is the whole reason the restoring force is cubic.
    assert_eq!(
        kc_mode_stretch(0.02, p2, l),
        4.0 * kc_mode_stretch(0.01, p2, l)
    );
}

// The tension-modulated string harness's oracle bars, carried from `tests/test_tension_string.py`
// at its own numbers (retirement plan §34): `omega0² = 3.947e5` (mode 1 of the harness string,
// rounded) and `eps = 5e7`. The stretch identity needs the core crate's grid and lives in
// `crates/physsynth-core/tests/string_nonlinear_harness.rs`.

#[test]
fn at_zero_eps_the_duffing_frequency_is_omega_zero_to_the_last_bit_at_three_scales() {
    // Carried from `test_duffing_frequency_linear_limit_is_exact`: `m = 0` and `K(0) = pi/2`
    // cancel, at three decades of `omega0²` and a large amplitude that must not matter.
    for w0sq in [1.0f64, 1e4, 3.947e5] {
        let w = duffing_frequency(2.3, w0sq, 0.0).unwrap();
        assert!(
            (w / w0sq.sqrt() - 1.0).abs() <= 1e-15,
            "{w} vs {}",
            w0sq.sqrt()
        );
    }
}

#[test]
fn the_expansion_agrees_at_small_amplitude_and_is_simply_wrong_at_large() {
    // Carried from `test_duffing_frequency_matches_expansion_at_small_amplitude`: elliptic form vs
    // the independent Lindstedt–Poincaré expansion. Agreement as A -> 0 is evidence both are right;
    // at A = 0.3 the expansion is ~74 % off, which is why it is not the oracle.
    let (w0sq, eps) = (3.947e5, 5.0e7);
    let exact = duffing_frequency(1e-3, w0sq, eps).unwrap();
    let approx = duffing_frequency_expansion(1e-3, w0sq, eps).unwrap();
    assert!(
        (exact - approx).abs() <= 1e-7 * approx,
        "{exact} vs {approx}"
    );
    let exact = duffing_frequency(0.3, w0sq, eps).unwrap();
    let approx = duffing_frequency_expansion(0.3, w0sq, eps).unwrap();
    println!("A = 0.3: elliptic {exact}, expansion {approx}");
    assert!((exact - approx).abs() / exact > 0.5, "{exact} vs {approx}");
}

#[test]
fn a_hardening_duffings_parameter_never_reaches_the_pole() {
    // Carried from `test_duffing_elliptic_parameter_never_reaches_the_singularity`. Hardening keeps
    // `m` in [0, 1/2], strictly away from K(m)'s pole at 1; in floating point it ROUNDS to 1/2 at
    // extreme amplitude, harmless since K(1/2) ~ 1.854. A softening spring would have no such bound.
    let (w0sq, eps) = (3.947e5, 5.0e7);
    for a in [0.0, 1.0, 1e4, 1e8] {
        let m = duffing_elliptic_parameter(a, w0sq, eps).unwrap();
        assert!((0.0..=0.5).contains(&m), "A = {a}: m = {m}");
    }
    assert!(duffing_elliptic_parameter(1.0, w0sq, eps).unwrap() < 0.5);
    assert!(duffing_frequency(1e8, w0sq, eps).unwrap().is_finite());
}

#[test]
fn the_waveform_starts_at_its_amplitude_and_is_a_cosine_at_zero_eps_at_the_harness_numbers() {
    // Carried from `test_duffing_displacement_starts_at_rest_and_degenerates_to_cosine`, on the
    // Python's 200-point window to 0.05 s and its `atol = 1e-14`.
    let w0sq = 3.947e5;
    let q0 = duffing_displacement(&[0.0], 0.07, w0sq, 5e7).unwrap()[0];
    assert!((q0 / 0.07 - 1.0).abs() <= 1e-14, "{q0}");
    let ts: Vec<f64> = (0..200).map(|i| 0.05 * i as f64 / 199.0).collect();
    let q = duffing_displacement(&ts, 0.07, w0sq, 0.0).unwrap();
    for (qi, &t) in q.iter().zip(&ts) {
        let cosine = 0.07 * (w0sq.sqrt() * t).cos();
        assert!((qi - cosine).abs() <= 1e-14, "t = {t}: {qi} vs {cosine}");
    }
}

#[test]
fn the_kirchhoff_carrier_coefficients_reduce_to_the_stiff_string_at_zero_ea() {
    // Carried from `test_kc_mode_coefficients_reduce_to_the_linear_string`: `EA = 0` gives
    // `eps = 0`, and `omega0² = c²p² + kappa²p⁴` is the linear stiff-string relation (written out
    // here, with the harness's `kappa = 2`, so the bar does not repeat the oracle's spelling).
    let p2 = 9.87;
    let (w0sq, eps) = kc_mode_coefficients(200.0, 2.0, 0.0, 0.005, p2, 1.0).unwrap();
    assert_eq!(eps, 0.0);
    let want = 200.0 * 200.0 * p2 + 2.0 * 2.0 * p2 * p2;
    assert!((w0sq / want - 1.0).abs() <= 1e-12, "{w0sq} vs {want}");
}

#[test]
fn the_duffing_oracle_refuses_a_negative_ea_and_a_negative_omega_zero_squared() {
    // Carried from `test_duffing_oracle_rejects_nonphysical_input`, with the prose asserted so the
    // two refusals cannot stand in for each other (§32.4 D).
    let err = kc_mode_coefficients(200.0, 2.0, -1.0, 0.005, 9.87, 1.0).unwrap_err();
    assert_eq!(err, "EA (axial stiffness) must be >= 0.");
    let err = duffing_frequency(0.1, -1.0, 1e7).unwrap_err();
    assert_eq!(err, "omega0_sq must be positive, got -1");
}

// -- radiation (the piston oracle) -------------------------------------------------------------

#[test]
fn the_piston_reaches_twice_the_free_space_monopole_in_the_rayleigh_limit() {
    // As ka -> 0 the bracket tends to (ka)^2/2 and R_a -> rho0 omega^2 / (2 pi c0) -- exactly twice
    // the free-space monopole, because a baffle radiates into 2 pi steradians rather than 4 pi.
    // The factor of two is the physics; getting it wrong would still give a smooth bounded curve.
    let (omega, a) = (2.0 * std::f64::consts::PI * 5.0, 1e-3);
    let r = piston_radiation_resistance(omega, a, RHO0_AIR, C0_AIR);
    let want = RHO0_AIR * omega * omega / (2.0 * std::f64::consts::PI * C0_AIR);
    assert!(
        (r / want - 1.0).abs() < 1e-6,
        "Rayleigh limit: {r} vs {want}"
    );
}

#[test]
fn the_pistons_two_branches_meet_at_their_threshold() {
    // This test asserted the OPPOSITE until 2026-09-03, deliberately: the shipped Python's series
    // cutoff sat at `ka = 1e-8`, three decades below where `1 - J1(2ka)/ka` becomes computable, and
    // just above it the function was 544% wrong. That bar is what found the defect (hurdles §14),
    // and this is what replaces it now the threshold has moved to 3e-2 with three series terms.
    //
    // Continuity across the seam is the property to assert rather than accuracy on either side: a
    // future threshold edit that lands back inside the cancellation would show up here as a step,
    // and nothing else in the suite would notice — every physics bar downstream is percentage-level
    // and the two branches differ by parts in 1e13.
    let (radius, c0) = (0.05, C0_AIR);
    let cut = PISTON_SERIES_CUTOFF_KA;
    for d in [0.999, 0.9999, 1.0, 1.0001, 1.001] {
        let ka = cut * d;
        let omega = ka * c0 / radius;
        let r = piston_radiation_resistance(omega, radius, RHO0_AIR, c0);
        // Evaluate both brackets directly, so this compares the two formulas rather than the
        // function against itself. `ka` is recomputed from `omega` the way the function does it,
        // because `omega * radius / c0` does not round-trip `ka * c0 / radius` exactly -- and the
        // scale is applied in the function's own association, since `(scale * ka2) * rest` and
        // `scale * (ka2 * rest)` are different doubles (§27).
        let ka = omega * radius / c0;
        let ka2 = ka * ka;
        let scale = RHO0_AIR * c0 / (std::f64::consts::PI * radius * radius);
        let series = scale * (ka2 * (0.5 - ka2 * (1.0 / 12.0 - ka2 / 144.0)));
        let direct = scale * (1.0 - j1(2.0 * ka) / ka);
        assert!(
            (series / direct - 1.0).abs() < 1e-11,
            "the branches disagree by {:.3e} at ka = {ka}; the threshold has moved back into the              cancellation",
            (series / direct - 1.0).abs()
        );
        assert!(
            r == series || r == direct,
            "the function took neither branch at ka = {ka}"
        );
    }
}

#[test]
fn the_pistons_series_is_the_brackets_own_taylor_expansion() {
    // The three terms are (ka)^2/2 - (ka)^4/12 + (ka)^6/144, and getting a coefficient wrong would
    // still give a smooth, monotone, plausibly-sized curve. Check them against the expansion
    // written out term by term rather than in Horner form -- a different arrangement of the same
    // series, so a transposed coefficient shows and a re-association does not.
    for &ka in &[1e-6, 1e-4, 1e-3, 1e-2, 2.9e-2] {
        let radius = 0.05;
        let scale = RHO0_AIR * C0_AIR / (std::f64::consts::PI * radius * radius);
        let omega = ka * C0_AIR / radius;
        let got = piston_radiation_resistance(omega, radius, RHO0_AIR, C0_AIR) / scale;
        let want = ka * ka / 2.0 - ka * ka * ka * ka / 12.0 + ka * ka * ka * ka * ka * ka / 144.0;
        assert!(
            (got / want - 1.0).abs() < 1e-14,
            "series mismatch at ka = {ka}: {got} vs {want}"
        );
    }
}

#[test]
fn the_piston_resistance_rises_and_saturates() {
    // R_a is monotone in ka up to the first maximum and tends to rho0 c0 / S as ka -> infinity,
    // because J1(2ka)/ka -> 0. That plateau is the plane-wave limit and is what makes the piston a
    // sensible reference resistance for a bore's bell.
    let (radius, c0) = (0.05, C0_AIR);
    let plane = RHO0_AIR * c0 / (std::f64::consts::PI * radius * radius);
    let big = piston_radiation_resistance(400.0 * c0 / radius, radius, RHO0_AIR, c0);
    assert!(
        (big / plane - 1.0).abs() < 5e-3,
        "far above the cutoff R_a should be the plane-wave value: {big} vs {plane}"
    );
    let small = piston_radiation_resistance(0.1 * c0 / radius, radius, RHO0_AIR, c0);
    assert!(
        small < 0.02 * plane,
        "well below the cutoff it should be tiny"
    );
}

#[test]
fn the_air_constants_match_the_cores() {
    // This crate cannot depend on `physsynth-core`, so the two numbers are literals in two files.
    // The values are checked here against the same digits `physsynth/core/radiation.py` defines, so
    // a change to one without the other is a red test rather than a silent physics shift.
    assert_eq!(RHO0_AIR, 1.2041);
    assert_eq!(C0_AIR, 343.0);
}
