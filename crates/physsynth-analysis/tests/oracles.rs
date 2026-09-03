//! `damping`, `dispersion` and `duffing` — their own bars.
//!
//! Three small modules with one property each that a parity test could not establish: the decay
//! oracle must return *no decay* for a lossless string, the dispersion curve must be flat when the
//! scheme is exact, and the Duffing solution must actually solve the Duffing equation.

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
use physsynth_analysis::modal::{discrete_mode_frequency, discrete_stiff_mode_frequency};

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
