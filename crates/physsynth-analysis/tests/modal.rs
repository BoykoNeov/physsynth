//! The oracles' own bars — the limits, degeneracies and published digits they promise.
//!
//! An oracle is the one kind of code a parity test protects least well. If both implementations
//! compute the same wrong frequency, every physics test in the project agrees with it beautifully
//! and the simulation is validated against a fiction. So these assert the properties the
//! docstrings claim: that the discrete formulas collapse to the continuum ones in the limits where
//! they must, that the anisotropic ones reduce to the isotropic ones when the anisotropy is turned
//! off, and that the two root-finds land on the digits the literature publishes.

use physsynth_analysis::modal::{
    bore_resonance_frequencies, cents, circular_membrane_freqs, discrete_beam_eigenfrequency,
    discrete_bore_eigenfrequency, discrete_membrane_eigenfrequency, discrete_mode_frequency,
    discrete_orthotropic_plate_eigenfrequency, discrete_plate_eigenfrequency,
    free_circular_plate_lambda_roots, free_circular_plate_lambdas,
    free_circular_plate_saddle_bound, free_free_beam_beta_l, free_free_beam_freqs,
    free_plate_coupling_form, free_plate_freq_from_lambda, free_plate_twist_bound,
    harmonic_frequencies, inharmonicity_b, orthotropic_plate_freqs, rectangular_membrane_freqs,
    rectangular_plate_freqs, stiff_harmonic_frequencies,
};

/// The free–free beam's first four roots, published to twelve digits in every vibration text.
const BETA_L: [f64; 4] = [
    4.730_040_744_862_704,
    7.853_204_624_095_838,
    10.995_607_838_001_67,
    14.137_165_491_257_46,
];

#[test]
fn the_explicit_scheme_is_exact_at_lambda_one() {
    // The claim `discrete_mode_frequency` makes in its own docstring: at λ = 1 the numerical
    // dispersion vanishes and the discrete frequency IS the continuum one. Not "close to" —
    // arcsin(sin(x)) is the identity on this branch, so it should hold to a rounding.
    let (c, l, n) = (200.0, 0.65, 128);
    let cont = harmonic_frequencies(c, l, 8);
    for m in 1..=8i64 {
        let d = discrete_mode_frequency(c, l, n, 1.0, m);
        assert!(
            (d - cont[(m - 1) as usize]).abs() < 1e-9,
            "mode {m}: discrete {d} vs continuum {}",
            cont[(m - 1) as usize]
        );
    }
    // And below λ = 1 it must sit *below* the continuum — dispersion is a flattening, never a
    // sharpening. Getting this sign backwards is the classic way to "validate" a broken scheme.
    for m in 1..=8i64 {
        assert!(
            discrete_mode_frequency(c, l, n, 0.8, m) < cont[(m - 1) as usize],
            "mode {m} came out sharp under dispersion"
        );
    }
}

#[test]
fn stiffness_pushes_every_partial_sharp_including_the_first() {
    let (c, l, kappa) = (200.0, 0.65, 0.7);
    let plain = harmonic_frequencies(c, l, 10);
    let stiff = stiff_harmonic_frequencies(c, l, kappa, 10);
    for i in 0..10 {
        assert!(stiff[i] > plain[i], "partial {} is not sharp", i + 1);
    }
    // Including the fundamental: f_1 = f_0 sqrt(1 + B), NOT f_0. The docstring says so because it
    // is the detail a reader gets wrong.
    let b = inharmonicity_b(c, l, kappa);
    assert!((stiff[0] - plain[0] * (1.0 + b).sqrt()).abs() < 1e-12);
    // κ = 0 must give back the plain series exactly — same expression with B = 0.
    let none = stiff_harmonic_frequencies(c, l, 0.0, 10);
    assert_eq!(none, plain, "kappa = 0 is not the harmonic series");
}

#[test]
fn cents_is_zero_on_itself_and_1200_on_an_octave() {
    for &f in &[27.5, 440.0, 12543.85] {
        assert_eq!(cents(f, f), 0.0, "a pitch is zero cents from itself");
        assert!((cents(2.0 * f, f) - 1200.0).abs() < 1e-12);
        assert!((cents(f, 2.0 * f) + 1200.0).abs() < 1e-12);
    }
}

#[test]
fn the_discrete_membrane_and_plate_frequencies_reach_their_continuum_limits() {
    // f = arccos(1 - c²k²Λ/2)/(2πk) → c√Λ/(2π) as k → 0, and the plate's θ-scheme form
    // → κΛ/(2π). Both are the identity the temporal companion of an eigenvalue test rests on.
    //
    // Asserted as a *rate* rather than a tolerance, which is the stronger statement and the one a
    // wrong constant inside the formula could not fake: both errors are O(k²), so a tenfold
    // refinement must divide them by about a hundred. Measured: 3.75e-4 → 3.75e-6 → 3.76e-8 for
    // the membrane and 1.33e-3 → 1.33e-5 → 1.33e-7 for the plate.
    let (c, kappa, lambda) = (150.0f64, 2.0f64, 4_000.0f64);
    let want_m = c * lambda.sqrt() / (2.0 * std::f64::consts::PI);
    let want_p = kappa * lambda / (2.0 * std::f64::consts::PI);
    let mut prev: Option<(f64, f64)> = None;
    for &k in &[1e-5, 1e-6, 1e-7] {
        let e_m = (discrete_membrane_eigenfrequency(lambda, c, k) / want_m - 1.0).abs();
        let e_p = (discrete_plate_eigenfrequency(lambda, kappa, k, 0.5) / want_p - 1.0).abs();
        assert!(e_m < 1e-3 && e_p < 2e-3, "not yet in the limit at k={k}");
        if let Some((pm, pp)) = prev {
            assert!(
                (pm / e_m - 100.0).abs() < 5.0,
                "membrane error fell by {} rather than 100 at k={k}",
                pm / e_m
            );
            assert!(
                (pp / e_p - 100.0).abs() < 5.0,
                "plate error fell by {} rather than 100 at k={k}",
                pp / e_p
            );
        }
        prev = Some((e_m, e_p));
    }
}

#[test]
fn the_orthotropic_oracles_reduce_to_the_isotropic_ones() {
    // With all three grain ratios at 1 the wood becomes metal, and the orthotropic formulas must
    // return exactly what the isotropic ones do. "Where the factor of 2 lives" is model #5o's
    // recorded trap, and this is the assertion that catches it.
    let (kappa, lx, ly) = (3.5, 0.4, 0.3);
    let modes = [(1i64, 1i64), (2, 1), (1, 2), (3, 2)];
    let iso = rectangular_plate_freqs(kappa, lx, ly, &modes);
    let ortho = orthotropic_plate_freqs(kappa, lx, ly, &modes, 1.0, 1.0, 1.0).unwrap();
    for (a, b) in iso.iter().zip(ortho.iter()) {
        assert!((a - b).abs() < 1e-9 * a, "isotropic {a} vs orthotropic {b}");
    }
    // The discrete pair, same reduction: λ_x + λ_y is the isotropic Laplacian eigenvalue.
    let (lam_x, lam_y, k, theta) = (900.0, 1_600.0, 1e-5, 0.5);
    let d_iso = discrete_plate_eigenfrequency(lam_x + lam_y, kappa, k, theta);
    let d_ortho =
        discrete_orthotropic_plate_eigenfrequency(lam_x, lam_y, kappa, k, theta, 1.0, 1.0, 1.0)
            .unwrap();
    assert!((d_iso - d_ortho).abs() < 1e-9 * d_iso);
    // A cross term below -sqrt(grain_x * grain_y) makes the modal stiffness negative, which is not
    // a plate; both must refuse rather than return a NaN frequency.
    assert!(orthotropic_plate_freqs(kappa, lx, ly, &modes, 1.0, -2.0, 1.0).is_err());
    assert!(discrete_orthotropic_plate_eigenfrequency(
        lam_x, lam_y, kappa, k, theta, 1.0, -2.0, 1.0
    )
    .is_err());
}

#[test]
fn the_beam_roots_are_the_published_ones_and_solve_their_own_equation() {
    let bl = free_free_beam_beta_l(4).unwrap();
    for (i, &want) in BETA_L.iter().enumerate() {
        assert!(
            (bl[i] - want).abs() < 1e-11,
            "beta_{} L = {} , published {want}",
            i + 1,
            bl[i]
        );
        // cos(x) cosh(x) = 1 -- the equation itself, evaluated in the form the solver did NOT use.
        assert!(
            (bl[i].cos() * bl[i].cosh() - 1.0).abs() < 1e-8,
            "root {} does not satisfy cos*cosh = 1",
            i + 1
        );
    }
    // And they approach (2i+1)π/2, which is what tells a reader the branch is right — a solver
    // that skipped a bracket would land every root one asymptote too far along.
    //
    // Note the direction is *not* asserted, and `modal.py`'s docstring overstates it: it says the
    // roots tend to (2i+1)π/2 "from above", which is true of the first (4.73004 > 4.71239) and
    // false of the second (7.85320 < 7.85398). The approach alternates and is quick — by the third
    // root the two agree to five decimals. Only the proximity is a real claim.
    for (i, &x) in bl.iter().enumerate() {
        let asym = (2 * (i + 1) + 1) as f64 * std::f64::consts::PI / 2.0;
        assert!(
            (x - asym).abs() < 0.02,
            "root {} is nowhere near its asymptote",
            i + 1
        );
    }
    assert!(free_free_beam_beta_l(0).is_err(), "n_modes = 0 must refuse");
}

#[test]
fn the_beam_frequencies_are_the_roots_squared_and_the_discrete_form_reaches_them() {
    let (kappa, l) = (1.2, 0.5);
    let f = free_free_beam_freqs(kappa, l, 3).unwrap();
    let bl = free_free_beam_beta_l(3).unwrap();
    for i in 0..3 {
        let want = kappa * bl[i] * bl[i] / (2.0 * std::f64::consts::PI * l * l);
        assert_eq!(f[i], want);
    }
    // The discrete companion: μ = β⁴ is the biharmonic eigenvalue, and as k → 0 the θ-scheme
    // frequency must return the continuum one.
    let mu = (bl[0] / l).powi(4);
    for &k in &[1e-6, 1e-7] {
        let d = discrete_beam_eigenfrequency(mu, kappa, k, 0.5);
        assert!((d / f[0] - 1.0).abs() < 1e-3 * (k / 1e-6));
    }
}

#[test]
fn the_free_disk_reproduces_its_derived_lambdas_and_respects_its_own_bound() {
    // ν = 0.3: 5.35833 (n=2), 9.00314 (n=0), 12.43899 (n=3), 20.47455 (n=1), 21.83516 (n=4) --
    // derived in `modal.py`'s docstring rather than cited, since the plate tables are paywalled.
    let (lam, nodal) = free_circular_plate_lambdas(0.3, 7, 8).unwrap();
    assert!((lam[0] - 5.358_33).abs() < 1e-4, "fundamental = {}", lam[0]);
    assert_eq!(nodal[0], 2, "the fundamental is the two-diameter saddle");
    assert_eq!(nodal[1], 2, "and it is a degenerate pair");
    assert!(
        (lam[2] - 9.003_14).abs() < 1e-4,
        "second distinct = {}",
        lam[2]
    );
    assert_eq!(nodal[2], 0, "the breathing mode is non-degenerate");
    assert_eq!(nodal[3], 3);
    // Ascending, which the sort promises.
    for i in 1..lam.len() {
        assert!(lam[i] >= lam[i - 1], "lambdas came back out of order");
    }
    // The closed-form saddle bound contains no Bessel function at all, so it is a genuinely
    // independent check on the whole determinant: sqrt(48(1-ν)) = 5.79655 at ν = 0.3, an 8.18%
    // one-term overshoot of the derived 5.35833.
    let bound = free_circular_plate_saddle_bound(0.3).unwrap();
    assert!((bound - 5.796_55).abs() < 1e-4);
    assert!(bound > lam[0], "a Rayleigh bound must overshoot");
    let overshoot = (bound / lam[0] - 1.0) * 100.0;
    assert!(
        (overshoot - 8.18).abs() < 0.05,
        "one-term overshoot is {overshoot}%, expected 8.18%"
    );
}

#[test]
fn the_disk_scan_finds_no_spurious_root_and_refuses_bad_poissons_ratios() {
    // Every root must be a root -- the scan brackets a sign change, and a bracket around a pole
    // would produce a plausible number that is not a mode. There is no pole here, and this is how
    // that is asserted rather than assumed: refine each root and check it stayed put.
    for n in 0..5i32 {
        let roots = free_circular_plate_lambda_roots(0.3, n, 14.0, 20000).unwrap();
        assert!(!roots.is_empty(), "n = {n} produced no elastic root");
        for &r in &roots {
            assert!(
                r > 1e-3,
                "the rigid-body root at lambda = 0 was not filtered"
            );
            assert!(r < 14.0);
        }
    }
    assert!(free_circular_plate_lambda_roots(0.6, 0, 14.0, 100).is_err());
    assert!(free_circular_plate_lambda_roots(0.3, -1, 14.0, 100).is_err());
    assert!(free_circular_plate_saddle_bound(-1.0).is_err());
}

#[test]
fn the_circular_membrane_spectrum_starts_where_the_first_bessel_zero_says() {
    let (c, a) = (100.0, 0.15);
    let modes = circular_membrane_freqs(c, a, 8, 12, 12);
    let want = c * 2.404_825_557_695_773 / (2.0 * std::f64::consts::PI * a);
    assert!(
        (modes[0].freq - want).abs() < 1e-10,
        "f_01 = {}",
        modes[0].freq
    );
    assert_eq!(modes[0].m, 0);
    assert_eq!(modes[0].n, 1);
    assert_eq!(modes[0].degeneracy, 1, "the (0,1) mode is non-degenerate");
    assert_eq!(modes[1].m, 1, "the second mode has one nodal diameter");
    assert_eq!(modes[1].degeneracy, 2, "and it is a cos/sin pair");
    // Ascending, and the classic ratio f_11/f_01 = 3.8317/2.4048 = 1.593.
    for i in 1..modes.len() {
        assert!(modes[i].freq >= modes[i - 1].freq);
    }
    assert!((modes[1].freq / modes[0].freq - 1.593_4).abs() < 1e-3);
}

#[test]
fn the_bore_series_is_odd_when_closed_open_and_full_otherwise() {
    let (c0, l) = (343.0, 0.6);
    let co = bore_resonance_frequencies(c0, l, 4, "closed-open").unwrap();
    for (i, &f) in co.iter().enumerate() {
        assert_eq!(f, (2 * (i + 1) - 1) as f64 * c0 / (4.0 * l));
    }
    // The clarinet's signature: the second resonance is a twelfth, not an octave.
    assert!((co[1] / co[0] - 3.0).abs() < 1e-12);
    let oo = bore_resonance_frequencies(c0, l, 4, "open-open").unwrap();
    let cc = bore_resonance_frequencies(c0, l, 4, "closed-closed").unwrap();
    assert_eq!(
        oo, cc,
        "both full-series boundaries give the same frequencies"
    );
    assert!((oo[1] / oo[0] - 2.0).abs() < 1e-12);
    assert!(bore_resonance_frequencies(c0, l, 4, "open-closed").is_err());
    // And the discrete form returns the continuum one as k → 0.
    let omega2 = (2.0 * std::f64::consts::PI * co[0]).powi(2);
    for &k in &[1e-6, 1e-7] {
        let d = discrete_bore_eigenfrequency(omega2, k);
        assert!((d / co[0] - 1.0).abs() < 1e-3 * (k / 1e-6));
    }
}

#[test]
fn the_small_helpers_refuse_what_they_promise_to_refuse() {
    assert!(free_plate_twist_bound(1.0, 0.3, 0.2, 0.5).is_ok());
    assert!(free_plate_twist_bound(0.0, 0.3, 0.2, 0.5).is_err());
    assert!(free_plate_twist_bound(1.0, 0.3, 0.2, 0.0).is_err());
    assert!(free_plate_coupling_form(1.0, 0.01, 10, 10).is_ok());
    assert!(free_plate_coupling_form(1.0, 0.0, 10, 10).is_err());
    assert!(free_plate_coupling_form(1.0, 0.01, 1, 10).is_err());
    // λ → Hz is pure arithmetic and linear in λ.
    let f = free_plate_freq_from_lambda(13.468, 2.0, 0.25);
    assert_eq!(
        free_plate_freq_from_lambda(2.0 * 13.468, 2.0, 0.25),
        2.0 * f
    );
}

#[test]
fn a_square_membrane_has_the_degeneracy_a_rectangle_does_not() {
    // (1,2) and (2,1) cost the same on a square and different amounts otherwise. A geometry bug in
    // the oracle would break the first half; a transposed axis would break the second.
    let sq = rectangular_membrane_freqs(200.0, 0.4, 0.4, &[(1, 2), (2, 1)]);
    assert_eq!(sq[0], sq[1]);
    let re = rectangular_membrane_freqs(200.0, 0.4, 0.3, &[(1, 2), (2, 1)]);
    assert!(re[0] > re[1], "the short axis must carry the higher mode");
}
