//! The resolution horizon of the BUILT models — the seven bars that read a model's own `k`, `h`,
//! `θ` and `c` rather than writing them down.
//!
//! Carried from `tests/test_resolution_horizon.py` (retirement plan §46), whose other 41 bars ask
//! only `modal`'s closed-form dispersion relations and live with the primitives in
//! `crates/physsynth-analysis/tests/horizon.rs`. These seven built a damped string, a plate or a
//! membrane through `tests/helpers.py`'s factories and read the result's attributes, so here they
//! read the native `Params` the same way — the string's `θ` is the model's, not a literal, and the
//! membrane's `k` is what `fs = c/(λh)` round-trips to, not `λh/c`. That is the only reason they
//! need this crate; the dependency on `physsynth-analysis` is test-only (plan §24.2). Nothing here
//! steps a model: the eigenvalues are closed form, so `Params::new` is as far as any bar goes.
//!
//! The two mechanisms, as the analysis file states them: the implicit θ-scheme's time and space
//! errors both flatten, so they COMPOUND onto a floor no sample rate passes; the explicit leapfrog's
//! time error is sharp and CANCELS the spatial droop at one Courant number (`λ = 1` in 1-D, the
//! diagonal at `λ = 1/√2` in 2-D).

use physsynth_analysis::horizon::{
    cancellation_courant, mode_family, pitch_error_cents, pitch_horizon, sinc_horizon_fraction,
};
use physsynth_analysis::modal;
use physsynth_core::membrane::{self, Domain};
use physsynth_core::plate::{self, PlateSpec};
use physsynth_core::string_damped;
use std::f64::consts::PI;

/// `tests/helpers.py`'s canonical string, plate and membrane.
const L_DEFAULT: f64 = 1.0;
const T_DEFAULT: f64 = 200.0;
const RHO_DEFAULT: f64 = 0.005;
const RHO_AREAL_DEFAULT: f64 = 0.005;
const KAPPA_PLATE: f64 = 20.0;
const CENTS: f64 = 5.0;

/// `1.0 / np.sqrt(2.0)`, spelled as the Python spelled it (the analysis file says why it matters).
fn mem_ceiling() -> f64 {
    1.0 / 2.0_f64.sqrt()
}

fn wave_speed() -> f64 {
    (T_DEFAULT / RHO_DEFAULT).sqrt()
}

/// `make_damped_string`: lossless, at the model's default `θ`, Courant number `lam` through
/// `fs = cN/(Lλ)`. `lam > 1` is allowed — the θ-scheme is unconditionally stable.
fn damped_string(n: i64, lam: f64, kappa: f64) -> string_damped::Params {
    let fs = wave_speed() * n as f64 / (L_DEFAULT * lam);
    string_damped::Params::new(
        L_DEFAULT,
        T_DEFAULT,
        RHO_DEFAULT,
        fs,
        n,
        kappa,
        0.0,
        0.0,
        string_damped::THETA,
        true,
    )
    .expect("an admissible damped string")
}

/// `make_membrane(domain="rectangle")`: a unit square whose Courant number is exactly `lam`,
/// through `fs = c/(λh)`.
fn membrane(n: i64, lam: f64) -> membrane::Params {
    let c = (T_DEFAULT / RHO_AREAL_DEFAULT).sqrt();
    let h = 1.0 / n as f64;
    membrane::Params::new(
        Some(Domain::Rectangle),
        T_DEFAULT,
        RHO_AREAL_DEFAULT,
        c / (lam * h),
        n,
        Some(1.0),
        Some(1.0),
        None,
        0.0,
    )
    .expect("an admissible membrane")
}

/// The spatial operator's horizon about the canonical string, with no timestep in it — the same
/// test fixture the analysis file carries, and a fixture rather than a library function for the
/// reason given there.
fn spatial_operator_horizon(n: i64, kappa: f64) -> usize {
    let h = L_DEFAULT / n as f64;
    let c = wave_speed();
    let (mut w_disc, mut w_cont) = (Vec::new(), Vec::new());
    for m in 1..n {
        let p2_disc = modal::dirichlet_axis_eigenvalue(m as f64, L_DEFAULT, h);
        let p2_cont = (m as f64 * PI / L_DEFAULT).powi(2);
        w_disc.push((c * c * p2_disc + kappa * kappa * p2_disc * p2_disc).sqrt());
        w_cont.push((c * c * p2_cont + kappa * kappa * p2_cont * p2_cont).sqrt());
    }
    pitch_horizon(&w_disc, &w_cont, CENTS).unwrap().0
}

/// The θ-scheme string's horizon, from the BUILT model's `k` and `θ`.
fn string_horizon(n: i64, lam: f64, kappa: f64) -> usize {
    let c = wave_speed();
    let s = damped_string(n, lam, kappa);
    let f_disc: Vec<f64> = (1..n)
        .map(|m| modal::discrete_stiff_mode_frequency(c, L_DEFAULT, n, kappa, s.k, m, s.theta))
        .collect();
    let f_cont = modal::stiff_harmonic_frequencies(c, L_DEFAULT, kappa, (n - 1) as usize);
    pitch_horizon(&f_disc, &f_cont, CENTS).unwrap().0
}

fn ideal_horizon(n: i64, lam: f64) -> usize {
    let c = wave_speed();
    let f_disc: Vec<f64> = (1..n)
        .map(|m| modal::discrete_mode_frequency(c, L_DEFAULT, n, lam, m))
        .collect();
    let f_cont: Vec<f64> = (1..n).map(|m| m as f64 * c / (2.0 * L_DEFAULT)).collect();
    pitch_horizon(&f_disc, &f_cont, CENTS).unwrap().0
}

/// The BUILT membrane's `(f_discrete, f_continuum)` over `modes`, from its own `h`, `c` and `k`.
fn built_membrane_frequencies(
    mem: &membrane::Params,
    n: i64,
    modes: &[(i64, i64)],
) -> (Vec<f64>, Vec<f64>) {
    let f_disc = modal::rectangular_discrete_eigenvalues(mem.h, n, n, modes)
        .into_iter()
        .map(|e| modal::discrete_membrane_eigenfrequency(e, mem.c, mem.k))
        .collect();
    let f_cont = modal::rectangular_membrane_freqs(mem.c, 1.0, 1.0, modes);
    (f_disc, f_cont)
}

fn membrane_family_horizon(n: i64, lam: f64, kind: &str) -> usize {
    let mem = membrane(n, lam);
    let (d, c) = built_membrane_frequencies(&mem, n, &mode_family(kind, n - 1).unwrap());
    pitch_horizon(&d, &c, CENTS).unwrap().0
}

// -- family 1, the implicit θ-scheme: the two errors COMPOUND, so the floor is real ----------------

#[test]
fn the_theta_string_horizon_rises_with_the_sample_rate_and_then_stops() {
    // Refining `k` buys modes until it buys nothing. Three claims, none of them a number: the
    // horizon never FALLS as `k` falls, it never passes the space floor, and by the smallest `k` in
    // the sweep it has arrived there — the third is what makes the first two more than a tautology.
    // Measured at κ = 0, 2, 8: [6 .. 21] under 21, [6 .. 19] under 19, [5 .. 15] under 15.
    let lams = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
    for kappa in [0.0, 2.0, 8.0] {
        let horizons: Vec<usize> = lams
            .iter()
            .map(|&lam| string_horizon(256, lam, kappa))
            .collect();
        let floor = spatial_operator_horizon(256, kappa);
        assert!(
            horizons.windows(2).all(|w| w[1] >= w[0]),
            "kappa={kappa}: a finer timestep lost modes: {horizons:?}"
        );
        assert!(
            *horizons.iter().max().unwrap() <= floor,
            "kappa={kappa}: {horizons:?} passed the space floor {floor} — the time and space errors \
             are supposed to compound, so this would mean they cancelled"
        );
        assert!(
            horizons[6] + 1 >= floor,
            "kappa={kappa}: a 64x refinement stalled at {}, short of the floor {floor}",
            horizons[6]
        );
    }
}

#[test]
fn the_canonical_theta_string_resolves_under_a_tenth_of_its_grid_and_stops_early() {
    // The floor arrives early. At the canonical `λ = 1` the string is time-limited (11 of 256
    // modes), and by `λ = 1/8` it is within a mode of its space floor (18 against 19) — eight times
    // the sample rate roughly doubles the band and then stops. The `>= floor - 1` here is met
    // EXACTLY, as it was in the Python: an integer bar the recorded run sits on.
    let (n, kappa) = (256, 2.0);
    let at_canonical = string_horizon(n, 1.0, kappa);
    let at_eighth = string_horizon(n, 0.125, kappa);
    let floor = spatial_operator_horizon(n, kappa);
    assert!(
        at_canonical < at_eighth,
        "8x the sample rate should still be buying modes"
    );
    assert!(
        at_eighth + 1 >= floor,
        "and by then it should have essentially stopped"
    );
    assert!(
        (at_canonical as f64 / n as f64) < 0.10,
        "the canonical lambda = 1 string resolves {at_canonical}/{n} of its grid to {CENTS} cents; \
         if this ever exceeds a tenth the fixture or the oracle changed"
    );
}

#[test]
fn the_plate_is_space_limited_at_every_sample_rate_the_suite_uses() {
    // A 40x change of sample rate moves the plate's horizon by at most one mode. The plate's
    // frequency goes like the Laplacian eigenvalue, so it carries TWICE the sinc droop, and the θ
    // average adds to it rather than cancelling: the horizon is the space floor already, which is
    // why `μ` is not the knob that fixes it. Measured: 1 and 1.
    let n = 32;
    let mut horizons = Vec::new();
    for mu in [2.0, 0.05] {
        let h = 1.0 / n as f64;
        let p = plate::Params::new(&PlateSpec {
            lx: 1.0,
            ly: 1.0,
            kappa: KAPPA_PLATE,
            rho: RHO_AREAL_DEFAULT,
            fs: KAPPA_PLATE / (mu * h * h),
            n,
            sigma: 0.0,
            ..PlateSpec::default()
        })
        .expect("an admissible plate");
        let (mut f_disc, mut f_cont) = (Vec::new(), Vec::new());
        for m in 1..n {
            let lam_disc = 2.0 * modal::dirichlet_axis_eigenvalue(m as f64, p.lx, p.h);
            let lam_cont = 2.0 * (m as f64 * PI / p.lx).powi(2);
            f_disc.push(modal::discrete_plate_eigenfrequency(
                lam_disc, p.kappa, p.k, p.theta,
            ));
            f_cont.push(p.kappa * lam_cont / (2.0 * PI));
        }
        let (horizon, monotone) = pitch_horizon(&f_disc, &f_cont, CENTS).unwrap();
        assert!(monotone);
        horizons.push(horizon);
    }
    assert!(
        horizons[0].abs_diff(horizons[1]) <= 1,
        "a 40x sample-rate change moved the plate's horizon from {} to {}; the plate is supposed \
         to be space-limited",
        horizons[0],
        horizons[1]
    );
}

// -- family 2, the explicit leapfrog: the cancellation, and only on the diagonal ---------------------

#[test]
fn the_membranes_cancellation_is_diagonal_only() {
    // At the 2-D Courant ceiling the diagonal modes are exact, and reading only that family would
    // say "the membrane is in tune at `λ = 1/√2`". The axial modes are not: they sit near the space
    // floor, a factor of eight or nine below. A margin measured on one mode family is a claim about
    // that family. The hand-picked bars are kept beside the derived ones (a derived bar without a
    // floor under it asserts less than the literal it replaced), and both derived bars are LOSSLESS
    // claims — the membrane is built with `σ = 0`.
    for n in [64, 128] {
        let diagonal = membrane_family_horizon(n, mem_ceiling(), "diagonal");
        let axial = membrane_family_horizon(n, mem_ceiling(), "axial");
        let family = (n - 1) as usize;
        assert!(
            diagonal as f64 >= 0.9 * family as f64,
            "the diagonal family should be essentially exact at the ceiling, got {diagonal}/{family}"
        );
        assert!(
            (axial as f64) < 0.25 * family as f64,
            "the axial family should NOT be, got {axial}/{family}"
        );
        assert!(
            diagonal > 5 * axial,
            "diagonal {diagonal} against axial {axial}"
        );
        // Derived: the diagonal's cancellation at the ceiling is an identity at every N, so the
        // horizon IS the family.
        assert_eq!(
            diagonal, family,
            "the diagonal horizon at the ceiling is the family"
        );
        // And the axial family gets `√2` times the string's space floor, to leading order in `1/N`,
        // never above it: higher-order terms can only add droop. Measured 7 under 7.58, 15 under
        // 15.17.
        let predicted = 2.0_f64.sqrt() * sinc_horizon_fraction(CENTS, 1).unwrap() * n as f64;
        let floor = predicted.floor() as usize;
        assert!(
            axial <= floor,
            "the axial horizon {axial} is above its closed form {predicted:.3}"
        );
        assert!(
            axial + 1 >= floor,
            "the axial horizon {axial} is more than one mode below its closed form {predicted:.3}"
        );
    }
}

#[test]
fn both_membrane_families_converge_to_the_same_space_floor() {
    // Away from the magic number the cancellation is gone and only the grid is left: the space floor
    // is where EVERY scheme ends up, and what distinguishes the families is only whether a Courant
    // number exists that beats it. Measured 10, 10 and the floor 10.
    let n = 128;
    let diagonal = membrane_family_horizon(n, 0.125, "diagonal");
    let axial = membrane_family_horizon(n, 0.125, "axial");
    let floor = spatial_operator_horizon(n, 0.0);
    assert!(
        diagonal.abs_diff(axial) <= 1,
        "far below the ceiling the two families should agree: {diagonal} against {axial}"
    );
    assert!(
        diagonal.abs_diff(floor) <= 2,
        "and both should sit on the 1-D space floor {floor}, got {diagonal}"
    );
}

#[test]
fn the_analytic_membrane_path_agrees_with_a_built_membrane() {
    // The seam the analysis file's membrane section rests on, asserted rather than assumed: that file
    // forms `k = λh/c` directly, the built membrane round-trips through `fs = c/(λh)`. They differ in
    // the last few bits of `k` (4.6874999999999994e-5 against 4.6875e-5 here) and by 3.9e-13 cents.
    let (n, lam) = (64, 0.6);
    let modes = physsynth_analysis::horizon::mode_block(6).unwrap();
    let mem = membrane(n, lam);
    let (d, c) = built_membrane_frequencies(&mem, n, &modes);
    let built = pitch_error_cents(&d, &c).unwrap();

    let c0 = (T_DEFAULT / RHO_AREAL_DEFAULT).sqrt();
    let h = L_DEFAULT / n as f64;
    let k = lam * h / c0;
    let analytic_d: Vec<f64> = modal::rectangular_discrete_eigenvalues(h, n, n, &modes)
        .into_iter()
        .map(|e| modal::discrete_membrane_eigenfrequency(e, c0, k))
        .collect();
    let analytic_c = modal::rectangular_membrane_freqs(c0, L_DEFAULT, L_DEFAULT, &modes);
    let analytic = pitch_error_cents(&analytic_d, &analytic_c).unwrap();
    let gap = built
        .iter()
        .zip(&analytic)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        gap < 1e-9,
        "the analytic path and a built membrane disagree by {gap:e}, more than the round trip \
         through fs"
    );
}

#[test]
fn the_string_is_the_same_formula_with_the_second_axis_dropped() {
    // Drop the second axis and `√(m⁴)/m²` is exactly 1 for every mode — the 1-D CFL limit. So in 1-D
    // the whole spectrum cancels at one Courant number and the string resolves its entire grid; in
    // 2-D only the diagonal attains the minimum, and the axial family's cancellation number lies
    // past the stability bound. Measured: 255 against the membrane's axial 30.
    for m in 1..200 {
        assert_eq!(
            cancellation_courant(m, 0).unwrap(),
            1.0,
            "the 1-D case is not exactly 1 at m={m}"
        );
    }
    let horizon = ideal_horizon(256, 1.0);
    assert!(
        horizon >= 254,
        "the 1-D cancellation should take the whole grid, got {horizon}"
    );
    let axial_at_ceiling = membrane_family_horizon(256, mem_ceiling(), "axial");
    assert!(
        (axial_at_ceiling as f64) < 0.2 * horizon as f64,
        "the 2-D axial family cannot reach its own cancellation number ({:.4} > {:.4}), so it \
         should get far less than the string's {horizon}; got {axial_at_ceiling}",
        cancellation_courant(64, 1).unwrap(),
        mem_ceiling()
    );
}
