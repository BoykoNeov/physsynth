//! The acoustic bore's acceptance harness at the Python suite's own rig (retirement plan §42).
//!
//! `tests/test_bore_{energy,modal,stability,radiation}.py` were the air column's validation suite;
//! this file is where their assertions went when they were deleted. It runs at `make_bore`'s rig —
//! a 0.5 m cylinder of radius 8 mm in ambient air (`rho0 = 1.2041`, `c0 = 343`), the sample rate
//! solved for the Courant number as `fs = c0 / (lam h)` — with the Python's Gaussian pressure bump
//! `1e-3 exp(-(x - c)^2 / (2 w^2))`. That is **not** `bore.rs`'s rig (a 0.6 m tube and a narrower,
//! unit bump), so that file keeps its own bars and this one is new, as the stiff string's was (§32).
//!
//! Four groups, in the order the Python argued them:
//!
//! - **Energy** — lossless, the discrete energy is flat to machine precision across the Courant
//!   range and every boundary combination; viscous, it falls monotonically and a single mode decays
//!   at the analytic `2 sigma` rate; it scales with the cross-section, as Joules must.
//! - **Resonances** — a closed-open tube is odd-harmonic only and an open-open one carries the full
//!   series; the operator's eigenvalues are the dispersionless continuum at `lam = 1`, track the
//!   measured spectrum, and converge at second order below it.
//! - **Stability** — nothing blows up anywhere in the valid range, `lam > 1` is refused, and so are
//!   non-physical parameters.
//! - **The radiating bell** — the field energy plus what the bell booked is conserved for every
//!   placement, a single reflection sheds exactly `1 - r^2`, a matched load is anechoic, the two
//!   limits of `R` are the open and the closed end, and the far-field read-out is clean.
//!
//! **The eigenvalue referee is a closed form, stronger than the solver the Python used.** The
//! Python read the resonances off SciPy's ARPACK. Recorded before deletion (§42.1), the free-node
//! problem `L phi = omega^2 C phi` has the exact spectrum
//! `omega_n^2 = (2 c0 / h)^2 sin^2((2n - 1) pi / 4N)` for the closed-open tube: ARPACK matched it
//! to 2e-15 relative on every rig here, LAPACK's dense solve to within 0.7 `eps omega_max^2`. So the
//! native solver is certified against that formula, written in the test, and the Python's bars are
//! then carried on the native solver's output.

use physsynth_analysis::modal;
use physsynth_analysis::radiation::piston_radiation_resistance;
use physsynth_analysis::spectrum;
use physsynth_core::bore::{Bore, End, ParamError, Params};
use physsynth_core::eig;
use physsynth_core::engine::{simulate, Resonator};
use physsynth_core::radiation::{AirParams, AirRadiation};
use physsynth_core::reduce;

const L: f64 = 0.5; // `BORE_LENGTH_DEFAULT`
const RADIUS: f64 = 0.008; // `BORE_RADIUS_DEFAULT`
const RHO0: f64 = 1.2041; // `RHO0_AIR`
const C0: f64 = 343.0; // `C0_AIR`

/// The lossless acceptance bar — the Python's (and every resonator's).
const DRIFT_TOL: f64 = 1e-10;

const CO: (End, End) = (End::Closed, End::Open);
const ALL_ENDS: [(End, End); 4] = [
    (End::Closed, End::Open),
    (End::Open, End::Open),
    (End::Closed, End::Closed),
    (End::Open, End::Closed),
];

/// `make_bore` / `make_radiating_bore`: the tube at Courant number exactly `lam`.
fn params(n: usize, lam: f64, bc: (End, End), sigma: f64, r_bell: f64, radius: f64) -> Params {
    let h = L / (n as f64);
    let fs = C0 / (lam * h);
    Params::new(L, fs, n, radius, Some(bc), sigma, r_bell, RHO0, C0)
        .expect("parameters should be accepted")
}

fn plain(n: usize, lam: f64, bc: (End, End)) -> Params {
    params(n, lam, bc, 0.0, 0.0, RADIUS)
}

fn bell(n: usize, lam: f64, bc: (End, End), r_bell: f64) -> Params {
    params(n, lam, bc, 0.0, r_bell, RADIUS)
}

/// The Python's `_bump`: `amplitude * np.exp(-((x - c) ** 2) / (2.0 * w * w))`, `c` and `w` as
/// fractions of the length. NumPy's `exp` and the platform's agreed on every entry of every bump
/// here (recorded, §42.1), so the initial conditions are the Python's to the bit on Windows.
fn bump(p: &Params, center_frac: f64, width_frac: f64) -> Vec<f64> {
    let c = center_frac * p.l;
    let w = width_frac * p.l;
    p.grid()
        .iter()
        .map(|&x| {
            let d = x - c;
            1e-3 * (-(d * d) / ((2.0 * w) * w)).exp()
        })
        .collect()
}

/// A bore at rest except for `p0` — the Python's `set_state(p0)` (`U^{-1/2} = 0`).
fn started(p: &Params, p0: &[f64]) -> Bore {
    let mut b = Bore::new(p.clone());
    b.set_state(p0, &vec![0.0; p.n]);
    b
}

/// `int(secs * bore.fs)`.
fn steps(p: &Params, secs: f64) -> usize {
    (secs * p.fs) as usize
}

/// `simulate(bore, num_steps, pickup_index)`, which cannot fail for a bore.
fn run(b: &mut Bore, n: usize, pickup: usize) -> physsynth_core::engine::SimResult {
    simulate(b, n, Some(pickup), 0).expect("an undriven bore cannot fail a step")
}

fn output(r: &physsynth_core::engine::SimResult) -> &[f64] {
    r.output.as_deref().expect("a pickup was requested")
}

/// `np.max` over an iterator: NaN-propagating, so a run that blew up cannot report a clean figure.
fn nan_max(it: impl Iterator<Item = f64>) -> f64 {
    it.fold(f64::NEG_INFINITY, |m, x| {
        if x.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(x)
        }
    })
}

fn max_abs(xs: &[f64]) -> f64 {
    nan_max(xs.iter().map(|x| x.abs()))
}

// -- the eigenvalue oracle ----------------------------------------------------------------------

/// `omega^2` on the free nodes, ascending, and the eigenvectors (column `j` of a row-major `m x m`).
///
/// `L` restricted to [`Params::dof`], against the diagonal compliance `C` — the generalized
/// problem `tests/helpers.bore_low_eigenfrequencies` handed to `eigsh`, solved densely.
fn free_spectrum(p: &Params) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let dof = p.dof();
    let m = dof.len();
    let (lop, _) = p.pressure_operator();
    let mut a = vec![0.0; m * m];
    for (r, &i) in dof.iter().enumerate() {
        for (c, &j) in dof.iter().enumerate() {
            a[r * m + c] = lop.get(i, j);
        }
    }
    let d: Vec<f64> = dof.iter().map(|&i| p.c[i]).collect();
    let (w2, vecs) = eig::generalized_eigen_diag(&a, &d, m).expect("the free problem is definite");
    (w2, vecs, dof)
}

/// The `n_modes` lowest discrete resonance frequencies (Hz), through the leapfrog's dispersion —
/// `bore_low_eigenfrequencies` for a tube with an open end.
fn low_frequencies(p: &Params, n_modes: usize) -> Vec<f64> {
    let (w2, _, _) = free_spectrum(p);
    w2[..n_modes]
        .iter()
        .map(|&w| modal::discrete_bore_eigenfrequency(w, p.k))
        .collect()
}

/// The closed-open tube's exact discrete spectrum, `(2 c0 / h)^2 sin^2((2n - 1) pi / 4N)`.
fn closed_open_omega2(p: &Params, n: usize) -> f64 {
    let s = (((2 * n - 1) as f64) * std::f64::consts::PI / (4.0 * p.n as f64)).sin();
    let g = 2.0 * p.c0 / p.h;
    (g * g) * (s * s)
}

/// The rigs whose spectra the Python read: `(N, lam, modes)`.
const SPECTRUM_RIGS: [(usize, f64, usize); 7] = [
    (200, 1.0, 6),
    (150, 0.7, 5),
    (50, 0.7, 3),
    (100, 0.7, 3),
    (200, 0.7, 3),
    (400, 0.7, 3),
    (150, 0.8, 1),
];

#[test]
fn the_free_spectrum_is_the_closed_form_and_the_solver_finds_it() {
    // The operator AND the solver in one bar: every eigenvalue the Python read off ARPACK, at every
    // rig it read one, is the closed form to within the dense solve's own floor. Recorded, LAPACK's
    // `eigh(A, B)` sat within 0.7 `eps omega_max^2` of it and ARPACK within 2e-15 relative.
    let eps = f64::EPSILON;
    for (n, lam, modes) in SPECTRUM_RIGS {
        let p = plain(n, lam, CO);
        let (w2, _, _) = free_spectrum(&p);
        let top = w2[w2.len() - 1];
        let mut worst = 0.0_f64;
        for (i, &w) in w2[..modes].iter().enumerate() {
            let exact = closed_open_omega2(&p, i + 1);
            let units = (w - exact).abs() / (eps * top);
            assert!(
                units < 4.0,
                "N={n} lam={lam} mode {}: {units:.2} eps*w_max off",
                i + 1
            );
            worst = worst.max(units);
        }
        println!("N={n} lam={lam}: worst {worst:.3} eps*w_max");
    }
}

#[test]
fn the_spectrum_follows_the_sound_speed_and_the_density_off_their_defaults() {
    // §42.4: every carried bar ran in ambient air, where a compliance written with the default
    // sound speed, or an inductance with the default density, is the right number. Off them the
    // first changes the wave speed and the second moves it by `sqrt(rho0 / 1.2041)`, and the tube
    // still conserves its energy perfectly — only its pitch is wrong. The closed form reads the
    // tube's own `c0`; its `rho0` must drop out.
    let eps = f64::EPSILON;
    for (rho0, c0) in [(1.18, 343.0), (1.2041, 340.0), (1.18, 340.0)] {
        let h = L / 150.0;
        let p = Params::new(L, c0 / (0.7 * h), 150, RADIUS, Some(CO), 0.0, 0.0, rho0, c0).unwrap();
        let (w2, _, _) = free_spectrum(&p);
        let top = w2[w2.len() - 1];
        for (i, &w) in w2[..5].iter().enumerate() {
            let exact = closed_open_omega2(&p, i + 1);
            let units = (w - exact).abs() / (eps * top);
            assert!(
                units < 4.0,
                "rho0={rho0} c0={c0} mode {}: {units:.2} eps*w_max off",
                i + 1
            );
        }
    }
}

// -- energy (`test_bore_energy.py`) ---------------------------------------------------------------

#[test]
fn a_lossless_tube_conserves_energy_across_the_courant_range() {
    // Algebraic conservation, not a lambda = 1 special case.
    for lam in [1.0, 0.99, 0.9, 0.7, 0.5] {
        let p = plain(200, lam, CO);
        let mut b = started(&p, &bump(&p, 0.3, 0.08));
        let r = run(&mut b, steps(&p, 0.3), 1);
        let drift = r.energy_drift();
        println!(
            "lam={lam}: drift {drift:e}, E0 {:e}, out_last {:e}",
            r.energy[0],
            output(&r)[r.energy.len() - 1]
        );
        assert!(drift < DRIFT_TOL, "drift {drift:.2e} at lambda={lam}");
    }
}

#[test]
fn a_lossless_tube_conserves_energy_for_every_boundary_combination() {
    // The closed wall (half-cell) and the open end (pressure-release pin) are both lossless.
    for bc in ALL_ENDS {
        let p = plain(200, 0.9, bc);
        let mut b = started(&p, &bump(&p, 0.3, 0.08));
        let r = run(&mut b, steps(&p, 0.3), 1);
        let drift = r.energy_drift();
        println!(
            "{bc:?}: drift {drift:e}, E_last {:e}",
            r.energy[r.energy.len() - 1]
        );
        assert!(drift < DRIFT_TOL, "{bc:?} drift {drift:.2e}");
    }
}

#[test]
fn a_lossless_tube_has_strictly_positive_energy() {
    let p = plain(200, 0.9, CO);
    let mut b = started(&p, &bump(&p, 0.3, 0.08));
    let r = run(&mut b, steps(&p, 0.2), 1);
    // `e > 0.0` is false for a NaN, so a run that blew up fails here as `np.all` would.
    for (i, &e) in r.energy.iter().enumerate() {
        assert!(e > 0.0, "E^{i} = {e:e} is not positive");
    }
    println!(
        "min E {:e}",
        r.energy.iter().copied().fold(f64::INFINITY, f64::min)
    );
}

#[test]
fn a_viscous_tube_loses_energy_monotonically() {
    let p = params(200, 0.9, CO, 40.0, 0.0, RADIUS);
    let mut b = started(&p, &bump(&p, 0.3, 0.08));
    let r = run(&mut b, steps(&p, 0.3), 1);
    let e0 = r.energy[0];
    let worst = nan_max(r.energy.windows(2).map(|w| w[1] - w[0]));
    println!(
        "max step / E0 {:e}, final / E0 {:e}",
        worst / e0,
        r.energy[r.energy.len() - 1] / e0
    );
    assert!(
        worst <= 1e-10 * e0,
        "max positive step {:.2e} * E0",
        worst / e0
    );
}

#[test]
fn a_single_mode_decays_at_twice_sigma() {
    // The viscous term damps the kinetic half; over a cycle the loss averages to `2 sigma` of the
    // total. The mode is the operator's own lowest eigenvector (the Python's came from ARPACK; the
    // ratio below does not care which solver, sign or normalization made it — recorded, ARPACK's
    // and LAPACK's vectors gave the same ratio to 3e-16).
    let sigma = 40.0;
    let secs = 0.15;
    let p = params(150, 0.8, CO, sigma, 0.0, RADIUS);
    let (_, vecs, dof) = free_spectrum(&p);
    let m = dof.len();
    let mut phi = vec![0.0; p.nodes()];
    for (r, &i) in dof.iter().enumerate() {
        phi[i] = vecs[r * m] * 1e-3;
    }
    let mut b = started(&p, &phi);
    let r = simulate(&mut b, steps(&p, secs), None, 0).unwrap();
    let measured = r.energy[r.energy.len() - 1] / r.energy[0];
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    println!("measured {measured:e}, expected {expected:e}, rel {rel:e}");
    assert!(
        rel < 0.02,
        "decay rate off by {:.3}% (got {measured:.3e}, want {expected:.3e})",
        100.0 * rel
    );
}

#[test]
fn the_energy_scales_with_the_cross_section() {
    // Joules: at a fixed pressure field the energy is pure compliance `~ S`, so doubling the area
    // doubles it. The Python wrote `rtol=1e-12` and inherited `np.isclose`'s `atol=1e-8` with it,
    // which was the bar that bit; measured, the ratio is 2 to two ulps, so this asserts the
    // relative claim the Python wrote down.
    let p1 = params(120, 0.9, CO, 0.0, 0.0, 0.008);
    let p2 = params(120, 0.9, CO, 0.0, 0.0, 0.008 * 2.0_f64.sqrt());
    let b1 = started(&p1, &bump(&p1, 0.3, 0.08));
    let b2 = started(&p2, &bump(&p2, 0.3, 0.08));
    let ratio = b2.energy() / b1.energy();
    println!("ratio {ratio:e}, minus 2 {:e}", ratio - 2.0);
    assert!(
        (ratio - 2.0).abs() <= 1e-12 * 2.0,
        "E ratio {ratio} for twice the area"
    );
}

// -- resonances (`test_bore_modal.py`) -----------------------------------------------------------

#[test]
fn the_pressure_operator_is_symmetric() {
    // `L = G^T M^-1 G` by construction, which is what makes the energy telescope.
    let p = plain(120, 1.0, CO);
    let (lop, _) = p.pressure_operator();
    let nodes = p.nodes();
    let asym = nan_max((0..nodes).flat_map(|i| {
        let lop = &lop;
        (0..nodes).map(move |j| (lop.get(i, j) - lop.get(j, i)).abs())
    }));
    println!("max |L - L^T| = {asym:e}");
    assert!(asym < 1e-12, "L not symmetric: {asym:.3e}");
}

/// Largest magnitude within two bins of `f` — the Python's `peak_near`, window `[i-2, i+3)`.
fn peak_near(mag: &[f64], df: f64, f: f64, half: usize) -> f64 {
    let i = (f / df).round_ties_even() as usize;
    let lo = 1.max(i.saturating_sub(half));
    nan_max(mag[lo..(i + half + 1).min(mag.len())].iter().copied())
}

/// `(odd, even)` peaks at the odd and even multiples of `f1`, `count` of each.
fn odd_even(sig: &[f64], fs: f64, f1: f64, count: usize) -> (Vec<f64>, Vec<f64>) {
    let s = spectrum::magnitude_spectrum(sig, fs, 2);
    let df = s.freqs[1] - s.freqs[0];
    let odd = (1..=count)
        .map(|n| peak_near(&s.mag, df, (2 * n - 1) as f64 * f1, 2))
        .collect();
    let even = (1..=count)
        .map(|n| peak_near(&s.mag, df, (2 * n) as f64 * f1, 2))
        .collect();
    (odd, even)
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_of(v: &[f64]) -> f64 {
    nan_max(v.iter().copied())
}

#[test]
fn a_closed_open_tube_is_odd_harmonic_only() {
    // The money test: a bump near the closed wall rings on f1, 3 f1, 5 f1 ... and the even
    // multiples sit at the noise floor — a gap of orders of magnitude, not a threshold.
    let p = plain(256, 1.0, CO);
    let mut b = started(&p, &bump(&p, 0.12, 0.06));
    let r = run(&mut b, steps(&p, 0.5), 1);
    let f1 = modal::bore_resonance_frequencies(C0, L, 1, "closed-open").unwrap()[0];
    let (odd, even) = odd_even(output(&r), r.fs, f1, 5);
    let ratio = min_of(&odd) / max_of(&even);
    println!("odd {odd:?}\neven {even:?}\nratio {ratio:e}");
    assert!(
        ratio > 1e3,
        "even harmonics not suppressed: min(odd)/max(even) = {ratio:.2e}"
    );
}

#[test]
fn an_open_open_tube_carries_the_full_series() {
    // lambda = 1 is dispersionless, so there is no horizon to read; the claim is the PRESENCE of the
    // even harmonics, on pitch.
    let p = plain(256, 1.0, (End::Open, End::Open));
    let mut b = started(&p, &bump(&p, 0.23, 0.05));
    let r = run(&mut b, steps(&p, 0.5), p.n / 3);
    let oracle = modal::bore_resonance_frequencies(C0, L, 4, "open-open").unwrap();
    let found = spectrum::measure_partials_near(output(&r), r.fs, &oracle, None);
    let err: Vec<f64> = found
        .iter()
        .zip(&oracle)
        .map(|(&f, &o)| modal::cents(f, o))
        .collect();
    println!("found {found:?}\ncents {err:?}");
    for (i, e) in err.iter().enumerate() {
        assert!(
            e.abs() < 2.0,
            "open-open partial {} off by {e} cents (want < 2)",
            i + 1
        );
    }
}

#[test]
fn at_lambda_one_the_discrete_resonances_are_the_continuum() {
    let p = plain(200, 1.0, CO);
    let fd = low_frequencies(&p, 6);
    let fc = modal::bore_resonance_frequencies(C0, L, 6, "closed-open").unwrap();
    let worst = nan_max(fd.iter().zip(&fc).map(|(&a, &b)| modal::cents(a, b).abs()));
    println!("fdisc {fd:?}\nworst {worst:e} cents");
    assert!(
        worst < 0.05,
        "lambda = 1 not dispersionless: up to {worst:.3} cents"
    );
}

#[test]
fn the_discrete_resonances_track_the_measured_spectrum() {
    let p = plain(150, 0.7, CO);
    let fd = low_frequencies(&p, 5);
    let mut b = started(&p, &bump(&p, 0.1, 0.06));
    let r = run(&mut b, steps(&p, 0.6), 1);
    let found = spectrum::measure_partials_near(output(&r), r.fs, &fd, None);
    let err: Vec<f64> = found
        .iter()
        .zip(&fd)
        .map(|(&f, &o)| modal::cents(f, o))
        .collect();
    println!("fdisc {fd:?}\nfound {found:?}\ncents {err:?}");
    let worst = nan_max(err.iter().map(|e| e.abs()));
    assert!(
        worst < 0.5,
        "measured spectrum off the oracle by up to {worst:.3} cents"
    );
}

#[test]
fn the_dispersion_error_converges_at_second_order() {
    let oracle = modal::bore_resonance_frequencies(343.0, 0.5, 3, "closed-open").unwrap();
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [50, 100, 200, 400] {
        let p = plain(n, 0.7, CO);
        let f = low_frequencies(&p, 3);
        hs.push(p.h);
        errs.push(nan_max(f.iter().zip(&oracle).map(|(&a, &b)| (a - b).abs())));
    }
    let orders: Vec<f64> = (0..3)
        .map(|i| (errs[i] / errs[i + 1]).ln() / (hs[i] / hs[i + 1]).ln())
        .collect();
    println!("errs {errs:?}\norders {orders:?}");
    for w in errs.windows(2) {
        assert!(
            w[1] - w[0] < 0.0,
            "dispersion error not decreasing: {errs:?}"
        );
    }
    assert!(orders[2] > 1.9, "convergence order {:.2} < 1.9", orders[2]);
}

// -- stability and construction (`test_bore_stability.py`) ----------------------------------------

fn finite(xs: &[f64]) -> bool {
    xs.iter().all(|x| x.is_finite())
}

#[test]
fn nothing_blows_up_across_the_valid_courant_range() {
    for lam in [0.999, 0.95, 0.9, 0.75, 0.5, 0.3, 0.1] {
        let p = plain(150, lam, CO);
        let mut b = started(&p, &bump(&p, 0.3, 0.08));
        let r = run(&mut b, steps(&p, 0.2), 1);
        println!(
            "lam={lam}: max |out| {:e}, drift {:e}",
            max_abs(output(&r)),
            r.energy_drift()
        );
        assert!(finite(output(&r)), "the pickup blew up at lambda={lam}");
        assert!(finite(&r.energy), "the energy blew up at lambda={lam}");
    }
}

#[test]
fn nothing_blows_up_for_any_boundary_combination() {
    for bc in ALL_ENDS {
        let p = plain(150, 0.9, bc);
        let mut b = started(&p, &bump(&p, 0.3, 0.08));
        let r = run(&mut b, steps(&p, 0.2), 1);
        println!("{bc:?}: max |out| {:e}", max_abs(output(&r)));
        assert!(finite(output(&r)) && finite(&r.energy), "{bc:?} blew up");
    }
}

#[test]
fn a_courant_number_above_one_is_refused_and_one_is_accepted() {
    // `lambda = c0 N / (fs L)`: dividing the rate by 1.05 forces lambda = 1.05.
    let n = 150;
    let fs_unstable = C0 * (n as f64) / (L * 1.05);
    let err = Params::new(L, fs_unstable, n, RADIUS, Some(CO), 0.0, 0.0, RHO0, C0).unwrap_err();
    assert!(matches!(err, ParamError::CflViolated(_)));
    assert_eq!(
        err.to_string(),
        "CFL violated: lambda = c0*k/h = 1.050000 > 1. \
         Reduce fs, refine the grid (increase N), or shorten the tube."
    );

    let p = plain(150, 1.0, CO);
    println!("lam - 1 = {:e}", p.lam - 1.0);
    assert!(
        (p.lam - 1.0).abs() <= 1e-12,
        "lambda = {} at the ceiling",
        p.lam
    );
}

#[test]
fn non_physical_parameters_are_refused() {
    // The Python's eleven cases on its `L = 0.5, fs = 4e5, N = 150` base, each by the variant AND
    // by the word its message was matched on. A bad boundary token arrives as `None`: the parse
    // happens at the caller, which is the only place that can quote the caller's object.
    let build = |l: f64,
                 fs: f64,
                 n: usize,
                 radius: f64,
                 rho0: f64,
                 c0: f64,
                 sigma: f64,
                 bc: Option<(End, End)>| {
        Params::new(l, fs, n, radius, bc, sigma, 0.0, rho0, c0).unwrap_err()
    };
    let base = |l, fs, radius, rho0, c0| build(l, fs, 150, radius, rho0, c0, 0.0, Some(CO));
    for err in [
        base(0.0, 4e5, RADIUS, RHO0, C0),
        base(-1.0, 4e5, RADIUS, RHO0, C0),
        base(L, 0.0, RADIUS, RHO0, C0),
        base(L, 4e5, 0.0, RHO0, C0),
        base(L, 4e5, -0.01, RHO0, C0),
        base(L, 4e5, RADIUS, 0.0, C0),
        base(L, 4e5, RADIUS, RHO0, -1.0),
    ] {
        assert_eq!(err, ParamError::NonPositiveScalar);
        assert_eq!(
            err.to_string(),
            "L, fs, radius, rho0, c0 must all be positive."
        );
    }
    let err = build(L, 4e5, 150, RADIUS, RHO0, C0, -0.1, Some(CO));
    assert_eq!(err, ParamError::NegativeSigma);
    assert_eq!(err.to_string(), "sigma (loss) must be >= 0.");
    let err = build(L, 4e5, 1, RADIUS, RHO0, C0, 0.0, Some(CO));
    assert_eq!(err, ParamError::TooFewSegments);
    assert_eq!(
        err.to_string(),
        "N must be >= 2 (need at least one interior node)."
    );
    for token in ["rigid", "leaky"] {
        assert_eq!(End::parse(token), None, "{token} parsed");
    }
    let err = build(L, 4e5, 150, RADIUS, RHO0, C0, 0.0, None);
    assert_eq!(err, ParamError::BadBoundary);
    assert!(err
        .to_string()
        .starts_with("each boundary end must be one of"));
}

// -- the radiating bell (`test_bore_radiation.py`) -----------------------------------------------

#[test]
fn the_bell_books_exactly_what_the_field_loses_at_every_placement() {
    // A big R (near-closed) keeps the tube ringing while it radiates a little every reflection; the
    // identity must hold for any R, and both-ends is what catches a left-end sign error.
    for bc in [
        (End::Closed, End::Radiating),
        (End::Radiating, End::Closed),
        (End::Radiating, End::Radiating),
    ] {
        let p = bell(200, 0.9, bc, 1e5);
        let mut b = started(&p, &bump(&p, 0.4, 0.06));
        let r = run(&mut b, steps(&p, 0.3), 1);
        let drift = r.energy_drift();
        println!(
            "{bc:?}: drift {drift:e}, radiated {:e}",
            b.radiated_energy()
        );
        assert!(drift < DRIFT_TOL, "{bc:?} drift {drift:.2e}");
        assert!(
            b.radiated_energy() > 0.0,
            "a radiating end must shed some energy"
        );
    }
}

#[test]
fn the_field_energy_falls_as_the_bell_radiates_while_the_total_holds() {
    let p = bell(200, 0.9, (End::Closed, End::Radiating), 5e4);
    let mut b = started(&p, &bump(&p, 0.4, 0.06));
    let mut field = Vec::new();
    let mut total = Vec::new();
    for _ in 0..steps(&p, 0.2) {
        b.step(None);
        field.push(b.acoustic_energy());
        total.push(b.energy());
    }
    let f0 = field[0];
    let last = field[field.len() - 1];
    let rise = nan_max(field.windows(2).map(|w| w[1] - w[0]));
    let range = max_of(&total) - min_of(&total);
    println!(
        "field last/first {:e}, max rise / f0 {:e}, total range / t0 {:e}",
        last / f0,
        rise / f0,
        range / total[0]
    );
    assert!(
        last < 0.99 * f0,
        "the air column's energy should fall as the bell radiates"
    );
    assert!(
        rise <= 1e-12 * f0,
        "the field energy must be monotone (passive)"
    );
    assert!(
        range < DRIFT_TOL * total[0],
        "the total must stay conserved"
    );
}

#[test]
fn an_absurd_resistance_is_still_stable() {
    // No guard beyond the interior CFL: `a + b > 0` always, at any R.
    for r_bell in [1e-2, 1e12] {
        let p = bell(200, 0.9, (End::Closed, End::Radiating), r_bell);
        let mut b = started(&p, &bump(&p, 0.4, 0.06));
        let r = run(&mut b, steps(&p, 0.3), 1);
        let peak = max_abs(output(&r));
        println!("R={r_bell:e}: max |out| {peak:e}");
        assert!(finite(output(&r)), "blew up at R={r_bell:.0e}");
        assert!(
            peak <= 2.0e-3,
            "amplitude grew at R={r_bell:.0e} (should not)"
        );
    }
}

/// `Bore(L=0.5, fs=1e6, N=400, radius=...).Z0` — the reference tube the reflection bars size `R` by.
fn z0_of_the_reference_tube() -> f64 {
    Params::new(L, 1e6, 400, RADIUS, Some(CO), 0.0, 0.0, RHO0, C0)
        .unwrap()
        .z0
}

/// A centred narrow pulse in a 400-segment tube at lambda = 1, with a radiating right end.
fn reflection_rig(r_bell: f64) -> Bore {
    let p = bell(400, 1.0, (End::Closed, End::Radiating), r_bell);
    let b = bump(&p, 0.5, 0.03);
    started(&p, &b)
}

#[test]
fn one_reflection_sheds_one_minus_r_squared() {
    // The teeth: an oracle independent of the coupling. At lambda = 1 the stationary pulse splits
    // into two exact halves; the right one hits the bell once and sheds `1 - r^2` of its E0/2, the
    // left one bounces off the wall losslessly and has not come back after N steps.
    let z0 = z0_of_the_reference_tube();
    for frac in [0.1, 0.3, 1.0, 3.0, 10.0] {
        let r_bell = frac * z0;
        let mut b = reflection_rig(r_bell);
        let e0 = b.energy();
        let n = b.params().n;
        run(&mut b, n, 1);
        let z = b.params().z0;
        let r = (r_bell - z) / (r_bell + z);
        let oracle = 0.5 * (1.0 - r * r);
        let measured = b.radiated_energy() / e0;
        println!(
            "R/Z0={frac}: shed {measured:e}, oracle {oracle:e}, rel {:e}",
            (measured - oracle).abs() / oracle
        );
        assert!(
            (measured - oracle).abs() < 0.02 * oracle,
            "R/Z0={frac}: shed {measured:.4} of E0, oracle {oracle:.4}"
        );
    }
}

#[test]
fn a_matched_load_is_anechoic() {
    // `R = Z0` (r = 0): after both halves have reached the bell, essentially everything has left.
    let z0 = z0_of_the_reference_tube();
    let mut b = reflection_rig(z0);
    let e0 = b.energy();
    let n = b.params().n;
    run(&mut b, 3 * n, 1);
    println!(
        "radiated/E0 {:e}, acoustic/E0 {:e}",
        b.radiated_energy() / e0,
        b.acoustic_energy() / e0
    );
    assert!(
        b.radiated_energy() > 0.97 * e0,
        "the anechoic load absorbed only {:.3}",
        b.radiated_energy() / e0
    );
    assert!(
        b.acoustic_energy() < 0.03 * e0,
        "a matched load should leave the tube nearly silent"
    );
}

/// `min(odd) / max(even)` over four of each, at the INTERIOR pickup, for a 256-segment bell.
fn odd_even_ratio(r_bell: f64, center_frac: f64) -> (f64, Vec<f64>, Vec<f64>) {
    let p = bell(256, 1.0, (End::Closed, End::Radiating), r_bell);
    let mut b = started(&p, &bump(&p, center_frac, 0.06));
    let r = run(&mut b, steps(&p, 0.5), 1);
    let f1 = modal::bore_resonance_frequencies(C0, L, 1, "closed-open").unwrap()[0];
    let (odd, even) = odd_even(output(&r), r.fs, f1, 4);
    (min_of(&odd) / max_of(&even), odd, even)
}

#[test]
fn a_realistic_bell_stays_an_odd_harmonic_clarinet() {
    // R << Z0: the interior spectrum is clean, the end node's cosmetic ripple does not leak in.
    let (ratio, odd, even) = odd_even_ratio(650.0, 0.1);
    println!("odd {odd:?}\neven {even:?}\nratio {ratio:e}");
    assert!(
        ratio > 1e3,
        "open-ish bell not odd-harmonic: min-odd/max-even = {ratio:.2e}"
    );
}

#[test]
fn a_rigid_bell_is_a_closed_wall() {
    // R -> infinity: closed-closed, the full series, so the odd-only ratio collapses.
    let (ratio, odd, even) = odd_even_ratio(1e10, 0.12);
    println!("odd {odd:?}\neven {even:?}\nratio {ratio:e}");
    assert!(
        ratio < 1.0,
        "a rigid bell still suppresses the even harmonics: ratio = {ratio:.2e}"
    );
}

/// The fraction of a signal's power in the top 5% of its spectrum, as the Python measured it:
/// `np.abs(np.fft.rfft(x - x.mean()))`, the mean NumPy's pairwise one.
fn nyquist_fraction(x: &[f64]) -> f64 {
    let mean = reduce::sum(x) / x.len() as f64;
    let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let (re, im) = spectrum::rfft(&centred);
    let power: Vec<f64> = re.iter().zip(&im).map(|(a, b)| a * a + b * b).collect();
    let cut = (0.95 * power.len() as f64) as usize;
    power[cut..].iter().sum::<f64>() / power.iter().sum::<f64>()
}

#[test]
fn the_far_field_readout_is_clean() {
    // The read-out is the bell's volume acceleration, whose Nyquist part has cancelled; the raw
    // heavily-damped end node carries the cosmetic ripple.
    let p = bell(256, 1.0, (End::Closed, End::Radiating), 650.0);
    let mut b = started(&p, &bump(&p, 0.1, 0.06));
    let mut far = Vec::new();
    let mut raw = Vec::new();
    for _ in 0..steps(&p, 0.3) {
        b.step(None);
        far.push(b.radiated_pressure());
        raw.push(b.p()[p.n]);
    }
    let ff = nyquist_fraction(&far);
    let fr = nyquist_fraction(&raw);
    println!("far {ff:e}, raw {fr:e}");
    assert!(
        ff < 1e-2,
        "the far-field read-out is not clean: Nyquist fraction {ff:.2e}"
    );
    assert!(
        ff < 0.5 * fr,
        "the read-out should be cleaner than the raw end node"
    );
}

#[test]
fn the_far_field_readout_is_the_rate_of_the_bell_flow() {
    // The read-out is a volume ACCELERATION: integrated over a run it gives back the bell's
    // outgoing volume velocity, which starts at zero — at EVERY step, not only the last. The
    // first draft compared the two at the end of the run, where the flow has died back to ~2e-10
    // of its peak, so a read-out with the wrong sign, without its `/ k` or doubled also matched
    // (§42.8). Compared step by step, while the flow is large, the sign and the scale are pinned.
    // A read-out that stopped differencing stays clean and nonzero, so neither neighbouring bar
    // sees any of these; only the viewer's Windows-exact freeze saw the first (§42.4).
    let p = bell(200, 1.0, (End::Closed, End::Radiating), 650.0);
    let mut b = started(&p, &bump(&p, 0.3, 0.06));
    let mut integral = 0.0;
    let mut peak: f64 = 0.0;
    let mut worst: f64 = 0.0;
    for _ in 0..steps(&p, 0.05) {
        b.step(None);
        integral += b.radiated_pressure() * p.k;
        peak = peak.max(b.u_out().abs());
        let err = (integral - b.u_out()).abs();
        worst = if err.is_nan() || worst.is_nan() {
            f64::NAN
        } else {
            worst.max(err)
        };
    }
    println!(
        "peak U_out {peak:e}, worst |sum - U_out| / peak {:e}",
        worst / peak
    );
    assert!(peak > 0.0, "the bell never moved any air");
    assert!(
        worst <= 1e-9 * peak,
        "the integrated read-out strays from U_out by {:e} of its peak",
        worst / peak
    );
}

#[test]
fn the_bell_drives_the_far_field_microphone() {
    // The read-out is exactly what the free-field radiator consumes: a delayed, nonzero pressure
    // once the wavefront reaches the listener 1 m away.
    let p = bell(200, 1.0, (End::Closed, End::Radiating), 650.0);
    let mut b = started(&p, &bump(&p, 0.3, 0.06));
    let mut air = AirRadiation::new(AirParams::new(p.fs, 1.0, RHO0, C0, true).unwrap());
    let mut out = Vec::new();
    for _ in 0..steps(&p, 0.05) {
        b.step(None);
        out.push(air.process(b.radiated_pressure()));
    }
    let first = out.iter().position(|&v| v != 0.0);
    println!("max |out| {:e}, first nonzero {first:?}", max_abs(&out));
    assert!(finite(&out));
    assert!(
        max_abs(&out) > 0.0,
        "the radiating bell should drive the far-field microphone"
    );
}

#[test]
fn a_physical_bell_radiates_lightly() {
    // The baffled-piston resistance at the clarinet's fundamental, for this bore's radius: a small
    // end radiates weakly, `R << Z0`.
    let p = plain(200, 1.0, CO);
    let f1 = modal::bore_resonance_frequencies(C0, p.l, 1, "closed-open").unwrap()[0];
    let r = piston_radiation_resistance(2.0 * std::f64::consts::PI * f1, p.radius, RHO0, C0);
    println!("R {r:e}, Z0 {:e}, R/Z0 {:e}", p.z0, r / p.z0);
    assert!(r > 0.0 && r < 1e-3 * p.z0, "R/Z0 = {:.2e}", r / p.z0);
}

#[test]
fn the_bells_refusals_carry_the_originals_words() {
    let build = |bc, r_bell| Params::new(0.5, 1e6, 200, RADIUS, bc, 0.0, r_bell, RHO0, C0);
    let err = build(Some((End::Closed, End::Radiating)), 0.0).unwrap_err();
    assert_eq!(err, ParamError::RadiatingNeedsResistance);
    assert!(err.to_string().contains("radiating"));
    let err = build(Some(CO), -1.0).unwrap_err();
    assert_eq!(err, ParamError::NegativeRBell);
    assert!(err.to_string().contains("R_bell"));
    assert_eq!(End::parse("flared"), None);
    let err = build(None, 0.0).unwrap_err();
    assert_eq!(err, ParamError::BadBoundary);
    assert!(err.to_string().contains("each boundary end"));
}

#[test]
fn a_resistance_without_a_radiating_end_changes_nothing() {
    // The Python compared its default `R_bell` with an explicit `0.0`, a claim about a keyword
    // default that has no native analogue. The claim under it is sharper here: a closed-open bore
    // given a NONZERO resistance it has no radiating end to use is the plain bore bit for bit.
    let a = plain(200, 0.9, CO);
    let b = bell(200, 0.9, CO, 650.0);
    let mut ba = started(&a, &bump(&a, 0.4, 0.06));
    let mut bb = started(&b, &bump(&b, 0.4, 0.06));
    let ra = run(&mut ba, 2000, 1);
    let rb = run(&mut bb, 2000, 1);
    assert_eq!(output(&ra), output(&rb));
    assert_eq!(ba.p(), bb.p());
    assert_eq!(ba.u(), bb.u());
    assert_eq!(bb.radiated_energy(), 0.0);
    println!("out_last {:e}", output(&ra)[2000]);
}

#[test]
fn the_bore_is_a_resonator() {
    // Through the engine's trait, as `simulate` reads it: the state is the N + 1 pressure nodes,
    // the pickup is the state's entry, and the timestep is the parameters'.
    let p = plain(200, 0.9, CO);
    let b = started(&p, &bump(&p, 0.4, 0.06));
    let r: &dyn Resonator = &b;
    assert_eq!(r.state().len(), p.nodes());
    assert_eq!(r.displacement_at(10), r.state()[10]);
    assert_ne!(r.displacement_at(10), 0.0);
    assert_eq!(r.timestep(), p.k);
    assert_eq!(r.energy(), b.energy());
}
