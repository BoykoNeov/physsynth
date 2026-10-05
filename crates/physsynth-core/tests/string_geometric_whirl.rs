//! The geometrically exact string's **whirling threshold** — carried from
//! `tests/test_geometric_whirl.py` (retirement plan §36).
//!
//! An isotropic string cannot whirl: `w -> -w` is a reflection symmetry, so a planar start stays
//! planar to the bit, and the rotation generator pins both Floquet multipliers at `+1` —
//! marginal, never exponential. Detune the polarizations (`kappa_u != kappa_w`) and whirling
//! becomes a **threshold** instability. Reduced to one mode pair, out-of-plane motion obeys a
//! Mathieu equation pumped at `2 Om`:
//!
//! ```text
//! dq_w'' + [ ww^2 + eps A^2/2 + (eps A^2/2) cos(2 Om t) ] dq_w = 0
//! eps = (a / 4 rho) p^4,  a = EA - T0,  Om^2 = wu^2 + (3/4) eps A^2,  delta = ww^2 - wu^2
//! unstable  <=>  0 < delta < eps A^2 / 2,   most unstable at  delta = eps A^2 / 4
//! ```
//!
//! So the whole file lives on the dimensionless coordinate `delta / (eps A^2)`: the tongue is
//! `(0, 0.5)` with its peak at `0.25`, whatever `kappa_w`, `A` or `N` are individually. Measured
//! (Windows, through the binding before the deletion, and reproduced here), the growth of the
//! out-of-plane seed over 0.06 s across that coordinate:
//!
//! ```text
//! delta/eps A^2   0.00    0.07    0.25    0.41    0.50    0.80
//! growth          1.00x  16.0x   76.3x   37.4x    8.4x    1.63x
//! ```
//!
//! **Why the exaggeration is a microscope, not a thumb on the scale.** A real whirling string is
//! detuned by under a hertz and whirls over seconds; this one is detuned by ~17 Hz at a 35 mm
//! amplitude and whirls in ~12 ms. Both are the same exaggeration: the tongue is dimensionless, so
//! holding `delta/(eps A^2)` fixed while scaling `delta` and `eps A^2` together preserves the
//! physics and only compresses the wall-clock (the growth rate `~ eps A^2 / 8 Om` buys the speed).
//!
//! **What is exact and what is a limit.** The tongue is a Kirchhoff–Carrier-limit oracle: `eps` is
//! quasi-static and `Om` leading-order Duffing, so the edges are approximate (the upper one is
//! soft: 8.4x still at `0.50`, dead by `0.80`) while the peak is sharp. The bars assert the
//! tongue's shape and scaling and report the edge rather than pinning it.
//!
//! **The discarded precession rate is the whirl growth rate.** At the tongue's centre the Mathieu
//! rate collapses to `eps A^2/(8 Om)` — the `Om_prec = eps A^2/(8 w0)` the model's plan derived by
//! naive averaging and rightly rejected for a degenerate string, under `w0 -> Om`. It does not
//! describe a degenerate string precessing; it describes how fast a detuned one whirls
//! (`docs/dev/geometrically-exact-string-plan.md`, the whirling section).
//!
//! **The honesty gate.** Every claim is about a SEEDED perturbation. An unseeded planar run at the
//! same extreme parameters keeps `max|w| == 0.0` exactly, which is what makes a growth ratio mean
//! anything.
//!
//! The parameters, and why each is what it is:
//!
//! - `N = 16`. Mode 1 carries the claim, its discrete `p^2` is within 0.3% of the continuum, and
//!   the tongue is refinement-invariant (`the_tongue_does_not_move_with_the_grid` holds it at 2x).
//! - `lam_long = 0.9`, `0.06 s`, seed `1e-3` of the driven amplitude: deep in the linear Mathieu
//!   regime at the start, and ~6 e-foldings by the end.
//! - [`TENSION_RISE`] = 1.5, tuned from both sides. Up, because the e-folding time is ~12 ms here
//!   and ~650 ms at the 4 mm the rest of the suite uses. Down, because model #9 found a PLANAR
//!   parametric instability (in-plane modal exchange) above `dT/T0 ~ 3` that also conserves
//!   energy, so the drift gate cannot tell it from whirling — only staying away from it can. At
//!   `kappa_u = 0`, `eps A^2 / wu^2 == dT/T0` exactly, which is why the driven plane is the
//!   flexible one; and every run asserts the driven field stays single-mode rather than trusting
//!   the margin.
//!
//! Nothing here is NumPy's or SciPy's own number (the Python's oracles were the analysis crate
//! through the binding, its runs the core), so nothing is frozen, as in §35. The projections are
//! left-to-right sums where the Python's `np.dot` was BLAS; growth ratios are not exact claims.

mod geometric_fixture;

use geometric_fixture::*;
use physsynth_analysis::damping::spatial_eigenvalue_p2;
use physsynth_analysis::duffing::kc_mode_coefficients;
use std::f64::consts::PI;
use std::sync::OnceLock;

const N: i64 = 16;
const LAM_LONG: f64 = 0.9;
const T_TOTAL: f64 = 0.06;
/// The out-of-plane seed, as a fraction of the driven amplitude.
const SEED_REL: f64 = 1e-3;
/// The nonlinearity strength as the single-mode tension rise `dT/T0` (see the header).
const TENSION_RISE: f64 = 1.5;
/// The sweep in `delta/(eps A^2)`: the edge, the rising flank, the predicted peak, the falling
/// flank, the predicted upper edge, and outside.
const TONGUE_FRACS: [f64; 6] = [0.0, 0.07, 0.25, 0.41, 0.5, 0.8];

/// `(p2, wu^2, eps)`: the Kirchhoff–Carrier reduction of mode 1 at `kappa_u = 0`, with the
/// DISCRETE `p^2` — the tongue is compared against a string on a grid, so the oracle carries the
/// scheme's own eigenvalue.
fn kc(n: i64) -> (f64, f64, f64) {
    let p2 = spatial_eigenvalue_p2(n, L / n as f64, 1);
    let (w0sq, eps) = kc_mode_coefficients(c(), 0.0, EA - T, RHO, p2, L).unwrap();
    (p2, w0sq, eps)
}

/// The amplitude at which `eps A^2 / wu^2 == rise == dT/T0` (exact at `kappa_u = 0`).
fn amp_for(rise: f64, n: i64) -> f64 {
    let (_, w0sq, eps) = kc(n);
    (rise * w0sq / eps).sqrt()
}

/// `ww^2 - wu^2 = kappa_w^2 (p^2)^2`: the scheme's `d_xxxx` is literally `D2 D2`.
fn delta(kappa_w: f64, n: i64) -> f64 {
    let (p2, _, _) = kc(n);
    kappa_w.powi(2) * p2.powi(2)
}

/// The `kappa_w` placing the string at `delta/(eps A^2) == frac`.
fn kappa_w_at(frac: f64, amp: f64, n: i64) -> f64 {
    let (p2, _, eps) = kc(n);
    (frac * eps * amp.powi(2) / p2.powi(2)).sqrt()
}

/// Gough's threshold amplitude `A_c = sqrt(2 delta / eps)`: the tongue's upper edge read in the
/// amplitude direction. A limit oracle, used to place runs, never asserted to cents.
fn a_crit(kappa_w: f64, n: i64) -> f64 {
    let (_, _, eps) = kc(n);
    (2.0 * delta(kappa_w, n) / eps).sqrt()
}

/// The Mathieu growth rate `(Om/2) sqrt(qM^2 - sigma^2)`, zero outside the tongue: `qM` is the
/// pump's strength and `sigma` the detuning from the principal resonance.
fn mathieu_rate(kappa_w: f64, amp: f64, n: i64) -> f64 {
    let (_, w0sq, eps) = kc(n);
    let ea2 = eps * amp.powi(2);
    let om = (w0sq + 0.75 * ea2).sqrt();
    let qm = ea2 / (4.0 * om.powi(2));
    let sigma = (delta(kappa_w, n) - ea2 / 4.0) / om.powi(2);
    (om / 2.0) * (qm.powi(2) - sigma.powi(2)).max(0.0).sqrt()
}

/// The exponential rate from the LAST two quarter-envelopes, past the seed's transient (the seed is
/// not the growing Floquet mode, so the first quarter carries the decaying partner).
fn growth_rate(quarters: &[f64; 4]) -> f64 {
    4.0 * (quarters[3] / quarters[2]).ln() / T_TOTAL
}

#[derive(Clone, Copy, PartialEq)]
enum Drive {
    /// The soft plane (`kappa_u = 0`), seeding the stiff one.
    U,
    /// The same string with the roles swapped: the `delta < 0` half of Gough's asymmetry.
    W,
}

#[derive(Clone, Copy, PartialEq)]
enum Seed {
    /// `dw = s A phi`, at rest — on a degenerate string exactly the rotation generator.
    Disp,
    /// `dw' = s A wu phi` from zero — injects angular momentum, which marginality needs.
    Vel,
}

#[derive(Clone, Debug)]
struct Run {
    growth: f64,
    saturation: f64,
    drift: f64,
    n_not_converged: usize,
    off_mode: f64,
    quarters: [f64; 4],
}

/// `_whirl_run`: drive one polarization in a plane, seed the other, watch it grow (or not).
fn whirl_run(kappa_w: f64, amp: f64, drive: Drive, seed: Seed, t_total: f64, n: i64) -> Run {
    let mut s = Geo {
        kappa: 0.0,
        kappa_w: Some(kappa_w),
        lam_long: Some(LAM_LONG),
        ..Geo::new(n)
    }
    .build();
    let x = s.p.grid();
    let sine = |m: f64| -> Vec<f64> { x.iter().map(|&xi| (m * PI * xi / L).sin()).collect() };
    let phi = sine(1.0);
    let norm = dot(&phi, &phi);
    let scaled = |a: f64| -> Vec<f64> { phi.iter().map(|p| a * p).collect() };
    let z = zeros(&s);

    let driven_ic = scaled(amp);
    let (seeded_disp, seeded_vel) = match seed {
        Seed::Disp => (scaled(SEED_REL * amp), z.clone()),
        Seed::Vel => (z.clone(), scaled(SEED_REL * amp * kc(n).1.sqrt())),
    };
    match drive {
        Drive::U => s.set_state(
            &driven_ic,
            &seeded_disp,
            &z,
            &[z.clone(), seeded_vel, z.clone()],
        ),
        Drive::W => s.set_state(
            &seeded_disp,
            &driven_ic,
            &z,
            &[seeded_vel, z.clone(), z.clone()],
        ),
    }

    let n_steps = (t_total * s.p.fs) as usize;
    let e0 = s.energy();
    let (mut e_lo, mut e_hi) = (e0, e0);
    let mut q_perp = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        step(&mut s);
        let perp = if drive == Drive::U { &s.w } else { &s.u };
        q_perp.push(dot(perp, &phi) / norm);
        let e = s.energy();
        // NaN-propagating, so a run that blew up cannot report a small drift (§29.3).
        e_lo = -nan_max(-e_lo, -e);
        e_hi = nan_max(e_hi, e);
    }

    let driven = if drive == Drive::U { &s.u } else { &s.w };
    let modes: Vec<f64> = [1.0, 3.0, 5.0]
        .iter()
        .map(|&m| (dot(driven, &sine(m)) / norm).abs())
        .collect();
    let seed_amp = SEED_REL * amp;
    let peak = max_abs(&q_perp);
    let len = q_perp.len();
    let quarters: [f64; 4] =
        std::array::from_fn(|i| max_abs(&q_perp[i * len / 4..(i + 1) * len / 4]) / seed_amp);
    Run {
        growth: peak / seed_amp,
        saturation: peak / amp,
        drift: (e_hi - e_lo) / e0.abs(),
        n_not_converged: s.n_not_converged,
        off_mode: nan_max(modes[1], modes[2]) / modes[0],
        quarters,
    }
}

/// The tongue: growth against `delta/(eps A^2)` at fixed amplitude, six runs, built once and
/// shared by the five bars that read it (the Python's module-scoped fixture).
fn tongue() -> &'static Vec<(f64, Run)> {
    static TONGUE: OnceLock<Vec<(f64, Run)>> = OnceLock::new();
    TONGUE.get_or_init(|| {
        let amp = amp_for(TENSION_RISE, N);
        TONGUE_FRACS
            .iter()
            .map(|&f| {
                let run = whirl_run(kappa_w_at(f, amp, N), amp, Drive::U, Seed::Disp, T_TOTAL, N);
                println!("tongue {f}: {run:?}");
                (f, run)
            })
            .collect()
    })
}

fn at(frac: f64) -> &'static Run {
    &tongue().iter().find(|(f, _)| *f == frac).unwrap().1
}

fn growth(frac: f64) -> f64 {
    at(frac).growth
}

#[test]
fn the_whirling_growth_maps_the_mathieu_tongue() {
    // Carried from `test_the_whirling_growth_maps_the_mathieu_tongue`, the headline. Whirling is
    // a PARAMETRIC resonance, not a thing that happens above an amplitude: unimodal, peaked at the
    // predicted 0.25 with no free parameter, dead outside (0, 0.5). The 0.00 end is marginal but
    // earns nothing here -- a displacement seed on a degenerate string is a planar mode in a
    // rotated plane -- `the_degenerate_string_is_marginal_not_exponential` tests marginality.
    let peak = TONGUE_FRACS
        .iter()
        .copied()
        .max_by(|a, b| growth(*a).total_cmp(&growth(*b)))
        .unwrap();
    assert_eq!(peak, 0.25, "the tongue must peak at delta/(eps A^2) = 0.25");
    assert!(
        growth(0.25) > 40.0,
        "the centre must whirl: {}",
        growth(0.25)
    );
    assert!(
        growth(0.8) < 2.0,
        "past the upper edge the seed must not grow: {}",
        growth(0.8)
    );
    assert!(growth(0.25) / growth(0.8) > 20.0);
    // Unimodal, with both flanks resolved: the shape is the claim, not just the peak.
    assert!(
        growth(0.0) < growth(0.07) && growth(0.07) < growth(0.25),
        "the rising flank is not monotone"
    );
    assert!(
        growth(0.25) > growth(0.41) && growth(0.41) > growth(0.5) && growth(0.5) > growth(0.8),
        "the falling flank is not monotone"
    );
    // It genuinely whirls: the seed reaches a macroscopic fraction of the driven amplitude.
    assert!(at(0.25).saturation > 20.0 * SEED_REL);
}

#[test]
fn the_growth_rate_is_the_mathieu_rate_the_plan_discarded_as_a_precession_rate() {
    // Carried from the test of the same name. `(Om/2) sqrt(qM^2 - sigma^2)` predicts how fast the
    // seed grows at every point of the tongue, nothing fitted; measured 5-11% LOW, systematically
    // -- a leading-order oracle (quasi-static eps, first-order Duffing Om) plus the seed's
    // non-growing component, not a bug. The residual's SIGN is asserted, so a change of story is
    // caught.
    let amp = amp_for(TENSION_RISE, N);
    for frac in [0.07, 0.25, 0.41] {
        let pred = mathieu_rate(kappa_w_at(frac, amp, N), amp, N);
        let meas = growth_rate(&at(frac).quarters);
        println!(
            "{frac}: predicted {pred:.3}/s, measured {meas:.3}/s ({:.4})",
            meas / pred
        );
        assert!(
            (meas / pred - 1.0).abs() <= 0.2,
            "at {frac}: {meas:.1}/s against {pred:.1}/s -- the tongue is in place but growing at \
             the wrong speed, so the mechanism is probably not the 2 Om pump"
        );
        assert!(
            meas < pred,
            "at {frac}: the residual changed sign ({meas:.1} vs {pred:.1})"
        );
    }
    // The centre's rate IS the discarded precession rate, at the hardened Om.
    let (_, w0sq, eps) = kc(N);
    let ea2 = eps * amp.powi(2);
    let om = (w0sq + 0.75 * ea2).sqrt();
    let centre = mathieu_rate(kappa_w_at(0.25, amp, N), amp, N);
    let want = ea2 / (8.0 * om);
    assert!(
        (centre - want).abs() <= 1e-9 * want.abs(),
        "{centre} vs {want}"
    );
}

#[test]
fn the_whirl_conserves_energy_and_stays_converged() {
    // Carried from the test of the same name: the gate that makes the tongue mean anything. A
    // parametric instability redistributes energy, it does not create it, so the lossless model
    // conserves straight through a 76x blow-up -- which is how a whirl is told from a diverging
    // solve. Energy is not enough on its own: model #9's IN-PLANE exchange (dT/T0 ~ 3) conserves
    // too, so the driven field must also stay single-mode.
    for (frac, r) in tongue() {
        assert!(
            r.drift < DRIFT_GATE,
            "at {frac} the energy drifted {:.2e}: the {:.1}x 'whirl' is divergence",
            r.drift,
            r.growth
        );
        assert_eq!(r.n_not_converged, 0, "a non-converged solve at {frac}");
        assert!(
            r.off_mode < 0.01,
            "at {frac} the driven field is {:.2}% off-mode: energy may be leaving mode 1 in plane",
            100.0 * r.off_mode
        );
    }
}

#[test]
fn whirling_must_be_seeded_and_never_leaks_from_a_planar_start() {
    // Carried from the test of the same name: the honesty gate at the parameters that would break
    // it (the tongue centre, 35 mm, a 17 Hz detuning). `w -> -w` is a symmetry of the detuned model
    // too -- the knob splits only the linear operator -- so an identically-zero `w` can never
    // acquire a bit, and every growth ratio here is a growth ratio rather than a leak.
    let amp = amp_for(TENSION_RISE, N);
    let mut s = Geo {
        kappa: 0.0,
        kappa_w: Some(kappa_w_at(0.25, amp, N)),
        lam_long: Some(LAM_LONG),
        ..Geo::new(N)
    }
    .build();
    assert!(!s.is_degenerate());
    let shape = mode_ic(&s, 1, amp);
    start_u(&mut s, &shape);
    let (mut w_max, mut u_max) = (0.0f64, 0.0f64);
    for _ in 0..(T_TOTAL * s.p.fs) as usize {
        step(&mut s);
        // Tracked, never sampled at the last step: a mode-1 string passes through u ~ 0 twice a
        // period, so the final state is at an arbitrary phase.
        w_max = nan_max(w_max, max_abs(&s.w));
        u_max = nan_max(u_max, max_abs(&s.u));
    }
    println!("unseeded: max|w| {w_max:e}, max|u| {:.7} A", u_max / amp);
    assert_eq!(
        w_max, 0.0,
        "an unseeded planar run leaked {w_max:.3e} into w"
    );
    assert!(
        u_max > 0.9 * amp,
        "the driven plane must ring at full amplitude"
    );
    // ...and the seeded run at the identical parameters whirls. Same string, one seed apart.
    assert!(growth(0.25) > 40.0);
}

#[test]
fn only_the_plane_of_the_lower_mode_whirls() {
    // Carried from `test_only_the_plane_of_the_lower_mode_whirls`: Gough's asymmetry. The tongue
    // is `0 < delta < eps A^2/2` with `delta` measured FROM THE DRIVEN PLANE, so driving the stiff
    // polarization flips its sign and puts the string outside the tongue at every amplitude. Same
    // string, same energy, distinguished only by which polarization carries the motion -- and a
    // free test of the u/w symmetry of the implementation.
    let amp = amp_for(TENSION_RISE, N);
    let stiff = whirl_run(
        kappa_w_at(0.25, amp, N),
        amp,
        Drive::W,
        Seed::Disp,
        T_TOTAL,
        N,
    );
    println!("stiff: {stiff:?}");
    let soft = at(0.25);
    assert!(soft.growth > 40.0);
    assert!(
        stiff.growth < 1.5,
        "driving the stiff plane must not whirl: {:.2}x",
        stiff.growth
    );
    assert!(
        soft.growth / stiff.growth > 30.0,
        "Gough's asymmetry is not holding"
    );
    assert!(stiff.drift < DRIFT_GATE && stiff.n_not_converged == 0);
}

#[test]
fn the_threshold_moves_as_the_square_root_of_the_detuning() {
    // Carried from `test_the_threshold_moves_as_the_square_root_of_the_detuning`. Three runs: A1
    // above A_c(delta_1) whirls; the SAME A1 at delta_2 = 2 delta_1 is now below the moved
    // threshold and is stable; sqrt(2) A1 at delta_2 whirls again. Run 2 alone fits "more
    // detuning is more stable", which is false; run 3 is what makes it the sqrt(delta) law. The
    // factors 2 and sqrt(2) are exact by construction, so asserting them guards the sweep's own
    // arithmetic; the physics is in the three growths.
    let amp_ref = amp_for(TENSION_RISE, N);
    let k1 = kappa_w_at(0.25, amp_ref, N);
    let k2 = kappa_w_at(0.5, amp_ref, N);
    let (a1, a2) = (1.2 * a_crit(k1, N), 1.2 * a_crit(k2, N));
    let d_ratio = delta(k2, N) / delta(k1, N);
    let a_ratio = a_crit(k2, N) / a_crit(k1, N);
    assert!((d_ratio - 2.0).abs() <= 1e-9 * 2.0, "{d_ratio}");
    assert!(
        (a_ratio - 2f64.sqrt()).abs() <= 1e-9 * 2f64.sqrt(),
        "{a_ratio}"
    );

    let runs = [
        whirl_run(k1, a1, Drive::U, Seed::Disp, T_TOTAL, N),
        whirl_run(k2, a1, Drive::U, Seed::Disp, T_TOTAL, N),
        whirl_run(k2, a2, Drive::U, Seed::Disp, T_TOTAL, N),
    ];
    let [above_k1, below_k2, above_k2] = &runs;
    println!(
        "A1 at d1 {:.3}x, A1 at d2 {:.3}x, A2 at d2 {:.3}x",
        above_k1.growth, below_k2.growth, above_k2.growth
    );
    assert!(
        above_k1.growth > 10.0,
        "A1 sits 1.2x above A_c(delta_1) and must whirl"
    );
    assert!(
        below_k2.growth < 2.5,
        "at delta_2 the same A1 is below the moved threshold and must be stable: {:.2}x",
        below_k2.growth
    );
    assert!(
        above_k2.growth > 10.0,
        "sqrt(2) more amplitude restores the position and must whirl again"
    );
    assert!(above_k1.growth / below_k2.growth > 5.0);
    assert!(above_k2.growth / below_k2.growth > 5.0);
    for r in &runs {
        assert!(r.drift < DRIFT_GATE && r.n_not_converged == 0);
        assert!(
            r.off_mode < 0.01,
            "dT/T0 reaches 2.16 here: {:.4}",
            r.off_mode
        );
    }
}

#[test]
fn the_degenerate_string_is_marginal_not_exponential() {
    // Carried from the test of the same name: the negative control, which needs the VELOCITY seed.
    // A degenerate string's second solution grows SECULARLY, linearly in t; a displacement seed is
    // the rotation generator and pins at 1.00x forever, which is rotational invariance, not
    // marginality. The discriminator is the envelope's shape over four quarters, no fit: secular
    // is 1 : 2 : 3 : 4, exponential a constant neighbour ratio.
    let amp = amp_for(TENSION_RISE, N);
    let deg = whirl_run(0.0, amp, Drive::U, Seed::Vel, T_TOTAL, N);
    let det = whirl_run(
        kappa_w_at(0.25, amp, N),
        amp,
        Drive::U,
        Seed::Vel,
        T_TOTAL,
        N,
    );
    let q_deg: Vec<f64> = deg.quarters.iter().map(|q| q / deg.quarters[0]).collect();
    let q_det: Vec<f64> = det.quarters.iter().map(|q| q / det.quarters[0]).collect();
    println!("degenerate {q_deg:.4?}, detuned {q_det:.4?}");

    // Secular: the envelope is linear in t. `pytest.approx(rel=0.12)`, which also kept
    // `abs = 1e-12` -- immaterial against values of order 1.
    for (got, want) in q_deg.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!(
            (got - want).abs() <= 0.12 * want,
            "the degenerate envelope must be 1:2:3:4, got {q_deg:.3?}"
        );
    }
    // Exponential: far past the secular envelope, with a constant neighbour ratio.
    assert!(q_det[3] > 5.0 * q_deg[3], "{} vs {}", q_det[3], q_deg[3]);
    let ratios: Vec<f64> = q_det.windows(2).map(|w| w[1] / w[0]).collect();
    assert!(
        ratios.iter().all(|r| *r > 2.0),
        "the detuned neighbour ratios {ratios:.3?} are not an exponential's"
    );
    assert!(deg.drift < DRIFT_GATE && det.drift < DRIFT_GATE);
    assert!(deg.n_not_converged == 0 && det.n_not_converged == 0);
}

#[test]
fn the_tongue_does_not_move_with_the_grid() {
    // Carried from the test of the same name: an instability that lives on the mesh is a bug. At
    // N = 32 and the same delta/(eps A^2) the tongue whirls and is dead in the same places (its
    // centre's kappa_w moves 0.13% while the grid halves). The RATE is mildly N-dependent -- eps
    // is a continuum coefficient -- so the position is the claim, not how fast it grows there.
    let n = 2 * N;
    let amp = amp_for(TENSION_RISE, n);
    let inside = whirl_run(kappa_w_at(0.25, amp, n), amp, Drive::U, Seed::Disp, 0.05, n);
    let outside = whirl_run(kappa_w_at(0.8, amp, n), amp, Drive::U, Seed::Disp, 0.05, n);
    println!(
        "N = {n}: inside {:.3}x, outside {:.3}x",
        inside.growth, outside.growth
    );
    assert!(
        inside.growth > 10.0,
        "the centre must still whirl at N = {n}"
    );
    assert!(
        outside.growth < 2.0,
        "outside must still be stable at N = {n}"
    );
    assert!(inside.growth / outside.growth > 5.0);
    for r in [&inside, &outside] {
        assert!(r.drift < DRIFT_GATE && r.n_not_converged == 0);
    }
}
