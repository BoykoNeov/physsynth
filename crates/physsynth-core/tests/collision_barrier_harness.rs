//! The barrier string's validation harness — model #8, carried from `tests/test_collision_energy.py`,
//! `tests/test_collision_modal.py` and `tests/test_collision_signature.py` (retirement plan §37), and
//! run at the retired Python helper `make_barrier_string`'s parameters: `L = 1`, `T = 200`,
//! `rho = 0.005` (so `c = 200 m/s`), a flexible string (`kappa = 0`), `theta = 0.28`,
//! `fs = c N / (L lam)`, `K = 1e6`, `alpha = 1.5`, a flat rail 2 mm below rest, and the binding's
//! solve defaults (`eta_tol = 1e-12`, `newton_tol = 1e-13`, 60 iterations). The pluck is the
//! Python's upward first mode, `amp sin(pi x / L)` on the string's own grid — bit-identical to
//! NumPy's (measured before the deletion), so the intermittent contact here is the Python's
//! contact, step for step.
//!
//! **A separate file from `collision_barrier.rs`**, which pins the shell (support, admittance block,
//! the `k ** 2` spelling, refusals) on a different fixture; this one carries the acceptance claims
//! at the Python's own runs, which are longer.
//!
//! **No outside referee is frozen.** The NumPy numbers these files compared against were the pluck
//! (`np.sin`, reproduced to the bit), the static equilibrium (`np.linalg.solve` — the native dense
//! LU below agrees with LAPACK's to 9e-16 of the deflection), and the spectral centroid (`np.fft`,
//! for which `physsynth_analysis::spectrum` is a tolerance port, and every centroid bar here is a
//! ratio with a measured margin). Everything else was already this crate, through the binding.
//!
//! Two things the Python file had no native line for and that are the point of carrying it:
//!
//! * **The static-equilibrium oracle.** Energy conservation proves only that the force and the
//!   potential telescope together; both could carry the same wrong scale and still conserve. Seated
//!   at the closed-form equilibrium `S u* = (K/rho) b`, `S = -L + (K/rho) I`, with `alpha = 1` (where
//!   the discrete gradient is exactly linear), the scheme must HOLD it — which pins the coupling's
//!   absolute magnitude. `S` is built from this file's own `K` and `rho`, never from the model's
//!   `g_mat` or `force_pref` (§29.3: a bar about a constant must not import it).
//! * **The single-node collapse.** With one contact node the vector solve (damped Newton + Armijo)
//!   and the scalar one (Newton + bracket + `brentq`) are different algorithms for the same unique
//!   root, so their agreement checks both.

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::collision::{penetration_of, solve_contact, BarrierString, ContactParams};
use physsynth_core::dense;
use physsynth_core::engine::{simulate, Resonator};
use physsynth_core::string_damped::{DampedStiffString, Params};
use std::f64::consts::PI;

/// The acceptance bar, unchanged — see CLAUDE.md.
const CONSERVE_TOL: f64 = 1e-10;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, f1 = 100 Hz
/// `THETA_DEFAULT`, written here rather than imported (§29.3).
const THETA: f64 = 0.28;
/// `BARRIER_K_DEFAULT`, `BARRIER_ALPHA_DEFAULT`, `BARRIER_HEIGHT_DEFAULT`.
const K_DEF: f64 = 1.0e6;
const ALPHA_DEF: f64 = 1.5;
const RAIL: f64 = -2.0e-3;
/// The binding's solve defaults.
const ETA_TOL: f64 = 1e-12;
const NEWTON_TOL: f64 = 1e-13;
const MAXITER: i64 = 60;

/// `wave_speed()` — 200 m/s.
fn c() -> f64 {
    (T / RHO).sqrt()
}

/// The retired `make_damped_string`.
fn string(n: i64, lam: f64, kappa: f64, sigma0: f64, sigma1: f64) -> DampedStiffString {
    let fs = c() * n as f64 / (L * lam);
    DampedStiffString::new(
        Params::new(L, T, RHO, fs, n, kappa, sigma0, sigma1, THETA, true).expect("valid string"),
    )
}

/// Everything `make_barrier_string` took, with its defaults.
#[derive(Clone, Copy)]
struct Rig {
    n: i64,
    lam: f64,
    k: f64,
    alpha: f64,
    hysteresis: f64,
    kappa: f64,
    sigma0: f64,
    sigma1: f64,
    newton_tol: f64,
}

const RIG: Rig = Rig {
    n: 80,
    lam: 0.9,
    k: K_DEF,
    alpha: ALPHA_DEF,
    hysteresis: 0.0,
    kappa: 0.0,
    sigma0: 0.0,
    sigma1: 0.0,
    newton_tol: NEWTON_TOL,
};

/// The retired `make_barrier_string`, against an arbitrary `(N+1)` profile (`-inf` = no barrier).
fn build(r: Rig, profile: &[f64]) -> BarrierString {
    let s = string(r.n, r.lam, r.kappa, r.sigma0, r.sigma1);
    BarrierString::new(
        s,
        profile,
        r.k,
        r.alpha,
        r.hysteresis,
        ETA_TOL,
        r.newton_tol,
        MAXITER,
    )
    .expect("valid barrier")
}

/// A flat rail `height` below rest — the Python's scalar `barrier`, broadcast.
fn rail(r: Rig, height: f64) -> BarrierString {
    build(r, &vec![height; r.n as usize + 1])
}

/// The Python's `_pluck`: `amp sin(pi x / L)` on the string's grid, released from rest.
fn pluck(bar: &mut BarrierString, amp: f64) {
    let x = bar.string.p.grid();
    let u0: Vec<f64> = x.iter().map(|&xi| amp * (PI * xi / L).sin()).collect();
    bar.set_state(&u0, &vec![0.0; x.len()]);
}

/// `np.max` — NaN-propagating, unlike `f64::max`.
fn nan_max(it: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for v in it {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// `max|E - E0| / |E0|`, NaN-propagating.
fn drift(e: &[f64]) -> f64 {
    nan_max(e.iter().map(|v| (v - e[0]).abs())) / e[0].abs()
}

/// The Python's `_run`: step `steps` times, returning the energy trace, and assert at EVERY step
/// that the vector solve stayed under its cap — energy exactness holds only at the converged root,
/// so a stall would corrupt the balance silently.
fn run(bar: &mut BarrierString, steps: usize) -> Vec<f64> {
    let mut e = Vec::with_capacity(steps + 1);
    e.push(bar.energy());
    for i in 1..=steps {
        bar.step();
        assert!(
            bar.s.newton_iters < bar.p.newton_maxiter,
            "contact solve stalled at step {i}"
        );
        e.push(bar.energy());
    }
    e
}

// == criterion 1 (money test): lossless coupled energy conservation =============================

#[test]
fn lossless_energy_is_conserved_through_genuine_contact() {
    // No string loss, elastic barrier: E_string + h sum phi is conserved exactly through contact.
    for &(k, alpha) in &[(1.0e6, 1.5), (5.0e5, 1.0), (2.0e6, 2.0), (1.0e6, 3.0)] {
        let mut bar = rail(
            Rig {
                k,
                alpha,
                lam: 0.4,
                ..RIG
            },
            RAIL,
        );
        pluck(&mut bar, 5.0e-3);
        let d = drift(&run(&mut bar, 6000));
        println!("lossless K={k:e} alpha={alpha}: drift {d:e}");
        assert!(
            d < CONSERVE_TOL,
            "energy drift {d:e} (K = {k}, alpha = {alpha})"
        );
    }
}

#[test]
fn the_string_actually_contacts_the_rail() {
    // Else conservation is a trivial free-string re-test and a vector-solve bug would hide.
    let mut bar = rail(Rig { lam: 0.4, ..RIG }, RAIL);
    pluck(&mut bar, 5.0e-3);
    let (mut contact_steps, mut max_force) = (0usize, 0.0f64);
    for _ in 0..6000 {
        bar.step();
        if bar.s.penetration.iter().any(|&p| p > 0.0) {
            contact_steps += 1;
        }
        max_force = nan_max(
            bar.s
                .contact_force
                .iter()
                .map(|f| f.abs())
                .chain(std::iter::once(max_force)),
        );
    }
    println!("contacts: {contact_steps} steps, peak force {max_force:e} N");
    assert!(
        contact_steps > 300,
        "only {contact_steps} contact steps — barrier barely touched"
    );
    assert!(
        max_force > 1.0,
        "peak contact force {max_force:e} N too small to test the coupling"
    );
}

// == criterion 2: conservation is discrete-gradient-limited =====================================

#[test]
fn the_drift_scales_with_the_newton_tolerance() {
    // Loosening the vector solve loosens the conservation: the applied force is the exact discrete
    // gradient only at the root, so the telescoping is what conserves.
    let drifts: Vec<f64> = [1e-13, 1e-9, 1e-5]
        .iter()
        .map(|&newton_tol| {
            let mut bar = rail(
                Rig {
                    lam: 0.4,
                    newton_tol,
                    ..RIG
                },
                RAIL,
            );
            pluck(&mut bar, 5.0e-3);
            drift(&run(&mut bar, 2500))
        })
        .collect();
    println!("drift vs tol 1e-13/1e-9/1e-5: {drifts:?}");
    assert!(
        drifts[0] < CONSERVE_TOL,
        "tight-tol drift {:e} not machine precision",
        drifts[0]
    );
    assert!(
        drifts[2] > drifts[0] * 100.0,
        "loosening the solve did not increase drift ({drifts:?}); conservation is not solve-limited"
    );
}

// == criterion 3: passivity survives the coupling ===============================================

#[test]
fn loss_only_ever_removes_energy() {
    for &(sigma0, sigma1, hysteresis) in &[(2.0, 0.0, 0.0), (0.0, 0.0, 5.0e4), (1.0, 0.05, 2.0e4)] {
        let mut bar = rail(
            Rig {
                lam: 0.4,
                sigma0,
                sigma1,
                hysteresis,
                ..RIG
            },
            RAIL,
        );
        pluck(&mut bar, 5.0e-3);
        let e = run(&mut bar, 5000);
        assert!(e.iter().all(|v| v.is_finite()), "non-finite energy");
        let rise = nan_max(e.windows(2).map(|w| w[1] - w[0]));
        println!(
            "loss ({sigma0}, {sigma1}, {hysteresis}): max rise {:e} E0",
            rise / e[0]
        );
        assert!(
            rise <= 1e-9 * e[0],
            "loss added energy (sigma0 = {sigma0}, sigma1 = {sigma1}, lam_h = {hysteresis}): \
             max rise {rise:e}"
        );
    }
}

// == criterion 4: a barrier out of reach is the bare string to the bit ===========================

#[test]
fn an_out_of_reach_barrier_is_the_bare_string_at_the_helpers_courant_number() {
    // At `lam = 0.9`, the helper's default — `collision_barrier.rs` runs the same anchor at 0.4.
    let n = 80;
    let mut bare = string(n, 0.9, 0.0, 0.3, 0.0);
    let mut bar = rail(
        Rig {
            n,
            lam: 0.9,
            sigma0: 0.3,
            ..RIG
        },
        -100.0,
    );
    let x = bare.p.grid();
    let phi: Vec<f64> = x.iter().map(|&xi| 5.0e-3 * (PI * xi).sin()).collect();
    let zeros = vec![0.0; x.len()];
    bare.set_state(&phi, &zeros);
    bar.set_state(&phi, &zeros);
    for _ in 0..500 {
        bare.step();
        bar.step();
    }
    assert!(bar.s.contact_force.iter().all(|&f| f == 0.0));
    assert!(!bar.s.penetration.iter().any(|&p| p > 0.0));
    assert_eq!(bar.string.u, bare.u);
}

#[test]
fn the_barrier_string_runs_through_the_resonator_interface() {
    // The Python duck-typed the engine's protocol; natively that is the trait, so it is driven
    // through `simulate` and each method is checked against the field it reports.
    let mut bar = rail(RIG, RAIL);
    pluck(&mut bar, 5.0e-3);
    let k = bar.string.p.k;
    let res = simulate(&mut bar, 200, Some(1), 50).expect("a barrier step cannot fail");
    assert_eq!(res.fs, 1.0 / k);
    assert_eq!(Resonator::state(&bar), bar.string.u);
    assert_eq!(Resonator::state(&bar).len(), bar.string.p.nodes());
    assert_eq!(Resonator::displacement_at(&bar, 1), bar.string.u[1]);
    assert_eq!(
        res.output.expect("a pickup was requested")[200],
        bar.string.u[1]
    );
    assert_eq!(res.energy[200], bar.energy());
}

#[test]
fn a_stiff_string_conserves_against_the_barrier_too() {
    // The biharmonic term is inside the string's energy; the contact scheme is agnostic to it.
    let mut bar = rail(
        Rig {
            lam: 0.4,
            kappa: 2.0,
            ..RIG
        },
        RAIL,
    );
    pluck(&mut bar, 5.0e-3);
    let d = drift(&run(&mut bar, 5000));
    println!("stiff string drift {d:e}");
    assert!(d < CONSERVE_TOL, "stiff-string-barrier energy drift {d:e}");
}

// == the static-equilibrium magnitude oracle =====================================================

const EPS_RAIL: f64 = 2.0e-3; // a hair ABOVE rest, so the whole interior is in contact
const K_BED: f64 = 3000.0;

/// `S u* = (K/rho) eps 1`, `S = -L + (K/rho) I` — the continuous augmented equilibrium, solved
/// densely from this file's own `K_BED` and `RHO`.
fn flat_equilibrium(bar: &BarrierString) -> Vec<f64> {
    let m = bar.string.p.interior();
    let kr = K_BED / RHO;
    let mut s = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            s[i * m + j] = -bar.string.p.op_l.get(i, j) + if i == j { kr } else { 0.0 };
        }
    }
    let rhs = vec![kr * EPS_RAIL * 1.0; m];
    let f = dense::lu_factor(s, m).expect("S is SPD");
    dense::lu_solve(&f, &rhs).expect("square")
}

/// The Python's `_seat_at_equilibrium`: both history levels at `u*`, zero velocity (`set_state`'s
/// consistent `u^{-1}` is the FREE string's, which is off-equilibrium).
fn seat(bar: &mut BarrierString, ustar: &[f64]) {
    let nodes = bar.string.p.nodes();
    let mut uf = vec![0.0; nodes];
    uf[1..nodes - 1].copy_from_slice(ustar);
    bar.set_state(&uf, &vec![0.0; nodes]);
    bar.string.u = uf.clone();
    bar.string.u_prev = uf;
    let mut pen = vec![0.0; bar.p.support_len()];
    penetration_of(&bar.p, &bar.string.u, &mut pen);
    bar.s.penetration = pen;
}

/// Worst `max|u - u*|` over `steps` steps.
fn hold(bar: &mut BarrierString, ustar: &[f64], steps: usize) -> f64 {
    let mut worst = 0.0f64;
    for _ in 0..steps {
        bar.step();
        let nodes = bar.string.p.nodes();
        let d = nan_max(
            bar.string.u[1..nodes - 1]
                .iter()
                .zip(ustar)
                .map(|(a, b)| (a - b).abs()),
        );
        worst = nan_max([worst, d].into_iter());
    }
    worst
}

fn bed() -> BarrierString {
    rail(
        Rig {
            n: 60,
            lam: 0.5,
            k: K_BED,
            alpha: 1.0,
            ..RIG
        },
        EPS_RAIL,
    )
}

#[test]
fn the_static_equilibrium_matches_the_closed_form() {
    // alpha = 1, full-interior spring bed: at rest eta+ = eta-, the discrete gradient takes its
    // Taylor branch, and the scheme's fixed point IS the continuous augmented equilibrium.
    let mut bar = bed();
    let ustar = flat_equilibrium(&bar);
    seat(&mut bar, &ustar);
    assert!(
        bar.s.penetration.iter().all(|&p| p > 0.0),
        "not in permanent contact — pick a larger eps"
    );
    let d = hold(&mut bar, &ustar, 2000);
    let scale = nan_max(ustar.iter().map(|v| v.abs()));
    println!("flat bed: held to {d:e} (deflection {scale:e})");
    assert!(
        d < 1e-13 * scale.max(1.0),
        "held equilibrium to only {d:e} (deflection {scale:e}) — coupling magnitude off"
    );
}

#[test]
fn the_static_equilibrium_negative_control_has_teeth() {
    // Doubling the whole coupling (the admittance block AND the injected force) moves the true
    // fixed point, so seating at the single-K equilibrium drifts by orders of magnitude.
    let mut good = bed();
    let ustar = flat_equilibrium(&good);
    seat(&mut good, &ustar);
    let good_drift = hold(&mut good, &ustar, 400);
    let mut bad = bed();
    seat(&mut bad, &ustar);
    bad.p.g_mat.iter_mut().for_each(|g| *g *= 2.0);
    bad.p.force_pref *= 2.0;
    let bad_drift = hold(&mut bad, &ustar, 400);
    println!("flat control: good {good_drift:e}, doubled {bad_drift:e}");
    assert!(
        bad_drift > 1e4 * good_drift,
        "negative control too weak: good drift {good_drift:e}, doubled-coupling {bad_drift:e}"
    );
}

// == the single-active-node collapse to the scalar solve =========================================

/// One finite node at `node`, the rest out of support; plucked at `amp`; each step, re-solve the
/// node's scalar contact equation from the free penetration the vector solve saw and compare.
/// Returns `(contact steps, peak force, worst mismatch)`.
fn collapse(r: Rig, node: usize, height: f64, amp: f64, steps: usize) -> (usize, f64, f64) {
    let mut profile = vec![f64::NEG_INFINITY; r.n as usize + 1];
    profile[node] = height;
    let mut bar = build(r, &profile);
    pluck(&mut bar, amp);
    // `(k^2/rho) (A^-1)_jj` — the one contact node's scalar admittance.
    let g = bar.p.g_mat[0];
    let cp = ContactParams {
        lam_h: 0.0,
        ..bar.p.contact
    };
    let (mut contact_steps, mut max_force, mut mismatch) = (0usize, 0.0f64, 0.0f64);
    for _ in 0..steps {
        let eta_prev = bar.p.b[0] - bar.string.u_prev[bar.p.support[0]];
        let seed = bar.s.penetration[0];
        bar.step();
        let (eta_v, f_v) = (bar.s.penetration[0], bar.s.contact_force[0]);
        // The free penetration the vector solve saw: eta_free = eta + g f (applied).
        let sol = solve_contact(eta_v + g * f_v, eta_prev, g, cp, seed, 1e-14, 60)
            .expect("the scalar residual is monotone");
        mismatch = nan_max([mismatch, (eta_v - sol.eta).abs()].into_iter());
        if f_v > 0.0 {
            contact_steps += 1;
        }
        max_force = nan_max([max_force, f_v].into_iter());
    }
    (contact_steps, max_force, mismatch)
}

#[test]
fn a_single_contact_node_collapses_to_the_scalar_solve() {
    let r = Rig {
        lam: 0.4,
        k: 8.0e5,
        ..RIG
    };
    let (steps, force, mismatch) = collapse(r, 40, -1.0e-3, 5.0e-3, 800);
    println!("point fret: {steps} contact steps, peak {force:e} N, mismatch {mismatch:e}");
    assert!(steps > 100, "point barrier barely touched ({steps} steps)");
    assert!(
        force > 1.0,
        "contact force {force:e} N too small to be a real test"
    );
    assert!(
        mismatch < 1e-13,
        "vector (m = 1) and scalar solvers disagree by {mismatch:e} — not the same root"
    );
}

#[test]
fn the_juari_thread_collapses_to_the_scalar_solve_at_its_shipped_numbers() {
    // The tanpura's cotton thread as the viewer renders it: one grazing node at K = 2e6, loss on,
    // an 8 mm first mode — the config-specific claim, not the generic one above.
    let r = Rig {
        n: 100,
        lam: 0.4,
        k: 2.0e6,
        sigma0: 0.5,
        ..RIG
    };
    let (steps, force, mismatch) = collapse(r, 9, 0.0, 8.0e-3, 1200);
    println!("juari: {steps} contact steps, peak {force:e} N, mismatch {mismatch:e}");
    assert!(steps > 100, "thread barely touched ({steps} steps)");
    assert!(
        force > 1.0,
        "contact force {force:e} N too small to be a real test"
    );
    assert!(
        mismatch < 1e-13,
        "juari config: vector (m = 1) and scalar solvers disagree by {mismatch:e}"
    );
}

// == physical signatures (the diagnostic tier) ===================================================

/// The Python's `_spectral_centroid`: amplitude-weighted mean frequency of the Hann-windowed
/// record — no mean removal (the mallet's differs).
fn centroid(sig: &[f64], fs: f64) -> f64 {
    let w = hann(sig.len());
    let x: Vec<f64> = sig.iter().zip(&w).map(|(s, w)| s * w).collect();
    let mag = rfft_mag(&x);
    let f = rfftfreq(sig.len(), 1.0 / fs);
    let num: f64 = f.iter().zip(&mag).map(|(f, m)| f * m).sum();
    let den: f64 = mag.iter().sum();
    num / den
}

/// The Python's `_pluck_and_record`: pluck, run, return the pickup at `max(1, int(0.1 N))` and the
/// per-step "any node in contact" flag.
fn record(bar: &mut BarrierString, amp: f64, steps: usize) -> (Vec<f64>, Vec<bool>) {
    pluck(bar, amp);
    let node = ((0.1 * bar.string.p.n as f64) as usize).max(1);
    let mut pickup = Vec::with_capacity(steps);
    let mut contact = Vec::with_capacity(steps);
    for _ in 0..steps {
        bar.step();
        pickup.push(bar.string.u[node]);
        contact.push(bar.s.penetration.iter().any(|&p| p > 0.0));
    }
    (pickup, contact)
}

fn fs_of(bar: &BarrierString) -> f64 {
    1.0 / bar.string.p.k
}

#[test]
fn the_barrier_brightens_the_tone() {
    // Same string, same pluck: a barrier in reach must raise the spectral centroid.
    let mut free = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.5,
            ..RIG
        },
        -100.0,
    );
    let mut buzz = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.5,
            ..RIG
        },
        -2.0e-3,
    );
    let (pf, _) = record(&mut free, 5.0e-3, 8000);
    let (pb, cb) = record(&mut buzz, 5.0e-3, 8000);
    assert!(
        cb.iter().any(|&c| c),
        "buzz case never contacted the barrier"
    );
    let (cf, cz) = (centroid(&pf, fs_of(&free)), centroid(&pb, fs_of(&buzz)));
    println!("brightness: free {cf}, buzz {cz}, ratio {}", cz / cf);
    assert!(
        cz > 1.3 * cf,
        "barrier did not brighten the tone: centroid free={cf:.0} buzz={cz:.0}"
    );
}

#[test]
fn a_closer_barrier_is_brighter() {
    let mut far = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.5,
            ..RIG
        },
        -4.0e-3,
    );
    let mut near = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.5,
            ..RIG
        },
        -1.0e-3,
    );
    let (pa, _) = record(&mut far, 5.0e-3, 8000);
    let (pn, _) = record(&mut near, 5.0e-3, 8000);
    let (ca, cn) = (centroid(&pa, fs_of(&far)), centroid(&pn, fs_of(&near)));
    println!("closer: far {ca}, near {cn}, ratio {}", cn / ca);
    assert!(
        cn > ca,
        "closer barrier not brighter: far={ca:.0} near={cn:.0}"
    );
}

#[test]
fn the_contact_is_intermittent() {
    // Repeated slaps, and never pinned: some steps must be contact-free.
    let mut bar = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.2,
            ..RIG
        },
        -2.0e-3,
    );
    let (_, contact) = record(&mut bar, 5.0e-3, 8000);
    let onsets = contact.windows(2).filter(|w| w[1] && !w[0]).count();
    let frac = contact.iter().filter(|&&c| c).count() as f64 / contact.len() as f64;
    println!("intermittency: {onsets} onsets, in contact {frac}");
    assert!(
        onsets >= 3,
        "expected repeated slaps, saw {onsets} contact onsets"
    );
    assert!(
        frac < 0.9,
        "string is essentially pinned (in contact {frac})"
    );
}

#[test]
fn a_harder_barrier_shortens_the_contact() {
    let mut soft = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.2,
            k: 1.0e5,
            ..RIG
        },
        -2.0e-3,
    );
    let mut hard = rail(
        Rig {
            lam: 0.4,
            sigma0: 0.2,
            k: 5.0e6,
            ..RIG
        },
        -2.0e-3,
    );
    let (_, cs) = record(&mut soft, 5.0e-3, 8000);
    let (_, ch) = record(&mut hard, 5.0e-3, 8000);
    let (ns, nh) = (
        cs.iter().filter(|&&c| c).count(),
        ch.iter().filter(|&&c| c).count(),
    );
    println!("hardness: soft {ns} steps in contact, hard {nh}");
    assert!(
        nh < ns,
        "harder barrier not shorter contact: soft={ns} hard={nh} (of 8000)"
    );
}
