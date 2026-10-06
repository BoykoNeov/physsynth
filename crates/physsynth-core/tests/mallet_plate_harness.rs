//! The mallet-struck plate's validation harness — model #7p, carried from
//! `tests/test_mallet_plate.py` and `tests/test_mallet_plate_signature.py` (retirement plan §38), at
//! the retired Python helper `make_mallet_plate`'s parameters: a 1 m square Kirchhoff plate at
//! `kappa = 20`, `rho = 0.005 kg/m^2`, `theta = 0.28`, `nu = 0.3`, `N = 24`, plate Courant number
//! `mu = 1` via `fs = kappa / (mu h^2)` (11,520 Hz), a 20 g mallet on a `K = 5e4`, `alpha = 2.3` felt
//! striking `(0.3, 0.4)` at 3 m/s, and the binding's solve defaults (`eta_tol = 1e-12`,
//! `newton_tol = 1e-14`, 60 iterations). Strike and pickup positions are physical coordinates and
//! snap to the nearest live node, as the Python's did.
//!
//! **Not `mallet.rs`'s fixture**, which is a 0.4 m plate at `kappa = 1`, `rho = 2`, `fs = 20 kHz`,
//! `N = 12` or `14`, run for up to 2,000 steps; that file keeps its plate section. The Python ran up
//! to 4,000 steps here, across four felts, three losses, three solve tolerances, a 48-cell rig at 16x
//! the conditioning, both boundaries, two curved outlines and five signatures.
//!
//! **No outside referee is frozen.** The NumPy numbers here were the start bump (`np.exp`, and on the
//! free plate `np.mean`'s pairwise sum — `reduce::sum` here), the analytic mode shapes (`np.sin`),
//! the mode projections and the free plate's weighted mean (`@` and `np.dot`, BLAS — left-to-right
//! sums here, inside bars with measured margins) and the spectral centroid (`np.fft`, a tolerance
//! port in `physsynth_analysis::spectrum`).

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::collision::{contact_force_dg, PowPath};
use physsynth_core::engine::{simulate, Resonator};
use physsynth_core::mallet::{MalletPlate, PlateParams};
use physsynth_core::plate::{self, Boundary, Domain, Plate};
use physsynth_core::reduce;
use std::f64::consts::PI;

/// The acceptance bar, unchanged — see CLAUDE.md.
const CONSERVE_TOL: f64 = 1e-10;

/// The free branch's bar, and a property of the energy READ-OUT rather than of the scheme: a point
/// strike feeds the free plate's `{1, x, y}` rigid nullspace, and the potential form cancels the
/// rigid part only to `eps`, leaving an error quadratic in the rigid displacement
/// (`the_free_readout_error_is_quadratic_in_the_rigid_drift`). A claim about THIS rig: it is paired
/// with a supported control on the same mallet, because the ratio between the branches is what
/// transfers and the constant is only this rig's value of it.
const FREE_READOUT_TOL: f64 = 1e-8;
const FREE_STEPS: usize = 2000;

const KAPPA: f64 = 20.0;
const RHO: f64 = 0.005;
const THETA: f64 = 0.28;
/// `MALLET_MASS_DEFAULT`, `MALLET_K_DEFAULT`, `MALLET_ALPHA_DEFAULT`, `MALLET_VELOCITY_DEFAULT`.
const MASS: f64 = 0.02;
const K_DEF: f64 = 5.0e4;
const ALPHA_DEF: f64 = 2.3;
const V0: f64 = 3.0;

/// The retired `make_plate` / `make_free_plate` / `make_mallet_plate` plate: unit square, Courant
/// number exactly `mu` by solving for `fs`. `nu` is passed only where the helper passed it.
fn plate_at(
    boundary: Boundary,
    domain: Domain,
    n: i64,
    mu: f64,
    sigma: f64,
    nu: Option<f64>,
) -> Plate {
    let h = 1.0 / n as f64;
    let spec = plate::PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho: RHO,
        fs: KAPPA / (mu * h * h),
        n,
        sigma,
        theta: THETA,
        boundary: Some(boundary),
        domain: Some(domain),
        nu,
        ..plate::PlateSpec::default()
    };
    Plate::new(plate::Params::new(&spec).expect("a valid plate"))
}

/// Everything `make_mallet_plate` took, with its defaults.
#[derive(Clone, Copy)]
struct Strike {
    boundary: Boundary,
    domain: Domain,
    n: i64,
    mu: f64,
    k: f64,
    mass: f64,
    alpha: f64,
    hysteresis: f64,
    x: f64,
    y: f64,
    v0: f64,
    gap: f64,
    sigma: f64,
    newton_tol: f64,
}

const STRIKE: Strike = Strike {
    boundary: Boundary::Supported,
    domain: Domain::Rectangle,
    n: 24,
    mu: 1.0,
    k: K_DEF,
    mass: MASS,
    alpha: ALPHA_DEF,
    hysteresis: 0.0,
    x: 0.3,
    y: 0.4,
    v0: V0,
    gap: 0.0,
    sigma: 0.0,
    newton_tol: 1e-14,
};

/// Strike `pl` as `s` describes. The plate's state is read, so a pre-set plate is honoured — set it
/// BEFORE this call, as the Python did, since the mallet's start is placed against the struck node.
fn strike_on(s: Strike, pl: Plate) -> MalletPlate {
    let p = PlateParams::new(
        &pl.p,
        s.mass,
        s.k,
        s.alpha,
        s.hysteresis,
        s.x,
        s.y,
        s.gap,
        1e-12,
        s.newton_tol,
        60,
    )
    .expect("valid mallet");
    MalletPlate::new(p, pl, s.gap, s.v0)
}

/// The retired `make_mallet_plate`.
fn mallet(s: Strike) -> MalletPlate {
    strike_on(
        s,
        plate_at(s.boundary, s.domain, s.n, s.mu, s.sigma, Some(0.3)),
    )
}

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

/// The physical coordinates of the live nodes, in live order — walked here from the mask, not
/// through the model's own `live_coords`, as the Python read them through `X[mask]`.
fn live_xy(p: &plate::Params) -> (Vec<f64>, Vec<f64>) {
    let flags = p.mask.flags();
    let xs = (0..flags.len())
        .filter(|&i| flags[i])
        .map(|i| p.x[i])
        .collect();
    let ys = (0..flags.len())
        .filter(|&i| flags[i])
        .map(|i| p.y[i])
        .collect();
    (xs, ys)
}

/// The retired `plate_bump`: a smooth off-centre bump on the live nodes, mean removed on the free
/// plate (`np.mean` is NumPy's pairwise sum, which `reduce::sum` reproduces).
fn plate_bump(p: &plate::Params, amp: f64) -> Vec<f64> {
    let (xs, ys) = live_xy(p);
    let width = 0.08 * p.lx;
    let u0: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| {
            let (dx, dy) = (x - 0.42 * p.lx, y - 0.38 * p.ly);
            amp * (-((dx * dx + dy * dy) / (width * width))).exp()
        })
        .collect();
    match p.boundary {
        Boundary::Supported => u0,
        Boundary::Free => {
            let mean = reduce::sum(&u0) / u0.len() as f64;
            u0.iter().map(|v| v - mean).collect()
        }
    }
}

/// The Python's `_run`: the per-step total energy `H`, including the start.
fn run(mal: &mut MalletPlate, steps: usize) -> Vec<f64> {
    let mut e = Vec::with_capacity(steps + 1);
    e.push(mal.energy());
    for _ in 0..steps {
        mal.step().expect("contact solve converged");
        e.push(mal.energy());
    }
    e
}

fn drift(e: &[f64]) -> f64 {
    nan_max(e.iter().map(|v| (v - e[0]).abs())) / e[0].abs()
}

// == criterion 0: the influence column, the only new arithmetic in the model =====================

#[test]
fn the_influence_column_reproduces_a_forced_step_at_every_node() {
    // Stepping force-free and then adding `influence * f` must be the same plate as stepping with
    // `f` in the right-hand side: `A (u_free + d) = rhs_0 + A d = rhs_0 + (k^2 f / den) e_node`.
    // Asserted over the WHOLE field, and computed here by hand rather than through the model's own
    // injector: a column right at the strike node and wrong elsewhere passes every other test.
    for boundary in [Boundary::Supported, Boundary::Free] {
        let mal = mallet(Strike {
            boundary,
            n: 16,
            ..STRIKE
        });
        let (node, influence) = (mal.params().node, &mal.params().influence);
        let force = 7.5; // N — an arbitrary nonzero

        // A generic non-rest state: the bump, advanced far enough that `u` and `u_prev` differ.
        let mut forced = plate_at(boundary, Domain::Rectangle, 16, 1.0, 0.0, Some(0.3));
        let bump = plate_bump(&forced.p, 1e-3);
        forced.set_state(&bump, &vec![0.0; forced.p.n_live]);
        for _ in 0..5 {
            forced.step(None);
        }
        let mut split = forced.clone();

        let mut f_ext = vec![0.0; forced.p.n_live];
        f_ext[node] = -force;
        forced.step(Some(&f_ext));
        split.step(None);

        let scale = nan_max(forced.u.iter().map(|v| v.abs()));
        assert!(scale > 0.0, "the reference field is identically zero");
        // Not `==`: `solve(rhs) + c*solve(e)` and `solve(rhs + c*e)` round differently.
        let worst = nan_max(
            forced
                .u
                .iter()
                .zip(&split.u)
                .zip(influence)
                .map(|((f, s), g)| (f - (s - g * force)).abs()),
        ) / scale;
        println!("{boundary:?}: column vs forced step {worst:e}");
        assert!(
            worst < 1e-14,
            "{boundary:?}: the column disagrees with a forced step by {worst:.3e}"
        );
    }
}

#[test]
fn one_newton_moves_the_strike_node_by_the_admittance() {
    // `g_s` is READ from the column, so its identity with the column's entry is structural; the
    // physics is that a unit force from rest moves the struck node by `g_s` metres one step later.
    // The Python wrote `approx(rel=1e-14)`, which keeps `abs=1e-12` — at `g_s ~ 1e-4` that is a
    // 1e-8 relative bar. The relative claim the comment made is the one asserted here.
    for boundary in [Boundary::Supported, Boundary::Free] {
        let mal = mallet(Strike {
            boundary,
            n: 16,
            ..STRIKE
        });
        let p = mal.params();
        assert_eq!(p.g_s, p.influence[p.node]);
        assert!(
            p.g_s > 0.0,
            "an SPD plate cannot have a negative driving-point admittance"
        );
        assert_eq!(p.g, p.g_s + p.g_h);

        let mut pl = plate_at(boundary, Domain::Rectangle, 16, 1.0, 0.0, Some(0.3));
        let mut f_ext = vec![0.0; pl.p.n_live];
        f_ext[p.node] = 1.0;
        pl.step(Some(&f_ext));
        let rel = (pl.u[p.node] - p.g_s).abs() / p.g_s;
        println!("{boundary:?}: g_s {:e}, one-newton mismatch {rel:e}", p.g_s);
        assert!(
            rel <= 1e-14,
            "{boundary:?}: one newton moved the node {:.6e}, admittance {:.6e}",
            pl.u[p.node],
            p.g_s
        );
    }
}

#[test]
fn the_strike_snaps_to_a_node_and_says_where() {
    // `pickup_index_at` counts LIVE nodes; `x`/`y` are full-grid arrays. A confusion between the
    // two reports a plausible point and builds a wrong column, so the reported point is checked
    // against the live coordinates walked independently of the model.
    let mal = mallet(STRIKE);
    let (pl, p) = (&mal.plate.p, mal.params());
    assert_eq!(p.node, plate::pickup_index_at(p.x_strike, p.y_strike, pl));
    let (xs, ys) = live_xy(pl);
    assert_eq!(p.x_strike, xs[p.node]);
    assert_eq!(p.y_strike, ys[p.node]);
    assert!((p.x_strike - 0.3).abs() <= pl.h && (p.y_strike - 0.4).abs() <= pl.h);
}

#[test]
fn the_struck_field_is_oriented_the_way_the_coordinates_are() {
    // `state` is the full field, and a transposed decode renders as a plausible picture (viewer
    // batch 18). The strike must show up at the full-grid index whose `x` IS `x_strike` — on a
    // square plate there is no other way to tell the axes apart.
    let mut mal = mallet(STRIKE);
    for _ in 0..12 {
        mal.step().expect("contact solve converged");
    }
    let field = mal.plate.state();
    assert!(field.iter().all(|v| v.is_finite()), "non-finite field");
    // First maximum, as `np.argmax` takes it.
    let mut idx = 0;
    for (i, v) in field.iter().enumerate() {
        if v.abs() > field[idx].abs() {
            idx = i;
        }
    }
    let p = mal.params();
    assert_eq!(
        mal.plate.p.x[idx], p.x_strike,
        "the field's x axis is not x's"
    );
    assert_eq!(
        mal.plate.p.y[idx], p.y_strike,
        "the field's y axis is not y's"
    );
}

#[test]
fn the_struck_plates_acceleration_is_the_second_difference_of_its_motion() {
    // Added with the batch, the human's call (retirement plan §38.4, breakages A and I). The strike
    // corrects the plate's displacement AND its stored acceleration — the field `pressure()`, the
    // radiated read-out, is built from. Skipping the second correction left every bar in the
    // workspace green: the energy never reads `accel`, and the one test that does calls the
    // injector directly rather than through a strike. So, through a real strike, at every node
    // and every step: `accel` must be `(u^{n+1} - 2 u^n + u^{n-1}) / k^2` of the motion the plate
    // actually made. Compared in displacement units (times `k^2`) against the field's own scale,
    // for the reason `mallet.rs`'s column test gives: the second difference is orders smaller.
    let mut mal = mallet(STRIKE);
    let k2 = mal.plate.p.k * mal.plate.p.k;
    let (mut worst, mut struck) = (0.0f64, 0usize);
    for _ in 0..1200 {
        let (u_n, u_nm1) = (mal.plate.u.clone(), mal.plate.u_prev.clone());
        mal.step().expect("contact solve converged");
        let u = &mal.plate.u;
        let scale = nan_max(u.iter().map(|v| v.abs()));
        if scale == 0.0 {
            continue;
        }
        let err = nan_max(
            (0..u.len())
                .map(|i| (mal.plate.accel[i] * k2 - (u[i] - 2.0 * u_n[i] + u_nm1[i])).abs()),
        );
        worst = nan_max([worst, err / scale].into_iter());
        struck += usize::from(mal.state().contact_force != 0.0);
    }
    println!("accel vs second difference: worst {worst:e} of max|u|, {struck} struck steps");
    assert!(struck > 100, "the felt barely engaged ({struck} steps)");
    assert!(
        worst <= 1e-14,
        "the stored acceleration is not the plate's own second difference: {worst:.3e} of max|u|"
    );
}

#[test]
fn a_mallet_set_against_a_moving_plate_reads_the_strike_node() {
    // Added with the batch, the human's call (retirement plan §38.4, breakage L). The starting
    // penetration is the plate's displacement AT THE STRIKE NODE minus the mallet's height, and
    // every other test starts the plate at rest, where every node reads zero — so reading the
    // wrong node was invisible. Here the plate is already displaced (the bump reaches the strike
    // node at ~6.2e-5 m) and the mallet starts 5e-6 m up: the felt is already compressed, which a
    // reading at any node the bump does not reach would deny.
    let mut pl = plate_at(
        Boundary::Supported,
        Domain::Rectangle,
        24,
        1.0,
        0.0,
        Some(0.3),
    );
    let bump = plate_bump(&pl.p, 1e-3);
    pl.set_state(&bump, &vec![0.0; pl.p.n_live]);
    let gap = 5e-6;
    let mal = strike_on(Strike { gap, ..STRIKE }, pl);
    let node = plate::pickup_index_at(0.3, 0.4, &mal.plate.p);
    let u_node = bump[node];
    println!(
        "pre-set plate: u at the strike node {u_node:e}, corner {:e}",
        bump[0]
    );
    assert!(
        u_node > 10.0 * gap,
        "the bump does not reach the strike node ({u_node:e})"
    );
    assert_eq!(mal.state().penetration, u_node - gap);
    assert!(
        mal.state().in_contact,
        "the felt starts compressed and the model denies it"
    );
}

// == criterion 1 (money test): lossless conservation on the supported branch =====================

#[test]
fn supported_lossless_energy_is_conserved() {
    // E_plate + mallet KE + averaged felt PE, flat to machine precision, over four felts.
    for &(k, mass, v0, alpha) in &[
        (5.0e4, 0.02, 3.0, 2.3),
        (2.0e4, 0.05, 3.0, 1.0),
        (3.0e4, 0.03, 4.0, 3.0),
        (5.0e4, 0.02, 3.0, 1.5),
    ] {
        let mut mal = mallet(Strike {
            k,
            mass,
            v0,
            alpha,
            ..STRIKE
        });
        let d = drift(&run(&mut mal, 4000));
        println!(
            "lossless K={k:e} M={mass} v0={v0} alpha={alpha}: drift {d:e}, fallbacks {}",
            mal.state().fallbacks
        );
        assert!(
            d < CONSERVE_TOL,
            "energy drift {d:e} (K={k}, M={mass}, v0={v0}, a={alpha})"
        );
    }
}

#[test]
fn the_bar_is_not_fitted_to_one_rig() {
    // `mu = 4, N = 48`: the shipped rig's sample rate and felt resolution (11,520 Hz) with sixteen
    // times the conditioning (`cond(A) ~ 1 + 64 theta mu^2`) and four times the live nodes. The
    // Python measured the node count, not the conditioning, as what moves this bar (a longer
    // energy reduction accumulates more rounding).
    let mut mal = mallet(Strike {
        n: 48,
        mu: 4.0,
        ..STRIKE
    });
    assert_eq!(mal.plate.p.n_live, 47 * 47);
    assert!(mal.params().steps_per_contact >= 8.0);
    let d = drift(&run(&mut mal, 2000));
    println!("N=48 mu=4: drift {d:e}");
    assert!(
        d < CONSERVE_TOL,
        "drift {d:.2e} at 16x conditioning, 4x the nodes"
    );
}

#[test]
fn the_strike_actually_couples() {
    // Without this, conservation is satisfied by a mallet that sailed past.
    let mut mal = mallet(STRIKE);
    let mut share = 0.0f64;
    for _ in 0..4000 {
        mal.step().expect("contact solve converged");
        share = nan_max([share, mal.plate.energy() / mal.energy()].into_iter());
    }
    println!("plate share at peak: {share}");
    assert!(
        share > 0.3,
        "the plate took only {share:.2} of the energy — coupling too weak"
    );
}

#[test]
fn the_applied_force_is_the_felt_law_at_this_files_stiffness() {
    // Added with the batch, not carried (retirement plan §38.4, breakage B) — the plate twin of the
    // drumhead harness's bar of the same name (§37.4). A felt stiffness read 1% high in BOTH the
    // force and the potential conserves perfectly, and workspace-wide only the linear-gong twin
    // anchor saw it — a comparison of two copies, which says they differ and not which is wrong.
    // So pin the magnitude: every step's applied force must be the discrete gradient of
    // `K [eta]+^(alpha+1) / (alpha+1)` between `eta^{n-1}` and `eta^{n+1}`, with `K` and `alpha`
    // written HERE rather than read from the model (§29.3). Lossless felt: equality, not a band.
    let mut mal = mallet(STRIKE);
    let node = mal.params().node;
    let mut in_contact = 0;
    for _ in 0..1200 {
        let eta_prev = mal.plate.u_prev[node] - mal.state().z_h_prev;
        mal.step().expect("contact solve converged");
        let s = mal.state();
        let law = contact_force_dg(
            s.penetration,
            eta_prev,
            K_DEF,
            ALPHA_DEF,
            1e-12,
            PowPath::Scalar,
        );
        assert_eq!(
            s.contact_force, law,
            "step {}: the applied force is not the felt law",
            s.n
        );
        in_contact += usize::from(s.contact_force > 0.0);
    }
    println!("felt law: {in_contact} steps with a force");
    assert!(
        in_contact > 100,
        "the felt barely engaged ({in_contact} steps)"
    );
}

#[test]
fn the_felt_is_resolved_on_the_shipped_defaults() {
    // The plate's sample rate comes out of its own Courant number, not out of anything the felt
    // cares about — so "is the contact resolved" is asked of the defaults directly.
    let mal = mallet(STRIKE);
    let spc = mal.params().steps_per_contact;
    println!("steps per contact: {spc}");
    assert!(
        spc >= 8.0,
        "only {spc:.1} steps through the felt half-period; the model warns below 8"
    );
}

// == criterion 2: conservation is discrete-gradient-limited ======================================

#[test]
fn the_drift_scales_with_the_newton_tolerance() {
    // On `make_plate`'s plate (no `nu` passed), as the Python built it.
    let drifts: Vec<f64> = [1e-14, 1e-10, 1e-6]
        .iter()
        .map(|&newton_tol| {
            let pl = plate_at(Boundary::Supported, Domain::Rectangle, 24, 1.0, 0.0, None);
            drift(&run(
                &mut strike_on(
                    Strike {
                        newton_tol,
                        ..STRIKE
                    },
                    pl,
                ),
                2500,
            ))
        })
        .collect();
    println!("drift vs tol 1e-14/1e-10/1e-6: {drifts:?}");
    assert!(
        drifts[0] < CONSERVE_TOL,
        "tight-tol drift {:e} is not machine precision",
        drifts[0]
    );
    assert!(
        drifts[2] > drifts[0] * 100.0,
        "loosening the solve did not increase drift ({drifts:?}); conservation is not solve-limited"
    );
}

// == criterion 3: passivity survives the coupling ================================================

#[test]
fn loss_only_ever_removes_energy() {
    for &(sigma, hysteresis) in &[(2.0, 0.0), (0.0, 5.0e3), (1.0, 2.0e3)] {
        let e = run(
            &mut mallet(Strike {
                sigma,
                hysteresis,
                ..STRIKE
            }),
            3000,
        );
        assert!(e.iter().all(|v| v.is_finite()), "non-finite energy");
        let rise = nan_max(e.windows(2).map(|w| w[1] - w[0]));
        let end = e[e.len() - 1] / e[0];
        println!(
            "loss sigma={sigma} lam_h={hysteresis}: max rise {:e} E0, end {end} E0",
            rise / e[0]
        );
        assert!(
            rise <= 1e-9 * e[0],
            "loss added energy (sigma={sigma}, lam_h={hysteresis})"
        );
        assert!(
            end < 0.99,
            "the run lost nothing measurable — the bar is testing itself"
        );
    }
}

// == criterion 4: a mallet that never touches leaves the plate bit-for-bit unchanged =============

#[test]
fn a_mallet_that_never_touches_is_the_bare_plate_to_the_bit() {
    // The one case where superposition is EXACT: `f == 0.0` makes every increment a signed zero and
    // `x - (+-0.0) == x`. Also the guard that the whole-field injection touches no node it should
    // not — the displacement AND the acceleration, which the injection writes separately.
    let steps = 400;
    let mut bare = plate_at(Boundary::Supported, Domain::Rectangle, 24, 1.0, 0.3, None);
    let seed = plate_bump(&bare.p, 1e-3);
    let zeros = vec![0.0; bare.p.n_live];
    bare.set_state(&seed, &zeros);

    let mut held = plate_at(Boundary::Supported, Domain::Rectangle, 24, 1.0, 0.3, None);
    held.set_state(&seed, &zeros);
    // A huge gap with the mallet moving away: it never reaches the plate inside the window.
    let mut mal = strike_on(
        Strike {
            v0: -1.0,
            gap: 10.0,
            ..STRIKE
        },
        held,
    );
    for _ in 0..steps {
        bare.step(None);
        mal.step().expect("free flight cannot fail");
    }
    assert_eq!(mal.state().contact_force, 0.0);
    assert!(!mal.state().in_contact);
    assert_eq!(mal.plate.u, bare.u);
    assert_eq!(mal.plate.accel, bare.accel);
}

// == the free branch, and whose drift it is ======================================================

#[test]
fn the_free_readout_error_is_quadratic_in_the_rigid_drift() {
    // NO MALLET IN THIS TEST. A free plate handed a uniform velocity translates for ever; the
    // potential form annihilates the rigid part mathematically and cancels it only to `eps`
    // numerically, leaving an error that goes like the SQUARE of the rigid displacement. What the
    // struck free plate does below is what the plate does, here with nothing striking it.
    let (mut errors, mut drifts) = (Vec::new(), Vec::new());
    for v_rigid in [0.0, 0.3, 1.0, 3.0] {
        let mut pl = plate_at(Boundary::Free, Domain::Rectangle, 24, 1.0, 0.0, Some(0.3));
        let bump = plate_bump(&pl.p, 1e-3);
        pl.set_state(&bump, &vec![v_rigid; pl.p.n_live]);
        let e0 = pl.energy();
        let mut worst = 0.0f64;
        for _ in 0..2000 {
            pl.step(None);
            worst = nan_max([worst, (pl.energy() - e0).abs()].into_iter());
        }
        errors.push(worst);
        let w = &pl.p.w;
        let mean = w.iter().zip(&pl.u).map(|(a, b)| a * b).sum::<f64>() / reduce::sum(w);
        drifts.push(mean.abs());
    }
    let ratios: Vec<f64> = errors[1..]
        .iter()
        .zip(&drifts[1..])
        .map(|(e, d)| e / (d * d))
        .collect();
    let spread = nan_max(ratios.iter().copied()) / -nan_max(ratios.iter().map(|r| -r));
    println!("read-out: errors {errors:?}, drifts {drifts:?}, ratios {ratios:?}, spread {spread}");
    // With no net momentum the read-out is exact to machine precision.
    assert!(
        errors[0] < 1e-14,
        "a free plate at rest in the mean drifts by {:.2e} J",
        errors[0]
    );
    assert!(
        spread < 1.3,
        "the read-out error is not quadratic in the rigid drift: error/drift^2 = {ratios:?}"
    );
}

#[test]
fn a_free_strike_conserves_within_the_readouts_reach() {
    // The struck cymbal, with the supported plate on the identical mallet as the control: it has no
    // rigid mode to translate along, so it must be orders better.
    let free = drift(&run(
        &mut mallet(Strike {
            boundary: Boundary::Free,
            ..STRIKE
        }),
        FREE_STEPS,
    ));
    let supported = drift(&run(&mut mallet(STRIKE), FREE_STEPS));
    println!("free {free:e}, supported {supported:e}");
    assert!(
        free < FREE_READOUT_TOL,
        "free-plate strike drifted {free:.2e}"
    );
    assert!(
        supported < CONSERVE_TOL,
        "supported control drifted {supported:.2e}"
    );
    assert!(
        supported < 1e-2 * free,
        "the two branches drift alike ({supported:.2e} vs {free:.2e}) — then the free excess is \
         not the rigid mode"
    );
}

#[test]
fn a_struck_free_plate_recoils() {
    // `K 1 = 0`, so projecting the step onto `1^T W` kills the stiffness term and leaves
    // `m^{n+1} = 2 m^n - m^{n-1}`: after the contact the weighted mean is EXACTLY linear in n.
    let mut mal = mallet(Strike {
        boundary: Boundary::Free,
        ..STRIKE
    });
    let total = reduce::sum(&mal.plate.p.w);
    let weight: Vec<f64> = mal.plate.p.w.iter().map(|w| w / total).collect();
    let mut marks = Vec::new();
    for step in 1..=2000 {
        mal.step().expect("contact solve converged");
        if [1000, 1500, 2000].contains(&step) {
            marks.push(
                weight
                    .iter()
                    .zip(&mal.plate.u)
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
            );
        }
    }
    let (first, second) = (marks[1] - marks[0], marks[2] - marks[1]);
    println!(
        "recoil: {first:e} then {second:e}, relative {:e}",
        (second - first).abs() / first.abs()
    );
    assert!(
        first.abs() > 1e-6,
        "the plate took no net momentum: mean moved {first:.2e} m"
    );
    assert!(
        (second - first).abs() <= 1e-9 * first.abs(),
        "the drift is not a constant velocity: {first:.9e} then {second:.9e}"
    );
}

#[test]
fn a_curved_free_plate_is_struck_too() {
    // The influence column is built from whatever `A` the plate factored; a staircased rim only
    // changes which nodes are live. Curved outlines are free-only, so the read-out bar applies.
    for domain in [Domain::Circle, Domain::Guitar] {
        let mut mal = mallet(Strike {
            boundary: Boundary::Free,
            domain,
            n: 20,
            ..STRIKE
        });
        let d = drift(&run(&mut mal, 1500));
        println!("{domain:?}: drift {d:e}, {} live nodes", mal.plate.p.n_live);
        assert!(
            d < FREE_READOUT_TOL,
            "{domain:?} plate strike drifted {d:.2e}"
        );
        assert!(
            mal.plate.p.n_live < 21 * 21,
            "a curved outline should have pruned some nodes"
        );
    }
}

#[test]
fn the_struck_plate_runs_through_the_resonator_interface() {
    // `mu = 0.5` as the Python had it: a coarse plate at the default would under-resolve the felt.
    let mut mal = mallet(Strike {
        n: 12,
        mu: 0.5,
        ..STRIKE
    });
    let k = mal.plate.p.k;
    let res = simulate(&mut mal, 300, Some(0), 100).expect("a converged strike");
    assert_eq!(Resonator::timestep(&mal), k);
    assert_eq!(res.fs, 1.0 / k);
    let full = Resonator::state(&mal);
    assert_eq!(full.len(), mal.plate.p.mask.flags().len());
    assert_eq!(full, mal.plate.state());
    assert_eq!(Resonator::displacement_at(&mal, 0), mal.plate.u[0]);
    assert_eq!(
        res.output.expect("a pickup was requested")[300],
        mal.plate.u[0]
    );
    assert!(mal.plate.pressure().is_finite());
}

// == physical signatures (the supported branch: a free plate translates, a moving target) ========

/// The Python's `_spectral_centroid`: mean removed, Hann window, guarded denominator.
fn centroid(sig: &[f64], fs: f64) -> f64 {
    let mean = sig.iter().sum::<f64>() / sig.len() as f64;
    let w = hann(sig.len());
    let x: Vec<f64> = sig.iter().zip(&w).map(|(s, w)| (s - mean) * w).collect();
    let mag = rfft_mag(&x);
    let f = rfftfreq(sig.len(), 1.0 / fs);
    let num: f64 = f.iter().zip(&mag).map(|(f, m)| f * m).sum();
    let den: f64 = mag.iter().sum();
    num / (den + 1e-30)
}

/// The retired `plate_mode_shape`, unit-normalised as `_mode_energies` normalised it: the exact
/// discrete mode `sin(m pi x / Lx) sin(n pi y / Ly)` of a supported plate at the live nodes.
fn mode(p: &plate::Params, m: u32, n: u32) -> Vec<f64> {
    let (xs, ys) = live_xy(p);
    let phi: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| (f64::from(m) * PI * x / p.lx).sin() * (f64::from(n) * PI * y / p.ly).sin())
        .collect();
    let norm = phi.iter().map(|v| v * v).sum::<f64>().sqrt();
    phi.iter().map(|v| v / norm).collect()
}

/// The Python's `_mode_energies`: peak |projection| onto each mode over `steps` steps.
fn mode_peaks(mal: &mut MalletPlate, modes: &[(u32, u32)], steps: usize) -> Vec<f64> {
    let basis: Vec<Vec<f64>> = modes
        .iter()
        .map(|&(m, n)| mode(&mal.plate.p, m, n))
        .collect();
    let mut peak = vec![0.0f64; modes.len()];
    for _ in 0..steps {
        mal.step().expect("contact solve converged");
        for (pk, phi) in peak.iter_mut().zip(&basis) {
            let proj: f64 = phi.iter().zip(&mal.plate.u).map(|(a, b)| a * b).sum();
            *pk = nan_max([*pk, proj.abs()].into_iter());
        }
    }
    peak
}

/// The Python's pickup at `(0.62 Lx, 0.71 Ly)`.
fn pickup(mal: &MalletPlate) -> usize {
    plate::pickup_index_at(0.62 * mal.plate.p.lx, 0.71 * mal.plate.p.ly, &mal.plate.p)
}

#[test]
fn a_strike_excites_many_modes() {
    // A plate's frequencies go like `m^2 + n^2`, so an impulse leaves a displacement comb that
    // falls away far faster than a drumhead's — hence 1% of the strongest, not the drum's 5%.
    let modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 3)];
    let peak = mode_peaks(&mut mallet(STRIKE), &modes, 3000);
    let top = nan_max(peak.iter().copied());
    let rel: Vec<f64> = peak.iter().map(|p| p / top).collect();
    let strong = rel.iter().filter(|&&r| r > 0.01).count();
    println!("modes: {rel:?}, {strong} above 1%");
    assert!(
        strong >= 6,
        "only {strong} of {} modes spoke: {rel:?}",
        modes.len()
    );
}

#[test]
fn a_centre_strike_nulls_the_antisymmetric_modes() {
    // `sin(2 pi x / Lx)` vanishes at `Lx/2`, and the sine product is an EXACT discrete eigenvector
    // of this plate — so the null is an identity of the scheme. Assert the snap first: with an even
    // N there is a node at exactly the centre, and without one the null would measure the snap.
    let mut centre = mallet(Strike {
        x: 0.5,
        y: 0.5,
        ..STRIKE
    });
    assert_eq!(
        (centre.params().x_strike, centre.params().y_strike),
        (0.5, 0.5),
        "the centre strike snapped off the centre; the null would be measuring the snap"
    );
    let modes = [(1, 1), (2, 1), (1, 2), (3, 3)];
    let centred = mode_peaks(&mut centre, &modes, 2500);
    let offset = mode_peaks(&mut mallet(STRIKE), &modes, 2500);
    println!(
        "centre/offset (2,1): {:e} / {:e}; (1,2): {:e} / {:e}; (3,3) centre {:e}",
        centred[1] / centred[0],
        offset[1] / offset[0],
        centred[2] / centred[0],
        offset[2] / offset[0],
        centred[3] / centred[0]
    );
    for (i, m) in [(1, (2, 1)), (2, (1, 2))] {
        assert!(
            centred[i] < 1e-10 * centred[0],
            "a centre strike drove {m:?} at {:.2e} of (1,1)",
            centred[i] / centred[0]
        );
        assert!(
            offset[i] > 1e-3 * offset[0],
            "the off-centre control failed to drive {m:?}; the null proves nothing"
        );
    }
    // (3,3) is symmetric about the centre and survives it: the null is about symmetry.
    assert!(
        centred[3] > 1e-3 * centred[0],
        "a symmetric mode should survive a centre strike"
    );
}

#[test]
fn a_harder_felt_makes_a_shorter_brighter_strike() {
    // Duration counts steps with a non-zero contact FORCE, as the Python did — not `in_contact`.
    let (mut durations, mut centroids) = (Vec::new(), Vec::new());
    for k in [2.0e4, 2.0e5] {
        let mut mal = mallet(Strike { k, ..STRIKE });
        let i = pickup(&mal);
        let mut sig = Vec::with_capacity(3000);
        let mut touching = 0usize;
        for _ in 0..3000 {
            mal.step().expect("contact solve converged");
            sig.push(mal.plate.u[i]);
            touching += usize::from(mal.state().contact_force != 0.0);
        }
        durations.push(touching);
        centroids.push(centroid(&sig, mal.plate.p.fs));
        assert!(
            mal.params().steps_per_contact >= 8.0,
            "K={k:e} under-resolves its own contact"
        );
    }
    println!("hardness: durations {durations:?}, centroids {centroids:?}");
    assert!(
        durations[1] < durations[0],
        "the harder felt was in contact longer ({} vs {} steps)",
        durations[1],
        durations[0]
    );
    assert!(
        centroids[1] > centroids[0],
        "the harder felt was not brighter ({:.1} vs {:.1} Hz)",
        centroids[1],
        centroids[0]
    );
}

#[test]
fn the_mallet_bounces_and_flies_clear() {
    // One contact, then separation: the felt is one-sided, so once the mallet has reversed and
    // cleared the surface it never comes back inside the window.
    let mut mal = mallet(STRIKE);
    let mut contact = Vec::with_capacity(4000);
    for _ in 0..4000 {
        mal.step().expect("contact solve converged");
        contact.push(mal.state().in_contact);
    }
    let starts = contact.windows(2).filter(|w| w[1] && !w[0]).count() + usize::from(contact[0]);
    println!(
        "bounce: {starts} contact(s), {} steps, exit {} m/s",
        contact.iter().filter(|&&c| c).count(),
        mal.mallet_velocity()
    );
    assert_eq!(starts, 1, "the mallet made contact {starts} times");
    assert!(
        !contact[contact.len() - 1],
        "the mallet never left the plate"
    );
    assert!(
        mal.mallet_velocity() > 0.0,
        "the mallet did not rebound: velocity {:.3e} m/s",
        mal.mallet_velocity()
    );
}

#[test]
fn the_felt_exponent_is_the_only_source_of_dynamic_timbre() {
    // At `alpha = 1` the whole system is linear (the contact's switching is scale-invariant), so the
    // response to a fourfold softer strike must be an exactly scaled copy; above it the felt stiffens
    // with penetration and the SHAPE of the response changes with how hard it is hit — graded.
    let trace = |alpha: f64, v0: f64| -> Vec<f64> {
        let mut mal = mallet(Strike {
            n: 20,
            v0,
            alpha,
            ..STRIKE
        });
        let i = pickup(&mal);
        (0..2000)
            .map(|_| {
                mal.step().expect("contact solve converged");
                mal.plate.u[i]
            })
            .collect()
    };
    let departures: Vec<f64> = [1.0, 1.5, 2.3, 3.0]
        .iter()
        .map(|&alpha| {
            let (loud, quiet) = (trace(alpha, 3.0), trace(alpha, 0.75));
            let diff = nan_max(loud.iter().zip(&quiet).map(|(l, q)| (l - 4.0 * q).abs()));
            diff / nan_max(loud.iter().map(|l| l.abs()))
        })
        .collect();
    println!("departures from scaling: {departures:?}");
    assert!(
        departures.iter().all(|d| d.is_finite()),
        "non-finite departure: {departures:?}"
    );
    assert!(
        departures[0] < 1e-10,
        "a linear felt did not give an exactly scaled response ({:.2e})",
        departures[0]
    );
    assert!(
        departures[1] > 0.05,
        "alpha = 1.5 should already break the scaling"
    );
    assert!(
        departures.windows(2).all(|w| w[0] <= w[1]),
        "the departure from scaling is not monotone in the felt exponent: {departures:?}"
    );
}
