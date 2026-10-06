//! The mallet-struck drumhead's validation harness — model #7, carried from
//! `tests/test_mallet_energy.py` and `tests/test_mallet_signature.py` (retirement plan §37), at the
//! retired Python helper `make_mallet`'s parameters: a 1 m square head at `T = 200 N/m`,
//! `rho = 0.005 kg/m^2` (so `c = 200 m/s`), `N = 40`, Courant number `lam = 0.5` via
//! `fs = c / (lam h)` (oversampling the stiff felt), a 20 g mallet on a `K = 5e4`, `alpha = 2.3` felt
//! striking the centre at 3 m/s, and the binding's solve defaults (`eta_tol = 1e-12`,
//! `newton_tol = 1e-14`, 60 iterations). Strike and pickup positions are physical coordinates and
//! snap to the nearest live node, as the Python's did.
//!
//! **Not `mallet.rs`'s fixture**, which is a `T = 100`, `rho = 0.26`, `N = 24` drumhead run for 1,200
//! steps; that file keeps the wall rig, the `** 2` pin and the plate model. The Python's own runs are
//! up to 6,000 steps on this head, across four felts, three losses, a missing mallet and a circular
//! head, and none of those had a native line.
//!
//! **No outside referee is frozen.** The NumPy numbers here were the plucked start
//! (`sin(pi X) sin(pi Y)`, reproduced to the bit before the deletion), the mode projections
//! (`basis @ u`, a BLAS product — left-to-right sums here, inside bars with measured margins) and the
//! spectral centroid (`np.fft`, a tolerance port in `physsynth_analysis::spectrum`).

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::collision::{contact_force_dg, PowPath};
use physsynth_core::engine::{simulate, Resonator};
use physsynth_core::mallet::{MalletMembrane, Params};
use physsynth_core::membrane::{self, Domain, Membrane};
use std::f64::consts::PI;

/// The acceptance bar, unchanged — see CLAUDE.md.
const CONSERVE_TOL: f64 = 1e-10;

const T: f64 = 200.0;
const RHO: f64 = 0.005; // areal -> c = 200 m/s
const RADIUS: f64 = 0.5;
/// `MALLET_MASS_DEFAULT`, `MALLET_K_DEFAULT`, `MALLET_ALPHA_DEFAULT`, `MALLET_VELOCITY_DEFAULT`.
const MASS: f64 = 0.02;
const K_DEF: f64 = 5.0e4;
const ALPHA_DEF: f64 = 2.3;
const V0: f64 = 3.0;

/// The retired `make_membrane`: Courant number exactly `lam` by solving for `fs`.
fn head(domain: Domain, n: i64, lam: f64, sigma: f64) -> Membrane {
    let c = (T / RHO).sqrt();
    let p = match domain {
        Domain::Rectangle => {
            let h = 1.0 / n as f64;
            membrane::Params::new(
                Some(domain),
                T,
                RHO,
                c / (lam * h),
                n,
                Some(1.0),
                Some(1.0),
                None,
                sigma,
            )
        }
        Domain::Circle => {
            let h = 2.0 * RADIUS / n as f64;
            membrane::Params::new(
                Some(domain),
                T,
                RHO,
                c / (lam * h),
                n,
                None,
                None,
                Some(RADIUS),
                sigma,
            )
        }
    };
    Membrane::new(p.expect("valid drumhead"))
}

/// Everything `make_mallet` took, with its defaults.
#[derive(Clone, Copy)]
struct Strike {
    domain: Domain,
    n: i64,
    lam: f64,
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
    domain: Domain::Rectangle,
    n: 40,
    lam: 0.5,
    k: K_DEF,
    mass: MASS,
    alpha: ALPHA_DEF,
    hysteresis: 0.0,
    x: 0.5,
    y: 0.5,
    v0: V0,
    gap: 0.0,
    sigma: 0.0,
    newton_tol: 1e-14,
};

/// Strike `mem` as `s` describes (its state is read, so a pre-set head is honoured).
fn strike_on(s: Strike, mem: Membrane) -> MalletMembrane {
    let p = Params::new(
        mem.params(),
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
    MalletMembrane::new(p, mem, s.gap, s.v0)
}

/// The retired `make_mallet`.
fn mallet(s: Strike) -> MalletMembrane {
    strike_on(s, head(s.domain, s.n, s.lam, s.sigma))
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

/// The Python's `_run`: the per-step total energy `H`, including the start.
fn run(mal: &mut MalletMembrane, steps: usize) -> Vec<f64> {
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

// == criterion 1 (money test): lossless coupled conservation =====================================

#[test]
fn lossless_energy_is_conserved_through_the_strike() {
    // The four felts vary alpha too: the wall rig covers alpha in {1, 2, 3} with no head, so this
    // closes the "alpha != 2.3 AND coupled" case — the force law times the head's admittance.
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
        let d = drift(&run(&mut mal, 6000));
        println!(
            "lossless K={k:e} M={mass} v0={v0} alpha={alpha}: drift {d:e}, fallbacks {}",
            mal.state().fallbacks
        );
        assert!(
            d < CONSERVE_TOL,
            "energy drift {d:e} (K = {k}, M = {mass}, v0 = {v0})"
        );
    }
}

#[test]
fn the_strike_actually_couples() {
    // The head must take a real share of the strike, else conservation is a linear re-test.
    let mut mal = mallet(STRIKE);
    let mut frac = 0.0f64;
    for _ in 0..4000 {
        mal.step().expect("contact solve converged");
        frac = nan_max([frac, mal.membrane.energy() / mal.energy()].into_iter());
    }
    println!("head share at peak: {frac}");
    assert!(
        frac > 0.3,
        "membrane took only {frac:.2} of the energy — coupling too weak to test"
    );
}

#[test]
fn the_applied_force_is_the_felt_law_at_this_files_stiffness() {
    // Added with the batch, not carried (retirement plan §37.4, breakage J): a felt stiffness read
    // 1% high in BOTH the force and the potential conserves perfectly and moves every signature too
    // little to see, and nothing in the workspace but the Windows-exact viewer freeze caught it.
    // So pin the magnitude directly: every step's applied force must be the discrete gradient of
    // `K [eta]+^(alpha+1) / (alpha+1)` between `eta^{n-1}` and `eta^{n+1}`, with `K` and `alpha`
    // written HERE rather than read from the model (§29.3). Lossless felt, so there is no
    // hysteretic term, and the reported force is exactly that evaluation — equality, not a band.
    let mut mal = mallet(STRIKE);
    let node = mal.params().node;
    let mut in_contact = 0;
    for _ in 0..1200 {
        let eta_prev = mal.membrane.u_prev[node] - mal.state().z_h_prev;
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

// == criterion 2: conservation is discrete-gradient-limited ======================================

#[test]
fn the_drift_scales_with_the_newton_tolerance() {
    let drifts: Vec<f64> = [1e-14, 1e-10, 1e-6]
        .iter()
        .map(|&newton_tol| {
            drift(&run(
                &mut mallet(Strike {
                    newton_tol,
                    ..STRIKE
                }),
                2500,
            ))
        })
        .collect();
    println!("drift vs tol 1e-14/1e-10/1e-6: {drifts:?}");
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
            5000,
        );
        assert!(e.iter().all(|v| v.is_finite()), "non-finite energy");
        let rise = nan_max(e.windows(2).map(|w| w[1] - w[0]));
        println!(
            "loss sigma={sigma} lam_h={hysteresis}: max rise {:e} E0, end {} E0",
            rise / e[0],
            e[e.len() - 1] / e[0]
        );
        assert!(
            rise <= 1e-9 * e[0],
            "loss added energy (sigma = {sigma}, lam_h = {hysteresis})"
        );
    }
}

// == criterion 4: a mallet that never arrives is the bare head to the bit ========================

#[test]
fn a_mallet_that_never_arrives_leaves_the_head_bit_identical() {
    // A plucked, lossy head wrapped in a mallet held 10 m off and moving AWAY: no contact, so the
    // coupling force is a true zero every step and the field must be the bare membrane's exactly.
    let (n, steps) = (40, 400);
    let mut bare = head(Domain::Rectangle, n, 0.5, 0.3);
    let p = bare.params().clone();
    let phi: Vec<f64> =
        p.x.iter()
            .zip(&p.y)
            .map(|(&x, &y)| (PI * x).sin() * (PI * y).sin() * 1e-3)
            .collect();
    let live = p.to_live(&phi);
    bare.set_displacement(&live);
    let mut struck = head(Domain::Rectangle, n, 0.5, 0.3);
    struck.set_displacement(&live);
    let mut mal = strike_on(
        Strike {
            v0: -1.0,
            gap: 10.0,
            ..STRIKE
        },
        struck,
    );
    for _ in 0..steps {
        bare.step();
        mal.step().expect("free flight cannot fail");
    }
    assert_eq!(mal.state().contact_force, 0.0);
    assert!(!mal.state().in_contact);
    assert_eq!(mal.membrane.u, bare.u);
}

#[test]
fn the_struck_head_runs_through_the_resonator_interface() {
    let mut mal = mallet(STRIKE);
    let k = mal.membrane.params().k;
    let res = simulate(&mut mal, 300, Some(0), 100).expect("a converged strike");
    assert_eq!(Resonator::timestep(&mal), k);
    assert_eq!(res.fs, 1.0 / k);
    let full = Resonator::state(&mal);
    assert_eq!(full.len(), mal.membrane.params().mask.flags().len());
    assert_eq!(full, mal.membrane.state());
    assert_eq!(Resonator::displacement_at(&mal, 0), mal.membrane.u[0]);
    assert_eq!(
        res.output.expect("a pickup was requested")[300],
        mal.membrane.u[0]
    );
}

#[test]
fn a_struck_circular_head_conserves_too() {
    // Energy is geometry-independent: the staircased disk, struck at its centre.
    let mut mal = mallet(Strike {
        domain: Domain::Circle,
        x: 0.0,
        y: 0.0,
        ..STRIKE
    });
    assert_eq!((mal.params().x_strike, mal.params().y_strike), (0.0, 0.0));
    let d = drift(&run(&mut mal, 4000));
    println!("circle drift {d:e}");
    assert!(d < CONSERVE_TOL, "circular-membrane energy drift {d:e}");
}

// == physical signatures =========================================================================

/// The Python's mallet `_spectral_centroid` — unlike the barrier's, it removes the mean first and
/// guards the denominator.
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

/// The normalised analytic eigenmode `sin(m pi x / Lx) sin(n pi y / Ly)` at the live nodes.
fn mode(mem: &Membrane, m: u32, n: u32) -> Vec<f64> {
    let p = mem.params();
    let (lx, ly) = (p.lx.expect("a rectangle"), p.ly.expect("a rectangle"));
    let (xs, ys) = (p.to_live(&p.x), p.to_live(&p.y));
    let phi: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| (f64::from(m) * PI * x / lx).sin() * (f64::from(n) * PI * y / ly).sin())
        .collect();
    let norm = phi.iter().map(|v| v * v).sum::<f64>().sqrt();
    phi.iter().map(|v| v / norm).collect()
}

fn project(phi: &[f64], u: &[f64]) -> f64 {
    phi.iter().zip(u).map(|(a, b)| a * b).sum()
}

#[test]
fn a_strike_excites_several_modes() {
    // Projected on the analytic eigenmodes — more robust than counting FFT peaks.
    let mut mal = mallet(Strike {
        k: 8.0e4,
        x: 0.3,
        y: 0.4,
        ..STRIKE
    });
    let modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 3)];
    let basis: Vec<Vec<f64>> = modes
        .iter()
        .map(|&(m, n)| mode(&mal.membrane, m, n))
        .collect();
    let mut peak = 0.0f64;
    let mut proj = vec![0.0f64; modes.len()];
    for _ in 0..6000 {
        mal.step().expect("contact solve converged");
        peak = nan_max([peak, mal.membrane.energy()].into_iter());
        for (p, phi) in proj.iter_mut().zip(&basis) {
            *p = nan_max([*p, project(phi, &mal.membrane.u).abs()].into_iter());
        }
    }
    // The head holds a real share at the peak of contact; the elastic mallet then bounces off with
    // much of it, so the share at the END is smaller — which is why this compares against it.
    let end = mal.energy();
    let top = nan_max(proj.iter().copied());
    let count = proj.iter().filter(|&&p| p > 0.1 * top).count();
    println!(
        "modes: peak/end {}, projections {proj:?}, {count} above 10%",
        peak / end
    );
    assert!(
        peak > 1e-2 * end,
        "the strike deposited almost no energy in the head"
    );
    assert!(
        count >= 3,
        "only {count} modes meaningfully excited — strike not broadband"
    );
}

#[test]
fn a_harder_mallet_is_shorter_and_brighter() {
    let mut soft = mallet(Strike {
        k: 1.0e4,
        x: 0.3,
        y: 0.4,
        ..STRIKE
    });
    let mut hard = mallet(Strike {
        k: 2.0e5,
        x: 0.3,
        y: 0.4,
        ..STRIKE
    });
    let i_s = soft.membrane.params().pickup_index_at(0.6, 0.55);
    let i_h = hard.membrane.params().pickup_index_at(0.6, 0.55);
    let (mut cs, mut ch) = (0usize, 0usize);
    let (mut ss, mut sh) = (Vec::with_capacity(6000), Vec::with_capacity(6000));
    for _ in 0..6000 {
        soft.step().expect("contact solve converged");
        hard.step().expect("contact solve converged");
        cs += usize::from(soft.state().in_contact);
        ch += usize::from(hard.state().in_contact);
        ss.push(soft.membrane.u[i_s]);
        sh.push(hard.membrane.u[i_h]);
    }
    let fs = soft.membrane.params().fs;
    let (bs, bh) = (centroid(&ss, fs), centroid(&sh, fs));
    println!("hardness: pickup {i_s}; contact soft {cs} hard {ch}; centroid soft {bs} hard {bh}");
    assert!(
        ch < cs,
        "stiffer felt should shorten contact: hard={ch} soft={cs} steps"
    );
    assert!(bh > bs, "stiffer felt should brighten the spectrum");
}

#[test]
fn the_mallet_bounces_off() {
    // Arrives moving in, leaves moving away: exactly one contact episode, then clear.
    let mut mal = mallet(Strike {
        k: 1.0e5,
        x: 0.3,
        y: 0.4,
        ..STRIKE
    });
    let mut contact = Vec::with_capacity(2500);
    for _ in 0..2500 {
        mal.step().expect("contact solve converged");
        contact.push(mal.state().in_contact);
    }
    let episodes = contact.windows(2).filter(|w| w[1] && !w[0]).count() + usize::from(contact[0]);
    println!(
        "bounce: {episodes} episode(s), {} steps in contact, exit {} m/s",
        contact.iter().filter(|&&c| c).count(),
        mal.mallet_velocity()
    );
    assert_eq!(
        episodes, 1,
        "expected a single bounce, saw {episodes} contact episodes"
    );
    assert!(
        !contact[contact.len() - 1],
        "mallet never separated from the head"
    );
    assert!(
        mal.mallet_velocity() > 0.0,
        "mallet did not rebound (still moving into the head)"
    );
}

#[test]
fn the_strike_position_combs_out_the_modes_it_sits_on() {
    // A centre strike of a SQUARE head is on the (2,1) mode's node line, so cannot excite it.
    let comb = |x: f64, y: f64| {
        let mut mal = mallet(Strike { x, y, ..STRIKE });
        let phi = mode(&mal.membrane, 2, 1);
        let mut p = 0.0f64;
        for _ in 0..4000 {
            mal.step().expect("contact solve converged");
            p = nan_max([p, project(&phi, &mal.membrane.u).abs()].into_iter());
        }
        p
    };
    let (centre, offset) = (comb(0.5, 0.5), comb(0.3, 0.4));
    println!(
        "mode comb: centre {centre:e}, offset {offset:e}, ratio {:e}",
        offset / (centre + 1e-30)
    );
    assert!(
        offset > 50.0 * (centre + 1e-30),
        "(2,1) mode should be nulled by a centre strike: centre={centre:e} offset={offset:e}"
    );
}

#[test]
fn the_mallet_and_its_head_share_a_timestep() {
    let mal = mallet(Strike {
        n: 32,
        x: 0.4,
        y: 0.4,
        v0: 2.0,
        ..STRIKE
    });
    assert_eq!(mal.params().k, mal.membrane.params().k);
    assert_eq!(
        Resonator::state(&mal).len(),
        mal.membrane.params().mask.flags().len()
    );
}
