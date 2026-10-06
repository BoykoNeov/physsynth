//! The jawari — the sitar and tanpura's curved buzzing bridge — carried from `tests/test_jawari.py`
//! (retirement plan §37).
//!
//! The jawari is **not new core physics**: it is model #8 (`BarrierString`) against a *curved*
//! barrier hugging the `x = 0` termination, `b(x) = -clearance - depth (x/d)^2` on
//! `0 < x <= d = 0.15 L` and `-inf` beyond. The string wraps onto the curve on each downswing and
//! its departure point travels along it, which is the shimmering, sustained, high-partial-rich
//! "life" of these instruments. So the gates are model #8's, re-exercised in the curved and
//! persistently wrapping regime the flat rail and the point fret never reach — conservation and the
//! static-equilibrium magnitude oracle on a CURVED profile — and then the two signatures that must
//! separate the curve from a flat rail or a clean string: brightness that is **sustained** deep into
//! the decay, and a contact edge that **travels**.
//!
//! The fixture is the retired `make_jawari_string`: `N = 100`, `lam = 0.4`, `K = 2e6`,
//! `alpha = 1.5`, `depth = 1 mm`, a grazing crest (`clearance = 0`), and the barrier string's solve
//! defaults; the pluck is an 8 mm first mode. The profile is built here rather than imported: it
//! was a test helper in Python (`tests/helpers.py`'s `jawari_barrier`), the viewer keeps its own
//! copy, and its square is a multiply because NumPy's array `** 2` is one. All of it — profile,
//! support and pluck — was checked bit-identical to the Python's before the deletion.
//!
//! The two 12,000-step runs (the jawari and its clean contrast) feed three bars, so they run once
//! per test binary behind a `OnceLock`, as the Python's tests would have shared them had it been
//! able to.

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::collision::{penetration_of, BarrierString};
use physsynth_core::dense;
use physsynth_core::string_damped::{DampedStiffString, Params};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// The acceptance bar, unchanged — see CLAUDE.md.
const CONSERVE_TOL: f64 = 1e-10;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005;
const THETA: f64 = 0.28;
/// `JAWARI_WIDTH_FRAC_DEFAULT`, `JAWARI_DEPTH_DEFAULT`, `JAWARI_K_DEFAULT`.
const WIDTH_FRAC: f64 = 0.15;
const DEPTH: f64 = 1.0e-3;
const K_DEF: f64 = 2.0e6;
const ALPHA_DEF: f64 = 1.5;
/// First-mode pluck amplitude: it swings down onto the bridge.
const AMP: f64 = 8.0e-3;

#[derive(Clone, Copy)]
struct Jaw {
    lam: f64,
    k: f64,
    alpha: f64,
    depth: f64,
    clearance: f64,
    sigma0: f64,
}

const JAW: Jaw = Jaw {
    lam: 0.4,
    k: K_DEF,
    alpha: ALPHA_DEF,
    depth: DEPTH,
    clearance: 0.0,
    sigma0: 0.0,
};

/// The retired `jawari_barrier`: the parabolic bridge on the grid `x`, `-inf` off its span.
fn jawari_profile(x: &[f64], depth: f64, clearance: f64) -> Vec<f64> {
    let d = WIDTH_FRAC * L;
    x.iter()
        .map(|&xi| {
            if xi > 0.0 && xi <= d {
                let r = xi / d;
                -clearance - depth * (r * r)
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// The retired `make_jawari_string`.
fn jawari(j: Jaw) -> BarrierString {
    let n = 100;
    let fs = (T / RHO).sqrt() * n as f64 / (L * j.lam);
    let s = DampedStiffString::new(
        Params::new(L, T, RHO, fs, n, 0.0, j.sigma0, 0.0, THETA, true).expect("valid string"),
    );
    let b = jawari_profile(&s.p.grid(), j.depth, j.clearance);
    BarrierString::new(s, &b, j.k, j.alpha, 0.0, 1e-12, 1e-13, 60).expect("valid bridge")
}

fn pluck(bar: &mut BarrierString) {
    let x = bar.string.p.grid();
    let u0: Vec<f64> = x.iter().map(|&xi| AMP * (PI * xi / L).sin()).collect();
    bar.set_state(&u0, &vec![0.0; x.len()]);
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

/// The amplitude-weighted mean frequency of the Hann-windowed record.
fn centroid(sig: &[f64], fs: f64) -> f64 {
    let w = hann(sig.len());
    let x: Vec<f64> = sig.iter().zip(&w).map(|(s, w)| s * w).collect();
    let mag = rfft_mag(&x);
    let f = rfftfreq(sig.len(), 1.0 / fs);
    let num: f64 = f.iter().zip(&mag).map(|(f, m)| f * m).sum();
    let den: f64 = mag.iter().sum();
    num / den
}

/// One run's record: the mid-string pickup, and the **wrap edge** per step — the furthest-in-contact
/// support index, `-1` when clear — whose travel is the jawari's precessing departure point.
struct Record {
    pickup: Vec<f64>,
    wrap: Vec<i64>,
}

/// The Python's `_run_pickup_wrap`: pluck, run, record.
fn record(bar: &mut BarrierString, steps: usize) -> Record {
    pluck(bar);
    let node = ((0.5 * bar.string.p.n as f64) as usize).max(1);
    let mut pickup = Vec::with_capacity(steps);
    let mut wrap = Vec::with_capacity(steps);
    for _ in 0..steps {
        bar.step();
        let edge = bar
            .contact_mask()
            .iter()
            .rposition(|&m| m)
            .map_or(-1, |j| j as i64);
        wrap.push(edge);
        pickup.push(bar.string.u[node]);
    }
    Record { pickup, wrap }
}

const LONG: usize = 12_000;

/// The jawari at `sigma0 = 0.5` and its clean contrast (the crest dropped 1 m, so it never
/// contacts), 12,000 steps each — shared by the brightness, sustain and travel bars.
fn shared() -> &'static (Record, Record, f64) {
    static RUNS: OnceLock<(Record, Record, f64)> = OnceLock::new();
    RUNS.get_or_init(|| {
        let mut jaw = jawari(Jaw { sigma0: 0.5, ..JAW });
        let mut clean = jawari(Jaw {
            sigma0: 0.5,
            clearance: 1.0,
            ..JAW
        });
        let fs = 1.0 / jaw.string.p.k;
        (record(&mut jaw, LONG), record(&mut clean, LONG), fs)
    })
}

/// `np.std` (population) of the wrap edge over the steps in contact.
fn edge_spread(wrap: &[i64]) -> f64 {
    let v: Vec<f64> = wrap
        .iter()
        .filter(|&&w| w >= 0)
        .map(|&w| w as f64)
        .collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64).sqrt()
}

// == money gate 1 — lossless conservation through the curved wrap ================================

#[test]
fn lossless_energy_is_conserved_through_the_curved_wrap() {
    for &alpha in &[1.0, 1.5, 2.0] {
        let mut bar = jawari(Jaw { alpha, ..JAW });
        pluck(&mut bar);
        let mut e = vec![bar.energy()];
        for i in 1..=6000 {
            bar.step();
            assert!(
                bar.s.newton_iters < bar.p.newton_maxiter,
                "contact solve stalled at step {i}"
            );
            e.push(bar.energy());
        }
        let d = nan_max(e.iter().map(|v| (v - e[0]).abs())) / e[0].abs();
        println!("jawari lossless alpha={alpha}: drift {d:e}");
        assert!(
            d < CONSERVE_TOL,
            "jawari energy drift {d:e} (alpha = {alpha})"
        );
    }
}

#[test]
fn the_string_actually_wraps_the_bridge() {
    // Else conservation is a free-string re-test: demand multi-node contact and a real force.
    let mut bar = jawari(Jaw { sigma0: 0.3, ..JAW });
    pluck(&mut bar);
    let (mut span, mut force, mut steps) = (0usize, 0.0f64, 0usize);
    for _ in 0..6000 {
        bar.step();
        let n = bar.contact_mask().iter().filter(|&&m| m).count();
        if n > 0 {
            steps += 1;
            span = span.max(n);
        }
        force = nan_max(bar.s.contact_force.iter().map(|f| f.abs()).chain([force]));
    }
    println!("wrap: {steps} steps, widest span {span} nodes, peak force {force:e} N");
    assert!(steps > 300, "bridge barely touched ({steps} steps)");
    assert!(
        span >= 3,
        "string never wrapped multiple nodes (max span {span})"
    );
    assert!(
        force > 1.0,
        "peak contact force {force:e} N too small to test the coupling"
    );
}

// == money gate 2 — the static-equilibrium magnitude oracle on a CURVED profile ==================

/// `alpha = 1` with the crest preloaded 2 mm ABOVE rest, so the whole curved span is in gentle
/// contact at rest and the discrete gradient sits on its no-warp Taylor branch.
const STATIC: Jaw = Jaw {
    lam: 0.5,
    k: 3000.0,
    alpha: 1.0,
    depth: 1.5e-3,
    clearance: -2.0e-3,
    sigma0: 0.0,
};

/// `S u* = (K/rho) b`, `S = -L + (K/rho) diag(mask)` over the bridge support — built from the
/// fixture's own `K` and `rho`, not from the model's admittance.
fn curved_equilibrium(bar: &BarrierString) -> Vec<f64> {
    let m = bar.string.p.interior();
    let kr = STATIC.k / RHO;
    let mut on = vec![false; m];
    let mut rhs = vec![0.0; m];
    for (j, &node) in bar.p.support.iter().enumerate() {
        on[node - 1] = true;
        rhs[node - 1] = kr * bar.p.b[j];
    }
    let mut s = vec![0.0; m * m];
    for i in 0..m {
        for jj in 0..m {
            let bed = if i == jj && on[i] { kr } else { 0.0 };
            s[i * m + jj] = -bar.string.p.op_l.get(i, jj) + bed;
        }
    }
    let f = dense::lu_factor(s, m).expect("S is SPD");
    dense::lu_solve(&f, &rhs).expect("square")
}

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

#[test]
fn the_curved_static_equilibrium_matches_the_closed_form() {
    let mut bar = jawari(STATIC);
    let ustar = curved_equilibrium(&bar);
    seat(&mut bar, &ustar);
    assert!(
        bar.s.penetration.iter().all(|&p| p > 0.0),
        "not all support in contact — raise the preload"
    );
    let d = hold(&mut bar, &ustar, 2000);
    let scale = nan_max(ustar.iter().map(|v| v.abs()));
    println!("curved bed: held to {d:e} (deflection {scale:e})");
    assert!(
        d < 1e-13 * scale.max(1.0),
        "held curved equilibrium to only {d:e} (deflection {scale:e}) — magnitude off"
    );
}

#[test]
fn the_curved_static_equilibrium_negative_control_has_teeth() {
    let mut good = jawari(STATIC);
    let ustar = curved_equilibrium(&good);
    seat(&mut good, &ustar);
    let good_drift = hold(&mut good, &ustar, 400);
    let mut bad = jawari(STATIC);
    seat(&mut bad, &ustar);
    bad.p.g_mat.iter_mut().for_each(|g| *g *= 2.0);
    bad.p.force_pref *= 2.0;
    let bad_drift = hold(&mut bad, &ustar, 400);
    println!("curved control: good {good_drift:e}, doubled {bad_drift:e}");
    assert!(
        bad_drift > 1e4 * good_drift,
        "negative control too weak: good {good_drift:e}, doubled-coupling {bad_drift:e}"
    );
}

// == signature 1 — the bridge buzzes =============================================================

#[test]
fn the_jawari_brightens_the_tone() {
    let (jaw, clean, fs) = shared();
    assert!(
        jaw.wrap.iter().any(|&w| w >= 0),
        "jawari never contacted the bridge"
    );
    let (cj, cc) = (centroid(&jaw.pickup, *fs), centroid(&clean.pickup, *fs));
    println!("jawari brightness: {cj}, clean {cc}, ratio {}", cj / cc);
    assert!(
        cj > 2.0 * cc,
        "bridge did not brighten the tone: clean={cc:.0} jawari={cj:.0}"
    );
}

// == signature 2 — shimmer: the brightness is SUSTAINED ==========================================

#[test]
fn the_jawari_brightness_is_sustained() {
    // A clean string's mid pickup sits near its fundamental throughout; the curved contact
    // re-injects highs every downswing, so late in the decay the jawari is still much brighter and
    // has not collapsed back toward its own fundamental.
    let (jaw, clean, fs) = shared();
    let half = LONG / 2;
    let (j_e, j_l) = (
        centroid(&jaw.pickup[..half], *fs),
        centroid(&jaw.pickup[half..], *fs),
    );
    let (c_e, c_l) = (
        centroid(&clean.pickup[..half], *fs),
        centroid(&clean.pickup[half..], *fs),
    );
    let (j_ratio, c_ratio) = (j_l / j_e, c_l / c_e);
    println!(
        "sustain: jawari {j_e} -> {j_l}, clean {c_e} -> {c_l}; late ratio {}",
        j_l / c_l
    );
    assert!(
        j_l > 2.5 * c_l,
        "late-window brightness not dominated by the bridge (jawari {j_l:.0} vs clean {c_l:.0})"
    );
    assert!(
        j_l > 0.7 * j_e,
        "jawari brightness collapsed (early {j_e:.0} -> late {j_l:.0})"
    );
    assert!(
        j_ratio > c_ratio,
        "jawari not more sustained than clean string (jawari {j_ratio:.2} vs {c_ratio:.2})"
    );
}

// == signature 3 — the contact point travels =====================================================

#[test]
fn the_wrap_edge_travels_more_than_on_a_flat_rail() {
    // On the curve the departure point sweeps along the bridge; on a flat rail at the same MINIMUM
    // clearance (the crest's height, on the same support) contact is a cluster near the antinode
    // crossing. Both buzz; only the curve travels. The rail is written before the pluck so the
    // continuation seed reads it; the admittance block is the curve's, as in the original, since
    // the support is unchanged.
    let (jaw, _, _) = shared();
    let mut flat = jawari(Jaw { sigma0: 0.5, ..JAW });
    let crest = flat.p.b[0];
    flat.p.b = vec![crest; flat.p.support_len()];
    let flat = record(&mut flat, LONG);
    let (sj, sf) = (edge_spread(&jaw.wrap), edge_spread(&flat.wrap));
    println!(
        "wrap travel: curve {sj}, flat {sf}, ratio {} (crest {crest:e})",
        sj / sf
    );
    assert!(
        sj > 1.5 * sf,
        "wrap edge does not travel more on the curve (curve {sj:.2} vs flat {sf:.2})"
    );
}

// == sanity — the profile and the support ========================================================

#[test]
fn the_jawari_profile_is_a_curved_ramp() {
    let x: Vec<f64> = (0..=100).map(|i| i as f64 * (1.0 / 100.0)).collect();
    let b = jawari_profile(&x, 1.0e-3, 0.0);
    assert!(
        !b[0].is_finite(),
        "the x = 0 termination must be off-support (-inf)"
    );
    let on: Vec<f64> = b.iter().copied().filter(|v| v.is_finite()).collect();
    assert!(!on.is_empty());
    assert!(
        on.windows(2).all(|w| w[1] < w[0]),
        "bridge must curve monotonically away from the crest"
    );
    for (xi, bi) in x.iter().zip(&b) {
        if *xi > WIDTH_FRAC {
            assert!(
                !bi.is_finite(),
                "no barrier beyond the bridge span (x = {xi})"
            );
        }
    }
}

#[test]
fn the_bridge_support_resolves_the_wrap_and_stays_under_the_dense_solve_cliff() {
    let bar = jawari(JAW);
    let m = bar.p.support_len();
    assert!(m >= 8, "too few bridge nodes to resolve the wrap ({m})");
    assert!(m < 100, "support over the dense-solve cliff ({m})");
}
