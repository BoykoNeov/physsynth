//! Native bars for the simply-supported orthotropic plate — model #5o, three bending ratios.
//!
//! Carried from `tests/test_plate_orthotropic.py` (retirement plan §25). Wood is roughly ten times
//! stiffer along the grain than across it, so the supported plate takes three ratios instead of
//! one stiffness and `B = L²` becomes
//!
//! ```text
//! B = g_x (δ_xx)² + 2 g_h (δ_xx δ_yy) + g_y (δ_yy)²
//! ```
//!
//! This model has a *closed-form* oracle rather than a convergence rate:
//! `sin(mπx/Lx) sin(nπy/Ly)` is an exact discrete eigenvector of `δ_xx` and `δ_yy` separately, so
//! it survives orthotropy and carries its analytic frequency with it. The order of trust is the
//! Python file's: the operator against the closed form, the isotropic default on the untouched
//! `L @ L` line, the ledger (which proves less here than it looks like it proves), the guard on
//! the cross term — then the findings, two of them negative.
//!
//! **Two numbers here come from outside this project's code**, and they were recorded on
//! 2026-09-29 before the Python file was deleted (NumPy 2.4.6, SciPy 1.17.1, wheel freshly
//! reinstalled): LAPACK's eigenvalues of the guard's two operators (`LAPACK_*`), and the gap
//! between SciPy's sparse `L @ L` and the general assembly (`SCIPY_SQUARING_GAP`). Everything else
//! the Python asserted already ran through the Rust binding, so it was Rust checking Rust.

use physsynth_analysis::modal::{
    cents, dirichlet_axis_eigenvalue, discrete_orthotropic_plate_eigenfrequency,
    discrete_plate_eigenfrequency, orthotropic_plate_freqs, rectangular_discrete_eigenvalues,
    rectangular_plate_freqs,
};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::eig::symmetric_eigenvalues;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::ops2d::{laplacian_from_mask, orthotropic_biharmonic, rectangle_mask};
use physsynth_core::plate::{
    grain_ratios_from_material, pickup_index_at, Boundary, ParamError, Params, Plate, PlateSpec,
};
use physsynth_core::sparse::Csr;
use std::f64::consts::PI;

/// `tests/helpers.py`'s `KAPPA_PLATE_DEFAULT` and `RHO_AREAL_DEFAULT`.
const KAPPA: f64 = 20.0;
const RHO: f64 = 0.005;
const THETA: f64 = 0.28;
/// Tier 1: the project's acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

/// Passivity is asserted against a roundoff bar relative to the initial energy, not a bare
/// `<= 0.0`. Every test here plucks from rest, so step 0 dissipates essentially nothing (the Python
/// file measured -2.70e-18 against -1.50e-10 for the next step) and its sign is decided by
/// summation order, which is how the Python's bare `<= 0.0` passed locally and failed on CI. The
/// bar is ~3e-17 here, still five orders below a genuine decrement.
const PASSIVITY_ROUNDOFF: f64 = 1e-12;

// -- fixtures ---------------------------------------------------------------------------------

/// `(g_x, g_h, g_y)` — the supported branch's three ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Grain {
    x: f64,
    cross: f64,
    y: f64,
}

/// ~ the spruce ratios, rounded.
const G_STRONG: Grain = Grain {
    x: 1.0,
    cross: 0.153,
    y: 0.0727,
};
/// Nothing is made of this — the core API takes ratios so a test can dial it anyway.
const G_WILD: Grain = Grain {
    x: 11.0,
    cross: 2.5,
    y: 0.9,
};
const ISOTROPIC: Grain = Grain {
    x: 1.0,
    cross: 1.0,
    y: 1.0,
};

/// `tests/helpers.py`'s `SPRUCE`, through the material helper.
fn spruce() -> Grain {
    material(1.0)
}

/// Spruce with every modulus scaled by `s`: the same ratios in exact arithmetic.
fn material(s: f64) -> Grain {
    let m = grain_ratios_from_material(s * 11.0e9, s * 0.8e9, 0.37, s * 0.7e9, 3.0e-3, 420.0)
        .expect("spruce is admissible");
    Grain {
        x: m.grain_x,
        cross: m.grain_cross,
        y: m.grain_y,
    }
}

/// `make_orthotropic_plate`: a supported plate at plate-Courant number `mu`, `fs` solved from it.
fn spec(n: i64, mu: f64, sigma: f64, (lx, ly): (f64, f64), g: Grain) -> PlateSpec {
    let h = lx / n as f64;
    PlateSpec {
        lx,
        ly,
        kappa: KAPPA,
        rho: RHO,
        fs: KAPPA / (mu * h * h),
        n,
        sigma,
        theta: THETA,
        grain_x: g.x,
        grain_cross: Some(g.cross),
        grain_y: g.y,
        ..PlateSpec::default()
    }
}

fn params(n: i64, mu: f64, sigma: f64, dims: (f64, f64), g: Grain) -> Params {
    Params::new(&spec(n, mu, sigma, dims, g)).expect("an admissible grained plate")
}

fn square(n: i64, mu: f64, g: Grain) -> Params {
    params(n, mu, 0.0, (1.0, 1.0), g)
}

/// `orthotropic_mode_freqs`: the theta-scheme frequency of each NAMED mode, on the plate's own
/// snapped `Ly`. Named rather than "the lowest k", because under a grain the ordering moves.
fn mode_freqs(p: &Params, modes: &[(i64, i64)]) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, n)| {
            let lam_x = dirichlet_axis_eigenvalue(m as f64, p.lx, p.h);
            let lam_y = dirichlet_axis_eigenvalue(n as f64, p.ly, p.h);
            discrete_orthotropic_plate_eigenfrequency(
                lam_x,
                lam_y,
                p.kappa,
                p.k,
                p.theta,
                p.grain_x,
                p.grain_cross,
                p.grain_y,
            )
            .expect("a definite grain")
        })
        .collect()
}

/// The full-grid field restricted to the live nodes, in the plate's own ordering.
fn to_live(full: &[f64], p: &Params) -> Vec<f64> {
    full.iter()
        .zip(p.mask.flags())
        .filter(|(_, &alive)| alive)
        .map(|(v, _)| *v)
        .collect()
}

/// `sin(mπx/Lx) sin(nπy/Ly)` sampled on the live nodes.
fn sine_field(p: &Params, m: i64, n: i64) -> Vec<f64> {
    let ncols = p.mask.ncols();
    let mut v = Vec::with_capacity(p.n_live);
    for (idx, &alive) in p.mask.flags().iter().enumerate() {
        if alive {
            let (i, j) = ((idx % ncols) as f64, (idx / ncols) as f64);
            v.push((m as f64 * PI * i * p.h / p.lx).sin() * (n as f64 * PI * j * p.h / p.ly).sin());
        }
    }
    v
}

/// A raised-cosine pluck centred at `(cx Lx, cy Ly)`, restricted to the live nodes.
fn pluck_at(p: &Params, (cx, cy): (f64, f64), amplitude: f64) -> Vec<f64> {
    let width = 0.25 * p.lx.min(p.ly);
    let full = raised_cosine_2d(&p.x, &p.y, (cx * p.lx, cy * p.ly), width, amplitude).unwrap();
    to_live(&full, p)
}

fn pluck(p: &Params) -> Vec<f64> {
    pluck_at(p, (0.4, 0.55), 1e-3)
}

fn started(p: Params, u0: &[f64]) -> Plate {
    let mut plate = Plate::new(p);
    plate.set_state(u0, &vec![0.0; u0.len()]);
    plate
}

/// `int(secs * fs)` steps — Python's truncation.
fn run(plate: &mut Plate, secs: f64, pickup: Option<usize>) -> SimResult {
    let steps = (secs * plate.p.fs) as usize;
    simulate(plate, steps, pickup, 0).expect("a linear plate cannot fail a step")
}

fn assert_monotone(e: &[f64], why: &str) {
    let bar = PASSIVITY_ROUNDOFF * e[0];
    let worst = e
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        worst <= bar,
        "{why}: max step {worst:.3e} exceeds {bar:.3e}"
    );
    assert!(e[e.len() - 1] < e[0], "{why}: energy did not fall at all");
}

fn dense(k: &Csr) -> Vec<f64> {
    let (n, m) = (k.nrows(), k.ncols());
    let mut d = vec![0.0; n * m];
    for i in 0..n {
        for idx in k.indptr()[i]..k.indptr()[i + 1] {
            d[i * m + k.indices()[idx]] += k.data()[idx];
        }
    }
    d
}

fn norm(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) / y).abs())
        .fold(0.0f64, f64::max)
}

/// Indices that sort `f` ascending — `np.argsort`. Callers assert no exact ties first, because
/// NumPy's default sort leaves the order of a tie unspecified.
fn argsort(f: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..f.len()).collect();
    idx.sort_by(|&a, &b| f[a].total_cmp(&f[b]));
    idx
}

fn assert_no_ties(f: &[f64]) {
    let mut s = f.to_vec();
    s.sort_by(f64::total_cmp);
    assert!(
        s.windows(2).all(|w| w[0] != w[1]),
        "an exact tie makes the ordering comparison ill-posed: {s:?}"
    );
}

// -- tier 1: the contract ---------------------------------------------------------------------

#[test]
fn the_analytic_sine_is_an_exact_eigenvector_of_the_grained_operator() {
    // The money test. `B v = (g_x λ_x² + 2 g_h λ_x λ_y + g_y λ_y²) v` on the analytic sine. A plate
    // has no `kappa = 0` anchor, so the operator is proved directly — and as a RESIDUAL, which an
    // approximate eigenvector cannot pass, rather than an eigen-solve, which would return a nearby
    // eigenvalue of some other vector.
    //
    // The bar is 1e-10, not the Python's 1e-11, because the Python's passed with 2.2x to spare:
    // measured worst 4.5e-12 (strong) and 3.0e-12 (wild). That is the rounding floor, not a
    // defect — `B`'s entries are ~1e8 (`g · 64/h⁴`) and the (1,1) eigenvalue ~1e3, so
    // `eps · |B| / q` is already ~1e-11. Any wiring error puts the residual near 1, so 1e-10
    // (22x) loses nothing.
    for g in [G_STRONG, G_WILD] {
        let p = params(24, 2.0, 0.0, (0.62, 0.43), g);
        let mut worst = 0.0f64;
        for (m, n) in [(1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (5, 4)] {
            let v = sine_field(&p, m, n);
            let lam_x = dirichlet_axis_eigenvalue(m as f64, p.lx, p.h);
            let lam_y = dirichlet_axis_eigenvalue(n as f64, p.ly, p.h);
            let q = g.x * lam_x * lam_x + 2.0 * g.cross * lam_x * lam_y + g.y * lam_y * lam_y;
            let bv = p.stiffness.matvec(&v);
            let r: Vec<f64> = bv.iter().zip(&v).map(|(b, x)| b - q * x).collect();
            worst = worst.max(norm(&r) / (q * norm(&v)));
        }
        assert!(
            worst < 1e-10,
            "{g:?}: sine is not an exact eigenvector, {worst:.2e}"
        );
    }
}

/// SciPy's `max|general - L @ L| / max|L @ L|` on the 16 x 16, `Lx = 0.7` grid, and how many
/// entries differ. Plain multiply-and-add with no transcendental, so it is a cross-platform claim.
const SCIPY_SQUARING_GAP: f64 = 1.70601310856e-16;
const SCIPY_SQUARING_NDIFF: usize = 195;

#[test]
fn the_isotropic_default_stays_on_the_untouched_squaring_path() {
    // The default plate must be BYTE-identical to `L @ L`. The general assembly agrees with it
    // only to rounding, and how closely is grid-dependent — which is why the default keeps its own
    // line. So this is asserted on a grid KNOWN to distinguish the two paths.
    //
    // The reference `L @ L` is built here by hand (dense, `k` ascending — the order SciPy's sorted
    // CSR product accumulates in), not by `biharmonic_from_mask`: that is what `Params` calls, and
    // comparing a builder with itself proves nothing. The hand loop is faithful because it
    // reproduces SciPy's gap to the general assembly EXACTLY, in value and in count.
    let (nx, lx) = (16usize, 0.7);
    let h = lx / nx as f64;
    let (l, _) = laplacian_from_mask(&rectangle_mask(nx, nx), h);
    let n = l.nrows();
    let ld = dense(&l);
    let mut squared = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                let (a, b) = (ld[i * n + k], ld[k * n + j]);
                if a != 0.0 && b != 0.0 {
                    s += a * b;
                }
            }
            squared[i * n + j] = s;
        }
    }
    let general = dense(&orthotropic_biharmonic(nx, nx, h, 1.0, 1.0, 1.0).0);
    let diffs: Vec<f64> = general
        .iter()
        .zip(&squared)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let gap = max_abs(&diffs) / max_abs(&squared);
    let ndiff = diffs.iter().filter(|d| **d != 0.0).count();
    assert!(
        0.0 < gap && gap < 1e-15,
        "this grid was chosen because the two differ; gap {gap:.2e} (0 means no teeth)"
    );
    assert_eq!(
        gap, SCIPY_SQUARING_GAP,
        "the reference is not SciPy's L @ L"
    );
    assert_eq!(ndiff, SCIPY_SQUARING_NDIFF);

    let spec = PlateSpec {
        lx,
        ly: lx,
        kappa: KAPPA,
        rho: RHO,
        fs: 40_000.0,
        n: nx as i64,
        theta: THETA,
        ..PlateSpec::default()
    };
    let p = Params::new(&spec).unwrap();
    assert!(p.grain_is_isotropic);
    let b = dense(&p.stiffness);
    assert_eq!(b, squared, "the default plate left the L @ L path");
    assert_ne!(
        b, general,
        "the default plate is built by the general assembly"
    );
}

#[test]
fn a_uniform_grain_is_an_isotropic_plate_of_stiffness_kappa_sqrt_r() {
    // `g = (r, r, r)` is an isotropic plate of stiffness `kappa sqrt(r)` — through the NEW path.
    // The twin is built at a MATCHED SAMPLE RATE: solving fs from its larger kappa would give it a
    // different timestep and a ~3% frequency gap that looks like a modelling bug.
    let r = 2.7;
    let a = square(
        24,
        2.0,
        Grain {
            x: r,
            cross: r,
            y: r,
        },
    );
    assert!(
        !a.grain_is_isotropic,
        "the uniform grain must take the general assembly"
    );
    let b = Params::new(&PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA * r.sqrt(),
        rho: RHO,
        fs: a.fs,
        n: 24,
        theta: a.theta,
        ..PlateSpec::default()
    })
    .unwrap();
    assert!(
        a.k == b.k && a.h == b.h,
        "the rig error, made impossible to make silently"
    );

    let modes = [(1, 1), (2, 1), (1, 2), (3, 2)];
    let f_iso: Vec<f64> = rectangular_discrete_eigenvalues(b.h, 24, 24, &modes)
        .iter()
        .map(|&lam| discrete_plate_eigenfrequency(lam, b.kappa, b.k, b.theta))
        .collect();
    let f_ortho = mode_freqs(&a, &modes);
    let rel = max_rel(&f_ortho, &f_iso);
    // Measured 3.6e-16; bar 1e-14 (28x).
    assert!(
        rel < 1e-14,
        "uniform grain is not an isotropic plate: {rel:.2e}"
    );

    // ... and it STEPS identically, to rounding: the two operators differ by a reassociation, so
    // the states separate by the last bit per step. Measured 1.9e-12 after 200 steps; bar 1e-11
    // (5x).
    let u0 = pluck(&a);
    let mut pa = started(a, &u0);
    let mut pb = started(b, &u0);
    for _ in 0..200 {
        pa.step(None);
        pb.step(None);
    }
    let d: Vec<f64> = pa.u.iter().zip(&pb.u).map(|(x, y)| x - y).collect();
    let gap = max_abs(&d) / max_abs(&pb.u);
    assert!(
        gap < 1e-11,
        "uniform grain diverges from its twin: {gap:.2e}"
    );
}

#[test]
fn the_continuum_law_reduces_to_the_isotropic_one_and_the_grain_is_not_vacuous() {
    let (lx, ly) = (0.62, 0.43);
    let modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)];
    let f_iso = rectangular_plate_freqs(KAPPA, lx, ly, &modes);
    let via = orthotropic_plate_freqs(KAPPA, lx, ly, &modes, 1.0, 1.0, 1.0).unwrap();
    assert!(
        max_rel(&via, &f_iso) < 1e-14,
        "the orthotropic law does not reduce to the isotropic one — suspect the factor of 2"
    );
    let g = G_STRONG;
    let grained = orthotropic_plate_freqs(KAPPA, lx, ly, &modes, g.x, g.cross, g.y).unwrap();
    assert!(
        grained.iter().zip(&f_iso).all(|(a, b)| a <= b),
        "every ratio is <= 1, so no mode may rise"
    );
    assert!(
        grained[0] < 0.5 * f_iso[0],
        "the fundamental must drop by more than an octave"
    );
    // Not every mode drops much. Compared on a SQUARE, so only the material differs: on the
    // rectangle, (1,3) puts its half-waves across the shorter side and a shape asymmetry would
    // impersonate a material one.
    let stiff = orthotropic_plate_freqs(KAPPA, lx, lx, &[(3, 1)], g.x, g.cross, g.y).unwrap()[0];
    let soft = orthotropic_plate_freqs(KAPPA, lx, lx, &[(1, 3)], g.x, g.cross, g.y).unwrap()[0];
    assert!(
        stiff > 2.5 * soft,
        "along the grain {stiff:.1} Hz vs across {soft:.1} Hz"
    );
}

#[test]
fn the_discrete_law_converges_to_the_continuum_at_second_order() {
    // The spatial operator is exact on these modes, so the whole error is the per-axis
    // `(4/h²)sin²(mπh/2L) -> (mπ/L)²` gap plus a temporal error that scales as h⁴ at fixed mu.
    let (lx, mu) = (1.0, 0.5);
    let modes = [(1i64, 1i64), (2, 1)];
    let g = G_STRONG;
    let f_cont = orthotropic_plate_freqs(KAPPA, lx, lx, &modes, g.x, g.cross, g.y).unwrap();
    let (mut hs, mut errs) = (Vec::new(), Vec::new());
    for n in [16, 32, 64, 128] {
        let h = lx / n as f64;
        let k = mu * h * h / KAPPA;
        let err = modes
            .iter()
            .zip(&f_cont)
            .map(|(&(m, nn), fc)| {
                let lam_x = dirichlet_axis_eigenvalue(m as f64, lx, h);
                let lam_y = dirichlet_axis_eigenvalue(nn as f64, lx, h);
                let fd = discrete_orthotropic_plate_eigenfrequency(
                    lam_x, lam_y, KAPPA, k, THETA, g.x, g.cross, g.y,
                )
                .unwrap();
                (fd - fc).abs()
            })
            .fold(0.0f64, f64::max);
        hs.push(h);
        errs.push(err);
    }
    assert!(
        errs.windows(2).all(|w| w[1] < w[0]),
        "errors not decreasing: {errs:?}"
    );
    let last = errs.len() - 1;
    let order = (errs[last - 1] / errs[last]).ln() / (hs[last - 1] / hs[last]).ln();
    // Measured 2.003.
    assert!(order > 1.8, "order {order:.2} < 1.8");
}

#[test]
fn the_time_stepper_rings_at_the_grained_frequency() {
    // The operator bars prove the matrix; this proves it is the matrix being stepped. It is also
    // the one bar in the file where the frequency oracle meets something it did not compute.
    let p = square(48, 1.0, G_STRONG);
    let f_11 = mode_freqs(&p, &[(1, 1)])[0];
    let pickup = pickup_index_at(0.3 * p.lx, 0.28 * p.ly, &p);
    let u0 = pluck(&p);
    let mut plate = started(p, &u0);
    let res = run(&mut plate, 0.5, Some(pickup));
    let out = res.output.as_ref().unwrap();
    let found = measure_partials_near(out, res.fs, &[f_11], Some(20.0))[0];
    let err = cents(found, f_11).abs();
    // Measured 0.023 cents (36.87521 Hz found, 36.87471 predicted); bar 5 cents.
    assert!(
        err < 5.0,
        "grained fundamental off by {err:.2} cents ({found:.2} vs {f_11:.2})"
    );
}

/// The frequency the STEPPER runs an exact eigenmode at, read off the mode's own recurrence.
///
/// On an exact eigenvector the θ-scheme is the three-term recurrence `a⁺ + a⁻ = 2c a` in the mode's
/// amplitude, with `c = cos(ωk)`. So `c` is a least-squares fit over the run's triples and `ω` is
/// `acos(c)/k` — no spectrum, no window, nothing the frequency oracles compute. It is what the plate
/// ACTUALLY rings at, to rounding.
fn stepped_frequency(plate: &mut Plate, phi: &[f64], steps: usize) -> f64 {
    let norm2: f64 = phi.iter().map(|v| v * v).sum();
    let amplitude = |pl: &Plate| pl.u.iter().zip(phi).map(|(u, f)| u * f).sum::<f64>() / norm2;
    let mut a = vec![amplitude(plate)];
    for _ in 0..steps {
        plate.step(None);
        a.push(amplitude(plate));
    }
    let (mut num, mut den) = (0.0, 0.0);
    for w in a.windows(3) {
        num += w[1] * (w[0] + w[2]);
        den += 2.0 * w[1] * w[1];
    }
    (num / den).acos() / (2.0 * PI * plate.p.k)
}

#[test]
fn both_frequency_oracles_read_the_theta_the_stepper_runs_at() {
    // Added at the human's call (retirement plan §46.4). Every bar in the workspace built its plates
    // at the default θ = 0.28, so `discrete_plate_eigenfrequency` and
    // `discrete_orthotropic_plate_eigenfrequency` could BOTH ignore their `θ` and use 0.28 and all
    // 101 test binaries stayed green: the one bar that read either oracle at another θ held them
    // against EACH OTHER (`the_orthotropic_oracles_reduce_to_the_isotropic_ones`), which sees a
    // defect in one copy and is blind to a defect both share — finding #78's twin again. The viewer
    // and the horizon read-outs pass a plate's own θ through these, so the defect is a wrong in-tune
    // limit for any plate run at a non-default θ.
    //
    // So each oracle is held against the plate ITSELF at four θs, the default among them: started on
    // an exact eigenmode at a large timestep (μ = 13, where `Qk²` is near one and θ moves the pitch
    // by percents), its stepped frequency must be the oracle's at the plate's own θ. Measured: every
    // one of the 16 cases agrees to <= 8.6e-14 relative (bar 1e-9); θ = 0.5 moves the isotropic
    // (1,1) 8.8% from the default's pitch, and the nearest θ, 0.25, still moves every mode by
    // >= 0.49% (the orthotropic (1,1); control bar 0.1%).
    let (n, mu) = (16, 13.0);
    for theta in [0.25, 0.28, 0.5, 1.0] {
        for (g, oracle) in [(ISOTROPIC, "isotropic"), (G_STRONG, "orthotropic")] {
            for (m, nn) in [(1, 1), (2, 1)] {
                let p = Params::new(&PlateSpec {
                    theta,
                    ..spec(n, mu, 0.0, (1.0, 1.0), g)
                })
                .expect("an admissible plate");
                let lam_x = dirichlet_axis_eigenvalue(m as f64, p.lx, p.h);
                let lam_y = dirichlet_axis_eigenvalue(nn as f64, p.ly, p.h);
                let predicted = if oracle == "isotropic" {
                    discrete_plate_eigenfrequency(lam_x + lam_y, p.kappa, p.k, p.theta)
                } else {
                    discrete_orthotropic_plate_eigenfrequency(
                        lam_x,
                        lam_y,
                        p.kappa,
                        p.k,
                        p.theta,
                        p.grain_x,
                        p.grain_cross,
                        p.grain_y,
                    )
                    .unwrap()
                };
                let at_default = if oracle == "isotropic" {
                    discrete_plate_eigenfrequency(lam_x + lam_y, p.kappa, p.k, THETA)
                } else {
                    discrete_orthotropic_plate_eigenfrequency(
                        lam_x,
                        lam_y,
                        p.kappa,
                        p.k,
                        THETA,
                        p.grain_x,
                        p.grain_cross,
                        p.grain_y,
                    )
                    .unwrap()
                };
                let phi = sine_field(&p, m, nn);
                let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
                let mut plate = started(p, &u0);
                let stepped = stepped_frequency(&mut plate, &phi, 400);
                let gap = (stepped - predicted).abs() / predicted;
                assert!(
                    gap < 1e-9,
                    "{oracle} ({m},{nn}) at theta = {theta}: the plate rings at {stepped:.12} Hz, \
                     the oracle says {predicted:.12} Hz ({gap:.2e} apart)"
                );
                // The control asserts that it differs: at μ = 13 a θ other than the default must
                // move the pitch, or the bar above could not see an oracle stuck at 0.28.
                if theta != THETA {
                    let moved = (stepped - at_default).abs() / stepped;
                    assert!(
                        moved > 1e-3,
                        "{oracle} ({m},{nn}): theta = {theta} moved the pitch only {moved:.2e} \
                         from the default's — the rig cannot tell theta apart"
                    );
                }
            }
        }
    }
}

#[test]
fn a_grained_plate_conserves_its_energy() {
    // B stays symmetric (the cross term is a product of two commuting symmetric factors), and the
    // energy uses the same matrix the update does — at timesteps no explicit scheme could run.
    for mu in [0.5, 2.0, 8.0] {
        let p = square(32, mu, G_STRONG);
        let u0 = pluck(&p);
        let mut plate = started(p, &u0);
        let drift = run(&mut plate, 0.5, None).energy_drift();
        // Measured 1.3e-13, 4.4e-12, 1.8e-13 at mu = 0.5, 2, 8: at worst 23x under the bar.
        assert!(drift < DRIFT_TOL, "drift {drift:.2e} at mu = {mu}");
    }
}

#[test]
fn a_lossy_grained_plate_is_passive_and_its_fundamental_decays_at_two_sigma() {
    // On ONE low mode: the top of the spectrum is effectively undamped at any usable timestep (the
    // shipped theta caveat), so a broadband pluck decays at whatever mixture it happened to hold.
    let p = params(24, 2.0, 4.0, (1.0, 1.0), G_STRONG);
    let two_sigma = 2.0 * p.sigma;
    let u0: Vec<f64> = sine_field(&p, 1, 1).iter().map(|v| v * 1e-3).collect();
    let mut plate = started(p, &u0);
    let e = run(&mut plate, 0.3, None).energy;
    assert_monotone(&e, "energy increased somewhere in a lossy run");
    let rate = -(e[e.len() - 1] / e[0]).ln() / 0.3;
    // Measured 0.45% off; bar 2% (4.4x). Deterministic physics, not rounding.
    assert!(
        (rate / two_sigma - 1.0).abs() < 0.02,
        "fundamental decays at {rate:.3}/s, want ~{two_sigma:.3}/s"
    );
}

// -- the guard: what replaces the isotropic operator's free definiteness ----------------------

/// LAPACK's (`np.linalg.eigvalsh`) smallest eigenvalue of the N = 8 operator at 0.98x the floor,
/// and of the one at 1.02x.
const LAPACK_INSIDE_MIN: f64 = 56.543_774_203_884_176;
const LAPACK_OUTSIDE_MIN: f64 = -1_639.643_772_293_387_2;
/// LAPACK's largest eigenvalue of the inside operator (the outside one's is 651749.84). A dense
/// solve's error is absolute, about `eps · lambda_max` ~ 1.4e-10 here, so the comparison is
/// measured in that unit: 20 of them, ~2.9e-9. The native solver sat 0.13 and 1.2 units away.
const LAPACK_INSIDE_MAX: f64 = 652_353.802_180_589;
const EIG_FLOOR_BAR: f64 = 20.0 * f64::EPSILON * LAPACK_INSIDE_MAX;

#[test]
fn the_cross_term_guard_is_sharp_and_rejected_at_construction() {
    // `g_h > -sqrt(g_x g_y)` or B is indefinite and the theta-scheme's "unconditional" stability,
    // which quietly assumes a definite B, is gone. Sharp: 1.02x the floor is indefinite, 0.98x not.
    let (gx, gy): (f64, f64) = (11.0, 0.9);
    let floor = -(gx * gy).sqrt();
    for cross in [floor * 1.02, floor] {
        let e = Params::new(&spec(
            8,
            2.0,
            0.0,
            (1.0, 1.0),
            Grain {
                x: gx,
                cross,
                y: gy,
            },
        ))
        .expect_err("an indefinite cross term");
        assert!(matches!(e, ParamError::IndefiniteCross { .. }), "{e:?}");
        assert!(e.to_string().contains("indefinite"), "{e}");
    }
    let ok = square(
        8,
        2.0,
        Grain {
            x: gx,
            cross: floor * 0.98,
            y: gy,
        },
    );
    let n = ok.n_live;
    let lo = symmetric_eigenvalues(&dense(&ok.stiffness), n).unwrap()[0];
    assert!(
        lo > 0.0,
        "just inside the guard the operator is indefinite ({lo:.3e})"
    );
    // Measured 1.9e-11 from LAPACK.
    assert!(
        (lo - LAPACK_INSIDE_MIN).abs() < EIG_FLOOR_BAR,
        "{lo} vs LAPACK {LAPACK_INSIDE_MIN}"
    );

    // ... and just outside it really is indefinite: the guard is not merely conservative.
    let bad = orthotropic_biharmonic(8, 8, 1.0 / 8.0, gx, floor * 1.02, gy).0;
    let lo = symmetric_eigenvalues(&dense(&bad), bad.nrows()).unwrap()[0];
    assert!(lo < 0.0, "the guard rejects a valid operator");
    // Measured 1.8e-10 from LAPACK: 16x under the bar, where the Python-era 1e-9 would be 5.7x.
    assert!(
        (lo - LAPACK_OUTSIDE_MIN).abs() < EIG_FLOOR_BAR,
        "{lo} vs LAPACK {LAPACK_OUTSIDE_MIN}"
    );
}

#[test]
fn degenerate_grain_ratios_are_refused() {
    for (x, y) in [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)] {
        let e = Params::new(&spec(8, 2.0, 0.0, (1.0, 1.0), Grain { x, cross: 1.0, y }))
            .expect_err("a non-positive ratio");
        assert!(matches!(e, ParamError::NonPositiveGrain(..)), "{e:?}");
        assert!(e.to_string().contains("must be positive"), "{e}");
    }
}

#[test]
fn a_grain_on_the_free_boundary_needs_the_split_and_says_so() {
    // The free plate needs the coupling and torsional rigidities SEPARATELY, not just H. A grain
    // without them is refused rather than completed from Poisson's ratio — a wrong default, not a
    // permitted unphysical choice.
    let common = PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho: RHO,
        fs: 40_000.0,
        n: 8,
        boundary: Some(Boundary::Free),
        grain_y: 0.5,
        ..PlateSpec::default()
    };
    let e = Params::new(&common).expect_err("a grain with no split");
    assert!(matches!(e, ParamError::FreeNeedsSplit), "{e:?}");
    assert!(e.to_string().contains("separately"), "{e}");
    let p = Params::new(&PlateSpec {
        grain_coupling: Some(0.1),
        grain_torsion: Some(0.3),
        ..common
    })
    .expect("with the split it builds");
    assert_eq!(
        p.grain_cross,
        0.1 + 2.0 * 0.3,
        "grain_cross must be derived from the split"
    );
}

// -- the material chain: where the factor of 2 lives -------------------------------------------

#[test]
fn an_isotropic_material_returns_exactly_no_grain_and_an_areal_density_by_name() {
    // `plate.rs`'s `isotropic_material_comes_back_at_exactly_one` pins the `(1, 1, 1)` and the
    // split; carried here are the clauses it does not assert.
    let (e, nu, t, rho) = (1.1e10, 0.3, 3.0e-3, 420.0);
    let s = grain_ratios_from_material(e, e, nu, e / (2.0 * (1.0 + nu)), t, rho).unwrap();
    assert_eq!((s.grain_x, s.grain_cross, s.grain_y), (1.0, 1.0, 1.0));
    let d = e * t * t * t / (12.0 * (1.0 - nu * nu));
    assert!((s.kappa / (d / (rho * t)).sqrt() - 1.0).abs() < 1e-14);
    // The density handed to Plate is AREAL and comes back by name, so the volume density cannot
    // be passed through by accident — that slip leaves every frequency right (kappa carries them)
    // and every energy wrong by 1/t, here 333x, and no detector in this file would see it.
    assert_eq!(s.rho_s, rho * t);
    assert!((rho / s.rho_s - 1.0 / t).abs() < 1e-9);
    assert_ne!(
        s.rho_s, rho,
        "if these coincide the trap is invisible and so is this test"
    );
}

#[test]
fn the_two_rival_cross_term_packagings_are_measurably_wrong() {
    // The literature invites `H = D_1` and `H = D_1 + D_xy`. Both are positive, both keep B
    // definite, and both give a stable, exactly conservative, WRONG plate. Priced here so a
    // transcription slip has a known signature: 0.30x and 0.65x at nu = 0.3.
    let (e, nu, t): (f64, f64, f64) = (1.1e10, 0.3, 3.0e-3);
    let d = e * t * t * t / (12.0 * (1.0 - nu * nu));
    let g = e / (2.0 * (1.0 + nu));
    let (d_1, d_xy) = (nu * d, g * t * t * t / 12.0);
    assert!(
        ((d_1 + 2.0 * d_xy) / d - 1.0).abs() < 1e-14,
        "the correct packaging"
    );
    assert!((d_1 / d - 0.30).abs() < 5e-3, "H = D_1 alone");
    assert!(((d_1 + d_xy) / d - 0.65).abs() < 5e-3, "H = D_1 + D_xy");
}

#[test]
fn spruce_is_not_a_stretched_isotropic_plate() {
    // `H / sqrt(D_x D_y) ~ 0.57`, not 1: a stretched isotropic plate would sit at exactly 1, so
    // the cross term is an independent axis. `plate.rs` asserts the ratio band too; the stiffness
    // ratio is this file's.
    let g = spruce();
    assert!(
        13.0 < g.x / g.y && g.x / g.y < 15.0,
        "stiffness ratio {:.2}",
        g.x / g.y
    );
    let ratio = g.cross / (g.x * g.y).sqrt();
    assert!(0.5 < ratio && ratio < 0.65, "H/sqrt(Dx Dy) = {ratio:.3}");
}

// -- findings ----------------------------------------------------------------------------------

#[test]
fn the_energy_ledger_cannot_see_a_wrongly_wired_grain() {
    // Three deliberate mutations — grain swapped end for end, factor of 2 dropped, cross term taken
    // as D_1 alone. Every one is a symmetric definite operator, so the ledger is GREEN for all
    // three (measured drifts 1.7e-13, 5.1e-13, 3.0e-13); only the modal oracle separates them.
    let t = spruce();
    let modes = [(1, 1), (2, 1), (1, 2), (2, 2)];
    let drift_of = |g: Grain| {
        let p = square(24, 2.0, g);
        let u0 = pluck(&p);
        let mut plate = started(p, &u0);
        run(&mut plate, 0.2, None).energy_drift()
    };
    let f_true = mode_freqs(&square(24, 2.0, t), &modes);
    assert!(drift_of(t) < DRIFT_TOL);

    let mutations = [
        (
            "swapped",
            Grain {
                x: t.y,
                cross: t.cross,
                y: t.x,
            },
        ),
        (
            "factor_2_dropped",
            Grain {
                cross: t.cross / 2.0,
                ..t
            },
        ),
        (
            "D_1_only",
            Grain {
                cross: t.cross * 0.30,
                ..t
            },
        ),
    ];
    for (name, g) in mutations {
        let drift = drift_of(g);
        assert!(
            drift < DRIFT_TOL,
            "'{name}' broke conservation; the finding is that it does NOT, so the mutation is wrong"
        );
        // Measured 1.25, 0.095 and 0.136 — the dropped factor of 2 moves the modes only 9.5%,
        // 1.9x over the 5% bar, and is the mutation this detector sees least well.
        let worst = max_rel(&mode_freqs(&square(24, 2.0, g), &modes), &f_true);
        assert!(
            worst > 0.05,
            "'{name}' is within 5% of the truth, {worst:.3}"
        );
    }

    // The control: the same material with every modulus scaled by 1.1, which lands the ratios on
    // the same values in exact arithmetic and a different last bit in doubles. Both detectors must
    // stay green (measured 2.5e-16 in frequency; `cross` and `y` differ in the last bit). The
    // Python control rebuilt the IDENTICAL plate and so asserted nothing; this one asserts that it
    // differs, so it cannot silently become that again.
    let c = material(1.1);
    assert_ne!(
        c, t,
        "the control is bit-identical to the truth and would assert nothing"
    );
    assert!(drift_of(c) < DRIFT_TOL);
    let rel = max_rel(&mode_freqs(&square(24, 2.0, c), &modes), &f_true);
    assert!(
        rel < 1e-14,
        "a last-bit reassociation moved the modes by {rel:.2e}"
    );
}

#[test]
fn a_square_plates_diagonal_modes_are_blind_to_the_grain_running_the_wrong_way() {
    // Swapping g_x and g_y while λ_x = λ_y leaves g_x λ_x² + g_y λ_y² untouched, so a check of a
    // square plate's fundamental or its diagonal modes PASSES a grain running 90 degrees wrong.
    // Invariant in exact arithmetic; in doubles 0 or ~2e-16 by mode (measured worst 1.9e-16), hence
    // a bound, not equality. The two ways out below measured 2.26x and 1.62x.
    let g = spruce();
    let swapped = Grain {
        x: g.y,
        cross: g.cross,
        y: g.x,
    };
    let right = square(32, 2.0, g);
    let wrong = square(32, 2.0, swapped);
    let diagonal = [(1, 1), (2, 2), (3, 3), (4, 4)];
    let gap = max_rel(
        &mode_freqs(&wrong, &diagonal),
        &mode_freqs(&right, &diagonal),
    );
    assert!(
        gap < 1e-14,
        "diagonal modes are supposed to be blind to the swap; {gap:.2e}"
    );

    // Way out 1: an off-diagonal mode sees it at once.
    let off = mode_freqs(&right, &[(2, 1)])[0] / mode_freqs(&wrong, &[(2, 1)])[0];
    assert!(
        (off - 1.0).abs() > 0.5,
        "the off-diagonal mode should catch it; {off:.3}"
    );

    // Way out 2: on a non-square plate even the fundamental does.
    let r2 = params(32, 2.0, 0.0, (0.62, 0.43), g);
    let w2 = params(32, 2.0, 0.0, (0.62, 0.43), swapped);
    let fund = mode_freqs(&w2, &[(1, 1)])[0] / mode_freqs(&r2, &[(1, 1)])[0];
    assert!(
        (fund - 1.0).abs() > 0.3,
        "a rectangle's fundamental should catch it; {fund:.3}"
    );
}

#[test]
fn the_grain_makes_the_theta_damping_anisotropic_and_the_ledger_stays_green() {
    // The theta average turns flat loss into 2σ(1 - θ Q k²). Isotropically Q depends on Λ alone, so
    // (4,1) and (1,4) on a square decay identically; with a grain Q depends on how the curvature
    // splits, and the pair splits in decay rate. MEASURED, each mode started alone, with the
    // isotropic plate as the control in the same rig. Python recorded 5.857 / 5.857 and
    // 6.024 / 7.751 per second against a nominal 8.
    let (sigma, mu, secs) = (4.0, 4.0, 0.3);
    let rate = |g: Grain, m: i64, n: i64| {
        let p = params(24, mu, sigma, (1.0, 1.0), g);
        let u0: Vec<f64> = sine_field(&p, m, n).iter().map(|v| v * 1e-3).collect();
        let mut plate = started(p, &u0);
        let e = run(&mut plate, secs, None).energy;
        assert_monotone(&e, "the ledger must stay monotone — passivity is untouched");
        -(e[e.len() - 1] / e[0]).ln() / secs
    };
    let (iso_a, iso_b) = (rate(ISOTROPIC, 4, 1), rate(ISOTROPIC, 1, 4));
    assert!(
        (iso_a - iso_b).abs() / iso_a < 1e-6,
        "the isotropic control is not degenerate ({iso_a:.6} vs {iso_b:.6})"
    );
    let (a, b) = (rate(G_STRONG, 4, 1), rate(G_STRONG, 1, 4));
    let split = (a - b).abs() / a.max(b);
    // Measured 22.3% (6.024 vs 7.750/s); the isotropic pair agreed to 2.7e-12. As in Python.
    assert!(
        split > 0.15,
        "the pair still decays together ({a:.4} vs {b:.4})"
    );
    assert!(
        a < 2.0 * sigma && b < 2.0 * sigma,
        "under-damped: the caveat, not a new defect"
    );
}

#[test]
fn the_cross_term_detunes_selectively_without_reordering_anything() {
    // Hold the along/across ratio at spruce's 13.75 and sweep g_h from 0.2x to 1.0x of
    // sqrt(g_x g_y) — the value a stretched isotropic plate is pinned at (spruce sits at 0.567x).
    // Half one: the sixteen low modes never reorder, at two resolutions. Half two: they move by
    // 1.3% to 29% (measured: (3,1) 2.3%, (2,4) 29.0%), a 22x spread — the cross term matters
    // where the direct stiffness is weakest.
    // And past isotropy (2x) it DOES reorder: the result is about the range, not an inert term.
    let modes: Vec<(i64, i64)> = (1..5).flat_map(|m| (1..5).map(move |n| (m, n))).collect();
    let (gx, gy): (f64, f64) = (1.0, 1.0 / 13.75);
    let stretched = (gx * gy).sqrt();
    let freqs = |n: i64, factor: f64| {
        let p = square(
            n,
            1.0,
            Grain {
                x: gx,
                cross: factor * stretched,
                y: gy,
            },
        );
        assert!(
            (p.grain_x / p.grain_y / 13.75 - 1.0).abs() < 1e-9,
            "the ratio is held FIXED"
        );
        let f = mode_freqs(&p, &modes);
        assert_no_ties(&f);
        f
    };

    let physical = [0.2, 0.567, 1.0];
    let order =
        |n: i64| -> Vec<Vec<usize>> { physical.iter().map(|&f| argsort(&freqs(n, f))).collect() };
    let (o24, o48) = (order(24), order(48));
    assert!(
        o24.iter().all(|o| *o == o24[0]),
        "the ordering moved inside the physical range"
    );
    assert_eq!(o24, o48, "the two grids disagree about the ordering");

    let (lo, hi) = (freqs(48, 0.2), freqs(48, 1.0));
    let shift: Vec<f64> = hi.iter().zip(&lo).map(|(h, l)| h / l - 1.0).collect();
    let at = |mode: (i64, i64)| shift[modes.iter().position(|&m| m == mode).unwrap()];
    assert!(
        at((3, 1)) < 0.05,
        "along the grain should barely notice; {:.3}",
        at((3, 1))
    );
    assert!(
        at((2, 4)) > 0.25,
        "across the grain should notice strongly; {:.3}",
        at((2, 4))
    );
    let spread = shift.iter().cloned().fold(f64::MIN, f64::max)
        / shift.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread > 15.0,
        "the leverage is nearly uniform ({spread:.1}x)"
    );

    let beyond = argsort(&freqs(48, 2.0));
    assert_ne!(
        beyond, o48[0],
        "even at 2x nothing reorders; the 'range' framing overstates it"
    );
}

#[test]
fn the_grain_is_in_the_partial_series_and_not_in_the_level() {
    // A coupled instrument listens at a point, which sees a weighted sum over modes. With raw pitch
    // removed (an isotropic twin matched on the fundamental, at a matched SAMPLE RATE), the level
    // at one node straddles 1 across five unrelated geometries — the spread is geometry, not grain
    // — while the partial series moves by 37%, which no pluck position affects. Measured: shape
    // 0.372; ratios 1.117, 0.811, 0.875, 0.918, 0.835; twin pitch-matched to 2.4e-5.
    let modes: Vec<(i64, i64)> = (1..4).flat_map(|m| (1..4).map(move |n| (m, n))).collect();
    let g = spruce();
    let sorted = |mut f: Vec<f64>| {
        f.sort_by(f64::total_cmp);
        f
    };
    let iso_freqs = |p: &Params| -> Vec<f64> {
        sorted(
            rectangular_discrete_eigenvalues(p.h, 32, 32, &modes)
                .iter()
                .map(|&lam| discrete_plate_eigenfrequency(lam, p.kappa, p.k, p.theta))
                .collect(),
        )
    };

    let grained = square(32, 1.0, g);
    let f_grained = sorted(mode_freqs(&grained, &modes));
    let f_iso = iso_freqs(&square(32, 1.0, ISOTROPIC));
    let scale = f_grained[0] / f_iso[0];

    // (1) the partial series, pitch removed — a property of the operator.
    let shape = f_grained
        .iter()
        .zip(&f_iso)
        .map(|(a, b)| {
            let want = b / f_iso[0];
            ((a / f_grained[0] - want) / want).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(
        shape > 0.2,
        "the grained partial series barely reshaped ({shape:.3})"
    );

    // (2) the RMS at one node, over five unrelated pluck/pickup geometries.
    let ring = |p: Params, pluck_xy: (f64, f64), pick_xy: (f64, f64)| {
        let u0 = pluck_at(&p, pluck_xy, 1e-3);
        let idx = pickup_index_at(pick_xy.0 * p.lx, pick_xy.1 * p.ly, &p);
        let mut plate = started(p, &u0);
        let res = run(&mut plate, 0.15, Some(idx));
        let out = res.output.unwrap();
        (out.iter().map(|v| v * v).sum::<f64>() / out.len() as f64).sqrt()
    };
    let twin = |a: &Params| {
        Params::new(&PlateSpec {
            lx: a.lx,
            ly: a.ly,
            kappa: KAPPA * scale,
            rho: RHO,
            fs: a.fs,
            n: 32,
            theta: a.theta,
            ..PlateSpec::default()
        })
        .unwrap()
    };
    let geometries = [
        ((0.40, 0.55), (0.31, 0.27)),
        ((0.50, 0.50), (0.50, 0.50)),
        ((0.25, 0.70), (0.60, 0.35)),
        ((0.60, 0.30), (0.20, 0.80)),
        ((0.45, 0.45), (0.70, 0.70)),
    ];
    let mut ratios = Vec::new();
    for (pluck_xy, pick_xy) in geometries {
        let a = square(32, 1.0, g);
        let b = twin(&a);
        assert!(a.fs == b.fs && a.k == b.k);
        ratios.push(ring(a, pluck_xy, pick_xy) / ring(b, pluck_xy, pick_xy));
    }

    // The twin really is pitch-matched at this sample rate, so nothing below is raw pitch.
    let f_twin = iso_freqs(&twin(&grained));
    assert!(
        (f_twin[0] / f_grained[0] - 1.0).abs() < 1e-3,
        "the twin is not pitch-matched"
    );

    let (lo, hi) = (
        ratios.iter().cloned().fold(f64::MAX, f64::min),
        ratios.iter().cloned().fold(f64::MIN, f64::max),
    );
    assert!(
        lo < 1.0 && 1.0 < hi,
        "the level ratios {ratios:?} sit on one side of 1: the level DOES carry the grain"
    );
    // A JUDGEMENT about what still counts as geometry noise, not a derived threshold; the
    // load-bearing assertion is the straddle above, which cannot be tuned.
    let worst = ratios
        .iter()
        .map(|r| (r - 1.0).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 0.3, "the level moved by more than 30% ({ratios:?})");
}
