//! Native bars for the completely free (FFFF) plate — model #5b: energy, operator, modes.
//!
//! Carried from `tests/test_free_plate_energy.py` and `tests/test_free_plate_modal.py` (retirement
//! plan §27). The free plate is the curved-Chladni resonator and has **no closed-form modal
//! oracle** — free edges have no `sin·sin` eigenvector — so it rests on three independent anchors
//! of increasing externality: the rigid-body nullspace `{1, x, y}` with the `K(xy) != 0` twist
//! counter-check, O(h²) self-convergence of the low eigenvalues, and Leissa's FFFF-square
//! frequency parameters, with the fundamental the twisting saddle rather than a drum bulge.
//!
//! **SciPy retires as a referee here, with its numbers.** Every eigenvalue the Python file used
//! came from ARPACK. Following §24.3's rule, the recorded referee is LAPACK's DENSE solve of the
//! same pencil (`W^{-1/2} K W^{-1/2}`, `dsyevr`), taken on 2026-09-29 before deletion (NumPy 2.4.6,
//! SciPy 1.17.1, wheel reinstalled); the ARPACK values are in retirement plan §27.1. A dense
//! solve's error is absolute, about `eps · mu_max`, and `mu_max` grows like `h^-4` — 2.2e-9 at
//! N = 20, 5.8e-7 at N = 80 — so every comparison is written in units of that floor rather than as
//! a bare number.

use physsynth_analysis::modal::{
    cents, discrete_beam_eigenfrequency, free_plate_ffff_square_lambdas,
    free_plate_freq_from_lambda, rectangular_discrete_eigenvalues,
};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::ops::free_beam_stiffness;
use physsynth_core::ops2d::{biharmonic_from_mask, free_plate_stiffness, rectangle_mask};
use physsynth_core::plate::{pickup_index_at, Boundary, Params, Plate, PlateSpec};
use physsynth_core::sparse::Csr;
use std::f64::consts::PI;

/// `tests/helpers.py`'s `KAPPA_PLATE_DEFAULT` and `RHO_AREAL_DEFAULT`.
const KAPPA: f64 = 20.0;
const RHO: f64 = 0.005;
/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

// -- the recorded referee ---------------------------------------------------------------------

/// LAPACK's lowest generalized eigenvalues `mu = ω²/κ²` of the free unit square at `nu = 0.3`:
/// `(N, mu_max, [three rigid, then the elastic ones])`. Independent of `mu` (the plate-Courant
/// number): `K` and `W` do not see the timestep.
const LAPACK: [(i64, f64, [f64; 8]); 5] = [
    (
        20,
        10_129_380.036_304_263,
        [
            3.913_448_117_630_575_5e-10,
            3.913_448_117_630_575_5e-10,
            3.913_448_117_630_575_5e-10,
            180.020_837_139_766_4,
            378.173_862_007_084_95,
            580.006_677_072_531,
            1_192.395_450_452_714_6,
            1_192.395_450_454_352_6,
        ],
    ),
    (
        32,
        66_811_230.475_849_25,
        [
            -8.992_805_631_363_962e-10,
            -8.992_805_631_363_962e-10,
            1.021_414_346_001_301_3e-8,
            180.905_919_659_404_62,
            382.246_828_724_384_7,
            586.110_966_653_963,
            1_205.379_175_897_516_2,
            1_205.379_175_908_629_8,
        ],
    ),
    (
        40,
        163_367_158.059_895_72,
        [
            -3.770_434_748_151_444e-9,
            -3.770_434_748_151_444e-9,
            -3.770_434_748_151_444e-9,
            181.102_691_790_034_1,
            383.110_836_066_117_7,
            587.432_315_729_770_7,
            1_208.142_866_487_93,
            1_208.142_866_487_93,
        ],
    ),
    (
        64,
        1_072_500_798.029_334_1,
        [
            3.081_653_385_286_272e-8,
            3.081_653_385_286_272e-8,
            3.081_653_385_286_272e-8,
            181.304_775_713_603_9,
            383.930_496_440_610_57,
            588.729_278_641_584_6,
            1_210.780_277_820_43,
            1_210.780_277_820_43,
        ],
    ),
    (
        80,
        2_619_484_720.081_145,
        [
            1.075_566_248_444_488_1e-7,
            1.075_566_248_444_488_1e-7,
            1.075_566_248_444_488_1e-7,
            181.347_234_981_736_1,
            384.075_390_548_542_8,
            588.977_354_647_219_5,
            1_211.253_349_974_708_7,
            1_211.253_349_974_708_7,
        ],
    ),
];

/// The recorded row for grid `n`.
fn lapack(n: i64) -> (f64, [f64; 8]) {
    let row = LAPACK.iter().find(|r| r.0 == n).expect("a recorded grid");
    (row.1, row.2)
}

/// The comparison unit: a dense solve's absolute floor on this pencil.
fn floor(n: i64) -> f64 {
    f64::EPSILON * lapack(n).0
}

// -- fixtures ---------------------------------------------------------------------------------

/// `make_free_plate`: a completely free unit square at plate-Courant number `mu`, `nu = 0.3`.
fn free(n: i64, mu: f64, sigma: f64, rho: f64) -> Params {
    let h = 1.0 / n as f64;
    Params::new(&PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho,
        fs: KAPPA / (mu * h * h),
        n,
        sigma,
        boundary: Some(Boundary::Free),
        nu: Some(0.3),
        ..PlateSpec::default()
    })
    .expect("an admissible free plate")
}

fn plate(n: i64) -> Params {
    free(n, 2.0, 0.0, RHO)
}

/// `free_plate_low_eigenfrequencies`' solve: the `count` lowest generalized eigenpairs of
/// `K φ = mu W φ` (rigid ones included) by shift-invert at `-1e-3 (13/a²)²`.
fn lowest(p: &Params, count: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let a = p.lx;
    let sigma = -1e-3 * (13.0 / (a * a)).powi(2);
    let got = eigsh_shift_invert(&p.stiffness, p.mass.as_ref(), sigma, count)
        .expect("the shifted pencil factors");
    (got.values, got.vectors)
}

/// The `n` lowest ELASTIC eigenvalues, the three rigid-body modes discarded.
fn elastic(p: &Params, n: usize) -> Vec<f64> {
    lowest(p, n + 3).0[3..].to_vec()
}

fn freq(mu: f64) -> f64 {
    KAPPA * mu.max(0.0).sqrt() / (2.0 * PI)
}

/// Assert native against LAPACK in units of the dense floor, returning the worst distance in them.
///
/// Every call site measured at most 0.38 floors (N = 20 … 80) against a bar of 20.
fn against_lapack(n: i64, got: &[f64], from: usize, units: f64) -> f64 {
    let (_, want) = lapack(n);
    let worst = got
        .iter()
        .zip(&want[from..])
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f64, f64::max)
        / floor(n);
    assert!(
        worst < units,
        "N={n}: {worst:.2} dense floors from LAPACK (bar {units})"
    );
    worst
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

/// A smooth off-centre bump; every free node is live, so nothing is clamped.
fn pluck(p: &Params) -> Vec<f64> {
    let a = p.lx;
    raised_cosine_2d(&p.x, &p.y, (0.4 * a, 0.55 * a), 0.25 * a, 1e-3).unwrap()
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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// A field over the `(ny+1) x (nx+1)` grid from the origin corner, row-major — `np.meshgrid`.
fn field(nx: usize, ny: usize, h: f64, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut v = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            v.push(f(i as f64 * h, j as f64 * h));
        }
    }
    v
}

// -- tier 1: energy ----------------------------------------------------------------------------

#[test]
fn a_lossless_free_plate_conserves_energy_across_mu() {
    // K symmetric and W SPD make the W-weighted theta-scheme conservative unconditionally.
    // Measured 2.3e-11 / 8.8e-13 / 4.8e-13 at mu 0.5 / 2 / 8. The first is 4.4x under the
    // acceptance bar, which is the project's contract and is recorded, not moved.
    for mu in [0.5, 2.0, 8.0] {
        let p = free(32, mu, 0.0, RHO);
        let u0 = pluck(&p);
        let mut pl = started(p, &u0);
        let drift = run(&mut pl, 1.0, None).energy_drift();
        assert!(drift < DRIFT_TOL, "drift {drift:.2e} at mu = {mu}");
    }
}

#[test]
fn energy_is_conserved_at_a_timestep_an_explicit_plate_could_not_run() {
    let p = free(32, 16.0, 0.0, RHO);
    assert!(p.mu > 0.25, "would blow up an explicit plate immediately");
    let u0 = pluck(&p);
    let mut pl = started(p, &u0);
    let drift = run(&mut pl, 1.0, None).energy_drift();
    assert!(drift < DRIFT_TOL, "drift {drift:.2e} at mu = 16");
}

#[test]
fn lossless_energy_stays_strictly_positive() {
    let p = plate(32);
    let u0 = pluck(&p);
    let mut pl = started(p, &u0);
    let e = run(&mut pl, 0.5, None).energy;
    assert!(e.iter().all(|&v| v > 0.0), "a non-positive energy");
}

#[test]
fn a_lossy_free_plate_decreases_monotonically() {
    let p = free(32, 2.0, 8.0, RHO);
    let u0 = pluck(&p);
    let mut pl = started(p, &u0);
    let e = run(&mut pl, 1.0, None).energy;
    let worst = e
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::NEG_INFINITY, f64::max)
        / e[0];
    assert!(worst <= 1e-10, "max positive step {worst:.2e} * E0");
    assert!(
        e[e.len() - 1] < e[0],
        "a plate that never moved is monotone too"
    );
}

/// The `which`-th elastic eigenvector (0 = the saddle fundamental), scaled to `1e-3`.
fn elastic_mode(p: &Params, which: usize) -> Vec<f64> {
    let (_, vecs) = lowest(p, which + 4);
    vecs[3 + which].iter().map(|v| v * 1e-3).collect()
}

#[test]
fn the_saddle_fundamental_decays_at_two_sigma() {
    // A single LOW mode (Q k² << 1) decays at ~2 sigma.
    let (sigma, secs) = (6.0, 0.5);
    let p = free(32, 2.0, sigma, RHO);
    let u0 = elastic_mode(&p, 0);
    let mut pl = started(p, &u0);
    let e = run(&mut pl, secs, None).energy;
    let measured = e[e.len() - 1] / e[0];
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    // Measured 0.33%.
    assert!(rel < 0.03, "low-mode decay off by {:.3}%", rel * 100.0);
}

#[test]
fn a_higher_mode_underdamps_relative_to_the_fundamental() {
    // The damping caveat: 2σ(1 - θ Q k²) falls with mode, so a higher mode keeps MORE energy —
    // the opposite of a real plate. Coarse timestep so θ Q k² is visible for the high mode.
    let (sigma, secs) = (6.0, 0.3);
    let retained: Vec<f64> = [0usize, 12]
        .iter()
        .map(|&which| {
            let p = free(32, 8.0, sigma, RHO);
            let u0 = elastic_mode(&p, which);
            let mut pl = started(p, &u0);
            let e = run(&mut pl, secs, None).energy;
            e[e.len() - 1] / e[0]
        })
        .collect();
    // Measured 0.027 against 0.053 — the higher mode keeps twice as much.
    assert!(
        retained[1] > retained[0],
        "high/low retained = {retained:?}"
    );
}

#[test]
fn energy_is_in_joules_and_scales_with_areal_density() {
    let (p1, p2) = (free(32, 2.0, 0.0, 0.005), free(32, 2.0, 0.0, 0.010));
    let e1 = started(p1.clone(), &pluck(&p1)).energy();
    let e2 = started(p2.clone(), &pluck(&p2)).energy();
    assert!(
        (e2 / e1 - 2.0).abs() <= 2e-12,
        "E(2 rho) / E(rho) = {}",
        e2 / e1
    );
}

// -- tier 2: the operator ----------------------------------------------------------------------

#[test]
fn the_energy_first_operator_is_symmetric() {
    let (k, _, _) = free_plate_stiffness(12, 10, 0.1, 0.3, 1.0, 1.0, None, None);
    let (d, n) = (dense(&k), k.nrows());
    let mut asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            asym = asym.max((d[i * n + j] - d[j * n + i]).abs());
        }
    }
    assert!(asym < 1e-12, "K not symmetric: {asym:.3e}");
}

/// `K` by explicit per-node loops — no Kronecker products — the reference for the ordering.
/// Dense, row-major `nn x nn`.
fn direct_k(nx: usize, ny: usize, h: f64, nu: f64) -> Vec<f64> {
    let nn = (nx + 1) * (ny + 1);
    let idx = |i: usize, j: usize| j * (nx + 1) + i;
    let inv_h2 = 1.0 / (h * h);
    let mut dxx = vec![0.0; nn * nn];
    let mut dyy = vec![0.0; nn * nn];
    for j in 0..=ny {
        for i in 1..nx {
            let r = idx(i, j);
            dxx[r * nn + idx(i - 1, j)] += inv_h2;
            dxx[r * nn + idx(i, j)] += -2.0 * inv_h2;
            dxx[r * nn + idx(i + 1, j)] += inv_h2;
        }
    }
    for j in 1..ny {
        for i in 0..=nx {
            let r = idx(i, j);
            dyy[r * nn + idx(i, j - 1)] += inv_h2;
            dyy[r * nn + idx(i, j)] += -2.0 * inv_h2;
            dyy[r * nn + idx(i, j + 1)] += inv_h2;
        }
    }
    let mut wa = vec![0.0; nn];
    for j in 0..=ny {
        let wy = if 0 < j && j < ny { h } else { 0.5 * h };
        for i in 0..=nx {
            let wx = if 0 < i && i < nx { h } else { 0.5 * h };
            wa[idx(i, j)] = wx * wy;
        }
    }
    // (Aᵀ diag(w) B)[p][q] = Σ_r A[r][p] w[r] B[r][q].
    let gram = |a: &[f64], w: &[f64], b: &[f64], rows: usize| {
        let mut out = vec![0.0; nn * nn];
        for r in 0..rows {
            for p in 0..nn {
                let arp = a[r * nn + p];
                if arp == 0.0 {
                    continue;
                }
                for q in 0..nn {
                    out[p * nn + q] += arp * w[r] * b[r * nn + q];
                }
            }
        }
        out
    };
    let bxx = gram(&dxx, &wa, &dxx, nn);
    let byy = gram(&dyy, &wa, &dyy, nn);
    let cross = gram(&dxx, &wa, &dyy, nn);
    let ncell = nx * ny;
    let mut dxy = vec![0.0; ncell * nn];
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            dxy[c * nn + idx(i, j)] += inv_h2;
            dxy[c * nn + idx(i + 1, j)] += -inv_h2;
            dxy[c * nn + idx(i, j + 1)] += -inv_h2;
            dxy[c * nn + idx(i + 1, j + 1)] += inv_h2;
        }
    }
    let ones = vec![1.0; ncell];
    let twist = gram(&dxy, &ones, &dxy, ncell);
    let mut k = vec![0.0; nn * nn];
    for p in 0..nn {
        for q in 0..nn {
            let i = p * nn + q;
            let sym = cross[i] + cross[q * nn + p];
            k[i] = bxx[i] + byy[i] + nu * sym + 2.0 * (1.0 - nu) * (h * h) * twist[i];
        }
    }
    k
}

#[test]
fn the_kronecker_assembly_matches_a_direct_per_node_build() {
    // Non-square grids, so an x <-> y swap would show.
    for (nx, ny, nu) in [
        (4usize, 4usize, 0.3),
        (5, 3, 0.3),
        (4, 5, 0.0),
        (3, 4, 0.49),
    ] {
        let h = 0.1;
        let (k, w, index_map) = free_plate_stiffness(nx, ny, h, nu, 1.0, 1.0, None, None);
        let kd = dense(&k);
        let scale = kd.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let diff = max_abs_diff(&kd, &direct_k(nx, ny, h, nu)) / scale;
        // Relative to the largest entry (~2e3 here), not the Python's absolute 1e-12, which the
        // measured 4.5e-13 met with only 2.2x to spare: it is one ulp of rounding (2.4e-16
        // relative) measured on the wrong scale. 1e-14 is 42x.
        assert!(
            diff < 1e-14,
            "({nx},{ny},{nu}): kron K != direct assembly by {diff:.2e}"
        );
        let trivial: Vec<i64> = (0..((nx + 1) * (ny + 1)) as i64).collect();
        assert_eq!(index_map, trivial, "every node is live, in C order");
        let wmin = (0..w.nrows())
            .map(|i| w.get(i, i))
            .fold(f64::INFINITY, f64::min);
        assert!(
            (wmin - 0.25 * h * h).abs() <= 1e-8 + 1e-5 * 0.25 * h * h,
            "corner weight h²/4"
        );
    }
}

#[test]
fn the_rigid_body_nullspace_is_exact_and_the_saddle_is_not_in_it() {
    // RELATIVE to ||K||_F (the absolute residual of K x grows as h shrinks); the discriminating
    // signal is the contrast — the rigid modes sit many orders below the saddle xy.
    let (nx, ny, h) = (14usize, 12usize, 0.1);
    let (k, _, _) = free_plate_stiffness(nx, ny, h, 0.3, 1.0, 1.0, None, None);
    let k_fro = norm(k.data());
    let rel = |v: Vec<f64>| norm(&k.matvec(&v)) / (k_fro * norm(&v));
    let one = rel(field(nx, ny, h, |_, _| 1.0));
    let x = rel(field(nx, ny, h, |x, _| x));
    let y = rel(field(nx, ny, h, |_, y| y));
    let xy = rel(field(nx, ny, h, |x, y| x * y));
    for (name, r) in [("1", one), ("x", x), ("y", y)] {
        assert!(r < 1e-12, "K {name} not in the nullspace: {r:.2e}");
    }
    // Measured: {1, x, y} at 5e-19 to 4e-18, xy at 1.1e-5 — twelve orders of contrast.
    // Only the (1 - nu) twist term feeds the saddle; a dropped-nu operator would kill it.
    assert!(xy > 1e-9, "K xy spuriously ~0 (dropped nu?): {xy:.2e}");
    assert!(xy > 1e6 * one.max(x).max(y), "nullspace contrast too small");
}

#[test]
fn the_saddles_energy_scales_exactly_with_one_minus_nu() {
    let (nx, ny, h) = (12usize, 10usize, 0.1);
    let xy = field(nx, ny, h, |x, y| x * y);
    let scaled: Vec<f64> = [0.0, 0.3, 0.49, -0.5]
        .iter()
        .map(|&nu| {
            let (k, _, _) = free_plate_stiffness(nx, ny, h, nu, 1.0, 1.0, None, None);
            norm(&k.matvec(&xy)) / (1.0 - nu)
        })
        .collect();
    // The Python's `np.allclose(rtol=1e-12)` also carried `atol=1e-8` (finding (d)); this is the
    // relative claim alone. Measured spread 5.7e-15 on values of ~4.0.
    let spread = scaled
        .iter()
        .map(|s| ((s - scaled[0]) / scaled[0]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        spread < 1e-12,
        "K xy not proportional to 1 - nu: {scaled:?}"
    );
}

#[test]
fn the_plates_bending_diagonal_is_the_free_beam_operator_along_each_axis() {
    // C2xᵀ Wa C2x + C2yᵀ Wa C2y == kron(M_y, S_x) + kron(S_y, M_x) with S, M the validated free
    // beam: the plate's bending inherits the beam's symmetry, per-line {1, x} and O(h²).
    //
    // SHARPENED against the plate's OWN assembly. The Python rebuilt the diagonal from
    // `_collocated_d2_1d`, but the Rust plate assembles its curvatures from the mask
    // (`free_plate_stiffness_from_mask`) and never calls `collocated_d2_1d` — a planted error in
    // that function turned only this bar red (retirement plan §27.3). So the diagonal is read off
    // the real builder with the coupling and torsion switched off, leaving exactly the two bending
    // terms. Non-square, so an axis swap would show. Measured 5.7e-14 (18x).
    let (nx, ny, h) = (9usize, 7usize, 0.1);
    let (bend, _, _) = free_plate_stiffness(nx, ny, h, 0.3, 1.0, 1.0, Some(0.0), Some(0.0));
    let (sx, mx) = free_beam_stiffness(nx, h);
    let (sy, my) = free_beam_stiffness(ny, h);
    let beam = my.kron(&sx).add(&sy.kron(&mx));
    let diff = max_abs_diff(&dense(&bend), &dense(&beam));
    assert!(
        diff < 1e-12,
        "the plate's bending diagonal != free-beam operator by {diff:.2e}"
    );
}

// -- tier 3: the spectrum ----------------------------------------------------------------------

#[test]
fn exactly_three_modes_are_rigid() {
    // A collocated centred u_xy would add a spurious (-1)^{i+j} near-zero mode. The cell-centred
    // twist must leave exactly {1, x, y}.
    let p = plate(20);
    let (vals, _) = lowest(&p, 8);
    let n_zero = vals
        .iter()
        .filter(|v| v.abs() < 1e-3 * vals[3].abs())
        .count();
    assert_eq!(n_zero, 3, "near-zero modes: {vals:?}");
    // Against LAPACK: the rigid ones are zero to the dense floor, the elastic ones agree.
    let (_, want) = lapack(20);
    for v in &vals[..3] {
        assert!(v.abs() < 100.0 * floor(20), "a rigid eigenvalue {v:e}");
    }
    assert!(want[..3].iter().all(|v| v.abs() < 100.0 * floor(20)));
    against_lapack(20, &vals[3..], 3, 20.0);
}

#[test]
fn the_low_eigenvalues_self_converge_at_second_order() {
    // Richardson on h, h/2, h/4 — needs no external table.
    let mus: Vec<Vec<f64>> = [20i64, 40, 80]
        .iter()
        .map(|&n| elastic(&plate(n), 4))
        .collect();
    let mut orders = Vec::new();
    for m in 0..4 {
        let d1 = mus[0][m] - mus[1][m];
        let d2 = mus[1][m] - mus[2][m];
        assert!(d2.abs() < d1.abs(), "mode {m} not converging: {mus:?}");
        orders.push((d1.abs() / d2.abs()).log2());
    }
    // Measured orders 2.15, 2.36, 2.26, 2.34.
    assert!(orders[0] > 1.8, "fundamental order {:.2} < 1.8", orders[0]);
    let lo = orders.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(lo > 1.6, "low-mode orders {orders:?}");
    for (i, n) in [20i64, 40, 80].iter().enumerate() {
        against_lapack(*n, &mus[i], 3, 20.0);
    }
}

#[test]
fn the_low_modes_match_leissas_ffff_square_and_improve_with_refinement() {
    // Percent-level absolute anchor. Matched by SORTED eigenvalue (modes 4 and 5 are a degenerate
    // pair).
    let lambdas = free_plate_ffff_square_lambdas();
    let mut worst = Vec::new();
    for n in [32i64, 64] {
        let p = plate(n);
        let mu = elastic(&p, lambdas.len());
        let err = mu
            .iter()
            .zip(&lambdas)
            .map(|(&m, &l)| {
                let f_oracle = free_plate_freq_from_lambda(l, p.kappa, p.lx);
                (freq(m) - f_oracle).abs() / f_oracle
            })
            .fold(0.0f64, f64::max);
        worst.push(err);
        against_lapack(n, &mu, 3, 20.0);
    }
    // Measured 0.25% at N = 32 and 0.026% at N = 64.
    assert!(
        worst[1] < 0.006,
        "Leissa off {:.3}% at N = 64",
        worst[1] * 100.0
    );
    assert!(
        worst[1] < worst[0],
        "error not decreasing with refinement: {worst:?}"
    );
}

#[test]
fn the_fundamental_is_the_saddle_not_a_bulge() {
    // The best qualitative catch for a nu-dropped operator: corners alternate in sign, the centre
    // is a node.
    let n = 40usize;
    let p = plate(n as i64);
    let (_, vecs) = lowest(&p, 6);
    let phi = &vecs[3];
    let scale = phi.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let at = |j: usize, i: usize| phi[j * (n + 1) + i] / scale;
    let (c00, c0n, cn0, cnn) = (at(0, 0), at(0, n), at(n, 0), at(n, n));
    let centre = at(n / 2, n / 2);
    assert!(
        c00 * cnn > 0.5,
        "corners (0,0) and (N,N) should share sign: {c00:.2}, {cnn:.2}"
    );
    assert!(
        c0n * cn0 > 0.5,
        "corners (0,N) and (N,0) should share sign: {c0n:.2}, {cn0:.2}"
    );
    assert!(
        c00 * c0n < -0.5,
        "adjacent corners should be opposite (diagonal nodal lines)"
    );
    // Measured corners +-1.000 and a centre of 2.5e-12.
    assert!(
        centre.abs() < 0.1,
        "the centre should be a node, got {centre:.3}"
    );
}

#[test]
fn the_time_stepper_rings_at_the_discrete_fundamental() {
    let p = free(40, 1.0, 0.0, RHO);
    let a = p.lx;
    let mu = elastic(&p, 2);
    against_lapack(40, &mu, 3, 20.0);
    let f_disc = discrete_beam_eigenfrequency(mu[0], p.kappa, p.k, p.theta);
    let u0 = raised_cosine_2d(&p.x, &p.y, (0.3 * a, 0.62 * a), 0.3 * a, 1e-3).unwrap();
    let pickup = pickup_index_at(0.18 * a, 0.22 * a, &p);
    let mut pl = started(p, &u0);
    let res = run(&mut pl, 0.5, Some(pickup));
    let out = res.output.as_ref().unwrap();
    let found = measure_partials_near(out, res.fs, &[f_disc], Some(20.0))[0];
    let err = cents(found, f_disc).abs();
    // Measured 0.080 cents.
    assert!(
        err < 8.0,
        "FFT fundamental off {err:.2} cents ({found:.2} vs {f_disc:.2})"
    );
}

#[test]
fn the_generalized_map_on_the_supported_operators_is_model_5() {
    // The unified scheme's supported special case, K = h²B and W = h²I, through the generalized
    // eigen-map, reproduces model #5's frequencies: the free machinery is a correct generalization.
    let n = 32usize;
    let h = 1.0 / n as f64;
    let (b, _) = biharmonic_from_mask(&rectangle_mask(n, n), h);
    let k_ss = b.scaled(h * h);
    let w_ss = Csr::identity(b.nrows()).scaled(h * h);
    let mu = eigsh_shift_invert(&k_ss, Some(&w_ss), 0.0, 6)
        .expect("K_ss is SPD")
        .values;
    let modes: Vec<(i64, i64)> = (1..4).flat_map(|m| (1..4).map(move |q| (m, q))).collect();
    let mut lam = rectangular_discrete_eigenvalues(h, n as i64, n as i64, &modes);
    lam.sort_by(f64::total_cmp);
    let rel = mu
        .iter()
        .zip(&lam)
        .map(|(&m, &l)| {
            let f_ss = KAPPA * l / (2.0 * PI);
            (freq(m) - f_ss).abs() / f_ss
        })
        .fold(0.0f64, f64::max);
    // Measured 8.6e-13.
    assert!(
        rel < 1e-9,
        "the generalized map on SS operators != model #5 by {rel:.2e}"
    );
}
