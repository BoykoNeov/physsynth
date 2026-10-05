//! Native bars for the free-edge orthotropic plate — model #5of, four bending constants.
//!
//! Carried from `tests/test_free_plate_orthotropic.py` (retirement plan §24). A pinned rim sees the
//! coupling and torsional rigidities only through `H = D_1 + 2 D_xy`; a free rim sees them apart,
//! so the free branch takes four numbers and the two boundaries stop being the same material
//! question. There is no closed-form spectrum, so the four constants are validated by four
//! independent probes, one per constant:
//!
//! * `grain_torsion` — the centred saddle's Rayleigh quotient: blind to the other three, numerator
//!   `4 g_xy ab` in closed form, and a one-sided bound on the fundamental;
//! * `grain_x` / `grain_y` — an exact reduction to the shipped 1-D free beam on fields constant
//!   along one axis, but only at zero coupling, and the way it breaks is anticlastic curvature;
//! * `grain_coupling` — the `(x², y²)` bilinear probe, the only one of the four that responds to
//!   the coupling rigidity at all.
//!
//! The energy ledger is nearly blind here and says so: any symmetric `K` conserves exactly, so a
//! wrong coefficient is perfectly stable and perfectly conservative.
//!
//! **SciPy retires as the oracle here, with its numbers.** Where the Python file's truth came out of
//! `scipy.linalg.eigh` (LAPACK) or `scipy.sparse.linalg.eigsh` (ARPACK), the values it produced at
//! these exact fixtures were recorded on 2026-09-29, before the file was deleted, and are asserted
//! below as `SCIPY_*` constants. Without them the native eigensolvers would be the only referee of
//! their own answers. The inequalities the Python file asserted are carried as well; the recorded
//! values are an addition, not a substitute. Where the Python used ARPACK, the recorded referee is
//! LAPACK's DENSE solve of the same pencil instead, because recording ARPACK found it the least
//! accurate of the four solvers on these plates (see `SCIPY_SPLIT_LAMBDAS`).

use physsynth_analysis::modal::{free_plate_coupling_form, free_plate_twist_bound};
use physsynth_core::eig::generalized_eigen_diag;
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::engine::simulate;
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::ops::free_beam_stiffness;
use physsynth_core::ops2d::free_plate_stiffness;
use physsynth_core::plate::{
    grain_ratios_from_material, Boundary, ParamError, Params, Plate, PlateSpec, VkParams, VkSpec,
};
use physsynth_core::sparse::Csr;

/// `tests/helpers.py`'s `KAPPA_PLATE_DEFAULT` and `RHO_AREAL_DEFAULT`.
const KAPPA: f64 = 20.0;
const RHO: f64 = 0.005;
/// Tier 1: the same acceptance bar as every other resonator.
const DRIFT_TOL: f64 = 1e-10;

// -- fixtures ---------------------------------------------------------------------------------

/// Spruce, as the Python helpers' `SPRUCE` material, split into the free branch's four constants.
fn spruce() -> [f64; 4] {
    let s = grain_ratios_from_material(11.0e9, 0.8e9, 0.37, 0.7e9, 3.0e-3, 420.0)
        .expect("spruce is admissible");
    [s.grain_x, s.grain_y, s.grain_coupling, s.grain_torsion]
}

/// The isotropic `nu = 0.3` split — `make_orthotropic_free_plate`'s defaults.
const ISOTROPIC: [f64; 4] = [1.0, 1.0, 0.3, 0.35];

/// `make_orthotropic_free_plate`: a free square of side 1 at plate-Courant number `mu`, grained.
fn free_spec(n: i64, mu: f64, g: [f64; 4]) -> PlateSpec {
    let a = 1.0;
    let h = a / n as f64;
    PlateSpec {
        lx: a,
        ly: a,
        kappa: KAPPA,
        rho: RHO,
        fs: KAPPA / (mu * h * h),
        n,
        boundary: Some(Boundary::Free),
        grain_x: g[0],
        grain_y: g[1],
        grain_coupling: Some(g[2]),
        grain_torsion: Some(g[3]),
        ..PlateSpec::default()
    }
}

fn free_params(n: i64, mu: f64, g: [f64; 4]) -> Params {
    Params::new(&free_spec(n, mu, g)).expect("an admissible grained free plate")
}

/// The rectangle's four-constant stiffness, with every constant given.
fn stiffness(nx: usize, ny: usize, h: f64, g: [f64; 4]) -> (Csr, Csr) {
    let (k, w, _) = free_plate_stiffness(nx, ny, h, 0.3, g[0], g[1], Some(g[2]), Some(g[3]));
    (k, w)
}

fn diag(w: &Csr) -> Vec<f64> {
    (0..w.nrows()).map(|i| w.get(i, i)).collect()
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Dense `K φ = mu W φ`, ascending, `W`-orthonormal vectors (column `j` of a row-major matrix).
fn pencil(k: &Csr, w: &Csr) -> (Vec<f64>, Vec<f64>) {
    let n = k.nrows();
    generalized_eigen_diag(&dense(k), &diag(w), n).expect("a symmetric pencil with positive mass")
}

fn column(vecs: &[f64], n: usize, j: usize) -> Vec<f64> {
    (0..n).map(|i| vecs[i * n + j]).collect()
}

/// A field over the `(ny+1) x (nx+1)` grid, row-major, with coordinates measured from the plate's
/// CENTROID — what makes `xy` orthogonal to `{1, x, y}`.
fn centred(nx: usize, ny: usize, h: f64, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut v = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            v.push(f(
                (i as f64 - 0.5 * nx as f64) * h,
                (j as f64 - 0.5 * ny as f64) * h,
            ));
        }
    }
    v
}

/// The retired `tests/helpers.py::free_plate_low_eigenfrequencies` as the eigenvalues
/// `mu = ω²/κ²`: the `n_modes` lowest ELASTIC ones, the three rigid-body modes discarded, by
/// shift-invert at a small negative shift scaled by the caller's guess at the fundamental's
/// frequency parameter.
fn low_elastic_mu(p: &Params, n_modes: usize, lam1_hint: f64) -> Vec<f64> {
    let a = p.lx;
    let mu1_est = (lam1_hint / (a * a)).powi(2);
    let got = eigsh_shift_invert(&p.stiffness, p.mass.as_ref(), -1e-3 * mu1_est, n_modes + 3)
        .expect("the shifted pencil factors");
    let mut v = got.values;
    v.sort_by(f64::total_cmp);
    v[3..].to_vec()
}

fn freq_of(mu: f64, kappa: f64) -> f64 {
    kappa * mu.max(0.0).sqrt() / (2.0 * std::f64::consts::PI)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

// -- assembly: the resonator builds what its parameters say ------------------------------------

#[test]
fn the_default_free_plate_is_the_isotropic_operator_and_its_split_is_nus() {
    // Carries `test_resonator_default_free_plate_is_bit_identical_to_the_helper`.
    let n = 16;
    let h = 1.0 / n as f64;
    let spec = PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho: RHO,
        fs: KAPPA / (2.0 * h * h),
        n,
        boundary: Some(Boundary::Free),
        nu: Some(0.3),
        ..PlateSpec::default()
    };
    let p = Params::new(&spec).unwrap();
    let (k, w, _) = free_plate_stiffness(p.n, p.ny, p.h, 0.3, 1.0, 1.0, None, None);
    assert_eq!(k.indices(), p.stiffness.indices());
    assert_eq!(k.data(), p.stiffness.data(), "default free plate moved");
    assert_eq!(diag(&w), p.w);
    assert_eq!(
        (p.grain_coupling, p.grain_torsion),
        (0.3, 0.35),
        "the nu-derived split"
    );
}

#[test]
fn a_grained_plate_builds_the_operator_its_own_parameters_imply() {
    // The SEAM: every other bar probes `free_plate_stiffness` directly, so a mis-wired field in the
    // constructor (`grain_y` where `grain_coupling` belongs) would survive all of them. Non-square
    // with all four constants distinct, because on a square an x/y transposition is invisible.
    let g = [2.3, 0.41, -0.37, 0.62];
    let spec = PlateSpec {
        lx: 0.62,
        ly: 0.42,
        kappa: KAPPA,
        rho: RHO,
        fs: 40_000.0,
        n: 13,
        boundary: Some(Boundary::Free),
        grain_x: g[0],
        grain_y: g[1],
        grain_coupling: Some(g[2]),
        grain_torsion: Some(g[3]),
        ..PlateSpec::default()
    };
    let p = Params::new(&spec).unwrap();
    assert_ne!(p.n, p.ny, "the fixture must be non-square");
    let (k, w, _) = free_plate_stiffness(p.n, p.ny, p.h, p.nu, g[0], g[1], Some(g[2]), Some(g[3]));
    assert_eq!(k.indices(), p.stiffness.indices());
    assert_eq!(
        k.data(),
        p.stiffness.data(),
        "resonator K != its own parameters"
    );
    assert_eq!(diag(&w), p.w);
    assert_eq!(p.grain_cross, -0.37 + 2.0 * 0.62);
    assert_eq!(p.nu, -0.37 / 2.3, "the implied nu_yx = g_1/g_x");
}

/// Explicit per-node assembly of the four-constant form — no Kronecker products, no sparse type.
/// The independent reference for the ordering; returns a dense row-major `nn x nn`.
fn direct_assembly(nx: usize, ny: usize, h: f64, g: [f64; 4]) -> Vec<f64> {
    let nn = (nx + 1) * (ny + 1);
    let idx = |i: usize, j: usize| j * (nx + 1) + i;
    let inv_h2 = 1.0 / (h * h);
    // Rows of the two curvatures (one row per node, zero rows on the edges they cannot see).
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
    let gram = |a: &[f64], w: &[f64], b: &[f64], rows: usize| -> Vec<f64> {
        let mut out = vec![0.0; nn * nn];
        for r in 0..rows {
            for p in 0..nn {
                let ap = a[r * nn + p] * w[r];
                if ap == 0.0 {
                    continue;
                }
                for q in 0..nn {
                    out[p * nn + q] += ap * b[r * nn + q];
                }
            }
        }
        out
    };
    let xx = gram(&dxx, &wa, &dxx, nn);
    let yy = gram(&dyy, &wa, &dyy, nn);
    let cross = gram(&dxx, &wa, &dyy, nn);
    let mut dxy = vec![0.0; nx * ny * nn];
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            dxy[c * nn + idx(i, j)] += inv_h2;
            dxy[c * nn + idx(i + 1, j)] += -inv_h2;
            dxy[c * nn + idx(i, j + 1)] += -inv_h2;
            dxy[c * nn + idx(i + 1, j + 1)] += inv_h2;
        }
    }
    let twist = gram(&dxy, &vec![1.0; nx * ny], &dxy, nx * ny);
    let mut k = vec![0.0; nn * nn];
    for p in 0..nn {
        for q in 0..nn {
            k[p * nn + q] = g[0] * xx[p * nn + q]
                + g[1] * yy[p * nn + q]
                + g[2] * (cross[p * nn + q] + cross[q * nn + p])
                + 4.0 * g[3] * (h * h) * twist[p * nn + q];
        }
    }
    k
}

#[test]
fn the_assembly_matches_a_direct_per_node_build_with_four_distinct_constants() {
    // Non-square grids AND g_x != g_y != g_1 != g_xy, so an axis swap cannot hide.
    let g = [2.3, 0.41, -0.37, 0.62];
    for (nx, ny) in [(5usize, 3usize), (4, 5), (6, 4)] {
        let h = 0.1;
        let (k, _, index_map) =
            free_plate_stiffness(nx, ny, h, 0.3, g[0], g[1], Some(g[2]), Some(g[3]));
        let reference = direct_assembly(nx, ny, h, g);
        let scale = max_abs(&reference);
        let got = dense(&k);
        let diff = got
            .iter()
            .zip(&reference)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            diff < 1e-12 * scale,
            "sparse != direct at {nx}x{ny}: {diff:.2e} (scale {scale:.2e})"
        );
        let want: Vec<i64> = (0..((nx + 1) * (ny + 1)) as i64).collect();
        assert_eq!(index_map, want);
    }
}

#[test]
fn the_grained_operator_stays_symmetric() {
    // Symmetry is what makes the energy an exact algebraic identity; it must survive orthotropy.
    let (k, _) = stiffness(12, 10, 0.1, spruce());
    let n = k.nrows();
    let d = dense(&k);
    let mut asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            asym = asym.max((d[i * n + j] - d[j * n + i]).abs());
        }
    }
    assert!(
        asym < 1e-12 * max_abs(&d),
        "K not symmetric with a grain: {asym:.3e}"
    );
}

/// One code path, stated rather than carried. `test_isotropic_split_is_bit_identical_on_every_grid`
/// asserted that `(1, 1, nu, (1-nu)/2)` passed explicitly reproduces the isotropic free plate byte
/// for byte. Natively the isotropic call IS that call: `free_plate_stiffness_from_mask` fills a
/// missing half of the split with `unwrap_or(nu)` and `unwrap_or(0.5 * (1.0 - nu))`, the same
/// expressions a caller would pass, into the same assembly. A carried equality would compare a
/// computation with itself (finding #78). This test pins the premise of that verdict instead: the
/// isotropic default and the explicit split agree, and the seam test above is what can fail.
#[test]
fn the_isotropic_split_is_the_isotropic_plate_by_construction() {
    // The Python file's seven-grid survey, which contains grids where the SUPPORTED branch's two
    // assemblies differ in the last bit — asserting only on a friendly grid would prove nothing.
    let grids = [
        (12usize, 12usize, 1.0 / 12.0),
        (24, 24, 1.0 / 24.0),
        (20, 14, 0.05),
        (17, 17, 1.0 / 17.0),
        (13, 9, 0.62 / 13.0),
        (16, 16, 0.7 / 16.0),
        (11, 7, 0.31 / 11.0),
    ];
    for (nx, ny, h) in grids {
        for nu in [0.3, 0.0, 0.49, -0.5] {
            let (a, wa, _) = free_plate_stiffness(nx, ny, h, nu, 1.0, 1.0, None, None);
            let (b, wb, _) =
                free_plate_stiffness(nx, ny, h, nu, 1.0, 1.0, Some(nu), Some(0.5 * (1.0 - nu)));
            assert_eq!(a.indices(), b.indices());
            assert_eq!(a.data(), b.data(), "{nx}x{ny} nu={nu}");
            assert_eq!(wa.data(), wb.data());
        }
    }
}

// -- the nullspace: three rigid-body modes, and a fourth if the torsion is switched off ----------

#[test]
fn the_rigid_body_nullspace_survives_the_grain() {
    let (nx, ny, h) = (14usize, 12usize, 0.1);
    let (k, _) = stiffness(nx, ny, h, spruce());
    let k_fro = norm(k.data());
    // Uncentred coordinates, as the Python file used: the claim is about {1, x, y}, not its basis.
    let field = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
        let mut v = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                v.push(f(i as f64 * h, j as f64 * h));
            }
        }
        v
    };
    let rel_of = |v: &[f64]| norm(&k.matvec(v)) / (k_fro * norm(v));
    let rigid = [
        ("1", rel_of(&field(&|_, _| 1.0))),
        ("x", rel_of(&field(&|x, _| x))),
        ("y", rel_of(&field(&|_, y| y))),
    ];
    for (name, r) in rigid {
        assert!(
            r < 1e-12,
            "K@{name} not in the nullspace with a grain: {r:.2e}"
        );
    }
    let saddle = rel_of(&field(&|x, y| x * y));
    let worst = rigid.iter().fold(0.0f64, |m, (_, r)| m.max(*r));
    assert!(saddle > 1e6 * worst, "nullspace contrast too small");
}

/// LAPACK's `eigh` on the zero-torsion pencil below: the largest eigenvalue, and the fifth lowest
/// as a fraction of it (the first four are the rigid trio and the saddle, at ~1e-16).
const SCIPY_ZERO_TORSION_SCALE: f64 = 4130580.2517834003;
const SCIPY_ZERO_TORSION_FIFTH: f64 = 0.0016883885424919722;

#[test]
fn zero_torsion_puts_the_saddle_into_the_nullspace() {
    // `grain_torsion = 0` is DEGENERATE, not stiff: the operator helper permits it (so it can be
    // built here) and the plate refuses it. The torsional rigidity is the only term that gives the
    // saddle `xy` any energy at all.
    let (k, w) = stiffness(10, 8, 0.05, [1.0, 0.5, 0.1, 0.0]);
    let (vals, _) = pencil(&k, &w);
    let scale = max_abs(&vals);
    let n_zero = vals.iter().filter(|v| v.abs() < 1e-9 * scale).count();
    assert_eq!(
        n_zero,
        4,
        "expected the 3 rigid modes + the saddle: {:?}",
        &vals[..6]
    );
    assert!(rel(scale, SCIPY_ZERO_TORSION_SCALE) < 1e-12, "{scale:?}");
    assert!(
        rel(vals[4] / scale, SCIPY_ZERO_TORSION_FIFTH) < 1e-9,
        "{:?}",
        vals[4] / scale
    );

    let e = Params::new(&free_spec(8, 2.0, [1.0, 1.0, 0.3, 0.0])).unwrap_err();
    assert!(matches!(e, ParamError::NonPositiveTorsion(_)));
    assert!(
        e.to_string().contains("joins the rigid-body nullspace"),
        "{e}"
    );
}

// -- detector 1 (grain_torsion): the saddle's Rayleigh quotient ---------------------------------

/// `((xy)ᵀK(xy), (xy)ᵀW(xy))` for the centred saddle.
fn twist_quotient(nx: usize, ny: usize, h: f64, g: [f64; 4]) -> (f64, f64) {
    let (k, w) = stiffness(nx, ny, h, g);
    let xy = centred(nx, ny, h, |x, y| x * y);
    let wxy: Vec<f64> = diag(&w).iter().zip(&xy).map(|(a, b)| a * b).collect();
    (dot(&xy, &k.matvec(&xy)), dot(&xy, &wxy))
}

#[test]
fn the_twist_quotient_is_blind_to_the_other_three_constants() {
    // The collocated second differences annihilate the saddle (it is linear along every grid line),
    // so g_x, g_y and the coupling cannot contribute in exact arithmetic. In doubles they cancel
    // rather than vanish — the Python measured 1.6e-13, and the bar is 1e-11.
    let (nx, ny, h) = (16, 12, 0.05);
    let q: Vec<f64> = [
        (1.0, 1.0, 0.3),
        (1.0, 0.073, 0.02),
        (3.1, 0.5, -0.4),
        (1.0, 1.0, 0.0),
    ]
    .iter()
    .map(|&(gx, gy, g1)| {
        let (num, den) = twist_quotient(nx, ny, h, [gx, gy, g1, 0.37]);
        num / den
    })
    .collect();
    let spread = (q.iter().cloned().fold(f64::MIN, f64::max)
        - q.iter().cloned().fold(f64::MAX, f64::min))
        / q[0].abs();
    assert!(
        spread < 1e-11,
        "R(xy) is not blind to g_x/g_y/g_1: {spread:.2e}"
    );
}

#[test]
fn the_twist_numerator_is_the_closed_form_and_exact_without_cancellation() {
    // (xy)ᵀK(xy) = 4 g_xy ab on any grid: Dxy(xy) = 1 on every cell. Exact in exact arithmetic,
    // so the bar is set at the measured roundoff (2.5e-14 clean, ~1e-13 with bending on).
    for (nx, ny, h) in [
        (16usize, 12usize, 0.05),
        (13, 9, 0.62 / 13.0),
        (24, 24, 1.0 / 24.0),
    ] {
        let (a, b) = (nx as f64 * h, ny as f64 * h);
        let g_xy = 0.37;
        let exact = 4.0 * g_xy * a * b;
        let (clean, _) = twist_quotient(nx, ny, h, [0.0, 0.0, 0.0, g_xy]);
        let (dirty, _) = twist_quotient(nx, ny, h, [1.0, 1.0, 0.3, g_xy]);
        assert!((clean - exact).abs() < 1e-13 * exact, "{clean} != {exact}");
        assert!((dirty - exact).abs() < 1e-11 * exact, "{dirty} vs {exact}");
        assert!(
            (clean - exact).abs() < (dirty - exact).abs(),
            "with the bending terms off nothing cancels, so clean must be closer: {clean:?} vs {dirty:?}"
        );
    }
}

#[test]
fn the_twist_quotient_converges_to_the_continuum_bound_at_h2() {
    // R(xy) -> 576 D_xy/(rho_s a²b²), second order, and the error lives in the MASS: the numerator
    // is exact, so the O(h²) is the trapezoidal (xy)ᵀW(xy) -> a³b³/144. Measured 1.95, 1.99, 2.00.
    let g_xy = 0.37;
    let errs: Vec<f64> = [8usize, 16, 32, 64]
        .iter()
        .map(|&n| {
            let (num, den) = twist_quotient(n, n, 1.0 / n as f64, [1.0, 1.0, 0.3, g_xy]);
            (num / den - 576.0 * g_xy).abs()
        })
        .collect();
    for w in errs.windows(2) {
        assert!(w[1] < w[0], "not converging: {errs:?}");
        let order = (w[0] / w[1]).log2();
        assert!(
            order > 1.8,
            "twist-quotient order {order} (want ~2): {errs:?}"
        );
    }
}

/// LAPACK's dense `eigh` fundamentals (Hz) at the two fixtures below. ARPACK, through
/// `free_plate_low_eigenfrequencies`, gave `42.81308434144431` and `18.111843019769445` — within
/// 1e-11 of these, so at this shift the two SciPy solvers agree.
const SCIPY_F1_ISOTROPIC: f64 = 42.81308434106075;
const SCIPY_F1_SPRUCE: f64 = 18.111843020350392;

#[test]
fn the_fundamental_is_below_the_twist_bound_and_the_bound_is_informative() {
    // Rayleigh: the first elastic frequency sits below 24 sqrt(D_xy/rho_s)/(2 pi ab). A bound is
    // only worth asserting if it is tight enough to catch something — ~5% on both arms. It is
    // ONE-SIDED, so a uniformly too-soft operator passes; the beam reduction is the two-sided guard.
    for (tag, g, hint, scipy) in [
        ("isotropic", ISOTROPIC, 13.5, SCIPY_F1_ISOTROPIC),
        ("spruce", spruce(), 6.0, SCIPY_F1_SPRUCE),
    ] {
        let p = free_params(32, 0.5, g);
        let bound = free_plate_twist_bound(p.kappa, p.lx, p.ly, p.grain_torsion).unwrap();
        let f1 = freq_of(low_elastic_mu(&p, 1, hint)[0], p.kappa);
        let margin = (bound - f1) / bound;
        assert!(
            f1 < bound,
            "{tag}: fundamental {f1:.3} Hz above the bound {bound:.3}"
        );
        assert!(
            margin < 0.15,
            "{tag}: bound is loose ({margin:.3}) — operator too soft?"
        );
        assert!(rel(f1, scipy) < 1e-9, "{tag}: {f1:?} vs LAPACK {scipy:?}");
    }
}

// -- detector 2 (grain_x / grain_y): the exact reduction to the shipped 1-D free beam ----------

/// LAPACK's `eigh` on the free beam's pencil at `h = 0.05`: the three lowest ELASTIC eigenvalues
/// for `N = 20`, `14` and `16` (the two rigid modes, at ~1e-10, are not recorded).
const SCIPY_BEAM: [(usize, [f64; 3]); 3] = [
    (
        20,
        [490.4243569588452, 3636.9393111973036, 13542.42247714206],
    ),
    (
        14,
        [2000.1683720140695, 14474.901553997235, 52237.46434186766],
    ),
    (16, [1183.777023142654, 8662.666674632012, 31709.2405546054]),
];

/// The free beam's dense pencil, checked against LAPACK's recorded values on the way out.
fn beam_pencil(n: usize, h: f64) -> (Vec<f64>, Vec<f64>) {
    let (s, m) = free_beam_stiffness(n, h);
    let (vals, vecs) = pencil(&s, &m);
    if h == 0.05 {
        let (_, want) = SCIPY_BEAM
            .iter()
            .find(|(nn, _)| *nn == n)
            .expect("a recorded beam");
        for (j, w) in want.iter().enumerate() {
            assert!(
                rel(vals[j + 2], *w) < 1e-10,
                "beam N={n} mode {j}: {} vs {w}",
                vals[j + 2]
            );
        }
    }
    (vals, vecs)
}

#[test]
fn a_field_constant_along_one_axis_has_the_free_beams_spectrum_exactly() {
    // At ZERO coupling, w = 1 ⊗ v (or v ⊗ 1) is an exact eigenvector: C2 across it and the twist
    // both vanish, and the trapezoidal weights of the other axis factor out of both sides, so the
    // generalized problem collapses to the 1-D beam's scaled by that axis's stiffness ratio. The
    // batch's only TWO-SIDED external anchor, borrowing the free beam's own oracle.
    let (nx, ny, h) = (20usize, 14usize, 0.05);
    for axis in ["x", "y"] {
        let (g_along, g_across) = if axis == "x" { (2.3, 0.6) } else { (0.6, 2.3) };
        let (k, w) = stiffness(nx, ny, h, [g_along, g_across, 0.0, 0.25]);
        let wd = diag(&w);
        let n_line = if axis == "x" { nx } else { ny };
        let (vals_b, vecs_b) = beam_pencil(n_line, h);
        for j in [2, 3, 4] {
            let v = column(&vecs_b, n_line + 1, j);
            let field: Vec<f64> = (0..=ny)
                .flat_map(|jj| (0..=nx).map(move |ii| (ii, jj)))
                .map(|(ii, jj)| if axis == "x" { v[ii] } else { v[jj] })
                .collect();
            let mu_pred = if axis == "x" { g_along } else { g_across } * vals_b[j];
            let target: Vec<f64> = wd
                .iter()
                .zip(&field)
                .map(|(a, b)| mu_pred * a * b)
                .collect();
            let resid: Vec<f64> = k
                .matvec(&field)
                .iter()
                .zip(&target)
                .map(|(a, b)| a - b)
                .collect();
            let r = norm(&resid) / norm(&target);
            assert!(
                r < 1e-10,
                "{axis}-independent beam mode {}: residual {r:.2e}",
                j - 1
            );
        }
    }
}

#[test]
fn coupling_breaks_the_beam_reduction_without_changing_its_energy() {
    // With a coupling rigidity the beam-like field stops being an eigenvector while its Rayleigh
    // quotient stays EXACTLY the beam's: `cross` annihilates it but `crossᵀ` survives on the free y
    // edges, where a cylindrically bent strip does not satisfy M_y = 0. A different SHAPE at the
    // same energy — anticlastic curvature, which is physics. Do not "fix" the residual.
    let (nx, ny, h) = (20usize, 14usize, 0.05);
    let (vals_b, vecs_b) = beam_pencil(nx, h);
    let v = column(&vecs_b, nx + 1, 2);
    let field: Vec<f64> = (0..=ny).flat_map(|_| v.iter().copied()).collect();
    let mu_pred = 2.3 * vals_b[2];
    let mut residuals = Vec::new();
    for g_1 in [0.0, 0.1, 0.3] {
        let (k, w) = stiffness(nx, ny, h, [2.3, 0.6, g_1, 0.25]);
        let ww: Vec<f64> = diag(&w).iter().zip(&field).map(|(a, b)| a * b).collect();
        let kw = k.matvec(&field);
        let resid: Vec<f64> = kw.iter().zip(&ww).map(|(a, b)| a - mu_pred * b).collect();
        residuals.push(norm(&resid) / (mu_pred * norm(&ww)));
        let rayleigh = dot(&field, &kw) / dot(&field, &ww);
        assert!(
            (rayleigh / mu_pred - 1.0).abs() < 1e-12,
            "the coupling changed the beam-like field's ENERGY at g_1={g_1}: {}",
            rayleigh / mu_pred
        );
    }
    assert!(
        residuals[0] < 1e-10 && 1e-10 < residuals[1] && residuals[1] < residuals[2],
        "the residual must be zero at g_1=0 and grow with coupling: {residuals:?}"
    );
}

// -- detector 3 (grain_coupling): the (x², y²) probe, the only one that sees D_1 --------------

#[test]
fn the_coupling_probe_hits_its_exact_discrete_value() {
    // P(x², y²) = 4 g_1 h² (Nx-1)(Ny-1) exactly: every other term dies on this pair, and the
    // collocated second differences return 2 at each interior node and 0 on the free edges. Built
    // with the other three constants at zero, so nothing cancels and the closed form is hit at
    // machine precision. The continuum 4 D_1 ab is short by exactly one boundary strip.
    let g_1 = 0.153;
    let grids = [
        (8usize, 8usize, 1.0 / 8.0),
        (32, 32, 1.0 / 32.0),
        (13, 9, 0.62 / 13.0),
        (20, 14, 0.05),
    ];
    let mut cont_errs = Vec::new();
    for (nx, ny, h) in grids {
        let (k, _) = stiffness(nx, ny, h, [0.0, 0.0, g_1, 0.0]);
        let x2 = centred(nx, ny, h, |x, _| x * x);
        let y2 = centred(nx, ny, h, |_, y| y * y);
        let val = dot(&x2, &k.matvec(&y2));
        let exact = free_plate_coupling_form(g_1, h, nx as i64, ny as i64).unwrap();
        assert!(
            (val - exact).abs() < 1e-13 * exact.abs(),
            "{nx}x{ny}: {val:?} != {exact:?}"
        );
        let cont = 4.0 * g_1 * (nx as f64 * h) * (ny as f64 * h);
        cont_errs.push((exact - cont).abs() / cont);
    }
    for ((nx, ny, _), err) in grids.iter().zip(&cont_errs).take(2) {
        let predicted = 1.0 - (1.0 - 1.0 / *nx as f64) * (1.0 - 1.0 / *ny as f64);
        assert!(
            (err - predicted).abs() < 1e-12,
            "continuum gap {err} != {predicted}"
        );
    }
}

#[test]
fn only_the_coupling_probe_sees_the_coupling_rigidity() {
    // Change g_1 alone: the twist and beam probes do not move, this one moves proportionally. It is
    // why the least glamorous detector is load-bearing.
    let (nx, ny, h) = (16usize, 12usize, 0.05);
    let twist: Vec<f64> = [0.0, 0.2]
        .iter()
        .map(|&g_1| twist_quotient(nx, ny, h, [1.0, 0.5, g_1, 0.3]).0)
        .collect();
    assert!(
        (twist[1] - twist[0]).abs() < 1e-11 * twist[0].abs(),
        "the twist probe saw g_1"
    );
    let (_, vecs_b) = beam_pencil(nx, h);
    let v = column(&vecs_b, nx + 1, 2);
    let field: Vec<f64> = (0..=ny).flat_map(|_| v.iter().copied()).collect();
    let energies: Vec<f64> = [0.0, 0.2]
        .iter()
        .map(|&g_1| {
            let (k, w) = stiffness(nx, ny, h, [1.0, 0.5, g_1, 0.3]);
            let ww: Vec<f64> = diag(&w).iter().zip(&field).map(|(a, b)| a * b).collect();
            dot(&field, &k.matvec(&field)) / dot(&field, &ww)
        })
        .collect();
    assert!(
        (energies[1] - energies[0]).abs() < 1e-12 * energies[0],
        "the beam probe saw g_1"
    );
    let probes: Vec<f64> = [0.1, 0.2]
        .iter()
        .map(|&g_1| free_plate_coupling_form(g_1, h, nx as i64, ny as i64).unwrap())
        .collect();
    assert!(
        (probes[1] / probes[0] - 2.0).abs() < 1e-12,
        "the probe is not linear in g_1"
    );
}

// -- the construction-time guard: a DIFFERENT set from the supported branch's -------------------

#[test]
fn free_grain_admissibility_is_rejected_at_construction() {
    // |g_1| >= sqrt(1.0 * 0.5) = 0.7071... is refused; so is a torsion that is not positive.
    for g_1 in [0.71, -0.71, 1.5] {
        let e = Params::new(&free_spec(8, 2.0, [1.0, 0.5, g_1, 0.3])).unwrap_err();
        assert!(
            matches!(e, ParamError::IndefiniteCoupling { .. }),
            "g_1={g_1}: {e}"
        );
        assert!(e.to_string().contains("grain_coupling"), "{e}");
    }
    for g_t in [0.0, -0.1] {
        let e = Params::new(&free_spec(8, 2.0, [1.0, 0.5, 0.1, g_t])).unwrap_err();
        assert!(
            matches!(e, ParamError::NonPositiveTorsion(_)),
            "g_t={g_t}: {e}"
        );
        assert!(e.to_string().contains("grain_torsion"), "{e}");
    }
    // ... and just inside it builds.
    let p = free_params(8, 2.0, [1.0, 0.5, 0.70, 0.3]);
    assert_eq!(p.grain_coupling, 0.70);
}

#[test]
fn the_two_boundaries_admissible_sets_differ() {
    // g_1 = 0.9, g_xy = 0.05, g_y = 0.5 gives H = 1.0, which the supported guard
    // (H > -sqrt(g_x g_y)) accepts — while the free branch rejects it, because
    // |g_1| > sqrt(g_x g_y). The guards are not nested either way: conditions on different objects.
    let common = |boundary| PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho: RHO,
        fs: 40_000.0,
        n: 8,
        boundary: Some(boundary),
        grain_y: 0.5,
        grain_coupling: Some(0.9),
        grain_torsion: Some(0.05),
        ..PlateSpec::default()
    };
    Params::new(&common(Boundary::Supported)).expect("the supported branch accepts H = 1.0");
    let e = Params::new(&common(Boundary::Free)).unwrap_err();
    assert!(matches!(e, ParamError::IndefiniteCoupling { .. }), "{e}");
}

/// LAPACK's `eigh` on the six pencils below, at `(Nx, Ny, g_y, g_xy, factor)`: the largest
/// eigenvalue, and the fourth lowest (the first elastic mode) as a fraction of it.
const SCIPY_GUARD: [(usize, usize, f64, f64, f64, f64, f64); 6] = [
    (
        10,
        8,
        0.5,
        0.2,
        0.98,
        8954877.863116127,
        0.00013448980076636273,
    ),
    (
        10,
        8,
        0.5,
        0.2,
        1.02,
        9090814.605794422,
        9.581698764209263e-05,
    ),
    (
        12,
        12,
        0.5,
        0.02,
        0.98,
        7354118.240790997,
        1.1514800379897023e-05,
    ),
    (
        12,
        12,
        0.5,
        0.02,
        1.02,
        7494212.513720713,
        1.129098496132885e-05,
    ),
    (
        10,
        8,
        1.0,
        0.35,
        0.98,
        13014560.80541308,
        0.00011871338667252027,
    ),
    (
        10,
        8,
        1.0,
        0.35,
        1.02,
        13206599.035528863,
        8.523609531991008e-05,
    ),
];

#[test]
fn the_guard_is_conservative_and_the_claim_is_one_sided() {
    // Inside the guard: semi-definite with exactly 3 zero modes. OUTSIDE it, on a coarse grid, the
    // discrete operator is STILL semi-definite — the pointwise bound is the sharp CONTINUUM
    // condition, and the discrete margin (4–20%, shrinking with refinement) is measured, not claimed
    // away. The arm past the bound is a claim about THESE resolutions: the discrete threshold was
    // bisected at 1.198 on 7x5 but 1.042 on 24x24, so refining would legitimately break it.
    for &(nx, ny, g_y, g_t, factor, scipy_scale, scipy_fourth) in &SCIPY_GUARD {
        let edge = g_y.sqrt();
        let (k, w) = stiffness(nx, ny, 0.05, [1.0, g_y, factor * edge, g_t]);
        let (vals, _) = pencil(&k, &w);
        let scale = max_abs(&vals);
        assert!(
            rel(scale, scipy_scale) < 1e-12,
            "{nx}x{ny} x{factor}: {scale:?}"
        );
        assert!(
            rel(vals[3] / scale, scipy_fourth) < 1e-9,
            "{:?}",
            vals[3] / scale
        );
        if factor < 1.0 {
            assert!(
                vals[0] > -1e-12 * scale,
                "indefinite INSIDE the guard: {:.2e}",
                vals[0] / scale
            );
            let n_zero = vals.iter().filter(|v| v.abs() < 1e-9 * scale).count();
            assert_eq!(n_zero, 3, "not 3 zero modes: {:?}", &vals[..5]);
        } else {
            assert!(
                vals[0] > -1e-12 * scale,
                "the discrete threshold moved BELOW the pointwise bound: {:.2e}",
                vals[0] / scale
            );
        }
    }
}

#[test]
fn the_split_api_refuses_every_ambiguous_call() {
    let base = PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho: RHO,
        fs: 40_000.0,
        n: 8,
        boundary: Some(Boundary::Free),
        ..PlateSpec::default()
    };
    let refused = |s: PlateSpec, words: &str| {
        let e = Params::new(&s).unwrap_err();
        assert!(e.to_string().contains(words), "want {words:?}: {e}");
        e
    };
    let half = refused(
        PlateSpec {
            grain_coupling: Some(0.3),
            ..base.clone()
        },
        "together",
    );
    assert_eq!(half, ParamError::HalfSplit);
    let half = refused(
        PlateSpec {
            grain_torsion: Some(0.35),
            ..base.clone()
        },
        "together",
    );
    assert_eq!(half, ParamError::HalfSplit);
    let both = refused(
        PlateSpec {
            nu: Some(0.3),
            grain_coupling: Some(0.3),
            grain_torsion: Some(0.35),
            ..base.clone()
        },
        "either nu or",
    );
    assert_eq!(both, ParamError::NuWithSplit);
    let contra = refused(
        PlateSpec {
            grain_cross: Some(0.9),
            grain_coupling: Some(0.3),
            grain_torsion: Some(0.35),
            ..base.clone()
        },
        "contradicts the split",
    );
    assert!(matches!(contra, ParamError::SplitContradiction { .. }));
    let unsplit = refused(
        PlateSpec {
            grain_y: 0.5,
            ..base.clone()
        },
        "separately",
    );
    assert_eq!(unsplit, ParamError::FreeNeedsSplit);
    // A consistent grain_cross alongside the split is accepted (1.0 == 0.3 + 2*0.35).
    let p = Params::new(&PlateSpec {
        grain_cross: Some(1.0),
        grain_coupling: Some(0.3),
        grain_torsion: Some(0.35),
        ..base
    })
    .unwrap();
    assert_eq!(p.nu, 0.3, "the implied nu_yx = g_1/g_x is exposed as nu");
}

// -- the physics: what a free edge can measure that a pinned one cannot -------------------------

#[test]
fn the_supported_plate_is_blind_to_the_split() {
    // Four splits of the same H: the supported operator is BIT-IDENTICAL, on a non-square grid
    // chosen because it distinguishes assemblies. The exact null control for the next bar.
    let mut base: Option<Csr> = None;
    for g_1 in [-0.1, 0.0, 0.05, 0.1] {
        let p = Params::new(&PlateSpec {
            lx: 0.62,
            ly: 0.42,
            kappa: KAPPA,
            rho: RHO,
            fs: 40_000.0,
            n: 13,
            boundary: Some(Boundary::Supported),
            grain_y: 0.073,
            grain_coupling: Some(g_1),
            grain_torsion: Some(0.5 * (0.153 - g_1)),
            ..PlateSpec::default()
        })
        .unwrap();
        match &base {
            None => base = Some(p.stiffness),
            Some(b) => {
                assert_eq!(b.indices(), p.stiffness.indices());
                assert_eq!(
                    b.data(),
                    p.stiffness.data(),
                    "supported B moved at g_1={g_1}"
                );
            }
        }
    }
}

/// LAPACK's dense `eigh` frequency parameters `lambda = a² sqrt(mu_4)` for the five splits below.
///
/// NOT the ARPACK values the Python test computed, and that is a finding: at its shift of `-1e-4`,
/// three orders closer to the rigid trio than `free_plate_low_eigenfrequencies` goes, ARPACK
/// returned `5.651951780872772, 6.002028112660742, 5.1679474052340035, 3.7629602597978966,
/// 0.9229586653986105` — up to 3.6e-7 off in lambda (7.2e-7 in mu) at `g_1 = -0.1`. The native
/// shift-invert at the same shift lands within 1e-10 of LAPACK, as does the native dense solver. The
/// Python bar (`ratio > 4`) never came near noticing.
const SCIPY_SPLIT_LAMBDAS: [f64; 5] = [
    5.6519497525961695,
    6.002031732893242,
    5.167921895584456,
    3.7629593552267893,
    0.9229587201886765,
];

#[test]
fn the_free_plate_is_not_blind_to_the_split() {
    // ... while the free plate's fundamental spans 6.5x across the same admissible splits. Fixed
    // g_x = 1, g_y = 0.073, H = 0.153 (spruce's own cross term); only the split varies. Reported as
    // a range, not a ratio: it is non-monotone (peaks at zero coupling), and its low end is a plate
    // whose torsion has been driven almost to zero.
    let g_h = 0.153;
    let mut lams = Vec::new();
    for (g_1, scipy) in [-0.1, 0.0, 0.05, 0.1, 0.15].iter().zip(SCIPY_SPLIT_LAMBDAS) {
        let p = free_params(24, 0.5, [1.0, 0.073, *g_1, 0.5 * (g_h - g_1)]);
        let got = eigsh_shift_invert(&p.stiffness, p.mass.as_ref(), -1e-4, 6).unwrap();
        let mut v = got.values;
        v.sort_by(f64::total_cmp);
        let lam = p.lx * p.lx * v[3].max(0.0).sqrt();
        assert!(
            // Measured 3.0e-10 (lambda, i.e. half the relative error in mu); 1e-8 keeps 30x.
            rel(lam, scipy) < 1e-8,
            "g_1={g_1}: {lam:?} vs LAPACK {scipy:?}"
        );
        lams.push(lam);
    }
    let ratio = lams.iter().cloned().fold(f64::MIN, f64::max)
        / lams.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        ratio > 4.0,
        "the free plate barely noticed the split ({ratio:.2}x): {lams:?}"
    );
}

/// LAPACK's `eigh` for the two plates below: eigenvalues 4..6 (the first three elastic).
const SCIPY_REORDER: [(f64, [f64; 3]); 2] = [
    (
        0.04,
        [19.73938899894824, 31.410093887282507, 147.46393382493966],
    ),
    (
        0.10,
        [32.685811024017326, 49.34847249600486, 182.93079211342484],
    ),
];

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

#[test]
fn the_grain_reorders_the_free_plates_modes() {
    // The fundamental CHANGES IDENTITY with the grain, which never happens on the supported branch.
    // A race between a twist mode governed by D_xy alone and a cross-grain bender governed by D_y:
    // below g_y/g_xy ~ 1.03 the bender leads, above it the twist. Real spruce sits at 1.154, on the
    // twist side but close enough that the ordering is a material property.
    let (n, g_xy) = (24usize, 0.063);
    let h = 1.0 / n as f64;
    let xy = centred(n, n, h, |x, y| x * y);
    let mut kinds = Vec::new();
    for (g_y, scipy) in SCIPY_REORDER {
        let (k, w) = stiffness(n, n, h, [1.0, g_y, 0.0, g_xy]);
        let (vals, vecs) = pencil(&k, &w);
        for (j, want) in scipy.iter().enumerate() {
            assert!(
                // Measured 6.0e-11: 625 unknowns put LAPACK's own floor, eps·mu_max/mu, at
                // ~2.5e-11 here. 1e-9 keeps 16x.
                rel(vals[3 + j], *want) < 1e-9,
                "g_y={g_y} mode {j}: {}",
                vals[3 + j]
            );
        }
        let corr = pearson(&xy, &column(&vecs, k.nrows(), 3)).abs();
        kinds.push(if corr > 0.9 { "twist" } else { "bend" });
    }
    assert_eq!(
        kinds,
        ["bend", "twist"],
        "below the crossing the bender leads, above it the twist"
    );
    let s = spruce();
    assert!(
        s[1] / s[3] > 1.0,
        "spruce should sit on the twist-first side"
    );
}

#[test]
fn the_grain_is_worth_more_here_than_the_supported_branch_suggested() {
    // On a pinned edge a grain detunes selectively, 1.3%–29%. On a free edge the same material moves
    // the fundamental by a FACTOR: spruce's twist mode sits at roughly sqrt(g_xy) of the isotropic
    // plate's, because the fundamental is governed by the torsional rigidity alone.
    let iso = free_params(32, 0.5, ISOTROPIC);
    let spr = free_params(32, 0.5, spruce());
    let f_iso = freq_of(low_elastic_mu(&iso, 1, 13.5)[0], iso.kappa);
    let f_spr = freq_of(low_elastic_mu(&spr, 1, 6.0)[0], spr.kappa);
    assert!(rel(f_iso, SCIPY_F1_ISOTROPIC) < 1e-9 && rel(f_spr, SCIPY_F1_SPRUCE) < 1e-9);
    let ratio = f_spr / f_iso;
    let predicted = (spr.grain_torsion / iso.grain_torsion).sqrt();
    assert!(
        0.3 < ratio && ratio < 0.5,
        "spruce/isotropic fundamental ratio {ratio:.3}"
    );
    assert!(
        (ratio / predicted - 1.0).abs() < 0.10,
        "the drop should track sqrt(g_xy) to ~10%: {ratio:.4} vs {predicted:.4}"
    );
}

// -- tier 1: the ledger. Nearly blind here, and that is the point of saying so -----------------

fn plucked(spec: &PlateSpec) -> Plate {
    let p = Params::new(spec).unwrap();
    let a = p.lx;
    let u0 = raised_cosine_2d(&p.x, &p.y, (0.4 * a, 0.55 * a), 0.25 * a, 1e-3).unwrap();
    let mut plate = Plate::new(p);
    let v0 = vec![0.0; u0.len()];
    plate.set_state(&u0, &v0);
    plate
}

#[test]
fn a_grained_free_plate_conserves_its_energy() {
    // Unconditional conservation survives four constants. This cannot VALIDATE them — any symmetric
    // K conserves exactly — so it is here to catch the time-stepper, not the physics.
    for mu in [0.5, 4.0] {
        let mut plate = plucked(&free_spec(24, mu, spruce()));
        let steps = (0.5 * plate.p.fs) as usize;
        let res = simulate(&mut plate, steps, None, 0).unwrap();
        let drift = res.energy_drift();
        assert!(drift < DRIFT_TOL, "drift {drift:.2e} at mu={mu}");
    }
}

#[test]
fn a_lossy_grained_free_plate_is_passive() {
    let spec = PlateSpec {
        sigma: 8.0,
        ..free_spec(24, 1.0, spruce())
    };
    let mut plate = plucked(&spec);
    let steps = (0.3 * plate.p.fs) as usize;
    let res = simulate(&mut plate, steps, None, 0).unwrap();
    let e = &res.energy;
    let worst = e.windows(2).map(|w| w[1] - w[0]).fold(f64::MIN, f64::max);
    assert!(
        worst <= 1e-12 * e[0],
        "energy increased: max step {worst:.3e}"
    );
    assert!(e[e.len() - 1] < e[0]);
}

/// LAPACK's dense `eigh`: the three lowest elastic `mu` for spruce at `N = 20, 40, 80`.
///
/// ARPACK, through `free_plate_low_eigenfrequencies`, was 2.1e-9 off these at `N = 40`
/// (`35.900296245874394` against `35.90029617939288`); the native shift-invert is within 2e-10.
///
/// At `N = 80` the referees swap. A dense solve's error is absolute, `~eps·mu_max`, and `mu_max`
/// grows like `h^-4`, so relative to the low modes LAPACK's own floor is ~1e-9 on 6,561 unknowns.
/// There the native shift-invert sits 2.1e-9 from LAPACK (`36.034532037851115`) and 2.1e-10 from
/// ARPACK (`36.034532030050784`) — the two SHIFTED solvers agree and the dense one is the outlier.
/// Hence [`SCIPY_CONVERGENCE_TOL`].
const SCIPY_CONVERGENCE: [[f64; 3]; 3] = [
    [32.21545250475522, 35.36299441093958, 168.00053598594178],
    [32.413155388996586, 35.90029617939288, 170.3775753847438],
    [32.46220813026825, 36.03453196098951, 170.9693123802085],
];

/// LAPACK's floor at `N = 80`, above: measured 2.1e-9 there (1.2e-10 at `N = 40`, 5e-12 at
/// `N = 20`), so 2e-8 keeps ~10x.
const SCIPY_CONVERGENCE_TOL: f64 = 2e-8;

#[test]
fn a_grained_free_plate_self_converges_at_second_order() {
    // Richardson on the low eigenvalues: the only TWO-SIDED check on the spectrum as a whole (the
    // twist bound is one-sided). Measured orders 2.01, 2.00, 2.01.
    let mus: Vec<Vec<f64>> = [20i64, 40, 80]
        .iter()
        .zip(SCIPY_CONVERGENCE)
        .map(|(&n, scipy)| {
            let m = low_elastic_mu(&free_params(n, 0.5, spruce()), 3, 6.0);
            for (j, (got, want)) in m.iter().zip(scipy).enumerate() {
                assert!(
                    rel(*got, want) < SCIPY_CONVERGENCE_TOL,
                    "N={n} mode {j}: {got:?} vs LAPACK {want:?}"
                );
            }
            m
        })
        .collect();
    for j in 0..3 {
        let (d1, d2) = (mus[0][j] - mus[1][j], mus[1][j] - mus[2][j]);
        assert!(d2.abs() < d1.abs(), "mode {j} not converging: {mus:?}");
        let order = (d1.abs() / d2.abs()).log2();
        assert!(order > 1.6, "mode {j} convergence order {order} (want ~2)");
    }
}

// -- the material chain, and the shipped nonlinear plate that shares this operator --------------

#[test]
fn the_material_split_adds_back_to_its_cross_term() {
    // grain_coupling + 2*grain_torsion == grain_cross. The torsional share (82%) and the isotropic
    // split `(nu, (1-nu)/2)` are asserted in `plate.rs`; this is the one clause they left out.
    let s = grain_ratios_from_material(11.0e9, 0.8e9, 0.37, 0.7e9, 3.0e-3, 420.0).unwrap();
    assert!((s.grain_coupling + 2.0 * s.grain_torsion - s.grain_cross).abs() < 1e-15);
}

#[test]
fn the_von_karman_plates_bending_operator_is_the_isotropic_free_plate() {
    // The nonlinear plate shares this operator and stays ISOTROPIC (an orthotropic von Kármán plate
    // needs a four-constant in-plane compliance and has no oracle — refused). Its bending operator
    // must be byte-identical to the default.
    let vk = VkParams::new(&VkSpec {
        lx: 1.0,
        ly: 1.0,
        young: 2.0e11,
        thickness: 1.0e-3,
        nu: 0.3,
        rho: RHO,
        fs: 200_000.0,
        n: 12,
        boundary: Some(Boundary::Free),
        ..VkSpec::default()
    })
    .unwrap();
    let (k, w, _) = free_plate_stiffness(12, 12, vk.lin.h, 0.3, 1.0, 1.0, None, None);
    assert_eq!(k.indices(), vk.lin.stiffness.indices());
    assert_eq!(
        k.data(),
        vk.lin.stiffness.data(),
        "the von Karman bending operator moved"
    );
    assert_eq!(diag(&w), vk.lin.w);
}

#[test]
fn an_implied_poisson_ratio_above_one_half_is_admissible() {
    // The isotropic (-1, 1/2) range is admissibility for ONE Poisson ratio; an orthotropic sheet is
    // bounded by nu_xy nu_yx < 1, i.e. this branch's guard g_1² < g_x g_y. Applying the isotropic
    // range to the IMPLIED ratio once rejected a valid material (g_1 = 0.70 at g_y = 0.5).
    let p = free_params(8, 2.0, [1.0, 0.5, 0.70, 0.3]);
    assert!(
        p.nu == 0.70 && p.nu > 0.5,
        "the implied nu_yx is exposed unclamped"
    );
    let (k, _, _) = free_plate_stiffness(8, 8, p.h, p.nu, 1.0, 0.5, Some(0.70), Some(0.3));
    assert_eq!(k.data(), p.stiffness.data());
}

#[test]
#[should_panic(expected = "Poisson")]
fn a_plain_isotropic_call_still_refuses_that_poisson_ratio() {
    // ... but where nu IS used — a plain isotropic call — the same 0.70 is still refused.
    let _ = free_plate_stiffness(8, 8, 0.1, 0.70, 1.0, 1.0, None, None);
}
