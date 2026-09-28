//! Native bars for `eigs` — the shift-invert block Krylov solver behind the viewer's mode markers.
//!
//! New numerics, so the project's rule applies (`CLAUDE.md`): correctness against spectra known in
//! closed form, before any payload calls it (retirement plan §23.11). The fixtures are chosen for
//! what goes wrong in this family of solvers:
//!
//! * a **repeated eigenvalue** — a single-vector Krylov space reports it once (the 2-D grid's
//!   `(p, q)` / `(q, p)` pairs, and a diagonal pencil with a triple);
//! * a **singular `K`** under a negative shift — the free plate's rigid modes and the closed
//!   bore's `omega = 0` mode, which the viewer reaches exactly that way;
//! * a **mass matrix** — the plate and the bore solve `K x = lambda M x`, not `K x = lambda x`;
//! * a **shift inside the spectrum** — "nearest sigma" is two-sided.

use physsynth_core::eig::symmetric_eigenvalues;
use physsynth_core::eigs::{eigsh_shift_invert, EigsError};
use physsynth_core::sparse::Csr;
use std::f64::consts::PI;

/// The 1-D second difference `(-1, 2, -1)` on `n` nodes, Dirichlet (`neumann = false`) or with
/// both ends reflected (`true`, first and last diagonal 1: the constant is in the nullspace).
fn second_difference(n: usize, neumann: bool) -> Csr {
    let rows = (0..n)
        .map(|i| {
            let mut r = Vec::new();
            if i > 0 {
                r.push((i - 1, -1.0));
            }
            let d = if neumann && (i == 0 || i == n - 1) {
                1.0
            } else {
                2.0
            };
            r.push((i, d));
            if i + 1 < n {
                r.push((i + 1, -1.0));
            }
            r
        })
        .collect();
    Csr::from_rows(n, n, rows)
}

/// The 2-D five-point Laplacian (negated, so positive definite) on an `n x n` interior, Dirichlet.
fn laplacian_2d(n: usize) -> Csr {
    let id = |i: usize, j: usize| i * n + j;
    let rows = (0..n * n)
        .map(|p| {
            let (i, j) = (p / n, p % n);
            let mut r = vec![(p, 4.0)];
            if i > 0 {
                r.push((id(i - 1, j), -1.0));
            }
            if i + 1 < n {
                r.push((id(i + 1, j), -1.0));
            }
            if j > 0 {
                r.push((id(i, j - 1), -1.0));
            }
            if j + 1 < n {
                r.push((id(i, j + 1), -1.0));
            }
            r
        })
        .collect();
    Csr::from_rows(n * n, n * n, rows)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// `||K x - lambda M x||` relative to `||K x||`, and `M`-orthonormality, for every returned pair.
fn assert_pairs_are_eigenpairs(k: &Csr, m: Option<&Csr>, got: &physsynth_core::eigs::Eigenpairs) {
    let mv = |x: &[f64]| m.map_or_else(|| x.to_vec(), |m| m.matvec(x));
    for (j, (lam, x)) in got.values.iter().zip(&got.vectors).enumerate() {
        let kx = k.matvec(x);
        let mx = mv(x);
        let res: f64 = kx
            .iter()
            .zip(&mx)
            .map(|(a, b)| (a - lam * b).powi(2))
            .sum::<f64>()
            .sqrt();
        // Scaled by the operator's size, not by `||K x||`: for the rigid mode that is ~1e-16 and
        // would demand a residual below rounding.
        let k_norm = (0..k.nrows())
            .map(|i| {
                k.data()[k.indptr()[i]..k.indptr()[i + 1]]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f64>()
            })
            .fold(0.0f64, f64::max);
        assert!(
            res <= 1e-10 * k_norm,
            "pair {j}: residual {res:e} vs ||K|| {k_norm:e}"
        );
        for (l, y) in got.vectors.iter().enumerate() {
            let d: f64 = y.iter().zip(&mx).map(|(a, b)| a * b).sum();
            let want = if j == l { 1.0 } else { 0.0 };
            assert!((d - want).abs() < 1e-9, "M-inner product ({j}, {l}) = {d}");
        }
    }
}

#[test]
fn the_1d_dirichlet_operator_matches_its_closed_form() {
    let n = 300;
    let k = second_difference(n, false);
    let got = eigsh_shift_invert(&k, None, 0.0, 8).unwrap();
    for (j, &lam) in got.values.iter().enumerate() {
        let m = (j + 1) as f64;
        let want = 4.0 * (m * PI / (2.0 * (n as f64 + 1.0))).sin().powi(2);
        assert!(rel(lam, want) < 1e-12, "mode {m}: {lam} vs {want}");
    }
    assert_pairs_are_eigenpairs(&k, None, &got);
    // the vectors are the discrete sines, up to sign
    for (j, x) in got.vectors.iter().enumerate() {
        let m = (j + 1) as f64;
        let s: Vec<f64> = (0..n)
            .map(|i| (m * (i as f64 + 1.0) * PI / (n as f64 + 1.0)).sin())
            .collect();
        let ns = s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let cos: f64 = x.iter().zip(&s).map(|(a, b)| a * b).sum::<f64>() / ns;
        assert!(
            (cos.abs() - 1.0).abs() < 1e-10,
            "mode {m}: |cos| = {}",
            cos.abs()
        );
    }
}

/// The 2-D grid's `(p, q)` / `(q, p)` pairs are EXACT repeats, and the whole list matches the
/// closed form in order. This is a closed-form bar, NOT the multiplicity discriminator: planted
/// `BLOCK = 1`, it still passes, because by the time 14 pairs converge the basis is long enough for
/// rounding to have fed each pair's second direction in. The triple below is the one that fails at
/// block 1 — measured, which is why both exist.
#[test]
fn the_2d_laplacian_returns_every_repeated_pair_with_its_multiplicity() {
    let n = 25;
    let k = laplacian_2d(n);
    let nev = 14;
    let got = eigsh_shift_invert(&k, None, 0.0, nev).unwrap();
    let s = |p: usize| 4.0 * (p as f64 * PI / (2.0 * (n as f64 + 1.0))).sin().powi(2);
    let mut want: Vec<f64> = (1..=n)
        .flat_map(|p| (1..=n).map(move |q| (p, q)))
        .map(|(p, q)| s(p) + s(q))
        .collect();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // a cut through a repeated pair is still a list of the nearest values
    for (j, (&g, &w)) in got.values.iter().zip(&want).enumerate() {
        assert!(rel(g, w) < 1e-11, "eigenvalue {j}: {g} vs {w}");
    }
    let repeats = got
        .values
        .windows(2)
        .filter(|p| rel(p[1], p[0]) < 1e-9)
        .count();
    assert!(
        repeats >= 5,
        "the (p, q)/(q, p) pairs must come back doubled: {repeats}"
    );
    assert_pairs_are_eigenpairs(&k, None, &got);
}

/// A diagonal pencil with a TRIPLE, and a shift inside the spectrum: "nearest" is two-sided, and
/// a multiplicity equal to the block size is still found whole. A diagonal operator gives rounding
/// nothing to mix, so this is the fixture a single-vector space fails (planted `BLOCK = 1`: fails).
#[test]
fn a_generalized_pencil_with_a_triple_is_solved_on_both_sides_of_the_shift() {
    let lambdas = [
        0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ];
    let n = lambdas.len();
    let mass: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (i as f64).sin()).collect();
    let stiff: Vec<f64> = (0..n).map(|i| lambdas[i] * mass[i]).collect();
    let k = Csr::diagonal(&stiff);
    let m = Csr::diagonal(&mass);
    let got = eigsh_shift_invert(&k, Some(&m), 1.1, 6).unwrap();
    let want = [0.5, 1.0, 1.0, 1.0, 2.0, 2.0];
    for (g, w) in got.values.iter().zip(want) {
        assert!(rel(*g, w) < 1e-12, "{:?}", got.values);
    }
    assert_pairs_are_eigenpairs(&k, Some(&m), &got);
}

/// The reflected operator is singular (the constant). A shift just below zero reaches it the way
/// the viewer reaches a free plate's rigid modes and a closed bore's `omega = 0` mode.
#[test]
fn a_singular_operator_is_solved_through_a_negative_shift() {
    let n = 120;
    let k = second_difference(n, true);
    let got = eigsh_shift_invert(&k, None, -1e-3, 5).unwrap();
    for (j, &lam) in got.values.iter().enumerate() {
        let want = 2.0 - 2.0 * (j as f64 * PI / n as f64).cos();
        assert!(
            (lam - want).abs() < 1e-12 * want.max(1.0),
            "mode {j}: {lam} vs {want}"
        );
    }
    assert_pairs_are_eigenpairs(&k, None, &got);
}

/// A mass matrix that is not diagonal-constant: the eigenvalues are those of the dense
/// `M^-1/2 K M^-1/2`, which the dense routine gives independently.
#[test]
fn a_varying_mass_matrix_matches_the_dense_similarity_transform() {
    let n = 60;
    let k = second_difference(n, false);
    let mass: Vec<f64> = (0..n).map(|i| 1.0 + 0.3 * (0.7 * i as f64).cos()).collect();
    let m = Csr::diagonal(&mass);
    let got = eigsh_shift_invert(&k, Some(&m), 0.0, 7).unwrap();
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            dense[i * n + j] = k.get(i, j) / (mass[i] * mass[j]).sqrt();
        }
    }
    let all = symmetric_eigenvalues(&dense, n).unwrap();
    for (j, &g) in got.values.iter().enumerate() {
        assert!(rel(g, all[j]) < 1e-11, "eigenvalue {j}: {g} vs {}", all[j]);
    }
    assert_pairs_are_eigenpairs(&k, Some(&m), &got);
}

/// Asking for nearly everything: the basis reaches the whole space, where the answer is exact.
#[test]
fn asking_for_n_minus_one_pairs_spans_the_whole_space() {
    let n = 10;
    let k = second_difference(n, false);
    let got = eigsh_shift_invert(&k, None, 0.0, n - 1).unwrap();
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            dense[i * n + j] = k.get(i, j);
        }
    }
    let all = symmetric_eigenvalues(&dense, n).unwrap();
    for (j, &g) in got.values.iter().enumerate() {
        assert!(rel(g, all[j]) < 1e-12, "eigenvalue {j}");
    }
    assert!(got.basis_size <= n);
}

#[test]
fn the_run_is_deterministic_to_the_bit() {
    let k = laplacian_2d(12);
    let a = eigsh_shift_invert(&k, None, 0.0, 6).unwrap();
    let b = eigsh_shift_invert(&k, None, 0.0, 6).unwrap();
    assert_eq!(a.values, b.values);
    assert_eq!(a.vectors, b.vectors);
}

#[test]
fn bad_shapes_and_counts_are_refused() {
    let k = second_difference(5, false);
    for nev in [0, 5, 6] {
        assert!(
            matches!(
                eigsh_shift_invert(&k, None, 0.0, nev),
                Err(EigsError::BadShape(_))
            ),
            "{nev}"
        );
    }
    let m = Csr::identity(4);
    assert!(matches!(
        eigsh_shift_invert(&k, Some(&m), 0.0, 2),
        Err(EigsError::BadShape(_))
    ));
    let rect = Csr::from_rows(2, 3, vec![vec![(0, 1.0)], vec![(1, 1.0)]]);
    assert!(matches!(
        eigsh_shift_invert(&rect, None, 0.0, 1),
        Err(EigsError::BadShape(_))
    ));
}

/// The viewer's size: a 99 x 99 interior (9,801 unknowns, the membrane's `n_live` ceiling is
/// 9,900) and the twelve markers it asks for. Closed form again, at the scale the payload runs.
#[test]
fn the_viewers_largest_membrane_problem_matches_the_closed_form() {
    let n = 99;
    let k = laplacian_2d(n);
    let t0 = std::time::Instant::now();
    let got = eigsh_shift_invert(&k, None, 0.0, 12).unwrap();
    eprintln!(
        "n = {}: {:?}, basis {}",
        n * n,
        t0.elapsed(),
        got.basis_size
    );
    let s = |p: usize| 4.0 * (p as f64 * PI / (2.0 * (n as f64 + 1.0))).sin().powi(2);
    let mut want: Vec<f64> = (1..=6)
        .flat_map(|p| (1..=6).map(move |q| s(p) + s(q)))
        .collect();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (j, (&g, &w)) in got.values.iter().zip(&want).enumerate() {
        assert!(rel(g, w) < 1e-11, "eigenvalue {j}: {g} vs {w}");
    }
}

/// The fixture that found the convergence bug: a FREE plate, whose `K - sigma W` is nearly
/// singular because the three rigid modes sit right beside the negative shift. Measured against
/// the full residual `||Op y - theta y||`, the solve error floored above the bar and the basis
/// grew to the whole space (minutes, not a second). The bar: the basis stays small, the rigid trio
/// comes back at zero, and every pair satisfies `K x = lambda W x`.
#[test]
fn a_free_plate_converges_through_its_nearly_singular_shift() {
    use physsynth_core::plate::{Boundary, Params, PlateSpec};
    let n = 40;
    let (kappa, lx) = (20.0, 1.0);
    let h = lx / n as f64;
    let spec = PlateSpec {
        lx,
        ly: 1.0,
        kappa,
        rho: 0.005,
        fs: kappa / (h * h),
        n,
        boundary: Some(Boundary::Free),
        nu: Some(0.3),
        ..PlateSpec::default()
    };
    let p = Params::new(&spec).unwrap();
    let w = p.mass.as_ref().unwrap();
    let shift = -1e-3 * (13.0f64 / (lx * 1.0)).powi(2);
    let got = eigsh_shift_invert(&p.stiffness, Some(w), shift, 9).unwrap();
    assert!(got.basis_size <= 120, "basis grew to {}", got.basis_size);
    let top = got.values[8];
    for (j, &v) in got.values.iter().take(3).enumerate() {
        assert!(v.abs() < 1e-8 * top, "rigid mode {j}: {v}");
    }
    assert!(
        got.values[3] > 1e-3 * top,
        "the first elastic mode is not rigid"
    );
    assert_pairs_are_eigenpairs(&p.stiffness, Some(w), &got);
}
