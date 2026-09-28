//! Native bars for `eig` — the symmetric eigenvalue routine `connection`'s stability guard needs.
//!
//! There is no Python counterpart to these: the Python side called `np.linalg.eigvals`, i.e.
//! LAPACK, and asserted nothing about it. So this file follows the project's rule for new numerics
//! (`CLAUDE.md`: correctness is asserted against closed-form physics, not against another
//! implementation) and checks the routine against spectra that are known exactly — a diagonal
//! matrix, a 2x2 in closed form, and the tridiagonal `(1, -2, 1)` operator whose eigenvalues are
//! `-4 sin^2(m pi / 2(n+1))` — plus the two similarity invariants that hold for every symmetric
//! matrix, on a case with no closed form at all.
//!
//! The clustered and degenerate cases are here on purpose: they are what ruled out power
//! iteration for the guard (see the module header), so a routine that quietly failed on them
//! would be failing at exactly the fixture the caller has.

use physsynth_core::eig::{
    symmetric_eigen, symmetric_eigenvalues, symmetric_max_eigenvalue, EigError,
};

/// Largest absolute difference between two same-length sequences.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "sequences must be the same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// A dense symmetric matrix from a closure over `(i, j)`, row-major.
fn build(n: usize, at: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = at(i, j);
        }
    }
    a
}

// -- shapes and refusals -------------------------------------------------------------------------

#[test]
fn a_ragged_matrix_is_refused() {
    let err = symmetric_eigenvalues(&[1.0, 2.0, 3.0], 2).unwrap_err();
    assert_eq!(err, EigError::BadShape);
    assert_eq!(
        err.to_string(),
        "matrix data must have exactly n * n entries."
    );
}

#[test]
fn an_empty_matrix_has_no_eigenvalues_and_no_largest_one() {
    assert!(symmetric_eigenvalues(&[], 0).unwrap().is_empty());
    assert_eq!(
        symmetric_max_eigenvalue(&[], 0).unwrap_err(),
        EigError::BadShape
    );
}

#[test]
fn a_one_by_one_matrix_is_its_own_eigenvalue() {
    assert_eq!(symmetric_eigenvalues(&[7.5], 1).unwrap(), vec![7.5]);
    assert_eq!(symmetric_max_eigenvalue(&[7.5], 1).unwrap(), 7.5);
}

// -- closed forms --------------------------------------------------------------------------------

#[test]
fn a_diagonal_matrix_returns_its_diagonal_sorted() {
    let d = [3.0, -1.0, 10.0, 0.0, 2.5];
    let a = build(5, |i, j| if i == j { d[i] } else { 0.0 });
    let values = symmetric_eigenvalues(&a, 5).unwrap();
    assert_eq!(values, vec![-1.0, 0.0, 2.5, 3.0, 10.0]);
}

#[test]
fn a_two_by_two_matches_the_quadratic_formula() {
    // [[a, b], [b, d]] has eigenvalues (a + d)/2 +- sqrt(((a - d)/2)^2 + b^2).
    let (a, b, d): (f64, f64, f64) = (4.0, -3.0, 1.0);
    let mid = 0.5 * (a + d);
    let half = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    let values = symmetric_eigenvalues(&[a, b, b, d], 2).unwrap();
    assert!(
        max_abs_diff(&values, &[mid - half, mid + half]) < 1e-14,
        "2x2 spectrum {values:?} does not match the closed form"
    );
}

#[test]
fn the_second_difference_operator_matches_its_analytic_spectrum() {
    // The Dirichlet `(1, -2, 1)` tridiagonal of order n has eigenvalues
    // `-4 sin^2(m pi / (2 (n + 1)))`, m = 1..n — the discrete spectrum every string model in this
    // crate is built on, so getting it wrong here would be getting the physics wrong twice.
    let n = 60;
    let a = build(n, |i, j| {
        if i == j {
            -2.0
        } else if i.abs_diff(j) == 1 {
            1.0
        } else {
            0.0
        }
    });
    let values = symmetric_eigenvalues(&a, n).unwrap();
    let mut exact: Vec<f64> = (1..=n)
        .map(|m| {
            let s = ((m as f64) * std::f64::consts::PI / (2.0 * (n as f64 + 1.0))).sin();
            -4.0 * s * s
        })
        .collect();
    exact.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!(
        max_abs_diff(&values, &exact) < 1e-13,
        "worst deviation {:.3e} from the analytic spectrum",
        max_abs_diff(&values, &exact)
    );
    assert!((symmetric_max_eigenvalue(&a, n).unwrap() - exact[n - 1]).abs() < 1e-13);
}

// -- the cases that ruled out power iteration -----------------------------------------------------

#[test]
fn a_degenerate_top_eigenvalue_is_returned_with_its_multiplicity() {
    // Two identical strings make the coupled operator's top eigenvalue exactly degenerate. A
    // routine that deflates one copy and misses the other would still satisfy the guard's
    // `lambda_max`, and would be wrong about the operator.
    let a = build(4, |i, j| match (i, j) {
        (0, 0) | (1, 1) => 5.0,
        (2, 2) | (3, 3) => 5.0,
        (0, 1) | (1, 0) => 2.0,
        (2, 3) | (3, 2) => 2.0,
        _ => 0.0,
    });
    let values = symmetric_eigenvalues(&a, 4).unwrap();
    assert!(
        max_abs_diff(&values, &[3.0, 3.0, 7.0, 7.0]) < 1e-14,
        "{values:?}"
    );
}

#[test]
fn an_identity_is_all_ones_rather_than_a_non_convergence() {
    // The fully degenerate case: every subdiagonal is already zero, so the QL sweep must exit
    // immediately for every eigenvalue rather than iterate on a zero shift.
    let n = 12;
    let a = build(n, |i, j| if i == j { 1.0 } else { 0.0 });
    let values = symmetric_eigenvalues(&a, n).unwrap();
    assert_eq!(values, vec![1.0; n]);
}

#[test]
fn a_clustered_spectrum_separates_to_full_precision() {
    // `lambda_2 / lambda_1 = 0.9969` on the suite's two-string fixture; this is that ratio made
    // far worse. Power iteration needs ~1e5 matrix-vector products to separate these; the
    // tridiagonal reduction does not iterate on the ratio at all.
    let d = [1.0, 1.0 - 1e-11, 1.0 - 2e-11, 0.5];
    let a = build(4, |i, j| if i == j { d[i] } else { 0.0 });
    let values = symmetric_eigenvalues(&a, 4).unwrap();
    assert_eq!(values[3], 1.0);
    assert!(values[2] < values[3], "the cluster collapsed: {values:?}");
    assert!(values[1] < values[2], "the cluster collapsed: {values:?}");
}

// -- invariants, where there is no closed form ----------------------------------------------------

#[test]
fn the_trace_and_the_frobenius_norm_survive_the_reduction() {
    // Similarity preserves `sum lambda_i` and (for a symmetric matrix) `sum lambda_i^2`. Together
    // they pin the spectrum of a matrix with no analytic one — a deterministic pseudo-random fill,
    // dense enough that every branch of the Householder loop runs.
    let n = 40;
    let mut seed = 12345u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = next();
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    let values = symmetric_eigenvalues(&a, n).unwrap();

    let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
    let frob: f64 = a.iter().map(|v| v * v).sum();
    let sum: f64 = values.iter().sum();
    let sum_sq: f64 = values.iter().map(|v| v * v).sum();
    assert!((sum - trace).abs() < 1e-12, "trace {trace} vs {sum}");
    assert!(
        (sum_sq - frob).abs() < 1e-12 * frob,
        "Frobenius {frob} vs {sum_sq}"
    );
    // Sorted ascending, which every caller relies on and `symmetric_max_eigenvalue` states.
    assert!(values.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(symmetric_max_eigenvalue(&a, n).unwrap(), values[n - 1]);
}

// -- eigenvectors (`symmetric_eigen`, EISPACK tred2/tql2) ------------------------------------------

/// A fixed, structureless symmetric matrix: no closed form, so only invariants can judge it.
fn scrambled(n: usize) -> Vec<f64> {
    build(n, |i, j| {
        let (a, b) = (i.min(j) as f64, i.max(j) as f64);
        (0.37 * a + 1.3 * b).sin() + if i == j { 0.1 * i as f64 } else { 0.0 }
    })
}

/// The vector path returns the value path's eigenvalues to the BIT: the extra bookkeeping writes
/// only the upper triangle and a separate matrix, which is what lets it inherit the bars above.
#[test]
fn the_vector_path_returns_the_same_doubles_as_the_value_path() {
    for n in [1, 2, 3, 7, 20, 41] {
        let a = scrambled(n);
        let (vals, _) = symmetric_eigen(&a, n).unwrap();
        assert_eq!(vals, symmetric_eigenvalues(&a, n).unwrap(), "n = {n}");
    }
}

#[test]
fn eigenvectors_are_orthonormal_and_satisfy_the_eigen_equation() {
    let n = 41;
    let a = scrambled(n);
    let (vals, v) = symmetric_eigen(&a, n).unwrap();
    let scale = vals.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for j in 0..n {
        for l in 0..n {
            let dot: f64 = (0..n).map(|i| v[i * n + j] * v[i * n + l]).sum();
            let want = if j == l { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-13, "V^T V at ({j}, {l}) = {dot}");
        }
        for i in 0..n {
            let av: f64 = (0..n).map(|k| a[i * n + k] * v[k * n + j]).sum();
            assert!(
                (av - vals[j] * v[i * n + j]).abs() < 1e-13 * scale,
                "A v != lambda v, j = {j}"
            );
        }
    }
}

/// The `(1, -2, 1)` operator's eigenvectors are `sin(m k pi / (n+1))`, up to sign and scale.
#[test]
fn the_second_difference_eigenvectors_are_the_discrete_sines() {
    let n = 30;
    let a = build(n, |i, j| match i.abs_diff(j) {
        0 => -2.0,
        1 => 1.0,
        _ => 0.0,
    });
    let (vals, v) = symmetric_eigen(&a, n).unwrap();
    let pi = std::f64::consts::PI;
    for (j, &lam) in vals.iter().enumerate() {
        // ascending: the most negative eigenvalue is the highest mode, m = n - j
        let m = (n - j) as f64;
        let want = -4.0 * (m * pi / (2.0 * (n as f64 + 1.0))).sin().powi(2);
        assert!((lam - want).abs() < 1e-13, "eigenvalue {j}");
        let s: Vec<f64> = (0..n)
            .map(|k| (m * (k as f64 + 1.0) * pi / (n as f64 + 1.0)).sin())
            .collect();
        let ns = s.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cos: f64 = (0..n).map(|k| v[k * n + j] * s[k] / ns).sum();
        assert!(
            (cos.abs() - 1.0).abs() < 1e-12,
            "mode {m}: |cos| = {}",
            cos.abs()
        );
    }
}

/// A repeated eigenvalue gets an orthonormal basis of its WHOLE eigenspace, not two copies of one
/// vector — which is what a Rayleigh-Ritz step over a degenerate pair depends on.
#[test]
fn a_repeated_eigenvalue_gets_an_orthonormal_basis_of_its_eigenspace() {
    let n = 6;
    let d = [3.0, 1.0, 3.0, 2.0, 3.0, 5.0];
    // rotate diag(d) by a fixed orthogonal matrix so the eigenspace is not coordinate-aligned
    let q = {
        let (vals, vecs) = symmetric_eigen(&scrambled(n), n).unwrap();
        let _ = vals;
        vecs
    };
    let a = build(n, |i, j| {
        (0..n).map(|k| q[i * n + k] * d[k] * q[j * n + k]).sum()
    });
    let (vals, v) = symmetric_eigen(&a, n).unwrap();
    assert!(max_abs_diff(&vals, &[1.0, 2.0, 3.0, 3.0, 3.0, 5.0]) < 1e-13);
    for j in 2..5 {
        for l in 2..5 {
            let dot: f64 = (0..n).map(|i| v[i * n + j] * v[i * n + l]).sum();
            assert!((dot - if j == l { 1.0 } else { 0.0 }).abs() < 1e-13);
        }
    }
}

#[test]
fn the_vector_path_refuses_a_ragged_matrix_and_accepts_an_empty_one() {
    assert_eq!(
        symmetric_eigen(&[1.0, 2.0, 3.0], 2).unwrap_err(),
        EigError::BadShape
    );
    let (vals, vecs) = symmetric_eigen(&[], 0).unwrap();
    assert!(vals.is_empty() && vecs.is_empty());
}

// -- the generalized problem with a diagonal mass (`generalized_eigen_diag`) ----------------------

/// The case the room scene needs: a FREE square plate's `K x = lambda W x`, whose spectrum has the
/// rigid trio at zero and exactly repeated pairs. Asserted: `V^T W V = I` (a projection under the
/// mass is only a projection with this), `K V = W V Lambda`, the trio, a repeated pair, and the
/// low values against the independent sparse shift-invert route.
#[test]
fn a_free_square_plates_pencil_is_solved_mass_orthonormally() {
    use physsynth_core::eig::generalized_eigen_diag;
    use physsynth_core::eigs::eigsh_shift_invert;
    use physsynth_core::plate::{Boundary, Params, PlateSpec};
    let n_seg = 8;
    let h = 1.0 / n_seg as f64;
    let spec = PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: 20.0,
        rho: 0.005,
        fs: 20.0 / (h * h),
        n: n_seg,
        boundary: Some(Boundary::Free),
        nu: Some(0.3),
        ..PlateSpec::default()
    };
    let p = Params::new(&spec).unwrap();
    let n = p.n_live;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = p.stiffness.get(i, j);
        }
    }
    let w = &p.w;
    let (vals, v) = generalized_eigen_diag(&a, w, n).unwrap();
    let scale = vals[n - 1].abs();
    for j in 0..n {
        for l in 0..n {
            let g: f64 = (0..n).map(|i| v[i * n + j] * w[i] * v[i * n + l]).sum();
            assert!(
                (g - if j == l { 1.0 } else { 0.0 }).abs() < 1e-10,
                "V^T W V ({j}, {l}) = {g}"
            );
        }
        for i in 0..n {
            let kv: f64 = (0..n).map(|k| a[i * n + k] * v[k * n + j]).sum();
            let r = kv - vals[j] * w[i] * v[i * n + j];
            assert!(r.abs() < 1e-9 * scale, "K v != lambda W v at mode {j}");
        }
    }
    assert!(
        vals[..3].iter().all(|x| x.abs() < 1e-9 * scale),
        "the rigid trio: {:?}",
        &vals[..3]
    );
    assert!(vals[3] > 1e-6 * scale);
    let pairs = vals
        .windows(2)
        .filter(|x| (x[1] - x[0]).abs() < 1e-9 * x[1].abs())
        .count();
    assert!(pairs >= 1, "a square free plate has exactly repeated pairs");
    let sparse = eigsh_shift_invert(&p.stiffness, p.mass.as_ref(), -1e-3 * 169.0, 9).unwrap();
    for (j, (&s, &d)) in sparse.values.iter().zip(&vals).enumerate().skip(3) {
        assert!(
            (s - d).abs() < 1e-9 * d.abs(),
            "mode {j}: sparse {s} vs dense {d}"
        );
    }
    // refusals
    assert_eq!(
        generalized_eigen_diag(&a, &w[..n - 1], n).unwrap_err(),
        EigError::BadShape
    );
    let mut bad = w.clone();
    bad[5] = 0.0;
    assert_eq!(
        generalized_eigen_diag(&a, &bad, n).unwrap_err(),
        EigError::MassNotPositive(5)
    );
}
