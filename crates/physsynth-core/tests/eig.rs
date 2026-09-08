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

use physsynth_core::eig::{symmetric_eigenvalues, symmetric_max_eigenvalue, EigError};

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
