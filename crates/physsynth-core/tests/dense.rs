//! Native bars for the dense LU (plan §16).
//!
//! What can be asserted without Python: that `P A = L U` holds for the factors handed back, that
//! the solve inverts the matrix, that the pivot is the largest sub-column entry rather than merely
//! a nonzero one, and that an exactly singular matrix reports the LAPACK `info` index instead of
//! panicking. The last is the one this project actually depends on being *stated* rather than
//! guessed at: `collision`'s Jacobian cannot be singular, so nothing else would ever exercise it.

use physsynth_core::dense::{lu_factor, lu_solve, DenseError, Lu};

fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

/// `L U` reassembled from the packed factor, then the row permutation undone, giving back `A`.
fn reconstruct(f: &Lu) -> Vec<f64> {
    let n = f.n;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..=i.min(j) {
                let l = if p == i { 1.0 } else { f.lu[i * n + p] };
                if p <= i && p <= j {
                    acc += l * f.lu[p * n + j];
                }
            }
            a[i * n + j] = acc;
        }
    }
    // Undo the interchanges in reverse order: the factorization swapped row i with row piv[i]
    // going forward, so `A` comes back by swapping in the opposite order.
    for i in (0..n).rev() {
        let p = f.piv[i];
        if p != i {
            for c in 0..n {
                a.swap(i * n + c, p * n + c);
            }
        }
    }
    a
}

/// A deterministic, well-conditioned test matrix with entries that are not powers of two — §14.3's
/// lesson, that a fixture built from 1.0s cannot see a rounding difference at all.
fn model_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (x, y) = (i as f64, j as f64);
            a[i * n + j] = if i == j {
                3.0 + 0.37 * x
            } else {
                0.13 / (1.0 + (x - y).abs()) - 0.07 * ((x + y) * 0.31).sin()
            };
        }
    }
    a
}

#[test]
fn factor_reconstructs_the_matrix() {
    for n in [1usize, 2, 3, 5, 13, 40] {
        let a = model_matrix(n);
        let f = lu_factor(a.clone(), n).expect("square");
        assert_eq!(f.info, 0, "n = {n} reported a singular pivot");
        let back = reconstruct(&f);
        for i in 0..n * n {
            assert!(
                (back[i] - a[i]).abs() <= 1e-13 * (1.0 + a[i].abs()),
                "n = {n}, entry {i}: {} vs {}",
                back[i],
                a[i]
            );
        }
    }
}

#[test]
fn solve_inverts_the_matrix() {
    for n in [1usize, 2, 7, 25] {
        let a = model_matrix(n);
        let x: Vec<f64> = (0..n).map(|i| 0.41 - 0.19 * (i as f64)).collect();
        let b = matvec(&a, n, &x);
        let f = lu_factor(a, n).expect("square");
        let got = lu_solve(&f, &b).expect("matching rhs");
        for i in 0..n {
            assert!(
                (got[i] - x[i]).abs() <= 1e-12 * (1.0 + x[i].abs()),
                "n = {n}, component {i}: {} vs {}",
                got[i],
                x[i]
            );
        }
    }
}

#[test]
fn pivoting_picks_the_largest_and_keeps_the_first_of_a_tie() {
    // Column 0 is [1, -4, 4]: the pivot must be row 1 (strictly larger magnitude), and the tie
    // between rows 1 and 2 must go to the FIRST, which is what LAPACK's IDAMAX does and what makes
    // the choice reproducible rather than a property of the loop direction.
    let a = vec![1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 4.0, 7.0, 9.0];
    let f = lu_factor(a, 3).expect("square");
    assert_eq!(f.piv[0], 1);
}

#[test]
fn a_zero_pivot_reports_the_lapack_info_index() {
    // Column 1 of the trailing 2x2 is all zeros, so U[1][1] is exactly zero: LAPACK sets info = 2.
    let a = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 1.0];
    let f = lu_factor(a, 3).expect("square");
    assert_eq!(f.info, 2, "the second leading minor is singular");
}

#[test]
fn a_bad_shape_is_refused_rather_than_guessed_at() {
    assert_eq!(
        lu_factor(vec![1.0, 2.0, 3.0], 2).err(),
        Some(DenseError::BadShape)
    );
    assert_eq!(lu_factor(vec![], 0).err(), Some(DenseError::BadShape));
    let f = lu_factor(model_matrix(3), 3).expect("square");
    assert_eq!(lu_solve(&f, &[1.0, 2.0]), Err(DenseError::BadShape));
}

#[test]
fn the_identity_is_its_own_factor() {
    // The case `collision` spends most of its life in: with no node in contact the Jacobian is
    // exactly `I`, no row is ever swapped, and the solve is the right-hand side unchanged. If this
    // ever stops being exact, the barrier stops being bit-identical to a bare string.
    let n = 6;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    let f = lu_factor(a, n).expect("square");
    assert!(f.piv.iter().enumerate().all(|(i, &p)| p == i));
    let b: Vec<f64> = (0..n).map(|i| 0.31 * (i as f64) - 0.7).collect();
    assert_eq!(lu_solve(&f, &b).unwrap(), b);
}
