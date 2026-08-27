//! Native bars for the banded Cholesky (plan §15).
//!
//! What can be asserted here is what does not need Python: that the factor reconstructs the matrix,
//! that the solve inverts it, that a non-SPD band is refused at the right minor, and — the one that
//! is about *this project* rather than about linear algebra — that a pentadiagonal band whose
//! second superdiagonal is numerically zero gives the same answer as the tridiagonal band it
//! describes. That case is `kappa = 0`, which every string model in the family accepts and which a
//! `kd = 2` loop can get wrong while passing every other test here.

use physsynth_core::banded::{cho_solve_banded_upper, cholesky_banded_upper, BandedError};

/// Expand an upper band to a dense symmetric matrix, row-major `n × n`.
fn dense(ab: &[f64], kd: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j.saturating_sub(kd)..=j {
            let v = ab[(kd + i - j) * n + j];
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

/// A pentadiagonal SPD band of the shape the θ-scheme builds: `(1 + sk) I - θ k² L`.
fn model_band(n: usize, kappa: f64) -> Vec<f64> {
    let mut ab = vec![0.0; 3 * n];
    for j in 0..n {
        ab[2 * n + j] = 4.0 + 6.0 * kappa + 0.013 * (j as f64);
        if j >= 1 {
            ab[n + j] = -1.0 - 4.0 * kappa;
        }
        if j >= 2 {
            ab[j] = kappa;
        }
    }
    ab
}

fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

#[test]
fn the_factor_reconstructs_the_matrix() {
    for &kappa in &[0.0, 0.4, 7.0] {
        let n = 24;
        let ab = model_band(n, kappa);
        let a = dense(&ab, 2, n);
        let u = cholesky_banded_upper(ab, 2, n).expect("SPD by construction");
        // (U^T U)[i][j] over the band, against the original.
        for i in 0..n {
            for j in i..(i + 3).min(n) {
                let mut s = 0.0;
                for r in 0..n {
                    let uri = if i >= r && i - r <= 2 {
                        u[(2 + r - i) * n + i]
                    } else {
                        0.0
                    };
                    let urj = if j >= r && j - r <= 2 {
                        u[(2 + r - j) * n + j]
                    } else {
                        0.0
                    };
                    s += uri * urj;
                }
                let want = a[i * n + j];
                assert!(
                    (s - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "kappa={kappa} ({i},{j}): {s} vs {want}"
                );
            }
        }
    }
}

#[test]
fn the_solve_inverts_the_matrix() {
    let n = 40;
    let ab = model_band(n, 2.5);
    let a = dense(&ab, 2, n);
    let u = cholesky_banded_upper(ab, 2, n).unwrap();
    let x_true: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin() * 1e-3).collect();
    let b = matvec(&a, n, &x_true);
    let x = cho_solve_banded_upper(&u, 2, n, &b).unwrap();
    let scale = x_true.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for i in 0..n {
        assert!((x[i] - x_true[i]).abs() <= 1e-12 * scale, "row {i}");
    }
}

#[test]
fn a_numerically_zero_second_band_is_the_tridiagonal_answer() {
    // `kappa = 0` is a legal string: the biharmonic term is guarded out and the second
    // superdiagonal is all zeros, but the caller still hands over a (3, n) band. The kd = 2 path
    // must then agree with the kd = 1 path to the BIT, not to a tolerance — the arithmetic it
    // performs on the extra band is exactly `x - 0.0 * y`, which changes nothing unless the loop
    // bounds are wrong.
    let n = 31;
    let ab2 = model_band(n, 0.0);
    let mut ab1 = vec![0.0; 2 * n];
    for j in 0..n {
        ab1[n + j] = ab2[2 * n + j];
        ab1[j] = ab2[n + j];
    }
    let u2 = cholesky_banded_upper(ab2, 2, n).unwrap();
    let u1 = cholesky_banded_upper(ab1, 1, n).unwrap();
    for j in 0..n {
        assert_eq!(u2[2 * n + j], u1[n + j], "diagonal {j}");
        assert_eq!(u2[n + j], u1[j], "superdiagonal {j}");
        assert_eq!(u2[j], 0.0, "the empty band must stay empty at {j}");
    }
    let b: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.7).cos()).collect();
    assert_eq!(
        cho_solve_banded_upper(&u2, 2, n, &b).unwrap(),
        cho_solve_banded_upper(&u1, 1, n, &b).unwrap()
    );
}

#[test]
fn a_diagonal_band_factors_to_the_square_roots_exactly() {
    // kd = 0 has no off-diagonal work at all, so the factor is elementwise sqrt with nothing
    // rounded on top of it. Exact equality is the right bar and it pins the indexing.
    let n = 6;
    let ab: Vec<f64> = vec![4.0, 9.0, 16.0, 0.25, 2.0, 1e-8];
    let u = cholesky_banded_upper(ab.clone(), 0, n).unwrap();
    for j in 0..n {
        assert_eq!(u[j], ab[j].sqrt());
    }
    // And the solve is `b / sqrt(a) / sqrt(a)`, which is NOT `b / a`: at a = 2 the two spellings
    // differ in the last bit (0.49999999999999994 against 0.5). Asserted in the form the algorithm
    // actually computes, because a later "this is just a division" simplification would be a
    // different answer, and the family's anchors compare bits.
    let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let x = cho_solve_banded_upper(&u, 0, n, &b).unwrap();
    for j in 0..n {
        assert_eq!(x[j], 1.0 / ab[j].sqrt() / ab[j].sqrt());
    }
    assert_ne!(
        x[4],
        1.0 / ab[4],
        "a = 2 is the witness that the shortcut is a different number"
    );
}

#[test]
fn a_non_positive_definite_band_names_the_minor_that_failed() {
    let n = 5;
    let mut ab = model_band(n, 0.0);
    ab[2 * n + 3] = -1.0; // the 4th diagonal entry
    match cholesky_banded_upper(ab, 2, n) {
        Err(BandedError::NotPositiveDefinite(i)) => assert_eq!(i, 4, "1-based, as LAPACK reports"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_zero_diagonal_is_refused_and_so_is_a_nan() {
    let n = 4;
    let mut ab = model_band(n, 0.0);
    ab[2 * n] = 0.0;
    assert_eq!(
        cholesky_banded_upper(ab.clone(), 2, n),
        Err(BandedError::NotPositiveDefinite(1))
    );
    // `!(ajj > 0.0)` rather than `ajj <= 0.0`: a NaN diagonal is false on both comparisons, and
    // taking its square root would hand back a band of NaNs that looks like a factorization.
    ab[2 * n] = f64::NAN;
    assert_eq!(
        cholesky_banded_upper(ab, 2, n),
        Err(BandedError::NotPositiveDefinite(1))
    );
}

#[test]
fn a_shape_that_is_not_a_band_is_refused() {
    assert_eq!(
        cholesky_banded_upper(vec![], 2, 0),
        Err(BandedError::BadShape)
    );
    assert_eq!(
        cholesky_banded_upper(vec![1.0; 7], 2, 3),
        Err(BandedError::BadShape)
    );
    let u = cholesky_banded_upper(model_band(5, 0.0), 2, 5).unwrap();
    assert_eq!(
        cho_solve_banded_upper(&u, 2, 5, &[1.0; 4]),
        Err(BandedError::BadShape)
    );
}

#[test]
fn the_solve_is_deterministic_across_repeats() {
    // The family's reduction anchors are `array_equal` assertions across MODELS, so this solver
    // returning the same bits for the same inputs is not a nicety — it is the property those
    // anchors rest on once every caller shares this code.
    let n = 50;
    let u = cholesky_banded_upper(model_band(n, 1.25), 2, n).unwrap();
    let b: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.91).sin()).collect();
    let first = cho_solve_banded_upper(&u, 2, n, &b).unwrap();
    for _ in 0..8 {
        assert_eq!(cho_solve_banded_upper(&u, 2, n, &b).unwrap(), first);
    }
}
