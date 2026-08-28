//! Native bars for the sparse LU (plan §24).
//!
//! What can be asserted without Python is that the factorization inverts the matrix it was handed,
//! that it does so on shapes the elimination has to work for rather than around — an unsymmetric
//! pattern, a matrix that *needs* a row pivot, one that fills in, one that is singular — and, the
//! one that is about this project rather than about linear algebra, that the beam's own
//! pentadiagonal operator factors with **no fill at all**. That last is the claim the module's
//! "natural order is enough here" decision rests on, and it is the one a later ordering change
//! would silently break.

use physsynth_core::ops::free_beam_stiffness;
use physsynth_core::sparse::Csr;
use physsynth_core::sparse_lu::{SparseLu, SparseLuError};

/// Build a `Csr` from a dense row-major matrix, dropping exact zeros.
fn csr_from_dense(a: &[f64], n: usize) -> Csr {
    let rows: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| a[i * n + j] != 0.0)
                .map(|j| (j, a[i * n + j]))
                .collect()
        })
        .collect();
    Csr::from_rows(n, n, rows)
}

/// Largest relative residual of `A x - b`.
fn residual(a: &Csr, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.matvec(x);
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    ax.iter()
        .zip(b)
        .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
        / scale
}

/// A deterministic right-hand side; no RNG, so a failure is reproducible from the test name alone.
fn rhs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + ((i * 37) % 11) as f64 - 0.25 * ((i * 5) % 7) as f64)
        .collect()
}

#[test]
fn solves_an_unsymmetric_pattern() {
    // Deliberately not symmetric in pattern OR value: a symmetric matrix can hide a transposed
    // index, which is the mistake this whole file exists to catch.
    #[rustfmt::skip]
    let a = vec![
        4.0, 1.0, 0.0, 2.0,
        0.0, 3.0, 1.0, 0.0,
        1.0, 0.0, 5.0, 1.0,
        0.0, 2.0, 0.0, 4.0,
    ];
    let m = csr_from_dense(&a, 4);
    let lu = SparseLu::factor(&m).expect("factor");
    let b = rhs(4);
    let x = lu.solve(&b).expect("solve");
    assert!(
        residual(&m, &x, &b) < 1e-14,
        "residual {}",
        residual(&m, &x, &b)
    );
}

#[test]
fn pivots_when_the_diagonal_is_zero() {
    // A zero pivot in the natural order: without row pivoting this factorization does not exist,
    // so a pass here is the pivot search doing its job rather than the matrix being forgiving.
    #[rustfmt::skip]
    let a = vec![
        0.0, 2.0, 0.0,
        1.0, 0.0, 3.0,
        0.0, 1.0, 1.0,
    ];
    let m = csr_from_dense(&a, 3);
    let lu = SparseLu::factor(&m).expect("factor");
    let b = rhs(3);
    let x = lu.solve(&b).expect("solve");
    assert!(residual(&m, &x, &b) < 1e-14);
}

#[test]
fn fills_in_where_the_arrow_points_the_wrong_way() {
    // The classic fill case: an arrow matrix with the dense row and column FIRST fills L and U
    // completely. Natural order gets no help here, which is the point -- the elimination must be
    // right when the pattern grows, not only when it does not.
    let n = 6;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 4.0 + i as f64;
        a[i * n] = if i == 0 { a[i * n + i] } else { 1.0 };
        a[i] = if i == 0 { a[i] } else { 1.0 };
    }
    let m = csr_from_dense(&a, n);
    let lu = SparseLu::factor(&m).expect("factor");
    let (lnz, unz) = lu.nnz();
    assert!(
        lnz + unz > m.nnz(),
        "the arrow must fill: {lnz} + {unz} vs {}",
        m.nnz()
    );
    let b = rhs(n);
    let x = lu.solve(&b).expect("solve");
    assert!(residual(&m, &x, &b) < 1e-13);
}

#[test]
fn a_singular_matrix_is_refused_at_its_column() {
    // Two identical rows: column 1 has no admissible pivot once column 0 is eliminated.
    #[rustfmt::skip]
    let a = vec![
        1.0, 2.0, 0.0,
        1.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ];
    let m = csr_from_dense(&a, 3);
    match SparseLu::factor(&m) {
        Err(SparseLuError::Singular(_)) => {}
        other => panic!("expected a singular refusal, got {other:?}"),
    }
}

#[test]
fn refuses_a_non_square_matrix() {
    let m = Csr::from_rows(2, 3, vec![vec![(0, 1.0)], vec![(1, 1.0)]]);
    assert!(matches!(SparseLu::factor(&m), Err(SparseLuError::BadShape)));
}

#[test]
fn refuses_a_right_hand_side_of_the_wrong_length() {
    let m = csr_from_dense(&[2.0, 0.0, 0.0, 3.0], 2);
    let lu = SparseLu::factor(&m).expect("factor");
    assert_eq!(lu.solve(&[1.0]), Err(SparseLuError::BadRhs));
}

/// `A = (1 + sigma k) W + theta k² kappa² K` — the beam's own matrix, exactly as `beam.rs` forms
/// it. Kept here so the no-fill claim is made about the shipped operator, not about a lookalike.
fn beam_matrix(n_intervals: usize) -> Csr {
    let h = 1.0 / n_intervals as f64;
    let (k_mat, w_mat) = free_beam_stiffness(n_intervals, h);
    let (kappa, k_step, theta) = (20.0, 1.0 / 48000.0, 0.28);
    let coeff = theta * k_step * k_step * kappa * kappa;
    w_mat.sub(&k_mat.scaled(-coeff)) // W + coeff K, spelled as SciPy's `+` rounds it
}

#[test]
fn the_beam_operator_factors_with_no_fill() {
    // The module's ordering decision in one assertion: the beam's A is pentadiagonal, so natural
    // order gives L and U exactly the band and nothing else -- `n` diagonal + 2 off-diagonals per
    // column in each, i.e. 3n - 3 stored in L (no diagonal) and 3n - 3 in U... counted directly
    // rather than predicted, because the two end rows are shorter.
    for n_intervals in [8usize, 16, 32, 64, 128, 200] {
        let m = beam_matrix(n_intervals);
        let n = n_intervals + 1;
        let lu = SparseLu::factor(&m).expect("factor");
        let (lnz, unz) = lu.nnz();
        let below = (m.nnz() - n) / 2; // A is symmetric, so half its off-diagonal is the band
        assert_eq!(
            lnz, below,
            "L filled in at N = {n_intervals}: {lnz} vs {below}"
        );
        assert_eq!(
            unz,
            below + n,
            "U filled in at N = {n_intervals}: {unz} vs {}",
            below + n
        );
        assert!(
            lu.is_natural(),
            "N = {n_intervals}: fill would follow a pivot"
        );
    }
}

#[test]
fn the_beam_operator_solves_to_machine_precision() {
    for n_intervals in [8usize, 32, 64] {
        let m = beam_matrix(n_intervals);
        let lu = SparseLu::factor(&m).expect("factor");
        let b = rhs(n_intervals + 1);
        let x = lu.solve(&b).expect("solve");
        let r = residual(&m, &x, &b);
        assert!(r < 1e-13, "N = {n_intervals}: residual {r:.3e}");
    }
}

#[test]
fn no_row_pivot_fires_on_the_beam_operator() {
    // SciPy reports `perm_r == perm_c` on this matrix -- it is SPD and SuperLU takes the diagonal
    // every time. The Rust side must reach the same discrete decision, because a pivot is a
    // different elimination rather than a different last bit (§24.2).
    for n_intervals in [8usize, 32, 64, 128, 200] {
        let lu = SparseLu::factor(&beam_matrix(n_intervals)).expect("factor");
        assert!(lu.is_natural(), "a row pivot fired at N = {n_intervals}");
    }
}

#[test]
fn strict_partial_pivoting_is_what_the_threshold_is_buying() {
    // The negative half of the test above, and the reason `DIAG_PIVOT_THRESH` is not SciPy's 1.0.
    // Ordered naturally, the beam's matrix stops being diagonally largest between N = 48 and
    // N = 64; at SciPy's own default the elimination would start swapping rows and filling U.
    // Asserted rather than remembered, so that raising the threshold cannot pass silently.
    assert!(SparseLu::factor_with_thresh(&beam_matrix(48), 1.0)
        .unwrap()
        .is_natural());
    for n_intervals in [64usize, 128, 200] {
        let m = beam_matrix(n_intervals);
        let strict = SparseLu::factor_with_thresh(&m, 1.0).expect("factor");
        assert!(
            !strict.is_natural(),
            "N = {n_intervals} did not need a pivot after all"
        );
        let (_, u_strict) = strict.nnz();
        let (_, u_diag) = SparseLu::factor(&m).expect("factor").nnz();
        assert!(
            u_strict >= u_diag,
            "strict pivoting cannot reduce fill here"
        );
    }
}

#[test]
fn the_beam_diagonal_clears_the_threshold_by_five_times() {
    // How much headroom the 0.1 default actually has on the shipped operator: the diagonal stays
    // the chosen pivot for any threshold below ~0.50 at every size, so the margin is a factor of
    // five and it stops shrinking. A bar rather than a comment, because the margin is the whole
    // safety argument for preferring the diagonal.
    for n_intervals in [8usize, 32, 64, 128, 200] {
        let m = beam_matrix(n_intervals);
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let natural = SparseLu::factor_with_thresh(&m, mid)
                .map(|f| f.is_natural())
                .unwrap_or(false);
            if natural {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!(
            lo > 0.5,
            "N = {n_intervals}: diagonal only survives to threshold {lo:.4}"
        );
    }
}
