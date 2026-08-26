//! Finite-difference operators and the discrete inner product (HANDOFF Appendix A).
//!
//! The Python original is `physsynth/core/operators.py`, and as of Phase 1 this is the whole of it:
//! four pointwise differences, the inner product and its norm, and the three sparse builders
//! (`second_difference_matrix`, `biharmonic_matrix`, `free_beam_stiffness`). Phase 0 deliberately
//! shipped only the two functions `string_ideal` calls; that stub is now filled in.
//!
//! All spatial operators act on a 1-D grid array `u` of length `N + 1` sampling `u(l*h)`.
//! The discrete inner product `<f, g> = h * sum_l f[l] g[l]` is the bookkeeping device behind
//! every energy proof in this project, so it lives next to the operators it pairs with.
//!
//! # Evaluation order is part of the specification here
//!
//! Each kernel below is written out longhand in **NumPy's** evaluation order, not in whichever
//! order reads best. NumPy evaluates `a - 2*b + c` as `(a - 2*b) + c`, and in floating point that
//! is a different function from `a - (2*b - c)`. Writing the order down deliberately is what lets
//! `tests/test_rust_parity_operators.py` assert bit-identity rather than a tolerance, which is a
//! far sharper detector of a transcription slip than any physics bar (plan §2.1, as corrected).
//!
//! The one place this needed measuring rather than reasoning is `delta_xxxx`'s `h**4` — see there.

use crate::sparse::Csr;

/// Forward spatial difference `delta_x+ u[l] = (u[l+1] - u[l]) / h`.
///
/// Returns the `N` inter-node strains (length `u.len() - 1`); element `l` lives on the segment
/// between nodes `l` and `l+1`.
///
/// # Panics
/// If `u` is empty.
pub fn delta_x_forward(u: &[f64], h: f64) -> Vec<f64> {
    assert!(!u.is_empty(), "delta_x_forward needs at least one node");
    u.windows(2).map(|w| (w[1] - w[0]) / h).collect()
}

/// Backward spatial difference `delta_x- u[l] = (u[l] - u[l-1]) / h` (length `u.len() - 1`).
///
/// The same array of numbers as [`delta_x_forward`], provided for notational symmetry — the two
/// names mark which end of the segment the caller means, not a different computation.
///
/// # Panics
/// If `u` is empty.
pub fn delta_x_backward(u: &[f64], h: f64) -> Vec<f64> {
    delta_x_forward(u, h)
}

/// Second spatial difference at interior nodes.
///
/// `delta_xx u[l] = (u[l+1] - 2 u[l] + u[l-1]) / h^2` for `l = 1 .. N-1`. Returns the `N - 1`
/// interior values (length `u.len() - 2`); boundary handling is the caller's responsibility.
///
/// # Panics
/// If `u` has fewer than two nodes.
pub fn delta_xx(u: &[f64], h: f64) -> Vec<f64> {
    assert!(u.len() >= 2, "delta_xx needs at least two nodes");
    let h2 = h * h;
    u.windows(3)
        .map(|w| ((w[2] - 2.0 * w[1]) + w[0]) / h2)
        .collect()
}

/// Fourth spatial difference at interior nodes `l = 2 .. N-2` (the stiffness operator).
///
/// `delta_xxxx u[l] = (u[l+2] - 4u[l+1] + 6u[l] - 4u[l-1] + u[l-2]) / h^4`. Returns the `N - 3`
/// values where the 5-point stencil fits without a ghost node (length `u.len() - 4`); boundary
/// rows are the caller's responsibility. This is the pure-operator reference that
/// [`biharmonic_matrix`] — built as `delta_xx` squared — must reproduce in the interior.
///
/// **`powf(4.0)`, not `h*h*h*h`.** Python's `h ** 4` calls libm's `pow`, which returns the
/// correctly-rounded fourth power; three chained multiplications round three times and land
/// somewhere else. Measured over `h = 1/N` for `N = 2 .. 3999`: the two disagree in **1400** of
/// 3998 cases, and `(h*h)*(h*h)` disagrees in 1934. So the spelling is not cosmetic, and this is
/// the one kernel in the module whose exactness rests on a library call agreeing across platforms
/// rather than on IEEE-754 alone. The parity test sweeps `N` for exactly that reason.
///
/// # Panics
/// If `u` has fewer than four nodes.
pub fn delta_xxxx(u: &[f64], h: f64) -> Vec<f64> {
    assert!(u.len() >= 4, "delta_xxxx needs at least four nodes");
    let h4 = h.powf(4.0);
    u.windows(5)
        .map(|w| ((((w[4] - 4.0 * w[3]) + 6.0 * w[2]) - 4.0 * w[1]) + w[0]) / h4)
        .collect()
}

/// Discrete inner product `<f, g> = h * sum_l f[l] g[l]`.
///
/// The multiplication by `h` happens **after** the sum, matching `float(h * np.dot(f, g))` — with
/// floating-point arithmetic the order is part of the definition, not a detail. The sum itself is
/// the one thing here that cannot match NumPy exactly: `np.dot` goes through BLAS, which
/// accumulates in an order no portable loop reproduces. That is the ~1e-15 the plan's §2.1
/// correction locates precisely at reductions.
///
/// # Panics
/// If `f` and `g` have different lengths.
pub fn inner(f: &[f64], g: &[f64], h: f64) -> f64 {
    assert_eq!(f.len(), g.len(), "inner() operands must have equal length");
    let mut acc = 0.0;
    for (a, b) in f.iter().zip(g.iter()) {
        acc += a * b;
    }
    h * acc
}

/// Squared discrete norm `||f||^2 = <f, f> = h * sum_l f[l]^2` (>= 0).
pub fn norm2(f: &[f64], h: f64) -> f64 {
    inner(f, f, h)
}

/// `(N-1) x (N-1)` Dirichlet second-difference operator `delta_xx` on interior nodes.
///
/// Acts on the interior unknowns `u[1 .. N-1]` with the boundary condition `u[0] = u[N] = 0` baked
/// into the stencil (the ghost values are simply zero). Tridiagonal: `-2/h^2` on the diagonal,
/// `1/h^2` on the off-diagonals. Symmetric negative-definite; its eigenvalues are
/// `-(4/h^2) sin^2(m pi / 2N)` with eigenvectors `sin(m pi l / N)`.
///
/// # Panics
/// If `N < 2`. The Python original has no explicit guard and fails inside NumPy instead
/// (`np.full` on a negative length); the binding turns this into the same `ValueError`.
pub fn second_difference_matrix(n_intervals: usize, h: f64) -> Csr {
    assert!(
        n_intervals >= 2,
        "N must be >= 2 (an interior grid needs at least one unknown)"
    );
    let n = n_intervals - 1;
    let main = -2.0 / (h * h);
    let off = 1.0 / (h * h);
    let rows = (0..n)
        .map(|i| {
            let mut row = Vec::with_capacity(3);
            if i > 0 {
                row.push((i - 1, off));
            }
            row.push((i, main));
            if i + 1 < n {
                row.push((i + 1, off));
            }
            row
        })
        .collect();
    Csr::from_rows(n, n, rows)
}

/// `(N-1) x (N-1)` simply-supported biharmonic operator `delta_xxxx` on interior nodes.
///
/// Built as `D2 @ D2` with `D2 = second_difference_matrix(N, h)`. Doing so (rather than hand-coding
/// the 5-point rows) bakes in *both* simply-supported conditions `u = 0` and `u_xx = 0`: the second
/// is exactly `delta_xx u = 0` at the ends, which the squared operator enforces via the zero ghost
/// of the inner `delta_xx`. Two payoffs:
///
/// - the boundary-adjacent diagonal comes out `5/h^4` (not the interior `6/h^4`) automatically;
/// - the discrete biharmonic *is* `(delta_xx)^2` exactly, so the energy identity
///   `<delta_xxxx f, g> = <delta_xx f, delta_xx g>` holds to machine precision (since `D2` is
///   symmetric: `(D2 D2 f) . g = (D2 f) . (D2 g)`). This is what keeps the lossless energy
///   conserved and makes `sin(m pi l / N)` an exact eigenvector (eigenvalue `+p^4`).
///
/// Symmetric positive-definite (pentadiagonal).
///
/// # Panics
/// If `N < 2`.
pub fn biharmonic_matrix(n_intervals: usize, h: f64) -> Csr {
    let d2 = second_difference_matrix(n_intervals, h);
    d2.matmul(&d2)
}

/// Energy-first free-free Euler–Bernoulli bending operator on the `N+1` nodes.
///
/// Returns `(K, W)` — the building block for the **free** flexural resonator. Unlike the
/// simply-supported [`biharmonic_matrix`] (clamped/Dirichlet ends), **both ends are free
/// unknowns**, so the operator acts on the *full* node set:
///
/// - `K` is the symmetric **positive-semidefinite** stiffness representing the bending energy
///   `∫(u_xx)² dx ≈ uᵀ K u`. It is assembled *from the energy* as a Gram product `K = D2ᵀ Wc D2`
///   with `D2` the `(N-1)×(N+1)` **interior** second-difference operator (curvature at nodes
///   `1 .. N-1`) and `Wc = h·I` the curvature-quadrature weight, i.e. `K = h · D2ᵀ D2`. Because
///   `D2` annihilates linear data, `K`'s nullspace is **exactly the rigid-body space `{1, x}`**
///   (translation + tilt) — the free-free natural BCs `u_xx = 0` and `u_xxx = 0` are enforced by
///   construction, with no hand-coded boundary rows.
/// - `W` is the diagonal **trapezoidal** mass (lumped quadrature): `h` at interior nodes, `h/2` at
///   the two end nodes. It sits on the LHS of `W u_tt = -kappa² K u`.
///
/// **Why no special end stiffness rows are needed:** the mass-normalised operator `W⁻¹K` comes out
/// as Bilbao's energy-conserving free-free bar exactly — end row `(1/h⁴)[2,-4,2]`, next row
/// `(1/h⁴)[-2,5,-4,1]`, interior `(1/h⁴)[1,-4,6,-4,1]`. The factor of two at the very end is
/// supplied by the `h/2` mass cell, **not** by a hand-written stiffness stencil.
///
/// # Panics
/// If `N < 2` (there would be no interior curvature to build from).
pub fn free_beam_stiffness(n_intervals: usize, h: f64) -> (Csr, Csr) {
    assert!(
        n_intervals >= 2,
        "N must be >= 2 (need at least one interior curvature)."
    );
    let n_curv = n_intervals - 1;
    let inv_h2 = 1.0 / (h * h);

    // D2: interior second difference. Row r (r = 0 .. N-2) is the curvature at node l = r+1, with
    // entries [1, -2, 1]/h² at columns l-1, l, l+1.
    let d2 = Csr::from_rows(
        n_curv,
        n_intervals + 1,
        (0..n_curv)
            .map(|r| {
                let l = r + 1;
                vec![(l - 1, inv_h2), (l, -2.0 * inv_h2), (l + 1, inv_h2)]
            })
            .collect(),
    );

    let k = d2.transpose().matmul(&d2).scaled(h); // = D2ᵀ (h I) D2

    let mut w = vec![h; n_intervals + 1];
    w[0] = 0.5 * h; // trapezoidal: half cells at the two free ends
    let last = w.len() - 1;
    w[last] = 0.5 * h;
    (k, Csr::diagonal(&w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_difference_of_a_ramp_is_its_slope() {
        let h = 0.25;
        let u: Vec<f64> = (0..5).map(|i| 3.0 * (i as f64) * h).collect();
        for d in delta_x_forward(&u, h) {
            assert!((d - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn backward_difference_is_the_same_numbers() {
        let h = 0.1;
        let u = [0.0, 1.5, -2.0, 3.25, 0.5];
        assert_eq!(delta_x_backward(&u, h), delta_x_forward(&u, h));
    }

    #[test]
    fn inner_product_is_the_weighted_sum() {
        let f = [1.0, 2.0, 3.0];
        let g = [4.0, 5.0, 6.0];
        // h * (4 + 10 + 18) = 0.5 * 32
        assert_eq!(inner(&f, &g, 0.5), 16.0);
    }

    #[test]
    fn inner_product_is_symmetric() {
        let f = [0.3, -1.7, 2.5, 0.0];
        let g = [1.1, 0.25, -0.5, 4.0];
        assert_eq!(inner(&f, &g, 0.125), inner(&g, &f, 0.125));
    }

    #[test]
    fn norm2_is_inner_with_itself_and_never_negative() {
        let f = [0.3, -1.7, 2.5, 0.0];
        assert_eq!(norm2(&f, 0.01), inner(&f, &f, 0.01));
        assert!(norm2(&f, 0.01) >= 0.0);
    }
}
