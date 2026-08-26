//! Finite-difference operators and the discrete inner product (HANDOFF Appendix A).
//!
//! The Python original is `physsynth/core/operators.py`. **Only what Phase 0 needs is here**:
//! `string_ideal` calls exactly two of its functions, so exactly two are ported. Phase 1 ports the
//! module properly — the remaining pointwise differences and the sparse matrix builders
//! (`second_difference_matrix`, `biharmonic_matrix`, `free_beam_stiffness`) — and binds the lot.
//! Resisting the urge to transcribe the whole file now is the point: an unported operator is
//! visibly missing, whereas an untested ported one looks finished.
//!
//! All spatial operators act on a 1-D grid array `u` of length `N + 1` sampling `u(l*h)`.
//! The discrete inner product `<f, g> = h * sum_l f[l] g[l]` is the bookkeeping device behind
//! every energy proof in this project, so it lives next to the operators it pairs with.

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

/// Discrete inner product `<f, g> = h * sum_l f[l] g[l]`.
///
/// The multiplication by `h` happens **after** the sum, matching `float(h * np.dot(f, g))` — with
/// floating-point arithmetic the order is part of the definition, not a detail.
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
}
