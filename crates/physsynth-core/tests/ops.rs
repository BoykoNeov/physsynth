//! Native physics bars for the difference operators (HANDOFF Appendix A, plan Phase 1).
//!
//! These are the *operator* half of the project's acceptance contract. They do not compare against
//! Python — `tests/test_rust_parity_operators.py` does that, and the plan is explicit that the
//! comparison is a diagnostic while these are the bar. Every assertion below is against closed-form
//! discrete algebra, so each one stays meaningful after the Python side is deleted.
//!
//! The three that matter most, because they are the properties the downstream models' energy
//! proofs actually rest on:
//!
//! * `sin(m pi l / N)` is an **exact** eigenvector of `delta_xx`, with the exact discrete
//!   eigenvalue `-(4/h^2) sin^2(m pi / 2N)`. No continuum approximation anywhere in that statement.
//! * `<B f, g> == <D2 f, D2 g>` to machine precision, because `B` *is* `D2 @ D2` and `D2` is
//!   symmetric. That identity is what conserves energy in the stiff string and the plate.
//! * `free_beam_stiffness`'s nullspace is exactly the rigid-body space `{1, x}` — which is how the
//!   free-free boundary conditions get enforced without a single hand-written boundary row.

use physsynth_core::ops::{
    biharmonic_matrix, delta_x_backward, delta_x_forward, delta_xx, delta_xxxx,
    free_beam_stiffness, inner, norm2, second_difference_matrix,
};

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// The grid `x_l = l*h` on `[0, L]` with `N` intervals.
fn grid(n: usize, l_len: f64) -> (Vec<f64>, f64) {
    let h = l_len / n as f64;
    ((0..=n).map(|l| l as f64 * h).collect(), h)
}

// -- pointwise differences -----------------------------------------------------------------------

#[test]
fn forward_difference_is_exact_on_a_linear_ramp() {
    let (x, h) = grid(8, 2.0);
    let u: Vec<f64> = x.iter().map(|xi| 3.0 * xi + 1.0).collect();
    let d = delta_x_forward(&u, h);
    assert_eq!(d.len(), u.len() - 1);
    for v in &d {
        assert!((v - 3.0).abs() < 1e-12, "slope came out {v}");
    }
    // The backward difference is the same numbers by construction; assert it rather than trust it,
    // because the two names exist to be used interchangeably in the energy proofs.
    assert_eq!(delta_x_backward(&u, h), d);
}

#[test]
fn second_difference_is_exact_on_a_quadratic() {
    let (x, h) = grid(6, 0.6);
    let u: Vec<f64> = x.iter().map(|xi| 2.0 * xi * xi - xi + 5.0).collect();
    for v in delta_xx(&u, h) {
        assert!((v - 4.0).abs() < 1e-9, "curvature came out {v}, want 4");
    }
}

#[test]
fn second_difference_has_the_exact_discrete_eigenvector() {
    // sin(m pi x / L) is an eigenvector of delta_xx with eigenvalue -(4/h^2) sin^2(m pi / 2N).
    // This pins the operator down with no continuum approximation in the statement at all.
    let n = 64;
    let l_len = 1.0;
    let (x, h) = grid(n, l_len);
    let m = 5.0;
    let v: Vec<f64> = x
        .iter()
        .map(|xi| (m * std::f64::consts::PI * xi / l_len).sin())
        .collect();
    let lhs = delta_xx(&v, h);
    let eig = -(4.0 / (h * h)) * (m * std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
    for (i, got) in lhs.iter().enumerate() {
        let want = eig * v[i + 1];
        assert!((got - want).abs() < 1e-9, "node {}: {got} vs {want}", i + 1);
    }
}

#[test]
fn fourth_difference_is_exact_on_a_quartic() {
    // The 5-point fourth difference is exact for polynomials of degree <= 5, so d^4/dx^4 of x^4 is
    // 24 on the nose. h is kept coarse on purpose: the operator divides by h^4, so a fine grid
    // makes this test a measurement of rounding rather than of the stencil.
    let (x, h) = grid(10, 10.0); // h = 1
    let u: Vec<f64> = x.iter().map(|xi| xi.powi(4)).collect();
    let d = delta_xxxx(&u, h);
    assert_eq!(d.len(), u.len() - 4);
    for v in &d {
        assert!((v - 24.0).abs() < 1e-8, "fourth difference came out {v}");
    }
}

#[test]
fn the_inner_product_is_the_energy_bookkeeping_it_claims_to_be() {
    let f = [1.0, -2.0, 3.0, 0.5];
    let g = [0.5, 1.0, -1.0, 2.0];
    let h = 0.01;
    assert!((inner(&f, &g, h) - inner(&g, &f, h)).abs() == 0.0);
    assert_eq!(norm2(&f, h), inner(&f, &f, h));
    assert!(norm2(&f, h) >= 0.0);
    assert_eq!(norm2(&[0.0; 5], h), 0.0);

    // And the VALUE, against the definition `h sum f_i g_i` -- carried from
    // `tests/test_operators.py` (retirement plan §43). Everything above is a property any scaling
    // of the sum also has: an inner product that dropped `h`, or used `h^2`, is still symmetric,
    // still non-negative on the diagonal, and still zero on zero. The sums are exact here (-3.5 and
    // 14.25), so the only rounding is the one multiply by `h`, and the recorded NumPy figures are
    // reproduced exactly.
    assert_eq!(inner(&f, &g, h), h * -3.5);
    assert_eq!(norm2(&f, h), h * 14.25);
    assert_eq!(inner(&f, &g, h), -0.035);
    assert_eq!(norm2(&f, h), 0.142_500_000_000_000_02);
}

// -- the assembled operators ---------------------------------------------------------------------

#[test]
fn the_second_difference_matrix_is_the_pointwise_operator_with_dirichlet_ghosts() {
    let n = 12;
    let (x, h) = grid(n, 1.0);
    let d2 = second_difference_matrix(n, h);
    assert_eq!((d2.nrows(), d2.ncols()), (n - 1, n - 1));
    assert!(d2.is_symmetric());

    // On a field that already vanishes at both ends, the matrix and the stencil must agree
    // exactly — the matrix's only extra content is the zero ghost, which such a field supplies.
    let v: Vec<f64> = x
        .iter()
        .map(|xi| (3.0 * std::f64::consts::PI * xi).sin())
        .collect();
    let interior: Vec<f64> = v[1..n].to_vec();
    let by_matrix = d2.matvec(&interior);
    let by_stencil = delta_xx(&v, h);
    for (a, b) in by_matrix.iter().zip(by_stencil.iter()) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn the_second_difference_matrix_has_the_exact_discrete_eigenpair() {
    let n = 48;
    let (x, h) = grid(n, 1.0);
    let d2 = second_difference_matrix(n, h);
    for m in [1.0, 7.0, 23.0] {
        let v: Vec<f64> = x[1..n]
            .iter()
            .map(|xi| (m * std::f64::consts::PI * xi).sin())
            .collect();
        let eig = -(4.0 / (h * h)) * (m * std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
        let got = d2.matvec(&v);
        let scale = eig.abs() * max_abs(&v);
        for (i, gi) in got.iter().enumerate() {
            assert!(
                (gi - eig * v[i]).abs() < 1e-10 * scale,
                "m={m}, row {i}: {gi} vs {}",
                eig * v[i]
            );
        }
    }
}

#[test]
fn the_biharmonic_is_the_second_difference_squared_and_conserves_the_energy_identity() {
    // <B f, g> == <D2 f, D2 g> to machine precision. This is not a nicety: it is the discrete
    // integration-by-parts that keeps the stiff string's lossless energy flat, and it holds
    // because B is literally D2 @ D2 with D2 symmetric.
    let n = 24;
    let h = 1.0 / n as f64;
    let d2 = second_difference_matrix(n, h);
    let b = biharmonic_matrix(n, h);
    assert_eq!((b.nrows(), b.ncols()), (n - 1, n - 1));
    assert!(b.is_symmetric());

    let f: Vec<f64> = (0..n - 1).map(|i| (0.3 * i as f64).sin()).collect();
    let g: Vec<f64> = (0..n - 1).map(|i| (0.17 * i as f64 + 1.0).cos()).collect();
    let lhs = inner(&b.matvec(&f), &g, h);
    let rhs = inner(&d2.matvec(&f), &d2.matvec(&g), h);
    assert!(
        (lhs - rhs).abs() <= 1e-12 * lhs.abs().max(rhs.abs()),
        "energy identity broken: {lhs} vs {rhs}"
    );
}

#[test]
fn the_biharmonic_carries_the_simply_supported_boundary_in_its_diagonal() {
    // The payoff of building B as D2 squared rather than hand-coding 5-point rows: the
    // boundary-adjacent diagonal is 5/h^4, not the interior 6/h^4, and nobody wrote that down.
    let n = 16;
    let h = 1.0 / n as f64;
    let b = biharmonic_matrix(n, h);
    let h4 = h.powf(4.0);
    let last = n - 2;
    assert!((b.get(0, 0) - 5.0 / h4).abs() < 1e-6 * (5.0 / h4));
    assert!((b.get(last, last) - 5.0 / h4).abs() < 1e-6 * (5.0 / h4));
    for i in 1..last {
        assert!(
            (b.get(i, i) - 6.0 / h4).abs() < 1e-6 * (6.0 / h4),
            "row {i} diagonal is {}, want 6/h^4",
            b.get(i, i)
        );
    }
    // And the interior rows are the plain 5-point stencil, i.e. delta_xxxx's own coefficients.
    for (offset, want) in [(2usize, 1.0), (1, -4.0)] {
        let got = b.get(n / 2, n / 2 - offset);
        assert!((got - want / h4).abs() < 1e-6 * (want / h4).abs(), "{got}");
    }
}

#[test]
fn the_biharmonic_eigenvalue_is_the_second_difference_eigenvalue_squared() {
    let n = 40;
    let (x, h) = grid(n, 1.0);
    let b = biharmonic_matrix(n, h);
    let m = 6.0;
    let v: Vec<f64> = x[1..n]
        .iter()
        .map(|xi| (m * std::f64::consts::PI * xi).sin())
        .collect();
    let mu = -(4.0 / (h * h)) * (m * std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
    let eig = mu * mu; // positive, as befits a bending operator
    let got = b.matvec(&v);
    let scale = eig * max_abs(&v);
    for (i, gi) in got.iter().enumerate() {
        assert!((gi - eig * v[i]).abs() < 1e-9 * scale, "row {i}: {gi}");
    }
}

// -- carried from `tests/test_stiff_string.py` (retirement plan §32) -------------------------------

/// `np.allclose(a, b, rtol, atol)`, both terms written down: `|a - b| <= atol + rtol |b|` per
/// element. The Python passed `rtol` at most, so every carried call keeps NumPy's hidden
/// `atol = 1e-8`, and a call that passed neither keeps `rtol = 1e-5` too (§16's (d)).
fn allclose(a: &[f64], b: &[f64], rtol: f64, atol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}

#[test]
fn the_biharmonic_rows_and_eigenvectors_at_the_stiff_string_fixture() {
    // Carried from `test_biharmonic_matrix_structure`, at its own N = 12, h = 0.1. The whole
    // boundary row is pinned -- `[5, -4, 1, 0, 0] / h^4`, the simply-supported ghost reflection --
    // where the bars above check only its diagonal; and the eigenvector claim at three modes.
    let (n, h) = (12usize, 0.1);
    let b = biharmonic_matrix(n, h);
    let h4 = h.powi(4);
    let row = |i: usize, len: usize| -> Vec<f64> { (0..len).map(|j| b.get(i, j) * h4).collect() };
    assert!(
        allclose(&row(0, 5), &[5.0, -4.0, 1.0, 0.0, 0.0], 1e-5, 1e-8),
        "boundary row {:?}",
        row(0, 5)
    );
    assert!(
        allclose(&row(2, 6), &[1.0, -4.0, 6.0, -4.0, 1.0, 0.0], 1e-5, 1e-8),
        "interior row {:?}",
        row(2, 6)
    );
    assert!(b.is_symmetric());
    for m in [1.0, 3.0, 5.0] {
        let phi: Vec<f64> = (1..n)
            .map(|l| (m * std::f64::consts::PI * l as f64 / n as f64).sin())
            .collect();
        let p4 =
            ((4.0 / (h * h)) * (m * std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2)).powi(2);
        let want: Vec<f64> = phi.iter().map(|v| p4 * v).collect();
        assert!(allclose(&b.matvec(&phi), &want, 1e-9, 1e-8), "mode {m}");
    }
}

#[test]
fn the_biharmonic_matrix_is_the_pure_stencil_away_from_the_boundary() {
    // Carried from `test_biharmonic_matrix_matches_pure_operator_in_interior` (N = 16, h = 0.05).
    // The Python drew the field from NumPy's seeded PCG64; the claim is about a structureless
    // field, not that stream, so any deterministic hash will do (§15).
    let (n, h) = (16usize, 0.05);
    let b = biharmonic_matrix(n, h);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut u: Vec<f64> = (0..=n)
        .map(|_| {
            // splitmix64, mapped to [-1, 1).
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect();
    u[0] = 0.0;
    u[n] = 0.0;
    let full = b.matvec(&u[1..n]); // interior nodes 1 .. N-1
    let stencil = delta_xxxx(&u, h); // nodes 2 .. N-2
    assert_eq!(stencil.len(), full.len() - 2);
    assert!(
        allclose(&full[1..full.len() - 1], &stencil, 1e-10, 1e-8),
        "matrix {:?} vs stencil {stencil:?}",
        &full[1..full.len() - 1]
    );
}

#[test]
fn the_second_difference_eigenpair_at_the_stiff_string_fixture() {
    // Carried from `test_second_difference_matrix_eigenvalues`, at its own N = 20, h = 0.05 and
    // modes {1, 4, 9}, with its per-element bound rather than the max-norm one above.
    let (n, h) = (20usize, 0.05);
    let d2 = second_difference_matrix(n, h);
    for m in [1.0, 4.0, 9.0] {
        let phi: Vec<f64> = (1..n)
            .map(|l| (m * std::f64::consts::PI * l as f64 / n as f64).sin())
            .collect();
        let p2 = (4.0 / (h * h)) * (m * std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
        let want: Vec<f64> = phi.iter().map(|v| -p2 * v).collect();
        assert!(allclose(&d2.matvec(&phi), &want, 1e-10, 1e-8), "mode {m}");
    }
}

// -- the free-free beam operator ------------------------------------------------------------------

#[test]
fn the_free_beam_mass_is_trapezoidal() {
    let n = 9;
    let h = 1.3 / n as f64;
    let (_, w) = free_beam_stiffness(n, h);
    assert_eq!((w.nrows(), w.ncols()), (n + 1, n + 1));
    assert_eq!(w.get(0, 0), 0.5 * h);
    assert_eq!(w.get(n, n), 0.5 * h);
    for i in 1..n {
        assert_eq!(w.get(i, i), h, "interior node {i} should carry a full cell");
    }
    assert_eq!(w.nnz(), n + 1, "the mass must be diagonal");
}

#[test]
fn the_free_beam_stiffness_annihilates_exactly_the_rigid_body_space() {
    // K's nullspace is {1, x} — translation and tilt. That IS the free-free boundary condition;
    // there are no hand-written boundary rows in the builder, and this is the test that says the
    // Gram construction supplied them.
    let n = 20;
    let (x, h) = grid(n, 1.0);
    let (k, _) = free_beam_stiffness(n, h);
    assert!(k.is_symmetric());

    // Neither rigid-body mode is annihilated *exactly*, and the reason is worth writing down
    // because the first draft of this test asserted that it was. `D2 @ 1` is exactly zero —
    // `(a - 2a) + a` cancels in IEEE-754 — but `K @ 1` is not the same computation: K's entries
    // are sums fixed at assembly time, so applying it sums a row of already-rounded numbers rather
    // than re-deriving the cancellation. Measured at N = 20: 8.2e-12 against an operator whose
    // scale is h * (1/h^4) = 1/h^3 = 8000, i.e. ~1e-15 relative. The claim the builder actually
    // supports is that the rigid-body space is annihilated to rounding, and the bar below says so
    // relative to what the same operator does to a field that genuinely bends.
    let quadratic: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
    let bend = k.matvec(&quadratic);
    assert!(max_abs(&bend) > 0.0, "a quadratic must actually bend");

    for (name, field) in [("translation", vec![1.0; n + 1]), ("tilt", x.clone())] {
        let residual = max_abs(&k.matvec(&field));
        assert!(
            residual < 1e-10 * max_abs(&bend),
            "{name} is not in the nullspace: residual {residual} against a bending response of {}",
            max_abs(&bend)
        );
    }
}

#[test]
fn the_mass_normalised_free_beam_is_bilbaos_energy_conserving_bar() {
    // W^-1 K must come out as end row (1/h^4)[2,-4,2], next row (1/h^4)[-2,5,-4,1], interior
    // (1/h^4)[1,-4,6,-4,1]. The factor of two at the very end comes from the h/2 mass cell, not
    // from a stiffness stencil — which is the whole reason the free edge needed no special-casing.
    let n = 12;
    let h = 1.0 / n as f64;
    let (k, w) = free_beam_stiffness(n, h);
    let h4 = h.powf(4.0);
    let row = |i: usize, js: &[usize]| -> Vec<f64> {
        js.iter().map(|&j| k.get(i, j) / w.get(i, i) * h4).collect()
    };
    let close = |got: &[f64], want: &[f64]| {
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    };
    close(&row(0, &[0, 1, 2]), &[2.0, -4.0, 2.0]);
    close(&row(1, &[0, 1, 2, 3]), &[-2.0, 5.0, -4.0, 1.0]);
    close(&row(6, &[4, 5, 6, 7, 8]), &[1.0, -4.0, 6.0, -4.0, 1.0]);
    close(&row(n, &[n, n - 1, n - 2]), &[2.0, -4.0, 2.0]);
    close(
        &row(n - 1, &[n, n - 1, n - 2, n - 3]),
        &[-2.0, 5.0, -4.0, 1.0],
    );
}

#[test]
fn the_free_beam_bending_energy_is_the_curvature_norm() {
    // u^T K u == h * sum(curvature^2) — the identity the operator was assembled from. If this
    // drifts, the beam's energy report is measuring something other than its own stiffness.
    let n = 18;
    let (x, h) = grid(n, 1.0);
    let (k, _) = free_beam_stiffness(n, h);
    let u: Vec<f64> = x
        .iter()
        .map(|xi| (2.0 * xi).cos() + 0.3 * xi * xi)
        .collect();
    let quad = inner(&k.matvec(&u), &u, 1.0); // <K u, u> with no extra weight
    let curvature = delta_xx(&u, h);
    let want = h * curvature.iter().map(|c| c * c).sum::<f64>();
    assert!(
        (quad - want).abs() <= 1e-10 * want.abs(),
        "bending energy {quad} vs curvature norm {want}"
    );
}

// -- carried from `tests/test_binding_surface.py` (retirement plan §48) ---------------------------

fn usizes(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_u64().expect("an index") as usize)
        .collect()
}

fn doubles(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

#[test]
fn the_biharmonic_is_scipys_own_product_of_the_second_difference() {
    // The bars above hold `B` to `D2 @ D2` at 1e-12 and to its stencil at 1e-6; this holds it to
    // the BIT, against the library the deleted Python body called. `B`'s entries are genuine
    // three-term sums, so the association is a claim, and SciPy's SMMP kernel is the referee:
    // `(D2 @ D2).tocsr()` with `sum_duplicates` and `sort_indices`, recorded from the crate's own
    // `D2` at eight sizes and two lengths (a tidy `h = 1/N`, and a 0.65 m string where `1/(h*h)`
    // rounds differently) into `reference/scipy_biharmonic.json` before the binding went. The
    // recorded `D2` is checked first, so a moved `D2` and a moved product are told apart.
    let rec: serde_json::Value =
        serde_json::from_str(include_str!("reference/scipy_biharmonic.json")).expect("the record");
    let cases = rec["cases"].as_array().expect("the cases");
    assert_eq!(cases.len(), 16, "the record changed shape");
    for case in cases {
        let n = case["n"].as_u64().expect("N") as usize;
        let length = case["length"].as_f64().expect("L");
        let h = length / n as f64;
        assert_eq!(
            h,
            case["h"].as_f64().expect("h"),
            "N = {n}, L = {length}: h"
        );

        let d2 = second_difference_matrix(n, h);
        assert_eq!(
            d2.indptr(),
            usizes(&case["d2_indptr"]),
            "N = {n}, L = {length}: D2"
        );
        assert_eq!(
            d2.indices(),
            usizes(&case["d2_indices"]),
            "N = {n}, L = {length}: D2"
        );
        assert_eq!(
            d2.data(),
            doubles(&case["d2_data"]),
            "N = {n}, L = {length}: D2's values"
        );

        let b = biharmonic_matrix(n, h);
        let shape = usizes(&case["shape"]);
        assert_eq!(
            (b.nrows(), b.ncols()),
            (shape[0], shape[1]),
            "N = {n}, L = {length}"
        );
        assert_eq!(
            b.indptr(),
            usizes(&case["indptr"]),
            "N = {n}, L = {length}: B's rows"
        );
        assert_eq!(
            b.indices(),
            usizes(&case["indices"]),
            "N = {n}, L = {length}: B's columns"
        );
        let want = doubles(&case["data"]);
        let worst = b
            .data()
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (a, w)| m.max((a - w).abs() / w.abs()));
        assert_eq!(
            b.data(),
            want,
            "B(N = {n}, L = {length}) is not SciPy's D2 @ D2; worst relative gap {worst:.3e}"
        );
    }
}
