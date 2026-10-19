//! Native bars for `eig::general_eigenvalues` — the unsymmetric Hessenberg-QR routine
//! `coupled_tuning` needs for a lossy one-step map (string mapping plan §16).
//!
//! As for the symmetric routine, the bars are spectra known exactly, not another implementation:
//! a triangular matrix, a rotation-scaling, a known spectrum hidden by a similarity transform, a
//! polynomial's companion matrix, and the ideal string's leapfrog map, whose roots sit on the unit
//! circle at the scheme's own dispersion relation. Plus the trace, which every similarity keeps.

use std::f64::consts::PI;

use physsynth_core::dense::{lu_factor, lu_solve};
use physsynth_core::eig::{general_eigenvalues, EigError};

type C = (f64, f64);

/// Match every expected root to a distinct computed one (greedy nearest) and return the worst
/// distance, relative to `scale`.
fn worst_match(expected: &[C], got: &[C], scale: f64) -> f64 {
    assert_eq!(expected.len(), got.len(), "root counts differ");
    let mut used = vec![false; got.len()];
    let mut worst = 0.0f64;
    for e in expected {
        let (mut best, mut at) = (f64::INFINITY, 0);
        for (i, g) in got.iter().enumerate() {
            let d = (e.0 - g.0).hypot(e.1 - g.1);
            if !used[i] && d < best {
                best = d;
                at = i;
            }
        }
        used[at] = true;
        worst = worst.max(best / scale);
    }
    worst
}

/// A deterministic well-conditioned matrix: the identity plus a small scrambled part.
fn mixer(n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let x = ((i * 7 + j * 13 + 3) % 17) as f64 / 17.0 - 0.5;
            s[i * n + j] = if i == j { 1.0 } else { 0.0 } + 0.4 * x;
        }
    }
    s
}

/// `S D S^-1`, row-major, with `S^-1` from the project's dense LU, column by column.
fn similar(d: &[f64], s: &[f64], n: usize) -> Vec<f64> {
    let lu = lu_factor(s.to_vec(), n).expect("square");
    let mut s_inv = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = lu_solve(&lu, &e).expect("solves");
        for i in 0..n {
            s_inv[i * n + j] = col[i];
        }
    }
    let mul = |a: &[f64], b: &[f64]| {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
        c
    };
    mul(&mul(s, d), &s_inv)
}

#[test]
fn a_ragged_matrix_is_refused_and_an_empty_one_has_no_eigenvalues() {
    assert_eq!(
        general_eigenvalues(&[1.0, 2.0, 3.0], 2),
        Err(EigError::BadShape)
    );
    assert_eq!(general_eigenvalues(&[], 0), Ok(Vec::new()));
}

#[test]
fn a_triangular_matrix_returns_its_diagonal() {
    let n = 5;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            a[i * n + j] = if i == j {
                i as f64 - 1.5
            } else {
                0.3 * (i + j) as f64
            };
        }
    }
    let got = general_eigenvalues(&a, n).unwrap();
    let expected: Vec<C> = (0..n).map(|i| (i as f64 - 1.5, 0.0)).collect();
    assert!(worst_match(&expected, &got, 1.0) < 1e-14);
}

#[test]
fn a_rotation_scaling_is_a_complex_pair() {
    let (re, im) = (0.6, 0.7);
    let got = general_eigenvalues(&[re, -im, im, re], 2).unwrap();
    assert!(worst_match(&[(re, im), (re, -im)], &got, 1.0) < 1e-15);
}

/// Damped roots `r e^{+-i theta}` and real roots, hidden behind a dense similarity: the shape of a
/// lossy one-step map's spectrum.
#[test]
fn a_known_spectrum_survives_a_similarity_transform() {
    let pairs: [(f64, f64); 5] = [
        (0.999, 0.03),
        (0.98, 0.4),
        (0.9, 1.3),
        (0.5, 2.8),
        (0.995, 0.031),
    ];
    let reals = [0.3, -0.7];
    let n = 2 * pairs.len() + reals.len();
    let mut d = vec![0.0; n * n];
    let mut expected = Vec::new();
    for (b, &(r, th)) in pairs.iter().enumerate() {
        let (c, s) = (r * th.cos(), r * th.sin());
        let i = 2 * b;
        d[i * n + i] = c;
        d[i * n + i + 1] = -s;
        d[(i + 1) * n + i] = s;
        d[(i + 1) * n + i + 1] = c;
        expected.push((c, s));
        expected.push((c, -s));
    }
    for (o, &v) in reals.iter().enumerate() {
        let i = 2 * pairs.len() + o;
        d[i * n + i] = v;
        expected.push((v, 0.0));
    }
    let a = similar(&d, &mixer(n), n);
    let got = general_eigenvalues(&a, n).unwrap();
    let worst = worst_match(&expected, &got, 1.0);
    assert!(worst < 1e-12, "worst root error {worst:.2e}");
}

/// `p(z) = (z - 0.5)(z + 2)(z - 3)(z^2 - 2z + 5)`: roots 0.5, -2, 3, 1 +- 2i.
#[test]
fn a_companion_matrix_returns_its_polynomials_roots() {
    // Coefficients of z^5 + c4 z^4 + ... + c0, highest first after the leading 1.
    let roots: [C; 5] = [(0.5, 0.0), (-2.0, 0.0), (3.0, 0.0), (1.0, 2.0), (1.0, -2.0)];
    let mut poly: Vec<C> = vec![(1.0, 0.0)];
    for r in roots {
        let mut next = vec![(0.0, 0.0); poly.len() + 1];
        for (i, &c) in poly.iter().enumerate() {
            next[i].0 += c.0;
            next[i].1 += c.1;
            // - r * c, into the next lower power
            next[i + 1].0 -= r.0 * c.0 - r.1 * c.1;
            next[i + 1].1 -= r.0 * c.1 + r.1 * c.0;
        }
        poly = next;
    }
    let n = 5;
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        a[j] = -poly[j + 1].0;
    }
    for i in 1..n {
        a[i * n + i - 1] = 1.0;
    }
    let got = general_eigenvalues(&a, n).unwrap();
    assert!(worst_match(&roots, &got, 3.0) < 1e-13);
}

/// The ideal string's leapfrog, fixed at both ends, as a one-step map on `[u^n, u^{n-1}]`: its
/// roots are `e^{+-i phi_m}` with `cos(phi_m) = 1 - 2 lambda^2 sin^2(m pi / 2N)`, all on the unit
/// circle (a lossless scheme). With a loss `sigma` every root moves to radius
/// `sqrt((1 - sigma k)/(1 + sigma k))`.
#[test]
fn the_ideal_strings_leapfrog_map_has_the_schemes_own_roots() {
    let (nseg, lam) = (24usize, 0.9);
    let d = nseg - 1; // interior nodes
    for sk in [0.0, 0.01] {
        let n = 2 * d;
        let mut a = vec![0.0; n * n];
        for i in 0..d {
            // u^{n+1}_i = (2 u_i - (1 - sk) u'_i + lam^2 (u_{i+1} - 2 u_i + u_{i-1})) / (1 + sk)
            let inv = 1.0 / (1.0 + sk);
            a[i * n + i] = (2.0 - 2.0 * lam * lam) * inv;
            if i > 0 {
                a[i * n + i - 1] = lam * lam * inv;
            }
            if i + 1 < d {
                a[i * n + i + 1] = lam * lam * inv;
            }
            a[i * n + d + i] = -(1.0 - sk) * inv;
            a[(d + i) * n + i] = 1.0;
        }
        let got = general_eigenvalues(&a, n).unwrap();
        let radius = ((1.0 - sk) / (1.0 + sk)).sqrt();
        let mut expected = Vec::new();
        for m in 1..=d {
            let s = (m as f64 * PI / (2.0 * nseg as f64)).sin();
            // The root pair of (1 + sk) z^2 - 2 (1 - 2 lam^2 s^2) z + (1 - sk) = 0: radius as above,
            // cos(phi) = (1 - 2 lam^2 s^2) / ((1 + sk) radius).
            let cos_phi = (1.0 - 2.0 * lam * lam * s * s) / ((1.0 + sk) * radius);
            let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
            expected.push((radius * cos_phi, radius * sin_phi));
            expected.push((radius * cos_phi, -radius * sin_phi));
        }
        let worst = worst_match(&expected, &got, 1.0);
        assert!(worst < 1e-13, "sk {sk}: worst root error {worst:.2e}");
    }
}

#[test]
fn the_roots_sum_to_the_trace() {
    let n = 12;
    let s = mixer(n);
    let mut a = similar(&s, &mixer(n), n);
    a[5] += 0.7;
    let got = general_eigenvalues(&a, n).unwrap();
    let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
    let sum: f64 = got.iter().map(|z| z.0).sum();
    let imag: f64 = got.iter().map(|z| z.1).sum();
    assert!(
        (sum - trace).abs() < 1e-12 * trace.abs().max(1.0),
        "{sum} vs {trace}"
    );
    assert!(imag.abs() < 1e-12, "conjugates do not pair: {imag}");
}
