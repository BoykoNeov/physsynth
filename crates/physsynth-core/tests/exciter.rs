//! Native validation of the excitation shapes.
//!
//! These have no energy ledger of their own — they are functions of a grid, and what can be wrong
//! with them is shape, support and endpoint behaviour. So that is what is asserted: the pluck is a
//! tent that vanishes at both ends and peaks where it was asked to, the raised cosines are exactly
//! zero outside their support and exactly `amplitude` at their centre, and the 2-D bump is
//! radially symmetric.
//!
//! The one behaviour worth stating out loud is the 1-D raised cosine's **end clamp**: a hump
//! placed within `width` of a boundary would otherwise leave a non-zero value sitting on a node
//! the resonator holds at zero, which is a silently inconsistent initial condition — the energy
//! would be computed from a displacement the first step then discards.

use physsynth_core::exciter::{raised_cosine, raised_cosine_2d, triangular_pluck, ExciterError};
use std::f64::consts::PI;

fn grid(n: usize, l: f64) -> Vec<f64> {
    let step = l / n as f64;
    let mut x: Vec<f64> = (0..=n).map(|i| i as f64 * step).collect();
    x[n] = l;
    x
}

#[test]
fn the_pluck_is_a_tent_that_vanishes_at_both_ends() {
    let x = grid(20, 1.0);
    let u = triangular_pluck(&x, 1.0, 0.25, 2.0).expect("a legal pluck");

    assert_eq!(u[0], 0.0);
    assert_eq!(u[20], 0.0);
    assert_eq!(u[5], 2.0, "the peak sits on the node at x = 0.25");
    // Monotone up to the peak, monotone down after it.
    for i in 1..=5 {
        assert!(u[i] > u[i - 1]);
    }
    for i in 6..=20 {
        assert!(u[i] < u[i - 1]);
    }
}

#[test]
fn the_pluck_puts_its_peak_between_nodes_when_asked_to() {
    // A pluck point that is not on a node still produces the exact tent sampled at the nodes: the
    // maximum is at whichever node is nearest, and both legs are still linear.
    let x = grid(10, 1.0);
    let u = triangular_pluck(&x, 1.0, 0.33, 1.0).expect("a legal pluck");
    let (peak_idx, _) = u
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |acc, (i, &v)| {
            if v > acc.1 {
                (i, v)
            } else {
                acc
            }
        });
    assert_eq!(peak_idx, 3);
    let slope = u[1] - u[0];
    for i in 2..=3 {
        assert!(
            (u[i] - u[i - 1] - slope).abs() < 1e-15,
            "the rise must be linear"
        );
    }
}

#[test]
fn a_pluck_outside_the_string_is_rejected() {
    let x = grid(10, 1.0);
    for bad in [0.0, 1.0, -0.2, 1.5] {
        let err = triangular_pluck(&x, 1.0, bad, 1.0).unwrap_err();
        assert_eq!(err, ExciterError::PluckOutside(1.0, bad));
        assert!(err.to_string().starts_with("pluck position must satisfy"));
    }
}

#[test]
fn the_raised_cosine_is_zero_outside_its_support() {
    let x = grid(100, 1.0);
    let u = raised_cosine(&x, 1.0, 0.5, 0.2, 1.5).expect("a legal hump");

    for (i, (&xi, &ui)) in x.iter().zip(u.iter()).enumerate() {
        if (xi - 0.5).abs() >= 0.2 {
            assert_eq!(ui, 0.0, "node {i} at x = {xi} is outside the support");
        }
    }
    assert_eq!(u[50], 1.5, "the centre carries the full amplitude");
}

#[test]
fn the_raised_cosine_clamps_the_ends_even_when_the_hump_reaches_them() {
    // Centred on the left end with a wide support: without the clamp, u[0] would be the full
    // amplitude sitting on a node the string holds at zero.
    let x = grid(40, 1.0);
    let u = raised_cosine(&x, 1.0, 0.0, 0.3, 1.0).expect("a legal hump");
    assert_eq!(u[0], 0.0);
    assert_eq!(u[40], 0.0);
    assert!(u[1] > 0.0, "the rest of the hump survives the clamp");
}

#[test]
fn the_raised_cosine_is_the_shape_it_claims_to_be() {
    let x = grid(64, 2.0);
    let u = raised_cosine(&x, 2.0, 1.0, 0.5, 1.0).expect("a legal hump");
    for (&xi, &ui) in x.iter().zip(u.iter()).skip(1).take(63) {
        let d = (xi - 1.0).abs();
        let want = if d < 0.5 {
            0.5 * (1.0 + (PI * d / 0.5).cos())
        } else {
            0.0
        };
        assert!((ui - want).abs() < 1e-15, "at x = {xi}: {ui} vs {want}");
    }
}

#[test]
fn a_non_positive_width_is_rejected_in_both_dimensions() {
    let x = grid(10, 1.0);
    assert_eq!(
        raised_cosine(&x, 1.0, 0.5, 0.0, 1.0).unwrap_err(),
        ExciterError::NonPositiveWidth
    );
    assert_eq!(
        raised_cosine_2d(&x, &x, (0.0, 0.0), -1.0, 1.0).unwrap_err(),
        ExciterError::NonPositiveWidth
    );
}

#[test]
fn the_two_dimensional_bump_is_radially_symmetric() {
    let n = 20usize;
    // Built as `(i - n/2) * h` rather than `-1 + 2i/n`: the latter is not exactly antisymmetric in
    // floating point (`-1 + 1.9` is not `-(-1 + 0.1)`), and this test would then be measuring the
    // grid's asymmetry rather than the bump's.
    let coords: Vec<f64> = (0..=n)
        .map(|i| (i as f64 - (n / 2) as f64) * (2.0 / n as f64))
        .collect();
    let mut fx = Vec::new();
    let mut fy = Vec::new();
    for &yv in &coords {
        for &xv in &coords {
            fx.push(xv);
            fy.push(yv);
        }
    }
    let f = raised_cosine_2d(&fx, &fy, (0.0, 0.0), 0.6, 1.0).expect("a legal bump");

    let nodes = n + 1;
    // Centred on the origin with an even segment count, so the grid is symmetric about it.
    for j in 0..nodes {
        for i in 0..nodes {
            let mirrored = f[(nodes - 1 - j) * nodes + (nodes - 1 - i)];
            assert_eq!(
                f[j * nodes + i],
                mirrored,
                "node ({j}, {i}) breaks symmetry"
            );
        }
    }
    assert_eq!(
        f[(n / 2) * nodes + n / 2],
        1.0,
        "the centre is the amplitude"
    );
    assert_eq!(f[0], 0.0, "a corner is well outside the support");
}
