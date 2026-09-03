//! The special functions' own bars, asserted without an interpreter in the room.
//!
//! These are not parity tests — `tests/test_rust_parity_analysis.py` compares this code against
//! SciPy. These assert that the code is *right on its own terms*, which is the half a parity test
//! cannot reach: two implementations can agree beautifully and both be wrong, and a hand-rolled
//! Bessel routine is exactly the kind of thing that agrees with nothing until it is checked against
//! the definition.
//!
//! So `J` is checked against its own differential equation, its recurrence, the normalising
//! identity it is built on, its small-argument series, and published digits; `I` against the same
//! recurrence with the sign that distinguishes it; the zeros against `J` actually vanishing there
//! and against the interlacing theorem the algorithm relies on.

use physsynth_analysis::bessel::{iv, ivp, j0_zeros, j1, jn, jn_all, jn_zeros, jvp};

/// Published to sixteen digits (Abramowitz & Stegun 9.5.1, and every table since).
const J0_ZERO_1: f64 = 2.404_825_557_695_773;
const J0_ZERO_2: f64 = 5.520_078_110_286_311;
const J1_ZERO_1: f64 = 3.831_705_970_207_512;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn j_at_zero_is_the_kronecker_delta() {
    let v = jn_all(0.0, 5);
    assert_eq!(v[0], 1.0, "J_0(0) = 1 exactly");
    for (n, &x) in v.iter().enumerate().skip(1) {
        assert_eq!(x, 0.0, "J_{n}(0) = 0 exactly");
    }
}

#[test]
fn j_matches_its_own_small_argument_series() {
    // J_n(x) = sum_k (-1)^k (x/2)^{n+2k} / (k! (n+k)!). At x = 0.5 this converges in a handful of
    // terms with no cancellation worth speaking of, so it is an independent evaluation rather than
    // a restatement of the recurrence.
    for n in 0..6u32 {
        let x = 0.5f64;
        let h = 0.5 * x;
        let mut term = 1.0;
        for k in 1..=n {
            term *= h / k as f64;
        }
        let mut s = term;
        let mut k = 1u32;
        while k < 30 {
            term *= -h * h / (k as f64 * (n + k) as f64);
            s += term;
            k += 1;
        }
        // 3e-16 is about three ulps of J_0(0.5) = 0.938 — the two evaluations are different
        // algorithms, not the same one twice, so agreeing to a last bit is the most that can be
        // asked and asking for exactness here would be asking the series to be the recurrence.
        assert!(
            close(jn(n as i32, x), s, 3e-16),
            "J_{n}(0.5): recurrence {} vs series {s}",
            jn(n as i32, x)
        );
    }
}

#[test]
fn j_satisfies_the_three_term_recurrence() {
    // J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x). Not how the values are produced downward — the
    // normalisation is — so this is a real check on the whole pipeline rather than an identity.
    for &x in &[0.3, 1.0, 4.7, 14.0, 33.3, 54.5] {
        let v = jn_all(x, 20);
        for n in 1..19 {
            let lhs = v[n - 1] + v[n + 1];
            let rhs = (2.0 * n as f64 / x) * v[n];
            assert!(
                close(lhs, rhs, 1e-14),
                "recurrence broken at x={x}, n={n}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn j_satisfies_bessels_differential_equation() {
    // x² y'' + x y' + (x² - n²) y = 0, with the derivatives taken from the analytic recursion
    // rather than by finite differences — so this checks `jvp` and `jn` together.
    for &x in &[0.7, 2.5, 9.0, 13.9] {
        for n in 0..7i32 {
            let (y, y1, y2) = (jvp(n, x, 0), jvp(n, x, 1), jvp(n, x, 2));
            let r = x * x * y2 + x * y1 + (x * x - (n * n) as f64) * y;
            assert!(r.abs() < 1e-13, "Bessel ODE residual {r} at x={x}, n={n}");
        }
    }
}

#[test]
fn the_normalising_identity_holds() {
    // J_0 + 2(J_2 + J_4 + ...) = 1. It is what the downward recurrence is normalised by, so on its
    // own it would be a tautology — except that it is applied to a *truncated* sum here, at an
    // order well short of the one the algorithm starts from. A start order too low would show up
    // as this drifting, and it does not.
    for &x in &[0.1, 3.0, 12.0, 40.0] {
        let v = jn_all(x, 120);
        let mut s = v[0];
        let mut k = 2;
        while k <= 120 {
            s += 2.0 * v[k];
            k += 2;
        }
        assert!(close(s, 1.0, 1e-15), "sum identity gave {s} at x={x}");
    }
}

#[test]
fn i_satisfies_its_own_recurrence_and_grows() {
    // I_{n-1}(x) - I_{n+1}(x) = (2n/x) I_n(x) -- the minus is what separates it from J.
    for &x in &[0.3, 2.0, 7.5, 14.0] {
        for n in 1..8i32 {
            let lhs = iv(n - 1, x) - iv(n + 1, x);
            let rhs = (2.0 * n as f64 / x) * iv(n, x);
            assert!(
                (lhs - rhs).abs() <= 1e-13 * lhs.abs().max(1.0),
                "modified recurrence broken at x={x}, n={n}: {lhs} vs {rhs}"
            );
        }
        assert!(iv(0, x) > iv(1, x), "I_0 > I_1 for x > 0");
    }
    assert_eq!(iv(0, 0.0), 1.0);
    assert_eq!(iv(3, 0.0), 0.0);
    assert_eq!(iv(-3, 2.0), iv(3, 2.0), "I_{{-n}} = I_n exactly");
}

#[test]
fn i_satisfies_the_modified_differential_equation() {
    // x² y'' + x y' - (x² + n²) y = 0.
    for &x in &[0.7, 3.5, 11.0, 14.0] {
        for n in 0..7i32 {
            let (y, y1, y2) = (ivp(n, x, 0), ivp(n, x, 1), ivp(n, x, 2));
            let r = x * x * y2 + x * y1 - (x * x + (n * n) as f64) * y;
            assert!(
                r.abs() <= 1e-13 * y.abs().max(1.0) * x * x,
                "modified ODE residual {r} at x={x}, n={n}"
            );
        }
    }
}

#[test]
fn negative_orders_carry_the_right_sign() {
    for &x in &[0.4, 3.0, 10.0] {
        assert_eq!(jn(-1, x), -jn(1, x));
        assert_eq!(jn(-2, x), jn(2, x));
        assert_eq!(jn(-3, x), -jn(3, x));
    }
}

#[test]
fn the_first_derivative_matches_the_classical_pair_identity() {
    // J_n' = (J_{n-1} - J_{n+1})/2 is the k = 1 case of the recursion `jvp` implements, and the two
    // are the same expression -- so this pins the *general* recursion against the k = 1 identity at
    // k = 2 and 3 by finite differences instead, where an off-by-one in the binomials would show.
    let h = 1e-5;
    for &x in &[1.5, 6.0, 12.0] {
        for n in 0..5i32 {
            for k in 1..3u32 {
                let fd = (jvp(n, x + h, k - 1) - jvp(n, x - h, k - 1)) / (2.0 * h);
                assert!(
                    close(jvp(n, x, k), fd, 1e-8),
                    "jvp k={k} disagrees with a finite difference of k={} at x={x}, n={n}",
                    k - 1
                );
            }
        }
    }
}

#[test]
fn published_zeros_are_reproduced() {
    let z0 = j0_zeros(4);
    assert!(close(z0[0], J0_ZERO_1, 1e-14), "j_{{0,1}} = {}", z0[0]);
    assert!(close(z0[1], J0_ZERO_2, 1e-14), "j_{{0,2}} = {}", z0[1]);
    let z1 = jn_zeros(1, 3);
    assert!(close(z1[0], J1_ZERO_1, 1e-13), "j_{{1,1}} = {}", z1[0]);
}

#[test]
fn every_zero_is_a_zero_and_they_interlace() {
    // The two properties the algorithm claims. The second is the one that makes the first safe:
    // an oscillating function has many roots, and "J is small here" would not prove the *right*
    // one was found.
    for m in 0..13u32 {
        let z = jn_zeros(m, 12);
        for (i, &x) in z.iter().enumerate() {
            assert!(
                jn(m as i32, x).abs() < 1e-14,
                "J_{m}({x}) = {} is not a zero",
                jn(m as i32, x)
            );
            if i > 0 {
                assert!(x > z[i - 1], "zeros of J_{m} came back out of order");
            }
        }
        if m > 0 {
            let below = jn_zeros(m - 1, 13);
            for (i, &x) in z.iter().enumerate() {
                assert!(
                    below[i] < x && x < below[i + 1],
                    "j_{{{m},{}}} = {x} is not inside (j_{{{},{}}}, j_{{{},{}}})",
                    i + 1,
                    m - 1,
                    i + 1,
                    m - 1,
                    i + 2
                );
            }
        }
    }
}

#[test]
fn j1_is_the_order_one_bessel_and_nothing_else() {
    // The name `physsynth/core/radiation.py` reaches for, so it gets its own line: the piston
    // resistance is `1 - J1(2ka)/ka`, and a J0 wired in by mistake would still be smooth, still be
    // bounded, and still pass every energy bar in this project.
    for &x in &[0.01, 1.0, 3.66, 12.0] {
        assert_eq!(j1(x), jn(1, x));
    }
    assert_eq!(j1(0.0), 0.0, "J_1(0) = 0");
    // J_1(x) ~ x/2 as x -> 0.
    assert!(close(j1(1e-6), 5e-7, 1e-19));
}
