//! `reduce::sum` and `ops::inner` held against what NumPy actually said — the outside referee for
//! both, frozen before the binding that could ask NumPy goes (retirement plan §48).
//!
//! `src/reduce.rs`'s own tests pin the blocking's *structure* — below eight it is a plain loop,
//! the ragged tail folds into the combined result, the split rounds down to a multiple of eight —
//! each against a deliberate mis-transcription. What none of them can say is that the structure
//! they pin is **NumPy's**: a transcription that got the algorithm wrong in a way none of those
//! three mis-transcriptions spells would pass them all. Until this file, the only thing comparing
//! the crate's sum with `np.sum` itself was `tests/test_binding_surface.py`, through two bridges'
//! read-outs; that comparison is what moved here, and it is sharper for being direct.
//!
//! The record is `reference/numpy_reductions.json`. Its INPUTS are not stored: they are an LCG both
//! languages rebuild bit for bit (u64 wrapping arithmetic, `(s >> 11) / 2^53`, then `2u - 1`, every
//! step exact), so only NumPy's answers are frozen, with each vector's first and last element kept
//! as a canary that the two generators still agree. A failure there is a broken generator, not a
//! broken sum, and the test says which.
//!
//! `np.dot` is a different kind of referee and is held differently. BLAS `ddot` fuses its
//! multiply-add and OpenBLAS picks the kernel by CPU, so there is no scalar recipe to transcribe and
//! the recorded value is a fact about the recording machine (finding #14). It is asserted at the
//! plan's Group A `1e-13`, which is what the binding's test asserted; measured worst gap at the
//! recording, 1.8e-15.

use physsynth_core::ops::inner;
use physsynth_core::reduce::{sum, sum_strided};
use serde_json::Value;

/// The recorder's generator: `2u - 1` with `u` the LCG `src/reduce.rs`'s tests use.
fn lcg(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(2_862_933_555_777_941_757)
        .wrapping_add(3_037_000_493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            2.0 * (((s >> 11) as f64) / ((1u64 << 53) as f64)) - 1.0
        })
        .collect()
}

fn left_to_right(a: &[f64]) -> f64 {
    let mut r = 0.0;
    for &x in a {
        r += x;
    }
    r
}

fn record() -> Value {
    serde_json::from_str(include_str!("reference/numpy_reductions.json")).expect("the record")
}

fn f(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("`{key}` is a number"))
}

fn u(v: &Value, key: &str) -> usize {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("`{key}` is an integer")) as usize
}

/// The inputs this run rebuilt, checked against the canaries before anything else is.
fn rebuilt(case: &Value) -> Vec<f64> {
    let a = lcg(u(case, "seed") as u64, u(case, "len"));
    assert_eq!(
        (a[0], a[a.len() - 1]),
        (f(case, "first"), f(case, "last")),
        "seed {}: the generator no longer reproduces the recorder's inputs -- fix the LCG, the \
         sums have not been compared yet",
        case["seed"]
    );
    a
}

#[test]
fn the_crate_sum_is_numpys_sum_to_the_bit() {
    let rec = record();
    let cases = rec["sums"].as_array().expect("the sums");
    let (mut contiguous, mut strided) = (0, 0);
    for case in cases {
        let a = rebuilt(case);
        let stride = u(case, "stride");
        let (got, what) = if stride == 1 {
            contiguous += 1;
            (sum(&a), format!("len {}", a.len()))
        } else {
            strided += 1;
            let (off, n) = (u(case, "off"), u(case, "n"));
            (
                sum_strided(&a, off, n, stride),
                format!("{n} elements at stride {stride}"),
            )
        };
        assert_eq!(
            got.to_bits(),
            f(case, "sum").to_bits(),
            "seed {} ({what}): the crate summed to {got:e}, NumPy to {:e}",
            case["seed"],
            f(case, "sum")
        );
    }
    // 25 lengths x 8 seeds, spanning every branch: below eight, one block, the 128/129 edge, and
    // recursion to depth six (4,641). Plus NumPy's strided path, which reduces a view in place.
    assert_eq!((contiguous, strided), (200, 24), "the record changed shape");
}

#[test]
fn the_record_is_not_a_left_to_right_loop_in_disguise() {
    // The control. Below eight the two spellings are one computation, so a record that only ever
    // agreed with a plain loop would pass a crate that had collapsed into one. Counted at the
    // recording: 87 of the 200 contiguous sums differ, every one of them at eight terms or more.
    let rec = record();
    let mut differ = 0;
    for case in rec["sums"].as_array().expect("the sums") {
        if u(case, "stride") != 1 {
            continue;
        }
        let a = rebuilt(case);
        if left_to_right(&a) != f(case, "sum") {
            assert!(a.len() >= 8, "NumPy left a {}-term sum unblocked", a.len());
            differ += 1;
        }
    }
    assert_eq!(
        differ, 87,
        "the record no longer separates the two spellings as recorded"
    );
}

#[test]
fn the_inner_product_agrees_with_numpys_dot_to_the_group_a_target() {
    // `h * np.dot(f, g)` went through BLAS, which accumulates in an order no portable loop
    // reproduces, so this is the tolerance the binding test held, not a bit.
    let rec = record();
    let dots = rec["dots"].as_array().expect("the dots");
    assert_eq!(dots.len(), 4);
    for case in dots {
        let n = u(case, "len");
        let h = f(case, "h");
        assert_eq!(h, 1.0 / (n - 1) as f64, "the recorded step is not 1/(n-1)");
        let fv = lcg(u(case, "seed_f") as u64, n);
        let gv = lcg(u(case, "seed_g") as u64, n);
        for (key, a, b) in [("fg", &fv, &gv), ("gf", &gv, &fv), ("ff", &fv, &fv)] {
            let (got, want) = (inner(a, b, h), f(case, key));
            let gap = (got - want).abs() / want.abs();
            assert!(
                gap <= 1e-13,
                "n = {n} {key}: inner {got:e} vs h*np.dot {want:e} ({gap:.3e})"
            );
        }
    }
}
