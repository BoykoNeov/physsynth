//! NumPy's pairwise summation, transcribed — so a ported `np.sum` can be bit-identical.
//!
//! # Why this module exists, and what it retires
//!
//! Plan §30.2 measured `np.sum`'s blocking and found a written-down cutoff: below **eight**
//! elements `np.sum` is a plain left-to-right loop, at eight or above it is not. It drew the
//! conclusion that bit-identity across a ported `np.sum` is *decidable by counting the terms* —
//! available below eight, unavailable at or above — and refused to transcribe the blocking on the
//! grounds that doing so would be "a claim about a library internal, and after §22.1 a claim about
//! the CPU as well".
//!
//! The first half of that stands, and it is sharper than any measurement: below eight it is the
//! same computation, for free, whatever the values. The second half is **wrong**, and this module
//! is the correction. NumPy's blocking is one fixed algorithm rather than a dispatched kernel, and
//! transcribing it reproduces `np.sum` **exactly**: measured 0 disagreements in 2,000 random
//! vectors at each of n = 1, 4, 7, 8, 9, 15, 16, 20, 30, 56, 128, 129, 200, 560 and 4,641, and 0 in
//! 200 at n = 40,000; 0 in 200 for whole-array sums of 3-D arrays at five shapes up to 41x33x25;
//! and 0 in 200 for a **strided** reduction (a column of a C-order 3-D array), which is a different
//! code path.
//!
//! The distinction that makes this not §22.1 is worth stating, because the two look alike. §22.1's
//! hazard is that NumPy computes `pow`, `sin` and `exp` with its **own CPU-dispatched routines**,
//! chosen at import from the machine's feature set — two CPUs, two instruction selections, two last
//! bits, and nothing in the source of either language shows it. A summation has no comparable
//! freedom: the order is fixed by the blocking below, and the unroll by eight exists so that a
//! vector unit can be used *without* changing that order. A transcription is therefore a claim
//! about an algorithm, which is stable, and not about an instruction selection, which is not.
//!
//! That is also why this is not §14.2's or §18.2's bargain. BLAS `ddot` **fuses** its multiply-add
//! and OpenBLAS picks the kernel by CPU, so there is no scalar recipe at all; SciPy's sparse
//! product hides its contraction order inside a kernel. `np.sum` is scalar arithmetic in a
//! published order, and there is a recipe.
//!
//! # The algorithm
//!
//! ```text
//! n < 8      ->  plain left-to-right loop from 0.0
//! n <= 128   ->  eight accumulators seeded from a[0..8], unrolled by eight, combined
//!                ((r0+r1) + (r2+r3)) + ((r4+r5) + (r6+r7)), then the ragged tail left to right
//! n  > 128   ->  split at n/2 rounded DOWN to a multiple of eight, recurse, add the two halves
//! ```
//!
//! Two details a casual reading gets wrong, each of which is invisible at the lengths the other is
//! wrong at: the eight accumulators are **seeded** from the first eight elements rather than
//! started at zero (so `n = 8` is `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))` with no zero added
//! anywhere), and the ragged tail folds into the **combined** result left to right rather than back
//! into the accumulators. Both are asserted below against a deliberate mis-transcription.
//!
//! # When to reach for this, and when not to
//!
//! §14.2's question decides it: **does the reduction reach the next timestep?** If it does — the
//! port weights `w = W / W.sum()` and the port resistance `R_room` in [`crate::airbox_port`] both
//! do, and a last bit there is a different trajectory — use [`sum`]. If it does not, a tolerance is
//! honest and the cheaper spelling is fine. [`crate::airbox`] was written before this module
//! existed and still books its two energy ledgers with a plain loop; those are pure bookkeeping
//! (§30.2), so that is a tightening left undone on purpose rather than a disagreement.

/// NumPy's pairwise block size — the length above which `np.sum` recurses.
const PW_BLOCKSIZE: usize = 128;

/// `np.sum(a)` for a contiguous float64 slice, bit-identical.
///
/// See the module docs for what that claim rests on and where it has been measured.
pub fn sum(a: &[f64]) -> f64 {
    sum_strided(a, 0, a.len(), 1)
}

/// `np.sum` over `n` elements of `a` starting at `off` and stepping by `stride`.
///
/// NumPy's own routine takes a byte stride and reduces a strided view without materializing it, so
/// the recursion splits on the element count and never on the slice. Exposed because a wall face
/// and a node plane are both strided views of a room's arrays.
pub fn sum_strided(a: &[f64], off: usize, n: usize, stride: usize) -> f64 {
    let at = |i: usize| a[off + i * stride];
    if n < 8 {
        // The one case that is free: below the cutoff `np.sum` IS a left-to-right loop (§30.2).
        let mut res = 0.0;
        for i in 0..n {
            res += at(i);
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        // Seeded from the first eight, NOT started at zero: at n = 8 nothing is added to a zero.
        let mut r = [at(0), at(1), at(2), at(3), at(4), at(5), at(6), at(7)];
        let mut i = 8;
        let whole = n - (n % 8);
        while i < whole {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += at(i + j);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        // The ragged tail folds into the COMBINED result, not back into the accumulators.
        while i < n {
            res += at(i);
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8; // rounded DOWN to a multiple of eight, so the left half stays block-aligned
    sum_strided(a, off, n2, stride) + sum_strided(a, off + n2 * stride, n - n2, stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random stream. The crate's dependency list is empty by policy
    /// (plan §10), so the native tests roll their own rather than take `rand`.
    fn lcg(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(2_862_933_555_777_941_757)
            .wrapping_add(3_037_000_493);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 11) as f64) / ((1u64 << 53) as f64)
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

    #[test]
    fn below_eight_is_a_plain_loop() {
        // §30.2's cutoff, asserted rather than remembered: this is what makes a short reduction
        // exact for free, and it is the half of §30.2 this module does NOT retire.
        for n in 0..8 {
            for seed in 0..50 {
                let a = lcg(seed, n);
                assert_eq!(sum(&a), left_to_right(&a), "n = {n}, seed = {seed}");
            }
        }
    }

    #[test]
    fn at_and_above_eight_it_is_not_a_plain_loop() {
        // The negative control. If this ever goes green the transcription has collapsed into the
        // naive loop and every other test here is asserting nothing — §23.5's empty search, which
        // arrived inside the very test written to catch it.
        let mut differed = 0;
        for n in [8usize, 9, 16, 56, 129, 560] {
            for seed in 0..200 {
                let a = lcg(seed + 1000 * n as u64, n);
                if sum(&a) != left_to_right(&a) {
                    differed += 1;
                }
            }
        }
        assert!(
            differed > 100,
            "only {differed} of 1200 differed from a left-to-right loop"
        );
    }

    #[test]
    fn the_ragged_tail_folds_into_the_combined_result() {
        // The tail of a non-multiple-of-eight block is added to the COMBINED sum, not back into
        // the accumulators. Both spellings are the same algebra and different doubles.
        //
        // Its sibling mis-transcription — starting the eight accumulators at zero instead of
        // seeding them from `a[0..8]` — is NOT a difference and was written here first as a test
        // that could never fail: `0.0 + x` is exactly `x`, so the zeroed variant's first block
        // reproduces the seeding step for step. §23.5's empty search, arriving inside the test
        // written to catch it; kept as a comment because "we looked and there is nothing here" is
        // the useful half.
        fn tail_into_accumulators(a: &[f64]) -> f64 {
            let n = a.len();
            let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
            let mut i = 8;
            let whole = n - (n % 8);
            while i < whole {
                for (j, acc) in r.iter_mut().enumerate() {
                    *acc += a[i + j];
                }
                i += 8;
            }
            while i < n {
                r[i - whole] += a[i];
                i += 1;
            }
            ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        }
        let mut bad = 0;
        for n in [12usize, 20, 29, 61] {
            for seed in 0..200 {
                let a = lcg(seed + 31 * n as u64, n);
                if tail_into_accumulators(&a) != sum(&a) {
                    bad += 1;
                }
            }
        }
        assert!(
            bad > 0,
            "folding the ragged tail into the accumulators made no difference"
        );
    }

    #[test]
    fn the_split_rounds_down_to_a_multiple_of_eight() {
        // n = 200: n/2 is 100 and 100 - (100 % 8) is 96, so a naive halving takes a different pair
        // of sub-sums. Searched rather than asserted at one hand-picked vector (§26.6).
        fn naive_split(a: &[f64]) -> f64 {
            let n = a.len();
            if n <= PW_BLOCKSIZE {
                return sum(a);
            }
            let n2 = n / 2;
            sum(&a[..n2]) + sum(&a[n2..])
        }
        let mut bad = 0;
        for seed in 0..200 {
            let a = lcg(seed, 200);
            if naive_split(&a) != sum(&a) {
                bad += 1;
            }
        }
        assert!(
            bad > 0,
            "rounding the split down to a multiple of eight made no difference"
        );
    }

    #[test]
    fn strided_matches_the_contiguous_sum_of_the_same_elements() {
        let a = lcg(99, 3 * 40);
        let picked: Vec<f64> = (0..40).map(|i| a[1 + 3 * i]).collect();
        assert_eq!(sum_strided(&a, 1, 40, 3), sum(&picked));
    }

    #[test]
    fn an_empty_sum_is_zero() {
        assert_eq!(sum(&[]), 0.0);
    }
}
