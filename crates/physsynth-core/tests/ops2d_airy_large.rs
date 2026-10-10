//! The Airy solve's backward error on the five large grids the room tests build (up to
//! 160 × 128) — split from `ops2d_scipy.rs` so that CI's unoptimised job can leave it out: 116 s
//! there against ~15 s optimised, the human's call (retirement plan §49). It pins no arithmetic
//! spelling and the solve has no `debug_assert!` an unoptimised run would add.
//!
//! SuperLU's answers were not recorded on these grids (2.7 MB, also the human's call), so this is
//! the whole of the claim here: the crate's sparse LU solves `B_F f = Wa · source` to machine
//! precision in the backward sense, whatever the grid, while the forward error is free to grow
//! like the clamped biharmonic's `N⁴` condition number.

use physsynth_core::ops2d::AiryStressSolver;

mod airy_fixture;
use airy_fixture::*;

#[test]
fn the_crates_airy_solve_is_backward_stable_on_the_large_grids() {
    for (nx, ny, h) in [
        (40usize, 32usize, 0.025),
        (48, 48, 1.0 / 120.0),
        (80, 64, 0.0125),
        (96, 96, 1.0 / 240.0),
        (160, 128, 0.00625),
    ] {
        let airy = AiryStressSolver::new(nx, ny, h).expect("SPD");
        let raw = lcg(11, airy.n_nodes());
        let f_rs = airy.solve(&source(&airy, &raw)).expect("solve");
        let rhs = live_and_load(&airy, nx, ny, h, &raw);
        let be = backward_error(airy.bf(), &live(&airy, &f_rs), &rhs);
        eprintln!("{nx}x{ny}: backward error {be:.3e}");
        assert!(
            be <= 1e-13,
            "the crate's solve is not backward stable at {nx}x{ny}: {be:.3e}"
        );
    }
}
