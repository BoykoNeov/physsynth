//! Native bars for the distributed barrier — model #8, and the last model of Phase 3 (plan §23).
//!
//! `tests/collision.rs` pins the primitives and both solves; this file pins the **shell** built on
//! them, in the same division of labour `tests/mallet.rs` has with the scalar solve. What the shell
//! owns is construction (broadcast the profile, pick the support, solve `m` admittance columns,
//! form `k**2/rho`), the two penetration gathers, the rank-`m` force injection, and the barrier's
//! two-time-averaged potential energy.
//!
//! # The money test is conservation, and it is the whole point of the discrete gradient
//!
//! A lossless string against an elastic barrier conserves `E_string + h Σ φ(η)` to machine
//! precision, and it does so *only* because the contact force is the discrete gradient and the
//! stored energy is the two-time average of the potential. Either one alone drifts at `O(k²)`. So
//! the assertion below is not a smoke test — it is the one bar that fails if the shell pairs the
//! right force with the wrong energy, which is a mistake no trajectory comparison can see (both
//! implementations would make it identically).
//!
//! # What is deliberately NOT here
//!
//! Cross-language agreement. Nothing in this crate can see NumPy, so the comparisons live in
//! `tests/test_rust_parity_collision.py`. What can be pinned natively is the arithmetic *spelling*
//! this port had to choose — `k ** 2` against `k * k` — because that choice is invisible to any
//! energy bar and is the kind of thing §17.2 found the optimiser silently erasing.

use physsynth_core::collision::{BarrierError, BarrierString};
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::string_damped::{self, DampedStiffString};

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s
const K: f64 = 1.0e6;
const DRIFT_TOL: f64 = 1e-10; // CLAUDE.md's acceptance bar

/// The barrier rig's string: `lam` well under 1 to give the stiff contact headroom.
fn string(n: i64, lam: f64, sigma0: f64, sigma1: f64) -> DampedStiffString {
    let c = (T / RHO).sqrt();
    let fs = c * (n as f64) / (L * lam);
    let p = string_damped::Params::new(L, T, RHO, fs, n, 0.0, sigma0, sigma1, 0.28, true)
        .expect("valid string");
    DampedStiffString::new(p)
}

/// A flat rail `height` below the rest line, under the whole string.
fn rail(n: usize, height: f64) -> Vec<f64> {
    vec![height; n + 1]
}

/// A point fret: one finite node, everything else out of support.
fn fret(n: usize, node: usize, height: f64) -> Vec<f64> {
    let mut b = vec![f64::NEG_INFINITY; n + 1];
    b[node] = height;
    b
}

fn barrier(profile: Vec<f64>, alpha: f64, lam_h: f64, sigma0: f64, sigma1: f64) -> BarrierString {
    let n = profile.len() - 1;
    BarrierString::new(
        string(n as i64, 0.4, sigma0, sigma1),
        &profile,
        K,
        alpha,
        lam_h,
        1e-12,
        1e-13,
        60,
    )
    .expect("valid barrier")
}

/// Plucked half-sine, 5 mm — the shape every Python fixture starts from.
fn pluck(bar: &mut BarrierString) {
    let nodes = bar.string.p.nodes();
    let u0: Vec<f64> = (0..nodes)
        .map(|i| 5.0e-3 * (std::f64::consts::PI * (i as f64) / ((nodes - 1) as f64)).sin())
        .collect();
    bar.set_state(&u0, &vec![0.0; nodes]);
}

// -- criterion 1 (money test): a lossless elastic barrier conserves ------------------------------

#[test]
fn a_lossless_barrier_conserves_the_total_energy() {
    for &alpha in &[1.0, 1.5, 2.3] {
        let mut bar = barrier(rail(80, -2.0e-3), alpha, 0.0, 0.0, 0.0);
        pluck(&mut bar);
        let e0 = bar.energy();
        let mut worst: f64 = 0.0;
        for _ in 0..4000 {
            bar.step();
            worst = worst.max(((bar.energy() - e0) / e0).abs());
        }
        assert!(
            worst < DRIFT_TOL,
            "alpha = {alpha}: lossless energy drifted {worst:.3e}"
        );
    }
}

#[test]
fn a_point_fret_conserves_too_and_it_is_a_different_code_path() {
    // `m = 1` makes the admittance block 1x1 and the dense LU a scalar divide. Same physics, a
    // solve that shares almost nothing with the rail's — so conservation here is a separate bar,
    // not a repeat.
    let mut bar = barrier(fret(80, 26, -2.0e-4), 1.5, 0.0, 0.0, 0.0);
    pluck(&mut bar);
    let e0 = bar.energy();
    let mut worst: f64 = 0.0;
    for _ in 0..4000 {
        bar.step();
        worst = worst.max(((bar.energy() - e0) / e0).abs());
    }
    assert!(worst < DRIFT_TOL, "point fret drifted {worst:.3e}");
}

// -- criterion 2: passivity ---------------------------------------------------------------------

#[test]
fn hysteresis_and_string_loss_only_ever_remove_energy() {
    for &(lam_h, sigma0, sigma1) in &[(2.0e4, 0.0, 0.0), (0.0, 0.5, 0.05), (2.0e4, 0.5, 0.05)] {
        let mut bar = barrier(rail(80, -2.0e-3), 1.5, lam_h, sigma0, sigma1);
        pluck(&mut bar);
        let mut prev = bar.energy();
        for n in 0..2000 {
            bar.step();
            let e = bar.energy();
            assert!(
                e <= prev + 1.0e-12 * prev.abs(),
                "lam_h = {lam_h}, sigma = ({sigma0}, {sigma1}): energy rose at step {n}"
            );
            prev = e;
        }
    }
}

// -- criterion 3: the barrier is one-sided ------------------------------------------------------

#[test]
fn an_out_of_reach_barrier_is_the_bare_string_to_the_bit() {
    // The one anchor `collision` owns, natively. With the rail at -100 m nothing ever penetrates,
    // the contact force is exactly zero, and the injected correction must be exactly zero too —
    // not "small". A shell that wrote `u += force_pref * 0.0` through a path that touched the
    // values would still pass an energy bar and would fail this.
    let mut bar = barrier(rail(80, -100.0), 1.5, 0.0, 0.3, 0.0);
    let mut bare = string(80, 0.4, 0.3, 0.0);
    let nodes = bar.string.p.nodes();
    let u0: Vec<f64> = (0..nodes)
        .map(|i| 5.0e-3 * (std::f64::consts::PI * (i as f64) / ((nodes - 1) as f64)).sin())
        .collect();
    let zeros = vec![0.0; nodes];
    bar.set_state(&u0, &zeros);
    bare.set_state(&u0, &zeros);
    for _ in 0..1000 {
        bar.step();
        bare.step();
    }
    assert!(bar.s.contact_force.iter().all(|&f| f == 0.0));
    assert_eq!(bar.string.u, bare.u);
    assert_eq!(bar.energy(), bare.energy());
}

#[test]
fn the_barrier_pushes_the_string_away_from_it_and_never_toward_it() {
    // The contact force density is `>= 0` everywhere, always: a one-sided spring cannot pull.
    let mut bar = barrier(rail(80, -2.0e-3), 1.5, 0.0, 0.0, 0.0);
    pluck(&mut bar);
    let mut ever_in_contact = false;
    for _ in 0..2000 {
        bar.step();
        for (j, &f) in bar.s.contact_force.iter().enumerate() {
            assert!(f >= 0.0, "the barrier pulled at node {j}: {f:e}");
            if f > 0.0 {
                ever_in_contact = true;
            }
        }
    }
    assert!(
        ever_in_contact,
        "the fixture never made contact — it proves nothing"
    );
}

// -- the support, which is a geometry decision no energy bar can see ----------------------------

#[test]
fn the_support_is_the_finite_interior_and_nothing_else() {
    let n = 80;
    let mut profile = vec![f64::NEG_INFINITY; n + 1];
    // Two finite interior nodes, plus both clamped ends and one NaN — none of which may join.
    profile[0] = -1.0e-4;
    profile[n] = -1.0e-4;
    profile[27] = -1.0e-4;
    profile[54] = -2.0e-4;
    profile[40] = f64::NAN;
    let bar = barrier(profile, 1.5, 0.0, 0.0, 0.0);
    assert_eq!(bar.p.support, vec![27, 54]);
    assert_eq!(bar.p.int_idx, vec![26, 53]);
    assert_eq!(bar.p.b, vec![-1.0e-4, -2.0e-4]);
}

#[test]
fn the_admittance_block_is_the_symmetric_positive_definite_one_the_solve_assumes() {
    // `G = force_pref * (A^-1)_{S,S}` with `A` SPD, so `G` is symmetric and positive-diagonal.
    // The Newton proof in `collision.py`'s header rests on exactly that, and a transposed gather
    // would break it silently — the solve would still converge, just to a different root.
    //
    // To a tolerance, not to the bit, and that is a fact about the construction rather than a
    // slack bar: each column is its own back-substitution, so `(A^-1)_ij` and `(A^-1)_ji` are
    // computed by different arithmetic and agree only to rounding. Observed worst here ~1e-16
    // relative; a transposed gather would be O(1) wrong, which is 1e16 times this bar.
    let bar = barrier(rail(40, -2.0e-3), 1.5, 0.0, 0.0, 0.0);
    let m = bar.p.support_len();
    assert_eq!(m, 39);
    let mut worst: f64 = 0.0;
    for i in 0..m {
        assert!(bar.p.g_mat[i * m + i] > 0.0, "G_{i}{i} is not positive");
        for j in 0..m {
            let (a, b) = (bar.p.g_mat[i * m + j], bar.p.g_mat[j * m + i]);
            let scale = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
            worst = worst.max((a - b).abs() / scale);
        }
    }
    assert!(
        worst < 1e-12,
        "G is not symmetric: worst relative gap {worst:e}"
    );
}

// -- the spelling the optimiser would happily erase ---------------------------------------------

/// The first sample rate in a deterministic sweep at which `k ** 2 / rho` and `k * k / rho` are
/// different doubles, where `k = 1/fs` — or `None` if this machine's `pow` never separates them.
///
/// Two things about this search are load-bearing, and both were found by it going wrong.
///
/// **It searches rather than hardcoding**, which is §22.1's lesson applied before it could bite:
/// which arguments `pow` rounds differently from a multiply is a property of the C library the
/// runner links, so a witness measured on one machine would be a bar decided by that machine
/// (§21.6). Searching makes the test bite wherever the distinction exists and stay quiet where it
/// does not.
///
/// **It goes through [`scalar_pow`] rather than writing `k.powf(2.0)`**, and that is §17.2 for the
/// third time. Written the obvious way this loop passed in debug and found *nothing* in release —
/// LLVM folds a literal `2.0` exponent into `x * x`, which makes the search's own predicate
/// `x * x != x * x`, so the test silently asserted nothing and reported a green tick. The
/// `#[inline(never)]` inside `scalar_pow` is what stops that, and it is the same helper the port
/// itself uses, so the search now measures the thing under test rather than a rewrite of it.
///
/// The predicate is the whole expression, `/ RHO` included, and that too came from a failure: the
/// first witness the sweep returned separated `k ** 2` from `k * k` and then had the difference
/// **absorbed by the division**, so the negative control below compared a value against itself.
fn pow_multiply_witness() -> Option<f64> {
    let mut fs = 20_000.0f64;
    for _ in 0..400_000 {
        let k = 1.0 / fs;
        if scalar_pow(k, 2.0) / RHO != k * k / RHO {
            return Some(fs);
        }
        fs += 0.7;
    }
    None
}

#[test]
fn the_force_prefactor_is_the_pow_spelling_and_not_a_multiply() {
    // `collision.py` writes `string.k ** 2 / string.rho` — `float.__pow__`, i.e. libm's `pow`.
    // `bow.py` writes `self.k * self.k` at the same spot. Those are different doubles for a small
    // fraction of sample rates (86 of 200,000 in this project's range, on the machine this was
    // written on), so the two ports must spell it differently, and this pins which one this model
    // got.
    let Some(fs) = pow_multiply_witness() else {
        // Not a failure: on a C library whose `pow` agrees with the multiply everywhere in this
        // range, the two spellings are the same computation and there is nothing to distinguish.
        // Saying so is more useful than a green tick that meant nothing.
        println!("this machine's pow never separates k**2 from k*k in 20-300 kHz; nothing to pin");
        return;
    };
    let k = 1.0 / fs;
    assert_ne!(
        scalar_pow(k, 2.0) / RHO,
        k * k / RHO,
        "the witness search returned a non-witness"
    );
    let p = string_damped::Params::new(L, T, RHO, fs, 80, 0.0, 0.0, 0.0, 0.28, true)
        .expect("valid string");
    let s = DampedStiffString::new(p);
    let bar = BarrierString::new(s, &rail(80, -2.0e-3), K, 1.5, 0.0, 1e-12, 1e-13, 60)
        .expect("valid barrier");
    assert_eq!(bar.p.force_pref, scalar_pow(k, 2.0) / RHO);
    assert_ne!(bar.p.force_pref, k * k / RHO);
}

// -- construction -------------------------------------------------------------------------------

#[test]
fn construction_rejects_the_unphysical_in_the_originals_order() {
    let mk = |stiffness, alpha, hysteresis| {
        BarrierString::new(
            string(80, 0.4, 0.0, 0.0),
            &rail(80, -2.0e-3),
            stiffness,
            alpha,
            hysteresis,
            1e-12,
            1e-13,
            60,
        )
        .map(|_| ())
    };
    assert_eq!(mk(0.0, 1.5, 0.0), Err(BarrierError::NonPositiveStiffness));
    assert_eq!(mk(-1.0, 1.5, 0.0), Err(BarrierError::NonPositiveStiffness));
    assert_eq!(mk(K, 0.5, 0.0), Err(BarrierError::AlphaTooSmall));
    assert_eq!(mk(K, 1.5, -1.0), Err(BarrierError::NegativeHysteresis));
    // A call wrong in more than one way reports the fault the original would report first.
    assert_eq!(mk(0.0, 0.5, -1.0), Err(BarrierError::NonPositiveStiffness));
    assert!(mk(K, 1.0, 0.0).is_ok());
}

#[test]
fn a_barrier_with_no_finite_interior_node_is_refused() {
    let out = BarrierString::new(
        string(80, 0.4, 0.0, 0.0),
        &vec![f64::NEG_INFINITY; 81],
        K,
        1.5,
        0.0,
        1e-12,
        1e-13,
        60,
    )
    .map(|_| ());
    assert_eq!(out, Err(BarrierError::EmptySupport));
    // The two clamped ends do not count, which is the interesting half of the check.
    let mut ends = vec![f64::NEG_INFINITY; 81];
    ends[0] = -1.0e-4;
    ends[80] = -1.0e-4;
    let out = BarrierString::new(
        string(80, 0.4, 0.0, 0.0),
        &ends,
        K,
        1.5,
        0.0,
        1e-12,
        1e-13,
        60,
    )
    .map(|_| ());
    assert_eq!(out, Err(BarrierError::EmptySupport));
}

#[test]
fn set_state_resets_the_seed_and_the_step_count_but_not_the_last_force() {
    // Faithful to the original rather than tidy: `set_state` refreshes the continuation seed and
    // zeroes `n`, and leaves `contact_force` and `newton_iters` where the last step left them.
    // Stepped until the rail is *actually being pressed*, rather than for a round number of
    // steps: the claim is that `set_state` leaves a NONZERO force alone, and the string spends
    // most of its period clear of the barrier, so a fixed step count tests the claim only by luck.
    let mut bar = barrier(rail(80, -2.0e-3), 1.5, 0.0, 0.0, 0.0);
    pluck(&mut bar);
    let mut stepped = 0;
    while stepped < 4000 && !bar.s.contact_force.iter().any(|&f| f > 0.0) {
        bar.step();
        stepped += 1;
    }
    let stale = bar.s.contact_force.clone();
    let iters = bar.s.newton_iters;
    assert!(
        stale.iter().any(|&f| f > 0.0),
        "the fixture never touched the rail in {stepped} steps"
    );
    pluck(&mut bar);
    assert_eq!(bar.s.n, 0);
    assert_eq!(bar.s.contact_force, stale);
    assert_eq!(bar.s.newton_iters, iters);
    // The seed is the penetration of the freshly-set field, not the one 200 steps of contact left.
    for (j, &node) in bar.p.support.iter().enumerate() {
        assert_eq!(bar.s.penetration[j], bar.p.b[j] - bar.string.u[node]);
    }
}

#[test]
fn the_contact_mask_is_the_sign_of_the_penetration() {
    let mut bar = barrier(rail(80, -2.0e-3), 1.5, 0.0, 0.0, 0.0);
    pluck(&mut bar);
    let mut ever = false;
    for _ in 0..800 {
        bar.step();
        let mask = bar.contact_mask();
        for (j, &m) in mask.iter().enumerate() {
            assert_eq!(m, bar.s.penetration[j] > 0.0);
            ever |= m;
        }
    }
    assert!(
        ever,
        "nothing was ever in contact — the mask proves nothing"
    );
}
