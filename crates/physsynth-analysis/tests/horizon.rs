//! The resolution horizon's own bars — the identities its docstrings claim, checked as maths.
//!
//! `tests/analysis_frozen_values.py` holds what the Python implementation in `tests/helpers.py`
//! said about these seven functions before it was replaced, and that record catches a
//! transcription error, a wrong branch and a regression. It cannot catch an error the Python made
//! too — both sides would agree on the same wrong number and every horizon claim in the project
//! would be validated against a fiction. That is what this file is for, and `docs/dev/
//! rust-migration-plan.md` §37.11 is the precedent: a native bar found a 544% defect in the free
//! circular plate's oracle that the Python had always had and no parity test could see.
//!
//! Everything in the first half is an identity or an inequality out of the docstrings, so nothing in
//! it is a fixture. The two exceptions are the published grid fractions (5.925% and 8.378%), which
//! are *derived* claims quoted in the plan and worth pinning where a reader will find them.
//!
//! The second half ("The schemes' horizons") is carried from `tests/test_resolution_horizon.py`,
//! retired at retirement plan §46: the primitives asked about the string, the plate, the grained
//! plate and the membrane through `modal`'s closed-form dispersion relations. The seven of that
//! file's bars that read a BUILT model are in `crates/physsynth-core/tests/horizon_models.rs`. Every
//! figure the Python printed was reproduced here to the last digit, root finds included.

use physsynth_analysis::horizon::{
    block_weight, cancellation_courant, mode_block, mode_family, pitch_error_cents, pitch_horizon,
    sinc_horizon_fraction,
};

/// `1/√2`, the 2-D CFL ceiling, spelled the way the formula reaches it.
///
/// `√2 / 2` and `1 / √2` are the same real number and **not** the same double — they differ by one
/// ulp — and `cancellation_courant` computes `√(2m⁴) / 2m²`, which is the first spelling. So the
/// constant here is written that way rather than as `FRAC_1_SQRT_2`, and the comparisons below
/// still carry a tolerance: the intermediate `2m⁴` rounds differently as `m` climbs.
const CFL_2D: f64 = std::f64::consts::SQRT_2 / 2.0;

// -- pitch_error_cents ---------------------------------------------------------------------------

#[test]
fn an_exact_scheme_has_no_pitch_error_at_all() {
    // Not a tolerance and deliberately so: `log2(x/x)` is `log2(1.0)`, and 1.0 is the one argument
    // every conforming log2 returns exactly zero for. A nonzero reading here is a wrong formula,
    // not a rounding.
    let f = [110.0, 220.5, 331.0, 447.25];
    for e in pitch_error_cents(&f, &f).unwrap() {
        assert_eq!(
            e, 0.0,
            "an exact scheme reported a pitch error of {e} cents"
        );
    }
}

#[test]
fn the_sign_says_flat_and_the_size_is_the_cent() {
    // A cent is the 1200th root of two, so a ratio of exactly 2^(1/1200) must read +1.000 cents and
    // its reciprocal -1.000. Sign convention: negative means the scheme is FLAT, which is the
    // direction every scheme in this project errs.
    let one_cent = 2.0_f64.powf(1.0 / 1200.0);
    let got = pitch_error_cents(&[440.0 * one_cent, 440.0 / one_cent], &[440.0, 440.0]).unwrap();
    assert!((got[0] - 1.0).abs() < 1e-12, "sharp by {} cents", got[0]);
    assert!((got[1] + 1.0).abs() < 1e-12, "flat by {} cents", got[1]);
}

#[test]
fn a_length_mismatch_is_refused_rather_than_zipped_short() {
    // Rust's `zip` stops at the shorter side, so the guard is the only thing between a caller who
    // passed the wrong family and an answer about its first few modes. Assert the refusal.
    let err = pitch_error_cents(&[1.0, 2.0], &[1.0]).unwrap_err();
    assert!(err.contains("shape mismatch"), "{err}");
}

// -- pitch_horizon -------------------------------------------------------------------------------

#[test]
fn the_horizon_is_a_leading_prefix_and_not_the_last_mode_inside() {
    // The distinction the returned `monotone` flag exists for. This error curve steps outside the
    // bound at index 2 and back inside at index 4: the prefix reading is 2, the "last mode inside"
    // reading would be 5, and the two are different answers to different questions.
    let cont = [100.0, 100.0, 100.0, 100.0, 100.0];
    let one_cent = 2.0_f64.powf(1.0 / 1200.0);
    let disc = [
        100.0,
        100.0 * one_cent,
        100.0 * one_cent.powi(9), // outside a 5-cent bound
        100.0 * one_cent.powi(9),
        100.0 * one_cent.powi(2), // back inside
    ];
    let (horizon, monotone) = pitch_horizon(&disc, &cont, 5.0).unwrap();
    assert_eq!(
        horizon, 2,
        "the prefix stops at the first mode outside the bound"
    );
    assert!(
        !monotone,
        "this curve is not monotone and the flag must say so"
    );
}

#[test]
fn a_curve_that_never_leaves_the_bound_resolves_every_mode_it_was_given() {
    let cont = [100.0; 6];
    let (horizon, monotone) = pitch_horizon(&cont, &cont, 5.0).unwrap();
    assert_eq!(horizon, 6);
    assert!(monotone, "a flat-zero error curve is monotone");
}

#[test]
fn a_flat_error_curve_is_monotone_despite_the_last_bit() {
    // The `-1e-12` slack in the monotone test earns its place here: two modes whose error differs
    // only in the last bit must not read as a curve that turns back on itself.
    let cont = [100.0, 100.0, 100.0];
    let disc = [100.5, 100.5 * (1.0 + 1e-16), 100.5];
    let (_, monotone) = pitch_horizon(&disc, &cont, 500.0).unwrap();
    assert!(
        monotone,
        "a last-bit wobble is not a non-monotone error curve"
    );
}

#[test]
fn a_genuine_dip_far_below_a_cent_is_still_reported_as_not_monotone() {
    // The other side of the slack above, added at the human's call (retirement plan §46.4): a slack
    // loosened from 1e-12 to 1e-3 cents was seen by nothing but the viewer's frozen payloads, which
    // compare the flag as a bool. A curve that really dips, by a billionth of a cent, must not read
    // as monotone. Together the two bars hold the slack between the last bit (~2.5e-13 cents at
    // these frequencies) and 1e-9.
    let cont = [100.0, 100.0, 100.0];
    let cents = [2.0, 2.0 - 1e-9, 2.0];
    let disc: Vec<f64> = cents
        .iter()
        .map(|c| 100.0 * 2.0_f64.powf(c / 1200.0))
        .collect();
    let read = pitch_error_cents(&disc, &cont).unwrap();
    assert!(
        read[0] - read[1] > 0.5e-9,
        "the fixture lost its dip in the rounding: {read:?}"
    );
    let (_, monotone) = pitch_horizon(&disc, &cont, 5.0).unwrap();
    assert!(
        !monotone,
        "a dip of 1e-9 cents is a real fall, not a last-bit wobble"
    );
}

#[test]
fn a_nonpositive_cents_bound_is_refused() {
    for bad in [0.0, -1.0, f64::NAN] {
        let err = pitch_horizon(&[100.0], &[100.0], bad).unwrap_err();
        assert!(err.contains("must be positive"), "{bad}: {err}");
    }
}

// -- sinc_horizon_fraction -----------------------------------------------------------------------

#[test]
fn the_returned_fraction_actually_solves_the_equation_it_claims_to() {
    // The bar that needs no second implementation and no recorded number: whatever `brentq` came
    // back with, put it into `(sin u / u)^power` and check it lands on `2^(-cents/1200)`. This is
    // the whole definition, so a wrong bracket, a wrong power or a wrong target fails here.
    for &cents in &[0.5, 1.0, 5.0, 25.0, 100.0] {
        for power in 1..=4i64 {
            let frac = sinc_horizon_fraction(cents, power).unwrap();
            let u = frac * std::f64::consts::PI / 2.0;
            let got = (u.sin() / u).powi(power as i32);
            let want = 2.0_f64.powf(-cents / 1200.0);
            assert!(
                (got - want).abs() < 1e-12,
                "cents={cents} power={power}: sinc^{power}({u}) = {got}, wanted {want}"
            );
        }
    }
}

#[test]
fn a_plate_resolves_the_same_share_of_its_grid_as_a_string_at_half_the_cents() {
    // The identity in the docstring, and the reason the plate's horizon is not a separate story:
    // `sinc(u)^2 = 2^(-c/1200)` and `sinc(u) = 2^(-(c/2)/1200)` are the same equation.
    //
    // Exact in exact arithmetic and asserted on a tolerance anyway, because the two sides are two
    // separate root finds stopping at `brentq`'s own tolerance rather than one computation used
    // twice. The measured gap over these five bounds is ~1e-16 and the bar is four orders above it.
    for &cents in &[0.5, 1.0, 5.0, 25.0, 100.0] {
        let plate = sinc_horizon_fraction(cents, 2).unwrap();
        let string_half = sinc_horizon_fraction(cents / 2.0, 1).unwrap();
        assert!(
            (plate - string_half).abs() < 1e-12,
            "cents={cents}: plate {plate} vs string-at-half {string_half}"
        );
        let string = sinc_horizon_fraction(cents, 1).unwrap();
        assert!(
            plate < string,
            "cents={cents}: the plate must resolve less of its grid"
        );
    }
}

#[test]
fn the_published_grid_fractions_are_what_the_plan_quotes() {
    // Two derived numbers the plan cites in prose (§3.2) — a string resolving 8.378% of its grid at
    // five cents and a plate 5.925%. Pinned here so a reader who finds them in the document can see
    // where they come from, and so that changing the definition has to change these too.
    let string = sinc_horizon_fraction(5.0, 1).unwrap();
    let plate = sinc_horizon_fraction(5.0, 2).unwrap();
    assert!((string - 0.083_78).abs() < 1e-5, "string fraction {string}");
    assert!((plate - 0.059_250).abs() < 1e-5, "plate fraction {plate}");
}

#[test]
fn a_tighter_bound_resolves_less_of_the_grid() {
    // Monotone in `cents`, which is the property that makes "the horizon" a meaningful thing to ask
    // for at all: a stricter tuning requirement can never buy you more modes.
    let mut prev = 0.0;
    for &cents in &[0.1, 0.5, 1.0, 5.0, 25.0, 100.0] {
        let frac = sinc_horizon_fraction(cents, 1).unwrap();
        assert!(frac > prev, "cents={cents}: {frac} did not exceed {prev}");
        prev = frac;
    }
}

#[test]
fn the_sinc_fraction_refuses_a_nonpositive_bound_and_a_zeroth_power() {
    assert!(sinc_horizon_fraction(0.0, 1)
        .unwrap_err()
        .contains("must be positive"));
    assert!(sinc_horizon_fraction(-5.0, 1)
        .unwrap_err()
        .contains("must be positive"));
    assert!(sinc_horizon_fraction(5.0, 0)
        .unwrap_err()
        .contains("positive number of sinc"));
}

// -- mode_family and mode_block ------------------------------------------------------------------

#[test]
fn the_three_families_are_the_index_sequences_they_are_named_for() {
    assert_eq!(
        mode_family("axial", 4).unwrap(),
        vec![(1, 1), (2, 1), (3, 1), (4, 1)]
    );
    assert_eq!(
        mode_family("axial_y", 4).unwrap(),
        vec![(1, 1), (1, 2), (1, 3), (1, 4)]
    );
    assert_eq!(
        mode_family("diagonal", 4).unwrap(),
        vec![(1, 1), (2, 2), (3, 3), (4, 4)]
    );
}

#[test]
fn the_two_axial_families_are_transposes_of_one_another() {
    // On an isotropic square they are exactly degenerate — which is why the plan warns that a grain
    // makes them two different measurements. The transpose relation is the thing that degeneracy
    // rests on, so assert it rather than the degeneracy.
    let x = mode_family("axial", 6).unwrap();
    let y = mode_family("axial_y", 6).unwrap();
    let flipped: Vec<(i64, i64)> = y.iter().map(|&(m, n)| (n, m)).collect();
    assert_eq!(x, flipped);
}

#[test]
fn an_unknown_family_and_an_empty_one_are_both_refused() {
    assert!(mode_family("axial_x", 4)
        .unwrap_err()
        .contains("unknown mode family"));
    assert!(mode_family("axial", 0)
        .unwrap_err()
        .contains("at least one mode"));
    assert!(mode_block(0)
        .unwrap_err()
        .contains("at least one mode per axis"));
}

#[test]
fn a_block_is_the_whole_square_of_indices_ordered_by_continuum_frequency() {
    for m_max in 1..=6i64 {
        let block = mode_block(m_max).unwrap();
        assert_eq!(block.len() as i64, m_max * m_max, "m_max={m_max}");
        let mut keys: Vec<(i64, i64, i64)> =
            block.iter().map(|&(m, n)| (m * m + n * n, m, n)).collect();
        let sorted = {
            let mut k = keys.clone();
            k.sort();
            k
        };
        assert_eq!(
            keys, sorted,
            "m_max={m_max}: the block came back out of order"
        );
        keys.dedup();
        assert_eq!(
            keys.len() as i64,
            m_max * m_max,
            "m_max={m_max}: a repeated index pair"
        );
    }
}

#[test]
fn the_ordering_key_breaks_its_ties_by_index_and_not_by_sort_stability() {
    // `(1, 2)` and `(2, 1)` are degenerate on a square, so their relative order is decided by the
    // tie-break and by nothing else. Written down here because a reader comparing this list to a
    // Python `sorted` needs to know the rule is stated rather than inherited.
    let block = mode_block(2).unwrap();
    assert_eq!(block, vec![(1, 1), (1, 2), (2, 1), (2, 2)]);
}

// -- block_weight and the corner argument --------------------------------------------------------

#[test]
fn the_block_weight_dips_at_a_closed_form_position_in_n() {
    // The claim the whole corner argument rests on, and it is sharper than "there is a dip
    // somewhere". Minimising `(m⁴ + t²)/(m² + t)` over `t = n²` gives `t* = m²(√2 − 1)`, so the
    // weight's minimum sits at `n* = m·√(√2 − 1) ≈ 0.6436 m` — always strictly inside `(0, m)`,
    // which is why the maximum over a block can never be interior.
    //
    // On the INTEGER grid that interior minimum is only reachable from `m = 3` up. At `m = 2` the
    // continuous minimiser is 1.287 and the nearest index below it is `n = 1`, which is the edge of
    // the block, so the discrete argmin lands on the boundary. The corner argument survives that
    // untouched — it is about the maximum, not the minimum — but a reader checking the docstring's
    // "interior minimum in n" against `m = 2` will find the edge, and this is where that is
    // written down.
    let ratio = ((2.0_f64).sqrt() - 1.0).sqrt();
    for m in 2..=12i64 {
        let ws: Vec<f64> = (1..=40).map(|n| block_weight(m, n).unwrap()).collect();
        let argmin = ws
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i64
            + 1; // back to a 1-based mode index
        let star = m as f64 * ratio;
        assert!(
            argmin == star.floor() as i64 || argmin == star.ceil() as i64,
            "m={m}: the weight dips at n={argmin}, not at an integer next to n*={star:.3}"
        );
        assert!(
            (m == 2) == (argmin == 1),
            "m={m}: the discrete minimum is at the block edge, and only m=2 may be"
        );
    }
}

#[test]
fn a_blocks_worst_mode_is_always_one_of_its_corners() {
    // The property that makes a block readable at all: its horizon is some corner family's
    // horizon. If the maximum ever sat inside, no family's prefix would describe the block.
    for m_max in 2..=12i64 {
        let block = mode_block(m_max).unwrap();
        let worst = block
            .iter()
            .max_by(|a, b| {
                block_weight(a.0, a.1)
                    .unwrap()
                    .partial_cmp(&block_weight(b.0, b.1).unwrap())
                    .unwrap()
            })
            .copied()
            .unwrap();
        let corners = [(m_max, 1), (1, m_max), (m_max, m_max)];
        assert!(
            corners.contains(&worst),
            "m_max={m_max}: the worst mode is {worst:?}, which is not a corner"
        );
    }
}

#[test]
fn the_weight_is_symmetric_in_its_two_indices() {
    // `(m⁴ + n⁴)/(m² + n²)` cannot tell the axes apart, which is the isotropy the plan says a grain
    // destroys. Exact rather than approximate: the two expressions are the same additions in the
    // opposite order, and floating-point addition is commutative.
    for m in 1..=10i64 {
        for n in 1..=10i64 {
            assert_eq!(block_weight(m, n).unwrap(), block_weight(n, m).unwrap());
        }
    }
}

#[test]
fn the_weight_refuses_a_zero_index_where_the_courant_number_accepts_one() {
    // The asymmetry between these two is deliberate and easy to read as a bug. `n = 0` is a
    // meaningful 1-D degenerate case for `cancellation_courant` and is not one here: the weight of
    // a mode that does not exist is not a number anyone should be handed.
    assert!(block_weight(1, 0).unwrap_err().contains("starts at 1"));
    assert!(block_weight(0, 1).unwrap_err().contains("starts at 1"));
    assert!(cancellation_courant(1, 0).is_ok());
}

// -- cancellation_courant ------------------------------------------------------------------------

#[test]
fn the_one_dimensional_case_is_exactly_the_cfl_limit() {
    // Exact equality, and this is a case where that is a statement about the arithmetic rather than
    // a hope. With `n = 0` the expression is `√(m⁴) / m²`; `m⁴` is an integer well inside the range
    // a double represents exactly, `sqrt` of an exact square is exact, and `x / x` is 1.0. Every
    // rounding on the path is a no-op, which is the question to ask before writing `==`.
    for m in 1..=64i64 {
        assert_eq!(cancellation_courant(m, 0).unwrap(), 1.0, "m={m}");
    }
}

#[test]
fn the_diagonal_sits_on_the_two_dimensional_cfl_ceiling_for_every_mode() {
    // The claim that makes the membrane's "magic Courant number" not a coincidence: the ceiling is
    // where the diagonal family cancels, at every mode number rather than at one.
    for m in 1..=64i64 {
        let lam = cancellation_courant(m, m).unwrap();
        assert!(
            (lam - CFL_2D).abs() < 1e-15,
            "m={m}: {lam} against the ceiling {CFL_2D}"
        );
    }
}

#[test]
fn no_mode_of_a_stable_membrane_is_ever_sharp() {
    // `λ ≤ 1/√2 ≤ cancellation_courant(m, n)` for every mode, so a membrane run at or below its own
    // stability ceiling has every mode flat or (the diagonal, at the ceiling) exact. This is the
    // second half of that chain and the one that is not the CFL condition.
    for m in 1..=24i64 {
        for n in 0..=24i64 {
            let lam = cancellation_courant(m, n).unwrap();
            assert!(
                lam >= CFL_2D - 1e-15,
                "mode ({m}, {n}) cancels at {lam}, below the ceiling {CFL_2D} — it would be sharp"
            );
        }
    }
}

#[test]
fn the_ceiling_is_the_minimum_of_this_function_over_the_whole_spectrum() {
    // Stated in the docstring as `t² + (1 - t)²` minimised at `t = 1/2`, which is the diagonal. The
    // previous test says nothing dips below the ceiling; this one says the ceiling is *attained*,
    // so it is the minimum rather than merely a lower bound.
    let mut best = f64::INFINITY;
    for m in 1..=24i64 {
        for n in 0..=24i64 {
            best = best.min(cancellation_courant(m, n).unwrap());
        }
    }
    assert!(
        (best - CFL_2D).abs() < 1e-15,
        "the spectrum's minimum is {best}, not {CFL_2D}"
    );
}

#[test]
fn the_axial_family_climbs_toward_an_unreachable_one() {
    // It rises toward `λ = 1`, which is above the 2-D ceiling and therefore never run. That is why
    // the ceiling buys the diagonal family the whole grid and the axial family nothing.
    let mut prev = 0.0;
    for m in 1..=64i64 {
        let lam = cancellation_courant(m, 1).unwrap();
        assert!(
            lam >= prev,
            "m={m}: the axial family stopped climbing at {lam}"
        );
        assert!(
            lam < 1.0,
            "m={m}: the axial family reached {lam}, which is the 1-D limit"
        );
        prev = lam;
    }
    assert!(
        prev > 0.999,
        "the axial family should approach 1; it reached {prev}"
    );
}

#[test]
fn the_courant_number_refuses_a_zero_first_index_and_a_negative_second() {
    assert!(cancellation_courant(0, 1)
        .unwrap_err()
        .contains("starts at 1"));
    assert!(cancellation_courant(1, -1)
        .unwrap_err()
        .contains("use n = 0"));
}

// =================================================================================================
// The schemes' horizons — carried from `tests/test_resolution_horizon.py` (retirement plan §46)
// =================================================================================================
//
// Everything above checks a primitive against its own definition. Everything below asks the
// primitives about a SCHEME — the string, the plate, the grained plate and the membrane — through
// `modal`'s closed-form dispersion relations, with no model built: the eigenvalues are closed form,
// so assembling a quarter-million-unknown operator to read them back would answer the same question
// slowly. The seven bars that read a BUILT model's `k`, `h`, `θ` or `c` live in
// `crates/physsynth-core/tests/horizon_models.rs`, because this crate cannot construct one.
//
// **Nothing here is a frozen integer.** A horizon is a property of `(model, N, k, params)`, so what
// is asserted is family behaviour: which way a horizon moves when a knob turns, that one family has
// a floor no sample rate passes, that the other's errors cancel at a magic Courant number, and the
// closed forms the measurement can be checked against from outside itself. The two mechanisms, so a
// reader does not assume the first is universal:
//
// * **Implicit θ-scheme** (stiff and damped strings, both plates, the beam). The time error gives
//   `ω/ω_c = 1/√(1 + θk²Q) < 1` and the spatial operator `sinc(u) < 1`. Both flatten, so they
//   COMPOUND: there is a hard space floor and refining the timestep cannot pass it.
// * **Explicit leapfrog** (ideal string, membrane). `sin(ωk/2) = λ sin(mπh/2L)`, whose time factor
//   is SHARP and cancels the spatial droop exactly at `λ = 1` (and along the diagonal at `λ = 1/√2`
//   in 2-D). Here refining the timestep moves the horizon the WRONG way.
//
// Three spellings in the fixtures below are the Python's on purpose, each measured:
//
// * `π⁴` is `PI * PI * PI * PI`, left to right. That is what `np.pi**4` returns (97.40909103400242);
//   `PI.powi(4)` squares twice and lands one ulp higher.
// * the 2-D ceiling is `1.0 / 2.0_f64.sqrt()`, the Python's `1.0 / np.sqrt(2.0)`, and NOT this
//   file's `CFL_2D` (`√2/2`), which is one ulp away. The two meet in a `< 1e-15` bar, so the ulp is
//   inside it either way; the spelling is kept so the carried bars compare what the Python compared.
// * every `argmax` is the FIRST maximum, as `np.argmax` and Python's `max` are. Rust's `max_by`
//   returns the LAST, and the two axial corners `(M, 1)` and `(1, M)` are exact twins, so a named
//   witness would otherwise depend on which way an iterator breaks a tie.

use physsynth_analysis::modal;
use physsynth_analysis::root::brentq;
use std::f64::consts::PI;

/// `tests/helpers.py`'s canonical string and plate: L = 1 m, T = 200 N, ρ = 0.005 kg/m (c = 200
/// m/s), the plate's κ = 20 at the θ-scheme's default θ = 0.28, the membrane's areal ρ = 0.005.
const L_DEFAULT: f64 = 1.0;
const T_DEFAULT: f64 = 200.0;
const RHO_DEFAULT: f64 = 0.005;
const RHO_AREAL_DEFAULT: f64 = 0.005;
const KAPPA_PLATE: f64 = 20.0;
const PLATE_THETA: f64 = 0.28;
/// A sustained tone's pitch JND is around this; the bound is an argument, not a law.
const CENTS: f64 = 5.0;
/// `scipy.optimize.brentq`'s defaults, which the Python's root finds relied on by not passing them.
const SCIPY_XTOL: f64 = 2e-12;
const SCIPY_RTOL: f64 = 8.881_784_197_001_252e-16;
const SCIPY_MAXITER: usize = 100;

fn mem_ceiling() -> f64 {
    1.0 / 2.0_f64.sqrt()
}

/// Every Courant number the suite runs a membrane at.
fn mem_lams() -> [f64; 5] {
    [mem_ceiling(), 0.7, 0.6, 0.5, 0.45]
}

const MEM_GRIDS: [i64; 5] = [64, 128, 256, 512, 1024];
const MEM_BLOCKS: std::ops::RangeInclusive<i64> = 2..=12;

fn wave_speed() -> f64 {
    (T_DEFAULT / RHO_DEFAULT).sqrt()
}

/// The index of the FIRST maximum, as `np.argmax` returns it (see the section header).
fn argmax_first(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn abs_cents(f_disc: &[f64], f_cont: &[f64]) -> Vec<f64> {
    pitch_error_cents(f_disc, f_cont)
        .unwrap()
        .into_iter()
        .map(f64::abs)
        .collect()
}

/// `np.linspace(start, stop, num)`: `i * step + start`, the last entry overwritten by `stop`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

/// The horizon of the SPATIAL operator alone, about the canonical string — `tests/helpers.py`'s
/// `spatial_operator_horizon`, kept a test fixture rather than a library function for the reason
/// that file gave: it hard-codes `L`, `c` and a 1-D Dirichlet second difference, so in the library
/// it would answer about the default string whatever the caller meant (resolution-horizon plan
/// §7.7). `k` appears nowhere: this is what is left when the timestep is refined to nothing.
fn spatial_operator_horizon(n: i64, kappa: f64, cents: f64) -> (usize, bool) {
    let h = L_DEFAULT / n as f64;
    let c = wave_speed();
    let (mut w_disc, mut w_cont) = (Vec::new(), Vec::new());
    for m in 1..n {
        let p2_disc = modal::dirichlet_axis_eigenvalue(m as f64, L_DEFAULT, h);
        let p2_cont = (m as f64 * PI / L_DEFAULT).powi(2);
        w_disc.push((c * c * p2_disc + kappa * kappa * p2_disc * p2_disc).sqrt());
        w_cont.push((c * c * p2_cont + kappa * kappa * p2_cont * p2_cont).sqrt());
    }
    pitch_horizon(&w_disc, &w_cont, cents).unwrap()
}

/// The explicit ideal string's horizon at Courant number `lam`, from its own dispersion relation.
fn ideal_horizon(n: i64, lam: f64) -> (usize, bool) {
    let c = wave_speed();
    let f_disc: Vec<f64> = (1..n)
        .map(|m| modal::discrete_mode_frequency(c, L_DEFAULT, n, lam, m))
        .collect();
    let f_cont: Vec<f64> = (1..n).map(|m| m as f64 * c / (2.0 * L_DEFAULT)).collect();
    pitch_horizon(&f_disc, &f_cont, CENTS).unwrap()
}

/// `(f_discrete, f_continuum)` for a square supported plate, analytically. `k = μh²/κ` is
/// `make_plate`'s sample rate inverted.
fn plate_family_frequencies(n: i64, mu: f64, modes: &[(i64, i64)]) -> (Vec<f64>, Vec<f64>) {
    let h = L_DEFAULT / n as f64;
    let k = mu * h * h / KAPPA_PLATE;
    let f_disc = modal::rectangular_discrete_eigenvalues(h, n, n, modes)
        .into_iter()
        .map(|lam| modal::discrete_plate_eigenfrequency(lam, KAPPA_PLATE, k, PLATE_THETA))
        .collect();
    let f_cont = modal::rectangular_plate_freqs(KAPPA_PLATE, L_DEFAULT, L_DEFAULT, modes);
    (f_disc, f_cont)
}

/// `(f_discrete, f_continuum)` for a square membrane, analytically: `k = λh/c` is `make_membrane`'s
/// sample rate inverted. Deliberately not the BUILT path in `horizon_models.rs`, which round-trips
/// through `fs = c/(λh)` and moves the answer in the last few bits; the seam between the two is
/// asserted there (`the_analytic_membrane_path_agrees_with_a_built_membrane`).
fn membrane_frequencies(n: i64, lam: f64, modes: &[(i64, i64)]) -> (Vec<f64>, Vec<f64>) {
    let c = (T_DEFAULT / RHO_AREAL_DEFAULT).sqrt();
    let h = L_DEFAULT / n as f64;
    let k = lam * h / c;
    let f_disc = modal::rectangular_discrete_eigenvalues(h, n, n, modes)
        .into_iter()
        .map(|e| modal::discrete_membrane_eigenfrequency(e, c, k))
        .collect();
    let f_cont = modal::rectangular_membrane_freqs(c, L_DEFAULT, L_DEFAULT, modes);
    (f_disc, f_cont)
}

fn membrane_cents(n: i64, lam: f64, modes: &[(i64, i64)]) -> Vec<f64> {
    let (d, c) = membrane_frequencies(n, lam, modes);
    pitch_error_cents(&d, &c).unwrap()
}

fn sinc_squared(n: i64) -> Vec<f64> {
    (1..n)
        .map(|m| {
            let u = m as f64 * PI / (2.0 * n as f64);
            (u.sin() / u).powi(2)
        })
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// -- the primitive, and the case the conservative reading exists for -------------------------------

#[test]
fn the_prefix_and_the_last_mode_inside_differ_exactly_where_the_flag_says_so() {
    // Modes 1-2 in tune, mode 3 far out, mode 4 back in tune. "The last mode inside the bound" is
    // 4; the leading prefix is 2. They differ exactly here, which is why the flag is returned.
    let cont = [100.0, 200.0, 300.0, 400.0];
    let disc: Vec<f64> = cont
        .iter()
        .zip([1.0, 1.0, 0.9, 1.0])
        .map(|(c, r)| c * r)
        .collect();
    let (horizon, monotone) = pitch_horizon(&disc, &cont, CENTS).unwrap();
    assert_eq!(
        horizon, 2,
        "the prefix must stop at the first mode outside the bound"
    );
    assert!(
        !monotone,
        "a non-monotone error curve must be reported, not hidden behind one int"
    );
    let last_inside = pitch_error_cents(&disc, &cont)
        .unwrap()
        .iter()
        .rposition(|e| e.abs() <= CENTS)
        .unwrap()
        + 1;
    assert_eq!(last_inside, 4);
    assert_ne!(last_inside, horizon);
}

#[test]
fn the_horizon_refuses_mismatched_families_rather_than_zipping_them_short() {
    // `pitch_error_cents` had this bar and `pitch_horizon` did not; the horizon reaches the same
    // guard through it, and a family of two read against a family of one is a caller error.
    let err = pitch_horizon(&[100.0, 200.0], &[100.0], CENTS).unwrap_err();
    assert!(err.contains("shape mismatch"), "{err}");
}

// -- the one closed form: the space floor is checkable from OUTSIDE the measurement ----------------

#[test]
fn the_wave_space_floor_matches_its_closed_form() {
    // The measured floor against `sin(u)/u = 2^(-cents/1200)`, analytic rather than recorded. The
    // tolerance is one mode: the horizon is an integer count and the closed form is not, so they can
    // differ by the quantisation and no more. Measured: the gap runs -0.36 to -0.98 modes.
    for cents in [5.0, 25.0, 100.0] {
        for n in [64, 128, 256, 512, 1024] {
            let (horizon, monotone) = spatial_operator_horizon(n, 0.0, cents);
            assert!(
                monotone,
                "a pure wave's sinc droop is monotone in the mode index"
            );
            let predicted = sinc_horizon_fraction(cents, 1).unwrap() * n as f64;
            assert!(
                (horizon as f64 - predicted).abs() <= 1.0,
                "N={n}, {cents} cents: measured floor {horizon} against the closed form \
                 {predicted:.2} — more than the integer quantisation apart"
            );
        }
    }
}

#[test]
fn the_wave_space_floor_is_a_fraction_of_the_grid_and_not_a_frequency() {
    // Doubling `N` doubles the floor; nothing about `c`, `L` or `fs` enters it. The bar is DERIVED:
    // on a grid of `N` the fraction can only be read to `1/N`, so the coarsest grid sets how much
    // spread a genuinely constant fraction may show. Measured spread 0.0049 against 0.0078.
    let grids = [128, 256, 512, 1024];
    let fractions: Vec<f64> = grids
        .iter()
        .map(|&n| spatial_operator_horizon(n, 0.0, CENTS).0 as f64 / n as f64)
        .collect();
    let spread = fractions.iter().cloned().fold(f64::MIN, f64::max)
        - fractions.iter().cloned().fold(f64::MAX, f64::min);
    let quantisation = 1.0 / 128.0;
    assert!(
        spread <= quantisation,
        "the floor is not a constant fraction of the grid: {fractions:?} spread {spread:.5} \
         against the coarsest grid's own quantisation {quantisation:.5}"
    );
}

#[test]
fn stiffness_lowers_the_floor_because_the_biharmonic_errs_at_fourth_power() {
    // `Q = c²p² + κ²p⁴`, and the `p⁴` term carries twice the sinc droop. So a stiffer string
    // resolves a SMALLER share of its own grid, and the share keeps shrinking under refinement.
    let share = |n: i64, kappa: f64| spatial_operator_horizon(n, kappa, CENTS).0 as f64 / n as f64;
    let (k0, k2, k8) = (share(512, 0.0), share(512, 2.0), share(512, 8.0));
    assert!(
        k0 > k2 && k2 > k8,
        "stiffness did not lower the floor: {k0} {k2} {k8}"
    );
    let (coarse, fine) = (share(128, 8.0), share(1024, 8.0));
    assert!(
        fine < coarse,
        "a stiff string's resolved share should shrink under refinement: {coarse} -> {fine}"
    );
}

// -- family 2, the explicit leapfrog: the two errors CANCEL, and only at one Courant number --------

#[test]
fn the_explicit_string_beats_its_own_space_floor_at_lambda_one() {
    // An implicit scheme can never pass the space floor. The explicit one does, by an order of
    // magnitude, because at `λ = 1` its sharp time error cancels the flat space error identically.
    let n = 256;
    let (horizon, _) = ideal_horizon(n, 1.0);
    let floor = spatial_operator_horizon(n, 0.0, CENTS).0;
    assert!(
        horizon > 8 * floor,
        "lambda = 1 resolved {horizon} of {} modes against a space floor of {floor}",
        n - 1
    );
    assert!(
        horizon as i64 >= n - 2,
        "lambda = 1 is the dispersionless case — essentially every mode"
    );
}

#[test]
fn refining_the_explicit_timestep_makes_the_string_worse() {
    // Below `λ = 1` the cancellation is partial and gets less complete as `λ` falls: lowering the
    // Courant number for safety buys nothing and costs pitch. Measured 255, 131, 66, 48, 32, 24.
    let lams = [1.0, 0.99, 0.95, 0.9, 0.75, 0.5];
    let horizons: Vec<usize> = lams.iter().map(|&lam| ideal_horizon(256, lam).0).collect();
    assert!(
        horizons.windows(2).all(|w| w[1] <= w[0]),
        "the horizon did not fall monotonically as lambda fell: {horizons:?}"
    );
    assert!(
        (horizons[5] as f64) < horizons[0] as f64 / 5.0,
        "halving the Courant number should cost most of the band: {horizons:?}"
    );
}

// -- the 2-D spectrum split by mode family: the plate ------------------------------------------------

#[test]
fn the_plate_diagonal_family_is_the_strings_spatial_droop_squared() {
    // A plate's frequency goes like the Laplacian eigenvalue rather than its square root, so its
    // spatial droop is `sinc(u)²`. Along the DIAGONAL that is exact (both axes carry the same `u`);
    // the AXIAL family `(m, 1)` carries an undrooped `p₁²` and only approaches it, like `1/N²`.
    for n in [64, 256] {
        let h = L_DEFAULT / n as f64;
        let sinc_sq = sinc_squared(n);
        let diagonal: Vec<f64> = modal::rectangular_discrete_eigenvalues(
            h,
            n,
            n,
            &mode_family("diagonal", n - 1).unwrap(),
        )
        .iter()
        .zip(1..n)
        .map(|(lam, m)| lam / (2.0 * (m as f64 * PI / L_DEFAULT).powi(2)))
        .collect();
        let dev = max_abs_diff(&diagonal, &sinc_sq);
        assert!(
            dev < 1e-14,
            "N={n}: the diagonal family's eigenvalue ratio is sinc(u)^2 exactly, off by {dev:e}"
        );
        let axial: Vec<f64> =
            modal::rectangular_discrete_eigenvalues(h, n, n, &mode_family("axial", n - 1).unwrap())
                .iter()
                .zip(1..n)
                .map(|(lam, m)| lam / ((m * m + 1) as f64 * (PI / L_DEFAULT).powi(2)))
                .collect();
        let gap = max_abs_diff(&axial, &sinc_sq);
        assert!(
            0.0 < gap && gap < 1e-3,
            "N={n}: the axial family should be near sinc^2, not equal to it: {gap:e}"
        );
    }
}

#[test]
fn the_plate_space_floor_matches_its_closed_form() {
    // The plate's analogue of the string's closed-form bar, measured against `sinc(u)²` with `μ`
    // taken to nothing. The tolerance is the integer quantisation and the second assertion fixes the
    // DIRECTION: a timestep can only cost modes. Measured 2026-09-07 and again here,
    // `predicted - horizon` runs 0.168 to 0.948 — clear of zero and of one.
    //
    // The direction bar's stated mechanism is not the whole reason it passes: the axial family's
    // SPACE floor is itself a hair above `sinc²` (its undrooped cross-axis term dilutes the droop),
    // and on an isotropic plate that excess is under a third of a mode. Put a grain on the plate and
    // the soft axis crosses — `the_closed_form_stops_being_an_upper_bound_once_the_plate_has_a_grain`.
    for kind in ["diagonal", "axial"] {
        for n in [64, 128, 256, 512] {
            for cents in [1.0, 5.0, 25.0] {
                let modes = mode_family(kind, n - 1).unwrap();
                let (d, c) = plate_family_frequencies(n, 1e-5, &modes);
                let (horizon, monotone) = pitch_horizon(&d, &c, cents).unwrap();
                assert!(
                    monotone,
                    "the {kind} family's droop is monotone in the mode index"
                );
                let predicted = sinc_horizon_fraction(cents, 2).unwrap() * n as f64;
                assert!(
                    (horizon as f64 - predicted).abs() <= 1.0,
                    "{kind}, N={n}, {cents} cents: measured floor {horizon} against the closed \
                     form {predicted:.2}"
                );
                assert!(
                    horizon as f64 <= predicted,
                    "{kind}, N={n}: measured {horizon} ABOVE the space floor {predicted:.2}, which \
                     a timestep cannot buy"
                );
            }
        }
    }
}

#[test]
fn the_plates_two_families_agree_in_index_unlike_the_membranes() {
    // The implicit plate has no magic Courant number to cancel at, so both families sit on the same
    // `sinc²` floor and have the SAME horizon in their own mode index. A `k -> 0` claim, so asserted
    // over four decades of `μ`, and its margin measured: the shared horizon must sit clear of an
    // integer boundary on both sides in both families. Worst measured 0.0438 cents at (1 cent,
    // N = 256), against a bar of 0.02.
    for n in [64, 128, 256, 512] {
        for cents in [1.0, 5.0, 25.0] {
            let mut horizon = 0;
            for mu in [1e-5, 1e-3, 1e-1] {
                let h: Vec<usize> = ["diagonal", "axial"]
                    .iter()
                    .map(|kind| {
                        let modes = mode_family(kind, n - 1).unwrap();
                        let (d, c) = plate_family_frequencies(n, mu, &modes);
                        pitch_horizon(&d, &c, cents).unwrap().0
                    })
                    .collect();
                assert_eq!(
                    h[0], h[1],
                    "N={n}, {cents} cents, mu={mu}: the families should be identical at the floor"
                );
                horizon = h[0];
            }
            let mut margins = Vec::new();
            for kind in ["diagonal", "axial"] {
                let (d, c) = plate_family_frequencies(n, 1e-5, &mode_family(kind, n).unwrap());
                let err = abs_cents(&d, &c);
                margins.push(cents - err[horizon - 1]);
                margins.push(err[horizon] - cents);
            }
            let worst = margins.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                worst > 0.02,
                "N={n}, {cents} cents: the shared horizon sits {worst:.4} cents from an integer \
                 boundary — too close to call the two families equal rather than adjacent"
            );
        }
    }
}

#[test]
fn a_finite_timestep_breaks_the_family_tie_toward_the_axial_one() {
    // At index `m` the axial mode sits at about half the diagonal's frequency, so it takes about
    // half the time droop: raise `k` and the axial family reaches further. Measured 54 against 44,
    // under a floor of 67.
    let (n, cents) = (512, 25.0);
    let horizon = |kind: &str, mu: f64| {
        let (d, c) = plate_family_frequencies(n, mu, &mode_family(kind, n - 1).unwrap());
        pitch_horizon(&d, &c, cents).unwrap().0
    };
    let (diagonal, axial) = (horizon("diagonal", 2.0), horizon("axial", 2.0));
    assert!(
        axial > diagonal,
        "the time droop should cost the diagonal family more: {diagonal} {axial}"
    );
    let floor = horizon("diagonal", 1e-5);
    assert!(
        axial < floor,
        "neither family may pass the space floor {floor}: {axial}"
    );
}

#[test]
fn the_plates_families_differ_by_a_factor_of_two_in_pitch() {
    // At a fixed continuum frequency `ρ² = m² + n²`, the droop weight `(m⁴ + n⁴)/(m² + n²)` is
    // maximal on the axis and minimal on the diagonal, by a factor of two. So at equal pitch an axial
    // mode is twice as flat, approached from below; a finite timestep adds a family-independent
    // droop and pulls the ratio toward one. Measured 1.9675 / 1.9926 / 1.9987 at the floor.
    for m in [8, 16, 32] {
        let axial_m = (m as f64 * 2.0_f64.sqrt()).round() as i64;
        let ratio = |mu: f64| {
            let (d, c) = plate_family_frequencies(512, mu, &[(m, m), (axial_m, 1)]);
            let err = abs_cents(&d, &c);
            err[1] / err[0] * (c[0] / c[1])
        };
        let floor_ratio = ratio(1e-5);
        assert!(
            (floor_ratio - 2.0).abs() < 0.05,
            "({m},{m}) against ({axial_m},1): the axial mode should be twice as flat at equal \
             pitch, got {floor_ratio:.4}"
        );
        assert!(
            ratio(2.0) < floor_ratio,
            "a finite timestep must pull the ratio toward one"
        );
    }
}

#[test]
fn the_sorted_2d_spectrum_is_not_monotone_so_its_prefix_is_not_a_horizon() {
    // Sorted by frequency the plate's spectrum interleaves the families and the error does not rise
    // with pitch: `(3,1)` is LOWER in frequency than `(2,3)` and WORSE in error. `pitch_horizon`
    // still returns an integer over such a list, and its flag is the only thing saying so.
    for n in [96, 256] {
        let modes = mode_block(8).unwrap();
        let (d, c) = plate_family_frequencies(n, 0.5, &modes);
        let (_, monotone) = pitch_horizon(&d, &c, CENTS).unwrap();
        assert!(
            !monotone,
            "N={n}: a frequency-sorted 2-D spectrum is not monotone in pitch error"
        );
        let err = abs_cents(&d, &c);
        let fall = err.windows(2).position(|w| w[1] - w[0] < 0.0).unwrap();
        assert_eq!(
            (modes[fall], modes[fall + 1]),
            ((3, 1), (2, 3)),
            "N={n}: the first fall should be the named witness"
        );
        assert!(err[fall] > err[fall + 1]);
    }
}

#[test]
fn a_2d_blocks_worst_mode_is_its_diagonal_corner() {
    // The weight `(m⁴ + n⁴)/(m² + n²)` has an interior minimum in `n`, so its maximum over a block
    // sits at a corner, and the diagonal corner `(M, M)` beats the axial `(M, 1)` for every
    // `M >= 2`. So a block's horizon IS its diagonal family's, and that has a prefix. Measured: the
    // corner leads the runner-up by 2.4% at the thinnest (`M = 8`, `N = 256`).
    for n in [96, 256] {
        for m_max in 2..=8 {
            let modes = mode_block(m_max).unwrap();
            let (d, c) = plate_family_frequencies(n, 0.5, &modes);
            let worst = modes[argmax_first(&abs_cents(&d, &c))];
            assert_eq!(
                worst,
                (m_max, m_max),
                "N={n}: the {m_max}x{m_max} block's worst mode should be its diagonal corner"
            );
        }
    }
}

// -- the plate with a grain --------------------------------------------------------------------------
//
// In MODE INDEX all three families sit on the same `sinc²` floor the isotropic plate has, and the
// diagonal one sits on it exactly for any grain. In HERTZ the horizon is per-direction, because the
// same index is a different frequency on each axis. What the grain really costs is the isotropic
// plate's DIRECTION bar: `sinc_horizon_fraction(cents, 2) * N` stops being an upper bound.
//
// `_ortho_space_ratio` and `_ortho_crossing` form `q_disc / q_cont` HERE, from the axis eigenvalues
// alone, so the identities they check are about this test's formula and not about `modal.rs`. The
// library's two orthotropic oracles are reached only by `ortho_family_frequencies`, and the one bar
// that holds the two against each other is
// `the_integer_horizon_is_the_floor_of_that_crossing_which_is_where_a_one_mode_bar_breaks`.

#[derive(Clone, Copy)]
struct Grain {
    x: f64,
    cross: f64,
    y: f64,
}

const GRAIN_ISO: Grain = Grain {
    x: 1.0,
    cross: 1.0,
    y: 1.0,
};
/// About the spruce ratios.
const GRAIN_SPRUCE: Grain = Grain {
    x: 1.0,
    cross: 0.153,
    y: 0.0727,
};
/// Nothing is made of this.
const GRAIN_WILD: Grain = Grain {
    x: 11.0,
    cross: 2.5,
    y: 0.9,
};
/// `cross` must exceed `-√(x·y)` or the modal stiffness goes non-positive; this sits just inside that
/// guard, the ONLY fixture here that is not a plausible material, and the one that flips two signs.
const GRAIN_NEAR_GUARD: Grain = Grain {
    x: 1.0,
    cross: -0.9,
    y: 1.0,
};
const GRAINS: [(&str, Grain); 4] = [
    ("isotropic", GRAIN_ISO),
    ("spruce", GRAIN_SPRUCE),
    ("wild", GRAIN_WILD),
    ("near-guard", GRAIN_NEAR_GUARD),
];
/// Shared by every sweep below on purpose: widening this one list from four grids to nine is what
/// found the quantisation hazard at `N = 80`.
const GRAIN_GRIDS: [i64; 9] = [48, 64, 80, 96, 128, 160, 256, 384, 512];
const GRAIN_BOUNDS: [f64; 3] = [1.0, 5.0, 25.0];
const GRAIN_FAMILIES: [&str; 3] = ["diagonal", "axial", "axial_y"];

fn grain_ratio(n: i64, m: f64, nn: f64, g: Grain) -> f64 {
    let h = L_DEFAULT / n as f64;
    let lam_x = modal::dirichlet_axis_eigenvalue(m, L_DEFAULT, h);
    let lam_y = modal::dirichlet_axis_eigenvalue(nn, L_DEFAULT, h);
    let a = (m / L_DEFAULT).powi(2);
    let b = (nn / L_DEFAULT).powi(2);
    let q_disc = g.x * lam_x.powi(2) + 2.0 * g.cross * lam_x * lam_y + g.y * lam_y.powi(2);
    let q_cont = (PI * PI * PI * PI) * (g.x * a.powi(2) + 2.0 * g.cross * a * b + g.y * b.powi(2));
    (q_disc / q_cont).sqrt()
}

/// `f_disc / f_cont` in the `k -> 0` limit, taken ANALYTICALLY: the residual time droop at
/// `μ = 1e-5` is around `1e-10`, four orders above the `1e-15` identity the diagonal satisfies.
fn ortho_space_ratio(n: i64, modes: &[(i64, i64)], g: Grain) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, nn)| grain_ratio(n, m as f64, nn as f64, g))
        .collect()
}

/// The mode index where this family's space droop reaches `cents`, as a REAL number: everything in
/// the eigenvalue formula extends to a real index, so the quantisation can be taken out of the
/// comparison with the closed form. `brentq` at `xtol = 1e-12`, as the Python passed it.
fn ortho_crossing(kind: &str, n: i64, g: Grain, cents: f64) -> f64 {
    let err = |m: f64| {
        let (mm, nn) = match kind {
            "diagonal" => (m, m),
            "axial" => (m, 1.0),
            "axial_y" => (1.0, m),
            _ => unreachable!("{kind}"),
        };
        (1200.0 * grain_ratio(n, mm, nn, g).log2()).abs() - cents
    };
    brentq(
        err,
        1.0 + 1e-9,
        n as f64 - 1.0,
        1e-12,
        SCIPY_RTOL,
        SCIPY_MAXITER,
    )
    .unwrap()
}

/// `(f_discrete, f_continuum)` for a square orthotropic plate, through `modal`'s two oracles.
fn ortho_family_frequencies(
    n: i64,
    mu: f64,
    modes: &[(i64, i64)],
    g: Grain,
) -> (Vec<f64>, Vec<f64>) {
    let h = L_DEFAULT / n as f64;
    let k = mu * h * h / KAPPA_PLATE;
    let f_disc = modes
        .iter()
        .map(|&(m, nn)| {
            let lam_x = modal::dirichlet_axis_eigenvalue(m as f64, L_DEFAULT, h);
            let lam_y = modal::dirichlet_axis_eigenvalue(nn as f64, L_DEFAULT, h);
            modal::discrete_orthotropic_plate_eigenfrequency(
                lam_x,
                lam_y,
                KAPPA_PLATE,
                k,
                PLATE_THETA,
                g.x,
                g.cross,
                g.y,
            )
            .unwrap()
        })
        .collect();
    let f_cont =
        modal::orthotropic_plate_freqs(KAPPA_PLATE, L_DEFAULT, L_DEFAULT, modes, g.x, g.cross, g.y)
            .unwrap();
    (f_disc, f_cont)
}

/// The small-`u` pitch-error weight of mode `(m, n)`, grain and all: the isotropic weights collapse
/// it to `(m⁴ + n⁴)/(m² + n²)`. A proxy, so the bars that reason with it also check the measurement.
fn ortho_weight(m: i64, n: i64, g: Grain) -> f64 {
    let (mf, nf) = (m as f64, n as f64);
    let num = g.x * m.pow(6) as f64
        + g.cross * mf * mf * nf * nf * (m * m + n * n) as f64
        + g.y * n.pow(6) as f64;
    let den = g.x * m.pow(4) as f64 + 2.0 * g.cross * mf * mf * nf * nf + g.y * n.pow(4) as f64;
    num / den
}

fn closed_form(cents: f64, n: i64) -> f64 {
    sinc_horizon_fraction(cents, 2).unwrap() * n as f64
}

#[test]
fn the_grained_diagonal_droop_is_sinc_squared_for_any_grain() {
    // On a square the diagonal mode carries the same `u` on both axes, so every grain weight
    // multiplies the discrete and the continuum stiffness by the same factor and divides straight
    // back out. It holds even at `cross = -0.9`, where a `0/0` would be a fair worry. Measured worst
    // 1.2e-15 (near-guard, N = 512).
    for (name, g) in GRAINS {
        for n in [64, 512] {
            let ratio = ortho_space_ratio(n, &mode_family("diagonal", n - 1).unwrap(), g);
            let dev = max_abs_diff(&ratio, &sinc_squared(n));
            assert!(
                dev < 1e-14,
                "{name}, N={n}: the diagonal droop should be sinc(u)^2 independent of the grain, \
                 off by {dev:.3e}"
            );
        }
    }
}

#[test]
fn the_axial_deviation_from_sinc_squared_takes_the_sign_of_the_cross_term() {
    // `(m, 1)` carries a drooped `lam_x` against an undrooped `lam_y`, so the weights no longer
    // cancel. The sign of what is left is the sign of `cross`, the only term mixing the two: above
    // `sinc²` for every real wood, below it at the near-guard fixture. And a real grain splits the
    // two axes by an order of magnitude (spruce at N = 512: 4.8e-7 stiff, 6.7e-6 soft).
    for (name, g) in GRAINS {
        for n in [128, 512] {
            let sinc_sq = sinc_squared(n);
            let mut maxima = Vec::new();
            for kind in ["axial", "axial_y"] {
                let ratio = ortho_space_ratio(n, &mode_family(kind, n - 1).unwrap(), g);
                let dev: Vec<f64> = ratio.iter().zip(&sinc_sq).map(|(r, s)| r - s).collect();
                assert!(
                    dev[0].abs() <= 1e-14,
                    "mode (1,1) is on the floor by symmetry"
                );
                let rest = &dev[1..];
                let signed = if g.cross > 0.0 {
                    rest.iter().all(|&d| d > 0.0)
                } else {
                    rest.iter().all(|&d| d < 0.0)
                };
                assert!(
                    signed,
                    "{name}, {kind}, N={n}: the axial deviation from sinc^2 should everywhere \
                     take the sign of grain_cross ({})",
                    g.cross
                );
                maxima.push(rest.iter().map(|d| d.abs()).fold(0.0, f64::max));
            }
            let (stiff, soft) = (maxima[0], maxima[1]);
            if g.x == g.y {
                assert!(
                    (stiff - soft).abs() <= 1e-12 * soft,
                    "equal axes must not split"
                );
            } else {
                assert!(
                    soft > 5.0 * stiff,
                    "{name}, N={n}: the soft axis should deviate far more than the stiff one, got \
                     {soft:.3e} against {stiff:.3e}"
                );
            }
        }
    }
}

#[test]
fn the_grained_diagonal_crossing_is_the_closed_form_with_nothing_left_over() {
    // The strongest form of "one floor": not within a mode, EQUAL. The diagonal droop is `sinc²` for
    // any grain, and `sinc_horizon_fraction(cents, 2) * N` is where that reaches `cents`, so the
    // crossing is the closed form — over all 108 fixtures. Measured worst 4.6e-13 relative, which is
    // the root find's `xtol` against crossings of 1 to 30.
    for (name, g) in GRAINS {
        for cents in GRAIN_BOUNDS {
            for n in GRAIN_GRIDS {
                let predicted = closed_form(cents, n);
                let crossing = ortho_crossing("diagonal", n, g, cents);
                assert!(
                    (crossing - predicted).abs() <= 1e-12 * predicted,
                    "{name}, N={n}, {cents} cents: the diagonal crossing {crossing:.9} should BE \
                     the closed form {predicted:.9}"
                );
            }
        }
    }
}

#[test]
fn the_grained_axial_crossing_stays_within_half_a_mode_and_closes_like_one_over_n() {
    // The axial families mix a drooped axis with an undrooped one, so their crossing only approaches
    // the closed form: over the 108 fixtures the gap runs -0.199 to +0.399 modes, under half a mode,
    // signed as `cross` is. Stated in MODES, not as a fraction: at N = 48 the prediction is only 1.6
    // modes, and a percentage there would be about how small the prediction is.
    //
    // It closes like `1/N`, but that is ASYMPTOTIC and the shipped grids are not in the asymptote:
    // `gap * N` reaches its per-grain limit (18.9 isotropic, 39.7 spruce, 52.4 wild, -17.0
    // near-guard) only past N ~ 2000. So the rate is measured where it exists and the BOUND covers
    // the shipped grids. Thinnest sign margin measured: 0.0011 modes (spruce).
    for (name, g) in GRAINS {
        for cents in GRAIN_BOUNDS {
            for n in GRAIN_GRIDS {
                let predicted = closed_form(cents, n);
                for kind in ["axial", "axial_y"] {
                    let gap = ortho_crossing(kind, n, g, cents) - predicted;
                    assert!(
                        gap.abs() < 0.5,
                        "{name}, {kind}, N={n}, {cents} cents: the axial floor is {gap:+.4} modes \
                         from the isotropic closed form {predicted:.3}"
                    );
                    assert!(
                        (gap > 0.0) == (g.cross > 0.0) && gap != 0.0,
                        "{name}, {kind}, N={n}: the gap {gap:+.4} should take the sign of \
                         grain_cross"
                    );
                }
            }
        }
        let tail: Vec<f64> = [1024, 2048, 4096, 8192]
            .iter()
            .map(|&n| (ortho_crossing("axial_y", n, g, 1.0) - closed_form(1.0, n)) * n as f64)
            .collect();
        let (hi, lo) = (
            tail.iter().cloned().fold(f64::MIN, f64::max),
            tail.iter().cloned().fold(f64::MAX, f64::min),
        );
        assert!(
            hi / lo < 1.02,
            "{name}: gap*N should have settled by N=1024, got {tail:?}"
        );
        let coarse = (ortho_crossing("axial_y", 128, g, 1.0) - closed_form(1.0, 128)) * 128.0;
        assert!(
            coarse.abs() < tail[3].abs(),
            "{name}: the shipped grids should sit BELOW that limit, not at it ({coarse:.3} against \
             {:.3})",
            tail[3]
        );
    }
}

#[test]
fn the_integer_horizon_is_the_floor_of_that_crossing_which_is_where_a_one_mode_bar_breaks() {
    // `pitch_horizon` counts leading modes, so it returns exactly `floor(crossing)` — 324 of 324
    // fixtures. That makes the integer reading's distance from the closed form the sum of a real
    // quantity (the gap, under half a mode) and an artefact (where the prediction falls between
    // integers), and the artefact can dominate: at N = 80 and one cent the prediction is 2.120 and
    // the near-guard crossing 1.926, so a 0.19-mode deficit reads as a whole mode lost.
    //
    // This is also the one bar that holds `modal`'s two orthotropic oracles (through
    // `ortho_family_frequencies`) against the formula this file writes out (through
    // `ortho_crossing`). Measured: no crossing sits closer than 4.4e-4 of a mode to an integer.
    let mut exceptions = Vec::new();
    for (name, g) in GRAINS {
        for cents in GRAIN_BOUNDS {
            for n in GRAIN_GRIDS {
                let predicted = closed_form(cents, n);
                for kind in GRAIN_FAMILIES {
                    let modes = mode_family(kind, n - 1).unwrap();
                    let (d, c) = ortho_family_frequencies(n, 1e-5, &modes, g);
                    let (horizon, _) = pitch_horizon(&d, &c, cents).unwrap();
                    let crossing = ortho_crossing(kind, n, g, cents);
                    assert_eq!(
                        horizon,
                        crossing.floor() as usize,
                        "{name}, {kind}, N={n}, {cents} cents: the integer horizon should be \
                         floor({crossing:.4})"
                    );
                    if (horizon as f64 - predicted).abs() > 1.0 {
                        exceptions.push((name, kind, n, cents));
                    }
                }
            }
        }
    }
    assert!(
        exceptions
            .iter()
            .all(|&(_, _, n, cents)| n == 80 && cents == 1.0),
        "the quantisation hazard should be the recorded N=80 one-cent case, got {exceptions:?}"
    );
    assert_eq!(
        exceptions.len(),
        2,
        "both near-guard axial families cross the integer at N=80; got {exceptions:?}"
    );
}

#[test]
fn the_closed_form_stops_being_an_upper_bound_once_the_plate_has_a_grain() {
    // The isotropic plate's direction bar does not survive a grain. An axial family's SPACE floor
    // already sits a hair above `sinc²`; a grain roughly doubles that on the soft axis, and at coarse
    // grids it clears the integer. Eight of the 108 fixtures cross, all the `(1, n)` family of the
    // two positively-grained plates at N = 48, 64 and 128, by up to 0.304 modes.
    let mut crossings = Vec::new();
    for (name, g) in GRAINS {
        for cents in GRAIN_BOUNDS {
            for n in GRAIN_GRIDS {
                let predicted = closed_form(cents, n);
                for kind in GRAIN_FAMILIES {
                    let modes = mode_family(kind, n - 1).unwrap();
                    let (d, c) = ortho_family_frequencies(n, 1e-5, &modes, g);
                    let (horizon, _) = pitch_horizon(&d, &c, cents).unwrap();
                    if horizon as f64 > predicted {
                        crossings.push((name, g.cross, kind, n, horizon as f64 - predicted));
                    }
                }
            }
        }
    }
    assert!(
        !crossings.is_empty(),
        "no fixture exceeded the isotropic closed form; check the grain reaches the model first"
    );
    assert!(
        crossings.iter().all(|c| c.2 == "axial_y"),
        "only the soft-axis family should cross the closed form, got {crossings:?}"
    );
    assert!(
        crossings.iter().all(|c| c.1 > 0.0),
        "a negative cross term sits below the closed form and cannot cross it, got {crossings:?}"
    );
    assert!(
        crossings.iter().all(|c| c.4 < 1.0),
        "the excess is still under one mode"
    );
}

#[test]
fn the_two_axial_families_sit_a_root_stiffness_apart_in_hertz() {
    // Both axial families have the same horizon in their own index, and index `m` is not the same
    // frequency on the two axes: `f(m,1)/f(1,m) -> √(g_x/g_y)`, from below, like `1/m²`. So a
    // horizon quoted in hertz is a per-axis number. At m = 80 both grains are inside a cent of the
    // limit (measured 0.53 spruce, 0.69 wild).
    for (name, g) in [("spruce", GRAIN_SPRUCE), ("wild", GRAIN_WILD)] {
        let target = (g.x / g.y).sqrt();
        let mut orders = Vec::new();
        let mut last_ratio = 0.0;
        for m in [10, 20, 40, 80] {
            let f = |mode: (i64, i64)| {
                modal::orthotropic_plate_freqs(
                    KAPPA_PLATE,
                    L_DEFAULT,
                    L_DEFAULT,
                    &[mode],
                    g.x,
                    g.cross,
                    g.y,
                )
                .unwrap()[0]
            };
            let ratio = f((m, 1)) / f((1, m));
            let gap = target - ratio;
            assert!(
                gap > 0.0,
                "{name}, m={m}: the ratio approaches sqrt(g_x/g_y) from below"
            );
            orders.push(gap * m as f64 * m as f64);
            last_ratio = ratio;
        }
        let hi = orders.iter().cloned().fold(f64::MIN, f64::max);
        let lo = orders.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            hi / lo < 1.05,
            "{name}: the gap should close like 1/m^2, got gap*m^2 = {orders:?}"
        );
        let residual = 1200.0 * (target / last_ratio).log2();
        assert!(
            residual < 1.0,
            "{name}: by m=80 the ratio should be within a cent of sqrt(g_x/g_y), got {residual:.3}"
        );
    }
}

#[test]
fn a_grained_blocks_worst_mode_is_still_its_diagonal_corner() {
    // The corner argument survives every wood, and the grain narrows its margin: `ortho_weight(m,
    // m)` is EXACTLY `m²` for any grain while the axial corner stays below it. Checked against the
    // MEASURED error, because the weight is a small-`u` proxy; only `mode_block`'s set is used, its
    // order being isotropic. Thinnest lead measured: 0.34% (spruce, M = 8, N = 256).
    for (name, g) in [
        ("isotropic", GRAIN_ISO),
        ("spruce", GRAIN_SPRUCE),
        ("wild", GRAIN_WILD),
    ] {
        for n in [96, 256] {
            for m_max in 2..=8 {
                let modes = mode_block(m_max).unwrap();
                let (d, c) = ortho_family_frequencies(n, 0.5, &modes, g);
                let worst = modes[argmax_first(&abs_cents(&d, &c))];
                assert_eq!(
                    worst,
                    (m_max, m_max),
                    "{name}, N={n}: the {m_max}x{m_max} block's worst mode should be its diagonal \
                     corner"
                );
                let square = (m_max * m_max) as f64;
                assert!(
                    (ortho_weight(m_max, m_max, g) - square).abs() <= 1e-12 * square,
                    "the diagonal corner's weight is m^2 for any grain"
                );
                assert!(ortho_weight(m_max, 1, g) < square);
            }
        }
    }
}

#[test]
fn a_blocks_worst_error_is_grain_blind_in_space_and_not_at_a_finite_timestep() {
    // A block's worst mode is its diagonal corner and the diagonal droop is `sinc²` whatever the
    // grain, so the block's worst SPACE error is the same number on every plate. But only as
    // `k -> 0`: the time droop `1/√(1 + θk²Q)` reads the modal stiffness, which IS the grain, so at a
    // working timestep the grain returns, ordered by stiffness. Its size COLLAPSES onto
    // `(m_max/N)²` rather than being a number to hand-pick: over 20 fixtures the constant is
    // 44.2..45.2 (spread 2.3% against the 5% bar).
    let positive = [
        ("isotropic", GRAIN_ISO),
        ("spruce", GRAIN_SPRUCE),
        ("wild", GRAIN_WILD),
    ];
    let mut constants = Vec::new();
    for n in [96, 128, 256, 512] {
        for m_max in 2..=6 {
            let modes = mode_block(m_max).unwrap();
            let floor: Vec<f64> = positive
                .iter()
                .map(|&(_, g)| {
                    ortho_space_ratio(n, &modes, g)
                        .iter()
                        .map(|r| (1200.0 * r.log2()).abs())
                        .fold(0.0, f64::max)
                })
                .collect();
            let spread = floor.iter().cloned().fold(f64::MIN, f64::max)
                - floor.iter().cloned().fold(f64::MAX, f64::min);
            assert!(
                spread < 1e-12,
                "N={n}, {m_max}x{m_max}: a block's SPACE error must not depend on the grain, got \
                 {floor:?}"
            );
            let stepped: Vec<f64> = positive
                .iter()
                .map(|&(_, g)| {
                    let (d, c) = ortho_family_frequencies(n, 0.5, &modes, g);
                    abs_cents(&d, &c).into_iter().fold(0.0, f64::max)
                })
                .collect();
            let (iso, spruce, wild) = (stepped[0], stepped[1], stepped[2]);
            assert!(
                wild > iso && iso > spruce,
                "N={n}, {m_max}x{m_max}: the stepped error should order by modal stiffness, got \
                 {stepped:?}"
            );
            let lo = stepped.iter().cloned().fold(f64::MAX, f64::min);
            let hi = stepped.iter().cloned().fold(f64::MIN, f64::max);
            constants.push((hi - lo) / lo / (m_max as f64 / n as f64).powi(2));
        }
    }
    let hi = constants.iter().cloned().fold(f64::MIN, f64::max);
    let lo = constants.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        hi / lo < 1.05,
        "the grain's share of a block's stepped error should collapse onto (m_max/N)^2; got a \
         {:.3}x spread over {constants:?}",
        hi / lo
    );
}

#[test]
fn the_corner_rule_breaks_at_exactly_minus_one_over_m_max_squared() {
    // Push `cross` negative and an off-diagonal mode overtakes the corner. The threshold is not
    // "negative": it is `-1/m_max²`, and it tightens as the block grows. At `cross = -0.9` reading a
    // block through its corner understates its error by a fifth or more (measured 1.20x to 1.28x).
    for m_max in [2, 3, 4, 6, 8] {
        let threshold = -1.0 / (m_max * m_max) as f64;
        let modes = mode_block(m_max).unwrap();
        let worst_by_weight = |cross: f64| {
            let g = Grain {
                x: 1.0,
                cross,
                y: 1.0,
            };
            let w: Vec<f64> = modes.iter().map(|&(m, n)| ortho_weight(m, n, g)).collect();
            modes[argmax_first(&w)]
        };
        assert_eq!(
            worst_by_weight(threshold * 0.99),
            (m_max, m_max),
            "just inside {threshold:.6} the diagonal corner must still be the worst mode"
        );
        assert_ne!(
            worst_by_weight(threshold * 1.01),
            (m_max, m_max),
            "just past {threshold:.6} an off-diagonal mode must overtake it"
        );

        // ...and the proxy agrees with the measured error where the two are not degenerate:
        // `(m, n)` and `(n, m)` are twins at `g_x == g_y`, so compare the index SETS.
        let (d, c) = ortho_family_frequencies(256, 0.5, &modes, GRAIN_NEAR_GUARD);
        let err = abs_cents(&d, &c);
        let measured = modes[argmax_first(&err)];
        let predicted = worst_by_weight(-0.9);
        let as_set = |(a, b): (i64, i64)| (a.min(b), a.max(b));
        assert_eq!(
            as_set(measured),
            as_set(predicted),
            "{m_max}x{m_max}: the measured worst mode {measured:?} and the weight's {predicted:?} \
             differ"
        );
        let corner = err[modes.iter().position(|&mn| mn == (m_max, m_max)).unwrap()];
        let top = err.iter().cloned().fold(0.0, f64::max);
        assert!(
            top > 1.15 * corner,
            "{m_max}x{m_max}: reading this block through its corner should understate its error, \
             got {top:.4} against {corner:.4} cents"
        );
    }
}

// -- the membrane's block ----------------------------------------------------------------------------
//
// The block weight is the same function for both models, and a monotone square root cannot reorder
// a block, so IN SPACE the plate's corner rule transfers unchanged. What breaks it is the term the
// implicit plate does not have: the explicit scheme's sharp time error lets each mode cancel at its
// own Courant number, the diagonal's being the CFL ceiling itself. Above `λ = 1/√(M² + 1)` — beneath
// the ceiling for every block — the worst mode is the AXIAL corner. Inverted, not weakened.

#[test]
fn the_membranes_space_floor_is_the_strings_and_not_the_plates_half() {
    // The plate and the membrane share one spatial operator and differ only in the power its
    // eigenvalue carries into the frequency, so on the diagonal the membrane's droop is exactly
    // `sinc(u)` — the string's — where the plate's is `sinc(u)²`.
    let n = 128;
    let modes = mode_family("diagonal", 40).unwrap();
    let h = L_DEFAULT / n as f64;
    let eig = modal::rectangular_discrete_eigenvalues(h, n, n, &modes);
    let mut dev: f64 = 0.0;
    let mut cents_gap: f64 = 0.0;
    for (&(m, _), e) in modes.iter().zip(&eig) {
        let cont = 2.0 * (m as f64 * PI / L_DEFAULT).powi(2);
        let u = m as f64 * PI / (2.0 * n as f64);
        dev = dev.max(((e / cont).sqrt() - u.sin() / u).abs());
        let plate = 1200.0 * (e / cont).log2();
        let membrane = 600.0 * (e / cont).log2();
        cents_gap = cents_gap.max((plate - 2.0 * membrane).abs());
    }
    assert!(
        dev < 1e-14,
        "the membrane's diagonal space droop is not the string's sinc(u): {dev:e}"
    );
    assert!(cents_gap < 1e-12, "{cents_gap:e}");
}

#[test]
fn the_space_only_corner_rule_transfers_from_the_plate_unchanged() {
    // `block_weight` is one function and both models read it; the square root sits outside, on the
    // whole ratio. So with no timestep in the comparison the membrane's heaviest mode is the
    // plate's: the DIAGONAL corner, not merely some corner. Integers.
    for m_max in [2, 3, 8, 24, 64] {
        let modes = mode_block(m_max).unwrap();
        let weights: Vec<f64> = modes
            .iter()
            .map(|&(m, n)| block_weight(m, n).unwrap())
            .collect();
        assert_eq!(
            modes[argmax_first(&weights)],
            (m_max, m_max),
            "the {m_max}x{m_max} block's heaviest mode should be its diagonal corner"
        );
    }
}

/// `|measured cancellation Courant number - closed form|` on four grids, by `brentq` on the signed
/// pitch error at SciPy's default tolerances.
fn cancellation_residuals(mode: (i64, i64)) -> Vec<f64> {
    let predicted = cancellation_courant(mode.0, mode.1).unwrap();
    let (lo, hi) = (0.7 * predicted, (1.3 * predicted).min(0.9999));
    [64, 128, 256, 512]
        .iter()
        .map(|&n| {
            let root = brentq(
                |lam| membrane_cents(n, lam, &[mode])[0],
                lo,
                hi,
                SCIPY_XTOL,
                SCIPY_RTOL,
                SCIPY_MAXITER,
            )
            .unwrap();
            (root - predicted).abs()
        })
        .collect()
}

#[test]
fn the_diagonal_cancellation_number_is_an_identity_at_every_grid() {
    // On the diagonal `λ√S` at the ceiling is `sin(u)` and the scheme's own `arcsin` undoes it: an
    // identity for every N, not a limit. So the residual does not fall with the grid — it is already
    // at the root finder's floor — and asserting a rate here would be asserting `brentq`. Measured
    // worst 4.3e-8 ((1,1) at N = 512).
    for mode in [(1, 1), (2, 2), (5, 5), (17, 17)] {
        let residuals = cancellation_residuals(mode);
        assert!(
            residuals.iter().all(|&r| r < 1e-6),
            "{mode:?}: the diagonal should cancel at the ceiling on every grid, got {residuals:?}"
        );
    }
}

#[test]
fn every_off_diagonal_mode_has_its_own_cancellation_courant_number() {
    // The signed pitch error crosses zero at `√(m⁴ + n⁴)/(m² + n²)` — leading order in `1/N²` off the
    // diagonal, so the crossing APPROACHES the closed form. An 8x grid step shrinks a `1/N²`
    // residual by 64x (measured 62.1 to 64.3); a factor of 40 is the loose reading of that.
    for mode in [(2, 1), (3, 1), (8, 1), (3, 2), (5, 3), (7, 4)] {
        let r = cancellation_residuals(mode);
        assert!(
            r[0] < 3e-3,
            "{mode:?}: even the coarsest grid should be close, got {r:?}"
        );
        assert!(
            r[3] < 3e-5,
            "{mode:?}: the finest grid should be closer, got {r:?}"
        );
        assert!(
            r[0] > 40.0 * r[3],
            "{mode:?}: a 1/N^2 residual should shrink ~64x over an 8x grid step, got {r:?}"
        );
    }
}

#[test]
fn the_2d_cfl_ceiling_is_attained_by_the_diagonal_and_by_nothing_else() {
    // `cancellation_courant²` is `t² + (1-t)²` with `t = m²/ρ²`, minimised at `t = 1/2`: the ceiling
    // IS the smallest cancellation number the spectrum has, attained by the diagonal and by nothing
    // else. Sharper than `the_ceiling_is_the_minimum_of_this_function_over_the_whole_spectrum` above,
    // which says nothing about WHO attains it. The nearest off-diagonal mode sits 1.0e-4 above.
    let ceiling = mem_ceiling();
    let mut worst = (f64::INFINITY, (0, 0));
    for m in 1..=60 {
        for n in 1..=60 {
            let lam_c = cancellation_courant(m, n).unwrap();
            if (lam_c, (m, n)) < worst {
                worst = (lam_c, (m, n));
            }
            assert!(
                lam_c >= ceiling - 1e-15,
                "({m},{n}) cancels below the CFL bound: {lam_c}"
            );
            assert_eq!(
                (lam_c - ceiling).abs() < 1e-15,
                m == n,
                "({m},{n}) attains the ceiling without being diagonal, or fails to while being it"
            );
        }
    }
    assert!(
        (worst.0 - ceiling).abs() < 1e-15,
        "the minimum should be the ceiling, got {worst:?}"
    );
}

#[test]
fn no_mode_is_ever_sharp_on_a_stable_membrane_and_the_measurement_agrees() {
    // Every mode's cancellation number is at or above the ceiling and a stable run is at or below
    // it, so every mode is flat or exactly in tune — never sharp. The arithmetic is the assertion,
    // here WITHOUT a tolerance as the Python wrote it; the measurement is corroboration (the largest
    // signed error at the ceiling, measured 4.4e-10 cents at N = 256).
    let ceiling = mem_ceiling();
    for m in 1..=40 {
        for n in 1..=40 {
            assert!(ceiling <= cancellation_courant(m, n).unwrap(), "({m},{n})");
        }
    }
    for n_grid in [64, 128, 256] {
        let top = membrane_cents(n_grid, ceiling, &mode_block(16).unwrap())
            .into_iter()
            .fold(f64::MIN, f64::max);
        assert!(
            top < 1e-8,
            "N={n_grid}: a mode came out sharp at the ceiling, max = {top:+.3e} cents"
        );
    }
}

#[test]
fn a_membrane_blocks_worst_mode_is_always_a_corner() {
    // Subtracting the sharp time term `λ²ρ²` from the weight leaves an error still maximised on the
    // block's boundary. The two-corner algebra never looks at `(M, n)` for `1 < n < M`, where the
    // time term could in principle move the interior minimum far enough to win. It does not, for
    // any block or stable Courant number. Integers again, at the leading order.
    for m_max in [2, 5, 13, 40, 120] {
        let modes = mode_block(m_max).unwrap();
        let corners = [(m_max, m_max), (m_max, 1), (1, m_max)];
        for lam in linspace(0.01, mem_ceiling(), 48) {
            let err: Vec<f64> = modes
                .iter()
                .map(|&(m, n)| {
                    (block_weight(m, n).unwrap() - lam * lam * (m * m + n * n) as f64).abs()
                })
                .collect();
            let worst = modes[argmax_first(&err)];
            assert!(
                corners.contains(&worst),
                "the {m_max}x{m_max} block's worst mode at lambda={lam:.4} is {worst:?}, which is \
                 not a corner — the block reading has no licence at all if this fails"
            );
        }
    }
}

#[test]
fn the_corner_claim_also_holds_when_the_model_is_asked_rather_than_the_expansion() {
    // The measured floor under the bar above, which asks the leading order only. The sweep runs
    // ACROSS the flip, so it sees the diagonal corner win as well as the axial one.
    for m_max in MEM_BLOCKS {
        let modes = mode_block(m_max).unwrap();
        let corners = [(m_max, m_max), (m_max, 1), (1, m_max)];
        for n in [64, 256] {
            for lam in linspace(0.05, mem_ceiling(), 24) {
                let err: Vec<f64> = membrane_cents(n, lam, &modes)
                    .into_iter()
                    .map(f64::abs)
                    .collect();
                let worst = modes[argmax_first(&err)];
                assert!(
                    corners.contains(&worst),
                    "N={n}, lambda={lam:.4}, M={m_max}: the model's worst mode is {worst:?}, which \
                     the expansion says cannot happen"
                );
            }
        }
    }
}

/// `|error(M, 1)| - |error(M, M)|` on a membrane: positive where the axial corner is worse.
fn corner_gap(n: i64, m_max: i64, lam: f64) -> f64 {
    let e = membrane_cents(n, lam, &[(m_max, 1), (m_max, m_max)]);
    e[0].abs() - e[1].abs()
}

fn corner_flip(n: i64, m_max: i64) -> f64 {
    let predicted = 1.0 / ((m_max * m_max) as f64 + 1.0).sqrt();
    brentq(
        |lam| corner_gap(n, m_max, lam),
        0.7 * predicted,
        (1.3 * predicted).min(mem_ceiling() - 1e-9),
        SCIPY_XTOL,
        SCIPY_RTOL,
        SCIPY_MAXITER,
    )
    .unwrap()
}

#[test]
fn the_membrane_blocks_worst_corner_flips_at_one_over_root_m_squared_plus_one() {
    // Setting the two corners' errors equal collapses to `(M² - 1)(λ² - 1/(M² + 1)) = 0`: below
    // `λ = 1/√(M² + 1)` the diagonal corner is worst (the plate's answer), above it the axial one.
    // Leading order in `1/N²`; measured residual at N = 512 runs 2.8e-6 (M = 2) to 3.0e-5 (M = 12).
    for m_max in MEM_BLOCKS {
        let predicted = 1.0 / ((m_max * m_max) as f64 + 1.0).sqrt();
        let crossing = corner_flip(512, m_max);
        assert!(
            (crossing - predicted).abs() < 2e-4,
            "M={m_max}: the corner flip should sit at 1/sqrt(M^2+1) = {predicted:.6}, measured \
             {crossing:.6}"
        );
        assert!(
            corner_gap(512, m_max, 0.9 * predicted) < 0.0,
            "M={m_max}: below the flip the DIAGONAL corner is worst"
        );
        assert!(
            corner_gap(512, m_max, 1.1 * predicted) > 0.0,
            "M={m_max}: above the flip the AXIAL corner is worst"
        );
    }
}

#[test]
fn the_corner_flip_converges_to_the_closed_form_like_one_over_n_squared() {
    // A different kind of claim from the bar above: the rate. A 16x grid step should shrink the
    // residual by about 256 (measured 247, each doubling 3.8x to 4.0x); asserted loosely, because the
    // finest grid's residual nears `brentq`'s own tolerance.
    let predicted = 1.0 / 65.0_f64.sqrt();
    let residuals: Vec<f64> = MEM_GRIDS
        .iter()
        .map(|&n| (corner_flip(n, 8) - predicted).abs())
        .collect();
    assert!(
        residuals.windows(2).all(|w| w[1] < w[0]),
        "the residual did not fall monotonically with the grid: {residuals:?}"
    );
    assert!(
        residuals[0] > 50.0 * residuals[4],
        "a 16x grid step should shrink a 1/N^2 residual by ~256x, got {residuals:?}"
    );
}

#[test]
fn the_flip_is_below_the_ceiling_for_every_block_so_the_worst_corner_is_axial() {
    // `1/√(M² + 1) < 1/√2` for every `M >= 2`, and it falls as the block grows, so the window in
    // which a membrane's block behaves like a plate's never contains a Courant number anyone would
    // choose. Thinnest lead measured: 0.3% (M = 2).
    for m_max in MEM_BLOCKS {
        assert!(1.0 / ((m_max * m_max) as f64 + 1.0).sqrt() < mem_ceiling());
        let modes = mode_block(m_max).unwrap();
        let axial = [(m_max, 1), (1, m_max)];
        for lam in mem_lams() {
            for n in [64, 256, 512] {
                let err: Vec<f64> = membrane_cents(n, lam, &modes)
                    .into_iter()
                    .map(f64::abs)
                    .collect();
                let worst = modes[argmax_first(&err)];
                assert!(
                    axial.contains(&worst),
                    "N={n}, lambda={lam:.4}, M={m_max}: the worst mode is {worst:?}, not an axial \
                     corner — a caller reading the plate's rule here would name the wrong mode"
                );
            }
        }
    }
}

#[test]
fn the_two_axial_corners_are_exactly_degenerate() {
    // So the claim is "an axial corner", never a particular one: `mode_block`'s order makes
    // `argmax` return `(1, M)` and never `(M, 1)` — a tie-break, not a result. On a square the two
    // are the same mode to the bit, because the eigenvalue's two `sin²` terms are added in the
    // opposite order and floating-point addition commutes.
    for m_max in [2, 5, 9, 12] {
        for n in [64, 256] {
            for lam in mem_lams() {
                let e = membrane_cents(n, lam, &[(m_max, 1), (1, m_max)]);
                assert_eq!(
                    e[0], e[1],
                    "the axial twins differ at N={n}, M={m_max}, lambda={lam}"
                );
            }
        }
    }
}

#[test]
fn the_membranes_axial_family_is_monotone_so_a_block_horizon_means_something() {
    // Knowing the worst mode is the axial corner licenses reading a block ONLY if the axial family
    // has a leading prefix: the error must rise with `M`, and it does at every Courant number in the
    // stable range. Without this the section proves which corner is worst and still cannot quote a
    // block horizon. Smallest step measured: 0.0098 cents (at the ceiling).
    let modes = mode_family("axial", 40).unwrap();
    for lam in mem_lams() {
        let (d, c) = membrane_frequencies(256, lam, &modes);
        let (_, monotone) = pitch_horizon(&d, &c, CENTS).unwrap();
        assert!(
            monotone,
            "the axial family is not monotone at lambda={lam:.4}; a block horizon read through its \
             axial corner would be meaningless"
        );
        let err = abs_cents(&d, &c);
        assert!(
            err.windows(2).all(|w| w[1] - w[0] > 0.0),
            "and strictly so at lambda={lam:.4}, not merely within the flag's tolerance"
        );
    }
}
