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
//! Everything here is an identity or an inequality out of the docstrings, so nothing in it is a
//! fixture. The two exceptions are the published grid fractions (5.925% and 8.378%), which are
//! *derived* claims quoted in the plan and worth pinning where a reader will find them.

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
