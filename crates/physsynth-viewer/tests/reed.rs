//! The reed through the payload builder — `test_web_backend.py`'s reed section.
//!
//! The reed is the acoustic dual of the bow, but its balance is a STRONGER claim: its loss channels
//! are measured independently, so the residual can genuinely fail. These are written to catch a
//! broken wiring: a residual on a sum cannot see a dead summand, and a balance passes on silence,
//! so the channels and the speaking are asserted separately from the residual.

mod common;

use common::{f, ok, sim};
use physsynth_viewer::bow::BOW_BALANCE_TOL;
use physsynth_viewer::reed::{
    REED_ANIM_MAX, REED_AUDIO_MAX, REED_N_MAX, REED_N_MIN, REED_SPEAK_GATE, REED_SWEEP_N,
    REED_WORK_MAX,
};
use physsynth_viewer::string::ANIM_WIN_MAX;
use serde_json::{json, Value};

fn reed(overrides: Value) -> Value {
    let mut p = json!({"model": "reed", "audio_duration": 0.3, "N": 96});
    common::merge(&mut p, overrides);
    p
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect()
}

#[test]
fn balance_is_a_measured_residual_and_the_channels_are_load_bearing() {
    let e = ok(&reed(json!({})))["energy"].clone();
    assert_eq!(e["kind"], "balance");
    let b = &e["balance"]["measured"];
    assert!(f(&b["residual"]) < BOW_BALANCE_TOL, "{b}");
    assert_eq!(b["pass"], true);
    // measured loss dropped => the jet + damping fraction reappears
    assert!(f(&b["naive_residual"]) > 0.5, "{b}");
}

#[test]
fn every_balance_channel_is_non_trivially_populated() {
    let bud = ok(&reed(json!({})))["meta"]["budget"].clone();
    assert!(f(&bud["mouth_work"]) > 0.0);
    assert!(f(&bud["jet_frac"]) > 0.1);
    assert!(f(&bud["damping_frac"]) > 0.0);
    assert!(f(&bud["stored_frac"]) > 0.0);
    let tot = f(&bud["jet_frac"])
        + f(&bud["damping_frac"])
        + f(&bud["radiated_frac"])
        + f(&bud["stored_frac"]);
    assert!((tot - 1.0).abs() <= 2e-3, "{bud}");
}

#[test]
fn balance_has_no_sigma_gate_and_that_is_not_an_oversight() {
    for domain in ["radiating", "open"] {
        let b = ok(&reed(json!({"domain": domain})))["energy"]["balance"].clone();
        assert!(b.get("measured").is_some());
        assert!(b.get("lossless").is_none() && b.get("lossy").is_none());
        assert!(f(&b["measured"]["residual"]) < BOW_BALANCE_TOL);
    }
}

#[test]
fn closes_the_balance_with_the_bell_radiating() {
    let rad = ok(&reed(json!({"domain": "radiating"})))["meta"].clone();
    let op = ok(&reed(json!({"domain": "open"})))["meta"].clone();
    assert!(f(&rad["budget"]["radiated_frac"]) > 0.02);
    assert_eq!(f(&op["budget"]["radiated_frac"]), 0.0);
    assert!(!rad["r_ratio"].is_null() && op["r_ratio"].is_null());
}

#[test]
fn actually_speaks_above_threshold_and_is_silent_below() {
    let loud = ok(&reed(json!({"gamma": 0.51})))["meta"].clone();
    let quiet = ok(&reed(json!({"gamma": 0.20})))["meta"].clone();
    assert_eq!(loud["speaks"], true);
    assert!(f(&loud["ac_level"]) > 10.0 * REED_SPEAK_GATE);
    assert_eq!(quiet["speaks"], false);
    assert!(f(&quiet["ac_level"]) < REED_SPEAK_GATE);
    assert_eq!(loud["beating"]["beats"], true);
    assert_eq!(quiet["beating"]["beats"], false);
}

#[test]
fn blowing_threshold_brackets_one_third() {
    let sw = ok(&reed(json!({})))["meta"]["sweep"].clone();
    let (lo, hi) = (f(&sw["bracket"][0]), f(&sw["bracket"][1]));
    assert!(lo <= 0.36 && hi >= 0.30, "{}", sw["bracket"]);
    let level = nums(&sw["level"]);
    let (mn, mx) = (
        level.iter().copied().fold(f64::INFINITY, f64::min),
        level.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(mn < 0.01 && 0.01 < mx);
    assert!(mx / mn.max(1e-12) > 100.0);
    assert!(
        level.windows(2).all(|w| w[1] >= w[0] - 1e-9),
        "monotone in gamma"
    );
    assert!(
        nums(&sw["gamma"]).contains(&(1.0 / 3.0)),
        "1/3 on the grid EXACTLY"
    );
}

#[test]
fn sweep_is_pinned_off_the_render_grid() {
    let a = ok(&reed(json!({"N": 64})))["meta"]["sweep"].clone();
    let b = ok(&reed(json!({"N": 200, "audio_duration": 0.2})))["meta"]["sweep"].clone();
    assert_eq!(a["sweep_N"], REED_SWEEP_N);
    assert_eq!(a["bracket"], b["bracket"]);
    assert_eq!(a["level"], b["level"]);
}

#[test]
fn the_sweep_moves_with_everything_that_moves_the_reed_or_the_bell() {
    // The reference guarded a memo key with this; there is no memo here (the sweep is cheap), and
    // the claim underneath it stands on its own: f_reed, q_reed, the bell and L all move the curve.
    let base = ok(&reed(json!({})))["meta"]["sweep"]["level"].clone();
    for over in [
        json!({"f_reed": 1500.0}),
        json!({"q_reed": 8.0}),
        json!({"bell_ratio_exp": -1.2}),
        json!({"L": 0.7}),
    ] {
        assert_ne!(
            ok(&reed(over.clone()))["meta"]["sweep"]["level"],
            base,
            "{over}"
        );
    }
}

#[test]
fn pitch_is_set_by_the_air_column_not_the_reed() {
    let pit = ok(&reed(json!({})))["meta"]["sweep"]["pitch"].clone();
    assert!((f(&pit["reed_change_pct"]) - 50.0).abs() < 1e-9);
    let pc = f(&pit["pitch_change_pct"]);
    assert!(pc.abs() < 6.0 && pc > 0.0);
    let cents = nums(&pit["cents"]);
    assert!(cents[1] > cents[0], "a stiffer reed lands closer to c/4L");
    assert!(
        cents.iter().all(|&c| c < 0.0),
        "reed compliance always flattens"
    );
}

#[test]
fn is_a_clarinet_odd_harmonics_dominate() {
    let sp = ok(&reed(json!({})))["meta"]["spectrum"].clone();
    assert_eq!(sp["applies"], true);
    assert!(f(&sp["odd_even"]) > 100.0);
    assert!(f(&sp["third_second"]) > 1.0);
    assert!(f(&sp["crest"]) < 1.5);
}

#[test]
fn beating_is_debounced_to_one_slam_per_period() {
    let bt = ok(&reed(json!({"gamma": 0.51})))["meta"]["beating"].clone();
    assert_eq!(bt["beats"], true);
    assert!((f(&bt["per_period"]) - 1.0).abs() <= 0.15, "{bt}");
    assert!(0.2 < f(&bt["duty"]) && f(&bt["duty"]) < 0.6);
    assert_eq!(f(&bt["min_opening"]), 0.0);
}

#[test]
fn ships_the_far_field_caveat_beside_the_mouthpiece_audio() {
    let sp = ok(&reed(json!({})))["meta"]["spectrum"].clone();
    let ff = &sp["far_field"];
    assert!(f(&ff["quieter_by"]) > 20.0);
    assert!(f(&ff["crest"]) > f(&sp["crest"]));
    let op = ok(&reed(json!({"domain": "open"})))["meta"]["spectrum"].clone();
    assert_eq!(op["applies"], true);
    assert!(op["far_field"].is_null());
    assert!(op.get("far_note").is_some());
}

#[test]
fn below_threshold_withdraws_the_spectrum_claims() {
    let sp = ok(&reed(json!({"gamma": 0.20})))["meta"]["spectrum"].clone();
    assert_eq!(sp["applies"], false);
    assert!(sp.get("odd_even").is_none());
    assert!(sp.get("note").is_some());
}

#[test]
fn announces_its_mouth_end_for_the_viz_without_touching_the_bore_boundary() {
    assert_eq!(
        ok(&reed(json!({})))["meta"]["ends"],
        json!(["reed", "radiating"])
    );
    assert_eq!(
        ok(&reed(json!({"domain": "open"})))["meta"]["ends"],
        json!(["reed", "open"])
    );
}

#[test]
fn animation_is_paced_on_the_transit_and_captured_from_the_settled_tail() {
    let d = ok(&reed(json!({})));
    let fpt = f(&d["meta"]["frames_per_transit"]);
    assert!(8.0 < fpt && fpt < 20.0);
    let t = nums(&d["frame_times"]);
    let end = f(&d["meta"]["num_steps"]) / f(&d["fs_sim"]);
    assert!((t[t.len() - 1] - end).abs() <= 1e-3 * end);
    assert!(t[0] > 0.5 * t[t.len() - 1], "a tail, not the attack");
}

#[test]
fn guards_reject_out_of_range_configurations_cleanly() {
    for bad in [
        json!({"N": REED_N_MAX, "audio_duration": REED_AUDIO_MAX + 0.5}),
        json!({"N": REED_N_MAX + 1}),
        json!({"N": REED_N_MIN - 1}),
        json!({"audio_duration": REED_AUDIO_MAX + 0.1}),
        json!({"animation_window": REED_ANIM_MAX + 0.01}),
        json!({"gamma": 0.0}),
        json!({"gamma": 2.0}),
        json!({"f_reed": 200.0}),
        json!({"domain": "bell"}),
    ] {
        assert!(sim(&reed(bad.clone())).get("error").is_some(), "{bad}");
    }
    const { assert!(REED_ANIM_MAX < ANIM_WIN_MAX) };
    const { assert!(REED_WORK_MAX > 0) };
}

#[test]
fn ignores_params_that_belong_to_other_models() {
    let base = ok(&reed(json!({})));
    let other = ok(&reed(json!({
        "sigma0": 0.9, "sigma1": 0.05, "kappa": 4.0, "depth": 0.002, "alpha": 1.5, "K": 8000,
    })));
    assert_eq!(other["fs_sim"], base["fs_sim"]);
    assert_eq!(
        other["meta"]["budget"]["jet_frac"],
        base["meta"]["budget"]["jet_frac"]
    );
    assert!(f(&other["energy"]["balance"]["measured"]["residual"]) < BOW_BALANCE_TOL);
}
