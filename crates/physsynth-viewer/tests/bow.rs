//! The bowed string through the payload builder — `test_web_backend.py`'s bow section.
//!
//! The first driven model: its energy panel is the BALANCE `E - E0 == work - loss`, which replaces
//! both older verdicts because for a driven model both would lie. Its second panel scores
//! `slip_fraction == beta` only inside the Helmholtz window, and labels the regime outside it.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_viewer::bow::{BOW_BALANCE_TOL, BOW_N_MAX, BOW_SLIP_MATCH_TOL};
use physsynth_viewer::energy::LOSSLESS_TOL;
use serde_json::{json, Value};

fn bow_p(overrides: Value) -> Value {
    let mut p = json!({
        "model": "bow", "N": 64, "lambda": 0.9, "kappa": 0.0, "audio_duration": 1.0,
        "sigma0": 0.5, "sigma1": 0.05, "force": 1.0, "v_bow": 0.1,
        "bow_position": 0.13, "sharpness": 60.0, "pickup_position": 0.33,
    });
    common::merge(&mut p, overrides);
    p
}

fn bow_quiet(overrides: Value) -> Value {
    let mut p = bow_p(json!({"sigma0": 0.0, "sigma1": 0.0, "audio_duration": 0.4}));
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

/// `(n_frames, width, field)` of a payload's frames.
fn field(r: &Value) -> (usize, usize, Vec<f32>) {
    let fr = &r["frames"];
    (
        fr["n_frames"].as_u64().unwrap() as usize,
        fr["width"].as_u64().unwrap() as usize,
        decode_f32(fr["b64"].as_str().unwrap()),
    )
}

#[test]
fn lossless_balance_is_the_money_number() {
    let e = ok(&bow_quiet(json!({})))["energy"].clone();
    assert_eq!(e["kind"], "balance");
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["balance"]["lossless"]["residual"]) < BOW_BALANCE_TOL);
    assert_eq!(e["balance"]["lossless"]["pass"], true);
    assert!(f(&e["balance"]["work_total"]) > 0.0, "nothing was tested");
}

#[test]
fn balance_replaces_both_older_verdicts_because_both_would_lie() {
    let quiet = ok(&bow_quiet(json!({})))["energy"].clone();
    let lossy = ok(&bow_p(json!({})))["energy"].clone();
    for e in [&quiet, &lossy] {
        assert_eq!(e["kind"], "balance");
        assert!(e.get("lossless").is_none() && e.get("lossy").is_none());
        assert!(
            e.get("convergence").is_none(),
            "exact for any Newton residual — no gate"
        );
    }
    // ... and prove they WOULD have lied. From rest E0 = 0, so the drift falls back to max|E|.
    let v = nums(&quiet["value"]);
    assert_eq!(v[0], 0.0);
    let apparent = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(apparent > 1e3 * LOSSLESS_TOL, "{apparent:e}");
    let lv = nums(&lossy["value"]);
    assert!(
        lv[lv.len() - 1] > lv[0],
        "a bowed note GAINS energy from rest"
    );
}

#[test]
fn lossy_reports_inferred_dissipation_not_a_tautological_residual() {
    let b = ok(&bow_p(json!({})))["energy"]["balance"].clone();
    assert!(b.get("lossless").is_none());
    assert!(b["lossy"].get("residual").is_none());
    assert_eq!(b["lossy"]["non_negative"], true);
    assert_eq!(b["lossy"]["monotone"], true);
    assert_eq!(b["lossy"]["pass"], true);
    assert!(f(&b["lossy"]["dissipation_total"]) > 0.0);
}

#[test]
fn balance_curves_share_one_decimation() {
    let e = ok(&bow_p(json!({})))["energy"].clone();
    let b = &e["balance"];
    let n = e["time"].as_array().unwrap().len();
    let (w, d, x) = (
        nums(&b["work"]),
        nums(&b["delta_energy"]),
        nums(&b["dissipation"]),
    );
    assert!(w.len() == n && d.len() == n && x.len() == n);
    for i in 0..n {
        assert!((x[i] - (w[i] - d[i])).abs() <= 1e-12);
    }
}

#[test]
fn helmholtz_slip_fraction_matches_beta() {
    let sp = ok(&bow_p(json!({})))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "bow");
    assert_eq!(sp["helmholtz"], true);
    assert!((f(&sp["slips_per_period"]) - 1.0).abs() <= 0.25);
    assert!((f(&sp["slip_fraction"]) - f(&sp["beta"])).abs() < BOW_SLIP_MATCH_TOL);
    assert_eq!(sp["slip_matches_beta"], true);
}

#[test]
fn slip_fraction_tracks_beta_as_the_bow_moves() {
    for x in [0.13, 0.2, 0.25] {
        let sp = ok(&bow_p(json!({"bow_position": x, "force": 0.4})))["meta"]["spectrum"].clone();
        assert_eq!(sp["helmholtz"], true, "beta ~ {x}");
        assert!((f(&sp["slip_fraction"]) - f(&sp["beta"])).abs() < BOW_SLIP_MATCH_TOL);
    }
}

#[test]
fn pitch_is_the_strings_not_the_bows() {
    let slow = ok(&bow_p(json!({})))["meta"]["spectrum"].clone();
    let fast = ok(&bow_p(json!({"v_bow": 0.2, "force": 0.8})))["meta"]["spectrum"].clone();
    assert!(f(&slow["pitch_cents"]).abs() < 60.0);
    if fast["helmholtz"] == true {
        assert!((f(&fast["f_detected"]) - f(&slow["f_detected"])).abs() < 0.05 * f(&slow["f1"]));
    }
}

#[test]
fn out_of_window_is_labelled_not_failed() {
    let r = ok(&bow_p(json!({"force": 0.02})));
    let sp = &r["meta"]["spectrum"];
    assert!(f(&r["meta"]["helmholtz_number"]) < 1.0);
    assert_eq!(sp["helmholtz"], false);
    assert!(sp["slip_matches_beta"].is_null());
    assert!(sp["note"].as_str().unwrap().contains("not scored"));
    assert!(sp.get("slip_error").is_none() && sp.get("pitch_cents").is_none());
    assert_eq!(sp["regime"], "never_sticks");
    assert!(f(&sp["slip_fraction"]) > 0.95);
    // energy is STRUCTURAL: off-window motion is still exactly balanced
    assert_eq!(r["energy"]["balance"]["lossy"]["pass"], true);
}

#[test]
fn zero_force_leaves_the_string_at_rest() {
    let r = ok(&bow_quiet(json!({"force": 0.0})));
    assert_eq!(f(&r["energy"]["balance"]["work_total"]), 0.0);
    assert_eq!(f(&r["field_amp"]), 0.0);
    assert!(field(&r).2.iter().all(|&v| v == 0.0));
}

#[test]
fn frames_decode_to_a_string_with_fixed_ends() {
    let r = ok(&bow_p(json!({})));
    let (n, w, fl) = field(&r);
    assert_eq!(w, r["grid"]["x"].as_array().unwrap().len());
    assert_eq!(w, 65);
    for row in 0..n {
        assert_eq!(fl[row * w], 0.0);
        assert_eq!(fl[row * w + w - 1], 0.0);
    }
    let peak = fl.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    assert!((peak - f(&r["field_amp"])).abs() <= 1e-5 * peak);
    assert!(f(&r["field_amp"]) > 1e-6, "the settled corner, not rest");
}

#[test]
fn short_render_animates_the_attack_from_rest() {
    // With less audio than the animation window the settle window is EMPTY and capture starts at
    // step 0, so the first frame is the string at rest; the default is the other branch.
    let r = ok(&bow_p(json!({"audio_duration": 0.05})));
    let (_, w, fl) = field(&r);
    assert_eq!(f(&r["frame_times"][0]), 0.0);
    assert!(fl[..w].iter().all(|&v| v == 0.0));
    assert!(fl.iter().any(|&v| v != 0.0));
    let d = ok(&bow_p(json!({})));
    let (_, dw, dfl) = field(&d);
    assert_eq!(f(&d["frame_times"][0]), 0.0);
    assert!(
        dfl[..dw].iter().any(|&v| v != 0.0),
        "the settled window must not start from rest"
    );
}

#[test]
fn helmholtz_number_is_reported_never_asserted() {
    let r = ok(&bow_p(json!({})));
    assert!(f(&r["meta"]["helmholtz_number"]) > 1.0);
    assert_eq!(r["energy"]["balance"]["lossy"]["pass"], true);
}

#[test]
fn bad_params_give_error_payload() {
    for bad in [
        json!({"N": BOW_N_MAX + 1}),
        json!({"force": -1.0}),
        json!({"force": 99.0}),
        json!({"v_bow": 0.0}),
        json!({"sharpness": 0.0}),
        json!({"bow_position": 0.0}),
        json!({"bow_position": 2.0}),
        json!({"audio_duration": 0.0}),
        json!({"audio_duration": 99.0}),
    ] {
        let r = sim(&bow_p(bad.clone()));
        assert!(
            !r["error"]["message"].as_str().unwrap_or("").is_empty(),
            "{bad}"
        );
    }
}

#[test]
fn work_budget_is_its_own() {
    let r = sim(&bow_p(
        json!({"N": 256, "lambda": 0.5, "audio_duration": 3.0}),
    ));
    assert!(r["error"]["message"]
        .as_str()
        .unwrap()
        .contains("root-find"));
}

#[test]
fn the_horizon_is_the_strings_it_drives() {
    // An exciter inherits the dispersion of what it drives: a bowed string reads like the damped
    // string underneath it (the horizon section's inventory row, carried here with its scene).
    let bowed = ok(&json!({"model": "bow", "N": 64, "audio_duration": 0.1}))["horizon"].clone();
    let plain = ok(&json!({"model": "damped", "N": 64, "kappa": 0.0, "audio_duration": 0.3}))
        ["horizon"]
        .clone();
    assert_eq!(bowed["scheme"], "implicit theta-scheme");
    assert_eq!(bowed["scheme"], plain["scheme"]);
    assert_eq!(
        common::band(&bowed, 5.0)["modes"],
        common::band(&plain, 5.0)["modes"]
    );
    assert!(bowed["of"].as_str().unwrap().contains("string"));
}
