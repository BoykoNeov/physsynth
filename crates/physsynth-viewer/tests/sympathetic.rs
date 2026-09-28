//! Sympathetic strings through the payload builder — `test_web_backend.py`'s sympathetic section.
//!
//! A closed, undriven, linear-leapfrog system: conservation is automatic and passes even a flipped
//! coupling sign, so it is table stakes. What these pin is what energy cannot see — the bridge held
//! EXACTLY still by the antisymmetric mode, the frequency-selective transfer, and the piano
//! unison's two-stage decay.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::sympathetic::{
    SYMP_N_MAX, SYMP_SIGMA_BODY_MAX, SYMP_WEINREICH_DETUNE_MAX, SYMP_WORK_MAX,
};
use physsynth_viewer::AUDIO_FS;
use serde_json::{json, Value};

fn normal(overrides: Value) -> Value {
    let mut p = json!({
        "model": "sympathetic", "domain": "normal", "N": 60, "lambda": 0.9,
        "T": 200.0, "rho": 0.005, "L": 1.0, "K": 8000.0,
        "pluck_position": 0.3, "pickup_position": 0.1, "audio_duration": 0.4,
    });
    common::merge(&mut p, overrides);
    p
}

fn transfer(overrides: Value) -> Value {
    let mut p = normal(json!({"domain": "transfer", "K": 1500.0, "audio_duration": 1.5}));
    common::merge(&mut p, overrides);
    p
}

fn wein(overrides: Value) -> Value {
    let mut p = normal(json!({
        "domain": "weinreich", "K": 6000.0, "sigma_body": 20.0, "detune": 0.0,
        "audio_duration": 1.5,
    }));
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

// -- the normal-mode oracle

#[test]
fn antisymmetric_mode_keeps_the_bridge_bit_exact_still() {
    let sp = ok(&normal(json!({})))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "sympathetic");
    assert_eq!(sp["regime"], "normal");
    assert_eq!(f(&sp["anti_max"]), 0.0, "not small — exactly zero");
    assert_eq!(sp["anti_exact_zero"], true);
    assert_eq!(f(&sp["body_frac_anti"]), 0.0);
}

#[test]
fn symmetric_mode_is_the_contrast_that_makes_the_zero_mean_something() {
    let sp = ok(&normal(json!({})))["meta"]["spectrum"].clone();
    assert!(f(&sp["sym_max"]) > 1e6 * f(&sp["anti_max"]).max(1e-30));
    assert!(f(&sp["sym_max"]) > 0.0);
    assert!(f(&sp["body_frac_sym"]) > 0.05);
}

#[test]
fn detune_is_ignored_in_the_normal_regime() {
    let sp = ok(&normal(json!({"detune": 5.0})))["meta"]["spectrum"].clone();
    assert_eq!(f(&sp["anti_max"]), 0.0);
    assert_eq!(sp["anti_exact_zero"], true);
}

#[test]
fn normal_conserves_through_the_wrapper_ordinary_drift() {
    let e = ok(&normal(json!({})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], true);
    assert_ne!(e["kind"], "balance");
    assert!(e.get("convergence").is_none());
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

#[test]
fn frames_decode_to_two_mirror_strings_with_clamped_nuts() {
    let d = ok(&normal(json!({})));
    let fr = &d["frames"];
    assert_eq!(fr["dims"], 1);
    assert_eq!(fr["fields"], json!(["string A", "string B"]));
    assert_eq!(fr["field_labels"].as_array().unwrap().len(), 2);
    let (n, w) = (
        fr["n_frames"].as_u64().unwrap() as usize,
        fr["width"].as_u64().unwrap() as usize,
    );
    let buf = decode_f32(fr["b64"].as_str().unwrap());
    for k in 0..n {
        let a = &buf[(2 * k) * w..(2 * k + 1) * w];
        let b = &buf[(2 * k + 1) * w..(2 * k + 2) * w];
        assert!(
            a.iter().zip(b).all(|(x, y)| *y == -*x),
            "B is not A's antiphase"
        );
        assert_eq!(a[0], 0.0);
        assert_eq!(b[0], 0.0);
    }
}

#[test]
fn audio_is_the_plucked_string_pickup_not_silence() {
    let a = ok(&normal(json!({})))["audio"].clone();
    assert!(a["n"].as_u64().unwrap() > 0);
    assert_eq!(f(&a["fs"]), AUDIO_FS);
    let x = decode_f32(a["b64"].as_str().unwrap());
    assert!(x.iter().all(|v| v.is_finite()) && x.iter().any(|&v| v != 0.0));
}

// -- transfer

#[test]
fn transfer_tuned_unison_drains_most_of_the_energy() {
    let sp = ok(&transfer(json!({"detune": 0.0})))["meta"]["spectrum"].clone();
    assert_eq!(sp["regime"], "transfer");
    assert_eq!(sp["tuned"], true);
    assert!(f(&sp["peak_neighbour"]) > 0.5);
    assert!(f(&sp["frac1"][0]) < 0.05);
    assert!(f(&sp["frac0"][0]) > 0.9);
}

#[test]
fn transfer_detuned_neighbour_stays_quiet() {
    let tuned = ok(&transfer(json!({"detune": 0.0})))["meta"]["spectrum"].clone();
    let detuned = ok(&transfer(json!({"detune": 4.0})))["meta"]["spectrum"].clone();
    assert!(f(&detuned["peak_neighbour"]) < 0.25);
    assert!(f(&tuned["peak_neighbour"]) > 3.0 * f(&detuned["peak_neighbour"]));
}

#[test]
fn transfer_conserves_and_the_fractions_stay_physical() {
    let d = ok(&transfer(json!({"detune": 0.0})));
    assert!(f(&d["energy"]["lossless"]["drift"]) < LOSSLESS_TOL);
    for key in ["frac0", "frac1"] {
        assert!(nums(&d["meta"]["spectrum"][key])
            .iter()
            .all(|&v| (-1e-9..=1.0 + 1e-9).contains(&v)));
    }
}

// -- refusals

#[test]
fn lambda_must_be_below_one() {
    let r = sim(&normal(json!({"lambda": 1.0})));
    assert!(r["error"]["message"]
        .as_str()
        .unwrap()
        .contains("lambda must be in (0, 1)"));
}

#[test]
fn over_stiff_bridge_is_rejected_by_the_core_guard() {
    let r = sim(&normal(json!({"K": 1.0e6})));
    assert!(r["error"]["message"].as_str().unwrap().contains("unstable"));
}

#[test]
fn bad_params_give_error_payload() {
    for bad in [
        json!({"N": SYMP_N_MAX + 1}),
        json!({"domain": "nonesuch"}),
        json!({"detune": -1.0, "domain": "transfer"}),
        json!({"detune": 99.0, "domain": "transfer"}),
        json!({"K": -5.0}),
        json!({"domain": "weinreich", "sigma_body": 999.0}),
        json!({"domain": "weinreich", "detune": 5.0}),
    ] {
        assert!(sim(&normal(bad.clone())).get("error").is_some(), "{bad}");
    }
}

#[test]
fn normal_work_budget_counts_both_runs() {
    let r = sim(&normal(
        json!({"audio_duration": 3.0, "T": 800.0, "N": 160}),
    ));
    assert!(r["error"]["message"]
        .as_str()
        .unwrap()
        .contains("work budget"));
    let default_steps = (0.4 * (200.0f64 / 0.005).sqrt() * 60.0 / 0.9).round_ties_even() as i64;
    assert!(2 * default_steps < SYMP_WORK_MAX);
}

// -- weinreich

#[test]
fn weinreich_two_stage_knee_prompt_faster_than_aftersound() {
    let sp = ok(&wein(json!({})))["meta"]["spectrum"].clone();
    assert_eq!(sp["regime"], "weinreich");
    assert!(f(&sp["prompt_rate"]) > 0.5);
    assert!(f(&sp["prompt_rate"]) > 5.0 * f(&sp["aftersound_rate"]).max(1e-6));
}

#[test]
fn weinreich_unison_aftersound_is_lossless_rising_with_detune() {
    let unison = ok(&wein(json!({"detune": 0.0})))["meta"]["spectrum"].clone();
    let mistuned = ok(&wein(json!({"detune": 0.3})))["meta"]["spectrum"].clone();
    assert!(f(&unison["aftersound_rate"]) < 0.1);
    assert!(f(&mistuned["aftersound_rate"]) > 2.5 * f(&unison["aftersound_rate"]).max(1e-3));
}

#[test]
fn weinreich_strike_both_decays_away_no_aftersound() {
    let sp = ok(&wein(json!({})))["meta"]["spectrum"].clone();
    assert!(f(&sp["floor_one"]) > 0.3);
    assert!(f(&sp["both_final"]) < 0.15);
    assert!(f(&sp["floor_one"]) > 2.0 * f(&sp["both_final"]));
}

#[test]
fn weinreich_lossy_body_reports_passivity_without_a_decay_oracle() {
    let e = ok(&wein(json!({})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(e["lossy"].get("measured_2sigma").is_none());
    assert_ne!(e["kind"], "balance");
    assert!(e.get("convergence").is_none());
}

#[test]
fn weinreich_zero_body_loss_flips_to_the_drift_check() {
    let d = ok(&wein(json!({"sigma_body": 0.0})));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let sp = &d["meta"]["spectrum"];
    assert_eq!(sp["sigma_zero"], true);
    assert!(f(&sp["aftersound_rate"]).abs() < 0.05 && f(&sp["prompt_rate"]) < 0.5);
}

#[test]
fn weinreich_envelopes_are_finite_and_normalized() {
    let sp = ok(&wein(json!({})))["meta"]["spectrum"].clone();
    for key in ["env_one", "env_both"] {
        let arr = &sp[key];
        assert!((0.85..=1.05).contains(&f(&arr[0])), "{key}");
        assert!(arr
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.as_f64().is_some_and(|x| (-1e-9..=1.05).contains(&x))));
    }
}

#[test]
fn weinreich_audio_is_the_struck_string_pickup() {
    let d = ok(&wein(json!({})));
    assert_eq!(f(&d["audio"]["fs"]), AUDIO_FS);
    assert!(decode_f32(d["audio"]["b64"].as_str().unwrap())
        .iter()
        .any(|&v| v != 0.0));
    assert_eq!(d["frames"]["dims"], 1);
    assert_eq!(d["frames"]["fields"].as_array().unwrap().len(), 2);
}

#[test]
fn weinreich_work_budget_counts_both_runs() {
    let r = sim(&wein(json!({"audio_duration": 3.0, "T": 800.0, "N": 160})));
    assert!(r["error"]["message"]
        .as_str()
        .unwrap()
        .contains("work budget"));
}

#[test]
fn weinreich_detune_range_is_fine_not_semitones() {
    const { assert!(SYMP_WEINREICH_DETUNE_MAX < 1.0) };
    assert!(
        sim(&wein(json!({"detune": SYMP_WEINREICH_DETUNE_MAX + 0.5})))
            .get("error")
            .is_some()
    );
    assert!(sim(&wein(json!({"sigma_body": SYMP_SIGMA_BODY_MAX + 1.0})))
        .get("error")
        .is_some());
}
