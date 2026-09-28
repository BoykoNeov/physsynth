//! The string on a lumped modal body through the payload builder — `test_web_backend.py`'s body
//! section, plus the `R = 0` anchor that ties the radiation-loaded body back to this one.
//!
//! The claim is the three-way energy split `E_string + E_body + E_conn`: the total conserves
//! through the coupling while `E_string` alone does not, and the body colours the radiated spectrum.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_core::engine::Resonator;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_ideal::{Boundary, IdealString, Params};
use physsynth_viewer::body::{
    BODY_AUDIO_MAX, BODY_DISTANCE_MAX, BODY_N_MAX, BODY_SIGMA_BODY_MAX, BODY_WORK_MAX,
};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::AUDIO_FS;
use serde_json::{json, Value};

fn body(overrides: Value) -> Value {
    let mut p = json!({"model": "body", "audio_duration": 0.5});
    common::merge(&mut p, overrides);
    p
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

#[test]
fn conserves_through_the_coupling_while_the_string_alone_does_not() {
    let d = ok(&body(json!({"bridge_stiffness": 8000})));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let ex = &d["meta"]["exchange"];
    assert!(f(&ex["total_drift"]) < LOSSLESS_TOL);
    assert!(f(&ex["string_frac_max"]) > 0.9);
    assert!(
        f(&ex["string_frac_min"]) < 0.6,
        "E_string alone must visibly slosh out"
    );
    assert!(
        f(&ex["body_frac_peak"]) > 0.4,
        "the body must carry a large fraction at the peak"
    );
}

/// `K = 0` severs the spring: the string field is bit-identical to a bare fixed/free string
/// plucked the same way, the body never moves and nothing radiates.
#[test]
fn k_zero_decouples_the_string_bit_for_bit() {
    let d = ok(&body(
        json!({"bridge_stiffness": 0.0, "audio_duration": 0.3}),
    ));
    let ex = &d["meta"]["exchange"];
    assert_eq!(
        f(&ex["body_frac_peak"]),
        0.0,
        "a decoupled body must never move"
    );
    assert!((f(&ex["string_frac_min"]) - 1.0).abs() < 1e-12);
    assert_eq!(f(&d["audio"]["peak"]), 0.0, "a still body radiates nothing");
    // The Python reference emitted NaN here (0/0 on a silent body) and its server would have
    // answered 500; the port reports "no reading" instead.
    assert!(d["meta"]["spectrum"]["omega2_consistency"].is_null());

    let fs = f(&d["fs_sim"]);
    let width = d["frames"]["width"].as_u64().unwrap() as usize;
    let frames = decode_f32(d["frames"]["b64"].as_str().unwrap());
    let last = &frames[frames.len() - width..];
    let times = nums(&d["frame_times"]);
    let last_step = (times[times.len() - 1] * fs).round_ties_even() as usize;

    let c = (200.0f64 / 0.005).sqrt();
    let p = Params::new(
        1.0,
        200.0,
        0.005,
        c * 100.0 / 0.9,
        100,
        0.0,
        Some((Boundary::Fixed, Boundary::Free)),
    )
    .unwrap();
    let x = p.grid();
    let mut s = IdealString::new(p);
    let u0 = triangular_pluck(&x, 1.0, 0.3, 1e-3).unwrap();
    s.set_state(&u0, &vec![0.0; u0.len()]);
    for _ in 0..last_step {
        Resonator::step(&mut s).unwrap();
    }
    let bare: Vec<f32> = Resonator::state(&s).iter().map(|&v| v as f32).collect();
    assert_eq!(last, &bare[..]);
}

#[test]
fn exchange_fractions_carry_e_conn_and_must_not_be_stacked() {
    let ex = ok(&body(json!({"bridge_stiffness": 8000})))["meta"]["exchange"].clone();
    assert_eq!(ex["kind"], "body");
    let n = ex["time"].as_array().unwrap().len();
    for key in [
        "time",
        "e_string_frac",
        "e_body_frac",
        "e_conn_frac",
        "total_frac",
    ] {
        let len = ex[key].as_array().unwrap().len();
        assert!(len == n && len > 100, "{key}");
    }
    let (es, eb) = (nums(&ex["e_string_frac"]), nums(&ex["e_body_frac"]));
    let (ec, tot) = (nums(&ex["e_conn_frac"]), nums(&ex["total_frac"]));
    let mut min_sb = f64::INFINITY;
    for i in 0..n {
        assert!((es[i] + eb[i] + ec[i] - tot[i]).abs() <= 1e-9 + 1e-5 * tot[i].abs());
        assert!((tot[i] - 1.0).abs() <= 1e-9 + 1e-5);
        min_sb = min_sb.min(es[i] + eb[i]);
    }
    assert!(
        min_sb < 0.95,
        "E_conn carries a real share; stacking es+eb hides it"
    );
}

#[test]
fn exchange_slosh_is_prompt_and_windowed() {
    let ex = ok(&body(
        json!({"bridge_stiffness": 8000, "audio_duration": 1.0}),
    ))["meta"]["exchange"]
        .clone();
    assert!((f(&ex["window"]) - 0.4).abs() <= 0.01);
    let first = f(&ex["first_peak_ms"]);
    assert!(
        0.0 < first && first < 40.0,
        "the slosh is prompt (tens of ms): {first}"
    );
}

#[test]
fn terminus_glides_from_free_toward_clamped_as_the_bridge_stiffens() {
    let lo = ok(&body(json!({"bridge_stiffness": 200})))["meta"]["spectrum"].clone();
    let hi = ok(&body(json!({"bridge_stiffness": 8000})))["meta"]["spectrum"].clone();
    assert!((f(&lo["f1_free"]) - 50.0).abs() < 1e-9);
    assert!((f(&lo["f1_clamped"]) - 100.0).abs() < 1e-9);
    assert!(f(&lo["f1_free"]) < f(&lo["terminus_f1"]));
    assert!(f(&lo["terminus_f1"]) < f(&hi["terminus_f1"]));
    assert!(f(&hi["terminus_f1"]) < f(&lo["f1_clamped"]));
    assert!(
        (f(&hi["terminus_f1"]) - 90.7).abs() <= 1.5,
        "the probe's measured K=8000 glide"
    );
}

#[test]
fn monopole_omega2_is_a_consistency_check_near_one_not_an_oracle() {
    let spx = ok(&body(json!({"bridge_stiffness": 8000})))["meta"]["spectrum"].clone();
    assert!((f(&spx["omega2_consistency"]) - 1.0).abs() <= 0.05);
}

#[test]
fn one_over_r_scales_level_and_latency_only_never_the_spectrum_shape() {
    let a = ok(&body(json!({"distance": 1.0, "audio_duration": 0.4})))["meta"]["spectrum"].clone();
    let b = ok(&body(json!({"distance": 4.0, "audio_duration": 0.4})))["meta"]["spectrum"].clone();
    let (ga, gb) = (f(&a["gain_times_r"]), f(&b["gain_times_r"]));
    assert!((ga - gb).abs() <= 1e-6 * gb.abs());
    let (la, lb) = (f(&a["latency_ms"]), f(&b["latency_ms"]));
    assert!((lb - 4.0 * la).abs() <= 0.05 * 4.0 * la);
    assert_eq!(a["f"], b["f"]);
    assert_eq!(
        a["mag"], b["mag"],
        "distance must not change the spectrum shape"
    );
}

#[test]
fn sigma_body_gates_the_verdict_and_drops_the_decay_oracle() {
    let e = ok(&body(json!({"bridge_stiffness": 8000, "sigma_body": 20.0})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(e["lossy"].get("measured_2sigma").is_none());
    assert!(e["lossy"].get("oracle_2sigma").is_none());
}

#[test]
fn guard_is_the_exact_bound_surfaced_as_a_clean_error() {
    let over = sim(&json!({"model": "body", "bridge_stiffness": 500_000}));
    assert_eq!(over["error"]["kind"], "construction");
    assert!(over["error"]["message"]
        .as_str()
        .unwrap()
        .contains("lambda_max"));
    let lam = sim(&json!({"model": "body", "lambda": 1.0}));
    assert_eq!(lam["error"]["kind"], "param");
    let neg = sim(&json!({"model": "body", "bridge_stiffness": -1.0}));
    assert_eq!(neg["error"]["kind"], "param");
}

#[test]
fn frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped() {
    let d = ok(&body(json!({"audio_duration": 0.3})));
    let width = d["frames"]["width"].as_u64().unwrap() as usize;
    assert_eq!(width, d["grid"]["x"].as_array().unwrap().len());
    let frames = decode_f32(d["frames"]["b64"].as_str().unwrap());
    let n_frames = frames.len() / width;
    assert_eq!(d["frames"]["n_frames"], n_frames);
    assert_eq!(d["meta"]["n_frames"], n_frames);
    assert!(frames.iter().all(|v| v.is_finite()));
    for row in frames.chunks_exact(width) {
        assert_eq!(row[0], 0.0, "the nut (node 0) is clamped");
    }
}

#[test]
fn audio_is_the_far_field_pressure_real_and_normalized() {
    let d = ok(&body(json!({"audio_duration": 0.4})));
    let au = &d["audio"];
    let sig = decode_f32(au["b64"].as_str().unwrap());
    assert_eq!(f(&au["fs"]), AUDIO_FS);
    assert_eq!(au["n"], sig.len());
    assert!(!sig.is_empty());
    assert!(sig.iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6));
    assert!(
        f(&au["peak"]) > 0.0,
        "a coupled, radiating body must produce a pressure"
    );
}

#[test]
fn work_budget_and_the_n_ceiling_are_reachable() {
    let over = sim(
        &json!({"model": "body", "N": BODY_N_MAX, "lambda": 0.9, "rho": 0.001,
                           "audio_duration": BODY_AUDIO_MAX}),
    );
    assert_eq!(over["error"]["kind"], "param");
    assert!(over["error"]["message"]
        .as_str()
        .unwrap()
        .contains("budget"));
    assert_eq!(
        BODY_WORK_MAX, 200_000,
        "the step backstop the message quotes"
    );
    for bad in [
        json!({"model": "body", "N": BODY_N_MAX + 1}),
        json!({"model": "body", "sigma_body": BODY_SIGMA_BODY_MAX + 1.0}),
        json!({"model": "body", "distance": BODY_DISTANCE_MAX + 1.0}),
    ] {
        assert_eq!(sim(&bad)["error"]["kind"], "param", "{bad}");
    }
}

#[test]
fn ignores_params_that_belong_to_other_models() {
    let base = ok(&body(json!({"bridge_stiffness": 8000})));
    let noisy = ok(&body(
        json!({"bridge_stiffness": 8000, "K": 2.0e6, "alpha": 2.3,
                                "depth": 1e-3, "kappa": 5.0, "EA": 1e4}),
    ));
    assert_eq!(noisy["meta"]["exchange"], base["meta"]["exchange"]);
    assert_eq!(noisy["meta"]["spectrum"], base["meta"]["spectrum"]);
    assert_eq!(noisy["energy"], base["energy"]);
}

/// The radiation-loaded body at `R = 0` reproduces this scene EXACTLY, from one shared param
/// dict: the rank-1 air correction is multiplied by `R`, so at zero it is a no-op.
#[test]
fn radbody_at_r_zero_is_bit_identical_to_the_readout_only_body() {
    let shared = json!({"audio_duration": 0.35, "bridge_stiffness": 8000, "N": 90,
                        "lambda": 0.85, "pluck_position": 0.27, "amplitude": 1.5e-3,
                        "distance": 1.4});
    let mut lp = json!({"model": "radbody", "radiation_R": 0.0, "sweep_points": 2,
                        "sweep_cap": 0.2});
    common::merge(&mut lp, shared.clone());
    let mut rp = json!({"model": "body"});
    common::merge(&mut rp, shared);
    let (loaded, readout) = (ok(&lp), ok(&rp));
    assert_eq!(loaded["energy"], readout["energy"]);
    assert_eq!(loaded["audio"]["b64"], readout["audio"]["b64"]);
    assert_eq!(loaded["frames"]["b64"], readout["frames"]["b64"]);
    for key in [
        "time",
        "e_string_frac",
        "e_body_frac",
        "e_conn_frac",
        "total_frac",
    ] {
        assert_eq!(
            loaded["meta"]["exchange"][key], readout["meta"]["exchange"][key],
            "{key}"
        );
    }
    assert_eq!(f(&loaded["meta"]["exchange"]["rad_frac_end"]), 0.0);
}
