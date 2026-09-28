//! The von Kármán plate through the payload builder — `test_web_backend.py`'s vk section.
//!
//! No analytic modal oracle: energy conservation at a CONVERGED fixed point is the correctness
//! test, and the pitch hardens with amplitude above the linear markers.

mod common;

use common::{f, ok, sim};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::vk::{VK_N_MAX, VK_WOVERE_MAX};
use serde_json::{json, Value};

fn vk(overrides: Value) -> Value {
    let mut p = json!({
        "model": "vk", "domain": "supported",
        "E": 2.0e11, "e": 1.0e-3, "nu": 0.3, "rho": 7800.0,
        "Lx": 0.15, "Ly": 0.15, "N": 14, "fs": 48000.0, "sigma": 0.0,
        "nonlinear": true, "w_over_e": 3.0,
        "pluck_x": 0.5, "pluck_y": 0.5, "pluck_width": 0.28,
        "pickup_x": 0.47, "pickup_y": 0.53,
        "audio_duration": 0.12, "animation_window": 0.01, "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

#[test]
fn the_supported_gong_conserves_converges_and_hardens() {
    let d = ok(&vk(json!({})));
    assert_eq!(d["model"], "vk");
    assert_eq!(d["boundary"], "supported");
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert_eq!(e["convergence"]["all_converged"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let sp = &d["meta"]["spectrum"];
    assert_eq!(sp["kind"], "vk");
    assert!(!sp["f0_detected"].is_null());
    assert!(f(&sp["shift_pct"]) > 5.0, "genuine hardening");
}

/// `nonlinear = false` is the linear plate: no convergence block, no glide.
#[test]
fn the_linear_toggle_has_no_convergence_block_and_no_shift() {
    for flag in [json!(false), json!("false")] {
        let d = ok(&vk(json!({"nonlinear": flag, "w_over_e": 0.5})));
        assert_eq!(d["nonlinear"], false, "{flag}");
        assert!(d["energy"].get("convergence").is_none());
        assert_eq!(d["energy"]["lossless"]["pass"], true);
        assert!(f(&d["meta"]["spectrum"]["shift_pct"]).abs() < 1.0);
    }
}

/// The free-edge crash is a mode wash: it conserves, and reports no fundamental rather than a
/// lying number.
#[test]
fn the_free_cymbal_conserves_without_a_fundamental() {
    let d = ok(&vk(
        json!({"domain": "free", "Lx": 0.2, "Ly": 0.2, "N": 14, "w_over_e": 3.0}),
    ));
    assert_eq!(d["boundary"], "free");
    assert_eq!(d["energy"]["convergence"]["all_converged"], true);
    assert!(f(&d["energy"]["lossless"]["drift"]) < LOSSLESS_TOL);
    let sp = &d["meta"]["spectrum"];
    assert_eq!(sp["kind"], "vk");
    assert!(sp["f0_detected"].is_null() && sp["shift_pct"].is_null());
}

/// Lossy: passivity, with the convergence gate still riding along.
#[test]
fn a_lossy_plate_reports_passivity_and_keeps_its_convergence_gate() {
    let e = ok(&vk(
        json!({"sigma": 3.0, "w_over_e": 2.0, "audio_duration": 0.12}),
    ))["energy"]
        .clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert!(e.get("lossless").is_none());
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(e.get("convergence").is_some());
}

#[test]
fn bad_params_give_an_error_payload() {
    for bad in [
        json!({"fs": 500.0}),
        json!({"fs": 200000.0}),
        json!({"N": 1}),
        json!({"N": VK_N_MAX + 1}),
        json!({"domain": "clamped"}),
        json!({"E": 0.0}),
        json!({"e": 0.0}),
        json!({"w_over_e": 0.0}),
        json!({"w_over_e": VK_WOVERE_MAX + 1.0}),
        json!({"audio_duration": 0.0}),
    ] {
        let d = sim(&vk(bad.clone()));
        assert!(
            !d["error"]["message"].as_str().unwrap_or("").is_empty(),
            "{bad}"
        );
    }
}
