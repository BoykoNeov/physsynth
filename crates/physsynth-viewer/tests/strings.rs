//! The linear strings through the payload builder — the string half of `test_web_backend.py`.
//!
//! These pin the contract the front-end depends on: the energy *signature* survives the wrapper
//! unaltered, the frame and audio bookkeeping decodes to the right values (not just the right
//! length), the energy report is gated by loss, and every bad request comes back as a clean error
//! payload rather than a panic.

mod common;

use common::{base_params, decode_f32, f, ok, sim};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::AUDIO_FS;
use serde_json::json;

// -- criterion 1: the energy signature survives the wrapper

#[test]
fn lossless_drift_survives_wrapper() {
    let payload = ok(&base_params(json!({})));
    let energy = &payload["energy"];
    assert_eq!(energy["sigma_is_zero"], true);
    assert!(energy.get("lossy").is_none());
    assert!(f(&energy["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(energy["lossless"]["pass"], true);
}

#[test]
fn all_three_models_build() {
    for model in ["ideal", "stiff", "damped"] {
        let extra = if model == "ideal" {
            json!({"model": model})
        } else {
            json!({"model": model, "kappa": 1.0})
        };
        let payload = ok(&base_params(extra));
        assert_eq!(payload["model"], model);
        assert!(!payload["meta"]["partials"].is_null(), "{model}");
        assert!(
            f(&payload["energy"]["lossless"]["drift"]) < LOSSLESS_TOL,
            "{model}"
        );
    }
}

// -- frame / audio bookkeeping

#[test]
fn frame_bookkeeping() {
    let payload = ok(&base_params(json!({})));
    let frames = &payload["frames"];
    let n = frames["n_frames"].as_u64().unwrap() as usize;
    let w = frames["width"].as_u64().unwrap() as usize;
    assert_eq!(w, payload["grid"]["x"].as_array().unwrap().len()); // N + 1
    assert_eq!(decode_f32(frames["b64"].as_str().unwrap()).len(), n * w);
    assert_eq!(n, payload["frame_times"].as_array().unwrap().len());
    assert!(n >= 2);
    assert!(f(&payload["anim_dt"]) > 0.0);
}

#[test]
fn frames_decode_to_field_values_and_boundary() {
    // A length-only check passes on byte-order garbage, so pin the VALUES: the decoded peak is
    // `field_amp`, and the fixed ends stay clamped every frame.
    let payload = ok(&base_params(json!({})));
    let n = payload["frames"]["n_frames"].as_u64().unwrap() as usize;
    let w = payload["frames"]["width"].as_u64().unwrap() as usize;
    let grid = decode_f32(payload["frames"]["b64"].as_str().unwrap());
    let peak = grid.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let amp = f(&payload["field_amp"]);
    assert!((peak - amp).abs() <= 1e-5 * amp + 1e-9, "{peak} vs {amp}");
    for row in 0..n {
        assert!(f64::from(grid[row * w]).abs() < 1e-9);
        assert!(f64::from(grid[row * w + w - 1]).abs() < 1e-9);
    }
    assert!(peak > 0.0);
}

#[test]
fn audio_resampled_and_normalized() {
    let payload = ok(&base_params(json!({"N": 64})));
    let audio = &payload["audio"];
    assert_eq!(f(&audio["fs"]), AUDIO_FS);
    let samples = decode_f32(audio["b64"].as_str().unwrap());
    assert_eq!(samples.len() as u64, audio["n"].as_u64().unwrap());
    assert!(samples.iter().all(|s| s.is_finite()));
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.0 && peak <= 1.0 + 1e-6);
    assert!(f(&audio["peak"]) > 0.0);
}

#[test]
fn high_n_audio_stays_in_browser_range() {
    let payload = ok(&base_params(json!({"N": 512, "audio_duration": 0.2})));
    assert!(f(&payload["fs_sim"]) > AUDIO_FS); // the sim ran far above the browser cap...
    assert_eq!(f(&payload["audio"]["fs"]), AUDIO_FS); // ...but the delivered audio did not
}

// -- loss-gated energy

#[test]
fn lossy_reports_passivity_not_drift() {
    let payload = ok(&base_params(
        json!({"model": "damped", "sigma0": 2.0, "sigma1": 1e-4, "kappa": 1.0}),
    ));
    let energy = &payload["energy"];
    assert_eq!(energy["sigma_is_zero"], false);
    assert!(energy.get("lossless").is_none());
    let lossy = &energy["lossy"];
    assert_eq!(lossy["monotone"], true);
    assert!(f(&lossy["measured_2sigma"]) > 0.0);
    assert!((f(&lossy["oracle_2sigma"]) - 4.0).abs() < 1e-12); // 2 * sigma0
}

// -- clean error payloads

#[test]
fn cfl_violation_is_clean_error() {
    let payload = sim(&base_params(json!({"model": "ideal", "lambda": 1.5})));
    assert_eq!(payload["error"]["kind"], "construction");
    let msg = payload["error"]["message"].as_str().unwrap().to_lowercase();
    assert!(msg.contains("lambda") || msg.contains("cfl"), "{msg}");
}

#[test]
fn stiff_admits_lambda_above_one() {
    let payload = ok(&base_params(
        json!({"model": "stiff", "kappa": 1.0, "lambda": 1.5}),
    ));
    assert!((f(&payload["lambda"]) - 1.5).abs() < 1.5e-6);
}

#[test]
fn bad_params_give_error_payload() {
    for bad in [
        json!({"N": 1}),
        json!({"N": 99999}),
        json!({"pluck_position": 0.0}),
        json!({"pickup_position": 1.0}),
        json!({"audio_duration": 0.0}),
        json!({"animation_window": -1.0}),
        json!({"model": "banana"}),
        json!({"rho": 0.0}),
    ] {
        let payload = sim(&base_params(bad.clone()));
        let msg = payload["error"]["message"].as_str().unwrap_or("");
        assert!(!msg.is_empty(), "{bad} gave {payload}");
    }
}

#[test]
fn none_params_does_not_crash() {
    ok(&json!({}));
    // `params or {}` — a body that is not an object runs on the defaults too.
    ok(&json!(null));
}

// -- the reference's messages, verbatim, because the front-end shows them

#[test]
fn refusal_messages_are_the_references_word_for_word() {
    let cases = [
        (json!({"N": 1}), "param", "N must be in [2, 2000], got 1."),
        (
            json!({"model": "banana"}),
            "param",
            "unknown model 'banana' (expected 'ideal' | 'stiff' | 'damped' | 'tension' | 'bow').",
        ),
        (
            json!({"lambda": 1.5}),
            "construction",
            "CFL violated: lambda = c*k/h = 1.500000 > 1. Reduce fs, refine the grid (increase \
             N), or lower the wave speed.",
        ),
        (
            json!({"model": "stiff", "kappa": 1.0, "theta": 0.0}),
            "construction",
            "theta must be in (0, 1], got 0.0.",
        ),
        (
            json!({"N": null}),
            "param",
            "N must be an integer, got None.",
        ),
        (
            json!({"N": [64]}),
            "param",
            "N must be an integer, got [64].",
        ),
        (
            json!({"N": "64.0"}),
            "param",
            "N must be an integer, got '64.0'.",
        ),
        (
            json!({"amplitude": {"a": 1}}),
            "param",
            "'amplitude' must be a number, got {'a': 1}.",
        ),
        (
            json!({"pickup_position": "nan"}),
            "param",
            "pickup_position must be in (0, 1), got nan.",
        ),
        (
            json!({"lambda": 0.0}),
            "param",
            "lambda must be > 0, got 0.0.",
        ),
        (
            json!({"model": "damped", "sigma0": -1.0}),
            "construction",
            "sigma0 (frequency-independent loss) must be >= 0.",
        ),
        (
            json!({"L": "inf"}),
            "construction",
            "L, T, rho, fs must all be positive.",
        ),
        // `True` is an int in Python: N = 1, refused by the RANGE check, not the type check.
        (
            json!({"N": true}),
            "param",
            "N must be in [2, 2000], got 1.",
        ),
        (
            json!({"model": null}),
            "param",
            "unknown model 'None' (expected 'ideal' | 'stiff' | 'damped' | 'tension' | 'bow').",
        ),
    ];
    for (bad, kind, message) in cases {
        let payload = sim(&base_params(bad.clone()));
        assert_eq!(payload["error"]["kind"], kind, "{bad}");
        assert_eq!(payload["error"]["message"], message, "{bad}");
    }
}

#[test]
fn float_and_int_coercions_accept_what_python_accepted() {
    // A deep link can carry strings; `float()` and `int()` took them, so the payload must be the
    // same scene as the numeric request.
    let numeric = ok(&base_params(json!({"N": 64, "amplitude": 0.002})));
    for spelled in [
        json!({"N": "64", "amplitude": "0.002"}),
        json!({"N": 64.7, "amplitude": "  0.002 "}),
    ] {
        let p = ok(&base_params(spelled.clone()));
        assert_eq!(p["audio"], numeric["audio"], "{spelled}");
        assert_eq!(p["energy"], numeric["energy"], "{spelled}");
    }
}

#[test]
fn params_belonging_to_other_models_are_ignored() {
    let plain = ok(&base_params(json!({})));
    let leaky = ok(&base_params(
        json!({"kappa": 5.0, "sigma0": 3.0, "EA": 1e5, "domain": "circle"}),
    ));
    assert_eq!(plain, leaky);
}
