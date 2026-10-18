//! The damped string's musical domain through the payload builder —
//! `docs/dev/parameter-mapping-plan.md` §6.
//!
//! The mapping itself is held by core's `tests/string_tuning.rs`, which runs the simulation. What
//! is held here is the wiring: the musical domain builds from the targets and not from the
//! physics sliders the front-end still sends, the physics domain is untouched by the musical keys,
//! and the drift warning and the refusals reach the payload.

mod common;

use common::{base_params, f, ok, sim};
use physsynth_core::string_tuning::{tune_damped_string, Targets};
use serde_json::{json, Value};

/// A musical request, with the physics keys set to values the mapping must ignore.
fn musical(overrides: Value) -> Value {
    let mut p = base_params(json!({
        "model": "damped", "domain": "musical",
        // Decoys: a musical build that read any of these would play a different string.
        "T": 750.0, "kappa": 6.0, "sigma0": 3.0, "sigma1": 0.009, "lambda": 0.4,
        "N": 128, "theta": 0.5, "L": 0.65, "rho": 0.0012,
        "f0": 220.0, "inharmonicity": 3e-4, "t60_low": 4.0, "t60_high": 1.0,
        "f_ref": 1500.0, "sample_rate": 48000.0,
        "audio_duration": 2.0,
    }));
    common::merge(&mut p, overrides);
    p
}

#[test]
fn the_musical_domain_builds_from_the_targets_not_the_sliders() {
    let payload = ok(&musical(json!({})));
    let t = &payload["tuning"];
    let want = tune_damped_string(
        0.65,
        0.0012,
        48000.0,
        128,
        0.5,
        &Targets {
            f0: 220.0,
            inharmonicity: 3e-4,
            t60_low: 4.0,
            t60_high: 1.0,
            f_ref: 1500.0,
        },
    )
    .unwrap();
    assert_eq!(f(&t["T"]), want.t);
    assert_eq!(f(&t["kappa"]), want.kappa);
    assert_eq!(f(&t["sigma0"]), want.sigma0);
    assert_eq!(f(&t["sigma1"]), want.sigma1);
    assert_eq!(t["anchor"], want.anchor as i64);
    assert_eq!(f(&payload["fs_sim"]), 48000.0);
    // The Courant number is a read-out here: c k N / L of the MAPPED wave speed.
    let lam = want.c / 48000.0 * 128.0 / 0.65;
    assert!(
        (f(&payload["lambda"]) - lam).abs() < 1e-6,
        "{}",
        payload["lambda"]
    );
}

#[test]
fn the_rendered_pluck_plays_the_fundamental_and_the_anchor() {
    // The payload's own partial overlay, whose reference in this domain is the asked-for law.
    let payload = ok(&musical(json!({})));
    let anchor = payload["tuning"]["anchor"].as_u64().unwrap() as usize;
    let cents = payload["meta"]["partials"]["cents"].as_array().unwrap();
    let analytic = payload["meta"]["partials"]["analytic"].as_array().unwrap();
    assert!((f(&analytic[0]) - 220.0).abs() < 1e-9);
    for m in [1, anchor] {
        let c = f(&cents[m - 1]);
        assert!(c.abs() < 0.5, "partial {m}: {c} cents");
    }
    assert!(f(&payload["meta"]["f1"]) == 220.0);
}

#[test]
fn musical_keys_do_not_touch_the_physics_domain() {
    // The front-end sends every slider to every model. A damped request in the physics domain
    // must come back byte-for-byte as if the musical keys were absent.
    let plain = base_params(json!({"model": "damped", "kappa": 1.0, "sigma0": 0.5}));
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"domain": "physical", "f0": 330.0, "inharmonicity": 0.01, "t60_low": 9.0,
               "t60_high": 0.1, "f_ref": 4000.0, "sample_rate": 96000.0}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    assert!(a.get("tuning").is_none());
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn a_coarse_low_stiff_string_warns_with_the_number() {
    let payload = ok(&musical(json!({
        "N": 64, "L": 1.0, "rho": 0.005, "theta": 0.28, "sample_rate": 44100.0,
        "f0": 82.4, "inharmonicity": 1e-2, "f_ref": 2000.0, "audio_duration": 0.3,
    })));
    let t = &payload["tuning"];
    assert_eq!(t["warn"], true, "{t}");
    let worst = f(&t["worst_drift_cents"]);
    assert!(worst > 15.0 && worst < 35.0, "{worst}");
    let msg = t["message"].as_str().unwrap();
    assert!(msg.contains("cents off"), "{msg}");
}

#[test]
fn a_fine_string_reports_no_warning() {
    let payload = ok(&musical(json!({
        "N": 256, "sample_rate": 96000.0, "f0": 440.0, "inharmonicity": 1e-4,
        "f_ref": 2000.0, "audio_duration": 0.3,
    })));
    assert_eq!(payload["tuning"]["warn"], false, "{}", payload["tuning"]);
}

#[test]
fn unrepresentable_targets_come_back_as_clean_errors() {
    let cases = [
        (json!({"sample_rate": 1000.0}), "sample_rate must be in"),
        (json!({"t60_high": 8.0}), "negative loss"),
        (json!({"f_ref": 100.0}), "above the fundamental"),
        (json!({"inharmonicity": -1e-3}), "negative stiffness"),
        (json!({"f0": 0.0}), "f0 must be positive"),
    ];
    for (bad, needle) in cases {
        let payload = sim(&musical(bad.clone()));
        assert_eq!(payload["error"]["kind"], "param", "{bad}: {payload}");
        let msg = payload["error"]["message"].as_str().unwrap();
        assert!(msg.contains(needle), "{bad}: {msg}");
    }
}
