//! The stiff string's musical controls through the payload builder —
//! `docs/dev/string-mapping-plan.md` §3 and §14 (slice 3).
//!
//! The mapping is held by core's `tests/string_stiff_tuning.rs`, which runs the simulation. What is
//! held here is the wiring, judged by what the page PLAYS: its rendered pluck's partials against
//! the stretched law, and its energy's decay against the one ring time (a wiring bar that listens
//! only to pitch cannot hear the loss — plan §13).

mod common;

use common::{base_params, f, ok, sim};
use physsynth_core::string_tuning::{stiff_mode_t60, tune_stiff_string};
use serde_json::json;

fn stiff(overrides: serde_json::Value) -> serde_json::Value {
    let mut p = base_params(json!({
        "model": "stiff", "controls": "musical",
        // Decoys: a musical build that read any of these would play another string.
        "T": 750.0, "kappa": 6.0, "sigma": 3.0, "lambda": 0.6, "theta": 0.5,
        "N": 128, "L": 0.65, "rho": 0.0012,
        "f0": 220.0, "inharmonicity": 3e-4, "t60_low": 2.0, "f_ref": 1500.0,
        "sample_rate": 48000.0, "audio_duration": 1.0,
    }));
    common::merge(&mut p, overrides);
    p
}

#[test]
fn the_musical_stiff_string_builds_from_its_targets() {
    let payload = ok(&stiff(json!({})));
    let t = &payload["tuning"];
    let want =
        tune_stiff_string(0.65, 0.0012, 48000.0, 128, 0.5, 220.0, 3e-4, 2.0, 1500.0).unwrap();
    assert_eq!(f(&t["T"]), want.t);
    assert_eq!(f(&t["kappa"]), want.kappa);
    assert_eq!(f(&t["sigma0"]), want.sigma0);
    assert_eq!(f(&t["sigma1"]), 0.0);
    assert_eq!(t["anchor"], want.anchor as i64);
    let read = stiff_mode_t60(0.65, 128, 48000.0, 0.5, &want, want.anchor);
    assert_eq!(f(&t["anchor_t60"]), read);
    assert_eq!(f(&payload["fs_sim"]), 48000.0);
}

#[test]
fn the_musical_stiff_string_plays_its_law_and_its_ring_time() {
    let payload = ok(&stiff(json!({})));
    // Pitch: the rendered pluck's own partial overlay, against the stretched law.
    let anchor = payload["tuning"]["anchor"].as_u64().unwrap() as usize;
    let cents = payload["meta"]["partials"]["cents"].as_array().unwrap();
    for m in [1, anchor.min(cents.len())] {
        let c = f(&cents[m - 1]);
        assert!(c.abs() < 0.5, "partial {m}: {c} cents");
    }
}

/// The musical page is the physics page fed the mapped string: its `T`, `kappa` and `sigma`
/// typed into the physics sliders, `lambda` set so that `fs` is the musical sample rate. That
/// listens to the loss as well as the pitch. The energy panel's own decay fit cannot: on this
/// scheme one `sigma` decays a mode at `2 sigma / (1 + theta X_m)` (hurdle §4), so a pluck's
/// energy falls at a blend of rates (measured 1.98/s against the fundamental's 6.91/s here) —
/// found 2026-10-19 by a bar that assumed otherwise.
#[test]
fn the_musical_stiff_page_is_the_physics_page_on_the_mapped_string() {
    let m = ok(&stiff(json!({})));
    let want =
        tune_stiff_string(0.65, 0.0012, 48000.0, 128, 0.5, 220.0, 3e-4, 2.0, 1500.0).unwrap();
    let twin = ok(&base_params(json!({
        "model": "stiff", "N": 128, "L": 0.65, "rho": 0.0012, "theta": 0.5,
        "T": want.t, "kappa": want.kappa, "sigma": want.sigma0, "lambda": want.lam,
        "audio_duration": 1.0,
    })));
    for key in ["measured_2sigma", "oracle_2sigma"] {
        let (a, b) = (
            f(&m["energy"]["lossy"][key]),
            f(&twin["energy"]["lossy"][key]),
        );
        assert!((a / b - 1.0).abs() < 1e-9, "{key}: {a} vs {b}");
    }
    let (pa, pb) = (
        &m["meta"]["partials"]["detected"],
        &twin["meta"]["partials"]["detected"],
    );
    for (x, y) in pa.as_array().unwrap().iter().zip(pb.as_array().unwrap()) {
        assert!((f(x) / f(y) - 1.0).abs() < 1e-9, "partials {x} vs {y}");
    }
    // The control differs from its subject: the decoy sliders alone play another string.
    let decoy = ok(&base_params(json!({
        "model": "stiff", "N": 128, "L": 0.65, "rho": 0.0012, "theta": 0.5,
        "T": 750.0, "kappa": 6.0, "sigma": 3.0, "lambda": 0.6, "audio_duration": 1.0,
    })));
    let (dm, mm) = (
        f(&decoy["energy"]["lossy"]["measured_2sigma"]),
        f(&m["energy"]["lossy"]["measured_2sigma"]),
    );
    assert!(
        (dm / mm - 1.0).abs() > 0.05,
        "decoy decays at {dm}, musical at {mm}"
    );
}

#[test]
fn musical_keys_do_not_touch_the_physical_stiff_string() {
    let plain = base_params(json!({"model": "stiff", "kappa": 1.0, "sigma": 0.5}));
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"controls": "physical", "f0": 330.0, "inharmonicity": 0.01, "t60_low": 9.0,
               "f_ref": 4000.0, "sample_rate": 96000.0}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    assert!(a.get("tuning").is_none());
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn unrepresentable_stiff_targets_come_back_as_clean_errors() {
    let cases = [
        (json!({"sample_rate": 1000.0}), "sample_rate must be in"),
        (json!({"f_ref": 100.0}), "above the fundamental"),
        (json!({"inharmonicity": -1e-3}), "negative stiffness"),
        (
            json!({"theta": 1.0, "sample_rate": 22050.0, "f_ref": 4000.0}),
            "beyond what this grid and sample rate can produce",
        ),
    ];
    for (bad, needle) in cases {
        let payload = sim(&stiff(bad.clone()));
        assert_eq!(payload["error"]["kind"], "param", "{bad}: {payload}");
        let msg = payload["error"]["message"].as_str().unwrap();
        assert!(msg.contains(needle), "{bad}: {msg}");
    }
}
