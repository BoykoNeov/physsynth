//! The wind pages' musical controls through the payload builder — `docs/dev/wind-mapping-plan.md`.
//!
//! The bore's mapping is held by core's `tests/bore_tuning.rs`, which reads the one-step map and
//! the render. What is held here is the wiring, judged by what the page PLAYS: its measured
//! partials against the odd series of the target, its ring time off the energy it has shed, and
//! the musical page as the physics page fed the mapped tube.

mod common;

use common::{f, ok, sim};
use physsynth_core::wind_tuning::tune_bore;
use serde_json::{json, Value};

/// The page's speed of sound, written out, not imported.
const C0: f64 = 343.0;

/// A bore request off the page's defaults, with decoys a musical build must not read: at
/// `L = 1.5` the tube sounds 57 Hz, and `10^-1` is a bell forty times lossier than the mapping's.
fn bore(overrides: Value) -> Value {
    let mut p = json!({
        "model": "bore", "controls": "musical", "domain": "radiating",
        "N": 100, "L": 1.5, "bell_ratio_exp": -1.0,
        "f0": 200.0, "t60_low": 2.0,
        "pickup_position": 0.23, "audio_duration": 0.5, "animation_window": 0.02,
        "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

/// Numbers within `rel` of each other, everything else equal, recursively; `b64` strings (float32
/// renderings of numbers that may differ in their last bits) are skipped.
fn assert_close(a: &Value, b: &Value, rel: f64, path: &str) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let (kx, ky): (Vec<_>, Vec<_>) = (x.keys().collect(), y.keys().collect());
            assert_eq!(kx, ky, "{path}: keys differ");
            for (k, v) in x {
                if k != "b64" {
                    assert_close(v, &y[k], rel, &format!("{path}.{k}"));
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: lengths differ");
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                assert_close(u, v, rel, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(_), Value::Number(_)) => {
            let (u, v) = (f(a), f(b));
            let scale = u.abs().max(v.abs()).max(1e-300);
            // An absolute floor for roundoff-sized read-outs (the energy drift, ~1e-15).
            assert!((u - v).abs() <= rel * scale + 1e-12, "{path}: {u} vs {v}");
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

fn refusal(p: &Value) -> String {
    let payload = sim(p);
    assert_eq!(payload["error"]["kind"], "param", "{payload}");
    payload["error"]["message"].as_str().unwrap().to_owned()
}

// -- the bore ---------------------------------------------------------------------------------

#[test]
fn musical_keys_do_not_touch_the_physical_bore() {
    for domain in ["radiating", "open"] {
        let mut named = bore(json!({"controls": "physical", "domain": domain, "L": 0.7}));
        let a = ok(&named);
        named.as_object_mut().unwrap().remove("controls");
        named.as_object_mut().unwrap().remove("f0");
        named.as_object_mut().unwrap().remove("t60_low");
        assert_eq!(a, ok(&named), "{domain}");
        assert!(a.get("tuning").is_none());
    }
}

#[test]
fn the_musical_bore_is_the_physics_bore_on_the_mapped_tube() {
    for (domain, f0, t60) in [
        ("radiating", 200.0, 2.0),
        ("radiating", 61.7, 0.3),
        ("open", 440.0, 2.0),
    ] {
        let musical = ok(&bore(json!({"domain": domain, "f0": f0, "t60_low": t60})));
        let tu = tune_bore(f0, (domain == "radiating").then_some(t60), C0).unwrap();
        let mut twin = bore(json!({"controls": "physical", "domain": domain, "L": tu.l}));
        if let Some(r) = tu.r_ratio {
            twin["bell_ratio_exp"] = json!(r.log10());
        }
        let mut stripped = musical.clone();
        let t = stripped
            .as_object_mut()
            .unwrap()
            .remove("tuning")
            .expect("a tuning block");
        // 10^log10(r) is the ratio to a few ulps, so the twin is close, not bit-equal.
        assert_close(&stripped, &ok(&twin), 1e-9, domain);
        assert_eq!(f(&t["L"]), tu.l);
        assert_eq!(t["r_ratio"], tu.r_ratio.map_or(Value::Null, |r| json!(r)));
    }
}

#[test]
fn the_musical_bore_plays_the_odd_series_of_its_pitch() {
    // A build that read the decoy L would sound 57 Hz: three of the partials below would be a
    // semitone and more away.
    for (f0, n) in [(200.0, 100), (97.3, 64)] {
        let payload = ok(&bore(json!({"f0": f0, "N": n, "audio_duration": 1.0})));
        let measured = payload["meta"]["spectrum"]["partials"]["measured"]
            .as_array()
            .unwrap();
        assert!(measured.len() >= 3);
        for (i, v) in measured.iter().enumerate() {
            let want = (2 * i + 1) as f64 * f0;
            let c = 1200.0 * (f(v) / want).log2();
            assert!(
                c.abs() < 1.0,
                "f0 {f0}: partial {} at {} Hz, {c} cents",
                i + 1,
                f(v)
            );
        }
    }
}

#[test]
fn the_musical_bore_rings_its_ring_time() {
    // At lambda = 1 the whole state returns every round trip scaled by |rho|, so after a whole
    // number of round trips (f0 D is whole: 100, 20 and 200 periods) the air keeps exactly
    // 10^(-6 D / T60) of the energy it started with: the rest has left through the bell.
    for (t60, d) in [(2.0, 0.5), (0.4, 0.1), (9.0, 1.0)] {
        let payload = ok(&bore(json!({"t60_low": t60, "audio_duration": d})));
        let left = 1.0 - f(&payload["meta"]["radiated_frac"]);
        let want = 10f64.powf(-6.0 * d / t60);
        assert!(
            (left / want - 1.0).abs() < 1e-4,
            "T60 {t60}: {left} of the energy left after {d} s, want {want}"
        );
    }
}

#[test]
fn the_open_end_sets_the_pitch_alone() {
    let payload = ok(&bore(
        json!({"domain": "open", "f0": 300.0, "t60_low": 0.3}),
    ));
    let t = &payload["tuning"];
    assert_eq!(t["t60"], Value::Null);
    assert_eq!(t["r_ratio"], Value::Null);
    assert!(t["message"].as_str().unwrap().contains("rings forever"));
    assert_eq!(f(&payload["meta"]["radiated_frac"]), 0.0);
    // The ring-time slider is not read: a value no bell could hold changes nothing.
    let other = ok(&bore(
        json!({"domain": "open", "f0": 300.0, "t60_low": 1e6}),
    ));
    assert_eq!(payload, other);
}

#[test]
fn a_pitch_outside_the_tube_is_refused_with_the_range() {
    for f0 in [40.0, 900.0] {
        let msg = refusal(&bore(json!({"f0": f0})));
        assert!(msg.contains("f0 must be in [42.88, 857.5] Hz"), "{msg}");
    }
}

#[test]
fn a_ring_time_past_the_bell_names_the_longest_at_that_pitch() {
    // At 800 Hz the tube is 0.107 m and the bell's floor R/Z0 = 1e-4 rings 21.59 s.
    let msg = refusal(&bore(json!({"f0": 800.0, "t60_low": 30.0, "N": 32})));
    assert!(msg.contains("the longest is 21.59 s"), "{msg}");
    // Just inside it renders.
    ok(&bore(json!({"f0": 800.0, "t60_low": 21.0, "N": 32})));
}

#[test]
fn the_budget_refusal_names_the_highest_pitch_that_fits() {
    // N = 256, 1.5 s: fs = 4 N f0, so (300000 - 256) / (4 * 256 * 1.5) = 195.15 Hz fits (printed
    // 195.1).
    let msg = refusal(&bore(json!({"f0": 400.0, "N": 256, "audio_duration": 1.5})));
    assert!(
        msg.contains("the highest pitch that fits is 195.1 Hz"),
        "{msg}"
    );
    ok(&bore(json!({"f0": 195.0, "N": 256, "audio_duration": 1.5})));
}
