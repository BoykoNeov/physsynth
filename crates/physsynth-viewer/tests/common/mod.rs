//! Shared helpers for the viewer's payload tests — what the top of `tests/test_web_backend.py` was.
#![allow(dead_code)]

use serde_json::{json, Map, Value};

/// `simulate_to_payload` on a params object.
pub fn sim(params: &Value) -> Value {
    physsynth_viewer::simulate_to_payload(params)
}

/// The canonical short run (c = 200, f1 = 100 Hz), with overrides merged over it.
pub fn base_params(overrides: Value) -> Value {
    let mut p = json!({
        "model": "ideal",
        "L": 1.0, "T": 200.0, "rho": 0.005,
        "N": 64, "lambda": 1.0, "sigma": 0.0,
        "pluck_position": 0.3, "amplitude": 1e-3, "pickup_position": 0.1,
        "audio_duration": 0.3, "animation_window": 0.05, "playback_speed": 0.02,
    });
    merge(&mut p, overrides);
    p
}

/// `dict.update`.
pub fn merge(p: &mut Value, overrides: Value) {
    let m: &mut Map<String, Value> = p.as_object_mut().expect("params are an object");
    if let Value::Object(o) = overrides {
        for (k, v) in o {
            m.insert(k, v);
        }
    }
}

/// A successful payload, or a panic naming the refusal.
pub fn ok(params: &Value) -> Value {
    let payload = sim(params);
    assert!(payload.get("error").is_none(), "{}", payload["error"]);
    payload
}

/// Inverse of `_b64f32` — what the front-end's `DataView` does.
pub fn decode_f32(b64: &str) -> Vec<f32> {
    let bytes = decode_b64(b64);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Standard base64 decode (padding allowed).
pub fn decode_b64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    };
    let mut out = Vec::new();
    for chunk in s.as_bytes().chunks(4) {
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= if c == b'=' { 0 } else { val(c) } << (18 - 6 * i);
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    out
}

/// A JSON number as `f64`, panicking with the path on anything else.
pub fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// The read-out block of a successful payload.
pub fn horizon(params: &Value) -> Value {
    let payload = ok(params);
    payload["horizon"].clone()
}

/// The band at `cents` of a prefix read-out.
pub fn band(block: &Value, cents: f64) -> Value {
    assert_eq!(block["kind"], "prefix", "{block}");
    block["bands"]
        .as_array()
        .expect("bands")
        .iter()
        .find(|b| b["cents"].as_f64() == Some(cents))
        .expect("a band at that bound")
        .clone()
}
