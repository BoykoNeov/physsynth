//! The web viewer's backend: a request's params in, a JSON payload out.
//!
//! The Rust edition of `web/serialize.py` (retirement plan §5, phase D). [`simulate_to_payload`] is
//! the whole contract the front-end in `web/static/` speaks, and it is socket-free and
//! graphics-free so that it is tested directly; `main.rs` is a thin HTTP shell over it.
//!
//! # The contract
//!
//! * **Never an exception.** A bad request comes back as `{"error": {"kind", "message"}}` — `param`
//!   when a viewer guard refused it (a cap, a range, a budget), `construction` when the physics
//!   refused it (a core constructor's own guard: CFL, a non-physical parameter). The split is kept
//!   because the front-end words the two differently, and in Rust it is no longer two exception
//!   types that keep it for free: it is the two arms of [`Refusal`].
//! * **Never a NaN.** Every float goes through [`py::num`], and a non-finite one refuses the whole
//!   payload with kind `internal` rather than shipping as `null` (which is what `serde_json` would
//!   silently have made of it — see [`py::NONFINITE`]).
//! * **Every successful payload carries a `horizon`**, either a builder's own block or a refusal
//!   naming why the scene has none ([`horizon::absent_reason`]).
//!
//! # What is ported so far
//!
//! The port is going model by model (retirement plan §23). A model key whose builder is not here
//! yet is refused with kind `unported`, so a scene can never render from a half-finished payload;
//! the Python backend stays the live viewer until the list below is complete.

pub mod bow;
pub mod energy;
pub mod horizon;
pub mod py;
pub mod resample;
pub mod server;
pub mod string;
pub mod tension;

use serde_json::{json, Value};

/// The fixed audio output rate (catch #1): `fs_sim` rides `N` and `c` and can pass the browser's
/// `AudioBuffer` cap, so the pickup is always resampled to this.
pub const AUDIO_FS: f64 = 48_000.0;

/// Every model key the front-end offers, and whether its builder has been ported.
pub const MODELS: [(&str, bool); 22] = [
    ("ideal", true),
    ("stiff", true),
    ("damped", true),
    ("tension", true),
    ("bow", true),
    ("geometric", false),
    ("sympathetic", false),
    ("jawari", false),
    ("juari", false),
    ("fret", false),
    ("bore", false),
    ("reed", false),
    ("membrane", false),
    ("mallet", false),
    ("plate", false),
    ("vk", false),
    ("body", false),
    ("platebody", false),
    ("radbody", false),
    ("airload", false),
    ("airbox", false),
    ("vkroom", false),
];

/// Why a request produced no scene.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// A viewer guard: a cap, a range, a budget, an unreadable value.
    Param(String),
    /// A core constructor's own guard.
    Construction(String),
    /// A model key whose builder has not been ported yet.
    Unported(String),
    /// A solve inside a step failed, or a payload held a non-finite number. The reference raised
    /// here and the browser saw a dropped connection; this is a payload saying so instead.
    Internal(String),
}

/// `_fnum`: read a float param, refusing (as `param`) anything `float()` would not take.
pub fn fnum(p: &Value, key: &str, default: f64) -> Result<f64, Refusal> {
    match p.get(key) {
        None => Ok(default),
        Some(v) => py::py_float(v).ok_or_else(|| {
            Refusal::Param(format!(
                "{} must be a number, got {}.",
                py::repr_str(key),
                py::py_repr(v)
            ))
        }),
    }
}

/// `str(p.get("model", "ideal"))`.
pub fn model_of(p: &Value) -> String {
    p.get("model")
        .map_or_else(|| "ideal".to_owned(), py::py_str)
}

/// `_resample_normalize`: the pickup at [`AUDIO_FS`], peak-normalized to 0.9, plus the raw peak.
///
/// `peak` is the physical displacement amplitude (metres) before normalization, so the loudness
/// story stays physical although the audio is scaled to ~0.9 full scale.
pub fn resample_normalize(x: &[f64], fs_in: f64) -> (Vec<f64>, f64) {
    let peak = if x.is_empty() { 0.0 } else { py::max_abs(x) };
    let mut y = x.to_vec();
    if x.len() > 1 && (fs_in - AUDIO_FS).abs() > 1e-6 {
        let den = py::round_int(fs_in).max(1) as u64;
        let (up, down) = resample::limit_denominator(py::round_int(AUDIO_FS) as u64, den, 2000);
        if up > 0 && down > 0 {
            y = resample::resample_poly(&y, up, down);
        }
    }
    let m = if y.is_empty() { 0.0 } else { py::max_abs(&y) };
    if m > 0.0 {
        for v in &mut y {
            *v = 0.9 * *v / m;
        }
    }
    (y, peak)
}

/// The dispatch on model key.
fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let model = model_of(p);
    match MODELS.iter().find(|(k, _)| *k == model) {
        Some((_, false)) => Err(Refusal::Unported(format!(
            "the {model:?} scene has not been ported to the Rust viewer yet \
             (docs/dev/python-retirement-plan.md section 23)."
        ))),
        _ if model == "tension" => tension::build_payload(p),
        _ if model == "bow" => bow::build_payload(p),
        // `ideal`, `stiff`, `damped` — and every unknown key, which the string builder refuses
        // with the reference's message after reading the params it reads first.
        _ => string::build_payload(p),
    }
}

/// params -> JSON payload. Never panics on bad input: a refusal becomes `{"error": {...}}`.
pub fn simulate_to_payload(params: &Value) -> Value {
    // `params or {}`: a missing body runs on the defaults.
    let empty = json!({});
    let p = if params.is_object() { params } else { &empty };
    let mut payload = match build_payload(p) {
        Ok(v) => v,
        Err(Refusal::Param(m)) => return json!({"error": {"kind": "param", "message": m}}),
        Err(Refusal::Construction(m)) => {
            return json!({"error": {"kind": "construction", "message": m}})
        }
        Err(Refusal::Unported(m)) => return json!({"error": {"kind": "unported", "message": m}}),
        Err(Refusal::Internal(m)) => return json!({"error": {"kind": "internal", "message": m}}),
    };
    if py::has_nonfinite(&payload) {
        return json!({"error": {
            "kind": "internal",
            "message": "the payload held a non-finite number; refusing rather than shipping it \
                        as null — this is a bug in the viewer, not in the request.",
        }});
    }
    if payload.get("horizon").is_none() {
        payload["horizon"] = horizon::horizon_none(horizon::absent_reason(&model_of(p)), None);
    }
    payload
}
