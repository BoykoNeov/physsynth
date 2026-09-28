//! Python's semantics where the payload builder depended on them — the coercions, the rounding,
//! the reprs, and the two NumPy idioms the viewer leans on.
//!
//! `web/serialize.py` read its request with `float(p.get(key, default))`, `int(...)` and `str(...)`,
//! and those accept more than "a JSON number": a numeric string, a bool (`float(True) == 1.0`), a
//! float for an integer (`int(64.7) == 64`). The front-end never sends most of these, but a deep
//! link can, and the reference's *refusal messages* quote the offending value in Python's `repr`,
//! which `tests/` matched on. So each coercion here is the Python one, including what it refuses.
//!
//! Two differences are deliberate and are the only ones:
//!
//! * `int(float("inf"))` is an `OverflowError` in Python, which `serialize.py` did not catch — the
//!   request died with a 500. Here it is the same refusal a NaN gets. A request can only carry an
//!   infinity as a string (`"inf"`), since JSON has no literal for one.
//! * A dict in a refusal message prints its keys sorted, because `serde_json`'s map is ordered by
//!   key rather than by insertion. Only a malformed request can reach it.

use serde_json::{Map, Number, Value};

/// Marker a non-finite float becomes, so the top level can refuse the payload rather than ship it.
///
/// `serde_json` turns a non-finite `f64` into `null` *silently* (`Value::from(f64::NAN)` is
/// `Null`), where the Python server's `json.dumps(..., allow_nan=False)` raised. Every float in a
/// payload goes through [`num`], which emits this string instead, and
/// [`crate::simulate_to_payload`] walks the finished tree for it. A "never NaN" test is then a
/// test of something rather than of `serde_json`'s quiet substitution.
pub const NONFINITE: &str = "\u{0}non-finite float\u{0}";

/// A float for the payload: a JSON number, or [`NONFINITE`] if it is not finite.
pub fn num(x: f64) -> Value {
    match Number::from_f64(x) {
        Some(n) => Value::Number(n),
        None => Value::String(NONFINITE.to_owned()),
    }
}

/// An integer for the payload.
pub fn int(i: i64) -> Value {
    Value::Number(Number::from(i))
}

/// `None` for `None`, else [`num`] — the `float | None` fields.
pub fn opt_num(x: Option<f64>) -> Value {
    x.map_or(Value::Null, num)
}

/// True when a payload holds a [`NONFINITE`] marker anywhere.
pub fn has_nonfinite(v: &Value) -> bool {
    match v {
        Value::String(s) => s == NONFINITE,
        Value::Array(a) => a.iter().any(has_nonfinite),
        Value::Object(m) => m.values().any(has_nonfinite),
        _ => false,
    }
}

// -- coercions ------------------------------------------------------------------------------------

/// `float(v)`: `None` exactly where Python raises `TypeError` or `ValueError`.
pub fn py_float(v: &Value) -> Option<f64> {
    match v {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        Value::String(s) => parse_float_str(s),
        _ => None,
    }
}

/// Why `int(v)` refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntError {
    /// `TypeError` or `ValueError` — what `serialize.py` caught.
    Refused,
    /// `ValueError: cannot convert float NaN to integer`.
    NaN,
    /// `OverflowError` — uncaught in the reference; see the module header.
    Infinite,
}

/// `int(v)` for a JSON value: bools are 0/1, floats truncate, strings must be integer literals.
///
/// Values past `i64` saturate — Python's ints do not, but every caller range-checks the result
/// against a bound some fourteen orders of magnitude smaller, so the refusal is the same one.
pub fn py_int(v: &Value) -> Result<i64, IntError> {
    match v {
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if n.as_u64().is_some() {
                Ok(i64::MAX)
            } else {
                float_to_int(n.as_f64().ok_or(IntError::Refused)?)
            }
        }
        Value::String(s) => parse_int_str(s).ok_or(IntError::Refused),
        _ => Err(IntError::Refused),
    }
}

/// `int(x)` for a float: truncation toward zero, with Python's two refusals.
pub fn float_to_int(x: f64) -> Result<i64, IntError> {
    if x.is_nan() {
        Err(IntError::NaN)
    } else if x.is_infinite() {
        Err(IntError::Infinite)
    } else {
        Ok(x.trunc() as i64)
    }
}

/// `round(x)` with no `ndigits`: to the nearest integer, ties to even (banker's rounding).
///
/// Not `f64::round`, which rounds ties away from zero. `round(2.5) == 2` in Python, and the index
/// arithmetic (`pickup_idx`, step counts, strides) inherits that.
pub fn round_int(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `round(x, ndigits)` on a float: the correctly-rounded decimal, ties to even, read back.
///
/// Rust's fixed-precision formatting is correctly rounded on the exact binary value with ties to
/// even, which is exactly CPython's `float.__round__` (both are dtoa in mode 3); checked against
/// Python on halfway values, a negative zero and 1e300 before this was written. Non-finite values
/// pass through, as in Python.
pub fn round_nd(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}")
        .parse()
        .expect("a formatted finite float parses")
}

/// `_as_bool(v, default)`: a bool is itself, absent/`null` is the default, a number is its
/// truthiness, and anything else is `str(v).strip().lower() in ("1", "true", "yes", "on")`.
pub fn as_bool(v: Option<&Value>, default: bool) -> bool {
    match v {
        None | Some(Value::Null) => default,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_none_or(|x| x != 0.0),
        Some(other) => matches!(
            py_str(other).trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
    }
}

/// `str(v)` for a JSON value: a string is itself, anything else its repr.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// `repr(v)` for a JSON value, as Python prints the object `json.loads` would have made.
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => {
            if n.is_f64() {
                physsynth_core::fmt::py_float(n.as_f64().unwrap_or(f64::NAN))
            } else {
                n.to_string()
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(m) => repr_dict(m),
    }
}

fn repr_dict(m: &Map<String, Value>) -> String {
    let parts: Vec<String> = m
        .iter()
        .map(|(k, v)| format!("{}: {}", repr_str(k), py_repr(v)))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// `repr(s)` for a str: single quotes unless the text holds a `'` and no `"`.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Strip Python's digit-group underscores: one at a time, and only between two digits.
fn strip_underscores(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' {
            let prev = i.checked_sub(1).map(|j| b[j]);
            let next = b.get(i + 1).copied();
            if !(prev.is_some_and(|p| p.is_ascii_digit())
                && next.is_some_and(|n| n.is_ascii_digit()))
            {
                return None;
            }
        } else {
            out.push(c as char);
        }
    }
    Some(out)
}

/// `float(str)`: surrounding whitespace, a sign, `inf`/`infinity`/`nan` in any case, underscores.
fn parse_float_str(s: &str) -> Option<f64> {
    let t = strip_underscores(s.trim())?;
    let body = t.strip_prefix(['+', '-']).unwrap_or(&t);
    let special = matches!(
        body.to_ascii_lowercase().as_str(),
        "inf" | "infinity" | "nan"
    );
    // Rust's parser is correctly rounded like CPython's; what it accepts beyond Python's grammar
    // is nothing a decimal literal can spell, so the only guard needed is on the characters.
    if !special
        && !body
            .bytes()
            .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return None;
    }
    t.parse().ok()
}

/// `int(str)`: surrounding whitespace, a sign, decimal digits with single underscores.
fn parse_int_str(s: &str) -> Option<i64> {
    let t = strip_underscores(s.trim())?;
    let (neg, digits) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, &t[..]),
    };
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mag: i64 = digits.parse().unwrap_or(i64::MAX);
    Some(if neg { -mag } else { mag })
}

// -- NumPy idioms ---------------------------------------------------------------------------------

/// `np.linspace(0, last, num).astype(int)` — the decimation index every trace panel uses.
///
/// NumPy computes `i * step` with `step = last / (num - 1)` and then *overwrites* the final entry
/// with `last`; the cast truncates. Reproduced exactly, since a different index is a different
/// sample in the shipped trace.
pub fn linspace_idx(last: usize, num: usize) -> Vec<usize> {
    match num {
        0 => Vec::new(),
        1 => vec![0],
        _ => {
            let step = last as f64 / (num - 1) as f64;
            let mut idx: Vec<usize> = (0..num).map(|i| (i as f64 * step) as usize).collect();
            idx[num - 1] = last;
            idx
        }
    }
}

/// `_finite_list`: floats to a JSON list, non-finite values to `null`, optionally rounded.
pub fn finite_list(arr: &[f64], ndigits: Option<usize>) -> Value {
    Value::Array(
        arr.iter()
            .map(|&v| {
                if !v.is_finite() {
                    Value::Null
                } else {
                    num(ndigits.map_or(v, |nd| round_nd(v, nd)))
                }
            })
            .collect(),
    )
}

/// `_finite_list` over an index selection — `_finite_list(arr[idx], ...)`.
pub fn finite_list_at(arr: &[f64], idx: &[usize], ndigits: Option<usize>) -> Value {
    let picked: Vec<f64> = idx.iter().map(|&i| arr[i]).collect();
    finite_list(&picked, ndigits)
}

/// `np.max(np.abs(a))` — NaN-propagating, and `0.0` for an empty slice where the caller guarded.
pub fn max_abs(a: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in a {
        let v = v.abs();
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// `f"{n:,}"` — an integer with comma thousands separators, as the budget refusals print it.
pub fn commas(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// `np.mean` of a 1-D float array: NumPy's pairwise sum, divided by the length.
pub fn np_mean(a: &[f64]) -> f64 {
    physsynth_core::reduce::sum(a) / a.len() as f64
}

/// `np.dot` of two vectors, left to right.
///
/// **Not** NumPy's value to the bit: `np.dot` is BLAS `ddot`, which fuses its multiply-adds and
/// picks its kernel by CPU (findings §14.2), so there is no scalar recipe that reproduces it. Every
/// caller here feeds a gated or rounded quantity, and the one-time comparison measures what the
/// difference reaches.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// `np.linalg.norm` of a vector — `sqrt(dot(x, x))`, with [`dot`]'s caveat.
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `scipy.ndimage.uniform_filter1d(x, size, mode="nearest")` — a sliding mean, edge-extended.
///
/// SciPy keeps a RUNNING SUM and divides on output: `tmp += new - old; out = tmp / size`. That
/// order was found by trying the three candidates against the installed SciPy 1.17.1 on random
/// data (sizes 1..257, lengths 5..5000): this one matched every sample, while
/// dividing inside the update or compensating the sum each missed tens of thousands.
pub fn uniform_filter1d_nearest(x: &[f64], size: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let size = size.max(1);
    let s1 = size / 2;
    let s2 = size - s1 - 1;
    let mut ext = vec![x[0]; s1];
    ext.extend_from_slice(x);
    ext.extend(std::iter::repeat_n(x[n - 1], s2));
    let mut out = Vec::with_capacity(n);
    let mut tmp = 0.0;
    for v in &ext[..size] {
        tmp += v;
    }
    out.push(tmp / size as f64);
    for ll in 1..n {
        tmp += ext[ll + size - 1] - ext[ll - 1];
        out.push(tmp / size as f64);
    }
    out
}

// -- base64 ---------------------------------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding — `base64.b64encode`.
pub fn b64encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `_b64f32`: little-endian float32, base64 — what `app.js` decodes with a `DataView`.
///
/// `as f32` is round-to-nearest-even, which is NumPy's `astype('<f4')`.
pub fn b64f32(a: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(a.len() * 4);
    for &v in a {
        bytes.extend_from_slice(&(v as f32).to_le_bytes());
    }
    b64encode(&bytes)
}

/// `_b64u8`: one byte per element — the decimated domain mask.
pub fn b64u8(a: &[u8]) -> String {
    b64encode(a)
}
