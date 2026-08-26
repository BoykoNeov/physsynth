//! Formatting a float the way Python's `repr()` does — for error messages, and only for those.
//!
//! # Why this exists
//!
//! The project's convention, set in Phase 0, is that a ported rejection reproduces the Python
//! original's message **verbatim**, because `tests/` matches on the text. Most ported messages
//! interpolate a float with an explicit precision (`{:.6}`), where Rust and Python already agree.
//! `exciter::triangular_pluck` does not: it interpolates bare, `f"... (L={L}), got {position}."`,
//! and there the two languages disagree on the very first case anyone tries.
//!
//! ```text
//! Python  repr(1.0)  -> "1.0"        Rust  format!("{}", 1.0)  -> "1"
//! Python  repr(1e-5) -> "1e-05"      Rust  format!("{}", 1e-5) -> "0.00001"
//! ```
//!
//! Rust's `Debug` for `f64` is much closer — it is the same shortest-round-trip algorithm, and it
//! keeps the `.0` — so it is the starting point. What is left is the exponent spelling: Python
//! always signs it and pads it to two digits (`1e+16`, `1e-05`), where `Debug` writes `1e16` and
//! `1e-5`. That, plus `NaN` versus `nan`, is the whole difference over the range this project
//! could plausibly reach.
//!
//! **This is for messages only.** Nothing numeric depends on it, and it should never be used to
//! serialise a value — a round-trip through text is not how any state crosses this boundary.

/// `repr(x)` as Python would print it.
///
/// Handles the three ways Rust's `Debug` and Python's `repr` diverge: `NaN`/`nan`, an unsigned
/// exponent, and a one-digit exponent.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }

    let s = format!("{x:?}");
    let Some(epos) = s.find(['e', 'E']) else {
        return s;
    };
    let (mantissa, exp) = s.split_at(epos);
    let exp = &exp[1..];
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("+", exp.strip_prefix('+').unwrap_or(exp)),
    };
    if digits.len() < 2 {
        format!("{mantissa}e{sign}0{digits}")
    } else {
        format!("{mantissa}e{sign}{digits}")
    }
}

/// `f"{x:.<prec>e}"` as Python formats it.
///
/// Rust's `{:e}` writes the exponent bare (`1.234e-5`, `0.000e0`); Python signs it and pads it to
/// two digits (`1.234e-05`, `0.000e+00`). Same divergence [`py_float`] fixes for `repr`, and the
/// same fix — only the exponent is touched, never the mantissa, so the digits are Rust's own
/// correctly-rounded ones.
pub fn py_exp(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let s = format!("{x:.prec$e}");
    let Some(epos) = s.find('e') else {
        return s;
    };
    let (mantissa, exp) = s.split_at(epos);
    let exp = &exp[1..];
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("+", exp.strip_prefix('+').unwrap_or(exp)),
    };
    if digits.len() < 2 {
        format!("{mantissa}e{sign}0{digits}")
    } else {
        format!("{mantissa}e{sign}{digits}")
    }
}

#[cfg(test)]
mod tests {
    use super::{py_exp, py_float};

    #[test]
    fn integral_values_keep_their_point_zero() {
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py_float(-3.0), "-3.0");
        assert_eq!(py_float(0.0), "0.0");
    }

    #[test]
    fn ordinary_values_are_the_shortest_round_trip() {
        assert_eq!(py_float(0.65), "0.65");
        assert_eq!(py_float(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(py_float(1e-4), "0.0001");
    }

    #[test]
    fn exponents_are_signed_and_two_digits() {
        assert_eq!(py_float(1e-5), "1e-05");
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(1.5e-7), "1.5e-07");
        assert_eq!(py_float(1e100), "1e+100");
    }

    #[test]
    fn the_non_finite_spellings_are_pythons() {
        assert_eq!(py_float(f64::NAN), "nan");
        assert_eq!(py_float(f64::INFINITY), "inf");
        assert_eq!(py_float(f64::NEG_INFINITY), "-inf");
    }
    #[test]
    fn the_exponent_form_matches_pythons_format_spec() {
        // f"{0.0:.3e}" == "0.000e+00";  f"{1e-14:.1e}" == "1.0e-14" -- the two the refusal uses.
        assert_eq!(py_exp(0.0, 3), "0.000e+00");
        assert_eq!(py_exp(1e-14, 1), "1.0e-14");
        assert_eq!(py_exp(1.2345e-5, 3), "1.234e-05");
        assert_eq!(py_exp(-6.02e23, 2), "-6.02e+23");
        assert_eq!(py_exp(1e100, 1), "1.0e+100");
        assert_eq!(py_exp(f64::NAN, 3), "nan");
    }
}
