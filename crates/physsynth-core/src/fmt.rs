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

/// `f"{x:.<prec>g}"` as Python formats it.
///
/// The third spelling this project's refusals need, after [`py_float`] and [`py_exp`]: the grain
/// guards print their computed floor and ceiling with `:.6g`, and the split-contradiction message
/// prints `:.12g`. Rust has no `{:g}`, so the rule is transcribed — it is CPython's, and it is
/// short: render at `prec` significant digits, choose scientific when the decimal exponent is
/// below `-4` or at least `prec`, and strip the trailing zeros either way.
///
/// The digits themselves are Rust's own correctly-rounded ones; only the *choice* of notation and
/// the trailing-zero strip are transcribed, exactly as [`py_exp`] touches only the exponent.
pub fn py_general(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    // `%g` treats a precision of 0 as 1 -- CPython's `format_float_short` does, and so does C.
    let p = prec.max(1);

    // Render once in scientific notation to learn the exponent AFTER rounding: 9.99 at two
    // significant digits is 1.0e+01, not 9.9e+00, and it is the rounded exponent that picks the
    // notation.
    let sci = format!("{:.*e}", p - 1, x);
    let epos = sci.find('e').expect("{:e} always writes an exponent");
    let exp: i32 = sci[epos + 1..]
        .parse()
        .expect("{:e} writes a decimal exponent");

    if exp < -4 || exp >= p as i32 {
        let mantissa = strip_trailing_zeros(&sci[..epos]);
        let (sign, digits) = if exp < 0 {
            ("-", (-exp).to_string())
        } else {
            ("+", exp.to_string())
        };
        if digits.len() < 2 {
            format!("{mantissa}e{sign}0{digits}")
        } else {
            format!("{mantissa}e{sign}{digits}")
        }
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_trailing_zeros(&format!("{x:.decimals$}"))
    }
}

/// Drop a fractional part's trailing zeros, and the point with them if nothing is left.
///
/// Only touches a string that already has a `.`; an integral rendering is returned unchanged,
/// which is what makes `py_general(0.0, 6)` come back `"0"` rather than `"0."`.
fn strip_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_owned();
    }
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

#[cfg(test)]
mod tests {
    use super::{py_exp, py_float, py_general};

    #[test]
    fn general_format_matches_pythons_g() {
        // Every expectation is `format(v, '.6g')` / `format(v, '.12g')` read off CPython. The
        // interesting rows are the ones where the two notations meet: 999999.5 rounds UP to an
        // exponent of 6 at six significant digits and so flips to scientific, while at twelve it
        // does not -- which is why the exponent has to be read AFTER the rounding.
        let cases: &[(f64, &str, &str)] = &[
            (0.0, "0", "0"),
            (-0.0, "-0", "-0"),
            (1.0, "1", "1"),
            (-1.0, "-1", "-1"),
            (0.5, "0.5", "0.5"),
            (-0.5, "-0.5", "-0.5"),
            (1e-5, "1e-05", "1e-05"),
            (1.23456789, "1.23457", "1.23456789"),
            (-1.23456789e-7, "-1.23457e-07", "-1.23456789e-07"),
            (123456789.0, "1.23457e+08", "123456789"),
            (1e16, "1e+16", "1e+16"),
            (0.1, "0.1", "0.1"),
            (999999.5, "1e+06", "999999.5"),
            (9.99e-5, "9.99e-05", "9.99e-05"),
            (0.25, "0.25", "0.25"),
            (1.0 / 3.0, "0.333333", "0.333333333333"),
        ];
        for &(v, six, twelve) in cases {
            assert_eq!(py_general(v, 6), six, "{v:?} at .6g");
            assert_eq!(py_general(v, 12), twelve, "{v:?} at .12g");
        }
    }

    #[test]
    fn general_format_of_the_two_grain_guards() {
        // The values these actually print: -sqrt(g_x*g_y) and sqrt(g_x*g_y) for a spruce-ish
        // plate, and a contradicted split's H. Pinned because the message is compared to the
        // Python original's character for character.
        assert_eq!(py_general(-(1.0f64 * 0.073).sqrt(), 6), "-0.270185");
        assert_eq!(py_general((1.0f64 * 0.073).sqrt(), 6), "0.270185");
        assert_eq!(py_general(0.0269 + 2.0 * 0.063, 12), "0.1529");
    }

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
