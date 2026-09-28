//! Python's number formats, where a refusal message or a label quotes a number.
//!
//! Every expected string below was printed by CPython 3.12 (`f"{x:g}"`, `f"{x:.2e}"`) — the
//! reference's spellings, which a Rust `format!` does not reproduce (`1e-5`, `3e0`, `1.23457e6`).

use physsynth_viewer::py::{fmt_g, sci};

#[test]
fn g_format_matches_python() {
    for (x, want) in [
        (3.0, "3"),
        (2.5, "2.5"),
        (1e-5, "1e-05"),
        (50.0, "50"),
        (0.1, "0.1"),
        (123456.0, "123456"),
        (1234567.0, "1.23457e+06"),
        (0.00012345, "0.00012345"),
        (1.23456789, "1.23457"),
        (0.0, "0"),
    ] {
        assert_eq!(fmt_g(x), want, "{x}");
    }
}

#[test]
fn e_format_matches_python() {
    for (x, want) in [
        (3.0, "3.00e+00"),
        (1e-5, "1.00e-05"),
        (1234567.0, "1.23e+06"),
        (0.00012345, "1.23e-04"),
        (1.2e9, "1.20e+09"),
    ] {
        assert_eq!(sci(x, 2), want, "{x}");
    }
    assert_eq!(sci(1000.0, 3), "1.000e+03");
}
