//! The resolution-horizon read-out — the viewer half of `docs/dev/resolution-horizon-plan.md`.
//!
//! The physics is asserted against the primitives in `physsynth-analysis`; what is asserted here is
//! that the viewer asks them the right question about the right scene. The membrane's and the
//! plate's 2-D rows are here; the von Kármán plate's arrive with that scene's builder.

mod common;

use common::{band, base_params, f, horizon, ok, sim};
use physsynth_analysis::horizon::sinc_horizon_fraction;
use physsynth_viewer::horizon::{HORIZON_ABSENT, HORIZON_BANDS, HORIZON_CENTS_DEFAULT};
use physsynth_viewer::py::{has_nonfinite, num, NONFINITE};
use physsynth_viewer::MODELS;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// A short rectangular membrane — the reference's `_membrane_params(domain="rectangle", ...)`.
fn rect_membrane(n: i64, extra: Value) -> Value {
    let mut p = json!({
        "model": "membrane", "domain": "rectangle",
        "T": 200.0, "rho": 0.005, "radius": 0.5, "N": n, "lambda": 0.6, "sigma": 0.0,
        "pluck_x": 0.4, "pluck_y": 0.55, "pluck_width": 0.45, "amplitude": 1e-3,
        "pickup_x": 0.65, "pickup_y": 0.6,
        "audio_duration": 0.05, "animation_window": 0.04, "playback_speed": 0.02,
    });
    common::merge(&mut p, extra);
    p
}

/// The reference's `_plate_params`, short.
fn plate(extra: Value) -> Value {
    let mut p = json!({
        "model": "plate", "domain": "supported",
        "kappa": 20.0, "rho": 0.005, "Lx": 1.0, "Ly": 1.0,
        "N": 40, "mu": 1.0, "sigma": 0.0, "nu": 0.3,
        "pluck_x": 0.4, "pluck_y": 0.55, "pluck_width": 0.3, "amplitude": 1e-3,
        "pickup_x": 0.62, "pickup_y": 0.58,
        "audio_duration": 0.2, "animation_window": 0.02, "playback_speed": 0.02,
    });
    common::merge(&mut p, extra);
    p
}

/// The models whose builders compute a read-out; with `HORIZON_ABSENT` this partitions the list.
const HORIZON_MEASURED: [&str; 11] = [
    "ideal", "stiff", "damped", "bow", "jawari", "juari", "fret", "membrane", "mallet", "plate",
    "vk",
];

/// The model keys the front-end's `<select id="model">` offers, read out of the markup.
fn offered_models() -> BTreeSet<String> {
    let html = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../web/static/index.html"
    ))
    .expect("the front-end's index.html");
    let start = html
        .find("<select id=\"model\">")
        .expect("the model <select> moved; this guard derives its population from it");
    let rest = &html[start..];
    let select = &rest[..rest.find("</select>").expect("a closed select")];
    select
        .split("<option value=\"")
        .skip(1)
        .map(|s| s[..s.find('"').expect("a closed attribute")].to_owned())
        .collect()
}

#[test]
fn every_model_the_viewer_offers_is_classified() {
    let offered = offered_models();
    assert!(offered.len() >= 20, "{offered:?}");
    let measured: BTreeSet<String> = HORIZON_MEASURED.iter().map(|s| (*s).to_owned()).collect();
    let absent: BTreeSet<String> = HORIZON_ABSENT
        .iter()
        .map(|(k, _)| (*k).to_owned())
        .collect();
    assert!(measured.is_disjoint(&absent));
    let union: BTreeSet<String> = measured.union(&absent).cloned().collect();
    assert_eq!(union, offered);
    // The dispatch table covers exactly the same population, so a model offered in the markup can
    // never fall through to the string builder's "unknown model".
    let dispatched: BTreeSet<String> = MODELS.iter().map(|(k, _)| (*k).to_owned()).collect();
    assert_eq!(dispatched, offered);
}

#[test]
fn a_prefix_block_carries_every_field_the_frontend_reads() {
    let block = horizon(&base_params(json!({})));
    assert_eq!(block["kind"], "prefix");
    let keys: BTreeSet<&str> = block
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let want: BTreeSet<&str> = [
        "kind",
        "scheme",
        "dims",
        "of",
        "n_modes",
        "f_max",
        "nyquist",
        "default_cents",
        "bands",
    ]
    .into_iter()
    .collect();
    assert_eq!(keys, want);
    assert_eq!(f(&block["default_cents"]), HORIZON_CENTS_DEFAULT);
    let cents: Vec<f64> = block["bands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| f(&b["cents"]))
        .collect();
    assert_eq!(cents, HORIZON_BANDS.to_vec());
    let band_keys: BTreeSet<&str> = [
        "cents",
        "modes",
        "hz",
        "saturated",
        "limited_by",
        "limit_hz",
        "limit_cents",
        "index",
        "family",
        "family_tied",
        "families",
        "monotone",
    ]
    .into_iter()
    .collect();
    for b in block["bands"].as_array().unwrap() {
        let k: BTreeSet<&str> = b.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(k, band_keys);
        assert!(b["hz"].is_null() || f(&b["hz"]) > 0.0);
        let fams = b["families"].as_array().unwrap();
        assert!(!fams.is_empty());
        for fam in fams {
            let k: BTreeSet<&str> = fam
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(k, ["index", "monotone", "name"].into_iter().collect());
        }
    }
}

#[test]
fn is_absent_from_an_error_payload() {
    let payload = sim(&base_params(json!({"N": 1})));
    assert!(payload.get("error").is_some() && payload.get("horizon").is_none());
}

#[test]
fn the_explicit_string_at_lambda_one_is_in_tune_across_the_whole_grid() {
    let block = horizon(&base_params(json!({"model": "ideal", "lambda": 1.0})));
    let b = band(&block, 5.0);
    assert_eq!(b["saturated"], true);
    assert_eq!(b["modes"], 63);
    assert_eq!(block["n_modes"], 63); // N - 1, the whole resolvable spectrum
    assert!(b["limited_by"].is_null() && b["limit_hz"].is_null());
    assert_eq!(b["hz"], block["f_max"]);
}

#[test]
fn refining_the_explicit_timestep_makes_the_read_out_worse() {
    let modes: Vec<u64> = [1.0, 0.9, 0.75, 0.5]
        .iter()
        .map(|&lam| {
            band(
                &horizon(&base_params(json!({"model": "ideal", "lambda": lam}))),
                5.0,
            )["modes"]
                .as_u64()
                .unwrap()
        })
        .collect();
    let mut sorted = modes.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(modes, sorted);
    assert!((modes[3] as f64) < modes[0] as f64 / 5.0, "{modes:?}");
}

#[test]
fn the_theta_string_cannot_pass_its_own_space_floor() {
    let floor = sinc_horizon_fraction(5.0, 1).unwrap() * 128.0;
    let got: Vec<f64> = [1.0, 0.5, 0.125]
        .iter()
        .map(|&lam| {
            f(&band(
                &horizon(&base_params(json!({
                    "model": "damped", "N": 128, "kappa": 0.0, "sigma0": 0.0, "sigma1": 0.0,
                    "lambda": lam,
                }))),
                5.0,
            )["modes"])
        })
        .collect();
    assert!(got.windows(2).all(|w| w[0] <= w[1]), "{got:?}"); // more rate never costs modes
    let top = got.iter().copied().fold(0.0, f64::max);
    assert!(top <= floor + 1.0, "{got:?} vs {floor}"); // and never passes the floor
    assert!(top >= floor - 3.0, "{got:?} vs {floor}"); // a floor nothing approaches is not one
}

#[test]
fn is_built_from_the_scheme_and_not_from_the_display_arrays() {
    for n in [64, 128, 256] {
        let block = horizon(&base_params(
            json!({"model": "damped", "N": n, "lambda": 1.0}),
        ));
        assert_eq!(block["n_modes"], n - 1);
    }
    // the plate panel ships 6 markers; the read-out's mode set is the whole grid's
    let plate_block = horizon(&plate(json!({})));
    assert_eq!(plate_block["n_modes"], 39 * 39);
}

/// One select carries three plates, and only the supported rectangle has a horizon: the free
/// plate's reference is a table, the guitar's error is its staircase.
#[test]
fn the_plate_key_covers_three_plates_and_only_one_has_a_horizon() {
    assert_eq!(
        horizon(&plate(json!({"domain": "supported"})))["kind"],
        "prefix"
    );
    for (domain, needle) in [("free", "tabulated"), ("guitar", "staircased")] {
        let block = horizon(&plate(
            json!({"domain": domain, "N": 24, "audio_duration": 0.05}),
        ));
        assert_eq!(block["kind"], "none", "{domain}");
        assert!(
            block["reason"].as_str().unwrap().contains(needle),
            "{block}"
        );
    }
}

#[test]
fn a_tighter_bound_can_only_shorten_the_claim() {
    for params in [
        base_params(json!({"model": "ideal", "lambda": 0.8})),
        base_params(json!({"model": "stiff", "N": 96, "kappa": 4.0})),
        rect_membrane(24, json!({})),
        plate(json!({})),
    ] {
        let bands = horizon(&params)["bands"].as_array().unwrap().clone();
        for pair in bands.windows(2) {
            let (tight, loose) = (&pair[0], &pair[1]);
            assert!(f(&tight["cents"]) < f(&loose["cents"]));
            assert!(f(&tight["modes"]) <= f(&loose["modes"]));
            assert!(f(&tight["index"]) <= f(&loose["index"]));
            if !tight["hz"].is_null() {
                assert!(f(&tight["hz"]) <= f(&loose["hz"]));
            }
        }
    }
}

#[test]
fn the_hertz_ceiling_stops_below_the_first_mode_that_is_out_of_tune() {
    for params in [
        base_params(json!({"model": "damped", "N": 96})),
        rect_membrane(24, json!({})),
    ] {
        let b = band(&horizon(&params), 5.0);
        assert_eq!(b["saturated"], false);
        assert!(!b["hz"].is_null(), "{b}");
        assert!(f(&b["limit_hz"]) > f(&b["hz"]), "{b}");
        assert!(f(&b["limit_cents"]).abs() > f(&b["cents"]), "{b}");
    }
}

/// Why a 2-D scene ships two numbers: a block is not a family, and the index reading counts only
/// the square block whose corners are all in tune — the smaller claim, by a factor of four at 25
/// cents on this membrane.
#[test]
fn the_2d_readings_differ_and_the_index_one_is_the_conservative_one() {
    let block = horizon(&rect_membrane(24, json!({})));
    assert_eq!(block["dims"], 2);
    for b in block["bands"].as_array().unwrap() {
        assert!(f(&b["index"]) <= f(&b["modes"]), "{b}");
    }
    let loose = band(&block, 25.0);
    assert!(f(&loose["modes"]) > f(&loose["index"]), "{loose}");
}

/// At the 2-D CFL ceiling the membrane's DIAGONAL family is exact and the axial ones are not, so
/// the corner that limits a block is axial — the plate's rule inverted (plan §10.6). Below the
/// ceiling the three families land on one integer and the payload says they tie.
#[test]
fn a_membranes_worst_corner_is_axial_at_the_courant_ceiling() {
    let lmax = 1.0 / 2.0f64.sqrt();
    let ceiling = band(&horizon(&rect_membrane(32, json!({"lambda": lmax}))), 5.0);
    let fam = ceiling["family"].as_str().unwrap();
    assert!(
        fam.contains("axial") && !fam.contains("diagonal"),
        "{ceiling}"
    );
    let idx = |name: &str| {
        ceiling["families"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["name"] == name)
            .map(|x| f(&x["index"]))
            .unwrap()
    };
    let diag = idx("diagonal (m, m)");
    assert!(
        diag > idx("axial (m, 1)") && diag > idx("axial (1, n)"),
        "{ceiling}"
    );
    let low = band(&horizon(&rect_membrane(32, json!({"lambda": 0.3}))), 5.0);
    assert_eq!(low["family_tied"], true, "{low}");
}

/// A staircased disk is refused with the mechanism named, not quoted.
#[test]
fn a_circular_membrane_is_refused_as_a_staircase() {
    let mut p = rect_membrane(40, json!({"domain": "circle"}));
    common::merge(&mut p, json!({"audio_duration": 0.05}));
    let block = horizon(&p);
    assert_eq!(block["kind"], "none");
    assert!(
        block["reason"].as_str().unwrap().contains("staircased"),
        "{block}"
    );
    assert_eq!(block["of"], "the membrane");
}

#[test]
fn the_1d_prefix_and_index_readings_are_the_same_list() {
    for params in [
        base_params(json!({"model": "ideal", "lambda": 0.8})),
        base_params(json!({"model": "stiff", "kappa": 3.0})),
        base_params(json!({"model": "damped", "N": 96})),
    ] {
        let block = horizon(&params);
        assert_eq!(block["dims"], 1);
        for b in block["bands"].as_array().unwrap() {
            assert_eq!(b["modes"], b["index"], "{b}");
            let names: Vec<&str> = b["families"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x["name"].as_str().unwrap())
                .collect();
            assert_eq!(names, ["harmonic"]);
        }
    }
}

#[test]
fn every_refusal_names_its_mechanism_not_just_its_absence() {
    let keywords = [
        ("tension", "nonlinear"),
        ("geometric", "nonlinear"),
        ("bore", "Webster"),
        ("reed", "Webster"),
        ("sympathetic", "spring-coupled bridge"),
        ("body", "spring-coupled bridge"),
        ("radbody", "spring-coupled bridge"),
        ("airload", "spring-coupled bridge"),
        ("platebody", "spring-coupled bridge"),
        ("airbox", "direction-dependent"),
        ("vkroom", "direction-dependent"),
    ];
    let table: BTreeSet<&str> = HORIZON_ABSENT.iter().map(|(k, _)| *k).collect();
    let named: BTreeSet<&str> = keywords.iter().map(|(k, _)| *k).collect();
    assert_eq!(table, named);
    for (model, needle) in keywords {
        let reason = HORIZON_ABSENT.iter().find(|(k, _)| *k == model).unwrap().1;
        assert!(reason.contains(needle), "{model}: {reason}");
        assert!(
            reason.ends_with('.') && reason.len() > 100,
            "{model}: {reason}"
        );
    }
}

// -- "never a NaN", with teeth

#[test]
fn a_non_finite_float_is_a_marker_the_top_level_refuses_not_a_silent_null() {
    // What serde_json alone would do — the reason the marker exists.
    assert_eq!(Value::from(f64::NAN), Value::Null);
    assert_eq!(num(f64::NAN), json!(NONFINITE));
    assert_eq!(num(f64::INFINITY), json!(NONFINITE));
    assert!(has_nonfinite(&json!({"a": [1.0, {"b": NONFINITE}]})));
    assert!(!has_nonfinite(&json!({"a": [1.0, null, "text"]})));
}

#[test]
fn survives_strict_json() {
    for params in [
        base_params(json!({})),
        base_params(json!({"model": "damped", "N": 96})),
        rect_membrane(24, json!({})),
        rect_membrane(40, json!({"domain": "circle"})),
        plate(json!({})),
    ] {
        let payload = ok(&params);
        assert!(!has_nonfinite(&payload));
        let text = serde_json::to_string(&payload["horizon"]).unwrap();
        assert!(
            !text.contains("NaN") && !text.contains("Infinity"),
            "{text}"
        );
    }
}
