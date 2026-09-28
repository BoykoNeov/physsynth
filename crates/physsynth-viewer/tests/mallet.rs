//! A mallet striking the drumhead through the payload builder — `test_web_backend.py`'s mallet
//! section (model #7).
//!
//! A closed mass + felt + membrane system, so the verdict is conservation; the headline is the
//! contact episode — restitution near 1 and a head that keeps almost nothing — never tuned away.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::mallet::MALLET_N_MAX;
use serde_json::{json, Value};

/// A short mallet run: small grid, brief audio (FDTD plus a root-find per step).
fn mal(overrides: Value) -> Value {
    let mut p = json!({
        "model": "mallet", "domain": "circle",
        "T": 200.0, "rho": 0.005, "radius": 0.5,
        "N": 40, "lambda": 0.5, "sigma": 0.0,
        "mass": 0.02, "stiffness": 5.0e4, "alpha": 2.3, "hysteresis": 0.0,
        "strike_velocity": 3.0, "pluck_x": 0.5, "pluck_y": 0.5,
        "pickup_x": 0.65, "pickup_y": 0.6,
        "audio_duration": 0.3, "animation_window": 0.04, "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

#[test]
fn a_lossless_strike_conserves_through_the_wrapper() {
    let d = ok(&mal(json!({})));
    assert_eq!(d["model"], "mallet");
    assert_eq!(d["frames"]["dims"], 2);
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(e.get("lossy").is_none());
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

/// A point mass is an inefficient exciter: it bounces off with restitution near 1 and the head
/// keeps a tiny fraction, after a large transient dimple mid-contact.
#[test]
fn bounces_with_near_unity_restitution_and_the_head_barely_rings() {
    let sp = ok(&mal(json!({"N": 60})))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "mallet");
    assert_eq!(sp["separated"], true);
    assert!(f(&sp["restitution"]) > 0.99);
    assert!(f(&sp["final_head_pct"]) < 0.5);
    assert!(f(&sp["peak_head_pct"]) > 30.0);
    assert!(f(&sp["contact_ms"]) > 0.0 && f(&sp["peak_force"]) > 0.0);
}

/// The strike marker is the SNAPPED contact node, in fractions — where the felt landed.
#[test]
fn the_strike_marker_reports_the_snapped_node_in_fractions() {
    let sp = ok(&mal(json!({"N": 40, "pluck_x": 0.5, "pluck_y": 0.5})))["meta"]["spectrum"].clone();
    for key in ["strike_fx", "strike_fy"] {
        let v = f(&sp[key]);
        assert!(0.0 < v && v < 1.0 && (v - 0.5).abs() <= 0.03, "{key} = {v}");
    }
}

/// The audio is the membrane's modal ring, not the near-field dimple: its dominant peak rides the
/// discrete fundamental.
#[test]
fn the_audio_is_the_ring_not_the_dimple() {
    let d = ok(&mal(json!({"N": 48, "audio_duration": 0.4})));
    let a: Vec<f64> = decode_f32(d["audio"]["b64"].as_str().unwrap())
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    assert!(a.iter().all(|v| v.is_finite()));
    let peak = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!((peak - 0.9).abs() <= 1e-4, "{peak}");
    let w = hann(a.len());
    let x: Vec<f64> = a.iter().zip(&w).map(|(s, wi)| s * wi).collect();
    let mag = rfft_mag(&x);
    let freqs = rfftfreq(a.len(), 1.0 / f(&d["audio"]["fs"]));
    let top = (0..mag.len()).fold(0, |b, i| if mag[i] > mag[b] { i } else { b });
    let f1 = f(&d["meta"]["f1"]);
    assert!((freqs[top] - f1).abs() < 0.1 * f1, "{} vs {f1}", freqs[top]);
}

/// Loss (sigma or felt hysteresis) gives pure passivity with the `2 sigma` line ABSENT: the
/// struck system's energy floors at the departing mallet's kinetic energy.
#[test]
fn a_lossy_strike_reports_passivity_without_a_decay_oracle() {
    for extra in [
        json!({"N": 48, "sigma": 6.0}),
        json!({"N": 48, "hysteresis": 3.0e4}),
    ] {
        let e = ok(&mal(extra.clone()))["energy"].clone();
        assert_eq!(e["sigma_is_zero"], false, "{extra}");
        assert!(e.get("lossless").is_none());
        assert_eq!(e["lossy"]["monotone"], true);
        assert!(e["lossy"].get("measured_2sigma").is_none());
        assert!(e["lossy"].get("oracle_2sigma").is_none());
    }
}

#[test]
fn hysteresis_lowers_restitution() {
    let elastic = f(&ok(&mal(json!({"N": 48})))["meta"]["spectrum"]["restitution"]);
    let hyst =
        f(&ok(&mal(json!({"N": 48, "hysteresis": 3.0e4})))["meta"]["spectrum"]["restitution"]);
    assert!(hyst < elastic, "{hyst} vs {elastic}");
}

#[test]
fn bad_params_give_an_error_payload() {
    for bad in [
        json!({"lambda": 1.0}),
        json!({"N": MALLET_N_MAX + 1}),
        json!({"mass": 0.0}),
        json!({"stiffness": -1.0}),
        json!({"alpha": 0.5}),
        json!({"strike_velocity": 0.0}),
        json!({"pluck_x": 1.5}),
        json!({"audio_duration": 5.0}),
    ] {
        let d = sim(&mal(bad.clone()));
        assert!(d["error"].get("message").is_some(), "{bad}");
    }
}

#[test]
fn a_small_geometry_is_rejected_by_the_work_budget_but_short_audio_fits() {
    let heavy = sim(&mal(
        json!({"domain": "rectangle", "Lx": 0.6, "Ly": 0.8, "N": 80,
                                "audio_duration": 2.0}),
    ));
    assert!(heavy["error"]["message"]
        .as_str()
        .unwrap()
        .contains("node-steps"));
    ok(&mal(
        json!({"domain": "rectangle", "Lx": 0.6, "Ly": 0.8, "N": 80,
                   "audio_duration": 0.2}),
    ));
}

/// The two `N` defaults, carried: the pre-check reads 60 and the membrane underneath reads 80, so
/// a request with no `N` renders the drum at 80 (the reference's behaviour, pinned).
#[test]
fn with_no_n_the_drum_is_built_at_the_membranes_default() {
    let mut p = mal(json!({"audio_duration": 0.02}));
    p.as_object_mut().unwrap().remove("N");
    let d = ok(&p);
    let lam = 0.5;
    let fs_at = |n: f64| (200.0f64 / 0.005).sqrt() / (lam * (2.0 * 0.5 / n));
    assert!(
        (f(&d["fs_sim"]) - fs_at(80.0)).abs() < 1e-3,
        "{}",
        d["fs_sim"]
    );
}

/// An exciter inherits the horizon of what it drives: a struck rectangle reports exactly the
/// membrane's read-out, naming the struck membrane. The reference's test of this row said so in
/// its docstring and asserted only the bow; this is the mallet half it described.
#[test]
fn a_struck_rectangle_reads_exactly_like_the_membrane_on_its_own() {
    let shared = json!({"domain": "rectangle", "Lx": 1.0, "Ly": 1.0, "N": 24, "lambda": 0.5,
                        "audio_duration": 0.02});
    let struck = ok(&mal(shared.clone()))["horizon"].clone();
    let mut mp = mal(shared);
    common::merge(&mut mp, json!({"model": "membrane"}));
    let plain = ok(&mp)["horizon"].clone();
    assert_eq!(struck["kind"], "prefix");
    assert_eq!(struck["of"], "the struck membrane");
    assert_eq!(plain["of"], "the membrane");
    let strip = |mut v: Value| {
        v.as_object_mut().unwrap().remove("of");
        v
    };
    assert_eq!(strip(struck), strip(plain));
}
