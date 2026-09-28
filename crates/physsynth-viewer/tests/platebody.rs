//! A string on a distributed plate body through the payload builder — `test_web_backend.py`'s
//! platebody section.
//!
//! The same three-way split as the lumped body (the total conserves while `E_string` sloshes), plus
//! the plate's 2-D field, plus the OPPOSITE per-boundary terminus story: the supported soundboard
//! lands just below `c/2L`, the free cymbal overshoots it.

mod common;

use common::{decode_b64, decode_f32, f, ok, sim};
use physsynth_core::engine::Resonator;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_ideal::{Boundary, IdealString, Params};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::platebody::{
    PLATEBODY_AUDIO_MAX, PLATEBODY_DISTANCE_MAX, PLATEBODY_K_MAX, PLATEBODY_NPLATE_MAX,
    PLATEBODY_NSTRING_MAX, PLATEBODY_SIGMA_MAX, PLATEBODY_WORK_MAX,
};
use physsynth_viewer::AUDIO_FS;
use serde_json::{json, Value};

/// The free cymbal (the headline body) at 0.5 s.
fn pb(overrides: Value) -> Value {
    let mut p = json!({"model": "platebody", "domain": "free", "audio_duration": 0.5});
    common::merge(&mut p, overrides);
    p
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

#[test]
fn conserves_through_the_coupling_on_both_boundaries() {
    for (boundary, plate_peak) in [("free", 0.83), ("supported", 0.77)] {
        let d = ok(&pb(json!({"domain": boundary, "bridge_stiffness": 3000})));
        let e = &d["energy"];
        assert_eq!(e["sigma_is_zero"], true);
        assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL, "{boundary}");
        assert_eq!(e["lossless"]["pass"], true);
        let ex = &d["meta"]["exchange"];
        assert!(f(&ex["total_drift"]) < LOSSLESS_TOL);
        assert!(f(&ex["string_frac_max"]) > 0.9);
        assert!(
            f(&ex["string_frac_min"]) < 0.6,
            "E_string alone must visibly slosh"
        );
        assert!(
            (f(&ex["body_frac_peak"]) - plate_peak).abs() <= 0.05,
            "{boundary}"
        );
    }
}

/// Supported lands just below `c/2L`, free overshoots it; the probe is duration-robust, and the
/// overshoot needs a stiff bridge — a soft one leaves even the free end below `c/2L`.
#[test]
fn the_terminus_is_the_opposite_story_per_boundary() {
    let supp = ok(&pb(
        json!({"domain": "supported", "bridge_stiffness": 3000}),
    ))["meta"]["spectrum"]
        .clone();
    let free =
        ok(&pb(json!({"domain": "free", "bridge_stiffness": 3000})))["meta"]["spectrum"].clone();
    assert!((f(&supp["f1_free"]) - 50.0).abs() < 1e-9);
    assert!((f(&supp["f1_clamped"]) - 100.0).abs() < 1e-9);
    assert!(
        (f(&supp["terminus_f1"]) - 96.5).abs() <= 2.0,
        "supported lands just below c/2L"
    );
    assert!(
        (f(&free["terminus_f1"]) - 116.7).abs() <= 2.0,
        "free OVERSHOOTS c/2L"
    );
    assert!(f(&supp["terminus_f1"]) < f(&supp["f1_clamped"]));
    assert!(f(&supp["f1_clamped"]) < f(&free["terminus_f1"]));
    let free2 = ok(&pb(json!({"domain": "free", "bridge_stiffness": 3000,
                              "audio_duration": 2.0})))["meta"]["spectrum"]
        .clone();
    assert!(
        (f(&free2["terminus_f1"]) - f(&free["terminus_f1"])).abs() <= 1.0,
        "duration-robust"
    );
    assert!(f(&free2["terminus_f1"]) > f(&free2["f1_clamped"]));
    let soft =
        ok(&pb(json!({"domain": "free", "bridge_stiffness": 500})))["meta"]["spectrum"].clone();
    assert!(
        f(&soft["terminus_f1"]) < f(&soft["f1_clamped"]),
        "a soft bridge does NOT overshoot"
    );
    assert!(
        f(&soft["terminus_f1"]) < f(&free["terminus_f1"]),
        "the terminus climbs with K"
    );
}

/// `K = 0` severs the spring: the plate never moves, nothing radiates, and the string field is
/// bit-identical to a bare fixed/free string. The omega² read-out is `null` (0/0), where the
/// reference shipped a NaN its own server refused.
#[test]
fn k_zero_decouples_the_string_bit_for_bit() {
    let d = ok(&pb(json!({"bridge_stiffness": 0.0, "audio_duration": 0.3})));
    let ex = &d["meta"]["exchange"];
    assert_eq!(f(&ex["body_frac_peak"]), 0.0);
    assert!((f(&ex["string_frac_min"]) - 1.0).abs() < 1e-12);
    assert_eq!(f(&d["audio"]["peak"]), 0.0);
    assert!(d["meta"]["spectrum"]["omega2_consistency"].is_null());
    let fs = f(&d["fs_sim"]);
    let st = &d["string"];
    let width = st["width"].as_u64().unwrap() as usize;
    let frames = decode_f32(st["b64"].as_str().unwrap());
    let last = &frames[frames.len() - width..];
    let times = nums(&d["frame_times"]);
    let last_step = (times[times.len() - 1] * fs).round_ties_even() as usize;
    let c = (200.0f64 / 0.005).sqrt();
    let p = Params::new(
        1.0,
        200.0,
        0.005,
        c * 100.0 / 0.9,
        100,
        0.0,
        Some((Boundary::Fixed, Boundary::Free)),
    )
    .unwrap();
    let x = p.grid();
    let mut s = IdealString::new(p);
    let u0 = triangular_pluck(&x, 1.0, 0.3, 1e-3).unwrap();
    s.set_state(&u0, &vec![0.0; u0.len()]);
    for _ in 0..last_step {
        Resonator::step(&mut s).unwrap();
    }
    let bare: Vec<f32> = Resonator::state(&s).iter().map(|&v| v as f32).collect();
    assert_eq!(last, &bare[..]);
}

/// The plate is a real 2-D field that rings, and the supported plate's exterior clamps to zero in
/// every frame; the string strip rides along with the nut clamped.
#[test]
fn the_heatmap_is_a_real_2d_field_that_rings_and_stays_masked() {
    let d = ok(&pb(
        json!({"domain": "supported", "bridge_stiffness": 3000, "audio_duration": 0.3}),
    ));
    let (fr, gr) = (&d["frames"], &d["grid"]);
    assert_eq!(fr["dims"], 2);
    assert_eq!(gr["dims"], 2);
    let nf = fr["n_frames"].as_u64().unwrap() as usize;
    let (ny, nx) = (
        fr["ny"].as_u64().unwrap() as usize,
        fr["nx"].as_u64().unwrap() as usize,
    );
    let field = decode_f32(fr["b64"].as_str().unwrap());
    let mask = decode_b64(d["mask"]["b64"].as_str().unwrap());
    assert_eq!(fr["width"], nx);
    assert_eq!(d["mask"]["nx"], nx);
    assert_eq!(d["mask"]["ny"], ny);
    assert!(field.iter().all(|v| v.is_finite()));
    assert!(
        field.iter().any(|&v| v != 0.0),
        "the plate must actually ring"
    );
    for frame in 0..nf {
        for (cell, &live) in mask.iter().enumerate() {
            if live == 0 {
                assert_eq!(field[frame * nx * ny + cell], 0.0);
            }
        }
    }
    let st = &d["string"];
    let width = st["width"].as_u64().unwrap() as usize;
    assert_eq!(st["n_frames"], nf);
    assert_eq!(width, st["x"].as_array().unwrap().len());
    let sf = decode_f32(st["b64"].as_str().unwrap());
    assert!(
        sf.chunks_exact(width).all(|row| row[0] == 0.0),
        "the nut is clamped"
    );
}

#[test]
fn the_exchange_fractions_carry_e_conn_and_must_not_be_stacked() {
    let ex = ok(&pb(json!({"bridge_stiffness": 3000})))["meta"]["exchange"].clone();
    assert_eq!(ex["kind"], "platebody");
    let n = ex["time"].as_array().unwrap().len();
    for key in [
        "time",
        "e_string_frac",
        "e_body_frac",
        "e_conn_frac",
        "total_frac",
    ] {
        let len = ex[key].as_array().unwrap().len();
        assert!(len == n && len > 100, "{key}");
    }
    let (es, eb) = (nums(&ex["e_string_frac"]), nums(&ex["e_body_frac"]));
    let (ec, tot) = (nums(&ex["e_conn_frac"]), nums(&ex["total_frac"]));
    let mut min_sb = f64::INFINITY;
    for i in 0..n {
        assert!((es[i] + eb[i] + ec[i] - tot[i]).abs() <= 1e-9 + 1e-5 * tot[i].abs());
        assert!((tot[i] - 1.0).abs() <= 1e-9 + 1e-5);
        min_sb = min_sb.min(es[i] + eb[i]);
    }
    assert!(min_sb < 0.98, "E_conn carries a real share");
    let first = f(&ex["first_peak_ms"]);
    assert!(0.0 < first && first < 40.0, "{first}");
}

/// The omega² sanity is ~1 only against the plate's VOLUME displacement; the plate's own low modes
/// are the spectrum markers.
#[test]
fn the_monopole_omega2_uses_the_volume_displacement() {
    for boundary in ["free", "supported"] {
        let spx = ok(&pb(json!({"domain": boundary, "bridge_stiffness": 3000})))["meta"]
            ["spectrum"]
            .clone();
        assert!(
            (f(&spx["omega2_consistency"]) - 1.0).abs() <= 0.05,
            "{boundary}"
        );
        let modes = nums(&spx["body_modes"]);
        assert!(!modes.is_empty() && modes[0] > 0.0);
    }
}

#[test]
fn one_over_r_scales_level_and_latency_only_never_the_spectrum_shape() {
    let a = ok(&pb(json!({"distance": 1.0, "audio_duration": 0.4})))["meta"]["spectrum"].clone();
    let b = ok(&pb(json!({"distance": 4.0, "audio_duration": 0.4})))["meta"]["spectrum"].clone();
    let (ga, gb) = (f(&a["gain_times_r"]), f(&b["gain_times_r"]));
    assert!((ga - gb).abs() <= 1e-6 * gb.abs());
    let (la, lb) = (f(&a["latency_ms"]), f(&b["latency_ms"]));
    assert!((lb - 4.0 * la).abs() <= 0.05 * 4.0 * la);
    assert_eq!(a["f"], b["f"]);
    assert_eq!(a["mag"], b["mag"]);
}

#[test]
fn sigma_plate_gates_the_verdict_and_drops_the_decay_oracle() {
    let e = ok(&pb(json!({"bridge_stiffness": 3000, "sigma_plate": 20.0})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(e["lossy"].get("measured_2sigma").is_none());
    assert!(e["lossy"].get("oracle_2sigma").is_none());
}

/// The exact guard, on both boundaries, including the high-n_plate x high-K corner.
#[test]
fn the_guard_is_the_exact_bound_surfaced_as_a_clean_error_on_both_boundaries() {
    for boundary in ["free", "supported"] {
        let over = sim(&json!({"model": "platebody", "domain": boundary,
                               "bridge_stiffness": 500_000}));
        assert_eq!(over["error"]["kind"], "construction", "{boundary}");
        let corner = sim(&json!({"model": "platebody", "domain": boundary,
                                 "n_plate": PLATEBODY_NPLATE_MAX,
                                 "bridge_stiffness": PLATEBODY_K_MAX}));
        assert_eq!(corner["error"]["kind"], "construction", "{boundary} corner");
    }
    for (bad, kind) in [
        (json!({"model": "platebody", "lambda": 1.0}), "param"),
        (
            json!({"model": "platebody", "bridge_stiffness": -1.0}),
            "param",
        ),
        (json!({"model": "platebody", "domain": "clamped"}), "param"),
    ] {
        assert_eq!(sim(&bad)["error"]["kind"], kind, "{bad}");
    }
}

#[test]
fn the_audio_is_the_far_field_pressure_real_and_normalized() {
    let d = ok(&pb(json!({"audio_duration": 0.4})));
    let au = &d["audio"];
    let sig = decode_f32(au["b64"].as_str().unwrap());
    assert_eq!(f(&au["fs"]), AUDIO_FS);
    assert_eq!(au["n"], sig.len());
    assert!(!sig.is_empty());
    assert!(sig.iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6));
    assert!(f(&au["peak"]) > 0.0);
}

#[test]
fn the_work_budget_and_the_ceilings_are_reachable() {
    let over = sim(
        &json!({"model": "platebody", "N": PLATEBODY_NSTRING_MAX, "lambda": 0.9,
                           "rho": 0.001, "n_plate": PLATEBODY_NPLATE_MAX,
                           "audio_duration": PLATEBODY_AUDIO_MAX}),
    );
    assert_eq!(over["error"]["kind"], "param");
    assert!(over["error"]["message"]
        .as_str()
        .unwrap()
        .contains("budget"));
    assert_eq!(PLATEBODY_WORK_MAX, 1.0e8);
    for bad in [
        json!({"model": "platebody", "N": PLATEBODY_NSTRING_MAX + 1}),
        json!({"model": "platebody", "n_plate": PLATEBODY_NPLATE_MAX + 1}),
        json!({"model": "platebody", "sigma_plate": PLATEBODY_SIGMA_MAX + 1.0}),
        json!({"model": "platebody", "distance": PLATEBODY_DISTANCE_MAX + 1.0}),
    ] {
        assert_eq!(sim(&bad)["error"]["kind"], "param", "{bad}");
    }
}

#[test]
fn it_ignores_params_that_belong_to_other_models() {
    let base = ok(&pb(json!({"bridge_stiffness": 3000})));
    let noisy = ok(&pb(
        json!({"bridge_stiffness": 3000, "K": 2.0e6, "alpha": 2.3, "depth": 1e-3,
                              "kappa": 5.0, "EA": 1e4}),
    ));
    assert_eq!(noisy["meta"]["exchange"], base["meta"]["exchange"]);
    assert_eq!(noisy["meta"]["spectrum"], base["meta"]["spectrum"]);
    assert_eq!(noisy["energy"], base["energy"]);
}
