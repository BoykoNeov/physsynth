//! The 2-D drumhead through the payload builder — `test_web_backend.py`'s membrane section.
//!
//! What a heatmap scene adds to the string contract: a decimated 2-D field whose mask is decimated
//! by the SAME stride, a display budget, the snapped geometry, and a spectrum panel marked with the
//! operator's own discrete eigenfrequencies.

mod common;

use common::{decode_b64, decode_f32, f, ok, sim};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::membrane::{membrane_lambda_max, DISPLAY_MAX, MEMBRANE_N_MAX};
use serde_json::{json, Value};

/// A short 2-D run: a small grid and brief audio keep each case fast.
fn mem(overrides: Value) -> Value {
    let mut p = json!({
        "model": "membrane", "domain": "circle",
        "T": 200.0, "rho": 0.005, "radius": 0.5,
        "N": 40, "lambda": 0.6, "sigma": 0.0,
        "pluck_x": 0.4, "pluck_y": 0.55, "pluck_width": 0.45, "amplitude": 1e-3,
        "pickup_x": 0.65, "pickup_y": 0.6,
        "audio_duration": 0.2, "animation_window": 0.04, "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

fn dim(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

#[test]
fn lossless_drift_survives_the_wrapper() {
    let d = ok(&mem(json!({})));
    assert_eq!(d["model"], "membrane");
    assert_eq!(d["frames"]["dims"], 2);
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(e.get("lossy").is_none());
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

#[test]
fn frame_bookkeeping_is_2d_for_both_domains() {
    for extra in [
        json!({"domain": "circle", "radius": 0.5}),
        json!({"domain": "rectangle", "Lx": 1.0, "Ly": 0.8}),
    ] {
        let d = ok(&mem(extra.clone()));
        let fr = &d["frames"];
        let (nf, nx, ny) = (dim(&fr["n_frames"]), dim(&fr["nx"]), dim(&fr["ny"]));
        assert!(nx <= DISPLAY_MAX && ny <= DISPLAY_MAX, "{extra}");
        assert_eq!(decode_f32(fr["b64"].as_str().unwrap()).len(), nf * nx * ny);
        assert_eq!(nf, d["frame_times"].as_array().unwrap().len());
        assert!(nf >= 2);
        assert_eq!(d["grid"]["dims"], 2);
        assert_eq!(dim(&d["grid"]["nx"]), nx);
    }
}

#[test]
fn spatial_decimation_shrinks_a_field_above_the_display_budget() {
    // N = 80: the mask is 81 x 81 (> 64), so the display grid must be coarser than the sim grid.
    let d = ok(&mem(json!({"N": 80, "audio_duration": 0.12})));
    let (nx, ny) = (dim(&d["frames"]["nx"]), dim(&d["frames"]["ny"]));
    assert!(nx < 81 && ny < 81);
    assert!(nx <= DISPLAY_MAX && ny <= DISPLAY_MAX);
}

/// The 2-D analogue of the string boundary test: a length-only check passes on byte-order
/// garbage. Every mask == 0 cell is 0 in every frame (aligned decimation), the decoded peak is the
/// `field_amp` the colour scale uses, and a disk fills about pi/4 of its box.
#[test]
fn frames_decode_to_field_values_aligned_with_the_mask() {
    let d = ok(&mem(json!({"N": 64, "audio_duration": 0.12})));
    let fr = &d["frames"];
    let (nf, nx, ny) = (dim(&fr["n_frames"]), dim(&fr["nx"]), dim(&fr["ny"]));
    let field = decode_f32(fr["b64"].as_str().unwrap());
    let mask = decode_b64(d["mask"]["b64"].as_str().unwrap());
    assert_eq!(dim(&d["mask"]["nx"]), nx);
    assert_eq!(dim(&d["mask"]["ny"]), ny);
    assert_eq!(mask.len(), nx * ny);
    let peak = field.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let amp = f(&d["field_amp"]);
    assert!((peak - amp).abs() <= 1e-5 * amp + 1e-12, "{peak} vs {amp}");
    assert!(peak > 0.0, "the strike moved the interior");
    for frame in 0..nf {
        for (cell, &live) in mask.iter().enumerate() {
            if live == 0 {
                assert_eq!(field[frame * nx * ny + cell], 0.0, "dead cell {cell} moved");
            }
        }
    }
    let live = mask.iter().filter(|&&m| m == 1).count() as f64 / mask.len() as f64;
    assert!(0.6 < live && live < 0.85, "{live}");
}

#[test]
fn rectangle_extent_uses_the_snapped_ly() {
    let d = ok(&mem(
        json!({"domain": "rectangle", "Lx": 1.2, "Ly": 0.8, "N": 48}),
    ));
    let g = &d["grid"];
    assert!((f(&g["extent_x"]) - 1.2).abs() <= 1e-9);
    assert!((f(&g["extent_y"]) - 0.8).abs() <= 0.05);
}

/// The FFT rings at the discrete fundamental — the markers come from the eigensolver, so this is
/// also the end-to-end check that it found the right lowest mode.
#[test]
fn the_spectrum_rings_at_the_discrete_fundamental() {
    let sp = ok(&mem(json!({"N": 48, "audio_duration": 0.3})))["meta"]["spectrum"].clone();
    assert!(!sp.is_null());
    assert!(!sp["modes_discrete"].as_array().unwrap().is_empty());
    assert!(!sp["modes_continuum"].as_array().unwrap().is_empty());
    assert!(f(&sp["f1_discrete"]) > 0.0);
    assert!(f(&sp["cents_fundamental"]).abs() < 5.0);
    assert!(!sp["cents_geometry"].is_null());
}

/// A grid disk keeps the square grid's symmetry, not the circle's: its cos/sin pairs of ODD
/// angular order are exact repeats (the square group's two-dimensional representation), and the
/// EVEN ones split a little. The marker list must carry both members of every exact pair — the
/// eigensolver's block exists for this (retirement plan §23.11) — and the split ones as two lines.
#[test]
fn the_disks_repeated_modes_come_back_as_pairs() {
    let sp = ok(&mem(json!({"N": 48, "audio_duration": 0.05})))["meta"]["spectrum"].clone();
    let m: Vec<f64> = sp["modes_discrete"]
        .as_array()
        .unwrap()
        .iter()
        .map(f)
        .collect();
    assert_eq!(m.len(), 12);
    // at N = 48, (angular, radial) = (1, 1), (3, 1), (1, 2) are exact; (2, 1) splits by 0.15 Hz
    let pairs = m.windows(2).filter(|w| w[0] == w[1]).count();
    assert_eq!(pairs, 3, "{m:?}");
    assert!(
        m[4] - m[3] > 0.1 && m[4] - m[3] < 0.2,
        "the even pair splits: {m:?}"
    );
    // the continuum list expands the same degeneracy, so the two line up pair for pair
    let c: Vec<f64> = sp["modes_continuum"]
        .as_array()
        .unwrap()
        .iter()
        .map(f)
        .collect();
    assert_eq!(c.len(), m.len());
    assert_eq!(c[1], c[2], "the (1, 1) Bessel mode is a cos/sin pair");
}

#[test]
fn a_lossy_run_reports_passivity() {
    let e = ok(&mem(json!({"sigma": 6.0, "audio_duration": 0.3})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert!(e.get("lossless").is_none());
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(f(&e["lossy"]["measured_2sigma"]) > 0.0);
    assert!((f(&e["lossy"]["oracle_2sigma"]) - 12.0).abs() < 1e-12);
}

#[test]
fn bad_params_give_an_error_payload() {
    for bad in [
        json!({"lambda": 1.0}),
        json!({"N": MEMBRANE_N_MAX + 1}),
        json!({"N": 1}),
        json!({"domain": "hexagon"}),
        json!({"radius": 0.0}),
        json!({"pluck_x": 0.0}),
        json!({"pickup_y": 1.0}),
        json!({"audio_duration": 0.0}),
    ] {
        let d = sim(&mem(bad.clone()));
        assert!(
            !d["error"]["message"].as_str().unwrap_or("").is_empty(),
            "{bad}"
        );
    }
}

#[test]
fn the_cfl_ceiling_is_the_2d_bar() {
    let lam = ((membrane_lambda_max() - 0.01) * 1000.0).round() / 1000.0;
    ok(&mem(json!({"lambda": lam})));
    let bad = sim(&mem(json!({"lambda": 0.9})));
    let msg = bad["error"]["message"].as_str().unwrap().to_lowercase();
    assert!(msg.contains("cfl") || msg.contains("sqrt(2)"), "{msg}");
}

#[test]
fn a_thin_rectangle_is_rejected_by_the_live_node_guard() {
    let d = sim(&mem(
        json!({"domain": "rectangle", "Lx": 0.3, "Ly": 2.0, "N": 100}),
    ));
    assert!(d["error"]["message"]
        .as_str()
        .unwrap()
        .contains("interior nodes"));
}

#[test]
fn a_small_geometry_is_rejected_by_the_work_budget_but_not_banned() {
    let d = sim(&mem(
        json!({"domain": "circle", "radius": 0.2, "N": 100, "audio_duration": 2.0}),
    ));
    assert!(d["error"]["message"]
        .as_str()
        .unwrap()
        .contains("node-steps"));
    ok(&mem(
        json!({"domain": "circle", "radius": 0.2, "N": 100, "audio_duration": 0.3}),
    ));
}
