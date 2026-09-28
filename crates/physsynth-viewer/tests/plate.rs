//! The Kirchhoff plate through the payload builder — `test_web_backend.py`'s plate and guitar
//! sections (models #5, #5b, #5g).
//!
//! The guitar's tiers, in the reference's order of weight: the PARITY FLIP (the claim, and the
//! only tier that can fail), the display mask staying connected where the solver's is, the
//! rectangle path untouched, the reported diagnostics self-consistent — and energy, which is a
//! regression tier only, because it is geometry-blind.

mod common;

use common::{decode_b64, decode_f32, f, ok, sim};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::membrane::DISPLAY_MAX;
use physsynth_viewer::plate::{
    GUITAR_CENTRELINE_TOL, GUITAR_SWEEP_N_MAX, GUITAR_SWEEP_POINTS, GUITAR_WAIST_MAX, PLATE_N_MAX,
};
use serde_json::{json, Value};
use std::sync::OnceLock;

fn plate(overrides: Value) -> Value {
    let mut p = json!({
        "model": "plate", "domain": "supported",
        "kappa": 20.0, "rho": 0.005, "Lx": 1.0, "Ly": 1.0,
        "N": 40, "mu": 1.0, "sigma": 0.0, "nu": 0.3,
        "pluck_x": 0.4, "pluck_y": 0.55, "pluck_width": 0.3, "amplitude": 1e-3,
        "pickup_x": 0.62, "pickup_y": 0.58,
        "audio_duration": 0.2, "animation_window": 0.02, "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

fn guitar(overrides: Value) -> Value {
    let mut p = plate(json!({
        "domain": "guitar", "Lx": 0.37, "Ly": 0.48, "N": 24, "mu": 8.0,
        "waist": 0.42, "asym": 0.30, "pluck_x": 0.25, "pluck_y": 0.35,
        "audio_duration": 0.05, "animation_window": 0.01,
    }));
    common::merge(&mut p, overrides);
    p
}

/// The shipped guitar, rendered once — the reference's module-scoped fixture.
fn guitar_payload() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&guitar(json!({}))))
}

fn dim(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

/// 4-connected components of a decoded display mask.
fn components(mask: &[u8], ny: usize, nx: usize) -> usize {
    let mut seen = vec![false; mask.len()];
    let mut n = 0;
    for s in 0..mask.len() {
        if mask[s] == 0 || seen[s] {
            continue;
        }
        n += 1;
        seen[s] = true;
        let mut stack = vec![s];
        while let Some(c) = stack.pop() {
            let (y, x) = (c / nx, c % nx);
            let mut nb = Vec::new();
            if y + 1 < ny {
                nb.push(c + nx);
            }
            if y > 0 {
                nb.push(c - nx);
            }
            if x + 1 < nx {
                nb.push(c + 1);
            }
            if x > 0 {
                nb.push(c - 1);
            }
            for q in nb {
                if mask[q] != 0 && !seen[q] {
                    seen[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    n
}

// == the rectangle =================================================================================

#[test]
fn both_rectangle_boundaries_conserve_and_book_keep_in_2d() {
    for boundary in ["supported", "free"] {
        let d = ok(&plate(json!({"domain": boundary})));
        assert_eq!(d["model"], "plate");
        assert_eq!(d["boundary"], boundary);
        let e = &d["energy"];
        assert_eq!(e["sigma_is_zero"], true);
        assert!(e.get("lossy").is_none());
        assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL, "{boundary}");
        assert_eq!(e["lossless"]["pass"], true);
        let fr = &d["frames"];
        let (nf, nx, ny) = (dim(&fr["n_frames"]), dim(&fr["nx"]), dim(&fr["ny"]));
        assert!(nx <= DISPLAY_MAX && ny <= DISPLAY_MAX);
        assert_eq!(fr["dims"], 2);
        assert_eq!(decode_f32(fr["b64"].as_str().unwrap()).len(), nf * nx * ny);
        assert_eq!(
            decode_b64(d["mask"]["b64"].as_str().unwrap()).len(),
            nx * ny
        );
        assert_eq!(nf, d["frame_times"].as_array().unwrap().len());
        assert!(nf >= 2);
        assert_eq!(d["grid"]["domain"], "rectangle");
        // the outline path adds keys and changes no number
        assert_eq!(d["outline"], "rectangle");
        assert!(d["meta"].get("claim").is_none());
        let info = &d["meta"]["outline_info"];
        assert_eq!(info["n_pruned"], 0);
        assert_eq!(f(&info["prune_depth_h"]), 0.0);
        assert!(f(&info["area_deficit"]).abs() < 1e-9);
        assert!(info["n_live"].as_u64() <= info["n_box"].as_u64());
    }
}

/// Simply supported: the FFT rings on the discrete line and the Navier tier is tight.
#[test]
fn the_supported_spectrum_is_the_tight_tier() {
    let sp = ok(&plate(
        json!({"domain": "supported", "N": 40, "audio_duration": 0.3}),
    ))["meta"]["spectrum"]
        .clone();
    assert_eq!(sp["kind"], "plate");
    assert!(!sp["modes_discrete"].as_array().unwrap().is_empty());
    assert!(!sp["modes_continuum"].as_array().unwrap().is_empty());
    assert!(f(&sp["cents_fundamental"]).abs() < 5.0);
    assert!(f(&sp["cents_geometry"]).abs() < 15.0);
}

/// The free plate's reference is the Leissa SQUARE anchor: present for a square, absent off it.
#[test]
fn the_free_spectrum_has_the_leissa_anchor_only_when_square() {
    let sq = ok(&plate(json!({"domain": "free", "Lx": 1.0, "Ly": 1.0})));
    assert!(!sq["meta"]["spectrum"]["modes_continuum"]
        .as_array()
        .unwrap()
        .is_empty());
    let rect = ok(&plate(json!({"domain": "free", "Lx": 1.4, "Ly": 0.7})));
    assert_eq!(rect["meta"]["spectrum"]["modes_continuum"], json!([]));
}

#[test]
fn a_lossy_plate_reports_passivity() {
    let e = ok(&plate(json!({"sigma": 6.0, "audio_duration": 0.25})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert!(e.get("lossless").is_none());
    assert_eq!(e["lossy"]["monotone"], true);
    assert!((f(&e["lossy"]["oracle_2sigma"]) - 12.0).abs() < 1e-12);
}

#[test]
fn bad_params_give_an_error_payload() {
    for bad in [
        json!({"mu": 0.0}),
        json!({"mu": -1.0}),
        json!({"N": 1}),
        json!({"N": PLATE_N_MAX + 1}),
        json!({"domain": "clamped"}),
        json!({"kappa": 0.0}),
        json!({"Lx": 0.0}),
        json!({"audio_duration": 0.0}),
    ] {
        let d = sim(&plate(bad.clone()));
        assert!(
            !d["error"]["message"].as_str().unwrap_or("").is_empty(),
            "{bad}"
        );
    }
}

/// `fs = kappa / (mu h²)` explodes at LOW mu, and the refusal points there.
#[test]
fn low_mu_is_rejected_by_the_work_budget() {
    let d = sim(&plate(json!({"mu": 0.25, "N": 80, "audio_duration": 2.0})));
    let msg = d["error"]["message"].as_str().unwrap().to_lowercase();
    assert!(msg.contains("node-steps") && msg.contains("mu"), "{msg}");
}

// == the guitar ====================================================================================

/// THE CLAIM: deepening the waist swaps the fundamental from an even bender to an odd twist, and
/// the detector is one parity that reads ±1 exactly and flips once.
#[test]
fn the_waist_swaps_the_fundamental() {
    let claim = &guitar_payload()["meta"]["claim"];
    assert_eq!(claim["kind"], "waist_crossing");
    let rows = claim["rows"].as_array().unwrap();
    assert!(rows.len() >= GUITAR_SWEEP_POINTS - 2);
    for r in rows {
        assert!((f(&r["parity"]).abs() - 1.0).abs() < 1e-6, "{r}");
    }
    assert_eq!(claim["n_flips"], 1);
    let (lo, hi) = (f(&claim["crossing"][0]), f(&claim["crossing"][1]));
    assert!(0.0 < lo && lo < hi && hi < GUITAR_WAIST_MAX);
    for r in rows {
        let w = f(&r["waist"]);
        if w <= lo {
            assert!(f(&r["parity"]) > 0.0, "{r}");
        }
        if w >= hi {
            assert!(f(&r["parity"]) < 0.0, "{r}");
        }
    }
    assert_eq!(claim["shipped_side"], "twist");
}

/// The crossing is an INTERVAL, and the slider's dead bands are why.
#[test]
fn the_crossing_is_a_bracket_not_a_number() {
    let claim = &guitar_payload()["meta"]["claim"];
    assert!(f(&claim["crossing"][1]) > f(&claim["crossing"][0]));
    let q = &claim["quantisation"];
    let (blo, bhi) = (f(&q["dead_band"][0]), f(&q["dead_band"][1]));
    let w = f(&claim["shipped_waist"]);
    assert!(blo <= w && w <= bhi && bhi > blo, "{q}");
    let distinct = f(&q["distinct"]);
    assert!(1.0 < distinct && distinct < GUITAR_WAIST_MAX / f(&q["sampling"]));
}

/// A centre-line strike cannot excite the odd family at all — so the overlap ships as a number.
#[test]
fn the_strike_overlap_is_reported_because_it_can_hide_the_claim() {
    let claim = &guitar_payload()["meta"]["claim"];
    assert_eq!(claim["centreline_warning"], false);
    let rows = claim["rows"].as_array().unwrap();
    assert!(rows.iter().all(|r| (0.0..=1.0).contains(&f(&r["a1"]))));
    let hi = f(&claim["crossing"][1]);
    let past_max = |rows: &[Value]| {
        rows.iter()
            .filter(|r| f(&r["waist"]) >= hi)
            .map(|r| f(&r["a1"]))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    assert!(
        past_max(rows) > 0.5,
        "the default strike really does hit the twist"
    );

    let on_centre = ok(&guitar(json!({"pluck_x": 0.5})));
    let flagged = &on_centre["meta"]["claim"];
    assert_eq!(flagged["centreline_warning"], true);
    assert!((f(&flagged["pluck_x"]) - 0.5).abs() < GUITAR_CENTRELINE_TOL);
    let frows = flagged["rows"].as_array().unwrap();
    let fhi = f(&flagged["crossing"][1]);
    let fpast = frows
        .iter()
        .filter(|r| f(&r["waist"]) >= fhi)
        .map(|r| f(&r["a1"]))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        fpast < 1e-4,
        "on the centre line the twist is genuinely not struck: {fpast}"
    );
}

#[test]
fn the_display_mask_is_connected_and_pooled() {
    let d = guitar_payload();
    let (nx, ny) = (dim(&d["frames"]["nx"]), dim(&d["frames"]["ny"]));
    let mask = decode_b64(d["mask"]["b64"].as_str().unwrap());
    assert_eq!(mask.len(), nx * ny);
    let live = mask.iter().filter(|&&m| m != 0).count();
    assert!(0 < live && live < mask.len());
    assert_eq!(components(&mask, ny, nx), 1);
    assert_eq!(d["grid"]["domain"], "guitar");
    assert_eq!(d["outline"], "guitar");
    assert_eq!(d["boundary"], "free");
}

/// End to end on a configuration the sliders can reach, where point-sampling draws two lobes.
#[test]
fn a_reachable_narrow_deep_waisted_guitar_still_draws_one_plate() {
    let d = ok(&guitar(
        json!({"Lx": 0.15, "Ly": 0.70, "N": 33, "waist": 0.88, "asym": 0.0,
                              "mu": 32.0, "audio_duration": 0.05, "animation_window": 0.005}),
    ));
    let (nx, ny) = (dim(&d["frames"]["nx"]), dim(&d["frames"]["ny"]));
    let mask = decode_b64(d["mask"]["b64"].as_str().unwrap());
    assert_eq!(components(&mask, ny, nx), 1);
}

#[test]
fn the_guitar_reports_its_own_outline_self_consistently() {
    let info = &guitar_payload()["meta"]["outline_info"];
    assert_eq!(info["domain"], "guitar");
    let (area, outline) = (f(&info["area"]), f(&info["outline_area"]));
    assert!(area < outline, "the staircase always undershoots");
    assert!((f(&info["area_deficit"]) - (area / outline - 1.0)).abs() <= 1e-5);
    let deficit = f(&info["area_deficit"]);
    assert!(-0.30 < deficit && deficit < -0.01);
    let fill = f(&info["n_live"]) / f(&info["n_box"]);
    assert!(0.3 < fill && fill < 0.7);
    assert!(info["n_pruned"].as_u64().unwrap() >= 1);
    let depth = f(&info["prune_depth_h"]);
    assert!(0.0 < depth && depth <= 1.0001);
}

/// Regression tier ONLY — energy is geometry-blind, and it says so in its name.
#[test]
fn the_guitar_energy_conserves_but_this_tier_is_not_evidence() {
    let e = &guitar_payload()["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

#[test]
fn the_sweep_grid_is_capped_independently_of_the_audio_grid() {
    let fine = ok(&guitar(json!({"N": 56, "mu": 16.0})));
    assert_eq!(fine["meta"]["claim"]["sweep_N"], GUITAR_SWEEP_N_MAX);
    let coarse = ok(&guitar(json!({"N": 20, "mu": 8.0})));
    assert_eq!(coarse["meta"]["claim"]["sweep_N"], 20);
}

/// A rectangle law must not be drawn on a plate that is not a rectangle — even a guitar whose
/// bounding box is square — while the free square keeps the anchor it is entitled to.
#[test]
fn a_curved_outline_gets_no_continuum_reference_whatever_its_box() {
    let square_box = ok(&guitar(json!({"Lx": 0.60, "Ly": 0.60, "N": 24, "mu": 32.0,
                                       "audio_duration": 0.05})));
    assert_eq!(square_box["meta"]["spectrum"]["modes_continuum"], json!([]));
    assert!(square_box["meta"]["spectrum"]["cents_geometry"].is_null());
    let rect = ok(&plate(
        json!({"domain": "free", "Lx": 1.0, "Ly": 1.0, "N": 24, "mu": 4.0,
                                "audio_duration": 0.2}),
    ));
    assert!(!rect["meta"]["spectrum"]["modes_continuum"]
        .as_array()
        .unwrap()
        .is_empty());
}

/// A 0.01 s render is legal, and its fundamental readout must be present and POSITIVE.
#[test]
fn a_short_render_reports_a_positive_fundamental() {
    for (lx, ly, n) in [(0.15, 0.80, 16), (0.30, 0.80, 24)] {
        let d = ok(&guitar(
            json!({"Lx": lx, "Ly": ly, "N": n, "mu": 32.0, "waist": 0.42,
                                  "asym": 0.30, "audio_duration": 0.01,
                                  "animation_window": 0.002}),
        ));
        let det = &d["meta"]["spectrum"]["f1_detected"];
        assert!(!det.is_null(), "suppressed, not measured ({lx}, {ly}, {n})");
        assert!(f(det) > 0.0);
    }
}

/// The claim panel and the spectrum panel quote the SAME hertz — the discrete theta-scheme's —
/// at both ends of the mu slider.
#[test]
fn the_claim_and_spectrum_panels_quote_the_same_hertz() {
    for mu in [2.0, 32.0] {
        let d = ok(&guitar(json!({"N": 24, "mu": mu, "audio_duration": 0.05})));
        let you = &d["meta"]["claim"]["you"];
        let f1 = f(&d["meta"]["f1"]);
        assert!((f(&you["f1"]) - f1).abs() <= 1e-6 * f1 + 1e-3, "mu {mu}");
        assert!(f(&you["f2"]) > f(&you["f1"]));
        assert!((f(&you["parity"]).abs() - 1.0).abs() < 1e-6);
    }
}

/// A long body pushes the crossing past the slider, and the panel says "none", not "at".
#[test]
fn a_long_body_pushes_the_crossing_past_the_slider() {
    let long_body = ok(&guitar(json!({"Lx": 0.20, "Ly": 1.00, "N": 24, "mu": 16.0,
                                      "audio_duration": 0.05})));
    let claim = &long_body["meta"]["claim"];
    assert_eq!(claim["n_flips"], 0);
    assert!(claim["crossing"].is_null());
    assert_eq!(claim["shipped_side"], "none");
    let rows = claim["rows"].as_array().unwrap();
    assert!(rows.len() >= GUITAR_SWEEP_POINTS - 2);
    assert!(rows
        .iter()
        .all(|r| (f(&r["parity"]).abs() - 1.0).abs() < 1e-6));
    let squat = ok(&guitar(json!({"Lx": 0.42, "Ly": 0.46, "N": 24, "mu": 16.0,
                                  "audio_duration": 0.05})));
    assert_eq!(squat["meta"]["claim"]["n_flips"], 1);
    assert!(f(&squat["meta"]["claim"]["crossing"][1]) < 0.30);
}

#[test]
fn guitar_bad_params_give_an_error_payload_and_guitar_means_free() {
    for bad in [
        json!({"waist": GUITAR_WAIST_MAX + 0.01}),
        json!({"waist": -0.1}),
        json!({"asym": 1.5}),
        json!({"domain": "guitar-supported"}),
    ] {
        let d = sim(&guitar(bad.clone()));
        assert!(
            !d["error"]["message"].as_str().unwrap_or("").is_empty(),
            "{bad}"
        );
    }
    let d = ok(&guitar(json!({"audio_duration": 0.02})));
    assert_eq!(d["boundary"], "free");
    assert_eq!(d["outline"], "guitar");
}

// == the pooled display decimation, pinned on synthetic masks ======================================

/// Two bouts joined by a one-node neck on an ODD column: point-sampling at stride >= 2 severs it,
/// pooling keeps it. Built big enough to FORCE a real stride, and asserts it got one — a toy that
/// took stride 1 would exercise no pooling and pass without testing anything.
#[test]
fn pooling_keeps_a_one_node_isthmus_that_point_sampling_severs() {
    use physsynth_core::ops2d::Mask;
    use physsynth_viewer::membrane::decimate_field_mask;
    use physsynth_viewer::plate::{display_components, pool_field_mask};
    let (ny, nx) = (2 * DISPLAY_MAX, DISPLAY_MAX - 8);
    let (upper, lower) = (ny / 2 - 5, ny / 2 + 5);
    let mut live = vec![false; ny * nx];
    for j in 0..ny {
        for i in 0..nx {
            live[j * nx + i] = j < upper || j >= lower || i == 1;
        }
    }
    let flags: Vec<u8> = live.iter().map(|&b| u8::from(b)).collect();
    assert_eq!(display_components(&flags, ny, nx), 1);
    let frames = vec![vec![1.0; ny * nx]];
    let (_, sampled, sy, sx) = decimate_field_mask(&frames, &live, ny, nx);
    assert!(ny / sy >= 2, "the toy must force a real stride");
    assert_eq!(
        display_components(&sampled, sy, sx),
        2,
        "point-sampling severs the neck"
    );
    let (_, pooled, py, px) = pool_field_mask(&frames, &Mask::new(ny, nx, live));
    assert_eq!(display_components(&pooled, py, px), 1, "pooling keeps it");
}

/// On an all-live mask every block's representative is live, so pooling IS point-sampling, bit
/// for bit — across the strides the display budget produces.
#[test]
fn pooling_reduces_to_point_sampling_on_an_all_live_mask() {
    use physsynth_core::ops2d::Mask;
    use physsynth_viewer::membrane::decimate_field_mask;
    use physsynth_viewer::plate::pool_field_mask;
    let mut strides = std::collections::BTreeSet::new();
    let mut seed = 0x2026_0826u64;
    for (ny, nx) in [(40, 40), (80, 65), (129, 100), (200, 51), (400, 90)] {
        let frames: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                (0..ny * nx)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                    })
                    .collect()
            })
            .collect();
        let live = vec![true; ny * nx];
        let (wf, wm, wy, wx) = decimate_field_mask(&frames, &live, ny, nx);
        let (gf, gm, gy, gx) = pool_field_mask(&frames, &Mask::new(ny, nx, live));
        strides.insert(ny.max(nx).div_ceil(DISPLAY_MAX));
        assert_eq!((gy, gx), (wy, wx));
        assert_eq!(gm, wm);
        assert_eq!(gf, wf, "bit-identical, not close");
    }
    assert!(strides.len() >= 3, "only strides {strides:?} exercised");
}
