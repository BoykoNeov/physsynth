//! The wind pages' musical controls through the payload builder — `docs/dev/wind-mapping-plan.md`.
//!
//! The bore's mapping is held by core's `tests/bore_tuning.rs`, which reads the one-step map and
//! the render. What is held here is the wiring, judged by what the page PLAYS: its measured
//! partials against the odd series of the target, its ring time off the energy it has shed, and
//! the musical page as the physics page fed the mapped tube. The reed's search is held by core's
//! `tests/reed_tuning.rs`; here, its page plays the pitch to another ear, its bell is the one the
//! FOUND tube needs, and each refusal says what to change.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_core::wind_tuning::tune_bore;
use serde_json::{json, Value};
use std::f64::consts::{LN_10, PI};

/// The page's speed of sound, written out, not imported.
const C0: f64 = 343.0;

/// A bore request off the page's defaults, with decoys a musical build must not read: at
/// `L = 1.5` the tube sounds 57 Hz, and `10^-1` is a bell forty times lossier than the mapping's.
fn bore(overrides: Value) -> Value {
    let mut p = json!({
        "model": "bore", "controls": "musical", "domain": "radiating",
        "N": 100, "L": 1.5, "bell_ratio_exp": -1.0,
        "f0": 200.0, "t60_low": 2.0,
        "pickup_position": 0.23, "audio_duration": 0.5, "animation_window": 0.02,
        "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

/// Numbers within `rel` of each other, everything else equal, recursively; `b64` strings (float32
/// renderings of numbers that may differ in their last bits) are skipped.
fn assert_close(a: &Value, b: &Value, rel: f64, path: &str) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let (kx, ky): (Vec<_>, Vec<_>) = (x.keys().collect(), y.keys().collect());
            assert_eq!(kx, ky, "{path}: keys differ");
            for (k, v) in x {
                if k != "b64" {
                    assert_close(v, &y[k], rel, &format!("{path}.{k}"));
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: lengths differ");
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                assert_close(u, v, rel, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(_), Value::Number(_)) => {
            let (u, v) = (f(a), f(b));
            let scale = u.abs().max(v.abs()).max(1e-300);
            // An absolute floor for roundoff-sized read-outs (the energy drift, ~1e-15).
            assert!((u - v).abs() <= rel * scale + 1e-12, "{path}: {u} vs {v}");
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

fn refusal(p: &Value) -> String {
    let payload = sim(p);
    assert_eq!(payload["error"]["kind"], "param", "{payload}");
    payload["error"]["message"].as_str().unwrap().to_owned()
}

// -- the bore ---------------------------------------------------------------------------------

#[test]
fn musical_keys_do_not_touch_the_physical_bore() {
    for domain in ["radiating", "open"] {
        let mut named = bore(json!({"controls": "physical", "domain": domain, "L": 0.7}));
        let a = ok(&named);
        named.as_object_mut().unwrap().remove("controls");
        named.as_object_mut().unwrap().remove("f0");
        named.as_object_mut().unwrap().remove("t60_low");
        assert_eq!(a, ok(&named), "{domain}");
        assert!(a.get("tuning").is_none());
    }
}

#[test]
fn the_musical_bore_is_the_physics_bore_on_the_mapped_tube() {
    for (domain, f0, t60) in [
        ("radiating", 200.0, 2.0),
        ("radiating", 61.7, 0.3),
        ("open", 440.0, 2.0),
    ] {
        let musical = ok(&bore(json!({"domain": domain, "f0": f0, "t60_low": t60})));
        let tu = tune_bore(f0, (domain == "radiating").then_some(t60), C0).unwrap();
        let mut twin = bore(json!({"controls": "physical", "domain": domain, "L": tu.l}));
        if let Some(r) = tu.r_ratio {
            twin["bell_ratio_exp"] = json!(r.log10());
        }
        let mut stripped = musical.clone();
        let t = stripped
            .as_object_mut()
            .unwrap()
            .remove("tuning")
            .expect("a tuning block");
        // 10^log10(r) is the ratio to a few ulps, so the twin is close, not bit-equal.
        assert_close(&stripped, &ok(&twin), 1e-9, domain);
        assert_eq!(f(&t["L"]), tu.l);
        assert_eq!(t["r_ratio"], tu.r_ratio.map_or(Value::Null, |r| json!(r)));
    }
}

#[test]
fn the_musical_bore_plays_the_odd_series_of_its_pitch() {
    // A build that read the decoy L would sound 57 Hz: three of the partials below would be a
    // semitone and more away.
    for (f0, n) in [(200.0, 100), (97.3, 64)] {
        let payload = ok(&bore(json!({"f0": f0, "N": n, "audio_duration": 1.0})));
        let measured = payload["meta"]["spectrum"]["partials"]["measured"]
            .as_array()
            .unwrap();
        assert!(measured.len() >= 3);
        for (i, v) in measured.iter().enumerate() {
            let want = (2 * i + 1) as f64 * f0;
            let c = 1200.0 * (f(v) / want).log2();
            assert!(
                c.abs() < 1.0,
                "f0 {f0}: partial {} at {} Hz, {c} cents",
                i + 1,
                f(v)
            );
        }
    }
}

#[test]
fn the_musical_bore_rings_its_ring_time() {
    // At lambda = 1 the whole state returns every round trip scaled by |rho|, so after a whole
    // number of round trips (f0 D is whole: 100, 20 and 200 periods) the air keeps exactly
    // 10^(-6 D / T60) of the energy it started with: the rest has left through the bell.
    for (t60, d) in [(2.0, 0.5), (0.4, 0.1), (9.0, 1.0)] {
        let payload = ok(&bore(json!({"t60_low": t60, "audio_duration": d})));
        let left = 1.0 - f(&payload["meta"]["radiated_frac"]);
        let want = 10f64.powf(-6.0 * d / t60);
        assert!(
            (left / want - 1.0).abs() < 1e-4,
            "T60 {t60}: {left} of the energy left after {d} s, want {want}"
        );
    }
}

#[test]
fn the_open_end_sets_the_pitch_alone() {
    let payload = ok(&bore(
        json!({"domain": "open", "f0": 300.0, "t60_low": 0.3}),
    ));
    let t = &payload["tuning"];
    assert_eq!(t["t60"], Value::Null);
    assert_eq!(t["r_ratio"], Value::Null);
    assert!(t["message"].as_str().unwrap().contains("rings forever"));
    assert_eq!(f(&payload["meta"]["radiated_frac"]), 0.0);
    // The ring-time slider is not read: a value no bell could hold changes nothing.
    let other = ok(&bore(
        json!({"domain": "open", "f0": 300.0, "t60_low": 1e6}),
    ));
    assert_eq!(payload, other);
}

#[test]
fn a_pitch_outside_the_tube_is_refused_with_the_range() {
    for f0 in [40.0, 900.0] {
        let msg = refusal(&bore(json!({"f0": f0})));
        assert!(msg.contains("f0 must be in [42.88, 857.5] Hz"), "{msg}");
    }
}

#[test]
fn a_ring_time_past_the_bell_names_the_longest_at_that_pitch() {
    // At 800 Hz the tube is 0.107 m and the bell's floor R/Z0 = 1e-4 rings 21.59 s.
    let msg = refusal(&bore(json!({"f0": 800.0, "t60_low": 30.0, "N": 32})));
    assert!(msg.contains("the longest is 21.59 s"), "{msg}");
    // Just inside it renders.
    ok(&bore(json!({"f0": 800.0, "t60_low": 21.0, "N": 32})));
}

#[test]
fn the_budget_refusal_names_the_highest_pitch_that_fits() {
    // N = 256, 1.5 s: fs = 4 N f0, so (300000 - 256) / (4 * 256 * 1.5) = 195.15 Hz fits (printed
    // 195.1).
    let msg = refusal(&bore(json!({"f0": 400.0, "N": 256, "audio_duration": 1.5})));
    assert!(
        msg.contains("the highest pitch that fits is 195.1 Hz"),
        "{msg}"
    );
    ok(&bore(json!({"f0": 195.0, "N": 256, "audio_duration": 1.5})));
}

// -- the reed ---------------------------------------------------------------------------------

/// A reed request with every reed parameter off the page's defaults, and decoys a musical build
/// must not read: `L = 1.5` and a bell `10^-1`.
fn reed(overrides: Value) -> Value {
    let mut p = json!({
        "model": "reed", "controls": "musical", "domain": "radiating",
        "N": 100, "L": 1.5, "bell_ratio_exp": -1.0,
        "f0": 196.0, "t60_low": 3.0,
        "gamma": 0.6, "f_reed": 2200.0, "q_reed": 5.0,
        "audio_duration": 0.6, "animation_window": 0.02, "playback_speed": 0.02,
    });
    common::merge(&mut p, overrides);
    p
}

/// The Hann-windowed Fourier peak within 3 % of `f0`: a grid, then a golden section. Not the
/// page's reader, which is an autocorrelation.
fn fourier_peak(x0: &[f64], fs: f64, f0: f64) -> f64 {
    let n = x0.len();
    let m = x0.iter().sum::<f64>() / n as f64;
    let x: Vec<f64> = x0
        .iter()
        .enumerate()
        .map(|(i, v)| (v - m) * (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect();
    let power = |fr: f64| {
        let w = 2.0 * PI * fr / fs;
        let (mut re, mut im) = (0.0, 0.0);
        for (i, v) in x.iter().enumerate() {
            re += v * (w * i as f64).cos();
            im -= v * (w * i as f64).sin();
        }
        re * re + im * im
    };
    let (lo, hi) = (0.97 * f0, 1.03 * f0);
    let step = (hi - lo) / 60.0;
    let best =
        (0..=60)
            .map(|j| lo + step * j as f64)
            .fold(lo, |b, fr| if power(fr) > power(b) { fr } else { b });
    let (mut a, mut b) = (best - step, best + step);
    let g = 0.5 * (5f64.sqrt() - 1.0);
    for _ in 0..50 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if power(c) > power(d) {
            b = d;
        } else {
            a = c;
        }
    }
    0.5 * (a + b)
}

#[test]
fn musical_keys_do_not_touch_the_physical_reed() {
    for domain in ["radiating", "open"] {
        let mut named = reed(json!({"controls": "physical", "domain": domain, "L": 0.6}));
        let a = ok(&named);
        for k in ["controls", "f0", "t60_low"] {
            named.as_object_mut().unwrap().remove(k);
        }
        assert_eq!(a, ok(&named), "{domain}");
        assert!(a.get("tuning").is_none());
    }
}

/// What the page plays, heard by an ear that is not the page's: the rendered audio's last 40 %.
/// A build that read the decoy L would play near 57 Hz.
#[test]
fn the_musical_reed_plays_its_pitch_to_another_ear() {
    for (domain, f0) in [("radiating", 196.0), ("open", 130.81)] {
        let payload = ok(&reed(json!({"domain": domain, "f0": f0})));
        let t = &payload["tuning"];
        assert!((1200.0 * (f(&t["played"]) / f0).log2()).abs() < 0.1, "{t}");
        // The search tuned the rig the page renders: its last run and the render read alike, to
        // the bit.
        assert_eq!(t["render_played"], t["played"], "{t}");
        assert!(f(&t["pull_cents"]) < 0.0, "the reed pulls flat: {t}");
        let audio: Vec<f64> = decode_f32(payload["audio"]["b64"].as_str().unwrap())
            .into_iter()
            .map(f64::from)
            .collect();
        let tail = &audio[(0.6 * audio.len() as f64) as usize..];
        let heard = fourier_peak(tail, f(&payload["audio"]["fs"]), f0);
        let c = 1200.0 * (heard / f0).log2();
        assert!(
            c.abs() < 1.0,
            "{domain} {f0}: the audio sounds {heard} Hz ({c:+.3} cents)"
        );
        assert_eq!(payload["meta"]["speaks"], true);
        // The odd-harmonic panel sits on the note played, not on the tube's own c0/4L.
        let marker = f(&payload["meta"]["spectrum"]["markers"][0]);
        assert!(
            (marker - f(&t["played"])).abs() < 1e-3,
            "{marker} vs {}",
            t["played"]
        );
    }
}

/// The bell belongs to the tube the search FOUND (plan §6): `R/Z0 = tanh(3 ln10 L / (c0 T60))`
/// at the payload's own L — not at `c0 / 4 f0`, which the reed's pull makes ~4 % too long here.
/// The twin below cannot see this: it reads the same number the page built.
#[test]
fn the_reed_bell_is_built_for_the_tube_the_search_found() {
    let payload = ok(&reed(json!({})));
    let t = &payload["tuning"];
    let l = f(&t["L"]);
    let want = (3.0 * LN_10 * l / (C0 * 3.0)).tanh();
    assert!((f(&t["r_ratio"]) / want - 1.0).abs() < 1e-12, "{t}");
    assert!((f(&payload["meta"]["r_ratio"]) / want - 1.0).abs() < 1e-12);
    // The control: the unpulled tube's bell is another number.
    let unpulled = (3.0 * LN_10 * (C0 / (4.0 * 196.0)) / (C0 * 3.0)).tanh();
    assert!((unpulled / want - 1.0).abs() > 1e-3);
}

#[test]
fn the_musical_reed_is_the_physics_reed_on_the_found_tube() {
    for domain in ["radiating", "open"] {
        let musical = ok(&reed(json!({"domain": domain})));
        let t = musical["tuning"].clone();
        let mut twin = reed(json!({"controls": "physical", "domain": domain, "L": t["L"]}));
        if domain == "radiating" {
            twin["bell_ratio_exp"] = json!(f(&t["r_ratio"]).log10());
        }
        let mut stripped = musical.clone();
        stripped.as_object_mut().unwrap().remove("tuning");
        let mut physics = ok(&twin);
        // The harmonic and beating panels are anchored on the played note in musical mode and on
        // c0/4L in physics, so they are the one place the twins differ by design.
        for p in [&mut stripped, &mut physics] {
            let meta = p["meta"].as_object_mut().unwrap();
            meta.remove("spectrum");
            meta.remove("beating");
        }
        assert_close(&stripped, &physics, 1e-9, domain);
    }
}

#[test]
fn the_open_reed_reads_no_ring_time() {
    let a = ok(&reed(json!({"domain": "open"})));
    let b = ok(&reed(json!({"domain": "open", "t60_low": 1e6})));
    assert_eq!(a, b);
    assert_eq!(a["tuning"]["t60"], Value::Null);
}

#[test]
fn a_reed_refusal_says_what_to_change() {
    let cases = [
        (json!({"gamma": 0.2}), "blow harder"),
        (json!({"f0": 800.0}), "out of this reed's reach"),
        (
            json!({"f0": 45.0, "audio_duration": 0.2, "gamma": 0.51, "f_reed": 2500.0}),
            "too short to read a note this low",
        ),
        (
            json!({"f0": 440.0, "gamma": 0.36, "f_reed": 3500.0, "q_reed": 8.0, "N": 64,
                   "audio_duration": 0.2, "t60_low": 40.0}),
            "does not play the tube's note",
        ),
        (
            json!({"f0": 400.0, "N": 256, "audio_duration": 1.0}),
            // 300000 / (4 * 256 * 1.0) = 292.97
            "must stay below 293.0 Hz",
        ),
        (
            json!({"f0": 600.0, "t60_low": 60.0, "N": 64, "audio_duration": 0.3}),
            "longer than the bell can hold on the",
        ),
        // Under the early ceiling, the found tube is still too short for the budget.
        (
            json!({"f0": 290.0, "N": 256, "audio_duration": 1.0}),
            "this pitch needs runs",
        ),
    ];
    for (over, want) in cases {
        let msg = refusal(&reed(over.clone()));
        assert!(msg.contains(want), "{over}: {msg}");
    }
}

/// A refusal that names a limit names one the page can play: the highest note out of reach
/// names, and the audio length a low note asks for.
#[test]
fn a_named_reed_limit_is_one_the_page_plays() {
    let num_after = |msg: &str, key: &str| -> f64 {
        let rest = &msg[msg.find(key).unwrap_or_else(|| panic!("{msg}")) + key.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap();
        rest[..end].parse().unwrap()
    };
    let msg = refusal(&reed(json!({"f0": 800.0})));
    let highest = num_after(&msg, "the highest note it plays is ");
    ok(&reed(json!({"f0": highest - 0.5})));
    let low = json!({"f0": 45.0, "gamma": 0.51, "f_reed": 2500.0});
    let mut short = low.clone();
    short["audio_duration"] = json!(0.2);
    let msg = refusal(&reed(short));
    let enough = num_after(&msg, "at least ");
    let mut long = low;
    long["audio_duration"] = json!(enough);
    ok(&reed(long));
}

/// The grid's buzz near fs/2 (plan §7): warned of when it is in hearing and within 30 dB of the
/// note. Cases far from the line: +9.6 dB (a heavy bell on a 45 Hz note at N = 64), -36.9 dB in
/// hearing (11.6 kHz), and above hearing at any level.
#[test]
fn an_audible_grid_buzz_is_warned_of() {
    let defaults = json!({"gamma": 0.51, "f_reed": 2500.0, "q_reed": 4.0, "audio_duration": 0.5});
    let case = |over: Value| {
        let mut p = reed(defaults.clone());
        common::merge(&mut p, over);
        ok(&p)["tuning"].clone()
    };
    let loud = case(json!({"f0": 45.0, "N": 64, "t60_low": 0.3}));
    assert!(
        f(&loud["buzz_db"]) > -20.0 && f(&loud["buzz_hz"]) < 20_000.0,
        "{loud}"
    );
    assert_eq!(loud["warn"], true);
    assert!(
        loud["message"].as_str().unwrap().contains("raise N"),
        "{loud}"
    );
    let quiet = case(json!({"f0": 45.0, "N": 128, "t60_low": 2.0}));
    assert!(
        f(&quiet["buzz_db"]) < -35.0 && f(&quiet["buzz_hz"]) < 20_000.0,
        "{quiet}"
    );
    assert_eq!(quiet["warn"], false, "{quiet}");
    // Loud (-15 dB) but at 31 kHz: the 48 kHz audio's windowed-sinc filter cuts it.
    let above = case(json!({"f0": 60.0, "N": 256, "t60_low": 0.3}));
    assert!(
        f(&above["buzz_db"]) > -25.0 && f(&above["buzz_hz"]) > 25_000.0,
        "{above}"
    );
    assert_eq!(above["warn"], false, "{above}");
}

/// A grid too coarse for the reed's own vibration names the N and the f_reed that fix it, on
/// both control sets — and each named value plays.
#[test]
fn a_grid_too_coarse_for_the_reed_names_the_n_that_plays() {
    let physics = json!({"model": "reed", "controls": "physical", "domain": "radiating",
        "N": 32, "L": 1.9, "gamma": 0.51, "f_reed": 2500.0, "q_reed": 4.0,
        "bell_ratio_exp": -2.6, "audio_duration": 0.5, "animation_window": 0.02,
        "playback_speed": 0.02});
    let msg = refusal(&physics);
    assert!(
        msg.contains("raise N to at least 44") && msg.contains("below f_reed = 1838 Hz"),
        "{msg}"
    );
    for over in [json!({"N": 44}), json!({"f_reed": 1838.0})] {
        let mut p = physics.clone();
        common::merge(&mut p, over);
        ok(&p);
    }
    let msg = refusal(&reed(json!({"f0": 45.0, "N": 32})));
    assert!(
        msg.contains("too coarse for the reed's own vibration"),
        "{msg}"
    );
}
