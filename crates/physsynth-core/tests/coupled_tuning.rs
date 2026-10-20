//! `coupled_tuning`: the tension and string loss at which a string on a bridge — the coupled
//! instrument, not the string alone — plays a target pitch and rings a target time
//! (`docs/dev/string-mapping-plan.md` §16).
//!
//! The rigs are the viewer pages': an ideal string fixed at the nut and free at the bridge, on a
//! spring to a body of four guitar-ish modes (or five, for two strings), each about the string's
//! own mass, `N = 100`, `lambda = 0.9`, so the sample rate follows the tension.
//!
//! The round trip is read from the integrator, not from the solver: the run starts on the solved
//! mode (`damped_mode`'s eigenvector) and steps with the bridge's own `step()`. A single damped
//! mode obeys `s_{n+D} = 2 r^D cos(D theta) s_n - r^2D s_{n-D}` exactly, so a least-squares fit of
//! that recurrence to the run gives its pitch and decay, and the fit's residual says whether the
//! start really was one mode.

use std::f64::consts::{LN_10, PI};

use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::{StringBodyBridge, SympatheticStrings};
use physsynth_core::coupled_tuning::{
    damped_mode, string_sigma, tune_coupled, Coupled, CoupledError, CoupledTuning,
};
use physsynth_core::radiation::RadiatedBody;
use physsynth_core::string_ideal::{Boundary, IdealString, Params as StringParams};

const L: f64 = 1.0;
const RHO: f64 = 0.005;
/// The pages' grid.
const N: i64 = 100;
/// The two-string rigs' grid: half the pages', because their map is twice the size and its full
/// spectrum is what the tuning scans (CI's unoptimised lane runs this file). No bar here depends on
/// N; the pages' own bars run at theirs.
const N_TWO: i64 = 50;
const LAM: f64 = 0.9;
const BODY: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
const SYMP: [f64; 5] = [137.0, 213.0, 330.0, 471.0, 620.0];

/// The rig's sample rate at clamped fundamental `f_cl`: `c = 2 L f_cl`, `fs = c N / (L lambda)`.
fn fs_on(f_cl: f64, n: i64) -> f64 {
    2.0 * L * f_cl * n as f64 / (L * LAM)
}

fn fs_at(f_cl: f64) -> f64 {
    fs_on(f_cl, N)
}

fn string_on(f_cl: f64, sigma: f64, n: i64) -> Option<IdealString> {
    let p = StringParams::new(
        L,
        RHO * (2.0 * L * f_cl).powi(2),
        RHO,
        fs_on(f_cl, n),
        n,
        sigma,
        Some((Boundary::Fixed, Boundary::Free)),
    );
    p.ok().map(IdealString::new)
}

fn string_at(f_cl: f64, sigma: f64) -> Option<IdealString> {
    string_on(f_cl, sigma, N)
}

fn modal(freqs: &[f64], fs: f64, sigma: f64) -> ModalBody {
    let m = freqs.len();
    ModalBody::new(
        body::Params::new(
            freqs.to_vec(),
            fs,
            vec![sigma; m],
            vec![0.02; m],
            vec![1.0; m],
            None,
        )
        .expect("valid body"),
    )
}

fn body_rig(
    k_spring: f64,
    sigma_body: f64,
) -> impl Fn(f64, f64) -> Option<StringBodyBridge<ModalBody>> {
    move |f_cl, sigma| {
        let s = string_at(f_cl, sigma)?;
        StringBodyBridge::new(s, modal(&BODY, fs_at(f_cl), sigma_body), k_spring).ok()
    }
}

fn radiating_rig(r: f64) -> impl Fn(f64, f64) -> Option<StringBodyBridge<RadiatedBody>> {
    move |f_cl, sigma| {
        let s = string_at(f_cl, sigma)?;
        let loaded = RadiatedBody::new(modal(&BODY, fs_at(f_cl), 0.0), r).ok()?;
        StringBodyBridge::new(s, loaded, 8000.0).ok()
    }
}

fn two_string_rig(
    k_spring: f64,
    sigma_body: f64,
) -> impl Fn(f64, f64) -> Option<SympatheticStrings> {
    move |f_cl, sigma| {
        let strings = vec![
            string_on(f_cl, sigma, N_TWO)?,
            string_on(f_cl, sigma, N_TWO)?,
        ];
        let b = modal(&SYMP, fs_on(f_cl, N_TWO), sigma_body);
        SympatheticStrings::new(strings, b, vec![k_spring, k_spring]).ok()
    }
}

/// Pitch (Hz), ring time (s) and the fit's worst residual relative to the run's peak, read off
/// `steps` steps of the integrator started on the damped mode nearest `guess` (a root of the
/// one-step map; `f_hint` is roughly its pitch, which only picks the fit's stride).
fn played<R: Coupled>(
    rig: &mut R,
    guess: (f64, f64),
    f_hint: f64,
    probe: usize,
    steps: usize,
) -> (f64, f64, f64) {
    let (_, v) = damped_mode(rig, guess).expect("the mode");
    let start: Vec<f64> = v.iter().map(|c| c.0).collect();
    rig.set_levels(&start);
    let mut s = vec![rig.levels()[probe]];
    for _ in 0..steps {
        rig.advance();
        s.push(rig.levels()[probe]);
    }
    let k = rig.k();
    // The recurrence over a stride of D steps, s_{n+D} = 2 r^D cos(D theta) s_n - r^2D s_{n-D},
    // with D a quarter period. One step turns the mode by only ~0.03 rad, and a one-step fit
    // then resolves the pitch to ~1e-9 at best (its two columns are nearly parallel); a quarter
    // period keeps them orthogonal. The report's pitch only picks D, and any D with
    // D theta in (0, pi) gives the same answer.
    let stride = ((0.25 / (f_hint * k)).round() as usize).max(1);
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for n in stride..s.len() - stride {
        let (x1, x2, y) = (s[n], -s[n - stride], s[n + stride]);
        a11 += x1 * x1;
        a12 += x1 * x2;
        a22 += x2 * x2;
        b1 += x1 * y;
        b2 += x2 * y;
    }
    let det = a11 * a22 - a12 * a12;
    let alpha = (b1 * a22 - b2 * a12) / det;
    let beta = (a11 * b2 - a12 * b1) / det;
    let peak = s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0f64;
    for n in stride..s.len() - stride {
        worst = worst.max((s[n + stride] - alpha * s[n] + beta * s[n - stride]).abs());
    }
    let d = stride as f64;
    let r = beta.sqrt().powf(1.0 / d);
    let f = (alpha / (2.0 * beta.sqrt())).acos() / (2.0 * PI * k * d);
    let t60 = if r < 1.0 {
        3.0 * LN_10 * k / -r.ln()
    } else {
        f64::INFINITY
    };
    (f, t60, worst / peak)
}

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

/// The pitch and the ring time are the targets, read off the integrator, on every rig the pages
/// build: the plain body (two springs, lossy and lossless), the radiating body and two strings.
#[test]
fn the_tuned_instrument_plays_and_rings_its_targets() {
    let check = |name: &str, f0: f64, t60: Option<f64>, t: &CoupledTuning, got: (f64, f64, f64)| {
        let (f, rings, residual) = got;
        eprintln!(
            "{name}: f_cl {:.4}, sigma {:.4}, plays {:+.2e} rel, reports {:+.2e} rel, t60 {:.6} s \
             (fit {:.6}), residual {residual:.1e}, share {:.3}",
            t.f_cl,
            t.sigma,
            f / f0 - 1.0,
            t.note.f / f0 - 1.0,
            t.note.t60,
            rings,
            t.note.share
        );
        assert!(
            residual < 1e-9,
            "{name}: the start was not one mode ({residual:.2e})"
        );
        assert!(
            ((f - f0) / f0).abs() < 1e-9,
            "{name}: plays {f} Hz for {f0} ({:+.3e} c)",
            cents(f, f0)
        );
        assert!(
            ((t.note.f - f0) / f0).abs() < 1e-11,
            "{name}: reports {} Hz",
            t.note.f
        );
        // Decay rates, not ring times: a lossless note's ring time is infinite in the report and
        // merely enormous in a fit.
        let rate = |t60: f64| 3.0 * LN_10 / t60;
        assert!(
            (rate(rings) - rate(t.note.t60)).abs() < 1e-9 * 2.0 * PI * f0,
            "{name}: rings {rings} s, reported {} s",
            t.note.t60
        );
        if let Some(target) = t60 {
            assert!(
                ((t.note.t60 - target) / target).abs() < 1e-8,
                "{name}: rings {} s for {target}",
                t.note.t60
            );
            assert!(
                ((rings - target) / target).abs() < 1e-6,
                "{name}: the run rings {rings} s"
            );
        }
    };
    for (f0, k_spring, sigma_body, t60) in [
        (100.0, 8000.0, 0.0, Some(1.5)),
        (233.08, 3000.0, 20.0, Some(0.2)),
        (82.41, 8000.0, 0.0, None),
    ] {
        let rig = body_rig(k_spring, sigma_body);
        let t = tune_coupled(f0, t60, &rig).expect("tunes");
        let mut r = rig(t.f_cl, t.sigma).unwrap();
        let got = played(&mut r, t.root, t.note.f, 99, 3000);
        check(&format!("body {f0} K {k_spring}"), f0, t60, &t, got);
    }
    let rig = radiating_rig(133.0);
    let t = tune_coupled(146.83, Some(0.5), &rig).expect("tunes");
    let mut r = rig(t.f_cl, t.sigma).unwrap();
    check(
        "radiating",
        146.83,
        Some(0.5),
        &t,
        played(&mut r, t.root, t.note.f, 99, 3000),
    );

    let rig = two_string_rig(6000.0, 20.0);
    let t = tune_coupled(100.0, Some(1.0), &rig).expect("tunes");
    let mut r = rig(t.f_cl, t.sigma).unwrap();
    check(
        "two strings",
        100.0,
        Some(1.0),
        &t,
        played(&mut r, t.root, t.note.f, 49, 2000),
    );
}

/// The string alone is the wrong answer twice over. Tuned to its clamped note, the coupled
/// instrument plays well over a semitone flat; given the loss that would make it ring 1.5 s by
/// itself, the coupled note rings about twice that. Each control asserts its own miss.
#[test]
fn the_string_alone_misses_the_pitch_and_the_ring_time() {
    let rig = body_rig(8000.0, 0.0);
    let f0 = 100.0;
    let t = tune_coupled(f0, Some(1.5), &rig).expect("tunes");

    // The clamped-note string's own note sits near 0.9 f0 (Part 0 probe: 90.8 Hz): aim there, at
    // the tuned note's radius, and the mode found is the one it plays.
    let mut textbook = rig(f0, t.sigma).unwrap();
    let k = textbook.k();
    let (radius, theta) = (t.root.0.hypot(t.root.1), 2.0 * PI * 0.9 * f0 * k);
    let guess = (radius * theta.cos(), radius * theta.sin());
    let (f, _, residual) = played(&mut textbook, guess, 0.9 * f0, 99, 3000);
    assert!(residual < 1e-9);
    assert!(
        cents(f, f0) < -100.0,
        "the clamped-note string plays {:+.1} c",
        cents(f, f0)
    );

    let alone = string_sigma(Some(1.5), 1.0 / fs_at(t.f_cl));
    assert!(
        t.sigma > 1.5 * alone,
        "solved {} against alone {alone}",
        t.sigma
    );
    let mut textbook = rig(t.f_cl, alone).unwrap();
    let (_, rings, residual) = played(&mut textbook, t.root, t.note.f, 99, 3000);
    assert!(residual < 1e-9);
    assert!(rings > 2.0 * 1.5, "the string-alone loss rings {rings} s");
}

/// In a split band (110 Hz on the 8000 N/m body) the target is still played exactly, and the
/// louder neighbour is reported; outside one (90 Hz) there is none. On a body resonance itself
/// (196 Hz on the 3000 N/m body) the note on the target is mostly body.
#[test]
fn in_a_split_band_the_target_plays_and_the_louder_neighbour_is_named() {
    let rig = body_rig(8000.0, 0.0);
    let t = tune_coupled(110.0, Some(1.0), &rig).expect("tunes");
    assert!(((t.note.f - 110.0) / 110.0).abs() < 1e-11);
    let loud = t.louder_neighbour().expect("110 Hz is in a split band");
    assert!(loud.share > t.note.share);
    assert!(
        cents(loud.f, 110.0).abs() > 100.0,
        "the neighbour is a different note: {} Hz",
        loud.f
    );

    let t = tune_coupled(90.0, Some(1.0), &rig).expect("tunes");
    assert!(t.louder_neighbour().is_none(), "90 Hz: {:?}", t.others);
    assert!(t.note.share > 0.5);

    let t = tune_coupled(196.0, None, body_rig(3000.0, 0.0)).expect("tunes");
    assert!(t.note.share < 0.2, "share {}", t.note.share);
    let loud = t
        .louder_neighbour()
        .expect("a body resonance is the deepest split");
    assert!(loud.share > 0.5, "neighbour share {}", loud.share);
}

/// On two identical strings the note is the bridge-still mode — all string, no body — and the
/// symmetric hybrid a pluck also sounds is reported beside it.
#[test]
fn two_strings_play_the_bridge_still_mode() {
    let rig = two_string_rig(6000.0, 20.0);
    let t = tune_coupled(100.0, Some(1.0), &rig).expect("tunes");
    assert!(t.note.share > 1.0 - 1e-9, "share {}", t.note.share);
    let hybrid = t
        .others
        .iter()
        .find(|o| o.share > 0.3)
        .expect("the symmetric hybrid");
    assert!(cents(hybrid.f, 100.0) < -50.0, "hybrid at {} Hz", hybrid.f);
    // The bridge stays still, so the body's loss cannot reach the note and the string's loss is
    // the single-string one; the hybrid moves the body a little and rings shorter.
    let alone = string_sigma(Some(1.0), 1.0 / fs_on(t.f_cl, N_TWO));
    assert!(
        ((t.sigma - alone) / alone).abs() < 1e-7,
        "sigma {} against {alone}",
        t.sigma
    );
    assert!(hybrid.t60 < t.note.t60, "hybrid rings {} s", hybrid.t60);
}

/// With nothing to share it: at `K = 0` the bridge is two separate parts, the note is the
/// free-ended string's, and the solved loss is exactly [`string_sigma`]'s single-string one.
#[test]
fn a_decoupled_string_gets_the_single_string_loss() {
    for t60 in [0.1, 1.0, 8.0] {
        let t = tune_coupled(55.0, Some(t60), body_rig(0.0, 0.0)).expect("tunes");
        assert!(t.note.share > 1.0 - 1e-12);
        let alone = string_sigma(Some(t60), 1.0 / fs_at(t.f_cl));
        assert!(
            ((t.sigma - alone) / alone).abs() < 1e-8,
            "{t60}: {} against {alone}",
            t.sigma
        );
        // A fixed-free string sounds c / 4L, so its clamped note is an octave above the
        // undamped note, and a decay a pulls the damped note below that: f_u^2 = f^2 + (a/2pi)^2.
        // At 0.1 s that is 2 %, a real string's physics; the grid's own dispersion here is 1e-5.
        let a = 3.0 * LN_10 / t60;
        let f_u = (55.0f64.powi(2) + (a / (2.0 * PI)).powi(2)).sqrt();
        assert!(
            (t.f_cl / (2.0 * f_u) - 1.0).abs() < 1e-4,
            "{t60}: f_cl {} against {}",
            t.f_cl,
            2.0 * f_u
        );
    }
}

/// A pitch no lossless string reaches can still be tuned with a ring time: on the radiating body
/// the guard refuses every string slacker than about 61 Hz, so nothing plays 56.57 Hz lossless,
/// but a 0.05 s ring's string loss pulls a mostly-string note down onto it just inside the wall.
#[test]
fn a_pitch_only_a_lossy_string_reaches_tunes() {
    let air = radiating_rig(133.0);
    let f0 = 40.0 * 2f64.powf(6.0 / 12.0);
    assert_eq!(tune_coupled(f0, None, &air), Err(CoupledError::NoNote));
    let t = tune_coupled(f0, Some(0.05), &air).expect("a lossy string reaches it");
    let mut r = air(t.f_cl, t.sigma).unwrap();
    let (f, rings, residual) = played(&mut r, t.root, t.note.f, 99, 3000);
    assert!(
        residual < 1e-9,
        "the start was not one mode ({residual:.2e})"
    );
    assert!(((f - f0) / f0).abs() < 1e-9, "plays {f} Hz");
    assert!(((rings - 0.05) / 0.05).abs() < 1e-6, "rings {rings} s");
    assert!(t.note.share > 0.99, "share {}", t.note.share);
}

#[test]
fn the_refusals() {
    let rig = body_rig(8000.0, 0.0);
    assert_eq!(
        tune_coupled(0.0, None, &rig),
        Err(CoupledError::BadPitch(0.0))
    );
    assert_eq!(
        tune_coupled(100.0, Some(-1.0), &rig),
        Err(CoupledError::BadRingTime(-1.0))
    );
    // The guard refuses every clamped note below about 61 Hz at this spring, and a 14 Hz target
    // needs one below 56 Hz.
    assert_eq!(
        tune_coupled(14.0, Some(1.0), &rig),
        Err(CoupledError::Refused)
    );
    // Near the wall the search starts at it and still finds the note ...
    let t = tune_coupled(60.0, Some(1.0), &rig).expect("tunes past the wall");
    assert!(((t.note.f - 60.0) / 60.0).abs() < 1e-11);
    // ... but the slackest string the guard allows already plays above 50 Hz, lossless ...
    assert_eq!(tune_coupled(50.0, None, &rig), Err(CoupledError::NoNote));
    // ... and only a heavy string loss pulls a note down to it. The heavier the loss the tighter
    // the string, so the note rings longest at the guard's wall: 0.0338 s (0.033 s tunes there,
    // 0.035 s does not, both below the floor). Its ring time is solved onto the pitch: read off
    // the wide search's grid it was 0.0377 s, 10 % out, which refused 0.035 s as too short.
    match tune_coupled(50.0, Some(1.0), &rig) {
        Err(CoupledError::RingTooLong { longest }) => {
            assert!((0.033..0.035).contains(&longest), "longest {longest} s");
        }
        other => panic!("expected RingTooLong, got {other:?}"),
    }

    // The air alone empties the radiating body's note within a few seconds: it cannot ring 100.
    let air = radiating_rig(133.0);
    let lossless = tune_coupled(146.83, None, &air).expect("tunes").note.t60;
    match tune_coupled(146.83, Some(100.0), &air) {
        Err(CoupledError::RingTooLong { longest }) => {
            assert!(longest < 100.0);
            assert!(
                (longest / lossless - 1.0).abs() < 1e-8,
                "{longest} against {lossless}"
            );
        }
        other => panic!("expected RingTooLong, got {other:?}"),
    }
    // Below the floor the search is not known to find every tuning, so it is not run (the human's
    // call, plan §16); at the floor it is.
    assert_eq!(
        tune_coupled(100.0, Some(1e-4), &rig),
        Err(CoupledError::RingBelowFloor(1e-4))
    );
    // The literal, not the constant: a bar about a constant must not import it.
    let below = 0.05 * (1.0 - 1e-12);
    assert_eq!(
        tune_coupled(100.0, Some(below), &rig),
        Err(CoupledError::RingBelowFloor(below))
    );
    assert!(tune_coupled(100.0, Some(0.05), &rig).is_ok());
}

/// The notes the scan follows are real notes, read off the run. Two ways of following a root from
/// one scan point to the next went wrong in plants (plan §16) that this file's other bars missed:
///
/// - **By rank in the band, not by nearness.** On the 3000 N/m body, 269.09 Hz lies just above
///   the ceiling of the 261 Hz body mode raised by the spring (the hybrid levels off at 267.9 Hz):
///   the only note that reaches it is the mostly-body one at a slack string (share 0.064,
///   `f_cl = 265.2`). Near `f_cl = 470` the 118 Hz note leaves the band's lower edge and every
///   note above it changes rank, so a rank-follower took the jump from 257 to 304 Hz for a
///   crossing and returned a 0.71-share "note" at `f_cl = 471.5` that does not exist.
/// - **Without the radius the change in loss predicts.** At 100.79 Hz a 0.05 s ring needs a heavy
///   string loss; following the root without moving it inward settled on another root and refused
///   the tuning as too short.
#[test]
fn the_followed_notes_are_real_notes() {
    let soft = body_rig(3000.0, 0.0);
    let play = |f0: f64, t60: Option<f64>| {
        let t = tune_coupled(f0, t60, &soft).expect("tunes");
        let mut r = soft(t.f_cl, t.sigma).unwrap();
        let (f, rings, residual) = played(&mut r, t.root, t.note.f, 99, 3000);
        assert!(residual < 1e-9, "{f0}: the start was not one mode");
        assert!(((f - f0) / f0).abs() < 1e-9, "{f0}: plays {f} Hz");
        if let Some(target) = t60 {
            assert!(
                ((rings - target) / target).abs() < 1e-6,
                "{f0}: rings {rings} s"
            );
        }
        t
    };

    // The probe sweep's semitone grid, so a plant's recorded outcome is this fixture's.
    let f0 = 40.0 * 2f64.powf(33.0 / 12.0);
    let t = play(f0, None);
    assert!(t.note.share < 0.1, "share {}", t.note.share);
    assert!((t.f_cl / 265.2026 - 1.0).abs() < 1e-5, "f_cl {}", t.f_cl);

    let t = play(40.0 * 2f64.powf(16.0 / 12.0), Some(0.05));
    assert!(t.note.share > 0.5, "share {}", t.note.share);
}

/// The cases a lossless reference got wrong (plan §16): the radiating body between its air-moved
/// modes, where the lossless operator's 213.6 Hz note does not exist, and the plain body beside a
/// resonance with a very short ring. Each must tune, and play and ring its targets off the run.
#[test]
fn the_hard_cases_tune_and_play() {
    let check = |name: &str, f0: f64, t60: Option<f64>, t: &CoupledTuning, got: (f64, f64, f64)| {
        let (f, rings, residual) = got;
        eprintln!(
            "{name}: f_cl {:.4}, sigma {:.4}, share {:.3}, plays {:+.2e} rel, rings {rings:.6} s",
            t.f_cl,
            t.sigma,
            t.note.share,
            f / f0 - 1.0
        );
        assert!(
            residual < 1e-9,
            "{name}: the start was not one mode ({residual:.2e})"
        );
        assert!(
            ((f - f0) / f0).abs() < 1e-9,
            "{name}: plays {f} Hz for {f0}"
        );
        if let Some(target) = t60 {
            assert!(
                ((rings - target) / target).abs() < 1e-6,
                "{name}: rings {rings} s"
            );
        }
        // No note twice: every other note is a different root.
        for (i, a) in t.others.iter().enumerate() {
            assert!(
                (a.f - t.note.f).abs() > 1e-6 * f0,
                "{name}: a neighbour is the note itself"
            );
            for b in &t.others[i + 1..] {
                assert!(
                    (a.f - b.f).abs() > 1e-6 * f0,
                    "{name}: {} Hz listed twice",
                    a.f
                );
            }
        }
    };
    let air = radiating_rig(133.0);
    for t60 in [None, Some(0.3)] {
        let t = tune_coupled(213.6, t60, &air).expect("the air-moved band tunes");
        let mut r = air(t.f_cl, t.sigma).unwrap();
        let got = played(&mut r, t.root, t.note.f, 99, 3000);
        check(&format!("radiating 213.6 {t60:?}"), 213.6, t60, &t, got);
    }
    let soft = body_rig(3000.0, 0.0);
    // Beside the 110 and 196 Hz resonances the note is mostly body (share about 0.1) and the body
    // is lossless: more string loss pushes its motion further onto the body, so its decay peaks.
    // A half-second ring is below the peak and tunes. A twentieth is past it for that note, but
    // not for every note: past a loss threshold another note, mostly string, reaches the pitch at
    // a string 7 to 9 semitones tighter than the lossless tuning, and rings it (the human's fifth
    // answer, plan §16).
    for (f0, f_cl, share) in [(113.1, 134.89, 0.71), (201.6, 294.11, 0.55)] {
        let t = tune_coupled(f0, Some(0.5), &soft).expect("tunes beside a resonance");
        let mut r = soft(t.f_cl, t.sigma).unwrap();
        let got = played(&mut r, t.root, t.note.f, 99, 3000);
        check(&format!("soft body {f0} 0.5 s"), f0, Some(0.5), &t, got);
        let t = tune_coupled(f0, Some(0.05), &soft).expect("another note rings a twentieth");
        let mut r = soft(t.f_cl, t.sigma).unwrap();
        let got = played(&mut r, t.root, t.note.f, 99, 3000);
        check(&format!("soft body {f0} 0.05 s"), f0, Some(0.05), &t, got);
        assert!((t.f_cl / f_cl - 1.0).abs() < 1e-3, "{f0}: f_cl {}", t.f_cl);
        assert!(
            (t.note.share - share).abs() < 0.01,
            "{f0}: share {}",
            t.note.share
        );
    }
}
