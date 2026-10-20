//! The reed's played note — `src/wind_tuning.rs`, `docs/dev/wind-mapping-plan.md` §6.
//!
//! [`tune_reed_length`] searches the tube's length until the note the reed PLAYS lands on the
//! target, reading each run with [`read_played_note`]. Held here: the landed note re-read by
//! another ear (a Hann-windowed Fourier peak, written below) on the same run, with every reed
//! parameter off the page's defaults; the reader against a wave built to fool a crossing count;
//! and one bar per refusal, each on a rig the plan's sweep found it on.

use physsynth_core::bore::{self as bore, Bore, End};
use physsynth_core::reed::{self as reed, ReedBore};
use physsynth_core::wind_tuning::{
    bell_ratio, read_played_note, tune_reed_length, Heard, ReedSearch, ReedTuningError,
};
use std::f64::consts::PI;

/// The reed page's air, written out, not imported.
const RHO0: f64 = 1.2041;
const C0: f64 = 343.0;
const RADIUS: f64 = 0.008;
/// The reed page's settled tail and speaking gate, written out.
const SETTLE_FRAC: f64 = 0.4;
const SPEAK_GATE: f64 = 0.02;

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

#[derive(Clone, Copy)]
struct Rig {
    n: usize,
    gamma: f64,
    f_reed: f64,
    q_reed: f64,
    /// `None`: the ideal open end.
    t60: Option<f64>,
    secs: f64,
}

/// The page's reed on a tube of length `l`, the bell rebuilt from `l`.
fn build(r: &Rig, l: f64) -> ReedBore {
    let z0 = RHO0 * C0 / (PI * RADIUS * RADIUS);
    let (right, r_bell) = match r.t60 {
        Some(t) => (End::Radiating, bell_ratio(l, t, C0) * z0),
        None => (End::Open, 0.0),
    };
    let p = bore::Params::new(
        l,
        C0 * r.n as f64 / l,
        r.n,
        RADIUS,
        Some((End::Closed, right)),
        0.0,
        r_bell,
        RHO0,
        C0,
    )
    .expect("the bore constructs");
    let b = Bore::new(p);
    let (mu, h0) = (0.03, 4.0e-4);
    let p_closing = mu * (2.0 * PI * r.f_reed).powi(2) * h0;
    let rp = reed::Params::new(
        b.params(),
        r.f_reed,
        r.q_reed,
        mu,
        1.5e-4,
        1.5e-2,
        h0,
        1e-10,
        60,
    )
    .expect("the reed constructs");
    ReedBore::new(rp, b, r.gamma * p_closing)
}

/// One run: the mouthpiece pressure, the sample rate and `p_closing`.
fn run(r: &Rig, l: f64) -> (Vec<f64>, f64, f64) {
    let mut rb = build(r, l);
    let k = rb.params().k;
    let steps = ((r.secs / k).round() as usize).max(1);
    let mut sig = Vec::with_capacity(steps);
    for _ in 0..steps {
        rb.step().expect("the reed steps");
        sig.push(rb.mouthpiece_pressure());
    }
    (sig, 1.0 / k, rb.params().p_closing)
}

fn tune(r: &Rig, f0: f64, search: &ReedSearch) -> Result<(f64, f64, f64), ReedTuningError> {
    tune_reed_length(f0, C0, search, |l| {
        let (sig, fs, pc) = run(r, l);
        Ok(read_played_note(
            &sig,
            fs,
            pc,
            SETTLE_FRAC,
            SPEAK_GATE,
            C0 / (4.0 * l),
        ))
    })
    .map(|t| (t.l, t.played, t.tube_note))
}

/// Another ear: the peak of the Hann-windowed Fourier magnitude within 3 % of `f0`, a grid then
/// a golden section.
fn fourier_peak(x0: &[f64], fs: f64, f0: f64) -> f64 {
    let n = x0.len();
    let m = x0.iter().sum::<f64>() / n as f64;
    let x: Vec<f64> = x0
        .iter()
        .enumerate()
        .map(|(i, v)| (v - m) * (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect();
    let power = |f: f64| {
        let w = 2.0 * PI * f / fs;
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
            .fold(lo, |b, f| if power(f) > power(b) { f } else { b });
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

/// The landed note, re-read by the other ear on the same run's tail: within a cent, flat of the
/// tube's own note, at three pitches with every reed parameter off the page's defaults.
#[test]
fn the_reed_plays_the_pitch_asked_for_by_another_ear() {
    let fixtures = [
        (
            147.0,
            Rig {
                n: 48,
                gamma: 0.45,
                f_reed: 1800.0,
                q_reed: 3.0,
                t60: Some(1.5),
                secs: 0.5,
            },
        ),
        (
            262.0,
            Rig {
                n: 80,
                gamma: 0.7,
                f_reed: 3100.0,
                q_reed: 9.0,
                t60: None,
                secs: 0.5,
            },
        ),
        (
            98.0,
            Rig {
                n: 48,
                gamma: 0.6,
                f_reed: 2700.0,
                q_reed: 6.0,
                t60: Some(20.0),
                secs: 0.6,
            },
        ),
    ];
    for (f0, rig) in fixtures {
        let (l, played, tube_note) = tune(&rig, f0, &ReedSearch::default()).expect("it tunes");
        assert!(cents(played, f0).abs() < 0.1, "f0 {f0}: read {played}");
        assert!(
            played < tube_note,
            "f0 {f0}: the reed pulls flat of {tube_note}"
        );
        let (sig, fs, _) = run(&rig, l);
        let tail = &sig[((1.0 - SETTLE_FRAC) * sig.len() as f64) as usize..];
        let heard = fourier_peak(tail, fs, f0);
        assert!(
            cents(heard, f0).abs() < 1.0,
            "f0 {f0}: the other ear hears {heard} Hz ({:+.3} cents)",
            cents(heard, f0)
        );
    }
}

/// An odd-harmonic wave with its third at 60 % of the fundamental, in antiphase, and a little
/// ripple: it crosses its mean three times a period, so a plain crossing count reads the third
/// harmonic as the note. The period is still one.
#[test]
fn the_reader_hears_the_fundamental_under_a_loud_third_harmonic() {
    let (fs, f) = (44_100.0, 151.3);
    let sig: Vec<f64> = (0..(0.5 * fs) as usize)
        .map(|i| {
            let w = 2.0 * PI * f * i as f64 / fs;
            w.sin() - 0.6 * (3.0 * w).sin() + 0.05 * (31.0 * w).sin()
        })
        .collect();
    match read_played_note(&sig, fs, 1.0, SETTLE_FRAC, SPEAK_GATE, 160.0) {
        Heard::Note {
            f: got,
            drift_cents,
        } => {
            assert!(cents(got, f).abs() < 0.01, "read {got}");
            assert!(drift_cents.abs() < 0.01);
        }
        other => panic!("{other:?}"),
    }
    // The control: a plain mean-crossing count of the same wave is not the note.
    let tail = &sig[(0.6 * sig.len() as f64) as usize..];
    let ups = tail
        .windows(2)
        .filter(|p| p[0] < 0.0 && p[1] >= 0.0)
        .count() as f64;
    let naive = ups * fs / tail.len() as f64;
    assert!(cents(naive, f) > 600.0, "a crossing count reads {naive}");
}

#[test]
fn the_reader_names_silence_shortness_and_no_repeat() {
    let fs = 20_000.0;
    let quiet = vec![1e-6; 20_000];
    assert!(matches!(
        read_played_note(&quiet, fs, 1.0, SETTLE_FRAC, SPEAK_GATE, 100.0),
        Heard::Silent { .. }
    ));
    // 0.2 s: each half-tail holds 0.04 s, less than two periods an octave below 60 Hz.
    let short: Vec<f64> = (0..4_000)
        .map(|i| (2.0 * PI * 60.0 * i as f64 / fs).sin())
        .collect();
    assert_eq!(
        read_played_note(&short, fs, 1.0, SETTLE_FRAC, SPEAK_GATE, 60.0),
        Heard::TooShort
    );
    // A deterministic noise (an LCG) speaks and repeats nowhere.
    let mut s = 12_345u64;
    let noise: Vec<f64> = (0..20_000)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    assert_eq!(
        read_played_note(&noise, fs, 1.0, SETTLE_FRAC, SPEAK_GATE, 100.0),
        Heard::Aperiodic
    );
}

// -- one bar per refusal, each on a rig the plan's sweep found it on ----------------------------

fn rig(n: usize, gamma: f64, f_reed: f64, q_reed: f64, t60: Option<f64>, secs: f64) -> Rig {
    Rig {
        n,
        gamma,
        f_reed,
        q_reed,
        t60,
        secs,
    }
}

#[test]
fn a_reed_blown_too_softly_is_refused_as_silent() {
    let r = rig(64, 0.2, 2500.0, 4.0, Some(40.0), 0.2);
    assert!(matches!(
        tune(&r, 45.0, &ReedSearch::default()),
        Err(ReedTuningError::Silent { .. })
    ));
}

#[test]
fn a_render_too_short_for_a_low_note_is_refused() {
    let r = rig(64, 0.3, 2500.0, 4.0, Some(40.0), 0.2);
    assert!(matches!(
        tune(&r, 45.0, &ReedSearch::default()),
        Err(ReedTuningError::TooShort { .. })
    ));
}

#[test]
fn a_squeak_above_the_tube_is_refused_not_tuned_toward() {
    // A stiff reed near its onset on a short tube sounds 694 Hz over a tube whose own note is
    // 440: steady, and not the tube's fundamental.
    let r = rig(64, 0.36, 3500.0, 8.0, Some(40.0), 0.2);
    match tune(&r, 440.0, &ReedSearch::default()) {
        Err(ReedTuningError::Squeak {
            played, tube_note, ..
        }) => assert!(cents(played, tube_note) > 25.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_note_that_wanders_is_refused_as_unsettled() {
    let r = rig(64, 0.36, 1500.0, 2.0, Some(0.3), 0.2);
    assert!(matches!(
        tune(&r, 110.0, &ReedSearch::default()),
        Err(ReedTuningError::Unsettled { .. })
    ));
}

#[test]
fn a_pitch_past_the_shortest_tube_names_the_highest_note() {
    let r = rig(64, 0.51, 1000.0, 4.0, Some(40.0), 0.2);
    match tune(&r, 440.0, &ReedSearch::default()) {
        Err(ReedTuningError::OutOfReach { l, played }) => {
            assert_eq!(l, 0.1);
            assert!(played < 440.0 && cents(played, 440.0) > -50.0, "{played}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_reed_that_sounds_but_never_repeats_is_refused() {
    let r = rig(64, 0.3, 1000.0, 4.0, Some(0.3), 1.0);
    assert!(matches!(
        tune(&r, 45.0, &ReedSearch::default()),
        Err(ReedTuningError::Aperiodic { .. })
    ));
}

#[test]
fn a_search_out_of_runs_is_refused_with_its_last_note() {
    // One run cannot land: the first guess is the tube's own note, and the reed pulls flat of it.
    let r = rig(48, 0.45, 1800.0, 3.0, Some(1.5), 0.5);
    let one = ReedSearch {
        max_runs: 1,
        ..ReedSearch::default()
    };
    match tune(&r, 147.0, &one) {
        Err(ReedTuningError::NoConvergence { played, .. }) => assert!(played < 147.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_non_positive_pitch_is_refused_by_name() {
    let r = rig(48, 0.45, 1800.0, 3.0, Some(1.5), 0.5);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            tune(&r, bad, &ReedSearch::default()),
            Err(ReedTuningError::NonPositive("f0"))
        );
    }
}
