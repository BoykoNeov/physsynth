//! Musical controls for the acoustic bore — `docs/dev/wind-mapping-plan.md`.
//!
//! The bore page takes a tube length and the bell's resistance as a ratio `R/Z0`; a musician sets
//! a pitch and a ring time. This module maps one onto the other. It is an optional helper:
//! [`crate::bore::Params::new`] and its raw arguments are untouched.
//!
//! # Exact at every partial, by a special case
//!
//! The mapping is for a closed–radiating tube at `lambda = 1` (the bore page's only setting), and
//! there it is exact, in pitch and in ring time, at **every** partial. At `lambda = 1` the
//! staggered scheme carries a travelling wave without dispersion; the closed half-cell reflects it
//! with `+1`; and the bell's centred 1×1 solve ([`crate::bore::radiate_node`]) reflects it with
//! exactly
//!
//! ```text
//! rho = (R - Z0) / (R + Z0)
//! ```
//!
//! at every frequency. So one round trip, `2N` steps, multiplies the whole state by `-|rho|`: mode
//! `m` sounds at `(2m - 1) c0 / 4L` and every mode rings for
//!
//! ```text
//! T60 = (2L / c0) ln(1000) / (-ln|rho|)
//! ```
//!
//! The plan's §2 measured both on the one-step map's eigenvalues to `1e-9`. Inverting:
//!
//! ```text
//! L = c0 / (4 f0),     R/Z0 = tanh(3 ln10 L / (c0 T60))
//! ```
//!
//! # The branch
//!
//! `rho` and `1/rho` have the same modulus, so a ring time has two bells: `R < Z0` (`rho < 0`, the
//! open-like end) and `R > Z0` (`rho > 0`). The second makes the bell a closed end — the tube then
//! sounds a whole series an octave up, even harmonics included — so it is not the instrument asked
//! for. `tanh < 1` picks `R < Z0` by construction.
//!
//! Without a bell (the ideal open end) the tube loses nothing and rings forever; the pitch alone is
//! mapped there, the human's call (plan §3).
//!
//! # The reed: the note you hear, by a search on the tube's length
//!
//! A reed on the same tube plays flat of `c0 / 4L` — 62 cents on the reed page's default, a few
//! semitones on a short tube with a soft reed (plan §2) — and no closed form says by how much: the
//! note is a self-oscillation of a nonlinear system. The human's call (plan §3) is that the pitch
//! means the note the reed *plays*, so [`tune_reed_length`] searches the length: it runs the
//! caller's own rig, reads the note off the run's settled tail ([`read_played_note`]) and moves `L`
//! until the note lands within [`ReedSearch::tol_cents`]. The bell rides along: with a ring time,
//! the caller rebuilds it from each trial `L` ([`bell_ratio`]), not from `c0 / 4 f0`.
//!
//! A run is accepted only as **the tube's fundamental**: speaking, settled, and at or below the
//! tube's own note plus [`ReedSearch::squeak_cents`]. Every played fundamental the probe saw was
//! flat of the tube's note, and a squeak can be perfectly steady (plan §6), so "settled" alone
//! would let a squeak steer the search.

use std::f64::consts::LN_10;

/// What [`tune_bore`] gives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoreTuning {
    /// Tube length (m), `c0 / 4 f0`.
    pub l: f64,
    /// The bell's `R / Z0`, in `(0, 1)` — `None` when no ring time was asked for (the open end).
    pub r_ratio: Option<f64>,
}

/// Why a pitch or a ring time cannot be mapped.
#[derive(Debug, Clone, PartialEq)]
pub enum WindTuningError {
    /// An input was zero, negative or not finite. Names the input.
    NonPositive(&'static str),
}

impl std::fmt::Display for WindTuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindTuningError::NonPositive(name) => {
                write!(f, "{name} must be positive and finite.")
            }
        }
    }
}

impl std::error::Error for WindTuningError {}

/// Map a pitch and, with a bell, a ring time onto a closed–radiating bore at `lambda = 1`.
///
/// `t60` is the seconds for every partial's amplitude to fall 60 dB; `None` maps the pitch alone.
///
/// # Errors
/// [`WindTuningError::NonPositive`] when `f0`, `c0` or a given `t60` is not positive and finite.
pub fn tune_bore(f0: f64, t60: Option<f64>, c0: f64) -> Result<BoreTuning, WindTuningError> {
    for (name, v) in [("f0", Some(f0)), ("c0", Some(c0)), ("t60", t60)] {
        if let Some(v) = v {
            if !(v > 0.0 && v.is_finite()) {
                return Err(WindTuningError::NonPositive(name));
            }
        }
    }
    let l = c0 / (4.0 * f0);
    Ok(BoreTuning {
        l,
        r_ratio: t60.map(|t| bell_ratio(l, t, c0)),
    })
}

/// The bell's `R/Z0` that makes every partial of a closed–radiating bore of length `l` at
/// `lambda = 1` ring for `t60` seconds, on the open-like branch — `tanh(3 ln10 L / (c0 T60))`.
pub fn bell_ratio(l: f64, t60: f64, c0: f64) -> f64 {
    (3.0 * LN_10 * l / (c0 * t60)).tanh()
}

/// Partial `m` (from 1) of a closed–open or closed–radiating bore at `lambda = 1`, in Hz.
pub fn bore_partial(m: usize, l: f64, c0: f64) -> f64 {
    (2 * m - 1) as f64 * c0 / (4.0 * l)
}

/// The ring time (s) every partial of a closed–radiating bore at `lambda = 1` has with the bell at
/// `R/Z0 = r_ratio` — [`tune_bore`]'s inverse on its branch.
///
/// `-ln|rho| = 2 atanh(x)` with `x = min(r_ratio, 1/r_ratio)`, so a bell past `Z0` is read on the
/// other branch: the same ring time, but a closed end and another series. Infinite at `r_ratio = 0`
/// (nothing leaves), zero at `r_ratio = 1` (the anechoic bell sends nothing back).
pub fn bore_t60(l: f64, r_ratio: f64, c0: f64) -> f64 {
    let x = r_ratio.min(1.0 / r_ratio);
    3.0 * LN_10 * l / (c0 * x.atanh())
}

// == the reed: the played note ===================================================================

/// What one run of a reed rig sounded, read off its settled tail by [`read_played_note`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Heard {
    /// A note: its frequency over the tail, and how far the tail's second half sits from its
    /// first — the settledness.
    Note {
        /// Frequency over the whole tail (Hz).
        f: f64,
        /// Second half against the first, in cents.
        drift_cents: f64,
    },
    /// Below the speaking gate.
    Silent {
        /// The tail's AC rms over `p_closing`.
        level: f64,
    },
    /// Speaking, but a half-tail is too short to hold two periods of a note an octave below the
    /// tube's own.
    TooShort,
    /// Speaking, but nothing in the tail repeats: no lag correlates past [`PERIODIC_MIN`].
    Aperiodic,
}

/// Why [`period_frequency`] read no frequency.
enum NoPeriod {
    TooShort,
    Aperiodic,
}

/// A lag is the period when its autocorrelation is within this share of the best one.
pub const PERIOD_SHARE: f64 = 0.9;
/// Below this best autocorrelation there is no periodic note to read.
pub const PERIODIC_MIN: f64 = 0.5;

/// Normalised autocorrelation of `x` (mean already removed) at lag `t`.
fn autocorrelation(x: &[f64], t: usize) -> f64 {
    if t == 0 || t >= x.len() {
        return f64::NAN;
    }
    let (mut s, mut a, mut b) = (0.0, 0.0, 0.0);
    for i in 0..x.len() - t {
        s += x[i] * x[i + t];
        a += x[i] * x[i];
        b += x[i + t] * x[i + t];
    }
    s / (a * b).sqrt()
}

/// The lag in `lo..=hi` with the largest autocorrelation, and that value.
fn best_lag(x: &[f64], lo: usize, hi: usize) -> (usize, f64) {
    let mut best = (lo, autocorrelation(x, lo));
    for t in lo + 1..=hi {
        let r = autocorrelation(x, t);
        if r > best.1 {
            best = (t, r);
        }
    }
    best
}

/// The frequency (Hz) whose period repeats in `x` at sample rate `fs`, by autocorrelation, looked
/// for from two octaves above `f_hint` to two below.
///
/// **Not zero crossings.** A reed's wave is square-ish with odd harmonics as strong as a third of
/// the fundamental, and it rings at its edges: a plain crossing count read a note 27 cents flat as
/// 82 cents sharp, and a hysteresis count read a strong third harmonic as the note, an octave and
/// a half up (plan §6). Both called a played fundamental a squeak.
///
/// The search runs on a block-averaged copy (about 32 samples per period of `f_hint`), takes the
/// **smallest** lag within [`PERIOD_SHARE`] of the best (a period's multiples correlate as well as
/// the period), refines it at the full rate, and reads the final period at the largest multiple of
/// it that fits in half of `x`, with a parabola through the peak — `m` periods measured at once.
///
/// `x` must hold two periods of a note an octave below `f_hint` (`4 fs / f_hint` samples): a lag
/// the window cannot hold twice is not searched, and a period missing from the search would leave
/// a harmonic standing in for it.
fn period_frequency(x0: &[f64], fs: f64, f_hint: f64) -> Result<f64, NoPeriod> {
    if (x0.len() as f64) < 4.0 * fs / f_hint {
        return Err(NoPeriod::TooShort);
    }
    let n = x0.len() as f64;
    let mean = x0.iter().sum::<f64>() / n;
    let x: Vec<f64> = x0.iter().map(|v| v - mean).collect();
    let d = ((fs / (32.0 * f_hint)).floor() as usize).max(1);
    let xd: Vec<f64> = x
        .chunks_exact(d)
        .map(|c| c.iter().sum::<f64>() / d as f64)
        .collect();
    let lo = ((fs / (4.0 * f_hint) / d as f64).floor() as usize).max(1);
    let hi = ((4.0 * fs / f_hint / d as f64).ceil() as usize).min(xd.len() / 2);
    if hi < lo + 2 {
        return Err(NoPeriod::TooShort);
    }
    let r: Vec<f64> = (lo..=hi).map(|t| autocorrelation(&xd, t)).collect();
    let r_best = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // NaN (a silent copy) is no repeat either.
    if r_best.is_nan() || r_best <= PERIODIC_MIN {
        return Err(NoPeriod::Aperiodic);
    }
    let coarse = (1..r.len() - 1)
        .find(|&i| r[i] >= r[i - 1] && r[i] >= r[i + 1] && r[i] >= PERIOD_SHARE * r_best)
        .map(|i| (lo + i) * d)
        .ok_or(NoPeriod::Aperiodic)?;
    let (period, _) = best_lag(&x, coarse.saturating_sub(d + 1).max(1), coarse + d + 1);
    let m = (x.len() / 2 / period).max(1);
    let (peak, _) = best_lag(
        &x,
        (m * period).saturating_sub(m + 1).max(2),
        m * period + m + 1,
    );
    let (y0, y1, y2) = (
        autocorrelation(&x, peak - 1),
        autocorrelation(&x, peak),
        autocorrelation(&x, peak + 1),
    );
    let den = y0 - 2.0 * y1 + y2;
    let delta = if den != 0.0 {
        0.5 * (y0 - y2) / den
    } else {
        0.0
    };
    let f = fs * m as f64 / (peak as f64 + delta);
    if f.is_finite() {
        Ok(f)
    } else {
        Err(NoPeriod::Aperiodic)
    }
}

/// Read the note a reed rig played: `signal` is its mouthpiece pressure at `fs`, the tail is the
/// last `settle_frac` of it, and the rig speaks when the tail's AC rms over `p_closing` exceeds
/// `speak_gate` — the reed page's own two numbers, so this and the page's verdict cannot disagree.
/// `tube_note` (the tube's own `c0 / 4L`) only centres the search: a note two octaves either side
/// of it is not looked for.
pub fn read_played_note(
    signal: &[f64],
    fs: f64,
    p_closing: f64,
    settle_frac: f64,
    speak_gate: f64,
    tube_note: f64,
) -> Heard {
    let from = ((1.0 - settle_frac) * signal.len() as f64) as usize;
    let tail = &signal[from.min(signal.len())..];
    if tail.len() < 4 {
        return Heard::TooShort;
    }
    let n = tail.len() as f64;
    let m = tail.iter().sum::<f64>() / n;
    let level = (tail.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt() / p_closing;
    if level.is_nan() || level <= speak_gate {
        return Heard::Silent { level };
    }
    let half = tail.len() / 2;
    let read = || -> Result<Heard, NoPeriod> {
        let a = period_frequency(&tail[..half], fs, tube_note)?;
        let b = period_frequency(&tail[half..], fs, tube_note)?;
        Ok(Heard::Note {
            f: period_frequency(tail, fs, tube_note)?,
            drift_cents: 1200.0 * (b / a).log2(),
        })
    };
    match read() {
        Ok(h) => h,
        Err(NoPeriod::TooShort) => Heard::TooShort,
        Err(NoPeriod::Aperiodic) => Heard::Aperiodic,
    }
}

/// How [`tune_reed_length`] searches; [`ReedSearch::default`] is what the reed page runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReedSearch {
    /// Accept a played note this close to the target (cents). The reader agrees with itself to
    /// ~0.01 cents over half a second, ~0.2 cents over a quarter at 56 Hz (plan §2).
    pub tol_cents: f64,
    /// A note further than this between its tail's two halves is not settled (cents).
    pub settle_cents: f64,
    /// A note this far above the tube's own `c0 / 4L` is not the tube's fundamental (cents).
    pub squeak_cents: f64,
    /// The shortest tube the caller can build (m).
    pub l_min: f64,
    /// The longest tube the caller can build (m).
    pub l_max: f64,
    /// Runs before giving up.
    pub max_runs: usize,
}

impl Default for ReedSearch {
    fn default() -> Self {
        ReedSearch {
            tol_cents: 0.1,
            settle_cents: 1.0,
            squeak_cents: 25.0,
            l_min: 0.1,
            l_max: 2.0,
            max_runs: 12,
        }
    }
}

/// The tube [`tune_reed_length`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReedTuning {
    /// Tube length (m).
    pub l: f64,
    /// The note the reed plays on it (Hz), as read.
    pub played: f64,
    /// The tube's own note `c0 / 4L` (Hz): the reed pulls the played note flat of it.
    pub tube_note: f64,
    /// Runs the search took.
    pub runs: usize,
}

/// Why no tube length puts the reed's played note on the target. Each carries the tube length
/// where it was found.
#[derive(Debug, Clone, PartialEq)]
pub enum ReedTuningError {
    /// An input was zero, negative or not finite. Names the input.
    NonPositive(&'static str),
    /// The reed does not speak.
    Silent {
        /// Tube length (m).
        l: f64,
        /// The tail's AC rms over `p_closing`.
        level: f64,
    },
    /// The run is too short to hold the periods a frequency needs.
    TooShort {
        /// Tube length (m).
        l: f64,
    },
    /// The reed speaks, but nothing it sounds repeats.
    Aperiodic {
        /// Tube length (m).
        l: f64,
    },
    /// The reed sounds well above the tube's own note: not the tube's fundamental.
    Squeak {
        /// Tube length (m).
        l: f64,
        /// What it sounds (Hz).
        played: f64,
        /// The tube's own note (Hz).
        tube_note: f64,
    },
    /// The note moves between the tail's two halves by more than [`ReedSearch::settle_cents`].
    Unsettled {
        /// Tube length (m).
        l: f64,
        /// What it sounds over the tail (Hz).
        played: f64,
        /// Second half against the first (cents).
        drift_cents: f64,
    },
    /// The target lies past what the reed plays on the shortest or the longest tube.
    OutOfReach {
        /// The length at the end of the range (m).
        l: f64,
        /// The note the reed plays there (Hz): the highest or the lowest it reaches.
        played: f64,
    },
    /// [`ReedSearch::max_runs`] runs and the note is not within tolerance.
    NoConvergence {
        /// The last length tried (m).
        l: f64,
        /// What it sounded (Hz).
        played: f64,
    },
    /// The caller's rig failed to run (its message).
    Run(String),
}

impl std::fmt::Display for ReedTuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReedTuningError::NonPositive(name) => write!(f, "{name} must be positive and finite."),
            ReedTuningError::Silent { l, level } => write!(
                f,
                "the reed does not speak on a {l:.3} m tube (level {level:.3}): blow harder."
            ),
            ReedTuningError::TooShort { l } => write!(
                f,
                "the run is too short to read the note on a {l:.3} m tube: lengthen it."
            ),
            ReedTuningError::Aperiodic { l } => write!(
                f,
                "the reed speaks on a {l:.3} m tube, but nothing it sounds repeats: no note."
            ),
            ReedTuningError::Squeak {
                l,
                played,
                tube_note,
            } => write!(
                f,
                "the reed squeaks on a {l:.3} m tube: it sounds {played:.1} Hz, above the tube's \
                 own {tube_note:.1} Hz."
            ),
            ReedTuningError::Unsettled {
                l,
                played,
                drift_cents,
            } => write!(
                f,
                "the note does not settle on a {l:.3} m tube: about {played:.1} Hz, moving \
                 {drift_cents:+.1} cents across the run."
            ),
            ReedTuningError::OutOfReach { l, played } => write!(
                f,
                "out of reach: on a {l:.3} m tube, the end of the range, the reed plays \
                 {played:.2} Hz."
            ),
            ReedTuningError::NoConvergence { l, played } => write!(
                f,
                "the search did not land on the pitch: last {played:.2} Hz on a {l:.3} m tube."
            ),
            ReedTuningError::Run(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ReedTuningError {}

/// Find the tube length at which a reed plays `f0`, the human's call (plan §3).
///
/// `play(l)` runs the caller's rig on a tube of length `l` — the bell rebuilt from `l` when a ring
/// time is set — and reads it with [`read_played_note`]. The search starts at the tube's own
/// `c0 / 4 f0`, rescales `L` by the played note's miss (the note falls about as `1 / L`), then
/// takes secant steps on `ln f` against `ln L`. A step past the range plays the range's end
/// instead, and a fundamental there that still misses names the reach.
///
/// # Errors
/// [`ReedTuningError`]: the first run that is not the tube's fundamental ends the search with its
/// reason, as do a target past the range and [`ReedSearch::max_runs`] runs without landing.
pub fn tune_reed_length<F>(
    f0: f64,
    c0: f64,
    search: &ReedSearch,
    mut play: F,
) -> Result<ReedTuning, ReedTuningError>
where
    F: FnMut(f64) -> Result<Heard, String>,
{
    for (name, v) in [("f0", f0), ("c0", c0)] {
        if !(v > 0.0 && v.is_finite()) {
            return Err(ReedTuningError::NonPositive(name));
        }
    }
    let cents = |a: f64, b: f64| 1200.0 * (a / b).log2();
    let mut l = c0 / (4.0 * f0);
    let mut last: Option<(f64, f64)> = None;
    let mut run = 0;
    loop {
        run += 1;
        let at = l.clamp(search.l_min, search.l_max);
        let (f, drift_cents) = match play(at).map_err(ReedTuningError::Run)? {
            Heard::Silent { level } => return Err(ReedTuningError::Silent { l: at, level }),
            Heard::TooShort => return Err(ReedTuningError::TooShort { l: at }),
            Heard::Aperiodic => return Err(ReedTuningError::Aperiodic { l: at }),
            Heard::Note { f, drift_cents } => (f, drift_cents),
        };
        let tube_note = c0 / (4.0 * at);
        if cents(f, tube_note) > search.squeak_cents {
            return Err(ReedTuningError::Squeak {
                l: at,
                played: f,
                tube_note,
            });
        }
        if drift_cents.abs() > search.settle_cents {
            return Err(ReedTuningError::Unsettled {
                l: at,
                played: f,
                drift_cents,
            });
        }
        if cents(f, f0).abs() < search.tol_cents {
            return Ok(ReedTuning {
                l: at,
                played: f,
                tube_note,
                runs: run,
            });
        }
        if at != l {
            return Err(ReedTuningError::OutOfReach { l: at, played: f });
        }
        if run >= search.max_runs {
            return Err(ReedTuningError::NoConvergence { l: at, played: f });
        }
        let (x2, y2) = (at.ln(), f.ln());
        l = match last {
            Some((x1, y1)) if x2 != x1 && y2 != y1 => {
                (x2 + (f0.ln() - y2) * (x2 - x1) / (y2 - y1)).exp()
            }
            _ => at * f / f0,
        };
        last = Some((x2, y2));
    }
}
