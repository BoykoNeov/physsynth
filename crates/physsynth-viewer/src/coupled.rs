//! Musical controls for the strings on a bridge — the `body`, `radbody`, `sympathetic` and
//! `platebody` pages (`docs/dev/string-mapping-plan.md` §16–§18).
//!
//! The pitch slider sets the note the COUPLED instrument plays, and the ring time a loss on the
//! string, solved together by core's [`tune_coupled`] through the page's own build: the rig the
//! solver tunes is the rig the page then runs. Each page splits its build into a check of the
//! request and an assembly at a given tension and string loss; the physics path assembles at the
//! `T` slider and a lossless string, as it always did.
//!
//! The human's options (plan §17): the ring time can be switched off (a lossless string, only the
//! pitch solved — the fast way), and a split band (a wolf note) can be refused instead of played.

use physsynth_core::coupled_tuning::{tune_coupled, Coupled, CoupledTuning, Note};
use serde_json::{json, Value};

use crate::py::{commas, num, py_repr, round_nd};
use crate::{fnum, model_of, Refusal};

/// The pitch default on the coupled pages (Hz): the shared 110 Hz sits on the body's first mode, a
/// wolf (the agent's call, plan §17).
pub const COUPLED_F0_DEFAULT: f64 = 98.0;
/// The ring-time default (s).
pub const COUPLED_T60_DEFAULT: f64 = 3.0;

/// True when a coupled page's request asks for musical controls, on its `controls` select.
pub fn is_musical(p: &Value) -> bool {
    matches!(
        model_of(p).as_str(),
        "body" | "radbody" | "sympathetic" | "platebody"
    ) && p.get("controls").and_then(Value::as_str) == Some("musical")
}

/// What a coupled page's musical request asks for.
pub struct Targets {
    /// The pitch (Hz).
    pub f0: f64,
    /// The ring time (s), or `None` when the ring-time toggle is off: a lossless string.
    pub t60: Option<f64>,
    /// Refuse a split band rather than play it (the wolf toggle).
    pub refuse_split: bool,
}

/// A boolean toggle, absent meaning `default`.
pub(crate) fn toggle(p: &Value, key: &str, default: bool) -> Result<bool, Refusal> {
    match p.get(key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => Err(Refusal::Param(format!(
            "{key} must be true or false, got {}.",
            py_repr(v)
        ))),
    }
}

/// A string select, absent meaning the first of `options`.
pub(crate) fn select(
    p: &Value,
    key: &str,
    options: &[&'static str],
) -> Result<&'static str, Refusal> {
    match p.get(key) {
        None => Ok(options[0]),
        Some(v) => options
            .iter()
            .copied()
            .find(|o| v.as_str() == Some(o))
            .ok_or_else(|| {
                Refusal::Param(format!(
                    "{key} must be one of {options:?}, got {}.",
                    py_repr(v)
                ))
            }),
    }
}

/// The targets from the request: `f0`, the `ring_time` toggle (off by default) and `t60_low`, the
/// `split` select.
pub fn targets(p: &Value) -> Result<Targets, Refusal> {
    let f0 = fnum(p, "f0", COUPLED_F0_DEFAULT)?;
    // Off unless asked: a lossless string tunes in seconds, an exact ring time in tens of seconds
    // (minutes on two strings) — the human's eleventh answer, plan §17.
    let ring = toggle(p, "ring_time", false)?;
    let t60 = if ring {
        Some(fnum(p, "t60_low", COUPLED_T60_DEFAULT)?)
    } else {
        None
    };
    let refuse_split = select(p, "split", &["play", "refuse"])? == "refuse";
    Ok(Targets {
        f0,
        t60,
        refuse_split,
    })
}

/// The tension (N) at which a string of density `rho` and length `l` has clamped fundamental
/// `f_cl`: `c = 2 L f_cl`, `T = rho c^2`. The solver's builds and the page's own go through this one
/// function, so the string that is tuned and the string that runs are the same to the bit.
pub fn tension_at(rho: f64, l: f64, f_cl: f64) -> f64 {
    let c = 2.0 * l * f_cl;
    rho * c * c
}

/// The steps a run of `audio` seconds takes at sample rate `fs`, `runs` times over.
fn steps(audio: f64, fs: f64, runs: i64) -> i64 {
    runs * crate::py::round_int(audio * fs).max(1)
}

/// Refuse at once when even the slackest string that could sound `f0` would overrun the page's
/// work budget — before a tuning that can take minutes (the human's tenth answer, plan §17).
///
/// The solver sounds `f0` only with a clamped fundamental above `f0 / 1.3`, and the page's sample
/// rate is `c N / (L lambda) = 2 f_cl N / lambda`, so that is the cheapest run there can be. A
/// ceiling, not a promise: the tuning can land on a tighter string (196 Hz on the body page at
/// `N = 100` passes here and is refused after it).
pub fn check_budget_before(
    f0: f64,
    n: i64,
    lam: f64,
    audio: f64,
    runs: i64,
    work_max: i64,
) -> Result<(), Refusal> {
    let fs_min = 2.0 * (f0 / 1.3) * n as f64 / lam;
    let need = steps(audio, fs_min, runs);
    if need <= work_max {
        return Ok(());
    }
    let f0_max = 1.3 * work_max as f64 / (runs as f64 * audio) * lam / (2.0 * n as f64);
    Err(Refusal::Param(format!(
        "work budget exceeded before tuning: any string that sounds {} Hz on this bridge runs at \
         {} Hz or more, so the run takes at least {} steps > {}{}. No pitch above about {} Hz \
         can fit these settings, and one below can still be refused once tuned, when its note \
         needs a tighter string. Lower the pitch, shorten the audio, lower N, or raise lambda.",
        round_nd(f0, 2),
        round_nd(fs_min, 0),
        commas(need),
        commas(work_max),
        runs_note(runs),
        round_nd(f0_max, 1),
    )))
}

/// The budget refusal after the tuning, in the musical controls' terms: the tension slider is
/// hidden there, so the message names what is not.
pub fn over_budget(n_steps: i64, work_max: i64, fs: f64, audio: f64, runs: i64) -> Refusal {
    let audio_max = work_max as f64 / runs as f64 / fs;
    Refusal::Param(format!(
        "work budget exceeded ({} steps > {}{}): the tuned string runs at {} Hz, so {} s of \
         audio does not fit. Shorten the audio to {} s or less, lower N, raise lambda, or lower \
         the pitch.",
        commas(n_steps),
        commas(work_max),
        runs_note(runs),
        round_nd(fs, 0),
        round_nd(audio, 2),
        round_nd(audio_max, 2),
    ))
}

/// [`check_budget_before`] for the plate page, whose budget counts node-steps: every step updates
/// the plate's `n_live` nodes.
pub fn check_node_budget_before(
    f0: f64,
    n: i64,
    lam: f64,
    audio: f64,
    n_live: i64,
    work_max: f64,
) -> Result<(), Refusal> {
    let fs_min = 2.0 * (f0 / 1.3) * n as f64 / lam;
    let need = n_live as f64 * steps(audio, fs_min, 1) as f64;
    if need <= work_max {
        return Ok(());
    }
    let f0_max = 1.3 * work_max / (n_live as f64 * audio) * lam / (2.0 * n as f64);
    Err(Refusal::Param(format!(
        "work budget exceeded before tuning: any string that sounds {} Hz on this bridge runs at \
         {} Hz or more, so the run takes at least ~{:.0}M node-steps > {:.0}M. No pitch above \
         about {} Hz can fit these settings, and one below can still be refused once tuned, when \
         its note needs a tighter string. Lower the pitch, shorten the audio, lower N or n_plate, \
         or raise lambda.",
        round_nd(f0, 2),
        round_nd(fs_min, 0),
        need / 1e6,
        work_max / 1e6,
        round_nd(f0_max, 1),
    )))
}

/// [`over_budget`] for the plate page, in node-steps.
pub fn over_node_budget(work: f64, work_max: f64, fs: f64, audio: f64, n_live: i64) -> Refusal {
    let audio_max = work_max / n_live as f64 / fs;
    Refusal::Param(format!(
        "over budget: ~{:.0}M > {:.0}M node-steps: the tuned string runs at {} Hz, so {} s of \
         audio does not fit. Shorten the audio to {} s or less, lower n_plate or N, raise lambda, \
         or lower the pitch.",
        work / 1e6,
        work_max / 1e6,
        round_nd(fs, 0),
        round_nd(audio, 2),
        round_nd(audio_max, 2),
    ))
}

fn runs_note(runs: i64) -> &'static str {
    if runs > 1 {
        ", this regime runs twice"
    } else {
        ""
    }
}

/// [`tune_coupled`] with the page's options: its refusals as the page's, and a split band refused
/// when the wolf toggle says so.
pub fn tune<R: Coupled>(
    t: &Targets,
    build: impl Fn(f64, f64) -> Option<R>,
) -> Result<CoupledTuning, Refusal> {
    let tu = tune_coupled(t.f0, t.t60, build).map_err(|e| Refusal::Param(e.to_string()))?;
    if t.refuse_split {
        if let Some(loud) = tu.louder_neighbour() {
            return Err(Refusal::Param(format!(
                "this pitch sits in a split band (a wolf note): the note on {} Hz carries {:.0} % \
                 of the string's motion, and a louder one at {} Hz carries {:.0} %. Move the \
                 pitch, or set the split band to play.",
                round_nd(t.f0, 2),
                100.0 * tu.note.share,
                round_nd(loud.f, 2),
                100.0 * loud.share,
            )));
        }
    }
    Ok(tu)
}

fn note(n: &Note) -> Value {
    json!({
        "f": num(n.f),
        // An undamped system's note rings forever.
        "t60": if n.t60.is_finite() { num(n.t60) } else { Value::Null },
        "share": num(n.share),
    })
}

/// The `tuning` block of a coupled page: what was asked, the tension and string loss it maps to,
/// the note put on the target, its louder neighbour if the target sits in a split band, and the
/// page's other notes. `extra` adds a page's own entries (the detune, the sweep's grid).
pub fn tuning_block(t: &Targets, tu: &CoupledTuning, tension: f64, extra: Value) -> Value {
    let loud = tu.louder_neighbour();
    let others: Vec<Value> = tu.others.iter().map(note).collect();
    let mut b = json!({
        "kind": "coupled",
        "f0": num(t.f0),
        "t60": t.t60.map_or(Value::Null, num),
        "ring_time": t.t60.is_some(),
        "f_cl": num(tu.f_cl),
        "T": num(tension),
        "sigma": num(tu.sigma),
        // The note's root of the one-step map, which core's `damped_mode` takes: a test reads the
        // note off a rig of its own with it.
        "root": [num(tu.root.0), num(tu.root.1)],
        "note": note(&tu.note),
        "louder": loud.as_ref().map_or(Value::Null, note),
        "split": loud.is_some(),
        "others": others,
    });
    if let (Value::Object(m), Value::Object(x)) = (&mut b, extra) {
        m.extend(x);
    }
    b
}
