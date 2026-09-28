//! The resolution-horizon read-out — "how far up the frequency range can this configuration be
//! trusted?" — `serialize.py`'s horizon section.
//!
//! None of the physics is here: every number comes from `physsynth_analysis::{horizon, modal}`.
//! What this decides is *which* horizon a scene reports and when it must refuse to report one, and
//! each of the five rules the reference wrote down is a trap the resolution-horizon plan paid for:
//!
//! 1. **never read a horizon off a display array** — the mode set is the scheme's whole resolvable
//!    spectrum, built from its dispersion relation, not the 12 partials a panel ships;
//! 2. **a horizon in hertz is family-dependent, in mode index it is not** — a 2-D scene reports both;
//! 3. **a prefix over the sorted 2-D spectrum is not a horizon** — the hertz ceiling is "the lowest
//!    frequency at which some mode is out of tune", computed as a first failure;
//! 4. **a horizon of zero is live** — `hz` is then `null`, never `0.0` (a measurement) or NaN;
//! 5. **a scene with no reference refuses, with the reason** — `kind` is a two-value union.

use physsynth_analysis::{horizon, modal};
use serde_json::{json, Map, Value};

use crate::py::{int, num, round_nd};

/// The three bounds shipped, in cents; the middle one is the plan's default.
pub const HORIZON_BANDS: [f64; 3] = [1.0, 5.0, 25.0];
/// The bound the front-end shows first.
pub const HORIZON_CENTS_DEFAULT: f64 = 5.0;
/// Mode-set ceiling; the `n_live` guards keep every scene under it.
pub const HORIZON_MAX_MODES: usize = 12_000;

/// Refusal: a staircased outline.
pub const HORIZON_STAIRCASE: &str = "this domain is staircased onto the grid, so the error being \
measured is the SHAPE, not the scheme: the continuum reference is a frequency for a different \
outline, and a cents comparison would mix a geometry error into a dispersion one. That needs a \
geometry-convergence study rather than this primitive (plan section 5).";
/// Refusal: a free plate.
pub const HORIZON_FREE_PLATE: &str = "a free plate's continuum reference is a table of tabulated \
values (Narita/Leissa), not a formula over all modes, so a horizon exists only over the tabulated \
set (plan section 5).";
/// Refusal: a nonlinear resonator.
pub const HORIZON_NONLINEAR: &str = "this resonator is nonlinear: its partials move with \
amplitude, so there is no linear modal oracle to compare against. What it has instead is a \
*refinement* horizon — one point of which is measured in docs/dev/vk-newton-plan.md section 13 — \
and that is a different measurement from this one (plan section 5).";
/// Refusal: a string on a spring-coupled bridge.
pub const HORIZON_COUPLED: &str = "the string in this scene terminates on a spring-coupled bridge \
rather than a fixed end, so its continuum partials are shifted by the coupling the scene exists to \
show. A cents comparison against the uncoupled series would report that shift as a discretisation \
error.";
/// Refusal: the bore family.
pub const HORIZON_BORE: &str = "the bore's resolution question is the Webster area function's, \
not a modal dispersion one: a reflection oracle exists but it is a different comparison, so no \
horizon is quoted (plan section 5).";
/// Refusal: a 3-D room.
pub const HORIZON_ROOM: &str = "a 3-D room's dispersion is direction-dependent, like the \
membrane's but with more directions, and what is recorded for it is the lattice light cone and \
the defective corner mode at the CFL ceiling rather than a pitch horizon (plan section 5). The \
string in the same scene is bridge-coupled, so it has none to quote either.";
/// The fallback for a model key in neither table.
pub const HORIZON_GENERIC: &str = "this scene has no closed-form modal reference to compare \
against, so no resolution horizon is quoted for it (docs/dev/resolution-horizon-plan.md section \
5).";

/// Every model key whose builder does not compute a block, and why. With the builders that do,
/// this partitions the model list the front-end offers (a test derives that list from the markup).
pub const HORIZON_ABSENT: [(&str, &str); 11] = [
    ("tension", HORIZON_NONLINEAR),
    ("geometric", HORIZON_NONLINEAR),
    ("bore", HORIZON_BORE),
    ("reed", HORIZON_BORE),
    ("sympathetic", HORIZON_COUPLED),
    ("body", HORIZON_COUPLED),
    ("radbody", HORIZON_COUPLED),
    ("airload", HORIZON_COUPLED),
    ("platebody", HORIZON_COUPLED),
    ("airbox", HORIZON_ROOM),
    ("vkroom", HORIZON_ROOM),
];

/// The refusal a model key gets when its builder set none.
pub fn absent_reason(model: &str) -> &'static str {
    HORIZON_ABSENT
        .iter()
        .find(|(k, _)| *k == model)
        .map_or(HORIZON_GENERIC, |(_, r)| r)
}

/// The refusing arm: a reason, never a number.
pub fn horizon_none(reason: &str, of: Option<&str>) -> Value {
    let mut m = Map::new();
    m.insert("kind".into(), json!("none"));
    m.insert("reason".into(), json!(reason));
    if let Some(of) = of {
        m.insert("of".into(), json!(of));
    }
    Value::Object(m)
}

/// A mode sequence along which a leading prefix means something: `(name, (discrete, continuum))`.
pub type Family = (String, (Vec<f64>, Vec<f64>));

/// NumPy's sort order for floats: NaN after everything.
fn np_order(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(&b).expect("neither is NaN"),
    }
}

/// `round(x, n)` as a JSON number.
fn rnum(x: f64, nd: usize) -> Value {
    num(round_nd(x, nd))
}

/// Assemble the read-out from one exhaustive mode set plus the families to read an index along.
#[allow(clippy::too_many_arguments)]
pub fn horizon_report(
    f_disc: &[f64],
    f_cont: &[f64],
    labels: &[String],
    families: &[Family],
    scheme: &str,
    dims: i64,
    of: &str,
    nyquist: f64,
) -> Value {
    // `np.argsort(f_cont, kind="stable")`.
    let mut order: Vec<usize> = (0..f_cont.len()).collect();
    order.sort_by(|&i, &j| np_order(f_cont[i], f_cont[j]));
    let fd: Vec<f64> = order.iter().map(|&i| f_disc[i]).collect();
    let fc: Vec<f64> = order.iter().map(|&i| f_cont[i]).collect();
    let lab: Vec<&str> = order.iter().map(|&i| labels[i].as_str()).collect();
    let err = horizon::pitch_error_cents(&fd, &fc).expect("the two sets have one length");
    let last = *fc.last().expect("a report has modes");

    let mut bands = Vec::new();
    for cents in HORIZON_BANDS {
        // The reference wrote `~(abs(err) <= cents)` so that a NaN — a mode the scheme cannot
        // represent — counts OUT of tune; `> cents || is_nan()` is the same predicate over every
        // double, spelled the way clippy accepts.
        let outside = err.iter().position(|e| e.abs() > cents || e.is_nan());
        let mut band = Map::new();
        match outside {
            Some(first) => {
                band.insert("modes".into(), int(first as i64));
                band.insert(
                    "hz".into(),
                    if first > 0 {
                        rnum(fc[first - 1], 1)
                    } else {
                        Value::Null
                    },
                );
                band.insert("saturated".into(), Value::Bool(false));
                band.insert("limited_by".into(), json!(lab[first]));
                band.insert("limit_hz".into(), rnum(fc[first], 1));
                band.insert(
                    "limit_cents".into(),
                    if err[first].is_finite() {
                        rnum(err[first], 2)
                    } else {
                        Value::Null
                    },
                );
            }
            None => {
                band.insert("modes".into(), int(fc.len() as i64));
                band.insert("hz".into(), rnum(last, 1));
                band.insert("saturated".into(), Value::Bool(true));
                band.insert("limited_by".into(), Value::Null);
                band.insert("limit_hz".into(), Value::Null);
                band.insert("limit_cents".into(), Value::Null);
            }
        }
        let mut rows: Vec<(usize, String, bool)> = families
            .iter()
            .map(|(name, (ffd, ffc))| {
                let (index, monotone) = horizon::pitch_horizon(ffd, ffc, cents)
                    .expect("families are built with matching lengths and a positive bound");
                (index, name.clone(), monotone)
            })
            .collect();
        rows.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        let index = rows[0].0;
        let tied: Vec<&(usize, String, bool)> = rows.iter().filter(|r| r.0 == index).collect();
        band.insert("cents".into(), num(cents));
        band.insert("index".into(), int(index as i64));
        band.insert(
            "family".into(),
            json!(tied
                .iter()
                .map(|r| r.1.as_str())
                .collect::<Vec<_>>()
                .join(", ")),
        );
        band.insert("family_tied".into(), Value::Bool(tied.len() > 1));
        band.insert(
            "families".into(),
            Value::Array(
                rows.iter()
                    .map(|r| json!({"name": r.1, "index": int(r.0 as i64), "monotone": r.2}))
                    .collect(),
            ),
        );
        band.insert("monotone".into(), Value::Bool(tied.iter().all(|r| r.2)));
        bands.push(Value::Object(band));
    }

    json!({
        "kind": "prefix",
        "scheme": scheme,
        "dims": int(dims),
        "of": of,
        "n_modes": int(fc.len() as i64),
        "f_max": rnum(last, 1),
        "nyquist": rnum(nyquist, 1),
        "default_cents": num(HORIZON_CENTS_DEFAULT),
        "bands": bands,
    })
}

/// The scheme a 1-D string is stepped with — what its dispersion relation is a function of.
pub enum StringScheme {
    /// The explicit leapfrog, at Courant number `lam`.
    Explicit { lam: f64 },
    /// The implicit theta-scheme with stiffness `kappa` and timestep `k`.
    Theta { kappa: f64, theta: f64, k: f64 },
}

/// What the read-out needs to know about a string.
pub struct StringInfo {
    /// Both ends pinned (`fixed` or `supported`) — the only case with a sine-series reference.
    pub pinned: bool,
    pub n: usize,
    pub c: f64,
    pub l: f64,
    pub fs: f64,
    pub scheme: StringScheme,
}

/// The 1-D string family's read-out.
pub fn string_block(s: &StringInfo, of: &str) -> Value {
    if !s.pinned {
        return horizon_none(
            "this string does not terminate on two pinned ends: one end is the spring-coupled \
             bridge the scene is about, so its continuum partials are shifted by the coupling \
             itself. Comparing them with the uncoupled harmonic series in cents would report that \
             shift as a discretisation error — two mechanisms in one number, which is the same \
             reason a staircased outline is refused below.",
            Some(of),
        );
    }
    let n = s.n;
    if n < 2 {
        return horizon_none("this grid carries no interior modes.", Some(of));
    }
    if n - 1 > HORIZON_MAX_MODES {
        return horizon_none(
            &format!(
                "this grid carries {} modes, past the {HORIZON_MAX_MODES} the read-out examines; \
                 no horizon is reported rather than one read off a truncated spectrum.",
                n - 1
            ),
            Some(of),
        );
    }
    let modes: Vec<i64> = (1..n as i64).collect();
    let labels: Vec<String> = modes.iter().map(|m| format!("partial {m}")).collect();
    let (f_disc, f_cont, scheme) = match s.scheme {
        StringScheme::Theta { kappa, theta, k } => (
            modes
                .iter()
                .map(|&m| {
                    modal::discrete_stiff_mode_frequency(s.c, s.l, n as i64, kappa, k, m, theta)
                })
                .collect::<Vec<_>>(),
            modal::stiff_harmonic_frequencies(s.c, s.l, kappa, n - 1),
            "implicit theta-scheme",
        ),
        StringScheme::Explicit { lam } => (
            modes
                .iter()
                .map(|&m| modal::discrete_mode_frequency(s.c, s.l, n as i64, lam, m))
                .collect(),
            modal::harmonic_frequencies(s.c, s.l, n - 1),
            "explicit leapfrog",
        ),
    };
    let families = vec![("harmonic".to_owned(), (f_disc.clone(), f_cont.clone()))];
    horizon_report(
        &f_disc,
        &f_cont,
        &labels,
        &families,
        scheme,
        1,
        of,
        s.fs / 2.0,
    )
}
