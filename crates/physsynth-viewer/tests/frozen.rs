//! The Python viewer's own outputs, frozen before it was deleted (retirement plan §23.19).
//!
//! Every scene of phase D was checked against the live Python serializer by parsing both payloads
//! and comparing the trees: exact, unless a path falls under one of the named tolerance classes
//! below. That reference is gone now, so its outputs are kept here instead, recorded from the
//! Python on a freshly reinstalled wheel: 588 requests over 26 corpora in `tests/frozen/`.
//!
//! The payloads were 180 MB, so the freeze is a DIGEST that loses no exactness:
//!
//! - every key, string, bool, null, and int-versus-float type is kept as it was;
//! - every number is kept, except inside a numeric list longer than 16, which becomes its length
//!   and an FNV-1a hash of each element's type tag and bits;
//! - a base64 buffer becomes its byte length and an FNV-1a hash of its bytes — every `….b64`, and
//!   the geometric string's `orbit.u` / `orbit.w`, which are buffers under other names (a scan of
//!   every recorded payload for long base64 strings found those two and no others);
//! - a buffer or long list under a tolerance class keeps its hash (so an exact match is still
//!   recognised as exact), 64 samples plus its argmax, and the max and min of 64 blocks. The class's
//!   bar bounds those soundly: an extreme cannot move further than the worst element does.
//! - Python's `NaN`, which its server could not serialize and this crate ships as `null`, is
//!   `{"$nan": 0}`.
//!
//! # Two modes, because exactness is a claim about the platform
//!
//! The recording platform is `x86_64-pc-windows-msvc`. There, and in the `frozen-windows` CI job,
//! every comparison above runs. Everywhere else — the Linux CI job — only what cannot pass through
//! a transcendental is compared: keys, types, lengths, ints, strings and bools. Float values,
//! hashes and samples are skipped. Measured 2026-09-28: a GitHub Windows runner reproduced all 588
//! cases to the bit; the Linux runner differed in 158, all of them floats except one, and the
//! parametric scene's own growth carried a last-bit difference to ~1e-6 in its audio. No tolerance
//! that admits that would assert anything anywhere else, so none is attempted.
//!
//! What this catches: a transcription error, a wrong branch, a regression. What it cannot catch is
//! an error the Python made too — the per-scene tests beside this file are for that.

mod common;

use common::decode_b64;
use physsynth_viewer::{simulate_to_payload, MODELS};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

// -- tolerance classes: `compare.py`'s, carried as they were (none widened) -----------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tol {
    /// 2 float32 eps of the buffer's PEAK.
    F32Peak,
    Abs(f64),
    Rel(f64),
    /// Any value; the leaf is meaningless in the reference too.
    Any,
}

/// `path` minus one trailing `[digits]`, if it has one.
fn strip_index(path: &str) -> Option<&str> {
    let body = path.strip_suffix(']')?;
    let open = body.rfind('[')?;
    let digits = &body[open + 1..];
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(&body[..open])
}

/// A key made only of `[a-z_]`, as the reference's regexes spell `[a-z_]+`.
fn lower_key(s: &str, allow_empty: bool) -> bool {
    (allow_empty || !s.is_empty()) && s.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
}

/// The tolerance class a leaf falls under, or `None` for exact. The model-restricted classes come
/// first and apply to one scene only, so a loosening one scene needs cannot hide a regression in
/// another.
fn classify(model: &str, path: &str) -> Option<Tol> {
    if model == "vkroom" {
        // The room-loaded gong's radiated-energy LEDGER: the binding accumulated it through NumPy's
        // dot, the core through its own read-out sum (retirement plan §16: a reduction that does
        // not feed back takes the crate's spelling), so the total and every fraction of it differ
        // in the last bit; measured <= 1.6e-15 relative, every state array identical.
        if strip_index(path) == Some("energy.value")
            || strip_index(path)
                .and_then(|b| b.strip_prefix("meta.ledger."))
                .and_then(|k| k.strip_suffix("_frac"))
                .is_some_and(|k| lower_key(k, false))
        {
            return Some(Tol::Rel(1e-13));
        }
        // Differences of two nearly equal ledgers (~5e-13 and ~1e-17): the ledger's own last bit
        // is the whole of the difference; measured <= 1.7e-16 absolute.
        if matches!(
            path,
            "energy.lossless.drift" | "meta.ledger.total_drift" | "meta.ledger.residual_max"
        ) {
            return Some(Tol::Abs(1e-15));
        }
    }
    // resample_poly's taps call sin and Cephes i0; NumPy's sin is CPU-dispatched and the Bessel is
    // a different expansion here, so a tap can differ in its last bit. Audio is float32, and the
    // bar is 2 float32 eps of the buffer's PEAK — a sample that should cancel to zero carries
    // ~1e-33 of noise in both, so a per-sample ulp count is the wrong scale.
    if path == "audio.b64" {
        return Some(Tol::F32Peak);
    }
    // Off-mode FRACTIONS of the driven amplitude, from BLAS ddot/nrm2 on a residual u - q*shape
    // that cancels to ~1e-7 (or ~1e-14 for a pure mode): the dot's last bit is amplified by the
    // cancellation, so relative error is the wrong scale; the trajectories themselves are exact.
    let unindexed = strip_index(path).unwrap_or(path);
    if let Some(rest) = unindexed.strip_prefix("meta.spectrum.") {
        if let Some((side, leaf)) = rest.split_once('.') {
            if matches!(side, "above" | "below")
                && matches!(leaf, "off" | "env" | "level" | "floor")
            {
                return Some(Tol::Abs(1e-15));
            }
        }
    }
    if matches!(
        path,
        "meta.spectrum.cascade.grid_scale" | "meta.spectrum.purity.off_mode"
    ) {
        return Some(Tol::Abs(1e-15));
    }
    if let Some(rest) = path.strip_prefix("meta.spectrum.sweep.points") {
        if let Some(i) = rest.strip_suffix(".floor") {
            if strip_index(&format!("x{i}")) == Some("x") {
                return Some(Tol::Abs(1e-15));
            }
        }
    }
    // Magnitude spectra NORMALIZED to their band maximum, through an FFT that is not pocketfft
    // (Bluestein at the record's own length): last-bit differences, so the scale is the peak (1.0).
    if let Some(key) = unindexed.strip_prefix("meta.spectrum.") {
        if key.strip_suffix("mag").is_some_and(|k| lower_key(k, true)) {
            return Some(Tol::Abs(1e-12));
        }
    }
    if let Some(b) = strip_index(path) {
        if let Some(name) = b
            .strip_prefix("meta.spectrum.spectra.")
            .and_then(|k| k.strip_suffix(".mag"))
        {
            if lower_key(name, false) {
                return Some(Tol::Abs(1e-12));
            }
        }
        // The bore's band spectrum, normalized to the max of a band that can hold only ~1e-6 of
        // the record's peak (anechoic bell: the pulse leaves; measured 1.06e-6). An FFT's rounding
        // is relative to the RECORD, so the band normalization amplifies it ~1e6x: measured 2.8e-11.
        if b == "meta.spectrum.spectrum.mag" {
            return Some(Tol::Abs(1e-9));
        }
    }
    // The bore's O(h^2) ratio at lambda = 1, where BOTH departures are rounding residue (each
    // rounds to 0.0 cents in both payloads): the ratio of two noises, meaningless in the reference
    // too (2.667 there, 1.2 here for one case). The front-end reads only order[0].
    if path == "meta.dispersion.order[8]" {
        return Some(Tol::Any);
    }
    match path {
        // np.polyfit is an SVD least squares; this is the closed form about the mean.
        "energy.lossy.measured_2sigma" => Some(Tol::Rel(1e-12)),
        // The fret's fitted decay rate: np.polyfit (SVD) vs the closed form, and ABSOLUTE because
        // at sigma0 = 0 the rate is a fit to conserved energy (~6e-12 of noise) where no relative
        // bar means anything; 1e-12 is 1e-12 of a ~1 s^-1 rate.
        "energy.decay_triple.rate" => Some(Tol::Abs(1e-12)),
        // 2 sigma0 <2KE/E>, where KE is a BLAS ddot of the centred velocity (blocked) against a
        // left-to-right dot: a last-bit difference per step, averaged over the run.
        "energy.decay_triple.corrected" => Some(Tol::Rel(1e-13)),
        _ => None,
    }
}

// -- the digest comparison ------------------------------------------------------------------------

fn fnv(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// One list element as the freezer hashed it: a type tag, then 8 bytes. `-0.0` is hashed as `0.0`,
/// because the reference compared with Python's `==`.
fn num_bytes(v: &Value) -> Option<[u8; 9]> {
    let mut out = [0u8; 9];
    match v {
        Value::Null => out[0] = 2,
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64()?;
            out[0] = 1;
            out[1..].copy_from_slice(&(if x == 0.0 { 0.0f64 } else { x }).to_le_bytes());
        }
        Value::Number(n) => out[1..].copy_from_slice(&n.as_i64()?.to_le_bytes()),
        _ => return None,
    }
    Some(out)
}

/// Exact comparison is a claim about the platform's math library as much as about this crate:
/// `sin`, `exp` and `ln` come from the C runtime, and a last-bit difference in one of them moves the
/// last bits of every step downstream. The payloads were recorded on `x86_64-pc-windows-msvc`, and a
/// GitHub Windows runner reproduced all 588 of them to the bit; the Linux runner differed in 158,
/// every difference but one a float (retirement plan §23.19). So the exact claim is made where it
/// holds, and every other platform compares structure only.
const EXACT: bool = cfg!(all(
    target_os = "windows",
    target_env = "msvc",
    target_arch = "x86_64"
));

#[derive(Default)]
struct Verdict {
    /// Compare float values, hashes and samples (the recording platform), or only what does not
    /// pass through a transcendental: keys, types, lengths, strings, bools, ints.
    exact: bool,
    /// Differences inside a stated tolerance (or a NaN shipped as the null it has to be).
    soft: Vec<String>,
    hard: Vec<String>,
    /// Of `hard`: scalar floats (with the worst relative gap), hashed lists, hashed buffers — so a
    /// failure says whether it is a last-bit drift or a change of structure.
    n_float: usize,
    worst_rel: f64,
    n_list: usize,
    n_buf: usize,
    /// Samples or block extremes of a classed buffer or list, outside the class's bar.
    n_sampled: usize,
    /// Everything else — keys, lengths, types, ints, strings, bools: a change of STRUCTURE, never
    /// a last bit. Listed first in a failure, because it is the one that matters.
    other: Vec<String>,
}

impl Verdict {
    fn new() -> Self {
        Self {
            exact: EXACT,
            ..Self::default()
        }
    }

    /// A structural difference: recorded in `hard` and, for the failure message, in `other`.
    fn structural(&mut self, msg: String) {
        self.other.push(msg.clone());
        self.hard.push(msg);
    }
}

/// The samples and block extremes of a classed buffer or list, against the got values.
fn cmp_sampled(want: &Value, got: &[f64], path: &str, tol: Tol, v: &mut Verdict) {
    let n = want["n"].as_u64().unwrap() as usize;
    if got.len() != n {
        v.structural(format!("{path}: length {n} vs {}", got.len()));
        return;
    }
    if !v.exact {
        return;
    }
    let (maxes, mins) = (
        want["max"].as_array().unwrap(),
        want["min"].as_array().unwrap(),
    );
    let bar_of = |x: f64| -> f64 {
        match tol {
            Tol::F32Peak => {
                let peak = maxes
                    .iter()
                    .chain(mins)
                    .map(|m| m.as_f64().unwrap().abs())
                    .fold(0.0, f64::max);
                2.0 * f64::from(f32::EPSILON) * peak
            }
            Tol::Abs(t) => t,
            Tol::Rel(t) => t * x.abs(),
            Tol::Any => f64::INFINITY,
        }
    };
    let mut worst = 0.0f64;
    let mut check = |label: &str, w: f64, g: f64, v: &mut Verdict| {
        let d = (w - g).abs();
        worst = worst.max(d);
        if d > bar_of(w) {
            v.n_sampled += 1;
            v.hard.push(format!("{path}{label}: {w} vs {g}"));
        }
    };
    for s in want["at"].as_array().unwrap() {
        let i = s[0].as_u64().unwrap() as usize;
        check(&format!("[{i}]"), s[1].as_f64().unwrap(), got[i], v);
    }
    let nb = maxes.len();
    for b in 0..nb {
        let seg = &got[b * n / nb..(b + 1) * n / nb];
        let (mx, mn) = seg
            .iter()
            .fold((f64::NEG_INFINITY, f64::INFINITY), |(a, b), &x| {
                (a.max(x), b.min(x))
            });
        check(
            &format!(" block {b} max"),
            maxes[b].as_f64().unwrap(),
            mx,
            v,
        );
        check(&format!(" block {b} min"), mins[b].as_f64().unwrap(), mn, v);
    }
    v.soft.push(format!("{path}: within, worst {worst:e}"));
}

fn cmp(want: &Value, got: &Value, path: &str, model: &str, v: &mut Verdict) {
    let join = |k: &str| {
        if path.is_empty() {
            k.to_owned()
        } else {
            format!("{path}.{k}")
        }
    };
    if let Some(m) = want.as_object().filter(|m| m.len() == 1) {
        let (tag, body) = m.iter().next().unwrap();
        match tag.as_str() {
            "$nan" => {
                if got.is_null() {
                    v.soft.push(format!("{path}: NaN shipped as null"));
                } else {
                    v.structural(format!("{path}: NaN in the reference, {got} here"));
                }
                return;
            }
            "$buf" => {
                let Some(s) = got.as_str() else {
                    v.structural(format!("{path}: a buffer in the reference, {got} here"));
                    return;
                };
                let raw = decode_b64(s);
                let (n, h) = (
                    body[0].as_u64().unwrap() as usize,
                    body[1].as_str().unwrap(),
                );
                if raw.len() != n {
                    v.structural(format!("{path}: {n} bytes vs {}", raw.len()));
                } else if v.exact && fnv(raw.iter().copied()) != h {
                    v.n_buf += 1;
                    v.hard.push(format!(
                        "{path}: buffer differs ({n} bytes vs {})",
                        raw.len()
                    ));
                }
                return;
            }
            "$audio" => {
                let Some(s) = got.as_str() else {
                    v.structural(format!("{path}: audio in the reference, {got} here"));
                    return;
                };
                let raw = decode_b64(s);
                if v.exact && fnv(raw.iter().copied()) == body["fnv"].as_str().unwrap() {
                    return;
                }
                let xs: Vec<f64> = raw
                    .chunks_exact(4)
                    .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect();
                let tol = classify(model, path).expect("an $audio leaf is classed");
                cmp_sampled(body, &xs, path, tol, v);
                return;
            }
            "$nums" | "$numsc" => {
                let Some(list) = got.as_array() else {
                    v.structural(format!("{path}: a list in the reference, {got} here"));
                    return;
                };
                let bytes: Option<Vec<u8>> = list
                    .iter()
                    .map(num_bytes)
                    .collect::<Option<Vec<_>>>()
                    .map(|b| b.concat());
                let (n, h) = if tag == "$nums" {
                    (
                        body[0].as_u64().unwrap() as usize,
                        body[1].as_str().unwrap(),
                    )
                } else {
                    (
                        body["n"].as_u64().unwrap() as usize,
                        body["fnv"].as_str().unwrap(),
                    )
                };
                if list.len() != n {
                    v.structural(format!("{path}: {n} elements vs {}", list.len()));
                    return;
                }
                if !v.exact || bytes.as_deref().map(fnv_of) == Some(h.to_owned()) {
                    return;
                }
                if tag == "$nums" {
                    v.n_list += 1;
                    v.hard.push(format!(
                        "{path}: list differs ({n} elements vs {})",
                        list.len()
                    ));
                    return;
                }
                // Classed: every element must be a float to be compared under the bar.
                let xs: Option<Vec<f64>> = list
                    .iter()
                    .map(|x| {
                        x.as_f64()
                            .filter(|_| x.as_number().is_some_and(|n| n.is_f64()))
                    })
                    .collect();
                let Some(xs) = xs else {
                    v.structural(format!("{path}: a non-float element"));
                    return;
                };
                let tol = classify(model, &format!("{path}[0]")).expect("a $numsc list is classed");
                cmp_sampled(body, &xs, path, tol, v);
                return;
            }
            _ => {}
        }
    }
    match (want, got) {
        (Value::Object(a), Value::Object(b)) => {
            let (ka, kb): (BTreeSet<&String>, BTreeSet<&String>) =
                (a.keys().collect(), b.keys().collect());
            if ka != kb {
                v.structural(format!(
                    "{path}: keys only in the reference {:?}, only here {:?}",
                    ka.difference(&kb).collect::<Vec<_>>(),
                    kb.difference(&ka).collect::<Vec<_>>()
                ));
            }
            for k in ka.intersection(&kb) {
                cmp(&a[*k], &b[*k], &join(k), model, v);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                v.structural(format!("{path}: length {} vs {}", a.len(), b.len()));
                return;
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                cmp(x, y, &format!("{path}[{i}]"), model, v);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            if a.is_f64() != b.is_f64() {
                v.structural(format!("{path}: int-vs-float {a} vs {b}"));
                return;
            }
            // `Number`'s `==` is `f64`'s for floats, so `-0.0 == 0.0` here as in the reference
            if a == b {
                return;
            }
            if !a.is_f64() {
                v.structural(format!("{path}: int {a} vs {b}"));
                return;
            }
            if !v.exact {
                return;
            }
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let within = match classify(model, path) {
                Some(Tol::Abs(t)) => (x - y).abs() <= t,
                Some(Tol::Rel(t)) => (x - y).abs() / x.abs().max(1e-300) <= t,
                Some(Tol::Any) => true,
                Some(Tol::F32Peak) | None => false,
            };
            if within {
                v.soft.push(format!("{path}: {x} vs {y}"));
            } else {
                v.n_float += 1;
                v.worst_rel = v.worst_rel.max((x - y).abs() / x.abs().max(1e-300));
                v.hard.push(format!("{path}: {a} vs {b}"));
            }
        }
        _ => {
            if want != got {
                v.structural(format!("{path}: {want} vs {got}"));
            }
        }
    }
}

fn fnv_of(b: &[u8]) -> String {
    fnv(b.iter().copied())
}

// -- the fixtures ---------------------------------------------------------------------------------

fn frozen_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/frozen")
}

fn load(corpus: &str) -> Map<String, Value> {
    let p = frozen_dir().join(format!("{corpus}.json"));
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["corpus"], corpus);
    v["cases"].as_object().unwrap().clone()
}

fn corpora() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(frozen_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Run one corpus; panic listing every case with a difference outside its class.
fn check(corpus: &str) {
    let cases = load(corpus);
    let (mut exact, mut within) = (0, 0);
    let mut failing = Vec::new();
    for (key, case) in &cases {
        let req = &case["request"];
        let model = physsynth_viewer::model_of(req);
        let got = simulate_to_payload(req);
        let mut v = Verdict::new();
        cmp(&case["expect"], &got, "", &model, &mut v);
        if !v.hard.is_empty() {
            // structure first: it is never a last bit
            let shown: Vec<&String> = v
                .other
                .iter()
                .chain(v.hard.iter().filter(|h| !v.other.contains(h)))
                .take(8)
                .collect();
            failing.push(format!(
                "{key}: {} differences ({} structural; {} scalar floats, worst relative {:.1e}; \
                 {} hashed lists; {} buffers; {} sampled), first {shown:?}",
                v.hard.len(),
                v.other.len(),
                v.n_float,
                v.worst_rel,
                v.n_list,
                v.n_buf,
                v.n_sampled
            ));
        } else if v.soft.is_empty() {
            exact += 1;
        } else {
            within += 1;
        }
    }
    if std::env::var_os("FROZEN_REPORT").is_some() {
        eprintln!(
            "{corpus}: {exact} exact, {within} within a stated tolerance, {} failing (of {})",
            failing.len(),
            cases.len()
        );
    }
    assert!(failing.is_empty(), "{corpus}:\n{}", failing.join("\n"));
}

macro_rules! corpus_tests {
    ($($name:ident),* $(,)?) => {
        $(#[test] fn $name() { check(stringify!($name)); })*
        /// Every fixture file has a test, and every test a fixture file.
        const TESTED: &[&str] = &[$(stringify!($name)),*];
    };
}

corpus_tests!(
    airbox, airload, body, bore, browser, browser4, browser5, browser5b, browser5c, browser5d,
    browser5e, browser6a, browser6b, contact, d2, geom, mallet, membrane, plate, platebody,
    radbody, reed, strings, symp, vk, vkroom,
);

#[test]
fn every_fixture_file_is_run() {
    let tested: BTreeSet<String> = TESTED.iter().map(|s| (*s).to_owned()).collect();
    let files: BTreeSet<String> = corpora().into_iter().collect();
    assert_eq!(tested, files);
}

/// The population comes from the model list and from the front-end's own domain table, so a scene
/// or a regime cannot ship unfrozen. A case covers a domain only by NAMING it: a request that
/// leaves the domain out runs whatever the default is, which is not a claim about any one of them.
#[test]
fn every_model_and_every_domain_the_viewer_offers_is_frozen() {
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    let mut models: BTreeSet<String> = BTreeSet::new();
    for c in corpora() {
        for case in load(&c).values() {
            let req = &case["request"];
            let m = physsynth_viewer::model_of(req);
            // the room's select IS its wall termination, sent as `walls` (`gatherParams`)
            let key = if m == "airbox" { "walls" } else { "domain" };
            if let Some(d) = req.get(key).and_then(Value::as_str) {
                seen.insert((m.clone(), d.to_owned()));
            }
            // a refusal does not count as covering a scene
            if case["expect"].get("error").is_none() {
                models.insert(m);
            }
        }
    }
    let want: BTreeSet<String> = MODELS.iter().map(|(m, _)| (*m).to_owned()).collect();
    assert_eq!(
        want.difference(&models).collect::<Vec<_>>(),
        Vec::<&String>::new(),
        "models with no successful frozen case"
    );
    let app = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/static/app.js"),
    )
    .unwrap();
    let offered = domain_opts(&app);
    assert!(offered.len() >= 25, "{offered:?}");
    let missing: Vec<_> = offered.difference(&seen).collect();
    assert!(
        missing.is_empty(),
        "domains no frozen case names: {missing:?}"
    );
}

/// `DOMAIN_OPTS` in `app.js`, read as text: `model: [["value", "label"], ...]` entries up to the
/// closing `};`. Comment lines are skipped.
fn domain_opts(app: &str) -> BTreeSet<(String, String)> {
    let start = app
        .find("const DOMAIN_OPTS = {")
        .expect("DOMAIN_OPTS moved; this guard derives its population from it");
    let body = &app[start..];
    let body = &body[..body.find("\n};").unwrap()];
    let mut out = BTreeSet::new();
    let mut model = String::new();
    for line in body.lines().skip(1) {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        if let Some((head, _)) = t.split_once(": [") {
            if lower_key(head, false) {
                model = head.to_owned();
            }
        }
        let mut rest = t;
        while let Some(i) = rest.find("[\"") {
            rest = &rest[i + 2..];
            let end = rest.find('"').unwrap();
            out.insert((model.clone(), rest[..end].to_owned()));
            rest = &rest[end..];
        }
    }
    out
}

// -- the comparator bites -------------------------------------------------------------------------

fn cheap_case() -> (Value, Value, String) {
    let cases = load("strings");
    let (_, case) = cases
        .iter()
        .find(|(_, c)| c["expect"].get("error").is_none())
        .unwrap();
    let req = case["request"].clone();
    let model = physsynth_viewer::model_of(&req);
    (case["expect"].clone(), simulate_to_payload(&req), model)
}

fn hard(want: &Value, got: &Value, model: &str, exact: bool) -> usize {
    let mut v = Verdict {
        exact,
        ..Verdict::default()
    };
    cmp(want, got, "", model, &mut v);
    v.hard.len()
}

/// A change of STRUCTURE fails in both modes, on every platform.
#[test]
fn the_comparator_fails_each_structural_corruption_in_both_modes() {
    let (want, got, model) = cheap_case();
    assert_eq!(hard(&want, &got, &model, false), 0);
    for exact in [false, true] {
        // int -> float of the same value
        let mut g = got.clone();
        let n = g["frames"]["n_frames"].as_i64().unwrap();
        g["frames"]["n_frames"] = serde_json::json!(n as f64);
        assert_eq!(hard(&want, &g, &model, exact), 1);

        // an int moved by one
        let mut g = got.clone();
        g["frames"]["n_frames"] = serde_json::json!(n + 1);
        assert_eq!(hard(&want, &g, &model, exact), 1);

        // a key added
        let mut g = got.clone();
        g["meta"]["extra"] = Value::Bool(true);
        assert_eq!(hard(&want, &g, &model, exact), 1);

        // a buffer one float shorter
        let mut g = got.clone();
        let raw = decode_b64(g["frames"]["b64"].as_str().unwrap());
        g["frames"]["b64"] = Value::String(encode_b64(&raw[..raw.len() - 4]));
        assert_eq!(hard(&want, &g, &model, exact), 1);

        // a string changed
        let mut g = got.clone();
        g["model"] = Value::String("not-a-model".into());
        assert_eq!(hard(&want, &g, &model, exact), 1);
    }
}

/// A change of VALUE fails in exact mode. The baseline must be exact first, which is a claim about
/// the recording platform — elsewhere the payload itself may differ in the last bit, so this half
/// runs where [`EXACT`] holds (the Windows CI job) and nowhere else.
#[test]
fn the_comparator_fails_each_value_corruption_in_exact_mode() {
    if !EXACT {
        eprintln!("not the recording platform: the exact half runs in the Windows CI job");
        return;
    }
    let (want, got, model) = cheap_case();
    assert_eq!(hard(&want, &got, &model, true), 0);

    // one scalar, one ulp
    let mut g = got.clone();
    let x = g["fs_sim"].as_f64().unwrap();
    g["fs_sim"] = serde_json::json!(f64::from_bits(x.to_bits() + 1));
    assert_eq!(hard(&want, &g, &model, true), 1);
    assert_eq!(
        hard(&want, &g, &model, false),
        0,
        "structure mode skips float values"
    );

    // one byte of a buffer
    let mut g = got.clone();
    let s = g["frames"]["b64"].as_str().unwrap().to_owned();
    let flipped = if s.starts_with('A') { "B" } else { "A" };
    g["frames"]["b64"] = Value::String(format!("{flipped}{}", &s[1..]));
    assert_eq!(hard(&want, &g, &model, true), 1);

    // one sample of the audio, far outside its peak-scaled bar
    let mut g = got.clone();
    let raw = decode_b64(g["audio"]["b64"].as_str().unwrap());
    let mut xs: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let i = xs.len() / 2; // a sampled index (i * n / 64 at i = 32)
    xs[i] += 0.01;
    let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
    g["audio"]["b64"] = Value::String(encode_b64(&bytes));
    assert!(hard(&want, &g, &model, true) >= 1);
}

/// Standard base64 with padding — only for the corruption test above.
fn encode_b64(b: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for k in 0..4 {
            if k <= c.len() {
                s.push(A[(n >> (18 - 6 * k)) as usize & 63] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

/// The class table's matchers, pinned on the paths the reference's regexes were written for —
/// including the near misses that must stay exact.
#[test]
fn the_classes_match_what_the_references_regexes_matched() {
    let cases: &[(&str, &str, Option<Tol>)] = &[
        ("ideal", "audio.b64", Some(Tol::F32Peak)),
        ("ideal", "frames.b64", None),
        ("tension", "meta.spectrum.above.off", Some(Tol::Abs(1e-15))),
        (
            "tension",
            "meta.spectrum.below.floor[3]",
            Some(Tol::Abs(1e-15)),
        ),
        ("tension", "meta.spectrum.above.peak", None),
        (
            "tension",
            "meta.spectrum.cascade.grid_scale",
            Some(Tol::Abs(1e-15)),
        ),
        (
            "tension",
            "meta.spectrum.purity.off_mode",
            Some(Tol::Abs(1e-15)),
        ),
        (
            "tension",
            "meta.spectrum.sweep.points[4].floor",
            Some(Tol::Abs(1e-15)),
        ),
        ("tension", "meta.spectrum.sweep.points.floor", None),
        ("ideal", "meta.spectrum.mag[10]", Some(Tol::Abs(1e-12))),
        ("ideal", "meta.spectrum.zoom_mag", Some(Tol::Abs(1e-12))),
        (
            "ideal",
            "meta.spectrum.spectra.fret.mag[2]",
            Some(Tol::Abs(1e-12)),
        ),
        ("ideal", "meta.spectrum.spectra.fret.mag", None),
        (
            "bore",
            "meta.spectrum.spectrum.mag[7]",
            Some(Tol::Abs(1e-9)),
        ),
        ("bore", "meta.spectrum.spectrum.freq[7]", None),
        ("bore", "meta.dispersion.order[8]", Some(Tol::Any)),
        ("bore", "meta.dispersion.order[0]", None),
        (
            "ideal",
            "energy.lossy.measured_2sigma",
            Some(Tol::Rel(1e-12)),
        ),
        ("fret", "energy.decay_triple.rate", Some(Tol::Abs(1e-12))),
        (
            "fret",
            "energy.decay_triple.corrected",
            Some(Tol::Rel(1e-13)),
        ),
        ("vkroom", "energy.value[3]", Some(Tol::Rel(1e-13))),
        ("vk", "energy.value[3]", None),
        (
            "vkroom",
            "meta.ledger.radiated_frac[0]",
            Some(Tol::Rel(1e-13)),
        ),
        ("vkroom", "meta.ledger.radiated_frac", None),
        ("vkroom", "energy.lossless.drift", Some(Tol::Abs(1e-15))),
        ("ideal", "energy.lossless.drift", None),
        ("vkroom", "meta.ledger.residual_max", Some(Tol::Abs(1e-15))),
    ];
    for (model, path, want) in cases {
        assert_eq!(classify(model, path), *want, "{model} {path}");
    }
}
