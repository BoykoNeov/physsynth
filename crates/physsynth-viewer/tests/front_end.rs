//! Agreements between the front-end (`web/static/`) and the headless browser check
//! (`examples/verify_headless.rs`) that no payload test can see.
//!
//! Both populations are DERIVED from the files rather than written down here: a literal third copy
//! would be one more thing to drift.

use physsynth_viewer::MODELS;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn read(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Every `"…"` literal between `start` and the next `end` in `text`.
fn quoted_after(text: &str, start: &str, end: &str) -> Vec<String> {
    let i = text
        .find(start)
        .unwrap_or_else(|| panic!("`{start}` moved; this guard derives its population from it"));
    let body = &text[i + start.len()..];
    let body = &body[..body.find(end).unwrap()];
    body.split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect()
}

/// The panel emits a decision word and the headless check grades it. The vocabulary lives in two
/// files, so renaming a branch in one would leave the other grading a word that is never emitted —
/// a check that passes because it is looking at nothing.
#[test]
fn the_canvas_mark_vocabulary_is_the_same_word_in_both_files() {
    let app = read("../../web/static/app.js");
    let harness = read("examples/verify_headless.rs");

    let emitted: BTreeSet<String> = app
        .split("markHorizon(\"")
        .skip(1)
        .filter_map(|s| s.split_once('"').map(|(w, _)| w.to_owned()))
        .filter(|w| w.bytes().all(|b| b.is_ascii_lowercase() || b == b'-'))
        .collect();
    assert_eq!(
        emitted.len(),
        5,
        "one per branch; a lost branch is a silent hole: {emitted:?}"
    );
    let graded: BTreeSet<String> =
        quoted_after(&harness, "const HORIZON_MARK_REASONS: &[&str] =", "];")
            .into_iter()
            .collect();
    assert_eq!(graded, emitted);

    // The two arms the panel actually PAINTS, as opposed to reporting in text.
    assert!(emitted.contains("drawn") && emitted.contains("zero-modes"));
    assert!(app.contains(r#"reason === "drawn" || reason === "zero-modes""#));
}

/// The check loads every scene the viewer serves at least once, so a model added to the viewer
/// cannot go unrendered in a real browser.
#[test]
fn the_headless_check_renders_every_model() {
    let harness = read("examples/verify_headless.rs");
    let cases = quoted_after(&harness, "const CASES: &[(&str, &str)] = &[", "\n];");
    let rendered: BTreeSet<String> = cases
        .iter()
        .filter_map(|q| q.strip_prefix("model="))
        .map(|q| q.split('&').next().unwrap().to_owned())
        .collect();
    let served: BTreeSet<String> = MODELS.iter().map(|(m, _)| (*m).to_owned()).collect();
    assert_eq!(rendered, served);
}
