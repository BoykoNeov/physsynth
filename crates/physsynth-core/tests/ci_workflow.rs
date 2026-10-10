//! The workflow file's own guard: shell continuations, and the test targets the steps name.
//!
//! Carried from `tests/test_ci_workflow.py` at retirement plan §50, because the workflow outlives
//! Python and the defects it catches are made by editing the workflow, which the last deletion
//! does more of than any step before it.
//!
//! `docs/dev/rust-migration-plan.md` §19.7 found a CI step that had been failing since the previous
//! batch for a reason nothing in the repo could see: a `run:` block whose line continuation had been
//! written as the two characters backslash-`n` instead of a backslash and a newline. The shell then
//! received that pair as an argument, `pytest` reported "file or directory not found", and the job
//! went red, loudly, for a reason that read like a missing test file rather than a typo in YAML.
//!
//! Two things make that shape recur rather than being a one-off. It is invisible to a YAML parser,
//! because the sequence is a valid pair of characters inside a block scalar; and it is introduced by
//! *tooling* rather than by typing: any editor, script or patch that passes a string through one
//! round of escaping too few produces exactly it. The same bug was reintroduced while §20's step was
//! being added.
//!
//! **No YAML parser is used, deliberately, and not only to avoid a dependency.** After parsing, a
//! literal backslash-`n` and a real newline are both just characters in a string, and the
//! distinction this file exists for is gone. The raw text is where the two are still different.
//!
//! Each check is a function over the text, and each runs twice: on the real workflow, and on
//! planted text where it must FIRE. The planted half is permanent proof that a check can fail; a
//! guard that passes quietly over nothing is the failure it exists to catch, one level down.
//!
//! Nothing here is about the *content* of a step. That belongs to the steps themselves, which are
//! claims about a batch and are meant to be edited. This file asserts only what no reader would
//! check by eye and no other tool checks at all.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn workflow() -> String {
    let path = repo_root().join(".github").join("workflows").join("ci.yml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read the workflow at {}: {e}", path.display()))
}

/// The two characters backslash and `n`, built from the backslash's code point so that no round
/// of escaping between an editor and this file can turn the needle into the thing it looks for.
fn backslash_n() -> String {
    [char::from(92u8), 'n'].iter().collect()
}

/// Line numbers (1-based) holding a literal backslash-`n`, anywhere in the file.
fn literal_backslash_n_lines(text: &str) -> Vec<usize> {
    let needle = backslash_n();
    text.lines()
        .enumerate()
        .filter(|(_, line)| line.contains(&needle))
        .map(|(i, _)| i + 1)
        .collect()
}

/// One line of shell that a `run:` key hands to the runner.
struct RunLine<'a> {
    number: usize,
    /// The whole line as written, indentation included: what the length check measures.
    raw: &'a str,
    /// The shell text: the line itself in a block, the value after `run: ` on a one-line run.
    shell: &'a str,
    /// Inside a `run: |` block, where a continuation can be swallowed. A one-line `run:` has no
    /// continuation to lose.
    in_block: bool,
}

/// Every line of shell the workflow runs, in order.
///
/// A block starts at a line ending in `run: |` or `run: |-` and ends at the first non-empty line
/// indented no deeper than the key. A one-line run is `run: <command>`, with or without the
/// list item's `- ` in front.
fn run_lines(text: &str) -> Vec<RunLine<'_>> {
    let mut out = Vec::new();
    let (mut inside, mut block_indent) = (false, 0);
    for (i, line) in text.lines().enumerate() {
        let stripped = line.trim();
        let indent = line.len() - line.trim_start().len();
        if inside && !stripped.is_empty() && indent <= block_indent {
            inside = false;
        }
        if stripped.ends_with("run: |") || stripped.ends_with("run: |-") {
            inside = true;
            block_indent = indent;
            continue;
        }
        if inside {
            out.push(RunLine {
                number: i + 1,
                raw: line,
                shell: stripped,
                in_block: true,
            });
            continue;
        }
        let key = stripped.strip_prefix("- ").unwrap_or(stripped);
        if let Some(command) = key.strip_prefix("run: ") {
            out.push(RunLine {
                number: i + 1,
                raw: line,
                shell: command.trim(),
                in_block: false,
            });
        }
    }
    out
}

/// The same escaping failure with the opposite sign, found in §24 and pre-existing on three steps.
///
/// §19.7's variant leaves a literal backslash-`n` behind. The variant found while adding the beam's
/// step *swallows* the continuation instead: the same tooling round-trip that turns `\` + newline
/// into two visible characters can also consume both, and what is left is two shell lines JOINED
/// into one. It is harmless while what follows a continuation is another argument; the moment it
/// sits between two commands (`pip install ...` and `pytest ...`) the second becomes an argument of
/// the first and the step silently stops doing half its job. That is invisible to the
/// backslash-`n` check (there is no backslash left), to a YAML parser and to the eye.
///
/// A length limit tells the two shapes apart without asserting anything about content. Measured at
/// §50: the longest `run:` block line is 120 characters (`rust-debug`'s first `release_only=`
/// line), and the joined lines §24 found were 300 to 850. The Python's limit was 120 with `>`,
/// so that line sat exactly on it and the next test name added there would have tripped it. 160
/// leaves the legitimate lines room and is still about half the shortest join found. A join of two
/// SHORT lines is under any limit; this catches the shape that was actually found, not every join.
const RUN_LINE_LIMIT: usize = 160;

/// `(line number, length)` of every `run:` block line over the limit.
fn over_long_run_lines(text: &str, limit: usize) -> Vec<(usize, usize)> {
    run_lines(text)
        .into_iter()
        .filter(|r| r.in_block && r.raw.chars().count() > limit)
        .map(|r| (r.number, r.raw.chars().count()))
        .collect()
}

/// What the run lines name: repository paths, and `cargo test -p <crate> --test <file>` targets.
#[derive(Debug, Default, PartialEq)]
struct Named {
    paths: Vec<String>,
    /// `(package, test)`. The package is `None` when the line gave `--test` with no `-p`.
    targets: Vec<(Option<String>, String)>,
}

/// Top-level directories whose names, as a token's prefix, mark it as a path into this repository.
const REPO_DIRS: &[&str] = &["crates/", "tests/", "web/", "docs/", "scripts/", ".github/"];

/// The tokens of every run line that are names to check.
///
/// Shell comment lines are skipped: they are prose, and the workflow's history comments name
/// files that were deleted on purpose. A token holding `$` or `*` is a *query*, not a name (the
/// `rust-debug` step builds `crates/physsynth-core/tests/$n.rs` and globs `tests/*.rs` so its list
/// is derived at job time), and asking whether it exists is a category error.
fn named_in_runs(text: &str) -> Named {
    let mut named = Named::default();
    for run in run_lines(text) {
        if run.shell.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = run
            .shell
            .split_whitespace()
            .map(|t| t.trim_matches(|c| matches!(c, '"' | '\'' | ';' | ')' | '(')))
            .collect();
        let mut package: Option<String> = None;
        for (k, token) in tokens.iter().enumerate() {
            if token.contains('$') || token.contains('*') {
                continue;
            }
            // `targets="$targets --test $n"` names no target: the query rule covers the operand.
            let next = tokens
                .get(k + 1)
                .copied()
                .filter(|t| !t.contains('$') && !t.contains('*'));
            match *token {
                "-p" | "--package" => {
                    if let Some(p) = next {
                        package = Some(p.to_owned());
                    }
                }
                "--test" => {
                    if let Some(t) = next {
                        named.targets.push((package.clone(), t.to_owned()));
                    }
                }
                _ => {
                    let bare = token.strip_prefix("./").unwrap_or(token);
                    if REPO_DIRS.iter().any(|d| bare.starts_with(d)) {
                        named.paths.push(bare.to_owned());
                    }
                }
            }
        }
    }
    named
}

/// Everything `named` names that is not in the repository at `root`, as readable descriptions.
fn missing(named: &Named, root: &Path) -> Vec<String> {
    let crates = root.join("crates");
    let mut out = Vec::new();
    for p in &named.paths {
        if !root.join(p).exists() {
            out.push(format!("path `{p}`"));
        }
    }
    for (package, test) in &named.targets {
        let file = format!("{test}.rs");
        let found = match package {
            Some(p) => crates.join(p).join("tests").join(&file).is_file(),
            None => std::fs::read_dir(&crates)
                .map(|dir| {
                    dir.flatten()
                        .any(|c| c.path().join("tests").join(&file).is_file())
                })
                .unwrap_or(false),
        };
        if !found {
            let place = package.as_deref().unwrap_or("any crate");
            out.push(format!("test target `{test}` in {place}"));
        }
    }
    out
}

// -- the real workflow ---------------------------------------------------------------------------

#[test]
fn no_line_contains_a_literal_backslash_n() {
    let offenders = literal_backslash_n_lines(&workflow());
    assert!(
        offenders.is_empty(),
        "a literal backslash-n in the workflow at line(s) {offenders:?} -- a line continuation \
         that lost a round of escaping, so the shell will receive it as an argument"
    );
}

#[test]
fn no_run_block_line_is_a_swallowed_continuation() {
    let text = workflow();
    // POSITIVE CONTROL. The Python had none: if the block detection broke, the scan saw no lines
    // and passed. `rust-debug` runs this command inside a `run: |` block, so a scanner that still
    // works must have seen it there.
    assert!(
        run_lines(&text)
            .iter()
            .any(|r| r.in_block && r.shell == "cargo test -p physsynth-analysis --tests"),
        "the run-block scanner did not see `cargo test -p physsynth-analysis --tests` inside a \
         block, so it has stopped finding blocks and this guard is checking nothing"
    );
    let offenders = over_long_run_lines(&text, RUN_LINE_LIMIT);
    assert!(
        offenders.is_empty(),
        "`run:` line(s) over {RUN_LINE_LIMIT} characters, as (line, length): {offenders:?} -- \
         almost certainly two shell lines joined by a continuation that lost its backslash AND its \
         newline"
    );
}

/// The general form of §19.7's failure, and the one that catches a rename too.
///
/// The Python asked only about `tests/...py` tokens, and after phase F the workflow names none:
/// its last one, `tests/test_binding_surface.py`, goes with the binding. So the population is
/// widened (the human's call, retirement plan §50) to every repository path and every
/// `cargo test --test` target a run line names. A count on this population would drain; a NAMED
/// control does not: the exact viewer freeze is the `frozen-windows` job's whole purpose and runs
/// as `-p physsynth-viewer --test frozen`, so a scan that still works must find that pair.
#[test]
fn every_path_and_test_target_the_workflow_names_exists() {
    let named = named_in_runs(&workflow());
    let control = (Some("physsynth-viewer".to_owned()), "frozen".to_owned());
    assert!(
        named.targets.contains(&control),
        "the scan did not find `-p physsynth-viewer --test frozen`, which the `frozen-windows` job \
         runs -- so the scan has stopped matching and this guard is checking nothing. It found: \
         {named:?}"
    );
    let gone = missing(&named, &repo_root());
    assert!(
        gone.is_empty(),
        "the workflow names things that are not in the repository: {gone:?}"
    );
}

// -- planted text: each check must fire ----------------------------------------------------------

/// A two-step job in the workflow's own shape, with `extra` appended to the first `run:` block.
fn planted(extra: &str) -> String {
    format!(
        "jobs:\n  a:\n    steps:\n      - name: one\n        run: |\n          echo first\n\
         {extra}\
         \n      - name: two\n        run: cargo test -p physsynth-core --test deps\n"
    )
}

#[test]
fn a_planted_backslash_n_is_found() {
    let line = format!("          pytest a.py {}          b.py\n", backslash_n());
    let text = planted(&line);
    assert_eq!(literal_backslash_n_lines(&text), vec![7]);
    assert!(literal_backslash_n_lines(&planted("")).is_empty());
}

#[test]
fn a_planted_joined_line_is_found_only_inside_a_run_block() {
    // A join is a long line INSIDE a block. The same length elsewhere (a comment, a matrix
    // expression) is not a shell line and must not be flagged, and the block ends at the dedent.
    let joined = format!("          pip install x {}\n", "y ".repeat(150));
    let text = planted(&joined);
    let length = joined.trim_end_matches('\n').len();
    assert!(length > RUN_LINE_LIMIT);
    assert_eq!(
        over_long_run_lines(&text, RUN_LINE_LIMIT),
        vec![(7, length)]
    );

    let outside = format!("# {}\n{}", "z".repeat(300), planted(""));
    assert!(over_long_run_lines(&outside, RUN_LINE_LIMIT).is_empty());

    // A one-line `run:` cannot swallow a continuation, so its length is not this check's business.
    let one_line = planted("").replace("--test deps", &"--test deps ".repeat(20));
    assert!(over_long_run_lines(&one_line, RUN_LINE_LIMIT).is_empty());
}

#[test]
fn the_scan_reads_blocks_and_one_line_runs_and_skips_queries_and_comments() {
    let text = planted(
        "          # tests/deleted_on_purpose.py is history\n\
         \x20         ls crates/physsynth-core/tests/$n.rs tests/*.py\n\
         \x20         targets=\"$targets --test $n\"\n\
         \x20         pytest ./tests/gone.py \"web/static\"\n\
         \x20         cargo test --release -p physsynth-viewer --test frozen\n",
    );
    let named = named_in_runs(&text);
    assert_eq!(
        named,
        Named {
            paths: vec!["tests/gone.py".to_owned(), "web/static".to_owned()],
            targets: vec![
                (Some("physsynth-viewer".to_owned()), "frozen".to_owned()),
                (Some("physsynth-core".to_owned()), "deps".to_owned()),
            ],
        }
    );
    // Against the real repository: the two real names are found, the two missing ones are not.
    assert_eq!(missing(&named, &repo_root()), vec!["path `tests/gone.py`"]);
}

#[test]
fn a_missing_test_target_is_reported_with_and_without_a_package() {
    let named = Named {
        paths: vec![],
        targets: vec![
            (
                Some("physsynth-viewer".to_owned()),
                "no_such_test".to_owned(),
            ),
            (Some("physsynth-core".to_owned()), "frozen".to_owned()),
            (None, "frozen".to_owned()),
            (None, "no_such_test".to_owned()),
        ],
    };
    // `frozen` is the viewer's, not core's: a wrong package is a missing target.
    assert_eq!(
        missing(&named, &repo_root()),
        vec![
            "test target `no_such_test` in physsynth-viewer",
            "test target `frozen` in physsynth-core",
            "test target `no_such_test` in any crate",
        ]
    );
}
