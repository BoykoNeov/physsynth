//! The Rust half of the portability contract (`docs/dev/rust-migration-plan.md` §2.2).
//!
//! `tests/test_stability.py::test_core_dependency_allowlist` used to be the only thing asserting
//! that the DSP core pulls in nothing but the declared numeric stack. It cannot see inside a
//! compiled extension module — to `sys.modules`, `physsynth_rs` is one opaque third-party binary
//! whose own dependency tree is invisible. So the rule migrates here, where the tree is visible,
//! and keeps its shape: a **hardcoded** allowlist, so that adding a dependency is a reviewed edit
//! in two places rather than a silent one in `Cargo.toml`.
//!
//! # What is checked, and what is deliberately not
//!
//! Only `physsynth-core`'s own transitive **normal** and **build** dependencies. Not the
//! workspace's — the workspace also contains `physsynth-py`, which pulls pyo3, numpy, syn, libc
//! and thirty more, so a workspace-scoped check would pass while proving nothing at all. And not
//! dev-dependencies: `serde_json` is here to parse `cargo metadata` for this very test, which is
//! the Rust equivalent of pytest being installed. The exclusion is *self-testing* — the last
//! assertion below fails if `serde_json` ever shows up in the walked set, which is what would
//! happen if the edge-kind filter broke.

use std::collections::{BTreeSet, HashMap};
use std::process::Command;

/// Crates `physsynth-core` may depend on, transitively, at build or run time.
///
/// Empty, and that is the intended state today: Phase 0's physics is `f64` arithmetic over slices,
/// which needs nothing. §2.2 permits "the numeric stack and nothing else" — no I/O, no logging, no
/// plugin framework — but seeding this list with crates nobody has actually reviewed would
/// pre-approve them. The first numeric dependency the core takes (`ndarray`, a LAPACK binding, a
/// sparse solver — the plan's §4 expects all three eventually) arrives as a deliberate edit here,
/// in the same commit that adds it to `Cargo.toml`, with the reason written down.
const ALLOWED: &[&str] = &[];

/// Categories that must never appear, whatever the allowlist says.
///
/// A second, blunter net under the first. The allowlist catches everything by construction, so
/// this exists for the case where someone edits the allowlist without thinking — the names below
/// are the ones §2.2 calls out by category, and seeing one of them turned on should read as an
/// argument to be had rather than a line to be added.
const NEVER: &[&str] = &[
    "tokio",
    "async-std",
    "reqwest",
    "hyper",
    "log",
    "tracing",
    "env_logger",
    "clap",
    "serde",
    "nih_plug",
    "cpal",
    "rodio",
    "hound",
    "eframe",
    "egui",
    "winit",
    "wgpu",
];

/// One package in the `cargo metadata` resolve graph.
struct Node {
    name: String,
    /// Package ids this node depends on via a non-dev edge.
    deps: Vec<String>,
}

/// Run `cargo metadata` for THIS crate and return (root id, id -> node).
fn resolve_graph() -> (String, HashMap<String, Node>) {
    let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let out = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            manifest,
        ])
        .output()
        .expect("`cargo metadata` must be runnable — it is how this rule is checked at all");
    assert!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let meta: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("cargo metadata emits JSON");
    let resolve = &meta["resolve"];

    let mut graph = HashMap::new();
    for node in resolve["nodes"]
        .as_array()
        .expect("resolve.nodes is a list")
    {
        let id = node["id"]
            .as_str()
            .expect("every node has an id")
            .to_owned();
        let name = meta["packages"]
            .as_array()
            .expect("packages is a list")
            .iter()
            .find(|p| p["id"].as_str() == Some(&id))
            .and_then(|p| p["name"].as_str())
            .unwrap_or("<unknown>")
            .to_owned();

        // `deps` (not `dependencies`) carries `dep_kinds`, which is what distinguishes a dev edge
        // from a normal one. A null `kind` means normal; "build" is a build-script dependency and
        // still ships influence into the artifact; "dev" is test-only and is skipped.
        let mut deps = Vec::new();
        for dep in node["deps"].as_array().into_iter().flatten() {
            let keep = dep["dep_kinds"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|k| matches!(k["kind"].as_str(), None | Some("build")));
            if keep {
                if let Some(pkg) = dep["pkg"].as_str() {
                    deps.push(pkg.to_owned());
                }
            }
        }
        graph.insert(id, Node { name, deps });
    }

    let root = resolve["root"]
        .as_str()
        .expect("a single-crate metadata query has a root")
        .to_owned();
    (root, graph)
}

/// Every crate reachable from `physsynth-core` by normal/build edges, excluding itself.
fn shipped_dependencies() -> BTreeSet<String> {
    let (root, graph) = resolve_graph();
    let mut seen = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut stack = vec![root.clone()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let node = match graph.get(&id) {
            Some(n) => n,
            None => continue,
        };
        if id != root {
            names.insert(node.name.clone());
        }
        stack.extend(node.deps.iter().cloned());
    }
    names
}

#[test]
fn core_depends_only_on_the_allowlist() {
    let allowed: BTreeSet<&str> = ALLOWED.iter().copied().collect();
    let shipped = shipped_dependencies();
    let leaked: Vec<&String> = shipped
        .iter()
        .filter(|n| !allowed.contains(n.as_str()))
        .collect();
    assert!(
        leaked.is_empty(),
        "physsynth-core pulled crate(s) outside the allowlist {allowed:?}: {leaked:?}\n\
         If one of these belongs, add it to ALLOWED in this file in the same commit that adds it \
         to Cargo.toml, with the reason. That two-step is the rule, not friction."
    );
}

#[test]
fn core_pulls_in_nothing_from_the_forbidden_categories() {
    let shipped = shipped_dependencies();
    for name in NEVER {
        assert!(
            !shipped.contains(*name),
            "physsynth-core depends on `{name}` — §2.2 forbids I/O, logging, async runtimes, \
             audio backends, GUI toolkits and plugin frameworks in the headless core"
        );
    }
}

#[test]
fn dev_dependencies_are_excluded_from_the_rule() {
    // Not a tautology: this is the only thing standing between the edge-kind filter above and a
    // vacuous pass. `serde_json` IS a dependency of this crate — a dev one — and if the filter
    // stopped distinguishing kinds it would appear in the walked set and the first test would go
    // red for the wrong reason. Assert the distinction directly, so the failure names itself.
    let shipped = shipped_dependencies();
    assert!(
        !shipped.contains("serde_json"),
        "the dev-dependency serde_json leaked into the shipped set — the dep_kinds filter in \
         resolve_graph() is broken, and the allowlist test above is now checking the wrong graph"
    );
}
