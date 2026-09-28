//! The dependency rule for the viewer crate — the third copy, and the first non-empty list.
//!
//! Same shape as `physsynth-core/tests/deps.rs` and `physsynth-analysis/tests/deps.rs`, for the
//! reason the analysis copy gives: `cargo metadata` is rooted at the package the test lives in, and
//! the rule's point is that a dependency is a reviewed edit *in the package that takes it*.
//!
//! # Why this list is not empty, and why it is this short
//!
//! The viewer is the one crate here whose job is I/O: it binds a socket, reads the front-end off
//! disk and writes JSON. The socket and the files are `std`. JSON is the one thing taken from
//! outside, and `serde_json` is the name retirement plan §5 recommended (it was already a
//! dev-dependency of both physics crates, so it is not new to the tree). Everything else below is
//! what `serde_json` itself pulls: its integer and float formatters (`itoa`, `zmij`), the
//! derive-free half of serde (`serde_core`) and `memchr`.
//!
//! No HTTP crate. The plan named `tiny_http` as the obvious candidate and said to measure at the
//! batch. Measured 2026-09-28, `tiny_http 0.12` pulls `ascii`, `chunked_transfer`, `httpdate` and
//! `log` — the last on the core's NEVER list by category — for a server that handles one request
//! per connection on localhost. `std::net` plus a thread per connection is what the Python
//! `ThreadingHTTPServer` it replaces was.

use std::collections::{BTreeSet, HashMap};
use std::process::Command;

/// Crates `physsynth-viewer` may depend on, transitively, at build or run time.
const ALLOWED: &[&str] = &[
    // This workspace's physics — the viewer is their client.
    "physsynth-core",
    "physsynth-analysis",
    // JSON, and what it pulls.
    "serde_json",
    "serde_core",
    "itoa",
    "zmij",
    "memchr",
];

/// Names that must never appear, whatever the allowlist says.
///
/// The binding is here by name: the viewer is what makes the binding deletable, so it must never
/// come to depend on it. The rest are what a server is tempted by — an async runtime, an HTTP
/// framework, a logging facade.
const NEVER: &[&str] = &[
    "physsynth-py",
    "pyo3",
    "numpy",
    "tokio",
    "async-std",
    "hyper",
    "reqwest",
    "axum",
    "actix-web",
    "tiny_http",
    "log",
    "tracing",
    "env_logger",
];

/// A platform condition that holds on no platform, so an edge carrying it is never compiled.
///
/// The one place this copy of the walker departs from the other two, and it had to: `serde_json`
/// declares `serde` under `[target.'cfg(any())'.dependencies]` purely to constrain which `serde`
/// version may coexist with it. `cargo metadata`'s resolve lists that edge with the cfg as data
/// rather than evaluating it, so the first run of this test reported `serde`, `serde_derive`,
/// `syn`, `quote`, `proc-macro2` and `unicode-ident` as shipped when `cargo tree` (which does
/// evaluate) shows none of them. Only the literal never-true cfg is skipped — a real platform cfg
/// is still walked, which over-reports on the safe side.
const NEVER_TRUE_CFG: &str = "cfg(any())";

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
            let keep = dep["dep_kinds"].as_array().into_iter().flatten().any(|k| {
                matches!(k["kind"].as_str(), None | Some("build"))
                    && k["target"].as_str() != Some(NEVER_TRUE_CFG)
            });
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

/// Every crate reachable from `physsynth-analysis` by normal/build edges, excluding itself.
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
fn viewer_depends_only_on_the_allowlist() {
    let allowed: BTreeSet<&str> = ALLOWED.iter().copied().collect();
    let shipped = shipped_dependencies();
    let leaked: Vec<&String> = shipped
        .iter()
        .filter(|n| !allowed.contains(n.as_str()))
        .collect();
    assert!(
        leaked.is_empty(),
        "physsynth-viewer pulled crate(s) outside the allowlist {allowed:?}: {leaked:?}
         If one of these belongs, add it to ALLOWED in this file in the same commit that adds it          to Cargo.toml, with the reason. That two-step is the rule, not friction."
    );
}

#[test]
fn viewer_pulls_in_nothing_from_the_forbidden_names() {
    let shipped = shipped_dependencies();
    for name in NEVER {
        assert!(
            !shipped.contains(*name),
            "physsynth-viewer depends on `{name}` — see NEVER in this file for why it is refused"
        );
    }
}

#[test]
fn the_allowlist_is_not_stale() {
    // A name allowed but no longer pulled is a pre-approval nobody reviewed for its next use.
    let shipped = shipped_dependencies();
    for name in ALLOWED {
        assert!(
            shipped.contains(*name),
            "`{name}` is allowed but no longer a dependency — remove it from ALLOWED"
        );
    }
}
