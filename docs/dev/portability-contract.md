# Portability Contract — `physsynth/core`

> What the DSP core may and may not depend on, why, and how it's enforced.
> This is the rule that keeps the eventual systems-language port cheap — and the answer to
> "discipline drift." Treat a violation as a bug, not a style nit.

> **Superseded in part, 2026-08-26.** Python is being retired for Rust — see
> `docs/dev/rust-migration-plan.md`. Two clauses below no longer hold: Python does **not** stay the
> permanent reference oracle (it is retired model by model, oracle last), and the "check the new
> kernel to ~1e-15" bar is **not achievable** across languages — the acceptance bar is the physics
> harness (energy drift < 1e-10, partials within a cent, convergence order ≈ 2). Everything else
> here — the one-way dependency arrow, the no-I/O rule, the hot-path style — stands, and is what
> makes the migration mechanical rather than a redesign.

## Why this exists

Non-negotiables #3 (prototype in Python, port later) and #4 (headless core) commit us to a plan:
**research the physics in Python, then port the hot kernel to a systems language at the plugin
stage.** That port is mechanical and safe *only if* `core/` stays portable. It stops being portable
through small, innocent leaks — a plotting import "just for debugging," a global cache, a config
read — i.e. **discipline drift**.

Drift is a code-organization problem, not a language one: a messy Rust/Julia core drifts just as
easily. So the fix is not a different language — it is an **enforced boundary**, defined here and
checked by tests. That is also why we are staying in Python now rather than switching: the switch
would not have fixed the thing we were worried about.

## The dependency rule — one direction only

```
        exciter ──▶ resonator (± nonlinear coupling) ──▶ body / radiation
                          │  (this is core/)
                          ▼
        analysis · viz · io · scripts · tests   ── depend on core, NEVER the reverse
```

`core/` is a leaf. Everything else may import it; it imports nothing of ours except itself.

## `core/` MAY use

- **NumPy and SciPy** — they map cleanly to the target (BLAS/LAPACK, Eigen, `ndarray`, sparse
  solvers). This is the numeric vocabulary the port will mirror.
- **Stdlib that is a struct / typing convenience:** `dataclasses`, `typing`, `enum`, `math`,
  `__future__`. These erase or translate trivially.
- **Relative imports within `core/`** (`from .operators import ...`).

## `core/` MUST NOT use

- **Plotting** (`matplotlib`, `plotly`), **audio I/O** (`sounddevice`, `pyaudio`, `soundfile`),
  **GUI** (`PyQt*`, `PySide*`, `pygame`, `tkinter`). These belong in `viz/` and future wrappers.
- **File / network / disk / environment I/O, or logging to disk.** A resonator is a *pure function*
  of its constructor arguments and its state — it reads no config, opens no file, touches no clock.
- **Global mutable state or module-level singletons.** All state lives on the resonator instance
  (`self.u`, `self.u_prev`, …), so a C++/Rust struct can hold it 1:1.
- **Heavy frameworks as hard dependencies** (`torch`, `jax`, `pandas`). Numba/JAX may *later*
  accelerate a specific kernel behind the same interface — that is an optimization to flag for
  review, never a baseline import of `core/`.
- **Dynamic / reflective constructs that don't port:** `eval`/`exec`, monkeypatching, metaclass
  magic, and Python-level per-element loops or exotic broadcasting in the hot path.

## Hot-path style (so the port is transcription, not redesign)

- The per-step kernel is **vectorized array arithmetic** — no Python per-element loop in `step()`.
- **State is explicit** on the object and passed in/out; no hidden globals.
- **Deterministic:** same inputs → same outputs. Cross-language agreement to ~1e-15 is one of our
  strongest correctness checks, and it only works if the Python side is reproducible.

## Enforcement

The Python guards (`tests/test_stability.py`) are gone: retirement plan §50 deleted the file when
the Python package was down to re-export shims. What enforces the contract now is in the crates,
where the dependency tree is visible rather than hidden inside one compiled extension module.

| Guard | Holds |
|-------|-------|
| `crates/physsynth-core/tests/deps.rs` | **The core depends on nothing.** Its transitive normal and build dependencies, read from `cargo metadata`, must sit inside a hardcoded allowlist that is empty today — adding a crate is a reviewed edit here in the same commit as `Cargo.toml`. A second list names the forbidden categories outright (async runtimes, HTTP, logging, CLI, serde, audio backends, GUI toolkits, plugin frameworks). A self-test proves dev-dependencies are excluded, so the walk is the right graph. |
| `crates/physsynth-analysis/tests/deps.rs` | The same rule, also with an empty list, for the measuring instrument: the core's guard is scoped to its own package, so a new crate inherits none of its enforcement. |
| `crates/physsynth-viewer/tests/deps.rs` | The viewer's allowlist — the two physics crates, `serde_json` and what it pulls, nothing else — for the one crate whose job is I/O. Its forbidden list names the binding: the viewer is what made the binding deletable, so it must never come to depend on it. |

**What the crate rule does not see.** It is a rule about *dependencies*. The core's own source
could still use the standard library's file, network or console calls (`std::fs`, `std::net`,
`println!`) and nothing would fail. None appears today (checked at §50); a guard for it was
offered and declined (the human's call, 2026-10-10), so it is a known gap, not an oversight.

## When we *do* port

The validation harness **is** the contract for the port: the ported kernel is correct iff it
reproduces the same numbers — lossless energy drift < 1e-10 (in practice ~1e-15), partials within
~1 cent, convergence order ≈ 2. Keep the Python implementation as the **reference oracle** and check
the new kernel against it to ~1e-15. The systems-language choice itself — **C++/JUCE** (mature
plugin path) vs **Rust/`nih-plug`** (safer, no-GC, smaller ecosystem) — is a *plugin-stage*
decision, deliberately not made now.
