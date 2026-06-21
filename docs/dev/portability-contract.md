# Portability Contract — `physsynth/core`

> What the DSP core may and may not depend on, why, and how it's enforced.
> This is the rule that keeps the eventual systems-language port cheap — and the answer to
> "discipline drift." Treat a violation as a bug, not a style nit.

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

## Enforcement (`tests/test_stability.py`)

| Test | Guards |
|------|--------|
| `test_core_is_headless` | Blocklist of common offenders (matplotlib, audio, GUI) — named, with a clear failure message. |
| `test_core_dependency_allowlist` | **No deps beyond the numeric stack:** importing every `core/` submodule must add *no* third-party package beyond `numpy`/`scipy` **and their transitive closure** (the baseline is captured by importing the numeric stack first, so scipy's own unavoidable baggage — `charset_normalizer`, `cython_runtime`, the hash-suffixed mypyc runtime — is permitted, while a real leak like `torch`/`requests`/`PIL` is not). Auto-discovers new submodules, so it catches *any* future leak. |
| `test_core_does_not_import_sibling_layers` | `core/` must not import `physsynth.viz` / `analysis` / `io` — enforces the one-way dependency arrow. |

## When we *do* port

The validation harness **is** the contract for the port: the ported kernel is correct iff it
reproduces the same numbers — lossless energy drift < 1e-10 (in practice ~1e-15), partials within
~1 cent, convergence order ≈ 2. Keep the Python implementation as the **reference oracle** and check
the new kernel against it to ~1e-15. The systems-language choice itself — **C++/JUCE** (mature
plugin path) vs **Rust/`nih-plug`** (safer, no-GC, smaller ecosystem) — is a *plugin-stage*
decision, deliberately not made now.
