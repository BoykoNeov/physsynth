---
name: rust-phase0-state
description: "Phase 0 of the Rust migration is BUILT (2026-08-26) — the binding's state buffers must be Python-owned numpy arrays (a Rust Vec view is a use-after-free that reads plausibly), the state comes out BIT-IDENTICAL, and step 5 of the ritual can never fire on schedule"
metadata: 
  node_type: memory
  type: project
  originSessionId: cd7b3b41-e492-4138-9e69-06b9fd5f5668
  modified: 2026-08-26T12:32:04.286Z
---

**Phase 0 landed 2026-08-26**: `crates/physsynth-core` (zero dependencies) + `crates/physsynth-py`
(PyO3, exposed as `physsynth_rs`), `string_ideal` ported, CI job added. Plan §9 records it.
Context: [[rust-migration-state]].

**The result was better than the plan budgeted for: the WHOLE suite is green under the flag —
1,954 passed, 0 failed**, not just the string's 38. The plan expected a failure list to record as
the binding's real surface spec; there wasn't one. `connection.py`'s private-name reach, the
sympathetic/collision/plate/room couplings and all 403 web tests all pass. So §3.1's "~255 call
sites need a designed binding" is *answered* for this model, not deferred.

**But "green under the flag" is only worth the swap actually happening**, and nothing in the 38
tests mentions Rust — so a mistyped variable or a refactored-away swap block would run **Python**
and pass. `test_the_rust_swap_matches_the_environment` in `test_stability.py` now asserts both
directions off the env var (set → the class *is* `physsynth_rs.IdealString`; unset → it is the
Python one). Same reason the parity CI step runs a bare `import physsynth_rs` first: that file
opens with `importorskip`, so a failed install would skip it and exit 0 having asserted nothing.

**The model splits three ways and Phase 2 should keep the split**: a validated immutable `Params`
(all derivation + every rejection), **kernels as free functions over `&[f64]`** (no state), and a
native owning struct with `Vec` state for `cargo test`. The binding does **not** wrap the owning
struct — it calls the kernels, because its buffers must be something else. ↓

**THE finding — who owns the buffers.** Python's `step()` *rebinds* `self.u`; it does not overwrite
it. So a reference held across a step is a valid **snapshot**, and a write *through* `.u` reaches
the string (that is how `connection.py` applies a bridge force, 4 sites). A Rust `Vec` cannot
honour both and **both wrong answers are silent**: a copy loses the write into a temporary; a
zero-copy view over a reassigned `Vec` is a **use-after-free that still returns the old contents** —
i.e. it looks exactly like a correct snapshot until the allocator reuses the page (measured). Fix:
the binding holds **`Py<PyArray1>` — numpy arrays owned by Python** — and `step()` allocates a fresh
one and rebinds. Free, because that is the allocation pattern Python already had. **No physics test
in the repo can see any of this** (they all go through `state`, which copies), so the parity file
asserts it directly against *both* implementations.

**"Not bit-for-bit" was half wrong, and the useful half.** The *state* is **bit-identical** over
4,000 steps × 4 boundary spellings × loss on/off — an explicit step is pure elementwise arithmetic,
so IEEE fixes it exactly **provided the operation order matches**, which is why the Rust kernels are
written in NumPy's evaluation order longhand. Only *reductions* can't match (`np.dot` → BLAS):
energy agrees to **7e-16**. So assert exactness wherever the arithmetic permits it — far sharper
than a tolerance and free. **The line is "does the step contain a reduction", NOT "is it Group A"**
— `body`'s modal displacement and `mallet`'s contact force are both `np.dot` *inside the timestep*,
so they sit in the 1e-15 bucket despite solving nothing. Read the update before asserting exactness.

**Speed, honestly:** the 38 ideal-string tests went **23.7s → 4.2s** (5.6×, genuinely all-Rust);
the whole suite **421.8s → 382.4s**. Quote the first, not the second — the 9% is one uncontrolled
back-to-back pair with 21 of 22 models still Python, i.e. not separable from noise. Both runs 1,954
green, which is the useful half. The [[test-suite-performance]] "Rust makes the gate fast" claim
stays a **prediction** until Phase 2 has moved ten models.

**Step 5 (delete the Python model) cannot fire per model.** Not just the viewer ([[rust-migration-state]]):
`connection.py` — a **Phase 5** model — imports `IdealString` and touches the **private**
`_bc_right` and `_second_diff`, so those are part of the binding surface. Phase 0 correctly ends
with **both implementations alive**. The lever is the `PHYSSYNTH_RS` env switch at the bottom of
`physsynth/core/string_ideal.py`: because every consumer imports the one name, flipping it swings
`connection.py`, `body.py` *and* the viewer onto Rust at once. Deletion is not the milestone.

**Two things to do BEFORE writing physics next time.** (1) Probe the toolchain — a throwaway
extension proving zero-copy numpy views work — because §3.1 prices 160 call sites on it. (2) Build
**`abi3-py311`** so the dev machine (3.14) and CI (3.12) run the *same binary*; pyo3 0.29 /
rust-numpy 0.29 / maturin 1.15 all support it. Keep **two Python distributions** — root stays
hatchling/pure-Python so the baseline stays installable alone; the binding is its own maturin dist.

**`test_core_dependency_allowlist` failed first, on `physsynth_rs` — that is the tripwire working**
(see [[stiff-string-state]] for why it is hardcoded). Its rule now also lives Rust-side in
`crates/physsynth-core/tests/deps.rs` against `cargo metadata`, scoped to **physsynth-core's own
resolve set** (a workspace-scoped check passes vacuously — the workspace contains pyo3) and walking
**normal/build edges only**, with a self-test asserting the dev-only `serde_json` stays out.
`ALLOWED` is deliberately **empty** so Phase 1's first numeric crate is a written-down decision.

**Correction the plan needed:** `tests/test_string_ideal*.py` **never existed**. Tests here are
filed by *criterion*, not by model — the ideal string's 38 live in `test_energy` · `test_modal` ·
`test_convergence` · `test_dispersion` · `test_stability`. Look them up; do not guess a filename.
