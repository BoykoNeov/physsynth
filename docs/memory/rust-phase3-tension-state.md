---
name: rust-phase3-tension-state
description: "Rust migration Phase 3 batch 4 — model #9 (tension string); a reduction that a ROOT-FIND branches on, a telemetry attribute sharper than the trajectory, an agreement window set by amplitude, and a CI step red since the last batch"
metadata: 
  node_type: memory
  type: project
  originSessionId: a8b75527-17df-450b-bbcb-ddb1076fea44
  modified: 2026-08-27T11:28:25.924Z
---

Phase 3 batch 4 (2026-08-27): `physsynth/core/string_nonlinear.py` — model #9, the
tension-modulated (Kirchhoff–Carrier) string — is ported. Third of the four theta-scheme strings;
only `string_geometric` remains, and it needs Group D. See plan §19.

**The first model whose update matrix moves every step.** `A = A0 - beta D2` depends on the
tension, so the banded factor is inside a scalar root-find, not at construction — `banded` (§15)
meets `root` (§13.3) for the first time, ~7 refactor+solve per step.

**The finding — §18.10 predicted the obstacle and got the fix wrong, and §18.3 had already written
the right answer.** §18.10 said the port would have to match BLAS `ddot` head-on. Measured: it
cannot (612/800 real `_stretch` vectors differ). But the alternative was not "accept a tolerance" —
because **the stretch feeds a `brentq` residual, so a last-bit difference changes an INTEGER**: the
two spellings take a different number of residual evaluations on **1,400 of 5,000 steps**. So
`_stretch`/`_stretch_int` moved to `portable.dot`, unconditionally, on the default path — §18.3's
rule (*a decision justified by "nothing downstream depends on this" expires when something
downstream ports*) arriving one batch after it was written. Under the alternative the parity bar
would have been ~2e-12, set by `tension_tol=1e-13` rather than by the reduction — four orders
looser than the ulp it was meant to guard. **The question at each reduction gains a clause: not
only "does this reach the next timestep" but "does anything downstream BRANCH on it".**

**A public read-out can be a strictly sharper detector than the state.** The batch's one real
porting error was a mis-associated `_stretch_int` (`dot + (a+b)` instead of `(dot+a)+b`). The
trajectory stayed **bit-identical for 2,000 steps** through it; `delta_tension` caught it. Reason
is quantitative, not luck: `beta = k^2 dT/(2 rho)` is ~4e-9, so a last bit of `dT` never reaches an
O(1) band entry. So the parity file compares all four telemetry attributes — two of them integers —
**exactly, and first**. Before writing a parity file ask **which quantity is nearest a branch**.

**A fourth agreement regime, and the first set by an operating point inside the model.** #9 is
parametrically unstable above `dT/T0 ~ 3` (real physics). Same code, same solver gap, two answers
by amplitude alone: sub-threshold 2.0e-14 / 3.0e-13 / 9.7e-11 at 100 / 1,000 / 20,000 steps; above
threshold 1.4e-13 / **2.6e-3** / 7.3e-1. Both conserve energy to 1e-10 throughout. So a parity
fixture's **amplitude is part of its claim** — the third form of [[rust-phase3-collision-state]]'s
fixture question (§16.4) and the first about amplitude rather than stiffness or persistence.

**§16.4 again, third time, new reason each time:** at amplitude 1e-6 the root-find never separates,
so a gentle fixture is green **with the port's reduction wrong**. Both halves are assertions in the
parity file, not comments.

**A CI step had been red since the previous batch.** `tests/test_rust_parity_mallet.py \n ...` — a
literal backslash-n instead of a line continuation — meant bash handed pytest a path `n`, so the
"Rust vs Python parity" step exited 4. Committed with §18. Fixed here. **A shell continuation
inside a YAML block scalar is checked by nothing**, and it fails loudly for the wrong reason.

Smaller: the `dict[float,...]` memo is performance-only and deliberately NOT ported; `apply_Ainv`
**raises** and the refusal is interface (three coupled models call it on whatever string they get);
the `RuntimeWarning` goes through Python's `warnings` so `pytest.warns` sees it; the
bracket-doubling loop is **dormant** (0 across ten fixtures — §16.6's Armijo hazard again) and is
driven natively instead. Also measured: `np.float64(x)**2 == math.pow(x,2.0)` in 200,000/200,000,
so `scalar_pow` is the right spelling for the end terms.

Measured: `cargo test --workspace` **228** (both profiles); 13 parity files **1,194** flagged /
1,193+1 skipped unflagged; `test_rust_parity_tension.py` **90** both ways;
`test_tension_string.py` **39** against Rust.

Related: [[rust-phase3-strings-state]] · [[rust-phase3-banded-state]] · [[rust-migration-state]] ·
[[tension-string-state]]
