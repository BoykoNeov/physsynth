---
name: horizon-promotion-state
description: "The tuning-horizon helpers move from tests/ into the library; both of the plan's stated blockers were false, and a DELETION's constraint does not transfer to a MOVE"
metadata: 
  node_type: memory
  type: project
  originSessionId: f8335a9a-ed4f-4a59-8975-cf06e03a8884
  modified: 2026-09-07T15:24:56.163Z
---

The resolution-horizon primitives left `tests/helpers.py` for
`physsynth/analysis/horizon.py` over `crates/physsynth-analysis/src/horizon.rs`, 2026-09-07.
Seven functions — `pitch_error_cents`, `pitch_horizon`, `sinc_horizon_fraction`, `mode_family`,
`mode_block`, `cancellation_courant`, `block_weight`. Plan section is
`M:\claud_projects\physical synthesis\docs\dev\resolution-horizon-plan.md` §11; the batch is
ledger finding #70. Four plan sections (§6, §8.9, §9.8, §10.10) had named this "not done".

**§6 gave two reasons it would be hard and NEITHER was a reason.** That is the batch's content.

- **"`brentq` would be a dependency decision in the analysis crate."** It is not. That crate
  already `#[path]`-includes `physsynth-core/src/root.rs`, a line-for-line transcription of
  SciPy's `brentq.c`, for `modal.rs` — and `lib.rs`'s header argues out at length why an include
  and not a Cargo edge. The empty `ALLOWED` in `crates/physsynth-analysis/tests/deps.rs` stayed
  empty; `Cargo.toml` did not move. **A dependency worry stated about a function NAME is not a
  dependency worry** — ask what the crate already *compiles*.
- **"Freezing is impossible, no Python implementation is left."** That fact belongs to the
  modules **deleted** in `rust-migration-plan.md` §44. These seven were **promoted**, and a
  promotion has a live Python body until the commit that replaces it. `scripts/freeze_horizon.py`
  walked that one-commit window: 12 cases, Python and Rust measured in the same pass. No guard
  amendment; the contract over the other 62 fixtures is untouched. **A constraint inherited from a
  deletion does not transfer to a move** — see [[rust-deletion-phase-state]],
  [[analysis-freeze-state]].

**Every float case recorded a gap of EXACTLY 0.0**, including `sinc_horizon_fraction`, where two
hazards stacked: the Python used SciPy's *implicit* `brentq` defaults (`xtol=2e-12`,
`rtol≈8.88e-16`, now spelled out in the Rust), and the objective calls `sin`, which NumPy computes
with its own CPU-dispatched routine ([[numpy-libm-cpu-dispatch]]). Both were measured, not assumed.
That is a claim about **this machine**, which is why the frozen bar stays a `1e-13` tolerance.

**Five frozen rows record a STRING, not a gap, and it is the strong arm.** `pitch_horizon` returns
`(int, bool)`; the two index builders return lists of integer pairs. No float in the answer means
the whole comparison is the exact one on `ints` and `structure`. The canary that used to reject any
non-float now separates that from a real incomparability, and additionally checks such a row really
has no floats **and does have integers** — otherwise the exemption becomes a way to freeze a row
asserting nothing. **A recorded 0.0 on those rows would have meant "nothing float was compared",
not bit-identity.**

**`ANALYSIS_MODULES` stopped being a hand-written tuple** and is derived with
`pkgutil.iter_modules`. Adding `"horizon"` by hand would have restated the hole one number higher.
Same move as rust-migration ledger #67.

**A native bar caught a docstring overclaim** (27 bars in
`crates/physsynth-analysis/tests/horizon.rs`, all identities rather than fixtures). `mode_block`
said the error weight "has an interior minimum in `n`". Minimising `(m⁴+t²)/(m²+t)` over `t=n²`
gives `t* = m²(√2−1)`, so the minimum is at `n* = m·√(√2−1) ≈ 0.6436 m` — interior for the
*continuous* function, and on the integer grid only reachable from **m ≥ 3**: at `m = 2` the
minimiser is 1.287 and the nearest index below it is the block's own edge. The corner argument is
untouched (it is about the **maximum**, still always a corner).

**`spatial_operator_horizon` did NOT move**, and that is the design rule: it hardcodes
`L_DEFAULT`, `wave_speed()` and a 1-D Dirichlet second difference, so it silently answers about the
canonical string. Honest as a test fixture, a lie in a library. **A promoted helper must not carry
a fixture inside it** — which is why `mode_family` returns index pairs only. There are **no
re-exports** left in `helpers.py`: two routes to one name, and ruff's `--fix` deletes an unused
import (ledger #66).

**The unrecoverable part is the SEQUENCING.** Record and commit before the edit that removes the
second implementation; make the generator refuse loudly afterwards rather than emit a half-file.
`scripts/freeze_horizon.py` now exits with that message.

Unblocked next: the viewer read-out ("this configuration is trustworthy to 2 kHz") —
`web/serialize.py` can import the module like any other oracle. See [[web-viewer-state]].
**DONE 2026-09-07 — [[viewer-horizon-readout-state]]**, and the surprise was how many of the
viewer's models have to refuse rather than answer.
