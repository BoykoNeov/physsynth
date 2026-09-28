---
name: rust-phase7-spectrum-state
description: "Phase 7 batch 1, the partial detector — a SECOND flag (analysis swaps the instrument, not the model), a decision whose margin is EXACTLY ZERO but on the exact axis, and the CI collapse from 49m to 9m"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2d381787-81fe-408c-aa3e-260eac2cbf68
  modified: 2026-09-03T09:41:25.043Z
---

`analysis/spectrum.py` in Rust, 2026-09-03 — the first module out of `physsynth/analysis`, so it
brought a **third crate** (`crates/physsynth-analysis`). Plan §36; ledger #28 and #29. See
[[rust-phase5-connection-state]] for where `core/` finished and [[rust-migration-state]] for the
destination.

**The batch's finding: a decision's MARGIN and its AXIS are separate questions, and a
decision-valued module ports safely only when they run opposite.** This file splits along a seam no
earlier finding predicts — not reduction-vs-step, not solver group, not values-vs-stored-order, but
two *axes of one computation*:

- **Frequency axis** (`freqs`, `df`, window bounds, `min_separation_hz`, every comparison among
  them) is `+ - * /` alone → IEEE-754 pins it → bit-exact on any machine, no CPU claim.
- **Magnitude axis** is unmatchable: NumPy's own dispatched `cos` in the Hann window, pocketfft,
  `hypot`. Differs in 4,074 of 4,097 bins at 3.2e-16 of the peak.

**Measured BEFORE writing anything** (a pytest plugin over the 18 dependent test files, 384 real
`measure_partials_near` calls, 92,261 candidate peaks): winner-vs-runner-up **≥1.4e12 ulps**,
local-max guard **≥1.6e10**, candidate ordering ≥7.6e7 with **zero** exact ties — versus a
separation test whose margin is **exactly zero**. The tight one is on the exact axis. That is the
licence, and it is a number rather than a hope.

**A flag's meaning is a property of what it does NOT swap.** `PHYSSYNTH_RS` swaps models;
`PHYSSYNTH_RS_ANALYSIS` swaps the instrument. Under the first alone a Rust model is measured by a
Python detector against an unmoved oracle — that is what makes the acceptance gate mean anything.
§5 scheduled Phase 7 late for this reason and §35.3 replanned the order **without re-taking the
argument**. Before widening any flag, ask what the existing runs relied on staying still.

**Other things worth carrying:**

- The zero-margin comparison **does not always clear** — at 100 kHz with `nfft=16` it *rejects*. So
  the claim is **agreement, never outcome**. It survives only because `1.0/(n*(1.0/fs))` was
  transcribed and not tidied to `fs/n`: different numbers for ~**1 random sample rate in 8**, but
  identical at every rate this project uses, so a hand-picked witness would have blessed the tidy
  form. Pins must **search** — same scar as [[rust-phase5-matrices-state]].
- **Removing a transcendental from a decision beats spelling it portably.**
  `2**ceil(log2(n))` → integer `next_power_of_two`, checked equal for every length 1..2^20 and every
  2^k boundary to 2^31. Retires the claim instead of relocating it (contrast
  [[numpy-libm-cpu-dispatch]]).
- The guard from [[spectrum-detector-guard]] is **live**: it fires on 14 of 384 real suite calls.
  Not a defensive branch — port it exactly, and put both recorded witnesses in the parity fixture.
- **A loop-order change is a cache change first.** Hoisting the FFT twiddle out of the block loop
  cut transcendental calls from `(n/2)log2(n)` to `n-1` and made long transforms **5x slower**. One
  size is not a measurement. Final: 1.25x at 2^10, 0.51x at 2^18 (a textbook radix-2 against
  pocketfft — the honest price of an empty dependency allowlist).
- **A new crate inherits the portability contract's convention and none of its enforcement** —
  `deps.rs` is rooted at its own package by name. `physsynth-analysis` carries its own allowlist.
