---
name: parameter-mapping-state
description: "Parameter mapping (hurdle §12) STARTED 2026-10-18 for the damped string: string_tuning + viewer 'musical' domain; exact at two partials only, rest drift reported (allow + warn)"
metadata:
  node_type: memory
  type: project
  originSessionId: 2981c66c-a170-406c-8d54-2accb5e17c88
  modified: 2026-10-18T07:34:12.058Z
---

Batch done 2026-10-18 (`docs/dev/parameter-mapping-plan.md`, status DONE). The human chose: damped
string first, helpers **and** viewer sliders, corrected to the simulation (not textbook), and
after Part 0 "allow and warn" for drift.

- `crates/physsynth-core/src/string_tuning.rs`: `tune_damped_string(l, rho, fs, n, theta, &Targets)`
  → `T, kappa, sigma0, sigma1`. Each mode's root pair is LINEAR in `(k²Q_m, kσ_m)`, so the inverse
  is three 2×2 solves, exact at mode 1 and the anchor (mode nearest `f_ref`). `partial_drift`
  reports the rest; `HORIZON_CENTS = 5`.
- Viewer: damped string's domain select `physical` / `musical`; keys `f0, inharmonicity, t60_low,
  t60_high, f_ref, sample_rate`; payload `tuning` block only in musical mode. Physics path
  byte-identical (bar asserts it); freeze untouched.
- Headless case `string_damped_musical`.

**Why:** next mapping batches (plate "size/thickness/material", other strings) should reuse the
pattern: invert the scheme's own per-mode recurrence, not the continuum.

**How to apply:**
- A wiring bar must listen to the OUTPUT (rendered pitch), not compare a reported block to core —
  a build ignoring the mapping still reports it (planted 2026-10-18).
- Every slider default must sit on its step grid: 44,100 on a 1,000-step grid from 8,000 silently
  rendered 44,000; only the headless status line showed it.
- Fit a decaying single mode in the well-conditioned form (second difference on `a_n`,
  `a_n − a_{n−1}`; `ln_1p`), else T60 reads ~1e-5 off at low pitch.
- A NEW viewer domain trips `frozen.rs`' coverage guard (the freeze is the deleted Python's answers). The human's rule: add it to `BORN_AFTER_THE_FREEZE` with the covering test file; never record Rust output into the freeze.
- Verify_headless from Bash via `cmd //c start` hangs on the space in the repo path — launch it with
  PowerShell `Start-Process -PassThru` (gives the PID) instead.

Related: [[theta-loss-lock-state]], [[resolution-horizon-state]], [[unphysical-params-are-a-feature]],
[[damped-string-state]].
