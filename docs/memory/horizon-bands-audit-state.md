---
name: horizon-bands-audit-state
description: "The hand-picked-band audit (2026-09-07) — a band's limiter follows from its REFERENCE, so only 2 of 13 were pitch-bounded, and a derived band with no floor is a tautology"
metadata: 
  node_type: memory
  type: project
  originSessionId: ed2e5f64-eb18-4b5e-b87c-d723bb0b2486
  modified: 2026-09-07T04:56:17.512Z
---

The follow-on `docs/dev/resolution-horizon-plan.md` §6 named as "mechanical" — converting the test
suites' hand-picked mode bands into ones derived from [[theta-loss-lock-state]]'s successor, the
pitch horizon — is **done 2026-09-07** and it was not mechanical, not "several suites", and the
example §6 gave was not a pitch band at all. `docs/dev/resolution-horizon-plan.md` §7 is the audit;
`M:\claud_projects\physical synthesis\docs\dev\scientific-hurdles.md` §15 records the closure.

**The organising finding: a band's limiter follows from what the test compares AGAINST, not from
the model.** Four kinds of reference, three of which have no horizon:

- **continuum, inexact scheme** — the real pitch horizon. **2 of 13 bands.** Both derived.
- **the scheme's OWN discrete oracle** — the dispersion is already inside the reference, so there
  is no continuum in the comparison to be flat against. Horizon irrelevant *by construction*; the
  limiter is detectability of the n-th partial (pluck amplitudes fall like `1/n²`). 5 bands.
- **`λ = 1`** — the explicit family's time error cancels the spatial droop exactly, so the scheme
  has no horizon to read. This disqualified **more** tests than dispersion did. 3 bands.
- **the continuum of a DIFFERENT shape** — staircased circle/guitar outline; a cents reading mixes
  two errors. 1 band. See [[guitar-plate-state]].

**Six things worth carrying:**

1. **A FLOOR is not a HORIZON.** The plan's `N = 128` row reads 10; that is the *space floor* (what
   the string resolves as `k → 0`). The λ=1 fixture's actual horizon is **5** at 5 cents. Reading
   the floor as the horizon overstates a fixture by 2x — I wrote it wrong first and only caught it
   by measuring.
2. **The §6 exemplar was a DECAY-RATE band.** `test_sigma1_makes_high_partials_die_faster`'s
   `[1..16]` claims rate *monotonicity*; its limiter is the turnover at `~m=32`. Deriving it from
   pitch would have shrunk 16 → 5. §4's "rate error and pitch error are one factor and its square
   root" locks them for a **single mode**, never for a band — a monotonicity claim does not need
   its modes in tune (mode 16 there is 45 cents flat, mode 32 is 240 cents flat, and that is fine).
3. **A derived band with no floor asserts LESS than the literal it replaced.** If the horizon is
   measured from the same two arrays the test then asserts over, `max(err[:horizon]) < bound` is a
   **tautology** — it passes at horizon 1. Keep the old literal as `assert horizon >= floor`; that
   is the only line that can fail. Also assert `horizon < window`, because `pitch_horizon` returns
   the array length when nothing is outside, i.e. a lower bound wearing a measurement's clothes.
4. **`spatial_operator_horizon(N, kappa, cents)` looks general and is not** — it hardcodes
   `L_DEFAULT` and `wave_speed()` and a 1-D Dirichlet second difference, so on any other geometry
   or boundary it silently answers about the default *string*. Call `pitch_horizon` on the arrays
   the test already builds instead.
5. **Every literal in the suite was conservative, never over-claiming** (10 vs a real 48, 4 vs 6,
   1 vs 2). The batch was scoped on the suspicion that some band asserted past where its scheme is
   in tune. None did. **Amended by [[plate-mode-family-split-state]]:** the plate's band, once
   derivable, came out **exactly saturated** — horizon 2, band 2, zero headroom. Still not
   over-claiming, but the blanket "conservative" was written before that one could be measured.

6. **A window inside an assertion is a claim about the eigensolver.** The beam's band comes from
   `eigsh`, and hurdles §11 records ARPACK oracles that were not run-to-run reproducible (fixed in
   the migration plan §24.9). Widening `window` from 5 to 24 put 19 unchecked ARPACK modes inside a
   `monotone` boolean. Measured before trusting it: bit-identical over five fresh processes, and
   the smallest gap between consecutive per-mode errors is **0.214 cents** — the boolean's margin,
   the same question as [[test-suite-performance]]'s "a bar with 2% margin is a future false
   failure", asked of a flag rather than a float. The horizon also reads 6 at window 12, 24 and 40,
   which is stronger than stability: it does not depend on the window.

**Left open — CLOSED 2026-09-07, see [[plate-mode-family-split-state]].** The 2-D plate's band
(`test_plate_modal.py::test_low_modes_within_one_cent`) was refused here because `pitch_horizon`
counts a prefix of ONE mode family and its four modes mix axial with diagonal, "families measured a
factor of nine apart". **The refusal was right; that reason was backwards.** On the plate the two
families are *identical* in mode index — the factor of nine is the explicit scheme's cancellation
at its Courant ceiling, which the implicit plate has no analogue of. What really breaks the prefix
is that the frequency-sorted list is **not monotone** in the error. The band was derivable after
all, through the fact that a block's worst mode is its **diagonal corner**. See [[plate-state]].
