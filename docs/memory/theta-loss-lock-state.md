---
name: theta-loss-lock-state
description: "scientific-hurdles §4 probed 2026-09-06 and NOT built — the θ-scheme's decay suppression and its pitch flattening are ONE factor and its square root, so the fix repairs the quieter half"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c0c4a48-d743-45a8-a700-73c01ddab289
  modified: 2026-09-06T20:31:40.261Z
---

The human picked `docs/dev/scientific-hurdles.md` **§4** (the θ-scheme's damping rate suppression)
on 2026-09-06, on the strength of one sentence: "the smallest physics change with an audible payoff
in the whole register". **The Part 0 probe killed that sentence before any code was written**, and
the decision was to record it and not build. Plan and full tables:
`M:\claud_projects\physical synthesis\docs\dev\theta-loss-compensation-plan.md`; probes at
`M:\claud_projects\temp\theta-loss\`.

**The finding, in one line: the two errors are the same number.** The θ-scheme's lossless
amplification gives `sin²(ωk/2) = k²Q / (4(1 + θk²Q))`, so the discrete **frequency** carries
`1/√(1 + θk²Q)` while the decay rate carries the whole factor. Therefore

```
pitch error (cents) = 600 · log₂(S)          S = the decay-rate suppression
```

verified to **0.57 cents** over 108 string configurations wherever `S > 0.99`, and it loosens
outside that regime in the direction that *strengthens* the conclusion. A mode whose decay time is
10% long — about one decay JND — is **82 cents flat**, some fifteen pitch JNDs. **The compensation
repairs the quieter half of a single defect and leaves the louder half exactly where it was.** The
only knob that fixes both is a smaller `k`.

Four things worth carrying forward:

* **The ordering claim never fires in tune.** Swept 90 configurations with `(σ₀, σ₁)` derived the
  way a user derives them (`loss_coefficients_from_T60` at the fundamental and mode 20), the
  turnover's best case was mode 26 at **330 cents flat**. Realistic `σ₁` is 1.8e-3–3.7e-3 against
  the suite fixture's 1e-4, and **larger σ₁ pushes the turnover further out**, not nearer.
* **My prior that "the plate is the bigger payoff because it has no Courant bound" was wrong**, and
  so was the advisor's. Larger suppression is larger *mistuning*. Under a lock, a bigger error on
  one axis is not a bigger opportunity.
* **The one independently visible artifact is a TIMESTEP artifact.** A broadband strike's tail
  rings 7.41× too long at `μ=2`, 1.17× at `μ=0.5`, 1.00× at `μ=0.05`.
* **§4's observations all stand; only the payoff framing is retracted** — the same split §5's probe
  drew when it kept the observations and threw out the mechanism. Write a falsification that way and
  it lands as a correction rather than as a repudiation of the whole section.

**The method lesson: when a probe is asked to justify a fix, make it able to KILL the fix.** The
decisive probe was not "how big is the error" but "what else is wrong with the same mode, and which
is more audible". A fix's payoff is never the size of the error it removes — it is the size relative
to whatever error survives it.

The compensation stays **scoped and costed** in the plan's §4 (five parts, including the `kd` 2→3
bandwidth change that `apply_ainv`'s three external consumers would have to be re-routed for) so a
future decision is made against a real number. Nothing was started.

Related: [[resolution-horizon-state]] (the thread this opened, and the human's chosen follow-on),
[[damped-string-state]], [[vk-newton-state]], [[plate-state]].
