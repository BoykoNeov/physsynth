---
name: unphysical-params-are-a-feature
description: "feedback — don't design APIs that make physically-inconsistent parameter combinations unrepresentable; unrealistic instruments are an interesting sonic experiment (HANDOFF §12.J), so offer realism via helpers, never impose it"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 16bcf34c-2240-4459-a026-52fa69e62fbc
---

When a core API could either (a) take **effective coefficients** that permit physically
inconsistent combinations, or (b) take a **materials surface** that makes inconsistency
unrepresentable — **default to (a)**, and provide consistency as an *optional helper*.

The human, on model #9's `EA` decision (2026-07-16), choosing the effective-coefficient surface:

> "I don't think it is a minus to build a string in the software with unrealistic properties, yes,
> we aim generally towards realism, but this might be an interesting experiment, producing
> interesting sound."

**Why:** "You could build a physically impossible instrument" is **not a defect to be designed out**
— it is HANDOFF **§12.J, hyperreal instruments** ("physics beyond real materials — negative
stiffness, time-varying or morphing geometry, non-Euclidean or fractal resonators"). A string with
steel's bending stiffness and rubber's axial stiffness is a sound nobody has heard, and it is
reachable *only* because the surface is effective coefficients. Locking the API to a consistent
materials triple would foreclose a stated long-term direction to prevent a "problem" that is
actually a feature. Realism is the *default aim*, not a constraint to enforce in the type system.

**How to apply:** Expose the effective coefficients the physics actually uses. Add a pure helper
(the `radiation.py` `R_a`-helpers / `string_coefficients_from_material` pattern) that derives a
consistent set from real material + geometry, and have it report the governing dimensionless ratio.
The helper **offers** realism; it must never **impose** it. Two supporting arguments that recur:
the family usually *already* exposes effective coefficients (so the "new" inconsistency isn't new),
and a materials surface's consistency guarantee is often **fiction** for real composite objects
(wound strings have no single `E`/`radius`/`ρ_v` — the literature characterizes them by exactly the
effective coefficients). Note this cuts the other way where the object really *is* homogeneous:
model #6's plate genuinely is a sheet of thickness `e`, and there the materials surface was right.
Don't over-generalize either way — ask.

See [[tension-string-state]] (where this was decided), [[von-karman-plate-state]] (the opposite
call, correctly).
