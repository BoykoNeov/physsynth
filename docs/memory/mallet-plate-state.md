---
name: mallet-plate-state
description: "Model #7p mallet-on-a-plate (2026-09-06) — an IMPLICIT resonator hands over a COLUMN, not a scalar; a struck FREE plate's energy read-out decays QUADRATICALLY in its rigid drift; the felt exponent is the sole source of dynamic timbre"
metadata: 
  node_type: memory
  type: project
  originSessionId: 53b6adaf-b700-458c-bde9-add6c2938cb6
  modified: 2026-09-06T11:17:53.174Z
---

Model **#7p, `MalletPlate`** — the mallet struck a plate, built 2026-09-06. Closes the gap
HANDOFF §14.1 named ("there is no mallet-on-a-plate composition to reach for"). Rust only, no
Python original ever existed: `crates/physsynth-core/src/mallet.rs`,
`crates/physsynth-py/src/mallet.rs`, re-exported by `physsynth/core/mallet.py`.
Full record: `M:\claud_projects\physical synthesis\docs\dev\mallet-plate-plan.md`.

**The one new object is an admittance, and for an implicit resonator it is a VECTOR.** The membrane
is explicit, so a nodal force reaches only its own node next step and `g_s` is a scalar with a
closed form. The plate solves `A u^{n+1} = rhs`, so the force reaches every node and what stands in
is the **influence column** `(k²/force_den) A⁻¹ e_node` — one back-substitution at construction,
`g_s = influence[node]`, and the whole column is how the force is spread back. This is the bow's
manoeuvre against the implicit stiff string, and `docs/dev/hammer-collision-plan.md` predicted it
four batches early. Everything else is model #7's, already proven.

**The guard on a column must be a WHOLE-FIELD claim.** A column right at the strike node and wrong
elsewhere satisfies a drive-point check, passes every energy bar (a wrong field is just a different
self-consistent trajectory) and sounds wrong. Assert `plate.step(f_ext)` == `plate.step(None)` +
`influence*f` at every node, on both branches. Not bitwise — a sparse LU back-substitution is not a
linear map over doubles — but a **miss** is exact, because `f == 0.0` makes every increment a signed
zero and `x - (±0.0) == x`.

**THE FINDING: a struck FREE plate's energy read-out decays quadratically in its rigid drift, and
it is the PLATE's, not the mallet's.** A point strike feeds the `{1, x, y}` nullspace, the plate
translates for ever (exactly: `1ᵀW` kills the stiffness term, so the weighted mean is linear in the
step index), and the potential form `κ²(Ku).g` — which annihilates the rigid part *mathematically*
— cancels it only to `eps`. Being a **quadratic** form, the leftover error goes like the **square**
of the drift: measured on a **bare** plate with no exciter, `error/drift²` is constant to three
digits (2.90e-10 → 3.01e-10) across 30× of drift, against 6.4e-16 J at zero momentum. The mallet is
only the first thing in the project that ever gave a free plate net momentum — every existing
free-plate test starts zero-mean. **No rig tuning fixes it**: relative drift = plate share ×
cancellation, and making the plate heavier shrinks both, so "conserves to 1e-10 over a long window"
and "the plate actually takes the strike" are not simultaneously available. Free branch runs on a
documented **read-out** bar (1e-8 / 2000 steps, measured 9.8e-10) with the supported branch as a
control at 3e-13. Cross-referenced into `docs/dev/plate-free-edge-plan.md`.

**The felt exponent is the ONLY source of dynamic timbre, and it is graded.** At `alpha = 1` the
whole system is linear (plate linear, mallet a mass, one-sided switching scale-invariant), so the
response is proportional to strike velocity to **7.6e-13**; departure is 0.19 / 0.62 / 0.71 at
`alpha` = 1.5 / 2.3 / 3.0. Everything a player calls *dynamics* lives in one exponent.

**It refuses a `VKPlate` and that refusal is a batch, not a cast.** The von Kármán step is
nonlinear → not affine in `f_ext` → the influence column does not exist. The gong needs an outer
contact solve around a full Picard/Newton plate solve per residual evaluation. `StringVKPlateBridge`
does not transfer: its `F = K η^n` is sweep-invariant and sits outside the Picard loop, which is
exactly what a discrete-gradient force (implicit in `η^{n+1}`, and implicit is what makes it
conserve) is not.

**That batch is BUILT — [[mallet-gong-state]], model #7g, 2026-09-06 — and it falsified this
batch's two numbers about it.** The cost is **1.9–2.3×** a bare gong step, not the 10–100× predicted,
and the outer iteration **does** have a closed-form derivative. Both errors have one shape:
superposition fails, so the influence column is not the *answer*, but it fails **by a little**
(`k²`-scaled), so the linear column is an excellent **frozen tangent** exactly where it is a useless
exact solution. "Not exact" was read as "not useful". Also falsified from here: the felt-exponent
headline above is a claim about a **linear** resonator — on a gong, `alpha = 1` makes the felt's
contribution exactly `0.0` and the plate still moves the timbre by 2.12.

**What moves an energy bar here is the NODE COUNT, not the conditioning.** `mu` (the plate's own
Courant number) sets `cond(A) ~ 1 + 64θmu²` on its own, but a `mu` sweep at fixed `fs` needs
`N² ~ mu`, so it moves conditioning and problem size together. Separated (hold `N`, scale `kappa`
with `mu` to keep `fs`): conditioning does **nothing** (1.6e-13 to 2.3e-12 across 256×,
non-monotone); the node count does, because `energy()` is a reduction — 1.9e-13 at 256 live nodes to
1.7e-11 at 4489. Superposition is insensitive to both (4e-16 → 1.8e-15).

Traps: **`pickup_index_at` returns a LIVE index while `Params::x`/`y` are FULL-GRID** — share the
traversal (`live_coords`) so the mixup is unrepresentable, or you get a plausible strike point and a
wrong column. **`_accel` must be corrected too** (else `pressure()` reports the unstruck plate), and
its guard needs its own normalisation — a displacement error arrives there as `e/k²` while the
acceleration's own scale is `(ωk)²` smaller. The plate's `fs` comes from **its** Courant number and
knows nothing about the felt, so `make_mallet_plate` defaults to `mu = 1.0` and a test asks
`steps_per_contact >= 8` of the shipped defaults directly.

Related: [[mallet-collision-state]] · [[plate-state]] · [[free-plate-state]] ·
[[string-vk-bridge-state]] · [[bow-state]] · [[von-karman-plate-state]]
