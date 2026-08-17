---
name: string-vk-room-chain-state
description: "The three-way chain string→bridge→room-loaded gong→room (shipped 2026-08-17) — zero core edits, no third fixed point, batch 6's headline dies on its own rig, and band overlap replaces amplitude"
metadata: 
  node_type: memory
  type: project
  originSessionId: 30b86e83-59ee-4ecf-aaed-7c395e9ed3b1
  modified: 2026-08-17T12:00:22.403Z
---

`string → StringVKPlateBridge → RoomLoadedVKPlate | RoomSuspendedVKPlate → AirBox`. Shipped
2026-08-17. The thing [[air-box-state]] batch 6 and [[string-vk-bridge-state]] each deferred to the
other. Plan: `docs/dev/string-vk-plate-room-plan.md`; tests in `tests/test_airbox_vk.py`;
`scripts/diagnose_string_vk_room.py`.

**Zero core edits.** It composes as-is on all four combinations (baffled/suspended ×
supported/free). The wrappers' `__getattr__` already reaches every name the guard assembles from,
and the guard is **bit-identical** to the bare plate's — safe because the air load is
*dissipative*: it enters `A`, never `G0`. The only source edit is a stale comment. 26 new tests,
14 s.

**The bridge plan's "third fixed point" does not exist.** There is one. `F = K η^n` is
sweep-invariant and enters the RHS outside the Picard loop; the room's terms go into `rhs_fixed`;
`TᵀRT` folds into `A` once. Bare and loaded bridges take the **same** sweep count at the same
pluck (4/4 loud, 3/3 quiet). Phrase it **"the room adds no outer iteration"**, never "the room does
not affect convergence" — batch 6's finding that *coarsening* the room breaks the plate's fixed
point stands beside it, not against it.

**The obvious headline died against a shipped batch's own numbers.** Batch 6's `sigma_shape` with a
string in place of its strike, on batch 6's rig, at the **same peak `w/e` = 3**: flat to four
digits — **0.27% against a 0.28% control**, where the strike gives 46.0% against 1.4% — and the
loud arm's spread lands **below its own quiet control** (1.0027 vs 1.0028). That
is what makes it a null result rather than a small one — a dead claim measured against a *shipped*
batch is the strongest kind this project produces. Radiated fraction **0.0042%** of the scene over
the 120 ms window, so **the room is a read-out here, not a variable** — which is why the
replacement claim had to come from the plate's MOTION, not from anybody's books.

**Why: `w/e` is not an amplitude when the drive is a POINT FORCE.** A point force on a free plate
feeds the `{1, x, y}` rigid nullspace; rigid motion stretches nothing; the von Kármán coupling is
a functional of *stretching*. At matched peak displacement the string-driven plate carries **~190×
less energy** than a struck one (158× raw, **190×** normalised by `(w/e)²`). **The un-normalised
781× was wrong** — the struck arm was then started at `w/e = 3` and *rang UP* to a peak of 5.213,
so 781× compared 2.997 against 5.213 and called it "the same peak". Fixed BOTH ways: start the
struck arm at `w0/e = 1.727` so its peak lands at 2.730, **and** normalise by `(w/e)²`. Batch 6's
own "magnitudes don't survive changed conditions" landing inside the batch citing it. And the struck arm's 0.00% rigid
share is an **identity, not a measurement** (a displacement start gives the nullspace no velocity),
so the contrast measured on both sides is string-driven free vs string-driven supported.

**The two boundaries are out of reach by OPPOSITE mechanisms** — the sentence no earlier batch could
write. Free (cymbal): reaches `w/e = 3` easily, 95.5% of it bounce. Supported (gong): no nullspace
to hide in, so reaches only `w/e = 0.121` at the same pluck, 25× less.

**THE CLAIM — band overlap, not amplitude.** Hold the plate; move the *string's* fundamental across
the plate's first flexural mode by **length only** (so wave impedance `√(Tρ)` is fixed), at
constant pluck **energy**. Rigid share collapses 95.5% → 4.5%; departure from the plate's own
linear self spans **5894× between the worst and best overlap** (7082× at the hard pluck) — a
**range, not a trend** (the column is non-monotone). All at peak displacements inside where the model is faithful. **Reproduced
across two rigs 7× apart in sample rate** (57.9 kHz / 100 mm plate and 8 kHz / 300 mm plate), which
is what makes it a mechanism rather than a tuning. **Nobody tuned the cheap rig there** — it is the
LINEAR chain test's rig, chosen long before this batch (300 mm plate, first flexural mode **35.57
Hz** measured), and it lands at `f1/f_el = 2.566` with a 4.3% rigid share. **Be precise about how
far the agreement goes:** the TRANSITION reproduces quantitatively at the low-ratio end (0.321 →
95.4% cheap vs 0.28 → 95.5% expensive), but the residual floor above it agrees only in order
(2.566 → 4.3% vs 2.00 → 1.7%, ~2.5× apart). Claim the transition, not the floor.

**The optimum MOVES with the pluck** — the plate's energy share peaks at `f1/f_el = 1` soft
(18.85%) and at `2` hard (84.06%) — a second, smaller claim riding on the
first, and something no linear chain can do. Neither peak *value* is claimed as a magnitude.

**Three negative controls, all measured.** (1) Bridge stiffness is not the lever: 10× in `K` moves
the rigid share <1 point — the coupling **saturates in K**, and the guard refuses beyond it
(`K = 1e5` → margin 3.25). (2) The rigid share is **amplitude-invariant** to three figures across
plucks 186× apart in energy. (3) The ratio survives where the magnitude does not (7082× hard,
5894× defensible — quote the defensible).

**This upgrades [[string-vk-bridge-state]]'s closing line** ("want the gong impression, use a
mallet, not a budget") from assertion to mechanism, and sharpens it: **not a budget question at
all** — a string cannot deliver the *shape* at any budget unless its band reaches the plate's.

**The family's insufficiency rule gains a new KIND.** A wrong string reaction (`beta_s`) is caught
by the scene total (2.8e-15 → 8.8e-2) and invisible to the money test. But a `drive_index` that
merely **differs between two runs** moves the comparison **1.7× with every detector at machine
precision**, because each run is internally consistent so there is nothing inconsistent to detect.
**The three detectors are jointly insufficient against a COMPARISON, not only against a
coefficient.** Hence: pass `drive_index` explicitly whenever two chains are compared, and build the
linear twin through the same construction path with `nonlinear=False`.

**Corner refused honestly:** the thin-plate arm (plate thinned so its modes descend onto a fixed
string) shows the same lever monotonically, but arrives at `w/e = 54`, far outside von Kármán's
moderate-rotation regime. Corroboration, never the claim — moving the *string* is the honest form.

**Cost:** 10.6–10.9 ms/step on batch 6's rig; the diagnose script ~12 min. The bridge plan's fear
that a third fixed point would make this unaffordable never materialised.

**An observable thrown away mid-build:** the suite test first asserted on the plate's ENERGY
share and failed, non-monotone — because the plate's energy **counts rigid bouncing as energy
the plate received**, the batch's own headline turned back on itself. Also: never sample it at
the last step; it oscillates as the plate trades with the string. Take the max over the run.
