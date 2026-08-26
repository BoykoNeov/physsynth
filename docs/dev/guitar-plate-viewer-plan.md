# Viewer batch 20 plan — the guitar plate on screen: the waist swaps the fundamental

**Status: PROBED, not yet built.** Every number below was measured in
`M:/claud_projects/temp/guitar-viewer/` before a line of viewer code was written — the same order
batch 19 and model #5g itself used, and for the same reason: two of the five probes changed the
design, and one of them would have shipped a green batch that showed the wrong picture.

This batch surfaces model #5g (`docs/dev/guitar-plate-plan.md`, built 2026-08-26) in the web viewer.
The viewer plan's standing rule — *a new core model reopens the built-but-unshown gap* — named it as
a candidate the day #5g landed.

---

## 1. The gate, and the plan's own amendment was wrong about it

`docs/dev/web-viewer-plan.md`'s 2026-08-26 amendment says #5g "arrives with a frontend requirement of
its own … the viewer's field renderers all assume a **rectangular** field, and a masked outline needs
dead nodes drawn as absent rather than as zero — which is a new capability, not a new parameter."

**That is false, and it is worth correcting in place rather than quietly building around.** The
frontend has drawn dead nodes as panel background since the membrane batch: `app.js` tests
`maskData[p] === 0` in all three of its 2-D paint paths, and the circular drumhead exercises it every
render. The `mask` block has been in the payload since batch 13.

What *is* new is smaller and more specific, and the batch should claim only this:

- **No free-plate payload has ever shipped a non-trivial mask.** `platebody` ships
  `"domain": "rectangle"` and an all-ones mask; `plate` likewise. Wiring a real outline through is a
  parameter-level change, not a capability — but it is not "already covered" either.
- **The display path has no connectivity guard, and the core does.** That gap is §4, and it is the
  one thing in this batch that blocks.

⇒ The rule that authorises this batch is therefore *model-newness alone*, which #5g satisfies
outright. Do not claim a capability that already shipped.

---

## 2. The claim: the waist does not detune the plate — it SWAPS the fundamental

The plan for #5g framed the waist as "the knob that makes the shape a guitar rather than an ellipse".
Probed, it is sharper than that, and the answer was not implied by either precedent — #5o (the
supported orthotropic plate) detuned selectively *without* reordering, #5of (the free orthotropic
plate) *did* reorder.

The guitar's waist reorders, and it does so at the very bottom of the spectrum:

- Below a critical waist the fundamental is a **long bending mode** — no sign change across the
  width, two nodal lines across the length. It is **even** under the plate's left-right mirror.
- Above it the fundamental is the **twist** (the saddle) — one sign change across the width, none
  along. It is **odd** under the same mirror.

Measured at N = 44, `Lx = 0.37`, `Ly = 0.48`, ν = 0.3, mirror parity of mode 1 (exactly ±1):

| waist | 0.10 | 0.20 | 0.25 | 0.26 | 0.30 | 0.42 |
|---|---|---|---|---|---|---|
| parity of mode 1 | +1.00000 | +1.00000 | +1.00000 | **−1.00000** | −1.00000 | −1.00000 |

### 2.1 The detector is the PARITY, and that is the whole reason this claim is cheap

The outline is `|x| < W(y)`: mirror-symmetric about the centre line whatever `waist` and `asym` do.
So every mode is exactly even or exactly odd in `x`, the two families do not couple, and a crossing
between one of each is a **true crossing, not an avoided one** — nothing is available to open a gap.

That makes the detector one scalar per plate: `⟨φ₁, mirror(φ₁)⟩ / ⟨φ₁, φ₁⟩`, which measures
**+1.00000 or −1.00000** and nothing in between. No shape tracking, no overlap threshold, no
eigenvector matching across a degenerate pair. Across 12 configurations (N = 24…48 × 16/24/32 sweep
points) the sweep found **exactly one flip**, every time.

### 2.2 Where the crossing sits, and what moves it

Grid-converged (fine waist scan, minimum of `f₂/f₁`):

| N | 24 | 32 | 44 | 60 | 76 |
|---|---|---|---|---|---|
| crossing waist | 0.240 | 0.250 | 0.250 | 0.260 | 0.260 |

It is **not a universal constant** — it is a balance between elongation (which favours the bender)
and the waist (which favours the twist), so a longer body needs a deeper waist to swap:

| `Ly/Lx` | 1.081 | 1.189 | 1.297 | 1.486 | 1.676 |
|---|---|---|---|---|---|
| crossing waist | 0.183 | 0.200 | 0.250 | 0.400 | 0.567 |

Poisson's ratio moves it mildly (0.280 at ν = 0.15, 0.250 at ν = 0.30, 0.220 at ν = 0.45).

A second crossing sits between modes 5 and 6 near waist ≈ 0.6; the batch shows the first one and
mentions the second, because two crossings on one panel is a chart, not a claim.

### 2.3 What NOT to ship as evidence

`min f₂/f₁` over the sweep reads 1.00777, 1.00046, 1.00462, 1.00297, 1.00068 at N = 24…76 —
non-monotone, and **not a gap**. §5 explains why: those numbers say only how close the grid's
representable waists happened to land to the crossing. Refining the scan proves it — six successive
window bisections all return the *identical* residual `1.641e-03`, because the mask stops changing.

⇒ State the crossing as **an interval between two adjacent representable waists**. Never as a number
with a residual attached, and never with `min f₂/f₁` beside it as if it were a convergence measure.

---

## 3. The waist slider is QUANTISED, and a viewer that does not say so reads as broken

The outline is a staircase, so `waist` only changes the plate when a node crosses the rim. Between
those points the mask is bit-identical and *nothing* moves — not the picture, not the audio, not one
digit of the spectrum. Distinct outlines as `waist` sweeps 0.00 → 0.90:

| N | 16 | 24 | 32 | 44 | 60 | 80 |
|---|---|---|---|---|---|---|
| distinct plates | 50 | 97 | 182 | 285 | 469 | 675 |
| mean waist step | 0.018 | 0.009 | 0.005 | 0.0032 | 0.0019 | 0.0013 |
| **widest dead band** | **0.107** | 0.043 | 0.037 | 0.014 | 0.009 | 0.007 |

At N = 16 more than a tenth of the slider's travel produces no change at all. This is not a defect to
hide behind a coarser slider step — it is what a staircased outline *is*, and it belongs on the panel
next to the crossing interval, because the crossing's bracket width **is** this quantum.

---

## 4. THE BLOCKING TRAP: the picture can split into two guitars while the plate is one

`_decimate_field_mask` point-samples: `mask_full[::stride, ::stride]`. Subsampling a **convex** shape
is benign — that is why the drumhead never found this. A guitar's **waist** is concave, and the row
that joins the two bouts is exactly the row a stride can skip.

Hunted over 53,613 core-accepted configurations (`Lx = 0.37`; `Ly` 0.40–1.0; `asym` 0, 0.3, 0.6;
N = 16…80; waist 0.40–0.99):

- **128 render as two disconnected lobes** while the solver has one connected plate.
- **2,865 render the waist as a single pixel.**

*Re-hunted after the build, inside the viewer's own slider ranges* (`Lx` 0.15–0.80, `Ly` 0.20–1.00,
`asym` 0–0.8, N = 16…80, **waist ≤ 0.88** — the shipped cap): **249 of 140,349** reachable
configurations split under point-sampling, and **0** under pooling. The first hunt's splits all sat
past waist 0.93, which made it tempting to record the guard as spare once the cap was set at 0.88.
It is not: a narrow body with a deep waist reaches it with no exotic parameter at all
(`Lx = 0.15`, `Ly = 0.70`, N = 33, waist 0.88 is the smallest such case, and it is now a test).

Every detector stays green through this. Energy conserves (geometry-blind, twice recorded).
The nullspace is 3-dimensional, because the *solver's* plate is fine. The audio is correct. The core's
own `_count_components` guard passes, because it guards the solver mask. Only the picture is wrong,
and it is wrong in a way that looks deliberate — two lobes read as a design, not a bug. Same failure
class as batch 18's transposed slice decode.

### 4.1 The fix is a proof, not a retry loop

**Max-pool the display mask instead of point-sampling it:** a display cell is live iff **any** solver
node in its `s × s` block is live.

This cannot split a connected plate, and the argument is two lines: any 4-connected path in the
solver mask visits nodes differing by one in a single coordinate, so their blocks differ by zero or
one in that coordinate, so the block sequence is itself a 4-connected path of live display cells.
Connectivity is preserved by construction — there is no parameter to tune and no case to miss.

Confirmed rather than assumed: over the split-prone slice, **35 point-sampled splits → 0 pooled
splits**, narrowest pooled waist 1 px (pooling preserves the join; it cannot manufacture resolution).

**The disk is left alone, and that is a decision with a reason.** Pooling never splits the disk
either (checked at N = 32…200), so the membrane has no bug here — a convex outline cannot lose its
join. Pooling *would* change its rendering (40–85 extra live cells at stride ≥ 2), and #5o's
precedent is that perturbing a shipped number for no correctness gain is a real cost. So the pooled
path is the guitar's, and the membrane's stays as shipped.

### 4.2 The field's semantics, decided explicitly

Pooling the mask without deciding the field would render every block whose representative node is
dead as a live cell holding `0.0` — a fake node, at exactly the waist where it matters.

**Rule: point-sample where the representative node is live; fall back to the block's live-node mean
only where it is not.** Consequences, stated because `field_amp` drives the colour scale:

- On a rectangle every representative is live, so this is **bit-identical to today** and `plate`'s
  shipped payload does not move.
- In the guitar's interior it is likewise identical. Only rim and waist cells take the fallback.
- Every displayed value is either a solver value or a mean of solver values — never a zero standing
  in for "no data".

### 4.3 And the guard stays anyway

The pooled mask's component count is asserted in the payload path regardless of the proof. A proof
about the code as written is not a proof about the code as edited.

---

## 5. THE SECOND TRAP: the shipped strike cannot express the claim

The claim is carried by mode 1 changing parity. A raised-cosine strike centred on the plate's centre
line has **exactly zero** overlap with every odd mode — so past the crossing the fundamental would be
struck with zero amplitude while the modal panel says the modes swapped. The picture and the sound
would silently disagree with the claim, and every ledger would stay green.

This is not a corner. Measured modal amplitudes at the shipped default waist 0.42 (N = 40, the
fundamental is the odd/twist mode there):

| `pluck_x` | 0.30 | **0.40 (plate's shipped default)** | 0.45 | 0.48 | 0.50 |
|---|---|---|---|---|---|
| `|a₁|` | 8.12e-6 | **4.19e-6** | 2.07e-6 | 8.22e-7 | **1.7e-14** |
| `|a₂|` | 4.43e-5 | 5.04e-5 | 5.20e-5 | 5.25e-5 | 5.26e-5 |

At the plate's own default the fundamental is already **12× weaker** than the second partial; one
slider step from `0.4` to `0.5` and it is gone outright. A pickup on the centre line nulls it too
(mode-1 reading −2.4e-10 against mode-2's +2.80).

### 5.1 And there is NO strike point that works on both sides of the crossing

This is the part the trap's first statement missed. `pluck_y` matters as much as `pluck_x`, and it
matters in the *opposite* direction, because the two branches have different nodal geometry along the
length. `|a₁| / max_{j≤4} |a_j|`, `pluck_width` = 0.45:

| `pluck_x`, `pluck_y` | waist 0.15 (bender) | waist 0.42 (twist) | waist 0.60 (twist) |
|---|---|---|---|
| 0.25, 0.55 (near the waist) | **1.000** | 0.241 | 0.306 |
| 0.40, 0.55 (near the shipped default) | 0.476 | 0.083 | 0.095 |
| 0.25, 0.35 (in the upper bout) | 0.371 | **1.000** | **1.000** |
| 0.18, 0.35 | 0.479 | **1.000** | **1.000** |
| 0.40, 0.35 | 0.023 | 1.000 | 1.000 |

A strike at the waist shows the bender and hides the twist; a strike in a bout does the reverse.
⇒ **Ship the compromise and show the number, do not hide the conflict.** The guitar's defaults become
`pluck_x = 0.25`, `pluck_y = 0.35` (0.371 / 1.000 / 1.000 — the fundamental is audible on both sides
and dominant on one), and the claim panel reports **the strike's measured overlap with each branch**
alongside the frequencies. A user who drags the strike onto the centre line then watches the number
go to zero instead of watching the claim quietly stop being true.

---

## 6. What the plate reports about its own outline, and why it must be on screen

`Plate` already exposes `n_pruned`, `area`, `outline_area`, `area_deficit` and `prune_depth_max` on
every plate. #5g is emphatic that the area correction is **reported and never silently applied** — so
omitting it from the panel would be the same mistake in the other direction. Measured on the shipped
default guitar (`Lx = 0.37`, `Ly = 0.48`, waist 0.42):

| N | 16 | 24 | 32 | 40 | 48 | 64 | 80 |
|---|---|---|---|---|---|---|---|
| live nodes | 186 | 416 | 763 | 1175 | 1687 | 3026 | 4725 |
| fill of the bounding box | 49.7% | 52.0% | 53.8% | 54.1% | 54.6% | 55.4% | 55.6% |
| **area deficit** | **−21.6%** | −15.5% | −10.8% | −9.3% | −7.7% | −5.4% | −4.6% |
| pruned nodes | 4 | 2 | 2 | 2 | 2 | 2 | 2 |
| deepest prune (`h`) | 0.769 | 0.739 | 0.711 | 0.712 | 0.713 | 0.706 | 0.703 |

At the coarse end the plate the viewer is simulating is **a fifth smaller than the guitar drawn on
screen**, and its frequencies are correspondingly high. A panel that does not say so lets a coarse,
unconverged plate look finished.

---

## 7. Cost, and the defaults are not on a cliff

The guitar fills ~50–56% of its bounding box, so it costs about **half** a rectangle at the same `N`.
Against the existing `PLATE_WORK_MAX = 7.0e8` at μ = 1 and 2 s of audio, N ≤ 40 fits; higher `N` is
reached by raising μ (fs = κ/(μh²)), which is the plate's existing knob and its existing hint text.

The claim sweep is cheap: 24 waist points at N = 40 is **2.07 s** (16 points, 1.40 s; 32 points,
2.71 s). It is a separate short-grid pass, decoupled from the audio run — batch 18's pattern.

**The shipped default `waist = 0.42` sits well past the crossing** at the default aspect, so the
plate opens in the twist regime and the crossing is something the user slides *back* to find. Unlike
batch 19's amplitude sliders, this default is not on a cliff, and the batch should say so rather than
leaving the reader to wonder.

---

## 8. Touch list

- `web/serialize.py` — a `guitar` model id routed through the existing plate builder (so `plate`'s
  payload is untouched); `_pool_field_mask` (§4.1/§4.2); the outline diagnostics block (§6); the
  waist-sweep claim block (§2) with the branch-overlap numbers (§5.1); the pooled-mask connectivity
  assert (§4.3).
- `web/static/index.html` — the model option, the `waist` / `asym` sliders, the outline panel.
- `web/static/app.js` — `MODEL_RANGES.guitar` (N max 40 at μ = 1 — the shared slider defaults to 128
  and `PLATE_N_MAX` is 80, so an entry is mandatory, not optional); the claim panel renderer; the
  centre-line strike warning.
- `tests/test_web_backend.py` — the tiers of §9.
- **Not** `physsynth/core/` — this batch adds no physics. If it needs a core edit, that is a signal
  the batch has drifted.

## 9. Validation tiers

1. **The parity flip** (the claim). Exactly one flip across the sweep; parity is ±1 to 1e-9 either
   side. This is the tier that can fail.
2. **The display mask is connected** wherever the solver's is — asserted on the split-prone
   configurations the hunt found, which point-sampling fails and pooling passes.
3. **The rectangle payload is unchanged.** `plate` with `outline = rectangle` must be *bit-identical*
   to today's `plate` — the pooling rule reduces to point-sampling when every node is live, and #5g's
   own §9.1 is the precedent for measuring that rather than asserting it (a one-grid bit-identity is
   not measured; run it across the grids and both boundaries).
4. **The reported diagnostics match the core**, so the panel cannot drift from `Plate`.
5. **Energy** — regression tier only, and the test says so in its own docstring. It is geometry-blind,
   recorded three times now (#4, #5of, #5g).

## 10. Refusals

- **No supported guitar.** The core refuses it with a reason (`B = L @ L` makes its spectrum the
  membrane's squared); the viewer surfaces the refusal rather than routing around it.
- **No clamped rim.** #5g §7's refusal stands; it is a core batch, not a viewer one.
- **No von Kármán outline.** Likewise.
- **No area-corrected frequencies.** Reported, never applied — #5g §5.2's rule, and applying it in
  the viewer would be applying it silently at one remove.

---

## 11. Build record (2026-08-26)

Everything above this line is the plan as probed. This section is what building it actually found.

### 11.1 It is a DOMAIN of the plate, and the codebase had already answered that

§8's touch list assumed a new `guitar` model id. Building it exposed why that is wrong *here*, and
the reason is mechanical rather than aesthetic: `Lx` and `Ly` are gated by `data-domain`, which hides
them outright for any model with no secondary select — and the guitar needs both. The plate's select
already carries `supported | free`, so the outline became its third value.

That reads as a mixed axis (a shape where the other two are boundaries) until you notice **the three
options are exactly the three plates that exist**: §10's refusal means "guitar" can only ever mean
"guitar, free". The membrane's select is a shape too. `payload.model` stays `"plate"`, and §9's tier
3 was measured rather than asserted — **bit-identical across 20 configurations** (2 boundaries ×
5 grids × 2 μ) against `git show HEAD:web/serialize.py`, with `outline` and `meta.outline_info` the
only new keys.

**And one invariant had to be repaired to do it, which is the part worth keeping.**
`hasRegimeRanges(model)` re-ranges sliders on *any* domain change as soon as *any* `model:regime` key
exists — while the plan file's own comment says a plate supported→free switch must NOT reset the
user's sliders. Adding `plate:guitar` would have broken that **silently**: nothing fails, the sliders
just quietly snap back on a boundary switch. The gate is now "re-range only when the regime being
**entered or left** declares ranges", which costs one tracked variable. The generalisation is that a
mechanism keyed on the *model* cannot express an invariant stated about a *regime*, and the first new
regime is where you find out.

### 11.2 The blocking trap was real, and the fix is a proof

§4's hunt held up: **128 of 53,613** core-accepted configurations render as two disconnected lobes.
Max-pooling killed every one (35 → 0 over the split-prone slice), the disk is unaffected, and the
component count is asserted in the payload path anyway.

The test for it needed three attempts, and the failures are the same failure each time — *a test that
passes without testing what it claims*, which is the exact family §9.6 of the #5g plan is organised
against. A 7×5 toy with a one-node isthmus: point-sampling did **not** sever it (the bouts either
side of the skipped row are themselves adjacent after sampling). A diagonal neck: not 4-connected to
begin with. What actually severs is a neck **one node wide sitting on an odd column** — missed by
point-sampling at every stride above 1. And the toy has to be big enough that the display budget
picks a stride above 1 at all, or the helper is a no-op and the test asserts nothing. Both are now
asserted in the test body, including the stride it got.

### 11.3 What shipped

- **`_pool_field_mask`** (§4.1/§4.2), **`_display_components`**, **`_outline_block`** (§6),
  **`_mirror_parity`**, **`_guitar_claim_block`** (§2) and **`_waist_quantisation`** (§3) in
  `web/serialize.py`; the outline threaded through `_build_plate`.
- The claim panel `drawWaistCrossing` — two sorted branches with the crossing drawn as a **band**,
  parity encoded in the marker fill (with a legend, because the sorted curves *touch* rather than
  cross and without it the picture reads as the opposite of the claim), and the **strike-overlap bar
  track** of §5 under the plot.
- `updateGuitarHint`, which fires the centre-line warning **before** a render is paid for.
- **16 tests**, group `web_guitar`. Two of them say in their own docstrings that they are not
  evidence: the energy ledger (geometry-blind, recorded now for the fourth time) and — implicitly —
  anything downstream of it.

### 11.4 Verified in a real browser, not only in the payload

Rendered at `?model=plate&domain=guitar`: the outline draws as a guitar with the exterior as panel
background, the claim panel shows the crossing band with `you` at 0.42 on the twist side, and the
energy panel reads *conserved*, drift 1.08e-13. Re-rendered at `waist=0.15&pluck_x=0.5`: the sidebar
hint turns **red** with the centre-line warning, the readout switches to "a long BENDER (even under
the mirror)", and the strike track goes flat and red across the whole sweep — the picture of a claim
that is true and cannot be heard, which is precisely what §5 exists to make visible.

### 11.4a Four gaps a second advisor pass found after the batch was working

All four were invisible to the 17 tests that were green at the time, which is why they are recorded
rather than quietly fixed.

1. **The two panels quoted different Hertz for the same plate.** The spectrum panel marks the
   θ-scheme's *discrete* eigenfrequencies; the claim sweep returned the *continuum* limit (no `k`, no
   `θ`). At the guitar's shipped μ = 2 that is 0.01 % and invisible — but the plate Courant slider
   runs to **32**, where it is **3.0 %, or 51 cents**. Two numbers on one screen disagreeing with
   each other, with nothing failing. The sweep now runs at the shipped plate's own μ and through
   `discrete_beam_eigenfrequency`, and the claim block carries the **shipped plate's own row**
   (`you`) computed on the shipped plate rather than read off the nearest sweep sample — so the two
   panels are tied together by construction. Pinned at both ends of the slider.
2. **`scripts/verify_web_headless.py` carries an enumerated case list**, and the batch had not
   touched it. It had `plate_supported` and `plate_free`; batch 19 closed a never-run-cases gap two
   batches wide, and this would have reopened it at width one on the batch after. Added.
3. **The no-crossing branch said the opposite of the truth.** With no flip in the sweep the readout
   fell through to "right at the crossing" — when in fact there is no crossing at all. Reachable, and
   §2.2 says exactly how: the crossing waist climbs with elongation (0.567 already at aspect 1.68),
   so a long enough body pushes it past the slider's cap. `none` is now its own value with its own
   sentence, and it is a test.
4. **Every render and every test used one aspect ratio.** The claim's single strongest sensitivity
   had never been exercised — which is *why* (3) survived. Two more aspects are now asserted: a long
   body (no crossing, and the panel says so) and a near-square one (crossing below 0.30).

**One known cost, recorded rather than fixed.** `_pool_field_mask`'s fallback branch allocates two
full float copies of the animation buffer, where the point-sampling path took a view — and on a
guitar the branch always fires. At a near-`MAX_FRAMES` run that is a transient few hundred MB. Fine
for a local dev tool; not fine to rediscover as a mystery.

### 11.5 Still out

- The **clamped** rim and **von Kármán on an outline** — both #5g §7 refusals, both core batches.
- ~~The display split is plausibly unreachable through the viewer.~~ **Wrong, and measured wrong
  after the build.** §4 now carries the corrected hunt: **249 of 140,349** configurations inside the
  viewer's own slider ranges, at waist ≤ 0.88, render as two guitars under point-sampling. The guard
  is load-bearing, not spare. The tempting inference — "the cap excludes the failures the hunt
  found" — was drawn from a hunt run at *one* `Lx`; widening it to the sliders' range broke it
  immediately. A guard that is only argued for is a guard that is being talked out of.
