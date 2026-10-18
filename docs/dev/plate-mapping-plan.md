# Parameter mapping, part two — musical controls for the plates

`docs/dev/scientific-hurdles.md` §12, second piece. The first piece
(`docs/dev/parameter-mapping-plan.md`) gave the damped string pitch, inharmonicity and two ring
times; §12 named the plate's "size / thickness / material" vocabulary as the next candidate.

**Status: PLANNED 2026-10-18** — Part 0 next.

**The human's four decisions, 2026-10-18:**

1. **Plates: all three the viewer offers** — the simply supported rectangle, the free rectangle,
   the guitar outline.
2. **Direction: both, behind a switch.** Material and size are always set; then *either* the
   thickness (pitch is read out) *or* a target pitch (the thickness is worked out). The pitch
   target is corrected to the simulation, as the string's is.
3. **Ring time: add "highs fade faster".** The plate today has one loss coefficient, so one ring
   time. This batch adds a second, frequency-dependent loss term to the plate **model** (§2), so a
   plate takes two ring times like the string.
4. **Materials: metals, glass *and* wood with grain** — isotropic presets plus orthotropic woods.

---

## 1. What a musician sets, and what the model needs

| control | meaning | drives |
|---|---|---|
| material | a preset (§4) or custom constants | `E` (or `E_L, E_R, G_LR, ν_LR`), `ν`, density `ρ_v` |
| `Lx`, `Ly` | size (the guitar's width and length) | unchanged — already inputs |
| set by | `thickness` or `pitch` | which of the next two is the input |
| thickness `t` (m) | — | `κ = sqrt(D_x / ρ_s)`, `ρ_s = ρ_v t` |
| `f0` (Hz) | pitch of the lowest elastic mode | `κ`, then `t` from `κ` and the material |
| `t60_low` (s) | ring time of that mode | `σ₀` (mostly) |
| `t60_high` (s) | ring time at the reference frequency | `σ₁` (mostly) |
| `f_ref` (Hz) | where "high" is measured; picks the anchor mode | — |

`N`, `fs`, `θ`, the outline's `waist` and `asym` stay inputs. The raw constructor stays as it is:
the mapping is an optional helper returning the raw coefficients (the human's rule since model
#9: realism is offered, never imposed).

**Every grain ratio is independent of the thickness** — `D_x, D_y, D_1, D_xy` all scale as `t³`
and are reported as ratios to `D_x`. So the thickness moves `κ` and `ρ_s` and nothing else: the
mode *shapes* of a given material and outline do not depend on `t`, which is what makes §3 exact.

**Out of scope:** the von Kármán plate (gong, cymbal) — its anchor bar (a linear `VkPlate` is
bit-identical to `Plate`) holds at `σ₁ = 0`, and musical controls for it are a later batch; the
membrane; the circle outline (core has it, the viewer does not offer it); smoothing a moving
control; learned mappings.

---

## 2. The model change — a frequency-dependent loss on the plate

Today: `u_tt = −κ² ΔΔu − 2σ u_t`, with `σ` entering `A = (1 + σk)I + θk²κ²B` and the right-hand
side as `σk u⁻`. The change is Bilbao's second loss term:

```text
u_tt = −κ² ΔΔu − 2σ₀ u_t + 2σ₁ Δ u_t
```

`σ` keeps its name and meaning (it *is* `σ₀`); `σ₁ ≥ 0` is new, default `0`. The term is centred
in time, `2σ₁ Δ (u⁺ − u⁻)/(2k)`, so it adds `−σ₁ k Δ` to `A` and `−σ₁ k Δ u⁻` to the right-hand
side, keeps `A` symmetric positive definite, and dissipates `σ₁ ‖∇ δ_t· u‖²` per step — passive by
construction. The energy is unchanged; the **balance** (energy drop = dissipated work, per step,
to rounding) is the bar.

**Why the Laplacian and not the stiffness.** For a plate `ω ≈ κ p²`, so `σ_m = σ₀ + σ₁ p²` grows
*linearly* with frequency — the standard choice (Bilbao, *Numerical Sound Synthesis*, the plate
chapter). A loss proportional to the stiffness would grow as `ω²`.

**Where the Laplacian comes from, per branch:**

- **Supported.** The masked Dirichlet Laplacian `L` is already built (`Params.laplacian`) and
  `B = L·L`, so `L` and `B` share every eigenvector *exactly*: each mode stays a mode, with
  `σ_m = σ₀ + σ₁ λ_m`, `λ_m` the mode's `−L` eigenvalue. Damping is proportional; §3 is exact.
- **Free and guitar.** No Laplacian exists for a free edge. The natural one is the Dirichlet
  energy `∫|∇u|²` assembled cell by cell on the live mask, the same quadrature `K` and `W` use,
  with the free (natural) boundary — positive semidefinite, constants in its null space (a rigid
  translation is not damped by `σ₁`, only by `σ₀`). It does **not** share eigenvectors with `K`:
  the damping is non-proportional and the modes couple, weakly. **This is Part 0's main
  question** (§7): how far off target does §3's one-mode inversion land when the decay rate is
  taken as the mode's Rayleigh quotient `φᵀGφ / φᵀWφ`? Light loss makes that first-order exact,
  so the error should be second order in the loss — Part 0 measures it. If it is not small enough,
  the options go to the human: refine by iteration, accept and report, or use a
  stiffness-proportional loss on the free branches (exact, but a different frequency law).

**Nothing that exists may move.** `σ₁ = 0` must leave every plate trajectory, the 588-request
viewer freeze and the `VkPlate` anchor bit-identical. So the term is **skipped**, not multiplied
by zero, when `σ₁ = 0` — `A`'s sparsity pattern (and hence its factorization) stays the one it is
today, and no `0.0 · x` reaches a `-0.0` or a NaN. Of the 66 `PlateSpec` literals, all but two
files use `..Default::default()`; `mallet.rs` and `mallet_plate_harness.rs` get the field by hand.

---

## 3. The inversion — exact, linear, three unknowns

The θ-scheme with both loss terms gives each mode `m` (stiffness eigenvalue `β_m` of `B`, or of
`(K, W)`; loss eigenvalue `λ_m`) the string's recurrence with

```text
X_m = k² κ² β_m          Y_m = k (σ₀ + σ₁ λ_m)
a = 1 + θX + Y,   b = −2 + (1 − 2θ)X,   cc = 1 + θX − Y
```

and `β_m`, `λ_m` do not depend on `κ`, `σ₀`, `σ₁` or the thickness (§1). A target `(f, T60)` is
the root pair `r e^{±iφ}`, `φ = 2πfk`, `r = 10^(−3k/T60)`, and both `cc/a = r²` and
`−b/a = 2r cos φ` are linear in `(X, Y)` — so linear in `(κ², σ₀, σ₁)`.

**The plate has one stiffness constant, not two.** The string fit `c²` *and* `κ²` to two pitches;
the plate's whole spectrum moves with `κ` alone, so the anchor mode's pitch is not free — the
geometry and material decide it. The equations are therefore:

- **set by pitch:** mode 1's pitch and ring time (two equations) and the anchor's ring time
  (one) — three linear equations in `κ², σ₀, σ₁`. Exact at both anchors where §2's damping is
  proportional. The thickness then follows: `t = κ sqrt(12 ρ_v (1 − ν_xy ν_yx) / E_x)`.
- **set by thickness:** `κ` comes forward from the material; the two ring times give `σ₀, σ₁` by
  a 2×2 solve; the pitch of every mode is a **read-out** of the same recurrence.

"Mode 1" is the **lowest elastic mode**: on the free branches the three rigid motions (`β = 0`)
are skipped. On the guitar, the waist decides whether that mode bends or twists; the inversion
does not care which. The anchor `M` is the mode whose read-out frequency is nearest `f_ref`; a
degenerate pair (the square's `(1,2)` and `(2,1)`) shares `β` and `λ`, so either serves.

Supported: `β_m = λ_m²`, both closed form. Free and guitar: one generalized eigen-solve of
`(K, W)` at `κ = 1` per geometry (the viewer already runs one for its spectrum panel), plus the
Rayleigh quotients of `G` (§2).

---

## 4. Materials

Isotropic presets — steel, brass, bell bronze, aluminium, glass — each `(E, ν, ρ_v)`; and
orthotropic woods — spruce, maple and at least one more — each `(E_L, E_R, G_LR, ν_LR, ρ_v)` fed
to the existing `plate::grain_ratios_from_material`, plus "custom" for either kind. **Every value
is taken from a cited source in Part 1** (the USDA *Wood Handbook* for the woods; a standard
engineering table for the rest), with the citation beside the table in code.

- **Grain direction.** The guitar's length runs along `y` (`guitar_mask(…, ly, lx, …)`), and a
  top's grain runs along its length, so the woods enter with `E_x = E_R`, `E_y = E_L` there. The
  rectangles take a grain-direction choice (along `x` or `y`).
- **Free branch with grain** needs the split (`grain_coupling`, `grain_torsion`), which
  `grain_ratios_from_material` returns; isotropic material keeps the `nu` path.
- **Areal density is the trap** `GrainSpec`'s doc names: passing `ρ_v` where `ρ_s` belongs leaves
  every pitch right and every energy wrong by a factor of `t`. A bar must read a quantity that
  depends on `ρ_s` for a fixed strike, and a planted `ρ_v` must turn it red (§5).
- A steel plate with a guitar outline is physically odd and allowed.

---

## 5. Validation — the bars

Every bar runs the simulation, never the mapping's algebra backwards.

1. **The new loss term, before any mapping.** Balance to rounding per step on all three branches
   with `σ₁ > 0`; monotone decay; on the supported branch each mode's measured decay equals
   `σ₀ + σ₁ λ_m` through the recurrence; `σ₁ = 0` bit-identical to today (state, energy, `A`).
   A start-up bar of its own (CLAUDE.md: several first-step defects hid from every conservation
   bar).
2. **Round trip, single mode,** both directions, all three branches: start in exactly mode 1 (then
   `M`), project each step, read `r` and `φ` in the well-conditioned form (second difference on
   `(a_n, a_n − a_{n−1})`, `ln_1p`). Tolerance stated per branch once Part 0 has measured the free
   branches' coupling.
3. **Round trip, struck,** through the analysis crate's spectrum detector.
4. **Off every default:** `Lx ≠ Ly`, `N`, `fs`, `θ ∈ {0.25, 0.28, 0.5, 1}`, `waist`, `asym`, every
   preset, both grain directions. Grep the plate fixtures for each range first.
5. **Textbook control:** the continuum `κ` from `f₁₁ = (πκ/2)(1/Lx² + 1/Ly²)` and the continuum
   T60 law miss by a measured amount, and the bar asserts the miss exists.
6. **Refusals,** one bar each (§6).
7. **Thickness round trip:** set by pitch → `t`; set by that `t` → the same pitch.

**Plants** (each red, each time-limited): swap `σ₀`/`σ₁`; drop the `σ₁` term from `A` but not the
right-hand side (and the reverse); continuum `λ_m`; anchor `M ± 1`; a rigid mode taken as mode 1;
`ρ_v` passed as `ρ_s`; `E_L`/`E_R` swapped; hard-wired `θ`; `k = fs`.

---

## 6. Refusals

`Err` with a plain reason when: a 2×2 / 3×3 is singular or gives `κ² ≤ 0`, `σ₀ < 0`, `σ₁ < 0`
(ring time rising with frequency); the anchor is mode 1 (`f_ref` at or below the fundamental); a
target pitch is beyond what the scheme can produce at this `fs` and `θ`; any input non-positive or
non-finite; the material is inadmissible (`MaterialError`). The drift of the other partials is
**reported, not refused**, as the string's is.

---

## 7. The viewer

**Backend** (`crates/physsynth-viewer/src/plate.rs`): the plate's `domain` key already means
supported / free / guitar, so the control set gets **its own key**, `controls: physical |
musical`, and in musical mode `set_by: thickness | pitch`. A request without `controls` takes
today's path byte for byte. In musical mode `fs` is an explicit input and `mu` a read-out; the
existing cost ceilings (`PLATE_MU_MAX`, `PLATE_WORK_MAX`, `PLATE_NLIVE_MAX`) still refuse. The
payload gains a `tuning` block: material, thickness, `ρ_s`, `κ`, `σ₀`, `σ₁`, the anchor, targets
against measured partials.

**One drift read-out, not two.** The string batch found that its tuning hint and the resolution
strip measured against different references and named different horizons. The plate has no
stretched-law target, so the off-anchor pitch read-out **is** the existing horizon strip (the
continuum plate with the mapped constants); only the ring times at partials other than the two
anchors get a new read-out.

**The coverage guard.** `crates/physsynth-viewer/tests/frozen.rs` reads only each model's one
`DOMAIN_OPTS` select, so a second control on the plate would escape it. The guard is extended to
read the new control-set select too, and the new combinations go into `BORN_AFTER_THE_FREEZE`
with the test file that covers them — the human's rule from the string batch.

**Front-end:** a "musical controls" select on the plate panel; material select (with grain
direction for woods); thickness-or-pitch switch; read-only derived values. `verify_headless`
gains a musical-plate case per branch.

---

## 8. Order of work

- **Part 0 — probe** (scratch crate under `W:\temp\claude`, nothing in the repo): (a) `σ₀`-only
  inversion at mode 1 through the real `Plate` on all three branches — exact where the loss is
  proportional to the mass, which `σ₀` is; (b) a scratch copy of the step with §2's term, the
  supported branch exact, and the free branches' first-order error measured over a grid of
  loss, `N`, outline and material; (c) the textbook miss. Anything surprising goes to the human.
- **Part 1 — model:** `σ₁` on `Plate` (both branches, the free Laplacian), its bars (§5.1).
- **Part 2 — materials:** the cited preset tables.
- **Part 3 — mapping:** the helper, both directions, refusals, bars §5.2–7, plants.
- **Part 4 — viewer backend,** guard extension, new viewer tests, freeze unchanged.
- **Part 5 — front-end** and the headless browser check.
- **Part 6 — docs:** hurdle §12, HANDOFF §8, memory.

Commit and push at each part's end, `scripts/cargo-test-nice.ps1 -Full` before each.
