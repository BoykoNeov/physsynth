# Parameter mapping, part two — musical controls for the plates

`docs/dev/scientific-hurdles.md` §12, second piece. The first piece
(`docs/dev/parameter-mapping-plan.md`) gave the damped string pitch, inharmonicity and two ring
times; §12 named the plate's "size / thickness / material" vocabulary as the next candidate.

**Status: DONE 2026-10-18** — Parts 0–6 (§9–§14); the human chose the Laplacian loss the same day.
Nothing queued from this plan.

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

**Clients that rebuild the step must carry `σ₁` or refuse it** — the `σ₁ = 0` bit-identity bar
cannot see a client that silently drops a non-zero `σ₁`. Found by grep, 2026-10-18:

- `airbox_wrap::theta_matrix` rebuilds `A` from `plate::Params` for the room seam (both the
  linear and the von Kármán seam). It takes the right-hand side from `plate::step_rhs`, so with
  `σ₁ > 0` it would carry the term on one side and not the other. **Carries it** (one shared
  spelling of `A`, which `theta_matrix`'s own doc already insists on).
- `mallet::PlateParams` uses the plate's own factorization and `step_rhs` — carries it for free;
  a bar asserts it (a struck plate with `σ₁ > 0` decays as the bare plate does).
- `plate::initial_previous` ignores `σ₀` today (the start-up is the lossless Taylor step); `σ₁`
  is treated the same way, and the start-up bar says so.
- `connection` (string ↔ plate bridge) steps the plate through `Plate::step`, so it carries the
  term; its `plate_drive_compliance` is a stability guard on `G0 = M + (θ − ¼)k²κ²S`, where loss
  does not enter (its own doc: a dissipative term enters the step operator, not `G0`).
- `VkPlate` and the von Kármán room seam never see `σ₁`: `VkSpec` has no such field and builds its
  linear part with `σ₁ = 0`. The viewer's `platebody` passes `sigma` only, so `σ₁ = 0` there too.

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
  rectangles take a grain-direction choice (along `x` or `y`). **Turning the grain turns the
  Poisson ratio too:** `grain_ratios_from_material` takes the major ratio `ν_xy`, which with the
  grain along `y` is `ν_RL = ν_LR E_R / E_L`, not `ν_LR`. Passed unrotated, spruce's
  `1 − ν_xy ν_yx` goes negative and the plate is refused as inadmissible. A plant.
- **Grain and an isotropic `σ₁` scatter the ring times.** Two partials at the same pitch can have
  different `λ_m`, so the off-anchor ring-time read-out is a scatter, not a curve.
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
`ρ_v` passed as `ρ_s`; `E_L`/`E_R` swapped; `ν` not rotated with the grain; hard-wired `θ`;
`k = fs`.

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

- **Part 0 — probe** (scratch crate under `W:\temp\claude`, nothing in the repo):
  - (a) `σ₀`-only inversion at mode 1 through the real `Plate` on all three branches — exact
    where the loss is proportional to the mass, which `σ₀` is.
  - (b) Two cheap checks first: does `L` commute with the **grained** supported `B` (built by
    `orthotropic_biharmonic`, not `L·L` — if not, "exact on supported" holds only isotropically);
    how far from commuting are `W⁻¹K` and `W⁻¹G` on the free branches.
  - (c) The free branches' error **measured on the scheme's own root** — the quadratic
    eigenproblem's root nearest the target, by inverse iteration on small dense grids — not by
    projecting a run onto a `(K, W)` mode, which mixes neighbour leakage into the answer. Stress
    the worst case: anchors inside dense mode clusters, near-square rectangles, the guitar near
    its waist crossing. Coupling between near-equal modes is first order, not second.
  - (d) Rigid tilt: `x` and `y` are not in `G`'s null space, so `σ₁` damps a tilt and a strike
    pumps energy into it. Measure how much.
  - (e) Which textbook miss is well above measurement — mode 1's pitch (one `κ`, likely tiny) or
    the anchor's ring time. The control differs per plate: `f₁₁` supported, Leissa's table for
    the free square, nothing for the guitar.
  - The result goes to the human as **two columns** — Laplacian loss (constant loss factor,
    inexact on free edges, damps tilt) against stiffness-proportional loss (exact everywhere,
    rigid motion untouched, ring time falling as frequency squared). Not pre-decided.
  - Choose `M` from the **lossless** spectrum and record its margin to the next mode (a discrete
    output). Skip the rigid modes by count — three, after `count_components == 1` — with a check
    on the gap above them, not by a threshold.
- **Part 1 — model:** `σ₁` on `Plate` (both branches, the free Laplacian), its bars (§5.1).
- **Part 2 — materials:** the cited preset tables.
- **Part 3 — mapping:** the helper, both directions, refusals, bars §5.2–7, plants.
- **Part 4 — viewer backend,** guard extension, new viewer tests, freeze unchanged.
- **Part 5 — front-end** and the headless browser check.
- **Part 6 — docs:** hurdle §12, HANDOFF §8, memory.

Commit and push at each part's end, `scripts/cargo-test-nice.ps1 -Full` before each.

---

## 9. Part 0 result (2026-10-18)

Probe: a scratch crate depending on core by path, nothing in the repo changed. It builds the real
`plate::Params`, adds §2's loss operator (`−L` supported; the cell-assembled Dirichlet energy `G`
free), inverts the three targets of §3 with each mode's loss taken as its Rayleigh quotient
`φᵀGφ`, then finds the scheme's **own** root nearest each target by nonlinear inverse iteration
on `T(z) = z²A − zR₁ + R₀` (dense, complex) — no time-stepping, no projection. Eleven plates
(supported square, rectangle and spruce; free square, near-square `0.5 × 0.52`, rectangle and
spruce; guitar at waist 0.20, 0.42, 0.60 and in spruce), `N ∈ {12, 16, 20}`,
`fs ∈ {44.1k, 96k}`, `θ ∈ {0.25, 0.28, 0.5, 1}`, `f0 ∈ {110, 330}`, three loss levels
(`T60` pairs `(4, 1)`, `(1, 0.2)`, `(0.3, 0.05)` s), `f_ref ∈ {6, 15} × f0`; 2,294 runs (anchors
past 40 % of the grid's top mode skipped).

- **Supported — exact, grain included.** The commutator `‖LB − BL‖` is `0` isotropically and
  `≤ 5e-18` relative with spruce grain (`orthotropic_biharmonic` is a Kronecker build, so it
  shares `L`'s eigenvectors). Worst miss over every supported run: mode 1's pitch `1.3e-8`
  cents, its T60 `1.5e-10`, the anchor's T60 `1.7e-11`.
- **Free and guitar — not exact, far below hearing.** Relative commutator `2e-3`–`1.2e-2`
  (shrinking with `N`). Worst miss: mode 1's pitch **0.05 cents** (guitar, `N = 12`; 0.013 on the
  free rectangles), its T60 `4.5e-5`; the anchor's T60 **0.3 %** (free spruce, an anchor in a
  near-degenerate pair: margin to the next mode `≤ 1.2e-3`), `≤ 4e-4` everywhere else. The error
  grows with the loss (light / mid / heavy, free square: `2.6e-7`, `8.6e-7`, `1.4e-5`) — the
  second-order behaviour expected of a first-order (Rayleigh) inversion, except inside clusters.
  Audibility thresholds are ~5 cents (the string batch's warning) and several percent in decay
  time, so the margin is ~100× in pitch and ~10× in ring time at the very worst.
- **The textbook mapping misses a lot.** Supported: mode 1 is **5–17 cents** flat-or-sharp
  (coarser grids worse) and the anchor's T60 **5–11 %** off. Free square (Leissa's 13.468 and the
  continuum loss law `σ = σ₀ + σ₁ ω/κ`): **8–21 cents** and **27–30 %** in T60 — the continuum
  law's `p² = ω/κ` is a supported-plate identity, wrong for the free plate's twist fundamental.
  Unlike the string (0.25 cents at the fundamental), the correction pays at the **fundamental**
  as well as at the anchor.
- **Rigid tilt is damped by `σ₁`.** Translation is in `G`'s null space; a tilt is not. A tilt's
  loss rate is **0.39–0.91×** mode 1's (`σ₁` part), so a struck free plate's slow rotation stops
  where today it drifts on (with `σ₀ = 0`). Mode 1 does not couple to the tilt on the rectangles
  (symmetry: `0.000`); on the guitar the normalized coupling is up to **0.15**. The net momentum
  an elastic vibration hands the tilt over its whole decay is zero (`d/dt (tᵀWu̇) = −σ₁ tᵀGu̇`
  integrates to `−σ₁ tᵀG Δu`, and the elastic `Δu` returns to zero), so this is a sub-audio
  effect on the rigid motion, not on the partials.
- **Rigid modes skip cleanly by count:** the gap `β₄ / max|β₁..₃|` is `≥ 2e10` everywhere.

**The decision this hands the human:** keep §2's Laplacian loss (ring time falling in proportion
to frequency, the constant-loss-factor law of metals and Bilbao's standard; free and guitar plates
inexact by ≤ 0.05 cents and ≤ 0.3 % in T60, reported; tilt damped), or a stiffness-proportional
loss (exact on every plate, rigid motion untouched; ring time falling as frequency **squared** —
with 4 s at 110 Hz and 1 s at 1.65 kHz it gives 0.14 s at 5 kHz where the linear law gives
0.38 s, and 2.4 s at 800 Hz against 1.7 s).

**The human's call, 2026-10-18: the Laplacian loss** (ring time falling in proportion to
frequency). The free branches' inexactness is reported and held to stated tolerances by the bars;
the tilt damping is documented, not removed.

---

## 10. Part 1 result — the loss term (2026-10-18)

`PlateSpec::sigma1` / `Params::sigma1`, refused when negative or non-finite (`NegativeSigma1`);
`Params::loss` holds the form only when `σ₁ > 0`; `plate::loss_form` builds it for any plate
(`−L` supported, `ops2d::free_gradient_form` free). `A` gains `σ₁kG`, `step_rhs` gains `σ₁kGu⁻`,
both absent at `σ₁ = 0`. `airbox_wrap::theta_matrix` carries the same term.

Bars, `crates/physsynth-core/tests/plate_loss.rs` (nine) and one in `tests/airbox_grid.rs`:

- every supported Navier mode follows its scalar recurrence with `σ₀ + σ₁λ`, `λ` from the 1-D
  Dirichlet eigenvalues written in the test, to `1e-9` — isotropic and grained, `θ ∈ {0.28, 0.5,
  1}`, three modes; the same run without `σ₁` misses by `> 1e-3`;
- the energy books close to `1e-10` on all three branches at two `θ`s, the free gradient form
  re-assembled in the test; `σ₁`'s own share of the work is asserted above 5 % of the start;
- `σ₁` alone is passive on all three branches;
- a free translation keeps its speed (`1e-9`), a tilt loses energy — the documented behaviour;
- a free mode follows its Rayleigh-quotient recurrence: worst `4.5e-5` (guitar, mode 15) against
  a `1e-4` bar, `≥ 0.09` without `σ₁`;
- refusals; the form carried only when used; the start-up is the lossless Taylor step either way;
- the mallet's influence column still reproduces a forced step (`1e-14`);
- a room-mounted plate with no air load is bit-identical to the bare plate with `σ₁ > 0`.

**Plants, all red:** the right-hand-side half dropped, sign-flipped, or applied to `u` not `u⁻`;
the `A` half dropped or doubled; the supported form `+L`; the free form doubled, missing its
`y`-edges, or with rows not summing to zero; the room seam dropping its `A` half. The `A`-half
plants were re-run against `plate_loss.rs` alone (the room bar catches them too, and stops cargo
before the rest) — red there as well.

`σ₁ = 0` leaves everything as it was: the full suite, including the 588-request viewer freeze, is
green unchanged (1,850 tests).

---

## 11. Part 2 result — materials (2026-10-18)

`crates/physsynth-core/src/plate_tuning.rs`: `Material` (isotropic, or quarter-sawn wood with
`E_L, E_R, G_LR, ν_LR`), `PRESETS`, `plate_material(material, grain, t)` → `PlateMaterial`
(`κ`, `ρ_s`, and the grain in the form each boundary takes, written by `apply`), and
`thickness_for_kappa`. Isotropic material keeps the plate's isotropic path (`nu`, no grain).

- **Sources.** Woods: USDA *Wood Handbook* FPL-GTR-190 (2010), Tables 5-1, 5-2, 5-3a — Sitka
  and Engelmann spruce, western redcedar, sugar maple, coast Douglas-fir. Two handbook footnotes
  applied: `E_L` = bending modulus + 10 % (shear), density = `SG · 1000 · 1.12` (specific gravity
  is on volume at 12 % moisture). Metals and glass: Engineering ToolBox's Young's modulus,
  Poisson's ratio and density tables; where those give a range the preset takes a value inside it
  and says which (brass 110 GPa of 102–125, bronze 110 of 96–120, glass 70 of 50–90).
- **The grain turns the Poisson ratio:** along `y` the major ratio is `ν_LR E_R / E_L`.
- Bars, `tests/plate_tuning.rs` (seven): every preset builds every plate with its grain either
  way; Sitka retyped from the handbook (both footnotes); the isotropic `κ` and `ρ_s = ρt` written
  out; turning a wood's grain swaps `D_x` and `D_y` and keeps `H`, `D_1`, `D_xy`; the thickness
  round trip; refusals; sources and unique names.
- **Plants, all red:** `ν` not rotated; a volume density handed out; either footnote dropped;
  `E_L`/`E_R` swapped.

---

## 12. Part 3 result — the mapping (2026-10-18)

`plate_tuning::tune_plate(geometry, material, grain, targets)` → `PlateTuning` (material at the
tuned thickness, `σ₀`, `σ₁`, the fundamental's and the anchor's frequencies in the scheme, the
anchor's margin to its neighbours, both modes' `β` and `λ`, and `exact`). `SetBy::Pitch` or
`SetBy::Thickness`; `PlateTuning::apply` writes it into a spec. Supported modes come from the closed
form (no solve); free ones from `eigsh_shift_invert` — four near a small negative shift (three rigid
motions skipped by count, with a `10⁶` gap check) and eight around the anchor's stiffness.

**Mode 1 alone fixes the stiffness.** Its two equations involve only its own `(X, Y)`, so the 3×3
of §3 splits: a 2×2 for mode 1, then one product equation for the anchor's `Y`, then a 2×2 for
`(σ₀, σ₁)`.

**A refusal found by the bars: the high ring time has a range.** With the loss rising in proportion
to `λ`, fixing `t60_low` bounds `t60_high` on both sides — shortest at `σ₀ = 0`, longest at
`σ₁ = 0` — and the refusal (`HighRingTimeOutOfRange`) names both ends. The long end lies a little
**above** `t60_low` (the scheme suppresses a high mode's decay rate, hurdle §4). On the free
rectangle at 180 Hz with 2.5 s low, the anchor near 1.5 kHz can ring no shorter than 0.93 s: the
free plate's twist fundamental has a large loss eigenvalue for its frequency, so its `λ` ratio to
the anchor is smaller than the frequency ratio.

Bars, `tests/plate_tuning.rs` (six more, thirteen in all):

- **supported, 96 fixtures** (steel, spruce both ways; `N ∈ {12, 20}`, `fs ∈ {44.1k, 96k}`, four
  `θ`, both directions): both anchors read back from the running scheme — worst `1.3e-7` cents and
  `2e-11` in T60 — and the anchor is the partial nearest `f_ref`;
- **free and guitar** (steel, spruce along `y`; `θ ∈ {0.28, 1}`): worst `1.9e-4` cents, `1.9e-6`
  in T60, against Part 0's worst-case 0.1 cents / 1 %;
- **struck** brass plate through the analysis crate's detector: both anchors within `5e-4` cents;
- **textbook control**: the continuum `f₁₁` stiffness is **15 cents** flat where the mapping lands
  at `1e-10`;
- thickness round trip (set by pitch → `t` → set by `t` gives the pitch back, `1e-10`);
- refusals, including both ends of the ring-time range landing on the one-loss plates.

**Plants, all red:** `σ₀`/`σ₁` swapped in `apply`; `θ` hard-wired; the continuum `λ`; the anchor
chosen farthest from `f_ref`; a rigid motion taken as mode 1; a 20 dB ring time; the lossless pitch
in place of the 2×2; the thickness from `κ²`; the free `λ` doubled; `σ₀`'s sign. (The material
plants of §11 re-ran with them, red.)

---

## 13. Part 4 result — the viewer backend (2026-10-18)

`crates/physsynth-viewer/src/plate.rs`: a plate request with `controls: "musical"` is built by
`build_musical` from `material` (a preset name, or `custom` with `custom_E_gpa`, `custom_nu`,
`custom_density`), `grain` (`x` / `y`; default `y` on the guitar, `x` elsewhere), `set_by`
(`pitch` with `f0`, or `thickness` with `thickness_mm`), `t60_low`, `t60_high`, `f_ref` and
`sample_rate`, plus the existing `Lx`, `Ly`, `N`, `theta`, `waist`, `asym`, `domain`. Its
`kappa`, `rho`, `mu`, `sigma`, `nu` are ignored; `mu` is a read-out still held to `PLATE_MU_MAX`
(a refusal that says to raise the sample rate or reduce `N`), and `PLATE_NLIVE_MAX` and
`PLATE_WORK_MAX` apply unchanged. Custom material is isotropic only.

- The payload gains a `tuning` block: material, source, grain, how it was set, thickness, `ρ_s`,
  `κ`, `σ₀`, `σ₁`, the fundamental and the anchor in the scheme, the anchor's margin (`crowded`
  below 1 %), `exact`, `mu`, and the pitch and ring time of the lowest eight partials
  (`plate_tuning::lowest_modes`, new in core). Their pitch drift is left to the existing horizon
  strip, as §7 decided.
- The energy verdict: lossless only when both losses are zero, and the single-rate `2σ` oracle
  line dropped when `σ₁ > 0` (each mode decays at its own rate).
- The animation run's plate is now a clone of the freshly built one rather than a second build, so
  the mapping runs once; on the physical path that is the same plate, and the freeze is unchanged.

Bars, `crates/physsynth-viewer/tests/plate_musical.rs` (eight): built from the targets, not the
decoy sliders (and `mu` read out from the mapped `κ`); the rendered audio sounds `f0` (supported
steel, spruce guitar) through the analysis crate's detector; the run's **first and last energy**
equal a core plate built from the mapping and struck the same way (the areal density, and both
losses at the mapped stiffness); musical keys leave a physical request byte-identical; set by
thickness reads the pitch out near the continuum `f₁₁`; a free plate says it is not exact and a
wood reports its grain; the energy verdict without the single-rate oracle; eight refusals.

**Plants, all red:** the slider `κ` read in place of the mapped one; a volume density; `σ₁`
dropped; the switch never matching.

**Moved to Part 5:** the coverage guard's extension. It reads `app.js`, and the musical select does
not exist until the front-end adds it, so the guard and the select land together.

---

## 14. Part 5 result — the front-end and the coverage guard (2026-10-18)

`web/static/`: a second select, **Controls**, shown for the plate and filled from a new
`CONTROL_OPTS` table (`physical` / `musical`). `musicalOn(model)` answers for both models that have
musical controls (the damped string through its domain select, the plate through this one), and
two new gates use it: `data-musical` shows a group only when that model's musical controls are on,
`data-hide-musical` hides a physics slider then (OR-ed onto the earlier passes for an element they
rewrite, recomputed for one they never touch — the latch the `data-hide-domain` pass once had).

- The string's musical group (pitch, both ring times, the reference, the sample rate) is **shared**
  with the plate; inharmonicity stays string-only.
- A **Plate material** group: material (the core's preset names plus `custom`), grain (`auto`, `x`,
  `y`), set by (pitch / thickness), a thickness slider, and three custom-constant sliders. The
  input that is not set hides; the custom sliders show only for `custom`. In musical mode the
  plate's `κ`, `ρ`, `μ`, `ν`, `σ` and the then-empty Physical group hide.
- The read-out under the group: the material and grain, the thickness (or the pitch it gives), `κ`,
  `ρ_s`, both losses, `μ`, the anchor, how exact the simulated plate is, the ring times of the first
  eight partials, and the source.
- A deep link takes `controls`, `material`, `set_by`, `grain`; a value no option carries is reported
  in the page's own notes, which the headless check fails on.
- `fieldset.group` gets `min-width: 0` and its selects `width: 100%`: a wide option text had pushed
  the group past the panel — seen only on the headless screenshot.

**The coverage guard** (`tests/frozen.rs`) now reads `CONTROL_OPTS` as well as `DOMAIN_OPTS`, naming
the plate's options `controls:physical` and `controls:musical`; both are in `BORN_AFTER_THE_FREEZE`,
covered by `tests/plate_musical.rs` (the physical one by its byte-identity bar). Planted — the
musical exemption removed — it goes red naming the pair.

`tests/plate_musical.rs` gains a ninth bar: the material menu in `index.html` is exactly the core's
preset list plus `custom`.

**Headless browser check:** two new cases, `plate_musical` (brass, supported, set by pitch) and
`plate_musical_guitar` (Sitka spruce guitar top, so the read-out must say it is not exact); with
every plate case and `string_damped_musical` (the shared group) — 8/8 green.

**Review fixes (2026-10-18, before calling the batch done):**

- **The viewer bars sent the server's own fallbacks** for every musical key (f0 220, ring times
  2 / 0.5, f_ref 2000, 44.1 kHz, θ 0.28, steel), so a misspelled key or a hard-wired `θ` would have
  passed them all — CLAUDE.md's first rule. Every value moved off its fallback; planted, a
  misspelled `t60_low` and a hard-wired `θ` both go red now.
- **Two inputs had no bar:** the custom material's three sliders and an explicit grain. Each now has
  one against core `tune_plate`, the grain one on the worked-out thickness (spruce along `x` and
  `y` differ by more than 5 %). Planted (a misspelled custom density, `grain: y` read as `x`): red.
- **The viewer had never built a grained plate, and three read-outs assumed it never would:**
  - the supported plate's discrete modes squared the Laplacian's eigenvalues (`B = L·L`);
  - the continuum markers used the isotropic Navier law (and the free square's isotropic Leissa
    table);
  - the guitar's waist sweep rebuilt each plate from `nu` alone, so a spruce top's crossing panel
    described an isotropic one.

  Now a grained supported plate reads `B` itself and takes the orthotropic Navier markers, a grained
  free plate gets no tabulated marker (none exists), and the sweep carries the four free-branch
  constants. Isotropic plates take their old code paths, so the freeze is unchanged. Three bars, each
  planted red.

`tests/plate_musical.rs` has thirteen bars. Full suite green (1,876); headless 3/3 on the musical
and guitar cases.
