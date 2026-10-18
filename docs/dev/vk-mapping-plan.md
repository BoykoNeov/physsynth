# Parameter mapping, part four — musical controls for the gong and the cymbal

`docs/dev/scientific-hurdles.md` §12, fourth piece. The damped string
(`docs/dev/parameter-mapping-plan.md`), the three linear plates (`docs/dev/plate-mapping-plan.md`)
and the drum (`docs/dev/membrane-mapping-plan.md`) have musical controls; §12 names the von Kármán
plate (model #6, the supported gong and the free-edge cymbal) as still raw.

**Status: PLANNED 2026-10-18.** Part 0 next.

**The human's four decisions, 2026-10-18:**

1. **The pitch you set is the settled note** — the one the plate sinks to as it rings out, i.e.
   its gentle-strike (`w → 0`) pitch. Met exactly, by the plate mapping's own inversion. A harder
   strike starts above it, and the page says by how much.
2. **The strike stays "how hard"** — today's strike-strength slider, in plate thicknesses. The page
   reads out how far above the settled note a strike starts. No search for a target bend.
3. **One page:** the gong/cymbal page (`model: "vk"`). The cymbal-in-a-room page stays as it is.
4. **The cymbal gets pitch and ring times too,** the same controls as the gong: its settled lowest
   tone is met (first-order exact, as the free plate's), even where a crash hides it.

Not asked, taken from the plate batch: metals and glass only; set by thickness *or* pitch; two
ring times, through a second loss term the gong does not have yet (§2).

---

## 1. What a musician sets, and what the model needs

| control | meaning | drives |
|---|---|---|
| material | a metal or glass preset (`plate_tuning::PRESETS`, isotropic only) | `E`, `ν`, `ρ_v` |
| size | `Lx`, `Ly` | unchanged — already inputs |
| set by | `thickness` or `pitch` | which of the next two is the input |
| thickness `e` (m) | — | `κ`, `ρ_s`, **and where the bend starts** |
| `f0` (Hz) | the settled pitch of the lowest elastic mode | `κ`, then `e` |
| `t60_low` (s) | ring time of that mode, gentle strike | `σ₀` (mostly) |
| `t60_high` (s) | ring time at the reference, gentle strike | `σ₁` (mostly) |
| `f_ref` (Hz) | where "high" is measured; picks the anchor mode | — |
| strike | amplitude in thicknesses (`w_over_e`, today's slider) | the initial field |

`N`, `fs`, boundary (supported gong / free cymbal), strike position and width, pickup, and the
nonlinearity switch stay as they are. The raw constructor stays as it is: the mapping is an
optional helper returning raw coefficients.

**Why the thickness matters here and did not on the linear plate.** The linear plate takes `κ` and
an areal density; its thickness is a read-out. The gong takes `E`, `e`, `ν` and the **volume**
density `ρ_v`, re-derives `κ`, `ρ_s = ρ_v e` and the membrane coefficient `E e`, and the bend sets
in at a deflection of about one `e`. So the mapping must hand back a thickness (the plate mapping's
`thickness_for_kappa`) and the material's `ρ_v` — the plate batch's areal/volume density trap, in
the other direction: passing `ρ_s` as `ρ` leaves every pitch right and every energy wrong by `e`.

**Why the strike knob means the same thing at every pitch** (to be measured, Part 0 (d)). Scale
`w = e ŵ`, `x = L x̂`, `t = t̂ / f0`: the von Kármán equations become one equation in `ŵ` whose
only material constant is `E e³ / D = 12 (1 − ν²)`. So at a fixed strike in thicknesses, the bend in
cents depends on the outline, the boundary, `ν`, and the grid — not on the metal's stiffness, its
density, the size or the thickness. Set by pitch at a fixed `fs`, even the discrete scheme is the
same dimensionless problem (`f0 k` fixed). If Part 0 confirms it, the strike slider bends a steel
gong at 80 Hz and a brass one at 400 Hz by the same interval, and the page can say so.

**Out of scope:** the cymbal-in-a-room page (decision 3); tuning at a chosen strike strength
(decision 1); a target bend (decision 2); strings other than the damped one; smoothing a moving
control; learned mappings.

---

## 2. The model change — the second loss term reaches the gong

Today `VkParams::new` builds its linear half with `..PlateSpec::default()`, so `σ₁` is always `0`:
the gong has one loss coefficient, hence one ring time.

The change: **`VkSpec::sigma1`**, default `0`, forwarded to the linear `PlateSpec`. That is all the
scheme needs, because every coupled step already runs on the linear half's own pieces:

- `VkCoupledStep::new` takes `step_rhs(…, &p.lin)` (which carries the loss's `+σ₁kG u⁻` half) and
  inverts `p.lin.lu` (whose `A` carries `σ₁kG`); Picard, Newton and `auto` all go through it.
- `mallet` (model #7g): its contact column is `vk.lin.lu.solve(e)` and its trial step is `vk_step`;
  the room-loaded gong's chord uses `VkCoupledStep::with_rhs` against `grid.lu_loaded()`.
- `airbox_wrap::VkSeam`: its operator is `theta_matrix(&plate.p.lin)`, which already adds
  `σ₁kG` (plate batch), and its right-hand side the plate's `step_rhs`.
- `connection` (string ↔ gong): `BridgePlate for VkPlate` steps through `VkPlate::step` and reads
  `p.lin` for its guard — as it reads the linear plate's, which already carries `σ₁`.

**Grep the clients again at Part 1** — "routes through `lin`" above is read from call sites, not
proven; a bar per client with `σ₁ > 0` is the proof (§5.1).

**The energy needs no correction.** The plate's `σ₁` term is centred and implicit (`2σ₁ G δ_t· u`):
it changes no stored energy and adds `−2kσ₁⟨G δ_t· u, δ_t· u⟩` of work per step. The membrane
energy is untouched. But every existing conservation bar on the gong runs at `σ₁ = 0`, so the
balance gets its own bar at a high membrane share (`≥ 50 %` of the energy in the stress field),
where a wrong coupling would show.

**Nothing that exists may move.** `σ₁ = 0` keeps the plate's own skip (the loss form is not even
built), so every gong, mallet-gong, bridge and room trajectory and the 588-request viewer freeze
stay bit-identical.

---

## 3. The mapping — the plate's, not a fork of it

`nonlinear = false` *is* the linear plate (`VkParams.lin`), so the gentle-strike gong is the plate
`tune_plate` already inverts exactly (supported) or to first order (free, Rayleigh quotient,
≤ 0.05 cents and ≤ 0.3 % ring time in the plate batch). The new code is a thin adapter in
`plate_tuning.rs`:

```text
tune_vk_plate(geometry: &VkSpec, material: &Material, targets: &Targets) -> VkTuning
```

- refuses a wood (the gong is isotropic only);
- builds the `PlateSpec` geometry the gong's own `lin` would get (`lx, ly, n, fs, θ`, boundary,
  rectangle, `ν`) and calls `tune_plate` with it;
- returns the `PlateTuning` plus what `VkSpec` takes: `E`, `ν`, **`ρ_v`**, thickness, `σ₀`, `σ₁`,
  and `VkTuning::apply(&mut VkSpec)`.

**One spelling caveat.** `plate_material` forms `D` as `e·t·t·t / (12(1 − ν·ν))`, `VkParams::new` as
`E·pow(e, 3) / (12(1 − pow(ν, 2)))`: the two `κ`s may differ in the last bit. Exactness is to
rounding either way (`~1e-9` cents in the earlier batches); a bar comparing the gong's `lin` with
`tune_plate`'s plate must allow it, or be written against the gong's own `lin`.

**What "exact" means with the nonlinearity on.** The coupling force is cubic in `w`, so the mapped
pitch and ring times are the plate's in the limit of a gentle strike. A bar must show it on a
**live nonlinear run** at a tiny strike (the bend falls as the square of the strike), not only with
`nonlinear = false`.

---

## 4. The strike read-out — how far above the settled note a strike starts

The page already reads the gong's fundamental by zero-crossing spacing over the whole run
(`f0_detected`, supported only) and reports its rise over the linear fundamental. The rise
measured today at the page's server defaults (steel, 0.4 m, 1 mm, `N` 20, 32 kHz, lossless):

| strike (thicknesses) | ¼ | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| rise | +0.4 % | +6.1 % | +19.8 % | +35.8 % | +53.1 % | +92.0 % |
| ≈ semitones | 0.07 | 1.0 | 3.1 | 5.3 | 7.4 | 11.3 |

With loss the note glides down as the plate rings, so a whole-run average is not "where it
starts". Part 0 (e) picks the read-out: the zero-crossing pitch over an early window and a late
one, or a short-window track. The cymbal (free edge) has no clean fundamental once struck hard, so
its read-out may have to say "no single pitch" — Part 0 measures whether the gentle strike's
lowest mode can still be read there.

---

## 5. Validation — the bars

Every bar runs the simulation, never the mapping's algebra backwards.

1. **The loss term on the gong.** `σ₁ = 0` bit-identical to today (state, both stress caches,
   energy), both boundaries, Picard and `auto`; with `σ₁ > 0` the energy drop equals the two
   losses' work per step to `1e-10` of the start at a membrane share `≥ 50 %`, both boundaries;
   passive; the mallet-gong ledger, the bridge and the gong in a room each close with `σ₁ > 0`.
2. **Gentle-strike round trip, nonlinearity on.** Start the supported gong in exactly mode 1 (then
   the anchor) at a strike of `1e-3` thicknesses, project, read `r` and `φ` in the well-conditioned
   form: pitch and ring time on target (bars as the plate's). The free cymbal through the plate
   batch's root measurement, within its first-order bound.
3. **The bend is live:** at a strike of 2 thicknesses the starting pitch is measurably above `f0`
   — a control that the gentle bar is not passing because the coupling is off.
4. **Settling:** with `σ₀ > 0` the supported gong's late-window pitch approaches `f0` from above
   (the strongest page bar: the musician's note is where the gong ends up).
5. **Separability** (if Part 0 confirms it): two metals, two sizes, the same `f0`, `fs`, `N`,
   strike — the same starting bend within a bound Part 0 sets; `ν` named as the only material
   dependence.
6. **Off every default:** `Lx ≠ Ly`, `N`, `fs`, every metal, both boundaries, both set-by modes.
7. **The volume density is live:** the strike's energy equals a core gong built from the mapping
   with `ρ_v`; set by pitch then by its thickness gives the pitch back.
8. **Refusals:** a wood; the plate mapping's own (anchor at the fundamental, ring-time range,
   out of reach); the page's work budget at the mapped thickness.

**Plants** (each red, each time-limited): `σ₁` not forwarded to `lin`; `ρ_s` passed as `ρ_v`; the
nonlinearity off in the gentle bar's run; the thickness from `κ` of the wrong material; `σ₀`/`σ₁`
swapped; the late window replaced by the whole run; `ν` hard-wired.

---

## 6. The page

`crates/physsynth-viewer/src/vk.rs`: the gong's `domain` key already means supported / free, so the
control set takes the plate's key, `controls: physical | musical`, and `set_by: thickness | pitch`.
A request without `controls` takes today's path byte for byte. In musical mode `E`, `e`, `nu`,
`rho`, `sigma` are ignored (decoys in the bars); `material`, `set_by`, `thickness` or `f0`,
`t60_low`, `t60_high`, `f_ref` drive the build; `fs`, `N`, `Lx`, `Ly`, the strike and the pickup
stay. The energy verdict drops the single-rate `2σ` oracle line when `σ₁ > 0`.

Payload: a `tuning` block (material and source, how it was set, thickness, `σ₀`, `σ₁`, the settled
fundamental and the anchor in the scheme, the anchor's margin, `exact`, the lowest partials' pitch
and ring time) and the strike read-out (§4): the starting pitch, its interval above `f0`, and on the
cymbal what can and cannot be read.

Front-end: `CONTROL_OPTS` gains `vk`; the shared musical group shows; the plate page's material
menu filtered to isotropic presets; the strike slider stays. Whether `fs` stays this page's own
slider or becomes the shared group's `sample_rate` is read from the front-end at Part 3. The
coverage guard: `controls:physical` and `controls:musical` for `vk` go into
`BORN_AFTER_THE_FREEZE` with their covering test file. A headless case `vk_musical`.

---

## 7. Order of work

- **Part 0 — probe** (scratch crate under `W:\temp\claude`, nothing in the repo; `σ₁` set by
  rebuilding a `VkParams`' public `lin` with it):
  - (a) `σ₁` through the coupled step: the balance at a membrane share `≥ 50 %`, both boundaries,
    Picard and Newton; `σ₁ = 0` bit-identical.
  - (b) Gentle-strike exactness with the nonlinearity on: mode-1 and anchor runs at strikes
    `1e-3`, `1e-2`, `0.1` thicknesses — how fast the bend vanishes, which strike a bar can use.
  - (c) The cymbal: the mapped settled pitch through the scheme's roots, and whether a gentle
    strike's lowest mode reads cleanly from a run.
  - (d) Separability: steel and brass, two sizes, two thicknesses, same strike — the bend in cents;
    then `ν` alone.
  - (e) The strike read-out: early/late windows against a short-window track, lossless and lossy;
    the settling curve.
  - (f) Cost against pitch: renders over the `f0` range the controls will allow, at the page's
    `fs` range — wall-clock, sweeps, non-converged steps, Newton fallbacks. The refusals come from
    these numbers, not from the sweep-count budget alone.
  - The result goes to the human with any decision it forces.
- **Part 1 — model:** `VkSpec::sigma1`, its bars (§5.1).
- **Part 2 — mapping:** `tune_vk_plate`, bars §5.2–§5.8, plants.
- **Part 3 — page,** backend then front-end, guard, headless case.
- **Part 4 — docs:** hurdle §12, HANDOFF §8, memory.

Commit and push at each part's end, `scripts/cargo-test-nice.ps1 -Full` before each.
