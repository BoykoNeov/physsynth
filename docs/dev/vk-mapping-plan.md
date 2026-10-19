# Parameter mapping, part four — musical controls for the gong and the cymbal

`docs/dev/scientific-hurdles.md` §12, fourth piece. The damped string
(`docs/dev/parameter-mapping-plan.md`), the three linear plates (`docs/dev/plate-mapping-plan.md`)
and the drum (`docs/dev/membrane-mapping-plan.md`) have musical controls; §12 names the von Kármán
plate (model #6, the supported gong and the free-edge cymbal) as still raw.

**Status: PLANNED 2026-10-18; Part 0 (probe) DONE 2026-10-19** (§8); the human's four answers
to it in §8.8. Part 1 next.

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

---

## 8. Part 0 result (2026-10-19)

Scratch crate under `W:\temp\claude` (deleted at the batch's end); `σ₁` set by rebuilding a
`VkParams`' public `lin` with it. Steel unless named, `N` 20, `couple_max_iter` 50,
`couple_method` `Auto`.

### 8.1 (a) The second loss term through the coupled step — works as planned

Steel 0.4 × 0.35 m, 48 kHz, `σ₀` 1.5, `σ₁` 2e-3, both boundaries, both methods. The loss balance
(energy lost per step against the two loss terms' work) closes to the last bits, with the
membrane carrying over half the energy:

| boundary | strike | membrane share | worst balance, Picard | worst balance, Newton |
|---|---|---|---|---|
| supported | 4.5 e | 64 % | 1.39e-13 | 3.96e-14 |
| free | 3 e | 57 % | 2.59e-14 | 7.01e-13 |

No step failed to settle. Part 1 needs nothing beyond §2.

### 8.2 (b), (c) The settled note is met exactly in the gentle limit — both boundaries

The bend above the target grows as the strike's square: ≈ 7.6e-5 cents at 1e-3 e, 7.6e-3 at 1e-2,
0.76 at 0.1, 12–23 at 0.5 (supported, steel 0.4 m at 200 Hz: `e` 6.66 mm, `σ₀` 1.396, `σ₁`
7.37e-3). At 1e-3 e the ring time is off by ≈ 3e-8 relative; the linear control reads ≈ 1e-9
cents. The cymbal (bronze, brass; mode 1 and the anchor read by projection onto the scheme's own
free modes) gives ≈ 1e-4 cents and ≈ 1e-7 in ring time at 1e-3 e. **The bars can run the
nonlinearity on at 1e-3 e** and assert the targets to ≈ 1e-3 cents and 1e-6 in ring time.
(Eigen-solver note: a shift exactly at an eigenvalue does not converge — shift at `0.999 β`.)

### 8.3 (d) The bend at a fixed strike depends on outline, `ν`, grid and `f0·k` only

At strikes 1, 2, 4 e (200 Hz, 32 kHz): steel 0.4 m and 0.25 m agree to every printed digit
(+101.816, +310.845, +734.606 cents). Brass and aluminium (`ν` ≈ 0.33) bend 1–2 % less, glass
(`ν` 0.22) ≈ 3 % more. A 0.4 × 0.3 outline bends much less at small strikes (+85 against +102 c).
`f0·k` from 0.0005 to 0.0125 moves the bend 312.6 → 308.8 c; `N` 12 → 32 moves it 314.8 → 309.4 c.

### 8.4 (e) The read-out — early and late windows; the cymbal only after it settles

Lossy (2 s / 0.4 s), target 200 Hz; cents above the target in 50 ms windows starting at:

| | 0 | 50 | 100 | 200 | 400 | 600 | 800 | 950 ms | whole run |
|---|---|---|---|---|---|---|---|---|---|
| supported 2 e, pickup | +277 | +219 | +166 | +95 | +26 | +7 | +2 | +0 | +58 |
| supported 4 e, pickup | +658 | +532 | +436 | +286 | +101 | +28 | +7 | +3 | +169 |
| free 2 e, pickup | +2480 | +1458 | +1087 | +1031 | +1018 | +1017 | +1017 | +1017 | +1156 |
| free 2 e, mode 1 | +2764 | +1548 | +543 | −16 | −9 | −14 | −4 | +0 | |
| free 4 e, mode 1 | +2374 | +393 | −35 | −88 | +2 | −7 | +3 | +8 | |

- **The gong:** the pickup's zero crossings and the mode-1 projection agree to a few cents (the
  4 e start: +658 against +654); an early window gives where the note starts, a late one shows it
  settling onto the target. The whole-run figure the page shows today (+58 c) is neither.
- **The cymbal:** the pickup locks onto a higher tone (+1017 c) for the whole run, so a pickup
  read-out names the wrong tone. Projected onto the lowest free mode, the tone settles within
  ≈ ±20 c after ≈ 200 ms (±90 c at 4 e); before that there is no single pitch — the crash.

### 8.5 (f) Cost — three classes, and the cymbal's depends on the grid

Each case is classed by whether its steps settle, not by wall-clock (the machine's speed moved
≈ 6× between runs for the same work):

- **clean** — the fast method settles every step on its own (≈ 6–20 linear solves per step);
- **rescued** — Newton takes over on some or all steps and every step still settles; the answer
  is right and slower (tens to hundreds of solves per step, most of them early in the sound);
- **broken** — some steps never settle (≈ 9 000–23 000 solves per step, *minutes* per step at
  `N` 32); the answer is not a solution of the scheme.

Runs stopped after 20 unsettled steps or 2 minutes; a case stopped by the clock with no unsettled
step is classed by what it had done. Steel 0.4 m set by pitch, ring times 2 s / 1 s, 0.5 s of
sound; `f0·k` is pitch ÷ sample rate.

**The gong (supported), `N` 20,** strikes 2 and 6 e:

| `f0·k` | ≤ 0.0083 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|
| strike 2 | clean | clean | clean | clean | clean |
| strike 6 | clean | rescued (26–49 steps) | rescued (524) | rescued (809–1 496) | **broken** |

**The cymbal (free), `N` 20,** strikes 1, 2 and 6 e:

| `f0·k` | ≤ 0.0031 | 0.0042 | 0.0063 | 0.0083 | 0.0125 | 0.0167 | ≥ 0.025 |
|---|---|---|---|---|---|---|---|
| strike 1 | clean | clean | clean | clean | rescued | rescued | **broken** |
| strike 2 | clean | clean | clean | rescued | rescued | **broken** | **broken** |
| strike 6 | rescued | rescued | **broken** | **broken** | **broken** | **broken** | **broken** |

**The cymbal against the grid** (32 kHz):

| case | `N` 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|
| 200 Hz (`f0·k` 0.0063), strike 2 | clean | clean | clean | rescued | rescued |
| 200 Hz, strike 6 | broken | broken | broken | broken | broken |
| 400 Hz (0.0125), strike 2 | rescued | rescued | rescued | **broken** | **broken** |
| 400 Hz, strike 6 | broken | broken | broken | broken | broken |

Readings:

- Pairs with the same `f0·k` line up in both tables (16 kHz / 800 Hz and 32 kHz / 1600 Hz broke
  alike; 16 kHz / 400 and 32 kHz / 800 needed alike hundreds of rescues). With §8.3's
  separability, **the trouble is a function of `f0·k`, the strike, the grid and the boundary** —
  evidence from steel 0.4 m only, a hypothesis until a second size and metal are run.
- **The grid matters for the cymbal**, finer being worse. The gong's grid dependence is not swept.
- Refusing on the run is too late: at `N` 32 one unsettled step took 4–6 minutes. A guard must
  predict from the settings.
- One case was refused by the mapping, not the solver: 1600 Hz at 16 kHz cannot have a 1 s high
  ring time (the anchor's range there is 1.4–7.0 s) — the plate batch's two-sided range.

### 8.6 A hazard on today's page — older than this batch

`crates/physsynth-viewer/src/vk.rs`'s work budget assumes the worst step is 50 fast sweeps
(`n_live × steps × couple_max_iter ≤ 2e9`); it does not count Newton's rescue, and a broken step
costs ≈ 400× that. Today's physical page reaches the broken region: the free plate with `e` 20 mm
(≈ 400 Hz), strike 2, 32 kHz, `N` 32, 1 s, is under the budget (≈ 1.1e3 × 3.2e4 × 50 ≈ 1.8e9)
and, by §8.5, would run for hours. Not measured end to end — read off the budget and the table.

### 8.7 Decisions this forces (to the human)

1. **The guard's shape** — refuse what is predicted *broken*; for *rescued*, warn with a cost
   estimate or refuse; whether to name the sample rate that would fit.
2. **The cymbal's read-out** — its start is a crash, so no starting bend: show "settles onto the
   note after ≈ X ms" instead, or show nothing for it and the start/settle pair for the gong only.
3. **The hazard of §8.6** — guard the physical page too (the same rule, from the same table), or
   leave it as it is.
4. **The musical page's default pitch** — today's defaults (steel 0.4 m, 1 mm) sit near 30 Hz.

### 8.8 The human's answers (2026-10-19)

1. **Refuse and warn:** refuse what is predicted *broken*; run *rescued* with a warning and a
   rough time estimate; in both, name a sample rate that would fit.
2. **Gong: start and settle; cymbal: settle.** The gong reads where the note starts and how it
   glides onto the target (early and late windows); the cymbal reads "settles onto the note after
   ≈ X ms", from the lowest free mode's projection.
3. **The same guard on the physical page,** so neither set of controls can hold the server for
   hours.
4. **Default pitch 100 Hz.**

The guard is a prediction from `f0·k`, the strike, `N` and the boundary, built from a measured
table. Before Part 3 writes it: the gong's grid dependence, and a second size and metal to test
the `f0·k` claim (§8.5). Status: Part 1 next.
