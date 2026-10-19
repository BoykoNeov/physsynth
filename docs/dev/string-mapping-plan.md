# Parameter mapping — musical controls for the other strings

`docs/dev/scientific-hurdles.md` §12, fifth piece. The damped string (`parameter-mapping-plan.md`),
the plates, the drum, the gong and the cymbal have musical controls; every other string page still
takes raw physics. This plan covers them, sorted by how much new derivation each needs.

**Status: slices 1–3 DONE 2026-10-19** (§9–§14): the bow, the tension string, the ideal string,
the three barrier pages and the stiff string all have musical controls. Next batch, per §8.1:
the four coupled ideal pages (`sympathetic`, `body`, `platebody`, `radbody`); after it, the
geometric string (§6). **Slice 4 started 2026-10-19 (§16):** the probe overturned §4's premise, and
the human chose to tune through the coupling.

---

## 1. The pages, and the string each one is built on

- **`ideal`** — model #1, explicit, `λ ≤ 1`, one `σ`. Pitch and one ring time; **every**
  partial exact at `λ = 1`.
- **`stiff`** — #2, θ-scheme with `κ`, one `σ`. Pitch, inharmonicity, one ring time (mode 1 only).
- **`bow`** — #3 (`damped::Params`), `σ₀` and `σ₁`. `tune_damped_string` as is; the bow's pull on
  the pitch read out.
- **`tension`** — #9 = #3 + `EA`, `σ₀` and `σ₁`. `tune_damped_string` in the gentle limit (`EA = 0`
  is #3 bit for bit).
- **`jawari`, `juari`, barrier** — #3 with `κ = 0`, `σ₁ = 0`, one `σ₀`. Pitch and one ring time at
  mode 1.
- **`sympathetic`, `body`, `platebody`, `radbody`** — #1 on a bridge or a body, *fixed/free and
  lossless* (`σ = 0` hard-coded on all four), `λ < 1` required. The first draft of this line said
  "one `σ`" and "the string alone; the shift read out": both wrong, see §16.
- **`geometric`** — #10, its own `Params`, Newton, four losses. A separate batch (§6).

## 2. The ideal string (#1) — exact everywhere, by a special case

The explicit scheme's loss is the same centred difference as #3's (`string_ideal.rs`, `1 ± σk`),
so its mode recurrence is §2 of the damped plan at `θ = 0`, `κ = 0`, `σ₁ = 0`. Two consequences:

- `r² = (1 − σk)/(1 + σk)` for **every** mode, so one ring time is met by every partial exactly,
  not only the fundamental.
- At `λ = 1` the scheme has no dispersion, so every partial is exactly harmonic. The pitch target
  fixes `c = 2Lf₀` up to the loss's pull, which the same 2×2 removes.

`tune_damped_string` refuses `θ = 0` (`BadTheta`), so #1 gets its own small function, not a
relaxed one. **The grid question (§8.2)**: at a fixed sample rate `λ ≈ 2 f₀ N / fs`, so `λ ≤ 1`
caps the grid at `N ≤ fs / (2 f₀)` — about 50 segments at 440 Hz and 44.1 kHz. Today's page lets
`fs` follow the pitch (`fs = c N / (L λ)`) and resamples the audio to 48 kHz, so keeping `λ = 1`
costs nothing new.

## 3. The stiff string (#2) — one new derivation

One `σ` and an inharmonicity target. Mode 1's `(f₀, T60)` fixes `(Q₁, σ)` by the existing 2×2.
At the anchor `M`, `Y = σk` is then **known**, and only the pitch is a target:

```text
cos φ = −b / (2 sqrt(a cc)),   a = u + Y,  cc = u − Y,  u = 1 + θX,  b = −2 + (1 − 2θ)X
⇒ (2 − (1 − 2θ)X)² = 4 cos²φ (u² − Y²),   with sign(2 − (1 − 2θ)X) = sign(cos φ)
```

a quadratic in `X`. **A branch choice** — the probe (§7) must show that exactly one root is
positive and has the right sign over the whole fixture grid, or name where two do; the code
refuses rather than guesses when two qualify. The anchor's ring time is not a target: it is what
one `σ` gives (the θ-scheme lengthens it slightly up the series, hurdle §4) and is read out.

## 4. Strings with something attached

The bow, the bridges with buzz (`jawari`, `juari`, barrier), the sympathetic strings and the three
string-on-a-body pages. Each page's own string is one of §2–§3 or #3, so the string alone can be
tuned exactly. What attaches to it moves the pitch:

- the bow pulls the played note (flat as the force rises);
- a buzzing bridge shortens the string's speaking length while it touches;
- a body or a bridge loads the string's end and moves every mode a little.

**Default (pending §8.3): tune the string alone and report the shift**, as the drum was tuned in
vacuo and the air's pull shown on its own page. Correcting *through* the attachment needs the
coupled system's modes, as the drum-in-air question does.

*Overturned for the four coupled ideal pages (§16):* their bodies weigh about what the string does,
so "a little" is 2–13 semitones, and at some pitches the note splits. The human chose to tune
through the coupling there. §8.3 still stands for the bow and the buzzing bridges.

## 5. The tension string (#9) — the settled note

`EA = 0` is model #3 bit for bit, so `tune_damped_string` gives the gentle-limit tuning exactly,
the same adapter pattern as `tune_vk_plate`. The pitch you set is the **settled** note (the gong's
precedent); a hard pluck starts sharp and glides onto it, and the page reads the start and the
settle.

- **Musical mode is for the `duffing` regime only.** The `parametric` regime forces the loss to
  zero on purpose, so two ring-time sliders would be silently ignored there — refused instead.
- **`EA` (pending §8.4):** stays a raw slider, comes from a material and gauge (the existing
  `string_nonlinear::string_coefficients_from_material`, with cited presets), or becomes a glide
  target.

## 6. The geometrically exact string (#10) — its own batch

Its longitudinal Courant number is `sqrt(EA/T)` times the transverse one — 12–25× at realistic
`EA/T` of 150–600 — so most musical settings land past hurdle §6's Newton wall (`λ_long ≈ 4`). It
would need a pre-run guard like the gong's. Any shared material helper must mind
`EA_#9 ↔ (EA − T₀)_#10`.

## 7. Order of work

Each slice: Part 0 probe (a scratch crate under `W:\temp\claude`, nothing in the repo), core
function and bars, viewer page, browser check, docs. Commit and push per slice,
`scripts/cargo-test-nice.ps1 -Full` before each.

1. **Direct reuse:** `bow`, `tension` (duffing). No new maths.
2. **Mode 1 only:** `ideal` (§2) and the three contact pages.
3. **The quadratic:** `stiff` (§3).
4. **Coupled ideal pages:** `sympathetic`, `body`, `platebody`, `radbody` (§4).
5. **Separately:** `geometric` (§6).

**Bars carried from the damped batch** (`parameter-mapping-plan.md` §5, §9): round trip read from
the *simulation*; every parameter off its default; a textbook control asserting its miss; one bar
per refusal; plants, each with a time limit; a wiring bar that listens to the rendered pitch, not
the reported block.

**Viewer traps:**

- `domain` is already taken on `tension` (duffing/parametric), `geometric` and `sympathetic`. Use
  `controls: "musical"` (the gong page's key) on every new page, and leave the damped string's
  `domain: "musical"` as it is.
- `string.rs`' `Built::tuning` is typed to the damped `Tuning`, and `tuning_block` has an
  `unreachable!` for any other model. Generalize both before any other `build_resonator` model
  returns `Some`.
- Each new musical page: a `BORN_AFTER_THE_FREEZE` entry naming its test file, a bar that the
  physical path is byte-identical with every musical key present, and slider defaults on their
  step grids.

## 8. The human's four answers, 2026-10-19

1. **Scope: the solo strings first** — §7 slices 1–3 (`bow`, `tension`, `ideal`, the three
   contact pages, `stiff`). The coupled ideal pages are the next batch; `geometric` after that.
2. **The ideal string's grid: the sample rate follows the pitch.** `λ = 1` is kept, every partial
   exact; the audio is resampled to 48 kHz as the page already does.
3. **Attachments: tune the string alone and show the shift** — the drum-in-vacuo precedent.
4. **`EA` from a material and a gauge**: cited presets (steel, nylon, gut) and a diameter give the
   linear density and `EA` through `string_coefficients_from_material`.

   *Correction, 2026-10-19:* the first build also kept the inharmonicity a free target, with the
   gauge's own bending stiffness only read out. That was the agent's choice, wrongly recorded here
   as the human's. Asked, **the human chose the gauge to set the bending stiffness too** (§15).

## 9. Slice 1, Part 0 result (2026-10-19)

Probe: a scratch crate under `W:\temp\claude` depending on core and analysis by path.

**The bow.** `tune_damped_string`'s string under the page's default bow (position 0.13, 0.1 m/s,
1 N, sharpness 60), 15 fixtures (`f₀ ∈ {110, 196, 440}`, three ring-time pairs, `fs ∈ {22.05k,
44.1k}`, `N ∈ {64, 100}`) and a force sweep:

- **The bowed note is not the string's mode 1, and the offset moves with the grid.** In clean
  one-slip motion it reads −13.3 to +7.7 cents off the tuned pitch. It is stable to 0.02 cents
  from a 0.5 s to a 6 s window, so it is not the detector. But the same 110 Hz string reads −13.3
  cents at 44.1 kHz and +3.9 cents at 22.05 kHz (`N = 100` both). So the page must not call the
  offset "the bow's pull". *What sets it is not established* — §11 falsified the first
  explanation offered here.
- Clean bowing is not guaranteed at the defaults: 196 Hz, `N = 100`, 1 N gives two slips a period.
  That is the physics page's behaviour too (Schelleng's window), already labelled there.
- **The work budget bites.** The page caps a run at 60,000 friction steps. At 44.1 kHz that is
  1.36 s, below the page's 2 s default audio, so a musical bow at the shared sample-rate default
  would be refused out of the box.

**The tension string** (`ρ` and `EA` from a gauge, the rest from `tune_damped_string`; three
materials, four strings each, `fs = 44.1k`, `N = 128`, `θ = 0.28`):

- **Exact in the gentle limit**, as predicted: a 1e-7 m single mode with the mapped losses reads
  ≤ 2.8e-8 cents and ≤ 1e-11 in ring time off its targets. `EA` does not touch the linear string.
- **The start shift is what a gauge makes it.** A 1 mm pluck starts 0.1–4.7 cents sharp; 3 mm,
  0.6–41 cents; 1 cm, 7–372 cents (a thick steel string). The page's `dT/T₀ ≤ 4.45` guard was never
  reached here.
- **The gauge's own inharmonicity** spans 1.7e-6 (thin nylon) to 1.8e-3 (1 mm steel). The
  mapping's default target (1e-4) is far from either end. That is the read-out §8.4 asked for.

**Presets** (the build's citations):

| material | density (kg/m³) | Young's modulus | source |
|---|---|---|---|
| steel (music wire) | 7800 | 190 GPa | [a] |
| nylon | 1080 | `32.0 + 0.0353 σ − 0.0269 ρ` GPa, `σ` in MPa | [b] |
| gut | 1322 | 6.0 GPa | [c] |

- [a] MakeItFrom, ASTM A228 music wire (cold-worked).
- [b] Lynch-Aird & Woodhouse, *Materials* 10 (2017) 497: Eq. 19, and Table 1's lower-density
  group.
- [c] Lynch-Aird & Woodhouse, *Materials* 11 (2018) 2160: Table 1, and the 5.5–6.5 GPa range
  they cite.

**Part 1 done (core), 2026-10-19.** `string_tuning::tune_tension_string` and `STRING_MATERIALS`,
held by `crates/physsynth-core/tests/string_tension_tuning.rs`, seven bars. Every material plays
both anchors to within 4.3e-11 in pitch and 3.5e-11 in ring time. A 3 mm pluck at 196 Hz on a
0.6 mm string starts +12.7 cents sharp on steel and +2.4 on gut; steel's settles onto its target
to 1e-5 cents. Eight plants (the diameter used as the radius, a density off by 1e-4, the stress
in Pa in nylon's fit, the fit's sign, the gauge's `κ` imposed, `π²` dropped from the gauge's
inharmonicity, `EA` dropped, the stress over the wrong area) each turned a bar red. The file runs
in 35 s unoptimised, after the round trip was cut from 0.25 s to 0.06 s of simulation with no
loss of precision (155 s before), so it stays in CI's debug lane.

Nylon's modulus is a strong function of stress (4 to over 10 GPa on one string), so a fixed value
would be wrong for every string but one. The fit is evaluated at the tuned stress `T/A`. That is
possible because the tension the mapping returns does not depend on `EA`.

## 10. Slice 1, Parts 3–4: the two pages (2026-10-19)

The bow and the tension string take `controls: "musical"` (`string.rs`' `is_musical`, which keeps
the damped string's `domain` key); the tension string adds `string_material` and `diameter_mm`.
`Built::tuning` is now a `Musical` struct carrying its own `theta`, so the `unreachable!` the
reviewer flagged is gone. Held by `crates/physsynth-viewer/tests/bow_tension_musical.rs` (ten
bars) and two headless browser cases, `string_bow_musical` and `string_tension_musical`.

- **The bow** plays +3.9 cents off its 110 Hz target, in the payload bar and in the browser at
  22,050 Hz (a first browser run read +3.5: it had snapped to 22,100 Hz, below). The read-out
  says the offset moves with N and the sample rate as well as the bow, and that its cause is not
  established (§11).
- **The tension page measures exactly what the physics page measures** on the mapped string, to
  1e-9 in every spectrum field: its `T`, `rho`, `EA` and `kappa` typed into the physics sliders,
  with `lambda` chosen so `fs` matches. That twin is the wiring bar. A 3 mm steel pluck starts
  +12.3 cents sharp and the settled note sits 0.0035 cents off 196 Hz.
- **Found:** at that gentle pluck both pages read the start shift 2.7 % below the Duffing closed
  form (1.403 Hz against 1.443). The physics page's bars run a 2 cm pluck, where the shift is
  ~20 times larger. This is the page's 12-period measurement, not the wiring, so no musical bar
  bounds it.
- **Found:** the bow's 22,050 Hz default was not on the sample-rate slider's 100 Hz grid, and
  the browser quietly ran 22,100 Hz. That is the damped batch's "a slider grid is a value
  filter" again. The step is now 50 Hz (22,050, 44,100, 48,000 and 96,000 all on it), and the
  headless case asserts the rate.
- **The shared read-out paragraph** was gated on the damped string's domain, so it would have
  stayed hidden on both new pages. The gate is gone (its group is already musical-only), and
  the paragraph is cleared before a page with its own read-out draws.
- **Plants, viewer** (each turned a bar red): the bow's reference pitch from the `κ` slider; the
  lossless flag ignored; `rho` from the slider; `EA` from the slider; the tension page keyed on
  `domain`; the bow's 44.1 kHz default; the musical bow built as a plain string; the parametric
  regime allowing musical controls; the gauge block dropped.

## 11. The bowed offset's first explanation, falsified (2026-10-19)

§9 first said the bowed offset came from the partials between the two anchors drifting with the
grid. The reviewer asked for the check, and the probe (each partial's pitch from
`string_tuning::mode_frequency`, the 110 Hz string of §9, cents from the target law) says no:

| fs, N | bowed offset | worst drift, partials 2–17 | partials 21 / 30 / 40 |
|---|---|---|---|
| 44.1 kHz, 100 | −13.3 c | +1.1 c | −2.0 / −18.0 / −62 c |
| 22.05 kHz, 100 | +3.9 c | +3.0 c | −5.4 / −45.6 / −143 c |
| 44.1 kHz, 64 | −5.9 c | +3.1 c | −5.6 / −48.4 / −158 c |

The in-between partials stay within about 3 cents everywhere, and the setting with the least
drift bows furthest off. The partials above the anchor do fall flat with the grid, but nothing
here shows they set the offset either. The page, this plan, the hurdles register and the memory
now say only what was measured: the offset moves with `N` and the sample rate as well as with the
bow, and what sets it is not established.

**Follow-ups the same day (the reviewer's points):**

- In musical mode, two refusals named controls that are hidden there. The tension page's
  stability refusal said "lower EA"; it now says to lower the amplitude or choose a thinner or
  softer string. Both work-budget refusals said "lower N"; with the sample rate fixed, N does not
  change the step count, so they now name the audio length or the sample rate. The physics wording
  is unchanged byte for byte (the freeze holds the refusals).
- Two headless cases load each page with every slider at its default:
  `string_bow_musical_defaults` and `string_tension_musical_defaults`. The tension page's defaults
  sit at 56,000 of its 60,000 steps; both render.

## 12. The human's two answers on stress, 2026-10-19

The reviewer found that the material sliders can tune a string far outside what its cited data
supports. Asked, the human chose **warn and keep running** for both cases:

1. **Nylon's modulus outside the stress the paper measured.** The 2017 paper's own
   tension-modulation example runs from 30 MPa (about 3 GPa) to 220 MPa (nearly 11 GPa). Outside
   that range the fit is an extrapolation, and the read-out says so.
2. **A string tuned past its material's breaking stress.** Cited limits:
   - steel music wire: 2,450 MPa ultimate tensile strength (MakeItFrom, ASTM A228);
   - gut and nylon: about 280 MPa. Lynch-Aird & Woodhouse, "The sound of a harp string"
     (euphonics.org, 2023), Fig. 1: the top gut harp strings run at about 275 MPa, "very close to
     their breaking stress limit", and "the breaking stress limit for nylon strings is fairly
     similar to that for gut". An approximate figure, read from a plot, and stated as such.

   The simulation still runs. It is a model, and a model string does not snap.

**Built the same day.** `StringMaterial` carries `breaking_mpa` and its source, and nylon's
`Modulus::StressFit` carries `measured_mpa: (30, 220)`. `GaugeTuning` reports
`modulus_extrapolated` and `past_breaking`. The viewer's gauge block turns them into plain-words
warnings, and the read-out shows them in amber. Two core bars and one viewer bar cover them; the
viewer bar is a nylon string at 1 kHz, past both limits and still rendered. Three plants (the
range never checked, the breaking stress compared in pascals, the range compared in pascals) each
turned a bar red.

## 13. Slice 2: the ideal string and the flexible string (2026-10-19)

**Part 0 probe** (48 configurations: `f0 ∈ {27.5, 110, 440, 1000}`, `N ∈ {16, 64, 128, 512}`,
`T60 ∈ {0.1, 1, 10}`). The reviewer's equation for the ideal string at `λ = 1`,
`cos(2π f0 k) = cos(π/N) cosh(a k)`, `a = 3 ln 10 / T60`, solved for the time step:

- **Exact everywhere.** The fundamental's pitch and every partial's ring time read back from the
  simulation to 3e-11 or better, with `λ = 1` to the last printed digit. No setting needs a
  refusal: the root always exists, so the string is never too damped to sound.
- **The partials above the fundamental run SHARP with loss**, not flat as first estimated. The
  fundamental is pinned, and a damped oscillator's frequency falls by a larger share at a low
  pitch than at a high one. At 27.5 Hz they run +98 to +128 cents sharp with a 0.1 s ring, +1.4
  cents with 1 s, and nothing measurable with 10 s. This is physics, not the scheme.
- **The audio path's sample-rate rounding** (the resampler takes `round(fs)`) costs up to 0.6
  cents at the lowest rates (`fs ≈ 880 Hz`: a 27.5 Hz note on 16 segments) and under 0.04 above
  14 kHz. It is the shared audio path's, left as it is and noted here.

**Part 1, core:** `string_tuning::tune_ideal_string` (bisection on the time step, written through
half-angles so the difference near `cos = 1` keeps its digits), `ideal_mode_frequency`, and
`tune_flexible_string` for the barrier pages' `κ = 0`, `σ₁ = 0` string (mode 1 only, the damped
mapping's per-mode solve). Held by `crates/physsynth-core/tests/string_ideal_tuning.rs`:

- the ideal round trip over 54 fixtures: pitch to 5.7e-13, every partial's ring time (modes 1, 3
  and `N/2`) to 5.4e-9;
- the sharp partials read from the simulation, matching `ideal_mode_frequency` to 1e-8;
- the textbook ideal string (`c = 2 L f0`, `σ = 3 ln 10 / T60`) 149 cents flat on a 0.1 s thud at
  27.5 Hz, the corrected one exact;
- the flexible round trip over 72 fixtures, pitch to 2.7e-12, ring time to 1.9e-11;
- the refusals.

Seven plants each turned a bar red: the loss without its `tanh`; `c` over `N + 1` segments; the
half angle doubled; the loss dropped from the mode pitch; `ln 10` dropped from the ring time; the
flexible loss not divided by `k`; mode 2's eigenvalue in place of mode 1's. The file runs in 1 s
optimised and 13.5 s unoptimised.

**Parts 3–4, the pages.** The ideal page and the three barrier pages take `controls: "musical"`
with a pitch and one ring time (`f0`, `t60_low`). The ideal string runs at `λ = 1` with its
sample rate following the pitch (2·f₀·N). The barrier pages take the sample-rate slider, and their
`λ` read-out is refused at 1 or above, as the physics path refuses it, with the sample rate that
would do. `string.rs`' `Musical` now carries a `Mapped` (damped, ideal or flexible). The barrier
pages share one `string_setup`, whose physics arithmetic is the reference's: the full exact
freeze of `contact.json` is unchanged. Held by
`crates/physsynth-viewer/tests/ideal_contact_musical.rs` (eight bars) and two headless cases,
`string_ideal_musical` and `jawari_musical`.

- **Each barrier page in musical mode is its physics page on the mapped string**, to 1e-9 in every
  payload field (float32 renderings aside). The pages keep their internal fundamental at the
  physics path's `c / 2L`, which sits within a fraction of a cent of the target. Using the exact
  target there moved display-only numbers (the spectrum's range, the animation stride) and broke
  the twin for no physical gain.
- **The ideal page's energy decays at 6 ln 10 / T60** to 1.4e-6, which proves the loss it plays is
  the mapped one. Found by a plant: a build reading the `sigma` slider passed every other bar,
  because they listened to pitch and the loss barely moves pitch. The damped batch's lesson again:
  a wiring bar must listen to the output, and to the output the mapping sets.
- **A jawari at `N = 64`** warns that its partials above the fundamental run flat of m·f₀, up to
  −55 cents at the 16th. Its physics page has the same flattening; the musical page says so.
- **Plants, viewer:** the ideal string's loss from the slider (caught only after the energy bar
  was added); its sample rate from the lossless law; the barrier pages never musical; the
  `λ ≥ 1` refusal removed; the tuning block dropped; the wrong θ; the barrier pages keyed on
  `domain`. All seven turn a bar red.

## 14. Slice 3: the stiff string (2026-10-19)

**Part 0 probe** (1,728 configurations: two strings, four grids, θ ∈ {0.25, 0.28, 0.5, 1},
`f0 ∈ {55, 220, 880}`, `B ∈ {0, 1e-4, 1e-2}`, three ring times, two reference frequencies). Mode
1's `(f0, T60)` gives `(X₁, Y)` by the damped 2×2. With `Y = σk` then known, the anchor's pitch
alone gives `X_M` through §3's quadratic:

- **Exactly one valid root** in all 1,686 representable configurations; **two in none**. The other
  42 have no valid root at all: the anchor's pitch is past the scheme's ceiling (θ = 1 caps a pitch
  below fs/6). The same "unreachable" refusal as the damped string's.
- **Why one:** lossless, `cos φ` falls strictly as `X` rises (its derivative is
  `−1/(2(1 + θX)²)`), so each pitch has one stiffness. The squaring's second root belongs to the
  mirrored branch. The sign test is load-bearing: a plant that dropped it let that root through,
  and two bars went red.

**Part 1, core:** `string_tuning::tune_stiff_string` (returns a `Tuning` with `σ₁ = 0`, so the
drift read-out reads it as model #2) and `stiff_mode_t60` (the anchor's ring time, a read-out).
Held by `crates/physsynth-core/tests/string_stiff_tuning.rs` over 72 fixtures:

- fundamental pitch to 8.3e-11 and ring time to 1.0e-9, anchor pitch to 1.5e-12, and the anchor
  ring-time read-out matching the simulation to 1.0e-9;
- the textbook mapping 164 cents flat at the anchor (82.4 Hz, θ = 1, N = 64), the corrected one
  exact;
- the refusals, including the θ = 1 ceiling, which is reachable at θ = 1/4.

Six plants turned bars red: the sign test dropped, the anchor solved lossless, `β`'s sign flipped,
the anchor's eigenvalue off by one mode, θ dropped from the ring-time read-out, σ halved. The file
runs in 1.8 s optimised and 26 s unoptimised.

**Parts 3–4, the page.** `stiff` takes `controls: "musical"`: pitch, inharmonicity, one ring time,
the reference and the sample rate. The read-out names the anchor's ring time as what one loss
leaves it. Held by `crates/physsynth-viewer/tests/stiff_musical.rs` (five bars) and the headless
case `string_stiff_musical`.

- **The wiring bar is a twin again:** the musical page's energy decay and every detected partial
  equal the physics page's on the mapped string, to 1e-9.
- **Found:** the first draft asserted that the energy panel's decay fit equals the fundamental's
  ring time. It read 1.98/s against 6.91/s. On this scheme one σ decays mode `m` at
  `2σ/(1 + θX_m)` (hurdle §4), so a pluck's energy falls at a blend of rates. The ideal string's
  energy bar holds only because there every mode shares one rate.
- **Plants, viewer:** σ from the slider, κ from the slider, the page keyed on `domain`, the anchor
  ring time dropped. All four red.

## 15. The gauge sets the stretch too (2026-10-19, the human's call)

A side note pointed out that §8.4's "the inharmonicity stays a free target" was the agent's
choice, recorded as the human's. Asked, the human chose **the gauge to set the bending stiffness
too**: a thick steel string now stretches its overtones as a real one would, and the tension page
has no inharmonicity slider.

**The mapping** (`tune_tension_string`, now taking `GaugeTargets`: pitch, two ring times, the
reference). The gauge fixes `κ² = E r² / (4 ρ_v)`. Mode 1's `(f0, T60_low)` gives `(X₁, Y₁)`, so
`c² = X₁/(k² p₁²) − κ² p₁²`; nylon's modulus depends on the stress `ρ_v c²`, so the two are
solved around each other. At the anchor the stiffness is now known, so `X_M` is too, and the ring
time alone fixes `Y_M = (1 + θX_M)(1 − r²)/(1 + r²)`. Then `(σ₀, σ₁)` follow from `Y₁` and `Y_M`.

- **Exact:** the fundamental's pitch and ring time, and the anchor's ring time (round trip: pitch
  1.9e-12, ring times 1.7e-11). The anchor's **pitch** is the gauge's, a read-out that matches the
  simulation to 3.6e-11. "High" is the partial nearest `f_ref` on the string's own stretched law.
- **What it changes:** a 110 Hz steel string's 18th-ish partial runs +25 cents sharp on 0.3 mm
  wire and +270 cents on 1 mm wire (B = 1.6e-4 against 1.7e-3).
- **New refusals:** a gauge too stiff for its pitch (bending alone would sound mode 1 above the
  target: no positive tension), and a high ring time longer than the low one allows.
- **Plants:** no nylon iteration; the diameter as the radius in κ; θ dropped from the anchor's
  root product; κ ignored in `c²`; the anchor picked on the harmonic law. The last one **survived**
  the round trip, which measures whichever anchor it is handed, until a bar pinned the anchor to
  the stretched law. All five are red now.
- **A consequence on the page:** with the gauge's own κ, the grid's flattening of the high
  partials is no longer absorbed. The free κ of the first build quietly compensated for it, holding
  the anchor on the target law. The default tension page now warns that partials between f0 and
  the anchor run up to 25 cents flat of the gauge's stretched law (in tune to partial 7, `N =
  128`). A finer grid or a higher sample rate brings them in, as the warning says.

## 16. Slice 4: the coupled ideal pages — probe, and the human's two answers (2026-10-19)

**What the four pages build** (`body.rs`, `radbody.rs`, `sympathetic.rs`, `platebody.rs`): an
ideal string, fixed at the nut and free at the far end, with `σ = 0` hard-coded, on a bridge spring
to a body. The body's modal mass (0.02 kg) is about the string's own `ρL`, chosen so the body
"genuinely reacts". `λ < 1` is required: at `λ = 1` the spring pushes the Nyquist mode unstable.

**Part 0 probe** (a scratch crate under `W:\temp\claude`; the pages' defaults, `N = 100`,
`λ = 0.9`, pitches 55–440 Hz). The coupled lossless operator is the core's own
`symmetrized_operator`, and each eigenvalue `μ` plays at `cos(2πfk) = 1 − k²μ/2`. That matches the
body page's own `terminus_f1` read-out to about 0.1 Hz at every pitch.

- **The shift is not small.** At the body page's spring (8000 N/m), a string whose clamped note is
  100 Hz plays at 90.8 Hz (−168 cents); 196 Hz plays −253 c, 440 Hz −432 c. The plate page moves
  further and in both directions: 100 Hz plays +268 c on the free plate, 440 Hz −1309 c.
- **At some pitches the note splits.** Near a body resonance two or three coupled modes share the
  string's motion (shares 0.30 / 0.42 / 0.24 at 262 Hz), a wolf note: no single note is "the" one.
- **The two-string page** has one clean note: the antisymmetric mode keeps the bridge still, so it
  is a string on a spring to ground (100 Hz → −43 c at 8000 N/m, −208 c at 1500 N/m).
- **The stability guard refuses** low pitches at stiff springs (55 Hz at 8000 N/m).
- The radiating page has no played-pitch read-out at all.

**The human's two answers** (asked with the numbers above):

1. **The pitch slider sets the note the instrument plays.** Tension is solved so the coupled
   system plays the target. Where the note splits, the page shows both notes and their strengths
   rather than pretending there is one. The plate page comes last: it needs its own calculation.
2. **Ring time: add a string loss.** Musical mode gets a ring-time slider that puts loss on the
   string itself; the physics pages are unchanged. The ring time is approximate, because the body's
   loss adds to it unevenly across notes.

**The agent's calls, not the human's** (to be confirmed or overridden):

- "The note" is the coupled mode carrying the largest share of the string's motion, in the band
  the page already uses for `terminus_f1`.
- On the two-string page that is the bridge-still mode; both notes a single pluck sounds are shown.
  `detune` stays a tension ratio from string A's tuned tension.
- In a split region the page renders (it does not refuse) and names both notes.
- The radiating page's `R` sweep holds the mapped `T` and `σ`; it runs at its own fixed `N`, so its
  string is slightly off pitch, and the page says so.

**Method (reviewer, 2026-10-19).** The integrator's own one-step map, built column by column from
`step()` as `symmetrize` is built from `apply_a`, so every loss is included. The lossless symmetric
solve locates and selects the mode; complex shift-invert on the map refines its damped `z`
(pitch `arg z / 2πk`, decay `ln|z| / k`).

**Part 0, the reviewer's four checks:**

- **A lossy string broke every bridge's books — fixed.** The string step divides its update by
  `1 + σk` and the spring's pull on the end node was subtracted afterwards, undivided, so the
  string felt `(1 + σk) F` while the body felt `F`. Total plus the scheme's own dissipation missed
  by 1.5e-4 of `E0` at `σ = 5` and 7e-4 at `σ = 69` (body loss alone: 5e-14). The suite never saw
  it: its lossy-string bars ask only for a monotone decrease, which held. Now
  `connection::end_correction` divides the pull too, in all three bridges; lossless, that is a
  division by exactly `1.0`, so no frozen fixture moved (full suite, 1990 tests, green). The bar is
  `crates/physsynth-core/tests/connection_lossy_string.rs`: the books close to 1e-10 on the modal
  body, the radiating body, two strings with *different* losses, and both plates. Plants: the
  undivided pull (all four red); every string divided by the first string's loss (the two-string
  bar red).
- **The one-step map is right and cheap.** Its damped `z` predicts a simulation started on that
  mode to 5e-13. Loss pulls the pitch by cents (90.773 → 90.658 Hz, −2.2 c, at `σ = 35` with body
  loss 20), so the refinement matters. Building the two-string map (410 × 410) took 6 ms and the
  refinement 15 ms. **Trap:** a Rayleigh-quotient iteration from an arbitrary start converged to
  the wrong mode on two strings; start from the lossless eigenvector, fixed shift first.
- **The strongest note rises with `T` between jumps** (no decrease over 728 steps of 5 cents), but
  it **jumps over gaps**: at the body page's spring, 99.5 → 125.1, 181 → 223 and 239 → 290 Hz. About
  11 of the 36 semitones between 61 and 500 Hz can never be the strongest note. The gaps are the
  split regions, and they are wide.
- **The guard is a wall below:** at 8000 N/m it refuses every clamped target under 61 Hz; at
  3000 N/m none in 40–500 Hz.
