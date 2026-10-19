# Parameter mapping — musical controls for the other strings

`docs/dev/scientific-hurdles.md` §12, fifth piece. The damped string (`parameter-mapping-plan.md`),
the plates, the drum, the gong and the cymbal have musical controls; every other string page still
takes raw physics. This plan covers them, sorted by how much new derivation each needs.

**Status: PLANNED 2026-10-19 — the human answered §8 the same day; slice 1 next.** Nothing built.

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
- **`sympathetic`, `body`, `platebody`, `radbody`** — #1 on a bridge or a body, one `σ`. The string
  alone; the coupling's shift read out (§8).
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
   linear density and `EA` through `string_coefficients_from_material`. The inharmonicity stays a
   free target; the bending stiffness the gauge alone would imply is read out beside the `κ` the
   mapping chose, not imposed.

## 9. Slice 1, Part 0 result (2026-10-19)

Probe: a scratch crate under `W:\temp\claude` depending on core and analysis by path.

**The bow.** `tune_damped_string`'s string under the page's default bow (position 0.13, 0.1 m/s,
1 N, sharpness 60), 15 fixtures (`f₀ ∈ {110, 196, 440}`, three ring-time pairs, `fs ∈ {22.05k,
44.1k}`, `N ∈ {64, 100}`) and a force sweep:

- **The bowed note is not the string's mode 1, and the offset moves with the grid.** In clean
  one-slip motion it reads −13.3 to +7.7 cents off the tuned pitch. It is stable to 0.02 cents
  from a 0.5 s to a 6 s window, so it is not the detector. But the same 110 Hz string reads −13.3
  cents at 44.1 kHz and +3.9 cents at 22.05 kHz (`N = 100` both). A bowed (Helmholtz) note's
  period is set by the whole series, and the partials between the two anchors drift with the grid
  (the damped plan's §7 drift). So the read-out is "the bow and the grid together", and the page
  must say so, not "the bow's pull".
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
