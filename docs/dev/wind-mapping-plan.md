# Parameter mapping — musical controls for the wind instruments

`docs/dev/scientific-hurdles.md` §12, sixth piece. The strings, plates, drums, gong and cymbal have
musical controls (`parameter-mapping-plan.md`, `plate-mapping-plan.md`, `membrane-mapping-plan.md`,
`vk-mapping-plan.md`, `string-mapping-plan.md`). The two wind pages — the bore (`bore`) and the reed
on it (`reed`) — still take raw physics: a tube length, a grid, and the bell's resistance as a
ratio `R/Z0`. This plan covers both.

**Status: DONE 2026-10-20.** Slice 1 (the bore, §5) and slice 2 (the reed, §6) both built.

---

## 1. The pages

- **`bore`** — a closed–open cylinder at `λ = 1` (`BORE_LAMBDA`), so `fs = c0 N / L`. Controls `L`,
  `N`, the far end (`radiating` bell or ideal `open`), `bell_ratio_exp = log10(R/Z0)`. A Gaussian
  bump rings down through the bell.
- **`reed`** — the same tube with the dynamic single reed at the closed end (`ReedBore`). Controls
  `L`, `N`, `gamma = p_mouth / p_closing`, `f_reed`, `q_reed`, the far end, the bell. A blown note:
  it speaks above `gamma ≈ 1/3` and holds.

Both pages already use their `domain` select for the far end, so the control set rides on
`controls` (the plate's precedent).

## 2. Part 0 — what the probe measured (2026-10-20)

Scratch crate `W:\temp\claude\wind-probe` (nothing in the repo).

**The bore is exact, at every partial, in pitch and ring time.** The bore's one-step map, built
column by column from `Bore::step` and solved with `eig::general_eigenvalues`, over
`L ∈ {0.1, 0.5, 2}`, `N ∈ {32, 64, 128, 256}`, `R/Z0 ∈ {1e-4, 1e-3, 1e-2, 0.05, 0.3}`: modes 1–4
sit at `(2m − 1) c0 / 4L` to `≤ 1.4e-9` cents, and every one rings for

```text
T60 = (2L / c0) ln(1000) / (−ln|ρ|),   ρ = (R − Z0) / (R + Z0)
```

to `≤ 4e-9` relative. Reason: at `λ = 1` the staggered scheme carries a travelling wave exactly, the
closed half-cell reflects it with `+1`, and the bell's centred 1×1 solve reflects it with exactly
`ρ` at every frequency — so one round trip (`2N` steps) multiplies the whole state by `−|ρ|`. The
inversion is closed form:

```text
L = c0 / (4 f0),     R/Z0 = tanh(3 ln10 · L / (c0 T60))
```

**The branch.** `ρ` and `1/ρ` give the same `|ρ|`. The root with `R > Z0` makes the bell a closed
end — the tube then sounds an octave up with the even harmonics back. `tanh < 1` picks `R < Z0`
(`ρ < 0`) by construction.

**The reed pulls the note flat, and by a lot.** Played note (interpolated zero crossings of the
settled mouthpiece pressure, 1 s runs, last 40 %) against `c0 / 4L`:

| what moves | played note |
|---|---|
| default (`L` 0.5, `N` 128, `γ` 0.51, `f_reed` 2500, `q` 4, `R/Z0` 2.5e-3) | −62.1 cents |
| `γ` 0.36 → 0.9 | −54 → −62 → −59 (peak at ~0.6; 8 cents in all) |
| `f_reed` 3500 → 1500 | −32 → −161 (1000 Hz: not settled, −310 / +26 in two windows) |
| `q_reed` 1 → 12 | −64 → −63 (3 cents) |
| bell `R/Z0` 0 → 1e-2 | −62.3 → −61.7 |
| `N` 64 → 256 | −62.05 → −62.13 |
| `L` 0.1 → 2 | −283 → −16 |
| `L` 0.1 with `f_reed` 1000 | −1164 (the reed, not the tube, sets the note) |
| `γ` = 1.0 | +1849 (statically shut — not a note of the tube) |

The reed's stiffness and the tube's length set the pull; how hard you blow moves it ~8 cents; the
grid barely at all.

## 3. The human's answers (2026-10-20)

1. **Reed pitch = the note you hear.** The page solves the tube length so that the note the reed
   *plays* lands on the target, re-solving when the blowing pressure or the reed changes; it
   refuses, naming why, where the reed does not speak or the note does not settle. (The other
   option, the tube's own note with the pull reported, was declined.)
2. **The reed's bell in musical mode = the tube's ring time**, through the same exact formula as
   the bore page. Blowing pressure (`γ`), `f_reed` and `q_reed` stay as they are.
3. **Bore page, ideal open end, musical mode:** the pitch works; the ring-time control is hidden,
   with a line saying the end loses nothing and the tube rings forever.

**The agent's own calls** (not the human's; labelled so a later reader can tell them apart):

- The musical ring-time slider runs 0.05–60 s. The ceiling the page cannot pass is the bell range's
  floor `R/Z0 = 1e-4`, which is `≈ 201 s · L` — so at high pitch the ring time is refused naming
  the longest ring time at that pitch. Below the page's resonant gate (`R/Z0 > 0.05`, a ring time
  of `≈ 0.4 s · L`) the tube still rings, briefly; the signature panel already labels that regime.
- Pitch range follows the page's `L ∈ [0.1, 2] m`: 42.9–857.5 Hz on the bore.

## 4. Order of work

Each slice: core function and bars, viewer page, browser check, docs; commit and push per slice,
`scripts/cargo-test-nice.ps1 -Full` before each.

1. **The bore** — `wind_tuning::tune_bore` (closed form), the page's musical set.
2. **The reed** — `wind_tuning::tune_reed`, a search on `L` over short runs of the rendered rig
   (same `N`, `γ`, reed and bell — the page's fixed-`N` sweep says nothing about the render).
   Probe the solver's response first: near the threshold the note settles slowly and may not be
   monotone in `L`.

**Bars carried** (`string-mapping-plan.md` §7): round trip read from the *simulation*; every
parameter off its default; one bar per refusal; plants, each with a time limit; a wiring bar that
listens to the rendered pitch, not the reported block. The bore's ring time is read at
round-trip-aligned steps (multiples of `2N`): the bell sheds energy once per round trip, so the
energy is a staircase between them.

## 5. Slice 1 — the bore (done 2026-10-20)

- **Core** `wind_tuning::tune_bore(f0, t60, c0)` → `L`, `R/Z0` (`None` without a ring time), with
  `bore_partial` and `bore_t60` as read-outs. Bars `crates/physsynth-core/tests/bore_tuning.rs`:
  the one-step map's lowest five partials on `(2m − 1) f0` to `1e-6` cents and on the ring time to
  `1e-8`; the render repeating every round trip as `−|ρ|` to `1e-10` of its peak; the other root
  plays nothing within 600 cents of `f0`; refusals by name. Plants: a length off by `1e-4`, the
  ring-time constant `3.0 → 3.1`, the other root — each red.
- **The render's round trip needed a two-step mean.** The bell's grid-Nyquist ripple (a real
  eigenvalue near `−1` at `R ≪ Z0`, recorded in the bore's batch 2) does not repeat inverted: at
  `1.5e-9` of the bump it failed a `1e-10` bar at every node, not only the bell's. The mean of two
  consecutive steps cancels it and leaves every partial's round trip intact.
- **Viewer** `controls: musical` on the bore page (`f0`, `t60_low`; `L` and the bell slider hide;
  on the open end the ring time hides too). Refusals name the pitch range (42.88–857.5 Hz), the
  longest ring time the bell's `R/Z0 = 1e-4` floor holds at that pitch, and — over the step budget,
  since `fs = 4 N f0` — the highest pitch that fits. Bars `crates/physsynth-viewer/tests/
  wind_musical.rs`: the physics set with musical keys riding along changes no byte; the musical
  page equals the physics page on the mapped tube to `1e-9`; its measured partials sit within a
  cent of the odd series; after a whole number of round trips the air keeps `10^(−6D/T60)` of its
  energy. Plants: the page ignoring the mapped length, or the mapped bell — each red. Browser
  check: `bore_musical`, `bore_musical_open`.

## 6. Slice 2 — the reed (done 2026-10-20)

**The search.** `wind_tuning::tune_reed_length(f0, c0, &ReedSearch, play)` runs the caller's rig
on a trial length, starting at `c0/4f0`, rescales `L` by the miss, then takes secant steps on
`ln f` against `ln L`; it lands within 0.1 cents in 2–6 runs (0.01–2 s; ~0.35 s at a 1 s render,
N = 128). The page's `play` is its own rig run for the render's own step count, with the bell
rebuilt from each trial length — so the render IS the search's last run, and the page re-reads
it to prove it (`render_played == played`, to the bit).

**A run counts only as the tube's fundamental**: speaking (the page's `REED_SPEAK_GATE` over its
`REED_SETTLE_FRAC` tail), settled (the tail's two halves within 1 cent), and at or below the
tube's own `c0/4L` + 25 cents. Every fundamental the probe saw was flat of the tube's note, and a
squeak can be perfectly steady (γ = 1.0: +1849 cents in both windows) — "settled" alone would let
a squeak steer the search (the reviewer's catch).

**The reader took three tries** (`read_played_note`):

| reader | what went wrong |
|---|---|
| mean zero crossings | the wave rings at its edges: a note 27 cents flat read +82 cents sharp, called a squeak |
| Schmitt trigger (±½ rms) | a strong third harmonic crosses the band three times a period: real notes read +1900 cents |
| **autocorrelation** (kept) | — |

The kept reader searches lags from two octaves above the tube's note to two below on a
block-averaged copy, takes the *smallest* lag within 0.9 of the best (a period's multiples
correlate as well), refines at the full rate and reads the period at the largest multiple that
fits in half the window. A window must hold two periods of a note an octave below the tube's, or
it is **too short** (a missing lag would let a harmonic stand in). Over the probe's 3,024
settings it tuned 1,377 (the crossing reader 931) and every disagreement with the spectrum's
loudest peak was an odd harmonic (3rd/5th/7th/9th) or the grid's top-frequency ripple.

**Another ear.** A Hann-windowed Fourier peak on the same run's tail: within 1 cent at ordinary
settings on renders of 0.5 s and up; 1–4 cents at the extremes (γ = 1, the onset, ≤ 60 Hz with
a heavy bell); up to ~9 cents on 0.2 s renders, where the note is still settling. The page says
so, and warns under 0.5 s.

**Refusals**, each with what to change: silent (blow harder), too short (names the shortest
audio), aperiodic, squeak (or, near fs/2, the grid's ripple — raise N), unsettled, out of reach
(the highest note on the 0.1 m tube, the lowest on the 2 m one), no convergence, the step budget
(before the search, from `fs = 4 N f_tube > 4 N f0`), and a ring time longer than the bell holds
on the found tube.

**The onset barely moves with the ring time** (γ 0.34–0.43 over T60 0.05–40 s): a short ring time
does not silence this reed. Defaults D3 and 2 s, as on the bore — the agent's calls.

**Bars.** Core `crates/physsynth-core/tests/reed_tuning.rs` (11): three pitches with every reed
parameter off default, re-read by the Fourier ear within a cent and flat of the tube; the reader
under an antiphase third harmonic (a crossing count reads it 3×); silence, shortness and noise at
the reader; one bar per refusal on a rig the sweep found it on. Plants: no squeak check, no settle
check, tolerance 30 cents, the best lag instead of the smallest, no reach check — each red.
Viewer `wind_musical.rs` (7 more): the physical set untouched by musical keys; the audio heard by
the Fourier ear within a cent, the panel's markers on the played note; the bell recomputed from the
payload's L with the formula written out (the twin cannot see a bell built for the unpulled tube
— only this bar went red on that plant); the twin on the found tube (spectrum and beating panels
excepted: anchored on the played note by design); the open end ignoring the ring time; six
refusals' wording. Plants: the bell from `c0/4f0`, the page ignoring the found L, the panel on the
tube's note, the search's bell from `c0/4f0` — each red. Browser: `reed_musical`,
`reed_musical_open`.

**Seen, not acted on.** The reed page's grid hint reads "λ = 1.00 (implicit scheme — no CFL
limit)", but the bore under the reed is the explicit leapfrog at its CFL limit; and at a low N on
a long tube the mouthpiece pressure carries a strong grid-Nyquist ripple that can sit in the
audible band (fs/2 ≈ 5.7 kHz at N = 64, 1.9 m) — both pre-date this plan.
