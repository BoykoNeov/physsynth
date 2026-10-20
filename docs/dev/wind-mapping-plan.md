# Parameter mapping — musical controls for the wind instruments

`docs/dev/scientific-hurdles.md` §12, sixth piece. The strings, plates, drums, gong and cymbal have
musical controls (`parameter-mapping-plan.md`, `plate-mapping-plan.md`, `membrane-mapping-plan.md`,
`vk-mapping-plan.md`, `string-mapping-plan.md`). The two wind pages — the bore (`bore`) and the reed
on it (`reed`) — still take raw physics: a tube length, a grid, and the bell's resistance as a
ratio `R/Z0`. This plan covers both.

**Status: started 2026-10-20.** Slice 1 (the bore) **DONE 2026-10-20** (§5); slice 2 (the reed)
next.

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
