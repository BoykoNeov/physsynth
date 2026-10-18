# Parameter mapping, part three — musical controls for the drum

`docs/dev/scientific-hurdles.md` §12, third piece. The damped string
(`docs/dev/parameter-mapping-plan.md`) and the three linear plates
(`docs/dev/plate-mapping-plan.md`) have musical controls; §12 names the membrane as still raw.

**Status: planned 2026-10-18.**

**The human's three decisions, 2026-10-18:**

1. **Shapes: both** the viewer offers — the round drumhead and the rectangle.
2. **Inputs: head presets and a switch.** A head type (cited film presets) and a size are always
   set; then *either* the tension (pitch is read out) *or* a target pitch (the tension is worked
   out), as the plate's thickness-or-pitch switch. Two ring times, as the string and the plate.
3. **Pages: the drum, the mallet-struck drum — and a new drum-in-a-room page.** The viewer has no
   page for the core's `RoomGrid<MembraneSeam>` today; this batch builds one, with the musical
   controls on it. There the air lowers the pitch and adds damping, so the targets are met by the
   drum *before* the air and the page reports how far the room moves them.

---

## 1. What a musician sets, and what the model needs

| control | meaning | drives |
|---|---|---|
| head | a preset (§4) or a custom areal density | `ρ` (kg/m²) |
| size | radius (circle), or `Lx`, `Ly` (rectangle) | unchanged — already inputs |
| set by | `tension` or `pitch` | which of the next two is the input |
| tension `T` (N/m) | — | `c = sqrt(T / ρ)` |
| `f0` (Hz) | pitch of the lowest mode | `c`, then `T = ρ c²` |
| `t60_low` (s) | ring time of that mode | `σ₀` (mostly) |
| `t60_high` (s) | ring time at the reference frequency | `σ₁` (mostly) |
| `f_ref` (Hz) | where "high" is measured; picks the anchor mode | — |

`N` and `fs` stay inputs (`fs` becomes explicit on the drum page, where today it is derived from a
`λ` slider; `λ` becomes a read-out held to §2's bound). The raw constructor stays as it is: the
mapping is an optional helper returning raw coefficients (realism offered, never imposed).

**The head's density does not move a pitch.** Only `c` does; `ρ` scales every energy and every
force the mallet and the air see. So the trap the plate named (`ρ_v` for `ρ_s`) has a twin here: a
wrong `ρ` leaves every pitch right. A bar must read a `ρ`-dependent quantity (§5).

**Out of scope:** the von Kármán plate (gong, cymbal); strings other than the damped one; air
correction of the targets in the room (reported, not compensated — §7.3); smoothing a moving
control; learned mappings.

---

## 2. The model change — a frequency-dependent loss on the membrane

Today: `u_tt = c² Δu − 2σ u_t`, stepped **explicitly**; `σ` enters only the `(1 + σk)` divisor.
The change is Bilbao's second loss term, in its **explicit** form:

```text
δ_tt u = c² L u − 2σ₀ δ_t· u + 2σ₁ L δ_t− u
u⁺ (1 + σ₀k) = 2u − (1 − σ₀k) u⁻ + c²k² L u + 2σ₁k L (u − u⁻)
```

`σ` keeps its name and meaning (it *is* `σ₀`); `σ₁ ≥ 0` is new, default `0`.

**Why the backward difference, not the centred one the plate got.** The plate was already implicit,
so a centred term cost nothing. The membrane is explicit, and two clients are built on that:

- `mallet::Params` uses the membrane's driving-point admittance `k² / (ρh²(1 + σk))` — the bare
  nodal mass, no solve (`mallet.rs` §238–280). A centred `σ₁` puts `−σ₁kL` into `u⁺`'s coefficient
  and that formula stops being true.
- `airbox_wrap::MembraneSeam::a_bare` is `(1 + σk) I`, "its emptiness is the tier's main design
  fact".

The backward difference touches only `u` and `u⁻`, so `u⁺`'s coefficient stays `(1 + σ₀k)` and both
clients are untouched. It goes **inside `step_numerator_into`**, so `step_into`, `Membrane::step`
(hence the mallet, which steps through it) and `MembraneSeam::rhs` all carry it from one spelling.

**Its two costs.**

- **A tighter stability bound.** Per mode (`μ` an eigenvalue of `−L`, `μ ≤ 8/h²`) the scheme is
  stable iff `c²k²μ + 4σ₁kμ ≤ 4`, i.e. `λ² + 4σ₁k/h² ≤ 1/2`. Construction refuses past it. At
  `σ₁ = 0` the old `CflViolated` path and message stay byte for byte; a new variant names the loss
  when `σ₁ > 0` is what crosses the line.
- **The energy needs a correction.** Writing `δ_t− = δ_t· − (k/2)δ_tt`, the extra piece moves into
  the stored energy:

  ```text
  E = ρh² [ ½‖δ_t− u‖² − (σ₁k/2)⟨−L δ_t− u, δ_t− u⟩ + (c²/2)⟨−L u, u⁻⟩ ]
  E⁺ − E = −k ρh² [ 2σ₀‖δ_t· u‖² + 2σ₁⟨−L δ_t· u, δ_t· u⟩ ]
  ```

  Its non-negativity is exactly the bound above. The correction is **skipped**, not multiplied by
  zero, at `σ₁ = 0`, so the energy every existing bar, the mallet's books and the room's books read
  stays bit-identical. The balance (energy drop = dissipated work, per step) is the bar.

**The frequency law is forced.** On a membrane `ω ≈ c√μ`, so `σ₀ + σ₁μ` grows as frequency
**squared** — the damped string's law, not the plate's linear one (there `ω ∝ μ`). No local
operator gives a linear law here. Part 0 puts numbers on it.

**Nothing that exists may move.** `σ₁ = 0` leaves every membrane, mallet and room trajectory, and
the 588-request viewer freeze, bit-identical: the term is skipped, not zero-multiplied.
`Params::new` is positional with nine arguments and many call sites; `σ₁` arrives as a setter
(`Params::with_sigma1`) that re-checks the bound, not a tenth argument.

**Clients that rebuild the step must carry `σ₁` or refuse it.** Grep, 2026-10-18:

- `mallet` — steps through `Membrane::step`, and its contact resolve uses the unchanged admittance.
  Carries it; a bar asserts the mallet's ledger closes with `σ₁ > 0`.
- `airbox_wrap::MembraneSeam` — `rhs` is `step_numerator_into`; `a_bare` stays `(1 + σ₀k)I`; its
  `energy` is `Membrane::energy`. Carries it; a bar asserts the room's books close to `1e-10`.
- `engine` — generic over `step()`; nothing to do.
- `initial_previous` ignores `σ₀` (the lossless Taylor start); `σ₁` is treated the same way, and a
  start-up bar says so.

---

## 3. The inversion — exact on both shapes

Each mode `m` (eigenvalue `μ_m` of `−L`) obeys

```text
(1 + s₀) z² + (−2 + X + Y) z + (1 − s₀ − Y) = 0,   s₀ = σ₀k,  X = c²k²μ_m,  Y = 2σ₁kμ_m
```

A target `(f, T60)` is the root pair `r e^{±iφ}`, `φ = 2πfk`, `r = 10^(−3k/T60)`:

```text
1 − s₀ − Y = r² (1 + s₀)          (ring time)
2 − X − Y  = 2r cos φ (1 + s₀)    (pitch)
```

Both linear in `(c², σ₀, σ₁)`. **`σ₁` sits in the pitch equation** — the loss lowers the pitch — so
`c²` must be solved *with* the losses in, and a plant using the lossless pitch must go red.

- The two ring-time equations (mode 1 and the anchor `M`) are a 2×2 in `(σ₀, σ₁)` that does not
  involve `c²`. Then mode 1's pitch equation gives `c²`. The anchor's pitch is a read-out (one wave
  speed, so one free pitch — as the plate).
- **set by tension:** `c` comes forward; the 2×2 gives the losses; every pitch is a read-out.
- **Exact on both shapes.** The loss operator *is* the stiffness operator `−L`, staircased circle
  included — they share every eigenvector by construction. No first-order error like the free
  plate's, no rigid modes to skip.
- **The anchor is a discrete output.** The square grid keeps the circle's `m = 1` pairs degenerate
  and splits its `m = 2` pairs (the grid's four-fold symmetry). `M` is chosen by sorted value and
  count from the lossless spectrum, never by a `(m, n)` label, and its margin to the next mode is
  reported.
- Eigenvalues: the rectangle's in closed form (1-D Dirichlet sums); the circle's from
  `eigs::eigsh_shift_invert` on `−L` (a few near zero for mode 1, a few around the anchor).
- **Refusals:** a singular 2×2; `c² ≤ 0`, `σ₀ < 0`, `σ₁ < 0` (the high ring time's two-sided range,
  as the plate's — both ends named); the anchor is mode 1; §2's bound violated at the mapped values
  (with the `fs` or `N` that would fix it); non-positive or non-finite input.

---

## 4. Heads

Presets of single- and two-ply polyester (PET) film at typical head thicknesses, and calfskin if a
citable areal density exists, each giving `ρ` (kg/m²) — plus "custom" (a density slider). **Every
value from a cited source found in Part 2** (film density from the manufacturer's datasheet, ply
thicknesses from head makers' published specs, any drum measurement from the acoustics literature),
the citation beside the table in code. No number from memory.

Tension is shown, not preset — a drummer tunes by ear; the read-out says what tension that is.

---

## 5. Validation — the bars

Every bar runs the simulation, never the mapping's algebra backwards.

1. **The new loss term, before any mapping.** Balance to rounding per step on both shapes with
   `σ₁ > 0`; monotone decay; each rectangle mode's measured decay and pitch equal §3's recurrence
   with `μ` from the 1-D Dirichlet eigenvalues written in the test; `σ₁ = 0` bit-identical to today
   (state and energy); the stability bound — a run just inside stays bounded, just outside grows,
   construction refuses the second; a start-up bar.
2. **Round trip, single mode,** both directions, both shapes: start in exactly mode 1 (then `M`),
   project, read `r` and `φ` in the well-conditioned form (`ln_1p`, second difference on
   `(a_n, a_n − a_{n−1})`).
3. **Round trip, struck,** through the analysis crate's spectrum detector.
4. **Off every default:** `Lx ≠ Ly`, radius, `N`, `fs`, every head, both directions; grep the
   membrane fixtures for each range first.
5. **Textbook control:** the continuum `c` from `f₀₁ = j₀₁ c / (2πa)` misses the simulated
   fundamental by the staircase's several cents at the viewer's `N` — the batch's payoff on the
   circle. Measured in Part 0 and asserted to exist.
6. **The head density is live:** a strike's energy (or the mallet's contact time) differs between
   two heads at the same pitch, and equals a core membrane built with that head's `ρ`.
7. **Tension round trip:** set by pitch → `T`; set by that `T` → the same pitch.
8. **Clients:** the mallet's ledger and the room's books with `σ₁ > 0`.

**Plants** (each red, each time-limited): `σ₀`/`σ₁` swapped; the `σ₁` term on `u⁺` (centred
spelling) or on `u` only; the energy correction dropped or its sign flipped; the old bound kept
with `σ₁ > 0`; the lossless pitch for `c²`; the continuum `μ`; anchor `M ± 1`; a degenerate twin
counted once; `ρ` from the wrong head; `k = fs`.

---

## 6. The two existing pages

**Drum page** (`crates/physsynth-viewer/src/membrane.rs`): the membrane's `domain` key already means
circle / rectangle, so the control set takes the plate's key, `controls: physical | musical`, and
`set_by: tension | pitch`. A request without `controls` takes today's path byte for byte. In musical
mode `fs` is an input and `λ` a read-out against §2's bound. The energy verdict drops the
single-rate `2σ` oracle line when `σ₁ > 0` (`energy_block(…, sigma == 0.0, 2.0 * sigma, …)`). The
payload gains a `tuning` block: head, source, tension, `c`, `σ₀`, `σ₁`, `λ`, the anchor and its
margin, the lowest partials' pitch and ring time. The horizon strip's continuum reference now
differs from the scheme by the loss's pitch shift too; Part 0 measures whether that is visible.

**Mallet page** (`crates/physsynth-viewer/src/mallet.rs`): builds its drum through
`build_membrane`, so the musical path arrives with it. The mallet's contact shifts pitch only while
it touches; after it leaves the head rings at the mapped partials, so the read-out is measured on
the ring-down. A mallet-specific bar: the struck drum's ring-down sounds `f0`.

**The coverage guard** reads `CONTROL_OPTS`; the membrane and mallet entries go into
`BORN_AFTER_THE_FREEZE` with their covering test files.

---

## 7. The new page — the drum in a room

### 7.1 What exists

Core: `RoomGrid<MembraneSeam>`, `baffled` (flush in a wall, a source) and `suspended` (hanging in
the air, an object), from air-box batch 5 (`docs/dev/air-box-membrane-plan.md`): a circular head
needed a port fix; the air load puts the membrane in a solve (`A` gains the load); the footprint
ceiling is two air cells; a membrane has no coincidence frequency — `c/c₀` decides subsonic or
supersonic at every mode. The viewer's gong-in-a-room page (`vkroom.rs`) is the template: a 3-D
slice set, a mounting select, the room setting the sample rate.

### 7.2 What it shows

The drum's head as a heatmap, the room as three slices, the mic's audio, the energy books (drum,
air, radiated, the cross-ledger residual) — and the musical read-out: targets, the drum's partials
*in vacuo* (exact), and the partials measured in the room. The headline is the shift itself: the
air's added mass lowers the pitch, most at the fundamental, and adds radiation damping; a slack
head (low `c/c₀`) radiates little, a tight one more.

### 7.3 Decisions this page forces (Part 0 measures, the human decides)

- **The room sets `fs`** (the 3-D CFL), so the drum's `λ` is a read-out and §2's bound may refuse
  a heavily damped head on a coarse air grid.
- **Cost.** The gong page runs ~23 s at its default; a drum's head is cheaper than a von Kármán
  plate but the room is the same `h⁻⁴`. Part 0 times a default and picks caps.
- **Mapping in the room:** in-vacuo targets with the shift reported (planned), or a later batch
  that corrects for the air. Not pre-decided beyond this batch.
- Musical controls only, or the physical set too (the page is new; there is no freeze to keep).

**The human's calls after Part 0 (2026-10-18):**

1. **Tune the drum, show the air.** The musical targets are met by the drum in vacuo; the page
   measures each mode's pitch and ring time in the room (the head projected on its in-vacuo modes,
   §9) and shows the shift. One run, no iteration.
2. **Walls: absorbing by default, rigid offered** — absorbing (`ζ = 1` on every face but a baffle)
   is where the shift belongs to the drum; rigid lets the room's resonances take over.
3. **Mounting: both,** in a wall (baffled) and hanging (suspended).
4. **Controls: both, behind the switch** the drum and plate pages use.

---

## 8. Order of work

- **Part 0 — probe** (scratch crate under `W:\temp\claude`, nothing in the repo):
  - (a) The scheme with `σ₁` (backward difference) in scratch on the real `Params.l`: the per-mode
    recurrence against a run, both shapes; the energy with its correction balances; the bound —
    just inside bounded, just outside grows (on the circle, whose `μ_max < 8/h²`, measure the
    margin the sufficient bound leaves).
  - (b) Inversion exactness on both shapes over `N`, `fs`, sizes, loss levels.
  - (c) The frequency-squared law in numbers: e.g. 4 s at 110 Hz and 1 s at 1 kHz ⇒ ? at 3 kHz.
  - (d) Textbook miss on the circle (`j₀₁`) and the rectangle, at the viewer's `N` range.
  - (e) The circle's anchor margins (split `m = 2` pairs) at the viewer's `N`.
  - (f) The room: pitch and ring-time shift of a mapped drum, baffled and suspended, at two
    tensions; the cost of a default run; the `fs` the room forces against §2's bound.
  - The result goes to the human with the room decisions of §7.3.
- **Part 1 — model:** `σ₁` on `Membrane` (setter, bound, step, energy), its bars (§5.1), the
  clients' bars (§5.8).
- **Part 2 — heads:** cited presets.
- **Part 3 — mapping:** `membrane_tuning.rs`, both directions, refusals, bars §5.2–7, plants.
- **Part 4 — drum and mallet pages,** backend then front-end, guard, headless cases.
- **Part 5 — the room page,** backend, front-end, headless case.
- **Part 6 — docs:** hurdle §12, HANDOFF §8, memory.

Commit and push at each part's end, `scripts/cargo-test-nice.ps1 -Full` before each.

---

## 9. Part 0 result (2026-10-18)

Probe: a scratch crate depending on core and analysis by path, nothing in the repo changed. It
adds §2's term to a copy of the step (calling core's own `step_numerator_into` and then the
backward-difference `2σ₁kL(u − u⁻)`), runs it on real `membrane::Params`, and mounts real heads in
`RoomGrid<MembraneSeam>`.

- **The recurrence is the scheme.** A run started in one exact mode (rectangle: `sin·sin`;
  circle: `eigsh_shift_invert` on `−L`) follows §3's per-mode recurrence to `1.3e-10` cents in
  pitch and `7e-13` in ring time, both shapes, three modes each.
- **The energy books close only with the correction.** Struck, `σ₀ = 1.5`, `σ₁ = 2e-4`: per-step
  balance `2.9e-15` (rectangle), `3.6e-15` (circle) of the start, monotone; `σ₁`'s own share of
  the work is 15 % of the start. Without the correction term the balance misses by `4e-6`–`7e-6`
  of the start — far above rounding, so a dropped correction is visible to the balance bar.
- **`σ₁ = 0` leaves the state bit-identical** to core's `Membrane::step` (500 steps).
- **The stability bound is right, and the sufficient form is tight.** The exact limit is
  `c²k²μ_max + 4σ₁kμ_max ≤ 4`; at `0.999×` it a run decays, at `1.001×` it grows by `10¹⁴`–`10¹⁵`
  over 20k steps. The sufficient `λ² + 4σ₁k/h² ≤ ½` (`μ_max ≤ 8/h²`) gives away only
  0.16–0.28 % of `σ₁` (`μ_max h²` is 7.98–7.996). The refusal uses the sufficient form.
- **The inversion is exact on both shapes.** 192 admissible fixtures (rectangles 0.30 and
  0.45 m, circles 0.12 and 0.33 m; `N ∈ {30, 60}`, `fs ∈ {44.1k, 96k}`, `f0 ∈ {80, 220}`, three
  loss levels, anchors near 3× and 8× the fundamental): worst `1e-8` cents and `5e-12` in ring
  time, read back from the recurrence; a run on every eighth rectangle agrees to `4e-9`.
- **The loss's own pitch shift is small at musical losses** (0.001–0.04 cents for the modes
  measured), so the plant "lossless pitch used for `c²`" needs a heavy loss to go red.
- **The frequency law, in numbers** (4 s at 110 Hz, 1 s at 1 kHz): the drum's law gives 3.2 s at
  300 Hz, 0.31 s at 2 kHz, 0.14 s at 3 kHz, 0.05 s at 5 kHz; the plate's linear law would give
  2.4, 0.54, 0.37, 0.23 s. Highs die fast — the price of a local operator.
- **The textbook mapping misses the round drum by 10–25 cents flat** (`c` from `j₀₁`: 25 cents
  at `N = 40`, 16 at 60, 14 at 80, 10 at 100) — the staircase rim. Ring times off by 0.3–2.5 %.
  The rectangle's textbook miss is small (1.2 cents at `N = 30`, 0.1 at 80). The round drum is
  where this batch pays.
- **The circle's spectrum has near-degenerate clusters.** Exact pairs (the `m ≥ 1` cos/sin twins)
  come out as multiplicity 2; the grid also splits some pairs and brings unrelated modes close —
  smallest relative gaps `4.6e-5` to `6.7e-4` in frequency, between 2× and 5× the fundamental. An
  anchor inside such a cluster is still exact (each mode has its own `μ`), and its neighbour rings
  within a hair of it; the margin is reported.

**The room** (rigid walls, `fs = 40 kHz`, `h_air = 16.5 mm`, round head 0.12 m, 0.26 kg/m²,
`T = 3000 N/m`, `N = 24`, 0.4 s). Every peak listed, not one detector window — a ±40 % window
first reported "+560 cents" by picking a different peak.

| room (m) | mount | fundamental | shift | wall-clock |
|---|---|---|---|---|
| in vacuo | — | 337.0 Hz | — | — |
| 0.46 × 0.46 × 0.15 | baffled | 315.3 Hz | −115 c | 1.7 s |
| 0.46 × 0.46 × 0.15 | suspended | 256.8 Hz | −470 c | 1.8 s |
| 0.66 × 0.66 × 0.40 | baffled | 290.1 Hz | −260 c | 8.2 s |
| 0.66 × 0.66 × 0.40 | suspended | 265.4 Hz | −413 c | 8.1 s |
| 0.99 × 0.86 × 0.73 | baffled | 289.1 Hz *and* 232.0 Hz | −267 c / −640 c | 39.0 s |
| 0.99 × 0.86 × 0.73 | suspended | 272.5 Hz | −370 c | 38.8 s |

In the largest room the baffled head's fundamental **splits in two** around a room resonance — the
head and the room share the note, so "the drum's pitch in the room" is not one number there.
Cost grows as the room's volume over `h³`: ~4 s, ~20 s, ~100 s per second of sound.

**Each mode labelled** (the head projected on its in-vacuo eigenvectors, the peak of each modal
coordinate — the table above lists the six biggest peaks, which are not modes). Mode 0 is the
fundamental; modes 1–2 the first cos/sin pair; shifts in cents from the in-vacuo partial.

| head (N/m) | room | walls | mount | mode 0 | modes 1–2 | modes 3–4 | head energy kept |
|---|---|---|---|---|---|---|---|
| 1000 | 0.46 | rigid | baffled | +63 | −178 | −100 | 0.93 |
| 3000 | 0.46 | rigid | baffled | −115 | −661 | −99…−116 | 0.81 |
| 8000 | 0.46 | rigid | baffled | −209 | −198 | −116…−137 | 0.53 |
| 3000 | 0.46 | absorbing | baffled | −250 | −153 | −104 | 0.02 |
| 3000 | 0.66 | absorbing | baffled | −293 | −159 | −105 | 0.02 |
| 3000 | 0.46 | absorbing | suspended | −442 | −231 | −152 | 0.02 |
| 3000 | 0.66 | absorbing | suspended | −391 | −222 | −149 | 0.12 |
| 8000 | 0.46 | absorbing | baffled | −265 | −176 | −124 | 0.00 |
| 8000 | 0.66 | absorbing | baffled | −283 | −173 | −122 | 0.00 |

(Absorbing: `Wall::Impedance(ρ₀c₀)` on every face but the baffle, which stays rigid.)

- **Rigid walls: the room's resonances set the note.** Shifts jump by mode and by room size
  (modes 1–2 at −661 cents next to −99 for modes 3–4; a slack head's fundamental pushed *up*
  63 cents by the small sealed room). Not a property of the drum.
- **Absorbing walls: the shift belongs to the drum.** Two room sizes agree within 1–6 cents above
  the fundamental and within 40–50 cents at it (the rigid baffle and oblique incidence still
  reflect). The air lowers the fundamental by **2½–4½ semitones** (baffled about −2½ to −3,
  suspended about −4) and the higher modes by one to two — mass loading of a light head, both
  faces when suspended. The order of the in-vacuo partials survives.
- **Radiation damping dominates once the walls absorb:** the head keeps 2 % of its energy over
  0.4 s at 3000 N/m (≈ a 1.4 s ring time from the air alone) and none at 8000. With rigid walls
  the energy comes back from the room and the head keeps 53–93 %. Either way the air, not `σ₀`
  or `σ₁`, can set the ring time; a target longer than the air allows cannot be met.
- **The room's sample rate does not bite §2's bound.** At 40 kHz the 0.12 m head (`N = 24`) runs at
  `λ ≈ 0.27`, so `σ₁` may reach `≈ 0.43 m²/s`; the mapped `σ₁` for musical ring times is
  `~1e-3`.
- **So "targets met before the air" would leave the room drum 1–5 semitones flat**, and with
  absorbing walls its ring times bounded by radiation. §7.3's mapping question is the page's main
  decision.

---

## 10. Part 1 result — the loss term (2026-10-18)

`membrane::Params::sigma1`, set by `Params::with_sigma1` (refuses a negative or non-finite value,
`NegativeSigma1`, and `λ² + 4σ₁k/h² > ½`, `LossCflViolated`; `σ₁ = 0` returns the membrane as it
was, the plain CFL path untouched). `step_numerator_into` gains `2σ₁kL(u − u⁻)` and `energy` the
correction `−(σ₁k/2)⟨−Lδ_t−u, δ_t−u⟩`, both skipped at `σ₁ = 0`. No client changed: the mallet
steps through `Membrane::step`, the room seam's right-hand side is `step_numerator_into` and its
energy `Membrane::energy`.

Bars, `crates/physsynth-core/tests/membrane_loss.rs` (nine), `−L` re-applied from the mask in the
test and the rectangle's eigenvalues closed form:

- three rectangle modes follow their recurrence with `σ₀ + σ₁μ` to `1e-9` of the amplitude; the
  same run without `σ₁` misses by `> 1e-4`;
- the energy books close per step to `1e-10` of the start on both shapes, `σ₁`'s own work above
  5 %; judged by the `σ₁ = 0` energy formula the same run misses by `> 1e-7` (the control);
- `σ₁` alone is passive on both shapes;
- `σ₁ = 0` is the old membrane bit for bit (state and energy), and a membrane exactly at the plain
  CFL ceiling stays legal with `σ₁ = 0` and is refused with any `σ₁ > 0`;
- the start-up is the lossless Taylor step either way;
- the bound: accepted at `0.999×` the sufficient limit, refused at `1.001×`; the grid's top mode
  stays bounded at `0.99×` the *exact* limit and grows past `10³` at `1.01×` (set by hand past the
  refusal), the sufficient limit within 1 % below the exact one;
- the struck drum's books (coupled total plus the head's `σ₁` work) close to `1e-10`;
- the drum in a room, baffled and suspended: scene total plus the head's loss work flat to
  `1e-10`, the port's two ledgers agree to `1e-12`.

**Plants, all red** (a copy of the file restored after each, compared byte for byte): the term on
`u` alone; at half strength; with its sign flipped; the energy correction dropped; its sign
flipped; the bound not checked.

---

## 11. Parts 2 and 3 result — heads and the mapping (2026-10-18)

`crates/physsynth-core/src/membrane_tuning.rs`: `HEADS` (four clear polyester-film heads by ply
construction — single-ply 3, 7.5 and 10 mil, two-ply 7 + 7 mil), each `areal_density()` =
`FILM_DENSITY · t`. Sources: Remo's product pages for the constructions (Ambassador Hazy Snare Side,
Diplomat, Ambassador, Emperor — accessed 2026-10-18) and DuPont's *Mylar* typical-properties sheet
for the film density, 1.390 g/cm³ (ASTM D1505). The makers' pages say "film" without naming it;
PET is assumed. Clear heads only — a coating or a collar adds mass the table does not carry.

`tune_membrane(geometry, rho, targets)` → `MembraneTuning` (the built `Params` with both losses,
the tension, `σ₀`, `σ₁`, the fundamental and the anchor in the scheme, the anchor's margin, both
modes' `μ`). `SetBy::Pitch` or `SetBy::Tension`. The rectangle's eigenvalues come from the closed
form, the circle's from `eigsh_shift_invert` (three near zero, ten around the anchor). The two
ring-time equations give `(σ₀, σ₁)` by a 2×2 whichever is set; set by pitch, mode 1's pitch
equation then gives `c²` with the losses in. The tuned membrane is built through `with_sigma1`, so
the loss-tightened bound refuses as `TuningError::Membrane(LossCflViolated)`.

Bars, `crates/physsynth-core/tests/membrane_tuning.rs` (seven):

- every head retyped from its plies and 1.390 g/cm³; names unique; sources named;
- **64 round trips** read back from the running scheme (rectangles 0.30 and 0.45 m, circles 0.12
  and 0.33 m; `N ∈ {30, 56}`, `fs ∈ {44.1k, 96k}`, two heads, set by pitch and by a tension 10 %
  off the pitch run's): worst **`5.7e-9` cents** and **`4.9e-12`** in ring time, against bars of
  `1e-6` and `1e-7`;
- the anchor is the partial nearest `f_ref`, three references on each shape;
- **struck** round head through the analysis crate's detector: the mapped drum `−0.0002` cents
  from 180 Hz; the textbook mapping (`c` from `j₀₁`) on the same grid `−25.0` cents;
- two heads at the same targets: same wave speed and losses, tension and energy in the ratio of
  their densities, the energy bit-identical to the raw constructor's membrane;
- set by pitch, then by its tension: the pitch back to `1e-9` cents;
- refusals: the anchor at the fundamental, `f_ref` past Nyquist, both ends of the high ring
  time's range (and just inside each tunes), non-positive inputs, the plain CFL, the loss CFL.

**Plants, all red:** `σ₀`/`σ₁` swapped into the membrane; the lossless pitch used for `c²`; the
rectangle's `x` eigenvalues on the `y` axis; the anchor farthest from `f_ref`; a 20 dB ring time;
the tension without the density; the film density 1395.
