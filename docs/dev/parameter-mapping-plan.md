# Parameter mapping — musical controls for the damped string

`docs/dev/scientific-hurdles.md` §12 ("raw physics is not a musician's interface"), opened
2026-10-18 on the human's call. HANDOFF §8 budgets "real work" for it; §12D proposes learning it.
This batch does the first, hand-derived piece for one model.

**The human's three decisions, 2026-10-18:**

1. **Instrument: the damped stiff string (model #3)** first. Plates and the membrane come later.
2. **Reach: helpers *and* viewer sliders** in this batch, not helpers alone.
3. **Accuracy: corrected to the simulation**, not textbook. "Play 220 Hz and ring for 2 s" means
   the *rendered* string does that, not a continuum string the scheme then detunes.

---

## 1. What a musician sets, and what the model needs

| musical control | meaning | model coefficient it drives |
|---|---|---|
| `f0` (Hz) | pitch of the fundamental | tension `T` (via `c`, given `ρ`, `L`) |
| `inharmonicity` `B` | how far the partials stretch above `m·f0` — 0 is a perfect string, ~1e-4 a piano mid-range, ~1e-2 a bar-like string | `κ` |
| `t60_low` (s) | ring time of the fundamental | `σ₀` (mostly) |
| `t60_high` (s) | ring time at the reference frequency | `σ₁` (mostly) |
| `f_ref` (Hz) | where "high" is measured; picks the anchor mode | — |

`L`, `ρ`, `N`, `fs`, `θ` stay as they are: inputs to the mapping, not outputs. The raw
coefficients and `Params::new` are **untouched** — the mapping is an optional helper that returns
them (the human's rule since model #9's `EA` call: realism is offered, never imposed). It refuses only what
model #3 cannot represent (§3).

"Stretched" means the textbook stiff-string law, which the targets are written in:

```text
f_m = m f0 sqrt((1 + B m²) / (1 + B))          (so f_1 = f0 exactly)
```

**Out of scope:** smoothing a moving control (a real-time-stage concern, HANDOFF §8); any other
model; learned mappings (§12D); a "material" or "size" vocabulary (that is the plate's batch).

---

## 2. The inversion is exact, not iterative

Model #3's mode `m` (an exact discrete eigenvector: simply supported, `sin(mπx/L)`) obeys the
two-step recurrence `a z² + b z + cc = 0` with, writing `X = k² Q_m`, `Y = k σ_m`,
`Q_m = c² p_m² + κ² p_m⁴`, `σ_m = σ₀ + σ₁ p_m²` and `p_m²` the **discrete** `−D₂` eigenvalue:

```text
a  = 1 + θX + Y
b  = −2 + (1 − 2θ) X
cc = 1 + θX − Y
```

(`physsynth-analysis::damping::decay_roots_ac`, the oracle the damped string's decay bars use).
A mode that sounds at `f` and rings for `T60` is the root pair `z = r e^{±iφ}` with
`φ = 2π f k` and `r = 10^(−3k/T60)` (amplitude down 60 dB, a factor 1000). Product and sum of the
roots give `cc/a = r²` and `−b/a = 2r cos φ`, which are **linear in `X` and `Y`**:

```text
θ(1 − r²) X − (1 + r²) Y            = r² − 1
−[(1 − 2θ) + 2θ r cos φ] X − 2r cos φ Y = 2r cos φ − 2
```

So each target `(f, T60)` at one mode gives that mode's exact `(Q_m, σ_m)` by a 2×2 solve. Two
anchor modes — mode 1 and the anchor `M` nearest `f_ref` on the stretched law — then give `(c², κ²)`
and `(σ₀, σ₁)` by two more 2×2 solves. No iteration, and the loss's effect on pitch (the damped
root sits slightly flat of the lossless one) is inside the same equations.

Lossless check: `r = 1` forces `Y = 0` and leaves `4s = X(1 − 4θs)`, `s = sin²(φ/2)` — the
θ-scheme dispersion relation `docs/dev/theta-loss-compensation-plan.md` §2 starts from.

**What "exact" covers.** Modes 1 and `M` land on target to rounding. Every *other* mode lands where
the scheme puts it, which is off the stretched law by the scheme's own dispersion — the pitch error
of hurdle §15 and the rate suppression of §4, which this mapping does not and cannot remove (§4
showed they are one factor). That residual is **measured and reported**, not hidden (§5, §6).

The existing `loss_coefficients_from_t60` (analysis crate, continuum, frozen fixture) stays exactly
as it is: it is the *textbook* mapping, and this batch uses it as the comparison the corrected one
must beat.

---

## 3. Refusals — what model #3 cannot be

The mapping returns `Err` with a plain reason, never a silently wrong string, when:

- the 2×2 at either anchor is singular, or gives `X ≤ 0` (no positive stiffness sounds there);
- the anchor frequency is beyond what the scheme can produce at this `fs`, `θ` (for `θ ≥ 1/4` the
  lossless pitch is capped at `sin²(φ/2) < 1/(4θ)`);
- `c² ≤ 0`, `κ² < 0` (targets shrink the partials, `B < 0`), `σ₀ < 0` or `σ₁ < 0` (`T60` rising
  with frequency — no passive string does that; `Params::new` refuses it too);
- `M = 1` (`f_ref` at or below the fundamental: one anchor cannot separate two unknowns);
- any input non-positive or non-finite.

The resolution horizon (hurdle §15) is **reported**, not refused: the payload says up to which
partial the non-anchor modes stay within ±5 cents of the stretched law. Whether an anchor above
that horizon should refuse is a Part 0 question, put to the human with the measurements.

---

## 4. Where the code lives

- **`crates/physsynth-core/src/string_tuning.rs`** (new): the pure mapping. Core because a plugin or
  a WASM build calls it at runtime and core is the portable, dependency-free crate. It brings its
  own discrete `p_m²` — the analysis crate's `spatial_eigenvalue_p2` is not reachable from core.
- Returns a struct: `t` (given `ρ`), `kappa`, `sigma0`, `sigma1`, the anchor `M`, and the derived
  `c`, `λ`, `B` as the model will report it.
- Tests that need the analysis oracles live in core's `tests/` through the existing test-only
  dependency.

---

## 5. Validation — the bars

Every bar runs **the simulation**, never the mapping's own algebra backwards (CLAUDE.md: a bar
about a formula must not import the formula).

1. **Round trip, single mode.** Map targets → `Params`; start the string in exactly mode 1 (then
   exactly mode `M`); step it; project the state onto the mode each step; read `r` and `φ` from the
   measured amplitudes. Assert frequency and T60 equal their targets to ~1e-9 relative.
2. **Round trip, plucked.** A broadband pluck, the analysis crate's spectrum detector on the
   output: mode 1 and mode `M` within a measurement tolerance stated with its reason.
3. **Off every default.** `L`, `ρ`, `N`, `fs`, `θ` (`0.25`, `0.28`, `0.5`, `1`), `B` (0 and
   non-zero), `f_ref` across several anchors. Grep model #3's fixtures for each range first.
4. **Textbook control.** The continuum mapping (`c = 2 L f0`, `loss_coefficients_from_t60`) on the
   same targets misses by a measured amount, and the bar **asserts the miss exists** — so the
   corrected mapping visibly differs from its control, and the size of the miss is printed.
5. **Refusals,** one bar per §3 case, each asserting the specific error.
6. **Off-anchor report.** For each fixture, the worst cents error of modes `2..M−1` and above `M`
   against the stretched law — printed and pinned loosely, so a change in it is noticed.

**Plants** (each must turn a bar red; give each a time limit): swap `σ₀` and `σ₁`; hard-wire `θ`
to 0.5; flip the sign of `B`; continuum `p_m²` (`(mπ/L)²`) in place of the discrete one; anchor
`M ± 1`; `k = fs` instead of `1/fs`; `r` from a 20 dB instead of 60 dB drop.

---

## 6. The viewer

**Backend** (`crates/physsynth-viewer/src/string.rs`, model `"damped"` only): new **optional**
request keys `f0`, `inharmonicity`, `t60_low`, `t60_high`, `f_ref`, `fs`. When `f0` is present the
string is built from the mapping: `T`, `kappa`, `sigma0`, `sigma1` come from it, and `fs` is
explicit (in this mode `λ` becomes a read-out instead of an input, because `f0` sets `c`). The
payload gains a `tuning` block: the derived physics values, the anchor mode, the targets against
the measured partials, and the horizon read-out. Refusals surface as the viewer's normal parameter
refusal.

**The freeze must not move.** No existing key's default or meaning changes; a request without `f0`
takes today's path byte for byte. `crates/physsynth-viewer/tests/frozen.rs` (588 payloads) stays
green unchanged. New requests get their own viewer tests.

**Front-end** (`web/static/`): a "musical controls" toggle on the damped-string panel that swaps
the `T`/`κ`/`σ₀`/`σ₁` sliders for pitch / inharmonicity / two ring times / reference frequency, and
shows the derived physics values read-only underneath. The browser check
(`verify_headless`) gains a step that renders in musical mode.

---

## 7. Order of work

- **Part 0 — probe** (Rust, scratch crate under `W:\temp\claude`, nothing in the repo): confirm the
  §2 inversion round-trips through `DampedStiffString` to rounding across the §5.3 grid; measure the
  textbook miss and the off-anchor residual. Anything surprising goes to the human before Part 1.
- **Part 1 — core:** `string_tuning.rs`, unit tests, refusals.
- **Part 2 — the bars:** §5's round trips and controls, then the plants.
- **Part 3 — viewer backend:** optional keys, `tuning` block, new viewer tests, freeze unchanged.
- **Part 4 — front-end:** the toggle, the read-outs, the headless browser check.
- **Part 5 — docs:** hurdle §12 updated, HANDOFF §8's mapping line, memory.

Commit and push at each part's end, `scripts/cargo-test-nice.ps1 -Full` before each.
