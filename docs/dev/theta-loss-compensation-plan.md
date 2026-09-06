# The θ-scheme's rate suppression — the probe, and the claim it falsified

`docs/dev/scientific-hurdles.md` §4, opened 2026-09-06 on the human's call. This document is the
Part 0 probe and its result. **The probe killed the payoff claim the batch was chosen on**, and §4
assigns that decision to the human.

**The human's decision, 2026-09-06: record the finding, do not build the compensation, and open the
wider thread instead** — the probe's real content is that the *pitch* error bounds every accuracy
claim this project makes about a mode, and no model has that boundary written down. §4 below stays
as a scoped, costed option so that a future decision is made against a real number; nothing in it
has been started.

Probe scripts: `M:\claud_projects\temp\theta-loss\probe.py`, `probe_lock.py`, `probe_tail.py`,
`probe_turnover.py`. Nothing in the repository was changed to run them.

---

## 1. What §4 says, and what was going to be built

Every implicit θ-scheme string and plate here discretizes loss as `−2σ δ_t· u` (plus
`+2σ₁ δ_t· δ_xx u` on the string) while the stiffness sits under the θ average. In the modal domain
with `Q = c²p² + κ²p⁴` the per-step decay is

```
Γ_m ≈ 2 σ_eff(m) / (1 + θ k² Q_m)        (continuum: 2 σ_eff(m))
```

so every mode's decay is **suppressed** by `S_m = 1/(1 + θk²Q_m) ≤ 1`, increasingly with `m`. Model
#3's whole audible claim — "highs die faster" — therefore turns over somewhere and reverses.

§4 derives the cure and calls it "the smallest physics change with an audible payoff in the whole
register": pre-multiply the loss operator by the θ-denominator,

```
−2σ δ_t· u   →   −2 (σ₀ I − σ₁ D₂)(I + θ k² 𝓛) δ_t· u      with  𝓛 = −c²D₂ + κ²D₂²
```

which makes `σ_eff → σ_eff(1 + θk²Q)` in the modal domain and `g_m = (1 − σ_eff k)/(1 + σ_eff k)`
for **every** mode — the continuum rate up to the mode-independent bilinear warp. The planned batch
was: an oracle branch, a `Params` flag defaulting off, a bandwidth change in `banded.rs`
(pentadiagonal → heptadiagonal), and the test that cannot exist today, "the rate rises monotonically
over the whole resolved band".

---

## 2. The probe result: the rate suppression and the pitch error are THE SAME NUMBER

The θ-scheme's lossless amplification factor satisfies

```
1 − cos(ωk) = k²Q / (2 (1 + θk²Q))      ⇒      sin²(ωk/2) = k²Q / (4 (1 + θk²Q))
```

so in the small-argument limit `ω/ω_c = 1/√(1 + θk²Q) = √S`, while the decay rate carries `S`
itself. **The discrete frequency is suppressed by the square root of the same factor that suppresses
the decay rate.** Written in the unit a listener uses:

```
pitch error (cents) = 600 · log₂(S)          S = the decay-rate suppression
```

Measured over 108 string configurations — `κ ∈ {0, 0.5, 2, 5}`, `θ ∈ {0.28, 0.5, 1}`,
`λ ∈ {0.5, 1, 2}`, `m ∈ {4, 16, 40}` at `N = 128` — against `analysis/modal.py`'s discrete and
continuum frequencies:

| band | worst \|actual − 600 log₂ S\| |
|---|---|
| `S > 0.999` | 0.57 cents |
| `S > 0.99` | 0.57 cents |
| `S > 0.95` | 11.2 cents |
| `S > 0.90` | 59.4 cents |

The lock is analytic, not coincidental, and it is **exact where any "fix this" claim would live**.
Outside the small-argument regime it loosens, and it loosens in the direction that strengthens the
conclusion: the *actual* pitch error is then larger than `600 log₂ S`, not smaller.

### 2.1 What that costs the payoff claim

The two errors arrive together and cannot be separated, so the question is only which is more
audible. Taking pitch JND ≈ 5 cents on a sustained tone and decay-time JND ≈ 10–20% (both cited
figures, not measured here — at 10 cents and 5% the ratio changes but not the direction):

| T60 comes out too long by | `S` | pitch is flat by |
|---|---|---|
| 1% | 0.9901 | **−8.6 cents** |
| 2% | 0.9804 | −17.1 cents |
| 5% | 0.9524 | −42.2 cents |
| 10% | 0.9091 | **−82.5 cents** |
| 20% | 0.8333 | −157.8 cents |
| 100% | 0.5 | −600 cents |

By the time the decay error reaches one JND (~10% long), the mode is **more than three quarters of
a semitone flat** — of order fifteen pitch JNDs. The compensation repairs the *less* audible half of
a single defect, and the half it leaves behind is the one a listener notices first.

### 2.2 The plate does not escape it, and the coarse-timestep plate is worse in both

The prior expectation — that the plate is the bigger payoff, because `Q = κ²p⁴` has no Courant bound
on `k/h²` — does not survive the lock. Larger suppression is larger *mistuning*. On the suite's own
supported-rectangle fixtures at `σ = 1`:

| fixture | mode | `S` | pitch error |
|---|---|---|---|
| `N=16, μ=2` | (1,1) | 0.9934 | −4.0 cents |
| `N=16, μ=2` | (4,4) | 0.3941 | −604 cents |
| `N=16, μ=2` | (15,15) | 0.0140 | −3243 cents |
| `N=16, μ=0.05` | (15,15) | 0.9579 | −26 cents |

### 2.3 The one artifact that is independently visible — and it is a timestep artifact

A mode damped 71× too slowly still carries energy, so the surviving question was *cleanliness*
rather than fidelity: after a broadband strike, does the tail belong to modes the scheme resolves,
or to junk that should have died? Measured by fitting the log-energy tail of a narrow strike:

| fixture | corner mode's own T60 | measured tail |
|---|---|---|
| plate `N=16, μ=2` | 71.3× too long | **7.41× too long** |
| plate `N=16, μ=0.5` | 5.4× too long | 1.17× too long |
| plate `N=16, μ=0.05` | 1.0× | 1.00× |
| string `N=128, λ=1`, σ₀=2, σ₁=1e-4 | 2.21× (m=127) | 1.27× too long |

So there *is* a gross artifact — but only at a large timestep, and it vanishes as `k` shrinks. `k`
is the same knob that fixes the pitch. The string barely shows it at all.

### 2.4 The ordering claim never fires in tune — 90 configurations, the best case is 330 cents off

The lock compares one mode's rate against its own pitch, but σ₁'s actual job is the *ordering*.
`test_sigma1_makes_high_partials_die_faster` asserts the rise only over `[1..16]` precisely because
the rate turns over. So: does the turnover ever land inside a band a listener would call in tune?

Swept over `κ ∈ {0.5, 2, 5}`, `N ∈ {128, 256, 512}`, `λ ∈ {1, 0.5}` and five T60 target pairs, with
`(σ₀, σ₁)` derived the way a user would derive them — `loss_coefficients_from_T60` at the
fundamental and mode 20 — the turnover mode and its pitch error:

| κ | N | T60 lo → hi | σ₀ | σ₁ | turnover | pitch there | m at −5 cents |
|---|---|---|---|---|---|---|---|
| 0.5 | 128 | 5.0 → 0.8 | 1.362 | 1.97e-3 | 127 | −1684 cents | 6 |
| 2.0 | 128 | 5.0 → 0.8 | 1.362 | 1.97e-3 | 52 | −748 cents | 6 |
| 2.0 | 512 | 5.0 → 0.8 | 1.363 | 1.85e-3 | 100 | −521 cents | 21 |
| 5.0 | 128 | 10.0 → 0.5 | 0.654 | 3.71e-3 | 32 | −574 cents | 6 |
| 5.0 | 512 | 5.0 → 0.8 | 1.363 | 1.86e-3 | 62 | −456 cents | 16 |

**Best case over the entire sweep: the turnover at m = 26, where the pitch is 330 cents flat** — a
minor third. The pitch passes 5 cents at mode 6 to 24 depending on the grid; the ordering artifact
arrives hundreds of cents later, every time. Realistic σ₁ makes this *worse*, not better: the suite's
fixture uses `σ₁ = 1e-4` while a real T60 pair asks for `1.8e-3` to `3.7e-3`, and a larger σ₁ pushes
the turnover further out of reach (the numerator grows like `p²`, the denominator like `p⁴`).

---

## 3. What is falsified, and what still stands

**§4's observations all hold.** The turnover exists. The rates are suppressed. The plate inherits it
with a fourth-power denominator and inherits it worse. §4's and `tests/helpers.py`'s descriptions of
the *mechanism* are correct — unlike §5, where the probe found the mechanism itself wrong.

**What is falsified is the payoff framing** — "the one artifact a later calibration would fit around
rather than through", and "the smallest physics change with an audible payoff in the whole register".
There is no register in which it is audible before the same mode's pitch error is audible by a wide
margin, because the two are one factor and one square root apart.

**What a build would still buy, stated honestly:**

* The rate's error constant improves by roughly seven orders of magnitude — from `θk²Q` (order
  `1e-1` at the top of the band) to the bilinear warp's mode-independent `σ²k²/3` (order `1e-9`).
  Under HANDOFF non-negotiable #1, "accuracy first", removing one systematic error is worth doing
  even when a second remains, and the compensated scheme is strictly closer to the physical string:
  identical in one coordinate, better in the other.
* `loss_coefficients_from_T60` starts delivering what it promises. Today it is 2.75% long at its own
  second target mode and 19% long by mode 40 — real errors, below the decay JND at the target.
* The coarse-timestep plate's 7.4× tail goes away, on a fixture whose partials are already a fifth
  flat.
* The test §4 says cannot exist today — the rate rising monotonically over the whole resolved band —
  becomes writable, and would be a genuine new bar on the scheme.

**What it does not buy:** anything a listener can hear that a smaller `k` would not fix better, and
`k` fixes both halves at once.

---

## 4. The build, if the human wants it — scoped but not started

Recorded here so the decision is made against a real cost, not a vague one. Five parts, string
first because the plate has the harder passivity question.

* **Part 1 — the oracle branch.** `decay_roots_ac` in `crates/physsynth-analysis/src/damping.rs`
  takes `compensated: bool`; `σ_eff → σ_eff · (1 + θk²Q)`. Threaded through all four consumers and
  the pyfunction defaults in `crates/physsynth-py/src/analysis.rs`, exposed as a *keyword* on the
  existing Python names rather than a new name — `__all__` must not grow, or
  `test_analysis_frozen.py`'s derived guard fails and cannot be satisfied (there is no second
  implementation left to freeze against; the guard's own message says the honest option is a native
  bar). The 62 frozen fixtures call positionally and keep hitting the default branch.
* **Part 2 — the scheme.** `Params::new` in `string_damped.rs` gains `compensate_loss: bool`. Build
  `M = (σ₀I − σ₁D₂)(I + θk²𝓛)` explicitly with `Csr::matmul` and **assert `M.is_symmetric()` at
  construction**, refusing like `NotFactorable` does — for a product of two symmetric matrices that
  check *is* the commutation test, and it carries to the masked plate for free. `A = I + kM + θk²𝓛`;
  `step_rhs` gains the matching `+ k M u_prev`.
  Commutation holds by construction on the supported string: `ops::biharmonic_matrix` is literally
  `d2.matmul(&d2)`, so both factors are polynomials in one SPD matrix.
* **Part 3 — the bandwidth, which is where this batch would silently break.** `M` carries a `D₂³`
  term whenever `σ₁κ ≠ 0`, so `A` goes pentadiagonal → heptadiagonal, `kd` 2 → 3. The literal `2` is
  written into `apply_ainv`'s `cho_solve_banded_upper` call and into `Params.chol`'s doc comment,
  and `apply_ainv` has three external consumers — `bow`, `collision::BarrierString` and the
  `connection` bridges — plus the binding and the viewer. Store `kd` in `Params` and grep every
  reader of `chol` and every hardcoded `2` across both crates, the binding and the Python side
  before changing anything. This is the shape of the mallet batch's scar: an override added in one
  part and not routed in another is worse than one never added, because unrouted is at least
  self-consistent.
* **Part 4 — the tests.** The monotone-rate bar over the whole resolved band. Per-mode simulated
  decay against the compensated oracle at the existing 5e-4, at the modes where model #3 turns over
  today (32, 64, 100). Passivity at large σ. Two free anchors: `σ₀ = σ₁ = 0` with the flag **on**
  must be `array_equal` to the flag off, and the flag-off path must be *untouched code* rather than a
  re-derivation that happens to agree — the damped↔stiff↔nonlinear↔geometric `array_equal` chain
  depends on it. The θ/κ/mode-independence of the compensated decay factor is an **exact** claim and
  belongs as a native bar in `crates/physsynth-analysis/tests/`, not as a simulated test: a
  simulated envelope still carries θ-dependent frequency and the lossless-start transient.
* **Part 5 — the plate, only if Part 4's numbers justify it.** `sigma` compensation on the supported
  rectangle. `SparseLu` means no bandwidth question at all, but the guitar outline's masked
  operators need not commute, which is exactly what Part 2's `is_symmetric` guard would catch.

**Deliberately not in scope even if built:** `string_stiff`, `string_nonlinear` and
`string_geometric` do not get the flag. That leaves the `σ₁ = 0`, `EA = 0` and `EA = T` reductions
valid only with compensation off — a known asymmetry, not a bug, and it must be written into the
anchor tests' comments if this ships.
