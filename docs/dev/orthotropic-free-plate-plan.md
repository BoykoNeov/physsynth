# Orthotropic free plate — plan (model #5of: the *free* plate gets a grain)

> **Status: IMPLEMENTED (2026-08-17).** Four-constant orthotropy in
> `operators2d.free_plate_stiffness` (one code path), `grain_coupling`/`grain_torsion` on `Plate`'s
> free branch, `plate.grain_ratios_from_material` returning the split, oracles
> `modal.free_plate_twist_bound` / `free_plate_coupling_form`, suite in
> `tests/test_free_plate_orthotropic.py` (30 tests), diagnostics in
> `scripts/diagnose_orthotropic_free_plate.py`. Check 1 (§2) was the gate and cleared before any
> core edit. The claims in §1 and §5 were **pre-registered before running** and are left standing;
> three died, and §8 records what the build itself changed.
>
> **What the gate changed, and it is most of what this batch is worth so far:**
> - **The isotropic collapse is bit-exact on all 7 grids and all 4 Poisson values (Q0)** — so this
>   branch gets a **single code path**, the opposite of #5o's conclusion for the supported branch.
>   The reason is structural, not luck: the four coefficients multiply the *same four matrix
>   products in the same order*, at `1.0`, `1.0`, `nu` and an exactly-representable halving.
> - **Q3's sharpness prediction died.** The pointwise condition `g_1² < g_x g_y` is *not* the sharp
>   discrete boundary: the discrete operator stays semi-definite 4–20% past it, by a margin that
>   **shrinks under refinement** (19.8% on 7x5, 4.2% on 24x24). So it is the sharp *continuum*
>   condition and a *provably sufficient, measurably conservative* discrete guard — which is what
>   makes it the right thing to ship, and a genuine contrast with #5o's guard, which was sharp.
> - **Q7 died twice over, and is demoted from a detector to a documented physical note.** The
>   coupling term leaves a `y`-independent field's Rayleigh quotient **exactly** unchanged (it is a
>   *shape* change, not an energy change), and the resulting eigenvalue shift is **downward for a
>   strip and upward for a square**. The narrow-strip `(1 - nu_xy nu_yx)` limit is approached only
>   roughly (7.5% against 9.0% predicted) and confounded with cross-width resolution. It is not an
>   oracle and will not be asserted as one.
> - **Q8 got stronger than planned, and it matters because it is the only detector of `g_1`.** The
>   coupling probe is not `O(h)`: it has the **exact discrete closed form**
>   `4 g_1 h² (Nx-1)(Ny-1)`, hit to 6e-16 when the other three terms are switched off (and to
>   ~1e-13 when they are on and have to cancel). It is a tier-2 machine-precision test, not a
>   tier-3 convergence one.
> - **Both headlines lived (§5), and H1's control is exact.** At fixed `H` the free plate's
>   fundamental moves **6.5x** while the supported plate's operator is **bit-identical** (`0.0`).
>   The mode-order swap H2 predicted is exhibited, with the crossing measured at
>   `g_y/g_xy = 1.025` — real spruce sits 12.6% above it, on the twist-first side.

## 0. What this batch is, in one line

Model #5o gave the plate a grain direction, but only for the branch whose rim is pinned all round
(simply supported). This batch gives the grain to the **completely free** plate (model #5b) — the
boundary a real guitar or violin top actually has before it is glued in, and the one luthiers tap.

It was named as the next step by #5o itself (`orthotropic-plate-plan.md` §3, "Also out of scope"),
for a stated reason: the free branch assembles from the **strain energy** and therefore needs the
coupling and torsional rigidities *separately*, not just their combination `H`. That is the whole
technical content of this batch, and §1.1 is where it earns or loses its oracles.

## 1. The physics, and what the free boundary demands that the supported one did not

Orthotropic Kirchhoff bending energy (Timoshenko & Woinowsky-Krieger, orthotropic-plate chapter;
Leissa, *Vibration of Plates*, NASA SP-160, ch. 11):

    U = 1/2 ∫∫ [ D_x w_xx² + 2 D_1 w_xx w_yy + D_y w_yy² + 4 D_xy w_xy² ] dA

    D_1 = nu_yx D_x = nu_xy D_y      (coupling rigidity)
    D_xy = G_xy t³ / 12              (torsional rigidity)
    H    = D_1 + 2 D_xy              (the PDE's cross term — model #5o's third number)

The bilinear form the free branch needs is the polarization of that:

    P(f, g) = ∫∫ [ D_x f_xx g_xx + D_y f_yy g_yy + D_1 (f_xx g_yy + f_yy g_xx)
                   + 4 D_xy f_xy g_xy ] dA

**Why `H` is not enough here, stated so it can be checked rather than believed.** Take the
Euler–Lagrange equation of that energy on a domain with *no* boundary terms and only `H = D_1 + 2D_xy`
survives — which is exactly why the supported plate needs three numbers and not four, and why its
sine stayed an exact eigenvector. Integrating the twist term by parts twice is what merges `D_1` and
`2 D_xy`, and *that integration by parts leaves boundary terms which vanish only because the
supported rim pins them*. On a free rim they do not vanish: the free edge conditions are

    M_n = 0        (zero bending moment)      -> sees D_1 explicitly
    V_n = 0        (zero Kirchhoff shear)     -> mixes D_xy into the edge condition
    corner force   2·(2 D_xy) w_xy = 0        -> pure D_xy

so two materials with identical `H` and different splits are the *same* supported plate and
**different** free plates. Pre-registered prediction, measured as §2 Q5: the split is worth tens of
percent on the free plate's fundamental at fixed `H`. If that comes back at a fraction of a percent
the batch's premise is wrong and the honest outcome is a paragraph saying so.

Dimensionless ratios against a reference rigidity `D_ref = D_x`, matching #5o's convention
(`kappa² = D_ref/rho_s` carries the scale):

    g_x = 1,   g_y = D_y/D_x,   g_1 = D_1/D_x,   g_xy = D_xy/D_x,   g_h = g_1 + 2 g_xy

**Isotropic reduction.** `D_x = D_y = D`, `D_1 = nu D`, `D_xy = (1-nu) D / 2` gives
`4 D_xy = 2(1-nu) D` and `H = D` — i.e. the shipped free-plate form
`f_xx g_xx + f_yy g_yy + nu(f_xx g_yy + f_yy g_xx) + 2(1-nu) f_xy g_xy` exactly. Which is the
first gate question, because the shipped isotropic path must not move.

### 1.1 Four constants, and one detector each — the part that decides whether this batch is buildable

The free plate has no closed-form spectrum (model #5b already lives with that). What makes this
batch acceptable under the project's rule (b) — *a new model needs an oracle, not a convergence
rate* — is that the four constants separate into four independent probes, three of them exact.

**(a) `g_xy` — the twist quotient, and the numerator is exact on every grid.** The centred saddle
`w = x·y` (coordinates from the plate's centroid) is odd in both axes, so it is `W`-orthogonal to
the rigid-body space `{1, x, y}` *exactly* by grid symmetry, which makes the Rayleigh quotient a
legitimate upper bound on the first elastic eigenvalue. On the saddle `w_xx = w_yy = 0` and
`w_xy = 1`, so every term except the torsional one vanishes:

    R(xy) = (xy)ᵀK(xy) / (xy)ᵀW(xy),      continuum limit  576 D_xy / (rho_s a²b²)
    i.e.  omega_twist ≤ 24 sqrt(D_xy/rho_s) / (a b)

Two separate claims fall out, and they are pre-registered separately because they can fail
separately. **(i)** `R(xy)` must be *exactly independent* of `g_x`, `g_y` and `g_1` — the collocated
second differences annihilate the saddle and the cross term with it — a machine-precision test that
isolates the new torsional constant and nothing else. **(ii)** the numerator should be
**exact on any grid**: `hᵀ Dxy(xy) = 1` on every cell, so `(xy)ᵀK(xy) = 4 g_xy h² · N_x N_y = 4 g_xy ab`
with no discretization error at all; the `O(h²)` in `R(xy)` should live entirely in the trapezoidal
mass `(xy)ᵀW(xy) → a³b³/144`. Predicted, to be measured as Q1/Q2.

This is the free plate's *physical* claim as well as its test. The frequency above depends on the
torsional rigidity **alone** — it is why the wood literature reads `G_xy` off a tapped free plate
(Caldersmith, *Vibrations of orthotropic rectangular plates*, Acustica 56 (1984) 144–152;
McIntyre & Woodhouse, on measuring orthotropic sheet constants). For an isotropic square at
`nu = 0.3` it gives `omega a² sqrt(rho_s/D) ≤ 24 sqrt(0.35) = 14.20` against Leissa's tabulated
13.47 for the FFFF fundamental — a 5.4% one-term Rayleigh overshoot, i.e. the bound is *informative*
and not vacuous. Its blind spot is named in §7: it is **one-sided**, so an operator that is
uniformly too soft passes it.

**(b) `g_x`, `g_y` — an exact reduction to the shipped 1-D free beam, but only at zero coupling.**
Feed the operator a field `w(x)` constant in `y`. The `y`-curvature and the twist both annihilate it
identically, and the generalized problem collapses: with `w = 1_y ⊗ v`,

    K w = g_x · kron(m_y, S_x v)        W w = kron(m_y, M_x v)
    =>  K w = mu W w   <=>   S_x v = mu M_x v      (the 1-D free-beam problem, exactly)

with `S_x, M_x = operators.free_beam_stiffness` — the already-validated model #5b-pre, which has its
own closed-form `cosh·cos` oracle. So at `g_1 = 0` the free plate's `y`-independent spectrum must
equal the free beam's, scaled by `sqrt(g_x)`, to machine precision. Setting `g_1 = 0` is permitted
(the standing rule: the core API admits combinations no material has, and realism is offered by a
helper rather than imposed).

**And at `g_1 ≠ 0` it must NOT hold — for a physical reason, which is the trap this batch has to
avoid writing as a bug.** The coupling term does *not* vanish on a `y`-independent field: it
contributes on the two `y` free edges (the transpose of the cross operator hits the trapezoidal
edge weights, which do not cancel there). That is **anticlastic curvature** — bend a strip one way
and it curls the other — and it is exactly why a wide plate is not a beam of rigidity `E_x t³/12`.
Pre-registered: the beam reduction is exact at `g_1 = 0`, breaks at `g_1 > 0`, and breaks *upward*
(a stiffer, higher mode). Q6/Q7. Anyone who "fixes" the Q7 discrepancy has broken the physics.

**(c) `g_1` — a closed-form bilinear probe that no other detector can see.** On the pair
`f = x²`, `g = y²` every term dies except the coupling one, and its continuum value is known
exactly:

    P(x², y²) = ∫∫ D_1 (f_xx g_yy + f_yy g_xx) dA = 4 D_1 · a b

Discretely the collocated second differences give `2` at every interior node and `0` on the
respective free edges, so the discrete form returns `4 g_1 h² (N_x-1)(N_y-1)` — the same number
short of one boundary strip, i.e. `O(h)` relative. A convergent probe rather than an exact one, but
it pins the coupling rigidity with an *analytically known coefficient*, and it is the only detector
of the four that responds to `g_1` at all in isolation. Q8.

### 1.2 Admissibility is a different condition than the supported branch's

The theta-scheme is unconditionally stable for `theta >= 1/4` provided the spatial operator is
positive semi-definite. Here that is a condition on the pointwise curvature form
`[[g_x, g_1], [g_1, g_y]]` plus the torsional term:

    g_x > 0,   g_y > 0,   g_1² < g_x g_y,   g_xy > 0

Isotropic material sits at `g_1² = nu² < 1` and `g_xy = (1-nu)/2 > 0`; note the plate form needs
only `|nu| < 1`, looser than the `(-1, 1/2)` the code already enforces for 3-D thermodynamic
admissibility. **This is not the same set as #5o's supported guard `g_h > -sqrt(g_x g_y)`.** Every
`H` the supported plate accepts has *some* admissible split (take `g_1` small and `g_xy` large), but
a caller who fixes both halves can land outside — so the free branch needs its own construction-time
rejection, and the two guards must not be conflated. Sharpness measured at Q3.

## 2. Check 1 — the gate. No core edit before this table is filled

`check1_orthotropic_free.py` (scratch, not shipped), on the 7-grid survey #5o used plus a spruce
material point. Predictions are written before running.

| # | Question | Prediction | Result |
|---|---|---|---|
| Q0 | does the general 4-constant assembly at `(1, 1, nu, (1-nu)/2)` reproduce the shipped free `K` **bit-for-bit**, on all 7 grids | bit-exact everywhere (same terms, same order, multipliers `1.0` and an exact halving) — **unlike** #5o's supported case, so a single code path is possible here | **bit-exact**, all 7 grids x `nu` ∈ {0.3, 0, 0.49, −0.5}. Single code path |
| Q1 | is `R(xy)` exactly independent of `g_x`, `g_y`, `g_1` | machine precision | **1.6e-13** relative over four wildly different `(g_x, g_y, g_1)`. Not 1e-16: the bending terms are *numerically cancelling*, not absent |
| Q2 | is the twist numerator `(xy)ᵀK(xy) = 4 g_xy a b` exact on every grid, and does `R(xy)` approach `576 g_xy/(a²b²)` at `O(h²)` | numerator exact; quotient `O(h²)` through the mass only | numerator **7.5e-14** relative — and **2.5e-14 even with the other three terms switched off** (§8.2: `h²` times `1/h⁴` carries roundoff whether or not anything cancels, so "exact" here means *in exact arithmetic*). Quotient orders **1.95, 1.99, 2.00** — `O(h²)` confirmed, and it does live in the mass |
| Q3 | is `g_1² < g_x g_y` (with `g_xy > 0`) the sharp semi-definiteness boundary | indefinite at 1.02x, PSD-with-3-zeros at 0.98x | **DIED.** Still PSD at 1.02x. Discrete threshold bisected: **1.042 (24x24) … 1.198 (7x5)** — conservative by 4–20%, shrinking with refinement. Separately `g_xy = 0` adds a **4th** zero mode (the saddle joins the nullspace) and `g_xy < 0` is violently indefinite (−7.3e2) |
| Q4 | does the grain **reorder** the free plate's modes | **yes** — and this is the direct contrast with #5o, where ordering never moved. Spruce puts the twist at `lambda ≈ 6.0` and cross-grain bending at `lambda ≈ 6.1`: a near-tie the solver has to break | **yes.** Spruce: twist 5.687, cross-grain bend 5.976 (5.1% apart) against isotropic 13.44 / 19.53 (45% apart). Swap exhibited at `g_y = 0.0546` (bend 5.191 *below* twist 5.642). Isotropic λ₁ = 13.444 vs Leissa 13.47 ✓ |
| Q5 | at fixed `H`, how far does the fundamental move across the admissible split range | tens of percent (the batch's premise) | **6.51x** (λ₁ 0.924 … 6.013) over the admissible split at `g_h = 0.153`; **1.6x** if restricted to the plausible half `|g_1| ≤ 0.1`. Non-monotone, peaking at `g_1 = 0` |
| Q6 | at `g_1 = 0`, does the `y`-independent spectrum equal `sqrt(g_x)` x the shipped free beam's, exactly | machine precision | **yes** — eigen-residuals 2.2e-12, 2.2e-13, 6.5e-14 on the first three beam modes at `g_x = 2.3` |
| Q7 | at `g_1 > 0`, by how much does it break, and in which direction | breaks upward (anticlastic stiffening); physical, not a bug | **DIED twice.** The Rayleigh quotient is **exactly unchanged** (`1.00000000`) — a shape change, not an energy change. The eigenvalue moves **down** for strips (0.925, 0.926, 0.933 at 4:1, 10:1, 20:1 against the `1 − g_1²/(g_x g_y) = 0.910` strip limit) and **up** for a square (1.183). Demoted to a note |
| Q8 | does the `(x², y²)` probe return `4 g_1 ab` at `O(h)`, and does the material chain land `g_1 + 2 g_xy` on #5o's `grain_cross` exactly | yes; exact agreement (same expression) | **better than predicted.** Exact *discrete* form `4 g_1 h²(Nx−1)(Ny−1)` to **5.9e-16 … 1.9e-14** (other terms off). Continuum `4 g_1 ab` is the `O(h)` statement (orders 0.95→0.99). Spruce `g_1 + 2 g_xy = 0.152914644628`, identical expression to #5o's `grain_cross` |
| Q5b | is the supported plate *exactly* invariant to the split at fixed `H` | yes, by construction | **`0.0` exactly**, four splits, on a grid #5o chose because it distinguishes assemblies. This is H1's null control and it is not approximate |

### 2.1 The three results that change the build

**Q0 -> one code path.** The free branch routes the isotropic default through the general assembly.
#5o's decision went the other way for the supported branch and this plan predicted the difference
before measuring it; the test pins bit-identity on the same grids, including the two where #5o's
supported paths differ, so it cannot pass vacuously.

**Q3 -> the guard is conservative, and says so.** Ship the pointwise condition
`g_x > 0`, `g_y > 0`, `g_1² < g_x g_y`, `g_xy > 0`. It is *provably* sufficient (the energy is a sum
of pointwise forms in `(w_xx, w_yy)` plus a non-negative twist term, so positive-definiteness of the
2x2 block gives semi-definiteness on any grid) and measurably conservative by 4–20% on coarse grids
with the margin vanishing as `h -> 0`. The assertion in the suite is therefore **one-sided**: inside
the guard, semi-definite on every grid tested; the exact discrete boundary is *not* claimed. The
`g_xy = 0` finding is the more useful half — it is not merely inadmissible, it puts the saddle
**into** the nullspace, which is the cleanest possible statement of what the torsional rigidity is
for.

**Q7 -> deleted as a detector, kept as a sentence.** The detector list in §1.1 is now three probes,
not four, plus the isotropic bit-identity. The physics stays in the docstring: a coupling rigidity
makes a free plate's beam-like mode a *different shape* rather than a different energy, and whether
that lands above or below the beam depends on the aspect ratio.

## 3. Scope, and the refusals

**In scope:** four-constant orthotropy in `operators2d.free_plate_stiffness`; the free branch of
`Plate` (lifting its `NotImplementedError`); the material chain returning the split; the twist-bound
and coupling-probe oracles in `analysis/modal.py`; one new test module; one diagnostic script.

**Refused, and each for a stated reason:**

- **Orthotropic von Kármán (nonlinear wood).** Still out, and still for #5o's reason: the Airy
  stress solver inverts an *in-plane* biharmonic that assumes isotropic in-plane compliance, which
  an orthotropic sheet replaces with a four-constant compliance tensor — new machinery with no
  closed-form oracle. This batch is its prerequisite (it shares the strain-energy assembly), not its
  substitute. `VKPlate` keeps calling `free_plate_stiffness` with isotropic arguments and must come
  out **bit-identical**; that is a test, not an assumption.
- **A full tabulated FFFF-orthotropic spectrum as an absolute anchor.** The obvious external
  anchors (Narita 1981; the orthotropic entries behind Leissa ch. 11) are paywalled, and this
  project's own rule is that cited digits beat memory — so no recalled table will be entered as an
  oracle. §1.1's four probes are derived here and independently checkable, which is why they are the
  spine instead. If open cited digits turn up during the build they go in as a *fourth* tier-3
  anchor, not as the basis.
- **A guitar-shaped (non-rectangular) top.** The rectangle is what has a separable operator and
  the beam reduction. Arbitrary outlines are the staircasing problem the membrane batch already
  paid for, in a 4th-order operator. Named because it is the thing this unlocks *next*, not now.
- **Frequency-dependent (anisotropic) loss.** The plate's damping caveat is already broader than
  the stiff string's, and a grain makes decay direction-dependent. Out; the caveat text gets one
  sentence saying the grain now enters it.

## 4. Build order — oracles first, core untouched until Q0 clears

1. **Check 1** (`M:\claud_projects\temp\...`, throwaway): fills §2. Any of Q0/Q1/Q6 failing changes
   the design before a line of core moves.
2. **Oracles** in `analysis/modal.py`: `free_plate_twist_bound` (the `24 sqrt(D_xy/rho_s)/(ab)`
   continuum bound) and `orthotropic_coupling_form` (the `4 D_1 ab` probe value). Core untouched.
3. **Operator**: `free_plate_stiffness` gains `grain_x, grain_y, grain_coupling, grain_torsion`,
   defaulting to the `nu`-derived isotropic split. Single code path if Q0 is bit-exact; a separate
   path if it is not (#5o's precedent).
4. **Resonator**: `Plate`'s free branch accepts the split and drops the `NotImplementedError`.
   `GrainSpec` grows the two fields — see §6.
5. **Tests**: `tests/test_free_plate_orthotropic.py`. The shipped `test_free_plate_modal.py` and
   `test_vk_free.py` must pass **unchanged**.
6. **Diagnostics**: `scripts/diagnose_orthotropic_free_plate.py` — Chladni patterns for spruce vs
   isotropic at matched `H`, the split sweep of Q5, the beam-reduction break of Q7.
7. **Docs**: this file's record section, HANDOFF §12B (which currently names this batch as the open
   one), the module docstrings, and the memory file.

## 5. The headlines, pre-registered so they can die

**H1 — the free plate can read the split that the supported plate provably cannot.** Same `H`, same
`D_x`, same `D_y`, two different `(D_1, D_xy)` splits: the supported plate's spectrum is *identically
unchanged* (its eigenvalue depends on `g_h` only — already proven, not measured), while the free
plate's fundamental moves by tens of percent. **Killed if** the free spread is small — then `H` is
effectively sufficient for both boundaries and the batch is a formality.

**H2 — the grain reorders the free plate's modes, though it never reorders the supported plate's.**
#5o's surviving result was that the cross term detunes selectively but never changes mode order,
anywhere between solid wood and isotropic material. The free plate should behave differently,
because its low spectrum is a race between a twist mode governed by `D_xy` alone and two
beam-like bending modes governed by `D_x` and `D_y`: spruce ratios put the twist and the cross-grain
bending within ~1% of each other. **Killed if** the ordering is stable across the same sweep — in
which case #5o's "ordering is grain-invariant" generalizes to both boundaries, which is a *better*
result than the one predicted here and should be stated as such.

**H3 (a claim about our own docs, and the safest of the three).** The free plate's fundamental is
the saddle/twist — model #5b's finding. With a grain, `D_xy` is roughly `0.06 D_x` for spruce, so the
saddle's frequency drops by ~4x relative to the isotropic plate of the same along-grain stiffness.
If H2 lives, "the fundamental is the twist" stops being a fact about free plates and becomes a fact
about *isotropic* free plates — a shipped sentence this batch would have to correct.

## 6. API: the one decision worth arguing about

`Plate` currently takes `grain_x, grain_cross, grain_y` (three, supported-only) and `nu` (free-only).
The free branch needs the split, so `grain_cross` cannot be its input. The design:

- add `grain_coupling` (`D_1/D_ref`) and `grain_torsion` (`D_xy/D_ref`), both `None` by default;
- both `None` -> today's behaviour on both branches, byte for byte;
- supplied -> the supported branch derives `grain_cross = g_1 + 2 g_xy` (so one material chain feeds
  both boundaries), and a caller who *also* passes a contradicting `grain_cross` is refused rather
  than silently overridden;
- `nu` becomes `None`-defaulted. Passing both `nu` and a split is **refused** — otherwise `nu` is a
  silently ignored argument, which is the exact failure class this project keeps finding. With a
  split supplied, `self.nu` is set to the *implied* `nu_yx = g_1/g_x`, which reduces to `nu` in the
  isotropic case, so nothing downstream reads a fiction;
- one-of-two supplied is refused (no defaulting the other half).

`GrainSpec` grows `grain_coupling` and `grain_torsion`. It is a `NamedTuple` and six shipped call
sites unpack it **positionally** (`_, _, gx, gh, gy = ...`), so appending fields breaks them: those
six get migrated to attribute access in the same commit. Recorded here because it is a deliberate
break of a shipped internal surface, not an accident — and because attribute access is what the
`GrainSpec` docstring already argues for.

## 7. Blind spots, stated before the tests are written

The plate/room family's standing rule is that no single detector is sufficient, and every batch has
found a new *kind* of hole. The ones visible from here:

- **The twist bound is one-sided.** A uniformly-too-soft operator passes it. Covered only by the
  combination with Q6's exact beam reduction (two-sided) and `O(h²)` self-convergence.
- **No detector sees all four constants.** By construction: that is what makes them independent
  probes. The hazard is the inverse — a bug in one *term* is invisible to three of the four tests,
  so the coupling probe (the weakest, `O(h)`) is the only guard on `g_1` and must not be dropped for
  being loose.
- **The energy ledger is blind here, and more so than in #5o.** Any symmetric `K` conserves energy
  exactly, so a wrong coefficient is perfectly stable and perfectly conservative. #5o already said
  the correctness claim rides on the modal oracle; with four constants and no closed-form spectrum
  it rides on §1.1's four probes, and the ledger's role shrinks to catching the time-stepper.
- **The isotropic bit-identity is a weak test of the general path** — it exercises exactly one point
  of a four-dimensional space, and that point is symmetric in `x`/`y`. A non-square grid with
  `g_x ≠ g_y` and `g_1 ≠ 0` is needed to catch an axis swap, and the shipped explicit-loop reference
  assembly in `test_free_plate_modal.py` should be extended to four constants rather than trusted.

## 8. The build record — what shipping it changed

Everything §2 measured survived into tests. Four things the *build* added, in descending order of
how much they change what a reader should believe.

**9.1 A guard I wrote rejected a valid material, and the test that caught it is the one worth
keeping.** `free_plate_stiffness` validated its `nu` argument against `(-1, 1/2)` unconditionally —
correct for the isotropic case, wrong the moment a split supersedes it, because an orthotropic
sheet's implied `nu_yx = D_1/D_x` is bounded by `nu_xy nu_yx < 1` and not by one half. A free plate
at `g_1 = 0.70`, `g_y = 0.5` — inside this branch's own admissibility guard — was refused by the
*isotropic* bound one level down. `nu` is now validated only where it actually supplies a missing
half of the split, and `test_an_implied_poisson_ratio_above_one_half_is_admissible` pins both
directions: the orthotropic call builds, the plain isotropic call with that `nu` is still refused.
The general lesson is the one this family keeps relearning from a different angle: **a superseded
argument that is still being validated is a live constraint**, and the two admissibility notions
here (3-D thermodynamic, 2-D orthotropic-plate) are genuinely different sets.

**9.2 "Exact" needed downgrading twice, in the same direction.** Two closed forms that are exact in
exact arithmetic are not exact in doubles, for the same reason: the terms are products of `h²` with
`1/h⁴`, so the sums carry ordinary roundoff even when the algebra is a clean cancellation.

| quantity | predicted | measured (clean) | measured (with bending on) |
|---|---|---|---|
| twist numerator `4 g_xy ab` | machine | **2.5e-14** rel | ~1e-13 rel |
| `R(xy)` blindness to `g_x, g_y, g_1` | machine | — | **1.6e-13** rel |
| coupling probe `4 g_1 h²(Nx-1)(Ny-1)` | `O(h)` (wrong) | **5.9e-16 … 1.9e-14** | ~1e-12 |

The bars in the suite are set at the measured magnitudes with the reason written beside them, and
each test *also* asserts the ordering — the clean build must be the closer of the two — so a future
change that introduces cancellation where there was none fails rather than passing at a loose bar.

**9.3 The split sweep changes the fundamental's identity, not just its frequency.** The plan's Q5
reported a 6.5x span in `lambda_1` and read it as detuning. The shipped diagnostics label each mode,
and the low end of the sweep is a *different mode*: at zero and negative coupling the fundamental is
the cross-grain bender, and from `g_1 = 0.05` upward it is the twist. So H1's number is a
mode-crossing plus a detuning, which is a stronger statement than the plan made and a more careful
one than "the fundamental moves 6.5x".

**9.4 The twist bound's overshoot is nearly material-independent.** Measured margin between the
Rayleigh bound and the true fundamental: **5.27%** isotropic, **5.55%** spruce — despite the two
plates' fundamentals differing by a factor of 2.4. So the bound's tightness is a property of the
one-term trial function rather than of the material, which is what makes a single `< 15%` bar
meaningful across both arms instead of needing one bar per material.

**Not done, and not silently:** the free plate's O(h²) self-convergence is asserted on the grained
plate (orders 2.011, 2.001, 2.006), but there is still **no absolute external anchor** for the
orthotropic free spectrum — §3's refusal stands, and the three probes plus the beam reduction are
what stand in for it. If open cited digits for FFFF orthotropic plates turn up later, they go in as
an added tier-3 anchor and nothing here needs to move.

## 9. References

- S. Bilbao, *Numerical Sound Synthesis* — plates, energy methods, stability.
- S. Timoshenko & S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed. — the orthotropic
  strain energy and the free-edge conditions.
- A. W. Leissa, *Vibration of Plates*, NASA SP-160 — FFFF isotropic tables (already the anchor for
  model #5b); ch. 11 for the orthotropic case.
- G. W. Caldersmith, "Vibrations of orthotropic rectangular free plates", *Acustica* **56** (1984)
  144–152 — the free orthotropic plate as the luthier's measuring instrument.
- N. H. Fletcher & T. D. Rossing, *The Physics of Musical Instruments*, ch. 3 — the wood-plate
  twist/bending mode picture.
