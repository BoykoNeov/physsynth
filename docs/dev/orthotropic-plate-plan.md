# Orthotropic plate — plan (model #5o: the plate gets a grain direction)

> **Status: IMPLEMENTED (2026-08-17).** `operators2d.orthotropic_biharmonic`, three `grain_*`
> keywords on `Plate` (supported branch), `plate.grain_ratios_from_material`, oracles
> `modal.orthotropic_plate_freqs` / `discrete_orthotropic_plate_eigenfrequency` /
> `dirichlet_axis_eigenvalue`, suite in `tests/test_plate_orthotropic.py` (22 tests), diagnostics in
> `scripts/diagnose_orthotropic_plate.py`. Written after
> the check-1 paper/numeric pass, which is the gate this batch had to clear before any core edit; it
> cleared it on every count (§2).
>
> **What changed against the plan, and it is most of what the batch is worth:**
> - **§2 Q1 was under-measured.** The isotropic collapse is *grid-dependent*: bit-exact on some
>   grids, 1.7–2.4e-16 on others. Neither can be relied on, which strengthens rather than weakens
>   the design decision (§4 Step 2). The test now pins the default path on a grid **chosen because
>   the two assemblies differ there**, and says so — a grid where they coincide would pass while
>   proving nothing.
> - **The headline (§5) died.** The cross term does **not** reorder the modes anywhere between solid
>   wood and isotropic material. What replaced it is better: it is a *selective detuner*, moving
>   modes by 1.3% to 29% over the same sweep — 22× spread — with the leverage largest exactly where
>   the direct stiffness is weakest. See §7.1.
> - **The second finding (§5) also died as posed, and its replacement is the useful one:** a bridge
>   point does notice the grain, but only in the *partial series*, not in the *level* — and the
>   level number the first draft quoted was one draw from a ±20% geometry-dependent spread. See §7.2.
> - **One test was a tautology and was rewritten as a measurement** (§7.3).

## 0. What this batch is, in one line

Every plate in the project so far is made of a material that is equally stiff in every direction.
Real soundboards are wood, and wood is roughly ten times stiffer along the grain than across it.
This batch gives the **simply-supported** plate (model #5) three bending stiffnesses instead of
one — along, across, and a cross term — and validates the result against the closed-form
orthotropic Navier law.

The reason this is the batch and not one of the more exciting neighbours is the project's own
acceptance contract: **(a)** switching the new term off must reproduce a shipped model, and
**(b)** the new model must have a *closed-form* oracle, not a convergence rate. Orthotropy is the
only candidate on the shortlist that has both. (§8 records what was rejected and why.)

## 1. The physics, and the convention this batch commits to

Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., orthotropic-plate chapter;
Leissa, *Vibration of Plates* (NASA SP-160), ch. 11:

    rho_s w_tt = -( D_x w_xxxx + 2 H w_xxyy + D_y w_yyyy ) - 2 rho_s sigma w_t

    H = D_1 + 2 D_xy,    D_1 = nu_yx D_x = nu_xy D_y,    D_xy = G_xy t^3 / 12

**The factor of 2 sits in the PDE (multiplying `H`), not inside `H`.** This is the single most
mis-cited thing in the orthotropic literature and the whole batch is anchored on it, so it is
written down here and asserted in a test rather than trusted. §2 Q3 measures what the two rival
packagings would cost: `D_1` alone is 0.30x the correct term and `D_1 + D_xy` is 0.65x — a 3.3x and
a 1.5x error in the cross-grain stiffness, both of which produce a perfectly stable,
energy-conserving, completely wrong plate.

Setting `D_x = H = D_y = D` recovers `D (w_xxxx + 2 w_xxyy + w_yyyy) = D grad^4 w` exactly — the
shipped model #5.

### 1.1 Why the simply-supported branch keeps its exact eigenvector

`Plate`'s supported branch builds `B = L @ L` from the *Dirichlet* Laplacian, which is the
tensor-product 5-point operator on a rectangle. Splitting it as `L = Dxx + Dyy` and noting the two
commute on a separable rectangle gives

    L @ L = Dxx@Dxx + 2 Dxx@Dyy + Dyy@Dyy

which is **exactly the isotropic case of the orthotropic assembly**. So orthotropy is not a new
stencil: it is the same three matrix products with three coefficients instead of one,

    B_ortho = g_x (Dxx@Dxx) + 2 g_h (Dxx@Dyy) + g_y (Dyy@Dyy)

and `sin(m pi x/Lx) sin(n pi y/Ly)` stays an **exact discrete eigenvector**, because it is already
an exact eigenvector of `Dxx` and `Dyy` *separately*. That is what buys the closed-form oracle, and
it is the reason this batch is cheap.

Discrete eigenvalue, with the positive 1-D Dirichlet eigenvalues
`lam_x = (4/h^2) sin^2(m pi h / 2Lx)` and likewise `lam_y`:

    Q_mn = kappa^2 ( g_x lam_x^2 + 2 g_h lam_x lam_y + g_y lam_y^2 )

Continuum limit:

    f_mn = (pi/2) sqrt( [D_x (m/Lx)^4 + 2H (m/Lx)^2 (n/Ly)^2 + D_y (n/Ly)^4] / rho_s )

### 1.2 The stability guard is not the CFL — it is definiteness of the cross term

The theta-scheme is unconditionally stable for `theta >= 1/4` **provided the spatial operator is
positive semi-definite**, which the isotropic `L @ L` is for free (a square). With three
coefficients it is a condition:

    g_x a^2 + 2 g_h a b + g_y b^2 > 0  for all a, b > 0   <=>   g_h > -sqrt(g_x g_y)

Measured sharp to 2% either side (§2 Q4). Rejected at construction. Note this is *permissive* by
design: real materials sit well inside it — spruce at 0.567 of `sqrt(D_x D_y)` and isotropic
material at exactly 1.0 — and the guard admits the whole band including negative `H`, per the
standing project rule that unphysical parameter combinations are a feature of the core API and
realism is offered through a helper, not imposed. **No hard upper admissibility bound on `H` is
claimed here**: this batch measured two points (0.567 and 1.0) and did not establish a ceiling, and
§7.1 deliberately probes 2x, which is past solid wood but is not asserted to be unreachable.

## 2. Check 1 — the gate, and what it returned

`check1_oracle.py`, run before any core edit, on a 13x9 grid:

| Question | Result |
|---|---|
| Q0 does the tensor split `Dxx + Dyy` reproduce the repo's own `laplacian_from_mask`, index order and all | **0.0** exactly |
| Q1 does `B_ortho(D,D,D)` collapse to `D * L@L` | **2.6e-16** relative — *reassociation only, not bit-exact* (but see below: this turned out to be grid-dependent) |
| Q2 is the sine still an exact discrete eigenvector, with the predicted eigenvalue | worst **1.9e-13** over 12 modes |
| Q2b does the continuum closed form collapse to the shipped isotropic law | **2.6e-16** |
| Q3 does the material chain land `H` on `D` at isotropic constants | **0.0** exactly; rivals at 0.30x, 0.65x |
| Q4 is the definiteness boundary `g_h = -sqrt(g_x g_y)` sharp | indefinite at 1.02x, SPD at 0.98x |

**Q1 is load-bearing and is the reason for a design decision — and the check-1 number understated
it.** Re-measured across seven grids during the build:

| grid | `h` | collapse |
|---|---|---|
| 12×12, Lx=1 | 1/12 | **bit-exact** |
| 24×24, Lx=1 | 1/24 | **bit-exact** |
| 20×14, Lx=1 | 0.05 | **bit-exact** |
| 17×17, Lx=1 | 1/17 | **bit-exact** |
| 13×9, Lx=0.62 | 0.047692… | 2.4e-16 |
| 16×16, Lx=0.7 | 0.04375 | 1.7e-16 |
| 11×7, Lx=0.31 | 0.028181… | 2.4e-16 |

So the two assemblies are *sometimes* bit-identical and sometimes not, depending on whether `1/h²`
lands exactly. **Neither behaviour can be relied on**, which is exactly why the isotropic default
stays on the **untouched `L @ L` code path** rather than being re-routed — otherwise every shipped
plate number would move in its last digit on some grids and not others, which is worse than moving
everywhere.

The intended pin was a stored byte-exact baseline of today's plate. **It was replaced with a live
in-run assertion** — the default plate's `B` must be byte-equal to `L @ L` *and* byte-unequal to the
general assembly, made on a grid known to distinguish them. That is stronger than a stored file
(it cannot rot, and it matches the repo's existing bit-identity idiom), and it exists only because
the grid survey above found grids where the two paths differ. Asserting it on a `Lx = 1` grid would
have passed vacuously.

Also from check 1, for the headline: a spruce-ish material (11 GPa along grain, 0.8 GPa across,
`nu_xy = 0.37`, `G_xy = 0.7 GPa`) gives `D_x/D_y = 13.75` and `H/sqrt(D_x D_y) = 0.567`, and drops
the fundamental from 43.5 Hz to 19.7 Hz. **That 0.567 is the interesting number**: a plate that was
merely "isotropic, stretched" would sit at exactly 1.0. Real wood does not, so `H` is an
independent axis and not a consequence of the stiffness ratio. §5 builds the headline on it.

## 3. Scope — and one refusal, written down

**In scope:** `Plate`, `boundary="supported"`, model #5. New oracles in `analysis/modal.py`. A
material-chain helper. One new test module.

**Refused this batch: the nonlinear (von Karman) plate, model #6.** Not for cost. `VKPlate`'s
linear bending operator would take the change in the same three lines, but its Airy stress solver
inverts an *in-plane* biharmonic that assumes isotropic in-plane compliance. Orthotropic von Karman
replaces that with a four-constant compliance tensor and a different stress function equation —
genuinely new machinery with no closed-form modal oracle to check it against, i.e. it fails
criterion (b) that selected this batch. It is its own batch, and it should be taken *after* the
free-edge orthotropic plate, which shares the strain-energy assembly it would need.

**Also out of scope:** the free branch (`free_plate_stiffness` assembles from strain energy and
would need `D_1` and `D_xy` separately, not just their combination `H` — a real extension, and the
one that unlocks a guitar-shaped top). Deliberately deferred so this batch keeps its exact oracle.

An earlier concern that orthotropy would land in an operator *shared* by all three plates, and so
put the shipped `nonlinear=False` bit-identity at risk, was checked in the code and **does not
hold**: `Plate` and `VKPlate` each build `B = L @ L` inline in their own constructor, and the free
branch goes through a different function entirely. The change is local to one branch of one class.

## 4. Build order and the test plan

### Step 1 — oracles first, core untouched

`analysis/modal.py`:

- `orthotropic_plate_freqs(kappa, Lx, Ly, modes, g_x, g_h, g_y)` — the continuum law of §1.1.
- `discrete_orthotropic_plate_eigenfrequency(lam_x, lam_y, kappa, k, theta, g_x, g_h, g_y)` —
  the theta-scheme frequency from `Q_mn`. A *separate* function, not an optional argument on
  `discrete_plate_eigenfrequency`, because the isotropic one takes a single Laplacian eigenvalue
  and this one needs the two axes separately.

Both must reduce to the shipped isotropic functions at `g = (1,1,1)` — asserted, not assumed.

### Step 2 — the core change

`Plate.__init__` gains three dimensionless keyword floats, `grain_x`, `grain_cross`, `grain_y`,
all defaulting to `1.0`, scaling the reference rigidity `D_ref = kappa^2 rho`. Explicit floats
rather than a tuple: three positional numbers of the same magnitude in a fixed order is exactly the
kind of surface that absorbs a silent swap.

    if isotropic (all three exactly 1.0):  B = (L @ L)      <-- today's line, untouched
    else:                                  B = g_x Dxx@Dxx + 2 g_h Dxx@Dyy + g_y Dyy@Dyy

Guards: `grain_x > 0`, `grain_y > 0`, `grain_cross > -sqrt(grain_x * grain_y)`.

`kappa` keeps its exact present meaning (it sets `mu` and the reported explicit-scheme margin).
Docstring must say that with grain, `mu` is computed from the *reference* rigidity and the
stiffest axis is `mu * sqrt(max(g_x, g_y))` — the implicit scheme has no limit either way, but the
reported number would otherwise quietly under-state.

Helper, module-level in `plate.py` (pure arithmetic, no I/O, headless rule intact):
`grain_ratios_from_material(E_x, E_y, nu_xy, G_xy, thickness, rho)` -> `(kappa, g_x, g_h, g_y)`,
implementing §1's chain including the reciprocity `nu_yx = nu_xy E_y / E_x`.

### Step 3 — the anisotropic damping caveat goes in the docstring

The shipped docstring already warns that the theta average turns frequency-independent loss into
frequency-*dependent* loss, mode `m` decaying at `2 sigma (1 - theta Q k^2)`. With grain this
becomes **anisotropic**: `Q` is no longer a function of the Laplacian eigenvalue alone, so two
modes that were degenerate under isotropy now decay at *different* rates depending on how their
curvature splits along and across the grain. Passivity is untouched (the operator is SPD by the
§1.2 guard), which means **the energy ledger stays green and cannot see this** — the modal
frequency oracle carries the correctness claim. Say so explicitly; do not let a green energy report
read as a validation of the damping.

### Step 4 — tests (`tests/test_plate_orthotropic.py`, new file)

Tier 1, the contract:

- **T1 operator money test.** Assembled `B_ortho` eigenvalue on the analytic sine equals
  `g_x lam_x^2 + 2 g_h lam_x lam_y + g_y lam_y^2` to machine precision, over a spread of `(m,n)`
  and a strongly anisotropic `g`.
- **T2 modal oracle.** Frequencies measured from a run match
  `discrete_orthotropic_plate_eigenfrequency` at machine precision, and
  `orthotropic_plate_freqs` at sub-cent on a fine grid. O(h^2) convergence to the continuum.
- **T3a byte-exact default.** A default-constructed plate reproduces a **pre-recorded baseline**
  of today's plate byte-for-byte (the batch-5 seam discipline). This is the assertion that the
  default provably takes the untouched path — "the numbers agree closely" is not it.
- **T3b new-path collapse.** `g = (r, r, r)` for `r != 1` equals an isotropic plate with
  `kappa' = kappa sqrt(r)` to ~1e-15. This is the meaningful statement about the *new* assembly,
  and the reason it is stated at 1e-15 and not 0 is Q1.
- **T4 energy.** Lossless drift < 1e-10 (tier 1 bar, unchanged); lossy monotone decreasing.
- **T5 guard.** Construction rejects `grain_cross <= -sqrt(g_x g_y)`, accepts just above, and
  rejects non-positive `grain_x`/`grain_y`.

Tier 2, the findings — §5.

### Step 5 — deliberate falsification of the coefficient wiring

Four constants with clashing published conventions is precisely the class of slip no energy ledger
sees, because a wrong-but-consistent coefficient is carried self-consistently through every step —
the `rho_v`-vs-`rho_s` hazard from the air-box batch, which was a 1000x error that stayed green.
So falsify it on purpose and record, per detector, what each mutation does:

1. swap `g_x` and `g_y` (the grain runs the wrong way),
2. drop the factor of 2 on the cross term (the `D_1 + D_xy` packaging),
3. use `D_1` alone as `H` (the other rival packaging),
4. pure reassociation of the same three products (the control — must stay green).

Expected: the energy ledger is green for all four; the modal oracle kills 1-3 and passes 4. If
anything else happens, that is the batch's real finding and it goes in the docstring.

## 5. The headline, stated so it can die

**Naive claim, which this batch expects to *not* survive:** "grain stretches the mode spacing" —
i.e. an orthotropic plate is an isotropic plate with one axis scaled, and `D_x/D_y` tells you
everything.

**The claim actually being tested:** `H` is an independent axis. Hold `D_x/D_y` fixed at the
spruce value 13.75 and sweep `H` across `sqrt(D_x D_y)` (the stretched-isotropic value; real spruce
sits at 0.567 of it, per §2). If the *ordering* of the modes reshuffles across that sweep, then no
amount of stiffness-ratio intuition reaches the answer and the cross term has to be modelled. If
the ordering does not move, that is also worth publishing and the claim dies — the repo's habit.

**Second measurement, the one most likely to produce a negative result worth more than the
positive:** does a **bridge point** notice the grain at all? `StringPlateBridge` couples at a
single node, and a point coupling sees a weighted sum over modes, so the anisotropy may partly
average out at the terminus even while the mode field is dramatically different. Measure it against
an isotropic plate matched on the fundamental; do not assume either way. This has the same shape as
the earlier finding that a displacement ratio is not an amplitude when the drive is a point force,
and if it holds it directly qualifies what orthotropy buys the *chain* as opposed to the *field*.

**A control that has to be in the sweep:** the mode-ordering claim must be checked at more than one
grid resolution, because a reshuffle between two nearly-degenerate modes is exactly what a
discretisation error can manufacture. Two grids at least; if they disagree, the reshuffle is
numerical and the claim dies.

## 6. Cost and the affordability check

Cheap. The operator assembly is three sparse products instead of one squaring, at the same
sparsity (13-point). No new solver, no new stencil, no iteration, no coupling. The `splu` factor
and back-substitute per step are unchanged in cost. The mode sweep of §5 is the only multi-run
item; it is a handful of short runs at a coarse grid plus one confirmation at a finer one.

No test in this batch should need the slow lane.

## 7. Build results — what measurement returned

### 7.1 The headline died. The replacement is sharper.

**Planned claim (§5):** the cross term `H` is an independent axis, so sweeping it at fixed
`D_x/D_y` reorders the modes and no stiffness-ratio intuition reaches the answer.

**Measured:** across the whole physically spanned range — from 0.2× the stretched-isotropic value
`sqrt(D_x D_y)`, past solid spruce at **0.567×**, to isotropic material at **exactly 1.0×** — all
sixteen low modes stay in **the same order**, at both grid resolutions. On the question of mode
ordering the naive intuition is simply right and the extra freedom buys nothing.

**What it does instead — a selective detuner.** Over that same sweep the modes shift by between
**1.3% and 29%**, a 22× spread in leverage, with a clean mechanism. The cross term enters as
`2 g_h λ_x λ_y` against direct terms `g_x λ_x²` and `g_y λ_y²`:

| mode | shift over the sweep | why |
|---|---|---|
| (3,1) | **+2.3%** | bends along the grain; the huge `g_x λ_x²` swamps the cross term |
| (2,2) | +16.9% | balanced |
| (2,3) | +26.6% | |
| (2,4) | **+29.0%** | bends across the grain, where `g_y` is 13.75× smaller and the cross term is comparable to it |

**The cross term matters exactly where the direct stiffness is weakest** — the opposite of where a
"stretched isotropic" picture would put it, and it is why the ordering survives: every mode moves
the same *direction*, and the gaps are wide enough to absorb a 22× difference in how far.

**The ordering result is a statement about the range, not about the term being inert.** Push the
cross term to 2× the stretched-isotropic value — past any solid wood — and (2,4) and (3,1) do swap.
The test asserts that too, so "no reordering" cannot be read as "the term does nothing".

### 7.2 The second finding died as posed, and its replacement is the practical one

**Planned:** a bridge point may barely notice the grain, because a point coupling averages over
modes.

**As posed this is wrong** — the frequencies move by more than an octave, so a bridge cannot fail to
notice. But with pitch removed (compared against an isotropic plate matched on the fundamental), the
grain sorts cleanly into one channel and out of the other:

- **the partial series carries all of it**: mode ratios above the fundamental move **37%**, a
  property of the operator that no pluck or pickup position can affect;
- **the level carries none of it**: the RMS at a single node lands at 1.116, 0.809, 0.873, 0.916,
  0.834 of the matched isotropic plate across five unrelated pluck/pickup geometries. It
  **straddles 1** — the spread is the geometry talking, not the grain.

The first draft of that test quoted **one** of those five numbers ("the single-node level moves
11.6%") as if it measured the wood. It measured where the pluck landed. The test now runs all five
and asserts the straddle, so the finding cannot be restated as a magnitude.

Practical consequence: couple this plate into a chain as an instrument body and judge it by how
loud the terminus rings, and **you cannot tell wood from metal**. The grain is audible as tuning,
not as output.

### 7.3 A test that proved nothing, and how it was caught

The anisotropic-damping test (§4 Step 3) first computed both decay rates from the same closed-form
expression it was claiming and compared them to each other. That is a tautology: it can only fail
if arithmetic fails. Rewritten to actually run the two modes and fit their energy decay, with the
isotropic plate as a control in the same rig:

| | (4,1) | (1,4) | apart |
|---|---|---|---|
| isotropic control | 5.857 /s | 5.857 /s | **identical to six figures**, as degeneracy requires |
| grained | 6.024 /s | 7.751 /s | **29%** |

(nominal rate `2σ = 8`/s; both arms under-damped, which is the plate's shipped caveat, not a new
defect — the finding is the *split*.) The control is what makes it a measurement: without it, two
different numbers prove nothing about the grain.

### 7.4 The detector result, and the falsification that produced it

Per §5 Step 5, the coefficient wiring was falsified on purpose: grain swapped end for end, the
factor of 2 dropped, the cross term taken as `D_1` alone, plus a pure-reassociation control.

**Every one of the three wrong plates conserves energy to machine precision.** They are all
symmetric definite operators, so the ledger has nothing to complain about — the wrong coefficient is
carried self-consistently through every step, exactly the shape of the volume-vs-areal density slip
that stayed green for a whole batch in the air-box family. Only the modal oracle separates them,
and it separates them by tens of percent. This is the fourth or fifth time in this project that the
energy report has been the wrong detector for the thing being built; it is now near enough a rule
that **a new coefficient needs an oracle, not a ledger**.

**And then the oracle turned out to have a blind spot of its own — a sharp, avoidable one.** The
diagnostic script printed the swapped-grain plate's fundamental as *identical* to the correct
plate's. It is not a coincidence and not a bug: on a **square** plate, swapping `g_x` and `g_y`
while `λ_x = λ_y` leaves `g_x λ_x² + g_y λ_y²` untouched, so **every diagonal mode `(m,m)` is
invariant under the grain running 90 degrees wrong** (0 or 2.2e-16 depending on the mode — exact in
real arithmetic, last-bit in floating point, which is why the test asserts a tight bound rather than
`array_equal`; it was written as `array_equal` first and (2,2) alone failed it).

The consequence is the practical one: **a validation that checks a square plate's fundamental, or
its first few diagonal modes, passes a plate whose grain runs the wrong way.** Two ways out, both
asserted: use an off-diagonal mode — the (2,1)/(1,2) pair swaps outright, 126% apart — or use a
non-square plate, where the fundamental alone moves 62%. The suite's falsification test uses
off-diagonal modes; this records that that was not an arbitrary choice.

So the tally for this batch is **three detectors and three different blind spots**: the energy
ledger cannot see any wrong coefficient; the modal oracle cannot see a swapped grain *if you ask it
only about diagonal modes on a square plate*; and a square plate cannot distinguish a material
asymmetry from a shape one at all (§7.5).

### 7.5 Two rig errors worth recording, because both looked like physics

- **A twin built at matched `mu` instead of matched `fs` — made twice.** `make_plate` solves the
  sample rate from `kappa`, so the isotropic twin of a `g = (r,r,r)` plate silently got a different
  timestep and the frequencies disagreed by ~3%. That reads as a modelling bug and is entirely a rig
  error. Caught, fixed, documented — **and then made again** in the level-versus-partials test,
  where the twin ran at 12.0 kHz against the grained plate's 20.5 kHz. There it happened not to move
  the numbers (the ratios are identical to three figures either way), which is luck rather than
  design and is exactly why the fix is now an `assert a.fs == b.fs` in both tests rather than a
  note. A helper that derives one thing from another will keep producing this until the comparison
  asserts the shared quantity.
- **A shape asymmetry impersonating a material one.** Comparing "curvature along the grain" against
  "curvature across it" on a **0.62 × 0.43** rectangle gives 1.6×, because the across-grain mode
  puts its half-waves along the *shorter* side and the extra wavenumber cancels most of the softer
  stiffness. On a square the same comparison is 2.95×. The comparison is only about the material if
  the geometry is symmetric.

### 7.6 A trap found in review and closed, rather than documented

`grain_ratios_from_material` takes the material's **volume** density and `Plate` takes an **areal**
one. The first version returned `(kappa, g_x, g_h, g_y)` and warned about the difference in its
docstring. That is the wrong shape of fix: the plausible caller mistake — passing the volume density
straight through to `Plate` — leaves **every frequency correct**, because `kappa` carries them, and
**every energy wrong by a factor of the thickness** (333x for a 3 mm sheet). Frequencies right,
energies wrong is exactly the class of error §7.4 shows the ledger cannot see *and* §7.1's oracle
cannot see either, because the modal oracle only reads `kappa`.

So the helper now returns a named `GrainSpec` carrying `rho_s` explicitly. The trap is removed
rather than warned about, and the change is loud for any existing caller (a 4-way unpack raises
rather than silently mis-binding). Asserted in the suite, including that `rho_s != rho` — if those
ever coincided the trap would be invisible and so would its test.

---

## 8. What was rejected to get here, and why

- **Energy-port reformulation** — high leverage but no new observable and it touches every
  coupling. Not a measurement batch.
- **Mass-spring networks**, **tuned marimba bar** — both have the reduction anchor, but the check
  is a convergence rate or published measured ratios, not a closed form. Criterion (b) fails.
- **Snare / rattle** — reuses the existing contact model, thin.
- **Curved shell (bell, bowl)** — the most exciting on the list and the weakest oracle: shallow-
  shell frequency formulas are themselves approximations, so there is nothing exact to check
  against. It is the right batch *after* the free-edge orthotropic plate, which builds the
  strain-energy machinery it needs.
- **Thermoviscous wall loss in the bore** — satisfies both criteria cleanly and is the genuine
  runner-up. Chosen against only because the body-coupling family is the most developed part of the
  codebase and grain drops straight into machinery that already runs.
