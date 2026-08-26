# Guitar-shaped plate plan — model #5g: the outline stops being a rectangle

**Status: PROBED, not yet built** (2026-08-26). Every number below was measured by the probes in
`M:/claud_projects/temp/guitar-plate/` before a line of core code was written, which is the point:
the batch was *probed before it was planned*, as batch 19 was.

HANDOFF §12B names this refusal explicitly — "a **non-rectangular** (guitar-shaped) outline, which
is the membrane batch's staircasing problem in a 4th-order operator". This plan discharges it.

---

## 1. What the batch is, and what it is not

Every plate the core has ever built is a rectangle. `Plate` (model #5, supported and free), the
orthotropic pair (#5o, #5of) and the von Kármán plate (#6) all derive `Ny` from `Lx/N` and assemble
on the full `(Nx+1)×(Ny+1)` bounding box. The membrane (#4) is the only 2-D resonator that already
takes an arbitrary **mask**, and it is the one that learned what a staircased boundary costs.

This batch gives the *plate* a mask. The deliverable is a soundboard outline — the shape a guitar
top actually has, with two bouts and a waist — carrying both plate boundaries.

**It is not a new PDE.** The bending operator, the energy ledger, the θ-scheme and the four grain
constants are all unchanged. What changes is the set of nodes they are assembled on.

---

## 2. The supported branch is a NEGATIVE result, and it ships as one

`biharmonic_from_mask` already accepts an arbitrary mask, because `B = L @ L` is built from the
membrane's masked Laplacian. So a simply-supported guitar plate costs *zero* new operator code.

It also buys *nothing*. `eig(L²) = eig(L)²` is an identity, not a coincidence: the Navier plate's
spectrum on any outline is the **square of the membrane's spectrum on the same outline**. Measured
at 2.8e-12 on the guitar mask — and that number measures `eigsh` convergence, not correctness,
because the identity **cannot fail**.

⇒ The guitar outline tells the simply-supported plate nothing the drumhead did not already know.
State this with the identity; do not dress it up as a passing test. (Same discipline as §12H batch
3's lesson: a detector that cannot fail is not a detector.)

The **free** branch is where the batch's content is, and it is where all the work below goes.

---

## 3. The free branch: the assembly generalises, and it does so bit-identically

`free_plate_stiffness` is a Kronecker assembly on a full rectangle — `kron(iy, c2x)` and friends —
so it cannot see a mask at all. The generalisation, probed and working:

| piece | rectangle rule | masked rule |
|---|---|---|
| curvature `C2x`, `C2y` | `[1,-2,1]/h²` at interior rows, **zero at the two end rows** | `[1,-2,1]/h²` **iff both neighbours along that axis are live**, else a zero row |
| twist `Dxy` | one row per cell of the `Nx·Ny` grid | one row per **live cell** (all four corners live) |
| area weight `Wa` | `kron(m_y, m_x)`: `h²` interior, `h²/2` edge, `h²/4` corner | `h² · (live adjacent cells)/4` |

The area rule is the load-bearing one and it is *not* a new convention — it **is** the trapezoidal
weight, rewritten. An interior node touches 4 cells (`h²`), an edge node 2 (`h²/2`), a corner 1
(`h²/4`). The three cases fall out of one expression instead of being three cases.

**Measured consequence: on a rectangle mask the masked assembly reproduces `free_plate_stiffness`
bit-identically — a printed `max|dK| = 0.0` and `max|dW| = 0.0`.** Not "to 2e-16": exactly zero.
The four coefficients multiply the same four matrix products in the same order, and the area
weights are products of powers of two with the same `h·h`.

⇒ **Structural call: `free_plate_stiffness` becomes a thin wrapper** that builds a rectangle mask
and calls the general routine. One code path, and every shipped free-plate number — #5b's Leissa
match at 0.01%, #5of's four probes, the cymbal in the room — is preserved *by construction* rather
than by re-measurement.

**Gate before collapsing them (advisor, and it is not optional):** the bit-identity is confirmed at
**one** grid and **one** ν so far. Re-run it across the orthotropic batch's **7-grid × 4-ν survey**
*and* at least one **non-default grain split** before deleting the Kronecker path. If any cell of
that survey is non-zero, keep two paths — the #5o precedent is that a reassociation that agrees only
to ~2e-16 still perturbs every shipped number in the last digit, and that is a real cost.

---

## 4. The trap the staircase actually sprang: nodes that carry no area

A curved outline produces live nodes that touch **no complete cell** — the one-node spikes at the
tips of the shape. Their area weight is exactly `0`, so the mass matrix `W` is **singular** and
`A = (1+σk)W + θk²κ²K` cannot be factored. Measured on the guitar outline: **4, 3, 2, 2 such nodes**
at N = 16, 24, 32, 40. Two nodes are enough to kill the solve.

**The fix is a prune to a fixed point:** a node is an unknown only if it touches at least one live
cell; dropping nodes can orphan more, so iterate. Converged in **one sweep** at every grid probed,
dropping 2–4 nodes.

⇒ **The mask is not the outline.** The outline is a predicate on coordinates; the mask is the set of
nodes that carry area. `Plate` must store the pruned mask and report the drop count, because §5's
pricing depends on it.

---

## 5. Validation — and the honest fact that only ONE tier can falsify anything

The three detectors this family reaches for are all blind here:

- **Energy conservation is geometry-blind.** The membrane batch recorded it (`energy ⊥ geometry`);
  #5of recorded it again (three visibly different plates conserving at 1.5–2.1e-13 with fundamentals
  of 5.68, 3.79, 5.99). A wrong outline conserves perfectly.
- **The supported identity cannot fail** (§2).
- **Nullspace = {1,x,y} is necessary, not sufficient.** Measured on the guitar mask at every grid:
  exactly **3** zero modes, rigid-body residual **6.7e-13**, saddle `xy` not null. Good — and it
  would look exactly the same for an assembly with the wrong boundary treatment.

So the batch needs an **absolute oracle on a curved boundary**, and there is exactly one available.

### 5.1 The disk oracle — derived here, not cited

The free **circular** plate exercises the identical staircase machinery and has a real answer. No
freely-available orthotropic-style table was needed: the frequency equation is derived in the batch
(the same policy #5of set when its tables turned out to be paywalled).

With `w = W(r)cos(nθ)`, `W(ρ) = A J_n(λρ) + B I_n(λρ)`, `λ = ka`, `ω = κλ²/a²`, the two free-edge
conditions at `ρ = 1` are (primes are `d/dρ`; the `a`-powers cancel after multiplying the second by
`a³`):

    (i)  M_r = 0:   W'' + ν(W' − n²W) = 0
    (ii) V_r = 0:   W''' + W'' − [1 + n²(2−ν)]W' + n²(3−ν)W = 0

**The derivation is checked three ways before it is trusted**, because a plausible-looking sign
error passes most checks:

1. **Rigid-body roots.** `W = 1` (n=0) and `W = ρ` (n=1) annihilate both lines exactly.
2. **The plate's own energy Rayleigh quotient.** For each candidate root, build `W` from the
   nullvector and evaluate `P/M` in polar coordinates. A genuine mode returns `λ⁴` exactly. **All**
   roots pass — the equation has no spurious roots.
3. **A closed-form bound with no Bessel functions in it.** The pure saddle `w = xy` is orthogonal to
   `{1, x, y}`, has `w_xx = w_yy = 0, w_xy = 1`, and gives `P = 2(1−ν)πa²`, `∫w² = πa⁶/24`, so

       Λ₁ ≤ sqrt(48(1−ν)) = 5.79655   (ν = 0.3)

   against a derived fundamental of **5.35833** — an **8.18%** overshoot, the same character as the
   free *rectangle*'s analogous twist bound (5.3–5.6%). This is the disk's version of #5of's twist
   probe, and it is the one line of the oracle that a reader can check by hand.

Derived spectrum, ν = 0.3, `Λ = ω a² sqrt(ρ_s/D)`:

| Λ | n | multiplicity |
|---|---|---|
| 5.35833 | 2 | pair |
| 9.00314 | 0 | single |
| 12.43899 | 3 | pair |
| 20.47455 | 1 | pair |
| 21.83516 | 4 | pair |

### 5.2 What the masked assembly measures against it

All **seven** lowest elastic modes, uniformly, converging **from above**:

| N | live | area deficit | mean abs error | area-corrected |
|---|---|---|---|---|
| 24 | 437 | −13.35% | 13.03% | 2.06% |
| 32 | 793 | −8.98% | 8.55% | 1.20% |
| 48 | 1789 | −6.28% | 6.04% | 0.61% |
| 64 | 3205 | −4.26% | 4.02% | 0.41% |
| 96 | 7209 | −3.02% | 2.89% | 0.22% |
| 128 | 12849 | −2.11% | 2.01% | 0.15% |
| 160 | 20069 | −1.76% | 1.68% | 0.11% |

**Two findings, and the second is the batch's headline.**

First, the rate is **O(h), not O(h²)** — set the convergence test up expecting second order and the
batch will be spent hunting a bug that is the boundary. The membrane already found the same tax on
its Bessel match; this is the 4th-order operator paying it too.

Second, and the part that generalises: **the staircase error is a DOMAIN-SIZE error, not an
operator error.** Read the third and fourth columns together — −13.35 / 13.03, −8.98 / 8.55,
−2.11 / 2.01. The frequency error tracks the *area deficit* to within a few percent of itself, and
it does so **mode-independently** (the seven per-mode errors at N=128 span +1.92 to +2.08). The
staircased disk is not a badly-modelled disk; it is a well-modelled *slightly smaller* disk.

Dividing the deficit out — `Λ_corr = Λ · area/(πa²)`, i.e. measuring the plate in its own effective
radius — drops the error by 6× to 15× and lifts the rate to roughly **O(h^1.5)**. Ship this as a
reported diagnostic, **not** as a silent correction inside the operator: it is a statement about the
mask, and hiding it would make a coarse plate look converged when it is not.

### 5.3 The zero-valued detector

The disk's `n ≥ 1` modes are **degenerate pairs**, and a square grid relates the axis-aligned and 45°
members by no symmetry at all — so it splits them. **The exact answer for that split is zero.**
Measured: 0.69%, 1.01%, 0.06%, 0.52%, 0.17%, 0.013% at N = 24…128 — small, shrinking, and **not
monotone**. Assert a ceiling that shrinks with `h`, never monotonicity.

### 5.4 The two cheap asserts that are currently missing

- **`mu.min() > -tol`.** The probes printed the 4th eigenvalue and never the 1st, so a small
  *negative* eigenvalue on a staircased domain is not yet ruled out.
- **One connected component.** A coarse grid plus an aggressive waist can pinch the outline into two
  lobes. That is silently *two plates* with a 6-D nullspace, and every other detector passes.

---

## 6. The outline itself

A half-width profile, so the region is simply connected and vertically convex by construction:

    W(t) = W0 · sin(πt) · [1 − waist·cos(4π(t−½))] · [1 + asym·(t−½)],   t = y/L ∈ [0,1]

`sin(πt)` closes both ends; the `cos(4π…)` term puts maxima at the two bouts and the minimum at the
waist; `asym` makes the lower bout wider. Defaults `waist = 0.42`, `asym = 0.30` give a
dreadnought-ish top. Exact lutherie geometry is not the point — a *parametric* outline is, because
the waist depth is the knob that makes the shape a guitar rather than an ellipse.

---

## 7. Refusals

- **No clamped branch.** A glued-in soundboard is nearer clamped than either shipped boundary, and
  `_clamped_d2_1d` exists — but it is 1-D and the Airy solver's, and generalising a clamped rim to a
  staircase is a second batch, not a corner of this one.
- **No von Kármán on the outline.** #6 assembles its bracket and Airy solve on the full grid; masking
  those is its own batch and would drag the Picard convergence question in with it.
- **No area correction inside the operator.** §5.2 — reported, never applied silently.

---

## 8. Touch list

- `physsynth/core/operators2d.py` — `guitar_mask`, `prune_mask`, and the masked
  `free_plate_stiffness` (the Kronecker path collapses into it **iff** §3's survey gate passes).
- `physsynth/core/plate.py` — `domain` parameter on `Plate` (`"rectangle" | "circle" | "guitar"`),
  the pruned mask, the drop count and the area deficit as reported attributes.
- `physsynth/analysis/modal.py` — `free_circular_plate_lambdas(nu, n_modes)`, the derived oracle of
  §5.1, with its three self-checks as asserts rather than comments.
- `tests/test_guitar_plate.py` — the tiers of §5.
- **Not** `web/` — a viewer batch follows this one (the plan's standing rule: a new core model
  reopens the built-but-unshown gap), and it is not in this batch's scope.
