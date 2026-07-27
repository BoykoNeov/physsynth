---
name: barrier-collision-state
description: model
metadata: 
  node_type: memory
  type: project
  originSessionId: da778206-15e5-4927-b163-24e4e57cc97f
---

Model #8 = **string vibrating against a one-sided distributed barrier** (`core/collision.py`), the
FIRST *distributed* contact model (fret buzz / tanpura jawari / snare / prepared-piano rattle).
Built & green (**693 tests**, 2026-07-10). The second consumer of the mallet's contact scheme
([[mallet-collision-state]]) — so it **promoted the `[DG]` primitives** (`contact_potential`,
`contact_force_elastic`, `contact_stiffness`, `contact_force_dg`, hysteresis pair,
`contact_force_total`, scalar `solve_contact`) into `core/collision.py`; **`mallet.py` imports them
back**, bit-identical (its 26 tests untouched). Plan: `docs/dev/collision-barrier-plan.md`.

**The genuinely new machinery = VECTOR contact solve.** The mallet's host (membrane) is *explicit* →
scalar local solve. The barrier's host is the *implicit* θ-scheme `DampedStiffString`, so a force at
node j propagates through `A⁻¹` to every node → the unknown is the whole penetration VECTOR over the
contact nodes 𝒞: `η_𝒞 = η_free − G·F(η_𝒞)`, `G = (k²/ρ)(A⁻¹)_𝒞` (dense SPD admittance block).
`new solve_contact_vector` = **damped Newton + Armijo line-search on ½‖r‖²**, NOT brentq (scalar-only,
doesn't transfer). Well-posedness PROVABLE: `J = I + G·diag(F')`, G SPD + diag(F')⪰0 → every
eigenvalue ≥1 everywhere → **unique root, global convergence, NO branch-picking** (stronger than the
bow). Active set **self-selects**: `F=F'=0` for η<0 → inactive nodes' J-columns vanish. **Zero edits
to `string_damped.py`** (bow lesson: `apply_Ainv` + external `u +=` suffices).

**Force is a DENSITY (advisor, load-bearing): `force_pref = k²/ρ`, NOT `k²/(ρh)`.** `φ'(η)` is
already N/m because barrier energy `V = h·Σⱼφ(ηⱼ)` makes φ an energy density. Update `+(k²/ρ)A⁻¹F`;
energy `h·Σφ(ηⱼ)` (two-time-avg ½(φ^n+φ^{n-1})). Sign: `η = b − u` (>0 in contact), force opposes
penetration. `[DG]` uses η^{n+1},η^{n-1} only (u^n absent → telescopes). Vector `[DG]` =
component-wise scalar (separable potential).

**Oracles (advisor reconcile — the frequency eigenvalue is WARPED, static equilibrium is EXACT):**
energy conservation proves only INTERNAL consistency (force+PE telescope, but both could carry a
compensating scale factor). The **magnitude** gate is the **STATIC-EQUILIBRIUM oracle**: α=1
full-interior linear spring bed, at rest η⁺=η⁻ → `[DG]` hits Taylor branch (no time-avg warp) → the
discrete fixed point EXACTLY equals the continuous augmented equilibrium `S u*=(K/ρ)b`,
`S=−L+(K/ρ)diag(mask)`, held to **3.4e-15** (seat BOTH history levels at u*; `set_state`'s u⁻¹ is the
free-string one → off-equilibrium). **Negative control**: `force_pref*=2` blows drift up >1e4× (gate
has teeth). The α=1 DG force is `K(η⁺+η⁻)/2` — time-weights (½,0,½) vs elastic θ-term (θ,1−2θ,θ) →
partials warped from continuous eig at finite k → frequency-eigenvalue oracle DIAGNOSTIC-ONLY.
**Scalar collapse**: one contacting node → vector m=1 == imported scalar `solve_contact` at
`g=G_jj=(k²/ρ)(A⁻¹)_jj` to **7e-16** (different solvers Newton+Armijo vs Newton+brentq → checks
same-root).

Gates: lossless drift **~2e-12** through real contact (peak ~700 N, ~1000 contact-steps); drift ∝
newton_tol; passivity (σ / λ_h Hunt–Crossley) monotone; out-of-reach barrier bit-identical to bare
string; buzz signatures (barrier brightens tone, closer/harder → brighter/shorter, intermittent
contact). Files: `tests/test_collision_{energy,modal,signature}.py` (12+3+4=19),
`scripts/diagnose_collision.py`, `make_barrier_string` helper.

**PLATFORM GOTCHA:** dense per-step solve uses `scipy.linalg.lu_solve`, NOT `np.linalg.solve` —
NumPy 2.4 threaded BLAS on this Windows box has a catastrophic cliff (0.05 ms at n≤90 → ~250 ms at
n≥100) that hung the N=120 diagnostic; scipy stays ~1 ms. Tests sit at N≤80 (m≤79, under the cliff) so
they ran fast either way. Correction vectorized to one matmul (`cols_mat @ F`). Full-support solve is
`|𝒞|×|𝒞|` dense per step (as designed, fine offline); active-set reduction to in-contact nodes is the
natural future optimization. Next contact/collision targets: 2D snare-on-membrane, moving/deformable
barrier (finger stopping), sub-grid contact interpolation.
