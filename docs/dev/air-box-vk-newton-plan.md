# The room seam's solver — giving the air-loaded gong the iteration the bare one already has

Plan document for `docs/dev/scientific-hurdles.md` §5's remaining scene, chosen by the human on
2026-09-06 over four alternatives. Written **after probing and before any code**, which is this
project's ritual, and — as with `docs/dev/vk-newton-plan.md` — the probing changed the batch.

The probe scripts are in `M:\claud_projects\temp\airbox-vk-newton\` (`probe_seam.py`,
`probe_wall.py`, `probe_wall2.py`, `probe_newton_proxy.py`, `probe_identity.py`,
`probe_outcome.py`). They drive the shipped models through the binding and add no core code, so
every number below is reproducible today.

---

## 1. What is actually blocked

`docs/dev/vk-newton-plan.md` §11.8 states it and §14.1 repeats it: the airbox von Kármán scenes
cannot run under Newton. The reason is one method. `_VKPlateSurface.solve`
(`crates/physsynth-py/src/airbox_wrap.rs:602`) runs its **own** Picard loop — a hand transcription
of `VKPlate.step`'s sweep, written in Python-object arithmetic, iterating against the room-**loaded**
factorization that the wrapper assembled and handed in as an argument. `couple_method` lives on
`VkParams` and that loop never consults it.

The seam exists for a good reason and it is not going away: once the air is attached, the bare
plate's `lu` is the wrong operator. `A_loaded = A_bare + (k / 2 rho_s) T^T R T` is assembled in
SciPy by `build` (`airbox_wrap.rs:747`), factored **once** by `splu`, and handed to `solve` as a
Python object. The model's own factorization is untouched, and
`test_the_loaded_factorization_is_never_assigned_to_the_model` exists to keep it that way.

So the blockage is structural rather than incidental: the core's Newton reaches its θ-scheme
operator as `self.p.lin.lu`, a Rust `SparseLu`, and the operator this scene needs is a SciPy object
the core cannot name.

## 2. What the probe measured, and the four things it settles

### 2.1 The seam's Picard loop is a FAITHFUL transcription — bit for bit, on both boundary arms

Drive `_VKPlateSurface` with the plate's own **unloaded** factorization (`splu(surface.a_bare())`)
and it reproduces `VKPlate.step` exactly. 40 steps, `w/e = 3`, 8 kHz, N=20, σ=0:

| boundary | `max abs dw` | `max abs dF` | `n_iters` mismatches |
|---|---|---|---|
| supported | **0.000e+00** | **0.000e+00** | 0 |
| free | **0.000e+00** | **0.000e+00** | 0 |

This is the fact the whole batch stands on. It means replacing the transcription with a call into
the core's own coupled step — parameterised by which factorization to back-substitute against — is
a **diff-free refactor**, not a re-derivation, and the existing suite is a sufficient guard for it.

Both boundary arms are measured because the seam branches on the boundary twice: `couple_factor`
picks up an extra `h^2` on a free edge, and `areas` comes from `wdiag` (non-uniform, `5.625e-05` at
the corner against a uniform `h*h = 2.250e-04`) rather than a uniform cell area. The **tier**
(baffled versus suspended) is *not* a seam branch — `solve` takes `lu` and `rhs_fixed` as arguments
and never asks which tier assembled them — so the four combinations this section's first draft
worried about are two, and both are measured.

### 2.2 The air load does not move the wall — it sits exactly where the bare plate's does

The natural hypothesis is that the room makes the iteration harder. It does not; if anything the
load is a PSD term added to an SPD matrix before factoring, so `A_loaded^-1` is *smaller* and Picard
should contract slightly better. Measured, 120 steps, cap 50, N=20, at the room fixture's own 8 kHz,
comparing the room-loaded gong against the bare gong built from the same material and struck
identically:

| w/e | room: max sweeps / mean / non-converged | bare: max / mean / non-converged |
|---|---|---|
| 1 | 11 / 7.0 / 0 | 11 / 7.0 / 0 |
| 2 | 21 / 9.6 / 0 | 21 / 9.7 / 0 |
| 3 | 44 / 13.6 / 0 | 44 / 13.6 / 0 |
| 3.5 | 50 / 16.1 / **1** | 50 / 16.1 / **1** |
| 4 | 50 / 19.4 / **2** | 50 / 19.4 / **2** |
| 4.5 | 50 / 50.0 / 1, **blew up on step 0** | 50 / 50.0 / 1, **blew up on step 0** |

The transitions line up cell for cell. **The room scene's wall is the bare plate's wall**, which is
what makes the fix pure plumbing and makes its payoff predictable from a measurement that already
exists.

Two side facts worth writing down. The wall sits at `w/e ~ 4` here rather than the 48 kHz map's
`~16` because `k^2` is 36x larger at 8 kHz — the `k^2` half of §5's scaling, confirmed again. And
the room fixture *cannot* simply be run at 48 kHz: the 3-D CFL shrinks the room's physical size at
fixed `N`, and `make_room_loaded_vk_plate` at 48 kHz raises
`the surface's footprint spans -0.052..0.218 m, outside face 'z0', which is 0..0.165 m there`.
The room sets the sample rate; that is not new, but it is why this scene's wall is at a *different
place on the strain axis* than every number in `vk-newton-plan.md`.

### 2.3 Newton clears this wall too — but it has its own, and the gain is SMALLER than at 48 kHz

Because §2.2 shows the walls coincide, the bare plate under Newton at 8 kHz predicts what the room
scene would get. Measured, 120 steps, N=20, σ=0, cap 50, with `n_solves` (back-substitutions) as the
machine-independent cost:

| w/e | Picard: non-conv / solves / drift | Newton: non-conv / solves / drift | verdict |
|---|---|---|---|
| 1 | 0 / 1,668 / 8.9e-14 | 0 / 2,466 / 2.1e-13 | Newton costs **1.48x** |
| 3 | 0 / 3,268 / 2.3e-13 | 0 / 3,996 / 3.6e-13 | 1.22x |
| 3.5 | **1** / 3,876 / 4.5e-09 | 0 / 4,434 / 1.7e-13 | Newton clears the cap |
| 4 | **2** / 4,668 / 3.0e-03 | 0 / 4,824 / 2.4e-13 | Newton clears the cap |
| 4.5 | **blew up** / 100 | 0 / 5,408 / 3.4e-13 | **Newton rescues** |
| 5 | **blew up** / 100 | 0 / 6,454 / 2.2e-13 | **Newton rescues** |
| 6 | **blew up** / 100 | 0 / 8,102 / 2.2e-13 | **Newton rescues** |
| 9 | **blew up** / 100 | 27 / 700,004 / 9.5e-02 | Newton survives, degraded |
| 12 | **blew up** / 100 | 118 / 2,444,286 / 1.8e+02 | **Newton's own wall** |

Two claims, and the second is new.

* **The batch is worth doing.** A scene that dies at `w/e = 4.5` runs clean to 6 with an energy
  drift of 2e-13, which is the tier-1 bar rather than a tolerance.
* **Newton's boundary at 8 kHz is between 6 and 9, so the amplitude gain here is about 1.6x** —
  materially less than the 2-4x `vk-newton-plan.md` §13 mapped at 48 kHz. That is a finding, not a
  disappointment, and Part 3 must report it as one: the map's factor does **not** transfer across
  sample rate, and this scene is the first place that could be seen, because it is the first fixture
  whose sample rate is set by something other than the plate.

The `w/e = 9` row is the interesting one for the map: Newton spends 700,004 back-substitutions and
still comes back with 27 non-converged steps and a 9.5% drift. Newton does not fail here by
diverging, it fails by becoming unaffordable — the same "not free" statement §13.3 makes, at the
point where it stops being a tax and becomes the bound.

### 2.4 A defect that is live TODAY, independent of Newton: the seam misreports its own failure

`couple_outcome()` distinguishes `capped` (the sweeps were still contracting; a bigger cap fixes
this step) from `expansive` (they were growing; no cap at any size helps). The test is deliberately
**positive** — the ratio must be finite *and* below one — so that a NaN cannot be filed as merely
short of sweeps.

The seam never writes `residual_ratio`. It is NaN from construction and stays NaN, so **every
non-converged step of a room-loaded gong is filed as `expansive`**. Measured, same strike, bare
against room, counting the outcomes of the non-converged steps only:

| w/e | room reports | bare reports |
|---|---|---|
| 3.5 | `{'expansive': 1}` | `{'capped': 1}` |
| 4 | `{'expansive': 2}` | `{'capped': 2}` |
| 4.5 | `{'expansive': 1, blew up}` | `{'expansive': 1, blew up}` |

The two rows that disagree are exactly the recoverable ones. The room tells you "no cap will help"
about a step a larger cap does fix. `n_solves` is the same story one step further: it stays `0` for
the life of a room-driven plate, and it is the *only* axis on which `VkStep`'s own documentation
permits Picard and Newton to be compared. Both must be plumbed regardless of which design wins;
neither is a consequence of Newton.

(`n_iters`, `converged` and `last_residual` **are** written by the seam, exactly as the model's own
step writes them. The gap is precisely the two fields added later — one by the Newton batch, one by
Part 0.)

## 3. The design, and the one way to get it subtly wrong

Three designs were considered.

* **A — teach the core to accept a supplied factorization.** A trait in `physsynth-core`, no new
  dependency; `VkCoupledStep` carries a `&dyn ThetaSolve` instead of reaching into `p.lin.lu`, and
  the binding implements it over the SciPy object the wrapper already holds. §2.1 says the Picard
  path is then diff-free by measurement, and Newton arrives for free because `vk_newton` is written
  against `VkCoupledStep` and nothing else.
* **B — transcribe Newton into the seam**, alongside the Picard transcription already there.
  Rejected: it duplicates the exact arithmetic `vk-newton-plan.md` Part 1's finite-difference
  apparatus exists to certify, and a second copy of a Jacobian is the failure this project has the
  most scar tissue about.
* **C — assemble and factor the loaded matrix in Rust.** Rejected *for this batch*: it swaps SciPy's
  SuperLU for the crate's own sparse LU under an `array_equal` anchor in `tests/test_airbox_vk.py`,
  and `docs/dev/rust-migration-findings.md` records that an ordering in front of a sparse LU is a
  divergence **no bar in this project can catch**. It is a real batch; it is not this one.

**A is the design.** The hazard is specific and worth stating before the code exists: `p.lin.lu` is
reached in **two** places — `VkCoupledStep::sweep_from` (the Picard sweep, and Newton's residual)
and `VkCoupledStep::jacobian_vector` (the correction inside GMRES). Route only the first through the
override and you get a Newton whose **residual is the loaded problem while its tangent is the bare
plate's**. That is a valid preconditioner, so it still converges, to the right answer, and it passes
every energy bar — it is simply slower, in a way nothing in the suite would notice. A native bar
that substitutes a deliberately *wrong* solver and asserts that **both** methods move is the guard.

## 4. The parts

**Part 1 — the core seam.** `ThetaSolve` in `plate.rs`, implemented for `SparseLu`;
`VkCoupledStep` carries one; a constructor that takes an already-assembled `rhs_lin` (the seam's
`rhs_fixed` already contains `f_ext` *and* the room's two load terms, so it cannot be rebuilt from
`step_rhs`). Native bars: substituting `&p.lin.lu` reproduces the shipped trajectory bit for bit,
and a wrong solver moves **both** call sites (§3's hazard).

**Part 2 — the binding.** `PyVKPlateSurface::solve` reaches the model's `VkParams` (it is
`PyVKPlate`, in the same crate, with `vk_params()` already `pub(crate)`), wraps the supplied `lu` in
a `ThetaSolve`, and calls the core — honouring `couple_method`, and writing `residual_ratio` and
`n_solves` along with the three fields it already writes. The transcribed loop is **deleted**; §2.1
is what makes that safe. Guards: `test_airbox_vk.py` unchanged and green, and in particular the
byte-exact `nonlinear=False` regression against `RoomLoadedPlate`, which the airbox family's
standing rule (no single detector is sufficient) says must stay primary.

**Part 3 — the measurement.** The room scene under Newton: where its wall moves, what it costs in
back-substitutions, and §2.3's finding that the amplitude gain at 8 kHz is smaller than the 48 kHz
map's. Plus the `capped`-versus-`expansive` misreport of §2.4, now fixed, asserted as a test rather
than a table.

**No default moves.** `couple_method` stays `"picard"` everywhere and every shipped number stays the
number it is.

## 5. Deliberately not done here — and one of these is half of what was asked

* **The mallet on a gong in a room.** `MalletVKPlate` calls `vk_step` directly with its own params,
  and its outer chord freezes the **linear** plate's drive-point admittance built from
  `A^-1 e_node`; against the room-loaded `A` that is a different column, and whether the chord's
  frozen tangent should be the loaded one is a real question with its own energy bars. Part 1 gives
  that composition the primitive it needs — a coupled step that can be pointed at a supplied
  factorization — but it does not perform it. **This is a follow-on batch, and it is named here
  because it is half of what the human asked for on 2026-09-06.**
* **No Rust assembly or factorization of the loaded matrix** (design C above).
* **No `f_ext` change.** `vk_step`'s `f` argument is still dead on the nonlinear path.
* **No viewer scene.** The `vkroom` scene keeps the path it has.
