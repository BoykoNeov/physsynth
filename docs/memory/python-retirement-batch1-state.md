---
name: python-retirement-batch1-state
description: "Phase C batch 1 (sympathetic strings, 2026-09-08) — found the plan's HOLE: 14 model classes live ONLY in the doomed binding crate, so phase F is gated on re-homing them; the guard needed a NEW eigensolver, and the symmetric route is a MEASURED claim about the operator"
metadata: 
  node_type: memory
  type: project
  originSessionId: 68b9a58f-3d77-4a17-af62-5841d60fe486
  modified: 2026-09-08T07:38:03.484Z
---

The first phase-C batch of [[python-retirement-state]]: `SympatheticStrings` moved into
`crates/physsynth-core/src/connection.rs`, 18 native bars replaced `tests/test_sympathetic.py`'s 15
functions, and the Python file went in the same commit. Recorded in
`docs/dev/python-retirement-plan.md` §11 (the hole) and §12 (the batch).

**The hole, which is worth more than the batch.** The plan counted the whole binding crate (19,024
lines) as deletable. It is not: `connection.rs` (4 bridge classes) and `airbox_wrap.rs` (10 room
wrappers) — 3,231 lines — are the **only implementation** of models the project ships. The
migration put them there on purpose, because they are polymorphic over their collaborators through
Python duck typing (one `body=` slot takes eight types), and a `#[pyclass]` downcast would have
narrowed them. That was the right call while Python existed; with Python gone the models go with
the interpreter. **Phase F is gated on re-homing all fourteen**, and that work is phase C by
another name — it lands with the bars.

**Why sympathetic strings was the cheap case:** every construction site in the repo passes
`IdealString`s and a `ModalBody`, so the duck-typed slot collapses to a concrete type and nothing
is lost. `StringBodyBridge` (eight body types) needs an enum or trait; the airbox wrappers are
worse, because the tier below them stores its matrices as Python objects *by design*.

**A port can need a whole new numerical routine, and the plan's test-count estimate cannot see
it.** The model's constructor refuses an over-stiff bridge by `k^2 lambda_max(A) < 4`, and Python
got `lambda_max` from LAPACK. The core crate's dependency list is empty and stays empty, so
`crates/physsynth-core/src/eig.rs` had to be written — Householder tridiagonalization plus
implicit-shift QL, 10 bars of its own.

**The symmetric routine is licensed by a MEASUREMENT, not by convenience.** `A = M^-1 K` is not
symmetric; `M^1/2 A M^-1/2` is, to 3.6e-18 relative (roundoff in the scaling multiply), with `M`
the trapezoidal mass diagonal **halved at the free end that carries the spring**. That is asserted
as its own bar — if it stopped holding, the guard would answer a *different question*, not the same
one imprecisely. Power iteration was rejected on measurement too: `lambda_2/lambda_1 = 0.9969`, and
two identical strings make the top eigenvalue exactly degenerate.

**A bit-identity bar whose referent is being deleted must be RESTATED, not ported.** "One string
equals `StringBodyBridge`" compared two implementations of one spring; with one implementation left
it is the code agreeing with itself. What it protected was the *shape* of the coupling, so it
became: one coupled step = one free step of each part **plus exactly one spring force at one node**
(weight `2k^2/(rho h)`, handed to the body unscaled), bit for bit. The `K=0` bar did **not** change
— both of its referents (a lone string, a lone body) exist natively. Same distinction as ledger
#64/#65 in `docs/dev/rust-migration-findings.md`.

**Absolute thresholds on a chosen amplitude are claims about the amplitude.** Three of the
antisymmetric bar's tolerances (`1e-13`, `1e-15` on a `1e-3` pluck) are now relative to the pluck
and to `E^0`.

**Verify the port against the implementation it replaces BEFORE deleting the Python — once, as a
scratch check, not a committed test.** Same fixture through both: every state array and both
energies **bit-identical**, `lambda_max` off by **16 ulps (2.3e-15)** — this crate's QL against
LAPACK's `dgeev`, the only number that could differ. That is why the new module keeps the binding's
expression order and `reduce::sum`.

Cost: the Python file was 59 s of suite time; the 18 native bars run in **2.1 s** debug.
The binding's `PySympatheticStrings` stays as it is — the viewer builds it, and
`tests/test_web_backend.py`'s 22 sympathetic bars cover that path until phase D.

**A bar built from your own derivation checks that derivation against ITSELF.** The symmetry bar
builds `M^1/2 A M^-1/2` from `mass_diagonal()` — the module's own mass derivation — so a wrong mass
that still symmetrizes (a uniform scale; the half cell on the wrong end of two *identical* strings)
passes it while scaling the whole spectrum. The fix was to freeze the cross-check's number:
`lambda_max = 1661856272.3158104` to 1e-12 relative, LAPACK-measured before the Python went. Value
plus tolerance, never a bit pattern (ledger #68) — 16 ulps of cross-implementation spread is the
portability floor. **Ask what a bar would still pass if the thing it is built from were wrong.**
