---
name: rust-phase5-geometric-state
description: "Phase 5 batch 5 — the geometrically exact string; the first divergence that changes NO digit (an ORDERING in front of the sparse LU, 13x fill), the first non-SPD Group D matrix, \"sum or max?\" added to the branch rule, and 15.5x — plus why `connection` moves AFTER `airbox`"
metadata: 
  node_type: memory
  type: project
  originSessionId: eca39c3d-2eec-4137-8c20-a69869a4004a
  modified: 2026-08-31T09:11:24.587Z
---

Phase 5 batch 5 (2026-08-31) ported `physsynth/core/string_geometric.py` — model #10, the **last of
the four theta-scheme strings**, so the whole string family is Rust and the three `array_equal`
anchors that chain it (`sigma1 = 0`, `EA = 0`, `EA = T`) now compare Rust against Rust. Plan §29.
See [[rust-phase5-plate-state]] for the batch before it and [[geometric-string-state]] for the model.

**The scope was flipped before any code was written, and that is the first finding.** §28.11 named
`connection` as next and "the cheapest of them", because it touches no private names. Wrong
instrument — the same complaint §28.2 had just made about a name grep. `connection` is
**polymorphic over its collaborators' TYPES**: its three bridges take Rust bodies/plates *and*
`airbox`'s duck-typed Python wrappers, and `test_airbox_{surface,dipole,vk}.py` pin
`bridge.stability_margin == bare.stability_margin` **exactly** across that boundary. It also needs a
dense nonsymmetric eigensolver (`np.linalg.eigvals`) with the dep list empty, and buys ~zero speed.
**So `connection` ports AFTER `airbox`.** (Note for then: `A = M⁻¹S` is symmetrizable, so `dgeev`
becomes a symmetric eigensolver plus a §25.3 margin measurement on the raise/no-raise decision.)
Three different searches have now been needed to find a blocking dependency — private names,
re-derivation, polymorphism — and **none of the three finds the others**.

**The batch's finding: a divergence that changes no digit.** Every finding since §14 is about
*which digits* two implementations produce. This one is about **how much work** they do. It is the
first Group D model that factors **inside** the solve loop (a fresh Newton Jacobian per iteration
per step; `beam`/`plate`/Airy all factor once at construction), so §24's natural column order stops
being free — and §24 wrote its own escape clause for exactly this. The unknowns are stacked **by
field** (`[u; w; v]`) while the nonlinearity couples the three fields at the same **cell**, so every
coupling sits `N-1` columns off the diagonal. At N = 128: natural order **33,895** nonzeros in
`L+U` and 2,068 µs, against SuperLU/COLAMD's 2,788 and 156 µs. Reordered by node — `(u_i, w_i, v_i)`
together, a **closed form in N** — 2,645 and 58 µs. **No bar in this project could have caught it:
the answers stay right and the model just gets slow.** If a port introduces an algorithmic choice,
the assertion has to be about the *work*.

The reordering is free **only because** every update-path operator is block diagonal by field, so no
reduction crosses a block and the permutation can live inside `SparseLu::factor_permuted`.

Corollaries worth carrying:

- **The first Group D matrix that is NOT SPD**, retiring `DIAG_PIVOT_THRESH`'s written
  justification. The replacement is measured — and *the proxy matters*: row dominance is set by
  `lam_long` alone (8.06 at 0.5, **0.285** at 8.0, not dominant at all) while `is_natural` — the
  thing that actually decides — holds over **854 Jacobians** with no pivot anywhere.
- **§19.2's branch rule gains a word: is it a sum or a max?** A `max` is order-independent by
  construction. Model #9's `brentq` bracket was a sum and flipped on 1,400/5,000 steps; here only
  *which side of the bar* one step lands on varies, at a rate set by **how far the mean iteration
  count sits from an integer** (0 flips in 20,000 at mean 1.00; 475 at 1.50). A flip costs two
  orders of trajectory and **nothing** on the energy — any root of the DG equation conserves exactly.
- **`portable.py` was not needed at all** — a first for this family. Every matrix arrives canonical.
- **A fixture can be wrong in the PHYSICS, not the coverage.** The first native fixtures fixed
  `fs = 48 kHz`, landing at `lam_long = 4.6`/9.2 — past the model's own cliff. Three bars went red
  and all three were *correct*. Build a two-speed model's fixture at the **fast** speed.
- **A shape is part of the interface** (`_chol_u` flat vs `(3, n)`), and **a rejection's type is**
  (`displacement_at` must raise `IndexError`, not `ValueError`).
- §19.7's YAML line continuation happened a **fourth** time from a **fourth** tool and never reached
  CI — §20.7's test catches it locally in seconds. Four occurrences, zero red runs since it exists.

**Speed: the nonlinear step is 15.5x**, retiring §28's 2.99x record by a wide margin. A Newton step
is a dozen tiny NumPy/SciPy calls per iteration and all of it is per-call overhead. **Models with an
inner iteration are where the real-time port lives.**

**How to apply:** before scoping the next batch, ask what its clients are *polymorphic over*, not
only what names they read. `airbox` is next (3,976 lines, six factorizations, already half-swapped
by §28.2's `splu`), then `connection`, then `analysis/`.
