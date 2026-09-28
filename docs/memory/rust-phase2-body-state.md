---
name: rust-phase2-body-state
description: "Rust migration Phase 2 batch 2 — the modal body; a leading UNDERSCORE is not a statement about the interface, and the speed crossover is about whether NumPy's hot path was already compiled, not about size"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c26d2cb-d5d4-4ef7-8d03-5cb3247880a7
  modified: 2026-08-26T16:37:50.885Z
---

Phase 2 batch 2 landed 2026-08-26 (plan §12): `physsynth/core/body.py` in full — the modal body,
203 lines, the **smallest** resonator in the project and the one with the **longest client list**.
That asymmetry is the batch. Continues [[rust-phase2-state]].

**The finding: a leading underscore says nothing about whether the port may change a name.** The
body keeps its modal acceleration in `_accel`, and **three modules assign to it once per
timestep** — `radiation.RadiatedBody`, the rational air load, and `airbox.RoomLoadedBody`. All
three use the same idiom: snapshot `q^{n-1}`, step, apply a rank-1 correction to `q^{n+1}`, then
**rewrite `_accel`** from the corrected second difference — and all three *rebind* `q` rather than
writing into it. So the body has **three** Python-owned state buffers where the string and membrane
had two, and the third is spelled private. Phase 0 found `connection.py` *reading* the string's
private names; this is one step worse. **Derive the buffer list from grepping for assignment, not
from what the author marked public.** A `_accel` setter that copied instead of adopting would drop
every wrapper's correction while energy, decay and modal frequencies all stayed green.

**The speed story is sharpened and one earlier claim is weakened.** §11.6 said Rust removes
per-step interpreter overhead, not arithmetic (membrane: 8.7x small → ~1.1x by 5,000 unknowns). The
body has **no compiled kernel underneath it at all** — no sparse matvec, no BLAS — just eight
elementwise NumPy ops on short arrays. Result: **~15x flat from 1 to 64 modes** (Python costs ~7 µs
per step whether the bank has one mode or sixty-four — that is dispatch, not arithmetic), decaying
only slowly to 3.6x at 4,096 modes. So **the crossover is not about size, it is about whether the
Python side's hot path was already in C.** Applied forward: the big wins are models whose steps are
many short NumPy expressions (exciters, lumped models, boundary corrections); the small wins are
models already dominated by a SciPy solve.

**And the whole-suite timing is noise.** Batch 1's matched pair read +3.1% (flagged slower), same
sign in all three shards. Batch 2's matched pair reads **+0.14% with the per-shard signs
disagreeing**, and between the two pairs the same machine ran the same suite **~20% faster**. The
drift is bigger than the effect. **Only the controlled per-step benchmark supports a speed claim**;
the suite number is good for "nothing blew up" and nothing else.

**Two smaller ones.** The step's `force` is visible *only* through the acceleration — a port that
reconstructed `q'' = -omega²q - 2σq'` passes every free-response test and silently drops the bridge
force, so the parity sweep drives the body as well as releasing it. And the CFL rejection names
`np.argmax(omega_k)`, not the first offender.

**`bore` was deliberately NOT started.** `bore.step(source=...)` takes a Python callable that
mutates the pressure field **in place, mid-step**, and `reed.ReedBore` is its hot caller. That is
an interface decision about how the **exciter seam** crosses the language boundary, and it commits
the project for `bow` and every continuous exciter after it. Same rule as §11.3: don't fix an
interface before the requirement that chooses it exists. So **`bore` + `reed` go together, next.**

Measured: `cargo test --workspace` **90 passed**; `tests/test_rust_parity_body.py` **51 passed**;
state bit-identical over 4,000 free and 2,000 driven steps incl. `_accel`; the body's 10 client test
files **268 passed** under the flag; whole suite **2,335 passed, 0 failed** both ways.

**Pre-work for the `bore` + `reed` batch, already done (plan §12.8) — three results change the
design.** (1) The Python-callable seam is **load-bearing, not transitional**: `test_reed_stability`
passes its own `lambda p: None` to assert the hook is inert, so the binding must accept an arbitrary
Python callable even after `reed` is ported. Design: `physsynth-core`'s `Bore` takes a **Rust
closure**, the binding wraps a Python callable into one — a PyO3 type inside the core would break
exactly what `deps.rs` guards. (2) `Lop`/`Cmat`/`dof` are **public attributes** reached by four
files with fancy indexing and fed to a generalized `eigsh`, so they are the membrane's `L` problem
again: the binding builds real SciPy objects itself, and Phase 1's triplet shim does NOT apply.
(3) The hook's position inside `step` (after the open-end pin, before the radiating drain, before
momentum) is **invisible to energy** — put `test_reed_signature.py` in the gate, per [[reed-state]].

**And a finding that is not about Rust at all:** `reed.py`'s comment claimed its node-0 compliance
equalled the bore's `_p_pref[0]`. It does not — the bore says `rho0 * c0**2` (a libm `pow`), the reed
says `rho0 * c0 * c0`, and they **disagree by one ulp in 3,531 of 3,552** tube/grid combinations.
This is the `h**4` finding in **Python-vs-Python** form, it long predates the migration, and the
trap is that a port which 'tidied' the two spellings into agreement would change a number the
acceptance runs were taken with. Comment corrected in place; code left alone.

**A limit of the gate (§12.9):** CI's Rust coverage is exactly four named test lists plus the parity
files. The three sharded jobs that cover the whole suite run on the **default Python path** —
`PHYSSYNTH_RS=1` is never set there. The whole-suite flagged run is a **manual measurement taken
per batch**, not a gate.
