---
name: rust-phase2-wind-state
description: "Rust migration Phase 2 batch 3 — bore + reed (the whole wind leg); a `&mut self` pymethod CANNOT hand control back to Python and still be read, and the reed's Brent fallback FIRES so SciPy's brentq had to be transcribed"
metadata: 
  node_type: memory
  type: project
  originSessionId: d917aa2e-ab9e-45ab-a399-cf715203459a
  modified: 2026-08-26T17:58:38.999Z
---

Phase 2 batch 3 landed 2026-08-26 (plan §13): `bore.py` + `reed.py`, 890 lines, the whole wind leg
as **one batch** — because the question was never how to transcribe either model but **how the
exciter seam crosses the language boundary**, and that needs both sides present. Continues
[[rust-phase2-body-state]]; the models themselves are [[bore-state]] and [[reed-state]].

**THE FINDING: a `&mut self` `#[pymethods]` function cannot hand control back to Python and still
be read.** PyO3 holds a `PyRefMut` on the object for the whole body of such a method. `Bore.step`
calls out mid-step through the `source` hook, and the reed's hook does `self.bore.p[0]` — a
perfectly ordinary read the original allows. The obvious binding refuses it with
**`RuntimeError: Already mutably borrowed`**. Fix: `step` takes the **object**
(`slf: &Bound<'_, Self>`) and borrows it in two short phases with the callback in between, holding
nothing. Three things to carry forward: it is **invisible to `cargo test`**, which never crosses the
boundary (the native struct takes the same hook happily); **every model that calls out mid-step
inherits the shape**, so `bow` should start from it rather than meet the error; and the **error path
is contract too** — when a Python callable raises, `step` propagates before committing and the bore
is left un-stepped, so the native hook is fallible so it can refuse the same way.

**The seam's answer, both halves.** The hook stays a **general Python callable** given a live
writable view of the uncommitted `p_next` (§12.8's pre-work: `test_reed_stability` passes its own
`lambda p: None`, so it is interface not scaffolding) — AND `PyReedBore` **requires a `PyBore`** and
injects through a Rust closure, so the clarinet's hot loop crosses once per `step()` instead of
twice. Handed the pure-Python `BorePy` it raises **`TypeError` rather than falling back**: a silent
fallback would be a Rust reed reporting Rust while blowing a Python tube.

**THE OTHER FINDING: the reed's bracketed Brent fallback FIRES, so it had to be transcribed.**
Measured before writing any Rust, over 4,000 steps: 0 at `p_mouth=1200`, **5 at the flagship 1500**,
13 at 1800, 0 below threshold, and **219 on a coarse `N=40` grid**. `physsynth-core`'s dependency
list is empty by design, so there was no SciPy to call — the choice was transcribe
`scipy/optimize/Zeros/brentq.c` (~90 lines) or drop the reed out of the bit-identical bucket.
**Checked before it was relied on:** the same transcription written in Python and run against the
real `brentq` on the reed's own residuals over **248 real calls** returned **bit-identical** roots
every time. Lives in `crates/physsynth-core/src/root.rs`.

**A branch choice is part of the trajectory, not a diagnostic.** If Rust stalls Newton on a
different step than Python the two separate **structurally**, not by rounding, and no energy bar
sees it. So `fallbacks` is compared **step for step** over 2,000 steps — a sampled comparison would
find the trajectories still identical long after the branches diverged, because the two roots agree
to ~1e-13. Measured: bit-identical over 4,000 steps in every configuration including the coarse grid
where Brent fires **270 times on exactly the same steps**.

**The stall test is `!(|r_new| < |r|)`, NOT `>=`.** The original spells it
`if not (abs(r_new) < abs(r))`, which is **true for a NaN** residual; the inverted spelling is false
and would iterate on a NaN forever. `clippy::neg_cmp_op_on_partial_ord` asks for the wrong one — the
`allow` carries the reason and a native test pins it.

**§12.8's one-ulp finding was diagnosed wrong and the correction is more useful.** It is
**associativity, not `pow`**: `c0**2 == c0*c0` exactly at 343 m/s (343² = 117649 is exact), so the
divergence is `rho0*(c0*c0)` vs `(rho0*c0)*c0`. BUT **`x**2 != x*x` in 79 of 200,007 random
doubles**, so `**2` still has to be spelled `powf(2.0)` in general, exactly as Phase 1 spelled
`h**4`. Both spellings preserved on both sides; both parity files assert it so a future tidy-up
fails loudly.

**Two smaller ones.** A bell at **both** ends books each end's energy inside its own
`_radiate_node`, so it accumulates `(E + e_l) + e_r` and never `E + (e_l + e_r)` — a claim about the
order of two additions, invisible to every energy bar, and the test *fails* if the chosen
configuration cannot distinguish the orders. And **`R_bell > 0` with neither end radiating is a
legal bore** whose `U_out` read-out pair still rotates, because the early exit keys on the
resistance and not on the ends.

Measured: `cargo test --workspace` **141 passed** (90 before); parity **148 bore + 104 reed**; the
bore's and reed's own tests under `PHYSSYNTH_RS=1` **97 passed**. `test_reed_signature.py` is in the
CI list deliberately — the step's internal ordering (pin → hook → drain → momentum) is load-bearing
and **no energy test can see it**.

**Two surface traps found only by TRYING things, not by reading code.** (1) `bernoulli_flow` is a
module-level function that `tests/test_reed_stability.py` imports **by name** — and that file is in
the flagged CI step, so without swapping the function too it would have gone on asserting the Python
jet while the run reported Rust. **Grep clients for direct imports of module-level functions, not
just for the class.** (2) `boundary=` unpacks **any** 2-sequence, so a **list** is legal — and a
list is what a JSON round-trip produces, which matters because `web/serialize.py` is a live client
for the whole migration. A tuple-only parse compiled and passed every test in the repo, because
every call site writes a literal tuple. One divergence there **cannot** be closed: PyO3 maps an
omitted keyword and an explicit `None` to the same Rust `None`, so `Bore(boundary=None)` raises in
Python and defaults in Rust — true of every object-typed defaulted parameter in the binding layer,
`string_ideal`'s included. Recorded as a test rather than hidden.

**The viewer is the wide client here, and it was checked by hand:** `tests/test_web_backend.py`
**408 passed** under the flag. It is the only caller that drives `Lop`/`Cmat` through a
**shift-invert** `eigsh` at `k = n_modes + 1` rather than the `k = 1` the tests use.

**Why:** These are the decisions and traps that cost real measurement time this batch and that the
next models (`radiation`, `engine`, then `bow` and `collision` in Phase 3) will hit again.

**How to apply:** Before porting a model that calls back into Python mid-step, start from the
two-phase object borrow. Before assuming a scalar solve is bit-reproducible, **measure whether the
original's fallback branch actually fires** and whether it called a SciPy routine — that decides
whether the parity file can assert equality or must drop to a tolerance. And keep deriving buffer
and surface lists by grepping clients rather than reading intent off underscores
([[rust-phase2-body-state]]).
