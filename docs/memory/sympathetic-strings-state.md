---
name: sympathetic-strings-state
description: "Sympathetic / coupled strings (SympatheticStrings, core/connection.py) — N fixed/free strings sharing ONE bridge point on a common ModalBody; scalar-sum body force = zero ModalBody edits; antisymmetric-normal-mode oracle (bridge exactly still) is the money test energy can't see"
metadata: 
  node_type: memory
  type: project
  originSessionId: c864443c-7545-4044-9bd0-8552e20034ea
  modified: 2026-07-18T13:08:13.695Z
---

**Sympathetic / coupled strings** — first of the coupled-string family (HANDOFF §12.B),
human-chosen next direction (2026-07-10) over 2D-snare/new-method/web-viewer. `SympatheticStrings`
in `core/connection.py` (alongside `StringBodyBridge`/`StringPlateBridge`). Substrate for piano
unisons + sitar/tanpura sympathetics.

**Topology = A0 (advisor steer, build FIRST): all N strings attach at the SAME body driving
point.** So the body feels the **scalar sum** `F = Σ_j F_j`, `F_j = K_j η_j`, `η_j = u_end,j − w_b`,
`w_b = Σ_i φ_i q_i` (the shared bridge displacement). ⇒ **`ModalBody.step(force=Σ F_j)` used AS-IS,
zero ModalBody edits** (scalar path bit-identical). Strings talk ONLY through the common `w_b`.
Rejected the draft (per-string `φ^(j)` vector-force hook): on a lumped modal bank the `φ^(j)` are
*invented* spatial inputs that weaken every oracle — the shared point removes that freedom and hands
the exact antisymmetric invariant. **A1 (distinct bridge points, ~20-line vector-force follow-on)
and grid-`Plate` soundboard are DEFERRED.**

**Still one linear leapfrog → energy exact.** `H = E_body + Σ_j E_string,j + Σ_j E_conn,j`,
`E_conn,j = ½ K_j η_j^n η_j^{n-1}`. Per-step increments telescope (string `−kF_jδu_end,j`, conn
`+kF_jδη_j`, body `+k(ΣF_j)δw_b`; `η_j=u_end,j−w_b` cancels) — the modal-bridge proof, one spring
per string. Step order EXACTLY mirrors `StringBodyBridge` (F from pre-step state; each string
`.step()` then `.u[-1]-=β_s,j F_j`; body `.step(force=ΣF_j)`) → single-string bit-identical to
`StringBodyBridge`. Per-string `β_s,j = 2k²/(ρ_j h_j)` (strings may differ in every param, share
only `k`).

**THE money test = the antisymmetric normal mode (advisor's #1 point: energy/passivity are
AUTOMATIC from the linear-leapfrog structure → they PASS a flipped coupling sign; NOT
discriminating).** Two identical strings, equal K, body at rest, IC `u_B = −u_A`: by symmetry
`F_A+F_B = −2K w_b` → a bridge at `w_b=0` feels zero force and stays there ∀t; `E_body ≡ 0`,
`u_B ≡ −u_A` to machine precision. **Verified BIT-EXACT** — diagnose prints `max|w_b| = 0.00e+00`
(IEEE: `−(2a)=2(−a)`, `K·(−x)=−(K·x)`, `sum([a,−a])=0.0` all exact), stronger than the ~1e-13 the
advisor predicted. Contrast = symmetric IC (`u_B=+u_A`): bridge swings, body loads to 58 %.
`test_antisymmetric_mode_keeps_bridge_still` asserts `allclose(u_B,−u_A,atol=1e-15)` every step +
`max|w_b|<1e-13` + `max E_body<1e-13·E0`.

**Guard = dense `k²λ_max(A)<4`** (NOT hand-rolled Woodbury for the rank-J spring — advisor). Extended
`_apply_A`/`_max_leapfrog_eigenvalue` to the STACKED state `[str_0 nodes 1..N_0, …, str_{J-1}, q_1..q_M]`
via `_offsets=cumsum`; all strings couple to shared `w_b`, body block gets `−φ_i(Σ_j K_jη_j)/m_i`.
Small (`ΣN_j+M`), one dense `eigvals` at construction. Cheap 2-DOF bound is the documented footgun.
λ<1 required per string (Nyquist marginal at λ=1, spring pushes unstable).

**Signatures.** Sympathetic transfer (threshold): softer bridge `K=1500` → frequency-SELECTIVE
coupling → tuned unison neighbour peaks **77.7 %** of total energy (near-complete coupled-oscillator
exchange) vs detuned ~4-semi (T=120) **12.1 %**, 6.4×; `K=8000` is broadband (poor selectivity,
detuned still 24 %). NB accidental partial-overlap resonances (e.g. T=80→30 %) so the test picks a
clean detune point. **Weinreich two-stage decay**: two mistuned unisons over a LOSSY bridge,
**strike ONE string** → excites symmetric+antisymmetric equally → symmetric loads lossy bridge (fast
prompt, ~½ energy) + antisymmetric barely loads it (slow aftersound plateau) → total-string-E log
plot shows the knee (plucking BOTH is pure-symmetric, single-slope — the fix). **NOW SHIPPED in the
web viewer** (batch 7): the anchor sub-claim is that at detune=0 the antisym tail is bit-exactly
lossless (the antisym mode is bridge-decoupled ⟹ body damping never activates on it ⟹ aftersound
slope≈0, rising with detune). See [[web-viewer-state]] Phase-D batch 7 for the viewer's
measured defaults (σ_body=20 not the diagnose's 10 — 10 gave a 0.6 s⁻¹ prompt, no visible knee).

Scope: purely LINEAR sympathetic resonance. Jawari/buzz = nonlinear contact, composes
`collision.py` model #8 LATER — kept out this batch (advisor).

Files: `SympatheticStrings` in `physsynth/core/connection.py`; `make_sympathetic` in `tests/helpers.py`
(detune via per-string `Ts`, downward to keep λ<1; fs from first tension); `tests/test_sympathetic.py`
(19 tests); `scripts/diagnose_sympathetic.py` (4-panel: normal-mode bridge trace + body-E, transfer
tuned/detuned, Weinreich). Full suite green. Next in family: A1 distinct bridge points, or
string↔string via grid-plate soundboard, or jawari (collision + sympathetic). **Now surfaced in the
web viewer** — [[web-viewer-state]] Phase D batch 6 (normal-mode bit-exact-w_b oracle + transfer
slosh) + batch 7 (Weinreich two-stage decay = the FIRST body-loss slider; passivity via
decay_oracle=False; strike-one plateau vs strike-both decay-away contrast). See [[body-bridge-state]],
[[free-plate-bridge-state]], [[barrier-collision-state]], [[commit-push-at-batch-end]],
[[respect-ruff-line-length]].
