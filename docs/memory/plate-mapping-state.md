---
name: plate-mapping-state
description: "Plate musical controls DONE 2026-10-18 (plate_tuning + viewer controls:musical + new plate loss sigma1); exact supported, first-order free; high T60 has a range"
metadata:
  node_type: memory
  type: project
  originSessionId: e4a7015b-6998-4c53-88f0-7af73c3a59f2
  modified: 2026-10-18T11:33:45.874Z
---

Batch done 2026-10-18 (`docs/dev/plate-mapping-plan.md`, status DONE, §9–§14). The human chose:
all three linear plates, thickness OR pitch behind a switch, a second loss term added to the plate
model, woods with grain; after Part 0, the Laplacian loss law (T60 ∝ 1/f) over stiffness-
proportional (exact but highs die as f²).

- Core: `plate::PlateSpec::sigma1` (Bilbao `2σ₁Δu_t`; supported form `−L`, free form
  `ops2d::free_gradient_form`), skipped at 0 so the freeze stays bit-identical; `airbox_wrap::
  theta_matrix` carries it. `plate_tuning.rs`: cited `PRESETS`, `plate_material`, `tune_plate`,
  `lowest_modes`.
- Viewer: plate key `controls: physical|musical` (its `domain` already = supported/free/guitar);
  front-end `CONTROL_OPTS` + `data-musical` / `data-hide-musical` gates; guard reads both tables.

**Why:** next mapping batches (membrane, VK plate, other strings) start from these facts.

**How to apply:**
- Exactness needs the loss to SHARE the stiffness's eigenvectors. Supported −L does (grain too —
  Kronecker build); a free edge's gradient form does not → Rayleigh quotient, ≤0.05 c / ≤0.3 % T60.
  Measure a non-proportional case on the scheme's OWN root (inverse iteration on z²A−zR₁+R₀), not by
  projecting a run onto a mode — projection mixes neighbour leakage in.
- One stiffness constant ⇒ only one free pitch; mode 1's own 2×2 fixes κ.
- σ₀≥0, σ₁≥0 bound t60_high on BOTH sides once t60_low is fixed; the long end sits ABOVE t60_low
  (hurdle §4 rate suppression). Bars that pick ring-time targets must stay inside that range.
- A wood's grain turned 90° turns ν too (`ν_RL = ν_LR E_R/E_L`); unrotated spruce is refused.
- Handbook footnotes: E_L = bending MOE +10 %; density = SG·1000·1.12 (SG on volume at 12 % MC).
- A viewer wiring bar for losses: compare the run's LAST energy with a core plate built from the
  mapping (first energy catches ρ_s; last catches σ₀/σ₁ dropped).
- The Bash tool fails on LONG heredocs ("unexpected EOF while looking for matching `''") — write the
  block with the Write tool to W:\temp\claude and `cat >>` it.
- No Python even for doc edits (I slipped once; Python text mode wrote CRLF — git normalised it).

Related: [[parameter-mapping-state]], [[theta-loss-lock-state]], [[resolution-horizon-state]],
[[orthotropic-plate-state]], [[free-plate-state]], [[unphysical-params-are-a-feature]].
