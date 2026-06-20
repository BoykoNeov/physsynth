# Physical Synthesis Simulator — Handoff & Specification

> **What this is.** A self-contained brief for continuing development of a physical-modeling
> sound-synthesis tool (standalone + DAW plugin). It is written so a fresh Claude Code session
> with no prior context can pick up and continue. It captures the decisions already made (so they
> are not re-litigated), the mathematical foundation, the validation-as-code strategy, and a
> concrete, testable first milestone.
>
> **Companion file:** `CLAUDE.md` is the lean, always-loaded version of the non-negotiables.
> This document is the deep reference behind it.

---

## 0. How to use this document

- Read sections 1–3 first; they are the load-bearing decisions.
- Section 4 is the math you implement from.
- Section 6 is the test suite — build it *before* expanding features.
- Section 10 is the very first thing to build. Start there.
- Section 11 lists the few decisions still genuinely open. Surface them to the human; don't guess silently.

---

## 1. Vision

A physical-modeling synthesis instrument that:

- Runs both **standalone** and as a **DAW plugin**.
- Incorporates **multiple physical-synthesis methods**, starting with one done well, then expanding
  in two directions: **breadth** (more methods) and **depth** (more physics per method).
- Is **interactive** and **beautifully visualized** — the vibrating object itself is the visual.

**Primary optimization target (decided): physical depth / accuracy over polish and over real-time.**
Fidelity comes first. Real-time performance is a *later port*, not a day-one constraint.

---

## 2. Decisions already made — do not re-litigate

1. **Accuracy-first.** Offline rendering at high sample rates / fine grids is acceptable and
   preferred while validating correctness. Real-time is deferred.
2. **Energy-based / passive numerical methods** are the mathematical foundation (the Bilbao
   framework — see §4 and §13). This is chosen over plain digital waveguides specifically because
   it (a) gives provable stability, (b) makes fidelity measurable, and (c) extends to the nonlinear
   models (gongs, cymbals, large plates) that are the eventual payoff.
3. **Prototype stack: Python (NumPy/SciPy).** Julia is an acceptable alternative if the developer
   prefers it (better for numerical PDEs; weaker agent ecosystem). **Not** C++/JUCE for the
   prototype phase — that is for the later real-time port.
4. **Headless DSP core**, fully decoupled from UI and visualization. The core is pure, importable,
   and testable with no audio I/O and no graphics. Everything else wraps it.
5. **Validation is code, not listening.** Correctness is asserted numerically against closed-form
   physics (see §6). This is the central reason the project suits an AI-driven workflow: the agent
   cannot hear bugs, but it can run `pytest` against analytic ground truth.
6. **Unifying abstraction:** every method decomposes into
   `exciter -> resonator (+- nonlinear coupling) -> body/radiation`.
   "Adding a method" usually means writing a new resonator or exciter behind a stable interface,
   not building a new synth.
7. **Depth before breadth for the first arc.** Deepen one resonator family (string) through its
   physics before sprawling into many methods.

---

## 3. Architecture

### 3.1 Layering

```
+-----------------------------------------------------------+
|  Wrappers:  standalone app  |  DAW plugin (later, native)  |
+-----------------------------------------------------------+
|  Visualization / diagnostics (notebook plots -> web later) |
+-----------------------------------------------------------+
|  HEADLESS DSP CORE  (pure, no I/O, no graphics, testable)  |
|    exciter | resonator | coupling | body | engine/loop     |
+-----------------------------------------------------------+
|  Validation harness  (energy, modal, convergence, ...)     |
+-----------------------------------------------------------+
```

The core never imports plotting or audio libraries. Visualization and wrappers depend on the core,
never the reverse. This keeps the physics portable to C++/Rust later without re-derivation.

### 3.2 Core interfaces (conceptual — refine in code)

- **Exciter**: produces an excitation signal or initial/boundary condition.
  `pluck`, `strike`, `bow`, `blow`. Continuous exciters expose per-sample force.
- **Resonator**: a discretized object that advances state one timestep given inputs.
  Must expose: `step()`, current `state` (for visualization snapshots), and an `energy()` method
  returning the discrete energy quantity (see §4) — energy reporting is mandatory, not optional.
- **Coupling**: nonlinear interactions between exciter and resonator (bow friction, reed/lip,
  hammer-felt) and between resonators (string-to-body, sympathetic strings).
- **Body / radiation**: maps resonator motion to output pressure (modal body, or convolution later).
- **Engine**: owns the timestep loop, parameter smoothing, and (eventually) the audio callback.

### 3.3 Real-time safety (for the later native port — note now, enforce later)

- Audio callback: no allocation, no locks, no blocking, ever.
- Parameter changes are smoothed; never apply a raw jump (zipper noise / clicks).
- Visualization reads model state via a **lock-free snapshot / double buffer** — never shares the
  audio thread's working memory directly.

---

## 4. Mathematical foundation: energy-based FDTD

This is the spine. Every model follows the same five moves; the ideal string shows all of them.

### 4.1 The recipe (apply to every model)

1. Write the continuous energy `E(t)`; confirm `dE/dt = (dissipation <= 0) + (boundary flux)`.
2. Discretize. **Put the potential-energy term across two time levels** (a product of the
   spatial gradient at step `n` with the gradient at step `n-1`). This is the non-obvious move that
   makes energy conservation *exact*, not approximate.
3. Form the discrete energy `E^n`; **demand `E^n >= 0`**. That requirement *is* the stability
   condition (CFL for explicit schemes; unconditional for implicit).
4. Implement, then assert: lossless run -> `E^n` flat to machine precision (~1e-13); lossy run ->
   `E^n` monotonically decreasing at the analytic rate.
5. Only then add the next feature.

### 4.2 Ideal string

PDE (transverse displacement `u(x,t)`, wave speed `c = sqrt(T/rho)`, domain `[0,L]`):

```
u_tt = c^2 * u_xx
```

Continuous energy = kinetic + potential (strain):

```
E(t) = (rho/2) * integral_0^L (u_t)^2 dx  +  (T/2) * integral_0^L (u_x)^2 dx
```

Differentiating, substituting the PDE, and integrating by parts gives `dE/dt = T*[u_t*u_x]` at the
boundaries only — interior terms cancel exactly. Fixed (Dirichlet) or free (Neumann) ends kill the
boundary term, so `dE/dt = 0`.

**Grid:** `u[l,n] ~= u(l*h, n*k)`, with `h = L/N` the spacing and `k = 1/fs` the timestep.

**Difference operators** (see Appendix A for the full set):

```
delta_tt u = (u[l,n+1] - 2u[l,n] + u[l,n-1]) / k^2
delta_xx u = (u[l+1,n] - 2u[l,n] + u[l-1,n]) / h^2
```

**Explicit scheme** `delta_tt u = c^2 * delta_xx u`, rearranged:

```
u[l,n+1] = 2u[l,n] - u[l,n-1] + lambda^2 * (u[l+1,n] - 2u[l,n] + u[l-1,n])
where  lambda = c*k/h          (the Courant number — everything hinges on this)
```

**Discrete energy** (the conserved quantity; `<f,g> = h * sum_l f[l]*g[l]`):

```
E^n = (1/2)*||delta_t- u^n||^2  +  (c^2/2)*< delta_x+ u^n , delta_x+ u^{n-1} >
       \___ kinetic ___/          \________ potential, ACROSS two time levels ________/
```

This satisfies `E^{n+1} - E^n = 0` (interior, lossless) as an algebraic identity.

**Stability from positivity.** The kinetic term is a sum of squares (>= 0). The potential term is a
cross-time product and is not obviously non-negative. Requiring `E^n >= 0` forces:

```
lambda = c*k/h <= 1            (the CFL condition)
```

- At `lambda = 1` exactly: zero numerical dispersion; the scheme is *exact* and coincides with the
  digital waveguide / d'Alembert solution. Tune toward `lambda = 1`.
- For `lambda < 1`: numerical dispersion (high partials travel too slowly), worse as `lambda` shrinks.

### 4.3 Damping -> passivity

Add loss: `u_tt = c^2 u_xx - 2*sigma*u_t`, `sigma >= 0`, discretizing the loss with a centered
difference. The energy balance becomes:

```
E^{n+1} - E^n = -2*sigma*k*||delta_t. u^n||^2  <= 0
```

Energy only decreases — the scheme is **passive**, mirroring the physics. Frequency-dependent
damping (high partials dying faster, as real strings do) needs a richer loss operator but the same
passivity bookkeeping.

### 4.4 Boundaries -> summation by parts (SBP)

The discrete analog of integration by parts:

```
< f , delta_xx g > = - < delta_x+ f , delta_x+ g > + (boundary terms)
```

Fixed/free ends kill the boundary terms (energy conserved). A lossy/radiating termination is built
by making the boundary term a controlled negative-definite quantity. **Most "mystery energy leaks"
in a new model live in the boundary handling** — check SBP first when `E^n` drifts.

### 4.5 Why this foundation (implicit + nonlinear)

- **Implicit schemes:** define the potential term via a time-average (mu) -> an implicit update
  (solve a sparse system per step) -> **unconditional stability** (no CFL limit) and better accuracy
  for stiff systems. Given accuracy-first, this trade is usually worth it.
- **Nonlinearity:** frequency-domain (von Neumann) stability analysis does **not** apply to nonlinear
  systems. Energy analysis does. This is the only robust route to stable nonlinear plates
  (von Karman equations -> gongs, cymbals), which are the eventual sonic payoff.

---

## 5. Method ladder (roadmap by method, accuracy-first ordering)

| # | Model | New physics | Validate against | Notes |
|---|-------|-------------|------------------|-------|
| 1 | Ideal string (FDTD) + analytic modal solver | wave equation | harmonic series `f_n = n*c/(2L)` | **First milestone.** Build the validation harness here. |
| 2 | Stiff string | + bending term `-kappa^2 * u_xxxx` | stretched partials `f_n = n*f_1*sqrt(1+B*n^2)` | First model that sounds like a real instrument (piano-like). CFL bound tightens. |
| 3 | Damped string | + frequency-dependent loss | measured decay rates per partial | Passivity test: energy decreases monotonically. |
| 4 | Membrane (2D) | 2D wave equation, circular rim | Bessel zeros `f_{mn} = c*j_{m,n}/(2*pi*a)` | Visual showpiece begins. |
| 5 | Plate (Kirchhoff) | biharmonic operator | `f_{mn} = (pi/2)*sqrt(D/rho_s)*[(m/Lx)^2+(n/Ly)^2]` | Chladni patterns as diagnostics. |
| 6 | Nonlinear plate | von Karman coupling | energy conservation (no analytic modes) | Gongs/cymbals. The deep end. |

Steps 1–3 are one resonator family deepening in physics. Breadth (modal percussion, bowed string,
wind/bore + reed) comes after the string family and the test harness are solid, reusing the same
exciter/resonator interfaces.

---

## 6. Validation & testing strategy (the heart of the AI workflow)

These become the CI suite. Each is a numeric assertion against closed-form physics — no ears needed.

1. **Energy conservation (lossless).** Run with `sigma = 0`. Assert `max|E^n - E^0| / E^0 < 1e-10`
   over the whole run. Drift = bug (wrong operator, boundary, or indexing). This is the strongest
   single correctness test.
2. **Passivity (lossy).** With `sigma > 0`, assert `E^{n+1} <= E^n` for all `n` (within tolerance),
   and that the decay rate matches the analytic value.
3. **Modal frequency match.** Excite, take the spectrum, compare detected partials to the analytic
   oracle for that model (table in §5). Assert per-partial error below a tolerance in cents.
4. **Convergence order.** Refine the grid (`h -> h/2`); error vs the analytic solution should shrink
   at the scheme's theoretical rate. If it doesn't, the scheme isn't what you think it is.
5. **Dispersion relation.** Measure phase velocity vs frequency; compare to theory. Confirms the
   `lambda = 1` exactness and quantifies dispersion for `lambda < 1`.
6. **A/B against recordings (later).** Spectral + per-partial decay comparison to real instruments,
   once analytic checks pass.

**Rule:** the energy report (`resonator.energy()`) and tests 1–4 must exist before model #2 is started.

---

## 7. Visualization plan

- **Phase 1 (now):** diagnostics *are* the visuals — energy-over-time traces, detected-vs-analytic
  partial plots, convergence plots, dispersion curves, mode shapes / Chladni patterns, spectrograms
  over a recording. These are both correctness tools and genuinely striking.
- **Later:** animated real-time model state (string displacement, membrane ripple, plate modes),
  moving to web/WASM for interactivity. Render reads state via lock-free snapshot (see §3.3).
- Visualization code lives outside the core and may import plotting freely.

---

## 8. Known problems / pitfalls

- **Numerical stability / CFL.** Explicit schemes blow up if `lambda > 1`; energy test catches it.
- **Boundary energy leaks.** The usual source of conservation drift. Check SBP.
- **Aliasing in nonlinear elements.** Bow, reed, nonlinear plate fold HF down as garbage —
  oversample around nonlinearities.
- **Parameter mapping.** Raw physics (Young's modulus, tension, mode damping) is not musician-friendly.
  Budget real work for mapping ugly physical params to a few intuitive macros, with smoothing.
- **CPU budget x polyphony.** A single FDTD plate can saturate a core. Decide which models are
  polyphonic. (Matters at the real-time stage, not now.)
- **Real-time safety.** No alloc/lock/block in the callback. (Real-time stage.)
- **Visualization/audio thread coordination.** Lock-free snapshot only. (Real-time stage.)
- **Testing without ears.** Mitigated by §6 — keep it first-class.
- **Plugin build/distribution.** Codesigning/notarization (macOS), installers, format wrappers.
  (Real-time/ship stage.)

---

## 9. Phases

- **Phase 0 — Scaffolding.** Repo, package layout, test runner, CI, the exciter/resonator/coupling/
  body interfaces with a null/no-op core, and the empty validation harness wired to CI.
- **Phase 1 — MVP (see §10).** Ideal string FDTD + analytic modal solver + energy/modal/convergence
  tests + diagnostic plots.
- **Phase 2 — Stiffness + frequency-dependent damping.** Models #2–3.
- **Phase 3 — 2D membrane + plate.** Models #4–5; mode-shape visualization.
- **Phase 4 — Nonlinear plate.** Model #6; energy-method stability for gongs/cymbals.
- **Phase 5 — Real-time port + plugin.** Port validated DSP to C++ (JUCE) or Rust (nih-plug/CLAP);
  standalone + VST3/AU. Verify current framework/format support at this point.

---

## 10. First milestone — concrete spec (START HERE)

**Goal:** an ideal-string solver plus a validation harness that proves it correct. The deliverable
is not "a string" — it is "a string *and* a rig that measures exactly how its partials deviate from
theory." This sets the project's culture.

**Inputs / parameters:**
- `L` (length, m), `T` (tension, N), `rho` (linear density, kg/m), `fs` (sample rate, Hz),
  `N` (spatial grid points), boundary type (`fixed` | `free`), excitation (pluck position + shape),
  `sigma` (loss; 0 for the lossless tests).
- Derive `c = sqrt(T/rho)`, `h = L/N`, `k = 1/fs`, `lambda = c*k/h`. Assert `lambda <= 1`.

**Solver:**
- Implement the explicit scheme in §4.2 first (simplest; exact at `lambda = 1`).
- Expose `step()`, `state` (displacement array, for plotting), and `energy()` returning `E^n` from §4.2.
- Provide an analytic modal reference: `f_n = n*c/(2L)` and (optionally) a modal-synthesis solver to
  cross-check the FDTD output.

**Acceptance criteria (the milestone is done when all pass):**
1. **Energy conserved:** lossless run, `max|E^n - E^0|/E^0 < 1e-10` over >= 2 seconds of output.
2. **Partials correct:** detected spectral peaks match `n*c/(2L)` within ~1 cent for the first
   ~10 partials at `lambda` near 1.
3. **Convergence:** halving `h` reduces partial error at the expected rate.
4. **No NaN / no blow-up** across a sweep of valid `lambda` in `(0, 1]`; and a deliberate `lambda > 1`
   case is rejected at construction time.
5. Diagnostic notebook/script renders: energy-vs-time, detected-vs-analytic partials, and the string
   displacement animation.

**Suggested layout (Python):**
```
physsynth/
  core/
    operators.py        # difference operators (Appendix A)
    string_ideal.py     # IdealString resonator: step(), state, energy()
    exciter.py          # pluck/strike
    engine.py           # timestep loop
  analysis/
    modal.py            # analytic oracles (harmonics, later Bessel/plate)
    spectrum.py         # partial detection
  viz/
    plots.py            # energy, partials, convergence, dispersion
    animate.py          # displacement animation
tests/
  test_energy.py        # criteria 1, 2
  test_convergence.py   # criterion 3
  test_stability.py     # criterion 4
notebooks/
  01_ideal_string.ipynb # criterion 5
CLAUDE.md
HANDOFF.md
```

---

## 11. Open decisions — surface to the human, do not silently guess

1. **Python vs Julia** for the prototype. (Default: Python, unless the human prefers Julia.)
2. **Explicit vs implicit reference solver.** Explicit at `lambda = 1` is exact and simplest to start;
   implicit gives unconditional stability and better stiff-string accuracy. Recommendation: start
   explicit for milestone 1, add an implicit path at the stiff-string stage.
3. **Which models are polyphonic** (affects the engine design eventually).
4. **First interactive visualization target:** notebook plots only for now, or jump to an interactive
   web view sooner?
5. **Tuning of test tolerances** (cents thresholds, energy-drift bound) with the human's fidelity bar.

---

## 12. Expansion horizon (future directions)

This is a map of where the project can grow, not a commitment. It is deliberately beyond current
scope. The two foundational decisions — **energy/passivity** and a **headless, interface-driven
core** — were chosen precisely because they keep most of these reachable: a new resonator, exciter,
or coupling drops in behind the existing interfaces, and energy bookkeeping keeps it stable. Pick
threads from here as the project matures; each bullet is a seed, not a spec.

### A. More methods (breadth)

- **Modal synthesis from data:** resonator banks driven by modal frequencies/dampings extracted
  from FEM meshes or measured impulse responses — lets *any* object become an instrument.
- **(Banded) digital waveguides:** bars, bowed bars, glass harmonica, singing bowls.
- **2D/3D waveguide meshes** for membranes, plates, rooms.
- **Mass–spring / mass-interaction networks** (CORDIS-ANIMA / ACROE paradigm): build arbitrary
  topologies of masses and springs — the natural substrate for a "build-your-own-instrument" UX.
- **Finite element method (FEM)** for arbitrary, non-analytic geometries.
- **Wind/brass:** full bore + reed/lip nonlinearity, toneholes, jet models; vocal-tract coupling
  (didgeridoo, beatbox, talking instruments).
- **Vocal synthesis:** articulatory vocal tract as a waveguide with a glottal-flow exciter (speech, singing).
- **Friction/contact/foley:** scraping, rolling, rubbing, rattles — environmental and sound-design use.

### B. Deeper physics (depth)

- **Full instrument chains:** string → bridge → body → radiated air, with realistic coupling.
- **Nonlinearities everywhere:** tension modulation, geometric (von Kármán) nonlinearity, and
  **collisions/contact** — string–fret, hammer/mallet, beating reeds, snares, prepared-piano rattles,
  bottleneck-slide friction.
- **Sympathetic resonance / coupled systems:** piano soundboard with all strings, sitar jawari,
  sympathetic strings, multi-string coupling.
- **Material realism:** anisotropy, layered/composite materials, inhomogeneity/aging, and
  temperature/humidity dependence.
- **Tube acoustics:** thermoviscous losses, radiation impedance at the bell.

### C. Numerical-methods frontier

- **Port-Hamiltonian formulation (high leverage):** model every subsystem as an energy-exchanging
  set of ports. It generalizes the energy/passivity philosophy already chosen and lets heterogeneous
  subsystems (string + nonlinear bridge + air) be coupled while *provably* preserving passivity.
  This is the structural way to scale §3's coupling without each new junction risking instability.
- **Implicit/energy schemes throughout**, higher-order and spectral methods for accuracy.
- **Model-order reduction** (POD, balanced truncation): FEM-level accuracy at modal-synthesis cost.
- **Differentiable physical models (high leverage):** make the solver differentiable so physical
  parameters can be fit by gradient descent — see §D.
- **ML surrogates:** neural operators / DeepONet / PINNs that approximate an expensive solver for
  real-time playability while keeping physically meaningful controls.

### D. Inverse problems — calibration & sound matching

- **Auto-calibration:** given a recording of a real instrument, fit the model's physical parameters
  to reproduce it (leans on differentiable simulation from §C).
- **Automatic perceptual mapping:** learn the map from raw physics (tension, stiffness, damping) to a
  few intuitive macros — directly attacks the parameter-mapping pitfall in §8.

### E. Performance, real-time, platforms

- **GPU acceleration:** FDTD grids and meshes map naturally to GPU (CUDA / Metal / compute shaders /
  WebGPU); the bridge from accuracy-first offline to interactive real-time.
- **WASM + WebGPU + SIMD** for a browser-native interactive build.
- **Embedded / hardware:** Bela, Daisy, eurorack module, or FPGA for ultra-low-latency playable
  instruments; mobile.
- **Voice management:** polyphony, voice stealing, and per-method polyphony budgets.

### F. Expressive control & haptics

- **Expressive controllers:** MPE, MIDI 2.0, breath/wind controllers, pressure/force sensors —
  mapping real gesture (bow speed/pressure/position, breath, embouchure) to the exciter.
- **Haptic / force-feedback** controllers that let the player *feel* the model — closes the loop with
  the mass-interaction paradigm (§A).
- **Native microtonality / alternate tunings:** physical models tune continuously — a built-in strength.

### G. Visualization, UX, and the instrument sandbox

- **Real-time 3D** rendering of the vibrating object with modal/Chladni overlays, slow-motion, and
  synchronized spectrograms.
- **"Build-your-own-instrument" sandbox:** draw a shape, assign materials, hear it — the dream UX,
  built on FEM (§A) + mass-interaction (§A) + the headless core.
- **Educational mode:** expose the live physics and math, extending the diagnostics-as-visuals
  philosophy already in §7.
- **VR/AR:** walk around and play a giant resonating object in spatial audio.

### H. Spatial audio, radiation & room

- **3D radiation modeling** with directivity, plus HRTF / ambisonics output.
- **Room coupling:** place the modeled instrument inside a modeled acoustic space.

### I. Ecosystem & product

- **Instrument/preset library** of calibrated real instruments, with a shareable format.
- **Multi-format plugin** (VST3 / AU / AAX / CLAP) + standalone + a reusable DSP **SDK/library**.
- **Patching / scripting language** for defining models (a physical-modeling DSL; or integrate with
  Faust, which already has a physical-modeling library).
- **Open-source core** with a commercial plugin on top, if that fits the goals.

### J. Research / novel territory

- **Hyperreal instruments:** physics beyond real materials — negative stiffness, time-varying or
  morphing geometry (a string becoming a tube mid-note), non-Euclidean or fractal resonators.
- **Mega-instruments:** many objects coupled into one playable physical system.

---

## 13. References

- S. Bilbao, *Numerical Sound Synthesis: Finite Difference Schemes and Simulation in Musical
  Acoustics*, Wiley, 2009. **Primary reference** — strings, bars, membranes, plates, nonlinear
  plates, all with stability analysis.
- J. O. Smith, *Physical Audio Signal Processing*, CCRMA (free online) — digital waveguides,
  Karplus-Strong, scattering junctions.
- A. Chaigne & J. Kergomard, *Acoustics of Musical Instruments* — ground-truth instrument physics.
- K. Karplus & A. Strong, "Digital Synthesis of Plucked-String and Drum Timbres," CMJ, 1983 — the
  simplest entry point if a quick breadth demo is ever wanted.

> Verify any framework/format/version specifics (JUCE, CLAP, nih-plug, plugin formats) at the
> real-time/ship stage — that tooling moves and is outside this document's scope.

---

## Appendix A — Difference operators (notation)

Time step `k`, grid spacing `h`.

```
Forward time:     delta_t+  u[n] = (u[n+1] - u[n]) / k
Backward time:    delta_t-  u[n] = (u[n]   - u[n-1]) / k
Centered time:    delta_t.  u[n] = (u[n+1] - u[n-1]) / (2k)
Second time:      delta_tt  u[n] = (u[n+1] - 2u[n] + u[n-1]) / k^2   = delta_t+ delta_t-
Time averages:    mu_t+, mu_t-, mu_t.   (analogous; used for implicit schemes)

Spatial operators delta_x+, delta_x-, delta_xx, mu_x*  are defined identically over l.
Fourth-order (stiffness): delta_xxxx = delta_xx delta_xx   (for the -kappa^2 u_xxxx term).
```

Inner product over interior grid points: `<f,g> = h * sum_l f[l]*g[l]`; norm `||f||^2 = <f,f>`.
