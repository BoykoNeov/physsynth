# Memory index

One line per memory — a hook to decide relevance, not the content. Open the file for the detail.

## Working rules & preferences

- [Commit & push at batch end](commit-push-at-batch-end.md) — batch end → memory+docs, commit, push; "commit" always pushes
- [Parity files run UNFLAGGED](parity-files-run-unflagged.md) — SUPERSEDED by phase A: no flag exists; still reinstall the wheel first
- [Respect ruff line length](respect-ruff-line-length.md) — ≤100 chars in the FIRST draft; CI fails fast on `ruff check .`
- [Destructive undo discipline](destructive-undo-discipline.md) — never undo a temp edit with `git checkout --`; it discarded a whole batch
- [Identify processes before killing](identify-processes-before-killing.md) — read the command line first; the user runs their own Python here
- [Port reclaim modus operandi](port-reclaim-modus-operandi.md) — reclaim a busy port only from a stale run of THIS program
- [Unphysical params are a feature](unphysical-params-are-a-feature.md) — effective-coefficient APIs that permit odd combos; realism via helpers

## Rust migration and Python retirement

- [Rust migration state](rust-migration-state.md) — Python retired for Rust (2026-08-26); 1e-15 cross-language promise RETRACTED
- [Rust Phase 0](rust-phase0-state.md) — binding buffers must be Python-owned arrays (a Vec view is a use-after-free)
- [Rust Phase 1](rust-phase1-state.md) — `operators`; dep list stays EMPTY; `h**4` ≠ `h*h*h*h`
- [Rust Phase 2 b1](rust-phase2-state.md) — first loop; the first thing to break bit-identity is a SOLVER; port in halves
- [Rust Phase 2 body](rust-phase2-body-state.md) — a leading underscore is not an interface statement
- [Rust Phase 2 wind](rust-phase2-wind-state.md) — a branch choice is part of the trajectory
- [Rust Phase 2 radiation](rust-phase2-radiation-state.md) — bit-identity ends at a BLAS reduction that feeds back
- [Rust Phase 3 banded](rust-phase3-banded-state.md) — an anchor between two classes BINDS them into one unit
- [Rust Phase 3 strings](rust-phase3-strings-state.md) — both obstacles were evaluation ORDER, fixed Python-side
- [Rust Phase 3 collision](rust-phase3-collision-state.md) — a vectorized NumPy function is TWO computations
- [Rust Phase 3 tension](rust-phase3-tension-state.md) — a reduction a root-find branches on is not a last bit
- [Rust Phase 2 mallet](rust-phase2-mallet-state.md) — LLVM folds constant exponents: test spelling in BOTH profiles
- [Rust Phase 3 bow](rust-phase3-bow-state.md) — a hand hoist is the third two-spelling hazard
- [Rust Phase 3 barrier](rust-phase3-barrier-state.md) — bit-identity across a sum is a claim about its LENGTH
- [Rust Phase 4 beam](rust-phase4-beam-state.md) — SuperLU is supernodal, so Group D runs on measured tolerance
- [Rust Phase 5 outline](rust-phase5-outline-state.md) — a DISCRETE output: measure the margin first
- [Rust Phase 5 matrices](rust-phase5-matrices-state.md) — values and stored ORDER are two questions
- [Rust Phase 5 von Karman](rust-phase5-vk-state.md) — re-associate is the third remedy; read the energy
- [Rust Phase 5 plate](rust-phase5-plate-state.md) — an anchor bound the model to a client that re-derives its matrix
- [Rust Phase 5 geometric](rust-phase5-geometric-state.md) — a divergence that changes no digit (sparse-LU ordering)
- [Rust Phase 5 airbox](rust-phase5-airbox-state.md) — `np.sum` is left-to-right below 8 elements
- [Rust Phase 5 ports](rust-phase5-ports-state.md) — pairwise `np.sum` IS transcribable
- [Rust Phase 5 wrappers](rust-phase5-wrappers-state.md) — the tier BELOW decides what porting this tier means
- [Rust Phase 5 membrane](rust-phase5-membrane-state.md) — a getter without a setter makes an attribute read-only
- [Rust Phase 5 connection](rust-phase5-connection-state.md) — a ported caller that computes nothing is SLOWER
- [Rust Phase 7 spectrum](rust-phase7-spectrum-state.md) — the analysis flag swaps the instrument, not the model
- [Rust Phase 7 oracles](rust-phase7-oracles-state.md) — a native bar found a 544% defect the Python always had
- [Rust Phase 7 rotating-wave](rust-phase7-rotating-wave-state.md) — translation OVER; a one-fixture margin is one claim
- [Python retirement](python-retirement-state.md) — both carve-outs REVERSED 2026-09-07; Python to ZERO kills 19k lines of Rust
- [Retirement batch 1](python-retirement-batch1-state.md) — the HOLE: models living only in the doomed binding
- [Retirement batch 2](rust-retirement-batch2-state.md) — the hole is 16 classes / 3 files; order forced: ports → wrappers → bridges
- [Retirement batch 3](rust-retirement-batch3-state.md) — port tier done; a SHAPE refusal becomes a type, a VALUE one stays
- [Retirement batch 4](rust-retirement-batch4-state.md) — six wrappers → one generic; `np.allclose` keeps `rtol=1e-5`
- [Retirement batch 5](rust-retirement-batch5-state.md) — the gong seam; per-step iteration counts matched SuperLU to the bit
- [Retirement batch 6](rust-retirement-batch6-state.md) — the mallet's room MODE as its own type; a two-way-differing twin bar is vacuous
- [Retirement batch 7](rust-retirement-batch7-state.md) — all three bridges native (`BridgeBody`, one generic `StringPlateBridge`); the hole is CLOSED; finding 79
- [Retirement phase A](retirement-phase-a-state.md) — flag GONE; banded deleted; default suite == old flagged; reconcile counts via a worktree
- [Viewer stays Python](viewer-stays-python.md) — SUPERSEDED; only coverage proves a line runs, not a grep
- [Analysis freeze](analysis-freeze-state.md) — 62 fixtures frozen from the Python before it was deleted
- [Deletion phase](rust-deletion-phase-state.md) — all 11 units gone, 23,396 lines, zero physics bars retired
- [Deletion: the beam](rust-deletion-beam-state.md) — an empty `parametrize` collects as a SKIP; delete a drained table
- [Airbox native bars](rust-airbox-native-bars.md) — the audit read a directory, not the runner
- [Deletion 10: the last body](rust-deletion-connection-state.md) — a parity file is HARVESTED; a freeze promotes digits
- [Deletion in halves](rust-deletion-split-guards.md) — a split deletion empties `deleted_bodies`
- [CI flagged-suite collapse](ci-flagged-suite-collapse.md) — HISTORY: flagged job gone at phase A; filter AFTER a computed split
- [NumPy libm CPU dispatch](numpy-libm-cpu-dispatch.md) — NumPy's own transcendentals: bit-identity depends on the CI machine
- [CI runner variance](ci-runner-variance.md) — runners vary ~1.6x; compare within a job
- [Test suite performance](test-suite-performance.md) — bulk-bound; shards computed from the glob; never pass `-q`

## Models

- [Milestone 1](milestone-1-state.md) — ideal string + validation harness; remote `origin` + CI on push
- [HANDOFF §11 closed](handoff-decisions-closed.md) — all five closed 2026-08-10; tolerances STAND
- [Stiff string](stiff-string-state.md) — #2, implicit θ-scheme; also the core-dep allowlist policy
- [Damped string](damped-string-state.md) — #3, frequency-dependent loss; per-mode oracle
- [Membrane](membrane-state.md) — #4, 2-D drumhead; energy ⊥ geometry; λ≤1/√2
- [Plate](plate-state.md) — #5, Kirchhoff SS rectangle; B = L@L
- [Orthotropic plate](orthotropic-plate-state.md) — #5o; where the factor of 2 lives; three detectors, three blind spots
- [Beam](beam-state.md) — free-free Euler-Bernoulli; free ends via half-cell MASSES
- [Free plate](free-plate-state.md) — #5b FFFF; nullspace {1,x,y}; matched Narita/Leissa 0.01%
- [Free orthotropic plate](free-plate-orthotropic-state.md) — needs FOUR constants; grain DOES reorder modes
- [Guitar plate](guitar-plate-state.md) — #5g; the mask is not the outline; staircase = domain-size error
- [VK Newton Part 0](vk-newton-part0-state.md) — cap-vs-wall verdict; classify `capped` positively
- [VK Newton Part 1](vk-newton-part1-state.md) — an FD check cannot see the wrong MAP
- [VK Newton Part 2](vk-newton-part2-state.md) — Newton behind `couple_method`; clears three walls
- [VK Newton Part 3](vk-newton-part3-state.md) — the outcome is NOT monotone; bisection lies
- [VK Newton Part 4](vk-newton-part4-state.md) — grep for the CLAIM, not the files
- [VK Newton Part 5](vk-newton-part5-state.md) — a censored grid gave a clean false null
- [VK Newton Part 6](vk-newton-part6-state.md) — default moves as a FALLBACK; the seed is the risk
- [VK Newton batch](vk-newton-state.md) — the driver is STRAIN, not grid; a third of the wall is the cap
- [Von Kármán plate](von-karman-plate-state.md) — #6, gong + cymbal; conservative implicit Picard
- [Tension string](tension-string-state.md) — #9 Kirchhoff–Carrier; exact Duffing oracle
- [Geometric string](geometric-string-state.md) — #10 (u,w,v); `λ_long` is THE trap; whirl = Mathieu tongue
- [Bow](bow-state.md) — first continuous nonlinear exciter; energy = BALANCE
- [Bore](bore-state.md) — linear bore + bell; staggered p-U leapfrog
- [Reed](reed-state.md) — self-oscillating reed; balance not sufficient → signature oracle
- [Mallet collision](mallet-collision-state.md) — #7, first contact; the 0/0 Taylor branch
- [Mallet-plate](mallet-plate-state.md) — #7p; an implicit resonator hands over a COLUMN
- [Mallet-gong](mallet-gong-state.md) — #7g nested solve; predicted 10-100x cost was ~2x
- [Mallet-gong in a room](mallet-vk-room-state.md) — ownership of the STEP; sample rate flips solver ranking
- [Barrier collision](barrier-collision-state.md) — #8 fret buzz; scipy `lu_solve` not `np.linalg.solve`
- [Jawari](jawari-state.md) — a configuration of #8, zero core code
- [Juari](juari-state.md) — single-node point contact; headline = the tuning curve
- [Sympathetic strings](sympathetic-strings-state.md) — N strings on one bridge point; antisymmetric-mode oracle
- [Body/bridge](body-bridge-state.md) — modal body + string terminus + grid plate as a body
- [Free-plate bridge](free-plate-bridge-state.md) — free plate as a distributed body
- [String↔VK plate bridge](string-vk-bridge-state.md) — failure mode migrates to Picard non-convergence
- [String→gong→room chain](string-vk-room-chain-state.md) — zero core edits; the claim is band overlap
- [Air box](air-box-state.md) — 3-D FDTD room; no single detector of the three suffices
- [Radiation](radiation-state.md) — three air tiers; batch 2's R ≠ batch 3's R
- [Airbox VK Newton](airbox-vk-newton-state.md) — the air load does not move the wall
- [Plate mode-family split](plate-mode-family-split-state.md) — the blocker is a non-monotone sorted list
- [Hand-picked band audit](horizon-bands-audit-state.md) — a floor is not a horizon
- [Orthotropic horizon](horizon-orthotropic-state.md) — one floor in mode index (`sinc²`)
- [Theta-scheme loss lock](theta-loss-lock-state.md) — probed, NOT built; the fix repairs the quieter half
- [Membrane block corner](horizon-membrane-block-state.md) — the CFL ceiling IS the cancellation Courant number
- [Resolution horizon](resolution-horizon-state.md) — two families with opposite signs
- [Horizon promotion](horizon-promotion-state.md) — a deletion's constraint does not transfer to a move
- [Spectrum detector guard](spectrum-detector-guard.md) — could return a NEGATIVE frequency; guard is local-max

## Viewer

- [Web viewer](web-viewer-state.md) — Phase 3.5 viewer; model list closed by b20; backend does NOT hot-reload
- [Airbox viewer](airbox-viewer-state.md) — `dims:3` is a slice set; the room sets fs
- [Gong-in-room viewer](vkroom-viewer-state.md) — a hidden Chrome tab renders a correct field blank
- [Guitar plate viewer](guitar-plate-viewer-state.md) — waist SWAPS the fundamental; reported as an interval
- [Viewer horizon read-out](viewer-horizon-readout-state.md) — half the viewer must REFUSE (free ends)
- [Horizon on the canvas](viewer-horizon-canvas-state.md) — off-panel draws nothing; both 2-D panels are off-panel
