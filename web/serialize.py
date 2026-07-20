"""Pure serialization seam for the web viewer (wrapper, NOT core).

``simulate_to_payload(params)`` is the whole backend: a params dict -> build a string resonator
(``ideal`` / ``stiff`` / ``damped`` dispatch) -> pluck -> run the existing
:func:`physsynth.core.engine.simulate` -> a JSON-able payload. It is socket-free and graphics-free,
so it is unit-tested directly (``tests/test_web_backend.py``). The HTTP shell (``web/server.py``) is
a thin wrapper over this function.

Design decisions baked in (see ``docs/dev/web-viewer-plan.md``, advisor review 2):
  * **catch #1** — the pickup is resampled to a fixed :data:`AUDIO_FS` (48 kHz); ``fs_sim`` rides
    N and c (sliders) and can blow past the browser ``AudioBuffer`` cap. ``audio.fs`` means 48 kHz.
  * **catch #2** — the animation captures only ``animation_window`` seconds at a stride that
    resolves the *fundamental* (~:data:`FRAMES_PER_PERIOD` frames/period), decoupled from the audio
    length; the frontend plays it in slow motion. A wall-clock 60 fps stride would alias the wiggle.
  * **catch #4** — energy is reported as a *correctness drift* only when lossless; with loss it is
    reported as a passivity (monotone-decrease + decay-rate) check, not a scary "drift" number.
  * **catch #5** — per-model constructor params and partials oracle
    (see :func:`_build_resonator`).

Depends on ``physsynth.core`` + ``physsynth.analysis``; never imported by the core.
"""

from __future__ import annotations

import base64
import math
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly
from scipy.sparse.linalg import eigsh

from physsynth.analysis import damping, dispersion, duffing, modal, spectrum
from physsynth.analysis.rotating_wave import rotating_wave_history, solve_rotating_wave
from physsynth.core.body import ModalBody
from physsynth.core.bore import C0_AIR, RHO0_AIR, Bore
from physsynth.core.bow import BowedString
from physsynth.core.collision import BarrierString
from physsynth.core.connection import SympatheticStrings
from physsynth.core.engine import SimResult, simulate
from physsynth.core.exciter import raised_cosine_2d, triangular_pluck
from physsynth.core.mallet import MalletMembrane
from physsynth.core.membrane import Membrane
from physsynth.core.plate import Plate, VKPlate
from physsynth.core.reed import ReedBore
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_geometric import GeometricString
from physsynth.core.string_ideal import IdealString
from physsynth.core.string_nonlinear import TensionModulatedString
from physsynth.core.string_stiff import StiffString

# -- constants -------------------------------------------------------------------------------------

AUDIO_FS = 48_000.0          # fixed audio output rate (catch #1)
FRAMES_PER_PERIOD = 12       # animation temporal resolution (catch #2)
N_ENERGY_POINTS = 600        # decimated energy-trace length for the plot
N_PARTIALS = 12              # partials overlaid on the spectrum
MAX_FRAMES = 1500            # payload guard (frame count ceiling)
LOSSLESS_TOL = 1e-10         # the project's energy-drift bar (HANDOFF §6)
MONOTONE_TOL = 1e-9          # relative per-step energy-increase tolerance for the passivity check

# clamps — a local tool will OOM / hang on absurd inputs
N_MIN, N_MAX = 2, 2000
AUDIO_DUR_MAX = 10.0
ANIM_WIN_MAX = 2.0
SPEED_MAX = 8.0

# --- membrane (2D, Phase B) clamps + display budget ---
# Cost is ~cubic in N (nlive ~ N², fs rides N so steps ~ N, plus an eigsh shift-invert
# factorization of L) — the string's N_MAX is catastrophic in 2D, so the membrane gets its own,
# much lower ceiling and a shorter audio cap, both verified to keep a worst-case local render to a
# few seconds.
# Cost must be bounded by the *actual problem size*, not N alone. Profiled: the cost is pure FDTD
# (eigsh is ~0.06 s, negligible), and it has TWO independent drivers the sliders can push:
#   • per-step cost ∝ n_live, with a razor-sharp ~3.2× CACHE CLIFF at n_live ≈ 10_000
#     (87 µs → 281 µs/step as the working set crosses L2). A thin rectangle reaches it fast:
#     n_live ~ N²·(Ly/Lx), so Lx=0.3, Ly=2.0, N=100 is ~66k nodes. ⇒ keep n_live strictly BELOW
#     the cliff.
#   • step count ∝ fs = c/(λ·h) ∝ 1/min_dimension: a small drum / fine grid inflates steps
#     (the audio is resampled to 48 kHz regardless, so a high sim rate buys no fidelity — pure
#     cost).
# So: an n_live cap below the cache cliff (keeps per-step on the fast side, where time ∝ work),
# and a work-budget cap on n_live × total steps (covers both the audio and the slow-mo animation
# runs).
MEMBRANE_N_MAX = 100
MEMBRANE_NLIVE_MAX = 9_900     # below the n_live≈10_000 L2 cliff; admits a square/disk at N=100
# node-steps budget: square@N=100, 2 s audio ≈ 6.7e8 ≈ ~6 s locally
MEMBRANE_WORK_MAX = 7.0e8
MEMBRANE_AUDIO_MAX = 2.0
MEMBRANE_LAMBDA_MAX = 1.0 / math.sqrt(2.0)   # 2D CFL ceiling (5-point Laplacian: λ ≤ 1/√2)

# Mallet → membrane (model #7): reuses the membrane FDTD + heatmap, so its budget mirrors the
# membrane's — but every step also runs a scalar contact root-find (a near-constant ~20 µs on top
# of the FDTD), so a coupled step is ~1.5-3× a bare membrane step (the factor is largest at low N,
# where the root-find dominates the small grid). Halving the node-step budget keeps the worst
# passing render at the membrane's ~5-6 s wall-clock despite that per-step factor.
MALLET_N_MAX = 80
MALLET_WORK_MAX = 3.5e8
MALLET_AUDIO_MAX = 2.0
MALLET_DIAG_POINTS = 240     # decimated contact-episode trace length for the contact panel

# heatmap display grid is decimated to <= this per axis (data-size trap)
DISPLAY_MAX = 64
N_MEMBRANE_MODES = 12        # discrete eigenmodes marked on the spectrum panel
N_SPEC_POINTS = 420          # decimated magnitude-spectrum length for the plot

# --- Kirchhoff plate (2D, model #5 / #5b) clamps + display budget ---
# The plate is implicit (unconditionally stable for theta >= 1/4), so its time control is the
# *plate Courant number* mu = kappa*k/h^2 (the explicit-scheme stability parameter, reported only):
# fs = kappa/(mu*h^2) rides N^2 (h = Lx/N), so a fine grid inflates the step count exactly like the
# membrane. Same cost drivers (n_live per-step + fs step-count), so it reuses the membrane's cliff /
# work-budget shape with plate-appropriate ceilings.
PLATE_N_MAX = 80
PLATE_NLIVE_MAX = 9_900        # below the n_live~10_000 L2 cliff (shared with the membrane)
PLATE_WORK_MAX = 7.0e8         # n_live x total steps (audio + animation)
PLATE_AUDIO_MAX = 2.0
PLATE_MU_MAX = 32.0            # implicit -> no CFL; large mu is just coarse (dispersive), stable
N_PLATE_MODES = 6             # discrete eigenmodes marked on the spectrum panel

# --- von Karman nonlinear plate (2D, model #6) clamps + Picard-aware budget ---
# The nonlinear plate is the expensive corner: each step Picard-iterates (up to couple_max_iter
# sweeps of an F-solve + an A-solve), so the true problem size is n_live x steps x couple_max_iter.
# Energy conservation holds *only at the Picard fixed point*, so the defaults sit squarely in the
# proven convergent regime (N ~ 24, w/e ~ 1-3, drift ~ 1e-13) and N/amplitude are capped hard.
VK_N_MAX = 32
VK_NLIVE_MAX = 1_600           # ~ (VK_N_MAX+1)^2; a free VK plate is every-node-live
VK_AUDIO_MAX = 1.0
VK_FS_MIN, VK_FS_MAX = 8_000.0, 96_000.0   # oversample around the nonlinearity (HANDOFF §8)
VK_COUPLE_MAX_ITER = 50        # Picard safety cap (also the worst-case cost multiplier)
VK_WORK_MAX = 2.0e9            # n_live x steps x couple_max_iter (Picard-aware)
VK_WOVERE_MAX = 6.0            # strike amplitude in thickness units (w/e); the hardening knob
N_VK_MODES = 6               # linear eigenmodes marked on the spectrum panel

# --- tension-modulated string (model #9) — the Kirchhoff–Carrier hardening demo ---
# The headline is the *amplitude shift* omega(A) - omega(A->0), NOT an absolute frequency. A
# measured omega(A) carries the theta-scheme's linear temporal dispersion error, and omega(A->0)
# carries the same one, so their difference cancels it and isolates the nonlinear physics. That is
# the model's own oracle (test_amplitude_shift_matches_duffing); duffing_frequency alone matches
# only "loosely". Calibrated at the string defaults (L=1, T=200, rho=0.005 -> c=200, f1=100 Hz),
# N=128, EA=1e5:
#   A=0.001 -> shift  0.05 Hz (  0.8 c)  <- the string path's inherited default: a near-null; the
#                                           flagship panel would render blank and look broken
#   A=0.02  -> shift 16.9  Hz (270   c)  <- shipped default: plainly visible, and sub-breakup
#   A=0.06  -> shift 80+   Hz            <- the slider cap; still sub-breakup (see TENSION_DT_MAX)
TENSION_EA_DEFAULT = 1.0e5      # EA/T = 500 at the default T; real strings sit at 150-600
TENSION_EA_MAX = 2.0e5
TENSION_AMP_DEFAULT = 0.02      # amplitude IS this model's independent variable — see the table
TENSION_AMP_MAX = 0.06          # slider bound only; the load-bearing guard is TENSION_DT_MAX below
TENSION_REF_AMP = 1e-5          # FIXED absolute reference amplitude (a ratio of A courts zc noise)
# 12 fundamental periods per measurement run. More does not help: at 6/12/24 periods the shift lands
# 1.47e-3 / 1.17e-3 / 1.03e-3 from the oracle, so the residual is the scheme's genuine O(h²)+O(k²)
# gap to the continuum Duffing, NOT crossing noise. 12 buys the accuracy at half the cost.
TENSION_MEASURE_PERIODS = 12
# Purity gate: sub-threshold single-mode runs sit at off-mode ~1e-11, broken-up ones at >1e-3
# (the core's own signature tests), so any bar in between discriminates without tuning.
TENSION_OFFMODE_MAX = 1e-6

# THE load-bearing guard. Single-mode motion is parametrically unstable above a threshold in
# dT/T0 = EA·A²·p1²/(4T) — this project's own discovery (model #9). Above it the mode disintegrates
# into its neighbours (energy-conserving: it is *physics*, not a blow-up), and the Duffing reduction
# stops describing the motion, so the shift oracle stops meaning anything. The threshold is NOT the
# "~3" of the core's tests — that was measured at mode 3 and is not mode-invariant. Measured here
# for mode 1 over a run 1000 fundamental periods long (3x the longest this model now admits, since
# TENSION_AUDIO_MAX caps exposure at ~300 periods), at N=128:
#   A=0.06 -> dT/T0=4.44 -> worst off-mode 3.2e-13   PURE
#   A=0.07 -> dT/T0=6.05 -> worst off-mode 5.9e-02   BREAKS UP
# so mode 1's threshold lies in (4.44, 6.05]. Also verified PURE at the cap with kappa=0 (off-mode
# 3.2e-13): stiffness detunes the Mathieu resonance, so kappa=0 *looked* like the dangerous corner —
# measured, it moves mode 1's threshold nowhere. Bounding dT/T0 (not amplitude!) keeps the panel
# honest: A alone is a proxy — EA and T move dT/T0 just as hard, and EA=2e5 at A=0.06 would break up
# with an amplitude-only cap none the wiser (the membrane's "bound the actual problem size" lesson).
# Because the cap makes breakup unreachable, the purity gate is a *guarantee*, not dead code.
# The instability deserves a viewer panel of its own (mode cascade, the Mathieu tongue) — deferred.
TENSION_DT_MAX = 4.45
# Cost: the tension solve is a scalar root-find (a banded re-solve per iteration) every step, so a
# step costs ~176 µs at N=128 — about 2x a 2D membrane step, for a 1D string. The string path's
# N_MAX=2000 / 10 s would be ~4M root-finds (minutes of hang), so this model needs its own budget.
# Steps rise with N on their own (fs = c·N/(L·λ)), so a total-step budget already penalizes a fine
# grid. Default (N=128, 1 s) ~ 34k steps ~ 6 s; the budget admits ~60k ~ 11 s as the worst pass.
TENSION_N_MAX = 256
TENSION_AUDIO_MAX = 3.0
TENSION_WORK_MAX = 60_000       # total steps: audio + both measurement runs + animation

# --- bowed string (Phase D batch 2) ---
# Defaults are the core test rig's known-good Helmholtz point (tests/helpers.make_bowed_string):
# beta=0.13, force=1.0, v_bow=0.1, a=60 on a kappa=0 flexible string -> f1 = c/2L = 100 Hz.
# Loss is ON by default and that is load-bearing, the OPPOSITE of the tension string: sigma0 > 0
# lets the note settle to a steady Helmholtz limit cycle instead of growing without bound, and
# sigma1 > 0 damps the high partials so the corner stays clean (one slip per period) rather than
# raucous (~18 slips/period). sigma1's string-path slider max (0.01) is far too small — re-ranged.
BOW_POSITION_DEFAULT = 0.13     # m; beta = x_bow/L, and the slip fraction of the period ~ beta
BOW_V_DEFAULT = 0.1             # m/s
BOW_FORCE_DEFAULT = 1.0         # N (peak of the friction curve)
BOW_SHARPNESS_DEFAULT = 60.0    # s^2/m^2
BOW_SIGMA0_DEFAULT = 0.5
BOW_SIGMA1_DEFAULT = 0.05
BOW_FORCE_MAX = 5.0
BOW_V_MAX = 1.0
BOW_SHARPNESS_MAX = 200.0
BOW_BALANCE_TOL = 1e-11         # lossless |dE - bow_work|/scale bar (tests/test_bow_energy.py)
# Helmholtz-window gating for the slip panel (advisor): slip_fraction == beta is a *verdict* only
# for clean one-slip-per-period motion. Outside Schelleng's window the note goes raucous and the
# match legitimately fails — real physics, not a bug — so the panel reports it as a LABEL, never a
# FAIL (the free-cymbal `null` "crash cascade" precedent).
BOW_SLIPS_LO, BOW_SLIPS_HI = 0.85, 1.25   # same window as test_one_slip_per_period
BOW_SLIP_MATCH_TOL = 0.05                 # same bar as test_slip_fraction_matches_beta
# The statistics tail is the settled last 40% of the run — the core's own choice (_bow_to_steady),
# and it has to be this long for a *rate*. `slips_per_period` is an integer onset count divided by
# the periods in the window, so a short window quantizes it coarsely: over 6 periods it can only
# land on multiples of 1/6, and clean one-slip motion reads 0.83 or 1.0 depending on where the
# window edge falls — straddling the 0.85 gate and mislabelling a perfectly clean note as raucous.
# Over a 40% tail (~40 periods at the default) the quantization is ~0.025.
BOW_TAIL_FRAC = 0.4
BOW_TRACE_PERIODS = 3                     # periods *drawn* — a plot, not the statistics window
# Cost: like the tension string, every step is a scalar Newton root-find (plus a rank-1 correction),
# so the string path's N_MAX=2000 / 10 s budget would hang. Same shape of budget, same reasoning.
BOW_N_MAX = 256
BOW_AUDIO_MAX = 3.0
BOW_WORK_MAX = 60_000           # total steps: audio + animation

# == geometrically-exact string (model #10) ========================================================
#
# Four regimes, one per claim; the secondary select carries them (see GEOM_REGIMES).
GEOM_REGIMES = ("planar", "rotating", "whirl", "phantom")
# lam_long = c_long k / h IS this model's trap, and nothing enforces it: the theta-scheme is
# unconditionally stable, so an unresolved longitudinal wave returns quiet nonsense with no CFL
# error at all. The core WARNS above 1 (LAM_LONG_WARN) rather than rejecting — the scheme really is
# stable there — but the viewer must never render the headline in the warned regime, so here it is
# a hard cap. It is also the whole cost story: c_long/c = sqrt(EA/T) ~ 22 for a real string, so
# resolving it forces fs ~ 22x a transverse-only model's.
GEOM_LAM_LONG_MAX = 1.0
GEOM_LAM_LONG_DEFAULT = 0.9
GEOM_N_MAX = 32
# Every step is a vector Newton solve over 3 coupled fields: ~2 ms at small amplitude, ~4 ms on the
# whirl (a bigger amplitude buys more iterations). The default whirl is 4,770 steps ~ 19 s, so this
# ceiling is ~25 s — by far the slowest model in the viewer, and irreducibly so (see the header
# comment on _build_payload_geometric). Bound the STEPS, not N: N and the window and lam_long all
# move the step count, and only the product is the cost (the membrane's lesson).
GEOM_WORK_MAX = 6_000
# The whirl's amplitude is set by dT/T0, never by A directly — A is a proxy, and EA and T move the
# tension excess just as hard (the tension string's lesson). kappa_u = 0 makes eps A^2/omega_u^2 ==
# dT/T0 EXACTLY, so this is model #9's own planar-breakup coordinate: the whirl rig's 1.5 is a
# measured half-margin to #9's ~3, and 2.16 is measured good (off-mode 0.2%).
GEOM_DT_DEFAULT = 1.5
GEOM_DT_MAX = 2.2
# The Mathieu tongue's dimensionless coordinate, frac = delta / (eps A^2), where delta = w_w^2 -
# w_u^2 is the detuning the kappa_w knob buys. Unstable <=> 0 < frac < 1/2, peak at 1/4. THE control
# of this model: the measured map is 1.00 -> 14.7 -> 76.3 -> 37.4 -> 8.4 -> 1.63x growth at
# frac = 0 / .07 / .25 / .41 / .5 / .8. The upper edge is SOFT (leading-order eps), so the slider
# runs past 1/2 on purpose — watching the growth die is the point.
GEOM_TONGUE_DEFAULT = 0.25
GEOM_TONGUE_MAX = 1.0
# The out-of-plane seed, as a fraction of the driven amplitude. Kept small: it must stay a
# perturbation, or the linearised tongue is not the right oracle for what the run does.
#
# BOTH kinds are offered, and the difference between them IS a claim of this model — measured here,
# because the received rule ("a velocity seed, never a displacement one: a displaced w is the
# rotation generator, so it pins growth at 1.00x and would draw a line even inside the tongue") is
# only half right, and the wrong half is the memorable half. Measured growth at t = 0.06 s:
#
#     frac        0       0.07     0.25      0.5      0.8
#     disp     1.00x    14.69x   60.17x    6.08x    1.17x
#     vel      6.88x    28.52x   63.00x    0.85x    0.78x
#
# Inside the tongue a DISPLACED seed grows perfectly well (60x at the peak) — the pinning happens
# only at frac = 0, on the DEGENERATE string, where dw = dA phi at rest is exactly the rotation
# generator and the run is just the same planar motion in a rotated plane (Tier A/1 restated). So:
#   * displacement (default) reads the TONGUE cleanly — 1.00x at frac=0 is "a degenerate string
#     cannot whirl", the claim, and the map peaks at 1/4 and dies by 0.8 exactly as predicted.
#   * velocity injects angular momentum, so the degenerate string grows SECULARLY (linearly, 6.88x)
#     rather than exponentially. That is marginality, not whirling — and the log-y envelope panel
#     shows the difference free: secular growth BENDS on log axes, exponential growth is straight.
GEOM_SEED_FRAC = 1e-3
GEOM_AMP_DEFAULT = 4e-3         # planar / rotating: small enough that the BVP continuation is quick
GEOM_ROTATING_PERIODS = 2.0     # revolutions drawn — the circle closes on itself
GEOM_PLANAR_WINDOW = 0.02       # s; the line needs no growth, so it needs no length
GEOM_ORBIT_POINTS = 1_500       # (u, w) trail points shipped — 2 floats each, cheap
GEOM_ENV_POINTS = 400           # decimated whirl-envelope length for the log-y plot
GEOM_GROWTH_FRAC = 8            # growth = max|w| over the last 1/8 of the run / over the first 1/8

# -- the phantom regime ----------------------------------------------------------------------------
#
# Model #9's FIRST refusal, discharged: a scalar tension has nowhere to put a combination tone, so
# the phantom is the sharpest statement of what it means for the tension to be a FIELD. The readout
# is the bridge force EA v_x(0) — what actually radiates in a piano, and the honest place to look:
# below the first longitudinal resonance the v response is quasi-static, so EA v_x carries r^2
# almost directly and the combination tones are in it by construction rather than by luck.
#
# This regime reproduces tests/test_geometric_phantom.py's rig EXACTLY (same N / lam_long / window /
# two-mode IC / v = 0 start / blind band-limited detector). That is deliberate and load-bearing: the
# viewer then inherits the suite's validation for free. In particular v is NOT initialised at its
# quasi-static equilibrium — a held string really has settled, so v = 0 radiates a startup transient
# that is the LARGEST feature in the full bridge spectrum, and pre-solving it away is tempting. It
# would put a number on screen that no test backs, and it buys nothing: the free longitudinal modes
# sit at n c_long/(2L) ~ 2236 Hz while the phantoms live below 500 Hz, so the transient is 4.5x
# above the band and the panel never shows it. That separation is why the rig band-limits, and it
# is why the phantom band is PURELY forced response (which is also the piano physics).
GEOM_PHANTOM_WINDOW = 0.10
"""Seconds of bridge force measured. **Not a slider, and not padding** — the test rig's own window.

Halving it does not merely cost precision. At 0.05 s the raw bins are 20 Hz wide, the ``2 f1``
phantom (the weakest of the four) sits inside its neighbours' leakage skirts, and it is mislocated
by **0.52 Hz** — against 0.013-0.16 Hz for the other three, and 0.039 Hz for the worst of the four
at 0.1 s. Its margin to ``f2`` collapses from 170x to 8x with it. So the window is physics, and the
`animation_window` slider is ignored (and hidden) in this regime.
"""
GEOM_PHANTOM_AMP_DEFAULT = 1.5e-3
GEOM_PHANTOM_KAPPA_DEFAULT = 8.0
"""Stiffness for the phantom regime — **4x the piano-ish default**, and the choice is load-bearing.

A phantom is discriminating only because it lands where no partial is. On a harmonic string (B = 0)
every phantom coincides with a partial *exactly* — ``f2 - f1 = f1``, ``2 f1 = f2``, ``f1 + f2 = f3``
— so there is nothing to see. Inharmonicity ``B = pi^2 kappa^2/(c^2 L^2)`` opens the gaps in
proportion to it. This exaggerates the CONTRAST, not the EFFECT: the mechanism is ``r^2`` pumping
``v``, which is completely kappa-independent — kappa only decides whether the gap a phantom lands
in is visible. A microscope, not a thumb on the scale.

At the kappa = 2 default the gap is 0.89 Hz, which does not merely blur — it makes the claim
**wrong**: ``f1``/``f2`` are measured from this run so they are the *hardened* ones, and hardening
drives the phantom at ``f1 + f2`` up by a measured 1.29 Hz, which EXCEEDS the gap. The phantom
crosses ``f3`` and the panel would confidently report a phantom landing on a partial. Hence the
kappa slider is live here (sliding it down and watching the discrimination die is the point) but
the verdict is GATED on the measured defect below.
"""
GEOM_PHANTOM_DEFECT_MIN = 3.0
"""Hz of defect ``f2 - 2 f1`` below which the panel LABELS instead of scoring.

The tests' own floor, and ~77x the 0.039 Hz peak-location error measured at this window — so above
it the displacement is real, and below it the phantoms have collapsed onto the partials and there is
no discrimination left to claim. Label-not-fail, the bow's Schelleng-window precedent: a harmonic
string genuinely has no phantom signature, which is physics rather than a solver failure.

**The gate is one-sided (``>=``, not ``abs``), and that is load-bearing — measured here, and it is
NOT what the name "inharmonicity defect" suggests.** ``f2 - 2 f1`` is not pure stiffness: the
theta-scheme's temporal dispersion drags mode 2 *flat*, so it contributes a NEGATIVE defect, and
what the panel measures is the difference of the two. Off the linear discrete ladder at
``lam_long = 0.9``::

    N        kappa = 0     kappa = 2     kappa = 8
    16        -0.965        -0.677        +3.571
    24        -0.430        -0.137        +4.168
    32        -0.242        +0.052        +4.377

At ``kappa = 0`` the defect is pure numerical dispersion and converges as O(h^2) (0.965/0.242 = 4.0
across a 2x refinement, exactly). Two consequences. First, the ``kappa = 2`` trap is *worse* than
recorded: at N=32 dispersion very nearly CANCELS the true inharmonicity (+0.05 Hz net), and by N=16
it overwhelms it and the defect goes negative. Second, a coarse-enough grid gives a large *negative*
defect — phantoms displaced to the wrong side of the partials by an artifact rather than by physics
— which an ``abs()`` gate would happily score. Requiring a positive defect means "the partials are
stretched by real stiffness, by enough to see", which is the only version of this claim worth
making. (The measured in-run defect at N=32/kappa=8 is 4.574 vs the ladder's 4.377: the extra ~0.2
is the amplitude hardening, which *widens* the defect and so works against the claim.)

Scale caveat: 3.0 Hz is absolute, so a longer/slacker string (lower f0) trips it at a kappa that
would have been fine — that under-claims, which is the safe direction for a gate that only labels.
"""
GEOM_PHANTOM_BAND = 4.8
"""Display band, as a multiple of ``f1`` — **the trap the diagnose rig had to learn the hard way**.

On a 0-2 kHz axis (out to the first longitudinal mode) the whole claim is sub-pixel: ``f1 + f2`` is
11.4 Hz from the 3rd partial, the two marker families sit on top of each other, and the figure shows
the OPPOSITE of what it says. 4.8 f1 is the tightest band that still contains all four combinations
(``2 f2`` is the highest). The discrimination is the picture; frame it.
"""
GEOM_PHANTOM_ZOOM_HZ = 26.0
"""Half-width (Hz) of the *second*, zoomed strip — the same trap one level down.

The full band is ~485 Hz across a 430 px canvas, so the 4.6 Hz defect is ~4 px: on the wide strip
the difference tone reads as sitting ON f1, which is half the claim rendered backwards. Measured
combo-vs-ladder separations across the wide strip are 4 / 3 / 9 / 25 px — only the top two read.
The wide strip carries "peaks at the four combinations"; a zoom around the f1 / (f2-f1) pair carries
"and NOT on the partial" (4.56 Hz over a 56.6 Hz axis is ~31 px — plainly two lines). Both halves
need a picture, so the panel draws both.
"""
GEOM_PHANTOM_DISPLAY_PAD = 32
"""Zero-pad factor for the **drawn traces only** — the detector keeps the rig's default.

At the rig's 2x pad the bins are 4.85 Hz wide, so the 56.6 Hz zoom strip gets **12 points**: not a
picture. Zero-padding densifies the bin grid without adding real resolution — exactly what a *plot*
needs and nothing a *measurement* may lean on. So it is confined to the display path: `detect_peaks`
and the peak-magnitude lookup keep `magnitude_spectrum`'s 2x default, which is what the test rig
measures (and where the 0.039 Hz peak-location error comes from — parabolic refinement, independent
of the display grid). At 32x the zoom strip gets ~190 points and the wide strip ~1600 pre-pooling.
"""
GEOM_PHANTOM_WORK_MAX = 16_500
"""Steps. Its own budget: **~45 s** at the default, well past GEOM_WORK_MAX's ~25 s.

Measured, not extrapolated: the bare Newton loop is 2.2 ms/step x 15,900 = 35.6 s, and the panel
telemetry (energy every step, the two modal projections, the field maxima) carries it to ~45 s. The
window is fixed physics (see GEOM_PHANTOM_WINDOW) and fs rides N and lam_long, so this is really a
cap on N x lam_long at the fixed window. **It is by far the slowest render in the viewer** — the
headless verifier's wait had to grow for it.
"""

# == sympathetic / coupled strings (SympatheticStrings, model in core/connection.py) ===============
#
# Several fixed/free strings share ONE bridge point on a common ModalBody (piano unisons, sitar
# sympathetics; HANDOFF §12.B). The viewer fixes **J = 2** — the two-string oracles (the
# antisymmetric normal mode, sympathetic transfer) are exactly what is validated, and a free
# string-count slider would break them. It is a CLOSED, UNDRIVEN, linear-leapfrog system, so it
# rides the ORDINARY energy panel: sigma = 0 -> the conservation drift check; a lossy BODY (the
# weinreich regime below) flips it to passivity. But conservation and passivity are AUTOMATIC from
# the linear-leapfrog structure — they pass even a flipped coupling sign — so they are table-stakes,
# NOT the claim. The claim lives in the SECOND panel. Three regimes carry it:
#   * normal    — the discriminating oracle. Runs BOTH ICs (antisymmetric u_B = -u_A, keeping w_b
#                 exactly still, vs the symmetric contrast that swings the bridge) and plots both
#                 w_b traces; a flat zero alone would read as "broken". The bit-exact w_b == 0 holds
#                 only if the two strings are identical, so the detune slider is gated OUT of here.
#   * transfer  — pluck one string; a tuned unison neighbour drains most of the energy, a detuned
#                 one barely responds. The per-string energy-fraction panel carries the slosh.
#   * weinreich — the piano-unison two-stage decay. Two near-unison strings over a LOSSY bridge
#                 (a body-loss slider, the first in the viewer); strike ONE and the symmetric normal
#                 mode loads the lossy bridge and dies fast (the "prompt"), while the antisymmetric
#                 mode barely loads it and lingers (the "aftersound"). The string-energy envelope
#                 shows the fast-then-slow knee. At detune = 0 the antisymmetric mode is bit-exactly
#                 bridge-decoupled (the normal-mode oracle again), so the body's damping never
#                 activates on it and the aftersound is EXACTLY lossless (slope -> 0) — the sharp
#                 sub-claim; a nonzero detune gives the realistic finite aftersound. Its contrast is
#                 plucking BOTH strings (the pure symmetric mode): single-slope, no aftersound.
SYMP_REGIMES = ("normal", "transfer", "weinreich")
SYMP_J = 2                       # fixed: the validated oracles are two-string
SYMP_N_MAX = 160                 # each step = 2 ideal-string leapfrogs + a body step (no root-find)
SYMP_LAM_DEFAULT = 0.9           # < 1 REQUIRED: the bridge spring pushes the Nyquist mode unstable
SYMP_K_DEFAULT = 8000.0          # normal-mode bridge (the diagnose rig's stiff K)
SYMP_K_TRANSFER = 1500.0         # transfer bridge: a SOFTER spring is frequency-SELECTIVE (the
#                                  resonant transfer is the point; a stiff one couples broadly)
SYMP_K_WEINREICH = 6000.0        # two-stage-decay bridge (the diagnose rig's lossy-bridge K)
SYMP_SIGMA_BODY_DEFAULT = 20.0   # body-loss default: a visible prompt + a long aftersound in ~2 s
SYMP_SIGMA_BODY_MAX = 80.0       # heavier only shortens the prompt to an invisible cliff
SYMP_WEINREICH_DETUNE_MAX = 0.4  # semitones (~40 cents): a piano unison is mistuned by a few cents,
#                                  so weinreich needs a FINE detune range, distinct from transfer's
#                                  0..12 semitones (its slider step is 0.01 semi ~ 1 cent)
SYMP_BODY_FREQS = np.array([137.0, 213.0, 330.0, 471.0, 620.0])  # diagnose rig's off-harmonic modes
SYMP_BODY_MASS = 0.02            # ~ the string's rho*L, so the body genuinely reacts
SYMP_AMP = 1e-3                  # pluck amplitude; the bridge-stillness claim is scale-invariant
SYMP_NORMAL_ANIM_WIN = 0.06      # s of animation for the normal mode (a standing oscillation)
SYMP_AUDIO_MAX = 3.0
SYMP_TRACE_POINTS = 600          # decimated length of the w_b / energy-fraction traces
# The w_b trace window is DECOUPLED from the audio run (catch #2 again). The bridge swings at
# the string fundamental AND the body modes (up to ~620 Hz); index-decimating a full 2 s / 44k-step
# run to 600 points sits their frequencies above the decimation's Nyquist, so the symmetric trace
# aliases into a solid full-scale BAND and the flat antisymmetric oracle line is muddied where
# they cross. A short window resolves the individual swings — "swings vs dead-flat" is the picture,
# not "a nonzero band vs zero". The full run still feeds the audio + the energy drift.
SYMP_TRACE_WINDOW = 0.10         # s of w_b actually plotted in the normal-mode panel
# Total steps. The normal regime runs TWICE (antisymmetric + its symmetric contrast, both needed for
# the panel); transfer runs ONCE over the full slosh. fs = c0 N/(L lam) rides c0 = sqrt(T/rho), so a
# high tension inflates the step count — bound N x T x duration, not N alone (the membrane's
# "bound the actual problem size" lesson). At the defaults (T=200, N=100, lam=0.9, 2 s) transfer is
# ~44k steps, normal ~89k; each step is cheap (ideal-string leapfrogs, no root-find), so this budget
# is really a frame/payload guard rather than a wall-clock one.
SYMP_WORK_MAX = 130_000


class ParamError(ValueError):
    """A bad request parameter (out of the allowed range) — surfaced as a clean error payload."""


# -- small helpers ---------------------------------------------------------------------------------


def _b64f32(arr: NDArray[np.float64]) -> str:
    """Pack a numeric array as base64 little-endian float32 (10x smaller than JSON numbers)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def _b64u8(arr: NDArray[np.bool_] | NDArray[np.uint8]) -> str:
    """Pack a small integer/boolean array as base64 uint8 (e.g. the decimated domain mask)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8))
    return base64.b64encode(a.tobytes()).decode("ascii")


def _finite_list(arr: NDArray[np.float64], ndigits: int | None = None) -> list[float | None]:
    """Array -> JSON-safe list, mapping non-finite values to ``None`` (JSON has no NaN)."""
    out: list[float | None] = []
    for v in np.asarray(arr, dtype=float).ravel():
        if not math.isfinite(v):
            out.append(None)
        else:
            out.append(round(float(v), ndigits) if ndigits is not None else float(v))
    return out


def _fnum(p: dict[str, Any], key: str, default: float) -> float:
    """Read a float param, raising :class:`ParamError` (not a bare ValueError) on garbage."""
    try:
        return float(p.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"{key!r} must be a number, got {p.get(key)!r}.") from exc


# -- model dispatch (catch #5) ---------------------------------------------------------------------


class _Built:
    """Bundle returned by :func:`_build_resonator` (the resonator + derived scalars it needs)."""

    def __init__(self, res: Any, c: float, L: float, N: int, fs: float,
                 sigma_zero: bool, oracle_2sigma: float) -> None:
        self.res = res
        self.c = c
        self.L = L
        self.N = N
        self.fs = fs
        self.sigma_zero = sigma_zero
        self.oracle_2sigma = oracle_2sigma


def _build_resonator(p: dict[str, Any]) -> _Built:
    """Construct a fresh resonator from params. Raises ParamError / ValueError (caught upstream).

    ``fs_sim = c*N/(L*lambda)`` so the requested ``lambda`` is reproduced exactly; the core ctor is
    the single source of truth for the CFL / non-physical guards (it raises ValueError, surfaced
    upstream as a clean error payload).
    """
    model = str(p.get("model", "ideal"))
    L = _fnum(p, "L", 1.0)
    T = _fnum(p, "T", 200.0)
    rho = _fnum(p, "rho", 0.005)
    lam = _fnum(p, "lambda", 1.0)
    try:
        N = int(p.get("N", 128))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {N_MAX}], got {N}.")
    if min(L, T, rho) <= 0:
        raise ParamError("L, T, rho must all be positive.")
    if lam <= 0:
        raise ParamError(f"lambda must be > 0, got {lam}.")

    c = math.sqrt(T / rho)
    fs = c * N / (L * lam)
    common = {"L": L, "T": T, "rho": rho, "fs": fs, "N": N}

    if model == "ideal":
        sigma = _fnum(p, "sigma", 0.0)
        res = IdealString(**common, boundary="fixed", sigma=sigma)
        return _Built(res, c, L, N, fs, sigma == 0.0, 2.0 * sigma)
    if model == "stiff":
        sigma = _fnum(p, "sigma", 0.0)
        res = StiffString(
            **common, kappa=_fnum(p, "kappa", 0.0), sigma=sigma, theta=_fnum(p, "theta", 0.28)
        )
        return _Built(res, c, L, N, fs, sigma == 0.0, 2.0 * sigma)
    if model == "damped":
        s0 = _fnum(p, "sigma0", 0.0)
        s1 = _fnum(p, "sigma1", 0.0)
        res = DampedStiffString(
            **common, kappa=_fnum(p, "kappa", 0.0), sigma0=s0, sigma1=s1,
            theta=_fnum(p, "theta", 0.28),
        )
        # oracle_2sigma is the frequency-INDEPENDENT base rate (sigma1 adds a per-mode term on top).
        return _Built(res, c, L, N, fs, s0 == 0.0 and s1 == 0.0, 2.0 * s0)
    if model == "tension":
        s0 = _fnum(p, "sigma0", 0.0)
        s1 = _fnum(p, "sigma1", 0.0)
        EA = _fnum(p, "EA", TENSION_EA_DEFAULT)
        if not (0.0 <= EA <= TENSION_EA_MAX):
            raise ParamError(f"EA must be in [0, {TENSION_EA_MAX:.0f}] N, got {EA}.")
        res = TensionModulatedString(
            **common, kappa=_fnum(p, "kappa", 0.0), EA=EA, sigma0=s0, sigma1=s1,
            theta=_fnum(p, "theta", 0.28),
        )
        return _Built(res, c, L, N, fs, s0 == 0.0 and s1 == 0.0, 2.0 * s0)
    if model == "bow":
        s0 = _fnum(p, "sigma0", BOW_SIGMA0_DEFAULT)
        s1 = _fnum(p, "sigma1", BOW_SIGMA1_DEFAULT)
        force = _fnum(p, "force", BOW_FORCE_DEFAULT)
        v_bow = _fnum(p, "v_bow", BOW_V_DEFAULT)
        sharp = _fnum(p, "sharpness", BOW_SHARPNESS_DEFAULT)
        if not (0.0 <= force <= BOW_FORCE_MAX):
            raise ParamError(f"force must be in [0, {BOW_FORCE_MAX}] N, got {force}.")
        if not (0.0 < v_bow <= BOW_V_MAX):
            raise ParamError(f"v_bow must be in (0, {BOW_V_MAX}] m/s, got {v_bow}.")
        if not (0.0 < sharp <= BOW_SHARPNESS_MAX):
            raise ParamError(f"sharpness must be in (0, {BOW_SHARPNESS_MAX}], got {sharp}.")
        string = DampedStiffString(
            **common, kappa=_fnum(p, "kappa", 0.0), sigma0=s0, sigma1=s1,
            theta=_fnum(p, "theta", 0.28),
        )
        res = BowedString(
            string=string, bow_position=_fnum(p, "bow_position", BOW_POSITION_DEFAULT),
            v_bow=v_bow, force=force, sharpness=sharp,
        )
        return _Built(res, c, L, N, fs, s0 == 0.0 and s1 == 0.0, 2.0 * s0)
    raise ParamError(
        f"unknown model {model!r} (expected 'ideal' | 'stiff' | 'damped' | 'tension' | 'bow')."
    )


def _partials_oracle(model: str, p: dict[str, Any], c: float, L: float) -> NDArray[np.float64]:
    """Per-model continuum partials oracle (catch #5)."""
    if model in ("stiff", "damped"):
        return modal.stiff_harmonic_frequencies(c, L, _fnum(p, "kappa", 0.0), N_PARTIALS)
    return modal.harmonic_frequencies(c, L, N_PARTIALS)


# -- sub-blocks ------------------------------------------------------------------------------------


def _resample_normalize(x: NDArray[np.float64], fs_in: float) -> tuple[NDArray[np.float64], float]:
    """Resample the pickup to :data:`AUDIO_FS` (catch #1) and peak-normalize. Returns (audio, peak).

    ``peak`` is the raw physical displacement amplitude (metres), pre-normalization, so the loudness
    story stays physical even though the audio is scaled to ~0.9 full-scale.
    """
    x = np.asarray(x, dtype=float)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if x.size > 1 and abs(fs_in - AUDIO_FS) > 1e-6:
        frac = Fraction(int(round(AUDIO_FS)), max(1, int(round(fs_in)))).limit_denominator(2000)
        if frac.numerator > 0 and frac.denominator > 0:
            x = resample_poly(x, frac.numerator, frac.denominator)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 0:
        x = 0.9 * x / m
    return x, peak


def _balance_verdict(
    E: NDArray[np.float64], w: NDArray[np.float64], sigma_zero: bool, idx: NDArray[np.int_],
    measured_loss: dict[str, NDArray[np.float64]] | None = None,
) -> dict[str, Any]:
    """The energy-BALANCE verdict for an *actively driven* model: ``E - E0 == work_in - loss``.

    A third verdict type beside the sigma=0 drift check and the sigma>0 passivity check — and it
    *replaces* both, because for a driven model neither is merely weaker, both are actively wrong:
    at sigma=0 the bow pumps energy in, so ``energy_drift`` is enormous by design and the lossless
    branch would report a catastrophic FAIL; at sigma>0 the energy *rises* from rest to the
    Helmholtz limit cycle, so the monotone/passivity branch would FAIL too. Either would read as
    "implementation broken" when nothing is.

    The verdict is itself sigma-gated, and the lossy branch is deliberately NOT a residual:

    - ``sigma == 0``: nothing dissipates, so the balance must close exactly. The headline is
      ``max|(E - E0) - w| / (|E| + |w|)`` against :data:`BOW_BALANCE_TOL` — the money number
      (~6e-15). Same normalization as ``test_bow_energy.test_lossless_energy_balance`` (per-sample
      scale, max over steps) computed on the **full** per-step arrays: taking the max over the
      decimated arrays would sample fewer steps and understate it.
    - ``sigma > 0``: the string's dissipation is never measured separately — it is *inferred* as
      the residual ``w - (E - E0)``. A "balance residual" here would therefore be identically zero
      BY CONSTRUCTION: a tautology, a green tick that cannot fail. The honest content is the core's
      own criterion 2 (``test_bow_energy.test_loss_only_removes_energy``): the inferred dissipation
      must be ``>= 0`` (loss never *adds* energy) and monotone non-decreasing (no step adds energy).

    ``measured_loss`` (the reed) switches the verdict from *inferred* to **measured** dissipation
    and drops the sigma gate entirely — the two go together, and getting either half wrong is a live
    trap. The reed measures every channel independently (``jet_loss = sum k dp_bar U_B``,
    ``reed_damp_work = sum k Mr g y'^2``, with the bell's radiation booked inside ``E``), so:

    - the residual ``dE - (mouth_work - sum(measured_loss))`` is a number that can genuinely FAIL,
      unlike the bow's lossy branch where dissipation is defined as ``w - dE`` and the residual is
      therefore identically zero by construction — a green tick that cannot fail;
    - and it is genuine in EVERY regime, so there is no sigma gate. This is the trap: the bow's
      ``sigma_zero`` branch scores ``max|dE - w|/scale``, which is only valid because a lossless
      string has ``dE == w``. The reed's jet and lip-damping channels are on even at bore-sigma = 0,
      so ``dE - mouth_work == -(jet + damp)`` — roughly 60 % of the scale. Routing the reed through
      the bow's branch would report a catastrophic "IMBALANCE" on a perfectly balanced model.
    """
    dE = E - E[0]
    if measured_loss is not None:
        loss_total = np.zeros_like(dE)
        for arr in measured_loss.values():
            loss_total = loss_total + np.asarray(arr, dtype=float)
        scale = np.abs(E) + np.abs(w) + 1e-30
        residual = float(np.max(np.abs(dE - (w - loss_total)) / scale))
        # The bow-style residual — measured loss DROPPED — is reported beside it on purpose. It is
        # large (the jet+damp fraction), and that is the evidence the channels are load-bearing:
        # a residual that stays tiny with a summand removed would not be testing the summand.
        naive = float(np.max(np.abs(dE - w) / scale))
        return {
            "work": _finite_list(w[idx]),
            "delta_energy": _finite_list(dE[idx]),
            "dissipation": _finite_list(loss_total[idx]),
            "channels": {k: _finite_list(np.asarray(v, dtype=float)[idx])
                         for k, v in measured_loss.items()},
            "channel_totals": {k: float(np.asarray(v, dtype=float)[-1])
                               for k, v in measured_loss.items()},
            "work_total": float(w[-1]),
            "measured": {
                "residual": residual,
                "naive_residual": naive,
                "tol": BOW_BALANCE_TOL,
                "pass": bool(residual < BOW_BALANCE_TOL),
            },
        }
    dissipation = w - dE                       # inferred, not measured — see above
    block: dict[str, Any] = {
        "work": _finite_list(w[idx]),
        "delta_energy": _finite_list(dE[idx]),
        "dissipation": _finite_list(dissipation[idx]),
        "work_total": float(w[-1]),
    }
    if sigma_zero:
        scale = np.abs(E) + np.abs(w) + 1e-30
        residual = float(np.max(np.abs(dE - w) / scale))
        block["lossless"] = {
            "residual": residual,
            "tol": BOW_BALANCE_TOL,
            "pass": bool(residual < BOW_BALANCE_TOL),
        }
        return block
    scale_w = abs(float(w[-1])) + 1.0
    d_step = np.diff(dissipation)
    block["lossy"] = {
        "dissipation_total": float(dissipation[-1]),
        "non_negative": bool(float(dissipation[-1]) >= -BOW_BALANCE_TOL * scale_w),
        "monotone": bool(float(np.min(d_step)) >= -1e-9 * scale_w) if d_step.size else True,
        "worst_step": float(np.min(d_step)) if d_step.size else 0.0,
    }
    block["lossy"]["pass"] = bool(block["lossy"]["non_negative"] and block["lossy"]["monotone"])
    return block


def _energy_block(
    res: SimResult,
    sigma_zero: bool,
    oracle_2sigma: float,
    convergence: dict[str, Any] | None = None,
    balance_work: NDArray[np.float64] | None = None,
    decay_oracle: bool = True,
    split: dict[str, NDArray[np.float64]] | None = None,
    measured_loss: dict[str, NDArray[np.float64]] | None = None,
) -> dict[str, Any]:
    """Energy report, gated by loss (catch #4): drift-vs-tol when lossless, passivity when lossy.

    ``convergence`` (von Kármán only) folds in the Picard convergence gate: the energy identity
    telescopes *only at the fixed point*, so a run with any non-converged step has a drift number
    that is iteration noise, not physics. When supplied, the lossless verdict additionally requires
    every step to have converged, and the block carries a ``convergence`` sub-block so the frontend
    can say "did not converge — verdict N/A" instead of a lying pass/fail.

    ``balance_work`` (driven models — the bow) switches the block to the **balance** verdict, which
    replaces both branches above; see :func:`_balance_verdict`. It is the cumulative exciter work
    per step, so it is decimated on the *same* ``idx`` as the energy trace and the three curves
    stay aligned. Note there is deliberately no convergence gate on the balance: the bow applies
    its friction force exactly and reads the power from the true post-correction velocity, so the
    balance is exact for *any* Newton residual — that is the model's whole trick, and copying von
    Kármán's Picard gate here would gate on something that cannot spoil the number.

    ``decay_oracle=False`` drops the measured-vs-oracle ``2σ`` line from the lossy verdict, leaving
    pure **passivity** (monotone non-increasing). Used by the mallet: it is a *closed* system whose
    total energy includes the mallet's kinetic energy, and once the felt separates the mallet flies
    off force-free at ~constant velocity (restitution ≈ 1 — a point mass barely couples to the
    head). So E(t) sits on a near-constant floor of ``½M·v₀²`` with only a ~0.01 % membrane wiggle,
    and ``_fit_decay`` over that near-flat trace reports ``measured_2σ ≈ 0`` against a nonzero
    ``oracle_2σ`` — a lying "broken match" over physics that is actually fine. There is no closed
    form for mallet + hysteresis + membrane decay, so passivity *is* the honest lossy verdict.

    ``split`` (the bore) carries named CHANNELS of the same total — ``{"acoustic": ...,
    "radiated": ...}`` — decimated on the same ``idx`` as the total so all three curves stay
    aligned (the ``balance_work`` precedent). It changes no verdict: the bore's bell is a *booked*
    channel, so ``res.energy`` is already the conserving sum and the lossless branch fires on it
    unchanged. The split exists because a flat "conserved ✓" beside an audibly decaying pickup
    reads as a bug — the panel needs acoustic falling and radiated rising to make the flat sum
    legible.

    ``measured_loss`` (the reed) upgrades the balance from the bow's *inferred* dissipation to
    independently **measured** channels and drops the sigma gate; see :func:`_balance_verdict`.
    """
    t, E = res.time, res.energy
    idx = np.linspace(0, len(E) - 1, min(len(E), N_ENERGY_POINTS)).astype(int)
    block: dict[str, Any] = {
        "sigma_is_zero": sigma_zero,
        "time": _finite_list(t[idx], 6),
        "value": _finite_list(E[idx]),
    }
    if split is not None:
        block["split"] = {k: _finite_list(np.asarray(v, dtype=float)[idx])
                          for k, v in split.items()}
    if convergence is not None:
        block["convergence"] = convergence
    if balance_work is not None:
        block["kind"] = "balance"
        block["balance"] = _balance_verdict(E, np.asarray(balance_work, dtype=float), sigma_zero,
                                            idx, measured_loss=measured_loss)
        return block

    if sigma_zero:
        drift = res.energy_drift
        converged = convergence["all_converged"] if convergence is not None else True
        block["lossless"] = {
            "drift": drift, "tol": LOSSLESS_TOL,
            "pass": bool(drift < LOSSLESS_TOL and converged),
        }
        return block

    E0 = float(E[0])
    dE = np.diff(E)
    max_rel_inc = float(np.max(dE) / E0) if E0 > 0 else float(np.max(dE)) if dE.size else 0.0
    lossy: dict[str, Any] = {
        "monotone": bool(max_rel_inc <= MONOTONE_TOL),
        "max_rel_increase": max_rel_inc,
    }
    if decay_oracle:
        lossy["measured_2sigma"] = _fit_decay(t, E)
        lossy["oracle_2sigma"] = oracle_2sigma
    block["lossy"] = lossy
    return block


def _fit_decay(t: NDArray[np.float64], E: NDArray[np.float64]) -> float | None:
    """Slope of log(E) vs t over the non-tiny portion = ``2 sigma_eff`` (E ~ e^{-2 sigma t})."""
    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    mask = E > E[0] * 1e-6
    if mask.sum() < 2:
        return None
    slope = float(np.polyfit(t[mask], np.log(E[mask]), 1)[0])
    return -slope


def _partials_block(
    pickup: NDArray[np.float64], fs: float, model: str, p: dict[str, Any], c: float, L: float
) -> dict[str, Any] | None:
    """Detected-vs-analytic partials (cents error), using the full-rate pickup for accuracy."""
    analytic = _partials_oracle(model, p, c, L)
    analytic = analytic[analytic < 0.95 * (fs / 2.0)]
    if analytic.size == 0:
        return None
    detected = spectrum.measure_partials_near(pickup, fs, analytic)
    cents = modal.cents(detected, analytic)
    return {
        "analytic": _finite_list(analytic, 4),
        "detected": _finite_list(detected, 4),
        "cents": _finite_list(cents, 4),
    }


# -- main entry point ------------------------------------------------------------------------------


def simulate_to_payload(params: dict[str, Any]) -> dict[str, Any]:
    """params dict -> JSON-able payload. Never raises: bad input -> ``{"error": {...}}``."""
    try:
        return _build_payload(params or {})
    except ParamError as exc:
        return {"error": {"kind": "param", "message": str(exc)}}
    except ValueError as exc:  # core construction guards (CFL violated, non-physical params)
        return {"error": {"kind": "construction", "message": str(exc)}}


def _build_payload(p: dict[str, Any]) -> dict[str, Any]:
    """Dispatch on model: the 1D string family, the 2D membrane, or the 2D plate family."""
    model = str(p.get("model", "ideal"))
    if model == "membrane":
        return _build_payload_membrane(p)
    if model == "mallet":
        return _build_payload_mallet(p)
    if model == "plate":
        return _build_payload_plate(p)
    if model == "vk":
        return _build_payload_vk(p)
    if model == "tension":
        return _build_payload_tension(p)
    if model == "bow":
        return _build_payload_bow(p)
    if model == "geometric":
        return _build_payload_geometric(p)
    if model == "sympathetic":
        return _build_payload_sympathetic(p)
    if model == "jawari":
        return _build_payload_jawari(p)
    if model == "fret":
        return _build_payload_fret(p)
    if model == "bore":
        return _build_payload_bore(p)
    if model == "reed":
        return _build_payload_reed(p)
    return _build_payload_string(p)


def _build_payload_string(p: dict[str, Any]) -> dict[str, Any]:
    model = str(p.get("model", "ideal"))
    audio_dur = _fnum(p, "audio_duration", 2.0)
    anim_win = _fnum(p, "animation_window", 0.06)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    amplitude = _fnum(p, "amplitude", 1e-3)
    pluck_frac = _fnum(p, "pluck_position", 0.3)
    pickup_frac = _fnum(p, "pickup_position", 0.1)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= AUDIO_DUR_MAX):
        raise ParamError(f"audio_duration must be in (0, {AUDIO_DUR_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < pluck_frac < 1.0):
        raise ParamError(f"pluck_position must be in (0, 1), got {pluck_frac}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")

    b = _build_resonator(p)
    c, L, N, fs = b.c, b.L, b.N, b.fs
    f1_base = c / (2.0 * L)
    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)

    # --- audio run: full duration, pickup, no snapshots ------------------------------------------
    b.res.set_state(triangular_pluck(b.res.x, L, pluck_frac * L, amplitude=amplitude))
    n_audio = max(1, round(audio_dur * fs))
    audio_res = simulate(b.res, num_steps=n_audio, pickup_index=pickup_idx)
    pickup = np.asarray(audio_res.output, dtype=float)
    if not np.all(np.isfinite(pickup)):
        raise ParamError(
            "simulation produced non-finite output (instability) — adjust parameters."
        )

    # --- animation run: fresh resonator, short window, fundamental-resolving stride (catch #2) ----
    anim = _build_resonator(p).res
    anim.set_state(triangular_pluck(anim.x, L, pluck_frac * L, amplitude=amplitude))
    anim_stride = max(1, round((fs / f1_base) / fpp))
    n_anim = max(anim_stride, round(anim_win * fs))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    anim_res = simulate(anim, num_steps=n_anim, snapshot_stride=anim_stride)
    frames = np.array([st for _, st in anim_res.snapshots], dtype=float)
    frame_steps = np.array([i for i, _ in anim_res.snapshots], dtype=float)

    # --- audio + assembly ------------------------------------------------------------------------
    audio48, peak = _resample_normalize(pickup, fs)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0

    return {
        "model": model,
        "fs_sim": round(fs, 3),
        "lambda": round(float(getattr(b.res, "lam", float("nan"))), 6),
        "grid": {"x": _finite_list(b.res.x, 6)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(audio_res, b.sigma_zero, b.oracle_2sigma),
        "meta": {
            "c": round(c, 3),
            "f1": round(f1_base, 3),
            "num_steps": int(n_audio),
            "n_frames": int(frames.shape[0]),
            "partials": _partials_block(pickup, fs, model, p, c, L),
        },
    }


# == tension-modulated string (1D nonlinear, model #9) =============================================


def _mode1_shape(res: Any) -> NDArray[np.float64]:
    """The exact discrete eigenvector ``sin(pi x / L)`` on the resonator's grid.

    The Kirchhoff-Carrier -> Duffing reduction is a *single-mode* ansatz, so the oracle is only
    meaningful from a single-mode start. A triangular pluck is broadly multi-mode and would make the
    measured shift a lying number (model #6's ``mode11`` lesson, in a new model).
    """
    return np.sin(np.pi * np.arange(res.N + 1) / res.N)


def _tension_dt_over_t(EA: float, amplitude: float, p2: float, T: float) -> float:
    """Peak tension excess ``dT/T0`` of a single mode — exact, closed form, no stepping.

    The stretch of one mode is ``I = A² p² L/2`` — see
    :func:`~physsynth.analysis.duffing.kc_mode_stretch`, exact on the discrete grid too — and the
    tension is ``T0 + (EA/2L) I``, so the ``L`` cancels. This is what lets the guard run *before*
    any stepping.
    """
    return float(EA * amplitude**2 * p2 / (4.0 * T))


def _tension_measure_steps(fs: float, L: float, c: float) -> int:
    """Step count of one measurement run: ``TENSION_MEASURE_PERIODS`` linear fundamental periods."""
    return max(64, int(round(TENSION_MEASURE_PERIODS * fs * 2.0 * L / c)))


def _interp_zero_cross_frequency(sig: NDArray[np.float64], fs: float) -> float:
    """Fundamental (Hz) from mean zero-crossing spacing, each crossing linearly interpolated.

    :func:`_zero_cross_fundamental` quantizes every crossing to a whole sample; because the mean
    spacing telescopes to ``(zc[-1] - zc[0])/(M-1)``, that quantization lands *directly* on the
    headline shift rather than averaging away (~0.1 Hz on a ~17 Hz shift). Interpolating removes it.
    Kept separate so the von Kármán path stays bit-for-bit.
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    idx = np.where(np.diff(np.signbit(sig)))[0]
    if idx.size < 3:
        return float("nan")
    a, b = sig[idx], sig[idx + 1]
    span = a - b
    frac = np.divide(a, span, out=np.zeros_like(a), where=span != 0.0)
    t = (idx.astype(float) + frac) / fs
    return float(1.0 / (2.0 * float(np.mean(np.diff(t)))))


def _measure_tension_mode1(p: dict[str, Any], amplitude: float) -> dict[str, float]:
    """Frequency, worst off-mode fraction and peak ``dT/T0`` of a **lossless, short** mode-1 run.

    Deliberately NOT the audio run, and deliberately lossless. ``duffing_frequency_shift(A)``
    predicts the frequency at a *fixed* amplitude ``A``; a lossy tension string is a downward
    chirp (as ``A`` decays, ``omega(A) -> omega_0``), so the mean crossing spacing of the pickup
    reports an *amplitude-averaged* frequency that undershoots the oracle badly. The audio run stays
    lossy on purpose — the audible glide is the model's signature — but the *number* is from here.

    Purity is measured against the **fixed** ``||u_0||``, never the instantaneous ``||u||``: a
    single
    mode passes through ``u ~ 0`` twice a period, where roundoff would report a spurious ``1.0``.
    """
    b = _build_resonator({**p, "sigma0": 0.0, "sigma1": 0.0})
    res = b.res
    shape = _mode1_shape(res)
    res.set_state(amplitude * shape)
    scale = float(np.linalg.norm(amplitude * shape))
    denom = float(np.dot(shape, shape))
    n_steps = _tension_measure_steps(b.fs, b.L, b.c)

    q = np.empty(n_steps)
    worst_off, peak_dt = 0.0, 0.0
    for i in range(n_steps):
        u = res.u  # read-only view; `.state` would copy the field every step for nothing
        q[i] = float(np.dot(u, shape)) / denom
        worst_off = max(worst_off, float(np.linalg.norm(u - q[i] * shape)) / scale)
        peak_dt = max(peak_dt, res.tension / res.T - 1.0)
        res.step()
    return {
        "f": _interp_zero_cross_frequency(q, b.fs),
        "off_mode": worst_off,
        "dT_over_T": peak_dt,
        "not_converged": float(res.n_not_converged),
    }


def _tension_spectrum_block(
    p: dict[str, Any], pickup: NDArray[np.float64], fs: float, w0sq: float, eps: float,
    amplitude: float,
) -> dict[str, Any]:
    """The money panel: measured amplitude shift vs the **exact** Duffing closed form.

    Two honesty gates, both following precedent. *Purity* — this project's own discovery is that
    single-mode motion is parametrically unstable above ``dT/T0 ~ 3``; past that the mode breaks up
    (energy is still conserved — it is physics, not a blow-up) and the Duffing reduction no longer
    describes the motion, so the shift is reported as ``null`` rather than a lying number (model
    #6's free-cymbal precedent). *Convergence* is folded into the energy block, not here.
    """
    f_lin = math.sqrt(w0sq) / (2.0 * math.pi)
    shift_oracle = duffing.duffing_frequency_shift(amplitude, w0sq, eps) / (2.0 * math.pi)

    run_a = _measure_tension_mode1(p, amplitude)
    run_ref = _measure_tension_mode1(p, TENSION_REF_AMP)
    off = max(run_a["off_mode"], run_ref["off_mode"])
    pure = off < TENSION_OFFMODE_MAX
    measured = math.isfinite(run_a["f"]) and math.isfinite(run_ref["f"])

    shift_meas: float | None = None
    rel_err: float | None = None
    if pure and measured:
        shift_meas = run_a["f"] - run_ref["f"]
        if shift_oracle > 0.0:
            rel_err = abs(shift_meas - shift_oracle) / shift_oracle

    fmax = max(4.0 * f_lin, 1.25 * (f_lin + shift_oracle))
    pooled = _pooled_spectrum(pickup, fs, fmax)
    freq, mag = ([], []) if pooled is None else (
        _finite_list(pooled[0], 3), _finite_list(pooled[1], 5)
    )
    return {
        "kind": "tension",
        "freq": freq,
        "mag": mag,
        "fmax": round(fmax, 3),
        "f_linear": round(f_lin, 4),
        "f_hardened": round(run_a["f"], 4) if measured else None,
        "f_reference": round(run_ref["f"], 4) if measured else None,
        "shift_measured": round(shift_meas, 4) if shift_meas is not None else None,
        "shift_oracle": round(shift_oracle, 4),
        "shift_rel_error": rel_err,
        "shift_cents": (
            round(float(modal.cents(f_lin + shift_meas, f_lin)), 3)
            if shift_meas is not None and f_lin + shift_meas > 0 else None
        ),
        "dT_over_T": round(run_a["dT_over_T"], 4),
        "purity": {"off_mode": run_a["off_mode"], "tol": TENSION_OFFMODE_MAX, "pure": bool(pure)},
    }


def _build_payload_tension(p: dict[str, Any]) -> dict[str, Any]:
    audio_dur = _fnum(p, "audio_duration", 2.0)
    anim_win = _fnum(p, "animation_window", 0.06)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    amplitude = _fnum(p, "amplitude", TENSION_AMP_DEFAULT)
    pickup_frac = _fnum(p, "pickup_position", 0.1)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= TENSION_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {TENSION_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < amplitude <= TENSION_AMP_MAX):
        raise ParamError(f"amplitude must be in (0, {TENSION_AMP_MAX}] m, got {amplitude}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")
    try:
        n_req = int(p.get("N", 128))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc
    if n_req > TENSION_N_MAX:
        raise ParamError(
            f"N must be <= {TENSION_N_MAX} for the tension string (got {n_req}): every step runs a "
            "tension root-find, so a fine grid is far costlier here than on a linear string."
        )

    b = _build_resonator(p)
    c, L, N, fs = b.c, b.L, b.N, b.fs
    res = b.res
    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)

    # Duffing coefficients of mode 1 on the DISCRETE grid (p2 = (4/h^2) sin^2(pi/2N)) — the
    # eigenvalue the stepper actually rings at. The continuum (pi/L)^2 would fold O(h^2) error into
    # the oracle the measurement is then scored against.
    p2 = damping.spatial_eigenvalue_p2(N, res.h, 1)
    w0sq, eps = duffing.kc_mode_coefficients(
        c=c, kappa=res.kappa, EA=res.EA, rho=res.rho, p2=p2, L=L
    )
    f_lin = math.sqrt(w0sq) / (2.0 * math.pi)
    f_hard_est = f_lin + duffing.duffing_frequency_shift(amplitude, w0sq, eps) / (2.0 * math.pi)

    # Bound dT/T0, not amplitude: EA and T move it just as hard as A does (see TENSION_DT_MAX). The
    # single-mode stretch is exact — I = A² p2 L/2, so dT/T0 = (EA/2L)·I / T = EA·A²·p2/(4T) — so
    # this costs nothing and is checked *before* any stepping.
    dt_over_t = _tension_dt_over_t(res.EA, amplitude, p2, res.T)
    if dt_over_t > TENSION_DT_MAX:
        raise ParamError(
            f"dT/T0 = {dt_over_t:.2f} exceeds {TENSION_DT_MAX} — above this the single mode is "
            "parametrically unstable and breaks up into its neighbours, so the Duffing shift stops "
            "describing the motion. (The breakup is real, energy-conserving physics; it wants a "
            "panel of its own.) Lower the amplitude or EA."
        )

    n_measure = _tension_measure_steps(fs, L, c)
    n_audio_req = max(1, round(audio_dur * fs))
    work = n_audio_req + 2 * n_measure + round(anim_win * fs)
    if work > TENSION_WORK_MAX:
        raise ParamError(
            f"work budget exceeded ({work:,} steps > {TENSION_WORK_MAX:,}): every step runs a "
            "tension root-find (~176 µs at N=128). Lower N or audio_duration."
        )

    shape = _mode1_shape(res)
    res.set_state(amplitude * shape)
    # Measured AT THE IC, which is the PEAK: the mode starts at maximum displacement with zero
    # velocity, so all of E is potential and the stretch is maximal. The stretch oscillates twice a
    # period, so reading this off the *final* state would instead report wherever the run happened
    # to stop (0.115 vs 8e-6 for a 0.5 s vs 0.4 s render — same physics, meaningless difference).
    e0 = res.energy()
    nl_fraction = res.nonlinear_energy() / e0 if e0 > 0 else 0.0

    n_audio = max(1, round(audio_dur * fs))
    audio_res = simulate(res, num_steps=n_audio, pickup_index=pickup_idx)
    pickup = np.asarray(audio_res.output, dtype=float)
    if not np.all(np.isfinite(pickup)):
        raise ParamError(
            "simulation produced non-finite output (instability) — adjust parameters."
        )

    # Animation stride rides the *hardened* estimate, not the linear f1: at a large amplitude the
    # motion runs up to ~2x faster, and a linear-f1 stride would under-resolve it (catch #2).
    anim = _build_resonator(p).res
    anim.set_state(amplitude * _mode1_shape(anim))
    anim_stride = max(1, round((fs / f_hard_est) / fpp))
    n_anim = max(anim_stride, round(anim_win * fs))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    anim_res = simulate(anim, num_steps=n_anim, snapshot_stride=anim_stride)
    frames = np.array([st for _, st in anim_res.snapshots], dtype=float)
    frame_steps = np.array([i for i, _ in anim_res.snapshots], dtype=float)

    audio48, peak = _resample_normalize(pickup, fs)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    n_bad = int(res.n_not_converged)
    tol = res.tension_tol
    convergence = {
        "all_converged": n_bad == 0,
        "n_not_converged": n_bad,
        "tension_tol": tol,
        "bracket_expansions": int(res.bracket_expansions),
        "detail": (
            f"tension root-find did not converge: {n_bad} step(s), tol {tol:.0e}\n"
            "energy verdict N/A — lower the amplitude or EA"
        ),
        "note": f"  ·  tension solve converged (tol {tol:.0e})",
    }

    return {
        "model": "tension",
        "fs_sim": round(fs, 3),
        "lambda": round(float(res.lam), 6),
        "grid": {"x": _finite_list(res.x, 6)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(audio_res, b.sigma_zero, b.oracle_2sigma, convergence=convergence),
        "meta": {
            "c": round(c, 3),
            "f1": round(f_lin, 3),
            "num_steps": int(n_audio),
            "n_frames": int(frames.shape[0]),
            # nested under meta, like every other model's spectrum — the frontend reads
            # payload.meta.spectrum (a top-level key here would silently render nothing).
            "spectrum": _tension_spectrum_block(p, pickup, fs, w0sq, eps, amplitude),
            "EA_over_T": round(res.EA_over_T, 3),
            # A nonlinearity HIDES at small amplitude, where the run merely re-plays the linear
            # scheme (model #6's lesson) — so report the PEAK fraction of E the stretch term holds.
            "nonlinear_fraction": round(float(nl_fraction), 6),
        },
    }


# == bowed string (Phase D batch 2) ================================================================
#
# The project's first *actively driven* model in the viewer, and the reason the energy panel grows a
# third verdict type. Everything else shown so far is passive: it either conserves (sigma=0 drift)
# or dissipates (sigma>0 passivity). A bow pumps energy in through a nonlinear friction curve, so
# BOTH of those verdicts are wrong for it — see :func:`_balance_verdict`. What replaces them is the
# balance `E - E0 == bow_work - loss`, which is exact to ~6e-15 at sigma=0.
#
# The animation needs no new viz: Helmholtz motion IS the string's shape, so the existing 1-D line
# path draws the travelling corner for free. The second panel is the stick-slip trace, whose oracle
# (slip fraction == beta) is claimed only inside the Helmholtz window — see
# :func:`_bow_stickslip_block`.


def _run_bow(
    bow: BowedString,
    num_steps: int,
    pickup_index: int | None = None,
    snapshot_stride: int = 0,
    snapshot_from: int = 0,
) -> tuple[SimResult, NDArray[np.float64], NDArray[np.float64]]:
    """Run a :class:`BowedString` like :func:`engine.simulate`, also tracking work and ``v_rel``.

    The engine captures energy (+ optional pickup/snapshots) but is generic over ``Resonator``, so
    it knows nothing of the bow's cumulative ``bow_work`` — the balance verdict's second curve — or
    its per-step ``v_rel``, which the stick-slip panel reads. Capturing them here keeps ``core``
    untouched and the engine generic; the same move :func:`_run_vk` makes for its Picard telemetry.

    ``snapshot_from`` delays frame capture to a given step, which is what lets the bow animate its
    *settled* motion out of the single audio run. Every other model re-runs a second resonator for
    the animation because its window is the attack, at ``t = 0``; the bow's window is the tail, and
    the audio run already passes through it — so re-running would silently double the cost of a
    model whose every step is a root-find.

    Returns ``(SimResult, work, v_rel)`` with ``work`` aligned step-for-step with ``res.energy``
    (both length ``num_steps + 1``), so the balance curves decimate on a single shared index.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")
    n = num_steps + 1
    energy = np.empty(n)
    work = np.empty(n)
    v_rel = np.empty(n)
    output = np.empty(n) if pickup_index is not None else None
    snapshots: list[tuple[int, NDArray[np.float64]]] = []

    def _snap(i: int) -> bool:
        if not snapshot_stride or i < snapshot_from:
            return False
        return (i - snapshot_from) % snapshot_stride == 0

    energy[0], work[0], v_rel[0] = bow.energy(), bow.bow_work, bow.v_rel
    if output is not None:
        output[0] = bow.displacement_at(pickup_index)  # type: ignore[arg-type]
    if _snap(0):
        snapshots.append((0, bow.state))
    for i in range(1, n):
        bow.step()
        energy[i], work[i], v_rel[i] = bow.energy(), bow.bow_work, bow.v_rel
        if output is not None:
            output[i] = bow.displacement_at(pickup_index)  # type: ignore[arg-type]
        if _snap(i):
            snapshots.append((i, bow.state))

    res = SimResult(
        time=np.arange(n) * bow.k, energy=energy, output=output, fs=1.0 / bow.k, snapshots=snapshots
    )
    return res, work, v_rel


def _bow_stickslip_block(
    v_rel: NDArray[np.float64],
    pickup: NDArray[np.float64],
    fs: float,
    v_bow: float,
    beta: float,
    f1: float,
) -> dict[str, Any]:
    """Stick-slip panel: the ``v_rel`` trace, and the Helmholtz oracle ``slip_fraction == beta``.

    Helmholtz motion is a two-state cycle: the string *sticks* to the bow (co-moving, ``v_rel ~ 0``)
    for a fraction ``1 - beta`` of the period, then *slips* back once, for a fraction ``beta``
    (= bow-to-nut distance / L). One slip per period, and the slip fraction equals ``beta`` — a
    closed-form prediction with the bow-position slider sitting right on top of its free parameter.
    Detector and bars are the core's own (``tests/test_bow_modal.py``): an absolute threshold at
    half the bow speed, which captures the whole slip *duration* (the smooth friction curve rounds
    the corner, so a relative-to-peak threshold would only catch the trough).

    **The oracle is only honest inside Schelleng's window, and that gating is the point.** Bow force
    has to sit between a minimum and a maximum that *narrow* as the bow moves off the bridge; drag
    force / beta / v_bow outside that window and the motion legitimately stops being Helmholtz —
    it crushes, or breaks into raucous multi-slip motion (~18 slips per period). That is real
    physics faithfully reproduced, **not** a bug, so scoring ``slip_fraction`` against ``beta``
    there would paint a red FAIL on a correct simulation — tension's sigma-divergence and von
    Kármán's broad-strike trap, a third time. So: ``slips_per_period`` is always *reported* as an
    observation; the beta-match is claimed as a *verdict* only when the motion is actually
    one-slip-per-period, and otherwise the panel says "outside the Helmholtz window" and scores
    nothing (the free cymbal's ``null`` "crash cascade" precedent).

    The pitch check rides the same gate, for the same reason: a raucous note's "pitch" may be a
    subharmonic, so comparing it to ``f1`` off-window would report a confident, meaningless number.
    """
    n_tail = int(min(len(v_rel), max(4, round(BOW_TAIL_FRAC * len(v_rel)))))
    tail = v_rel[-n_tail:]
    slipping = np.abs(tail) >= 0.5 * v_bow          # the core's absolute half-bow-speed threshold
    slip_fraction = float(np.mean(slipping))
    onsets = int(np.sum((~slipping[:-1]) & (slipping[1:])))
    n_periods = len(tail) * f1 / fs
    slips_per_period = float(onsets / n_periods) if n_periods > 0 else 0.0
    helmholtz = bool(BOW_SLIPS_LO < slips_per_period < BOW_SLIPS_HI)

    # Trace: a few periods of the settled tail, decimated for the plot.
    n_show = int(min(len(tail), max(8, round(BOW_TRACE_PERIODS * fs / f1))))
    trace = tail[-n_show:]
    step = max(1, len(trace) // N_ENERGY_POINTS)
    trace = trace[::step]

    block: dict[str, Any] = {
        "kind": "bow",
        "v_rel": _finite_list(trace),
        "dt": float(step / fs),
        "v_bow": float(v_bow),
        "stick_threshold": float(0.5 * v_bow),
        "beta": float(beta),
        "slip_fraction": round(slip_fraction, 4),
        "slips_per_period": round(slips_per_period, 3),
        "helmholtz": helmholtz,
        "slip_tol": BOW_SLIP_MATCH_TOL,
    }
    if helmholtz:
        block["slip_matches_beta"] = bool(abs(slip_fraction - beta) < BOW_SLIP_MATCH_TOL)
        block["slip_error"] = round(slip_fraction - beta, 4)
        # The bow does NOT set the pitch — the string does. Same lesson as the reed and the bore.
        sig = pickup[-n_tail:] - float(np.mean(pickup[-n_tail:]))
        detected = float(
            spectrum.measure_partials_near(sig, fs, np.array([f1]), search_hz=0.15 * f1)[0]
        )
        block["f_detected"] = round(detected, 3) if math.isfinite(detected) else None
        block["f1"] = round(f1, 3)
        block["pitch_cents"] = (
            round(1200.0 * math.log2(detected / f1), 2)
            if math.isfinite(detected) and detected > 0 else None
        )
    else:
        block["slip_matches_beta"] = None
        # A zero onset count is ambiguous on its own — never sticking and never slipping both give
        # zero stick->slip transitions — so let the slip fraction say which it was.
        block["regime"] = "never_sticks" if slip_fraction > 0.95 else "multi_slip"
        what = (
            "the string never sticks to the bow (slip fraction ≈ 1), so no stick-slip cycle forms"
            if slip_fraction > 0.95 else
            f"{slips_per_period:.1f} slips per period (clean bowing is 1)"
        )
        block["note"] = (
            f"outside the Helmholtz window — {what}. Real physics, not a solver failure: "
            "Schelleng's playable force window has a floor and a ceiling, and both narrow as the "
            "bow moves off the bridge. The slip = beta oracle only describes one-slip motion, so "
            "it is not scored here."
        )
    return block


def _build_payload_bow(p: dict[str, Any]) -> dict[str, Any]:
    audio_dur = _fnum(p, "audio_duration", 2.0)
    anim_win = _fnum(p, "animation_window", 0.06)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    pickup_frac = _fnum(p, "pickup_position", 0.33)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= BOW_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {BOW_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")
    try:
        n_req = int(p.get("N", 100))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc
    if n_req > BOW_N_MAX:
        raise ParamError(
            f"N must be <= {BOW_N_MAX} for the bowed string (got {n_req}): every step runs a "
            "friction root-find, so a fine grid is far costlier here than on a linear string."
        )

    b = _build_resonator(p)
    c, L, N, fs = b.c, b.L, b.N, b.fs
    bow: BowedString = b.res
    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)
    # The string's own fundamental — the pitch the bow locks to, not one the bow chooses. Reduces
    # to c/2L at kappa=0 (the default flexible string) and stays correct if stiffness is dialled in.
    f1 = float(modal.stiff_harmonic_frequencies(c, L, _fnum(p, "kappa", 0.0), 1)[0])

    n_audio = max(1, round(audio_dur * fs))
    if n_audio > BOW_WORK_MAX:
        raise ParamError(
            f"work budget exceeded ({n_audio:,} steps > {BOW_WORK_MAX:,}): every step runs a "
            "friction root-find. Lower N or audio_duration."
        )

    anim_stride = max(1, round((fs / f1) / fpp))
    n_anim = min(n_audio, max(anim_stride, round(anim_win * fs)))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))

    # ONE run, and the frames come out of it. Animate the SETTLED motion, not the attack: the bow
    # starts from rest (there is no pluck to place), so the opening frames are a near-flat string —
    # the corner has not formed yet — and a window at t=0 would show almost nothing moving. The
    # audio run already passes through the settled tail, so capture there rather than re-running a
    # second resonator the way every other (attack-windowed) model does: that would double the cost
    # of a model whose every step is a root-find, and the work budget would not see it.
    n_settle = max(0, n_audio - n_anim)
    audio_res, work, v_rel = _run_bow(
        bow, n_audio, pickup_index=pickup_idx, snapshot_stride=anim_stride, snapshot_from=n_settle
    )
    pickup = np.asarray(audio_res.output, dtype=float)
    if not np.all(np.isfinite(pickup)):
        raise ParamError("simulation produced non-finite output (instability) — adjust parameters.")

    frames = np.array([st for _, st in audio_res.snapshots], dtype=float)
    # relative to the window start, so the animation clock begins at 0 like every other model's
    frame_steps = np.array([i - n_settle for i, _ in audio_res.snapshots], dtype=float)

    audio48, peak = _resample_normalize(pickup, fs)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0

    return {
        "model": "bow",
        "fs_sim": round(fs, 3),
        "lambda": round(float(bow.string.lam), 6),
        "grid": {"x": _finite_list(bow.string.x, 6)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        # balance_work switches the panel to the third verdict type; see _balance_verdict.
        "energy": _energy_block(audio_res, b.sigma_zero, b.oracle_2sigma, balance_work=work),
        "meta": {
            "c": round(c, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_audio),
            "n_frames": int(frames.shape[0]),
            "spectrum": _bow_stickslip_block(v_rel, pickup, fs, bow.v_bow, bow.beta, f1),
            "bow_x": round(float(bow.x_bow), 4),
            "beta": round(float(bow.beta), 4),
            # Reported, never asserted: above 1 the friction curve is multivalued — the regime of
            # real sustained bowing — but it is not a stability limit (friction is bounded, the
            # scheme is stable and the balance exact for any root), so it is a diagnostic only.
            "helmholtz_number": round(float(bow.helmholtz_number), 3),
            "fallbacks": int(bow.fallbacks),
        },
    }


# == geometrically-exact string (1D nonlinear, 3 fields — model #10) ===============================
#
# The viewer's first VIZ-ONLY model, and the reason is physics, not a shortcut. This string has a
# longitudinal wave at c_long = sqrt(EA/rho); for a real string (EA/T ~ 500) that is ~22x the
# transverse c you actually hear. Resolving it — lam_long <= 1, the model's central trap — forces
# fs ~ 22x a transverse-only model's, and every step is a vector Newton solve over 3 coupled fields
# (~2-4 ms). One second of listenable audio is ~10 minutes of compute, and there is no cheat:
# lam_long > 1 is exactly the silent-garbage regime this model exists to warn about. So it renders
# pictures, not sound. (The phantom-partials bridge-force spectrum — EA v_x(0), the model's true
# audible signature — is a separate ~30 s run at N=32/kappa=8 and lands in its own later increment.)
#
# What replaces the audio is the orbit: model #9 has ONE polarization, so its only orbit is a point
# on a line. Three regimes, three claims, cheapest first:
#   * planar   — a bit-exact straight line. max|w| == 0.0, not "small": it is the w -> -w reflection
#                symmetry, and it is also the honesty gate for the whirl below (without it, every
#                growth ratio is partly measuring a seed leak).
#   * rotating — a true circle, from the converged rotating-wave BVP (Tier B): an EXACT solution of
#                the scheme, so it is round from frame 1 and needs no growth to look like anything.
#                Its oracle is its own roundness, and the longitudinal field holding still (psi is a
#                nonzero *static* stretch — asserting v == 0 would assert the physics away).
#   * whirl    — the Mathieu tongue. NOT drawn as an opening orbit: at the affordable 0.06 s the
#                growth is ~60x, but from a 1e-3 relative seed that only reaches max|w|/max|u| ~
#                0.08, which on equal axes still reads as a line. (Saturation — an orbit you can
#                SEE open —
#                needs ~0.1 s from this seed, ~7.5k steps, past the work budget; the diagnose script
#                needs 0.22 s because its velocity seed carries no omega_u factor and so starts a
#                thousandfold lower.) The honest cheap signature is the ENVELOPE of max|w| on
#                log-y —
#                a straight line there IS the instability — plus the energy drift holding ~1e-12
#                THROUGH the blow-up, which is what separates redistribution from a diverging solve.


def _geom_regime(p: dict[str, Any]) -> str:
    regime = str(p.get("domain", "rotating"))
    if regime not in GEOM_REGIMES:
        raise ParamError(f"regime must be one of {GEOM_REGIMES}, got {regime!r}.")
    return regime


def _geom_long_kinetic(res: GeometricString) -> float:
    """Kinetic energy of the LONGITUDINAL field alone (J) — the rotating wave's money number.

    Deliberately not "is ``v`` zero": ``psi``, the static stretch, is a *nonzero* equilibrium the
    helix holds, so ``v == 0`` would assert the physics away (the plan's own criterion was wrong
    here). What vanishes on a true rotating wave is the longitudinal *motion*.
    """
    dt_v = (res.v[1:-1] - res.v_prev[1:-1]) / res.k
    return 0.5 * res.rho * res.h * float(np.dot(dt_v, dt_v))


def _sliding_max(a: NDArray[np.float64], win: int) -> NDArray[np.float64]:
    """Centred sliding-window maximum — the ENVELOPE of an oscillating signal.

    The whirl's ``max|w|`` is a growing *oscillation*, not a growing number: every node crosses zero
    twice a period, so the instantaneous spatial max is non-monotone (it reads 4.6e-8 -> 1.8e-8 ->
    2.9e-7 -> 1.5e-7 -> 8.5e-7 on the default run) and a log-y plot of it is a mess of spikes that
    hides the straight line underneath. A max over a ~1-period window recovers the envelope,
    whose slope on log axes is the Mathieu growth rate. This is the family's recurring trap in
    another dress: never read an oscillating field at one phase.
    """
    win = int(min(max(win, 1), a.size))
    if win <= 1:
        return a.copy()
    view = np.lib.stride_tricks.sliding_window_view(a, win)
    env = np.asarray(view.max(axis=1), dtype=float)
    lead = (win - 1) // 2
    tail = a.size - env.size - lead
    return np.concatenate([np.full(lead, env[0]), env, np.full(max(tail, 0), env[-1])])[: a.size]


def _geom_tongue(
    *, c: float, EA: float, T: float, rho: float, L: float, N: int, dt_over_t0: float, frac: float
) -> dict[str, float]:
    """The whirl recipe: amplitude, ``kappa_w``, the driven frequency and the predicted rate.

    Whirling is a **Mathieu tongue** and its parameters are NOT free — picking them by eye gives
    1.1x growth and a picture of nothing. Reduce to one mode pair (a sine is exact for both ``d_xx``
    and ``d_xxxx`` under simple support, so the two polarizations share it even when detuned) and
    linearise out of plane about ``q_u = A cos(Omega t)``::

        d_tt q_w + [w_w^2 + eps A^2/2 + (eps A^2/2) cos(2 Omega t)] q_w = 0

    — a pump at ``2 Omega`` on the principal resonance, unstable iff ``0 < delta < eps A^2/2`` with
    ``delta = w_w^2 - w_u^2``, peaking at ``delta = eps A^2/4``. ``eps`` is model #9's OWN
    ``kc_mode_coefficients`` under ``EA -> a`` (the quartic is isotropic, so the planar reduction is
    untouched by the third field).

    Two things this encodes that cost a cycle each to learn: the driven plane must be the SOFT one
    (``kappa_u = 0 < kappa_w``, so ``delta > 0`` — Gough's real signature; the same string driven on
    the stiff plane gives 1.00x), and the tongue is **refinement-invariant** in the coordinate
    ``frac``, so ``kappa_w`` must be recomputed from ``p2`` at the actual N (39.05 / 39.01 / 39.00
    N = 16 / 24 / 32) rather than pinned to a number that was right at one grid.

    The predicted rate is ``(Om/2) sqrt(qM^2 - sigma^2)``, pump strength ``qM = eps A^2/(4 Om^2)``
    and detuning ``sigma = (delta - eps A^2/4)/Om^2`` — the tongue and its rate profile are the same
    formula read twice, so the rate peaks at ``frac = 1/4`` and dies at both edges. ``Om`` is the
    **planar** Duffing frequency ``sqrt(w0^2 + 3/4 eps A^2)``: the driven motion here is a plane
    oscillation, NOT the circular ``sqrt(w0^2 + eps A^2)`` of the rotating wave.
    """
    p2 = damping.spatial_eigenvalue_p2(N, L / N, 1)
    omega0_sq, eps = duffing.kc_mode_coefficients(c=c, kappa=0.0, EA=EA - T, rho=rho, p2=p2, L=L)
    amplitude = float(np.sqrt(dt_over_t0 * omega0_sq / eps))   # kappa_u=0 => eps A^2/w0^2 == dT/T0
    kappa_w = float(np.sqrt(frac * eps * amplitude**2 / p2**2)) if frac > 0 else 0.0
    ea2 = eps * amplitude**2
    omega = float(np.sqrt(omega0_sq + 0.75 * ea2))
    q_m = ea2 / (4.0 * omega**2)
    sigma = (frac * ea2 - ea2 / 4.0) / omega**2       # delta == frac * eps A^2, by construction
    rate = float((omega / 2.0) * np.sqrt(max(q_m**2 - sigma**2, 0.0)))
    return {
        "amplitude": amplitude, "kappa_w": kappa_w, "omega": omega, "predicted_rate": rate,
        "omega0_sq": float(omega0_sq),
    }


def _build_geometric(p: dict[str, Any], *, kappa: float, kappa_w: float) -> GeometricString:
    """Construct a :class:`GeometricString` with ``fs`` driven by **lam_long**, not by lambda.

    The reverse of models #1-#9, on purpose: ``lam = c k / h`` is the familiar knob and it is the
    wrong one here — at the ``lam = 0.5`` a reader of the string family reaches for first,
    ``lam_long`` is silently ~11 and the model returns garbage that conserves nothing. So the slider
    IS ``lam_long`` and ``lam`` is whatever falls out (~0.04).
    """
    L = _fnum(p, "L", 1.0)
    T = _fnum(p, "T", 200.0)
    rho = _fnum(p, "rho", 0.005)
    EA = _fnum(p, "EA", 1.0e5)
    lam_long = _fnum(p, "lam_long", GEOM_LAM_LONG_DEFAULT)
    theta = _fnum(p, "theta", 0.28)
    try:
        N = int(p.get("N", 16))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc
    if N > GEOM_N_MAX:
        raise ParamError(
            f"N must be <= {GEOM_N_MAX} for the geometrically-exact string (got {N}): each step "
            "a vector Newton solve over three coupled fields, and fs rides N."
        )
    if not (0.0 < lam_long <= GEOM_LAM_LONG_MAX):
        raise ParamError(
            f"lam_long must be in (0, {GEOM_LAM_LONG_MAX}], got {lam_long}. This is the model's "
            "central trap: above 1 the longitudinal wave is under-resolved, and because the "
            "theta-scheme is unconditionally stable it fails SILENTLY — no CFL error, just quiet "
            "nonsense that stops conserving."
        )
    if EA <= 0:
        raise ParamError(f"EA must be positive, got {EA}.")
    c_long = math.sqrt(EA / rho)
    fs = c_long * N / (L * lam_long)
    return GeometricString(
        L=L, T=T, rho=rho, fs=fs, N=N, EA=EA, kappa=kappa, kappa_w=kappa_w,
        sigma0=_fnum(p, "sigma0", 0.0), sigma1=_fnum(p, "sigma1", 0.0), theta=theta,
    )


class _GeomRun:
    """Per-step telemetry of a geometric run (the fields are what the panels are made of)."""

    def __init__(self, n: int, width: int) -> None:
        self.E = np.empty(n + 1)
        self.u_probe = np.empty(n + 1)
        self.w_probe = np.empty(n + 1)
        self.u_max = np.empty(n + 1)
        self.w_max = np.empty(n + 1)
        self.long_kin = 0.0
        self.frames: list[NDArray[np.float64]] = []
        self.frame_steps: list[int] = []
        self.width = width
        # Phantom regime only (see _run_geometric's `modal`): the bridge force and the two modal
        # projections. Index 0 is the IC and is dropped before analysis, so these hold exactly the
        # n_steps post-step samples the test rig measures.
        self.bridge: NDArray[np.float64] | None = None
        self.q1: NDArray[np.float64] | None = None
        self.q2: NDArray[np.float64] | None = None


def _bridge_force(res: GeometricString) -> float:
    """The longitudinal end force ``EA v_x(0) = EA v[1]/h`` (N) — the piano's radiating channel."""
    return res.EA * res.v[1] / res.h


def _run_geometric(
    res: GeometricString,
    n_steps: int,
    *,
    probe: int,
    anim_stride: int,
    modal: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    long_kin: bool = True,
) -> _GeomRun:
    """Step the string, capturing everything the panels need in ONE pass.

    Energy every step is free here, so it is taken every step: ``energy()`` costs ~0.15 ms against
    a ~2-4 ms Newton step, so the drift gate — what separates a whirl (redistribution) from
    a diverging solve — is ~4% overhead. The orbit trail is captured at full rate too (2 floats a
    step) and decimated at the end; only the 3-field snapshots ride the animation stride.

    ``modal=(sin1, sin2)`` additionally records the bridge force and the two modal amplitudes (the
    phantom regime). Modal *projections*, not a point probe: they isolate ``f1`` and ``f2`` from
    each other, so each is a single clean peak and the parabolic refinement is not fighting a
    neighbour's leakage skirt. Three dot products a step against a ~2 ms Newton solve — free.

    ``long_kin=False`` skips the longitudinal-kinetic tracking, which only the rotating regime
    reports. It is an allocation and a dot product per step for a number nobody reads, and the
    phantom regime is 15,900 steps long — this is the one regime where that arithmetic shows up.
    """
    run = _GeomRun(n_steps, res.N + 1)
    if modal is not None:
        sin1, sin2 = modal
        d1, d2 = float(np.dot(sin1, sin1)), float(np.dot(sin2, sin2))
        run.bridge = np.empty(n_steps + 1)
        run.q1, run.q2 = np.empty(n_steps + 1), np.empty(n_steps + 1)

    def _cap(i: int) -> None:
        st = res.state
        run.frames.append(np.stack([st.u, st.w, st.v]))
        run.frame_steps.append(i)

    def _sample(i: int) -> None:
        run.E[i] = res.energy()
        run.u_probe[i] = res.u[probe]
        run.w_probe[i] = res.w[probe]
        run.u_max[i] = float(np.max(np.abs(res.u)))
        run.w_max[i] = float(np.max(np.abs(res.w)))
        if run.bridge is not None:
            run.bridge[i] = _bridge_force(res)
            run.q1[i] = np.dot(res.u, sin1) / d1
            run.q2[i] = np.dot(res.u, sin2) / d2

    _sample(0)
    _cap(0)
    for i in range(1, n_steps + 1):
        res.step()
        _sample(i)
        if long_kin:
            run.long_kin = max(run.long_kin, _geom_long_kinetic(res))
        if i % anim_stride == 0:
            _cap(i)
    if not np.all(np.isfinite(run.E)):
        raise ParamError("simulation produced non-finite energy (instability) — adjust parameters.")
    return run


def _geom_orbit_block(run: _GeomRun, n_frames: int) -> dict[str, Any]:
    """The (u, w) trail at the probe node — the plot model #9 structurally cannot draw."""
    n = run.u_probe.size
    idx = np.linspace(0, n - 1, min(n, GEOM_ORBIT_POINTS)).astype(int)
    return {
        "u": _b64f32(run.u_probe[idx]),
        "w": _b64f32(run.w_probe[idx]),
        "n": int(idx.size),
        # The trail accumulates as the animation plays, so the frontend needs to know how far along
        # the trail one frame is. Float on purpose: the two strides do not divide evenly.
        "per_frame": float(idx.size / max(n_frames, 1)),
    }


def _geom_whirl_block(
    run: _GeomRun, fs: float, f_osc: float, tongue: dict[str, Any]
) -> dict[str, Any]:
    """The whirl panel: the log-y envelope of max|w|, the growth ratio, and the tongue coordinate.

    The growth ratio uses the same definition as the diagnose rig — the max over the last eighth of
    the run against the max over the first eighth — so the viewer and the figures report one number.
    It is read off the **spatial** max, not a probe node, so it cannot be an artifact of where the
    probe sits.

    The measured rate comes from the **last two** quarter-envelopes, never the whole run: the seed
    is not the growing Floquet mode, so quarter one is contaminated by its decaying partner. It is
    reported against the closed-form Mathieu rate as a Tier-C number — the match is 5-11% and
    *systematically low* (leading-order eps, plus the seed's non-growing component), so this is a
    "reported, not asserted" comparison and the panel must not dress it up as a pass/fail.
    """
    w = run.w_max
    env = _sliding_max(w, max(1, round(fs / max(f_osc, 1e-9))))
    idx = np.linspace(0, env.size - 1, min(env.size, GEOM_ENV_POINTS)).astype(int)
    eighth = max(1, w.size // GEOM_GROWTH_FRAC)
    first = float(np.max(w[:eighth]))
    last = float(np.max(w[-eighth:]))
    growth = last / first if first > 0 else 0.0

    q = max(1, w.size // 4)
    quarters = [float(np.max(w[i * q : (i + 1) * q])) for i in range(4)]
    t_total = (w.size - 1) / fs
    measured = (float(4.0 * np.log(quarters[3] / quarters[2]) / t_total)
                if quarters[2] > 0 and quarters[3] > 0 else None)
    predicted = float(tongue["predicted_rate"])
    return {
        "kind": "whirl",
        "time": _finite_list(idx / fs, 6),
        "envelope": _finite_list(env[idx]),
        "growth": growth,
        "w_over_u": float(last / max(float(np.max(run.u_max)), 1e-300)),
        "seeded": bool(first > 0.0),
        "measured_rate": measured,
        "rate_ratio": (measured / predicted) if (measured is not None and predicted > 0) else None,
        **tongue,
    }


def _geom_phantom_dt_over_t0(res: GeometricString, amplitude: float) -> float:
    """The static tension excess ``dT/T0`` of the two-mode IC — the amplitude bound, exact and free.

    Bound ``dT/T0``, never ``A`` (the tension string's lesson): ``A`` is a proxy, and ``EA`` and
    ``T`` move the tension excess just as hard, so an amplitude-only cap would let ``EA = 2e5``
    break up with none the wiser. For ``u = A(phi_1 + phi_2)`` the modes are orthogonal in the
    stretch integral (``int phi_1' phi_2' = 0``), so the excess is just the sum of theirs::

        dT = (a/(2L)) int u_x^2 dx = a A^2 (p2_1 + p2_2) / 4

    with ``a = EA - T0`` the model's own nonlinear coefficient and ``p2_n`` the DISCRETE spatial
    eigenvalue at the actual N (so the bound is refinement-invariant, as the whirl's tongue is).
    """
    p2_1 = damping.spatial_eigenvalue_p2(res.N, res.L / res.N, 1)
    p2_2 = damping.spatial_eigenvalue_p2(res.N, res.L / res.N, 2)
    return float((res.EA - res.T) * amplitude**2 * (p2_1 + p2_2) / (4.0 * res.T))


def _geom_phantom_block(run: _GeomRun, res: GeometricString) -> dict[str, Any]:
    """The phantom panel: the bridge spectrum, the four combination tones, and the defect verdict.

    Every number here is *measured in this one run*. In particular the combinations are built from
    the MEASURED ``f1``, ``f2`` and never predicted: those partials carry the theta-scheme's
    temporal dispersion *and* the amplitude hardening, and the phantom rides on whatever the
    partials actually are — predicting them would fold both errors into the oracle and measure the
    formula instead of the string.

    The headline is the **inharmonicity defect** ``f2 - 2 f1``, which is the primary form of the
    Conklin signature and needs no oracle at all. For a harmonic string the low phantoms coincide
    with partials exactly (``f2 - f1 = f1``, ``2 f1 = f2``), so the distance from the difference
    tone to ``f1`` and from ``2 f1`` to ``f2`` are BOTH exactly ``|f2 - 2 f1|`` — nonzero only
    because kappa > 0. It also carries **no hardening confound**: hardening moves the phantoms and
    the partials together (measured, it slightly *widens* the defect: it works against the claim).

    The plan's other form — ``f1 + f2`` landing ~9B f1 below ``f3`` — is deliberately NOT computed
    here. Mode 3 is not excited, so it needs the discrete ladder as an oracle, and that oracle has
    to be *earned* by a second amp -> 0 run. The step count is amplitude-independent, so that run
    costs the same ~45 s as this one: 90 s for a strictly weaker statement. The ladder is still
    shipped, as shown-not-scored reference markers (where partials would be), which is what makes
    "the phantoms land in the gaps" a picture rather than an assertion.
    """
    fs = res.fs
    bridge, q1, q2 = run.bridge[1:], run.q1[1:], run.q2[1:]   # drop the IC sample (v = 0)
    f1 = float(spectrum.detect_peaks(q1, fs, 1, f_min=10.0)[0])
    f2 = float(spectrum.detect_peaks(q2, fs, 1, f_min=10.0)[0])
    combos = {"f2-f1": f2 - f1, "2f1": 2.0 * f1, "f1+f2": f1 + f2, "2f2": 2.0 * f2}
    defect = f2 - 2.0 * f1
    f_long1 = float(res.c_long / (2.0 * res.L))

    # Blind detection, band-limited below the first free longitudinal mode. detect_peaks, NOT
    # measure_partials_near: anchoring a search window on the frequency under test is how a spectrum
    # test passes by construction. Nothing here tells the detector where a phantom should be. The
    # band limit is a physical cut, not a fudge — see the regime's header comment.
    peaks = spectrum.detect_peaks(bridge, fs, 40, f_min=10.0)
    peaks = peaks[peaks < 0.9 * f_long1]
    freqs, mag, _ = spectrum.magnitude_spectrum(bridge, fs)
    mags = np.array([mag[int(np.argmin(np.abs(freqs - pk)))] for pk in peaks])

    combo_err: float | None = None
    dominance: float | None = None
    if peaks.size >= 4:
        strongest = np.sort(peaks[np.argsort(mags)[::-1][:4]])
        combo_err = float(np.max(np.abs(strongest - np.sort(np.array(list(combos.values()))))))
        is_combo = np.array([any(abs(pk - v) < 1.0 for v in combos.values()) for pk in peaks])
        if is_combo.any() and (~is_combo).any():
            dominance = float(mags[is_combo].min() / mags[~is_combo].max())

    def _nearest(target: float) -> float | None:
        return float(peaks[int(np.argmin(np.abs(peaks - target)))]) if peaks.size else None

    near_f1, near_f2 = _nearest(f1), _nearest(f2)
    peak_diff, peak_2f1 = _nearest(combos["f2-f1"]), _nearest(combos["2f1"])
    # The two displacements are the same physical number approached from opposite sides — the
    # difference tone sits `defect` BELOW f1, and 2f1 sits `defect` below f2.
    disp = [abs(peak_diff - f1) if peak_diff is not None else None,
            abs(peak_2f1 - f2) if peak_2f1 is not None else None]

    # EA = T0 kills the nonlinear coefficient a = EA - T0 outright, so the three fields decouple and
    # v, started at rest, never leaves it: the bridge force is zero *identically*, not "small". That
    # is the harness control (the core suite's own), and it is a RESULT rather than an error — the
    # phantom channel does not exist on a linear string. It must be checked before the verdict
    # below, because with no peaks at all `resolved` would otherwise paint the claim green over an
    # empty spectrum. detect_peaks returns [] on a zero signal rather than raising, so nothing else
    # catches this.
    bridge_max = float(np.max(np.abs(bridge))) if bridge.size else 0.0
    linear = bool(bridge_max == 0.0)
    resolved = bool(defect >= GEOM_PHANTOM_DEFECT_MIN) and not linear and peaks.size >= 4
    # The drawn traces ride a DENSER grid than the detector — see GEOM_PHANTOM_DISPLAY_PAD. Both
    # bands come off this one FFT.
    d_freqs, d_mag, _ = spectrum.magnitude_spectrum(
        bridge, fs, zero_pad_factor=GEOM_PHANTOM_DISPLAY_PAD
    )
    wide = _pool_band(d_freqs, d_mag, 0.0, GEOM_PHANTOM_BAND * f1)
    # The zoom strip: the f1 / (f2-f1) pair, where the wide strip's ~1.2 Hz/px renders the claim's
    # second half backwards. Centred between the two so both are on screen whatever the defect is.
    z_lo = max(0.0, min(f1, combos["f2-f1"]) - GEOM_PHANTOM_ZOOM_HZ)
    z_hi = max(f1, combos["f2-f1"]) + GEOM_PHANTOM_ZOOM_HZ
    zoom = _pool_band(d_freqs, d_mag, z_lo, z_hi)
    return {
        "kind": "phantom",
        "f1": f1, "f2": f2, "defect": defect,
        "combos": {k: float(v) for k, v in combos.items()},
        # Shown, never scored: the DISCRETE ladder — where *this string on this grid* puts its
        # partials, carrying the scheme's own spatial eigenvalue and theta dispersion (the continuum
        # sqrt(1 + B n^2) would fold both errors into the markers). The eye reads the combinations
        # landing in the gaps between them; f1/f2's ladder entries also sit slightly below the
        # measured ones, and that gap IS the amplitude hardening, visible for free.
        "ladder": _finite_list(
            dispersion.stiff_dispersion_frequencies(
                res.c, res.L, res.N, res.kappa_u, res.k, res.theta, np.arange(1, 6)
            ), 4
        ),
        "peaks": _finite_list(peaks, 4),
        "combo_err": combo_err,
        "dominance": dominance,
        "nearest_to_f1": near_f1,
        "nearest_to_f2": near_f2,
        "displacements": disp,
        "resolved": resolved,
        "linear": linear,
        "bridge_max": bridge_max,
        "n_peaks": int(peaks.size),
        "defect_min": GEOM_PHANTOM_DEFECT_MIN,
        "kappa": round(float(res.kappa_u), 3),
        "f_long1": round(f_long1, 1),
        "band": [0.0, round(GEOM_PHANTOM_BAND * f1, 3)],
        "zoom": [round(z_lo, 3), round(z_hi, 3)],
        "wide_freq": _finite_list(wide[0], 3) if wide else [],
        "wide_mag": _finite_list(wide[1], 6) if wide else [],
        "zoom_freq": _finite_list(zoom[0], 3) if zoom else [],
        "zoom_mag": _finite_list(zoom[1], 6) if zoom else [],
    }


def _geom_audio_block(run: _GeomRun, res: GeometricString, regime: str) -> dict[str, Any]:
    """``{audio, audio_note}``. Only the phantom regime has any, and only 0.1 s of it.

    The other three regimes ship ``audio: None`` rather than a stub the player would click on: their
    windows are set by what the *picture* needs, and stretching one to a listenable second is ~10
    minutes of compute (see the module-level regime comment).

    The phantom regime is the exception because its window is already 0.1 s of the **bridge force**
    ``EA v_x(0)`` — the channel that actually radiates in a piano — so the clip is free, already
    measured, and is the honest form of this model's long-promised audio. Two things it is NOT, both
    stated in ``audio_note`` rather than fixed:

    * **It is 0.1 s** — a blip, not a note. Longer is not a budget choice; it is the same ~22x
      oversampling that makes this model viz-only.
    * **It is dominated by the longitudinal startup transient**, not by the phantoms. ``v = 0`` is
      not the longitudinal equilibrium, so the run opens with a broadband radiating transient that
      is the largest feature in the *full* bridge spectrum. That is a property of the IC the test
      rig shares, and pre-solving it away would put a number on screen no test backs. The phantoms
      live 4.5x below it in frequency, which is why the panel band-limits and the ear does not.
    """
    if regime != "phantom" or run.bridge is None:
        return {
            "audio": None,
            "audio_note": (
                f"viz-only: c_long/c = {math.sqrt(res.EA / res.T):.0f}x, so resolving the "
                f"longitudinal wave forces fs = {res.fs / 1000:.0f} kHz and one second of sound "
                "would be ~10 minutes of compute."
            ),
        }
    audio48, peak = _resample_normalize(run.bridge[1:], res.fs)
    return {
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "audio_note": (
            f"the bridge force EA·v_x(0) — the piano's radiating channel — {GEOM_PHANTOM_WINDOW}s "
            f"of it (peak {peak:.3g} N). A blip, not a note: fs = {res.fs / 1000:.0f} kHz here, so "
            "a second would be ~10 minutes of compute. What you hear is mostly the longitudinal "
            f"startup transient at ~{res.c_long / (2 * res.L):.0f} Hz (v = 0 is not the "
            "longitudinal equilibrium); the phantoms are ~4.5x lower, where the panel looks."
        ),
    }


def _build_payload_geometric(p: dict[str, Any]) -> dict[str, Any]:
    regime = _geom_regime(p)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    probe_frac = _fnum(p, "pickup_position", 0.25)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < probe_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {probe_frac}.")

    kappa = _fnum(p, "kappa", 0.0)
    tongue: dict[str, Any] = {}
    diag: dict[str, Any] = {}

    if regime == "whirl":
        # kappa_u = 0 is FORCED, not a default: only the SOFT plane whirls (delta > 0 requires the
        # driven polarization to be the lower one). The same string driven on the stiff plane gives
        # 1.00x — the sharpest claim here, and no energy or amplitude measurement can make it.
        dt_over_t0 = _fnum(p, "dt_over_t0", GEOM_DT_DEFAULT)
        frac = _fnum(p, "tongue_position", GEOM_TONGUE_DEFAULT)
        if not (0.0 < dt_over_t0 <= GEOM_DT_MAX):
            raise ParamError(
                f"dT/T0 must be in (0, {GEOM_DT_MAX}], got {dt_over_t0:.3f}. Above that the DRIVEN "
                "polarization stops being single-mode (model #9's own planar parametric breakup), "
                "and the tongue oracle silently stops describing the run."
            )
        if not (0.0 <= frac <= GEOM_TONGUE_MAX):
            raise ParamError(f"tongue_position must be in [0, {GEOM_TONGUE_MAX}], got {frac}.")
        kappa = 0.0
    elif regime == "phantom":
        amplitude = _fnum(p, "amplitude", GEOM_PHANTOM_AMP_DEFAULT)
        if not (0.0 < amplitude <= 0.05):
            raise ParamError(f"amplitude must be in (0, 0.05] m, got {amplitude}.")
    else:
        amplitude = _fnum(p, "amplitude", GEOM_AMP_DEFAULT)
        if not (0.0 < amplitude <= 0.05):
            raise ParamError(f"amplitude must be in (0, 0.05] m, got {amplitude}.")

    # --- build + seed, per regime ----------------------------------------------------------------
    if regime == "whirl":
        probe_res = _build_geometric(p, kappa=0.0, kappa_w=0.0)   # cheap: only to read c/L/N back
        tg = _geom_tongue(
            c=probe_res.c, EA=probe_res.EA, T=probe_res.T, rho=probe_res.rho, L=probe_res.L,
            N=probe_res.N, dt_over_t0=dt_over_t0, frac=frac,
        )
        amplitude = tg["amplitude"]
        res = _build_geometric(p, kappa=0.0, kappa_w=tg["kappa_w"])
        shape = np.sin(np.pi * res.x / res.L)
        seed = _fnum(p, "seed_frac", GEOM_SEED_FRAC)
        seed_velocity = _as_bool(p.get("seed_velocity"), False)
        # seed = 0 is the HONESTY GATE: unseeded, at the tongue centre, max|w| is 0.0 bit-exact.
        # Without it every growth ratio here is partly measuring a leak. See GEOM_SEED_FRAC for why
        # both seed kinds exist and what each one is honest about.
        if seed_velocity:
            # The omega_u factor makes the two seeds comparable: it gives the velocity kick the same
            # initial out-of-plane DISPLACEMENT scale (~seed*A) the displacement seed starts with,
            # so switching kind changes the physics, not merely the size of the perturbation.
            res.set_state(amplitude * shape,
                          w_dot=seed * amplitude * math.sqrt(tg["omega0_sq"]) * shape)
        else:
            res.set_state(amplitude * shape, seed * amplitude * shape)
        f_osc = tg["omega"] / (2.0 * math.pi)
        n_steps = max(1, round(_fnum(p, "animation_window", 0.06) * res.fs))
        tongue = {
            "tongue_position": round(frac, 4), "dt_over_t0": round(dt_over_t0, 4),
            "amplitude": amplitude, "kappa_w": round(tg["kappa_w"], 3),
            "predicted_rate": tg["predicted_rate"], "seed_velocity": seed_velocity,
            "degenerate": bool(frac == 0.0),
            "in_tongue": bool(0.0 < frac < 0.5), "peak_at": 0.25,
        }
    elif regime == "rotating":
        res = _build_geometric(p, kappa=kappa, kappa_w=kappa)   # a helix needs a DEGENERATE string
        wave = solve_rotating_wave(
            L=res.L, T=res.T, rho=res.rho, EA=res.EA, fs=res.fs, N=res.N, theta=res.theta,
            amplitude=amplitude, mode=1, kappa=kappa,
        )
        # Assign the history DIRECTLY, never via set_state: set_state's y^{-1} is a 2nd-order Taylor
        # start — consistent, not exact — and its O(k^3) history error lands straight in the
        # longitudinal field. Exact history gives long_kin ~2e-26; set_state gives ~1e-16, which
        # still *looks* like machine precision while being ten orders worse.
        u0, w0, v0, up, wp, vp = rotating_wave_history(wave, fs=res.fs)
        res.u, res.w, res.v = u0, w0, v0
        res.u_prev, res.w_prev, res.v_prev = up, wp, vp
        f_osc = float(wave.frequency)
        n_steps = max(1, round(GEOM_ROTATING_PERIODS * res.fs / f_osc))
        # `converged` is reported, not assumed: the whole claim of this regime is that the seed
        # is an EXACT solution of the scheme, and a BVP that stopped early is simply not one —
        # the roundness would then measure the continuation's failure, not the physics.
        diag = {"bvp_frequency": round(f_osc, 4), "bvp_iterations": int(wave.iterations),
                "bvp_converged": bool(wave.converged)}
    elif regime == "phantom":
        # A DEGENERATE string (kappa_w = kappa_u): the motion is planar by construction, and planar
        # is the phantom's precondition, not an accident — a circular mode holds r^2 static and
        # pumps nothing (Tier A/3). Two modes, not one: a single mode gives only 2 f1, and the sum
        # and difference tones are the whole point — they are what cannot be mistaken for a harmonic
        # of anything.
        res = _build_geometric(p, kappa=kappa, kappa_w=kappa)
        sin1 = np.sin(np.pi * res.x / res.L)
        sin2 = np.sin(2.0 * np.pi * res.x / res.L)
        res.set_state(amplitude * (sin1 + sin2))
        dt_over_t0 = _geom_phantom_dt_over_t0(res, amplitude)
        if dt_over_t0 > GEOM_DT_MAX:
            raise ParamError(
                f"dT/T0 = {dt_over_t0:.2f} exceeds {GEOM_DT_MAX} at A = {amplitude:.4f} m: the "
                "two-mode motion stops being two modes (model #9's planar parametric breakup), and "
                "the partials f1, f2 the phantoms are measured against stop being clean single "
                "peaks. Lower the amplitude or EA."
            )
        f_osc = float(dispersion.stiff_dispersion_frequencies(
            res.c, res.L, res.N, res.kappa_u, res.k, res.theta, np.array([1])
        )[0])
        n_steps = max(1, round(GEOM_PHANTOM_WINDOW * res.fs))   # fixed window: physics, not a knob
        diag = {"dt_over_t0": round(dt_over_t0, 4)}
    else:                                                          # planar
        res = _build_geometric(p, kappa=kappa, kappa_w=kappa)
        res.set_state(amplitude * np.sin(np.pi * res.x / res.L))
        f_osc = float(modal.stiff_harmonic_frequencies(res.c, res.L, kappa, 1)[0])
        n_steps = max(1, round(_fnum(p, "animation_window", GEOM_PLANAR_WINDOW) * res.fs))

    work_max = GEOM_PHANTOM_WORK_MAX if regime == "phantom" else GEOM_WORK_MAX
    if n_steps > work_max:
        raise ParamError(
            f"work budget exceeded ({n_steps:,} steps > {work_max:,}): every step is a vector "
            "Newton solve over three coupled fields, and fs is forced ~22x higher than a "
            "transverse-only string's by the longitudinal wave (lam_long <= 1). "
            + ("The phantom window is fixed physics, so lower N or lam_long."
               if regime == "phantom" else "Lower N or the animation window.")
        )

    probe = min(max(1, round(probe_frac * res.N)), res.N - 1)
    anim_stride = max(1, round((res.fs / f_osc) / fpp))
    if n_steps // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_steps / MAX_FRAMES))
    run = _run_geometric(
        res, n_steps, probe=probe, anim_stride=anim_stride,
        modal=(sin1, sin2) if regime == "phantom" else None,
        long_kin=(regime == "rotating"),
    )

    frames = np.array(run.frames, dtype=float)                     # (n_frames, 3, N+1)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    e0 = float(run.E[0])
    sim = SimResult(
        time=np.arange(run.E.size) / res.fs, energy=run.E, output=None, fs=res.fs, snapshots=[],
    )
    sigma_zero = _fnum(p, "sigma0", 0.0) == 0.0 and _fnum(p, "sigma1", 0.0) == 0.0

    if regime == "whirl":
        spectrum_block: dict[str, Any] | None = _geom_whirl_block(run, res.fs, f_osc, tongue)
    elif regime == "phantom":
        spectrum_block = {**_geom_phantom_block(run, res), **diag}
    elif regime == "rotating":
        r = np.hypot(run.u_probe, run.w_probe)
        spectrum_block = {
            "kind": "rotating",
            "roundness": float((r.max() - r.min()) / r.mean()) if r.mean() > 0 else 0.0,
            # The longitudinal field does not MOVE (psi, the static stretch, is nonzero and held).
            "long_kin_over_e": float(run.long_kin / e0) if e0 > 0 else 0.0,
            **diag,
        }
    else:
        spectrum_block = {
            "kind": "planar",
            # Bit-exact zero: the w -> -w reflection symmetry, not a small number. Also the honesty
            # gate the whirl's growth ratios rest on.
            "max_w": float(np.max(run.w_max)),
            "exact_zero": bool(np.max(run.w_max) == 0.0),
        }

    return {
        "model": "geometric",
        "regime": regime,
        "fs_sim": round(res.fs, 3),
        "lambda": round(float(res.lam), 6),
        "lam_long": round(float(res.lam_long), 6),
        "grid": {"x": _finite_list(res.x, 6)},
        # dims stays 1 (a 1-D model) but there are THREE stacked fields per frame; the frontend
        # dispatches on `fields`, so the string path's (n_frames, width) contract is untouched.
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[2]) if frames.ndim == 3 else 0,
            "fields": ["u", "w", "v"],
            "dims": 1,
        },
        "frame_times": _finite_list(np.array(run.frame_steps, dtype=float) / res.fs, 6),
        "anim_dt": float(anim_stride / res.fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "orbit": _geom_orbit_block(run, int(frames.shape[0])),
        **_geom_audio_block(run, res, regime),
        "energy": _energy_block(sim, sigma_zero, 2.0 * _fnum(p, "sigma0", 0.0)),
        "meta": {
            "c": round(float(res.c), 3),
            "c_long": round(float(math.sqrt(res.EA / res.rho)), 1),
            "f1": round(f_osc, 3),
            "num_steps": int(n_steps),
            "n_frames": int(frames.shape[0]),
            "probe_x": round(float(res.x[probe]), 4),
            "spectrum": spectrum_block,
        },
    }


# == sympathetic / coupled strings (1D multi-field, model in core/connection.py) ==================
#
# The viewer's first MULTI-STRING model, and the second customer of the stacked-strip `drawFields`
# viz (the geometric string's u/w/v was the first). Frames are (n_frames, J, N+1) with
# `fields: ["string A", "string B"]` and `dims` still 1, exactly mirroring the geometric contract,
# so the string path stays bit-for-bit and `drawFields` generalizes from 3 fixed strips to J. See
# the SYMP_* constants block above for the two regimes and why energy is table-stakes here.


def _symp_regime(p: dict[str, Any]) -> str:
    regime = str(p.get("domain", "normal"))
    if regime not in SYMP_REGIMES:
        raise ParamError(f"regime must be one of {SYMP_REGIMES}, got {regime!r}.")
    return regime


def _build_sympathetic(
    p: dict[str, Any], *, detune_semitones: float = 0.0, k_default: float = SYMP_K_DEFAULT,
    sigma_body: float = 0.0,
) -> tuple[SympatheticStrings, float, float, int, float, float]:
    """Build a 2-string :class:`SympatheticStrings` on a shared modal body.

    ``fs = c0 N/(L lam)`` is set from string 0's tension (string 0 sits at the requested ``lam``);
    string 1 is detuned **downward** by ``detune_semitones`` (lower tension keeps its ``lam < 1``).
    ``k_default`` is the regime's fallback bridge stiffness when the request omits ``K`` (normal =
    8000, transfer = 1500, weinreich = 6000) — it MUST match the default the caller uses for the
    panel label, or a request with no ``K`` would run one stiffness while the panel claims another.
    ``sigma_body`` is the modal-body loss (the weinreich two-stage decay; ``0`` -> a lossless body,
    keeping the normal/transfer paths bit-identical). It only *adds* dissipation: the core stability
    guard is built from the lossless leapfrog operator (it uses ``omega``, never ``sigma``), so a
    lossy body cannot destabilise a config the ``sigma = 0`` guard passed. The core ctor is the
    single source of truth for the ``lam < 1`` guard and the exact dense stability bound — both
    raise ``ValueError``, surfaced upstream as a clean construction error.
    Returns ``(symp, c0, L, N, fs, lam)``.
    """
    L = _fnum(p, "L", 1.0)
    T = _fnum(p, "T", 200.0)
    rho = _fnum(p, "rho", 0.005)
    lam = _fnum(p, "lambda", SYMP_LAM_DEFAULT)
    K = _fnum(p, "K", k_default)
    try:
        N = int(p.get("N", 100))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= SYMP_N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {SYMP_N_MAX}] for sympathetic strings, got {N}.")
    if min(L, T, rho) <= 0:
        raise ParamError("L, T, rho must all be positive.")
    if not (0.0 < lam < 1.0):
        raise ParamError(
            f"lambda must be in (0, 1), got {lam}: the string's Nyquist mode is marginal at "
            "lambda = 1 and the bridge spring pushes it unstable, so the coupled system needs "
            "headroom below it."
        )
    if K < 0:
        raise ParamError(f"bridge stiffness K must be >= 0, got {K}.")

    c0 = math.sqrt(T / rho)
    fs = c0 * N / (L * lam)
    tensions = [T, T * 2.0 ** (-detune_semitones / 6.0)]   # f ~ sqrt(T), so a semitone is 2^(1/12)
    strings = [
        IdealString(L=L, T=Tj, rho=rho, fs=fs, N=N, boundary=("fixed", "free"))
        for Tj in tensions
    ]
    body = ModalBody(freqs=SYMP_BODY_FREQS, fs=fs, masses=SYMP_BODY_MASS, sigmas=sigma_body,
                     phi=1.0)
    symp = SympatheticStrings(strings=strings, body=body, Ks=[K, K])
    return symp, c0, L, N, fs, float(strings[0].lam)


class _SympRun:
    """Per-step telemetry of a sympathetic run — the fields, the shared bridge, and the energies."""

    def __init__(self, n: int, J: int, width: int) -> None:
        self.E = np.empty(n + 1)                 # total (strings + body + connections)
        self.wb = np.empty(n + 1)                # shared bridge displacement w_b
        self.e_body = np.empty(n + 1)            # body energy alone
        self.e_str = np.empty((J, n + 1))        # per-string energy (the transfer panel)
        self.pickup = np.empty(n + 1)            # string displacement at the pickup node (audio)
        self.frames: list[NDArray[np.float64]] = []
        self.frame_steps: list[int] = []
        self.width = width


def _run_sympathetic(
    symp: SympatheticStrings,
    n_steps: int,
    *,
    pickup_idx: int,
    anim_stride: int,
    frame_until: int,
    pickup_string: int = 0,
) -> _SympRun:
    """Step the coupled system, capturing everything the panels need in ONE pass.

    ``SympatheticStrings.state`` is string 0's field only, and :func:`simulate` gives neither the J
    stacked fields, nor ``w_b(t)``, nor the per-string energy — all three are what this model's
    panels are made of — so the run is hand-rolled (the geometric/mallet pattern). Frames (the J
    stacked displacement fields) are captured at ``anim_stride`` up to ``frame_until`` steps only
    (the normal regime animates a short window while its w_b trace runs long); the scalar telemetry
    is captured every step.
    """
    J = symp.J
    run = _SympRun(n_steps, J, symp.strings[0].N + 1)

    def _sample(i: int) -> None:
        run.E[i] = symp.energy()
        run.wb[i] = symp._bridge_displacement()
        run.e_body[i] = symp.body.energy()
        for j in range(J):
            run.e_str[j, i] = symp.string_energy(j)
        run.pickup[i] = float(symp.strings[pickup_string].u[pickup_idx])

    def _cap(i: int) -> None:
        run.frames.append(np.stack([s.u.copy() for s in symp.strings]))
        run.frame_steps.append(i)

    _sample(0)
    if frame_until >= 1:
        _cap(0)
    for i in range(1, n_steps + 1):
        symp.step()
        _sample(i)
        if i <= frame_until and i % anim_stride == 0:
            _cap(i)
    if not np.all(np.isfinite(run.E)):
        raise ParamError("simulation produced non-finite energy (instability) — adjust parameters.")
    return run


def _symp_finish(
    *,
    regime: str,
    frames_run: _SympRun,
    energy_E: NDArray[np.float64],
    pickup: NDArray[np.float64],
    fields: list[str],
    field_labels: list[str],
    grid_x: NDArray[np.float64],
    fs: float,
    lam: float,
    anim_stride: int,
    c0: float,
    f1: float,
    n_steps: int,
    probe_x: float,
    spectrum_block: dict[str, Any],
    playback_speed: float,
    energy_sigma_zero: bool = True,
    energy_decay_oracle: bool = True,
) -> dict[str, Any]:
    """Assemble the payload common to all regimes (frames + audio + the energy panel).

    ``energy_sigma_zero`` / ``energy_decay_oracle`` parametrise the energy verdict: the lossless
    normal/transfer regimes ride the conservation-drift check (``sigma_zero=True``), while the
    weinreich regime with a lossy body flips to passivity (``sigma_zero=False``,
    ``decay_oracle=False`` — its total energy is a two-rate decay to a nonzero aftersound floor,
    with no single-exponential oracle, exactly the mallet's ``decay_oracle=False`` case).
    """
    frames = np.array(frames_run.frames, dtype=float)             # (n_frames, J, N+1)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    sim = SimResult(
        time=np.arange(energy_E.size) / fs, energy=energy_E, output=None, fs=fs, snapshots=[],
    )
    audio48, peak = _resample_normalize(pickup, fs)
    return {
        "model": "sympathetic",
        "regime": regime,
        "fs_sim": round(fs, 3),
        "lambda": round(lam, 6),
        "grid": {"x": _finite_list(grid_x, 6)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[2]) if frames.ndim == 3 else 0,
            "fields": fields,
            "field_labels": field_labels,
            "dims": 1,
        },
        "frame_times": _finite_list(np.array(frames_run.frame_steps, dtype=float) / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        # normal/transfer are lossless (conservation-drift verdict); weinreich has a lossy body ->
        # passivity with no decay oracle (a two-rate decay to a nonzero floor). A closed undriven
        # system never needs the balance panel.
        "energy": _energy_block(sim, sigma_zero=energy_sigma_zero, oracle_2sigma=0.0,
                                decay_oracle=energy_decay_oracle),
        "meta": {
            "c": round(c0, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_steps),
            "n_frames": int(frames.shape[0]),
            "probe_x": round(probe_x, 4),
            "spectrum": spectrum_block,
        },
    }


def _build_payload_sympathetic(p: dict[str, Any]) -> dict[str, Any]:
    regime = _symp_regime(p)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    pickup_frac = _fnum(p, "pickup_position", 0.1)
    pluck_frac = _fnum(p, "pluck_position", 0.3)
    audio_dur = _fnum(p, "audio_duration", 2.0)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")
    if not (0.0 < pluck_frac < 1.0):
        raise ParamError(f"pluck_position must be in (0, 1), got {pluck_frac}.")
    if not (0.0 < audio_dur <= SYMP_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {SYMP_AUDIO_MAX}] s, got {audio_dur}.")

    if regime == "transfer":
        return _symp_transfer(p, playback_speed, pickup_frac, pluck_frac, audio_dur, fpp)
    if regime == "weinreich":
        return _symp_weinreich(p, playback_speed, pickup_frac, pluck_frac, audio_dur, fpp)
    return _symp_normal(p, playback_speed, pickup_frac, pluck_frac, audio_dur, fpp)


def _symp_normal(
    p: dict[str, Any], playback_speed: float, pickup_frac: float, pluck_frac: float,
    audio_dur: float, fpp: int,
) -> dict[str, Any]:
    """The discriminating oracle: run BOTH ICs and plot both bridge traces.

    The antisymmetric start ``u_B = -u_A`` keeps the shared bridge exactly still (``w_b == 0``
    bit-exact, ``E_body == 0``) forever — a claim energy conservation cannot see (it is automatic
    from the leapfrog structure). Its symmetric contrast ``u_B = +u_A`` swings the bridge and loads
    the body. The zero is only meaningful against that contrast, so both are run and both traces are
    shipped. The animation is the antisymmetric run (the two strings vibrate in exact antiphase over
    a dead bridge).
    """
    K = _fnum(p, "K", SYMP_K_DEFAULT)
    sy_anti, c0, L, N, fs, lam = _build_sympathetic(p)
    sy_sym, *_ = _build_sympathetic(p)
    f1 = c0 / (2.0 * L)
    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)
    x = sy_anti.strings[0].x
    pluck = triangular_pluck(x, L, pluck_frac * L, amplitude=SYMP_AMP)

    n_steps = max(1, round(audio_dur * fs))
    if 2 * n_steps > SYMP_WORK_MAX:
        raise ParamError(
            f"work budget exceeded ({2 * n_steps:,} steps > {SYMP_WORK_MAX:,}): the normal-mode "
            "regime runs twice (antisymmetric + its symmetric contrast). Lower the audio duration, "
            "N, or the tension."
        )
    frame_until = max(1, round(SYMP_NORMAL_ANIM_WIN * fs))
    anim_stride = max(1, round((fs / f1) / fpp))
    if frame_until // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(frame_until / MAX_FRAMES))

    sy_anti.strings[0].set_state(pluck)
    sy_anti.strings[1].set_state(-pluck)              # exact float negation -> bit-exact w_b == 0
    run_a = _run_sympathetic(sy_anti, n_steps, pickup_idx=pickup_idx, anim_stride=anim_stride,
                             frame_until=frame_until)
    sy_sym.strings[0].set_state(pluck)
    sy_sym.strings[1].set_state(pluck)                # symmetric: both push the bridge together
    run_s = _run_sympathetic(sy_sym, n_steps, pickup_idx=pickup_idx, anim_stride=1, frame_until=0)

    # The PLOTTED trace is a short window (see SYMP_TRACE_WINDOW) so the symmetric bridge's swings
    # are resolved rather than aliased into a band; anti_max / sym_max / the body fraction are still
    # measured over the FULL run.
    n_trace = min(n_steps, max(1, round(SYMP_TRACE_WINDOW * fs)))
    idx = np.linspace(0, n_trace, min(n_trace + 1, SYMP_TRACE_POINTS)).astype(int)
    e0 = float(run_a.E[0])
    anti_max = float(np.max(np.abs(run_a.wb)))
    sym_max = float(np.max(np.abs(run_s.wb)))
    spectrum_block: dict[str, Any] = {
        "kind": "sympathetic",
        "regime": "normal",
        "time": _finite_list(idx / fs, 6),
        "wb_anti": _finite_list(run_a.wb[idx]),
        "wb_sym": _finite_list(run_s.wb[idx]),
        "anti_max": anti_max,
        "sym_max": sym_max,
        "anti_exact_zero": bool(anti_max == 0.0),
        "body_frac_anti": float(np.max(run_a.e_body) / e0) if e0 > 0 else 0.0,
        "body_frac_sym": float(np.max(run_s.e_body) / e0) if e0 > 0 else 0.0,
        "K": round(K, 1),
    }
    return _symp_finish(
        regime="normal", frames_run=run_a, energy_E=run_a.E, pickup=run_a.pickup,
        fields=["string A", "string B"],
        field_labels=["string A — plucked +", "string B — antiphase −"],
        grid_x=x, fs=fs, lam=lam, anim_stride=anim_stride, c0=c0, f1=f1, n_steps=n_steps,
        probe_x=float(x[pickup_idx]), spectrum_block=spectrum_block, playback_speed=playback_speed,
    )


def _symp_transfer(
    p: dict[str, Any], playback_speed: float, pickup_frac: float, pluck_frac: float,
    audio_dur: float, fpp: int,
) -> dict[str, Any]:
    """Sympathetic transfer: pluck string A; a tuned neighbour drains the energy, a detuned one does
    not.

    A single run (string A plucked, string B at rest) over the full slosh. The per-string
    energy-fraction panel is the hero — the beat rate moves with the detune and the bridge K, so
    the panel carries it robustly — while the animation shows string A ringing down as string B
    rings up. Detune (semitones down, ``0`` = unison) is live here; at unison the two-string swap
    is near-complete, and it collapses as the neighbour is detuned off the partial.
    """
    K = _fnum(p, "K", SYMP_K_TRANSFER)
    detune = _fnum(p, "detune", 0.0)
    if not (0.0 <= detune <= 12.0):
        raise ParamError(f"detune must be in [0, 12] semitones, got {detune}.")
    sy, c0, L, N, fs, lam = _build_sympathetic(
        p, detune_semitones=detune, k_default=SYMP_K_TRANSFER,
    )
    f1 = c0 / (2.0 * L)
    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)
    x = sy.strings[0].x

    n_steps = max(1, round(audio_dur * fs))
    if n_steps > SYMP_WORK_MAX:
        raise ParamError(
            f"work budget exceeded ({n_steps:,} steps > {SYMP_WORK_MAX:,}). Lower the audio "
            "duration, N, or the tension."
        )
    anim_stride = max(1, round((fs / f1) / fpp))
    if n_steps // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_steps / MAX_FRAMES))

    sy.strings[0].set_state(triangular_pluck(x, L, pluck_frac * L, amplitude=SYMP_AMP))
    run = _run_sympathetic(sy, n_steps, pickup_idx=pickup_idx, anim_stride=anim_stride,
                           frame_until=n_steps)

    idx = np.linspace(0, n_steps, min(n_steps + 1, SYMP_TRACE_POINTS)).astype(int)
    frac0 = run.e_str[0] / run.E
    frac1 = run.e_str[1] / run.E
    spectrum_block: dict[str, Any] = {
        "kind": "sympathetic",
        "regime": "transfer",
        "time": _finite_list(idx / fs, 6),
        "frac0": _finite_list(frac0[idx]),
        "frac1": _finite_list(frac1[idx]),
        "peak_neighbour": float(np.max(frac1)),
        "detune": round(detune, 2),
        "tuned": bool(detune < 0.05),
        "K": round(K, 1),
    }
    return _symp_finish(
        regime="transfer", frames_run=run, energy_E=run.E, pickup=run.pickup,
        fields=["string A", "string B"],
        field_labels=["string A — plucked", "string B — sympathetic"],
        grid_x=x, fs=fs, lam=lam, anim_stride=anim_stride, c0=c0, f1=f1, n_steps=n_steps,
        probe_x=float(x[pickup_idx]), spectrum_block=spectrum_block, playback_speed=playback_speed,
    )


# -- weinreich two-stage decay: helpers + the regime builder ---------------------------------------
#
# The string energy E_A + E_B EXCLUDES E_conn and E_body, so it oscillates (string<->spring sloshing
# + sym/antisym beating) even in a lossless tail; a pointwise read or a raw polyfit measures that
# ripple, not the decay. So the trace is a sliding-MEAN envelope over one fundamental period (the
# whirl's sliding-envelope precedent, mean not max: max sits on the slosh peaks, the mean is the
# average stored string energy). E_str is also a SUM of two exponentials (a fast symmetric mode + a
# slow/flat antisymmetric one), so log(E_str) CURVES through the knee -- a naive early-window fit
# reads the curvature, not the prompt rate. The antisymmetric plateau is a real floor, so the prompt
# (fast) rate is fit on log(E_env - floor) over the steep part, and the aftersound (slow) rate on
# log(E_env) over the late window. None of the rates is a validated oracle (there is no closed form
# for the coupled modal decay over an off-harmonic body) -- they are reported, like the mallet's
# contact time. What IS a sharp claim is the aftersound rate at detune = 0: the antisymmetric mode
# bit-exactly bridge-decoupled (the normal-mode oracle), so it never loads the lossy body and its
# tail is lossless in the discrete scheme -> slope ~ 0, rising clearly with detune.


def _weinreich_envelope(e: NDArray[np.float64], win: int) -> NDArray[np.float64]:
    """Sliding-mean envelope of the (normalized) string energy over ``win`` samples (~1 period)."""
    return uniform_filter1d(np.asarray(e, dtype=float), size=max(1, int(win)), mode="nearest")


def _neg_log_slope(t: NDArray[np.float64], e: NDArray[np.float64]) -> float:
    """Decay rate ``-d/dt log(e)`` over a window (>= 0 for a decaying segment). 0 if too short."""
    e = np.asarray(e, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = e > (float(np.max(e)) * 1e-9 if e.size else 1.0)
    if mask.sum() < 2:
        return 0.0
    return float(-np.polyfit(t[mask], np.log(e[mask]), 1)[0])


def _weinreich_rates(
    t: NDArray[np.float64], env: NDArray[np.float64]
) -> tuple[float, float, float]:
    """``(prompt_rate, aftersound_rate, floor)`` from the string-energy envelope.

    ``floor`` is the tail-mean aftersound level. ``aftersound_rate`` is the late-window (last 40 %)
    slope -- the sharp claim (~0 at detune = 0). ``prompt_rate`` is the floor-subtracted early slope
    (over the steep part well above the floor), a report-only "how fast the prompt dies" number.
    """
    n = len(env)
    floor = float(np.mean(env[int(0.80 * n):])) if n else 0.0
    aftersound = _neg_log_slope(t[int(0.60 * n):], env[int(0.60 * n):]) if n else 0.0
    excess = env - floor
    steep = excess > 0.03
    prompt = _neg_log_slope(t[steep], excess[steep]) if steep.sum() > 3 else 0.0
    return prompt, aftersound, floor


def _symp_weinreich(
    p: dict[str, Any], playback_speed: float, pickup_frac: float, pluck_frac: float,
    audio_dur: float, fpp: int,
) -> dict[str, Any]:
    """Weinreich two-stage decay: strike ONE of two near-unison strings over a LOSSY bridge.

    Striking one string excites the symmetric and antisymmetric normal modes in equal measure. The
    symmetric mode loads the lossy body and dies fast (the "prompt"); the antisymmetric mode barely
    loads it and lingers (the "aftersound"), so the string-energy envelope shows the fast-then-slow
    knee of a real piano unison. Its contrast, run alongside, is plucking BOTH strings -- the pure
    symmetric mode, which loads the body fully and decays away single-slope with no aftersound (the
    "fix"). Both runs are needed: the strike-one plateau alone could be mistaken for a noise floor,
    but strike-both proves the plateau is the un-decaying antisymmetric mode (remove that excitation
    and the energy DOES decay away). The energy verdict is passivity with ``decay_oracle=False``
    (a two-rate decay to a nonzero floor; ``sigma_body = 0`` flips it back to the drift check).
    """
    K = _fnum(p, "K", SYMP_K_WEINREICH)
    sigma_body = _fnum(p, "sigma_body", SYMP_SIGMA_BODY_DEFAULT)
    detune = _fnum(p, "detune", 0.0)
    if not (0.0 <= sigma_body <= SYMP_SIGMA_BODY_MAX):
        raise ParamError(
            f"body loss must be in [0, {SYMP_SIGMA_BODY_MAX}], got {sigma_body}."
        )
    if not (0.0 <= detune <= SYMP_WEINREICH_DETUNE_MAX):
        raise ParamError(
            f"detune must be in [0, {SYMP_WEINREICH_DETUNE_MAX}] semitones for the weinreich "
            f"regime (a piano unison is mistuned by a few cents), got {detune}."
        )
    sy_one, c0, L, N, fs, lam = _build_sympathetic(
        p, detune_semitones=detune, k_default=SYMP_K_WEINREICH, sigma_body=sigma_body,
    )
    sy_both, *_ = _build_sympathetic(
        p, detune_semitones=detune, k_default=SYMP_K_WEINREICH, sigma_body=sigma_body,
    )
    f1 = c0 / (2.0 * L)
    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)
    x = sy_one.strings[0].x
    pluck = triangular_pluck(x, L, pluck_frac * L, amplitude=SYMP_AMP)

    n_steps = max(1, round(audio_dur * fs))
    if 2 * n_steps > SYMP_WORK_MAX:
        raise ParamError(
            f"work budget exceeded ({2 * n_steps:,} steps > {SYMP_WORK_MAX:,}): the weinreich "
            "regime runs twice (strike-one + its strike-both contrast). Lower the audio duration, "
            "N, or the tension."
        )
    anim_stride = max(1, round((fs / f1) / fpp))
    if n_steps // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_steps / MAX_FRAMES))

    sy_one.strings[0].set_state(pluck)                    # strike ONE (sym + antisym, equal parts)
    run_one = _run_sympathetic(sy_one, n_steps, pickup_idx=pickup_idx, anim_stride=anim_stride,
                               frame_until=n_steps)
    sy_both.strings[0].set_state(pluck)
    sy_both.strings[1].set_state(pluck)                   # strike BOTH: the pure symmetric mode
    run_both = _run_sympathetic(sy_both, n_steps, pickup_idx=pickup_idx, anim_stride=1,
                                frame_until=0)

    t_full = np.arange(n_steps + 1) / fs
    win = max(1, round(fs / f1))
    e_one = run_one.e_str[0] + run_one.e_str[1]
    e_both = run_both.e_str[0] + run_both.e_str[1]
    env_one = _weinreich_envelope(e_one / float(e_one[0]), win)
    env_both = _weinreich_envelope(e_both / float(e_both[0]), win)
    prompt_rate, aftersound_rate, floor_one = _weinreich_rates(t_full, env_one)

    idx = np.linspace(0, n_steps, min(n_steps + 1, SYMP_TRACE_POINTS)).astype(int)
    spectrum_block: dict[str, Any] = {
        "kind": "sympathetic",
        "regime": "weinreich",
        "time": _finite_list(idx / fs, 6),
        "env_one": _finite_list(env_one[idx]),
        "env_both": _finite_list(env_both[idx]),
        "prompt_rate": round(prompt_rate, 3),
        "aftersound_rate": round(aftersound_rate, 4),
        "floor_one": round(floor_one, 4),
        "both_final": round(float(env_both[-1]), 4),
        "sigma_body": round(sigma_body, 2),
        "detune": round(detune, 3),
        "K": round(K, 1),
        "sigma_zero": bool(sigma_body == 0.0),
    }
    return _symp_finish(
        regime="weinreich", frames_run=run_one, energy_E=run_one.E, pickup=run_one.pickup,
        fields=["string A", "string B"],
        field_labels=["string A — struck", "string B — silent → sympathetic"],
        grid_x=x, fs=fs, lam=lam, anim_stride=anim_stride, c0=c0, f1=f1, n_steps=n_steps,
        probe_x=float(x[pickup_idx]), spectrum_block=spectrum_block, playback_speed=playback_speed,
        energy_sigma_zero=bool(sigma_body == 0.0), energy_decay_oracle=False,
    )


# == jawari / buzzing bridge (model #8 in its curved configuration) ================================
#
# The sitar & tanpura bridge. NOT new core physics: a :class:`BarrierString` (model #8, string
# against a one-sided distributed nonlinear barrier) whose barrier is a *parabola* hugging the
# termination. The string wraps onto the curve each downswing and its departure point travels along
# it, re-injecting high partials — the shimmer. Load-bearing decisions, all measured before wiring:
#
#   * **The claim is the shimmer, not energy.** Energy conservation through contact passes for a
#     flat rail too (model #8 already gates it) — table-stakes green, not what makes it a jawari.
#     The headline is the LATE-window spectral-centroid elevation over a clean string, and the
#     second panel is the travelling wrap edge.
#   * **The energy panel is the damped string's, unchanged — `decay_oracle` stays TRUE.** The bridge
#     is a *lossless elastic* barrier: it redistributes energy into the highs but dissipates none,
#     so every mode still decays at exactly ``2 sigma0`` and the flat-loss oracle holds (measured
#     2sigma = 1.009 against oracle 1.000). This is the OPPOSITE of the mallet (a closed system on a
#     ½Mv₀² floor) and of weinreich (two rates to a nonzero floor), which is why neither template
#     applies. ``sigma0`` gates the verdict: 0 -> drift *through* the curved wrap (1.0e-12), > 0 ->
#     passivity + the 2sigma oracle.
#   * **`sigma1` and `hysteresis` are fixed at 0, deliberately not exposed.** Both would break the
#     oracle above for nothing: sigma1 makes the highs decay faster than 2*sigma0 (measured
#     2.77 against a 1.00 oracle) and hysteresis is contact *damping*, which fights the very
#     re-injection the model is about. The shimmer needs neither (the 3.4x headline is at sigma1=0).
#   * **The IC is a mode-1 sine and there is NO pluck-position slider.** The headline is a CONTRAST
#     against a clean string, so the clean baseline must be spectrally pure: a mode-1 start puts the
#     clean centroid on f1 and the bridge's highs stand out. A raised-cosine pluck gives the clean
#     string its own highs and shrinks the contrast (the #6 mode11 / #9 single-mode lesson again).
#   * **`amplitude / depth` is the control, not `depth`.** What decides whether the string wraps is
#     the near-termination downswing (~``amplitude * pi * width_frac``) against the curve's drop:
#     the dimensionless ``downswing / depth`` is refinement- and units-invariant (the tension
#     ``dT/T0`` / geometric ``frac`` pattern). Measured, it is a ONE-SIDED FLOOR, not a window:
#     ratio 15.1/7.5/3.8/1.9/0.9/0.6 -> elevation 3.48/3.75/3.44/3.15/2.33/1.63x, and the wrap
#     contracts toward the crest (max node 14 -> 11 -> 8) as it falls. Below the floor the bridge
#     grazes: a legitimate stiff point contact, just not a jawari -> **LABEL, never FAIL** (the
#     bow's Schelleng-window rule).
#   * **Only the jawari and clean runs are paid for.** The flat-rail control that separates "buzzes"
#     from "travels" is already validated in ``tests/test_jawari.py`` (wrap-edge std 4.89 curve vs
#     2.35 flat); the panel cites that rather than paying for a third run, and the shipped sweep
#     (nodes 0..14) is self-evidently travelling.


JAWARI_N_MAX = 128            # support ~= width_frac*N nodes; the dense contact solve is |C|x|C|
JAWARI_AUDIO_MAX = 1.5        # ~143 us/step at fs = c*N/(L*lam) with lam = 0.4 -> fs = 50 kHz
JAWARI_WORK_MAX = 150_000     # total steps across BOTH runs (jawari + the clean contrast)
JAWARI_LAM_DEFAULT = 0.4      # sub-unity: the coupled contact solve wants headroom below Nyquist
JAWARI_K_DEFAULT = 2.0e6      # N/m^alpha — stiff wood/bone bridge
JAWARI_ALPHA_DEFAULT = 1.5    # Hertzian-ish contact exponent
JAWARI_WIDTH_DEFAULT = 0.15   # bridge span as a fraction of L, hugging the termination
JAWARI_DEPTH_DEFAULT = 1.0e-3         # crest-to-far-edge drop of the parabola (m)
JAWARI_AMP_DEFAULT = 8.0e-3           # mode-1 amplitude (m) — the test suite's AMP
JAWARI_SIGMA0_DEFAULT = 0.5           # loss defaults ON: "sustained" is meaningless without decay
JAWARI_DEPTH_MAX = 8.0e-3
JAWARI_AMP_MAX = 4.0e-2
JAWARI_WIDTH_MAX = 0.4
JAWARI_ELEVATION_GATE = 2.5   # late-window centroid elevation over the clean string (the headline)
JAWARI_RATIO_FLOOR = 1.5      # downswing/depth below which the bridge grazes — LABEL, not FAIL


def _jawari_profile(x: NDArray[np.float64], L: float, *, width_frac: float, depth: float,
                    clearance: float) -> NDArray[np.float64]:
    """Parabolic bridge hugging ``x = 0``: ``b = -clearance - depth*(x/d)^2`` on ``0 < x <= d``.

    ``-inf`` off the span (out of support). Mirrors ``tests/helpers.jawari_barrier`` exactly so the
    viewer runs the validated geometry rather than a lookalike.
    """
    d = width_frac * L
    b = np.full_like(np.asarray(x, dtype=float), -np.inf)
    on = (x > 0.0) & (x <= d)
    b[on] = -clearance - depth * (x[on] / d) ** 2
    return b


def _build_jawari(p: dict[str, Any], *, clearance: float) -> tuple[BarrierString, dict[str, Any]]:
    """Construct a jawari :class:`BarrierString` from params (the clean contrast passes a big
    ``clearance``, which drops the whole curve out of the string's reach).

    Returns ``(bar, info)`` where ``info`` carries the derived scalars the panels quote.
    """
    L = _fnum(p, "L", 1.0)
    T = _fnum(p, "T", 200.0)
    rho = _fnum(p, "rho", 0.005)
    lam = _fnum(p, "lambda", JAWARI_LAM_DEFAULT)
    # Deliberately NOT `K` and NOT `alpha`. Both names are already taken by other models with wildly
    # different scales and meanings — `K` is the sympathetic bridge SPRING (~8000 N/m; this one
    # is 2e6 N/m^alpha, 250x stiffer) and `alpha` is the mallet's felt exponent (2.3 vs 1.5 here).
    # The frontend sends every slider, hidden ones included, so reading either would silently
    # render a
    # different bridge the moment the user had visited those models. A distinct name is the fix; the
    # contact exponent is not a headline control here, so it stays fixed at the validated 1.5 rather
    # than earning a slider (and a third chance to leak).
    K = _fnum(p, "bridge_stiffness", JAWARI_K_DEFAULT)
    alpha = JAWARI_ALPHA_DEFAULT
    width_frac = _fnum(p, "width_frac", JAWARI_WIDTH_DEFAULT)
    depth = _fnum(p, "depth", JAWARI_DEPTH_DEFAULT)
    sigma0 = _fnum(p, "sigma0", JAWARI_SIGMA0_DEFAULT)
    try:
        N = int(p.get("N", 100))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= JAWARI_N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {JAWARI_N_MAX}] for the jawari, got {N}.")
    if min(L, T, rho) <= 0:
        raise ParamError("L, T, rho must all be positive.")
    if not (0.0 < lam < 1.0):
        raise ParamError(
            f"lambda must be in (0, 1), got {lam}: the coupled contact solve needs headroom below "
            "the string's marginal Nyquist mode."
        )
    if not (0.0 < width_frac <= JAWARI_WIDTH_MAX):
        raise ParamError(f"width_frac must be in (0, {JAWARI_WIDTH_MAX}], got {width_frac}.")
    if not (0.0 < depth <= JAWARI_DEPTH_MAX):
        raise ParamError(f"depth must be in (0, {JAWARI_DEPTH_MAX}] m, got {depth}.")
    if K <= 0:
        raise ParamError(f"bridge_stiffness must be positive, got {K}.")
    if sigma0 < 0:
        raise ParamError(f"sigma0 must be >= 0, got {sigma0}.")

    c = math.sqrt(T / rho)
    fs = c * N / (L * lam)
    # sigma1 = hysteresis = 0 on purpose: both would break the flat-loss 2*sigma0 oracle, and the
    # shimmer needs neither (see the section note above).
    string = DampedStiffString(L=L, T=T, rho=rho, fs=fs, N=N, kappa=0.0, sigma0=sigma0,
                               sigma1=0.0)
    barrier = _jawari_profile(string.x, L, width_frac=width_frac, depth=depth,
                              clearance=clearance)
    bar = BarrierString(string=string, barrier=barrier, stiffness=K, alpha=alpha, hysteresis=0.0)
    info = {
        "c": c, "L": L, "N": N, "fs": fs, "lam": float(string.lam), "sigma0": sigma0,
        "width_frac": width_frac, "depth": depth, "K": K, "alpha": alpha,
        "support": int(np.isfinite(barrier).sum()),
    }
    return bar, info


class _JawariRun:
    """Per-step telemetry of a jawari run — the pickup, the wrap edge and the energy."""

    def __init__(self, n: int) -> None:
        self.E = np.empty(n + 1)
        self.pickup = np.empty(n + 1)
        self.wrap = np.full(n + 1, -1.0)      # furthest-in-contact node; -1 = clear of the bridge
        self.n_contact = np.zeros(n + 1)
        self.frames: list[NDArray[np.float64]] = []
        self.frame_steps: list[int] = []


def _run_jawari(bar: BarrierString, n_steps: int, *, pickup_idx: int, anim_stride: int,
                frame_until: int, capture_wrap: bool = True) -> _JawariRun:
    """Step the barrier string, capturing pickup + wrap edge + energy + frames in ONE pass.

    Hand-rolled rather than :func:`simulate` because ``contact_mask()`` — the travelling departure
    point, which is the second panel — is not part of a ``SimResult`` (the geometric/mallet/
    sympathetic pattern).
    """
    run = _JawariRun(n_steps)

    def _sample(i: int) -> None:
        run.E[i] = bar.energy()
        run.pickup[i] = bar.string.displacement_at(pickup_idx)
        if capture_wrap:
            m = bar.contact_mask()
            n_c = int(np.count_nonzero(m))
            run.n_contact[i] = n_c
            if n_c:
                run.wrap[i] = float(np.max(np.where(m)[0]))

    _sample(0)
    if frame_until >= 1:
        run.frames.append(bar.string.u.copy())
        run.frame_steps.append(0)
    for i in range(1, n_steps + 1):
        bar.step()
        _sample(i)
        if i <= frame_until and i % anim_stride == 0:
            run.frames.append(bar.string.u.copy())
            run.frame_steps.append(i)
    if not np.all(np.isfinite(run.E)):
        raise ParamError("simulation produced non-finite energy (instability) — adjust parameters.")
    return run


def _spectral_centroid(sig: NDArray[np.float64], fs: float) -> float:
    """Amplitude-weighted mean frequency (Hz) — the brightness proxy, identical to the suite's."""
    sig = np.asarray(sig, dtype=float)
    if sig.size < 4:
        return float("nan")
    mag = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    total = float(np.sum(mag))
    return float(np.sum(freqs * mag) / total) if total > 0 else float("nan")


def _jawari_band_spectrum(sig: NDArray[np.float64], fs: float, f_max: float,
                          n_points: int = 240) -> tuple[list[float], list[float | None], float]:
    """Band-limited magnitude spectrum, pooled to ``n_points`` for the plot. Returns (f, mag, norm).

    Pooling is by MAX over each bin (the shimmer is a forest of narrow high partials; a mean would
    average them into the noise floor). The normalizer is returned so the jawari and clean traces
    can share one scale — their *relative* height is the claim.
    """
    sig = np.asarray(sig, dtype=float)
    mag = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    keep = freqs <= f_max
    freqs, mag = freqs[keep], mag[keep]
    if freqs.size <= n_points:
        return _finite_list(freqs, 3), _finite_list(mag), float(mag.max() if mag.size else 1.0)
    edges = np.linspace(0, freqs.size, n_points + 1).astype(int)
    fo = np.array([freqs[a:b].mean() for a, b in zip(edges[:-1], edges[1:], strict=True)
                   if b > a])
    mo = np.array([mag[a:b].max() for a, b in zip(edges[:-1], edges[1:], strict=True)
                   if b > a])
    return _finite_list(fo, 3), _finite_list(mo), float(mo.max() if mo.size else 1.0)


def _jawari_shimmer_block(jaw: _JawariRun, clean: _JawariRun, fs: float, f1: float,
                          info: dict[str, Any], amplitude: float) -> dict[str, Any]:
    """The claim: late-window brightness held up by the bridge, and the wrap edge travelling.

    Both windows are halves of the SAME run, so the early/late comparison needs no second
    simulation. The late-window *elevation* over the clean string is the robust headline; the
    late/early *ratio* is reported but deliberately never gated — it wobbles 0.9-1.3 with the decay
    rate and window placement (recorded in the model's own notes), so gating it would make a correct
    render flaky.
    """
    half = jaw.pickup.size // 2
    j_e = _spectral_centroid(jaw.pickup[:half], fs)
    j_l = _spectral_centroid(jaw.pickup[half:], fs)
    c_e = _spectral_centroid(clean.pickup[:half], fs)
    c_l = _spectral_centroid(clean.pickup[half:], fs)
    elevation = float(j_l / c_l) if c_l > 0 else float("nan")

    f_max = min(0.45 * fs, max(20.0 * f1, 2000.0))
    jf, jm, jn = _jawari_band_spectrum(jaw.pickup[half:], fs, f_max)
    cf, cm, cn = _jawari_band_spectrum(clean.pickup[half:], fs, f_max)
    norm = max(jn, cn) or 1.0
    jm = [None if v is None else v / norm for v in jm]
    cm = [None if v is None else v / norm for v in cm]

    contacting = jaw.wrap >= 0.0
    wrap = jaw.wrap[contacting]
    downswing = amplitude * math.pi * info["width_frac"]
    ratio = float(downswing / info["depth"]) if info["depth"] > 0 else float("inf")

    return {
        "kind": "jawari",
        "centroid": {
            "jawari_early": round(j_e, 1), "jawari_late": round(j_l, 1),
            "clean_early": round(c_e, 1), "clean_late": round(c_l, 1),
        },
        "elevation": round(elevation, 3),
        "elevation_gate": JAWARI_ELEVATION_GATE,
        "shimmering": bool(elevation > JAWARI_ELEVATION_GATE),
        # report-only, never gated (see the docstring)
        "sustain_ratio": round(float(j_l / j_e), 3) if j_e > 0 else None,
        "clean_sustain_ratio": round(float(c_l / c_e), 3) if c_e > 0 else None,
        "downswing": downswing,
        "depth": info["depth"],
        "ratio": round(ratio, 2),
        "ratio_floor": JAWARI_RATIO_FLOOR,
        "grazing": bool(ratio < JAWARI_RATIO_FLOOR),
        "wrap": {
            "std": round(float(np.std(wrap)), 2) if wrap.size else None,
            "min_node": int(wrap.min()) if wrap.size else None,
            "max_node": int(wrap.max()) if wrap.size else None,
            "support": int(info["support"]),
            "duty": round(float(contacting.mean()), 4),
            "mean_nodes": round(float(jaw.n_contact[contacting].mean()), 2) if wrap.size else None,
            # the flat-rail control is validated in tests/test_jawari.py, not re-run here
            "flat_rail_std": 2.35,
            "curve_std_suite": 4.89,
        },
        "spectra": {
            "f_max": round(f_max, 1),
            "jawari": {"f": jf, "mag": jm},
            "clean": {"f": cf, "mag": cm},
        },
        "f1": round(f1, 3),
    }


def _build_payload_jawari(p: dict[str, Any]) -> dict[str, Any]:
    playback_speed = _fnum(p, "playback_speed", 0.02)
    pickup_frac = _fnum(p, "pickup_position", 0.5)
    audio_dur = _fnum(p, "audio_duration", 0.24)
    anim_win = _fnum(p, "animation_window", 0.06)
    amplitude = _fnum(p, "amplitude", JAWARI_AMP_DEFAULT)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")
    if not (0.0 < audio_dur <= JAWARI_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {JAWARI_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < amplitude <= JAWARI_AMP_MAX):
        raise ParamError(f"amplitude must be in (0, {JAWARI_AMP_MAX}] m, got {amplitude}.")

    jaw, info = _build_jawari(p, clearance=0.0)
    fs, L, c, N = info["fs"], info["L"], info["c"], info["N"]
    f1 = c / (2.0 * L)
    n_steps = max(1, round(audio_dur * fs))
    # BOTH runs are paid for (the clean contrast is what makes the elevation a number), so the work
    # budget counts them together — a per-run cap would silently licence twice the wall clock.
    if 2 * n_steps > JAWARI_WORK_MAX:
        raise ParamError(
            f"this configuration needs {2 * n_steps} steps across the jawari and clean runs "
            f"(budget {JAWARI_WORK_MAX}): every step is a vector contact solve over the bridge "
            f"support. Shorten audio_duration, lower N, or raise lambda."
        )

    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)
    anim_stride = max(1, round((fs / f1) / fpp))
    n_anim = min(n_steps, max(anim_stride, round(anim_win * fs)))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))

    shape = _mode1_shape(jaw.string) * amplitude
    jaw.set_state(shape)
    run = _run_jawari(jaw, n_steps, pickup_idx=pickup_idx, anim_stride=anim_stride,
                      frame_until=n_anim)

    # The clean contrast: the SAME string with the bridge dropped out of reach. Sharing the string
    # parameters is what makes the centroid ratio attributable to the bridge and nothing else.
    clean, _ = _build_jawari(p, clearance=1.0)
    clean.set_state(_mode1_shape(clean.string) * amplitude)
    clean_run = _run_jawari(clean, n_steps, pickup_idx=pickup_idx, anim_stride=1, frame_until=0,
                            capture_wrap=False)

    frames = np.array(run.frames, dtype=float)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    audio48, peak = _resample_normalize(run.pickup, fs)
    sim = SimResult(time=np.arange(run.E.size) / fs, energy=run.E, output=None, fs=fs, snapshots=[])
    # THE INDEXING TRAP. `BarrierString._b` and `contact_mask()` are both over the SUPPORT (the
    # ~15 nodes the finite barrier covers), not over the grid — so shipping either straight to the
    # frontend, which indexes everything by grid node, silently misplaces it. The barrier would
    # arrive as a 15-long array drawn across the first 15 nodes, and the wrap marker would land one
    # node short of the contact it marks (off by exactly the first support offset, which near a
    # termination looks entirely plausible). Both are scattered back onto the grid here.
    barrier = np.full(jaw.string.N + 1, np.nan)
    barrier[jaw._support] = jaw._b
    support_nodes = np.asarray(jaw._support, dtype=int)
    wrap_grid = [int(support_nodes[int(w)]) if w >= 0 else -1
                 for w in (run.wrap[i] for i in run.frame_steps)]

    return {
        "model": "jawari",
        "fs_sim": round(fs, 3),
        "lambda": round(info["lam"], 6),
        "grid": {"x": _finite_list(jaw.string.x, 6), "barrier": _finite_list(barrier)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(np.array(run.frame_steps, dtype=float) / fs, 6),
        # grid-node indices (see the scatter above); the wrap STATISTICS in meta stay
        # support-relative, because that is the frame the model's own tests report them in.
        "wrap_frames": wrap_grid,
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        # The damped string's own verdict, unchanged: the bridge is elastic and lossless, so the
        # flat-loss 2*sigma0 oracle survives the wrap (see the section note).
        "energy": _energy_block(sim, sigma_zero=bool(info["sigma0"] == 0.0),
                                oracle_2sigma=2.0 * info["sigma0"]),
        "meta": {
            "c": round(c, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_steps),
            "n_frames": int(frames.shape[0]),
            "probe_x": round(float(jaw.string.x[pickup_idx]), 4),
            "bridge_span": round(info["width_frac"] * L, 4),
            "spectrum": _jawari_shimmer_block(run, clean_run, fs, f1, info, amplitude),
        },
    }


# == fret / flat rail (model #8 on its OWN terms) ==================================================
#
# Batch 8 shipped the jawari, which is a *configuration* of model #8 — a curved barrier hugging the
# termination, onto which the string wraps and stays. The general case is the **flat rail / fret**:
# string-fret buzz, prepared-piano rattle. It is the jawari's physical OPPOSITE — not one departure
# point gliding persistently along a curve, but **slap-and-release**. That contrast is the batch's
# structural point; without it the batch collapses into "the jawari with a flatter barrier".
# All-wrapper again: ``physsynth/core`` stays untouched, and the geometry is the one
# ``tests/test_collision_signature.py`` already validates (flat rail, ``lam = 0.4``).
#
# Load-bearing decisions, ALL measured before wiring (probe ``temp/fret-viewer-probe/``):
#
#   * **The gated claim is INTERMITTENCY.** At the default (flat rail, clearance 2 mm, sigma0 = 0.5,
#     N = 100) the string makes **1.24 contact episodes per period at a 15.4 % duty** — it slaps and
#     springs off, never pinned. So the money panel is a **contact raster** (an x-vs-t spark map),
#     which next to the shipped jawari reads instantly as the opposite regime. Everything else on
#     screen — the decay reading, the brightness, the pitch — is diagnostic (see below).
#   * **The active set is genuinely a VECTOR, and that is the second number on screen.** Of m = 99
#     support nodes, up to **69 are simultaneously in contact** (mean 42 while touching), at
#     **Newton iters max 2, mean 1.16**. That cheapness IS the ``lambda_min(J) >= 1`` proof showing
#     up as a measurement: unique root, global convergence, no branch-picking.
#   * **THE JAWARI'S 2*sigma0 DECAY ORACLE DOES NOT TRANSFER — and the reason is physics.** The rail
#     is *equally* lossless (sigma = 0 drift 7.2e-13 through genuine contact) yet decays 6-9 % fast.
#     The loss identity is ``dE/dt = -2 sigma0 (2 KE)``, so "rate == 2 sigma0" is really an
#     **equipartition** assumption (``<KE> = E/2``) — true for harmonic motion, false for a string
#     being slapped by a stiff one-sided spring. Measured ``<2KE/E>`` (with the **centered**
#     velocity the identity actually uses) against the measured rate ratio: 1.0000/0.9999,
#     1.0069/1.0086, 1.0725/1.0724, 1.0872/1.0875 at out-of-reach / 4 / 2 / 1 mm — agreeing to
#     0.01-0.17 %. This
#     also explains the jawari's 1.009 retroactively: its gentle wrap stays near-equipartitioned, so
#     the naive oracle passed there by ACCIDENT rather than by transfer. *Generalizable: an oracle
#     that holds in one configuration of a model may be passing on an accident of that
#     configuration — re-measure it per configuration instead of inheriting it.*
#   * **...but that identity is a CONSISTENCY CHECK, not a gate.** Both sides come from one run of
#     one identity, so what their agreement certifies — "barrier lossless + accounting intact" — is
#     already certified at machine precision by the sigma = 0 drift and the sigma > 0 passivity. And
#     the agreement is NOT uniform: with the *backward* velocity the alpha = 1 case disagrees by
#     3.0 %, collapsing to 0.41 % under centering (the residual is the known alpha = 1
#     discrete-gradient ``(1/2, 0, 1/2)`` weighting against the string's theta-centering).
#     Calibrating
#     a tight tolerance on the pretty 0.01 % number would ship a test failing at other alpha/lam/N
#     for non-bug reasons. So: **gate on drift (sigma = 0) + passivity (sigma > 0), and report the
#     rate as a diagnostic TRIPLE** — rate, 2*sigma0, and 2*sigma0*<2KE/E> side by side, which
#     *shows* the physics without pretending to be a precision oracle. ``decay_oracle`` is therefore
#     FALSE here and TRUE for the jawari — the one place the two configurations of model #8 differ.
#   * **BRIGHTNESS IS NON-MONOTONE IN CLEARANCE — the panel must not claim "closer is brighter".**
#     Centroid elevation over an out-of-reach control: 2.50x at 4 mm, 2.71x at 3 mm, **3.33x at
#     2 mm**, 2.59x at 1 mm, 2.83x at 0.5 mm — it PEAKS at an intermediate clearance, falling as
#     the string starts to pin. ``test_closer_barrier_is_brighter`` compares 4 mm vs 1 mm only, and
#     by a hair (2.50 vs 2.59), so a monotone panel label would be **disproved by its own slider**.
#     Reported with the peak named; the monotone claim stays scoped to the test's two-point pair.
#     *Generalizable: a signature validated on a two-point comparison is not a monotone law — the
#     slider samples the range the test never did.*
#   * **PITCH IS DIAGNOSTIC-ONLY — do NOT build a shortened-length oracle.** Measured +69/+225/+598/
#     +690/+1271 cents at clearance 4/3/2/1/0.5 mm: large and real, but ``f = c/(2 L_eff)`` would
#     OVERCLAIM: ``test_contact_is_intermittent`` is explicit that the string is *not* pinned
#     to a shorter length. It is shipped under the name of the thing actually measured — the upward
#     **zero-crossing rate** — because a buzz adds crossings within a period, so calling it "pitch"
#     would launder a mixed quantity into a note name. Gates nothing (the Schelleng-window /
#     grazing-ratio precedent, fourth customer).
#   * **The static-equilibrium magnitude oracle is CITED, not led.** ``S u* = (K/rho) b`` held to
#     3.4e-15 is model #8's exact money test and has never been in the viewer — but it is alpha = 1,
#     non-musical, and its "result" is a flat line that does not move. It rides the verdict panel as
#     a magnitude credential, never as a co-headline beside the buzz. Lead audible, cite the oracle.
#   * **RAIL_FRAC CAN SILENTLY KILL THE CLAIM, AND IT IS A SLIDER.** Peak swing over the rail falls
#     as the rail shortens toward the nut (5.33/5.00/2.94/1.24 mm at rail_frac 1.0/0.5/0.2/0.08), so
#     at the default 2 mm clearance the rail goes OUT OF REACH: duty 15.3/18.2/8.8/**0.0**/0.0 % at
#     rail_frac 1.0/0.5/0.2/0.12/0.08, with zero episodes below ~0.15. A user dragging one slider
#     reaches a blank raster and an empty claim with nothing wrong. Hence the floor at 0.2, enforced
#     **here** and not merely as a slider min (the frontend sends what it likes). Out of reach is
#     still LABELLED, never failed — it is a correct render of a fret nobody can touch.
#   * **Cost: the most expensive model in the viewer per second of audio, and the dense solve is NOT
#     the reason.** Measured 189/201/257/261/320 us/step at N = 64/80/96/100/128, ~2x the jawari's
#     143. Halving the rail span halves m but buys only ~20 %, so the ``|C|x|C|`` solve is not
#     dominant — the string step and the rank-m correction are. Since ``fs = cN/(L lam)`` also buys
#     the sample rate, budget the PRODUCT: 1 s of audio costs 6 s (N=64) -> 13 s (N=100) -> 20 s
#     (N=128) of wall clock. And the out-of-reach control is **0.95x** the fret run's cost per step
#     (no contact solve to skip — same reason), so keeping it live costs ~1.95x. It is kept
#     but **separately bounded and short**: the elevation is a spectral centroid and does not need
#     the full render. ``FRET_WORK_MAX`` counts ``n_audio + n_control`` (the jawari/batch-9 rule).
#     *Absolute us/step drifts run to run on this machine (331 vs 530 for one config, one session),
#     so budget on the committed figures and measured ratios, never on a single fresh timing.*
#   * **Slider names must not collide — the leak family, sixth member.** ``K`` is the sympathetic
#     bridge spring, ``alpha`` the mallet's felt exponent, ``bridge_stiffness``/``depth``/
#     ``width_frac`` the jawari's. The rail takes its own ``clearance``/``rail_frac``/
#     ``rail_stiffness`` — and every one resets in the frontend's ``_default``, or a visit to
#     another model silently re-renders a different fret.

FRET_N_MAX = 100              # ~261 us/step at N=100; see the cost note above
FRET_AUDIO_MAX = 0.6          # s — 0.6 s at N=100 is ~30k steps ~= 9 s wall
# The control is SHORT, exactly as the plan settled: the elevation is a spectral centroid and does
# not need the full render. MEASURED: the control's centroid is window-invariant — 99.996 / 99.997 /
# 100.002 / 100.026 Hz over 0.4 / 0.2 / 0.1 / 0.05 s (a mode-1 pluck with no rail is a pure
# sinusoid), giving an elevation of 4.682 / 4.682 / 4.681 / 4.680. A first cut read this as the
# control being too short and doubled it, because the *comparison* truncated the FRET pickup to the
# control's length (`min(len_a, len_b)`) — and the fret centroid is NOT window-invariant (468.2 over
# 0.4 s, 442.2 over 0.2 s). The bug was coupling the two windows, not the control's length; the
# right fix measures each signal over its own full run. *Generalizable: when a ratio moves after a
# window change, find out WHICH side moved before paying to lengthen the other.*
FRET_CONTROL_MAX = 0.2        # s — ample margin over the ~0.05 s where the centroid has converged
FRET_WORK_MAX = 60_000        # total steps across the fret run AND the control
FRET_LAM_DEFAULT = 0.4        # the validated collision-signature lambda
FRET_K_DEFAULT = 2.0e6        # N/m^alpha — a stiff metal fret
FRET_ALPHA = 1.5              # Hertzian-ish; fixed, NOT exposed (see the jawari's `alpha` note)
FRET_CLEARANCE_DEFAULT = 2.0e-3       # m — the brightness peak AND the intermittency default
FRET_CLEARANCE_MAX = 8.0e-3
FRET_RAIL_FRAC_DEFAULT = 1.0          # rail under the whole string -> support m = N-1
FRET_RAIL_FRAC_MIN = 0.2              # below ~0.15 the rail is out of reach (measured; see above)
FRET_AMP_DEFAULT = 5.0e-3             # m — the probe's mode-1 amplitude
FRET_AMP_MAX = 2.0e-2
FRET_SIGMA0_DEFAULT = 0.5             # loss ON (the jawari's rule: "sustained" needs a decay)
FRET_PICKUP_DEFAULT = 0.05            # the knee: 78 % of the best elevation at 2.3x the level
FRET_FRAMES_PER_PERIOD = 8            # transit-paced; f1 pacing recovers only 38.8 % of episodes
FRET_RASTER_COLS_PER_PERIOD = 10      # below this the debounce rounds under a column -> FRAGMENTS
FRET_RASTER_MAX_COLS = 1200
FRET_RASTER_MAX_ROWS = 128            # x-binning is free for the picture, fatal for the number
FRET_DEBOUNCE_FRAC = 0.10             # merge episodes closer than 10 % of a period (reed's rule)
FRET_EPISODES_MIN = 0.5               # per period; measured >= 1.00 everywhere reachable
# The "stopped releasing" bar. It is a GUARANTEE, not a live label — MEASURED unreachable across the
# whole slider range, and for a structural reason worth stating: a lossless one-sided spring always
# pushes back, so this string can never come to REST on the rail. Softening the rail raises the duty
# monotonically (0.188/0.300/0.434/0.456/0.466/0.466 at rail_stiffness 2e6...2e1) but it asymptotes
# just under **0.5**, which is the *free-sinusoid* limit — a string crossing the rail line untouched
# — i.e. the no-rail limit, NOT pinning. Stiffening it drives the duty the other way (the rail
# repels rather than admits). A first cut put this bar at 0.5 and would have shipped a "pinned"
# label sitting exactly on the soft-rail limit, firing on the one regime where the rail does LEAST.
# So the bar is where not-releasing actually lives, and the fact that nothing reaches it is itself
# the result: the intermittency is structural, not a tuning accident. (The recorded rule: a guard
# that cannot trip in the reachable range is a guarantee — do not widen a range to make it fire.)
FRET_DUTY_MAX = 0.9
FRET_CENTROID_FMAX = 8000.0           # the probe's brightness band
FRET_BRIGHTNESS_PEAK = 2.0e-3         # m — where the measured elevation peaks (3.33x); NAMED, not
#                                       claimed as a law: the curve is non-monotone either side.


def _fret_profile(x: NDArray[np.float64], L: float, *, clearance: float,
                  rail_frac: float) -> NDArray[np.float64]:
    """Flat rail at ``-clearance`` under ``0 < x <= rail_frac*L``; ``-inf`` (out of support) beyond.

    Mirrors the probe's and ``tests/test_collision_signature.py``'s geometry exactly, so the viewer
    runs the validated configuration rather than a lookalike (the jawari's own rule).
    """
    b = np.full_like(np.asarray(x, dtype=float), -np.inf)
    on = (x > 0.0) & (x <= rail_frac * L)
    b[on] = -clearance
    return b


def _build_fret(p: dict[str, Any], *, out_of_reach: bool = False) -> tuple[BarrierString,
                                                                          dict[str, Any]]:
    """Construct the flat-rail :class:`BarrierString`. ``out_of_reach`` drops the rail far below
    the string's swing: the brightness/pitch CONTROL — same string, same everything, no contact.
    """
    L = _fnum(p, "L", 1.0)
    T = _fnum(p, "T", 200.0)
    rho = _fnum(p, "rho", 0.005)
    lam = _fnum(p, "lambda", FRET_LAM_DEFAULT)
    # Deliberately NOT `K`/`alpha`/`bridge_stiffness` — see the leak note in the section comment.
    K = _fnum(p, "rail_stiffness", FRET_K_DEFAULT)
    clearance = _fnum(p, "clearance", FRET_CLEARANCE_DEFAULT)
    rail_frac = _fnum(p, "rail_frac", FRET_RAIL_FRAC_DEFAULT)
    sigma0 = _fnum(p, "sigma0", FRET_SIGMA0_DEFAULT)
    try:
        N = int(p.get("N", 100))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= FRET_N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {FRET_N_MAX}] for the fret, got {N}.")
    if min(L, T, rho) <= 0:
        raise ParamError("L, T, rho must all be positive.")
    if not (0.0 < lam < 1.0):
        raise ParamError(
            f"lambda must be in (0, 1), got {lam}: the coupled contact solve needs headroom below "
            "the string's marginal Nyquist mode."
        )
    if not (0.0 < clearance <= FRET_CLEARANCE_MAX):
        raise ParamError(f"clearance must be in (0, {FRET_CLEARANCE_MAX}] m, got {clearance}.")
    # Enforced HERE, not merely as a slider min: the frontend sends whatever it likes, and below
    # ~0.15 the rail is out of the string's reach -> a blank raster and an empty claim (measured).
    if not (FRET_RAIL_FRAC_MIN <= rail_frac <= 1.0):
        raise ParamError(
            f"rail_frac must be in [{FRET_RAIL_FRAC_MIN}, 1.0], got {rail_frac}: a shorter rail "
            "sits under a smaller share of the string's swing, and below ~0.15 nothing touches it."
        )
    if K <= 0:
        raise ParamError(f"rail_stiffness must be positive, got {K}.")
    if sigma0 < 0:
        raise ParamError(f"sigma0 must be >= 0, got {sigma0}.")

    c = math.sqrt(T / rho)
    fs = c * N / (L * lam)
    # sigma1 = hysteresis = 0, as for the jawari: hysteresis is contact DAMPING (it would fight the
    # re-injection the model is about) and sigma1 makes the highs decay off any flat-loss reading.
    string = DampedStiffString(L=L, T=T, rho=rho, fs=fs, N=N, kappa=0.0, sigma0=sigma0, sigma1=0.0)
    b = _fret_profile(string.x, L, clearance=(1.0 if out_of_reach else clearance),
                      rail_frac=rail_frac)
    bar = BarrierString(string=string, barrier=b, stiffness=K, alpha=FRET_ALPHA, hysteresis=0.0)
    info = {
        "c": c, "L": L, "N": N, "fs": fs, "lam": float(string.lam), "sigma0": sigma0,
        "clearance": clearance, "rail_frac": rail_frac, "K": K, "alpha": FRET_ALPHA,
        "rho": rho, "h": float(string.h), "support": int(bar._support.size),
    }
    return bar, info


class _FretRun:
    """Per-step telemetry of a fret run, captured in ONE pass.

    The contact history is reduced into the raster **as it is produced** rather than stored: a full
    ``(n_steps, m)`` mask would be ~9 MB at the cap for no gain, since every scalar the panel quotes
    is an accumulation anyway. What IS kept at full rate is the per-step ``n_active`` / ``iters`` /
    ``contact_any``, because the duty, the episode count and ``|C|max`` must be computed from the
    signal and never read off the picture (an OR-reduced raster over-states the duty at every finite
    resolution — measured 28.8/21.4/18.4 % at 400/800/1600 columns against a true 15.5 %).
    """

    def __init__(self, n: int, m: int, n_cols: int) -> None:
        self.E = np.empty(n + 1)
        self.pickup = np.empty(n + 1)
        self.n_active = np.zeros(n + 1, dtype=np.int32)
        self.iters = np.zeros(n + 1, dtype=np.int32)
        # Per-cell MAX contact force -> the optional linear grey. Contact force spans median 10.8 N
        # to max 77.2 N, a dynamic range of only 7.1x, so no log mapping is needed; and the faintest
        # half of the contact pixels carry 14.2 % of the impulse, so a binary mask would not badly
        # over-weight grazing touches either. The grey is a refinement, not a correctness fix.
        self.raster = np.zeros((m, n_cols))
        self.frames: list[NDArray[np.float64]] = []
        self.frame_steps: list[int] = []
        # <2 KE / E> with the CENTERED velocity the loss identity actually uses (see the section
        # note): sum and count, accumulated online over three time levels rather than by storing the
        # field history.
        self.equi_sum = 0.0
        self.equi_count = 0


def _run_fret(bar: BarrierString, n_steps: int, *, pickup_idx: int, anim_stride: int,
              frame_until: int, n_cols: int, capture_contact: bool = True) -> _FretRun:
    """Step the barrier string, capturing audio + energy + contact raster + traces + frames in ONE
    pass (the geometric/mallet/sympathetic/jawari pattern — ``simulate`` exposes none of the contact
    internals, and they ARE the panels).

    The animation is a *stride of this same run*, not a second resonator: unlike the bow, whose
    window is a settled tail, the fret buzzes from ``t = 0``, so the frames the animation wants are
    already flowing past. A second run would silently double the cost of a root-find-per-step model.
    """
    m = bar.penetration.size
    run = _FretRun(n_steps, m if capture_contact else 0, n_cols if capture_contact else 0)
    s = bar.string
    rho, h, k2 = s.rho, s.h, 2.0 * bar.k

    def _sample(i: int) -> None:
        run.E[i] = bar.energy()
        run.pickup[i] = s.displacement_at(pickup_idx)
        if capture_contact:
            mask = bar.contact_mask()
            run.n_active[i] = int(np.count_nonzero(mask))
            run.iters[i] = int(bar.newton_iters)
            if run.n_active[i]:
                col = min(n_cols - 1, (i * n_cols) // (n_steps + 1))
                f = np.abs(bar.contact_force)
                np.maximum(run.raster[:, col], np.where(mask, f, 0.0), out=run.raster[:, col])

    _sample(0)
    if frame_until >= 1:
        run.frames.append(s.u.copy())
        run.frame_steps.append(0)
    for i in range(1, n_steps + 1):
        # u^{i-2}, saved before the roll: the centered velocity at level i-1 is
        # (u^i - u^{i-2}) / 2k, and the energy it pairs with is E[i-1], already sampled.
        u_prev2 = s.u_prev.copy()
        bar.step()
        _sample(i)
        if capture_contact and run.E[i - 1] > 0.0:
            v_c = (s.u[1:-1] - u_prev2[1:-1]) / k2
            ke_c = 0.5 * rho * h * float(np.dot(v_c, v_c))
            run.equi_sum += 2.0 * ke_c / run.E[i - 1]
            run.equi_count += 1
        if i <= frame_until and i % anim_stride == 0:
            run.frames.append(s.u.copy())
            run.frame_steps.append(i)
    if not np.all(np.isfinite(run.E)):
        raise ParamError("simulation produced non-finite energy (instability) — adjust parameters.")
    return run


def _fret_episodes(contact: NDArray[np.bool_], min_gap: int) -> int:
    """Contact episodes, merging any two separated by fewer than ``min_gap`` samples.

    The reed's debounce rule, second customer. Raw onsets and debounced episodes agree at the
    default (31 vs 31), but the chatter regime is one slider away and the rule is nearly free.
    """
    on = np.flatnonzero(contact[1:] & ~contact[:-1])
    if on.size == 0:
        return 0
    off = np.flatnonzero(~contact[1:] & contact[:-1])
    if off.size and off[0] < on[0]:
        off = off[1:]
    n = 1
    for i in range(1, on.size):
        prev_off = off[i - 1] if i - 1 < off.size else on[i]
        if on[i] - prev_off >= min_gap:
            n += 1
    return n


def _fret_contact_block(run: _FretRun, info: dict[str, Any], fs: float, f1: float, n_steps: int,
                        n_cols: int, support_x: NDArray[np.float64]) -> dict[str, Any]:
    """The GATED claim: the string slaps the rail and springs off, and the solve is a vector one.

    Every scalar here is computed at FULL RATE on the FULL support. Only the *image* may be binned:
    dropping the support from 99 to 33 x-bins leaves the ink fill identical (6.11 %) — the raster
    reads the same — but collapses ``|C|max`` from 69 to 23, and ``|C|max`` is precisely the number
    that makes this a vector Newton rather than the mallet's scalar.
    """
    contact = run.n_active > 0
    n_periods = n_steps / (fs / f1)
    episodes = _fret_episodes(contact, max(1, int(FRET_DEBOUNCE_FRAC * fs / f1)))
    raw_onsets = int(np.count_nonzero(contact[1:] & ~contact[:-1]))
    duty = float(contact.mean())
    per_period = float(episodes / n_periods) if n_periods > 0 else 0.0
    touching = run.n_active[contact]

    # The image (and ONLY the image) is binned when the support outruns the raster's rows.
    raster, rows = run.raster, run.raster.shape[0]
    x_binned = rows > FRET_RASTER_MAX_ROWS
    if x_binned:
        edges = np.linspace(0, rows, FRET_RASTER_MAX_ROWS + 1).astype(int)
        raster = np.array([raster[a:b].max(axis=0) for a, b in zip(edges[:-1], edges[1:],
                                                                   strict=True) if b > a])
    fmax = float(raster.max()) if raster.size else 0.0
    if fmax > 0.0:
        # Any contact stays >= 1 so the grey never rounds a genuine touch away: the uint8 map is a
        # strict refinement of the binary mask, which is what the panel's honesty rests on.
        img = np.where(raster > 0.0, np.maximum(1.0, np.rint(255.0 * raster / fmax)), 0.0)
    else:
        img = np.zeros_like(raster)

    # The traces share the raster's columns so they line up under it. Reduced by MAX, not by
    # sampling: |C| is an intermittent spike train, and picked samples would under-report it for
    # exactly the reason the duty must not be read off the picture.
    def _col_max(a: NDArray[np.float64]) -> list[float | None]:
        edges = np.linspace(0, a.size, n_cols + 1).astype(int)
        return _finite_list(np.array([float(a[p:q].max()) for p, q in
                                      zip(edges[:-1], edges[1:], strict=True) if q > p]), 3)

    return {
        "kind": "fret",
        # -- the gated claim --------------------------------------------------------------
        "duty": round(duty, 4),
        "episodes": int(episodes),
        "raw_onsets": raw_onsets,
        "episodes_per_period": round(per_period, 3),
        "episodes_min": FRET_EPISODES_MIN,
        "duty_max": FRET_DUTY_MAX,
        "intermittent": bool(duty > 0.0 and per_period >= FRET_EPISODES_MIN
                             and duty <= FRET_DUTY_MAX),
        # Both are correct renders, not failures — a rail nobody reaches and a string lying on one.
        # `out_of_reach` is live (clearance 6 mm, or a small amplitude, or a short rail);
        # `pinned` is a measured GUARANTEE that never fires — see the FRET_DUTY_MAX note.
        "out_of_reach": bool(duty == 0.0),
        "pinned": bool(duty >= FRET_DUTY_MAX),
        # -- the vector Newton ------------------------------------------------------------
        "support": int(info["support"]),
        "active_max": int(run.n_active.max()),
        "active_mean_touching": round(float(touching.mean()), 2) if touching.size else None,
        "iters_max": int(run.iters.max()),
        "iters_mean": round(float(run.iters[1:].mean()), 3) if n_steps else None,
        # -- the raster -------------------------------------------------------------------
        "raster": {
            "b64": base64.b64encode(img.astype(np.uint8).tobytes()).decode("ascii"),
            "n_rows": int(img.shape[0]) if img.size else 0,
            "n_cols": int(img.shape[1]) if img.size else 0,
            "x_binned": bool(x_binned),
            "x0": round(float(support_x[0]), 4),
            "x1": round(float(support_x[-1]), 4),
            "t0": 0.0,
            "t1": round(n_steps / fs, 6),
            "force_max": round(fmax, 3),
            "cols_per_period": round(n_cols / n_periods, 2) if n_periods > 0 else None,
        },
        "trace": {"active": _col_max(run.n_active.astype(float)),
                  "iters": _col_max(run.iters.astype(float))},
        "n_periods": round(n_periods, 2),
    }


def _fret_decay_triple(run: _FretRun, fs: float, sigma0: float) -> dict[str, Any]:
    """The DIAGNOSTIC triple that replaces the jawari's 2*sigma0 oracle (see the section note).

    ``rate`` is what the run does; ``2 sigma0`` is what equipartition would predict; and
    ``2 sigma0 <2KE/E>`` is the same prediction with the equipartition assumption removed. The third
    matching the first to ~0.1 % is the whole point — and it is *reported*, never gated, because
    both sides come from one run of one identity and the agreement is not uniform in alpha.
    """
    t = np.arange(run.E.size) / fs
    rate = _fit_decay(t, run.E)
    equi = (run.equi_sum / run.equi_count) if run.equi_count else None
    corrected = (2.0 * sigma0 * equi) if equi is not None else None
    return {
        "rate": rate,
        "oracle_2sigma": 2.0 * sigma0,
        "equipartition": round(equi, 5) if equi is not None else None,
        "corrected": corrected,
        "ratio": round(rate / (2.0 * sigma0), 4) if (rate and sigma0 > 0.0) else None,
        "agreement": (round(abs(rate - corrected) / corrected, 5)
                      if (rate and corrected and corrected > 0.0) else None),
    }


def _fret_signature_block(fret_pickup: NDArray[np.float64], ctrl_pickup: NDArray[np.float64],
                          fs: float, f1: float, info: dict[str, Any]) -> dict[str, Any]:
    """Brightness elevation and the crossing rate — BOTH diagnostic, neither gated.

    Each signal is measured over its OWN full run, never truncated to the other's length. The
    control is shorter and that is fine — and load-bearing to get right: its centroid is
    window-invariant (99.996-100.026 Hz over an 8x range of windows) because a mode-1 pluck with no
    rail is a pure sinusoid, so the whole elevation is harmonic content the rail added. The fret's
    centroid is NOT window-invariant (468.2 over 0.4 s vs 442.2 over 0.2 s), so coupling the two
    windows would silently report the fret's *short-window* brightness under the full run's label.
    """
    a, b = fret_pickup, ctrl_pickup
    f_bright = _fret_centroid(a, fs)
    c_bright = _fret_centroid(b, fs)
    elevation = float(f_bright / c_bright) if c_bright > 0 else float("nan")
    f_cross = _interp_zero_cross_frequency(a - float(np.mean(a)), fs)
    c_cross = _interp_zero_cross_frequency(b - float(np.mean(b)), fs)
    cents = (1200.0 * math.log2(f_cross / c_cross)
             if (f_cross > 0 and c_cross > 0) else None)
    return {
        # `drawDiagnostics` dispatches on meta.spectrum.kind; without it the fret falls through
        # every branch to drawPartials, which reads per-partial data this model never ships.
        "kind": "fret",
        "centroid_fret": round(f_bright, 1),
        "centroid_control": round(c_bright, 1),
        "elevation": round(elevation, 3),
        "f1": round(f1, 3),
        "clearance": info["clearance"],
        # NAMED, not claimed: the elevation peaks at an intermediate clearance and falls either side
        # (2.50/2.71/3.33/2.59/2.83x at 4/3/2/1/0.5 mm), so no monotone label survives the slider.
        "peak_clearance": FRET_BRIGHTNESS_PEAK,
        "monotone": False,
        # Deliberately NOT called pitch: a buzz adds zero crossings *within* a period, so this mixes
        # the fundamental with the rail's contribution. Large and real; an oracle for nothing.
        "crossing_rate": round(f_cross, 2),
        "crossing_rate_control": round(c_cross, 2),
        "crossing_cents": round(cents, 1) if cents is not None else None,
        "crossing_is_pitch": False,
        # model #8's exact money test (tests/test_barrier_collision.py), cited as the magnitude
        # credential of the contact force — alpha = 1, static, and not a thing this render re-runs.
        "static_oracle": {"claim": "S u* = (K/rho) b", "residual": 3.4e-15, "alpha": 1.0},
    }


def _fret_centroid(sig: NDArray[np.float64], fs: float) -> float:
    """Band-limited amplitude-weighted mean frequency — the probe's brightness proxy verbatim."""
    sig = np.asarray(sig, dtype=float)
    if sig.size < 4:
        return float("nan")
    mag = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    sel = freqs <= FRET_CENTROID_FMAX
    total = float(np.sum(mag[sel]))
    return float(np.sum(freqs[sel] * mag[sel]) / total) if total > 0 else float("nan")


def _build_payload_fret(p: dict[str, Any]) -> dict[str, Any]:
    playback_speed = _fnum(p, "playback_speed", 0.02)
    pickup_frac = _fnum(p, "pickup_position", FRET_PICKUP_DEFAULT)
    audio_dur = _fnum(p, "audio_duration", 0.4)
    anim_win = _fnum(p, "animation_window", 0.1)
    amplitude = _fnum(p, "amplitude", FRET_AMP_DEFAULT)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRET_FRAMES_PER_PERIOD)))

    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")
    if not (0.0 < audio_dur <= FRET_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {FRET_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < amplitude <= FRET_AMP_MAX):
        raise ParamError(f"amplitude must be in (0, {FRET_AMP_MAX}] m, got {amplitude}.")

    bar, info = _build_fret(p)
    fs, L, c, N = info["fs"], info["L"], info["c"], info["N"]
    f1 = c / (2.0 * L)
    n_steps = max(1, round(audio_dur * fs))
    # Short (see the FRET_CONTROL_MAX note), but still BUDGETED: the control is 0.95x the fret run's
    # cost per step — there is no contact solve to skip, because the string step and the rank-m
    # correction dominate either way — so it is never assumed free.
    n_control = max(1, round(min(audio_dur, FRET_CONTROL_MAX) * fs))
    if n_steps + n_control > FRET_WORK_MAX:
        raise ParamError(
            f"this configuration needs {n_steps + n_control} steps across the fret run and its "
            f"out-of-reach control (budget {FRET_WORK_MAX}): every step is a vector contact solve "
            f"over up to {info['support']} rail nodes. Shorten audio_duration, lower N, or raise "
            f"lambda."
        )

    pickup_idx = min(max(1, round(pickup_frac * N)), N - 1)
    anim_stride = max(1, round((fs / f1) / fpp))
    n_anim = min(n_steps, max(anim_stride, round(anim_win * fs)))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    # >= 10 columns per period: below that the debounce window rounds to under one column and the
    # episode count degenerates to raw onsets and FRAGMENTS the slaps (83 and 60 apparent episodes
    # at 200 and 100 columns against a truth of 49). A too-coarse raster looks BUSIER, not emptier —
    # which is the more dangerous failure. The count is computed at full rate regardless; this is
    # what keeps the picture honest next to it.
    n_periods = max(1e-9, n_steps / (fs / f1))
    n_cols = int(min(FRET_RASTER_MAX_COLS, max(16, round(FRET_RASTER_COLS_PER_PERIOD * n_periods))))

    shape = _mode1_shape(bar.string) * amplitude
    bar.set_state(shape)
    run = _run_fret(bar, n_steps, pickup_idx=pickup_idx, anim_stride=anim_stride,
                    frame_until=n_anim, n_cols=n_cols)

    # The control: the SAME string with the rail dropped out of reach. Sharing every string
    # parameter is what makes the elevation attributable to the rail and nothing else.
    ctrl, _ = _build_fret(p, out_of_reach=True)
    ctrl.set_state(_mode1_shape(ctrl.string) * amplitude)
    ctrl_run = _run_fret(ctrl, n_control, pickup_idx=pickup_idx, anim_stride=1, frame_until=0,
                         n_cols=0, capture_contact=False)

    frames = np.array(run.frames, dtype=float)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    audio48, peak = _resample_normalize(run.pickup, fs)
    sim = SimResult(time=np.arange(run.E.size) / fs, energy=run.E, output=None, fs=fs, snapshots=[])
    # THE INDEXING TRAP (batch 8): `_b` and `contact_mask()` are over the SUPPORT, not the grid, so
    # both are scattered back onto grid coordinates before shipping. Contact STATISTICS stay
    # support-relative — that is the frame the model's own tests report them in.
    rail = np.full(bar.string.N + 1, np.nan)
    rail[bar._support] = bar._b
    support_x = bar.string.x[bar._support]

    # decay_oracle=False, and the one place the two configurations of model #8 disagree: the flat
    # rail breaks equipartition (6-9 % fast) where the jawari's gentle wrap did not. The triple
    # rides alongside so the panel SHOWS why, instead of printing a mismatch it cannot explain.
    energy = _energy_block(sim, sigma_zero=bool(info["sigma0"] == 0.0),
                           oracle_2sigma=2.0 * info["sigma0"], decay_oracle=False)
    energy["decay_triple"] = _fret_decay_triple(run, fs, info["sigma0"])

    return {
        "model": "fret",
        "fs_sim": round(fs, 3),
        "lambda": round(info["lam"], 6),
        "grid": {"x": _finite_list(bar.string.x, 6), "barrier": _finite_list(rail)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(np.array(run.frame_steps, dtype=float) / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": energy,
        "meta": {
            "c": round(c, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_steps),
            "n_control_steps": int(n_control),
            "n_frames": int(frames.shape[0]),
            "probe_x": round(float(bar.string.x[pickup_idx]), 4),
            "clearance": info["clearance"],
            "rail_frac": info["rail_frac"],
            "rail_span": round(info["rail_frac"] * L, 4),
            "amplitude": amplitude,
            "contact": _fret_contact_block(run, info, fs, f1, n_steps, n_cols, support_x),
            "spectrum": _fret_signature_block(run.pickup, ctrl_run.pickup, fs, f1, info),
        },
    }


# == bore (wind leg) — the acoustic tube + radiating bell ==========================================
#
# The first WIND model in the viewer and a new field type: **pressure** along a tube, not
# displacement along a string. Linear, so nothing here root-finds per step — the reed (which
# *contains* a ``Bore`` and draws on the same picture) inherits a validated viz.
#
# THE STRUCTURAL FACT THE BATCH TURNS ON: the bell's loss is **booked**. ``Bore.energy()`` is
# ``acoustic_energy() + radiated_energy``, so a sigma = 0 tube with a *radiating* end still
# CONSERVES (measured drift 3.3e-16 – 7.4e-15) while the bell sheds 9.8 % of E0 at a physical
# ``R/Z0 ~ 3e-4`` and 100 % at the matched (anechoic) ``R = Z0``. It is the first lossy model in
# the viewer that does NOT give up the conservation verdict — contrast the bow, whose sigma-loss is
# *inferred* and which therefore cannot score its lossy branch at all.
#
# ...which is exactly why the energy panel must plot the SPLIT (``_energy_block(split=...)``):
# acoustic falling, radiated rising, sum flat. A flat green "conserved, drift 1e-15 ✓" beside an
# audibly decaying pickup reads as a bug and hides the very physics the batch is about.
#
# THE COROLLARY THAT CONSTRAINS THE SLIDERS: the *viscous* sigma is NOT booked — there is no
# viscous accumulator in ``bore.py``, so ``energy()`` conserves only at sigma = 0. A sigma slider
# would silently re-introduce an inferred channel and demote the verdict back to a bare monotone
# check, buying a loss the bell already provides physically. So **sigma is fixed at 0 and never
# exposed; R/Z0 is the loss control.** Generalizable, and batch 10's reed inherits it: a
# measured-channel verdict is only as strong as the *least*-booked channel you let the user switch
# on.

BORE_N_MIN, BORE_N_MAX = 32, 256
BORE_AUDIO_MAX = 1.5
BORE_ANIM_MAX = 0.1           # NOT the shared ANIM_WIN_MAX = 2.0 — see the cost note in the builder
BORE_WORK_MAX = 300_000       # total steps across the render AND the reflection run
BORE_LAMBDA = 1.0             # pinned: lambda is an OPERATOR claim here, not a render knob
BORE_RADIUS = 0.008           # m — scales absolute energy and Z0 only; R/Z0 is the real control
BORE_AMP = 1e-3               # Pa — the model is LINEAR, so amplitude is pure scale, not a slider
BORE_ODD_EVEN_GATE = 1e3      # measured 3.6e4 at the SHORTEST allowed duration (see the block)
BORE_R_RATIO_MIN, BORE_R_RATIO_MAX = 1e-4, 30.0
# Above this R/Z0 the bell absorbs so hard that no standing wave forms — the partial/odd-harmonic
# claims stop applying (they are not *wrong*, there is simply nothing to measure). Labelled, never
# failed: the anechoic null is a correct render, not a broken one.
BORE_RESONANT_RATIO = 0.05
BORE_LAMBDA_CURVE = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.97, 1.0)


def _bore_bump(x: NDArray[np.float64], L: float, *, center_frac: float,
               width_frac: float = 0.06, amplitude: float = BORE_AMP) -> NDArray[np.float64]:
    """The suite's own excitation — a Gaussian pressure bump (``tests/test_bore_modal._bump``)."""
    c, w = center_frac * L, width_frac * L
    return amplitude * np.exp(-((x - c) ** 2) / (2.0 * w * w))


def _bore_eigenfrequencies(bore: Bore, n_modes: int) -> NDArray[np.float64]:
    """The ``n_modes`` lowest discrete resonances (Hz) of ``bore``, from the OPERATOR.

    Mirrors ``tests/helpers.bore_low_eigenfrequencies`` (which the wrapper cannot import): solve
    ``L phi = omega^2 C phi`` on the free (non-open) pressure nodes and map each eigenvalue through
    the leapfrog dispersion. This is the oracle that structurally certifies the half-cell wall — it
    owes nothing to any FFT window, which is the whole reason it is worth carrying alongside the
    measured spectrum.
    """
    dof = bore.dof
    Lfree = bore.Lop[dof][:, dof]
    Cfree = bore.Cmat[dof][:, dof]
    n_open = int(bore._open_left) + int(bore._open_right)
    if n_open == 0:
        shift = -1e-3 * (math.pi * bore.c0 / bore.L) ** 2  # a closed tube has an omega = 0 mode
        w2 = eigsh(Lfree, k=n_modes + 1, M=Cfree, sigma=shift, which="LM",
                   return_eigenvectors=False)
        w2 = np.sort(w2)[1:n_modes + 1]
    else:
        w2 = eigsh(Lfree, k=n_modes, M=Cfree, sigma=0.0, which="LM", return_eigenvectors=False)
        w2 = np.sort(w2)
    return np.asarray(modal.discrete_bore_eigenfrequency(w2, bore.k))


def _build_bore(p: dict[str, Any]) -> tuple[Bore, dict[str, Any]]:
    """Construct the bore from params. ``lambda`` is pinned; ``sigma`` is not a param (see above).

    The control is the **dimensionless** ``R/Z0``, not a raw ``R``: ``Z0 = rho0 c0 / S`` moves with
    the bore radius, so a raw-R slider would mean something different at every geometry (the
    tension string's ``dT/T0`` / geometric ``frac`` / jawari ``downswing/depth`` pattern, fourth
    customer). It arrives as a base-10 EXPONENT because the interesting range spans a physical
    clarinet's 3e-4 to the anechoic 1 and past it — five decades no linear slider can carry.
    """
    L = _fnum(p, "L", 0.5)
    N = int(_fnum(p, "N", 128))
    ratio_exp = _fnum(p, "bell_ratio_exp", -3.5)
    domain = str(p.get("domain", "radiating"))

    if domain not in ("radiating", "open"):
        raise ParamError(f"bore end must be 'radiating' or 'open', got {domain!r}.")
    if not (0.1 <= L <= 2.0):
        raise ParamError(f"L must be in [0.1, 2.0] m, got {L}.")
    if not (BORE_N_MIN <= N <= BORE_N_MAX):
        raise ParamError(f"N must be in [{BORE_N_MIN}, {BORE_N_MAX}] for the bore, got {N}.")
    ratio = 10.0 ** ratio_exp
    if not (BORE_R_RATIO_MIN <= ratio <= BORE_R_RATIO_MAX):
        raise ParamError(
            f"the bell's R/Z0 must be in [{BORE_R_RATIO_MIN}, {BORE_R_RATIO_MAX}], got {ratio:.3e} "
            f"(bell_ratio_exp = {ratio_exp})."
        )

    radiating = domain == "radiating"
    Z0 = RHO0_AIR * C0_AIR / (math.pi * BORE_RADIUS * BORE_RADIUS)
    fs = C0_AIR / (BORE_LAMBDA * (L / N))
    bore = Bore(
        L=L, fs=fs, N=N, radius=BORE_RADIUS,
        boundary=("closed", "radiating" if radiating else "open"),
        sigma=0.0, R_bell=(ratio * Z0) if radiating else 0.0,
    )
    info = {
        "L": L, "N": N, "fs": fs, "lam": bore.lam, "Z0": Z0, "c0": C0_AIR,
        "ratio": ratio, "radiating": radiating, "R_bell": bore.R_bell,
        "ends": ["closed", "radiating" if radiating else "open"],
        "resonant": (not radiating) or ratio <= BORE_RESONANT_RATIO,
    }
    return bore, info


class _BoreRun:
    """Per-step telemetry of a bore run — the pickup and BOTH energy channels, kept separate."""

    def __init__(self, n: int, width: int) -> None:
        self.E_ac = np.empty(n + 1)     # acoustic energy in the tube
        self.E_rad = np.empty(n + 1)    # cumulative energy shed to the far field (booked)
        self.pickup = np.empty(n + 1)
        self.env = np.zeros(width)      # running max|p(x)| over the FULL run -> the static envelope
        self.frames: list[NDArray[np.float64]] = []
        self.frame_steps: list[int] = []
        self.frame_rad: list[float] = []  # per-frame cumulative radiated energy -> the mouth glow


def _run_bore(bore: Bore, n_steps: int, *, pickup_idx: int, anim_stride: int,
              frame_until: int) -> _BoreRun:
    """Step the bore once, capturing pickup + the two energy channels + frames + the envelope.

    ONE sim, not two (the jawari's pattern rather than the string path's): the animation window is
    a prefix of the audio run at the same initial condition, so a second sim would buy an identical
    picture at twice the wall clock. The envelope is a running max over the *whole* run — no single
    instantaneous frame can show the formed node/antinode structure.
    """
    run = _BoreRun(n_steps, bore.N + 1)

    def _sample(i: int) -> None:
        run.E_ac[i] = bore.acoustic_energy()
        run.E_rad[i] = bore.radiated_energy
        run.pickup[i] = bore.pressure_at(pickup_idx)
        np.maximum(run.env, np.abs(bore.p), out=run.env)

    def _frame(i: int) -> None:
        run.frames.append(bore.p.copy())
        run.frame_steps.append(i)
        run.frame_rad.append(bore.radiated_energy)

    _sample(0)
    if frame_until >= 0:
        _frame(0)
    for i in range(1, n_steps + 1):
        bore.step()
        _sample(i)
        if i <= frame_until and i % anim_stride == 0:
            _frame(i)
    if not np.all(np.isfinite(run.E_ac)):
        raise ParamError("simulation produced non-finite energy (instability) — adjust parameters.")
    return run


def _bore_reflection_block(info: dict[str, Any]) -> dict[str, Any]:
    """The money panel: one bounce off the bell against the closed-form ``r = (R-Z0)/(R+Z0)``.

    A **centred** Gaussian splits into two counter-propagating halves; the right-going half bounces
    once off the bell and sheds exactly ``1/2 (1 - r^2)`` of E0. Measured worst absolute error over
    ``R/Z0`` in [0.03, 30] is 1.4e-16, with an exact anechoic null at ``R = Z0`` (shed = 0.500000 —
    the entire right-going half absorbed, r = 0).

    The **curve** is free: ``r = (ratio - 1)/(ratio + 1)`` is geometry-invariant, so the analytic
    sweep costs no simulation at all. Only ONE point is measured — the user's own ``R/Z0`` — and it
    is a second initial condition, hence a second run, which the work budget counts.
    """
    # 1.0 is inserted EXACTLY: the anechoic null is the curve's whole point, and a logspace that
    # merely straddles it would draw a peak of 0.499973 — a plot that quietly misses its own claim.
    # The span must cover the SLIDER's range, not just the interesting part of the curve: the
    # default is a physical clarinet at R/Z0 ~ 3e-4, and a curve starting at 1e-2 would put the
    # measured point off the left edge of its own panel — invisible at the shipped defaults.
    ratios = np.sort(np.append(np.logspace(-4.0, 1.5, 79), 1.0))
    curve = 0.5 * (1.0 - ((ratios - 1.0) / (ratios + 1.0)) ** 2)
    block: dict[str, Any] = {
        "radiating": bool(info["radiating"]),
        "curve": {"ratio": _finite_list(ratios, 5), "shed": _finite_list(curve, 6)},
    }
    if not info["radiating"]:
        # An "open" end is pressure-release: r = -1, nothing is shed, and there is no bell to score.
        block["note"] = "the ideal open end reflects perfectly (r = -1) - no radiation to measure."
        return block

    L, N, ratio = info["L"], info["N"], info["ratio"]
    b = Bore(L=L, fs=info["fs"], N=N, radius=BORE_RADIUS, boundary=("closed", "radiating"),
             sigma=0.0, R_bell=info["R_bell"])
    b.set_state(_bore_bump(b.x, L, center_frac=0.5, width_frac=0.04))
    E0 = b.energy()
    for _ in range(N):  # lambda = 1 -> one node per step; centre -> bell -> back = N steps
        b.step()
    shed = float(b.radiated_energy / E0) if E0 > 0 else float("nan")
    r = (info["R_bell"] - info["Z0"]) / (info["R_bell"] + info["Z0"])
    oracle = 0.5 * (1.0 - r * r)
    block.update({
        "ratio": ratio,
        "r": round(float(r), 6),
        "oracle": round(float(oracle), 6),
        "measured": round(shed, 6),
        "abs_error": abs(shed - float(oracle)),
        "tol": 1e-9,
        "pass": bool(abs(shed - float(oracle)) < 1e-9),
        "anechoic": bool(abs(ratio - 1.0) < 1e-9),
        "steps": int(N),
    })
    return block


def _bore_dispersion_block(L: float) -> dict[str, Any]:
    """Cents-vs-lambda from the EIGENVALUE oracle — no time-stepping, so it costs milliseconds.

    A lambda slider *on the render* is a trap: ``fs = c0 N / (lambda L)``, so steps scale as
    ``1/lambda`` and at N_MAX / 1.5 s the budget is exhausted at lambda = 0.878 — before the
    reflection run. And the payoff is not there: 0.07–0.67 cents is inaudible and invisible on a
    spectrum, and making it *watchable* needs lambda ~ 0.5–0.7, exactly where the cost explodes and
    exactly where CLAUDE.md's "tune toward lambda = 1" says not to live.

    So the claim is computed where it lives — in the operator. Each point is one ``eigsh`` call on
    a fixed grid (only ``fs`` moves), and the departure from the continuum falls by 4.01x across a
    2x refinement, i.e. exactly O(h^2), collapsing to 0.0000 cents at lambda = 1.
    **Generalizable: when a claim is about the operator, compute it from the operator — don't buy
    it with wall clock by rendering audio nobody can hear the difference in.**
    """
    n_modes, n_coarse, n_fine = 5, 64, 128
    cont = modal.bore_resonance_frequencies(C0_AIR, L, n_modes, "closed-open")

    def _worst(lam: float, N: int) -> float:
        """Worst |cents| of the first ``n_modes`` eigenfrequencies against the continuum."""
        b = Bore(L=L, fs=C0_AIR / (lam * (L / N)), N=N, radius=BORE_RADIUS,
                 boundary=("closed", "open"), sigma=0.0)
        return float(np.max(np.abs(modal.cents(_bore_eigenfrequencies(b, n_modes), cont))))

    # Two grids at every lambda, because the CLAIM is the ratio: a single curve shows a departure,
    # the pair shows it is second-order in h (4.01x across a 2x refinement).
    coarse = np.array([_worst(lam, n_coarse) for lam in BORE_LAMBDA_CURVE])
    fine = np.array([_worst(lam, n_fine) for lam in BORE_LAMBDA_CURVE])
    order = [round(float(c / f), 3) if f > 0 else None
             for c, f in zip(coarse, fine, strict=True)]
    return {
        "lambda": [round(lam, 4) for lam in BORE_LAMBDA_CURVE],
        "coarse": _finite_list(coarse, 5),
        "fine": _finite_list(fine, 5),
        "order": order,
        "n_coarse": n_coarse, "n_fine": n_fine, "n_modes": n_modes,
    }


def _bore_signature_block(pickup: NDArray[np.float64], fs: float,
                          info: dict[str, Any]) -> dict[str, Any]:
    """The clarinet signature: odd harmonics only, and the partials against BOTH oracles.

    Two oracles, deliberately: the continuum ``f_n = (2n-1) c0/4L`` is the physics, and the
    discrete **eigenvalue** frequencies are what the scheme can actually produce. At lambda = 1 they
    agree to 0.0000 cents at every N, which is the point — but the measured spectrum must be read
    through the suite's parabolic-refined :func:`spectrum.measure_partials_near`, not a bin
    peak-pick: a crude peak-pick reported 1.69 cents at N = 100/200 and 0.00 at N = 64/128, which is
    not physics but which bin the fundamental landed on (batch 1's interpolation lesson, second
    customer). Refined, every measured partial is <= 0.007 cents.

    The odd/even ratio is set by the FFT **window**, not by N or by physics: 2.29e5 at 0.5 s but
    3.6e4 at 0.25 s, flat in N to three digits at each. So the gate sits at the SHORTEST allowed
    duration — gating on the 0.5 s number would fail a legitimately-correct short render (the bow's
    "a rate needs a long window" lesson in spectral clothing).
    """
    L = info["L"]
    n_partials = 5
    cont = modal.bore_resonance_frequencies(C0_AIR, L, n_partials, "closed-open")
    cont = cont[cont < 0.45 * fs]
    ideal = Bore(L=L, fs=fs, N=info["N"], radius=BORE_RADIUS, boundary=("closed", "open"),
                 sigma=0.0)
    eig = _bore_eigenfrequencies(ideal, n_partials)[: cont.size]
    measured = spectrum.measure_partials_near(pickup, fs, cont)

    f1 = float(cont[0])
    freqs, mag, _ = spectrum.magnitude_spectrum(pickup, fs)
    df = float(freqs[1] - freqs[0])

    def _peak_near(f: float) -> float:
        i = int(round(f / df))
        return float(mag[max(1, i - 2): i + 3].max())

    odd = [_peak_near((2 * n - 1) * f1) for n in range(1, 6)]
    even = [_peak_near(2 * n * f1) for n in range(1, 6)]
    ratio = float(min(odd) / max(even)) if max(even) > 0 else float("inf")

    f_max = min(0.45 * fs, 14.0 * f1)
    sf, sm, snorm = _jawari_band_spectrum(pickup, fs, f_max)
    sm = [None if v is None else v / (snorm or 1.0) for v in sm]

    return {
        "kind": "bore",
        "f1": round(f1, 3),
        # The claims below need a standing wave. At a heavily-absorbing bell there is none —
        # LABEL, never fail (the jawari's grazing-ratio precedent).
        "applies": bool(info["resonant"]),
        "odd_even": {
            "ratio": ratio, "gate": BORE_ODD_EVEN_GATE,
            "pass": bool(ratio > BORE_ODD_EVEN_GATE),
        },
        "partials": {
            "continuum": _finite_list(cont, 4),
            "eigen": _finite_list(eig, 4),
            "measured": _finite_list(measured, 4),
            "cents_vs_continuum": _finite_list(modal.cents(measured, cont), 4),
            "cents_vs_eigen": _finite_list(modal.cents(measured, eig), 4),
            "eigen_vs_continuum": _finite_list(modal.cents(eig, cont), 5),
        },
        "spectrum": {"f_max": round(f_max, 1), "f": sf, "mag": sm},
    }


def _build_payload_bore(p: dict[str, Any]) -> dict[str, Any]:
    playback_speed = _fnum(p, "playback_speed", 0.02)
    pickup_frac = _fnum(p, "pickup_position", 0.1)
    audio_dur = _fnum(p, "audio_duration", 0.5)
    anim_win = _fnum(p, "animation_window", 0.03)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < pickup_frac < 1.0):
        raise ParamError(f"pickup_position must be in (0, 1), got {pickup_frac}.")
    if not (0.0 < audio_dur <= BORE_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {BORE_AUDIO_MAX}] s, got {audio_dur}.")
    # NOT the shared ANIM_WIN_MAX = 2.0. THE COST HOLE: at N = 256 a 2 s window is 351,232 animation
    # steps — over BORE_WORK_MAX on its own — and the MAX_FRAMES re-stride does NOT save it (it caps
    # the frames emitted, not the steps simulated). Generalizable: a frame-count ceiling is not a
    # cost ceiling.
    if not (0.0 < anim_win <= BORE_ANIM_MAX):
        raise ParamError(f"animation_window must be in (0, {BORE_ANIM_MAX}] s, got {anim_win}.")

    bore, info = _build_bore(p)
    fs, L, N = info["fs"], info["L"], info["N"]
    f1 = C0_AIR / (4.0 * L)
    n_steps = max(1, round(audio_dur * fs))
    n_refl = N if info["radiating"] else 0
    if n_steps + n_refl > BORE_WORK_MAX:
        raise ParamError(
            f"this configuration needs {n_steps + n_refl} steps across the render and the "
            f"reflection run (budget {BORE_WORK_MAX}). At lambda = 1, fs = c0*N/L — so N buys the "
            f"sample rate, not just the grid. Shorten audio_duration or lower N."
        )

    pickup_idx = min(max(0, round(pickup_frac * N)), N - 1)
    # THE TRAP: pace on the TRANSIT, not on f1. One transit is L/c0 = 1.46 ms but f1 = c0/4L is
    # 5.83 ms (four transits), so the string family's f_ref = f1 gives a measured 2.98–3.05 frames
    # per transit at every N — the bounce-and-flip picture aliases into noise, and playback_speed
    # cannot rescue it because the frames are already decimated in *sim* time. f_ref = c0/L makes
    # "frames per period" read as "frames per transit": 11.6–12.8, flat in N. This needs no new
    # mechanism — the bow substitutes f_hard_est here and the tension string f_osc.
    # Generalizable: pace the animation on the timescale of the CLAIM the picture makes, not on the
    # fundamental. They coincide for a string and differ 4x for a bore.
    anim_stride = max(1, round((fs / (C0_AIR / L)) / fpp))
    n_anim = min(n_steps, max(anim_stride, round(anim_win * fs)))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))

    bore.set_state(_bore_bump(bore.x, L, center_frac=0.12))
    run = _run_bore(bore, n_steps, pickup_idx=pickup_idx, anim_stride=anim_stride,
                    frame_until=n_anim)

    frames = np.array(run.frames, dtype=float)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    audio48, peak = _resample_normalize(run.pickup, fs)
    # THE VERDICT RIDES ON THE TOTAL. acoustic alone sheds ~10 % through the bell, so feeding it to
    # the lossless branch would fail the drift check and demote the batch's own headline; the sum is
    # what conserves, and the split is carried alongside for the panel.
    E_total = run.E_ac + run.E_rad
    sim = SimResult(time=np.arange(E_total.size) / fs, energy=E_total, output=None, fs=fs,
                    snapshots=[])

    return {
        "model": "bore",
        "fs_sim": round(fs, 3),
        "lambda": round(info["lam"], 6),
        "grid": {"x": _finite_list(bore.x, 6), "envelope": _finite_list(run.env)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(np.array(run.frame_steps, dtype=float) / fs, 6),
        # Per-frame cumulative radiated energy — the mouth glow, i.e. the field-side dual of the
        # energy panel's radiated curve, so energy leaving is SEEN leaving. Normalized to E0 so the
        # frontend needs no units. Deliberately NOT shared with the audio-run split: different sim
        # strides, different lengths.
        "radiated_frames": _finite_list(
            np.array(run.frame_rad, dtype=float) / (E_total[0] or 1.0), 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(
            sim, sigma_zero=True, oracle_2sigma=0.0,
            split={"acoustic": run.E_ac, "radiated": run.E_rad},
        ),
        "meta": {
            "c": round(C0_AIR, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_steps),
            "n_frames": int(frames.shape[0]),
            "probe_x": round(float(bore.x[pickup_idx]), 4),
            "ends": info["ends"],
            "radiating": bool(info["radiating"]),
            # None, not the slider's echo, when there is no bell: reporting R/Z0 beside an ideal
            # open end would put a live-looking number on a control that is doing nothing.
            "r_ratio": info["ratio"] if info["radiating"] else None,
            "Z0": info["Z0"],
            "R_bell": info["R_bell"],
            "radiated_frac": round(float(run.E_rad[-1] / (E_total[0] or 1.0)), 6),
            "transit": round(L / C0_AIR, 8),
            "frames_per_transit": round(float((fs * L / C0_AIR) / anim_stride), 2),
            "reflection": _bore_reflection_block(info),
            "dispersion": _bore_dispersion_block(L),
            "spectrum": _bore_signature_block(run.pickup, fs, info),
        },
    }


# == reed (wind, Phase D batch 10) =================================================================
#
# The dynamic single reed on batch 9's tube: a self-oscillating exciter, the acoustic dual of the
# bow. It reuses the bore's field type wholesale (pressure along S(x), transit-paced animation,
# per-end rendering via meta.ends) and adds a mouth end type, a genuine-residual balance, and a
# signature panel. Core untouched; ReedBore contains the Bore.

REED_N_MIN, REED_N_MAX = 32, 256
REED_AUDIO_MAX = 1.0
REED_ANIM_MAX = 0.1
REED_WORK_MAX = 300_000       # bounds the RENDER (n_anim + n_audio) only — the sweeps are separate
REED_LAMBDA = 1.0
REED_RADIUS = 0.008
REED_NEWTON_TOL = 1e-10       # pinned and unexposed: the balance is linear in the residual, so a
                              # loose solve degrades the headline directly rather than masking it
REED_SETTLE_FRAC = 0.4        # the settled tail = the last 40 % of a run (the bow's choice)

# The sweeps are pinned OFF the render slider. Measured N-invariant to the 4th significant digit
# (gamma = 0.338 -> 0.00431/0.00429/0.00428/0.00427/0.00426 at N = 64/96/128/200/256; pitch leverage
# 3.05 % at every N), so a fixed modest N gives the same answer at a fraction of the cost — and,
# the correctness half, the headline threshold stops moving when the user drags N.
REED_SWEEP_N = 64
# >= 0.8 s is REQUIRED, not a tolerance. Near the onset the reed critically slows down, and at a
# 0.4 s window gamma = 0.355 reads 0.01123 — an apparently half-speaking point — where converged it
# is 0.00441, i.e. silent. A settling time measured deep in the oscillating regime (0.04 s at
# gamma = 0.5) does not bound the settling time near the bifurcation.
REED_SWEEP_SECS = 0.8
# 1/3 is on the grid EXACTLY — batch 9's anechoic-null lesson (a curve that merely straddles its
# own special value misses the claim it exists to make). Dalmont/Kergomard's small-oscillation
# threshold is gamma ~ 1/3, so the panel must be able to say which side of it the reed is on.
REED_SWEEP_GAMMAS = (0.20, 0.30, 1.0 / 3.0, 0.355, 0.372, 0.45, 0.51, 0.61)
REED_SPEAK_GATE = 0.02        # AC rms / p_closing; the measured floor is ~0.004 and speaking ~0.06
REED_PITCH_SECS = 0.6
REED_PITCH_FREEDS = (2000.0, 3000.0)
_REED_SWEEP_MEMO: dict[tuple, dict[str, Any]] = {}


def _build_reed(p: dict[str, Any]) -> tuple[ReedBore, dict[str, Any]]:
    """Build the clarinet: a ``ReedBore`` on batch 9's ``Bore``.

    ``gamma = p_mouth / p_closing`` is the star control (the dimensionless-coordinate rule's fifth
    customer after ``dT/T0``, ``frac``, ``downswing/depth`` and ``R/Z0``): ``p_mouth`` alone means
    nothing without the reed that resists it, and ``p_closing = mu wr^2 H0`` moves with every reed
    parameter. So the slider is gamma and ``p_mouth`` is derived.

    Bore viscous sigma stays fixed at 0 and unexposed — batch 9's corollary, inherited exactly: it
    is the one channel ``bore.py`` does not book, and exposing it would silently demote the balance
    from a measured residual to an inferred one.
    """
    L = _fnum(p, "L", 0.5)
    N = int(_fnum(p, "N", 128))
    gamma = _fnum(p, "gamma", 0.51)
    f_reed = _fnum(p, "f_reed", 2500.0)
    q_reed = _fnum(p, "q_reed", 4.0)
    ratio_exp = _fnum(p, "bell_ratio_exp", -2.6)
    domain = str(p.get("domain", "radiating"))

    if domain not in ("radiating", "open"):
        raise ParamError(f"reed bore end must be 'radiating' or 'open', got {domain!r}.")
    if not (0.1 <= L <= 2.0):
        raise ParamError(f"L must be in [0.1, 2.0] m, got {L}.")
    if not (REED_N_MIN <= N <= REED_N_MAX):
        raise ParamError(f"N must be in [{REED_N_MIN}, {REED_N_MAX}] for the reed, got {N}.")
    if not (0.02 <= gamma <= 1.2):
        raise ParamError(
            f"gamma = p_mouth/p_closing must be in [0.02, 1.2], got {gamma}. The note speaks above "
            "gamma ~ 0.36; at gamma >= 1 the reed is held statically shut."
        )
    if not (800.0 <= f_reed <= 4000.0):
        raise ParamError(f"f_reed must be in [800, 4000] Hz, got {f_reed}.")
    if not (0.5 <= q_reed <= 20.0):
        raise ParamError(f"q_reed must be in [0.5, 20], got {q_reed}.")
    ratio = 10.0 ** ratio_exp
    if not (BORE_R_RATIO_MIN <= ratio <= BORE_R_RATIO_MAX):
        raise ParamError(
            f"the bell's R/Z0 must be in [{BORE_R_RATIO_MIN}, {BORE_R_RATIO_MAX}], got {ratio:.3e}."
        )

    radiating = domain == "radiating"
    Z0 = RHO0_AIR * C0_AIR / (math.pi * REED_RADIUS * REED_RADIUS)
    fs = C0_AIR / (REED_LAMBDA * (L / N))
    bore = Bore(
        L=L, fs=fs, N=N, radius=REED_RADIUS,
        boundary=("closed", "radiating" if radiating else "open"),
        sigma=0.0, R_bell=(ratio * Z0) if radiating else 0.0,
    )
    reed = _reed_at_gamma(bore, gamma, f_reed, q_reed)
    info = {
        "L": L, "N": N, "fs": fs, "lam": bore.lam, "Z0": Z0, "c0": C0_AIR,
        "gamma": gamma, "f_reed": f_reed, "q_reed": q_reed,
        "p_mouth": reed.p_mouth, "p_closing": reed.p_closing, "H0": reed.H0,
        "ratio": ratio, "radiating": radiating, "R_bell": bore.R_bell,
        # meta.ends[0] is a VIZ label set here, not read off bore._bc_left (which stays "closed" —
        # the reed rides that end's live half-cell DOF and the core requires it).
        "ends": ["reed", "radiating" if radiating else "open"],
    }
    return reed, info


def _reed_at_gamma(bore: Bore, gamma: float, f_reed: float, q_reed: float) -> ReedBore:
    """A ``ReedBore`` blown at a given ``gamma``, deriving ``p_mouth`` from its own ``p_closing``.

    ``p_closing = mu wr^2 H0`` is built from the same defaults ``ReedBore`` uses, so this stays a
    single source of truth for the mapping gamma -> p_mouth across the render and both sweeps.
    """
    mu, H0 = 0.03, 4.0e-4
    p_closing = mu * (2.0 * math.pi * f_reed) ** 2 * H0
    return ReedBore(bore=bore, p_mouth=gamma * p_closing, f_reed=f_reed, q_reed=q_reed,
                    newton_tol=REED_NEWTON_TOL)


class _ReedRun:
    """Per-step telemetry of a reed run — every balance channel, the opening, and the field."""

    def __init__(self, n: int, width: int) -> None:
        self.E = np.empty(n + 1)            # total book: acoustic + reed + booked radiation
        self.E_ac = np.empty(n + 1)
        self.E_rad = np.empty(n + 1)
        self.mouth = np.empty(n + 1)        # cumulative mouth work (active breath in)
        self.jet = np.empty(n + 1)          # cumulative Bernoulli jet loss
        self.damp = np.empty(n + 1)         # cumulative reed (lip) damping work
        self.pickup = np.empty(n + 1)       # mouthpiece pressure — the square wave
        self.far = np.empty(n + 1)          # bell far-field pressure, for the honest caveat
        self.opening = np.empty(n + 1)      # reed channel opening H+ (m)
        self.env = np.zeros(width)
        self.frames: list[NDArray[np.float64]] = []
        self.frame_steps: list[int] = []
        self.frame_open: list[float] = []
        self.frame_rad: list[float] = []


def _run_reed(reed: ReedBore, n_steps: int, *, anim_stride: int, frame_from: int) -> _ReedRun:
    """Step the reed once, capturing everything the three panels need.

    ONE run, and the animation window is captured out of its TAIL via ``frame_from`` — batch 2's
    ``snapshot_from`` lesson, which applies verbatim here: the reed also starts from rest and climbs
    to a limit cycle, so the opening frames are silence, and re-running a second resonator would
    silently double the cost of a root-find-per-step model in a way the work budget cannot see.

    ``simulate()` is unusable here for the same reason it was for the bow and the mallet: it yields
    neither the balance channels, nor the opening trace, nor the stacked field — and all three ARE
    the panels.
    """
    bore = reed.bore
    run = _ReedRun(n_steps, bore.N + 1)

    def _sample(i: int) -> None:
        run.E[i] = reed.energy()
        run.E_ac[i] = bore.acoustic_energy()
        run.E_rad[i] = bore.radiated_energy
        run.mouth[i] = reed.mouth_work
        run.jet[i] = reed.jet_loss
        run.damp[i] = reed.reed_damp_work
        run.pickup[i] = reed.mouthpiece_pressure()
        run.far[i] = reed.pressure()
        run.opening[i] = reed.reed_opening()
        if i >= frame_from:
            np.maximum(run.env, np.abs(bore.p), out=run.env)

    _sample(0)
    for i in range(1, n_steps + 1):
        reed.step()
        _sample(i)
        if i >= frame_from and (i - frame_from) % anim_stride == 0:
            run.frames.append(bore.p.copy())
            run.frame_steps.append(i)
            run.frame_open.append(run.opening[i])
            run.frame_rad.append(run.E_rad[i])
    if not np.all(np.isfinite(run.E)):
        raise ParamError("simulation produced non-finite energy (instability) — adjust parameters.")
    return run


def _reed_ac_level(reed: ReedBore, secs: float) -> float:
    """Settled-tail AC rms of the mouthpiece pressure, normalized by ``p_closing``.

    Mean-REMOVED, and that is load-bearing rather than tidy: below threshold the mouthpiece still
    carries a steady DC pressure (measured 8.2 Pa at gamma = 0.338 against an AC 13.0 Pa), so a raw
    rms reports a "note" where there is only a static pressure. Normalizing by ``p_closing`` then
    collapses every reed geometry onto one dimensionless axis.
    """
    n = max(2, int(secs / reed.k))
    sig = np.empty(n)
    for i in range(n):
        reed.step()
        sig[i] = reed.mouthpiece_pressure()
    tail = sig[int((1.0 - REED_SETTLE_FRAC) * n):]
    return float(np.sqrt(np.mean((tail - tail.mean()) ** 2))) / reed.p_closing


def _reed_sweep_block(info: dict[str, Any]) -> dict[str, Any]:
    """The signature panel: the blowing threshold and the pitch leverage, both at a FIXED N.

    Memoized on every quantity that moves either curve. THE KEY IS THE TRAP: the threshold lives in
    gamma, but ``p_closing = mu wr^2 H0``, so a key missing ``f_reed`` (which IS exposed) would
    return stale numbers the moment the user drags that slider — passing on the defaults and wrong
    on interaction, the "wrong without being broken" family one level up into a cache.
    """
    key = (round(info["L"], 6), round(info["f_reed"], 6), round(info["q_reed"], 6),
           round(info["H0"], 10), round(REED_RADIUS, 6), bool(info["radiating"]),
           round(info["R_bell"], 6), REED_SWEEP_N, REED_SWEEP_SECS)
    if key in _REED_SWEEP_MEMO:
        return _REED_SWEEP_MEMO[key]

    L, f_reed, q_reed = info["L"], info["f_reed"], info["q_reed"]
    end = "radiating" if info["radiating"] else "open"
    fs = C0_AIR / (L / REED_SWEEP_N)

    def _fresh(gamma: float, fr: float = f_reed) -> ReedBore:
        bore = Bore(L=L, fs=fs, N=REED_SWEEP_N, radius=REED_RADIUS, boundary=("closed", end),
                    sigma=0.0, R_bell=info["R_bell"])
        return _reed_at_gamma(bore, gamma, fr, q_reed)

    levels = [_reed_ac_level(_fresh(g), REED_SWEEP_SECS) for g in REED_SWEEP_GAMMAS]
    speaks = [bool(v > REED_SPEAK_GATE) for v in levels]
    # The bracket = the last silent gamma and the first speaking one. Reported, never FAILed below
    # threshold: a reed blown too gently is correct physics, not a broken model (the bow's
    # Schelleng rule and the jawari's grazing rule, third customer).
    lo = hi = None
    for g, sp in zip(REED_SWEEP_GAMMAS, speaks, strict=True):
        if not sp:
            lo = g
        elif hi is None:
            hi = g

    f1 = C0_AIR / (4.0 * L)
    pitches = []
    for fr in REED_PITCH_FREEDS:
        r = _fresh(0.51, fr)
        n = max(4, int(REED_PITCH_SECS / r.k))
        sig = np.empty(n)
        for i in range(n):
            r.step()
            sig[i] = r.displacement_at(1)
        tail = sig[n // 2:]
        got = spectrum.measure_partials_near(tail - tail.mean(), 1.0 / r.k, np.array([f1]))
        pitches.append(float(got[0]))
    leverage = None
    if all(np.isfinite(pitches)) and pitches[0] > 0:
        # Stated as LEVERAGE, not as the suite's binary "< 6 %": a +50 % reed sweep buys ~+3 % of
        # pitch, and the residual trend is itself physics (reed compliance acts as an end
        # correction, so a stiffer reed lands closer to c/4L).
        leverage = {
            "f_reed": [round(f, 1) for f in REED_PITCH_FREEDS],
            "reed_change_pct": round(100.0 * (REED_PITCH_FREEDS[1] / REED_PITCH_FREEDS[0] - 1), 1),
            "f0": [round(f, 3) for f in pitches],
            "pitch_change_pct": round(100.0 * (pitches[1] / pitches[0] - 1.0), 2),
            "cents": [round(1200.0 * math.log2(f / f1), 1) for f in pitches],
        }

    block = {
        "sweep_N": REED_SWEEP_N,
        "sweep_secs": REED_SWEEP_SECS,
        "gamma": list(REED_SWEEP_GAMMAS),
        "level": [round(v, 6) for v in levels],
        "speaks": speaks,
        "gate": REED_SPEAK_GATE,
        "bracket": [lo, hi],
        "pitch": leverage,
        "f1": round(f1, 3),
        # The bracket MOVES with the bell, and the panel must say at which bell it was measured.
        # Measured at N = 64: R/Z0 = 3e-4 and 2.5e-3 bracket (0.30, 0.338]; 1e-2 gives
        # (0.338, 0.355]; 2.4e-2 gives (0.355, 0.372]; and by 6.3e-2 the note never speaks at all.
        # That is correct physics — a lossier bell needs a harder blow — and it is exactly why the
        # memo key has to carry R_bell.
        "r_ratio": info["ratio"] if info["radiating"] else None,
    }
    _REED_SWEEP_MEMO[key] = block
    return block


def _reed_beating_block(opening: NDArray[np.float64], fs: float, f1: float) -> dict[str, Any]:
    """The reed's closure statistics over the settled tail: duty first, debounced count beside it.

    THE DUTY IS THE PRIMARY NUMBER because it needs no event definition at all. A raw per-period
    closure count measures the CHATTER, not the event: at gamma ~ 0.5 each period holds a 35-sample
    precursor, a 66-sample re-opening and then the main 291-sample closure, so counting zero
    crossings reports ~1.94 closures per period for what is plainly one beat. Merging episodes
    separated by less than 10 % of a period recovers 1.00.
    """
    tail = np.asarray(opening[int((1.0 - REED_SETTLE_FRAC) * opening.size):], dtype=float)
    if tail.size < 4 or f1 <= 0:
        return {"beats": False, "duty": 0.0, "per_period": None}
    period = max(1, int(round(fs / f1)))
    shut = tail <= 0.0
    duty = float(np.mean(shut))

    gap = max(1, int(0.10 * period))
    episodes, i = 0, 0
    while i < shut.size:
        if shut[i]:
            episodes += 1
            j = i
            while j < shut.size:
                if shut[j]:
                    j += 1
                    continue
                run_open = 0
                while j + run_open < shut.size and not shut[j + run_open]:
                    run_open += 1
                if run_open >= gap:
                    break
                j += run_open
            i = j
        else:
            i += 1
    periods = tail.size / period
    return {
        "beats": bool(duty > 0.0),
        "duty": round(duty, 4),
        "per_period": round(episodes / periods, 3) if periods > 0 else None,
        "min_opening": float(tail.min()),
    }


def _build_payload_reed(p: dict[str, Any]) -> dict[str, Any]:
    playback_speed = _fnum(p, "playback_speed", 0.02)
    audio_dur = _fnum(p, "audio_duration", 0.5)
    anim_win = _fnum(p, "animation_window", 0.03)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < audio_dur <= REED_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {REED_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= REED_ANIM_MAX):
        raise ParamError(f"animation_window must be in (0, {REED_ANIM_MAX}] s, got {anim_win}.")

    reed, info = _build_reed(p)
    fs, L = info["fs"], info["L"]
    f1 = C0_AIR / (4.0 * L)
    n_steps = max(1, round(audio_dur * fs))
    if n_steps > REED_WORK_MAX:
        raise ParamError(
            f"this render needs {n_steps} steps (budget {REED_WORK_MAX}). At lambda = 1, "
            f"fs = c0*N/L — so N buys the sample rate, not just the grid. Shorten audio_duration "
            f"or lower N."
        )

    # Pace on the TRANSIT, not f1 — batch 9's rule, and the probe confirms it transfers unchanged
    # (f_ref = c0/L gives 11.6–12.8 frames/transit at every N; f_ref = f1 gives ~3.0 and aliases
    # the travelling pressure step, which is this batch's best picture).
    anim_stride = max(1, round((fs / (C0_AIR / L)) / fpp))
    n_anim = min(n_steps, max(anim_stride, round(anim_win * fs)))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    frame_from = max(0, n_steps - n_anim)

    run = _run_reed(reed, n_steps, anim_stride=anim_stride, frame_from=frame_from)

    frames = np.array(run.frames, dtype=float)
    field_amp = float(np.max(np.abs(frames))) if frames.size else 0.0
    audio48, peak = _resample_normalize(run.pickup, fs)
    sim = SimResult(time=np.arange(run.E.size) / fs, energy=run.E, output=None, fs=fs,
                    snapshots=[])

    tail0 = int((1.0 - REED_SETTLE_FRAC) * run.pickup.size)
    tail = run.pickup[tail0:]
    ac_level = float(np.sqrt(np.mean((tail - tail.mean()) ** 2))) / info["p_closing"]
    spoke = bool(ac_level > REED_SPEAK_GATE)
    E_stored = run.E - run.E_rad          # acoustic + reed: plateaus in the limit cycle
    breath = float(run.mouth[-1]) or 1.0

    return {
        "model": "reed",
        "fs_sim": round(fs, 3),
        "lambda": round(info["lam"], 6),
        "grid": {"x": _finite_list(reed.bore.x, 6), "envelope": _finite_list(run.env)},
        "frames": {
            "b64": _b64f32(frames.ravel()),
            "n_frames": int(frames.shape[0]),
            "width": int(frames.shape[1]) if frames.ndim == 2 else 0,
            "dims": 1,
        },
        "frame_times": _finite_list(np.array(run.frame_steps, dtype=float) / fs, 6),
        "opening_frames": _finite_list(np.array(run.frame_open, dtype=float), 9),
        "radiated_frames": _finite_list(
            np.array(run.frame_rad, dtype=float) / (run.E[-1] or 1.0), 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        # sigma_zero is passed through but the measured-loss branch ignores it for the verdict:
        # the reed's jet and lip-damping channels are on in EVERY regime, so the residual is
        # genuine in every regime and there is nothing to gate. See _balance_verdict.
        "energy": _energy_block(
            sim, sigma_zero=True, oracle_2sigma=0.0,
            balance_work=run.mouth,
            measured_loss={"jet": run.jet, "reed damping": run.damp},
        ),
        "meta": {
            "c": round(C0_AIR, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_steps),
            "n_frames": int(frames.shape[0]),
            "ends": info["ends"],
            "radiating": bool(info["radiating"]),
            "r_ratio": info["ratio"] if info["radiating"] else None,
            "Z0": info["Z0"],
            "gamma": round(info["gamma"], 4),
            "p_mouth": round(info["p_mouth"], 2),
            "p_closing": round(info["p_closing"], 2),
            "H0": info["H0"],
            "f_reed": round(info["f_reed"], 1),
            "q_reed": round(info["q_reed"], 3),
            "transit": round(L / C0_AIR, 8),
            "frames_per_transit": round(float((fs * L / C0_AIR) / anim_stride), 2),
            "fallbacks": int(reed.fallbacks),
            # Where the breath goes. The measured ledger IS the batch's structural point, so it is
            # reported as fractions of the mouth work rather than left implicit in the curves.
            "budget": {
                "mouth_work": float(run.mouth[-1]),
                "jet_frac": round(float(run.jet[-1]) / breath, 4),
                "damping_frac": round(float(run.damp[-1]) / breath, 4),
                "radiated_frac": round(float(run.E_rad[-1]) / breath, 4),
                "stored_frac": round(float(E_stored[-1] - E_stored[0]) / breath, 4),
            },
            "speaks": spoke,
            "ac_level": round(ac_level, 6),
            "envelope_ratio": round(float(run.env[0] / max(run.env[-1], 1e-12)), 2),
            "beating": _reed_beating_block(run.opening, fs, f1),
            "spectrum": _reed_signature_block(run.pickup, run.far, fs, f1, spoke),
            "sweep": _reed_sweep_block(info),
        },
    }


def _reed_signature_block(pickup: NDArray[np.float64], far: NDArray[np.float64], fs: float,
                          f1: float, spoke: bool) -> dict[str, Any]:
    """Odd-harmonic dominance off the mouthpiece, with the far-field caveat measured beside it.

    The mouthpiece is the audio because it IS the square wave (measured crest 1.116 against the far
    field's 3.684, and f1/2f1 = 460 against 25.5). But it is emphatically not what a listener hears
    — radiation differentiates — so the far-field numbers ship alongside rather than letting "the
    iconic clarinet tone" quietly mean the pressure inside the mouthpiece.
    """
    out: dict[str, Any] = {"kind": "reed", "applies": bool(spoke)}
    if not spoke:
        out["note"] = "below threshold — no tone to analyse"
        return out
    tail = np.asarray(pickup[pickup.size // 2:], dtype=float)
    ac = tail - tail.mean()
    freqs, mag, _ = spectrum.magnitude_spectrum(ac, fs)
    df = float(freqs[1] - freqs[0])

    def _peak(f: float, m: NDArray[np.float64] = mag) -> float:
        i = int(round(f / df))
        return float(m[max(1, i - 3):i + 4].max())

    n_show = max(8, int(6.5 * f1 / df))
    out["freq"] = _finite_list(freqs[1:n_show], 3)
    out["mag"] = _finite_list(mag[1:n_show], 6)
    out["markers"] = [round(m * f1, 3) for m in (1, 2, 3, 4, 5)]
    out["odd_even"] = round(_peak(f1) / max(_peak(2 * f1), 1e-30), 1)
    out["third_second"] = round(_peak(3 * f1) / max(_peak(2 * f1), 1e-30), 2)
    rms = float(np.sqrt(np.mean(ac**2)))
    out["crest"] = round(float(np.abs(ac).max() / rms), 3) if rms > 0 else None
    ftail = np.asarray(far[far.size // 2:], dtype=float)
    fac = ftail - ftail.mean()
    frms = float(np.sqrt(np.mean(fac**2)))
    fpeak = float(np.abs(fac).max())
    # An ideal open end RADIATES NOTHING, so there is no far field to compare the mouthpiece with.
    # Emitting the comparison anyway divides by ~0 and puts "1.6e+33x quieter (crest null)" on
    # screen — a lying number where the honest statement is "not applicable". Withdraw it instead,
    # the batch-9 `applies = false` pattern (nothing to measure is not the same as wrong). Every
    # backend test passed with the bogus number in place; only the rendered readout showed it.
    if frms > 0.0 and fpeak > 1e-12 * max(float(np.abs(ac).max()), 1e-30):
        out["far_field"] = {
            "peak": round(float(np.abs(ftail).max()), 4),
            "crest": round(fpeak / frms, 3),
            "quieter_by": round(float(np.abs(ac).max()) / fpeak, 1),
        }
    else:
        out["far_field"] = None
        out["far_note"] = ("an ideal open end radiates nothing — there is no far field to "
                           "compare the mouthpiece with")
    return out


# == membrane (2D, Phase B) ========================================================================
#
# NOTE the string family and the 1-D bore live above this line; the 2D family starts here.
#
# The 2D path is split off from the string path (above) so the string contract stays bit-for-bit
# unchanged. What differs: frames are 2D heatmap fields (decimated to a <= DISPLAY_MAX display
# grid — the data-size trap), the excitation/pickup are (x, y) fractions, and the modal panel is a
# *magnitude spectrum with mode-marker lines* rather than per-partial cents bars. The reason for
# the latter (advisor review 3): the continuum Bessel oracle is off by ~O(h) staircase (≈9 cents
# at N=128) *by design*, so scoring it as "error" reads as a bug; and an off-centre struck drum
# rings high/odd modes only weakly, so `measure_partials_near` would lock onto noise for unexcited
# targets and report
# confident-but-meaningless cents. Instead we show the FFT with vertical lines at the **discrete**
# eigenfreqs (peaks landing on lines = self-consistency) and fainter lines at the continuum oracle
# (the staircase offset, shown not scored). Two headline numbers: the robust fundamental
# detected-vs-discrete cents, and the discrete-vs-continuum "geometry tier" gap. Energy stays the
# hard pass/fail signature (conservation σ=0 / passivity σ>0), reused verbatim from the string
# path.


def _build_membrane(p: dict[str, Any]) -> tuple[Membrane, float, float, float, str, dict[str, Any]]:
    """Construct a fresh :class:`Membrane` from params.

    Returns ``(res, c, fs, sigma, domain, geom)``. ``fs = c/(λ·h)`` reproduces the requested λ
    exactly (``h`` from the geometry: ``2·radius/N`` for a disk, ``Lx/N`` for a rectangle).
    ``geom`` holds the *snapped* geometry read back off the resonator (the ctor snaps ``Ly`` to
    whole cells).
    """
    domain = str(p.get("domain", "circle"))
    if domain not in ("circle", "rectangle"):
        raise ParamError(f"domain must be 'circle' or 'rectangle', got {domain!r}.")
    T = _fnum(p, "T", 200.0)
    rho = _fnum(p, "rho", 0.005)
    lam = _fnum(p, "lambda", 0.6)
    sigma = _fnum(p, "sigma", 0.0)
    try:
        N = int(p.get("N", 80))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= MEMBRANE_N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {MEMBRANE_N_MAX}] for the membrane, got {N}.")
    if min(T, rho) <= 0:
        raise ParamError("T, rho must both be positive.")
    if sigma < 0:
        raise ParamError(f"sigma (loss) must be >= 0, got {sigma}.")
    if not (0.0 < lam <= MEMBRANE_LAMBDA_MAX + 1e-9):
        raise ParamError(
            f"lambda must be in (0, 1/sqrt(2) = {MEMBRANE_LAMBDA_MAX:.4f}] (2D CFL), got {lam}."
        )

    c = math.sqrt(T / rho)
    if domain == "circle":
        radius = _fnum(p, "radius", 0.5)
        if radius <= 0:
            raise ParamError(f"radius must be positive, got {radius}.")
        h = 2.0 * radius / N
        fs = c / (lam * h)
        res = Membrane(domain="circle", T=T, rho=rho, fs=fs, N=N, radius=radius, sigma=sigma)
        geom = {"radius": radius}
    else:
        Lx = _fnum(p, "Lx", 1.0)
        Ly = _fnum(p, "Ly", 1.0)
        if min(Lx, Ly) <= 0:
            raise ParamError("Lx, Ly must both be positive.")
        h = Lx / N
        fs = c / (lam * h)
        res = Membrane(domain="rectangle", T=T, rho=rho, fs=fs, N=N, Lx=Lx, Ly=Ly, sigma=sigma)
        geom = {"Lx": float(res.Lx), "Ly": float(res.Ly)}  # snapped Ly read back off the ctor

    # Geometry gate (constructing res is cheap — it builds L but not eigsh): a thin rectangle's
    # n_live ~ N²·(Ly/Lx) explodes well before N_MAX bites. The per-step FDTD cost is bounded
    # separately by the work budget in _build_payload_membrane (which also covers the fs blow-up).
    if res.n_live > MEMBRANE_NLIVE_MAX:
        raise ParamError(
            f"this geometry has {res.n_live} interior nodes (> {MEMBRANE_NLIVE_MAX}); reduce N or "
            "use a less extreme aspect ratio."
        )
    return res, c, fs, sigma, domain, geom


def _frac_to_xy(domain: str, fx: float, fy: float, geom: dict[str, Any]) -> tuple[float, float]:
    """Map (fx, fy) in (0,1)² to a physical point. Disk: centred grid [-r, r]; rect: [0, L]."""
    if domain == "circle":
        a = geom["radius"]
        return (2.0 * fx - 1.0) * a, (2.0 * fy - 1.0) * a
    return fx * geom["Lx"], fy * geom["Ly"]


def _length_scale(domain: str, geom: dict[str, Any]) -> float:
    """Characteristic length for the excitation width (disk radius / shorter rectangle side)."""
    return geom["radius"] if domain == "circle" else min(geom["Lx"], geom["Ly"])


def _discrete_eigenfreqs(res: Membrane, c: float, k_request: int) -> NDArray[np.float64]:
    """Lowest discrete eigenfrequencies (Hz) via ``eigsh(-L)`` →
    ``discrete_membrane_eigenfrequency``.

    These are the frequencies the time-stepper *actually* rings at (the operator's own spectrum), so
    they are the honest marker lines for the FFT panel. ``k`` is clamped below ``n_live`` so a small
    grid can't crash the eigensolver (advisor catch).
    """
    k = min(int(k_request), res.n_live - 1)
    if k < 1:
        return np.asarray([], dtype=float)
    lam = eigsh(-res.L, k=k, sigma=0.0, which="LM", return_eigenvectors=False)
    f = modal.discrete_membrane_eigenfrequency(np.sort(np.asarray(lam, dtype=float)), c, res.k)
    return np.sort(np.asarray(f, dtype=float))


def _membrane_continuum_oracle(
    domain: str, c: float, geom: dict[str, Any], n_modes: int
) -> NDArray[np.float64]:
    """Lowest continuum (analytic) membrane frequencies — the *geometry-tier* reference.

    Circle: Bessel ``f_{mn}`` expanded by degeneracy (m>=1 are cos/sin pairs) so the sorted list
    lines up with the numerical spectrum. Rectangle: ``f_{mn} = (c/2)√((m/Lx)²+(n/Ly)²)`` over a
    grid of ``(m, n)``. For the disk this sits ~O(h) staircase below the discrete lines (shown, not
    scored); for the rectangle it nearly coincides (O(h²)).
    """
    if n_modes < 1:
        return np.asarray([], dtype=float)
    if domain == "circle":
        freqs: list[float] = []
        for _m, _n, f, deg in modal.circular_membrane_freqs(c, geom["radius"], n_modes):
            freqs.extend([f] * deg)
        return np.sort(np.asarray(freqs, dtype=float))[:n_modes]
    Lx, Ly = geom["Lx"], geom["Ly"]
    rng = range(1, n_modes + 1)
    modes = [(m, n) for m in rng for n in rng]
    f = modal.rectangular_membrane_freqs(c, Lx, Ly, modes)
    return np.sort(np.asarray(f, dtype=float))[:n_modes]


def _decimate_field_mask(
    frames_full: NDArray[np.float64], mask_full: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Spatially decimate a ``(nf, ny, nx)`` field and its ``(ny, nx)`` mask to the display grid.

    The 2D data-size trap (shared by every heatmap model): the same stride is applied to the field
    AND the mask so they stay aligned, and the shipped ``field_amp`` / colour scale come from the
    decimated frames the frontend actually receives (not the full field). Cf. advisor review 3.
    """
    ny_full, nx_full = mask_full.shape
    stride = max(1, math.ceil(max(ny_full, nx_full) / DISPLAY_MAX))
    return frames_full[:, ::stride, ::stride], mask_full[::stride, ::stride]


def _pool_band(
    freqs: NDArray[np.float64], mag: NDArray[np.float64], fmin: float, fmax: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Slice ``[fmin, fmax]`` out of a magnitude spectrum, max-pool to ~:data:`N_SPEC_POINTS`, and
    normalize to 0..1. Max-pooling (not mean) so spectral *peaks* survive the decimation.

    Split out of :func:`_pooled_spectrum` so a caller that has already paid for an FFT (the phantom
    panel, which needs one dense grid for two different bands) can pool it without running another.
    """
    keep = (freqs >= fmin) & (freqs <= fmax)
    f, m = freqs[keep], mag[keep]
    if m.size == 0:
        return None
    npts = min(N_SPEC_POINTS, int(m.size))
    edges = np.linspace(0, m.size, npts + 1).astype(int)
    f_ds = np.empty(npts)
    m_ds = np.empty(npts)
    for i in range(npts):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        f_ds[i] = f[lo:hi].mean()
        m_ds[i] = m[lo:hi].max()
    mmax = float(m_ds.max()) if m_ds.size else 0.0
    if mmax > 0:
        m_ds = m_ds / mmax
    return f_ds, m_ds


def _pooled_spectrum(
    pickup: NDArray[np.float64], fs: float, fmax: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Magnitude spectrum over ``[0, fmax]`` max-pooled to ~:data:`N_SPEC_POINTS` points.

    Shared by every heatmap model's spectrum panel. Returns ``(freq, mag)`` or ``None`` if empty.
    """
    freqs, mag, _ = spectrum.magnitude_spectrum(pickup, fs)
    return _pool_band(freqs, mag, 0.0, fmax)


def _zero_cross_fundamental(sig: NDArray[np.float64], fs: float) -> float:
    """Fundamental from the mean zero-crossing spacing — robust to the von Kármán pitch shift.

    The hardened fundamental drifts +tens of % with amplitude, so an FFT window on the *linear*
    frequency would miss it; zero-crossing spacing tracks the moving fundamental directly (the
    method proven in ``test_pitch_glide_hardening`` / ``diagnose_vk_plate``).
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    zc = np.where(np.diff(np.signbit(sig)))[0]
    if len(zc) < 3:
        return float("nan")
    return fs / (2.0 * float(np.mean(np.diff(zc))))


def _modal_spectrum_block(
    pickup: NDArray[np.float64],
    fs: float,
    f_disc: NDArray[np.float64],
    f_cont: NDArray[np.float64],
    kind: str,
) -> dict[str, Any] | None:
    """Magnitude spectrum (max-pooled) + mode-marker lines + the two headline cents numbers.

    Shared by the *linear* heatmap models (``membrane``, ``plate``): the FFT panel marks the
    **discrete** eigenfreqs (``f_disc`` — where the stepper actually rings) and the fainter
    continuum oracle (``f_cont`` — the geometry tier, shown not scored). ``kind`` selects the
    frontend readout wording (Bessel/rect staircase vs the plate's tight SS tier).
    """
    if f_disc.size == 0:
        return None
    fmax = float(np.max(f_disc)) * 1.25
    pooled = _pooled_spectrum(pickup, fs, fmax)
    if pooled is None:
        return None
    f_ds, m_ds = pooled

    # Robust headline: the fundamental is always excited -> detected-vs-discrete cents.
    f1 = float(f_disc[0])
    detected = float(
        spectrum.measure_partials_near(pickup, fs, np.asarray([f1]), search_hz=0.3 * f1)[0]
    )
    cents_fund = float(modal.cents(detected, f1)) if math.isfinite(detected) else None
    cents_geom = (
        float(modal.cents(f1, float(f_cont[0]))) if f_cont.size and f_cont[0] > 0 else None
    )

    return {
        "kind": kind,
        "freq": _finite_list(f_ds, 3),
        "mag": _finite_list(m_ds, 5),
        "fmax": round(fmax, 3),
        "modes_discrete": _finite_list(f_disc, 4),
        "modes_continuum": _finite_list(f_cont, 4),
        "f1_discrete": round(f1, 4),
        "f1_detected": round(detected, 4) if math.isfinite(detected) else None,
        "cents_fundamental": round(cents_fund, 4) if cents_fund is not None else None,
        "cents_geometry": round(cents_geom, 4) if cents_geom is not None else None,
    }


def _membrane_spectrum_block(
    pickup: NDArray[np.float64],
    fs: float,
    f_disc: NDArray[np.float64],
    f_cont: NDArray[np.float64],
) -> dict[str, Any] | None:
    """Membrane spectrum panel (thin wrapper over :func:`_modal_spectrum_block`, membrane kind)."""
    return _modal_spectrum_block(pickup, fs, f_disc, f_cont, "membrane")


def _build_payload_membrane(p: dict[str, Any]) -> dict[str, Any]:
    audio_dur = _fnum(p, "audio_duration", 2.0)
    anim_win = _fnum(p, "animation_window", 0.05)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    amplitude = _fnum(p, "amplitude", 1e-3)
    pluck_fx = _fnum(p, "pluck_x", 0.4)
    pluck_fy = _fnum(p, "pluck_y", 0.55)
    pluck_wfrac = _fnum(p, "pluck_width", 0.45)
    pickup_fx = _fnum(p, "pickup_x", 0.65)
    pickup_fy = _fnum(p, "pickup_y", 0.6)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= MEMBRANE_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {MEMBRANE_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    for name, v in (("pluck_x", pluck_fx), ("pluck_y", pluck_fy),
                    ("pickup_x", pickup_fx), ("pickup_y", pickup_fy)):
        if not (0.0 < v < 1.0):
            raise ParamError(f"{name} must be in (0, 1), got {v}.")
    if not (0.0 < pluck_wfrac <= 1.0):
        raise ParamError(f"pluck_width must be in (0, 1], got {pluck_wfrac}.")

    res, c, fs, sigma, domain, geom = _build_membrane(p)

    # Work budget: the FDTD cost is n_live × steps (both the audio run and the slow-mo animation
    # run), with a cache penalty at high n_live. fs ∝ 1/min_dimension, so a small drum or fine
    # grid inflates the step count past N_MAX/n_live alone — bound the product before stepping
    # anything.
    n_audio = max(1, round(audio_dur * fs))
    n_anim_est = max(1, round(anim_win * fs))
    work = res.n_live * (n_audio + n_anim_est)
    if work > MEMBRANE_WORK_MAX:
        raise ParamError(
            f"this configuration needs ~{work / 1e6:.0f}M node-steps (over the "
            f"~{MEMBRANE_WORK_MAX / 1e6:.0f}M budget); reduce N, raise lambda, shorten the "
            "audio/animation, or enlarge the geometry."
        )

    # Mode markers + the f1 used for the animation stride (the lowest discrete eigenfrequency).
    f_disc = _discrete_eigenfreqs(res, c, N_MEMBRANE_MODES)
    f_cont = _membrane_continuum_oracle(domain, c, geom, int(f_disc.size))
    f1 = float(f_disc[0]) if f_disc.size else c / (2.0 * _length_scale(domain, geom))

    # --- audio run: full duration, single (x, y) pickup node, no snapshots -----------------------
    wc = pluck_wfrac * _length_scale(domain, geom)
    pcx, pcy = _frac_to_xy(domain, pluck_fx, pluck_fy, geom)
    res.set_state(raised_cosine_2d(res.X, res.Y, (pcx, pcy), wc, amplitude=amplitude))
    pickup_idx = res.pickup_index_at(*_frac_to_xy(domain, pickup_fx, pickup_fy, geom))
    audio_res = simulate(res, num_steps=n_audio, pickup_index=pickup_idx)
    pickup = np.asarray(audio_res.output, dtype=float)
    if not np.all(np.isfinite(pickup)):
        raise ParamError(
            "simulation produced non-finite output (instability) — adjust parameters."
        )

    # --- animation run: fresh membrane, short window, fundamental-resolving stride (catch #2) -----
    anim = _build_membrane(p)[0]
    anim.set_state(raised_cosine_2d(anim.X, anim.Y, (pcx, pcy), wc, amplitude=amplitude))
    anim_stride = max(1, round((fs / f1) / fpp))
    n_anim = max(anim_stride, round(anim_win * fs))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    anim_res = simulate(anim, num_steps=n_anim, snapshot_stride=anim_stride)
    frames_full = np.array([st for _, st in anim_res.snapshots], dtype=float)  # (nf, ny, nx)
    frame_steps = np.array([i for i, _ in anim_res.snapshots], dtype=float)

    # --- spatial decimation to the display grid (the 2D data-size trap) --------------------------
    # Decimate field AND mask with the SAME stride/offset so they stay aligned; field_amp + the
    # heatmap colour scale come from the decimated frames we actually ship (advisor catch).
    ny_full, nx_full = res.mask.shape
    stride_s = max(1, math.ceil(max(ny_full, nx_full) / DISPLAY_MAX))
    frames_dec = frames_full[:, ::stride_s, ::stride_s]
    mask_dec = res.mask[::stride_s, ::stride_s]
    nf, ny_dec, nx_dec = frames_dec.shape
    field_amp = float(np.max(np.abs(frames_dec))) if frames_dec.size else 0.0

    audio48, peak = _resample_normalize(pickup, fs)
    ext_x, ext_y = (2.0 * geom["radius"], 2.0 * geom["radius"]) if domain == "circle" \
        else (geom["Lx"], geom["Ly"])

    return {
        "model": "membrane",
        "domain": domain,
        "fs_sim": round(fs, 3),
        "lambda": round(float(getattr(res, "lam", float("nan"))), 6),
        "grid": {
            "dims": 2, "nx": int(nx_dec), "ny": int(ny_dec),
            "extent_x": round(ext_x, 6), "extent_y": round(ext_y, 6), "domain": domain,
        },
        "frames": {
            "b64": _b64f32(frames_dec.ravel()),
            "n_frames": int(nf), "nx": int(nx_dec), "ny": int(ny_dec),
            "width": int(nx_dec), "dims": 2,
        },
        "mask": {"b64": _b64u8(mask_dec.ravel()), "nx": int(nx_dec), "ny": int(ny_dec)},
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(audio_res, sigma == 0.0, 2.0 * sigma),
        "meta": {
            "c": round(c, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_audio),
            "n_frames": int(nf),
            "spectrum": _membrane_spectrum_block(pickup, fs, f_disc, f_cont),
        },
    }


# == Mallet → membrane collision (model #7) ========================================================
#
# The FIRST contact model in the viewer: a lumped mass in one-sided nonlinear felt contact striking
# a drumhead. It reuses the membrane's 2D heatmap machinery wholesale (decimated field + mask), with
# three differences from the membrane path: (1) the head starts at REST — the mallet strikes it, so
# there is no raised-cosine IC; (2) the verdict is CONSERVATION, not a pluck's decay — it rides the
# ordinary energy panel (lossless → drift < 1e-10; σ>0 or hysteresis>0 → passivity), but with the
# 2σ decay oracle DROPPED (``decay_oracle=False``): the total energy includes the mallet's KE and
# after the felt separates the mallet flies off at ~constant velocity, so E sits on a near-constant
# ½M·v₀² floor and a fitted "2σ" would be a lie; (3) the headline is not the tone but the CONTACT —
# a point mass is an inefficient membrane exciter (restitution ≈ 1, the head keeps ~0.01 %), which
# the second panel shows and never tunes away. The core solve lives in :class:`MalletMembrane`.


def _build_mallet(
    p: dict[str, Any],
) -> tuple[MalletMembrane, Membrane, float, float, float, str, dict[str, Any], dict[str, Any]]:
    """Construct a fresh ``(MalletMembrane, Membrane)`` from params.

    Returns ``(mallet, membrane, c, fs, sigma, domain, geom, info)``. The membrane is force-free at
    construction (``u = 0``) — the mallet strikes a drum at REST — unlike every other 2D model,
    which sets a raised-cosine IC. ``info`` carries the derived contact quantities (mass, felt
    stiffness, strike velocity, the snapped strike position in fractions, and the rigid-wall
    steps-per-contact resolution estimate).
    """
    try:
        n_req = int(p.get("N", 60))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc
    if n_req > MALLET_N_MAX:
        raise ParamError(
            f"N must be <= {MALLET_N_MAX} for the mallet (each step also runs a contact "
            f"root-find), got {n_req}."
        )

    mem, c, fs, sigma, domain, geom = _build_membrane(p)

    mass = _fnum(p, "mass", 0.02)
    stiffness = _fnum(p, "stiffness", 5.0e4)
    alpha = _fnum(p, "alpha", 2.3)
    hysteresis = _fnum(p, "hysteresis", 0.0)
    v0 = _fnum(p, "strike_velocity", 3.0)
    strike_fx = _fnum(p, "pluck_x", 0.5)   # the strike (x, y) reuses the shared 2D strike sliders
    strike_fy = _fnum(p, "pluck_y", 0.5)

    if mass <= 0.0:
        raise ParamError(f"mallet mass must be > 0, got {mass}.")
    if stiffness <= 0.0:
        raise ParamError(f"felt stiffness must be > 0, got {stiffness}.")
    if alpha < 1.0:
        raise ParamError(f"felt exponent alpha must be >= 1, got {alpha}.")
    if hysteresis < 0.0:
        raise ParamError(f"hysteresis must be >= 0, got {hysteresis}.")
    if v0 <= 0.0:
        raise ParamError(f"strike velocity must be > 0, got {v0}.")
    for name, v in (("pluck_x", strike_fx), ("pluck_y", strike_fy)):
        if not (0.0 < v < 1.0):
            raise ParamError(f"{name} (strike position) must be in (0, 1), got {v}.")

    sx, sy = _frac_to_xy(domain, strike_fx, strike_fy, geom)
    mal = MalletMembrane(
        membrane=mem, mass=mass, stiffness=stiffness, alpha=alpha, hysteresis=hysteresis,
        strike_x=sx, strike_y=sy, strike_velocity=v0,
    )

    # Rigid-wall contact-resolution estimate (the felt half-period must span several steps or the
    # strike aliases). It UNDERSTATES the coupled contact duration — on a yielding membrane the head
    # relaxes and contact lasts ~20× the wall's π√(M/K) — but it is the right guard: it bounds
    # resolution of the stiff felt oscillation, which is what aliases. Energy conserves even when
    # under-resolved (the whole point of the energy method), so this is a NOTE, not an error.
    steps_per_contact = float(np.pi * np.sqrt(mass / stiffness) * fs)
    info = {
        "mass": mass, "stiffness": stiffness, "alpha": alpha, "hysteresis": hysteresis,
        "strike_velocity": v0,
        "strike_fx": round(float(_xy_to_frac(domain, mal.x_strike, mal.y_strike, geom)[0]), 4),
        "strike_fy": round(float(_xy_to_frac(domain, mal.x_strike, mal.y_strike, geom)[1]), 4),
        "steps_per_contact": round(steps_per_contact, 1),
        "resolved": bool(steps_per_contact >= 8.0),
    }
    return mal, mem, c, fs, sigma, domain, geom, info


def _xy_to_frac(domain: str, x: float, y: float, geom: dict[str, Any]) -> tuple[float, float]:
    """Inverse of :func:`_frac_to_xy`: physical (x, y) → (fx, fy) in (0,1)² (strike marker)."""
    if domain == "circle":
        a = geom["radius"]
        return (x / a + 1.0) / 2.0, (y / a + 1.0) / 2.0
    return x / geom["Lx"], y / geom["Ly"]


def _mallet_contact_block(
    time: NDArray[np.float64],
    mvel: NDArray[np.float64],
    head_e: NDArray[np.float64],
    force: NDArray[np.float64],
    in_contact: NDArray[np.bool_],
    ke0: float,
    fs: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    """The contact-episode verdict — the mallet's headline (``kind == "mallet"``).

    A point mass is an *inefficient* membrane exciter (physics, not a bug — advisor-confirmed): the
    local reactive near-field forms a dimple that relaxes and returns almost all the energy to the
    mallet, so it bounces off with **restitution ≈ 1** and the head keeps **~0.01 %** of the strike.
    This block reports that story rather than a tone spectrum (a soft felt low-passes the strike, so
    per-mode partial-locking would lock onto noise). Retention is read at the **peak** (during
    contact the transient dimple holds ~65 %; post-separation ~0.01 %), per the core signature test.
    """
    n = int(mvel.size)
    v0 = float(info["strike_velocity"])

    # Find the contact episode: first sustained contact start, then its first sustained end. The
    # in_contact flag can flicker at grazing, so require a run of >= GRAZE steps on each side.
    graze = 10
    start = None
    for i in range(n - graze):
        if in_contact[i] and bool(np.all(in_contact[i:i + graze])):
            start = i
            break
    sep = None
    if start is not None:
        for i in range(start + 1, n - graze):
            if (not in_contact[i]) and bool(not np.any(in_contact[i:i + graze])):
                sep = i
                break

    if start is None:
        # Never made real contact (K or v0 too small for the felt to load) — honest null, not a bug.
        restitution = 1.0
        contact_ms: float | None = None
        separated = False
        zoom_end = n - 1
    elif sep is None:
        # Still in contact at the end of the window — report what we have, flag it.
        restitution = abs(float(np.mean(mvel[-graze:]))) / v0
        contact_ms = None
        separated = False
        zoom_end = n - 1
    else:
        restitution = abs(float(np.mean(mvel[sep:min(sep + graze, n)]))) / v0
        contact_ms = float(time[sep] * 1e3)
        separated = True
        zoom_end = min(n - 1, int(sep * 2.0))   # auto-zoom to ~2× the contact episode

    peak_head_pct = float(np.max(head_e) / ke0 * 100.0) if ke0 > 0 else 0.0
    tail = head_e[int(n * 0.9):]
    final_head_pct = float(np.mean(tail) / ke0 * 100.0) if (ke0 > 0 and tail.size) else 0.0
    peak_force = float(np.max(force))

    # Decimate the traces over the auto-zoom window to a fixed length for the panel.
    hi = max(2, zoom_end + 1)
    idx = np.unique(np.linspace(0, hi - 1, min(hi, MALLET_DIAG_POINTS)).astype(int))
    return {
        "kind": "mallet",
        "t": _finite_list(time[idx], 6),
        "vel": _finite_list(mvel[idx], 5),
        "force": _finite_list(force[idx], 5),
        "v0": round(v0, 4),
        "restitution": round(float(restitution), 4),
        "separated": bool(separated),
        "contact_ms": (round(contact_ms, 2) if contact_ms is not None else None),
        "peak_head_pct": round(peak_head_pct, 3),
        "final_head_pct": round(final_head_pct, 4),
        "peak_force": round(peak_force, 3),
        "steps_per_contact": info["steps_per_contact"],
        "resolved": info["resolved"],
        "strike_fx": info["strike_fx"],
        "strike_fy": info["strike_fy"],
    }


def _build_payload_mallet(p: dict[str, Any]) -> dict[str, Any]:
    audio_dur = _fnum(p, "audio_duration", 1.0)
    anim_win = _fnum(p, "animation_window", 0.06)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    pickup_fx = _fnum(p, "pickup_x", 0.65)
    pickup_fy = _fnum(p, "pickup_y", 0.6)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= MALLET_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {MALLET_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    for name, v in (("pickup_x", pickup_fx), ("pickup_y", pickup_fy)):
        if not (0.0 < v < 1.0):
            raise ParamError(f"{name} must be in (0, 1), got {v}.")

    mal, mem, c, fs, sigma, domain, geom, info = _build_mallet(p)
    lam_h = float(info["hysteresis"])

    # Work budget: the same node-step product as the membrane (audio run + slow-mo animation run),
    # capped lower (MALLET_WORK_MAX) for the per-step contact root-find.
    n_audio = max(1, round(audio_dur * fs))
    n_anim_est = max(1, round(anim_win * fs))
    work = mem.n_live * (n_audio + n_anim_est)
    if work > MALLET_WORK_MAX:
        raise ParamError(
            f"this configuration needs ~{work / 1e6:.0f}M node-steps (over the "
            f"~{MALLET_WORK_MAX / 1e6:.0f}M mallet budget); reduce N, raise lambda, shorten the "
            "audio/animation, or enlarge the drum."
        )

    f_disc = _discrete_eigenfreqs(mem, c, N_MEMBRANE_MODES)
    f1 = float(f_disc[0]) if f_disc.size else c / (2.0 * _length_scale(domain, geom))

    # --- audio + energy + contact-diagnostics run: ONE manual, instrumented loop -----------------
    # simulate() would give energy + pickup but NOT the mallet's internal state (velocity, contact
    # force, in-contact flag) — and the contact story IS the headline — so we step by hand and read
    # both. The head starts at REST (no set_state); the mallet drops and strikes it.
    pickup_idx = mem.pickup_index_at(*_frac_to_xy(domain, pickup_fx, pickup_fy, geom))
    energy = np.empty(n_audio + 1)
    pickup = np.empty(n_audio + 1)
    mvel = np.empty(n_audio + 1)
    head_e = np.empty(n_audio + 1)
    force = np.empty(n_audio + 1)
    in_contact = np.zeros(n_audio + 1, dtype=bool)
    energy[0], pickup[0] = mal.energy(), mal.displacement_at(pickup_idx)
    mvel[0], head_e[0], force[0] = mal.mallet_velocity(), mem.energy(), mal.contact_force
    in_contact[0] = mal.in_contact
    for i in range(1, n_audio + 1):
        mal.step()
        energy[i], pickup[i] = mal.energy(), mal.displacement_at(pickup_idx)
        mvel[i], head_e[i], force[i] = mal.mallet_velocity(), mem.energy(), mal.contact_force
        in_contact[i] = mal.in_contact
    if not np.all(np.isfinite(pickup)):
        raise ParamError(
            "simulation produced non-finite output (instability) — adjust parameters."
        )

    time = np.arange(n_audio + 1, dtype=float) / fs
    audio_res = SimResult(time=time, energy=energy, output=pickup, fs=fs, snapshots=[])
    lossless = (sigma == 0.0 and lam_h == 0.0)
    ke0 = 0.5 * float(info["mass"]) * float(info["strike_velocity"]) ** 2
    contact = _mallet_contact_block(time, mvel, head_e, force, in_contact, ke0, fs, info)

    # --- animation run: fresh mallet from REST, short window, fundamental-resolving stride --------
    mal_anim = _build_mallet(p)[0]
    anim_stride = max(1, round((fs / f1) / fpp))
    n_anim = max(anim_stride, round(anim_win * fs))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    anim_res = simulate(mal_anim, num_steps=n_anim, snapshot_stride=anim_stride)
    frames_full = np.array([st for _, st in anim_res.snapshots], dtype=float)  # (nf, ny, nx)
    frame_steps = np.array([i for i, _ in anim_res.snapshots], dtype=float)

    # --- spatial decimation to the display grid (identical to the membrane path) ------------------
    ny_full, nx_full = mem.mask.shape
    stride_s = max(1, math.ceil(max(ny_full, nx_full) / DISPLAY_MAX))
    frames_dec = frames_full[:, ::stride_s, ::stride_s]
    mask_dec = mem.mask[::stride_s, ::stride_s]
    nf, ny_dec, nx_dec = frames_dec.shape
    field_amp = float(np.max(np.abs(frames_dec))) if frames_dec.size else 0.0

    audio48, peak = _resample_normalize(pickup, fs)
    ext_x, ext_y = (2.0 * geom["radius"], 2.0 * geom["radius"]) if domain == "circle" \
        else (geom["Lx"], geom["Ly"])

    return {
        "model": "mallet",
        "domain": domain,
        "fs_sim": round(fs, 3),
        "lambda": round(float(getattr(mem, "lam", float("nan"))), 6),
        "grid": {
            "dims": 2, "nx": int(nx_dec), "ny": int(ny_dec),
            "extent_x": round(ext_x, 6), "extent_y": round(ext_y, 6), "domain": domain,
        },
        "frames": {
            "b64": _b64f32(frames_dec.ravel()),
            "n_frames": int(nf), "nx": int(nx_dec), "ny": int(ny_dec),
            "width": int(nx_dec), "dims": 2,
        },
        "mask": {"b64": _b64u8(mask_dec.ravel()), "nx": int(nx_dec), "ny": int(ny_dec)},
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(audio_res, lossless, 2.0 * sigma, decay_oracle=False),
        "meta": {
            "c": round(c, 3),
            "f1": round(f1, 3),
            "num_steps": int(n_audio),
            "n_frames": int(nf),
            "spectrum": contact,
        },
    }


# == Kirchhoff plate (2D, model #5 / #5b) ==========================================================
#
# The plate reuses the membrane's 2D heatmap machinery (decimated field + mask, max-pooled spectrum
# with discrete/continuum marker lines) with two differences: the domain is always a *rectangle*
# (there is no disk plate), and the secondary "domain" select is repurposed as the plate
# **boundary** — ``"supported"`` (simply-supported #5, closed-form oracle) or ``"free"`` (FFFF #5b,
# the curved Chladni plate, no closed form → Leissa square anchor only). The time control is the
# plate Courant ``mu = kappa k / h²`` (implicit → no CFL; large mu is coarse-but-stable): ``fs =
# kappa / (mu h²)``, so cost explodes at *low* mu (high fs → step blow-up), the opposite of a max.


def _plate_discrete_eigenfreqs(res: Any, k_request: int) -> NDArray[np.float64]:
    """Lowest discrete plate eigenfrequencies (Hz) — the lines the stepper actually rings at.

    Duck-typed across :class:`Plate` / :class:`VKPlate` and both boundaries (attributes shared by
    both classes: ``.boundary``, ``.L`` / ``.K`` / ``.W``, ``.kappa``, ``.k``, ``.theta``):

    * ``"supported"`` — ``eigsh(-L)`` gives the **Laplacian** magnitude ``Λ``;
      :func:`modal.discrete_plate_eigenfrequency` squares it internally (``Q = κ²Λ²``), so we pass
      ``Λ`` — **not** ``B``'s eigenvalue ``Λ²`` (that would double-square).
    * ``"free"`` — the generalized eig ``K φ = μ W φ`` (``μ = ω²/κ²``); drop the **3** rigid
      zero-modes, pass the rest to :func:`modal.discrete_beam_eigenfrequency`. Duck-typed on the
      matrix ``.W`` (never ``.w`` / ``.wdiag`` — those names diverge between Plate and VKPlate). The
      shift is strictly negative (``K`` is only PSD; a >= 0 shift lands on the rigid nullspace).
    """
    if res.boundary == "supported":
        k = min(int(k_request), res.n_live - 1)
        if k < 1:
            return np.asarray([], dtype=float)
        lam = eigsh(-res.L, k=k, sigma=0.0, which="LM", return_eigenvectors=False)
        Lambda = np.sort(np.asarray(lam, dtype=float))
        f = modal.discrete_plate_eigenfrequency(Lambda, res.kappa, res.k, res.theta)
        return np.sort(np.asarray(f, dtype=float))
    k = min(int(k_request) + 3, res.n_live - 1)
    if k < 4:
        return np.asarray([], dtype=float)
    a2 = res.Lx * res.Ly  # ~ a² for a (near-)square; only seeds the shift magnitude
    mu1 = (13.0 / a2) ** 2  # lambda_1 ~ 13.47 -> a safe negative shift off the rigid nullspace
    vals = eigsh(
        res.K.tocsc(), k=k, M=res.W.tocsc(), sigma=-1e-3 * mu1, which="LM",
        return_eigenvectors=False,
    )
    vals = np.sort(np.asarray(vals, dtype=float))
    elastic = np.clip(vals[3:], 0.0, None)  # drop the 3 rigid-body modes {1, x, y}
    f = modal.discrete_beam_eigenfrequency(elastic, res.kappa, res.k, res.theta)
    return np.sort(np.asarray(f, dtype=float))


def _plate_continuum(res: Any, n_modes: int) -> NDArray[np.float64]:
    """Continuum plate frequencies — the *geometry-tier* reference (shown, not scored).

    ``"supported"`` — the Navier law ``f_{mn} = (π/2)κ[(m/Lx)²+(n/Ly)²]`` over an ``(m, n)`` grid
    (nearly coincident with the discrete lines — the SS plate is a *tight* tier, ~1 cent).
    ``"free"`` — the free plate has **no closed form**, so we fall back to the Leissa
    FFFF-**square** anchor and only when the plate is (near-)square; otherwise return empty (a
    reference that would be meaningless off-square is better omitted than mislabelled).
    """
    if n_modes < 1:
        return np.asarray([], dtype=float)
    if res.boundary == "supported":
        rng = range(1, n_modes + 1)
        modes = [(m, n) for m in rng for n in rng]
        f = modal.rectangular_plate_freqs(res.kappa, res.Lx, res.Ly, modes)
        return np.sort(np.asarray(f, dtype=float))[:n_modes]
    if abs(res.Lx - res.Ly) > 0.02 * max(res.Lx, res.Ly):
        return np.asarray([], dtype=float)  # Leissa anchor is square-only
    a2 = res.Lx * res.Ly  # ~ a² for a near-square plate
    f = modal.free_plate_ffff_square_lambdas() * res.kappa / (2.0 * np.pi * a2)
    return np.sort(np.asarray(f, dtype=float))


def _build_plate(p: dict[str, Any]) -> tuple[Plate, float, str, dict[str, Any]]:
    """Construct a fresh :class:`Plate`. Returns ``(res, fs, boundary, geom)``.

    The secondary "domain" select carries the plate **boundary** (``supported`` / ``free``). The
    time control is the plate Courant ``mu = kappa k / h²``: ``fs = kappa / (mu h²)`` with
    ``h = Lx/N`` reproduces the requested ``mu`` exactly. ``geom`` holds the *snapped* geometry read
    back off the ctor (``Ly`` is snapped to whole cells).
    """
    boundary = str(p.get("domain", "supported"))
    if boundary not in ("supported", "free"):
        raise ParamError(f"boundary must be 'supported' or 'free', got {boundary!r}.")
    kappa = _fnum(p, "kappa", 20.0)
    rho = _fnum(p, "rho", 0.005)  # areal density (kg/m²) — Plate.rho IS areal (cf. VKPlate)
    Lx = _fnum(p, "Lx", 1.0)
    Ly = _fnum(p, "Ly", 1.0)
    mu = _fnum(p, "mu", 1.0)
    sigma = _fnum(p, "sigma", 0.0)
    nu = _fnum(p, "nu", 0.3)
    theta = _fnum(p, "theta", 0.28)
    try:
        N = int(p.get("N", 60))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= PLATE_N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {PLATE_N_MAX}] for the plate, got {N}.")
    if min(kappa, rho, Lx, Ly) <= 0:
        raise ParamError("kappa, rho, Lx, Ly must all be positive.")
    if sigma < 0:
        raise ParamError(f"sigma (loss) must be >= 0, got {sigma}.")
    if not (0.0 < mu <= PLATE_MU_MAX):
        raise ParamError(f"mu (plate Courant) must be in (0, {PLATE_MU_MAX}], got {mu}.")

    h = Lx / N
    fs = kappa / (mu * h * h)
    res = Plate(
        Lx=Lx, Ly=Ly, kappa=kappa, rho=rho, fs=fs, N=N, sigma=sigma, boundary=boundary,
        nu=nu, theta=theta,
    )
    geom = {"Lx": float(res.Lx), "Ly": float(res.Ly)}
    if res.n_live > PLATE_NLIVE_MAX:
        raise ParamError(
            f"this geometry has {res.n_live} interior nodes (> {PLATE_NLIVE_MAX}); reduce N or use "
            "a less extreme aspect ratio."
        )
    return res, fs, boundary, geom


def _build_payload_plate(p: dict[str, Any]) -> dict[str, Any]:
    audio_dur = _fnum(p, "audio_duration", 1.0)
    anim_win = _fnum(p, "animation_window", 0.03)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    amplitude = _fnum(p, "amplitude", 1e-3)
    pluck_fx = _fnum(p, "pluck_x", 0.4)
    pluck_fy = _fnum(p, "pluck_y", 0.55)
    pluck_wfrac = _fnum(p, "pluck_width", 0.3)
    pickup_fx = _fnum(p, "pickup_x", 0.62)
    pickup_fy = _fnum(p, "pickup_y", 0.58)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= PLATE_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {PLATE_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    for name, v in (("pluck_x", pluck_fx), ("pluck_y", pluck_fy),
                    ("pickup_x", pickup_fx), ("pickup_y", pickup_fy)):
        if not (0.0 < v < 1.0):
            raise ParamError(f"{name} must be in (0, 1), got {v}.")
    if not (0.0 < pluck_wfrac <= 1.0):
        raise ParamError(f"pluck_width must be in (0, 1], got {pluck_wfrac}.")

    res, fs, boundary, geom = _build_plate(p)
    Lx, Ly = geom["Lx"], geom["Ly"]

    # Work budget (n_live × steps). fs = kappa/(mu h²) rides 1/mu and N², so cost explodes at LOW mu
    # / high N — the message points there (opposite of the membrane's lambda cap).
    n_audio = max(1, round(audio_dur * fs))
    n_anim_est = max(1, round(anim_win * fs))
    work = res.n_live * (n_audio + n_anim_est)
    if work > PLATE_WORK_MAX:
        raise ParamError(
            f"this configuration needs ~{work / 1e6:.0f}M node-steps (over the "
            f"~{PLATE_WORK_MAX / 1e6:.0f}M budget); RAISE mu, reduce N, or shorten the audio."
        )

    f_disc = _plate_discrete_eigenfreqs(res, N_PLATE_MODES)
    f_cont = _plate_continuum(res, int(f_disc.size))
    f1 = float(f_disc[0]) if f_disc.size else res.kappa / (2.0 * min(Lx, Ly) ** 2)

    # --- audio run: broad raised-cosine strike, single (x, y) pickup, no snapshots ---------------
    wc = pluck_wfrac * min(Lx, Ly)
    pcx, pcy = pluck_fx * Lx, pluck_fy * Ly
    res.set_state(raised_cosine_2d(res.X, res.Y, (pcx, pcy), wc, amplitude=amplitude))
    pickup_idx = res.pickup_index_at(pickup_fx * Lx, pickup_fy * Ly)
    audio_res = simulate(res, num_steps=n_audio, pickup_index=pickup_idx)
    pickup = np.asarray(audio_res.output, dtype=float)
    if not np.all(np.isfinite(pickup)):
        raise ParamError("simulation produced non-finite output (instability) — adjust parameters.")

    # --- animation run: fresh plate, short window, fundamental-resolving stride (catch #2) --------
    anim = _build_plate(p)[0]
    anim.set_state(raised_cosine_2d(anim.X, anim.Y, (pcx, pcy), wc, amplitude=amplitude))
    anim_stride = max(1, round((fs / f1) / fpp))
    n_anim = max(anim_stride, round(anim_win * fs))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    anim_res = simulate(anim, num_steps=n_anim, snapshot_stride=anim_stride)
    frames_full = np.array([st for _, st in anim_res.snapshots], dtype=float)  # (nf, ny, nx)
    frame_steps = np.array([i for i, _ in anim_res.snapshots], dtype=float)

    frames_dec, mask_dec = _decimate_field_mask(frames_full, res.mask)
    nf, ny_dec, nx_dec = frames_dec.shape
    field_amp = float(np.max(np.abs(frames_dec))) if frames_dec.size else 0.0

    audio48, peak = _resample_normalize(pickup, fs)

    return {
        "model": "plate",
        "boundary": boundary,
        "fs_sim": round(fs, 3),
        "mu": round(float(res.mu), 6),
        "grid": {
            "dims": 2, "nx": int(nx_dec), "ny": int(ny_dec),
            "extent_x": round(Lx, 6), "extent_y": round(Ly, 6), "domain": "rectangle",
        },
        "frames": {
            "b64": _b64f32(frames_dec.ravel()),
            "n_frames": int(nf), "nx": int(nx_dec), "ny": int(ny_dec),
            "width": int(nx_dec), "dims": 2,
        },
        "mask": {"b64": _b64u8(mask_dec.ravel()), "nx": int(nx_dec), "ny": int(ny_dec)},
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(audio_res, res.sigma == 0.0, 2.0 * res.sigma),
        "meta": {
            "kappa": round(res.kappa, 4),
            "f1": round(f1, 3),
            "num_steps": int(n_audio),
            "n_frames": int(nf),
            "spectrum": _modal_spectrum_block(pickup, fs, f_disc, f_cont, "plate"),
        },
    }


# == von Kármán nonlinear plate (2D, model #6) =====================================================
#
# The expensive corner: two coupled fields (transverse w + Airy stress F), a Picard fixed-point per
# step, and **no analytic modal oracle** — energy conservation *is* the correctness test, but it
# telescopes only at the Picard fixed point, so the verdict is convergence-gated (catch: a
# non-converged run has a drift number that is iteration noise, not physics). The material surface
# is ``(E, e, nu, rho)`` (rho volumetric; the linear plates took kappa directly), and ``e``
# (thickness) is the amplitude scale — the strike is ``w_over_e · e`` metres. The pitch **hardens**
# with amplitude, so the spectrum panel's marker lines are the *linear* (w→0) modes and the real
# peaks sit **above** them (the hardening shift — the opposite reading of a linear model, where
# peaks-on-lines = good). The hardened fundamental is read by zero-crossing spacing (robust),
# and **only for** ``boundary="supported"`` + a broad centered strike — a free-edge crash is a mode
# wash where zero-cross returns noise, so the free path reports a crash cascade, not a glide %.


def _as_bool(v: Any, default: bool) -> bool:
    """Coerce a JSON/query value to bool (accepts bool, ``"true"``/``"false"``, 1/0)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _build_vk(p: dict[str, Any]) -> tuple[VKPlate, float, str, bool, dict[str, Any]]:
    """Construct a fresh :class:`VKPlate`. Returns ``(res, fs, boundary, nonlinear, geom)``.

    Material surface ``(E, e, nu, rho)`` (``rho`` volumetric → ``rho_s = rho·e`` inside the ctor);
    ``kappa, D, Y_mem`` are derived. The time control is ``fs`` **directly** (oversample around the
    nonlinearity, HANDOFF §8) — not a Courant number, because ``kappa`` is itself derived. The
    "domain" select carries the **boundary** (``supported`` gong / ``free`` cymbal).
    """
    boundary = str(p.get("domain", "supported"))
    if boundary not in ("supported", "free"):
        raise ParamError(f"boundary must be 'supported' or 'free', got {boundary!r}.")
    E = _fnum(p, "E", 2.0e11)
    e = _fnum(p, "e", 1.0e-3)
    nu = _fnum(p, "nu", 0.3)
    rho = _fnum(p, "rho", 7800.0)  # VKPlate.rho is VOLUMETRIC (kg/m³); rho_s = rho·e derived
    Lx = _fnum(p, "Lx", 0.4)
    Ly = _fnum(p, "Ly", 0.4)
    fs = _fnum(p, "fs", 32_000.0)
    sigma = _fnum(p, "sigma", 0.0)
    nonlinear = _as_bool(p.get("nonlinear", True), True)
    try:
        N = int(p.get("N", 20))
    except (TypeError, ValueError) as exc:
        raise ParamError(f"N must be an integer, got {p.get('N')!r}.") from exc

    if not (N_MIN <= N <= VK_N_MAX):
        raise ParamError(f"N must be in [{N_MIN}, {VK_N_MAX}] for the von Kármán plate, got {N}.")
    if min(E, e, rho, Lx, Ly) <= 0:
        raise ParamError("E, e, rho, Lx, Ly must all be positive.")
    if sigma < 0:
        raise ParamError(f"sigma (loss) must be >= 0, got {sigma}.")
    if not (VK_FS_MIN <= fs <= VK_FS_MAX):
        raise ParamError(f"fs must be in [{VK_FS_MIN:.0f}, {VK_FS_MAX:.0f}] Hz, got {fs}.")

    res = VKPlate(
        Lx=Lx, Ly=Ly, E=E, e=e, nu=nu, rho=rho, fs=fs, N=N, sigma=sigma,
        boundary=boundary, nonlinear=nonlinear, couple_max_iter=VK_COUPLE_MAX_ITER,
    )
    geom = {"Lx": float(res.Lx), "Ly": float(res.Ly)}
    if res.n_live > VK_NLIVE_MAX:
        raise ParamError(
            f"this plate has {res.n_live} live nodes (> {VK_NLIVE_MAX}); reduce N or "
            "the aspect ratio."
        )
    return res, fs, boundary, nonlinear, geom


def _run_vk(
    vk: VKPlate, num_steps: int, pickup_index: int | None = None, snapshot_stride: int = 0
) -> tuple[SimResult, dict[str, Any]]:
    """Run a :class:`VKPlate` like :func:`engine.simulate` but also track Picard convergence.

    Mirrors the engine's capture (energy always, optional pickup + snapshots) and additionally
    accumulates the convergence stats the energy verdict is gated on: how many steps failed to reach
    ``couple_tol``, the worst residual, and the max Picard sweeps used. Returns
    ``(SimResult, convergence)``.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")
    n = num_steps + 1
    energy = np.empty(n)
    output = np.empty(n) if pickup_index is not None else None
    snapshots: list[tuple[int, NDArray[np.float64]]] = []
    n_not_conv = 0
    worst_res = 0.0
    max_iters = 0

    energy[0] = vk.energy()
    if output is not None:
        output[0] = vk.displacement_at(pickup_index)  # type: ignore[arg-type]
    if snapshot_stride:
        snapshots.append((0, vk.state))
    for i in range(1, n):
        vk.step()
        if not vk.converged:
            n_not_conv += 1
        worst_res = max(worst_res, vk.last_residual)
        max_iters = max(max_iters, vk.n_iters)
        energy[i] = vk.energy()
        if output is not None:
            output[i] = vk.displacement_at(pickup_index)  # type: ignore[arg-type]
        if snapshot_stride and (i % snapshot_stride == 0):
            snapshots.append((i, vk.state))

    res = SimResult(
        time=np.arange(n) * vk.k, energy=energy, output=output, fs=1.0 / vk.k, snapshots=snapshots
    )
    convergence = {
        "all_converged": n_not_conv == 0,
        "n_not_converged": int(n_not_conv),
        "worst_residual": float(worst_res),
        "max_iters": int(max_iters),
        "couple_tol": float(vk.couple_tol),
    }
    return res, convergence


def _vk_spectrum_block(
    pickup: NDArray[np.float64], fs: float, f_lin: NDArray[np.float64], f0: float
) -> dict[str, Any] | None:
    """von Kármán spectrum panel: FFT + **linear** (w→0) marker lines + the hardening shift.

    Unlike the linear models, the marker lines are the *linear* eigenfrequencies and the real peaks
    sit **above** them by the amplitude hardening — so ``kind="vk"`` tells the frontend to read the
    gap as a *hardening shift*, never a "cents error". ``f0`` is the honest hardened fundamental
    (from zero-crossings; ``nan`` on the free-edge crash where it is meaningless). ``shift_pct`` is
    percent rise of ``f0`` over the linear fundamental (``None`` when ``f0`` is unavailable).
    """
    if f_lin.size == 0:
        return None
    f1_lin = float(f_lin[0])
    fmax = float(np.max(f_lin)) * 1.6
    if math.isfinite(f0) and f0 > 0:
        fmax = max(fmax, f0 * 1.6)
    pooled = _pooled_spectrum(pickup, fs, fmax)
    if pooled is None:
        return None
    f_ds, m_ds = pooled
    shift_pct = (
        100.0 * (f0 / f1_lin - 1.0) if (math.isfinite(f0) and f0 > 0 and f1_lin > 0) else None
    )
    return {
        "kind": "vk",
        "freq": _finite_list(f_ds, 3),
        "mag": _finite_list(m_ds, 5),
        "fmax": round(fmax, 3),
        "modes_linear": _finite_list(f_lin, 4),
        "f1_linear": round(f1_lin, 4),
        "f0_detected": round(float(f0), 4) if math.isfinite(f0) else None,
        "shift_pct": round(shift_pct, 2) if shift_pct is not None else None,
    }


def _vk_strike(
    res: VKPlate, boundary: str, amplitude: float, pcx: float, pcy: float, wc: float
) -> NDArray[np.float64]:
    """Excitation field for a :class:`VKPlate`.

    ``"supported"`` — the **(1,1) eigenmode** ``sin(πx/Lx) sin(πy/Ly)`` (peak ``amplitude``). A pure
    mode is what makes the hardened fundamental read cleanly off zero-crossings: a broad multi-mode
    strike would make the zero-cross count *overcount* and report a bogus glide (this mirrors
    ``diagnose_vk_plate``'s sweep, which uses ``mode11`` for exactly this reason). Pluck position is
    therefore ignored for the supported gong. ``"free"`` — a broad centered raised-cosine *crash* at
    ``(pcx, pcy)`` (a mode wash; no clean fundamental — the frontend shows a cascade, not a glide).
    """
    if boundary == "supported":
        field = amplitude * np.sin(np.pi * res.X / res.Lx) * np.sin(np.pi * res.Y / res.Ly)
    else:
        field = raised_cosine_2d(res.X, res.Y, (pcx, pcy), wc, amplitude=amplitude)
    field[~res.mask] = 0.0
    return field


def _build_payload_vk(p: dict[str, Any]) -> dict[str, Any]:
    audio_dur = _fnum(p, "audio_duration", 0.5)
    anim_win = _fnum(p, "animation_window", 0.02)
    playback_speed = _fnum(p, "playback_speed", 0.02)
    w_over_e = _fnum(p, "w_over_e", 2.0)
    pluck_fx = _fnum(p, "pluck_x", 0.5)
    pluck_fy = _fnum(p, "pluck_y", 0.5)
    pluck_wfrac = _fnum(p, "pluck_width", 0.28)
    pickup_fx = _fnum(p, "pickup_x", 0.47)
    pickup_fy = _fnum(p, "pickup_y", 0.53)
    fpp = max(1, int(_fnum(p, "frames_per_period", FRAMES_PER_PERIOD)))

    if not (0.0 < audio_dur <= VK_AUDIO_MAX):
        raise ParamError(f"audio_duration must be in (0, {VK_AUDIO_MAX}] s, got {audio_dur}.")
    if not (0.0 < anim_win <= ANIM_WIN_MAX):
        raise ParamError(f"animation_window must be in (0, {ANIM_WIN_MAX}] s, got {anim_win}.")
    if not (0.0 < playback_speed <= SPEED_MAX):
        raise ParamError(f"playback_speed must be in (0, {SPEED_MAX}], got {playback_speed}.")
    if not (0.0 < w_over_e <= VK_WOVERE_MAX):
        raise ParamError(f"w_over_e must be in (0, {VK_WOVERE_MAX}], got {w_over_e}.")
    for name, v in (("pluck_x", pluck_fx), ("pluck_y", pluck_fy),
                    ("pickup_x", pickup_fx), ("pickup_y", pickup_fy)):
        if not (0.0 < v < 1.0):
            raise ParamError(f"{name} must be in (0, 1), got {v}.")
    if not (0.0 < pluck_wfrac <= 1.0):
        raise ParamError(f"pluck_width must be in (0, 1], got {pluck_wfrac}.")

    res, fs, boundary, nonlinear, geom = _build_vk(p)
    Lx, Ly = geom["Lx"], geom["Ly"]

    # Picard-aware work budget: n_live × steps × couple_max_iter (worst-case sweeps every step).
    n_audio = max(1, round(audio_dur * fs))
    n_anim_est = max(1, round(anim_win * fs))
    iters = res.couple_max_iter if nonlinear else 1
    work = res.n_live * (n_audio + n_anim_est) * iters
    if work > VK_WORK_MAX:
        raise ParamError(
            f"this configuration needs up to ~{work / 1e6:.0f}M coupled node-solves (over the "
            f"~{VK_WORK_MAX / 1e6:.0f}M budget); reduce N, lower fs, or shorten the audio."
        )

    # Linear (w→0) eigenmodes: the marker lines. The hardened peaks sit ABOVE these.
    f_lin = _plate_discrete_eigenfreqs(res, N_VK_MODES)
    f1_lin = float(f_lin[0]) if f_lin.size else res.kappa / (2.0 * min(Lx, Ly) ** 2)

    # --- audio run: broad centered strike of w_over_e·e metres, single (x, y) pickup -------------
    amplitude = w_over_e * res.e
    wc = pluck_wfrac * min(Lx, Ly)
    pcx, pcy = pluck_fx * Lx, pluck_fy * Ly
    res.set_state(_vk_strike(res, boundary, amplitude, pcx, pcy, wc))
    pickup_idx = res.pickup_index_at(pickup_fx * Lx, pickup_fy * Ly)
    audio_res, convergence = _run_vk(res, n_audio, pickup_index=pickup_idx)
    pickup = np.asarray(audio_res.output, dtype=float)
    if not np.all(np.isfinite(pickup)):
        raise ParamError("simulation produced non-finite output (instability) — adjust parameters.")

    # Hardened fundamental via zero-crossings — ONLY honest for supported + a broad centered strike
    # (a free-edge crash is a mode wash where zero-cross returns noise; report a cascade, not a %).
    f0 = _zero_cross_fundamental(pickup, fs) if boundary == "supported" else float("nan")

    # --- animation run: fresh plate, short window, fundamental-resolving stride ------------------
    anim = _build_vk(p)[0]
    anim.set_state(_vk_strike(anim, boundary, amplitude, pcx, pcy, wc))
    anim_stride = max(1, round((fs / max(f1_lin, 1.0)) / fpp))
    n_anim = max(anim_stride, round(anim_win * fs))
    if n_anim // anim_stride > MAX_FRAMES:
        anim_stride = max(1, math.ceil(n_anim / MAX_FRAMES))
    anim_res, _ = _run_vk(anim, n_anim, snapshot_stride=anim_stride)
    frames_full = np.array([st for _, st in anim_res.snapshots], dtype=float)  # (nf, ny, nx)
    frame_steps = np.array([i for i, _ in anim_res.snapshots], dtype=float)

    frames_dec, mask_dec = _decimate_field_mask(frames_full, res.mask)
    nf, ny_dec, nx_dec = frames_dec.shape
    field_amp = float(np.max(np.abs(frames_dec))) if frames_dec.size else 0.0

    audio48, peak = _resample_normalize(pickup, fs)

    return {
        "model": "vk",
        "boundary": boundary,
        "nonlinear": nonlinear,
        "fs_sim": round(fs, 3),
        "grid": {
            "dims": 2, "nx": int(nx_dec), "ny": int(ny_dec),
            "extent_x": round(Lx, 6), "extent_y": round(Ly, 6), "domain": "rectangle",
        },
        "frames": {
            "b64": _b64f32(frames_dec.ravel()),
            "n_frames": int(nf), "nx": int(nx_dec), "ny": int(ny_dec),
            "width": int(nx_dec), "dims": 2,
        },
        "mask": {"b64": _b64u8(mask_dec.ravel()), "nx": int(nx_dec), "ny": int(ny_dec)},
        "frame_times": _finite_list(frame_steps / fs, 6),
        "anim_dt": float(anim_stride / fs),
        "playback_speed": playback_speed,
        "field_amp": field_amp,
        "audio": {"b64": _b64f32(audio48), "fs": AUDIO_FS, "peak": peak, "n": int(audio48.size)},
        "energy": _energy_block(
            audio_res, res.sigma == 0.0, 2.0 * res.sigma,
            convergence=convergence if nonlinear else None,
        ),
        "meta": {
            "kappa": round(res.kappa, 4),
            "e": res.e,
            "f1": round(f1_lin, 3),
            "num_steps": int(n_audio),
            "n_frames": int(nf),
            "spectrum": _vk_spectrum_block(pickup, fs, f_lin, f0),
        },
    }
