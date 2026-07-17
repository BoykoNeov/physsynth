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
from scipy.signal import resample_poly
from scipy.sparse.linalg import eigsh

from physsynth.analysis import damping, duffing, modal, spectrum
from physsynth.core.engine import SimResult, simulate
from physsynth.core.exciter import raised_cosine_2d, triangular_pluck
from physsynth.core.membrane import Membrane
from physsynth.core.plate import Plate, VKPlate
from physsynth.core.string_damped import DampedStiffString
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
    raise ParamError(
        f"unknown model {model!r} (expected 'ideal' | 'stiff' | 'damped' | 'tension')."
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


def _energy_block(
    res: SimResult,
    sigma_zero: bool,
    oracle_2sigma: float,
    convergence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Energy report, gated by loss (catch #4): drift-vs-tol when lossless, passivity when lossy.

    ``convergence`` (von Kármán only) folds in the Picard convergence gate: the energy identity
    telescopes *only at the fixed point*, so a run with any non-converged step has a drift number
    that is iteration noise, not physics. When supplied, the lossless verdict additionally requires
    every step to have converged, and the block carries a ``convergence`` sub-block so the frontend
    can say "did not converge — verdict N/A" instead of a lying pass/fail.
    """
    t, E = res.time, res.energy
    idx = np.linspace(0, len(E) - 1, min(len(E), N_ENERGY_POINTS)).astype(int)
    block: dict[str, Any] = {
        "sigma_is_zero": sigma_zero,
        "time": _finite_list(t[idx], 6),
        "value": _finite_list(E[idx]),
    }
    if convergence is not None:
        block["convergence"] = convergence

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
    block["lossy"] = {
        "monotone": bool(max_rel_inc <= MONOTONE_TOL),
        "max_rel_increase": max_rel_inc,
        "measured_2sigma": _fit_decay(t, E),
        "oracle_2sigma": oracle_2sigma,
    }
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
    if model == "plate":
        return _build_payload_plate(p)
    if model == "vk":
        return _build_payload_vk(p)
    if model == "tension":
        return _build_payload_tension(p)
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


# == membrane (2D, Phase B) ========================================================================
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


def _pooled_spectrum(
    pickup: NDArray[np.float64], fs: float, fmax: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Magnitude spectrum over ``[0, fmax]`` max-pooled to ~:data:`N_SPEC_POINTS` points.

    Max-pooling (not mean) so spectral *peaks* survive the decimation, then normalized to 0..1.
    Shared by every heatmap model's spectrum panel. Returns ``(freq, mag)`` or ``None`` if empty.
    """
    freqs, mag, _ = spectrum.magnitude_spectrum(pickup, fs)
    keep = (freqs >= 0.0) & (freqs <= fmax)
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
