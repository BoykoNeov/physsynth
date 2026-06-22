"""Pure serialization seam for the web viewer (wrapper, NOT core).

``simulate_to_payload(params)`` is the whole backend: a params dict -> build a string resonator
(``ideal`` / ``stiff`` / ``damped`` dispatch) -> pluck -> run the existing
:func:`physsynth.core.engine.simulate` -> a JSON-able payload. It is socket-free and graphics-free,
so it is unit-tested directly (``tests/test_web_backend.py``). The HTTP shell (``web/server.py``) is
a thin wrapper over this function.

Design decisions baked in (see ``docs/dev/web-viewer-plan.md``, advisor review 2):
  * **catch #1** — the pickup is resampled to a fixed :data:`AUDIO_FS` (48 kHz); ``fs_sim`` rides N
    and c (sliders) and can blow past the browser ``AudioBuffer`` cap. ``audio.fs`` means 48 kHz.
  * **catch #2** — the animation captures only ``animation_window`` seconds at a stride that
    resolves the *fundamental* (~:data:`FRAMES_PER_PERIOD` frames/period), decoupled from the audio
    length; the frontend plays it in slow motion. A wall-clock 60 fps stride would alias the wiggle.
  * **catch #4** — energy is reported as a *correctness drift* only when lossless; with loss it is
    reported as a passivity (monotone-decrease + decay-rate) check, not a scary "drift" number.
  * **catch #5** — per-model constructor params and partials oracle (see :func:`_build_resonator`).

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

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import SimResult, simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_ideal import IdealString
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


class ParamError(ValueError):
    """A bad request parameter (out of the allowed range) — surfaced as a clean error payload."""


# -- small helpers ---------------------------------------------------------------------------------


def _b64f32(arr: NDArray[np.float64]) -> str:
    """Pack a numeric array as base64 little-endian float32 (10x smaller than JSON numbers)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
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
    raise ParamError(f"unknown model {model!r} (expected 'ideal' | 'stiff' | 'damped').")


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


def _energy_block(res: SimResult, sigma_zero: bool, oracle_2sigma: float) -> dict[str, Any]:
    """Energy report, gated by loss (catch #4): drift-vs-tol when lossless, passivity when lossy."""
    t, E = res.time, res.energy
    idx = np.linspace(0, len(E) - 1, min(len(E), N_ENERGY_POINTS)).astype(int)
    block: dict[str, Any] = {
        "sigma_is_zero": sigma_zero,
        "time": _finite_list(t[idx], 6),
        "value": _finite_list(E[idx]),
    }
    if sigma_zero:
        drift = res.energy_drift
        block["lossless"] = {"drift": drift, "tol": LOSSLESS_TOL, "pass": drift < LOSSLESS_TOL}
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
        raise ParamError("simulation produced non-finite output (instability) — adjust parameters.")

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
