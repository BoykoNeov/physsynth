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
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import SimResult, simulate
from physsynth.core.exciter import raised_cosine_2d, triangular_pluck
from physsynth.core.membrane import Membrane
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

# --- membrane (2D, Phase B) clamps + display budget ---
# Cost is ~cubic in N (nlive ~ N², fs rides N so steps ~ N, plus an eigsh shift-invert factorization
# of L) — the string's N_MAX is catastrophic in 2D, so the membrane gets its own, much lower ceiling
# and a shorter audio cap, both verified to keep a worst-case local render to a few seconds.
# Cost must be bounded by the *actual problem size*, not N alone. Profiled: the cost is pure FDTD
# (eigsh is ~0.06 s, negligible), and it has TWO independent drivers the sliders can push:
#   • per-step cost ∝ n_live, with a razor-sharp ~3.2× CACHE CLIFF at n_live ≈ 10_000 (87 µs →
#     281 µs/step as the working set crosses L2). A thin rectangle reaches it fast: n_live ~
#     N²·(Ly/Lx), so Lx=0.3, Ly=2.0, N=100 is ~66k nodes. ⇒ keep n_live strictly BELOW the cliff.
#   • step count ∝ fs = c/(λ·h) ∝ 1/min_dimension: a small drum / fine grid inflates steps (the
#     audio is resampled to 48 kHz regardless, so a high sim rate buys no fidelity — pure cost).
# So: an n_live cap below the cache cliff (keeps per-step on the fast side, where time ∝ work), and a
# work-budget cap on n_live × total steps (covers both the audio and the slow-mo animation runs).
MEMBRANE_N_MAX = 100
MEMBRANE_NLIVE_MAX = 9_900     # below the n_live≈10_000 L2 cliff; admits a square/disk at N=100
MEMBRANE_WORK_MAX = 7.0e8      # node-steps budget: square@N=100, 2 s audio ≈ 6.7e8 ≈ ~6 s locally
MEMBRANE_AUDIO_MAX = 2.0
MEMBRANE_LAMBDA_MAX = 1.0 / math.sqrt(2.0)   # 2D CFL ceiling (5-point Laplacian: λ ≤ 1/√2)
DISPLAY_MAX = 64             # heatmap display grid is decimated to <= this per axis (data-size trap)
N_MEMBRANE_MODES = 12        # discrete eigenmodes marked on the spectrum panel
N_SPEC_POINTS = 420          # decimated magnitude-spectrum length for the plot


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
    """Dispatch on geometry: the 1D string family vs the 2D membrane (Phase B)."""
    if str(p.get("model", "ideal")) == "membrane":
        return _build_payload_membrane(p)
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


# == membrane (2D, Phase B) ========================================================================
#
# The 2D path is split off from the string path (above) so the string contract stays bit-for-bit
# unchanged. What differs: frames are 2D heatmap fields (decimated to a <= DISPLAY_MAX display grid —
# the data-size trap), the excitation/pickup are (x, y) fractions, and the modal panel is a *magnitude
# spectrum with mode-marker lines* rather than per-partial cents bars. The reason for the latter
# (advisor review 3): the continuum Bessel oracle is off by ~O(h) staircase (≈9 cents at N=128) *by
# design*, so scoring it as "error" reads as a bug; and an off-centre struck drum rings high/odd modes
# only weakly, so `measure_partials_near` would lock onto noise for unexcited targets and report
# confident-but-meaningless cents. Instead we show the FFT with vertical lines at the **discrete**
# eigenfreqs (peaks landing on lines = self-consistency) and fainter lines at the continuum oracle
# (the staircase offset, shown not scored). Two headline numbers: the robust fundamental
# detected-vs-discrete cents, and the discrete-vs-continuum "geometry tier" gap. Energy stays the hard
# pass/fail signature (conservation σ=0 / passivity σ>0), reused verbatim from the string path.


def _build_membrane(p: dict[str, Any]) -> tuple[Membrane, float, float, float, str, dict[str, Any]]:
    """Construct a fresh :class:`Membrane` from params.

    Returns ``(res, c, fs, sigma, domain, geom)``. ``fs = c/(λ·h)`` reproduces the requested λ
    exactly (``h`` from the geometry: ``2·radius/N`` for a disk, ``Lx/N`` for a rectangle). ``geom``
    holds the *snapped* geometry read back off the resonator (the ctor snaps ``Ly`` to whole cells).
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
    """Lowest discrete eigenfrequencies (Hz) via ``eigsh(-L)`` → ``discrete_membrane_eigenfrequency``.

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


def _membrane_spectrum_block(
    pickup: NDArray[np.float64],
    fs: float,
    f_disc: NDArray[np.float64],
    f_cont: NDArray[np.float64],
) -> dict[str, Any] | None:
    """Magnitude spectrum (max-pooled) + mode-marker lines + the two headline cents numbers."""
    if f_disc.size == 0:
        return None
    fmax = float(np.max(f_disc)) * 1.25
    freqs, mag, _ = spectrum.magnitude_spectrum(pickup, fs)
    keep = (freqs >= 0.0) & (freqs <= fmax)
    f, m = freqs[keep], mag[keep]
    if m.size == 0:
        return None

    # Max-pool to ~N_SPEC_POINTS so spectral *peaks* survive the decimation (mean would wash them out).
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

    # Robust headline: the (0,1) fundamental is always excited -> detected-vs-discrete cents.
    f1 = float(f_disc[0])
    detected = float(
        spectrum.measure_partials_near(pickup, fs, np.asarray([f1]), search_hz=0.3 * f1)[0]
    )
    cents_fund = float(modal.cents(detected, f1)) if math.isfinite(detected) else None
    cents_geom = (
        float(modal.cents(f1, float(f_cont[0]))) if f_cont.size and f_cont[0] > 0 else None
    )

    return {
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
    # run), with a cache penalty at high n_live. fs ∝ 1/min_dimension, so a small drum or fine grid
    # inflates the step count past N_MAX/n_live alone — bound the product before stepping anything.
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
        raise ParamError("simulation produced non-finite output (instability) — adjust parameters.")

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
