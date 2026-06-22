"""Web-viewer backend: the pure ``simulate_to_payload`` seam (wrapper, not core).

These exercise the serialization layer directly — no socket — so the contract the frontend depends
on is pinned by tests (HANDOFF §6 spirit, applied to the wrapper). The energy *signature* must
survive the wrapper unaltered (criterion 1); the rest pins frame/audio bookkeeping, the loss-gated
energy report (advisor catch #4), and clean error payloads for guard violations (no 500/NaN).

This file imports ``web.serialize`` (the wrapper) and ``physsynth`` (the core) — it does NOT touch
``physsynth.core`` internals, so the core dependency-allowlist guard is unaffected.
"""

import base64

import numpy as np
import pytest

from web.serialize import AUDIO_FS, LOSSLESS_TOL, simulate_to_payload


def _decode_f32(b64: str) -> np.ndarray:
    """Inverse of serialize._b64f32 — what the frontend's DataView does, in numpy."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def _base_params(**overrides):
    """Canonical short run (c=200, f1=100 Hz); short durations keep the suite fast."""
    p = {
        "model": "ideal",
        "L": 1.0, "T": 200.0, "rho": 0.005,
        "N": 64, "lambda": 1.0, "sigma": 0.0,
        "pluck_position": 0.3, "amplitude": 1e-3, "pickup_position": 0.1,
        "audio_duration": 0.3, "animation_window": 0.05, "playback_speed": 0.02,
    }
    p.update(overrides)
    return p


# -- criterion 1: the energy signature survives the wrapper ---------------------------------------


def test_lossless_drift_survives_wrapper():
    """Lossless ideal string: the serializer must report drift < 1e-10 (HANDOFF §6.1)."""
    payload = simulate_to_payload(_base_params())
    assert "error" not in payload, payload.get("error")
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is True
    assert "lossy" not in energy
    assert energy["lossless"]["drift"] < LOSSLESS_TOL
    assert energy["lossless"]["pass"] is True


@pytest.mark.parametrize("model", ["ideal", "stiff", "damped"])
def test_all_three_models_build(model):
    """Per-model dispatch (catch #5): each model returns a well-formed lossless payload."""
    extra = {"kappa": 1.0} if model in ("stiff", "damped") else {}
    payload = simulate_to_payload(_base_params(model=model, **extra))
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == model
    assert payload["meta"]["partials"] is not None
    # lossless (no loss params set) -> the drift report, not the passivity one
    assert payload["energy"]["lossless"]["drift"] < LOSSLESS_TOL


# -- frame / audio bookkeeping --------------------------------------------------------------------


def test_frame_bookkeeping():
    """Frame width == grid length; frame count matches the decoded buffer and ~window/stride."""
    p = _base_params()
    payload = simulate_to_payload(p)
    frames, grid = payload["frames"], payload["grid"]["x"]
    n, w = frames["n_frames"], frames["width"]
    assert w == len(grid)                      # 1D field width == node count (N+1)
    flat = _decode_f32(frames["b64"])
    assert flat.size == n * w                  # base64 buffer matches the declared shape
    assert n == len(payload["frame_times"])
    assert n >= 2                              # a real animation, not a single frame
    assert payload["anim_dt"] > 0


def test_frames_decode_to_field_values_and_boundary():
    """The decoded *values* (not just buffer length) must be sane.

    ``app.js`` decodes ``frames.b64`` as little-endian float32 and scales the canvas by
    ``field_amp``. A length-only check passes even on byte-order garbage (right size, wrong
    values) — the one silent failure a code-read can't rule out — so pin the values: the decoded
    peak must equal ``field_amp``, and the fixed endpoints must stay clamped at ~0 every frame.
    """
    payload = simulate_to_payload(_base_params())
    n, w = payload["frames"]["n_frames"], payload["frames"]["width"]
    grid = _decode_f32(payload["frames"]["b64"]).astype(float).reshape(n, w)
    # decoded peak == the field_amp the front-end y-scale uses (garbage bytes would diverge wildly)
    assert np.max(np.abs(grid)) == pytest.approx(payload["field_amp"], rel=1e-5, abs=1e-9)
    assert np.max(np.abs(grid[:, 0])) < 1e-9     # fixed boundary, left end clamped
    assert np.max(np.abs(grid[:, -1])) < 1e-9    # fixed boundary, right end clamped
    assert np.max(np.abs(grid)) > 0.0            # the pluck actually moved the interior


def test_audio_resampled_and_normalized():
    """Audio is at the fixed 48 kHz (catch #1), finite, and peak-normalized to <= 1."""
    payload = simulate_to_payload(_base_params(N=64))
    audio = payload["audio"]
    assert audio["fs"] == AUDIO_FS
    samples = _decode_f32(audio["b64"])
    assert samples.size == audio["n"]
    assert np.all(np.isfinite(samples))
    assert 0.0 < np.max(np.abs(samples)) <= 1.0 + 1e-6
    assert audio["peak"] > 0.0                 # raw physical displacement amplitude reported


def test_high_N_audio_stays_in_browser_range():
    """fs_sim rides N; the *audio* rate must stay 48 kHz regardless (the point of catch #1)."""
    payload = simulate_to_payload(_base_params(N=512, audio_duration=0.2))
    assert "error" not in payload, payload.get("error")
    assert payload["fs_sim"] > AUDIO_FS        # sim ran far above the browser cap...
    assert payload["audio"]["fs"] == AUDIO_FS  # ...but the delivered audio did not


# -- loss-gated energy (catch #4) -----------------------------------------------------------------


def test_lossy_reports_passivity_not_drift():
    """With loss, energy must be reported as monotone decrease, not a scary 'drift' figure."""
    payload = simulate_to_payload(_base_params(model="damped", sigma0=2.0, sigma1=1e-4, kappa=1.0))
    assert "error" not in payload, payload.get("error")
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is False
    assert "lossless" not in energy
    lossy = energy["lossy"]
    assert lossy["monotone"] is True
    assert lossy["measured_2sigma"] > 0.0      # energy genuinely decays
    assert lossy["oracle_2sigma"] == pytest.approx(4.0)  # 2 * sigma0


# -- clean error payloads (no 500 / NaN) ----------------------------------------------------------


def test_cfl_violation_is_clean_error():
    """lambda > 1 for the explicit ideal string -> the core ctor guard surfaces as an error."""
    payload = simulate_to_payload(_base_params(model="ideal", **{"lambda": 1.5}))
    assert "error" in payload
    assert payload["error"]["kind"] == "construction"
    msg = payload["error"]["message"].lower()
    assert "lambda" in msg or "cfl" in msg


def test_stiff_admits_lambda_above_one():
    """The implicit stiff scheme has no CFL limit -> lambda > 1 must succeed, not error."""
    payload = simulate_to_payload(_base_params(model="stiff", kappa=1.0, **{"lambda": 1.5}))
    assert "error" not in payload, payload.get("error")
    assert payload["lambda"] == pytest.approx(1.5, rel=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        {"N": 1},
        {"N": 99999},
        {"pluck_position": 0.0},
        {"pickup_position": 1.0},
        {"audio_duration": 0.0},
        {"animation_window": -1.0},
        {"model": "banana"},
        {"rho": 0.0},
    ],
)
def test_bad_params_give_error_payload(bad):
    """Out-of-range params return a clean error payload, never an exception or a 500."""
    payload = simulate_to_payload(_base_params(**bad))
    assert "error" in payload
    assert payload["error"]["message"]


def test_none_params_does_not_crash():
    """Defensive: empty/None input runs on defaults rather than throwing."""
    payload = simulate_to_payload({})
    assert "error" not in payload, payload.get("error")
