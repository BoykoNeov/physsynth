"""Web-viewer backend: the pure ``simulate_to_payload`` seam (wrapper, not core).

These exercise the serialization layer directly — no socket — so the contract the frontend
depends on is pinned by tests (HANDOFF §6 spirit, applied to the wrapper). The energy *signature*
must survive the wrapper unaltered (criterion 1); the rest pins frame/audio bookkeeping, the
loss-gated energy report (advisor catch #4), and clean error payloads for guard violations
(no 500/NaN).

This file imports ``web.serialize`` (the wrapper) and ``physsynth`` (the core) — it does NOT touch
``physsynth.core`` internals, so the core dependency-allowlist guard is unaffected.
"""

import base64

import numpy as np
import pytest

import web.serialize as web_serialize
from web.serialize import (
    AUDIO_FS,
    BODY_AUDIO_MAX,
    BODY_DISTANCE_MAX,
    BODY_N_MAX,
    BODY_SIGMA_BODY_MAX,
    BODY_WORK_MAX,
    BORE_ANIM_MAX,
    BORE_AUDIO_MAX,
    BORE_N_MAX,
    BORE_N_MIN,
    BORE_ODD_EVEN_GATE,
    BOW_BALANCE_TOL,
    BOW_N_MAX,
    BOW_SLIP_MATCH_TOL,
    DISPLAY_MAX,
    FRET_AUDIO_MAX,
    FRET_BRIGHTNESS_PEAK,
    FRET_CONTROL_MAX,
    FRET_DUTY_MAX,
    FRET_EPISODES_MIN,
    FRET_N_MAX,
    FRET_RAIL_FRAC_MIN,
    FRET_RASTER_COLS_PER_PERIOD,
    FRET_RASTER_MAX_ROWS,
    FRET_WORK_MAX,
    GEOM_DT_MAX,
    GEOM_LAM_LONG_MAX,
    GEOM_N_MAX,
    GEOM_PHANTOM_DEFECT_MIN,
    GEOM_PHANTOM_WINDOW,
    GEOM_PHANTOM_WORK_MAX,
    GEOM_WORK_MAX,
    JAWARI_AMP_MAX,
    JAWARI_DEPTH_MAX,
    JAWARI_ELEVATION_GATE,
    JAWARI_N_MAX,
    JAWARI_RATIO_FLOOR,
    JAWARI_WORK_MAX,
    JUARI_AMP_MAX,
    JUARI_ELEVATION_GATE,
    JUARI_N_MAX,
    JUARI_SWEEP_DUR,
    JUARI_WORK_MAX,
    LOSSLESS_TOL,
    MALLET_N_MAX,
    MEMBRANE_LAMBDA_MAX,
    MEMBRANE_N_MAX,
    PLATE_N_MAX,
    PLATEBODY_AUDIO_MAX,
    PLATEBODY_DISTANCE_MAX,
    PLATEBODY_K_MAX,
    PLATEBODY_NPLATE_MAX,
    PLATEBODY_NSTRING_MAX,
    PLATEBODY_SIGMA_MAX,
    PLATEBODY_WORK_MAX,
    REED_ANIM_MAX,
    REED_AUDIO_MAX,
    REED_N_MAX,
    REED_N_MIN,
    REED_SPEAK_GATE,
    REED_SWEEP_N,
    REED_WORK_MAX,
    SYMP_N_MAX,
    SYMP_SIGMA_BODY_MAX,
    SYMP_WEINREICH_DETUNE_MAX,
    SYMP_WORK_MAX,
    TENSION_AMP_MAX,
    TENSION_DT_MAX,
    TENSION_N_MAX,
    TENSION_OFFMODE_MAX,
    VK_N_MAX,
    VK_WOVERE_MAX,
    _measure_tension_mode1,
    _tension_dt_over_t,
    _tension_spectrum_block,
    simulate_to_payload,
)


def _decode_f32(b64: str) -> np.ndarray:
    """Inverse of serialize._b64f32 — what the frontend's DataView does, in numpy."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def _decode_u8(b64: str) -> np.ndarray:
    """Inverse of serialize._b64u8 — the decimated domain mask the frontend reads."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8)


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


# == tension-modulated string (1D nonlinear, model #9) ============================================
# The panel's headline is the amplitude shift against the model's EXACT Duffing closed form, so
# these pin the oracle through the wrapper, both honesty gates, and the cost budget it needs.

TENSION_P: dict = {"model": "tension", "N": 128, "audio_duration": 0.4, "amplitude": 0.02}


def test_tension_shift_matches_the_exact_duffing_oracle():
    """**The money test.** Lead with the *shift*, never an absolute frequency: a measured ω(A)
    carries the θ-scheme's linear temporal dispersion error, and ω(A→0) carries the same one, so
    their difference cancels it and isolates the nonlinear physics (the core's own oracle test,
    re-run through the wrapper)."""
    sp = simulate_to_payload(TENSION_P)["meta"]["spectrum"]
    assert sp["kind"] == "tension"
    assert sp["shift_oracle"] > 1.0, "the shift should be a real, audible number of Hz"
    assert sp["shift_measured"] == pytest.approx(sp["shift_oracle"], rel=1e-2)
    # the residual is the scheme's O(h²)+O(k²) gap to the continuum Duffing, not crossing noise
    assert sp["shift_rel_error"] < 1e-2
    assert sp["f_hardened"] > sp["f_linear"], "hardening only — transverse motion can only stretch"


def test_tension_shift_is_immune_to_loss():
    """The shift must come from a **lossless** pair of runs, never from the audio pickup.

    ``duffing_frequency_shift(A)`` predicts the frequency at a *fixed* amplitude; a lossy tension
    string is a downward-gliding chirp, so zero-crossing the decaying pickup would report an
    amplitude-*averaged* frequency that undershoots the oracle — and the panel would diverge from
    the oracle as σ rose, reading as a bug that isn't one. The audio stays lossy on purpose (the
    glide is the model's signature); the *number* must not notice.
    """
    quiet = simulate_to_payload({**TENSION_P})["meta"]["spectrum"]
    lossy = simulate_to_payload({**TENSION_P, "sigma0": 5.0, "sigma1": 0.002})["meta"]["spectrum"]
    assert not simulate_to_payload({**TENSION_P, "sigma0": 5.0})["energy"]["sigma_is_zero"]
    # bit-identical: the measurement run forces sigma0 = sigma1 = 0 regardless of the request
    assert lossy["shift_measured"] == quiet["shift_measured"]
    assert lossy["f_hardened"] == quiet["f_hardened"]


def test_tension_energy_survives_the_wrapper_and_the_nonlinearity_is_engaged():
    """Criterion 1 through a nonlinear model — plus the model #6 lesson: a nonlinearity *hides* at
    small amplitude, where the test merely re-runs the linear scheme. So assert the stretch term
    actually holds a real fraction of E at the shipped default."""
    r = simulate_to_payload(TENSION_P)
    assert r["energy"]["lossless"]["drift"] < LOSSLESS_TOL
    assert r["energy"]["lossless"]["pass"] is True
    assert r["energy"]["convergence"]["all_converged"] is True
    assert r["meta"]["nonlinear_fraction"] > 0.01, "the nonlinearity must be engaged, not hiding"
    assert r["meta"]["EA_over_T"] == pytest.approx(500.0)


def test_tension_ea_zero_collapses_to_the_linear_string():
    """``EA = 0`` is model #3 bit-for-bit — the free regression the nonlinearity ships with."""
    r = simulate_to_payload({**TENSION_P, "EA": 0.0})
    sp = r["meta"]["spectrum"]
    assert r["meta"]["nonlinear_fraction"] == 0.0
    assert sp["shift_oracle"] == 0.0
    assert sp["shift_measured"] == pytest.approx(0.0, abs=1e-6)
    assert sp["dT_over_T"] == 0.0


def test_tension_dt_over_t_matches_the_closed_form():
    """``dT/T0 = EA·A²·p²/(4T)`` is exact for a single mode (the ``L`` cancels), which is what lets
    the guard run *before* any stepping."""
    sp = simulate_to_payload(TENSION_P)["meta"]["spectrum"]
    p2 = web_serialize.damping.spatial_eigenvalue_p2(128, 1.0 / 128, 1)
    # abs=1e-4: the payload rounds to 4 dp, and the run's *measured* peak is what is reported
    assert sp["dT_over_T"] == pytest.approx(_tension_dt_over_t(1e5, 0.02, p2, 200.0), abs=1e-4)


def test_purity_gate_nulls_the_shift_when_the_mode_breaks_up(monkeypatch):
    """Above its threshold the single mode parametrically disintegrates into its neighbours — real,
    energy-conserving physics, but the Duffing reduction stops describing the motion, so the shift
    must read ``null`` and never a lying number.

    Driven directly, past the ``dT/T0`` guard: that guard is exactly what makes this unreachable
    through the public path (which is the point — a guard that cannot trip is a *guarantee*), so the
    gate is exercised at the layer its logic actually lives.
    """
    monkeypatch.setattr(web_serialize, "TENSION_MEASURE_PERIODS", 20)
    p = {"model": "tension", "N": 64}
    broke = _measure_tension_mode1(p, 0.20)      # dT/T0 ~ 49 — far above threshold
    assert broke["off_mode"] > TENSION_OFFMODE_MAX, "expected genuine parametric breakup"
    assert _measure_tension_mode1(p, 0.005)["off_mode"] < TENSION_OFFMODE_MAX, "sub-threshold: pure"

    fs = 200.0 * 64 / 1.0
    w0sq, eps = web_serialize.duffing.kc_mode_coefficients(
        c=200.0, kappa=0.0, EA=1e5, rho=0.005,
        p2=web_serialize.damping.spatial_eigenvalue_p2(64, 1.0 / 64, 1), L=1.0,
    )
    sig = np.sin(2 * np.pi * 100.0 * np.arange(4096) / fs)
    sp = _tension_spectrum_block(p, sig, fs, w0sq, eps, 0.20)
    assert sp["purity"]["pure"] is False
    assert sp["shift_measured"] is None, "a broken-up mode must report no shift, not a wrong one"
    assert sp["shift_rel_error"] is None
    assert sp["shift_oracle"] > 0.0, "the oracle is still reported — only the measurement is void"


def test_tension_dt_guard_is_not_an_amplitude_proxy():
    """The guard bounds ``dT/T0``, not amplitude: EA and T move it just as hard as A does, so an
    amplitude-only cap would let ``EA = 2e5`` break up with the panel none the wiser."""
    at_cap = {**TENSION_P, "amplitude": TENSION_AMP_MAX, "EA": 1e5, "audio_duration": 0.2}
    assert simulate_to_payload(at_cap)["meta"]["spectrum"]["dT_over_T"] <= TENSION_DT_MAX
    # same (legal) amplitude, stiffer string -> over the threshold -> rejected
    err = simulate_to_payload({**at_cap, "EA": 2e5})["error"]
    assert err["kind"] == "param"
    assert "dT/T0" in err["message"]


def test_tension_lossy_reports_passivity():
    r = simulate_to_payload({**TENSION_P, "sigma0": 4.0})
    assert r["energy"]["sigma_is_zero"] is False
    assert r["energy"]["lossy"]["monotone"] is True
    assert r["energy"]["lossy"]["measured_2sigma"] == pytest.approx(8.0, rel=0.25)


def test_tension_frames_decode_to_a_mode1_sine_with_fixed_ends():
    """The IC is the mode-1 sine the Duffing reduction needs (a triangular pluck is broadly
    multi-mode and would make the shift a lying number). Pins the decoded *values*, not just the
    size — a length-only check cannot catch byte-order garbage."""
    r = simulate_to_payload(TENSION_P)
    fr = r["frames"]
    field = _decode_f32(fr["b64"]).reshape(fr["n_frames"], fr["width"])
    assert fr["dims"] == 1
    assert fr["width"] == 129
    assert field[0][0] == pytest.approx(0.0, abs=1e-12)
    assert field[0][-1] == pytest.approx(0.0, abs=1e-12)
    # frame 0 IS the initial condition: a half sine peaking at mid-span
    assert field[0][64] == pytest.approx(0.02, rel=1e-4)
    assert np.max(np.abs(field)) == pytest.approx(r["field_amp"], rel=1e-5)


@pytest.mark.parametrize(
    "bad",
    [
        {"amplitude": 0.0},
        {"amplitude": TENSION_AMP_MAX + 0.01},
        {"EA": -1.0},
        {"EA": 5e5},
        {"N": TENSION_N_MAX + 1},
        {"audio_duration": 6.0},
        {"N": 256, "audio_duration": 3.0},   # over the work budget (every step is a root-find)
        {"pickup_position": 1.5},
    ],
)
def test_tension_bad_params_give_error_payload(bad):
    r = simulate_to_payload({**TENSION_P, **bad})
    assert "error" in r, f"{bad} should be rejected"
    assert r["error"]["kind"] in ("param", "construction")
    assert isinstance(r["error"]["message"], str) and r["error"]["message"]


# == membrane (2D, Phase B) =======================================================================


def _membrane_params(**overrides):
    """Short 2D run — small grid + brief audio keep the suite fast (eigsh + FDTD per call)."""
    p = {
        "model": "membrane", "domain": "circle",
        "T": 200.0, "rho": 0.005, "radius": 0.5,
        "N": 40, "lambda": 0.6, "sigma": 0.0,
        "pluck_x": 0.4, "pluck_y": 0.55, "pluck_width": 0.45, "amplitude": 1e-3,
        "pickup_x": 0.65, "pickup_y": 0.6,
        "audio_duration": 0.2, "animation_window": 0.04, "playback_speed": 0.02,
    }
    p.update(overrides)
    return p


def test_membrane_lossless_drift_survives_wrapper():
    """The energy signature must survive the 2D wrapper too
    (HANDOFF §6.1 — conservation ⊥ geometry).
    """
    payload = simulate_to_payload(_membrane_params())
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == "membrane" and payload["frames"]["dims"] == 2
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is True and "lossy" not in energy
    assert energy["lossless"]["drift"] < LOSSLESS_TOL
    assert energy["lossless"]["pass"] is True


@pytest.mark.parametrize("domain", ["circle", "rectangle"])
def test_membrane_frame_bookkeeping_2d(domain):
    """2D frames: declared {nx, ny} match the decoded buffer, frame_times, and the <=64 budget."""
    extra = {"radius": 0.5} if domain == "circle" else {"Lx": 1.0, "Ly": 0.8}
    payload = simulate_to_payload(_membrane_params(domain=domain, **extra))
    assert "error" not in payload, payload.get("error")
    fr = payload["frames"]
    nf, nx, ny = fr["n_frames"], fr["nx"], fr["ny"]
    assert nx <= DISPLAY_MAX and ny <= DISPLAY_MAX        # display budget enforced
    flat = _decode_f32(fr["b64"])
    assert flat.size == nf * nx * ny                      # buffer matches the declared 3D shape
    assert nf == len(payload["frame_times"]) and nf >= 2
    assert payload["grid"]["dims"] == 2 and payload["grid"]["nx"] == nx


def test_membrane_spatial_decimation_shrinks_field():
    """For a grid above the display budget, the shipped field must be strictly decimated."""
    # N=80 -> mask is 81x81 (> 64), so the display grid must be coarser than the sim grid.
    payload = simulate_to_payload(_membrane_params(N=80, audio_duration=0.12))
    assert "error" not in payload, payload.get("error")
    nx, ny = payload["frames"]["nx"], payload["frames"]["ny"]
    assert nx < 81 and ny < 81                            # decimation actually happened
    assert nx <= DISPLAY_MAX and ny <= DISPLAY_MAX


def test_membrane_frames_decode_to_field_values_and_mask():
    """Decoded *values* are sane: peak == field_amp, and dead (exterior) nodes stay clamped at 0.

    The 2D analogue of the string boundary test — a length-only check passes on byte-order
    garbage. The decimated mask must align with the decimated field (same stride): every mask==0
    cell is 0 in every frame, and the decoded peak equals the ``field_amp`` the heatmap colour
    scale uses.
    """
    payload = simulate_to_payload(_membrane_params(N=64, audio_duration=0.12))
    fr = payload["frames"]
    nf, nx, ny = fr["n_frames"], fr["nx"], fr["ny"]
    field = _decode_f32(fr["b64"]).astype(float).reshape(nf, ny, nx)
    mask = _decode_u8(payload["mask"]["b64"]).astype(bool).reshape(ny, nx)
    assert payload["mask"]["nx"] == nx and payload["mask"]["ny"] == ny
    assert np.max(np.abs(field)) == pytest.approx(payload["field_amp"], rel=1e-5, abs=1e-12)
    assert np.max(np.abs(field)) > 0.0                    # the strike actually moved the interior
    # exterior (mask==0) is clamped to 0 in every frame (Dirichlet rim, aligned decimation)
    dead = ~mask
    if dead.any():
        assert np.max(np.abs(field[:, dead])) == 0.0
    # a disk fills ~pi/4 of its bounding box — a sanity bound on the live fraction
    assert 0.6 < mask.mean() < 0.85


def test_membrane_rectangle_aspect_uses_snapped_ly():
    """Rectangle extent comes from the ctor-snapped Ly (cells stay square), not the raw slider."""
    payload = simulate_to_payload(_membrane_params(domain="rectangle", Lx=1.2, Ly=0.8, N=48))
    assert "error" not in payload, payload.get("error")
    g = payload["grid"]
    assert g["extent_x"] == pytest.approx(1.2, abs=1e-9)
    assert g["extent_y"] == pytest.approx(0.8, abs=0.05)  # snapped to whole cells, ~0.8


def test_membrane_spectrum_block_fundamental_is_self_consistent():
    """The FFT must ring at the discrete fundamental (the honest oracle) to well under a cent."""
    payload = simulate_to_payload(_membrane_params(N=48, audio_duration=0.3))
    sp = payload["meta"]["spectrum"]
    assert sp is not None
    assert len(sp["modes_discrete"]) > 0 and len(sp["modes_continuum"]) > 0
    assert sp["f1_discrete"] > 0
    # detected-vs-discrete is the robust signature (the (0,1) mode is always excited)
    assert abs(sp["cents_fundamental"]) < 5.0
    # discrete-vs-continuum is the geometry tier (staircase); reported but NOT a pass/fail bar
    assert sp["cents_geometry"] is not None


def test_membrane_lossy_reports_passivity():
    """σ>0 → energy reported as monotone passive decay at 2σ, not a scary 'drift' (catch #4)."""
    payload = simulate_to_payload(_membrane_params(sigma=6.0, audio_duration=0.3))
    assert "error" not in payload, payload.get("error")
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is False and "lossless" not in energy
    lossy = energy["lossy"]
    assert lossy["monotone"] is True
    assert lossy["measured_2sigma"] > 0.0
    assert lossy["oracle_2sigma"] == pytest.approx(12.0)   # 2 * sigma


@pytest.mark.parametrize(
    "bad",
    [
        {"lambda": 1.0},                       # > 1/sqrt(2): violates the 2D CFL
        {"N": MEMBRANE_N_MAX + 1},             # past the membrane cost ceiling
        {"N": 1},
        {"domain": "hexagon"},
        {"radius": 0.0},
        {"pluck_x": 0.0},
        {"pickup_y": 1.0},
        {"audio_duration": 0.0},
    ],
)
def test_membrane_bad_params_give_error_payload(bad):
    """Out-of-range 2D params return a clean error payload, never an exception/500/NaN."""
    payload = simulate_to_payload(_membrane_params(**bad))
    assert "error" in payload
    assert payload["error"]["message"]


def test_membrane_cfl_ceiling_is_the_2d_bar():
    """λ just under 1/√2 builds; the message names the 2D CFL when it doesn't."""
    ok = simulate_to_payload(_membrane_params(**{"lambda": round(MEMBRANE_LAMBDA_MAX - 0.01, 3)}))
    assert "error" not in ok, ok.get("error")
    bad = simulate_to_payload(_membrane_params(**{"lambda": 0.9}))
    assert "error" in bad
    assert "cfl" in bad["error"]["message"].lower() or "sqrt(2)" in bad["error"]["message"].lower()


def test_membrane_thin_rectangle_rejected_by_nlive_guard():
    """A thin rectangle blows n_live ~ N²·(Ly/Lx) past the cache cliff — must reject, not hang.

    N alone is bounded but n_live is not: Lx=0.3, Ly=2.0, N=100 is ~66k interior nodes (the cost
    is per-step ∝ n_live with a sharp cache cliff). The geometry guard catches it *before* eigsh /
    simulate run, so the rejection is instant.
    """
    payload = simulate_to_payload(_membrane_params(domain="rectangle", Lx=0.3, Ly=2.0, N=100))
    assert "error" in payload
    assert "interior nodes" in payload["error"]["message"]


def test_membrane_small_geometry_rejected_by_work_budget():
    """A small drum inflates fs (∝ 1/min_dimension) → step count past the work budget.
    Reject.
    """
    # radius 0.2 at N=100 stays under the n_live cap but fs ≈ 83 kHz → ~1.3e9 node-steps >
    # budget.
    payload = simulate_to_payload(_membrane_params(domain="circle", radius=0.2, N=100,
                                                   audio_duration=2.0))
    assert "error" in payload
    assert "node-steps" in payload["error"]["message"]
    # the SAME geometry with short audio fits the budget (work scales with duration) — not a hard
    # ban
    ok = simulate_to_payload(_membrane_params(domain="circle", radius=0.2, N=100,
                                              audio_duration=0.3))
    assert "error" not in ok, ok.get("error")


# == Mallet → membrane collision (model #7) =======================================================


def _mallet_params(**overrides):
    """Short mallet run — small grid + brief audio keep the suite fast (FDTD + a root-find/step)."""
    p = {
        "model": "mallet", "domain": "circle",
        "T": 200.0, "rho": 0.005, "radius": 0.5,
        "N": 40, "lambda": 0.5, "sigma": 0.0,
        "mass": 0.02, "stiffness": 5.0e4, "alpha": 2.3, "hysteresis": 0.0,
        "strike_velocity": 3.0, "pluck_x": 0.5, "pluck_y": 0.5,
        "pickup_x": 0.65, "pickup_y": 0.6,
        "audio_duration": 0.3, "animation_window": 0.04, "playback_speed": 0.02,
    }
    p.update(overrides)
    return p


def test_mallet_lossless_conserves_through_the_wrapper():
    """The money test: a lossless, elastic strike conserves H to machine precision (drift < 1e-10).

    Model #7 is a *closed* mass+felt+membrane system, so the verdict is CONSERVATION — it rides the
    ordinary lossless drift panel, no new verdict type (the bow needed a balance panel only because
    it is driven from rest with E₀ = 0).
    """
    payload = simulate_to_payload(_mallet_params())
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == "mallet" and payload["frames"]["dims"] == 2
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is True and "lossy" not in energy
    assert energy["lossless"]["drift"] < LOSSLESS_TOL
    assert energy["lossless"]["pass"] is True


def test_mallet_bounces_with_near_unity_restitution_and_head_barely_rings():
    """A point mass is an inefficient membrane exciter: it bounces off with restitution ≈ 1 and the
    head keeps only ~0.01 % of the strike (the core signature, HANDOFF §7 finding).

    This is the headline the contact panel shows — physics, never tuned to ring louder.
    """
    sp = simulate_to_payload(_mallet_params(N=60))["meta"]["spectrum"]
    assert sp["kind"] == "mallet"
    assert sp["separated"] is True                        # the felt lets go within the window
    assert sp["restitution"] > 0.99                       # near-elastic rebound
    assert sp["final_head_pct"] < 0.5                     # head retains a tiny fraction post-strike
    assert sp["peak_head_pct"] > 30.0                     # but a big transient dimple mid-contact
    assert sp["contact_ms"] > 0.0 and sp["peak_force"] > 0.0


def test_mallet_strike_marker_reports_the_snapped_node_in_fractions():
    """The strike marker coords are the SNAPPED contact node (in (0,1) fractions) — where the felt
    actually landed, not the raw slider — so the heatmap dot sits on the real node.
    """
    sp = simulate_to_payload(_mallet_params(N=40, pluck_x=0.5, pluck_y=0.5))["meta"]["spectrum"]
    assert 0.0 < sp["strike_fx"] < 1.0 and 0.0 < sp["strike_fy"] < 1.0
    assert sp["strike_fx"] == pytest.approx(0.5, abs=0.03)   # snaps near the requested centre
    assert sp["strike_fy"] == pytest.approx(0.5, abs=0.03)


def test_mallet_audio_is_the_ring_not_the_dimple():
    """The pickup audio must be the membrane's modal ring (a real tone at the fundamental), not the
    near-field dimple relaxation — a peak near f1_discrete confirms it (advisor sanity check).
    """
    payload = simulate_to_payload(_mallet_params(N=48, audio_duration=0.4))
    assert "error" not in payload, payload.get("error")
    a = _decode_f32(payload["audio"]["b64"]).astype(float)
    assert np.all(np.isfinite(a)) and np.max(np.abs(a)) == pytest.approx(0.9, abs=1e-4)
    spec = np.abs(np.fft.rfft(a * np.hanning(a.size)))
    freqs = np.fft.rfftfreq(a.size, 1.0 / payload["audio"]["fs"])
    f_peak = freqs[np.argmax(spec)]
    f1 = payload["meta"]["f1"]
    assert abs(f_peak - f1) < 0.1 * f1                    # dominant peak rides the fundamental


def test_mallet_lossy_reports_passivity_without_a_decay_oracle():
    """σ>0 (or hysteresis>0): the verdict is pure passivity — the 2σ decay oracle is DROPPED.

    A closed struck system's total energy sits on a near-constant ½M·v₀² floor once the felt
    separates, so a fitted 2σ would read ~0 against a nonzero oracle — a lying 'broken match'. The
    honest lossy verdict is monotone non-increasing, and the 2σ fields must be absent so the
    frontend can render passivity instead.
    """
    for extra in ({"sigma": 6.0}, {"hysteresis": 3.0e4}):
        energy = simulate_to_payload(_mallet_params(N=48, **extra))["energy"]
        assert energy["sigma_is_zero"] is False and "lossless" not in energy
        lossy = energy["lossy"]
        assert lossy["monotone"] is True
        assert "measured_2sigma" not in lossy and "oracle_2sigma" not in lossy


def test_mallet_hysteresis_lowers_restitution():
    """Hunt-Crossley hysteresis is passive (removes energy on loading AND unloading), so it must
    lower the rebound: restitution(λ_h>0) < restitution(λ_h=0).
    """
    elastic = simulate_to_payload(_mallet_params(N=48))["meta"]["spectrum"]["restitution"]
    hyst = simulate_to_payload(_mallet_params(N=48, hysteresis=3.0e4))["meta"]["spectrum"]
    assert hyst["restitution"] < elastic


@pytest.mark.parametrize(
    "bad",
    [
        {"lambda": 1.0},                       # > 1/sqrt(2): violates the 2D CFL
        {"N": MALLET_N_MAX + 1},               # past the mallet's tighter N ceiling
        {"mass": 0.0},                         # non-physical felt / mass
        {"stiffness": -1.0},
        {"alpha": 0.5},                        # felt exponent must be >= 1
        {"strike_velocity": 0.0},
        {"pluck_x": 1.5},                      # strike position out of (0, 1)
        {"audio_duration": 5.0},               # past the audio cap
    ],
)
def test_mallet_bad_params_give_error_payload(bad):
    payload = simulate_to_payload(_mallet_params(**bad))
    assert "error" in payload and "message" in payload["error"]


def test_mallet_small_geometry_rejected_by_work_budget():
    """A squat rectangle at N=80 with 2 s audio inflates the node-step product past the mallet
    budget (halved vs the membrane for the per-step root-find). Reject; short audio fits.
    """
    heavy = simulate_to_payload(_mallet_params(domain="rectangle", Lx=0.6, Ly=0.8, N=80,
                                               audio_duration=2.0))
    assert "error" in heavy and "node-steps" in heavy["error"]["message"]
    ok = simulate_to_payload(_mallet_params(domain="rectangle", Lx=0.6, Ly=0.8, N=80,
                                            audio_duration=0.2))
    assert "error" not in ok, ok.get("error")


# == Kirchhoff plate (2D, model #5 / #5b) =========================================================


def _plate_params(**overrides):
    """Short plate run — the secondary 'domain' select carries the plate boundary."""
    p = {
        "model": "plate", "domain": "supported",
        "kappa": 20.0, "rho": 0.005, "Lx": 1.0, "Ly": 1.0,
        "N": 40, "mu": 1.0, "sigma": 0.0, "nu": 0.3,
        "pluck_x": 0.4, "pluck_y": 0.55, "pluck_width": 0.3, "amplitude": 1e-3,
        "pickup_x": 0.62, "pickup_y": 0.58,
        "audio_duration": 0.2, "animation_window": 0.02, "playback_speed": 0.02,
    }
    p.update(overrides)
    return p


@pytest.mark.parametrize("boundary", ["supported", "free"])
def test_plate_lossless_drift_survives_wrapper(boundary):
    """Both plate boundaries conserve energy through the wrapper (HANDOFF §6.1)."""
    payload = simulate_to_payload(_plate_params(domain=boundary))
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == "plate" and payload["boundary"] == boundary
    assert payload["frames"]["dims"] == 2
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is True and "lossy" not in energy
    assert energy["lossless"]["drift"] < LOSSLESS_TOL
    assert energy["lossless"]["pass"] is True


@pytest.mark.parametrize("boundary", ["supported", "free"])
def test_plate_frame_bookkeeping_2d(boundary):
    """2D frames + mask: declared shapes match the decoded buffers and the display budget."""
    payload = simulate_to_payload(_plate_params(domain=boundary))
    fr = payload["frames"]
    nf, nx, ny = fr["n_frames"], fr["nx"], fr["ny"]
    assert nx <= DISPLAY_MAX and ny <= DISPLAY_MAX
    assert _decode_f32(fr["b64"]).size == nf * nx * ny
    mask = _decode_u8(payload["mask"]["b64"])
    assert mask.size == nx * ny
    assert nf == len(payload["frame_times"]) and nf >= 2
    assert payload["grid"]["dims"] == 2 and payload["grid"]["domain"] == "rectangle"


def test_plate_supported_spectrum_is_tight_tier():
    """Simply-supported: the FFT rings on the discrete line, and the continuum (Navier) tier is
    tight (~1 cent) — unlike the membrane's O(h) staircase.
    """
    payload = simulate_to_payload(_plate_params(domain="supported", N=40, audio_duration=0.3))
    sp = payload["meta"]["spectrum"]
    assert sp is not None and sp["kind"] == "plate"
    assert len(sp["modes_discrete"]) > 0 and len(sp["modes_continuum"]) > 0
    assert abs(sp["cents_fundamental"]) < 5.0
    assert abs(sp["cents_geometry"]) < 15.0            # SS Navier tier is tight (not a staircase)


def test_plate_free_spectrum_has_leissa_only_when_square():
    """The free plate's continuum reference is the Leissa square anchor — present for a square,
    empty for an off-square rectangle (no closed form to mislabel).
    """
    sq = simulate_to_payload(_plate_params(domain="free", Lx=1.0, Ly=1.0))
    assert "error" not in sq, sq.get("error")
    assert len(sq["meta"]["spectrum"]["modes_continuum"]) > 0
    rect = simulate_to_payload(_plate_params(domain="free", Lx=1.4, Ly=0.7))
    assert "error" not in rect, rect.get("error")
    assert rect["meta"]["spectrum"]["modes_continuum"] == []


def test_plate_lossy_reports_passivity():
    """σ>0 → monotone passive decay at 2σ, not a drift number (catch #4)."""
    payload = simulate_to_payload(_plate_params(sigma=6.0, audio_duration=0.25))
    assert "error" not in payload, payload.get("error")
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is False and "lossless" not in energy
    assert energy["lossy"]["monotone"] is True
    assert energy["lossy"]["oracle_2sigma"] == pytest.approx(12.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"mu": 0.0},
        {"mu": -1.0},
        {"N": 1},
        {"N": PLATE_N_MAX + 1},
        {"domain": "clamped"},
        {"kappa": 0.0},
        {"Lx": 0.0},
        {"audio_duration": 0.0},
    ],
)
def test_plate_bad_params_give_error_payload(bad):
    """Out-of-range plate params return a clean error payload, never an exception/500."""
    payload = simulate_to_payload(_plate_params(**bad))
    assert "error" in payload
    assert payload["error"]["message"]


def test_plate_low_mu_rejected_by_work_budget():
    """fs = κ/(μh²) explodes at LOW μ → step blow-up; the guard rejects and points at μ."""
    payload = simulate_to_payload(_plate_params(mu=0.25, N=80, audio_duration=2.0))
    assert "error" in payload
    msg = payload["error"]["message"].lower()
    assert "node-steps" in msg and "mu" in msg


# == von Kármán nonlinear plate (2D, model #6) ====================================================


def _vk_params(**overrides):
    """Short von Kármán run — small N + brief audio (each step Picard-iterates two solves)."""
    p = {
        "model": "vk", "domain": "supported",
        "E": 2.0e11, "e": 1.0e-3, "nu": 0.3, "rho": 7800.0,
        "Lx": 0.15, "Ly": 0.15, "N": 14, "fs": 48000.0, "sigma": 0.0,
        "nonlinear": True, "w_over_e": 3.0,
        "pluck_x": 0.5, "pluck_y": 0.5, "pluck_width": 0.28,
        "pickup_x": 0.47, "pickup_y": 0.53,
        "audio_duration": 0.12, "animation_window": 0.01, "playback_speed": 0.02,
    }
    p.update(overrides)
    return p


def test_vk_supported_conserves_converges_and_hardens():
    """Supported gong, w/e=3: lossless energy conserves at a *converged* Picard fixed point, and
    the fundamental hardens ABOVE its linear value (the amplitude pitch glide).
    """
    payload = simulate_to_payload(_vk_params())
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == "vk" and payload["boundary"] == "supported"
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is True
    assert energy["convergence"]["all_converged"] is True
    assert energy["lossless"]["drift"] < LOSSLESS_TOL
    assert energy["lossless"]["pass"] is True
    sp = payload["meta"]["spectrum"]
    assert sp["kind"] == "vk"
    assert sp["f0_detected"] is not None and sp["shift_pct"] > 5.0   # genuine hardening


def test_vk_linear_toggle_has_no_convergence_block_and_no_shift():
    """nonlinear=False reproduces the *linear* plate: no Picard convergence block, ~0 hardening."""
    payload = simulate_to_payload(_vk_params(nonlinear=False, w_over_e=0.5))
    assert "error" not in payload, payload.get("error")
    assert payload["nonlinear"] is False
    assert "convergence" not in payload["energy"]
    assert payload["energy"]["lossless"]["pass"] is True
    assert abs(payload["meta"]["spectrum"]["shift_pct"]) < 1.0       # no coupling -> no glide


def test_vk_free_cymbal_conserves_without_a_fundamental():
    """Free-edge cymbal: energy conserves (converged), but the crash is a mode wash with no clean
    fundamental — f0/shift are reported as unavailable, never a lying number.
    """
    payload = simulate_to_payload(_vk_params(domain="free", Lx=0.2, Ly=0.2, N=14, w_over_e=3.0))
    assert "error" not in payload, payload.get("error")
    assert payload["boundary"] == "free"
    assert payload["energy"]["convergence"]["all_converged"] is True
    assert payload["energy"]["lossless"]["drift"] < LOSSLESS_TOL
    sp = payload["meta"]["spectrum"]
    assert sp["kind"] == "vk"
    assert sp["f0_detected"] is None and sp["shift_pct"] is None


def test_vk_lossy_reports_passivity():
    """σ>0 → monotone passive decay; the convergence block still rides along (catch: gate both)."""
    payload = simulate_to_payload(_vk_params(sigma=3.0, w_over_e=2.0, audio_duration=0.12))
    assert "error" not in payload, payload.get("error")
    energy = payload["energy"]
    assert energy["sigma_is_zero"] is False and "lossless" not in energy
    assert energy["lossy"]["monotone"] is True
    assert "convergence" in energy                       # verdict gate available in the lossy path


@pytest.mark.parametrize(
    "bad",
    [
        {"fs": 500.0},
        {"fs": 200000.0},
        {"N": 1},
        {"N": VK_N_MAX + 1},
        {"domain": "clamped"},
        {"E": 0.0},
        {"e": 0.0},
        {"w_over_e": 0.0},
        {"w_over_e": VK_WOVERE_MAX + 1.0},
        {"audio_duration": 0.0},
    ],
)
def test_vk_bad_params_give_error_payload(bad):
    """Out-of-range von Kármán params return a clean error payload, never an exception/500/NaN."""
    payload = simulate_to_payload(_vk_params(**bad))
    assert "error" in payload
    assert payload["error"]["message"]


# == bowed string (1D nonlinear exciter) ==========================================================
# The viewer's first ACTIVELY DRIVEN model, and the reason the energy panel grows a third verdict
# type. These pin (a) the balance money number, (b) that the balance *replaces* the two older
# verdicts rather than joining them — because for a driven model both are actively wrong, not
# merely weaker — (c) that the lossy branch is not a tautological residual, and (d) that the
# Helmholtz oracle is claimed only where it is honest.

BOW_P: dict = {"model": "bow", "N": 64, "lambda": 0.9, "kappa": 0.0, "audio_duration": 1.0,
               "sigma0": 0.5, "sigma1": 0.05, "force": 1.0, "v_bow": 0.1,
               "bow_position": 0.13, "sharpness": 60.0, "pickup_position": 0.33}
BOW_QUIET: dict = {**BOW_P, "sigma0": 0.0, "sigma1": 0.0, "audio_duration": 0.4}


def test_bow_lossless_balance_is_the_money_number():
    """**The money test.** The bow stores no energy (friction is memoryless), so with no loss every
    joule the bow's work put in must sit in the string: ``E - E0 == bow_work``, exactly. It holds
    for *any* Newton residual — the force is applied exactly and the power read from the true
    post-correction velocity — which is why (unlike von Kármán) there is no convergence gate."""
    e = simulate_to_payload(BOW_QUIET)["energy"]
    assert e["kind"] == "balance"
    assert e["sigma_is_zero"] is True
    assert e["balance"]["lossless"]["residual"] < BOW_BALANCE_TOL
    assert e["balance"]["lossless"]["pass"] is True
    assert e["balance"]["work_total"] > 0.0, "the bow did no net work — nothing was tested"


def test_bow_balance_replaces_both_older_verdicts_because_both_would_lie():
    """The *reason* the balance is a third verdict type, pinned rather than asserted in a comment.

    A driven model breaks both older checks, and not subtly: at sigma=0 the bow pumps energy in, so
    the conservation drift is astronomically past the 1e-10 bar; at sigma>0 the energy *rises* from
    rest to the Helmholtz limit cycle, so the passivity monotone check fails. Either would paint a
    red badge on a perfectly correct run, so the block must ship neither.
    """
    quiet = simulate_to_payload(BOW_QUIET)["energy"]
    lossy = simulate_to_payload(BOW_P)["energy"]
    for e in (quiet, lossy):
        assert e["kind"] == "balance"
        assert "lossless" not in e and "lossy" not in e, "the old verdicts must not ride along"
        assert "convergence" not in e, "the balance is exact for any Newton residual — no gate"
    # ... and prove they WOULD have lied: reconstruct what the lossless branch would have reported.
    # The bow starts from REST, so E0 = 0 and a *ratio* drift is not even defined — which is the
    # point twice over. SimResult.energy_drift falls back to max|E| there, and against the 1e-10
    # conservation bar that is a failure by orders of magnitude on a perfectly correct run.
    v = [x for x in quiet["value"] if x is not None]
    assert v[0] == 0.0, "the bow starts from rest — that is why a ratio drift is meaningless"
    apparent_drift = max(abs(x) for x in v)
    assert apparent_drift > 1e3 * LOSSLESS_TOL, (
        f"expected the conservation check to blow past its bar, got {apparent_drift:.2e}"
    )
    # and the lossy energy rises (non-monotone), which the passivity branch would have failed
    lv = [x for x in lossy["value"] if x is not None]
    assert lv[-1] > lv[0], "a bowed note must GAIN energy from rest — passivity is the wrong check"


def test_bow_lossy_reports_inferred_dissipation_not_a_tautological_residual():
    """With loss on, dissipation is never measured — it is *inferred* as ``bow_work - dE``. So a
    "balance residual" here would be identically zero BY CONSTRUCTION: a green tick that cannot
    fail. The honest content is the core's own criterion 2: the inferred loss is >= 0 and only ever
    grows."""
    b = simulate_to_payload(BOW_P)["energy"]["balance"]
    assert "lossless" not in b, "a lossy residual is a tautology — it must not be shipped"
    assert "residual" not in b["lossy"]
    assert b["lossy"]["non_negative"] is True and b["lossy"]["monotone"] is True
    assert b["lossy"]["pass"] is True
    assert b["lossy"]["dissipation_total"] > 0.0, "a lossy bowed string must dissipate"


def test_bow_balance_curves_share_one_decimation():
    """The three curves are only comparable if they are sampled at the same instants."""
    e = simulate_to_payload(BOW_P)["energy"]
    b = e["balance"]
    n = len(e["time"])
    assert len(b["work"]) == len(b["delta_energy"]) == len(b["dissipation"]) == n
    # dissipation == work - dE pointwise, by definition — the curve the panel draws
    for w, d, x in zip(b["work"], b["delta_energy"], b["dissipation"], strict=True):
        assert x == pytest.approx(w - d, abs=1e-12)


def test_bow_helmholtz_slip_fraction_matches_beta():
    """The panel's oracle: Helmholtz motion sticks for ``1-beta`` of the period and slips once, for
    a fraction ``beta``. The bow-position slider sits directly on the oracle's free parameter."""
    sp = simulate_to_payload(BOW_P)["meta"]["spectrum"]
    assert sp["kind"] == "bow"
    assert sp["helmholtz"] is True
    assert sp["slips_per_period"] == pytest.approx(1.0, abs=0.25)
    assert abs(sp["slip_fraction"] - sp["beta"]) < BOW_SLIP_MATCH_TOL
    assert sp["slip_matches_beta"] is True


@pytest.mark.parametrize("bow_position", [0.13, 0.2, 0.25])
def test_bow_slip_fraction_tracks_beta_as_the_bow_moves(bow_position):
    """Not one point but a *trend*: beta is the star control, so the slip fraction must follow it.
    ``force = 0.4`` is inside Schelleng's window across this range (the core's own choice)."""
    sp = simulate_to_payload(
        {**BOW_P, "bow_position": bow_position, "force": 0.4}
    )["meta"]["spectrum"]
    assert sp["helmholtz"] is True, f"expected clean Helmholtz at beta~{bow_position}"
    assert abs(sp["slip_fraction"] - sp["beta"]) < BOW_SLIP_MATCH_TOL


def test_bow_pitch_is_the_strings_not_the_bows():
    """The bow does not choose the pitch — the string does (the bore/reed lesson, on a string).
    Doubling the bow speed must not move it."""
    slow = simulate_to_payload(BOW_P)["meta"]["spectrum"]
    fast = simulate_to_payload({**BOW_P, "v_bow": 0.2, "force": 0.8})["meta"]["spectrum"]
    assert abs(slow["pitch_cents"]) < 60.0, "bowed pitch should lock to f1"
    if fast["helmholtz"]:
        assert abs(fast["f_detected"] - slow["f_detected"]) < 0.05 * slow["f1"]


def test_bow_out_of_window_is_labelled_not_failed():
    """**The honesty gate.** Schelleng's playable force window has a floor as well as a ceiling,
    and both narrow as the bow moves off the bridge. Starve the bow of force and it can no longer
    capture the string: it slides over it, no stick-slip cycle forms, and ``slip == beta`` stops
    describing anything. That is real physics faithfully reproduced, NOT a solver failure — so it
    must not be scored (tension's sigma-divergence and von Karman's broad-strike trap, a third
    time). The balance, a property of the *scheme* rather than of the parameters, must still pass.
    """
    r = simulate_to_payload({**BOW_P, "force": 0.02})
    sp = r["meta"]["spectrum"]
    assert r["meta"]["helmholtz_number"] < 1.0, "expected a starved bow, below the force floor"
    assert sp["helmholtz"] is False, "expected this corner to leave the Helmholtz window"
    assert sp["slip_matches_beta"] is None, "an off-window beta-match must not be scored"
    assert "note" in sp and "not scored" in sp["note"]
    assert "slip_error" not in sp and "pitch_cents" not in sp
    # a zero onset count is ambiguous, so the panel must say WHICH way it left the window
    assert sp["regime"] == "never_sticks" and sp["slip_fraction"] > 0.95
    # energy is STRUCTURAL, Helmholtz purity is DYNAMICAL: off-window motion is still exactly
    # balanced — the scheme does not care whether the note is musical
    assert r["energy"]["balance"]["lossy"]["pass"] is True


def test_bow_zero_force_leaves_the_string_at_rest():
    """``force = 0`` decouples the bow entirely — the free regression the exciter ships with. The
    bow starts from REST (there is no pluck), so with no friction nothing ever moves."""
    r = simulate_to_payload({**BOW_QUIET, "force": 0.0})
    assert r["energy"]["balance"]["work_total"] == 0.0
    assert r["field_amp"] == 0.0
    assert np.all(_decode_f32(r["frames"]["b64"]) == 0.0)


def test_bow_frames_decode_to_a_string_with_fixed_ends():
    """The 1D byte-order test: right size, wrong values would pass a length check. Also pins that
    the animation shows SETTLED motion — from rest the first frames would be near-flat."""
    r = simulate_to_payload(BOW_P)
    f = r["frames"]
    field = _decode_f32(f["b64"]).reshape(f["n_frames"], f["width"])
    assert f["width"] == len(r["grid"]["x"]) == BOW_P["N"] + 1
    assert np.all(field[:, 0] == 0.0) and np.all(field[:, -1] == 0.0), "fixed ends must not move"
    assert np.max(np.abs(field)) == pytest.approx(r["field_amp"], rel=1e-5)
    assert r["field_amp"] > 1e-6, "the animation window must show the settled corner, not rest"


def test_bow_helmholtz_number_is_reported_never_asserted():
    """Above 1 the friction curve is multivalued — the regime of real sustained bowing — but it is
    NOT a stability limit (friction is bounded, the scheme stable and the balance exact for any
    root). So it must be a reported diagnostic, and a value above 1 must render fine."""
    r = simulate_to_payload(BOW_P)
    assert r["meta"]["helmholtz_number"] > 1.0
    assert "error" not in r
    assert r["energy"]["balance"]["lossy"]["pass"] is True


@pytest.mark.parametrize(
    "bad",
    [
        {"N": BOW_N_MAX + 1},
        {"force": -1.0},
        {"force": 99.0},
        {"v_bow": 0.0},
        {"sharpness": 0.0},
        {"bow_position": 0.0},
        {"bow_position": 2.0},
        {"audio_duration": 0.0},
        {"audio_duration": 99.0},
    ],
)
def test_bow_bad_params_give_error_payload(bad):
    """Out-of-range bow params return a clean error payload, never an exception/500/NaN."""
    payload = simulate_to_payload({**BOW_P, **bad})
    assert "error" in payload
    assert payload["error"]["message"]


def test_bow_work_budget_is_its_own():
    """Every step is a friction root-find, so the string path's N_MAX/duration would hang. The
    guard must name the reason, not just refuse."""
    payload = simulate_to_payload({**BOW_P, "N": 256, "lambda": 0.5, "audio_duration": 3.0})
    assert "error" in payload
    assert "root-find" in payload["error"]["message"]


# == geometrically-exact string (1D nonlinear, 3 fields — model #10) ===============================
# The viewer's first VIZ-ONLY model. Its three regimes are three different claims, so they get
# three different panels. What these pin: (a) the two bit-exact zeros the model rests on (planar
# max|w|, and the unseeded whirl's honesty gate) really are zero and not merely small; (b) the
# rotating wave is an exact solution — round, with a longitudinal field that leans without moving;
# (c) the whirl grows only inside the Mathieu tongue, and the energy conserves straight THROUGH the
# growth (which is what separates redistribution from a diverging solve); (d) lam_long — the trap
# nothing in the core enforces — is a hard cap here.
#
# N = 8 and short windows on purpose: every step is a 3-field vector Newton solve (~2 ms), so this
# section pays for physical time, not for grid. The tongue is refinement-invariant in the
# dimensionless coordinate frac = delta/(eps A^2), so claims survive the coarse grid — kappa_w is
# recomputed from p2 at the actual N, not pinned to a value that was right at some other grid.
GEOM_P: dict = {"model": "geometric", "N": 8, "lam_long": 0.9, "T": 200.0, "rho": 0.005,
                "EA": 1.0e5, "L": 1.0, "theta": 0.28, "kappa": 0.0, "sigma0": 0.0, "sigma1": 0.0,
                "pickup_position": 0.25, "amplitude": 4e-3, "animation_window": 0.01}
GEOM_WHIRL: dict = {**GEOM_P, "domain": "whirl", "dt_over_t0": 1.5, "tongue_position": 0.25,
                    "animation_window": 0.03}


def _geom(**overrides):
    return simulate_to_payload({**GEOM_P, **overrides})


def test_geometric_planar_max_w_is_bit_exact_zero():
    """The orbit model #9 *can* draw — and the reason it is a claim, not a small number.

    A planar run leaves the out-of-plane field at **bit-exact** zero forever: that is ``w -> -w``
    reflection symmetry of the geometric nonlinearity, not a quantity that happens to be tiny. It is
    also the gate the whirl's growth ratios rest on: without it, "|w| grew 60x" would be partly a
    measurement of how much the in-plane motion leaks out of plane.
    """
    d = _geom(domain="planar")
    sp = d["meta"]["spectrum"]
    assert sp["kind"] == "planar"
    assert sp["max_w"] == 0.0, "not 'small' — exactly zero, or the symmetry is broken"
    assert sp["exact_zero"] is True


def test_geometric_planar_conserves_through_the_wrapper():
    """Criterion 1 for a three-field model: the energy signature survives serialization."""
    e = _geom(domain="planar")["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL
    assert e["lossless"]["pass"] is True


def test_geometric_rotating_wave_is_a_true_circle():
    """Tier B: the helix is an EXACT solution of the scheme, so it is round from the first frame.

    No growth, no transient, nothing to wait for — which is why it, and not the whirl, is the
    orbit this batch leads with. Its own roundness is the oracle; there is no external formula to
    compare against, because the converged BVP *is* the answer the scheme would give.
    """
    d = _geom(domain="rotating")
    sp = d["meta"]["spectrum"]
    assert sp["kind"] == "rotating"
    assert sp["roundness"] < 1e-6, "the radius wanders — this is not a rotating wave"
    assert sp["bvp_frequency"] > 0.0
    # Reported, not assumed: a BVP that stopped early is not an exact solution, so the roundness
    # would be measuring the continuation's failure rather than the physics. The panel says so.
    assert sp["bvp_converged"] is True
    assert sp["bvp_iterations"] > 0, "0 iterations would be a lying number, not a fast solve"


def test_geometric_rotating_wave_longitudinal_field_leans_but_does_not_move():
    """``psi`` is a NONZERO static stretch — so the claim is about motion, not about ``v`` itself.

    Asserting ``v == 0`` would assert the physics away (the helix holds a real stretch; the plan's
    own criterion was wrong here). What vanishes is the longitudinal *kinetic* energy: the field
    leans into the stretch it should already be holding, and then stays there.
    """
    d = _geom(domain="rotating")
    assert d["meta"]["spectrum"]["long_kin_over_e"] < 1e-12


def test_geometric_frames_decode_to_three_fields_with_clamped_ends():
    """The 2D path's byte-order test, in the shape this model needs.

    A length-only check cannot catch endianness garbage (right size, wrong values), so decode and
    look: three stacked fields per frame, ends clamped in ALL THREE components, and a ``u`` that
    actually carries the mode-1 amplitude that was asked for.
    """
    d = _geom(domain="planar")
    fr = d["frames"]
    assert fr["dims"] == 1 and fr["fields"] == ["u", "w", "v"]
    assert fr["width"] == len(d["grid"]["x"])
    buf = np.frombuffer(base64.b64decode(fr["b64"]), dtype="<f4")
    assert buf.size == fr["n_frames"] * 3 * fr["width"]
    f = buf.reshape(fr["n_frames"], 3, fr["width"])
    assert np.all(f[:, :, 0] == 0.0) and np.all(f[:, :, -1] == 0.0), "ends clamped in u, w AND v"
    assert np.max(np.abs(f[:, 0, :])) == pytest.approx(GEOM_P["amplitude"], rel=0.02)
    assert np.all(f[:, 1, :] == 0.0), "planar: w is zero in every frame"


def test_geometric_orbit_trail_is_the_probe_node_and_indexable_by_frame():
    """The trail the animation draws: it must decode, and one frame must map onto part of it."""
    d = _geom(domain="rotating")
    orb = d["orbit"]
    u = np.frombuffer(base64.b64decode(orb["u"]), dtype="<f4")
    w = np.frombuffer(base64.b64decode(orb["w"]), dtype="<f4")
    assert u.size == w.size == orb["n"]
    assert orb["per_frame"] > 0
    # It is a circle, so the two polarizations carry the SAME amplitude — the helix's signature.
    assert np.max(np.abs(w)) == pytest.approx(np.max(np.abs(u)), rel=0.05)


def test_geometric_whirl_grows_inside_the_tongue_and_conserves_through_it():
    """**The money test.** A parametric instability is energy REDISTRIBUTION, not energy creation.

    So a correct lossless scheme conserves straight *through* a blow-up that grows ``max|w|`` by
    orders of magnitude — and that is precisely what separates a whirl from the other thing that
    makes a field grow like this, a diverging solve. Energy alone is not sufficient (model #9's
    in-plane exchange conserves too), which is why the tongue tests below exist; but it is
    necessary, and it is nearly free (``energy()`` is ~0.15 ms against a ~2 ms step).
    """
    d = simulate_to_payload(GEOM_WHIRL)
    sp, e = d["meta"]["spectrum"], d["energy"]
    assert sp["kind"] == "whirl" and sp["in_tongue"] is True
    assert sp["growth"] > 3.0, "no growth at the tongue's peak — the recipe is off the tongue"
    assert e["lossless"]["drift"] < LOSSLESS_TOL
    assert e["lossless"]["pass"] is True


def test_geometric_whirl_needs_no_new_energy_verdict_unlike_the_bow():
    """The whirl rides the ORDINARY lossless drift check — no third verdict type.

    Worth pinning, because the bowed string needed one: a driven model breaks both older ones, so
    the balance had to replace them. Nothing drives this string — it is seeded and then left alone —
    so the plain conservation check is not merely adequate, it is the whole claim.
    """
    e = simulate_to_payload(GEOM_WHIRL)["energy"]
    assert "kind" not in e, "the whirl must not claim a balance/other verdict type"
    assert "balance" not in e
    assert "lossless" in e


def test_geometric_whirl_is_dead_outside_the_tongue():
    """Off the tongue there is no exponential growth *at any amplitude*: parameters are not free.

    This is the claim that makes the tongue a tongue rather than "more amplitude = more whirl". The
    upper edge is SOFT (the analysis is leading-order in eps), so the bar is generous: what must not
    happen is the orders-of-magnitude growth the tongue's interior shows.
    """
    d = simulate_to_payload({**GEOM_WHIRL, "tongue_position": 0.8})
    sp = d["meta"]["spectrum"]
    assert sp["in_tongue"] is False
    assert sp["growth"] < 3.0
    assert sp["predicted_rate"] == 0.0, "outside the tongue the closed-form rate is exactly zero"


def test_geometric_degenerate_string_cannot_whirl():
    """An isotropic string provably cannot whirl — rotational symmetry forces ``w_w == w_u``.

    With the default (displacement) seed this reads 1.00x, for a sharp reason: ``dw = dA phi`` at
    rest **is** the rotation generator, so the run is the same planar motion in a rotated plane.
    """
    d = simulate_to_payload({**GEOM_WHIRL, "tongue_position": 0.0})
    sp = d["meta"]["spectrum"]
    assert sp["degenerate"] is True and sp["in_tongue"] is False
    assert sp["growth"] == pytest.approx(1.0, abs=0.15)
    assert sp["kappa_w"] == 0.0


def test_geometric_velocity_seed_makes_the_degenerate_string_marginal_not_stable():
    """The received rule ("a velocity seed, NEVER a displacement one") is only half right.

    Measured, here and across the tongue: a *displaced* seed grows perfectly well inside the tongue
    (the pinning at 1.00x happens only at ``delta = 0``, where it is the rotation generator). What
    the velocity seed is genuinely for is the degenerate string: it injects angular momentum, so the
    marginal mode grows **secularly** — linear in t, not exponential. That is not whirling, and the
    panel must not call it whirling; the difference is the envelope's shape — hence the log-y.
    """
    disp = simulate_to_payload({**GEOM_WHIRL, "tongue_position": 0.0})["meta"]["spectrum"]
    vel = simulate_to_payload({**GEOM_WHIRL, "tongue_position": 0.0,
                               "seed_velocity": True})["meta"]["spectrum"]
    assert disp["seed_velocity"] is False and vel["seed_velocity"] is True
    assert vel["growth"] > disp["growth"], "a velocity kick must do what a rotation cannot"
    # Secular, not exponential: it grows, but nothing like the tongue's interior at the same cost.
    tongue = simulate_to_payload(GEOM_WHIRL)["meta"]["spectrum"]
    assert vel["growth"] < tongue["growth"]


def test_geometric_unseeded_whirl_is_the_honesty_gate():
    """Bit-exact zero at the tongue's centre with no seed — without this, growth measures a leak.

    In-plane motion cannot excite the out-of-plane field by itself, tongue or no tongue. Every
    growth ratio in this section is a ratio *to* the seed, so if the unseeded run were merely small
    rather than exactly zero, each of them would be partly measuring numerical leakage.
    """
    d = simulate_to_payload({**GEOM_WHIRL, "seed_frac": 0.0})
    sp = d["meta"]["spectrum"]
    assert sp["seeded"] is False
    assert max(sp["envelope"]) == 0.0
    assert d["energy"]["lossless"]["pass"] is True, "still conserves with nothing to redistribute"


def test_geometric_whirl_rate_matches_the_mathieu_prediction_and_runs_low():
    """Tier C: the rate matches the closed form to ~10%, and is *systematically* below it.

    ``(Om/2) sqrt(qM^2 - sigma^2)`` is leading-order in eps, and the seed contains a non-growing
    component, so the measurement is expected to under-run the prediction. The bar is deliberately
    loose: this is reported to the user and never scored, and inventing a tight pass/fail here would
    be inventing a claim the physics does not make.
    """
    sp = simulate_to_payload(GEOM_WHIRL)["meta"]["spectrum"]
    assert sp["predicted_rate"] > 0.0
    assert sp["measured_rate"] is not None
    assert sp["rate_ratio"] == pytest.approx(1.0, abs=0.35)


def test_geometric_has_no_audio_and_says_why():
    """The viewer's first viz-only model — the payload must say so, not ship a stub.

    A stub clip would be a click, not a note, and a silent player reads as a bug. The reason is
    physical and irreducible: c_long/c ~ 22, so resolving the longitudinal wave (which is what
    lam_long <= 1 means) forces fs ~ 22x a transverse-only string's.
    """
    d = _geom(domain="planar")
    assert d["audio"] is None
    assert "22" in d["audio_note"] and "minutes" in d["audio_note"]


def test_geometric_lam_long_above_one_is_rejected():
    """The trap nothing in the core enforces — the theta-scheme is unconditionally stable, so an
    under-resolved longitudinal wave fails SILENTLY (no CFL error, just quiet nonsense that stops
    conserving). The core only warns; the viewer must never render a headline in that regime."""
    d = _geom(domain="planar", lam_long=GEOM_LAM_LONG_MAX + 0.5)
    assert "error" in d
    assert "lam_long" in d["error"]["message"]


def test_geometric_lam_long_is_the_knob_and_lambda_is_derived():
    """The inversion of models #1-#9, and the reason the trap bites: the familiar lambda reads a
    reassuring ~0.04 in exactly the regime that works, so it cannot be the control."""
    d = _geom(domain="planar")
    assert d["lam_long"] == pytest.approx(GEOM_P["lam_long"], rel=1e-6)
    assert d["lambda"] < 0.1
    # c_long/c = sqrt(EA/T) ~ 22: the whole cost story of this model, in one ratio.
    assert d["meta"]["c_long"] / d["meta"]["c"] == pytest.approx(22.4, rel=0.05)


def test_geometric_work_budget_is_its_own():
    """A 3-field Newton solve per step at ~22x the sample rate: the string budget would hang."""
    d = _geom(domain="planar", animation_window=0.3)
    assert "error" in d
    assert "work budget" in d["error"]["message"]
    assert f"{GEOM_WORK_MAX:,}" in d["error"]["message"]


@pytest.mark.parametrize(
    "bad",
    [
        {"N": GEOM_N_MAX + 1},                     # fs rides N, every step is a Newton solve
        {"lam_long": 0.0},                           # the trap's guard, lower edge
        {"domain": "nonsense"},                      # unknown regime
        {"EA": 0.0},                                 # a string with no axial stiffness
        {"domain": "whirl", "dt_over_t0": GEOM_DT_MAX + 1.0},   # the driven mode would break up
        {"domain": "whirl", "tongue_position": -0.1},           # off the coordinate
        {"pickup_position": 1.5},                    # the orbit probe must be ON the string
    ],
)
def test_geometric_bad_params_give_error_payload(bad):
    """Guards surface as clean error payloads — never a 500, never a NaN render."""
    d = _geom(**bad)
    assert "error" in d, f"{bad} should have been rejected"
    assert d["error"]["kind"] in ("param", "construction")


# == phantom partials (model #10's 4th regime — the bridge-force spectrum) =========================
# Model #9's FIRST refusal, discharged. #9's tension is a spatial *scalar*, so it has no
# longitudinal field and nowhere to put a combination tone; #10's tension is a FIELD, and its
# nonlinear excess carries `a r^2 v_x / 2` — quadratic in the transverse fields, linear in the
# longitudinal one — so two transverse partials at f1, f2 drive v at f_i +- f_j. Those are Conklin's
# phantom partials, read off the bridge force EA v_x(0): what actually radiates in a piano.
#
# This regime reproduces tests/test_geometric_phantom.py's rig exactly (same lam_long / 0.10 s
# window / two-mode IC / v = 0 start / blind band-limited detector), so it inherits that suite's
# validation. What is pinned HERE is what the *wrapper* adds and could break: the verdict gating
# (three ways to not have a claim, each with its own honest label), the fixed window, the
# display/measurement split, and the two budgets.
#
# COST: N = 16 rather than the viewer's default 32, at ~23 s for the shared fixture. The claim is
# NOT refinement-invariant (see the coarse-grid test below — dispersion eats the defect), so N = 16
# is the cheapest grid where it is still real: defect 3.79 Hz against the 3.0 gate.
GEOM_PHANTOM: dict = {"model": "geometric", "domain": "phantom", "N": 16, "lam_long": 0.9,
                      "T": 200.0, "rho": 0.005, "EA": 1.0e5, "L": 1.0, "theta": 0.28,
                      "kappa": 8.0, "sigma0": 0.0, "sigma1": 0.0, "pickup_position": 0.25,
                      "amplitude": 1.5e-3,
                      # Deliberately absurd, and ignored: the window is fixed physics. Pinned by
                      # test_phantom_window_is_fixed_physics_and_ignores_the_animation_slider.
                      "animation_window": 0.3}


@pytest.fixture(scope="module")
def phantom():
    """The phantom run (~23 s), shared by every test that reads it."""
    return simulate_to_payload(dict(GEOM_PHANTOM))


def test_phantom_peaks_are_the_quadratic_combinations_of_the_measured_partials(phantom):
    """**The mechanism.** Every peak the longitudinal field carries is a *quadratic* combination of
    the transverse partials.

    This is ``a r^2 v_x/2`` made visible: ``r^2`` is quadratic in ``u``, so a transverse spectrum
    ``{f1, f2}`` drives ``v`` at ``{f2-f1, 2f1, f1+f2, 2f2}`` and at nothing else. The four
    strongest in-band peaks are those four combinations to ~0.04 Hz, and they dominate the strongest
    non-combination peak by ~5x (that runner-up is the broadband ``v = 0`` startup transient, not
    physics).

    The combinations are built from the MEASURED ``f1``, ``f2`` — never predicted. Those partials
    carry the theta-scheme's dispersion *and* the amplitude hardening, and the phantom rides on
    whatever the partials actually are; predicting them would fold both errors into the oracle and
    measure the formula instead of the string.
    """
    sp = phantom["meta"]["spectrum"]
    assert sp["kind"] == "phantom"
    assert sp["resolved"] is True
    assert sp["n_peaks"] >= 4
    assert sp["combo_err"] < 0.15, (
        f"the 4 strongest in-band peaks are not the 4 quadratic combinations "
        f"(max err {sp['combo_err']:.4f} Hz)"
    )
    assert sp["dominance"] > 3.0, (
        f"the combinations do not dominate the spectrum: weakest combo vs strongest non-combo = "
        f"{sp['dominance']:.2f}x"
    )
    # Measured, never predicted — the combinations are exact functions of the measured partials...
    assert sp["combos"]["f1+f2"] == pytest.approx(sp["f1"] + sp["f2"], rel=1e-12)
    assert sp["combos"]["f2-f1"] == pytest.approx(sp["f2"] - sp["f1"], rel=1e-12)
    # ...and NOT of the ladder, which is shipped only as a shown-not-scored reference.
    assert abs(sp["combos"]["f1+f2"] - (sp["ladder"][0] + sp["ladder"][1])) > 0.05


def test_phantom_no_longitudinal_peak_sits_on_a_transverse_partial(phantom):
    """The other half of the claim, and the more discriminating half.

    A *linear* coupling between the fields — the bug this excludes — would reproduce the transverse
    partials in ``v`` directly. ``f1`` and ``f2`` are the only partials actually excited, so they
    are the only two that can be checked without an oracle: no in-band longitudinal peak comes near
    either.
    """
    sp = phantom["meta"]["spectrum"]
    for name in ("f1", "f2"):
        near = sp[f"nearest_to_{name}"]
        assert abs(near - sp[name]) > 3.0, (
            f"a longitudinal peak sits on the transverse partial {name} = {sp[name]:.3f}: nearest "
            f"peak {near:.3f}. The coupling is supposed to be quadratic."
        )


def test_phantom_headline_is_the_defect_measured_from_both_sides(phantom):
    """**The headline**: the displacement that puts a phantom in a *gap* is ``f2 - 2 f1``, and it is
    one number measured in one run — no oracle, no confound.

    For a harmonic string the low phantoms coincide with partials *exactly* (``f2 - f1 = f1`` and
    ``2 f1 = f2``), so the distance from the difference tone to ``f1`` and from ``2 f1`` to ``f2``
    are BOTH exactly ``|f2 - 2 f1|`` — the defect, nonzero only because kappa > 0. Approaching the
    same physical number from opposite sides is what makes it a measurement rather than an
    assertion.

    No hardening confound either: hardening moves the phantoms and the partials *together*, and
    measured it slightly WIDENS the defect, so it works against the claim rather than for it.
    """
    sp = phantom["meta"]["spectrum"]
    defect = sp["defect"]
    assert defect == pytest.approx(sp["f2"] - 2 * sp["f1"], rel=1e-12)
    assert defect > GEOM_PHANTOM_DEFECT_MIN
    d1, d2 = sp["displacements"]
    assert d1 == pytest.approx(defect, abs=0.3)
    assert d2 == pytest.approx(defect, abs=0.3)
    assert min(d1, d2) > 3.0, "the phantoms must land in the gaps, not on the partials"


def test_phantom_conserves_through_the_wrapper(phantom):
    """Criterion 1 survives 7,950 Newton steps of the most arithmetic-heavy regime in the viewer."""
    e = phantom["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL
    assert e["lossless"]["pass"] is True


def test_phantom_window_is_fixed_physics_and_ignores_the_animation_slider(phantom):
    """The 0.10 s window is not a preference, so the slider that sets every other regime's window is
    ignored here (and hidden in the UI).

    The fixture asks for ``animation_window = 0.3`` — 3x — and gets 0.10 s of physics anyway.
    Halving the window does not merely cost precision: at 0.05 s the raw bins are 20 Hz wide, the
    ``2 f1`` phantom (the weakest of the four) sits inside its neighbours' leakage skirts and is
    mislocated by 0.52 Hz against 0.039 Hz for the worst of the four at 0.1 s, and its margin to
    ``f2`` collapses from 170x to 8x.
    """
    assert GEOM_PHANTOM["animation_window"] == 0.3, "the fixture must ask for the wrong window"
    assert phantom["meta"]["num_steps"] == round(GEOM_PHANTOM_WINDOW * phantom["fs_sim"])


def test_phantom_display_grid_is_denser_than_the_measurement_grid(phantom):
    """The display/measurement split, pinned structurally.

    Zero-padding densifies the bin grid without adding real resolution: exactly what a *plot* needs
    and nothing a *measurement* may lean on. So it is confined to the drawn traces, while the
    detector keeps ``magnitude_spectrum``'s default — which is what the test rig measures, and where
    the 0.039 Hz peak-location error comes from (parabolic refinement, independent of the display
    grid).

    At the rig's 2x pad the bins are ~5 Hz and the ~57 Hz zoom band would hold about a dozen points:
    not a picture. This asserts the zoom strip is an order of magnitude denser than that, which can
    only be true if the display path pads separately.
    """
    sp = phantom["meta"]["spectrum"]
    zoom_hz = sp["zoom"][1] - sp["zoom"][0]
    detector_nfft = 2 ** np.ceil(np.log2(phantom["meta"]["num_steps"] * 2))
    detector_df = phantom["fs_sim"] / detector_nfft
    assert len(sp["zoom_freq"]) > 10 * (zoom_hz / detector_df), (
        f"the zoom strip has {len(sp['zoom_freq'])} points over {zoom_hz:.1f} Hz — no denser than "
        f"the detector's {detector_df:.2f} Hz bins, so the defect will render sub-pixel"
    )
    assert len(sp["wide_freq"]) > 100


def test_phantom_ships_the_audio_that_batch_3_deferred(phantom):
    """The debt: model #10 went viz-only, promising the bridge force as its true audible signature.

    This is the one geometric regime with sound, and only because its window is *already* 0.1 s of
    the radiating channel — so the clip is free rather than affordable. It is honest about being a
    blip: 0.1 s, and dominated by the longitudinal startup transient rather than by the phantoms.
    """
    assert phantom["audio"] is not None
    assert phantom["audio"]["fs"] == AUDIO_FS
    assert phantom["audio"]["n"] == pytest.approx(GEOM_PHANTOM_WINDOW * AUDIO_FS, rel=0.02)
    assert phantom["audio"]["peak"] > 0.0
    assert np.all(np.isfinite(_decode_f32(phantom["audio"]["b64"])))
    assert "bridge force" in phantom["audio_note"]
    # Every other regime still ships none, and says why rather than shipping a stub.
    planar = _geom(domain="planar")
    assert planar["audio"] is None
    assert "viz-only" in planar["audio_note"]


def test_phantom_linear_string_has_no_channel_to_put_a_phantom_in():
    """**The harness control** — and it is nearly free (``EA = T0`` makes ``c_long = c``, so ``fs``
    collapses from 159 kHz to 1.8 kHz and the run is 178 steps).

    ``a = EA - T0 = 0`` kills the whole nonlinear excess, so the three fields decouple and ``v``,
    started at rest, never leaves it: the bridge force is zero *identically*, not "small". This is
    what rules out the readout manufacturing its own result — same IC, same readout, same blind
    detector, and the only change is the one coefficient the phantom is supposed to come from.

    It also pins the verdict ORDER, which is a real trap rather than a nicety: ``detect_peaks``
    returns an empty list on a zero signal rather than raising, so without the ``linear`` check
    firing first, a defect above the gate would paint "the 4 strongest peaks ARE the 4 combinations"
    over a spectrum with no peaks in it at all.
    """
    d = simulate_to_payload({**GEOM_PHANTOM, "N": 8, "EA": 200.0})   # EA == T0
    sp = d["meta"]["spectrum"]
    assert sp["linear"] is True
    assert sp["bridge_max"] == 0.0, "EA = T0 must leave v identically zero — not merely small"
    assert sp["n_peaks"] == 0
    assert sp["resolved"] is False, "no channel means no claim, whatever the defect says"
    assert d["audio"]["peak"] == 0.0


def test_phantom_labels_a_grid_too_coarse_to_show_the_stiffness():
    """Label-not-fail, and the reason is a measured surprise worth pinning: **N is a second control
    on the defect, not just kappa** — at kappa = 8, plenty of stiffness, N = 8 still cannot show it.

    ``f2 - 2 f1`` is not pure stiffness. The theta-scheme's temporal dispersion drags mode 2 *flat*,
    so it contributes a NEGATIVE defect, and what is measured is the difference of the two. Off the
    linear ladder at kappa = 8 the defect runs +0.38 / +3.57 / +4.38 Hz at N = 8 / 16 / 32, and at
    kappa = 0 it is pure dispersion (-0.97 / -0.43 / -0.24, converging as O(h^2)). So at N = 8 the
    grid eats the physics and there is nothing to discriminate — a fact about the run rather than a
    solver failure, so the panel labels it (the bow's Schelleng-window precedent).

    Note this is NOT the linear control above: the channel is wide open (peaks are detected, the
    phantoms are there), they have merely collapsed onto the partials.
    """
    d = simulate_to_payload({**GEOM_PHANTOM, "N": 8})
    sp = d["meta"]["spectrum"]
    assert sp["linear"] is False, "the phantom channel is open — this is not the a = 0 control"
    assert sp["n_peaks"] >= 4, "the phantoms are present; they are merely not discriminating"
    assert sp["defect"] < GEOM_PHANTOM_DEFECT_MIN
    assert sp["resolved"] is False


def test_phantom_defect_gate_is_one_sided_not_absolute():
    """The gate is ``defect >= 3.0``, never ``abs(defect) >= 3.0``, and that is load-bearing.

    A coarse-enough grid gives a large *negative* defect — phantoms displaced to the wrong side of
    the partials by numerical dispersion rather than by physics — which an ``abs()`` gate would
    happily score. Requiring a POSITIVE defect means "the partials are stretched by real stiffness,
    by enough to see", which is the only version of this claim worth making. Driven directly rather
    than through a run: the reachable slider range makes a large negative defect hard to reach, and
    a guard that cannot trip in range is a guarantee, not dead code (the tension string's lesson).
    """
    assert GEOM_PHANTOM_DEFECT_MIN > 0
    for defect in (-10.0, -3.5, -0.5, 0.0, 2.9):
        assert not (defect >= GEOM_PHANTOM_DEFECT_MIN), (
            f"defect {defect} must not be scorable: negative means the scheme's dispersion, not "
            f"stiffness, is displacing the phantoms"
        )
    assert 3.79 >= GEOM_PHANTOM_DEFECT_MIN     # the measured N=16 defect clears it


@pytest.mark.parametrize("bad", [
    # The window is fixed, so the only way past the budget is N x lam_long: fs = c_long N/(L lam).
    {"N": 32, "lam_long": 0.2},
    # dT/T0 = a A^2 (p2_1 + p2_2)/(4 T) — bound the tension excess, never the amplitude (A is a
    # proxy; EA and T move it just as hard). At A = 0.02 this lands ~2.4 against the 2.2 cap.
    {"amplitude": 0.02},
])
def test_phantom_budget_and_amplitude_guards_give_clean_error_payloads(bad):
    """Both phantom-specific guards reject at construction — no 500, no NaN render, no 45 s wait."""
    d = simulate_to_payload({**GEOM_PHANTOM, **bad})
    assert "error" in d, f"{bad} should have been rejected"
    assert d["error"]["kind"] in ("param", "construction")


def test_phantom_has_its_own_work_budget_because_it_is_the_slowest_render():
    """Its budget is ~2.75x the other regimes'. The window is fixed physics and fs rides the
    longitudinal wave, so ~15,900 vector Newton steps (~45 s) is the floor at the default N = 32 —
    GEOM_WORK_MAX would reject the regime's own default parameters."""
    default_steps = round(GEOM_PHANTOM_WINDOW * 159_009)      # N=32, lam_long=0.9, EA=1e5, rho=5e-3
    assert GEOM_PHANTOM_WORK_MAX > GEOM_WORK_MAX
    assert default_steps < GEOM_PHANTOM_WORK_MAX              # the N=32 default fits...
    assert default_steps > GEOM_WORK_MAX                      # ...and would not have, before


# == sympathetic / coupled strings (1D, J=2 — model in core/connection.py) =========================
# The viewer's first MULTI-STRING model, and the second customer of the stacked-strip drawFields.
# It is a CLOSED, UNDRIVEN, linear-leapfrog system, so energy conservation + passivity are AUTOMATIC
# and pass even a flipped coupling sign — table-stakes, not the claim. What these pin is the claim
# energy cannot see: (a) the antisymmetric normal mode keeps the shared bridge EXACTLY still
# (w_b == 0 bit-exact) while the symmetric contrast swings it and loads the body; (b) sympathetic
# transfer drains a tuned neighbour far more than a detuned one; plus the frames/audio bookkeeping,
# the ordinary drift verdict (no balance panel, no decay oracle), and clean error payloads.
SYMP_NORMAL: dict = {"model": "sympathetic", "domain": "normal", "N": 60, "lambda": 0.9,
                     "T": 200.0, "rho": 0.005, "L": 1.0, "K": 8000.0,
                     "pluck_position": 0.3, "pickup_position": 0.1, "audio_duration": 0.4}
SYMP_TRANSFER: dict = {**SYMP_NORMAL, "domain": "transfer", "K": 1500.0, "audio_duration": 1.5}


def _symp(**overrides):
    return simulate_to_payload({**SYMP_NORMAL, **overrides})


def test_symp_antisymmetric_mode_keeps_the_bridge_bit_exact_still():
    """THE discriminating oracle — the claim energy conservation cannot see.

    Two identical strings started ``u_B = -u_A`` keep the shared bridge at **bit-exact**
    zero forever (IEEE: the float negation of A's IC is exact, ``sum([a, -a]) == 0.0``), so the
    body feels no force and stays at rest. A flipped coupling sign or a mis-summed force would move
    the bridge and fail this at once, while energy conservation would pass either way.
    """
    sp = _symp()["meta"]["spectrum"]
    assert sp["kind"] == "sympathetic" and sp["regime"] == "normal"
    assert sp["anti_max"] == 0.0, "not 'small' — exactly zero, or the coupling sign/sum is wrong"
    assert sp["anti_exact_zero"] is True
    assert sp["body_frac_anti"] == 0.0, "a still bridge cannot load the body"


def test_symp_symmetric_mode_is_the_contrast_that_makes_the_zero_mean_something():
    """Its opposite: ``u_B = +u_A`` swings the bridge and floods the body — a flat zero alone reads
    as 'broken', so both traces are shipped and the symmetric one must be genuinely large."""
    sp = _symp()["meta"]["spectrum"]
    assert sp["sym_max"] > 1e6 * max(sp["anti_max"], 1e-30), "the symmetric bridge barely moved"
    assert sp["sym_max"] > 0.0
    assert sp["body_frac_sym"] > 0.05, "the symmetric mode should load the body appreciably"


def test_symp_detune_is_ignored_in_the_normal_regime():
    """The bit-exact zero needs the two strings identical, so the detune slider is gated OUT of
    normal — passing it must not perturb the oracle (the backend never reads it here)."""
    sp = _symp(detune=5.0)["meta"]["spectrum"]
    assert sp["anti_max"] == 0.0 and sp["anti_exact_zero"] is True


def test_symp_normal_conserves_through_the_wrapper_ordinary_drift():
    """A closed undriven system: the ordinary conservation-drift verdict, no balance panel and no
    decay oracle (those are for driven / decaying models)."""
    e = _symp()["energy"]
    assert e["sigma_is_zero"] is True
    assert e.get("kind") != "balance"
    assert "convergence" not in e
    assert e["lossless"]["drift"] < LOSSLESS_TOL
    assert e["lossless"]["pass"] is True


def test_symp_frames_decode_to_two_mirror_strings_with_clamped_nuts():
    """Frames are (n_frames, J=2, N+1); dims stays 1 (the string path is untouched). In the
    antisymmetric animation string B is the exact negation of string A, and both are clamped (0) at
    the nut (left, fixed) with a free bridge end."""
    d = _symp()
    fr = d["frames"]
    assert fr["dims"] == 1 and fr["fields"] == ["string A", "string B"]
    assert len(fr["field_labels"]) == 2
    buf = _decode_f32(fr["b64"]).reshape(fr["n_frames"], 2, fr["width"])
    assert np.array_equal(buf[:, 1, :], -buf[:, 0, :]), "string B is not the antiphase of string A"
    assert np.all(buf[:, :, 0] == 0.0), "the nut (fixed left end) is not clamped"


def test_symp_audio_is_the_plucked_string_pickup_not_silence():
    """Sympathetic strings are NOT viz-only (fs ~ 22 kHz, unlike the geometric string): audio is a
    real string pickup. Body pressure would go silent on the antisymmetric mode, but the string
    still rings, so the pickup carries it."""
    a = _symp()["audio"]
    assert a is not None and a["n"] > 0 and a["fs"] == AUDIO_FS
    x = _decode_f32(a["b64"])
    assert np.all(np.isfinite(x)) and float(np.max(np.abs(x))) > 0.0


def test_symp_transfer_tuned_unison_drains_most_of_the_energy():
    """Pluck string A; at unison the tuned neighbour drains most of the total energy (the classic
    near-complete coupled-oscillator exchange), and the fraction starts at ~0 and rises."""
    d = simulate_to_payload({**SYMP_TRANSFER, "detune": 0.0})
    sp = d["meta"]["spectrum"]
    assert sp["kind"] == "sympathetic" and sp["regime"] == "transfer" and sp["tuned"] is True
    assert sp["peak_neighbour"] > 0.5, f"tuned neighbour barely rang ({sp['peak_neighbour']:.2f})"
    assert sp["frac1"][0] < 0.05, "the neighbour starts at rest"
    assert sp["frac0"][0] > 0.9, "the plucked string starts with essentially all the energy"


def test_symp_transfer_detuned_neighbour_stays_quiet():
    """Its contrast: a neighbour ~4 semitones off unison barely responds — the coupling is
    frequency-selective, which is why a sympathetic string lights up only for the right note."""
    tuned = simulate_to_payload({**SYMP_TRANSFER, "detune": 0.0})["meta"]["spectrum"]
    detuned = simulate_to_payload({**SYMP_TRANSFER, "detune": 4.0})["meta"]["spectrum"]
    assert detuned["peak_neighbour"] < 0.25, "detuned neighbour rang too much"
    assert tuned["peak_neighbour"] > 3.0 * detuned["peak_neighbour"]


def test_symp_transfer_conserves_and_the_fractions_stay_physical():
    """Transfer is lossless too — the ordinary drift verdict — and each per-string energy fraction
    is a fraction (in [0, 1] up to the body/connection share)."""
    d = simulate_to_payload({**SYMP_TRANSFER, "detune": 0.0})
    assert d["energy"]["lossless"]["drift"] < LOSSLESS_TOL
    sp = d["meta"]["spectrum"]
    for arr in (sp["frac0"], sp["frac1"]):
        assert all(-1e-9 <= v <= 1.0 + 1e-9 for v in arr)


def test_symp_lambda_must_be_below_one():
    """lambda = 1 leaves the string's Nyquist mode marginal and the bridge spring pushes it
    unstable, so the coupled system needs headroom below it (a clean error, not a 500)."""
    r = simulate_to_payload({**SYMP_NORMAL, "lambda": 1.0})
    assert "error" in r and "lambda must be in (0, 1)" in r["error"]["message"]


def test_symp_over_stiff_bridge_is_rejected_by_the_core_guard():
    """K is gated by the core's exact dense leapfrog guard (k^2 lambda_max(A) < 4), surfaced as a
    clean construction error."""
    r = simulate_to_payload({**SYMP_NORMAL, "K": 1.0e6})
    assert "error" in r and "unstable" in r["error"]["message"]


@pytest.mark.parametrize("bad", [
    {"N": SYMP_N_MAX + 1}, {"domain": "nonesuch"}, {"detune": -1.0, "domain": "transfer"},
    {"detune": 99.0, "domain": "transfer"}, {"K": -5.0},
    {"domain": "weinreich", "sigma_body": 999.0}, {"domain": "weinreich", "detune": 5.0},
])
def test_symp_bad_params_give_error_payload(bad):
    r = simulate_to_payload({**SYMP_NORMAL, **bad})
    assert "error" in r, f"expected a clean error payload for {bad}"


def test_symp_normal_work_budget_counts_both_runs():
    """The normal regime runs TWICE (antisymmetric + symmetric contrast), so its budget is on
    ``2 * n_steps``; a long-enough audio duration trips it with a clean error."""
    r = simulate_to_payload({**SYMP_NORMAL, "audio_duration": 3.0, "T": 800.0, "N": 160})
    assert "error" in r and "work budget" in r["error"]["message"]
    # and the default sits comfortably inside it
    assert 2 * round(0.4 * (200.0 / 0.005) ** 0.5 * 60 / 0.9) < SYMP_WORK_MAX


# -- weinreich two-stage decay (the piano-unison regime, the first body-loss slider) ---------------
# Strike ONE of two near-unison strings over a LOSSY bridge: the symmetric mode loads the bridge and
# dies fast (prompt), the antisymmetric mode barely loads it and lingers (aftersound), so the
# string-energy envelope shows the fast-then-slow knee. The sharp sub-claim is at detune = 0: the
# antisymmetric mode is bit-exactly bridge-decoupled (the normal-mode oracle), so it never loads the
# lossy body and its tail is LOSSLESS -> aftersound slope ~ 0, rising clearly with detune. The
# contrast (strike BOTH = the pure symmetric mode) decays away single-slope with no aftersound. The
# energy verdict is passivity with decay_oracle=False (no single-exponential oracle); sigma_body = 0
# flips it back to the drift check.
SYMP_WEIN: dict = {**SYMP_NORMAL, "domain": "weinreich", "K": 6000.0, "sigma_body": 20.0,
                   "detune": 0.0, "audio_duration": 1.5}


def _wein(**overrides):
    return simulate_to_payload({**SYMP_WEIN, **overrides})


def test_symp_weinreich_two_stage_knee_prompt_faster_than_aftersound():
    """The two-stage decay: the prompt (fast, symmetric mode) dies far faster than the aftersound
    (slow, antisymmetric mode). A single exponential is also passive-monotone but has no knee, so
    prompt >> aftersound IS the claim energy conservation cannot see."""
    sp = _wein()["meta"]["spectrum"]
    assert sp["kind"] == "sympathetic" and sp["regime"] == "weinreich"
    assert sp["prompt_rate"] > 0.5, "the prompt should decay at a visible rate"
    assert sp["prompt_rate"] > 5.0 * max(sp["aftersound_rate"], 1e-6), "no two-stage knee"


def test_symp_weinreich_unison_aftersound_is_lossless_rising_with_detune():
    """THE sharp sub-claim, tied to the batch-6 normal-mode oracle. At detune = 0 the antisymmetric
    mode is bit-exactly bridge-decoupled, so the body's damping never activates on it and the
    aftersound is lossless (slope ~ 0). A few cents of mistune loads the bridge a little, so the
    aftersound decays slowly — clearly faster than at unison."""
    unison = _wein(detune=0.0)["meta"]["spectrum"]
    mistuned = _wein(detune=0.3)["meta"]["spectrum"]
    assert unison["aftersound_rate"] < 0.1, "the unison aftersound should be near-lossless"
    assert mistuned["aftersound_rate"] > 2.5 * max(unison["aftersound_rate"], 1e-3), (
        "detuning must speed the aftersound up — the antisymmetric mode now loads the bridge"
    )


def test_symp_weinreich_strike_both_decays_away_no_aftersound():
    """The contrast that makes the plateau mean something: strike-one keeps a big fraction of its
    energy in the un-decaying antisymmetric mode (the aftersound floor), while strike-both (the pure
    symmetric mode) loads the bridge fully and falls away. The plateau is physics, not a floor."""
    sp = _wein()["meta"]["spectrum"]
    assert sp["floor_one"] > 0.3, "strike-one should retain a real aftersound"
    assert sp["both_final"] < 0.15, "strike-both should decay away"
    assert sp["floor_one"] > 2.0 * sp["both_final"], "the aftersound contrast is not clear"


def test_symp_weinreich_lossy_body_reports_passivity_without_a_decay_oracle():
    """A lossy body -> the passivity verdict (monotone), NOT the drift check and NOT a fitted-2σ
    oracle: the total energy is a two-rate decay to a nonzero aftersound floor, so a single measured
    2σ against a flat oracle would be a lying 'broken match' (the mallet's decay_oracle=False)."""
    e = _wein()["energy"]
    assert e["sigma_is_zero"] is False
    assert "lossy" in e and e["lossy"]["monotone"] is True
    assert "measured_2sigma" not in e["lossy"], "the flat-loss oracle must be dropped"
    assert e.get("kind") != "balance" and "convergence" not in e


def test_symp_weinreich_zero_body_loss_flips_to_the_drift_check():
    """The body-loss slider genuinely toggles the verdict type: at sigma_body = 0 the body is
    lossless, the whole closed system conserves, and the panel reports the conservation-drift check
    (with both rates ~ 0 — no loss, no two-stage decay)."""
    d = _wein(sigma_body=0.0)
    e = d["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True
    sp = d["meta"]["spectrum"]
    assert sp["sigma_zero"] is True
    assert abs(sp["aftersound_rate"]) < 0.05 and sp["prompt_rate"] < 0.5, "nothing should decay"


def test_symp_weinreich_envelopes_are_finite_and_normalized():
    """Both string-energy envelopes are normalized by their own initial energy, so they start near 1
    (the 1-period centered mean dips slightly over the prompt's fast decay) and stay finite and
    physical (in [0, ~1.05]) — the log-y panel plots these directly."""
    sp = _wein()["meta"]["spectrum"]
    for key in ("env_one", "env_both"):
        arr = sp[key]
        assert 0.85 <= arr[0] <= 1.05, f"{key} does not start near 1 (got {arr[0]})"
        assert all(v is not None and -1e-9 <= v <= 1.05 for v in arr)


def test_symp_weinreich_audio_is_the_struck_string_pickup():
    """The struck string rings audibly (the two-stage decay is an audible piano signature), so the
    audio is a real pickup, not silence — and the frames carry both strings (B starts silent)."""
    d = _wein()
    a = d["audio"]
    assert a is not None and a["fs"] == AUDIO_FS
    assert float(np.max(np.abs(_decode_f32(a["b64"])))) > 0.0
    fr = d["frames"]
    assert fr["dims"] == 1 and len(fr["fields"]) == 2


def test_symp_weinreich_work_budget_counts_both_runs():
    """Like the normal regime, weinreich runs TWICE (strike-one + its strike-both contrast), so the
    budget is on 2 * n_steps and a long-enough duration trips it with a clean error."""
    r = _wein(audio_duration=3.0, T=800.0, N=160)
    assert "error" in r and "work budget" in r["error"]["message"]


def test_symp_weinreich_detune_range_is_fine_not_semitones():
    """A piano unison is mistuned by a few cents, so the weinreich detune range is small (0..0.4
    semis), distinct from transfer's 0..12 — a transfer-scale detune is rejected here."""
    assert SYMP_WEINREICH_DETUNE_MAX < 1.0
    assert "error" in _wein(detune=SYMP_WEINREICH_DETUNE_MAX + 0.5)
    assert "error" in _wein(sigma_body=SYMP_SIGMA_BODY_MAX + 1.0)


# =================================================================================================
# Jawari / buzzing bridge (model #8, curved) — the claim is the shimmer, not the energy
# =================================================================================================
#
# The viewer runs the SAME rig as tests/test_jawari.py (mode-1 sine at 8 mm, sigma0 = 0.5, the
# curved parabola, a clean contrast at clearance = 1 m), so the suite's own validated numbers come
# back through the wrapper and act as a free end-to-end oracle: elevation ~3.4x and wrap-edge std
# ~4.9 are not new claims invented here, they are the model's, re-measured through the payload.


def _jaw(**over):
    return web_serialize.simulate_to_payload({"model": "jawari", "audio_duration": 0.24, **over})


def test_jawari_reproduces_the_suites_shimmer_and_wrap_numbers():
    """The end-to-end oracle: the payload's headline numbers ARE tests/test_jawari.py's. The late
    window is ~3.4x brighter than the clean string (the suite gates > 2.5x) and the wrap edge
    sweeps with std ~4.9 (the suite's flat rail at matched clearance pins at ~2.35). If the wrapper
    perturbed the rig — a different IC, a stale param, a shifted window — these would drift."""
    sp = _jaw()["meta"]["spectrum"]
    assert sp["kind"] == "jawari"
    assert sp["elevation"] == pytest.approx(3.44, abs=0.05)
    assert sp["shimmering"] is True and sp["elevation"] > JAWARI_ELEVATION_GATE
    assert sp["wrap"]["std"] == pytest.approx(4.89, abs=0.05)
    assert sp["wrap"]["min_node"] == 0 and sp["wrap"]["max_node"] == 14
    # the clean contrast must be spectrally pure — that is what makes the elevation attributable
    assert sp["centroid"]["clean_late"] == pytest.approx(sp["f1"], rel=0.02)


def test_jawari_energy_keeps_the_flat_loss_oracle_unlike_the_mallet():
    """The load-bearing verdict decision. The bridge is a LOSSLESS elastic barrier: it moves energy
    into the highs but dissipates none, so every mode still decays at exactly 2*sigma0 and the
    flat-loss oracle survives the wrap. This is why neither the mallet's nor weinreich's
    decay_oracle=False applies here — dropping it would throw away a strictly stronger verdict."""
    e = _jaw(sigma0=0.5)["energy"]
    assert e["sigma_is_zero"] is False
    assert e["lossy"]["monotone"] is True
    assert e["lossy"]["oracle_2sigma"] == pytest.approx(1.0)
    assert e["lossy"]["measured_2sigma"] == pytest.approx(1.0, rel=0.02)


def test_jawari_sigma0_gates_the_verdict_and_conserves_through_the_curved_wrap():
    """sigma0 = 0 flips the panel to the conservation-drift check, and the drift survives a
    SUSTAINED many-node curved contact (~1e-12, the money gate of the jawari batch) — not a
    contact-free run that would prove nothing about the barrier."""
    d = _jaw(sigma0=0.0)
    e = d["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True
    assert d["meta"]["spectrum"]["wrap"]["duty"] > 0.1, "must actually be in contact"


def test_jawari_grazing_config_is_labelled_not_failed():
    """Below the downswing/depth floor the string only grazes the crest: a legitimate stiff point
    contact, just not a jawari. It must render with an honest label — never an error and never a
    green shimmer badge (the bow's Schelleng-window rule: real physics gets labelled, never
    failed)."""
    sp = _jaw(depth=6.0e-3)["meta"]["spectrum"]
    assert sp["grazing"] is True and sp["ratio"] < JAWARI_RATIO_FLOOR
    assert sp["shimmering"] is False and sp["elevation"] < JAWARI_ELEVATION_GATE
    # the mechanism, not just the score: the wrap contracts toward the crest
    assert sp["wrap"]["max_node"] < 14


def test_jawari_ratio_is_the_control_and_amplitude_moves_it_as_hard_as_depth():
    """downswing/depth is the dimensionless coordinate (the tension dT/T0 pattern): depth is only
    half of it, and quartering the amplitude degrades the shimmer exactly as deepening the bridge
    does. A depth-only guard would let an under-plucked string through none the wiser."""
    base = _jaw()["meta"]["spectrum"]
    quiet = _jaw(amplitude=2.0e-3)["meta"]["spectrum"]
    assert quiet["ratio"] == pytest.approx(base["ratio"] / 4.0, abs=0.01)  # ratio is 2-dp rounded
    assert quiet["elevation"] < base["elevation"]


def test_jawari_ignores_the_mallet_alpha_and_the_sympathetic_K():
    """Param-collision guard. The frontend sends EVERY slider, so a user who visited the mallet or
    the sympathetic strings would otherwise ship alpha = 2.3 and K = 8000 into this model — a
    different felt exponent and a 250x softer bridge — and the render would silently change with
    nothing on screen to say so. The jawari reads neither name; it uses bridge_stiffness."""
    base = _jaw()["meta"]["spectrum"]
    leaked = _jaw(alpha=2.3, K=8000)["meta"]["spectrum"]
    assert leaked["elevation"] == base["elevation"]
    assert leaked["wrap"]["std"] == base["wrap"]["std"]
    # ... while the name it DOES read still bites
    assert _jaw(bridge_stiffness=2.0e5)["meta"]["spectrum"]["elevation"] != base["elevation"]


def test_jawari_payload_carries_the_bridge_profile_and_the_wrap_marker():
    """The animation's two extras: the barrier profile (finite on its support, null off it — the
    frontend draws the solid the string rests on) and the per-frame departure node."""
    d = _jaw()
    b = d["grid"]["barrier"]
    assert len(b) == len(d["grid"]["x"])
    finite = [v for v in b if v is not None]
    assert len(finite) == d["meta"]["spectrum"]["wrap"]["support"]
    assert all(v <= 0.0 for v in finite), "the bridge sits at or below the rest line"
    assert finite == sorted(finite, reverse=True), "a parabola falling away from the crest"
    assert len(d["wrap_frames"]) == d["frames"]["n_frames"]
    # wrap_frames are GRID node indices, not support-relative ones: the core reports contact over
    # its ~15-node support, and shipping that frame straight through puts the marker one node off
    # the contact it marks — plausible-looking, and wrong. Every marked node must be ON the bridge.
    assert all(w == -1 or b[w] is not None for w in d["wrap_frames"])
    assert any(w >= 0 for w in d["wrap_frames"]), "the string must contact during the window"


def test_jawari_late_spectra_share_one_scale():
    """Both traces are normalized by the SAME peak, because their relative height up the band is
    the entire claim — per-trace normalization would render two similar curves and delete it."""
    sp = _jaw()["meta"]["spectrum"]["spectra"]
    jm = [v for v in sp["jawari"]["mag"] if v is not None]
    cm = [v for v in sp["clean"]["mag"] if v is not None]
    assert max(max(jm), max(cm)) == pytest.approx(1.0)
    assert max(jm) < 1.0 or max(cm) < 1.0, "one trace must be the normalizer, not both"
    assert len(sp["jawari"]["f"]) == len(sp["jawari"]["mag"])


def test_jawari_audio_is_real_and_finite():
    d = _jaw()
    a = d["audio"]
    assert a is not None and a["fs"] == AUDIO_FS
    sig = _decode_f32(a["b64"])
    assert np.all(np.isfinite(sig)) and float(np.max(np.abs(sig))) > 0.0


def test_jawari_work_budget_counts_both_runs_and_the_guards_are_reachable():
    """The clean contrast is a SECOND full run, so the budget is on 2 * n_steps — a per-run cap
    would silently licence twice the wall clock. Worst passing render measured at ~34 s."""
    r = _jaw(audio_duration=1.5, N=128, width_frac=0.4)
    assert "error" in r and str(JAWARI_WORK_MAX) in r["error"]["message"]
    assert "error" in _jaw(N=JAWARI_N_MAX + 1)
    assert "error" in _jaw(depth=JAWARI_DEPTH_MAX + 1e-3)
    assert "error" in _jaw(amplitude=JAWARI_AMP_MAX + 1e-3)
    assert "error" in _jaw(bridge_stiffness=0.0)


def test_jawari_sustain_ratio_is_reported_but_never_gates():
    """The late/early ratio wobbles 0.9-1.3 with the decay rate and window placement, so it is
    printed and never scored — gating it would make a correct render flaky."""
    sp = _jaw()["meta"]["spectrum"]
    assert sp["sustain_ratio"] is not None and sp["clean_sustain_ratio"] is not None
    assert sp["shimmering"] == (sp["elevation"] > JAWARI_ELEVATION_GATE)


# =================================================================================================
# Juari / tanpura cotton thread (model #8, a single-node POINT contact) — the claim is the
# TUNING CURVE (buzz vs thread position), which the jawari's distributed wrap cannot express
# =================================================================================================
#
# Most tests run FAST (small N, a tiny non-canonical sweep_duration): the buzz is POSITION-dependent
# at any duration, so the wiring — a curve that varies with position, a marker on it, the flat clean
# and jawari reference lines, the honest grid resolution — is checkable cheaply. The one physics
# claim that needs the SETTLED regime (the sweet spot sits near the nut) gets its own slower test.


def _jua(**over):
    # Small N + a tiny non-canonical sweep_duration: the per-node construction (a fresh factorized
    # string per thread position) dominates the cost, so N is the lever that keeps these fast.
    base = {"model": "juari", "N": 40, "sweep_duration": 0.04, "audio_duration": 0.05}
    base.update(over)
    return web_serialize.simulate_to_payload(base)


def test_juari_tuning_curve_is_position_selective():
    """The headline and the clean separator from the jawari: the buzz VARIES with thread position
    (the jawari's is a single position-independent number). The clean baseline is spectrally pure,
    so the elevation is attributable to the thread; and every swept array is the same length."""
    sp = _jua()["meta"]["spectrum"]
    assert sp["kind"] == "juari"
    tn = sp["tuning"]
    assert len(tn["node"]) == len(tn["frac"]) == len(tn["x"]) == len(tn["elevation"])
    elev = [e for e in tn["elevation"] if e is not None]
    assert max(elev) - min(elev) > 0.6, "the tuning curve must be position-selective, not flat"
    assert sp["centroid"]["clean_late"] == pytest.approx(sp["f1"], rel=0.05)


def test_juari_thread_marker_sits_on_the_drawn_curve():
    """The selected thread's elevation IS one of the swept points, so the frontend marker lands on
    the curve rather than floating beside it — the selected node is always folded into the sweep."""
    sp = _jua(thread_position=0.1)["meta"]["spectrum"]
    node = sp["thread"]["node"]
    assert node in sp["tuning"]["node"]
    i = sp["tuning"]["node"].index(node)
    assert sp["thread"]["elevation"] == sp["tuning"]["elevation"][i] == sp["elevation"]


def test_juari_energy_keeps_the_flat_loss_oracle_like_the_jawari():
    """A single-node point contact is still a LOSSLESS elastic barrier: it moves energy into the
    highs but dissipates none, so decay_oracle stays TRUE and every mode decays at exactly 2*sigma0
    (the jawari's decision, carried unchanged — not the mallet's decay_oracle=False)."""
    e = _jua(sigma0=0.5)["energy"]
    assert e["sigma_is_zero"] is False
    assert e["lossy"]["monotone"] is True
    assert e["lossy"]["measured_2sigma"] == pytest.approx(e["lossy"]["oracle_2sigma"], rel=0.05)


def test_juari_sigma0_gates_and_conserves_through_the_point_contact():
    """sigma0 = 0 flips to the conservation-drift check, and the drift survives a REAL point contact
    (the string must actually be caught on the thread, not clear of it — else the gate proves
    nothing about the barrier)."""
    d = _jua(sigma0=0.0)
    e = d["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True
    assert sum(d["contact_frames"]) > 0, "the string must contact the thread during the window"
    assert len(d["contact_frames"]) == d["frames"]["n_frames"]


def test_juari_thread_snaps_to_a_grid_node_and_reports_the_resolution():
    """The point contact lives ON the grid, so thread_position snaps to a node and the tuning
    resolution IS the node spacing — the honesty the panel foregrounds. The marker is scattered onto
    grid coordinates (thread_x == x[thread_node]), the single-node analogue of the jawari's
    support->grid scatter."""
    d = _jua(N=40, thread_position=0.1)
    node = d["grid"]["thread_node"]
    assert 1 <= node <= 39
    assert d["grid"]["thread_x"] == pytest.approx(d["grid"]["x"][node], abs=1e-6)
    q = d["meta"]["spectrum"]["quantization"]
    assert q["h"] == pytest.approx(1.0 / 40, abs=1e-4)
    assert q["near_nut_nodes"] == round(q["near_nut_frac"] * 40)


def test_juari_reference_lines_are_clean_1x_and_a_flat_jawari():
    """The two guide lines that make the separator read at a glance: clean at exactly 1.0x, and the
    jawari as a SINGLE number (it is position-independent — that is the whole contrast). The jawari
    reference must genuinely buzz (> 1x), i.e. the curved bridge is in reach at these params."""
    ref = _jua()["meta"]["spectrum"]["reference"]
    assert ref["clean"] == 1.0
    assert ref["jawari"] is not None and ref["jawari"] > 1.0


def test_juari_tuning_curve_is_decoupled_from_the_audio_length():
    """The load-bearing cost/UX decision: the sweep runs at a CANONICAL fixed duration, so
    lengthening the sound does not shift the tuning-curve map (nor make it more expensive). Two
    renders differing ONLY in audio_duration return the identical curve."""
    a = _jua(audio_duration=0.06)["meta"]["spectrum"]["tuning"]["elevation"]
    b = _jua(audio_duration=0.1)["meta"]["spectrum"]["tuning"]["elevation"]
    assert a == b


def test_juari_ignores_the_mallet_alpha_and_the_sympathetic_K():
    """Param-collision guard (the jawari's lesson). The frontend sends every slider, so a user who
    visited the mallet or sympathetic strings would otherwise leak alpha = 2.3 / K = 8000. The juari
    reads neither; it uses bridge_stiffness (fixed alpha 1.5)."""
    base = _jua()["meta"]["spectrum"]["tuning"]["elevation"]
    leaked = _jua(alpha=2.3, K=8000)["meta"]["spectrum"]["tuning"]["elevation"]
    assert leaked == base
    changed = _jua(bridge_stiffness=2.0e5)["meta"]["spectrum"]["tuning"]["elevation"]
    assert changed != base


def test_juari_audio_is_real_and_finite():
    a = _jua()["audio"]
    assert a is not None and a["fs"] == AUDIO_FS
    sig = _decode_f32(a["b64"])
    assert np.all(np.isfinite(sig)) and float(np.max(np.abs(sig))) > 0.0


def test_juari_below_signal_is_labelled_not_failed():
    """A thread at mid-string is a legitimate position that simply buzzes little (the even partials
    fall on its node). It must render with an honest 'weak' label — never an error, never a silent
    buzzing badge (the bow/jawari label-not-fail rule)."""
    d = _jua(thread_position=0.5)
    assert "error" not in d
    sp = d["meta"]["spectrum"]
    assert sp["buzzing"] == (sp["elevation"] > JUARI_ELEVATION_GATE)
    assert sp["elevation"] is not None


def test_juari_guards_are_clean_error_payloads():
    """Every out-of-range control comes back as a labelled error payload, not a 500 or a NaN."""
    assert "error" in _jua(thread_position=1.4)
    assert "error" in _jua(thread_position=0.0)
    assert "error" in _jua(N=JUARI_N_MAX + 1)
    assert "error" in _jua(**{"lambda": 1.5})
    assert "error" in _jua(amplitude=JUARI_AMP_MAX + 1e-3)
    assert "error" in _jua(bridge_stiffness=0.0)
    assert "error" in _jua(sweep_duration=JUARI_SWEEP_DUR + 0.1)
    assert "error" in _jua(N=128, sweep_duration=JUARI_SWEEP_DUR, audio_duration=JUARI_WORK_MAX)


def test_juari_sweet_spot_sits_near_the_nut_settled():
    """The one claim that needs the SETTLED buzz (slower): the sweet spot is near the nut, where a
    real tanpura thread lives — NOT mid-string. Measured at the canonical sweep duration; the exact
    node wobbles ~2 with the window (it is at the grid-resolution limit), so only the near-nut
    REGION is pinned, not a single node."""
    sp = _jua(N=40, sweep_duration=0.16, audio_duration=0.06)["meta"]["spectrum"]
    assert sp["sweet_spot"]["frac"] <= 0.25, "the peak buzz must sit near the nut"
    # The magnitude oracle (the sim is deterministic): the settled sweet-spot buzz is ~3.1x and the
    # curved-jawari reference ~3.5x at these params, so a well-placed thread genuinely rivals the
    # whole bridge. Bracketed, not point-pinned, to tolerate platform FFT noise while still catching
    # a real physics regression — a halved buzz (~1.5x) or a dead reference would break this, where
    # "elevation > gate" alone would keep a green BUZZING badge on broken physics.
    assert 2.7 < sp["sweet_spot"]["elevation"] < 3.5, sp["sweet_spot"]["elevation"]
    assert 3.0 < sp["reference"]["jawari"] < 3.9, sp["reference"]["jawari"]


# =================================================================================================
# Acoustic bore + radiating bell (the wind leg) — the claim is that the loss is BOOKED
# =================================================================================================
#
# Every other lossy model in the viewer gives up the conservation verdict. This one does not: the
# bell's shed energy is accumulated into ``Bore.energy()``, so sigma = 0 with a *radiating* end
# still conserves while up to 100 % of E0 leaves the tube. The discriminating tests below are the
# ones that would pass on a *broken* wiring if written loosely — a total-only drift check passes
# even if radiation is silently disabled, so the split is asserted to actually move.


def _bore(**over):
    return web_serialize.simulate_to_payload({"model": "bore", "audio_duration": 0.25, **over})


def test_bore_conserves_with_the_bell_radiating_and_the_split_actually_moves():
    """THE headline, and it needs both halves. Total drift < 1e-10 is necessary but NOT sufficient:
    it would also pass with the bell wired to shed nothing at all. So the split must show acoustic
    energy falling, radiated rising, and the two summing flat — that is the picture the panel makes
    and the only assertion that distinguishes "booked" from "absent"."""
    d = _bore(bell_ratio_exp=0.0)          # anechoic: the whole pulse leaves
    e = d["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True

    ac = np.array(e["split"]["acoustic"], dtype=float)
    rad = np.array(e["split"]["radiated"], dtype=float)
    assert ac[-1] < 0.05 * ac[0], "the matched bell must drain the tube"
    assert rad[-1] > 0.95 * ac[0], "and the drained energy must be BOOKED, not lost"
    assert np.all(np.diff(rad) >= -1e-18), "radiated energy is cumulative — never decreases"
    total = ac + rad
    assert np.max(np.abs(total - total[0])) / total[0] < LOSSLESS_TOL
    assert d["meta"]["radiated_frac"] == pytest.approx(1.0, abs=1e-6)


def test_bore_a_lightly_radiating_clarinet_sheds_a_little_and_still_conserves():
    """The physical bell (R/Z0 ~ 3e-4), which is the default: a slow leak, not a drain. The verdict
    is unchanged — which is the point. A model that could only conserve when nothing radiated would
    be proving something much weaker."""
    d = _bore()
    assert d["energy"]["lossless"]["pass"] is True
    assert 0.0 < d["meta"]["radiated_frac"] < 0.5
    assert d["meta"]["radiating"] is True and d["meta"]["ends"] == ["closed", "radiating"]


def test_bore_reflection_matches_the_closed_form_including_the_anechoic_null():
    """The money oracle: one bounce off the bell sheds 1/2 (1 - r^2) with r = (R - Z0)/(R + Z0).
    Measured to ~1e-16, and at the matched R = Z0 the reflection vanishes and exactly HALF the
    energy (the entire right-going half of a centred pulse) is absorbed."""
    rb = _bore()["meta"]["reflection"]
    assert rb["radiating"] is True and rb["pass"] is True
    assert rb["abs_error"] < 1e-12
    assert rb["measured"] == pytest.approx(rb["oracle"], abs=1e-12)

    an = _bore(bell_ratio_exp=0.0)["meta"]["reflection"]
    assert an["anechoic"] is True
    assert an["r"] == pytest.approx(0.0, abs=1e-12)
    assert an["measured"] == pytest.approx(0.5, abs=1e-12)
    # the analytic curve is free (no simulation) and must contain the null EXACTLY, not straddle it
    shed = np.array(an["curve"]["shed"], dtype=float)
    assert float(shed.max()) == pytest.approx(0.5, abs=1e-15)


def test_bore_r_over_z0_is_the_control_and_it_monotonically_buys_loss():
    """R/Z0 is the units- and geometry-invariant coordinate (Z0 moves with the bore radius, so a
    raw-R slider would mean something different at every geometry). Turning it up toward the match
    must monotonically increase what the bell sheds."""
    fracs = [_bore(bell_ratio_exp=x)["meta"]["radiated_frac"] for x in (-3.5, -2.0, -1.0, 0.0)]
    assert fracs == sorted(fracs)
    assert fracs[0] < 0.5 < fracs[-1]


def test_bore_ideal_open_end_is_the_lossless_contrast_with_no_bell_to_score():
    """The 'open' regime is pressure-release (r = -1): a perfect mirror, nothing radiates. It is
    the conservation baseline, and the reflection panel must say there is nothing to measure rather
    than report a meaningless number — and R/Z0 must not echo a live-looking value at a control
    that is doing nothing."""
    d = _bore(domain="open")
    assert d["energy"]["lossless"]["pass"] is True
    assert d["meta"]["radiated_frac"] == pytest.approx(0.0, abs=1e-15)
    assert d["meta"]["radiating"] is False and d["meta"]["r_ratio"] is None
    assert d["meta"]["ends"] == ["closed", "open"]
    assert d["meta"]["reflection"]["radiating"] is False
    assert "note" in d["meta"]["reflection"]


def test_bore_is_a_clarinet_odd_harmonics_only_at_the_shortest_allowed_render():
    """The odd/even ratio is set by the FFT WINDOW, not by N or by physics (2.29e5 at 0.5 s but
    3.6e4 at 0.25 s, flat in N at each), so the gate is checked at the SHORTEST allowed duration —
    gating on the long-window number would fail a legitimately-correct short render."""
    sp = _bore(audio_duration=0.25)["meta"]["spectrum"]
    assert sp["applies"] is True
    assert sp["odd_even"]["pass"] is True
    assert sp["odd_even"]["ratio"] > BORE_ODD_EVEN_GATE
    assert sp["f1"] == pytest.approx(343.0 / (4.0 * 0.5), rel=1e-3)


def test_bore_partials_land_on_the_eigenvalue_oracle_which_is_exact_at_lambda_one():
    """Two oracles. The EIGENVALUE frequencies owe nothing to any FFT window and are what
    structurally certifies the half-cell wall: at lambda = 1 they sit on the continuum to 0.0000
    cents at every N. The measured spectrum must be read through the parabolic refinement — a bin
    peak-pick invents a bogus N-dependence (1.69 cents at N = 100/200, 0.00 at N = 64/128, which is
    not physics but which bin the fundamental landed on)."""
    for N in (64, 128, 200):
        pa = _bore(N=N)["meta"]["spectrum"]["partials"]
        assert np.max(np.abs(pa["eigen_vs_continuum"])) < 1e-3, "lambda = 1 is dispersionless"
        assert np.max(np.abs(pa["cents_vs_eigen"])) < 0.05


def test_bore_heavily_absorbing_bell_is_labelled_not_failed():
    """At the anechoic match there is no standing wave, so the odd-harmonic and partial claims stop
    applying — there is nothing to measure, which is not the same as being wrong. It must render
    with an honest label, never an error and never a silent pass (the jawari's grazing
    precedent)."""
    d = _bore(bell_ratio_exp=0.0)
    assert "error" not in d
    sp = d["meta"]["spectrum"]
    assert sp["applies"] is False
    assert sp["odd_even"]["pass"] is False, "and the gate is honest about it rather than hidden"


def test_bore_dispersion_is_an_eigenvalue_computation_showing_second_order_departure():
    """The lambda claim is about the OPERATOR, so it is computed from the operator — no
    time-stepping, hence no cost, hence no lambda slider on the render (where steps scale as
    1/lambda and the budget dies at lambda = 0.878). The departure falls 4x across a 2x refinement
    (exactly O(h^2)) and collapses to zero at lambda = 1."""
    dp = _bore()["meta"]["dispersion"]
    assert len(dp["lambda"]) == len(dp["coarse"]) == len(dp["fine"]) == len(dp["order"])
    assert dp["lambda"][-1] == 1.0
    assert dp["coarse"][-1] < 1e-3 and dp["fine"][-1] < 1e-3, "dispersionless at lambda = 1"
    assert dp["coarse"][0] > 1.0, "and visibly off it at lambda = 0.5"
    assert dp["order"][0] == pytest.approx(4.0, abs=0.1)


def test_bore_animation_is_paced_on_the_transit_not_the_fundamental():
    """THE TRAP. One transit is L/c0 = 1.46 ms but f1 = c0/4L is 5.83 ms — four transits. Pacing on
    f1 the way every string model does gives ~3 frames per transit and aliases the bounce-and-flip
    picture into noise, which playback_speed cannot rescue because the frames are already decimated
    in sim time. f_ref = c0/L keeps it at ~12 frames/transit, flat in N."""
    for N in (64, 128, 256):
        m = _bore(N=N)["meta"]
        assert 10.0 <= m["frames_per_transit"] <= 14.0
        assert m["transit"] == pytest.approx(0.5 / 343.0, rel=1e-4)  # rounded for display


def test_bore_frame_and_grid_bookkeeping_line_up():
    """1D frame width == the pressure grid, one radiated sample per frame (the mouth glow), and the
    envelope over the whole run on the same grid — a stride mismatch here would paint the glow and
    the envelope against the wrong nodes."""
    d = _bore(N=64)
    x = d["grid"]["x"]
    assert d["frames"]["dims"] == 1
    assert d["frames"]["width"] == len(x) == 65
    assert len(d["grid"]["envelope"]) == len(x)
    assert d["frames"]["n_frames"] == len(d["frame_times"]) == len(d["radiated_frames"])
    frames = np.frombuffer(base64.b64decode(d["frames"]["b64"]), dtype="<f4")
    assert frames.size == d["frames"]["n_frames"] * d["frames"]["width"]
    assert np.all(np.isfinite(frames))
    # the closed end is a pressure ANTINODE (p free and large) and the open end a NODE — that
    # asymmetry IS the odd-harmonic claim, so a viz that pinned both ends would draw it backwards
    assert np.array(d["grid"]["envelope"], dtype=float)[0] > 0.0
    open_env = np.array(_bore(N=64, domain="open")["grid"]["envelope"], dtype=float)
    assert open_env[-1] == pytest.approx(0.0)


def test_bore_radiated_frames_are_cumulative_and_normalized_for_the_mouth_glow():
    """The field-side dual of the energy panel: the glow tracks the BOOKED radiated energy, so it
    is monotone and expressed as a fraction of E0 — the frontend needs no units."""
    rf = np.array(_bore(bell_ratio_exp=0.0)["radiated_frames"], dtype=float)
    assert rf[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(rf) >= -1e-15)
    assert 0.0 < rf[-1] <= 1.0


def test_bore_audio_is_real_finite_and_normalized():
    a = _bore()["audio"]
    assert a["fs"] == AUDIO_FS and a["peak"] > 0.0
    sig = np.frombuffer(base64.b64decode(a["b64"]), dtype="<f4")
    assert sig.size == a["n"] and np.all(np.isfinite(sig))
    assert float(np.max(np.abs(sig))) <= 1.0 + 1e-6


def test_bore_work_budget_counts_the_reflection_run_and_the_guards_are_reachable():
    """N buys the sample rate, not just the grid: at lambda = 1, fs = c0*N/L, so CPU per second of
    audio scales as N^2. The reflection run is a SECOND initial condition and is counted, even
    though at ~N steps it is negligible — the jawari's rule holds regardless of size."""
    assert "error" in _bore(N=BORE_N_MAX, audio_duration=BORE_AUDIO_MAX + 0.5)
    assert "error" in _bore(N=BORE_N_MAX + 1)
    assert "error" in _bore(N=BORE_N_MIN - 1)
    assert "error" in _bore(audio_duration=BORE_AUDIO_MAX + 0.1)
    assert "error" in _bore(domain="reed")
    assert "error" in _bore(bell_ratio_exp=3.0)
    assert "error" in _bore(pickup_position=1.0)


def test_bore_animation_window_has_its_own_cap_not_the_shared_one():
    """The shared ANIM_WIN_MAX = 2.0 is a COST HOLE the bore's budget does not cover: at N = 256 a
    2 s window is 351k animation steps, over BORE_WORK_MAX on its own, and the MAX_FRAMES re-stride
    does not save it — it caps the frames emitted, not the steps simulated. A frame-count ceiling
    is not a cost ceiling."""
    assert BORE_ANIM_MAX < web_serialize.ANIM_WIN_MAX
    assert "error" in _bore(animation_window=BORE_ANIM_MAX + 0.01)
    assert "error" not in _bore(animation_window=BORE_ANIM_MAX)


def test_bore_ignores_params_that_belong_to_other_models():
    """Every model gets the whole slider panel. The bore must not read the string family's sigma0
    (its own sigma is deliberately unexposed and pinned at 0), nor kappa, nor the jawari's depth —
    a silently-honoured stale param would change the physics with nothing on screen to say so."""
    base = _bore()
    other = _bore(sigma0=0.9, sigma1=0.05, kappa=4.0, depth=0.002, T=500)
    assert other["energy"]["sigma_is_zero"] is True
    assert other["fs_sim"] == base["fs_sim"]
    assert other["meta"]["radiated_frac"] == pytest.approx(base["meta"]["radiated_frac"])


# =================================================================================================
# The dynamic single reed (wind, batch 10) — the claim is that the balance is MEASURED, and speaks
# =================================================================================================
#
# The reed is the acoustic dual of the bow, but its energy balance is a *stronger* claim: the bow's
# dissipation is INFERRED as ``work - dE``, so its lossy residual is identically zero by
# construction. The reed measures ``jet_loss`` and ``reed_damp_work`` independently and books the
# bell's radiation into ``energy()``, so its residual can genuinely fail. The tests below are
# written to be the ones that would catch a *broken* wiring: a residual on a sum cannot see a dead
# summand, and a balance alone passes on silence — so the channels and the speaking are asserted
# separately from the residual.


def _reed(**over):
    return web_serialize.simulate_to_payload(
        {"model": "reed", "audio_duration": 0.3, "N": 96, **over})


def test_reed_balance_is_a_measured_residual_and_the_channels_are_load_bearing():
    """The batch's structural claim. The residual closes to ~1e-14 with every channel accounted —
    and the SAME run with the measured loss dropped (the bow's ``dE - work``) is enormous. That
    contrast is the evidence the channels are load-bearing: a residual that stayed tiny with a
    summand removed would not be testing the summand at all. Batch 9's "a sum cannot see a dead
    summand", aimed here at the residual itself."""
    e = _reed()["energy"]
    assert e["kind"] == "balance"
    b = e["balance"]["measured"]
    assert b["residual"] < BOW_BALANCE_TOL, b
    assert b["pass"] is True
    # measured loss dropped => the jet + damping fraction reappears, ~60-95 % of scale
    assert b["naive_residual"] > 0.5, b


def test_reed_every_balance_channel_is_non_trivially_populated():
    """A dead or wrong-sign reed rings down and reports residual ~= 0 on silence — reed-state's
    "balance is necessary but NOT sufficient". So each channel is asserted to actually carry
    energy, not merely to sum correctly."""
    m = _reed()["meta"]
    bud = m["budget"]
    assert bud["mouth_work"] > 0.0
    assert bud["jet_frac"] > 0.1          # Bernoulli dissipation is the dominant sink (~0.75)
    assert bud["damping_frac"] > 0.0      # lip damping
    assert bud["stored_frac"] > 0.0       # something is actually ringing in the tube
    tot = bud["jet_frac"] + bud["damping_frac"] + bud["radiated_frac"] + bud["stored_frac"]
    assert tot == pytest.approx(1.0, abs=2e-3), bud


def test_reed_balance_has_no_sigma_gate_and_that_is_not_an_oversight():
    """The trap this batch had to avoid. The bow's sigma = 0 branch scores ``max|dE - work|``,
    valid only because a lossless string has ``dE == work``. The reed's jet and lip-damping
    channels are on in EVERY regime, so routing it through that branch reports a catastrophic
    IMBALANCE on a perfectly balanced model. The measured branch must therefore replace both the
    lossless and lossy branches, not sit beside them."""
    for domain in ("radiating", "open"):
        b = _reed(domain=domain)["energy"]["balance"]
        assert "measured" in b
        assert "lossless" not in b and "lossy" not in b
        assert b["measured"]["residual"] < BOW_BALANCE_TOL


def test_reed_closes_the_balance_with_the_bell_radiating():
    """Batch 9's booked-radiation property carried onto a driven model: the bell sheds real energy
    and the book still closes, because ``energy()`` accumulates it. The radiating run must actually
    radiate (else this passes on a dead bell) while the open run must not radiate at all."""
    rad = _reed(domain="radiating")["meta"]
    op = _reed(domain="open")["meta"]
    assert rad["budget"]["radiated_frac"] > 0.02, rad["budget"]
    assert op["budget"]["radiated_frac"] == 0.0
    assert rad["r_ratio"] is not None and op["r_ratio"] is None


def test_reed_actually_speaks_above_threshold_and_is_silent_below():
    """The sufficiency gate. Channels-populated catches a *dead* reed; it does not catch a
    wrong-sign one that rings down with populated channels. Only the signature does — so this is an
    assertion, not a reported number."""
    loud = _reed(gamma=0.51)["meta"]
    quiet = _reed(gamma=0.20)["meta"]
    assert loud["speaks"] is True
    assert loud["ac_level"] > 10.0 * REED_SPEAK_GATE
    assert quiet["speaks"] is False
    assert quiet["ac_level"] < REED_SPEAK_GATE
    # ...and below threshold the reed never slams shut, while above it does.
    assert loud["beating"]["beats"] is True
    assert quiet["beating"]["beats"] is False


def test_reed_blowing_threshold_brackets_one_third():
    """Dalmont/Kergomard's small-oscillation rule. The level curve is what carries the claim — it
    jumps three orders of magnitude across the onset — and the derived bracket must straddle ~1/3.
    Deliberately NOT asserted at gamma = 1/3 itself: that point sits exactly ON the threshold
    (measured 0.0196 against a 0.02 gate at the default bell), so its label is a knife edge by
    construction. Assert the curve's shape, never the knife edge."""
    sw = _reed()["meta"]["sweep"]
    lo, hi = sw["bracket"]
    assert lo is not None and hi is not None
    assert lo <= 0.36 and hi >= 0.30, sw["bracket"]
    assert min(sw["level"]) < 0.01 < max(sw["level"])
    assert max(sw["level"]) / max(min(sw["level"]), 1e-12) > 100.0
    # monotone in gamma: blowing harder never makes the note quieter
    assert all(b >= a - 1e-9
               for a, b in zip(sw["level"][:-1], sw["level"][1:], strict=True))
    assert 1.0 / 3.0 in sw["gamma"]      # on the grid EXACTLY (batch 9's anechoic-null lesson)


def test_reed_sweep_is_pinned_off_the_render_grid():
    """The sweep is a property of (bore, reed, gamma), not of the display grid — measured
    N-invariant to the 4th significant digit. Pinning it at REED_SWEEP_N both bounds its cost and,
    the correctness half, stops the headline threshold moving when the user drags N."""
    a = _reed(N=64)["meta"]["sweep"]
    b = _reed(N=200, audio_duration=0.2)["meta"]["sweep"]
    assert a["sweep_N"] == REED_SWEEP_N
    assert a["bracket"] == b["bracket"]
    assert a["level"] == b["level"]


def test_reed_sweep_memo_key_carries_everything_that_moves_p_closing():
    """THE CACHE TRAP. The threshold lives in gamma, but ``p_closing = mu wr^2 H0`` — so a memo key
    missing ``f_reed`` (which IS an exposed slider) returns stale numbers the moment the user drags
    it: green on the defaults, wrong on interaction. Same for the bell, which genuinely moves the
    bracket (a lossier bell needs a harder blow)."""
    base = _reed()["meta"]["sweep"]
    assert _reed(f_reed=1500.0)["meta"]["sweep"]["level"] != base["level"]
    assert _reed(q_reed=8.0)["meta"]["sweep"]["level"] != base["level"]
    assert _reed(bell_ratio_exp=-1.2)["meta"]["sweep"]["level"] != base["level"]
    assert _reed(L=0.7)["meta"]["sweep"]["level"] != base["level"]


def test_reed_pitch_is_set_by_the_air_column_not_the_reed():
    """Stated as LEVERAGE rather than the suite's binary "< 6 %": a +50 % sweep of the reed buys
    only ~+3 % of pitch. The residual trend is itself physics — the reed's compliance acts as an
    end correction, so a stiffer reed lands CLOSER to c/4L (less negative cents)."""
    pit = _reed()["meta"]["sweep"]["pitch"]
    assert pit["reed_change_pct"] == pytest.approx(50.0)
    assert abs(pit["pitch_change_pct"]) < 6.0
    assert pit["pitch_change_pct"] > 0.0
    assert pit["cents"][1] > pit["cents"][0]      # stiffer reed -> closer to c/4L
    assert all(c < 0.0 for c in pit["cents"])     # reed compliance always flattens


def test_reed_is_a_clarinet_odd_harmonics_dominate():
    sp = _reed()["meta"]["spectrum"]
    assert sp["applies"] is True
    assert sp["odd_even"] > 100.0                 # fundamental dwarfs the 2nd harmonic
    assert sp["third_second"] > 1.0               # 3rd (odd) beats the 2nd (even)
    assert sp["crest"] < 1.5                      # ~1.0 is a perfect square wave


def test_reed_beating_is_debounced_to_one_slam_per_period():
    """A raw per-period closure count measures the CHATTER, not the event: each period holds a
    short precursor, a brief re-opening, then the main closure, so zero crossings report ~1.94 for
    what is plainly one beat. The duty needs no event definition and is the primary number."""
    bt = _reed(gamma=0.51)["meta"]["beating"]
    assert bt["beats"] is True
    assert bt["per_period"] == pytest.approx(1.0, abs=0.15), bt
    assert 0.2 < bt["duty"] < 0.6
    assert bt["min_opening"] == 0.0


def test_reed_ships_the_far_field_caveat_beside_the_mouthpiece_audio():
    """The mouthpiece IS the square wave and is the audio, but it is emphatically not what a
    listener hears — radiation differentiates. The far field is spikier and orders quieter, and
    those numbers ship so that "the iconic clarinet tone" cannot quietly mean the pressure inside
    the mouthpiece."""
    sp = _reed()["meta"]["spectrum"]
    ff = sp["far_field"]
    assert ff["quieter_by"] > 20.0
    assert ff["crest"] > sp["crest"]
    # ...and on an ideal open end it is WITHDRAWN, not faked. Nothing radiates there, so the
    # comparison divides by ~0: the first cut shipped "1.6e+33x quieter (crest null)" and every
    # backend test passed — only the rendered readout showed it. Batch 9's `applies = false`
    # pattern: nothing to measure is not the same as wrong.
    op = _reed(domain="open")["meta"]["spectrum"]
    assert op["applies"] is True          # the note still speaks; only the far field is absent
    assert op["far_field"] is None
    assert "far_note" in op


def test_reed_below_threshold_withdraws_the_spectrum_claims():
    """An honesty gate that fires and is LABELLED, never failed (batch 9's ``applies = false``):
    below threshold there is no tone, so there is nothing to measure — which is not the same as
    being wrong."""
    sp = _reed(gamma=0.20)["meta"]["spectrum"]
    assert sp["applies"] is False
    assert "odd_even" not in sp
    assert "note" in sp


def test_reed_announces_its_mouth_end_for_the_viz_without_touching_the_bore_boundary():
    """``meta.ends[0] = "reed"`` is a VIZ label set by the payload — the core still requires the
    bore's left end to be literally "closed" (the reed rides that half-cell DOF). Batch 9 built the
    ``meta.ends`` switch with two cases precisely so this would be an addition, not a rewrite."""
    assert _reed()["meta"]["ends"] == ["reed", "radiating"]
    assert _reed(domain="open")["meta"]["ends"] == ["reed", "open"]


def test_reed_animation_is_paced_on_the_transit_and_captured_from_the_settled_tail():
    """Two inherited lessons in one payload. Batch 9: pace on the transit, not f1 (f1 pacing gives
    ~3 frames/transit and aliases the travelling pressure step, this batch's best picture). Batch 2:
    the window is the TAIL, because the reed starts from rest and the opening frames are silence —
    and it comes out of the audio run, never a second resonator."""
    d = _reed()
    assert 8.0 < d["meta"]["frames_per_transit"] < 20.0
    t = d["frame_times"]
    # the frames end at the end of the run and start well into it: a tail, not the attack
    assert t[-1] == pytest.approx(d["meta"]["num_steps"] / d["fs_sim"], rel=1e-3)
    assert t[0] > 0.5 * t[-1]


def test_reed_guards_reject_out_of_range_configurations_cleanly():
    """The render budget bounds n_anim + n_audio; the sweeps are a separate fixed cost, so neither
    cap can launder the other. The animation window has its own ceiling for batch 9's reason — the
    shared 2 s is a cost hole this budget cannot cover."""
    assert "error" in _reed(N=REED_N_MAX, audio_duration=REED_AUDIO_MAX + 0.5)
    assert "error" in _reed(N=REED_N_MAX + 1)
    assert "error" in _reed(N=REED_N_MIN - 1)
    assert "error" in _reed(audio_duration=REED_AUDIO_MAX + 0.1)
    assert "error" in _reed(animation_window=REED_ANIM_MAX + 0.01)
    assert "error" in _reed(gamma=0.0)
    assert "error" in _reed(gamma=2.0)
    assert "error" in _reed(f_reed=200.0)
    assert "error" in _reed(domain="bell")
    assert REED_ANIM_MAX < web_serialize.ANIM_WIN_MAX
    assert REED_WORK_MAX > 0


def test_reed_ignores_params_that_belong_to_other_models():
    """Every model gets the whole slider panel. The reed must not read the string family's sigma0
    (bore viscous sigma is deliberately unexposed and pinned at 0 — batch 9's least-booked-channel
    rule), nor the jawari's depth, nor the mallet's alpha."""
    base = _reed()
    other = _reed(sigma0=0.9, sigma1=0.05, kappa=4.0, depth=0.002, alpha=1.5, K=8000)
    assert other["fs_sim"] == base["fs_sim"]
    assert other["meta"]["budget"]["jet_frac"] == pytest.approx(base["meta"]["budget"]["jet_frac"])
    assert other["energy"]["balance"]["measured"]["residual"] < BOW_BALANCE_TOL


# =================================================================================================
# Fret / flat rail (model #8 on its OWN terms) — the claim is INTERMITTENCY, not the energy
# =================================================================================================
#
# The jawari's physical opposite: not a departure point gliding persistently along a curve, but
# slap-and-release. The viewer runs the geometry tests/test_collision_signature.py validates (flat
# rail, lam = 0.4, mode-1 pluck), so the probe's committed numbers come back through the payload and
# act as a free end-to-end oracle. Everything on screen except the intermittency is diagnostic: the
# decay is a consistency TRIPLE (the jawari's 2*sigma0 oracle does NOT transfer), the brightness is
# non-monotone in its own slider, and the crossing rate is deliberately not called pitch.


def _fret(**over):
    return web_serialize.simulate_to_payload({"model": "fret", "audio_duration": 0.4, **over})


def test_fret_reproduces_the_probes_intermittency_numbers():
    """The end-to-end oracle: the payload's headline numbers ARE the probe's. The string slaps the
    rail ~1.24x per period at a ~15.4 % duty and springs off — it is never pinned. If the wrapper
    perturbed the rig (a different IC, a stale param, a shifted window) these would drift."""
    ct = _fret()["meta"]["contact"]
    assert ct["kind"] == "fret"
    assert ct["duty"] == pytest.approx(0.154, abs=0.005)
    assert ct["episodes_per_period"] == pytest.approx(1.24, abs=0.05)
    assert ct["intermittent"] is True
    assert ct["out_of_reach"] is False and ct["pinned"] is False


def test_fret_active_set_is_a_vector_and_the_newton_is_cheap():
    """The second headline, and the reason this is model #8 proper rather than the mallet's scalar
    collapse: of 99 support nodes up to 69 are simultaneously in contact. The Newton costs max 2 /
    mean ~1.16 iterations — that cheapness IS the lambda_min(J) >= 1 proof (unique root, global
    convergence, no branch-picking) showing up as a measurement."""
    ct = _fret()["meta"]["contact"]
    assert ct["support"] == 99
    assert ct["active_max"] == pytest.approx(69, abs=3)
    assert ct["active_max"] > 1, "a scalar active set would be the mallet, not a vector Newton"
    assert ct["iters_max"] <= 3
    assert ct["iters_mean"] == pytest.approx(1.16, abs=0.05)


def test_fret_sigma0_gates_the_verdict_and_conserves_through_genuine_contact():
    """sigma0 = 0 flips the panel to the conservation-drift check, and the drift survives real
    many-node contact (~7e-13) — not a contact-free run that would prove nothing about the rail."""
    d = _fret(sigma0=0.0)
    e = d["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True
    assert d["meta"]["contact"]["duty"] > 0.1, "must actually be in contact"


def test_fret_drops_the_jawari_decay_oracle_and_ships_the_triple_instead():
    """THE batch's load-bearing verdict decision, and the one place the two configurations of model
    #8 disagree. The rail is *equally* lossless (see the sigma = 0 drift above) yet decays 6-9 %
    fast, because "rate == 2*sigma0" silently assumes EQUIPARTITION (<KE> = E/2) — true for harmonic
    motion, false for a string slapped by a stiff one-sided spring. So the flat-loss oracle line is
    dropped (it would print a mismatch it cannot explain) and a diagnostic triple replaces it."""
    e = _fret(sigma0=0.5)["energy"]
    assert e["sigma_is_zero"] is False
    assert e["lossy"]["monotone"] is True
    assert "measured_2sigma" not in e["lossy"], "the jawari's oracle must NOT be inherited"
    assert "oracle_2sigma" not in e["lossy"]
    t = e["decay_triple"]
    assert t["oracle_2sigma"] == pytest.approx(1.0)
    # the naive oracle is measurably WRONG here — that is the whole point of dropping it
    assert t["ratio"] > 1.03
    # ...and removing the equipartition assumption recovers the measured rate to well under 1 %
    assert t["agreement"] < 0.01
    assert t["corrected"] == pytest.approx(t["rate"], rel=0.01)


def test_fret_equipartition_correction_tracks_the_rate_across_clearances():
    """The mechanism, not one lucky point: as the rail is brought closer the decay runs further
    ahead of 2*sigma0, and the <2KE/E> correction follows it every time. This is what licenses
    REPORTING the triple — and its being one identity measured twice is why it is not GATED."""
    ratios = []
    for clearance in (4.0e-3, 2.0e-3, 1.0e-3):
        t = _fret(audio_duration=0.25, clearance=clearance)["energy"]["decay_triple"]
        assert t["agreement"] < 0.01
        ratios.append(t["ratio"])
    assert ratios[0] < ratios[-1], "a closer rail should break equipartition harder"
    assert ratios[0] > 1.0


def test_fret_out_of_reach_rail_is_labelled_not_failed():
    """A rail the string never reaches is a correct render, not a broken one (the bow's Schelleng /
    jawari's grazing rule, fourth customer). It must come back as a labelled payload with an empty
    claim — never an error, and never a green intermittency badge over a blank raster."""
    ct = _fret(clearance=6.0e-3)["meta"]["contact"]
    assert ct["out_of_reach"] is True
    assert ct["intermittent"] is False
    assert ct["duty"] == 0.0 and ct["episodes"] == 0 and ct["active_max"] == 0


def test_fret_rail_frac_floor_is_enforced_server_side():
    """rail_frac is a slider that can silently kill the claim: peak swing over the rail falls as the
    rail shortens, so below ~0.15 nothing touches it at all. The floor is enforced HERE and not
    merely as a slider min, because the frontend sends whatever it likes."""
    assert "error" in _fret(rail_frac=FRET_RAIL_FRAC_MIN - 0.05)
    ct = _fret(rail_frac=FRET_RAIL_FRAC_MIN)["meta"]["contact"]
    assert ct["duty"] > 0.0 and ct["intermittent"] is True, "the floor must still reach the rail"


def test_fret_intermittency_is_structural_not_a_tuned_accident():
    """The duty is bounded well away from "pinned" across the WHOLE rail_stiffness range, and for a
    structural reason: a lossless one-sided spring always pushes back, so the string can never come
    to REST on the rail. Softening the rail raises the duty monotonically but it asymptotes just
    under 0.5 — the FREE-SINUSOID limit (a string crossing the rail line untouched), i.e. the
    no-rail limit, NOT pinning. This is why `pinned` is a guarantee rather than a live label."""
    duties = []
    for stiffness in (2.0e6, 2.0e4, 2.0e2):
        ct = _fret(audio_duration=0.08, rail_stiffness=stiffness,
                   clearance=5.0e-4)["meta"]["contact"]
        assert ct["intermittent"] is True and ct["pinned"] is False
        duties.append(ct["duty"])
    assert duties[0] < duties[-1], "a softer rail admits the string for longer"
    assert max(duties) < 0.5 < FRET_DUTY_MAX


def test_fret_scalars_are_computed_at_full_rate_not_read_off_the_raster():
    """An OR-reduced raster is an honest map of WHERE and WHEN and a BIASED estimator of HOW MUCH: a
    column is lit if *any* step in it touched, so the image is a dilation in time and its apparent
    duty overstates the true one at every finite resolution (28.8/21.4/18.4 % at 400/800/1600
    columns against a true 15.5 %). So the duty is computed from the signal, and here it is checked
    to be strictly BELOW what the picture would suggest."""
    ct = _fret()["meta"]["contact"]
    img = _decode_u8(ct["raster"]["b64"]).reshape(ct["raster"]["n_rows"], ct["raster"]["n_cols"])
    column_duty = float((img > 0).any(axis=0).mean())
    assert ct["duty"] < column_duty, "the raster dilates in time; the scalar must not inherit that"
    assert ct["duty"] == pytest.approx(0.154, abs=0.005)


def test_fret_raster_resolves_at_least_ten_columns_per_period():
    """Below ~10 columns/period the debounce window rounds to under one column and the episode count
    degenerates to raw onsets and FRAGMENTS the slaps (83 and 60 apparent episodes at 200 and 100
    columns against a truth of 49). A too-coarse raster looks BUSIER, not emptier — the more
    dangerous failure — so the column count tracks the duration rather than being hardcoded."""
    for duration in (0.1, 0.4, 0.6):
        r = _fret(audio_duration=duration)["meta"]["contact"]["raster"]
        assert r["cols_per_period"] >= FRET_RASTER_COLS_PER_PERIOD - 0.5
        assert r["n_rows"] <= FRET_RASTER_MAX_ROWS


def test_fret_raster_decodes_to_the_grid_and_greys_without_losing_contacts():
    """Shape and framing are part of the contract (batch 8's indexing trap: `_b` and contact_mask()
    are over the SUPPORT, not the grid). The uint8 grey is a strict REFINEMENT of the binary mask —
    any genuine contact stays >= 1 so a faint touch can never round away to "no contact"."""
    d = _fret()
    ct = d["meta"]["contact"]
    r = ct["raster"]
    img = _decode_u8(r["b64"])
    assert img.size == r["n_rows"] * r["n_cols"]
    assert r["n_rows"] == ct["support"] and r["x_binned"] is False
    assert 0 < img.max() <= 255 and img.min() == 0
    assert r["force_max"] > 0.0
    # the rail is scattered back onto GRID coordinates, so it indexes like every other field
    assert len(d["grid"]["barrier"]) == len(d["grid"]["x"])
    assert r["x0"] > 0.0 and r["x1"] < d["grid"]["x"][-1]
    assert len(ct["trace"]["active"]) == r["n_cols"] == len(ct["trace"]["iters"])


def test_fret_brightness_is_reported_with_its_non_monotonicity_named():
    """The elevation reproduces the probe exactly (4.682x at the 0.05 L pickup) against an unusually
    clean baseline: a mode-1 pluck with no rail is a pure sinusoid whose centroid reads EXACTLY f1,
    so the whole elevation is harmonic content the rail added. But it PEAKS at an intermediate
    clearance and falls either side, so no monotone label may be shipped — the slider samples a
    range the two-point suite test never did, and would disprove it."""
    sp = _fret()["meta"]["spectrum"]
    assert sp["elevation"] == pytest.approx(4.682, abs=0.02)
    assert sp["centroid_control"] == pytest.approx(sp["f1"], rel=0.001)
    assert sp["monotone"] is False
    assert sp["peak_clearance"] == FRET_BRIGHTNESS_PEAK
    far = _fret(audio_duration=0.25, clearance=4.0e-3)["meta"]["spectrum"]["elevation"]
    peak = _fret(audio_duration=0.25, clearance=2.0e-3)["meta"]["spectrum"]["elevation"]
    near = _fret(audio_duration=0.25, clearance=1.0e-3)["meta"]["spectrum"]["elevation"]
    assert peak > far and peak > near, "brightness must peak at an INTERMEDIATE clearance"


def test_fret_signature_block_carries_its_dispatch_kind():
    """`drawDiagnostics` switches on `meta.spectrum.kind`, and every branch is an equality test —
    so a missing key is not a missing panel, it is the WRONG panel: the fret falls through to
    drawPartials, which reads per-partial arrays this model never ships. The contact block carries
    its own `kind` for the same reason; both are pinned so neither can be dropped silently."""
    payload = _fret()
    assert payload["meta"]["spectrum"]["kind"] == "fret"
    assert payload["meta"]["contact"]["kind"] == "fret"


def test_fret_crossing_rate_is_never_called_pitch():
    """It rises steeply and is real, but f = c/(2 L_eff) would OVERCLAIM: the suite's
    test_contact_is_intermittent is explicit that the string is NOT pinned to a shorter length. A
    buzz adds zero crossings *within* a period, so this mixes the fundamental with the rail's
    contribution — it ships under the name of the thing actually measured, and gates nothing."""
    sp = _fret()["meta"]["spectrum"]
    assert "pitch" not in sp and "cents" not in sp
    assert sp["crossing_is_pitch"] is False
    assert sp["crossing_rate_control"] == pytest.approx(sp["f1"], rel=0.01)
    assert sp["crossing_rate"] > sp["crossing_rate_control"]
    # the static-equilibrium oracle is CITED as the magnitude credential, never re-run as a headline
    assert sp["static_oracle"]["residual"] < 1e-13


def test_fret_control_window_is_independent_of_the_fret_window():
    """The control's centroid is window-INVARIANT (a rail-free mode-1 pluck is a pure sinusoid), so
    a short control is free — but only if the comparison does not truncate the FRET pickup to match
    it. The fret's centroid is NOT window-invariant (468 over 0.4 s vs 442 over 0.2 s), so coupling
    the two windows would silently report the short-window brightness under the full run's label."""
    long_run = _fret(audio_duration=0.4)["meta"]["spectrum"]
    assert long_run["centroid_control"] == pytest.approx(100.0, abs=0.1)
    # the control is capped well below the fret run, and the elevation still reads the probe's value
    assert long_run["elevation"] == pytest.approx(4.682, abs=0.02)
    short = _fret(audio_duration=0.15)["meta"]["spectrum"]
    assert short["centroid_control"] == pytest.approx(100.0, abs=0.2)
    assert short["centroid_fret"] < long_run["centroid_fret"], "the fret window is NOT invariant"


def test_fret_ignores_the_names_that_belong_to_other_models():
    """The leak family, sixth member. `K` is the sympathetic bridge spring, `alpha` the mallet's
    felt exponent, `bridge_stiffness`/`depth`/`width_frac` the jawari's — and the frontend sends
    slider, hidden ones included. Merely VISITING those models must not silently re-render a
    different fret with nothing on screen to say so."""
    base = _fret(audio_duration=0.08)
    other = _fret(audio_duration=0.08, alpha=2.3, K=8000, depth=1.0e-3, bridge_stiffness=1.0,
                  width_frac=0.4, kappa=4.0, sigma1=0.05)
    assert other["fs_sim"] == base["fs_sim"]
    assert other["meta"]["contact"]["duty"] == base["meta"]["contact"]["duty"]
    assert other["meta"]["contact"]["active_max"] == base["meta"]["contact"]["active_max"]
    assert other["meta"]["spectrum"]["elevation"] == base["meta"]["spectrum"]["elevation"]


def test_fret_work_budget_counts_both_runs_and_the_guards_are_reachable():
    """The most expensive model in the viewer per second of audio (~2x the jawari), and the dense
    |C|x|C| solve is NOT the reason — the string step and the rank-m correction are. So the budget
    counts the fret run AND its control, and it must be REACHABLE: lowering lambda raises
    fs = cN/(L*lam) and trips it inside the shipped slider range."""
    assert "error" in _fret(N=FRET_N_MAX + 1)
    assert "error" in _fret(audio_duration=FRET_AUDIO_MAX + 0.05)
    assert "error" in _fret(clearance=0.0)
    assert "error" in _fret(rail_stiffness=0.0)
    at_cap = _fret(audio_duration=FRET_AUDIO_MAX)
    assert "error" not in at_cap
    assert at_cap["meta"]["num_steps"] + at_cap["meta"]["n_control_steps"] <= FRET_WORK_MAX
    err = _fret(audio_duration=FRET_AUDIO_MAX, **{"lambda": 0.2})
    assert "error" in err and "steps" in err["error"]["message"]


def test_fret_control_is_bounded_and_the_animation_is_a_stride_of_one_run():
    """Unlike the bow (whose window is a settled tail), the fret buzzes from t = 0, so the animation
    is a stride of the SAME run — a second resonator would silently double the cost of a
    root-find-per-step model. And the control never outgrows its own cap."""
    d = _fret(audio_duration=0.4)
    assert d["meta"]["n_control_steps"] <= round(FRET_CONTROL_MAX * d["fs_sim"])
    assert d["meta"]["n_control_steps"] < d["meta"]["num_steps"]
    # frames land on the stride, starting at t = 0 (no settling window to skip)
    assert d["frame_times"][0] == 0.0
    assert d["frames"]["n_frames"] == len(d["frame_times"])
    assert d["frames"]["width"] == len(d["grid"]["x"])
    field = _decode_f32(d["frames"]["b64"])
    assert field.size == d["frames"]["n_frames"] * d["frames"]["width"]
    assert np.all(np.isfinite(field))


def test_fret_audio_is_a_near_termination_pickup_and_is_real():
    """The buzz signature is strongest nearest the termination (6.03/4.68/3.46/2.56/2.78x elevation
    at 0.02/0.05/0.10/0.25/0.50 L) while the level rises the other way; 0.05 L is the knee at 78 %
    of the best available elevation for 2.3x the level. Audio is the string pickup, as for every
    string model."""
    d = _fret()
    audio = _decode_f32(d["audio"]["b64"])
    assert d["audio"]["fs"] == AUDIO_FS
    assert audio.size == d["audio"]["n"] and np.all(np.isfinite(audio))
    assert 0.0 < d["audio"]["peak"] <= 1.0
    assert np.max(np.abs(audio)) == pytest.approx(0.9, abs=0.05)
    assert d["meta"]["probe_x"] == pytest.approx(0.05, abs=0.01)


def test_fret_episode_debounce_merges_chatter_without_inventing_it():
    """The reed's debounce rule, second customer: episodes closer than 10 % of a period are one
    event. At the default the raw onsets and debounced episodes agree closely (the string is not
    chattering), so the rule must not be silently deflating a real count — but it stays in place
    because the chatter regime is one slider away."""
    ct = _fret()["meta"]["contact"]
    assert ct["episodes"] <= ct["raw_onsets"]
    assert ct["episodes"] >= 0.9 * ct["raw_onsets"], "the default must not be over-merged"
    assert ct["episodes_per_period"] >= FRET_EPISODES_MIN


# =================================================================================================
# String -> modal body / bridge + radiation read-out (batch 12) — the COUPLING is the content
# =================================================================================================
#
# The first viewer of the third stage. The claim is the three-way energy split E_string + E_body +
# E_conn (the total conserves through the coupling while E_string alone does not) and the body
# colouring the radiated spectrum — NOT a new signal (for one source the far-field is the body
# pressure, scaled by 1/r and delayed). The measured numbers below come back through the payload as
# an end-to-end oracle: if the wrapper perturbed the rig they would drift.


def _body(**over):
    return web_serialize.simulate_to_payload({"model": "body", "audio_duration": 0.5, **over})


def test_body_conserves_through_the_coupling_while_the_string_alone_does_not():
    """THE batch, in one test: at sigma_body = 0 the TOTAL drifts < 1e-10 (energy is conserved
    through the spring coupling), yet E_string alone swings from ~100 % to a small fraction of the
    total — it is emphatically not conserved on its own. Shipping both numbers is the point: a flat
    total beside a sloshing E_string is the coupled-oscillator signature, not a bug."""
    d = _body(bridge_stiffness=8000)
    e = d["energy"]
    assert e["sigma_is_zero"] is True
    assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True
    ex = d["meta"]["exchange"]
    assert ex["total_drift"] < LOSSLESS_TOL
    # the contrast: the string's own energy fraction is NOT flat — it gives a big chunk to the body
    assert ex["string_frac_max"] > 0.9
    assert ex["string_frac_min"] < 0.6, "E_string alone must visibly not conserve (it sloshes out)"
    assert ex["body_frac_peak"] > 0.4, "the body must carry a large, visible fraction at the peak"


def test_body_k_zero_decouples_the_string_bit_for_bit():
    """The anchor. K = 0 severs the spring: the string evolves as a bare fixed/free IdealString, the
    body never moves (E_body = 0), and nothing radiates. The string FIELD is bit-identical to a
    stand-alone string plucked the same way — the sharpest cheap proof the coupling is a clean
    add-on that vanishes at K = 0 (the sympathetic all-zero-K / vk nonlinear=False precedent)."""
    d = _body(bridge_stiffness=0.0, audio_duration=0.3)
    ex = d["meta"]["exchange"]
    assert ex["body_frac_peak"] == 0.0, "a decoupled body must never move"
    assert ex["string_frac_min"] == pytest.approx(1.0), "the string keeps 100 % of the energy"
    assert d["audio"]["peak"] == 0.0, "a still body radiates nothing"

    # bit-identity of the string field at a matched step
    fs = d["fs_sim"]
    frames = _decode_f32(d["frames"]["b64"]).reshape(-1, d["frames"]["width"])
    last_step = round(d["frame_times"][-1] * fs)
    from physsynth.core.engine import simulate
    from physsynth.core.exciter import triangular_pluck
    from physsynth.core.string_ideal import IdealString
    c = np.sqrt(200.0 / 0.005)
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=c * 100 / 0.9, N=100,
                    boundary=("fixed", "free"), sigma=0.0)
    s.set_state(triangular_pluck(s.x, 1.0, 0.3, amplitude=1e-3))
    bare = simulate(s, num_steps=last_step, snapshot_stride=1).snapshots[-1][1].astype("<f4")
    assert float(np.max(np.abs(frames[-1] - bare))) == 0.0


def test_body_exchange_fractions_carry_e_conn_and_must_not_be_stacked():
    """The money panel ships four channels {e_string, e_body, e_conn, total}. E_conn is the cross-
    time spring energy: it swings its own (up to ~10 %) share and can dip negative, so E_string and
    E_body do NOT sum to the flat total — stacking them to 1 would hide E_conn. The test pins that
    E_conn is load-bearing (E_string + E_body departs from 1 by the E_conn amount)."""
    ex = _body(bridge_stiffness=8000)["meta"]["exchange"]
    assert ex["kind"] == "body"
    for key in ("time", "e_string_frac", "e_body_frac", "e_conn_frac", "total_frac"):
        assert len(ex[key]) == len(ex["time"]) and len(ex[key]) > 100
    es = np.array(ex["e_string_frac"])
    eb = np.array(ex["e_body_frac"])
    ec = np.array(ex["e_conn_frac"])
    tot = np.array(ex["total_frac"])
    # the three channels sum to the (flat ~1) total — E_conn is exactly the gap, never dropped
    assert np.allclose(es + eb + ec, tot, atol=1e-9)
    assert np.allclose(tot, 1.0, atol=1e-9)
    assert float(np.min(es + eb)) < 0.95, "E_conn carries a real share; stacking es+eb hides it"


def test_body_exchange_slosh_is_prompt_and_windowed():
    """The exchange trace is the FIRST 0.4 s of the run (~9 sloshes; the full 2 s reads as noise),
    and the body first peaks within tens of ms — the fast coupled exchange the panel must resolve.
    The scalars are measured over the full run, so they are robust to the window."""
    ex = _body(bridge_stiffness=8000, audio_duration=1.0)["meta"]["exchange"]
    assert ex["window"] == pytest.approx(0.4, abs=0.01)
    assert 0.0 < ex["first_peak_ms"] < 40.0, "the slosh is prompt (tens of ms), not a slow drift"


def test_body_terminus_glides_from_free_toward_clamped_as_the_bridge_stiffens():
    """The honest secondary readout: a soft bridge lets the free end swing (fundamental near the
    free c/4L = 50 Hz); stiffening it pulls the terminus toward the clamped c/2L = 100 Hz. It
    ASYMPTOTES BELOW 100 (the guard caps K and the body is finite-mass), so the readout never claims
    to reach it — the test asserts the monotone glide and the sub-clamped ceiling, not a target."""
    lo = _body(bridge_stiffness=200)["meta"]["spectrum"]
    hi = _body(bridge_stiffness=8000)["meta"]["spectrum"]
    assert lo["f1_free"] == pytest.approx(50.0) and lo["f1_clamped"] == pytest.approx(100.0)
    assert lo["f1_free"] < lo["terminus_f1"] < hi["terminus_f1"] < lo["f1_clamped"]
    assert hi["terminus_f1"] == pytest.approx(90.7, abs=1.5), "the probe's measured K=8000 glide"


def test_body_monopole_omega2_is_a_consistency_check_near_one_not_an_oracle():
    """The radiated pressure is the body's volume ACCELERATION, so its spectrum is (2 pi f)^2 times
    the body displacement spectrum — bin for bin, because a = phi. That makes the ratio ~1.00 a
    SANITY check on the _accel read / weights / byte order, and it is labelled 'consistency', never
    a radiation law. A broken acceleration read or a stale weight would move it off 1."""
    spx = _body(bridge_stiffness=8000)["meta"]["spectrum"]
    assert spx["omega2_consistency"] == pytest.approx(1.0, abs=0.05)


def test_body_one_over_r_scales_level_and_latency_only_never_the_spectrum_shape():
    """The far-field law: gain = rho0/(4 pi r), so gain*r is constant and the latency r/c0 grows
    with r, but the (normalised) spectrum SHAPE is computed from the raw volume acceleration and is
    therefore byte-for-byte identical across distances — the readout's 'shape unchanged' claim, made
    testable."""
    a = _body(distance=1.0, audio_duration=0.4)["meta"]["spectrum"]
    b = _body(distance=4.0, audio_duration=0.4)["meta"]["spectrum"]
    assert a["gain_times_r"] == pytest.approx(b["gain_times_r"])
    assert b["latency_ms"] == pytest.approx(4.0 * a["latency_ms"], rel=0.05)
    assert a["f"] == b["f"] and a["mag"] == b["mag"], "distance must not change the spectrum shape"


def test_body_sigma_body_gates_the_verdict_and_drops_the_decay_oracle():
    """sigma_body = 0 gives the conservation-drift check; sigma_body > 0 flips to PASSIVITY with NO
    2*sigma oracle line — decided by measurement (the off-harmonic coupled decay is multi-rate, so a
    single fitted rate against a flat oracle would lie; the jawari check gave the opposite answer
    because that barrier was lossless-elastic). The total must be monotone non-increasing."""
    d = _body(bridge_stiffness=8000, sigma_body=20.0)
    e = d["energy"]
    assert e["sigma_is_zero"] is False
    assert e["lossy"]["monotone"] is True
    assert "measured_2sigma" not in e["lossy"] and "oracle_2sigma" not in e["lossy"]


def test_body_guard_is_the_exact_bound_surfaced_as_a_clean_error():
    """The K slider's ceiling is the core's EXACT coupled guard (k^2 lambda_max(A) < 4), not the
    2-DOF footgun. Push K well past it and the wrapper returns a clean construction-error payload
    (never a 500/NaN); lambda >= 1 and a negative stiffness are clean param errors."""
    over = web_serialize.simulate_to_payload({"model": "body", "bridge_stiffness": 500_000})
    assert over["error"]["kind"] == "construction"
    assert "lambda_max" in over["error"]["message"]
    lam = web_serialize.simulate_to_payload({"model": "body", "lambda": 1.0})
    assert lam["error"]["kind"] == "param"
    neg = web_serialize.simulate_to_payload({"model": "body", "bridge_stiffness": -1.0})
    assert neg["error"]["kind"] == "param"


def test_body_frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped():
    """1D string field: the frame width matches the grid, every frame is finite, and node 0 (the
    nut) is exactly 0 in every frame — the string-boundary test (catches a byte-order scramble or a
    field/grid stride mismatch), here for the coupled string whose other end is the live bridge."""
    d = _body(audio_duration=0.3)
    x = d["grid"]["x"]
    frames = _decode_f32(d["frames"]["b64"]).reshape(-1, d["frames"]["width"])
    assert d["frames"]["width"] == len(x)
    assert d["frames"]["n_frames"] == frames.shape[0] == d["meta"]["n_frames"]
    assert np.all(np.isfinite(frames))
    assert float(np.max(np.abs(frames[:, 0]))) == 0.0, "the nut (node 0) is clamped"


def test_body_audio_is_the_far_field_pressure_real_and_normalized():
    """Audio = the retarded far-field pressure, resampled to 48 kHz and peak-normalized. It must be
    finite, at 48 kHz, within full scale; the reported peak is the raw physical pressure (Pa)."""
    d = _body(audio_duration=0.4)
    au = d["audio"]
    sig = _decode_f32(au["b64"])
    assert au["fs"] == AUDIO_FS and au["n"] == sig.size > 0
    assert np.all(np.isfinite(sig)) and float(np.max(np.abs(sig))) <= 1.0 + 1e-6
    assert au["peak"] > 0.0, "a coupled, radiating body must produce a nonzero far-field pressure"


def test_body_work_budget_and_the_n_ceiling_are_reachable():
    """The clamps a local render can hit: too many steps -> a clean work-budget ParamError, N above
    the body ceiling and the loss/distance ranges -> clean param errors. All surfaced, never a hang
    or a 500. The step cap is reachable within the audio cap by a light, fast string (small rho ->
    high fs), so it is a live backstop, not dead code shadowed by the duration cap."""
    over = web_serialize.simulate_to_payload(
        {"model": "body", "N": BODY_N_MAX, "lambda": 0.9, "rho": 0.001,
         "audio_duration": BODY_AUDIO_MAX})
    assert over["error"]["kind"] == "param" and "budget" in over["error"]["message"]
    assert BODY_WORK_MAX == 200_000, "the step backstop the message quotes"
    big_n = web_serialize.simulate_to_payload({"model": "body", "N": BODY_N_MAX + 1})
    assert big_n["error"]["kind"] == "param"
    loud = web_serialize.simulate_to_payload(
        {"model": "body", "sigma_body": BODY_SIGMA_BODY_MAX + 1.0})
    assert loud["error"]["kind"] == "param"
    far = web_serialize.simulate_to_payload({"model": "body", "distance": BODY_DISTANCE_MAX + 1.0})
    assert far["error"]["kind"] == "param"


def test_body_ignores_params_that_belong_to_other_models():
    """The K-collision guard: the bridge spring is 'bridge_stiffness', NOT the jawari/sympathetic
    'K' nor the mallet 'alpha'. Passing those stale names (as the frontend does when the user has
    visited other models) must not change a single number — otherwise a switch renders a different
    bridge with nothing on screen to say so (the recurring MODEL_RANGES leak)."""
    base = _body(bridge_stiffness=8000)
    noisy = _body(bridge_stiffness=8000, K=2.0e6, alpha=2.3, depth=1e-3, kappa=5.0, EA=1e4)
    assert noisy["meta"]["exchange"] == base["meta"]["exchange"]
    assert noisy["meta"]["spectrum"] == base["meta"]["spectrum"]
    assert noisy["energy"] == base["energy"]


# =================================================================================================
# String -> DISTRIBUTED plate body (models #5/#5b) + radiation (batch 13) — the body you WATCH ring
# =================================================================================================
#
# Batch 12's lumped ModalBody swapped for a distributed grid Plate, so the third stage finally has a
# PICTURE: the soundboard (supported #5) / cymbal (free #5b) ringing on the heatmap. The claim:
# the same three-way split (total conserves through the coupling while E_string sloshes) PLUS the
# 2D field, and the OPPOSITE per-boundary terminus story (supported lands near c/2L; free OVERSHOOTS
# it). The measured probe numbers (platebody-viewer-probe) come back through the payload end-to-end.


def _pb(**over):
    """Default to the free cymbal (the batch's headline body); 0.5 s keeps the suite fast."""
    return web_serialize.simulate_to_payload(
        {"model": "platebody", "domain": "free", "audio_duration": 0.5, **over})


def test_platebody_conserves_through_the_coupling_on_both_boundaries():
    """THE batch, in one test, for each body: at sigma_plate = 0 the TOTAL drifts < 1e-10 (energy is
    conserved through the spring coupling to the distributed plate), yet E_string alone swings from
    ~100 % to a small fraction — it is emphatically not conserved on its own. The free cymbal is the
    bigger slosh (E_plate ~83 %) than the supported soundboard (~77 %), both measured."""
    for boundary, plate_peak in (("free", 0.83), ("supported", 0.77)):
        d = _pb(domain=boundary, bridge_stiffness=3000)
        e = d["energy"]
        assert e["sigma_is_zero"] is True
        assert e["lossless"]["drift"] < LOSSLESS_TOL and e["lossless"]["pass"] is True
        ex = d["meta"]["exchange"]
        assert ex["total_drift"] < LOSSLESS_TOL
        assert ex["string_frac_max"] > 0.9
        assert ex["string_frac_min"] < 0.6, "E_string alone must visibly not conserve (it sloshes)"
        assert ex["body_frac_peak"] == pytest.approx(plate_peak, abs=0.05), "measured plate share"


def test_platebody_terminus_is_the_OPPOSITE_story_per_boundary():
    """The batch's sharp per-boundary claim (advisor): read from a near-nut pickup, the supported
    soundboard is a near-rigid termination so its fundamental lands JUST BELOW c/2L = 100 Hz (~98),
    while the FREE plate loads the end as a reactive mass-spring and OVERSHOOTS c/2L, landing ~117.
    A single 'toward clamped c/2L' readout (batch 12's) would be wrong for the free edge — this pins
    that the two boundaries straddle 100 Hz in opposite directions, matching the core diagnostic."""
    supp = _pb(domain="supported", bridge_stiffness=3000)["meta"]["spectrum"]
    free = _pb(domain="free", bridge_stiffness=3000)["meta"]["spectrum"]
    assert supp["f1_free"] == pytest.approx(50.0) and supp["f1_clamped"] == pytest.approx(100.0)
    assert supp["terminus_f1"] == pytest.approx(96.5, abs=2.0), "supported lands just below c/2L"
    assert free["terminus_f1"] == pytest.approx(116.7, abs=2.0), "free OVERSHOOTS c/2L"
    assert supp["terminus_f1"] < supp["f1_clamped"] < free["terminus_f1"], "they straddle 100 Hz"
    # DURATION-ROBUST: the terminus is a dedicated near-nut-pluck probe, NOT read off the user run —
    # the user's 0.3 L pluck flips free's argmax to ~99 at 2 s (the avoided-crossing doublet). The
    # probe must read the SAME overshoot at 2 s as at 0.5 s, or the "OVERSHOOTS" readout would lie.
    free2 = _pb(domain="free", bridge_stiffness=3000, audio_duration=2.0)["meta"]["spectrum"]
    assert free2["terminus_f1"] == pytest.approx(free["terminus_f1"], abs=1.0), "duration-robust"
    assert free2["terminus_f1"] > free2["f1_clamped"], "free still OVERSHOOTS at 2 s (not flipped)"
    # K-DEPENDENT: the terminus is NOT a fixed per-boundary label — a SOFT bridge leaves the free
    # end near its quarter-wave, so even the free plate reads BELOW c/2L at low K (overshoot needs
    # stiff). Pins that the frontend must derive "overshoots/near/climbing" from the value, not the
    # boundary alone (else a low-K panel would claim OVERSHOOTS while showing ~60).
    free_soft = _pb(domain="free", bridge_stiffness=500)["meta"]["spectrum"]
    assert free_soft["terminus_f1"] < free_soft["f1_clamped"], "a soft bridge does NOT overshoot"
    assert free_soft["terminus_f1"] < free["terminus_f1"], "the terminus climbs with K"


def test_platebody_k_zero_decouples_the_string_bit_for_bit():
    """The anchor. K = 0 severs the spring: the plate never moves (E_plate = 0), nothing radiates,
    and the string FIELD is bit-identical to a bare fixed/free IdealString plucked the same way —
    the sharpest cheap proof the distributed-body coupling is a clean add-on that vanishes at K = 0
    (the batch-12 / sympathetic all-zero-K precedent, here with the plate as the body)."""
    d = _pb(bridge_stiffness=0.0, audio_duration=0.3)
    ex = d["meta"]["exchange"]
    assert ex["body_frac_peak"] == 0.0, "a decoupled plate must never move"
    assert ex["string_frac_min"] == pytest.approx(1.0), "the string keeps 100 % of the energy"
    assert d["audio"]["peak"] == 0.0, "a still plate radiates nothing"
    fs = d["fs_sim"]
    st = d["string"]
    str_frames = _decode_f32(st["b64"]).reshape(-1, st["width"])
    last_step = round(d["frame_times"][-1] * fs)
    from physsynth.core.engine import simulate
    from physsynth.core.exciter import triangular_pluck
    from physsynth.core.string_ideal import IdealString
    c = np.sqrt(200.0 / 0.005)
    s = IdealString(L=1.0, T=200.0, rho=0.005, fs=c * 100 / 0.9, N=100,
                    boundary=("fixed", "free"), sigma=0.0)
    s.set_state(triangular_pluck(s.x, 1.0, 0.3, amplitude=1e-3))
    bare = simulate(s, num_steps=last_step, snapshot_stride=1).snapshots[-1][1].astype("<f4")
    assert float(np.max(np.abs(str_frames[-1] - bare))) == 0.0


def test_platebody_heatmap_is_a_real_2d_field_that_rings_and_stays_masked():
    """The batch's NEW content: the plate is a 2D heatmap you watch ring (batch 12 had only the 1D
    string). The field is dims = 2, its width/nx/ny line up with the decoded buffer and the mask, it
    is finite and genuinely nonzero (the plate moves), and — for the supported soundboard — every
    exterior (mask == 0) node clamps to 0 in every frame (the 2D boundary test; catches a byte-order
    scramble or a field/mask stride mismatch). The string strip rides along, same count."""
    d = _pb(domain="supported", bridge_stiffness=3000, audio_duration=0.3)
    fr, gr = d["frames"], d["grid"]
    assert fr["dims"] == 2 and gr["dims"] == 2
    nf, ny, nx = fr["n_frames"], fr["ny"], fr["nx"]
    field = _decode_f32(fr["b64"]).astype(float).reshape(nf, ny, nx)
    mask = _decode_u8(d["mask"]["b64"]).astype(bool).reshape(ny, nx)
    assert fr["width"] == nx and d["mask"]["nx"] == nx and d["mask"]["ny"] == ny
    assert np.all(np.isfinite(field))
    assert float(np.max(np.abs(field))) > 0.0, "the plate must actually ring"
    ext = ~mask
    if ext.any():
        assert float(np.max(np.abs(field[:, ext]))) == 0.0, "exterior nodes clamp to 0 every frame"
    # the string strip rides along: same frame count, N+1 wide, nut clamped
    st = d["string"]
    assert st["n_frames"] == nf and st["width"] == len(st["x"])
    str_frames = _decode_f32(st["b64"]).reshape(nf, st["width"])
    assert float(np.max(np.abs(str_frames[:, 0]))) == 0.0, "the nut (node 0) is clamped"


def test_platebody_exchange_fractions_carry_e_conn_and_must_not_be_stacked():
    """The money panel reuses batch-12's four channels {e_string, e_body(=plate), e_conn, total}.
    E_conn is the cross-time spring energy: it swings its own share and can dip negative, so the
    three channels sum to the flat ~1 total with E_conn exactly the gap — stacking es+eb would hide
    it. `kind` is 'platebody' so the frontend can pick the right per-boundary readout."""
    ex = _pb(bridge_stiffness=3000)["meta"]["exchange"]
    assert ex["kind"] == "platebody"
    for key in ("time", "e_string_frac", "e_body_frac", "e_conn_frac", "total_frac"):
        assert len(ex[key]) == len(ex["time"]) and len(ex[key]) > 100
    es = np.array(ex["e_string_frac"])
    eb = np.array(ex["e_body_frac"])
    ec = np.array(ex["e_conn_frac"])
    tot = np.array(ex["total_frac"])
    assert np.allclose(es + eb + ec, tot, atol=1e-9)
    assert np.allclose(tot, 1.0, atol=1e-9)
    assert float(np.min(es + eb)) < 0.98, "E_conn carries a real share; stacking es+eb hides it"
    assert 0.0 < ex["first_peak_ms"] < 40.0, "the slosh is prompt (tens of ms), not a slow drift"


def test_platebody_monopole_omega2_uses_volume_displacement_not_the_driving_point():
    """The omega^2 sanity is ~1.00 ONLY against the plate VOLUME displacement (Q'' = Q_vol'' for a
    distributed body). The batch-12 denominator w_b (here the single driving-point node) gives ~0.4,
    not ~1 — the real correction this batch makes. A ~1.00 here proves the pressure read / area
    weights / byte order are right; it is labelled 'consistency', never a radiation law."""
    for boundary in ("free", "supported"):
        spx = _pb(domain=boundary, bridge_stiffness=3000)["meta"]["spectrum"]
        assert spx["omega2_consistency"] == pytest.approx(1.0, abs=0.05)
        # the plate's own low modes are the spectrum markers (shown, not scored)
        assert len(spx["body_modes"]) >= 1 and spx["body_modes"][0] > 0.0


def test_platebody_one_over_r_scales_level_and_latency_only_never_the_spectrum_shape():
    """The far-field law carries over from batch 12: gain*r is constant, the latency r/c0 grows with
    r, but the normalised spectrum SHAPE (from the raw volume acceleration) is byte-for-byte the
    same across distances — the 'distance changes level + latency only' claim, made testable."""
    a = _pb(distance=1.0, audio_duration=0.4)["meta"]["spectrum"]
    b = _pb(distance=4.0, audio_duration=0.4)["meta"]["spectrum"]
    assert a["gain_times_r"] == pytest.approx(b["gain_times_r"])
    assert b["latency_ms"] == pytest.approx(4.0 * a["latency_ms"], rel=0.05)
    assert a["f"] == b["f"] and a["mag"] == b["mag"], "distance must not change the spectrum shape"


def test_platebody_sigma_plate_gates_the_verdict_and_drops_the_decay_oracle():
    """sigma_plate = 0 gives the conservation-drift check; sigma_plate > 0 flips to PASSIVITY, NO
    2*sigma oracle line — decided by measurement (the off-modal coupled decay is multi-rate, so a
    single fitted rate vs a flat oracle would lie). The total must be monotone non-increasing."""
    d = _pb(bridge_stiffness=3000, sigma_plate=20.0)
    e = d["energy"]
    assert e["sigma_is_zero"] is False
    assert e["lossy"]["monotone"] is True
    assert "measured_2sigma" not in e["lossy"] and "oracle_2sigma" not in e["lossy"]


def test_platebody_guard_is_the_exact_bound_surfaced_as_a_clean_error_on_both_boundaries():
    """The K ceiling is the core's EXACT Sherman-Morrison guard, not the 2-DOF footgun. It is the
    SAME for supported and free (the string end-node term dominates), and SHRINKS as n_plate grows
    — so a high-n_plate x high-K corner trips it. Both surface as clean construction-error payloads
    (never a 500/NaN); lambda >= 1 and a negative stiffness are clean param errors."""
    for boundary in ("free", "supported"):
        over = web_serialize.simulate_to_payload(
            {"model": "platebody", "domain": boundary, "bridge_stiffness": 500_000})
        assert over["error"]["kind"] == "construction"
        corner = web_serialize.simulate_to_payload(
            {"model": "platebody", "domain": boundary, "n_plate": PLATEBODY_NPLATE_MAX,
             "bridge_stiffness": PLATEBODY_K_MAX})
        assert corner["error"]["kind"] == "construction", "high n_plate x high K trips the guard"
    lam = web_serialize.simulate_to_payload({"model": "platebody", "lambda": 1.0})
    assert lam["error"]["kind"] == "param"
    neg = web_serialize.simulate_to_payload({"model": "platebody", "bridge_stiffness": -1.0})
    assert neg["error"]["kind"] == "param"
    bad = web_serialize.simulate_to_payload({"model": "platebody", "domain": "clamped"})
    assert bad["error"]["kind"] == "param", "an unknown boundary is a clean param error"


def test_platebody_audio_is_the_far_field_pressure_real_and_normalized():
    """Audio = the retarded far-field pressure, resampled to 48 kHz and peak-normalized: finite, at
    48 kHz, within full scale; the reported peak is the raw physical pressure (Pa)."""
    d = _pb(audio_duration=0.4)
    au = d["audio"]
    sig = _decode_f32(au["b64"])
    assert au["fs"] == AUDIO_FS and au["n"] == sig.size > 0
    assert np.all(np.isfinite(sig)) and float(np.max(np.abs(sig))) <= 1.0 + 1e-6
    assert au["peak"] > 0.0, "a coupled, radiating plate must produce a nonzero far-field pressure"


def test_platebody_work_budget_and_the_ceilings_are_reachable():
    """The clamps a local render can hit, all surfaced as clean param errors (never a hang/500): the
    node-step budget, N over the string cap, n_plate out of range, the loss/distance caps."""
    over = web_serialize.simulate_to_payload(
        {"model": "platebody", "N": PLATEBODY_NSTRING_MAX, "lambda": 0.9, "rho": 0.001,
         "n_plate": PLATEBODY_NPLATE_MAX, "audio_duration": PLATEBODY_AUDIO_MAX})
    assert over["error"]["kind"] == "param" and "budget" in over["error"]["message"]
    assert PLATEBODY_WORK_MAX == 1.0e8, "the node-step backstop the message quotes"
    big_n = web_serialize.simulate_to_payload(
        {"model": "platebody", "N": PLATEBODY_NSTRING_MAX + 1})
    assert big_n["error"]["kind"] == "param"
    big_np = web_serialize.simulate_to_payload(
        {"model": "platebody", "n_plate": PLATEBODY_NPLATE_MAX + 1})
    assert big_np["error"]["kind"] == "param"
    loud = web_serialize.simulate_to_payload(
        {"model": "platebody", "sigma_plate": PLATEBODY_SIGMA_MAX + 1.0})
    assert loud["error"]["kind"] == "param"
    far = web_serialize.simulate_to_payload(
        {"model": "platebody", "distance": PLATEBODY_DISTANCE_MAX + 1.0})
    assert far["error"]["kind"] == "param"


def test_platebody_ignores_params_that_belong_to_other_models():
    """The K-collision guard: the bridge spring is 'bridge_stiffness', NOT the jawari/symp 'K'
    nor the mallet 'alpha'. Passing those stale names (as the frontend does after visiting other
    models) must not change a single number — the recurring MODEL_RANGES leak, one body up."""
    base = _pb(bridge_stiffness=3000)
    noisy = _pb(bridge_stiffness=3000, K=2.0e6, alpha=2.3, depth=1e-3, kappa=5.0, EA=1e4)
    assert noisy["meta"]["exchange"] == base["meta"]["exchange"]
    assert noisy["meta"]["spectrum"] == base["meta"]["spectrum"]
    assert noisy["energy"] == base["energy"]
