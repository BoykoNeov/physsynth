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
    DISPLAY_MAX,
    LOSSLESS_TOL,
    MEMBRANE_LAMBDA_MAX,
    MEMBRANE_N_MAX,
    PLATE_N_MAX,
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
