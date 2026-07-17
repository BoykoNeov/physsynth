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
    BOW_BALANCE_TOL,
    BOW_N_MAX,
    BOW_SLIP_MATCH_TOL,
    DISPLAY_MAX,
    GEOM_DT_MAX,
    GEOM_LAM_LONG_MAX,
    GEOM_N_MAX,
    GEOM_WORK_MAX,
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
