//! The barrier family (model #8) in its three configurations — `serialize.py`'s jawari, juari and
//! fret builders.
//!
//! * **jawari** — a curved bridge hugging the termination: the string wraps onto it and its
//!   departure point travels, re-injecting high partials. The claim is the SHIMMER (late-window
//!   brightness over a clean string), not energy; the bridge is lossless-elastic, so the damped
//!   string's `2 sigma0` decay oracle survives.
//! * **juari** — the tanpura's cotton thread: ONE barrier node, whose buzz is position-SELECTIVE.
//!   The claim is the TUNING CURVE (buzz against thread position), swept at a canonical settled
//!   duration so the map does not move with the audio slider.
//! * **fret** — a flat rail: slap-and-release, the jawari's physical opposite. The gated claim is
//!   INTERMITTENCY (episodes per period at a finite duty), and the decay reading is a diagnostic
//!   triple, because the flat rail breaks equipartition and the `2 sigma0` oracle with it.
//!
//! Every brightness number here is a spectral centroid of `rfft` at the record's own length
//! (`physsynth_analysis::spectrum::rfft`), so those are tolerance ports of pocketfft's; the
//! trajectories and every contact statistic are exact.

use std::f64::consts::PI;

use physsynth_analysis::spectrum::{hann, rfft_mag, rfftfreq};
use physsynth_core::collision::BarrierString;
use physsynth_core::fmt::py_float;
use physsynth_core::reduce::sum;
use physsynth_core::string_damped::{self as damped, DampedStiffString};
use physsynth_core::string_stiff::THETA_DEFAULT;
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, fit_decay, EnergyOpts};
use crate::horizon::string_block;
use crate::py::{
    b64encode, b64f32, dot, finite_list, int, linspace_idx, max_abs, np_mean, num, opt_num, py_int,
    py_repr, py_str, round_int, round_nd,
};
use crate::string::{
    construction, damped_info, int_at_least_one, ANIM_WIN_MAX, FRAMES_PER_PERIOD, MAX_FRAMES,
    N_MIN, SPEED_MAX,
};
use crate::tension::{interp_zero_cross_frequency, mode1_shape};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Jawari grid ceiling; the dense contact solve is `|C| x |C|`.
pub const JAWARI_N_MAX: i64 = 128;
/// Longest jawari audio, seconds.
pub const JAWARI_AUDIO_MAX: f64 = 1.5;
/// Total steps across BOTH jawari runs (the bridge and the clean contrast).
pub const JAWARI_WORK_MAX: i64 = 150_000;
/// Sub-unity: the coupled contact solve wants headroom below Nyquist.
pub const JAWARI_LAM_DEFAULT: f64 = 0.4;
/// Bridge stiffness, N/m^alpha — stiff wood or bone.
pub const JAWARI_K_DEFAULT: f64 = 2.0e6;
/// Hertzian-ish contact exponent (fixed, not a slider).
pub const JAWARI_ALPHA_DEFAULT: f64 = 1.5;
/// Bridge span as a fraction of L.
pub const JAWARI_WIDTH_DEFAULT: f64 = 0.15;
/// Crest-to-far-edge drop of the parabola (m).
pub const JAWARI_DEPTH_DEFAULT: f64 = 1.0e-3;
/// Mode-1 amplitude (m).
pub const JAWARI_AMP_DEFAULT: f64 = 8.0e-3;
/// Loss on: "sustained" is meaningless without decay.
pub const JAWARI_SIGMA0_DEFAULT: f64 = 0.5;
/// Depth ceiling (m).
pub const JAWARI_DEPTH_MAX: f64 = 8.0e-3;
/// Amplitude ceiling (m).
pub const JAWARI_AMP_MAX: f64 = 4.0e-2;
/// Width ceiling (fraction of L).
pub const JAWARI_WIDTH_MAX: f64 = 0.4;
/// Late-window centroid elevation over the clean string: the headline gate.
pub const JAWARI_ELEVATION_GATE: f64 = 2.5;
/// `downswing / depth` below which the bridge grazes — LABEL, not FAIL.
pub const JAWARI_RATIO_FLOOR: f64 = 1.5;

/// Juari grid ceiling.
pub const JUARI_N_MAX: i64 = 128;
/// Sub-unity lambda, as the jawari's.
pub const JUARI_LAM_DEFAULT: f64 = 0.4;
/// Mode-1 amplitude (m).
pub const JUARI_AMP_DEFAULT: f64 = 8.0e-3;
/// Amplitude ceiling (m).
pub const JUARI_AMP_MAX: f64 = 4.0e-2;
/// Loss on by default.
pub const JUARI_SIGMA0_DEFAULT: f64 = 0.5;
/// Thread position (fraction of L); near the measured sweet spot.
pub const JUARI_THREAD_DEFAULT: f64 = 0.10;
/// A generic listening point (0.5 hides the evens on the pickup's own node).
pub const JUARI_PICKUP_DEFAULT: f64 = 0.83;
/// The SETTLED-buzz window (s): canonical, not the audio slider.
pub const JUARI_SWEEP_DUR: f64 = 0.24;
/// Longest juari audio, seconds.
pub const JUARI_AUDIO_MAX: f64 = 0.6;
/// Steps across the sweep, the jawari reference, the main run and both cleans.
pub const JUARI_WORK_MAX: i64 = 240_000;
/// Elevation below which a position is "weak".
pub const JUARI_ELEVATION_GATE: f64 = 2.0;
/// The near-nut band whose honest position count is quoted.
pub const JUARI_NEAR_NUT_FRAC: f64 = 0.15;
/// Thread positions swept for the tuning curve (fraction of L).
pub const JUARI_SWEEP_FRACS: [f64; 10] =
    [0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.21, 0.30, 0.45, 0.65];

/// Fret grid ceiling (~261 us/step at N = 100 in the reference).
pub const FRET_N_MAX: i64 = 100;
/// Longest fret audio, seconds.
pub const FRET_AUDIO_MAX: f64 = 0.6;
/// The out-of-reach control is short: its centroid is window-invariant.
pub const FRET_CONTROL_MAX: f64 = 0.2;
/// Total steps across the fret run and the control.
pub const FRET_WORK_MAX: i64 = 60_000;
/// The validated collision-signature lambda.
pub const FRET_LAM_DEFAULT: f64 = 0.4;
/// Rail stiffness, N/m^alpha — a stiff metal fret.
pub const FRET_K_DEFAULT: f64 = 2.0e6;
/// Contact exponent (fixed).
pub const FRET_ALPHA: f64 = 1.5;
/// Clearance (m): the brightness peak and the intermittency default.
pub const FRET_CLEARANCE_DEFAULT: f64 = 2.0e-3;
/// Clearance ceiling (m).
pub const FRET_CLEARANCE_MAX: f64 = 8.0e-3;
/// Rail under the whole string.
pub const FRET_RAIL_FRAC_DEFAULT: f64 = 1.0;
/// Below ~0.15 the rail is out of reach (measured), so the floor is enforced here.
pub const FRET_RAIL_FRAC_MIN: f64 = 0.2;
/// Mode-1 amplitude (m).
pub const FRET_AMP_DEFAULT: f64 = 5.0e-3;
/// Amplitude ceiling (m).
pub const FRET_AMP_MAX: f64 = 2.0e-2;
/// Loss on (the jawari's rule).
pub const FRET_SIGMA0_DEFAULT: f64 = 0.5;
/// The knee: 78 % of the best elevation at 2.3x the level.
pub const FRET_PICKUP_DEFAULT: f64 = 0.05;
/// Transit-paced frames per period.
pub const FRET_FRAMES_PER_PERIOD: f64 = 8.0;
/// Below this the debounce rounds under a column and FRAGMENTS the slaps.
pub const FRET_RASTER_COLS_PER_PERIOD: f64 = 10.0;
/// Raster width ceiling.
pub const FRET_RASTER_MAX_COLS: f64 = 1200.0;
/// Raster height ceiling: x-binning is free for the picture, fatal for the number.
pub const FRET_RASTER_MAX_ROWS: usize = 128;
/// Merge episodes closer than 10 % of a period (the reed's rule).
pub const FRET_DEBOUNCE_FRAC: f64 = 0.10;
/// Episodes per period; measured >= 1.00 everywhere reachable.
pub const FRET_EPISODES_MIN: f64 = 0.5;
/// The "stopped releasing" bar — a guarantee that never fires (see the reference's note).
pub const FRET_DUTY_MAX: f64 = 0.9;
/// The brightness band (Hz).
pub const FRET_CENTROID_FMAX: f64 = 8000.0;
/// Where the measured elevation peaks (m): named, not claimed as a law.
pub const FRET_BRIGHTNESS_PEAK: f64 = 2.0e-3;

fn pf(x: f64) -> String {
    py_float(x)
}

/// `N` read as `int(p.get("N", 100))`, refused as `param`.
fn requested_n(p: &Value) -> Result<i64, Refusal> {
    let v = p.get("N").cloned().unwrap_or(json!(100));
    py_int(&v).map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&v))))
}

/// Python's `min(L, T, rho) <= 0` guard.
fn check_positive(l: f64, t: f64, rho: f64) -> Result<(), Refusal> {
    let mut mn = l;
    for v in [t, rho] {
        if v < mn {
            mn = v;
        }
    }
    if mn <= 0.0 {
        return Err(Refusal::Param("L, T, rho must all be positive.".into()));
    }
    Ok(())
}

fn check_lambda(lam: f64) -> Result<(), Refusal> {
    if !(0.0 < lam && lam < 1.0) {
        return Err(Refusal::Param(format!(
            "lambda must be in (0, 1), got {}: the coupled contact solve needs headroom below the \
             string's marginal Nyquist mode.",
            pf(lam)
        )));
    }
    Ok(())
}

/// A lossless-bridge damped string (`kappa = 0`, `sigma1 = 0`, theta at its default).
fn damped_string(
    l: f64,
    t: f64,
    rho: f64,
    fs: f64,
    n: i64,
    sigma0: f64,
) -> Result<DampedStiffString, Refusal> {
    let sp = damped::Params::new(l, t, rho, fs, n, 0.0, sigma0, 0.0, THETA_DEFAULT, true)
        .map_err(construction)?;
    Ok(DampedStiffString::new(sp))
}

/// A `BarrierString` with the binding's defaults (`eta_tol` 1e-12, Newton 1e-13 / 60).
fn barrier_string(
    string: DampedStiffString,
    barrier: &[f64],
    stiffness: f64,
    alpha: f64,
) -> Result<BarrierString, Refusal> {
    BarrierString::new(string, barrier, stiffness, alpha, 0.0, 1e-12, 1e-13, 60)
        .map_err(construction)
}

/// Displace a barrier string from rest into mode 1 at `amplitude`.
fn pluck_mode1(bar: &mut BarrierString, amplitude: f64) {
    let shape: Vec<f64> = mode1_shape(bar.string.p.n)
        .iter()
        .map(|s| s * amplitude)
        .collect();
    let v0 = vec![0.0; shape.len()];
    bar.set_state(&shape, &v0);
}

/// The barrier profile scattered onto the grid (`nan` off the support) — the INDEXING TRAP: the
/// model's `b` and `contact_mask()` are over the SUPPORT, never the grid.
fn barrier_on_grid(bar: &BarrierString) -> Vec<f64> {
    let mut g = vec![f64::NAN; bar.string.p.nodes()];
    for (&i, &b) in bar.p.support.iter().zip(&bar.p.b) {
        g[i] = b;
    }
    g
}

/// `_spectral_centroid`: amplitude-weighted mean frequency of `|rfft(sig * hann)|`.
pub fn spectral_centroid(sig: &[f64], fs: f64) -> f64 {
    centroid_below(sig, fs, f64::INFINITY)
}

/// The centroid over the bins at or below `fmax` (`_fret_centroid` with `FRET_CENTROID_FMAX`).
fn centroid_below(sig: &[f64], fs: f64, fmax: f64) -> f64 {
    if sig.len() < 4 {
        return f64::NAN;
    }
    let w = hann(sig.len());
    let x: Vec<f64> = sig.iter().zip(&w).map(|(s, wi)| s * wi).collect();
    let mag = rfft_mag(&x);
    let freqs = rfftfreq(sig.len(), 1.0 / fs);
    let sel: Vec<usize> = (0..freqs.len()).filter(|&i| freqs[i] <= fmax).collect();
    let m: Vec<f64> = sel.iter().map(|&i| mag[i]).collect();
    let fm: Vec<f64> = sel.iter().map(|&i| freqs[i] * mag[i]).collect();
    let total = sum(&m);
    if total > 0.0 {
        sum(&fm) / total
    } else {
        f64::NAN
    }
}

/// `_jawari_band_spectrum`: `[0, f_max]` of `|rfft(sig * hann)|`, max-pooled to `n_points`.
/// Returns `(f, mag, norm)`; the caller divides `mag` by a SHARED norm.
fn band_spectrum(sig: &[f64], fs: f64, f_max: f64, n_points: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let w = hann(sig.len());
    let x: Vec<f64> = sig.iter().zip(&w).map(|(s, wi)| s * wi).collect();
    let mag = rfft_mag(&x);
    let freqs = rfftfreq(sig.len(), 1.0 / fs);
    let (f, m): (Vec<f64>, Vec<f64>) = freqs
        .iter()
        .zip(&mag)
        .filter(|(&fr, _)| fr <= f_max)
        .map(|(&fr, &mg)| (fr, mg))
        .unzip();
    let top = |v: &[f64]| {
        if v.is_empty() {
            1.0
        } else {
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
    };
    if f.len() <= n_points {
        let norm = top(&m);
        return (f, m, norm);
    }
    let edges = linspace_idx(f.len(), n_points + 1);
    let (mut fo, mut mo) = (Vec::new(), Vec::new());
    for e in edges.windows(2) {
        let (a, b) = (e[0], e[1]);
        if b > a {
            fo.push(np_mean(&f[a..b]));
            mo.push(m[a..b].iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
    }
    let norm = top(&mo);
    (fo, mo, norm)
}

/// Two band spectra on ONE scale: `norm = max(jn, cn) or 1.0`.
fn shared_spectra(a: &[f64], b: &[f64], fs: f64, f_max: f64) -> (Value, Value) {
    let (af, am, an) = band_spectrum(a, fs, f_max, 240);
    let (bf, bm, bn) = band_spectrum(b, fs, f_max, 240);
    let mx = if bn > an { bn } else { an };
    let norm = if mx == 0.0 { 1.0 } else { mx };
    let scale = |v: &[f64]| v.iter().map(|x| x / norm).collect::<Vec<f64>>();
    (
        json!({"f": finite_list(&af, Some(3)), "mag": finite_list(&scale(&am), None)}),
        json!({"f": finite_list(&bf, Some(3)), "mag": finite_list(&scale(&bm), None)}),
    )
}

/// `np.std` (population), NumPy's two-pass pairwise form.
fn np_std(a: &[f64]) -> f64 {
    let m = np_mean(a);
    let sq: Vec<f64> = a.iter().map(|v| (v - m) * (v - m)).collect();
    (sum(&sq) / a.len() as f64).sqrt()
}

// == the jawari ====================================================================================

/// `_jawari_profile`: `-clearance - depth (x/d)²` on `0 < x <= d`, `-inf` off the span.
fn jawari_profile(x: &[f64], l: f64, width_frac: f64, depth: f64, clearance: f64) -> Vec<f64> {
    let d = width_frac * l;
    x.iter()
        .map(|&xi| {
            if xi > 0.0 && xi <= d {
                let r = xi / d;
                -clearance - depth * (r * r)
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// What `_build_jawari` reports beside the barrier string.
struct JawInfo {
    c: f64,
    l: f64,
    n: i64,
    fs: f64,
    lam: f64,
    sigma0: f64,
    width_frac: f64,
    depth: f64,
    support: usize,
}

/// `_build_jawari`: `clearance = 1.0` drops the whole curve out of reach (the clean contrast).
fn build_jawari(p: &Value, clearance: f64) -> Result<(BarrierString, JawInfo), Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", JAWARI_LAM_DEFAULT)?;
    // NOT `K` and NOT `alpha`: both names belong to other models' sliders at other scales.
    let k = fnum(p, "bridge_stiffness", JAWARI_K_DEFAULT)?;
    let width_frac = fnum(p, "width_frac", JAWARI_WIDTH_DEFAULT)?;
    let depth = fnum(p, "depth", JAWARI_DEPTH_DEFAULT)?;
    let sigma0 = fnum(p, "sigma0", JAWARI_SIGMA0_DEFAULT)?;
    let n = requested_n(p)?;
    if !(N_MIN..=JAWARI_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {JAWARI_N_MAX}] for the jawari, got {n}."
        )));
    }
    check_positive(l, t, rho)?;
    check_lambda(lam)?;
    if !(0.0 < width_frac && width_frac <= JAWARI_WIDTH_MAX) {
        return Err(Refusal::Param(format!(
            "width_frac must be in (0, {}], got {}.",
            pf(JAWARI_WIDTH_MAX),
            pf(width_frac)
        )));
    }
    if !(0.0 < depth && depth <= JAWARI_DEPTH_MAX) {
        return Err(Refusal::Param(format!(
            "depth must be in (0, {}] m, got {}.",
            pf(JAWARI_DEPTH_MAX),
            pf(depth)
        )));
    }
    if k <= 0.0 {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be positive, got {}.",
            pf(k)
        )));
    }
    if sigma0 < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma0 must be >= 0, got {}.",
            pf(sigma0)
        )));
    }
    let c = (t / rho).sqrt();
    let fs = c * n as f64 / (l * lam);
    let string = damped_string(l, t, rho, fs, n, sigma0)?;
    let lam_s = string.p.lam;
    let barrier = jawari_profile(&string.p.grid(), l, width_frac, depth, clearance);
    let support = barrier.iter().filter(|b| b.is_finite()).count();
    let bar = barrier_string(string, &barrier, k, JAWARI_ALPHA_DEFAULT)?;
    Ok((
        bar,
        JawInfo {
            c,
            l,
            n,
            fs,
            lam: lam_s,
            sigma0,
            width_frac,
            depth,
            support,
        },
    ))
}

/// `_JawariRun`: the pickup, the wrap edge (support-relative) and the energy, per step.
struct BarRun {
    e: Vec<f64>,
    pickup: Vec<f64>,
    /// Furthest-in-contact SUPPORT index, -1 when clear of the bridge.
    wrap: Vec<f64>,
    n_contact: Vec<f64>,
    frames: Vec<Vec<f64>>,
    frame_steps: Vec<usize>,
}

/// `_run_jawari`.
fn run_barrier(
    bar: &mut BarrierString,
    n_steps: usize,
    pickup_idx: usize,
    anim_stride: usize,
    frame_until: usize,
    capture_wrap: bool,
) -> Result<BarRun, Refusal> {
    let mut r = BarRun {
        e: Vec::with_capacity(n_steps + 1),
        pickup: Vec::with_capacity(n_steps + 1),
        wrap: vec![-1.0; n_steps + 1],
        n_contact: vec![0.0; n_steps + 1],
        frames: Vec::new(),
        frame_steps: Vec::new(),
    };
    let sample = |bar: &BarrierString, r: &mut BarRun, i: usize| {
        r.e.push(bar.energy());
        r.pickup.push(bar.string.u[pickup_idx]);
        if capture_wrap {
            let m = bar.contact_mask();
            let nc = m.iter().filter(|&&b| b).count();
            r.n_contact[i] = nc as f64;
            if nc > 0 {
                r.wrap[i] = m.iter().rposition(|&b| b).expect("nc > 0") as f64;
            }
        }
    };
    sample(bar, &mut r, 0);
    if frame_until >= 1 {
        r.frames.push(bar.string.u.clone());
        r.frame_steps.push(0);
    }
    for i in 1..=n_steps {
        bar.step();
        sample(bar, &mut r, i);
        if i <= frame_until && i % anim_stride == 0 {
            r.frames.push(bar.string.u.clone());
            r.frame_steps.push(i);
        }
    }
    if !r.e.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }
    Ok(r)
}

/// The shared 1-D frames / audio / energy half of a barrier payload.
struct Common {
    frames: Value,
    frame_times: Value,
    field_amp: f64,
    audio: Value,
    energy_time: Vec<f64>,
}

fn common_half(r: &BarRun, fs: f64) -> Common {
    let flat: Vec<f64> = r.frames.iter().flatten().copied().collect();
    let n_frames = r.frames.len();
    let (audio48, peak) = resample_normalize(&r.pickup, fs);
    let times: Vec<f64> = r.frame_steps.iter().map(|&i| i as f64 / fs).collect();
    Common {
        frames: json!({
            "b64": b64f32(&flat),
            "n_frames": int(n_frames as i64),
            "width": int(if n_frames > 0 { r.frames[0].len() as i64 } else { 0 }),
            "dims": int(1),
        }),
        frame_times: finite_list(&times, Some(6)),
        field_amp: if flat.is_empty() { 0.0 } else { max_abs(&flat) },
        audio: json!({
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        }),
        energy_time: (0..r.e.len()).map(|i| i as f64 / fs).collect(),
    }
}

/// The front half every barrier builder shares: playback, pickup, audio, animation, amplitude.
struct Front {
    playback_speed: f64,
    pickup_frac: f64,
    audio_dur: f64,
    anim_win: f64,
    amplitude: f64,
    fpp: i64,
}

/// The first five reads, in the reference's order. `frames_per_period` comes separately, because
/// the juari reads `thread_position` between `amplitude` and it (the order decides which of two
/// bad values is reported).
fn read_front(
    p: &Value,
    pickup_default: f64,
    audio_default: f64,
    anim_default: f64,
    amp_default: f64,
) -> Result<Front, Refusal> {
    Ok(Front {
        playback_speed: fnum(p, "playback_speed", 0.02)?,
        pickup_frac: fnum(p, "pickup_position", pickup_default)?,
        audio_dur: fnum(p, "audio_duration", audio_default)?,
        anim_win: fnum(p, "animation_window", anim_default)?,
        amplitude: fnum(p, "amplitude", amp_default)?,
        fpp: 0,
    })
}

fn finish_front(mut f: Front, p: &Value, fpp_default: f64) -> Result<Front, Refusal> {
    f.fpp = int_at_least_one(fnum(p, "frames_per_period", fpp_default)?)?;
    Ok(f)
}

fn check_front(f: &Front, audio_max: f64, amp_max: f64) -> Result<(), Refusal> {
    if !(0.0 < f.playback_speed && f.playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(f.playback_speed)
        )));
    }
    if !(0.0 < f.pickup_frac && f.pickup_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pickup_position must be in (0, 1), got {}.",
            pf(f.pickup_frac)
        )));
    }
    if !(0.0 < f.audio_dur && f.audio_dur <= audio_max) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(audio_max),
            pf(f.audio_dur)
        )));
    }
    if !(0.0 < f.anim_win && f.anim_win <= ANIM_WIN_MAX) {
        return Err(Refusal::Param(format!(
            "animation_window must be in (0, {}] s, got {}.",
            pf(ANIM_WIN_MAX),
            pf(f.anim_win)
        )));
    }
    if !(0.0 < f.amplitude && f.amplitude <= amp_max) {
        return Err(Refusal::Param(format!(
            "amplitude must be in (0, {}] m, got {}.",
            pf(amp_max),
            pf(f.amplitude)
        )));
    }
    Ok(())
}

/// The animation stride and window shared by the barrier scenes.
fn anim_plan(fs: f64, f_ref: f64, fpp: i64, n_steps: i64, anim_win: f64) -> (i64, i64) {
    let mut stride = round_int((fs / f_ref) / fpp as f64).max(1);
    let n_anim = n_steps.min(stride.max(round_int(anim_win * fs)));
    if n_anim / stride > MAX_FRAMES {
        stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    (stride, n_anim)
}

fn pickup_index(frac: f64, n: i64) -> usize {
    round_int(frac * n as f64).max(1).min(n - 1) as usize
}

/// `_jawari_shimmer_block`.
fn shimmer_block(
    jaw: &BarRun,
    clean: &BarRun,
    fs: f64,
    f1: f64,
    info: &JawInfo,
    amplitude: f64,
) -> Value {
    let half = jaw.pickup.len() / 2;
    let j_e = spectral_centroid(&jaw.pickup[..half], fs);
    let j_l = spectral_centroid(&jaw.pickup[half..], fs);
    let c_e = spectral_centroid(&clean.pickup[..half], fs);
    let c_l = spectral_centroid(&clean.pickup[half..], fs);
    let elevation = if c_l > 0.0 { j_l / c_l } else { f64::NAN };
    let f_max = (0.45 * fs).min((20.0 * f1).max(2000.0));
    let (js, cs) = shared_spectra(&jaw.pickup[half..], &clean.pickup[half..], fs, f_max);

    let contacting: Vec<bool> = jaw.wrap.iter().map(|&w| w >= 0.0).collect();
    let wrap: Vec<f64> = jaw.wrap.iter().copied().filter(|&w| w >= 0.0).collect();
    let touching: Vec<f64> = (0..contacting.len())
        .filter(|&i| contacting[i])
        .map(|i| jaw.n_contact[i])
        .collect();
    let downswing = amplitude * PI * info.width_frac;
    let ratio = if info.depth > 0.0 {
        downswing / info.depth
    } else {
        f64::INFINITY
    };
    let duty = contacting.iter().filter(|&&b| b).count() as f64 / contacting.len() as f64;
    json!({
        "kind": "jawari",
        "centroid": {
            "jawari_early": num(round_nd(j_e, 1)), "jawari_late": num(round_nd(j_l, 1)),
            "clean_early": num(round_nd(c_e, 1)), "clean_late": num(round_nd(c_l, 1)),
        },
        "elevation": num(round_nd(elevation, 3)),
        "elevation_gate": num(JAWARI_ELEVATION_GATE),
        "shimmering": elevation > JAWARI_ELEVATION_GATE,
        "sustain_ratio": if j_e > 0.0 { num(round_nd(j_l / j_e, 3)) } else { Value::Null },
        "clean_sustain_ratio": if c_e > 0.0 { num(round_nd(c_l / c_e, 3)) } else { Value::Null },
        "downswing": num(downswing),
        "depth": num(info.depth),
        "ratio": num(round_nd(ratio, 2)),
        "ratio_floor": num(JAWARI_RATIO_FLOOR),
        "grazing": ratio < JAWARI_RATIO_FLOOR,
        "wrap": {
            "std": if wrap.is_empty() { Value::Null } else { num(round_nd(np_std(&wrap), 2)) },
            "min_node": if wrap.is_empty() { Value::Null } else {
                int(wrap.iter().copied().fold(f64::INFINITY, f64::min) as i64)
            },
            "max_node": if wrap.is_empty() { Value::Null } else {
                int(wrap.iter().copied().fold(f64::NEG_INFINITY, f64::max) as i64)
            },
            "support": int(info.support as i64),
            "duty": num(round_nd(duty, 4)),
            "mean_nodes": if wrap.is_empty() { Value::Null } else {
                num(round_nd(np_mean(&touching), 2))
            },
            "flat_rail_std": num(2.35),
            "curve_std_suite": num(4.89),
        },
        "spectra": {"f_max": num(round_nd(f_max, 1)), "jawari": js, "clean": cs},
        "f1": num(round_nd(f1, 3)),
    })
}

/// `_build_payload_jawari`.
pub fn build_payload_jawari(p: &Value) -> Result<Value, Refusal> {
    let f = read_front(p, 0.5, 0.24, 0.06, JAWARI_AMP_DEFAULT)?;
    let f = finish_front(f, p, FRAMES_PER_PERIOD)?;
    check_front(&f, JAWARI_AUDIO_MAX, JAWARI_AMP_MAX)?;

    let (mut jaw, info) = build_jawari(p, 0.0)?;
    let (fs, l, c, n) = (info.fs, info.l, info.c, info.n);
    let f1 = c / (2.0 * l);
    let n_steps = round_int(f.audio_dur * fs).max(1);
    if 2 * n_steps > JAWARI_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs {} steps across the jawari and clean runs (budget \
             {JAWARI_WORK_MAX}): every step is a vector contact solve over the bridge support. \
             Shorten audio_duration, lower N, or raise lambda.",
            2 * n_steps
        )));
    }
    let pickup_idx = pickup_index(f.pickup_frac, n);
    let (stride, n_anim) = anim_plan(fs, f1, f.fpp, n_steps, f.anim_win);
    pluck_mode1(&mut jaw, f.amplitude);
    let run = run_barrier(
        &mut jaw,
        n_steps as usize,
        pickup_idx,
        stride as usize,
        n_anim as usize,
        true,
    )?;
    // The clean contrast: the SAME string with the bridge out of reach.
    let (mut clean, _) = build_jawari(p, 1.0)?;
    pluck_mode1(&mut clean, f.amplitude);
    let clean_run = run_barrier(&mut clean, n_steps as usize, pickup_idx, 1, 0, false)?;

    let com = common_half(&run, fs);
    let support = &jaw.p.support;
    let wrap_grid: Vec<Value> = run
        .frame_steps
        .iter()
        .map(|&i| {
            let w = run.wrap[i];
            int(if w >= 0.0 {
                support[w as usize] as i64
            } else {
                -1
            })
        })
        .collect();
    let x = jaw.string.p.grid();
    Ok(json!({
        "model": "jawari",
        "horizon": string_block(&damped_info(&jaw.string), "the string under the jawari"),
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(info.lam, 6)),
        "grid": {"x": finite_list(&x, Some(6)), "barrier": finite_list(&barrier_on_grid(&jaw), None)},
        "frames": com.frames,
        "frame_times": com.frame_times,
        "wrap_frames": wrap_grid,
        "anim_dt": num(stride as f64 / fs),
        "playback_speed": num(f.playback_speed),
        "field_amp": num(com.field_amp),
        "audio": com.audio,
        "energy": energy_block(&com.energy_time, &run.e, info.sigma0 == 0.0, 2.0 * info.sigma0,
                               EnergyOpts::default()),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(run.frames.len() as i64),
            "probe_x": num(round_nd(x[pickup_idx], 4)),
            "bridge_span": num(round_nd(info.width_frac * l, 4)),
            "spectrum": shimmer_block(&run, &clean_run, fs, f1, &info, f.amplitude),
        },
    }))
}

// == the juari =====================================================================================

/// `_juari_snap`: a thread fraction to its nearest INTERIOR node.
fn juari_snap(frac: f64, n: i64) -> i64 {
    round_int(frac * n as f64).max(1).min(n - 1)
}

/// What `_build_juari` reports.
struct JuaInfo {
    c: f64,
    l: f64,
    n: i64,
    fs: f64,
    lam: f64,
    sigma0: f64,
}

/// `_build_juari`: the thread at grid `node`, grazing rest (or 1 m below it, for the clean run).
fn build_juari(p: &Value, node: i64, reach: bool) -> Result<(BarrierString, JuaInfo), Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", JUARI_LAM_DEFAULT)?;
    let k = fnum(p, "bridge_stiffness", JAWARI_K_DEFAULT)?;
    let sigma0 = fnum(p, "sigma0", JUARI_SIGMA0_DEFAULT)?;
    let n = requested_n(p)?;
    if !(N_MIN..=JUARI_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {JUARI_N_MAX}] for the juari, got {n}."
        )));
    }
    check_positive(l, t, rho)?;
    check_lambda(lam)?;
    if k <= 0.0 {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be positive, got {}.",
            pf(k)
        )));
    }
    if sigma0 < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma0 must be >= 0, got {}.",
            pf(sigma0)
        )));
    }
    if !(1..=n - 1).contains(&node) {
        return Err(Refusal::Param(format!(
            "thread node must be an interior node in [1, {}], got {node}.",
            n - 1
        )));
    }
    let c = (t / rho).sqrt();
    let fs = c * n as f64 / (l * lam);
    let string = damped_string(l, t, rho, fs, n, sigma0)?;
    let lam_s = string.p.lam;
    let mut barrier = vec![f64::NEG_INFINITY; string.p.nodes()];
    barrier[node as usize] = if reach { 0.0 } else { -1.0 };
    let bar = barrier_string(string, &barrier, k, JAWARI_ALPHA_DEFAULT)?;
    Ok((
        bar,
        JuaInfo {
            c,
            l,
            n,
            fs,
            lam: lam_s,
            sigma0,
        },
    ))
}

/// `_juari_pickup`: step, returning the pickup trace only.
fn juari_pickup(
    bar: &mut BarrierString,
    n_steps: usize,
    pickup_idx: usize,
) -> Result<Vec<f64>, Refusal> {
    let mut out = Vec::with_capacity(n_steps + 1);
    out.push(bar.string.u[pickup_idx]);
    for _ in 0..n_steps {
        bar.step();
        out.push(bar.string.u[pickup_idx]);
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }
    Ok(out)
}

/// `_juari_late_centroid`: the second half's spectral centroid.
fn late_centroid(pickup: &[f64], fs: f64) -> f64 {
    spectral_centroid(&pickup[pickup.len() / 2..], fs)
}

/// The juari's `N_req` pre-read: `int(N)` only when `str(N).lstrip("-")` is all digits, else 100.
fn juari_n_req(p: &Value) -> Result<i64, Refusal> {
    let v = p.get("N").cloned().unwrap_or(json!(100));
    let s = py_str(&v);
    let stripped = s.trim_start_matches('-');
    if !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit()) {
        py_int(&v).map_err(|_| {
            Refusal::Construction(format!(
                "invalid literal for int() with base 10: {}",
                crate::py::repr_str(&s)
            ))
        })
    } else {
        Ok(100)
    }
}

/// `_build_payload_juari`.
pub fn build_payload_juari(p: &Value) -> Result<Value, Refusal> {
    let f = read_front(p, JUARI_PICKUP_DEFAULT, 0.24, 0.06, JUARI_AMP_DEFAULT)?;
    let thread_frac = fnum(p, "thread_position", JUARI_THREAD_DEFAULT)?;
    let f = finish_front(f, p, FRAMES_PER_PERIOD)?;
    // The sweep duration is canonical; the tests override it to keep the ~11-run sweep fast.
    let sweep_dur = fnum(p, "sweep_duration", JUARI_SWEEP_DUR)?;
    if !(0.0 < sweep_dur && sweep_dur <= JUARI_SWEEP_DUR) {
        return Err(Refusal::Param(format!(
            "sweep_duration must be in (0, {}] s, got {}.",
            pf(JUARI_SWEEP_DUR),
            pf(sweep_dur)
        )));
    }
    check_front(&f, JUARI_AUDIO_MAX, JUARI_AMP_MAX)?;
    if !(0.0 < thread_frac && thread_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "thread_position must be in (0, 1), got {}.",
            pf(thread_frac)
        )));
    }

    let n_req = juari_n_req(p)?;
    let pre = if (N_MIN..=JUARI_N_MAX).contains(&n_req) {
        n_req
    } else {
        100
    };
    let sel_node = juari_snap(thread_frac, pre);
    let (mut main_bar, info) = build_juari(p, sel_node, true)?;
    let (fs, l, c, n) = (info.fs, info.l, info.c, info.n);
    let sel_node = juari_snap(thread_frac, n); // re-snap on the validated N
    let f1 = c / (2.0 * l);
    let mut nodes: Vec<i64> = JUARI_SWEEP_FRACS
        .iter()
        .map(|&fr| juari_snap(fr, n))
        .collect();
    nodes.push(sel_node);
    nodes.sort_unstable();
    nodes.dedup();

    let sweep_steps = round_int(sweep_dur * fs).max(1);
    let n_steps = round_int(f.audio_dur * fs).max(1);
    let total = (nodes.len() as i64 + 2) * sweep_steps + 2 * n_steps;
    if total > JUARI_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs {total} steps (budget {JUARI_WORK_MAX}): the tuning-curve \
             sweep is {} thread positions at the settled-buzz duration. Lower N, raise lambda, or \
             shorten audio_duration.",
            nodes.len()
        )));
    }
    let pickup_idx = pickup_index(f.pickup_frac, n);

    // The canonical clean baseline at the sweep duration.
    let (mut clean_sweep, _) = build_juari(p, sel_node, false)?;
    pluck_mode1(&mut clean_sweep, f.amplitude);
    let clean_late_sweep = late_centroid(
        &juari_pickup(&mut clean_sweep, sweep_steps as usize, pickup_idx)?,
        fs,
    );

    // The jawari reference — a CONVENIENCE overlay: any failure drops the guide line, never the
    // render (the reference caught ParamError and ValueError here).
    let jawari_ref = (|| -> Result<Option<f64>, Refusal> {
        let (mut jr, _) = build_jawari(p, 0.0)?;
        pluck_mode1(&mut jr, f.amplitude);
        let late = late_centroid(
            &juari_pickup(&mut jr, sweep_steps as usize, pickup_idx)?,
            fs,
        );
        Ok((clean_late_sweep > 0.0).then(|| round_nd(late / clean_late_sweep, 3)))
    })()
    .unwrap_or(None);

    // The tuning curve.
    let n_tune = requested_n(p)?;
    let l_tune = fnum(p, "L", 1.0)?;
    let (mut fracs, mut xs, mut elev) = (Vec::new(), Vec::new(), Vec::new());
    for &node in &nodes {
        let (mut bar, _) = build_juari(p, node, true)?;
        pluck_mode1(&mut bar, f.amplitude);
        let cen = late_centroid(
            &juari_pickup(&mut bar, sweep_steps as usize, pickup_idx)?,
            fs,
        );
        fracs.push(round_nd(node as f64 / n_tune as f64, 4));
        xs.push(round_nd(node as f64 / n_tune as f64 * l_tune, 4));
        elev.push(if clean_late_sweep > 0.0 {
            round_nd(cen / clean_late_sweep, 3)
        } else {
            f64::NAN
        });
    }

    // The main run and its own clean contrast, at the audio duration.
    let (stride, n_anim) = anim_plan(fs, f1, f.fpp, n_steps, f.anim_win);
    pluck_mode1(&mut main_bar, f.amplitude);
    let run = run_barrier(
        &mut main_bar,
        n_steps as usize,
        pickup_idx,
        stride as usize,
        n_anim as usize,
        true,
    )?;
    let (mut clean_audio, _) = build_juari(p, sel_node, false)?;
    pluck_mode1(&mut clean_audio, f.amplitude);
    let clean_run = run_barrier(&mut clean_audio, n_steps as usize, pickup_idx, 1, 0, false)?;
    let com = common_half(&run, fs);

    // -- the signature block
    let juari_late = late_centroid(&run.pickup, fs);
    let clean_late = late_centroid(&clean_run.pickup, fs);
    let sel_i = nodes
        .iter()
        .position(|&x| x == sel_node)
        .expect("sel_node was added");
    let sel_elev = elev[sel_i];
    // `np.nanargmax`: the first maximum, NaNs ignored.
    let peak_i = (0..elev.len())
        .filter(|&i| !elev[i].is_nan())
        .fold(None, |b: Option<usize>, i| match b {
            Some(j) if elev[j] >= elev[i] => Some(j),
            _ => Some(i),
        })
        .ok_or_else(|| Refusal::Construction("All-NaN slice encountered".into()))?;
    let near_nut_nodes = round_int(JUARI_NEAR_NUT_FRAC * n as f64).max(1).min(n - 1);
    let swept_near_nut = fracs
        .iter()
        .filter(|&&fr| fr <= JUARI_NEAR_NUT_FRAC)
        .count();
    let f_max = (0.45 * fs).min((20.0 * f1).max(2000.0));
    let half = run.pickup.len() / 2;
    let (js, cs) = shared_spectra(&run.pickup[half..], &clean_run.pickup[half..], fs, f_max);
    let nodes_json: Vec<Value> = nodes.iter().map(|&x| int(x)).collect();
    let tuning = json!({
        "node": nodes_json,
        "frac": fracs.iter().map(|&v| num(v)).collect::<Vec<_>>(),
        "x": xs.iter().map(|&v| num(v)).collect::<Vec<_>>(),
        "elevation": elev.iter().map(|&v| num(v)).collect::<Vec<_>>(),
    });
    let sig = json!({
        "kind": "juari",
        "tuning": tuning,
        "thread": {
            "node": int(sel_node),
            "frac": num(round_nd(sel_node as f64 / n as f64, 4)),
            "x": num(round_nd(sel_node as f64 / n as f64 * l, 4)),
            "elevation": num(sel_elev),
        },
        "sweet_spot": {
            "node": int(nodes[peak_i]),
            "frac": num(fracs[peak_i]),
            "x": num(xs[peak_i]),
            "elevation": num(elev[peak_i]),
        },
        "elevation": num(sel_elev),
        "elevation_gate": num(JUARI_ELEVATION_GATE),
        "buzzing": sel_elev > JUARI_ELEVATION_GATE,
        "reference": {"clean": num(1.0), "jawari": opt_num(jawari_ref)},
        "centroid": {
            "juari_late": num(round_nd(juari_late, 1)),
            "clean_late": num(round_nd(clean_late, 1)),
        },
        "quantization": {
            "h": num(round_nd(l / n as f64, 5)),
            "near_nut_frac": num(JUARI_NEAR_NUT_FRAC),
            "near_nut_nodes": int(near_nut_nodes),
            "swept_near_nut": int(swept_near_nut as i64),
        },
        "spectra": {"f_max": num(round_nd(f_max, 1)), "juari": js, "clean": cs},
        "f1": num(round_nd(f1, 3)),
    });
    let x = main_bar.string.p.grid();
    let contact_frames: Vec<Value> = run
        .frame_steps
        .iter()
        .map(|&i| int(i64::from(run.n_contact[i] > 0.0)))
        .collect();
    Ok(json!({
        "model": "juari",
        "horizon": string_block(&damped_info(&main_bar.string), "the string under the thread"),
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(info.lam, 6)),
        "grid": {
            "x": finite_list(&x, Some(6)),
            "thread_node": int(sel_node),
            "thread_x": num(round_nd(x[sel_node as usize], 6)),
        },
        "frames": com.frames,
        "frame_times": com.frame_times,
        "contact_frames": contact_frames,
        "anim_dt": num(stride as f64 / fs),
        "playback_speed": num(f.playback_speed),
        "field_amp": num(com.field_amp),
        "audio": com.audio,
        "energy": energy_block(&com.energy_time, &run.e, info.sigma0 == 0.0, 2.0 * info.sigma0,
                               EnergyOpts::default()),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(run.frames.len() as i64),
            "probe_x": num(round_nd(x[pickup_idx], 4)),
            "spectrum": sig,
        },
    }))
}

// == the fret ======================================================================================

/// What `_build_fret` reports.
struct FretInfo {
    c: f64,
    l: f64,
    n: i64,
    fs: f64,
    lam: f64,
    sigma0: f64,
    clearance: f64,
    rail_frac: f64,
    support: usize,
}

/// `_build_fret`: a flat rail at `-clearance` under `0 < x <= rail_frac L` (1 m down when out of
/// reach, the control).
fn build_fret(p: &Value, out_of_reach: bool) -> Result<(BarrierString, FretInfo), Refusal> {
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", FRET_LAM_DEFAULT)?;
    let k = fnum(p, "rail_stiffness", FRET_K_DEFAULT)?;
    let clearance = fnum(p, "clearance", FRET_CLEARANCE_DEFAULT)?;
    let rail_frac = fnum(p, "rail_frac", FRET_RAIL_FRAC_DEFAULT)?;
    let sigma0 = fnum(p, "sigma0", FRET_SIGMA0_DEFAULT)?;
    let n = requested_n(p)?;
    if !(N_MIN..=FRET_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {FRET_N_MAX}] for the fret, got {n}."
        )));
    }
    check_positive(l, t, rho)?;
    check_lambda(lam)?;
    if !(0.0 < clearance && clearance <= FRET_CLEARANCE_MAX) {
        return Err(Refusal::Param(format!(
            "clearance must be in (0, {}] m, got {}.",
            pf(FRET_CLEARANCE_MAX),
            pf(clearance)
        )));
    }
    if !(FRET_RAIL_FRAC_MIN..=1.0).contains(&rail_frac) {
        return Err(Refusal::Param(format!(
            "rail_frac must be in [{}, 1.0], got {}: a shorter rail sits under a smaller share of \
             the string's swing, and below ~0.15 nothing touches it.",
            pf(FRET_RAIL_FRAC_MIN),
            pf(rail_frac)
        )));
    }
    if k <= 0.0 {
        return Err(Refusal::Param(format!(
            "rail_stiffness must be positive, got {}.",
            pf(k)
        )));
    }
    if sigma0 < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma0 must be >= 0, got {}.",
            pf(sigma0)
        )));
    }
    let c = (t / rho).sqrt();
    let fs = c * n as f64 / (l * lam);
    let string = damped_string(l, t, rho, fs, n, sigma0)?;
    let lam_s = string.p.lam;
    let depth = if out_of_reach { 1.0 } else { clearance };
    let span = rail_frac * l;
    let b: Vec<f64> = string
        .p
        .grid()
        .iter()
        .map(|&x| {
            if x > 0.0 && x <= span {
                -depth
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let bar = barrier_string(string, &b, k, FRET_ALPHA)?;
    let support = bar.p.support_len();
    Ok((
        bar,
        FretInfo {
            c,
            l,
            n,
            fs,
            lam: lam_s,
            sigma0,
            clearance,
            rail_frac,
            support,
        },
    ))
}

/// `_FretRun`: per-step telemetry, with the contact raster reduced as it is produced.
struct FretRun {
    base: BarRun,
    n_active: Vec<f64>,
    iters: Vec<f64>,
    /// `m` rows x `n_cols` columns, row-major.
    raster: Vec<f64>,
    m: usize,
    n_cols: usize,
    equi_sum: f64,
    equi_count: usize,
}

/// `_run_fret`.
#[allow(clippy::too_many_arguments)]
fn run_fret(
    bar: &mut BarrierString,
    n_steps: usize,
    pickup_idx: usize,
    anim_stride: usize,
    frame_until: usize,
    n_cols: usize,
    capture: bool,
) -> Result<FretRun, Refusal> {
    let m = if capture { bar.p.support_len() } else { 0 };
    let n_cols = if capture { n_cols } else { 0 };
    let mut r = FretRun {
        base: BarRun {
            e: Vec::with_capacity(n_steps + 1),
            pickup: Vec::with_capacity(n_steps + 1),
            wrap: Vec::new(),
            n_contact: Vec::new(),
            frames: Vec::new(),
            frame_steps: Vec::new(),
        },
        n_active: vec![0.0; n_steps + 1],
        iters: vec![0.0; n_steps + 1],
        raster: vec![0.0; m * n_cols],
        m,
        n_cols,
        equi_sum: 0.0,
        equi_count: 0,
    };
    let (rho, h, k2) = (bar.string.p.rho, bar.string.p.h, 2.0 * bar.p.contact.k);
    let sample = |bar: &BarrierString, r: &mut FretRun, i: usize| {
        r.base.e.push(bar.energy());
        r.base.pickup.push(bar.string.u[pickup_idx]);
        if capture {
            let mask = bar.contact_mask();
            let na = mask.iter().filter(|&&b| b).count();
            r.n_active[i] = na as f64;
            r.iters[i] = bar.s.newton_iters as f64;
            if na > 0 {
                let col = (r.n_cols - 1).min((i * r.n_cols) / (n_steps + 1));
                for (row, (&on, f)) in mask.iter().zip(&bar.s.contact_force).enumerate() {
                    let v = if on { f.abs() } else { 0.0 };
                    let cell = &mut r.raster[row * r.n_cols + col];
                    // `np.maximum`: NaN-propagating.
                    if v > *cell || v.is_nan() {
                        *cell = v;
                    }
                }
            }
        }
    };
    sample(bar, &mut r, 0);
    if frame_until >= 1 {
        r.base.frames.push(bar.string.u.clone());
        r.base.frame_steps.push(0);
    }
    for i in 1..=n_steps {
        // u^{i-2}, saved before the roll: the CENTERED velocity at level i-1.
        let u_prev2 = bar.string.u_prev.clone();
        bar.step();
        sample(bar, &mut r, i);
        if capture && r.base.e[i - 1] > 0.0 {
            let u = &bar.string.u;
            let v_c: Vec<f64> = (1..u.len() - 1).map(|j| (u[j] - u_prev2[j]) / k2).collect();
            let ke_c = 0.5 * rho * h * dot(&v_c, &v_c);
            r.equi_sum += 2.0 * ke_c / r.base.e[i - 1];
            r.equi_count += 1;
        }
        if i <= frame_until && i % anim_stride == 0 {
            r.base.frames.push(bar.string.u.clone());
            r.base.frame_steps.push(i);
        }
    }
    if !r.base.e.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }
    Ok(r)
}

/// `_fret_episodes`: contact episodes, merging any two separated by fewer than `min_gap` samples.
fn fret_episodes(contact: &[bool], min_gap: usize) -> usize {
    let on: Vec<usize> = (0..contact.len() - 1)
        .filter(|&i| contact[i + 1] && !contact[i])
        .collect();
    if on.is_empty() {
        return 0;
    }
    let mut off: Vec<usize> = (0..contact.len() - 1)
        .filter(|&i| !contact[i + 1] && contact[i])
        .collect();
    if !off.is_empty() && off[0] < on[0] {
        off.remove(0);
    }
    let mut n = 1;
    for i in 1..on.len() {
        let prev_off = if i - 1 < off.len() { off[i - 1] } else { on[i] };
        if on[i] as i64 - prev_off as i64 >= min_gap as i64 {
            n += 1;
        }
    }
    n
}

/// `_fret_contact_block`: the GATED claim — slap-and-release, and a genuinely vector solve.
fn contact_block(
    r: &FretRun,
    info: &FretInfo,
    fs: f64,
    f1: f64,
    n_steps: i64,
    support_x: &[f64],
) -> Value {
    let contact: Vec<bool> = r.n_active.iter().map(|&a| a > 0.0).collect();
    let n_periods = n_steps as f64 / (fs / f1);
    let min_gap = ((FRET_DEBOUNCE_FRAC * fs / f1) as usize).max(1);
    let episodes = fret_episodes(&contact, min_gap);
    let raw_onsets = (0..contact.len() - 1)
        .filter(|&i| contact[i + 1] && !contact[i])
        .count();
    let duty = contact.iter().filter(|&&b| b).count() as f64 / contact.len() as f64;
    let per_period = if n_periods > 0.0 {
        episodes as f64 / n_periods
    } else {
        0.0
    };
    let touching: Vec<f64> = (0..contact.len())
        .filter(|&i| contact[i])
        .map(|i| r.n_active[i])
        .collect();

    // The image (and ONLY the image) is binned when the support outruns the raster's rows.
    let (mut rows, cols) = (r.m, r.n_cols);
    let mut raster = r.raster.clone();
    let x_binned = rows > FRET_RASTER_MAX_ROWS;
    if x_binned {
        let edges = linspace_idx(rows, FRET_RASTER_MAX_ROWS + 1);
        let mut out = Vec::new();
        let mut new_rows = 0;
        for e in edges.windows(2) {
            let (a, b) = (e[0], e[1]);
            if b > a {
                for col in 0..cols {
                    let mut mx = f64::NEG_INFINITY;
                    for row in a..b {
                        mx = mx.max(raster[row * cols + col]);
                    }
                    out.push(mx);
                }
                new_rows += 1;
            }
        }
        raster = out;
        rows = new_rows;
    }
    let fmax = if raster.is_empty() {
        0.0
    } else {
        raster.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    let img: Vec<u8> = raster
        .iter()
        .map(|&v| {
            if fmax > 0.0 && v > 0.0 {
                // Any contact stays >= 1, so the grey never rounds a genuine touch away.
                (255.0 * v / fmax).round_ties_even().max(1.0) as u8
            } else {
                0
            }
        })
        .collect();
    let col_max = |a: &[f64]| -> Value {
        let edges = linspace_idx(a.len(), r.n_cols + 1);
        let v: Vec<f64> = edges
            .windows(2)
            .filter(|e| e[1] > e[0])
            .map(|e| {
                a[e[0]..e[1]]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        finite_list(&v, Some(3))
    };
    let iters_after: Vec<f64> = r.iters[1..].to_vec();
    let has_img = !img.is_empty();
    json!({
        "kind": "fret",
        "duty": num(round_nd(duty, 4)),
        "episodes": int(episodes as i64),
        "raw_onsets": int(raw_onsets as i64),
        "episodes_per_period": num(round_nd(per_period, 3)),
        "episodes_min": num(FRET_EPISODES_MIN),
        "duty_max": num(FRET_DUTY_MAX),
        "intermittent": duty > 0.0 && per_period >= FRET_EPISODES_MIN && duty <= FRET_DUTY_MAX,
        "out_of_reach": duty == 0.0,
        "pinned": duty >= FRET_DUTY_MAX,
        "support": int(info.support as i64),
        "active_max": int(r.n_active.iter().copied().fold(0.0, f64::max) as i64),
        "active_mean_touching": if touching.is_empty() {
            Value::Null
        } else {
            num(round_nd(np_mean(&touching), 2))
        },
        "iters_max": int(r.iters.iter().copied().fold(0.0, f64::max) as i64),
        "iters_mean": if n_steps > 0 { num(round_nd(np_mean(&iters_after), 3)) } else { Value::Null },
        "raster": {
            "b64": b64encode(&img),
            "n_rows": int(if has_img { rows as i64 } else { 0 }),
            "n_cols": int(if has_img { cols as i64 } else { 0 }),
            "x_binned": x_binned,
            "x0": num(round_nd(support_x[0], 4)),
            "x1": num(round_nd(support_x[support_x.len() - 1], 4)),
            "t0": num(0.0),
            "t1": num(round_nd(n_steps as f64 / fs, 6)),
            "force_max": num(round_nd(fmax, 3)),
            "cols_per_period": if n_periods > 0.0 {
                num(round_nd(r.n_cols as f64 / n_periods, 2))
            } else {
                Value::Null
            },
        },
        "trace": {"active": col_max(&r.n_active), "iters": col_max(&r.iters)},
        "n_periods": num(round_nd(n_periods, 2)),
    })
}

/// `_fret_decay_triple`: rate, `2 sigma0`, and `2 sigma0 <2KE/E>` side by side — never gated.
fn decay_triple(r: &FretRun, fs: f64, sigma0: f64) -> Value {
    let t: Vec<f64> = (0..r.base.e.len()).map(|i| i as f64 / fs).collect();
    let rate = fit_decay(&t, &r.base.e);
    let equi = (r.equi_count > 0).then(|| r.equi_sum / r.equi_count as f64);
    let corrected = equi.map(|e| 2.0 * sigma0 * e);
    // Python's truthiness: None and 0.0 are both false.
    let rate_t = rate.filter(|&v| v != 0.0);
    json!({
        "rate": opt_num(rate),
        "oracle_2sigma": num(2.0 * sigma0),
        "equipartition": opt_num(equi.map(|e| round_nd(e, 5))),
        "corrected": opt_num(corrected),
        "ratio": opt_num(rate_t.filter(|_| sigma0 > 0.0).map(|v| round_nd(v / (2.0 * sigma0), 4))),
        "agreement": opt_num(match (rate_t, corrected) {
            (Some(rv), Some(cv)) if cv != 0.0 && cv > 0.0 => Some(round_nd((rv - cv).abs() / cv, 5)),
            _ => None,
        }),
    })
}

/// `_fret_signature_block`: brightness elevation and the crossing rate — both diagnostic.
fn signature_block(
    fret_pickup: &[f64],
    ctrl_pickup: &[f64],
    fs: f64,
    f1: f64,
    info: &FretInfo,
) -> Value {
    let f_bright = centroid_below(fret_pickup, fs, FRET_CENTROID_FMAX);
    let c_bright = centroid_below(ctrl_pickup, fs, FRET_CENTROID_FMAX);
    let elevation = if c_bright > 0.0 {
        f_bright / c_bright
    } else {
        f64::NAN
    };
    let centred = |a: &[f64]| {
        let m = np_mean(a);
        a.iter().map(|v| v - m).collect::<Vec<f64>>()
    };
    let f_cross = interp_zero_cross_frequency(&centred(fret_pickup), fs);
    let c_cross = interp_zero_cross_frequency(&centred(ctrl_pickup), fs);
    let cents = (f_cross > 0.0 && c_cross > 0.0).then(|| 1200.0 * (f_cross / c_cross).log2());
    json!({
        "kind": "fret",
        "centroid_fret": num(round_nd(f_bright, 1)),
        "centroid_control": num(round_nd(c_bright, 1)),
        "elevation": num(round_nd(elevation, 3)),
        "f1": num(round_nd(f1, 3)),
        "clearance": num(info.clearance),
        "peak_clearance": num(FRET_BRIGHTNESS_PEAK),
        "monotone": false,
        "crossing_rate": num(round_nd(f_cross, 2)),
        "crossing_rate_control": num(round_nd(c_cross, 2)),
        "crossing_cents": opt_num(cents.map(|v| round_nd(v, 1))),
        "crossing_is_pitch": false,
        "static_oracle": {"claim": "S u* = (K/rho) b", "residual": num(3.4e-15), "alpha": num(1.0)},
    })
}

/// `_build_payload_fret`.
pub fn build_payload_fret(p: &Value) -> Result<Value, Refusal> {
    let f = read_front(p, FRET_PICKUP_DEFAULT, 0.4, 0.1, FRET_AMP_DEFAULT)?;
    let f = finish_front(f, p, FRET_FRAMES_PER_PERIOD)?;
    check_front(&f, FRET_AUDIO_MAX, FRET_AMP_MAX)?;

    let (mut bar, info) = build_fret(p, false)?;
    let (fs, l, c, n) = (info.fs, info.l, info.c, info.n);
    let f1 = c / (2.0 * l);
    let n_steps = round_int(f.audio_dur * fs).max(1);
    let n_control = round_int(f.audio_dur.min(FRET_CONTROL_MAX) * fs).max(1);
    if n_steps + n_control > FRET_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs {} steps across the fret run and its out-of-reach control \
             (budget {FRET_WORK_MAX}): every step is a vector contact solve over up to {} rail \
             nodes. Shorten audio_duration, lower N, or raise lambda.",
            n_steps + n_control,
            info.support
        )));
    }
    let pickup_idx = pickup_index(f.pickup_frac, n);
    let (stride, n_anim) = anim_plan(fs, f1, f.fpp, n_steps, f.anim_win);
    let n_periods = (n_steps as f64 / (fs / f1)).max(1e-9);
    let n_cols = FRET_RASTER_MAX_COLS
        .min((round_int(FRET_RASTER_COLS_PER_PERIOD * n_periods) as f64).max(16.0))
        as usize;

    pluck_mode1(&mut bar, f.amplitude);
    let run = run_fret(
        &mut bar,
        n_steps as usize,
        pickup_idx,
        stride as usize,
        n_anim as usize,
        n_cols,
        true,
    )?;
    let (mut ctrl, _) = build_fret(p, true)?;
    pluck_mode1(&mut ctrl, f.amplitude);
    let ctrl_run = run_fret(&mut ctrl, n_control as usize, pickup_idx, 1, 0, 0, false)?;

    let com = common_half(&run.base, fs);
    let x = bar.string.p.grid();
    let support_x: Vec<f64> = bar.p.support.iter().map(|&i| x[i]).collect();
    let mut energy = energy_block(
        &com.energy_time,
        &run.base.e,
        info.sigma0 == 0.0,
        2.0 * info.sigma0,
        EnergyOpts {
            no_decay_oracle: true,
            ..EnergyOpts::default()
        },
    );
    if let Value::Object(m) = &mut energy {
        m.insert("decay_triple".into(), decay_triple(&run, fs, info.sigma0));
    }
    let mut meta = Map::new();
    meta.insert("c".into(), num(round_nd(c, 3)));
    meta.insert("f1".into(), num(round_nd(f1, 3)));
    meta.insert("num_steps".into(), int(n_steps));
    meta.insert("n_control_steps".into(), int(n_control));
    meta.insert("n_frames".into(), int(run.base.frames.len() as i64));
    meta.insert("probe_x".into(), num(round_nd(x[pickup_idx], 4)));
    meta.insert("clearance".into(), num(info.clearance));
    meta.insert("rail_frac".into(), num(info.rail_frac));
    meta.insert("rail_span".into(), num(round_nd(info.rail_frac * l, 4)));
    meta.insert("amplitude".into(), num(f.amplitude));
    meta.insert(
        "contact".into(),
        contact_block(&run, &info, fs, f1, n_steps, &support_x),
    );
    meta.insert(
        "spectrum".into(),
        signature_block(&run.base.pickup, &ctrl_run.base.pickup, fs, f1, &info),
    );
    Ok(json!({
        "model": "fret",
        "horizon": string_block(&damped_info(&bar.string), "the fretted string"),
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(info.lam, 6)),
        "grid": {"x": finite_list(&x, Some(6)), "barrier": finite_list(&barrier_on_grid(&bar), None)},
        "frames": com.frames,
        "frame_times": com.frame_times,
        "anim_dt": num(stride as f64 / fs),
        "playback_speed": num(f.playback_speed),
        "field_amp": num(com.field_amp),
        "audio": com.audio,
        "energy": energy,
        "meta": meta,
    }))
}
