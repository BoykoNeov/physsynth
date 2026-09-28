//! A string terminated on a distributed plate body, radiating — `serialize.py`'s platebody builder.
//!
//! The body scene with its lumped modal body swapped for a grid plate (the supported soundboard or
//! the free cymbal), so the third stage of exciter -> resonator -> body finally has a PICTURE: the
//! plate lighting up as the pluck's energy transfers into it. The coupling and its exact stability
//! guard live in the core (`connection::StringPlateBridge`); the guard firing is a clean
//! construction error. Two read-outs are measured decisions of the reference:
//!
//! * the terminus fundamental comes from a DEDICATED near-nut probe, because both the user's pluck
//!   and the free end mislead once the end pins;
//! * the `omega²` sanity check divides by the plate's VOLUME displacement, not the driving point.

use physsynth_core::connection::StringPlateBridge;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::fmt::py_float;
use physsynth_core::plate::{self as pl, Boundary as PlateBoundary, Plate, PlateSpec};
use physsynth_core::radiation::{AirParams, AirRadiation};
use physsynth_core::reduce::sum;
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use serde_json::{json, Value};

use crate::body::{np_extreme, omega2_consistency, opt_finite, pooled_spectrum, terminus_f1};
use crate::energy::{energy_block, EnergyOpts};
use crate::membrane::decimate_field_mask;
use crate::plate::discrete_eigenfreqs;
use crate::py::{
    b64f32, b64u8, dot, finite_list, int, linspace_idx, max_abs, num, py_int, py_repr, py_str,
    repr_str, round_int, round_nd,
};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::{
    construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN, SPEED_MAX,
};
use crate::tension::N_SPEC_POINTS;
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// String grid ceiling.
pub const PLATEBODY_NSTRING_MAX: i64 = 160;
/// Plate grid range.
pub const PLATEBODY_NPLATE_MIN: i64 = 8;
/// See [`PLATEBODY_NPLATE_MIN`].
pub const PLATEBODY_NPLATE_MAX: i64 = 24;
/// Default Courant number (below 1: the bridge spring pushes the Nyquist mode).
pub const PLATEBODY_LAM_DEFAULT: f64 = 0.9;
/// Default bridge spring (N/m): visible slosh, well under the guard's ceiling.
pub const PLATEBODY_K_DEFAULT: f64 = 3000.0;
/// Slider cap; the exact guard is the real gate.
pub const PLATEBODY_K_MAX: f64 = 12000.0;
/// Plate loss ceiling.
pub const PLATEBODY_SIGMA_MAX: f64 = 80.0;
/// Listener distance default (m).
pub const PLATEBODY_DISTANCE_DEFAULT: f64 = 1.0;
/// Listener distance ceiling (m).
pub const PLATEBODY_DISTANCE_MAX: f64 = 8.0;
/// Pluck amplitude default (m).
pub const PLATEBODY_AMP_DEFAULT: f64 = 1e-3;
/// Longest audio, seconds.
pub const PLATEBODY_AUDIO_MAX: f64 = 3.0;
/// `n_live x steps` backstop.
pub const PLATEBODY_WORK_MAX: f64 = 1.0e8;
/// Seconds of string+plate animation (the plate rings fast).
pub const PLATEBODY_ANIM_WIN: f64 = 0.03;
/// Seconds of the exchange slosh plotted.
pub const PLATEBODY_EXCHANGE_WINDOW: f64 = 0.4;
/// Exchange trace length.
pub const PLATEBODY_TRACE_POINTS: usize = 600;
/// Plate stiffness, fixed server-side.
pub const PLATEBODY_KAPPA: f64 = 20.0;
/// Plate areal density (kg/m²).
pub const PLATEBODY_RHO_PLATE: f64 = 0.005;
/// Poisson's ratio (free edge only).
pub const PLATEBODY_NU: f64 = 0.3;
/// Square plate side (m).
pub const PLATEBODY_SIDE: f64 = 1.0;
/// Near-nut string pickup for the terminus fundamental.
pub const PLATEBODY_PICKUP_FRAC: f64 = 0.23;
/// Near-nut pluck for the terminus probe.
pub const PLATEBODY_TERMINUS_PLUCK: f64 = 0.137;
/// The terminus probe's fixed window (s).
pub const PLATEBODY_TERMINUS_SECS: f64 = 0.6;
/// Low plate modes marked on the spectrum (not scored).
pub const PLATEBODY_MARKER_MODES: usize = 6;

fn pf(x: f64) -> String {
    py_float(x)
}

/// Python's `min` over a sequence.
fn py_min(vals: &[f64]) -> f64 {
    let mut m = vals[0];
    for &v in &vals[1..] {
        if v < m {
            m = v;
        }
    }
    m
}

type Bridge = StringPlateBridge<Plate>;

/// What `_build_platebody_bridge` reports beside the bridge.
struct Info {
    c: f64,
    l: f64,
    n: i64,
    n_plate: i64,
    fs: f64,
    lam: f64,
    k: f64,
    sigma_plate: f64,
    boundary: PlateBoundary,
    n_live: usize,
}

/// `_build_platebody_bridge`.
fn build(p: &Value) -> Result<(Bridge, Info), Refusal> {
    let dv = p.get("domain").cloned().unwrap_or(json!("free"));
    let dom = py_str(&dv);
    if dom != "supported" && dom != "free" {
        return Err(Refusal::Param(format!(
            "boundary must be 'supported' or 'free', got {}.",
            repr_str(&dom)
        )));
    }
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", PLATEBODY_LAM_DEFAULT)?;
    let k = fnum(p, "bridge_stiffness", PLATEBODY_K_DEFAULT)?;
    let sigma_plate = fnum(p, "sigma_plate", 0.0)?;
    let nv = p.get("N").cloned().unwrap_or(json!(100));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    let npv = p.get("n_plate").cloned().unwrap_or(json!(16));
    let n_plate = py_int(&npv).map_err(|_| {
        Refusal::Param(format!(
            "n_plate must be an integer, got {}.",
            py_repr(&npv)
        ))
    })?;

    if !(N_MIN..=PLATEBODY_NSTRING_MAX).contains(&n) {
        // the reference's message has no closing period
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {PLATEBODY_NSTRING_MAX}] for the string, got {n}"
        )));
    }
    if !(PLATEBODY_NPLATE_MIN..=PLATEBODY_NPLATE_MAX).contains(&n_plate) {
        return Err(Refusal::Param(format!(
            "n_plate must be in [{PLATEBODY_NPLATE_MIN}, {PLATEBODY_NPLATE_MAX}], got {n_plate}."
        )));
    }
    if py_min(&[l, t, rho]) <= 0.0 {
        return Err(Refusal::Param("L, T, rho must all be positive.".into()));
    }
    if !(0.0 < lam && lam < 1.0) {
        return Err(Refusal::Param(format!(
            "lambda must be in (0, 1), got {}: the string's Nyquist mode is marginal at lambda = \
             1 and the bridge spring pushes it unstable, so the coupled system needs headroom \
             below it.",
            pf(lam)
        )));
    }
    if k < 0.0 {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be >= 0, got {}.",
            pf(k)
        )));
    }
    if !(0.0..=PLATEBODY_SIGMA_MAX).contains(&sigma_plate) {
        return Err(Refusal::Param(format!(
            "sigma_plate must be in [0, {}], got {}.",
            pf(PLATEBODY_SIGMA_MAX),
            pf(sigma_plate)
        )));
    }
    let c = (t / rho).sqrt();
    let fs = c * n as f64 / (l * lam);
    let sp = ideal::Params::new(
        l,
        t,
        rho,
        fs,
        n,
        0.0,
        Some((Boundary::Fixed, Boundary::Free)),
    )
    .map_err(construction)?;
    let string = IdealString::new(sp);
    let lam_s = string.params().lam;
    let boundary = if dom == "free" {
        PlateBoundary::Free
    } else {
        PlateBoundary::Supported
    };
    // `nu` is passed to the free plate only, as the reference's two constructor calls do.
    let spec = PlateSpec {
        lx: PLATEBODY_SIDE,
        ly: PLATEBODY_SIDE,
        kappa: PLATEBODY_KAPPA,
        rho: PLATEBODY_RHO_PLATE,
        fs,
        n: n_plate,
        sigma: sigma_plate,
        boundary: Some(boundary),
        nu: (boundary == PlateBoundary::Free).then_some(PLATEBODY_NU),
        ..PlateSpec::default()
    };
    let plate = Plate::new(pl::Params::new(&spec).map_err(construction)?);
    let n_live = plate.p.n_live;
    // The exact stability guard fires here, as a construction error.
    let bridge = StringPlateBridge::new(string, plate, k, None).map_err(construction)?;
    Ok((
        bridge,
        Info {
            c,
            l,
            n,
            n_plate,
            fs,
            lam: lam_s,
            k,
            sigma_plate,
            boundary,
            n_live,
        },
    ))
}

/// `_platebody_qvol`: the plate's VOLUME displacement — `h² sum(u)` supported, `w . u` free.
fn qvol(plate: &Plate) -> f64 {
    match plate.p.boundary {
        PlateBoundary::Supported => plate.p.h * plate.p.h * sum(&plate.u),
        PlateBoundary::Free => dot(&plate.p.w, &plate.u),
    }
}

/// `_platebody_terminus_f1`: the coupled fundamental from a dedicated near-nut-pluck probe.
fn terminus_probe(p: &Value, c: f64, l: f64, fs: f64, n_string: i64) -> Result<f64, Refusal> {
    let (mut bridge, _) = build(p)?;
    let x = bridge.string().params().grid();
    let u0 = triangular_pluck(&x, l, PLATEBODY_TERMINUS_PLUCK * l, PLATEBODY_AMP_DEFAULT)
        .map_err(construction)?;
    bridge.string_mut().set_state(&u0, &vec![0.0; u0.len()]);
    let n = round_int(PLATEBODY_TERMINUS_SECS * fs).max(1) as usize;
    let idx = round_int(PLATEBODY_PICKUP_FRAC * n_string as f64) as usize;
    let mut pick = Vec::with_capacity(n + 1);
    pick.push(bridge.string().u[idx]);
    for _ in 0..n {
        let _ = bridge.step(&mut ());
        pick.push(bridge.string().u[idx]);
    }
    Ok(terminus_f1(&pick, fs, c, l))
}

/// `_build_payload_platebody`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let amplitude = fnum(p, "amplitude", PLATEBODY_AMP_DEFAULT)?;
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let distance = fnum(p, "distance", PLATEBODY_DISTANCE_DEFAULT)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;
    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    if !(0.0 < pluck_frac && pluck_frac < 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_position must be in (0, 1), got {}.",
            pf(pluck_frac)
        )));
    }
    if !(0.0 < audio_dur && audio_dur <= PLATEBODY_AUDIO_MAX) {
        // the reference's message has no closing period
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}",
            pf(PLATEBODY_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < distance && distance <= PLATEBODY_DISTANCE_MAX) {
        return Err(Refusal::Param(format!(
            "distance must be in (0, {}] m, got {}.",
            pf(PLATEBODY_DISTANCE_MAX),
            pf(distance)
        )));
    }

    let (mut bridge, info) = build(p)?;
    let (c, l, fs, lam) = (info.c, info.l, info.fs, info.lam);
    let f1_base = c / (2.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    let work = info.n_live as i64 * n_steps;
    if work as f64 > PLATEBODY_WORK_MAX {
        return Err(Refusal::Param(format!(
            "over budget: ~{:.0}M > {:.0}M node-steps. Lower the audio duration, n_plate, or N.",
            work as f64 / 1e6,
            PLATEBODY_WORK_MAX / 1e6
        )));
    }
    let mut anim_stride = round_int((fs / f1_base) / fpp as f64).max(1);
    let frame_until = anim_stride.max(round_int(PLATEBODY_ANIM_WIN * fs));
    if frame_until / anim_stride > MAX_FRAMES {
        anim_stride = ((frame_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }

    let x = bridge.string().params().grid();
    let u0 = triangular_pluck(&x, l, pluck_frac * l, amplitude).map_err(construction)?;
    bridge.string_mut().set_state(&u0, &vec![0.0; u0.len()]);
    let air_p = AirParams::new(fs, distance, RHO0_AIR, C0_AIR, true).map_err(construction)?;
    let (gain, latency, residual) = (
        air_p.gain,
        air_p.latency_samples,
        air_p.retardation_residual,
    );
    let mut air = AirRadiation::new(air_p);

    // -- one instrumented run: split energies, Q_vol, raw Q'', far field, BOTH fields
    let n = n_steps as usize;
    let mut ch: [Vec<f64>; 7] = Default::default(); // E, string, plate, conn, qvol, Q'', p
    let (mut frames_plate, mut frames_string, mut frame_steps) =
        (Vec::new(), Vec::new(), Vec::new());
    let k_spring = bridge.stiffness();
    let mut sample = |b: &Bridge, ch: &mut [Vec<f64>; 7]| {
        ch[0].push(b.energy());
        ch[1].push(b.string().energy());
        ch[2].push(b.plate().energy());
        ch[3].push(0.5 * k_spring * b.stretch(false) * b.stretch(true));
        ch[4].push(qvol(b.plate()));
        ch[5].push(b.pressure());
        ch[6].push(air.process(b.pressure()));
    };
    sample(&bridge, &mut ch);
    if frame_until >= 1 {
        frames_plate.push(bridge.plate().state());
        frames_string.push(bridge.string().u.clone());
        frame_steps.push(0usize);
    }
    for i in 1..=n {
        let _ = bridge.step(&mut ());
        sample(&bridge, &mut ch);
        if i as i64 <= frame_until && i as i64 % anim_stride == 0 {
            frames_plate.push(bridge.plate().state());
            frames_string.push(bridge.string().u.clone());
            frame_steps.push(i);
        }
    }
    let [total, e_string, e_plate, e_conn, qv, qaccel, pressure] = ch;
    if !total.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability); adjust parameters.".into(),
        ));
    }

    // -- the money panel: the E_string <-> E_plate exchange ("body" keys: the page's drawBodyEnergy)
    let frac = |a: &[f64]| {
        a.iter()
            .zip(&total)
            .map(|(x, t)| x / t)
            .collect::<Vec<f64>>()
    };
    let (es, ep, ec, ones) = (frac(&e_string), frac(&e_plate), frac(&e_conn), frac(&total));
    let n_exch = n.min(round_int(PLATEBODY_EXCHANGE_WINDOW * fs).max(1) as usize);
    let eidx = linspace_idx(n_exch, (n_exch + 1).min(PLATEBODY_TRACE_POINTS));
    let pick = |a: &[f64]| eidx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    let t_ex: Vec<f64> = eidx.iter().map(|&i| i as f64 / fs).collect();
    let e0 = total[0];
    let quarter = (n / 4).max(1);
    let mut first = 0;
    for i in 0..quarter {
        if e_plate[i] > e_plate[first] {
            first = i;
        }
    }
    let exchange = json!({
        "kind": "platebody",
        "time": finite_list(&t_ex, Some(6)),
        "e_string_frac": finite_list(&pick(&es), None),
        "e_body_frac": finite_list(&pick(&ep), None),
        "e_conn_frac": finite_list(&pick(&ec), None),
        "total_frac": finite_list(&pick(&ones), None),
        "window": num(round_nd(n_exch as f64 / fs, 4)),
        "body_frac_peak": num(round_nd(np_extreme(&ep, true), 4)),
        "string_frac_min": num(round_nd(np_extreme(&es, false), 4)),
        "string_frac_max": num(round_nd(np_extreme(&es, true), 4)),
        "first_peak_ms": num(round_nd(first as f64 / fs * 1000.0, 1)),
        "total_drift": num(if e0 != 0.0 {
            (np_extreme(&total, true) - np_extreme(&total, false)) / e0.abs()
        } else {
            f64::NAN
        }),
        "K": num(round_nd(info.k, 1)),
    });

    // -- the radiated-pressure spectrum and the per-boundary read-outs
    let f_max = (0.45 * fs).min((10.0 * f1_base).max(1500.0));
    let (sf, sm) = pooled_spectrum(&qaccel, fs, f_max, N_SPEC_POINTS);
    let mut modes = discrete_eigenfreqs(&bridge.plate().p, PLATEBODY_MARKER_MODES)?;
    modes.retain(|&f| f <= f_max);
    modes.truncate(PLATEBODY_MARKER_MODES);
    let boundary = match info.boundary {
        PlateBoundary::Supported => "supported",
        PlateBoundary::Free => "free",
    };
    let spectrum = json!({
        "kind": "platebody",
        "boundary": boundary,
        "f_max": num(round_nd(f_max, 1)),
        "f": finite_list(&sf, Some(3)),
        "mag": finite_list(&sm, None),
        "body_modes": finite_list(&modes, Some(1)),
        // `null` when the plate never moves (K = 0): the reference shipped NaN, which its own
        // strict JSON writer refused — the body scene's decision, carried (§23.10).
        "omega2_consistency": opt_finite(round_nd(omega2_consistency(&qaccel, &qv, fs, f_max), 3)),
        "terminus_f1": opt_finite(round_nd(terminus_probe(p, c, l, fs, info.n)?, 2)),
        "f1_free": num(round_nd(c / (4.0 * l), 2)),
        "f1_clamped": num(round_nd(c / (2.0 * l), 2)),
        "distance": num(round_nd(distance, 3)),
        "gain": num(gain),
        "gain_times_r": num(round_nd(gain * distance, 6)),
        "latency_ms": num(round_nd(latency as f64 / fs * 1000.0, 3)),
        "retardation_residual": num(round_nd(residual, 4)),
    });

    // -- frames (both fields), audio, the energy verdict
    let pp = &bridge.plate().p;
    let (ny, nx) = (pp.mask.nrows(), pp.mask.ncols());
    let (plate_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&frames_plate, pp.mask.flags(), ny, nx);
    let nf = frames_plate.len();
    let field_amp = if plate_dec.is_empty() {
        0.0
    } else {
        max_abs(&plate_dec)
    };
    let str_flat: Vec<f64> = frames_string.iter().flatten().copied().collect();
    let str_amp = if str_flat.is_empty() {
        0.0
    } else {
        max_abs(&str_flat)
    };
    let (audio48, peak) = resample_normalize(&pressure, fs);
    let time: Vec<f64> = (0..total.len()).map(|i| i as f64 / fs).collect();
    let times: Vec<f64> = frame_steps.iter().map(|&i| i as f64 / fs).collect();
    Ok(json!({
        "model": "platebody",
        "boundary": boundary,
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(lam, 6)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(PLATEBODY_SIDE, 6)),
            "extent_y": num(round_nd(PLATEBODY_SIDE, 6)),
            "domain": "rectangle",
        },
        "frames": {
            "b64": b64f32(&plate_dec),
            "n_frames": int(nf as i64), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "width": int(nx_dec as i64), "dims": int(2),
        },
        "mask": {"b64": b64u8(&mask_dec), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64)},
        "string": {
            "b64": b64f32(&str_flat),
            "n_frames": int(frames_string.len() as i64),
            "width": int(frames_string.first().map_or(0, |f| f.len() as i64)),
            "amp": num(str_amp),
            "x": finite_list(&x, Some(6)),
        },
        "frame_times": finite_list(&times, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&time, &total, info.sigma_plate == 0.0, 0.0, EnergyOpts {
            no_decay_oracle: true,
            ..EnergyOpts::default()
        }),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1_base, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(nf as i64),
            "boundary": boundary,
            "n_plate": int(info.n_plate),
            "n_live": int(info.n_live as i64),
            "probe_x": num(round_nd(PLATEBODY_PICKUP_FRAC * l, 4)),
            "exchange": exchange,
            "spectrum": spectrum,
        },
    }))
}
