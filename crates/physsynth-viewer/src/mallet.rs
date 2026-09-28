//! A mallet striking the drumhead — `serialize.py`'s mallet builder (model #7), the viewer's first
//! contact model.
//!
//! It reuses the membrane's 2-D machinery whole, with three differences: the head starts at REST
//! (the mallet strikes it, so there is no raised-cosine start); the verdict is conservation with
//! the `2 sigma` decay line DROPPED (after separation the mallet flies off at nearly constant speed,
//! so the total energy floors at its kinetic energy and a fitted rate would be a lying zero); and
//! the headline is the CONTACT episode rather than the tone — a point mass is an inefficient
//! exciter, bouncing off with restitution near 1 while the head keeps about 0.01 %.

use physsynth_core::engine::simulate;
use physsynth_core::fmt::py_float;
use physsynth_core::mallet::{self as mal, MalletMembrane};
use physsynth_core::pyfloat::scalar_pow;
use serde_json::{json, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::membrane::{
    build_membrane, decimate_field_mask, discrete_eigenfreqs, horizon_block, xy_to_frac, Geom,
    N_MEMBRANE_MODES,
};
use crate::py::{
    b64f32, b64u8, finite_list, int, linspace_idx, max_abs, np_mean, num, py_int, py_repr,
    round_int, round_nd,
};
use crate::string::{
    construction, int_at_least_one, ANIM_WIN_MAX, FRAMES_PER_PERIOD, MAX_FRAMES, SPEED_MAX,
};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling (each step also runs a contact root-find).
pub const MALLET_N_MAX: i64 = 80;
/// Node-steps across the audio and animation runs.
pub const MALLET_WORK_MAX: f64 = 3.5e8;
/// Longest audio, seconds.
pub const MALLET_AUDIO_MAX: f64 = 2.0;
/// Length of the decimated contact-episode traces.
pub const MALLET_DIAG_POINTS: usize = 240;
/// A contact start or end must hold this many steps (the flag flickers at grazing).
const GRAZE: usize = 10;

fn pf(x: f64) -> String {
    py_float(x)
}

/// What `_build_mallet` reports beside the struck membrane.
pub struct Info {
    /// Mallet mass (kg).
    pub mass: f64,
    /// Felt hysteresis.
    pub hysteresis: f64,
    /// Strike velocity (m/s).
    pub strike_velocity: f64,
    /// The snapped strike position, as fractions (rounded to 4 places).
    pub strike_fx: f64,
    /// See `strike_fx`.
    pub strike_fy: f64,
    /// Rigid-wall steps per contact half-period, rounded to 0.1.
    pub steps_per_contact: f64,
    /// Whether that is at least 8.
    pub resolved: bool,
}

/// `_build_mallet`: the struck membrane, at rest, plus the contact quantities.
///
/// The `N` pre-check reads a default of 60 while the membrane underneath reads 80, as the
/// reference does: with no `N` the pre-check passes 60 and the drum is built at 80.
pub fn build_mallet(p: &Value) -> Result<(MalletMembrane, f64, f64, f64, Geom, Info), Refusal> {
    let nv = p.get("N").cloned().unwrap_or(json!(60));
    let n_req = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    if n_req > MALLET_N_MAX {
        return Err(Refusal::Param(format!(
            "N must be <= {MALLET_N_MAX} for the mallet (each step also runs a contact \
             root-find), got {n_req}."
        )));
    }
    let b = build_membrane(p)?;
    let (c, fs, sigma, geom) = (b.c, b.fs, b.sigma, b.geom);

    let mass = fnum(p, "mass", 0.02)?;
    let stiffness = fnum(p, "stiffness", 5.0e4)?;
    let alpha = fnum(p, "alpha", 2.3)?;
    let hysteresis = fnum(p, "hysteresis", 0.0)?;
    let v0 = fnum(p, "strike_velocity", 3.0)?;
    // the strike (x, y) reuses the shared 2-D strike sliders
    let strike_fx = fnum(p, "pluck_x", 0.5)?;
    let strike_fy = fnum(p, "pluck_y", 0.5)?;

    if mass <= 0.0 {
        return Err(Refusal::Param(format!(
            "mallet mass must be > 0, got {}.",
            pf(mass)
        )));
    }
    if stiffness <= 0.0 {
        return Err(Refusal::Param(format!(
            "felt stiffness must be > 0, got {}.",
            pf(stiffness)
        )));
    }
    if alpha < 1.0 {
        return Err(Refusal::Param(format!(
            "felt exponent alpha must be >= 1, got {}.",
            pf(alpha)
        )));
    }
    if hysteresis < 0.0 {
        return Err(Refusal::Param(format!(
            "hysteresis must be >= 0, got {}.",
            pf(hysteresis)
        )));
    }
    if v0 <= 0.0 {
        return Err(Refusal::Param(format!(
            "strike velocity must be > 0, got {}.",
            pf(v0)
        )));
    }
    for (name, v) in [("pluck_x", strike_fx), ("pluck_y", strike_fy)] {
        if !(0.0 < v && v < 1.0) {
            return Err(Refusal::Param(format!(
                "{name} (strike position) must be in (0, 1), got {}.",
                pf(v)
            )));
        }
    }

    let (sx, sy) = geom.frac_to_xy(strike_fx, strike_fy);
    // The binding's defaults: gap 0, eta_tol 1e-12, newton_tol 1e-14, 60 iterations.
    let params = mal::Params::new(
        b.res.params(),
        mass,
        stiffness,
        alpha,
        hysteresis,
        sx,
        sy,
        0.0,
        1e-12,
        1e-14,
        60,
    )
    .map_err(construction)?;
    let (fx, fy) = xy_to_frac(&geom, params.x_strike, params.y_strike);
    let mallet = MalletMembrane::new(params, b.res, 0.0, v0);

    // Rigid-wall contact resolution: the felt half-period must span several steps or the strike
    // aliases. It understates the coupled contact on a yielding head, and it is a NOTE, not an
    // error — the energy conserves either way.
    let steps_per_contact = std::f64::consts::PI * (mass / stiffness).sqrt() * fs;
    let info = Info {
        mass,
        hysteresis,
        strike_velocity: v0,
        strike_fx: round_nd(fx, 4),
        strike_fy: round_nd(fy, 4),
        steps_per_contact: round_nd(steps_per_contact, 1),
        resolved: steps_per_contact >= 8.0,
    };
    Ok((mallet, c, fs, sigma, geom, info))
}

/// `_mallet_contact_block`: the contact-episode verdict, the mallet's headline.
#[allow(clippy::too_many_arguments)]
fn contact_block(
    time: &[f64],
    mvel: &[f64],
    head_e: &[f64],
    force: &[f64],
    in_contact: &[bool],
    ke0: f64,
    info: &Info,
) -> Value {
    let n = mvel.len();
    let v0 = info.strike_velocity;
    let sustained = |i: usize, want: bool| in_contact[i..i + GRAZE].iter().all(|&c| c == want);
    let start = (0..n.saturating_sub(GRAZE)).find(|&i| in_contact[i] && sustained(i, true));
    let sep = start.and_then(|s| {
        (s + 1..n.saturating_sub(GRAZE)).find(|&i| !in_contact[i] && sustained(i, false))
    });

    let (restitution, contact_ms, separated, zoom_end) = match (start, sep) {
        // never made real contact: an honest null, not a bug
        (None, _) => (1.0, None, false, n - 1),
        // still in contact at the end of the window
        (Some(_), None) => (np_mean(&mvel[n - GRAZE..]).abs() / v0, None, false, n - 1),
        (Some(_), Some(sep)) => (
            np_mean(&mvel[sep..(sep + GRAZE).min(n)]).abs() / v0,
            Some(time[sep] * 1e3),
            true,
            (n - 1).min((sep as f64 * 2.0) as usize),
        ),
    };

    let head_max = head_e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let peak_head_pct = if ke0 > 0.0 {
        head_max / ke0 * 100.0
    } else {
        0.0
    };
    let tail = &head_e[(n as f64 * 0.9) as usize..];
    let final_head_pct = if ke0 > 0.0 && !tail.is_empty() {
        np_mean(tail) / ke0 * 100.0
    } else {
        0.0
    };
    let peak_force = force.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // `np.unique(np.linspace(0, hi - 1, min(hi, POINTS)).astype(int))`
    let hi = 2.max(zoom_end + 1);
    let mut idx = linspace_idx(hi - 1, hi.min(MALLET_DIAG_POINTS));
    idx.sort_unstable();
    idx.dedup();
    let pick = |a: &[f64]| idx.iter().map(|&i| a[i]).collect::<Vec<f64>>();
    json!({
        "kind": "mallet",
        "t": finite_list(&pick(time), Some(6)),
        "vel": finite_list(&pick(mvel), Some(5)),
        "force": finite_list(&pick(force), Some(5)),
        "v0": num(round_nd(v0, 4)),
        "restitution": num(round_nd(restitution, 4)),
        "separated": separated,
        "contact_ms": contact_ms.map_or(Value::Null, |v| num(round_nd(v, 2))),
        "peak_head_pct": num(round_nd(peak_head_pct, 3)),
        "final_head_pct": num(round_nd(final_head_pct, 4)),
        "peak_force": num(round_nd(peak_force, 3)),
        "steps_per_contact": num(info.steps_per_contact),
        "resolved": info.resolved,
        "strike_fx": num(info.strike_fx),
        "strike_fy": num(info.strike_fy),
    })
}

/// `_build_payload_mallet`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 1.0)?;
    let anim_win = fnum(p, "animation_window", 0.06)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pickup_fx = fnum(p, "pickup_x", 0.65)?;
    let pickup_fy = fnum(p, "pickup_y", 0.6)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= MALLET_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(MALLET_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.0 < anim_win && anim_win <= ANIM_WIN_MAX) {
        return Err(Refusal::Param(format!(
            "animation_window must be in (0, {}] s, got {}.",
            pf(ANIM_WIN_MAX),
            pf(anim_win)
        )));
    }
    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    for (name, v) in [("pickup_x", pickup_fx), ("pickup_y", pickup_fy)] {
        if !(0.0 < v && v < 1.0) {
            return Err(Refusal::Param(format!(
                "{name} must be in (0, 1), got {}.",
                pf(v)
            )));
        }
    }

    let (mut mallet, c, fs, sigma, geom, info) = build_mallet(p)?;
    let lam_h = info.hysteresis;
    let n_live = mallet.membrane.params().n_live();

    let n_audio = round_int(audio_dur * fs).max(1);
    let n_anim_est = round_int(anim_win * fs).max(1);
    let work = n_live as i64 * (n_audio + n_anim_est);
    if work as f64 > MALLET_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs ~{:.0}M node-steps (over the ~{:.0}M mallet budget); \
             reduce N, raise lambda, shorten the audio/animation, or enlarge the drum.",
            work as f64 / 1e6,
            MALLET_WORK_MAX / 1e6
        )));
    }

    let f_disc = discrete_eigenfreqs(&mallet.membrane, c, N_MEMBRANE_MODES)?;
    let f1 = if f_disc.is_empty() {
        c / (2.0 * geom.length_scale())
    } else {
        f_disc[0]
    };

    // -- the audio run: ONE instrumented loop, because the contact story is the headline and
    // `simulate` exposes none of the mallet's own state. The head starts at rest.
    let (qx, qy) = geom.frac_to_xy(pickup_fx, pickup_fy);
    let pickup_idx = mallet.membrane.params().pickup_index_at(qx, qy);
    let n = n_audio as usize;
    let mut energy = Vec::with_capacity(n + 1);
    let mut pickup = Vec::with_capacity(n + 1);
    let mut mvel = Vec::with_capacity(n + 1);
    let mut head_e = Vec::with_capacity(n + 1);
    let mut force = Vec::with_capacity(n + 1);
    let mut in_contact = Vec::with_capacity(n + 1);
    let sample = |m: &MalletMembrane,
                  e: &mut Vec<f64>,
                  pu: &mut Vec<f64>,
                  mv: &mut Vec<f64>,
                  he: &mut Vec<f64>,
                  fo: &mut Vec<f64>,
                  ic: &mut Vec<bool>| {
        e.push(m.energy());
        pu.push(m.membrane.displacement_at(pickup_idx));
        mv.push(m.mallet_velocity());
        he.push(m.membrane.energy());
        fo.push(m.state().contact_force);
        ic.push(m.state().in_contact);
    };
    sample(
        &mallet,
        &mut energy,
        &mut pickup,
        &mut mvel,
        &mut head_e,
        &mut force,
        &mut in_contact,
    );
    for _ in 0..n {
        // A failed contact solve raised RuntimeError in the reference, which nothing caught.
        mallet
            .step()
            .map_err(|e| Refusal::Internal(e.to_string()))?;
        sample(
            &mallet,
            &mut energy,
            &mut pickup,
            &mut mvel,
            &mut head_e,
            &mut force,
            &mut in_contact,
        );
    }
    if !pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }
    let time: Vec<f64> = (0..=n).map(|i| i as f64 / fs).collect();
    let lossless = sigma == 0.0 && lam_h == 0.0;
    let ke0 = 0.5 * info.mass * scalar_pow(info.strike_velocity, 2.0);
    let contact = contact_block(&time, &mvel, &head_e, &force, &in_contact, ke0, &info);

    // -- the animation run: a fresh mallet from rest, a fundamental-resolving stride
    let mut anim = build_mallet(p)?.0;
    let mut anim_stride = round_int((fs / f1) / fpp as f64).max(1);
    let n_anim = anim_stride.max(round_int(anim_win * fs));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let anim_run = simulate(&mut anim, n_anim as usize, None, anim_stride as usize)
        .map_err(Refusal::Internal)?;
    let frames_full: Vec<Vec<f64>> = anim_run.snapshots.iter().map(|(_, s)| s.clone()).collect();
    let frame_times: Vec<f64> = anim_run
        .snapshots
        .iter()
        .map(|(i, _)| *i as f64 / fs)
        .collect();

    let mp = mallet.membrane.params();
    let (ny, nx) = mp.shape();
    let (frames_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&frames_full, mp.mask.flags(), ny, nx);
    let nf = frames_full.len();
    let field_amp = if frames_dec.is_empty() {
        0.0
    } else {
        max_abs(&frames_dec)
    };
    let (audio48, peak) = resample_normalize(&pickup, fs);
    let (ext_x, ext_y) = match geom {
        Geom::Circle { radius } => (2.0 * radius, 2.0 * radius),
        Geom::Rectangle { lx, ly } => (lx, ly),
    };
    Ok(json!({
        "model": "mallet",
        "horizon": horizon_block(&geom, mp, "the struck membrane"),
        "domain": geom.domain(),
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(mp.lam, 6)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(ext_x, 6)), "extent_y": num(round_nd(ext_y, 6)),
            "domain": geom.domain(),
        },
        "frames": {
            "b64": b64f32(&frames_dec),
            "n_frames": int(nf as i64), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "width": int(nx_dec as i64), "dims": int(2),
        },
        "mask": {"b64": b64u8(&mask_dec), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64)},
        "frame_times": finite_list(&frame_times, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&time, &energy, lossless, 2.0 * sigma, EnergyOpts {
            no_decay_oracle: true,
            ..EnergyOpts::default()
        }),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(nf as i64),
            "spectrum": contact,
        },
    }))
}
