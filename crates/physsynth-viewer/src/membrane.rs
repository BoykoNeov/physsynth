//! The 2-D drumhead — `serialize.py`'s membrane builder, the first heatmap scene.
//!
//! What differs from the strings: frames are 2-D fields decimated to a display grid of at most
//! [`DISPLAY_MAX`] a side (with the mask decimated by the SAME stride, so the two stay aligned),
//! the excitation and pickup are `(x, y)` fractions, and the spectrum panel marks the operator's
//! own **discrete** eigenfrequencies rather than scoring partials. Those markers come from
//! `physsynth_core::eigs` — the one number in this payload that goes through an iteration, so
//! the one place it agrees with the reference to a tolerance rather than the bit (retirement plan
//! §23.11).

use physsynth_analysis::{modal, spectrum};
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::engine::simulate;
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::fmt::py_float;
use physsynth_core::membrane::{self as mem, Domain, Membrane};
use serde_json::{json, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::horizon::{grid2d_block, horizon_none, Grid2dScheme, HORIZON_STAIRCASE};
use crate::py::{
    b64f32, b64u8, finite_list, int, max_abs, num, py_int, py_repr, py_str, round_int, round_nd,
};
use crate::string::{
    construction, int_at_least_one, ANIM_WIN_MAX, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN, SPEED_MAX,
};
use crate::tension::pooled_spectrum;
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling.
pub const MEMBRANE_N_MAX: i64 = 100;
/// Live-node ceiling: below the `n_live ~ 10,000` cache cliff; admits a square or disk at N = 100.
pub const MEMBRANE_NLIVE_MAX: usize = 9_900;
/// Node-steps across the audio and animation runs.
pub const MEMBRANE_WORK_MAX: f64 = 7.0e8;
/// Longest audio, seconds.
pub const MEMBRANE_AUDIO_MAX: f64 = 2.0;
/// Discrete eigenmodes marked on the spectrum panel.
pub const N_MEMBRANE_MODES: usize = 12;
/// Display grid ceiling, per side, for every heatmap scene.
pub const DISPLAY_MAX: usize = 64;

/// The 2-D CFL ceiling, `1 / sqrt(2)`, spelled as the reference's division.
pub fn membrane_lambda_max() -> f64 {
    1.0 / 2.0f64.sqrt()
}

fn pf(x: f64) -> String {
    py_float(x)
}

/// Python's two-argument `min`: the first argument unless the second is strictly smaller, so a
/// NaN in either place is kept or dropped exactly as the reference's guard would.
fn py_min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// The snapped geometry read back off the resonator.
#[derive(Debug, Clone, Copy)]
pub enum Geom {
    /// A disk of this radius.
    Circle {
        /// Radius (m).
        radius: f64,
    },
    /// A rectangle, `ly` snapped to whole cells.
    Rectangle {
        /// Width (m).
        lx: f64,
        /// Height (m), snapped.
        ly: f64,
    },
}

impl Geom {
    fn domain(&self) -> &'static str {
        match self {
            Geom::Circle { .. } => "circle",
            Geom::Rectangle { .. } => "rectangle",
        }
    }

    /// `_length_scale`: the disk's radius, or the rectangle's shorter side.
    fn length_scale(&self) -> f64 {
        match *self {
            Geom::Circle { radius } => radius,
            Geom::Rectangle { lx, ly } => py_min2(lx, ly),
        }
    }

    /// `_frac_to_xy`: `(fx, fy)` in `(0, 1)²` to a physical point.
    fn frac_to_xy(&self, fx: f64, fy: f64) -> (f64, f64) {
        match *self {
            Geom::Circle { radius } => ((2.0 * fx - 1.0) * radius, (2.0 * fy - 1.0) * radius),
            Geom::Rectangle { lx, ly } => (fx * lx, fy * ly),
        }
    }
}

/// What `_build_membrane` returns beside the resonator.
pub struct Built {
    /// The resonator, at rest.
    pub res: Membrane,
    /// Wave speed (m/s).
    pub c: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Loss.
    pub sigma: f64,
    /// Snapped geometry.
    pub geom: Geom,
}

/// `_build_membrane`.
pub fn build_membrane(p: &Value) -> Result<Built, Refusal> {
    let dv = p.get("domain").cloned().unwrap_or(json!("circle"));
    let domain = py_str(&dv);
    if domain != "circle" && domain != "rectangle" {
        return Err(Refusal::Param(format!(
            "domain must be 'circle' or 'rectangle', got {}.",
            crate::py::repr_str(&domain)
        )));
    }
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lam = fnum(p, "lambda", 0.6)?;
    let sigma = fnum(p, "sigma", 0.0)?;
    let nv = p.get("N").cloned().unwrap_or(json!(80));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;

    if !(N_MIN..=MEMBRANE_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {MEMBRANE_N_MAX}] for the membrane, got {n}."
        )));
    }
    if py_min2(t, rho) <= 0.0 {
        return Err(Refusal::Param("T, rho must both be positive.".into()));
    }
    if sigma < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma (loss) must be >= 0, got {}.",
            pf(sigma)
        )));
    }
    let lmax = membrane_lambda_max();
    if !(0.0 < lam && lam <= lmax + 1e-9) {
        return Err(Refusal::Param(format!(
            "lambda must be in (0, 1/sqrt(2) = {lmax:.4}] (2D CFL), got {}.",
            pf(lam)
        )));
    }

    let c = (t / rho).sqrt();
    let (params, geom, fs) = if domain == "circle" {
        let radius = fnum(p, "radius", 0.5)?;
        if radius <= 0.0 {
            return Err(Refusal::Param(format!(
                "radius must be positive, got {}.",
                pf(radius)
            )));
        }
        let h = 2.0 * radius / n as f64;
        let fs = c / (lam * h);
        let params = mem::Params::new(
            Some(Domain::Circle),
            t,
            rho,
            fs,
            n,
            None,
            None,
            Some(radius),
            sigma,
        )
        .map_err(construction)?;
        (params, Geom::Circle { radius }, fs)
    } else {
        let lx = fnum(p, "Lx", 1.0)?;
        let ly = fnum(p, "Ly", 1.0)?;
        if py_min2(lx, ly) <= 0.0 {
            return Err(Refusal::Param("Lx, Ly must both be positive.".into()));
        }
        let h = lx / n as f64;
        let fs = c / (lam * h);
        let params = mem::Params::new(
            Some(Domain::Rectangle),
            t,
            rho,
            fs,
            n,
            Some(lx),
            Some(ly),
            None,
            sigma,
        )
        .map_err(construction)?;
        let geom = Geom::Rectangle {
            lx: params.lx.expect("a rectangle has Lx"),
            ly: params.ly.expect("a rectangle has Ly"),
        };
        (params, geom, fs)
    };
    let n_live = params.n_live();
    if n_live > MEMBRANE_NLIVE_MAX {
        return Err(Refusal::Param(format!(
            "this geometry has {n_live} interior nodes (> {MEMBRANE_NLIVE_MAX}); reduce N or use \
             a less extreme aspect ratio."
        )));
    }
    Ok(Built {
        res: Membrane::new(params),
        c,
        fs,
        sigma,
        geom,
    })
}

/// `_discrete_eigenfreqs`: the lowest discrete eigenfrequencies (Hz), from `eigsh(-L)` mapped
/// through the leapfrog dispersion. These are where the stepper actually rings.
pub fn discrete_eigenfreqs(res: &Membrane, c: f64, k_request: usize) -> Result<Vec<f64>, Refusal> {
    let p = res.params();
    let k = k_request.min(p.n_live().saturating_sub(1));
    if k < 1 {
        return Ok(Vec::new());
    }
    let neg_l = p.l.scaled(-1.0);
    let pairs = eigsh_shift_invert(&neg_l, None, 0.0, k)
        .map_err(|e| Refusal::Construction(e.to_string()))?;
    let mut f: Vec<f64> = pairs
        .values
        .iter()
        .map(|&lam| modal::discrete_membrane_eigenfrequency(lam, c, p.k))
        .collect();
    f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(f)
}

/// `_membrane_continuum_oracle`: the analytic frequencies, the geometry-tier reference.
pub fn continuum_oracle(geom: &Geom, c: f64, n_modes: usize) -> Vec<f64> {
    if n_modes < 1 {
        return Vec::new();
    }
    let mut f: Vec<f64> = match *geom {
        Geom::Circle { radius } => modal::circular_membrane_freqs(c, radius, n_modes, 12, 12)
            .iter()
            .flat_map(|m| std::iter::repeat_n(m.freq, m.degeneracy as usize))
            .collect(),
        Geom::Rectangle { lx, ly } => {
            let nm = n_modes as i64;
            let modes: Vec<(i64, i64)> = (1..=nm)
                .flat_map(|m| (1..=nm).map(move |n| (m, n)))
                .collect();
            modal::rectangular_membrane_freqs(c, lx, ly, &modes)
        }
    };
    f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    f.truncate(n_modes);
    f
}

/// `_modal_spectrum_block`: the pooled spectrum, the discrete and continuum markers, and the two
/// headline cents numbers. Shared with the linear plate (`kind` picks the wording).
pub fn modal_spectrum_block(
    pickup: &[f64],
    fs: f64,
    f_disc: &[f64],
    f_cont: &[f64],
    kind: &str,
) -> Value {
    if f_disc.is_empty() {
        return Value::Null;
    }
    let fmax = f_disc.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.25;
    let Some((f_ds, m_ds)) = pooled_spectrum(pickup, fs, fmax) else {
        return Value::Null;
    };
    let f1 = f_disc[0];
    let detected = spectrum::measure_partials_near(pickup, fs, &[f1], Some(0.3 * f1))[0];
    let cents_fund = (detected > 0.0).then(|| modal::cents(detected, f1));
    let cents_geom = (!f_cont.is_empty() && f_cont[0] > 0.0).then(|| modal::cents(f1, f_cont[0]));
    let opt = |v: Option<f64>| v.map_or(Value::Null, |x| num(round_nd(x, 4)));
    json!({
        "kind": kind,
        "freq": finite_list(&f_ds, Some(3)),
        "mag": finite_list(&m_ds, Some(5)),
        "fmax": num(round_nd(fmax, 3)),
        "modes_discrete": finite_list(f_disc, Some(4)),
        "modes_continuum": finite_list(f_cont, Some(4)),
        "f1_discrete": num(round_nd(f1, 4)),
        "f1_detected": opt((detected > 0.0).then_some(detected)),
        "cents_fundamental": opt(cents_fund),
        "cents_geometry": opt(cents_geom),
    })
}

/// `_horizon_membrane_block`: a rectangle is measured, a staircased disk is refused.
pub fn horizon_block(b: &Built, of: &str) -> Value {
    match b.geom {
        Geom::Circle { .. } => horizon_none(HORIZON_STAIRCASE, Some(of)),
        Geom::Rectangle { lx, ly } => {
            let p = b.res.params();
            grid2d_block(
                p.h,
                p.k,
                lx,
                ly,
                p.fs,
                Grid2dScheme::Membrane { c: p.c },
                of,
            )
        }
    }
}

/// Decimate a `(nf, ny, nx)` field stack and its mask by the same stride (`_decimate_field_mask`,
/// point-sampling). Returns `(frames, mask, ny_dec, nx_dec)`.
pub fn decimate_field_mask(
    frames: &[Vec<f64>],
    mask: &[bool],
    ny: usize,
    nx: usize,
) -> (Vec<f64>, Vec<u8>, usize, usize) {
    let s = ny.max(nx).div_ceil(DISPLAY_MAX).max(1);
    let rows: Vec<usize> = (0..ny).step_by(s).collect();
    let cols: Vec<usize> = (0..nx).step_by(s).collect();
    let mut out = Vec::with_capacity(frames.len() * rows.len() * cols.len());
    for fr in frames {
        for &i in &rows {
            for &j in &cols {
                out.push(fr[i * nx + j]);
            }
        }
    }
    let mut m = Vec::with_capacity(rows.len() * cols.len());
    for &i in &rows {
        for &j in &cols {
            m.push(u8::from(mask[i * nx + j]));
        }
    }
    (out, m, rows.len(), cols.len())
}

/// `_build_payload_membrane`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 2.0)?;
    let anim_win = fnum(p, "animation_window", 0.05)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let amplitude = fnum(p, "amplitude", 1e-3)?;
    let pluck_fx = fnum(p, "pluck_x", 0.4)?;
    let pluck_fy = fnum(p, "pluck_y", 0.55)?;
    let pluck_wfrac = fnum(p, "pluck_width", 0.45)?;
    let pickup_fx = fnum(p, "pickup_x", 0.65)?;
    let pickup_fy = fnum(p, "pickup_y", 0.6)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= MEMBRANE_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(MEMBRANE_AUDIO_MAX),
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
    for (name, v) in [
        ("pluck_x", pluck_fx),
        ("pluck_y", pluck_fy),
        ("pickup_x", pickup_fx),
        ("pickup_y", pickup_fy),
    ] {
        if !(0.0 < v && v < 1.0) {
            return Err(Refusal::Param(format!(
                "{name} must be in (0, 1), got {}.",
                pf(v)
            )));
        }
    }
    if !(0.0 < pluck_wfrac && pluck_wfrac <= 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_width must be in (0, 1], got {}.",
            pf(pluck_wfrac)
        )));
    }

    let mut b = build_membrane(p)?;
    let (c, fs, sigma, geom) = (b.c, b.fs, b.sigma, b.geom);
    let n_live = b.res.params().n_live();

    let n_audio = round_int(audio_dur * fs).max(1);
    let n_anim_est = round_int(anim_win * fs).max(1);
    let work = n_live as i64 * (n_audio + n_anim_est);
    if work as f64 > MEMBRANE_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs ~{:.0}M node-steps (over the ~{:.0}M budget); reduce N, \
             raise lambda, shorten the audio/animation, or enlarge the geometry.",
            work as f64 / 1e6,
            MEMBRANE_WORK_MAX / 1e6
        )));
    }

    let f_disc = discrete_eigenfreqs(&b.res, c, N_MEMBRANE_MODES)?;
    let f_cont = continuum_oracle(&geom, c, f_disc.len());
    let f1 = if f_disc.is_empty() {
        c / (2.0 * geom.length_scale())
    } else {
        f_disc[0]
    };

    // -- the audio run
    let wc = pluck_wfrac * geom.length_scale();
    let (pcx, pcy) = geom.frac_to_xy(pluck_fx, pluck_fy);
    let excite = |res: &mut Membrane| -> Result<(), Refusal> {
        let pp = res.params();
        let field =
            raised_cosine_2d(&pp.x, &pp.y, (pcx, pcy), wc, amplitude).map_err(construction)?;
        let live = pp.to_live(&field);
        res.set_displacement(&live);
        Ok(())
    };
    excite(&mut b.res)?;
    let (qx, qy) = geom.frac_to_xy(pickup_fx, pickup_fy);
    let pickup_idx = b.res.params().pickup_index_at(qx, qy);
    let audio_run =
        simulate(&mut b.res, n_audio as usize, Some(pickup_idx), 0).map_err(Refusal::Internal)?;
    let pickup = audio_run.output.as_deref().expect("a pickup was requested");
    if !pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }

    // -- the animation run: a fresh membrane, a short window, a fundamental-resolving stride
    let mut anim = build_membrane(p)?.res;
    excite(&mut anim)?;
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

    // -- decimation to the display grid: field AND mask, same stride
    let (ny, nx) = b.res.params().shape();
    let (frames_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&frames_full, b.res.params().mask.flags(), ny, nx);
    let nf = frames_full.len();
    let field_amp = if frames_dec.is_empty() {
        0.0
    } else {
        max_abs(&frames_dec)
    };

    let (audio48, peak) = resample_normalize(pickup, fs);
    let (ext_x, ext_y) = match geom {
        Geom::Circle { radius } => (2.0 * radius, 2.0 * radius),
        Geom::Rectangle { lx, ly } => (lx, ly),
    };
    let lam = b.res.params().lam;
    Ok(json!({
        "model": "membrane",
        "horizon": horizon_block(&b, "the membrane"),
        "domain": geom.domain(),
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(lam, 6)),
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
        "energy": energy_block(&audio_run.time, &audio_run.energy, sigma == 0.0, 2.0 * sigma,
                               EnergyOpts::default()),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(nf as i64),
            "spectrum": modal_spectrum_block(pickup, fs, &f_disc, &f_cont, "membrane"),
        },
    }))
}
