//! The gong in the room — `serialize.py`'s vkroom builder: the von Kármán plate as a radiating
//! SURFACE inside the 3-D room, baffled in a wall or suspended mid-room.
//!
//! THE CLAIM: a loud plate's radiation is time-varying at fixed geometry and a quiet one's is not.
//! The von Kármán coupling is quadratic, so the SHAPE of the motion evolves during one strike, and a
//! surface radiates by its shape. The observable is the shape-radiation efficiency per window
//! (a fixed quadratic form, the room's own resistive operator on the plate's coupled velocity),
//! against a control twin that is the same scene with the coupling off.
//!
//! The per-mode shares behind `modal_drift` depend on the basis inside each exactly repeated pair
//! of the square free plate's modes — in SciPy's output and in this one's. Measured on the
//! reference (retirement plan §23.17): rotating every repeated pair leaves every shipped digit
//! unchanged, so no solver's choice of basis reaches the payload. The modes must be MASS-
//! orthonormal, which `eig::generalized_eigen_diag` guarantees.

use physsynth_core::airbox::{self as ab, AirBox, Wall};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{RoomGrid, VkSeam};
use physsynth_core::eig::generalized_eigen_diag;
use physsynth_core::fmt::py_float;
use physsynth_core::plate::{Boundary, VkParams, VkPlate, VkSpec};
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::reduce::sum;
use serde_json::{json, Value};

use crate::airbox::{index, np_percentile, slice_planes, snap, take_slices, Plane};
use crate::energy::{energy_block, EnergyOpts};
use crate::membrane::decimate_field_mask;
use crate::py::{
    as_bool, b64f32, b64u8, commas, dot, finite_list, int, np_mean, num, py_int, py_repr, py_str,
    repr_str, round_int, round_nd, sci,
};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, SPEED_MAX};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// The two room tiers the `domain` select carries.
pub const VKROOM_TIERS: [&str; 2] = ["baffled", "suspended"];
/// Fixed: the honest cymbal (the supported gong resolves too few modes here to carry the claim).
pub const VKROOM_BOUNDARY: &str = "free";
/// Square plate side (m).
pub const VKROOM_PLATE_L: f64 = 0.10;
/// Plate grid default.
pub const VKROOM_PLATE_N_DEFAULT: i64 = 16;
/// Plate grid range.
pub const VKROOM_PLATE_N_MIN: i64 = 8;
/// See [`VKROOM_PLATE_N_MIN`].
pub const VKROOM_PLATE_N_MAX: i64 = 16;
/// Strike width, as a fraction of Lx — fixed: 0.19 runs at 109 sweeps and 0.18 is dead.
pub const VKROOM_STRIKE_WIDTH: f64 = 0.20;
/// Strike amplitude default (thicknesses).
pub const VKROOM_WOVERE_DEFAULT: f64 = 3.0;
/// Strike amplitude range.
pub const VKROOM_WOVERE_MIN: f64 = 0.01;
/// See [`VKROOM_WOVERE_MIN`].
pub const VKROOM_WOVERE_MAX: f64 = 3.0;
/// Measured dead; 3.2 runs at 109 of 120 sweeps.
pub const VKROOM_WOVERE_CLIFF: f64 = 3.4;
/// The room's Courant fraction — a constant, not a slider.
pub const VKROOM_CFL: f64 = 0.90;
/// Air grid default (m): the coarsest cell whose plate still converges.
pub const VKROOM_H_DEFAULT: f64 = 0.0114;
/// Air grid range (m) — a stability limit.
pub const VKROOM_H_MIN: f64 = 0.0090;
/// See [`VKROOM_H_MIN`].
pub const VKROOM_H_MAX: f64 = 0.0125;
/// Room side default (m, cube).
pub const VKROOM_SIZE_DEFAULT: f64 = 0.35;
/// Room side range (m).
pub const VKROOM_SIZE_MIN: f64 = 0.25;
/// See [`VKROOM_SIZE_MIN`].
pub const VKROOM_SIZE_MAX: f64 = 0.60;
/// Normalized wall impedance default.
pub const VKROOM_ZETA_DEFAULT: f64 = 4.0;
/// Normalized wall impedance ceiling.
pub const VKROOM_ZETA_MAX: f64 = 50.0;
/// Audio default (s).
pub const VKROOM_AUDIO_DEFAULT: f64 = 0.12;
/// Longest audio (s).
pub const VKROOM_AUDIO_MAX: f64 = 0.30;
/// Picard sweep cap (the core default 50 caps out on the loud arm).
pub const VKROOM_SWEEP_CAP: i64 = 120;
/// Observation windows — itself a claim.
pub const VKROOM_WINDOWS: usize = 4;
/// Air cells per structural wavelength below which a mode is not resolved.
pub const VKROOM_CELLS_PER_WAVE: f64 = 5.0;
/// A free plate's nullspace is exactly `{1, x, y}`.
pub const VKROOM_NULLSPACE_DIM: usize = 3;
/// Room node ceiling.
pub const VKROOM_NODE_MAX: usize = 200_000;
/// `nodes x steps`, counting the control twin's room.
pub const VKROOM_ROOM_WORK_MAX: f64 = 8.0e8;
/// `n_live x steps x sweep cap`.
pub const VKROOM_PLATE_WORK_MAX: f64 = 5.0e8;
/// Modal projections per window.
pub const VKROOM_PROJ_SAMPLES: f64 = 400.0;
/// Energy-channel samples.
pub const VKROOM_TRACE_POINTS: f64 = 400.0;
/// Seconds of plate animation.
pub const VKROOM_ANIM_WIN: f64 = 0.004;
/// Seconds of room slices.
pub const VKROOM_ROOM_ANIM_WIN: f64 = 0.0015;
/// One slice frame per cell of acoustic travel.
pub const VKROOM_CELLS_PER_FRAME: f64 = 1.0;
/// The far end of the mic's travel, off-axis.
pub const VKROOM_MIC_FAR: [f64; 3] = [0.90, 0.86, 0.93];
/// Mic default.
pub const VKROOM_MIC_DEFAULT: f64 = 0.75;
/// Percentile of the live slice field that sets the asinh reference.
pub const VKROOM_SCALE_PCTL: f64 = 55.0;

fn pf(x: f64) -> String {
    py_float(x)
}

type Grid = RoomGrid<VkSeam>;

/// The mode set: vectors (as columns), frequencies (Hz), and which the air grid resolves.
type Modes = (Vec<Vec<f64>>, Vec<f64>, Vec<bool>);

/// The scene: the wrapped plate, its room, and the scalars.
struct Scene {
    grid: Grid,
    room: AirBox,
    suspended: bool,
    fs: f64,
    h: f64,
    lam_air: f64,
    nodes: usize,
    zeta: f64,
    size: f64,
    n_plate: i64,
}

/// `_build_vkroom_scene`.
fn build(p: &Value, nonlinear: bool) -> Result<Scene, Refusal> {
    let tv = p.get("domain").cloned().unwrap_or(json!("suspended"));
    let tier = py_str(&tv);
    if !VKROOM_TIERS.contains(&tier.as_str()) {
        return Err(Refusal::Param(format!(
            "domain must be one of ('baffled', 'suspended'), got {}.",
            repr_str(&tier)
        )));
    }
    let size = fnum(p, "room_size", VKROOM_SIZE_DEFAULT)?;
    let h = fnum(p, "air_h", VKROOM_H_DEFAULT)?;
    let zeta = fnum(p, "wall_zeta", VKROOM_ZETA_DEFAULT)?;
    let nv = p
        .get("plate_N")
        .cloned()
        .unwrap_or(json!(VKROOM_PLATE_N_DEFAULT));
    let n_plate = py_int(&nv).map_err(|_| {
        Refusal::Param(format!("plate_N must be an integer, got {}.", py_repr(&nv)))
    })?;
    if !(VKROOM_SIZE_MIN..=VKROOM_SIZE_MAX).contains(&size) {
        return Err(Refusal::Param(format!(
            "room_size must be in [{}, {}] m, got {}.",
            pf(VKROOM_SIZE_MIN),
            pf(VKROOM_SIZE_MAX),
            pf(size)
        )));
    }
    if !(VKROOM_H_MIN..=VKROOM_H_MAX).contains(&h) {
        return Err(Refusal::Param(format!(
            "air_h must be in [{}, {}] m, got {}. This range is a STABILITY limit, not a taste: \
             the room sets the sample rate, and at h = 0.014 the plate's Picard iteration stops \
             converging while h >= 0.017 is an immediate NaN.",
            pf(VKROOM_H_MIN),
            pf(VKROOM_H_MAX),
            pf(h)
        )));
    }
    if !(0.0 < zeta && zeta <= VKROOM_ZETA_MAX) {
        return Err(Refusal::Param(format!(
            "wall_zeta must be in (0, {}], got {}.",
            pf(VKROOM_ZETA_MAX),
            pf(zeta)
        )));
    }
    if !(VKROOM_PLATE_N_MIN..=VKROOM_PLATE_N_MAX).contains(&n_plate) {
        return Err(Refusal::Param(format!(
            "plate_N must be in [{VKROOM_PLATE_N_MIN}, {VKROOM_PLATE_N_MAX}], got {n_plate}."
        )));
    }
    let lam_air = VKROOM_CFL / 3.0f64.sqrt();
    let fs = C0_AIR / (lam_air * h);
    let n_room = round_int(size / h);
    let side = n_room as f64 * h;
    let z = ab::impedance_from_zeta(zeta, RHO0_AIR, C0_AIR);
    let rp = ab::Params::new(
        [side; 3],
        fs,
        h,
        [Wall::from_z(z); 6],
        None,
        RHO0_AIR,
        C0_AIR,
    )
    .map_err(construction)?;
    let mut room = AirBox::new(rp);
    let rn = room.p.n;
    let nodes = (rn[0] + 1) * (rn[1] + 1) * (rn[2] + 1);
    if nodes > VKROOM_NODE_MAX {
        return Err(Refusal::Param(format!(
            "the room is {} nodes (> {}). Shrink room_size — air_h cannot be coarsened here, \
             because it is what holds the plate's iteration together.",
            commas(nodes as i64),
            commas(VKROOM_NODE_MAX as i64)
        )));
    }
    let spec = VkSpec {
        lx: VKROOM_PLATE_L,
        ly: VKROOM_PLATE_L,
        young: 2.0e11,
        thickness: 1.0e-3,
        nu: 0.3,
        rho: 7800.0,
        fs,
        n: n_plate,
        boundary: Some(Boundary::Free),
        nonlinear,
        couple_max_iter: VKROOM_SWEEP_CAP,
        ..VkSpec::default()
    };
    let plate = VkPlate::new(VkParams::new(&spec).map_err(construction)?);
    let seam = VkSeam::new(plate);
    let suspended = tier == "suspended";
    let grid = if suspended {
        RoomGrid::suspended(seam, &mut room, "z", n_room / 2, None, Spreading::Bilinear)
    } else {
        RoomGrid::baffled(seam, &mut room, "z0", None, Spreading::Bilinear)
    }
    .map_err(construction)?;
    Ok(Scene {
        grid,
        room,
        suspended,
        fs,
        h,
        lam_air,
        nodes,
        zeta,
        size,
        n_plate,
    })
}

/// `_vkroom_strike`: a centred raised-Gaussian of peak `w_over_e e`, on the live nodes.
fn strike(plate: &VkPlate, w_over_e: f64) -> Vec<f64> {
    let lin = &plate.p.lin;
    let w = VKROOM_STRIKE_WIDTH * lin.lx;
    let field: Vec<f64> = lin
        .x
        .iter()
        .zip(&lin.y)
        .map(|(&x, &y)| {
            let (dx, dy) = (x - 0.5 * lin.lx, y - 0.5 * lin.ly);
            w_over_e * plate.p.thickness * (-((dx * dx + dy * dy) / (w * w))).exp()
        })
        .collect();
    plate.p.to_live(&field)
}

/// `_vkroom_modes`: mass-orthonormal free-plate modes, their frequencies, and which the air
/// grid resolves (at least five air cells per structural wavelength). Vectors as columns.
fn modes(plate: &VkPlate, h_air: f64) -> Result<Modes, Refusal> {
    let lin = &plate.p.lin;
    let n = lin.n_live;
    let k2 = scalar_pow(lin.kappa, 2.0);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = lin.stiffness.get(i, j) * k2;
        }
    }
    let (vals, v) =
        generalized_eigen_diag(&a, &lin.w, n).map_err(|e| Refusal::Internal(e.to_string()))?;
    let two_pi = 2.0 * std::f64::consts::PI;
    let f: Vec<f64> = vals.iter().map(|&x| x.max(0.0).sqrt() / two_pi).collect();
    let resolved: Vec<bool> = f
        .iter()
        .map(|&fi| {
            let beta = (two_pi * fi.max(1e-9) / lin.kappa).sqrt();
            (two_pi / beta) / h_air >= VKROOM_CELLS_PER_WAVE
        })
        .collect();
    let vecs: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| v[i * n + j]).collect())
        .collect();
    Ok((vecs, f, resolved))
}

/// One run's record.
#[derive(Default)]
struct Run {
    e_total: Vec<f64>,
    e_plate: Vec<f64>,
    e_acoustic: Vec<f64>,
    e_dissipated: Vec<f64>,
    radiated: Vec<f64>,
    injected: Vec<f64>,
    p_mic: Vec<f64>,
    sigma_shape: Vec<f64>,
    sigma_res: Vec<f64>,
    sigma_mono: Vec<f64>,
    shares: Vec<Vec<f64>>,
    in_band: Vec<f64>,
    plate_frames: Vec<Vec<f64>>,
    plate_steps: Vec<usize>,
    slices: Vec<Vec<f64>>,
    slice_steps: Vec<usize>,
    n_not_converged: i64,
    worst_residual: f64,
    max_iters: usize,
    sweep_total: usize,
}

/// The run's cadences and what it captures.
struct Plan<'a> {
    n_steps: usize,
    mic: usize,
    planes: &'a [Plane],
    e_stride: usize,
    proj_stride: usize,
    plate_stride: usize,
    plate_until: usize,
    slice_stride: usize,
    slice_until: usize,
    capture: bool,
}

/// `_run_vkroom`: the coupled scene, the claim's windowed quadratic forms, both ledgers, frames.
fn run(
    grid: &mut Grid,
    room: &mut AirBox,
    suspended: bool,
    plan: &Plan,
    vecs: &[Vec<f64>],
    resolved: &[bool],
) -> Result<Run, Refusal> {
    let windows = VKROOM_WINDOWS;
    let n_steps = plan.n_steps;
    let faces = if suspended { 2.0 } else { 1.0 };
    let lin = grid.seam.plate.p.lin.clone();
    let areas = lin.w.clone();
    let area = lin.lx * lin.ly;
    let k = lin.k;
    let load = grid.port.load_matrix().clone();
    let m = n_steps / plan.e_stride + 1;
    let per = (n_steps / windows).max(1);
    let n_modes = vecs.len();
    let mut acc = vec![vec![0.0f64; n_modes]; windows];
    let mut sums = vec![[0.0f64; 5]; windows]; // p_shape, p_res, p_mono, v2, v2_res
    let mut counts = vec![0usize; windows];
    let res_idx: Vec<usize> = (0..n_modes).filter(|&j| resolved[j]).collect();
    let mut r = Run::default();

    let sample_energy = |g: &Grid, room: &AirBox, r: &mut Run| {
        r.e_plate.push(g.seam.plate.energy());
        r.e_acoustic.push(room.acoustic_energy());
        r.e_dissipated.push(room.dissipated);
        r.radiated.push(g.radiated_energy());
        r.injected.push(room.injected);
        r.e_total.push(g.energy() + room.energy());
    };
    let mut sample_claim = |g: &Grid, i: usize| {
        let w = (i / per).min(windows - 1);
        let pl = &g.seam.plate;
        let v: Vec<f64> =
            pl.u.iter()
                .zip(&pl.u_prev)
                .map(|(a, b)| (a - b) / k)
                .collect();
        let av: Vec<f64> = areas.iter().zip(&v).map(|(a, x)| a * x).collect();
        let c: Vec<f64> = vecs.iter().map(|col| dot(col, &av)).collect();
        let mut v_res = vec![0.0; v.len()];
        for &j in &res_idx {
            for (vr, x) in v_res.iter_mut().zip(&vecs[j]) {
                *vr += x * c[j];
            }
        }
        let quad = |x: &[f64]| dot(x, &load.matvec(x));
        sums[w][0] += faces * quad(&v);
        sums[w][1] += faces * quad(&v_res);
        sums[w][2] += faces * RHO0_AIR * C0_AIR * scalar_pow(sum(&av), 2.0) / area;
        sums[w][3] += dot(&av, &v) / area;
        let avr: Vec<f64> = areas.iter().zip(&v_res).map(|(a, x)| a * x).collect();
        sums[w][4] += dot(&avr, &v_res) / area;
        for (a, cj) in acc[w].iter_mut().zip(&c) {
            *a += cj * cj;
        }
        counts[w] += 1;
    };

    sample_energy(grid, room, &mut r);
    sample_claim(grid, 0);
    if plan.capture {
        r.p_mic.push(room.pressure[plan.mic]);
        if plan.plate_until >= 1 {
            r.plate_frames.push(grid.seam.plate.state());
            r.plate_steps.push(0);
        }
        if plan.slice_until >= 1 {
            r.slices.push(take_slices(room, plan.planes));
            r.slice_steps.push(0);
        }
    }
    for i in 1..=n_steps {
        grid.step(room, None)
            .map_err(|e| Refusal::Internal(e.to_string()))?;
        room.step();
        let pl = &grid.seam.plate;
        if !pl.converged {
            r.n_not_converged += 1;
        }
        if pl.last_residual > r.worst_residual {
            r.worst_residual = pl.last_residual;
        }
        r.max_iters = r.max_iters.max(pl.n_iters);
        r.sweep_total += pl.n_iters;
        if plan.capture {
            r.p_mic.push(room.pressure[plan.mic]);
        }
        if i % plan.proj_stride == 0 {
            sample_claim(grid, i);
        }
        if i % plan.e_stride == 0 && i / plan.e_stride < m {
            sample_energy(grid, room, &mut r);
        }
        if plan.capture && i <= plan.plate_until && i % plan.plate_stride == 0 {
            r.plate_frames.push(grid.seam.plate.state());
            r.plate_steps.push(i);
        }
        if plan.capture && i <= plan.slice_until && i % plan.slice_stride == 0 {
            r.slices.push(take_slices(room, plan.planes));
            r.slice_steps.push(i);
        }
    }
    // Non-finite is the OTHER failure mode, distinct from non-convergence.
    if !r.e_total.iter().all(|v| v.is_finite())
        || (plan.capture && !r.p_mic.iter().all(|v| v.is_finite()))
    {
        return Err(Refusal::Param(format!(
            "simulation produced non-finite energy (the plate's iteration diverged). Lower \
             w_over_e — it is capped at {} because {} was measured dead — or refine air_h.",
            pf(VKROOM_WOVERE_MAX),
            pf(VKROOM_WOVERE_CLIFF)
        )));
    }
    let denom = RHO0_AIR * C0_AIR * area;
    for w in 0..windows {
        let n = counts[w].max(1) as f64;
        let (v2, v2_res) = (sums[w][3] / n, sums[w][4] / n);
        r.sigma_shape.push(if v2 > 0.0 {
            (sums[w][0] / n) / (denom * v2)
        } else {
            0.0
        });
        r.sigma_res.push(if v2_res > 0.0 {
            (sums[w][1] / n) / (denom * v2_res)
        } else {
            0.0
        });
        r.sigma_mono.push(if v2 > 0.0 {
            (sums[w][2] / n) / (denom * v2)
        } else {
            0.0
        });
        r.in_band.push(if v2 > 0.0 { v2_res / v2 } else { 0.0 });
        let total = sum(&acc[w]);
        r.shares.push(if total > 0.0 {
            acc[w].iter().map(|a| a / total).collect()
        } else {
            acc[w].clone()
        });
    }
    Ok(r)
}

/// `_vkroom_spread`: max over min across the windows.
fn spread(x: &[f64]) -> f64 {
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    if lo > 0.0 {
        x.iter().copied().fold(f64::NEG_INFINITY, f64::max) / lo
    } else {
        0.0
    }
}

/// `_vkroom_claim_block`: the separation between the loud plate and its linear twin.
fn claim_block(main: &Run, twin: &Run, resolved: &[bool], freqs: &[f64], fs: f64) -> Value {
    let drift = |r: &Run| {
        let (a, b) = (&r.shares[r.shares.len() - 1], &r.shares[0]);
        let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
        0.5 * sum(&d)
    };
    let n_res = resolved.iter().filter(|&&b| b).count();
    let f_res_max = freqs
        .iter()
        .zip(resolved)
        .filter(|(_, &b)| b)
        .map(|(&f, _)| f)
        .fold(f64::NEG_INFINITY, f64::max);
    let ms = np_mean(&main.sigma_shape);
    json!({
        "kind": "vkroom",
        "sigma_shape": finite_list(&main.sigma_shape, Some(6)),
        "sigma_resolved": finite_list(&main.sigma_res, Some(6)),
        "sigma_mono": finite_list(&main.sigma_mono, Some(9)),
        "sigma_shape_twin": finite_list(&twin.sigma_shape, Some(6)),
        "sigma_resolved_twin": finite_list(&twin.sigma_res, Some(6)),
        "spread": num(round_nd(spread(&main.sigma_shape), 6)),
        "spread_resolved": num(round_nd(spread(&main.sigma_res), 6)),
        "spread_twin": num(round_nd(spread(&twin.sigma_shape), 6)),
        "spread_resolved_twin": num(round_nd(spread(&twin.sigma_res), 6)),
        "modal_drift": num(round_nd(drift(main), 6)),
        "modal_drift_twin": num(round_nd(drift(twin), 6)),
        "in_band": finite_list(&main.in_band, Some(6)),
        "n_resolved": int(n_res as i64),
        "n_modes": int(resolved.len() as i64),
        "f_resolved_max": num(round_nd(if n_res > 0 { f_res_max } else { 0.0 }, 2)),
        "windows": int(main.sigma_shape.len() as i64),
        "window_s": num(round_nd(
            (main.p_mic.len() as f64 - 1.0) / fs / (main.sigma_shape.len().max(1) as f64),
            6,
        )),
        "mono_ratio": num(round_nd(
            if ms > 0.0 { np_mean(&main.sigma_mono) / ms } else { 0.0 },
            12,
        )),
    })
}

/// `_build_payload_vkroom`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let w_over_e = fnum(p, "w_over_e", VKROOM_WOVERE_DEFAULT)?;
    let audio_dur = fnum(p, "audio_duration", VKROOM_AUDIO_DEFAULT)?;
    let mic_frac = fnum(p, "mic_position", VKROOM_MIC_DEFAULT)?;
    let slice_frac = fnum(p, "slice_position", 0.5)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;
    let nonlinear = as_bool(p.get("nonlinear"), true);

    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    if !(VKROOM_WOVERE_MIN..=VKROOM_WOVERE_MAX).contains(&w_over_e) {
        return Err(Refusal::Param(format!(
            "w_over_e must be in [{}, {}], got {}. The ceiling is not a taste: at this rig {} was \
             measured DEAD (the plate's iteration diverges at step 0), and it is a cliff rather \
             than a slope -- 3.2 still runs, at 109 of 120 sweeps.",
            pf(VKROOM_WOVERE_MIN),
            pf(VKROOM_WOVERE_MAX),
            pf(w_over_e),
            pf(VKROOM_WOVERE_CLIFF)
        )));
    }
    if !(0.0 < audio_dur && audio_dur <= VKROOM_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(VKROOM_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(0.15..=1.0).contains(&mic_frac) {
        return Err(Refusal::Param(format!(
            "mic_position must be in [0.15, 1.0], got {}.",
            pf(mic_frac)
        )));
    }
    if !(0.0..=1.0).contains(&slice_frac) {
        return Err(Refusal::Param(format!(
            "slice_position must be in [0, 1], got {}.",
            pf(slice_frac)
        )));
    }

    let mut sc = build(p, nonlinear)?;
    let fs = sc.fs;
    let n_steps = (round_int(audio_dur * fs).max(VKROOM_WINDOWS as i64)) as usize;
    let twin_runs = if nonlinear { 2.0 } else { 1.0 };
    let room_work = sc.nodes as f64 * n_steps as f64 * twin_runs;
    let n_live = sc.grid.seam.plate.p.lin.n_live;
    let plate_work = n_live as f64 * n_steps as f64 * VKROOM_SWEEP_CAP as f64;
    if room_work > VKROOM_ROOM_WORK_MAX {
        return Err(Refusal::Param(format!(
            "the room's budget is exceeded ({} node-steps > {}; the control twin's room counts \
             too). Shrink room_size or shorten audio_duration -- air_h cannot be coarsened here.",
            sci(room_work, 2),
            sci(VKROOM_ROOM_WORK_MAX, 2)
        )));
    }
    if plate_work > VKROOM_PLATE_WORK_MAX {
        return Err(Refusal::Param(format!(
            "the plate's budget is exceeded ({} coupled node-solves > {}, priced at the \
             {VKROOM_SWEEP_CAP}-sweep cap rather than the measured mean). Shorten audio_duration \
             or lower plate_N.",
            sci(plate_work, 2),
            sci(VKROOM_PLATE_WORK_MAX, 2)
        )));
    }

    let (vecs, freqs, resolved) = modes(&sc.grid.seam.plate, sc.h)?;
    // The FOURTH mode: a free plate's nullspace is exactly {1, x, y}.
    let f_lin = if freqs.len() > VKROOM_NULLSPACE_DIM {
        freqs[VKROOM_NULLSPACE_DIM]
    } else {
        1.0
    };

    // -- the geometry: the plate's centre and an off-axis mic
    let la = sc.room.p.l_actual;
    let z_plate = if sc.suspended {
        (sc.room.p.n[2] / 2) as f64 * sc.h
    } else {
        0.0
    };
    let centre = [0.5 * la[0], 0.5 * la[1], z_plate];
    let mic_want: [f64; 3] =
        std::array::from_fn(|d| centre[d] + mic_frac * (VKROOM_MIC_FAR[d] * la[d] - centre[d]));
    let mic_index = index(&sc.room, mic_want);
    let mic_flat = ab::flat(
        sc.room.p.p_shape(),
        mic_index[0],
        mic_index[1],
        mic_index[2],
    );
    let planes = slice_planes(&sc.room, slice_frac);
    let per_frame: usize = planes
        .iter()
        .map(|pl| {
            pl.json["nu"].as_u64().unwrap() as usize * pl.json["nv"].as_u64().unwrap() as usize
        })
        .sum();

    // TWO animation clocks: the plate pane on its first flexural mode, the slices on transit.
    let ns = n_steps as i64;
    let mut plate_stride = round_int((fs / f_lin.max(1.0)) / fpp as f64).max(1);
    let plate_until = ns.min(plate_stride.max(round_int(VKROOM_ANIM_WIN * fs)));
    if plate_until / plate_stride > MAX_FRAMES {
        plate_stride = ((plate_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let mut slice_stride = round_int(VKROOM_CELLS_PER_FRAME / sc.lam_air).max(1);
    let slice_until = ns.min(slice_stride.max(round_int(VKROOM_ROOM_ANIM_WIN * fs)));
    if slice_until / slice_stride > MAX_FRAMES {
        slice_stride = ((slice_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let e_stride = ((n_steps as f64 / VKROOM_TRACE_POINTS).ceil() as usize).max(1);
    let proj_stride =
        ((n_steps as f64 / (VKROOM_WINDOWS as f64 * VKROOM_PROJ_SAMPLES)).ceil() as usize).max(1);

    let u0 = strike(&sc.grid.seam.plate, w_over_e);
    sc.grid.set_state(&u0, &vec![0.0; u0.len()]);
    let plan = Plan {
        n_steps,
        mic: mic_flat,
        planes: &planes,
        e_stride,
        proj_stride,
        plate_stride: plate_stride as usize,
        plate_until: plate_until as usize,
        slice_stride: slice_stride as usize,
        slice_until: slice_until as usize,
        capture: true,
    };
    let main = run(
        &mut sc.grid,
        &mut sc.room,
        sc.suspended,
        &plan,
        &vecs,
        &resolved,
    )?;

    // -- the control twin: the same scene with the coupling off, and it is the anchor
    let twin = if nonlinear {
        let mut tw = build(p, false)?;
        let t0 = strike(&tw.grid.seam.plate, w_over_e);
        tw.grid.set_state(&t0, &vec![0.0; t0.len()]);
        let tplan = Plan {
            plate_stride: 1,
            plate_until: 0,
            slice_stride: 1,
            slice_until: 0,
            capture: false,
            ..plan
        };
        Some(run(
            &mut tw.grid,
            &mut tw.room,
            tw.suspended,
            &tplan,
            &vecs,
            &resolved,
        )?)
    } else {
        None
    };
    let twin_ref = twin.as_ref().unwrap_or(&main);

    // -- the two ledgers (neither is the gate) and the convergence record (which is)
    let total = &main.e_total;
    let e0 = total[0];
    let scale = e0.abs().max(1e-300);
    let resid: Vec<f64> = main
        .radiated
        .iter()
        .zip(&main.injected)
        .map(|(a, b)| (a - b).abs() / scale)
        .collect();
    let frac = |a: &[f64]| {
        a.iter()
            .zip(total)
            .map(|(x, t)| x / t)
            .collect::<Vec<f64>>()
    };
    let e_steps: Vec<f64> = (0..total.len())
        .map(|j| (j * e_stride) as f64 / fs)
        .collect();
    let fmax = |a: &[f64]| a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let fmin = |a: &[f64]| a.iter().copied().fold(f64::INFINITY, f64::min);
    let last = total.len() - 1;
    let ledger = json!({
        "kind": "vkroom",
        "time": finite_list(&e_steps, Some(6)),
        "e_plate_frac": finite_list(&frac(&main.e_plate), None),
        "e_acoustic_frac": finite_list(&frac(&main.e_acoustic), None),
        "e_dissipated_frac": finite_list(&frac(&main.e_dissipated), None),
        "radiated_frac": finite_list(&frac(&main.radiated), None),
        "total_frac": finite_list(&frac(total), None),
        "residual": finite_list(&resid, Some(3)),
        "residual_max": num(fmax(&resid)),
        "e_stride": int(e_stride as i64),
        "proj_stride": int(proj_stride as i64),
        "n_samples": int(total.len() as i64),
        "acoustic_frac_peak": num(round_nd(fmax(&frac(&main.e_acoustic)), 8)),
        "radiated_frac_end": num(round_nd(main.radiated[last] / total[last], 8)),
        "total_drift": num((fmax(total) - fmin(total)) / scale),
    });
    let plate = &sc.grid.seam.plate;
    let convergence = json!({
        "all_converged": main.n_not_converged == 0,
        "n_not_converged": int(main.n_not_converged),
        "worst_residual": num(main.worst_residual),
        "max_iters": int(main.max_iters as i64),
        "mean_iters": num(round_nd(main.sweep_total as f64 / n_steps.max(1) as f64, 2)),
        "couple_tol": num(plate.p.couple_tol),
        "cap": int(VKROOM_SWEEP_CAP),
        "twin_max_iters": int(twin_ref.max_iters as i64),
    });
    let claim = claim_block(&main, twin_ref, &resolved, &freqs, fs);

    let lin = &plate.p.lin;
    let (ny, nx) = (lin.mask.nrows(), lin.mask.ncols());
    let (plate_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&main.plate_frames, lin.mask.flags(), ny, nx);
    let nf = main.plate_frames.len();
    let plate_amp = plate_dec.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let slices: Vec<f64> = main.slices.iter().flatten().copied().collect();
    let live: Vec<f64> = slices
        .iter()
        .filter(|&&v| v != 0.0)
        .map(|v| v.abs())
        .collect();
    let slice_amp = slices.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut reference = if live.is_empty() {
        0.0
    } else {
        np_percentile(&live, VKROOM_SCALE_PCTL)
    };
    if reference <= 0.0 || reference.is_nan() {
        reference = if slice_amp > 0.0 { slice_amp } else { 1.0 };
    }
    let (audio48, peak) = resample_normalize(&main.p_mic, fs); // the audio IS the mic
    let plate_times: Vec<f64> = main.plate_steps.iter().map(|&i| i as f64 / fs).collect();
    let slice_times: Vec<f64> = main.slice_steps.iter().map(|&i| i as f64 / fs).collect();
    let planes_json: Vec<Value> = planes
        .iter()
        .map(|pl| Value::Object(pl.json.clone()))
        .collect();
    let mic_snapped = snap(&sc.room, mic_want);
    let rl = sc.room.p.l_actual;
    let tier = if sc.suspended { "suspended" } else { "baffled" };
    Ok(json!({
        "model": "vkroom",
        "tier": tier,
        "boundary": VKROOM_BOUNDARY,
        "nonlinear": nonlinear,
        "fs_sim": num(round_nd(fs, 3)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(lin.lx, 6)), "extent_y": num(round_nd(lin.ly, 6)),
            "domain": "rectangle",
        },
        "frames": {
            "b64": b64f32(&plate_dec),
            "n_frames": int(nf as i64), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "width": int(nx_dec as i64), "dims": int(2),
        },
        "mask": {"b64": b64u8(&mask_dec), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64)},
        "frame_times": finite_list(&plate_times, Some(6)),
        "anim_dt": num(plate_stride as f64 / fs),
        "room_frames": {
            "b64": b64f32(&slices),
            "n_frames": int(main.slices.len() as i64),
            "width": int(per_frame as i64),
            "dims": int(3),
            "kind": "slices",
            "planes": planes_json,
            "stride_floats": int(per_frame as i64),
            "times": finite_list(&slice_times, Some(6)),
            "anim_dt": num(slice_stride as f64 / fs),
            "amp": num(slice_amp),
            "scale": {"map": "asinh", "ref": num(reference), "amp": num(slice_amp),
                      "pctl": num(VKROOM_SCALE_PCTL)},
        },
        "playback_speed": num(playback_speed),
        "field_amp": num(plate_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&e_steps, total, true, 0.0, EnergyOpts {
            no_decay_oracle: true,
            convergence: Some(convergence.clone()),
            ..EnergyOpts::default()
        }),
        "meta": {
            "kappa": num(round_nd(lin.kappa, 4)),
            "e": num(plate.p.thickness),
            "f1": num(round_nd(f_lin, 3)),
            "num_steps": int(n_steps as i64),
            "n_frames": int(nf as i64),
            "w_over_e": num(w_over_e),
            "strike_width": num(VKROOM_STRIKE_WIDTH),
            "plate": {
                "L": num(VKROOM_PLATE_L), "N": int(sc.n_plate), "n_live": int(n_live as i64),
                "boundary": VKROOM_BOUNDARY, "cliff_w_over_e": num(VKROOM_WOVERE_CLIFF),
            },
            "room": {
                "L": rl.iter().map(|&v| num(round_nd(v, 4))).collect::<Vec<_>>(),
                "L_requested": num(round_nd(sc.size, 4)),
                "N": sc.room.p.n.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
                "h": num(sc.h),
                "nodes": int(sc.nodes as i64),
                "cfl": num(VKROOM_CFL),
                "lam_air": num(round_nd(sc.lam_air, 6)),
                "zeta": num(sc.zeta),
                "c0": num(round_nd(C0_AIR, 3)),
                "mic_at": mic_snapped.iter().map(|&v| num(round_nd(v, 4))).collect::<Vec<_>>(),
                "mic_cell": mic_index.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
                "room_work": num(room_work),
                "plate_work": num(plate_work),
            },
            "ledger": ledger,
            "convergence": convergence,
            "claim": claim,
        },
    }))
}
