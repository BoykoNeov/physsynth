//! The drum in the room — `docs/dev/membrane-mapping-plan.md` §7 and §13: a round head mounted in
//! a wall (baffled) or hanging in the air (suspended) inside the 3-D room, its physical or musical
//! controls, and what the air does to it.
//!
//! Born after the Python viewer, so it has no frozen case; `tests/drumroom.rs` covers it.
//!
//! THE READ-OUT: the air lowers a light head's pitch by one to five semitones (mass loading, both
//! faces when hung) and drains its energy (plan §9). The page measures that per mode. The head is
//! projected onto its own in-vacuo modes every step; the same projection is taken from a bare twin
//! — the identical head with no room, the same duration and the same search — and the shift is
//! room minus twin, so the detector's own bias cancels. The mapping's exact values are the target
//! column beside them.
//!
//! Three defences the measurement needs, each from Part 0:
//! - **a degenerate pair is one two-dimensional mode**: its two channels' power spectra are summed
//!   before a peak is picked, so the eigensolver's basis inside the pair cannot reach the payload;
//! - **each mode is searched only between its in-vacuo neighbours**: energy leaking from a lower
//!   mode into a higher mode's channel once out-shouted it by 1,800 cents;
//! - **two peaks of similar height read "split"** (the head and a room resonance sharing a note),
//!   and none reads "unresolved" — never a number for either.
//!
//! The room sets the sample rate (the 3-D CFL), so the drum's Courant number is a read-out and the
//! request's `sample_rate` and `lambda` are ignored; a refusal that would say "raise fs" says to
//! refine the air grid instead.

use physsynth_analysis::spectrum::{magnitude_spectrum, parabolic_refine};
use physsynth_core::airbox::{self as ab, AirBox, Wall};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{MembraneSeam, RoomGrid};
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::fmt::py_float;
use physsynth_core::membrane::{self as mem, Domain, Membrane};
use physsynth_core::membrane_tuning::{mode_frequency, mode_t60, Geometry};
use serde_json::{json, Value};

use crate::airbox::{index, np_percentile, slice_planes, snap, take_slices, Plane};
use crate::energy::{energy_block, EnergyOpts};
use crate::membrane::{decimate_field_mask, musical_from_request, tuning_block};
use crate::membrane::{Musical, MEMBRANE_N_MAX};
use crate::plate::is_musical;
use crate::py::{
    b64f32, b64u8, commas, dot, finite_list, int, num, py_int, py_repr, py_str, repr_str,
    round_int, round_nd, sci,
};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::{construction, int_at_least_one, FRAMES_PER_PERIOD, MAX_FRAMES, SPEED_MAX};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// The two mountings the `domain` select carries.
pub const DRUMROOM_TIERS: [&str; 2] = ["baffled", "suspended"];
/// The two wall terminations the `walls` select carries; the first is the default.
pub const DRUMROOM_WALLS: [&str; 2] = ["absorbing", "rigid"];
/// The room's Courant fraction — a constant, not a slider.
pub const DRUMROOM_CFL: f64 = 0.90;
/// Air grid default (m): 40 kHz, plan §9's rig.
pub const DRUMROOM_H_DEFAULT: f64 = 0.0165;
/// Air grid range (m).
pub const DRUMROOM_H_MIN: f64 = 0.010;
/// See [`DRUMROOM_H_MIN`].
pub const DRUMROOM_H_MAX: f64 = 0.025;
/// Room side default (m, cube).
pub const DRUMROOM_SIZE_DEFAULT: f64 = 0.48;
/// Room side range (m).
pub const DRUMROOM_SIZE_MIN: f64 = 0.30;
/// See [`DRUMROOM_SIZE_MIN`].
pub const DRUMROOM_SIZE_MAX: f64 = 0.80;
/// Head radius default (m).
pub const DRUMROOM_RADIUS_DEFAULT: f64 = 0.12;
/// Head radius range (m).
pub const DRUMROOM_RADIUS_MIN: f64 = 0.05;
/// See [`DRUMROOM_RADIUS_MIN`].
pub const DRUMROOM_RADIUS_MAX: f64 = 0.30;
/// Head grid default.
pub const DRUMROOM_N_DEFAULT: i64 = 24;
/// Head grid range. Not below 10: at `N` 8 and 9 the grid splits the (2,1) pair by 2.4 and 4.8 %,
/// too close to the 7 % gap to the next distinct mode for [`DRUMROOM_PAIR_TOL`] to sit between.
pub const DRUMROOM_N_MIN: i64 = 10;
/// See [`DRUMROOM_N_MIN`].
pub const DRUMROOM_N_MAX: i64 = 40;
/// Audio default (s).
pub const DRUMROOM_AUDIO_DEFAULT: f64 = 0.35;
/// Longest audio (s).
pub const DRUMROOM_AUDIO_MAX: f64 = 1.0;
/// Room node ceiling.
pub const DRUMROOM_NODE_MAX: usize = 300_000;
/// `nodes x steps`.
pub const DRUMROOM_ROOM_WORK_MAX: f64 = 2.0e9;
/// Mode groups (distinct in-vacuo frequencies) the air read-out reports.
pub const DRUMROOM_MODES: usize = 6;
/// Eigenpairs solved for, enough to fill [`DRUMROOM_MODES`] groups with every pair complete.
const DRUMROOM_EIGS: usize = 14;
/// Eigenvalues within this fraction of a group's first, in frequency, join it. The square grid
/// splits a circle's `cos 2θ`/`sin 2θ` pair by its staircase rim, by an amount that depends on
/// `N` alone and not monotonically (3 cents at the default `N`, 54 at `N` 12), while the next
/// distinct mode stays more than 7 % above the pair's lower member. Every `N` the page allows
/// is checked by `mod tests` below.
pub const DRUMROOM_PAIR_TOL: f64 = 0.04;
/// A second peak at least this fraction of the first, inside a mode's window, reads "split".
pub const DRUMROOM_SPLIT_RATIO: f64 = 0.5;
/// A ring time is quoted only past this much decay over the run (dB).
pub const DRUMROOM_MIN_DECAY_DB: f64 = 6.0;
/// ... and only for a fit at least this straight (R² of the log-energy line).
pub const DRUMROOM_MIN_R2: f64 = 0.9;
/// The fit stops once the energy has fallen this far (dB): the default head's fundamental falls in
/// a straight line to about −40 dB, then flattens onto the slower modes' leakage.
pub const DRUMROOM_FIT_DEPTH_DB: f64 = 25.0;
/// Energy-channel samples.
pub const DRUMROOM_TRACE_POINTS: f64 = 400.0;
/// Seconds of head animation.
pub const DRUMROOM_ANIM_WIN: f64 = 0.02;
/// Seconds of room slices.
pub const DRUMROOM_ROOM_ANIM_WIN: f64 = 0.004;
/// One slice frame per cell of acoustic travel.
pub const DRUMROOM_CELLS_PER_FRAME: f64 = 1.0;
/// The far end of the mic's travel, off-axis.
pub const DRUMROOM_MIC_FAR: [f64; 3] = [0.90, 0.86, 0.93];
/// Mic default.
pub const DRUMROOM_MIC_DEFAULT: f64 = 0.75;
/// Percentile of the live slice field that sets the asinh reference.
pub const DRUMROOM_SCALE_PCTL: f64 = 55.0;

fn pf(x: f64) -> String {
    py_float(x)
}

type Grid = RoomGrid<MembraneSeam>;

/// The scene: the wrapped head, its room, its bare twin, and the scalars.
struct Scene {
    grid: Grid,
    room: AirBox,
    twin: Membrane,
    suspended: bool,
    walls: &'static str,
    fs: f64,
    h: f64,
    lam_air: f64,
    nodes: usize,
    size: f64,
    radius: f64,
    musical: Option<Musical>,
}

/// A refusal from the head's constructor, with the room's own remedy: here the room sets the
/// sample rate, so "raise fs" means a finer air grid.
fn head_refusal(msg: String) -> Refusal {
    if msg.contains("CFL") {
        Refusal::Param(format!(
            "{msg} Here the room sets the sample rate: refine the air grid (a smaller air_h) or \
             coarsen the head's grid (a smaller N)."
        ))
    } else {
        Refusal::Param(msg)
    }
}

/// The scene a request describes.
fn build(p: &Value) -> Result<Scene, Refusal> {
    let tv = p.get("domain").cloned().unwrap_or(json!("baffled"));
    let tier = py_str(&tv);
    if !DRUMROOM_TIERS.contains(&tier.as_str()) {
        return Err(Refusal::Param(format!(
            "domain must be one of ('baffled', 'suspended'), got {}.",
            repr_str(&tier)
        )));
    }
    let wv = p
        .get("walls")
        .and_then(Value::as_str)
        .unwrap_or(DRUMROOM_WALLS[0]);
    let walls = *DRUMROOM_WALLS.iter().find(|w| **w == wv).ok_or_else(|| {
        Refusal::Param(format!(
            "walls must be one of ('absorbing', 'rigid'), got {}.",
            repr_str(wv)
        ))
    })?;
    let size = fnum(p, "room_size", DRUMROOM_SIZE_DEFAULT)?;
    let h = fnum(p, "air_h", DRUMROOM_H_DEFAULT)?;
    let radius = fnum(p, "radius", DRUMROOM_RADIUS_DEFAULT)?;
    let nv = p.get("N").cloned().unwrap_or(json!(DRUMROOM_N_DEFAULT));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    for (name, v, lo, hi) in [
        ("room_size", size, DRUMROOM_SIZE_MIN, DRUMROOM_SIZE_MAX),
        ("air_h", h, DRUMROOM_H_MIN, DRUMROOM_H_MAX),
        ("radius", radius, DRUMROOM_RADIUS_MIN, DRUMROOM_RADIUS_MAX),
    ] {
        if !(lo..=hi).contains(&v) {
            return Err(Refusal::Param(format!(
                "{name} must be in [{}, {}] m, got {}.",
                pf(lo),
                pf(hi),
                pf(v)
            )));
        }
    }
    if !(DRUMROOM_N_MIN..=DRUMROOM_N_MAX.min(MEMBRANE_N_MAX)).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{DRUMROOM_N_MIN}, {DRUMROOM_N_MAX}] for the drum in the room, got {n}."
        )));
    }

    let lam_air = DRUMROOM_CFL / 3.0f64.sqrt();
    let fs = C0_AIR / (lam_air * h);
    let n_room = round_int(size / h);
    let side = n_room as f64 * h;
    let suspended = tier == "suspended";
    // Part 0's walls: absorbing is the characteristic impedance on every face, but a baffle stays
    // rigid — it is the wall the drum is mounted in.
    let z1 = ab::impedance_from_zeta(1.0, RHO0_AIR, C0_AIR);
    let mut wall_set = if walls == "absorbing" {
        [Wall::Impedance(z1); 6]
    } else {
        [Wall::Rigid; 6]
    };
    if !suspended {
        wall_set[4] = Wall::Rigid;
    }
    let rp = ab::Params::new([side; 3], fs, h, wall_set, None, RHO0_AIR, C0_AIR)
        .map_err(construction)?;
    let mut room = AirBox::new(rp);
    let rn = room.p.n;
    let nodes = (rn[0] + 1) * (rn[1] + 1) * (rn[2] + 1);
    if nodes > DRUMROOM_NODE_MAX {
        return Err(Refusal::Param(format!(
            "the room is {} nodes (> {}). Shrink room_size or coarsen air_h.",
            commas(nodes as i64),
            commas(DRUMROOM_NODE_MAX as i64)
        )));
    }

    // The head: physical (T, rho, sigma) or musical (head, targets) — at the ROOM's sample rate.
    let (params, musical) = if is_musical(p) {
        let geometry = Geometry {
            domain: Domain::Circle,
            lx: 0.0,
            ly: 0.0,
            radius,
            n,
            fs,
        };
        let m = musical_from_request(p, &geometry).map_err(|e| match e {
            Refusal::Param(msg) => head_refusal(msg),
            other => other,
        })?;
        (m.tuning.params.clone(), Some(m))
    } else {
        let t = fnum(p, "T", 3000.0)?;
        let rho = fnum(p, "rho", 0.26)?;
        let sigma = fnum(p, "sigma", 0.0)?;
        if !(t > 0.0 && rho > 0.0) {
            return Err(Refusal::Param("T, rho must both be positive.".into()));
        }
        if sigma < 0.0 {
            return Err(Refusal::Param(format!(
                "sigma (loss) must be >= 0, got {}.",
                pf(sigma)
            )));
        }
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
        .map_err(|e| head_refusal(e.to_string()))?;
        (params, None)
    };
    let twin = Membrane::new(params.clone());
    let seam = MembraneSeam::new(Membrane::new(params));
    let grid = if suspended {
        RoomGrid::suspended(seam, &mut room, "z", n_room / 2, None, Spreading::Bilinear)
    } else {
        RoomGrid::baffled(seam, &mut room, "z0", None, Spreading::Bilinear)
    }
    .map_err(|e| {
        let msg = e.to_string();
        if msg.contains("footprint") {
            // the core's advice names its own `origin=`, which the page does not offer
            Refusal::Param(format!(
                "the head (radius {} m) does not fit the room (side {} m). Shrink radius or \
                 enlarge room_size.",
                pf(radius),
                pf(round_nd(side, 4))
            ))
        } else {
            construction(e)
        }
    })?;
    Ok(Scene {
        grid,
        room,
        twin,
        suspended,
        walls,
        fs,
        h,
        lam_air,
        nodes,
        size,
        radius,
        musical,
    })
}

/// One mode group: the eigenvalues of `−L` within [`DRUMROOM_PAIR_TOL`] of the first, ascending,
/// and their eigenvectors.
struct ModeGroup {
    mus: Vec<f64>,
    vectors: Vec<Vec<f64>>,
}

/// The lowest [`DRUMROOM_MODES`] distinct modes of the head. The last group the solve returns is
/// dropped, because it may hold only one of a pair.
fn mode_groups(p: &mem::Params) -> Result<Vec<ModeGroup>, Refusal> {
    let nev = DRUMROOM_EIGS.min(p.n_live().saturating_sub(1));
    let r = p.radius.expect("the room's drum is round");
    let pairs = eigsh_shift_invert(&p.l.scaled(-1.0), None, -1e-3 * (2.4 / r).powi(2), nev)
        .map_err(|e| Refusal::Internal(format!("{e:?}")))?;
    let mu_tol = (1.0 + DRUMROOM_PAIR_TOL) * (1.0 + DRUMROOM_PAIR_TOL);
    let mut groups: Vec<ModeGroup> = Vec::new();
    for (mu, v) in pairs.values.iter().zip(pairs.vectors) {
        match groups.last_mut() {
            Some(g) if *mu <= g.mus[0] * mu_tol => {
                g.mus.push(*mu);
                g.vectors.push(v);
            }
            _ => groups.push(ModeGroup {
                mus: vec![*mu],
                vectors: vec![v],
            }),
        }
    }
    groups.pop();
    groups.truncate(DRUMROOM_MODES);
    Ok(groups)
}

/// The strike: a raised cosine at the `(pluck_x, pluck_y)` fractions of the disk, `pluck_width`
/// of the radius wide.
fn strike(
    p: &mem::Params,
    radius: f64,
    fx: f64,
    fy: f64,
    w: f64,
    amp: f64,
) -> Result<Vec<f64>, Refusal> {
    let centre = ((2.0 * fx - 1.0) * radius, (2.0 * fy - 1.0) * radius);
    let field = raised_cosine_2d(&p.x, &p.y, centre, w * radius, amp).map_err(construction)?;
    Ok(p.to_live(&field))
}

/// What one mode's channel says: a frequency, or a split, or a peak the window does not hold.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Peak {
    /// One clear peak (Hz).
    Clear(f64),
    /// Two peaks of similar height (Hz): the stronger first.
    Split(f64, f64),
    /// The channel's strongest peak lies outside the window (Hz).
    Outside(f64),
    /// No peak at all.
    Unresolved,
}

/// The peak of a mode group's channels inside `[lo, hi)` Hz, power spectra summed over the group.
/// When a stronger peak lies outside the window the channel is not this mode's: that is reported,
/// not the window's best.
fn group_peak(channels: &[Vec<f64>], fs: f64, lo: f64, hi: f64) -> Peak {
    let spectra: Vec<_> = channels
        .iter()
        .map(|c| magnitude_spectrum(c, fs, 2))
        .collect();
    let (freqs, nfft) = (&spectra[0].freqs, spectra[0].nfft);
    let mag: Vec<f64> = (0..freqs.len())
        .map(|i| {
            spectra
                .iter()
                .map(|s| s.mag[i] * s.mag[i])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut all: Vec<usize> = (1..mag.len().saturating_sub(1))
        .filter(|&i| mag[i] > mag[i - 1] && mag[i] >= mag[i + 1])
        .collect();
    all.sort_by(|&a, &b| mag[b].total_cmp(&mag[a]));
    let Some(&strongest) = all.first() else {
        return Peak::Unresolved;
    };
    let inside = |i: usize| freqs[i] >= lo && freqs[i] < hi;
    if !inside(strongest) {
        return Peak::Outside(parabolic_refine(&mag, strongest, fs, nfft));
    }
    let maxima: Vec<usize> = all.into_iter().filter(|&i| inside(i)).collect();
    let first = strongest;
    let f1 = parabolic_refine(&mag, first, fs, nfft);
    match maxima
        .iter()
        .skip(1)
        .find(|&&i| i.abs_diff(first) > 3 && mag[i] >= DRUMROOM_SPLIT_RATIO * mag[first])
    {
        Some(&second) => Peak::Split(f1, parabolic_refine(&mag, second, fs, nfft)),
        None => Peak::Clear(f1),
    }
}

/// A mode group's ring time from its channels: the energy proxy `a_n² − a_{n+1} a_{n−1}` (exactly
/// `A² r^{2n} sin²φ` for one damped mode), averaged over blocks of one period, fitted on a log
/// line. `Ok(t60)` for a clean decay of at least [`DRUMROOM_MIN_DECAY_DB`]; `Err(reason)` else.
fn group_t60(channels: &[Vec<f64>], fs: f64, f: f64) -> Result<f64, &'static str> {
    let n = channels[0].len();
    let block = ((fs / f).round() as usize).max(4);
    let proxy: Vec<f64> = (1..n - 1)
        .map(|i| {
            channels
                .iter()
                .map(|a| a[i] * a[i] - a[i + 1] * a[i - 1])
                .sum()
        })
        .collect();
    let means: Vec<f64> = proxy
        .chunks_exact(block)
        .map(|c| c.iter().sum::<f64>() / block as f64)
        .collect();
    if means.len() < 4 {
        return Err("too short");
    }
    if means.iter().any(|&m| m <= 0.0) {
        return Err("beats");
    }
    // Fit only the first DRUMROOM_FIT_DEPTH_DB of decay: below it a fast mode's channel reaches a
    // floor of slower modes' leakage, which bends the line.
    let floor = means[0] * 10f64.powf(-DRUMROOM_FIT_DEPTH_DB / 10.0);
    let end = means
        .iter()
        .position(|&m| m <= floor)
        .map_or(means.len(), |i| i + 1);
    let means = &means[..end];
    if means.len() < 4 {
        return Err("too short");
    }
    let t: Vec<f64> = (0..means.len())
        .map(|j| (j as f64 + 0.5) * block as f64 / fs)
        .collect();
    let y: Vec<f64> = means.iter().map(|m| m.ln()).collect();
    let m = t.len() as f64;
    let (tm, ym) = (t.iter().sum::<f64>() / m, y.iter().sum::<f64>() / m);
    let stt: f64 = t.iter().map(|x| (x - tm) * (x - tm)).sum();
    let sty: f64 = t.iter().zip(&y).map(|(x, v)| (x - tm) * (v - ym)).sum();
    let syy: f64 = y.iter().map(|v| (v - ym) * (v - ym)).sum();
    let slope = sty / stt;
    let r2 = if syy > 0.0 {
        sty * sty / (stt * syy)
    } else {
        0.0
    };
    let decay_db = -10.0 * slope * (t[t.len() - 1] - t[0]) / std::f64::consts::LN_10;
    if slope >= 0.0 || decay_db < DRUMROOM_MIN_DECAY_DB {
        return Err("too little decay");
    }
    if r2 < DRUMROOM_MIN_R2 {
        return Err("beats");
    }
    Ok(-6.0 * std::f64::consts::LN_10 / slope)
}

/// One run's record.
#[derive(Default)]
struct Run {
    e_total: Vec<f64>,
    e_head: Vec<f64>,
    e_acoustic: Vec<f64>,
    e_dissipated: Vec<f64>,
    radiated: Vec<f64>,
    injected: Vec<f64>,
    p_mic: Vec<f64>,
    channels: Vec<Vec<Vec<f64>>>,
    head_frames: Vec<Vec<f64>>,
    head_steps: Vec<usize>,
    slices: Vec<Vec<f64>>,
    slice_steps: Vec<usize>,
}

/// The run's cadences and what it captures.
struct Plan<'a> {
    n_steps: usize,
    mic: usize,
    planes: &'a [Plane],
    e_stride: usize,
    head_stride: usize,
    head_until: usize,
    slice_stride: usize,
    slice_until: usize,
}

/// Project `u` on every group's vectors, appending to its channels.
fn project(groups: &[ModeGroup], u: &[f64], channels: &mut [Vec<Vec<f64>>]) {
    for (g, ch) in groups.iter().zip(channels.iter_mut()) {
        for (v, c) in g.vectors.iter().zip(ch.iter_mut()) {
            c.push(dot(v, u));
        }
    }
}

fn empty_channels(groups: &[ModeGroup], cap: usize) -> Vec<Vec<Vec<f64>>> {
    groups
        .iter()
        .map(|g| g.vectors.iter().map(|_| Vec::with_capacity(cap)).collect())
        .collect()
}

/// The head in the room: both ledgers, the mic, frames, and the modal channels.
fn run(
    grid: &mut Grid,
    room: &mut AirBox,
    plan: &Plan,
    groups: &[ModeGroup],
) -> Result<Run, Refusal> {
    let mut r = Run {
        channels: empty_channels(groups, plan.n_steps + 1),
        ..Run::default()
    };
    let sample_energy = |g: &Grid, room: &AirBox, r: &mut Run| {
        r.e_head.push(g.seam.membrane.energy());
        r.e_acoustic.push(room.acoustic_energy());
        r.e_dissipated.push(room.dissipated);
        r.radiated.push(g.radiated_energy());
        r.injected.push(room.injected);
        r.e_total.push(g.energy() + room.energy());
    };
    sample_energy(grid, room, &mut r);
    project(groups, &grid.seam.membrane.u, &mut r.channels);
    r.p_mic.push(room.pressure[plan.mic]);
    r.head_frames.push(grid.seam.membrane.state());
    r.head_steps.push(0);
    r.slices.push(take_slices(room, plan.planes));
    r.slice_steps.push(0);
    for i in 1..=plan.n_steps {
        grid.step(room, None)
            .map_err(|e| Refusal::Internal(e.to_string()))?;
        room.step();
        r.p_mic.push(room.pressure[plan.mic]);
        project(groups, &grid.seam.membrane.u, &mut r.channels);
        if i % plan.e_stride == 0 {
            sample_energy(grid, room, &mut r);
        }
        if i <= plan.head_until && i % plan.head_stride == 0 {
            r.head_frames.push(grid.seam.membrane.state());
            r.head_steps.push(i);
        }
        if i <= plan.slice_until && i % plan.slice_stride == 0 {
            r.slices.push(take_slices(room, plan.planes));
            r.slice_steps.push(i);
        }
    }
    if !r.e_total.iter().all(|v| v.is_finite()) || !r.p_mic.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }
    Ok(r)
}

/// The bare twin: the same head, no room, the same projection.
fn run_twin(twin: &mut Membrane, n_steps: usize, groups: &[ModeGroup]) -> Vec<Vec<Vec<f64>>> {
    let mut ch = empty_channels(groups, n_steps + 1);
    project(groups, &twin.u, &mut ch);
    for _ in 0..n_steps {
        twin.step();
        project(groups, &twin.u, &mut ch);
    }
    ch
}

fn peak_json(pk: Peak) -> Value {
    match pk {
        Peak::Clear(f) => json!({"status": "clear", "f": num(round_nd(f, 3))}),
        Peak::Split(a, b) => json!({
            "status": "split", "f": num(round_nd(a, 3)), "f2": num(round_nd(b, 3)),
        }),
        Peak::Outside(f) => json!({"status": "outside window", "f": num(round_nd(f, 3))}),
        Peak::Unresolved => json!({"status": "unresolved"}),
    }
}

fn t60_json(t: Result<f64, &'static str>) -> Value {
    match t {
        Ok(v) => json!({"t60": num(round_nd(v, 4))}),
        Err(why) => json!({"t60": Value::Null, "why": why}),
    }
}

/// A ring time, or `null` for a lossless head, which has none to aim at.
fn target_t60_json(t60: f64) -> Value {
    if t60.is_finite() {
        num(round_nd(t60, 4))
    } else {
        Value::Null
    }
}

/// The scheme's exact frequency and ring time of the in-vacuo mode with eigenvalue `mu` of `−L`
/// (the ring time is infinite for a lossless head).
fn scheme_mode(p: &mem::Params, mu: f64) -> (f64, f64) {
    let (k, s) = (p.k, p.sigma * p.k);
    let (x, y) = (p.c * p.c * k * k * mu, 2.0 * p.sigma1 * k * mu);
    (
        mode_frequency(s, x, y, k).unwrap_or(f64::NAN),
        mode_t60(s, y, k),
    )
}

/// One measured unit of a mode group: the whole group when its members share one eigenvalue, or
/// one member when the grid has split them.
struct Unit {
    target: (f64, f64),
    twin_pk: Peak,
    room_pk: Peak,
    twin_ring: Result<f64, &'static str>,
    room_ring: Result<f64, &'static str>,
    shift: Option<f64>,
}

/// The windows a unit is searched in: `[lo_twin, hi)` in the bare twin, `[lo_room, hi)` in the
/// room.
struct Window {
    lo_twin: f64,
    lo_room: f64,
    hi: f64,
}

fn measure(twin: &[Vec<f64>], room: &[Vec<f64>], fs: f64, target: (f64, f64), w: &Window) -> Unit {
    let twin_pk = group_peak(twin, fs, w.lo_twin, w.hi);
    let room_pk = group_peak(room, fs, w.lo_room, w.hi);
    let shift = match (room_pk, twin_pk) {
        (Peak::Clear(a), Peak::Clear(b)) => Some(1200.0 * (a / b).log2()),
        _ => None,
    };
    let f_room = match room_pk {
        Peak::Clear(v) | Peak::Split(v, _) => v,
        Peak::Outside(_) | Peak::Unresolved => target.0,
    };
    Unit {
        target,
        twin_pk,
        room_pk,
        twin_ring: group_t60(twin, fs, target.0),
        room_ring: group_t60(room, fs, f_room),
        shift,
    }
}

/// A group's peak from its units: the one unit's, or both members' of a grid pair when each reads
/// clear, or else the first member's that does not.
fn units_peak(units: &[Unit], pk: impl Fn(&Unit) -> Peak) -> Value {
    if units.len() == 1 {
        return peak_json(pk(&units[0]));
    }
    let fs: Vec<f64> = units
        .iter()
        .filter_map(|u| match pk(u) {
            Peak::Clear(f) => Some(f),
            _ => None,
        })
        .collect();
    if fs.len() == units.len() {
        let mut v = json!({"status": "grid pair", "f": num(round_nd(fs[0], 3))});
        v["f2"] = num(round_nd(fs[fs.len() - 1], 3));
        return v;
    }
    let bad = units.iter().map(&pk).find(|p| !matches!(p, Peak::Clear(_)));
    peak_json(bad.unwrap_or(Peak::Unresolved))
}

/// A group's ring time from its units: the mean when every unit has one, else the first reason.
fn units_ring(units: &[Unit], ring: impl Fn(&Unit) -> Result<f64, &'static str>) -> Value {
    let rings: Vec<_> = units.iter().map(ring).collect();
    match rings.iter().find_map(|r| r.err()) {
        Some(why) => t60_json(Err(why)),
        None => {
            let ok: Vec<f64> = rings.into_iter().flatten().collect();
            t60_json(Ok(ok.iter().sum::<f64>() / ok.len() as f64))
        }
    }
}

/// The air read-out: per mode, the target (the scheme's exact value), the twin's and the room's
/// measured pitch and ring time, and the shift between the last two.
///
/// A group whose members share one eigenvalue is measured on its power-summed channels, which no
/// rotation of the pair's basis changes. A pair the grid has split (`grid_split_cents > 0`) is
/// measured member by member: the room is a cube centred on the head, so the two members, of
/// different symmetry, do not mix, and each member's channel holds one frequency. Its shift is the
/// members' mean.
///
/// The windows: below, the previous group's frequency — in the room, where it actually landed,
/// because the air can lower a mode past its in-vacuo neighbour's; above, the next group's
/// in-vacuo frequency, which the air only lowers. The fundamental's window reaches an octave down.
fn air_block(p: &mem::Params, groups: &[ModeGroup], main: &Run, twin: &[Vec<Vec<f64>>]) -> Value {
    let fs = p.fs;
    let members: Vec<Vec<(f64, f64)>> = groups
        .iter()
        .map(|g| g.mus.iter().map(|&mu| scheme_mode(p, mu)).collect())
        .collect();
    let mut prev_room: Option<f64> = None;
    let mut modes: Vec<Value> = Vec::new();
    for (j, (g, ms)) in groups.iter().zip(&members).enumerate() {
        let n = ms.len() as f64;
        let f = ms.iter().map(|t| t.0).sum::<f64>() / n;
        let t60 = ms.iter().map(|t| t.1).sum::<f64>() / n;
        let split = 1200.0 * (ms[ms.len() - 1].0 / ms[0].0).log2();
        let distinct = g.mus[g.mus.len() - 1] > g.mus[0] * (1.0 + 1e-9);
        let lo_twin = if j == 0 {
            0.5 * f
        } else {
            members[j - 1][members[j - 1].len() - 1].0
        };
        let w = Window {
            lo_twin,
            lo_room: if j == 0 {
                0.5 * f
            } else {
                prev_room.unwrap_or(lo_twin)
            },
            hi: members.get(j + 1).map_or(1.3 * f, |m| m[0].0),
        };
        let units: Vec<Unit> = if distinct {
            (0..ms.len())
                .map(|i| {
                    let (t, r) = (&twin[j][i..=i], &main.channels[j][i..=i]);
                    measure(t, r, fs, ms[i], &w)
                })
                .collect()
        } else {
            vec![measure(&twin[j], &main.channels[j], fs, (f, t60), &w)]
        };
        prev_room = units
            .iter()
            .filter_map(|u| match u.room_pk {
                Peak::Clear(v) | Peak::Split(v, _) => Some(v),
                _ => None,
            })
            .reduce(f64::max);
        let shift = units
            .iter()
            .map(|u| u.shift)
            .collect::<Option<Vec<f64>>>()
            .map(|s| s.iter().sum::<f64>() / s.len() as f64);
        let mut mode = json!({
            "multiplicity": int(g.vectors.len() as i64),
            "grid_split_cents": num(round_nd(split, 2)),
            "target": {"f": num(round_nd(f, 3)), "t60": target_t60_json(t60)},
            "twin": {"peak": units_peak(&units, |u| u.twin_pk),
                     "ring": units_ring(&units, |u| u.twin_ring)},
            "room": {"peak": units_peak(&units, |u| u.room_pk),
                     "ring": units_ring(&units, |u| u.room_ring)},
            "shift_cents": shift.map_or(Value::Null, |s| num(round_nd(s, 2))),
        });
        if units.len() > 1 {
            mode["members"] = units
                .iter()
                .map(|u| {
                    json!({
                        "target": {"f": num(round_nd(u.target.0, 3)),
                                   "t60": target_t60_json(u.target.1)},
                        "twin": {"peak": peak_json(u.twin_pk), "ring": t60_json(u.twin_ring)},
                        "room": {"peak": peak_json(u.room_pk), "ring": t60_json(u.room_ring)},
                        "shift_cents": u.shift.map_or(Value::Null, |s| num(round_nd(s, 2))),
                    })
                })
                .collect();
        }
        modes.push(mode);
    }
    let (e0, e1) = (main.e_head[0], main.e_head[main.e_head.len() - 1]);
    json!({
        "kind": "drumroom",
        "modes": modes,
        "head_energy_kept": num(round_nd(if e0 > 0.0 { e1 / e0 } else { 0.0 }, 6)),
    })
}

/// True when the request asks for the musical controls (`controls: "musical"`).
fn musical_requested(p: &Value) -> bool {
    is_musical(p)
}

/// The drum-in-a-room payload.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let audio_dur = fnum(p, "audio_duration", DRUMROOM_AUDIO_DEFAULT)?;
    let mic_frac = fnum(p, "mic_position", DRUMROOM_MIC_DEFAULT)?;
    let slice_frac = fnum(p, "slice_position", 0.5)?;
    let amplitude = fnum(p, "amplitude", 1e-3)?;
    let pluck_fx = fnum(p, "pluck_x", 0.4)?;
    let pluck_fy = fnum(p, "pluck_y", 0.55)?;
    let pluck_w = fnum(p, "pluck_width", 0.45)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;
    if !(0.0 < playback_speed && playback_speed <= SPEED_MAX) {
        return Err(Refusal::Param(format!(
            "playback_speed must be in (0, {}], got {}.",
            pf(SPEED_MAX),
            pf(playback_speed)
        )));
    }
    if !(0.0 < audio_dur && audio_dur <= DRUMROOM_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(DRUMROOM_AUDIO_MAX),
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
    for (name, v) in [("pluck_x", pluck_fx), ("pluck_y", pluck_fy)] {
        if !(0.0 < v && v < 1.0) {
            return Err(Refusal::Param(format!(
                "{name} must be in (0, 1), got {}.",
                pf(v)
            )));
        }
    }
    if !(0.0 < pluck_w && pluck_w <= 1.0) {
        return Err(Refusal::Param(format!(
            "pluck_width must be in (0, 1], got {}.",
            pf(pluck_w)
        )));
    }

    let mut sc = build(p)?;
    let fs = sc.fs;
    let n_steps = round_int(audio_dur * fs).max(16) as usize;
    let room_work = sc.nodes as f64 * n_steps as f64;
    if room_work > DRUMROOM_ROOM_WORK_MAX {
        return Err(Refusal::Param(format!(
            "the room's budget is exceeded ({} node-steps > {}). Shrink room_size, coarsen \
             air_h or shorten audio_duration.",
            sci(room_work, 2),
            sci(DRUMROOM_ROOM_WORK_MAX, 2)
        )));
    }
    let hp = sc.grid.seam.membrane.params().clone();
    let groups = mode_groups(&hp)?;
    let f1 = groups
        .first()
        .map(|g| scheme_mode(&hp, g.mus[0]).0)
        .filter(|f| f.is_finite())
        .unwrap_or(1.0);

    // -- the geometry: the head's centre and an off-axis mic
    let la = sc.room.p.l_actual;
    let z_head = if sc.suspended {
        (sc.room.p.n[2] / 2) as f64 * sc.h
    } else {
        0.0
    };
    let centre = [0.5 * la[0], 0.5 * la[1], z_head];
    let mic_want: [f64; 3] =
        std::array::from_fn(|d| centre[d] + mic_frac * (DRUMROOM_MIC_FAR[d] * la[d] - centre[d]));
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

    // TWO animation clocks: the head on its fundamental, the slices on acoustic transit.
    let ns = n_steps as i64;
    let mut head_stride = round_int((fs / f1.max(1.0)) / fpp as f64).max(1);
    let head_until = ns.min(head_stride.max(round_int(DRUMROOM_ANIM_WIN * fs)));
    if head_until / head_stride > MAX_FRAMES {
        head_stride = ((head_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let mut slice_stride = round_int(DRUMROOM_CELLS_PER_FRAME / sc.lam_air).max(1);
    let slice_until = ns.min(slice_stride.max(round_int(DRUMROOM_ROOM_ANIM_WIN * fs)));
    if slice_until / slice_stride > MAX_FRAMES {
        slice_stride = ((slice_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let e_stride = ((n_steps as f64 / DRUMROOM_TRACE_POINTS).ceil() as usize).max(1);

    let u0 = strike(&hp, sc.radius, pluck_fx, pluck_fy, pluck_w, amplitude)?;
    let v0 = vec![0.0; u0.len()];
    sc.grid.set_state(&u0, &v0);
    sc.twin.set_state(&u0, &v0);
    let plan = Plan {
        n_steps,
        mic: mic_flat,
        planes: &planes,
        e_stride,
        head_stride: head_stride as usize,
        head_until: head_until as usize,
        slice_stride: slice_stride as usize,
        slice_until: slice_until as usize,
    };
    let main = run(&mut sc.grid, &mut sc.room, &plan, &groups)?;
    let twin = run_twin(&mut sc.twin, n_steps, &groups);
    let air = air_block(&hp, &groups, &main, &twin);

    // -- the ledgers
    let total = &main.e_total;
    let e0 = total[0];
    let scale = e0.abs().max(1e-300);
    let resid: Vec<f64> = main
        .radiated
        .iter()
        .zip(&main.injected)
        .map(|(a, b)| (a - b).abs() / scale)
        .collect();
    let frac = |a: &[f64]| a.iter().map(|x| x / scale).collect::<Vec<f64>>();
    let e_steps: Vec<f64> = (0..total.len())
        .map(|j| (j * e_stride) as f64 / fs)
        .collect();
    let fmax = |a: &[f64]| a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ledger = json!({
        "kind": "drumroom",
        "time": finite_list(&e_steps, Some(6)),
        "e_head_frac": finite_list(&frac(&main.e_head), None),
        "e_acoustic_frac": finite_list(&frac(&main.e_acoustic), None),
        "e_dissipated_frac": finite_list(&frac(&main.e_dissipated), None),
        "radiated_frac": finite_list(&frac(&main.radiated), None),
        "total_frac": finite_list(&frac(total), None),
        "residual_max": num(fmax(&resid)),
        "e_stride": int(e_stride as i64),
    });
    let head_lossless = hp.sigma == 0.0 && hp.sigma1 == 0.0;
    let energy = energy_block(
        &e_steps,
        total,
        head_lossless,
        0.0,
        EnergyOpts {
            no_decay_oracle: true,
            ..EnergyOpts::default()
        },
    );
    let tuning = match &sc.musical {
        Some(m) => Some(tuning_block(&hp, m)?),
        None => None,
    };

    // -- frames
    let (ny, nx) = hp.shape();
    let (head_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&main.head_frames, hp.mask.flags(), ny, nx);
    let nf = main.head_frames.len();
    let head_amp = head_dec.iter().fold(0.0f64, |m, v| m.max(v.abs()));
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
        np_percentile(&live, DRUMROOM_SCALE_PCTL)
    };
    if reference <= 0.0 || reference.is_nan() {
        reference = if slice_amp > 0.0 { slice_amp } else { 1.0 };
    }
    let (audio48, peak) = resample_normalize(&main.p_mic, fs); // the audio IS the mic
    let head_times: Vec<f64> = main.head_steps.iter().map(|&i| i as f64 / fs).collect();
    let slice_times: Vec<f64> = main.slice_steps.iter().map(|&i| i as f64 / fs).collect();
    let planes_json: Vec<Value> = planes
        .iter()
        .map(|pl| Value::Object(pl.json.clone()))
        .collect();
    let mic_snapped = snap(&sc.room, mic_want);
    let rl = sc.room.p.l_actual;
    let tier = if sc.suspended { "suspended" } else { "baffled" };
    let ext = 2.0 * sc.radius;
    let mut payload = json!({
        "model": "drumroom",
        "tier": tier,
        "walls": sc.walls,
        "controls": if musical_requested(p) { "musical" } else { "physical" },
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(hp.lam, 6)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(ext, 6)), "extent_y": num(round_nd(ext, 6)),
            "domain": "circle",
        },
        "frames": {
            "b64": b64f32(&head_dec),
            "n_frames": int(nf as i64), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "width": int(nx_dec as i64), "dims": int(2),
        },
        "mask": {"b64": b64u8(&mask_dec), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64)},
        "frame_times": finite_list(&head_times, Some(6)),
        "anim_dt": num(head_stride as f64 / fs),
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
                      "pctl": num(DRUMROOM_SCALE_PCTL)},
        },
        "playback_speed": num(playback_speed),
        "field_amp": num(head_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak),
            "n": int(audio48.len() as i64),
        },
        "energy": energy,
        "meta": {
            "c": num(round_nd(hp.c, 4)),
            "f1": num(round_nd(f1, 3)),
            "num_steps": int(n_steps as i64),
            "n_frames": int(nf as i64),
            "head": {
                "radius": num(sc.radius), "N": int(hp.n as i64), "n_live": int(hp.n_live() as i64),
                "T": num(round_nd(hp.t, 4)), "rho": num(round_nd(hp.rho, 6)),
                "sigma0": num(hp.sigma), "sigma1": num(hp.sigma1),
                "c_over_c0": num(round_nd(hp.c / C0_AIR, 4)),
            },
            "room": {
                "L": rl.iter().map(|&v| num(round_nd(v, 4))).collect::<Vec<_>>(),
                "L_requested": num(round_nd(sc.size, 4)),
                "N": sc.room.p.n.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
                "h": num(sc.h),
                "nodes": int(sc.nodes as i64),
                "cfl": num(DRUMROOM_CFL),
                "lam_air": num(round_nd(sc.lam_air, 6)),
                "walls": sc.walls,
                "c0": num(round_nd(C0_AIR, 3)),
                "mic_at": mic_snapped.iter().map(|&v| num(round_nd(v, 4))).collect::<Vec<_>>(),
                "mic_cell": mic_index.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
                "room_work": num(room_work),
            },
            "ledger": ledger,
            "air": air,
        },
    });
    if let Some(t) = tuning {
        payload["tuning"] = t;
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `N` the page allows groups the head's lowest modes as the circle's: (0,1), (1,1)
    /// pair, (2,1) pair, (0,2), (3,1) pair, (1,2) pair — the grid's split of the (2,1) pair
    /// inside one group, the (0,2) mode outside it.
    #[test]
    fn every_allowed_grid_groups_the_modes_as_the_circle_does() {
        for n in DRUMROOM_N_MIN..=DRUMROOM_N_MAX {
            let p = mem::Params::new(
                Some(Domain::Circle),
                3000.0,
                0.26,
                40_000.0,
                n,
                None,
                None,
                Some(0.12),
                0.0,
            )
            .unwrap();
            let groups = mode_groups(&p).unwrap();
            let mult: Vec<usize> = groups.iter().map(|g| g.vectors.len()).collect();
            assert_eq!(mult, [1, 2, 2, 1, 2, 2], "N = {n}");
        }
    }

    /// A channel whose strongest peak lies below the window reads "outside", with that peak's
    /// frequency, even when a weaker one sits inside — not the window's best.
    #[test]
    fn a_peak_below_the_window_is_reported_outside_not_replaced() {
        let fs = 40_000.0;
        let tone =
            |f: f64, a: f64, i: usize| a * (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin();
        let ch: Vec<f64> = (0..14_000)
            .map(|i| tone(700.0, 1.0, i) + tone(760.0, 0.05, i))
            .collect();
        match group_peak(std::slice::from_ref(&ch), fs, 717.0, 888.0) {
            Peak::Outside(f) => assert!((f - 700.0).abs() < 0.5, "{f}"),
            other => panic!("{other:?}"),
        }
        match group_peak(&[ch], fs, 650.0, 888.0) {
            Peak::Clear(f) => assert!((f - 700.0).abs() < 0.5, "{f}"),
            other => panic!("{other:?}"),
        }
    }
}
