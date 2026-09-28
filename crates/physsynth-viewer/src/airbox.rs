//! String -> body -> a 3-D room — `serialize.py`'s airbox builder, the first `dims: 3` payload.
//!
//! Three things unlike every other scene, all measured by the reference: the cost is `h^-4` (the
//! 3-D CFL runs the wrong way, so a coarser grid forces a HIGHER sample rate); the ROOM sets the
//! sample rate for the whole scene, so the string's lambda becomes a derived read-out; and
//! `dims: 3` is a SLICE SET — three decimated orthogonal planes per frame — never a volume.
//!
//! The claim is an INDEX, never a magnitude: on the 7-point stencil a mic's first nonzero sample
//! sits exactly on the Manhattan distance in cells from the port (the lattice light cone). The
//! money test beside it is the cross-ledger residual `|radiated - injected|`, because the coupling
//! term cancels out of the scene total identically and the total cannot see a wrong coupling.

use physsynth_core::airbox::{self as ab, AirBox, Wall};
use physsynth_core::airbox_wrap::RoomLoadedBody;
use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::StringBodyBridge;
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::fmt::py_float;
use physsynth_core::string_ideal::{self as ideal, Boundary, IdealString};
use serde_json::{json, Map, Value};

use crate::body::{BODY_AMP_DEFAULT, BODY_K_DEFAULT, BODY_SIGMA_BODY_DEFAULT, BODY_SIGMA_BODY_MAX};
use crate::energy::{energy_block, EnergyOpts};
use crate::membrane::DISPLAY_MAX;
use crate::py::{
    b64f32, commas, finite_list, fmt_g, int, num, py_int, py_repr, py_str, repr_str, round_int,
    round_nd, sci,
};
use crate::radbody::{BODY_BODY_FREQS, BODY_BODY_MASS};
use crate::reed::{C0_AIR, RHO0_AIR};
use crate::string::{construction, MAX_FRAMES, N_MIN, SPEED_MAX};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// The shipped room's proportions (m at `room_size = 1`).
pub const AIRBOX_ASPECT: [f64; 3] = [1.2, 0.9, 0.8];
/// Room scale default.
pub const AIRBOX_SIZE_DEFAULT: f64 = 1.0;
/// Room scale range.
pub const AIRBOX_SIZE_MIN: f64 = 0.5;
/// See [`AIRBOX_SIZE_MIN`].
pub const AIRBOX_SIZE_MAX: f64 = 2.5;
/// Air grid spacing default (m); the snap is the resolution.
pub const AIRBOX_H_DEFAULT: f64 = 0.03;
/// Air grid spacing range (m).
pub const AIRBOX_H_MIN: f64 = 0.02;
/// See [`AIRBOX_H_MIN`].
pub const AIRBOX_H_MAX: f64 = 0.10;
/// Fraction of the 3-D CFL ceiling; not the ceiling itself.
pub const AIRBOX_CFL_DEFAULT: f64 = 0.9;
/// CFL fraction range.
pub const AIRBOX_CFL_MIN: f64 = 0.3;
/// See [`AIRBOX_CFL_MIN`].
pub const AIRBOX_CFL_MAX: f64 = 0.99;
/// Normalized wall impedance default for `absorbing`.
pub const AIRBOX_ZETA_DEFAULT: f64 = 3.0;
/// Normalized wall impedance ceiling.
pub const AIRBOX_ZETA_MAX: f64 = 50.0;
/// The wall tokens.
pub const AIRBOX_WALLS: [&str; 3] = ["rigid", "absorbing", "open"];
/// String grid ceiling.
pub const AIRBOX_N_MAX: i64 = 140;
/// Audio default (s).
pub const AIRBOX_AUDIO_DEFAULT: f64 = 0.6;
/// Longest audio (s).
pub const AIRBOX_AUDIO_MAX: f64 = 2.0;
/// Room node ceiling.
pub const AIRBOX_NODE_MAX: usize = 400_000;
/// `nodes x steps`.
pub const AIRBOX_WORK_MAX: f64 = 1.2e9;
/// Seconds of animation (the wavefront crosses the room in ~3.5 ms).
pub const AIRBOX_ANIM_WIN: f64 = 0.014;
/// The animation clock is acoustic transit: one frame per cell of travel.
pub const AIRBOX_CELLS_PER_FRAME: f64 = 1.0;
/// Energy-channel samples.
pub const AIRBOX_TRACE_POINTS: f64 = 600.0;
/// The body's port, as a fraction of the room.
pub const AIRBOX_PORT_FRAC: [f64; 3] = [0.25, 0.30, 0.35];
/// The far end of the mic's travel — deliberately off-axis.
pub const AIRBOX_MIC_FAR: [f64; 3] = [0.92, 0.85, 0.90];
/// Mic position default.
pub const AIRBOX_MIC_DEFAULT: f64 = 0.8;
/// Mic position range.
pub const AIRBOX_MIC_MIN: f64 = 0.15;
/// See [`AIRBOX_MIC_MIN`].
pub const AIRBOX_MIC_MAX: f64 = 1.0;
/// Amplitude thresholds shown beside the cone.
pub const AIRBOX_ARRIVAL_FRACS: [f64; 2] = [1e-3, 2e-2];
/// Percentile of the live field that sets the asinh reference.
pub const AIRBOX_SCALE_PCTL: f64 = 55.0;

fn pf(x: f64) -> String {
    py_float(x)
}

/// `np.percentile(a, q)`, NumPy's default `linear` method, transcribed with its own arithmetic:
/// the virtual index `(n - 1) q` (the method's own override — NOT the general `n q + (alpha + q
/// (1 - alpha - beta)) - 1`, which differs from it in the last bit, measured) and the two-sided
/// lerp that switches at `t = 0.5`.
pub fn np_percentile(a: &[f64], q_percent: f64) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let q = q_percent / 100.0;
    let virt = (n as f64 - 1.0) * q;
    let prev = virt.floor();
    let lo = (prev.max(0.0) as usize).min(n - 1);
    let hi = ((prev + 1.0).max(0.0) as usize).min(n - 1);
    let t = virt - prev;
    let (a0, b0) = (v[lo], v[hi]);
    let diff = b0 - a0;
    if t >= 0.5 {
        b0 - diff * (1.0 - t)
    } else {
        a0 + diff * t
    }
}

/// What the scene reports beside the bridge and the room.
struct Info {
    c: f64,
    l: f64,
    fs: f64,
    lam_string: f64,
    lam_air: f64,
    sigma_body: f64,
    h: f64,
    cfl: f64,
    nodes: usize,
    walls_token: String,
    room_l: [f64; 3],
    room_l_requested: [f64; 3],
    room_n: [usize; 3],
    walls_label: String,
    zeta: Option<f64>,
    port_at: [f64; 3],
}

type Bridge = StringBodyBridge<RoomLoadedBody>;

/// `_airbox_walls`: the wall the core wants, the label the read-out prints, and zeta.
fn walls_of(p: &Value) -> Result<(Wall, String, Option<f64>), Refusal> {
    let wv = p.get("walls").cloned().unwrap_or(json!("rigid"));
    let walls = py_str(&wv);
    if !AIRBOX_WALLS.contains(&walls.as_str()) {
        return Err(Refusal::Param(format!(
            "walls must be one of ('rigid', 'absorbing', 'open'), got {}.",
            repr_str(&walls)
        )));
    }
    let zeta = fnum(p, "wall_zeta", AIRBOX_ZETA_DEFAULT)?;
    match walls.as_str() {
        "rigid" => Ok((Wall::Rigid, walls, None)),
        "open" => Ok((Wall::Open, walls, None)),
        _ => {
            if !(0.0 < zeta && zeta <= AIRBOX_ZETA_MAX) {
                return Err(Refusal::Param(format!(
                    "wall_zeta must be in (0, {}], got {}.",
                    pf(AIRBOX_ZETA_MAX),
                    pf(zeta)
                )));
            }
            let z = ab::impedance_from_zeta(zeta, RHO0_AIR, C0_AIR);
            Ok((
                Wall::Impedance(z),
                format!("absorbing (zeta = {})", fmt_g(zeta)),
                Some(zeta),
            ))
        }
    }
}

/// `_build_airbox_scene`: string -> bridge -> room-loaded body -> room, and the scalars.
fn build(p: &Value) -> Result<(Bridge, AirBox, Info), Refusal> {
    let size = fnum(p, "room_size", AIRBOX_SIZE_DEFAULT)?;
    let h = fnum(p, "air_h", AIRBOX_H_DEFAULT)?;
    let cfl = fnum(p, "air_cfl", AIRBOX_CFL_DEFAULT)?;
    let l = fnum(p, "L", 1.0)?;
    let t = fnum(p, "T", 200.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let k = fnum(p, "bridge_stiffness", BODY_K_DEFAULT)?;
    let sigma_body = fnum(p, "sigma_body", BODY_SIGMA_BODY_DEFAULT)?;
    let nv = p.get("N").cloned().unwrap_or(json!(100));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;

    if !(AIRBOX_SIZE_MIN..=AIRBOX_SIZE_MAX).contains(&size) {
        return Err(Refusal::Param(format!(
            "room_size must be in [{}, {}], got {}.",
            pf(AIRBOX_SIZE_MIN),
            pf(AIRBOX_SIZE_MAX),
            pf(size)
        )));
    }
    if !(AIRBOX_H_MIN..=AIRBOX_H_MAX).contains(&h) {
        return Err(Refusal::Param(format!(
            "air_h must be in [{}, {}] m, got {}.",
            pf(AIRBOX_H_MIN),
            pf(AIRBOX_H_MAX),
            pf(h)
        )));
    }
    if !(AIRBOX_CFL_MIN..=AIRBOX_CFL_MAX).contains(&cfl) {
        return Err(Refusal::Param(format!(
            "air_cfl must be in [{}, {}], got {}: it is the fraction of the 3-D CFL ceiling, and \
             1.0 is refused because lambda = 1/sqrt(3) is where the corner mode goes DEFECTIVE — \
             a flat energy there is not a stability certificate.",
            pf(AIRBOX_CFL_MIN),
            pf(AIRBOX_CFL_MAX),
            pf(cfl)
        )));
    }
    if !(N_MIN..=AIRBOX_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {AIRBOX_N_MAX}] for the room, got {n}."
        )));
    }
    let mut mn = l;
    for v in [t, rho] {
        if v < mn {
            mn = v;
        }
    }
    if mn <= 0.0 {
        return Err(Refusal::Param("L, T, rho must all be positive.".into()));
    }
    if k < 0.0 {
        return Err(Refusal::Param(format!(
            "bridge_stiffness must be >= 0, got {}.",
            pf(k)
        )));
    }
    if !(0.0..=BODY_SIGMA_BODY_MAX).contains(&sigma_body) {
        return Err(Refusal::Param(format!(
            "sigma_body must be in [0, {}], got {}.",
            pf(BODY_SIGMA_BODY_MAX),
            pf(sigma_body)
        )));
    }

    let (wall, walls_label, zeta) = walls_of(p)?;
    let room_l = [
        size * AIRBOX_ASPECT[0],
        size * AIRBOX_ASPECT[1],
        size * AIRBOX_ASPECT[2],
    ];
    let lam_air = cfl / 3.0f64.sqrt();
    let fs = C0_AIR / (lam_air * h); // the room pins the scene's sample rate
    let c = (t / rho).sqrt();
    let lam_string = c * n as f64 / (l * fs); // DERIVED, not a slider
    if !(0.0 < lam_string && lam_string < 1.0) {
        return Err(Refusal::Param(format!(
            "the string's lambda comes out {lam_string:.4}, and it must be in (0, 1). The ROOM \
             sets the sample rate here (fs = c0/(lambda_air h) = {} Hz), so the string's lambda \
             is derived, not dialled: lower N, refine air_h, or raise air_cfl.",
            commas(format!("{fs:.0}").parse::<i64>().unwrap_or(0))
        )));
    }
    let rp =
        ab::Params::new(room_l, fs, h, [wall; 6], None, RHO0_AIR, C0_AIR).map_err(construction)?;
    let mut room = AirBox::new(rp);
    let rn = room.p.n;
    let nodes = (rn[0] + 1) * (rn[1] + 1) * (rn[2] + 1);
    if nodes > AIRBOX_NODE_MAX {
        return Err(Refusal::Param(format!(
            "the room is {} nodes (> {}). Coarsen air_h or shrink room_size — in 3-D the node \
             count grows as h^-3.",
            commas(nodes as i64),
            commas(AIRBOX_NODE_MAX as i64)
        )));
    }

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
    let m = BODY_BODY_FREQS.len();
    let bp = body::Params::new(
        BODY_BODY_FREQS.to_vec(),
        fs,
        vec![sigma_body; m],
        vec![BODY_BODY_MASS; m],
        vec![1.0; m],
        None,
    )
    .map_err(construction)?;
    let la = room.p.l_actual;
    let at = [
        AIRBOX_PORT_FRAC[0] * la[0],
        AIRBOX_PORT_FRAC[1] * la[1],
        AIRBOX_PORT_FRAC[2] * la[2],
    ];
    let loaded =
        RoomLoadedBody::new(ModalBody::new(bp), &mut room, at, None).map_err(construction)?;
    // The exact coupled guard fires here, as a construction error.
    let bridge = StringBodyBridge::new(string, loaded, k).map_err(construction)?;
    let snapped_at = snap(&room, at);
    let info = Info {
        c,
        l,
        fs,
        lam_string,
        lam_air,
        sigma_body,
        h,
        cfl,
        nodes,
        walls_token: py_str(&p.get("walls").cloned().unwrap_or(json!("rigid"))),
        room_l: la.map(|v| round_nd(v, 4)),
        room_l_requested: room_l.map(|v| round_nd(v, 4)),
        room_n: rn,
        walls_label,
        zeta,
        port_at: snapped_at.map(|v| round_nd(v, 4)),
    };
    Ok((bridge, room, info))
}

/// `room.node_index(point)` — the points this scene asks about are inside the room by design.
pub(crate) fn index(room: &AirBox, point: [f64; 3]) -> [usize; 3] {
    ab::node_index(point, room.p.h, room.p.n).expect("the scene's points lie inside the room")
}

/// `room.snapped(point)`: the nearest node's coordinates.
pub(crate) fn snap(room: &AirBox, point: [f64; 3]) -> [f64; 3] {
    index(room, point).map(|i| i as f64 * room.p.h)
}

/// One slice plane's geometry.
pub(crate) struct Plane {
    pub(crate) json: Map<String, Value>,
    axis: usize,
    at: usize,
    stride: usize,
}

/// `_airbox_slice_planes`: the three named orthogonal planes and their decimation strides.
pub(crate) fn slice_planes(room: &AirBox, frac: f64) -> Vec<Plane> {
    let n = room.p.n.map(|v| v + 1);
    let idx: Vec<usize> = n
        .iter()
        .map(|&m| (round_int(frac * (m as f64 - 1.0)).max(0) as usize).min(m - 1))
        .collect();
    let hx = room.p.h;
    [
        ("xy", "x", "y", n[0], n[1], 2usize),
        ("xz", "x", "z", n[0], n[2], 1),
        ("yz", "y", "z", n[1], n[2], 0),
    ]
    .iter()
    .map(|&(name, u, v, nu_full, nv_full, axis)| {
        let at = idx[axis];
        let stride = nu_full.max(nv_full).div_ceil(DISPLAY_MAX).max(1);
        let mut m = Map::new();
        m.insert("name".into(), json!(name));
        m.insert("u".into(), json!(u));
        m.insert("v".into(), json!(v));
        m.insert("nu_full".into(), int(nu_full as i64));
        m.insert("nv_full".into(), int(nv_full as i64));
        m.insert("axis".into(), int(axis as i64));
        m.insert("at".into(), int(at as i64));
        m.insert("stride".into(), int(stride as i64));
        m.insert("nu".into(), int(nu_full.div_ceil(stride) as i64));
        m.insert("nv".into(), int(nv_full.div_ceil(stride) as i64));
        m.insert("at_m".into(), num(round_nd(at as f64 * hx, 4)));
        m.insert(
            "extent".into(),
            json!([
                num(round_nd((nu_full as f64 - 1.0) * hx, 4)),
                num(round_nd((nv_full as f64 - 1.0) * hx, 4))
            ]),
        );
        Plane {
            json: m,
            axis,
            at,
            stride,
        }
    })
    .collect()
}

/// `_airbox_take_slices`: one frame, the three decimated planes head to tail.
pub(crate) fn take_slices(room: &AirBox, planes: &[Plane]) -> Vec<f64> {
    let shape = room.p.p_shape();
    let mut out = Vec::new();
    for pl in planes {
        let s = pl.stride;
        match pl.axis {
            2 => {
                for i in (0..shape[0]).step_by(s) {
                    for j in (0..shape[1]).step_by(s) {
                        out.push(room.pressure[ab::flat(shape, i, j, pl.at)]);
                    }
                }
            }
            1 => {
                for i in (0..shape[0]).step_by(s) {
                    for k in (0..shape[2]).step_by(s) {
                        out.push(room.pressure[ab::flat(shape, i, pl.at, k)]);
                    }
                }
            }
            _ => {
                for j in (0..shape[1]).step_by(s) {
                    for k in (0..shape[2]).step_by(s) {
                        out.push(room.pressure[ab::flat(shape, pl.at, j, k)]);
                    }
                }
            }
        }
    }
    out
}

/// `_build_payload_airbox`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let pluck_frac = fnum(p, "pluck_position", 0.3)?;
    let amplitude = fnum(p, "amplitude", BODY_AMP_DEFAULT)?;
    let audio_dur = fnum(p, "audio_duration", AIRBOX_AUDIO_DEFAULT)?;
    let mic_frac = fnum(p, "mic_position", AIRBOX_MIC_DEFAULT)?;
    let slice_frac = fnum(p, "slice_position", 0.5)?;
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
    if !(0.0 < audio_dur && audio_dur <= AIRBOX_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(AIRBOX_AUDIO_MAX),
            pf(audio_dur)
        )));
    }
    if !(AIRBOX_MIC_MIN..=AIRBOX_MIC_MAX).contains(&mic_frac) {
        return Err(Refusal::Param(format!(
            "mic_position must be in [{}, {}], got {}.",
            pf(AIRBOX_MIC_MIN),
            pf(AIRBOX_MIC_MAX),
            pf(mic_frac)
        )));
    }
    if !(0.0..=1.0).contains(&slice_frac) {
        return Err(Refusal::Param(format!(
            "slice_position must be in [0, 1], got {}.",
            pf(slice_frac)
        )));
    }

    let (mut bridge, mut room, info) = build(p)?;
    let (fs, c, l) = (info.fs, info.c, info.l);
    let f1_base = c / (2.0 * l);
    let n_steps = round_int(audio_dur * fs).max(1);
    let work = info.nodes as i64 * n_steps;
    if work as f64 > AIRBOX_WORK_MAX {
        return Err(Refusal::Param(format!(
            "work budget exceeded ({} node-steps > {}). In 3-D the CFL runs the wrong way — a \
             coarser grid forces a HIGHER sample rate — so cost grows as h^-4: coarsen air_h only \
             together with a shorter audio_duration, or shrink room_size.",
            sci(work as f64, 2),
            sci(AIRBOX_WORK_MAX, 2)
        )));
    }

    // -- the geometry the claim is measured against
    let port_at = info.port_at;
    let la = room.p.l_actual;
    let mic_want: [f64; 3] =
        std::array::from_fn(|d| port_at[d] + mic_frac * (AIRBOX_MIC_FAR[d] * la[d] - port_at[d]));
    let mic_index = index(&room, mic_want);
    let port_index = index(&room, port_at);
    let manhattan: i64 = (0..3)
        .map(|d| (mic_index[d] as i64 - port_index[d] as i64).abs())
        .sum();
    let mic_snapped = snap(&room, mic_want);
    let port_snapped = snap(&room, port_at);
    let diff: [f64; 3] = std::array::from_fn(|d| mic_snapped[d] - port_snapped[d]);
    let eucl_m = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    let manh_m = diff[0].abs() + diff[1].abs() + diff[2].abs();

    let planes = slice_planes(&room, slice_frac);
    let per_frame: usize = planes
        .iter()
        .map(|pl| {
            pl.json["nu"].as_u64().unwrap() as usize * pl.json["nv"].as_u64().unwrap() as usize
        })
        .sum();
    // The animation clock is the ROOM's: one frame per cell of wavefront travel.
    let mut anim_stride = round_int(AIRBOX_CELLS_PER_FRAME / info.lam_air).max(1);
    let frame_until = n_steps.min(anim_stride.max(round_int(AIRBOX_ANIM_WIN * fs)));
    if frame_until / anim_stride > MAX_FRAMES {
        anim_stride = ((frame_until as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }

    let x = bridge.string().params().grid();
    let u0 = triangular_pluck(&x, l, pluck_frac * l, amplitude).map_err(construction)?;
    bridge.string_mut().set_state(&u0, &vec![0.0; u0.len()]);
    let e_stride = ((n_steps as f64 / AIRBOX_TRACE_POINTS).ceil() as usize).max(1);

    // -- the run: five booked channels and two ledgers every e_stride, mic and port every step
    let n = n_steps as usize;
    let m = n / e_stride + 1;
    let mut ch: [Vec<f64>; 8] = Default::default(); // E, string, body, conn, ac, dis, rad, inj
    let (mut p_mic, mut q_port) = (Vec::with_capacity(n + 1), Vec::with_capacity(n + 1));
    let (mut slices, mut frame_steps) = (Vec::new(), Vec::new());
    let k_spring = bridge.stiffness();
    let mic_flat = ab::flat(room.p.p_shape(), mic_index[0], mic_index[1], mic_index[2]);
    let sample_energy = |b: &Bridge, r: &AirBox, ch: &mut [Vec<f64>; 8]| {
        ch[1].push(b.string().energy());
        ch[2].push(b.body().body.energy());
        ch[3].push(0.5 * k_spring * b.stretch(false) * b.stretch(true));
        ch[4].push(r.acoustic_energy());
        ch[5].push(r.dissipated);
        ch[6].push(b.body().radiated_energy());
        ch[7].push(r.injected);
        ch[0].push(b.energy() + r.energy());
    };
    sample_energy(&bridge, &room, &mut ch);
    p_mic.push(room.pressure[mic_flat]);
    q_port.push(bridge.body().volume_velocity());
    if frame_until >= 1 {
        slices.push(take_slices(&room, &planes));
        frame_steps.push(0usize);
    }
    for i in 1..=n {
        bridge
            .step(&mut room)
            .map_err(|e| Refusal::Internal(e.to_string()))?;
        room.step();
        p_mic.push(room.pressure[mic_flat]);
        q_port.push(bridge.body().volume_velocity());
        if i % e_stride == 0 && i / e_stride < m {
            sample_energy(&bridge, &room, &mut ch);
        }
        if i as i64 <= frame_until && i as i64 % anim_stride == 0 {
            slices.push(take_slices(&room, &planes));
            frame_steps.push(i);
        }
    }
    let [total, e_string, e_body, e_conn, e_ac, e_dis, radiated, injected] = ch;
    if !total.iter().all(|v| v.is_finite()) || !p_mic.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite energy (instability) — adjust parameters.".into(),
        ));
    }

    // -- the money panel: five booked channels and the cross-ledger residual guarding them
    let e0 = total[0];
    let scale = e0.abs().max(1e-300);
    let resid: Vec<f64> = radiated
        .iter()
        .zip(&injected)
        .map(|(r, j)| (r - j).abs() / scale)
        .collect();
    let frac = |a: &[f64]| {
        a.iter()
            .zip(&total)
            .map(|(x, t)| x / t)
            .collect::<Vec<f64>>()
    };
    let e_steps: Vec<f64> = (0..total.len())
        .map(|j| (j * e_stride) as f64 / fs)
        .collect();
    let fold_max = |a: &[f64]| a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let fold_min = |a: &[f64]| a.iter().copied().fold(f64::INFINITY, f64::min);
    let last = total.len() - 1;
    let ledger = json!({
        "kind": "airbox",
        "time": finite_list(&e_steps, Some(6)),
        "e_string_frac": finite_list(&frac(&e_string), None),
        "e_body_frac": finite_list(&frac(&e_body), None),
        "e_conn_frac": finite_list(&frac(&e_conn), None),
        "e_acoustic_frac": finite_list(&frac(&e_ac), None),
        "e_dissipated_frac": finite_list(&frac(&e_dis), None),
        "total_frac": finite_list(&frac(&total), None),
        "residual": finite_list(&resid, Some(3)),
        "residual_max": num(fold_max(&resid)),
        "e_stride": int(e_stride as i64),
        "n_samples": int(total.len() as i64),
        "acoustic_frac_peak": num(round_nd(fold_max(&frac(&e_ac)), 6)),
        "dissipated_frac_end": num(round_nd(e_dis[last] / total[last], 6)),
        "radiated_frac_end": num(round_nd(radiated[last] / total[last], 6)),
        "total_drift": num((fold_max(&total) - fold_min(&total)) / scale),
        "walls": info.walls_label,
    });

    // -- the claim: the lattice light cone, an exact integer of cells
    let first = |a: &[f64]| a.iter().position(|&v| v != 0.0).map_or(-1, |i| i as i64);
    let (first_q, first_p) = (first(&q_port), first(&p_mic));
    let measured = if first_q >= 0 && first_p >= 0 {
        first_p - first_q
    } else {
        -1
    };
    let peak = p_mic.iter().fold(0.0f64, |mx, v| mx.max(v.abs()));
    let thresholds: Vec<Value> = AIRBOX_ARRIVAL_FRACS
        .iter()
        .map(|&fr| {
            let hit = if peak > 0.0 {
                p_mic.iter().position(|v| v.abs() > fr * peak)
            } else {
                None
            };
            let step = match hit {
                Some(h) if first_q >= 0 => h as i64 - first_q,
                _ => -1,
            };
            json!({"frac": num(fr), "steps": int(step)})
        })
        .collect();
    let span = (manhattan * 3)
        .max(measured + 20)
        .max(60)
        .min(p_mic.len() as i64 - 1);
    let lo = first_q.max(0) as usize;
    let hi = (lo + span as usize + 1).min(p_mic.len());
    let trace = &p_mic[lo..hi];
    let tmax = trace.iter().fold(0.0f64, |mx, v| mx.max(v.abs()));
    let trace_n: Vec<f64> = if tmax > 0.0 {
        trace.iter().map(|v| v / tmax).collect()
    } else {
        trace.to_vec()
    };
    let arrival = json!({
        "kind": "airbox",
        "manhattan_cells": int(manhattan),
        "measured_cells": int(measured),
        "match": measured == manhattan,
        "first_injection_step": int(first_q),
        "first_mic_step": int(first_p),
        "euclid_m": num(round_nd(eucl_m, 4)),
        "euclid_steps": num(round_nd(eucl_m / C0_AIR * fs, 2)),
        "manhattan_m": num(round_nd(manh_m, 4)),
        "manhattan_steps_physical": num(round_nd(manh_m / C0_AIR * fs, 2)),
        "thresholds": thresholds,
        "trace": finite_list(&trace_n, Some(6)),
        "trace_peak": num(tmax),
        "span": int(span),
        "mic_cell": mic_index.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
        "port_cell": port_index.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
        "mic_at": mic_snapped.iter().map(|&v| num(round_nd(v, 4))).collect::<Vec<_>>(),
        "lam_air": num(round_nd(info.lam_air, 6)),
    });

    // -- the slice set: dims 3, with the compressed signed map's reference a MEASUREMENT
    let flat_frames: Vec<f64> = slices.iter().flatten().copied().collect();
    let live: Vec<f64> = flat_frames
        .iter()
        .filter(|&&v| v != 0.0)
        .map(|v| v.abs())
        .collect();
    let field_amp = flat_frames.iter().fold(0.0f64, |mx, v| mx.max(v.abs()));
    let mut reference = if live.is_empty() {
        0.0
    } else {
        np_percentile(&live, AIRBOX_SCALE_PCTL)
    };
    if reference <= 0.0 || reference.is_nan() {
        reference = if field_amp > 0.0 { field_amp } else { 1.0 };
    }
    let (audio48, peak_a) = resample_normalize(&p_mic, fs); // the audio IS the mic
    let times: Vec<f64> = frame_steps.iter().map(|&i| i as f64 / fs).collect();
    let planes_json: Vec<Value> = planes
        .into_iter()
        .map(|pl| Value::Object(pl.json))
        .collect();
    let lossless = info.sigma_body == 0.0 && info.walls_token != "absorbing";
    Ok(json!({
        "model": "airbox",
        "fs_sim": num(round_nd(fs, 3)),
        "lambda": num(round_nd(info.lam_string, 6)),
        "grid": {"x": finite_list(&x, Some(6))},
        "frames": {
            "b64": b64f32(&flat_frames),
            "n_frames": int(slices.len() as i64),
            "width": int(per_frame as i64),
            "dims": int(3),
            "kind": "slices",
            "planes": planes_json,
            "stride_floats": int(per_frame as i64),
            "scale": {"map": "asinh", "ref": num(reference), "amp": num(field_amp),
                      "pctl": num(AIRBOX_SCALE_PCTL)},
        },
        "frame_times": finite_list(&times, Some(6)),
        "anim_dt": num(anim_stride as f64 / fs),
        "playback_speed": num(playback_speed),
        "field_amp": num(field_amp),
        "audio": {
            "b64": b64f32(&audio48),
            "fs": num(AUDIO_FS),
            "peak": num(peak_a),
            "n": int(audio48.len() as i64),
        },
        "energy": energy_block(&e_steps, &total, lossless, 0.0, EnergyOpts {
            no_decay_oracle: true,
            ..EnergyOpts::default()
        }),
        "meta": {
            "c": num(round_nd(c, 3)),
            "f1": num(round_nd(f1_base, 3)),
            "num_steps": int(n_steps),
            "n_frames": int(slices.len() as i64),
            "room": {
                "L": info.room_l.iter().map(|&v| num(v)).collect::<Vec<_>>(),
                "L_requested": info.room_l_requested.iter().map(|&v| num(v)).collect::<Vec<_>>(),
                "N": info.room_n.iter().map(|&v| int(v as i64)).collect::<Vec<_>>(),
                "h": num(info.h),
                "nodes": int(info.nodes as i64),
                "cfl": num(info.cfl),
                "lam_air": num(round_nd(info.lam_air, 6)),
                "lam_string": num(round_nd(info.lam_string, 6)),
                "walls": info.walls_label,
                "zeta": info.zeta.map_or(Value::Null, num),
                "port_at": info.port_at.iter().map(|&v| num(v)).collect::<Vec<_>>(),
                "c0": num(round_nd(C0_AIR, 3)),
                "work": num(work as f64),
            },
            "ledger": ledger,
            "arrival": arrival,
        },
    }))
}
