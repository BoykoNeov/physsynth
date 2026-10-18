//! The Kirchhoff plate — `serialize.py`'s plate builder (models #5, #5b and #5g).
//!
//! One model key, three plates, carried by the secondary "domain" select: `supported` (the
//! simply-supported rectangle, closed-form oracle), `free` (the free rectangle, whose reference is
//! the tabulated Leissa square) and `guitar` (a free plate with a guitar outline, whose headline is
//! the waist crossing: deepening the waist swaps the fundamental from a bending mode to a twist).
//! It reuses the membrane's 2-D machinery; what it adds is the time control `mu = kappa k / h²`
//! (implicit, so no CFL — cost explodes at LOW mu), the outline report, and for the guitar a
//! pooled display decimation that cannot split a concave outline.
//!
//! The guitar's claim reads eigenVECTORS (the fundamental's mirror parity, and how hard the strike
//! hits each of the first modes). Both are invariant to each vector's sign by construction — a
//! quadratic form and an absolute value — so the solver's arbitrary signs cannot reach the payload
//! (retirement plan §23.11).

use physsynth_analysis::modal;
use physsynth_core::eigs::{eigsh_shift_invert, Eigenpairs};
use physsynth_core::engine::simulate;
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::fmt::py_float;
use physsynth_core::ops2d::{embed, guitar_mask, prune_to_area_carrying, Mask};
use physsynth_core::plate::{self as pl, linspace0, Boundary, Domain, Plate, PlateSpec};
use physsynth_core::plate_tuning::{
    lowest_modes, mode_frequency, mode_t60, preset, tune_plate, GrainAlong, Material, PlateTuning,
    SetBy, Targets, PRESETS,
};
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::reduce::sum;
use serde_json::{json, Map, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::horizon::{
    grid2d_block, horizon_none, Grid2dScheme, HORIZON_FREE_PLATE, HORIZON_STAIRCASE,
};
use crate::membrane::{decimate_field_mask, modal_spectrum_block, DISPLAY_MAX};
use crate::py::{
    b64f32, b64u8, finite_list, int, max_abs, num, py_int, py_repr, py_str, repr_str, round_int,
    round_nd,
};
use crate::string::{
    construction, int_at_least_one, ANIM_WIN_MAX, FRAMES_PER_PERIOD, MAX_FRAMES, MUSICAL_FS_MAX,
    MUSICAL_FS_MIN, N_MIN, SPEED_MAX,
};
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling.
pub const PLATE_N_MAX: i64 = 80;
/// Live-node ceiling (shared with the membrane's reason).
pub const PLATE_NLIVE_MAX: usize = 9_900;
/// Node-steps across the audio and animation runs.
pub const PLATE_WORK_MAX: f64 = 7.0e8;
/// Longest audio, seconds.
pub const PLATE_AUDIO_MAX: f64 = 2.0;
/// Implicit, so no CFL; a large `mu` is merely coarse.
pub const PLATE_MU_MAX: f64 = 32.0;
/// Discrete eigenmodes marked on the spectrum panel.
pub const N_PLATE_MODES: usize = 6;
/// Waists swept for the crossing panel.
pub const GUITAR_SWEEP_POINTS: usize = 24;
/// The sweep's own grid ceiling, decoupled from the audio run.
pub const GUITAR_SWEEP_N_MAX: i64 = 40;
/// Past this the outline staircases into two plates on any grid offered.
pub const GUITAR_WAIST_MAX: f64 = 0.88;
/// `|pluck_x - 1/2|` below this and the odd (twist) family is barely struck.
pub const GUITAR_CENTRELINE_TOL: f64 = 0.04;

fn pf(x: f64) -> String {
    py_float(x)
}

/// Python's `min` over a sequence: the first element unless a later one is strictly smaller.
fn py_min(vals: &[f64]) -> f64 {
    let mut m = vals[0];
    for &v in &vals[1..] {
        if v < m {
            m = v;
        }
    }
    m
}

/// `np.linspace(0, stop, num)` for any `num` (the core's `linspace0` wants two or more).
fn linspace_from0(stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => linspace0(stop, num),
    }
}

/// `np.sort` for finite floats.
fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// The eigen-solve's refusal, as the reference's uncaught ARPACK error would have surfaced.
fn eig_err(e: physsynth_core::eigs::EigsError) -> Refusal {
    Refusal::Internal(e.to_string())
}

/// `-1e-3 * (13 / (Lx Ly))²`: a safe negative shift off a free plate's rigid nullspace.
fn free_shift(lx: f64, ly: f64) -> f64 {
    -1e-3 * scalar_pow(13.0 / (lx * ly), 2.0)
}

/// `_plate_discrete_eigenfreqs`: the lowest discrete plate eigenfrequencies (Hz).
pub fn discrete_eigenfreqs(p: &pl::Params, k_request: usize) -> Result<Vec<f64>, Refusal> {
    match p.boundary {
        Boundary::Supported => {
            let k = k_request.min(p.n_live.saturating_sub(1));
            if k < 1 {
                return Ok(Vec::new());
            }
            // A grained plate's B is not L·L, so its frequencies are not functions of the
            // Laplacian eigenvalue: read B itself. Born with the musical controls (plate mapping
            // plan §14) — before them the viewer never built a grained plate.
            if !p.grain_is_isotropic {
                let pairs = eigsh_shift_invert(&p.stiffness, None, 0.0, k).map_err(eig_err)?;
                return Ok(sorted(
                    pairs
                        .values
                        .iter()
                        .map(|&v| modal::discrete_beam_eigenfrequency(v, p.kappa, p.k, p.theta))
                        .collect(),
                ));
            }
            let neg_l = p
                .laplacian
                .as_ref()
                .expect("a supported plate carries its Laplacian")
                .scaled(-1.0);
            let pairs = eigsh_shift_invert(&neg_l, None, 0.0, k).map_err(eig_err)?;
            Ok(sorted(
                pairs
                    .values
                    .iter()
                    .map(|&l| modal::discrete_plate_eigenfrequency(l, p.kappa, p.k, p.theta))
                    .collect(),
            ))
        }
        Boundary::Free => {
            let k = (k_request + 3).min(p.n_live.saturating_sub(1));
            if k < 4 {
                return Ok(Vec::new());
            }
            let pairs = free_modes(p, k)?;
            Ok(sorted(
                pairs.values[3..]
                    .iter()
                    .map(|&v| {
                        modal::discrete_beam_eigenfrequency(v.max(0.0), p.kappa, p.k, p.theta)
                    })
                    .collect(),
            ))
        }
    }
}

/// `eigsh(K, k, M=W, sigma=<negative>)` on a free plate, ascending.
fn free_modes(p: &pl::Params, k: usize) -> Result<Eigenpairs, Refusal> {
    let w = p
        .mass
        .as_ref()
        .expect("a free plate carries its lumped mass");
    eigsh_shift_invert(&p.stiffness, Some(w), free_shift(p.lx, p.ly), k).map_err(eig_err)
}

/// `_plate_continuum`: the rectangle's Navier law, the free square's Leissa anchor, or nothing —
/// a curved outline gets no reference, whatever its bounding box.
fn continuum(p: &pl::Params, n_modes: usize) -> Vec<f64> {
    if n_modes < 1 || p.domain != Domain::Rectangle {
        return Vec::new();
    }
    match p.boundary {
        Boundary::Supported => {
            let nm = n_modes as i64;
            let modes: Vec<(i64, i64)> = (1..=nm)
                .flat_map(|m| (1..=nm).map(move |n| (m, n)))
                .collect();
            // A grained plate (the musical controls' woods) takes the orthotropic Navier law.
            let freqs = if p.grain_is_isotropic {
                modal::rectangular_plate_freqs(p.kappa, p.lx, p.ly, &modes)
            } else {
                match modal::orthotropic_plate_freqs(
                    p.kappa,
                    p.lx,
                    p.ly,
                    &modes,
                    p.grain_x,
                    p.grain_cross,
                    p.grain_y,
                ) {
                    Ok(f) => f,
                    Err(_) => return Vec::new(),
                }
            };
            let mut f = sorted(freqs);
            f.truncate(n_modes);
            f
        }
        Boundary::Free => {
            // Leissa's table is the ISOTROPIC free square; a grained free plate has no tabulated
            // reference, so it gets none rather than an isotropic one for a different plate.
            if !p.grain_is_isotropic || (p.lx - p.ly).abs() > 0.02 * p.lx.max(p.ly) {
                return Vec::new();
            }
            let a2 = p.lx * p.ly;
            sorted(
                modal::free_plate_ffff_square_lambdas()
                    .iter()
                    .map(|l| l * p.kappa / (2.0 * std::f64::consts::PI * a2))
                    .collect(),
            )
        }
    }
}

/// True when a plate request asks for musical controls (`controls: "musical"`).
///
/// Its own key, not `domain`: the plate's `domain` already means supported / free / guitar, and
/// keyed on a value rather than on the presence of `f0`, because the front-end sends every slider
/// it has to every model.
pub fn is_musical(p: &Value) -> bool {
    p.get("controls").and_then(Value::as_str) == Some("musical")
}

/// What a musical-controls plate was built from, for its `tuning` block.
#[derive(Debug, Clone)]
pub struct Musical {
    /// The mapping's result.
    pub tuning: PlateTuning,
    /// The targets it met.
    pub targets: Targets,
    /// The material's name as requested (`custom` for the sliders).
    pub material: String,
    /// Where the material's numbers come from.
    pub source: &'static str,
    /// The grain direction, `None` for isotropic material.
    pub grain: Option<GrainAlong>,
}

/// The musical-controls plate (`docs/dev/plate-mapping-plan.md` §7): material, size, grid, sample
/// rate and either a thickness or a pitch, plus two ring times, mapped by [`tune_plate`]. The
/// request's `kappa`, `rho`, `mu`, `sigma` and `nu` are ignored here — the material and the targets
/// decide them — and `mu` becomes a read-out, still held to [`PLATE_MU_MAX`].
fn build_musical(
    p: &Value,
    boundary: Boundary,
    outline: Domain,
    waist: f64,
    asym: f64,
) -> Result<(Plate, f64, Option<Musical>), Refusal> {
    let lx = fnum(p, "Lx", 1.0)?;
    let ly = fnum(p, "Ly", 1.0)?;
    let theta = fnum(p, "theta", 0.28)?;
    let nv = p.get("N").cloned().unwrap_or(json!(60));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    if !(N_MIN..=PLATE_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {PLATE_N_MAX}] for the plate, got {n}."
        )));
    }
    if py_min(&[lx, ly]) <= 0.0 {
        return Err(Refusal::Param("Lx, Ly must both be positive.".into()));
    }
    let fs = fnum(p, "sample_rate", 44100.0)?;
    if !(MUSICAL_FS_MIN..=MUSICAL_FS_MAX).contains(&fs) {
        return Err(Refusal::Param(format!(
            "sample_rate must be in [{}, {}] Hz, got {}.",
            pf(MUSICAL_FS_MIN),
            pf(MUSICAL_FS_MAX),
            pf(fs)
        )));
    }
    let name = p
        .get("material")
        .and_then(Value::as_str)
        .unwrap_or("steel")
        .to_owned();
    let (material, source) = if name == "custom" {
        let material = Material::Isotropic {
            e: fnum(p, "custom_E_gpa", 100.0)? * 1e9,
            nu: fnum(p, "custom_nu", 0.3)?,
            rho: fnum(p, "custom_density", 5000.0)?,
        };
        (material, "custom constants")
    } else {
        let pr = preset(&name).ok_or_else(|| {
            let names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
            Refusal::Param(format!(
                "material must be one of {} or 'custom', got {}.",
                names.join(", "),
                repr_str(&name)
            ))
        })?;
        (pr.material, pr.source)
    };
    let default_grain = if outline == Domain::Guitar { "y" } else { "x" };
    let along = match p
        .get("grain")
        .and_then(Value::as_str)
        .unwrap_or(default_grain)
    {
        "x" => GrainAlong::X,
        "y" => GrainAlong::Y,
        other => {
            return Err(Refusal::Param(format!(
                "grain must be 'x' or 'y', got {}.",
                repr_str(other)
            )))
        }
    };
    let set_by = match p.get("set_by").and_then(Value::as_str).unwrap_or("pitch") {
        "pitch" => SetBy::Pitch(fnum(p, "f0", 220.0)?),
        "thickness" => SetBy::Thickness(fnum(p, "thickness_mm", 2.0)? * 1e-3),
        other => {
            return Err(Refusal::Param(format!(
                "set_by must be 'pitch' or 'thickness', got {}.",
                repr_str(other)
            )))
        }
    };
    let targets = Targets {
        set_by,
        t60_low: fnum(p, "t60_low", 2.0)?,
        t60_high: fnum(p, "t60_high", 0.5)?,
        f_ref: fnum(p, "f_ref", 2000.0)?,
    };
    let geometry = PlateSpec {
        lx,
        ly,
        fs,
        n,
        theta,
        boundary: Some(boundary),
        domain: Some(outline),
        waist,
        asym,
        ..PlateSpec::default()
    };
    let tuning = tune_plate(&geometry, &material, along, &targets)
        .map_err(|e| Refusal::Param(e.to_string()))?;
    let mut spec = geometry;
    tuning.apply(&mut spec);
    let params = pl::Params::new(&spec).map_err(construction)?;
    if params.n_live > PLATE_NLIVE_MAX {
        return Err(Refusal::Param(format!(
            "this geometry has {} interior nodes (> {PLATE_NLIVE_MAX}); reduce N or use a less \
             extreme aspect ratio.",
            params.n_live
        )));
    }
    if params.mu > PLATE_MU_MAX {
        return Err(Refusal::Param(format!(
            "this pitch on this grid needs mu = {} (> {}): too coarse in time. Raise the sample \
             rate or reduce N.",
            pf(round_nd(params.mu, 3)),
            pf(PLATE_MU_MAX)
        )));
    }
    let grain = matches!(material, Material::Wood { .. }).then_some(along);
    Ok((
        Plate::new(params),
        fs,
        Some(Musical {
            tuning,
            targets,
            material: name,
            source,
            grain,
        }),
    ))
}

/// `_build_plate`: the plate, its sample rate, and what musical controls built it from.
pub fn build_plate(p: &Value) -> Result<(Plate, f64, Option<Musical>), Refusal> {
    let dv = p.get("domain").cloned().unwrap_or(json!("supported"));
    let domain = py_str(&dv);
    if !["supported", "free", "guitar"].contains(&domain.as_str()) {
        return Err(Refusal::Param(format!(
            "plate domain must be 'supported', 'free' or 'guitar', got {}.",
            repr_str(&domain)
        )));
    }
    let boundary = if domain == "supported" {
        Boundary::Supported
    } else {
        Boundary::Free
    };
    let outline = if domain == "guitar" {
        Domain::Guitar
    } else {
        Domain::Rectangle
    };
    let waist = fnum(p, "waist", 0.42)?;
    let asym = fnum(p, "asym", 0.30)?;
    if outline == Domain::Guitar {
        if !(0.0..=GUITAR_WAIST_MAX).contains(&waist) {
            return Err(Refusal::Param(format!(
                "waist must be in [0, {}], got {}. Deeper than that and the outline staircases \
                 into two disconnected plates on any grid this viewer offers.",
                pf(GUITAR_WAIST_MAX),
                pf(waist)
            )));
        }
        if !(-1.0 < asym && asym < 1.0) {
            return Err(Refusal::Param(format!(
                "asym must be in (-1, 1), got {}.",
                pf(asym)
            )));
        }
    }
    if is_musical(p) {
        return build_musical(p, boundary, outline, waist, asym);
    }
    let kappa = fnum(p, "kappa", 20.0)?;
    let rho = fnum(p, "rho", 0.005)?;
    let lx = fnum(p, "Lx", 1.0)?;
    let ly = fnum(p, "Ly", 1.0)?;
    let mu = fnum(p, "mu", 1.0)?;
    let sigma = fnum(p, "sigma", 0.0)?;
    let nu = fnum(p, "nu", 0.3)?;
    let theta = fnum(p, "theta", 0.28)?;
    let nv = p.get("N").cloned().unwrap_or(json!(60));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;

    if !(N_MIN..=PLATE_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {PLATE_N_MAX}] for the plate, got {n}."
        )));
    }
    if py_min(&[kappa, rho, lx, ly]) <= 0.0 {
        return Err(Refusal::Param(
            "kappa, rho, Lx, Ly must all be positive.".into(),
        ));
    }
    if sigma < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma (loss) must be >= 0, got {}.",
            pf(sigma)
        )));
    }
    if !(0.0 < mu && mu <= PLATE_MU_MAX) {
        return Err(Refusal::Param(format!(
            "mu (plate Courant) must be in (0, {}], got {}.",
            pf(PLATE_MU_MAX),
            pf(mu)
        )));
    }
    let h = lx / n as f64;
    let fs = kappa / (mu * h * h);
    let plate = make_plate(
        lx, ly, kappa, rho, fs, n, sigma, boundary, nu, theta, outline, waist, asym,
    )?;
    if plate.p.n_live > PLATE_NLIVE_MAX {
        return Err(Refusal::Param(format!(
            "this geometry has {} interior nodes (> {PLATE_NLIVE_MAX}); reduce N or use a less \
             extreme aspect ratio.",
            plate.p.n_live
        )));
    }
    Ok((plate, fs, None))
}

/// `Plate(Lx=, Ly=, kappa=, rho=, fs=, N=, sigma=, boundary=, nu=, theta=, domain=, waist=,
/// asym=)`, a construction refusal as the binding's `ValueError`.
#[allow(clippy::too_many_arguments)]
fn make_plate(
    lx: f64,
    ly: f64,
    kappa: f64,
    rho: f64,
    fs: f64,
    n: i64,
    sigma: f64,
    boundary: Boundary,
    nu: f64,
    theta: f64,
    domain: Domain,
    waist: f64,
    asym: f64,
) -> Result<Plate, Refusal> {
    let spec = PlateSpec {
        lx,
        ly,
        kappa,
        rho,
        fs,
        n,
        sigma,
        theta,
        boundary: Some(boundary),
        domain: Some(domain),
        waist,
        asym,
        nu: Some(nu),
        ..PlateSpec::default()
    };
    let params = pl::Params::new(&spec).map_err(construction)?;
    Ok(Plate::new(params))
}

/// The live-node values of a full node field.
fn to_live(field: &[f64], mask: &Mask) -> Vec<f64> {
    field
        .iter()
        .zip(mask.flags())
        .filter_map(|(&v, &alive)| alive.then_some(v))
        .collect()
}

/// `_pool_field_mask`: decimate for display WITHOUT ever disconnecting the outline — a display
/// cell is live iff any node in its block is; the field is point-sampled where the block's
/// representative node is live and falls back to the block's live-node mean only where it is not.
pub fn pool_field_mask(frames: &[Vec<f64>], mask: &Mask) -> (Vec<f64>, Vec<u8>, usize, usize) {
    let (ny, nx) = (mask.nrows(), mask.ncols());
    let s = ny.max(nx).div_ceil(DISPLAY_MAX).max(1);
    if s == 1 {
        let flat: Vec<f64> = frames.iter().flatten().copied().collect();
        let m: Vec<u8> = mask.flags().iter().map(|&b| u8::from(b)).collect();
        return (flat, m, ny, nx);
    }
    let (py, px) = (ny.div_ceil(s), nx.div_ceil(s));
    let live = |j: usize, i: usize| j < ny && i < nx && mask.at(j, i);
    let mut mask_dec = vec![false; py * px];
    let mut counts = vec![0usize; py * px];
    for bj in 0..py {
        for bi in 0..px {
            for dj in 0..s {
                for di in 0..s {
                    if live(bj * s + dj, bi * s + di) {
                        mask_dec[bj * px + bi] = true;
                        counts[bj * px + bi] += 1;
                    }
                }
            }
        }
    }
    let mut out = Vec::with_capacity(frames.len() * py * px);
    for fr in frames {
        let val = |j: usize, i: usize| {
            if j < ny && i < nx {
                fr[j * nx + i]
            } else {
                0.0
            }
        };
        for bj in 0..py {
            for bi in 0..px {
                let (j0, i0) = (bj * s, bi * s);
                let need = mask_dec[bj * px + bi] && !live(j0, i0);
                if need {
                    let mut total = 0.0;
                    for dj in 0..s {
                        for di in 0..s {
                            let (j, i) = (j0 + dj, i0 + di);
                            let l = if live(j, i) { 1.0 } else { 0.0 };
                            total += val(j, i) * l;
                        }
                    }
                    let c = counts[bj * px + bi];
                    out.push(if c > 0 { total / c.max(1) as f64 } else { 0.0 });
                } else {
                    out.push(val(j0, i0));
                }
            }
        }
    }
    (out, mask_dec.iter().map(|&b| u8::from(b)).collect(), py, px)
}

/// `_display_components`: 4-connected component count of a decimated mask.
pub fn display_components(mask: &[u8], ny: usize, nx: usize) -> usize {
    let mut seen = vec![false; ny * nx];
    let mut n = 0;
    for start in 0..ny * nx {
        if mask[start] == 0 || seen[start] {
            continue;
        }
        n += 1;
        seen[start] = true;
        let mut stack = vec![start];
        while let Some(c) = stack.pop() {
            let (y, x) = (c / nx, c % nx);
            let mut visit = |j: usize, i: usize| {
                let q = j * nx + i;
                if mask[q] != 0 && !seen[q] {
                    seen[q] = true;
                    stack.push(q);
                }
            };
            if y + 1 < ny {
                visit(y + 1, x);
            }
            if y > 0 {
                visit(y - 1, x);
            }
            if x + 1 < nx {
                visit(y, x + 1);
            }
            if x > 0 {
                visit(y, x - 1);
            }
        }
    }
    n
}

/// `_outline_block`: what the plate reports about its own outline, read off the core.
fn outline_block(p: &pl::Params) -> Value {
    json!({
        "domain": p.domain.name(),
        "n_live": int(p.n_live as i64),
        "n_box": int(p.mask.flags().len() as i64),
        "n_pruned": int(p.n_pruned as i64),
        "area": num(round_nd(p.area, 8)),
        "outline_area": num(round_nd(p.outline_area, 8)),
        "area_deficit": num(round_nd(p.area_deficit, 6)),
        "prune_depth_h": num(round_nd(p.prune_depth_max / p.h, 4)),
        "waist": num(round_nd(p.waist, 4)),
        "asym": num(round_nd(p.asym, 4)),
    })
}

/// `_mirror_parity`: `<phi, mirror(phi)> / <phi, phi>` — reads ±1 exactly on the mirror-symmetric
/// outline, and is invariant to the vector's sign.
fn mirror_parity(p: &pl::Params, vec: &[f64]) -> f64 {
    let full = embed(vec, &p.index_map);
    let (ny, nx) = (p.mask.nrows(), p.mask.ncols());
    let sq: Vec<f64> = full.iter().map(|v| v * v).collect();
    let mut mir = Vec::with_capacity(full.len());
    for j in 0..ny {
        for i in 0..nx {
            mir.push(full[j * nx + i] * full[j * nx + (nx - 1 - i)]);
        }
    }
    let denom = sum(&sq);
    if denom != 0.0 {
        sum(&mir) / denom
    } else {
        0.0
    }
}

/// `_guitar_elastic_modes`: the three elastic eigenpairs above the rigid trio.
fn guitar_elastic_modes(p: &pl::Params) -> Result<(Vec<f64>, Vec<Vec<f64>>), Refusal> {
    let pairs = free_modes(p, 6)?;
    let vals = pairs.values[3..].iter().map(|v| v.max(0.0)).collect();
    Ok((vals, pairs.vectors[3..].to_vec()))
}

/// `_waist_quantisation`: the waist slider is quantised — report the dead band around the current
/// waist and a lower bound on how many distinct plates the slider reaches.
fn waist_quantisation(p: &pl::Params) -> Value {
    let h = p.lx / p.n as f64;
    let ny = round_int(p.ly / h).max(1) as usize;
    let (nrows, ncols) = (ny + 1, p.n + 1);
    let xs = linspace_from0(p.lx, ncols);
    let ys = linspace_from0(p.ly, nrows);
    let mut xc = Vec::with_capacity(nrows * ncols);
    let mut yc = Vec::with_capacity(nrows * ncols);
    for &yv in &ys {
        for &xv in &xs {
            xc.push(xv - 0.5 * p.lx);
            yc.push(yv);
        }
    }
    let mask_at = |w: f64| {
        prune_to_area_carrying(&guitar_mask(&xc, &yc, p.ly, p.lx, w, p.asym, nrows, ncols)).0
    };
    let here = mask_at(p.waist);
    let edge = |direction: f64| -> f64 {
        let mut same = p.waist;
        let mut other = (p.waist + direction * GUITAR_WAIST_MAX).clamp(0.0, GUITAR_WAIST_MAX);
        if mask_at(other) == here {
            return other;
        }
        for _ in 0..20 {
            let mid = 0.5 * (same + other);
            if mask_at(mid) == here {
                same = mid;
            } else {
                other = mid;
            }
        }
        same
    };
    // `np.arange(0.0, GUITAR_WAIST_MAX + 0.5 * step, step)`: `i * step`, length by ceil.
    let step = 0.001;
    let count = ((GUITAR_WAIST_MAX + 0.5 * step) / step).ceil() as usize;
    let mut prev: Option<Mask> = None;
    let mut distinct = 0;
    for i in 0..count {
        let m = mask_at(i as f64 * step);
        if prev.as_ref() != Some(&m) {
            distinct += 1;
            prev = Some(m);
        }
    }
    json!({
        "dead_band": [num(round_nd(edge(-1.0), 5)), num(round_nd(edge(1.0), 5))],
        "distinct": int(distinct),
        "sampling": num(step),
    })
}

/// `_guitar_claim_block`: the waist sweep — deepening the waist SWAPS the fundamental.
fn guitar_claim_block(
    shipped: &pl::Params,
    pluck_fx: f64,
    pluck_fy: f64,
    pluck_wfrac: f64,
) -> Result<Value, Refusal> {
    let n_sweep = (shipped.n as i64).min(GUITAR_SWEEP_N_MAX);
    // A coarser GRID, never a different SCHEME: the sweep keeps the shipped plate's `mu`.
    let h_sweep = shipped.lx / n_sweep as f64;
    let fs_sweep = shipped.kappa / (shipped.mu * h_sweep * h_sweep);
    let waists = linspace_from0(GUITAR_WAIST_MAX, GUITAR_SWEEP_POINTS);
    let mut rows: Vec<Map<String, Value>> = Vec::new();
    let mut parities: Vec<f64> = Vec::new();
    let mut row_waists: Vec<f64> = Vec::new();
    for &w in &waists {
        let swept = if shipped.grain_is_isotropic {
            make_plate(
                shipped.lx,
                shipped.ly,
                shipped.kappa,
                shipped.rho,
                fs_sweep,
                n_sweep,
                0.0,
                Boundary::Free,
                shipped.nu,
                shipped.theta,
                Domain::Guitar,
                w,
                shipped.asym,
            )
        } else {
            // A grained top (the musical controls' woods) sweeps as ITSELF: the free branch's
            // four constants carried over, not an isotropic plate at its derived nu.
            pl::Params::new(&PlateSpec {
                lx: shipped.lx,
                ly: shipped.ly,
                kappa: shipped.kappa,
                rho: shipped.rho,
                fs: fs_sweep,
                n: n_sweep,
                theta: shipped.theta,
                boundary: Some(Boundary::Free),
                domain: Some(Domain::Guitar),
                waist: w,
                asym: shipped.asym,
                grain_x: shipped.grain_x,
                grain_y: shipped.grain_y,
                grain_coupling: Some(shipped.grain_coupling),
                grain_torsion: Some(shipped.grain_torsion),
                ..PlateSpec::default()
            })
            .map(Plate::new)
            .map_err(construction)
        };
        let Ok(q) = swept else {
            continue; // this waist staircases into two plates on the sweep's grid
        };
        let qp = &q.p;
        let (vals, mut vecs) = guitar_elastic_modes(qp)?;
        let freqs: Vec<f64> = vals
            .iter()
            .map(|&v| modal::discrete_beam_eigenfrequency(v, qp.kappa, qp.k, qp.theta))
            .collect();
        let wdiag = &qp.w;
        for v in vecs.iter_mut() {
            let wv: Vec<f64> = v.iter().zip(wdiag).map(|(a, b)| b * a).collect();
            let nrm = pl::dot(v, &wv);
            if nrm > 0.0 {
                let s = nrm.sqrt();
                for x in v.iter_mut() {
                    *x /= s;
                }
            }
        }
        let field = raised_cosine_2d(
            &qp.x,
            &qp.y,
            (pluck_fx * qp.lx, pluck_fy * qp.ly),
            pluck_wfrac * qp.lx.min(qp.ly),
            1e-3,
        )
        .map_err(construction)?;
        let strike = to_live(&field, &qp.mask);
        let ws: Vec<f64> = wdiag.iter().zip(&strike).map(|(a, b)| a * b).collect();
        let amps: Vec<f64> = vecs.iter().take(4).map(|v| pl::dot(v, &ws).abs()).collect();
        let mx = amps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scale = if mx == 0.0 { 1.0 } else { mx };
        let parity = round_nd(mirror_parity(qp, &vecs[0]), 6);
        let mut row = Map::new();
        row.insert("waist".into(), num(round_nd(w, 4)));
        row.insert("f1".into(), num(round_nd(freqs[0], 3)));
        row.insert("f2".into(), num(round_nd(freqs[1], 3)));
        row.insert("parity".into(), num(parity));
        row.insert("a1".into(), num(round_nd(amps[0] / scale, 5)));
        row.insert(
            "a2".into(),
            num(if amps.len() > 1 {
                round_nd(amps[1] / scale, 5)
            } else {
                0.0
            }),
        );
        row.insert("area_deficit".into(), num(round_nd(qp.area_deficit, 6)));
        parities.push(parity);
        row_waists.push(round_nd(w, 4));
        rows.push(row);
    }

    // The shipped plate's OWN row, on the shipped plate — what the "you" marker points at.
    let (you_vals, you_vecs) = guitar_elastic_modes(shipped)?;
    let you_f: Vec<f64> = you_vals
        .iter()
        .map(|&v| modal::discrete_beam_eigenfrequency(v, shipped.kappa, shipped.k, shipped.theta))
        .collect();
    let you = json!({
        "f1": num(round_nd(you_f[0], 3)),
        "f2": num(round_nd(you_f[1], 3)),
        "parity": num(round_nd(mirror_parity(shipped, &you_vecs[0]), 6)),
    });

    let flips: Vec<usize> = (1..parities.len())
        .filter(|&i| parities[i] * parities[i - 1] < 0.0)
        .collect();
    // The crossing is an INTERVAL between two adjacent representable waists.
    let bracket = flips.first().map(|&i| (row_waists[i - 1], row_waists[i]));
    let side = match bracket {
        None => "none",
        Some((_, hi)) if shipped.waist > hi => "twist",
        Some((lo, _)) if shipped.waist < lo => "bender",
        Some(_) => "at",
    };
    Ok(json!({
        "kind": "waist_crossing",
        "sweep_N": int(n_sweep),
        "rows": rows,
        "n_flips": int(flips.len() as i64),
        "crossing": bracket.map_or(Value::Null, |(a, b)| json!([num(a), num(b)])),
        "shipped_waist": num(round_nd(shipped.waist, 4)),
        "you": you,
        "shipped_side": side,
        "quantisation": waist_quantisation(shipped),
        "pluck_x": num(round_nd(pluck_fx, 4)),
        "centreline_warning": (pluck_fx - 0.5).abs() < GUITAR_CENTRELINE_TOL,
    }))
}

/// `_horizon_plate_block` for the linear plate: only the supported, isotropic rectangle is
/// measured.
pub fn horizon_block(p: &pl::Params, of: &str) -> Value {
    if p.domain != Domain::Rectangle {
        return horizon_none(HORIZON_STAIRCASE, Some(of));
    }
    if p.boundary != Boundary::Supported {
        return horizon_none(HORIZON_FREE_PLATE, Some(of));
    }
    if !p.grain_is_isotropic {
        return horizon_none(
            "this plate has a grain, so both its continuum oracle and the order its spectrum \
             comes in are the orthotropic ones (plan section 9) rather than the isotropic forms \
             this read-out builds. In mode index the floor is the same; in hertz it is \
             per-direction.",
            Some(of),
        );
    }
    grid2d_block(
        p.h,
        p.k,
        p.lx,
        p.ly,
        p.fs,
        Grid2dScheme::Plate {
            kappa: p.kappa,
            theta: p.theta,
        },
        of,
    )
}

/// `_build_payload_plate`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 1.0)?;
    let anim_win = fnum(p, "animation_window", 0.03)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let amplitude = fnum(p, "amplitude", 1e-3)?;
    // The guitar's strike defaults differ, and the reason is the claim: past the crossing the
    // fundamental is odd under the mirror, and a centre-line strike barely touches it.
    let dv = p.get("domain").cloned().unwrap_or(json!("supported"));
    let guitar = py_str(&dv) == "guitar";
    let pluck_fx = fnum(p, "pluck_x", if guitar { 0.25 } else { 0.4 })?;
    let pluck_fy = fnum(p, "pluck_y", if guitar { 0.35 } else { 0.55 })?;
    let pluck_wfrac = fnum(p, "pluck_width", 0.3)?;
    let pickup_fx = fnum(p, "pickup_x", 0.62)?;
    let pickup_fy = fnum(p, "pickup_y", 0.58)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= PLATE_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(PLATE_AUDIO_MAX),
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

    let (mut res, fs, musical) = build_plate(p)?;
    // The animation run's plate: a copy of the freshly built one rather than a second build, so a
    // musical plate's mapping (and its mode solves) runs once.
    let fresh = res.clone();
    let (lx, ly) = (res.p.lx, res.p.ly);

    let n_audio = round_int(audio_dur * fs).max(1);
    let n_anim_est = round_int(anim_win * fs).max(1);
    let work = res.p.n_live as i64 * (n_audio + n_anim_est);
    if work as f64 > PLATE_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs ~{:.0}M node-steps (over the ~{:.0}M budget); RAISE mu, \
             reduce N, or shorten the audio.",
            work as f64 / 1e6,
            PLATE_WORK_MAX / 1e6
        )));
    }

    let f_disc = discrete_eigenfreqs(&res.p, N_PLATE_MODES)?;
    let f_cont = continuum(&res.p, f_disc.len());
    let f1 = if f_disc.is_empty() {
        res.p.kappa / (2.0 * scalar_pow(py_min(&[lx, ly]), 2.0))
    } else {
        f_disc[0]
    };

    // -- the audio run: a broad raised-cosine strike, one pickup
    let wc = pluck_wfrac * py_min(&[lx, ly]);
    let (pcx, pcy) = (pluck_fx * lx, pluck_fy * ly);
    let excite = |plate: &mut Plate| -> Result<(), Refusal> {
        let field = raised_cosine_2d(&plate.p.x, &plate.p.y, (pcx, pcy), wc, amplitude)
            .map_err(construction)?;
        let u0 = to_live(&field, &plate.p.mask);
        let v0 = vec![0.0; u0.len()];
        plate.set_state(&u0, &v0);
        Ok(())
    };
    excite(&mut res)?;
    let pickup_idx = pl::pickup_index_at(pickup_fx * lx, pickup_fy * ly, &res.p);
    let audio_run =
        simulate(&mut res, n_audio as usize, Some(pickup_idx), 0).map_err(Refusal::Internal)?;
    let pickup = audio_run.output.as_deref().expect("a pickup was requested");
    if !pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }

    // -- the animation run: a fresh plate, a fundamental-resolving stride
    let mut anim = fresh;
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

    let rp = &res.p;
    let (frames_dec, mask_dec, ny_dec, nx_dec) = if rp.domain == Domain::Rectangle {
        let (ny, nx) = (rp.mask.nrows(), rp.mask.ncols());
        decimate_field_mask(&frames_full, rp.mask.flags(), ny, nx)
    } else {
        // A concave outline needs the pooled path, or the picture can split where the plate
        // does not — and the proof of that is about the code as written, so it is checked.
        let out = pool_field_mask(&frames_full, &rp.mask);
        let parts = display_components(&out.1, out.2, out.3);
        if parts != 1 {
            return Err(Refusal::Param(format!(
                "the {} outline renders as {parts} disconnected pieces at the display resolution \
                 even though the plate is one piece -- refuse rather than draw two guitars. \
                 Reduce waist or N.",
                rp.domain.name()
            )));
        }
        out
    };
    let nf = frames_full.len();
    let field_amp = if frames_dec.is_empty() {
        0.0
    } else {
        max_abs(&frames_dec)
    };
    let (audio48, peak) = resample_normalize(pickup, fs);

    let mut meta = Map::new();
    meta.insert("kappa".into(), num(round_nd(rp.kappa, 4)));
    meta.insert("f1".into(), num(round_nd(f1, 3)));
    meta.insert("num_steps".into(), int(n_audio));
    meta.insert("n_frames".into(), int(nf as i64));
    meta.insert(
        "spectrum".into(),
        modal_spectrum_block(pickup, fs, &f_disc, &f_cont, "plate"),
    );
    meta.insert("outline_info".into(), outline_block(rp));
    if rp.domain == Domain::Guitar {
        meta.insert(
            "claim".into(),
            guitar_claim_block(rp, pluck_fx, pluck_fy, pluck_wfrac)?,
        );
    }
    let boundary = match rp.boundary {
        Boundary::Supported => "supported",
        Boundary::Free => "free",
    };
    // A frequency-dependent loss decays each mode at its own rate, so the single `2 sigma` oracle
    // line would be a fitted lie; only a musical plate carries one.
    let energy = energy_block(
        &audio_run.time,
        &audio_run.energy,
        rp.sigma == 0.0 && rp.sigma1 == 0.0,
        2.0 * rp.sigma,
        EnergyOpts {
            no_decay_oracle: rp.sigma1 > 0.0,
            ..EnergyOpts::default()
        },
    );
    let tuning = match &musical {
        Some(m) => Some(tuning_block(rp, m)?),
        None => None,
    };
    let mut payload = json!({
        "model": "plate",
        "horizon": horizon_block(rp, "the plate"),
        "boundary": boundary,
        "outline": rp.domain.name(),
        "fs_sim": num(round_nd(fs, 3)),
        "mu": num(round_nd(rp.mu, 6)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(lx, 6)), "extent_y": num(round_nd(ly, 6)),
            "domain": rp.domain.name(),
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
        "energy": energy,
        "meta": meta,
    });
    if let Some(t) = tuning {
        payload["tuning"] = t;
    }
    Ok(payload)
}

/// How many partials the `tuning` block reports pitch and ring time for.
pub const TUNING_PARTIALS: usize = 8;

/// Below this relative gap to its nearest neighbour the anchor is called crowded: on the free
/// plates that is where the mapping's miss is largest (plan §9).
pub const CROWDED_ANCHOR: f64 = 0.01;

/// The `tuning` block of a musical-controls payload: what was asked, what it mapped to, and what
/// the partials the mapping does not pin do. Their pitch drift is the horizon strip's business
/// (plan §7: one drift read-out, not two); their ring times are only here.
fn tuning_block(p: &pl::Params, m: &Musical) -> Result<Value, Refusal> {
    let (tu, tg) = (&m.tuning, &m.targets);
    let (k, theta, kappa2) = (p.k, p.theta, p.kappa * p.kappa);
    let modes = lowest_modes(p, TUNING_PARTIALS).map_err(|e| Refusal::Param(e.to_string()))?;
    let partials: Vec<Value> = modes
        .iter()
        .filter_map(|md| {
            let x = k * k * kappa2 * md.beta;
            let y = k * (tu.sigma0 + tu.sigma1 * md.lambda);
            let f = mode_frequency(x, y, k, theta)?;
            Some(json!({
                "f": num(round_nd(f, 3)),
                "t60": num(round_nd(mode_t60(x, y, k, theta), 4)),
            }))
        })
        .collect();
    let crowded = tu.anchor_margin < CROWDED_ANCHOR;
    let mut message = if tu.exact {
        "The simulated plate meets the pitch and both ring times exactly.".to_owned()
    } else {
        "Free edges couple the modes slightly: the simulated plate meets the targets to within \
         about 0.05 cents and 0.3 % in ring time."
            .to_owned()
    };
    if crowded {
        message.push_str(&format!(
            " The anchor partial ({} Hz) sits within {:.1} % of another partial, where the \
             ring-time target is least certain; moving f_ref picks a clearer one.",
            pf(round_nd(tu.anchor_frequency, 1)),
            100.0 * tu.anchor_margin
        ));
    }
    let (set_by, f0_target, thickness_target) = match tg.set_by {
        SetBy::Pitch(f) => ("pitch", num(f), Value::Null),
        SetBy::Thickness(t) => ("thickness", Value::Null, num(round_nd(t * 1e3, 6))),
    };
    Ok(json!({
        "material": m.material,
        "source": m.source,
        "grain": match m.grain {
            Some(GrainAlong::X) => json!("x"),
            Some(GrainAlong::Y) => json!("y"),
            None => Value::Null,
        },
        "set_by": set_by,
        "f0_target": f0_target,
        "thickness_mm_target": thickness_target,
        "thickness_mm": num(round_nd(tu.thickness * 1e3, 4)),
        "rho_s": num(round_nd(tu.material.rho_s, 6)),
        "kappa": num(round_nd(tu.material.kappa, 6)),
        "sigma0": num(round_nd(tu.sigma0, 6)),
        "sigma1": num(round_nd(tu.sigma1, 8)),
        "f0": num(round_nd(tu.f0, 3)),
        "t60_low": num(tg.t60_low),
        "t60_high": num(tg.t60_high),
        "f_ref": num(tg.f_ref),
        "anchor_frequency": num(round_nd(tu.anchor_frequency, 3)),
        "anchor_margin": num(round_nd(tu.anchor_margin, 4)),
        "exact": tu.exact,
        "crowded": crowded,
        "sample_rate": num(p.fs),
        "mu": num(round_nd(p.mu, 4)),
        "partials": partials,
        "message": message,
    }))
}
