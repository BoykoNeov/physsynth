//! The von Kármán nonlinear plate — `serialize.py`'s vk builder (model #6: the gong and the cymbal).
//!
//! Two coupled fields and a fixed-point iteration per step, and NO analytic modal oracle: energy
//! conservation is the correctness test, and it holds only at the fixed point, so the verdict is
//! gated on convergence. The pitch HARDENS with amplitude, so the spectrum panel's marker lines are
//! the LINEAR (w -> 0) modes and the real peaks sit above them; the hardened fundamental is read by
//! zero-crossing spacing, and only on the supported gong — a free-edge crash is a mode wash.

use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::fmt::py_float;
use physsynth_core::plate::{self as pl, Boundary, VkParams, VkPlate, VkSpec};
use physsynth_core::plate_tuning::{
    lowest_modes, mode_frequency, mode_t60, preset, tune_vk_plate, Material, SetBy, Targets,
    VkTuning, PRESETS,
};
use physsynth_core::pyfloat::scalar_pow;
use serde_json::{json, Value};

use crate::energy::{energy_block, EnergyOpts};
use crate::horizon::{horizon_none, HORIZON_NONLINEAR};
use crate::membrane::decimate_field_mask;
use crate::plate::{
    discrete_eigenfreqs, free_modes, horizon_block, is_musical, CROWDED_ANCHOR, TUNING_PARTIALS,
};
use crate::py::{
    as_bool, b64f32, b64u8, finite_list, int, max_abs, np_mean, num, py_int, py_repr, py_str,
    repr_str, round_int, round_nd,
};
use crate::string::{
    construction, int_at_least_one, ANIM_WIN_MAX, FRAMES_PER_PERIOD, MAX_FRAMES, N_MIN, SPEED_MAX,
};
use crate::tension::pooled_spectrum;
use crate::{fnum, resample_normalize, Refusal, AUDIO_FS};

/// Grid ceiling.
pub const VK_N_MAX: i64 = 32;
/// Live-node ceiling (a free plate is every-node-live).
pub const VK_NLIVE_MAX: usize = 1_600;
/// Longest audio, seconds.
pub const VK_AUDIO_MAX: f64 = 1.0;
/// Oversample around the nonlinearity: the sample rate is the time control, in this range.
pub const VK_FS_MIN: f64 = 8_000.0;
/// See [`VK_FS_MIN`].
pub const VK_FS_MAX: f64 = 96_000.0;
/// Picard safety cap, and the worst-case cost multiplier.
pub const VK_COUPLE_MAX_ITER: i64 = 50;
/// `n_live x steps x couple_max_iter`.
pub const VK_WORK_MAX: f64 = 2.0e9;
/// Strike amplitude in thickness units — the hardening knob.
pub const VK_WOVERE_MAX: f64 = 6.0;
/// Linear eigenmodes marked on the spectrum panel.
pub const N_VK_MODES: usize = 6;

/// The musical controls' default settled pitch (Hz) — `docs/dev/vk-mapping-plan.md` §8.8.
pub const VK_MUSICAL_F0: f64 = 100.0;

/// The run's cap on coupled work, in node-solves (`n_live` × linear solves), counted as it runs.
///
/// The old budget, [`VK_WORK_MAX`], assumes the worst step is [`VK_COUPLE_MAX_ITER`] Picard sweeps
/// of about two solves each and never counts Newton's rescue, which can cost hundreds of times that
/// (plan §8.6). This is that budget's own promise, kept by counting: twice [`VK_WORK_MAX`].
pub const VK_SOLVE_MAX: f64 = 2.0 * VK_WORK_MAX;

/// The gong's cost table — linear solves per step, by `f₁·k` row ([`GONG_F0K`]) and strike column
/// ([`GONG_STRIKE`]); `None` is a cell where a step never settles. Plan §11.1, the worst of
/// `N` 12, 20 and 32 (the class is the same at all three, and with `ν` 0 or 0.45 and an oblong
/// outline, §11.2). Read at the cell **above** a request on both axes, the conservative corner.
const GONG_SOLVES: [[Option<f64>; 4]; 5] = [
    [Some(8.0), Some(9.5), Some(13.0), Some(17.0)],
    [Some(9.5), Some(12.0), Some(17.5), Some(24.5)],
    [Some(11.5), Some(16.0), Some(25.0), Some(42.0)],
    [Some(14.5), Some(21.0), Some(44.0), Some(128.0)],
    [Some(19.0), Some(32.0), Some(154.0), None],
];
/// [`GONG_SOLVES`]'s rows: the lowest mode's frequency times the time step.
const GONG_F0K: [f64; 5] = [0.003, 0.006, 0.0125, 0.025, 0.05];
/// [`GONG_SOLVES`]'s columns: the strike in thicknesses.
const GONG_STRIKE: [f64; 4] = [1.0, 2.0, 4.0, 6.0];
/// A lossless run costs more per step as it goes (1.6× the first 3 000 steps' rate, plan §11.3):
/// the margin a prediction from the table carries.
const GONG_LOSSLESS_RISE: f64 = 1.6;

/// What [`GONG_SOLVES`] says about one gong.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Predicted {
    /// Every step settles, at about this many solves each.
    Cost(f64),
    /// A step does not settle — measured, or past the last row in a column that already broke
    /// there (the cost rises on both axes, so a broken cell stays broken further out).
    Broken,
    /// Past the last measured row in a column that still settled: no claim either way. The run's
    /// own guard (`Guard`) is all that stands there.
    Unmeasured,
}

/// What the table predicts for a gong at `f0k` struck `strike` thicknesses deep (`strike` is held
/// to [`VK_WOVERE_MAX`], the last column, before this is asked).
fn gong_cost(f0k: f64, strike: f64) -> Predicted {
    let col = GONG_STRIKE
        .iter()
        .position(|&c| strike <= c)
        .unwrap_or(GONG_STRIKE.len() - 1);
    match GONG_F0K.iter().position(|&r| f0k <= r) {
        Some(row) => GONG_SOLVES[row][col].map_or(Predicted::Broken, Predicted::Cost),
        None => match GONG_SOLVES[GONG_F0K.len() - 1][col] {
            None => Predicted::Broken,
            Some(_) => Predicted::Unmeasured,
        },
    }
}

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

/// What a musical-controls gong was built from, for its `tuning` block.
struct Musical {
    /// The mapping's result.
    tuning: VkTuning,
    /// The targets it met.
    targets: Targets,
    /// The material's name as requested (`custom` for the sliders).
    material: String,
    /// Where the material's numbers come from.
    source: &'static str,
}

/// The gong, its sample rate and boundary, whether the coupling is on, and what musical controls
/// built it from.
type Built = (VkPlate, f64, Boundary, bool, Option<Musical>);

/// `_build_vk`.
fn build(p: &Value) -> Result<Built, Refusal> {
    let bv = p.get("domain").cloned().unwrap_or(json!("supported"));
    let boundary = py_str(&bv);
    if boundary != "supported" && boundary != "free" {
        return Err(Refusal::Param(format!(
            "boundary must be 'supported' or 'free', got {}.",
            repr_str(&boundary)
        )));
    }
    if is_musical(p) {
        let b = if boundary == "supported" {
            Boundary::Supported
        } else {
            Boundary::Free
        };
        return build_musical(p, b);
    }
    let young = fnum(p, "E", 2.0e11)?;
    let e = fnum(p, "e", 1.0e-3)?;
    let nu = fnum(p, "nu", 0.3)?;
    let rho = fnum(p, "rho", 7800.0)?;
    let lx = fnum(p, "Lx", 0.4)?;
    let ly = fnum(p, "Ly", 0.4)?;
    let fs = fnum(p, "fs", 32_000.0)?;
    let sigma = fnum(p, "sigma", 0.0)?;
    let nonlinear = as_bool(p.get("nonlinear"), true);
    let nv = p.get("N").cloned().unwrap_or(json!(20));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;

    if !(N_MIN..=VK_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {VK_N_MAX}] for the von Kármán plate, got {n}."
        )));
    }
    if py_min(&[young, e, rho, lx, ly]) <= 0.0 {
        return Err(Refusal::Param(
            "E, e, rho, Lx, Ly must all be positive.".into(),
        ));
    }
    if sigma < 0.0 {
        return Err(Refusal::Param(format!(
            "sigma (loss) must be >= 0, got {}.",
            pf(sigma)
        )));
    }
    if !(VK_FS_MIN..=VK_FS_MAX).contains(&fs) {
        return Err(Refusal::Param(format!(
            "fs must be in [{VK_FS_MIN:.0}, {VK_FS_MAX:.0}] Hz, got {}.",
            pf(fs)
        )));
    }
    let b = if boundary == "supported" {
        Boundary::Supported
    } else {
        Boundary::Free
    };
    let spec = VkSpec {
        lx,
        ly,
        young,
        thickness: e,
        nu,
        rho,
        fs,
        n,
        sigma,
        boundary: Some(b),
        nonlinear,
        couple_max_iter: VK_COUPLE_MAX_ITER,
        ..VkSpec::default()
    };
    let params = VkParams::new(&spec).map_err(construction)?;
    if params.lin.n_live > VK_NLIVE_MAX {
        return Err(Refusal::Param(format!(
            "this plate has {} live nodes (> {VK_NLIVE_MAX}); reduce N or the aspect ratio.",
            params.lin.n_live
        )));
    }
    Ok((VkPlate::new(params), fs, b, nonlinear, None))
}

/// The musical-controls gong (`docs/dev/vk-mapping-plan.md` §6): a metal or glass (or custom
/// isotropic constants), a size, a grid, a sample rate and either a thickness or a settled pitch,
/// plus two ring times, mapped by [`tune_vk_plate`]. The request's `E`, `e`, `nu`, `rho` and
/// `sigma` are ignored here — the material and the targets decide them.
fn build_musical(p: &Value, b: Boundary) -> Result<Built, Refusal> {
    let lx = fnum(p, "Lx", 0.4)?;
    let ly = fnum(p, "Ly", 0.4)?;
    let fs = fnum(p, "fs", 32_000.0)?;
    let nonlinear = as_bool(p.get("nonlinear"), true);
    let nv = p.get("N").cloned().unwrap_or(json!(20));
    let n = py_int(&nv)
        .map_err(|_| Refusal::Param(format!("N must be an integer, got {}.", py_repr(&nv))))?;
    if !(N_MIN..=VK_N_MAX).contains(&n) {
        return Err(Refusal::Param(format!(
            "N must be in [{N_MIN}, {VK_N_MAX}] for the von Kármán plate, got {n}."
        )));
    }
    if py_min(&[lx, ly]) <= 0.0 {
        return Err(Refusal::Param("Lx, Ly must both be positive.".into()));
    }
    if !(VK_FS_MIN..=VK_FS_MAX).contains(&fs) {
        return Err(Refusal::Param(format!(
            "fs must be in [{VK_FS_MIN:.0}, {VK_FS_MAX:.0}] Hz, got {}.",
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
            e: fnum(p, "custom_E_gpa", 200.0)? * 1e9,
            nu: fnum(p, "custom_nu", 0.3)?,
            rho: fnum(p, "custom_density", 7800.0)?,
        };
        (material, "custom constants")
    } else {
        let pr = preset(&name).ok_or_else(|| {
            let names: Vec<&str> = PRESETS
                .iter()
                .filter(|p| matches!(p.material, Material::Isotropic { .. }))
                .map(|p| p.name)
                .collect();
            Refusal::Param(format!(
                "material must be one of {} or 'custom', got {}.",
                names.join(", "),
                repr_str(&name)
            ))
        })?;
        (pr.material, pr.source)
    };
    let set_by = match p.get("set_by").and_then(Value::as_str).unwrap_or("pitch") {
        "pitch" => SetBy::Pitch(fnum(p, "f0", VK_MUSICAL_F0)?),
        "thickness" => SetBy::Thickness(fnum(p, "thickness_mm", 1.0)? * 1e-3),
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
    let geometry = VkSpec {
        lx,
        ly,
        fs,
        n,
        boundary: Some(b),
        nonlinear,
        couple_max_iter: VK_COUPLE_MAX_ITER,
        ..VkSpec::default()
    };
    let tuning =
        tune_vk_plate(&geometry, &material, &targets).map_err(|e| Refusal::Param(e.to_string()))?;
    let mut spec = geometry;
    tuning.apply(&mut spec);
    let params = VkParams::new(&spec).map_err(construction)?;
    if params.lin.n_live > VK_NLIVE_MAX {
        return Err(Refusal::Param(format!(
            "this plate has {} live nodes (> {VK_NLIVE_MAX}); reduce N or the aspect ratio.",
            params.lin.n_live
        )));
    }
    Ok((
        VkPlate::new(params),
        fs,
        b,
        nonlinear,
        Some(Musical {
            tuning,
            targets,
            material: name,
            source,
        }),
    ))
}

/// `_vk_strike`: the (1,1) mode on the gong (a pure mode is what makes the hardened fundamental
/// read cleanly off zero-crossings), a broad raised-cosine crash on the cymbal; zero off the mask.
fn strike(
    vk: &VkPlate,
    boundary: Boundary,
    amplitude: f64,
    centre: (f64, f64),
    wc: f64,
) -> Result<Vec<f64>, Refusal> {
    let lin = &vk.p.lin;
    let mut field: Vec<f64> = match boundary {
        Boundary::Supported => lin
            .x
            .iter()
            .zip(&lin.y)
            .map(|(&x, &y)| {
                amplitude
                    * (std::f64::consts::PI * x / lin.lx).sin()
                    * (std::f64::consts::PI * y / lin.ly).sin()
            })
            .collect(),
        Boundary::Free => {
            raised_cosine_2d(&lin.x, &lin.y, centre, wc, amplitude).map_err(construction)?
        }
    };
    for (v, &alive) in field.iter_mut().zip(lin.mask.flags()) {
        if !alive {
            *v = 0.0;
        }
    }
    Ok(field)
}

/// One run's record: energy, optional pickup, projection and snapshots, the Picard convergence
/// record, and what the coupled solves cost.
struct Run {
    time: Vec<f64>,
    energy: Vec<f64>,
    pickup: Vec<f64>,
    projection: Vec<f64>,
    snapshots: Vec<(usize, Vec<f64>)>,
    convergence: Value,
    /// Linear solves over every step.
    solves: usize,
    /// Steps Newton had to rescue.
    rescued: usize,
}

/// What a run may spend, and what a refusal tells the user to change.
struct Guard {
    /// Node-solves left of [`VK_SOLVE_MAX`], shared by the audio and animation runs.
    left: f64,
    boundary: Boundary,
}

impl Guard {
    /// The first step that never settles stops the render: its answer is not a solution of the
    /// scheme, and on a fine grid one such step can take minutes (plan §11.2 — every broken
    /// cymbal failed on its first step).
    fn unsettled(&self, step: usize) -> Refusal {
        let what = match self.boundary {
            Boundary::Supported => {
                "strike more gently (w_over_e) or raise the sample rate (fs), which shortens each \
                 step"
            }
            Boundary::Free => {
                "widen the strike (pluck_width), strike more gently (w_over_e), use a coarser grid \
                 (N) or raise the sample rate (fs) — a narrow strike or an oblong cymbal is what \
                 breaks it first"
            }
        };
        Refusal::Param(format!(
            "step {step} did not settle: the coupled solve failed even with Newton's rescue, so \
             the render was stopped rather than return a sound that is not the plate's. To fix \
             it, {what}."
        ))
    }

    fn over_budget(&self, step: usize, of: usize) -> Refusal {
        Refusal::Param(format!(
            "this render was stopped at step {step} of {of}: its coupled solves passed the \
             ~{:.0}M node-solve budget (Newton was rescuing its steps). Shorten the audio, reduce \
             N, strike more gently{}.",
            VK_SOLVE_MAX / 1e6,
            if self.boundary == Boundary::Free {
                ", or widen the strike"
            } else {
                ", or raise the sample rate"
            }
        ))
    }
}

/// `_run_vk`: `simulate`, plus the convergence stats the energy verdict is gated on. A nonlinear
/// run is charged against `guard` and stopped at its first unsettled step.
fn run(
    vk: &mut VkPlate,
    num_steps: usize,
    pickup: Option<usize>,
    project: Option<&[f64]>,
    snapshot_stride: usize,
    guard: &mut Guard,
) -> Result<Run, Refusal> {
    let n = num_steps + 1;
    let mut r = Run {
        // `np.arange(n) * vk.k` — not `i / fs`, which can differ in the last bit
        time: (0..n).map(|i| i as f64 * vk.p.lin.k).collect(),
        energy: Vec::with_capacity(n),
        pickup: Vec::new(),
        projection: Vec::new(),
        snapshots: Vec::new(),
        convergence: Value::Null,
        solves: 0,
        rescued: 0,
    };
    let proj = |vk: &VkPlate, w: &[f64]| vk.u.iter().zip(w).map(|(a, b)| a * b).sum::<f64>();
    let n_live = vk.p.lin.n_live as f64;
    let (mut n_not_conv, mut worst, mut max_iters) = (0i64, 0.0f64, 0usize);
    r.energy.push(vk.energy());
    if let Some(i) = pickup {
        r.pickup.push(vk.u[i]);
    }
    if let Some(w) = project {
        r.projection.push(proj(vk, w));
    }
    if snapshot_stride > 0 {
        r.snapshots.push((0, vk.state()));
    }
    for i in 1..n {
        vk.step(None)
            .map_err(|e| Refusal::Internal(e.to_string()))?;
        if vk.p.nonlinear {
            if !vk.converged {
                return Err(guard.unsettled(i));
            }
            guard.left -= n_live * vk.n_solves as f64;
            if guard.left < 0.0 {
                return Err(guard.over_budget(i, num_steps));
            }
        }
        r.solves += vk.n_solves;
        r.rescued += vk.n_fallbacks;
        if let Some(w) = project {
            r.projection.push(proj(vk, w));
        }
        if !vk.converged {
            n_not_conv += 1;
        }
        // Python's `max(worst, x)`: keeps `worst` unless `x` is strictly larger.
        if vk.last_residual > worst {
            worst = vk.last_residual;
        }
        max_iters = max_iters.max(vk.n_iters);
        r.energy.push(vk.energy());
        if let Some(j) = pickup {
            r.pickup.push(vk.u[j]);
        }
        if snapshot_stride > 0 && i % snapshot_stride == 0 {
            r.snapshots.push((i, vk.state()));
        }
    }
    r.convergence = json!({
        "all_converged": n_not_conv == 0,
        "n_not_converged": int(n_not_conv),
        "worst_residual": num(worst),
        "max_iters": int(max_iters as i64),
        "couple_tol": num(vk.p.couple_tol),
    });
    Ok(r)
}

/// `_zero_cross_fundamental`: `fs / (2 mean(diff(zero crossings)))`, robust to the hardening.
fn zero_cross_fundamental(sig: &[f64], fs: f64) -> f64 {
    let m = np_mean(sig);
    let neg: Vec<bool> = sig.iter().map(|v| (v - m).is_sign_negative()).collect();
    let zc: Vec<usize> = (0..neg.len().saturating_sub(1))
        .filter(|&i| neg[i] != neg[i + 1])
        .collect();
    if zc.len() < 3 {
        return f64::NAN;
    }
    let d: Vec<f64> = zc.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    fs / (2.0 * np_mean(&d))
}

/// `_vk_spectrum_block`: the pooled spectrum, the LINEAR markers and the hardening shift.
fn spectrum_block(pickup: &[f64], fs: f64, f_lin: &[f64], f0: f64) -> Value {
    if f_lin.is_empty() {
        return Value::Null;
    }
    let f1_lin = f_lin[0];
    let mut fmax = f_lin.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.6;
    if f0.is_finite() && f0 > 0.0 {
        fmax = fmax.max(f0 * 1.6);
    }
    let Some((f_ds, m_ds)) = pooled_spectrum(pickup, fs, fmax) else {
        return Value::Null;
    };
    let shift = (f0.is_finite() && f0 > 0.0 && f1_lin > 0.0).then(|| 100.0 * (f0 / f1_lin - 1.0));
    json!({
        "kind": "vk",
        "freq": finite_list(&f_ds, Some(3)),
        "mag": finite_list(&m_ds, Some(5)),
        "fmax": num(round_nd(fmax, 3)),
        "modes_linear": finite_list(f_lin, Some(4)),
        "f1_linear": num(round_nd(f1_lin, 4)),
        "f0_detected": if f0.is_finite() { num(round_nd(f0, 4)) } else { Value::Null },
        "shift_pct": shift.map_or(Value::Null, |s| num(round_nd(s, 2))),
    })
}

/// The gong's pre-run guard (plan §11.4): the table's solves per step at the lowest mode's `f₁`
/// (`None` where the table makes no claim, left to the run's own guard), or a refusal naming a
/// sample rate, and a strike, that would pass. `dur` is the audio and animation together (s).
fn gong_guard(
    f1: f64,
    fs: f64,
    strike: f64,
    n_live: usize,
    dur: f64,
) -> Result<Option<f64>, Refusal> {
    let work = |fs: f64, s: f64| n_live as f64 * dur * fs * s * GONG_LOSSLESS_RISE;
    // `Some(prediction)` when the guard lets this gong run.
    let passes = |fs: f64, strike: f64| -> Option<Option<f64>> {
        match gong_cost(f1 / fs, strike) {
            Predicted::Cost(s) => (work(fs, s) <= VK_SOLVE_MAX).then_some(Some(s)),
            Predicted::Unmeasured => Some(None),
            Predicted::Broken => None,
        }
    };
    if let Some(p) = passes(fs, strike) {
        return Ok(p);
    }
    let why = match gong_cost(f1 / fs, strike) {
        Predicted::Cost(s) => format!(
            "would need about {:.0}M node-solves with Newton rescuing its steps, over the \
             ~{:.0}M budget",
            work(fs, s) / 1e6,
            VK_SOLVE_MAX / 1e6
        ),
        _ => format!(
            "is past the measured edge where a step stops settling (f₁·k = {:.4}; plan §11.1)",
            f1 / fs
        ),
    };
    let mut fixes = Vec::new();
    let first = (fs / 1000.0).floor() as i64 + 1;
    if let Some(f) = (first..=(VK_FS_MAX / 1000.0) as i64)
        .map(|i| i as f64 * 1000.0)
        .find(|&f| passes(f, strike).is_some())
    {
        fixes.push(format!("raise fs to {} Hz", pf(f)));
    }
    if let Some(&c) = GONG_STRIKE
        .iter()
        .rev()
        .find(|&&c| c < strike && passes(fs, c).is_some())
    {
        fixes.push(format!("strike at most {} thicknesses", pf(c)));
    }
    if fixes.is_empty() {
        fixes.push("lower the pitch, shorten the audio or reduce N".into());
    }
    Err(Refusal::Param(format!(
        "this gong (lowest mode {} Hz at fs = {} Hz, struck {} thicknesses deep) {why}. To fix \
         it, {}.",
        pf(round_nd(f1, 1)),
        pf(fs),
        pf(strike),
        fixes.join(", or ")
    )))
}

/// The pitch (Hz) of a signal from its interpolated zero crossings about its mean; NaN with fewer
/// than three.
fn zc_pitch(sig: &[f64], fs: f64) -> f64 {
    let m = np_mean(sig);
    let t: Vec<f64> = (0..sig.len().saturating_sub(1))
        .filter(|&i| (sig[i] - m).is_sign_negative() != (sig[i + 1] - m).is_sign_negative())
        .map(|i| {
            let (a, b) = (sig[i] - m, sig[i + 1] - m);
            i as f64 + a / (a - b)
        })
        .collect();
    if t.len() < 3 {
        return f64::NAN;
    }
    fs * (t.len() - 1) as f64 / (2.0 * (t[t.len() - 1] - t[0]))
}

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

/// Within this many cents of the settled note, the cymbal is called settled.
const SETTLED_CENTS: f64 = 25.0;

/// Below this share of the strike's shape in the cymbal's lowest mode, its read-out says there is
/// no settled note to read rather than read the leakage. Measured on a 0.35 × 0.27 bronze cymbal:
/// the page's default strike puts 3.2 % there and still reads the settle cleanly (120 ms, −4 cents
/// at a strike of 2 e; −0.25 cents struck gently); a centred strike puts none and read +54 cents.
const MODE1_SHARE_MIN: f64 = 0.01;

/// How far above the settled note a strike starts, and where it ends up (plan §4, §8.8 answer 2).
///
/// The gong reads its pickup over an early and a late window — both read within a few cents of
/// the mode-1 projection in Part 0. The cymbal has no starting pitch (its start is a crash, and the
/// pickup locks onto a higher tone), so it reads the projection onto its lowest elastic mode and
/// reports when that settles onto the note.
fn strike_readout(
    audio: &Run,
    fs: f64,
    f0: f64,
    boundary: Boundary,
    mode1_share: Option<f64>,
) -> Value {
    let win = (0.04 * fs).max(4.0 * fs / f0).round() as usize;
    match boundary {
        Boundary::Supported => {
            let sig = &audio.pickup;
            if sig.len() < 2 * win {
                return json!({
                    "kind": "gong",
                    "message": "cannot be read: the render is too short for a start and an end; \
                                lengthen it.",
                });
            }
            let start = cents(zc_pitch(&sig[..win], fs), f0);
            let end = cents(zc_pitch(&sig[sig.len() - win..], fs), f0);
            let w_ms = 1e3 * win as f64 / fs;
            let message = if !start.is_finite() || !end.is_finite() {
                "cannot be read: too few cycles in a window to read a pitch.".to_owned()
            } else if end.abs() <= 10.0 {
                format!(
                    "starts {start:+.0} cents above the note and has settled onto it ({end:+.1} \
                     cents) by the end."
                )
            } else {
                format!(
                    "starts {start:+.0} cents from the note and is still {end:+.0} cents off it \
                     at the end; lengthen the sound or shorten the ring time to hear it settle."
                )
            };
            json!({
                "kind": "gong",
                "window_ms": num(round_nd(w_ms, 2)),
                "start_cents": num(round_nd(start, 1)),
                "end_cents": num(round_nd(end, 2)),
                "message": message,
            })
        }
        Boundary::Free => {
            let share = mode1_share.unwrap_or(0.0);
            if share < MODE1_SHARE_MIN {
                return json!({
                    "kind": "cymbal",
                    "mode1_share": num(round_nd(share, 4)),
                    "settle_ms": Value::Null,
                    "message": format!(
                        "barely reaches the lowest tone ({:.1} % of the strike's shape — that tone \
                         is a twist whose nodal lines cross the centre), so there is no settled \
                         note to read; move the strike off the centre lines.",
                        100.0 * share
                    ),
                });
            }
            let a = &audio.projection;
            let n_win = a.len() / win.max(1);
            let reads: Vec<f64> = (0..n_win)
                .map(|j| cents(zc_pitch(&a[j * win..(j + 1) * win], fs), f0))
                .collect();
            let ok = |c: &f64| c.is_finite() && c.abs() <= SETTLED_CENTS;
            let settle = (0..n_win).find(|&j| reads[j..].iter().all(ok));
            let w_ms = 1e3 * win as f64 / fs;
            let message = match settle {
                Some(0) => "leaves the lowest tone on the note from the start (a gentle strike)."
                    .to_owned(),
                Some(j) => format!(
                    "makes a crash at first — no single pitch — then the lowest tone settles onto \
                     the note after about {:.0} ms.",
                    j as f64 * w_ms
                ),
                None => format!(
                    "leaves the lowest tone more than {SETTLED_CENTS} cents off the note at the \
                     end; lengthen the sound or strike more gently."
                ),
            };
            json!({
                "kind": "cymbal",
                "mode1_share": num(round_nd(share, 4)),
                "window_ms": num(round_nd(w_ms, 2)),
                "settle_ms": settle.map_or(Value::Null, |j| num(round_nd(j as f64 * w_ms, 1))),
                "end_cents": reads.last().map_or(Value::Null, |&c| num(round_nd(c, 2))),
                "message": message,
            })
        }
    }
}

/// What the coupled solves cost, against a gentle strike at the same `f₁·k` (the table's strike-1
/// column) — a multiple, not seconds: the same work ran 6× apart on one machine (plan §8.5).
fn cost_block(
    audio: &Run,
    steps: usize,
    f0k: f64,
    predicted: Option<f64>,
    nonlinear: bool,
) -> Value {
    let per_step = audio.solves as f64 / steps.max(1) as f64;
    // Past the last row the strike-1 column is unmeasured; its last measured cell is the floor.
    let gentle = match gong_cost(f0k, 1.0) {
        Predicted::Cost(s) => s,
        _ => GONG_SOLVES[GONG_F0K.len() - 1][0].unwrap_or(1.0),
    };
    let multiple = per_step / gentle;
    let message = if !nonlinear {
        "the coupling is off: one linear solve a step.".to_owned()
    } else if audio.rescued == 0 {
        format!(
            "every step settled on the fast method, about {multiple:.1}× a gentle strike's \
             cost."
        )
    } else {
        format!(
            "slow but correct: Newton rescued {} of {steps} steps, about {multiple:.1}× a gentle \
             strike's cost. Raising the sample rate or striking more gently makes it cheaper.",
            audio.rescued
        )
    };
    json!({
        "solves_per_step": num(round_nd(per_step, 2)),
        "rescued_steps": int(audio.rescued as i64),
        "multiple_of_gentle": num(round_nd(multiple, 2)),
        "predicted_multiple": predicted.map_or(Value::Null, |s| num(round_nd(s / gentle, 2))),
        "message": message,
    })
}

/// The `tuning` block of a musical-controls payload: what was asked, what it mapped to, and the
/// ring times of the partials the mapping does not pin, all in the gentle-strike limit.
fn tuning_block(p: &pl::Params, m: &Musical) -> Result<Value, Refusal> {
    let (tu, tg) = (&m.tuning.plate, &m.targets);
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
        "Struck gently, the gong meets the pitch and both ring times exactly; a harder strike \
         starts above the note and glides down onto it."
            .to_owned()
    } else {
        "Struck gently, the cymbal meets the targets to within about 0.05 cents and 0.3 % in ring \
         time (free edges couple the modes slightly); a harder strike is a crash that settles \
         onto them."
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
        "set_by": set_by,
        "f0_target": f0_target,
        "thickness_mm_target": thickness_target,
        "thickness_mm": num(round_nd(tu.thickness * 1e3, 4)),
        "E_gpa": num(round_nd(m.tuning.young / 1e9, 4)),
        "nu": num(m.tuning.nu),
        "rho": num(m.tuning.rho_v),
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
        "partials": partials,
        "message": message,
    }))
}

/// `_build_payload_vk`.
pub fn build_payload(p: &Value) -> Result<Value, Refusal> {
    let audio_dur = fnum(p, "audio_duration", 0.5)?;
    let anim_win = fnum(p, "animation_window", 0.02)?;
    let playback_speed = fnum(p, "playback_speed", 0.02)?;
    let w_over_e = fnum(p, "w_over_e", 2.0)?;
    let pluck_fx = fnum(p, "pluck_x", 0.5)?;
    let pluck_fy = fnum(p, "pluck_y", 0.5)?;
    let pluck_wfrac = fnum(p, "pluck_width", 0.28)?;
    let pickup_fx = fnum(p, "pickup_x", 0.47)?;
    let pickup_fy = fnum(p, "pickup_y", 0.53)?;
    let fpp = int_at_least_one(fnum(p, "frames_per_period", FRAMES_PER_PERIOD)?)?;

    if !(0.0 < audio_dur && audio_dur <= VK_AUDIO_MAX) {
        return Err(Refusal::Param(format!(
            "audio_duration must be in (0, {}] s, got {}.",
            pf(VK_AUDIO_MAX),
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
    if !(0.0 < w_over_e && w_over_e <= VK_WOVERE_MAX) {
        return Err(Refusal::Param(format!(
            "w_over_e must be in (0, {}], got {}.",
            pf(VK_WOVERE_MAX),
            pf(w_over_e)
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

    let (mut vk, fs, boundary, nonlinear, musical) = build(p)?;
    let (lx, ly) = (vk.p.lin.lx, vk.p.lin.ly);

    // Picard-aware budget: worst-case sweeps every step.
    let n_audio = round_int(audio_dur * fs).max(1);
    let n_anim_est = round_int(anim_win * fs).max(1);
    let iters = if nonlinear {
        vk.p.couple_max_iter as i64
    } else {
        1
    };
    let work = vk.p.lin.n_live as i64 * (n_audio + n_anim_est) * iters;
    if work as f64 > VK_WORK_MAX {
        return Err(Refusal::Param(format!(
            "this configuration needs up to ~{:.0}M coupled node-solves (over the ~{:.0}M \
             budget); reduce N, lower fs, or shorten the audio.",
            work as f64 / 1e6,
            VK_WORK_MAX / 1e6
        )));
    }

    // The LINEAR (w -> 0) modes: the markers the hardened peaks sit above.
    let f_lin = discrete_eigenfreqs(&vk.p.lin, N_VK_MODES)?;
    let f1_lin = if f_lin.is_empty() {
        vk.p.lin.kappa / (2.0 * scalar_pow(py_min(&[lx, ly]), 2.0))
    } else {
        f_lin[0]
    };

    // The gong's measured cost (plan §11): refuse what does not settle, and what the table says
    // will pass the budget, before a step is paid for. The cymbal cannot be predicted; it is
    // stopped as it runs (`Guard`).
    let predicted = if nonlinear && boundary == Boundary::Supported {
        gong_guard(f1_lin, fs, w_over_e, vk.p.lin.n_live, audio_dur + anim_win)?
    } else {
        None
    };
    let mut guard = Guard {
        left: VK_SOLVE_MAX,
        boundary,
    };

    // -- the audio run: a strike of w_over_e thicknesses
    let amplitude = w_over_e * vk.p.thickness;
    let wc = pluck_wfrac * py_min(&[lx, ly]);
    let centre = (pluck_fx * lx, pluck_fy * ly);
    let excite = |vk: &mut VkPlate| -> Result<(), Refusal> {
        let field = strike(vk, boundary, amplitude, centre, wc)?;
        let u0 = vk.p.to_live(&field);
        let v0 = vec![0.0; u0.len()];
        vk.set_state(&u0, &v0)
            .map_err(|e| Refusal::Internal(e.to_string()))
    };
    excite(&mut vk)?;
    let pickup_idx = pl::pickup_index_at(pickup_fx * lx, pickup_fy * ly, &vk.p.lin);
    // The cymbal's read-out follows its lowest elastic mode (plan §8.4): the pickup locks onto a
    // higher tone through the whole crash.
    let mode1: Option<Vec<f64>> = match (&musical, boundary) {
        (Some(_), Boundary::Free) => {
            let lin = &vk.p.lin;
            let pairs = free_modes(lin, 4)?;
            Some(
                pairs.vectors[3]
                    .iter()
                    .zip(&lin.w)
                    .map(|(a, b)| a * b)
                    .collect(),
            )
        }
        _ => None,
    };
    // How much of the strike's shape the lowest mode holds (`|φᵀWu₀| / ‖u₀‖_W`, φ W-orthonormal):
    // that mode is the free plate's twist, so a strike on a centre line hardly excites it, and
    // the read-out then follows leakage rather than the note.
    let mode1_share = mode1.as_ref().map(|wv| {
        let a0: f64 = wv.iter().zip(&vk.u).map(|(a, b)| a * b).sum();
        let norm: f64 = vk.u.iter().zip(&vk.p.lin.w).map(|(u, w)| w * u * u).sum();
        a0.abs() / norm.sqrt()
    });
    let audio = run(
        &mut vk,
        n_audio as usize,
        Some(pickup_idx),
        mode1.as_deref(),
        0,
        &mut guard,
    )?;
    if !audio.pickup.iter().all(|v| v.is_finite()) {
        return Err(Refusal::Param(
            "simulation produced non-finite output (instability) — adjust parameters.".into(),
        ));
    }
    // The hardened fundamental: only honest for the gong's clean mode.
    let f0 = if boundary == Boundary::Supported {
        zero_cross_fundamental(&audio.pickup, fs)
    } else {
        f64::NAN
    };

    // -- the animation run: a fresh plate, a fundamental-resolving stride
    let mut anim = build(p)?.0;
    excite(&mut anim)?;
    let mut anim_stride = round_int((fs / f1_lin.max(1.0)) / fpp as f64).max(1);
    let n_anim = anim_stride.max(round_int(anim_win * fs));
    if n_anim / anim_stride > MAX_FRAMES {
        anim_stride = ((n_anim as f64 / MAX_FRAMES as f64).ceil() as i64).max(1);
    }
    let anim_run = run(
        &mut anim,
        n_anim as usize,
        None,
        None,
        anim_stride as usize,
        &mut guard,
    )?;
    let frames_full: Vec<Vec<f64>> = anim_run.snapshots.iter().map(|(_, s)| s.clone()).collect();
    let frame_times: Vec<f64> = anim_run
        .snapshots
        .iter()
        .map(|(i, _)| *i as f64 / fs)
        .collect();
    let lin = &vk.p.lin;
    let (ny, nx) = (lin.mask.nrows(), lin.mask.ncols());
    let (frames_dec, mask_dec, ny_dec, nx_dec) =
        decimate_field_mask(&frames_full, lin.mask.flags(), ny, nx);
    let nf = frames_full.len();
    let field_amp = if frames_dec.is_empty() {
        0.0
    } else {
        max_abs(&frames_dec)
    };
    let (audio48, peak) = resample_normalize(&audio.pickup, fs);
    let of = "the von Karman plate";
    let horizon = if nonlinear {
        horizon_none(HORIZON_NONLINEAR, Some(of))
    } else {
        horizon_block(lin, of)
    };
    let (sigma, sigma1) = (lin.sigma, lin.sigma1);
    let mut payload = json!({
        "model": "vk",
        "horizon": horizon,
        "boundary": if boundary == Boundary::Supported { "supported" } else { "free" },
        "nonlinear": nonlinear,
        "fs_sim": num(round_nd(fs, 3)),
        "grid": {
            "dims": int(2), "nx": int(nx_dec as i64), "ny": int(ny_dec as i64),
            "extent_x": num(round_nd(lx, 6)), "extent_y": num(round_nd(ly, 6)),
            "domain": "rectangle",
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
        // A frequency-dependent loss (musical controls only) decays each mode at its own rate, so
        // the single `2 sigma` oracle line would be a fitted lie there.
        "energy": energy_block(&audio.time, &audio.energy, sigma == 0.0 && sigma1 == 0.0,
                               2.0 * sigma,
                               EnergyOpts {
                                   convergence: nonlinear.then(|| audio.convergence.clone()),
                                   no_decay_oracle: sigma1 > 0.0,
                                   ..EnergyOpts::default()
                               }),
        "meta": {
            "kappa": num(round_nd(lin.kappa, 4)),
            "e": num(vk.p.thickness),
            "f1": num(round_nd(f1_lin, 3)),
            "num_steps": int(n_audio),
            "n_frames": int(nf as i64),
            "spectrum": spectrum_block(&audio.pickup, fs, &f_lin, f0),
        },
    });
    // The musical controls' additions; a physical request's payload is the frozen one.
    if let Some(m) = &musical {
        let f0k = f1_lin / fs;
        payload["controls"] = json!("musical");
        payload["tuning"] = tuning_block(lin, m)?;
        payload["strike_readout"] =
            strike_readout(&audio, fs, m.tuning.plate.f0, boundary, mode1_share);
        payload["cost"] = cost_block(&audio, n_audio as usize, f0k, predicted, nonlinear);
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gong_table_is_read_at_its_conservative_corner_and_rises_on_both_axes() {
        // A request between rows and columns takes the cell above it on both axes, and every cell
        // costs at least its neighbours below and to the left — so the corner is the worst case.
        for r in 0..GONG_F0K.len() {
            for c in 0..GONG_STRIKE.len() {
                let here = GONG_SOLVES[r][c].unwrap_or(f64::INFINITY);
                if r > 0 {
                    assert!(here >= GONG_SOLVES[r - 1][c].unwrap_or(f64::INFINITY));
                }
                if c > 0 {
                    assert!(here >= GONG_SOLVES[r][c - 1].unwrap_or(f64::INFINITY));
                }
            }
        }
        assert_eq!(gong_cost(0.004, 3.0), Predicted::Cost(17.5));
        assert_eq!(gong_cost(0.0001, 0.5), Predicted::Cost(8.0));
        assert_eq!(gong_cost(0.026, 5.0), Predicted::Broken);
        // Past the last row: a column that broke there stays broken; one that settled is no claim.
        assert_eq!(gong_cost(0.06, 6.0), Predicted::Broken);
        assert_eq!(gong_cost(0.06, 5.0), Predicted::Broken);
        assert_eq!(gong_cost(0.06, 4.0), Predicted::Unmeasured);
        assert_eq!(gong_cost(0.06, 1.0), Predicted::Unmeasured);
    }

    fn little_gong(boundary: Boundary) -> VkPlate {
        let spec = VkSpec {
            lx: 0.3,
            ly: 0.3,
            young: 2.0e11,
            thickness: 1.0e-3,
            nu: 0.3,
            rho: 7800.0,
            fs: 32_000.0,
            n: 10,
            boundary: Some(boundary),
            ..VkSpec::default()
        };
        let mut vk = VkPlate::new(VkParams::new(&spec).unwrap());
        let u0 = vec![2.0e-3; vk.p.lin.n_live];
        let field = vk.p.to_full(&u0);
        let u0 = vk.p.to_live(&field);
        vk.set_state(&u0, &vec![0.0; u0.len()]).unwrap();
        vk
    }

    #[test]
    fn the_work_cap_stops_a_run_and_says_so() {
        // The cap is counted, not timed: a budget of a few steps' solves stops the run there, and
        // the same run with the full budget completes.
        for b in [Boundary::Supported, Boundary::Free] {
            let mut vk = little_gong(b);
            let n_live = vk.p.lin.n_live as f64;
            let mut tight = Guard {
                left: 5.0 * n_live,
                boundary: b,
            };
            let err = run(&mut vk, 200, None, None, 0, &mut tight)
                .err()
                .expect("stopped");
            let Refusal::Param(msg) = err else {
                panic!("not a parameter refusal")
            };
            assert!(msg.contains("node-solve budget"), "{msg}");
            let mut vk = little_gong(b);
            let mut full = Guard {
                left: VK_SOLVE_MAX,
                boundary: b,
            };
            let r = run(&mut vk, 200, None, None, 0, &mut full).expect("within budget");
            assert!(r.solves > 200, "the coupled steps cost nothing?");
            assert!(full.left < VK_SOLVE_MAX, "the run was not charged");
        }
    }
}
