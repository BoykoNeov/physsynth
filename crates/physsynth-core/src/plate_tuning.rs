//! Musical controls for the Kirchhoff plates — `docs/dev/plate-mapping-plan.md`.
//!
//! The second piece of `docs/dev/scientific-hurdles.md` §12 after [`crate::string_tuning`]: a
//! musician sets a material, a size and either a thickness or a pitch, plus two ring times, and
//! this module returns what [`crate::plate::PlateSpec`] takes. Like the string's mapping it is an
//! optional helper — the raw constructor is untouched, and realism is offered, never imposed.
//!
//! # Materials
//!
//! [`PRESETS`] lists isotropic metals and glass and orthotropic woods, each with the source its
//! numbers come from. A wood's in-plane axes are the longitudinal (`L`, along the grain) and radial
//! (`R`) ones — a quarter-sawn plate, as instrument tops are — so its four in-plane constants are
//! `E_L`, `E_R`, `G_LR` and `ν_LR`, fed to [`crate::plate::grain_ratios_from_material`].
//!
//! **Every grain ratio is independent of the thickness**: all four rigidities scale as `t³` and
//! the plate takes them as ratios to `D_x`. The thickness moves only `κ` (linearly) and the areal
//! density `ρ_s = ρ t`, so a material and an outline fix the mode *shapes* once — which is what
//! makes the pitch mapping exact (plan §3).
//!
//! **Areal, not volume, density.** A plate takes `ρ_s` (kg/m²). Passing the material's `ρ`
//! (kg/m³) instead leaves every frequency right and every energy wrong by a factor of `t` — the
//! trap [`crate::plate::GrainSpec`] names, which no modal oracle and no energy ledger can see.
//! [`PlateMaterial::rho_s`] is the only density this module hands out.
//!
//! # The mapping: corrected to the scheme
//!
//! [`tune_plate`] meets its targets in the *simulated* plate. Each mode `m` (stiffness eigenvalue
//! `β_m` of `B`, or of `(K, W)`; loss eigenvalue `λ_m` of the loss form) obeys the string's
//! two-step recurrence with `X = k² κ² β_m` and `Y = k (σ₀ + σ₁ λ_m)`:
//!
//! ```text
//! a = 1 + θX + Y,   b = −2 + (1 − 2θ)X,   cc = 1 + θX − Y
//! ```
//!
//! and a target `(f, T60)` is the root pair `r e^{±iφ}`, `φ = 2πfk`, `r = 10^(−3k/T60)`, whose
//! product and sum `cc/a = r²`, `−b/a = 2r cos φ` are linear in `(X, Y)`.
//!
//! **The plate has one stiffness constant, so only one pitch is free.** The string fitted `c²` and
//! `κ²` to two pitches; the plate's whole spectrum moves with `κ` alone. Set by pitch, mode 1's
//! `(f, T60)` fixes its `(X, Y)` by a 2×2 solve, hence `κ`; the anchor's pitch is then whatever the
//! geometry makes it, and only its ring time is a target — one product equation gives its `Y`.
//! Set by thickness, `κ` comes from the material and both `Y`s from the product equations. Either
//! way the two `Y`s give `(σ₀, σ₁)` by a 2×2 solve. No iteration.
//!
//! **What "exact" covers.** On the supported plate the loss form `−L` shares every eigenvector
//! with `B`, so mode 1 and the anchor land on their targets to rounding. On the free plates it does
//! not, and `λ_m` is the mode's Rayleigh quotient — first-order exact. Plan §9 measured the miss
//! through the scheme's own roots: ≤ 0.05 cents in pitch, ≤ 0.3 % in ring time, the worst inside a
//! near-degenerate cluster. [`PlateTuning::exact`] says which a caller has.
//!
//! # The gong and the cymbal
//!
//! [`tune_vk_plate`] is a thin adapter for the von Kármán plate (`docs/dev/vk-mapping-plan.md`):
//! its linear half is this module's plate, so the targets are met in the gentle-strike limit and a
//! harder strike starts above them. It hands back `E`, `ν` and the **volume** density, which is
//! what the gong takes — the areal-density trap above, in the other direction.

use crate::eigs::{eigsh_shift_invert, EigsError};
use crate::plate::{
    grain_ratios_from_material, loss_form, Boundary, Domain, MaterialError, ParamError, Params,
    PlateSpec, VkSpec,
};
use std::f64::consts::{LN_10, PI};

/// A material's elastic constants and density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Material {
    /// The same in every direction.
    Isotropic {
        /// Young's modulus (Pa).
        e: f64,
        /// Poisson's ratio.
        nu: f64,
        /// Volume density (kg/m³).
        rho: f64,
    },
    /// Quarter-sawn wood: the grain (`L`) and the radial direction (`R`) in the plane.
    Wood {
        /// Young's modulus along the grain (Pa).
        e_l: f64,
        /// Young's modulus across it, radially (Pa).
        e_r: f64,
        /// Shear modulus in the `LR` plane (Pa).
        g_lr: f64,
        /// Poisson's ratio `ν_LR`: lateral (`R`) strain from a stress along the grain.
        nu_lr: f64,
        /// Volume density (kg/m³).
        rho: f64,
    },
}

/// Which plate axis a wood's grain runs along. Ignored for isotropic material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainAlong {
    /// Along `x` (the plate's `Lx`; the guitar's width).
    X,
    /// Along `y` (the plate's `Ly`; the guitar's length, where a top's grain runs).
    Y,
}

/// A named material and where its numbers come from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Preset {
    /// The name a caller selects it by.
    pub name: &'static str,
    /// The constants.
    pub material: Material,
    /// The source, in a few words.
    pub source: &'static str,
}

/// `Engineering ToolBox`, the three tables the isotropic presets are read from (accessed
/// 2026-10-18): "Young's Modulus" (`young-modulus-d_417`), "Poisson's Ratio"
/// (`poissons-ratio-d_1224`) and "Density of Solids" (`density-solids-d_1265`). Where the table
/// gives a range, the preset takes a value inside it, and the comment on the preset says which.
const TOOLBOX: &str = "Engineering ToolBox: Young's modulus, Poisson's ratio, density of solids";

/// USDA Forest Products Laboratory, *Wood Handbook* (FPL-GTR-190, 2010), chapter 5: Table 5-1
/// (elastic ratios at ~12 % moisture), Table 5-2 (Poisson's ratios), Table 5-3a (specific gravity
/// and static-bending modulus at 12 %). Two of the handbook's own footnotes are applied:
///
/// - Table 5-1's: `E_L` is the bending modulus **plus 10 %** (the bending test includes shear
///   deflection).
/// - Table 5-3a's: specific gravity is oven-dry weight over volume at 12 % moisture, so the density
///   at 12 % — the wood's water included — is `SG · 1000 · 1.12` kg/m³.
const WOOD_HANDBOOK: &str = "USDA Wood Handbook FPL-GTR-190 (2010), Tables 5-1, 5-2, 5-3a";

/// A handbook wood: bending modulus (MPa) and specific gravity from Table 5-3a, ratios from 5-1
/// and 5-2.
const fn wood(moe_mpa: f64, sg: f64, er_over_el: f64, glr_over_el: f64, nu_lr: f64) -> Material {
    let e_l = moe_mpa * 1.0e6 * 1.1;
    Material::Wood {
        e_l,
        e_r: er_over_el * e_l,
        g_lr: glr_over_el * e_l,
        nu_lr,
        rho: sg * 1000.0 * 1.12,
    }
}

/// The materials a caller can name.
pub const PRESETS: &[Preset] = &[
    // ASTM-A36 structural steel 200 GPa; mild steel ν 0.303; "Steel" 7.82e3.
    Preset {
        name: "steel",
        material: Material::Isotropic {
            e: 200.0e9,
            nu: 0.303,
            rho: 7820.0,
        },
        source: TOOLBOX,
    },
    // Brass 102-125 GPa (110 taken); 70-30 brass ν 0.331; brasses 8.40-8.75e3 (8530 taken).
    Preset {
        name: "brass",
        material: Material::Isotropic {
            e: 110.0e9,
            nu: 0.331,
            rho: 8530.0,
        },
        source: TOOLBOX,
    },
    // Bronze 96-120 GPa (110 taken); ν 0.34; bronzes 8.74-8.89e3 (8800 taken).
    Preset {
        name: "bronze",
        material: Material::Isotropic {
            e: 110.0e9,
            nu: 0.34,
            rho: 8800.0,
        },
        source: TOOLBOX,
    },
    // Aluminum 69 GPa; ν 0.334; 2.7e3.
    Preset {
        name: "aluminium",
        material: Material::Isotropic {
            e: 69.0e9,
            nu: 0.334,
            rho: 2700.0,
        },
        source: TOOLBOX,
    },
    // Glass 50-90 GPa (70 taken); soda glass ν 0.22; common glass 2.4-2.8e3 (2500 taken).
    Preset {
        name: "glass",
        material: Material::Isotropic {
            e: 70.0e9,
            nu: 0.22,
            rho: 2500.0,
        },
        source: TOOLBOX,
    },
    // Sitka spruce: MOE 10,800 MPa, SG 0.40; E_R/E_L 0.078, G_LR/E_L 0.064; ν_LR 0.372.
    Preset {
        name: "sitka spruce",
        material: wood(10_800.0, 0.40, 0.078, 0.064, 0.372),
        source: WOOD_HANDBOOK,
    },
    // Engelmann spruce: MOE 8,900, SG 0.35; 0.128, 0.124; ν_LR 0.422.
    Preset {
        name: "engelmann spruce",
        material: wood(8_900.0, 0.35, 0.128, 0.124, 0.422),
        source: WOOD_HANDBOOK,
    },
    // Western redcedar: MOE 7,700, SG 0.32; 0.081, 0.087; ν_LR 0.378.
    Preset {
        name: "western redcedar",
        material: wood(7_700.0, 0.32, 0.081, 0.087, 0.378),
        source: WOOD_HANDBOOK,
    },
    // Sugar maple: MOE 12,600, SG 0.63; 0.132, 0.111; ν_LR 0.424.
    Preset {
        name: "sugar maple",
        material: wood(12_600.0, 0.63, 0.132, 0.111, 0.424),
        source: WOOD_HANDBOOK,
    },
    // Douglas-fir (coast): MOE 13,400, SG 0.48; 0.068, 0.064; ν_LR 0.292.
    Preset {
        name: "douglas-fir",
        material: wood(13_400.0, 0.48, 0.068, 0.064, 0.292),
        source: WOOD_HANDBOOK,
    },
];

/// The preset called `name`, if there is one.
pub fn preset(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

/// Everything a [`PlateSpec`] needs from a material at one thickness.
///
/// Isotropic material keeps the plate's isotropic path (`nu` set, every grain ratio `1`, no split)
/// so a steel plate built here is the very plate the raw constructor builds from the same `κ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlateMaterial {
    /// `sqrt(D_x / ρ_s)` (m²/s).
    pub kappa: f64,
    /// Areal density `ρ t` (kg/m²) — what the plate's `rho` is.
    pub rho_s: f64,
    /// Poisson's ratio, isotropic only.
    pub nu: Option<f64>,
    /// `D_x / D_x`, which is `1`.
    pub grain_x: f64,
    /// `D_y / D_x`.
    pub grain_y: f64,
    /// `H / D_x`, for the supported branch; `None` isotropically.
    pub grain_cross: Option<f64>,
    /// `D_1 / D_x`, for the free branch; `None` isotropically.
    pub grain_coupling: Option<f64>,
    /// `D_xy / D_x`, for the free branch; `None` isotropically.
    pub grain_torsion: Option<f64>,
}

impl PlateMaterial {
    /// Write the material into `spec` for its boundary: `κ`, `ρ_s`, and the grain in the form that
    /// boundary takes (`grain_cross` supported, the split free — passing both is a contradiction
    /// the plate refuses).
    pub fn apply(&self, spec: &mut PlateSpec) {
        spec.kappa = self.kappa;
        spec.rho = self.rho_s;
        spec.nu = self.nu;
        spec.grain_x = self.grain_x;
        spec.grain_y = self.grain_y;
        let supported = spec.boundary == Some(Boundary::Supported);
        spec.grain_cross = if supported { self.grain_cross } else { None };
        spec.grain_coupling = if supported { None } else { self.grain_coupling };
        spec.grain_torsion = if supported { None } else { self.grain_torsion };
    }
}

/// `material` at `thickness` (m), with a wood's grain along `along`.
///
/// **Turning the grain turns the Poisson ratio.** [`grain_ratios_from_material`] takes the major
/// ratio `ν_xy` (strain along `y` from stress along `x`). With the grain along `y` that is
/// `ν_RL = ν_LR E_R / E_L`, by the reciprocity `ν_LR / E_L = ν_RL / E_R` — not `ν_LR`, which would
/// make a spruce plate's `1 − ν_xy ν_yx` negative and the plate inadmissible.
///
/// # Errors
/// A non-positive constant or thickness, a non-finite one, or an inadmissible Poisson pair.
pub fn plate_material(
    material: &Material,
    along: GrainAlong,
    thickness: f64,
) -> Result<PlateMaterial, MaterialError> {
    if !thickness.is_finite() {
        return Err(MaterialError::NonPositive);
    }
    match *material {
        Material::Isotropic { e, nu, rho } => {
            if !(e > 0.0 && rho > 0.0 && thickness > 0.0) || !e.is_finite() || !rho.is_finite() {
                return Err(MaterialError::NonPositive);
            }
            if !(-1.0 < nu && nu < 0.5) {
                return Err(MaterialError::Inadmissible {
                    den: 1.0 - nu * nu,
                    nu_xy: nu,
                    nu_yx: nu,
                });
            }
            let d = e * thickness * thickness * thickness / (12.0 * (1.0 - nu * nu));
            let rho_s = rho * thickness;
            Ok(PlateMaterial {
                kappa: (d / rho_s).sqrt(),
                rho_s,
                nu: Some(nu),
                grain_x: 1.0,
                grain_y: 1.0,
                grain_cross: None,
                grain_coupling: None,
                grain_torsion: None,
            })
        }
        Material::Wood {
            e_l,
            e_r,
            g_lr,
            nu_lr,
            rho,
        } => {
            if ![e_l, e_r, g_lr, nu_lr, rho].iter().all(|v| v.is_finite()) {
                return Err(MaterialError::NonPositive);
            }
            let g = match along {
                GrainAlong::X => grain_ratios_from_material(e_l, e_r, nu_lr, g_lr, thickness, rho),
                GrainAlong::Y => {
                    grain_ratios_from_material(e_r, e_l, nu_lr * e_r / e_l, g_lr, thickness, rho)
                }
            }?;
            Ok(PlateMaterial {
                kappa: g.kappa,
                rho_s: g.rho_s,
                nu: None,
                grain_x: g.grain_x,
                grain_y: g.grain_y,
                grain_cross: Some(g.grain_cross),
                grain_coupling: Some(g.grain_coupling),
                grain_torsion: Some(g.grain_torsion),
            })
        }
    }
}

/// The thickness (m) at which `material`, grain along `along`, has stiffness coefficient `kappa`.
///
/// `κ` is linear in the thickness (`D ∝ t³`, `ρ_s ∝ t`), so this is `κ / κ(1 m)`.
///
/// # Errors
/// As [`plate_material`], and a non-positive or non-finite `kappa`.
pub fn thickness_for_kappa(
    material: &Material,
    along: GrainAlong,
    kappa: f64,
) -> Result<f64, MaterialError> {
    if !(kappa > 0.0 && kappa.is_finite()) {
        return Err(MaterialError::NonPositive);
    }
    Ok(kappa / plate_material(material, along, 1.0)?.kappa)
}

// -- the mapping ----------------------------------------------------------------------------------

/// Which of thickness and pitch the musician sets; the other is read out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SetBy {
    /// The thickness (m); the fundamental's pitch is read out.
    Thickness(f64),
    /// The fundamental's pitch (Hz); the thickness is worked out.
    Pitch(f64),
}

/// What a musician sets, beyond the material and the outline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Targets {
    /// Thickness or pitch.
    pub set_by: SetBy,
    /// Ring time of the lowest elastic mode: seconds for its amplitude to fall 60 dB.
    pub t60_low: f64,
    /// Ring time of the anchor — the mode nearest `f_ref` — in seconds.
    pub t60_high: f64,
    /// Where "high" is measured (Hz); picks the anchor. Must lie above the fundamental.
    pub f_ref: f64,
}

/// One mode as the mapping sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeData {
    /// Stiffness eigenvalue: of `B` supported, of `(K, W)` free (`κ`-free, `1/m⁴`).
    pub beta: f64,
    /// Loss eigenvalue: of `−L` supported, the Rayleigh quotient `φᵀGφ` free (`1/m²`).
    pub lambda: f64,
}

/// The coefficients that meet the targets, and what a caller shows beside them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlateTuning {
    /// The material at the tuned thickness: `κ`, `ρ_s` and the grain.
    pub material: PlateMaterial,
    /// Thickness (m) — the input when set by thickness, worked out when set by pitch.
    pub thickness: f64,
    /// Frequency-independent loss (1/s).
    pub sigma0: f64,
    /// Frequency-dependent loss (m²/s).
    pub sigma1: f64,
    /// The lowest elastic mode's frequency (Hz) in the scheme, loss included — the target when
    /// set by pitch, the read-out when set by thickness.
    pub f0: f64,
    /// The anchor's frequency (Hz) in the scheme, loss included.
    pub anchor_frequency: f64,
    /// The relative gap from the anchor to the nearest mode at another frequency — a small one
    /// means the anchor sits in a cluster, where the free plates' miss is largest.
    pub anchor_margin: f64,
    /// The lowest elastic mode, as the mapping used it.
    pub fundamental: ModeData,
    /// The anchor.
    pub anchor: ModeData,
    /// `true` on the supported plate (proportional loss: both anchors exact to rounding), `false`
    /// on the free plates (Rayleigh quotient: first-order exact).
    pub exact: bool,
}

impl PlateTuning {
    /// Write the tuned material and both losses into `spec`.
    pub fn apply(&self, spec: &mut PlateSpec) {
        self.material.apply(spec);
        spec.sigma = self.sigma0;
        spec.sigma1 = self.sigma1;
    }
}

/// Why a set of targets was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum TuningError {
    /// A target was not a positive finite number; names it.
    NonPositive(&'static str),
    /// The material, at the thickness asked for or worked out, is not admissible.
    Material(MaterialError),
    /// The plate itself would not build.
    Plate(ParamError),
    /// The modes could not be computed.
    Modes(EigsError),
    /// A free plate's three rigid motions did not separate from its elastic modes.
    RigidModesNotSeparated,
    /// The frequency is beyond what this scheme can sound at this `fs` and `θ`.
    OutOfReach {
        /// What was asked (Hz).
        f: f64,
        /// Which target.
        what: &'static str,
    },
    /// `f_ref` picks the fundamental itself, so one anchor would have to fix two losses.
    AnchorIsTheFundamental {
        /// The reference asked for (Hz).
        f_ref: f64,
        /// The fundamental (Hz).
        f0: f64,
    },
    /// The equations had no solution, or a non-positive stiffness.
    Singular,
    /// The high ring time needs a negative loss. With the loss rising in proportion to the
    /// anchor's loss eigenvalue, `t60_high` is bounded on both sides once `t60_low` is fixed: the
    /// shortest comes with `σ₀ = 0` (all of the fundamental's loss frequency-dependent), the longest
    /// with `σ₁ = 0` (none of it). Outside that, one of the two would have to be negative — no
    /// passive plate.
    HighRingTimeOutOfRange {
        /// The high ring time asked for (s).
        asked: f64,
        /// The shortest these settings allow (s), at `σ₀ = 0`.
        shortest: f64,
        /// The longest (s), at `σ₁ = 0`.
        longest: f64,
    },
    /// A wood was handed to the von Kármán plate, which is isotropic only.
    NotIsotropic,
}

impl std::fmt::Display for TuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuningError::NonPositive(what) => write!(f, "{what} must be positive and finite."),
            TuningError::Material(e) => write!(f, "{e}"),
            TuningError::Plate(e) => write!(f, "{e}"),
            TuningError::Modes(e) => write!(f, "the plate's modes could not be found: {e:?}"),
            TuningError::RigidModesNotSeparated => write!(
                f,
                "this free plate's rigid motions do not separate from its lowest mode; refine N."
            ),
            TuningError::OutOfReach { f: hz, what } => write!(
                f,
                "{what} = {hz} Hz is beyond what this grid and sample rate can sound; raise the \
                 sample rate or lower the target."
            ),
            TuningError::AnchorIsTheFundamental { f_ref, f0 } => write!(
                f,
                "f_ref = {f_ref} Hz picks the fundamental ({f0} Hz) itself; put it above the \
                 fundamental so the high ring time has a partial of its own."
            ),
            TuningError::Singular => write!(f, "these targets have no plate that meets them."),
            TuningError::HighRingTimeOutOfRange {
                asked,
                shortest,
                longest,
            } => write!(
                f,
                "t60_high = {asked} s is out of reach: with this fundamental's ring time the anchor \
                 can ring between {shortest:.4} s and {longest:.4} s (a passive loss that rises \
                 with frequency)."
            ),
            TuningError::NotIsotropic => write!(
                f,
                "the gong and the cymbal are isotropic: pick a metal or glass, not a wood."
            ),
        }
    }
}

impl std::error::Error for TuningError {}

impl From<MaterialError> for TuningError {
    fn from(e: MaterialError) -> Self {
        TuningError::Material(e)
    }
}

/// `r² − 1` for a ring time, through `exp_m1` (`r` is a hair below 1).
fn r2_minus_1(t60: f64, k: f64) -> f64 {
    (-6.0 * LN_10 * k / t60).exp_m1()
}

/// The `X` a lossless mode at `f` needs; `None` beyond the scheme's reach.
fn lossless_x(f: f64, k: f64, theta: f64) -> Option<f64> {
    let s = (PI * f * k).sin().powi(2);
    (f > 0.0 && f * k < 0.5 && 4.0 * theta * s < 1.0).then(|| 4.0 * s / (1.0 - 4.0 * theta * s))
}

/// The lossless frequency (Hz) of a mode with `X`; `None` beyond the scheme's reach.
fn lossless_f(x: f64, k: f64, theta: f64) -> Option<f64> {
    let s = x / (4.0 * (1.0 + theta * x));
    (s < 1.0).then(|| s.sqrt().asin() / (PI * k))
}

/// The frequency (Hz) of the recurrence's root pair at `(X, Y)`; `None` for a non-oscillating one.
pub fn mode_frequency(x: f64, y: f64, k: f64, theta: f64) -> Option<f64> {
    let (a, b, cc) = (
        1.0 + theta * x + y,
        -2.0 + (1.0 - 2.0 * theta) * x,
        1.0 + theta * x - y,
    );
    let disc = 4.0 * a * cc - b * b;
    (disc > 0.0).then(|| disc.sqrt().atan2(-b) / (2.0 * PI * k))
}

/// The ring time (s) of the recurrence's root pair at `(X, Y)`, from `|z|² = cc / a`.
pub fn mode_t60(x: f64, y: f64, k: f64, theta: f64) -> f64 {
    let ln_r2 = (-2.0 * y / (1.0 + theta * x + y)).ln_1p();
    -6.0 * LN_10 * k / ln_r2
}

/// Supported: closed form. `B`'s and `L`'s shared eigenvectors are `sin(pπx/Lx) sin(qπy/Ly)` on
/// the interior nodes, with `d_p = (4/h²) sin²(pπ/2N)` and `e_q` likewise along `y`, so
/// `β = g_x d² + 2 g_c d e + g_y e²` and `λ = d + e` — every mode, no solve. Ascending in `β`.
fn supported_modes(p: &Params) -> Vec<ModeData> {
    let h2 = p.h * p.h;
    let (gx, gc, gy) = if p.grain_is_isotropic {
        (1.0, 1.0, 1.0)
    } else {
        (p.grain_x, p.grain_cross, p.grain_y)
    };
    let mut out = Vec::with_capacity((p.n - 1) * (p.ny - 1));
    for pm in 1..p.n {
        let d = 4.0 / h2 * (pm as f64 * PI / (2.0 * p.n as f64)).sin().powi(2);
        for qm in 1..p.ny {
            let e = 4.0 / h2 * (qm as f64 * PI / (2.0 * p.ny as f64)).sin().powi(2);
            out.push(ModeData {
                beta: gx * d * d + 2.0 * gc * d * e + gy * e * e,
                lambda: d + e,
            });
        }
    }
    out.sort_by(|a, b| a.beta.total_cmp(&b.beta));
    out
}

/// Free: the sparse solve near `shift`, each mode's loss eigenvalue its Rayleigh quotient under
/// the loss form (the vectors come back `W`-orthonormal).
fn free_modes(p: &Params, shift: f64, nev: usize) -> Result<Vec<ModeData>, TuningError> {
    let nev = nev.min(p.n_live - 1);
    let w = p.mass.as_ref().expect("the free branch assembles W");
    let pairs =
        eigsh_shift_invert(&p.stiffness, Some(w), shift, nev).map_err(TuningError::Modes)?;
    let g = loss_form(p);
    Ok(pairs
        .values
        .iter()
        .zip(&pairs.vectors)
        .map(|(&beta, v)| {
            let gv = g.matvec(v);
            ModeData {
                beta,
                lambda: v.iter().zip(&gv).map(|(a, b)| a * b).sum(),
            }
        })
        .collect())
}

/// The rigid motions, skipped by count: a connected free plate has exactly three (`1`, `x`, `y`).
const RIGID_MODES: usize = 3;

/// How many modes around the anchor's stiffness the free plate's solve returns.
const ANCHOR_CANDIDATES: usize = 8;

/// The lowest `count` elastic modes of `p`, ascending — the three rigid motions skipped on the
/// free branch. For a caller that reports what the mapping does to the partials it does not pin:
/// each one's pitch and ring time are [`mode_frequency`] and [`mode_t60`] at
/// `X = k² κ² β`, `Y = k (σ₀ + σ₁ λ)`.
///
/// # Errors
/// [`TuningError::Modes`] when the free plate's solve fails.
pub fn lowest_modes(p: &Params, count: usize) -> Result<Vec<ModeData>, TuningError> {
    match p.boundary {
        Boundary::Supported => Ok(supported_modes(p).into_iter().take(count).collect()),
        Boundary::Free => {
            let shift = -1e-3 * (13.0 / (p.lx * p.ly)).powi(2);
            let all = free_modes(p, shift, RIGID_MODES + count)?;
            Ok(all.into_iter().skip(RIGID_MODES).collect())
        }
    }
}

/// Map `targets` onto the plate `geometry` describes, built of `material` with its grain along
/// `along`.
///
/// `geometry` supplies the outline, size, grid, sample rate and `θ` (`lx, ly, n, fs, theta,
/// boundary, domain, waist, asym`); its stiffness, density, grain and losses are ignored and come
/// back in the result, which [`PlateTuning::apply`] writes into a spec.
///
/// # Errors
/// Any [`TuningError`].
pub fn tune_plate(
    geometry: &PlateSpec,
    material: &Material,
    along: GrainAlong,
    targets: &Targets,
) -> Result<PlateTuning, TuningError> {
    let positive = |v: f64| v > 0.0 && v.is_finite();
    match targets.set_by {
        SetBy::Thickness(t) if !positive(t) => return Err(TuningError::NonPositive("thickness")),
        SetBy::Pitch(f) if !positive(f) => return Err(TuningError::NonPositive("f0")),
        _ => {}
    }
    for (v, what) in [
        (targets.t60_low, "t60_low"),
        (targets.t60_high, "t60_high"),
        (targets.f_ref, "f_ref"),
    ] {
        if !positive(v) {
            return Err(TuningError::NonPositive(what));
        }
    }

    // The modes depend on the grain and the outline, not on the thickness: build the plate at a
    // unit thickness with the losses off, and read them there.
    let unit = plate_material(material, along, 1.0)?;
    let mut spec = geometry.clone();
    unit.apply(&mut spec);
    spec.sigma = 0.0;
    spec.sigma1 = 0.0;
    let p = Params::new(&spec).map_err(TuningError::Plate)?;
    let (k, theta) = (p.k, p.theta);
    let (fund, all) = match p.boundary {
        Boundary::Supported => {
            let all = supported_modes(&p);
            (all[0], Some(all))
        }
        Boundary::Free => {
            let shift = -1e-3 * (13.0 / (p.lx * p.ly)).powi(2);
            let low = free_modes(&p, shift, RIGID_MODES + 1)?;
            let rigid = low[..RIGID_MODES]
                .iter()
                .fold(0.0f64, |m, d| m.max(d.beta.abs()));
            if low.len() <= RIGID_MODES || low[RIGID_MODES].beta <= 1e6 * rigid {
                return Err(TuningError::RigidModesNotSeparated);
            }
            (low[RIGID_MODES], None)
        }
    };

    // The stiffness — from mode 1's (pitch, T60) by a 2x2, or from the material — and mode 1's Y.
    let em1 = r2_minus_1(targets.t60_low, k);
    let y_of = |x: f64, em: f64| -em * (1.0 + theta * x) / (2.0 + em);
    let (x1, y1, thickness) = match targets.set_by {
        SetBy::Pitch(f0) => {
            lossless_x(f0, k, theta).ok_or(TuningError::OutOfReach { f: f0, what: "f0" })?;
            let cs = 2.0 * (-3.0 * LN_10 * k / targets.t60_low).exp() * (2.0 * PI * f0 * k).cos();
            // θ(1 − r²) X − (1 + r²) Y = r² − 1
            // −[(1 − 2θ) + θ 2r cos φ] X − 2r cos φ Y = 2r cos φ − 2
            let (a11, a12, b1) = (-theta * em1, -(2.0 + em1), em1);
            let (a21, a22, b2) = (-((1.0 - 2.0 * theta) + theta * cs), -cs, cs - 2.0);
            let det = a11 * a22 - a12 * a21;
            let x = (b1 * a22 - a12 * b2) / det;
            let y = (a11 * b2 - b1 * a21) / det;
            if !(det != 0.0 && x > 0.0 && x.is_finite() && y.is_finite()) {
                return Err(TuningError::Singular);
            }
            let kappa = (x / (k * k * fund.beta)).sqrt();
            (x, y, thickness_for_kappa(material, along, kappa)?)
        }
        SetBy::Thickness(t) => {
            let kappa = plate_material(material, along, t)?.kappa;
            let x = k * k * kappa * kappa * fund.beta;
            (x, y_of(x, em1), t)
        }
    };
    let kappa2 = x1 / (k * k * fund.beta);
    let f0_lossless = lossless_f(x1, k, theta).ok_or(TuningError::OutOfReach {
        f: f64::INFINITY,
        what: "the fundamental",
    })?;

    // The anchor: nearest f_ref on the LOSSLESS spectrum, so the choice does not depend on the
    // losses it then helps fix.
    let x_ref = lossless_x(targets.f_ref, k, theta).ok_or(TuningError::OutOfReach {
        f: targets.f_ref,
        what: "f_ref",
    })?;
    let near = match all {
        Some(all) => all,
        None => free_modes(&p, x_ref / (k * k * kappa2), ANCHOR_CANDIDATES)?,
    };
    let freq = |m: &ModeData| lossless_f(k * k * kappa2 * m.beta, k, theta);
    let mut best: Option<(ModeData, f64)> = None;
    for m in near.iter().filter(|m| m.beta > fund.beta * (1.0 + 1e-9)) {
        if let Some(fm) = freq(m) {
            if best.is_none_or(|(_, fb)| (fm - targets.f_ref).abs() < (fb - targets.f_ref).abs()) {
                best = Some((*m, fm));
            }
        }
    }
    let (anchor, fa) = best.ok_or(TuningError::OutOfReach {
        f: targets.f_ref,
        what: "f_ref",
    })?;
    if (targets.f_ref - f0_lossless).abs() <= (targets.f_ref - fa).abs() {
        return Err(TuningError::AnchorIsTheFundamental {
            f_ref: targets.f_ref,
            f0: f0_lossless,
        });
    }
    let anchor_margin = near
        .iter()
        .filter_map(freq)
        .filter(|&f| (f - fa).abs() > 1e-9 * fa)
        .fold(f64::INFINITY, |m, f| m.min((f - fa).abs() / fa));

    // The anchor's Y from its product equation; then (σ₀, σ₁) from the two Ys.
    let xm = x1 * anchor.beta / fund.beta;
    let ym = y_of(xm, r2_minus_1(targets.t60_high, k));
    let sigma1 = (ym - y1) / (k * (anchor.lambda - fund.lambda));
    let sigma0 = y1 / k - sigma1 * fund.lambda;
    if !(sigma0.is_finite() && sigma1.is_finite()) {
        return Err(TuningError::Singular);
    }
    if sigma0 < 0.0 || sigma1 < 0.0 {
        // The anchor's Y runs from y1 λ_M/λ₁ (σ₀ = 0) down to y1 (σ₁ = 0).
        return Err(TuningError::HighRingTimeOutOfRange {
            asked: targets.t60_high,
            shortest: mode_t60(xm, y1 * anchor.lambda / fund.lambda, k, theta),
            longest: mode_t60(xm, y1, k, theta),
        });
    }
    Ok(PlateTuning {
        material: plate_material(material, along, thickness)?,
        thickness,
        sigma0,
        sigma1,
        f0: mode_frequency(x1, y1, k, theta).ok_or(TuningError::Singular)?,
        anchor_frequency: mode_frequency(xm, ym, k, theta).ok_or(TuningError::Singular)?,
        anchor_margin,
        fundamental: fund,
        anchor,
        exact: p.boundary == Boundary::Supported,
    })
}

// -- the von Kármán plate -------------------------------------------------------------------------

/// What a [`VkSpec`] needs to meet a set of targets, and the plate mapping behind it.
///
/// The gong takes its material as `E`, `ν` and the **volume** density and derives `κ`, `ρ_s` and
/// the membrane coefficient itself, so this hands back those rather than the plate's `κ` and
/// `ρ_s`: writing [`PlateMaterial::rho_s`] where `rho_v` belongs leaves every pitch right and every
/// energy, and the bend's onset, wrong by the thickness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VkTuning {
    /// The linear plate's mapping: pitch, anchor and both losses, met in the gentle-strike limit.
    pub plate: PlateTuning,
    /// Young's modulus (Pa).
    pub young: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Volume density (kg/m³).
    pub rho_v: f64,
}

impl VkTuning {
    /// Write the material, the thickness and both losses into `spec`.
    pub fn apply(&self, spec: &mut VkSpec) {
        spec.young = self.young;
        spec.nu = self.nu;
        spec.rho = self.rho_v;
        spec.thickness = self.plate.thickness;
        spec.sigma = self.plate.sigma0;
        spec.sigma1 = self.plate.sigma1;
    }
}

/// Map `targets` onto the von Kármán plate `geometry` describes, built of the isotropic
/// `material` (`docs/dev/vk-mapping-plan.md` §3).
///
/// With the coupling off the gong *is* its linear half, so the gentle-strike gong is the plate
/// [`tune_plate`] inverts: this builds the [`PlateSpec`] geometry `VkParams::new` would give that
/// half (`lx, ly, n, fs, θ`, the boundary, a rectangle) and calls it. The pitch and ring times are
/// therefore met as a strike tends to zero — exactly on the supported gong, to first order on the
/// free cymbal ([`PlateTuning::exact`]); a harder strike starts higher and glides down onto them.
///
/// The gong re-derives `κ` from `E e³ / 12(1 − ν²)` in its own spelling, which can differ from
/// [`plate_material`]'s in the last bit; the mapping's exactness is to rounding either way.
///
/// `geometry`'s material, thickness and losses are ignored and come back in the result.
///
/// # Errors
/// [`TuningError::NotIsotropic`] for a wood, and any of [`tune_plate`]'s.
pub fn tune_vk_plate(
    geometry: &VkSpec,
    material: &Material,
    targets: &Targets,
) -> Result<VkTuning, TuningError> {
    let Material::Isotropic { e, nu, rho } = *material else {
        return Err(TuningError::NotIsotropic);
    };
    let plate = PlateSpec {
        lx: geometry.lx,
        ly: geometry.ly,
        fs: geometry.fs,
        n: geometry.n,
        theta: geometry.theta,
        boundary: geometry.boundary,
        domain: Some(Domain::Rectangle),
        ..PlateSpec::default()
    };
    Ok(VkTuning {
        plate: tune_plate(&plate, material, GrainAlong::X, targets)?,
        young: e,
        nu,
        rho_v: rho,
    })
}
