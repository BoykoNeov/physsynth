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

use crate::plate::{grain_ratios_from_material, Boundary, MaterialError, PlateSpec};

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
