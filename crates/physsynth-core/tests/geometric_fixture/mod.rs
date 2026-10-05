//! The geometrically exact string's shared test fixture — the retired Python helper
//! `make_geometric_string` and its initial conditions, used by `string_geometric_harness.rs` and
//! `string_geometric_long.rs` (retirement plan §35). See the harness file's header for the
//! parameters and why they are spelled the way they are.

#![allow(dead_code)]

use physsynth_core::string_geometric::{GeometricString, ParamError, Params};
use std::f64::consts::PI;

/// The acceptance bar, unchanged — see CLAUDE.md.
pub const DRIFT_GATE: f64 = 1e-10;

pub const L: f64 = 1.0;
pub const T: f64 = 200.0;
pub const RHO: f64 = 0.005; // -> c = 200 m/s, f0 = 100 Hz
/// `KAPPA_DEFAULT`.
pub const KAPPA: f64 = 2.0;
/// `THETA_DEFAULT`, written here rather than imported (§29.3).
pub const THETA: f64 = 0.28;
/// `EA_DEFAULT`: `EA/T = 500`, a real steel string's ratio.
pub const EA: f64 = 1.0e5;
/// `GEO_NEWTON_TOL`, which is also the class default.
pub const NEWTON_TOL: f64 = 1e-15;
/// `NEWTON_MAXITER_DEFAULT`.
pub const NEWTON_MAXITER: i64 = 60;
/// `GEO_LAM_LONG_DEFAULT`: the fast field sets the timestep.
pub const LAM_LONG: f64 = 0.5;
/// `TENSION_TOL_DEFAULT`, for the model #9 twin.
pub const TENSION_TOL: f64 = 1e-13;

/// `wave_speed()` — 200 m/s.
pub fn c() -> f64 {
    (T / RHO).sqrt()
}

/// The retired `make_geometric_string`'s keyword arguments, with its defaults.
#[derive(Clone)]
pub struct Geo {
    pub n: i64,
    /// A sample rate given outright, as the Python's guard tests and anchor did.
    pub fs: Option<f64>,
    pub lam: Option<f64>,
    pub lam_long: Option<f64>,
    pub kappa: f64,
    pub kappa_w: Option<f64>,
    pub ea: f64,
    pub sigma0: f64,
    pub sigma1: f64,
    pub sigma0_long: Option<f64>,
    pub sigma1_long: Option<f64>,
    pub theta: f64,
    pub newton_tol: f64,
    pub newton_maxiter: i64,
    pub allow_softening: bool,
}

impl Geo {
    pub fn new(n: i64) -> Self {
        Geo {
            n,
            fs: None,
            lam: None,
            lam_long: None,
            kappa: KAPPA,
            kappa_w: None,
            ea: EA,
            sigma0: 0.0,
            sigma1: 0.0,
            sigma0_long: None,
            sigma1_long: None,
            theta: THETA,
            newton_tol: NEWTON_TOL,
            newton_maxiter: NEWTON_MAXITER,
            allow_softening: false,
        }
    }

    /// The helper's sample rate, spelled in its order: `c N / (L lam)` when the transverse `lam`
    /// is given, else `sqrt(EA/rho) N / (L lam_long)`.
    pub fn fs(&self) -> f64 {
        if let Some(fs) = self.fs {
            return fs;
        }
        match self.lam {
            Some(lam) => c() * self.n as f64 / (L * lam),
            None => {
                (self.ea / RHO).sqrt() * self.n as f64 / (L * self.lam_long.unwrap_or(LAM_LONG))
            }
        }
    }

    pub fn params(&self) -> Result<Params, ParamError> {
        Params::new(
            L,
            T,
            RHO,
            self.fs(),
            self.n,
            self.ea,
            self.kappa,
            self.kappa_w,
            self.sigma0,
            self.sigma1,
            self.sigma0_long,
            self.sigma1_long,
            self.theta,
            true,
            self.newton_tol,
            self.newton_maxiter,
            self.allow_softening,
        )
    }

    pub fn build(&self) -> GeometricString {
        GeometricString::new(self.params().expect("fixture must construct"))
    }
}

/// `GeometricString(L, T, rho, fs=12800, N=32, EA=ea)` with the **class** defaults (`kappa = 0`),
/// which is what the Python's guard tests built — not the helper's.
pub fn class_default(ea: f64) -> Geo {
    Geo {
        fs: Some(12800.0),
        kappa: 0.0,
        ea,
        ..Geo::new(32)
    }
}

/// `geometric_mode_ic`: `amp * sin(m pi x / L)` on the full grid, in the Python's order.
pub fn mode_ic(s: &GeometricString, m: usize, amp: f64) -> Vec<f64> {
    s.p.grid()
        .iter()
        .map(|&x| amp * ((m as f64 * PI) * x / L).sin())
        .collect()
}

/// `geometric_pluck_ic`: a triangle of height `amp` at `x = 0.2`, spelled as the Python spells it.
pub fn pluck_ic(s: &GeometricString, amp: f64) -> Vec<f64> {
    let at = 0.2;
    s.p.grid()
        .iter()
        .map(|&x| {
            amp * if x <= at {
                x / at
            } else {
                (1.0 - x) / (1.0 - at)
            }
        })
        .collect()
}

pub fn zeros(s: &GeometricString) -> Vec<f64> {
    vec![0.0; s.p.nodes()]
}

/// `set_state(u0, w0, v0)` from rest.
pub fn start(s: &mut GeometricString, u0: &[f64], w0: &[f64], v0: &[f64]) {
    let z = zeros(s);
    s.set_state(u0, w0, v0, &[z.clone(), z.clone(), z]);
}

/// `set_state(u0)`: the other two fields at zero.
pub fn start_u(s: &mut GeometricString, u0: &[f64]) {
    let z = zeros(s);
    start(s, u0, &z, &z);
}

pub fn step(s: &mut GeometricString) {
    s.step().expect("the Newton Jacobian must factor");
}

pub fn step_n(s: &mut GeometricString, n: usize) {
    for _ in 0..n {
        step(s);
    }
}

pub fn max_abs(v: &[f64]) -> f64 {
    // NaN-propagating, like `np.max(np.abs(v))` (§29.3).
    v.iter().fold(0.0f64, |m, x| {
        if x.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

pub fn ptp(v: &[f64]) -> f64 {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn rel_drift(e: f64, e0: f64) -> f64 {
    (e - e0).abs() / e0.abs()
}

/// A deterministic standard-normal draw, standing in for the Python's `rng.normal`.
///
/// The bars it feeds are claims about **every** state, so nothing needs NumPy's PCG64 numbers —
/// only their breadth (§15). Box–Muller over a splitmix64 hash rather than a uniform field, so that
/// at the largest strain scale a cell's `1 + v_x` can go negative and the stretch terms' second
/// branch is reached, as the Python's normal draws could.
pub fn normal(seed: u64, i: usize) -> f64 {
    let uniform = |j: u64| {
        let mut z = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(j)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let (u1, u2) = (uniform(2 * i as u64), uniform(2 * i as u64 + 1));
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// `rng.normal(size=(3, N)) * scale`, flattened field-major like the model's strains.
pub fn draw(seed: u64, offset: usize, n: usize, scale: f64) -> Vec<f64> {
    (0..3 * n)
        .map(|i| normal(seed, offset + i) * scale)
        .collect()
}
