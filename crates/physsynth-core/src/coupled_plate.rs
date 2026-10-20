//! The string-on-plate bridge's notes, read off its modes instead of its one-step map (string
//! mapping plan §18).
//!
//! [`tune_coupled`](crate::coupled_tuning::tune_coupled) reads every note off the integrator's
//! one-step map. On the plate page that map has `2 (N + n_live)` states — 778 at the page's
//! defaults, 3140 on its largest grids — and its whole spectrum took 2.7 s and 37 s, so a tuning,
//! a few hundred spectra, took many minutes. [`ModalPlateBridge`] gives the solver the same roots,
//! exactly to roundoff, in milliseconds.
//!
//! # The scheme as one equation
//!
//! The bridge's step is linear, so a mode `x^n = z^n x` of it solves `P(z) x = 0` with
//!
//! `P(z) = (z - 2 + 1/z) M_theta + k^2 K + k (z - 1/z) C`, where
//!
//! * `M_theta = M_s (+) (M_p + theta k^2 K_p)`: the string's lumped mass, and the plate's with its
//!   implicit θ-scheme on the mass side;
//! * `K = K_s (+) K_p + K_spring (e_end - e_dp)(e_end - e_dp)^T`: the string's `T/h` second
//!   difference with the free end's half row, the plate's stiffness, and the spring, which the
//!   step reads at time `n`;
//! * `C = sigma_s M_s (+) sigma_p M_p`: each part's frequency-independent loss on its own mass. The
//!   string's end node is included: [`connection`](crate::connection) divides the spring's pull by
//!   `1 + sigma k` as the string divides its own update.
//!
//! The string and the plate each diagonalize on their own. The pencils `(K_s, M_s)` and
//! `(K_p, M_p)` are symmetric, and their eigenvectors diagonalize `M_theta` and `C` too, part by
//! part. Neither depends on the tension or the sample rate — the string's eigenvalues scale with
//! `T`, the plate's not at all — so [`PlateModes::of`] computes them once and they serve every
//! tension a search tries. In those coordinates mode `i` is the scalar recurrence
//!
//! `q_i(z) = (mu_i + k gamma_i) z^2 + (k^2 kappa_i - 2 mu_i) z + (mu_i - k gamma_i)`
//!
//! (`mu = 1` on the string and `1 + theta k^2 kappa` on the plate, `kappa` the eigenvalue,
//! `gamma` the part's loss), and the spring couples the modes through rank one:
//!
//! `det P(z) ∝ prod_i q_i(z) * (1 + k^2 K_spring z sum_i v_i^2 / q_i(z))`,
//!
//! `v_i` the mode's amplitude at its end of the spring. Its `2 (N + n_live)` roots are the map's.
//!
//! # The roots: Aberth, after deflation
//!
//! All the roots at once, by Aberth's iteration on that product — `p'/p` costs one pass over the
//! modes, so a sweep is `O(n^2)` — started from each mode's own two roots, nudged off them. It
//! settles in about ten sweeps.
//!
//! Two kinds of mode must be taken out first, as a divide-and-conquer eigensolver deflates them.
//! Without that, two starts converged onto one root and a note was lost: 98 Hz on the free 8 x 8
//! plate, the very note being tuned.
//!
//! * A plate eigenvalue that repeats — a square plate's `(m, n)` and `(n, m)` — acts on the spring
//!   as one mode carrying the group's summed `v^2`; the rest of the group is orthogonal to the
//!   drive point and keeps its own roots.
//! * A mode the spring moves by less than roundoff keeps its own roots: one with a node at the
//!   drive point, and every mode at `K = 0`. Left in, its root is its pole to roundoff, where
//!   `p'/p` divides by `q = 0`.
//!
//! # The string share
//!
//! The modes are mass-orthonormal, so a coupled mode's amplitudes `x_i ∝ v_i z / q_i(z)` give its
//! mass-weighted motion directly: the share is the string modes' `sum |x_i|^2` over all of them —
//! the same weights as the one-step map's share ([`StringPlateBridge::mass_diagonal`]). A root of a
//! deflated mode is that mode alone: share 1 on the string, 0 on the plate.
//!
//! # Checked
//!
//! Against the one-step map's own spectrum, on both edges, lossless and lossy, from the lightest
//! string loss to the scheme's cap, `K` up to 12,000 N/m: every root in the page's band, to about
//! 1e-11 in pitch. On the largest free plate the map's dense spectrum was 4e-10 off on two notes,
//! and an independent solve of `P(z) x = 0` on the sparse matrices sided with this module.

use crate::connection::StringPlateBridge;
use crate::coupled_tuning::{Coupled, CoupledError, WideGrid};
use crate::eig::{self, EigError};
use crate::plate::{Boundary, Plate};
use crate::sparse::Csr;

/// Aberth's sweep cap; it settles in about ten.
const MAX_SWEEPS: usize = 200;
/// A root has settled when Aberth's correction is below this (the roots sit near `|z| = 1`)...
const SETTLED: f64 = 1e-15;
/// ...or when it is below this and has stopped shrinking: a root at roundoff's floor. A near-double
/// root is known only to about the square root of roundoff, and its correction wanders there — a
/// pair at `z = -0.99984`, by the Nyquist, held at 8e-12 for 200 sweeps on the viewer bar's plate
/// rig (`N = 40`, `lambda = 0.85`). A simple root's correction shrinks quadratically to
/// [`SETTLED`] long before this test can fire.
const FLOOR: f64 = 1e-9;
/// Plate eigenvalues within this of each other, relative to the largest, are one repeated
/// eigenvalue. Exact repeats differ by roundoff, about 1e-16 of it.
const REPEATED: f64 = 1e-13;
/// A mode the spring moves by less than this (in `z`, whose roots sit near `|z| = 1`) keeps its
/// own roots: 3e-13 of the pitch at most, against the solver's 1e-11. Left in, such a mode's root
/// is its pole to roundoff, and an iterate converging onto it divides by `q = 0`; at `K = 0` every
/// mode is one. A mode with a node at the drive point moves by about 1e-30.
const UNMOVED: f64 = 1e-14;

/// The plate's wide search ([`WideGrid`]): two passes, both on the body pages' tension steps and
/// with the damped solve — string-loss levels twice as close as the body pages' (a ratio of 1.4778
/// for their 2.1838), then, only when no note is found yet, four times as close (1.216).
///
/// The plate's notes lie a few hertz apart. At a short ring two of them merge and leave the band
/// between two of the body pages' loss levels, so no seed forms; elsewhere the Broyden's full
/// steps land on a neighbouring note and stall. On the probe sweep's 352 ring-time targets the
/// body pages' search refused 24 that a reference search on a 4x-finer grid found, every one
/// confirmed on the one-step map. Either pass alone still refused some (the finer one, two the
/// body pages' search solves: the outcome is not monotone in the grid); the two in turn refused
/// none (plan §18).
pub const PLATE_WIDE: &[WideGrid] = &[
    WideGrid {
        x_step: 1.1068,
        q_step: 1.4778,
        damped: true,
    },
    WideGrid {
        x_step: 1.1068,
        q_step: 1.216,
        damped: true,
    },
];

/// Why [`PlateModes`] cannot describe a bridge.
#[derive(Debug, Clone, PartialEq)]
pub enum PlateModesError {
    /// The plate has the frequency-dependent loss `sigma1`, whose form the plate's modes do not
    /// diagonalize.
    FrequencyDependentLoss,
    /// The free plate's mass is not lumped (diagonal).
    MassNotLumped,
    /// A modal eigenproblem failed.
    Eig(EigError),
    /// The bridge is not the one the modes were computed from: a different string grid, plate or
    /// drive point.
    Mismatch,
}

impl std::fmt::Display for PlateModesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlateModesError::FrequencyDependentLoss => {
                write!(
                    f,
                    "the plate's frequency-dependent loss has no modal form here."
                )
            }
            PlateModesError::MassNotLumped => write!(f, "the plate's mass is not lumped."),
            PlateModesError::Eig(e) => write!(f, "a modal eigenproblem failed: {e:?}."),
            PlateModesError::Mismatch => {
                write!(
                    f,
                    "the bridge is not the one these modes were computed from."
                )
            }
        }
    }
}

impl std::error::Error for PlateModesError {}

/// What the modes depend on: everything spatial. The tension, the losses, `theta` and the
/// timestep are read off each rig.
#[derive(Debug, Clone, PartialEq)]
struct Key {
    n: usize,
    h: f64,
    rho: f64,
    stiffness: Csr,
    masses: Vec<f64>,
    kappa: f64,
    force_denominator: f64,
    drive: usize,
}

/// The string's and the plate's own modes, mass-normalized, each with its amplitude at its end of
/// the spring.
#[derive(Debug, Clone)]
pub struct PlateModes {
    /// Per string mode: its eigenvalue of `K_s / T` against `M_s`, and its amplitude at the free
    /// end.
    string: Vec<(f64, f64)>,
    /// Per plate mode, ascending: its eigenvalue of `K_p` against `M_p`, and its amplitude at the
    /// drive point.
    plate: Vec<(f64, f64)>,
    key: Key,
}

impl PlateModes {
    /// The modes of `bridge`'s string and plate. Any bridge with the same string grid and density,
    /// the same plate and the same drive point can use them, at any tension, timestep and loss.
    ///
    /// # Errors
    /// [`PlateModesError::FrequencyDependentLoss`] or [`PlateModesError::MassNotLumped`] for a
    /// plate this form does not cover; [`PlateModesError::Eig`] if an eigenproblem fails.
    pub fn of(bridge: &StringPlateBridge<Plate>) -> Result<PlateModes, PlateModesError> {
        let key = key_of(bridge);
        let p = &bridge.plate().p;
        if p.loss.is_some() {
            return Err(PlateModesError::FrequencyDependentLoss);
        }
        if p.boundary == Boundary::Free && p.mass.as_ref().is_none_or(|w| w.nnz() != p.n_live) {
            return Err(PlateModesError::MassNotLumped);
        }

        // The string: (K_s / T) = (1/h) D^T D with the free end's half row, against M_s.
        let n = key.n;
        let mut ms = vec![key.rho * key.h; n];
        ms[n - 1] *= 0.5;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            let main = if i + 1 == n { 1.0 } else { 2.0 };
            a[i * n + i] = main / key.h / ms[i];
            if i + 1 < n {
                let off = -1.0 / key.h / (ms[i] * ms[i + 1]).sqrt();
                a[i * n + i + 1] = off;
                a[(i + 1) * n + i] = off;
            }
        }
        let (vals, vecs) = eig::symmetric_eigen(&a, n).map_err(PlateModesError::Eig)?;
        let end = ms[n - 1].sqrt();
        let string = (0..n)
            .map(|j| (vals[j], vecs[(n - 1) * n + j] / end))
            .collect();

        // The plate: K_p = (force denominator) kappa^2 S against its node masses — the step's own
        // scaling, so `M_p + theta k^2 K_p` is what it solves with.
        let m = p.n_live;
        let scale = key.force_denominator * (key.kappa * key.kappa);
        let mp = &key.masses;
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for q in key.stiffness.indptr()[i]..key.stiffness.indptr()[i + 1] {
                let j = key.stiffness.indices()[q];
                a[i * m + j] = scale * key.stiffness.data()[q] / (mp[i] * mp[j]).sqrt();
            }
        }
        let (vals, vecs) = eig::symmetric_eigen(&a, m).map_err(PlateModesError::Eig)?;
        let dp = key.drive;
        let at_drive = mp[dp].sqrt();
        let plate = (0..m)
            .map(|j| (vals[j], vecs[dp * m + j] / at_drive))
            .collect();
        Ok(PlateModes { string, plate, key })
    }
}

fn key_of(bridge: &StringPlateBridge<Plate>) -> Key {
    let sp = bridge.string().params();
    let p = &bridge.plate().p;
    Key {
        n: sp.n,
        h: sp.h,
        rho: sp.rho,
        stiffness: p.stiffness.clone(),
        masses: p.node_masses(),
        kappa: p.kappa,
        force_denominator: p.force_denominator(),
        drive: bridge.drive_index(),
    }
}

/// One mode's scalar recurrence and coupling: `mu`, `kappa`, `gamma`, `v^2` (module header), and
/// which part it belongs to.
#[derive(Debug, Clone, Copy)]
struct Mode {
    mu: f64,
    kappa: f64,
    gamma: f64,
    v2: f64,
    string: bool,
}

type C = (f64, f64);

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cinv(a: C) -> C {
    let d = a.0 * a.0 + a.1 * a.1;
    (a.0 / d, -a.1 / d)
}

fn cabs(a: C) -> f64 {
    a.0.hypot(a.1)
}

impl Mode {
    /// `q(z)` and `q'(z)`. `q` is spelled in `d = z - 1`, `mu d^2 + k gamma d (d + 2) + k^2 kappa
    /// (1 + d)`: the roots sit near `z = 1`, where the plain polynomial loses digits to
    /// cancellation.
    fn q(&self, k: f64, z: C) -> (C, C) {
        let d = (z.0 - 1.0, z.1);
        let d2 = cmul(d, d);
        let kg = k * self.gamma;
        let kk = k * k * self.kappa;
        let val = (
            self.mu * d2.0 + kg * (d2.0 + 2.0 * d.0) + kk * (1.0 + d.0),
            self.mu * d2.1 + kg * (d2.1 + 2.0 * d.1) + kk * d.1,
        );
        let a2 = self.mu + kg;
        let der = (2.0 * a2 * z.0 + kk - 2.0 * self.mu, 2.0 * a2 * z.1);
        (val, der)
    }

    /// The two roots of `q`.
    fn roots(&self, k: f64) -> (C, C) {
        let a2 = self.mu + k * self.gamma;
        let a1 = k * k * self.kappa - 2.0 * self.mu;
        let a0 = self.mu - k * self.gamma;
        let disc = a1 * a1 - 4.0 * a2 * a0;
        if disc < 0.0 {
            let re = -a1 / (2.0 * a2);
            let im = (-disc).sqrt() / (2.0 * a2);
            ((re, im), (re, -im))
        } else {
            let s = disc.sqrt();
            (((-a1 + s) / (2.0 * a2), 0.0), ((-a1 - s) / (2.0 * a2), 0.0))
        }
    }
}

/// A string-on-plate bridge whose notes [`tune_coupled`](crate::coupled_tuning::tune_coupled)
/// reads off [`PlateModes`] (module header). Its integrator is the bridge's own: the map-based
/// methods of [`Coupled`] step it, so [`damped_mode`](crate::coupled_tuning::damped_mode) still
/// starts a run on a solved mode.
#[derive(Debug, Clone)]
pub struct ModalPlateBridge<'m> {
    bridge: StringPlateBridge<Plate>,
    modes: &'m PlateModes,
    /// The modes the spring couples, and the deflated ones (module header).
    coupled: Vec<Mode>,
    deflated: Vec<Mode>,
    /// `k^2 K_spring`.
    spring: f64,
}

impl<'m> ModalPlateBridge<'m> {
    /// `bridge` read through `modes`.
    ///
    /// # Errors
    /// [`PlateModesError::Mismatch`] when `modes` were computed from a different string grid,
    /// plate or drive point; [`PlateModesError::FrequencyDependentLoss`] as in
    /// [`PlateModes::of`].
    pub fn new(
        bridge: StringPlateBridge<Plate>,
        modes: &'m PlateModes,
    ) -> Result<Self, PlateModesError> {
        if key_of(&bridge) != modes.key {
            return Err(PlateModesError::Mismatch);
        }
        let p = &bridge.plate().p;
        if p.loss.is_some() {
            return Err(PlateModesError::FrequencyDependentLoss);
        }
        let sp = bridge.string().params();
        let k = bridge.timestep();
        let spring = k * k * bridge.stiffness();
        let mut candidates: Vec<Mode> = modes
            .string
            .iter()
            .map(|&(w2, v)| Mode {
                mu: 1.0,
                kappa: sp.t * w2,
                gamma: sp.sigma,
                v2: v * v,
                string: true,
            })
            .collect();
        let mut deflated = Vec::new();
        let lmax = modes.plate.iter().map(|m| m.0.abs()).fold(0.0, f64::max);
        let mut i = 0;
        while i < modes.plate.len() {
            let mut j = i + 1;
            while j < modes.plate.len() && modes.plate[j].0 - modes.plate[i].0 <= REPEATED * lmax {
                j += 1;
            }
            let lam = modes.plate[i].0;
            let mode = Mode {
                mu: 1.0 + p.theta * (k * k) * lam,
                kappa: lam,
                gamma: p.sigma,
                v2: modes.plate[i..j].iter().map(|m| m.1 * m.1).sum(),
                string: false,
            };
            deflated.extend(std::iter::repeat_n(mode, j - i - 1));
            candidates.push(mode);
            i = j;
        }
        // A mode the spring moves by less than roundoff keeps its own roots: near its pole the
        // coupling factor is `1 + spring z v^2 / (q'(r) (z - r))`, so the spring shifts the root by
        // about `spring v^2 |r| / |q'(r)|`.
        let (coupled, unmoved): (Vec<Mode>, Vec<Mode>) = candidates.into_iter().partition(|m| {
            let (r, _) = m.roots(k);
            let (_, qd) = m.q(k, r);
            spring * m.v2 * cabs(r) > UNMOVED * cabs(qd)
        });
        deflated.extend(unmoved);
        Ok(ModalPlateBridge {
            bridge,
            modes,
            coupled,
            deflated,
            spring,
        })
    }

    /// The bridge.
    pub fn bridge(&self) -> &StringPlateBridge<Plate> {
        &self.bridge
    }

    /// The bridge, mutably — how a caller plucks it.
    pub fn bridge_mut(&mut self) -> &mut StringPlateBridge<Plate> {
        &mut self.bridge
    }

    /// The bridge, released.
    pub fn into_bridge(self) -> StringPlateBridge<Plate> {
        self.bridge
    }

    /// The modes it reads.
    pub fn modes(&self) -> &'m PlateModes {
        self.modes
    }

    /// For `p = det P = prod q * g`, `g = 1 + spring z sum v^2 / q` the coupling factor: Newton's
    /// step `p / p'`, and `g / g'`.
    ///
    /// The step is spelled `g / (g sum q'/q + g')`, never through `g'/g`: an iterate can land on a
    /// root to the last bit, `g = 0` exactly, and `g'/g` is then `0/0`. It did, at 2656.8 Hz on the
    /// viewer bar's plate rig (`N = 40`, `lambda = 0.85`, the supported 10 x 10 plate).
    fn newton(&self, z: C) -> (C, C) {
        let k = self.bridge.timestep();
        let (mut lq, mut s, mut sd) = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
        for m in &self.coupled {
            let (qv, qd) = m.q(k, z);
            let inv = cinv(qv);
            let t = cmul(qd, inv);
            lq = (lq.0 + t.0, lq.1 + t.1);
            s = (s.0 + m.v2 * inv.0, s.1 + m.v2 * inv.1);
            let u = cmul(qd, cmul(inv, inv));
            sd = (sd.0 - m.v2 * u.0, sd.1 - m.v2 * u.1);
        }
        let zs = cmul(z, s);
        let g = (1.0 + self.spring * zs.0, self.spring * zs.1);
        let zsd = cmul(z, sd);
        let gd = (self.spring * (s.0 + zsd.0), self.spring * (s.1 + zsd.1));
        let lg = cmul(lq, g);
        let step = cmul(g, cinv((lg.0 + gd.0, lg.1 + gd.1)));
        (step, cmul(g, cinv(gd)))
    }
}

impl Coupled for ModalPlateBridge<'_> {
    fn k(&self) -> f64 {
        self.bridge.timestep()
    }
    fn n_dofs(&self) -> usize {
        self.bridge.dofs()
    }
    fn n_string_dofs(&self) -> usize {
        self.bridge.string().params().n
    }
    fn masses(&self) -> Vec<f64> {
        self.bridge.mass_diagonal()
    }
    fn string_loss(&self) -> f64 {
        self.bridge.string().params().sigma
    }
    fn levels(&self) -> Vec<f64> {
        self.bridge.levels()
    }
    fn set_levels(&mut self, x: &[f64]) {
        self.bridge.set_levels(x);
    }
    fn advance(&mut self) {
        self.bridge.advance();
    }

    /// Every root of `det P(z)`: the coupled modes' by Aberth, the deflated modes' their own.
    fn spectrum(&mut self) -> Result<Vec<(f64, f64)>, CoupledError> {
        let k = self.bridge.timestep();
        let mut zs: Vec<C> = Vec::with_capacity(2 * (self.coupled.len() + self.deflated.len()));
        for (i, m) in self.coupled.iter().enumerate() {
            let (r1, r2) = m.roots(k);
            // Off the poles, where p'/p is infinite, and apart, so no two starts coincide.
            let nudge = 1e-6 * (1.0 + i as f64 * 1e-3);
            zs.push((r1.0 * (1.0 + nudge), r1.1 * (1.0 + nudge) + 0.3 * nudge));
            zs.push((r2.0 * (1.0 - nudge), r2.1 * (1.0 + nudge) - 0.3 * nudge));
        }
        let mut settled = vec![false; zs.len()];
        let mut last = vec![f64::INFINITY; zs.len()];
        let mut sweeps = 0;
        while settled.iter().any(|s| !s) {
            if sweeps == MAX_SWEEPS {
                return Err(CoupledError::Unconverged("the plate bridge's modal roots"));
            }
            sweeps += 1;
            for j in 0..zs.len() {
                if settled[j] {
                    continue;
                }
                let z = zs[j];
                let newton = self.newton(z).0;
                let mut pull = (0.0, 0.0);
                for (l, &zl) in zs.iter().enumerate() {
                    if l != j {
                        let t = cinv((z.0 - zl.0, z.1 - zl.1));
                        pull = (pull.0 + t.0, pull.1 + t.1);
                    }
                }
                let den = cmul(newton, pull);
                let w = cmul(newton, cinv((1.0 - den.0, -den.1)));
                zs[j] = (z.0 - w.0, z.1 - w.1);
                if !(zs[j].0.is_finite() && zs[j].1.is_finite()) {
                    return Err(CoupledError::Unconverged("the plate bridge's modal roots"));
                }
                let step = cabs(w);
                settled[j] = step < SETTLED || (step < FLOOR && step > 0.9 * last[j]);
                last[j] = step;
            }
        }
        for m in &self.deflated {
            let (r1, r2) = m.roots(k);
            zs.push(r1);
            zs.push(r2);
        }
        Ok(zs)
    }

    /// The string modes' share of the mode at `z`, from its modal amplitudes (module header); a
    /// root nearer a deflated mode's own than any coupled one is that mode alone: 1 on the string,
    /// 0 on the plate.
    fn string_share(&mut self, z: (f64, f64)) -> Result<f64, CoupledError> {
        let k = self.bridge.timestep();
        let to_coupled = if self.coupled.is_empty() {
            f64::INFINITY
        } else {
            cabs(self.newton(z).1)
        };
        let nearest = self
            .deflated
            .iter()
            .map(|m| {
                let (qv, qd) = m.q(k, z);
                (cabs(qv) / cabs(qd), m.string)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((d, string)) = nearest {
            if d < to_coupled {
                return Ok(if string { 1.0 } else { 0.0 });
            }
        }
        let (mut string, mut total) = (0.0, 0.0);
        for m in &self.coupled {
            let (qv, _) = m.q(k, z);
            let a = m.v2 / (qv.0 * qv.0 + qv.1 * qv.1);
            if m.string {
                string += a;
            }
            total += a;
        }
        Ok(string / total)
    }

    fn wide_grids(&self) -> &'static [WideGrid] {
        PLATE_WIDE
    }
}
