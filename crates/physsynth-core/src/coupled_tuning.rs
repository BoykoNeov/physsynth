//! Musical controls for a string on a bridge: the tension and string loss at which the *coupled*
//! instrument plays a pitch and rings a time (`docs/dev/string-mapping-plan.md` §16).
//!
//! The viewer's coupled pages hang an ideal string, fixed at the nut and free at the far end, on a
//! spring to a body whose modal mass is about the string's own. So the body does not nudge the
//! note, it moves it: 2 to 13 semitones from the string's clamped `c / 2L`, and near a body
//! resonance the string's motion splits between two or three coupled notes (a wolf). The human's
//! answers (plan §16): the pitch slider sets the note the instrument plays, solved through the
//! coupling; in a split band one of the notes is put exactly on the target, even if it is the
//! quieter one, and the louder neighbour is reported; the ring time is a loss on the string, solved
//! so that the played note rings exactly the time set, refused when the body or the air alone
//! already make it die faster.
//!
//! # The method
//!
//! A tension is a clamped fundamental `f_cl = c / 2L`; the search runs over `f_cl`, and over the
//! string's loss `sigma` when there is a ring time.
//!
//! Everything is read off the integrator's own one-step map `Phi` on the state `[x^n, x^{n-1}]`,
//! built column by column from `step()` — as the stability guard's operator is built from
//! `apply_a`, so neither can disagree with what runs. Each root `z` of `Phi` is a damped mode,
//! played at `arg z / 2 pi k` and decaying at `-ln|z| / k`, with every loss in it: the string's,
//! the body's and the air's.
//!
//! 1. **Which notes can reach the target.** Over the `f_cl` for which the page's own pitch band
//!    `(f_cl / 4, 1.3 f_cl)` holds the target, a scan computes `Phi`'s whole spectrum
//!    ([`eig::general_eigenvalues`]) with a lossless string, follows each damped root in the band
//!    from one scan point to the next, and marks every root that crosses the target. Below
//!    `f_cl / 1.3` the string's second mode cannot reach the target, so what crosses it is the
//!    string's fundamental or a body mode.
//! 2. **Which one plays it.** Each crossing is solved for its `f_cl`, and its *string share* — the
//!    fraction of its mass-weighted motion `m |x^n|^2` on string degrees of freedom — is read off
//!    its eigenvector. The largest share wins: outside a split band that is simply the strongest
//!    note; inside one it is the better of the hybrids that can sit on the target (the agent's
//!    reading of the human's answer, plan §16).
//! 3. **The ring time.** A Newton on `(ln f_cl, sigma)` puts both the pitch and the decay on their
//!    targets. The string's loss is solved, not set: the string carries only its share of the note,
//!    so a loss set for the string alone left a 1.5 s target ringing 3.1 s on the body page's
//!    default (plan §16, the human's fourth answer).
//!
//! **Why not the lossless operator.** A first build selected and followed modes by the guard's
//! symmetric operator, whose eigenvectors are cheap. On the radiating body at its default air load
//! that is no guide: the air's rank-1 dashpot reorganizes the body's modes, a note the lossless
//! operator put at 213.6 Hz sounded nowhere near it when plucked, and two lossless modes refined to
//! the same damped root. So the damped spectrum itself is the reference, and a root is followed by
//! continuation: each refinement starts from the last root, moved to the radius the change in the
//! string's loss predicts, never from a lossless guess (plan §16).
//!
//! The stability guard refuses the slackest strings (a heavy spring at a low sample rate), and a
//! refused build is a wall of the search, never an error mid-way.

use std::cell::Cell;
use std::f64::consts::{LN_10, PI};

use crate::body::ModalBody;
use crate::connection::{BridgeBody, StringBodyBridge, SympatheticStrings};
use crate::eig;
use crate::radiation::RadiatedBody;

/// Points of the scan over `f_cl`, across its range of about 2.4 octaves: a step of 5.3 %, small
/// enough that a root moves less between two points than it is far from its neighbours.
const SCAN_POINTS: usize = 32;
/// The pitch's stopping point, relative: 1.7e-8 cents.
///
/// Not tighter, because the damped pitch is not known tighter. One step turns the mode by only
/// `theta = 2 pi f k` (about 0.03 rad on these grids), so a roundoff-sized error in `z` from the
/// one-step map, which is not normal, is a relative error of order 1e-14 to 1e-13 in `f`. A
/// 1e-13 stop sat on that floor and, at `K = 0`, never settled.
const PITCH_TOL: f64 = 1e-11;
/// The Newton's stopping point on the damped decay rate, relative. Looser than the pitch's for
/// the same reason at a smaller scale: the rate is `-ln|z| / k`, and `1 - |z|` is only `a k`
/// (2e-5 for an 8 s ring on these grids), so roundoff in `|z|` is a relative 1e-12 to 1e-11 in it.
const RING_TOL: f64 = 1e-9;
/// Iteration caps: the bracketed solve, the Newton, the inverse iteration.
const BRACKET_ITERS: usize = 200;
const NEWTON_ITERS: usize = 30;
/// Points of the scan of the decay along the string's loss, over about six decades.
const SIGMA_POINTS: usize = 25;
const INVERSE_ITERS: usize = 60;

/// A body whose whole dynamic state is its two modal time levels — a free-air body with no load
/// state of its own. The bare [`ModalBody`] and the resistance-loaded [`RadiatedBody`] qualify;
/// the reactive load (an air mass with its own state) and a room do not, so they cannot be tuned
/// this way.
pub trait LevelBody: BridgeBody<Room = (), Error = std::convert::Infallible> {
    /// The modal body whose levels are the state.
    fn levels_mut(&mut self) -> &mut ModalBody;
}

impl LevelBody for ModalBody {
    fn levels_mut(&mut self) -> &mut ModalBody {
        self
    }
}

impl LevelBody for RadiatedBody {
    fn levels_mut(&mut self) -> &mut ModalBody {
        self.body_mut()
    }
}

/// A coupled string system whose one-step map can be read off its own step.
///
/// The degrees of freedom are the stability guard's: each string's nodes `1..=N` (node 0 is
/// clamped), strings in order, then the body's modes.
pub trait Coupled {
    /// The shared timestep (s).
    fn k(&self) -> f64;
    /// Degrees of freedom: string nodes, then body modes.
    fn n_dofs(&self) -> usize;
    /// How many leading degrees of freedom are string.
    fn n_string_dofs(&self) -> usize;
    /// The diagonal mass `M` in degree-of-freedom order.
    fn masses(&self) -> Vec<f64>;
    /// The first string's loss coefficient (1/s).
    fn string_loss(&self) -> f64;
    /// `[x^n, x^{n-1}]`.
    fn levels(&self) -> Vec<f64>;
    /// Set `[x^n, x^{n-1}]`; clamped nodes are written zero.
    fn set_levels(&mut self, x: &[f64]);
    /// One step of the integrator.
    fn advance(&mut self);
}

impl<B: LevelBody> Coupled for StringBodyBridge<B> {
    fn k(&self) -> f64 {
        self.timestep()
    }
    fn n_dofs(&self) -> usize {
        self.dofs()
    }
    fn n_string_dofs(&self) -> usize {
        self.string().params().n
    }
    fn masses(&self) -> Vec<f64> {
        self.mass_diagonal()
    }
    fn string_loss(&self) -> f64 {
        self.string().params().sigma
    }
    fn levels(&self) -> Vec<f64> {
        let (s, b) = (self.string(), self.body().modal());
        let mut x = s.u[1..].to_vec();
        x.extend_from_slice(b.q());
        x.extend_from_slice(&s.u_prev[1..]);
        x.extend_from_slice(b.q_prev());
        x
    }
    fn set_levels(&mut self, x: &[f64]) {
        let d = self.dofs();
        let n = self.string().params().n;
        let s = self.string_mut();
        s.u[0] = 0.0;
        s.u_prev[0] = 0.0;
        s.u[1..].copy_from_slice(&x[..n]);
        s.u_prev[1..].copy_from_slice(&x[d..d + n]);
        self.body_mut()
            .levels_mut()
            .set_levels(&x[n..d], &x[d + n..]);
    }
    fn advance(&mut self) {
        let Ok(()) = self.step(&mut ());
    }
}

impl Coupled for SympatheticStrings {
    fn k(&self) -> f64 {
        self.timestep()
    }
    fn n_dofs(&self) -> usize {
        self.dofs()
    }
    fn n_string_dofs(&self) -> usize {
        self.strings().iter().map(|s| s.params().n).sum()
    }
    fn masses(&self) -> Vec<f64> {
        self.mass_diagonal()
    }
    fn string_loss(&self) -> f64 {
        self.strings()[0].params().sigma
    }
    fn levels(&self) -> Vec<f64> {
        let mut x = Vec::with_capacity(2 * self.dofs());
        for s in self.strings() {
            x.extend_from_slice(&s.u[1..]);
        }
        x.extend_from_slice(self.body().q());
        for s in self.strings() {
            x.extend_from_slice(&s.u_prev[1..]);
        }
        x.extend_from_slice(self.body().q_prev());
        x
    }
    fn set_levels(&mut self, x: &[f64]) {
        let d = self.dofs();
        let mut off = 0;
        for j in 0..self.count() {
            let s = self.string_mut(j);
            let n = s.params().n;
            s.u[0] = 0.0;
            s.u_prev[0] = 0.0;
            s.u[1..].copy_from_slice(&x[off..off + n]);
            s.u_prev[1..].copy_from_slice(&x[d + off..d + off + n]);
            off += n;
        }
        self.body_mut().set_levels(&x[off..d], &x[d + off..]);
    }
    fn advance(&mut self) {
        self.step();
    }
}

/// The ideal string's loss for a ring time, *alone*: every mode of the explicit scheme decays by
/// `r^2 = (1 - sigma k) / (1 + sigma k)` per step, so a string by itself rings `t60` exactly at
/// `sigma = tanh(a k) / k`, `a = 3 ln 10 / t60`. `None` is lossless. On a bridge the string
/// carries only part of the note, so [`tune_coupled`] solves the loss instead; this is its
/// decoupled limit, and what a bar at `K = 0` compares against.
pub fn string_sigma(t60: Option<f64>, k: f64) -> f64 {
    match t60 {
        None => 0.0,
        Some(t) => (3.0 * LN_10 / t * k).tanh() / k,
    }
}

/// One coupled note.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Note {
    /// Its pitch with every loss (Hz).
    pub f: f64,
    /// Its ring time with every loss (s); infinite when nothing in the system loses energy.
    pub t60: f64,
    /// The fraction of its mass-weighted motion `m |x^n|^2` on the string(s).
    pub share: f64,
}

/// The tension and string loss that put the coupled instrument on a pitch and a ring time, and
/// what it then plays.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledTuning {
    /// The clamped fundamental `c / 2L` of the solution (Hz): the page's tension is
    /// `rho (2 L f_cl)^2`.
    pub f_cl: f64,
    /// The string's loss coefficient at the solution (1/s).
    pub sigma: f64,
    /// The chosen note's root of the one-step map, `(re, im)` — what [`damped_mode`] takes.
    pub root: (f64, f64),
    /// The note put on the target.
    pub note: Note,
    /// The other coupled notes in the page's band `(f_cl / 4, 1.3 f_cl)`, ascending in pitch.
    pub others: Vec<Note>,
}

impl CoupledTuning {
    /// The loudest other note, when one carries more of the string's motion than the target's:
    /// the target sits in a split band.
    pub fn louder_neighbour(&self) -> Option<Note> {
        self.others
            .iter()
            .copied()
            .filter(|o| o.share > self.note.share)
            .fold(None, |best: Option<Note>, o| match best {
                Some(b) if b.share >= o.share => Some(b),
                _ => Some(o),
            })
    }
}

/// Why no tension and loss put the coupled instrument on the targets.
#[derive(Debug, Clone, PartialEq)]
pub enum CoupledError {
    /// The target pitch is not positive and finite.
    BadPitch(f64),
    /// The ring time is not positive (a `None` ring time is the lossless string).
    BadRingTime(f64),
    /// The stability guard refuses every tension that could sound the target.
    Refused,
    /// No coupled note reaches the target at any tension the guard allows.
    NoNote,
    /// The body's and the air's losses alone make the note die faster than the target.
    RingTooLong {
        /// The longest ring time (s) the note can have: the one with a lossless string.
        longest: f64,
    },
    /// Even the heaviest string loss the scheme allows does not make the note die that fast.
    RingTooShort {
        /// The shortest ring time (s) reached, at that loss.
        shortest: f64,
    },
    /// A solve did not settle; names which.
    Unconverged(&'static str),
}

impl std::fmt::Display for CoupledError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoupledError::BadPitch(p) => write!(f, "the pitch must be positive, got {p}."),
            CoupledError::BadRingTime(t) => write!(f, "the ring time must be positive, got {t}."),
            CoupledError::Refused => write!(
                f,
                "the bridge spring is too stiff for this pitch: the coupled system is unstable at \
                 every tension that could sound it. Soften the spring or raise the pitch."
            ),
            CoupledError::NoNote => write!(
                f,
                "no coupled note reaches this pitch at any tension the bridge allows; soften the \
                 spring or raise the pitch."
            ),
            CoupledError::RingTooLong { longest } => write!(
                f,
                "the body and the air alone make this note die within {longest:.3} s, so it \
                 cannot ring longer; lower the ring time or the body's loss."
            ),
            CoupledError::RingTooShort { shortest } => write!(
                f,
                "even the heaviest string loss leaves this note ringing {shortest:.3} s; raise \
                 the ring time."
            ),
            CoupledError::Unconverged(stage) => {
                write!(f, "the coupled tuning did not converge ({stage}).")
            }
        }
    }
}

impl std::error::Error for CoupledError {}

/// A damped root of the one-step map, read as a note.
#[derive(Debug, Clone, Copy)]
struct Root {
    z: C,
    /// Pitch (Hz).
    f: f64,
    /// Decay rate (1/s), `-ln|z| / k`.
    a: f64,
}

impl Root {
    fn of(z: C, k: f64) -> Root {
        Root {
            z,
            f: z.1.atan2(z.0) / (2.0 * PI * k),
            a: -cabs(z).ln() / k,
        }
    }
}

/// Solve for the tension and the string's loss at which the coupled instrument plays `f0` and, with
/// a ring time, rings `t60` (the human's fourth answer, plan §16).
///
/// `build(f_cl, sigma)` builds the page's rig at clamped fundamental `f_cl` with string loss
/// `sigma` and the page's own body and air losses, or `None` when the stability guard refuses it
/// (the guard reads the lossless operator, so `sigma` never decides that). `t60 = None` is a
/// lossless string: only the pitch is solved, and the note rings whatever the body and air give.
///
/// # Errors
/// See [`CoupledError`].
pub fn tune_coupled<R: Coupled>(
    f0: f64,
    t60: Option<f64>,
    build: impl Fn(f64, f64) -> Option<R>,
) -> Result<CoupledTuning, CoupledError> {
    if !(f0 > 0.0 && f0.is_finite()) {
        return Err(CoupledError::BadPitch(f0));
    }
    if let Some(t) = t60 {
        if !(t > 0.0) {
            return Err(CoupledError::BadRingTime(t));
        }
    }
    // The page's band holds f0 when f_cl / 4 < f0 < 1.3 f_cl; a hair inside, so a hit at either
    // end still lies in it.
    let mut lo = f0 / 1.3 * (1.0 + 1e-9);
    let hi = 4.0 * f0 * (1.0 - 1e-9);
    if build(hi, 0.0).is_none() {
        return Err(CoupledError::Refused);
    }
    if build(lo, 0.0).is_none() {
        // The guard relaxes as the tension rises (the timestep shrinks with it): find the wall.
        let (mut a, mut b) = (lo, hi);
        while b / a - 1.0 > 1e-12 {
            let m = (a * b).sqrt();
            if build(m, 0.0).is_some() {
                b = m;
            } else {
                a = m;
            }
        }
        lo = b;
    }

    // 1. The damped roots in the band at every scan point, lossless string.
    let grid: Vec<f64> = (0..SCAN_POINTS)
        .map(|i| lo * (hi / lo).powf(i as f64 / (SCAN_POINTS - 1) as f64))
        .collect();
    let mut scan = Vec::with_capacity(SCAN_POINTS);
    for &x in &grid {
        let mut rig = build(x, 0.0).ok_or(CoupledError::Refused)?;
        scan.push(band_roots(&mut rig, x)?);
    }

    // 2. Every root that crosses f0 between two scan points, solved for its f_cl; keep the
    //    largest string share.
    let mut best: Option<(f64, Root, f64)> = None; // (f_cl, root, share)
    for w in 0..SCAN_POINTS - 1 {
        for (p, q) in follow(&scan[w], &scan[w + 1]) {
            let (ga, gb) = (p.f - f0, q.f - f0);
            if ga != 0.0 && ga.signum() == gb.signum() {
                continue;
            }
            let (xa, xb) = (grid[w], grid[w + 1]);
            // The root at x, found from its straight-line prediction between the two ends.
            let at = |x: f64| -> Result<Root, CoupledError> {
                let t = (x / xa).ln() / (xb / xa).ln();
                let guess = (p.z.0 + t * (q.z.0 - p.z.0), p.z.1 + t * (q.z.1 - p.z.1));
                let mut rig = build(x, 0.0).ok_or(CoupledError::Refused)?;
                nearest_root(&mut rig, guess)
            };
            let x = if ga == 0.0 {
                xa
            } else {
                bracket(xa, xb, |x| at(x).map(|r| r.f - f0))?
            };
            let root = at(x)?;
            let mut rig = build(x, 0.0).ok_or(CoupledError::Refused)?;
            let share = share_of(&mut rig, root.z)?;
            if best.is_none_or(|(_, _, s)| share > s) {
                best = Some((x, root, share));
            }
        }
    }
    let (x_pitch, root, share) = best.ok_or(CoupledError::NoNote)?;

    // 3. The ring time, following the root from where step 2 left it.
    let (x, sigma, z) = match t60 {
        None => (x_pitch, 0.0, root.z),
        Some(t) => pitch_and_ring(f0, 3.0 * LN_10 / t, x_pitch, root, share, &build)?,
    };

    // What the solution plays: every root in the band, the chosen one nearest the followed root.
    let mut rig = build(x, sigma).ok_or(CoupledError::Refused)?;
    let k = rig.k();
    let roots = band_roots(&mut rig, x)?;
    let chosen = (0..roots.len())
        .min_by(|&i, &j| cabs(csub(roots[i].z, z)).total_cmp(&cabs(csub(roots[j].z, z))))
        .ok_or(CoupledError::NoNote)?;
    let mut note = None;
    let mut others = Vec::new();
    for (i, r) in roots.iter().enumerate() {
        // The followed root is the converged one; QR's copy of it agrees to roundoff.
        let z = if i == chosen { z } else { r.z };
        let r = Root::of(z, k);
        let n = Note {
            f: r.f,
            t60: if r.a > 0.0 {
                3.0 * LN_10 / r.a
            } else {
                f64::INFINITY
            },
            share: share_of(&mut rig, z)?,
        };
        if i == chosen {
            note = Some(n);
        } else {
            others.push(n);
        }
    }
    Ok(CoupledTuning {
        f_cl: x,
        sigma,
        root: z,
        note: note.ok_or(CoupledError::NoNote)?,
        others,
    })
}

/// The one-step map's roots in the page's band `(x / 4, 1.3 x)`, upper half-plane, ascending in
/// pitch.
fn band_roots<R: Coupled>(rig: &mut R, x: f64) -> Result<Vec<Root>, CoupledError> {
    let mut roots: Vec<Root> = all_roots(rig)?
        .into_iter()
        .filter(|r| x / 4.0 < r.f && r.f < 1.3 * x)
        .collect();
    roots.sort_by(|a, b| a.f.total_cmp(&b.f));
    Ok(roots)
}

/// Every root of the one-step map in the upper half-plane (one of each conjugate pair).
fn all_roots<R: Coupled>(rig: &mut R) -> Result<Vec<Root>, CoupledError> {
    let k = rig.k();
    let (phi, dim) = one_step_map(rig);
    let ev = eig::general_eigenvalues(&phi, dim)
        .map_err(|_| CoupledError::Unconverged("the one-step map's spectrum"))?;
    Ok(ev
        .into_iter()
        .filter(|z| z.1 > 0.0)
        .map(|z| Root::of(z, k))
        .collect())
}

/// The root of `rig`'s one-step map nearest a predicted one: how a followed note is found at a
/// new tension or loss. The whole spectrum is cheap (9 ms at 208 states, 60 ms at 410) and, unlike
/// an inverse iteration from the prediction, cannot settle on a root further away.
fn nearest_root<R: Coupled>(rig: &mut R, guess: C) -> Result<Root, CoupledError> {
    all_roots(rig)?
        .into_iter()
        .min_by(|a, b| cabs(csub(a.z, guess)).total_cmp(&cabs(csub(b.z, guess))))
        .ok_or(CoupledError::NoNote)
}

/// Pair each root at one scan point with its continuation at the next: nearest first in the
/// `(pitch, decay / 2 pi)` plane, each root used once. A root that leaves or enters the band has
/// no partner and is dropped, which only loses a crossing at the band's own edge.
fn follow(a: &[Root], b: &[Root]) -> Vec<(Root, Root)> {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, p) in a.iter().enumerate() {
        for (j, q) in b.iter().enumerate() {
            let d = (p.f - q.f).hypot((p.a - q.a) / (2.0 * PI));
            pairs.push((d, i, j));
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let (mut used_a, mut used_b) = (vec![false; a.len()], vec![false; b.len()]);
    let mut out = Vec::new();
    for (_, i, j) in pairs {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            out.push((a[i], b[j]));
        }
    }
    out
}

/// The string share of the mode at root `z`: its eigenvector by inverse iteration, then
/// `sum_string m |x^n|^2 / sum m |x^n|^2`.
fn share_of<R: Coupled>(rig: &mut R, z: C) -> Result<f64, CoupledError> {
    let (d, ns, m) = (rig.n_dofs(), rig.n_string_dofs(), rig.masses());
    let (phi, dim) = one_step_map(rig);
    let (_, v) = root_near(&phi, dim, z, None)?;
    let weight = |i: usize| m[i] * (v[i].0 * v[i].0 + v[i].1 * v[i].1);
    let total: f64 = (0..d).map(weight).sum();
    Ok((0..ns).map(weight).sum::<f64>() / total)
}

/// The tension and string loss for the damped pitch `f0` and decay rate `a0`, following the note
/// from the lossless string's root (`root` at `x0`).
///
/// **The decay is not monotone in the string's loss.** A note that is partly body (a hybrid beside
/// a resonance) decays faster as the string's loss rises only until the loss pushes its motion off
/// the string and onto the body; past that it decays *slower*. At 113.1 Hz on the 3000 N/m body
/// (share 0.12) no loss came within a factor of three of a 0.05 s ring, and a Newton that assumed
/// "more loss, faster decay" wandered instead of saying so (plan §16). So the loss is scanned first:
///
/// 1. The decay along `sigma` at `x0`, from zero to the cap `sigma k = 1/2` (past which one step
///    removes a third of the string's motion and the centred loss stops being a damping), the root
///    followed from point to point. At `sigma = 0` it is the least the note can have — the body's
///    and the air's — and a target below it rings longer than the instrument allows. Its largest
///    value is the most the note can have; a target above it rings shorter than the instrument
///    allows.
/// 2. Otherwise the first `sigma` where the decay reaches the target is bracketed and solved, and a
///    Newton on `(ln f_cl, sigma)` with a finite-difference Jacobian corrects the pitch the loss
///    pulled, `sigma` held below the decay's peak.
fn pitch_and_ring<R: Coupled>(
    f0: f64,
    a0: f64,
    x0: f64,
    root: Root,
    share: f64,
    build: &impl Fn(f64, f64) -> Option<R>,
) -> Result<(f64, f64, C), CoupledError> {
    if root.a >= a0 {
        return Err(CoupledError::RingTooLong {
            longest: 3.0 * LN_10 / root.a,
        });
    }
    let k0 = build(x0, 0.0).map(|r| r.k()).ok_or(CoupledError::Refused)?;
    let cap = 0.5 / k0;
    // The last root found and the string loss it was found at: each refinement starts there,
    // moved to the radius the change in the string's loss predicts for this share,
    // `|z| ~ ((1 - sigma k) / (1 + sigma k))^(share / 2)`.
    let last = Cell::new((root.z, 0.0f64));
    let damped_at = |x: f64, sigma: f64| -> Result<Root, CoupledError> {
        let mut rig = build(x, sigma).ok_or(CoupledError::Refused)?;
        let k = rig.k();
        let (z_last, s_last) = last.get();
        let r = |s: f64| {
            let sk = (s * k).min(0.99);
            ((1.0 - sk) / (1.0 + sk)).powf(0.5 * share)
        };
        let scale = r(sigma) / r(s_last);
        let root = nearest_root(&mut rig, (z_last.0 * scale, z_last.1 * scale))?;
        last.set((root.z, sigma));
        Ok(root)
    };

    // 1. The decay along sigma, log-spaced from a thousandth of the loss the share alone would
    //    need up to the cap, the root followed upward.
    let first = (((a0 - root.a) / share.max(1e-3)) * k0).tanh() / k0 * 1e-3;
    let lo = first.min(cap * 1e-6);
    let grid: Vec<f64> = (0..SIGMA_POINTS)
        .map(|i| lo * (cap / lo).powf(i as f64 / (SIGMA_POINTS - 1) as f64))
        .collect();
    let mut decay = Vec::with_capacity(SIGMA_POINTS);
    let mut roots = Vec::with_capacity(SIGMA_POINTS);
    for &s in &grid {
        let r = damped_at(x0, s)?;
        decay.push(r.a);
        roots.push(r.z);
    }
    let peak = (0..SIGMA_POINTS)
        .max_by(|&i, &j| decay[i].total_cmp(&decay[j]))
        .unwrap_or(0);
    if decay[peak] < a0 {
        return Err(CoupledError::RingTooShort {
            shortest: 3.0 * LN_10 / decay[peak],
        });
    }

    // 2. The first crossing below the peak, solved on log sigma with the root followed from the
    //    bracket's lower end, then the Newton.
    let w = (0..=peak)
        .find(|&i| decay[i] >= a0)
        .ok_or(CoupledError::Unconverged("the ring-time scan"))?;
    let (s_lo, z_lo) = if w == 0 {
        (0.0, root.z)
    } else {
        (grid[w - 1], roots[w - 1])
    };
    let s_hi = grid[w];
    let sigma1 = if s_lo == 0.0 {
        s_hi
    } else {
        bracket(s_lo, s_hi, |s| {
            last.set((z_lo, s_lo));
            damped_at(x0, s).map(|r| r.a - a0)
        })?
    };
    last.set((z_lo, s_lo));
    let sigma_ceiling = grid[peak];
    let mut u = [x0.ln(), sigma1];
    let resid = |u: [f64; 2]| -> Result<[f64; 2], CoupledError> {
        let r = damped_at(u[0].exp(), u[1])?;
        Ok([(r.f / f0).ln(), (r.a / a0).ln()])
    };
    let mut r = resid(u)?;
    for _ in 0..NEWTON_ITERS {
        if r[0].abs() <= PITCH_TOL && r[1].abs() <= RING_TOL {
            return Ok((u[0].exp(), u[1], last.get().0));
        }
        let base = last.get();
        let (h0, h1) = (1e-7, 1e-7 * u[1].max(1.0));
        let r0 = resid([u[0] + h0, u[1]])?;
        last.set(base);
        let r1 = resid([u[0], u[1] + h1])?;
        last.set(base);
        let jac = [
            [(r0[0] - r[0]) / h0, (r1[0] - r[0]) / h1],
            [(r0[1] - r[1]) / h0, (r1[1] - r[1]) / h1],
        ];
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        if !(det.abs() > 0.0 && det.is_finite()) {
            return Err(CoupledError::Unconverged("a singular Jacobian"));
        }
        let du = [
            -(jac[1][1] * r[0] - jac[0][1] * r[1]) / det,
            -(-jac[1][0] * r[0] + jac[0][0] * r[1]) / det,
        ];
        // At most a tenth in f_cl per step, and sigma kept inside (0, the decay's peak]: the start
        // is within cents and a hair of the targets, so a larger step is an overshoot.
        let next_sigma = (u[1] + du[1]).clamp(0.5 * u[1], sigma_ceiling);
        u = [u[0] + du[0].clamp(-0.1, 0.1), next_sigma];
        r = resid(u)?;
    }
    Err(CoupledError::Unconverged("the pitch-and-ring Newton"))
}

/// A bracketed root of `g` on log `x`: bisection-safe regula falsi (Illinois).
fn bracket(
    mut a: f64,
    mut b: f64,
    g: impl Fn(f64) -> Result<f64, CoupledError>,
) -> Result<f64, CoupledError> {
    let mut ga = g(a)?;
    let mut gb = g(b)?;
    let mut side = 0i8;
    for _ in 0..BRACKET_ITERS {
        if ga == 0.0 {
            return Ok(a);
        }
        if gb == 0.0 {
            return Ok(b);
        }
        let (la, lb) = (a.ln(), b.ln());
        let mut lm = (la * gb - lb * ga) / (gb - ga);
        if !(lm > la.min(lb) && lm < la.max(lb)) {
            lm = 0.5 * (la + lb);
        }
        let m = lm.exp();
        if m <= a.min(b) || m >= a.max(b) {
            return Ok(m);
        }
        let gm = g(m)?;
        if gm.signum() == gb.signum() {
            b = m;
            gb = gm;
            if side == 1 {
                ga *= 0.5;
            }
            side = 1;
        } else {
            a = m;
            ga = gm;
            if side == -1 {
                gb *= 0.5;
            }
            side = -1;
        }
        if (b / a - 1.0).abs() <= 1e-15 {
            return Ok(if ga.abs() < gb.abs() { a } else { b });
        }
    }
    Err(CoupledError::Unconverged("a bracketed solve"))
}

/// The damped mode of the one-step map nearest `z_guess`: its root `z` (re, im) and eigenvector,
/// the state `[x^n, x^{n-1}]`, complex, scaled to a largest entry of magnitude 1. The rig is left
/// in an arbitrary state.
///
/// Public so that a test can start the integrator on a tuned note's mode and read its pitch and
/// decay off the run, independently of this solver.
///
/// # Errors
/// [`CoupledError::Unconverged`] if the inverse iteration fails.
pub fn damped_mode<R: Coupled>(
    r: &mut R,
    z_guess: (f64, f64),
) -> Result<((f64, f64), Vec<(f64, f64)>), CoupledError> {
    let (phi, dim) = one_step_map(r);
    root_near(&phi, dim, z_guess, None)
}

/// Inverse iteration on `Phi` for the root nearest `guess`, in two rounds: a fixed shift at the
/// guess until the estimate settles, then one at the settled estimate to polish it. Each shift sits
/// a relative 1e-10 inside the guess: a lossless mode's root is exactly on the unit circle, and a
/// shift there made `Phi - shift` exactly singular (a body mode decoupled at `K = 0` did, with
/// exactly zero pivots).
fn root_near(
    phi: &[f64],
    dim: usize,
    guess: C,
    start: Option<Vec<C>>,
) -> Result<(C, Vec<C>), CoupledError> {
    let mut v = start.unwrap_or_else(|| {
        (0..dim)
            .map(|i| {
                (
                    1.0 + 0.5 * (0.7 * i as f64).cos(),
                    0.3 * (1.3 * i as f64).sin(),
                )
            })
            .collect()
    });
    let mut z = guess;
    for _round in 0..2 {
        let shift = (z.0 * (1.0 - 1e-10), z.1 * (1.0 - 1e-10));
        let mut a: Vec<C> = phi.iter().map(|&p| (p, 0.0)).collect();
        for i in 0..dim {
            a[i * dim + i] = csub(a[i * dim + i], shift);
        }
        let piv = clu_factor(&mut a, dim);
        for _ in 0..INVERSE_ITERS {
            let w = clu_solve(&a, &piv, dim, &v);
            if !w.iter().all(|c| c.0.is_finite() && c.1.is_finite()) {
                return Err(CoupledError::Unconverged("a damped-mode solve"));
            }
            // The new estimate: shift + v_p / w_p at the largest entry of w.
            let p = (0..dim)
                .max_by(|&s, &t| cabs(w[s]).total_cmp(&cabs(w[t])))
                .unwrap_or(0);
            let norm = cabs(w[p]);
            if norm == 0.0 {
                return Err(CoupledError::Unconverged("a damped-mode solve"));
            }
            let z_new = cadd(shift, cdiv(v[p], w[p]));
            v = w.iter().map(|c| (c.0 / norm, c.1 / norm)).collect();
            let moved = cabs(csub(z_new, z));
            z = z_new;
            if moved <= 1e-16 {
                break;
            }
        }
    }
    // Settled to roundoff without meeting the bound exactly is still settled; the bars read the
    // mode off a simulation, which is what would catch an iteration that had not.
    Ok((z, v))
}

/// The one-step map `Phi` (row-major, `2d x 2d`): column `i` is the state one step after unit
/// state `e_i`. Built from the integrator's own step; the rig is left in an arbitrary state.
fn one_step_map<R: Coupled>(r: &mut R) -> (Vec<f64>, usize) {
    let dim = 2 * r.n_dofs();
    let mut phi = vec![0.0; dim * dim];
    let mut e = vec![0.0; dim];
    for i in 0..dim {
        e[i] = 1.0;
        r.set_levels(&e);
        r.advance();
        for (row, v) in r.levels().iter().enumerate() {
            phi[row * dim + i] = *v;
        }
        e[i] = 0.0;
    }
    (phi, dim)
}

// -- a complex dense LU, for the shift-invert alone ----------------------------------------------

type C = (f64, f64);

fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}
fn csub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cabs(a: C) -> f64 {
    a.0.hypot(a.1)
}

/// In-place LU with partial pivoting; returns the row permutation.
fn clu_factor(a: &mut [C], n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for col in 0..n {
        let p = (col..n)
            .max_by(|&s, &t| cabs(a[s * n + col]).total_cmp(&cabs(a[t * n + col])))
            .unwrap_or(col);
        if p != col {
            for c in 0..n {
                a.swap(col * n + c, p * n + c);
            }
            perm.swap(col, p);
        }
        let d = a[col * n + col];
        if d == (0.0, 0.0) {
            continue;
        }
        for row in col + 1..n {
            let factor = cdiv(a[row * n + col], d);
            a[row * n + col] = factor;
            if factor != (0.0, 0.0) {
                for c in col + 1..n {
                    let t = cmul(factor, a[col * n + c]);
                    a[row * n + c] = csub(a[row * n + c], t);
                }
            }
        }
    }
    perm
}

fn clu_solve(a: &[C], perm: &[usize], n: usize, b: &[C]) -> Vec<C> {
    let mut x: Vec<C> = perm.iter().map(|&p| b[p]).collect();
    for row in 0..n {
        let mut s = x[row];
        for c in 0..row {
            s = csub(s, cmul(a[row * n + c], x[c]));
        }
        x[row] = s;
    }
    for row in (0..n).rev() {
        let mut s = x[row];
        for c in row + 1..n {
            s = csub(s, cmul(a[row * n + c], x[c]));
        }
        x[row] = cdiv(s, a[row * n + row]);
    }
    x
}
