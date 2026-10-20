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
//! 4. **Every tension and loss.** Step 3 follows the note step 2 chose, from where it plays the
//!    pitch with a lossless string. A heavy string loss can put another note on the targets that
//!    no lossless string has on the pitch — at a much tighter string, eight semitones above the
//!    lossless tuning at 269.09 Hz with a 0.05 s ring. So a grid over `(f_cl, sigma k)` is searched
//!    too, and its note displaces step 3's when it has more of the string's motion (the human's
//!    fifth answer, plan §16: the tension may jump, and the search is slower). Checked against a
//!    reference search 64 times as dense on 658 targets with ring times from 0.05 to 20 s: it found
//!    every tuning the reference did. Not below that — at 0.01 and 0.02 s, a ring of one or two
//!    cycles, it missed 7 of 72 tunings the reference found — so a ring time under
//!    [`RING_FLOOR`] is refused (the human's call, plan §16).
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
use crate::connection::{BridgeBody, StringBodyBridge, StringPlateBridge, SympatheticStrings};
use crate::eig;
use crate::plate::Plate;
use crate::radiation::RadiatedBody;

/// Points of the scan over `f_cl`, across its range of about 2.4 octaves: a step of 5.3 %, small
/// enough that a root moves less between two points than it is far from its neighbours — except
/// two near-unison strings' pair of notes, which [`crossings_between`] scans again, finer.
const SCAN_POINTS: usize = 32;
/// A solved crossing is on the pitch when its root plays `f0` to this, relative. The solve stops
/// at about 1e-14; a crossing that joined two different notes misses by their gap, about 2 %.
const ON_PITCH: f64 = 1e-9;
/// Points of each finer scan of one scan step, both ends included: steps of 0.66 %.
const REFINE_POINTS: usize = 9;
/// How many times an interval is scanned again, finer, at most.
const REFINE_DEPTH: usize = 2;
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

/// The shortest ring time (s) [`tune_coupled`] takes. From here to 20 s the wide search found
/// every tuning a reference 64 times as dense found, on 658 targets; at 0.01 and 0.02 s, a ring of
/// one or two cycles, it missed 7 of 72. Below it the solver refuses rather than risk calling a
/// tuning that exists impossible (the human's call, plan §16).
pub const RING_FLOOR: f64 = 0.05;

/// The wide search's grid (plan §16, the human's fifth answer): a step of at most this ratio in
/// `f_cl`, 10.7 %, and string-loss levels `q = sigma k` from [`Q_FLOOR`] up to the scheme's cap
/// `1/2` in steps of [`Q_STEP`], 2.18, plus the lossless string. Eight times coarser along each
/// axis than the reference search it was checked against, which it matched on all 376 targets of
/// the probe sweep.
const X_STEP: f64 = 1.1068;
const Q_FLOOR: f64 = 1e-7;
const Q_STEP: f64 = 2.1838;
/// The wide search's solve: a Broyden on `(ln f_cl, ln q)`. Iteration cap, and how many steps in
/// a row without a new smallest residual end it. 10 is a margin, not a measured edge: the
/// prototype, on one grid shared by every target, lost a tuning at 201.59 Hz and 0.05 s at 6 (24
/// steps from its seed); this search, whose grid starts at each target's guard wall, loses none
/// of the probe sweep's 376 at 6 either.
const WIDE_ITERS: usize = 40;
const WIDE_STALL: usize = 10;
/// Two wide-search solutions are one note when their `f_cl` agree to this, relative, and their
/// string loss to [`SAME_SIGMA`].
const SAME_X: f64 = 1e-6;
const SAME_SIGMA: f64 = 1e-5;
/// A wide-search note displaces the continuation's only when its string share is larger by
/// this, relative: the same note re-found has the same share to roundoff, and must not replace it.
const SHARE_MARGIN: f64 = 1e-6;

/// How the wide search ([`tune_coupled`]'s step 4) covers the `(f_cl, q)` plane and solves its
/// seeds. A rig picks its own passes through [`Coupled::wide_grids`]; every rig but the plate
/// keeps the one pass [`WideGrid::DEFAULT`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WideGrid {
    /// The largest ratio between neighbouring tensions' `f_cl`.
    pub x_step: f64,
    /// The ratio between neighbouring string-loss levels `q = sigma k`.
    pub q_step: f64,
    /// Solve each seed by a damped Newton — a fresh Jacobian every step, each step halved until
    /// the residual falls — instead of the Broyden.
    pub damped: bool,
}

impl WideGrid {
    /// The grid checked on the body pages (`f_cl` ratio 1.1068, `q` ratio 2.1838) and the Broyden
    /// solve.
    pub const DEFAULT: WideGrid = WideGrid {
        x_step: X_STEP,
        q_step: Q_STEP,
        damped: false,
    };
}

/// The damped Newton's halvings of one step before it gives up: a step down to 1/256 of the
/// Newton's.
const WIDE_HALVINGS: usize = 8;

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

    /// Every root of the one-step map, conjugate pairs complete; the solver reads the upper half.
    ///
    /// The default builds the map from [`Coupled::advance`] and computes its whole spectrum with
    /// [`eig::general_eigenvalues`] — what every rig used until the plate: its map is several
    /// hundred states and that took seconds a spectrum, so a rig may compute the same roots another
    /// way ([`crate::coupled_plate`]).
    ///
    /// # Errors
    /// [`CoupledError::Unconverged`] when the spectrum does not settle.
    fn spectrum(&mut self) -> Result<Vec<(f64, f64)>, CoupledError>
    where
        Self: Sized,
    {
        let (phi, dim) = one_step_map(self);
        eig::general_eigenvalues(&phi, dim)
            .map_err(|_| CoupledError::Unconverged("the one-step map's spectrum"))
    }

    /// The string share of the mode at root `z`: `sum_string m |x^n|^2 / sum m |x^n|^2` over its
    /// eigenvector, with [`Coupled::masses`] as the weights. The default finds the eigenvector by
    /// inverse iteration on the one-step map.
    ///
    /// # Errors
    /// [`CoupledError::Unconverged`] when the eigenvector does not settle.
    fn string_share(&mut self, z: (f64, f64)) -> Result<f64, CoupledError>
    where
        Self: Sized,
    {
        let (d, ns, m) = (self.n_dofs(), self.n_string_dofs(), self.masses());
        let (phi, dim) = one_step_map(self);
        let (_, v) = root_near(&phi, dim, z, None)?;
        let weight = |i: usize| m[i] * (v[i].0 * v[i].0 + v[i].1 * v[i].1);
        let total: f64 = (0..d).map(weight).sum();
        Ok((0..ns).map(weight).sum::<f64>() / total)
    }

    /// How the wide search covers this rig's plane: the passes in order, each run only when
    /// neither the continuation nor the passes before found a note; the one pass
    /// [`WideGrid::DEFAULT`] unless the rig says otherwise.
    fn wide_grids(&self) -> &'static [WideGrid] {
        &[WideGrid::DEFAULT]
    }
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

/// The string-on-plate bridge, read the default way: its spectrum from the one-step map. Seconds
/// a spectrum at the plate page's grids, so the page tunes through [`ModalPlateBridge`], which
/// computes the same roots from the modes; this is the reference the bars hold it to.
///
/// [`ModalPlateBridge`]: crate::coupled_plate::ModalPlateBridge
impl Coupled for StringPlateBridge<Plate> {
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
        let (s, p) = (self.string(), self.plate());
        let mut x = s.u[1..].to_vec();
        x.extend_from_slice(&p.u);
        x.extend_from_slice(&s.u_prev[1..]);
        x.extend_from_slice(&p.u_prev);
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
        // The plate's step reads only `u` and `u_prev`; `accel` is a read-out of the last step.
        let p = self.plate_mut();
        p.u.copy_from_slice(&x[n..d]);
        p.u_prev.copy_from_slice(&x[d + n..]);
    }
    fn advance(&mut self) {
        let Ok(()) = self.step(&mut ());
    }
    /// The plate's own, as its modal twin's: the slow reference searches the same way.
    fn wide_grids(&self) -> &'static [WideGrid] {
        crate::coupled_plate::PLATE_WIDE
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
    /// The ring time is below [`RING_FLOOR`], where the search is not known to find every
    /// tuning.
    RingBelowFloor(f64),
    /// The stability guard refuses every tension that could sound the target.
    Refused,
    /// No coupled note reaches the target at any tension the guard allows — with a ring time, at
    /// any string loss either.
    NoNote,
    /// The note cannot ring that long: with a lossless string, the body's and the air's losses
    /// alone end it sooner; or no lossless string has a note on the pitch, and the string loss
    /// that brings one there ends it sooner.
    RingTooLong {
        /// The longest ring time (s): the lossless string's note's, exact — the note with the most
        /// string motion, so a quieter note at the pitch, if there is one, is not covered by it;
        /// or, when only a lossy string reaches the pitch, the longest the wide search saw there,
        /// each extreme solved onto the pitch.
        longest: f64,
    },
    /// The note cannot ring that short: the heaviest string loss the scheme allows does not make
    /// it die that fast, and no other tension and loss put a faster note on the targets.
    RingTooShort {
        /// The shortest ring time (s): the lossless string's note's, at the heaviest loss, exact;
        /// or, when only a lossy string reaches the pitch, the shortest the wide search saw
        /// there, each extreme solved onto the pitch.
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
            CoupledError::RingBelowFloor(t) => write!(
                f,
                "the ring time must be at least {RING_FLOOR} s, got {t}: below that the search \
                 is not known to find every tuning."
            ),
            CoupledError::Refused => write!(
                f,
                "the bridge spring is too stiff for this pitch: the coupled system is unstable at \
                 every tension that could sound it. Soften the spring or raise the pitch."
            ),
            CoupledError::NoNote => write!(
                f,
                "no coupled note reaches this pitch at any tension the bridge allows, nor at any \
                 string loss; soften the spring or raise the pitch."
            ),
            CoupledError::RingTooLong { longest } => write!(
                f,
                "the search found no tension and string loss giving a note at this pitch that \
                 rings this long: the note with the most string motion here cannot ring longer \
                 than {longest:.3} s (the body's and the air's own losses, or the string loss it \
                 takes to bring it to this pitch, end it sooner); lower the ring time or the \
                 body's loss."
            ),
            CoupledError::RingTooShort { shortest } => write!(
                f,
                "the search found no tension and string loss giving a note at this pitch that \
                 dies this fast: the note with the most string motion here rings at least \
                 {shortest:.3} s even at the heaviest string loss the scheme allows; raise the \
                 ring time."
            ),
            CoupledError::Unconverged(stage) => {
                write!(
                    f,
                    "the coupled tuning did not converge ({stage}); a tuning may still exist \
                     here, so a pitch or ring time a little different may find one."
                )
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
        // Negated so that a NaN ring time, which compares false against everything, is refused.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(t > 0.0) {
            return Err(CoupledError::BadRingTime(t));
        }
        if t < RING_FLOOR {
            return Err(CoupledError::RingBelowFloor(t));
        }
    }
    // The page's band holds f0 when f_cl / 4 < f0 < 1.3 f_cl; a hair inside, so a hit at either
    // end still lies in it.
    let mut lo = f0 / 1.3 * (1.0 + 1e-9);
    let hi = 4.0 * f0 * (1.0 - 1e-9);
    let wide_grids = build(hi, 0.0).ok_or(CoupledError::Refused)?.wide_grids();
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
        let ends = ((grid[w], &scan[w][..]), (grid[w + 1], &scan[w + 1][..]));
        for (x, root) in crossings_between(f0, ends.0, ends.1, REFINE_DEPTH, &build)? {
            let mut rig = build(x, 0.0).ok_or(CoupledError::Refused)?;
            let share = share_of(&mut rig, root.z)?;
            if best.is_none_or(|(_, _, s)| share > s) {
                best = Some((x, root, share));
            }
        }
    }
    let (x, sigma, z) = match t60 {
        None => {
            let (x_pitch, root, _) = best.ok_or(CoupledError::NoNote)?;
            (x_pitch, 0.0, root.z)
        }
        Some(t) => {
            let a0 = 3.0 * LN_10 / t;
            // 3. The ring time, following the root from where step 2 left it, and the share the
            //    note has there.
            let narrow = best
                .ok_or(CoupledError::NoNote)
                .and_then(|(x_pitch, root, share)| {
                    let (x, sigma, z) = pitch_and_ring(f0, a0, x_pitch, root, share, &build)?;
                    let mut rig = build(x, sigma).ok_or(CoupledError::Refused)?;
                    Ok((x, sigma, z, share_of(&mut rig, z)?))
                });
            // 4. The wide search, for a note of more share than that (the human's fifth answer):
            //    the rig's passes in order, the next only while no note is found at all.
            let top = narrow.as_ref().map_or(f64::INFINITY, |n| 2.0 * a0 / n.3);
            let mut better = None;
            let mut crossings = Vec::new();
            for &grid in wide_grids {
                let wide = wide_search(f0, a0, lo, hi, top, grid, &build);
                better = wide
                    .notes
                    .into_iter()
                    .filter(|w| match &narrow {
                        Ok(n) => {
                            let same = (w.x / n.0 - 1.0).abs() < SAME_X
                                && (w.sigma / n.1 - 1.0).abs() < SAME_SIGMA;
                            !same && w.share > n.3 * (1.0 + SHARE_MARGIN)
                        }
                        Err(_) => true,
                    })
                    .max_by(|p, q| p.share.total_cmp(&q.share));
                crossings.extend(wide.crossings);
                if better.is_some() || narrow.is_ok() {
                    break;
                }
            }
            match (better, narrow) {
                (Some(w), _) => (w.x, w.sigma, w.z),
                (None, Ok((x, sigma, z, _))) => (x, sigma, z),
                (None, Err(e)) => {
                    return Err(ring_refusal(e, f0, a0, &crossings, &build));
                }
            }
        }
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

/// The crossings of `f0` between two scan points, each given as `(f_cl, its band roots)`: every
/// paired root ([`follow`]) whose pitch changes side of `f0`, solved for its `f_cl`.
///
/// Pairing by pitch can swap two notes that sit closer together than they move in one step — two
/// strings a fifth of a semitone apart on the two-string page, whose notes were 4.5 Hz apart and
/// moved 5 Hz a step (plan §17) — and a "crossing" then joins the lower note at one end to the
/// upper at the other. Its solve converges on the tension where the nearest root jumps from one
/// note to the other: a tuning that is not on the pitch, which the solver returned as one (98 Hz
/// asked, 99.65 Hz played). So each solution is checked against the pitch ([`ON_PITCH`]), and an
/// interval with one that misses is scanned again at [`REFINE_POINTS`] points, up to
/// [`REFINE_DEPTH`] times, its crossings taken from there; a miss at the last depth is dropped.
/// A swap of two notes that cross the pitch in the same direction always leaves a miss; a swap of
/// two crossing in opposite directions would leave none, and was not seen: the notes rise with the
/// tension.
fn crossings_between<R: Coupled>(
    f0: f64,
    (xa, ra): (f64, &[Root]),
    (xb, rb): (f64, &[Root]),
    depth: usize,
    build: &impl Fn(f64, f64) -> Option<R>,
) -> Result<Vec<(f64, Root)>, CoupledError> {
    let mut out = Vec::new();
    let mut missed = false;
    for (p, q) in follow(ra, rb) {
        let (ga, gb) = (p.f - f0, q.f - f0);
        if ga != 0.0 && ga.signum() == gb.signum() {
            continue;
        }
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
        if (root.f / f0 - 1.0).abs() <= ON_PITCH {
            out.push((x, root));
        } else {
            missed = true;
        }
    }
    if !missed || depth == 0 {
        return Ok(out);
    }
    let xs: Vec<f64> = (0..REFINE_POINTS)
        .map(|i| xa * (xb / xa).powf(i as f64 / (REFINE_POINTS - 1) as f64))
        .collect();
    let mut scan = Vec::with_capacity(REFINE_POINTS);
    for &x in &xs {
        let mut rig = build(x, 0.0).ok_or(CoupledError::Refused)?;
        scan.push(band_roots(&mut rig, x)?);
    }
    let mut out = Vec::new();
    for w in 0..REFINE_POINTS - 1 {
        out.extend(crossings_between(
            f0,
            (xs[w], &scan[w]),
            (xs[w + 1], &scan[w + 1]),
            depth - 1,
            build,
        )?);
    }
    Ok(out)
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
    let ev = rig.spectrum()?;
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

/// The string share of the mode at root `z` ([`Coupled::string_share`]).
fn share_of<R: Coupled>(rig: &mut R, z: C) -> Result<f64, CoupledError> {
    rig.string_share(z)
}

/// The tension and string loss for the damped pitch `f0` and decay rate `a0`, following the note
/// from the lossless string's root (`root` at `x0`).
///
/// **The decay is not monotone in the string's loss.** A note that is partly body (a hybrid beside
/// a resonance) decays faster as the string's loss rises only until the loss pushes its motion off
/// the string and onto the body; past that it decays *slower*. At 113.1 Hz on the 3000 N/m body
/// (share 0.12) no loss came within a factor of three of a 0.05 s ring for this note, and a Newton
/// that assumed "more loss, faster decay" wandered instead of saying so (plan §16); another note,
/// at a much tighter string, rings it, and [`wide_search`] finds that one. So the loss is scanned
/// first:
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
        // A refusal here is the Newton stepping past the guard's wall, not every tension refused:
        // reported as Refused, it told the plate page to soften a spring that tunes (plan §18).
        let mut rig =
            build(x, sigma).ok_or(CoupledError::Unconverged("the pitch-and-ring Newton"))?;
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

/// A note the wide search put on the targets.
#[derive(Debug, Clone, Copy)]
struct WideNote {
    x: f64,
    sigma: f64,
    z: C,
    share: f64,
}

/// Where a root crosses the pitch along an edge of the wide search's grid: the edge's ends
/// `(f_cl, q)`, the root at each, and the decay read off the straight line between them.
#[derive(Debug, Clone, Copy)]
struct Crossing {
    from: (f64, f64),
    to: (f64, f64),
    r: Root,
    m: Root,
    estimate: f64,
}

/// What the wide search found: the notes on the targets, and where it saw a root cross the pitch
/// with a lossy string, for the refusal when there are none.
struct Wide {
    notes: Vec<WideNote>,
    crossings: Vec<Crossing>,
}

/// The root of `v` nearest `(f, a)` in the `(pitch, decay / 2 pi)` plane — [`follow`]'s
/// measure.
fn nearest_note(v: &[Root], f: f64, a: f64) -> Option<Root> {
    let d = |r: &Root| (r.f - f).hypot((r.a - a) / (2.0 * PI));
    v.iter().copied().min_by(|p, q| d(p).total_cmp(&d(q)))
}

/// Every note that plays `f0` and decays at `a0` at some tension in `[lo, hi]` and string loss up
/// to `sigma_top` (the human's fifth answer, plan §16).
///
/// The continuation of [`pitch_and_ring`] follows a note from where it reaches `f0` with a
/// lossless string. Some notes never do: past a loss threshold a *pair* of notes appears on `f0`
/// at a much tighter string (at 269.09 Hz on the 3000 N/m body, a 0.05 s ring has a note of share
/// 0.54 at `f_cl = 425.9`, 8 semitones above the lossless tuning). So the grid covers the plane:
/// `f_cl` from `lo` to `hi` in steps of at most `grid.x_step`, and `q = sigma k` at zero and from
/// [`Q_FLOOR`] up to the cap in steps of `grid.q_step` ([`WideGrid`]; [`X_STEP`] and [`Q_STEP`]
/// unless the rig asks for finer). A cell is a seed when one root, matched to its nearest at the
/// other three corners, has its pitch error and its decay error each change sign across them. Each
/// seed is solved by [`wide_solve`] ([`wide_solve_damped`] when the grid asks); a seed whose cell,
/// widened by one cell each way, already holds a solution is skipped.
///
/// `sigma_top` prunes: a note's decay is at least the string's loss times its string share (the
/// string's loss acts only on the string's part of the motion, and the body and the air only add
/// to it; on every tuning of the probe sweep `sigma share / a0 <= 1.002`). So a note of more
/// share than one already found needs less loss than `a0 / share`, and the caller passes twice
/// that. A refused build or a failed spectrum is a hole in the grid, never an error.
fn wide_search<R: Coupled>(
    f0: f64,
    a0: f64,
    lo: f64,
    hi: f64,
    sigma_top: f64,
    grid: WideGrid,
    build: &impl Fn(f64, f64) -> Option<R>,
) -> Wide {
    let nx = (((hi / lo).ln() / grid.x_step.ln()).ceil() as usize).max(1) + 1;
    let xs: Vec<f64> = (0..nx)
        .map(|i| lo * (hi / lo).powf(i as f64 / (nx - 1) as f64))
        .collect();
    let mut qs = vec![0.0];
    let mut q = Q_FLOOR;
    while q <= 0.5 {
        qs.push(q);
        q *= grid.q_step;
    }
    let nq = qs.len();
    // The timestep does not depend on the loss: one lossless build per column.
    let ks: Vec<Option<f64>> = xs.iter().map(|&x| build(x, 0.0).map(|r| r.k())).collect();
    let wanted = |i: usize, j: usize| match (ks[i], ks[i + 1]) {
        // The cell's least loss is at its slack corner, where the timestep is largest.
        (Some(k), Some(_)) => qs[j] / k <= sigma_top,
        _ => false,
    };
    let mut pts: Vec<Vec<Option<Vec<Root>>>> = vec![vec![None; nq]; nx];
    let mut done = vec![vec![false; nq]; nx];
    for i in 0..nx - 1 {
        for j in 0..nq - 1 {
            if !wanted(i, j) {
                continue;
            }
            for (a, b) in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)] {
                if !done[a][b] {
                    done[a][b] = true;
                    pts[a][b] = ks[a].and_then(|k| {
                        let mut rig = build(xs[a], qs[b] / k)?;
                        band_roots(&mut rig, xs[a]).ok()
                    });
                }
            }
        }
    }

    // The seeds, and where a root crosses the pitch along an edge with loss in it.
    let mut seeds = Vec::new();
    let mut crossings = Vec::new();
    for i in 0..nx {
        for j in 0..nq {
            let Some(here) = &pts[i][j] else { continue };
            // The edge to the right on a lossy level, the edge upward on every level.
            let right = (j > 0 && i + 1 < nx).then(|| (i + 1, j));
            let up = (j + 1 < nq).then(|| (i, j + 1));
            for (a, b) in right.into_iter().chain(up) {
                let Some(there) = &pts[a][b] else { continue };
                for r in here {
                    let Some(m) = nearest_note(there, r.f, r.a) else {
                        continue;
                    };
                    if (r.f - f0) * (m.f - f0) < 0.0 {
                        crossings.push(Crossing {
                            from: (xs[i], qs[j]),
                            to: (xs[a], qs[b]),
                            r: *r,
                            m,
                            estimate: r.a + (m.a - r.a) * (f0 - r.f) / (m.f - r.f),
                        });
                    }
                }
            }
            if i + 1 == nx || j + 1 == nq || !wanted(i, j) {
                continue;
            }
            let corners = [(i + 1, j), (i, j + 1), (i + 1, j + 1)];
            if corners.iter().any(|&(a, b)| pts[a][b].is_none()) {
                continue;
            }
            for r in here {
                let (mut ef, mut ea) = (vec![r.f - f0], vec![r.a - a0]);
                for &(a, b) in &corners {
                    if let Some(m) = pts[a][b].as_deref().and_then(|v| nearest_note(v, r.f, r.a)) {
                        ef.push(m.f - f0);
                        ea.push(m.a - a0);
                    }
                }
                let spans = |v: &[f64]| v.iter().any(|e| *e <= 0.0) && v.iter().any(|e| *e >= 0.0);
                if spans(&ef) && spans(&ea) {
                    let xc = (xs[i] * xs[i + 1]).sqrt();
                    let qc = if j == 0 {
                        0.5 * qs[1]
                    } else {
                        (qs[j] * qs[j + 1]).sqrt()
                    };
                    seeds.push((xc, qc, *r, i, j));
                }
            }
        }
    }

    let mut found: Vec<(f64, f64, Root)> = Vec::new(); // (x, q, root)
    for (x, q, r, i, j) in seeds {
        let (xl, xh) = (xs[i.saturating_sub(1)], xs[(i + 2).min(nx - 1)]);
        let (ql, qh) = (qs[j.saturating_sub(1)], qs[(j + 2).min(nq - 1)]);
        if found
            .iter()
            .any(|s| xl <= s.0 && s.0 <= xh && ql <= s.1 && s.1 <= qh)
        {
            continue;
        }
        let solved = if grid.damped {
            wide_solve_damped(f0, a0, x, q, r, build)
        } else {
            wide_solve(f0, a0, x, q, r, build)
        };
        if let Some(s) = solved {
            let dup = found
                .iter()
                .any(|t| (t.0 / s.0 - 1.0).abs() < SAME_X && (t.1 / s.1 - 1.0).abs() < SAME_SIGMA);
            if !dup {
                found.push(s);
            }
        }
    }
    let notes = found
        .into_iter()
        .filter_map(|(x, q, r)| {
            let k = build(x, 0.0)?.k();
            let mut rig = build(x, q / k)?;
            let share = share_of(&mut rig, r.z).ok()?;
            Some(WideNote {
                x,
                sigma: q / k,
                z: r.z,
                share,
            })
        })
        .collect();
    Wide { notes, crossings }
}

/// A Broyden solve on `(ln f_cl, ln q)` for pitch `f0` and decay `a0`, from `(x, q)` on the root
/// nearest `start`, followed by nearness: a fresh finite-difference Jacobian at the start and
/// after a step that made the residual worse, otherwise a rank-one update, so one spectrum per
/// step. Steps are clamped to 5 % in `f_cl` and a factor 1.65 in `q`. `None` when it leaves the
/// page's band for `f0` or the loss cap, a build is refused, the note grows, or [`WIDE_STALL`]
/// steps in a row bring no new smallest residual.
fn wide_solve<R: Coupled>(
    f0: f64,
    a0: f64,
    x: f64,
    q: f64,
    start: Root,
    build: &impl Fn(f64, f64) -> Option<R>,
) -> Option<(f64, f64, Root)> {
    let (xlo, xhi) = (f0 / 1.3, 4.0 * f0);
    let res = |u: [f64; 2], near: Root| -> Option<([f64; 2], Root)> {
        let (x, q) = (u[0].exp(), u[1].exp());
        if !(xlo < x && x < xhi && q <= 0.5) {
            return None;
        }
        let k = build(x, 0.0)?.k();
        let mut rig = build(x, q / k)?;
        let r = nearest_note(&band_roots(&mut rig, x).ok()?, near.f, near.a)?;
        (r.a > 0.0).then_some(([(r.f / f0).ln(), (r.a / a0).ln()], r))
    };
    let norm = |r: [f64; 2]| r[0].hypot(r[1]);
    let mut u = [x.ln(), q.max(Q_FLOOR).ln()];
    let (mut r, mut root) = res(u, start)?;
    let mut jac = [[0.0; 2]; 2];
    let mut fresh = true;
    let (mut best, mut since) = (norm(r), 0);
    for _ in 0..WIDE_ITERS {
        if r[0].abs() < PITCH_TOL && r[1].abs() < RING_TOL {
            return Some((u[0].exp(), u[1].exp(), root));
        }
        if fresh {
            let h = 1e-7;
            let (r0, _) = res([u[0] + h, u[1]], root)?;
            let (r1, _) = res([u[0], u[1] + h], root)?;
            jac = [
                [(r0[0] - r[0]) / h, (r1[0] - r[0]) / h],
                [(r0[1] - r[1]) / h, (r1[1] - r[1]) / h],
            ];
        }
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        if !(det.abs() > 0.0 && det.is_finite()) {
            return None;
        }
        let du = [
            (-(jac[1][1] * r[0] - jac[0][1] * r[1]) / det).clamp(-0.05, 0.05),
            (-(-jac[1][0] * r[0] + jac[0][0] * r[1]) / det).clamp(-0.5, 0.5),
        ];
        let next = [u[0] + du[0], u[1] + du[1]];
        let (rn, root_n) = res(next, root)?;
        let worse = norm(rn) >= norm(r);
        // The rank-one update: afterwards the Jacobian maps du to the change it made.
        let dr = [rn[0] - r[0], rn[1] - r[1]];
        let dd = du[0] * du[0] + du[1] * du[1];
        for (row, d) in jac.iter_mut().zip(dr) {
            let e = d - (row[0] * du[0] + row[1] * du[1]);
            row[0] += e * du[0] / dd;
            row[1] += e * du[1] / dd;
        }
        fresh = worse && !fresh;
        (u, r, root) = (next, rn, root_n);
        if norm(r) < best {
            (best, since) = (norm(r), 0);
        } else {
            since += 1;
            if since >= WIDE_STALL {
                return None;
            }
        }
    }
    None
}

/// [`wide_solve`]'s damped alternative, for a rig whose notes crowd ([`WideGrid::damped`]): a
/// fresh finite-difference Jacobian every step, and the step — clamped as there — halved up to
/// [`WIDE_HALVINGS`] times until the residual falls, so the note is followed in steps small enough
/// to stay on it. `None` as there, or when no halving makes the residual fall.
///
/// On the plate, whose notes lie a few hertz apart, the Broyden's full steps landed on a
/// neighbouring note and stalled from every seed of 21 targets where a tuning exists (plan §18).
fn wide_solve_damped<R: Coupled>(
    f0: f64,
    a0: f64,
    x: f64,
    q: f64,
    start: Root,
    build: &impl Fn(f64, f64) -> Option<R>,
) -> Option<(f64, f64, Root)> {
    let (xlo, xhi) = (f0 / 1.3, 4.0 * f0);
    let res = |u: [f64; 2], near: Root| -> Option<([f64; 2], Root)> {
        let (x, q) = (u[0].exp(), u[1].exp());
        if !(xlo < x && x < xhi && q <= 0.5) {
            return None;
        }
        let k = build(x, 0.0)?.k();
        let mut rig = build(x, q / k)?;
        let r = nearest_note(&band_roots(&mut rig, x).ok()?, near.f, near.a)?;
        (r.a > 0.0).then_some(([(r.f / f0).ln(), (r.a / a0).ln()], r))
    };
    let norm = |r: [f64; 2]| r[0].hypot(r[1]);
    let mut u = [x.ln(), q.max(Q_FLOOR).ln()];
    let (mut r, mut root) = res(u, start)?;
    for _ in 0..WIDE_ITERS {
        if r[0].abs() < PITCH_TOL && r[1].abs() < RING_TOL {
            return Some((u[0].exp(), u[1].exp(), root));
        }
        let h = 1e-7;
        let (r0, _) = res([u[0] + h, u[1]], root)?;
        let (r1, _) = res([u[0], u[1] + h], root)?;
        let jac = [
            [(r0[0] - r[0]) / h, (r1[0] - r[0]) / h],
            [(r0[1] - r[1]) / h, (r1[1] - r[1]) / h],
        ];
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        if !(det.abs() > 0.0 && det.is_finite()) {
            return None;
        }
        let du = [
            (-(jac[1][1] * r[0] - jac[0][1] * r[1]) / det).clamp(-0.05, 0.05),
            (-(-jac[1][0] * r[0] + jac[0][0] * r[1]) / det).clamp(-0.5, 0.5),
        ];
        let mut t = 1.0;
        let mut took = None;
        for _ in 0..=WIDE_HALVINGS {
            let next = [u[0] + t * du[0], u[1] + t * du[1]];
            if let Some((rn, root_n)) = res(next, root) {
                if norm(rn) < norm(r) {
                    took = Some((next, rn, root_n));
                    break;
                }
            }
            t *= 0.5;
        }
        (u, r, root) = took?;
    }
    None
}

/// The decay of the root at `c`, solved onto the pitch along its edge: a bracketed solve on the
/// edge's own coordinate (log `f_cl` along a level, log `q` up a column), the root at each point
/// the nearest to the straight line between the edge's two. `None` when a build or a spectrum on
/// the edge fails.
fn exact_decay<R: Coupled>(
    f0: f64,
    c: &Crossing,
    build: &impl Fn(f64, f64) -> Option<R>,
) -> Option<f64> {
    let along_x = c.from.1 == c.to.1;
    // The lossless end of a column is moved to a thousandth of the next level, for the log.
    let (va, vb) = if along_x {
        (c.from.0, c.to.0)
    } else {
        (c.from.1.max(1e-3 * c.to.1), c.to.1)
    };
    let at = |v: f64| -> Result<Root, CoupledError> {
        let (x, q) = if along_x {
            (v, c.from.1)
        } else {
            (c.from.0, v)
        };
        let t = (v / va).ln() / (vb / va).ln();
        let (f, a) = (c.r.f + t * (c.m.f - c.r.f), c.r.a + t * (c.m.a - c.r.a));
        let k = build(x, 0.0).ok_or(CoupledError::Refused)?.k();
        let mut rig = build(x, q / k).ok_or(CoupledError::Refused)?;
        nearest_note(&band_roots(&mut rig, x)?, f, a).ok_or(CoupledError::NoNote)
    };
    let v = bracket(va, vb, |v| at(v).map(|r| r.f - f0)).ok()?;
    at(v).ok().map(|r| r.a)
}

/// The refusal when no tension and loss ring `a0` (decay rate) at the pitch.
///
/// When a lossless string has a note on the pitch, the continuation's own refusal stands: it is
/// exact, and still true of the note the lossless string plays there (the wide search found no
/// other note to put on the targets). When none has, the notes the wide search saw on the pitch
/// with a lossy string decide: none — no note; all faster than the target — it cannot ring that
/// long; all slower — it cannot ring that short; some each way and still no solution — the search
/// did not converge.
///
/// The extremes are solved onto the pitch first ([`exact_decay`]): read off a straight line across
/// a coarse edge, a decay was 10 % out at 50 Hz on the 8000 N/m body (0.0377 s for 0.0338 s),
/// enough to call a ring between the two too short when the note cannot ring that long. The least
/// and the most are each re-solved until the extreme is one already solved.
fn ring_refusal<R: Coupled>(
    narrow: CoupledError,
    f0: f64,
    a0: f64,
    crossings: &[Crossing],
    build: &impl Fn(f64, f64) -> Option<R>,
) -> CoupledError {
    if narrow != CoupledError::NoNote || crossings.is_empty() {
        return narrow;
    }
    // (decay, solved): a failed solve keeps its estimate, counted as solved so it is not retried.
    let mut decays: Vec<(f64, bool)> = crossings.iter().map(|c| (c.estimate, false)).collect();
    let mut extreme = |most: bool| loop {
        let i = (0..decays.len())
            .max_by(|&i, &j| {
                let o = decays[i].0.total_cmp(&decays[j].0);
                if most {
                    o
                } else {
                    o.reverse()
                }
            })
            .unwrap_or(0);
        if decays[i].1 {
            return decays[i].0;
        }
        decays[i] = (
            exact_decay(f0, &crossings[i], build).unwrap_or(decays[i].0),
            true,
        );
    };
    let least = extreme(false);
    let most = extreme(true);
    if least > a0 {
        CoupledError::RingTooLong {
            longest: 3.0 * LN_10 / least,
        }
    } else if most < a0 {
        CoupledError::RingTooShort {
            shortest: 3.0 * LN_10 / most,
        }
    } else {
        CoupledError::Unconverged("the ring-time search")
    }
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
#[allow(clippy::type_complexity)]
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
