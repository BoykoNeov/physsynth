//! The `lam_long` edges of the geometrically exact string — the measurement behind
//! `docs/dev/scientific-hurdles.md` §6, which is still open.
//!
//! ```text
//! cargo run --release -p physsynth-core --example geometric_lam_long [-- --full]
//! ```
//!
//! Prints tables; writes nothing. The native successor of `scripts/sweep_geometric_lam_long.py`
//! (deleted in `docs/dev/python-retirement-plan.md` §22), and deliberately only half of it: that
//! script ran every cell twice, once with the pre-§1 Jacobian as an A/B control, and the answer —
//! the fix did not move the edge, 9 cells of 9 — is recorded in §6. What stays useful while §6 is
//! open is the *shape* of the measurement on the model as it is today.
//!
//! **Two columns, not one.** In the failing regime the energy drift is a step function of the
//! Newton iteration count, so drift alone cannot see where the solve starts to fail. Every row
//! reports the stalled-step count (steps that exhausted `newton_maxiter`) beside the drift, and
//! that is what separates the model's two thresholds:
//!
//! - **the convergence edge** — the smallest `lam_long` at which any step stalls;
//! - **the energy edge** — the smallest `lam_long` whose lossless drift breaks [`DRIFT_GATE`].
//!
//! Between them the solve stalls and the energy still conserves, so a flat energy is not a
//! convergence certificate there. `tests/test_geometric_energy.py` pins one point of that band.
//!
//! A non-finite drift counts as past the gate. The Python original compared `drift > gate`, which
//! is false for NaN, so a run that blew up to NaN before ever exceeding the gate would have been
//! read as conserving; none of the recorded cells did, so no recorded number depends on it.

use physsynth_core::string_geometric as geo;
use physsynth_core::string_stiff::THETA_DEFAULT;

// The canonical string of this family's tests: c = 200 m/s, f1 = 100 Hz, EA/T = 500 so
// c_long/c = 22.4. `kappa` is the suite's 2.0 rather than 0 so the numbers stay comparable.
const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005;
const EA: f64 = 1.0e5;
const KAPPA: f64 = 2.0;

/// CLAUDE.md's lossless bar. The energy edge is defined against this and nothing else.
const DRIFT_GATE: f64 = 1e-10;

#[derive(Clone, Copy)]
enum Ic {
    /// A single simply-supported eigenmode.
    Mode(usize),
    /// A triangular pluck at 0.2 L: broadband, so the stretch varies cell to cell.
    Pluck,
}

impl Ic {
    fn label(self) -> String {
        match self {
            Ic::Mode(m) => format!("mode-{m}"),
            Ic::Pluck => "pluck".to_string(),
        }
    }

    fn shape(self, x: &[f64], amp: f64) -> Vec<f64> {
        match self {
            Ic::Mode(m) => x
                .iter()
                .map(|&xi| amp * (m as f64 * std::f64::consts::PI * xi / L).sin())
                .collect(),
            Ic::Pluck => x
                .iter()
                .map(|&xi| {
                    let s = xi / L;
                    amp * if s <= 0.2 { s / 0.2 } else { (1.0 - s) / 0.8 }
                })
                .collect(),
        }
    }
}

struct Run {
    drift: f64,
    steps: usize,
    iters_per_step: f64,
    peak_iters: usize,
    stalled: usize,
}

/// One trajectory. `duration` is physical seconds, not steps: `lam_long` sets `fs`, so a fixed
/// step count would compare different amounts of physics at each point of the sweep.
fn run(n_cells: i64, lam_long: f64, ic: Ic, amp: f64, duration: f64) -> Run {
    let fs = (EA / RHO).sqrt() * n_cells as f64 / (L * lam_long);
    let p = geo::Params::new(
        L,
        T,
        RHO,
        fs,
        n_cells,
        EA,
        KAPPA,
        None,
        0.0,
        0.0,
        None,
        None,
        THETA_DEFAULT,
        true,
        geo::NEWTON_TOL_DEFAULT,
        geo::NEWTON_MAXITER_DEFAULT,
        false,
    )
    .expect("the sweep's fixtures construct");
    let x = p.grid();
    let nodes = p.nodes();
    let mut s = geo::GeometricString::new(p);
    let zero = vec![0.0; nodes];
    s.set_state(
        &ic.shape(&x, amp),
        &zero,
        &zero,
        &[zero.clone(), zero.clone(), zero.clone()],
    );

    let e0 = s.energy();
    let steps = (duration * fs).round() as usize;
    let mut peak_iters = 0;
    for _ in 0..steps {
        s.step().expect("the Newton solve must factor");
        peak_iters = peak_iters.max(s.newton_iters);
    }
    Run {
        drift: (s.energy() - e0).abs() / e0.abs(),
        steps,
        iters_per_step: s.total_newton_iters as f64 / steps as f64,
        peak_iters,
        stalled: s.n_not_converged,
    }
}

fn past_gate(drift: f64) -> bool {
    !drift.is_finite() || drift > DRIFT_GATE
}

fn show(edge: Option<f64>) -> String {
    edge.map_or_else(|| "none".to_string(), |v| format!("{v:.1}"))
}

/// The deliverable: two edges per cell.
fn edge_table(duration: f64, lams: &[f64], cases: &[(i64, f64, Ic)]) {
    println!("\n== the edges: duration={duration}s gate={DRIFT_GATE:e} ==");
    println!(
        "{:>4} {:>8} {:>7} | {:>11} {:>11}",
        "N", "amp", "ic", "convergence", "energy"
    );
    for &(n_cells, amp, ic) in cases {
        let mut convergence = None;
        let mut energy = None;
        for &lam_long in lams {
            let r = run(n_cells, lam_long, ic, amp, duration);
            if convergence.is_none() && r.stalled > 0 {
                convergence = Some(lam_long);
            }
            if past_gate(r.drift) {
                energy = Some(lam_long);
                break;
            }
        }
        println!(
            "{n_cells:4} {amp:8.1e} {:>7} | {:>11} {:>11}",
            ic.label(),
            show(convergence),
            show(energy)
        );
    }
}

/// Drift and convergence against `lam_long` in full, for three initial conditions.
fn sweep_table(n_cells: i64, duration: f64, amp: f64, lams: &[f64]) {
    println!("\n== the sweep: N={n_cells} duration={duration}s amp={amp:e} ==");
    println!(
        "{:8} {:>8} {:>10} {:>8} {:>5} {:>8} {:>6}",
        "case", "lam_long", "drift", "it/step", "peak", "stalled", "steps"
    );
    for ic in [Ic::Mode(3), Ic::Mode(1), Ic::Pluck] {
        for &lam_long in lams {
            let r = run(n_cells, lam_long, ic, amp, duration);
            println!(
                "{:8} {lam_long:8.1} {:10.2e} {:8.2} {:5} {:8} {:6}",
                ic.label(),
                r.drift,
                r.iters_per_step,
                r.peak_iters,
                r.stalled,
                r.steps
            );
        }
    }
}

fn main() {
    let full = std::env::args().any(|a| a == "--full");
    let cases = [
        (16, 4e-3, Ic::Mode(3)),
        (16, 8e-3, Ic::Mode(3)),
        (16, 4e-3, Ic::Pluck),
        (24, 4e-3, Ic::Mode(3)),
        (24, 8e-3, Ic::Mode(3)),
        (32, 4e-3, Ic::Mode(3)),
        (32, 8e-3, Ic::Mode(3)),
        (32, 4e-3, Ic::Pluck),
        (32, 1e-2, Ic::Mode(3)),
    ];
    let lams = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    edge_table(0.004, &lams, &cases);
    if full {
        sweep_table(
            32,
            0.02,
            4e-3,
            &[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 9.5, 11.0],
        );
    }
}
