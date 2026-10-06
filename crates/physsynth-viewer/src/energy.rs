//! The energy panel — `_energy_block`, `_balance_verdict` and `_fit_decay`.
//!
//! One report with three verdicts, and which one a scene gets is the whole content of the panel
//! (the viewer plan's catch #4, and the driven-model extension):
//!
//! * **lossless** — `max|E - E0| / |E0|` against the project's `1e-10` bar;
//! * **lossy** — passivity: the largest per-step *increase*, relative to `E0`, must stay under
//!   `1e-9`, plus the measured decay rate against the oracle's `2 sigma` when one exists;
//! * **balance** — for a driven model, `E - E0 == work_in - loss`, which replaces both of the above
//!   because for a bow or a reed neither is merely weak, both are wrong (see [`balance_verdict`]).
//!
//! The trace is decimated to [`N_ENERGY_POINTS`] with NumPy's `linspace` index, but every *verdict*
//! is computed on the full per-step arrays: a max over a decimated trace samples fewer steps and
//! understates the number it reports.

use crate::py::{finite_list_at, linspace_idx, num};
use serde_json::{json, Map, Value};

/// Decimated energy-trace length for the plot.
pub const N_ENERGY_POINTS: usize = 600;
/// The project's energy-drift bar (HANDOFF §6).
pub const LOSSLESS_TOL: f64 = 1e-10;
/// Relative per-step energy-increase tolerance for the passivity check.
pub const MONOTONE_TOL: f64 = 1e-9;
/// Lossless `|dE - work| / scale` bar for a driven model (the retired
/// `tests/test_bow_energy.py`'s, now `physsynth-core/tests/bow_harness.rs`'s).
pub const BOW_BALANCE_TOL: f64 = 1e-11;

/// The optional parts of an energy report; `Default` is the plain string's.
#[derive(Default)]
pub struct EnergyOpts<'a> {
    /// The von Kármán Picard gate: `{"all_converged": bool, ...}`, carried and AND-ed into the
    /// lossless verdict (the identity telescopes only at the fixed point).
    pub convergence: Option<Value>,
    /// Cumulative exciter work per step — switches the block to the balance verdict.
    pub balance_work: Option<&'a [f64]>,
    /// `true` drops the measured-vs-oracle `2 sigma` line (the mallet: a closed system whose
    /// energy floors at the mallet's kinetic energy, so a fitted rate would be a lying zero).
    pub no_decay_oracle: bool,
    /// Named channels of the same total (the bore's acoustic / radiated split), decimated alike.
    pub split: Option<Vec<(&'a str, &'a [f64])>>,
    /// Independently measured dissipation channels (the reed) — see [`balance_verdict`].
    pub measured_loss: Option<Vec<(&'a str, &'a [f64])>>,
}

/// `max|E - E0| / |E0|`, or `max|E|` when `E0 == 0` — `SimResult.energy_drift`, NaN-propagating.
pub fn energy_drift(e: &[f64]) -> f64 {
    let e0 = e[0];
    let mut m = f64::NEG_INFINITY;
    for &v in e {
        let d = if e0 == 0.0 { v.abs() } else { (v - e0).abs() };
        if d.is_nan() {
            return f64::NAN;
        }
        if d > m {
            m = d;
        }
    }
    if e0 == 0.0 {
        m
    } else {
        m / e0.abs()
    }
}

/// `np.max` over a non-empty slice, NaN-propagating.
fn np_max(a: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// `np.min` over a non-empty slice, NaN-propagating.
fn np_min(a: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if v < m {
            m = v;
        }
    }
    m
}

/// The energy report. `time` and `energy` are the full per-step arrays.
pub fn energy_block(
    time: &[f64],
    energy: &[f64],
    sigma_zero: bool,
    oracle_2sigma: f64,
    opts: EnergyOpts<'_>,
) -> Value {
    let idx = linspace_idx(energy.len() - 1, energy.len().min(N_ENERGY_POINTS));
    let mut block = Map::new();
    block.insert("sigma_is_zero".into(), Value::Bool(sigma_zero));
    block.insert("time".into(), finite_list_at(time, &idx, Some(6)));
    block.insert("value".into(), finite_list_at(energy, &idx, None));
    if let Some(split) = &opts.split {
        let m: Map<String, Value> = split
            .iter()
            .map(|(k, v)| ((*k).to_owned(), finite_list_at(v, &idx, None)))
            .collect();
        block.insert("split".into(), Value::Object(m));
    }
    let converged = opts
        .convergence
        .as_ref()
        .is_none_or(|c| c["all_converged"].as_bool().unwrap_or(false));
    if let Some(c) = opts.convergence {
        block.insert("convergence".into(), c);
    }
    if let Some(w) = opts.balance_work {
        block.insert("kind".into(), json!("balance"));
        block.insert(
            "balance".into(),
            balance_verdict(energy, w, sigma_zero, &idx, opts.measured_loss.as_deref()),
        );
        return Value::Object(block);
    }

    if sigma_zero {
        let drift = energy_drift(energy);
        block.insert(
            "lossless".into(),
            json!({
                "drift": num(drift),
                "tol": num(LOSSLESS_TOL),
                "pass": drift < LOSSLESS_TOL && converged,
            }),
        );
        return Value::Object(block);
    }

    let e0 = energy[0];
    let de: Vec<f64> = energy.windows(2).map(|w| w[1] - w[0]).collect();
    // `np.max(dE) / E0 if E0 > 0 else np.max(dE) if dE.size else 0.0`
    let max_rel_inc = if e0 > 0.0 {
        np_max(&de) / e0
    } else if !de.is_empty() {
        np_max(&de)
    } else {
        0.0
    };
    let mut lossy = Map::new();
    lossy.insert("monotone".into(), Value::Bool(max_rel_inc <= MONOTONE_TOL));
    lossy.insert("max_rel_increase".into(), num(max_rel_inc));
    if !opts.no_decay_oracle {
        lossy.insert(
            "measured_2sigma".into(),
            fit_decay(time, energy).map_or(Value::Null, num),
        );
        lossy.insert("oracle_2sigma".into(), num(oracle_2sigma));
    }
    block.insert("lossy".into(), Value::Object(lossy));
    Value::Object(block)
}

/// `-slope` of `log E` against `t` over the samples above `1e-6 E0` — `2 sigma_eff`.
///
/// `np.polyfit(t, log E, 1)` solves the same least-squares line through an SVD; this is the closed
/// form about the mean, which is the same line and differs from LAPACK's only in the last bits.
/// `None` when fewer than two samples qualify.
pub fn fit_decay(t: &[f64], e: &[f64]) -> Option<f64> {
    let floor = e[0] * 1e-6;
    let pts: Vec<(f64, f64)> = t
        .iter()
        .zip(e)
        .filter(|(_, &ev)| ev > floor)
        .map(|(&tv, &ev)| (tv, ev.ln()))
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let (t, y): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
    Some(-lstsq_slope(&t, &y))
}

/// The slope of the least-squares line through `(x, y)` — `np.polyfit(x, y, 1)[0]`.
///
/// `polyfit` solves this through LAPACK's SVD; this is the closed form about the mean, the same
/// line, differing from LAPACK's in the last bits (measured: 3e-14 relative at worst).
pub fn lstsq_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let xm = x.iter().sum::<f64>() / n;
    let ym = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (&xv, &yv) in x.iter().zip(y) {
        sxy += (xv - xm) * (yv - ym);
        sxx += (xv - xm) * (xv - xm);
    }
    sxy / sxx
}

/// The energy-BALANCE verdict for an actively driven model: `E - E0 == work_in - loss`.
///
/// * lossless (`sigma_zero`): `max|(E - E0) - w| / (|E| + |w|)` against [`BOW_BALANCE_TOL`];
/// * lossy: the dissipation is *inferred* as `w - (E - E0)`, so a residual would be zero by
///   construction — the honest content is that it is non-negative and never decreases;
/// * `measured_loss` (the reed): the channels are measured independently, so the residual
///   `dE - (w - sum(loss))` can genuinely fail, in every regime, and there is no sigma gate. The
///   bow-style residual with the loss dropped is shipped beside it as evidence the channels are
///   load-bearing.
pub fn balance_verdict(
    e: &[f64],
    w: &[f64],
    sigma_zero: bool,
    idx: &[usize],
    measured_loss: Option<&[(&str, &[f64])]>,
) -> Value {
    let e0 = e[0];
    let de: Vec<f64> = e.iter().map(|&v| v - e0).collect();
    let scale: Vec<f64> = e
        .iter()
        .zip(w)
        .map(|(&ev, &wv)| ev.abs() + wv.abs() + 1e-30)
        .collect();
    let w_last = *w.last().expect("a run has at least two samples");

    if let Some(channels) = measured_loss {
        let mut loss_total = vec![0.0; de.len()];
        for (_, arr) in channels {
            for (lt, &a) in loss_total.iter_mut().zip(arr.iter()) {
                *lt += a;
            }
        }
        let residual = np_max(
            &(0..de.len())
                .map(|i| (de[i] - (w[i] - loss_total[i])).abs() / scale[i])
                .collect::<Vec<_>>(),
        );
        let naive = np_max(
            &(0..de.len())
                .map(|i| (de[i] - w[i]).abs() / scale[i])
                .collect::<Vec<_>>(),
        );
        let mut ch = Map::new();
        let mut totals = Map::new();
        for (k, arr) in channels {
            ch.insert((*k).to_owned(), finite_list_at(arr, idx, None));
            totals.insert((*k).to_owned(), num(*arr.last().unwrap_or(&0.0)));
        }
        return json!({
            "work": finite_list_at(w, idx, None),
            "delta_energy": finite_list_at(&de, idx, None),
            "dissipation": finite_list_at(&loss_total, idx, None),
            "channels": ch,
            "channel_totals": totals,
            "work_total": num(w_last),
            "measured": {
                "residual": num(residual),
                "naive_residual": num(naive),
                "tol": num(BOW_BALANCE_TOL),
                "pass": residual < BOW_BALANCE_TOL,
            },
        });
    }

    let dissipation: Vec<f64> = w.iter().zip(&de).map(|(&wv, &d)| wv - d).collect();
    let mut block = Map::new();
    block.insert("work".into(), finite_list_at(w, idx, None));
    block.insert("delta_energy".into(), finite_list_at(&de, idx, None));
    block.insert(
        "dissipation".into(),
        finite_list_at(&dissipation, idx, None),
    );
    block.insert("work_total".into(), num(w_last));
    if sigma_zero {
        let residual = np_max(
            &(0..de.len())
                .map(|i| (de[i] - w[i]).abs() / scale[i])
                .collect::<Vec<_>>(),
        );
        block.insert(
            "lossless".into(),
            json!({
                "residual": num(residual),
                "tol": num(BOW_BALANCE_TOL),
                "pass": residual < BOW_BALANCE_TOL,
            }),
        );
        return Value::Object(block);
    }
    let scale_w = w_last.abs() + 1.0;
    let d_step: Vec<f64> = dissipation.windows(2).map(|p| p[1] - p[0]).collect();
    let diss_last = *dissipation.last().expect("non-empty");
    let non_negative = diss_last >= -BOW_BALANCE_TOL * scale_w;
    let (monotone, worst) = if d_step.is_empty() {
        (true, 0.0)
    } else {
        let m = np_min(&d_step);
        (m >= -1e-9 * scale_w, m)
    };
    block.insert(
        "lossy".into(),
        json!({
            "dissipation_total": num(diss_last),
            "non_negative": non_negative,
            "monotone": monotone,
            "worst_step": num(worst),
            "pass": non_negative && monotone,
        }),
    );
    Value::Object(block)
}
