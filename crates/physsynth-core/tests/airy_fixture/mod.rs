//! The Airy solve's shared test fixture — the recorder's generator, the area weight and the
//! backward error — used by `ops2d_scipy.rs` (the recorded grids, both profiles) and
//! `ops2d_airy_large.rs` (the five large grids, optimised only; retirement plan §49).

#![allow(dead_code)]

use physsynth_core::ops2d::AiryStressSolver;
use physsynth_core::sparse::Csr;

/// The recorder's generator: `2u - 1` with `u` the LCG `src/reduce.rs`'s tests use.
pub fn lcg(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(2_862_933_555_777_941_757)
        .wrapping_add(3_037_000_493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            2.0 * (((s >> 11) as f64) / ((1u64 << 53) as f64)) - 1.0
        })
        .collect()
}

/// `Wa = kron(m_y, m_x)` — interior `h²`, edge `h²/2`, corner `h²/4` — in `np.kron`'s order.
pub fn area_weights(nx: usize, ny: usize, h: f64) -> Vec<f64> {
    let mut mx = vec![h; nx + 1];
    mx[0] = 0.5 * h;
    mx[nx] = 0.5 * h;
    let mut my = vec![h; ny + 1];
    my[0] = 0.5 * h;
    my[ny] = 0.5 * h;
    my.iter()
        .flat_map(|wy| mx.iter().map(move |wx| wy * wx))
        .collect()
}

/// The infinity norm, `max_i Σ_j |B_ij|` — `np.max(np.abs(B).sum(axis=1))`.
pub fn norm_inf(m: &Csr) -> f64 {
    (0..m.nrows())
        .map(|i| {
            m.data()[m.indptr()[i]..m.indptr()[i + 1]]
                .iter()
                .map(|v| v.abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| {
        if x.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

/// The live part of a source and its `Wa`-weighted load, assembled here and not by the crate, so
/// the backward error below also checks the solve's own load assembly.
pub fn live_and_load(
    airy: &AiryStressSolver,
    nx: usize,
    ny: usize,
    h: f64,
    raw: &[f64],
) -> Vec<f64> {
    let wa = area_weights(nx, ny, h);
    (0..airy.n_nodes())
        .filter(|&p| airy.index_map()[p] >= 0)
        .map(|p| wa[p] * raw[p])
        .collect()
}

pub fn source(airy: &AiryStressSolver, raw: &[f64]) -> Vec<f64> {
    raw.iter()
        .zip(airy.index_map())
        .map(|(&v, &m)| if m >= 0 { v } else { 0.0 })
        .collect()
}

pub fn live(airy: &AiryStressSolver, full: &[f64]) -> Vec<f64> {
    full.iter()
        .zip(airy.index_map())
        .filter(|(_, &m)| m >= 0)
        .map(|(&v, _)| v)
        .collect()
}

/// `||B f - rhs||_∞ / (||B||_∞ ||f||_∞)` — the backward error a stable solve leaves at epsilon.
pub fn backward_error(bf: &Csr, f_live: &[f64], rhs: &[f64]) -> f64 {
    let r: Vec<f64> = bf
        .matvec(f_live)
        .iter()
        .zip(rhs)
        .map(|(a, b)| a - b)
        .collect();
    max_abs(&r) / (norm_inf(bf) * max_abs(f_live).max(1e-300))
}
