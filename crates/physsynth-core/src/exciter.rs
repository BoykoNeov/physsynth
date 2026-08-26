//! Excitations — the initial-condition shapes that start a resonator moving.
//!
//! Port of `physsynth/core/exciter.py`. These are pure functions of a grid: they return a field
//! sampled on the resonator's nodes, which the caller hands to `set_state`. No state, no history,
//! nothing to conserve — the energy contract belongs to the resonator that receives the shape.
//!
//! # Two shapes, two jobs
//!
//! [`triangular_pluck`] has a corner, so it excites the full harmonic series and is the right
//! excitation for the modal/partial-detection tests. [`raised_cosine`] (and its 2-D sibling) is
//! `C^1` and band-limited, which is what a grid-convergence study wants — a cornered shape's
//! high partials converge too slowly to read an order of accuracy off.
//!
//! # The one function here whose exactness is not free
//!
//! Everything in this module is elementwise arithmetic, so by the plan's §2.1 rule it should be
//! bit-identical to NumPy. Two of the three are: `sqrt` is correctly rounded by IEEE-754, and
//! NumPy's `x ** 2` takes its own fast path to `np.square`, i.e. `x * x`. But the raised cosines
//! call **`cos`**, and NumPy does not use the platform libm for that — it has vectorised
//! implementations with their own (~1 ulp) error budget. So the raised cosines are the first
//! kernels in this port whose agreement rests on two transcendental implementations matching, the
//! way `delta_xxxx`'s rests on two `pow`s (plan §10.3). `tests/test_rust_parity_ops2d.py`
//! measures it rather than assuming it.

/// A rejected excitation. `Display` reproduces the Python original's message verbatim, because
/// the suite matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ExciterError {
    /// The pluck point was not strictly inside the string. Carries `(L, position)`.
    PluckOutside(f64, f64),
    /// A raised cosine was given a non-positive half-width.
    NonPositiveWidth,
}

impl std::fmt::Display for ExciterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExciterError::PluckOutside(l, p) => write!(
                f,
                "pluck position must satisfy 0 < position < L (L={}), got {}.",
                crate::fmt::py_float(*l),
                crate::fmt::py_float(*p)
            ),
            ExciterError::NonPositiveWidth => write!(f, "width must be > 0."),
        }
    }
}

impl std::error::Error for ExciterError {}

/// Triangular plucked-string initial displacement — a tent peaked at `position`, zero at both ends.
///
/// Linear rise from the left end to the pluck point, linear fall to the right end. The corner is
/// the point: it excites every harmonic, which is what the modal test needs.
///
/// The node exactly at `position` takes the *left* branch, as `x <= xp` does in the original.
pub fn triangular_pluck(
    x: &[f64],
    l: f64,
    position: f64,
    amplitude: f64,
) -> Result<Vec<f64>, ExciterError> {
    if !(position > 0.0 && position < l) {
        return Err(ExciterError::PluckOutside(l, position));
    }
    let xp = position;
    Ok(x.iter()
        .map(|&xv| {
            if xv <= xp {
                (amplitude * xv) / xp
            } else {
                (amplitude * (l - xv)) / (l - xp)
            }
        })
        .collect())
}

/// Smooth (`C^1`) raised-cosine displacement hump, zero outside `[center - width, center + width]`.
///
/// Band-limited compared with [`triangular_pluck`] — no corner, so no slowly-converging high
/// partials — which makes it the cleaner excitation for a grid-convergence study.
///
/// The two end nodes are clamped to zero on the way out, so a hump placed near a boundary cannot
/// leave a non-zero node sitting on a fixed end. `l` is accepted and unused, matching the
/// original's signature; the string's length does not enter the shape.
pub fn raised_cosine(
    x: &[f64],
    l: f64,
    center: f64,
    width: f64,
    amplitude: f64,
) -> Result<Vec<f64>, ExciterError> {
    let _ = l;
    if width <= 0.0 {
        return Err(ExciterError::NonPositiveWidth);
    }
    let mut u0: Vec<f64> = x
        .iter()
        .map(|&xv| {
            let d = (xv - center).abs();
            if d < width {
                (amplitude * 0.5) * (1.0 + ((std::f64::consts::PI * d) / width).cos())
            } else {
                0.0
            }
        })
        .collect();
    if let Some(first) = u0.first_mut() {
        *first = 0.0;
    }
    if let Some(last) = u0.last_mut() {
        *last = 0.0;
    }
    Ok(u0)
}

/// Radial raised-cosine hump on a 2-D grid, zero outside radius `width`.
///
/// The 2-D analogue of [`raised_cosine`]: a band-limited bump centred at `center = (xc, yc)`,
/// falling smoothly to zero at radial distance `width`. `x` and `y` are flat row-major node fields
/// (see `ops2d`) and the result has their shape.
///
/// Unlike the 1-D version there is **no end clamping** — a 2-D domain's boundary is a rim, not two
/// nodes, and the caller selects live nodes through the mask anyway.
///
/// # Panics
/// If `x` and `y` have different lengths.
pub fn raised_cosine_2d(
    x: &[f64],
    y: &[f64],
    center: (f64, f64),
    width: f64,
    amplitude: f64,
) -> Result<Vec<f64>, ExciterError> {
    assert_eq!(x.len(), y.len(), "x and y must be the same field shape");
    if width <= 0.0 {
        return Err(ExciterError::NonPositiveWidth);
    }
    let (xc, yc) = center;
    Ok(x.iter()
        .zip(y.iter())
        .map(|(&xv, &yv)| {
            // `(X - xc) ** 2` in NumPy takes the `np.square` fast path, so this is `d * d` there
            // too; `np.sqrt` is IEEE-correctly-rounded. The `cos` below is the inexact one.
            let dx = xv - xc;
            let dy = yv - yc;
            let d = (dx * dx + dy * dy).sqrt();
            if d < width {
                (amplitude * 0.5) * (1.0 + ((std::f64::consts::PI * d) / width).cos())
            } else {
                0.0
            }
        })
        .collect())
}
