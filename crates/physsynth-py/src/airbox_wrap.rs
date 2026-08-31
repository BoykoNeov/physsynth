//! `airbox.py`'s wrapper tier: the resonators a room loads, and the two seams they load through.
//!
//! Plan section 32. This is the third and last slice of `airbox.py` — section 30 ported the room,
//! section 31 the three ports, and this batch ports what sits on top of them: `RoomLoadedBody`,
//! `RoomLoadedPlate`, `RoomSuspendedPlate`, `RoomLoadedVKPlate`, `RoomSuspendedVKPlate` and the two
//! surface adapters (`_PlateSurface`, `_VKPlateSurface`) the four plate wrappers drive. The
//! membrane tier stays Python for now: nothing in `connection.py` needs it and no exact anchor
//! binds it to these classes.
//!
//! ## Why this file computes through Python and does not own its arithmetic
//!
//! Every other ported module holds its own state and does its own arithmetic. This one cannot, and
//! the reason is an interface decision the tier **below** made deliberately in section 31: a port
//! stores `T`, `R` and `load_matrix` as `Py<PyAny>` — plain Python slots — because eight tests
//! replace them wholesale to switch a coupling off, halve it, or flip its sign
//! (`inst.port.T = -inst.port.T`, `port.load_matrix = (port.T.T @ diags(port.R) @ port.T).tocsr()`),
//! and two more replace the port's *methods* on the instance. A wrapper that cached its own copy
//! of any of them would read a matrix the test had already replaced, and every one of those tests
//! would pass having asserted nothing — section 23.6 through yet another door.
//!
//! It is not only the ports. Three tests assign `inst._lu_loaded = splu(a)`, replacing the
//! factorization with one of their own, and two call `inst._surface.rhs(None)` and
//! `inst._surface.a_bare()` directly. So the factorization, the seam and the port are all objects
//! this tier **holds and calls**, never things it is.
//!
//! The line drawn here, therefore:
//!
//! * **Through Python:** every sparse product (`B @ u`, `T.T @ pbar`, `load_matrix @ u`), the
//!   assembly of `a_loaded`, the factorization and its `solve`, `np.dot` (BLAS fuses its
//!   multiply-add — section 14.2), and every call on a port, a seam or a model.
//! * **In Rust:** the control flow, the guards, the ledgers, and the elementwise arithmetic
//!   between those calls. Elementwise `+ - * /` on f64 is the same operation in both languages
//!   with no reassociation available, so the transcription is exact by construction — which is
//!   what lets the five-term theta-scheme right-hand side be a Rust fold over slices.
//!
//! The consequence is worth stating rather than discovering: with the sparse kernels still in
//! SciPy there is no arithmetic left for Rust to win, so this tier is expected to come out
//! **near neutral** on speed. That is section 11.6's rule (the win is per-call overhead) with the
//! calls still present, and it is the price of not emptying ten tests.
//!
//! ## Collaborators are looked up as module globals, exactly as the reference does
//!
//! `RoomLoadedPlate.__init__` calls the *module-global* `SurfacePort`, `_PlateSurface` and `splu`,
//! so these classes import `physsynth.core.airbox` and read the same three names at call time.
//! That is the faithful transcription, and it is also what makes the parity file able to build a
//! Rust wrapper over a **Python** port and seam (section 28.4's trap, queued a batch in advance by
//! section 31.11) and to put both languages' wrappers on one factorization (section 24.4).

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

// -- small shared helpers -------------------------------------------------------------------

/// `physsynth.core.airbox`, for the module-global collaborator names (see the module docstring).
fn airbox_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("physsynth.core.airbox")
}

/// A 1-D float64 array as a `Vec<f64>`.
///
/// The fast arm is a **downcast**, which costs no Python call at all; the slow one is
/// `np.ascontiguousarray(obj, float64)` for a list, a strided view or an integer array. That split
/// is worth its two lines: the reference reaches NumPy through operators that are already C, where
/// a port reaches it through the interpreter, so an import and a call per extraction is seven
/// import-and-calls per step and it is measurable (§32).
fn vec1(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        if let Ok(slice) = arr.readonly().as_slice() {
            return Ok(slice.to_vec());
        }
    }
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, np.getattr("float64")?))?;
    let arr: Bound<'_, PyArray1<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 1-D float64 array.")))?;
    let ro = arr.readonly();
    Ok(ro.as_slice()?.to_vec())
}

/// A fresh NumPy array from a `Vec`, as the object the next Python call will take.
fn pyarr(py: Python<'_>, v: Vec<f64>) -> Py<PyAny> {
    PyArray1::from_vec(py, v).into_any().unbind()
}

/// `np.zeros(n)`.
fn zeros(py: Python<'_>, n: usize) -> Py<PyAny> {
    pyarr(py, vec![0.0; n])
}

/// `float(np.dot(a, b))` — BLAS, and deliberately not transcribed: `ddot` fuses its multiply-add
/// and admits no scalar recipe (plan section 14.2).
fn np_dot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    py.import("numpy")?.call_method1("dot", (a, b))?.extract()
}

/// A float attribute off a collaborator, read live rather than cached — the reference reads
/// `p.theta`, `p.sigma`, `p.kappa` on every step, so a caller who mutates one mid-run sees the
/// same thing through either implementation.
fn f64_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

/// Whether the model's `boundary` is the supported branch.
fn is_supported(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.getattr("boundary")?.extract::<String>()? == "supported")
}

/// `np.isclose(a, b, rtol=1e-12, atol=0.0)` — `|a - b| <= rtol |b|`, NaN false either way.
fn same_rate(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs()
}

/// `_require_same_rate` — refuse a resonator whose timestep is not the room's, naming it as the
/// caller knows it. The message is the reference's, `fs` formatted through Python's own `str`.
fn require_same_rate(
    model: &Bound<'_, PyAny>,
    room: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<()> {
    if same_rate(f64_attr(model, "k")?, f64_attr(room, "k")?) {
        return Ok(());
    }
    let model_fs = model.getattr("fs")?.str()?;
    let room_fs = room.getattr("fs")?.str()?;
    Err(PyValueError::new_err(format!(
        "sample-rate mismatch: {label} fs = {model_fs} but room fs = {room_fs}. The port's solve \
         is a single timestep shared by both, so they must agree exactly."
    )))
}

/// The wrapper's `spreading` argument on its way to the port: omitted means this tier's own
/// default, `"bilinear"`; an explicit `None` is the caller's and the port refuses it.
///
/// The arm order is section 31.7's, which is section 24.7's: PyO3 wraps the DEFAULT expression, so
/// `Some(None)` is "argument omitted" and a bare `None` is what the caller wrote.
fn spreading_arg(py: Python<'_>, spreading: Option<Option<Py<PyAny>>>) -> Py<PyAny> {
    match spreading {
        Some(None) => PyString::new(py, "bilinear").into_any().unbind(),
        None => py.None(),
        Some(Some(o)) => o,
    }
}

/// `origin`, whose default *is* `None` in the reference, so omitted and explicit agree.
fn origin_arg(py: Python<'_>, origin: Option<Py<PyAny>>) -> Py<PyAny> {
    origin.unwrap_or_else(|| py.None())
}

// -- the seams ------------------------------------------------------------------------------

/// Shared by both surface adapters: the live-node coordinates a port spreads onto.
///
/// `np.column_stack((p.X[p.mask], p.Y[p.mask]))` — the model's own C-order over `mask`, which is
/// what makes `T`'s columns line up with the state vector by construction.
fn surface_coords(py: Python<'_>, model: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mask = model.getattr("mask")?;
    let x = model.getattr("X")?.get_item(&mask)?;
    let y = model.getattr("Y")?.get_item(&mask)?;
    let np = py.import("numpy")?;
    let pair = PyTuple::new(py, [x, y])?;
    Ok(np.call_method1("column_stack", (pair,))?.unbind())
}

/// `(1 + sigma k) I + theta k^2 kappa^2 B`, or its `W` form — the **unloaded** system matrix, in
/// the reference's own operand order. Shared verbatim by the two seams, which is what keeps
/// `nonlinear=False` a byte-exact reduction rather than two transcriptions kept in step by hand
/// (plan section 15.2, and section 28.4's answer to it one tier down).
fn assemble_a_bare(py: Python<'_>, model: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let k = f64_attr(model, "k")?;
    let sk = f64_attr(model, "sigma")? * k;
    let kappa = f64_attr(model, "kappa")?;
    let coeff = f64_attr(model, "theta")? * k * k * kappa * kappa;
    if is_supported(model)? {
        let n_live: usize = model.getattr("n_live")?.extract()?;
        let scipy = py.import("scipy.sparse")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("format", "csc")?;
        let eye = scipy.call_method("identity", (n_live,), Some(&kwargs))?;
        let b = model.getattr("B")?;
        return Ok(eye.mul(1.0 + sk)?.add(b.mul(coeff)?)?.unbind());
    }
    let w = model.getattr("W")?;
    let k_mat = model.getattr("K")?;
    Ok(w.mul(1.0 + sk)?.add(k_mat.mul(coeff)?)?.unbind())
}

/// The force-free theta-scheme right-hand side — `Plate.step`'s own arithmetic, in its own operand
/// order, as a fold over slices. The sparse products go through SciPy; everything between them is
/// elementwise and therefore exact in either language.
fn linear_rhs(py: Python<'_>, model: &Bound<'_, PyAny>, k: f64) -> PyResult<Vec<f64>> {
    let sk = f64_attr(model, "sigma")? * k;
    let k2 = k * k;
    let kappa = f64_attr(model, "kappa")?;
    let kappa2 = kappa * kappa;
    let theta = f64_attr(model, "theta")?;
    let c1 = (1.0 - 2.0 * theta) * k2;
    let c2 = theta * k2;

    let u_obj = model.getattr("u")?;
    let v_obj = model.getattr("u_prev")?;
    let u = vec1(py, &u_obj, "u")?;
    let v = vec1(py, &v_obj, "u_prev")?;

    if is_supported(model)? {
        let b = model.getattr("B")?;
        let bu = vec1(py, &b.matmul(&u_obj)?, "B @ u")?;
        let bv = vec1(py, &b.matmul(&v_obj)?, "B @ u_prev")?;
        return Ok((0..u.len())
            .map(|i| {
                // 2 u + c1 (-kappa^2 (B u)) - u^{n-1} + c2 (-kappa^2 (B u^{n-1})) + sk u^{n-1}
                let a = 2.0 * u[i] + c1 * (-kappa2 * bu[i]);
                let b = a - v[i];
                let c = b + c2 * (-kappa2 * bv[i]);
                c + sk * v[i]
            })
            .collect());
    }

    // The free branch: W is the lumped mass and K the energy-first stiffness, so the mass rides on
    // the left of three of the four terms and is divided out by the solve.
    let w = model.getattr("W")?;
    let k_mat = model.getattr("K")?;
    let pred: Vec<f64> = (0..u.len()).map(|i| 2.0 * u[i] - v[i]).collect();
    let wpred = vec1(
        py,
        &w.matmul(pyarr(py, pred).bind(py))?,
        "W @ (2u - u_prev)",
    )?;
    let ku = vec1(py, &k_mat.matmul(&u_obj)?, "K @ u")?;
    let kv = vec1(py, &k_mat.matmul(&v_obj)?, "K @ u_prev")?;
    let wv = vec1(py, &w.matmul(&v_obj)?, "W @ u_prev")?;
    Ok((0..u.len())
        .map(|i| {
            let a = wpred[i] + c1 * (-kappa2 * ku[i]);
            let b = a + c2 * (-kappa2 * kv[i]);
            b + sk * wv[i]
        })
        .collect())
}

/// `rhs + k^2 f_ext / denominator` — the external-force path, added the way the model's own step
/// adds it, so a zero air load is a clean reduction to the bare model.
fn add_f_ext(
    py: Python<'_>,
    rhs: Vec<f64>,
    f_ext: Option<&Bound<'_, PyAny>>,
    k2: f64,
    denominator: f64,
) -> PyResult<Vec<f64>> {
    let Some(f) = f_ext else { return Ok(rhs) };
    if f.is_none() {
        return Ok(rhs);
    }
    let f = vec1(py, f, "f_ext")?;
    if f.len() != rhs.len() {
        return Err(PyValueError::new_err(format!(
            "f_ext must have shape ({},), got ({},).",
            rhs.len(),
            f.len()
        )));
    }
    Ok((0..rhs.len())
        .map(|i| rhs[i] + k2 * f[i] / denominator)
        .collect())
}

/// The seam between a room port and the grid resonator it loads — the `Plate` side.
///
/// The reference's docstring on `physsynth.core.airbox._PlateSurface` is the specification; the
/// contract is the six members, and `u_prev` is a **live read** the caller must take before
/// `commit`.
#[pyclass(dict, name = "_PlateSurface", module = "physsynth_rs")]
pub struct PyPlateSurface {
    model: Py<PyAny>,
    k: f64,
    areas: Py<PyAny>,
    denominator: f64,
}

#[pymethods]
impl PyPlateSurface {
    #[new]
    fn new(py: Python<'_>, plate: Py<PyAny>) -> PyResult<Self> {
        let p = plate.bind(py);
        let k = f64_attr(p, "k")?;
        let (areas, denominator) = if is_supported(p)? {
            let n_live: usize = p.getattr("n_live")?.extract()?;
            let h = f64_attr(p, "h")?;
            // Per-node mass rho_s h^2: Plate.step divides f_ext by exactly this.
            (pyarr(py, vec![h * h; n_live]), f64_attr(p, "rho")? * h * h)
        } else {
            // Lumped cell areas (h^2, h^2/2, h^2/4) -- no dead rim. W lives inside A and is
            // divided out by the solve, so the denominator is the areal density alone.
            let w = p.getattr("w")?.call_method0("copy")?;
            (w.unbind(), f64_attr(p, "rho")?)
        };
        Ok(Self {
            model: plate,
            k,
            areas,
            denominator,
        })
    }

    #[getter]
    fn model(&self, py: Python<'_>) -> Py<PyAny> {
        self.model.clone_ref(py)
    }

    #[getter]
    fn k(&self) -> f64 {
        self.k
    }

    #[getter]
    fn areas(&self, py: Python<'_>) -> Py<PyAny> {
        self.areas.clone_ref(py)
    }

    #[getter]
    fn denominator(&self) -> f64 {
        self.denominator
    }

    /// `(coords, areas)` — live-node positions (m) and their areas (m^2), in model order.
    fn surface(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let coords = surface_coords(py, self.model.bind(py))?;
        Ok(PyTuple::new(py, [coords, self.areas.clone_ref(py)])?
            .into_any()
            .unbind())
    }

    /// The unloaded system matrix.
    fn a_bare(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        assemble_a_bare(py, self.model.bind(py))
    }

    /// `u^{n-1}` — read once per step, before `commit`.
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.model.bind(py).getattr("u_prev")?.unbind())
    }

    /// The force-free theta-scheme RHS plus the `f_ext` path — the room's terms are not here.
    #[pyo3(signature = (f_ext))]
    fn rhs(&self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let model = self.model.bind(py);
        let rhs = linear_rhs(py, model, self.k)?;
        let bound = f_ext.map(|f| f.into_bound(py));
        let rhs = add_f_ext(py, rhs, bound.as_ref(), self.k * self.k, self.denominator)?;
        Ok(pyarr(py, rhs))
    }

    /// Roll `u^{n-1} <- u^n <- u^{n+1}` and refresh the plate's acceleration cache.
    fn commit(&self, py: Python<'_>, u_next: Py<PyAny>) -> PyResult<()> {
        let model = self.model.bind(py);
        let next = u_next.bind(py);
        let u_obj = model.getattr("u")?;
        let v_obj = model.getattr("u_prev")?;
        let u = vec1(py, &u_obj, "u")?;
        let v = vec1(py, &v_obj, "u_prev")?;
        let un = vec1(py, next, "u_next")?;
        let k2 = self.k * self.k;
        let accel: Vec<f64> = (0..u.len())
            .map(|i| (un[i] - 2.0 * u[i] + v[i]) / k2)
            .collect();
        model.setattr("_accel", pyarr(py, accel))?;
        model.setattr("u_prev", u_obj)?;
        model.setattr("u", next)?;
        let n: i64 = model.getattr("n")?.extract()?;
        model.setattr("n", n + 1)?;
        Ok(())
    }
}

/// The seam on the `Membrane` side — model #4, and three differences from the plate, each of them a
/// fact about the model rather than a convenience (the reference's docstring is the specification).
///
/// * **The mass is uniform.** Every live node carries `rho h^2`; there is no lumped `W` and no
///   free-boundary branch, because a membrane's rim is clamped and dead. So `denominator` is one
///   number and the load's `(k / 2 rho h^2) T^T R T` is a plain scaling.
/// * **There is no `_accel`** to refresh, so `commit` is the two-level roll and nothing else.
/// * **`rhs`'s `f_ext` term has no counterpart in the model.** `Plate.step` has its own `f_ext`
///   path, so the plate seam's copy of it can be checked against the original; `Membrane.step()`
///   takes no force at all. That term is therefore arithmetic with nothing in the model to be
///   bit-identical *to*, and the parity file has to drive it deliberately -- a run that only ever
///   passes `f_ext=None` compares the shared half twice and never reaches this one.
#[pyclass(dict, name = "_MembraneSurface", module = "physsynth_rs")]
pub struct PyMembraneSurface {
    model: Py<PyAny>,
    k: f64,
    areas: Py<PyAny>,
    denominator: f64,
}

#[pymethods]
impl PyMembraneSurface {
    #[new]
    fn new(py: Python<'_>, membrane: Py<PyAny>) -> PyResult<Self> {
        let m = membrane.bind(py);
        let k = f64_attr(m, "k")?;
        let n_live: usize = m.getattr("n_live")?.extract()?;
        let h = f64_attr(m, "h")?;
        // Per-node mass rho h^2, uniform: the clamped rim is dead, not lightly weighted. Both
        // spellings are the reference's own left-to-right fold -- `(rho * h) * h`, and `h * h`
        // rather than a squaring call, which is section 23.4's hazard and section 17.2's.
        let areas = pyarr(py, vec![h * h; n_live]);
        let denominator = f64_attr(m, "rho")? * h * h;
        Ok(Self {
            model: membrane,
            k,
            areas,
            denominator,
        })
    }

    #[getter]
    fn model(&self, py: Python<'_>) -> Py<PyAny> {
        self.model.clone_ref(py)
    }

    #[getter]
    fn k(&self) -> f64 {
        self.k
    }

    #[getter]
    fn areas(&self, py: Python<'_>) -> Py<PyAny> {
        self.areas.clone_ref(py)
    }

    #[getter]
    fn denominator(&self) -> f64 {
        self.denominator
    }

    /// `(coords, areas)` — the **live** nodes only, which is the moving surface: a membrane's rim
    /// is clamped and dead, so the radiating surface is one cell inside the nominal boundary.
    fn surface(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let coords = surface_coords(py, self.model.bind(py))?;
        Ok(PyTuple::new(py, [coords, self.areas.clone_ref(py)])?
            .into_any()
            .unbind())
    }

    /// `(1 + sigma k) I` — and its emptiness is the tier's main design fact: model #4 is a **pure
    /// explicit** update, and the air load's unknown is `u^{n+1}`, so putting the load in `A` buys
    /// `radiated == injected` as an identity at the price of a factorization the model never had.
    fn a_bare(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let m = self.model.bind(py);
        let sk = f64_attr(m, "sigma")? * f64_attr(m, "k")?;
        let n_live: usize = m.getattr("n_live")?.extract()?;
        let scipy = py.import("scipy.sparse")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("format", "csc")?;
        let eye = scipy.call_method("identity", (n_live,), Some(&kwargs))?;
        Ok(eye.mul(1.0 + sk)?.unbind())
    }

    /// `u^{n-1}` — read once per step, before `commit` (see the plate seam).
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.model.bind(py).getattr("u_prev")?.unbind())
    }

    /// `2 u^n - (1 - sigma k) u^{n-1} + c^2 k^2 L u^n`, plus `k^2 f_ext / rho h^2`.
    ///
    /// The first part is `Membrane.step`'s own numerator in its own operand order, so a zero air
    /// load and no force reduce to the bare membrane exactly.
    #[pyo3(signature = (f_ext))]
    fn rhs(&self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let m = self.model.bind(py);
        let sk = f64_attr(m, "sigma")? * self.k;
        let c = f64_attr(m, "c")?;
        // `m.c * m.c * self.k * self.k` left to right: ((c c) k) k, NOT (c k) squared.
        let c2k2 = c * c * self.k * self.k;

        let u_obj = m.getattr("u")?;
        let v_obj = m.getattr("u_prev")?;
        let u = vec1(py, &u_obj, "u")?;
        let v = vec1(py, &v_obj, "u_prev")?;
        let lu = vec1(py, &m.getattr("L")?.matmul(&u_obj)?, "L @ u")?;
        let rhs: Vec<f64> = (0..u.len())
            .map(|i| (2.0 * u[i] - (1.0 - sk) * v[i]) + c2k2 * lu[i])
            .collect();

        let bound = f_ext.map(|f| f.into_bound(py));
        let rhs = add_f_ext(py, rhs, bound.as_ref(), self.k * self.k, self.denominator)?;
        Ok(pyarr(py, rhs))
    }

    /// Roll `u^{n-1} <- u^n <- u^{n+1}`. There is no acceleration cache to refresh, which is the
    /// same gap that leaves `Membrane` without a `pressure()` read-out.
    fn commit(&self, py: Python<'_>, u_next: Py<PyAny>) -> PyResult<()> {
        let m = self.model.bind(py);
        let u_obj = m.getattr("u")?;
        m.setattr("u_prev", u_obj)?;
        m.setattr("u", u_next)?;
        let n: i64 = m.getattr("n")?.extract()?;
        m.setattr("n", n + 1)?;
        Ok(())
    }
}

/// The seam on the `VKPlate` side — the linear half is the model's own `_linear_rhs`, and the
/// Picard loop is `VKPlate.step`'s arithmetic with the room's terms held fixed.
#[pyclass(dict, name = "_VKPlateSurface", module = "physsynth_rs")]
pub struct PyVKPlateSurface {
    model: Py<PyAny>,
    k: f64,
    areas: Py<PyAny>,
    denominator: f64,
}

#[pymethods]
impl PyVKPlateSurface {
    #[new]
    fn new(py: Python<'_>, plate: Py<PyAny>) -> PyResult<Self> {
        let p = plate.bind(py);
        let k = f64_attr(p, "k")?;
        let areas = if is_supported(p)? {
            let n_live: usize = p.getattr("n_live")?.extract()?;
            let h = f64_attr(p, "h")?;
            pyarr(py, vec![h * h; n_live])
        } else {
            p.getattr("wdiag")?.call_method0("copy")?.unbind()
        };
        // Read off the model rather than recomputed, so this and VKPlate.step()'s own f_ext path
        // are the same expression by construction -- AREAL density, never rho_v.
        let denominator = f64_attr(p, "force_denominator")?;
        Ok(Self {
            model: plate,
            k,
            areas,
            denominator,
        })
    }

    #[getter]
    fn model(&self, py: Python<'_>) -> Py<PyAny> {
        self.model.clone_ref(py)
    }

    #[getter]
    fn k(&self) -> f64 {
        self.k
    }

    #[getter]
    fn areas(&self, py: Python<'_>) -> Py<PyAny> {
        self.areas.clone_ref(py)
    }

    #[getter]
    fn denominator(&self) -> f64 {
        self.denominator
    }

    fn surface(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let coords = surface_coords(py, self.model.bind(py))?;
        Ok(PyTuple::new(py, [coords, self.areas.clone_ref(py)])?
            .into_any()
            .unbind())
    }

    fn a_bare(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        assemble_a_bare(py, self.model.bind(py))
    }

    #[getter]
    fn u_prev(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.model.bind(py).getattr("u_prev")?.unbind())
    }

    /// The model's own linear theta-scheme RHS, plus the `f_ext` path.
    ///
    /// The linear half is `VKPlate._linear_rhs` itself, so it is bit-identical to the model by
    /// construction rather than by inspection — and, because that method and `_PlateSurface.rhs`
    /// are the same expression in the same operand order, bit-identical to the linear seam's as
    /// well, which is what makes the `nonlinear=False` regression a byte-exact claim.
    #[pyo3(signature = (f_ext))]
    fn rhs(&self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let model = self.model.bind(py);
        let rhs = model.call_method0("_linear_rhs")?;
        let bound = f_ext.map(|f| f.into_bound(py));
        let Some(f) = bound.as_ref().filter(|f| !f.is_none()) else {
            return Ok(rhs.unbind());
        };
        let base = vec1(py, &rhs, "_linear_rhs()")?;
        let out = add_f_ext(py, base, Some(f), self.k * self.k, self.denominator)?;
        Ok(pyarr(py, out))
    }

    /// One step's solve — `(w^{n+1}, F^{n+1})`, Picard-iterated on the **loaded** operator.
    ///
    /// `lu` is an argument and is never assigned to the model; `F^{n+1}` comes back as a return
    /// value; and the model's `n_iters` / `converged` / `last_residual` are written here exactly as
    /// its own step writes them.
    fn solve(&self, py: Python<'_>, lu: Py<PyAny>, rhs_fixed: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let p = self.model.bind(py);
        let lu = lu.bind(py);
        let rhs_fixed = rhs_fixed.bind(py);

        if !p.getattr("nonlinear")?.extract::<bool>()? {
            let w_next = lu.call_method1("solve", (rhs_fixed,))?;
            p.setattr("n_iters", 1)?;
            p.setattr("converged", true)?;
            p.setattr("last_residual", 0.0)?;
            // F and F_prev are both zeros on this path and nothing writes them, so handing F back
            // makes commit()'s roll a structural no-op rather than a coincidental one.
            let f = p.getattr("F")?;
            return Ok(PyTuple::new(py, [w_next, f])?.into_any().unbind());
        }

        let k2 = self.k * self.k;
        // Coupling force -> RHS factor, the model's own: "supported" k^2 l/rho_s (scalar mass);
        // "free" k^2 h^2 l/rho_s (A carries W's h^2, and the /W is the solve's, not ours).
        let mut couple_factor = k2 / f64_attr(p, "rho_s")?;
        if !is_supported(p)? {
            let h = f64_attr(p, "h")?;
            couple_factor *= h * h;
        }
        let w_prev_full = p.call_method1("_to_full", (p.getattr("u_prev")?,))?;
        let f_prev_full = p.getattr("F_prev")?;

        let u = vec1(py, &p.getattr("u")?, "u")?;
        let v = vec1(py, &p.getattr("u_prev")?, "u_prev")?;
        // Predictor w^{n+1}_(0).
        let mut w_j: Py<PyAny> = pyarr(
            py,
            (0..u.len())
                .map(|i| 2.0 * u[i] - v[i])
                .collect::<Vec<f64>>(),
        );
        let mut f_new_full: Py<PyAny> = p.getattr("F")?.unbind(); // fallback, unused once the loop runs

        let tol = f64_attr(p, "couple_tol")?;
        let max_iter: usize = p.getattr("couple_max_iter")?.extract()?;
        p.setattr("n_iters", 0)?;
        p.setattr("converged", false)?;
        let np = py.import("numpy")?;
        let norm = np.getattr("linalg")?.getattr("norm")?;

        for sweep in 1..=max_iter {
            p.setattr("n_iters", sweep)?;
            let w_j_full = p.call_method1("_to_full", (w_j.clone_ref(py),))?;
            let f_new = p.call_method1("_airy_F", (&w_j_full,))?; // F^{n+1}_(j)
                                                                  // mu_{t.} w and mu_{t.} F. Kept as NumPy objects rather than folded in Rust: this loop
                                                                  // crosses the language boundary once per sweep already, and a copy in and a copy out
                                                                  // per average is what turned the von Karman wrapper from 1.09x into 0.91x (§32).
            let w_avg = w_j_full.add(&w_prev_full)?.mul(0.5)?;
            let f_avg = f_new.add(&f_prev_full)?.mul(0.5)?;
            f_new_full = f_new.unbind();
            let bracket = p.call_method1("bracket", (w_avg, f_avg))?;
            let coupling = p.call_method1("_to_live", (bracket,))?;
            let rhs = rhs_fixed.add(coupling.mul(couple_factor)?)?;
            let w_next = lu.call_method1("solve", (rhs,))?;
            let diff = w_next.sub(w_j.bind(py))?;
            let incr: f64 = norm.call1((diff,))?.extract()?;
            let scale: f64 = norm.call1((&w_next,))?.extract()?;
            w_j = w_next.unbind();
            // Python's `max` returns its first argument when the comparison is false, which is
            // what a NaN scale would hit; f64::max would hand back the floor instead.
            let denom = if 1e-30 > scale { 1e-30 } else { scale };
            let residual = incr / denom;
            p.setattr("last_residual", residual)?;
            if residual <= tol {
                p.setattr("converged", true)?;
                break;
            }
        }
        Ok(PyTuple::new(py, [w_j, f_new_full])?.into_any().unbind())
    }

    /// Roll **both** histories: `w^{n-1} <- w^n <- w^{n+1}` and `F^{n-1} <- F^n <- F^{n+1}`.
    fn commit(&self, py: Python<'_>, u_next: Py<PyAny>, f_next: Py<PyAny>) -> PyResult<()> {
        let p = self.model.bind(py);
        let f_old = p.getattr("F")?;
        p.setattr("F_prev", f_old)?;
        p.setattr("F", f_next)?;
        let u_old = p.getattr("u")?;
        p.setattr("u_prev", u_old)?;
        p.setattr("u", u_next)?;
        let n: i64 = p.getattr("n")?.extract()?;
        p.setattr("n", n + 1)?;
        Ok(())
    }
}

// -- the wrappers ---------------------------------------------------------------------------

/// Which port tier a plate wrapper is on: one loaded face, or two faces of an interior plane.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    /// `SurfacePort` — one face, `free_pressure()` a vector, the ledger `surface_pressure`.
    Baffled,
    /// `InteriorSurfacePort` — two faces, `free_pressure()` a pair, the ledger `pressure_jump`.
    Suspended,
}

/// Everything the six grid wrappers hold. The differences between them are two enum arms, a
/// ledger name and the model's own attribute name; the arithmetic is one function.
struct Wrap {
    model: Py<PyAny>,
    /// What the wrapper calls the resonator it holds -- "plate" or "membrane". It is three things
    /// at once and they must agree: the `#[getter]` the class exposes, the label
    /// `_require_same_rate` puts in its message, and the name `__getattr__` refuses to delegate so
    /// the lookup cannot recurse. Getting the third wrong takes the wrapper's own model away from
    /// it, silently: the miss falls through to a delegation the model itself cannot answer.
    model_name: &'static str,
    room: Py<PyAny>,
    k: f64,
    surface: Py<PyAny>,
    denominator: f64,
    port: Py<PyAny>,
    load_scale: f64,
    nnz_growth: f64,
    lu_loaded: Py<PyAny>,
    lu_nnz: i64,
    radiated_energy: f64,
    nodal_volume_velocity: Py<PyAny>,
    /// `surface_pressure` on the baffled tier, `pressure_jump` on the suspended one.
    pbar: Py<PyAny>,
    volume_velocity: f64,
    n: i64,
    tier: Tier,
}

/// How many numbers the port's ledgers carry: `node_count` on a face, `face_count` on a plane.
fn ledger_len(py: Python<'_>, port: &Py<PyAny>, tier: Tier) -> PyResult<usize> {
    let name = match tier {
        Tier::Baffled => "node_count",
        Tier::Suspended => "face_count",
    };
    port.bind(py).getattr(name)?.extract()
}

/// The shared constructor: seam, port, loaded matrix, factorization and empty ledgers.
///
/// `seam_name` and `port_name` are read as module globals off `physsynth.core.airbox`, so the
/// wrapper builds whatever implementation that module currently exposes — which is the reference's
/// own behaviour and is what section 28.4 says a parity file must then override explicitly.
#[allow(clippy::too_many_arguments)]
fn build(
    py: Python<'_>,
    model: Py<PyAny>,
    model_name: &'static str,
    room: Py<PyAny>,
    seam_name: &str,
    port_name: &str,
    port_kwargs: &Bound<'_, PyDict>,
    tier: Tier,
) -> PyResult<Wrap> {
    let p = model.bind(py);
    require_same_rate(p, room.bind(py), model_name)?;
    let k = f64_attr(p, "k")?;
    let module = airbox_module(py)?;

    let surface = module.getattr(seam_name)?.call1((model.clone_ref(py),))?;
    let pair = surface.call_method0("surface")?;
    let coords = pair.get_item(0)?;
    let areas = pair.get_item(1)?;
    let denominator: f64 = surface.getattr("denominator")?.extract()?;

    port_kwargs.set_item("room", room.clone_ref(py))?;
    port_kwargs.set_item("coords", coords)?;
    port_kwargs.set_item("areas", areas)?;
    let port = module.getattr(port_name)?.call((), Some(port_kwargs))?;

    // A_loaded = A + (k / 2 rho_s) T^T R T -- SPD (PSD added to SPD), factored ONCE.
    let a_bare = surface.call_method0("a_bare")?;
    let load_scale = 0.5 * k / denominator;
    let load_matrix = port.getattr("load_matrix")?;
    let a_loaded = a_bare
        .add(load_matrix.mul(load_scale)?)?
        .call_method0("tocsc")?;
    // Drop the structural zeros the load's sparsity pattern contributes where its value is 0, so a
    // zero-area surface (T = 0) factors the plate's OWN matrix and reduces to the bare plate.
    a_loaded.call_method0("eliminate_zeros")?;
    let loaded_nnz: f64 = a_loaded.getattr("nnz")?.extract()?;
    let bare_nnz: f64 = a_bare.call_method0("tocsc")?.getattr("nnz")?.extract()?;
    let lu_loaded = module.getattr("splu")?.call1((&a_loaded,))?;
    let l_nnz: i64 = lu_loaded.getattr("L")?.getattr("nnz")?.extract()?;
    let u_nnz: i64 = lu_loaded.getattr("U")?.getattr("nnz")?.extract()?;

    let port = port.unbind();
    let n_ledger = ledger_len(py, &port, tier)?;
    Ok(Wrap {
        model,
        model_name,
        room,
        k,
        surface: surface.unbind(),
        denominator,
        port,
        load_scale,
        nnz_growth: loaded_nnz / bare_nnz,
        lu_loaded: lu_loaded.unbind(),
        lu_nnz: l_nnz + u_nnz,
        radiated_energy: 0.0,
        nodal_volume_velocity: zeros(py, n_ledger),
        pbar: zeros(py, n_ledger),
        volume_velocity: 0.0,
        n: 0,
        tier,
    })
}

impl Wrap {
    /// Advance one step: read the port, solve the **loaded** system, queue the injection.
    ///
    /// `nonlinear` picks the seam's solve: the linear seam returns `w^{n+1}` alone and commits it,
    /// the von Karman seam Picard-iterates and returns `(w^{n+1}, F^{n+1})`.
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>, von_karman: bool) -> PyResult<()> {
        let port = self.port.bind(py);
        port.call_method0("require_ready")?; // before mutating anything
        let free = port.call_method0("free_pressure")?; // read u^{n+1/2}, BEFORE room.step()
                                                        // The baffled tier reads one pressure per node; the suspended tier reads a pair and works
                                                        // with the JUMP across the plane.
        let free = match self.tier {
            Tier::Baffled => free,
            Tier::Suspended => {
                let lo = free.get_item(0)?;
                let hi = free.get_item(1)?;
                hi.sub(lo)?
            }
        };

        let k2 = self.k * self.k;
        let surface = self.surface.bind(py);
        // ONCE, and before commit() -- see the seam's docstring.
        let u_nm1 = surface.getattr("u_prev")?;
        let rhs = surface.call_method1("rhs", (f_ext,))?;

        // The air load: the known open-circuit term, plus the u^{n-1} half of the centered
        // velocity (its u^{n+1} half is already inside the factorization).
        let t = port.getattr("T")?;
        let load = t.getattr("T")?.matmul(&free)?;
        let load = vec1(py, &load, "T.T @ pbar_free")?;
        let carry = port.getattr("load_matrix")?.matmul(&u_nm1)?;
        let carry = vec1(py, &carry, "load_matrix @ u^{n-1}")?;
        let base = vec1(py, &rhs, "rhs")?;
        let rhs: Vec<f64> = (0..base.len())
            .map(|i| base[i] - k2 * load[i] / self.denominator + self.load_scale * carry[i])
            .collect();
        let rhs = pyarr(py, rhs);

        let lu = self.lu_loaded.bind(py);
        let u_next = if von_karman {
            let pair = surface.call_method1("solve", (lu, rhs))?;
            let u_next = pair.get_item(0)?;
            let f_next = pair.get_item(1)?;
            surface.call_method1("commit", (&u_next, f_next))?;
            u_next
        } else {
            let u_next = lu.call_method1("solve", (rhs,))?;
            // _accel already carries the load -- it was IN the solve, so no post-solve refresh.
            surface.call_method1("commit", (&u_next,))?;
            u_next
        };

        let prev = vec1(py, &u_nm1, "u^{n-1}")?;
        let next = vec1(py, &u_next, "u^{n+1}")?;
        let half = 2.0 * self.k;
        let vel: Vec<f64> = (0..next.len())
            .map(|i| (next[i] - prev[i]) / half)
            .collect();
        let q = t.matmul(pyarr(py, vel).bind(py))?;
        let r = vec1(py, &port.getattr("R")?, "R")?;
        let free_v = vec1(py, &free, "pbar_free")?;
        let q_v = vec1(py, &q, "q")?;
        // Baffled: pbar = pbar_free + R q. Suspended: d_pbar = d_free + 2 R q, and the 2 is the
        // two loaded faces -- the operand order is the reference's, (2 R) q rather than 2 (R q).
        let pbar: Vec<f64> = match self.tier {
            Tier::Baffled => (0..q_v.len()).map(|i| free_v[i] + r[i] * q_v[i]).collect(),
            Tier::Suspended => (0..q_v.len())
                .map(|i| free_v[i] + (2.0 * r[i]) * q_v[i])
                .collect(),
        };
        let pbar = pyarr(py, pbar);
        port.call_method1("inject", (&q,))?;
        self.radiated_energy += self.k * np_dot(py, pbar.bind(py), &q)?;
        self.volume_velocity = physsynth_core::reduce::sum(&q_v);
        self.nodal_volume_velocity = q.unbind();
        self.pbar = pbar;
        self.n += 1;
        Ok(())
    }

    /// Total discrete energy `E_plate + integral pbar . q dt` (Joules) — an explicit override, not
    /// a delegation, because the delegated number is the total *without* its coupling channel.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let e: f64 = self.model.bind(py).call_method0("energy")?.extract()?;
        Ok(e + self.radiated_energy)
    }

    fn reset_books(&mut self, py: Python<'_>) -> PyResult<()> {
        let n_ledger = ledger_len(py, &self.port, self.tier)?;
        self.radiated_energy = 0.0;
        self.nodal_volume_velocity = zeros(py, n_ledger);
        self.pbar = zeros(py, n_ledger);
        self.volume_velocity = 0.0;
        self.port.bind(py).call_method0("reset")?;
        self.n = 0;
        Ok(())
    }

    /// Delegate a read accessor to the plate, so a wrapper is a drop-in wherever a bare model is
    /// expected -- notably `StringPlateBridge`, which reassembles the plate's `G0` block out of
    /// exactly those. NOTHING exposed above may shadow a name that bridge reads.
    fn getattr(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if name == self.model_name {
            // Nothing to delegate through yet -- never recurse.
            return Err(PyAttributeError::new_err(name.to_string()));
        }
        Ok(self.model.bind(py).getattr(name)?.unbind())
    }
}

/// Generate the shared `#[pymethods]` surface of a grid wrapper: the state it sets on itself, the
/// four overrides and the delegation. Written once because the reference's six classes are one
/// class with two enum arms, and because an `array_equal` anchor across them (section 15.2) means
/// a difference between two transcriptions would be a failure rather than a divergence.
///
/// `$model` is the attribute the wrapper calls its resonator, and it MUST be the same string
/// `build` was handed: the getter emitted here and `Wrap::getattr`'s recursion guard are two
/// halves of one decision (see `Wrap::model_name`).
macro_rules! grid_wrapper {
    ($cls:ident, $model:ident, $ledger:ident, { $($extra:item)* }) => {
        #[pymethods]
        impl $cls {
            #[getter]
            fn $model(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.model.clone_ref(py)
            }

            #[getter]
            fn room(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.room.clone_ref(py)
            }

            #[getter]
            fn port(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.port.clone_ref(py)
            }

            #[getter]
            fn k(&self) -> f64 {
                self.w.k
            }

            #[getter]
            fn _surface(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.surface.clone_ref(py)
            }

            #[getter]
            fn _denominator(&self) -> f64 {
                self.w.denominator
            }

            #[getter]
            fn _load_scale(&self) -> f64 {
                self.w.load_scale
            }

            #[getter]
            fn nnz_growth(&self) -> f64 {
                self.w.nnz_growth
            }

            /// Settable: three tests refactor a rescaled load and hand the wrapper the result.
            #[getter]
            fn _lu_loaded(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.lu_loaded.clone_ref(py)
            }

            #[setter]
            fn set__lu_loaded(&mut self, value: Py<PyAny>) {
                self.w.lu_loaded = value;
            }

            #[getter]
            fn lu_nnz(&self) -> i64 {
                self.w.lu_nnz
            }

            #[getter]
            fn radiated_energy(&self) -> f64 {
                self.w.radiated_energy
            }

            #[setter]
            fn set_radiated_energy(&mut self, value: f64) {
                self.w.radiated_energy = value;
            }

            #[getter]
            fn nodal_volume_velocity(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.nodal_volume_velocity.clone_ref(py)
            }

            #[getter]
            fn $ledger(&self, py: Python<'_>) -> Py<PyAny> {
                self.w.pbar.clone_ref(py)
            }

            #[getter]
            fn volume_velocity(&self) -> f64 {
                self.w.volume_velocity
            }

            #[getter]
            fn n(&self) -> i64 {
                self.w.n
            }

            /// Settable, and the reference gets that for free by being Python. A `#[getter]` with
            /// no `#[setter]` is a data descriptor whose `__set__` raises, so a plain attribute
            /// the reference lets a caller advance becomes read-only the moment it is ported --
            /// and `test_airbox_membrane.py`'s hand-rolled lagged-velocity control does exactly
            /// that (`inst.n += 1`) while it drives the seam itself.
            #[setter]
            fn set_n(&mut self, value: i64) {
                self.w.n = value;
            }

            fn energy(&self, py: Python<'_>) -> PyResult<f64> {
                self.w.energy(py)
            }

            /// Set the model's initial state and reset this port's coupling ledger to zero.
            ///
            /// `v0`'s default is `0.0`, so an omitted argument and an explicit `None` are NOT the
            /// same call — section 24.7's arm order, and PyO3 wraps the default expression, so
            /// `Some(None)` is the omitted one.
            #[pyo3(signature = (u0, v0=None::<Py<PyAny>>))]
            fn set_state(
                &mut self,
                py: Python<'_>,
                u0: Py<PyAny>,
                v0: Option<Option<Py<PyAny>>>,
            ) -> PyResult<()> {
                let model = self.w.model.bind(py);
                match v0 {
                    Some(None) => model.call_method1("set_state", (u0, 0.0))?,
                    None => model.call_method1("set_state", (u0, py.None()))?,
                    Some(Some(v)) => model.call_method1("set_state", (u0, v))?,
                };
                self.w.reset_books(py)
            }

            /// Zero the model state and the coupling ledger — reuse on a new run.
            fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
                let model = self.w.model.bind(py);
                let n_live: usize = model.getattr("n_live")?.extract()?;
                model.call_method1("set_state", (zeros(py, n_live),))?;
                self.w.reset_books(py)
            }

            fn _reset_books(&mut self, py: Python<'_>) -> PyResult<()> {
                self.w.reset_books(py)
            }

            fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
                self.w.getattr(py, name)
            }

            $($extra)*
        }
    };
}

/// A `Plate` **loaded by a room** through one face of it — the baffled tier.
#[pyclass(dict, name = "RoomLoadedPlate", module = "physsynth_rs")]
pub struct PyRoomLoadedPlate {
    w: Wrap,
}

grid_wrapper!(PyRoomLoadedPlate, plate, surface_pressure, {
    #[new]
    #[pyo3(signature = (*, plate, room, face, origin=None, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        plate: Py<PyAny>,
        room: Py<PyAny>,
        face: Py<PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("face", face)?;
        kwargs.set_item("origin", origin_arg(py, origin))?;
        kwargs.set_item("spreading", spreading_arg(py, spreading))?;
        Ok(Self {
            w: build(
                py,
                plate,
                "plate",
                room,
                "_PlateSurface",
                "SurfacePort",
                &kwargs,
                Tier::Baffled,
            )?,
        })
    }

    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<()> {
        self.w.step(py, f_ext, false)
    }

    /// The plate's **monopole** read-out, reflecting the load — right for free, because the load
    /// was inside the solve and `_accel` carries it with no post-solve refresh.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.w.model.bind(py).call_method0("pressure")?.extract()
    }
});

/// A `Plate` suspended **inside** a room on an interior plane — the dipole tier.
#[pyclass(dict, name = "RoomSuspendedPlate", module = "physsynth_rs")]
pub struct PyRoomSuspendedPlate {
    w: Wrap,
}

grid_wrapper!(PyRoomSuspendedPlate, plate, pressure_jump, {
    #[new]
    #[pyo3(signature = (*, plate, room, plane, index, origin=None, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        plate: Py<PyAny>,
        room: Py<PyAny>,
        plane: Py<PyAny>,
        index: Py<PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("plane", plane)?;
        kwargs.set_item("index", index)?;
        kwargs.set_item("origin", origin_arg(py, origin))?;
        kwargs.set_item("spreading", spreading_arg(py, spreading))?;
        Ok(Self {
            w: build(
                py,
                plate,
                "plate",
                room,
                "_PlateSurface",
                "InteriorSurfacePort",
                &kwargs,
                Tier::Suspended,
            )?,
        })
    }

    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<()> {
        self.w.step(py, f_ext, false)
    }

    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.w.model.bind(py).call_method0("pressure")?.extract()
    }
});

/// A `VKPlate` loaded through one face — the nonlinear plate on the baffled tier.
///
/// Note what is deliberately absent: `pressure()`. Model #6 has none, and that absence is
/// load-bearing rather than incidental — `StringVKPlateBridge` composes with this class precisely
/// because the bridge batch supplied the two things model #6 lacked and *refused* the third.
#[pyclass(dict, name = "RoomLoadedVKPlate", module = "physsynth_rs")]
pub struct PyRoomLoadedVKPlate {
    w: Wrap,
}

grid_wrapper!(PyRoomLoadedVKPlate, plate, surface_pressure, {
    #[new]
    #[pyo3(signature = (*, plate, room, face, origin=None, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        plate: Py<PyAny>,
        room: Py<PyAny>,
        face: Py<PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("face", face)?;
        kwargs.set_item("origin", origin_arg(py, origin))?;
        kwargs.set_item("spreading", spreading_arg(py, spreading))?;
        Ok(Self {
            w: build(
                py,
                plate,
                "plate",
                room,
                "_VKPlateSurface",
                "SurfacePort",
                &kwargs,
                Tier::Baffled,
            )?,
        })
    }

    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<()> {
        self.w.step(py, f_ext, true)
    }
});

/// A `VKPlate` suspended inside a room on an interior plane — the free cymbal.
#[pyclass(dict, name = "RoomSuspendedVKPlate", module = "physsynth_rs")]
pub struct PyRoomSuspendedVKPlate {
    w: Wrap,
}

grid_wrapper!(PyRoomSuspendedVKPlate, plate, pressure_jump, {
    #[new]
    #[pyo3(signature = (*, plate, room, plane, index, origin=None, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        plate: Py<PyAny>,
        room: Py<PyAny>,
        plane: Py<PyAny>,
        index: Py<PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("plane", plane)?;
        kwargs.set_item("index", index)?;
        kwargs.set_item("origin", origin_arg(py, origin))?;
        kwargs.set_item("spreading", spreading_arg(py, spreading))?;
        Ok(Self {
            w: build(
                py,
                plate,
                "plate",
                room,
                "_VKPlateSurface",
                "InteriorSurfacePort",
                &kwargs,
                Tier::Suspended,
            )?,
        })
    }

    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<()> {
        self.w.step(py, f_ext, true)
    }
});

/// A `Membrane` **loaded by a room** through one face of it — the baffled drumhead.
///
/// `RoomLoadedPlate` with model #4 in place of model #5. The arithmetic is `Wrap::step`'s,
/// unchanged: the two tiers differ in the seam they drive and in nothing else, which is why this
/// batch adds no stepping code at all.
///
/// Note what is deliberately absent, here and on the suspended arm: **`pressure()`**. Model #4 has
/// no monopole read-out, because it caches no acceleration, and the reference's mixin says so in
/// as many words. A getter or method added here would be a permanent shadow over `__getattr__`
/// (section 32.6) and would answer a question the model cannot.
#[pyclass(dict, name = "RoomLoadedMembrane", module = "physsynth_rs")]
pub struct PyRoomLoadedMembrane {
    w: Wrap,
}

grid_wrapper!(PyRoomLoadedMembrane, membrane, surface_pressure, {
    #[new]
    #[pyo3(signature = (*, membrane, room, face, origin=None, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        membrane: Py<PyAny>,
        room: Py<PyAny>,
        face: Py<PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("face", face)?;
        kwargs.set_item("origin", origin_arg(py, origin))?;
        kwargs.set_item("spreading", spreading_arg(py, spreading))?;
        Ok(Self {
            w: build(
                py,
                membrane,
                "membrane",
                room,
                "_MembraneSurface",
                "SurfacePort",
                &kwargs,
                Tier::Baffled,
            )?,
        })
    }

    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<()> {
        self.w.step(py, f_ext, false)
    }
});

/// A `Membrane` hanging **in** the room on an interior plane — the frame drum.
///
/// `RoomSuspendedPlate` with model #4: the head radiates from both faces, is driven by the
/// pressure **jump** across it, and is an *object* rather than a source, which is the cut
/// `InteriorSurfacePort` registers. Both doublings -- the load matrix and the `pbar` term -- are
/// `Tier::Suspended`'s, already in `Wrap::step`.
#[pyclass(dict, name = "RoomSuspendedMembrane", module = "physsynth_rs")]
pub struct PyRoomSuspendedMembrane {
    w: Wrap,
}

grid_wrapper!(PyRoomSuspendedMembrane, membrane, pressure_jump, {
    #[new]
    #[pyo3(signature = (*, membrane, room, plane, index, origin=None, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        membrane: Py<PyAny>,
        room: Py<PyAny>,
        plane: Py<PyAny>,
        index: Py<PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("plane", plane)?;
        kwargs.set_item("index", index)?;
        kwargs.set_item("origin", origin_arg(py, origin))?;
        kwargs.set_item("spreading", spreading_arg(py, spreading))?;
        Ok(Self {
            w: build(
                py,
                membrane,
                "membrane",
                room,
                "_MembraneSurface",
                "InteriorSurfacePort",
                &kwargs,
                Tier::Suspended,
            )?,
        })
    }

    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<Py<PyAny>>) -> PyResult<()> {
        self.w.step(py, f_ext, false)
    }
});

/// A `ModalBody` **loaded by a room** — the lumped tier, and the one wrapper with no factorization
/// at all: its solve is a single division, and `1 + G R_room >= 1` makes it unconditionally
/// passive.
#[pyclass(dict, name = "RoomLoadedBody", module = "physsynth_rs")]
pub struct PyRoomLoadedBody {
    body: Py<PyAny>,
    room: Py<PyAny>,
    port: Py<PyAny>,
    k: f64,
    g: f64,
    corr: Py<PyAny>,
    radiated_energy: f64,
    volume_velocity: f64,
    port_pressure: f64,
    n: i64,
}

#[pymethods]
impl PyRoomLoadedBody {
    #[new]
    #[pyo3(signature = (*, body, room, at, radius))]
    fn new(
        py: Python<'_>,
        body: Py<PyAny>,
        room: Py<PyAny>,
        at: Py<PyAny>,
        radius: Py<PyAny>,
    ) -> PyResult<Self> {
        let b = body.bind(py);
        require_same_rate(b, room.bind(py), "body")?;
        let k = f64_attr(b, "k")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("room", room.clone_ref(py))?;
        kwargs.set_item("at", at)?;
        kwargs.set_item("radius", radius)?;
        let port = airbox_module(py)?
            .getattr("RoomPort")?
            .call((), Some(&kwargs))?
            .unbind();

        // Rank-1 precomputes, from PUBLIC body attributes. The (1 + sigma_i k) carries the body's
        // implicit damping denominator into the load. `np.sum` is transcribed (section 31.2), so
        // this reduction is exact rather than a call.
        let a = vec1(py, &b.getattr("a")?, "a")?;
        let m = vec1(py, &b.getattr("m")?, "m")?;
        let sigma = vec1(py, &b.getattr("sigma")?, "sigma")?;
        let one_plus_sk: Vec<f64> = sigma.iter().map(|s| 1.0 + s * k).collect();
        let terms: Vec<f64> = (0..a.len())
            .map(|i| a[i] * a[i] / (m[i] * one_plus_sk[i]))
            .collect();
        let g = 0.5 * k * physsynth_core::reduce::sum(&terms);
        let corr: Vec<f64> = (0..a.len())
            .map(|i| k * k * a[i] / (m[i] * one_plus_sk[i]))
            .collect();

        Ok(Self {
            body,
            room,
            port,
            k,
            g,
            corr: pyarr(py, corr),
            radiated_energy: 0.0,
            volume_velocity: 0.0,
            port_pressure: 0.0,
            n: 0,
        })
    }

    #[getter]
    fn body(&self, py: Python<'_>) -> Py<PyAny> {
        self.body.clone_ref(py)
    }

    #[getter]
    fn room(&self, py: Python<'_>) -> Py<PyAny> {
        self.room.clone_ref(py)
    }

    #[getter]
    fn port(&self, py: Python<'_>) -> Py<PyAny> {
        self.port.clone_ref(py)
    }

    #[getter]
    fn k(&self) -> f64 {
        self.k
    }

    #[getter]
    fn _G(&self) -> f64 {
        self.g
    }

    #[getter]
    fn _corr(&self, py: Python<'_>) -> Py<PyAny> {
        self.corr.clone_ref(py)
    }

    #[getter]
    fn radiated_energy(&self) -> f64 {
        self.radiated_energy
    }

    #[setter]
    fn set_radiated_energy(&mut self, value: f64) {
        self.radiated_energy = value;
    }

    #[getter]
    fn volume_velocity(&self) -> f64 {
        self.volume_velocity
    }

    #[getter]
    fn port_pressure(&self) -> f64 {
        self.port_pressure
    }

    #[getter]
    fn n(&self) -> i64 {
        self.n
    }

    /// Advance one step: read the port, advance the body, solve the load, queue the injection.
    #[pyo3(signature = (force=0.0))]
    fn step(&mut self, py: Python<'_>, force: f64) -> PyResult<()> {
        let b = self.body.bind(py);
        let port = self.port.bind(py);
        port.call_method0("require_ready")?; // before mutating anything
        let pbar_free: f64 = port.call_method0("free_pressure")?.extract()?; // BEFORE room.step()
        let q_nm1 = vec1(py, &b.getattr("q_prev")?, "q^{n-1}")?; // before step() rolls history

        b.call_method1("step", (force,))?; // commit the force-free next state

        let q_obj = b.getattr("q")?;
        let q_free = pyarr(py, q_nm1.clone());
        let delta = q_obj.sub(q_free.bind(py))?;
        let u_free = np_dot(py, &b.getattr("a")?, &delta)? / (2.0 * self.k);

        let r_room = f64_attr(port, "R_room")?;
        let u = (u_free - self.g * pbar_free) / (1.0 + self.g * r_room);
        let pbar = pbar_free + r_room * u;

        // Rank-1 correction of q^{n+1}, then refresh q'' from the *corrected* second difference so
        // pressure() carries the load.
        let q = vec1(py, &q_obj, "q")?;
        let corr = vec1(py, self.corr.bind(py), "_corr")?;
        let q_next: Vec<f64> = (0..q.len()).map(|i| q[i] - pbar * corr[i]).collect();
        let prev = vec1(py, &b.getattr("q_prev")?, "q^n")?;
        let k2 = self.k * self.k;
        let accel: Vec<f64> = (0..q_next.len())
            .map(|i| (q_next[i] - 2.0 * prev[i] + q_nm1[i]) / k2)
            .collect();
        b.setattr("q", pyarr(py, q_next))?;
        b.setattr("_accel", pyarr(py, accel))?;

        port.call_method1("inject", (u,))?;
        self.radiated_energy += self.k * pbar * u;
        self.volume_velocity = u;
        self.port_pressure = pbar;
        self.n += 1;
        Ok(())
    }

    /// Total discrete energy `E_body + integral pbar U dt` (Joules) — an explicit override, not a
    /// delegation. Not monotone: the port's near-field reactance hands energy back every cycle.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let e: f64 = self.body.bind(py).call_method0("energy")?.extract()?;
        Ok(e + self.radiated_energy)
    }

    /// Radiated pressure read-out `sum_i a_i q_i''`, reflecting the room load.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.body.bind(py).call_method0("pressure")?.extract()
    }

    /// Set the body's initial modal state and reset this port's coupling ledger to zero.
    #[pyo3(signature = (q0, v0=None::<Py<PyAny>>))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        q0: Py<PyAny>,
        v0: Option<Option<Py<PyAny>>>,
    ) -> PyResult<()> {
        let b = self.body.bind(py);
        match v0 {
            Some(None) => b.call_method1("set_state", (q0, 0.0))?,
            None => b.call_method1("set_state", (q0, py.None()))?,
            Some(Some(v)) => b.call_method1("set_state", (q0, v))?,
        };
        self.reset_books(py)
    }

    /// Zero the body state and the coupling ledger — reuse on a new run.
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        self.body.bind(py).call_method1("set_state", (0.0,))?;
        self.reset_books(py)
    }

    fn _reset_books(&mut self, py: Python<'_>) -> PyResult<()> {
        self.reset_books(py)
    }

    /// Delegate read-only body accessors so a `RoomLoadedBody` is a drop-in wherever a bare
    /// `ModalBody` is expected (e.g. `StringBodyBridge`).
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if name == "body" {
            // Nothing to delegate through yet -- never recurse.
            return Err(PyAttributeError::new_err(name.to_string()));
        }
        Ok(self.body.bind(py).getattr(name)?.unbind())
    }
}

impl PyRoomLoadedBody {
    fn reset_books(&mut self, py: Python<'_>) -> PyResult<()> {
        self.radiated_energy = 0.0;
        self.volume_velocity = 0.0;
        self.port_pressure = 0.0;
        self.port.bind(py).call_method0("reset")?;
        self.n = 0;
        Ok(())
    }
}
