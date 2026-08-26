//! The binding over `physsynth_core::bore` — the acoustic bore wearing the Python interface.
//!
//! # The exciter seam crosses the language boundary here, and that is why this batch waited
//!
//! `Bore.step(source=...)` hands the freshly-updated pressure field to a caller who mutates it in
//! place, between the pressure and momentum sub-steps. The plan (§12.5) deliberately deferred the
//! bore until `reed` existed, because *how* that hook crosses into Rust commits the project for
//! `bow` and every continuous exciter after it.
//!
//! The answer, measured rather than assumed (§12.8): the hook stays a **general Python callable**.
//! `tests/test_reed_stability.py` passes its own `lambda p: None` to assert the hook is inert when
//! unused, so it is not merely the reed's private channel and porting `reed` removes the *hot*
//! crossing, not the capability. So:
//!
//! - `physsynth_core::bore::Source` is a plain Rust closure — a PyO3 type inside `physsynth-core`
//!   would break exactly what `crates/physsynth-core/tests/deps.rs` guards.
//! - This layer wraps a Python callable into one of those, and hands it a **live view** of the
//!   in-progress `p_next`.
//! - `reed::PyReedBore` bypasses it entirely: it holds a native bore and injects through the Rust
//!   closure, so the reed's own steps never cross back.
//!
//! `p_next` is a Python-owned `PyArray1` allocated per step, not a Rust `Vec` with a temporary
//! view over it. That is §9.3 again: the callable may keep the reference, and a view over a `Vec`
//! this function is about to drop is a use-after-free that *reads plausibly*. The allocation is
//! free — `step` rebinds `self.p` to a fresh array anyway, exactly as the original does.
//!
//! # Two classes of buffer, as on the membrane
//!
//! - **`p`, `U`, `U_prev`** — rebound every step; Python-owned, settable.
//! - **`x`, `x_u`, `S_node`, `S_seg`, `Lop`, `Cmat`, `dof`** — immutable after construction, built
//!   once and handed back by `clone_ref`.
//!
//! `Lop` and `Cmat` are the membrane's `L` problem again, and the measurement that settles it is
//! §12.8's: `tests/helpers.py`, `tests/test_bore_energy.py`, `tests/test_bore_modal.py` and
//! `web/serialize.py` all reach for them, with fancy indexing (`bore.Lop[dof][:, dof]`), with `.T`
//! and `.max()`, and as the `M=` argument of a generalized `eigsh`. They must be real
//! `scipy.sparse` objects on the instance, built once — so the constructor builds them here, as
//! `membrane` does. `physsynth-py` is a SciPy client; `physsynth-core` is not.
//!
//! # The hook makes this the first re-entrant method in the binding
//!
//! A `&mut self` `#[pymethods]` function holds a `PyRefMut` on the object for its whole body, and
//! `step` hands control back to Python in the middle of itself. The reed's hook reads
//! `self.bore.p[0]` — a perfectly ordinary read the original allows — and PyO3 refuses it with
//! `RuntimeError: Already mutably borrowed`, because the bore is still borrowed by the `step` that
//! is calling the hook.
//!
//! So `step` takes the **object** (`slf: &Bound<'_, Self>`) and borrows it in two short phases with
//! the callback in between, holding nothing. That restores the original's semantics exactly: while
//! the hook runs, the bore is fully readable — `p` is still the uncommitted `p^n`, `U` is still
//! `U^{n+1/2}`, and the step commits only afterwards.
//!
//! This is a general fact about the exciter seam rather than a bore detail, and it is the reason
//! this had to be a design decision rather than a transcription: **every model that calls back into
//! Python mid-step needs this shape**, and the failure is not subtle at runtime but is completely
//! invisible to `cargo test`, which never crosses the boundary.
//!
//! # Private names are part of the surface, for the third time
//!
//! `_open_left` and `_open_right` are read by `tests/test_bore_radiation.py` and by
//! `web/serialize.py`; `_bc_left` is read by `reed.py`, which refuses to build on a bore whose
//! mouthpiece end is not `"closed"`. Measured by grep, not guessed — plan §1.2 and §12.2.

use crate::{as_1d_f64, csr_triplets, state_slice};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::bore as core;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

/// Parse the `boundary=` argument into a `(left, right)` pair.
///
/// Mirrors `(boundary, boundary) if isinstance(boundary, str) else boundary`: a bare string sets
/// both ends, a 2-sequence sets them independently. Anything else yields `None`, which
/// `Params::new` turns into the rejection *at the position in the check order where Python raises
/// it* — after the scalar, `N`, `sigma` and `R_bell` checks, and before the CFL test.
pub(crate) fn parse_boundary(obj: &Bound<'_, PyAny>) -> Option<(core::End, core::End)> {
    if let Ok(s) = obj.cast::<PyString>() {
        let e = core::End::parse(&s.to_cow().ok()?)?;
        return Some((e, e));
    }
    let seq = obj.cast::<PyTuple>().ok()?;
    if seq.len() != 2 {
        return None;
    }
    let left = core::End::parse(
        &seq.get_item(0)
            .ok()?
            .cast::<PyString>()
            .ok()?
            .to_cow()
            .ok()?,
    )?;
    let right = core::End::parse(
        &seq.get_item(1)
            .ok()?
            .cast::<PyString>()
            .ok()?
            .to_cow()
            .ok()?,
    )?;
    Some((left, right))
}

/// Build the `Params`, raising the Python original's messages — including the one that has to
/// quote the caller's object.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_params(
    boundary: &Bound<'_, PyAny>,
    l: f64,
    fs: f64,
    n: i64,
    radius: f64,
    sigma: f64,
    r_bell: f64,
    rho0: f64,
    c0: f64,
) -> PyResult<core::Params> {
    // `N < 2` is checked before the boundary is parsed, and a negative `N` must reach that check
    // rather than dying in the cast.
    let n_usize = if n < 2 { 0usize } else { n as usize };
    let bc = parse_boundary(boundary);
    core::Params::new(l, fs, n_usize, radius, bc, sigma, r_bell, rho0, c0).map_err(|e| match e {
        core::ParamError::BadBoundary => {
            let shown = boundary
                .repr()
                .map(|r| r.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unrepresentable>".to_owned());
            PyValueError::new_err(format!(
                "each boundary end must be one of ('closed', 'open', 'radiating'), got {shown}."
            ))
        }
        other => PyValueError::new_err(other.to_string()),
    })
}

/// A discretized acoustic tube — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with `physsynth.core.bore.Bore`; the
/// docstring on that class is the reference.
#[pyclass(name = "Bore", module = "physsynth_rs")]
pub struct PyBore {
    pub(crate) params: core::Params,
    /// The `boundary=` argument exactly as passed, so `.boundary` echoes a str or a tuple the way
    /// the original's does.
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    x_u: Py<PyArray1<f64>>,
    s_node: Py<PyArray1<f64>>,
    s_seg: Py<PyArray1<f64>>,
    lop: Py<PyAny>,
    cmat: Py<PyAny>,
    dof: Py<PyAny>,
    pressure: Py<PyArray1<f64>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    pub(crate) radiated_energy: f64,
    u_out: f64,
    u_out_prev: f64,
    n: usize,
    /// Divergence workspace, hoisted out of the timestep.
    scratch: Vec<f64>,
}

impl PyBore {
    /// Validate an array being assigned to one of the three state buffers and take ownership.
    fn adopt_state(
        &self,
        value: &Bound<'_, PyAny>,
        name: &str,
        want: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != want {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({want},), got ({},).",
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }

    /// The pressure sub-step, written into a freshly allocated Python-owned array.
    fn pressure_step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let out = PyArray1::<f64>::zeros(py, self.params.nodes(), false);
        {
            let p_bound = self.pressure.bind(py);
            let u_bound = self.u.bind(py);
            let p_ro = p_bound.readonly();
            let u_ro = u_bound.readonly();
            let mut rw = out.readwrite();
            let slot = rw
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("p_next must be contiguous."))?;
            core::pressure_into(
                state_slice(&p_ro, "p")?,
                state_slice(&u_ro, "U")?,
                slot,
                &mut self.scratch,
                &self.params,
            );
        }
        Ok(out)
    }

    /// The drain, the momentum sub-step, and the commit — everything after the `source` hook.
    fn finish_step(&mut self, py: Python<'_>, p_next: Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        {
            let p_bound = self.pressure.bind(py);
            let p_ro = p_bound.readonly();
            let mut rw = p_next.readwrite();
            let slot = rw
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("p_next must be contiguous."))?;
            if let Some(u_out) = core::apply_radiating_ends(
                slot,
                state_slice(&p_ro, "p")?,
                &mut self.radiated_energy,
                &self.params,
            ) {
                self.u_out_prev = self.u_out;
                self.u_out = u_out;
            }
        }

        let mut u_next = vec![0.0; self.params.n];
        {
            let u_bound = self.u.bind(py);
            let u_ro = u_bound.readonly();
            let p_ro = p_next.readonly();
            core::momentum_into(
                state_slice(&p_ro, "p_next")?,
                state_slice(&u_ro, "U")?,
                &mut u_next,
                &self.params,
            );
        }

        // `self.U_prev = self.U` hands the *same object* over, exactly as the original does, so a
        // caller holding a reference to `.U` across a step finds it under `.U_prev` afterwards.
        let fresh = PyArray1::from_vec(py, u_next).unbind();
        self.u_prev = std::mem::replace(&mut self.u, fresh);
        self.pressure = p_next.unbind();
        self.n += 1;
        Ok(())
    }

    /// The current pressure field as a plain `Vec`, for a native caller that needs a snapshot.
    pub(crate) fn pressure_vec(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let bound = self.pressure.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "p")?.to_vec())
    }

    /// Pressure at one node without copying the field.
    pub(crate) fn pressure_node(&self, py: Python<'_>, idx: usize) -> PyResult<f64> {
        let bound = self.pressure.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "p")?[idx])
    }
}

#[pymethods]
impl PyBore {
    #[new]
    #[pyo3(signature = (
        *, L, fs, N, radius=0.008, boundary=None, sigma=0.0, R_bell=0.0,
        rho0=1.2041, c0=343.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        L: f64,
        fs: f64,
        N: i64,
        radius: f64,
        boundary: Option<Bound<'_, PyAny>>,
        sigma: f64,
        R_bell: f64,
        rho0: f64,
        c0: f64,
    ) -> PyResult<Self> {
        // The original's default is the tuple `("closed", "open")` — the ideal clarinet — and
        // `.boundary` echoes whatever was passed, so the default has to be a real tuple object.
        let boundary = match boundary {
            Some(obj) => obj,
            None => PyTuple::new(py, ["closed", "open"])?.into_any(),
        };
        let params = build_params(&boundary, L, fs, N, radius, sigma, R_bell, rho0, c0)?;

        // Built once — `airbox`-style per-step rebuilds are what the membrane's header warns
        // about, and `helpers.bore_low_eigenfrequencies` slices these on every modal test.
        let (lop_csr, cmat_csr) = params.pressure_operator();
        let scipy = py.import("scipy.sparse")?;
        let build = |m: &physsynth_core::sparse::Csr| -> PyResult<Py<PyAny>> {
            let (data, indices, indptr, shape) = csr_triplets(py, m)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("shape", shape)?;
            Ok(scipy
                .call_method("csr_matrix", ((data, indices, indptr),), Some(&kwargs))?
                .unbind())
        };
        let lop = build(&lop_csr)?;
        let cmat = build(&cmat_csr)?;

        // `np.nonzero(free)[0].astype(np.int64)`.
        let np = py.import("numpy")?;
        let dof_list: Vec<i64> = params.dof().iter().map(|&i| i as i64).collect();
        let dof = np
            .call_method1("array", (dof_list, np.getattr("int64")?))?
            .unbind();

        let nodes = params.nodes();
        let n_seg = params.n;
        Ok(PyBore {
            x: PyArray1::from_vec(py, params.grid()).unbind(),
            x_u: PyArray1::from_vec(py, params.grid_u()).unbind(),
            s_node: PyArray1::from_slice(py, &params.s_node).unbind(),
            s_seg: PyArray1::from_slice(py, &params.s_seg).unbind(),
            lop,
            cmat,
            dof,
            pressure: PyArray1::from_vec(py, vec![0.0; nodes]).unbind(),
            u: PyArray1::from_vec(py, vec![0.0; n_seg]).unbind(),
            u_prev: PyArray1::from_vec(py, vec![0.0; n_seg]).unbind(),
            boundary: boundary.unbind(),
            radiated_energy: 0.0,
            u_out: 0.0,
            u_out_prev: 0.0,
            n: 0,
            scratch: vec![0.0; nodes],
            params,
        })
    }

    // -- parameters --------------------------------------------------------------------------

    #[getter]
    fn L(&self) -> f64 {
        self.params.l
    }
    #[getter]
    fn fs(&self) -> f64 {
        self.params.fs
    }
    #[getter]
    fn N(&self) -> usize {
        self.params.n
    }
    #[getter]
    fn radius(&self) -> f64 {
        self.params.radius
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.params.sigma
    }
    #[getter]
    fn R_bell(&self) -> f64 {
        self.params.r_bell
    }
    #[getter]
    fn rho0(&self) -> f64 {
        self.params.rho0
    }
    #[getter]
    fn c0(&self) -> f64 {
        self.params.c0
    }
    #[getter]
    fn h(&self) -> f64 {
        self.params.h
    }
    #[getter]
    fn k(&self) -> f64 {
        self.params.k
    }
    #[getter]
    fn lam(&self) -> f64 {
        self.params.lam
    }
    #[getter]
    fn Z0(&self) -> f64 {
        self.params.z0
    }

    /// The left-end token. Private in the original and read by `reed.py`, which refuses to build
    /// on a bore whose mouthpiece end is not `"closed"`.
    #[getter]
    fn _bc_left(&self) -> &'static str {
        self.params.bc_left.name()
    }
    #[getter]
    fn _bc_right(&self) -> &'static str {
        self.params.bc_right.name()
    }
    /// Private in the original and read by `tests/test_bore_radiation.py` and `web/serialize.py`.
    #[getter]
    fn _open_left(&self) -> bool {
        self.params.open_left()
    }
    #[getter]
    fn _open_right(&self) -> bool {
        self.params.open_right()
    }
    #[getter]
    fn _rad_left(&self) -> bool {
        self.params.rad_left()
    }
    #[getter]
    fn _rad_right(&self) -> bool {
        self.params.rad_right()
    }

    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.x.clone_ref(py)
    }
    #[getter]
    fn x_u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.x_u.clone_ref(py)
    }
    #[getter]
    fn S_node(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.s_node.clone_ref(py)
    }
    #[getter]
    fn S_seg(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.s_seg.clone_ref(py)
    }

    /// The pressure stiffness `L = G^T M^{-1} G`, as a `scipy.sparse.csr_matrix`.
    #[getter]
    fn Lop(&self, py: Python<'_>) -> Py<PyAny> {
        self.lop.clone_ref(py)
    }
    /// The diagonal node mass `C`, as a `scipy.sparse.csr_matrix`.
    #[getter]
    fn Cmat(&self, py: Python<'_>) -> Py<PyAny> {
        self.cmat.clone_ref(py)
    }
    /// Indices of the free (non-open) pressure nodes; `int64`, as the original's are.
    #[getter]
    fn dof(&self, py: Python<'_>) -> Py<PyAny> {
        self.dof.clone_ref(py)
    }

    // -- state -------------------------------------------------------------------------------

    /// Pressure field `p^n` — the live array, writable in place *and* rebindable.
    #[getter]
    fn p(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.pressure.clone_ref(py)
    }
    #[setter]
    fn set_p(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.pressure = self.adopt_state(value, "p", self.params.nodes())?;
        Ok(())
    }

    /// Volume velocity `U^{n+1/2}`.
    #[getter]
    fn U(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_U(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt_state(value, "U", self.params.n)?;
        Ok(())
    }

    /// Volume velocity `U^{n-1/2}` — after a step this *is* the object `.U` was.
    #[getter]
    fn U_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u_prev.clone_ref(py)
    }
    #[setter]
    fn set_U_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev = self.adopt_state(value, "U_prev", self.params.n)?;
        Ok(())
    }

    /// Energy shed to the far field through radiating ends.
    #[getter]
    fn radiated_energy(&self) -> f64 {
        self.radiated_energy
    }
    #[setter]
    fn set_radiated_energy(&mut self, value: f64) {
        self.radiated_energy = value;
    }

    #[getter]
    fn _U_out(&self) -> f64 {
        self.u_out
    }
    #[setter]
    fn set__U_out(&mut self, value: f64) {
        self.u_out = value;
    }
    #[getter]
    fn _U_out_prev(&self) -> f64 {
        self.u_out_prev
    }
    #[setter]
    fn set__U_out_prev(&mut self, value: f64) {
        self.u_out_prev = value;
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: usize) {
        self.n = value;
    }

    /// Current pressure field `p^n` (a copy, safe to store for plotting).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.pressure_vec(py)?).unbind())
    }

    // -- initial conditions ------------------------------------------------------------------

    /// Set the initial pressure field `p^0` (and optional half-node volume velocity).
    #[pyo3(signature = (p0, u0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        p0: &Bound<'_, PyAny>,
        u0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let nodes = self.params.nodes();
        let n_seg = self.params.n;
        let mut p = as_1d_f64(py, p0, "p0", nodes)?;
        // `np.broadcast_to(np.asarray(u0, float), (N,))`: a scalar (the default) fills, a
        // full-length array is taken as is.
        let u_prev = match u0 {
            None => vec![0.0; n_seg],
            Some(obj) => match obj.extract::<f64>() {
                Ok(scalar) => vec![scalar; n_seg],
                Err(_) => as_1d_f64(py, obj, "u0", n_seg)?,
            },
        };

        core::apply_open_ends(&mut p, &self.params);
        let mut u = vec![0.0; n_seg];
        core::momentum_into(&p, &u_prev, &mut u, &self.params);

        self.pressure = PyArray1::from_vec(py, p).unbind();
        self.u_prev = PyArray1::from_vec(py, u_prev).unbind();
        self.u = PyArray1::from_vec(py, u).unbind();
        self.radiated_energy = 0.0;
        self.u_out = 0.0;
        self.u_out_prev = 0.0;
        self.n = 0;
        Ok(())
    }

    // -- time stepping -----------------------------------------------------------------------

    /// Advance one timestep: pressure from the current velocity, then velocity from it.
    ///
    /// `source` is an optional callable `source(p_next)` invoked on the freshly-updated pressure
    /// field **after** the open-end pin and **before** the radiating drain and the momentum
    /// sub-step. It receives a live, writable view of the array this step is about to commit, so
    /// an in-place correction reaches the bore — which is the whole point of the hook.
    #[pyo3(signature = (source=None))]
    fn step(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        source: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // The receiver is the *object*, not `&mut self`, and the borrow is taken and dropped
        // around each phase on purpose. See the note on re-entrancy in the module header.
        let p_next = slf.borrow_mut().pressure_step(py)?;
        if let Some(f) = source {
            f.call1((&p_next,))?;
        }
        slf.borrow_mut().finish_step(py, p_next)
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Energy stored in the air column (Joules), excluding what has already radiated away.
    fn acoustic_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let p_bound = self.pressure.bind(py);
        let u_bound = self.u.bind(py);
        let up_bound = self.u_prev.bind(py);
        let p_ro = p_bound.readonly();
        let u_ro = u_bound.readonly();
        let up_ro = up_bound.readonly();
        Ok(core::acoustic_energy(
            state_slice(&p_ro, "p")?,
            state_slice(&u_ro, "U")?,
            state_slice(&up_ro, "U_prev")?,
            &self.params,
        ))
    }

    /// Total conserved energy `E_bore + radiated_energy` (Joules).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.acoustic_energy(py)? + self.radiated_energy)
    }

    /// Pressure at node `index` — a microphone pickup for spectral analysis.
    ///
    /// Negative indices count from the end, as they do on the NumPy array this replaces.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let nodes = self.params.nodes() as i64;
        let idx = if index < 0 { index + nodes } else { index };
        if idx < 0 || idx >= nodes {
            return Err(PyIndexError::new_err(format!(
                "index {index} is out of bounds for {nodes} nodes"
            )));
        }
        self.pressure_node(py, idx as usize)
    }

    /// Alias for `displacement_at` in the natural acoustic quantity.
    fn pressure_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.displacement_at(py, index)
    }

    /// Far-field monopole read-out: the bell's net volume acceleration `dU_out/dt` (m³/s²).
    fn pressure(&self) -> f64 {
        (self.u_out - self.u_out_prev) / self.params.k
    }
}
