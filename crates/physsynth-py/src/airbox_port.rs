//! The binding over `physsynth_core::airbox_port` — the room's ports wearing the Python interface.
//!
//! `RoomPort`, `SurfacePort` and `InteriorSurfacePort`. `_PatchPort`, the private base the two
//! distributed tiers share, has no counterpart here: it exists in the reference to hold code, and
//! its behaviour lives in `physsynth_core::airbox_port`'s free functions instead. Nothing outside
//! `airbox.py` names it (grepped: one comment in `tests/test_airbox_dipole.py`), so there is
//! nothing to be compatible with.
//!
//! # Everything the room can do, the room is asked to do
//!
//! A port reaches its room through Python attribute access and Python method calls —
//! `room.node_index(at)`, `room._plane_axis(plane)`, `room._register_cut(...)`,
//! `room._ports.append(self)`, `room._pending_ports.append(...)` — and never through the Rust
//! room's `Params`. Three reasons, in increasing order of how much they cost to get wrong:
//!
//! 1. **A port must accept either room.** §29.1 found `connection` polymorphic over its
//!    collaborators' types; the port tier is the same shape one level down. `AirBoxPy` and
//!    `_rs.AirBox` are both rooms, the parity file builds one of each, and a caller may run with
//!    the flag off for one and on for the other.
//! 2. **The refusals are the room's to write.** `node_index` raises "outside the room" and
//!    `_plane_axis` raises "unknown plane"; calling them gets both messages exactly right for free,
//!    in whichever implementation the caller actually has.
//! 3. **`_register_cut` mutates the room.** It is the one registration that writes, it is called
//!    last on purpose so that a refusal leaves the room as it was found, and re-implementing it
//!    here would be a second writer of `_cut_mask` — §30.4's bug class with an extra door.
//!
//! # The seam, from the other side
//!
//! §30.3 enumerated the fifteen names the ports reach *into the room* through, and the lesson was
//! *grep for assignment, not only for reference*. Applied to this tier the same search finds four
//! attributes a test **writes on a port** — `port.T`, `port.load_matrix`, `port.R` and
//! `port.areas`, all replaced with arbitrary SciPy or NumPy objects to switch the coupling off or
//! halve it — plus `port._queued_at`, which `AirBox.set_state` resets. All five are therefore
//! `Py<PyAny>` slots with a setter, never Rust-owned values, and nothing in this file reads them
//! back after construction except `net_area`, which reads `areas` live so that zeroing it works.
//!
//! And one door further on, which no attribute search finds at all:
//! `tests/test_airbox_dipole.py::test_a_sign_flip_is_invisible_to_every_energy_quantity`
//! **replaces the port's methods** —
//!
//! ```python
//! port.free_pressure = lambda: tuple(reversed(free()))
//! port.inject = lambda q: inject(-q)
//! ```
//!
//! — on the instance, and the wrapper above then calls the lambdas. A `#[pyclass]` has no
//! `__dict__` and refuses that outright, so all three classes here carry `dict`. The precedence
//! works out because CPython's lookup rules do not care which language defined the class: a
//! `#[getter]`/`#[setter]` pair is a *data* descriptor and wins over the instance dict (so
//! `port.T = x` still runs the setter), while a `#[pymethod]` is a *non-data* descriptor and loses
//! to it (so an assigned lambda shadows the method, which is the whole point).
//!
//! Attributes that nothing writes — `nodes`, `w`, `R_room`, `index`, `_flat` — are exposed with a
//! getter and **no** setter, so an assignment raises `AttributeError` instead of quietly leaving a
//! cached index array disagreeing with the one the room is handed. That is a deliberate narrowing
//! of the reference, which would accept it silently, and it is the loud direction.
//!
//! # One cosmetic deviation, stated rather than hidden
//!
//! The two "shares node" refusals interpolate `np.unravel_index(...)`, whose result is a tuple of
//! NumPy scalars — so under NumPy 2 the reference renders it `(np.int64(4), np.int64(3),
//! np.int64(3))`. This file writes `(4, 3, 3)`. Reproducing the other spelling would be a claim
//! about a NumPy version's scalar `repr`, which is the sort of library-internal claim §22.1 taught
//! this migration not to make for a *number*; making it for a *message* would be worse value
//! still. Both suites match the substring `"shares node"` and neither reads the tuple.

use crate::shape::shape_repr;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn};
use physsynth_core::airbox_port as core;
use physsynth_core::airbox_port::{RoomView, Spreading};
use physsynth_core::fmt::{py_float, py_general};
use physsynth_core::reduce;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// -- reading a room ------------------------------------------------------------------------------

/// Read borrows of every array a port kernel needs, held together so the slices stay alive.
///
/// §30.5's lesson, applied one tier up and with more at stake: the room's copying draft cost a
/// constant factor because its copy was proportional to its work, but a port's whole value is that
/// it is `O(port nodes)` in a room that may be a quarter of a million. Copying `p`, `ux`, `uy` and
/// `uz` to read twenty of their entries is not a constant factor, it is an asymptotic one. So the
/// arrays are **borrowed** and indexed.
struct RoomGuards<'py> {
    p: PyReadonlyArrayDyn<'py, f64>,
    u: [PyReadonlyArrayDyn<'py, f64>; 3],
    w: [PyReadonlyArray1<'py, f64>; 3],
    node_w: PyReadonlyArrayDyn<'py, f64>,
    beta: PyReadonlyArrayDyn<'py, f64>,
    n: [usize; 3],
    h: f64,
    k: f64,
    rho0: f64,
    c0: f64,
    has_walls: bool,
}

impl RoomGuards<'_> {
    fn view(&self) -> PyResult<RoomView<'_>> {
        Ok(RoomView {
            n: self.n,
            h: self.h,
            k: self.k,
            rho0: self.rho0,
            c0: self.c0,
            p: dyn_slice(&self.p, "p")?,
            u: [
                dyn_slice(&self.u[0], "ux")?,
                dyn_slice(&self.u[1], "uy")?,
                dyn_slice(&self.u[2], "uz")?,
            ],
            w: [
                self.w[0].as_slice().map_err(non_contiguous)?,
                self.w[1].as_slice().map_err(non_contiguous)?,
                self.w[2].as_slice().map_err(non_contiguous)?,
            ],
            node_w: dyn_slice(&self.node_w, "_W")?,
            beta: dyn_slice(&self.beta, "_beta")?,
            has_walls: self.has_walls,
        })
    }
}

/// One field of a [`RoomGuards`] as a slice, or the contiguity refusal naming it.
fn dyn_slice<'a>(a: &'a PyReadonlyArrayDyn<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("room.{name} must be contiguous.")))
}

fn non_contiguous<E>(_: E) -> PyErr {
    PyValueError::new_err("room._w must hold contiguous 1-D weight arrays.")
}

/// Fetch a float64 array attribute as a read borrow.
fn borrow_any<'py>(room: &Bound<'py, PyAny>, name: &str) -> PyResult<PyReadonlyArrayDyn<'py, f64>> {
    let obj = room.getattr(name)?;
    let arr: Bound<'py, numpy::PyArrayDyn<f64>> = obj.cast_into().map_err(|_| {
        PyValueError::new_err(format!("room.{name} must be a float64 NumPy array."))
    })?;
    Ok(arr.readonly())
}

/// Fetch a 1-D float64 array attribute as a read borrow.
fn borrow_1d<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let arr: Bound<'py, numpy::PyArray1<f64>> = obj
        .clone()
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 1-D float64 NumPy array.")))?;
    Ok(arr.readonly())
}

/// Borrow everything a kernel needs from a room, whichever implementation it is.
fn guards<'py>(room: &Bound<'py, PyAny>) -> PyResult<RoomGuards<'py>> {
    let n_obj = room.getattr("N")?;
    let n_vec: Vec<usize> = n_obj.extract()?;
    if n_vec.len() != 3 {
        return Err(PyValueError::new_err("room.N must be a triple."));
    }
    let w_obj = room.getattr("_w")?;
    let w_items: Vec<Bound<'py, PyAny>> = w_obj.try_iter()?.collect::<PyResult<_>>()?;
    if w_items.len() != 3 {
        return Err(PyValueError::new_err(
            "room._w must hold three weight arrays.",
        ));
    }
    Ok(RoomGuards {
        p: borrow_any(room, "p")?,
        u: [
            borrow_any(room, "ux")?,
            borrow_any(room, "uy")?,
            borrow_any(room, "uz")?,
        ],
        w: [
            borrow_1d(&w_items[0], "room._w[0]")?,
            borrow_1d(&w_items[1], "room._w[1]")?,
            borrow_1d(&w_items[2], "room._w[2]")?,
        ],
        node_w: borrow_any(room, "_W")?,
        beta: borrow_any(room, "_beta")?,
        n: [n_vec[0], n_vec[1], n_vec[2]],
        h: room.getattr("h")?.extract()?,
        k: room.getattr("k")?.extract()?,
        rho0: room.getattr("rho0")?.extract()?,
        c0: room.getattr("c0")?.extract()?,
        has_walls: room.getattr("_has_walls")?.extract()?,
    })
}

// -- shared plumbing ------------------------------------------------------------------------------

/// The node index triple as Python `intp` arrays — the fancy index the room is handed.
fn nodes_to_py(py: Python<'_>, nodes: &core::Nodes) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let intp = np.getattr("intp")?;
    let mut cols = Vec::with_capacity(3);
    for axis in nodes.iter() {
        let vals: Vec<i64> = axis.iter().map(|&i| i as i64).collect();
        let arr = PyArray1::from_vec(py, vals).into_any();
        cols.push(np.call_method1("asarray", (arr, &intp))?);
    }
    Ok(PyTuple::new(py, cols)?.into_any().unbind())
}

/// A flat offset vector as a Python `intp` array — `_flat`.
fn flat_to_py(py: Python<'_>, flat: &[usize]) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let vals: Vec<i64> = flat.iter().map(|&i| i as i64).collect();
    let arr = PyArray1::from_vec(py, vals).into_any();
    Ok(np
        .call_method1("asarray", (arr, np.getattr("intp")?))?
        .unbind())
}

/// A `Vec<f64>` as a fresh 1-D NumPy array.
fn f64_to_py(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    PyArray1::from_vec(py, values).into_any().unbind()
}

/// A `Csr` as the `scipy.sparse.csr_matrix` the reference holds.
fn csr_to_py(py: Python<'_>, m: &physsynth_core::sparse::Csr) -> PyResult<Py<PyAny>> {
    let scipy = py.import("scipy.sparse")?;
    let (data, indices, indptr, shape) = crate::csr_triplets(py, m)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    Ok(scipy
        .call_method("csr_matrix", ((data, indices, indptr),), Some(&kwargs))?
        .unbind())
}

/// Refuse a second solve inside one room step — `require_ready`, shared verbatim by both tiers.
fn require_ready_inner(room: &Bound<'_, PyAny>, queued_at: i64, message: String) -> PyResult<()> {
    let n: i64 = room.getattr("n")?.extract()?;
    if queued_at == n {
        return Err(PyRuntimeError::new_err(message));
    }
    Ok(())
}

/// Append `(nodes, weights, U)` to the room's pending-port queue.
fn queue(room: &Bound<'_, PyAny>, nodes: &Py<PyAny>, w: Py<PyAny>, u: f64) -> PyResult<()> {
    let py = room.py();
    let entry = PyTuple::new(
        py,
        [
            nodes.clone_ref(py),
            w,
            u.into_pyobject(py)?.into_any().unbind(),
        ],
    )?;
    room.getattr("_pending_ports")?
        .call_method1("append", (entry,))?;
    Ok(())
}

/// The shared disjointness refusal, over whichever ports the room already holds.
fn check_disjoint(
    room: &Bound<'_, PyAny>,
    flat: &[usize],
    shape: [usize; 3],
    describe: &dyn Fn(usize, [usize; 3], String) -> String,
) -> PyResult<()> {
    let ports = room.getattr("_ports")?;
    for other in ports.try_iter()? {
        let other = other?;
        let other_flat: Vec<i64> = other.getattr("_flat")?.extract()?;
        let other_flat: Vec<usize> = other_flat.iter().map(|&v| v as usize).collect();
        let (first, count) = core::shared_nodes(flat, &other_flat);
        if let Some(first) = first {
            let node = core::unravel(first, shape);
            let other_index = other.getattr("index")?.str()?.to_string();
            return Err(PyValueError::new_err(describe(count, node, other_index)));
        }
    }
    Ok(())
}

/// The open-face refusal, shared by both tiers — only the wording differs.
fn touched_open_faces(
    room: &Bound<'_, PyAny>,
    nodes: &core::Nodes,
    n: [usize; 3],
) -> PyResult<Vec<String>> {
    let np = room.py().import("numpy")?;
    let any_open: bool = np
        .call_method1("any", (room.getattr("_open")?,))?
        .extract()?;
    if !any_open {
        return Ok(Vec::new());
    }
    let walls = room.getattr("walls")?;
    let mut touched = Vec::new();
    for face in core::FACES {
        let z: f64 = walls.get_item(face)?.extract()?;
        if z != 0.0 {
            continue;
        }
        let axis = core::AXES
            .iter()
            .position(|&c| c == face.as_bytes()[0] as char)
            .unwrap();
        let end = if face.as_bytes()[1] == b'0' {
            0
        } else {
            n[axis]
        };
        if nodes[axis].contains(&end) {
            touched.push(format!("'{face}'"));
        }
    }
    Ok(touched)
}

/// Render a list of quoted face names the way Python renders a `list[str]`.
fn face_list(faces: &[String]) -> String {
    format!("[{}]", faces.join(", "))
}

// -- the module helpers ---------------------------------------------------------------------------

/// `physsynth.core.airbox._free_pressure_nodes(room, nodes)`.
///
/// Kept as a module-level function rather than folded into the two classes, because it is what the
/// ports and the reference *share* — and because the parity file asserts it against the room's own
/// `_divergence()`-then-closure. §23.6 warns that porting a helper can silently empty exactly that
/// kind of comparison: the Python `_free_pressure_nodes_py` therefore stays in `airbox.py` as the
/// asserted reference and this is compared against it explicitly, rather than the two swapping
/// places and the test comparing Rust with Rust.
#[pyfunction]
#[pyo3(name = "_free_pressure_nodes")]
pub fn free_pressure_nodes(
    py: Python<'_>,
    room: &Bound<'_, PyAny>,
    nodes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let g = guards(room)?;
    let view = g.view()?;
    let items: Vec<Bound<'_, PyAny>> = nodes.try_iter()?.collect::<PyResult<_>>()?;
    if items.len() != 3 {
        return Err(PyValueError::new_err(
            "nodes must be an (ix, iy, iz) index triple.",
        ));
    }
    let mut cols: Vec<Vec<usize>> = Vec::with_capacity(3);
    for item in &items {
        let vals: Vec<i64> = item.extract()?;
        cols.push(vals.iter().map(|&v| v as usize).collect());
    }
    let out = core::free_pressure_nodes(&view, &[&cols[0], &cols[1], &cols[2]]);
    Ok(f64_to_py(py, out))
}

/// `physsynth.core.airbox._face_axes(face)`.
///
/// `(normal axis, end, in-plane axis 0, in-plane axis 1)`. Pure integer arithmetic off a
/// six-element table, so there is nothing here to be bit-identical about -- what has to match is
/// the refusal, whose message the reference formats from `FACES` and which
/// `test_airbox_surface.py` matches on.
#[pyfunction]
#[pyo3(name = "_face_axes")]
pub fn face_axes_py(face: &str) -> PyResult<(usize, usize, usize, usize)> {
    core::face_axes(face).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown face '{face}'; expected one of ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')."
        ))
    })
}

/// `physsynth.core.airbox.impedance_from_zeta(zeta, rho0=RHO0_AIR, c0=C0_AIR)`.
///
/// `float(zeta) * rho0 * c0`, left-folded as Python folds it -- `(zeta rho0) c0`. Both defaults
/// are the module's ambient air, and they are spelled here rather than read off the module so that
/// a caller who passes neither gets the same two constants either implementation is built with.
#[pyfunction]
#[pyo3(name = "impedance_from_zeta")]
#[pyo3(signature = (zeta, *, rho0 = 1.2041, c0 = 343.0))]
pub fn impedance_from_zeta_py(zeta: f64, rho0: f64, c0: f64) -> f64 {
    physsynth_core::airbox::impedance_from_zeta(zeta, rho0, c0)
}

/// `physsynth_core::reduce::sum` — NumPy's pairwise blocking, exposed so the claim can be tested.
///
/// Not used by `airbox.py`. It exists because the claim it rests on is the riskiest thing in this
/// batch: §30.2 declined to transcribe `np.sum`'s blocking partly on the grounds that it might be
/// a claim about the machine (§22.1), and this batch takes the opposite view — that a summation
/// ORDER is fixed by an algorithm where a transcendental's last bit is fixed by an instruction
/// selection. That view is measured here on this machine and asserted in
/// `tests/test_rust_parity_airbox_port.py`, so if it is ever false on a CI runner the failure is
/// one named test saying exactly what broke, rather than every exact assertion in the port tier
/// going red at once for no visible reason.
#[pyfunction]
#[pyo3(name = "_pairwise_sum")]
pub fn pairwise_sum(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("a must be a contiguous 1-D float64 array."))?;
    Ok(reduce::sum(s))
}

// -- RoomPort -------------------------------------------------------------------------------------

/// One two-way terminal into an `AirBox` — the room's Thevenin equivalent at a spot.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.airbox.RoomPort`; the docstring on that class is the reference.
#[pyclass(name = "RoomPort", module = "physsynth_rs", dict)]
pub struct PyRoomPort {
    room: Py<PyAny>,
    index: (usize, usize, usize),
    radius: Option<f64>,
    nodes: Py<PyAny>,
    node_idx: core::Nodes,
    flat: Py<PyAny>,
    flat_idx: Vec<usize>,
    w: Py<PyAny>,
    r_room: f64,
    queued_at: i64,
}

#[pymethods]
impl PyRoomPort {
    #[new]
    #[pyo3(signature = (*, room, at, radius))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        at: Bound<'_, PyAny>,
        radius: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        // The room owns "outside the room", so ask it rather than re-deriving the message.
        let index: (usize, usize, usize) = room.call_method1("node_index", (&at,))?.extract()?;
        let idx = [index.0, index.1, index.2];
        let g = guards(&room)?;
        let view = g.view()?;

        let radius_val = if radius.is_none() {
            None
        } else {
            Some(radius.extract::<f64>()?)
        };
        let node_idx = match radius_val {
            None => [vec![idx[0]], vec![idx[1]], vec![idx[2]]],
            Some(r) => {
                if r <= 0.0 || !r.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "port radius must be a positive length, got {}.",
                        radius.repr()?
                    )));
                }
                let nodes = core::ball_nodes(view.n, view.h, idx, r);
                if nodes[0].len() == 1 {
                    return Err(PyValueError::new_err(format!(
                        "port radius {} m is smaller than the grid can resolve (h = {}): the ball \
                         contains only the centre node {:?}, so this would silently be a point \
                         port with a grid-dependent load magnitude. Coarsen the request, refine h, \
                         or pass radius=None to ask for a point port on purpose.",
                        py_float(r),
                        py_float(view.h),
                        idx
                    )));
                }
                nodes
            }
        };

        // The two refusals, in the reference's order: open faces before disjointness.
        let touched = touched_open_faces(&room, &node_idx, view.n)?;
        if !touched.is_empty() {
            return Err(PyValueError::new_err(format!(
                "port at {:?} touches the open (pressure-release) face(s) {}, where p is pinned to \
                 0: pbar_free and R_room are both exactly zero, so the body would radiate into a \
                 short circuit \u{2014} perfectly conservative, perfectly silent, and invisible to \
                 the energy report. Move the port off that face, or give the face a finite \
                 impedance.",
                idx,
                face_list(&touched)
            )));
        }
        let shape = view.node_shape();
        let flat_idx = core::ravel(&[&node_idx[0], &node_idx[1], &node_idx[2]], shape);
        let h = view.h;
        check_disjoint(&room, &flat_idx, shape, &|count, node, other_index| {
            format!(
                "port at {idx:?} shares node {node:?} with the existing port at {other_index} \
                 ({count} node(s) in common). Overlapping ports are not independent within a step, \
                 so each one's solve uses a pressure that never occurred and the energy ledgers \
                 stop matching. Note grid snapping: two nearby centres collapse onto one node at \
                 h = {}.",
                py_float(h)
            )
        })?;

        let cols = [&node_idx[0][..], &node_idx[1][..], &node_idx[2][..]];
        let (w, big_w) = core::port_weights(&view, &cols);
        let r_room = core::r_room(&view, &cols, &w, &big_w);

        let port = Self {
            room: room.clone().unbind(),
            index,
            radius: radius_val,
            nodes: nodes_to_py(py, &node_idx)?,
            node_idx,
            flat: flat_to_py(py, &flat_idx)?,
            flat_idx,
            w: f64_to_py(py, w),
            r_room,
            queued_at: -1,
        };
        Ok(port)
    }

    /// Register with the room, which is the reference's last line of `__init__`.
    ///
    /// It has to be `__init__` rather than the tail of `#[new]` for a mechanical reason: a `#[new]`
    /// returns a `Self` and there is no Python object yet to hand to `room._ports.append`. Python
    /// calls `__new__` then `__init__` with the same arguments, so the construction happens in the
    /// former and the one side effect that needs an object in the latter. The arguments are
    /// swallowed here because `__new__` has already validated every one of them.
    #[pyo3(signature = (**_kwargs))]
    fn __init__(slf: Bound<'_, Self>, _kwargs: Option<Bound<'_, PyDict>>) -> PyResult<()> {
        let room = slf.borrow().room.clone_ref(slf.py());
        room.bind(slf.py())
            .getattr("_ports")?
            .call_method1("append", (&slf,))?;
        Ok(())
    }

    #[getter]
    fn room(&self, py: Python<'_>) -> Py<PyAny> {
        self.room.clone_ref(py)
    }

    #[getter]
    fn index(&self) -> (usize, usize, usize) {
        self.index
    }

    #[getter]
    fn radius(&self) -> Option<f64> {
        self.radius
    }

    #[getter]
    fn nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.nodes.clone_ref(py)
    }

    #[getter]
    fn w(&self, py: Python<'_>) -> Py<PyAny> {
        self.w.clone_ref(py)
    }

    #[getter]
    fn R_room(&self) -> f64 {
        self.r_room
    }

    #[getter]
    fn _flat(&self, py: Python<'_>) -> Py<PyAny> {
        self.flat.clone_ref(py)
    }

    #[getter]
    fn _queued_at(&self) -> i64 {
        self.queued_at
    }

    /// Settable because `AirBox.set_state` resets every registered port's mark (§30.3's rule
    /// applied to this tier: grep for assignment, not only for reference).
    #[setter]
    fn set__queued_at(&mut self, value: i64) {
        self.queued_at = value;
    }

    /// How many grid nodes the port actually covers (clipping at walls included).
    #[getter]
    fn node_count(&self) -> usize {
        self.node_idx[0].len()
    }

    /// The port's discrete volume `sum_n W_n` (m^3) — the staircased ball, made visible.
    #[getter]
    fn volume(&self, py: Python<'_>) -> PyResult<f64> {
        let room = self.room.bind(py);
        let g = guards(room)?;
        let view = g.view()?;
        let vals: Vec<f64> = self.flat_idx.iter().map(|&f| view.node_w[f]).collect();
        Ok(reduce::sum(&vals))
    }

    /// The open-circuit centered pressure `pbar_free` this port would feel with `q = 0`.
    fn free_pressure(&self, py: Python<'_>) -> PyResult<f64> {
        let room = self.room.bind(py);
        let g = guards(room)?;
        let view = g.view()?;
        let cols = [
            &self.node_idx[0][..],
            &self.node_idx[1][..],
            &self.node_idx[2][..],
        ];
        let pbar = core::free_pressure_nodes(&view, &cols);
        // `self.w` is read back from the Python object rather than from a cache: it is a plain
        // array attribute and the reference would honour an assignment to it.
        let w = self.w.bind(py);
        let w: Vec<f64> = w.extract()?;
        if w.len() != pbar.len() {
            return Err(PyValueError::new_err(
                "port.w must have one weight per port node.",
            ));
        }
        let terms: Vec<f64> = w.iter().zip(pbar.iter()).map(|(a, b)| a * b).collect();
        Ok(reduce::sum(&terms))
    }

    /// Raise if this port's previous injection is still pending — i.e. no `room.step()`.
    fn require_ready(&self, py: Python<'_>) -> PyResult<()> {
        let room = self.room.bind(py);
        let n: i64 = room.getattr("n")?.extract()?;
        require_ready_inner(
            room,
            self.queued_at,
            format!(
                "port at {:?} was asked to solve twice within one room step (room.n = {n}). A port \
                 does not step its room \u{2014} the caller does, once, after every port has \
                 solved:  for inst in instruments: inst.step(...)  then  room.step(). Without it \
                 the room is frozen and the body is loaded by a stale field, silently.",
                [self.index.0, self.index.1, self.index.2]
            ),
        )
    }

    /// Queue this port's volume velocity `q` (m^3/s) for the room's next `AirBox.step`.
    fn inject(&mut self, py: Python<'_>, q: f64) -> PyResult<()> {
        self.require_ready(py)?;
        let room = self.room.bind(py);
        queue(room, &self.nodes, self.w.clone_ref(py), q)?;
        self.queued_at = room.getattr("n")?.extract()?;
        Ok(())
    }

    /// Forget any pending-injection mark — for reusing the port on a fresh run.
    fn reset(&mut self) {
        self.queued_at = -1;
    }
}

// -- the distributed tier --------------------------------------------------------------------------

/// Everything `_PatchPort` holds, shared by the two distributed classes.
struct Patch {
    room: Py<PyAny>,
    spreading: Spreading,
    in_plane_axes: (usize, usize),
    coords: Py<PyAny>,
    areas: Py<PyAny>,
    n_surface: usize,
    origin: (f64, f64),
    face_coords: Py<PyAny>,
    nodes: Py<PyAny>,
    node_idx: core::Nodes,
    index: (usize, usize, usize),
    flat: Py<PyAny>,
    t: Py<PyAny>,
    r: Py<PyAny>,
    load_matrix: Py<PyAny>,
    footprint_empty: usize,
    where_: String,
    queued_at: i64,
}

/// What `_accept_surface` hands back: the placed footprint, its areas and origin, and the three
/// Python objects the port stores (`coords`, `areas`, `_face_coords`).
type AcceptedSurface = (
    Vec<[f64; 2]>,
    Vec<f64>,
    (f64, f64),
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
);

/// The surface protocol: validate, place the footprint in the plane, return its `face_coords`.
#[allow(clippy::too_many_arguments)]
fn accept_surface(
    py: Python<'_>,
    room: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
    areas: &Bound<'_, PyAny>,
    origin: Option<&Bound<'_, PyAny>>,
    spreading: Spreading,
    in_plane_axes: (usize, usize),
    where_: &str,
) -> PyResult<AcceptedSurface> {
    let np = py.import("numpy")?;
    let f64ty = np.getattr("float64")?;
    let coords_arr = np.call_method1("asarray", (coords, &f64ty))?;
    let coords_shape: Vec<usize> = coords_arr.getattr("shape")?.extract()?;
    if coords_shape.len() != 2 || coords_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "coords must be an (n_surface, 2) array of in-plane node positions (m), got shape {}.",
            shape_repr(&coords_shape)
        )));
    }
    let areas_arr = np.call_method1("asarray", (areas, &f64ty))?;
    let areas_shape: Vec<usize> = areas_arr.getattr("shape")?.extract()?;
    if areas_shape.len() != 1 || areas_shape[0] != coords_shape[0] {
        return Err(PyValueError::new_err(format!(
            "areas must have shape {} (one per surface node), got {}.",
            shape_repr(&[coords_shape[0]]),
            shape_repr(&areas_shape)
        )));
    }
    let coords_c = np.call_method1("ascontiguousarray", (&coords_arr,))?;
    let areas_c = np.call_method1("ascontiguousarray", (&areas_arr,))?;
    let coords_flat: Vec<f64> = coords_c.call_method0("ravel")?.extract()?;
    let areas_vec: Vec<f64> = areas_c.extract()?;
    if areas_vec.iter().any(|&a| a < 0.0 || !a.is_finite()) {
        return Err(PyValueError::new_err(
            "surface node areas must be finite and >= 0 (m^2).",
        ));
    }
    let n_surface = coords_shape[0];
    let coords_pairs: Vec<[f64; 2]> = (0..n_surface)
        .map(|i| [coords_flat[2 * i], coords_flat[2 * i + 1]])
        .collect();

    let n: Vec<usize> = room.getattr("N")?.extract()?;
    let h: f64 = room.getattr("h")?.extract()?;
    let (t0, t1) = in_plane_axes;
    let extent = [n[t0] as f64 * h, n[t1] as f64 * h];

    let origin_pair = match origin {
        None => {
            // Centred: the footprint's midpoint lands on the plane's midpoint, so the grid's own
            // mirror maps the surface to itself. Not an aesthetic default — it is what makes the
            // load equivariant and what lets the scene be symmetric.
            let mut out = [0.0f64; 2];
            for (d, slot) in out.iter_mut().enumerate() {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for c in &coords_pairs {
                    lo = lo.min(c[d]);
                    hi = hi.max(c[d]);
                }
                *slot = 0.5 * (extent[d] - (lo + hi));
            }
            (out[0], out[1])
        }
        Some(obj) => {
            let vals: Vec<f64> = obj
                .try_iter()?
                .map(|v| v?.extract::<f64>())
                .collect::<PyResult<_>>()?;
            if vals.len() != 2 {
                let shown: Vec<String> = vals.iter().map(|&v| py_float(v)).collect();
                let inner = if vals.len() == 1 {
                    format!("{},", shown[0])
                } else {
                    shown.join(", ")
                };
                return Err(PyValueError::new_err(format!(
                    "origin must be an (o0, o1) pair in the plane's own axes, got ({inner})."
                )));
            }
            (vals[0], vals[1])
        }
    };

    let face_coords: Vec<[f64; 2]> = coords_pairs
        .iter()
        .map(|c| [c[0] + origin_pair.0, c[1] + origin_pair.1])
        .collect();
    let tol = 1e-9 * h;
    let axes = [t0, t1];
    for d in 0..2 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for c in &face_coords {
            lo = lo.min(c[d]);
            hi = hi.max(c[d]);
        }
        if lo < -tol || hi > extent[d] + tol {
            let o = if d == 0 { origin_pair.0 } else { origin_pair.1 };
            return Err(PyValueError::new_err(format!(
                "the surface's footprint spans {}..{} m along {}, outside {where_}, which is \
                 0..{} m there. Move it with origin= (it currently sits at {} m on that axis), or \
                 enlarge the room.",
                py_general(lo, 6),
                py_general(hi, 6),
                core::AXES[axes[d]],
                py_general(extent[d], 6),
                py_general(o, 6),
            )));
        }
    }
    let _ = spreading;
    let fc_flat: Vec<f64> = face_coords.iter().flat_map(|c| [c[0], c[1]]).collect();
    let fc_obj = PyArray1::from_vec(py, fc_flat)
        .reshape([n_surface, 2])?
        .into_any()
        .unbind();
    Ok((
        face_coords,
        areas_vec,
        origin_pair,
        coords_c.unbind(),
        areas_c.unbind(),
        fc_obj,
    ))
}

/// Parse the `spreading` argument, distinguishing an omitted one from an explicit `None`.
///
/// §24.7: PyO3 collapses those two, so a plain `Option` would silently build the default where the
/// reference raises. `Option<Option<_>>` keeps them apart.
fn parse_spreading(spreading: Option<Option<Py<PyAny>>>, py: Python<'_>) -> PyResult<Spreading> {
    // The arm order is the surprising half and is exactly `beam`'s: PyO3 wraps the DEFAULT
    // expression, so `Some(None)` is "argument omitted" and a bare `None` is the caller's literal
    // `None`. Written the obvious way round it silently accepts `spreading=None` and builds the
    // bilinear default where the reference raises — which is what the first draft did, and eleven
    // tests in `test_airbox_surface.py` caught it because they pass `spreading=spreading` from a
    // fixture whose default is None.
    let obj = match spreading {
        Some(None) => return Ok(Spreading::Bilinear),
        None => py.None(),
        Some(Some(o)) => o,
    };
    let bound = obj.bind(py);
    let parsed = bound
        .extract::<String>()
        .ok()
        .and_then(|s| Spreading::parse(&s));
    parsed.ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown spreading {}; expected one of ('bilinear', 'nearest'). 'nearest' is the \
             measured negative control of the symmetry argument, not a configuration.",
            bound.repr().map(|r| r.to_string()).unwrap_or_default()
        ))
    })
}

/// The footprint refusal, shared by both distributed tiers.
fn check_footprint(
    patch_where: &str,
    spreading: Spreading,
    h: f64,
    unfed: usize,
    foot: usize,
) -> PyResult<()> {
    if unfed == 0 {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{unfed} of {foot} air node(s) under the surface's footprint on {patch_where} are fed by \
         no surface node, so the acoustic source would be a comb at the grid scale. The footprint \
         is measured span-wise (per row and per column of air nodes), so this is about the \
         surface's spacing and not its outline: too coarse for h_air = {} m (spreading='{}'). \
         Refine the surface, or coarsen the air grid.",
        py_general(h, 6),
        spreading.name()
    )))
}

/// The in-plane rim refusal.
fn check_in_plane_rim(
    i0: &[i64],
    i1: &[i64],
    axes: (usize, usize),
    n: [usize; 3],
    where_: &str,
) -> PyResult<()> {
    for (idx, ax) in [(i0, axes.0), (i1, axes.1)] {
        let lo = *idx.iter().min().unwrap_or(&0);
        let hi = *idx.iter().max().unwrap_or(&0);
        if lo < 1 || hi > n[ax] as i64 - 1 {
            return Err(PyValueError::new_err(format!(
                "the surface's spread stencil reaches air node index {lo}..{hi} along {} on \
                 {where_}, but a node on the plane's own rim (0 or {}) touches a SECOND wall: it \
                 carries half the node weight W and the sum of two wall admittances, so R_j stops \
                 being uniform across the patch and the spreading operator's reflection \
                 equivariance stops holding. Keep the footprint plus one air cell strictly inside \
                 the plane -- move it with origin=, enlarge the room, or shrink the surface.",
                core::AXES[ax],
                n[ax]
            )));
        }
    }
    Ok(())
}

/// The shared parts of both distributed constructors, from the spread entries to the load matrix.
struct Assembled {
    node_idx: core::Nodes,
    index: (usize, usize, usize),
    flat_idx: Vec<usize>,
    t: physsynth_core::sparse::Csr,
    r: Vec<f64>,
    load: physsynth_core::sparse::Csr,
    in_plane: (Vec<i64>, Vec<i64>),
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    view: &RoomView<'_>,
    face_coords: &[[f64; 2]],
    areas: &[f64],
    spreading: Spreading,
    axes: (usize, usize),
    where_: &str,
    place: &dyn Fn(&[i64], &[i64]) -> core::Nodes,
    r_from: &dyn Fn(&core::Nodes) -> core::Nodes,
    scale: f64,
) -> PyResult<Assembled> {
    let (t0, t1) = axes;
    let (rows, cols, vals) = core::spread(
        face_coords,
        areas,
        view.h,
        [view.n[t0], view.n[t1]],
        spreading,
    );
    let (i0, i1, plane) = core::plane_nodes(&rows, view.n[t1]);
    check_in_plane_rim(&i0, &i1, axes, view.n, where_)?;
    let t = core::build_t(&rows, &cols, &vals, &plane, face_coords.len());

    let node_idx = place(&i0, &i1);
    let index = (node_idx[0][0], node_idx[1][0], node_idx[2][0]);
    let flat_idx = core::ravel(
        &[&node_idx[0], &node_idx[1], &node_idx[2]],
        view.node_shape(),
    );
    let r_nodes = r_from(&node_idx);
    let r = core::patch_resistance(view, &[&r_nodes[0], &r_nodes[1], &r_nodes[2]]);
    let load = core::load_matrix(&t, &r, scale);
    Ok(Assembled {
        node_idx,
        index,
        flat_idx,
        t,
        r,
        load,
        in_plane: (i0, i1),
    })
}

/// A surface mounted flush in one of the room's walls.
///
/// Attribute-for-attribute compatible with `physsynth.core.airbox.SurfacePort`.
#[pyclass(name = "SurfacePort", module = "physsynth_rs", dict)]
pub struct PySurfacePort {
    patch: Patch,
    face: String,
    axis: usize,
}

/// A surface hanging on an interior plane of faces, radiating from both sides.
///
/// Attribute-for-attribute compatible with `physsynth.core.airbox.InteriorSurfacePort`.
#[pyclass(name = "InteriorSurfacePort", module = "physsynth_rs", dict)]
pub struct PyInteriorSurfacePort {
    patch: Patch,
    plane: String,
    axis: usize,
    face_index: usize,
    nodes_lo: Py<PyAny>,
    nodes_hi: Py<PyAny>,
    lo_idx: core::Nodes,
    in_plane: Py<PyAny>,
}

#[pymethods]
impl PySurfacePort {
    #[getter]
    fn room(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.room.clone_ref(py)
    }

    #[getter]
    fn spreading(&self) -> &'static str {
        self.patch.spreading.name()
    }

    #[getter]
    fn in_plane_axes(&self) -> (usize, usize) {
        self.patch.in_plane_axes
    }

    #[getter]
    fn coords(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.coords.clone_ref(py)
    }

    /// Settable: two tests zero a surface's areas to switch its coupling off.
    #[getter]
    fn areas(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.areas.clone_ref(py)
    }

    #[setter]
    fn set_areas(&mut self, value: Py<PyAny>) {
        self.patch.areas = value;
    }

    #[getter]
    fn n_surface(&self) -> usize {
        self.patch.n_surface
    }

    #[getter]
    fn origin(&self) -> (f64, f64) {
        self.patch.origin
    }

    #[getter]
    fn nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.nodes.clone_ref(py)
    }

    #[getter]
    fn index(&self) -> (usize, usize, usize) {
        self.patch.index
    }

    #[getter]
    fn _flat(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.flat.clone_ref(py)
    }

    #[getter]
    fn _face_coords(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.face_coords.clone_ref(py)
    }

    #[getter]
    fn _where(&self) -> String {
        self.patch.where_.clone()
    }

    #[getter]
    fn footprint_empty(&self) -> usize {
        self.patch.footprint_empty
    }

    /// Settable: `test_airbox_surface` rebuilds it from a rescaled `R`.
    #[getter]
    fn T(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.t.clone_ref(py)
    }

    #[setter]
    fn set_T(&mut self, value: Py<PyAny>) {
        self.patch.t = value;
    }

    #[getter]
    fn R(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.r.clone_ref(py)
    }

    #[setter]
    fn set_R(&mut self, value: Py<PyAny>) {
        self.patch.r = value;
    }

    #[getter]
    fn load_matrix(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.load_matrix.clone_ref(py)
    }

    #[setter]
    fn set_load_matrix(&mut self, value: Py<PyAny>) {
        self.patch.load_matrix = value;
    }

    #[getter]
    fn _queued_at(&self) -> i64 {
        self.patch.queued_at
    }

    #[setter]
    fn set__queued_at(&mut self, value: i64) {
        self.patch.queued_at = value;
    }

    /// How many air nodes the surface's spread source actually covers.
    #[getter]
    fn node_count(&self) -> usize {
        self.patch.node_idx[0].len()
    }

    /// The **radiating** area `sum_n area_n` (m^2) — which is not the bounding rectangle.
    ///
    /// Reads `areas` live rather than from a cache, so zeroing it works.
    #[getter]
    fn net_area(&self, py: Python<'_>) -> PyResult<f64> {
        let areas: Vec<f64> = self.patch.areas.bind(py).extract()?;
        Ok(reduce::sum(&areas))
    }

    /// Raise if this port's previous injection is still pending.
    fn require_ready(&self, py: Python<'_>) -> PyResult<()> {
        let room = self.patch.room.bind(py);
        let n: i64 = room.getattr("n")?.extract()?;
        require_ready_inner(
            room,
            self.patch.queued_at,
            format!(
                "the surface port on {} was asked to solve twice within one room step \
                 (room.n = {n}). A port does not step its room -- the caller does, once, \
                 after every port has solved:  for inst in instruments: inst.step(...)  \
                 then  room.step(). Without it the room is frozen and the surface is \
                 loaded by a stale field, silently.",
                self.patch.where_
            ),
        )
    }

    /// Forget any pending-injection mark — for reusing the port on a fresh run.
    fn reset(&mut self) {
        self.patch.queued_at = -1;
    }

    /// Register with the room; see `PyRoomPort::__init__` for why it is not in `#[new]`.
    fn _register(slf: Bound<'_, Self>) -> PyResult<()> {
        let room = slf.borrow().patch.room.clone_ref(slf.py());
        room.bind(slf.py())
            .getattr("_ports")?
            .call_method1("append", (&slf,))?;
        Ok(())
    }

    #[new]
    #[pyo3(signature = (*, room, face, coords, areas, origin=None::<Py<PyAny>>, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        face: &str,
        coords: Bound<'_, PyAny>,
        areas: Bound<'_, PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let (axis, end, t0, t1) = core::face_axes(face).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown face '{face}'; expected one of ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')."
            ))
        })?;
        let spreading = parse_spreading(spreading, py)?;
        let where_ = format!("face '{face}'");
        let origin = origin.map(|o| o.into_bound(py));
        let (face_coords, areas_vec, origin_pair, coords_obj, areas_obj, fc_obj) = accept_surface(
            py,
            &room,
            &coords,
            &areas,
            origin.as_ref(),
            spreading,
            (t0, t1),
            &where_,
        )?;

        let g = guards(&room)?;
        let view = g.view()?;
        let n_axis_end = if end == 0 { 0usize } else { view.n[axis] };
        let asm = assemble(
            &view,
            &face_coords,
            &areas_vec,
            spreading,
            (t0, t1),
            &where_,
            &|i0, i1| {
                let mut out: core::Nodes = [Vec::new(), Vec::new(), Vec::new()];
                out[axis] = vec![n_axis_end; i0.len()];
                out[t0] = i0.iter().map(|&v| v as usize).collect();
                out[t1] = i1.iter().map(|&v| v as usize).collect();
                out
            },
            &|nodes| nodes.clone(),
            1.0,
        )?;

        let (unfed, foot) = core::footprint_unfed(&asm.node_idx[t0], &asm.node_idx[t1], view.n[t1]);
        check_footprint(&where_, spreading, view.h, unfed, foot)?;
        let touched = touched_open_faces(&room, &asm.node_idx, view.n)?;
        if !touched.is_empty() {
            return Err(PyValueError::new_err(format!(
                "the surface on {where_} touches the open (pressure-release) face(s) {}, where p \
                 is pinned to 0: pbar_free and every R_j are exactly zero, so the surface would \
                 radiate into a short circuit -- perfectly conservative, perfectly silent, and \
                 invisible to the energy report. Give that face a finite impedance, or mount the \
                 surface elsewhere.",
                face_list(&touched)
            )));
        }
        let count = asm.node_idx[0].len();
        check_disjoint(
            &room,
            &asm.flat_idx,
            view.node_shape(),
            &|shared, node, other_index| {
                format!(
                    "the surface on {where_} ({count} nodes) shares node {node:?} with the \
                     existing port at {other_index} ({shared} node(s) in common). Overlapping \
                     ports are not independent within a step, so each one's solve uses a pressure \
                     that never occurred and the energy ledgers stop matching. Note the acoustic \
                     source is up to one air cell LARGER than the surface itself (bilinear spreads \
                     outboard), so footprints that merely look separate can still collide."
                )
            },
        )?;

        Ok(Self {
            patch: Patch {
                room: room.clone().unbind(),
                spreading,
                in_plane_axes: (t0, t1),
                coords: coords_obj,
                areas: areas_obj,
                n_surface: face_coords.len(),
                origin: origin_pair,
                face_coords: fc_obj,
                nodes: nodes_to_py(py, &asm.node_idx)?,
                node_idx: asm.node_idx,
                index: asm.index,
                flat: flat_to_py(py, &asm.flat_idx)?,
                t: csr_to_py(py, &asm.t)?,
                r: f64_to_py(py, asm.r),
                load_matrix: csr_to_py(py, &asm.load)?,
                footprint_empty: unfed,
                where_,
                queued_at: -1,
            },
            face: face.to_owned(),
            axis,
        })
    }

    /// The reference's last line: register with the room. See `PyRoomPort::__init__`.
    #[pyo3(signature = (**_kwargs))]
    fn __init__(slf: Bound<'_, Self>, _kwargs: Option<Bound<'_, PyDict>>) -> PyResult<()> {
        Self::_register(slf)
    }

    #[getter]
    fn face(&self) -> &str {
        &self.face
    }

    #[getter]
    fn axis(&self) -> usize {
        self.axis
    }

    /// The open-circuit centered pressure **vector** `pbar_free` over the patch, `O(patch)`.
    fn free_pressure(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let room = self.patch.room.bind(py);
        let g = guards(room)?;
        let view = g.view()?;
        let n = &self.patch.node_idx;
        let out = core::free_pressure_nodes(&view, &[&n[0], &n[1], &n[2]]);
        Ok(f64_to_py(py, out))
    }

    /// Queue the **per-node** volume-velocity vector `q` (m^3/s) for the room's next step.
    fn inject(&mut self, py: Python<'_>, q: Bound<'_, PyAny>) -> PyResult<()> {
        let np = py.import("numpy")?;
        let q = np.call_method1("asarray", (q, np.getattr("float64")?))?;
        let shape: Vec<usize> = q.getattr("shape")?.extract()?;
        let count = self.patch.node_idx[0].len();
        if shape.len() != 1 || shape[0] != count {
            return Err(PyValueError::new_err(format!(
                "q must be the per-node volume-velocity vector, shape {}, got {}. (Pass q = port.T \
                 @ v, not the scalar sum -- the scalar is exactly what the lumped tier would have \
                 coupled through, i.e. the negative control.)",
                shape_repr(&[count]),
                shape_repr(&shape)
            )));
        }
        self.require_ready(py)?;
        let room = self.patch.room.bind(py);
        queue(room, &self.patch.nodes, q.unbind(), 1.0)?;
        self.patch.queued_at = room.getattr("n")?.extract()?;
        Ok(())
    }
}

#[pymethods]
impl PyInteriorSurfacePort {
    #[getter]
    fn room(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.room.clone_ref(py)
    }

    #[getter]
    fn spreading(&self) -> &'static str {
        self.patch.spreading.name()
    }

    #[getter]
    fn in_plane_axes(&self) -> (usize, usize) {
        self.patch.in_plane_axes
    }

    #[getter]
    fn coords(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.coords.clone_ref(py)
    }

    /// Settable: two tests zero a surface's areas to switch its coupling off.
    #[getter]
    fn areas(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.areas.clone_ref(py)
    }

    #[setter]
    fn set_areas(&mut self, value: Py<PyAny>) {
        self.patch.areas = value;
    }

    #[getter]
    fn n_surface(&self) -> usize {
        self.patch.n_surface
    }

    #[getter]
    fn origin(&self) -> (f64, f64) {
        self.patch.origin
    }

    #[getter]
    fn nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.nodes.clone_ref(py)
    }

    #[getter]
    fn index(&self) -> (usize, usize, usize) {
        self.patch.index
    }

    #[getter]
    fn _flat(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.flat.clone_ref(py)
    }

    #[getter]
    fn _face_coords(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.face_coords.clone_ref(py)
    }

    #[getter]
    fn _where(&self) -> String {
        self.patch.where_.clone()
    }

    #[getter]
    fn footprint_empty(&self) -> usize {
        self.patch.footprint_empty
    }

    /// Settable: `test_airbox_surface` rebuilds it from a rescaled `R`.
    #[getter]
    fn T(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.t.clone_ref(py)
    }

    #[setter]
    fn set_T(&mut self, value: Py<PyAny>) {
        self.patch.t = value;
    }

    #[getter]
    fn R(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.r.clone_ref(py)
    }

    #[setter]
    fn set_R(&mut self, value: Py<PyAny>) {
        self.patch.r = value;
    }

    #[getter]
    fn load_matrix(&self, py: Python<'_>) -> Py<PyAny> {
        self.patch.load_matrix.clone_ref(py)
    }

    #[setter]
    fn set_load_matrix(&mut self, value: Py<PyAny>) {
        self.patch.load_matrix = value;
    }

    #[getter]
    fn _queued_at(&self) -> i64 {
        self.patch.queued_at
    }

    #[setter]
    fn set__queued_at(&mut self, value: i64) {
        self.patch.queued_at = value;
    }

    /// How many air nodes the surface's spread source actually covers.
    #[getter]
    fn node_count(&self) -> usize {
        self.patch.node_idx[0].len()
    }

    /// The **radiating** area `sum_n area_n` (m^2) — which is not the bounding rectangle.
    ///
    /// Reads `areas` live rather than from a cache, so zeroing it works.
    #[getter]
    fn net_area(&self, py: Python<'_>) -> PyResult<f64> {
        let areas: Vec<f64> = self.patch.areas.bind(py).extract()?;
        Ok(reduce::sum(&areas))
    }

    /// Raise if this port's previous injection is still pending.
    fn require_ready(&self, py: Python<'_>) -> PyResult<()> {
        let room = self.patch.room.bind(py);
        let n: i64 = room.getattr("n")?.extract()?;
        require_ready_inner(
            room,
            self.patch.queued_at,
            format!(
                "the surface port on {} was asked to solve twice within one room step \
                 (room.n = {n}). A port does not step its room -- the caller does, once, \
                 after every port has solved:  for inst in instruments: inst.step(...)  \
                 then  room.step(). Without it the room is frozen and the surface is \
                 loaded by a stale field, silently.",
                self.patch.where_
            ),
        )
    }

    /// Forget any pending-injection mark — for reusing the port on a fresh run.
    fn reset(&mut self) {
        self.patch.queued_at = -1;
    }

    /// Register with the room; see `PyRoomPort::__init__` for why it is not in `#[new]`.
    fn _register(slf: Bound<'_, Self>) -> PyResult<()> {
        let room = slf.borrow().patch.room.clone_ref(slf.py());
        room.bind(slf.py())
            .getattr("_ports")?
            .call_method1("append", (&slf,))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, room, plane, index, coords, areas, origin=None::<Py<PyAny>>, spreading=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        plane: &str,
        index: i64,
        coords: Bound<'_, PyAny>,
        areas: Bound<'_, PyAny>,
        origin: Option<Py<PyAny>>,
        spreading: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        // The room owns "unknown plane", exactly as it owns "outside the room".
        let axis: usize = room.call_method1("_plane_axis", (plane,))?.extract()?;
        let n: Vec<usize> = room.getattr("N")?.extract()?;
        let n_face = n[axis] as i64;
        if !(1..=n_face - 2).contains(&index) {
            return Err(PyValueError::new_err(format!(
                "interior surface index {index} on plane '{plane}' is out of range 1..{} (the room \
                 has {n_face} face(s) there). The two node planes straddling the surface must BOTH \
                 be strictly interior: a node plane on a wall carries half the node weight W and \
                 the wall's admittance in beta, so R_j would differ between the two sides and the \
                 load would stop being 2 T^T R T with a single R.",
                n_face - 2
            )));
        }
        let face_index = index as usize;
        let spreading = parse_spreading(spreading, py)?;
        let (t0, t1) = match axis {
            0 => (1usize, 2usize),
            1 => (0, 2),
            _ => (0, 1),
        };
        let where_ = format!("the plane '{plane}' cross-section at index {index}");
        let origin = origin.map(|o| o.into_bound(py));
        let (face_coords, areas_vec, origin_pair, coords_obj, areas_obj, fc_obj) = accept_surface(
            py,
            &room,
            &coords,
            &areas,
            origin.as_ref(),
            spreading,
            (t0, t1),
            &where_,
        )?;

        let g = guards(&room)?;
        let view = g.view()?;
        // The combined node set is both straddling planes, low first — what the disjointness check
        // and the room's bookkeeping see. `R` is built from the LOW plane alone: one resistance for
        // both, which the rim refusal above is exactly what makes true.
        let asm = assemble(
            &view,
            &face_coords,
            &areas_vec,
            spreading,
            (t0, t1),
            &where_,
            &|i0, i1| {
                let mut out: core::Nodes = [Vec::new(), Vec::new(), Vec::new()];
                for offset in 0..2usize {
                    for m in 0..i0.len() {
                        out[axis].push(face_index + offset);
                        out[t0].push(i0[m] as usize);
                        out[t1].push(i1[m] as usize);
                    }
                }
                out
            },
            &|nodes| {
                let half = nodes[0].len() / 2;
                [
                    nodes[0][..half].to_vec(),
                    nodes[1][..half].to_vec(),
                    nodes[2][..half].to_vec(),
                ]
            },
            2.0,
        )?;
        let half = asm.node_idx[0].len() / 2;
        let lo_idx: core::Nodes = [
            asm.node_idx[0][..half].to_vec(),
            asm.node_idx[1][..half].to_vec(),
            asm.node_idx[2][..half].to_vec(),
        ];
        let hi_idx: core::Nodes = [
            asm.node_idx[0][half..].to_vec(),
            asm.node_idx[1][half..].to_vec(),
            asm.node_idx[2][half..].to_vec(),
        ];
        let index_triple = (lo_idx[0][0], lo_idx[1][0], lo_idx[2][0]);

        let (unfed, foot) = core::footprint_unfed(&lo_idx[t0], &lo_idx[t1], view.n[t1]);
        check_footprint(&where_, spreading, view.h, unfed, foot)?;
        let count = asm.node_idx[0].len();
        check_disjoint(
            &room,
            &asm.flat_idx,
            view.node_shape(),
            &|shared, node, other_index| {
                format!(
                    "the surface on {where_} ({count} nodes) shares node {node:?} with the \
                     existing port at {other_index} ({shared} node(s) in common). Overlapping \
                     ports are not independent within a step, so each one's solve uses a pressure \
                     that never occurred and the energy ledgers stop matching. Note the acoustic \
                     source is up to one air cell LARGER than the surface itself (bilinear spreads \
                     outboard), so footprints that merely look separate can still collide."
                )
            },
        )?;

        let (i0, i1) = &asm.in_plane;
        let in_plane = PyTuple::new(
            py,
            [
                flat_to_py(py, &i0.iter().map(|&v| v as usize).collect::<Vec<_>>())?,
                flat_to_py(py, &i1.iter().map(|&v| v as usize).collect::<Vec<_>>())?,
            ],
        )?
        .into_any()
        .unbind();

        let out = Self {
            patch: Patch {
                room: room.clone().unbind(),
                spreading,
                in_plane_axes: (t0, t1),
                coords: coords_obj,
                areas: areas_obj,
                n_surface: face_coords.len(),
                origin: origin_pair,
                face_coords: fc_obj,
                nodes: nodes_to_py(py, &asm.node_idx)?,
                node_idx: asm.node_idx,
                index: index_triple,
                flat: flat_to_py(py, &asm.flat_idx)?,
                t: csr_to_py(py, &asm.t)?,
                r: f64_to_py(py, asm.r),
                load_matrix: csr_to_py(py, &asm.load)?,
                footprint_empty: unfed,
                where_,
                queued_at: -1,
            },
            plane: plane.to_owned(),
            axis,
            face_index,
            nodes_lo: nodes_to_py(py, &lo_idx)?,
            nodes_hi: nodes_to_py(py, &hi_idx)?,
            lo_idx,
            in_plane,
        };
        Ok(out)
    }

    /// The reference's last two lines: cut, then register. The cut is **last** among the things
    /// that can fail, because it is the only registration that mutates the room — if any refusal
    /// above fires, the room is left exactly as it was found.
    #[pyo3(signature = (**_kwargs))]
    fn __init__(slf: Bound<'_, Self>, _kwargs: Option<Bound<'_, PyDict>>) -> PyResult<()> {
        Self::_cut(slf.clone())?;
        Self::_register(slf)
    }

    /// Register this surface's blocked faces with the room.
    fn _cut(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let (room, axis, face_index, in_plane) = {
            let b = slf.borrow();
            (
                b.patch.room.clone_ref(py),
                b.axis,
                b.face_index,
                b.in_plane.clone_ref(py),
            )
        };
        let items: Vec<Bound<'_, PyAny>> =
            in_plane.bind(py).try_iter()?.collect::<PyResult<_>>()?;
        room.bind(py).call_method1(
            "_register_cut",
            (&slf, axis, face_index, &items[0], &items[1]),
        )?;
        Ok(())
    }

    #[getter]
    fn plane(&self) -> &str {
        &self.plane
    }

    #[getter]
    fn axis(&self) -> usize {
        self.axis
    }

    #[getter]
    fn face_index(&self) -> usize {
        self.face_index
    }

    #[getter]
    fn nodes_lo(&self, py: Python<'_>) -> Py<PyAny> {
        self.nodes_lo.clone_ref(py)
    }

    #[getter]
    fn nodes_hi(&self, py: Python<'_>) -> Py<PyAny> {
        self.nodes_hi.clone_ref(py)
    }

    #[getter]
    fn _in_plane(&self, py: Python<'_>) -> Py<PyAny> {
        self.in_plane.clone_ref(py)
    }

    /// How many velocity faces the surface cuts — half of `node_count`.
    #[getter]
    fn face_count(&self) -> usize {
        self.lo_idx[0].len()
    }

    /// The **cut** area (m^2) — `face_count * h^2`, and not `net_area`.
    #[getter]
    fn blocked_area(&self, py: Python<'_>) -> PyResult<f64> {
        let h: f64 = self.patch.room.bind(py).getattr("h")?.extract()?;
        Ok(self.lo_idx[0].len() as f64 * physsynth_core::pyfloat::scalar_pow(h, 2.0))
    }

    /// The open-circuit centered pressure on the **low** and **high** node planes, `O(patch)`.
    fn free_pressure(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let room = self.patch.room.bind(py);
        let g = guards(room)?;
        let view = g.view()?;
        let half = self.lo_idx[0].len();
        let n = &self.patch.node_idx;
        let hi: core::Nodes = [
            n[0][half..].to_vec(),
            n[1][half..].to_vec(),
            n[2][half..].to_vec(),
        ];
        let lo_vals =
            core::free_pressure_nodes(&view, &[&self.lo_idx[0], &self.lo_idx[1], &self.lo_idx[2]]);
        let hi_vals = core::free_pressure_nodes(&view, &[&hi[0], &hi[1], &hi[2]]);
        Ok(
            PyTuple::new(py, [f64_to_py(py, lo_vals), f64_to_py(py, hi_vals)])?
                .into_any()
                .unbind(),
        )
    }

    /// Queue the **per-face** volume-velocity vector `q` (m^3/s) as a `-q`/`+q` pair.
    fn inject(&mut self, py: Python<'_>, q: Bound<'_, PyAny>) -> PyResult<()> {
        let np = py.import("numpy")?;
        let q = np.call_method1("asarray", (q, np.getattr("float64")?))?;
        let shape: Vec<usize> = q.getattr("shape")?.extract()?;
        let faces = self.lo_idx[0].len();
        if shape.len() != 1 || shape[0] != faces {
            return Err(PyValueError::new_err(format!(
                "q must be the per-FACE volume-velocity vector, shape {}, got {}. Note this is \
                 HALF the node count ({}): the two node planes share one q, with opposite signs.",
                shape_repr(&[faces]),
                shape_repr(&shape),
                self.patch.node_idx[0].len()
            )));
        }
        self.require_ready(py)?;
        let neg = q.call_method0("__neg__")?;
        let room = self.patch.room.bind(py);
        queue(room, &self.nodes_lo, neg.unbind(), 1.0)?;
        queue(room, &self.nodes_hi, q.unbind(), 1.0)?;
        self.patch.queued_at = room.getattr("n")?.extract()?;
        Ok(())
    }
}
