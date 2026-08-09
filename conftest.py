"""Root conftest.

Its mere presence puts the repository root on ``sys.path`` (pytest prepend import mode), so
``import physsynth`` works without an editable install. Test files additionally import the
sibling ``tests/helpers.py`` module (the ``tests`` directory is added to the path for each test).

It also pins the BLAS thread count **inside xdist workers only** -- see below.
"""

import os

# -- BLAS threads under xdist ----------------------------------------------------------------------
#
# SciPy/NumPy ship OpenBLAS, which by default opens one thread per core (16 on the dev box). That is
# a win for a SERIAL run -- measured ~14% FASTER on tests/test_geometric_limits.py than with one
# thread -- so the default `pytest` invocation deliberately leaves it alone.
#
# Under `pytest -n N` it inverts: N workers x 16 threads each oversubscribes the machine badly, and
# the threads spend their time contending rather than computing. Each worker therefore gets exactly
# ONE BLAS thread and the parallelism comes from the worker pool instead.
#
# This must run before NumPy is imported -- OpenBLAS reads these variables when its shared library
# loads, not per call. The root conftest is imported before any test module (verified: at this point
# `"numpy" in sys.modules` is False in both the controller and every worker), which makes this the
# last safe moment. `setdefault` keeps an explicitly exported value from the caller winning.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")
