"""Importing the package caps BLAS thread counts (prevents CPU oversubscription).

Training parallelises across labels via joblib's threading backend; nested BLAS
threads would oversubscribe the CPU and freeze the UI / starve the API loop.
``app/__init__.py`` sets these to 1 before numpy is imported. This pins that.
"""

import os

import app


def test_blas_threads_capped_on_import():
    assert app.__version__  # importing the package runs the thread-cap setup
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ.get(var) == "1", f"{var} should be capped to 1 on import"
