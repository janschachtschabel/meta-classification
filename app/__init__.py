"""api_v3 — lightweight, secure, CPU-only metadata text classification."""

import os

# Cap BLAS/thread-pool threads BEFORE numpy/scikit-learn are imported anywhere.
# Training parallelises across labels via joblib's 'threading' backend; if each
# per-label fit also let BLAS spawn one thread per core, the threads would nest
# and oversubscribe the CPU — freezing the desktop UI and starving the API event
# loop. One BLAS thread per worker is the correct setup here. setdefault keeps it
# overridable via real environment variables.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

__version__ = "3.1.0"
