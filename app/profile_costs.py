"""What a training run costs: the measured anchors and the arithmetic over them.

Split out of ``profiles.py`` (audit ARC-4), which carried three reasons to change in
one file. This one changes when somebody **re-measures** — every constant here is a
figure from a named benchmark run, and the docstrings say which. It is a stdlib-only
leaf, so read-only dataset inspection can estimate a run without reaching the server's
configuration or its memory module.
"""

from __future__ import annotations

# What a run costs, anchored on the ONE full-scale measurement this project has:
# faecher_300k_auto, 156 373 rows x 60 labels, `auto`, 40.2 min wall-clock (README,
# "How large a dataset does this handle?"). Row scaling is linear — benchmark_row_scaling
# measured exponent 1.00 for memory and vectorization — so minutes scale with the rows.
ANCHOR_ROWS = 156_373
ANCHOR_MINUTES = 40.2
# The anchor's head fits ran on 9 threads: 16 cores under the default 60 % CPU budget.
ANCHOR_THREADS = 9
# Relative to `auto`, from the head-fit and vectorization-pass counts in that same table
# (~4 min / 40 min / ~1.2 h at the anchor size). The RATIO is solid because it follows
# from the loop in tuning.cross_val_evaluate; the absolute minutes are an estimate.
_PROFILE_COST = {"fast": 0.1, "auto": 1.0, "best": 1.8}
# Share of the anchor's minutes spent in head fits: 7 fit units at ~4.9 min of the ~43
# the README's cost table adds up to. The rest is vectorization, which runs on one core
# whatever the thread count.
_HEAD_FIT_SHARE = 0.79
# Serial fraction of a head fit (Amdahl), measured 2026-09-11 with
# benchmark_training_memory.py: one thread against six, the same fit — 60.4 s / 19.2 s at
# 30k rows (0.18), 387 s / 120.5 s at 100k rows (0.17). Threads do not divide the time.
_HEAD_FIT_SERIAL = 0.175
# What one row adds to a run at its deploy fit, measured on data_300k.csv (2026-09-11):
# the text the run keeps (422 MB for 156 174 rows) and the feature matrix (436 MB for
# 156 373 rows at 365 non-zeros per row; word-only ~70 non-zeros).
_TEXT_BYTES_PER_ROW = 2_800
_MATRIX_BYTES_PER_ROW = 2_900
_WORD_ONLY_MATRIX_BYTES_PER_ROW = 600
# What a training in a child process (APIV3_TRAINING_ISOLATION=process) holds before it
# has read a row: its own interpreter with numpy, scipy, scikit-learn and pandas. 166 MB
# in the Linux container, 155 MB on Windows (scripts/benchmark_training_isolation.py:
# a fresh process holding the worker's imports, "API process at start").
CHILD_PROCESS_BASE_BYTES = 166 * 1024 * 1024


def head_fit_seconds_ratio(threads: int, than: int) -> float:
    """How much longer one head fit takes on ``threads`` than on ``than`` threads."""

    def seconds(count: int) -> float:
        return _HEAD_FIT_SERIAL + (1 - _HEAD_FIT_SERIAL) / max(1, count)

    return seconds(threads) / seconds(than)


def estimated_minutes(profile_name: str, n_rows: int, threads: int | None = None) -> float | None:
    """Roughly how long training ``n_rows`` rows on this profile takes, in minutes.

    An **estimate**, not a schedule, and it is worth knowing what it cannot see: the
    label count (the head's coefficients are ``n_labels x n_features``), the speed of the
    machine's cores, and the corpus's non-zeros per document — 365 on the anchor run
    against 278 on a shorter corpus. What it is good for is the decision it exists to
    support: whether a run is a coffee break or an afternoon.

    ``threads`` is the head-fit thread count the run will get (``CapacityPlan``); the
    head-fit share of the anchor's minutes then scales along the measured thread curve.
    ``None`` keeps the anchor's 9.

    ``None`` for a profile the cost model has no factor for: a profile added to
    ``config.yaml`` has no measured cost until somebody measures it, and answering with
    ``auto``'s number would put a figure on screen that nothing supports.
    """
    factor = _PROFILE_COST.get(profile_name)
    if factor is None:
        return None
    minutes = ANCHOR_MINUTES * factor * (n_rows / ANCHOR_ROWS)
    if threads is not None:
        slowdown = head_fit_seconds_ratio(threads, ANCHOR_THREADS)
        minutes *= (1 - _HEAD_FIT_SHARE) + _HEAD_FIT_SHARE * slowdown
    return round(minutes, 1)

