"""Where the decision cuts sit, and what they do to probabilities.

A trained head answers with a probability per label; this module owns the second
half of that answer — the threshold each label is read at, how those thresholds are
searched for, what they are called once a bundle stores them, and how they turn a
probability matrix into a set of asserted labels.

Kept apart from ``tuning`` because the two change for different reasons: the C
search changes when the question is which model to fit, this changes when the
question is where to cut. It imports nothing of ours, so the dependency runs one
way — ``tuning`` and ``deploy`` read it, it reads neither.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

# Nineteen fixed cuts, 0.05 apart. Coarse on purpose, which is a measurement and not an
# oversight: plan item C2 proposed replacing this with every observed score as a
# candidate (one descending sweep per label finds the F1-argmax cut exactly). Measured
# 2026-09-10 on data_30k_ai, thresholds tuned on a validation split and read on a
# held-out one, that is worth **+0.0037 in-sample / -0.0115 held out** (seed 42) and
# **+0.0042 / -0.0076** (seed 7) in macro F1 — the sign is the same both times. The
# finer search wins where it is fitted and loses where it counts, because the extra
# resolution buys the noise in the tuning split. The in-sample number is the check on
# the sweep itself: a search over every cut cannot lose to a search over 19 of them on
# the rows both saw, so if it had not won there the comparison would have been void.
# `scripts/benchmark_threshold_grid.py` re-runs it, on any seed or target. That was a
# measurement about QUALITY and it still stands: `_cut_scores` below sweeps these same 19
# cuts, it just counts them in one pass instead of scoring each one separately.
_DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _cut_scores(
    y: np.ndarray, proba: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """F1 of every (label, cut) pair as an ``(n_labels, len(grid))`` table, and each
    label's positives. ``grid`` must ascend.

    Counting instead of comparing. The search used to build a full ``(n, n_labels)``
    decision matrix per cut and hand it to sklearn, which derived counts from it — 19
    times for the global sweep and 19 more per label, i.e. a Python call and a full
    column pass per label per cut. Measured at 156,373 x 60 that is 33.8 s, and with
    ``select_c_on_tuned_thresholds`` (the default) the whole search runs once per C
    candidate.

    The counts are what the answer needs, and they come out of one pass: ``searchsorted``
    bins each probability by how many cuts it clears, and a reverse-cumulated
    ``bincount`` turns that into asserted rows and hits for all 19 cuts at once. F1 is
    then ``2 TP / (asserted + positives)``, which is what sklearn computes for a binary
    column — including the 0 it returns via ``zero_division`` when a label has neither.

    One column at a time on purpose: bucketing the whole matrix in one call would
    allocate an ``(n, n_labels)`` index array, 600 MB at 250,000 x 300 — the very
    allocation ``apply_thresholds`` and ``data.prepare_targets`` go out of their way
    to avoid.
    """
    size = len(grid)
    positives = y.sum(axis=0).astype(np.float64)
    scores = np.zeros((proba.shape[1], size), dtype=np.float64)
    for col in range(proba.shape[1]):
        # `bucket` counts the cuts a probability clears, so cut g asserts the row exactly
        # when g < bucket; reverse-cumulating the bucket counts gives every cut's totals.
        bucket = np.searchsorted(grid, proba[:, col], side="right")
        asserted = np.bincount(bucket, minlength=size + 1)[::-1].cumsum()[::-1][1:]
        hits = np.bincount(bucket, weights=y[:, col],
                           minlength=size + 1)[::-1].cumsum()[::-1][1:]
        denominator = asserted + positives[col]
        np.divide(2.0 * hits, denominator, out=scores[col], where=denominator > 0)
    return scores, positives


def tune_threshold_columns(
    y_val: np.ndarray,
    proba: np.ndarray,
    *,
    per_label: bool = True,
    grid: np.ndarray | None = None,
    shrink_k: float | None = None,
    keep_global: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """The global threshold, and one threshold per COLUMN of ``proba``.

    A threshold belongs to a column; the ``uri -> threshold`` dict a bundle stores is
    that same answer named. Keeping the two apart lets a caller that has no label names
    — the C search, which wants to score each candidate under its own thresholds — reach
    the numbers without carrying the vocabulary along.

    With ``per_label=False`` every column carries the global value: an array has no way
    to say "absent", and the global cut is what the absent entry would have meant.

    ``shrink_k`` blends each label's own cut toward the global one by how much that cut
    can be trusted: weight ``n_pos / (n_pos + shrink_k)``, where ``n_pos`` is the label's
    positives in ``y_val``. A cut fitted on four positives is mostly noise and a cut
    fitted on six hundred is not, and the F1-argmax rule cannot tell the difference on
    its own. ``None`` disables it, which is what every caller did before it existed.

    ``keep_global`` (bool per column) gives those columns the global cut whatever their
    positives say: a label that reached the training minimum only through AI-marked rows
    has a handful of real positives, and a run can decide its own cut would rest on too
    few (``provenance.RowProvenance.keep_global``). ``None`` keeps every own cut.
    """
    grid = _DEFAULT_GRID if grid is None else np.sort(np.asarray(grid, dtype=float))
    if grid.size == 0:
        # No cut to beat it, so the global threshold keeps the value it starts at — which
        # is what the scan below produced for an empty grid before it was a scan.
        return 0.5, np.full(proba.shape[1], 0.5)

    cut_f1, positives = _cut_scores(y_val, proba, grid)
    # A macro average IS the mean of the per-label F1 over every column, so the global cut
    # falls out of the same table as the per-label ones. It used to be 19 more full passes
    # over the matrix, each building a decision matrix sklearn then re-derived counts from.
    # `argmax` takes the first maximum, i.e. the LOWEST cut that reaches it — the tie-break
    # the ascending scan with its strict `>` had.
    best_global = float(grid[int(np.argmax(cut_f1.mean(axis=0)))])

    columns = np.full(proba.shape[1], best_global, dtype=float)
    if per_label:
        own = grid[np.argmax(cut_f1, axis=1)]
        if shrink_k is not None:
            trust = positives / (positives + shrink_k)
            own = trust * own + (1 - trust) * best_global
        # No positives to tune on: every threshold scores f1=0, and the argmax would hand
        # the label the grid MINIMUM (0.05) — a near-zero threshold that fires on
        # everything. Keep the global. Under ``shrink_k`` this is the same rule at weight
        # 0, not a second case. A column the caller pinned keeps the global cut too.
        chosen = positives > 0
        if keep_global is not None:
            chosen &= ~np.asarray(keep_global, dtype=bool)
        columns[chosen] = own[chosen]

    return best_global, columns


def tuned_score(
    y_true: np.ndarray, proba: np.ndarray, *, per_label: bool, shrink_k: float | None = None,
    keep_global: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Macro F1 a candidate reaches under thresholds tuned for ITSELF, and those
    thresholds.

    One vectorised comparison against a per-column vector — the same decision
    ``apply_thresholds`` makes once the columns carry names.
    """
    global_t, columns = tune_threshold_columns(
        y_true, proba, per_label=per_label, shrink_k=shrink_k, keep_global=keep_global)
    return macro_f1(y_true, (proba >= columns).astype(np.int8)), global_t, columns


def name_threshold_columns(columns: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Attach label URIs to threshold columns — the form a bundle persists."""
    return {uri: float(t) for uri, t in zip(classes, columns, strict=True)}


def tune_thresholds(
    y_val: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    *,
    per_label: bool = True,
    grid: np.ndarray | None = None,
    shrink_k: float | None = None,
    keep_global: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the global (and optionally per-label) threshold maximizing F1 on val.

    The named form of :func:`tune_threshold_columns` — what a bundle persists. An empty
    dict means "every label decides at the global cut", which is what ``per_label=False``
    produces and what ``apply_thresholds`` falls back to.
    """
    best_global, columns = tune_threshold_columns(
        y_val, proba, per_label=per_label, grid=grid, shrink_k=shrink_k, keep_global=keep_global)
    if not per_label:
        return best_global, {}
    return best_global, name_threshold_columns(columns, classes)


def apply_thresholds(
    proba: np.ndarray, classes: list[str], global_threshold: float, per_label: dict[str, float]
) -> np.ndarray:
    """Turn probabilities into a binary prediction matrix using thresholds."""
    # int8, not the int64 numpy gives for `int`: this is the same shape as `proba` and
    # holds only 0/1, so int64 spends 8 bytes per bit — 600 MB per allocation at 250k rows
    # x 300 labels, freed and re-made once per threshold candidate, in the middle of the
    # phase the training memory budget protects. `data.prepare_targets` makes the same
    # argument for the target matrix; sklearn scores an int8 indicator identically.
    preds = np.zeros_like(proba, dtype=np.int8)
    for col, uri in enumerate(classes):
        threshold = per_label.get(uri, global_threshold)
        preds[:, col] = (proba[:, col] >= threshold).astype(np.int8)
    return preds
