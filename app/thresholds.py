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

_DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def tune_threshold_columns(
    y_val: np.ndarray,
    proba: np.ndarray,
    *,
    per_label: bool = True,
    grid: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """The global threshold, and one threshold per COLUMN of ``proba``.

    A threshold belongs to a column; the ``uri -> threshold`` dict a bundle stores is
    that same answer named. Keeping the two apart lets a caller that has no label names
    — the C search, which wants to score each candidate under its own thresholds — reach
    the numbers without carrying the vocabulary along.

    With ``per_label=False`` every column carries the global value: an array has no way
    to say "absent", and the global cut is what the absent entry would have meant.
    """
    grid = _DEFAULT_GRID if grid is None else grid

    best_global, best_global_f1 = 0.5, -1.0
    for threshold in grid:
        score = macro_f1(y_val, (proba >= threshold).astype(int))
        if score > best_global_f1:
            best_global_f1, best_global = score, float(threshold)

    columns = np.full(proba.shape[1], best_global, dtype=float)
    if per_label:
        for col in range(proba.shape[1]):
            truth = y_val[:, col]
            if truth.sum() == 0:
                # No positives to tune on: every threshold scores f1=0, and the
                # ">" update would hand the label the grid MINIMUM (0.05) —
                # near-zero threshold, fires on everything. Keep the global.
                continue
            scores = proba[:, col]
            best_t, best_f1 = best_global, -1.0
            for threshold in grid:
                score = f1_score(truth, (scores >= threshold).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1, best_t = score, float(threshold)
            columns[col] = best_t

    return best_global, columns


def tuned_score(
    y_true: np.ndarray, proba: np.ndarray, *, per_label: bool
) -> tuple[float, float, np.ndarray]:
    """Macro F1 a candidate reaches under thresholds tuned for ITSELF, and those
    thresholds.

    One vectorised comparison against a per-column vector — the same decision
    ``apply_thresholds`` makes once the columns carry names.
    """
    global_t, columns = tune_threshold_columns(y_true, proba, per_label=per_label)
    return macro_f1(y_true, (proba >= columns).astype(int)), global_t, columns


def name_threshold_columns(columns: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Attach label URIs to threshold columns — the form a bundle persists."""
    return {uri: float(t) for uri, t in zip(classes, columns, strict=False)}


def tune_thresholds(
    y_val: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    *,
    per_label: bool = True,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the global (and optionally per-label) threshold maximizing F1 on val.

    The named form of :func:`tune_threshold_columns` — what a bundle persists. An empty
    dict means "every label decides at the global cut", which is what ``per_label=False``
    produces and what ``apply_thresholds`` falls back to.
    """
    best_global, columns = tune_threshold_columns(y_val, proba, per_label=per_label, grid=grid)
    if not per_label:
        return best_global, {}
    return best_global, name_threshold_columns(columns, classes)


def apply_thresholds(
    proba: np.ndarray, classes: list[str], global_threshold: float, per_label: dict[str, float]
) -> np.ndarray:
    """Turn probabilities into a binary prediction matrix using thresholds."""
    preds = np.zeros_like(proba, dtype=int)
    for col, uri in enumerate(classes):
        threshold = per_label.get(uri, global_threshold)
        preds[:, col] = (proba[:, col] >= threshold).astype(int)
    return preds
