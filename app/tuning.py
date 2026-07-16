"""Automatic optimization: regularization selection, threshold tuning, metrics.

Deliberately uses small, exhaustive grids (over ``C`` and over thresholds)
evaluated on a held-out validation split. For these tiny, mostly 1-D search
spaces a grid is simpler, deterministic and at least as good as a sampler like
Optuna. The held-out *test* split (never seen here) is used only for the final
reported metrics, so numbers are not optimistically biased.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from .classifier import make_head
from .errors import TrainingInputError
from .vectorizers import TfidfBackend

_DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def select_c(
    x_train, y_train, x_val, y_val, c_grid: list[float], n_jobs: int = 1,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float, float], None] | None = None,
    solver: str = "liblinear",
):
    """Fit the head for each C, score macro-F1 on val; return the best.

    Stops early (between C candidates) if ``should_stop`` returns True.
    Calls ``on_step(index, total, c, best_f1)`` after each candidate (sub-progress).
    Returns ``(best_c, best_f1, fitted_head)``.

    :raises ValueError: if ``c_grid`` is empty (e.g. a misconfigured profile),
        instead of failing with an opaque IndexError / a ``None`` head downstream.
    """
    if len(c_grid) == 0:
        raise ValueError("empty C grid: at least one C candidate is required")
    best_c: float = c_grid[0]
    best_f1 = -1.0
    best_head = None
    total = len(c_grid)
    for index, c in enumerate(c_grid, start=1):
        if should_stop is not None and should_stop():
            break
        head = make_head(c, n_jobs=n_jobs, solver=solver)
        head.fit(x_train, y_train)
        proba = head.predict_proba(x_val)
        score = macro_f1(y_val, (proba >= 0.5).astype(int))
        if score > best_f1:
            best_f1, best_c, best_head = score, c, head
        if on_step is not None:
            on_step(index, total, c, best_f1)
    return best_c, best_f1, best_head


def cross_val_evaluate(
    make_vectorizer: Callable[[], TfidfBackend],
    texts: list[str],
    y: np.ndarray,
    classes: list[str],
    *,
    k: int,
    c_grid: list[float],
    seed: int = 42,
    n_jobs: int = 1,
    solver: str = "liblinear",
    tune_threshold: bool = True,
    per_label: bool = True,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, str], None] | None = None,
) -> tuple[float, float, dict[str, float], dict] | None:
    """k-fold out-of-fold evaluation using ALL rows for both training and metrics.

    Every row is predicted exactly once by a model that did not train on it; the
    vectorizer is refit per fold, so there is no feature leakage. ``C`` is picked by
    OOF macro-F1, thresholds are tuned on the OOF probabilities, and the metrics are
    computed on them. Returns ``(best_c, global_threshold, per_label_thresholds,
    metrics)`` (the caller then fits the deploy model on 100% of the data).

    ``on_step(done, total, detail)`` fires after EVERY head fit (k x |grid| of
    them), so the caller can show real progress across a run that takes minutes.

    Fairness note: ``C`` and the thresholds are selected on the same OOF predictions
    the metrics report, so those two choices carry a mild in-sample optimism; the
    honest core — no row is ever scored by a model that saw it — holds. Use the
    classic train/val/test split for strict separation. Returns ``None`` if cancelled.

    :raises ValueError: if ``c_grid`` is empty.
    :raises TrainingInputError: if ``k`` exceeds the number of rows.
    """
    if len(c_grid) == 0:
        raise ValueError("empty C grid: at least one C candidate is required")
    n, n_labels = y.shape
    if k > n:
        raise TrainingInputError(f"cv_folds={k} exceeds the {n} usable rows; reduce cv_folds.")
    folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))
    # OOF probabilities per C, each row filled exactly once (float32 bounds RAM).
    oof = {c: np.zeros((n, n_labels), dtype=np.float32) for c in c_grid}
    total_fits = k * len(c_grid)
    done = 0
    for fold, (tr, te) in enumerate(folds, start=1):
        if should_stop is not None and should_stop():
            return None
        vec = make_vectorizer()
        x_tr = vec.fit_transform([texts[i] for i in tr])
        x_te = vec.transform([texts[i] for i in te])
        for c in c_grid:
            # Checked per C candidate (not only per fold) so /train/stop keeps its
            # "between the C fits" promise in CV mode as well.
            if should_stop is not None and should_stop():
                return None
            head = make_head(c, n_jobs=n_jobs, solver=solver)
            head.fit(x_tr, y[tr])
            oof[c][te] = head.predict_proba(x_te).astype(np.float32)
            done += 1
            if on_step is not None:
                on_step(done, total_fits, f"Fold {fold}/{k}: C={c} ({done}/{total_fits} fits)")
    best_c = max(c_grid, key=lambda c: macro_f1(y, (oof[c] >= 0.5).astype(int)))
    proba = oof[best_c]
    if tune_threshold:
        global_t, per_label_t = tune_thresholds(y, proba, classes, per_label=per_label)
    else:
        global_t, per_label_t = 0.5, {}
    metrics = compute_metrics(y, proba, classes, global_t, per_label_t)
    return best_c, global_t, per_label_t, metrics


def tune_thresholds(
    y_val: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    *,
    per_label: bool = True,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the global (and optionally per-label) threshold maximizing F1 on val."""
    grid = _DEFAULT_GRID if grid is None else grid

    best_global, best_global_f1 = 0.5, -1.0
    for threshold in grid:
        score = macro_f1(y_val, (proba >= threshold).astype(int))
        if score > best_global_f1:
            best_global_f1, best_global = score, float(threshold)

    per_label_thresholds: dict[str, float] = {}
    if per_label:
        for col, uri in enumerate(classes):
            truth = y_val[:, col]
            scores = proba[:, col]
            best_t, best_f1 = best_global, -1.0
            for threshold in grid:
                score = f1_score(truth, (scores >= threshold).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1, best_t = score, float(threshold)
            per_label_thresholds[uri] = best_t

    return best_global, per_label_thresholds


def apply_thresholds(
    proba: np.ndarray, classes: list[str], global_threshold: float, per_label: dict[str, float]
) -> np.ndarray:
    """Turn probabilities into a binary prediction matrix using thresholds."""
    preds = np.zeros_like(proba, dtype=int)
    for col, uri in enumerate(classes):
        threshold = per_label.get(uri, global_threshold)
        preds[:, col] = (proba[:, col] >= threshold).astype(int)
    return preds


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    global_threshold: float,
    per_label: dict[str, float],
) -> dict:
    """Macro/micro P/R/F1 plus per-label F1, computed on the given split."""
    preds = apply_thresholds(proba, classes, global_threshold, per_label)
    per_label_f1 = f1_score(y_true, preds, average=None, zero_division=0)
    return {
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "n_labels": len(classes),
        "per_label_f1": {uri: float(score) for uri, score in zip(classes, per_label_f1, strict=False)},
    }
