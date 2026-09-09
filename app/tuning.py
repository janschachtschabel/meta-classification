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


def is_single_label(task_type: str) -> bool:
    """binary/multiclass = exactly one label per prediction (serving uses argmax)."""
    return task_type in ("binary", "multiclass")


def _argmax_onehot(proba: np.ndarray) -> np.ndarray:
    """One-hot argmax decision — the rule serving applies to binary/multiclass
    (``ClassifierModel.predict`` returns the single best label there and
    ignores thresholds entirely)."""
    preds = np.zeros_like(proba, dtype=int)
    preds[np.arange(proba.shape[0]), proba.argmax(axis=1)] = 1
    return preds


def _default_decision(proba: np.ndarray, task_type: str) -> np.ndarray:
    """Decision used while SELECTING (before thresholds exist): argmax for
    single-label tasks, the neutral 0.5 cut for multilabel."""
    return _argmax_onehot(proba) if is_single_label(task_type) else (proba >= 0.5).astype(int)


def select_c(
    x_train, y_train, x_val, y_val, c_grid: list[float], n_jobs: int = 1,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float, float], None] | None = None,
    solver: str = "liblinear",
    task_type: str = "multilabel",
    tol: float | None = None,
):
    """Fit the head for each C, score macro-F1 on val; return the best.

    The score uses the decision rule serving will apply for ``task_type``
    (argmax for binary/multiclass, 0.5 threshold for multilabel), so C is
    optimized for the rule that actually answers requests.
    Stops early (between C candidates) if ``should_stop`` returns True.
    Calls ``on_step(index, total, c, best_f1)`` after each candidate (sub-progress).
    ``tol`` loosens the convergence of these throw-away fits only — the caller's
    deploy fit is a separate call and never sees it.
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
        head = make_head(c, n_jobs=n_jobs, solver=solver, tol=tol)
        head.fit(x_train, y_train)
        proba = head.predict_proba(x_val)
        score = macro_f1(y_val, _default_decision(proba, task_type))
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
    task_type: str = "multilabel",
    matrix=None,
    tol: float | None = None,
    select_on_tuned_thresholds: bool = False,
) -> tuple[float, float, dict[str, float], dict] | None:
    """k-fold out-of-fold evaluation using ALL rows for both training and metrics.

    Every row is predicted exactly once by a model that did not train on it; by default
    the vectorizer is refit per fold, so there is no feature leakage.

    ``matrix`` short-circuits that: a matrix already fitted over ALL rows is sliced per
    fold instead, which turns k vectorization passes into one — 2.3 min each at 156 k
    rows. The price is that the vocabulary and the IDF weights were computed with the
    fold's test rows in view, so the out-of-fold metric carries a mild optimism. Whether
    that optimism is material at this scale is a measurement, not an opinion:
    ``scripts/benchmark_shared_vectorizer.py`` reports both modes, and the caller
    decides. The heads are still fit only on the training rows either way. ``C`` is picked by
    OOF macro-F1, thresholds are tuned on the OOF probabilities, and the metrics are
    computed on them. Selection and metrics use the decision rule serving applies
    for ``task_type`` — for binary/multiclass (argmax) threshold tuning is skipped
    entirely, since serving never reads thresholds there. Returns ``(best_c,
    global_threshold, per_label_thresholds, metrics)`` (the caller then fits the
    deploy model on 100% of the data).

    ``select_on_tuned_thresholds`` changes WHICH candidate wins: instead of ranking the
    candidates at a flat 0.5 cut and tuning only the winner, each is scored under
    thresholds tuned for itself and the (C, thresholds) pair is chosen together. No
    extra fits — every candidate's out-of-fold probabilities are already in memory. See
    ``Profile.select_c_on_tuned_thresholds`` for why it is off by default.

    ``on_step(done, total, detail)`` fires after EVERY head fit (k x |grid| of
    them), so the caller can show real progress across a run that takes minutes.

    ``tol`` applies to every fold's fit and to nothing else; the deploy fit the caller
    makes afterwards keeps scikit-learn's default. See ``Profile.selection_tol`` for
    what a looser one was measured to be worth.

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
    if matrix is not None and matrix.shape[0] != n:
        # Fold indices slice the matrix and `y` together, so a matrix built from a
        # different row set would line row i of one up against row j of the other and
        # still produce a plausible number. Refuse rather than mis-measure.
        raise ValueError(
            f"matrix has {matrix.shape[0]} rows but the labels have {n}; they must be "
            "the same rows in the same order."
        )
    folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))
    # OOF probabilities per C, each row filled exactly once (float32 bounds RAM).
    oof = {c: np.zeros((n, n_labels), dtype=np.float32) for c in c_grid}
    total_fits = k * len(c_grid)
    done = 0
    for fold, (tr, te) in enumerate(folds, start=1):
        if should_stop is not None and should_stop():
            return None
        if matrix is None:
            vec = make_vectorizer()
            x_tr = vec.fit_transform([texts[i] for i in tr])
            x_te = vec.transform([texts[i] for i in te])
        else:
            x_tr, x_te = matrix[tr], matrix[te]
        for c in c_grid:
            # Checked per C candidate (not only per fold) so /train/stop keeps its
            # "between the C fits" promise in CV mode as well.
            if should_stop is not None and should_stop():
                return None
            head = make_head(c, n_jobs=n_jobs, solver=solver, tol=tol)
            head.fit(x_tr, y[tr])
            oof[c][te] = head.predict_proba(x_te).astype(np.float32)
            done += 1
            if on_step is not None:
                on_step(done, total_fits, f"Fold {fold}/{k}: C={c} ({done}/{total_fits} fits)")
    thresholds_apply = tune_threshold and not is_single_label(task_type)
    if select_on_tuned_thresholds and thresholds_apply:
        best_c, global_t, columns = _best_under_own_thresholds(
            y, oof, c_grid, per_label=per_label
        )
        # The winner's thresholds ARE the ones it was selected on; re-deriving them
        # would repeat the same search for the same answer.
        per_label_t = name_threshold_columns(columns, classes) if per_label else {}
    else:
        best_c = max(c_grid, key=lambda c: macro_f1(y, _default_decision(oof[c], task_type)))
        if thresholds_apply:
            global_t, per_label_t = tune_thresholds(y, oof[best_c], classes, per_label=per_label)
        else:
            global_t, per_label_t = 0.5, {}
    metrics = compute_metrics(y, oof[best_c], classes, global_t, per_label_t, task_type=task_type)
    return best_c, global_t, per_label_t, metrics


def _best_under_own_thresholds(
    y_true: np.ndarray,
    proba_by_c: dict[float, np.ndarray],
    c_grid: list[float],
    *,
    per_label: bool,
) -> tuple[float, float, np.ndarray]:
    """The (C, global threshold, threshold columns) triple with the best macro F1,
    every candidate judged under thresholds tuned for itself.

    Ranking at a flat 0.5 cut and tuning only afterwards eliminates a candidate whose
    probabilities are ranked well but scaled low, before its thresholds ever exist.
    """
    best_c, best_f1 = c_grid[0], -1.0
    best_global, best_columns = 0.5, np.full(y_true.shape[1], 0.5)
    for c in c_grid:
        score, global_t, columns = _tuned_score(y_true, proba_by_c[c], per_label=per_label)
        if score > best_f1:
            best_f1, best_c, best_global, best_columns = score, c, global_t, columns
    return best_c, best_global, best_columns


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


def _tuned_score(
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


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    global_threshold: float,
    per_label: dict[str, float],
    task_type: str = "multilabel",
) -> dict:
    """Macro/micro P/R/F1 plus per-label F1, computed on the given split.

    Measured with the SAME decision rule serving applies: argmax for
    binary/multiclass (``ClassifierModel.predict`` ignores thresholds there),
    the tuned thresholds for multilabel. ``decision_rule`` records which rule
    produced the numbers, so bundles stay self-describing.

    ``predicted_labels_per_row`` vs ``true_labels_per_row`` expose over-assertion,
    which F1 alone hides: a wide label space pushes the F1-optimal per-label cut
    down, and the model starts asserting far more labels than the data carries.
    How strongly depends on the TARGET (a subject vocab behaves differently from a
    curriculum vocab on the same rows), so every bundle carries its own pair.
    """
    single = is_single_label(task_type)
    preds = (
        _argmax_onehot(proba) if single
        else apply_thresholds(proba, classes, global_threshold, per_label)
    )
    per_label_f1 = f1_score(y_true, preds, average=None, zero_division=0)
    return {
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "n_labels": len(classes),
        "decision_rule": "argmax" if single else "thresholds",
        "predicted_labels_per_row": round(float(preds.sum(axis=1).mean()), 3),
        "true_labels_per_row": round(float(y_true.sum(axis=1).mean()), 3),
        "per_label_f1": {uri: float(score) for uri, score in zip(classes, per_label_f1, strict=False)},
    }
