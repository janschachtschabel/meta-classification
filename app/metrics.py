"""What a reported score means: probabilities read with the rule serving applies.

Split out of ``tuning``, which chooses C: the two change for different reasons — the C
search when the question is which model to fit, this when the question is what a number
in a bundle means. Training and an evaluation against another dataset (``evaluate``)
both score through here, so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .thresholds import apply_thresholds


def is_single_label(task_type: str) -> bool:
    """binary/multiclass = exactly one label per prediction (serving uses argmax)."""
    return task_type in ("binary", "multiclass")


def argmax_onehot(proba: np.ndarray) -> np.ndarray:
    """One-hot argmax decision — the rule serving applies to binary/multiclass
    (``ClassifierModel.predict`` returns the single best label there and
    ignores thresholds entirely)."""
    preds = np.zeros_like(proba, dtype=np.int8)  # see apply_thresholds on the dtype
    preds[np.arange(proba.shape[0]), proba.argmax(axis=1)] = 1
    return preds


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    global_threshold: float,
    per_label: dict[str, float],
    task_type: str = "multilabel",
    *,
    scored: np.ndarray | None = None,
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

    ``scored`` (bool per class) limits the macro averages and ``per_label_f1`` to the
    classes a REAL row can validate: a class whose rows an LLM wrote has nothing to be
    scored on here, and its F1 would read 0 whatever the model does. Micro F1 keeps every
    class — asserting such a label for a real row is still a wrong answer. ``None``
    scores every class.
    """
    single = is_single_label(task_type)
    preds = (
        argmax_onehot(proba) if single
        else apply_thresholds(proba, classes, global_threshold, per_label)
    )
    columns = None if scored is None else np.flatnonzero(scored)
    names = classes if columns is None else [classes[i] for i in columns]
    per_label_f1 = f1_score(y_true, preds, average=None, zero_division=0, labels=columns)

    def macro(score) -> float:
        return float(score(y_true, preds, average="macro", zero_division=0, labels=columns))

    return {
        "f1_macro": macro(f1_score),
        "f1_micro": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "precision_macro": macro(precision_score),
        "recall_macro": macro(recall_score),
        "n_labels": len(classes),
        "decision_rule": "argmax" if single else "thresholds",
        "predicted_labels_per_row": round(float(preds.sum(axis=1).mean()), 3),
        "true_labels_per_row": round(float(y_true.sum(axis=1).mean()), 3),
        "per_label_f1": {uri: float(score) for uri, score in zip(names, per_label_f1, strict=False)},
    }
