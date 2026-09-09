"""Tests for where the decision cuts sit and what they do to probabilities."""

import numpy as np
import pytest

from app import thresholds
from app.tuning import compute_metrics


def test_tune_thresholds_finds_separating_value():
    n = 200
    y = np.zeros((n, 2), dtype=int)
    y[:100, 0] = 1
    y[100:, 1] = 1
    proba = np.empty((n, 2))
    proba[:100, 0], proba[100:, 0] = 0.8, 0.2
    proba[:100, 1], proba[100:, 1] = 0.2, 0.8

    global_t, per_label = thresholds.tune_thresholds(y, proba, ["c0", "c1"], per_label=True)
    assert 0.2 < global_t < 0.8
    assert set(per_label) == {"c0", "c1"}

    metrics = compute_metrics(y, proba, ["c0", "c1"], global_t, per_label)
    assert metrics["f1_macro"] > 0.99
    assert set(metrics["per_label_f1"]) == {"c0", "c1"}

def test_tune_thresholds_zero_positive_label_keeps_global(audit_grid=None):
    """A label with NO positives in the val split must fall back to the global
    threshold. Regression: `best_f1 = -1.0` let the first grid value (0.05) win
    because f1=0 > -1, serving rare labels with a near-zero threshold."""
    rng = np.random.default_rng(0)
    n = 100
    y = np.zeros((n, 2), dtype=int)
    y[:50, 0] = 1  # label 0: separable; label 1: zero positives in val
    proba = rng.uniform(0.0, 1.0, size=(n, 2))
    proba[:50, 0] = rng.uniform(0.7, 1.0, size=50)
    proba[50:, 0] = rng.uniform(0.0, 0.3, size=50)

    global_t, per_label = thresholds.tune_thresholds(y, proba, ["a", "b"], per_label=True)

    assert per_label["b"] == pytest.approx(global_t), (
        "zero-positive label must keep the global threshold, not the grid minimum"
    )

def test_tune_threshold_columns_is_what_the_uri_keyed_form_is_built_from():
    """Thresholds are a per-COLUMN quantity; the uri -> threshold dict is how the
    bundle stores them. Splitting the two lets a caller score a candidate by its own
    thresholds without carrying the label names into the C search, which is what
    plan item C1 needs. The dict must stay exactly the zip of the columns."""
    # Each label needs a DIFFERENT optimal cut, and at least one different from the
    # global one — on data where every column agrees with the global, a wrapper that
    # ignored the columns entirely would pass and prove nothing (it did, once).
    # Positives/negatives per column: 0.90/0.10, 0.50/0.45, 0.80/0.75.
    n = 60
    y = np.zeros((n, 3), dtype=int)
    proba = np.empty((n, 3))
    for col, (high, low) in enumerate([(0.90, 0.10), (0.50, 0.45), (0.80, 0.75)]):
        block = slice(col * 20, (col + 1) * 20)
        y[block, col] = 1
        proba[:, col] = low
        proba[block, col] = high

    classes = ["c0", "c1", "c2"]
    global_t, columns = thresholds.tune_threshold_columns(y, proba, per_label=True)
    wrapped_global, wrapped = thresholds.tune_thresholds(y, proba, classes, per_label=True)

    assert columns.shape == (3,)
    assert wrapped_global == global_t
    assert wrapped == {uri: float(t) for uri, t in zip(classes, columns, strict=True)}
    assert not np.all(columns == global_t), (
        "this fixture must produce per-label cuts that differ from the global one, "
        "otherwise the assertion above cannot tell a real zip from a constant"
    )

def test_tune_threshold_columns_without_per_label_repeats_the_global():
    """`per_label=False` means every column decides at the global cut. The dict form
    says that by staying empty (apply_thresholds falls back); the column form has to
    say it by carrying the value, because an array has no 'absent'."""
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    global_t, columns = thresholds.tune_threshold_columns(y, proba, per_label=False)
    assert np.all(columns == global_t)
    assert thresholds.tune_thresholds(y, proba, ["a", "b"], per_label=False) == (global_t, {})
