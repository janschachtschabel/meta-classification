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


def _uneven_label_support():
    """Four labels, deterministic cuts, built so shrinkage has something to say.

    Labels 0 and 1 both have an optimal cut of 0.65 but wildly different support — 60
    positives against 4 — which is the whole question shrinkage answers. Labels 2 and 3
    are populous and optimal at 0.35, which is what holds the GLOBAL cut at 0.35, away
    from the other two; without that the per-label answers would equal the global and
    shrinking toward it would be a no-op that proves nothing. 0.35 rather than 0.50 on
    purpose: with the global sitting at 0.50, an implementation that shrank toward a
    hardcoded 0.5 passed every assertion here (it did).
    """
    n = 120
    y = np.zeros((n, 4), dtype=int)
    proba = np.empty((n, 4))
    # (positive score, negative score, number of positives)
    for col, (high, low, n_pos) in enumerate(
        [(0.9, 0.6, 60), (0.9, 0.6, 4), (0.4, 0.3, 60), (0.4, 0.3, 60)]
    ):
        proba[:, col] = low
        y[:n_pos, col] = 1
        proba[:n_pos, col] = high
    return y, proba, np.array([60, 4, 60, 60])


def test_shrinkage_trusts_a_threshold_in_proportion_to_the_positives_behind_it():
    """A cut fitted on 4 positives is mostly noise; one fitted on 60 is not, and the
    F1-argmax rule cannot tell the two apart on its own.

    Labels 0 and 1 land on the SAME unshrunk cut, so the only thing separating them
    afterwards is how much support each had. The formula is pinned against the unshrunk
    answer, so a wrong weight, a wrong direction, or a shrinkage applied to the global
    instead of to the columns all fail.
    """
    y, proba, counts = _uneven_label_support()
    k = 10.0

    global_t, plain = thresholds.tune_threshold_columns(y, proba, per_label=True)
    shrunk_global, shrunk = thresholds.tune_threshold_columns(
        y, proba, per_label=True, shrink_k=k)

    assert shrunk_global == global_t, "shrinkage moves per-label cuts, never the global"
    assert plain[0] == plain[1] != global_t, (
        "the two labels must start from one cut that differs from the global, or the "
        "comparison below is not about support at all"
    )

    weight = counts / (counts + k)
    assert shrunk == pytest.approx(weight * plain + (1 - weight) * global_t)
    assert abs(shrunk[1] - global_t) < abs(shrunk[0] - global_t), (
        f"the label with 4 positives should end up nearer the global than the one with "
        f"60; got {shrunk[1]:.4f} and {shrunk[0]:.4f} against a global {global_t}"
    )


def test_shrinkage_off_by_default_leaves_every_threshold_where_it_was():
    y, proba, _ = _uneven_label_support()
    global_t, plain = thresholds.tune_threshold_columns(y, proba, per_label=True)
    assert thresholds.tune_threshold_columns(y, proba, per_label=True, shrink_k=None) == (
        global_t, pytest.approx(plain))


def test_shrinkage_leaves_a_label_with_no_positives_at_the_global_threshold():
    """The zero-positive case already fell back to the global; under shrinkage it is
    the limit of the same rule (weight 0), not a second special case."""
    y = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    proba = np.array([[0.9, 0.3], [0.8, 0.4], [0.2, 0.7], [0.1, 0.6]])
    global_t, columns = thresholds.tune_threshold_columns(y, proba, per_label=True, shrink_k=10.0)
    assert columns[1] == pytest.approx(global_t)
