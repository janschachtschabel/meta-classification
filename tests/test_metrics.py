"""What a reported score means when some labels have no real row to be scored on."""

import numpy as np
import pytest

from app.metrics import compute_metrics

CLASSES = ["a", "b", "c"]
# Four real rows; none of them carries "c" — its rows were all written by an LLM.
Y = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
# Right on a and b, plus one false alarm for c.
PROBA = np.array([[0.9, 0.1, 0.1], [0.9, 0.1, 0.8], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]])


def test_a_label_no_real_row_carries_has_no_score_and_no_weight_in_the_macro():
    """Its F1 would read 0 whatever the model does — a number about the data, not the
    model — and it would drag the macro average down by a third here."""
    metrics = compute_metrics(Y, PROBA, CLASSES, 0.5, {}, scored=np.array([True, True, False]))

    assert metrics["per_label_f1"] == {"a": 1.0, "b": 1.0}
    assert metrics["f1_macro"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["n_labels"] == 3


def test_a_false_alarm_for_an_unscored_label_still_costs_the_micro_score():
    """Asserting "c" for a real row that is not "c" is a wrong answer a user would see."""
    metrics = compute_metrics(Y, PROBA, CLASSES, 0.5, {}, scored=np.array([True, True, False]))

    assert metrics["f1_micro"] == pytest.approx(8 / 9)   # 4 hits, 1 false alarm


def test_scoring_every_label_is_the_unrestricted_computation():
    everything = compute_metrics(Y, PROBA, CLASSES, 0.5, {}, scored=np.ones(3, dtype=bool))

    assert everything == compute_metrics(Y, PROBA, CLASSES, 0.5, {})
    assert everything["f1_macro"] == pytest.approx(2 / 3)
    assert everything["per_label_f1"]["c"] == 0.0
