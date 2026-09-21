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


def test_an_all_true_provenance_mask_narrows_nothing_of_its_own():
    """`scored=None` and an all-True mask are the same request — score every label the
    row provenance allows.

    What the SPLIT can score is a second, unconditional narrowing, and "c" has no positive
    row here either way. This test used to assert the opposite (macro 2/3, `c` scored 0.0);
    that was the defect stated as a contract, not a property worth keeping.
    """
    everything = compute_metrics(Y, PROBA, CLASSES, 0.5, {}, scored=np.ones(3, dtype=bool))

    assert everything == compute_metrics(Y, PROBA, CLASSES, 0.5, {})
    assert everything["labels_not_scored"] == ["c"]


def test_decision_matrices_use_a_compact_dtype():
    """A 0/1 decision matrix is the same shape as the probabilities it comes from, and
    int64 spends 8 bytes per bit.

    ``data.prepare_targets`` already refuses that for the target matrix, with the
    arithmetic in its docstring: 1.34 GB instead of 168 MB at 600k rows x 300 labels. The
    decision matrices are the SAME shape and were left at int64 — and at 250k x 300 one of
    them is 600 MB, allocated and freed once per threshold candidate, in the middle of the
    phase the training memory budget exists to protect. scikit-learn scores an int8
    indicator matrix identically.
    """
    import numpy as np

    from app import thresholds, tuning
    from app.metrics import argmax_onehot

    proba = np.array([[0.9, 0.2, 0.7], [0.1, 0.8, 0.3]], dtype=np.float32)
    classes = ["a", "b", "c"]

    applied = thresholds.apply_thresholds(proba, classes, 0.5, {})
    assert applied.dtype == np.int8
    assert applied.tolist() == [[1, 0, 1], [0, 1, 0]], "the decision itself is unchanged"

    assert argmax_onehot(proba).dtype == np.int8
    assert tuning._default_decision(proba, "multilabel").dtype == np.int8


# Three labels, one of which ("c") no row of the split carries — not because an LLM wrote
# its rows, but because it is rare and the split put its positives in train. A dataset with
# no provenance marks at all looks exactly like this.
Y_SPLIT = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
PERFECT = np.array([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]])


def test_a_label_the_split_cannot_score_is_left_out_of_the_macro():
    """A perfect prediction must read 1.0, whatever the split happens to contain.

    Without the narrowing, "c" scores F1 0.0 — there is no threshold and no model that
    could make it score anything else — and drags a flawless run to 0.667. That number is
    a fact about the split, not about the model, and it is the headline the bundle carries.
    """
    metrics = compute_metrics(Y_SPLIT, PERFECT, CLASSES, 0.5, {})

    assert metrics["f1_macro"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["per_label_f1"] == {"a": 1.0, "b": 1.0}
    assert metrics["n_labels"] == 3, "the label space is still three wide"


def test_the_labels_left_out_of_the_macro_are_named():
    """Silently averaging over a subset would be the same defect with a nicer number: the
    bundle has to say which labels the split could not put a question to."""
    metrics = compute_metrics(Y_SPLIT, PERFECT, CLASSES, 0.5, {})

    assert metrics["labels_not_scored"] == ["c"]
    assert compute_metrics(Y_SPLIT[:2], PERFECT[:2], CLASSES, 0.5, {})[
        "labels_not_scored"] == ["b", "c"]


def test_marking_the_rows_does_not_change_what_the_same_split_scores():
    """The same rows, read once as a plain dataset and once as a marked one.

    `real_rows.row_provenance` narrows to the labels a real row carries, and the holdout
    scores real rows only — so on these rows the two masks are the same mask, and the two
    readings have to agree. They did not: a dataset gained macro F1 purely by carrying
    data-prep's mark columns, which is exactly the non-comparability `app/real_rows.py`
    says the narrowing exists to prevent.
    """
    unmarked = compute_metrics(Y_SPLIT, PERFECT, CLASSES, 0.5, {})
    marked = compute_metrics(Y_SPLIT, PERFECT, CLASSES, 0.5, {},
                             scored=np.array([True, True, False]))

    assert unmarked == marked


def test_a_split_that_carries_no_label_at_all_scores_zero_rather_than_nan():
    """Narrowing to nothing would hand `average="macro"` an empty label list, and sklearn
    answers that with nan — which `json.dumps` writes as the token `NaN`, i.e. a
    metrics.json no strict parser will read back. Score everything instead: nothing could
    be measured here, and 0.0 says so in a number."""
    empty = np.zeros((2, 3), dtype=np.int8)

    metrics = compute_metrics(empty, PERFECT[:2], CLASSES, 0.5, {})

    assert metrics["f1_macro"] == 0.0
    assert metrics["labels_not_scored"] == []
    assert set(metrics["per_label_f1"]) == set(CLASSES)
