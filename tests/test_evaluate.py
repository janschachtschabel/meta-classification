"""Scoring an existing model against a dataset.

"Model B beats model A" is only a statement if both were measured on the same rows.
The numbers are easy; what makes them honest is what the model's label space cannot
cover, and that is what most of these tests are about.

The classifier is a stub: what is under test is the alignment of truth to the model's
label space, not scikit-learn's arithmetic.
"""

import numpy as np

from app import evaluate


class _StubModel:
    """Answers with fixed probabilities, in a fixed label space."""

    def __init__(self, classes, rows, task_type="multilabel"):
        self.classes = classes
        self.task_type = task_type
        self.global_threshold = 0.5
        self.per_label_thresholds = {}
        self.uri_to_label = {uri: uri.upper() for uri in classes}
        self._rows = np.array(rows, dtype=float)

    def predict_proba(self, texts):
        return self._rows[: len(texts)]


def test_a_perfect_answer_scores_one():
    """The floor the rest is read against."""
    model = _StubModel(["a", "b"], [[0.9, 0.1], [0.1, 0.9]])

    result = evaluate.evaluate_model(model, ["x", "y"], [["a"], ["b"]])

    assert result["metrics"]["f1_macro"] == 1.0
    assert result["n_rows"] == 2
    assert result["rows_without_a_known_label"] == 0
    assert result["unknown_labels"] == []


def test_truth_is_aligned_to_the_model_label_space_not_the_dataset():
    """The dataset's label set is not the model's. Binarizing the data on its own terms
    would produce a matrix of a different width than the model's output — and silently
    scoring column 0 of one against column 0 of the other is how an evaluation ends up
    measuring nothing at all."""
    model = _StubModel(["a", "b"], [[0.9, 0.1]])

    result = evaluate.evaluate_model(model, ["x"], [["a", "zzz_unknown_to_the_model"]])

    assert result["metrics"]["n_labels"] == 2, "scored over the model's classes"
    assert result["metrics"]["f1_macro"] > 0, "the label it does know still counts"


def test_a_label_the_model_never_learned_is_reported_not_hidden():
    """It caps the achievable recall: those rows can never be answered correctly, and a
    score that does not mention them reads better than the model deserves."""
    model = _StubModel(["a"], [[0.9], [0.9]])

    result = evaluate.evaluate_model(model, ["x", "y"], [["a"], ["a", "physics"]])

    assert result["unknown_labels"] == ["physics"]
    assert result["n_rows"] == 2


def test_a_row_with_no_label_the_model_knows_is_excluded_and_counted():
    """Scoring it as a failure would blame the model for a label it was never given;
    dropping it quietly would flatter the model on a dataset it does not cover. Neither
    is a number to compare two models on — so the row is excluded AND reported."""
    model = _StubModel(["a"], [[0.9], [0.9]])

    result = evaluate.evaluate_model(model, ["x", "y"], [["a"], ["physics"]])

    assert result["n_rows"] == 1, "only the row the model could answer"
    assert result["rows_without_a_known_label"] == 1
    assert result["unknown_labels"] == ["physics"]


def test_nothing_to_score_is_said_plainly():
    """A dataset from a different vocabulary produces no comparable rows at all. An F1
    of 0.0 would read as "the model is terrible here" rather than "these two do not
    meet"."""
    model = _StubModel(["a"], [[0.9]])

    result = evaluate.evaluate_model(model, ["x"], [["physics"]])

    assert result["n_rows"] == 0
    assert result["metrics"] is None
    assert result["rows_without_a_known_label"] == 1


def test_how_much_of_the_label_space_the_data_covers_is_reported():
    """Found on a real 59-label model scored against an 8-row probe: f1_macro 0.068
    beside f1_micro 0.941. Both are correct — macro averages over ALL the model's
    classes, and 55 of them had no examples, so they score zero. Without the coverage
    beside it that headline reads as "this model is terrible" instead of "this dataset
    exercises four of its labels".
    """
    model = _StubModel(["a", "b", "c"], [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]])

    result = evaluate.evaluate_model(model, ["x", "y"], [["a"], ["b"]])

    assert result["labels_covered"] == 2
    assert result["metrics"]["n_labels"] == 3, "still scored over the full label space"
    assert result["metrics"]["f1_macro"] < result["metrics"]["f1_micro"], (
        "the uncovered label drags macro down — which is exactly what needs explaining"
    )


def test_the_decision_rule_is_the_one_serving_applies():
    """A multiclass model answers by argmax and reads no threshold. Measuring it with
    thresholds would score a rule the model does not use."""
    single = _StubModel(["a", "b"], [[0.4, 0.3]], task_type="multiclass")

    result = evaluate.evaluate_model(single, ["x"], [["a"]])

    assert result["metrics"]["decision_rule"] == "argmax"
    assert result["metrics"]["f1_macro"] > 0, "argmax picks 'a' even below 0.5"
