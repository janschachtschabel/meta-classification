"""`at_least_one`: when no label of a multilabel model passes its threshold, the best one anyway.

The thresholds keep a multilabel model from asserting what it is unsure of, and an empty
answer is the honest result of that. A caller who knows the text belongs SOMEWHERE in the
label set can ask for the nearest label instead -- without `top_k=1`, which cuts a real second
label from every answer that had two. The added label says what it is: `above_threshold:
false`, as a forced entry of a ranking does.
"""

import numpy as np

from app.classifier import ClassifierModel


class _Vectorizer:
    kind = "tfidf"

    def transform(self, texts):
        return np.zeros((len(list(texts)), 1), dtype=np.float32)


class _Head:
    """Fixed probabilities per call: one row of them, repeated for every text."""

    def __init__(self, row):
        self.row = row

    def predict_proba(self, x):
        return np.tile(self.row, (x.shape[0], 1))


def _model(row, task_type="multilabel") -> ClassifierModel:
    # No stored thresholds: every label is cut at the default 0.5.
    return ClassifierModel(
        vectorizer=_Vectorizer(),  # type: ignore[arg-type]
        head=_Head(row),  # type: ignore[arg-type]
        classes=["u:math", "u:phys", "u:chem"],
        task_type=task_type,
        avg_labels=1.0,
        uri_to_label={"u:math": "Mathe", "u:phys": "Physik", "u:chem": "Chemie"},
    )


def test_an_empty_answer_gets_the_best_label_marked_as_below_its_threshold():
    model = _model([0.30, 0.45, 0.10])

    plain = model.predict(["text"])[0]
    forced = model.predict(["text"], at_least_one=True)[0]

    assert plain == []
    assert [(p.uri, p.above_threshold) for p in forced] == [("u:phys", False)]
    assert forced[0].confidence == np.float32(0.45)


def test_an_answer_that_has_labels_keeps_all_of_them_unmarked():
    # The point of the option over top_k=1: a second real label stays.
    model = _model([0.90, 0.80, 0.10])

    answer = model.predict(["text"], at_least_one=True)[0]

    assert [(p.uri, p.above_threshold) for p in answer] == [("u:math", None), ("u:phys", None)]


def test_the_option_follows_a_label_filter_and_a_threshold_override():
    model = _model([0.30, 0.45, 0.20])

    filtered = model.predict(["text"], at_least_one=True, label_filter="chem")[0]
    lowered = model.predict(["text"], at_least_one=True, threshold=0.4)[0]

    assert [p.uri for p in filtered] == ["u:chem"], "the best AMONG the requested labels"
    assert [(p.uri, p.above_threshold) for p in lowered] == [("u:phys", None)], \
        "a label above the request's own cut is a normal answer"


def test_ranking_and_single_label_models_are_unaffected():
    multilabel = _model([0.30, 0.45, 0.10])
    multiclass = _model([0.30, 0.45, 0.25], task_type="multiclass")

    ranked = multilabel.predict(["text"], top_k=2, at_least_one=True)[0]
    argmax = multiclass.predict(["text"], at_least_one=True)[0]

    assert [(p.uri, p.above_threshold) for p in ranked] == [("u:phys", False), ("u:math", False)]
    assert [(p.uri, p.above_threshold) for p in argmax] == [("u:phys", None)]


def test_a_label_filter_that_matches_nothing_stays_empty():
    assert _model([0.30, 0.45, 0.10]).predict(["text"], at_least_one=True, label_filter="bio") == [[]]


def test_the_model_says_where_the_option_can_act():
    # applied_settings reports what was APPLIED: never for a ranking, never for argmax.
    multilabel = _model([0.30, 0.45, 0.10])
    multiclass = _model([0.30, 0.45, 0.25], task_type="multiclass")

    assert multilabel.applies_at_least_one(None) is True
    assert multilabel.applies_at_least_one(2) is False
    assert multiclass.applies_at_least_one(None) is False


def test_off_by_default():
    assert _model([0.30, 0.45, 0.10]).predict(["a", "b"]) == [[], []]
