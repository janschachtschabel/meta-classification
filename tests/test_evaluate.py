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
    beside f1_micro 0.941. The 0.068 was never a fact about the model — 55 of its classes
    had no example in the probe, and such a class scores 0.0 under every threshold and
    every prediction. The macro now covers the classes the data actually asks about, and
    `labels_covered` beside `labels_not_scored` says how narrow the question was.

    Until CORR-2 this test asserted `f1_macro < f1_micro` — that the uncovered labels drag
    the macro down. That drag was the defect, so the assertion went with it.
    """
    model = _StubModel(["a", "b", "c"], [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]])

    result = evaluate.evaluate_model(model, ["x", "y"], [["a"], ["b"]])

    assert result["labels_covered"] == 2
    assert result["metrics"]["n_labels"] == 3, "the label space is still three wide"
    assert result["metrics"]["labels_not_scored"] == ["c"]
    assert result["metrics"]["f1_macro"] == result["metrics"]["f1_micro"] == 1.0


def test_the_decision_rule_is_the_one_serving_applies():
    """A multiclass model answers by argmax and reads no threshold. Measuring it with
    thresholds would score a rule the model does not use."""
    single = _StubModel(["a", "b"], [[0.4, 0.3]], task_type="multiclass")

    result = evaluate.evaluate_model(single, ["x"], [["a"]])

    assert result["metrics"]["decision_rule"] == "argmax"
    assert result["metrics"]["f1_macro"] > 0, "argmax picks 'a' even below 0.5"


class _RecordingModel(_StubModel):
    """Remembers which texts it was asked to score."""

    def __init__(self, classes):
        super().__init__(classes, [[0.9, 0.1]] * 10)
        self.scored: list[str] = []

    def predict_proba(self, texts):
        self.scored.extend(texts)
        return super().predict_proba(texts)


class _Registry:
    def __init__(self, model, metadata=None):
        self.model, self.records = model, []
        self.metadata = metadata or {}

    def get(self, name):
        return self.model

    def info(self, name):
        """What the bundle records about the run that produced the model."""
        return {"name": name, "metadata": self.metadata}

    def append_evaluation(self, name, record):
        self.records.append(record)


def _evaluate_csv(tmp_path, lines, *, text_columns=("title",), metadata=None, **request):
    from app.settings import Settings

    (tmp_path / "eval.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    model = _RecordingModel(["a", "b"])
    registry = _Registry(model, metadata)
    evaluate.run_evaluation(
        {"model_name": "m", "dataset_name": "eval.csv", "text_columns": list(text_columns),
         "label_column": "labels", **request},
        Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False),
        registry, on_progress=lambda **_: None, should_stop=lambda: False,
    )
    return model, registry


def test_rows_an_llm_wrote_or_touched_are_skipped_and_counted(tmp_path):
    """An evaluation is a validation: a model measured on text an LLM wrote measures
    how well it learned that LLM."""
    model, registry = _evaluate_csv(tmp_path, [
        "title;labels;generated_for;example_for;enriched_fields",
        "Bruchrechnung Aufgabe;a;;;",
        "Erzeugter Text eins;a;a;;",
        "Photosynthese Versuch;b;;;",
        "Beispielzeile zwei;b;;b;",
        "Ergänzte Zeile drei;a;;;keywords",
    ])

    assert model.scored == ["Bruchrechnung Aufgabe", "Photosynthese Versuch"]
    assert registry.records[0]["ai_marked_rows_skipped"] == 3
    assert registry.records[0]["n_rows"] == 2


def test_a_dataset_of_ai_rows_only_cannot_be_evaluated_and_says_why(tmp_path):
    import pytest

    from app.errors import TrainingInputError

    with pytest.raises(TrainingInputError, match="AI-marked"):
        _evaluate_csv(tmp_path, ["title;labels;generated_for", "Erzeugter Text eins;a;a"])


def test_the_text_is_assembled_the_way_the_model_was_trained(tmp_path):
    """A model fit on title-twice-plus-keywords, scored on title-once, is measured on a
    distribution it was never tuned on -- and the UI sends no weights at all."""
    model, registry = _evaluate_csv(
        tmp_path,
        ["title;keywords;labels", "Bruchrechnung;Mathe Zahlen;a"],
        text_columns=("title", "keywords"),
        metadata={"text_columns": ["title", "keywords"], "text_column_weights": {"title": 2}},
    )

    assert model.scored == ["Bruchrechnung Bruchrechnung Mathe Zahlen"]
    assert registry.records[0]["text_column_weights"] == {"title": 2}


def test_a_weight_for_a_column_this_run_does_not_read_is_left_out(tmp_path):
    """The bundle may name more columns than the request reads; the weights narrow to
    the columns actually assembled, as a training request narrows them."""
    _, registry = _evaluate_csv(
        tmp_path,
        ["title;labels", "Bruchrechnung;a"],
        metadata={"text_column_weights": {"title": 2, "description": 3}},
    )

    assert registry.records[0]["text_column_weights"] == {"title": 2}


def test_asking_for_no_weighting_is_still_possible(tmp_path):
    """An explicit empty mapping means what it says: every column once."""
    model, registry = _evaluate_csv(
        tmp_path,
        ["title;labels", "Bruchrechnung;a"],
        metadata={"text_column_weights": {"title": 2}},
        text_column_weights={},
    )

    assert model.scored == ["Bruchrechnung"]
    assert registry.records[0]["text_column_weights"] == {}
