"""Tests for the runtime ClassifierModel (prediction path)."""

import numpy as np

from app import data
from app.classifier import ClassifierModel


class _SpyVectorizer:
    """Records the texts passed to transform; returns a dummy feature matrix."""

    kind = "tfidf"

    def __init__(self) -> None:
        self.seen: list[str] | None = None

    def transform(self, texts):
        self.seen = list(texts)
        return np.zeros((len(self.seen), 1), dtype=np.float32)


class _ConstHead:
    """Minimal head: constant per-label probabilities."""

    def predict_proba(self, x):
        return np.tile([0.9, 0.1], (x.shape[0], 1))


def _model(vectorizer: _SpyVectorizer) -> ClassifierModel:
    return ClassifierModel(
        vectorizer=vectorizer,  # type: ignore[arg-type]
        head=_ConstHead(),  # type: ignore[arg-type]
        classes=["u1", "u2"],
        task_type="multilabel",
        avg_labels=1.0,
        uri_to_label={"u1": "A", "u2": "B"},
    )


class _TextDependentVectorizer:
    """1 feature: 1.0 for non-empty text, 0.0 for the empty baseline text."""

    kind = "tfidf"

    def transform(self, texts):
        return np.array([[1.0 if t else 0.0] for t in texts], dtype=np.float32)


class _LinearHead:
    """p(u1) = 0.2 + 0.6*x -> empty text 0.2 (base rate), real text 0.8."""

    def predict_proba(self, x):
        p1 = 0.2 + 0.6 * x[:, 0]
        return np.stack([p1, 1.0 - p1], axis=1)


def test_baseline_diff_separates_text_signal_from_base_rate():
    """baseline_diff = confidence minus the empty-text prediction: it isolates
    what the TEXT contributes from the label's base rate (colleague-repo idea)."""
    import pytest

    model = ClassifierModel(
        vectorizer=_TextDependentVectorizer(),  # type: ignore[arg-type]
        head=_LinearHead(),  # type: ignore[arg-type]
        classes=["u1", "u2"],
        task_type="multilabel",
        avg_labels=1.0,
        uri_to_label={},
    )
    preds = model.predict(["echter text"], threshold=0.0, include_baseline_diff=True)[0]
    by_uri = {p.uri: p for p in preds}
    assert by_uri["u1"].confidence == pytest.approx(0.8)
    assert by_uri["u1"].baseline_diff == pytest.approx(0.6)   # text pushes u1 up
    assert by_uri["u2"].baseline_diff == pytest.approx(-0.6)  # ... and u2 down

    plain = model.predict(["echter text"], threshold=0.0)[0]
    assert all(p.baseline_diff is None for p in plain)  # opt-in only


def test_label_f1_exposes_per_label_reliability():
    """include_label_f1 attaches the label's training F1, so a caller can tell a
    confident-AND-reliable label from a confident-but-unreliable one. A label the
    metrics never scored stays None rather than being guessed. Opt-in only."""
    model = ClassifierModel(
        vectorizer=_SpyVectorizer(),  # type: ignore[arg-type]
        head=_ConstHead(),  # type: ignore[arg-type]
        classes=["u1", "u2"],
        task_type="multilabel",
        avg_labels=1.0,
        uri_to_label={"u1": "A", "u2": "B"},
        per_label_f1={"u1": 0.95},
    )
    preds = model.predict(["text"], threshold=0.0, include_label_f1=True)[0]
    by_uri = {p.uri: p for p in preds}
    assert by_uri["u1"].label_f1 == 0.95
    assert by_uri["u2"].label_f1 is None  # not in metrics -> honestly absent

    plain = model.predict(["text"], threshold=0.0)[0]
    assert all(p.label_f1 is None for p in plain)  # opt-in only


def test_explicit_top_k_is_ranking_not_cap():
    """An explicit top_k means 'the K most probable labels' (ranking), NOT a cap
    on the thresholded list — otherwise the field looks dead whenever only one
    label passes its threshold. Each ranked prediction carries above_threshold."""
    model = _model(_SpyVectorizer())  # ConstHead: p(u1)=0.9, p(u2)=0.1; thresholds 0.5
    ranked = model.predict(["text"], top_k=2)[0]
    assert [p.uri for p in ranked] == ["u1", "u2"]         # exactly K, ranked
    assert [p.above_threshold for p in ranked] == [True, False]  # honesty flag

    default = model.predict(["text"])[0]                    # no top_k: thresholds decide
    assert [p.uri for p in default] == ["u1"]
    assert default[0].above_threshold is None               # flag only in ranking mode


def test_default_has_no_average_cap():
    """Without top_k, EVERY label above its threshold is returned — the old
    round(avg_labels) auto-cap silently swallowed legitimate second labels."""

    class _TwoHotHead:
        def predict_proba(self, x):
            return np.tile([0.9, 0.8], (x.shape[0], 1))  # both above 0.5

    model = ClassifierModel(
        vectorizer=_SpyVectorizer(),  # type: ignore[arg-type]
        head=_TwoHotHead(),  # type: ignore[arg-type]
        classes=["u1", "u2"], task_type="multilabel",
        avg_labels=1.0,  # old behaviour would cap at round(1.0) = 1
        uri_to_label={},
    )
    assert len(model.predict(["text"])[0]) == 2  # both labels, no Ø cap


def test_predict_cleans_text_matching_training():
    """Prediction must clean input the same way training does.

    The TF-IDF vocabulary is fit on cleaned text (training runs clean_text on every
    row), so raw HTML/Markdown at inference would tokenize differently — a
    train/serve skew. predict_proba must apply clean_text before vectorizing.
    """
    spy = _SpyVectorizer()
    raw = "<p>Hallo <b>Welt</b></p>"
    _model(spy).predict_proba([raw])
    assert spy.seen == [data.clean_text(raw)] == ["Hallo Welt"]
