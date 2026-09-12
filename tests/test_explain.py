"""What the leave-one-out explanation attributes to a word.

The endpoint answers "why this label" by removing one word and re-predicting, so the
number it reports is a DIFFERENCE — and both sides of it have to be the same text apart
from that one word. The model here is scripted rather than fitted: with a real model the
test would measure how well that model happens to separate a toy sentence, not what the
attribution does.
"""

import numpy as np

from app.classifier import Prediction
from app.explain import explain_prediction


class ScriptedModel:
    """A model whose confidence is a known function of the text.

    0.5 for the word "säuren", 0.3 for "basen", and 0.2 for a full stop anywhere — the
    last one stands for everything the rejoined word list drops but the original text
    carries (punctuation, spacing, words past the 60-word cap).
    """

    classes = ["uri:chem"]
    uri_to_label = {"uri:chem": "Chemie"}
    per_label_f1 = {"uri:chem": 0.9}
    task_type = "multilabel"

    def score(self, text: str) -> float:
        lowered = text.lower()
        return (0.5 * ("säuren" in lowered) + 0.3 * ("basen" in lowered)
                + 0.2 * ("." in lowered))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.array([[self.score(text)] for text in texts], dtype=float)

    def baseline_proba(self) -> np.ndarray:
        return np.array([0.0])

    def predict(self, texts: list[str], **_kwargs) -> list[list[Prediction]]:
        return [[Prediction(uri="uri:chem", label="Chemie", confidence=self.score(text),
                            baseline_diff=self.score(text), label_f1=0.9)]
                for text in texts]


def _impacts(result: dict) -> dict[str, float]:
    words = result["word_importance"]["uri:chem"]["top_words"]
    return {entry["word"]: entry["impact"] for entry in words}


def test_a_words_impact_is_measured_against_the_text_it_was_removed_from():
    """Both sides of the difference are the same word list, minus one word. Measured
    against the ORIGINAL text instead, every impact carried the full stop the variants
    had dropped: the irrelevant "wir" showed 0.2, and "basen" 0.5 instead of 0.3."""
    result = explain_prediction(ScriptedModel(), "Wir mischen Säuren und Basen.", top_n_words=5)

    impacts = _impacts(result)
    assert impacts["Basen"] == 0.3, "its own contribution, not plus the full stop"
    assert impacts["Säuren"] == 0.5
    assert impacts["Wir"] == 0.0, "a word that changes nothing has no impact"
    assert impacts["und"] == 0.0


def test_the_reported_confidence_still_describes_the_text_as_given():
    """Only the attribution compares like with like. What the model says about the text
    is what it says about THE text — punctuation included."""
    model = ScriptedModel()
    text = "Wir mischen Säuren und Basen."
    result = explain_prediction(model, text, top_n_words=5)

    assert result["predictions"][0]["confidence"] == round(model.score(text), 4)
    assert result["all_scores"]["uri:chem"]["confidence"] == model.score(text)


def test_a_single_word_text_gets_no_attribution():
    """Removing the only word leaves nothing to compare against."""
    result = explain_prediction(ScriptedModel(), "Säuren", top_n_words=5)
    assert result["word_importance"] == {}
