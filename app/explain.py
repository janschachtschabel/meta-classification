"""Explainable prediction: leave-one-out word importance.

Core logic for ``POST /predict/explain``, kept out of the route module so routes
stay thin (auth, validation, HTTP mapping only). With a linear head the
leave-one-out confidence drop is a cheap, faithful importance signal.
"""

from __future__ import annotations

import re

from .classifier import ClassifierModel

# Cap the number of leave-one-out variants: cost is one predict_proba row per
# word, so unbounded input would turn one request into thousands of predictions.
_MAX_WORDS = 60
_WORD_RE = re.compile(r"\b\w+\b")


def explain_prediction(model: ClassifierModel, text: str, top_n_words: int) -> dict:
    """Classify ``text`` and attribute the result to its most influential words.

    Returns the predictions, per-label confidence for ALL labels (each with its
    ``baseline_diff`` and ``label_f1``), and — for the top predicted labels — the
    words whose removal drops the confidence most (leave-one-out).

    The drop is measured between the word list with and without that word, not against
    the original text: the variants carry no punctuation and at most ``_MAX_WORDS``
    words, and comparing across that difference put it into every impact.

    What it cannot separate is context. The model reads character n-grams too, so
    removing a word also destroys the n-grams spanning it — a connector inside a
    characteristic phrase ("Säuren und Basen") is charged for the phrase.
    """
    base = model.predict_proba([text])[0]
    baseline = model.baseline_proba()  # empty-text base rates: always shown here (diagnostic endpoint)
    scores = {
        uri: {
            "label": model.uri_to_label.get(uri, uri),
            "confidence": float(base[i]),
            "baseline_diff": round(float(base[i] - baseline[i]), 4),
            # How good the model is on this label AT ALL — the counterweight to a
            # high confidence. Unconditional here for the same reason baseline_diff
            # is: this endpoint exists to judge a prediction, not just to make one.
            "label_f1": model.per_label_f1.get(uri),
        }
        for i, uri in enumerate(model.classes)
    }
    predicted = model.predict([text], include_baseline_diff=True, include_label_f1=True)[0]

    words = _WORD_RE.findall(text)[:_MAX_WORDS]
    importance: dict[str, dict] = {}
    if len(words) > 1 and predicted:
        # The first variant is the FULL word list, and it is what the others are measured
        # against: an impact has to be the difference between the same text with and
        # without one word. Against the original text instead, every impact also carried
        # what rejoining drops — its punctuation, its spacing, and any word past the cap.
        variants = [" ".join(words)]
        variants += [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        variant_proba = model.predict_proba(variants)
        for pred in predicted[:5]:
            col = model.classes.index(pred.uri)
            whole = float(variant_proba[0, col])
            impacts = sorted(
                ({"word": words[i], "impact": round(whole - float(variant_proba[i + 1, col]), 4)}
                 for i in range(len(words))),
                key=lambda d: d["impact"], reverse=True,
            )
            importance[pred.uri] = {"label": pred.label, "top_words": impacts[:top_n_words]}

    return {
        "text": text,
        "predictions": [
            {"uri": p.uri, "label": p.label, "confidence": round(p.confidence, 4),
             "baseline_diff": round(p.baseline_diff or 0.0, 4),
             "label_f1": p.label_f1}
            for p in predicted
        ],
        "word_importance": importance,
        "all_scores": scores,
    }
