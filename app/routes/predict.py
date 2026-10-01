"""Prediction endpoints over JSON: predict, batch, several models, explain.

Classifying a whole uploaded CSV lives in ``predict_bulk`` — same feature area, but a
multipart upload streaming a file back is a different shape from a JSON request, and
it changes for different reasons.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from ..classifier import ClassifierModel, Prediction
from ..dataset_load import combine_text_columns
from ..explain import explain_prediction
from ..limiter import limiter, predict_limit
from ..schemas import ExplainRequest, MultiPredictRequest, PredictRequest
from ..security import require_role
from ._bundles import load_model, text_columns_for

router = APIRouter(tags=["Prediction"])


def _truncate(text: str) -> str:
    return (text[:200] + "...") if len(text) > 200 else text


def _pred_dict(p: Prediction) -> dict:
    d = {"uri": p.uri, "label": p.label, "confidence": round(p.confidence, 4)}
    if p.baseline_diff is not None:  # only present when requested (or in /predict/explain)
        d["baseline_diff"] = round(p.baseline_diff, 4)
    if p.label_f1 is not None:  # only when requested AND the bundle scored this label
        d["label_f1"] = round(p.label_f1, 4)
    if p.above_threshold is not None:  # ranking mode: keep forced entries honest
        d["above_threshold"] = p.above_threshold
    return d


def _texts_for(model_name: str, body: PredictRequest | MultiPredictRequest) -> list[str]:
    """What to classify: the texts as sent, or each record assembled the way the model's
    training text was -- its text columns, each repeated by its weight. Before, every client
    had to rebuild that itself, or classified a different text than the model was fit on
    (audit 2026-09-30, improvement 1)."""
    if body.records is None:
        assert body.texts is not None  # the request validates exactly one of the two
        return body.texts
    try:
        columns, weights = text_columns_for(model_name, None)
    except HTTPException as exc:
        raise HTTPException(400, f"Model '{model_name}' does not record which fields it was "
                                 "trained on; send `texts` instead.") from exc
    for at, record in enumerate(body.records):
        # An empty text gets the model's base-rate answer, which reads like a classification.
        if not any(column in record for column in columns):
            raise HTTPException(400, f"Record {at} carries none of the fields model "
                                     f"'{model_name}' was trained on: {', '.join(columns)}.")
    frame = pd.DataFrame(body.records, columns=columns).fillna("")
    return combine_text_columns(frame, columns, weights).tolist()


def _applied_settings(model: ClassifierModel, body: PredictRequest | MultiPredictRequest) -> dict:
    # Report what was actually APPLIED, per model. The rule is the model's — how a task
    # type decides, and therefore what ranking size that amounts to — and asking it here
    # rather than restating it is the point: this route had its own copy of "binary and
    # multiclass decide by argmax, so it is 1" (audit ARC-3).
    return {
        "threshold": body.threshold if body.threshold is not None else model.global_threshold,
        "top_k": model.applied_top_k(body.top_k),
        "classification_type": model.task_type,
        "auto_mode": body.top_k is None and body.threshold is None,
    }


def _build_response(model: ClassifierModel, body: PredictRequest, top_k: int | None) -> dict:
    texts = _texts_for(body.model_name, body)
    predictions = model.predict(
        texts, top_k=top_k, threshold=body.threshold, label_filter=body.label_filter,
        include_baseline_diff=body.include_baseline_diff,
        include_label_f1=body.include_label_f1,
    )
    results = [
        {"text": _truncate(text), "predictions": [_pred_dict(p) for p in row]}
        for text, row in zip(texts, predictions, strict=False)
    ]
    return {
        "model_name": body.model_name,
        "applied_settings": _applied_settings(model, body),
        "results": results,
    }


def _load_and_build_multi(body: MultiPredictRequest) -> dict:
    names = list(dict.fromkeys(body.model_names))  # dedupe, keep order
    models = {name: load_model(name) for name in names}  # loads run in the worker thread
    return _build_multi_response(models, body)


def _build_multi_response(models: dict[str, ClassifierModel], body: MultiPredictRequest) -> dict:
    # Per model: from records, each assembles its own text from its own columns and weights.
    texts = {name: _texts_for(name, body) for name in models}
    per_model = {
        name: model.predict(
            texts[name], top_k=body.top_k, threshold=body.threshold, label_filter=body.label_filter,
            include_baseline_diff=body.include_baseline_diff,
            include_label_f1=body.include_label_f1,
        )
        for name, model in models.items()
    }
    shown = texts[next(iter(models))]  # the echo: the first model's text
    results = [
        {
            "text": _truncate(text),
            "predictions_by_model": {
                name: [_pred_dict(p) for p in rows[i]] for name, rows in per_model.items()
            },
        }
        for i, text in enumerate(shown)
    ]
    return {
        "model_names": list(models),
        "applied_settings": {
            name: _applied_settings(model, body) for name, model in models.items()
        },
        "results": results,
    }


def _load_and_build(body: PredictRequest) -> dict:
    # _load (cold-cache disk read + skops deserialize) runs INSIDE the thread so
    # a cold model never blocks the single worker's event loop. HTTPExceptions it
    # raises propagate back through the await and are handled normally.
    return _build_response(load_model(body.model_name), body, body.top_k)


async def _predict(body: PredictRequest) -> dict:
    return await asyncio.to_thread(_load_and_build, body)


@router.post("/predict", summary="Classify texts")
@limiter.limit(predict_limit)
async def predict(request: Request, body: PredictRequest, _: str = Depends(require_role("readonly"))) -> dict:
    """Classify one or more texts with a trained model.

    For each text it returns labels with `uri`, a readable `label` and `confidence`
    (0–1), sorted by confidence. By default (no `top_k`) the per-label thresholds
    tuned during training decide the labels — multilabel returns every label above
    its threshold, multiclass/binary the single best label.

    **Parameters:** `model_name`; optionally `threshold` (overrides the trained value),
    `top_k` (`null` = the model **decides**: multilabel via its tuned per-label thresholds,
    multiclass via argmax; `N` = **ranking**: exactly the N most probable labels regardless
    of thresholds, each flagged with `above_threshold` for a multilabel model; `0` =
    ranking of the typical label count), `label_filter` (only labels containing the
    substring), `include_baseline_diff` (adds confidence minus the model's empty-text
    prediction per label), `include_label_f1` (adds each label's F1 from the training
    evaluation — how much a high confidence on THIS label is worth; left out for a label
    the bundle has no F1 for, such as one no real row could validate). The response
    includes `applied_settings` with the values actually applied. **Auth:** readonly.
    """
    return await _predict(body)


@router.post("/predict/batch", summary="Classify texts (deprecated alias of /predict)",
             deprecated=True)
@limiter.limit(predict_limit)
async def predict_batch(request: Request, body: PredictRequest, _: str = Depends(require_role("readonly"))) -> dict:
    """**Deprecated — use `POST /predict`, which is the same endpoint.**

    Same body, same auth, same rate limit, same code. `/predict` has always taken a
    list of `texts` (up to 1000), so this never offered a batching capability the
    other one lacked; the old summary implied it did. Kept so existing callers keep
    working. **Auth:** readonly.
    """
    return await _predict(body)


@router.post("/predict/multi", summary="Classify texts with several models (target fields) in one call")
@limiter.limit(predict_limit)
async def predict_multi(
    request: Request, body: MultiPredictRequest, _: str = Depends(require_role("readonly"))
) -> dict:
    """Classify each text with several models at once — one model per target field
    (e.g. subjects + resource type + educational context).

    Per text, `predictions_by_model` maps each model name to its predictions.
    **Evaluation stays per model:** every bundle carries its own metrics and tuned
    thresholds, and this endpoint applies each model's own thresholds — it only
    orchestrates, nothing is re-evaluated jointly. Options (`top_k`, `threshold`,
    `label_filter`, `include_baseline_diff`, `include_label_f1`) apply to all listed
    models; `top_k=0` resolves per model from its average label count. **Auth:** readonly.
    """
    return await asyncio.to_thread(_load_and_build_multi, body)


@router.post("/predict/explain", summary="Explainable classification")
@limiter.limit(predict_limit)
async def predict_explain(
    request: Request, body: ExplainRequest, _: str = Depends(require_role("readonly"))
) -> dict:
    """Classify a text and explain the result.

    In addition to the predictions: `all_scores` (confidence for **all** labels, each
    with `baseline_diff` and `label_f1`) and `word_importance` — the most influential
    words per predicted label, determined by leave-one-out (confidence drop when a word
    is removed). Both reliability signals are always included here, no flag needed;
    `label_f1` is `null` for a label the bundle has no F1 for (unlike `/predict`, which
    leaves it out).

    An `impact` is a difference between two *word lists*: the text's words rejoined by
    single spaces, with and without that one word — not against the `confidence` beside
    it, which describes the text as sent (punctuation, spacing, and any word past the
    60-word cap included). So the impacts are comparable with each other but do not add
    up to the confidence. With character n-grams in the model a word also carries the
    junction it sits in, which is why a connector can score above a content word.

    More expensive than `/predict`. **Auth:** readonly.
    """
    def load_and_explain() -> dict:
        return explain_prediction(load_model(body.model_name), body.text, body.top_n_words)

    result = await asyncio.to_thread(load_and_explain)
    return {**result, "model_name": body.model_name}
