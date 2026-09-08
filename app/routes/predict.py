"""Prediction endpoints: predict, batch, explainable prediction."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from ..classifier import ClassifierModel, Prediction
from ..explain import explain_prediction
from ..limiter import limiter, predict_limit
from ..registry import UnsafeModelError, get_registry
from ..schemas import ExplainRequest, MultiPredictRequest, PredictRequest
from ..security import require_role, safe_name

router = APIRouter(tags=["Prediction"])


def _load(model_name: str) -> ClassifierModel:
    safe_name(model_name, "model name")
    registry = get_registry()
    if not registry.exists(model_name):
        raise HTTPException(404, f"Model '{model_name}' not found.")
    try:
        return registry.get(model_name)
    except FileNotFoundError as exc:
        # Deleted in the window between exists() and get() (TOCTOU) -> 404, the
        # same clean response as a plainly missing model, never a 500.
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    except UnsafeModelError as exc:
        # A bundle that exists but can't be safely loaded (corrupt/version drift)
        # is a client-visible 422, not a 500 that leaks internals.
        raise HTTPException(422, "Model bundle is invalid or unloadable.") from exc


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


def _applied_settings(model: ClassifierModel, body: PredictRequest | MultiPredictRequest) -> dict:
    # Report what was actually APPLIED, per model: an explicit top_k resolves to
    # the ranking size; without one, multiclass/binary decide with argmax (1)
    # and multilabel with its tuned thresholds (no cap -> null).
    single = model.task_type in ("binary", "multiclass")
    applied_top_k = model.resolved_top_k(body.top_k)
    if applied_top_k is None and single:
        applied_top_k = 1
    return {
        "threshold": body.threshold if body.threshold is not None else model.global_threshold,
        "top_k": applied_top_k,
        "classification_type": model.task_type,
        "auto_mode": body.top_k is None and body.threshold is None,
    }


def _build_response(model: ClassifierModel, body: PredictRequest, top_k: int | None) -> dict:
    predictions = model.predict(
        body.texts, top_k=top_k, threshold=body.threshold, label_filter=body.label_filter,
        include_baseline_diff=body.include_baseline_diff,
        include_label_f1=body.include_label_f1,
    )
    results = [
        {"text": _truncate(text), "predictions": [_pred_dict(p) for p in row]}
        for text, row in zip(body.texts, predictions, strict=False)
    ]
    return {
        "model_name": body.model_name,
        "applied_settings": _applied_settings(model, body),
        "results": results,
    }


def _load_and_build_multi(body: MultiPredictRequest) -> dict:
    names = list(dict.fromkeys(body.model_names))  # dedupe, keep order
    models = {name: _load(name) for name in names}  # loads run in the worker thread
    return _build_multi_response(models, body)


def _build_multi_response(models: dict[str, ClassifierModel], body: MultiPredictRequest) -> dict:
    per_model = {
        name: model.predict(
            body.texts, top_k=body.top_k, threshold=body.threshold, label_filter=body.label_filter,
            include_baseline_diff=body.include_baseline_diff,
            include_label_f1=body.include_label_f1,
        )
        for name, model in models.items()
    }
    results = [
        {
            "text": _truncate(text),
            "predictions_by_model": {
                name: [_pred_dict(p) for p in rows[i]] for name, rows in per_model.items()
            },
        }
        for i, text in enumerate(body.texts)
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
    return _build_response(_load(body.model_name), body, body.top_k)


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
    of thresholds, each flagged with `above_threshold`; `0` = ranking of the typical label
    count), `label_filter` (only labels containing the substring), `include_baseline_diff`
    (adds confidence minus the model's empty-text prediction per label), `include_label_f1`
    (adds each label's F1 from the training evaluation — how much a high confidence on
    THIS label is worth). The response includes `applied_settings` with the values
    actually applied. **Auth:** readonly.
    """
    return await _predict(body)


@router.post("/predict/batch", summary="Classify multiple texts (batch)")
@limiter.limit(predict_limit)
async def predict_batch(request: Request, body: PredictRequest, _: str = Depends(require_role("readonly"))) -> dict:
    """Like `POST /predict` — also takes a list of `texts` and is intended for larger
    batches. **Auth:** readonly."""
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
    is removed). Both reliability signals are always included here, no flag needed.
    More expensive than `/predict`. **Auth:** readonly.
    """
    def load_and_explain() -> dict:
        return explain_prediction(_load(body.model_name), body.text, body.top_n_words)

    result = await asyncio.to_thread(load_and_explain)
    return {**result, "model_name": body.model_name}
