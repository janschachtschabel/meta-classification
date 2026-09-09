"""Corrections editors make to predictions, and the CSV a training run reads back.

Thin, like every route module: the store and the export shape live in ``app.feedback``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from .. import feedback as feedback_store
from ..limiter import default_limit, export_limit, limiter
from ..schemas import FeedbackRequest
from ..security import require_role, safe_name

router = APIRouter(tags=["Feedback"])


@router.post("/feedback", summary="Record a correction to a prediction")
@limiter.limit(default_limit)
async def record_feedback(
    request: Request, body: FeedbackRequest, _: str = Depends(require_role("readonly"))
) -> dict:
    """Write down that a prediction was wrong, and what the right answer was.

    **The recognition rate improves with use, or it improves only when somebody produces
    a new export.** This is the first half of that loop; `GET /feedback/export` is the
    second. Corrections are appended and never dropped — this is training data, not a
    log, so the oldest is worth as much as the newest.

    `corrected` may be empty: "none of these apply" is a real thing to say. It is
    recorded, and left out of the export, because a row without labels is not trainable.

    **Auth:** readonly — correcting an answer is part of classifying, and the editors
    who spot the mistakes are the ones without an admin key.
    """
    safe_name(body.model_name, "model name")
    collected = feedback_store.append(body.model_dump())
    return {"status": "recorded", "collected": collected}


@router.get("/feedback/export", summary="Corrections as a training-ready CSV",
            response_model=None)
@limiter.limit(export_limit)
async def export_feedback(
    request: Request, _: str = Depends(require_role("admin"))
) -> Response:
    """Every correction as a CSV `/train` can read directly.

    Columns `text` and `labels`, with the separators `TrainRequest` defaults to — so it
    trains with the form's own defaults and needs nothing explained. Corrections with no
    corrected label are left out: the loader drops label-less rows, so including them
    would overstate what the file contributes.

    **Auth:** admin — posting one correction is part of the job; walking off with every
    text an editor ever pasted is not.
    """
    return Response(
        content=feedback_store.to_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="feedback.csv"'},
    )
