"""Corrections editors make to predictions: recorded, seen, backed up, restored, trained on.

Thin, like every route module: the store and the export shape live in ``app.feedback``,
reading the collection back (view, backup, import, dataset) in ``app.feedback_review``.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from .. import feedback as feedback_store
from .. import feedback_review
from ..errors import FeedbackWriteError
from ..limiter import default_limit, export_limit, limiter
from ..schemas import FeedbackRequest
from ..schemas.datasets import CorrectionsDatasetRequest
from ..security import require_role, safe_name
from ..settings import Settings, get_settings

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

    Answers **503** if the correction could not be written, naming the setting to fix.
    Never a 200 it cannot back up: this is training data, and a silent loss is the one
    failure nobody notices until the next run does not improve.
    """
    safe_name(body.model_name, "model name")
    try:
        # Off the event loop: a process's first correction counts what is on disk (R02).
        collected = await asyncio.to_thread(feedback_store.append, body.model_dump())
    except FeedbackWriteError as exc:
        # 503 rather than 500: the request was valid, the storage is not ready, and
        # retrying is the right move — which a sanitized "Internal server error" says
        # to nobody. The message is written for whoever has to fix it.
        raise HTTPException(503, str(exc)) from exc
    return {"status": "recorded", "collected": collected}


@router.get("/feedback/export", summary="Corrections as a training-ready CSV",
            response_model=None)
@limiter.limit(export_limit)
async def export_feedback(
    request: Request,
    limit: int | None = Query(
        None, ge=1, le=100_000,
        description="At most this many corrections, oldest first. Omitted = all of them.",
    ),
    offset: int = Query(
        0, ge=0,
        description=(
            "Exported corrections to skip, oldest first. With `limit` this pages the "
            "collection: take a page, add its size to the offset, ask again. Counts "
            "EXPORTED rows, so a correction recorded without a label does not shift the "
            "page. Position rather than a timestamp because several corrections can share "
            "one clock tick."
        ),
    ),
    format: Literal["csv", "jsonl"] = Query(  # noqa: A002 - the query parameter's public name
        "csv",
        description=(
            "`csv` (default): the training file described here. `jsonl`: the backup -- every "
            "correction with every field (model, what was predicted, when), one JSON object per "
            "line, as `POST /feedback/import` takes it back. `limit`/`offset` apply to `csv` only."
        ),
    ),
    _: str = Depends(require_role("admin")),
) -> Response:
    """Every correction as a CSV `/train` can read directly, or as a complete backup.

    Columns `text` and `labels`, with the separators `TrainRequest` defaults to — so it
    trains with the form's own defaults and needs nothing explained. Corrections with no
    corrected label are left out: the loader drops label-less rows, so including them
    would overstate what the file contributes.

    Streamed row by row. The collection is uncapped on purpose — it is training data, not
    a log — so a response that built the whole document first grew with the volume rather
    than with the answer, and held three copies of it while doing so.

    **Auth:** admin — posting one correction is part of the job; walking off with every
    text an editor ever pasted is not.
    """
    if format == "jsonl":
        return StreamingResponse(
            feedback_review.backup_lines(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="feedback-backup.jsonl"'},
        )
    return StreamingResponse(
        feedback_store.iter_csv(limit=limit, offset=offset),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="feedback.csv"'},
    )


@router.get("/feedback", summary="The recorded corrections, newest first")
@limiter.limit(export_limit)  # reads the whole collection, like the export
async def list_feedback(
    request: Request,
    model: str | None = Query(None, max_length=200, description="Only the corrections of this model."),
    limit: int = Query(50, ge=1, le=500, description="Corrections per page, newest first."),
    offset: int = Query(0, ge=0, description="Corrections to skip, newest first."),
    _: str = Depends(require_role("admin")),
) -> dict:
    """What editors corrected, newest first: `total` (of the filter), `models` (every model
    with its count, for a filter) and `items` -- each with when, which model, the text (the
    first 500 characters; `text_truncated` says when there is more), what was predicted and
    what is right.

    **Auth:** admin -- the view shows every text an editor ever pasted, like the export.
    """
    return await asyncio.to_thread(feedback_review.page, model, limit, offset)


def _backup_lines(upload: UploadFile) -> Iterator[str]:
    """The upload's lines, read from its spool file one at a time.

    Split at ``\n`` only. ``str.splitlines`` also splits at U+2028, U+2029 and U+0085, which
    the backup writes unescaped inside a text (``ensure_ascii=False``) -- such a correction
    came back as two invalid halves, lost on restore. One line at a time, too: the bytes, the
    decoded text and the list of lines no longer sit in memory side by side. The size is
    bounded before this runs, by the body guard (``max_upload_mb``).
    """
    upload.file.seek(0)
    reader = io.TextIOWrapper(upload.file, encoding="utf-8-sig", newline="\n")
    try:
        yield from reader
    finally:
        reader.detach()  # the upload owns its file and closes it


@router.post("/feedback/import", summary="Restore corrections from a backup")
@limiter.limit(export_limit)
async def import_feedback(
    request: Request,
    file: UploadFile = File(..., description="A backup from `GET /feedback/export?format=jsonl`."),
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Add the corrections of a backup that are not collected yet -- from another instance,
    or this one before a volume was replaced.

    Every line must be a correction as `POST /feedback` would accept it, plus its
    `recorded_at`; one that is not is counted in `invalid` and skipped. Exact repeats are
    counted in `duplicates` and skipped, so importing a backup twice adds nothing.

    Answers **400** for a file that is not UTF-8 text, **503** when the new corrections would
    pass the cap (`APIV3_MAX_FEEDBACK_MB`) or cannot be written -- nothing is added then.
    **Auth:** admin.
    """
    try:
        return await asyncio.to_thread(feedback_review.import_lines, _backup_lines(file))
    except UnicodeDecodeError as exc:
        # Met while reading, before anything is written: the import is all or nothing.
        raise HTTPException(400, "The backup is not UTF-8 text (one JSON object per line).") from exc
    except FeedbackWriteError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/feedback/dataset", summary="Save the corrections as a dataset")
@limiter.limit(export_limit)
async def corrections_as_dataset(
    request: Request,
    body: CorrectionsDatasetRequest,
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Write the training CSV of `GET /feedback/export` into the data directory as a new
    dataset, so it can be inspected, trained on, or downloaded and combined with the data the
    model learned from. Columns `text` and `labels`, separators `;` and `,` -- the training
    form's defaults.

    Answers **409** if a dataset of that name exists (it is never replaced), **400** for a bad
    name, a `.csv.gz` name (the file is plain CSV) or when no correction carries a label yet.
    **Auth:** admin.
    """
    name = body.dataset_name
    if name.casefold().endswith(".gz"):
        raise HTTPException(400, "The corrections are written as plain CSV: use a name ending in .csv.")
    if not name.casefold().endswith(".csv"):
        name += ".csv"
    safe_name(name, "dataset name")
    try:
        rows = await asyncio.to_thread(feedback_review.write_dataset, settings.data_dir, name)
    except FileExistsError as exc:
        raise HTTPException(409, f"Dataset '{name}' already exists.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"dataset_name": name, "rows": rows}
