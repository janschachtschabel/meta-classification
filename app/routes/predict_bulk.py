"""Classifying a whole CSV: upload, stream the answers back.

Split from ``predict`` when it landed: a multipart upload that streams a file back is
a different shape from a JSON request-response, and the two change for different
reasons — one when the prediction contract moves, the other when the bulk pipeline does.

The pipeline itself (chunked reading, text assembly, CSV writing) is ``predict_csv``;
this file is the thin route around it: auth, validation, staging, HTTP mapping.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .. import predict_csv as predict_csv_mod
from ..bundle_meta import as_mapping, as_names
from ..errors import TrainingInputError
from ..limiter import limiter, predict_limit
from ..registry import get_registry
from ..security import require_role, spool_upload_capped
from ..settings import Settings, get_settings
from .predict import load_model

router = APIRouter(tags=["Prediction"])

# What may appear in a Content-Disposition filename. The name is echoed into a
# response header, so anything else — quotes and CR/LF above all — is replaced
# rather than trusted.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _download_name(upload_name: str | None, model_name: str) -> str:
    """The filename the answer is offered under, derived from the uploaded one."""
    stem = _UNSAFE_IN_FILENAME.sub("-", Path(upload_name or model_name).stem).strip("-")
    return f"{stem or model_name}-predictions.csv"


def _text_columns_for(model_name: str, requested: list[str] | None) -> tuple[list[str], dict]:
    """Which columns to read, and how often each is repeated — from the bundle.

    How a text is assembled is part of what the model was fit on, so the columns and
    their weights are read from the bundle rather than asked of the caller. An explicit
    ``text_columns`` overrides the names (a newer export may call them something else);
    the weights then narrow to those columns, exactly as a training request narrows them.
    """
    metadata = as_mapping(get_registry().info(model_name).get("metadata"))
    columns = requested or as_names(metadata.get("text_columns"))
    if not columns:
        raise HTTPException(
            400, "This bundle does not record which text columns it was trained on; "
                 "pass text_columns explicitly.")
    weights = as_mapping(metadata.get("text_column_weights"))
    return columns, {col: weight for col, weight in weights.items() if col in columns}


@router.post("/predict/csv", summary="Classify every row of a CSV (upload → CSV download)")
@limiter.limit(predict_limit)
async def predict_csv(
    request: Request,
    file: UploadFile = File(..., description="A CSV carrying the text columns the model was trained on"),
    model_name: str = Form(..., description="The model to classify with"),
    text_columns: list[str] | None = Form(None, description="Override the bundle's text columns"),
    separator: str = Form(";", description="CSV field separator"),
    threshold: float | None = Form(None, description="Override the model's tuned threshold"),
    top_k: int | None = Form(None, description="Ranking mode: the N most probable labels per row"),
    _: str = Depends(require_role("readonly")),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Classify every row of an uploaded CSV and stream the answers back as CSV.

    The daily editorial job is "classify these 500 new items", not one text. `/predict`
    can do it, but only if the caller assembles each row's text — and **how a text is
    assembled is part of what the model was fit on**, so that is the one thing not to
    leave to the caller: the text columns and their repetition weights are read from
    the bundle (`text_columns` overrides the names if a newer export renamed them).

    The answer is `row,uri,label,confidence,above_threshold`, one line per predicted
    label, where `row` is the 0-based number of the input row — that is how the answers
    are joined back onto the original file. **A row the model asserts nothing for still
    gets a line**, with the prediction fields empty, so "which items did it refuse" is
    readable off the result.

    Nothing is materialised on either side: the CSV is read in chunks and the answer
    leaves as it is produced. Size limit and rate limit as for the other uploads.
    **Auth:** readonly.
    """
    model = await asyncio.to_thread(load_model, model_name)
    columns, weights = _text_columns_for(model_name, text_columns)

    # Spooled to our own file rather than read from the upload's own handle: the
    # response body is produced AFTER this function returns, and the request's
    # temporary file is not ours to keep alive that long.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(prefix=".predict-", suffix=".csv.tmp", dir=settings.data_dir)
    os.close(handle)
    path = Path(staged)
    try:
        await spool_upload_capped(file, settings.max_upload_mb * 1024 * 1024, path)
        # Before a byte is streamed: once the response starts, the status line is
        # already 200 and a bad header could only arrive as garbage in the body.
        await asyncio.to_thread(predict_csv_mod.check_columns, path, columns, separator=separator)
    except TrainingInputError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise

    return StreamingResponse(
        predict_csv_mod.classify_csv(
            path, model, text_columns=columns, weights=weights,
            separator=separator, threshold=threshold, top_k=top_k,
        ),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="{_download_name(file.filename, model_name)}"'},
        background=BackgroundTask(path.unlink, missing_ok=True),
    )
