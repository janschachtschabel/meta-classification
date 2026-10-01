"""Model management: list, details, evaluate, delete, export, import.

The share-link routes, which serve models and datasets alike, are in
:mod:`app.routes.share`. A few lines past the ~300-line guide with one reason to change —
the model endpoints; what several route modules share already lives in ``routes/_*.py``
(audit 2026-09-27, M-2).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi import (
    Path as PathParam,
)

from .. import data as data_mod
from ..dataset_load import require_columns
from ..errors import TrainingInputError
from ..evaluate import run_evaluation
from ..jobs import job_runner
from ..limiter import default_limit, export_limit, limiter, train_limit
from ..registry import UnsafeModelError, get_registry
from ..responses import TrainStartedResponse
from ..schemas import EvaluateRequest, ExportRequest, ModelInfo
from ..security import require_role, safe_name, spool_upload_capped
from ..settings import Settings, get_settings
from ..sharing import get_share_store
from ._bundles import staged_zip_response
from ._jobs import start_or_queue
from ._paging import Limit, Offset, page

router = APIRouter(tags=["Models"])

# The name every model route takes in its path, described once for all of them.
ModelName = Annotated[str, PathParam(description=(
    "Name of a model as `GET /models` lists it: at most 100 characters and no path "
    "characters (400 otherwise). A name no model has answers 404."))]


@router.get("/models", summary="List all models")
async def list_models(
    limit: Limit = None, offset: Offset = 0,
    _: str = Depends(require_role("readonly")),
) -> list[str]:
    """Names of all available models (bundles in the model directory).

    Omitting `limit` returns every name, as before. `limit`/`offset` page through a large
    registry; the order is the registry's own and stable between calls. **Auth:** readonly.
    """
    return page(await asyncio.to_thread(get_registry().list), limit, offset)


@router.get("/models/{model_name}", summary="Model details & metrics")
async def model_info(model_name: ModelName, _: str = Depends(require_role("readonly"))) -> dict:
    """Configuration + training metadata/metrics of a model.

    Reads only the JSON metadata (weights are not loaded). Includes the backend,
    classes, thresholds, task type and `label_vocabulary`; under `metadata` the training
    record: the metrics (on the held-out test split, or out-of-fold for k-fold — see
    `metadata.evaluation`), per-label F1 and support, the text columns and weights the
    model expects, `info`, and `evaluations` (results of `POST /models/{name}/evaluate`).

    The macro averages and `per_label_f1` cover the labels those rows carry a positive
    for; `metrics.labels_not_scored` names the rest. Such a label scores F1 0.0 under
    every model, so averaging it in would measure the split. `f1_micro`, `n_labels` and
    the labels-per-row averages still cover the whole label space.

    A model trained on a dataset carrying data-prep's marks also has
    `metadata.synthetic_data`: the run's `mode` (`synthetic_rows`); how many rows an LLM
    wrote (`generated_rows`), showed as examples (`example_rows`) or completed
    (`enriched_rows`), how many generated rows were left out, once per text as the dedupe
    would have kept them (`excluded_generated_rows`)
    and how many trained without validating (`train_only_rows`); what the metrics were
    computed on (`validated_on`: `real_rows`, or `all_rows` with the reason in
    `fallback`; `scored_rows`, and `too_few_rows` when that count is too small for the
    numbers to compare with another run); the labels no real row could score (`labels_not_validated`
    — they have no F1); and the thin labels with the cut chosen for them (`thin_labels`,
    `thin_label_threshold`). **Auth:** readonly.
    """
    safe_name(model_name, "model name")
    try:
        return await asyncio.to_thread(get_registry().info, model_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    except UnsafeModelError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/models/{model_name}/labels", summary="Per-label diagnostics")
async def model_labels(model_name: ModelName, _: str = Depends(require_role("readonly"))) -> list[dict]:
    """Every label with its F1, its support (rows carrying it in the training data,
    AI-marked rows included) and its threshold, **weakest first**.

    The headline F1 says how good a model is on average; this says where it is weak,
    which is what decides whether one answer deserves a second look. `support` makes
    a score readable — 0.13 on 25 rows is a different statement from 0.13 on 5,000.
    `threshold` is `null` for binary/multiclass, where serving decides by argmax and
    reads no threshold. `f1` is `null` for a label the evaluated rows could not score —
    they carry no positive for it (`metrics.labels_not_scored`) or no REAL row does
    (`synthetic_data.labels_not_validated`), both in `GET /models/{name}` — and, like
    `support`, for bundles trained before it was recorded. Those labels come last:
    unknown is not the same as weak. **Auth:** readonly.
    """
    safe_name(model_name, "model name")
    try:
        return await asyncio.to_thread(get_registry().label_diagnostics, model_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    except UnsafeModelError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.put("/models/{model_name}/info", summary="Set the model's documentation")
@limiter.limit(default_limit)
async def set_model_info(
    request: Request,
    model_name: ModelName,
    body: ModelInfo,
    _: str = Depends(require_role("admin")),
) -> dict:
    """Store author, purpose, data provenance and license in the model bundle.

    These are the facts the pipeline cannot measure, and they matter when the model is
    handed to a third party: the metadata records the training file's NAME, not where
    that file came from. They travel inside the exported archive.

    **Replaces** the whole block — send every field you want kept; `{}` clears it.
    Editable at any time because documentation is presentation-only: nothing in the
    serving path reads it, so correcting an author name costs no retrain.
    The label vocabulary is deliberately not settable here — `GET /models/{name}`
    derives it from the class URIs. **Auth:** admin · rate limit active.
    """
    safe_name(model_name, "model name")
    try:
        stored = await asyncio.to_thread(
            get_registry().update_info, model_name, body.model_dump(exclude_none=True)
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    return {"status": "updated", "model_name": model_name, "info": stored}


@router.delete("/models/{model_name}", summary="Delete a model")
@limiter.limit(default_limit)
async def delete_model(request: Request, model_name: ModelName, _: str = Depends(require_role("admin"))) -> dict:
    """Remove a model from the in-memory cache and from disk (irreversible), and revoke
    its share links. **Auth:** admin · rate limit active."""
    safe_name(model_name, "model name")
    try:
        # rmtree of a large bundle is blocking disk work — off the event loop.
        await asyncio.to_thread(get_registry().delete, model_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    # A link names the model, not this bundle: left alive, it would hand out
    # whatever is trained or imported under the name next.
    get_share_store().revoke_for("model", model_name)
    return {"status": "deleted", "model_name": model_name}


@router.post("/models/{model_name}/export", summary="Export a model (download or share link)",
             response_model=None)
@limiter.limit(export_limit)
async def export_model(
    request: Request,
    model_name: ModelName,
    body: ExportRequest | None = Body(None),
    _: str = Depends(require_role("admin")),
) -> Response | dict:
    """Export a model as a portable, pickle-free ZIP bundle (sklearn + skops).

    Without a body / `generate_share_url=false`: a direct ZIP download. With
    `generate_share_url=true`: an expiring share link (`expires_hours`, 1–168),
    retrievable via `GET /share/{id}`. **Auth:** admin · rate limit active.
    """
    safe_name(model_name, "model name")
    registry = get_registry()
    if not registry.exists(model_name):
        raise HTTPException(404, f"Model '{model_name}' not found.")
    body = body or ExportRequest()
    if body.generate_share_url:
        share_id, expires_at = get_share_store().create("model", model_name, body.expires_hours)
        return {"share_url": f"/share/{share_id}", "share_id": share_id, "expires_at": expires_at}
    # Zipping a production bundle (100+ MB skops) takes seconds of CPU/disk —
    # run it in a worker thread so /health and predicts stay responsive (same
    # rationale as the predict cold-load offload).
    return await asyncio.to_thread(staged_zip_response, model_name)


@router.post("/models/import", summary="Import a model (file upload only)")
@limiter.limit(export_limit)
async def import_model(
    request: Request,
    file: UploadFile = File(..., description="A model bundle (.zip) exported by this API"),
    new_name: str | None = Form(
        None,
        description=(
            "Install the model under this name instead of the archive's file name (minus "
            "`.zip`). An existing name is refused (409)."
        ),
    ),
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Import a model bundle from an uploaded ZIP (no URL fetch -> no SSRF).

    The archive is validated (no path-traversal names, required files present) and
    loaded safely via skops — files with unknown/unsafe types are rejected.
    `new_name` overrides the model name. Size limit active. **Auth:** admin.
    """
    # In any letter case, like a dataset's suffix: `FAECHER.ZIP` is a ZIP too.
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "File must be a .zip model bundle.")
    name = new_name or Path(file.filename).stem
    safe_name(name, "model name")
    # A training still writing this name would race the import on the same
    # staging dir (mutual clobber / franken-bundle). Two independent signals:
    # the job STATUS (normal runs), and THREAD liveness — after stop(hard=true)
    # the status lies ("idle") while the abandoned thread keeps saving its bundle.
    status_busy = job_runner.is_running() and job_runner.snapshot().get("model_name") == name
    if status_busy or job_runner.active_model_name() == name:
        raise HTTPException(409, f"A training for model '{name}' is currently running; retry after it finishes.")
    # A QUEUED run holds the name too: it was accepted first, and the import
    # would make it fail when its turn comes.
    if name in job_runner.queued_names():
        raise HTTPException(409, f"A training for model '{name}' is queued; retry after it finishes, "
                                 "or import under another name.")
    # Spooled to disk rather than joined in memory: `read_upload_capped` holds the chunks
    # AND the joined copy, i.e. twice the 200 MB cap, before a byte lands — and the import
    # only ever moves those bytes onto disk anyway (audit API-5). Beside the bundles, under
    # the hidden `.*.tmp` name the startup sweep already cleans.
    registry = get_registry()
    registry.dir.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(prefix=".import-", suffix=".zip.tmp", dir=registry.dir)
    os.close(handle)
    archive_path = Path(staged)
    try:
        await spool_upload_capped(file, settings.max_upload_mb * 1024 * 1024, archive_path)
        # Validation loads both skops files — seconds of CPU; off the event loop.
        info = await asyncio.to_thread(registry.import_archive, name, archive_path)
    except FileExistsError as exc:
        raise HTTPException(409, f"Model '{name}' already exists.") from exc
    except (UnsafeModelError, ValueError) as exc:
        raise HTTPException(400, f"Invalid or unsafe model archive: {exc}") from exc
    finally:
        # The install copies what it keeps into the bundle dir, so the staged archive is
        # never needed again — on success or on any failure.
        archive_path.unlink(missing_ok=True)
    return {"status": "imported", **info}


@router.post("/models/{model_name}/evaluate", status_code=202,
             summary="Evaluate a model on a dataset (asynchronous)",
             response_model=TrainStartedResponse)
@limiter.limit(train_limit)
async def evaluate_model(
    request: Request,
    model_name: ModelName,
    body: EvaluateRequest,
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Score an existing model against a dataset and record the result in its bundle.

    **"Model B beats model A" is only a statement if both were measured on the same
    rows.** Until now that meant a script driving a running server; this records the
    answer where it can be found again — appended to the bundle's `evaluations`, never
    over the training metrics, which describe the run that produced the model.

    The scored text is assembled the way the model was trained: without
    `text_column_weights` the run takes the bundle's own weights, narrowed to
    `text_columns`, and the recorded result says which ones it used. An explicit `{}`
    scores every column once.

    The model's own label space decides what can be scored: labels the model never
    learned are reported (`unknown_labels`), and rows carrying only such labels are
    excluded and counted (`rows_without_a_known_label`) rather than scored as failures
    — blaming a model for a label it was never given is not a number to compare on.

    Rows an LLM wrote or touched (data-prep's marks) are never scored: measured on them,
    a model is measured on how well it learned that LLM. They are skipped and counted in
    `ai_marked_rows_skipped` (recorded only when rows were skipped); a dataset of nothing
    but such rows fails the job.

    Runs as a background job on the same single worker as training, so it queues behind
    a running one exactly the same way; watch it on `/train/status` and find the outcome
    in `/train/history` with `kind: "evaluation"`. **Auth:** admin.
    """
    safe_name(model_name, "model name")
    # The name is joined onto data_dir below; without this it escapes the directory and
    # the server reads whatever it is pointed at. /train guards its dataset name the
    # same way — this route was the one that did not.
    safe_name(body.dataset_name, "dataset name")
    try:
        dataset = data_mod.resolve_dataset(settings.data_dir, body.dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Dataset '{body.dataset_name}' not found.") from exc
    # The header only, before the job can queue, as /train does: the loader would refuse a
    # missing column anyway, but as a job error behind whatever runs first.
    try:
        await asyncio.to_thread(require_columns, dataset, body.text_columns, body.label_column,
                                separator=body.csv_separator)
    except TrainingInputError as exc:
        raise HTTPException(400, str(exc)) from exc
    registry = get_registry()
    if not registry.exists(model_name):
        raise HTTPException(404, f"Model '{model_name}' not found.")

    req = {"model_name": model_name, **body.model_dump()}
    return start_or_queue(run_evaluation, req, settings, registry,
                          model_name=model_name, request=req, profile="evaluation",
                          kind="evaluation")
