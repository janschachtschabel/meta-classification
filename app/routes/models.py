"""Model management: list, details, evaluate, delete, export, import, share download."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

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
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import data as data_mod
from ..evaluate import run_evaluation
from ..jobs import job_runner
from ..limiter import default_limit, export_limit, limiter, train_limit
from ..registry import UnsafeModelError, get_registry
from ..responses import TrainStartedResponse
from ..schemas import EvaluateRequest, ExportRequest, ModelInfo
from ..security import read_upload_capped, require_role, safe_name
from ..settings import Settings, get_settings
from ..sharing import get_share_store

router = APIRouter(tags=["Models"])


def _staged_zip_response(name: str) -> FileResponse:
    """Pack the bundle into a staging file and stream that file back.

    The archive is not built in memory: a production bundle is 50-180 MB and the byte
    path peaked at 2.78x that (measured). Staging goes next to the bundles rather than
    into the system temp — on a container /tmp is often tmpfs, i.e. RAM, which would
    give back exactly what this removes — and carries the same hidden ".*.tmp" name the
    startup sweep already cleans, so a download that dies mid-flight leaks nothing
    permanently. The response deletes it once the body is sent.
    """
    registry = get_registry()
    registry.dir.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(prefix=".export-", suffix=".zip.tmp", dir=registry.dir)
    os.close(handle)
    path = Path(staged)
    try:
        with path.open("wb") as stream:
            registry.export_to(name, stream)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(
        path, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


@router.get("/models", summary="List all models")
async def list_models(_: str = Depends(require_role("readonly"))) -> list[str]:
    """Names of all available models (bundles in the model directory). **Auth:** readonly."""
    return get_registry().list()


@router.get("/models/{model_name}", summary="Model details & metrics")
async def model_info(model_name: str, _: str = Depends(require_role("readonly"))) -> dict:
    """Configuration + training metadata/metrics of a model.

    Reads only the JSON metadata (weights are not loaded). Includes the backend,
    classes, thresholds, task type and the test metrics. **Auth:** readonly.
    """
    safe_name(model_name, "model name")
    try:
        return get_registry().info(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc


@router.get("/models/{model_name}/labels", summary="Per-label diagnostics")
async def model_labels(model_name: str, _: str = Depends(require_role("readonly"))) -> list[dict]:
    """Every label with its F1, its support (rows carrying it) and its threshold,
    **weakest first**.

    The headline F1 says how good a model is on average; this says where it is weak,
    which is what decides whether one answer deserves a second look. `support` makes
    a score readable — 0.13 on 25 rows is a different statement from 0.13 on 5,000.
    `threshold` is `null` for binary/multiclass, where serving decides by argmax and
    reads no threshold. **Auth:** readonly.
    """
    safe_name(model_name, "model name")
    try:
        return get_registry().label_diagnostics(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc


@router.put("/models/{model_name}/info", summary="Set the model's documentation")
@limiter.limit(default_limit)
async def set_model_info(
    request: Request,
    model_name: str,
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
async def delete_model(request: Request, model_name: str, _: str = Depends(require_role("admin"))) -> dict:
    """Remove a model from the in-memory cache and from disk (irreversible).
    **Auth:** admin · rate limit active."""
    safe_name(model_name, "model name")
    try:
        # rmtree of a large bundle is blocking disk work — off the event loop.
        await asyncio.to_thread(get_registry().delete, model_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    return {"status": "deleted", "model_name": model_name}


@router.post("/models/{model_name}/export", summary="Export a model (download or share link)",
             response_model=None)
@limiter.limit(export_limit)
async def export_model(
    request: Request,
    model_name: str,
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
    return await asyncio.to_thread(_staged_zip_response, model_name)


@router.post("/models/import", summary="Import a model (file upload only)")
@limiter.limit(export_limit)
async def import_model(
    request: Request,
    file: UploadFile = File(..., description="A model bundle (.zip) exported by this API"),
    new_name: str | None = Form(None),
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Import a model bundle from an uploaded ZIP (no URL fetch -> no SSRF).

    The archive is validated (no path-traversal names, required files present) and
    loaded safely via skops — files with unknown/unsafe types are rejected.
    `new_name` overrides the model name. Size limit active. **Auth:** admin.
    """
    if not file.filename or not file.filename.endswith(".zip"):
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
    data = await read_upload_capped(file, settings.max_upload_mb * 1024 * 1024)
    try:
        # Validation loads both skops files — seconds of CPU; off the event loop.
        info = await asyncio.to_thread(get_registry().import_zip, name, data)
    except FileExistsError as exc:
        raise HTTPException(409, f"Model '{name}' already exists.") from exc
    except (UnsafeModelError, ValueError) as exc:
        raise HTTPException(400, f"Invalid or unsafe model archive: {exc}") from exc
    return {"status": "imported", **info}


@router.get("/share", summary="List active share links")
async def list_share_links(_: str = Depends(require_role("admin"))) -> list[dict]:
    """Every share link that has not expired: id, kind, name, created and expiry.

    A share link is a bearer capability — the id IS the authorization — so this
    listing hands out the secrets themselves and stays admin-only, unlike the
    download route it describes. **Auth:** admin.
    """
    return get_share_store().list()


@router.delete("/share/{share_id}", summary="Revoke a share link")
@limiter.limit(default_limit)
async def revoke_share_link(
    request: Request, share_id: str, _: str = Depends(require_role("admin")),
) -> dict:
    """Withdraw a share link before it expires.

    What was already downloaded cannot be recalled, but the link stops working — the
    point of an expiring capability you can end early. **Auth:** admin · rate limit
    active.
    """
    if not get_share_store().revoke(share_id):
        raise HTTPException(404, "Share link not found or already expired.")
    return {"status": "revoked", "share_id": share_id}


@router.get("/share/{share_id}", summary="Download a shared resource")
@limiter.limit(export_limit)  # public endpoint: throttle share-id brute-forcing
async def download_shared(
    request: Request,
    share_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Download a previously exported share resource (model ZIP or dataset CSV).

    The share link is a **bearer capability**: the unguessable id (`secrets.token_urlsafe`,
    96 bits) plus its expiry ARE the authorization, so no API key is required — a link
    can be handed to someone without a key. Creating links stays admin-only; guard the
    id like a secret. Expired or unknown links return 404. **Auth:** none (bearer link).
    """
    info = get_share_store().resolve(share_id)
    if info is None:
        raise HTTPException(404, "Share link not found or expired.")
    if info["kind"] == "model":
        registry = get_registry()
        if not registry.exists(info["name"]):
            raise HTTPException(404, "Model no longer exists.")
        # Same blocking-zip offload as the authenticated export route.
        return await asyncio.to_thread(_staged_zip_response, info["name"])
    dataset_path = settings.data_dir / info["name"]
    if not dataset_path.exists():
        raise HTTPException(404, "Dataset no longer exists.")
    # Same gzip/CSV distinction as the authenticated export route: a share link is the path a
    # recipient WITHOUT a key uses, so it is the one most likely opened in a browser — where
    # a text/csv header on gzip bytes yields a decompressed file saved under its .gz name.
    media_type = "application/gzip" if data_mod.is_gzipped(info["name"]) else "text/csv"
    return FileResponse(dataset_path, filename=info["name"], media_type=media_type)


@router.post("/models/{model_name}/evaluate", status_code=202,
             summary="Evaluate a model on a dataset (asynchronous)",
             response_model=TrainStartedResponse)
@limiter.limit(train_limit)
async def evaluate_model(
    request: Request,
    model_name: str,
    body: EvaluateRequest,
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Score an existing model against a dataset and record the result in its bundle.

    **"Model B beats model A" is only a statement if both were measured on the same
    rows.** Until now that meant a script driving a running server; this records the
    answer where it can be found again — appended to the bundle's `evaluations`, never
    over the training metrics, which describe the run that produced the model.

    The model's own label space decides what can be scored: labels the model never
    learned are reported (`unknown_labels`), and rows carrying only such labels are
    excluded and counted (`rows_without_a_known_label`) rather than scored as failures
    — blaming a model for a label it was never given is not a number to compare on.

    Runs as a background job on the same single worker as training, so it queues behind
    a running one exactly the same way; watch it on `/train/status` and find the outcome
    in `/train/history` with `kind: "evaluation"`. **Auth:** admin.
    """
    safe_name(model_name, "model name")
    registry = get_registry()
    if not registry.exists(model_name):
        raise HTTPException(404, f"Model '{model_name}' not found.")
    if not (settings.data_dir / body.dataset_name).exists():
        raise HTTPException(404, f"Dataset '{body.dataset_name}' not found.")

    req = {"model_name": model_name, **body.model_dump()}
    try:
        position = job_runner.submit(
            run_evaluation, req, settings, registry,
            model_name=model_name, request=req, kind="evaluation",
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "status": "started" if position == 0 else "queued",
        "model_name": model_name,
        "profile": "evaluation",
        "status_url": "/train/status",
        "queue_position": position,
    }
