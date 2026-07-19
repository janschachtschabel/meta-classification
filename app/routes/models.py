"""Model management: list, details, delete, export, import, share download."""

from __future__ import annotations

import asyncio
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

from ..jobs import training_job
from ..limiter import default_limit, export_limit, limiter
from ..registry import UnsafeModelError, get_registry
from ..schemas import ExportRequest
from ..security import read_upload_capped, require_role, safe_name
from ..settings import Settings, get_settings
from ..sharing import get_share_store

router = APIRouter(tags=["Models"])


def _zip_response(name: str, blob: bytes) -> Response:
    """A ZIP download response with an attachment filename."""
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
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
    return _zip_response(model_name, await asyncio.to_thread(registry.export_zip, model_name))


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
    status_busy = training_job.is_running() and training_job.snapshot().get("model_name") == name
    if status_busy or training_job.active_model_name() == name:
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
        blob = await asyncio.to_thread(registry.export_zip, info["name"])
        return _zip_response(info["name"], blob)
    dataset_path = settings.data_dir / info["name"]
    if not dataset_path.exists():
        raise HTTPException(404, "Dataset no longer exists.")
    return FileResponse(dataset_path, filename=info["name"], media_type="text/csv")
