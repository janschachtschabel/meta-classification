"""Dataset management: list, inspect, analyze, validate, import/export, delete."""

from __future__ import annotations

import asyncio
import os
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

from .. import data as data_mod
from .. import dataset_stats as stats_mod
from ..errors import TrainingInputError
from ..limiter import default_limit, export_limit, limiter
from ..schemas import AnalyzeRequest, ExportRequest, ValidateRequest
from ..security import require_role, safe_name, spool_upload_capped
from ..settings import Settings, get_settings
from ..sharing import get_share_store

router = APIRouter(tags=["Datasets"])


def _dataset_path(dataset_name: str, settings: Settings) -> Path:
    """Resolve a dataset name to an existing dataset file, or raise 404.

    The data directory holds more than datasets — ``label_names.json`` is the
    authoritative display-name sidecar every training reads. Only the listing
    filtered on the suffixes, so inspect/export/share/delete reached any file a
    safe name could name. Checking membership here keeps every dataset route
    honest and gives them one shared 404.
    """
    safe_name(dataset_name, "dataset name")
    path = settings.data_dir / dataset_name
    if not data_mod.is_dataset_name(dataset_name) or not path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found.")
    return path


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@router.get("/datasets", summary="List datasets")
@limiter.limit(export_limit)  # row counts read every CSV in full -> throttle like the heavy endpoints
def list_datasets(
    request: Request,
    _: str = Depends(require_role("readonly")), settings: Settings = Depends(get_settings)
) -> list[dict]:
    """List all CSV files in the data directory with name, size and row count.
    **Auth:** readonly."""
    data_dir = settings.data_dir
    if not data_dir.exists():
        return []
    out = []
    # Both suffixes: pandas reads a gzipped CSV natively, and the WLO full exports are
    # 126-195 MB compressed against ~1.4 GB plain — so the compressed form is the normal one
    # for large datasets, not an exception.
    for csv in sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.csv.gz")]):
        stat = csv.stat()
        out.append({"name": csv.name, "size_human": _format_size(stat.st_size),
                    "rows": data_mod.count_rows(csv)})
    return out


@router.get("/datasets/{dataset_name}", summary="Dataset: columns & sample rows")
@limiter.limit(export_limit)  # the row count reads the whole file -> throttle like the heavy endpoints
def dataset_info(
    request: Request,
    dataset_name: str,
    separator: str = ";",
    _: str = Depends(require_role("readonly")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Show column names, row count and the first few sample rows of a dataset.

    `separator` is the CSV delimiter (default `;`). **Auth:** readonly.
    """
    if len(separator) != 1:
        # pandas treats a multi-char sep as a regex (python engine) -> ReDoS.
        raise HTTPException(400, "separator must be a single character.")
    path = _dataset_path(dataset_name, settings)
    try:
        shaped = stats_mod.sample_rows(path, separator=separator)
    except TrainingInputError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": dataset_name, "rows": data_mod.count_rows(path), **shaped}


@router.post("/datasets/analyze", summary="Analyze label distribution & text quality")
@limiter.limit(export_limit)  # parses the whole CSV -> throttle like the other heavy endpoints
def analyze(request: Request, req: AnalyzeRequest, _: str = Depends(require_role("admin")),
            settings: Settings = Depends(get_settings)) -> dict:
    """Comprehensive statistics for a dataset to plan training.

    Returns sample/label counts, text lengths, labels per sample, the most frequent and
    rare labels, plus a threshold analysis (how many labels have >= N examples). Helps to
    choose `min_samples_per_label` and a `label_filter`. **Auth:** admin.
    """
    path = _dataset_path(req.dataset_name, settings)
    try:
        return stats_mod.analyze_dataset(
            path, req.text_columns, req.label_column,
            separator=req.csv_separator, label_separator=req.label_separator, label_filter=req.label_filter,
        )
    except TrainingInputError as exc:
        # Crafted, safe message (e.g. wrong column name + available columns) -> 400.
        raise HTTPException(400, str(exc)) from exc


@router.post("/datasets/{dataset_name}/validate", summary="Validate a dataset for training")
@limiter.limit(export_limit)  # parses the whole CSV -> throttle like the other heavy endpoints
def validate(
    request: Request,
    dataset_name: str,
    req: ValidateRequest,
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Check that the given columns exist and report data-quality warnings
    (missing columns, rows without a label, very rare labels).

    Takes ONE JSON object like `/datasets/analyze` (**breaking change**: the
    former raw-array body + query-parameter contract was dropped when the two
    endpoints were aligned). **Auth:** admin.
    """
    path = _dataset_path(dataset_name, settings)
    try:
        return stats_mod.validate_dataset(
            path, req.text_columns, req.label_column,
            separator=req.csv_separator, label_separator=req.label_separator,
        )
    except TrainingInputError as exc:
        # Same crafted-400 mapping as analyze/dataset_info (empty/malformed CSV).
        raise HTTPException(400, str(exc)) from exc


@router.post("/datasets/import", summary="Import a dataset (file upload only)")
@limiter.limit(export_limit)
async def import_dataset(
    request: Request,
    file: UploadFile = File(...),
    new_name: str | None = Form(None),
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Upload a CSV (optionally gzipped) into the data directory (no URL fetch -> no SSRF).

    Accepts `.csv` and `.csv.gz`; the compressed form is read natively everywhere and is the
    practical choice for large exports. `new_name` overrides the file name (`.csv` is
    appended if it carries neither suffix). Size limit active; existing names are rejected
    with 409. **Auth:** admin.
    """
    accepted = data_mod.DATASET_SUFFIXES
    if not file.filename or not file.filename.endswith(accepted):
        raise HTTPException(400, "File must be a .csv or .csv.gz file.")
    name = new_name or file.filename
    if not name.endswith(accepted):
        name += ".csv"
    safe_name(name, "dataset name")
    target = settings.data_dir / name
    if target.exists():
        raise HTTPException(409, f"Dataset '{name}' already exists.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Spool beside the target, then rename: the rename is what makes a dataset exist,
    # so a failure before it leaves a ".part" file that no route can name (the listing
    # globs the CSV suffixes and _dataset_path checks them) rather than a truncated CSV
    # that lists, inspects and trains as if it were complete.
    staging = target.with_name(target.name + ".part")
    size = await spool_upload_capped(file, settings.max_upload_mb * 1024 * 1024, staging)
    try:
        await asyncio.to_thread(os.replace, staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return {"status": "imported", "dataset_name": name, "size_bytes": size}


@router.post("/datasets/{dataset_name}/export", summary="Export a dataset (download or share link)",
             response_model=None)
@limiter.limit(export_limit)
async def export_dataset(
    request: Request,
    dataset_name: str,
    body: ExportRequest | None = Body(None),
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> Response | dict:
    """Export a dataset as a CSV download or as an expiring share link
    (`generate_share_url=true`, `expires_hours` 1–168, retrievable via `GET /share/{id}`).
    **Auth:** admin."""
    path = _dataset_path(dataset_name, settings)
    body = body or ExportRequest()
    if body.generate_share_url:
        share_id, expires_at = get_share_store().create("dataset", dataset_name, body.expires_hours)
        return {"share_url": f"/share/{share_id}", "share_id": share_id, "expires_at": expires_at}
    # Announce gzip as gzip: served as text/csv, a browser may transparently decompress and
    # save the file under its .gz name, leaving something that no longer opens.
    media_type = "application/gzip" if data_mod.is_gzipped(dataset_name) else "text/csv"
    return FileResponse(path, filename=dataset_name, media_type=media_type)


@router.delete("/datasets/{dataset_name}", summary="Delete a dataset")
@limiter.limit(default_limit)
async def delete_dataset(
    request: Request, dataset_name: str, _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Delete a CSV file from the data directory (irreversible).
    **Auth:** admin · rate limit active."""
    _dataset_path(dataset_name, settings).unlink()
    # Same reason as model delete: the next import under this name must not be
    # reachable through a link that was handed out for this file.
    get_share_store().revoke_for("dataset", dataset_name)
    return {"status": "deleted", "dataset_name": dataset_name}
