"""Dataset management: list, inspect, analyze, validate, import/export, delete."""

from __future__ import annotations

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
from ..schemas import AnalyzeRequest, ExportRequest
from ..security import read_upload_capped, require_role, safe_name
from ..settings import Settings, get_settings
from ..sharing import get_share_store

router = APIRouter(tags=["Datasets"])


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
    for csv in sorted(data_dir.glob("*.csv")):
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
    safe_name(dataset_name, "dataset name")
    if len(separator) != 1:
        # pandas treats a multi-char sep as a regex (python engine) -> ReDoS.
        raise HTTPException(400, "separator must be a single character.")
    path = settings.data_dir / dataset_name
    if not path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found.")
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
    safe_name(req.dataset_name, "dataset name")
    path = settings.data_dir / req.dataset_name
    if not path.exists():
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found.")
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
    text_columns: list[str],
    label_column: str,
    separator: str = ";",
    label_separator: str = ",",
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Check that the given columns exist and report data-quality warnings
    (missing columns, rows without a label, very rare labels). **Auth:** admin."""
    safe_name(dataset_name, "dataset name")
    path = settings.data_dir / dataset_name
    if not path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found.")
    return stats_mod.validate_dataset(
        path, text_columns, label_column, separator=separator, label_separator=label_separator
    )


@router.post("/datasets/import", summary="Import a dataset (file upload only)")
@limiter.limit(export_limit)
async def import_dataset(
    request: Request,
    file: UploadFile = File(...),
    new_name: str | None = Form(None),
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Upload a CSV file into the data directory (no URL fetch -> no SSRF).

    `new_name` overrides the file name (`.csv` is appended). Size limit active;
    existing names are rejected with 409. **Auth:** admin.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a .csv file.")
    name = new_name or file.filename
    if not name.endswith(".csv"):
        name += ".csv"
    safe_name(name, "dataset name")
    target = settings.data_dir / name
    if target.exists():
        raise HTTPException(409, f"Dataset '{name}' already exists.")
    data = await read_upload_capped(file, settings.max_upload_mb * 1024 * 1024)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"status": "imported", "dataset_name": name, "size_bytes": len(data)}


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
    safe_name(dataset_name, "dataset name")
    path = settings.data_dir / dataset_name
    if not path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found.")
    body = body or ExportRequest()
    if body.generate_share_url:
        share_id, expires_at = get_share_store().create("dataset", dataset_name, body.expires_hours)
        return {"share_url": f"/share/{share_id}", "share_id": share_id, "expires_at": expires_at}
    return FileResponse(path, filename=dataset_name, media_type="text/csv")


@router.delete("/datasets/{dataset_name}", summary="Delete a dataset")
@limiter.limit(default_limit)
async def delete_dataset(
    request: Request, dataset_name: str, _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Delete a CSV file from the data directory (irreversible).
    **Auth:** admin · rate limit active."""
    safe_name(dataset_name, "dataset name")
    path = settings.data_dir / dataset_name
    if not path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found.")
    path.unlink()
    return {"status": "deleted", "dataset_name": dataset_name}
