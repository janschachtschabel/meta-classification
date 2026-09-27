"""Dataset management: list, inspect, analyze, validate, import/export, delete."""

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
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi import (
    Path as PathParam,
)
from fastapi.responses import FileResponse

from .. import data as data_mod
from .. import dataset_stats as stats_mod
from ..capacity import CapacityPlan
from ..errors import TrainingInputError
from ..limiter import default_limit, export_limit, limiter
from ..profiles import load_training_config
from ..schemas import AnalyzeRequest, ExportRequest, ValidateRequest
from ..security import require_role, safe_name, spool_upload_capped
from ..settings import Settings, get_settings
from ..sharing import get_share_store
from ._paging import Limit, Offset, page

router = APIRouter(tags=["Datasets"])

# The name every dataset route takes in its path, described once for all of them.
DatasetName = Annotated[str, PathParam(description=(
    "File name of a dataset in the data directory, as `GET /datasets` lists it: `.csv` or "
    "`.csv.gz`, no path characters (400 otherwise). A name no dataset has answers 404."))]


def _dataset_path(dataset_name: str, settings: Settings) -> Path:
    """Resolve a dataset name to an existing dataset file, or raise 404.

    The data directory holds more than datasets — ``label_names.json`` is the
    authoritative display-name sidecar every training reads. Only the listing
    filtered on the suffixes, so inspect/export/share/delete reached any file a
    safe name could name. Checking membership here keeps every dataset route
    honest and gives them one shared 404.
    """
    safe_name(dataset_name, "dataset name")
    try:
        return data_mod.resolve_dataset(settings.data_dir, dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found.") from exc


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
    limit: Limit = None, offset: Offset = 0,
    _: str = Depends(require_role("readonly")), settings: Settings = Depends(get_settings)
) -> list[dict]:
    """List all CSV files in the data directory with name, size and row count.

    Omitting `limit` returns every dataset, as before. Note that the row count reads each
    CSV in full, so a bound here is a real saving on a directory with many large files.
    **Auth:** readonly.
    """
    data_dir = settings.data_dir
    if not data_dir.exists():
        return []
    out = []
    # Both suffixes: pandas reads a gzipped CSV natively, and the WLO full exports are
    # 126-195 MB compressed against ~1.4 GB plain — so the compressed form is the normal one
    # for large datasets, not an exception.
    # Paged before the row counts are read, not after: each one reads a whole CSV, so
    # narrowing first is the difference between one file and all of them.
    for csv in page(sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.csv.gz")]), limit, offset):
        stat = csv.stat()
        out.append({"name": csv.name, "size_human": _format_size(stat.st_size),
                    "rows": data_mod.count_rows(csv)})
    return out


@router.get("/datasets/{dataset_name}", summary="Dataset: columns & sample rows")
@limiter.limit(export_limit)  # the row count reads the whole file -> throttle like the heavy endpoints
def dataset_info(
    request: Request,
    dataset_name: DatasetName,
    separator: str = Query(";", description="The CSV's field delimiter: exactly one character (else 400)."),
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
    rare labels, plus a threshold analysis (how many labels have >= N examples), the
    `min_samples_per_label` the automatic setting would choose for this size, and the
    estimated minutes per profile on this server. Helps to choose
    `min_samples_per_label`, a `label_filter` and a profile. **Auth:** admin.
    """
    path = _dataset_path(req.dataset_name, settings)
    # Planned before the analysis loads the file (see CapacityPlan.for_server).
    plan = CapacityPlan.for_server(settings, load_training_config(settings.config_file).profiles)
    try:
        return stats_mod.analyze_dataset(
            path, req.text_columns, req.label_column,
            separator=req.csv_separator, label_separator=req.label_separator, label_filter=req.label_filter,
            plan=plan,
        )
    except TrainingInputError as exc:
        # Crafted, safe message (e.g. wrong column name + available columns) -> 400.
        raise HTTPException(400, str(exc)) from exc


@router.post("/datasets/{dataset_name}/validate", summary="Validate a dataset for training")
@limiter.limit(export_limit)  # parses the whole CSV -> throttle like the other heavy endpoints
def validate(
    request: Request,
    dataset_name: DatasetName,
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
    file: UploadFile = File(
        ...,
        description="The dataset: a `.csv` or gzip-compressed `.csv.gz` file, within `max_upload_mb` (`GET /config`).",
    ),
    new_name: str | None = Form(
        None,
        description=(
            "Store the dataset under this file name instead of the uploaded one; `.csv` is "
            "appended when it ends in neither suffix. An existing name is refused (409)."
        ),
    ),
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
    staging = _staging_path(settings.data_dir, name)
    size = await spool_upload_capped(file, settings.max_upload_mb * 1024 * 1024, staging)
    try:
        await asyncio.to_thread(_publish_new_dataset, staging, target)
    except FileExistsError as exc:
        # Won the upload race and lost the name: the 409 above ran before the upload, so
        # this is the same answer, decided at the moment it can actually be decided.
        raise HTTPException(409, f"Dataset '{name}' already exists.") from exc
    return {"status": "imported", "dataset_name": name, "size_bytes": size}


def _staging_path(data_dir: Path, name: str) -> Path:
    """A staging file for one upload, beside the target and nameable by no route.

    Spool-then-rename is what makes the rename the moment a dataset exists, so a failure
    part-way leaves something no route can ask for rather than a truncated CSV that lists,
    inspects and trains as if it were complete. Two properties carry that:

    * a **leading dot**, which ``safe_name`` rejects, and a ``.part`` suffix, which the
      listing's CSV glob does not match and the startup sweep does clean;
    * a **unique** name. It used to be ``<name>.part``, i.e. the same file for every request
      importing one name — so two concurrent imports interleaved their bytes into whichever
      one reached the rename (audit SEC-12).
    """
    handle, staged = tempfile.mkstemp(prefix=".import-", suffix=".part", dir=data_dir)
    os.close(handle)
    return Path(staged)


def _publish_new_dataset(staging: Path, target: Path) -> None:
    """Move the staged upload into place, or refuse because the name was taken.

    ``os.replace`` is atomic but silently OVERWRITES, and the 409 check runs before the
    upload — so a name created during a long upload was replaced by it anyway. ``os.link``
    fails with ``FileExistsError`` instead, atomically, which is the answer this route
    already gives for a name that exists (audit SEC-12).

    :raises FileExistsError: the target appeared while the upload was running.
    """
    try:
        os.link(staging, target)
    except FileExistsError:
        staging.unlink(missing_ok=True)
        raise
    except OSError:
        # No hard links here (a filesystem or platform that refuses them). Fall back to the
        # atomic-but-overwriting rename after one more check: the race window shrinks from
        # the whole upload to these two statements, which is the best this can do without
        # links.
        if target.exists():
            staging.unlink(missing_ok=True)
            raise FileExistsError(target) from None
        try:
            os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        return
    staging.unlink(missing_ok=True)


@router.post("/datasets/{dataset_name}/export", summary="Export a dataset (download or share link)",
             response_model=None)
@limiter.limit(export_limit)
async def export_dataset(
    request: Request,
    dataset_name: DatasetName,
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
    request: Request, dataset_name: DatasetName, _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Delete a CSV file from the data directory (irreversible), and revoke its share
    links. **Auth:** admin · rate limit active."""
    # missing_ok: _dataset_path already answered 404 for a name that is not there, so
    # reaching here with the file gone means it went in between — a double click, not
    # a server fault.
    _dataset_path(dataset_name, settings).unlink(missing_ok=True)
    # Same reason as model delete: the next import under this name must not be
    # reachable through a link that was handed out for this file.
    get_share_store().revoke_for("dataset", dataset_name)
    return {"status": "deleted", "dataset_name": dataset_name}
