"""Training endpoints: profiles, start, status, stop."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..jobs import training_job
from ..limiter import limiter, train_limit
from ..profiles import load_training_config
from ..registry import get_registry
from ..responses import TrainStartedResponse, TrainStopResponse
from ..schemas import TrainRequest
from ..security import require_role, safe_name
from ..settings import Settings, get_settings
from ..training import run_training

router = APIRouter(tags=["Training"])

_REQ_KEYS = (
    "dataset_name", "model_name", "text_columns", "label_column",
    "csv_separator", "label_separator", "label_filter", "task_type",
    "min_samples_per_label", "cv_folds",
)


@router.get("/train/profiles", summary="List training profiles")
async def list_profiles(
    _: str = Depends(require_role("readonly")), settings: Settings = Depends(get_settings)
) -> dict:
    """Available quality/effort profiles (`fast` / `auto` / `thorough`).

    Per profile: name, description, the `C` values tried, threshold tuning
    (`tune_threshold`, `threshold_per_label`) and the TF-IDF feature shape
    (`use_char`, effective `max_word_features` / `max_char_features`) — so it is
    visible what distinguishes the profiles. Profiles are defined in `config.yaml`
    and freely extensible. **Auth:** readonly.
    """
    cfg = load_training_config(settings.config_file)
    return {
        "default_profile": cfg.default_profile,
        "profiles": [
            {
                "name": p.name,
                "description": p.description,
                "c_grid": p.c_grid,
                "tune_threshold": p.tune_threshold,
                "threshold_per_label": p.threshold_per_label,
                "use_char": p.use_char,
                "max_word_features": p.max_word_features or settings.tfidf_max_word_features,
                "max_char_features": (
                    (p.max_char_features or settings.tfidf_max_char_features) if p.use_char else None
                ),
            }
            for p in cfg.profiles.values()
        ],
    }


@router.post("/train", summary="Train a model (asynchronous)", response_model=TrainStartedResponse)
@limiter.limit(train_limit)
async def train(
    request: Request,
    body: TrainRequest,
    _: str = Depends(require_role("admin")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Start a training job in the background and return a job reference immediately.

    Poll progress (phase, percent, estimated time remaining) via `GET /train/status`;
    cancel via `POST /train/stop`.

    **Key body fields:**
    - `dataset_name`: a CSV in the data directory (upload beforehand via `POST /datasets/import`).
    - `text_columns`: columns merged into the input text.
    - `label_column`: the column holding the labels. An optional `<label>_DISPLAYNAME`
      column supplies human-readable names.
    - `optimize_parameters`: quality/effort profile `fast` | `auto` | `thorough`
      (controls the C grid + threshold tuning, and thus the training time).
    - `task_type`: optional override `multilabel` | `multiclass` | `binary`
      (default `auto` = automatic detection).
    - `label_filter`: optionally keep only labels containing this substring.
    - `min_samples_per_label`: minimum number of tagged items required for a label to be
      included (rarer labels are dropped). Empty/`null` = automatic (scales with the
      dataset size, ~20 for typical sets).
    - `cv_folds`: evaluation mode — `0` = classic train/val/test split, `>= 2` = k-fold
      cross-validation (every row trains and validates via out-of-fold metrics; the
      deployed model is fit on 100% of the data). `null` = config default (`split.cv_folds`).

    **Auto-optimization:** `C`, thresholds (global + per-label) and the task type are
    determined automatically; `min_samples_per_label` is auto-scaled unless set. All of
    it is stored in the model.

    **Errors:** 404 (dataset missing), 409 (model name already exists, or a training is
    already running). **Auth:** admin · rate limit active.
    """
    safe_name(body.dataset_name, "dataset name")
    safe_name(body.model_name, "model name")

    cfg = load_training_config(settings.config_file)
    try:
        profile = cfg.get(body.optimize_parameters)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not (settings.data_dir / body.dataset_name).exists():
        raise HTTPException(404, f"Dataset '{body.dataset_name}' not found.")
    if get_registry().exists(body.model_name):
        raise HTTPException(409, f"Model '{body.model_name}' already exists. Delete it or pick another name.")

    req = {key: getattr(body, key) for key in _REQ_KEYS}
    try:
        # The singleton registry is injected so the training save shares its disk
        # lock with every API-side registry operation.
        training_job.start(run_training, req, settings, cfg, profile, get_registry(),
                           model_name=body.model_name)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "status": "started",
        "model_name": body.model_name,
        "profile": profile.name,
        "status_url": "/train/status",
    }


@router.get("/train/status", summary="Current training status")
async def status(_: str = Depends(require_role("readonly"))) -> dict:
    """Live status of the running (or most recent) training.

    Fields: `status` (idle/running/completed/error/stopped), `phase`
    (loading → preparing → features → selecting → threshold → evaluating → saving → done),
    `phase_detail` (e.g. which `C` value is currently being tried), `progress` (0–100),
    `message`, `elapsed_seconds`, `eta_seconds` (estimated time remaining),
    `seconds_since_heartbeat` (age of the newest progress signal while running; it keeps
    growing when the training thread stalls silently — long values mean "possibly hung",
    while `elapsed_seconds` grows either way), `model_name`,
    `results` (metrics on completion), `error`. **Auth:** readonly.
    """
    return training_job.snapshot()


@router.post("/train/stop", summary="Stop the running training", response_model=TrainStopResponse)
async def stop(hard: bool = False, _: str = Depends(require_role("admin"))) -> dict:
    """Cancel the running training.

    `hard=false` (default): cooperative cancellation at the next checkpoint (between
    the C fits, or before the final training). `hard=true`: reset the status to `idle`
    immediately. **Auth:** admin.
    """
    training_job.stop(hard=hard)
    return {"status": "idle" if hard else "stopping"}
