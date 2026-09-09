"""Training endpoints: profiles, start, status, stop."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import job_history
from ..jobs import job_runner
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
    """Available quality/effort profiles (`fast` / `auto` / `best`, cheapest first).

    Per profile: name, description, the `C` values tried, threshold tuning
    (`tune_threshold`, `threshold_per_label`) and the TF-IDF feature shape
    (`use_char`, effective `max_word_features` / `max_char_features`) — so it is
    visible what distinguishes the profiles. Profiles are defined in `config.yaml`
    and freely extensible.

    Also returns the training-config defaults a `/train` request inherits when it
    omits the field: `default_text_column_weights` and `default_min_samples_per_label`.
    **Auth:** readonly.
    """
    cfg = load_training_config(settings.config_file)
    return {
        "default_profile": cfg.default_profile,
        # Training-config defaults a /train request inherits when it omits the field.
        # Exposed so a client (the admin UI does) can pre-fill its form with what
        # would actually happen, instead of hard-coding a guess.
        "default_text_column_weights": cfg.text_column_weights,
        "default_min_samples_per_label": cfg.min_samples_per_label,
        "profiles": [
            {
                "name": p.name,
                "description": p.description,
                "c_grid": p.c_grid,
                "tune_threshold": p.tune_threshold,
                "threshold_per_label": p.threshold_per_label,
                "use_char": p.use_char,
                # The evaluation mode this profile implies (0 = holdout, >=2 = k-fold);
                # null means it defers to split.cv_folds. A request still overrides it.
                "cv_folds": p.cv_folds,
                "max_word_features": p.max_word_features or settings.tfidf_max_word_features,
                "max_char_features": (
                    (p.max_char_features or settings.tfidf_max_char_features) if p.use_char else None
                ),
            }
            for p in cfg.profiles.values()
        ],
    }


# 202, not 200: the response says a job was ACCEPTED and points at /train/status —
# the training itself has not happened yet. Callers that only check `< 300` are
# unaffected; a caller checking `== 200` sees the contract it should have read.
@router.post("/train", summary="Train a model (asynchronous)", status_code=202,
             response_model=TrainStartedResponse)
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
    - `optimize_parameters`: quality/effort profile `fast` | `auto` | `best`, cheapest
      first (controls character n-grams, the C grid and the evaluation mode — i.e. the
      training time). `fast` evaluates on a holdout split and therefore deploys a model
      fit on 85% of the rows; `auto` and `best` deploy on 100%.
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
        # str(KeyError) reprs its message (stray quotes) — use the message itself.
        raise HTTPException(400, str(exc.args[0])) from exc

    if not (settings.data_dir / body.dataset_name).exists():
        raise HTTPException(404, f"Dataset '{body.dataset_name}' not found.")
    if get_registry().exists(body.model_name):
        raise HTTPException(409, f"Model '{body.model_name}' already exists. Delete it or pick another name.")

    req = {key: getattr(body, key) for key in _REQ_KEYS}
    # Not in _REQ_KEYS: it is a nested model, and the pipeline stores plain JSON.
    # exclude_none keeps the bundle from claiming fields the caller left unset.
    req["info"] = body.info.model_dump(exclude_none=True) if body.info else None
    try:
        # The singleton registry is injected so the training save shares its disk
        # lock with every API-side registry operation.
        position = job_runner.submit(
                           run_training, req, settings, cfg, profile, get_registry(),
                           model_name=body.model_name,
                           # Everything but the documentation block: `info` is what the
                           # model says about itself, not a parameter of the run.
                           # The profile is not in _REQ_KEYS (it is resolved separately),
                           # yet it is the dimension two runs differ on most — recorded
                           # under the field name a caller would resend, holding the
                           # profile that was actually used rather than what was asked.
                           request={**{k: v for k, v in req.items() if k != "info"},
                                    "optimize_parameters": profile.name})
    except RuntimeError as exc:
        # A full queue, or this name already running/queued: both are the caller's to
        # resolve, and both are conflicts with what the server is already doing.
        raise HTTPException(409, str(exc)) from exc
    return {
        "status": "started" if position == 0 else "queued",
        "model_name": body.model_name,
        "profile": profile.name,
        "status_url": "/train/status",
        "queue_position": position,
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
    return job_runner.snapshot()


@router.post("/train/stop", summary="Stop the running training", response_model=TrainStopResponse)
async def stop(hard: bool = False, _: str = Depends(require_role("admin"))) -> dict:
    """Cancel the running training.

    `hard=false` (default): cooperative cancellation at the next checkpoint (between
    the C fits, or before the final training). `hard=true`: reset the status to `idle`
    immediately. **Auth:** admin.
    """
    job_runner.stop(hard=hard)
    return {"status": "idle" if hard else "stopping"}


@router.get("/train/history", summary="Outcomes of finished training runs")
async def history(limit: int = 50, _: str = Depends(require_role("readonly"))) -> list[dict]:
    """What every finished run left behind, newest first.

    Per run: the model name, how it ended, when and for how long, the request it was
    started with, and the headline scores (`f1_macro`, `f1_micro`, `n_labels`). A run
    that failed carries its `error` — and that is the case with no bundle to inspect
    afterwards, so this is the only place the reason survives.

    Not the full metrics: `per_label_f1` is one entry per label, and comparing two runs
    needs the numbers a comparison is made on. The bundle keeps the rest.

    Bounded to the newest 200 runs on disk. A run killed mid-flight leaves no entry —
    it never finished. **Auth:** readonly.
    """
    return job_history.recent(limit=max(1, min(limit, 200)))
