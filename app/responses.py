"""Pydantic response models for the fixed-shape endpoints.

Only the endpoints whose response has a stable, always-present key set are modeled
here — it documents the contract in OpenAPI without risk. Endpoints with
conditional or dynamic keys (the predict results, ``/train/status`` snapshot,
dataset analyze/validate) deliberately keep ``dict`` returns: a ``response_model``
filters output to the model's fields, which would silently drop the keys those
responses omit or add situationally.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Always `healthy` while the process answers requests.")
    version: str = Field(..., description="Version of the running MetaClassify build.")


class ReadyResponse(BaseModel):
    status: str = Field(..., description="`ready` — the storage this process serves from is usable.")
    version: str = Field(..., description="Version of the running MetaClassify build.")


class ConfigResponse(BaseModel):
    """Non-sensitive configuration overview (never includes API keys)."""

    # "auto" or the configured number, as set; the effective_* fields are what a run uses.
    n_jobs: int | str = Field(
        ...,
        description=(
            "Head-fit threads as configured (`APIV3_N_JOBS`): `auto` = every core this process "
            "may use; a negative number counts back from all cores (-1 = all, -2 = all but one)."
        ),
    )
    cpu_max_percent: int = Field(
        ...,
        description=(
            "CPU budget of a training run, in percent of the cores this process may use "
            "(`APIV3_CPU_MAX_PERCENT`; 100 = no cap)."
        ),
    )
    effective_n_jobs: int = Field(
        ...,
        description=(
            "Threads a training's head fits start from: `n_jobs`, capped by `cpu_max_percent` of "
            "the cores this process may use (a container's CPU quota counts). The memory budget "
            "can lower it per fit."
        ),
    )
    train_memory_mb: int | str = Field(
        ...,
        description="Training memory budget as configured (`APIV3_TRAIN_MEMORY_MB`): `auto`, or MiB (0 = no cap).",
    )
    effective_train_memory_mb: int | None = Field(
        ...,
        description=(
            "The budget in force, in MiB: the configured value, or for `auto` 85% of the "
            "container's memory limit; null = no cap. A run that would outgrow it trains with "
            "fewer head-fit threads — slower, never a different model."
        ),
    )
    training_isolation: str = Field(
        ...,
        description=(
            "Where a training runs (`APIV3_TRAINING_ISOLATION`). `process`: in a child process, "
            "so its memory goes back to the OS when it ends and an out-of-memory kill ends the "
            "run, not the API. `thread`: inside the API process."
        ),
    )
    tfidf_max_word_features: int = Field(
        ...,
        description=(
            "Server default for the word-n-gram vocabulary cap (`APIV3_TFIDF_MAX_WORD_FEATURES`); "
            "a profile's or a request's `max_word_features` overrides it."
        ),
    )
    tfidf_max_char_features: int = Field(
        ...,
        description=(
            "Server default for the character-n-gram vocabulary cap "
            "(`APIV3_TFIDF_MAX_CHAR_FEATURES`); a profile's or a request's `max_char_features` "
            "overrides it."
        ),
    )
    max_models_in_memory: int = Field(
        ...,
        description=(
            "Configured model-cache size (`APIV3_MAX_MODELS_IN_MEMORY`): beyond it the least "
            "recently used model is evicted and loaded from disk again on its next use."
        ),
    )
    effective_max_models_in_memory: int = Field(
        ...,
        description="Models the cache actually holds: `max_models_in_memory`, raised to fit every warmup model.",
    )
    warmup_models: list[str] = Field(
        ...,
        description=(
            "Models loaded at startup (`APIV3_WARMUP_MODELS`), so their first prediction pays no "
            "cold load; one that cannot be loaded is skipped and logged."
        ),
    )
    auth_enabled: bool = Field(
        ...,
        description="Whether requests need an `X-API-Key`; false = every request has admin rights (local use only).",
    )
    rate_limit_enabled: bool = Field(
        ...,
        description=(
            "Whether the rate limits apply (`APIV3_RATE_LIMIT_ENABLED`); they count per client "
            "address, and a request over its limit gets 429."
        ),
    )
    max_upload_mb: int = Field(
        ...,
        description=(
            "Largest accepted upload in MiB (`APIV3_MAX_UPLOAD_MB`) — a dataset, a model archive "
            "or a CSV to classify; a bigger one is refused with 413."
        ),
    )


class TrainStartedResponse(BaseModel):
    status: str = Field(
        ..., description="`started` when the run began at once, `queued` when it waits behind another.",
    )
    model_name: str = Field(
        ..., description="The model the run trains — or, for an evaluation, the model it scores.",
    )
    profile: str = Field(
        ...,
        description=(
            "The profile the run uses (`fast`, `auto`, `best` or a custom one from config.yaml); "
            "`evaluation` for an evaluation run."
        ),
    )
    status_url: str = Field(
        ..., description="Where to follow the run: `/train/status`. Its outcome then appears in `/train/history`.",
    )
    # 0 = running now; N = N runs are ahead of it. Always present rather than only when
    # queued: a caller that has to check whether a field exists before reading it will
    # eventually forget to.
    queue_position: int = Field(..., description="0 = running now; N = N runs are ahead of it in the queue.")


class TrainStopResponse(BaseModel):
    status: str = Field(
        ...,
        description=(
            "`stopping` (hard=false): the stop is requested and the run ends at its next "
            "checkpoint — `/train/status` says when. `idle` (hard=true): a running run's status "
            "was reset at once."
        ),
    )
