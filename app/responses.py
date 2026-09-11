"""Pydantic response models for the fixed-shape endpoints.

Only the endpoints whose response has a stable, always-present key set are modeled
here — it documents the contract in OpenAPI without risk. Endpoints with
conditional or dynamic keys (the predict results, ``/train/status`` snapshot,
dataset analyze/validate) deliberately keep ``dict`` returns: a ``response_model``
filters output to the model's fields, which would silently drop the keys those
responses omit or add situationally.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    """Non-sensitive configuration overview (never includes API keys)."""

    # "auto" or the configured number, as set; the effective_* fields are what a run uses.
    n_jobs: int | str
    cpu_max_percent: int
    effective_n_jobs: int
    train_memory_mb: int | str
    effective_train_memory_mb: int | None
    tfidf_max_word_features: int
    tfidf_max_char_features: int
    max_models_in_memory: int
    effective_max_models_in_memory: int
    warmup_models: list[str]
    auth_enabled: bool
    rate_limit_enabled: bool
    max_upload_mb: int


class TrainStartedResponse(BaseModel):
    status: str
    model_name: str
    profile: str
    status_url: str
    # 0 = running now; N = N runs are ahead of it. Always present rather than only when
    # queued: a caller that has to check whether a field exists before reading it will
    # eventually forget to.
    queue_position: int


class TrainStopResponse(BaseModel):
    status: str
