"""System endpoints: health check, Prometheus metrics, safe configuration."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from .. import __version__
from ..jobs import job_runner
from ..registry import get_registry
from ..responses import ConfigResponse, HealthResponse
from ..security import require_role
from ..settings import Settings, get_settings

router = APIRouter(tags=["System"])

_START = time.monotonic()  # process start (module import) for the uptime gauge


@router.get("/health", summary="Health check (public)", response_model=HealthResponse)
async def health() -> dict:
    """Public health check for load balancers and container probes (no auth)."""
    return {"status": "healthy", "version": __version__}


@router.get("/metrics", summary="Prometheus metrics (public)", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Operational gauges in the Prometheus text format (hand-rolled: a client
    library would be a new runtime dependency for six gauges).

    Public like /health: a standard ServiceMonitor cannot send custom auth
    headers, and the exposed values are operational counters only (no model
    names, paths or data).
    """
    snap = job_runner.snapshot()
    registry = get_registry()
    progress = float(snap["progress"] or 0)
    lines = [
        "# HELP apiv3_info Build information (value is always 1).",
        "# TYPE apiv3_info gauge",
        f'apiv3_info{{version="{__version__}"}} 1',
        "# HELP apiv3_uptime_seconds Seconds since process start.",
        "# TYPE apiv3_uptime_seconds gauge",
        f"apiv3_uptime_seconds {time.monotonic() - _START:.1f}",
        "# HELP apiv3_models_total Trained model bundles on disk.",
        "# TYPE apiv3_models_total gauge",
        f"apiv3_models_total {len(registry.list())}",
        "# HELP apiv3_models_in_memory Models resident in the LRU cache.",
        "# TYPE apiv3_models_in_memory gauge",
        f"apiv3_models_in_memory {registry.in_memory_count()}",
        "# HELP apiv3_training_running 1 while a training job runs, else 0.",
        "# TYPE apiv3_training_running gauge",
        f"apiv3_training_running {1 if snap['status'] == 'running' else 0}",
        "# HELP apiv3_training_progress Progress of the current/last training (0-100).",
        "# TYPE apiv3_training_progress gauge",
        f"apiv3_training_progress {progress:.0f}",
    ]
    return PlainTextResponse(
        "\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@router.get("/config", summary="Safe configuration overview", response_model=ConfigResponse)
async def config(
    _: str = Depends(require_role("readonly")),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Non-sensitive configuration including resource parameters (no API keys).

    Includes ``n_jobs`` (CPU cores), the TF-IDF feature caps (RAM lever),
    ``max_models_in_memory`` and the auth / rate-limit status. **Auth:** readonly.
    """
    return {
        "n_jobs": settings.n_jobs,
        "cpu_max_percent": settings.cpu_max_percent,
        "effective_n_jobs": settings.effective_n_jobs(),  # what training will actually use
        "tfidf_max_word_features": settings.tfidf_max_word_features,
        "tfidf_max_char_features": settings.tfidf_max_char_features,
        "max_models_in_memory": settings.max_models_in_memory,
        # What the cache actually holds: raised to fit the warmup list (see settings).
        "effective_max_models_in_memory": settings.effective_max_models_in_memory(),
        "warmup_models": settings.warmup_models_list,
        "auth_enabled": settings.auth_enabled,
        "rate_limit_enabled": settings.rate_limit_enabled,
        "max_upload_mb": settings.max_upload_mb,
    }
