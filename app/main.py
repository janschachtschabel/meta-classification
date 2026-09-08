"""FastAPI application factory for MetaClassify (metadata text-classification API)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from . import __version__
from .limiter import limiter
from .registry import Registry, get_registry
from .routes import datasets, models, predict, system, training
from .settings import get_settings

logging.basicConfig(
    level=get_settings().log_level.upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("api_v3")


def _warmup_models(registry: Registry, names: list[str]) -> None:
    """Preload configured models into the LRU cache and run one empty-text
    prediction, so the first real /predict pays no cold skops-load. Best-effort:
    a missing or unreadable model is logged and skipped, never fatal to startup."""
    for name in names:
        try:
            registry.get(name).baseline_proba()
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort; never fail startup
            logger.warning("Warmup skipped model %r: %r", name, exc)
        else:
            logger.info("Warmed up model %r", name)


def _check_auth_configuration(settings) -> None:
    """Refuse to start an authenticated deployment that nobody can administer.

    With auth on and no admin key, ``_role_for_key`` can never return "admin": every
    request 401s and no model can ever be trained, imported or deleted. That reads
    like a client-side key problem and has cost real debugging time, so fail here
    with the variable name instead. A readonly-only deployment is legitimate, so
    only the admin key is required.
    """
    if not settings.auth_enabled:
        return
    if not settings.api_key_admin:
        raise RuntimeError(
            "APIV3_AUTH_ENABLED is true but APIV3_API_KEY_ADMIN is not set — every "
            "request would be rejected with 401. Set the key, or run with "
            "APIV3_AUTH_ENABLED=false for local use."
        )
    if settings.api_key_admin == settings.api_key_readonly:
        logger.warning(
            "APIV3_API_KEY_ADMIN and APIV3_API_KEY_READONLY are identical — the "
            "readonly role grants full admin access."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create storage dirs on startup, sweep model staging dirs orphaned by a
    crashed/killed save (a hidden ``.name.tmp`` whose atomic rename never ran),
    and preload any configured warmup models."""
    settings = get_settings()
    _check_auth_configuration(settings)
    settings.ensure_dirs()
    registry = get_registry()
    swept = registry.sweep_stale_tmp()
    if swept:
        logger.warning("Swept %d orphaned model staging dir(s) from a previous crash", swept)
    if settings.warmup_models_list:
        _warmup_models(registry, settings.warmup_models_list)
    logger.info(
        "api_v3 ready (models_dir=%s, auth=%s)", settings.models_dir, settings.auth_enabled
    )
    yield
    logger.info("api_v3 shutting down")


_TAGS_METADATA = [
    {"name": "System", "description": "Health check and (safe) configuration."},
    {"name": "Training", "description": "Train models asynchronously and monitor progress."},
    {"name": "Prediction", "description": "Classify texts with a trained model."},
    {"name": "Models", "description": "List, inspect, delete, export/import and share models."},
    {"name": "Datasets", "description": "Manage, analyze and validate CSV datasets."},
]

_DESCRIPTION = (
    "**MetaClassify** — torch-free, CPU-only text-classification API. Train on "
    "your metadata, serve multiple models via REST.\n\n"
    "**Authentication:** `X-API-Key` header. Role *readonly* for classification "
    "and status, *admin* for training and management actions. `/health` is public.\n\n"
    "**Typical flow:** provide a dataset → `POST /train` → `GET /train/status` "
    "(phase, progress, estimated time remaining) → `POST /predict`. Multiple models "
    "coexist; select per request via `model_name`."
)


# Self-hosted Swagger UI page: vendored assets + an external init script (no inline
# JS), so it renders same-origin under CSP without any CDN. Assets live in
# static/swagger and are pinned (see static/swagger/README.md).
_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MetaClassify — API docs</title>
  <link rel="stylesheet" href="/swagger-static/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="/swagger-static/swagger-ui-bundle.js"></script>
  <script src="/swagger-static/swagger-init.js"></script>
</body>
</html>"""


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback server-side and return a sanitized 500 body so internal
    details (exception text, paths) never leak to clients."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Emit the same ``{"detail": ...}`` envelope as every other error (slowapi's
    default handler uses a divergent ``{"error": ...}`` key)."""
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


def create_app() -> FastAPI:
    """Build and configure the FastAPI app."""
    settings = get_settings()
    # docs_url/redoc_url disabled: FastAPI's built-in pages load their bundle from
    # a CDN plus an inline init script. We serve Swagger ourselves (vendored assets
    # + external init) at /docs so it is same-origin, needs no CDN, and runs under
    # CSP. ReDoc (also CDN-backed, redundant with Swagger) stays off.
    app = FastAPI(
        title="MetaClassify",
        description=_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_tags=_TAGS_METADATA,
        docs_url=None,
        redoc_url=None,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Baseline hardening headers on every response. No HSTS (TLS is
        terminated at the reverse proxy, which should set it)."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # Everything is same-origin now (the admin UI and the self-hosted Swagger
        # page both load only vendored assets, no inline JS), so CSP applies
        # everywhere. Swagger UI injects its own styles and inline SVG/data: icons
        # at runtime, so /docs alone needs style-src 'unsafe-inline' + img-src data:;
        # every other route keeps the strict same-origin policy.
        if request.url.path.startswith("/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; base-uri 'self'; form-action 'self'; "
                "frame-ancestors 'none'; object-src 'none'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; base-uri 'self'; form-action 'self'; "
                "frame-ancestors 'none'; object-src 'none'"
            )
        if request.url.path.startswith("/ui"):
            # Always revalidate UI assets (304 when unchanged): browsers otherwise
            # keep executing a stale app.js from the heuristic cache after updates.
            response.headers["Cache-Control"] = "no-cache"
        return response

    if settings.cors_origins_list:
        origins = settings.cors_origins_list
        # A wildcard origin with credentials lets any site make authenticated
        # cross-origin requests; drop credentials so "*" stays a safe public CORS.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials="*" not in origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Self-hosted API docs (replaces the disabled CDN-backed /docs).
    app.mount(
        "/swagger-static",
        StaticFiles(directory=Path(__file__).parent / "static" / "swagger"),
        name="swagger-static",
    )

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui() -> HTMLResponse:
        return HTMLResponse(_SWAGGER_HTML)

    app.include_router(system.router)
    app.include_router(training.router)
    app.include_router(predict.router)
    app.include_router(models.router)
    app.include_router(datasets.router)

    if settings.ui_enabled:
        # Optional admin UI: plain static files (no build step, no external
        # assets), same-origin so the browser sends X-API-Key without CORS.
        app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static" / "ui", html=True),
                  name="ui")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
