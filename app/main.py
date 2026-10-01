"""FastAPI application factory for MetaClassify (metadata text-classification API)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from . import __version__, correlation, middleware
from .correlation import RequestIdFilter
from .docs_content import DESCRIPTION, FAVICON_SVG, SWAGGER_HTML, TAGS_METADATA
from .lifecycle import lifespan
from .limiter import limiter
from .log_filters import RedactShareTokens
from .routes import datasets, feedback, metadata, models, predict, predict_bulk, share, system, training
from .settings import get_settings

# Boot-time, but deliberately NOT in `lifecycle`: this has to run at import, before anything
# else logs, and `lifecycle` is imported above — putting it there would run it earlier still.
# `settings.log_level` is a Literal for the sake of this line, so a typo fails as a named
# pydantic error instead of a stdlib "Unknown level" crash loop.
logging.basicConfig(
    level=get_settings().log_level.upper(),
    # The request id is in every line so a sanitized 500 body can be traced to the
    # traceback that explains it (audit OPS-8). `-` outside a request.
    format="%(asctime)s %(name)s %(levelname)s [%(request_id)s] %(message)s",
)
# On the root handlers, not on one logger: uvicorn's access and error lines go through
# their own loggers, and correlating only our own would still leave the two halves apart.
for _handler in logging.getLogger().handlers:
    _handler.addFilter(RequestIdFilter())
    # Share ids are bearer capabilities and uvicorn's access log writes the URL, so the
    # capability was in the log for as long as it was valid (audit SEC-9).
    _handler.addFilter(RedactShareTokens())
logger = logging.getLogger("api_v3")


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback server-side and return a sanitized 500 body so internal
    details (exception text, paths) never leak to clients.

    The body carries the request id the traceback was logged under: sanitizing is right, and
    without it the operator has a report and no way to find the matching line (audit OPS-8).
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    request_id = correlation.of_request(request)
    # The header is set here rather than by the middleware because this response never
    # passes through it: ServerErrorMiddleware is outside the whole stack.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Emit the same ``{"detail": ...}`` envelope as every other error (slowapi's
    default handler uses a divergent ``{"error": ...}`` key).

    With ``Retry-After``, which slowapi omits unless ``headers_enabled`` is set: without it
    every client has to guess, and the usual guess is an immediate retry (audit API-6). The
    window is parsed off the limit slowapi already resolved, so the number is the real one
    rather than a constant that drifts from the configuration.
    """
    # slowapi's RateLimitExceeded carries the Limit it tripped, whose ``.limit`` is the
    # ``limits`` RateLimitItem; ``get_expiry()`` is that library's window in seconds
    # (verified: 30/minute -> 60, 5/second -> 1). The fallback covers slowapi changing
    # shape rather than asserting a private structure holds forever.
    item = getattr(getattr(exc, "limit", None), "limit", None)
    expiry = getattr(item, "get_expiry", None)
    seconds = int(expiry()) if callable(expiry) else 60
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": str(seconds)},
    )


async def _validation_error_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    """Report a 422 with ``detail`` as a string, like every other error in this API.

    FastAPI's default handler makes ``detail`` a list of objects on this status alone, so
    client code written against any other 4xx — ``body["detail"].startswith(...)`` — raises
    on exactly the status a client hits most while integrating (audit API-7). The per-field
    information is what a form needs, so it is kept under ``errors`` rather than dropped.

    Without the ``input`` each error carries: echoed, a text over the cap came back whole
    (10.4 MB in, 10.4 MB out), and a value pydantic refuses because no response can carry
    it -- a lone surrogate has no UTF-8 form, and the JSON encoder refuses NaN -- turned
    the 422 into a 500 (audit 2026-09-30, V05). Field, rule and message say what to fix.
    """
    errors = [{key: value for key, value in error.items() if key != "input"} for error in exc.errors()]
    where = ".".join(str(part) for part in errors[0].get("loc", ())) if errors else "request"
    first = errors[0].get("msg", "invalid") if errors else "invalid"
    detail = f"Validation error at {where}: {first}"
    if len(errors) > 1:
        detail += f" (and {len(errors) - 1} more)"
    return JSONResponse(
        status_code=422,
        # jsonable_encoder: an error's `ctx` can hold values json.dumps refuses (the
        # ValueError a validator raised).
        content={"detail": detail, "errors": jsonable_encoder(errors)},
    )


def create_app() -> FastAPI:
    """Build and configure the FastAPI app."""
    settings = get_settings()
    # docs_url/redoc_url disabled: FastAPI's built-in pages load their bundle from
    # a CDN plus an inline init script. We serve Swagger ourselves (vendored assets
    # + external init) at /docs so it is same-origin, needs no CDN, and runs under
    # CSP. ReDoc (also CDN-backed, redundant with Swagger) stays off.
    app = FastAPI(
        title="MetaClassify",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_tags=TAGS_METADATA,
        docs_url=None,
        redoc_url=None,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    middleware.install(app, settings)

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
            # How many rows a /predict/csv answer must cover: a page on another origin
            # cannot tell a stream cut short from a finished one without it.
            expose_headers=["X-Input-Rows"],
        )

    # Self-hosted API docs (replaces the disabled CDN-backed /docs).
    app.mount(
        "/swagger-static",
        StaticFiles(directory=Path(__file__).parent / "static" / "swagger"),
        name="swagger-static",
    )

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui() -> HTMLResponse:
        return HTMLResponse(SWAGGER_HTML)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(FAVICON_SVG, media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Send a human who opened the bare host somewhere useful (404 was the
        first thing the app said to anyone typing the URL)."""
        return RedirectResponse("/ui/" if settings.ui_enabled else "/docs")

    app.include_router(system.router)
    app.include_router(training.router)
    app.include_router(predict.router)
    app.include_router(predict_bulk.router)
    app.include_router(metadata.router)
    app.include_router(models.router)
    app.include_router(share.router)
    app.include_router(datasets.router)
    app.include_router(feedback.router)

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
