"""What every HTTP response carries, and what never reaches a route.

Split out of `main.py`: these are cross-cutting HTTP concerns, registered once and then
invisible, which is a different reason to change than which routers the app mounts. CORS stays
in the factory — it is conditional on configuration and belongs with the assembly.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import correlation, telemetry
from .settings import Settings


def install(app: FastAPI, settings: Settings) -> None:
    """Register the body-size ceiling, the request counters, the security headers and the
    correlation id, in that order.

    Starlette runs middleware in REVERSE registration order, so the last registered is the
    outermost. Both orderings here are load-bearing: the headers wrap the body check, so an
    oversized-body 413 still carries them; and the correlation id wraps everything, so every
    response has an id — including that 413 and the 500 the error handler builds — and the
    id is bound before any other layer can log.
    """
    @app.middleware("http")
    async def refuse_oversized_bodies(request: Request, call_next):
        """Refuse a body larger than the upload cap on its DECLARED size, before it
        is read.

        The per-route caps run too late to bound what reaches disk: FastAPI resolves
        ``UploadFile`` during dependency injection, so Starlette has already streamed the
        whole part into a spooled temp file by the time a route body runs — and it
        enforces ``max_part_size`` only for non-file parts. Under the chart's read-only
        root filesystem that spill lands in an emptyDir on node ephemeral storage, and
        ``POST /predict/csv`` needs only a readonly key.

        Content-Length is client-supplied and a chunked body carries none, so this is a
        cheap ceiling rather than the whole answer — the streaming caps in ``security``
        stay where they are and remain the real enforcement.
        """
        declared = request.headers.get("content-length")
        if declared and declared.isdigit():
            limit = settings.max_upload_mb * 1024 * 1024
            if int(declared) > limit:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {settings.max_upload_mb} MB limit."},
                )
        return await call_next(request)

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        """Count every request by route template and status, and time it (audit OPS-9).

        The TEMPLATE, not the path: Starlette puts the matched route in the scope during
        routing, which happens inside ``call_next``, so it is read afterwards. Anything that
        matched nothing is one bucket — keyed on the path, a scanner could mint an unbounded
        number of permanent series in a process that never restarts.
        """
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        template = getattr(route, "path", None) or telemetry.UNMATCHED
        telemetry.record(request.method, template, response.status_code,
                         time.perf_counter() - started)
        return response

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

    # Registered last, so it is the OUTERMOST: every response gets an id, including the
    # 413 the body check short-circuits and the 500 the error handler builds — and the id
    # is bound before anything else runs, so every log line written while handling this
    # request carries it (audit OPS-8).
    @app.middleware("http")
    async def correlate(request: Request, call_next):
        request_id = correlation.accept(request.headers.get("X-Request-ID"))
        # On the scope as well as in the context: the 500 handler runs outside this stack
        # (see correlation.of_request) and reads it from there.
        request.state.request_id = request_id
        token = correlation.set_current(request_id)
        try:
            response = await call_next(request)
        finally:
            correlation.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

