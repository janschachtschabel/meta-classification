"""What every HTTP response carries, and what never reaches a route.

Split out of `main.py`: these are cross-cutting HTTP concerns, registered once and then
invisible, which is a different reason to change than which routers the app mounts. CORS stays
in the factory — it is conditional on configuration and belongs with the assembly.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import correlation, telemetry
from .security import authenticate
from .settings import Settings

MiB = 1024 * 1024


class BodyGuard:
    """No request body is read before its caller is known, nor past its limit.

    FastAPI parses a body -- JSON into objects, multipart into spooled temp files -- before
    any dependency runs, and ``require_role`` is a dependency. So a caller without a key made
    the server read and parse whatever it sent, up to the upload cap: 48 MiB of `[{},...]`
    took the process from 232 to 1,485 MiB for a 401, and a chunked upload -- no
    Content-Length, which was all the old ceiling looked at -- landed whole in a temp file
    before its 413 (audit 2026-09-30, S01). Under the chart that temp file is node ephemeral
    storage, and a full one evicts the pod with its running training.

    So a request that carries a body -- declared, or chunked -- must first name a caller
    ``security.authenticate`` accepts. The ROLE stays the route's to decide: a readonly key can
    make an admin upload route read its body before the 403, within the caps and the rate
    limit. Then the body is held to its limit: ``max_upload_mb`` for a multipart upload,
    ``max_json_mb`` for anything else, on the declared size before a byte is read, and on
    the bytes as they arrive, which is what bounds a chunked body.

    Pure ASGI rather than ``@app.middleware`` like the others: only here can the body be seen
    arriving, piece by piece, instead of after the route has read it.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if not headers.get("transfer-encoding") and (declared or "0") == "0":
            await self.app(scope, receive, send)
            return
        try:
            authenticate(Request(scope), headers.get("x-api-key"), self.settings)
        except HTTPException as refusal:
            await _refusal(refusal.status_code, refusal.detail, refusal.headers)(scope, receive, send)
            return
        upload = headers.get("content-type", "").lower().startswith("multipart/form-data")
        limit_mb = self.settings.max_upload_mb if upload else self.settings.max_json_mb
        too_large = f"Request body exceeds the {limit_mb} MB limit{'' if upload else ' for a JSON body'}."
        if declared and declared.isdigit() and int(declared) > limit_mb * MiB:
            await _refusal(413, too_large)(scope, receive, send)
            return
        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit_mb * MiB:
                    # Raised into whoever is reading, before a route runs: FastAPI lets an
                    # HTTPException from the body read through (it wraps everything else into
                    # a 400), and Starlette's multipart parser closes its temp files first.
                    raise HTTPException(status_code=413, detail=too_large)
            return message

        await self.app(scope, counted, send)


def _refusal(status: int, detail: str, headers: Mapping[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail}, headers=headers)


def hardening_headers(path: str) -> dict[str, str]:
    """The baseline hardening headers for a response to ``path``. No HSTS (TLS is terminated
    at the reverse proxy, which should set it).

    A function as well as a middleware: the 500 for an unhandled error is built OUTSIDE the
    middleware stack (Starlette's ServerErrorMiddleware), so the handler that builds it adds
    them itself -- it went out without them (audit 2026-09-30, R05).
    """
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
    }
    # Everything is same-origin now (the admin UI and the self-hosted Swagger
    # page both load only vendored assets, no inline JS), so CSP applies
    # everywhere. Swagger UI injects its own styles and inline SVG/data: icons
    # at runtime, so /docs alone needs style-src 'unsafe-inline' + img-src data:;
    # every other route keeps the strict same-origin policy.
    if path.startswith("/docs"):
        headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:"
        )
    else:
        headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'"
        )
    if path.startswith("/ui"):
        # Always revalidate UI assets (304 when unchanged): browsers otherwise
        # keep executing a stale app.js from the heuristic cache after updates.
        headers["Cache-Control"] = "no-cache"
    return headers


def install(app: FastAPI, settings: Settings) -> None:
    """Register the body guard, the request counters, the security headers and the
    correlation id, in that order.

    Starlette runs middleware in REVERSE registration order, so the last registered is the
    outermost. Both orderings here are load-bearing: the headers wrap the body guard, so its
    401 or 413 still carries them; and the correlation id wraps everything, so every
    response has an id — including those and the 500 the error handler builds — and the
    id is bound before any other layer can log.
    """
    app.add_middleware(BodyGuard, settings=settings)

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        """Count every request by route template and status, and time it (audit OPS-9).

        The TEMPLATE, not the path: Starlette puts the matched route in the scope during
        routing, which happens inside ``call_next``, so it is read afterwards. Anything that
        matched nothing is one bucket — keyed on the path, a scanner could mint an unbounded
        number of permanent series in a process that never restarts.
        """
        started = time.perf_counter()

        def count(status: int) -> None:
            route = request.scope.get("route")
            template = getattr(route, "path", None) or telemetry.UNMATCHED
            telemetry.record(request.method, template, status, time.perf_counter() - started)

        try:
            response = await call_next(request)
        except Exception:
            # Unhandled: the 500 is built outside this stack, so it is counted here or not at
            # all -- and "the API is returning 500s" is the alert this counter exists for
            # (audit 2026-09-30, R05).
            count(500)
            raise
        count(response.status_code)
        return response

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Baseline hardening headers on every response (``hardening_headers``)."""
        response = await call_next(request)
        response.headers.update(hardening_headers(request.url.path))
        return response

    # Registered last, so it is the OUTERMOST: every response gets an id, including the
    # refusals the body guard short-circuits and the 500 the error handler builds — and the id
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

