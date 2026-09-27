"""One id per request, on the log lines, the response and the sanitized error body.

Sanitizing a 500 body is right — the traceback belongs in the log, not in the answer — but
it leaves the operator with a report ("it failed at 14:02") and no way to find the matching
traceback among a day of plain-text log lines (audit OPS-8). The other half of that trade is
this: the answer carries an id, and so does every log line written while handling it.

A stdlib-only leaf. ``contextvars`` is what makes it work without threading an argument
through every call: the value is per-task, and ``asyncio.to_thread`` copies the context, so
a line logged from the registry's worker thread carries the id of the request that started
it. An inbound ``X-Request-ID`` is honoured so a reverse proxy's or a client's id wins and
one id spans the whole hop.
"""

from __future__ import annotations

import logging
import secrets
from contextvars import ContextVar

# Short on purpose: 8 hex characters is 4 billion values, which is plenty to disambiguate
# within a log retention window, and it stays readable in a line prefix and in a bug report.
_ID_BYTES = 4
# A client-supplied id is echoed but never trusted as-is: it lands in log lines, so it is
# length-capped and stripped of anything that could forge a line or a field separator.
_MAX_INBOUND = 64
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")

NONE = "-"

_current: ContextVar[str] = ContextVar("apiv3_request_id", default=NONE)


def new_id() -> str:
    return secrets.token_hex(_ID_BYTES)


def accept(inbound: str | None) -> str:
    """The id to use for this request: a usable inbound one, else a fresh one."""
    if inbound:
        cleaned = "".join(c for c in inbound[:_MAX_INBOUND] if c in _SAFE)
        if cleaned:
            return cleaned
    return new_id()


def set_current(request_id: str) -> object:
    """Bind the id for this task; the returned token resets it (see ``reset``)."""
    return _current.set(request_id)


def reset(token: object) -> None:
    _current.reset(token)  # type: ignore[arg-type]


def current() -> str:
    return _current.get()


def of_request(request: object) -> str:
    """The id bound to this request, for code that runs outside the middleware stack.

    ``add_exception_handler(Exception, ...)`` is served by Starlette's
    ``ServerErrorMiddleware``, the outermost layer of all — so the sanitized 500 is built
    after the middleware has already unwound and ``current()`` is back to ``-``. The handler
    does get the ``Request``, and the scope is the same object the middleware saw, so the id
    is put there as well and read back here.
    """
    state = getattr(request, "state", None)
    return str(getattr(state, "request_id", NONE))


class RequestIdFilter(logging.Filter):
    """Puts ``request_id`` on every record so the format string can name it.

    A filter rather than a formatter subclass: it attaches the attribute for whatever
    handler and format the deployment configures, and a record logged outside a request
    (startup, the training child) gets ``-`` instead of raising.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current()
        return True
