"""No request body is read before the caller is known, nor past its limit (audit 2026-09-30, S01).

FastAPI parses a body -- JSON into objects, multipart into spooled temp files -- before any
dependency runs, and `require_role` is a dependency. So a caller WITHOUT a key made the server
read and parse whatever it sent, up to the upload cap: 48 MiB of `[{},{},...]` took the process
from 232 to 1,485 MiB for a 401, and a chunked upload -- which declares no Content-Length for the
old ceiling to check -- landed whole in a temp file before its 413.

These drive the ASGI app directly, with a `receive` that counts the bytes the app actually
consumed: "refused" is not enough, the property is "refused before it was read".
"""

import asyncio

import pytest

MiB = 1024 * 1024
REMOTE = ("203.0.113.9", 4711)  # TEST-NET-3: never loopback


@pytest.fixture
def make_app(monkeypatch, tmp_path):
    def build(**env):
        settings = {
            "APIV3_AUTH_ENABLED": "true", "APIV3_API_KEY_ADMIN": "admin-key",
            "APIV3_API_KEY_READONLY": "ro-key", "APIV3_DATA_DIR": str(tmp_path / "data"),
            "APIV3_MODELS_DIR": str(tmp_path / "models"), **env,
        }
        for key, value in settings.items():
            monkeypatch.setenv(key, value)
        from app.registry import get_registry
        from app.settings import get_settings

        get_settings.cache_clear()
        get_registry.cache_clear()
        from app.main import create_app

        return create_app()

    yield build
    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    get_registry.cache_clear()


def _call(app, path: str, headers: dict[str, str], chunks: list[bytes], client=REMOTE) -> dict:
    """POST ``chunks`` to ``path``; the status, the response headers, and the body bytes the
    app consumed before it answered."""
    pending = list(chunks)
    consumed = 0
    answered = asyncio.Event()
    start: dict = {}

    async def receive() -> dict:
        nonlocal consumed
        if pending:
            chunk = pending.pop(0)
            consumed += len(chunk)
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        await answered.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            start.update(message)
        elif message["type"] == "http.response.body" and not message.get("more_body"):
            answered.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "client": client, "server": ("testserver", 80),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    asyncio.run(app(scope, receive, send))
    return {"status": start["status"], "consumed": consumed,
            "headers": {k.decode(): v.decode() for k, v in start["headers"]}}


JSON = {"content-type": "application/json"}
UPLOAD = {"content-type": "multipart/form-data; boundary=x"}
CHUNKED = {"transfer-encoding": "chunked"}


def _declared(size: int) -> dict[str, str]:
    return {"content-length": str(size)}


def test_a_body_without_a_key_is_refused_before_a_byte_is_read(make_app):
    answer = _call(make_app(), "/predict", {**JSON, **_declared(20 * MiB)}, [b"x" * MiB] * 20)

    assert (answer["status"], answer["consumed"]) == (401, 0)


def test_a_chunked_upload_without_a_key_is_refused_before_a_byte_is_read(make_app):
    """Chunked: no Content-Length, which is all the old ceiling looked at."""
    answer = _call(make_app(), "/datasets/import", {**UPLOAD, **CHUNKED}, [b"x" * MiB] * 20)

    assert (answer["status"], answer["consumed"]) == (401, 0)


def test_a_wrong_key_is_refused_before_a_byte_is_read(make_app):
    answer = _call(make_app(), "/predict", {**JSON, **_declared(5 * MiB), "x-api-key": "guess"},
                   [b"x" * MiB] * 5)

    assert (answer["status"], answer["consumed"]) == (401, 0)


def test_keyless_mode_refuses_a_remote_body_before_a_byte_is_read(make_app):
    answer = _call(make_app(APIV3_AUTH_ENABLED="false"), "/predict", {**JSON, **_declared(5 * MiB)},
                   [b"x" * MiB] * 5)

    assert (answer["status"], answer["consumed"]) == (403, 0)


def test_a_json_body_is_held_to_its_own_limit_on_its_declared_size(make_app):
    """JSON is parsed into objects, which cost many times the bytes -- so it gets a limit of
    its own, far below the upload cap a dataset needs."""
    answer = _call(make_app(APIV3_MAX_JSON_MB="2"), "/predict",
                   {**JSON, **_declared(3 * MiB), "x-api-key": "ro-key"}, [b"x" * MiB] * 3)

    assert (answer["status"], answer["consumed"]) == (413, 0)


def test_a_chunked_json_body_is_cut_off_at_its_limit(make_app):
    answer = _call(make_app(APIV3_MAX_JSON_MB="2"), "/predict",
                   {**JSON, **CHUNKED, "x-api-key": "ro-key"}, [b"x" * MiB] * 20)

    assert answer["status"] == 413
    assert answer["consumed"] <= 3 * MiB, "read past the limit"


def test_a_chunked_upload_is_cut_off_at_the_upload_cap(make_app):
    """The spool into a temp file is what filled the chart's 1 Gi /tmp: a well-formed file
    part that never ends is streamed to disk for as long as it keeps coming."""
    part = (b'--x\r\nContent-Disposition: form-data; name="file"; filename="big.csv"\r\n'
            b"Content-Type: text/csv\r\n\r\n")
    answer = _call(make_app(APIV3_MAX_UPLOAD_MB="2"), "/datasets/import",
                   {**UPLOAD, **CHUNKED, "x-api-key": "admin-key"},
                   [part + b"x" * MiB] + [b"x" * MiB] * 19)

    assert answer["status"] == 413
    assert answer["consumed"] <= 3 * MiB, "read past the cap"


def test_a_refusal_still_carries_the_headers_every_response_has(make_app):
    answer = _call(make_app(), "/predict", {**JSON, **_declared(MiB)}, [b"x" * MiB])

    assert answer["headers"].get("x-content-type-options") == "nosniff"
    assert answer["headers"].get("x-request-id")
