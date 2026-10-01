"""What an operator can see when something goes wrong (audit OPS-8, OPS-9, API-6, API-7).

Sanitizing a 500 body is right and leaves the operator with nothing to search for; a
ServiceMonitor with six process gauges has nothing to alert on; a 429 without `Retry-After`
makes every client guess; and a `detail` that is a string on eight status codes and a list of
objects on the ninth breaks the client that reads it.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from app import telemetry
from app.correlation import RequestIdFilter, accept


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    get_settings.cache_clear()
    get_registry.cache_clear()
    telemetry.reset()
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client
    get_settings.cache_clear()
    get_registry.cache_clear()


# --- OPS-8: one id per request -------------------------------------------------------------


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")

    assert response.headers.get("X-Request-ID"), "no correlation id on the response"


def test_two_requests_get_different_ids(client):
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]

    assert first != second


def test_an_inbound_id_is_honoured_so_one_id_spans_the_hop(client):
    """A reverse proxy or a client that already has an id for this call should see it kept,
    or correlating across the hop means correlating two ids."""
    response = client.get("/health", headers={"X-Request-ID": "edge-abc123"})

    assert response.headers["X-Request-ID"] == "edge-abc123"


def test_an_inbound_id_cannot_forge_a_log_line(client):
    """The id reaches log lines, so a newline in it would let a client write its own."""
    response = client.get("/health", headers={"X-Request-ID": "ok\r\nlevel=CRITICAL fake"})

    got = response.headers["X-Request-ID"]
    assert "\n" not in got and "\r" not in got and " " not in got
    # And the cleaner keeps something usable rather than discarding the whole thing.
    assert got.startswith("oklevel")


def test_the_cleaner_falls_back_when_nothing_usable_survives():
    assert accept("!!!!") != "!!!!"
    assert len(accept("")) == 8
    assert accept("a" * 200) == "a" * 64  # capped, not rejected


def test_a_sanitized_500_names_the_id_its_traceback_was_logged_under(client, monkeypatch):
    """This is the whole point: the body says nothing about the failure, so it has to say
    which log line describes it."""
    from app.routes import system as system_routes

    def boom(*_args, **_kwargs):
        raise RuntimeError("internal detail that must not leak")

    monkeypatch.setattr(system_routes.job_runner, "snapshot", boom)
    response = client.get("/metrics")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error."
    assert "internal detail" not in response.text
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_a_log_record_carries_the_id_even_outside_a_request():
    """Startup and the training child log too; the filter must not raise there."""
    record = logging.LogRecord("api_v3", logging.INFO, __file__, 1, "msg", None, None)

    assert RequestIdFilter().filter(record) is True
    assert record.request_id == "-"


def test_the_log_format_actually_prints_the_id():
    """A filter that attaches an attribute no format string names is invisible."""
    from pathlib import Path

    main = (Path(__file__).parent.parent / "app" / "main.py").read_text(encoding="utf-8")

    assert "%(request_id)s" in main, "the id is attached to records and never printed"


# --- OPS-9: the traffic, not just the process ---------------------------------------------


def test_metrics_counts_requests_by_route_and_status(client):
    client.get("/health")
    client.get("/health")
    client.get("/models", headers={"X-API-Key": "ro-key"})

    body = client.get("/metrics").text

    assert 'apiv3_requests_total{method="GET",route="/health",status="200"} 2' in body
    assert 'apiv3_requests_total{method="GET",route="/models",status="200"} 1' in body


def test_metrics_reports_what_a_route_costs(client):
    client.get("/health")
    body = client.get("/metrics").text

    assert 'apiv3_request_duration_seconds_count{method="GET",route="/health"} 1' in body
    assert 'apiv3_request_duration_seconds_sum{method="GET",route="/health"}' in body


def test_a_failure_is_visible_as_one(client):
    """"The API is returning 500s" was invisible: no counter had a status in it."""
    client.get("/models/nope/labels", headers={"X-API-Key": "ro-key"})
    body = client.get("/metrics").text

    assert 'status="404"' in body


def test_scanning_urls_cannot_mint_unbounded_series(client):
    """Keyed on the path, a scanner would add a permanent series per URL it tried, in a
    process that never restarts. The route TEMPLATE is finite; unmatched is one bucket."""
    for n in range(5):
        client.get(f"/definitely-not-a-route-{n}")

    body = client.get("/metrics").text

    assert body.count("route=\"<unmatched>\"") >= 1
    assert "definitely-not-a-route" not in body


def test_inventing_methods_cannot_mint_unbounded_series(client):
    """S13 (audit 2026-09-30): the method was a label taken as sent, and to uvicorn's h11
    parser any token is a method -- 200 invented ones made 600 series. The methods HTTP
    defines are kept apart; anything else is one bucket, like an unmatched route."""
    for n in range(5):
        client.request(f"BREW{n}", "/health")
    client.request("PATCH", "/health")

    body = client.get("/metrics").text

    assert "BREW" not in body
    assert body.count('apiv3_requests_total{method="OTHER",route="/health"') == 1
    assert 'apiv3_requests_total{method="PATCH",route="/health"' in body


def test_the_queue_depth_is_exported(client):
    """`queued_names()` existed and was not exported, so "runs are piling up" could not
    be alerted on either."""
    body = client.get("/metrics").text

    assert "apiv3_training_queued" in body


# --- API-6: a 429 says when to come back ---------------------------------------------------


def test_a_rate_limited_response_says_how_long_to_wait(client, monkeypatch):
    """Patched down to 1/minute the way tests/test_api.py does it: the limit is resolved
    per request from the cached settings object, so this is what actually trips it."""
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "rate_limit_default", "1/minute")
    assert client.get("/metrics").status_code == 200
    last = client.get("/metrics")

    assert last.status_code == 429, "the limiter did not trip"
    assert last.headers.get("Retry-After") == "60", (
        "a 429 with no Retry-After makes every client guess, and the usual guess is an "
        f"immediate retry; got {last.headers.get('Retry-After')!r}"
    )


# --- API-7: `detail` has one type ---------------------------------------------------------


def test_a_validation_error_reports_detail_as_a_string_like_every_other_error(client):
    """FastAPI's default handler makes `detail` a list of objects on 422 alone, so client
    code doing `body["detail"].startswith(...)` raises on exactly the status a client hits
    most while integrating."""
    response = client.post("/predict", headers={"X-API-Key": "ro-key"}, json={})

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], str), f"detail is {type(body['detail']).__name__}"
    # The structured form is what tells a form which field was wrong — kept, not dropped.
    assert isinstance(body["errors"], list) and body["errors"], "the field details are gone"
    # Which field pydantic reports first is its business; that it reports a body field
            # at all is the contract.
    assert all(item.get("loc", [None])[0] == "body" for item in body["errors"])
