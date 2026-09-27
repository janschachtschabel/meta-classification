"""Callers can bound a listing, and an out-of-range bound is refused rather than ignored
(audit API-8).

None of the four list endpoints could be paged, and the fifth accepted any `limit` and quietly
substituted a different one. Both are the same problem seen from either side: the caller cannot
tell what it is going to get back.

The response shape stays a bare array. Adding an envelope would break every existing client for
a Low finding; optional `limit`/`offset` that default to "everything" break nobody and give a
caller with 3 000 models a way through them.
"""

import pytest
from fastapi.testclient import TestClient


def _bounds(schema: dict) -> tuple[int | None, int | None]:
    """(minimum, maximum) from a parameter schema, looking inside an anyOf when the
    parameter is optional — FastAPI nests the constraints on the integer branch there."""
    for branch in schema.get("anyOf", [schema]):
        if branch.get("type") == "integer":
            return branch.get("minimum"), branch.get("maximum")
    return schema.get("minimum"), schema.get("maximum")


ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}


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
    for n in range(7):
        (tmp_path / "data" / f"set{n}.csv").write_text("title;label\na;uri:x\n", encoding="utf-8")
    get_settings.cache_clear()
    get_registry.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
    get_registry.cache_clear()


def test_a_listing_returns_everything_by_default(client):
    """The existing contract, unchanged: no parameters means the whole list."""
    names = [d["name"] for d in client.get("/datasets", headers=RO).json()]

    assert len(names) == 7


def test_a_listing_can_be_bounded(client):
    first = client.get("/datasets?limit=3", headers=RO).json()

    assert len(first) == 3


def test_a_listing_can_be_paged_without_gaps_or_repeats(client):
    """The point of an offset: two pages must cover the whole list exactly once, which only
    holds if the order is stable between calls."""
    page_one = [d["name"] for d in client.get("/datasets?limit=4", headers=RO).json()]
    page_two = [d["name"] for d in client.get("/datasets?limit=4&offset=4", headers=RO).json()]
    everything = [d["name"] for d in client.get("/datasets", headers=RO).json()]

    assert page_one + page_two == everything


def test_an_offset_past_the_end_is_an_empty_page_not_an_error(client):
    response = client.get("/datasets?limit=5&offset=500", headers=RO)

    assert response.status_code == 200
    assert response.json() == []


def test_the_same_bounds_work_on_the_other_listings(client):
    for path, headers in (("/models", RO), ("/share", ADMIN)):
        response = client.get(f"{path}?limit=1&offset=0", headers=headers)
        assert response.status_code == 200, f"{path}: {response.text}"
        assert isinstance(response.json(), list)


@pytest.mark.parametrize("query", ["limit=0", "limit=-1", "offset=-1", "limit=2001"])
def test_a_bound_outside_the_allowed_range_is_refused(client, query):
    """422, not a silent substitution: a caller that asks for 5 000 and receives 200 without
    being told has no way to know the list was cut."""
    response = client.get(f"/datasets?{query}", headers=RO)

    assert response.status_code == 422, response.text


def test_the_history_limit_is_validated_instead_of_clamped(client):
    """`/train/history` accepted any number and used `max(1, min(limit, 200))`, so a request
    for 1 000 answered with 200 runs and said nothing about it."""
    over = client.get("/train/history?limit=1000", headers=RO)
    under = client.get("/train/history?limit=0", headers=RO)

    assert over.status_code == 422, over.text
    assert under.status_code == 422, under.text
    assert client.get("/train/history?limit=200", headers=RO).status_code == 200


def test_the_documented_bounds_are_the_enforced_ones(client):
    """A description that names a range the validation does not enforce is the same defect
    the clamp was."""
    schema = client.get("/openapi.json").json()
    for path in ("/datasets", "/models", "/share"):
        parameters = {p["name"]: p for p in schema["paths"][path]["get"].get("parameters", [])}
        assert "limit" in parameters and "offset" in parameters, path
        # An optional int is an anyOf(integer, null), so the bounds sit on the integer
        # branch rather than at the top level.
        assert _bounds(parameters["limit"]["schema"]) == (1, 2000), path
        assert _bounds(parameters["offset"]["schema"])[0] == 0, path
