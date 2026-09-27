"""A damaged bundle reports as damaged, not as a server fault (audit API-3).

`model_io._read_bundle` wraps its JSON parses in `UnsafeModelError` explicitly "so routes
answer 422/400 instead of a 500", which is why `POST /predict` on a corrupt bundle is a 422.
`model_report.read_documents` parsed the identical documents unguarded, so the two read-only
reports — the ones an operator reaches for precisely *because* a bundle looks wrong — answered
500 and said nothing.

A truncated `config.json` is the realistic shape: `Registry.exists` tests for that file alone,
so a bundle whose atomic write was interrupted mid-file is listed and looks present.
"""

import pytest
from fastapi.testclient import TestClient

RO = {"X-API-Key": "ro-key"}


def _client_with_a_broken_bundle(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))

    broken = tmp_path / "models" / "broken"
    broken.mkdir(parents=True)
    # Cut off mid-document, as an interrupted write leaves it.
    (broken / "config.json").write_text('{"backend_kind": "tfidf", "classes": [', encoding="utf-8")

    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    get_registry.cache_clear()
    from app.limiter import limiter
    from app.main import create_app

    limiter.reset()
    return TestClient(create_app())


@pytest.mark.parametrize("path", ["/models/broken", "/models/broken/labels"])
def test_a_corrupt_bundle_is_reported_as_unprocessable_not_as_a_server_error(
    monkeypatch, tmp_path, path
):
    client = _client_with_a_broken_bundle(monkeypatch, tmp_path)

    response = client.get(path, headers=RO)

    assert response.status_code == 422, response.text
    # The message has to name the bundle problem without leaking a path or a traceback.
    detail = response.json()["detail"]
    assert "broken" in detail
    assert str(tmp_path) not in detail


def test_a_corrupt_bundle_still_appears_in_the_listing(monkeypatch, tmp_path):
    """The listing must not fall over because one bundle is damaged — otherwise a single
    bad directory hides every healthy model from the operator trying to find it."""
    client = _client_with_a_broken_bundle(monkeypatch, tmp_path)

    response = client.get("/models", headers=RO)

    assert response.status_code == 200, response.text
