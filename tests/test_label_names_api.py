"""The label-name sidecar over HTTP (audit 2026-09-30, improvement 9).

Since B06 a container's `label_names.json` had to be copied into the volume with `docker cp`
or `kubectl cp` -- which not every cluster lets an editor do. An admin uploads it instead;
training reads exactly that file. An upload, never a URL: the app fetches nothing.
"""

import json

import pytest
from fastapi.testclient import TestClient

ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}
NAMES = {"http://w3id.org/openeduhub/vocabs/discipline/380": "Mathematik",
         "http://w3id.org/openeduhub/vocabs/discipline/080": "Biologie"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))
    (tmp_path / "data").mkdir()

    from app.limiter import limiter
    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    get_registry.cache_clear()
    limiter.reset()
    yield TestClient(create_app())
    get_settings.cache_clear()
    get_registry.cache_clear()


def _on_disk(tmp_path) -> dict:
    return json.loads((tmp_path / "data" / "label_names.json").read_text(encoding="utf-8"))


def test_uploaded_names_are_what_training_reads(client, tmp_path):
    from app.label_sidecar import load

    response = client.put("/label-names", headers=ADMIN, json=NAMES)

    assert response.status_code == 200, response.text
    assert response.json() == {"labels": 2}
    assert load(tmp_path / "data") == NAMES
    assert client.get("/label-names", headers=RO).json() == {"labels": 2, "names": NAMES}


def test_no_file_reads_as_no_names(client):
    assert client.get("/label-names", headers=RO).json() == {"labels": 0, "names": {}}


@pytest.mark.parametrize("body", [{}, {"http://x/1": " "}, {"http://x/1": 5}, ["not", "a", "mapping"]])
def test_an_unusable_upload_is_refused_and_the_names_kept(client, tmp_path, body):
    """W07's lesson: `{}` written over a good file empties every display name the next
    training would have used."""
    client.put("/label-names", headers=ADMIN, json=NAMES)

    response = client.put("/label-names", headers=ADMIN, json=body)

    assert response.status_code == 422, response.text
    assert _on_disk(tmp_path) == NAMES


def test_only_an_admin_replaces_the_names(client):
    assert client.put("/label-names", headers=RO, json=NAMES).status_code == 403


def test_a_write_that_fails_leaves_the_names_as_they_were(client, tmp_path, monkeypatch):
    import app.label_sidecar as sidecar

    client.put("/label-names", headers=ADMIN, json=NAMES)

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sidecar.os, "replace", refuse)
    response = client.put("/label-names", headers=ADMIN, json={"http://x/1": "Neu"})

    assert response.status_code == 503, response.text
    assert _on_disk(tmp_path) == NAMES
    assert [p.name for p in (tmp_path / "data").iterdir()] == ["label_names.json"]
