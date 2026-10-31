"""Which character separates a dataset's fields, found from its header instead of assumed.

Every request that reads a dataset defaults to `;`, and the admin UI never sent anything
else: a comma- or tab-separated upload showed one column holding the whole header, and could
not be trained from the UI at all. `GET /datasets/{name}` now detects the separator when the
caller names none, and says which one it used, so the UI can send it on.
"""

import gzip

import pytest
from fastapi.testclient import TestClient

from app.csv_dialect import detect_separator

ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}


@pytest.mark.parametrize(("header", "expected"), [
    ("title;keywords;labels", ";"),
    ("title,keywords,labels", ","),
    ("title\tkeywords\tlabels", "\t"),
    ("title|keywords|labels", "|"),
    ("title", ";"),                                   # one column: nothing to tell, the old default
    ('"Titel, mit Komma";labels', ";"),               # a comma inside a quoted name is no delimiter
])
def test_the_separator_is_the_one_the_header_uses(tmp_path, header, expected):
    path = tmp_path / "d.csv"
    path.write_text(header + "\nA;b\n", encoding="utf-8")

    assert detect_separator(path) == expected


def test_a_gzipped_dataset_is_read_decompressed(tmp_path):
    path = tmp_path / "d.csv.gz"
    path.write_bytes(gzip.compress(b"title,labels\nA,b\n"))

    assert detect_separator(path) == ","


# Clearing the settings cache re-points it at whatever APIV3_* the last-collected module set:
# a module sorting before tests/test_api.py must not do it, and this one sorts after it.
@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "comma.csv").write_text(
        'title,labels\n"Brüche, gekürzt",math\nVulkane,geo\n', encoding="utf-8")
    (tmp_path / "data" / "semi.csv").write_text("title;labels\nA;math\n", encoding="utf-8")

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


def test_the_dataset_info_detects_the_separator_and_says_so(client):
    comma = client.get("/datasets/comma.csv", headers=RO)
    semi = client.get("/datasets/semi.csv", headers=RO)

    assert comma.status_code == 200, comma.text
    assert comma.json()["columns"] == ["title", "labels"]
    assert comma.json()["separator"] == ","
    assert comma.json()["rows"] == 2
    assert comma.json()["sample"][0]["title"] == "Brüche, gekürzt"
    assert semi.json()["separator"] == ";" and semi.json()["columns"] == ["title", "labels"]


def test_a_named_separator_is_used_as_named(client):
    forced = client.get("/datasets/comma.csv?separator=;", headers=RO)
    bad = client.get("/datasets/comma.csv?separator=ab", headers=RO)

    assert forced.status_code == 200
    assert forced.json()["separator"] == ";"
    assert forced.json()["columns"] == ["title,labels"], "the caller's choice, even when it is wrong"
    assert bad.status_code == 400
