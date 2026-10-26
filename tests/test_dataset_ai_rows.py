"""How much of a dataset an LLM wrote or touched, said before a run starts.

The training form offers `synthetic_rows` (train on the AI-marked rows, or leave them out) but
said nothing about how many there were: the choice was made blind. `GET /datasets/{name}` --
the call the form already waits for -- now counts the marks, with the reader training uses
(`provenance.block_marks`), so the count and the run agree on what a mark is.
"""

import pytest
from fastapi.testclient import TestClient

from app import dataset_stats

ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}

MARKED = "\n".join([
    "title;labels;generated_for;example_for;enriched_fields",
    "Echte Zeile;a;;;",
    "Erzeugt;a;a;;",
    "Erzeugt und ergänzt;b;b;;title",
    "Vorlage;b;;b;",
    "Ergänzt;a;;;title",
    "Leerzeichen sind keine Marke;a;   ;;",
    # Read as written, like the loader reads them: "NA" names a label, so it is a mark.
    "NA ist eine Marke;a;NA;;",
]) + "\n"


def test_the_marks_are_counted_per_kind_and_per_row(tmp_path):
    path = tmp_path / "marked.csv"
    path.write_text(MARKED, encoding="utf-8")

    counts = dataset_stats.ai_rows(path, separator=";",
                                   columns=["title", "labels", "generated_for", "example_for",
                                            "enriched_fields"])

    assert counts == {"generated": 3, "example": 1, "enriched": 2, "marked": 5}


def test_a_dataset_without_mark_columns_has_no_count(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("title;labels\nText;a\n", encoding="utf-8")

    assert dataset_stats.ai_rows(path, separator=";", columns=["title", "labels"]) is None


def test_only_the_mark_columns_present_are_read(tmp_path):
    path = tmp_path / "some.csv"
    path.write_text("title;labels;enriched_fields\nA;a;title\nB;a;\n", encoding="utf-8")

    counts = dataset_stats.ai_rows(path, separator=";",
                                   columns=["title", "labels", "enriched_fields"])

    assert counts == {"generated": 0, "example": 0, "enriched": 1, "marked": 1}


# Clearing the settings cache re-points it at whatever APIV3_* the last-collected module set:
# a module sorting before tests/test_api.py (which relies on the cache it bound at import)
# must not do it. Hence this file's name, not test_ai_rows.py.
@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "marked.csv").write_text(MARKED, encoding="utf-8")
    (tmp_path / "data" / "plain.csv").write_text("title;labels\nText;a\n", encoding="utf-8")

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


def test_the_dataset_info_carries_the_count(client):
    marked = client.get("/datasets/marked.csv", headers=RO)
    plain = client.get("/datasets/plain.csv", headers=RO)

    assert marked.status_code == 200
    assert marked.json()["rows"] == 7
    assert marked.json()["ai_rows"] == {"generated": 3, "example": 1, "enriched": 2, "marked": 5}
    assert plain.status_code == 200
    assert "ai_rows" not in plain.json()
