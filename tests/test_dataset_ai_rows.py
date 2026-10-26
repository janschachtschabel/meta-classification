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


def test_quoted_fields_span_lines_and_hold_the_separator(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('title;labels;generated_for\n"Zeile eins\nZeile zwei; mit Semikolon";a;a\n'
                    '"ohne";b;\n', encoding="utf-8")

    counts = dataset_stats.ai_rows(path, separator=";", columns=["title", "labels", "generated_for"])

    assert counts == {"generated": 1, "example": 0, "enriched": 0, "marked": 1}


def test_a_gzipped_dataset_is_counted_decompressed(tmp_path):
    import gzip

    path = tmp_path / "marked.csv.gz"
    path.write_bytes(gzip.compress(MARKED.encode("utf-8")))

    counts = dataset_stats.ai_rows(path, separator=";", columns=["generated_for", "example_for",
                                                                   "enriched_fields"])

    assert counts == {"generated": 3, "example": 1, "enriched": 2, "marked": 5}


def test_a_cp1252_mark_is_a_mark(tmp_path):
    # Read as UTF-8 with replacement, not detected: detecting the encoding of a 533 MB export
    # took 16.5 s. A byte UTF-8 cannot decode still makes a cell non-empty, which is all a
    # mark is.
    path = tmp_path / "ansi.csv"
    path.write_bytes(b"title;labels;generated_for\nA;a;\xe4\nB;b;\n")

    counts = dataset_stats.ai_rows(path, separator=";", columns=["title", "labels", "generated_for"])

    assert counts["generated"] == 1


def test_the_count_is_kept_until_the_file_changes(tmp_path, monkeypatch):
    # Every pick of a dataset in the training form asks for it: a 533 MB export took 22 s a
    # time, read afresh on every pick.
    path = tmp_path / "marked.csv"
    path.write_text(MARKED, encoding="utf-8")
    columns = ["title", "labels", "generated_for", "example_for", "enriched_fields"]
    assert dataset_stats.ai_rows(path, separator=";", columns=columns)["marked"] == 5

    def boom(*args, **kwargs):
        raise AssertionError("file re-read although it is unchanged")

    monkeypatch.setattr("builtins.open", boom)
    assert dataset_stats.ai_rows(path, separator=";", columns=columns)["marked"] == 5
    monkeypatch.undo()

    path.write_text(MARKED + "Neu erzeugt;a;a;;\n", encoding="utf-8")
    assert dataset_stats.ai_rows(path, separator=";", columns=columns)["marked"] == 6


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
