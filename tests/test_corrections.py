"""The corrections an editor recorded: seen, backed up, brought back, and trained on.

`POST /feedback` wrote them down and `GET /feedback/export` handed out a training CSV -- both
API-only, so the loop the corrections exist for could not be closed from the admin UI. These
pin the four things it was missing: a paged view, a backup that keeps every field, an import of
that backup that adds nothing twice, and turning the corrections into a dataset a run can train.
"""

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from app import feedback, feedback_review
from app.errors import FeedbackWriteError

ADMIN = {"X-API-Key": "admin-key"}
RO = {"X-API-Key": "ro-key"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "_feedback_path", lambda: path)
    monkeypatch.setattr(feedback, "_count", None)
    return path


def _correction(text: str, corrected: list[str], model: str = "subjects") -> dict:
    return {"text": text, "model_name": model, "predicted": ["uri:math"],
            "corrected": corrected, "source": "ui"}


# ---------------------------------------------------------------------------- the view ---


def test_the_view_shows_the_newest_first_per_model(store):
    feedback.append(_correction("erste", ["uri:hist"], model="a"))
    feedback.append(_correction("zweite", ["uri:math"], model="b"))
    feedback.append(_correction("dritte", ["uri:geo"], model="a"))

    everything = feedback_review.page(model=None, limit=2, offset=0)
    only_a = feedback_review.page(model="a", limit=10, offset=0)
    second_page = feedback_review.page(model=None, limit=2, offset=2)

    assert [item["text"] for item in everything["items"]] == ["dritte", "zweite"]
    assert everything["total"] == 3
    assert everything["models"] == {"a": 2, "b": 1}
    assert [item["text"] for item in only_a["items"]] == ["dritte", "erste"]
    assert only_a["total"] == 2
    assert [item["text"] for item in second_page["items"]] == ["erste"]
    assert everything["items"][0]["recorded_at"]


def test_the_view_shortens_a_long_text_and_says_so(store):
    feedback.append(_correction("x" * 5000, ["uri:hist"]))

    item = feedback_review.page(model=None, limit=10, offset=0)["items"][0]

    assert len(item["text"]) == feedback_review.VIEW_TEXT_CHARS
    assert item["text_truncated"] is True


# ------------------------------------------------------------------- backup and import ---


def test_the_backup_keeps_every_field_and_imports_back(store, tmp_path, monkeypatch):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))
    feedback.append(_correction("Nichts passt", []))
    backup = list(feedback_review.backup_lines())

    other = tmp_path / "other.jsonl"
    monkeypatch.setattr(feedback, "_feedback_path", lambda: other)
    monkeypatch.setattr(feedback, "_count", None)
    result = feedback_review.import_lines(backup)

    assert result == {"imported": 2, "duplicates": 0, "invalid": 0}
    assert feedback.read_all() == [json.loads(line) for line in backup], "recorded_at included"


def test_an_import_adds_nothing_twice(store):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))
    backup = list(feedback_review.backup_lines())

    again = feedback_review.import_lines(backup + backup)

    assert again == {"imported": 0, "duplicates": 2, "invalid": 0}
    assert len(feedback.read_all()) == 1


def test_an_import_skips_what_a_correction_could_not_be(store):
    good = json.dumps({**_correction("gut", ["uri:hist"]), "recorded_at": "2026-10-01T08:00:00+00:00"})
    lines = [
        good,
        "{not json",
        json.dumps({"model_name": "subjects", "corrected": ["uri:hist"],
                    "recorded_at": "2026-10-01T08:00:00+00:00"}),                 # no text
        json.dumps({**_correction("böser Name", ["uri:hist"], model="../etc"),
                    "recorded_at": "2026-10-01T08:00:00+00:00"}),                 # not a safe name
        json.dumps({**_correction("ohne Zeit", ["uri:hist"])}),                   # no recorded_at
        "",
    ]

    result = feedback_review.import_lines(lines)

    assert result == {"imported": 1, "duplicates": 0, "invalid": 4}
    assert [entry["text"] for entry in feedback.read_all()] == ["gut"]


def test_an_import_past_the_cap_writes_nothing(store, monkeypatch):
    feedback.append(_correction("schon da", ["uri:hist"]))
    before = store.read_bytes()
    monkeypatch.setattr(feedback, "_max_bytes", lambda: len(before) + 50)
    lines = [json.dumps({**_correction(f"neu {i}", ["uri:hist"]),
                         "recorded_at": f"2026-10-01T08:00:0{i}+00:00"}) for i in range(5)]

    with pytest.raises(FeedbackWriteError):
        feedback_review.import_lines(lines)

    assert store.read_bytes() == before, "all or nothing"


# --------------------------------------------------------------------------- over HTTP ---


@pytest.fixture
def client(monkeypatch, tmp_path, store):
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


def test_the_view_and_the_backup_are_admin_only(client):
    assert client.get("/feedback", headers=RO).status_code == 403
    assert client.get("/feedback/export?format=jsonl", headers=RO).status_code == 403
    assert client.post("/feedback/import", headers=RO,
                       files={"file": ("f.jsonl", b"", "application/x-ndjson")}).status_code == 403
    assert client.post("/feedback/dataset", headers=RO, json={"dataset_name": "x"}).status_code == 403


def test_the_view_over_http(client):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))

    answer = client.get("/feedback?model=subjects&limit=10", headers=ADMIN)

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["total"] == 1 and body["items"][0]["corrected"] == ["uri:hist"]


def test_the_backup_round_trips_over_http(client):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))

    backup = client.get("/feedback/export?format=jsonl", headers=ADMIN)
    assert backup.status_code == 200
    assert "ndjson" in backup.headers["content-type"]
    again = client.post("/feedback/import", headers=ADMIN,
                        files={"file": ("feedback.jsonl", backup.content, "application/x-ndjson")})

    assert again.status_code == 200, again.text
    assert again.json() == {"imported": 0, "duplicates": 1, "invalid": 0}


def test_the_training_csv_is_still_the_default_export(client):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))

    answer = client.get("/feedback/export", headers=ADMIN)

    assert answer.text.splitlines()[0] == "text;labels"


def test_corrections_become_a_dataset_a_run_can_read(client, tmp_path):
    from app.dataset_load import load_dataset

    feedback.append(_correction("Der Wiener Kongress von 1815", ["uri:hist"]))
    feedback.append(_correction("Bruchrechnung und Gleichungen", ["uri:math", "uri:stats"]))
    feedback.append(_correction("Nichts passt", []))

    answer = client.post("/feedback/dataset", headers=ADMIN, json={"dataset_name": "korrekturen"})

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"dataset_name": "korrekturen.csv", "rows": 2}
    path = tmp_path / "data" / "korrekturen.csv"
    data = load_dataset(path, ["text"], "labels", separator=feedback.CSV_SEPARATOR,
                        label_separator=feedback.LABEL_SEPARATOR, min_text_length=1)
    assert sorted(data.texts) == ["Bruchrechnung und Gleichungen", "Der Wiener Kongress von 1815"]


def test_saving_as_a_dataset_refuses_a_taken_or_bad_name(client, tmp_path):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))
    (tmp_path / "data" / "da.csv").write_text("text;labels\n", encoding="utf-8")

    taken = client.post("/feedback/dataset", headers=ADMIN, json={"dataset_name": "da.csv"})
    bad = client.post("/feedback/dataset", headers=ADMIN, json={"dataset_name": "../x"})

    assert taken.status_code == 409
    assert bad.status_code in (400, 422)
    assert (tmp_path / "data" / "da.csv").read_text(encoding="utf-8") == "text;labels\n"


def test_saving_as_a_dataset_needs_a_labelled_correction(client):
    feedback.append(_correction("Nichts passt", []))

    answer = client.post("/feedback/dataset", headers=ADMIN, json={"dataset_name": "leer"})

    assert answer.status_code == 400


def test_the_csv_written_as_a_dataset_is_the_export(client, tmp_path):
    feedback.append(_correction("=HYPERLINK(1)", ["uri:hist"]))

    client.post("/feedback/dataset", headers=ADMIN, json={"dataset_name": "k"})

    written = (tmp_path / "data" / "k.csv").read_text(encoding="utf-8")
    assert written == "".join(feedback.iter_csv())
    assert list(csv.reader(io.StringIO(written), delimiter=";"))[1][0] == "\t=HYPERLINK(1)"
