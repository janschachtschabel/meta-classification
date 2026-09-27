"""Only a dataset may be named as a dataset (audit API-4).

The data directory holds more than datasets — `label_names.json` is the display-name sidecar
every training reads. `routes/datasets.py` funnels its five routes through one resolver that
checks the suffix; `/train` and `/models/{name}/evaluate` hand-rolled `safe_name` + `exists()`
and skipped it, so a non-dataset name was accepted with 202 and surfaced minutes later as a
job error instead of immediately as a 404.

Not a traversal hole — `safe_name` holds at every site. It is the difference between an answer
and a wild goose chase.
"""

import pytest
from fastapi.testclient import TestClient

ADMIN = {"X-API-Key": "admin-key"}

# The pipeline fields /train requires, so a 422 cannot be mistaken for the 404 under test.
REQUIRED = {"text_columns": ["text"], "label_column": "label"}


@pytest.fixture
def client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))

    data = tmp_path / "data"
    data.mkdir(parents=True)
    # The real sidecar, present in every deployment that uses display names.
    (data / "label_names.json").write_text('{"http://x/1": "Mathematik"}', encoding="utf-8")
    (data / "real.csv").write_text("text,label\nein Satz,http://x/1\n", encoding="utf-8")

    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    get_registry.cache_clear()
    from app.limiter import limiter
    from app.main import create_app

    limiter.reset()
    return TestClient(create_app())


def test_train_refuses_a_file_that_is_not_a_dataset(client):
    response = client.post(
        "/train", headers=ADMIN, json={"dataset_name": "label_names.json", "model_name": "m", **REQUIRED}
    )

    assert response.status_code == 404, response.text
    assert "label_names.json" in response.json()["detail"]


def test_evaluate_refuses_a_file_that_is_not_a_dataset(client, tmp_path):
    bundle = tmp_path / "models" / "m"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text('{"backend_kind": "tfidf"}', encoding="utf-8")

    response = client.post(
        "/models/m/evaluate", headers=ADMIN, json={"dataset_name": "label_names.json", **REQUIRED}
    )

    assert response.status_code == 404, response.text
    assert "label_names.json" in response.json()["detail"]


def test_a_missing_dataset_is_still_a_404(client):
    """The suffix check must not shadow the existence check — both are 404, same wording."""
    response = client.post(
        "/train", headers=ADMIN, json={"dataset_name": "nope.csv", "model_name": "m", **REQUIRED}
    )

    assert response.status_code == 404, response.text
    assert "nope.csv" in response.json()["detail"]


def test_the_policy_itself_is_one_function():
    """`data.resolve_dataset` is where the rule lives, so the four routes cannot drift again.

    It raises `FileNotFoundError` rather than an HTTP error on purpose: `app/data.py` is a core
    module and the routes own the HTTP mapping.
    """
    from pathlib import Path

    from app import data

    with pytest.raises(FileNotFoundError):
        data.resolve_dataset(Path("."), "label_names.json")
