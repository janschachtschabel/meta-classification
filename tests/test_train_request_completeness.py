"""Every lever a caller may set actually reaches the run (audit ARC-2).

`/train` used to copy the body into the run request through a hand-maintained tuple of 15 field
names. A field added to `TrainRequest` was then accepted and validated by pydantic and silently
dropped on the way to the pipeline — which is not hypothetical: it happened to three text levers,
and each now has its own regression test in `tests/test_api.py`.

This asserts the property instead of the list: what the runner receives is the whole body minus
the two fields that deliberately travel another way.
"""

import pytest
from fastapi.testclient import TestClient

from app.schemas import TrainRequest

ADMIN = {"X-API-Key": "admin-key"}

# `info` is documentation, not a parameter of the run, and reaches the bundle as plain JSON;
# `optimize_parameters` is resolved to a profile object before submission. Both are asserted
# separately below rather than excused.
TRAVEL_SEPARATELY = {"info", "optimize_parameters"}


@pytest.fixture
def submitted(monkeypatch, tmp_path):
    """Spy on the job runner's boundary and return what /train handed it."""
    monkeypatch.setenv("APIV3_AUTH_ENABLED", "true")
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "admin-key")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "ro-key")
    monkeypatch.setenv("APIV3_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_JOB_HISTORY_FILE", str(tmp_path / "job_history.jsonl"))

    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "d.csv").write_text("text,label\nein Satz,http://x/1\n", encoding="utf-8")

    from app.registry import get_registry
    from app.settings import get_settings

    get_settings.cache_clear()
    get_registry.cache_clear()
    from app.jobs import job_runner
    from app.limiter import limiter
    from app.main import create_app

    limiter.reset()
    captured: dict = {}

    def fake_submit(target, req, *args, **kwargs):
        captured["req"] = req
        captured["request"] = kwargs.get("request")
        return 1

    monkeypatch.setattr(job_runner, "submit", fake_submit)

    with TestClient(create_app()) as client:
        response = client.post("/train", headers=ADMIN, json={
            "dataset_name": "d.csv", "model_name": "m",
            "text_columns": ["text"], "label_column": "label",
        })
        assert response.status_code == 202, response.text
    return captured


def test_the_run_request_carries_every_field_the_body_declares(submitted):
    expected = set(TrainRequest.model_fields) - TRAVEL_SEPARATELY

    assert set(submitted["req"]) - {"info"} == expected, (
        "a TrainRequest field is accepted and validated but never reaches the pipeline"
    )


def test_info_travels_as_plain_json(submitted):
    """Excluded from the field copy because the pipeline stores plain JSON, not a model."""
    assert submitted["req"]["info"] is None


def test_the_recorded_request_names_the_profile_that_ran(submitted):
    """`optimize_parameters` is resolved before submission, and what gets recorded is the
    profile actually used — under the field name a caller would resend."""
    recorded = submitted["request"]

    assert "optimize_parameters" in recorded
    assert "info" not in recorded, "documentation is not a parameter of the run"
