"""Training in a child process: the API process never holds a run's memory.

A run in a thread of the API process leaves its memory behind in the process that serves
requests, and an OOM kill takes the whole API down with it. In a child process every
byte goes back to the OS when the run ends, and a kill ends the run, not the server.
What is pinned here: the job spec crosses the boundary intact and without secrets; the
worker stages the bundle and the parent publishes it under its own disk lock; a stop, a
kill and a child that dies each end the run the way the job runner expects. (Staging
itself: tests/test_registry_staging.py; the runner's side: tests/test_job_queue.py and
tests/test_training_memory.py.)
"""

import json
import sys
import time
from pathlib import Path

import pytest

from app import train_worker
from app.errors import TrainingProcessError
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.train_worker import job_spec, read_job, run_in_child, serve

FIXTURES = Path(__file__).parent / "fixtures"
SECRETS = {"api_key_admin", "api_key_readonly"}


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=FIXTURES, models_dir=tmp_path / "models", auth_enabled=False,
                    api_key_admin="admin-secret", api_key_readonly="readonly-secret")


def _config() -> TrainingConfig:
    profile = Profile("fast", "TF-IDF", True, True, [1.0, 2.0])
    return TrainingConfig(
        default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
        test_size=0.2, min_text_length=5, drop_duplicates=True, min_samples_per_label=2,
    )


def _request(**overrides) -> dict:
    return {
        "dataset_name": "tiny.csv", "model_name": "tiny_model",
        "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
        "label_column": "properties.ccm:taxonid", "csv_separator": ";",
        "label_separator": ",", "label_filter": None, **overrides,
    }


def _registry(settings: Settings) -> Registry:
    return Registry(settings.models_dir, settings.max_models_in_memory)


def _spec(tmp_path, **overrides) -> dict:
    """What the parent sends, after the JSON round trip the pipe puts it through."""
    config = _config()
    return json.loads(json.dumps(job_spec(_request(**overrides), _settings(tmp_path), config,
                                          config.get("fast"))))


def test_the_job_spec_crosses_the_boundary_intact_and_without_secrets(tmp_path):
    settings, config = _settings(tmp_path), _config()
    spec = job_spec(_request(), settings, config, config.get("fast"))
    text = json.dumps(spec)
    assert "admin-secret" not in text and "readonly-secret" not in text

    req, child_settings, child_config, profile = read_job(json.loads(text))
    assert req == _request()
    assert child_config == config
    assert profile == config.get("fast")
    assert child_settings.model_dump(exclude=SECRETS) == settings.model_dump(exclude=SECRETS)


def test_the_worker_stages_a_bundle_that_only_the_parent_publishes(tmp_path):
    messages: list[dict] = []
    assert serve(_spec(tmp_path), messages.append, should_stop=lambda: False) == 0

    assert any("progress" in message for message in messages)
    done = messages[-1]["done"]
    assert done["staged"] is True
    assert done["result"]["model_name"] == "tiny_model"
    registry = _registry(_settings(tmp_path))
    assert not registry.exists("tiny_model"), "staged is not published"
    registry.publish("tiny_model")
    prediction = registry.get("tiny_model").predict(["Bruchrechnung und Gleichungen"], top_k=1)
    assert prediction[0][0].uri == "uri:math"


def test_an_input_error_in_the_worker_reaches_the_operator_verbatim(tmp_path):
    messages: list[dict] = []
    assert serve(_spec(tmp_path, label_column="no_such_column"), messages.append,
                 should_stop=lambda: False) == 1
    failed = messages[-1]["failed"]
    assert failed["user_facing"] is True
    assert "no_such_column" in failed["message"]


def _run(tmp_path, settings: Settings | None = None, **kwargs) -> tuple[dict, Registry, list[dict]]:
    settings, config = settings or _settings(tmp_path), _config()
    registry = _registry(settings)
    progress: list[dict] = []
    result = run_in_child(_request(), settings, config, config.get("fast"), registry,
                          on_progress=lambda **fields: progress.append(fields), **kwargs)
    return result, registry, progress


def _staging(registry: Registry) -> list[Path]:
    return [p for p in registry.dir.iterdir() if p.name.startswith(".")] if registry.dir.exists() else []


def test_a_run_in_a_child_process_publishes_a_model_the_api_can_load(tmp_path):
    result, registry, progress = _run(tmp_path, should_stop=lambda: False)

    assert result["model_name"] == "tiny_model"
    assert registry.exists("tiny_model") and not _staging(registry)
    assert any(fields.get("peak_rss_mb") for fields in progress), "progress crossed over"
    prediction = registry.get("tiny_model").predict(["Das Römische Reich der Antike"], top_k=1)
    assert prediction[0][0].uri == "uri:hist"


def test_a_stop_ends_the_child_run_without_a_model(tmp_path):
    result, registry, _ = _run(tmp_path, should_stop=lambda: True)
    assert result == {}
    assert not registry.exists("tiny_model") and not _staging(registry)


def test_a_kill_ends_the_child_at_once(tmp_path):
    started = time.monotonic()
    result, registry, _ = _run(tmp_path, should_stop=lambda: False, kill_requested=lambda: True)
    assert result == {}
    assert not registry.exists("tiny_model") and not _staging(registry)
    assert time.monotonic() - started < 30


def test_a_child_that_dies_without_an_answer_says_so(tmp_path, monkeypatch):
    """What an OOM kill looks like from the parent: the child is gone, no result line.
    The job must end with a reason the operator can act on, not a silent nothing."""
    monkeypatch.setattr(train_worker, "_worker_command", lambda: [
        sys.executable, "-c", "import os, sys; sys.stdin.readline(); os._exit(9)"])
    with pytest.raises(TrainingProcessError) as caught:
        _run(tmp_path, should_stop=lambda: False)
    assert "exit code 9" in str(caught.value)


def test_a_child_killed_in_the_middle_of_a_line_still_says_why(tmp_path, monkeypatch):
    """The OOM killer does not wait for a line to end. Half a message must not turn the
    run's real cause into a parse error nobody can act on."""
    monkeypatch.setattr(train_worker, "_worker_command", lambda: [
        sys.executable, "-c",
        "import os, sys; sys.stdin.readline(); sys.stdout.write('{\"progress\": {\"pha');"
        " sys.stdout.flush(); os._exit(9)"])
    with pytest.raises(TrainingProcessError) as caught:
        _run(tmp_path, should_stop=lambda: False)
    assert "exit code 9" in str(caught.value)


def test_the_child_takes_the_log_level_the_way_the_server_does(tmp_path):
    """The server upper-cases APIV3_LOG_LEVEL; `logging` itself refuses "info". A child
    that took the setting as written died before its first line."""
    settings = _settings(tmp_path).model_copy(update={"log_level": "info"})
    result, registry, _ = _run(tmp_path, settings=settings, should_stop=lambda: False)
    assert result["model_name"] == "tiny_model" and registry.exists("tiny_model")


def test_the_train_route_runs_a_training_in_a_child_process(tmp_path, monkeypatch):
    """The route's wiring, end to end, with the process isolation the server defaults to."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.registry import get_registry
    from app.settings import get_settings

    monkeypatch.setenv("APIV3_AUTH_ENABLED", "false")
    monkeypatch.setenv("APIV3_DATA_DIR", str(FIXTURES))
    monkeypatch.setenv("APIV3_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("APIV3_TRAINING_ISOLATION", "process")
    get_settings.cache_clear()
    get_registry.cache_clear()
    try:
        client = TestClient(create_app())
        # The route reads the real config.yaml: its label minimum (20) is more than the
        # fixture's 12 rows per label.
        body = {**_request(model_name="child_model"), "optimize_parameters": "fast",
                "min_samples_per_label": 2}
        body.pop("label_filter")
        assert client.post("/train", json=body).status_code == 202
        deadline = time.time() + 120
        status = client.get("/train/status").json()
        while time.time() < deadline and status["status"] == "running":
            time.sleep(0.2)
            status = client.get("/train/status").json()
        assert status["status"] == "completed", status
        assert "child_model" in client.get("/models").text
    finally:
        get_settings.cache_clear()
        get_registry.cache_clear()
