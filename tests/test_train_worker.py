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
import os
import subprocess
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
    # The child counts this process' memory against the budget they share.
    assert spec["parent_pid"] == os.getpid()

    req, child_settings, child_config, profile = read_job(json.loads(text))
    assert req == _request()
    assert child_config == config
    assert profile == config.get("fast")
    assert child_settings.model_dump(exclude=SECRETS) == settings.model_dump(exclude=SECRETS)


def test_relative_paths_mean_the_same_place_in_the_child(tmp_path, monkeypatch):
    """The child runs in the package root, not in the API's working directory: a relative
    APIV3_MODELS_DIR named another place there, and the child staged where the parent
    never publishes from."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_dir=Path("data"), models_dir=Path("models"), auth_enabled=False)
    config = _config()
    spec = job_spec(_request(), settings, config, config.get("fast"))
    assert spec["settings"]["models_dir"] == str((tmp_path / "models").resolve())
    assert spec["settings"]["data_dir"] == str((tmp_path / "data").resolve())


def test_the_child_holds_no_api_key_not_even_from_its_environment(tmp_path, monkeypatch):
    """Leaving the keys out of the spec was not enough: the child inherited the
    environment, and its Settings read APIV3_API_KEY_* from there (and from .env) again."""
    monkeypatch.setenv("APIV3_API_KEY_ADMIN", "env-admin-secret")
    monkeypatch.setenv("APIV3_API_KEY_READONLY", "env-readonly-secret")
    _, child_settings, _, _ = read_job(_spec(tmp_path))
    assert child_settings.api_key_admin is None and child_settings.api_key_readonly is None

    env = train_worker._child_env()
    assert not [name for name in env if name.upper().startswith("APIV3_API_KEY")]
    assert env.get("PATH") == os.environ.get("PATH"), "everything else is inherited"


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
    # A child that never answers must fail this suite, not hang it.
    kwargs.setdefault("kill_requested", _deadline(180))
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
    # Reaped, its id is released before the publish waits for the disk lock: the status
    # must not add whatever process gets that id next.
    pids = [fields["worker_pid"] for fields in progress if "worker_pid" in fields]
    assert isinstance(pids[0], int) and pids[-1] is None
    prediction = registry.get("tiny_model").predict(["Das Römische Reich der Antike"], top_k=1)
    assert prediction[0][0].uri == "uri:hist"


def test_a_stop_ends_the_child_run_without_a_model(tmp_path):
    result, registry, _ = _run(tmp_path, should_stop=lambda: True)
    assert result == {}
    assert not registry.exists("tiny_model") and not _staging(registry)


def test_a_kill_ends_the_child_at_once(tmp_path, monkeypatch):
    """At once means the child hears the closed pipe and goes — not that the parent waits
    out the grace period and kills it. With a grace of a minute, the run still ends in
    seconds only if the child ended itself."""
    monkeypatch.setattr(train_worker, "_EXIT_GRACE_SECONDS", 60)
    started = time.monotonic()
    result, registry, _ = _run(tmp_path, should_stop=lambda: False, kill_requested=lambda: True)
    assert result == {}
    assert not registry.exists("tiny_model") and not _staging(registry)
    assert time.monotonic() - started < 8


def test_a_child_that_dies_without_an_answer_says_so(tmp_path, monkeypatch):
    """What an OOM kill looks like from the parent: the child is gone, no result line.
    The job must end with a reason the operator can act on, not a silent nothing."""
    monkeypatch.setattr(train_worker, "_worker_command", lambda: [
        sys.executable, "-c", "import os, sys; sys.stdin.readline(); os._exit(9)"])
    with pytest.raises(TrainingProcessError) as caught:
        _run(tmp_path, should_stop=lambda: False)
    assert "exit code 9" in str(caught.value)


def _deadline(seconds: float):
    """A kill_requested that fires after ``seconds``: a relay that would wait forever ends
    as a failed assertion instead of a hung suite."""
    end = time.monotonic() + seconds
    return lambda: time.monotonic() > end


# The child's bytes as the escapes of a Python bytes literal, written into its source.
@pytest.mark.parametrize("output", ["\\xff\\xfe not utf-8\\n", "0\\n", "null\\n"],
                         ids=["not-utf8", "a-number", "null"])
def test_output_that_is_no_message_cannot_hide_why_the_child_ended(tmp_path, monkeypatch, output):
    """Bytes that are not UTF-8 killed the reader thread and left the relay waiting for a
    line that never came; JSON that is not an object crashed the relay itself. Either way
    the run must still end with the child's exit code."""
    code = ("import os, sys; sys.stdin.readline(); "
            f"sys.stdout.buffer.write(b'{output}'); sys.stdout.flush(); os._exit(9)")
    monkeypatch.setattr(train_worker, "_worker_command", lambda: [sys.executable, "-c", code])
    with pytest.raises(TrainingProcessError, match="exit code 9"):
        _run(tmp_path, should_stop=lambda: False, kill_requested=_deadline(20))


def test_a_failure_in_the_parent_leaves_no_staged_bundle_behind(tmp_path):
    """Whatever ends the relay early — here the progress callback itself, halfway through
    the save — the child is stopped and what it staged is removed once it is gone, not
    left for the next start's sweep."""
    settings, config = _settings(tmp_path), _config()
    registry = _registry(settings)

    def on_progress(**fields):
        if fields.get("phase_detail") == "Writing head.skops":
            raise RuntimeError("the status store failed")

    with pytest.raises(RuntimeError, match="status store"):
        run_in_child(_request(), settings, config, config.get("fast"), registry,
                     on_progress=on_progress, should_stop=lambda: False)
    assert not _staging(registry)


def test_the_child_asks_to_be_the_one_the_oom_killer_takes(tmp_path):
    """Without a hint the kernel kills the LARGEST process, and with a big model cache that
    can be the API. The child raises its own oom_score_adj (no privilege needed to raise
    it); where there is no such file — Windows, macOS — nothing happens."""
    score = tmp_path / "oom_score_adj"
    score.write_text("0")
    train_worker._volunteer_for_the_oom_killer(score)
    assert score.read_text() == "1000"
    train_worker._volunteer_for_the_oom_killer(tmp_path / "no" / "such" / "file")


@pytest.mark.parametrize(("code", "oom"), [(-9, True), (137, True), (1, False), (3, False)])
def test_only_a_kill_by_signal_9_is_called_a_likely_oom(code, oom):
    """Ctrl+C on a dev server ends the child with code 1: calling that an out-of-memory
    kill would send the operator after memory that was never short."""
    message = train_worker._no_result_message(code)
    assert f"exit code {code}" in message
    assert ("out-of-memory" in message) is oom


def test_serve_shows_every_user_facing_error_as_it_is(tmp_path, monkeypatch):
    """The job runner shows every UserFacingError verbatim; the child must not sanitize a
    kind the thread path would have shown."""
    from app import training
    from app.errors import UserFacingError

    class Refused(UserFacingError):
        pass

    def refuse(*args, **kwargs):
        raise Refused("Model 'x' cannot be trained: the reason for the operator.")

    monkeypatch.setattr(training, "run_training", refuse)
    messages: list[dict] = []
    assert serve(_spec(tmp_path), messages.append, should_stop=lambda: False) == 1
    assert messages[-1]["failed"] == {
        "message": "Model 'x' cannot be trained: the reason for the operator.", "user_facing": True}


def _start_worker(tmp_path) -> subprocess.Popen:
    """The real worker, driven by hand: the job sent, its first line (worker_pid) read."""
    child = subprocess.Popen(train_worker._worker_command(), cwd=train_worker._APP_ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             encoding="utf-8")
    assert child.stdin is not None and child.stdout is not None
    child.stdin.write(json.dumps({"job": _spec(tmp_path)}) + "\n")
    child.stdin.flush()
    assert "worker_pid" in json.loads(child.stdout.readline())["progress"]
    return child


def _ended_alone(child: subprocess.Popen, within: float) -> None:
    """The child ended by itself, promptly, with the end-of-input code."""
    started = time.monotonic()
    try:
        assert child.wait(timeout=60) == 3
        assert time.monotonic() - started < within
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_a_child_whose_parent_is_gone_ends_at_once(tmp_path):
    """What an API process that died looks like from the child: its end of the pipe is
    closed. Nothing else would stop a training nobody is waiting for any more."""
    child = _start_worker(tmp_path)
    child.stdin.close()
    _ended_alone(child, within=10)


def test_a_line_the_child_cannot_read_does_not_cost_it_its_stop_listener(tmp_path):
    """The stop listener is also what ends a child whose parent is gone. A line it cannot
    parse killed it, and the end of input that followed went unheard: the child trained
    on to the end (exit 0) instead of ending at once (exit 3)."""
    child = _start_worker(tmp_path)
    child.stdin.write("not a message\n")
    child.stdin.close()
    _ended_alone(child, within=10)


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
        # Proof that the model came from a CHILD: an in-process run would have put it in
        # the cache on the way past; a published bundle is loaded on first use.
        assert get_registry().in_memory_count() == 0
    finally:
        get_settings.cache_clear()
        get_registry.cache_clear()
