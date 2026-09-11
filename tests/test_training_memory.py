"""What a training run reports about its memory.

A run on the full WLO export was OOM-killed on 2026-09-11 inside a phase that logs
nothing, so neither the status nor the log could say how close it had come, and nothing
recorded what the next run would need. These tests pin the reporting: live and peak RSS
in the status, the peak in the bundle and in the job history, one log line per phase.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from app import job_history
from app.jobs import JobRunner
from app.profiles import Profile, TrainingConfig
from app.registry import Registry
from app.settings import Settings
from app.training import run_training

FIXTURES = Path(__file__).parent / "fixtures"
needs_rss = pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "win32"),
    reason="no current-RSS reading on this platform (app.memory.rss_bytes reads 0)",
)


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=FIXTURES, models_dir=tmp_path / "models", auth_enabled=False)


def _config() -> TrainingConfig:
    profile = Profile("fast", "TF-IDF", True, True, [1.0, 2.0])
    return TrainingConfig(
        default_profile="fast", profiles={"fast": profile}, validation_size=0.2,
        test_size=0.2, min_text_length=5, drop_duplicates=True, min_samples_per_label=2,
    )


def _train(tmp_path, on_progress=lambda **_: None) -> Settings:
    """One real run of the tiny fixture, under the fixture's name `tiny_model`."""
    settings = _settings(tmp_path)
    config = _config()
    request = {
        "dataset_name": "tiny.csv", "model_name": "tiny_model",
        "text_columns": ["properties.cclom:title", "properties.cclom:general_keyword"],
        "label_column": "properties.ccm:taxonid", "csv_separator": ";",
        "label_separator": ",", "label_filter": None,
    }
    run_training(request, settings, config, config.get("fast"),
                 Registry(settings.models_dir, settings.max_models_in_memory),
                 on_progress=on_progress, should_stop=lambda: False)
    return settings


def _wait_until_finished(job: JobRunner) -> dict:
    deadline = time.time() + 5
    while time.time() < deadline and job.snapshot()["status"] == "running":
        time.sleep(0.01)
    return job.snapshot()


@needs_rss
def test_status_reads_memory_live_and_carries_the_runs_peak():
    """`rss_mb` is read when the status is asked for — inside a head fit the last
    progress update can be minutes old. `peak_rss_mb` is what the run reported."""
    job = JobRunner()
    idle = job.snapshot()
    assert idle["rss_mb"] > 0
    assert idle["peak_rss_mb"] is None

    reported, release = threading.Event(), threading.Event()

    def target(*, on_progress, should_stop):
        on_progress(phase="features", peak_rss_mb=4321)
        reported.set()
        release.wait(5)
        return {}

    job.start(target, model_name="m")
    try:
        assert reported.wait(2), "target never reported"
        running = job.snapshot()
    finally:
        release.set()
    assert running["peak_rss_mb"] == 4321
    assert running["rss_mb"] > 0


@needs_rss
def test_training_reports_its_peak_while_running_and_in_the_bundle(tmp_path):
    peaks: list[int] = []

    def on_progress(**fields):
        if "peak_rss_mb" in fields:
            peaks.append(fields["peak_rss_mb"])

    settings = _train(tmp_path, on_progress)

    assert peaks and min(peaks) > 0
    assert peaks == sorted(peaks), "a peak can only grow"
    metrics = json.loads((settings.models_dir / "tiny_model" / "metrics.json")
                         .read_text(encoding="utf-8"))
    assert 0 < metrics["resources"]["peak_rss_mb"] <= peaks[-1]


def test_the_job_history_keeps_the_runs_peak(tmp_path, monkeypatch):
    """The answer to "will 8 GB be enough next time" — also for a run that failed."""
    monkeypatch.setattr(job_history, "_history_path", lambda: tmp_path / "jobs.jsonl")

    def target(*, on_progress, should_stop):
        on_progress(phase="features", peak_rss_mb=2048)
        raise MemoryError

    job = JobRunner()
    job.start(target, model_name="m")
    assert _wait_until_finished(job)["status"] == "error"
    assert job_history.recent()[0]["peak_rss_mb"] == 2048


def test_every_phase_leaves_a_memory_line_in_the_log(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="api_v3.training"):
        _train(tmp_path)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("phase=")]
    phases = [line.split()[0] for line in lines]
    assert {"phase=loading", "phase=features", "phase=saving"} <= set(phases), lines
    assert all(" rss=" in line and " peak=" in line for line in lines), lines
