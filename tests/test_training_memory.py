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


def _train(tmp_path, on_progress=lambda **_: None, *, cv_folds: int = 0) -> Settings:
    """One real run of the tiny fixture, under the fixture's name `tiny_model`."""
    settings = _settings(tmp_path)
    config = _config()
    config.cv_folds = cv_folds
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


def _spy_on_head_fits(monkeypatch) -> list:
    """Record the n_jobs every head fit is built with, on both fit sites."""
    from app import deploy as deploy_mod
    from app import tuning as tuning_mod

    seen: list = []

    def make_spy(real):
        def spy(c, **kwargs):
            seen.append(kwargs.get("n_jobs"))
            return real(c, **kwargs)
        return spy

    monkeypatch.setattr(deploy_mod, "make_head", make_spy(deploy_mod.make_head))
    monkeypatch.setattr(tuning_mod, "make_head", make_spy(tuning_mod.make_head))
    return seen


@pytest.mark.parametrize("cv_folds", [0, 2], ids=["holdout", "cross-validation"])
def test_a_run_over_its_memory_budget_fits_one_head_at_a_time(tmp_path, monkeypatch, cv_folds):
    """The OOM-killed run's situation: the CPU budget would allow several threads, the
    memory budget does not. Every fit — C search and deploy fit, holdout and CV — must
    then drop to one thread instead of multiplying the matrix-sized solver buffers."""
    seen = _spy_on_head_fits(monkeypatch)
    monkeypatch.setattr(Settings, "effective_n_jobs", lambda self: 3)
    # 1 MiB: less than the process already holds, so no fit has any headroom.
    monkeypatch.setattr(Settings, "effective_train_memory_bytes", lambda self: 1024 * 1024)

    settings = _train(tmp_path, cv_folds=cv_folds)

    assert seen and set(seen) == {1}, seen
    metrics = json.loads((settings.models_dir / "tiny_model" / "metrics.json")
                         .read_text(encoding="utf-8"))
    assert metrics["resources"]["train_memory_budget_mb"] == 1
    assert metrics["resources"]["head_fit_threads"] == {"requested": 3, "min": 1, "max": 1}


def test_without_a_memory_budget_every_fit_keeps_the_cpu_threads(tmp_path, monkeypatch):
    seen = _spy_on_head_fits(monkeypatch)
    monkeypatch.setattr(Settings, "effective_n_jobs", lambda self: 3)
    monkeypatch.setattr(Settings, "effective_train_memory_bytes", lambda self: None)

    settings = _train(tmp_path)

    assert seen and set(seen) == {3}, seen
    resources = json.loads((settings.models_dir / "tiny_model" / "metrics.json")
                           .read_text(encoding="utf-8"))["resources"]
    assert resources["train_memory_budget_mb"] is None
    assert resources["head_fit_threads"] == {"requested": 3, "min": 3, "max": 3}


@pytest.mark.parametrize("cv_folds", [0, 2], ids=["holdout", "cross-validation"])
def test_a_throttled_run_says_so_in_its_progress(tmp_path, monkeypatch, cv_folds):
    """Fewer threads is slower: the status line has to explain a run that crawls."""
    monkeypatch.setattr(Settings, "effective_n_jobs", lambda self: 3)
    monkeypatch.setattr(Settings, "effective_train_memory_bytes", lambda self: 1024 * 1024)
    details: list[str] = []

    def on_progress(**fields):
        if fields.get("phase_detail"):
            details.append(fields["phase_detail"])

    _train(tmp_path, on_progress, cv_folds=cv_folds)

    search = [d for d in details if "C=" in d]
    assert search and all(d.endswith("· 1 thread") for d in search), details
    assert any("final model" in d and "1 thread" in d for d in details), details


def test_every_phase_leaves_a_memory_line_in_the_log(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="api_v3.training"):
        _train(tmp_path)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("phase=")]
    phases = [line.split()[0] for line in lines]
    assert {"phase=loading", "phase=features", "phase=saving"} <= set(phases), lines
    assert all(" rss=" in line and " peak=" in line for line in lines), lines
