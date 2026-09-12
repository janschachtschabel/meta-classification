"""What a training run holds in memory, and what it reports about it.

A run on the full WLO export was OOM-killed on 2026-09-11 inside a phase that logs
nothing, so neither the status nor the log could say how close it had come, and nothing
recorded what the next run would need. These tests pin the reporting (live and peak RSS
in the status, the peak in the bundle and in the job history, one log line per phase),
the head-fit threads a memory budget allows, and that no phase keeps the previous
phase's matrices alive while it builds its own.
"""

import json
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

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
def test_the_peak_is_never_below_the_memory_shown_beside_it():
    """`rss_mb` is read when the status is asked for, `peak_rss_mb` travels with progress
    updates — and inside a head fit the newest update can be minutes old. The card then
    showed a peak BELOW the figure next to it. The runner keeps its own maximum of what
    it reads, which is also the peak between two progress updates."""
    job = JobRunner()
    reported, release = threading.Event(), threading.Event()

    def target(*, on_progress, should_stop):
        on_progress(phase="features", peak_rss_mb=1)  # a peak from long ago
        reported.set()
        release.wait(5)
        return {}

    job.start(target, model_name="m")
    try:
        assert reported.wait(2), "target never reported"
        running = job.snapshot()
    finally:
        release.set()
    assert running["rss_mb"] > 1
    assert running["peak_rss_mb"] >= running["rss_mb"]


@needs_rss
def test_the_status_counts_the_training_process_memory_without_showing_its_id():
    """In a child process the training's memory is not the API's: the status adds the
    worker's RSS, because the container's limit applies to the sum."""
    job = JobRunner()
    idle = job.snapshot()["rss_mb"]
    reported, release = threading.Event(), threading.Event()

    def target(*, on_progress, should_stop):
        on_progress(worker_pid=os.getpid())  # stands in for a child: this very process
        reported.set()
        release.wait(5)
        return {}

    job.start(target, model_name="m")
    try:
        assert reported.wait(2), "target never reported"
        running = job.snapshot()
    finally:
        release.set()
    assert "worker_pid" not in running
    assert running["rss_mb"] >= 1.5 * idle


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


@pytest.mark.parametrize("cv_folds", [0, 2], ids=["holdout", "cross-validation"])
def test_the_status_shows_the_threads_a_fit_runs_on_before_it_runs(
    tmp_path, monkeypatch, cv_folds
):
    """A head fit on 300k rows runs for minutes: the thread count has to be on screen
    while it runs, not only in the line that reports it finished."""
    monkeypatch.setattr(Settings, "effective_n_jobs", lambda self: 3)
    monkeypatch.setattr(Settings, "effective_train_memory_bytes", lambda self: 1024 * 1024)
    updates: list[dict] = []
    _train(tmp_path, lambda **fields: updates.append(fields), cv_folds=cv_folds)

    shown = [u for u in updates if "head_fit_threads" in u]
    assert shown and all(u["head_fit_threads"] == 1 and u["threads_requested"] == 3
                         for u in shown), shown
    first_fit_done = next(i for i, u in enumerate(updates)
                          if "C=" in str(u.get("phase_detail", "")))
    assert updates.index(shown[0]) < first_fit_done, "shown only after the fit"


def test_an_idle_status_has_no_thread_count():
    snapshot = JobRunner().snapshot()
    assert snapshot["head_fit_threads"] is None
    assert snapshot["threads_requested"] is None


class _RecordingVectorizer:
    """Row ids as the only feature (what the scripted heads read back), and a weak
    reference to everything it builds — so a test can ask what is still alive."""

    built: list = []

    def fit_transform(self, texts):
        _RecordingVectorizer.built.append(weakref.ref(self))
        return self.transform(texts)

    def transform(self, texts):
        matrix = sparse.csr_matrix(np.array([[float(text)] for text in texts]))
        _RecordingVectorizer.built.append(weakref.ref(matrix))
        return matrix


def _alive() -> list:
    return [ref for ref in _RecordingVectorizer.built if ref() is not None]


def test_a_fold_releases_its_matrices_before_the_next_fold_vectorizes(scripted_c_search):
    """Fold k's matrices and vectorizer used to stay bound until fold k+1's were
    assigned — so the next fold's vectorization peak sat on top of them."""
    from app.tuning import cross_val_evaluate

    _RecordingVectorizer.built = []
    alive_when_fitting: list[int] = []

    class Checked(_RecordingVectorizer):
        def fit_transform(self, texts):
            alive_when_fitting.append(len(_alive()))
            return super().fit_transform(texts)

    texts = [str(i) for i in range(8)]
    result = cross_val_evaluate(Checked, texts, scripted_c_search.y, ["c0", "c1"],
                                k=2, c_grid=[1.0, 2.0], seed=0)

    assert result is not None
    assert alive_when_fitting == [0, 0], "a previous fold's objects were still alive"


def test_the_shared_matrix_is_released_before_the_deploy_fit_vectorizes(
    scripted_c_search, monkeypatch
):
    """With one matrix shared across the folds, that matrix is dead weight once the
    folds are done — it must not sit under the deploy vectorization's peak."""
    from app import deploy
    from app.prepare import Prepared
    from app.profiles import Profile

    _RecordingVectorizer.built = []
    alive_when_fitting: list[int] = []

    class Checked(_RecordingVectorizer):
        def fit_transform(self, texts):
            alive_when_fitting.append(len(_alive()))
            return super().fit_transform(texts)

    monkeypatch.setattr(deploy, "TfidfBackend", lambda **kwargs: Checked())
    everything = np.arange(8)
    prep = Prepared(
        texts=np.array([str(i) for i in everything]), y_all=scripted_c_search.y,
        classes=["c0", "c1"], task_type="multilabel", avg_labels=1.0, min_samples=1,
        text_column_weights={}, train_idx=everything, val_idx=everything,
        test_idx=everything, uri_to_label={"c0": "c0", "c1": "c1"},
    )
    profile = Profile("t", c_grid=[1.0, 2.0], cv_folds=2, refit_vectorizer_per_fold=False)
    deploy.fit_evaluate_deploy(prep, Settings(), profile, cv_folds=2,
                               on_progress=lambda **_: None, should_stop=lambda: False)

    assert len(alive_when_fitting) == 2  # the shared matrix, then the deploy fit
    assert alive_when_fitting[-1] == 0, "the shared matrix outlived the folds"


def test_every_phase_leaves_a_memory_line_in_the_log(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="api_v3.training"):
        _train(tmp_path)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("phase=")]
    phases = [line.split()[0] for line in lines]
    assert {"phase=loading", "phase=features", "phase=saving"} <= set(phases), lines
    assert all(" rss=" in line and " peak=" in line for line in lines), lines
