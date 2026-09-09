"""The record a finished training leaves behind.

Comparing two runs is the whole point: today that means opening two bundles and
reading their metrics.json, and a run that failed leaves nothing to open at all.
"""

import json

import pytest

from app import job_history


@pytest.fixture
def history(tmp_path, monkeypatch):
    """A history file of its own, so tests never touch the real one."""
    path = tmp_path / "jobs.jsonl"
    monkeypatch.setattr(job_history, "_history_path", lambda: path)
    return path


def _record(name: str, **fields) -> dict:
    return {"model_name": name, "status": "completed", "duration_seconds": 1.0, **fields}


def test_a_finished_run_is_readable_back_without_opening_a_bundle(history):
    job_history.append(_record("subjects_v1", f1_macro=0.74, n_labels=59))

    entries = job_history.recent()
    assert len(entries) == 1
    assert entries[0]["model_name"] == "subjects_v1"
    assert entries[0]["f1_macro"] == 0.74


def test_the_newest_run_comes_first(history):
    """The question a history answers is "what did I just run", not "what did I run
    first" — and the answer has to survive being read back from disk."""
    for name in ("first", "second", "third"):
        job_history.append(_record(name))

    assert [entry["model_name"] for entry in job_history.recent()] == ["third", "second", "first"]


def test_a_failed_run_is_recorded_too(history):
    """A failure is exactly the run with no bundle to inspect afterwards, so the
    history is the only place its reason can survive."""
    job_history.append(_record("broken", status="error", error="No label reaches 20 samples."))

    entry = job_history.recent()[0]
    assert entry["status"] == "error"
    assert entry["error"] == "No label reaches 20 samples."


def test_the_file_stays_bounded(history):
    """It is appended to forever on a long-lived server. Keeping the newest N is the
    behaviour that needs no operator to remember anything."""
    for index in range(job_history.MAX_RECORDS + 25):
        job_history.append(_record(f"run_{index}"))

    entries = job_history.recent(limit=1000)
    assert len(entries) == job_history.MAX_RECORDS
    assert entries[0]["model_name"] == f"run_{job_history.MAX_RECORDS + 24}", "newest kept"
    assert all(entry["model_name"] != "run_0" for entry in entries), "oldest dropped"


def test_the_history_outlives_the_process(history):
    """The point of writing it down: a restart, a deploy or a crash must not erase
    what was already run — the browser tab holding the queue was exactly the thing
    that could not be relied on."""
    job_history.append(_record("before_restart"))

    # A fresh module state is what a new process sees: nothing cached, only the file.
    entries = json.loads("[" + ",".join(history.read_text(encoding="utf-8").splitlines()) + "]")
    assert entries[0]["model_name"] == "before_restart"
    assert job_history.recent()[0]["model_name"] == "before_restart"


def test_a_damaged_history_does_not_take_anything_down(history):
    """It sits on a mounted volume next to the models. A half-written line must cost
    that line, not the endpoint that reads it — and never a training run, which only
    writes to it after the work is already done."""
    history.write_text('{"model_name": "good", "status": "completed"}\n{ truncated\n',
                       encoding="utf-8")

    entries = job_history.recent()
    assert [entry["model_name"] for entry in entries] == ["good"]

    job_history.append(_record("after_damage"))
    assert [entry["model_name"] for entry in job_history.recent()] == ["after_damage", "good"]


def test_recent_limits_what_it_returns(history):
    for index in range(10):
        job_history.append(_record(f"run_{index}"))

    assert len(job_history.recent(limit=3)) == 3
    assert job_history.recent(limit=3)[0]["model_name"] == "run_9"
