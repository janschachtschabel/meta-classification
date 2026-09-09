"""Queueing training runs on the server instead of in a browser tab.

"Train five label fields overnight" needed an open tab: the queue lived in the page,
and closing it — or a laptop going to sleep — lost every run that had not started yet.

Timing here is driven by Events, never by sleeps: a test that waits for a real
training would be both slow and flaky.
"""

import threading

import pytest

from app.jobs import MAX_QUEUED, JobRunner


def _blocking_target(started: threading.Event, release: threading.Event):
    def target(*, on_progress, should_stop):
        started.set()
        release.wait(5)
        return {"model_name": "x", "metrics": {}}

    return target


def _finished_target(done: list, name: str):
    def target(*, on_progress, should_stop):
        done.append(name)
        return {"model_name": name, "metrics": {}}

    return target


def test_a_run_submitted_while_idle_starts_immediately():
    job = JobRunner()
    started, release = threading.Event(), threading.Event()

    assert job.submit(_blocking_target(started, release), model_name="first") == 0
    assert started.wait(2)
    assert job.snapshot()["status"] == "running"
    release.set()


def test_a_second_run_is_queued_rather_than_refused():
    """Before, this was a 409 and the browser had to retry it later. The position is
    what the caller gets instead: accepted, and this is where it sits."""
    job = JobRunner()
    started, release = threading.Event(), threading.Event()
    job.submit(_blocking_target(started, release), model_name="first")
    assert started.wait(2)

    assert job.submit(lambda **_: None, model_name="second") == 1
    assert job.submit(lambda **_: None, model_name="third") == 2
    assert job.snapshot()["queued"] == ["second", "third"]
    release.set()


def test_the_queue_runs_in_order_after_the_first_finishes():
    """The point of the whole thing: the runs happen without anybody asking again."""
    job = JobRunner()
    started, release = threading.Event(), threading.Event()
    done: list[str] = []
    job.submit(_blocking_target(started, release), model_name="first")
    assert started.wait(2)
    job.submit(_finished_target(done, "second"), model_name="second")
    job.submit(_finished_target(done, "third"), model_name="third")

    release.set()
    for _ in range(500):
        if len(done) == 2:
            break
        threading.Event().wait(0.01)
    assert done == ["second", "third"]
    assert job.snapshot()["queued"] == []


def test_the_queue_is_bounded():
    """Unbounded, one script could enqueue a thousand overnight runs and the operator
    would have no way back except restarting the process."""
    job = JobRunner()
    started, release = threading.Event(), threading.Event()
    job.submit(_blocking_target(started, release), model_name="first")
    assert started.wait(2)

    for index in range(MAX_QUEUED):
        job.submit(lambda **_: None, model_name=f"queued_{index}")
    with pytest.raises(RuntimeError, match="queue is full"):
        job.submit(lambda **_: None, model_name="one_too_many")
    release.set()


def test_stopping_clears_the_queue():
    """"Stop" means "I want this to end", not "skip to the next one" — and that is what
    the browser-side queue did too, so the server keeps the promise the UI made."""
    job = JobRunner()
    started, release = threading.Event(), threading.Event()
    done: list[str] = []
    job.submit(_blocking_target(started, release), model_name="first")
    assert started.wait(2)
    job.submit(_finished_target(done, "second"), model_name="second")

    job.stop()
    assert job.snapshot()["queued"] == []
    release.set()
    for _ in range(200):
        if job._thread is not None and not job._thread.is_alive():
            break
        threading.Event().wait(0.01)
    assert done == [], "a stopped queue starts nothing else"


def test_a_name_that_is_already_running_or_queued_is_refused():
    """Two runs under one name is a run guaranteed to fail: /train refuses an existing
    model, so the second would be started only to die. Refuse it while it is still a
    request, when the caller can still do something about it."""
    job = JobRunner()
    started, release = threading.Event(), threading.Event()
    job.submit(_blocking_target(started, release), model_name="subjects")
    assert started.wait(2)

    with pytest.raises(RuntimeError, match="already"):
        job.submit(lambda **_: None, model_name="subjects")
    job.submit(lambda **_: None, model_name="other")
    with pytest.raises(RuntimeError, match="already"):
        job.submit(lambda **_: None, model_name="other")
    release.set()
