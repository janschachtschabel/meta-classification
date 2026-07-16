"""Single background training job with thread-safe state and cooperative stop.

Only one training runs at a time (the typical single-instance deployment). The
training function receives ``on_progress`` and ``should_stop`` callbacks so the
orchestration in ``training.py`` stays free of threading concerns.

The status snapshot reports the current phase/progress/message plus a rough
``eta_seconds`` and ``elapsed_seconds`` derived from progress (estimate).
``seconds_since_heartbeat`` is the age of the newest progress update: unlike
``elapsed_seconds`` (which grows regardless) it exposes a training thread that
went silent — hung, or crawling through a page-file-thrashing save.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from .errors import TrainingInputError

logger = logging.getLogger("api_v3.jobs")


def _idle_state() -> dict:
    return {
        "status": "idle",
        "progress": 0,
        "phase": "",
        "phase_detail": None,
        "message": "",
        "started_at": None,
        "elapsed_seconds": None,
        "eta_seconds": None,
        "seconds_since_heartbeat": None,
        "model_name": None,
        "results": None,
        "error": None,
    }


class TrainingJob:
    """Owns the background thread and the shared, lock-protected state dict."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _idle_state()
        self._start_ts: float | None = None
        # Monotonic time of the newest progress update — the liveness signal.
        self._heartbeat_ts: float | None = None
        # Bumped on start and on hard reset: a finishing zombie thread whose
        # generation is stale must not overwrite the current state.
        self._generation = 0

    def snapshot(self) -> dict:
        """Return a copy of the state, adding elapsed/ETA while running."""
        with self._lock:
            state = dict(self._state)
            if state["status"] == "running" and self._start_ts is not None:
                elapsed = time.monotonic() - self._start_ts
                state["elapsed_seconds"] = round(elapsed, 1)
                progress = state["progress"]
                if 0 < progress < 100:
                    state["eta_seconds"] = round(elapsed * (100 - progress) / progress, 1)
            if state["status"] == "running" and self._heartbeat_ts is not None:
                state["seconds_since_heartbeat"] = round(time.monotonic() - self._heartbeat_ts, 1)
            return state

    def is_running(self) -> bool:
        with self._lock:
            return self._state["status"] == "running"

    def update(self, **fields: object) -> None:
        with self._lock:
            # Every progress signal from the training thread doubles as a
            # liveness heartbeat; snapshot() reports the age of the newest one.
            self._heartbeat_ts = time.monotonic()
            self._apply(fields)

    def _apply(self, fields: dict) -> None:
        """Write state fields; the caller must hold ``self._lock``."""
        # Entering a new phase clears stale sub-step detail.
        if "phase" in fields and "phase_detail" not in fields:
            self._state["phase_detail"] = None
        self._state.update(fields)

    def _finish(self, generation: int, **fields: object) -> None:
        """Final status update from the runner thread — dropped if a hard stop
        (or a newer start) bumped the generation while the target was running."""
        with self._lock:
            if self._generation != generation:
                return
            self._apply(fields)

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def start(self, target: Callable, *args: object, model_name: str) -> None:
        """Launch ``target(*args, on_progress=..., should_stop=...)`` in a thread."""
        with self._lock:
            # Refuse while a previous thread is still executing — including one
            # abandoned by stop(hard=True), whose deploy fit / skops save keep
            # running after the status was reset to idle. Starting anyway would
            # run two full trainings at once, breaking the single-worker design.
            if self._state["status"] == "running" or (
                self._thread is not None and self._thread.is_alive()
            ):
                raise RuntimeError(
                    "A training job is already running (or a hard-stopped one is "
                    "still finishing in the background); retry once it completes."
                )
            self._stop.clear()
            self._state = _idle_state()
            self._state.update(
                status="running",
                phase="starting",
                model_name=model_name,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._start_ts = time.monotonic()
            # A run that hangs before its first progress update must still show
            # a growing heartbeat age, so the clock starts at launch.
            self._heartbeat_ts = self._start_ts
            self._generation += 1
            generation = self._generation

        def runner() -> None:
            try:
                results = target(*args, on_progress=self.update, should_stop=self.should_stop)
                if self.should_stop():
                    self._finish(generation, status="stopped", message="Training stopped.",
                                 eta_seconds=0)
                else:
                    self._finish(generation, status="completed", progress=100, phase="done",
                                 results=results, eta_seconds=0)
            except TrainingInputError as exc:
                # Crafted, user-facing message (wrong column, too few rows, ...):
                # showing it is the point — hiding it would mask the user's own typo.
                logger.warning("Training job %r rejected: %s", model_name, exc)
                self._finish(generation, status="error", phase="error", message=str(exc))
            except Exception:  # noqa: BLE001 - report any failure as job error
                # Log the full traceback server-side; never leak exception text
                # (which can carry paths/data internals) to the readonly status.
                logger.exception("Training job %r failed", model_name)
                self._finish(generation, status="error", phase="error",
                             message="Training failed; see server logs for details.")

        thread = threading.Thread(target=runner, daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()

    def stop(self, *, hard: bool = False) -> None:
        self._stop.set()
        if hard:
            with self._lock:
                # Invalidate the running thread's generation so its eventual
                # completion cannot overwrite this reset.
                self._generation += 1
                self._apply(dict(status="idle", phase="", message="Hard stopped.",
                                 progress=0, results=None))


# Module-level singleton used by the API routes.
training_job = TrainingJob()
