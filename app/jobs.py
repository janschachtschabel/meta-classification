"""One background job at a time: thread-safe state, a bounded queue, cooperative stop.

Named for what it is rather than for its first user. It runs trainings today and will
run evaluations next; both are a long CPU-bound pass over a dataset, and on a
single-instance deployment exactly one of them may hold the CPU. Further submissions
wait in the queue and the finishing thread starts the next. That queue used to live in
the browser, so closing the tab lost every run that had not started yet.

The job function receives ``on_progress`` and ``should_stop`` callbacks so the
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
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime

from . import job_history
from .errors import TrainingInputError

logger = logging.getLogger("api_v3.jobs")

# How many runs may wait behind the running one. "Train five label fields overnight" is
# the case this exists for; unbounded, one script could enqueue a thousand and leave the
# operator no way out except restarting the process.
MAX_QUEUED = 10


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
        "kind": "training",
        "results": None,
        "error": None,
    }


class JobRunner:
    """Owns the background thread, the queue, and the shared lock-protected state."""

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
        # Model name of the most recently started run — read via
        # active_model_name() while the runner thread is alive (hard stop resets
        # the STATUS, but the abandoned thread keeps writing this bundle).
        self._last_model_name: str | None = None
        # What the run was asked to do, kept for its history entry.
        self._last_request: dict | None = None
        # Runs accepted while another one holds the thread. Server-side on purpose:
        # the browser used to hold this list, so closing the tab lost every run that
        # had not started. Guarded by the same lock as the state.
        self._queue: deque[tuple] = deque()

    def snapshot(self) -> dict:
        """Return a copy of the state, adding elapsed/ETA while running.

        Carries the queued run names too: what is waiting is part of "what is going on
        here", and the status endpoint is where anyone looks for that.
        """
        with self._lock:
            state = dict(self._state)
            state["queued"] = [entry[2] for entry in self._queue]
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

    def _update_if_current(self, generation: int, **fields: object) -> None:
        """Progress update from the runner thread — dropped after a hard stop (or
        a newer start) bumped the generation, so an abandoned zombie thread can
        no longer mutate the reset state (/metrics would show running=0 with
        progress creeping otherwise)."""
        with self._lock:
            if self._generation != generation:
                return
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
        (or a newer start) bumped the generation while the target was running.

        This is the one place a run ends, whichever way it ended, so it is where the
        history entry is written. A run whose generation is stale writes nothing: it
        was superseded, and recording it would put a second outcome under a name the
        newer run owns.
        """
        with self._lock:
            if self._generation != generation:
                return
            self._apply(fields)
            record = job_history.record_for(
                self._state, self._last_request,
                round(time.monotonic() - self._start_ts, 1) if self._start_ts else None,
            )
        # Outside the lock: the disk write must not hold the state lock that every
        # /train/status read takes. And it must never turn a finished run into a
        # failed one — the work is already done and saved by the time we get here.
        try:
            job_history.append(record)
        except OSError:
            logger.warning("Could not record %r in the job history.", record.get("model_name"))

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def submit(
        self, target: Callable, *args: object, model_name: str, request: dict | None = None,
        kind: str = "training",
    ) -> int:
        """Accept a run: start it now, or queue it behind the one already going.

        Returns its position — ``0`` means it is running, ``N`` that ``N`` runs are
        ahead of it. Queueing rather than refusing is what lets several label fields be
        trained in one go without a browser tab staying open to shepherd them.

        :raises RuntimeError: when the queue is full, or when this name is already
            running or queued — a second run under one name could only fail, since
            ``/train`` refuses an existing model, so it is refused while it is still a
            request and the caller can still change it.
        """
        with self._lock:
            queued_names = [entry[2] for entry in self._queue]
            if model_name == self._running_name() or model_name in queued_names:
                raise RuntimeError(f"A run for model '{model_name}' is already running or queued.")
            if self._busy_locked():
                if len(self._queue) >= MAX_QUEUED:
                    raise RuntimeError(
                        f"The training queue is full ({MAX_QUEUED} runs waiting); "
                        "retry once some have finished."
                    )
                self._queue.append((target, args, model_name, request, kind))
                return len(self._queue)
        self.start(target, *args, model_name=model_name, request=request, kind=kind)
        return 0

    def _running_name(self) -> str | None:
        """Name of the run holding the thread right now (caller holds the lock)."""
        if self._state["status"] == "running":
            return self._state["model_name"]
        if self._thread is not None and self._thread.is_alive():
            return self._last_model_name
        return None

    def _busy_locked(self) -> bool:
        """Is a thread still executing? (caller holds the lock)

        Includes a thread abandoned by ``stop(hard=True)``, whose deploy fit and skops
        save keep running after the status was reset — starting another run then would
        put two full trainings on the CPU at once.
        """
        return self._state["status"] == "running" or (
            self._thread is not None and self._thread.is_alive()
        )

    def queued_names(self) -> list[str]:
        with self._lock:
            return [entry[2] for entry in self._queue]

    def _dispatch_next(self) -> None:
        """Start the next queued run — called by the finishing thread, as its last act.

        It has to be this thread: nothing else is awake when a run ends. The liveness
        guard in ``start`` would refuse (this thread is still alive), which is why the
        dispatch goes around it — legitimately, because the caller IS that thread and
        its own work returned before ``_finish``. What it hands over is a thread that is
        about to exit, not one still training.
        """
        while True:
            with self._lock:
                if not self._queue:
                    # No successor — and this thread is about to exit while still
                    # reporting alive, which `_busy_locked` reads as busy. A submit
                    # landing in that gap would queue behind a dispatcher that has
                    # already left and never run. Retire from the busy check here, under
                    # the SAME lock submit() takes, so such a submit sees an idle runner
                    # and starts the run itself. Guarded on identity so a newer thread's
                    # registration is never clobbered.
                    if self._thread is threading.current_thread():
                        self._thread = None
                    return
                target, args, model_name, request, kind = self._queue.popleft()
            try:
                self._launch(target, args, model_name, request, kind)
                return
            except Exception:  # noqa: BLE001 - one bad entry must not strand the queue
                logger.exception("Queued run %r could not be started; skipping it.", model_name)

    def active_model_name(self) -> str | None:
        """Name of the model a still-live runner thread is working on — also after
        a hard stop reset the STATUS to idle (the abandoned thread keeps writing
        its bundle). The import guard keys on this, not on the lying status."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._last_model_name
            return None

    def start(
        self, target: Callable, *args: object, model_name: str, request: dict | None = None,
        kind: str = "training",
    ) -> None:
        """Launch ``target(*args, on_progress=..., should_stop=...)`` in a thread NOW.

        Refuses while anything is still executing. ``submit`` is the entry point that
        queues instead; this one stays strict because it is what guarantees two full
        trainings never share the CPU.

        ``request`` is recorded with the run's outcome so the history says what was
        asked for, not only what came out. The runner is the only thing that knows when
        a run ends, so it is the only place that can write that record.
        """
        with self._lock:
            # Refuse while a previous thread is still executing — including one
            # abandoned by stop(hard=True), whose deploy fit / skops save keep
            # running after the status was reset to idle. Starting anyway would
            # run two full trainings at once, breaking the single-worker design.
            if self._busy_locked():
                raise RuntimeError(
                    "A training job is already running (or a hard-stopped one is "
                    "still finishing in the background); retry once it completes."
                )
        self._launch(target, args, model_name, request, kind)

    def _launch(
        self, target: Callable, args: tuple, model_name: str, request: dict | None,
        kind: str = "training",
    ) -> None:
        """Set the state up and put the run on a thread. No liveness guard: the two
        callers each establish it their own way — ``start`` by checking, and
        ``_dispatch_next`` by being the finishing thread itself."""

        def runner() -> None:
            try:
                results = target(*args, on_progress=on_progress, should_stop=self.should_stop)
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
                self._finish(generation, status="error", phase="error",
                             message=str(exc), error=str(exc))
            except Exception:  # noqa: BLE001 - report any failure as job error
                # Log the full traceback server-side; never leak exception text
                # (which can carry paths/data internals) to the readonly status.
                logger.exception("Training job %r failed", model_name)
                sanitized = "Training failed; see server logs for details."
                self._finish(generation, status="error", phase="error",
                             message=sanitized, error=sanitized)
            finally:
                # Last act of the thread, and in `finally` so a queue never strands
                # behind a run that failed in a way nobody anticipated.
                self._dispatch_next()

        with self._lock:
            self._stop.clear()
            self._state = _idle_state()
            self._state.update(
                status="running",
                phase="starting",
                model_name=model_name,
                # Without it an evaluation reads as a training that somehow produced
                # no model — in the history above all, where the two sit side by side.
                kind=kind,
                started_at=datetime.now(UTC).isoformat(),
            )
            self._start_ts = time.monotonic()
            # A run that hangs before its first progress update must still show
            # a growing heartbeat age, so the clock starts at launch.
            self._heartbeat_ts = self._start_ts
            self._generation += 1
            generation = self._generation
            self._last_model_name = model_name
            self._last_request = request
            # Register the thread INSIDE the same lock block as the state
            # transition: a hard stop + new start in the gap between two separate
            # blocks could otherwise pass the liveness guard and run two
            # trainings at once. Only thread.start() happens outside.
            thread = threading.Thread(target=runner, daemon=True)
            self._thread = thread

        def on_progress(**fields: object) -> None:
            self._update_if_current(generation, **fields)

        thread.start()

    def stop(self, *, hard: bool = False) -> None:
        """Cancel the running run and everything waiting behind it.

        The queue goes too: "stop" means "I want this to end", not "skip to the next
        one". The browser-side queue behaved this way already, so the server keeps the
        promise the UI had been making.
        """
        self._stop.set()
        with self._lock:
            self._queue.clear()
        if hard:
            with self._lock:
                # Nothing to reset when nothing runs — and resetting anyway would
                # discard the finished run's results, which is what the operator
                # was waiting for (a double click, or stopping a run that just
                # completed, used to blank the metrics).
                if self._state["status"] != "running":
                    return
                # Invalidate the running thread's generation so its eventual
                # completion cannot overwrite this reset.
                self._generation += 1
                self._apply(dict(status="idle", phase="", message="Hard stopped.",
                                 progress=0, results=None))


# Module-level singleton used by the API routes.
job_runner = JobRunner()
