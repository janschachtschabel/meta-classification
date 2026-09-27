"""How many long streaming responses may run at once (audit PERF-2).

`POST /predict/csv` returns a `StreamingResponse` wrapping a **sync** generator, so Starlette
iterates it through `iterate_in_threadpool` — one anyio worker held for the whole
classification, which is minutes for a 200 MB CSV. That is the same pool every `def` route and
every `asyncio.to_thread` call uses, nothing bounded how many could be in flight, and the rate
limit is per client address with a readonly key. Enough concurrent uploads and every other
endpoint waits behind them for a threadpool slot.

A plain counting slot rather than `anyio.CapacityLimiter`: anyio's token is held by the
borrowing *task*, and this one is acquired in the route and released when a generator finishes
in a worker thread — a different task, which anyio refuses. Stdlib, no dependency, and safe to
release from wherever the release happens.

Rejecting rather than queueing is the point. A queued request holds its connection, its
uploaded temp file and the client's patience while waiting for work that has not started; a
503 with `Retry-After` lets the caller decide.
"""

from __future__ import annotations

import threading


class Slots:
    """A counting limit that refuses instead of waiting.

    Not a `threading.Semaphore`: that one's `acquire(blocking=False)` is close, but releasing
    more times than acquired raises there, and here the release sits in a generator's
    ``finally`` that must be safe to reach twice (a client disconnect closes the generator,
    and the background task may follow).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._in_use = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_use >= self.capacity:
                return False
            self._in_use += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._in_use = max(0, self._in_use - 1)

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use


# Process-local, like the rate limiter, the model cache and the job runner — the chart pins one
# replica. Sized on first use from the configured maximum, so `get_settings()` is not read at
# import time.
_csv_slots: Slots | None = None
_csv_lock = threading.Lock()


def csv_slots() -> Slots:
    global _csv_slots
    if _csv_slots is None:
        with _csv_lock:
            if _csv_slots is None:
                from .settings import get_settings

                _csv_slots = Slots(get_settings().max_concurrent_csv)
    return _csv_slots


def reset_csv_slots() -> None:
    """Drop the sized limiter so the next call re-reads the settings. For tests."""
    global _csv_slots
    with _csv_lock:
        _csv_slots = None
