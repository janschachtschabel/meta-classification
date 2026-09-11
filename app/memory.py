"""What this process holds in memory, what it may hold, and how many head fits that allows.

A training run's peak is set by the label-wise head fits: every concurrent newton-cg
fit holds solver buffers the size of the feature matrix, so the thread count that is
right for the CPU can be wrong for the RAM (a run on the full WLO export was OOM-killed
that way; see docs/plans/2026-09-11-training-memory.md). ``ThreadBudget`` lowers the
thread count to what the memory budget allows, ``PeakSampler`` records what a run
actually needed.

Stdlib only (procfs, cgroupfs, ctypes): psutil would be a 14th runtime dependency for
three numbers. Every reading degrades to "unknown" (0 / None) instead of raising — this
code reports on a training run and must never be the thing that fails it.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("api_v3.training")

MiB = 1024 * 1024

# Matrix-sized buffers ONE concurrent per-label newton-cg fit holds. scikit-learn
# materialises hX = diag(h) @ X in every Newton iteration and keeps the previous
# iteration's alive while it builds the next (sklearn/linear_model/_linear_loss.py,
# sklearn/utils/optimize.py). Measured 1.8-2.5x per thread; the conservative end,
# because a budget that is too generous is an OOM kill and one too strict only costs time.
FIT_COPIES_PER_THREAD = 2.5

# cgroup v1 has no "max" keyword: an unlimited cgroup reports a huge page-aligned number.
_V1_UNLIMITED = 2**60


if sys.platform == "win32":

    class _MemoryCounters(ctypes.Structure):
        """PROCESS_MEMORY_COUNTERS (psapi.h)."""

        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    @functools.cache
    def _psapi() -> tuple[Any, Any]:
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        # HANDLE is pointer-sized. Without the restype the pseudo-handle (-1) comes back
        # truncated to a 32-bit int and every call fails, reading 0 everywhere.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_MemoryCounters), ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        return kernel32, psapi

    def _windows_working_set() -> int:
        kernel32, psapi = _psapi()
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return 0
        return int(counters.WorkingSetSize)


def rss_bytes() -> int:
    """Resident set size of this process in bytes; 0 when the platform gives no reading.

    macOS reads 0: its stdlib only knows the lifetime PEAK (``ru_maxrss``), and a peak
    passed off as a current reading would make every budget decision wrong.
    """
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", "rb") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        if sys.platform == "win32":
            return _windows_working_set()
    except (OSError, ValueError, IndexError):
        pass
    return 0


def memory_limit_bytes(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Memory limit the container's cgroup imposes, in bytes; None = unlimited/absent.

    cgroup v2 (``memory.max``: a number or "max") first, then v1
    (``memory/memory.limit_in_bytes``). Malformed or missing files mean "no limit",
    exactly like ``settings._cgroup_cpu_quota``: this runs during settings resolution.
    """
    try:
        value = (cgroup_root / "memory.max").read_text().strip()
        if value != "max" and int(value) > 0:
            return int(value)
    except (OSError, ValueError):
        pass
    try:
        limit = int((cgroup_root / "memory" / "memory.limit_in_bytes").read_text())
        if 0 < limit < _V1_UNLIMITED:
            return limit
    except (OSError, ValueError):
        pass
    return None


def matrix_bytes(matrix: Any) -> int:
    """Bytes a feature matrix holds: a sparse matrix's three arrays, or a dense buffer."""
    if hasattr(matrix, "indptr"):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(matrix.nbytes)


class PeakSampler:
    """The highest RSS seen while the context is open, sampled on a daemon thread.

    Progress callbacks fire between steps, but the largest allocations of a training run
    live and die INSIDE one step (a head fit holds its solver copies for minutes), so a
    reading taken only at the callbacks would miss exactly the number that decides
    whether a run fits into its memory limit.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self._interval = interval
        self._peak = rss_bytes()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PeakSampler:
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._record()

    def _record(self) -> int:
        # Both the sampler thread and readers record, so a reading a reader saw can
        # never be undercut by a later answer: a reported peak only grows.
        reading = rss_bytes()
        with self._lock:
            self._peak = max(self._peak, reading)
            return self._peak

    @property
    def peak_bytes(self) -> int:
        """Highest reading so far, including one taken now."""
        return self._record()


@dataclass
class ThreadBudget:
    """How many label-wise head fits may run at once in one training run.

    ``requested`` is the CPU's answer (``Settings.effective_n_jobs``); the memory budget
    can only lower it, never raise it. Each fit's count is decided as it starts, from the
    matrix it is about to fit on and what the process holds at that moment — the copies
    scale with exactly that matrix — and kept in ``chosen`` for the bundle's metadata.
    ``budget_bytes=None`` grants the request unchanged (no limit known or configured).
    """

    requested: int
    budget_bytes: int | None
    chosen: list[int] = field(default_factory=list)

    def for_matrix(self, matrix: Any) -> int:
        threads = self.requested
        if self.budget_bytes is not None:
            per_fit = FIT_COPIES_PER_THREAD * matrix_bytes(matrix)
            if per_fit > 0:
                held = rss_bytes()
                fits = int((self.budget_bytes - held) // per_fit)
                threads = max(1, min(self.requested, fits))
                if threads < self.requested and (not self.chosen or self.chosen[-1] != threads):
                    logger.info(
                        "Memory budget: head fits use %d of %d threads (budget %d MB, "
                        "process %d MB, matrix %d MB)", threads, self.requested,
                        self.budget_bytes // MiB, held // MiB, matrix_bytes(matrix) // MiB,
                    )
        self.chosen.append(threads)
        return threads

    def summary(self) -> dict[str, int] | None:
        """``{"requested", "min", "max"}`` over the fits so far; None before the first."""
        if not self.chosen:
            return None
        return {"requested": self.requested, "min": min(self.chosen), "max": max(self.chosen)}
