"""Process memory readings and the head-fit thread budget they drive (app/memory.py).

The readings themselves are platform facts, so these tests pin what can be pinned: that
RSS moves with a real allocation, that the sampler keeps a peak that is already gone,
that the cgroup files are parsed the way the kernel writes them, and the budget
arithmetic. How much memory a training run saves is benchmark evidence
(scripts/benchmark_training_memory.py), not a unit test.
"""

import time

import numpy as np
from scipy import sparse

from app import memory
from app.memory import PeakSampler, ThreadBudget, matrix_bytes, memory_limit_bytes, rss_bytes

MiB = 1024 * 1024


def _row_matrix(nonzeros: int) -> sparse.csr_matrix:
    """A 1 x n float32 CSR holding ``8 * nonzeros + 8`` bytes (data + indices + indptr)."""
    return sparse.csr_matrix(
        (np.ones(nonzeros, dtype=np.float32), np.arange(nonzeros, dtype=np.int32),
         np.array([0, nonzeros], dtype=np.int32)),
        shape=(1, nonzeros),
    )


def test_rss_grows_while_a_large_array_is_alive():
    before = rss_bytes()
    block = np.ones(64 * MiB // 8)  # ones() touches every page; empty() would not count
    during = rss_bytes()
    del block
    assert before > 0
    assert during - before >= 40 * MiB


def test_rss_of_another_process_is_readable_by_its_id():
    """A training in a child process: the status reads the child's memory by its id."""
    import os
    import subprocess
    import sys

    own = rss_bytes(os.getpid())
    assert abs(own - rss_bytes()) < 32 * MiB  # the same process, read both ways

    # b"x" * n writes every page; a zeroed bytearray may never become resident. The child
    # reports its own id: under a Windows venv, Popen's pid is the launcher's, not the
    # interpreter's — the reason the training worker reports its id the same way.
    child = subprocess.Popen(
        [sys.executable, "-c", "import os, sys; block = b'x' * (96 << 20); "
                               "print(os.getpid(), flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        pid = int(child.stdout.readline())
        assert rss_bytes(pid) >= 64 * MiB
    finally:
        child.communicate(timeout=10)
    # Ended: nothing raises, and none of its 96 MB is reported. (Linux reads 0; Windows
    # keeps a terminated process queryable while a handle to it is open — a few KB.)
    assert rss_bytes(pid) < MiB


def test_peak_sampler_keeps_a_peak_that_is_already_gone():
    """The head fit's solver copies live and die inside one call, where no progress
    callback ever looks — only a sampler running beside it sees them."""
    with PeakSampler(interval=0.01) as sampler:
        start = rss_bytes()
        block = np.ones(64 * MiB // 8)
        time.sleep(0.2)
        del block
    assert sampler.peak_bytes - start >= 40 * MiB


def test_cgroup_v2_memory_limit_detected(tmp_path):
    (tmp_path / "memory.max").write_text("8589934592\n")
    assert memory_limit_bytes(tmp_path) == 8 * 1024**3


def test_cgroup_v2_unlimited_returns_none(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    assert memory_limit_bytes(tmp_path) is None


def test_cgroup_v1_memory_limit_detected(tmp_path):
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("8589934592\n")
    assert memory_limit_bytes(tmp_path) == 8 * 1024**3


def test_cgroup_v1_unlimited_sentinel_returns_none(tmp_path):
    """v1 has no "max" keyword: an unlimited cgroup reports a huge page-aligned number."""
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert memory_limit_bytes(tmp_path) is None


def test_missing_or_malformed_cgroup_files_mean_no_limit(tmp_path):
    """Bare metal, Windows, macOS — and a garbled file must not take settings down."""
    assert memory_limit_bytes(tmp_path) is None
    (tmp_path / "memory.max").write_text("lots\n")
    assert memory_limit_bytes(tmp_path) is None


def test_matrix_bytes_counts_the_three_sparse_arrays_or_the_dense_buffer():
    matrix = _row_matrix(1000)
    assert matrix_bytes(matrix) == 8 * 1000 + 8
    assert matrix_bytes(np.zeros((10, 10))) == 800


def test_thread_budget_without_a_budget_grants_the_request():
    budget = ThreadBudget(requested=9, budget_bytes=None)
    assert budget.for_matrix(_row_matrix(1000)) == 9
    assert budget.chosen == [9]


def test_thread_budget_fits_the_threads_into_the_headroom(monkeypatch):
    monkeypatch.setattr(memory, "rss_bytes", lambda: 20 * MiB)
    budget = ThreadBudget(requested=9, budget_bytes=31 * MiB)
    # 11 MiB of headroom / (2.5 x ~1 MiB per concurrent fit) = 4 threads.
    assert budget.for_matrix(_row_matrix(MiB // 8)) == 4


def test_thread_budget_never_goes_below_one_thread(monkeypatch):
    """Over budget already: one fit at a time is the least a run can do and still finish."""
    monkeypatch.setattr(memory, "rss_bytes", lambda: 40 * MiB)
    budget = ThreadBudget(requested=9, budget_bytes=31 * MiB)
    assert budget.for_matrix(_row_matrix(MiB // 8)) == 1


def test_thread_budget_never_exceeds_the_cpu_request(monkeypatch):
    monkeypatch.setattr(memory, "rss_bytes", lambda: 0)
    budget = ThreadBudget(requested=3, budget_bytes=10_000 * MiB)
    assert budget.for_matrix(_row_matrix(1000)) == 3


def test_a_budget_shared_with_another_process_counts_its_memory(monkeypatch):
    """A training in a child process shares the container's limit with the API process
    that started it: what the API holds is spent from the same budget — as it was when
    the training ran inside the API process."""
    monkeypatch.setattr(memory, "rss_bytes",
                        lambda pid=None: {None: 20 * MiB, 4242: 3 * MiB}[pid])
    monkeypatch.setattr(memory, "_budget_shared_with", [])
    alone = ThreadBudget(requested=9, budget_bytes=31 * MiB)
    assert alone.for_matrix(_row_matrix(MiB // 8)) == 4  # 11 MiB headroom / 2.5 MiB

    memory.share_budget_with(4242)
    shared = ThreadBudget(requested=9, budget_bytes=31 * MiB)
    assert shared.for_matrix(_row_matrix(MiB // 8)) == 3  # 8 MiB headroom / 2.5 MiB


def test_thread_budget_summarizes_what_the_fits_got(monkeypatch):
    readings = iter([20 * MiB, 26 * MiB])
    monkeypatch.setattr(memory, "rss_bytes", lambda: next(readings))
    budget = ThreadBudget(requested=9, budget_bytes=31 * MiB)
    budget.for_matrix(_row_matrix(MiB // 8))  # 11 MiB headroom -> 4
    budget.for_matrix(_row_matrix(MiB // 8))  # 5 MiB headroom -> 1 (2.5 MiB per fit)
    assert budget.summary() == {"requested": 9, "min": 1, "max": 4}
    assert ThreadBudget(requested=9, budget_bytes=None).summary() is None
