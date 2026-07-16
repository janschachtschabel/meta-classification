"""Tests for runtime settings resolution (CPU budget for training)."""


from app.settings import Settings


def test_effective_n_jobs_caps_at_cpu_percent(monkeypatch):
    """Training parallelism is bounded by cpu_max_percent of the cores (default
    60%), regardless of how many cores n_jobs requests — BLAS is pinned to 1
    thread, so head-fit threads ARE the CPU footprint."""
    monkeypatch.setattr("os.cpu_count", lambda: 10)
    assert Settings(n_jobs=-1, cpu_max_percent=60).effective_n_jobs() == 6   # all cores -> capped
    assert Settings(n_jobs=-2, cpu_max_percent=60).effective_n_jobs() == 6   # min(9, 6)
    assert Settings(n_jobs=4, cpu_max_percent=60).effective_n_jobs() == 4    # below the cap
    assert Settings(n_jobs=-1, cpu_max_percent=100).effective_n_jobs() == 10  # 100 = cap disabled
    assert Settings(n_jobs=-1).effective_n_jobs() == 6  # the DEFAULT budget is 60%

    monkeypatch.setattr("os.cpu_count", lambda: 1)
    assert Settings(n_jobs=-1, cpu_max_percent=60).effective_n_jobs() == 1  # never below 1
