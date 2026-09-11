"""Tests for runtime settings resolution (CPU and memory budget for training)."""


import pytest
from pydantic import ValidationError

from app import settings as settings_mod
from app.settings import Settings

GiB = 1024**3


def test_effective_n_jobs_caps_at_cpu_percent(monkeypatch):
    """Training parallelism is bounded by cpu_max_percent of the available cores
    (default 60%), regardless of how many cores n_jobs requests — BLAS is pinned
    to 1 thread, so head-fit threads ARE the CPU footprint."""
    monkeypatch.setattr(settings_mod, "available_cpus", lambda: 10)
    assert Settings(n_jobs=-1, cpu_max_percent=60).effective_n_jobs() == 6   # all cores -> capped
    assert Settings(n_jobs=-2, cpu_max_percent=60).effective_n_jobs() == 6   # min(9, 6)
    assert Settings(n_jobs=4, cpu_max_percent=60).effective_n_jobs() == 4    # below the cap
    assert Settings(n_jobs=-1, cpu_max_percent=100).effective_n_jobs() == 10  # 100 = cap disabled
    assert Settings(n_jobs=-1).effective_n_jobs() == 6  # the DEFAULT budget is 60%

    monkeypatch.setattr(settings_mod, "available_cpus", lambda: 1)
    assert Settings(n_jobs=-1, cpu_max_percent=60).effective_n_jobs() == 1  # never below 1


def test_warmup_models_are_all_kept_resident():
    """Listing models in APIV3_WARMUP_MODELS is a statement that all of them should
    answer without a cold load. With the LRU sized independently (default 2), warming
    four models evicted two of them before the first request — the documented advice
    was to keep the list short rather than to hold what was asked for. The cache is
    therefore sized to fit at least the warmed models."""
    four = Settings(max_models_in_memory=2, warmup_models="subjects, level, type, curriculum")
    assert four.effective_max_models_in_memory() == 4

    # The configured cap still governs everything else: it is a RAM ceiling, and a
    # short warmup list must not shrink it.
    assert Settings(max_models_in_memory=5, warmup_models="subjects").effective_max_models_in_memory() == 5
    assert Settings(max_models_in_memory=2, warmup_models="").effective_max_models_in_memory() == 2
    assert Settings(max_models_in_memory=0, warmup_models="").effective_max_models_in_memory() == 1


def test_effective_n_jobs_uses_container_budget_not_host(monkeypatch):
    """The Kubernetes scenario: a 4-CPU-limited pod on a 64-core node. The budget
    must derive from the 4 CPUs the cgroup quota grants (-> 2 threads at the
    default 60%), not spawn ~38 threads into the quota (kernel throttling)."""
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(settings_mod, "_affinity_cpus", lambda: None)
    monkeypatch.setattr(settings_mod, "_cgroup_cpu_quota", lambda: 4.0)
    assert Settings(n_jobs=-1, cpu_max_percent=60).effective_n_jobs() == 2


def test_cgroup_v2_quota_detected(tmp_path):
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    assert settings_mod._cgroup_cpu_quota(tmp_path) == 4.0


def test_cgroup_v2_unlimited_returns_none(tmp_path):
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert settings_mod._cgroup_cpu_quota(tmp_path) is None


def test_cgroup_v1_quota_detected(tmp_path):
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("200000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert settings_mod._cgroup_cpu_quota(tmp_path) == 2.0


def test_cgroup_v1_unlimited_returns_none(tmp_path):
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert settings_mod._cgroup_cpu_quota(tmp_path) is None


def test_no_cgroup_files_returns_none(tmp_path):
    """Bare metal / Windows / macOS: no cgroup files -> no container limit."""
    assert settings_mod._cgroup_cpu_quota(tmp_path) is None


def test_cgroup_garbage_content_returns_none(tmp_path):
    """A malformed cpu.max must not crash settings resolution at import/startup."""
    (tmp_path / "cpu.max").write_text("not-a-number\n")
    assert settings_mod._cgroup_cpu_quota(tmp_path) is None


def test_available_cpus_quota_wins_over_host_count(monkeypatch):
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(settings_mod, "_affinity_cpus", lambda: None)
    monkeypatch.setattr(settings_mod, "_cgroup_cpu_quota", lambda: 4.0)
    assert settings_mod.available_cpus() == 4


def test_available_cpus_fractional_quota_floors_but_never_below_1(monkeypatch):
    """A 500m pod (quota 0.5) gets 1 CPU; 2.5 floors to 2 (a thread more than the
    quota just gets throttled, so round down)."""
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(settings_mod, "_affinity_cpus", lambda: None)
    monkeypatch.setattr(settings_mod, "_cgroup_cpu_quota", lambda: 0.5)
    assert settings_mod.available_cpus() == 1
    monkeypatch.setattr(settings_mod, "_cgroup_cpu_quota", lambda: 2.5)
    assert settings_mod.available_cpus() == 2


def test_available_cpus_affinity_mask_wins_when_smaller(monkeypatch):
    """taskset / the K8s static CPU manager pin processes via the affinity mask;
    it bounds the count exactly like a quota does."""
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(settings_mod, "_affinity_cpus", lambda: 8)
    monkeypatch.setattr(settings_mod, "_cgroup_cpu_quota", lambda: None)
    assert settings_mod.available_cpus() == 8


def test_available_cpus_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(settings_mod, "_affinity_cpus", lambda: None)
    monkeypatch.setattr(settings_mod, "_cgroup_cpu_quota", lambda: None)
    assert settings_mod.available_cpus() == 8


def test_train_memory_budget_is_set_in_megabytes(monkeypatch):
    monkeypatch.setenv("APIV3_TRAIN_MEMORY_MB", "1024")
    assert Settings().effective_train_memory_bytes() == 1024 * 1024**2


def test_train_memory_budget_defaults_to_the_container_limit_minus_headroom(monkeypatch):
    """The 8 GB container that OOM-killed the wlo_max run: without a setting, the budget
    is what the cgroup allows minus 15 % for the API, the interpreter and the allocator's
    slack — 6.8 GiB of 8."""
    monkeypatch.setattr(settings_mod, "memory_limit_bytes", lambda: 8 * GiB)
    assert Settings().effective_train_memory_bytes() == int(8 * GiB * 0.85)


def test_train_memory_budget_zero_disables_the_cap(monkeypatch):
    monkeypatch.setattr(settings_mod, "memory_limit_bytes", lambda: 8 * GiB)
    assert Settings(train_memory_mb=0).effective_train_memory_bytes() is None


def test_train_memory_budget_without_a_container_limit_changes_nothing(monkeypatch):
    """Bare metal, Windows, Docker without --memory: no limit known, no cap — the run
    keeps every thread the CPU budget gives it, exactly as before."""
    monkeypatch.setattr(settings_mod, "memory_limit_bytes", lambda: None)
    assert Settings().effective_train_memory_bytes() is None


def test_a_negative_train_memory_budget_is_refused():
    with pytest.raises(ValidationError):
        Settings(train_memory_mb=-1)


def test_the_suite_never_writes_the_repositorys_own_state_files():
    """conftest redirects the job history and the feedback store to a scratch
    dir, but not the share store: after a run, the developer's real
    share_links.json held three live links to the test model "odd_metrics".
    Every file the app appends to at runtime must point outside the repo."""
    s = Settings()
    for path in (s.share_links_file, s.job_history_file, s.feedback_file):
        assert not path.resolve().is_relative_to(settings_mod._BASE), path
