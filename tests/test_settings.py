"""Tests for runtime settings resolution (CPU budget for training)."""


from app import settings as settings_mod
from app.settings import Settings


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
