"""Application settings: storage paths, auth, CORS, limits, compute resources.

Values come from environment variables (prefix ``APIV3_``) and an optional
``.env`` file. Training *profiles* live separately in ``config.yaml``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .memory import MiB, memory_limit_bytes

# Anchor default paths to the api_v3 folder so the app works from any CWD.
_BASE = Path(__file__).resolve().parent.parent

# Share of a container's memory limit a training run may plan with when no explicit
# budget is set. The rest is for the API serving alongside, the interpreter and the
# allocator's slack; the head-fit thread factor is conservative on top of it.
_TRAIN_MEMORY_SHARE = 0.85


def _affinity_cpus() -> int | None:
    """CPUs in this process' scheduler affinity mask (Linux only; None elsewhere).

    taskset and the Kubernetes static CPU manager pin processes via this mask,
    so it bounds usable parallelism exactly like a quota does.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:
        return None
    try:
        return len(getaffinity(0)) or None
    except OSError:
        return None


def _cgroup_cpu_quota(cgroup_root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPU quota the container's cgroup imposes, in cores; None = unlimited/absent.

    Checks cgroup v2 (``cpu.max``: "<quota> <period>" or "max ...") first, then
    v1 (``cpu/cpu.cfs_quota_us`` / ``cpu.cfs_period_us``, -1 = unlimited).
    Malformed or missing files mean "no limit" — this runs during settings
    resolution and must never take the app down.
    """
    try:
        parts = (cgroup_root / "cpu.max").read_text().split()
        if parts and parts[0] != "max":
            period = int(parts[1]) if len(parts) > 1 else 100_000
            if period > 0:
                return int(parts[0]) / period
    except (OSError, ValueError):
        pass
    try:
        quota = int((cgroup_root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((cgroup_root / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """CPUs THIS process may actually use, never below 1.

    ``os.cpu_count()`` reports the NODE's cores inside a container (a CFS quota
    is invisible to it) — on a 64-core node a 4-CPU-limited pod would otherwise
    size ~38 fit threads into its quota and just get throttled by the kernel.
    Bound the count by the affinity mask and the cgroup quota (fractional
    quotas floor: an extra thread beyond the quota only adds throttling).
    """
    cores = os.cpu_count() or 1
    affinity = _affinity_cpus()
    if affinity is not None:
        cores = min(cores, affinity)
    quota = _cgroup_cpu_quota()
    if quota is not None:
        cores = min(cores, int(quota))
    return max(1, cores)


class Settings(BaseSettings):
    """Runtime configuration, loaded from env / .env (prefix ``APIV3_``)."""

    model_config = SettingsConfigDict(
        env_prefix="APIV3_",
        # Anchored like every other default path: a relative name is resolved against
        # the WORKING directory, so starting uvicorn from anywhere but api_v3/ silently
        # dropped the dotenv — and with it the API keys.
        env_file=_BASE / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage ---
    data_dir: Path = _BASE / "data"
    models_dir: Path = _BASE / "models"
    share_links_file: Path = _BASE / "share_links.json"
    # Outcomes of finished training runs. Beside the models rather than in the
    # models dir: it describes runs, not bundles, and a stray file there would
    # sit next to things the registry enumerates.
    job_history_file: Path = _BASE / "job_history.jsonl"
    # Corrections editors make to predictions. Append-only and uncapped: this is
    # training data, not a log, so the oldest entry is worth as much as the newest.
    feedback_file: Path = _BASE / "feedback.jsonl"
    config_file: Path = _BASE / "config.yaml"

    # --- RAM control: how many models stay resident (LRU eviction beyond this) ---
    max_models_in_memory: int = 2

    # --- Warmup: model names to preload into the LRU cache on startup and run one
    # empty-text prediction on, so the first real /predict for them pays no cold
    # skops-load. Comma-separated; empty = off. Names beyond max_models_in_memory
    # are loaded in order but only the last that many stay resident. ---
    warmup_models: str = ""

    # --- Admin UI (static single-page app served at /ui; the page itself is
    # public like /docs — every data request from it carries the X-API-Key) ---
    ui_enabled: bool = True

    # --- Auth ---
    auth_enabled: bool = True
    api_key_admin: str | None = None
    api_key_readonly: str | None = None

    # --- CORS (comma-separated allowlist; empty = no cross-origin) ---
    cors_allow_origins: str = ""

    # --- Limits ---
    max_upload_mb: int = 200
    random_seed: int = 42

    # --- Compute resources ---
    # Per-label head parallelism. "auto" (default) = every core this process may use;
    # an integer asks for that many, negatives following joblib (-1 = all, -2 = all
    # but one). Bounded by cpu_max_percent (below), and per fit by the memory budget
    # (train_memory_mb): the cores share ONE input matrix, but every concurrent fit
    # adds its own solver buffers (~2.5x the matrix).
    n_jobs: int | Literal["auto"] = "auto"
    # Hard CPU budget for a training run, as a percentage of the machine's cores
    # (BLAS is pinned to 1 thread, so head-fit threads ARE the CPU footprint).
    # Default 60: training never occupies more than ~60% of the CPU, keeping the
    # API responsive and leaving headroom for other work. 100 disables the cap.
    cpu_max_percent: int = Field(60, ge=1, le=100)
    # Memory budget for a training run, in MiB. Every concurrent head fit holds ~2.5x
    # the feature matrix in solver buffers, so the budget bounds the head-fit threads:
    # a run that would outgrow it trains with fewer threads (slower) instead of being
    # OOM-killed. "auto" (default) = 85 % of the container's cgroup memory limit when
    # there is one, otherwise no cap; 0 = no cap. Never changes the model, only the speed.
    train_memory_mb: Annotated[int, Field(ge=0)] | Literal["auto"] = "auto"
    # TF-IDF vocabulary caps = the main RAM/quality lever. Lower = less RAM.
    tfidf_max_word_features: int = 80_000
    tfidf_max_char_features: int = 120_000
    # Linear-head solver + joblib backend. 'newton-cg' (default) and 'saga' KEEP
    # the float32 matrix (RAM-safe at scale) and release the GIL, so 'threading'
    # trains all labels on ONE shared input matrix (each fit's solver buffers come on
    # top; train_memory_mb sizes the threads accordingly). 'newton-cg'
    # converges fast; 'saga' is slow on high-dim TF-IDF. 'lbfgs'/'liblinear'
    # upcast to float64 (2x matrix RAM) — ok on small data; pair the GIL-bound
    # 'liblinear' with the 'loky' backend.
    solver: str = "newton-cg"
    parallel_backend: str = "threading"

    # --- Logging ---
    log_level: str = "INFO"

    # --- Rate limiting ---
    rate_limit_enabled: bool = True
    rate_limit_predict: str = "300/minute"
    rate_limit_train: str = "5/minute"
    rate_limit_export: str = "30/minute"
    rate_limit_default: str = "120/minute"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS allowlist into a clean list."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def warmup_models_list(self) -> list[str]:
        """Model names to preload on startup (parsed from the comma-separated list)."""
        return [m.strip() for m in self.warmup_models.split(",") if m.strip()]

    def effective_max_models_in_memory(self) -> int:
        """Size of the model cache: the configured cap, but never smaller than the
        warmup list.

        Listing a model in ``warmup_models`` states that it should answer without a
        cold skops load. Sizing the LRU independently broke that promise silently —
        four warmed models on the default cap of 2 left two of them evicted before
        the first request. The cap keeps its meaning as the RAM ceiling for
        everything else; it is only lifted to hold what was explicitly asked for.
        """
        return max(1, self.max_models_in_memory, len(self.warmup_models_list))

    def effective_n_jobs(self) -> int:
        """Thread count for the label-wise head fits: the requested ``n_jobs``
        (joblib semantics for negatives) bounded by the ``cpu_max_percent``
        budget. Never below 1. Container-aware: cores = ``available_cpus()``
        (cgroup quota / affinity mask), not the host's count."""
        cores = available_cpus()
        n_jobs = self.n_jobs if isinstance(self.n_jobs, int) else -1  # "auto" = all cores
        requested = n_jobs if n_jobs > 0 else max(1, cores + 1 + n_jobs)
        budget = max(1, (cores * self.cpu_max_percent) // 100)
        return max(1, min(requested, budget))

    def effective_train_memory_bytes(self) -> int | None:
        """Memory a training run may plan its head-fit threads with; None = no cap.

        An explicit ``train_memory_mb`` wins (``0`` switches the cap off). ``"auto"`` is
        the container's cgroup limit minus headroom — the limit the kernel enforces by
        killing the process, which is exactly the failure this budget exists to avoid.
        """
        if isinstance(self.train_memory_mb, int):
            return self.train_memory_mb * MiB if self.train_memory_mb > 0 else None
        limit = memory_limit_bytes()
        return int(limit * _TRAIN_MEMORY_SHARE) if limit is not None else None

    def ensure_dirs(self) -> None:
        """Create storage directories if they do not exist."""
        for directory in (self.data_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()
