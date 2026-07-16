"""Application settings: storage paths, auth, CORS, limits, compute resources.

Values come from environment variables (prefix ``APIV3_``) and an optional
``.env`` file. Training *profiles* live separately in ``config.yaml``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor default paths to the api_v3 folder so the app works from any CWD.
_BASE = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, loaded from env / .env (prefix ``APIV3_``)."""

    model_config = SettingsConfigDict(
        env_prefix="APIV3_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage ---
    data_dir: Path = _BASE / "data"
    models_dir: Path = _BASE / "models"
    share_links_file: Path = _BASE / "share_links.json"
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
    # Per-label head parallelism. -1 = all CPU cores; negative values follow
    # joblib semantics (-2 = all but one). With the default 'threading' backend +
    # a float32-preserving solver, the cores share ONE sparse matrix, so adding
    # cores costs little extra RAM. The value is additionally bounded by
    # cpu_max_percent (below) — the effective thread count is the minimum.
    n_jobs: int = -1
    # Hard CPU budget for a training run, as a percentage of the machine's cores
    # (BLAS is pinned to 1 thread, so head-fit threads ARE the CPU footprint).
    # Default 60: training never occupies more than ~60% of the CPU, keeping the
    # API responsive and leaving headroom for other work. 100 disables the cap.
    cpu_max_percent: int = Field(60, ge=1, le=100)
    # TF-IDF vocabulary caps = the main RAM/quality lever. Lower = less RAM.
    tfidf_max_word_features: int = 80_000
    tfidf_max_char_features: int = 120_000
    # Linear-head solver + joblib backend. 'newton-cg' (default) and 'saga' KEEP
    # the float32 matrix (RAM-safe at scale) and release the GIL, so 'threading'
    # trains all labels on ONE shared matrix (all cores, ~1x RAM). 'newton-cg'
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

    def effective_n_jobs(self) -> int:
        """Thread count for the label-wise head fits: the requested ``n_jobs``
        (joblib semantics for negatives) bounded by the ``cpu_max_percent``
        budget. Never below 1."""
        cores = os.cpu_count() or 1
        requested = self.n_jobs if self.n_jobs > 0 else max(1, cores + 1 + self.n_jobs)
        budget = max(1, (cores * self.cpu_max_percent) // 100)
        return max(1, min(requested, budget))

    def ensure_dirs(self) -> None:
        """Create storage directories if they do not exist."""
        for directory in (self.data_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()
