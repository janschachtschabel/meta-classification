"""Shared rate-limiter instance and limit resolvers.

Kept in its own module so both ``main`` and route modules can import it
without creating a circular dependency.

Note: slowapi's default backend is in-memory, i.e. per-process. The app is
designed for single-worker deployment; document a shared store before scaling
to multiple workers.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from .settings import get_settings

limiter = Limiter(key_func=get_remote_address, default_limits=[])

_UNLIMITED = "1000000/minute"


def _resolve(name: str) -> str:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return _UNLIMITED
    return {
        "predict": settings.rate_limit_predict,
        "train": settings.rate_limit_train,
        "export": settings.rate_limit_export,
    }.get(name, settings.rate_limit_default)


def default_limit() -> str:
    """Fallback limit for state-changing endpoints without a category of their own
    (e.g. deletes) — wires ``rate_limit_default``, which was otherwise unused."""
    return _resolve("default")


def predict_limit() -> str:
    return _resolve("predict")


def train_limit() -> str:
    return _resolve("train")


def export_limit() -> str:
    return _resolve("export")
