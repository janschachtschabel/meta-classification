"""Make the ``app`` package importable from tests regardless of CWD."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """slowapi's in-memory store is process-global; clear it between tests so
    accumulated hits never cause order/speed-dependent spurious 429s."""
    from app.limiter import limiter

    limiter.reset()
    yield
