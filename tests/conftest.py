"""Make the ``app`` package importable from tests regardless of CWD, and keep
the suite hermetic against ambient configuration."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Hermeticity: a developer's shell APIV3_* variables or a local `.env` (the
# docker-compose quickstart tells users to create one!) must not leak into the
# suite — e.g. APIV3_CORS_ALLOW_ORIGINS="*" or RATE_LIMIT_ENABLED=false flip
# real assertions. Scrub stray vars BEFORE any test module imports the app, and
# pin the assertion-critical settings to their defaults as REAL env vars (real
# env beats `.env` in pydantic-settings, so a local dotenv cannot flip them
# either). Tests still override freely via monkeypatch.setenv.
for _var in [k for k in os.environ if k.startswith("APIV3_")]:
    del os.environ[_var]
os.environ["APIV3_CORS_ALLOW_ORIGINS"] = ""
os.environ["APIV3_RATE_LIMIT_ENABLED"] = "true"
# Any JobRunner a test constructs writes its outcome to the configured history file.
# Without this the suite appends to the developer's REAL one — found by reading that
# file after a live run and seeing "first", "second", "third" in it.
os.environ["APIV3_JOB_HISTORY_FILE"] = str(
    Path(tempfile.mkdtemp(prefix="apiv3-tests-")) / "job_history.jsonl"
)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """slowapi's in-memory store is process-global; clear it between tests so
    accumulated hits never cause order/speed-dependent spurious 429s."""
    from app.limiter import limiter

    limiter.reset()
    yield
