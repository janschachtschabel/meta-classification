"""Make the ``app`` package importable from tests regardless of CWD, and keep
the suite hermetic against ambient configuration."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
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
# Trainings through /train run in-process for the suite: an interpreter start per run
# adds seconds, and tests patch the pipeline in THIS process. tests/test_train_worker.py
# covers the child-process path the shipped default takes.
os.environ["APIV3_TRAINING_ISOLATION"] = "thread"
# Any JobRunner a test constructs writes its outcome to the configured history file.
# Without this the suite appends to the developer's REAL one — found by reading that
# file after a live run and seeing "first", "second", "third" in it.
_SCRATCH = Path(tempfile.mkdtemp(prefix="apiv3-tests-"))
os.environ["APIV3_JOB_HISTORY_FILE"] = str(_SCRATCH / "job_history.jsonl")
# Same reason: a test that posts feedback would otherwise append to the developer's
# real collection — which, unlike the history, is training data nobody wants seeded
# with "Der Wiener Kongress" from a test fixture.
os.environ["APIV3_FEEDBACK_FILE"] = str(_SCRATCH / "feedback.jsonl")
# And the share store: found three live links to the test model "odd_metrics" in
# the developer's real share_links.json. Bearer links are the last file to leak into.
os.environ["APIV3_SHARE_LINKS_FILE"] = str(_SCRATCH / "share_links.json")


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """slowapi's in-memory store is process-global; clear it between tests so
    accumulated hits never cause order/speed-dependent spurious 429s."""
    from app.limiter import limiter

    limiter.reset()
    yield


class _ScriptedHead:
    """A head whose probabilities are read from a table instead of learned."""

    def __init__(self, table, scored_rows: list[int]) -> None:
        self.table = table
        self._scored_rows = scored_rows

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        # Sparse when a scripted vectorizer produced it, dense when a test handed
        # `cross_val_evaluate` a shared matrix directly.
        dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        rows = dense[:, 0].astype(int)
        self._scored_rows.append(len(rows))
        return self.table[rows]


class ScriptedCSearch:
    """Two C candidates that the flat 0.5 cut and tuned thresholds disagree about.

    ``C=1`` is well scaled but misses row 0 of label 0: macro F1 6/7 and 1.0 on the two
    labels, and no threshold rescues it because the missed row scores exactly what the
    negatives score. ``C=2`` ranks every row perfectly but compresses the scores below
    0.5 — the flat cut predicts nothing and scores it 0.0, a tuned cut scores it 1.0.
    Selecting on the flat cut therefore discards the candidate that wins once its
    thresholds are set, which is what plan item C1 is about.

    The probabilities are scripted rather than fitted so the tests measure the
    SELECTION RULE: with real fits the answer would turn on how well two Cs happen to
    separate toy data. Row identity travels in column 0 of the feature matrix, which is
    what lets a CV fold's slice look its own rows up again.
    """

    # Macro F1 the well-scaled candidate reaches under EITHER rule — one missed
    # positive on label 0 (F1 6/7) and a perfect label 1, averaged. Named once so the
    # tests read as "the well-scaled candidate's score" rather than a magic constant.
    well_scaled_f1 = (6 / 7 + 1.0) / 2

    def __init__(self, scored_rows: list[int]) -> None:
        self.y = np.zeros((8, 2), dtype=int)
        self.y[:4, 0] = 1
        self.y[4:, 1] = 1
        self.tables = {
            1.0: np.array([[0.1, 0.1]] + [[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 4),
            2.0: np.array([[0.4, 0.05]] * 4 + [[0.05, 0.4]] * 4),
        }
        self.row_ids = np.arange(8, dtype=float).reshape(-1, 1)
        # Rows handed to predict_proba, one entry per call, in order.
        self.scored_rows = scored_rows

    def head(self, c, **kwargs):
        return _ScriptedHead(self.tables[c], self.scored_rows)


@pytest.fixture
def scripted_c_search(monkeypatch):
    """`tuning.make_head` scripted with :class:`ScriptedCSearch`'s two candidates."""
    from app import tuning

    case = ScriptedCSearch([])
    monkeypatch.setattr(tuning, "make_head", case.head)
    return case
