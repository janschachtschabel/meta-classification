"""The optional metadata warmstart, and why it is optional.

`wordfreq` loads its German frequency table on the first `zipf_frequency` call, not at import:
measured over four fresh interpreters, that costs **245-286 ms and ~58 MB of RSS**. End to end
the first `/metadata` request takes 328-381 ms against 10-11 ms warm, and the warmstart moves
that to startup.

The ~58 MB is why this is off by default rather than unconditional. It stays resident for the
life of the process, and this app budgets memory deliberately — `ThreadBudget` sizes how many
head fits may run at once against the cgroup limit, so a table nothing uses would quietly buy
a training fewer parallel fits. A deployment that serves `/metadata` pays the memory on its
first request either way and should turn the setting on; one that only trains or only
classifies should not pay it at all.
"""

import ast
import logging
from pathlib import Path

# The cache behind textprep.stem (split out so a fragment longer than any word stays out of it).
from app.metadata.textprep import cached_stem as stem
from app.settings import Settings

# `app.lifecycle` is imported inside each test rather than here on purpose. Several test
# modules configure themselves by setting APIV3_* in the environment at import time and then
# importing the app, whose `get_settings`/`get_registry` lru_caches bind to whatever the FIRST
# importer had set. pytest imports every module during collection before running anything, so
# importing it here would bind those caches to this module's (empty) configuration and leave
# `tests/test_metadata_api.py` answering 401 to its own keys.


def test_the_warmup_runs_the_text_pipeline():
    """Proof that it warms rather than merely returning.

    The stemmer's own cache is the observable: it only grows when real tokens have gone
    through `prepare` → candidates → keywords, which is the path that pulls in the frequency
    table, the sentence segmenter and the stemmer together.
    """
    from app.lifecycle import _warmup_metadata

    stem.cache_clear()  # otherwise this only passes while no earlier test stemmed these words

    _warmup_metadata()

    assert stem.cache_info().currsize > 0


def test_the_warmup_never_fails_startup(monkeypatch, caplog):
    """Best-effort, exactly like `_warmup_models`: a broken warmup is a log line, not a boot
    failure. Nothing here is required for the endpoint to work — it only moves a cost."""
    import app.lifecycle as lifecycle

    def explode(*_args, **_kwargs):
        raise RuntimeError("no frequency data")

    monkeypatch.setattr(lifecycle, "generate", explode)

    with caplog.at_level(logging.WARNING):
        lifecycle._warmup_metadata()

    assert any("warmup" in record.message.lower() for record in caplog.records)


def test_the_setting_is_off_by_default():
    assert Settings(_env_file=None).warmup_metadata is False


def test_the_lifespan_only_warms_metadata_when_the_setting_is_on():
    """An AST rule rather than a boot test: the cost this guards is ~58 MB of resident
    memory, and a refactor that drops the guard would be invisible in a passing suite."""
    lifecycle_py = Path(__file__).resolve().parents[1] / "app" / "lifecycle.py"
    source = lifecycle_py.read_text(encoding="utf-8")
    lifespan = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )

    guarded = [
        node for node in ast.walk(lifespan)
        if isinstance(node, ast.If)
        and any(
            isinstance(sub, ast.Attribute) and sub.attr == "warmup_metadata"
            for sub in ast.walk(node.test)
        )
        and any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "_warmup_metadata"
            for sub in ast.walk(node)
        )
    ]

    assert guarded, "lifespan must call _warmup_metadata() only under settings.warmup_metadata"



def test_simultaneous_first_requests_load_the_frequency_table_once(monkeypatch):
    """M07 (audit 2026-09-30): wordfreq loads its German table behind an lru_cache, which is
    thread-safe but not single-flight -- eight simultaneous first requests each read it: 3.3 s
    and a 554 MB peak, against 0.3 s and 101 MB for one load."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import wordfreq

    from app.metadata import phrases

    wordfreq.get_frequency_dict.cache_clear()
    wordfreq.get_frequency_list.cache_clear()
    wordfreq._wf_cache.clear()
    phrases._idf.cache_clear()
    monkeypatch.setattr(phrases, "_table_loaded", False)
    reads: list[str] = []
    real_read = wordfreq.read_cBpack

    def counting_read(filename):
        reads.append(filename)
        return real_read(filename)

    monkeypatch.setattr(wordfreq, "read_cBpack", counting_read)
    start = threading.Barrier(8)

    def first_request(index: int) -> float:
        start.wait()
        return phrases._idf(f"Unterrichtswort{index}")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(first_request, range(8)))

    assert len(reads) == 1, f"the table was read {len(reads)} times"
