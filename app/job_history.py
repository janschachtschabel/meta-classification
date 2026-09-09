"""What every finished training run leaves behind, on disk.

Comparing two runs means opening two bundles and reading their ``metrics.json`` — and a
run that FAILED leaves nothing to open at all, so its reason lived only in the browser
tab that happened to be watching. This file is where a run's outcome survives the tab,
the restart and the deploy.

Separate from ``jobs`` on purpose: that module owns a thread and its shared state, this
one owns a file. One changes when the run loop does, the other when the record does.

JSON Lines rather than one JSON array: an append is a line, and a half-written last line
costs that line instead of the whole document — which matters for a file a container can
be killed in the middle of writing.
"""

from __future__ import annotations

import json
import logging
import os
import threading

from .settings import get_settings

logger = logging.getLogger("api_v3.jobs")

# Enough to compare a week of runs, small enough to rewrite on every append (a few
# finishes an hour at most). Without a cap this grows for the life of the volume.
MAX_RECORDS = 200

# The file is rewritten whole on append (see `append`), so two runners finishing at the
# same moment must not interleave. Only ever held around the read-modify-write, never
# around anything that could block on the job state.
_lock = threading.Lock()


def _history_path():
    return get_settings().job_history_file


def _read_lines() -> list[dict]:
    """Every readable record, oldest first. Unreadable lines are dropped, not fatal.

    The file sits on a mounted volume next to the models. A line that did not finish
    being written must cost that line — never the endpoint that reads it, and never a
    training run, which writes here only after the work is already done.
    """
    path = _history_path()
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Job history at %s is unreadable; reporting none.", path)
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            logger.warning("Dropping an unparseable line from the job history at %s.", path)
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def append(record: dict) -> None:
    """Add one finished run, keeping only the newest :data:`MAX_RECORDS`.

    Written whole through a temporary file and a rename, the same discipline the model
    bundles and the share store use: a crash mid-write cannot leave a history that no
    longer parses.
    """
    path = _history_path()
    with _lock:
        entries = (_read_lines() + [record])[-MAX_RECORDS:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
            encoding="utf-8",
        )
        os.replace(tmp, path)


def recent(limit: int = 50) -> list[dict]:
    """The most recent runs, newest first — the order the question is asked in."""
    with _lock:
        return list(reversed(_read_lines()))[:limit]
