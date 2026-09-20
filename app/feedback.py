"""Corrections an editor makes to a prediction, and the CSV a training run reads back.

The recognition rate improves with use, or it improves only when somebody produces a
new export. This is the first half of that loop: what a person noticed, written down in
a shape ``/train`` can consume directly.

Append-only, and deliberately uncapped — unlike ``job_history``, which is a log bounded
at 200 entries. This is not a log: it IS the data the next run learns from, and the
oldest correction is worth exactly as much as the newest. So a line is appended and the
file is never rewritten, which also means a crash can cost at most the line being
written rather than the whole collection.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
from datetime import UTC, datetime

from .errors import FeedbackWriteError
from .settings import get_settings

logger = logging.getLogger("api_v3.feedback")

# The shape a training run reads. Both separators are what ``TrainRequest`` defaults to,
# so the export trains with the form's own defaults and needs nothing explained.
EXPORT_COLUMNS = ("text", "labels")
CSV_SEPARATOR = ";"
LABEL_SEPARATOR = ","

_lock = threading.Lock()
# How many corrections are on disk. Counted once from the file and then kept, because
# the alternative — re-reading it on every append — is quadratic on a collection that is
# meant to grow for the life of the volume. Safe to hold in memory: single-worker design,
# one writer.
_count: int | None = None


def _feedback_path():
    return get_settings().feedback_file


def append(record: dict) -> int:
    """Record one correction; returns how many have been collected in total.

    Stamped here rather than by the caller: when a correction was made is part of the
    record, and a client-supplied timestamp is a client-supplied claim.

    Raises :class:`~app.errors.FeedbackWriteError` if the correction could not be
    written. It does not come back as a count that is only true in memory.
    """
    global _count
    path = _feedback_path()
    entry = {**record, "recorded_at": datetime.now(UTC).isoformat()}
    with _lock:
        if _count is None:
            # First write of this process: establish the count from what a READER sees,
            # so a line that never finished being written is not counted as a correction.
            _count = len(read_all())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            # The one write in this app that may NOT degrade to a logged warning. The job
            # history can: by the time it is written the run is finished and saved, so the
            # record is only a record. This is the work itself, and nothing regenerates
            # it — an editor told "recorded" closes the tab. So the failure travels back
            # to whoever made the correction, with the setting to fix named and the
            # server's paths left in the log where they belong.
            logger.error("Could not write a correction to %s: %s", path, exc)
            raise FeedbackWriteError(
                "The correction was NOT recorded: its storage file could not be written. "
                "Check APIV3_FEEDBACK_FILE — it must point at a writable volume (under a "
                "read-only root filesystem the default path next to the code is not one). "
                "Nothing was lost yet; send the correction again once it is writable."
            ) from exc
        # Only now: a count raised past what is on disk would report collected work that
        # no export will ever contain.
        _count += 1
        return _count


def read_all() -> list[dict]:
    """Every readable correction, oldest first.

    The file is appended to by a running server and sits on a mounted volume, so a line
    that did not finish being written must cost that line — never the export that reads
    it.
    """
    path = _feedback_path()
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Feedback file at %s is unreadable; reporting none.", path)
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            logger.warning("Dropping an unparseable line from the feedback at %s.", path)
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _iter_entries():
    """Every readable correction, oldest first, one at a time.

    The line-by-line twin of :func:`read_all`, for the export: the file is uncapped by
    design, so materialising it to decide what to write scales with the collection rather
    than with the answer. Same forgiveness — a line that did not finish being written
    costs that line, never the export.
    """
    path = _feedback_path()
    if not path.exists():
        return
    try:
        handle = path.open(encoding="utf-8")
    except OSError:
        logger.warning("Feedback file at %s is unreadable; reporting none.", path)
        return
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                logger.warning("Dropping an unparseable line from the feedback at %s.", path)
                continue
            if isinstance(entry, dict):
                yield entry


def iter_csv(limit: int | None = None, offset: int = 0):
    """The corrections as CSV, yielded row by row.

    Streaming rather than returning a string: the response can start before the file has
    been read, and neither the entries nor the finished document is ever held whole.

    ``offset`` skips that many exportable corrections and ``limit`` caps how many follow,
    both oldest-first, so they compose into a cursor: take a page, add its size to the
    offset, ask again. The cursor is a POSITION and not the ``recorded_at`` stamp it would
    be natural to reach for — the clock is coarser than the writes (on Windows five
    appends in a row share one microsecond value), so a stamp cursor would either skip
    every correction sharing the boundary instant or hand it out twice. Duplicated rows in
    training data are not a harmless kind of wrong. Position is exact here precisely
    because the file is append-only and never rewritten.

    Rows without a corrected label are skipped as ever, and count against neither
    ``offset`` nor ``limit``: both address the exported rows, which is what a caller pages
    through.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=CSV_SEPARATOR, lineterminator="\n")

    def drain() -> str:
        text = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        return text

    writer.writerow(EXPORT_COLUMNS)
    yield drain()

    seen = written = 0
    for entry in _iter_entries():
        if limit is not None and written >= limit:
            return
        labels = [label for label in entry.get("corrected") or [] if isinstance(label, str)]
        text = entry.get("text")
        if not labels or not isinstance(text, str):
            continue
        seen += 1
        if seen <= offset:
            continue
        writer.writerow([text, LABEL_SEPARATOR.join(labels)])
        written += 1
        yield drain()
