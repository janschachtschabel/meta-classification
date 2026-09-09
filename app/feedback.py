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
    """
    global _count
    path = _feedback_path()
    entry = {**record, "recorded_at": datetime.now(UTC).isoformat()}
    with _lock:
        if _count is None:
            # First write of this process: establish the count from what a READER sees,
            # so a line that never finished being written is not counted as a correction.
            _count = len(read_all())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
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


def to_csv() -> str:
    """The corrections as a CSV a training run can read directly.

    Written with ``csv.writer`` because editorial text carries semicolons and quotes,
    and hand-joining would split one row across two and corrupt everything after it.

    Corrections with no corrected label are recorded but NOT exported: "none of these
    apply" is a real thing to say, and it is also not trainable — ``load_dataset`` drops
    label-less rows, so exporting them would overstate what the file contributes.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=CSV_SEPARATOR, lineterminator="\n")
    writer.writerow(EXPORT_COLUMNS)
    for entry in read_all():
        labels = [label for label in entry.get("corrected") or [] if isinstance(label, str)]
        text = entry.get("text")
        if not labels or not isinstance(text, str):
            continue
        writer.writerow([text, LABEL_SEPARATOR.join(labels)])
    return buffer.getvalue()
