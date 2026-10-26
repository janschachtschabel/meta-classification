"""The corrections read back as a whole: the admin view, the backup, its import, and a dataset.

``feedback`` is the store -- append-only, one JSON object per line -- and keeps the one job of
writing a correction down. Everything here reads the collection as a whole or adds to it in
bulk, so the corrections can be seen, carried between instances and trained on from the admin
UI, which before could only record them.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from . import dataset_files, feedback
from .schemas import ImportedCorrection
from .security import name_problem

# How much of a text the view carries: enough to recognise it, without shipping 100,000
# characters per row to a table. The backup and the dataset keep the whole text.
VIEW_TEXT_CHARS = 500


def _view(entry: dict) -> dict:
    raw = entry.get("text")
    text = raw if isinstance(raw, str) else ""
    return {
        "recorded_at": entry.get("recorded_at"),
        "model_name": entry.get("model_name"),
        "text": text[:VIEW_TEXT_CHARS],
        "text_truncated": len(text) > VIEW_TEXT_CHARS,
        "predicted": entry.get("predicted") or [],
        "corrected": entry.get("corrected") or [],
        "source": entry.get("source"),
    }


def page(model: str | None, limit: int, offset: int) -> dict:
    """The corrections newest first, ``limit`` of them after skipping ``offset``, optionally
    of one model only -- with the total and how many each model has, for the filter.

    Two passes over the file rather than one list of it: the collection is uncapped by design,
    and an admin page should cost the page, not the volume. A correction recorded between the
    passes can shift the window by one; the next request sees it.
    """
    models: Counter[str] = Counter()
    total = 0
    for entry in feedback.iter_entries():
        name = str(entry.get("model_name", ""))
        models[name] += 1
        total += model is None or name == model
    # Newest first over an oldest-first file: the window, counted from the start.
    first, last = max(0, total - offset - limit), max(0, total - offset)
    items: list[dict] = []
    seen = 0
    for entry in feedback.iter_entries():
        if seen >= last:
            break
        if model is not None and str(entry.get("model_name", "")) != model:
            continue
        if seen >= first:
            items.append(_view(entry))
        seen += 1
    items.reverse()
    return {"total": total, "models": dict(sorted(models.items())), "items": items}


def backup_lines() -> Iterator[str]:
    """Every correction as one JSON line, every field kept -- what ``import_lines`` takes back.

    Re-serialised rather than the file copied: a line a crash tore is left out, as every
    other reader leaves it out.
    """
    for entry in feedback.iter_entries():
        yield json.dumps(entry, ensure_ascii=False) + "\n"


def _identity(entry: dict) -> bytes:
    """What makes two corrections the same one: when, for which model, which text, what was
    said and what is right. The source is left out -- a restore names itself differently."""
    key = [entry.get(field) for field in ("recorded_at", "model_name", "text", "predicted", "corrected")]
    return hashlib.blake2b(json.dumps(key, ensure_ascii=False).encode("utf-8"), digest_size=16).digest()


def import_lines(lines: Iterable[str]) -> dict:
    """Add the corrections of a backup that are not already collected; all of them or none.

    Every line meets the rules a correction posted live meets (``FeedbackRequest``) plus a
    ``recorded_at``, and its model name the rule every name meets; a line that does not is
    counted as invalid and skipped, not fatal -- a backup from another version should bring
    back what it can. Exact repeats, of the collection or within the file, are counted and
    skipped: importing one backup twice must not double the training data.

    :raises FeedbackWriteError: the new corrections would pass the cap, or the file cannot be
        written -- in which case nothing was added.
    """
    known = {_identity(entry) for entry in feedback.iter_entries()}
    new: list[dict] = []
    duplicates = invalid = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            checked = ImportedCorrection.model_validate(raw)
        except (ValueError, ValidationError):
            invalid += 1
            continue
        if name_problem(checked.model_name, "model name"):
            invalid += 1
            continue
        # The stamp as it was written, not as the parser normalised it: a restored record is
        # the record, and the duplicate check compares exactly what the backup carried.
        entry = {**checked.model_dump(mode="json", exclude={"recorded_at"}), "recorded_at": raw["recorded_at"]}
        identity = _identity(entry)
        if identity in known:
            duplicates += 1
            continue
        known.add(identity)
        new.append(entry)
    if new:
        feedback.append_many(new)
    return {"imported": len(new), "duplicates": duplicates, "invalid": invalid}


def write_dataset(data_dir: Path, name: str) -> int:
    """Save the training export as a new dataset ``name`` in ``data_dir``; returns its rows.

    Exactly the CSV ``GET /feedback/export`` hands out, written through the path every new
    dataset takes (``dataset_files``): staged where no route can name it, then linked into
    place, so an existing dataset is refused and never replaced.

    :raises FileExistsError: a dataset of that name exists.
    :raises ValueError: no correction carries a label -- such a file could not train.
    """
    target = data_dir / name
    if target.exists():
        raise FileExistsError(target)
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = dataset_files.staging_path(data_dir, name)
    rows = -1  # the header is the first line written
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            for chunk in feedback.iter_csv():
                handle.write(chunk)
                rows += 1
        if rows < 1:
            raise ValueError("No correction carries a label yet, so there is nothing to train on.")
        dataset_files.publish_new(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return rows
