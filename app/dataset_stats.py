"""Read-only dataset inspection for the ``/datasets`` endpoints: a preview
sample plus label/text statistics and basic validation.

Split out of ``data`` so the load/clean/target-prep core stays focused; this
module only *consumes* that core (``dataset_load.load_dataset``, ``read_csv``) —
the dependency is one-directional (stats -> data), never the reverse.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .csv_encoding import is_gzipped
from .data import auto_min_samples, read_csv, split_labels
from .dataset_load import load_dataset
from .profile_costs import estimated_minutes
from .provenance import ENRICHED, EXAMPLE, GENERATED, MARK_COLUMNS, block_marks

if TYPE_CHECKING:
    # Annotation only: `analyze_dataset` is handed a plan and calls two methods on
    # it. Importing `capacity` for real would pull `Settings` into a module whose
    # whole job is reading a file it was given.
    from .capacity import CapacityPlan

_SHIPPED_PROFILES = ("fast", "auto", "best")


def sample_rows(path: str | Path, *, separator: str = ";", n: int = 5) -> dict:
    """Columns and the first ``n`` rows of a CSV (for ``GET /datasets/{name}``).

    Delegated from the route so encoding fallback + empty/malformed handling live
    in one place; raises ``TrainingInputError`` (→ 400) on an unreadable CSV.
    """
    sample = read_csv(path, sep=separator, dtype=str, nrows=n)
    return {"columns": list(sample.columns), "sample": sample.fillna("").to_dict(orient="records")}


# The AI-row counts per file version, like `data.count_rows`: (mtime_ns, size, counts).
_AI_ROWS_CACHE: dict[str, tuple[int, int, dict]] = {}
# Mark cells handed to `block_marks` at a time: a frame per block, never one of the whole file.
_MARK_BLOCK_ROWS = 50_000


def ai_rows(path: str | Path, *, separator: str, columns: list[str]) -> dict | None:
    """How many rows of a CSV data-prep marked as written or touched by an LLM, per kind,
    and how many carry any mark (a row can carry two); ``None`` without mark columns.

    ``columns`` is the header already read. The cells are judged by the rule a training run
    applies (``provenance.block_marks``), as written, so the training form's count and the
    run's agree; ``train`` mode, because the count describes the file, not one choice of
    what to do with it.

    Read like ``data.count_rows`` -- a ``csv.reader`` pass, cached per file version -- and
    not through ``read_csv``: on a 533 MB export that took 22.5 s on every call, 16.5 s of
    it deciding the encoding. A mark needs no encoding, only whether its cell is blank, and
    UTF-8 with replacement answers that for any bytes: one UTF-8 cannot decode still leaves
    a character.
    """
    present = [c for c in MARK_COLUMNS if c in columns]
    if not present:
        return None
    path = Path(path)
    stat = path.stat()
    key = f"{path}|{separator}"
    hit = _AI_ROWS_CACHE.get(key)
    if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
        return dict(hit[2])
    counts = {"generated": 0, "example": 0, "enriched": 0, "marked": 0}
    opener = gzip.open if is_gzipped(path) else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=separator)
        header = next(reader, [])
        where = {c: header.index(c) for c in present if c in header}
        block: dict[str, list[str]] = {c: [] for c in where}

        def count_block() -> None:
            if not where or not block[next(iter(where))]:
                return
            marks = block_marks(pd.DataFrame(block), mode="train")
            counts["generated"] += int(np.count_nonzero(marks & GENERATED))
            counts["example"] += int(np.count_nonzero(marks & EXAMPLE))
            counts["enriched"] += int(np.count_nonzero(marks & ENRICHED))
            counts["marked"] += int(np.count_nonzero(marks))
            for cells in block.values():
                cells.clear()

        for row in reader:
            for column, at in where.items():
                block[column].append(row[at] if at < len(row) else "")
            if where and len(block[next(iter(where))]) >= _MARK_BLOCK_ROWS:
                count_block()
        count_block()
    if len(_AI_ROWS_CACHE) > 256:
        _AI_ROWS_CACHE.clear()
    _AI_ROWS_CACHE[key] = (stat.st_mtime_ns, stat.st_size, counts)
    return dict(counts)


def analyze_dataset(
    path: str | Path,
    text_columns: list[str],
    label_column: str,
    *,
    separator: str = ";",
    label_separator: str = ",",
    label_filter: str | None = None,
    plan: CapacityPlan | None = None,
) -> dict:
    """Comprehensive dataset statistics for the /datasets/analyze endpoint.

    ``plan`` is what this server grants a run right now; with it the estimate counts the
    head-fit threads each profile would get, and says how many (``None``: the anchor's).
    """
    data = load_dataset(
        path,
        text_columns,
        label_column,
        separator=separator,
        label_separator=label_separator,
        min_text_length=0,
        drop_duplicates=False,
        label_filter=label_filter,
    )
    lengths = np.array([len(t) for t in data.texts]) if data.texts else np.array([0])
    per_sample = np.array([len(labs) for labs in data.label_lists]) if data.label_lists else np.array([0])
    counts: dict[str, int] = {}
    for labs in data.label_lists:
        for lab in labs:
            counts[lab] = counts.get(lab, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    total = len(data.texts)
    costed = [name for name in _SHIPPED_PROFILES if estimated_minutes(name, total) is not None]
    threads = {name: plan.head_fit_threads(name, total) for name in costed} if plan else {}
    return {
        "file": str(path),
        "total_samples": total,
        "unique_labels": len(counts),
        # The two numbers that decide whether to start a run at all, computed here so
        # nothing downstream re-derives them: the threshold that fits this size (the
        # heuristic the training request would apply for `null`), and what each profile
        # costs. The threshold analysis below says how many labels survive the
        # recommendation, which is what turns it from a number into a decision.
        "recommended_min_samples_per_label": auto_min_samples(total),
        "estimated_minutes": {
            name: estimated_minutes(name, total, threads.get(name)) for name in costed
        },
        # The threads each estimate assumed: the deploy fit's, under this server's CPU and
        # memory budgets. Fewer than requested is a run the memory budget will slow down.
        **({"planned_head_fit_threads": threads, "threads_requested": plan.requested_threads}
           if plan else {}),
        "text_statistics": {
            "mean_length_chars": float(lengths.mean()),
            "median_length_chars": float(np.median(lengths)),
        },
        "label_statistics": {
            "mean_labels_per_sample": float(per_sample.mean()),
            "max_labels_per_sample": int(per_sample.max()),
        },
        # The recommendation joins the fixed buckets: "keeps N of M labels" is the
        # sentence that turns it from a number into a decision, and on a small dataset
        # the heuristic answers 2, which none of the fixed buckets covers.
        "label_threshold_analysis": {
            f"labels_with_{t}+_samples": int(sum(1 for c in counts.values() if c >= t))
            for t in sorted({5, 10, 20, 35, 50, 100, auto_min_samples(total)})
        },
        "top_20_labels": dict(top[:20]),
        "rare_labels_under_10": {k: v for k, v in counts.items() if v < 10},
    }


def validate_dataset(
    path: str | Path,
    text_columns: list[str],
    label_column: str,
    *,
    separator: str = ";",
    label_separator: str = ",",
) -> dict:
    """Check column presence and basic data-quality warnings."""
    header = read_csv(path, sep=separator, nrows=0)
    available = set(header.columns)
    errors: list[str] = []
    missing = [c for c in text_columns if c not in available]
    if missing:
        errors.append(f"Missing text columns: {missing}")
    if label_column not in available:
        errors.append(f"Missing label column: {label_column}")
    if errors:
        return {"valid": False, "errors": errors, "warnings": [], "available_columns": sorted(available)}

    data = load_dataset(
        path, text_columns, label_column, separator=separator,
        label_separator=label_separator, min_text_length=0, drop_duplicates=False,
    )
    counts: dict[str, int] = {}
    for labs in data.label_lists:
        for lab in labs:
            counts[lab] = counts.get(lab, 0) + 1
    warnings: list[str] = []
    rare = [lab for lab, c in counts.items() if c < 10]
    if rare:
        warnings.append(f"{len(rare)} labels with fewer than 10 samples")
    # load_dataset silently DROPS unlabeled rows, so data.label_lists can never
    # contain an empty list — count them from the raw label column instead.
    # As written, like the loader reads it: "NA" is a label, not a missing cell (T08).
    label_only = read_csv(path, data.encoding, sep=separator, usecols=[label_column], dtype=str,
                          keep_default_na=False)
    empty = int(sum(1 for cell in label_only[label_column]
                    if not split_labels(cell, label_separator)))
    if empty:
        warnings.append(f"{empty} rows without labels")
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "statistics": {"rows_with_labels": len(data.texts), "unique_labels": len(counts)},
    }
