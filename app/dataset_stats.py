"""Read-only dataset inspection for the ``/datasets`` endpoints: a preview
sample plus label/text statistics and basic validation.

Split out of ``data`` so the load/clean/target-prep core stays focused; this
module only *consumes* that core (``dataset_load.load_dataset``, ``read_csv``) —
the dependency is one-directional (stats -> data), never the reverse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import auto_min_samples, read_csv, split_labels
from .dataset_load import load_dataset
from .profiles import CapacityPlan, estimated_minutes

_SHIPPED_PROFILES = ("fast", "auto", "best")


def sample_rows(path: str | Path, *, separator: str = ";", n: int = 5) -> dict:
    """Columns and the first ``n`` rows of a CSV (for ``GET /datasets/{name}``).

    Delegated from the route so encoding fallback + empty/malformed handling live
    in one place; raises ``TrainingInputError`` (→ 400) on an unreadable CSV.
    """
    sample = read_csv(path, sep=separator, dtype=str, nrows=n)
    return {"columns": list(sample.columns), "sample": sample.fillna("").to_dict(orient="records")}


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
    label_only = read_csv(path, sep=separator, usecols=[label_column], dtype=str)
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
