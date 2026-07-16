"""Dataset loading, cleaning, label preparation and train/val/test splitting.

Kept deliberately free of heavy ML imports so it loads fast and is easy to test.
Low-RAM oriented: CSVs are read with ``usecols`` and ``dtype=str``. Read-only
inspection/statistics for the API live in ``dataset_stats`` (which consumes this).
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from .errors import TrainingInputError


def read_csv(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV as UTF-8, falling back to cp1252 (Windows-1252) — common for
    German metadata exports — so a legitimate non-UTF-8 file is not an opaque
    failure. Empty/malformed CSVs surface as ``TrainingInputError`` (→ 400)
    rather than a raw pandas error (→ 500)."""
    try:
        try:
            return pd.read_csv(path, encoding="utf-8", **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp1252", **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise TrainingInputError(f"The CSV is empty or malformed: {exc}") from exc


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")  # [text](url) / ![alt](url) -> text/alt
_MD_MARK_RE = re.compile(r"[*_`~#>]+")  # emphasis / code / heading / quote markers
_WS_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize text for vectorization.

    Decodes HTML entities, strips HTML tags and common Markdown markup, removes
    control characters and collapses whitespace.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(str(value))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_MARK_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _clean_in_chunks(series: pd.Series, on_progress: Callable[[str], None] | None) -> pd.Series:
    """Apply clean_text per row; chunked so long runs can report row progress."""
    n = len(series)
    step = 50_000
    if n <= step or on_progress is None:
        return series.map(clean_text)
    parts: list[pd.Series] = []
    for start in range(0, n, step):
        parts.append(series.iloc[start:start + step].map(clean_text))
        done = min(start + step, n)
        on_progress(f"Cleaning texts … {done:,}/{n:,} rows")
    return pd.concat(parts)


def split_labels(value: object, separator: str = ",") -> list[str]:
    """Split a multilabel cell into a list of trimmed, non-empty labels."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(separator) if part.strip()]


# path -> (mtime_ns, size, rows). Bounded: cleared beyond 256 entries (the data
# dir holds a handful of CSVs; stale keys from deleted files are harmless).
_ROW_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}


def count_rows(path: str | Path) -> int:
    """Data rows (excluding the header) of a CSV.

    Cached by (mtime, size) so repeated dataset listings/info calls do not
    re-read unchanged multi-MB files line by line on every request.
    """
    path = Path(path)
    stat = path.stat()
    key = str(path)
    hit = _ROW_COUNT_CACHE.get(key)
    if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
        return hit[2]
    with open(path, encoding="utf-8", errors="ignore") as handle:
        rows = max(0, sum(1 for _ in handle) - 1)
    if len(_ROW_COUNT_CACHE) > 256:
        _ROW_COUNT_CACHE.clear()
    _ROW_COUNT_CACHE[key] = (stat.st_mtime_ns, stat.st_size, rows)
    return rows


@dataclass
class LoadedData:
    """Cleaned texts, their label lists, and an optional URI->label map."""

    texts: list[str]
    label_lists: list[list[str]]
    uri_to_label: dict[str, str]


def load_dataset(
    path: str | Path,
    text_columns: list[str],
    label_column: str,
    *,
    separator: str = ";",
    label_separator: str = ",",
    displayname_column: str | None = None,
    min_text_length: int = 5,
    drop_duplicates: bool = True,
    label_filter: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> LoadedData:
    """Load a CSV and return cleaned texts + label lists.

    Only the needed columns are read (low RAM). A ``<label>_DISPLAYNAME`` column
    (if present) is used to build a URI->human-readable-label mapping.
    """
    path = Path(path)
    header = read_csv(path, sep=separator, nrows=0)
    available = set(header.columns)

    text_cols = [c for c in text_columns if c in available]
    if not text_cols:
        raise TrainingInputError(
            f"No valid text columns. Requested {text_columns}; available {sorted(available)}"
        )
    if label_column not in available:
        raise TrainingInputError(f"Label column {label_column!r} not found in {sorted(available)}")

    dn_col = displayname_column or f"{label_column}_DISPLAYNAME"
    has_dn = dn_col in available
    usecols = list(dict.fromkeys([*text_cols, label_column, *([dn_col] if has_dn else [])]))

    def emit(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    emit("Reading CSV file …")
    df = read_csv(path, sep=separator, usecols=usecols, dtype=str, low_memory=False)

    combined = df[text_cols[0]].fillna("")
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna("")
    texts = _clean_in_chunks(combined, on_progress)

    label_series = df[label_column]
    uri_to_label: dict[str, str] = {}
    if has_dn:
        for uri_cell, name_cell in zip(label_series.fillna(""), df[dn_col].fillna(""), strict=False):
            uris = split_labels(uri_cell, label_separator)
            names = split_labels(name_cell, label_separator)
            for uri, name in zip(uris, names, strict=False):
                uri_to_label.setdefault(uri, name)

    label_lists = [split_labels(cell, label_separator) for cell in label_series]
    if label_filter:
        label_lists = [[lab for lab in labs if label_filter in lab] for labs in label_lists]

    emit("Filtering short/empty and duplicate rows …")
    out_texts: list[str] = []
    out_labels: list[list[str]] = []
    seen: set[str] = set()
    for text, labels in zip(texts.tolist(), label_lists, strict=False):
        if len(text) < min_text_length or not labels:
            continue
        if drop_duplicates:
            if text in seen:
                continue
            seen.add(text)
        out_texts.append(text)
        out_labels.append(labels)

    return LoadedData(texts=out_texts, label_lists=out_labels, uri_to_label=uri_to_label)


def detect_task_type(label_lists: list[list[str]], n_classes: int) -> str:
    """Infer 'binary' | 'multiclass' | 'multilabel' from the label structure."""
    max_per_sample = max((len(labs) for labs in label_lists), default=0)
    if max_per_sample <= 1:
        return "binary" if n_classes <= 2 else "multiclass"
    return "multilabel"


def auto_min_samples(n_samples: int, override: int | None = None) -> int:
    """Heuristic minimum samples per label, scaled to dataset size."""
    if override is not None:
        return max(1, override)
    if n_samples < 1_000:
        return 2
    if n_samples < 10_000:
        return 5
    if n_samples < 50_000:
        return 20
    return 35


def prepare_targets(
    label_lists: list[list[str]], min_samples: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Binarize labels, drop rare label columns and now-empty rows.

    Returns ``(Y, classes, row_keep_mask)``. Apply ``row_keep_mask`` to the
    texts to keep them aligned with ``Y``.
    """
    mlb = MultiLabelBinarizer(sparse_output=False)
    matrix = mlb.fit_transform(label_lists)
    col_keep = matrix.sum(axis=0) >= min_samples
    matrix = matrix[:, col_keep]
    classes = [c for c, keep in zip(mlb.classes_, col_keep, strict=False) if keep]
    row_keep = matrix.sum(axis=1) > 0
    return matrix[row_keep], classes, row_keep


def three_way_split(
    n: int, *, val_size: float, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) index arrays for a random split."""
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(rest_idx, test_size=rel_test, random_state=seed)
    return train_idx, val_idx, test_idx
