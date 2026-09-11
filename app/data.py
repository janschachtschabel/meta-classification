"""CSV reading, text cleaning, label preparation and train/val/test splitting.

Kept deliberately free of heavy ML imports so it loads fast and is easy to test.
The pieces here are what reads a CSV at all (``read_csv``: UTF-8 with a cp1252
fallback, empty/malformed files as ``TrainingInputError``) and what a row becomes.
Assembling a training dataset out of them — in blocks of rows, the one step that
holds a whole dataset — is ``dataset_load``; read-only inspection/statistics for the
API live in ``dataset_stats``. Both consume this module, never the reverse.
"""

from __future__ import annotations

import gzip
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from . import stratify
from .errors import TrainingInputError
from .label_names import is_container_label


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


def split_labels(value: object, separator: str = ",") -> list[str]:
    """Split a multilabel cell into a list of trimmed, non-empty, learnable labels.

    Container values are dropped here rather than downstream so that every consumer —
    training, dataset statistics, validation — sees the same label set. A row left with no
    label is then dropped by the caller, exactly as for ``label_filter``.
    """
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return []
    return [
        part.strip()
        for part in str(value).split(separator)
        if part.strip() and not is_container_label(part.strip())
    ]


# The suffixes a dataset file carries. Everything else in the data directory is
# not a dataset (label_names.json is the sidecar every training reads).
DATASET_SUFFIXES = (".csv", ".csv.gz")


def is_dataset_name(name: str) -> bool:
    """Does this file name denote a dataset the API may serve, share or delete?"""
    return name.lower().endswith(DATASET_SUFFIXES)


# path -> (mtime_ns, size, rows). Bounded: cleared beyond 256 entries (the data
# dir holds a handful of CSVs; stale keys from deleted files are harmless).
_ROW_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}


def is_gzipped(path: str | Path) -> bool:
    """Does this dataset path denote a gzip-compressed CSV?

    Keyed off the name, exactly like pandas' own ``compression="infer"``, so what the reader
    and the row counter consider compressed can never disagree.
    """
    return str(path).lower().endswith(".gz")


def count_rows(path: str | Path) -> int:
    """Data rows (excluding the header) of a CSV.

    A line break inside a quoted field does not start a record, and on this data that is the
    difference between a number and a wrong number: counting physical lines reported
    1,343,683 rows for the 340,630 records of ``data_300k.csv`` (3.94x) and 2,141,123 for the
    426,724 of the combined WLO export (5.02x), because descriptions are full of newlines.

    Quote PARITY per line is enough to track this and stays a single cheap pass — an escaped
    ``""`` contributes two quotes and so leaves the parity untouched. A full ``csv.reader``
    pass would also be exact but parses every field for a number nobody trains on.

    Gzip is decompressed first; counting newlines in the COMPRESSED bytes is meaningless.
    Cached by (mtime, size) so repeated listings do not re-read multi-MB files.
    """
    path = Path(path)
    stat = path.stat()
    key = str(path)
    hit = _ROW_COUNT_CACHE.get(key)
    if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
        return hit[2]
    opener = gzip.open if is_gzipped(path) else open
    rows = 0
    inside_quotes = False
    with opener(path, "rt", encoding="utf-8", errors="ignore", newline="") as handle:
        for line in handle:
            if line.count('"') % 2:
                inside_quotes = not inside_quotes
            if not inside_quotes:
                rows += 1
    rows = max(0, rows - 1)  # header
    if len(_ROW_COUNT_CACHE) > 256:
        _ROW_COUNT_CACHE.clear()
    _ROW_COUNT_CACHE[key] = (stat.st_mtime_ns, stat.st_size, rows)
    return rows


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

    The matrix is int8: it is dense (rows x labels) and holds only 0/1, so sklearn's
    default int64 would spend 8 bytes per bit — 1.34 GB at 600k rows x 300 labels
    versus 168 MB. sklearn's metrics and the OneVsRest fit accept int8 unchanged.
    """
    mlb = MultiLabelBinarizer(sparse_output=False)
    matrix = mlb.fit_transform(label_lists).astype(np.int8, copy=False)
    col_keep = matrix.sum(axis=0) >= min_samples
    matrix = matrix[:, col_keep]
    classes = [c for c, keep in zip(mlb.classes_, col_keep, strict=False) if keep]
    row_keep = matrix.sum(axis=1) > 0
    return matrix[row_keep], classes, row_keep


def three_way_split(
    n: int, *, val_size: float, test_size: float, seed: int, y: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) index arrays.

    Random by default. Given ``y`` (the n x n_labels target matrix) the split is
    multilabel-stratified instead, so each label keeps its share in all three parts —
    it is the rare ones that a shuffle misplaces, and macro F1 weights those equally.
    See ``stratify.stratified_partition``.
    """
    if y is not None:
        train, val, test = stratify.stratified_partition(
            y, [1.0 - val_size - test_size, val_size, test_size], seed=seed)
        return train, val, test
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(rest_idx, test_size=rel_test, random_state=seed)
    return train_idx, val_idx, test_idx
