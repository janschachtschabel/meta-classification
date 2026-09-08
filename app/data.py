"""Dataset loading, cleaning, label preparation and train/val/test splitting.

Kept deliberately free of heavy ML imports so it loads fast and is easy to test.
Low-RAM oriented: CSVs are read with ``usecols`` and ``dtype=str``. Read-only
inspection/statistics for the API live in ``dataset_stats`` (which consumes this).
"""

from __future__ import annotations

import gzip
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


def is_container_label(label: str) -> bool:
    """Does this value name a NAMESPACE rather than a concept?

    A trailing ``/`` means "members of", not "a member" — in URIs as in paths. Such a value
    is a tagging accident, never a class worth learning: measured on ``data_300k.csv``, the
    bare vocabulary root ``…/vocabs/discipline/`` was attached to 522 rows and trained as
    an ordinary label scoring F1 0.4096, diluting macro F1 and letting ``/predict`` answer
    with a label that carries no meaning.

    ``min_samples_per_label`` cannot catch this — 522 rows clears any sane threshold — so
    the guard has to be structural. It is deliberately narrow: a genuine broader concept
    has an id (``…/discipline/120``) and is kept, because the label hierarchy is real
    signal (the higher-education vocabulary averages 2.25 levels per row).
    """
    return label.endswith("/")


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


@dataclass
class LoadedData:
    """Cleaned texts, their label lists, and an optional URI->label map."""

    texts: list[str]
    label_lists: list[list[str]]
    uri_to_label: dict[str, str]


def _rejoin_split_names(fragments: list[str]) -> list[str]:
    """Undo a split caused by a separator character INSIDE a display name.

    Label URIs and their display names arrive as two lists sharing one separator, but
    only URIs are guaranteed free of it. German orthography then says which fragment is
    a continuation rather than a new name: a lowercase start, a preceding compound half
    ("Rechts-"), or an unclosed parenthesis.

    Measured on ``data_300k.csv``: this reconstructs 29.6% of the damaged rows, and where
    the result could be cross-checked against rows that were never damaged it agreed
    75/75 times — i.e. precise but not complete, which is why ``_pair_names`` still
    refuses to guess when it does not reconcile.
    """
    merged: list[str] = []
    for fragment in fragments:
        continues = bool(merged) and (
            fragment[:1].islower()
            or merged[-1].endswith("-")
            or merged[-1].count("(") > merged[-1].count(")")
        )
        if continues:
            merged[-1] = f"{merged[-1]}, {fragment}"
        else:
            merged.append(fragment)
    return merged


def _pair_names(uris: list[str], names: list[str]) -> list[tuple[str, str]]:
    """Pair URIs with display names, but ONLY when the two provably line up.

    The URI count is authoritative. A positional zip of unequal lists silently shifts
    every later name onto the wrong URI — on ``data_300k.csv`` that gave 34 of 119
    higher-education labels the name of a *different* subject, which reads as a
    confident statement rather than as missing data. So when the counts cannot be
    reconciled, this contributes nothing and callers fall back to the URI.
    """
    if len(uris) == len(names):
        return list(zip(uris, names, strict=True))
    repaired = _rejoin_split_names(names)
    if len(repaired) == len(uris):
        return list(zip(uris, repaired, strict=True))
    return []


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
    text_column_weights: dict[str, int] | None = None,
    label_names: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> LoadedData:
    """Load a CSV and return cleaned texts + label lists.

    Only the needed columns are read (low RAM). A ``<label>_DISPLAYNAME`` column
    (if present) is used to build a URI->human-readable-label mapping.

    ``text_column_weights`` maps a column to how often its text is repeated in the
    combined training text (default 1). Repetition is what a TF-IDF backend
    understands as "this field matters more": a title drowning in a long
    description gets its term frequency back. ``sublinear_tf`` damps it
    logarithmically, so a weight of 2 is worth ~1.7x, not 2x. Weights for columns
    the CSV does not have are ignored, exactly like the columns themselves.
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

    weights = text_column_weights or {}
    weighted_cols = [col for col in text_cols for _ in range(max(1, int(weights.get(col, 1))))]
    combined = df[weighted_cols[0]].fillna("")
    for col in weighted_cols[1:]:
        combined = combined + " " + df[col].fillna("")
    texts = _clean_in_chunks(combined, on_progress)

    label_series = df[label_column]
    uri_to_label: dict[str, str] = {}
    if has_dn:
        for uri_cell, name_cell in zip(label_series.fillna(""), df[dn_col].fillna(""), strict=False):
            for uri, name in _pair_names(
                split_labels(uri_cell, label_separator),
                split_labels(name_cell, label_separator),
            ):
                uri_to_label.setdefault(uri, name)
    if label_names:
        # An external vocabulary is authoritative: it overrides CSV-derived names and
        # fills the ones no row could attribute. Narrowed to labels this dataset uses,
        # so a full vocabulary file does not bloat every bundle.
        used = {uri for cell in label_series.fillna("") for uri in split_labels(cell, label_separator)}
        uri_to_label.update(
            {uri: name for uri, name in label_names.items() if uri in used and name}
        )

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
    n: int, *, val_size: float, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) index arrays for a random split."""
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(rest_idx, test_size=rel_test, random_state=seed)
    return train_idx, val_idx, test_idx
