"""Loading a training dataset: a CSV in, cleaned texts, label lists and display names out.

Split out of ``data``, which keeps the per-row pieces this is assembled from
(``read_csv``, ``clean_text``, ``split_labels``). Loading is the one step that holds a
whole dataset at once, so it is where a training run's first memory peak is decided.

The CSV is read in blocks of rows. Read whole, the file sat in memory three times over
— the frame, the combined text, the cleaned text: +1.5 GB for a 558 MB export
(docs/plans/2026-09-11-training-memory.md). In blocks, each of those is one block long,
and only what the dataset keeps accumulates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import clean_text, read_csv, split_labels
from .errors import TrainingInputError
from .label_names import pair_names

# Rows per block: the step the loader already reported its cleaning progress in.
CHUNK_ROWS = 50_000


@dataclass
class LoadedData:
    """Cleaned texts, their label lists, and an optional URI->label map."""

    texts: list[str]
    label_lists: list[list[str]]
    uri_to_label: dict[str, str]


def combine_text_columns(
    frame: pd.DataFrame, text_columns: list[str], weights: dict[str, int] | None = None
) -> pd.Series:
    """Assemble one text per row: each column repeated as often as its weight says.

    Extracted from ``load_dataset`` because classifying a CSV has to assemble its input
    EXACTLY the way training did. A model fit on "title title description" sits on a
    different feature distribution than one fit on "title description", and its tuned
    thresholds sit on that distribution too — two copies of this loop would drift into
    exactly that skew, silently and with plausible-looking numbers.

    Repetition is what a TF-IDF backend understands as "this field matters more": a
    title drowning in a long description gets its term frequency back. ``sublinear_tf``
    damps it logarithmically, so a weight of 2 is worth ~1.7x, not 2x.
    """
    weights = weights or {}
    repeated = [col for col in text_columns for _ in range(max(1, int(weights.get(col, 1))))]
    combined = frame[repeated[0]].fillna("")
    for col in repeated[1:]:
        combined = combined + " " + frame[col].fillna("")
    return combined


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
    chunk_rows: int = CHUNK_ROWS,
) -> LoadedData:
    """Load a CSV and return cleaned texts + label lists.

    Only the needed columns are read, ``chunk_rows`` rows at a time (low RAM). A
    ``<label>_DISPLAYNAME`` column (if present) is used to build a URI->human-readable-
    label mapping.

    ``text_column_weights`` maps a column to how often its text is repeated in the
    combined training text (default 1) — see :func:`combine_text_columns`. Weights for
    columns the CSV does not have are ignored, exactly like the columns themselves.

    UTF-8 first; a file that turns out not to be UTF-8 anywhere is read again from the
    start as cp1252 (common for German metadata exports), whatever was read before is
    discarded — the whole-file reader behaved the same way. The blocks before the first
    non-UTF-8 byte have been cleaned by then, so such a file costs up to one extra
    cleaning pass; resuming mid-file instead would mix two decodings of one file.
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
    for encoding in ("utf-8", "cp1252"):
        collector = _Collector(
            text_cols=text_cols, label_column=label_column, dn_col=dn_col if has_dn else None,
            label_separator=label_separator, label_filter=label_filter,
            min_text_length=min_text_length, drop_duplicates=drop_duplicates,
            weights=text_column_weights,
        )
        try:
            for block in _read_blocks(path, encoding, separator=separator, usecols=usecols,
                                      chunk_rows=chunk_rows):
                collector.add(block)
                emit(f"Reading and cleaning … {collector.rows_read:,} rows")
        except UnicodeDecodeError:
            if encoding == "cp1252":
                raise
            emit("Not UTF-8 — reading the file again as Windows-1252 …")
            continue
        return collector.result(label_names)
    raise AssertionError("unreachable: the cp1252 attempt returns or raises")


def _read_blocks(
    path: Path, encoding: str, *, separator: str, usecols: list[str], chunk_rows: int
) -> Iterator[pd.DataFrame]:
    """The CSV in blocks of rows, only the needed columns, every cell as text. Empty or
    malformed CSVs surface as ``TrainingInputError`` (-> 400), like ``data.read_csv``.

    ``engine="c"``: a separator longer than one character would otherwise switch pandas
    to its python engine and be read as a regular expression. The API allows one
    character; this refuses the rest (ValueError), as the whole-file read did.
    """
    try:
        with pd.read_csv(path, sep=separator, usecols=usecols, dtype=str, encoding=encoding,
                         chunksize=chunk_rows, engine="c") as reader:
            yield from reader
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise TrainingInputError(f"The CSV is empty or malformed: {exc}") from exc


@dataclass
class _Collector:
    """The dataset being loaded, one block of rows at a time.

    Everything that spans blocks lives here, so a block boundary can change nothing: the
    texts already kept (the first occurrence wins across the whole file), the display
    names (the first pairing wins), and the labels in use (the authoritative names are
    narrowed to them once, at the end).
    """

    text_cols: list[str]
    label_column: str
    dn_col: str | None  # None: the CSV has no display-name column
    label_separator: str
    label_filter: str | None
    min_text_length: int
    drop_duplicates: bool
    weights: dict[str, int] | None
    texts: list[str] = field(default_factory=list)
    label_lists: list[list[str]] = field(default_factory=list)
    uri_to_label: dict[str, str] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)
    rows_read: int = 0

    def add(self, frame: pd.DataFrame) -> None:
        cleaned = combine_text_columns(frame, self.text_cols, self.weights).map(clean_text)
        label_series = frame[self.label_column]
        if self.dn_col is not None:
            names = frame[self.dn_col].fillna("")
            for uri_cell, name_cell in zip(label_series.fillna(""), names, strict=False):
                for uri, name in pair_names(
                    split_labels(uri_cell, self.label_separator),
                    split_labels(name_cell, self.label_separator),
                ):
                    self.uri_to_label.setdefault(uri, name)
        label_lists = [split_labels(cell, self.label_separator) for cell in label_series]
        self.used.update(uri for labels in label_lists for uri in labels)
        if self.label_filter:
            label_lists = [[lab for lab in labs if self.label_filter in lab] for labs in label_lists]
        for text, labels in zip(cleaned.tolist(), label_lists, strict=False):
            if len(text) < self.min_text_length or not labels:
                continue
            if self.drop_duplicates:
                if text in self.seen:
                    continue
                self.seen.add(text)
            self.texts.append(text)
            self.label_lists.append(labels)
        self.rows_read += len(frame)

    def result(self, label_names: dict[str, str] | None) -> LoadedData:
        if label_names:
            # An external vocabulary is authoritative: it overrides CSV-derived names and
            # fills the ones no row could attribute. Narrowed to labels this dataset uses,
            # so a full vocabulary file does not bloat every bundle.
            self.uri_to_label.update(
                {uri: name for uri, name in label_names.items() if uri in self.used and name}
            )
        return LoadedData(texts=self.texts, label_lists=self.label_lists,
                          uri_to_label=self.uri_to_label)
