"""Loading a training dataset: a CSV in, cleaned texts, label lists and display names out.

Split out of ``data``, which keeps the per-row pieces this is assembled from
(``read_csv``, ``clean_text``, ``split_labels``). Loading is the one step that holds a
whole dataset at once, so it is where a training run's first memory peak is decided.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import clean_text, read_csv, split_labels
from .errors import TrainingInputError
from .label_names import pair_names


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
) -> LoadedData:
    """Load a CSV and return cleaned texts + label lists.

    Only the needed columns are read (low RAM). A ``<label>_DISPLAYNAME`` column
    (if present) is used to build a URI->human-readable-label mapping.

    ``text_column_weights`` maps a column to how often its text is repeated in the
    combined training text (default 1) — see :func:`combine_text_columns`. Weights for
    columns the CSV does not have are ignored, exactly like the columns themselves.
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

    texts = _clean_in_chunks(combine_text_columns(df, text_cols, text_column_weights), on_progress)

    label_series = df[label_column]
    uri_to_label: dict[str, str] = {}
    if has_dn:
        for uri_cell, name_cell in zip(label_series.fillna(""), df[dn_col].fillna(""), strict=False):
            for uri, name in pair_names(
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
