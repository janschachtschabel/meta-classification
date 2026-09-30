"""CSV reading, text cleaning, label preparation and train/val/test splitting.

Kept deliberately free of heavy ML imports so it loads fast and is easy to test.
The pieces here are what reads a CSV at all (``read_csv``: in the encoding the file's
bytes call for, see ``csv_encoding``; empty/malformed files as ``TrainingInputError``) and
what a row becomes.
Assembling a training dataset out of them — in blocks of rows, the one step that
holds a whole dataset — is ``dataset_load``; read-only inspection/statistics for the
API live in ``dataset_stats``. Both consume this module, never the reverse.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from . import stratify
from .csv_encoding import CsvEncoding, detect, is_gzipped
from .errors import TrainingInputError
from .label_names import is_container_label
from .markup import CONTROL_RE, MD_LINK_RE, MD_MARK_RE, strip_tags

# What a read of a few rows (``nrows``) judges the encoding on: a preview must not scan a
# whole export first, and the header plus a handful of rows sit well inside this.
PREVIEW_BYTES = 4 << 20


def read_csv(path: str | Path, encoding: CsvEncoding | None = None, **kwargs: object) -> pd.DataFrame:
    """Read a CSV in ``encoding``, or in the one its bytes call for (``csv_encoding.detect``;
    judged on the first ``PREVIEW_BYTES`` when only ``nrows`` rows are read).

    Empty/malformed CSVs, and one that is neither UTF-8 nor Windows-1252, surface as
    ``TrainingInputError`` (→ 400) rather than a raw pandas error (→ 500).
    """
    if encoding is None:
        encoding = detect(path, limit=PREVIEW_BYTES if kwargs.get("nrows") is not None else None)
    try:
        return pd.read_csv(path, encoding=encoding.name, encoding_errors=encoding.errors, **kwargs)
    except UnicodeDecodeError as exc:
        # Only a preview can get here: its rows ran past the part the decision was made on.
        raise TrainingInputError(f"The CSV is not valid {encoding.name} at byte {exc.start}.") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise TrainingInputError(f"The CSV is empty or malformed: {exc}") from exc


_WS_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize text for vectorization.

    Decodes HTML entities, strips HTML tags and common Markdown markup, removes
    control characters and collapses whitespace.

    The result is one line of single-spaced tokens, which is what the fitted vectorizers
    saw, so this may flatten freely — marker runs of any length, markers inside a word,
    and script bodies all go, none of which ``strip_markup_preserving_lines`` may touch.
    The order of the steps below is equally load-bearing; both places say why.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = strip_tags(str(value))
    text = MD_LINK_RE.sub(r"\1", text)
    text = MD_MARK_RE.sub("", text)
    # Control characters LAST, as they always were: removed earlier, a mangled byte
    # between "]" and "(" closes up into a Markdown link that then collapses to its
    # text. See app/markup.py's CONTROL_RE.
    text = CONTROL_RE.sub("", text)
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


def resolve_dataset(data_dir: Path, name: str) -> Path:
    """The path of an existing dataset, or ``FileNotFoundError``.

    One function so the rule cannot drift between the routes that apply it. The data directory
    holds more than datasets — ``label_names.json`` is the display-name sidecar every training
    reads — and a route that checked only ``exists()`` accepted it, answered 202, and surfaced
    the problem minutes later as a job error.

    Raises ``FileNotFoundError``, not an HTTP error: the caller owns the status code, and this
    module is one the routes delegate to. The name must already have been through
    ``security.safe_name``; that check belongs at the trust boundary and stays there.
    """
    path = data_dir / name
    if not is_dataset_name(name) or not path.exists():
        raise FileNotFoundError(name)
    return path


# path -> (mtime_ns, size, rows). Bounded: cleared beyond 256 entries (the data
# dir holds a handful of CSVs; stale keys from deleted files are harmless).
_ROW_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}


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
    """Binarize labels, drop the label columns there is nothing to learn from, and the
    rows that leaves without a label.

    Returns ``(Y, classes, row_keep_mask)``. Apply ``row_keep_mask`` to the
    texts to keep them aligned with ``Y``.

    A label needs ``min_samples`` rows WITH it and as many WITHOUT it. The second half is
    the mirror of the first: a label on every row teaches nothing, and sklearn fits it as a
    `_ConstantPredictor` -- a type the skops guard rightly refuses, so the run reported
    `completed` and its model could never be loaded (audit 2026-09-30, T01). Repeated until
    nothing changes, because dropping the rows a drop left empty takes negatives away from
    the labels that stay: a label missing only from rows whose labels are all too rare is,
    once those rows are gone, on every row that is left.

    Binarized SPARSE, filtered there, and densified only once the rare columns are
    gone — because the cost has to follow the labels that survive, not the ones that
    arrived. A free-text label column brings them in bulk: the WLO export has 321 702
    distinct keywords over 274 804 rows, and asking numpy for that densely (as int64,
    which is what the binarizer returns) is 659 GiB. It was killed by signal 9 in this
    function, seconds before the filter below would have left 8 279 columns.

    The result is dense int8: it holds only 0/1, and sklearn's default int64 would
    spend 8 bytes per bit — 1.34 GB at 600k rows x 300 labels versus 168 MB. The
    dtype is narrowed while still sparse, so densifying allocates the int8 size and
    never an int64 copy of it. sklearn's metrics and the OneVsRest fit accept int8
    unchanged, and everything downstream indexes ``Y`` as a dense array.
    """
    mlb = MultiLabelBinarizer(sparse_output=True)
    matrix = mlb.fit_transform(label_lists)
    classes = list(mlb.classes_)
    rows = np.arange(matrix.shape[0])
    while True:
        positives = np.asarray(matrix.sum(axis=0)).ravel()
        col_keep = (positives >= min_samples) & (len(rows) - positives >= min_samples)
        matrix = matrix[:, col_keep]
        classes = [c for c, keep in zip(classes, col_keep, strict=True) if keep]
        labelled = np.asarray(matrix.sum(axis=1)).ravel() > 0
        matrix, rows = matrix[labelled], rows[labelled]
        if col_keep.all() and labelled.all():
            break
    row_keep = np.zeros(len(label_lists), dtype=bool)
    row_keep[rows] = True
    return matrix.astype(np.int8).toarray(), classes, row_keep


def three_way_split(
    n: int, *, val_size: float, test_size: float, seed: int, y: np.ndarray | None = None,
    train_only: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) index arrays.

    Random by default. Given ``y`` (the n x n_labels target matrix) the split is
    multilabel-stratified instead, so each label keeps its share in all three parts —
    it is the rare ones that a shuffle misplaces, and macro F1 weights those equally.
    See ``stratify.stratified_partition``.

    ``train_only`` (bool per row) marks rows an LLM wrote or touched: the three parts are
    drawn from the OTHER rows in the proportions above, and every train-only row joins
    train. Without such a row the split is exactly the one made without the argument.
    """
    if train_only is not None and train_only.any():
        real = np.flatnonzero(~train_only)
        train, val, test = three_way_split(
            len(real), val_size=val_size, test_size=test_size, seed=seed,
            y=None if y is None else y[real])
        return np.concatenate([real[train], np.flatnonzero(train_only)]), real[val], real[test]
    if y is not None:
        train, val, test = stratify.stratified_partition(
            y, [1.0 - val_size - test_size, val_size, test_size], seed=seed)
        return train, val, test
    # The two parts are sized as COUNTS, derived from n once. Given a share instead,
    # sklearn rounds each call up on its own — and the second call's share is a share OF
    # THE REST, so the two roundings compound: 0.1/0.2 over 100 rows came out 69/10/21 and
    # 0.25/0.05 came out 70/24/6, i.e. a test split a fifth larger than the one the bundle
    # goes on to report as `n_test`. Rounding once against n keeps every part within a row
    # of its share, and the remainder makes train, so the three still sum to n. Where the
    # chained form already landed on these numbers — 0.15/0.15, the default every
    # `scripts/benchmark_*.py` measures with — sklearn derives the same integers from the
    # floats, so those splits keep the exact rows they had and stay comparable.
    indices = np.arange(n)
    n_val, n_test = round(n * val_size), round(n * test_size)
    train_idx, rest_idx = train_test_split(indices, test_size=n_val + n_test, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=n_test, random_state=seed)
    return train_idx, val_idx, test_idx
