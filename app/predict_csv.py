"""Classify every row of a CSV and stream the answers back as CSV.

The daily editorial job is "classify these 500 new items", not one text. Doing that
through ``/predict`` means the caller assembles the text per row — and the way a text
is assembled is part of what the model was fit on, so that is exactly the thing not to
leave to the caller. Here the columns and their weights come out of the bundle.

Nothing is materialised: the CSV is read in chunks, each chunk is classified and
written out, and the result leaves as it is produced. A 200 MB input therefore costs
one chunk of rows, not the file. It is parsed once completely before the stream starts
(``check_input``): a failure after the 200 can only cut the answer short, and it did,
silently (audit 2026-09-30, V06).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .classifier import ClassifierModel
from .csv_encoding import CsvEncoding, detect
from .data import read_csv
from .dataset_load import combine_text_columns
from .errors import TrainingInputError

# Rows per read/predict step. The model vectorizes a whole chunk at once, so bigger is
# faster and heavier; 500 keeps a chunk's feature matrix small enough to sit beside a
# loaded model on a 2 GB container.
CHUNK_ROWS = 500
# Rows per step of the check's parse: nothing is predicted there, so the text columns of
# this many rows are all it holds.
CHECK_ROWS = 50_000
OUTPUT_FIELDS = ("row", "uri", "label", "confidence", "above_threshold")


@dataclass(frozen=True)
class CheckedInput:
    """A CSV that will classify to its last row: how to decode it, and how many rows it has."""

    encoding: CsvEncoding
    rows: int


def check_input(path: Path, text_columns: list[str], *, separator: str,
                chunk_rows: int = CHECK_ROWS) -> CheckedInput:
    """Refuse a CSV that cannot be classified to its end, before anything is streamed.

    Once a streaming response has started, the status line is already 200 and a failure
    can only cut the answer short. That is what a broken row past the first chunk did: the
    parser failed mid-stream, and the caller got 500 of 700 rows, a clean end of transfer
    and no word of it (audit 2026-09-30, V06). So the whole file is parsed here, once --
    a fraction of what classifying it costs -- while a 400 is still possible, and the rows
    are counted for the caller to check the answer against. The encoding is decided here
    too, on the whole file (``csv_encoding``), because the stream cannot change its mind.

    :raises TrainingInputError: naming the columns that are missing and what is there, the
        row where the file stops parsing, or why it is neither UTF-8 nor Windows-1252.
    """
    encoding = detect(path)
    header = read_csv(path, encoding, sep=separator, nrows=0)
    missing = [column for column in text_columns if column not in header.columns]
    if missing:
        raise TrainingInputError(
            f"The CSV has no column {missing}; it has {sorted(header.columns)}. "
            "The model was trained on the columns it names, so those have to be present."
        )
    rows = sum(len(chunk) for chunk in _read_chunks(
        path, text_columns, separator=separator, chunk_rows=chunk_rows, encoding=encoding))
    return CheckedInput(encoding, rows)


def _row_cells(index: int, predictions: list) -> Iterator[list]:
    if not predictions:
        # 500 items go in and "which ones did the model refuse" has to be readable off
        # the result: a row that vanished is indistinguishable from one never sent.
        yield [index, "", "", "", ""]
        return
    for prediction in predictions:
        yield [
            index,
            prediction.uri,
            prediction.label,
            f"{prediction.confidence:.2f}",
            # Only ranking mode (explicit top_k) sets it; blank means "not applicable",
            # which is not the same statement as "did not pass".
            "" if prediction.above_threshold is None else str(prediction.above_threshold).lower(),
        ]


def classify_csv(
    path: Path,
    model: ClassifierModel,
    *,
    text_columns: list[str],
    weights: dict[str, int] | None = None,
    separator: str = ";",
    threshold: float | None = None,
    top_k: int | None = None,
    chunk_rows: int = CHUNK_ROWS,
    encoding: CsvEncoding | None = None,
) -> Iterator[str]:
    """Yield the result CSV in pieces: a header, then one piece per chunk of input rows.

    One output row per predicted label, carrying the 0-based number of the input row it
    came from — that number is how a caller joins the answers back onto their own file.
    A row the model asserts nothing for still gets a line.

    Call :func:`check_input` first: a generator cannot report a bad file. Its
    ``encoding`` is the one to pass here (decided again when omitted).
    """
    buffer = io.StringIO()
    # csv.writer, not string joining: a display name like 'Politik, "Wirtschaft"' is
    # ordinary WLO data and would otherwise split across columns and corrupt every
    # following row.
    writer = csv.writer(buffer, lineterminator="\n")

    def flush() -> str:
        text = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        return text

    writer.writerow(OUTPUT_FIELDS)
    yield flush()

    offset = 0
    for chunk in _read_chunks(path, text_columns, separator=separator, chunk_rows=chunk_rows,
                              encoding=encoding or detect(path)):
        # The raw assembly goes to the model, which cleans its own input — the same
        # single cleaning step training applied to the same combined string.
        texts = combine_text_columns(chunk, text_columns, weights).tolist()
        predictions = model.predict(texts, top_k=top_k, threshold=threshold)
        for index, row in enumerate(predictions):
            writer.writerows(_row_cells(offset + index, row))
        offset += len(texts)
        yield flush()


def _read_chunks(
    path: Path, text_columns: list[str], *, separator: str, chunk_rows: int, encoding: CsvEncoding,
) -> Iterator[pd.DataFrame]:
    """The CSV in blocks of rows, holding only the text columns (low RAM).

    Goes to pandas directly rather than through ``data.read_csv``: with ``chunksize``
    that call returns a reader, not a frame. pandas raises a parser error while the
    reader is ITERATED, at the block that holds the broken row -- the conversion has to
    cover the iteration, not just the call that makes the reader.
    """
    try:
        with pd.read_csv(
            path, sep=separator, usecols=text_columns, dtype=str, chunksize=chunk_rows,
            encoding=encoding.name, encoding_errors=encoding.errors,
        ) as reader:
            yield from reader
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise TrainingInputError(f"The CSV is empty or malformed: {exc}") from exc
