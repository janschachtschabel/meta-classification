"""Classify every row of a CSV and stream the answers back as CSV.

The daily editorial job is "classify these 500 new items", not one text. Doing that
through ``/predict`` means the caller assembles the text per row — and the way a text
is assembled is part of what the model was fit on, so that is exactly the thing not to
leave to the caller. Here the columns and their weights come out of the bundle.

Nothing is materialised: the CSV is read in chunks, each chunk is classified and
written out, and the result leaves as it is produced. A 200 MB input therefore costs
one chunk of rows, not the file.
"""

from __future__ import annotations

import codecs
import csv
import io
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .classifier import ClassifierModel
from .data import read_csv
from .dataset_load import combine_text_columns
from .errors import TrainingInputError

# Rows per read/predict step. The model vectorizes a whole chunk at once, so bigger is
# faster and heavier; 500 keeps a chunk's feature matrix small enough to sit beside a
# loaded model on a 2 GB container.
CHUNK_ROWS = 500
OUTPUT_FIELDS = ("row", "uri", "label", "confidence", "above_threshold")


def check_columns(path: Path, text_columns: list[str], *, separator: str) -> None:
    """Refuse a CSV that lacks a required text column, before anything is streamed.

    Once a streaming response has started, the status line is already 200 and a failure
    can only reach the caller as garbage appended to a half-written CSV. Reading just
    the header settles it while a 400 is still possible.

    :raises TrainingInputError: naming the columns that are missing and what is there.
    """
    # data.read_csv's own utf-8 -> cp1252 fallback settles the header; only the
    # chunked body read needs the encoding decided in advance.
    header = read_csv(path, sep=separator, nrows=0)
    missing = [column for column in text_columns if column not in header.columns]
    if missing:
        raise TrainingInputError(
            f"The CSV has no column {missing}; it has {sorted(header.columns)}. "
            "The model was trained on the columns it names, so those have to be present."
        )


def csv_encoding(path: Path) -> str:
    """``utf-8`` if the whole file decodes as it, else ``cp1252``.

    ``data.read_csv`` decides this by parsing the file twice. A chunked reader cannot:
    the second attempt would come after bytes had already been streamed to the caller.
    One incremental pass settles it up front, reading blocks and keeping none — and
    German metadata exports really are commonly cp1252, so guessing is not an option.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    with path.open("rb") as handle:
        try:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                decoder.decode(block)
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            return "cp1252"
    return "utf-8"


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
) -> Iterator[str]:
    """Yield the result CSV in pieces: a header, then one piece per chunk of input rows.

    One output row per predicted label, carrying the 0-based number of the input row it
    came from — that number is how a caller joins the answers back onto their own file.
    A row the model asserts nothing for still gets a line.

    Call :func:`check_columns` first: a generator cannot report a bad header.
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
    for chunk in _read_chunks(path, text_columns, separator=separator, chunk_rows=chunk_rows):
        # The raw assembly goes to the model, which cleans its own input — the same
        # single cleaning step training applied to the same combined string.
        texts = combine_text_columns(chunk, text_columns, weights).tolist()
        predictions = model.predict(texts, top_k=top_k, threshold=threshold)
        for index, row in enumerate(predictions):
            writer.writerows(_row_cells(offset + index, row))
        offset += len(texts)
        yield flush()


def _read_chunks(
    path: Path, text_columns: list[str], *, separator: str, chunk_rows: int
) -> Iterator[pd.DataFrame]:
    """The CSV in blocks of rows, holding only the text columns (low RAM).

    Goes to pandas directly rather than through ``data.read_csv``: with ``chunksize``
    that call returns a reader, not a frame, and its encoding fallback works by parsing
    the file a second time — which is precisely what a stream cannot do.

    A malformed row deep in the file therefore aborts a download that has already
    started. The row numbers in the output say where it stopped, which is the best a
    stream can offer; the header, the part worth a clean 400, is checked before any of
    this runs.
    """
    try:
        reader = pd.read_csv(
            path, sep=separator, usecols=text_columns, dtype=str, chunksize=chunk_rows,
            encoding=csv_encoding(path),
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise TrainingInputError(f"The CSV is empty or malformed: {exc}") from exc
    with reader:
        yield from reader
