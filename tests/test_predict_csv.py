"""Tests for classifying a whole CSV: text assembly, output shape, encodings.

The model is a stub here on purpose — what is under test is the CSV layer (assembly,
chunking, quoting, the row bookkeeping), not the classifier. One end-to-end run with a
real model lives in ``test_api.py``.
"""

import csv
import io
from pathlib import Path

import pytest

from app import data, dataset_load, predict_csv
from app.classifier import Prediction
from app.errors import TrainingInputError

TEXT_COLS = ["title", "keywords"]
LABEL_COL = "labels"


class _StubModel:
    """Answers with one canned prediction row per text and records what it was given."""

    def __init__(self, rows: list[list[Prediction]]) -> None:
        self.rows = rows
        self.seen: list[str] = []

    def predict(self, texts, **_kwargs):
        self.seen.extend(texts)
        return [self.rows[i % len(self.rows)] for i in range(len(texts))]


def _csv(tmp_path: Path, rows: list[tuple[str, str, str]], encoding: str = "utf-8") -> Path:
    path = tmp_path / "input.csv"
    body = "title;keywords;labels\n" + "".join(f"{t};{k};{lab}\n" for t, k, lab in rows)
    path.write_text(body, encoding=encoding)
    return path


def _run(path: Path, model, **kwargs) -> list[list[str]]:
    stream = predict_csv.classify_csv(
        path, model, text_columns=TEXT_COLS,
        weights=kwargs.pop("weights", None), separator=";", **kwargs,
    )
    return list(csv.reader(io.StringIO("".join(stream))))


def test_the_text_is_assembled_exactly_as_training_would_have_built_it(tmp_path):
    """A model fit on "title title keywords" sits on a different feature distribution
    than one fit on "title keywords" — and its tuned thresholds sit on that
    distribution with it. Classifying a CSV must therefore assemble its input the way
    training did, which is why both go through one function instead of two loops that
    merely look alike. This test is what makes that claim checkable.
    """
    rows = [
        ("Bruchrechnung im Unterricht", "Mathematik Brueche", "uri:math"),
        ("Photosynthese der Pflanzen", "Biologie Blatt", "uri:bio"),
        ("Der Wiener Kongress 1815", "Geschichte Europa", "uri:hist"),
    ]
    path = _csv(tmp_path, rows)
    weights = {"title": 2}

    training = dataset_load.load_dataset(
        path, TEXT_COLS, LABEL_COL, separator=";", text_column_weights=weights,
        drop_duplicates=False,
    )
    model = _StubModel([[Prediction(uri="uri:x", label="X", confidence=0.5)]])
    _run(path, model, weights=weights)

    # load_dataset cleans; the model cleans its own input, so the CSV path hands over
    # the raw assembly. Compare after the same cleaning step.
    assert [data.clean_text(text) for text in model.seen] == training.texts
    assert "Bruchrechnung im Unterricht Bruchrechnung im Unterricht" in model.seen[0], (
        "the weighted column is repeated, in place, before the next column"
    )


def test_a_row_that_gets_no_label_still_appears(tmp_path):
    """500 items go in, and the answer to "which ones did the model refuse" has to be
    readable off the result. A row that simply vanished would be indistinguishable from
    one the caller forgot to include."""
    path = _csv(tmp_path, [("Ein langer Titel hier", "stichwort", "uri:a")] * 2)
    model = _StubModel([[], [Prediction(uri="uri:a", label="A", confidence=0.75)]])

    rows = _run(path, model)
    assert rows[0] == ["row", "uri", "label", "confidence", "above_threshold"]
    assert rows[1] == ["0", "", "", "", ""], "the refused row is reported, not dropped"
    assert rows[2] == ["1", "uri:a", "A", "0.75", ""]


def test_a_label_containing_the_separator_survives(tmp_path):
    """Display names are free text out of a dataset: "Politik, Wirtschaft" is a real
    WLO label. Hand-joining the output with commas would split it across two columns
    and silently corrupt every row after it."""
    path = _csv(tmp_path, [("Ein langer Titel hier", "stichwort", "uri:a")])
    model = _StubModel([[Prediction(uri="uri:a", label='Politik, "Wirtschaft"', confidence=0.5)]])

    rows = _run(path, model)
    assert rows[1] == ["0", "uri:a", 'Politik, "Wirtschaft"', "0.50", ""]


def test_a_ranking_reports_whether_each_label_also_passed_its_threshold(tmp_path):
    """With an explicit top_k the model returns the N most probable labels whether or
    not they clear their cut. Dropping that flag would present a forced ranking as if
    every entry had been asserted."""
    path = _csv(tmp_path, [("Ein langer Titel hier", "stichwort", "uri:a")])
    model = _StubModel([[
        Prediction(uri="uri:a", label="A", confidence=0.9, above_threshold=True),
        Prediction(uri="uri:b", label="B", confidence=0.1, above_threshold=False),
    ]])

    rows = _run(path, model, top_k=2)
    assert [row[-1] for row in rows[1:]] == ["true", "false"]


def test_a_cp1252_export_reads_like_it_does_for_training(tmp_path):
    """German metadata exports are commonly cp1252, which is why the training reader
    falls back to it. A chunked reader cannot decide that by parsing twice — the second
    attempt would come after bytes were already streamed to the caller — so the
    encoding is settled up front, in one pass, before anything is sent."""
    path = _csv(tmp_path, [("Öffentliche Anhörung", "Bürgerbeteiligung", "uri:a")],
                encoding="cp1252")
    model = _StubModel([[Prediction(uri="uri:a", label="A", confidence=0.5)]])

    _run(path, model)
    assert model.seen == ["Öffentliche Anhörung Bürgerbeteiligung"]


def test_a_missing_text_column_is_refused_before_a_single_byte_is_streamed(tmp_path):
    """Once a streaming response starts, the status line is already 200 and a failure
    can only arrive as garbage in the body. The header is therefore checked first."""
    path = _csv(tmp_path, [("Ein langer Titel hier", "stichwort", "uri:a")])

    with pytest.raises(TrainingInputError, match="beschreibung"):
        predict_csv.check_columns(path, ["title", "beschreibung"], separator=";")
    predict_csv.check_columns(path, TEXT_COLS, separator=";")  # present: no complaint


def test_rows_are_numbered_across_chunk_boundaries(tmp_path):
    """The row number is how a caller joins the answers back onto their own file, so it
    counts input rows, not rows within a chunk."""
    path = _csv(tmp_path, [("Ein langer Titel hier", "stichwort", "uri:a")] * 5)
    model = _StubModel([[Prediction(uri="uri:a", label="A", confidence=0.5)]])

    rows = _run(path, model, chunk_rows=2)
    assert [row[0] for row in rows[1:]] == ["0", "1", "2", "3", "4"]
