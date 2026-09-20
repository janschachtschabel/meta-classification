"""Corrections an editor makes to a prediction, and the CSV they train from.

The recognition rate improves with use or it improves only with a new export. This is
the first half of the loop: what a person noticed, written down in a shape a training
run can read back.
"""

import csv
import io

import pytest

from app import feedback
from app.errors import FeedbackWriteError


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "_feedback_path", lambda: path)
    # The count is process state (see the module): a fresh file means a fresh process.
    monkeypatch.setattr(feedback, "_count", None)
    return path


def _correction(text: str, corrected: list[str], **fields) -> dict:
    return {"text": text, "model_name": "subjects", "predicted": ["uri:math"],
            "corrected": corrected, "source": "ui", **fields}


def test_a_correction_is_written_down_and_read_back(store):
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))

    entries = feedback.read_all()
    assert len(entries) == 1
    assert entries[0]["corrected"] == ["uri:hist"]
    assert entries[0]["recorded_at"], "stamped on the way in, not by the caller"


def test_corrections_are_never_dropped(store):
    """The job history is capped at 200 because it is a log. This is not a log — it IS
    the data the next training run learns from, and the oldest correction is exactly as
    valuable as the newest. Appended, never rewritten."""
    for index in range(250):
        feedback.append(_correction(f"text {index}", ["uri:hist"]))

    entries = feedback.read_all()
    assert len(entries) == 250
    assert entries[0]["text"] == "text 0", "the oldest survives"


def test_the_export_is_a_csv_a_training_run_can_read(store):
    """"Training-compatible" is checkable: the separators are the ones /train defaults
    to, and the two columns are the ones the form asks for."""
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))
    feedback.append(_correction("Bruchrechnung", ["uri:math", "uri:stats"]))

    exported = "".join(feedback.iter_csv())
    rows = list(csv.DictReader(io.StringIO(exported), delimiter=";"))
    assert list(rows[0]) == ["text", "labels"]
    assert rows[0]["text"] == "Der Wiener Kongress"
    assert rows[1]["labels"] == "uri:math,uri:stats", "the label separator /train defaults to"


def test_a_correction_with_no_label_is_kept_but_not_exported(store):
    """"None of these apply" is a real thing to say and worth recording. It is not
    trainable though — the loader drops label-less rows — so exporting it would
    overstate how many rows the file actually contributes."""
    feedback.append(_correction("Ein Text ohne passendes Label", []))
    feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))

    assert len(feedback.read_all()) == 2
    rows = list(csv.DictReader(io.StringIO("".join(feedback.iter_csv())), delimiter=";"))
    assert [row["text"] for row in rows] == ["Der Wiener Kongress"]


def test_a_text_with_the_separator_in_it_survives_the_export(store):
    """Editorial text contains semicolons and quotes. Hand-joining the columns would
    split one row across two and corrupt everything after it."""
    feedback.append(_correction('Ein Titel; mit "Zeichen"', ["uri:hist"]))

    rows = list(csv.DictReader(io.StringIO("".join(feedback.iter_csv())), delimiter=";"))
    assert rows[0]["text"] == 'Ein Titel; mit "Zeichen"'
    assert rows[0]["labels"] == "uri:hist"


def test_a_damaged_line_costs_that_line_only(store):
    """The file sits on a mounted volume and is appended to by a running server. A
    half-written last line must not take the export down with it."""
    store.write_text('{"text": "good", "corrected": ["uri:hist"]}\n{ truncated\n',
                     encoding="utf-8")

    assert [entry["text"] for entry in feedback.read_all()] == ["good"]
    feedback.append(_correction("after", ["uri:math"]))
    assert [entry["text"] for entry in feedback.read_all()] == ["good", "after"]


def test_a_correction_that_cannot_be_saved_fails_instead_of_vanishing(tmp_path, monkeypatch):
    """The one failure this module must not swallow.

    The job history logs a warning and moves on, because by then the run is finished and
    saved — the record is only a record. A correction is the opposite: it IS the work, and
    nothing regenerates it. An editor told "recorded" closes the tab, so a silent failure
    costs exactly the thing the feedback loop exists to collect. It fails loudly instead,
    with a message that says what to fix and without the server's paths in it.
    """
    blocked = tmp_path / "feedback.jsonl"
    blocked.mkdir()  # a directory where the file belongs: the append's own write fails
    monkeypatch.setattr(feedback, "_feedback_path", lambda: blocked)
    monkeypatch.setattr(feedback, "_count", None)

    with pytest.raises(FeedbackWriteError) as failure:
        feedback.append(_correction("Der Wiener Kongress", ["uri:hist"]))

    assert "APIV3_FEEDBACK_FILE" in str(failure.value), "name the setting an operator fixes"
    assert str(blocked) not in str(failure.value), "a client message carries no server paths"

    # And it counted nothing: the next correction is the first one collected, not the
    # second. A phantom count would report progress that is not on disk.
    monkeypatch.setattr(feedback, "_feedback_path", lambda: tmp_path / "writable.jsonl")
    assert feedback.append(_correction("Der Wiener Kongress", ["uri:hist"])) == 1


def test_an_empty_store_exports_a_header_not_nothing(store):
    """A zero-byte download reads as a broken endpoint. A header row says "nothing
    collected yet", which is the truth."""
    rows = "".join(feedback.iter_csv()).splitlines()
    assert rows == ["text;labels"]


def test_the_export_actually_loads_as_a_dataset(store, tmp_path):
    """"Training-compatible" is a claim, and this is the only way to check it: hand the
    export to the same loader a run uses, with the separators the request defaults to,
    and see the rows come back. Asserting a header string would prove nothing about
    whether it trains.
    """
    from app.dataset_load import load_dataset

    feedback.append(_correction("Der Wiener Kongress von 1815", ["uri:hist"]))
    feedback.append(_correction("Bruchrechnung und Gleichungen", ["uri:math", "uri:stats"]))
    feedback.append(_correction("Nichts passt hier", []))

    path = tmp_path / "feedback.csv"
    path.write_text("".join(feedback.iter_csv()), encoding="utf-8")

    data = load_dataset(path, ["text"], "labels",
                        separator=feedback.CSV_SEPARATOR,
                        label_separator=feedback.LABEL_SEPARATOR)

    assert data.texts == ["Der Wiener Kongress von 1815", "Bruchrechnung und Gleichungen"]
    assert data.label_lists == [["uri:hist"], ["uri:math", "uri:stats"]]


def test_the_export_streams_without_holding_the_file(store):
    """Row by row, not file-then-list-then-string.

    The store is deliberately uncapped — it is training data, and the oldest correction
    is worth as much as the newest — so it grows for the life of the volume. The export
    read it whole, built a list of dicts from it, then a complete CSV string, then handed
    that to the response: three copies of an unbounded file, on the event loop of a
    single-worker server.

    Streaming is checkable without measuring memory: a generator has produced its header
    before anything has read the rest of the file.
    """
    for index in range(50):
        feedback.append(_correction(f"text {index}", ["uri:hist"]))

    rows = feedback.iter_csv()
    assert next(rows).startswith("text;labels"), "the header comes before the file is read"
    assert "".join([next(rows), next(rows)]).count("\n") == 2, "then one row at a time"



def _exported(**kwargs) -> list[str]:
    document = "".join(feedback.iter_csv(**kwargs))
    return [row["text"] for row in csv.DictReader(io.StringIO(document), delimiter=";")]


def test_the_export_pages_by_position_because_the_clock_is_too_coarse(store):
    """`offset` + `limit` are the cursor; `recorded_at` could not be one.

    Five appends in a row share a single microsecond value on this platform, so a stamp
    cursor would either skip every correction recorded in the boundary instant or hand it
    out twice — and a duplicated row in training data is not a harmless kind of wrong.
    The file is append-only and never rewritten, which is exactly what makes a position
    stable.
    """
    for index in range(5):
        feedback.append(_correction(f"text {index}", ["uri:hist"]))

    stamps = {entry["recorded_at"] for entry in feedback.read_all()}
    assert len(stamps) < 5, "the premise: the clock does not separate these writes"

    assert _exported(limit=2) == ["text 0", "text 1"]
    assert _exported(offset=3) == ["text 3", "text 4"]
    # Paging covers every correction exactly once, which is the property that matters.
    assert _exported(offset=0, limit=2) + _exported(offset=2, limit=2) + _exported(offset=4) == [
        f"text {index}" for index in range(5)]


def test_paging_addresses_exported_rows_not_recorded_ones(store):
    """A "none of these apply" correction is recorded and never exported, so counting it
    against the offset would make a caller's next page skip a real row."""
    feedback.append(_correction("keine Labels", []))
    feedback.append(_correction("erste", ["uri:hist"]))
    feedback.append(_correction("zweite", ["uri:math"]))

    assert _exported() == ["erste", "zweite"]
    assert _exported(offset=1) == ["zweite"]
