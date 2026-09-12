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

    exported = feedback.to_csv()
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
    rows = list(csv.DictReader(io.StringIO(feedback.to_csv()), delimiter=";"))
    assert [row["text"] for row in rows] == ["Der Wiener Kongress"]


def test_a_text_with_the_separator_in_it_survives_the_export(store):
    """Editorial text contains semicolons and quotes. Hand-joining the columns would
    split one row across two and corrupt everything after it."""
    feedback.append(_correction('Ein Titel; mit "Zeichen"', ["uri:hist"]))

    rows = list(csv.DictReader(io.StringIO(feedback.to_csv()), delimiter=";"))
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
    rows = feedback.to_csv().splitlines()
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
    path.write_text(feedback.to_csv(), encoding="utf-8")

    data = load_dataset(path, ["text"], "labels",
                        separator=feedback.CSV_SEPARATOR,
                        label_separator=feedback.LABEL_SEPARATOR)

    assert data.texts == ["Der Wiener Kongress von 1815", "Bruchrechnung und Gleichungen"]
    assert data.label_lists == [["uri:hist"], ["uri:math", "uri:stats"]]
