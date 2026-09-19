"""The loader carries data-prep's provenance marks alongside the texts it keeps.

A marked row trains but never validates; the loader is where that is first known, and
the one place that still sees the rows a dedupe drops.
"""

import pandas as pd
import pytest

from app.dataset_load import load_dataset
from app.provenance import ENRICHED, EXAMPLE, GENERATED, SAME_TEXT

MARKED = ("title", "labels", "generated_for", "example_for", "enriched_fields")


def _write(tmp_path, rows, columns=MARKED):
    path = tmp_path / "marked.csv"
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, sep=";", index=False)
    return path


def test_a_dataset_without_mark_columns_carries_no_marks(tmp_path):
    path = _write(tmp_path, [("Bruchrechnung", "math"), ("Photosynthese", "bio")],
                  columns=("title", "labels"))

    data = load_dataset(path, ["title"], "labels")

    assert data.marks is None
    assert data.excluded_generated == 0
    assert data.texts == ["Bruchrechnung", "Photosynthese"]


def test_each_kept_row_carries_its_marks(tmp_path):
    path = _write(tmp_path, [
        ("Bruchrechnung lernen", "math", "", "", ""),
        ("Brüche kürzen Übung", "math", "math", "", ""),
        ("Gleichungen lösen", "math", "", "math", ""),
        ("Zellbiologie Grundlagen", "bio", "", "", "title"),
    ])

    data = load_dataset(path, ["title"], "labels")

    assert data.texts == ["Bruchrechnung lernen", "Brüche kürzen Übung",
                          "Gleichungen lösen", "Zellbiologie Grundlagen"]
    assert data.marks == [0, GENERATED, EXAMPLE, ENRICHED]


@pytest.mark.parametrize("drop_duplicates", [True, False])
def test_a_row_with_the_text_of_a_marked_row_is_train_only_too(tmp_path, drop_duplicates):
    """The generator's example carries the mark, and its paraphrases are in the data.
    A second row with the same text is recognised, not classified — even when the dedupe
    drops the marked copy and keeps the unmarked one."""
    path = _write(tmp_path, [
        ("Brüche kürzen", "math", "", "", ""),
        ("Brüche kürzen", "math", "", "math", ""),
        ("Gleichungen lösen", "math", "", "", ""),
    ])

    data = load_dataset(path, ["title"], "labels", drop_duplicates=drop_duplicates)

    expected = [SAME_TEXT, 0] if drop_duplicates else [SAME_TEXT, EXAMPLE, 0]
    assert data.marks == expected


def test_exclude_drops_generated_rows_before_the_dedupe(tmp_path):
    """Dropped after the dedupe, a generated first occurrence would take its real twin
    with it. And with the generated rows gone, an example is an ordinary row."""
    path = _write(tmp_path, [
        ("Brüche kürzen", "math", "math", "", ""),
        ("Brüche kürzen", "math", "", "", ""),
        ("Gleichungen lösen", "math", "", "math", ""),
        ("Zellbiologie", "bio", "", "", "title"),
    ])

    data = load_dataset(path, ["title"], "labels", synthetic_rows="exclude")

    assert data.texts == ["Brüche kürzen", "Gleichungen lösen", "Zellbiologie"]
    assert data.marks == [0, 0, ENRICHED]
    assert data.excluded_generated == 1


@pytest.mark.parametrize(("drop_duplicates", "expected"), [
    # The dedupe keeps the first copy, and a marked copy it drops still marks that one.
    (True, [SAME_TEXT, 0, GENERATED, ENRICHED]),
    (False, [SAME_TEXT, 0, GENERATED, EXAMPLE, ENRICHED, SAME_TEXT]),
])
@pytest.mark.parametrize("chunk_rows", [1, 2, 3])
def test_the_marks_do_not_depend_on_the_block_size(tmp_path, chunk_rows, drop_duplicates, expected):
    rows = [
        ("Säuren und Basen", "chem", "", "", ""),
        ("Gedichte der Romantik", "german", "", "", ""),
        ("Kaiser Augustus", "hist", "hist", "", ""),
        ("Säuren und Basen", "chem", "", "chem", ""),
        ("Grammatik Kommaregeln", "german", "", "", "title"),
        ("Kaiser Augustus", "hist", "", "", ""),
    ]
    path = _write(tmp_path, rows)
    whole = load_dataset(path, ["title"], "labels", drop_duplicates=drop_duplicates)

    blocks = load_dataset(path, ["title"], "labels", drop_duplicates=drop_duplicates,
                          chunk_rows=chunk_rows)

    assert (blocks.texts, blocks.marks) == (whole.texts, whole.marks)
    assert whole.marks == expected


def test_an_unknown_synthetic_rows_mode_is_refused(tmp_path):
    path = _write(tmp_path, [("Bruchrechnung", "math")], columns=("title", "labels"))

    with pytest.raises(ValueError, match="synthetic_rows"):
        load_dataset(path, ["title"], "labels", synthetic_rows="keep")


def test_only_generated_rows_that_would_have_trained_count_as_left_out(tmp_path):
    """A row too short to train on was never going to be used; counting it as "left
    out" would overstate what exclude removed (review #6)."""
    path = _write(tmp_path, [
        ("abc", "math", "math", "", ""),
        ("Brüche kürzen", "math", "math", "", ""),
        ("Gleichungen lösen", "math", "", "", ""),
    ])

    data = load_dataset(path, ["title"], "labels", synthetic_rows="exclude")

    assert data.excluded_generated == 1

