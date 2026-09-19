"""Reading data-prep's provenance marks: which rows an LLM wrote or touched."""

import numpy as np
import pandas as pd
import pytest

from app import provenance
from app.provenance import ENRICHED, EXAMPLE, GENERATED, SAME_TEXT, RowProvenance, block_marks


def test_blank_whitespace_and_missing_cells_are_no_marks():
    """A CSV read as text hands over NaN for an empty cell, and an export may write
    blanks: none of them says anything about the row."""
    frame = pd.DataFrame({"generated_for": ["", "   ", np.nan, None, "disc/A"]})

    assert block_marks(frame, mode="train").tolist() == [0, 0, 0, 0, GENERATED]


def test_a_frame_without_mark_columns_has_no_marks():
    frame = pd.DataFrame({"title": ["a", "b"]})

    marks = block_marks(frame, mode="train")

    assert marks.dtype == np.int8
    assert marks.tolist() == [0, 0]


def test_each_mark_sets_its_own_bit():
    frame = pd.DataFrame({
        "generated_for": ["disc/A", "", "", "disc/B"],
        "example_for": ["", "disc/A", "", ""],
        "enriched_fields": ["", "", "keywords", "keywords"],
    })

    assert block_marks(frame, mode="train").tolist() == [
        GENERATED, EXAMPLE, ENRICHED, GENERATED | ENRICHED]


def test_an_example_is_an_ordinary_row_once_the_generated_rows_are_gone():
    """`example_for` says "paraphrases of this row are in the data". Without the
    generated rows nothing derived from it is left, so it may validate again. The
    generated bit stays: dropping those rows is the loader's decision."""
    frame = pd.DataFrame({
        "generated_for": ["disc/A", ""],
        "example_for": ["", "disc/A"],
        "enriched_fields": ["", "keywords"],
    })

    assert block_marks(frame, mode="exclude").tolist() == [GENERATED, ENRICHED]


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="synthetic_rows"):
        block_marks(pd.DataFrame({"generated_for": ["x"]}), mode="keep")


def test_the_summary_counts_each_kind_and_names_the_labels_no_real_row_validates():
    marks = np.array([0, GENERATED, GENERATED, EXAMPLE, ENRICHED, SAME_TEXT, 0], dtype=np.int8)
    rows = RowProvenance(marks=marks, mode="train", excluded_generated=0,
                         scored=np.array([True, False, True]))

    summary = rows.summary(["a", "b", "c"], scored_rows=2)

    assert rows.train_only().tolist() == [False, True, True, True, True, True, False]
    assert summary == {
        "mode": "train",
        "generated_rows": 2,
        "example_rows": 1,
        "enriched_rows": 1,
        "excluded_generated_rows": 0,
        "train_only_rows": 5,
        "validated_on": "real_rows",
        "scored_rows": 2,
        "labels_not_validated": ["b"],
        "fallback": None,
    }


def test_a_fallback_says_the_metrics_include_ai_rows():
    rows = RowProvenance(marks=np.array([GENERATED], dtype=np.int8), mode="train",
                         excluded_generated=0, fallback="no real rows to validate on")

    summary = rows.summary(["a"], scored_rows=1)

    assert summary["validated_on"] == "all_rows"
    assert summary["fallback"] == "no real rows to validate on"
    assert summary["labels_not_validated"] == []


def test_the_contract_names_match_what_data_prep_writes():
    """A name that drifts between writer and reader is a leak nobody sees: the split
    and the validation would simply stop recognising the rows."""
    assert provenance.MARK_COLUMNS == ("generated_for", "example_for", "enriched_fields")
