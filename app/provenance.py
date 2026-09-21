"""Which rows an LLM wrote or touched, as data-prep marks them.

data-prep (``app/refine/provenance.py`` there) writes three columns, and this module is
their reader. The names are a contract between the two apps: one that drifts is a leak
nobody sees — the rows would simply stop being recognised, and a model would be scored
on text the same LLM wrote from the same examples it was trained on.

  - ``generated_for``: the row was written by an LLM (balancing, or a Runs export).
  - ``example_for``: a REAL row shown to the generator as an example — its paraphrases
    are in the data, so the row itself would be recognised rather than classified.
  - ``enriched_fields``: a REAL row whose empty or short fields an LLM filled.
    data-prep enriches exact twins alike -- rows equal in every field its enrichment
    reads -- so, as long as those fields are among the text columns trained on, a twin
    never keeps the untouched text beside its enriched copy: the text rule
    (``SAME_TEXT``) knows rows by their text.

Such rows train, but never validate. A leaf: imports nothing of ours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENERATED_FOR = "generated_for"
EXAMPLE_FOR = "example_for"
ENRICHED_FIELDS = "enriched_fields"
MARK_COLUMNS = (GENERATED_FOR, EXAMPLE_FOR, ENRICHED_FIELDS)

# One int8 per row carries every mark through each row drop of the pipeline, where a
# separate array per kind would have to be filtered alongside the texts four times.
GENERATED, EXAMPLE, ENRICHED = 1, 2, 4
# Not a mark of the row itself: it shares its cleaned text with a marked row. The same
# text on both sides of a split would carry the LLM's text into the validation.
SAME_TEXT = 8

# What a training run does with the generated rows: train on them (never validating on
# them), or leave them out when the dataset is read.
SYNTHETIC_MODES = ("train", "exclude")
# Where a thin label -- one that reached the training minimum only through AI-marked
# rows -- is cut: at its own threshold, tuned on its few real rows, or the global one.
THIN_MODES = ("own", "global")

_BITS = ((GENERATED_FOR, GENERATED), (EXAMPLE_FOR, EXAMPLE), (ENRICHED_FIELDS, ENRICHED))

# Below this many rows under the metrics, a run says so. The fallback below only fires on
# an EMPTY validation or test part, so one real row in each was accepted as a holdout
# split -- an F1 over a single row, thresholds tuned on another single row, and nothing
# anywhere saying the number is a coin flip. The bound is a judgement, and this is the
# reasoning: one row out of n moves a label's F1 by at least 1/n, so under ten rows a
# single one is worth more than 0.1 -- coarser than the differences this project compares
# runs at (README: +0.0145 for a field weighting, -0.0052 for halved feature caps).
# Flagged rather than refused: too few real rows is still better than measuring on text
# the model was trained on, which is what the fallback has to do when there are none.
_MIN_SCORED_ROWS = 10


def _marked(column: pd.Series) -> np.ndarray:
    # Blank, whitespace and a missing cell are no mark: an export may write spaces, and a
    # frame built in memory may hold NaN. The loader reads mark cells as written, so a
    # text such as "NA" or "NaN" names a label there, and is a mark.
    return (column.fillna("").astype(str).str.strip() != "").to_numpy()


def block_marks(frame: pd.DataFrame, *, mode: str) -> np.ndarray:
    """The marks of each row of ``frame`` as an int8 bitmask (``GENERATED`` ...).

    Under ``mode="exclude"`` an ``example_for`` mark is ignored: with the generated
    rows gone, nothing derived from the example is left in the data. The generated bit
    is still set — dropping those rows is the loader's decision, not this one's.
    """
    if mode not in SYNTHETIC_MODES:
        raise ValueError(f"synthetic_rows must be one of {SYNTHETIC_MODES}, got {mode!r}")
    marks = np.zeros(len(frame), dtype=np.int8)
    for column, bit in _BITS:
        if column not in frame.columns or (bit == EXAMPLE and mode == "exclude"):
            continue
        marks[_marked(frame[column])] |= bit
    return marks


@dataclass
class RowProvenance:
    """The marks of the rows a training run kept, and what they decided.

    ``validate`` holds the rows the metrics may be computed on and ``scored`` the
    classes a real row can validate; ``None`` means all of them. ``fallback`` says why
    the metrics include AI-marked rows after all — too few real rows to validate on —
    and is ``None`` whenever they do not; it answers WHICH rows were measured, while
    ``summary``'s ``too_few_rows`` answers HOW MANY, and a small dataset can need both.
    ``thin`` (per class) marks the labels with
    fewer real rows than the training minimum, and ``thin_mode`` is the run's choice of
    their cut.
    """

    marks: np.ndarray
    mode: str
    excluded_generated: int = 0
    validate: np.ndarray | None = None
    scored: np.ndarray | None = None
    fallback: str | None = None
    thin: np.ndarray | None = None
    thin_mode: str = "own"

    def train_only(self) -> np.ndarray:
        """Per row: used for training, never for validation."""
        return self.marks != 0

    def keep_global(self) -> np.ndarray | None:
        """The classes the threshold search gives the global cut: the thin ones, when
        the run chose that; ``None`` when every label keeps its own."""
        if self.thin_mode != "global" or self.thin is None or not self.thin.any():
            return None
        return self.thin

    def summary(self, classes: list[str], *, scored_rows: int) -> dict:
        """The ``synthetic_data`` block a bundle stores: what was trained on, and what
        the reported numbers were computed on.

        ``scored_rows`` is the count those numbers rest on — the test split for a holdout
        run, the real rows for k-fold — and ``too_few_rows`` is the reason string set when
        it is below :data:`_MIN_SCORED_ROWS`, ``None`` otherwise.
        """

        def count(bit: int) -> int:
            return int(np.count_nonzero(self.marks & bit))

        unvalidated = ([] if self.scored is None
                       else [c for c, ok in zip(classes, self.scored, strict=True) if not ok])
        thin = ([] if self.thin is None
                else [c for c, is_thin in zip(classes, self.thin, strict=True) if is_thin])
        return {
            "mode": self.mode,
            "generated_rows": count(GENERATED),
            "example_rows": count(EXAMPLE),
            "enriched_rows": count(ENRICHED),
            "excluded_generated_rows": self.excluded_generated,
            "train_only_rows": int(np.count_nonzero(self.train_only())),
            "validated_on": "all_rows" if self.fallback else "real_rows",
            "scored_rows": scored_rows,
            "labels_not_validated": unvalidated,
            "fallback": self.fallback,
            "too_few_rows": (
                None if scored_rows >= _MIN_SCORED_ROWS else
                f"the metrics rest on {scored_rows} row{'' if scored_rows == 1 else 's'}, "
                f"fewer than the {_MIN_SCORED_ROWS} a number comparable with another run "
                f"needs"
            ),
            "thin_labels": thin,
            "thin_label_threshold": self.thin_mode,
        }
