"""Which rows may validate a run on a dataset an LLM partly wrote.

``prepare`` loads, cleans and splits; this module answers the questions data-prep's
provenance marks raise on the way: where the holdout's validation and test rows come
from, when too few real rows force the metrics to include AI-marked ones, and which
labels a real row can score. Split out of ``prepare``, which changes for other reasons
(loading, targets) and was at the size guide.
"""

from __future__ import annotations

import numpy as np

from . import data as data_mod
from .profiles import TrainingConfig
from .provenance import RowProvenance


def split_rows(
    n: int, *, training_cfg: TrainingConfig, seed: int, y: np.ndarray | None,
    train_only: np.ndarray | None, cv_folds: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], str | None]:
    """The train/val/test split, and why AI-marked rows validate after all (``None``:
    they do not).

    k-fold folds the real rows itself (``tuning.cross_val_evaluate``) and never reads
    this split, so there it stays the one the dataset always got. The holdout draws val
    and test from the real rows. Too few of them to fill either -- a pure Runs export has
    none -- and the run falls back to every row and says so, rather than refusing the
    synthetic-only training data-prep's Runs push exists for.
    """

    def every_row() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return data_mod.three_way_split(
            n, val_size=training_cfg.validation_size, test_size=training_cfg.test_size,
            seed=seed, y=y)

    if train_only is None:
        return every_row(), None
    n_real = int(np.count_nonzero(~train_only))
    if cv_folds >= 2:
        reason = None if n_real >= cv_folds else f"only {n_real} real rows for {cv_folds} folds"
        return every_row(), reason
    try:
        splits = data_mod.three_way_split(
            n, val_size=training_cfg.validation_size, test_size=training_cfg.test_size,
            seed=seed, y=y, train_only=train_only)
    except ValueError:
        # scikit-learn refuses to split this few rows: the same shortage as below.
        splits = None
    if splits is None or not (len(splits[1]) and len(splits[2])):
        return every_row(), f"only {n_real} real rows, too few for a validation and a test part"
    return splits, None


def row_provenance(
    marks: np.ndarray | None, y_all: np.ndarray, *, mode: str, excluded: int,
    fallback: str | None, evaluated: np.ndarray | None = None,
    min_samples: int = 1, thin_mode: str = "own",
) -> RowProvenance | None:
    """What the marks decided -- or ``None`` when the dataset has none, and the run is
    the one it was before marks existed.

    A label is scored only when a real row of the rows the metrics are computed on
    carries it: ``evaluated`` -- the holdout's test split -- or, for k-fold, every real
    row. A label balancing lifted often has one real row, which the stratified split
    puts into train; scored on a test split without it, it would read F1 0.0.

    A label with fewer real rows than ``min_samples`` is thin: it reached the training
    minimum only through AI-marked rows, and ``thin_mode`` is the run's choice of its cut.
    """
    if marks is None or not (marks.any() or excluded):
        return None
    real = marks == 0
    validate = real if marks.any() and fallback is None else None
    scored = None
    if fallback is None:
        # Also once "exclude" dropped the last mark: every row is real then, and the rule
        # stays the one "train" gets, or the two runs of a dataset are not comparable.
        has_real = y_all[real if evaluated is None else evaluated].sum(axis=0) > 0
        scored = None if bool(has_real.all()) else has_real
    thin = y_all[real].sum(axis=0) < min_samples
    return RowProvenance(marks=marks, mode=mode, excluded_generated=excluded,
                         validate=validate, scored=scored, fallback=fallback,
                         thin=thin, thin_mode=thin_mode)
