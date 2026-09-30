"""Training data preparation: load, clean, build targets, split.

First stage of the training pipeline (see ``training.py`` for the orchestration):
turns a training request into cleaned texts, a binary label matrix, the detected
task type and train/val/test indices. Kept separate from the fitting stages so
each module has one reason to change.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as data_mod
from .csv_encoding import CsvEncoding
from .dataset_load import load_dataset
from .errors import TrainingInputError
from .profiles import TrainingConfig
from .provenance import RowProvenance
from .real_rows import row_provenance, split_rows
from .settings import Settings

logger = logging.getLogger(__name__)

# Optional sidecar in the data directory: {"<label uri>": "<display name>"}. Authoritative
# where present, because a CSV that separates both URIs and display names with the same
# character cannot be fully repaired from itself (see label_names.pair_names).
# Generate with `python scripts/fetch_vocab_labels.py`.
LABEL_NAMES_FILE = "label_names.json"


def _authoritative_label_names(settings: Settings) -> dict[str, str]:
    """Load the label-name sidecar; ``{}`` when absent or unusable.

    Never fatal: correct display names are a reporting nicety, while a training run is
    expensive. A malformed file is logged and ignored rather than failing the run.
    """
    path = Path(settings.data_dir) / LABEL_NAMES_FILE
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring %s: %r", LABEL_NAMES_FILE, exc)
        return {}
    if not isinstance(loaded, dict):
        logger.warning("Ignoring %s: expected a JSON object of uri -> name", LABEL_NAMES_FILE)
        return {}
    return {
        uri: name.strip()
        for uri, name in loaded.items()
        if isinstance(uri, str) and isinstance(name, str) and name.strip()
    }


def _drop_unlearnable(
    texts: np.ndarray,
    y_all: np.ndarray,
    classes: list[str],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    marks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[np.ndarray, np.ndarray, np.ndarray],
           np.ndarray | None]:
    """Drop label columns the TRAIN split cannot teach -- no positive there, or no
    negative -- then the rows those drops orphaned (all-zero targets), remapping the split
    indices. ``marks`` (provenance, one per row) follows the rows.

    A head needs both classes in the rows it is fitted on: a label present only in
    val/test cannot be trained, and one on every train row is fitted as a constant (the
    type the skops guard refuses, audit 2026-09-30 T01 -- the deploy fit runs on train
    and val, so both classes in train is what keeps it loadable). Rows whose ONLY labels
    were dropped mirror ``prepare_targets``'s row_keep contract: kept, they would score
    guaranteed misses in the metrics (fatal under the argmax rule for single-label tasks)
    and dilute ``avg_labels``. Repeated until stable, for the reason ``prepare_targets``
    gives: an orphaned TRAIN row was a negative of every label that stays.
    """
    train_idx, val_idx, test_idx = splits
    while True:
        positives = y_all[train_idx].sum(axis=0)
        learnable = (positives > 0) & (positives < len(train_idx))
        if bool(learnable.all()):
            return texts, y_all, classes, (train_idx, val_idx, test_idx), marks
        y_all = y_all[:, learnable]
        classes = [c for c, keep in zip(classes, learnable, strict=True) if keep]
        keep = y_all.sum(axis=1) > 0
        if bool(keep.all()):
            continue
        new_pos = np.cumsum(keep) - 1

        def remap(idx: np.ndarray, keep: np.ndarray = keep, new_pos: np.ndarray = new_pos) -> np.ndarray:
            return new_pos[idx[keep[idx]]]

        texts, y_all = texts[keep], y_all[keep]
        train_idx, val_idx, test_idx = remap(train_idx), remap(val_idx), remap(test_idx)
        marks = None if marks is None else marks[keep]


@dataclass
class Prepared:
    """Cleaned data and label targets, ready for feature extraction and fitting."""

    texts: np.ndarray
    y_all: np.ndarray
    classes: list[str]
    task_type: str
    avg_labels: float
    min_samples: int
    # The weights actually applied (request override or the narrowed config default),
    # so the persisted metadata describes the model rather than the request.
    text_column_weights: dict[str, int]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    uri_to_label: dict[str, str]
    # Which rows an LLM wrote or touched and what that decided (``provenance``); None for
    # a dataset without marks.
    provenance: RowProvenance | None = None
    # Labels with enough rows that were not trained, because too few rows lack them
    # (``prepare_targets``) -- recorded, so the bundle says what it left out and why.
    ubiquitous_labels: tuple[str, ...] = ()
    # How the dataset file was decoded (``csv_encoding``); the bundle records it.
    csv_encoding: CsvEncoding = CsvEncoding("utf-8")
    # Copies of a text dropped although their labels differed (``dataset_load``, T10).
    conflicting_duplicates: int = 0


def prepare_data(
    req: dict,
    settings: Settings,
    training_cfg: TrainingConfig,
    *,
    cv_folds: int,
    on_progress: Callable[..., None],
    should_stop: Callable[[], bool],
    stratified: bool = False,
) -> Prepared | None:
    """Load + clean the dataset, build label targets, detect the task type and
    split into train/val/test. Returns ``None`` if cancelled."""
    dataset_path = Path(settings.data_dir) / req["dataset_name"]

    # Request wins over the config default (mirrors min_samples_per_label / cv_folds).
    # `None` = "use the config default", `{}` = "explicitly no weighting". The config
    # default is a dataset CONVENTION, so it is narrowed to the columns this request
    # trains on — a global default must not break a CSV with different column names.
    # Request-level keys are validated at the schema instead, where an unknown key
    # is a typo the caller wants to hear about.
    req_weights = req.get("text_column_weights")
    text_column_weights = (
        {col: w for col, w in training_cfg.text_column_weights.items() if col in req["text_columns"]}
        if req_weights is None
        else dict(req_weights)
    )

    # --- Load + clean (read/clean/filter sub-steps shown via phase_detail) ---
    on_progress(phase="loading", progress=5, message="Loading and preparing dataset...")
    mode = req.get("synthetic_rows", "train")
    loaded = load_dataset(
        dataset_path,
        req["text_columns"],
        req["label_column"],
        separator=req["csv_separator"],
        label_separator=req["label_separator"],
        min_text_length=training_cfg.min_text_length,
        drop_duplicates=training_cfg.drop_duplicates,
        label_filter=req.get("label_filter"),
        text_column_weights=text_column_weights,
        label_names=_authoritative_label_names(settings),
        synthetic_rows=mode,
        on_progress=lambda msg: on_progress(phase_detail=msg),
    )
    if should_stop():
        return None
    n_texts = len(loaded.texts)
    if n_texts < 10:
        left_out = (f"; {loaded.excluded_generated} generated rows were left out "
                    "(synthetic_rows=exclude)" if loaded.excluded_generated else "")
        raise TrainingInputError(
            f"Too few usable rows after cleaning ({n_texts}){left_out}. Need at least 10.")

    # --- Labels + task type ---
    on_progress(phase="preparing", progress=20, message=f"{n_texts} texts. Preparing labels...")
    req_min = req.get("min_samples_per_label")
    override_min = req_min if req_min is not None else training_cfg.min_samples_per_label
    min_samples = data_mod.auto_min_samples(n_texts, override_min)
    y_all, classes, row_keep = data_mod.prepare_targets(loaded.label_lists, min_samples)
    counts = Counter(label for labels in loaded.label_lists for label in set(labels))
    # A label with enough rows that did not survive failed the other half of the rule: too
    # few rows WITHOUT it (prepare_targets). Positive counts never shrink there -- only rows
    # left without any label are dropped -- so this is exactly the set it removed for that.
    kept = set(classes)
    ubiquitous = sorted(label for label, n in counts.items() if n >= min_samples and label not in kept)
    if ubiquitous:
        logger.warning("Not trained, on (nearly) every row: %s", ", ".join(ubiquitous))
    if loaded.conflicting_duplicates:
        logger.warning("%d duplicate rows dropped whose labels differed from the kept copy's",
                       loaded.conflicting_duplicates)
    if not classes and ubiquitous:
        raise TrainingInputError(
            f"Nothing to learn: every label with enough rows is on every row, or missing from "
            f"fewer than min_samples_per_label={min_samples} of them ({', '.join(ubiquitous[:5])}"
            f"{', ...' if len(ubiquitous) > 5 else ''}). A classifier learns a label from the "
            "rows without it as much as from the rows with it."
        )
    if not classes:
        # Name the gap, not just the rule: "no label has >= 20" leaves the user
        # guessing whether they are one row short or need a different dataset.
        best = max(counts.values(), default=0)
        raise TrainingInputError(
            f"Not enough data to train: of {len(counts)} labels the most frequent one has "
            f"only {best} tagged rows, below min_samples_per_label={min_samples}. Lower "
            f"min_samples_per_label, or add rows for the labels you want to learn."
        )
    texts = np.array([t for t, keep in zip(loaded.texts, row_keep, strict=False) if keep], dtype=object)
    marks = None if loaded.marks is None else np.asarray(loaded.marks, dtype=np.int8)[row_keep]
    # Re-check AFTER dropping rare-label rows: the pre-filter guard above counts
    # rows that prepare_targets may have just removed, so the three-way split
    # could otherwise get an empty val/test and "succeed" with meaningless metrics.
    n_kept = len(texts)
    if n_kept < 10:
        raise TrainingInputError(
            f"Too few rows left after dropping the labels with fewer than {min_samples} rows "
            f"with them or without them ({n_kept} of {n_texts}). Lower min_samples_per_label "
            "or add more data."
        )
    # --- Train / val / test split ---
    train_only = marks != 0 if marks is not None and marks.any() else None
    (train_idx, val_idx, test_idx), fallback = split_rows(
        len(texts), training_cfg=training_cfg, seed=settings.random_seed,
        y=y_all if stratified else None, train_only=train_only, cv_folds=cv_folds,
    )
    if cv_folds < 2:
        # Classic split only: drop unlearnable label columns + the rows that
        # orphans (see _drop_unlearnable). CV trains on EVERY row and
        # prepare_targets already guarantees >= min_samples positives per kept
        # label, so dropping there would lose rare labels to a split that CV
        # does not even use.
        dropped = _drop_unlearnable(texts, y_all, classes, (train_idx, val_idx, test_idx), marks)
        if train_only is not None and fallback is None and not (
                len(dropped[3][1]) and len(dropped[3][2])):
            # The real rows' labels had no training positive, so the drop emptied val or
            # test: the same shortage as too few real rows, and the same fallback.
            (train_idx, val_idx, test_idx), _ = split_rows(
                len(texts), training_cfg=training_cfg, seed=settings.random_seed,
                y=y_all if stratified else None, train_only=None, cv_folds=cv_folds)
            fallback = (f"the {int(np.count_nonzero(~train_only))} real rows carry no label "
                        "left to validate on")
            dropped = _drop_unlearnable(texts, y_all, classes, (train_idx, val_idx, test_idx), marks)
        texts, y_all, classes, (train_idx, val_idx, test_idx), marks = dropped
        # Train rows can never be orphaned (their labels have train positives by
        # definition), but a val/test split could in theory lose all its rows.
        if min(len(val_idx), len(test_idx)) == 0:
            raise TrainingInputError(
                "The validation or test split lost all its labeled rows after "
                "dropping unlearnable labels; add more data or lower min_samples_per_label."
            )
    if fallback:
        logger.warning("AI-marked rows validate after all: %s", fallback)
    if len(classes) < 2:
        raise TrainingInputError("Fewer than 2 learnable labels; dataset too small/sparse.")
    forced = req.get("task_type")
    # After every drop: the type of the targets that are actually trained (T04).
    task_type = forced if forced in ("multilabel", "multiclass", "binary") else (
        data_mod.detect_task_type(y_all))
    if should_stop():
        return None
    # AFTER the drops, so the average describes the label space actually trained.
    avg_labels = float(y_all.sum(axis=1).mean())

    return Prepared(
        texts=texts, y_all=y_all, classes=classes, task_type=task_type,
        avg_labels=avg_labels, min_samples=min_samples,
        text_column_weights=text_column_weights,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        uri_to_label={k: v for k, v in loaded.uri_to_label.items() if k in set(classes)},
        provenance=row_provenance(marks, y_all, mode=mode, excluded=loaded.excluded_generated,
                                  dropped=loaded.excluded_rows,
                                  fallback=fallback,
                                  evaluated=test_idx if cv_folds < 2 else None,
                                  min_samples=min_samples,
                                  thin_mode=req.get("thin_label_threshold", "own")),
        ubiquitous_labels=tuple(ubiquitous),
        csv_encoding=loaded.encoding,
        conflicting_duplicates=loaded.conflicting_duplicates,
    )
