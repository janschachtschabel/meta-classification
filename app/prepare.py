"""Training data preparation: load, clean, build targets, split.

First stage of the training pipeline (see ``training.py`` for the orchestration):
turns a training request into cleaned texts, a binary label matrix, the detected
task type and train/val/test indices. Kept separate from the fitting stages so
each module has one reason to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as data_mod
from .errors import TrainingInputError
from .profiles import TrainingConfig
from .settings import Settings


@dataclass
class Prepared:
    """Cleaned data and label targets, ready for feature extraction and fitting."""

    texts: np.ndarray
    y_all: np.ndarray
    classes: list[str]
    task_type: str
    avg_labels: float
    min_samples: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    uri_to_label: dict[str, str]


def prepare_data(
    req: dict,
    settings: Settings,
    training_cfg: TrainingConfig,
    *,
    cv_folds: int,
    on_progress: Callable[..., None],
    should_stop: Callable[[], bool],
) -> Prepared | None:
    """Load + clean the dataset, build label targets, detect the task type and
    split into train/val/test. Returns ``None`` if cancelled."""
    dataset_path = Path(settings.data_dir) / req["dataset_name"]

    # --- Load + clean (read/clean/filter sub-steps shown via phase_detail) ---
    on_progress(phase="loading", progress=5, message="Loading and preparing dataset...")
    loaded = data_mod.load_dataset(
        dataset_path,
        req["text_columns"],
        req["label_column"],
        separator=req["csv_separator"],
        label_separator=req["label_separator"],
        min_text_length=training_cfg.min_text_length,
        drop_duplicates=training_cfg.drop_duplicates,
        label_filter=req.get("label_filter"),
        on_progress=lambda msg: on_progress(phase_detail=msg),
    )
    if should_stop():
        return None
    n_texts = len(loaded.texts)
    if n_texts < 10:
        raise TrainingInputError(f"Too few usable rows after cleaning ({n_texts}). Need at least 10.")

    # --- Labels + task type ---
    on_progress(phase="preparing", progress=20, message=f"{n_texts} texts. Preparing labels...")
    req_min = req.get("min_samples_per_label")
    override_min = req_min if req_min is not None else training_cfg.min_samples_per_label
    min_samples = data_mod.auto_min_samples(n_texts, override_min)
    y_all, classes, row_keep = data_mod.prepare_targets(loaded.label_lists, min_samples)
    if not classes:
        raise TrainingInputError(f"No label has >= {min_samples} samples; nothing to train.")
    texts = np.array([t for t, keep in zip(loaded.texts, row_keep, strict=False) if keep], dtype=object)
    kept_label_lists = [labs for labs, keep in zip(loaded.label_lists, row_keep, strict=False) if keep]
    # Re-check AFTER dropping rare-label rows: the pre-filter guard above counts
    # rows that prepare_targets may have just removed, so the three-way split
    # could otherwise get an empty val/test and "succeed" with meaningless metrics.
    n_kept = len(texts)
    if n_kept < 10:
        raise TrainingInputError(
            f"Too few rows left after dropping labels below {min_samples} samples "
            f"({n_kept} of {n_texts}). Lower min_samples_per_label or add more data."
        )
    forced = req.get("task_type")
    if forced in ("multilabel", "multiclass", "binary"):
        task_type = forced
    else:
        task_type = data_mod.detect_task_type(kept_label_lists, len(classes))
    avg_labels = float(y_all.sum(axis=1).mean())

    # --- Train / val / test split ---
    train_idx, val_idx, test_idx = data_mod.three_way_split(
        len(texts), val_size=training_cfg.validation_size,
        test_size=training_cfg.test_size, seed=settings.random_seed,
    )
    if cv_folds < 2:
        # Classic split only: drop label columns absent from the training split (a
        # head needs positives). CV trains on EVERY row and prepare_targets already
        # guarantees >= min_samples positives per kept label, so dropping there
        # would lose rare labels to a split that CV does not even use.
        learnable = y_all[train_idx].sum(axis=0) > 0
        if not learnable.all():
            y_all = y_all[:, learnable]
            classes = [c for c, keep in zip(classes, learnable, strict=False) if keep]
    if len(classes) < 2:
        raise TrainingInputError("Fewer than 2 learnable labels; dataset too small/sparse.")
    if should_stop():
        return None

    return Prepared(
        texts=texts, y_all=y_all, classes=classes, task_type=task_type,
        avg_labels=avg_labels, min_samples=min_samples,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        uri_to_label={k: v for k, v in loaded.uri_to_label.items() if k in set(classes)},
    )
