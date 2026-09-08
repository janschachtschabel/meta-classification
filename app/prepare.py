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
from .errors import TrainingInputError
from .profiles import TrainingConfig
from .settings import Settings

logger = logging.getLogger(__name__)

# Optional sidecar in the data directory: {"<label uri>": "<display name>"}. Authoritative
# where present, because a CSV that separates both URIs and display names with the same
# character cannot be fully repaired from itself (see data._pair_names).
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
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Drop label columns without positives in the TRAIN split, then the rows
    those drops orphaned (all-zero targets), remapping the split indices.

    A head needs positives to learn, so a label present only in val/test cannot
    be trained. Rows whose ONLY labels were dropped mirror ``prepare_targets``'s
    row_keep contract: kept, they would score guaranteed misses in the metrics
    (fatal under the argmax rule for single-label tasks) and dilute
    ``avg_labels``.
    """
    train_idx, val_idx, test_idx = splits
    learnable = y_all[train_idx].sum(axis=0) > 0
    if bool(learnable.all()):
        return texts, y_all, classes, splits
    y_all = y_all[:, learnable]
    classes = [c for c, keep in zip(classes, learnable, strict=False) if keep]
    keep = y_all.sum(axis=1) > 0
    if bool(keep.all()):
        return texts, y_all, classes, splits
    new_pos = np.cumsum(keep) - 1

    def remap(idx: np.ndarray) -> np.ndarray:
        return new_pos[idx[keep[idx]]]

    return texts[keep], y_all[keep], classes, (remap(train_idx), remap(val_idx), remap(test_idx))


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
    loaded = data_mod.load_dataset(
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
        # Name the gap, not just the rule: "no label has >= 20" leaves the user
        # guessing whether they are one row short or need a different dataset.
        counts = Counter(label for labels in loaded.label_lists for label in labels)
        best = max(counts.values(), default=0)
        raise TrainingInputError(
            f"Not enough data to train: of {len(counts)} labels the most frequent one has "
            f"only {best} tagged rows, below min_samples_per_label={min_samples}. Lower "
            f"min_samples_per_label, or add rows for the labels you want to learn."
        )
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

    # --- Train / val / test split ---
    train_idx, val_idx, test_idx = data_mod.three_way_split(
        len(texts), val_size=training_cfg.validation_size,
        test_size=training_cfg.test_size, seed=settings.random_seed,
    )
    if cv_folds < 2:
        # Classic split only: drop unlearnable label columns + the rows that
        # orphans (see _drop_unlearnable). CV trains on EVERY row and
        # prepare_targets already guarantees >= min_samples positives per kept
        # label, so dropping there would lose rare labels to a split that CV
        # does not even use.
        texts, y_all, classes, (train_idx, val_idx, test_idx) = _drop_unlearnable(
            texts, y_all, classes, (train_idx, val_idx, test_idx)
        )
        # Train rows can never be orphaned (their labels have train positives by
        # definition), but a val/test split could in theory lose all its rows.
        if min(len(val_idx), len(test_idx)) == 0:
            raise TrainingInputError(
                "The validation or test split lost all its labeled rows after "
                "dropping unlearnable labels; add more data or lower min_samples_per_label."
            )
    if len(classes) < 2:
        raise TrainingInputError("Fewer than 2 learnable labels; dataset too small/sparse.")
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
    )
