"""Score an existing model against a dataset it was not necessarily trained on.

"Model B beats model A" is only a statement if both were measured on the same rows.
Until now that meant a script driving a running server (``scripts/eval_holdout.py``),
which nothing recorded and nobody could repeat.

The arithmetic is not the hard part — ``metrics.compute_metrics`` already does it with
the decision rule serving applies. What makes the number honest is the label space: a
model can only be scored over ITS OWN classes, and a dataset carrying labels the model
never learned is a dataset it cannot fully answer. Those rows and labels are reported
rather than dropped, because an evaluation that hides them reads better than the model
deserves.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from .bundle_meta import as_mapping
from .dataset_load import load_dataset
from .errors import TrainingInputError
from .memory import MiB, held_bytes, memory_limit_bytes
from .metrics import compute_metrics
from .predict_csv import CHUNK_ROWS
from .registry import Registry
from .settings import Settings

# CHUNK_ROWS is borrowed rather than redefined: how many rows the model may vectorize at
# once is the same decision here as in `predict_csv`, which does the identical "score a
# whole CSV" work, and two copies of it would drift apart without anything noticing.

# What a row's truth is scored against. Reported in full when short, because the point
# is to see WHICH vocabulary the dataset speaks; a long list means the wrong dataset.
_MAX_REPORTED_UNKNOWN = 25


def _evaluation_bytes(rows: int, labels: int) -> int:
    """What an evaluation holds at once beyond the model: the truth matrix (int8), the
    probabilities (float64, as ``predict_proba`` returns them) and the decision matrix
    ``compute_metrics`` derives from them (int8)."""
    return rows * labels * (1 + 8 + 1)


def _refuse_if_the_evaluation_cannot_fit(rows: int, labels: int) -> None:
    """Stop an evaluation whose arrays cannot fit the container, before it allocates one.

    This job runs in the API process — ``run_evaluation`` is submitted without
    ``run_in_child``, unlike a training run, which takes the child precisely so an OOM
    kill ends the run and not the server. Here the kill takes the server with it, and
    every model it was serving. 300,000 rows against a 300-label model is ~858 MB of
    truth, probabilities and decisions before a single chunk has been vectorized.

    Only the container's LIMIT is weighed, not the training budget: that budget is what
    an operator granted a TRAINING run, and spending it on an evaluation would refuse
    evaluations the machine can easily hold. Outside a container there is no limit and
    nothing is refused. Counted against what is already held, which is the number the
    OOM killer compares too — and that reading is optimistic where it cannot be taken
    (``memory.held_bytes`` answers 0 rather than "unknown"), so this gate is a floor.
    """
    limit = memory_limit_bytes()
    if limit is None:
        return
    needed = _evaluation_bytes(rows, labels)
    held = held_bytes()
    if held + needed <= limit:
        return
    raise TrainingInputError(
        f"This evaluation cannot fit the memory the container has: {rows:,} rows x "
        f"{labels:,} labels need {needed // MiB:,} MB of truth, probabilities and "
        f"decisions on top of the {held // MiB:,} MB already held — more than the "
        f"container's limit of {limit // MiB:,} MB, which the kernel enforces by killing "
        f"the process. Evaluate on fewer rows, or give the container more memory."
    )


def evaluate_model(model, texts: list[str], label_lists: list[list[str]], *,
                   chunk_rows: int = CHUNK_ROWS) -> dict:
    """Score ``model`` on labelled rows, in the model's own label space.

    Truth is binarized over ``model.classes``, never over the dataset's own label set:
    the two are different widths, and lining up column 0 of one with column 0 of the
    other is how an evaluation ends up measuring nothing.

    Rows whose labels the model has never seen are excluded and counted. Scoring them
    as failures would blame the model for a label it was never given; dropping them
    silently would flatter it on a dataset it does not cover. ``metrics`` is ``None``
    when nothing comparable is left — an F1 of 0.0 would read as "terrible here" rather
    than "these two do not meet".

    ``labels_covered`` says how much of the label space the data exercises. The macro
    averages cover the classes the data carries a positive for — a class with none scores
    F1 0.0 under every model, so counting it would measure the probe — and
    ``metrics.labels_not_scored`` names the rest.

    Scored ``chunk_rows`` at a time: ``predict_proba`` vectorizes everything it is handed
    in ONE feature matrix, so an unchunked call builds the whole dataset's sparse matrix
    inside the serving process. The arrays that outlive a chunk are weighed first
    (:func:`_refuse_if_the_evaluation_cannot_fit`).
    """
    _refuse_if_the_evaluation_cannot_fit(len(texts), len(model.classes))
    column_of = {uri: index for index, uri in enumerate(model.classes)}
    unknown: set[str] = set()
    keep: list[int] = []
    truth = np.zeros((len(texts), len(model.classes)), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        known = [column_of[uri] for uri in labels if uri in column_of]
        unknown.update(uri for uri in labels if uri not in column_of)
        if not known:
            continue
        truth[row, known] = 1
        keep.append(row)

    result = {
        "n_rows": len(keep),
        "rows_without_a_known_label": len(texts) - len(keep),
        "unknown_labels": sorted(unknown)[:_MAX_REPORTED_UNKNOWN],
        "n_unknown_labels": len(unknown),
        # How many of the model's own labels this dataset actually exercises. Measured on
        # a real 59-label model against an 8-row probe: f1_macro 0.068 beside f1_micro
        # 0.941, because macro then averaged over ALL classes and 55 of them had no
        # examples. `compute_metrics` no longer counts those (CORR-2); this says how
        # narrow a question the remaining number answers.
        "labels_covered": int((truth.sum(axis=0) > 0).sum()),
        "metrics": None,
    }
    if not keep:
        return result

    # Written into one array rather than collected and concatenated: the pieces plus the
    # joined copy would be two of the largest thing here at once.
    proba = np.empty((len(keep), len(model.classes)), dtype=np.float64)
    for start in range(0, len(keep), chunk_rows):
        block = keep[start:start + chunk_rows]
        proba[start:start + len(block)] = model.predict_proba([texts[row] for row in block])
    result["metrics"] = compute_metrics(
        truth[keep], proba, model.classes,
        model.global_threshold, model.per_label_thresholds, model.task_type,
    )
    return result


def _weights_for(registry: Registry, name: str, req: dict) -> dict:
    """How often each text column is repeated in the evaluation text.

    A model was fit on text assembled a particular way, so a request that says nothing
    is scored the way the model was trained: the bundle's own weights, narrowed to the
    columns this run reads (as a training request narrows them). A caller who really
    wants every column once sends an explicit empty mapping.
    """
    asked = req.get("text_column_weights")
    if asked is not None:
        return dict(asked)
    recorded = as_mapping(as_mapping(registry.info(name).get("metadata"))
                          .get("text_column_weights"))
    return {col: weight for col, weight in recorded.items() if col in req["text_columns"]}


def run_evaluation(
    req: dict,
    settings: Settings,
    registry: Registry,
    *,
    on_progress: Callable[..., None],
    should_stop: Callable[[], bool],
) -> dict:
    """Load a dataset, score one model on it, and record the result in its bundle.

    Same shape as ``training.run_training`` so the one job runner can host either: both
    are a long CPU-bound pass over a dataset, and on a single-instance deployment
    exactly one of them may hold the CPU.

    The result is APPENDED to the bundle's ``evaluations`` list. It never touches the
    training metrics — those describe the run that produced the model and stay the
    bundle's own account of itself.
    """
    started = time.time()
    name = req["model_name"]
    weights = _weights_for(registry, name, req)
    on_progress(phase="loading", progress=5, message=f"Reading {req['dataset_name']} …")
    data = load_dataset(
        settings.data_dir / req["dataset_name"],
        req["text_columns"],
        req["label_column"],
        separator=req.get("csv_separator", ";"),
        label_separator=req.get("label_separator", ","),
        label_filter=req.get("label_filter"),
        text_column_weights=weights,
    )
    # A row an LLM wrote or touched is never scored: measured on it, a model is measured
    # on how well it learned that LLM (data-prep's marks, see ``provenance``).
    real = [i for i, mark in enumerate(data.marks or [0] * len(data.texts)) if not mark]
    skipped = len(data.texts) - len(real)
    texts = [data.texts[i] for i in real]
    label_lists = [data.label_lists[i] for i in real]
    if not texts:
        detail = f": all {skipped} are AI-marked" if skipped else ""
        raise TrainingInputError(
            f"No usable rows in '{req['dataset_name']}' for those columns{detail}."
        )
    if should_stop():
        return {}

    on_progress(phase="evaluating", progress=40,
                message=f"Scoring {name} on {len(texts)} rows …")
    model = registry.get(name)
    result = evaluate_model(model, texts, label_lists)
    if should_stop():
        return {}

    record = {
        "dataset": req["dataset_name"],
        "label_column": req["label_column"],
        "text_columns": list(req["text_columns"]),
        # What the scored text was built from: a number is only comparable against
        # another run that assembled its text the same way.
        "text_column_weights": weights,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.time() - started, 1),
        **result,
        **({"ai_marked_rows_skipped": skipped} if skipped else {}),
    }
    on_progress(phase="saving", progress=90, message="Recording the result …")
    registry.append_evaluation(name, record)
    # Reported back in the same shape a training returns, so the job history reads the
    # same for both kinds — the headline scores under the same keys.
    metrics = result["metrics"] or {}
    return {
        "model_name": name,
        "task_type": model.task_type,
        "n_labels": metrics.get("n_labels", 0),
        "metrics": metrics,
        "n_rows": result["n_rows"],
        "evaluation_time_seconds": record["duration_seconds"],
    }
