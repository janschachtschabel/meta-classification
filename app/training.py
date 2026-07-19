"""Training orchestration for TF-IDF + LogisticRegression.

Pipeline in three modules, one responsibility each:
  - ``prepare.py``  : load -> clean -> label targets -> split
  - ``deploy.py``   : auto-select C + tune thresholds -> honest metrics -> deploy fit
  - here            : orchestration + persisted metadata

Two evaluation modes (``split.cv_folds`` in config, or ``cv_folds`` per request):
the classic train/val/test split, or k-fold cross-validation (every row trains
AND validates via out-of-fold, deploy on 100% of the data).

Callback-driven (``on_progress`` / ``should_stop``) so it carries no threading
logic itself; ``jobs.TrainingJob`` runs it in the background.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from .classifier import ClassifierModel
from .deploy import Fitted, fit_evaluate_deploy
from .prepare import Prepared, prepare_data
from .profiles import Profile, TrainingConfig
from .registry import Registry
from .settings import Settings

logger = logging.getLogger("api_v3.training")


def _build_metadata(
    req: dict, settings: Settings, profile: Profile, prep: Prepared, fitted: Fitted, elapsed: float,
    *, cv_folds: int = 0,
) -> dict:
    """Assemble the persisted metadata/metrics document for a trained model."""
    n = int(len(prep.texts))
    if cv_folds >= 2:
        evaluation = f"{cv_folds}-fold cross-validation (out-of-fold metrics; deployed on all {n})"
        n_train, n_val, n_test = n, 0, 0  # every row trains + validates via OOF; no fixed holdout
    else:
        evaluation = "holdout train/val/test split (metrics on the untouched test split)"
        n_train = int(len(prep.train_idx))
        n_val = int(len(prep.val_idx))
        n_test = int(len(prep.test_idx))
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation": evaluation,
        "dataset": req["dataset_name"],
        "text_columns": req["text_columns"],
        "label_column": req["label_column"],
        "label_filter": req.get("label_filter"),
        "profile": profile.name,
        "backend": "tfidf",
        "solver": settings.solver,
        "best_C": fitted.best_c,
        "task_type": prep.task_type,
        "avg_labels_per_sample": prep.avg_labels,
        "min_samples_per_label": prep.min_samples,
        "tfidf": {
            "use_char": profile.use_char,
            "n_features": fitted.deploy_n_features,
            "train_nnz": fitted.deploy_nnz,
            "train_sparse_mb": fitted.deploy_sparse_mb,
        },
        "n_samples": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_labels": len(prep.classes),
        "metrics": fitted.metrics,
        "training_time_seconds": round(elapsed, 1),
    }


def run_training(
    req: dict,
    settings: Settings,
    training_cfg: TrainingConfig,
    profile: Profile,
    registry: Registry,
    *,
    on_progress: Callable[..., None],
    should_stop: Callable[[], bool],
) -> dict:
    """Train one TF-IDF + LogReg model end-to-end and persist it.

    Orchestrates three steps — prepare data, fit/evaluate/deploy, persist — and
    returns a short result summary. Returns an empty dict if cancelled mid-run.

    ``registry`` is injected (the route passes the ``get_registry()`` singleton)
    so the training save shares ONE disk lock with all API-side reads/deletes/
    imports — a second Registry instance would have its own locks and silently
    bypass that serialization.
    """
    start = time.time()
    # Request-level cv_folds wins over the config default (mirrors min_samples_per_label).
    req_cv = req.get("cv_folds")
    cv_folds = req_cv if req_cv is not None else training_cfg.cv_folds
    prep = prepare_data(
        req, settings, training_cfg, cv_folds=cv_folds,
        on_progress=on_progress, should_stop=should_stop,
    )
    if prep is None:
        return {}
    fitted = fit_evaluate_deploy(
        prep, settings, profile, cv_folds=cv_folds,
        on_progress=on_progress, should_stop=should_stop,
    )
    if fitted is None:
        return {}

    model = ClassifierModel(
        vectorizer=fitted.vectorizer,
        head=fitted.head,
        classes=prep.classes,
        task_type=prep.task_type,
        avg_labels=prep.avg_labels,
        uri_to_label=prep.uri_to_label,
        global_threshold=fitted.global_threshold,
        per_label_thresholds=fitted.per_label_thresholds,
    )
    elapsed = time.time() - start
    metadata = _build_metadata(req, settings, profile, prep, fitted, elapsed, cv_folds=cv_folds)
    # Bundle sub-steps feed the job heartbeat: a big skops dump can crawl for
    # many minutes under memory pressure, and phase/progress stay frozen then.
    registry.save(req["model_name"], model, metadata,
                  on_step=lambda detail: on_progress(phase_detail=detail))
    logger.info("Training done: %s f1_macro=%.4f in %.1fs",
                req["model_name"], fitted.metrics["f1_macro"], elapsed)

    return {
        "model_name": req["model_name"],
        "backend": "tfidf",
        "task_type": prep.task_type,
        "n_labels": len(prep.classes),
        "metrics": fitted.metrics,
        "training_time_seconds": round(elapsed, 1),
    }
