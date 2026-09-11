"""Training orchestration for TF-IDF + LogisticRegression.

Pipeline in three modules, one responsibility each:
  - ``prepare.py``  : load -> clean -> label targets -> split
  - ``deploy.py``   : auto-select C + tune thresholds -> honest metrics -> deploy fit
  - here            : orchestration + persisted metadata

Two evaluation modes (``split.cv_folds`` in config, or ``cv_folds`` per request):
the classic train/val/test split, or k-fold cross-validation (every row trains
AND validates via out-of-fold, deploy on 100% of the data).

Callback-driven (``on_progress`` / ``should_stop``) so it carries no threading
logic itself; ``jobs.JobRunner`` runs it in the background.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from .classifier import ClassifierModel
from .deploy import Fitted, fit_evaluate_deploy
from .errors import TrainingInputError
from .memory import MiB, PeakSampler, held_bytes
from .prepare import Prepared, prepare_data
from .profiles import Profile, TrainingConfig
from .registry import Registry
from .settings import Settings

logger = logging.getLogger("api_v3.training")


def _with_memory(on_progress: Callable[..., None], sampler: PeakSampler) -> Callable[..., None]:
    """``on_progress`` that also carries the run's peak RSS, and logs every phase entry
    with the memory it starts from.

    The head fits log nothing of their own, so these lines are what tells a post-mortem
    how far a killed run got and what it held on the way. Both figures are
    ``held_bytes`` — in a child process the API process is counted in, as in
    ``/train/status``, because the container's limit applies to the sum.
    """

    def report(**fields: object) -> None:
        peak_mb = sampler.peak_bytes // MiB or None
        if "phase" in fields:
            logger.info("phase=%s rss=%s MB peak=%s MB", fields["phase"],
                        f"{held_bytes() // MiB:,}", f"{peak_mb or 0:,}")
        on_progress(**{**fields, "peak_rss_mb": peak_mb})

    return report


def _build_metadata(
    req: dict, settings: Settings, profile: Profile, prep: Prepared, fitted: Fitted, elapsed: float,
    *, cv_folds: int = 0, resources: dict | None = None,
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
        # Anchored in the bundle: the model was fit on text where these fields are
        # repeated, so a caller who wants matching behaviour has to know about it.
        # Taken from `prep`, not from `req` — the request may have left it to the
        # config default, and the bundle must describe what was actually applied.
        "text_column_weights": prep.text_column_weights,
        "label_column": req["label_column"],
        "label_filter": req.get("label_filter"),
        "profile": profile.name,
        "backend": "tfidf",
        "solver": settings.solver,
        "best_C": fitted.best_c,
        # The searched grid makes best_C interpretable later: a best_C equal to the
        # first or last entry means the search hit its BOUNDARY — the real optimum
        # may lie outside, which is invisible from best_C alone.
        "c_grid": list(profile.c_grid),
        "task_type": prep.task_type,
        "avg_labels_per_sample": prep.avg_labels,
        "min_samples_per_label": prep.min_samples,
        "tfidf": {
            "use_char": profile.use_char,
            # Both caps were saturated on the 30k data (n_features == the sum), so
            # recording them is what makes n_features interpretable.
            "max_word_features": fitted.max_word_features,
            "max_char_features": fitted.max_char_features if profile.use_char else None,
            "n_features": fitted.deploy_n_features,
            "train_nnz": fitted.deploy_nnz,
            "train_sparse_mb": fitted.deploy_sparse_mb,
        },
        "n_samples": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_labels": len(prep.classes),
        # Rows carrying each label, over ALL data used — not over the evaluated split.
        # Under a holdout split compute_metrics only sees the test share, so a support
        # taken there would report ~15% of the examples while reading as the real count.
        # This is what makes a per-label F1 interpretable: 0.13 on 25 rows is a very
        # different statement from 0.13 on 5,000.
        "per_label_support": {
            uri: int(count)
            for uri, count in zip(prep.classes, prep.y_all.sum(axis=0), strict=True)
        },
        "metrics": fitted.metrics,
        "training_time_seconds": round(elapsed, 1),
        # What the run needed, so the next run of this size can be sized before it
        # starts rather than after it is killed. Describes the run, like the time above.
        **({"resources": resources} if resources else {}),
        # Author-supplied documentation, kept in its own block so a reader can tell a
        # human assertion from a measured fact. Omitted entirely when nothing was given.
        **({"info": req["info"]} if req.get("info") else {}),
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
    # /train refused the name when the run was SUBMITTED; a model imported while
    # it waited in the queue now holds it. Refuse before minutes of fitting, with
    # the reason (TrainingInputError reaches the operator verbatim).
    if registry.exists(req["model_name"]):
        raise TrainingInputError(
            f"Model '{req['model_name']}' already exists — it was created while this training "
            "was queued. Delete it or train under another name."
        )
    start = time.time()
    # Most specific wins: request > profile > config. The profile carries the mode that
    # suits its size class, but an explicit request value still overrides it.
    req_cv = req.get("cv_folds")
    cv_folds = next(
        value for value in (req_cv, profile.cv_folds, training_cfg.cv_folds) if value is not None
    )
    with PeakSampler() as sampler:
        report = _with_memory(on_progress, sampler)
        prep = prepare_data(
            req, settings, training_cfg, cv_folds=cv_folds,
            stratified=profile.stratified_splits,
            on_progress=report, should_stop=should_stop,
        )
        if prep is None:
            return {}
        fitted = fit_evaluate_deploy(
            prep, settings, profile, cv_folds=cv_folds,
            max_word_features=req.get("max_word_features"),
            max_char_features=req.get("max_char_features"),
            on_progress=report, should_stop=should_stop,
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
            # Set here too, not only when reading the bundle back: registry.save()
            # publishes THIS object straight into the LRU cache, so a freshly trained
            # model would otherwise serve without label_f1 until it is evicted.
            per_label_f1=dict(fitted.metrics.get("per_label_f1", {})),
        )
        elapsed = time.time() - start
        metadata = _build_metadata(
            req, settings, profile, prep, fitted, elapsed, cv_folds=cv_folds,
            resources={
                "peak_rss_mb": sampler.peak_bytes // MiB or None,
                "train_memory_budget_mb": fitted.train_memory_budget_mb,
                "head_fit_threads": fitted.head_fit_threads,
            },
        )
        # Bundle sub-steps feed the job heartbeat: a big skops dump can crawl for
        # many minutes under memory pressure, and phase/progress stay frozen then.
        registry.save(req["model_name"], model, metadata,
                      on_step=lambda detail: report(phase_detail=detail))
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
