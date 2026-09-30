"""Model selection, evaluation and the final deploy fit.

Second stage of the training pipeline (see ``training.py`` for the orchestration).
Two evaluation modes: the classic train/val/test split (select/tune on validation,
honest metrics on the held-out test, deploy on train+val) or k-fold
cross-validation (every row trains, every real one -- none an LLM wrote or touched --
validates via out-of-fold, deploy on 100% of the data).

All label-wise fits run under the caller-configured joblib backend: 'threading'
with a float32-preserving, GIL-releasing solver (newton-cg by default) trains all
labels in parallel on ONE shared input matrix. The solver's working buffers are
per fit, though (~2.5x the matrix each), so the thread count is also a memory
multiplier: a ``memory.ThreadBudget`` sizes every fit to the run's memory budget.

The memory feasibility check lived here until it outgrew the 35 lines this paragraph once
gave as the reason to keep it; it is :mod:`app.feasibility` now (audit 2026-09-27, M-2).
What stays is WHEN it is asked, which only the pipeline knows: before each selection fit
(the first of them also weighing the deploy fit, projected), and before the deploy fit,
without the out-of-fold buffers that are gone by then (audit 2026-09-30, T06). That keeps
the module a little past the ~300-line guide, holding one thing: the fit itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from joblib import parallel_backend
from sklearn.multiclass import OneVsRestClassifier

from .classifier import make_head
from .feasibility import projected, refuse_if_the_run_cannot_fit
from .memory import (
    MiB,
    ThreadBudget,
    matrix_bytes,
)
from .metrics import compute_metrics, is_single_label
from .prepare import Prepared
from .profiles import Profile
from .settings import Settings
from .thresholds import resolve_cuts, tune_thresholds
from .tuning import cross_val_evaluate, select_c
from .vectorizers import TfidfBackend

logger = logging.getLogger("api_v3.training")


def _sparse_mb(matrix) -> float:
    """Megabytes held by a scipy CSR matrix (data + index arrays)."""
    return matrix_bytes(matrix) / MiB


def _threads_note(budget: ThreadBudget | None) -> str:
    """' · 4 threads' for the fit that just ran: a run slowed by its memory budget
    explains itself in the status line instead of just looking slow."""
    if budget is None or not budget.chosen:
        return ""
    threads = budget.chosen[-1]
    return f" · {threads} thread{'s' if threads != 1 else ''}"


@dataclass
class Fitted:
    """A deployed model plus the figures needed to build its metadata."""

    head: OneVsRestClassifier
    vectorizer: TfidfBackend
    best_c: float
    global_threshold: float
    per_label_thresholds: dict[str, float]
    metrics: dict
    deploy_n_features: int
    deploy_nnz: int
    deploy_sparse_mb: float
    # The caps actually in force (request > profile > settings), so the persisted
    # metadata explains a feature count instead of leaving it to be guessed.
    max_word_features: int
    max_char_features: int
    # What the memory budget allowed: the budget itself (None = no cap) and the
    # requested/min/max thread count over all head fits (ThreadBudget.summary).
    train_memory_budget_mb: int | None = None
    head_fit_threads: dict[str, int] | None = None


def select_on_split(
    new_vectorizer: Callable[[], TfidfBackend],
    prep: Prepared,
    settings: Settings,
    profile: Profile,
    *,
    should_stop: Callable[[], bool],
    on_progress: Callable[..., None],
    thread_budget: ThreadBudget | None = None,
) -> tuple[float, float, dict[str, float], dict] | None:
    """Classic path: fit on train, auto-select C + tune thresholds on validation,
    report honest metrics on the held-out test split. Returns
    ``(best_c, global_threshold, per_label_thresholds, metrics)`` or ``None`` if
    cancelled. Runs under the caller's joblib backend context; ``thread_budget``
    sizes the C search's fits to the run's memory budget.
    """
    texts, classes = prep.texts, prep.classes
    y_train, y_val, y_test = (
        prep.y_all[prep.train_idx], prep.y_all[prep.val_idx], prep.y_all[prep.test_idx]
    )

    on_progress(phase="features", progress=35,
                message="Computing TF-IDF features (word + character n-grams)...")
    vectorizer = new_vectorizer()
    x_tr = vectorizer.fit_transform(texts[prep.train_idx].tolist())
    x_va = vectorizer.transform(texts[prep.val_idx].tolist())
    x_te = vectorizer.transform(texts[prep.test_idx].tolist())
    logger.info("TF-IDF train matrix: %d dims, nnz=%d, ~%.0f MB sparse (float32)",
                x_tr.shape[1], x_tr.nnz, _sparse_mb(x_tr))

    on_progress(phase="selecting", progress=55,
                message=f"Selecting regularization strength C – {len(profile.c_grid)} candidates on validation...")
    # Single-label serving is argmax, so thresholds are neither tuned nor stored there.
    thresholds_apply = profile.tune_threshold and not is_single_label(prep.task_type)
    # Thin labels take the global cut when the run chose that (provenance).
    keep_global = prep.provenance.keep_global() if prep.provenance else None
    best_c, val_f1, head, val_thresholds = select_c(
        x_tr, y_train, x_va, y_val, profile.c_grid, n_jobs=settings.effective_n_jobs(),
        should_stop=should_stop, solver=settings.solver, task_type=prep.task_type,
        tol=profile.selection_tol,
        select_on_tuned_thresholds=profile.select_c_on_tuned_thresholds and thresholds_apply,
        threshold_per_label=profile.threshold_per_label,
        threshold_shrink_k=profile.threshold_shrinkage_k,
        thread_budget=thread_budget,
        keep_global=keep_global,
        # Distribute the C search across 55->75% so progress (and thus the ETA
        # derived from it) keeps moving through the longest phase.
        on_step=lambda i, total, c, f: on_progress(
            progress=55 + round(20 * i / total),
            phase_detail=(f"Testing regularization: C={c} ({i}/{total}) – best F1 so far "
                          f"{f:.3f}{_threads_note(thread_budget)}"),
        ),
    )
    if should_stop():
        return None
    logger.info("Selected C=%s (val_f1_macro=%.4f)", best_c, val_f1)

    def tune_now() -> tuple[float, dict[str, float]]:
        # Only the branch that actually tunes reports a phase: the other two do no work.
        on_progress(phase="threshold", progress=75,
                    message="Tuning per-label classification thresholds (on validation)...")
        return tune_thresholds(
            y_val, head.predict_proba(x_va), classes, per_label=profile.threshold_per_label,
            shrink_k=profile.threshold_shrinkage_k, keep_global=keep_global,
        )

    global_t, per_label = resolve_cuts(
        applies=thresholds_apply, pretuned=val_thresholds, classes=classes,
        per_label=profile.threshold_per_label, tune=tune_now,
    )

    on_progress(phase="evaluating", progress=85,
                message="Evaluating on the held-out test split (honest metrics)...")
    metrics = compute_metrics(
        y_test, head.predict_proba(x_te), classes, global_t, per_label,
        task_type=prep.task_type,
        # val and test hold real rows only (prepare); a label whose rows an LLM wrote
        # has nothing to be scored on there.
        scored=prep.provenance.scored if prep.provenance else None,
    )
    return best_c, global_t, per_label, metrics


def fit_evaluate_deploy(
    prep: Prepared,
    settings: Settings,
    profile: Profile,
    *,
    cv_folds: int,
    max_word_features: int | None = None,
    max_char_features: int | None = None,
    on_progress: Callable[..., None],
    should_stop: Callable[[], bool],
) -> Fitted | None:
    """Select C + thresholds, then refit the deploy model on all available data.
    Returns ``None`` if cancelled.

    ``cv_folds >= 2`` runs k-fold cross-validation (every row trains, every real one
    validates via out-of-fold) and deploys on 100% of the data; otherwise the classic
    train/val/test split is used and the deploy model is refit on train+val. All
    label-wise fits share ONE input matrix under the configured joblib backend
    (threading + a float32 solver); each fit's solver buffers come on top, so every
    fit runs on the threads the memory budget allows (``settings.train_memory_mb``).

    ``max_word_features`` / ``max_char_features`` override the vocabulary caps for
    this run (most specific wins: request -> profile -> settings), so the main
    RAM/quality lever can be explored without changing the deployment.
    """
    word_cap = max_word_features or profile.max_word_features or settings.tfidf_max_word_features
    char_cap = max_char_features or profile.max_char_features or settings.tfidf_max_char_features

    def new_vectorizer() -> TfidfBackend:
        return TfidfBackend(
            use_char=profile.use_char,
            max_word_features=word_cap,
            max_char_features=char_cap,
        )

    texts, y_all = prep.texts, prep.y_all
    # The CPU budget (cpu_max_percent) caps the raw n_jobs here — BLAS is pinned
    # to 1 thread, so these head-fit threads are the training's CPU footprint.
    n_jobs = settings.effective_n_jobs()
    # ...and the memory budget caps them again, per fit: every concurrent fit holds
    # ~2.5x its matrix, so a thread count that suits the CPU can outgrow the RAM.
    budget_bytes = settings.effective_train_memory_bytes()
    validate = prep.provenance.validate if prep.provenance else None
    # k-fold allocates one full (validating rows x labels) float32 buffer PER C CANDIDATE
    # before the first fit and holds them all until the winner is picked. Independent of k,
    # and the term that grows fastest: 858 MB at 250k rows x 300 labels x 3 candidates,
    # against 72 MB of targets. The holdout path allocates none.
    oof_rows = len(y_all) if validate is None else int(validate.sum())
    oof_bytes = oof_rows * y_all.shape[1] * 4 * len(profile.c_grid) if cv_folds >= 2 else 0

    def weigh(matrix: object, oof: int) -> None:
        refuse_if_the_run_cannot_fit(n_labels=y_all.shape[1], targets_bytes=y_all.nbytes,
                                     matrix=matrix, budget_bytes=budget_bytes, oof_bytes=oof)

    # The deploy fit comes LAST and has the most rows: all of them under k-fold, train + val
    # on the holdout path. Weighed only when it came, a run that could never finish ran every
    # selection fit first -- hours near the budget -- and was refused at the end (audit
    # 2026-09-30, T06). So the first selection fit weighs it too, projected from its own
    # matrix, before anything is fitted.
    deploy_rows = len(texts) if cv_folds >= 2 else len(prep.train_idx) + len(prep.val_idx)
    first = [True]

    def before_selection_fit(matrix: object) -> None:
        if first[0]:
            first[0] = False
            weigh(projected(matrix, deploy_rows), 0)
        weigh(matrix, oof_bytes)

    thread_budget = ThreadBudget(
        requested=n_jobs, budget_bytes=budget_bytes,
        on_choice=lambda threads: on_progress(head_fit_threads=threads, threads_requested=n_jobs),
        # Every fit asks the budget for its thread count first, whichever path built its
        # matrix — so this is where a run too big to finish gets stopped, with the width
        # the vocabulary actually reached and the targets it actually built.
        before_fit=before_selection_fit,
    )

    with parallel_backend(settings.parallel_backend, n_jobs=n_jobs):
        if cv_folds >= 2:
            shared_matrix = None
            if not profile.refit_vectorizer_per_fold:
                # One pass instead of k, at the cost of the fold's test rows shaping the
                # vocabulary and IDF. Off for every shipped profile — see the flag's
                # comment in profiles.py for what was measured. The deploy fit still
                # vectorizes on its own; folding that in too is a further saving with its
                # own measurement to make.
                on_progress(phase="features", progress=40,
                            message="Vectorizing once for all folds (shared matrix)...")
                shared_matrix = new_vectorizer().fit_transform(texts.tolist())
            rows = "all rows train + validate" if validate is None else "all rows train, real rows validate"
            on_progress(phase="cross-validating", progress=45,
                        message=f"{cv_folds}-fold cross-validation ({rows})...")
            selected = cross_val_evaluate(
                new_vectorizer, texts.tolist(), y_all, prep.classes, matrix=shared_matrix,
                k=cv_folds, c_grid=profile.c_grid, seed=settings.random_seed,
                n_jobs=n_jobs, solver=settings.solver,
                tune_threshold=profile.tune_threshold, per_label=profile.threshold_per_label,
                tol=profile.selection_tol,
                select_on_tuned_thresholds=profile.select_c_on_tuned_thresholds,
                threshold_shrink_k=profile.threshold_shrinkage_k,
                stratified=profile.stratified_splits,
                should_stop=should_stop, task_type=prep.task_type,
                thread_budget=thread_budget,
                # Rows an LLM wrote or touched train in every fold and are never scored.
                validate=validate,
                scored=prep.provenance.scored if prep.provenance else None,
                keep_global=prep.provenance.keep_global() if prep.provenance else None,
                # Distribute the k x |grid| fits across 45->90% (the 30k CV run sat
                # at a frozen 45% for ~25 min, turning the ETA meaningless).
                on_step=lambda done, total, detail: on_progress(
                    progress=45 + round(45 * done / total),
                    phase_detail=f"{detail}{_threads_note(thread_budget)}",
                ),
            )
            # The folds are done with it; the deploy fit vectorizes on its own and must
            # not build its matrix on top of this one.
            shared_matrix = None
            deploy_idx = np.arange(len(texts))
        else:
            selected = select_on_split(
                new_vectorizer, prep, settings, profile,
                should_stop=should_stop, on_progress=on_progress, thread_budget=thread_budget,
            )
            deploy_idx = np.concatenate([prep.train_idx, prep.val_idx])
        if selected is None:
            return None
        best_c, global_t, per_label, metrics = selected

        # --- Deploy: refit on all selected data; keep the tuned thresholds ---
        on_progress(phase="saving", progress=92,
                    message="Training the final model and saving the bundle...")
        # Sub-step details double as heartbeats: each of these can run for
        # minutes with no other progress signal.
        vectorizer = new_vectorizer()
        on_progress(phase_detail="Computing deploy TF-IDF features...")
        x_deploy = vectorizer.fit_transform(texts[deploy_idx].tolist())
        # The out-of-fold buffers went with cross_val_evaluate: this fit holds none (T06).
        thread_budget.before_fit = lambda matrix: weigh(matrix, 0)
        final_head = make_head(best_c, n_jobs=thread_budget.for_matrix(x_deploy),
                               solver=settings.solver)
        on_progress(phase_detail="Fitting the final model on all deploy rows"
                                 f"{_threads_note(thread_budget)}...")
        final_head.fit(x_deploy, y_all[deploy_idx])

    return Fitted(
        head=final_head, vectorizer=vectorizer, best_c=best_c,
        global_threshold=global_t, per_label_thresholds=per_label, metrics=metrics,
        deploy_n_features=int(x_deploy.shape[1]), deploy_nnz=int(x_deploy.nnz),
        deploy_sparse_mb=round(_sparse_mb(x_deploy), 1),
        max_word_features=word_cap, max_char_features=char_cap,
        train_memory_budget_mb=budget_bytes // MiB if budget_bytes else None,
        head_fit_threads=thread_budget.summary(),
    )
