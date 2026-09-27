"""Classifier head and the runtime model object used for prediction.

The head is a plain scikit-learn ``OneVsRestClassifier(LogisticRegression)``.
LogisticRegression yields calibrated probabilities natively, so no extra
probability calibration step is needed (unlike SVMs).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .data import clean_text
from .metrics import is_single_label
from .vectorizers import TfidfBackend


def make_head(
    c: float = 1.0, max_iter: int = 1000, n_jobs: int = 1, solver: str = "liblinear",
    tol: float | None = None,
) -> OneVsRestClassifier:
    """Build a multilabel-capable logistic-regression head.

    ``n_jobs`` sets per-label parallelism. The float32-preserving, GIL-releasing
    solvers 'newton-cg' (the API's configured default) and 'saga' let joblib's
    'threading' backend train all labels on ONE shared input matrix; each
    concurrent fit's solver buffers (~2-2.5x the matrix for newton-cg) come on
    top, which is why training sizes ``n_jobs`` with ``memory.ThreadBudget``.
    'lbfgs' frees the GIL too but upcasts to float64 (2x matrix); 'liblinear' is
    GIL-bound and parallelises only via processes (copying the matrix per worker).

    ``tol`` is left to scikit-learn (1e-4) unless a caller asks for something
    else, and it is OMITTED rather than passed through as a default so there is
    exactly one place that decides what a shipped model is fit at. Loosening it
    is a search-time trade — see ``Profile.selection_tol``.
    """
    return OneVsRestClassifier(
        LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter, solver=solver,
                           **({} if tol is None else {"tol": tol})),
        n_jobs=n_jobs,
    )


@dataclass
class Prediction:
    """One predicted label for one input text.

    ``baseline_diff`` (only set when requested) is the confidence minus the
    model's empty-text prediction for the label: it separates what the TEXT
    contributes from the label's base rate. A high confidence with a diff near
    zero means the label fires for almost anything, not for this text.

    ``label_f1`` (only set when requested) is this label's F1 from the training
    evaluation — how well the model does on this label AT ALL, independent of
    the current text. Confidence answers "how sure here?", label_f1 answers "how
    much is that worth?": 0.95 on a label scoring 0.60 deserves a human look.
    """

    uri: str
    label: str
    confidence: float
    baseline_diff: float | None = None
    label_f1: float | None = None
    # Only set in ranking mode (explicit top_k): whether this label would also
    # pass its tuned threshold — keeps forced rankings honest.
    above_threshold: bool | None = None


@dataclass
class ClassifierModel:
    """A fitted, self-contained model: backend + head + thresholds + metadata."""

    vectorizer: TfidfBackend
    head: OneVsRestClassifier
    classes: list[str]
    task_type: str
    avg_labels: float
    uri_to_label: dict[str, str]
    global_threshold: float = 0.5
    per_label_thresholds: dict[str, float] = field(default_factory=dict)
    # Per-label F1 from the training evaluation (uri -> score), read from the
    # bundle's metrics.json. Reporting-only: never influences a decision, it just
    # travels with the prediction so a caller can weigh it. Empty for bundles
    # trained before this existed — the field is then simply absent.
    per_label_f1: dict[str, float] = field(default_factory=dict)
    # Lazily computed empty-text probabilities (deterministic per model, so
    # cached once); runtime-only, never persisted in the bundle.
    _baseline: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    # Guards the line above. This object is shared: the registry's LRU cache hands the same
    # instance to every request, and `def` routes run in worker threads.
    _baseline_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False)

    def baseline_proba(self) -> np.ndarray:
        """Per-label probabilities for an empty text — the model's base rates.

        Double-checked: the fast path is a plain attribute read, and only the first callers
        take the lock. Two concurrent first requests both found this unset and both paid the
        cold pass — benign, because the answer is deterministic, but it is the pass this
        cache exists to avoid (audit PERF-4).
        """
        if self._baseline is None:
            with self._baseline_lock:
                if self._baseline is None:
                    self._baseline = self.predict_proba([""])[0]
        return self._baseline

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Per-label probabilities of shape (n_texts, n_labels).

        Inputs are cleaned with the same ``clean_text`` used at training time, so the
        TF-IDF vocabulary (fit on cleaned text) sees matching tokens at inference
        instead of raw HTML/Markdown (a train/serve skew).
        """
        cleaned = [clean_text(t) for t in texts]
        return self.head.predict_proba(self.vectorizer.transform(cleaned))

    def _threshold_for(self, uri: str, override: float | None) -> float:
        if override is not None:
            return override
        return self.per_label_thresholds.get(uri, self.global_threshold)

    @property
    def _is_single_label(self) -> bool:
        """Whether exactly one label comes back — the taxonomy's rule, not a second copy."""
        return is_single_label(self.task_type)

    def resolved_top_k(self, top_k: int | None) -> int | None:
        """The ranking size an explicit ``top_k`` resolves to (``0`` = the
        training set's typical label count); ``None`` = threshold mode."""
        if top_k is None:
            return None
        if top_k > 0:
            return top_k
        return 1 if self._is_single_label else max(1, round(self.avg_labels))

    def applied_top_k(self, top_k: int | None) -> int | None:
        """The ranking size a caller should be *told* was applied.

        ``resolved_top_k`` answers a different question — what ``predict`` ranks by, where
        ``None`` means "use the thresholds instead". For reporting, a single-label task with
        no ``top_k`` still returns exactly one label, because argmax does: so the number
        applied is 1, not "no cap". Both answers come from the same place, because the
        route used to derive this one itself.
        """
        resolved = self.resolved_top_k(top_k)
        if resolved is None and self._is_single_label:
            return 1
        return resolved

    def predict(
        self,
        texts: list[str],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        label_filter: str | None = None,
        include_baseline_diff: bool = False,
        include_label_f1: bool = False,
    ) -> list[list[Prediction]]:
        """Classify texts, returning sorted predictions per text.

        Two modes:
        - ``top_k=None`` (default) — the DECISION: multilabel returns every label
          above its tuned per-label threshold; multiclass/binary return the single
          argmax label.
        - ``top_k`` set — a RANKING: exactly the K most probable labels regardless
          of thresholds (``0`` = the training set's typical label count). Each
          prediction then carries ``above_threshold`` so forced entries stay
          distinguishable from asserted ones (multilabel only).

        ``include_baseline_diff`` attaches confidence minus the empty-text
        baseline per prediction; ``include_label_f1`` attaches the label's
        training F1 (both diagnostic, both opt-in; see ``Prediction``).
        """
        proba = self.predict_proba(texts)
        baseline = self.baseline_proba() if include_baseline_diff else None
        single = self._is_single_label
        rank_k = self.resolved_top_k(top_k)
        results: list[list[Prediction]] = []
        for row in range(len(texts)):
            scored = [
                Prediction(
                    uri, self.uri_to_label.get(uri, uri), float(proba[row, col]),
                    baseline_diff=(
                        float(proba[row, col] - baseline[col]) if baseline is not None else None
                    ),
                    label_f1=self.per_label_f1.get(uri) if include_label_f1 else None,
                )
                for col, uri in enumerate(self.classes)
                if not (label_filter and label_filter not in uri)
            ]
            scored.sort(key=lambda p: p.confidence, reverse=True)
            if rank_k is not None:
                chosen = scored[:rank_k]
                if not single:
                    for p in chosen:
                        p.above_threshold = p.confidence >= self._threshold_for(p.uri, threshold)
            elif single:
                chosen = scored[:1]
            else:
                chosen = [p for p in scored if p.confidence >= self._threshold_for(p.uri, threshold)]
            results.append(chosen)
        return results
