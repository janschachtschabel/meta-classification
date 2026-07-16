"""Tests for threshold tuning, C selection and metric computation."""

import numpy as np
import pytest

from app import tuning
from app.vectorizers import TfidfBackend


def test_tune_thresholds_finds_separating_value():
    n = 200
    y = np.zeros((n, 2), dtype=int)
    y[:100, 0] = 1
    y[100:, 1] = 1
    proba = np.empty((n, 2))
    proba[:100, 0], proba[100:, 0] = 0.8, 0.2
    proba[:100, 1], proba[100:, 1] = 0.2, 0.8

    global_t, per_label = tuning.tune_thresholds(y, proba, ["c0", "c1"], per_label=True)
    assert 0.2 < global_t < 0.8
    assert set(per_label) == {"c0", "c1"}

    metrics = tuning.compute_metrics(y, proba, ["c0", "c1"], global_t, per_label)
    assert metrics["f1_macro"] > 0.99
    assert set(metrics["per_label_f1"]) == {"c0", "c1"}


def test_select_c_returns_fitted_head():
    rng = np.random.RandomState(1)
    group_a = rng.normal(0.0, 1.0, (40, 4))
    group_b = rng.normal(6.0, 1.0, (40, 4))
    x = np.vstack([group_a, group_b])
    y = np.zeros((80, 2), dtype=int)
    y[:40, 0] = 1
    y[40:, 1] = 1
    order = rng.permutation(80)
    x, y = x[order], y[order]
    x_tr, x_val, y_tr, y_val = x[:60], x[60:], y[:60], y[60:]

    best_c, best_f1, head = tuning.select_c(x_tr, y_tr, x_val, y_val, [0.5, 1.0, 2.0])
    assert best_c in (0.5, 1.0, 2.0)
    assert head is not None
    assert head.predict_proba(x_val).shape == (20, 2)
    assert best_f1 > 0.8


def test_select_c_empty_grid_raises():
    """An empty C grid (e.g. a misconfigured profile in config.yaml) must fail
    loudly with a clear message, not crash with an opaque IndexError/AttributeError
    deep inside the training orchestration."""
    x = np.zeros((4, 3))
    y = np.zeros((4, 2), dtype=int)
    y[:2, 0] = 1
    y[2:, 1] = 1
    with pytest.raises(ValueError, match="[Cc] grid"):
        tuning.select_c(x, y, x, y, [])


def test_cross_val_evaluate_oof_metric():
    """Out-of-fold CV: every row is predicted by a model that did not train on it;
    on separable data this yields a high honest OOF macro-F1 and a valid C."""
    texts = (["mathematik algebra gleichung bruch"] * 20
             + ["geschichte rom antike kaiser"] * 20
             + ["biologie zelle organismus erbgut"] * 20)
    y = np.zeros((60, 3), dtype=int)
    y[:20, 0] = 1
    y[20:40, 1] = 1
    y[40:, 2] = 1
    classes = ["uri:math", "uri:hist", "uri:bio"]

    best_c, global_t, per_label, metrics = tuning.cross_val_evaluate(
        lambda: TfidfBackend(use_char=False, max_word_features=200),
        texts, y, classes,
        k=5, c_grid=[1.0, 2.0], seed=42, n_jobs=1, solver="liblinear",
        tune_threshold=True, per_label=True,
    )
    assert best_c in (1.0, 2.0)
    assert 0.0 <= global_t <= 1.0
    assert set(per_label) == set(classes)
    assert metrics["f1_macro"] > 0.9


def test_tfidf_backend_fit_then_transform_and_unfitted_guard():
    """fit()+transform() (the split-free path) matches the fitted vocabulary, and
    transform() before fit fails loudly instead of returning garbage."""
    texts = ["mathematik algebra gleichung", "geschichte rom antike", "biologie zelle erbgut"]
    backend = TfidfBackend(use_char=True, max_word_features=100, min_df=1)
    backend.fit(texts)
    x = backend.transform(texts)
    assert x.shape[0] == 3 and x.shape[1] > 0
    assert x.dtype == np.float32  # RAM contract: features stay float32

    with pytest.raises(RuntimeError, match="before fit"):
        TfidfBackend().transform(["nie gefittet"])


def test_cross_val_evaluate_reports_granular_steps():
    """Every single head fit reports (done, total, detail) so the caller can map
    it to real progress — a k*|grid| run must not look frozen for minutes."""
    steps: list[tuple[int, int, str]] = []
    texts = (["mathematik algebra gleichung"] * 10 + ["geschichte rom antike"] * 10)
    y = np.zeros((20, 2), dtype=int)
    y[:10, 0] = 1
    y[10:, 1] = 1

    tuning.cross_val_evaluate(
        lambda: TfidfBackend(use_char=False, max_word_features=100),
        texts, y, ["a", "b"], k=2, c_grid=[0.5, 1.0],
        on_step=lambda done, total, detail: steps.append((done, total, detail)),
    )
    assert [(d, t) for d, t, _ in steps] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert all("Fold" in detail and "C=" in detail for _, _, detail in steps)


def test_cross_val_evaluate_stops_between_c_fits(monkeypatch):
    """/train/stop promises cancellation "between the C fits"; the CV path must honour
    that inside a fold too, not only between folds (a fold trains |grid| heads)."""
    fits: list[int] = []
    real_make_head = tuning.make_head

    def counting_make_head(*args, **kwargs):
        fits.append(1)
        return real_make_head(*args, **kwargs)

    monkeypatch.setattr(tuning, "make_head", counting_make_head)

    texts = [f"beispiel text nummer {i} mit inhalt" for i in range(12)]
    y = np.zeros((12, 2), dtype=int)
    y[:6, 0] = 1
    y[6:, 1] = 1
    checks = iter([False])  # fold-start check passes; every later check requests stop

    result = tuning.cross_val_evaluate(
        lambda: TfidfBackend(use_char=False, max_word_features=100),
        texts, y, ["a", "b"],
        k=2, c_grid=[0.5, 1.0, 2.0],
        should_stop=lambda: next(checks, True),
    )
    assert result is None
    assert len(fits) == 0  # stop honoured before the first C fit, not after the whole grid


def test_cross_val_evaluate_rejects_more_folds_than_rows():
    """k > n rows fails with a clear, user-facing message instead of an opaque
    sklearn error deep inside KFold."""
    from app.errors import TrainingInputError

    texts = ["kurzer beispieltext eins", "kurzer beispieltext zwei"]
    y = np.array([[1, 0], [0, 1]])
    with pytest.raises(TrainingInputError, match="cv_folds"):
        tuning.cross_val_evaluate(
            lambda: TfidfBackend(use_char=False, max_word_features=50),
            texts, y, ["a", "b"], k=5, c_grid=[1.0],
        )
