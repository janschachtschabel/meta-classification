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


def test_metrics_report_how_many_labels_the_model_actually_asserts():
    """Over-assertion is invisible in F1 but decides how a suggestion list feels, and
    it is a property of THIS training target — not of the dataset. Report the asserted
    vs. true label count per model so every bundle carries its own number.

    Here every row truly has 1 label, but both labels score above their cut, so the
    model asserts 2.0 against a true 1.0.
    """
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    proba = np.full((4, 2), 0.9)  # everything above a 0.5 cut -> 2 labels per row
    metrics = tuning.compute_metrics(y, proba, ["c0", "c1"], 0.5, {})
    assert metrics["predicted_labels_per_row"] == 2.0
    assert metrics["true_labels_per_row"] == 1.0


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


def test_tune_thresholds_zero_positive_label_keeps_global(audit_grid=None):
    """A label with NO positives in the val split must fall back to the global
    threshold. Regression: `best_f1 = -1.0` let the first grid value (0.05) win
    because f1=0 > -1, serving rare labels with a near-zero threshold."""
    rng = np.random.default_rng(0)
    n = 100
    y = np.zeros((n, 2), dtype=int)
    y[:50, 0] = 1  # label 0: separable; label 1: zero positives in val
    proba = rng.uniform(0.0, 1.0, size=(n, 2))
    proba[:50, 0] = rng.uniform(0.7, 1.0, size=50)
    proba[50:, 0] = rng.uniform(0.0, 0.3, size=50)

    global_t, per_label = tuning.tune_thresholds(y, proba, ["a", "b"], per_label=True)

    assert per_label["b"] == pytest.approx(global_t), (
        "zero-positive label must keep the global threshold, not the grid minimum"
    )


def test_compute_metrics_multiclass_measures_argmax_serving_rule():
    """Serving decides single-label tasks via argmax and ignores thresholds
    (ClassifierModel.predict). The reported metrics must measure THAT rule:
    with every probability below the threshold, the thresholded rule predicts
    nothing (f1=0) while serving still answers every row correctly via argmax."""
    y = np.array([[1, 0], [0, 1], [1, 0]])
    proba = np.array([[0.40, 0.30], [0.20, 0.45], [0.35, 0.10]])  # argmax all correct

    argmax_metrics = tuning.compute_metrics(y, proba, ["a", "b"], 0.5, {}, task_type="multiclass")
    assert argmax_metrics["f1_macro"] == pytest.approx(1.0)
    assert argmax_metrics["decision_rule"] == "argmax"

    thresholded = tuning.compute_metrics(y, proba, ["a", "b"], 0.5, {})
    assert thresholded["f1_macro"] == pytest.approx(0.0)  # the discrepancy B9 fixes
    assert thresholded["decision_rule"] == "thresholds"


def test_select_c_scores_the_argmax_rule_for_single_label_tasks(monkeypatch):
    """C selection must rank candidates by the decision rule serving will use:
    a multiclass model whose probabilities all sit below 0.5 is perfect under
    argmax but scores 0 under the thresholded rule."""

    class _FixedProbaHead:
        def __init__(self, proba):
            self._proba = proba

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return self._proba

    y_val = np.array([[1, 0], [0, 1]])
    proba = np.array([[0.40, 0.30], [0.20, 0.45]])
    monkeypatch.setattr(tuning, "make_head", lambda c, **kw: _FixedProbaHead(proba))

    x = np.zeros((2, 3))
    _, f1_multiclass, _ = tuning.select_c(x, y_val, x, y_val, [1.0], task_type="multiclass")
    _, f1_multilabel, _ = tuning.select_c(x, y_val, x, y_val, [1.0])
    assert f1_multiclass == pytest.approx(1.0)
    assert f1_multilabel == pytest.approx(0.0)


def test_cross_val_evaluate_multiclass_skips_threshold_tuning():
    """For single-label tasks serving never reads thresholds, so CV must not
    tune them (neutral 0.5/{} in the bundle) and its metrics carry the argmax rule."""
    texts = (["mathematik algebra gleichung bruch"] * 20
             + ["geschichte rom antike kaiser"] * 20)
    y = np.zeros((40, 2), dtype=int)
    y[:20, 0] = 1
    y[20:, 1] = 1

    result = tuning.cross_val_evaluate(
        lambda: TfidfBackend(use_char=False, max_word_features=200),
        texts, y, ["uri:math", "uri:hist"],
        k=4, c_grid=[1.0], seed=42, n_jobs=1, solver="liblinear",
        tune_threshold=True, per_label=True, task_type="multiclass",
    )
    assert result is not None
    _, global_t, per_label, metrics = result
    assert global_t == 0.5
    assert per_label == {}
    assert metrics["decision_rule"] == "argmax"
    assert metrics["f1_macro"] > 0.9


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


def _tiny_corpus(n: int = 60):
    """Two separable topics, enough rows for a 3-fold split."""
    texts = [f"bruchrechnung gleichung mathematik {i}" if i % 2 else
             f"photosynthese pflanze biologie {i}" for i in range(n)]
    y = np.zeros((n, 2), dtype=np.int8)
    y[1::2, 0] = 1
    y[0::2, 1] = 1
    return texts, y, ["uri:math", "uri:bio"]


def test_cross_validation_can_reuse_one_vectorization_instead_of_k():
    """The vectorizer is refit per fold so no fold's test rows influence its vocabulary
    or IDF. That costs k passes over the corpus — at 156 k rows, 2.3 min each. Passing a
    prefitted matrix trades that isolation for one pass; whether the trade is acceptable
    is what `scripts/benchmark_shared_vectorizer.py` measures.

    What this pins is the mechanism: given a matrix, the fold loop must NOT vectorize
    again, or the saving is imaginary.
    """
    texts, y, classes = _tiny_corpus()
    fits = []

    def make_vectorizer():
        fits.append(1)
        return TfidfBackend()

    shared = TfidfBackend()
    matrix = shared.fit_transform(texts)

    result = tuning.cross_val_evaluate(
        make_vectorizer, texts, y, classes, k=3, c_grid=[1.0], matrix=matrix,
    )

    assert result is not None
    best_c, global_t, per_label_t, metrics = result
    assert best_c == 1.0
    assert metrics["n_labels"] == 2
    assert fits == [], "a prefitted matrix means the fold loop vectorizes nothing"


def test_the_refit_path_still_vectorizes_once_per_fold():
    """The counterpart, so the saving above is measured against something real."""
    texts, y, classes = _tiny_corpus()
    fits = []

    def make_vectorizer():
        fits.append(1)
        return TfidfBackend()

    tuning.cross_val_evaluate(make_vectorizer, texts, y, classes, k=3, c_grid=[1.0])

    assert len(fits) == 3, "one vectorizer per fold"


def test_a_matrix_that_does_not_match_the_labels_is_refused():
    """The dangerous failure: fold indices slice both the matrix and `y`, so a matrix
    built from different rows would line row i of one up against row j of the other and
    still produce a plausible-looking number. Refuse it instead."""
    texts, y, classes = _tiny_corpus()
    shared = TfidfBackend()
    matrix = shared.fit_transform(texts[:40])

    with pytest.raises(ValueError, match="rows"):
        tuning.cross_val_evaluate(
            lambda: TfidfBackend(), texts, y, classes, k=3, c_grid=[1.0], matrix=matrix,
        )


def test_a_profile_can_ask_for_the_shared_matrix_and_defaults_to_not():
    """The flag is the mechanism's on-switch. It defaults to refitting because the gate
    asks for two targets and only one is measurable on this machine — see the constant's
    comment in profiles.py for the numbers."""
    from app.profiles import Profile, load_training_config
    from app.settings import get_settings

    assert Profile("x").refit_vectorizer_per_fold is True
    for profile in load_training_config(get_settings().config_file).profiles.values():
        assert profile.refit_vectorizer_per_fold is True, "no shipped profile shares yet"


def test_make_head_uses_sklearns_tolerance_unless_a_looser_one_is_asked_for():
    """The default has to stay sklearn's, because it is the one the DEPLOYED model
    is fit at. A looser tolerance is a search-time trade, never a shipping one."""
    from app.classifier import make_head

    assert make_head().estimator.tol == pytest.approx(1e-4)
    assert make_head(tol=1e-3).estimator.tol == pytest.approx(1e-3)


def test_select_c_fits_at_the_tolerance_it_was_given():
    texts = [f"alpha beta {i}" for i in range(40)] + [f"gamma delta {i}" for i in range(40)]
    y = np.zeros((80, 2), dtype=int)
    y[:40, 0] = 1
    y[40:, 1] = 1
    matrix = TfidfBackend().fit_transform(texts)
    _, _, head = tuning.select_c(matrix, y, matrix, y, [1.0], tol=1e-3)
    assert head.estimator.tol == pytest.approx(1e-3)


def test_cross_val_evaluate_fits_every_candidate_at_the_given_tolerance(monkeypatch):
    """Every fold and every C, not just the first: a tolerance that reaches only
    part of the search would make the benchmark's timing meaningless."""
    seen: list[float | None] = []
    real = tuning.make_head

    def spy(c, **kwargs):
        seen.append(kwargs.get("tol"))
        return real(c, **kwargs)

    monkeypatch.setattr(tuning, "make_head", spy)
    texts = [f"alpha beta {i}" for i in range(30)] + [f"gamma delta {i}" for i in range(30)]
    y = np.zeros((60, 2), dtype=int)
    y[:30, 0] = 1
    y[30:, 1] = 1
    tuning.cross_val_evaluate(
        TfidfBackend, texts, y, ["c0", "c1"], k=2, c_grid=[1.0, 2.0], tol=1e-3,
    )
    assert seen == [1e-3] * 4, f"expected 2 folds x 2 candidates at 1e-3, got {seen}"


def test_a_profile_can_loosen_the_selection_tolerance_and_defaults_not_to():
    """The flag is the mechanism's on-switch, and the shipped profiles do not use it
    until the benchmark's gate is met — see the field's comment in profiles.py."""
    from app.profiles import Profile, load_training_config
    from app.settings import get_settings

    assert Profile("x").selection_tol is None
    for profile in load_training_config(get_settings().config_file).profiles.values():
        assert profile.selection_tol is None, "no shipped profile loosens the search yet"


def test_tune_threshold_columns_is_what_the_uri_keyed_form_is_built_from():
    """Thresholds are a per-COLUMN quantity; the uri -> threshold dict is how the
    bundle stores them. Splitting the two lets a caller score a candidate by its own
    thresholds without carrying the label names into the C search, which is what
    plan item C1 needs. The dict must stay exactly the zip of the columns."""
    # Each label needs a DIFFERENT optimal cut, and at least one different from the
    # global one — on data where every column agrees with the global, a wrapper that
    # ignored the columns entirely would pass and prove nothing (it did, once).
    # Positives/negatives per column: 0.90/0.10, 0.50/0.45, 0.80/0.75.
    n = 60
    y = np.zeros((n, 3), dtype=int)
    proba = np.empty((n, 3))
    for col, (high, low) in enumerate([(0.90, 0.10), (0.50, 0.45), (0.80, 0.75)]):
        block = slice(col * 20, (col + 1) * 20)
        y[block, col] = 1
        proba[:, col] = low
        proba[block, col] = high

    classes = ["c0", "c1", "c2"]
    global_t, columns = tuning.tune_threshold_columns(y, proba, per_label=True)
    wrapped_global, wrapped = tuning.tune_thresholds(y, proba, classes, per_label=True)

    assert columns.shape == (3,)
    assert wrapped_global == global_t
    assert wrapped == {uri: float(t) for uri, t in zip(classes, columns, strict=True)}
    assert not np.all(columns == global_t), (
        "this fixture must produce per-label cuts that differ from the global one, "
        "otherwise the assertion above cannot tell a real zip from a constant"
    )


def test_tune_threshold_columns_without_per_label_repeats_the_global():
    """`per_label=False` means every column decides at the global cut. The dict form
    says that by staying empty (apply_thresholds falls back); the column form has to
    say it by carrying the value, because an array has no 'absent'."""
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    global_t, columns = tuning.tune_threshold_columns(y, proba, per_label=False)
    assert np.all(columns == global_t)
    assert tuning.tune_thresholds(y, proba, ["a", "b"], per_label=False) == (global_t, {})


def test_a_profile_can_pick_c_on_tuned_thresholds_and_defaults_not_to(tmp_path):
    """C1's on-switch. Off for every shipped profile until the benchmark's gate is
    met, so the mechanism can be measured without changing what anyone trains today.

    The yaml half is not decoration: every shipped profile omits the key, so a
    forgotten line in the parser would still read back the code default and let an
    assertion about the defaults pass while `config.yaml` was silently ignored.
    """
    from app.profiles import Profile, load_training_config
    from app.settings import get_settings

    assert Profile("x").select_c_on_tuned_thresholds is False
    for profile in load_training_config(get_settings().config_file).profiles.values():
        assert profile.select_c_on_tuned_thresholds is False, (
            "no shipped profile picks C on tuned thresholds yet"
        )

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "profiles:\n"
        "  tuned:\n"
        "    C_grid: [1.0]\n"
        "    select_c_on_tuned_thresholds: true\n",
        encoding="utf-8",
    )
    assert load_training_config(config_file).get("tuned").select_c_on_tuned_thresholds is True


class _ScriptedHead:
    """A head whose probabilities are read from a table instead of learned.

    The tests below pin down a SELECTION RULE, so the probabilities have to be an
    input to the test rather than an output of a solver — with real fits the answer
    would depend on how well two Cs happen to separate toy data. Row identity travels
    in column 0 of the feature matrix, which is what lets a CV fold's slice look its
    own rows up again.
    """

    def __init__(self, table: np.ndarray) -> None:
        self.table = table

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return self.table[np.asarray(x)[:, 0].astype(int)]


def _selection_disagreement_case():
    """Two candidates the flat cut and tuned thresholds disagree about.

    C=1 is well scaled but misses row 0 of label 0: macro F1 0.929, and no threshold
    can rescue it because the missed row scores exactly what the negatives score.
    C=2 ranks every row perfectly but compresses the scores below 0.5 — the flat cut
    predicts nothing and scores it 0.0, a tuned cut scores it 1.0.

    So selecting on the flat 0.5 cut throws away the candidate that wins once its
    thresholds are set, which is the whole of plan item C1.
    """
    y = np.zeros((8, 2), dtype=int)
    y[:4, 0] = 1
    y[4:, 1] = 1
    well_scaled = np.array([[0.1, 0.1]] + [[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 4)
    compressed = np.array([[0.4, 0.05]] * 4 + [[0.05, 0.4]] * 4)
    row_ids = np.arange(8, dtype=float).reshape(-1, 1)
    return row_ids, y, {1.0: well_scaled, 2.0: compressed}


def _script_the_heads(monkeypatch, tables):
    monkeypatch.setattr(tuning, "make_head", lambda c, **kwargs: _ScriptedHead(tables[c]))


def test_cross_val_evaluate_picks_the_c_that_wins_under_its_own_thresholds(monkeypatch):
    """With the flag on, each candidate is scored under thresholds tuned for itself
    and the (C, thresholds) pair is chosen together; off, today's flat 0.5 cut decides.
    The grid puts the tuned winner FIRST so a rule that kept the last candidate's
    thresholds could not pass by accident."""
    row_ids, y, tables = _selection_disagreement_case()
    _script_the_heads(monkeypatch, tables)
    classes = ["c0", "c1"]

    best_c, global_t, per_label, metrics = tuning.cross_val_evaluate(
        TfidfBackend, [""] * 8, y, classes, matrix=row_ids, k=2, c_grid=[2.0, 1.0],
    )
    assert best_c == 1.0, "the flat cut scores the compressed candidate 0.0"
    assert metrics["f1_macro"] == pytest.approx(6 / 7 / 2 + 0.5, abs=1e-3)

    best_c, global_t, per_label, metrics = tuning.cross_val_evaluate(
        TfidfBackend, [""] * 8, y, classes, matrix=row_ids, k=2, c_grid=[2.0, 1.0],
        select_on_tuned_thresholds=True,
    )
    assert best_c == 2.0, "under its own thresholds the compressed candidate is perfect"
    assert metrics["f1_macro"] == 1.0
    assert global_t == pytest.approx(0.1)
    assert per_label == {"c0": pytest.approx(0.1), "c1": pytest.approx(0.1)}


def test_cross_val_evaluate_ignores_the_tuned_rule_when_thresholds_are_off(monkeypatch):
    """`tune_threshold=False` means the bundle ships no thresholds, so selecting on
    thresholds it will not keep would optimize for a rule serving never applies."""
    row_ids, y, tables = _selection_disagreement_case()
    _script_the_heads(monkeypatch, tables)

    best_c, global_t, per_label, _ = tuning.cross_val_evaluate(
        TfidfBackend, [""] * 8, y, ["c0", "c1"], matrix=row_ids, k=2, c_grid=[2.0, 1.0],
        tune_threshold=False, select_on_tuned_thresholds=True,
    )
    assert (best_c, global_t, per_label) == (1.0, 0.5, {})


def test_cross_val_evaluate_ignores_the_tuned_rule_for_single_label_tasks(monkeypatch):
    """binary/multiclass serving is argmax and reads no threshold; tuning one to
    select on would optimize for a rule that never runs."""
    row_ids, y, tables = _selection_disagreement_case()
    _script_the_heads(monkeypatch, tables)

    best_c, global_t, per_label, _ = tuning.cross_val_evaluate(
        TfidfBackend, [""] * 8, y, ["c0", "c1"], matrix=row_ids, k=2, c_grid=[2.0, 1.0],
        task_type="multiclass", select_on_tuned_thresholds=True,
    )
    assert (global_t, per_label) == (0.5, {})
    assert best_c in (1.0, 2.0)
