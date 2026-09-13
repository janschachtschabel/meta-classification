"""Tests for the selection/deploy stage — what `deploy.py` asks of `tuning.py`.

A unit test of `tuning.select_c` cannot see a profile flag that is never passed on,
which is the likeliest way for a feature behind a flag to end up quietly inert.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from app import deploy, tuning
from app.errors import TrainingInputError
from app.prepare import Prepared
from app.profiles import Profile
from app.settings import Settings


class _RowIdVectorizer:
    """Hands each row's index back as its only feature.

    `select_on_split` passes its matrices straight to the head, so the scripted head
    behind `scripted_c_search` can look its rows up again and the test measures the
    selection rule instead of a solver on toy text.
    """

    def fit_transform(self, texts):
        return self.transform(texts)

    def transform(self, texts):
        # Sparse because `select_on_split` reports the matrix's sparse footprint.
        return sparse.csr_matrix(np.array([[float(text)] for text in texts]))


class _NoFit(AssertionError):
    """Raised instead of fitting, so a test can prove no fit was ever started."""


def _no_fit(*args, **kwargs):
    raise _NoFit("a head was fitted although the run should have been refused")


def _prepared(y: np.ndarray) -> Prepared:
    """All eight rows in every split.

    Deliberately degenerate: these tests are about which C and which thresholds come
    out, not about honest metrics, and the disagreement between the two selection
    rules only exists over the full row set.
    """
    everything = np.arange(len(y))
    return Prepared(
        texts=np.array([str(i) for i in everything]),
        y_all=y, classes=["c0", "c1"], task_type="multilabel",
        avg_labels=1.0, min_samples=1, text_column_weights={},
        train_idx=everything, val_idx=everything, test_idx=everything,
        uri_to_label={"c0": "c0", "c1": "c1"},
    )


def _select(case, *, on_tuned_thresholds: bool):
    profile = Profile("t", c_grid=[2.0, 1.0],
                      select_c_on_tuned_thresholds=on_tuned_thresholds)
    selected = deploy.select_on_split(
        _RowIdVectorizer, _prepared(case.y), Settings(), profile,
        should_stop=lambda: False, on_progress=lambda **kwargs: None,
    )
    assert selected is not None
    return selected


def test_select_on_split_passes_the_profile_flag_into_the_c_search(scripted_c_search):
    """End to end through deploy: the flag has to change which C is deployed and which
    thresholds the bundle gets, not merely exist on the profile."""
    best_c, _global_t, _per_label, metrics = _select(
        scripted_c_search, on_tuned_thresholds=False)
    assert best_c == 1.0
    assert metrics["f1_macro"] == pytest.approx(scripted_c_search.well_scaled_f1, abs=1e-3)

    best_c, global_t, per_label, metrics = _select(
        scripted_c_search, on_tuned_thresholds=True)
    assert best_c == 2.0
    assert (global_t, per_label) == (pytest.approx(0.1),
                                     {"c0": pytest.approx(0.1), "c1": pytest.approx(0.1)})
    assert metrics["f1_macro"] == 1.0


def test_select_on_split_does_not_score_the_validation_split_twice(scripted_c_search):
    """The C search already produced the winner's validation probabilities and, with
    the flag on, its thresholds. Today's second `predict_proba(x_va)` — run only to
    tune those same thresholds again — is then pure repetition."""
    _select(scripted_c_search, on_tuned_thresholds=False)
    without = list(scripted_c_search.scored_rows)
    scripted_c_search.scored_rows.clear()
    _select(scripted_c_search, on_tuned_thresholds=True)

    assert len(without) == 4, (
        "two candidates on validation, the winner's validation rows AGAIN for the "
        f"thresholds, then the test split; got {without}"
    )
    assert len(scripted_c_search.scored_rows) == 3, (
        f"the repeat of the validation split should be gone; got "
        f"{scripted_c_search.scored_rows}"
    )


def _cross_validate(case, monkeypatch, *, on_tuned_thresholds: bool):
    monkeypatch.setattr(deploy, "TfidfBackend", lambda **kwargs: _RowIdVectorizer())
    profile = Profile("t", c_grid=[2.0, 1.0], cv_folds=2,
                      select_c_on_tuned_thresholds=on_tuned_thresholds)
    fitted = deploy.fit_evaluate_deploy(
        _prepared(case.y), Settings(), profile, cv_folds=2,
        on_progress=lambda **kwargs: None, should_stop=lambda: False,
    )
    assert fitted is not None
    return fitted


def test_fit_evaluate_deploy_passes_the_profile_flag_into_the_cv_search(
    scripted_c_search, monkeypatch
):
    """The CV path needs its own wiring test: `cross_val_evaluate` growing the
    parameter does nothing until `fit_evaluate_deploy` hands the profile's flag over,
    and the holdout test above cannot see that.
    """
    assert _cross_validate(scripted_c_search, monkeypatch, on_tuned_thresholds=False).best_c == 1.0

    fitted = _cross_validate(scripted_c_search, monkeypatch, on_tuned_thresholds=True)
    assert fitted.best_c == 2.0
    assert fitted.global_threshold == pytest.approx(0.1)
    assert fitted.metrics["f1_macro"] == 1.0


def test_both_paths_hand_the_profile_shrinkage_to_the_threshold_search(
    scripted_c_search, monkeypatch
):
    """The mechanism is proved in tests/test_thresholds.py; what this pins down is that
    the profile's value reaches it. A knob defined on the profile and never passed on is
    the failure this file exists for — it has already happened once here, when
    `cross_val_evaluate` grew a parameter that `fit_evaluate_deploy` never handed over.
    """
    seen: dict[str, object] = {}

    def spy(name, real):
        def wrapper(*args, **kwargs):
            seen[name] = kwargs.get("threshold_shrink_k", "NOT PASSED")
            return real(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(deploy, "select_c", spy("holdout", deploy.select_c))
    monkeypatch.setattr(deploy, "cross_val_evaluate",
                        spy("cv", deploy.cross_val_evaluate))
    monkeypatch.setattr(deploy, "TfidfBackend", lambda **kwargs: _RowIdVectorizer())

    profile = Profile("t", c_grid=[2.0, 1.0], cv_folds=2, threshold_shrinkage_k=10.0)
    prep = _prepared(scripted_c_search.y)
    deploy.select_on_split(_RowIdVectorizer, prep, Settings(), profile,
                           should_stop=lambda: False, on_progress=lambda **k: None)
    deploy.fit_evaluate_deploy(prep, Settings(), profile, cv_folds=2,
                               on_progress=lambda **k: None, should_stop=lambda: False)

    assert seen == {"holdout": 10.0, "cv": 10.0}


def test_shrinkage_reaches_the_threshold_tuning_a_profile_falls_back_to(
    scripted_c_search, monkeypatch
):
    """`select_on_split` only calls `tune_thresholds` when a profile turns C1's
    selection rule off, so the default configuration never exercises that call site —
    and a shrinkage wired into the other two but not this one would look fully wired.
    """
    seen: list[object] = []
    real = deploy.tune_thresholds
    monkeypatch.setattr(deploy, "tune_thresholds",
                        lambda *a, **kw: (seen.append(kw.get("shrink_k", "NOT PASSED")),
                                          real(*a, **kw))[1])

    profile = Profile("t", c_grid=[2.0, 1.0], threshold_shrinkage_k=10.0,
                      select_c_on_tuned_thresholds=False)
    deploy.select_on_split(_RowIdVectorizer, _prepared(scripted_c_search.y), Settings(),
                           profile, should_stop=lambda: False, on_progress=lambda **k: None)

    assert seen == [10.0]


def _wide_vectorizer(n_features: int):
    """A vectorizer whose vocabulary is as wide as a real run's, and empty otherwise:
    the check reads the matrix's width, and nothing here should get as far as a fit."""
    class _Wide:
        def fit_transform(self, texts):
            return self.transform(texts)

        def transform(self, texts):
            return sparse.csr_matrix((len(texts), n_features), dtype=np.float32)

    return lambda **kwargs: _Wide()


def test_a_head_that_cannot_fit_the_budget_is_refused_before_any_fit(monkeypatch):
    """The fitted head is one float32 per label and feature, and nothing releases it:
    it IS the model. 8 279 keyword labels over a 200 000-term vocabulary are 6.3 GB of
    coefficients alone — more than the whole budget of the container that asked for the
    run, before the input matrix and the solver buffers. Left alone, such a run is
    killed by the OOM killer inside its first fit and reaches the operator as
    `exit code -9` with a peak from whenever the last progress update landed.

    The refusal waits for the matrix, because only it knows how wide the vocabulary
    really got — estimating from the caps refused runs whose vocabulary never
    approaches them. It must land before the first FIT, which is the expensive part
    and the one that dies.
    """
    labels = 8_279
    monkeypatch.setattr(deploy, "TfidfBackend", _wide_vectorizer(200_000))
    monkeypatch.setattr(tuning, "make_head", _no_fit)
    prep = replace(_prepared(np.zeros((8, labels), dtype=np.int8)),
                   classes=[f"uri:{i}" for i in range(labels)])

    with pytest.raises(TrainingInputError) as excinfo:
        deploy.fit_evaluate_deploy(prep, Settings(train_memory_mb=6_000),
                                   Profile("t", c_grid=[1.0]), cv_folds=0,
                                   on_progress=lambda **kwargs: None,
                                   should_stop=lambda: False)

    message = str(excinfo.value)
    assert f"{labels:,}" in message, "name the label count that caused it"
    assert "min_samples_per_label" in message, "name the lever that reduces labels"
    assert "max_word_features" in message, "name the lever that reduces features"
    assert "APIV3_TRAIN_MEMORY_MB" in message, "name the lever that raises the budget"


def test_the_same_run_is_refused_in_cross_validation_too(monkeypatch):
    """The CV path vectorizes on its own, so the holdout check does not cover it — and
    `best`, the profile most likely to be pointed at a big label set, is a CV profile."""
    labels = 8_279
    monkeypatch.setattr(deploy, "TfidfBackend", _wide_vectorizer(200_000))
    monkeypatch.setattr(tuning, "make_head", _no_fit)
    prep = replace(_prepared(np.zeros((8, labels), dtype=np.int8)),
                   classes=[f"uri:{i}" for i in range(labels)])

    with pytest.raises(TrainingInputError):
        deploy.fit_evaluate_deploy(prep, Settings(train_memory_mb=6_000),
                                   Profile("t", c_grid=[1.0], cv_folds=2), cv_folds=2,
                                   on_progress=lambda **kwargs: None,
                                   should_stop=lambda: False)


def test_without_a_memory_budget_no_run_is_refused(monkeypatch):
    """`train_memory_mb=0` switches the cap off, and a check that then invented a limit
    of its own would refuse runs the operator deliberately left unbounded."""
    monkeypatch.setattr(deploy, "TfidfBackend", _wide_vectorizer(200_000))
    monkeypatch.setattr(tuning, "make_head", _no_fit)
    prep = replace(_prepared(np.zeros((8, 8_279), dtype=np.int8)),
                   classes=[f"uri:{i}" for i in range(8_279)])

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the point is the TYPE
        deploy.fit_evaluate_deploy(prep, Settings(train_memory_mb=0),
                                   Profile("t", c_grid=[1.0]), cv_folds=0,
                                   on_progress=lambda **kwargs: None,
                                   should_stop=lambda: False)
    assert isinstance(excinfo.value, _NoFit), "the run was refused instead of reaching its fit"
