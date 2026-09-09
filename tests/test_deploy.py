"""Tests for the selection/deploy stage — what `deploy.py` asks of `tuning.py`.

A unit test of `tuning.select_c` cannot see a profile flag that is never passed on,
which is the likeliest way for a feature behind a flag to end up quietly inert.
"""

import numpy as np
import pytest
from scipy import sparse

from app import deploy
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
