"""Tests for the selection/deploy stage — what `deploy.py` asks of `tuning.py`.

A unit test of `tuning.select_c` cannot see a profile flag that is never passed on,
which is the likeliest way for a feature behind a flag to end up quietly inert.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from app import deploy, tuning
from app.errors import TrainingInputError
from app.memory import MiB
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


class _Matrix:
    """Reports a real run's matrix size without allocating one.

    ``matrix_bytes`` reads the three CSR arrays and nothing else, so a stand-in that
    knows its own size is enough — building 700 MB of non-zeros to assert arithmetic
    would measure the machine, not the rule.
    """

    def __init__(self, rows: int, features: int, size_bytes: int) -> None:
        self.shape = (rows, features)
        third = size_bytes // 3
        self.data = SimpleNamespace(nbytes=third)
        self.indices = SimpleNamespace(nbytes=third)
        self.indptr = SimpleNamespace(nbytes=size_bytes - 2 * third)


def test_a_run_is_refused_when_the_head_fits_but_the_run_does_not():
    """Weighing the coefficients alone only catches the extreme: at 200 000 features a
    6 000 MB budget is not exceeded until ~7 500 labels. Below that a run can still be
    impossible, because the targets and the matrix are held at the same time — the dense
    y is rows x labels, and one fit needs the matrix plus the solver's copies of it.

    4 000 labels over 250 000 rows: the head is 3 052 MB and fits; the head plus the
    targets plus a single fit's matrix copies is 6 456 MB and does not. Judged at ONE
    thread, the fewest the thread budget will ever drop to — anything above that is the
    throttle's job, not this gate's.
    """
    budget = 6_000 * MiB
    head_only = deploy.head_bytes(4_000, 200_000)
    assert head_only < budget, "the case is only interesting while the head itself fits"

    with pytest.raises(TrainingInputError) as excinfo:
        deploy.refuse_if_the_run_cannot_fit(
            n_labels=4_000, targets_bytes=250_000 * 4_000,
            matrix=_Matrix(250_000, 200_000, 700 * MiB), budget_bytes=budget)

    message = str(excinfo.value)
    assert "4,000" in message, "name the labels"
    assert "min_samples_per_label" in message


def test_a_run_that_fits_at_one_thread_is_not_refused():
    """The same shape one step smaller: 1 523 labels are 1 162 MB of coefficients, the
    targets 322 MB, the matrix and one fit's copies 2 450 MB — 3 934 MB against 6 000. The
    thread budget may still drop this run to a single fit at a time, and that is what it
    is for; refusing here would take a run that works."""
    deploy.refuse_if_the_run_cannot_fit(
        n_labels=1_523, targets_bytes=221_915 * 1_523,
        matrix=_Matrix(221_915, 200_000, 700 * MiB), budget_bytes=6_000 * MiB)


def test_a_run_over_the_containers_own_limit_is_refused_even_without_a_budget(monkeypatch):
    """`train_memory_mb=0` switches the THROTTLE off, and that is a legitimate choice: it
    says "use the cores, I know the machine". It cannot switch off the kernel. What the
    OOM killer enforces is the cgroup limit, and a run whose minimum working set exceeds
    that is not a tuning question — it ends with the container's memory, whatever the
    operator asked for.

    Counted against what is ALREADY held (this is the child process, with its matrix and
    targets built), because that is the number the kernel compares too.
    """
    monkeypatch.setattr(deploy, "memory_limit_bytes", lambda: 8_192 * MiB)
    monkeypatch.setattr(deploy, "held_bytes", lambda: 3_000 * MiB)

    with pytest.raises(TrainingInputError) as excinfo:
        deploy.refuse_if_the_run_cannot_fit(
            n_labels=6_000, targets_bytes=250_000 * 6_000,
            matrix=_Matrix(250_000, 200_000, 700 * MiB), budget_bytes=None)

    message = str(excinfo.value)
    assert "8,192 MB" in message, "name the limit that will kill it"
    assert "min_samples_per_label" in message


def test_a_run_inside_the_containers_limit_is_not_refused(monkeypatch):
    """The counterpart: without a budget, only the impossible is refused. A run that
    fits the container is the operator's business, however long it takes."""
    monkeypatch.setattr(deploy, "memory_limit_bytes", lambda: 8_192 * MiB)
    monkeypatch.setattr(deploy, "held_bytes", lambda: 1_000 * MiB)

    deploy.refuse_if_the_run_cannot_fit(
        n_labels=1_000, targets_bytes=200_000 * 1_000,
        matrix=_Matrix(200_000, 200_000, 600 * MiB), budget_bytes=None)


def test_no_limit_and_no_budget_refuses_nothing(monkeypatch):
    """Outside a container there is neither, and inventing one would refuse runs on a
    machine whose memory nobody has declared."""
    monkeypatch.setattr(deploy, "memory_limit_bytes", lambda: None)
    monkeypatch.setattr(deploy, "held_bytes", lambda: 3_000 * MiB)

    deploy.refuse_if_the_run_cannot_fit(
        n_labels=50_000, targets_bytes=250_000 * 50_000,
        matrix=_Matrix(250_000, 200_000, 700 * MiB), budget_bytes=None)


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


# --- rows an LLM wrote: the two evaluation paths must be TOLD which rows are real ---


class _Stop(Exception):
    """Ends a run right after the call a test inspects."""


def _marked_prepared() -> Prepared:
    from app.provenance import GENERATED, RowProvenance

    y = np.array([[1, 0], [0, 1]] * 4, dtype=np.int8)
    marks = np.array([0, 0, 0, 0, 0, 0, GENERATED, GENERATED], dtype=np.int8)
    prep = _prepared(y)
    prep.provenance = RowProvenance(marks=marks, mode="train", validate=marks == 0,
                                    scored=np.array([True, False]))
    return prep


def test_k_fold_is_told_which_rows_may_validate_and_which_labels_score(monkeypatch):
    prep, seen = _marked_prepared(), {}

    def cross_val(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(deploy, "cross_val_evaluate", cross_val)
    with pytest.raises(_Stop):
        deploy.fit_evaluate_deploy(prep, Settings(), Profile("t", c_grid=[1.0]), cv_folds=3,
                                   on_progress=lambda **kwargs: None, should_stop=lambda: False)

    assert seen["validate"] is prep.provenance.validate
    assert seen["scored"] is prep.provenance.scored


def test_the_holdout_scores_only_the_labels_a_real_row_can_validate(monkeypatch):
    prep, seen = _marked_prepared(), {}

    def metrics(*args, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(deploy, "compute_metrics", metrics)
    deploy.select_on_split(_RowIdVectorizer, prep, Settings(), Profile("t", c_grid=[1.0]),
                           should_stop=lambda: False, on_progress=lambda **kwargs: None)

    assert seen["scored"] is prep.provenance.scored



def test_both_paths_are_told_which_labels_keep_the_global_cut(monkeypatch):
    from app.provenance import GENERATED, RowProvenance

    y = np.array([[1, 0], [0, 1]] * 4, dtype=np.int8)
    marks = np.array([0, 0, 0, 0, 0, 0, GENERATED, GENERATED], dtype=np.int8)
    prep = _prepared(y)
    prep.provenance = RowProvenance(marks=marks, mode="train", validate=marks == 0,
                                    thin=np.array([False, True]), thin_mode="global")
    seen: dict = {}

    def cross_val(*args, **kwargs):
        seen["cv"] = kwargs.get("keep_global")
        raise _Stop

    def search(*args, **kwargs):
        seen["select"] = kwargs.get("keep_global")
        raise _Stop

    monkeypatch.setattr(deploy, "cross_val_evaluate", cross_val)
    monkeypatch.setattr(deploy, "select_c", search)
    with pytest.raises(_Stop):
        deploy.fit_evaluate_deploy(prep, Settings(), Profile("t", c_grid=[1.0]), cv_folds=3,
                                   on_progress=lambda **kwargs: None, should_stop=lambda: False)
    with pytest.raises(_Stop):
        deploy.select_on_split(_RowIdVectorizer, prep, Settings(), Profile("t", c_grid=[1.0]),
                               should_stop=lambda: False, on_progress=lambda **kwargs: None)

    assert seen["cv"].tolist() == [False, True]
    assert seen["select"].tolist() == [False, True]
