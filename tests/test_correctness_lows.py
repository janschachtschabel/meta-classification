"""Four small defects with one thing in common: each turns into something unhelpful much
later than it could (audit CORR-6, CORR-8, CORR-9, PERF-4, PERF-8).

A split configuration that cannot work becomes "Training failed; see server logs" minutes in;
a second implementation of the threshold policy agrees today and is one edit from not; an
unreachable `fit()` waits for the next caller who reaches for the obvious-looking name; and a
partition recomputes from a copy what it could carry forward.
"""

import threading
import time

import numpy as np
import pytest

from app.errors import TrainingInputError
from app.profiles import TrainingConfig

# --- CORR-6: a configuration mistake reads as one ------------------------------------------


@pytest.mark.parametrize(
    "val, test",
    [(0.6, 0.5), (0.5, 0.5), (0.9, 0.2), (1.0, 0.0)],
)
def test_a_split_that_leaves_no_training_rows_is_a_configuration_error(val, test):
    """`validation_size + test_size >= 1.0` leaves nothing to train on. scikit-learn's own
    ValueError escaped to the generic handler and reached the operator as "Training failed;
    see server logs" — a configuration mistake reported as a server fault."""
    from app.real_rows import split_rows

    cfg = TrainingConfig(profiles={}, validation_size=val, test_size=test)

    with pytest.raises(TrainingInputError) as caught:
        split_rows(100, training_cfg=cfg, seed=1, y=None, train_only=None, cv_folds=0)

    message = str(caught.value)
    assert "validation_size" in message and "test_size" in message, message
    # The numbers that are wrong, so the fix needs no guessing.
    assert str(val) in message and str(test) in message, message


def test_a_workable_split_is_untouched():
    from app.real_rows import split_rows

    cfg = TrainingConfig(profiles={}, validation_size=0.15, test_size=0.15)
    (train, val, test), fallback = split_rows(
        100, training_cfg=cfg, seed=1, y=None, train_only=None, cv_folds=0)

    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert fallback is None


def test_the_same_check_guards_the_marked_row_path(monkeypatch):
    """The `train_only` branch had an `except ValueError`, so it turned this into the
    all-rows fallback and trained on a configuration nobody meant."""
    from app.real_rows import split_rows

    cfg = TrainingConfig(profiles={}, validation_size=0.7, test_size=0.7)
    train_only = np.zeros(100, dtype=bool)

    with pytest.raises(TrainingInputError):
        split_rows(100, training_cfg=cfg, seed=1, y=None, train_only=train_only, cv_folds=0)


# --- CORR-8: the threshold policy is written once ------------------------------------------


def test_both_evaluation_modes_resolve_their_cuts_through_one_function():
    """`deploy` (holdout) and `tuning` (k-fold) each had the three-way branch: no tuning ->
    the default cut; tuned during the C search -> keep those; otherwise tune now. They agreed,
    and nothing made them keep agreeing."""
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app"
    for module in ("deploy.py", "tuning.py"):
        source = (app / module).read_text(encoding="utf-8")
        assert "resolve_cuts(" in source, f"{module} still resolves the cuts itself"


@pytest.mark.parametrize("per_label", [True, False])
def test_the_resolver_keeps_what_the_c_search_already_tuned(per_label):
    from app.thresholds import resolve_cuts

    called = []

    def tune():
        called.append(1)
        return 0.9, {"x": 0.9}

    cuts = resolve_cuts(
        applies=True, pretuned=(0.42, np.array([0.3, 0.7])), classes=["a", "b"],
        per_label=per_label, tune=tune,
    )

    assert not called, "re-tuned what the C search had already tuned on the same rows"
    assert cuts[0] == 0.42
    assert cuts[1] == ({"a": 0.3, "b": 0.7} if per_label else {})


def test_the_resolver_tunes_when_nothing_was_tuned_yet():
    from app.thresholds import resolve_cuts

    assert resolve_cuts(applies=True, pretuned=None, classes=["a"], per_label=True,
                        tune=lambda: (0.7, {"a": 0.7})) == (0.7, {"a": 0.7})


def test_a_single_label_task_gets_the_default_cut_and_no_tuning():
    from app.thresholds import resolve_cuts

    def tune():
        raise AssertionError("argmax decides here; there is nothing to tune")

    assert resolve_cuts(applies=False, pretuned=None, classes=["a"], per_label=True,
                        tune=tune) == (0.5, {})


# --- CORR-9: the method that bypassed the two-pass fit is gone ------------------------------


def test_the_backend_offers_no_fit_that_skips_the_two_pass_vocabulary():
    """`TfidfBackend.fit` called scikit-learn's `fit` directly, i.e. the 4-19x memory peak the
    two-pass design exists to avoid. It was unreachable from the pipeline and used by one
    test — so the next caller reaching for the obvious-looking name would have reintroduced a
    documented OOM."""
    from app.vectorizers import TfidfBackend

    assert not hasattr(TfidfBackend, "fit"), (
        "TfidfBackend.fit is back: it bypasses vocabulary.fit_transform_exact"
    )
    assert hasattr(TfidfBackend, "fit_transform"), "the two-pass entry point is gone"


# --- PERF-4: the baseline is computed once -------------------------------------------------


def test_two_first_requests_compute_the_baseline_once():
    """The model object is shared through the registry's LRU cache and `def` routes run in
    worker threads, so two first requests both found `_baseline` unset and both paid the cold
    pass. Deterministic, so the race was benign — the cost was the duplicate work."""
    from app.classifier import ClassifierModel

    computed = []
    started = threading.Event()

    class _SlowHead:
        def predict_proba(self, _x):
            computed.append(1)
            started.set()
            # Long enough that every other thread is inside baseline_proba by now: without
            # the lock they all find `_baseline` unset and all call through.
            time.sleep(0.2)
            return np.array([[0.25, 0.75]])

    class _Vec:
        def transform(self, _texts):
            return np.zeros((1, 2))

    model = ClassifierModel(
        head=_SlowHead(), vectorizer=_Vec(), classes=["a", "b"], task_type="multilabel",
        avg_labels=1.0, uri_to_label={"a": "A", "b": "B"}, global_threshold=0.5,
    )

    answers: list[np.ndarray] = []
    threads = [
        threading.Thread(target=lambda: answers.append(model.baseline_proba()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    assert started.wait(timeout=5), "no thread ever reached the computation"
    for thread in threads:
        thread.join(timeout=10)

    assert len(computed) == 1, f"the cold pass ran {len(computed)} times"
    assert len(answers) == 4
    assert all(np.array_equal(answers[0], other) for other in answers[1:])


# --- PERF-8: the partition carries its counts forward --------------------------------------


def _partition_by_recomputing(y, shares_in, seed):
    """`stratified_partition` as it stood before PERF-8: `remaining` derived from `unplaced` on
    every pass, and otherwise line for line the same — including the leftover pass that places
    rows carrying no label, and the per-subset sort.

    The whole function, not just its loop: a reference that stops at the loop compares 132 rows
    against 137 and reports a divergence of its own making, which is what this one did first.
    """
    from app.stratify import _pick_subset

    shares = np.asarray(shares_in, dtype=float)
    shares = shares / shares.sum()
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    total_debt = n * shares
    label_debt = np.outer(y.sum(axis=0), shares)
    subsets: list[list[int]] = [[] for _ in shares]
    unplaced = np.ones(n, dtype=bool)
    while True:
        remaining = (y[unplaced] > 0).sum(axis=0)
        live = np.flatnonzero(remaining > 0)
        if live.size == 0:
            break
        label = int(live[np.argmin(remaining[live])])
        for row in np.flatnonzero(unplaced & (y[:, label] > 0)):
            chosen = _pick_subset(label_debt[label], total_debt, rng)
            subsets[chosen].append(int(row))
            unplaced[row] = False
            label_debt[y[row] > 0, chosen] -= 1
            total_debt[chosen] -= 1
    for row in np.flatnonzero(unplaced):
        chosen = int(np.argmax(total_debt))
        subsets[chosen].append(int(row))
        total_debt[chosen] -= 1
    return [np.sort(np.array(rows, dtype=int)) for rows in subsets]


@pytest.mark.parametrize(
    "rows, labels, density",
    [(200, 12, 0.15), (500, 40, 0.15), (137, 7, 0.4), (300, 25, 0.03), (64, 3, 0.9)],
)
def test_counting_incrementally_partitions_exactly_as_recomputing_did(rows, labels, density):
    """`remaining` was recomputed as `(y[unplaced] > 0).sum(axis=0)` once per label — a fresh
    copy of the whole unplaced block each pass, measured at 6.3 s for 50k x 300. Carrying the
    counts forward is worth nothing unless it partitions identically, so that is what is
    asserted: against the old algorithm, on five shapes including a very sparse label space
    and rows that carry almost every label."""
    from app.stratify import stratified_partition

    generator = np.random.default_rng(rows * labels)
    y = (generator.random((rows, labels)) < density).astype(np.int8)
    shares = np.array([0.7, 0.15, 0.15])

    reference = _partition_by_recomputing(y, shares, seed=7)
    actual = stratified_partition(y, shares, seed=7)

    assert [list(part) for part in actual] == [list(part) for part in reference], (
        "the incremental count changed which rows land where"
    )


def test_a_row_carrying_no_label_at_all_does_not_hang_the_loop():
    """Termination now depends on the carried count rather than on freshly-derived truth,
    which is how dropping one decrement turned a wrong partition into an infinite loop while
    this file was being written. Rows with no labels are the natural case where the counts go
    to zero with rows still unplaced."""
    from app.stratify import stratified_partition

    y = np.zeros((20, 4), dtype=np.int8)
    y[:5, 0] = 1  # only five rows carry anything

    parts = stratified_partition(y, np.array([0.7, 0.3]), seed=3)

    # Every row lands somewhere — the 15 with no label go wherever the row count is furthest
    # behind, which is what the function documents. The point of the test is that it RETURNS.
    assert sum(len(part) for part in parts) == 20
    assert sorted(int(i) for part in parts for i in part) == list(range(20))
