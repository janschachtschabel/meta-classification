"""Tests for where the decision cuts sit and what they do to probabilities."""

import time

import numpy as np
import pytest

from app import thresholds
from app.metrics import compute_metrics


def test_tune_thresholds_finds_separating_value():
    n = 200
    y = np.zeros((n, 2), dtype=int)
    y[:100, 0] = 1
    y[100:, 1] = 1
    proba = np.empty((n, 2))
    proba[:100, 0], proba[100:, 0] = 0.8, 0.2
    proba[:100, 1], proba[100:, 1] = 0.2, 0.8

    global_t, per_label = thresholds.tune_thresholds(y, proba, ["c0", "c1"], per_label=True)
    assert 0.2 < global_t < 0.8
    assert set(per_label) == {"c0", "c1"}

    metrics = compute_metrics(y, proba, ["c0", "c1"], global_t, per_label)
    assert metrics["f1_macro"] > 0.99
    assert set(metrics["per_label_f1"]) == {"c0", "c1"}

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

    global_t, per_label = thresholds.tune_thresholds(y, proba, ["a", "b"], per_label=True)

    assert per_label["b"] == pytest.approx(global_t), (
        "zero-positive label must keep the global threshold, not the grid minimum"
    )

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
    global_t, columns = thresholds.tune_threshold_columns(y, proba, per_label=True)
    wrapped_global, wrapped = thresholds.tune_thresholds(y, proba, classes, per_label=True)

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

    global_t, columns = thresholds.tune_threshold_columns(y, proba, per_label=False)
    assert np.all(columns == global_t)
    assert thresholds.tune_thresholds(y, proba, ["a", "b"], per_label=False) == (global_t, {})


def _uneven_label_support():
    """Four labels, deterministic cuts, built so shrinkage has something to say.

    Labels 0 and 1 both have an optimal cut of 0.65 but wildly different support — 60
    positives against 4 — which is the whole question shrinkage answers. Labels 2 and 3
    are populous and optimal at 0.35, which is what holds the GLOBAL cut at 0.35, away
    from the other two; without that the per-label answers would equal the global and
    shrinking toward it would be a no-op that proves nothing. 0.35 rather than 0.50 on
    purpose: with the global sitting at 0.50, an implementation that shrank toward a
    hardcoded 0.5 passed every assertion here (it did).
    """
    n = 120
    y = np.zeros((n, 4), dtype=int)
    proba = np.empty((n, 4))
    # (positive score, negative score, number of positives)
    for col, (high, low, n_pos) in enumerate(
        [(0.9, 0.6, 60), (0.9, 0.6, 4), (0.4, 0.3, 60), (0.4, 0.3, 60)]
    ):
        proba[:, col] = low
        y[:n_pos, col] = 1
        proba[:n_pos, col] = high
    return y, proba, np.array([60, 4, 60, 60])


def test_shrinkage_trusts_a_threshold_in_proportion_to_the_positives_behind_it():
    """A cut fitted on 4 positives is mostly noise; one fitted on 60 is not, and the
    F1-argmax rule cannot tell the two apart on its own.

    Labels 0 and 1 land on the SAME unshrunk cut, so the only thing separating them
    afterwards is how much support each had. The formula is pinned against the unshrunk
    answer, so a wrong weight, a wrong direction, or a shrinkage applied to the global
    instead of to the columns all fail.
    """
    y, proba, counts = _uneven_label_support()
    k = 10.0

    global_t, plain = thresholds.tune_threshold_columns(y, proba, per_label=True)
    shrunk_global, shrunk = thresholds.tune_threshold_columns(
        y, proba, per_label=True, shrink_k=k)

    assert shrunk_global == global_t, "shrinkage moves per-label cuts, never the global"
    assert plain[0] == plain[1] != global_t, (
        "the two labels must start from one cut that differs from the global, or the "
        "comparison below is not about support at all"
    )

    weight = counts / (counts + k)
    assert shrunk == pytest.approx(weight * plain + (1 - weight) * global_t)
    assert abs(shrunk[1] - global_t) < abs(shrunk[0] - global_t), (
        f"the label with 4 positives should end up nearer the global than the one with "
        f"60; got {shrunk[1]:.4f} and {shrunk[0]:.4f} against a global {global_t}"
    )


def test_shrinkage_off_by_default_leaves_every_threshold_where_it_was():
    y, proba, _ = _uneven_label_support()
    global_t, plain = thresholds.tune_threshold_columns(y, proba, per_label=True)
    assert thresholds.tune_threshold_columns(y, proba, per_label=True, shrink_k=None) == (
        global_t, pytest.approx(plain))


def test_shrinkage_leaves_a_label_with_no_positives_at_the_global_threshold():
    """The zero-positive case already fell back to the global; under shrinkage it is
    the limit of the same rule (weight 0), not a second special case."""
    y = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    proba = np.array([[0.9, 0.3], [0.8, 0.4], [0.2, 0.7], [0.1, 0.6]])
    global_t, columns = thresholds.tune_threshold_columns(y, proba, per_label=True, shrink_k=10.0)
    assert columns[1] == pytest.approx(global_t)


def test_a_column_told_to_keep_the_global_cut_keeps_it_despite_its_positives():
    """A label that reached the minimum only through AI-marked rows has a handful of
    real positives; its own cut is tuned on those few. The run can decide to give it
    the global cut instead."""
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]])
    proba = np.array([[0.9, 0.2], [0.8, 0.1], [0.3, 0.12], [0.2, 0.14], [0.7, 0.13], [0.1, 0.05]])

    global_t, own = thresholds.tune_threshold_columns(y, proba)
    _, kept = thresholds.tune_threshold_columns(y, proba, keep_global=np.array([True, False]))

    assert own[0] != global_t, "test setup: column 0 tunes a cut of its own"
    assert kept[0] == global_t
    assert kept[1] == own[1]


def _naive_search(y, proba, grid):
    """The search written out the slow, obvious way: every cut scored with sklearn.

    This is the reference the vectorised sweep has to reproduce exactly — including how
    it breaks ties (ascending grid, strict `>`, so the LOWEST cut reaching the maximum
    wins) and how it treats a label with no positives (keep the global cut; every
    threshold scores 0 there, and the `>` rule would otherwise hand it the grid minimum).
    """
    from sklearn.metrics import f1_score

    best_global, best_global_f1 = 0.5, -1.0
    for threshold in grid:
        score = f1_score(y, (proba >= threshold).astype(np.int8),
                         average="macro", zero_division=0)
        if score > best_global_f1:
            best_global_f1, best_global = score, float(threshold)

    columns = np.full(proba.shape[1], best_global, dtype=float)
    for col in range(proba.shape[1]):
        truth = y[:, col]
        if int(truth.sum()) == 0:
            continue
        best_t, best_f1 = best_global, -1.0
        for threshold in grid:
            score = f1_score(truth, (proba[:, col] >= threshold).astype(np.int8),
                             zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, float(threshold)
        columns[col] = best_t
    return best_global, columns


def _case(seed, rows=400, labels=12, rate=0.15):
    rng = np.random.default_rng(seed)
    y = (rng.random((rows, labels)) < rate).astype(np.int8)
    # Signal plus noise, so the argmax cut differs per label instead of landing on one
    # value for all of them — a sweep that got the columns wrong would still pass then.
    proba = np.clip(rng.random((rows, labels)) * 0.7 + y * rng.random((rows, labels)) * 0.6,
                    0, 1).astype(np.float32)
    return y, proba


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_sweep_picks_the_cuts_the_slow_search_picks(seed):
    """A refactor for speed must not move a single threshold: the cuts decide what a
    model asserts, and a bundle trained before it has to keep behaving like one trained
    after."""
    y, proba = _case(seed)

    got_global, got_columns = thresholds.tune_threshold_columns(y, proba, per_label=True)
    want_global, want_columns = _naive_search(y, proba, thresholds._DEFAULT_GRID)

    assert got_global == pytest.approx(want_global)
    assert got_columns == pytest.approx(want_columns)


@pytest.mark.parametrize("y_builder", [
    pytest.param(lambda shape: np.zeros(shape, np.int8), id="no-positives-anywhere"),
    pytest.param(lambda shape: np.ones(shape, np.int8), id="positives-everywhere"),
])
def test_the_sweep_matches_the_slow_search_on_the_degenerate_targets(y_builder):
    """Empty and full columns are where a counts-based formula divides by zero and a
    loop over sklearn calls simply returns 0.0."""
    y = y_builder((60, 4))
    proba = np.linspace(0, 1, 240, dtype=np.float32).reshape(60, 4)

    got_global, got_columns = thresholds.tune_threshold_columns(y, proba, per_label=True)
    want_global, want_columns = _naive_search(y, proba, thresholds._DEFAULT_GRID)

    assert got_global == pytest.approx(want_global)
    assert got_columns == pytest.approx(want_columns)


def test_a_probability_exactly_on_a_cut_is_asserted():
    """`>=`, not `>`, all the way through — `apply_thresholds` serves it that way, so a
    sweep that counted `> t` would tune against a rule serving does not apply."""
    y = np.array([[1], [0]], dtype=np.int8)
    proba = np.array([[0.60], [0.55]], dtype=np.float32)

    # At 0.60 the first row is asserted and the second is not: a perfect split.
    assert thresholds.tune_threshold_columns(y, proba, per_label=True)[1][0] == 0.60


def test_the_search_does_not_scan_the_matrix_once_per_cut():
    """19 cuts x (a macro F1 over the whole matrix + one f1_score per label) is a Python
    call per label per cut over a full column each time. Measured before this test
    existed: 4.5 s at 25,000 x 48 and 33.8 s at 156,373 x 60 — and with
    `select_c_on_tuned_thresholds` (the default) that whole search runs once per C
    candidate, i.e. ~100 s of a 40-minute run at the README's anchor shape.

    The sweep counts instead of comparing: one `searchsorted` per column bins its
    probabilities into the 19 cuts, and two `bincount`s give true and predicted positives
    for every cut at once. The budget is generous — the point is the shape of the curve,
    not the hardware."""
    y, proba = _case(7, rows=25_000, labels=48, rate=0.04)

    start = time.perf_counter()
    thresholds.tune_threshold_columns(y, proba, per_label=True)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.5, f"the 25,000 x 48 sweep took {elapsed:.2f} s"
