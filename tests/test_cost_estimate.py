"""What a training run of a given size will cost — the estimate the UI shows before
anyone spends an afternoon on it."""

import pytest

from app.profiles import ANCHOR_MINUTES, ANCHOR_ROWS, estimated_minutes


def test_the_estimate_reproduces_the_run_it_is_anchored_on():
    """The model has exactly one measurement under it: faecher_300k_auto, 156 373 rows,
    `auto`, 40.2 min wall-clock. If the estimate does not return that at that size, the
    anchor and the arithmetic have drifted apart and every other number is decoration.
    """
    assert estimated_minutes("auto", ANCHOR_ROWS) == pytest.approx(ANCHOR_MINUTES)


def test_rows_scale_linearly_because_that_is_what_was_measured():
    """`scripts/benchmark_row_scaling.py` measured exponent 1.00 for memory and
    vectorization, so doubling the rows doubles the estimate. Anything else here would
    be a curve nobody measured."""
    assert estimated_minutes("auto", 2 * ANCHOR_ROWS) == pytest.approx(2 * ANCHOR_MINUTES)
    assert estimated_minutes("auto", ANCHOR_ROWS // 2) == pytest.approx(ANCHOR_MINUTES / 2)


def test_the_profiles_keep_the_order_their_names_promise():
    """A profile name is meant to be a truthful price tag. `fast` under `auto` under
    `best` is the whole point of the dial; an estimate that inverted it would sell the
    expensive rung as the cheap one."""
    rows = 50_000
    assert (estimated_minutes("fast", rows)
            < estimated_minutes("auto", rows)
            < estimated_minutes("best", rows))


def test_an_unknown_profile_gets_no_estimate_rather_than_a_made_up_one():
    """A profile added to config.yaml has no measured cost until someone measures it.
    Returning `auto`'s number for it would put a figure on the screen that nothing
    supports."""
    assert estimated_minutes("experimental", 10_000) is None
    assert estimated_minutes("auto", 0) == 0.0
