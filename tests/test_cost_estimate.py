"""What a training run of a given size will cost — the estimate the UI shows before
anyone spends an afternoon on it."""

import pytest

from app.profiles import (
    ANCHOR_MINUTES,
    ANCHOR_ROWS,
    ANCHOR_THREADS,
    CapacityPlan,
    estimated_minutes,
    head_fit_seconds_ratio,
)

MiB = 1024 * 1024


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


def test_the_anchor_thread_count_reproduces_the_anchor_run():
    """The anchor ran on 9 threads; telling the model so must not move its one number."""
    assert estimated_minutes("auto", ANCHOR_ROWS, threads=ANCHOR_THREADS) == pytest.approx(
        ANCHOR_MINUTES)


def test_fewer_head_fit_threads_take_longer_but_do_not_divide_the_time():
    """A run the memory budget holds to 3 threads is slower than on 9 — but not 3x
    slower: vectorization does not use the threads at all, and the fits themselves
    scale far from linearly."""
    rows = 300_000
    nine = estimated_minutes("auto", rows, threads=9)
    three = estimated_minutes("auto", rows, threads=3)
    one = estimated_minutes("auto", rows, threads=1)
    assert nine < three < one
    assert three < 3 * nine


@pytest.mark.parametrize(("measured", "where"), [(60.4 / 19.2, "30k rows"),
                                                 (387.0 / 120.5, "100k rows")])
def test_the_thread_curve_reproduces_the_measured_fits(measured, where):
    """benchmark_training_memory.py, 2026-09-11: one thread against six, the same fit.
    6 threads were 3.1-3.2x faster than 1, nowhere near 6x — the curve the estimate uses
    has to give back what was measured, like the anchor has to give back its 40.2 min."""
    assert head_fit_seconds_ratio(1, 6) == pytest.approx(measured, rel=0.05), where


def test_without_a_memory_budget_a_run_gets_every_thread_the_cpu_grants():
    plan = CapacityPlan(requested_threads=9, budget_bytes=None, held_bytes=300 * MiB,
                        use_char={"auto": True})
    assert plan.head_fit_threads("auto", 1_000_000) == 9


def test_the_8_gb_container_holds_a_300k_run_to_fewer_threads():
    """The owner's test container: 6000 MB budget. At 300k rows the word+char deploy
    matrix is ~0.9 GB, so each concurrent fit needs ~2.2 GB — one fits next to the
    texts and the matrix, nine would not. Word-only builds a fifth of the matrix."""
    plan = CapacityPlan(requested_threads=9, budget_bytes=6000 * MiB, held_bytes=300 * MiB,
                        use_char={"auto": True, "fast": False})
    char_threads = plan.head_fit_threads("auto", 300_000)
    word_threads = plan.head_fit_threads("fast", 300_000)
    assert 1 <= char_threads < word_threads <= 9
    assert plan.head_fit_threads("auto", 10_000) == 9, "a small run is not held back"


def test_a_run_that_cannot_fit_at_all_still_gets_one_thread():
    plan = CapacityPlan(requested_threads=9, budget_bytes=100 * MiB, held_bytes=300 * MiB,
                        use_char={"auto": True})
    assert plan.head_fit_threads("auto", 300_000) == 1


def test_a_run_in_a_child_process_plans_with_a_second_interpreter(monkeypatch):
    """In process mode a run starts its own interpreter with numpy, scipy, scikit-learn
    and pandas, and its budget counts this process too: it starts from the API's memory
    PLUS a fresh interpreter's. Planned from the API's alone, a run near a boundary was
    promised a thread its budget then withheld. 215k rows sit right at such a boundary."""
    from app import profiles as profiles_mod
    from app import settings as settings_mod
    from app.profiles import CHILD_PROCESS_BASE_BYTES, Profile
    from app.settings import Settings

    monkeypatch.setattr(profiles_mod, "rss_bytes", lambda: 300 * MiB)
    monkeypatch.setattr(settings_mod, "available_cpus", lambda: 16)
    profiles = {"auto": Profile("auto", use_char=True), "fast": Profile("fast", use_char=False)}

    def plan(isolation: str) -> CapacityPlan:
        return CapacityPlan.for_server(Settings(n_jobs=9, cpu_max_percent=100, train_memory_mb=6000,
                                                training_isolation=isolation), profiles)

    thread, process = plan("thread"), plan("process")
    assert thread.held_bytes == 300 * MiB
    assert process.held_bytes == 300 * MiB + CHILD_PROCESS_BASE_BYTES
    assert (thread.requested_threads, thread.budget_bytes) == (9, 6000 * MiB)
    assert thread.use_char == {"auto": True, "fast": False}
    assert (thread.head_fit_threads("auto", 215_000), process.head_fit_threads("auto", 215_000)) == (3, 2)
