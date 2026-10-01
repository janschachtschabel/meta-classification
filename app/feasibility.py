"""Whether a training run fits its memory before anything is fitted, and the refusal if not.

Split out of :mod:`app.deploy` (audit 2026-09-27, M-2). That module's docstring kept it whole
on the grounds that this seam was 35 lines; the check had since grown to 85 — weighing the
head, then the whole run, then the container's own limit — and was a fifth of the module
with a reason to change of its own (the memory model) and no use of deploy's internals.
``deploy`` still calls it at the one point it matters: just before the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import TrainingInputError
from .memory import (
    FIT_COPIES_PER_THREAD,
    MiB,
    head_bytes,
    held_bytes,
    matrix_bytes,
    memory_limit_bytes,
)


@dataclass(frozen=True)
class ProjectedMatrix:
    """A matrix not built yet, as the gate below weighs one: a shape, and what it will hold
    (``memory.matrix_bytes`` reads ``nbytes`` where there are no sparse arrays)."""

    shape: tuple[int, int]
    nbytes: int


def projected(matrix: Any, rows: int) -> ProjectedMatrix:
    """``matrix`` grown to ``rows`` rows at its own width and bytes per row.

    How the first selection fit weighs the deploy fit that comes last: the same texts'
    vocabulary over more rows. A wider vocabulary would come on top, so this errs towards
    letting a run start -- the deploy fit is weighed again, for real, when it comes.
    """
    per_row = matrix_bytes(matrix) / max(1, matrix.shape[0])
    return ProjectedMatrix((rows, matrix.shape[1]), int(per_row * rows))


def refuse_if_the_run_cannot_fit(
    *, n_labels: int, targets_bytes: int, matrix: Any, budget_bytes: int | None,
    oof_bytes: int = 0,
) -> None:
    """Stop a run that cannot fit its budget even one fit at a time.

    Three things are held together once fitting starts, and the run needs all of them:
    the head (``labels x features`` float32 coefficients — nothing releases it, it IS
    the model), the dense targets (``rows x labels``), and the input matrix plus the
    solver's copies of it. ``oof_bytes`` adds a fourth for the k-fold path: the
    out-of-fold probability buffers ``cross_val_evaluate`` allocates per C candidate
    before its first fit and holds until the winner is picked. Judged at ONE thread and
    ONE head — the floor, below which
    nothing can be traded away: above it the thread budget is doing its job, and this
    gate would be taking runs that work. So it is deliberately optimistic in one place:
    ``select_c`` keeps the best candidate's head while fitting the next one, so the
    holdout path with a multi-value C grid really holds two at once. Counting that would
    refuse runs a k-fold profile completes comfortably, since ``cross_val_evaluate``
    releases each fold's head (``tuning.py``, ``del head``).

    Weighing the head alone was not enough — at 200 000 features a 6 000 MB budget is
    only exceeded past ~7 500 labels, while a run of 4 000 labels over 250 000 rows is
    already impossible once its targets and matrix are counted. Nor was leaving the
    buffers out: at 250 000 rows x 300 labels x 3 candidates they are 858 MB against
    72 MB of targets, i.e. the largest term in the wide-label regime this gate exists
    for. Left to run, such a job
    is killed mid-fit and reaches the operator as `exit code -9` with a peak from
    whenever the last progress update landed.

    Weighed against two different things, because they mean different things. The
    training BUDGET is what the operator granted this run; exceeding it is a tuning
    answer, and ``train_memory_mb=0`` withdraws the question. The container's cgroup
    LIMIT is what the kernel enforces, whatever anyone configured — so that one is
    weighed even with the budget switched off, and against what is already held, since
    that is the number the OOM killer compares too.

    Wired into ``ThreadBudget.before_fit``, which every fit passes with its own matrix:
    only the matrix knows how wide the vocabulary actually got, and estimating from the
    caps instead refuses runs whose vocabulary never approaches them. Outside a
    container there is neither a limit nor, usually, a budget — and nothing is refused,
    because no one has said what the machine may spend.
    """
    n_features = matrix.shape[1]
    head = head_bytes(n_labels, n_features)
    held_matrix = matrix_bytes(matrix)
    copies = int(FIT_COPIES_PER_THREAD * held_matrix)
    # The matrix itself plus what one fit copies of it: the run holds both at once.
    fit = held_matrix + copies
    needed = head + targets_bytes + fit + oof_bytes
    detail = (f"{n_labels:,} labels x {n_features:,} features need {head // MiB:,} MB of "
              f"coefficients, {targets_bytes // MiB:,} MB of targets and {fit // MiB:,} MB "
              f"for the matrix and one fit's copies of it")
    levers = ("Raise min_samples_per_label to train fewer labels (the usual cause is a "
              "free-text label column), lower max_word_features / max_char_features, or "
              "give the container more memory")
    if oof_bytes:
        detail += (f", plus {oof_bytes // MiB:,} MB of out-of-fold probabilities held for "
                   f"every C candidate at once")
        levers += (", and a k-fold run can shorten its C grid or take the holdout path "
                   "(cv_folds: 0), which keeps no such buffers")

    if budget_bytes is not None and needed > budget_bytes:
        raise TrainingInputError(
            f"This run cannot fit the memory it has, even one label at a time: {detail} "
            f"— {needed // MiB:,} MB against a training budget of "
            f"{budget_bytes // MiB:,} MB. {levers} (APIV3_TRAIN_MEMORY_MB)."
        )

    limit = memory_limit_bytes()
    if limit is None:
        return
    # Already held: this is the training process with its matrix and targets built, and
    # the API process when it shares the budget. The head and the fit's copies come on
    # top of that, so the sum is what the kernel would be asked for.
    # None where the platform gives no reading: a floor of zero, which only ever makes
    # this gate more permissive, and `memory.held_bytes` logs that it is flying blind.
    held = held_bytes() or 0
    if held + head + copies <= limit:
        return
    raise TrainingInputError(
        f"This run cannot fit the container, even one label at a time: {detail}, on top "
        f"of the {held // MiB:,} MB already held — more than the container's own limit "
        f"of {limit // MiB:,} MB, which the kernel enforces by killing the process. "
        f"{levers}."
    )
