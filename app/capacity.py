"""What this server grants a training run right now.

Split out of ``profiles.py`` (audit ARC-4), which at 332 lines carried this, the cost
model and the config parsing — three things that change for three different reasons. This
is the piece that reads ``Settings``, so keeping it apart is what lets the cost model stay
a stdlib-only leaf. It reads the cost model; the cost model does not know it exists.

(The audit also said read-only dataset inspection therefore depended transitively on
``Settings``. It did not: that import was already behind ``TYPE_CHECKING``, and a fresh
interpreter importing ``dataset_stats`` never loaded ``app.settings``. The line count and
the three responsibilities are the reason for this split, not a broken dependency.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .memory import rss_bytes, threads_within
from .profile_costs import (
    _MATRIX_BYTES_PER_ROW,
    _TEXT_BYTES_PER_ROW,
    _WORD_ONLY_MATRIX_BYTES_PER_ROW,
    CHILD_PROCESS_BASE_BYTES,
)

if TYPE_CHECKING:
    from .profiles import Profile
    from .settings import Settings


@dataclass(frozen=True)
class CapacityPlan:
    """What this server grants a run right now: the CPU budget's threads, the memory
    budget, and what the process already holds.

    Predicts the deploy fit's thread count — the run's largest matrix, so its fewest
    threads — with the same arithmetic ``memory.ThreadBudget`` applies during the run,
    on per-row sizes measured instead of matrices that do not exist yet.
    """

    requested_threads: int
    budget_bytes: int | None
    held_bytes: int
    use_char: dict[str, bool]  # per profile name; word-only builds a fifth of the matrix

    @classmethod
    def for_server(cls, settings: Settings, profiles: dict[str, Profile]) -> CapacityPlan:
        """The plan for a run started now under ``settings``.

        Call it before loading anything of your own: what a run starts from is what this
        process holds now, not an analysis' transient copy of the dataset. A run in a
        child process starts a second interpreter as well, and its budget counts this
        process too.
        """
        held = rss_bytes()
        if settings.training_isolation == "process":
            held += CHILD_PROCESS_BASE_BYTES
        return cls(requested_threads=settings.effective_n_jobs(),
                   budget_bytes=settings.effective_train_memory_bytes(),
                   held_bytes=held,
                   use_char={name: profile.use_char for name, profile in profiles.items()})

    def head_fit_threads(self, profile_name: str, n_rows: int) -> int:
        per_row = (_MATRIX_BYTES_PER_ROW if self.use_char.get(profile_name, True)
                   else _WORD_ONLY_MATRIX_BYTES_PER_ROW)
        matrix = n_rows * per_row
        held = self.held_bytes + n_rows * _TEXT_BYTES_PER_ROW + matrix
        return threads_within(self.requested_threads, self.budget_bytes, held_bytes=held,
                              matrix_bytes=matrix)
