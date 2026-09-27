"""Sentence splitting stays linear in the length of one line (audit 2026-09-27, S-2).

`split_sentences` already treats every line break as a hard boundary, and a text of many
short lines is linear for exactly that reason. Nothing bounded how long ONE line may be, and
pysbd's German abbreviation replacer runs a whole-string `re.sub` per abbreviation candidate,
so its cost grows with candidates × length. Measured before the fix, 25k -> 100k characters
of prose on one line: 0.23 s -> 2.94 s (x12.7). A /metadata request at the documented caps
(100 texts of 100,000 characters) came to roughly five CPU-minutes — readonly-reachable, in a
single-worker process whose GIL every prediction waits on.
"""

import itertools
import random
import time
from collections.abc import Iterable, Iterator

import pytest

from app.metadata import textprep

_WORDS = [
    "Pflanze", "Licht", "Energie", "Zelle", "Chlorophyll", "Wasser", "Kohlendioxid", "Zucker",
    "Sauerstoff", "Blatt", "Wurzel", "Stoffwechsel", "Reaktion", "Enzym", "Membran",
    "Photosynthese", "Atmung", "Nährstoff", "Wachstum",
]

# Well inside pysbd's linear zone (it stayed linear up to ~25k characters per call), so the
# test pins the PROPERTY — the input to pysbd is bounded — and leaves the code free to tune
# its own bound below this without touching the test.
_LINEAR_ZONE = 10_000


def _sentences(seed: int = 7) -> Iterator[str]:
    rng = random.Random(seed)
    while True:
        yield " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 14))).capitalize() + "."


def _line(sentences: Iterable[str], chars: int) -> str:
    picked, size = [], 0
    for sentence in sentences:
        if size >= chars:
            break
        picked.append(sentence)
        size += len(sentence) + 1
    return " ".join(picked)[:chars]


def _prose_line(chars: int) -> str:
    return _line(_sentences(), chars)


def _recurring_prose_line(chars: int) -> str:
    """Prose whose sentences recur from a pool of 100 — for timing.

    pysbd compiles one regex per distinct sentence (to find its span), and Python's `re`
    caches 512 compiled patterns. With every sentence distinct, the small input's patterns
    stayed cached from the second run of a best-of-three while the big input's could not, so
    the ratio compared a warm run with a cold one: 9-11x plain, 38-124x under coverage, whose
    tracer slows the pure-Python regex compiler. Clearing the cache before every run instead
    — cold against cold — hid the unbounded code under coverage (~10x). Recurring sentences
    keep every pattern cached on both sides, so the ratio measures the splitting alone.
    """
    return _line(itertools.cycle(itertools.islice(_sentences(), 100)), chars)


def _unpunctuated_line(chars: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(chars // 7))[:chars]


class _Spy:
    """The real segmenter, recording the length of every string it is handed."""

    def __init__(self, real) -> None:
        self.real = real
        self.lengths: list[int] = []

    def segment(self, text: str) -> list[str]:
        self.lengths.append(len(text))
        return self.real.segment(text)


@pytest.mark.parametrize("shape", [_prose_line, _unpunctuated_line])
def test_pysbd_is_never_handed_a_whole_long_line(shape, monkeypatch):
    """The mechanism, checked without a clock: what bounds the cost is the length of each call."""
    spy = _Spy(textprep._segmenter)
    monkeypatch.setattr(textprep, "_segmenter", spy)

    textprep.split_sentences(shape(100_000))

    assert max(spy.lengths) <= _LINEAR_ZONE, (
        f"pysbd was handed {max(spy.lengths)} characters in one call; its abbreviation pass "
        "is super-linear in that length"
    )


def test_bounding_a_long_line_does_not_change_its_sentences(monkeypatch):
    """Differential: the same function with the bound lifted is the reference. A line of
    ordinary prose over the bound must split into exactly the sentences it did before."""
    line = _prose_line(12_000)
    bounded = textprep.split_sentences(line)
    monkeypatch.setattr(textprep, "MAX_SEGMENT_CHARS", 10**9, raising=False)

    assert bounded == textprep.split_sentences(line)


@pytest.mark.parametrize("shape", [_recurring_prose_line, _unpunctuated_line])
def test_one_long_line_splits_in_linear_time(shape):
    """The scaling measurement CLAUDE.md asks of every pattern on this path — as a RATIO, not a
    wall-clock budget, so a slow machine slows both sides alike (a fixed budget is what made
    test_a_kill_ends_the_child_at_once fail under load). Eight times the input: linear work
    takes ~8x, quadratic ~64x. Measured plain and under coverage: 4-9x bounded, 33-51x with
    the bound lifted."""

    def best_of_three(text: str) -> float:
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            textprep.split_sentences(text)
            timings.append(time.perf_counter() - started)
        return min(timings)

    ratio = best_of_three(shape(100_000)) / best_of_three(shape(12_500))

    assert ratio < 14, f"8x the input took {ratio:.1f}x the time"
