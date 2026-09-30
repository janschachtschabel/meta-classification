"""`app/metadata/textprep`: undoing wrapped lines in linear time, with unchanged results.

M01 (audit 2026-09-30): joining a wrapped line searched the WHOLE text joined so far for a
line-final hyphen and copied it once per line. Lines that start in lower case -- vocabulary
lists, "der ... und die ..." -- all join, so the cost grew with the square of the length:
10k characters 0.04 s, 100k 4.0 s (measured before the fix), and a request at the documented
caps came to 13-60 CPU-minutes. No scaling test reached this path.
"""

import random
import re
import statistics
import time

from app.metadata import textprep


def _wrapped(chars: int) -> str:
    """Distinct lines that all start in lower case, so every one continues the one before
    (identical lines would be dropped as repeats)."""
    lines, size = [], 0
    while size < chars:
        lines.append(f"der Begriff {len(lines)} und die Definition im Unterricht")
        size += len(lines[-1]) + 1
    return "\n".join(lines)


def test_joining_wrapped_lines_takes_linear_time():
    """A ratio, not a wall-clock budget (see test_metadata_sentences): eight times the input
    takes ~8x when linear and ~64x when quadratic."""

    def best_of_three(text: str) -> float:
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            textprep.clean_text(text)
            timings.append(time.perf_counter() - started)
        return min(timings)

    ratio = best_of_three(_wrapped(100_000)) / best_of_three(_wrapped(12_500))

    assert ratio < 14, f"8x the input took {ratio:.1f}x the time"


class _SearchSpy:
    def __init__(self, pattern: re.Pattern) -> None:
        self.pattern = pattern
        self.lengths: list[int] = []

    def search(self, text: str):
        self.lengths.append(len(text))
        return self.pattern.search(text)


def test_the_hyphen_check_never_reads_more_than_one_line(monkeypatch):
    """The root cause, deterministic: whatever has been joined, a line-final hyphen is a
    question about the END of it."""
    spy = _SearchSpy(textprep._WRAP_HYPHEN_RE)
    monkeypatch.setattr(textprep, "_WRAP_HYPHEN_RE", spy)
    text = _wrapped(20_000)

    textprep.clean_text(text)

    assert spy.lengths, "the hyphen check did not run"
    assert max(spy.lengths) <= max(len(line) for line in text.splitlines())


# The joiner as it was before M01, verbatim, as the reference the fix must agree with.
def _reference_join(joined: str, line: str) -> str:
    if textprep._WRAP_HYPHEN_RE.search(joined):
        next_word = line.split(maxsplit=1)[0].rstrip(".,")
        if line[0].islower() and next_word not in textprep._SUSPENDED_HYPHEN_NEXT:
            return joined[:-1] + line
        if line[0].isupper():
            return joined + line
    return f"{joined} {line}"


def _reference_join_wrapped_lines(lines: list[str]) -> list[str]:
    wrapped = textprep._is_hard_wrapped(lines)
    widths = [len(line) for line in lines if len(line) >= 20]
    full_width = 0.8 * statistics.median(widths) if widths else float("inf")
    joined: list[str] = []
    last = ""
    for line in lines:
        if (
            joined and last and line and not textprep.is_list_item(line)
            and not textprep._SENTENCE_END_RE.search(last)
            and (textprep._breaks_sentence(last, line) or (wrapped and len(last) >= full_width))
        ):
            joined[-1] = _reference_join(joined[-1], line)
        else:
            joined.append(line)
        last = line
    return joined


_WORDS = ["und", "oder", "bis", "Wald", "wasser", "Lern-", "Er-", "lebnis", "Nord-", "Süd-Dialog",
          "die", "Katze.", "Ende!", "Frage?", "Komma,", "x", "Übung", "über", "- Punkt", "1. Punkt",
          "a) klein", "Satz", "-", "é-", "3-"]


def _random_lines(rng: random.Random) -> list[str]:
    lines = []
    for _ in range(rng.randint(0, 14)):
        if rng.random() < 0.15:
            lines.append("")
            continue
        words = [rng.choice(_WORDS) for _ in range(rng.randint(1, 9))]
        lines.append(" ".join(words))
    return lines


def test_joining_gives_exactly_what_it_gave_before():
    """Behaviour-preserving: 20,000 generated line lists -- hyphens, suspended compounds,
    commas, sentence ends, list items, blank lines -- join exactly as before M01."""
    rng = random.Random(20260930)
    for _ in range(20_000):
        lines = _random_lines(rng)
        assert textprep.join_wrapped_lines(lines) == _reference_join_wrapped_lines(lines), lines
