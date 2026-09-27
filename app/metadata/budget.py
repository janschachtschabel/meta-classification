"""Fit extracted sentences into a character budget, and polish a title.

Ported from `static-metadata-generators` (`metagen/budget.py`), without the MMR selector: that
one ranks sentences by embedding similarity, which is the part of the source app's default
description method this port deliberately leaves out.
"""

from __future__ import annotations

from collections.abc import Sequence

_CLAUSE_BREAKS = (", ", "; ", ": ", " – ", " - ")
_QUOTES = "\"'„“”‚‘’»«"


def shorten(text: str, max_chars: int) -> str:
    """Cut at the last clause boundary (or word) so that the result plus "…" fits max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    window = text[: max_chars - 1]
    cut = max(window.rfind(b) for b in _CLAUSE_BREAKS)
    if cut < int(max_chars * 0.4):
        cut = window.rfind(" ")
    if cut <= 0:
        cut = len(window)
    return window[:cut].rstrip(" ,;:–-") + "…"


def clean_title(text: str, max_chars: int) -> str:
    title = " ".join(text.split())
    if len(title) > 1 and title[0] in _QUOTES and title[-1] in _QUOTES:
        title = title[1:-1].strip()
    return shorten(title.rstrip(".:;,"), max_chars)


def select_sentences(sentences: list[str], scores: Sequence[float], max_chars: int) -> str:
    """Greedily take the best-scored sentences that still fit; output keeps document order."""
    if not sentences:
        return ""
    order = sorted(range(len(sentences)), key=lambda i: -scores[i])  # stable: ties keep position
    chosen: list[int] = []
    total = 0
    for i in order:
        extra = len(sentences[i]) + (1 if chosen else 0)
        if total + extra <= max_chars:
            chosen.append(i)
            total += extra
    if not chosen:
        return shorten(sentences[order[0]], max_chars)
    return " ".join(sentences[i] for i in sorted(chosen))
