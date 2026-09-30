"""Fit extracted sentences into a character budget, and polish a title.

Ported from `static-metadata-generators` (`metagen/budget.py`), without the MMR selector: that
one ranks sentences by embedding similarity, which is the part of the source app's default
description method this port deliberately leaves out.
"""

from __future__ import annotations

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


# Sentence openers that point back at the sentence before: personal and demonstrative pronouns,
# pronominal adverbs, and connectives that presuppose what came first. Not the articles --
# "Das kleinste gemeinsame Vielfache ..." stands on its own, and that is how "Das" mostly opens.
_BACK_REFERENCES = frozenset({
    "er", "sie", "es", "ihm", "ihn", "ihr", "ihre", "ihnen", "sein", "seine", "dessen", "deren",
    "dies", "diese", "dieser", "dieses", "diesen", "diesem", "jene", "jener", "jenes",
    "damit", "dabei", "dadurch", "dafür", "dagegen", "daher", "darum", "deshalb", "deswegen",
    "davon", "darauf", "daraus", "darin", "dazu", "danach", "davor", "dort", "somit", "folglich",
    "außerdem", "zudem", "trotzdem", "dennoch", "allerdings", "jedoch",
})


def _points_back(sentence: str) -> bool:
    opener = sentence.lstrip(_QUOTES + " ").split(maxsplit=1)
    return bool(opener) and opener[0].rstrip(",:;").lower() in _BACK_REFERENCES


def lead(sentences: list[str], max_chars: int) -> str:
    """The first sentences, in order, as many as fit.

    A sentence too long for what is left of the budget is skipped and later ones may still
    come, as in the source app, whose quality numbers are those of that behaviour (the parity
    fixture holds two descriptions built that way). But once one was skipped, a sentence that
    opens by pointing back ("Sie", "Dies", "Damit") is not taken: its reference would be the
    skipped sentence (audit 2026-09-30, M06). When not even the first fits, it is shortened at
    a clause boundary.
    """
    if not sentences:
        return ""
    chosen: list[str] = []
    total = 0
    skipped = False
    for sentence in sentences:
        extra = len(sentence) + (1 if chosen else 0)
        if total + extra > max_chars or (skipped and _points_back(sentence)):
            skipped = True
            continue
        chosen.append(sentence)
        total += extra
    return " ".join(chosen) if chosen else shorten(sentences[0], max_chars)
