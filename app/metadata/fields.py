"""The three generators, one per field.

These are the default combination of `static-metadata-generators`
(`metagen/recommended.py`) — the methods its two test sets picked out of 52, and the only ones
ported. Each is a few lines because the work is in the layers below: the sentences and
headings come from `textprep`, the ranked candidates from `phrases`, the budget from `budget`.

The source app's ids, for tracing a change back to its measurements:
`title.heading_or_keywords`, `description.lead`, `keywords.tfidf`.
"""

from __future__ import annotations

import re

from .budget import clean_title, select_sentences
from .phrases import finalize_keywords, scored_tfidf
from .textprep import (
    GENERIC_TERMS,
    description_candidates,
    is_boilerplate,
    is_task_instruction,
    letter_ratio,
    tokenize,
)
from .types import Document, MetadataSettings

_FORM_FIELD_RE = re.compile(r"^(?:Name|Datum|Klasse)\s*:", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$")
_WEB_ADDRESS_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)


def keywords(doc: Document, settings: MetadataSettings) -> list[str]:
    """`keywords.tfidf` — noun phrases by frequency in the text and rarity in German.

    The most reliable of the three fields: q 1,8 on the source app's first test set, and the
    best keyword method on its second. Uses only words that occur in the text, so it cannot
    invent a term.
    """
    return finalize_keywords([c.text for c, _ in scored_tfidf(doc)], settings.n_keywords)


def description(doc: Document, settings: MetadataSettings) -> str:
    """`description.lead` — the first usable sentences, up to the character budget.

    Sentences are scored by position, so "best first" means "earliest", and the budget fitting
    then takes as many as fit. `description_candidates` is what makes this more than a prefix:
    headings, list items, task instructions and page chrome are not candidates, so the
    description starts at the first real sentence.
    """
    sentences = description_candidates(doc)
    return select_sentences(
        sentences, [-float(i) for i in range(len(sentences))], settings.desc_max
    )


def _is_title_like(line: str) -> bool:
    tokens = tokenize(line)
    if not (5 <= len(line) <= 100 and 1 <= len(tokens) <= 14) or line.endswith((".", ",", ";", ":")):
        return False
    if letter_ratio(line) < 0.6 or is_boilerplate(line) or is_task_instruction(line):
        return False
    if _FORM_FIELD_RE.match(line) or _DATE_RE.match(line) or _WEB_ADDRESS_RE.match(line):
        return False
    # Worksheet headers like "Arbeitsblatt Geschichte – Klasse 8" consist of generic words.
    generic = sum(t.lower() in GENERIC_TERMS for t in tokens)
    return generic * 2 < len(tokens)


def _detected_heading(doc: Document) -> str | None:
    """First title-like line at the start of the text, or None (e.g. for plain paragraphs)."""
    return next((line for line in doc.lines[:5] if _is_title_like(line)), None)


def _keyphrase_title(phrases: list[str]) -> str:
    """Title template from ranked keyphrases: "K1: K2 und K3", "K1 und K2" or "K1"."""
    if len(phrases) >= 3:
        return f"{phrases[0]}: {phrases[1]} und {phrases[2]}"
    return " und ".join(phrases)


def title(doc: Document, settings: MetadataSettings) -> str:
    """`title.heading_or_keywords` — a real heading if the text opens with one, else a template.

    Both halves are needed. A genuine heading is the best title there is (sem 0,82), but
    running text has none, and heading extraction then collapses to the level of the first
    sentence (0,58); the keyword template holds up on both (0,76 / 0,70) and was the best
    title on the second test set. Falls back to the first sentence when the text yields no
    keywords either.
    """
    found = _detected_heading(doc)
    if found:
        return clean_title(found, settings.title_max)
    template = _keyphrase_title(
        finalize_keywords([c.text for c, _ in scored_tfidf(doc)], 3)
    )
    if template:
        return clean_title(template, settings.title_max)
    sentences = description_candidates(doc)
    return clean_title(sentences[0], settings.title_max) if sentences else ""
