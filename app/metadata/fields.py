"""The three generators, one per field.

These are the default combination of `static-metadata-generators`
(`metagen/recommended.py`) — the methods its two test sets picked out of 52, and the only ones
ported. Each is a few lines because the work is in the layers below: the sentences and
headings come from `textprep`, the ranked candidates from `phrases`, the budget from `budget`.

The source app's ids, for tracing a change back to its measurements:
`title.heading_or_keywords`, `description.lead`, `keywords.tfidf`. The title's second half is no
longer the source app's: its template "K1: K2 und K3" joined three rare, unrelated keywords
("Valerie: Katzen und Micki"); docs/plans/2026-10-26-metadata-titles.md has the replacement and
its measurements.
"""

from __future__ import annotations

import re
from collections import Counter

from .budget import clean_title, lead
from .phrases import finalize_counted, finalize_keywords, scored_tfidf
from .textprep import (
    GENERIC_TERMS,
    capitalised,
    description_candidates,
    is_boilerplate,
    is_task_instruction,
    letter_ratio,
    opens_in_lower_case,
    stem,
    tokenize,
)
from .types import Document, MetadataSettings

_FORM_FIELD_RE = re.compile(r"^(?:Name|Datum|Klasse)\s*:", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$")
_WEB_ADDRESS_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
# The subject of a definitional opening ("Der Satz des Pythagoras ist ...", "Die binäre Suche
# ist ..."): the article, lowercase adjectives, a noun (any word with a capital, so "pH-Wert"
# counts), an optional genitive attribute, then a lowercase word -- usually the verb. With its
# article, because only then is the adjective's inflection in the text the right one. From the
# external `textmeta` experiment (`_DEFINITION_SUBJECT`).
_DEFINITION_SUBJECT_RE = re.compile(
    r"^((?:Der|Die|Das)\s+(?:[a-zäöüß][\w-]*\s+)*?[\w-]*[A-ZÄÖÜ][\w-]*"
    r"(?:\s+(?:des|der)\s+(?:[a-zäöüß][\w-]*\s+)*?[\w-]*[A-ZÄÖÜ][\w-]*)?)\s+[a-zäöü]"
)
# The title keyword is the most frequent among this many of the best-ranked: the ranking favours
# rare words, which name details ("Komplexitätsklasse" for a text on binary search).
_TITLE_KEYWORD_POOL = 8


def keywords(doc: Document, settings: MetadataSettings) -> list[str]:
    """`keywords.tfidf` — noun phrases by frequency in the text and rarity in German.

    The most reliable of the three fields: q 1,8 on the source app's first test set, and the
    best keyword method on its second. Uses only words that occur in the text, so it cannot
    invent a term.
    """
    return finalize_keywords([c.text for c, _ in scored_tfidf(doc)], settings.n_keywords)


def description(doc: Document, settings: MetadataSettings) -> str:
    """`description.lead` — the first usable sentences, up to the character budget.

    As many as fit, in order; after one that does not, none that points back at it
    (`budget.lead`).
    `description_candidates` is what makes this more than a prefix: headings, list items, task
    instructions and page chrome are not candidates, so the description starts at the first
    real sentence.
    """
    return capitalised(lead(description_candidates(doc), settings.desc_max))


def _is_title_like(line: str) -> bool:
    tokens = tokenize(line)
    if not (5 <= len(line) <= 100 and 1 <= len(tokens) <= 14) or line.endswith((".", ",", ";", ":")):
        return False
    if letter_ratio(line) < 0.6 or is_boilerplate(line) or is_task_instruction(line):
        return False
    # A heading opens with a capital; "von Anna Schmidt" is a byline, "valerie und ihre
    # katzen" no title to hand on.
    if opens_in_lower_case(line):
        return False
    if _FORM_FIELD_RE.match(line) or _DATE_RE.match(line) or _WEB_ADDRESS_RE.match(line):
        return False
    # Worksheet headers like "Arbeitsblatt Geschichte – Klasse 8" consist of generic words.
    generic = sum(t.lower() in GENERIC_TERMS for t in tokens)
    return generic * 2 < len(tokens)


def _detected_heading(doc: Document) -> str | None:
    """First title-like line at the start of the text, or None (e.g. for plain paragraphs)."""
    return next((line for line in doc.lines[:5] if _is_title_like(line)), None)


def _stem_counts(doc: Document) -> Counter[str]:
    if "stem_counts" not in doc.cache:
        doc.cache["stem_counts"] = Counter(stem(token) for token in tokenize(doc.text))
    return doc.cache["stem_counts"]


def _definition_subject(doc: Document) -> str | None:
    """The subject of the first description sentence, if it is definitional and the text comes
    back to it: one of its nouns recurs. "Die Lehrkraft erklärt heute etwas." is no title, and
    neither is "Die Schülerinnen und Schüler lernen ...": a generic noun (the list the keywords
    leave out) recurs precisely because it is generic, so only a specific one counts."""
    sentences = description_candidates(doc)
    match = _DEFINITION_SUBJECT_RE.match(sentences[0]) if sentences else None
    if not match:
        return None
    subject = match.group(1)
    nouns = [token for token in tokenize(subject)[1:]
             if not opens_in_lower_case(token) and token.lower() not in GENERIC_TERMS]
    counts = _stem_counts(doc)
    return subject if any(counts[stem(noun)] >= 2 for noun in nouns) else None


def _title_keyword(doc: Document) -> str | None:
    """The most frequent of the best-ranked keywords, the better-ranked one on a tie."""
    ranked = finalize_counted([(c.text, c.count) for c, _ in scored_tfidf(doc)], _TITLE_KEYWORD_POOL)
    return max(ranked, key=lambda pair: pair[1])[0] if ranked else None


def title(doc: Document, settings: MetadataSettings) -> str:
    """A real heading if the text opens with one; else the subject of a definitional opening;
    else the text's most frequent strong keyword; else its first sentence.

    A genuine heading is the best title there is (sem 0,82 in the source app). Running text has
    none, and there the source app's template "K1: K2 und K3" read as three unrelated words; on
    the gold set the order here scores 0,38 against its 0,18 (token F1, running text) and 0,68
    against 0,57 with headings (docs/plans/2026-10-26-metadata-titles.md). Every word is the
    text's own; only a lowercase opener is capitalised.
    """
    found = _detected_heading(doc) or _definition_subject(doc) or _title_keyword(doc)
    if found:
        return capitalised(clean_title(found, settings.title_max))
    sentences = description_candidates(doc)
    return capitalised(clean_title(sentences[0], settings.title_max)) if sentences else ""
