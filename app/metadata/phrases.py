"""Keyword candidates from German text, and the score that ranks them.

Ported from `static-metadata-generators` (`metagen/candidates.py` and the TF-IDF part of
`metagen/methods/kw_statistical.py`). The candidate finder is the heuristic one — capitalised
words extended left over nouns and adjectives — not the spaCy one: the source app measured
both, and the spaCy variants scored lower (T2 0,6 against 1,1) while needing a tagger, a
parser and a 40 MB model.

The score is TF-IDF in the literal sense, with German word frequencies standing in for a
document collection: how often a phrase occurs in THIS text, against how rare its words are
in German at large.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

from wordfreq import zipf_frequency

from .textprep import GENERIC_TERMS, MAX_TOKEN_CHARS, is_stopword, is_task_verb, stem, tokenize
from .types import Document

# Attributive adjectives carry a derivational suffix plus an inflection ending ("chemische").
_ADJECTIVE_RE = re.compile(
    r"(?:isch|lich|ig|al|iv|ar|bar|los|voll|haft|sam|ell|ös|end)(?:e|en|er|es|em)$", re.IGNORECASE
)
_STOP, _CAP, _ADJ, _LOW = "stop", "cap", "adj", "low"
# Punctuation between two words ends a phrase: "Mathematik, Physik, Chemie" holds three
# candidates, not the keyword "Mathematik Physik Chemie", which the text does not contain
# (audit 2026-09-30, M06). None of these can occur inside a token, so splitting at them leaves
# every token as it was.
_PHRASE_BREAK_RE = re.compile(r"[,;:()\[\]{}„“”\"»«–—/]")
_EDGE_PUNCTUATION = " .,;:!?\"'()[]{}„“”»«–-"


@dataclass
class Candidate:
    text: str  # most frequent surface form
    key: tuple[str, ...]  # stems of the tokens
    count: int
    first_pos: int  # index of the first sentence containing the candidate


def _tag(token: str, index: int, lowercase_forms: set[str]) -> str:
    # A fragment longer than any word ends a phrase like a stopword: never part of a keyword,
    # never in the caches behind stem() and _idf() (audit 2026-09-30, M04).
    if len(token) > MAX_TOKEN_CHARS or is_stopword(token):
        return _STOP
    if token[0].isupper():
        # A capitalised sentence opener that also occurs in lower case is no noun ("Besonders").
        if (index == 0 and token.lower() in lowercase_forms) or is_task_verb(token):
            return _LOW
        return _CAP
    return _ADJ if _ADJECTIVE_RE.search(token) else _LOW


def _phrases(tokens: list[str], tags: list[str], max_len: int) -> Iterator[list[str]]:
    """Token spans ending in a capitalised word, extended left over nouns and adjectives."""
    for end, tag in enumerate(tags):
        if tag != _CAP:
            continue
        for start in range(end, max(end - max_len, -1), -1):
            if start < end and tags[start] not in (_CAP, _ADJ):
                break
            yield tokens[start : end + 1]


def _keep(words: list[str], count: int) -> bool:
    if len(words) == 1:
        return len(words[0]) >= 3 and words[0].lower() not in GENERIC_TERMS
    first = words[0]
    # Capitalised word pairs are often accidental ("Pflanzen Energie" in a heading); keep them
    # only when repeated or when the first word is an adjective ("Französische Revolution").
    return first[0].islower() or count >= 2 or bool(_ADJECTIVE_RE.search(first))


def _tagged_sentences(doc: Document) -> list[list[tuple[list[str], list[str]]]]:
    """Per sentence, its stretches between punctuation marks as (tokens, word-class tags)
    (cached per document). A tag is decided by the token's place in the whole SENTENCE --
    a capitalised sentence opener is judged as one -- while a phrase stays inside one stretch.
    """
    if "tagged_sentences" not in doc.cache:
        sentence_parts = [[tokenize(part) for part in _PHRASE_BREAK_RE.split(s)] for s in doc.sentences]
        lowercase_forms = {t for parts in sentence_parts for tokens in parts for t in tokens
                           if t[0].islower()}
        tagged = []
        for parts in sentence_parts:
            offset, stretches = 0, []
            for tokens in parts:
                stretches.append((tokens, [_tag(t, offset + i, lowercase_forms)
                                           for i, t in enumerate(tokens)]))
                offset += len(tokens)
            tagged.append(stretches)
        doc.cache["tagged_sentences"] = tagged
    return doc.cache["tagged_sentences"]


def _collect(occurrences: Iterable[tuple[tuple[str, ...], str, int]]) -> list[Candidate]:
    """Merge (stem key, surface, sentence index) occurrences; the most frequent surface wins."""
    counts: Counter[tuple[str, ...]] = Counter()
    surfaces: dict[tuple[str, ...], Counter[str]] = {}
    first_pos: dict[tuple[str, ...], int] = {}
    for key, surface, pos in occurrences:
        counts[key] += 1
        surfaces.setdefault(key, Counter())[surface] += 1
        first_pos.setdefault(key, pos)
    return [
        Candidate(
            min(surfaces[key].items(), key=lambda item: (-item[1], len(item[0])))[0],
            key, count, first_pos[key],
        )
        for key, count in counts.items()
    ]


def heuristic_candidates(doc: Document, max_len: int = 3) -> list[Candidate]:
    """Noun-phrase candidates (cached per document), merged by stems, in order of appearance."""
    cache_key = ("heuristic_candidates", max_len)
    if cache_key not in doc.cache:
        occurrences = (
            (tuple(stem(t) for t in phrase), " ".join(phrase), pos)
            for pos, stretches in enumerate(_tagged_sentences(doc))
            for tokens, tags in stretches
            for phrase in _phrases(tokens, tags, max_len)
        )
        doc.cache[cache_key] = [
            c for c in _collect(occurrences) if _keep(c.text.split(" "), c.count)
        ]
    return doc.cache[cache_key]


# wordfreq loads its German table on first use behind an lru_cache, which is thread-safe but
# not single-flight: eight simultaneous first requests each read it -- 3.3 s and a 554 MB peak
# against 0.3 s and 101 MB for one load (audit 2026-09-30, M07). The first use is serialised;
# after it, the flag is all anyone reads.
_table_lock = threading.Lock()
_table_loaded = False


def _load_table_once() -> None:
    global _table_loaded
    if _table_loaded:
        return
    with _table_lock:
        if not _table_loaded:
            zipf_frequency("und", "de")
            _table_loaded = True


@lru_cache(maxsize=100_000)
def _idf(token: str) -> float:
    _load_table_once()
    # Zipf 7 ≈ "der", 3 ≈ rare word, 0 = unknown word: rare words weigh more, never below 0.5.
    return max(0.5, 7.5 - zipf_frequency(token, "de"))


def score_candidates(candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
    """TF-IDF-like score per candidate (frequency in text x rarity in German), best first."""
    scored = []
    for c in candidates:
        words = c.text.split(" ")
        idf = sum(_idf(w) for w in words) / len(words)
        score = (1 + math.log(c.count)) * idf * (1 + 0.15 * (len(c.key) - 1))
        if c.first_pos < 3:
            score *= 1.2
        scored.append((c, score))
    return sorted(scored, key=lambda item: -item[1])


def scored_tfidf(doc: Document) -> list[tuple[Candidate, float]]:
    """All heuristic candidates with their TF-IDF-like score (cached per document).

    Cached because the title method ranks the same candidates the keyword method does: a text
    with no heading builds its title from the three best keywords, so without this the whole
    ranking would run twice per document.
    """
    if "tfidf" not in doc.cache:
        doc.cache["tfidf"] = score_candidates(heuristic_candidates(doc))
    return doc.cache["tfidf"]


def keyword_key(keyword: str) -> frozenset[str]:
    """Stems of the content words: keywords with equal keys are the same keyword."""
    words = tokenize(keyword)
    return frozenset(stem(w) for w in words if not is_stopword(w)) or frozenset(map(stem, words))


def finalize_keywords(items: list[str], n: int) -> list[str]:
    """Keep ranked keywords unless their stems repeat an earlier one; capitalise; cut to n."""
    chosen: list[str] = []
    chosen_keys: list[frozenset[str]] = []
    for item in items:
        surface = " ".join(item.split()).strip(_EDGE_PUNCTUATION)
        key = keyword_key(surface)
        if not key:
            continue
        if any(key <= other for other in chosen_keys):
            continue
        chosen.append(surface[0].upper() + surface[1:])
        chosen_keys.append(key)
        if len(chosen) == n:
            break
    return chosen
