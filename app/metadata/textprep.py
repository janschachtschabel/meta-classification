"""German text preparation: cleaning, sentence splitting, tokens, stopwords and stemming.

Ported from `static-metadata-generators` (`metagen/text.py`) with two substitutions, both
argued in `docs/plans/2026-09-26-descriptive-metadata.md`: the stemmer comes from
`snowballstemmer` rather than nltk, whose import pulls `urllib.request` and `ssl` into a
process that must not be able to fetch a URL; and pysbd is imported under a scoped warning
filter, because its 0.3.4 regexes raise a SyntaxWarning that this project's
warnings-as-errors gate would otherwise turn into an import failure.

This is a separate concern from `app.data.clean_text`, which normalises the text a model is
trained and predicted on. That one has to produce exactly the features the vectorizer saw;
this one has to recover the sentences and headings a human would read, which is why it
undoes PDF line wrapping and drops page chrome instead of flattening everything.

A little past the ~300-line guide since the sentence splitter gained its length bound
(audit 2026-09-27, S-2). If it grows again, the splitter is the seam — ``split_sentences``,
``_bounded`` and the segmenter, with a test file of their own already.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
import warnings
from functools import lru_cache
from pathlib import Path

import snowballstemmer

from ..markup import strip_markup_preserving_lines
from .types import Document

with warnings.catch_warnings():
    # pysbd 0.3.4 writes '\s' in plain strings, so importing it raises SyntaxWarning while its
    # bytecode is compiled — once, on a cold cache, which is the worst kind of intermittent.
    # Scoped to this import: the project's global "warnings are errors" gate stays armed, and
    # a SyntaxWarning in our own code still fails the suite.
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

_RESOURCES = Path(__file__).parent / "resources"


def _load_wordlist(name: str) -> frozenset[str]:
    lines = (_RESOURCES / name).read_text(encoding="utf-8").splitlines()
    return frozenset(w.strip().lower() for w in lines if w.strip() and not w.startswith("#"))


STOPWORDS = _load_wordlist("stopwords_de.txt")
GENERIC_TERMS = _load_wordlist("generic_terms_de.txt")

# A token starts with a letter; hyphenated compounds such as "Calvin-Zyklus" stay one token.
_TOKEN_RE = re.compile(r"[^\W\d_]\w*(?:-\w+)*")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'“”„»«)]?$")
_LIST_MARKER_RE = re.compile(r"^(?:[-–•*·▪►]|\d{1,2}[.)]|[a-zA-Z][.)])\s")
_WRAP_HYPHEN_RE = re.compile(r"[^\W\d_]-$")
# After a line-final hyphen these words mean a suspended compound: "Lern- und Lehrmaterial".
_SUSPENDED_HYPHEN_NEXT = frozenset({"und", "oder", "bzw", "sowie", "bis", "als"})

# Specific phrases that mark web/page chrome wherever they appear.
_BOILERPLATE_PHRASES = (
    "alle rechte vorbehalten",
    "zum inhalt springen",
    "zur navigation springen",
    "hier klicken",
    "klicken sie hier",
    "klicke hier",
    "verwendet cookies",
    "verwenden cookies",
    "cookie-einstellungen",
    "cookie-richtlinie",
    "datenschutzerklärung",
    "newsletter abonnieren",
    "teilen auf",
    "seite drucken",
    "lizenzinformationen",
    "javascript aktivieren",
    "©",
    "automatischer zoom",  # toolbar of the browser's PDF viewer, copied along with the text
    "lizenziert unter",  # licence note in the footer of OER materials
    # Licence notes of worksheet platforms (Tutory).
    "frei oder unlizenziert",
    "um die lizenz zu sehen",
    "angaben zu den urhebern",
    # Link markers (bpb).
    "interner link:",
    "externer link:",
    # Imprint of publications; generic labels such as "Druck:" or "Satz …:" also start content.
    "redaktionsschluss",
    "autorinnen und redaktion",
    "autoren des comics",
    "text des comics",
    "mit illustrationen von",
    "fachdidaktische unterstützung",
    "wiss. beratung",
    "gestaltung und satz",
    "bildnachweis",
    "fotonachweis",
)
# Reference lists: footnotes ("↑"), ISBNs, "Name, Vorname (2014): …" and access dates.
_REFERENCE_RE = re.compile(
    r"^(?:[-–•*]\s*)?↑"
    r"|\bISBN\b"
    r"|^(?:[-–•*]\s*)?[A-ZÄÖÜ][\w'-]+, [A-ZÄÖÜ][^():]{0,80}\((?:\d{4}|o\. ?J\.)[a-z]?\):"
    r"|\((?:Stand|Zugriff(?: am)?|abgerufen am):? \d{1,2}\.\s?\d{1,2}\.\s?\d{2,4}\)"
)
# Lines made only of these words (plus stopwords) are navigation menus.
_MENU_WORDS = frozenset({
    "barrierefreiheit", "cookie", "cookies", "datenschutz", "drucken", "faq", "hilfe", "home",
    "impressum", "inhaltsverzeichnis", "kontakt", "login", "anmelden", "menü", "navigation",
    "newsletter", "registrieren", "sitemap", "startseite", "suche", "suchen", "teilen", "übersicht",
    "weiter", "zurück",
})
# Sentence-initial imperatives of worksheet tasks; such sentences describe work, not content.
_TASK_VERBS = frozenset({
    "analysiere", "analysiert", "arbeite", "arbeitet", "bearbeite", "bearbeitet", "begründe",
    "begründet", "berechne", "berechnet", "beschreibe", "beschreibt", "bestimme", "diskutiere",
    "diskutiert", "erkläre", "erklärt", "erstelle", "erstellt", "ergänze", "ergänzt", "fasse",
    "fasst", "finde", "findet", "formuliere", "formuliert", "gestalte", "gestaltet", "interpretiere",
    "interpretiert", "lies", "lest", "löse", "löst", "markiere", "markiert", "nenne", "nennt",
    "notiere", "notiert", "ordne", "ordnet", "recherchiere", "recherchiert", "schreibe", "schreibt",
    "stelle", "überlege", "überlegt", "überprüfe", "übersetze", "unterstreiche", "vergleiche",
    "vergleicht", "zeichne", "zeichnet",
})

_segmenter = pysbd.Segmenter(language="de", clean=False)
_stemmer = snowballstemmer.stemmer("german")

# pysbd's German abbreviation pass runs one whole-string re.sub per abbreviation candidate,
# so a single call costs candidates x length: 25k -> 100k characters of prose on one line
# took 0.23 s -> 2.94 s, while the same text as short lines stayed linear (audit 2026-09-27,
# S-2). A line longer than this is handed over in pieces. Far above any real paragraph — the
# parity fixture's longest line is 648 characters — and deep inside the range where pysbd
# measured linear (up to ~25k per call).
MAX_SEGMENT_CHARS = 4_000
# Where a piece may end: sentence-final punctuation, an optional closing quote, whitespace.
_PIECE_END_RE = re.compile(r"[.!?][\"'“”„»«)]?\s")


def _bounded(line: str) -> list[str]:
    """``line`` in pieces of at most ``MAX_SEGMENT_CHARS``, cut where a sentence ends if possible.

    Cutting after the last sentence-final punctuation before the bound leaves pysbd's answer
    unchanged for ordinary prose; failing that, after the last space; failing even that, at the
    bound itself. A cut can land after an abbreviation ("z. B. ") and end a sentence pysbd
    would have continued — only on a line over the bound, where the alternative was minutes of
    CPU. Every cut advances by at least one character, so the loop always ends.
    """
    pieces: list[str] = []
    start = 0
    while len(line) - start > MAX_SEGMENT_CHARS:
        window = line[start : start + MAX_SEGMENT_CHARS]
        ends = [match.end() for match in _PIECE_END_RE.finditer(window)]
        cut = ends[-1] if ends else (window.rfind(" ") + 1 or MAX_SEGMENT_CHARS)
        pieces.append(window[:cut])
        start += cut
    pieces.append(line[start:])
    return pieces


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def is_stopword(word: str) -> bool:
    return word.lower() in STOPWORDS


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Snowball stem of the lower-cased word; umlauts are folded, so "Brüche" == "Bruch"."""
    return _stemmer.stemWord(word.lower())


def letter_ratio(s: str) -> float:
    visible = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in visible) / len(visible) if visible else 0.0


def is_boilerplate(s: str) -> bool:
    lowered = s.lower()
    if any(phrase in lowered for phrase in _BOILERPLATE_PHRASES) or _REFERENCE_RE.search(s):
        return True
    tokens = [t.lower() for t in tokenize(s)]
    # Some menu words ("startseite") are themselves in the stopword list, so test both sets.
    return (
        0 < len(tokens) <= 4
        and any(t in _MENU_WORDS for t in tokens)
        and all(t in _MENU_WORDS or t in STOPWORDS for t in tokens)
    )


def is_list_item(line: str) -> bool:
    """Bullet or enumeration line such as "- …", "1. …" or "a) …"."""
    return bool(_LIST_MARKER_RE.match(line))


def is_task_verb(word: str) -> bool:
    return word.lower() in _TASK_VERBS


def is_task_instruction(s: str) -> bool:
    tokens = tokenize(s)
    return bool(tokens) and (tokens[0].lower() == "aufgabe" or is_task_verb(tokens[0]))


def is_description_sentence(s: str) -> bool:
    """True for complete, content-bearing sentences that may appear in a description."""
    if not 30 <= len(s) <= 350 or is_boilerplate(s) or not _SENTENCE_END_RE.search(s):
        return False
    if len(tokenize(s)) < 5 or letter_ratio(s) < 0.7 or is_list_item(s):
        return False
    return not is_task_instruction(s)


def _breaks_sentence(last: str, line: str) -> bool:
    """Signs, clear in any text, that a line break falls inside a sentence."""
    return bool(_WRAP_HYPHEN_RE.search(last)) or last.endswith(",") or line[0].islower()


def _is_hard_wrapped(lines: list[str]) -> bool:
    """Text copied from a PDF breaks lines inside sentences; web text keeps a paragraph per line."""
    # List items never continue a line, and markers like "a)" would count as lower-case starts.
    # strict=False is the point: a list zipped with its own tail is one pair shorter, and
    # the last line has no successor to be judged against.
    pairs = [(a, b) for a, b in zip(lines, lines[1:], strict=False)
             if a and b and not is_list_item(b)]
    inside = sum(not _SENTENCE_END_RE.search(a) and _breaks_sentence(a, b) for a, b in pairs)
    return len(pairs) >= 4 and inside >= 0.15 * len(pairs)


def _join(joined: str, line: str) -> str:
    if _WRAP_HYPHEN_RE.search(joined):
        next_word = line.split(maxsplit=1)[0].rstrip(".,")
        if line[0].islower() and next_word not in _SUSPENDED_HYPHEN_NEXT:
            return joined[:-1] + line  # "Er-" + "lebnis"
        if line[0].isupper():
            return joined + line  # "Nord-" + "Süd-Dialog"
    return f"{joined} {line}"


def join_wrapped_lines(lines: list[str]) -> list[str]:
    """Undo hard line breaks inside sentences, as in text copied from a PDF.

    A line continues the previous one if that has no sentence end and ends in a hyphen or a
    comma, or if the line starts in lower case. In hard-wrapped text, a line of nearly full
    width continues too, since German lines often go on with a capitalised noun. Empty lines,
    list items and short lines such as headings stay boundaries.
    """
    wrapped = _is_hard_wrapped(lines)
    widths = [len(line) for line in lines if len(line) >= 20]
    full_width = 0.8 * statistics.median(widths) if widths else float("inf")
    joined: list[str] = []
    last = ""  # the previous line as it was, before joining
    for line in lines:
        if (
            joined and last and line and not is_list_item(line) and not _SENTENCE_END_RE.search(last)
            and (_breaks_sentence(last, line) or (wrapped and len(last) >= full_width))
        ):
            joined[-1] = _join(joined[-1], line)
        else:
            joined.append(line)
        last = line
    return joined


def clean_text(raw: str) -> str:
    """Normalise the text, drop table rows, page chrome and repeated lines, undo hard line wraps."""
    text = unicodedata.normalize("NFC", strip_markup_preserving_lines(raw)).translate(_ZERO_WIDTH).replace(" ", " ")
    lines: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            lines.append("")  # paragraph boundary for join_wrapped_lines
        elif line not in seen and letter_ratio(line) >= 0.5 and not is_boilerplate(line):
            seen.add(line)
            lines.append(line)
    return "\n".join(line for line in join_wrapped_lines(lines) if line)


def split_sentences(text: str) -> list[str]:
    """Split into sentences; every line break is a hard boundary (headings, list items)."""
    sentences: list[str] = []
    for line in text.splitlines():
        # Per LINE, not per piece: the comma repair below may join across a piece boundary.
        start = len(sentences)
        for piece in _bounded(line):
            for segment in (s.strip() for s in _segmenter.segment(piece)):
                if not segment:
                    continue
                # pysbd also ends a sentence at "Ludwig XVI., der …"; no sentence starts with
                # "," or ";".
                if len(sentences) > start and segment[0] in ",;":
                    sentences[-1] += segment
                else:
                    sentences.append(segment)
    return sentences


def _is_question_heading(sentence: str, lines: set[str]) -> bool:
    # A question alone on its line is a section heading ("Warum ist … wichtig?").
    return sentence in lines and sentence.rstrip("\"'“”»«)").endswith("?")


def description_candidates(doc: Document) -> list[str]:
    """Sentences usable in a description; falls back to all sentences for very short texts."""
    lines = set(doc.lines)
    candidates = [
        s for s in doc.sentences if is_description_sentence(s) and not _is_question_heading(s, lines)
    ]
    return candidates if len(candidates) >= 2 else doc.sentences


def prepare(raw: str) -> Document:
    text = clean_text(raw)
    lines = text.split("\n") if text else []
    return Document(text=text, lines=lines, sentences=split_sentences(text))
