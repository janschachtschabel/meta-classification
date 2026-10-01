"""German sentence splitting for the metadata generators: pysbd, bounded and per thread.

Split out of `textprep` when that module grew past its guide a second time -- the seam its
docstring had named. Two things make this more than a call to pysbd, and both are pinned by
`tests/test_metadata_sentences.py`:

- pysbd's German abbreviation pass is super-linear in the length of ONE call, so a long line
  is handed over in pieces of at most `MAX_SEGMENT_CHARS` (audit 2026-09-27, S-2);
- pysbd keeps the text it segments on the Segmenter object, so each thread has its own (audit
  2026-09-30, M02).

pysbd is imported under a scoped warning filter, because its 0.3.4 regexes raise a
SyntaxWarning that this project's warnings-as-errors gate would otherwise turn into an import
failure.
"""

from __future__ import annotations

import re
import threading
import warnings

with warnings.catch_warnings():
    # pysbd 0.3.4 writes '\s' in plain strings, so importing it raises SyntaxWarning while its
    # bytecode is compiled — once, on a cold cache, which is the worst kind of intermittent.
    # Scoped to this import: the project's global "warnings are errors" gate stays armed, and
    # a SyntaxWarning in our own code still fails the suite.
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

# pysbd keeps the text it segments on the Segmenter, so one may not be shared by the threads
# that serve requests: parallel requests got each other's sentences (audit 2026-09-30, M02).
# One per thread, built on first use -- it takes microseconds.
_per_thread = threading.local()


def _segmenter() -> pysbd.Segmenter:
    try:
        return _per_thread.segmenter
    except AttributeError:
        _per_thread.segmenter = pysbd.Segmenter(language="de", clean=False)
        return _per_thread.segmenter


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


def split_sentences(text: str) -> list[str]:
    """Split into sentences; every line break is a hard boundary (headings, list items)."""
    sentences: list[str] = []
    for line in text.splitlines():
        # Per LINE, not per piece: the comma repair below may join across a piece boundary.
        start = len(sentences)
        for piece in _bounded(line):
            for segment in (s.strip() for s in _segmenter().segment(piece)):
                if not segment:
                    continue
                # pysbd also ends a sentence at "Ludwig XVI., der …"; no sentence starts with
                # "," or ";".
                if len(sentences) > start and segment[0] in ",;":
                    sentences[-1] += segment
                else:
                    sentences.append(segment)
    return sentences
