"""Which character separates a dataset file's fields: found from its header line.

The counterpart of ``csv_encoding``: both answer how a file wants to be read from the file
itself. Every request that reads a dataset defaults to ``;``, and the admin UI used to send
nothing else, so a comma- or tab-separated upload showed one column holding the whole header.
``GET /datasets/{name}`` detects the separator when none is named and reports it, and
``data.count_rows`` guesses with the same rule.

A stdlib-only leaf.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from .csv_encoding import is_gzipped

# The field separators a header is searched for, in order of preference: on a tie -- a
# single column, say -- the first wins, which is this API's default.
DELIMITERS = (";", ",", "\t", "|")
# A quoted header field may hold any of them ("Titel, mit Komma"): such stretches are blanked
# before counting. `[^"]*` cannot backtrack, so a long header costs one pass.
_QUOTED_RE = re.compile(r'"[^"]*"')


def separator_from_header(header: str) -> str:
    """Whichever of ``DELIMITERS`` occurs most often in ``header`` outside quotes."""
    unquoted = _QUOTED_RE.sub("", header)
    return max(DELIMITERS, key=unquoted.count)


def detect_separator(path: str | Path) -> str:
    """The field separator a CSV's header line uses (``separator_from_header``), ``;`` on a
    tie. Gzip is read decompressed.

    For a caller that names none (``GET /datasets/{name}``): every request model still
    defaults to ``;``, so a client that relies on that keeps it, and the admin UI sends on
    what this found.
    """
    path = Path(path)
    opener = gzip.open if is_gzipped(path) else open
    with opener(path, "rt", encoding="utf-8", errors="ignore", newline="") as handle:
        return separator_from_header(handle.readline())
