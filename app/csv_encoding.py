"""Which encoding a dataset file is in: decided once, for the whole file, from its bytes.

German metadata exports arrive as UTF-8 or as Windows-1252, and every reader used to decide
by trial: UTF-8, and on the first undecodable byte anywhere, the whole file again as cp1252.
One truncated umlaut in an otherwise clean UTF-8 export -- a `0xC3` whose second byte was cut
-- turned every "ä" of every row into "Ã¤". Training and test saw the same garbage, so the
metrics did not show it (audit 2026-09-30, T02). With a curly quote in the file as well, the
cp1252 attempt hit `0x9D`, which cp1252 does not define, and the run failed without a reason.

The bytes answer the question instead of a first error. Every byte >= 0x80 either belongs to
a valid UTF-8 multi-byte sequence or it does not, and the two counts point one way:

- none invalid                         -> UTF-8.
- valid ones are rare accidents        -> cp1252. A cp1252 text forms a valid pair only where a
  capital umlaut or "ß" runs into a curly quote, dash or ellipsis (`heiß“` is DF 93).
- a handful invalid among valid ones   -> UTF-8, those bytes read as U+FFFD, and counted.
- anything else                        -> refused, with the offset of the first bad byte:
  mixed encodings or a damaged export, and either way no reading of it is right.

A stdlib-only leaf: ``data``, ``dataset_load`` and ``predict_csv`` all read through it.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import TrainingInputError

# "A handful": a truncated entry or two in an export of any size. More is a systematic fault
# (a second encoding spliced in) that replacing bytes would hide rather than repair.
MAX_REPLACED = 10
# cp1252 is taken when UTF-8 accounts for at most this share of the non-ASCII evidence.
_ACCIDENTAL_SHARE = 0.05
BLOCK_BYTES = 1 << 20

_HIGH_BYTES = bytes(range(0x80, 0x100))
# Well-formed UTF-8 sequences of two to four bytes (RFC 3629 table 3-7): no overlongs, no
# surrogates, nothing past U+10FFFF. Each alternative starts at a distinct lead byte and has
# a fixed length, so the search is linear.
_MULTIBYTE = re.compile(
    rb"[\xc2-\xdf][\x80-\xbf]"
    rb"|\xe0[\xa0-\xbf][\x80-\xbf]|[\xe1-\xec\xee\xef][\x80-\xbf]{2}|\xed[\x80-\x9f][\x80-\xbf]"
    rb"|\xf0[\x90-\xbf][\x80-\xbf]{2}|[\xf1-\xf3][\x80-\xbf]{3}|\xf4[\x80-\x8f][\x80-\xbf]{2}"
)
# The five bytes Python's cp1252 codec (like Windows) leaves undefined.
_CP1252_UNDEFINED = re.compile(rb"[\x81\x8d\x8f\x90\x9d]")


@dataclass(frozen=True)
class CsvEncoding:
    """How to decode a file: ``name`` as pandas and ``open`` take it, and how many bytes
    that were not UTF-8 are read as U+FFFD (``replaced``, UTF-8 only)."""

    name: str
    replaced: int = 0

    @property
    def errors(self) -> str:
        return "replace" if self.replaced else "strict"

    def describe(self) -> dict[str, object]:
        """What a bundle records about the file it was trained on."""
        return {"name": self.name, "replaced_bytes": self.replaced}


def is_gzipped(path: str | Path) -> bool:
    """Does this dataset path denote a gzip-compressed CSV?

    Keyed off the name, exactly like pandas' own ``compression="infer"``, so what the reader
    and the other passes over the file consider compressed can never disagree.
    """
    return str(path).lower().endswith(".gz")


def _cut(data: bytes) -> int:
    """Where to end a block so no UTF-8 sequence is split between two: before a trailing
    lead byte whose continuation bytes have not all arrived."""
    end = len(data)
    start = end - 1
    while start >= 0 and end - start <= 3 and 0x80 <= data[start] <= 0xBF:
        start -= 1
    if start < 0:
        return end
    lead = data[start]
    needed = 2 if 0xC2 <= lead <= 0xDF else 3 if 0xE0 <= lead <= 0xEF else 4 if 0xF0 <= lead <= 0xF4 else 1
    return start if end - start < needed else end


@dataclass
class _Evidence:
    valid: int = 0             # well-formed multi-byte sequences
    invalid: int = 0           # bytes >= 0x80 in none
    first_invalid: int | None = None
    first_undefined: int | None = None  # first byte cp1252 does not define

    def add(self, part: bytes, offset: int) -> None:
        sequences = _MULTIBYTE.findall(part)
        high = len(part) - len(part.translate(None, _HIGH_BYTES))
        invalid = high - sum(map(len, sequences))
        self.valid += len(sequences)
        self.invalid += invalid
        if invalid and self.first_invalid is None:
            try:
                part.decode("utf-8")
            except UnicodeDecodeError as exc:
                self.first_invalid = offset + exc.start
        if self.first_undefined is None and (match := _CP1252_UNDEFINED.search(part)):
            self.first_undefined = offset + match.start()


def detect(path: str | Path, *, limit: int | None = None, block_bytes: int = BLOCK_BYTES) -> CsvEncoding:
    """Decide how to decode the file at ``path`` (decompressed first, if gzipped).

    ``limit`` judges only the first so many bytes -- for a preview that reads a few rows,
    which must not scan a whole export for them.

    :raises TrainingInputError: the file cannot be read correctly as either encoding; the
        message names the byte offset where it goes wrong.
    """
    evidence = _Evidence()
    offset = 0
    carry = b""
    opener = gzip.open if is_gzipped(path) else open
    with opener(path, "rb") as handle:
        while limit is None or offset + len(carry) < limit:
            block = handle.read(block_bytes if limit is None else min(block_bytes, limit - offset - len(carry)))
            data = carry + block
            cut = _cut(data) if block else len(data)
            evidence.add(data[:cut], offset)
            offset += cut
            carry = data[cut:]
            if not block:
                break
    return _decide(evidence)


def _decide(evidence: _Evidence) -> CsvEncoding:
    if not evidence.invalid:
        return CsvEncoding("utf-8")
    if evidence.valid <= _ACCIDENTAL_SHARE * (evidence.valid + evidence.invalid):
        if evidence.first_undefined is not None:
            raise TrainingInputError(
                f"The file is neither UTF-8 nor Windows-1252: byte {evidence.first_undefined} "
                "is one Windows-1252 does not define. Export the file as UTF-8."
            )
        return CsvEncoding("cp1252")
    if evidence.invalid <= MAX_REPLACED and evidence.valid > evidence.invalid:
        return CsvEncoding("utf-8", replaced=evidence.invalid)
    raise TrainingInputError(
        f"The file is mostly UTF-8, but {evidence.invalid} bytes are not UTF-8, the first at byte "
        f"{evidence.first_invalid} -- more than the {MAX_REPLACED} a damaged export may have "
        "replaced, or a second encoding mixed in. Export the file again as UTF-8."
    )
