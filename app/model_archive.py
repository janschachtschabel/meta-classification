"""The transportable form of a model: bundle directory <-> ZIP archive.

Split out of ``registry`` when the model card and the checksum manifest landed and
made the archive a second reason to change that file. The division is by concern,
not by size: everything here is a pure function over BYTES — build an archive,
validate an incoming one — while ``registry`` keeps what needs the disk and the
locks (staging, atomic publish, cache).

Validating before touching the filesystem is also what keeps a hostile archive from
reaching it: member names are allowlisted (which kills path traversal, dotfiles and
Windows drive-relative names in one rule), the decompressed size is bounded, and a
manifest, when present, must match.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib

from . import model_card
from .data import label_vocabulary
from .model_io import (
    CARD_FILE,
    MANIFEST_FILE,
    UnsafeModelError,
    build_manifest,
    verify_manifest,
)

# Every api_v3 bundle contains all four; requiring them lets a truncated/crafted
# archive be rejected cleanly (400) instead of crashing later in _read_bundle.
REQUIRED_FILES = {"config.json", "head.skops", "vectorizer.skops", "vocabulary.json"}
# Bundles contain exactly these files. Allowlisting member names (instead of
# pattern-blocking bad ones) also kills dotfiles and Windows drive-relative
# names like "C:evil" that slip past character blocklists.
ALLOWED_MEMBERS = REQUIRED_FILES | {"metrics.json", MANIFEST_FILE, CARD_FILE}
# Import guards. skops stores its members uncompressed, so legitimate exports
# deflate well (~16x measured on a tiny bundle) — a ratio alone would misfire.
# Reject only archives that are BOTH large in absolute terms (> floor) and
# inflate far beyond the upload size (memory-DoS via zip bomb). The guard covers
# the outer envelope only; the skops members are parsed by skops itself during
# validation — acceptable residual risk since import is admin-only + rate-limited.
_MAX_DECOMPRESSION_RATIO = 20
_DECOMPRESSION_FLOOR_BYTES = 64 * 1024 * 1024
# What a damaged archive actually raises, measured rather than assumed (see unpack).
_DAMAGED = (zipfile.BadZipFile, zlib.error, EOFError, ValueError)


def pack(name: str, members: dict[str, bytes]) -> bytes:
    """Build the archive for one bundle, adding the model card and the manifest.

    Both are generated here rather than read from disk, so they always describe THIS
    archive; the caller passes only the bundle's own files.
    """
    config = json.loads(members["config.json"])
    metadata = json.loads(members["metrics.json"]) if "metrics.json" in members else {}
    vocabulary = label_vocabulary(config.get("classes") or [])

    members = dict(members)
    members[CARD_FILE] = model_card.render(name, config, metadata, vocabulary).encode("utf-8")
    manifest = build_manifest(name, members, metadata)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, payload in sorted(members.items()):
            archive.writestr(member, payload)
        archive.writestr(MANIFEST_FILE, json.dumps(manifest, ensure_ascii=False, indent=2))
    return buffer.getvalue()


def unpack(data: bytes) -> dict[str, bytes]:
    """Validate an uploaded archive and return the files to install.

    The transport artifacts (manifest and card) are verified and then dropped: they
    describe one archive and are rebuilt on the next export, so keeping them on disk
    would make that export ship a stale copy beside the fresh one.

    :raises UnsafeModelError: for anything we will not install — an unreadable zip,
        an unexpected or missing member, a zip bomb, or a checksum that does not match.
    """
    try:
        archive_file = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UnsafeModelError(f"Not a valid zip archive: {exc}") from exc

    with archive_file as archive:
        names = archive.namelist()
        unexpected = set(names) - ALLOWED_MEMBERS
        if unexpected:
            raise UnsafeModelError(f"Unexpected archive members: {sorted(unexpected)}")
        if not REQUIRED_FILES.issubset(set(names)):
            raise UnsafeModelError(f"Archive missing required files: {sorted(REQUIRED_FILES)}")
        # The upload cap bounds only the COMPRESSED size; refuse archives that
        # inflate far beyond it (zip bomb -> memory DoS) before extracting.
        total_uncompressed = sum(info.file_size for info in archive.infolist())
        if total_uncompressed > max(_DECOMPRESSION_FLOOR_BYTES, _MAX_DECOMPRESSION_RATIO * len(data)):
            raise UnsafeModelError(
                f"Archive decompresses to {total_uncompressed} bytes from a "
                f"{len(data)}-byte upload; refusing (possible zip bomb)."
            )
        try:
            payloads = {member: archive.read(member) for member in names}
        except _DAMAGED as exc:
            # Damage in transit is the normal failure for a 50-180 MB download, and
            # it must read as "your file is broken", not as a server fault. Measured
            # on a real archive: a mangled member name raises BadZipFile, a flipped
            # data byte — the likeliest damage — raises zlib.error, and a truncated
            # stream raises ValueError or EOFError depending on where it was cut.
            raise UnsafeModelError(f"Archive member could not be read: {exc!r}") from exc

    if MANIFEST_FILE in payloads:
        # Absent for bundles exported before 3.2: verification applies when a
        # manifest is there, its absence is not an error.
        try:
            manifest = json.loads(payloads[MANIFEST_FILE])
        except ValueError as exc:
            raise UnsafeModelError(f"{MANIFEST_FILE} is not valid JSON: {exc!r}") from exc
        verify_manifest(manifest, {m: p for m, p in payloads.items() if m != MANIFEST_FILE})
    return {m: p for m, p in payloads.items() if m not in (MANIFEST_FILE, CARD_FILE)}
