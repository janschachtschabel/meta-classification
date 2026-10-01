"""The transportable form of a model: bundle directory <-> ZIP archive.

Split out of ``registry`` when the model card and the checksum manifest landed and
made the archive a second reason to change that file. The division is by concern,
not by size: everything here builds or validates the transportable form, while
``registry`` keeps the locks, the staging and the atomic publish. ``pack_into`` reads
the member files directly — a production bundle is 50-180 MB, and handing it over as
bytes cost 2.78x the bundle in peak memory (measured).

Validating before touching the filesystem is also what keeps a hostile archive from
reaching it: member names are allowlisted (which kills path traversal, dotfiles and
Windows drive-relative names in one rule), the decompressed size is bounded, and a
manifest, when present, must match.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import IO, BinaryIO

from . import model_card
from .bundle_meta import as_mapping
from .label_names import label_vocabulary
from .manifest import (
    CARD_FILE,
    MANIFEST_FILE,
    build_manifest,
    digest_file,
    verify_digests,
    verify_manifest,
)
from .model_io import UnsafeModelError

# One block per read while streaming a member to disk. 1 MiB is large enough that the syscall
# count is irrelevant next to the inflate, and small enough to be invisible in the peak.
_STREAM_BLOCK = 1024 * 1024

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
# The ratio is a multiplier, so on its own it stops bounding anything: at the default 200 MB
# upload cap it permits 4 GiB of declared expansion, and `unpack` returns every member's bytes,
# so all of it lands in the SERVING process's RAM on top of the upload buffer — beside the
# model LRU cache and whatever a training is holding. An absolute ceiling is what makes this a
# bound. 1 GiB sits well above any bundle this project can produce (the largest described is a
# 300-label model at 200k features: n_labels x n_features x 4 = ~240 MB of head, plus a ~3 MB
# vocabulary and the vectorizer) and a quarter of what the ratio alone waved through.
_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
# What a damaged archive actually raises, measured rather than assumed (see unpack).
_DAMAGED = (zipfile.BadZipFile, zlib.error, EOFError, ValueError)
# A STREAMED read raises one more: `ZipFile.open` seeks to the member's local header, so a
# corrupt central directory surfaces as OSError(EINVAL) from that seek rather than BadZipFile.
# Only ever applied to the read — an OSError from the destination file is a full disk or a
# read-only volume, i.e. a server fault, and must not read as "your archive is broken".
_DAMAGED_STREAM = (*_DAMAGED, OSError)


def _refuse_if_overexpanded(total_uncompressed: int, compressed: int) -> None:
    """Refuse an archive that declares more expansion than either limit allows.

    Two limits, and the tighter one wins. The ratio (with its floor) is what lets a small,
    genuinely compressible bundle through; the ceiling is what stops the ratio from scaling the
    allowance with the attacker's own upload. Split out of `unpack` so the policy can be tested
    as the arithmetic it is — reaching the ceiling through a real archive would need a ~75 MB
    upload just to clear the ratio first.

    The sizes come from the central directory, which is binding: CPython truncates a member to
    its declared `file_size`, so a lying header cannot deliver more than it claims.
    """
    allowance = min(
        max(_DECOMPRESSION_FLOOR_BYTES, _MAX_DECOMPRESSION_RATIO * compressed),
        _MAX_UNCOMPRESSED_BYTES,
    )
    if total_uncompressed > allowance:
        raise UnsafeModelError(
            f"Archive decompresses to {total_uncompressed} bytes from a "
            f"{compressed}-byte upload, over the {allowance}-byte allowance; "
            "refusing (possible zip bomb)."
        )


def _document(path: Path | None) -> dict:
    """One of the bundle's JSON documents, or an empty one if it will not parse.

    Generating the card made export the one operation that *parses* a bundle — which
    left the bundle most in need of being exported, a damaged one an operator wants to
    inspect elsewhere, as the one export refused. Only the generated card goes without
    it; the member itself still travels byte for byte.
    """
    if path is None:
        return {}
    try:
        return as_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def pack_into(target: BinaryIO, name: str, sources: dict[str, Path]) -> None:
    """Write one bundle's archive into ``target``, adding the model card and manifest.

    Both are generated here rather than read from disk, so they always describe THIS
    archive; the caller passes only the bundle's own files.

    Members stream from disk instead of being handed over as bytes. Measured on a real
    51 MB bundle, the old byte path peaked at 142 MB of Python heap — 2.78x the bundle —
    because it held every member, the zip built beside them, and the copy ``getvalue()``
    makes. ``ZipFile.write`` reads in blocks and the digests are computed the same way,
    so the peak no longer scales with the bundle.
    """
    config = _document(sources.get("config.json"))
    metadata = _document(sources.get("metrics.json"))
    vocabulary = label_vocabulary(config.get("classes") or [])
    card = model_card.render(name, config, metadata, vocabulary).encode("utf-8")

    digests = {member: digest_file(path) for member, path in sources.items()}
    digests[CARD_FILE] = hashlib.sha256(card).hexdigest()
    manifest = build_manifest(name, digests, metadata)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, path in sorted(sources.items()):
            archive.write(path, member)
        archive.writestr(CARD_FILE, card)
        archive.writestr(MANIFEST_FILE, json.dumps(manifest, ensure_ascii=False, indent=2))


def _open(source: Path | io.BytesIO) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise UnsafeModelError(f"Not a valid zip archive: {exc}") from exc


def _validated_names(archive: zipfile.ZipFile, compressed: int) -> list[str]:
    """The archive's members, or ``UnsafeModelError`` — the checks that run before a byte is
    extracted.

    Shared by both import paths so the member allowlist (which is what kills path traversal,
    dotfiles and Windows drive-relative names) and the expansion bound are stated once.
    """
    names = archive.namelist()
    unexpected = set(names) - ALLOWED_MEMBERS
    if unexpected:
        raise UnsafeModelError(f"Unexpected archive members: {sorted(unexpected)}")
    if not REQUIRED_FILES.issubset(set(names)):
        raise UnsafeModelError(f"Archive missing required files: {sorted(REQUIRED_FILES)}")
    # The upload cap bounds only the COMPRESSED size; refuse archives that
    # inflate far beyond it (zip bomb -> memory DoS) before extracting.
    _refuse_if_overexpanded(sum(info.file_size for info in archive.infolist()), compressed)
    return names


def _manifest_from(payload: bytes) -> object:
    try:
        return json.loads(payload)
    except ValueError as exc:
        raise UnsafeModelError(f"{MANIFEST_FILE} is not valid JSON: {exc!r}") from exc


def unpack(data: bytes) -> dict[str, bytes]:
    """Validate an in-memory archive and return the files to install.

    The transport artifacts (manifest and card) are verified and then dropped: they
    describe one archive and are rebuilt on the next export, so keeping them on disk
    would make that export ship a stale copy beside the fresh one.

    Holds every member at once, which is why the install path uses :func:`unpack_into`
    instead (audit API-5). Kept for callers that already have the bytes.

    :raises UnsafeModelError: for anything we will not install — an unreadable zip,
        an unexpected or missing member, a zip bomb, or a checksum that does not match.
    """
    with _open(io.BytesIO(data)) as archive:
        names = _validated_names(archive, len(data))
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
        # Absent for bundles exported before the manifest existed: verification
        # applies when a manifest is there, its absence is not an error.
        verify_manifest(_manifest_from(payloads[MANIFEST_FILE]),
                        {m: p for m, p in payloads.items() if m != MANIFEST_FILE})
    return {m: p for m, p in payloads.items() if m not in (MANIFEST_FILE, CARD_FILE)}


def _read_blocks(source: IO[bytes]) -> Iterator[bytes]:
    """The member's bytes in blocks, with a read failure reported as archive damage.

    A generator so the guard covers exactly the read: the caller's ``sink.write`` stays outside
    it, because a write failure is the server's problem and not the archive's.
    """
    while True:
        try:
            block = source.read(_STREAM_BLOCK)
        except _DAMAGED_STREAM as exc:
            raise UnsafeModelError(f"Archive member could not be read: {exc!r}") from exc
        if not block:
            return
        yield block


def unpack_into(archive_path: Path, target: Path) -> None:
    """Validate an archive on disk and write the installable members into ``target``.

    The memory-frugal half of the pair. ``unpack`` holds the whole archive *and* every member;
    at the 200 MB upload cap that was ~400 MB of buffer plus the members again, inside the
    process that serves predictions, for an operation that only ever moves bytes to disk
    (audit API-5). Here the zip is read from the file, one member at a time, in blocks.

    The checksums are computed while streaming and compared afterwards with the same rules the
    buffered path uses (``manifest.verify_digests``). A member is therefore written before it
    is known to be intact, so **every member is removed again if anything fails** — a caller
    must never be handed a partially verified bundle to publish.

    :raises UnsafeModelError: the same set as :func:`unpack`.
    """
    written: list[Path] = []
    try:
        with _open(archive_path) as archive:
            names = _validated_names(archive, archive_path.stat().st_size)
            digests: dict[str, str] = {}
            transport: dict[str, bytes] = {}
            for member in names:
                if member in (MANIFEST_FILE, CARD_FILE):
                    # Kilobytes of JSON and markdown, and the manifest is needed whole to
                    # compare against; the members it describes are the large ones.
                    try:
                        transport[member] = archive.read(member)
                    except _DAMAGED_STREAM as exc:
                        raise UnsafeModelError(
                            f"Archive member could not be read: {exc!r}") from exc
                    continue
                digest = hashlib.sha256()
                destination = target / member
                try:
                    source = archive.open(member)
                except _DAMAGED_STREAM as exc:
                    raise UnsafeModelError(f"Archive member could not be read: {exc!r}") from exc
                with source, destination.open("wb") as sink:
                    written.append(destination)
                    for block in _read_blocks(source):
                        digest.update(block)
                        sink.write(block)
                digests[member] = digest.hexdigest()

        if MANIFEST_FILE in transport:
            recorded = dict(digests)
            if CARD_FILE in transport:
                recorded[CARD_FILE] = hashlib.sha256(transport[CARD_FILE]).hexdigest()
            verify_digests(_manifest_from(transport[MANIFEST_FILE]), recorded)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
