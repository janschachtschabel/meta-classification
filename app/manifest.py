"""The manifest an exported bundle carries: a SHA-256 per member, checked on import.

Moved out of ``model_io`` when that module passed the ~300-line guide (audit 2026-09-30,
R08): this is the ZIP form's integrity -- what an export claims about its members, and how an
import holds an archive to it -- and it changes with the archive, while ``model_io`` changes
with the bundle's form on disk. Read by ``model_archive``; ``registry`` takes the two member
names from here.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .errors import UnsafeModelError
from .model_io import FORMAT_VERSION

# Transport-only members: generated per export, verified on import, never kept in the
# installed bundle (a stale copy on disk would be zipped alongside the fresh one).
MANIFEST_FILE = "manifest.json"
CARD_FILE = "README.md"


def digest_file(path: Path) -> str:
    """SHA-256 of a file, read in blocks — a single bundle member reaches ~120 MB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(name: str, digests: dict[str, str], metadata: dict) -> dict:
    """Describe an archive: a SHA-256 for every member, plus what it is.

    Deliberately built at EXPORT rather than at save time. ``metrics.json`` is
    mutable by design (``PUT /models/{name}/info``) and ``config.json`` is rewritten
    by the label-repair scripts, so a manifest stored next to them would be
    invalidated by every legitimate edit — and a load-time check would then refuse a
    perfectly good bundle. What actually needs protecting is the 50-180 MB download
    between two servers, and that is exactly the export/import boundary.
    """
    metrics = metadata.get("metrics") or {}
    return {
        "model_name": name,
        "app_version": __version__,
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "created_at": metadata.get("created_at"),
        "n_labels": metadata.get("n_labels"),
        "f1_macro": metrics.get("f1_macro"),
        "dataset": metadata.get("dataset"),
        "files": dict(sorted(digests.items())),
    }


def verify_manifest(manifest: object, members: dict[str, bytes]) -> None:
    """Check every member against the manifest; raise ``UnsafeModelError`` on any drift.

    Covers three failures with one comparison: a truncated download, a member altered
    in transit, and a member the manifest does not mention at all.
    """
    verify_digests(manifest, {
        member: hashlib.sha256(payload).hexdigest() for member, payload in members.items()
    })


def verify_digests(manifest: object, digests: dict[str, str]) -> None:
    """The same check, for a caller that hashed the members as it streamed them to disk.

    Split out so the buffered and streaming import paths compare against the manifest with
    one set of rules rather than two implementations of the same three failures (audit API-5).
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise UnsafeModelError(f"{MANIFEST_FILE} is malformed")
    expected: dict = manifest["files"]
    for member, digest in sorted(digests.items()):
        recorded = expected.get(member)
        if recorded is None:
            raise UnsafeModelError(f"{member} is not listed in {MANIFEST_FILE}")
        if digest != recorded:
            raise UnsafeModelError(
                f"{member} does not match its checksum in {MANIFEST_FILE} "
                "(the archive was altered or arrived incomplete)"
            )
    missing = sorted(set(expected) - set(digests))
    if missing:
        raise UnsafeModelError(f"{MANIFEST_FILE} lists files the archive lacks: {missing}")
