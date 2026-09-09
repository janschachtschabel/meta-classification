"""Authentication and input-safety helpers.

- API-key auth with two roles (``admin`` / ``readonly``), compared in constant
  time to avoid timing side channels.
- ``safe_name`` rejects path-traversal in user-supplied model/dataset names.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import Depends, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from .settings import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _role_for_key(key: str | None, settings: Settings) -> str | None:
    """Return the role for an API key, or ``None`` if invalid."""
    if not settings.auth_enabled:
        return "admin"  # Auth disabled -> full access (local use).
    # A non-ASCII key can never match an ASCII secret, and secrets.compare_digest
    # raises TypeError on non-ASCII str — treat it as invalid (401), not a 500.
    if not key or not key.isascii():
        return None
    if settings.api_key_admin and secrets.compare_digest(key, settings.api_key_admin):
        return "admin"
    if settings.api_key_readonly and secrets.compare_digest(key, settings.api_key_readonly):
        return "readonly"
    return None


def require_role(required: str = "readonly"):
    """Build a FastAPI dependency enforcing a minimum role.

    ``readonly`` endpoints accept both roles; ``admin`` endpoints require admin.
    """

    def dependency(
        key: str | None = Security(api_key_header),
        settings: Settings = Depends(get_settings),
    ) -> str:
        role = _role_for_key(key, settings)
        if role is None:
            raise HTTPException(
                status_code=401,
                detail="API key required. Provide the X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if required == "admin" and role != "admin":
            raise HTTPException(status_code=403, detail="Admin API key required for this endpoint.")
        return role

    return dependency


# A name becomes one path component. Filesystems cap that around 255 bytes, and a
# non-ASCII name costs several bytes per character — 100 leaves room to spare while
# staying far above any real model or dataset name.
_MAX_NAME_LENGTH = 100


def safe_name(name: str, kind: str = "name") -> str:
    """Validate a model/dataset name, rejecting path-traversal characters."""
    if len(name) > _MAX_NAME_LENGTH:
        # Without this the OS raises deep inside a write and the client sees an
        # opaque 500 instead of "your name is malformed".
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {kind}: too long ({len(name)} characters, maximum {_MAX_NAME_LENGTH}).",
        )
    if (
        not name
        or ".." in name
        or "/" in name
        or "\\" in name
        or ":" in name  # Windows drive-relative ("D:x") + NTFS ADS ("x:stream") escape the dir
        or name.startswith(".")
        # Every control character, not just NUL: the name reaches a Content-Disposition
        # header on the export path, where a CR or LF is a header split — and a name
        # carrying one is createable on the Linux deployment target.
        or any(char < " " or char == "\x7f" for char in name)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {kind}: {name!r}. Must not contain path characters (/, \\, :, ..).",
        )
    return name


async def spool_upload_capped(upload: UploadFile, max_bytes: int, target: Path) -> int:
    """Stream an upload straight to ``target``, aborting if it exceeds ``max_bytes``.

    For the uploads that are large by design — a WLO export is 126-195 MB gzipped —
    where :func:`read_upload_capped` would hold the chunks AND the joined copy, i.e.
    twice the file, before a byte reached disk. Nothing is held here but one block.

    A refusal or a write error removes the partial file: leaving the bytes on the
    volume would only move the problem the cap exists to prevent. Returns the number
    of bytes written.
    """
    total = 0
    try:
        with target.open("wb") as handle:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit.",
                    )
                handle.write(block)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return total


async def read_upload_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded file in chunks, aborting if it exceeds ``max_bytes``.

    Kept for the model import, which validates the archive as bytes before anything
    touches the filesystem — see ``model_archive.unpack``.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)
