"""Authentication and input-safety helpers.

- API-key auth with two roles (``admin`` / ``readonly``), compared in constant
  time to avoid timing side channels.
- ``safe_name`` rejects path-traversal in user-supplied model/dataset names.
"""

from __future__ import annotations

import secrets

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
        or "\x00" in name
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {kind}: {name!r}. Must not contain path characters (/, \\, :, ..).",
        )
    return name


async def read_upload_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded file in chunks, aborting if it exceeds ``max_bytes``."""
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
