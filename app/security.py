"""Authentication and input-safety helpers.

- API-key auth with two roles (``admin`` / ``readonly``), compared in constant
  time to avoid timing side channels. With auth disabled, a caller on THIS machine
  is admin and every other caller is refused: keyless mode is local use.
- ``safe_name`` rejects path-traversal in user-supplied model/dataset names.
"""

from __future__ import annotations

import ipaddress
import secrets
from pathlib import Path

from fastapi import Depends, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader

from .settings import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# What a proxy adds to say whom it relays: the de-facto header uvicorn reads, the standard
# one (RFC 7239), and nginx's.
_FORWARDING_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")


def _role_for_key(key: str | None, settings: Settings) -> str | None:
    """Return the role for an API key, or ``None`` if invalid.

    Only for auth ON. The keyless case is decided in ``require_role``, where the caller's
    address is known — answering it here as well was a second, unguarded route to admin.
    """
    # A non-ASCII key can never match an ASCII secret, and secrets.compare_digest
    # raises TypeError on non-ASCII str — treat it as invalid (401), not a 500.
    if not key or not key.isascii():
        return None
    if settings.api_key_admin and secrets.compare_digest(key, settings.api_key_admin):
        return "admin"
    if settings.api_key_readonly and secrets.compare_digest(key, settings.api_key_readonly):
        return "readonly"
    return None


def _is_loopback_client(request: Request) -> bool:
    """True when the request peer is a loopback address (127.0.0.0/8 or ::1) and no proxy
    relayed the request.

    The rule data-prep already applies, so both apps mean the same by "local". A missing
    client (an in-process ASGI call) counts as loopback; an unparseable host — a proxy's
    hostname — counts as the network, because behind a proxy the operator must set a key.

    A request carrying a forwarding header is never local (audit 2026-09-30, S04). uvicorn
    rewrites the peer from X-Forwarded-For for EVERY peer in FORWARDED_ALLOW_IPS, so with an
    ingress controller's pod CIDR there, any pod in it sending `X-Forwarded-For: 127.0.0.1`
    was keyless admin; and a reverse proxy on this machine connects from loopback whoever it
    relays. The rewritten peer is all the app sees — the header that caused it is the tell.
    """
    if any(header in request.headers for header in _FORWARDING_HEADERS):
        return False
    client = request.client
    if client is None:
        return True
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        return False


_KEYLESS_FROM_THE_NETWORK = (
    "APIV3_AUTH_ENABLED is false, which serves only callers on this machine. To serve anyone "
    "else, set APIV3_AUTH_ENABLED=true with APIV3_API_KEY_ADMIN and APIV3_API_KEY_READONLY."
)


def authenticate(request: Request, key: str | None, settings: Settings) -> str:
    """The caller's role -- or the refusal: 401 without a valid key, 403 for a caller from
    the network in keyless mode.

    One rule for the two places that ask: ``require_role`` (per route, with the role the
    route needs) and the body guard in ``middleware``, which asks before a request body is
    read (audit 2026-09-30, S01) -- a dependency runs only after FastAPI has parsed it.
    """
    if not settings.auth_enabled:
        # Local use only, as the docs have always said. With no key there is nothing that
        # tells the operator apart from anyone else who can reach the port, so one variable
        # plus a `-p 8000:8000` or the chart's ingress was an open admin API (audit
        # 2026-09-27, S-1).
        if _is_loopback_client(request):
            return "admin"
        raise HTTPException(status_code=403, detail=_KEYLESS_FROM_THE_NETWORK)
    role = _role_for_key(key, settings)
    if role is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return role


def require_role(required: str = "readonly"):
    """Build a FastAPI dependency enforcing a minimum role.

    ``readonly`` endpoints accept both roles; ``admin`` endpoints require admin.
    """

    def dependency(
        request: Request,
        key: str | None = Security(api_key_header),
        settings: Settings = Depends(get_settings),
    ) -> str:
        role = authenticate(request, key, settings)
        if required == "admin" and role != "admin":
            raise HTTPException(status_code=403, detail="Admin API key required for this endpoint.")
        return role

    return dependency


# A name becomes one path component, and BOTH limits matter. Filesystems cap a component
# at ~255 bytes, so the byte bound is the one the OS enforces; the character bound keeps
# the name readable and is far above any real model or dataset name. Counting characters
# alone was not enough: 100 four-byte characters are 400 bytes, which ext4 refuses with
# ENAMETOOLONG — an unhandled OSError, i.e. the opaque 500 this check exists to prevent.
_MAX_NAME_LENGTH = 100
_MAX_NAME_BYTES = 200  # under 255 with room for the ".part"/".tmp" suffixes staging adds


def safe_name(name: str, kind: str = "name") -> str:
    """Validate a model/dataset name, rejecting path-traversal characters."""
    if len(name) > _MAX_NAME_LENGTH:
        # Without this the OS raises deep inside a write and the client sees an
        # opaque 500 instead of "your name is malformed".
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {kind}: too long ({len(name)} characters, maximum {_MAX_NAME_LENGTH}).",
        )
    if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid {kind}: too long ({len(name.encode('utf-8'))} bytes, maximum "
                    f"{_MAX_NAME_BYTES}). Non-ASCII characters cost several bytes each."),
        )
    if (
        not name
        or ".." in name
        or "/" in name
        or "\\" in name
        or ":" in name  # Windows drive-relative ("D:x") + NTFS ADS ("x:stream") escape the dir
        or name.startswith(".")
        # A quote or semicolon ends the filename parameter of the Content-Disposition
        # header the model export builds by hand, so `x";filename="setup.exe` would serve
        # the bundle under a name of the caller's choosing — and that handler also serves
        # share links, i.e. it is reachable without a key.
        or '"' in name
        or ";" in name
        # Every control character, not just NUL: the name reaches a Content-Disposition
        # header on the export path, where a CR or LF is a header split — and a name
        # carrying one is createable on the Linux deployment target.
        or any(char < " " or char == "\x7f" for char in name)
    ):
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid {kind}: {name!r}. Must not contain path characters "
                    f"(/, \\, :, ..), quotes or semicolons."),
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
