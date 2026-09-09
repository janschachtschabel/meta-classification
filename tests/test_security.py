"""Unit tests for the security helpers: name sanitization, upload cap, share
expiry. API-level auth/traversal behaviour is covered in test_api.py."""

import asyncio
import io
from datetime import timedelta

import pytest
from fastapi import HTTPException, UploadFile

from app import sharing
from app.security import read_upload_capped, safe_name
from app.sharing import ShareStore


def test_safe_name_accepts_normal_names():
    for name in ("model1", "my-model_2.0", "taxonid", "Modell mit Leerzeichen"):
        assert safe_name(name) == name


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "..",  # traversal
        "e..vil",  # embedded traversal
        "a/b",  # forward slash
        "a\\b",  # backslash (Windows traversal)
        ".hidden",  # leading dot would collide with the registry's tmp staging dirs
        "a\x00b",  # null byte
        "D:evil.csv",  # Windows drive-relative name escapes the storage dir
        "data.csv:hidden",  # NTFS alternate-data-stream
    ],
)
def test_safe_name_rejects_path_characters(bad):
    with pytest.raises(HTTPException) as exc:
        safe_name(bad, "model name")
    assert exc.value.status_code == 400


def test_role_for_key_rejects_non_ascii_key_without_raising():
    """A non-ASCII X-API-Key must be treated as invalid (401), not crash the
    constant-time compare with a TypeError that surfaces as a sanitized 500."""
    from app.security import _role_for_key
    from app.settings import Settings

    settings = Settings(auth_enabled=True, api_key_admin="secret", api_key_readonly=None)
    assert _role_for_key("naïve-key", settings) is None
    assert _role_for_key("secret", settings) == "admin"


def test_read_upload_capped_rejects_oversized_upload():
    """The streamed reader aborts with 413 as soon as the cap is exceeded, so an
    oversized body never fully lands in memory."""
    upload = UploadFile(file=io.BytesIO(b"x" * (2 * 1024 * 1024)))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(read_upload_capped(upload, max_bytes=1024 * 1024))
    assert exc.value.status_code == 413

    small = UploadFile(file=io.BytesIO(b"ok"))
    assert asyncio.run(read_upload_capped(small, max_bytes=1024)) == b"ok"


def test_share_link_expires_and_is_purged(tmp_path, monkeypatch):
    """An expired share id resolves to None and is removed from the persisted
    store — expiry must not depend on a cleanup job."""
    store = ShareStore(tmp_path / "links.json")
    share_id, _ = store.create("dataset", "tiny.csv", expires_hours=1)
    assert store.resolve(share_id) is not None  # live link resolves

    real_now = sharing._now()
    monkeypatch.setattr(sharing, "_now", lambda: real_now + timedelta(hours=2))
    assert store.resolve(share_id) is None  # expired -> gone
    assert share_id not in (tmp_path / "links.json").read_text(encoding="utf-8")  # purged on disk


def test_rate_limits_disabled_resolve_to_unlimited(monkeypatch):
    """APIV3_RATE_LIMIT_ENABLED=false is the documented off-switch: every limit
    resolver must return an effectively unlimited rate."""
    from app.limiter import export_limit, predict_limit, train_limit
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "rate_limit_enabled", False)
    for resolve in (predict_limit, train_limit, export_limit):
        assert int(resolve().split("/")[0]) >= 1_000_000


def test_share_expiry_hours_clamped(tmp_path):
    """expires_hours is clamped to 1..168 so a crafted request cannot create
    an (effectively) immortal link."""
    store = ShareStore(tmp_path / "links.json")
    _, expires_at = store.create("dataset", "tiny.csv", expires_hours=999_999)
    from datetime import datetime

    assert datetime.fromisoformat(expires_at) <= sharing._now() + timedelta(hours=168, minutes=1)


def test_safe_name_rejects_control_characters():
    """A name reaches a Content-Disposition header on the export path, where a newline
    is a header split. The guard rejected NUL only; every other control character got
    through, including \r and \n. Pre-existing, and it matters more now that a second
    route leans on this function."""
    for name in ("foo\nbar", "foo\rbar", "foo\tbar", "foo\x1bbar"):
        with pytest.raises(HTTPException) as raised:
            safe_name(name, "model name")
        assert raised.value.status_code == 400
    safe_name("subjects_v2.1", "model name")  # ordinary names still pass
