"""Tests for the expiring, JSON-backed share-link store."""

from app.sharing import ShareStore


def test_create_resolve_and_persist(tmp_path):
    path = tmp_path / "links.json"
    share_id, expires_at = ShareStore(path).create("model", "m1", expires_hours=24)
    assert expires_at  # ISO expiry timestamp returned
    # A fresh instance reads the persisted link back from disk.
    info = ShareStore(path).resolve(share_id)
    assert info is not None
    assert info["kind"] == "model"
    assert info["name"] == "m1"


def test_unknown_link_returns_none(tmp_path):
    assert ShareStore(tmp_path / "links.json").resolve("does-not-exist") is None


def test_expired_link_is_purged_on_load(tmp_path):
    path = tmp_path / "links.json"
    path.write_text(
        '{"old": {"kind": "model", "name": "m", "expires_at": "2000-01-01T00:00:00+00:00"}}',
        encoding="utf-8",
    )
    store = ShareStore(path)  # _load() drops links whose expiry is in the past
    assert store.resolve("old") is None
