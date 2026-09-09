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


def test_links_can_be_listed_and_revoked(tmp_path):
    """A share link is a bearer capability valid for up to a week: whoever holds the
    id can download without a key. Creating one was possible, seeing or withdrawing it
    was not — the only way to take a link back was editing the JSON on the volume."""
    path = tmp_path / "links.json"
    store = ShareStore(path)
    model_id, _ = store.create("model", "subjects", expires_hours=24)
    dataset_id, _ = store.create("dataset", "data.csv", expires_hours=1)

    listed = store.list()
    assert {entry["share_id"] for entry in listed} == {model_id, dataset_id}
    by_id = {entry["share_id"]: entry for entry in listed}
    assert by_id[model_id]["kind"] == "model" and by_id[model_id]["name"] == "subjects"
    assert by_id[model_id]["created_at"], "an operator needs to see how old a link is"

    assert store.revoke(model_id) is True
    assert store.resolve(model_id) is None
    assert store.revoke(model_id) is False, "revoking twice is not an error, just a no-op"
    # Revocation is durable, not only in memory — the store is reloaded per process.
    assert {entry["share_id"] for entry in ShareStore(path).list()} == {dataset_id}


def test_links_stored_before_created_at_existed_still_list(tmp_path):
    """The field arrived with this feature; links already on a running server predate
    it. They must remain listable and revocable rather than crashing the overview."""
    path = tmp_path / "links.json"
    path.write_text(
        '{"legacy": {"kind": "model", "name": "m", "expires_at": "2099-01-01T00:00:00+00:00"}}',
        encoding="utf-8",
    )
    store = ShareStore(path)
    entry = store.list()[0]
    assert entry["share_id"] == "legacy"
    assert entry["created_at"] is None
    assert store.revoke("legacy") is True


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
