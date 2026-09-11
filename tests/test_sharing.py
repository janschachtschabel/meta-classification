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


def test_a_link_that_expires_while_the_process_runs_leaves_the_listing(tmp_path, monkeypatch):
    """Expiry was enforced in two of the three readers: `_load` purges at startup and
    `resolve` refuses at use. `list` — the overview an operator revokes FROM — had no
    check at all, so a link that died while the server was up was still displayed as
    outstanding. The one screen that answers "what is still handed out" answered wrong.
    """
    from datetime import timedelta

    import app.sharing as sharing

    store = sharing.ShareStore(tmp_path / "links.json")
    short, _ = store.create("model", "expiring", expires_hours=1)
    long_lived, _ = store.create("model", "still_valid", expires_hours=48)
    assert {entry["share_id"] for entry in store.list()} == {short, long_lived}

    later = sharing._now() + timedelta(hours=2)
    monkeypatch.setattr(sharing, "_now", lambda: later)
    assert [entry["share_id"] for entry in store.list()] == [long_lived]
    assert store.resolve(short) is None, "the two readers must agree on what is live"


def test_a_share_file_that_is_not_a_mapping_resets_instead_of_crashing(tmp_path):
    """The store is ours, but it sits on a mounted volume next to the models. Valid
    JSON of the wrong shape reached `.items()` and raised at import of the first share
    route — the same reset the unreadable-file path already handles."""
    path = tmp_path / "links.json"
    path.write_text('["not", "a", "mapping"]', encoding="utf-8")
    store = ShareStore(path)
    assert store.list() == []
    share_id, _ = store.create("model", "m", expires_hours=1)
    assert store.resolve(share_id) is not None, "the store stays usable after the reset"


def test_deleting_a_resource_revokes_every_link_to_it_and_only_those(tmp_path):
    """A link names its resource. Without revocation on delete, a later model or
    dataset under the same name was served by an old link to whoever held it --
    reproduced through the API before this change. Matched case-insensitively:
    on Windows "D.csv" reaches the file "d.csv", so a link made through that
    spelling points at the same file."""
    store = ShareStore(tmp_path / "links.json")
    doomed = [store.create("dataset", "d.csv", 1)[0], store.create("dataset", "D.csv", 1)[0]]
    kept = [store.create("dataset", "other.csv", 1)[0], store.create("model", "d.csv", 1)[0]]

    assert store.revoke_for("dataset", "d.csv") == 2
    assert all(store.resolve(sid) is None for sid in doomed)
    assert all(store.resolve(sid) is not None for sid in kept)
    # Persisted: a restart does not bring them back.
    assert all(ShareStore(tmp_path / "links.json").resolve(sid) is None for sid in doomed)
