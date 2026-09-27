"""Importing a model does not hold the archive, or its members, in memory (audit API-5).

The export path was rewritten to stream for exactly this reason, with a measured 2.78x peak
recorded in its docstring; the import path kept both buffers. `read_upload_capped` joined the
chunks and the joined copy — twice the cap, 400 MB at the default — and `unpack` then
materialised every member on top of that, inside the process that serves predictions.

SEC-5's 1 GiB ceiling bounds the worst case, which is why this is a Medium and not a High. It
does not make the ordinary case cheap: a legitimate 180 MB bundle cost ~360 MB of upload buffer
plus its own size again in member bytes, on an 8 GB container.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from app.errors import UnsafeModelError
from app.model_archive import CARD_FILE, MANIFEST_FILE, REQUIRED_FILES, unpack, unpack_into


def _members() -> dict[str, bytes]:
    return {
        "config.json": json.dumps({"classes": ["uri:a"], "task_type": "multilabel"}).encode(),
        "head.skops": b"head-bytes",
        "vectorizer.skops": b"vectorizer-bytes",
        "vocabulary.json": json.dumps({"word": {}, "char": {}}).encode(),
    }


def _archive(tmp_path: Path, *, with_manifest: bool = True, tamper: str | None = None) -> Path:
    members = _members()
    if with_manifest:
        # The card is listed too: `unpack` verifies every member except the manifest itself,
        # so a real manifest covers the card and a fixture that omits it is not a valid
        # archive — which is what this fixture got wrong first.
        members[CARD_FILE] = b"# card\n"
        digests = {name: hashlib.sha256(payload).hexdigest()
                   for name, payload in members.items()}
        members[MANIFEST_FILE] = json.dumps({"files": digests}).encode()
    if tamper:
        members[tamper] = members[tamper] + b"extra"
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_streaming_writes_exactly_the_installable_members(tmp_path):
    """The transport artifacts are verified and then dropped — keeping them on disk would make
    the next export ship a stale copy beside the fresh one."""
    target = tmp_path / "staged"
    target.mkdir()

    unpack_into(_archive(tmp_path), target)

    assert {p.name for p in target.iterdir()} == REQUIRED_FILES
    assert (target / "head.skops").read_bytes() == b"head-bytes"


def test_streaming_and_buffering_install_the_same_bytes(tmp_path):
    """One validation path, two ways of moving the bytes: the streamed result has to equal
    what the in-memory path returns, member for member."""
    archive_path = _archive(tmp_path)
    target = tmp_path / "staged"
    target.mkdir()

    buffered = unpack(archive_path.read_bytes())
    unpack_into(archive_path, target)

    assert {p.name for p in target.iterdir()} == set(buffered)
    for name, payload in buffered.items():
        assert (target / name).read_bytes() == payload, name


def test_a_tampered_member_is_refused_and_nothing_is_left_behind(tmp_path):
    """Checksums are verified while streaming, so the answer must be the same as before —
    and a refusal must not leave half a bundle staged for the caller to publish."""
    target = tmp_path / "staged"
    target.mkdir()

    with pytest.raises(UnsafeModelError, match="checksum"):
        unpack_into(_archive(tmp_path, tamper="head.skops"), target)

    assert not list(target.iterdir()), "a refused import left files behind"


def test_an_archive_without_a_manifest_still_installs(tmp_path):
    """Bundles exported before the manifest existed have none; its absence is not an error."""
    target = tmp_path / "staged"
    target.mkdir()

    unpack_into(_archive(tmp_path, with_manifest=False), target)

    assert {p.name for p in target.iterdir()} == REQUIRED_FILES


def test_the_member_allowlist_still_holds_on_the_streaming_path(tmp_path):
    """The path-traversal defence must not have a second door. `unpack_into` writes to disk,
    so here the allowlist is the difference between a refusal and a file outside the target."""
    members = _members()
    members["../escaped.json"] = b"payload"
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    target = tmp_path / "staged"
    target.mkdir()

    with pytest.raises(UnsafeModelError, match="[Uu]nexpected"):
        unpack_into(path, target)

    assert not (tmp_path / "escaped.json").exists(), "a member escaped the target directory"
    assert not list(target.iterdir())


def test_a_zip_bomb_is_refused_before_a_byte_is_written(tmp_path):
    """The whole point of checking the declared sizes first: the guard has to run before the
    extraction it is protecting."""
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.writestr(name, b"\0" * (80 * 1024 * 1024))
    target = tmp_path / "staged"
    target.mkdir()

    with pytest.raises(UnsafeModelError, match="possible zip bomb"):
        unpack_into(path, target)

    assert not list(target.iterdir())


def test_a_corrupt_archive_reads_as_a_broken_file_not_a_server_fault(tmp_path):
    path = tmp_path / "broken.zip"
    path.write_bytes(b"this is not a zip")
    target = tmp_path / "staged"
    target.mkdir()

    with pytest.raises(UnsafeModelError, match="[Nn]ot a valid zip"):
        unpack_into(path, target)


def test_the_route_spools_the_upload_instead_of_joining_it():
    """Asserted on the source: the difference is a peak, which a functional test cannot see.
    `read_upload_capped` holds the chunks and the joined copy — twice the cap — before a byte
    reaches disk, and `spool_upload_capped` exists for precisely this case."""
    source = (Path(__file__).resolve().parents[1] / "app" / "routes" / "models.py").read_text(
        encoding="utf-8")
    body = source[source.index("async def import_model"):]

    assert "spool_upload_capped" in body, "the import still joins the whole archive in memory"
    assert "read_upload_capped(" not in body, "the buffering call is still there"


def test_the_registry_never_holds_every_member_at_once():
    """`unpack` returning a dict of every member is what put the whole bundle in RAM a second
    time. The installing path must use the streaming one."""
    source = (Path(__file__).resolve().parents[1] / "app" / "registry.py").read_text(
        encoding="utf-8")

    assert "unpack_into" in source, "the registry still installs from a dict of member bytes"


def test_the_byte_based_entry_point_still_works(tmp_path):
    """Eleven tests and any external caller pass bytes; that contract stays, and it must go
    through the same validation and staging as the streaming one."""
    from app.model_archive import unpack as unpack_bytes

    files = unpack_bytes(_archive(tmp_path).read_bytes())

    assert set(files) == REQUIRED_FILES
    assert MANIFEST_FILE not in files and CARD_FILE not in files


def test_streaming_reads_the_archive_without_loading_it(tmp_path):
    """A zip opened from a path reads members lazily; opened from BytesIO the whole archive is
    already resident. Pinned so a well-meaning edit does not reintroduce the buffer."""
    import app.model_archive as archive_mod

    seen: list[object] = []
    real = zipfile.ZipFile

    class _Watched(real):  # type: ignore[misc,valid-type]
        def __init__(self, file, *args, **kwargs):
            seen.append(file)
            super().__init__(file, *args, **kwargs)

    archive_mod.zipfile.ZipFile = _Watched  # type: ignore[misc]
    try:
        target = tmp_path / "staged"
        target.mkdir()
        unpack_into(_archive(tmp_path), target)
    finally:
        archive_mod.zipfile.ZipFile = real  # type: ignore[misc]

    assert seen and not any(isinstance(item, io.BytesIO) for item in seen), (
        "unpack_into opened the archive from memory instead of from the file"
    )
