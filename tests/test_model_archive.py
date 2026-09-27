"""The member allowlist — `unpack`'s path-traversal defence (audit T-6, T-7).

`unpack`'s docstring calls the allowlist the thing that "kills path traversal, dotfiles and
Windows drive-relative names in one rule", and that claim is the security argument for the
whole import path. Every other guard in `unpack` was pinned — the zip bomb, a missing file, a
tampered member, three damage modes, the legacy no-manifest case — and nothing had ever fed it
an archive containing `../../etc/passwd`.

This file is also `model_archive`'s dedicated home (audit T-7): its other coverage is
scattered through `tests/test_training.py` (2,000+ lines) under names about importing and
exporting, and `tests/test_model_archive_limits.py` covers the expansion policy.
"""

import io
import json
import zipfile

import pytest

from app.errors import UnsafeModelError
from app.model_archive import ALLOWED_MEMBERS, REQUIRED_FILES, unpack

# Names an attacker would reach for. Every one of them is a *member name inside the zip*, which
# is the only place they could come from: the upload's own filename never touches a path.
TRAVERSALS = [
    "../../etc/passwd",
    "../config.json",
    "..\\..\\windows\\system32\\config.json",
    "/etc/passwd",
    "C:/Windows/system.ini",
    "C:config.json",          # Windows drive-relative: resolves against the CWD of that drive
    "subdir/config.json",
    "./config.json",
    ".ssh/authorized_keys",
    ".hidden",
    "config.json/../../evil",
]


def _members() -> dict[str, bytes]:
    """The smallest member set `unpack` accepts, so a test can add one name and nothing else."""
    files = {
        "config.json": json.dumps({"classes": ["uri:a"], "task_type": "multilabel"}).encode(),
        "head.skops": b"not-a-real-container",
        "vectorizer.skops": b"not-a-real-container",
        "vocabulary.json": json.dumps({"word": {}, "char": {}}).encode(),
    }
    assert set(files) == REQUIRED_FILES, "the minimal set drifted from REQUIRED_FILES"
    return files


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


@pytest.mark.parametrize("name", TRAVERSALS)
def test_an_archive_carrying_a_path_is_refused_before_anything_is_written(name):
    """The allowlist is a set of exact names, so any path — relative, absolute, dotted, or
    drive-relative — is simply not in it. Asserted per name rather than trusting that
    property, because the property is the whole defence."""
    members = _members()
    members[name] = b"payload"

    with pytest.raises(UnsafeModelError, match="[Uu]nexpected"):
        unpack(_zip(members))


@pytest.mark.parametrize("name", TRAVERSALS)
def test_a_traversal_name_is_not_in_the_allowlist(name):
    """The mechanism behind the test above, stated directly: if a future edit ever matched
    members by suffix or by `endswith`, several of these would start passing."""
    assert name not in ALLOWED_MEMBERS


def test_an_archive_of_only_the_required_members_is_accepted():
    """A guard that refuses real work is worse than none — and without this the tests above
    would pass just as well if `unpack` refused everything."""
    files = unpack(_zip(_members()))

    assert set(files) == REQUIRED_FILES
    assert files["head.skops"] == b"not-a-real-container"


def test_a_directory_entry_is_refused_too():
    """A zip can carry explicit directory entries; `extractall` would create them."""
    members = _members()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr("evil/", b"")

    with pytest.raises(UnsafeModelError):
        unpack(buffer.getvalue())


def test_the_allowlist_is_exact_names_and_nothing_else():
    """No globs, no prefixes, no separators: the property every test above rests on."""
    for name in ALLOWED_MEMBERS:
        assert "/" not in name and "\\" not in name and "*" not in name, name
        assert not name.startswith("."), name
