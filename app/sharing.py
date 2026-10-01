"""Persisted, expiring share links for exported models and datasets.

A share link maps a random id to a resource (model/dataset) and an expiry
timestamp. No file contents are stored; the link just authorizes a time-limited
download of an existing resource.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import threading
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from .errors import ShareStoreWriteError

logger = logging.getLogger("api_v3.sharing")


def _now() -> datetime:
    return datetime.now(UTC)


def _is_live(info: object) -> bool:
    """Is this link still valid? A malformed entry counts as dead.

    One definition for all three readers. It used to be written twice and forgotten
    once — ``list`` had no expiry check at all, so a link that died while the process
    was running was still shown as outstanding by the very screen an operator revokes
    from. Anything unparseable is dead rather than skipped, because a link whose
    expiry we cannot read is a link whose expiry we cannot enforce.
    """
    if not isinstance(info, dict):
        return False
    try:
        return datetime.fromisoformat(info["expires_at"]) > _now()
    except (KeyError, TypeError, ValueError):
        return False


class ShareStore:
    """Thread-safe JSON-backed store of share links."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._links: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            links = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(links, dict):
                # Valid JSON of the wrong shape: the file sits on a mounted volume,
                # and `.items()` on a list would raise at the first share route.
                raise ValueError(f"expected an object, got {type(links).__name__}")
        except (OSError, ValueError):
            # A corrupt/unreadable store resets to empty — but say so, don't lose
            # every active link silently.
            logger.warning("Discarding unreadable share-links file %s; all links reset.", self.path)
            return {}
        live = {key: value for key, value in links.items() if _is_live(value)}
        if len(live) != len(links):
            self._purge(live)
        return live

    def _persist(self, links: dict[str, dict]) -> None:
        """Write ``links`` as the store; the caller adopts them only once this returns.

        In that order since audit 2026-09-30, S05: a revocation that dropped the link in
        memory first answered 500 on a full volume, a second one 404 -- and after the next
        start the link served again, in time even a new model under the same name.

        Atomic (tmp + rename) so a crash mid-write cannot corrupt the store, matching the
        model-bundle publish discipline.
        """
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(links, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            # The path stays in the log; the answer names the setting to fix.
            logger.error("Could not write the share links to %s: %s", self.path, exc)
            raise ShareStoreWriteError(
                "The share links could not be saved, so nothing was changed. Check that "
                "APIV3_SHARE_LINKS_FILE is on a writable volume with free space, then retry."
            ) from exc

    def _purge(self, links: dict[str, dict]) -> None:
        """Drop expired links, written if possible: unlike a revocation, a purge that cannot
        be written is harmless -- every reader checks the expiry, and the next start purges
        again. `_persist` has logged the failure."""
        with contextlib.suppress(ShareStoreWriteError):
            self._persist(links)
        self._links = links

    def create(self, kind: str, name: str, expires_hours: int) -> tuple[str, str]:
        """Create a link; returns (share_id, expires_at_iso)."""
        expires_hours = max(1, min(expires_hours, 168))  # 1h .. 7 days
        share_id = secrets.token_urlsafe(12)
        expires_at = (_now() + timedelta(hours=expires_hours)).isoformat()
        with self._lock:
            links = {**self._links, share_id: {
                "kind": kind, "name": name, "expires_at": expires_at,
                # Free at creation and the one thing an overview cannot derive:
                # how long ago somebody handed this capability out.
                "created_at": _now().isoformat(),
            }}
            # Not handed out unless stored: it would work until the next start, then vanish.
            self._persist(links)
            self._links = links
        return share_id, expires_at

    def list(self) -> list[dict]:
        """Every live link, newest expiry last — the overview an operator revokes from.

        The share id is the capability itself, so this is admin-only at the route.
        ``created_at`` is ``None`` for links created before it was recorded.

        Expired entries are filtered, not deleted: a listing is a read, and ``resolve``
        already purges the one link it was asked about. What is left is cleared at the
        next start — the store is process-local, so a restart is the only writer that
        can see them all at once anyway.
        """
        with self._lock:
            return sorted(
                (
                    {"share_id": share_id, "kind": info.get("kind"), "name": info.get("name"),
                     "created_at": info.get("created_at"), "expires_at": info.get("expires_at")}
                    for share_id, info in self._links.items()
                    if _is_live(info)
                ),
                key=lambda entry: entry["expires_at"] or "",
            )

    def revoke(self, share_id: str) -> bool:
        """Withdraw a link; ``False`` if it was already gone.

        A link cannot be un-shared once downloaded, but it can be stopped from being
        used again — until now the only way was editing the JSON on the volume.
        """
        with self._lock:
            if share_id not in self._links:
                return False
            links = {key: info for key, info in self._links.items() if key != share_id}
            self._persist(links)
            self._links = links
            return True

    def revoke_for(self, kind: str, name: str) -> int:
        """Withdraw every link to one resource; returns how many.

        Called when the resource is deleted. A link names its resource rather than
        a version of it, so without this a later model or dataset under the same
        name was served by an old link to whoever held it. Compared case-
        insensitively: on Windows "D.csv" reaches the file "d.csv", so a link made
        through that spelling points at the same file (on Linux it revokes a link
        to a different file, which fails safe).
        """
        key = name.casefold()
        with self._lock:
            doomed = {
                share_id for share_id, info in self._links.items()
                if info.get("kind") == kind and str(info.get("name", "")).casefold() == key
            }
            if doomed:
                links = {share_id: info for share_id, info in self._links.items() if share_id not in doomed}
                self._persist(links)
                self._links = links
            return len(doomed)

    def resolve(self, share_id: str) -> dict | None:
        """Return link info, or None if missing/expired (expired ones are purged)."""
        with self._lock:
            info = self._links.get(share_id)
            if info is None:
                return None
            if not _is_live(info):
                self._purge({key: value for key, value in self._links.items() if key != share_id})
                return None
            return dict(info)


@lru_cache
def get_share_store() -> ShareStore:
    from .settings import get_settings

    return ShareStore(get_settings().share_links_file)
