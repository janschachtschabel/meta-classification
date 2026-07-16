"""Persisted, expiring share links for exported models and datasets.

A share link maps a random id to a resource (model/dataset) and an expiry
timestamp. No file contents are stored; the link just authorizes a time-limited
download of an existing resource.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("api_v3.sharing")


def _now() -> datetime:
    return datetime.now(timezone.utc)


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
        except (OSError, ValueError):
            # A corrupt/unreadable store resets to empty — but say so, don't lose
            # every active link silently.
            logger.warning("Discarding unreadable share-links file %s; all links reset.", self.path)
            return {}
        now = _now()
        live: dict[str, dict] = {}
        for key, value in links.items():
            try:
                if datetime.fromisoformat(value["expires_at"]) > now:
                    live[key] = value
            except (KeyError, TypeError, ValueError):
                continue  # skip one malformed entry rather than failing the whole store
        if len(live) != len(links):
            self._links = live
            self._persist()
        return live

    def _persist(self) -> None:
        # Atomic write (tmp + rename) so a crash mid-write cannot corrupt the store,
        # matching the model-bundle publish discipline.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._links, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def create(self, kind: str, name: str, expires_hours: int) -> tuple[str, str]:
        """Create a link; returns (share_id, expires_at_iso)."""
        expires_hours = max(1, min(expires_hours, 168))  # 1h .. 7 days
        share_id = secrets.token_urlsafe(12)
        expires_at = (_now() + timedelta(hours=expires_hours)).isoformat()
        with self._lock:
            self._links[share_id] = {"kind": kind, "name": name, "expires_at": expires_at}
            self._persist()
        return share_id, expires_at

    def resolve(self, share_id: str) -> dict | None:
        """Return link info, or None if missing/expired (expired ones are purged)."""
        with self._lock:
            info = self._links.get(share_id)
            if info is None:
                return None
            if datetime.fromisoformat(info["expires_at"]) <= _now():
                del self._links[share_id]
                self._persist()
                return None
            return dict(info)


@lru_cache
def get_share_store() -> ShareStore:
    from .settings import get_settings

    return ShareStore(get_settings().share_links_file)
