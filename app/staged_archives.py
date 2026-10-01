"""One staged archive per bundle state, shared by every download of it in flight.

An export is packed into a file beside the bundles and streamed from there -- a production
bundle is too large to build in memory. Each download used to pack its own full copy and keep
it until its client had read the last byte: twelve connections that never read held twelve
copies, and with the chart's defaults (a 5 Gi volume, bundles around 250 MB) about twenty
filled the volume through a public share link, after which every write failed -- trainings,
feedback, new links (audit 2026-09-30, S02). Two downloads of one bundle state are the same
bytes, so they share one file, and the last of them to finish deletes it.

A stdlib leaf: the registry decides what a bundle state is and packs it under its disk lock;
this module only counts who still reads which file.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class _Archive:
    path: Path
    readers: int = 0


class StagedArchives:
    """Staged archives in ``directory``, keyed by the state they were packed from.

    The files are hidden ``.export-*.zip.tmp`` names, which no route serves and the registry's
    startup sweep removes: a process that dies mid-download leaks nothing past its restart.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._lock = threading.Lock()
        self._archives: dict[Hashable, _Archive] = {}

    def acquire(self, key: Hashable, write: Callable[[BinaryIO], None]) -> tuple[Path, Callable[[], None]]:
        """The archive packed from ``key``'s state, and the call that says it is no longer read.

        ``write`` packs it, and runs only when no download holds one for this key -- under this
        object's lock, so a second download of the same state waits for the first packing
        rather than starting its own. Exports already take turns on the registry's disk lock,
        which the caller holds around this call, so the wait costs nothing new.

        The release counts once however often it is called: a response that fails to build is
        released by its builder, and must not then take the file away from another download.
        """
        with self._lock:
            archive = self._archives.get(key)
            if archive is None:
                archive = _Archive(self._pack(write))
                self._archives[key] = archive
            archive.readers += 1
        released = False

        def release() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                archive.readers -= 1
                if archive.readers:
                    return
                if self._archives.get(key) is archive:
                    del self._archives[key]
            # Outside the lock: a download starting now packs a new file under a new name.
            archive.path.unlink(missing_ok=True)

        return archive.path, release

    def _pack(self, write: Callable[[BinaryIO], None]) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        handle, staged = tempfile.mkstemp(prefix=".export-", suffix=".zip.tmp", dir=self._dir)
        os.close(handle)
        path = Path(staged)
        try:
            with path.open("wb") as stream:
                write(stream)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path
