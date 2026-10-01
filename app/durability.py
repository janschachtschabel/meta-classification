"""Making a write survive a power cut: sync the data, then the directory entry.

A rename is atomic but not durable on its own. Without an fsync, a file's blocks can still be
in the page cache when the rename reaches the disk, and after a power cut the name points at
an empty file -- ext4 with delayed allocation does exactly that for new files. Nothing here was
synced, so a published bundle could come back with empty skops files (audit 2026-09-30, R13).

The order is the guarantee: sync what a rename will expose, rename, then sync the directory
that holds the new name. A stdlib leaf.
"""

from __future__ import annotations

import os
from pathlib import Path


def sync_file(path: Path) -> None:
    """Flush ``path``'s data to disk. Opened for writing: Windows refuses to flush a handle
    that may not write."""
    with Path(path).open("rb+") as handle:
        os.fsync(handle.fileno())


def sync_dir(path: Path) -> None:
    """Flush ``path``'s entries -- a rename in it, or the names of the files it holds.

    A no-op on Windows, which cannot open a directory to sync it; NTFS journals its metadata.
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree(directory: Path) -> None:
    """Flush every file directly in ``directory``, then the directory's own entries."""
    for path in Path(directory).iterdir():
        if path.is_file():
            sync_file(path)
    sync_dir(directory)
