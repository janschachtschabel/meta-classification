"""The label-name sidecar: ``label_names.json`` in the data directory.

A plain ``{"<label uri>": "<display name>"}`` mapping. Training prefers it over the names
derived from a CSV's ``_DISPLAYNAME`` column, which a name holding the separator can shift
(see ``label_names.pair_names``). Split from ``prepare`` so the file has one home: training
reads it, an upload (``PUT /label-names``) writes it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from . import durability

logger = logging.getLogger(__name__)

FILE = "label_names.json"


def load(data_dir: Path) -> dict[str, str]:
    """The sidecar's names; ``{}`` when absent or unusable.

    Never fatal: correct display names are a reporting nicety, while a training run is
    expensive. A malformed file is logged and ignored rather than failing the run.
    """
    path = Path(data_dir) / FILE
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring %s: %r", FILE, exc)
        return {}
    if not isinstance(loaded, dict):
        logger.warning("Ignoring %s: expected a JSON object of uri -> name", FILE)
        return {}
    return {
        uri: name.strip()
        for uri, name in loaded.items()
        if isinstance(uri, str) and isinstance(name, str) and name.strip()
    }


def save(data_dir: Path, names: dict[str, str]) -> None:
    """Replace the sidecar whole, or leave it as it was: written beside it, synced, renamed
    into place. Written in place, a failure half-way left half a JSON document where the
    names were (audit 2026-09-30, W07 -- the same lesson as the fetch script)."""
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=directory, prefix=f".{FILE}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(sorted(names.items())), stream, ensure_ascii=False, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, directory / FILE)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    durability.sync_dir(directory)
