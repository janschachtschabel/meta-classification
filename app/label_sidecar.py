"""The label-name sidecar: ``label_names.json`` in the data directory.

A plain ``{"<label uri>": "<display name>"}`` mapping. Training prefers it over the names
derived from a CSV's ``_DISPLAYNAME`` column, which a name holding the separator can shift
(see ``label_names.pair_names``). Split from ``prepare`` so the file has one home: training
reads it there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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
