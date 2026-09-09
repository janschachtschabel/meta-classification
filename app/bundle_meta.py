"""Typed, forgiving reads of a bundle's own JSON documents.

``config.json`` and ``metrics.json`` travel *inside* importable bundles, so every
value in them is foreign input: an older exporter, a hand edit on the volume, or a
crafted upload. The serving path already read them this way (see ``per_label_f1``,
which moved here from ``model_io``); the reporting paths did not, and formatted the
values directly — so one wrong type took down the export that packs the model card,
and with it the unauthenticated share download that serves the same bytes.

Reporting data is data that may legitimately be missing: bundles predating a field
have none either. Dropping an unusable value therefore costs a line in a table,
which is why every accessor here degrades to "absent" instead of raising.
"""

from __future__ import annotations

import math


def as_mapping(value: object) -> dict:
    """A sub-document of a bundle document, or an empty one if it is not a mapping."""
    return value if isinstance(value, dict) else {}


def as_names(value: object) -> list[str]:
    """The strings in a list of names (label URIs, column names); the rest drops out."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def as_number(value: object) -> float | None:
    """A value that can be formatted as a number, else ``None``.

    ``bool`` is refused although it is an ``int``: JSON ``true`` would otherwise print
    as "1.0000" and read like something that was measured. NaN and infinity are refused
    for the same reason — and because neither survives a JSON response.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def as_count(value: object) -> int | None:
    """A row count from a bundle document, or ``None``.

    Kept an ``int`` rather than coerced through :func:`as_number`: a count is what it
    is, and turning 12 into 12.0 would change the shape of an API response.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def per_label_f1(metadata: dict) -> dict[str, float]:
    """Per-label F1 out of a bundle's metrics document; ``{}`` if absent or unusable.

    A non-mapping, a non-numeric score or a NaN would otherwise reach the model and
    break every later prediction (a crash, or a response body that is not valid JSON),
    and reach the card renderer, which sorts and formats these values.
    """
    scores = as_mapping(metadata.get("metrics")).get("per_label_f1")
    if not isinstance(scores, dict):
        return {}
    usable = {uri: as_number(score) for uri, score in scores.items() if isinstance(uri, str)}
    return {uri: score for uri, score in usable.items() if score is not None}
