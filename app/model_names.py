"""What a model is called: its name, its display name, and that neither is ever used twice.

A model's name is its bundle directory; its display name is a line of its documentation (the
``info`` block). People pick models by either, so a name that one model uses in any form --
name or display name, in any letter case -- is not available to another one. Two models whose
names differ only in case could not even both exist on Windows.
"""

from __future__ import annotations

from typing import Protocol


class _Models(Protocol):
    def list(self) -> list[str]: ...
    def info(self, name: str) -> dict: ...


def display_name(models: _Models, name: str) -> str | None:
    """The display name ``name`` documents, if any -- read forgivingly, like all bundle data."""
    try:
        info = models.info(name).get("metadata", {}).get("info") or {}
    except (OSError, ValueError, KeyError):
        return None
    value = info.get("display_name") if isinstance(info, dict) else None
    return value if isinstance(value, str) and value else None


def taken_by(models: _Models, candidate: str, *, other_than: str) -> str | None:
    """The model that already answers to ``candidate`` -- by name or display name, in any
    letter case -- or ``None``. ``other_than`` is the model being renamed or documented: it may
    keep, or change the case of, what it already is."""
    key = candidate.casefold()
    for name in models.list():
        if name == other_than:
            continue
        if name.casefold() == key or (display_name(models, name) or "").casefold() == key:
            return name
    return None
