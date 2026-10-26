"""What a model is called: its name, its display name, and that neither is ever used twice.

A model's name is its bundle directory; its display name is a line of its documentation (the
``info`` block). People pick models by either, so a name that one model uses in any form --
name or display name, in any letter case -- is not available to another one. Two models whose
names differ only in case could not even both exist on Windows.
"""

from __future__ import annotations

from typing import Protocol

from .errors import UnsafeModelError


class _Models(Protocol):
    def list(self) -> list[str]: ...
    def info(self, name: str) -> dict: ...
    def update_info(self, name: str, info: dict) -> dict: ...


def display_name(models: _Models, name: str) -> str | None:
    """The display name ``name`` documents, if any -- read forgivingly, like all bundle data.

    An unreadable bundle (``UnsafeModelError``) answers to no display name: the check reads
    every model, and one broken bundle must not block renaming or documenting the others.
    """
    try:
        info = models.info(name).get("metadata", {}).get("info") or {}
    except (OSError, ValueError, KeyError, UnsafeModelError):
        return None
    value = info.get("display_name") if isinstance(info, dict) else None
    # Trimmed as the request model trims it: a bundle's documentation is foreign data.
    return (value.strip() or None) if isinstance(value, str) else None


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


def release_taken_display_name(models: _Models, name: str) -> dict | None:
    """After an import: drop the display name the archive brought if another model already
    answers to it, keeping the rest of the documentation; returns what was dropped, or ``None``.

    Dropped rather than refused: the model is what the import is for, and the same model
    arriving on a second server is the common way to meet its own display name there. The
    caller reports it, so a display name never disappears silently.
    """
    shown = display_name(models, name)
    if shown is None:
        return None
    owner = taken_by(models, shown, other_than=name)
    if owner is None:
        return None
    info = dict(models.info(name).get("metadata", {}).get("info") or {})
    info.pop("display_name", None)
    models.update_info(name, info)
    return {"display_name": shown, "taken_by": owner}
