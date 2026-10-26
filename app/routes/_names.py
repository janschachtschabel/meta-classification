"""The 409 every route gives a name another model already answers to.

Names and display names exclude each other across models (``app.model_names``), and a name
comes into being four ways: a training, an import, a rename, a documentation save. One answer
for all four, so none of them is the way round the rule.
"""

from __future__ import annotations

import asyncio

from fastapi import HTTPException

from .. import model_names
from ..registry import Registry


async def refuse_a_taken_name(registry: Registry, candidate: str | None, *, other_than: str) -> None:
    """409 if a model other than ``other_than`` answers to ``candidate`` -- as its name or its
    display name, in any letter case. ``None`` or empty: nothing to check."""
    if not candidate:
        return
    if owner := await asyncio.to_thread(model_names.taken_by, registry, candidate, other_than=other_than):
        raise HTTPException(409, f"'{candidate}' is already the name or display name of model '{owner}'.")
