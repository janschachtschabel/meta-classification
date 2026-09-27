"""Optional bounds on a listing, declared once (audit API-8).

Not a route module — it defines no endpoints.

The listings answer bare JSON arrays, which is the shape every existing client parses. Wrapping
them in an envelope with a total and a cursor would be the textbook answer and would break all
of them, for a finding whose severity is Low. So the shape stays, and the caller gains a way to
ask for less: absent parameters mean the whole list, exactly as before.

The bounds are *validated*, not clamped. `/train/history` clamped — `max(1, min(limit, 200))` —
so a request for 1 000 got 200 back with nothing saying so, and a caller could not tell a short
page from the end of the data. 422 says which number was wrong.
"""

from __future__ import annotations

from typing import Annotated, TypeVar

from fastapi import Query

# 2000 rather than "unbounded": a limit is a promise about the response, and a caller asking for
# more than the largest plausible listing has probably computed it wrong. Omitting the parameter
# is still the way to say "everything".
Limit = Annotated[int | None, Query(
    ge=1, le=2000,
    description="How many entries to return. Omit for all of them; out-of-range is a 422.",
)]
Offset = Annotated[int, Query(
    ge=0,
    description="How many entries to skip before the first returned one (needs a stable order).",
)]

T = TypeVar("T")


def page(items: list[T], limit: int | None, offset: int) -> list[T]:
    """``items`` narrowed to one page. An offset past the end is an empty page, not an error —
    that is how a caller walking pages learns it has reached the end."""
    if offset:
        items = items[offset:]
    return items if limit is None else items[:limit]
