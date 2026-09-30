"""Descriptive metadata for a text: title, description, keywords.

Thin, like every route module: the generators live in ``app.metadata``. Its own module rather
than a flag on ``/predict`` — this reads no model, applies no threshold and answers for a text
that no model was ever trained for, so the two have nothing in common but the word "text".
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from ..concurrency import metadata_slots
from ..limiter import limiter, predict_limit
from ..metadata import MetadataSettings, generate
from ..schemas import MetadataRequest
from ..security import require_role

router = APIRouter(tags=["Metadata"])


def _truncate(text: str) -> str:
    # Same 200 characters as /predict echoes: enough to recognise the row, not a copy of the
    # input back over the wire.
    return (text[:200] + "...") if len(text) > 200 else text


def _generate_all(body: MetadataRequest) -> dict:
    settings = MetadataSettings(
        title_max=body.title_max, desc_max=body.desc_max, n_keywords=body.n_keywords
    )
    results = []
    for text in body.texts:
        proposal = generate(text, settings)
        results.append({
            "text": _truncate(text),
            "title": proposal.title,
            "description": proposal.description,
            "keywords": proposal.keywords,
        })
    return {"results": results}


@router.post("/metadata", summary="Propose a title, a description and keywords for texts")
@limiter.limit(predict_limit)
async def describe(
    request: Request, body: MetadataRequest, _: str = Depends(require_role("readonly"))
) -> dict:
    """Derive descriptive metadata from the text itself, to round out what `/predict` classifies.

    Classification says what a text is about in a trained vocabulary; this says what it is, in
    its own words, so an item can be stored as a complete record rather than as labels alone.
    The two are independent — **no model is involved here**, nothing is loaded from the volume,
    and a text from a domain no model was trained on is answered just as well.

    Per text it returns `title`, `description` and `keywords` (best first), each within the
    budgets from the request. **All three are extractive: every word occurs in the input.**
    They are proposals for an editor to check, not finished metadata — the comparison behind
    the chosen methods found that small generative models were no better and invented facts,
    while these cannot.

    - **Title** — a real heading if the text opens with one, otherwise the template
      `Keyword1: Keyword2 und Keyword3` from its three strongest keywords.
    - **Description** — the first usable sentences up to `desc_max`, whole sentences only.
      Headings, list items, task instructions ("Beschreibe …") and page chrome are skipped, so
      it starts at the first real sentence rather than at the top of the file.
    - **Keywords** — noun phrases weighted by how often they occur in this text against how
      rare they are in German.

    Empty fields are an answer: a text that yields nothing (blank, or only boilerplate) comes
    back with empty strings and an empty list, not an error.

    **German.** The stopwords, the noun-phrase rules and the word frequencies are German;
    other languages will return something, but nothing about it was measured.

    **Markup is removed, line breaks are not.** HTML and Markdown go, and so do the bodies of
    `script`, `style`, `nav` and `footer` — code and page chrome rather than prose. A
    block-level tag counts as a line break, so a scraped page can be sent as it is. What the
    generators cannot recover is structure that is already gone: a text flattened to a single
    line has no heading to find and no paragraph to start at, so keep the line breaks the
    source had.

    **Auth:** readonly.
    """
    # CPU-bound (12-22 ms per ordinary text, seconds for long ones) in a single-worker
    # process: on the event loop that is time in which nothing else, /health included, gets
    # answered -- and in the worker pool, a thread every other route shares.
    slots = metadata_slots()
    if not slots.try_acquire():
        raise HTTPException(
            503,
            f"Too many metadata requests running ({slots.capacity}). Retry shortly.",
            headers={"Retry-After": "10"},
        )
    try:
        return await asyncio.to_thread(_generate_all, body)
    finally:
        slots.release()
