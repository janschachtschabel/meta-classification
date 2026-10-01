"""The label-name sidecar over HTTP: read it, or replace it with an upload.

Since the image carries the label repairs (audit 2026-09-30, B06), the only way to get
`label_names.json` into a container was to copy it into the volume -- `docker cp`,
`kubectl cp` -- which not every cluster lets an editor do (improvement 9). An upload only:
the app fetches no URL, the SSRF boundary dataset and model imports keep as well.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, RootModel, StringConstraints

from .. import label_sidecar
from ..limiter import default_limit, limiter
from ..security import require_role
from ..settings import get_settings

router = APIRouter(tags=["Datasets"])

LabelUri = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
LabelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class LabelNames(RootModel[dict[LabelUri, LabelName]]):
    """`{"<label uri>": "<display name>"}`, at least one: an empty mapping written over a good
    file empties every display name the next training would use (audit 2026-09-30, W07)."""

    root: dict[LabelUri, LabelName] = Field(min_length=1, max_length=100_000)


@router.get("/label-names", summary="The label display names training uses")
async def get_label_names(_: str = Depends(require_role("readonly"))) -> dict:
    """The current `label_names.json` of the data directory: `{"labels": n, "names": {...}}`,
    empty when there is none. Training prefers these names over the ones a CSV's
    `_DISPLAYNAME` column yields. **Auth:** readonly."""
    names = await asyncio.to_thread(label_sidecar.load, get_settings().data_dir)
    return {"labels": len(names), "names": names}


@router.put("/label-names", summary="Replace the label display names training uses")
@limiter.limit(default_limit)
async def put_label_names(
    request: Request, body: LabelNames, _: str = Depends(require_role("admin")),
) -> dict:
    """Replace `label_names.json` whole, or leave it as it was: written beside it and renamed
    into place. Applies to trainings from now on; a bundle already trained keeps its names
    until `scripts/patch_bundle_labels.py` repairs it. **Auth:** admin · rate limit active.
    """
    try:
        await asyncio.to_thread(label_sidecar.save, get_settings().data_dir, body.root)
    except OSError as exc:
        raise HTTPException(503, "The label names could not be written; the previous ones are "
                                 "unchanged.") from exc
    return {"labels": len(body.root)}
