"""Share links: list, revoke, and the public download a link stands for.

Moved out of routes/models.py: a link serves a model OR a dataset, so the
routes that manage links are not model management. The export routes that
CREATE a link stay with their resource (models.py, datasets.py). Same tag as
before, so the API documentation is unchanged.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

from .. import data as data_mod
from ..limiter import default_limit, export_limit, limiter
from ..registry import get_registry
from ..security import require_role
from ..settings import Settings, get_settings
from ..sharing import get_share_store
from .models import staged_zip_response

router = APIRouter(tags=["Models"])


@router.get("/share", summary="List active share links")
async def list_share_links(_: str = Depends(require_role("admin"))) -> list[dict]:
    """Every share link that has not expired: id, kind, name, created and expiry.

    A share link is a bearer capability — the id IS the authorization — so this
    listing hands out the secrets themselves and stays admin-only, unlike the
    download route it describes. **Auth:** admin.
    """
    return get_share_store().list()


@router.delete("/share/{share_id}", summary="Revoke a share link")
@limiter.limit(default_limit)
async def revoke_share_link(
    request: Request, share_id: str, _: str = Depends(require_role("admin")),
) -> dict:
    """Withdraw a share link before it expires.

    What was already downloaded cannot be recalled, but the link stops working — the
    point of an expiring capability you can end early. **Auth:** admin · rate limit
    active.
    """
    if not get_share_store().revoke(share_id):
        raise HTTPException(404, "Share link not found or already expired.")
    return {"status": "revoked", "share_id": share_id}


@router.get("/share/{share_id}", summary="Download a shared resource")
@limiter.limit(export_limit)  # public endpoint: throttle share-id brute-forcing
async def download_shared(
    request: Request,
    share_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Download a previously exported share resource (model ZIP or dataset CSV).

    The share link is a **bearer capability**: the unguessable id (`secrets.token_urlsafe`,
    96 bits) plus its expiry ARE the authorization, so no API key is required — a link
    can be handed to someone without a key. Creating links stays admin-only; guard the
    id like a secret. Expired or unknown links return 404. **Auth:** none (bearer link).
    """
    info = get_share_store().resolve(share_id)
    if info is None:
        raise HTTPException(404, "Share link not found or expired.")
    if info["kind"] == "model":
        registry = get_registry()
        if not registry.exists(info["name"]):
            raise HTTPException(404, "Model no longer exists.")
        # Same blocking-zip offload as the authenticated export route.
        return await asyncio.to_thread(staged_zip_response, info["name"])
    dataset_path = settings.data_dir / info["name"]
    if not dataset_path.exists():
        raise HTTPException(404, "Dataset no longer exists.")
    # Same gzip/CSV distinction as the authenticated export route: a share link is the path a
    # recipient WITHOUT a key uses, so it is the one most likely opened in a browser — where
    # a text/csv header on gzip bytes yields a decompressed file saved under its .gz name.
    media_type = "application/gzip" if data_mod.is_gzipped(info["name"]) else "text/csv"
    return FileResponse(dataset_path, filename=info["name"], media_type=media_type)
