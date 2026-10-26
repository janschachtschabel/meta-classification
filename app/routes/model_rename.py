"""Renaming a model: correct a typo in its name, or name an import after what it is.

A route module of its own -- routes/models.py serves the bundles; this changes what one is
called. The rules for what a name may be live in ``app.model_names``; the move in
``Registry.rename``.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import Path as PathParam

from .. import model_names
from ..limiter import default_limit, limiter
from ..registry import BACKUP_SUFFIX, get_registry, is_backup_name
from ..schemas.training import RenameRequest
from ..security import require_role, safe_name
from ..sharing import get_share_store
from ._jobs import refuse_while_a_run_holds

router = APIRouter(tags=["Models"])

ModelName = Annotated[str, PathParam(description="The model's current name.")]


@router.post("/models/{model_name}/rename", summary="Rename a model")
@limiter.limit(default_limit)
async def rename_model(
    request: Request, model_name: ModelName, body: RenameRequest,
    _: str = Depends(require_role("admin")),
) -> dict:
    """Give a model another name. Its bundle, metrics and documentation stay as they are;
    the run history and the corrections keep the name they were recorded under.

    **The old name's share links are revoked** (`revoked_links` says how many): a link
    resolves by name, so it would otherwise serve whatever model is later given the old
    name. They are revoked BEFORE the move, so a link store that cannot be written (503)
    leaves the model where it was.

    The new name must not be used by any other model as its name or display name, in any
    letter case (409); a change of case alone is allowed. Refused (409) while a run of
    either name is running or queued, or an import of the new name is in progress.
    **Auth:** admin · rate limit active.
    """
    safe_name(model_name, "model name")
    new = body.new_name.strip()
    safe_name(new, "model name")
    if is_backup_name(new):
        raise HTTPException(400, f"A model name must not end in '{BACKUP_SUFFIX}': that marks the "
                                 "backups the label repair keeps, which no route serves.")
    if new == model_name:
        raise HTTPException(400, "The new name is the model's current name.")
    registry = get_registry()
    if not registry.exists(model_name):
        raise HTTPException(404, f"Model '{model_name}' not found.")
    if owner := await asyncio.to_thread(model_names.taken_by, registry, new, other_than=model_name):
        raise HTTPException(409, f"'{new}' is already the name or display name of model '{owner}'.")
    for name in (model_name, new):
        refuse_while_a_run_holds(name)
    try:
        # Held like an import of the new name, so a training or import of it is refused
        # while the bundle moves rather than publishing over it afterwards.
        with registry.importing(new):
            revoked = get_share_store().revoke_for("model", model_name)
            await asyncio.to_thread(registry.rename, model_name, new)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    except FileExistsError as exc:
        raise HTTPException(409, f"'{new}' is taken, or an import of it is in progress.") from exc
    return {"name": new, "revoked_links": revoked}
