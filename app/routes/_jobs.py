"""The job runner, over HTTP: accept a run, or say why it was refused.

Not a route module — it defines no endpoints. `/train` and `POST /models/{name}/evaluate`
had these twelve lines each, differing only in the value of one response key (audit ARC-5),
so how a run is queued and how a refusal is reported had to be changed in two places.
"""

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from ..jobs import job_runner


def start_or_queue(
    target: Callable[..., Any], *args: object, model_name: str, request: dict, profile: str,
    kind: str = "training",
) -> dict:
    """Submit the run and describe where it landed.

    ``profile`` is what the caller should see this run recorded under: a training's resolved
    profile name, or simply ``"evaluation"`` — the one thing the two callers differ on.

    :raises HTTPException: 409 when the queue is full or that name is already running or
        queued. Both are the caller's to resolve and both are conflicts with what the
        server is already doing, which is what 409 says.
    """
    try:
        position = job_runner.submit(
            target, *args, model_name=model_name, request=request, kind=kind,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "status": "started" if position == 0 else "queued",
        "model_name": model_name,
        "profile": profile,
        "status_url": "/train/status",
        "queue_position": position,
    }
