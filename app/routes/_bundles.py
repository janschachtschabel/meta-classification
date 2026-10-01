"""The registry, over HTTP: resolve a model name, or hand a bundle back as a download.

Not a route module — it defines no endpoints. It exists because three route modules need
these two things and a route importing a sibling route makes one endpoint's module the
owner of another's policy (audit ARC-1): `share.py` took its download staging from
`models.py`, and `predict_bulk.py` took the exists/get/TOCTOU mapping from `predict.py`.

Both functions are HTTP mapping, which is why they live under `routes/` rather than in a
core module: they raise `HTTPException` and build a `FileResponse`, and `CLAUDE.md` keeps
FastAPI out of the modules the routes delegate *to*. What is not HTTP — placing the staging
file and cleaning it up on failure — moved to `Registry.stage_export`, beside `export_to`.
"""

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..classifier import ClassifierModel
from ..errors import UnsafeModelError
from ..registry import get_registry
from ..security import safe_name


def load_model(model_name: str) -> ClassifierModel:
    """The model that name refers to, or the right 4xx — never a 500."""
    safe_name(model_name, "model name")
    registry = get_registry()
    if not registry.exists(model_name):
        raise HTTPException(404, f"Model '{model_name}' not found.")
    try:
        return registry.get(model_name)
    except FileNotFoundError as exc:
        # Deleted in the window between exists() and get() (TOCTOU) -> 404, the
        # same clean response as a plainly missing model, never a 500.
        raise HTTPException(404, f"Model '{model_name}' not found.") from exc
    except UnsafeModelError as exc:
        # A bundle that exists but can't be safely loaded (corrupt/version drift)
        # is a client-visible 422, not a 500 that leaks internals.
        raise HTTPException(422, "Model bundle is invalid or unloadable.") from exc


def staged_zip_response(name: str) -> FileResponse:
    """Stream the bundle's archive back from a staging file on disk.

    The archive is not built in memory: a production bundle is 50-180 MB and the byte path
    peaked at 2.78x that (measured). `Registry.stage_export` owns where the file goes and
    why, and shares it between downloads of the same bundle; this owns the response that
    streams it and releases this download's hold once the body is sent.
    """
    path, release = get_registry().stage_export(name)
    try:
        return FileResponse(
            path, media_type="application/zip",
            # Starlette writes the header, RFC 5987-encoded (`filename*=utf-8''…`) where the
            # name is not plain ASCII: a header is Latin-1, and built by hand "Fächer–2026"
            # made every export of that model a 500 (audit 2026-09-30, S03). `safe_name`,
            # which runs on every path to here, keeps quotes and line breaks out (SEC-8).
            filename=f"{name}.zip",
            # Also after a client disconnects: uvicorn then drops the remaining body
            # silently, the response runs to its end, and the task runs.
            background=BackgroundTask(release),
        )
    except BaseException:
        # Ours until a response holds it: each failed attempt used to leave a whole bundle
        # copy behind, which only the next start's sweep removed.
        release()
        raise
