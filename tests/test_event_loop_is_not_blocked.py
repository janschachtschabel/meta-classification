"""No route handler does blocking work on the event loop.

The app is single-worker by contract, so the loop is the whole server: anything that
holds it holds `/health`, `/metrics` and every request in flight. The heavy routes were
offloaded one by one (export, import, predict, the CSV reads), but the *cheap-looking*
readers were left calling straight through — and several of them take
``Registry._disk_lock``, which `export_to` holds for seconds while it compresses a 180 MB
bundle. A `GET /models/{name}` issued during an export therefore froze the process.

Pinned as a source rule rather than a timing test: a duration assertion on a shared
runner is a flaky test, while "this call may not appear directly in an async handler" is
exactly the property, and it stays true whatever the machine is doing. Sibling of
``tests/test_no_url_fetch.py``, which guards a security boundary the same way.
"""

import ast
from pathlib import Path

ROUTES = Path(__file__).resolve().parent.parent / "app" / "routes"

# Methods that do disk I/O, take a lock a long operation holds, or both. The value is
# what a reader should be told when the guard fires.
BLOCKING = {
    "info": "reads the bundle under the registry's disk lock",
    "label_diagnostics": "reads the bundle under the registry's disk lock",
    "list": "an iterdir plus a stat per bundle",
    "recent": "reads the job-history file",
    "read_all": "reads the whole feedback file",
}

# Receivers whose attributes the names above actually refer to. Without this, any local
# variable with a `.list()` method would trip the guard.
RECEIVERS = {"registry", "get_registry()", "job_history", "feedback_store"}


def _receiver_name(node: ast.expr) -> str | None:
    """`registry.info` -> "registry"; `get_registry().info` -> "get_registry()"."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return f"{node.func.id}()"
    return None


def _offenders_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.AsyncFunctionDef):
            continue
        for node in ast.walk(func):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            why = BLOCKING.get(node.func.attr)
            if why and _receiver_name(node.func.value) in RECEIVERS:
                found.append(
                    f"{path.name}:{node.lineno} in async {func.name}(): "
                    f"{_receiver_name(node.func.value)}.{node.func.attr}() {why}"
                )
    return found


def test_no_async_route_handler_calls_a_blocking_method_directly():
    """Offload with ``await asyncio.to_thread(...)``.

    Passing the method as a reference is what makes this detectable: after the offload
    there is no call expression left inside the handler for the guard to find.
    """
    offenders = sorted(o for path in ROUTES.glob("*.py") for o in _offenders_in(path))
    assert not offenders, (
        "blocking work on the event loop — wrap in await asyncio.to_thread(...):\n  "
        + "\n  ".join(offenders)
    )
