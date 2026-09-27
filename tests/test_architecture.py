"""Boundaries `CLAUDE.md` states, asserted rather than reviewed (audit ARC-1, ARC-3…ARC-6).

"Routes stay thin and delegate to core modules" is a rule nothing enforced, and it had
already bent three ways: two route modules importing a third, a response contract that
re-encoded a rule the model owns, and a read-only inspection module pulled transitively onto
`Settings` through a cost model it shares a file with.

Every check here reads the source tree, so it holds for code that no test exercises.
"""

import ast
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
ROUTES = APP / "routes"


def _imports(path: Path, *, runtime_only: bool = False) -> list[tuple[str, int]]:
    """Every module this file imports, as (dotted name, level), relative imports included.

    ``runtime_only`` skips what sits under ``if TYPE_CHECKING:``. That distinction is the
    whole point of the dependency check below: an annotation-only import costs nothing at
    run time, and counting it reported a coupling the interpreter does not have.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    if runtime_only:
        for node in ast.walk(tree):
            guard = getattr(node, "test", None)
            is_type_checking = (
                isinstance(guard, ast.Name) and guard.id == "TYPE_CHECKING"
            ) or (isinstance(guard, ast.Attribute) and guard.attr == "TYPE_CHECKING")
            if isinstance(node, ast.If) and is_type_checking:
                skip.update(id(child) for child in ast.walk(node))
    found = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
    return found


# --- ARC-1: no route module imports another route module ----------------------------------


ROUTE_MODULES = sorted(p for p in ROUTES.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", ROUTE_MODULES, ids=lambda p: p.name)
def test_no_route_module_imports_another_route_module(path):
    """A route importing a sibling route makes one endpoint's module the owner of another's
    policy — `share.py` took its download staging from `models.py`, and `predict_bulk.py`
    took the registry's exists/get/TOCTOU mapping from `predict.py`. Shared route plumbing
    goes in a module of its own; anything else belongs in a core module.
    """
    siblings = {p.stem for p in ROUTE_MODULES} - {path.stem}
    offenders = [
        name for name, level in _imports(path)
        # `from .models import x` is level 1 with module "models"; a leading underscore
        # marks the shared-plumbing modules, which are not routes.
        if level == 1 and name in siblings and not name.startswith("_")
    ]

    assert not offenders, f"{path.name} imports the route module(s) {offenders}"


def test_the_shared_route_plumbing_is_not_a_route():
    """The module holding what several routes need must not also define endpoints, or it is
    just a fourth route module for the others to depend on."""
    shared = sorted(ROUTES.glob("_*.py"))
    assert shared, "no shared-plumbing module: the helpers are still inside a route module"

    for path in shared:
        source = path.read_text(encoding="utf-8")
        assert "APIRouter(" not in source, f"{path.name} defines a router"
        assert not re.search(r"@router\.", source), f"{path.name} defines endpoints"


# --- ARC-3: the model owns how top_k resolves ---------------------------------------------


def test_the_route_does_not_re_decide_what_top_k_resolves_to():
    """`ClassifierModel` already knows that binary/multiclass decide by argmax — that is why
    `resolved_top_k` answers 1 for them. The predict route re-encoded the same rule for the
    case where no `top_k` was sent, so two places knew it and only one was tested.
    """
    route = (ROUTES / "predict.py").read_text(encoding="utf-8")

    assert not re.search(r'task_type in \("binary", "multiclass"\)', route), (
        "predict.py still decides what a single-label task's applied top_k is; ask the model"
    )
    assert "applied_top_k(" in route, "the route does not delegate the resolution to the model"


def test_what_counts_as_a_single_label_task_is_written_once():
    """`metrics.is_single_label` carries the taxonomy's rule with the docstring explaining
    it. The same tuple membership was spelled out three more times — in the model twice and
    in the report builder — so adding a task type meant finding all four."""
    spellings = {
        path.name: path.read_text(encoding="utf-8").count('in ("binary", "multiclass")')
        for path in APP.rglob("*.py")
    }
    written_in = {name: n for name, n in spellings.items() if n}

    assert written_in == {"metrics.py": 1}, (
        f"the single-label rule is spelled out in {written_in}; ask metrics.is_single_label"
    )


# --- ARC-4: read-only dataset inspection does not depend on Settings ----------------------


def test_reading_a_datasets_statistics_stays_clear_of_the_server_settings():
    """`dataset_stats` reads a file it was handed and estimates what a run would cost. It
    has no business loading the server's configuration to do that, and after the split it
    does not: the cost model is a stdlib-only leaf and `CapacityPlan` arrives as an
    argument, imported only for the annotation.

    The audit reported this as an existing violation. It was not one — the `Settings`
    import was already behind `TYPE_CHECKING`, and a fresh interpreter importing
    `dataset_stats` never loaded `app.settings`. What the split did remove is real, and
    smaller: `app.memory` and `app.capacity` no longer load either.
    """
    seen: set[str] = set()

    def reachable(module: str) -> set[str]:
        if module in seen:
            return seen
        seen.add(module)
        path = APP / f"{module.replace('.', '/')}.py"
        if not path.exists():
            return seen
        for name, level in _imports(path, runtime_only=True):
            if level and name:
                reachable(name.split(".")[0])
        return seen

    reached = reachable("dataset_stats")
    for unwanted in ("settings", "capacity", "memory"):
        assert unwanted not in reached, (
            f"dataset_stats now reaches app.{unwanted} at run time; reached {sorted(reached)}"
        )


def test_each_of_the_profile_modules_has_one_reason_to_change():
    """332 lines carrying the cost model (changes on a re-benchmark), the capacity plan (the
    only part importing Settings) and the config parsing (changes with config.yaml)."""
    for name in ("profiles", "profile_costs", "capacity"):
        path = APP / f"{name}.py"
        assert path.exists(), f"app/{name}.py is missing: profiles.py was not split"
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 300, f"app/{name}.py is {lines} lines"


# --- ARC-5: "start or queue a job" is written once ----------------------------------------


def test_starting_or_queueing_a_job_is_written_once():
    """`/train` and `/models/{name}/evaluate` had the same twelve lines — submit, map a
    RuntimeError to 409, build the same five-key response — differing only in the value of
    one key. A change to how a run is queued or how a refusal reads had to be made twice.
    """
    submits = sorted(
        path.name for path in ROUTE_MODULES
        if "job_runner.submit(" in path.read_text(encoding="utf-8")
    )

    assert submits == ["_jobs.py"], (
        f"job_runner.submit is called from {submits}; the submit-or-409 block belongs in one "
        "place, and every endpoint that starts a run should go through it"
    )


# --- ARC-6: the config is not re-parsed per request ---------------------------------------


def test_the_training_config_is_not_re_read_on_every_request():
    """Three handlers called this, two of them `async def` — a blocking file read plus a
    yaml parse on the event loop, per request, for a file that changes between deployments.
    Every other singleton in the app is cached."""
    profiles = (APP / "profiles.py").read_text(encoding="utf-8")

    assert "lru_cache" in profiles, "load_training_config parses config.yaml on every call"


def test_editing_the_config_still_takes_effect_without_a_restart():
    """The cache must key on the file's mtime, not just its path: a plain `lru_cache` would
    freeze the first parse for the process's life, which is a behaviour change nobody asked
    for and a trap for any test that rewrites a config at the same path."""
    profiles = (APP / "profiles.py").read_text(encoding="utf-8")

    assert "st_mtime" in profiles, (
        "the config cache does not consider the file's mtime, so an edit is ignored until "
        "the process restarts"
    )
