"""Supply-chain guards: the lock files agree, and nothing is imported undeclared.

``requirements.lock`` (human-maintained direct pins) is the input constraint for
the compiled ``requirements-hashes.lock`` (full transitive tree + sha256, used by
Docker/CI). If someone bumps the former and forgets to recompile the latter, the
shipped tree silently diverges from the tested one — this test turns that human
factor into a red gate.

The third test closes the hole the other two structurally cannot see: a package that is
imported directly but declared nowhere is not a pin that went stale, it is a pin that never
existed, so no amount of lock-file comparison notices it.
"""

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).parent.parent

_PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", re.MULTILINE)


def _pins(filename: str) -> dict[str, str]:
    text = (ROOT / filename).read_text(encoding="utf-8")
    return {name.lower(): version for name, version in _PIN_RE.findall(text)}


def test_hashes_lock_matches_direct_pins():
    direct = _pins("requirements.lock")
    hashed = _pins("requirements-hashes.lock")
    assert len(direct) == 17, f"expected the 17 direct pins, parsed {sorted(direct)}"
    stale = {name: (version, hashed.get(name))
             for name, version in direct.items() if hashed.get(name) != version}
    assert not stale, (
        f"requirements-hashes.lock is stale for {stale} — recompile it "
        "(command in the requirements.lock header)."
    )


def test_hashes_lock_actually_carries_hashes():
    """The parity test alone would pass a recompile that dropped every hash;
    assert the hashes are really present so '--require-hashes' has teeth."""
    text = (ROOT / "requirements-hashes.lock").read_text(encoding="utf-8")
    pins = _PIN_RE.findall(text)
    assert pins, "no pins parsed from requirements-hashes.lock"
    assert text.count("--hash=sha256:") >= len(pins), (
        "requirements-hashes.lock carries fewer sha256 hashes than pinned "
        "packages — hash-pinning is not actually in effect."
    )


def _canonical(name: str) -> str:
    """PEP 503 normalisation, so `PyYAML`, `pyyaml` and `py_yaml` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = re.findall(r"^([A-Za-z0-9_.-]+)", text, re.MULTILINE)
    return {_canonical(n) for n in names}


def _imported_top_level() -> dict[str, str]:
    """Every third-party package `app/` imports, mapped to the module that imports it."""
    found: dict[str, str] = {}
    for source in sorted((ROOT / "app").rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root != "app" and root not in sys.stdlib_module_names:
                    found.setdefault(root, str(source.relative_to(ROOT)))
    return found


def test_every_package_app_imports_is_declared():
    """A direct import has to be a declared dependency, not a transitive passenger.

    `starlette` was imported in two route modules and declared in no manifest (audit DEP-1):
    `requirements-hashes.lock` pinned what CI and the image installed while the dev venv
    resolved something three minor versions newer, and neither lock file could report it
    because an undeclared package is not a stale pin. Both call sites are the temp-file
    unlink after a streamed download, so a silent change there leaks temp files or 500s a
    download — in dev or in production, depending which side moved.

    The reverse direction is deliberately NOT asserted: `python-multipart` is declared without
    being imported, because FastAPI needs it to parse forms.
    """
    declared = _declared()
    distributions = packages_distributions()

    undeclared = {}
    for module, importer in sorted(_imported_top_level().items()):
        # A module can come from several distributions; declaring any one of them is enough.
        names = distributions.get(module, [module])
        if not any(_canonical(name) in declared for name in names):
            undeclared[module] = f"{importer} (provided by {', '.join(names)})"

    assert not undeclared, (
        "imported by app/ but declared in no manifest: "
        + "; ".join(f"{module} in {where}" for module, where in undeclared.items())
    )
