"""Documentation that the code can contradict is tested, not trusted (audit DOC-1..DOC-4).

Every claim here drifted silently at least once: a test count three generations old, a profile
key list missing five of thirteen, an "allowlist of the four bundle files" that holds seven, and
an export member list short by one file. The first is a number nobody can keep current by hand;
the rest are lists the code owns, so the code is asked.

DOC-2 is the one with teeth beyond tidiness: `stratified_splits` defaults to **False** when a
profile omits it, and this project's own benchmark puts that at −0.0050 macro F1 and ~1.8× the
run-to-run spread. Someone writing a custom profile from an incomplete field list gets the worse
setting and no hint why.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _profile_keys() -> set[str]:
    """Every `config.yaml` profile key `profiles.py` actually reads."""
    source = (ROOT / "app" / "profiles.py").read_text(encoding="utf-8")
    keys = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "raw"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
    return keys


def test_every_profile_key_is_documented():
    """A key the loader reads but the docs omit is a silent default, not a missing sentence."""
    documented = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    keys = _profile_keys()

    assert len(keys) >= 13, f"expected the loader to read at least 13 keys, parsed {sorted(keys)}"
    missing = sorted(k for k in keys if k not in documented)

    assert not missing, f"profile keys read by app/profiles.py but absent from configuration.md: {missing}"


def test_the_readme_states_the_import_allowlist_correctly():
    """Security-relevant text: it describes the size of the guard's surface."""
    from app.model_archive import ALLOWED_MEMBERS

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "allowlist of the four bundle files" not in readme, (
        f"the allowlist holds {len(ALLOWED_MEMBERS)} members, not four"
    )


def test_the_readme_lists_every_exported_member():
    """The interop contract for reading a bundle outside the app.

    What export packs is `ALLOWED_MEMBERS` minus the two it generates, so the README's list has
    to name all of them — `metrics.json` was missing.
    """
    from app.model_archive import ALLOWED_MEMBERS, CARD_FILE, MANIFEST_FILE

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    interop = readme[readme.index("## Model interop / export"):][:2000]

    for member in sorted(ALLOWED_MEMBERS - {MANIFEST_FILE, CARD_FILE}):
        assert f"`{member}`" in interop, f"{member} is exported but not named in the interop section"


def test_the_readme_does_not_pin_a_test_count():
    """A hand-written count is wrong the next time anyone adds a test.

    It claimed 331 while the suite collected 593, then 764. The command is the useful part;
    the number only ever misleads.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    stale = re.findall(r"pytest tests -q\s*#\s*(\d+)\s*tests", readme)

    assert not stale, f"README pins a test count ({stale}) that nothing keeps current"


def test_the_workspace_note_describes_the_bundle_format_the_code_writes():
    """The workspace-root CLAUDE.md sits OUTSIDE this git repo (audit DOC-7), so no CI run
    and no pull request has ever read it — and it is the file an agent loads first. Its
    member list omitted `vocabulary.json`, which is in `REQUIRED_FILES`: a bundle without it
    is refused, so the document described a format the code rejects.

    Skipped when it is absent, because a checkout of this repo alone does not have it. That
    is the point: the file is only reachable from the developer's workspace, which is exactly
    why it drifted.
    """
    from app.model_archive import REQUIRED_FILES

    note = Path(__file__).resolve().parents[2] / "CLAUDE.md"
    if not note.exists():
        pytest.skip("workspace-root CLAUDE.md is outside this repo")

    text = note.read_text(encoding="utf-8")
    line = next((row for row in text.splitlines() if row.startswith("- Model bundles")), None)
    assert line, "the workspace note no longer describes the bundle format at all"

    missing = sorted(name for name in REQUIRED_FILES if f"`{name}`" not in line)
    assert not missing, f"the workspace note omits required bundle files: {missing}"



def _interop_snippet() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    interop = readme[readme.index("## Model interop / export"):]
    code = interop[interop.index("```python") + len("```python"):]
    return code[:code.index("```")]


def test_the_readme_interop_snippet_loads_a_bundle_while_trusting_nothing(tmp_path, monkeypatch):
    """S07 (audit 2026-09-30): the snippet passed `trusted=sio.get_untrusted_types(file=...)`,
    which trusts every type a file declares -- the code-execution path the app's own loader
    refuses (`model_io._safe_skops_load`). A bundle of this app declares none, so
    `trusted=[]` loads it and refuses one that declares more. The snippet runs here against a
    bundle the app just trained, word AND char vectorizer, so the recipe is one that works."""
    from app.profiles import Profile, TrainingConfig
    from app.registry import Registry
    from app.settings import Settings
    from app.training import run_training

    code = _interop_snippet()
    loads = [node for node in ast.walk(ast.parse(code))
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "sio.load"]
    assert len(loads) == 2, "the snippet loads both skops files"
    for call in loads:
        trusted = next((kw.value for kw in call.keywords if kw.arg == "trusted"), None)
        assert isinstance(trusted, ast.List) and not trusted.elts, f"trusts more: {ast.unparse(call)}"

    subjects = {"uri:math": "Bruchrechnung mit Nennern und Zählern üben",
                "uri:bio": "Photosynthese im Blatt und Zellatmung im Versuch"}
    rows = [f"{text} Teil {i};{uri}" for uri, text in subjects.items() for i in range(12)]
    (tmp_path / "set.csv").write_text("title;labels\n" + "\n".join(rows) + "\n", encoding="utf-8")
    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "models", auth_enabled=False)
    profile = Profile("p", c_grid=[1.0], use_char=True)
    config = TrainingConfig(default_profile="p", profiles={"p": profile}, validation_size=0.2,
                            test_size=0.2, min_text_length=5, drop_duplicates=True,
                            min_samples_per_label=2)
    run_training({"dataset_name": "set.csv", "model_name": "m", "text_columns": ["title"],
                  "label_column": "labels", "csv_separator": ";", "label_separator": ",",
                  "label_filter": None},
                 settings, config, profile, Registry(settings.models_dir, 2),
                 on_progress=lambda **_: None, should_stop=lambda: False)

    monkeypatch.chdir(settings.models_dir / "m")
    namespace: dict = {}
    exec(compile(code, "README.md (interop snippet)", "exec"), namespace)

    assert namespace["char_vec"] is not None, "both halves of the vectorizer were exercised"
    assert namespace["proba"].shape == (1, len(namespace["cfg"]["classes"]))


# --- W01: the documented install is the tested tree -------------------------------------------


def test_the_readme_installs_the_tree_the_suite_was_run_against():
    """W01 (audit 2026-09-30): the README installed `requirements.txt -c requirements.lock`,
    which pins the 17 direct dependencies and resolves the rest fresh. A fresh venv got an
    anyio whose deprecation warning `filterwarnings = error` turns into collection errors in
    ten test modules, while CI -- installing the hashed tree -- stayed green. The README now
    installs what CI and the image install."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installs = [line.strip() for line in readme.splitlines() if line.strip().startswith("pip install")]

    assert not [line for line in installs if "-c requirements.lock" in line], installs
    assert sum("--require-hashes -r requirements-hashes.lock" in line for line in installs) >= 2, installs


def test_the_readme_names_every_profile_key():
    """W08 (audit 2026-09-30): the README's list of profile fields named 7 of the 13 the
    loader reads, and read as complete -- `stratified_splits`, `selection_tol` and the rest
    were findable only in docs/configuration.md."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    [sentence] = [line for line in readme.splitlines() if line.startswith("Add your own profiles")]
    keys = _profile_keys() - {"c_grid"}  # the lower-case spelling is an accepted alias

    assert not sorted(k for k in keys if f"`{k}`" not in sentence), sentence

