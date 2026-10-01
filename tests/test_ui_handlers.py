"""UI event handlers, executed under node rather than pattern-matched (audit 2026-09-30, W02).

A regex over the source cannot see a wiring fault. U01 shipped with a test asserting that
`latestOnly` is *present* in training.js, while the wrapper handed each handler the `change`
event where it expected its `isCurrent` predicate: the first `isCurrent()` threw, the column
pickers stayed empty, and no training could be started from the admin UI.

These tests load the real modules into node with the page's globals stubbed, fire a handler
the way the browser does, and assert on what it did. Skipped where node is absent, like
`test_ui_query.py`: the UI has no toolchain on purpose, so node is a convenience here, not a
build dependency.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "app" / "static" / "ui"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# The page as far as the handlers under test reach into it. Elements are created on first
# lookup and remembered, so a test can seed a value before firing and read one afterwards;
# `errors` collects what reached showError AND any rejection nobody handled -- the U01
# failure was the second kind, invisible to a user except as a picker that never filled.
_PRELUDE = r"""
"use strict";
const elements = {};
function el(sel) {
  if (!elements[sel]) {
    elements[sel] = {
      value: "", hidden: true, disabled: false, textContent: "", innerHTML: "",
      style: {}, dataset: {},
      classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
      addEventListener() {}, removeEventListener() {}, focus() {},
      querySelector: (inner) => el(`${sel} ${inner}`),
      querySelectorAll: () => [],
    };
  }
  return elements[sel];
}
const document = { querySelector: el, querySelectorAll: () => [], addEventListener() {} };
const $ = el;
const errors = [];
process.on("unhandledRejection", (err) => errors.push(`unhandled: ${err}`));
function showError(target, err) { errors.push(String((err && err.message) || err)); }
const pickers = [];
function createPillPicker(opts) {
  const picker = { options: null, setOptions(o) { this.options = o; }, selected: () => [] };
  pickers.push(picker);
  return picker;
}
const apiCalls = [];
const Api = { get: async (url) => { apiCalls.push(url); return { columns: ["title", "label"] }; } };
const t = (key) => key;
const esc = (s) => String(s);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function report(value) { console.log(`RESULT ${JSON.stringify(value)}`); }
"""


def _function(module: str, name: str) -> str:
    """One top-level function out of a module whose load would touch the page (app.js
    calls `boot()` as its last statement)."""
    source = (UI / module).read_text(encoding="utf-8")
    start = source.index(f"function {name}(")
    return source[start:source.index("\n}\n", start) + 2]


def _run(tmp_path: Path, body: str, modules: tuple[str, ...] = (), helpers: tuple[str, ...] = ()) -> dict:
    """Load `modules` whole and `helpers` as single functions after the prelude, run `body`
    as an async function, and return what it reported."""
    parts = [_PRELUDE]
    parts += [(UI / module).read_text(encoding="utf-8") for module in modules]
    parts += [_function(module, name) for module, name in (h.split(":") for h in helpers)]
    parts.append(f"(async () => {{\n{body}\n}})().catch((err) => report({{ crashed: String(err) }}));")
    script = tmp_path / "handler-test.cjs"
    script.write_text("\n".join(parts), encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        # node writes UTF-8; without the encoding, Windows decodes it as cp1252.
        ["node", str(script)], capture_output=True, encoding="utf-8", timeout=30, check=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, f"the script reported nothing: {done.stdout}{done.stderr}"
    return json.loads(lines[-1][len("RESULT "):])


@needs_node
def test_picking_a_dataset_fills_the_training_column_pickers(tmp_path):
    """U01: fired as the browser fires it -- with the event as its only argument."""
    result = _run(tmp_path, modules=("training.js",), helpers=("app.js:latestOnly",), body="""
        el("#train-dataset").value = "tiny.csv";
        latestOnly(loadDatasetColumns)({ type: "change", target: el("#train-dataset") });
        await sleep(400);
        report({ errors, calls: apiCalls, options: pickers.map((p) => p.options) });
    """)

    assert result["errors"] == []
    assert result["calls"] == ["/datasets/tiny.csv"]
    assert result["options"] == [["title", "label"], ["title", "label"]], (
        "the text and label pickers did not receive the dataset's columns"
    )


@needs_node
def test_the_wrapped_work_receives_only_its_predicate(tmp_path):
    """The contract every `latestOnly` caller relies on, whatever the listener passes."""
    result = _run(tmp_path, helpers=("app.js:latestOnly",), body="""
        const seen = [];
        latestOnly((...args) => seen.push(args.map((a) => typeof a)), 10)({ type: "change" });
        await sleep(60);
        report({ seen });
    """)

    assert result["seen"] == [["function"]]


@needs_node
def test_a_burst_of_changes_runs_the_work_once(tmp_path):
    """Arrow-keying through a <select> fires `change` per keypress; each read is a whole CSV."""
    result = _run(tmp_path, helpers=("app.js:latestOnly",), body="""
        let runs = 0;
        const listener = latestOnly(() => { runs += 1; }, 30);
        for (let i = 0; i < 5; i += 1) listener({ type: "change" });
        await sleep(120);
        report({ runs });
    """)

    assert result["runs"] == 1


@needs_node
def test_an_answer_overtaken_by_a_newer_choice_is_not_applied(tmp_path):
    """The slow first read resolves after the second one started: only the second lands."""
    result = _run(tmp_path, helpers=("app.js:latestOnly",), body="""
        const applied = [];
        const delays = [120, 10];
        let call = 0;
        const listener = latestOnly(async (isCurrent) => {
          const mine = call++;
          await sleep(delays[mine]);
          if (isCurrent()) applied.push(mine);
        }, 5);
        listener({ type: "change" });
        await sleep(30);
        listener({ type: "change" });
        await sleep(250);
        report({ applied });
    """)

    assert result["applied"] == [1]



# --- S06: a copied command carries the model's name and nothing else --------------------------


@needs_node
@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell to paste into")
@pytest.mark.parametrize("name", [
    "m'$(touch PWNED)'",
    "Fächer (neu) & `touch PWNED` $HOME",
    "plain_model",
])
def test_the_copied_curl_command_runs_nothing_but_curl(tmp_path, name):
    """S06 (audit 2026-09-30): "curl kopieren" put the name between single quotes as it is,
    and `safe_name` allows `'`, `$`, `(`, `)` and backticks -- a model called
    `m'$(touch PWNED)'` ran its command in the shell of whoever pasted the snippet, readonly
    key or not. Pasted into a real shell here, with `curl` a function that records what it
    was handed."""
    command = _run(tmp_path, helpers=("model-detail.js:curlFor",), body=f"""
        globalThis.location = {{ origin: "http://localhost:8000" }};
        report({{ command: curlFor({json.dumps(name)}) }});
    """)["command"]
    script = "\n".join([
        "curl() { for arg in \"$@\"; do printf '%s\\000' \"$arg\"; done > argv.bin; }",
        "KEY=test-key",
        command,
    ])

    # A file, not `sh -c`: on Windows an argument passes through the command line's codepage
    # on its way into MSYS, which turns the umlaut into something else before any quoting.
    (tmp_path / "paste.sh").write_bytes(script.encode("utf-8"))
    subprocess.run(["sh", "paste.sh"], cwd=tmp_path, check=True, timeout=30)  # noqa: S603, S607

    assert not (tmp_path / "PWNED").exists(), "the pasted command ran a command of the name's"
    argv = (tmp_path / "argv.bin").read_bytes().decode("utf-8").split("\0")[:-1]
    body = json.loads(argv[argv.index("-d") + 1])
    assert body["model_name"] == name
    assert "X-API-Key: test-key" in argv, "the key still comes from the shell variable"



# --- S09: signing out leaves nothing of the session on the page -------------------------------


@needs_node
def test_signing_out_hands_the_next_person_a_fresh_page(tmp_path):
    """S09 (audit 2026-09-30): signing out cleared the key and hid the app -- and the next
    person to sign in found the previous one's query text, its result and a share link, a
    bearer capability, still on the page. A reload is the only state that stays clean as views
    are added; the forms are reset first, because Firefox puts field values back on one."""
    result = _run(tmp_path, helpers=("app.js:signOut",), body="""
        const calls = [];
        Api.clearKey = () => calls.push("clearKey");
        globalThis.showLogin = () => calls.push("showLogin");
        const forms = [{ reset: () => calls.push("reset query-form") },
                       { reset: () => calls.push("reset train-form") }];
        document.querySelectorAll = (selector) => (selector === "form" ? forms : []);
        globalThis.location = { reload: () => calls.push("reload") };
        signOut();
        report({ calls });
    """)

    assert result["calls"] == ["clearKey", "reset query-form", "reset train-form", "reload"]
