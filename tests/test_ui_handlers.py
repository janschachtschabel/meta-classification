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
      addEventListener() {}, removeEventListener() {}, focus() {}, remove() {},
      insertAdjacentHTML() {},
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


def _run(tmp_path: Path, body: str, modules: tuple[str, ...] = (), helpers: tuple[str, ...] = (),
         prelude: str = _PRELUDE) -> dict:
    """Load `modules` whole and `helpers` as single functions after the prelude, run `body`
    as an async function, and return what it reported."""
    parts = [prelude]
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





# --- U06/U03: a single-text answer stays about the text that was classified ------------------

# The Query tab answering one text, with explain.js and feedback.js wired in as the page wires
# them. Every request is recorded with the text(s) it carried -- and the user types on while
# each one is on its way, which is what the audit did in the browser. The answer's buttons
# are fakes that keep the handler they are given, so a test clicks them as a user would.
_ONE_TEXT = """
    const CLASSIFIED = "Pythagoras im rechtwinkligen Dreieck";
    el("#query-text").value = CLASSIFIED;
    el("#query-topk").value = "";
    const sent = [];
    Api.post = async (url, body) => {
      sent.push({ url, texts: body.texts || [body.text] });
      el("#query-text").value = "Photosynthese in der Pflanze";
      if (url === "/metadata") return { results: [{ title: "", description: "", keywords: [] }] };
      if (url === "/feedback") return { collected: 1 };
      return { results: [{ predictions: [{ uri: "u:m", label: "Mathematik", confidence: 0.9 }] }] };
    };
    const gets = [];
    Api.get = async (url) => { gets.push(url); return [{ uri: "u:bio", label: "Biologie" }]; };
    globalThis.fmtFixed = (v, d) => Number(v).toFixed(d);
    globalThis.applyBarWidths = () => {};
    globalThis.closer = (close) => close;
    globalThis.I18n = { locale: () => "de" };
    const click = {};
    const button = (kind) => ({ dataset: { [kind]: "m" }, closest: () => el("#card"),
                                addEventListener: (type, run) => { click[kind] = run; } });
    const out = { innerHTML: "", querySelectorAll: (sel) => (sel === "[data-explain]"
      ? [button("explain")] : sel === "[data-correct]" ? [button("correct")] : []) };
    el("#card .correction [data-send]").addEventListener = (type, run) => { click.save = run; };
"""
_ONE_TEXT_MODULES = ("query.js", "explain.js", "feedback.js")
CLASSIFIED = "Pythagoras im rechtwinkligen Dreieck"


@needs_node
def test_a_single_answer_and_its_extras_describe_the_same_text(tmp_path):
    """U06 (audit 2026-09-30): the classification was asked for the text in the field, and the
    metadata and the "Why?" button read the field again once the answer was back -- so a user
    who went on typing got the classification of one text beside the metadata of another, and
    "Why?" explained an answer to a text the model never saw."""
    result = _run(tmp_path, modules=_ONE_TEXT_MODULES, body=_ONE_TEXT + """
        await runSingle(["m"], { classify: true, metadata: true }, out);
        await click.explain();
        report({ sent, errors });
    """)

    assert result["errors"] == []
    assert result["sent"] == [{"url": "/predict", "texts": [CLASSIFIED]},
                              {"url": "/metadata", "texts": [CLASSIFIED]},
                              {"url": "/predict/explain", "texts": [CLASSIFIED]}]


@needs_node
def test_a_correction_records_the_text_that_was_classified(tmp_path):
    """U03 (audit 2026-09-30): the correction read the text field when it was SAVED. The
    audit classified "Pythagoras ...", changed the field to "Photosynthese ..." and corrected
    to Biologie -- and the feedback file got "Photosynthese" with the Mathematik prediction:
    a row the next training reads as a true pair. Here the field holds the classified text
    when the form is used; the next test is about one that changed."""
    result = _run(tmp_path, modules=_ONE_TEXT_MODULES, body=_ONE_TEXT + """
        await runSingle(["m"], { classify: true, metadata: false }, out);
        el("#query-text").value = CLASSIFIED;
        await click.correct();
        el("#query-text").value = CLASSIFIED;
        await click.save();
        report({ sent, errors });
    """)

    assert result["errors"] == []
    assert [s["texts"] for s in result["sent"] if s["url"] == "/feedback"] == [[CLASSIFIED]]


@needs_node
def test_a_correction_is_refused_once_the_text_has_changed(tmp_path):
    """The other half of U03: once the field says something else, which text a correction is
    for is no longer clear -- the labels on screen belong to the old one, the user may mean
    the new one. Neither guess goes into the training data; the user is asked to classify
    again, both when opening the form and when saving one opened before the edit."""
    result = _run(tmp_path, modules=_ONE_TEXT_MODULES, body=_ONE_TEXT + """
        await runSingle(["m"], { classify: true, metadata: false }, out);
        let shown = "";
        el("#card").insertAdjacentHTML = (where, html) => { shown += html; };
        await click.correct();
        const refusedOpen = { gets: [...gets], shown };
        el("#query-text").value = CLASSIFIED;
        await click.correct();
        el("#query-text").value = "Photosynthese in der Pflanze";
        await click.save();
        report({ refusedOpen, gets, sent, errors,
                 message: el("#card .correction .fb-message").textContent });
    """)

    assert result["refusedOpen"]["gets"] == [], "the form opened for a text that was not classified"
    assert "feedback.textChanged" in result["refusedOpen"]["shown"]
    assert result["gets"] == ["/models/m/labels"], "the form did not open for the classified text"
    assert not [s for s in result["sent"] if s["url"] == "/feedback"], (
        "a correction was saved for a text that was not classified")
    assert result["message"] == "feedback.textChanged"
    assert result["errors"] == []



# --- U04: a weight the user set survives a change of the column selection ---------------------


@needs_node
def test_a_weight_set_to_one_survives_picking_another_column(tmp_path):
    """U04 (audit 2026-09-30): the rebuild kept only weights above 1, so a deliberate 1 on
    "title" -- whose configured default is 2 -- jumped back to 2 the moment another column
    was picked, and the run trained with 2. The inputs are read back out of the markup the
    rebuild writes, as the browser would hold them."""
    result = _run(tmp_path, modules=("training.js",), body="""
        defaultColWeights = { title: 2, keywords: 2 };
        let inputs = [];
        const parse = () => {
          inputs = [...el("#textcol-weights-fields").innerHTML
            .matchAll(/value="([^"]*)" data-weight="([^"]*)"/g)]
            .map(([, value, weight]) => ({ value, dataset: { weight } }));
        };
        document.querySelectorAll = (sel) =>
          (sel === "#textcol-weights-fields [data-weight]" ? inputs : []);
        renderTextColWeights(["title"]);
        parse();
        inputs[0].value = "1";
        renderTextColWeights(["title", "description"]);
        parse();
        report({ shown: Object.fromEntries(inputs.map((i) => [i.dataset.weight, i.value])),
                 sent: textColumnWeights(), errors });
    """)

    assert result["errors"] == []
    assert result["shown"] == {"title": "1", "description": "1"}
    # A dict sent at all is taken as it is (prepare.py), so a 1 is said by leaving it out.
    assert result["sent"] == {}



# --- U05: a batch of label fields outlasts the rate limit -------------------------------------

# api.js on its own: it declares the `Api` the main prelude stubs, so it gets a page of its own.
_API_PRELUDE = r"""
"use strict";
const t = (key) => key;
const window = { dispatchEvent() {} };
function report(value) { console.log(`RESULT ${JSON.stringify(value)}`); }
const answer = (status, body, headers = {}) => ({
  status, ok: status >= 200 && status < 300, headers: new Headers(headers),
  json: async () => JSON.parse(body), text: async () => body,
});
"""


@needs_node
def test_a_refusal_for_the_rate_limit_carries_the_wait_the_server_named(tmp_path):
    """The 429 handler sends Retry-After so that no client has to guess (main.py); the
    transport dropped it, and the one caller that has to wait could not know how long."""
    result = _run(tmp_path, modules=("api.js",), prelude=_API_PRELUDE, body="""
        globalThis.fetch = async () =>
          answer(429, '{"detail": "Rate limit exceeded: 5 per 1 minute"}', { "Retry-After": "60" });
        try { await Api.post("/train", {}); report({ thrown: false }); }
        catch (err) { report({ status: err.status, retryAfter: err.retryAfter }); }
    """)

    assert result == {"status": 429, "retryAfter": 60}


@needs_node
def test_a_batch_of_seven_label_fields_waits_out_the_rate_limit(tmp_path):
    """U05 (audit 2026-09-30): one model per label field, one POST /train each, against a
    limit of 5 a minute -- seven fields gave five 202s and a 429 that ended the batch. A
    retry hit the limit again, and later "already exists" for the five already queued. The
    timers are fakes that fire at once and record what they were asked to wait."""
    result = _run(tmp_path, modules=("training.js",), body="""
        const delays = [];
        globalThis.setTimeout = (run, ms) => { delays.push(ms); run(); return 0; };
        const toasts = [];
        globalThis.toast = (message) => toasts.push(message);
        el("#train-dataset").value = "data.csv";
        el("#train-name").value = "fach";
        pickers[0].values = () => ["title"];
        pickers[1].values = () => ["f1", "f2", "f3", "f4", "f5", "f6", "f7"];
        const posted = [];
        let refusedOnce = false;
        Api.post = async (url, body) => {
          posted.push(body.model_name);
          if (posted.length === 6 && !refusedOnce) {
            refusedOnce = true;
            throw Object.assign(new Error("Rate limit exceeded: 5 per 1 minute"),
                                { status: 429, retryAfter: 60 });
          }
          return { status: posted.length === 1 ? "running" : "queued" };
        };
        await onTrainStart({ preventDefault() {} });
        report({ posted, delays, toasts, errors, button: el("#train-btn").disabled });
    """)

    assert result["errors"] == [], "the batch ended in an error"
    assert result["posted"] == ["fach_f1", "fach_f2", "fach_f3", "fach_f4", "fach_f5",
                                "fach_f6", "fach_f6", "fach_f7"]
    assert result["delays"] == [60_000], "it did not wait the window the server named"
    assert "train.waitingForLimit" in result["toasts"], "the wait was not said"
    assert result["toasts"][-1] == "train.startedQueued"
    assert result["button"] is False


@needs_node
def test_a_rate_limit_that_does_not_lift_still_ends_the_batch(tmp_path):
    """Waiting is bounded: another client on the same address can keep the window full, and
    a batch retrying forever would hold the Start button for as long as that lasts."""
    result = _run(tmp_path, modules=("training.js",), body="""
        globalThis.setTimeout = (run) => { run(); return 0; };
        globalThis.toast = () => {};
        el("#train-dataset").value = "data.csv";
        el("#train-name").value = "fach";
        pickers[0].values = () => ["title"];
        pickers[1].values = () => ["f1", "f2"];
        let calls = 0;
        Api.post = async () => {
          calls += 1;
          throw Object.assign(new Error("Rate limit exceeded"), { status: 429, retryAfter: 60 });
        };
        await onTrainStart({ preventDefault() {} });
        report({ calls, errors });
    """)

    assert result["calls"] == 4, "one attempt and three waits, then the batch gives up"
    assert result["errors"] == ["train.partialFailure"]
