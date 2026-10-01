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
import re
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
      insertAdjacentHTML() {}, setAttribute() {},
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
    result = _run(tmp_path, modules=_ONE_TEXT_MODULES, helpers=("app.js:busy",),
                  body=_ONE_TEXT + """
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
    a row the next training reads as a true pair. This one checks the wiring -- the classified
    text travels from the answer through the form into the request -- and passes against the
    old code by construction: while the field still holds the classified text, reading it at
    save time sends the same. The next test is the U03 regression guard."""
    result = _run(tmp_path, modules=_ONE_TEXT_MODULES, helpers=("app.js:busy",),
                  body=_ONE_TEXT + """
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
    result = _run(tmp_path, modules=_ONE_TEXT_MODULES, helpers=("app.js:busy",),
                  body=_ONE_TEXT + """
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
  blob: async () => new Blob([body]),
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
    result = _run(tmp_path, modules=("training.js",), helpers=("app.js:busy",), body="""
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
    result = _run(tmp_path, modules=("training.js",), helpers=("app.js:busy",), body="""
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



# --- U08: state that outlived its moment ------------------------------------------------------

# The dataset panel: a <dialog> rebuilt on every open. Each open gets a fresh frame, and the
# frames of earlier opens are disconnected, as the browser leaves them when innerHTML replaces
# them. The reads of /datasets/{name} answer when a test says so.
_DATASET_PANEL = """
    const dialog = el("#dataset-detail");
    dialog.showModal = () => {};
    globalThis.openModal = (opened) => opened.showModal();
    const frames = [];
    dialog.querySelector = (sel) => {
      if (sel !== ".detail") return null;
      frames.forEach((f) => { f.isConnected = false; });
      const parts = {};
      const frame = {
        isConnected: true, innerHTML: "", focus() {},
        insertAdjacentHTML(where, html) { this.innerHTML += html; },
        querySelector: (inner) => (parts[inner] ||= { value: "", textContent: "", disabled: false,
          addEventListener(type, run) { this.run = run; }, remove() {} }),
        querySelectorAll: (inner) =>
          (inner === "#ds-text-cols option" ? [{ selected: true, value: "title" }] : []),
      };
      frames.push(frame);
      return frame;
    };
    const answers = {};
    Api.get = (url) => new Promise((resolve) => { answers[url] = resolve; });
    const analyzed = [];
    Api.post = async (url, body) => {
      analyzed.push(body.dataset_name);
      return { total_samples: 1, unique_labels: 1, label_threshold_analysis: {},
               recommended_min_samples_per_label: 5, rare_labels_under_10: {} };
    };
"""


@needs_node
def test_a_late_answer_for_one_dataset_does_not_retarget_another(tmp_path):
    """U08 (audit 2026-09-30): the panel kept the open dataset's name in a module-wide
    object, and every answer wrote it. Open A, close it while it is still reading, open B:
    A's late answer set the name back to A, and "Analyse" in B's panel analysed A."""
    result = _run(tmp_path, modules=("dataset-detail.js",), helpers=("app.js:busy",), body=_DATASET_PANEL + """
        const first = showDatasetDetail("a.csv");
        const second = showDatasetDetail("b.csv");
        answers["/datasets/b.csv"]({ columns: ["title", "label"], sample: [] });
        await second;
        answers["/datasets/a.csv"]({ columns: ["title", "label"], sample: [] });
        await first;
        await frames[1].querySelector("#ds-analyze").run();
        report({ analyzed, errors });
    """)

    assert result["errors"] == []
    assert result["analyzed"] == ["b.csv"]


@needs_node
def test_a_long_sample_cell_is_cut_before_it_is_escaped(tmp_path):
    """U08 (audit 2026-09-30): the sample table escaped a cell and then cut it to 120
    characters, so the cut could land inside an entity -- "...&b" was shown as "...&a", half
    of "&amp;". The real escape.js here: the prelude's pass-through would hide exactly this."""
    result = _run(tmp_path, modules=("escape.js", "dataset-detail.js"),
                  prelude=_PRELUDE.replace("const esc = (s) => String(s);\n", ""),
                  body=_DATASET_PANEL + """
        const shown = showDatasetDetail("a.csv");
        answers["/datasets/a.csv"]({ columns: ["text"], sample: [{ text: "a".repeat(118) + "&b<c" }] });
        await shown;
        report({ html: frames[0].innerHTML, errors });
    """)

    cells = re.findall(r"<td>(.*?)</td>", result["html"])
    assert cells == ["a" * 118 + "&amp;b"]


@needs_node
def test_a_double_click_creates_one_share_link(tmp_path):
    """U08 (audit 2026-09-30): "Share link" posted once per click, so a double click created
    two links -- two bearer capabilities, valid for a day, for one intent. Once the first
    answer is in, sharing works again."""
    result = _run(tmp_path, modules=("share.js",), body="""
        globalThis.location = { origin: "http://localhost" };
        globalThis.closer = (close) => close;
        globalThis.toastError = (err) => errors.push(String(err.message || err));
        const posts = [];
        const pending = [];
        Api.post = (url) => {
          posts.push(url);
          return new Promise((resolve) => pending.push(resolve));
        };
        const first = shareResource("models", "m", "#models-share");
        const second = shareResource("models", "m", "#models-share");
        pending.forEach((resolve) => resolve({ share_url: "/share/abc", expires_at: null }));
        await Promise.all([first, second]);
        const again = shareResource("models", "m", "#models-share");
        pending.slice(1).forEach((resolve) => resolve({ share_url: "/share/def", expires_at: null }));
        await again;
        report({ posts, errors });
    """)

    assert result["posts"] == ["/models/m/export", "/models/m/export"], (
        "a double click must post once, and a later click again")
    assert result["errors"] == []


@needs_node
@pytest.mark.parametrize("content_type, body", [
    ("text/html; charset=utf-8", "<html><body>Anmelden</body></html>"),
    ("application/json", "<html>"),
])
def test_an_answer_that_is_not_the_apis_says_so(tmp_path, content_type, body):
    """U08 (audit 2026-09-30): a JSON call handed back the Response itself when a 2xx was not
    JSON -- an SSO proxy's sign-in page, a captive portal -- and the caller failed on it as
    "names.map is not a function", or showed an empty list. A body that claims JSON and is
    not failed with the browser's own parser message."""
    result = _run(tmp_path, modules=("api.js",), prelude=_API_PRELUDE, body=f"""
        globalThis.fetch = async () =>
          answer(200, {json.dumps(body)}, {{ "content-type": {json.dumps(content_type)} }});
        try {{ const got = await Api.get("/models"); report({{ thrown: false, got: typeof got }}); }}
        catch (err) {{ report({{ thrown: true, status: err.status, message: err.message }}); }}
    """)

    assert result == {"thrown": True, "status": 200, "message": "errors.notApi"}


@needs_node
def test_a_download_still_gets_the_answer_itself(tmp_path):
    """The other side of the same seam: a download IS a non-JSON answer, and /predict/csv
    reports its completeness in a header (X-Input-Rows, U02)."""
    result = _run(tmp_path, modules=("api.js",), prelude=_API_PRELUDE, body="""
        globalThis.document = { createElement: () => ({ click() {} }) };
        globalThis.fetch = async () =>
          answer(200, "row,text\\n0,a", { "content-type": "text/csv", "X-Input-Rows": "1" });
        const { blob, headers } = await Api.downloadForm("/predict/csv", {}, "x.csv");
        report({ text: await blob.text(), rows: headers.get("X-Input-Rows") });
    """)

    assert result == {"text": "row,text\n0,a", "rows": "1"}


# A <select> as the browser keeps one: assigning its options picks the `selected` one or the
# first, and a value no option carries reads back as "" -- which is what a rebuild did to the
# user's choice.
_FAKE_SELECT = """
    const fakeSelect = (sel) => {
      const select = el(sel);
      let html = "", value = "";
      const options = () => [...html.matchAll(/<option value="([^"]*)"([^>]*)>/g)];
      Object.defineProperty(select, "innerHTML", { get: () => html, set: (markup) => {
        html = markup;
        const picked = options().find((o) => /\\bselected\\b/.test(o[2])) || options()[0];
        value = picked ? picked[1] : "";
      } });
      Object.defineProperty(select, "value", { get: () => value, set: (wanted) => {
        value = options().some((o) => o[1] === wanted) ? wanted : "";
      } });
      return select;
    };
    let datasets = [{ name: "a.csv" }, { name: "b.csv" }];
    Api.get = async (url) => (url === "/datasets" ? datasets : {
      profiles: [{ name: "fast", description: "" }, { name: "auto", description: "" }],
      default_profile: "auto", default_text_column_weights: {} });
    fakeSelect("#train-dataset");
    fakeSelect("#train-profile");
"""


@needs_node
def test_coming_back_to_the_training_tab_keeps_the_choices(tmp_path):
    """U08 (audit 2026-09-30): every visit to the tab rebuilt the dataset and profile lists,
    so switching to Models and back reset both -- while the column pickers still offered the
    columns of the dataset that was no longer shown as chosen."""
    result = _run(tmp_path, modules=("training.js",), body=_FAKE_SELECT + """
        await loadTrainingTab();
        el("#train-dataset").value = "b.csv";
        el("#train-profile").value = "fast";
        await loadTrainingTab();
        report({ dataset: el("#train-dataset").value, profile: el("#train-profile").value, errors });
    """)

    assert result == {"dataset": "b.csv", "profile": "fast", "errors": []}


@needs_node
def test_a_dataset_deleted_meanwhile_clears_its_columns(tmp_path):
    """The case a kept choice cannot cover: the dataset is gone. The list falls back to "pick
    one", and the pickers must not go on offering its columns."""
    result = _run(tmp_path, modules=("training.js",), body=_FAKE_SELECT + """
        await loadTrainingTab();
        el("#train-dataset").value = "b.csv";
        datasets = [{ name: "a.csv" }];
        await loadTrainingTab();
        await sleep(10);
        report({ dataset: el("#train-dataset").value, columns: pickers.map((p) => p.options), errors });
    """)

    assert result == {"dataset": "", "columns": [[], []], "errors": []}


@needs_node
def test_coming_back_to_the_query_tab_keeps_the_models_picked(tmp_path):
    """U08: the model list was rebuilt on every visit with the first model checked, so a user
    who had picked another one classified with the first after looking at the Models tab. An
    empty choice is a choice too -- metadata alone needs no model."""
    result = _run(tmp_path, modules=("query.js",), body="""
        Api.get = async () => ["m1", "m2", "m3"];
        const box = el("#query-models");
        let boxes = [];
        const read = () => {
          boxes = [...box.innerHTML.matchAll(/value="([^"]*)"( checked)?/g)]
            .map(([, value, checked]) => ({ value, checked: Boolean(checked) }));
        };
        box.querySelector = () => boxes[0] || null;
        document.querySelectorAll = (sel) =>
          (sel === 'input[name="query-model"]:checked' ? boxes.filter((b) => b.checked) : []);
        const picked = () => boxes.filter((b) => b.checked).map((b) => b.value);
        await loadQueryTab(); read();
        const first = picked();
        boxes[0].checked = false;
        boxes[2].checked = true;
        await loadQueryTab(); read();
        const kept = picked();
        boxes[2].checked = false;
        await loadQueryTab(); read();
        report({ first, kept, none: picked(), errors });
    """)

    assert result == {"first": ["m1"], "kept": ["m3"], "none": [], "errors": []}



# --- U09: what a screen reader is told, and where the focus is -------------------------------

# Toasts, with the page's two stacks and a model panel that holds the pair openModal gives
# every dialog. `document.querySelector("dialog[open]")` answers as the browser would.
_TOASTS = """
    document.createElement = () => ({ innerHTML: "", addEventListener() {},
                                      querySelector: () => ({ addEventListener() {} }) });
    const stack = () => ({ items: [], appendChild(item) { this.items.push(item.innerHTML); } });
    elements["#toast"] = stack();
    elements["#toast-alert"] = stack();
    const steps = [];
    const dialog = {
      open: false, html: "", regions: {},
      insertAdjacentHTML(where, html) { steps.push("regions"); this.html += html; },
      showModal() { steps.push("showModal"); this.open = true; },
      querySelector(sel) {
        const name = (sel.match(/data-toast="([^"]+)"/) || [])[1];
        if (!name || !this.html.includes(`data-toast="${name}"`)) return null;
        return (this.regions[name] ||= stack());
      },
    };
    document.querySelector = (sel) => (sel === "dialog[open]" ? (dialog.open ? dialog : null) : el(sel));
    const texts = (items) => items.map((html) => html.replace(/<[^>]+>/g, "").replace("&times;", ""));
"""


@needs_node
def test_a_message_raised_in_a_dialog_is_shown_inside_it(tmp_path):
    """U09 (audit 2026-09-30): an open modal dialog makes everything outside it inert -- out
    of the accessibility tree -- and paints it under the backdrop, both toast stacks
    included. "Copied", or why a delete failed, was neither heard nor properly seen while the
    model panel was open. Each dialog now holds a pair of its own, inserted empty BEFORE it
    opens: a live region has to exist before its message does to be announced."""
    result = _run(tmp_path, helpers=("toasts.js:pushToast", "toasts.js:toastRegion",
                           "toasts.js:toastError", "toasts.js:openModal"),
                  body=_TOASTS + """
        toastError({ message: "before" });
        openModal(dialog);
        toastError({ message: "inside" });
        dialog.open = false;
        toastError({ message: "after" });
        report({ page: texts(elements["#toast-alert"].items),
                 dialog: texts((dialog.regions["toast-alert"] || stack()).items), steps, errors });
    """)

    assert result["dialog"] == ["inside"], "the message went under the dialog"
    assert result["page"] == ["before", "after"]
    assert result["steps"] == ["regions", "showModal"]
    assert result["errors"] == []


# A button as the browser treats one under its focus fixup rule: when the focused element
# turns disabled, the focus falls to <body>, and enabling it again does not bring it back.
_FOCUSABLE = """
    document.body = { tag: "body" };
    document.activeElement = document.body;
    const focusable = (button) => {
      let disabled = false;
      Object.defineProperty(button, "disabled", { get: () => disabled, set: (value) => {
        disabled = value;
        if (value && document.activeElement === button) document.activeElement = document.body;
      } });
      button.isConnected = true;
      button.focus = () => { if (!disabled) document.activeElement = button; };
      return button;
    };
"""


@needs_node
def test_a_busy_button_gets_its_focus_back(tmp_path):
    """U09 (audit 2026-09-30): after every submit the focus was on <body> -- a keyboard or
    screen-reader user back at the top of the page, a whole shell away from the form. It is
    given back only where the button had it, and only if nothing else has taken it since."""
    result = _run(tmp_path, helpers=("app.js:busy",), body=_FOCUSABLE + """
        const pressed = focusable({ name: "pressed" });
        pressed.focus();
        const idle = busy(pressed);
        const during = { disabled: pressed.disabled, onBody: document.activeElement === document.body };
        idle();
        const back = document.activeElement === pressed;

        const elsewhere = { name: "field" };
        pressed.focus();
        const idle2 = busy(pressed);
        document.activeElement = elsewhere;          // the user tabbed on meanwhile
        idle2();
        report({ during, back, enabled: !pressed.disabled, kept: document.activeElement === elsewhere });
    """)

    assert result == {"during": {"disabled": True, "onBody": True}, "back": True,
                      "enabled": True, "kept": True}


@needs_node
def test_classifying_leaves_the_focus_on_the_classify_button(tmp_path):
    """The audit's own case, through the real handler: query.js disabled #query-btn for the
    request and the focus never came back."""
    result = _run(tmp_path, modules=("query.js",), helpers=("app.js:busy",), body=_FOCUSABLE + """
        el('input[name="query-mode"]:checked').value = "one";
        document.querySelectorAll = (sel) =>
          (sel === 'input[name="query-model"]:checked' ? [{ value: "m" }] : []);
        el("#query-text").value = "Pythagoras";
        el("#query-topk").value = "";
        Api.post = async () => ({ results: [{ predictions: [{ uri: "u", label: "M", confidence: 1 }] }] });
        globalThis.fmtFixed = (v, d) => Number(v).toFixed(d);
        globalThis.applyBarWidths = () => {};
        globalThis.bindExplainButtons = () => {};
        globalThis.bindCorrectionButtons = () => {};
        const button = focusable(el("#query-btn"));
        button.focus();
        await onQuery({ preventDefault() {} });
        report({ focused: document.activeElement === button, enabled: !button.disabled, errors });
    """)

    assert result == {"focused": True, "enabled": True, "errors": []}



# --- U10: the training card and its Stop button -----------------------------------------------

# `t` with its parameters, so a value that changes reads as changed; and a card whose row list
# counts what is rebuilt and which rows are written.
_STATUS_PRELUDE = _PRELUDE.replace(
    "const t = (key) => key;",
    "const t = (key, params) => (params ? `${key} ${JSON.stringify(params)}` : key);")
_STATUS_CARD = """
    let rebuilt = 0;
    Object.defineProperty(el("#train-status"), "innerHTML", { set() { rebuilt += 1; } });
    const rows = el("#train-status-rows");
    let built = 0;
    const written = [];
    rows.children = [];
    Object.defineProperty(rows, "innerHTML", { set(markup) {
      built += 1;
      rows.children = [...markup.matchAll(/<(dt|dd)>/g)].map((m, at) => {
        let text = "";
        return { get textContent() { return text; },
                 set textContent(value) { written.push(at); text = value; } };
      });
    } });
    let bound = 0;
    el("#train-stop").addEventListener = () => { bound += 1; };
    const running = (elapsed) => ({
      status: "running", phase: "C-Auswahl", phase_detail: "Fold 2/3", model_name: "fach",
      progress: 40, elapsed_seconds: elapsed, eta_seconds: 600 - elapsed, queued: ["fach_b"],
      rss_mb: 900, peak_rss_mb: 1200, head_fit_threads: 4, threads_requested: 4,
      seconds_since_heartbeat: 1 });
"""


@needs_node
def test_a_poll_tick_updates_the_training_card_in_place(tmp_path):
    """U10 (audit 2026-09-30): the card was rebuilt from scratch every 2.5 s while a run was
    going, so the focus fell off "Stop" between Tab and Enter, and a phase or model name
    being selected to copy lost its selection. The rows are now written in place, and only
    the ones whose text changed -- here the elapsed time and the estimate."""
    result = _run(tmp_path, modules=("train-status.js",), helpers=("app.js:applyBarWidths",),
                  prelude=_STATUS_PRELUDE, body=_STATUS_CARD + """
        renderTrainStatus(running(100));
        written.length = 0;
        renderTrainStatus(running(102.5));
        report({ rebuilt, built, written, bound, stopShown: !el("#train-stop").hidden, errors });
    """)

    assert result["rebuilt"] == 0, "the card was replaced"
    assert result["built"] == 1, "the row list was rebuilt on the second tick"
    assert result["written"] == [9, 11], "rows were rewritten that had not changed"
    assert result["bound"] == 0, "rendering bound a listener: Stop belongs to the page"
    assert result["stopShown"] is True
    assert result["errors"] == []


@needs_node
def test_stopping_asks_first_and_names_the_queue_it_empties(tmp_path):
    """U10: "Stop training" posted at once -- and stopping also empties the whole queue, so
    one click could throw away a batch of runs. It asks first, and says how many go with it."""
    result = _run(tmp_path, modules=("train-status.js", "training.js"),
                  helpers=("app.js:applyBarWidths",),
                  prelude=_STATUS_PRELUDE, body=_STATUS_CARD + """
        const asked = [];
        let answer = false;
        globalThis.confirm = (question) => { asked.push(question); return answer; };
        globalThis.toast = () => {};
        globalThis.toastError = (err) => errors.push(String(err.message || err));
        const posted = [];
        Api.post = async (url) => { posted.push(url); return {}; };
        renderTrainStatus(running(100));
        await stopTraining();
        const declined = [...posted];
        answer = true;
        await stopTraining();
        report({ asked, declined, posted, errors });
    """)

    assert result["declined"] == [], "a declined question still stopped the run"
    assert result["posted"] == ["/train/stop"]
    assert result["asked"][0].startswith("trainStatus.stopConfirmQueue")
    assert '"count":1' in result["asked"][0] and '"name":"fach"' in result["asked"][0]
    assert result["errors"] == []



@needs_node
def test_stopping_also_cancels_the_runs_this_page_has_not_sent(tmp_path):
    """Review of U05/U10 (2026-10-01): a batch waiting out the rate limit went on after Stop.
    Seven fields, five accepted, the sixth refused with 429 and the page waiting 60 s; Stop
    asked about the four runs on the server, the server stopped and cleared its queue -- and
    the page then sent the sixth and seventh as if nothing had happened. Stop now counts the
    runs the page still holds, and ends the wait instead of sitting it out. The timers here
    never fire on their own: the stop has to come while the page is waiting."""
    result = _run(tmp_path, modules=("train-status.js", "training.js"), helpers=("app.js:busy",),
                  prelude=_STATUS_PRELUDE, body="""
        const timers = [];
        globalThis.setTimeout = (run, ms) => { timers.push({ run, ms }); return timers.length; };
        const toasts = [];
        globalThis.toast = (message) => toasts.push(message);
        globalThis.toastError = (err) => errors.push(String(err.message || err));
        el("#train-dataset").value = "data.csv";
        el("#train-name").value = "fach";
        pickers[0].values = () => ["title"];
        pickers[1].values = () => ["f1", "f2", "f3", "f4", "f5", "f6", "f7"];
        const posted = [];
        Api.post = async (url, body) => {
          posted.push(body && body.model_name ? body.model_name : url);
          if (posted.length === 6) {
            throw Object.assign(new Error("Rate limit exceeded"), { status: 429, retryAfter: 60 });
          }
          return { status: "queued" };
        };
        const batch = onTrainStart({ preventDefault() {} });
        for (let i = 0; i < 100 && !timers.length; i += 1) await Promise.resolve();
        const waiting = { label: el("#train-btn").textContent, timers: timers.map((t) => t.ms) };
        lastStatus = { status: "running", model_name: "fach_f1",
                       queued: ["fach_f2", "fach_f3", "fach_f4", "fach_f5"] };
        const asked = [];
        globalThis.confirm = (question) => { asked.push(question); return true; };
        await stopTraining();
        timers.forEach((t) => t.run());   // the 60 s pass either way
        await batch;
        report({ posted, asked, waiting, toasts, errors, idle: !el("#train-btn").disabled });
    """)

    assert result["waiting"]["timers"] == [60_000]
    assert result["waiting"]["label"].startswith("train.waitingButton"), "the wait shows nowhere lasting"
    assert '"count":6' in result["asked"][0], "the question left out the runs the page still holds"
    assert result["posted"] == ["fach_f1", "fach_f2", "fach_f3", "fach_f4", "fach_f5", "fach_f6",
                                "/train/stop"], "runs were sent after the stop"
    assert result["toasts"][-1].startswith("train.batchCancelled")
    assert '"count":2' in result["toasts"][-1]
    assert result["idle"] is True
    assert result["errors"] == []



@needs_node
def test_a_choice_made_while_the_tab_reloads_is_kept(tmp_path):
    """Review of U08 (2026-10-01): the choices were read BEFORE the lists were fetched and put
    back after -- so a dataset picked while `GET /datasets` was still running (the select
    still offers the old list) was reverted when the answer came, and the run went out for
    the other dataset with this one's columns."""
    result = _run(tmp_path, modules=("training.js",), body=_FAKE_SELECT + """
        await loadTrainingTab();
        el("#train-dataset").value = "b.csv";
        let release;
        const slow = new Promise((resolve) => { release = resolve; });
        const answer = Api.get;
        Api.get = (url) => (url === "/datasets" ? slow.then(() => datasets) : answer(url));
        const reload = loadTrainingTab();
        el("#train-dataset").value = "a.csv";
        release();
        await reload;
        report({ dataset: el("#train-dataset").value, errors });
    """)

    assert result == {"dataset": "a.csv", "errors": []}


@needs_node
def test_a_model_ticked_while_the_list_reloads_stays_ticked(tmp_path):
    """The same on the Query tab: a model ticked during `GET /models` was unticked again."""
    result = _run(tmp_path, modules=("query.js",), body="""
        let release;
        const slow = new Promise((resolve) => { release = resolve; });
        let calls = 0;
        Api.get = async () => { calls += 1; if (calls > 1) await slow; return ["m1", "m2"]; };
        const box = el("#query-models");
        let boxes = [];
        const read = () => {
          boxes = [...box.innerHTML.matchAll(/value="([^"]*)"( checked)?/g)]
            .map(([, value, checked]) => ({ value, checked: Boolean(checked) }));
        };
        box.querySelector = () => boxes[0] || null;
        document.querySelectorAll = (sel) =>
          (sel === 'input[name="query-model"]:checked' ? boxes.filter((b) => b.checked) : []);
        await loadQueryTab(); read();
        const reload = loadQueryTab();
        boxes[1].checked = true;
        release();
        await reload; read();
        report({ checked: boxes.filter((b) => b.checked).map((b) => b.value), errors });
    """)

    assert result == {"checked": ["m1", "m2"], "errors": []}



@needs_node
def test_the_focus_stays_on_the_card_when_a_finished_run_hides_stop(tmp_path):
    """Review of U10 (2026-10-01): after a confirmed stop the button keeps the focus, and the
    next poll hides it -- which drops the focus to <body>, the U09 symptom over again. The
    card's heading takes it instead."""
    result = _run(tmp_path, modules=("train-status.js",), helpers=("app.js:applyBarWidths",),
                  prelude=_STATUS_PRELUDE, body=_STATUS_CARD + """
        const stop = el("#train-stop");
        const heading = el("#train-status-heading");
        heading.focus = () => { document.activeElement = heading; };
        renderTrainStatus(running(100));
        document.activeElement = stop;
        renderTrainStatus({ ...running(130), status: "stopped", queued: [] });
        report({ onHeading: document.activeElement === heading, stopHidden: stop.hidden, errors });
    """)

    assert result == {"onHeading": True, "stopHidden": True, "errors": []}


@needs_node
def test_the_queue_line_is_written_only_when_it_changes(tmp_path):
    """Review of U10: the line naming the queued runs was rewritten on every poll, so a name
    being selected to copy lost its selection every 2.5 s, as the rows did -- and the live
    region it is got a fresh node each time."""
    result = _run(tmp_path, modules=("train-status.js",), helpers=("app.js:applyBarWidths",),
                  prelude=_STATUS_PRELUDE, body=_STATUS_CARD + """
        const line = el("#train-queue");
        let text = "", writes = 0;
        Object.defineProperty(line, "textContent", { get: () => text,
                                                     set: (value) => { writes += 1; text = value; } });
        renderTrainStatus(running(100));
        renderTrainStatus(running(102.5));
        renderTrainStatus(running(105));
        report({ writes, text, errors });
    """)

    assert result["writes"] == 1, f"the queue line was written {result['writes']} times"
    assert "fach_b" in result["text"]
    assert result["errors"] == []
