"""What the Query view renders, executed rather than pattern-matched.

`query.js` is a plain script with no imports and no DOM access at load time, so node
can run it with the handful of globals it uses stubbed out. That makes the answers it
builds testable as behaviour — which is what the near-miss follow-up needed: the
failure it can produce is a thrown card, and a regex over the source cannot see that.

Skipped where node is absent: the UI has no toolchain on purpose, so node is a
convenience here, not a build dependency.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

QUERY_JS = Path(__file__).parent.parent / "app" / "static" / "ui" / "query.js"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# The globals query.js expects from the page, reduced to what a render needs: `t`
# returns its key (so an assertion can name the string it wants), `esc` passes through,
# which keeps the probe about structure rather than escaping, and `fmtFixed` stands in
# for the locale-aware formatter in i18n.js.
_HARNESS = """
const fs = require("fs"), vm = require("vm");
const source = fs.readFileSync(process.argv[2], "utf8");
// _outStub stands in for the results container: renderBulkTable writes the table into
// `innerHTML` and then binds the download button, so the stub has to answer both.
const sandbox = { t: (key) => key, esc: (s) => String(s), document: {}, console,
                  fmtFixed: (v, d) => Number(v).toFixed(d),
                  Api: { saveBlob: () => {} },
                  _outStub: () => ({ innerHTML: "",
                                     querySelector: () => ({ addEventListener: () => {} }) }) };
try {
  // Appended to the source so it runs in the same scope: `const`/function
  // declarations of a script stay lexical and never reach the sandbox object.
  const html = vm.runInNewContext(source + "\\n" + process.argv[3], sandbox);
  console.log(JSON.stringify({ ok: true, html: String(html) }));
} catch (err) {
  console.log(JSON.stringify({ ok: false, error: String(err) }));
}
"""


def _render(expression: str, tmp_path: Path) -> dict:
    """Evaluate one expression against the real query.js and return its result."""
    harness = tmp_path / "harness.cjs"
    harness.write_text(_HARNESS, encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["node", str(harness), str(QUERY_JS), expression],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(done.stdout)


@needs_node
def test_a_model_named_like_an_object_property_still_renders(tmp_path):
    """The near-miss lookup is keyed by model name, and a model may legitimately be
    called `constructor` or `toString` (`safe_name` rejects paths, not property names).
    Read straight off the object, such a name returns something from Object.prototype
    instead of "nothing came back" — and the card then throws, replacing the whole
    answer (including the models that DID return) with an error."""
    result = _render('renderPredictions({ constructor: [] }, {})', tmp_path)

    assert result["ok"], result.get("error")
    assert "query.noLabelAboveThreshold" in result["html"]
    assert "query.nearestBelow" not in result["html"], "nothing was nearly matched"


@needs_node
def test_the_nearest_misses_are_rendered_under_the_decline_note(tmp_path):
    """The point of the follow-up: a model that reported nothing says what it almost
    said, below the note explaining why the list is empty."""
    nearest = '{ m: [{ uri: "u", label: "Chemie", confidence: 0.4, baseline_diff: 0.1 }] }'
    result = _render(f'renderPredictions({{ m: [] }}, {nearest})', tmp_path)

    assert result["ok"], result.get("error")
    html = result["html"]
    assert html.index("query.noLabelAboveThreshold") < html.index("query.nearestBelow")
    assert "Chemie" in html


@needs_node
@pytest.mark.parametrize(
    ("models", "metadata", "mode", "expected"),
    [
        # The ask: descriptive metadata on its own, with no model chosen at all. The progress
        # key travels with the plan because the status region is the page's only live region:
        # announcing "classifying" while nothing is classified misinforms a screen reader.
        ("[]", "true", "one",
         {"classify": False, "metadata": True, "progress": "query.progressDescribing"}),
        ("[]", "true", "many",
         {"classify": False, "metadata": True, "progress": "query.progressDescribing"}),
        # …and everything in one submit, one model or several.
        ('["faecher"]', "true", "one",
         {"classify": True, "metadata": True, "progress": "query.progressStart"}),
        ('["faecher", "stufe"]', "true", "one",
         {"classify": True, "metadata": True, "progress": "query.progressStart"}),
        ('["faecher"]', "false", "one",
         {"classify": True, "metadata": False, "progress": "query.progressStart"}),
    ],
)
def test_what_a_submit_asks_for(models, metadata, mode, expected, tmp_path):
    """The decision `onQuery` acts on, as a value rather than a branch inside a submit
    handler — which is the only way to reach the model-free case without a form."""
    result = _render(f"JSON.stringify(queryPlan({models}, {metadata}, {mode!r}))", tmp_path)

    assert result["ok"], result.get("error")
    assert json.loads(result["html"]) == expected


@needs_node
@pytest.mark.parametrize(
    ("models", "metadata", "mode"),
    [
        ("[]", "false", "one"),    # nothing asked for at all
        ("[]", "false", "many"),
        # The metadata option is only HIDDEN in csv mode, so a box ticked before the
        # switch stays ticked. Honouring it here would run /predict/csv with no model.
        ("[]", "true", "csv"),
    ],
)
def test_a_submit_that_asks_for_nothing_is_refused(models, metadata, mode, tmp_path):
    result = _render(f"JSON.stringify(queryPlan({models}, {metadata}, {mode!r}))", tmp_path)

    assert result["ok"], result.get("error")
    assert "error" in json.loads(result["html"])


@needs_node
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("one", "query.error.nothingRequested"),
        ("many", "query.error.nothingRequested"),
        # In csv mode the metadata option is HIDDEN, so pointing at it would send the reader
        # looking for a checkbox that is not on screen. There the only way out is a model.
        ("csv", "query.error.noModel"),
    ],
)
def test_the_refusal_only_offers_a_way_out_that_exists_in_that_mode(mode, expected, tmp_path):
    result = _render(f"JSON.stringify(queryPlan([], false, {mode!r}))", tmp_path)

    assert result["ok"], result.get("error")
    assert json.loads(result["html"])["error"] == expected


@needs_node
def test_the_download_drops_the_label_columns_when_nothing_was_classified(tmp_path):
    """The table and the file have to agree about what a run produced. A `confidence` column
    holding 0 for every row reads as "the model was unsure", not as "no model ran"."""
    rows = ('[{ row: 0, text: "a", uri: "", label: "", confidence: 0, title: "T", '
            'description: "D", keywords: ["k"] }]')
    result = _render(f"bulkCsv({rows}, false)", tmp_path)

    assert result["ok"], result.get("error")
    assert result["html"].split("\n")[0] == "row,text,title,description,keywords"


@needs_node
def test_the_download_keeps_the_label_columns_when_a_model_ran(tmp_path):
    rows = '[{ row: 0, text: "a", uri: "u", label: "Physik", confidence: 0.9 }]'
    result = _render(f"bulkCsv({rows}, true)", tmp_path)

    assert result["ok"], result.get("error")
    assert result["html"].split("\n")[0] == "row,text,uri,label,confidence"


@needs_node
def test_the_bulk_table_drops_the_label_columns_when_nothing_was_classified(tmp_path):
    """A metadata-only run over a list has no labels to show, and a table claiming "0 labels
    assigned · 2 without label" would read as two failures rather than as a question never
    asked. The title and keyword columns stay — they are the answer."""
    rows = ('[{ row: 0, text: "a", uri: "", label: "", confidence: 0, title: "Erstes", '
            'keywords: ["k1"] }, { row: 1, text: "b", uri: "", label: "", confidence: 0, '
            'title: "Zweites", keywords: ["k2"] }]')
    # The table goes into `out.innerHTML`; the return value is only the status line.
    result = _render(
        f'(() => {{ const out = _outStub();'
        f' renderBulkTable({rows}, ["a", "b"], out, false); return out.innerHTML; }})()',
        tmp_path,
    )

    assert result["ok"], result.get("error")
    html = result["html"]
    assert "query.table.title" in html and "query.table.keywords" in html
    assert "query.table.label" not in html, "the label column survived a model-free run"
    assert "query.table.confidence" not in html
    assert "query.bulk.withoutLabel" not in html, "counted a missing label nobody asked for"


@needs_node
def test_the_bulk_table_still_shows_labels_when_a_model_ran(tmp_path):
    """The other half of the same switch: with a model, nothing about the table changes."""
    rows = '[{ row: 0, text: "a", uri: "u", label: "Physik", confidence: 0.9 }]'
    result = _render(
        f'(() => {{ const out = _outStub();'
        f' renderBulkTable({rows}, ["a"], out, true); return out.innerHTML; }})()',
        tmp_path,
    )

    assert result["ok"], result.get("error")
    assert "query.table.label" in result["html"]
    assert "Physik" in result["html"]


@needs_node
def test_the_metadata_card_renders_all_three_fields(tmp_path):
    """The option's whole output. Keywords are a list, so they have to survive as one —
    joining them into a sentence would lose the ranking the endpoint returns them in."""
    proposal = ('{ title: "Die Fotosynthese", description: "Pflanzen stellen ihre Nahrung '
                'selbst her.", keywords: ["Fotosynthese", "Chlorophyll"] }')
    result = _render(f"metadataCard({proposal})", tmp_path)

    assert result["ok"], result.get("error")
    html = result["html"]
    assert "Die Fotosynthese" in html
    assert "Pflanzen stellen ihre Nahrung selbst her." in html
    assert "Fotosynthese" in html and "Chlorophyll" in html


@needs_node
def test_a_field_the_text_gave_nothing_for_says_so(tmp_path):
    """The endpoint answers a thin text with empty fields rather than an error, so the card
    has to render that state. An empty <p> would read as a rendering bug; naming it as "the
    text yielded none" is the difference between a blank and an answer."""
    result = _render('metadataCard({ title: "", description: "", keywords: [] })', tmp_path)

    assert result["ok"], result.get("error")
    assert result["html"].count("query.metadata.none") == 3, result["html"]


@needs_node
def test_the_bulk_csv_carries_the_metadata_columns_when_asked(tmp_path):
    """The bulk answer is used by downloading it, so the metadata has to be IN the file.

    One row per predicted label means an input row's metadata repeats across its labels —
    correct for a flat export, and the reason the columns are appended rather than spliced
    in: a consumer that already parses this file by position keeps working.
    """
    row = ('{ row: 0, text: "t", uri: "u", label: "Chemie", confidence: 0.5, '
           'title: "Titel", description: "Beschreibung", keywords: ["a", "b"] }')
    result = _render(f"bulkCsv([{row}])", tmp_path)

    assert result["ok"], result.get("error")
    header, first = result["html"].split("\n")
    assert header == "row,text,uri,label,confidence,title,description,keywords"
    # Keywords are one cell, separated as the API's own label lists are.
    assert first.endswith('"Titel","Beschreibung","a; b"'), first


@needs_node
def test_the_bulk_csv_is_unchanged_when_no_metadata_was_asked_for(tmp_path):
    """Off by default means the file a consumer already parses does not change shape."""
    row = '{ row: 0, text: "t", uri: "u", label: "Chemie", confidence: 0.5 }'
    result = _render(f"bulkCsv([{row}])", tmp_path)

    assert result["ok"], result.get("error")
    assert result["html"].split("\n")[0] == "row,text,uri,label,confidence"


def test_the_metadata_option_is_hidden_where_it_cannot_be_delivered():
    """`/predict/csv` streams its answer from the server, and this view never holds the
    rows — so in CSV mode there is nothing to attach metadata to.

    The same rule the two reliability signals already follow, for the reason written beside
    them: offering a control that does nothing is a small lie the rest of this UI does not
    tell. Asserted because it is a decision, and a later mode added without thinking about
    it would silently re-introduce the dead control.
    """
    source = QUERY_JS.read_text(encoding="utf-8")
    handler = source[source.index("function onQueryModeChange"):]
    handler = handler[:handler.index("\n}\n")]

    assert '$("#query-metadata-option").hidden = mode === "csv"' in handler, handler


@needs_node
def test_the_bulk_table_shows_the_metadata_it_was_asked_for(tmp_path):
    """Ticking the box has to change what is on screen, not only what a download contains.

    The description is deliberately not a column — 500 characters per row would push the
    labels off the side — so the table carries the title and the keywords and the note says
    where the full text is.
    """
    rows = ('[{ row: 0, text: "t", uri: "u", label: "Chemie", confidence: 0.5, '
            'title: "Titel", description: "Lange Beschreibung", keywords: ["a", "b"] }]')
    result = _render(f"(() => {{ const out = _outStub(); renderBulkTable({rows}, "
                     f'["t"], out); return out.innerHTML; }})()', tmp_path)

    assert result["ok"], result.get("error")
    html = result["html"]
    assert "query.table.title" in html and "query.table.keywords" in html
    assert "Titel" in html and "a, b" in html
    assert "query.bulk.descriptionInDownload" in html


@needs_node
def test_the_bulk_table_keeps_its_shape_without_metadata(tmp_path):
    """The columns appear only when there is something to put in them."""
    rows = '[{ row: 0, text: "t", uri: "u", label: "Chemie", confidence: 0.5 }]'
    result = _render(f"(() => {{ const out = _outStub(); renderBulkTable({rows}, "
                     f'["t"], out); return out.innerHTML; }})()', tmp_path)

    assert result["ok"], result.get("error")
    assert "query.table.title" not in result["html"]
