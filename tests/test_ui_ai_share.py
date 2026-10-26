"""The admin UI says how much of a dataset an LLM wrote before anyone trains on it.

`GET /datasets/{name}` counts the AI-marked rows (`ai_rows`); the training form shows that
count beside the choice what to do with them, and the dataset detail view shows it too.
Same node harness as test_ui_handlers.py.
"""

from test_ui_handlers import _PRELUDE, _run, needs_node

_PRELUDE_HERE = (_PRELUDE
                 .replace("const t = (key) => key;",
                          "const t = (key, params) => params ? `${key} ${JSON.stringify(params)}` : key;")
                 .replace("const esc = (s) => String(s);", ""))

_STUBS = """
globalThis.fmtFixed = (v, digits) => v.toFixed(digits);
"""

MARKED_INFO = """{ name: "mix.csv", rows: 1000,
  columns: ["title", "label", "generated_for", "enriched_fields"],
  sample: [], ai_rows: { generated: 200, example: 0, enriched: 50, marked: 250 } }"""


@needs_node
def test_the_share_is_one_sentence_and_absent_without_marks(tmp_path):
    result = _run(tmp_path, prelude=_PRELUDE_HERE, modules=("escape.js",),
                  helpers=("dataset-detail.js:aiShareText",), body=_STUBS + f"""
        report({{ marked: aiShareText({MARKED_INFO}),
                  plain: aiShareText({{ rows: 10, columns: ["title"] }}),
                  empty: aiShareText({{ rows: 0, ai_rows: {{ marked: 0 }} }}) }});
    """)

    assert result["marked"] == ('datasetDetail.aiShare {"count":250,"rows":1000,"percent":"25.0",'
                                '"generated":200,"enriched":50,"example":0}')
    assert result["plain"] == ""
    assert result["empty"] == "", "no rows, no share to state"


@needs_node
def test_the_training_form_shows_the_share_beside_the_choice(tmp_path):
    result = _run(tmp_path, prelude=_PRELUDE_HERE, modules=("escape.js", "training.js"),
                  helpers=("dataset-detail.js:aiShareText",), body=_STUBS + f"""
        el("#train-dataset").value = "mix.csv";
        Api.get = async () => ({MARKED_INFO});
        await loadDatasetColumns();
        const marked = {{ text: el("#train-synthetic-share").textContent,
                          hidden: el("#train-synthetic-share").hidden,
                          section: el("#train-synthetic").hidden }};
        Api.get = async () => ({{ rows: 3, columns: ["title", "label"], sample: [] }});
        await loadDatasetColumns();
        report({{ marked, plain: el("#train-synthetic-share").hidden, errors }});
    """)

    assert result["marked"]["text"].startswith('datasetDetail.aiShare {"count":250')
    assert result["marked"]["hidden"] is False
    assert result["marked"]["section"] is False
    assert result["plain"] is True, "a dataset without marks shows no share"
    assert result["errors"] == []


@needs_node
def test_the_dataset_detail_shows_the_share_and_that_it_never_validates(tmp_path):
    result = _run(tmp_path, prelude=_PRELUDE_HERE, modules=("escape.js",),
                  helpers=("dataset-detail.js:aiShareText", "dataset-detail.js:aiShareSection"),
                  body=_STUBS + f"""
        report({{ marked: aiShareSection({MARKED_INFO}), plain: aiShareSection({{ rows: 3 }}) }});
    """)

    assert 'datasetDetail.aiShare {"count":250' in result["marked"]
    assert "datasetDetail.aiShareNote" in result["marked"]
    assert result["plain"] == ""
