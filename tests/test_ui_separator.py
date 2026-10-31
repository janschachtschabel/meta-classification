"""The admin UI reads a dataset with the separator its header uses, and says which.

The UI never sent a separator: every read took the API's default `;`, so a comma- or
tab-separated upload showed one column holding the whole header and could not be trained from
the UI. GET /datasets/{name} now detects it; the training form shows it in a field one can
change, and every request that reads the dataset sends it on.
Same node harness as test_ui_handlers.py.
"""

from test_ui_handlers import _DATASET_PANEL, _run, needs_node

_ANSWER = """
const urls = [];
Api.get = async (url) => {
  urls.push(url);
  const named = url.includes("separator=") ? decodeURIComponent(url.split("separator=")[1]) : null;
  return { columns: ["title", "labels"], separator: named ?? ",", sample: [], rows: 2 };
};
"""


@needs_node
def test_the_training_form_detects_then_follows_a_changed_separator(tmp_path):
    result = _run(tmp_path, modules=("training.js",), helpers=("dataset-detail.js:aiShareText",),
                  body=_ANSWER + """
        el("#train-dataset").value = "comma.csv";
        await loadDatasetColumns();
        const detected = el("#train-separator").value;
        el("#train-separator").value = ";";
        await loadDatasetColumns();               // the field changed: read with that one
        el("#train-dataset").value = "other.csv";
        await loadDatasetColumns();               // another dataset: detect again
        report({ urls, detected, after: el("#train-separator").value, errors });
    """)

    assert result["urls"] == ["/datasets/comma.csv", "/datasets/comma.csv?separator=%3B",
                              "/datasets/other.csv"]
    assert result["detected"] == ","
    assert result["after"] == ","
    assert result["errors"] == []


@needs_node
def test_a_run_and_its_preflight_send_the_separator(tmp_path):
    result = _run(tmp_path, modules=("training.js", "preflight.js"), helpers=("app.js:busy",), body="""
        globalThis.toast = () => {};
        globalThis.analysisHtml = () => "";
        globalThis.preflightSummary = () => "";
        el("#train-dataset").value = "comma.csv";
        el("#train-separator").value = "\\t";
        el("#train-name").value = "fach";
        pickers[0].values = () => ["title"];
        pickers[1].values = () => ["labels"];
        const posted = [];
        Api.post = async (url, body) => { posted.push([url, body.csv_separator]); return { status: "running" }; };
        await runPreflight();
        await onTrainStart({ preventDefault() {} });
        report({ posted, errors });
    """)

    assert ["/datasets/analyze", "\t"] in result["posted"], result
    assert ["/train", "\t"] in result["posted"], result


@needs_node
def test_the_dataset_panel_analyses_with_the_separator_it_was_shown_with(tmp_path):
    result = _run(tmp_path, modules=("dataset-detail.js",), helpers=("app.js:busy",),
                  body=_DATASET_PANEL + """
        const sent = [];
        Api.post = async (url, body) => {
          sent.push(body.csv_separator);
          return { total_samples: 1, unique_labels: 1, label_threshold_analysis: {},
                   recommended_min_samples_per_label: 5, rare_labels_under_10: {} };
        };
        const open = showDatasetDetail("comma.csv");
        answers["/datasets/comma.csv"]({ columns: ["title", "labels"], separator: ",", sample: [] });
        await open;
        await frames[0].querySelector("#ds-analyze").run();
        report({ sent, html: frames[0].innerHTML, errors });
    """)

    assert result["sent"] == [","]
    assert "datasetDetail.separator" in result["html"], "the panel says which separator it read with"
