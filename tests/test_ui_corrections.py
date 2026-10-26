"""The Corrections tab: what editors corrected, seen, backed up, restored and trained on.

Driven through the real module in node, with the API answered by the test -- the same harness
as test_ui_handlers.py.
"""

from test_ui_handlers import _PRELUDE, _run, needs_node

# The real escape (loaded as a module below) and a t() that shows its parameters: the shared
# prelude's pass-throughs are `const`s, which a stub on globalThis cannot override.
_PRELUDE_HERE = (_PRELUDE
                 .replace("const t = (key) => key;",
                          "const t = (key, params) => params ? `${key} ${JSON.stringify(params)}` : key;")
                 .replace("const esc = (s) => String(s);", ""))
_MODULES = ("escape.js", "corrections.js")

_STUBS = """
globalThis.fmtDate = (iso) => `date(${iso})`;
globalThis.toasts = [];
globalThis.toast = (msg) => toasts.push(msg);
globalThis.busy = () => () => {};
function answerPages(pages) {
  Api.get = async (url) => { apiCalls.push(url); return pages.shift(); };
}
const item = (n, extra = {}) => ({ recorded_at: `t${n}`, model_name: "fach", text: `Text ${n}`,
  text_truncated: false, predicted: ["uri:a"], corrected: ["uri:b"], source: "ui", ...extra });
"""


@needs_node
def test_the_tab_lists_the_corrections_and_offers_more(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answerPages([
          { total: 3, models: { fach: 2, stufe: 1 }, items: [item(3), item(2, { text: "<b>2</b>", text_truncated: true })] },
          { total: 3, models: { fach: 2, stufe: 1 }, items: [item(1, { corrected: [] })] },
        ]);
        await loadCorrections();
        const first = { html: el("#corrections-list").innerHTML, more: el("#corrections-more").hidden,
                        summary: el("#corrections-summary").textContent,
                        options: el("#corrections-model").innerHTML };
        // The harness has no real DOM: what the next page appends is what we can observe.
        const appended = [];
        el("#corrections-list tbody").insertAdjacentHTML = (where, html) => appended.push([where, html]);
        await loadCorrections(true);
        report({ first, html: appended.map(([, h]) => h).join(""), where: appended.map(([w]) => w),
                 more: el("#corrections-more").hidden,
                 calls: apiCalls, errors });
    """)

    first = result["first"]
    assert result["calls"] == ["/feedback?limit=50&offset=0", "/feedback?limit=50&offset=2"]
    assert first["html"].index("Text 3") < first["html"].index("&lt;b&gt;2"), "newest first, escaped"
    assert "corrections.truncated" in first["html"]
    assert first["more"] is False, "one more correction is waiting"
    assert 'corrections.summary {"shown":2,"total":3}' in first["summary"]
    assert 'value="fach"' in first["options"] and 'value="stufe"' in first["options"]
    assert "Text 1" in result["html"] and "Text 3" not in result["html"], "only the next page is added"
    assert result["where"] == ["beforeend"], "appended below, not replacing"
    assert "corrections.noneApply" in result["html"], "an empty correction says so"
    assert result["more"] is True
    assert result["errors"] == []


@needs_node
def test_filtering_by_model_starts_over(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answerPages([{ total: 1, models: { fach: 1 }, items: [item(1)] }]);
        el("#corrections-model").value = "fach & co";
        await loadCorrections();
        report({ calls: apiCalls });
    """)

    assert result["calls"] == ["/feedback?limit=50&offset=0&model=fach%20%26%20co"]


@needs_node
def test_empty_and_readonly_states(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answerPages([{ total: 0, models: {}, items: [] }]);
        await loadCorrections();
        const empty = el("#corrections-list").innerHTML;
        Api.get = async () => { const err = new Error("Forbidden"); err.status = 403; throw err; };
        await loadCorrections();
        report({ empty, denied: el("#corrections-list").innerHTML, more: el("#corrections-more").hidden });
    """)

    assert "corrections.empty" in result["empty"]
    assert "corrections.adminOnly" in result["denied"], "a readonly key is told why, not 'Forbidden'"
    assert result["more"] is True


@needs_node
def test_saving_as_a_dataset_and_importing_a_backup(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        const posted = [];
        Api.post = async (url, json) => { posted.push([url, json]); return { dataset_name: "k.csv", rows: 7 }; };
        Api.postForm = async (url, form) => { posted.push([url, form.get("file") instanceof Blob]);
                                              return { imported: 4, duplicates: 1, invalid: 2 }; };
        answerPages([{ total: 0, models: {}, items: [] }]);
        el("#corrections-dataset-name").value = "  k  ";
        await onCorrectionsDataset({ preventDefault() {} });
        el("#corrections-import-file").files = [Object.assign(new Blob(["{}"]), { name: "b.jsonl" })];
        await onCorrectionsImport({ preventDefault() {} });
        report({ posted, toasts, errors, listReloaded: apiCalls.length });
    """)

    assert result["posted"] == [["/feedback/dataset", {"dataset_name": "k"}], ["/feedback/import", True]]
    assert 'corrections.dataset.saved {"name":"k.csv","rows":7}' in result["toasts"]
    assert 'corrections.import.done {"imported":4,"duplicates":1,"invalid":2}' in result["toasts"]
    assert result["listReloaded"] == 1, "the list shows what the import added"
    assert result["errors"] == []


@needs_node
def test_an_empty_name_or_no_file_is_refused_before_any_request(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        const posted = [];
        Api.post = async (...a) => { posted.push(a); };
        Api.postForm = async (...a) => { posted.push(a); };
        el("#corrections-dataset-name").value = "   ";
        await onCorrectionsDataset({ preventDefault() {} });
        el("#corrections-import-file").files = [];
        await onCorrectionsImport({ preventDefault() {} });
        report({ posted, errors });
    """)

    assert result["posted"] == []
    assert result["errors"] == ["corrections.dataset.nameMissing", "corrections.import.fileMissing"]
