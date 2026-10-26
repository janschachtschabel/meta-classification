"""The admin UI's side of model names: the display name shown, the rename with its warning.

Same node harness as test_ui_handlers.py; the API is answered by the test.
"""

from test_ui_handlers import _PRELUDE, _run, needs_node

# The real escape and a t() that shows its parameters (the shared prelude's are `const`
# pass-throughs a globalThis stub cannot override).
_PRELUDE_HERE = (_PRELUDE
                 .replace("const t = (key) => key;",
                          "const t = (key, params) => params ? `${key} ${JSON.stringify(params)}` : key;")
                 .replace("const esc = (s) => String(s);", ""))

_STUBS = """
globalThis.fmtFixed = (v) => String(v);
globalThis.renderShareLinks = () => {};
globalThis.toasts = [];
globalThis.toast = (msg) => toasts.push(msg);
globalThis.toastError = (err) => errors.push(String(err.message || err));
globalThis.busy = () => () => {};
globalThis.closer = (close) => close;
const MODEL = { name: "fach", task_type: "multilabel", format_version: 2,
  metadata: { n_labels: 3, metrics: {},
              info: { display_name: "Fächer <Schule>", author: "Redaktion" } } };
function answer(routes) {
  Api.get = async (url) => {
    apiCalls.push(url);
    if (!(url in routes)) throw new Error(`unexpected ${url}`);
    return routes[url];
  };
}
"""
_MODULES = ("escape.js", "models.js")


@needs_node
def test_the_models_table_shows_the_display_name_and_offers_a_rename(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answer({ "/models": ["fach"], "/models/fach": MODEL });
        await loadModels();
        report({ html: el("#models-list").innerHTML, errors });
    """)

    html = result["html"]
    assert "Fächer &lt;Schule&gt;" in html, "the display name, escaped"
    assert ">fach<" in html, "and the technical name beside it"
    assert 'data-rename="fach"' in html
    assert result["errors"] == []


@needs_node
def test_renaming_warns_about_the_links_it_will_revoke_and_renames(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answer({ "/share": [{ kind: "model", name: "fach" }, { kind: "model", name: "fach" },
                            { kind: "dataset", name: "fach" }, { kind: "model", name: "other" }],
                 "/models": [], });
        await renameModel("fach");
        const form = el("#models-share").innerHTML;
        const posted = [];
        Api.post = async (url, json) => { posted.push([url, json]); return { name: "fach_neu", revoked_links: 2 }; };
        el("#models-share #rename-new").value = " fach_neu ";
        await submitRename({ preventDefault() {} }, "fach", el("#models-share"));
        report({ form, posted, toasts, errors, reloaded: apiCalls.includes("/models") });
    """)

    assert 'models.rename.linksWarning {"count":2}' in result["form"]
    assert 'value="fach"' in result["form"], "the field starts from the current name"
    assert result["posted"] == [["/models/fach/rename", {"new_name": "fach_neu"}]]
    assert 'models.rename.done {"name":"fach_neu","count":2}' in result["toasts"], "count: a plural entry"
    assert result["reloaded"], "the table shows the new name"
    assert result["errors"] == []


@needs_node
def test_a_rename_without_links_or_without_admin_rights_says_so(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answer({ "/share": [] });
        await renameModel("fach");
        const none = el("#models-share").innerHTML;
        Api.get = async () => { const err = new Error("Forbidden"); err.status = 403; throw err; };
        await renameModel("fach");
        report({ none, unknown: el("#models-share").innerHTML });
    """)

    assert "models.rename.noLinks" in result["none"]
    assert "models.rename.linksUnknown" in result["unknown"], "the warning stays when links cannot be counted"


@needs_node
def test_an_unchanged_or_empty_name_is_refused_before_any_request(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        const posted = [];
        Api.post = async (...a) => { posted.push(a); };
        for (const value of ["fach", "   "]) {
          el("#models-share #rename-new").value = value;
          await submitRename({ preventDefault() {} }, "fach", el("#models-share"));
        }
        report({ posted, errors });
    """)

    assert result["posted"] == []
    assert result["errors"] == ["models.rename.unchanged", "models.rename.empty"]


@needs_node
def test_the_documentation_form_carries_the_display_name(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answer({ "/models/fach": MODEL });
        await editModelInfo("fach");
        report({ html: el("#models-share").innerHTML });
    """)

    assert 'id="info-display_name"' in result["html"]
    assert 'value="Fächer &lt;Schule&gt;"' in result["html"]


@needs_node
def test_saving_the_documentation_shows_the_new_display_name_in_the_table(tmp_path):
    result = _run(tmp_path, modules=_MODULES, prelude=_PRELUDE_HERE, body=_STUBS + """
        answer({ "/models/fach": MODEL, "/models": ["fach"] });
        await editModelInfo("fach");
        const box = el("#models-share");
        el("#models-share #info-display_name").value = " Fächer ";
        const put = [];
        Api.put = async (url, json) => { put.push([url, json]); return {}; };
        await submitInfo({ preventDefault() {} }, "fach", box);
        report({ put, toasts, errors, reloaded: apiCalls.includes("/models") });
    """)

    assert result["put"][0][0] == "/models/fach/info"
    assert result["put"][0][1]["display_name"] == "Fächer"
    assert "models.info.saved" in result["toasts"]
    assert result["reloaded"], "the table reads the display name from the saved documentation"
    assert result["errors"] == []


@needs_node
def test_the_detail_shows_the_documentation(tmp_path):
    result = _run(tmp_path, prelude=_PRELUDE_HERE, modules=("escape.js",),
                  helpers=("model-detail.js:definitionList", "model-detail.js:documentationSection"),
                  body="""
        const documented = documentationSection({ metadata: { info: {
          display_name: "Fächer", description: "Für <Schulmaterial>", license: "CC BY 4.0" } } });
        const bare = documentationSection({ metadata: {} });
        report({ documented, bare });
    """)

    assert "modelDetail.row.displayName" in result["documented"]
    assert "Für &lt;Schulmaterial&gt;" in result["documented"]
    assert "CC BY 4.0" in result["documented"]
    assert "modelDetail.row.author" not in result["documented"], "an unstated field leaves no row"
    assert "modelDetail.noDocumentation" in result["bare"]


@needs_node
def test_the_query_tab_labels_a_model_by_its_display_name(tmp_path):
    result = _run(tmp_path, prelude=_PRELUDE_HERE, modules=("escape.js", "models.js", "query.js"),
                  body=_STUBS + """
        answer({ "/models": ["fach", "stufe"], "/models/fach": MODEL,
                 "/models/stufe": { name: "stufe", metadata: {} } });
        await loadQueryTab();
        report({ html: el("#query-models").innerHTML, errors });
    """)

    html = result["html"]
    assert 'value="fach"' in html and "Fächer &lt;Schule&gt;" in html
    assert 'value="stufe"' in html
    assert result["errors"] == []


@needs_node
def test_a_blank_or_padded_display_name_from_a_bundle_is_trimmed(tmp_path):
    # Bundle data is foreign: an imported archive can carry a display name the API would refuse.
    result = _run(tmp_path, prelude=_PRELUDE_HERE, helpers=("models.js:modelDisplayName",), body="""
        const named = (value) => modelDisplayName({ metadata: { info: { display_name: value } } });
        report({ blank: named("   "), padded: named("  Fächer "), number: named(7) });
    """)

    assert result == {"blank": "", "padded": "Fächer", "number": ""}
