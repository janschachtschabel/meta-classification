/* The Query tab: classify one text, a list of texts, or a whole CSV file.

   Split from app.js when the bulk modes landed — the shell (login, tabs, toasts) and
   the training view stayed there. The three modes share a form on purpose: picking a
   model, a top-k and reading the answer is the same job whether it is one text or five
   hundred; only where the texts come from differs. */
"use strict";

// /predict takes up to 1000 texts. Half that: the wait between progress updates stays
// short on a long paste, and two round trips cost far less than one stalled minute.
const QUERY_BATCH = 500;
// Above this a paste is really a file, and the CSV mode streams instead of holding
// every answer in the page. Say so rather than melting the browser.
const QUERY_MAX_LINES = 5000;
const QUERY_TABLE_LIMIT = 200;

const queryMode = () => document.querySelector('input[name="query-mode"]:checked').value;
// English needs one rule and this UI has no i18n layer (every string in it is literal
// English, a decision that predates this view). "1 rows" is still wrong.
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function onQueryModeChange() {
  const mode = queryMode();
  $("#query-input-one").hidden = mode !== "one";
  $("#query-input-many").hidden = mode !== "many";
  $("#query-input-csv").hidden = mode !== "csv";
  // The two reliability signals are per-prediction extras of the JSON endpoints; the
  // CSV answer has no column for them, and offering a control that does nothing is a
  // small lie the rest of this UI does not tell.
  $("#query-signals").hidden = mode === "csv";
  $("#query-btn").textContent =
    { one: "Classify", many: "Classify all", csv: "Classify the file" }[mode];
  $("#query-results").innerHTML = "";
}

async function loadQueryTab() {
  const sel = $("#query-models");
  try {
    const names = await Api.get("/models");
    sel.innerHTML = names.map((n) => `<option>${esc(n)}</option>`).join("");
    if (!names.length) $("#query-results").innerHTML =
      `<p class="muted">No models yet — train one on the Training tab first.</p>`;
    else sel.options[0].selected = true;
  } catch (err) { $("#query-results").innerHTML = `<p class="error">${esc(err.message)}</p>`; }
}

/* ---------- shared request pieces ---------- */

function querySettings() {
  const body = {
    include_baseline_diff: $("#query-diff").checked,
    include_label_f1: $("#query-f1").checked,
  };
  const topk = $("#query-topk").value;
  if (topk !== "") body.top_k = Number(topk);
  return body;
}

function selectedModels() {
  return [...$("#query-models").selectedOptions].map((o) => o.value);
}

function queryErrorHtml(err) {
  // 422 here means the bundle exists but cannot be loaded — in practice a
  // pre-format-2 model. Say what to do about it; the raw message does not.
  const hint = err.status === 422
    ? ` This model was trained with an older bundle format and has to be retrained
        (the Models tab marks it).` : "";
  // role="alert": the results region is not a live region — announcing a whole table
  // would be noise — so a failure has to carry its own announcement.
  return `<p class="error" role="alert">${esc(err.message)}${esc(hint)}</p>`;
}

/* ---------- one text ---------- */

function renderPredictions(byModel) {
  return Object.entries(byModel).map(([name, preds]) => `
    <div class="card"><h3>${esc(name)}</h3>${preds.length ? preds.map((p) => `
      <div class="pred${p.above_threshold === false ? " below-t" : ""}">
        <span class="name">${esc(p.label)}</span>
        <span class="bar"><span data-width="${Math.round(p.confidence * 100)}"></span></span>
        <span class="val">${p.confidence.toFixed(3)}${p.baseline_diff !== undefined
          ? ` <span class="muted">diff ${p.baseline_diff >= 0 ? "+" : ""}${p.baseline_diff.toFixed(3)}</span>` : ""}${
          // != null covers both: absent when not asked for, null when the bundle
          // carries no score for this label (older or partially scored models).
          p.label_f1 != null ? ` <span class="muted">F1 ${p.label_f1.toFixed(3)}</span>` : ""}${
          p.above_threshold === false ? ` <span class="muted">· below threshold</span>` : ""}</span>
      </div>`).join("")
      : `<p class="muted">No label above the model's threshold.</p>`}</div>`).join("");
}

async function runSingle(models, out) {
  const body = { ...querySettings(), texts: [$("#query-text").value] };
  let byModel;
  if (models.length === 1) {
    const r = await Api.post("/predict", { ...body, model_name: models[0] });
    byModel = { [models[0]]: r.results[0].predictions };
  } else {
    const r = await Api.post("/predict/multi", { ...body, model_names: models });
    byModel = r.results[0].predictions_by_model;
  }
  out.innerHTML = renderPredictions(byModel);
  applyBarWidths(out);
}

/* ---------- many texts ---------- */

const csvCell = (value) => `"${String(value).replace(/"/g, '""')}"`;

function bulkCsv(rows) {
  return ["row,text,uri,label,confidence"]
    .concat(rows.map((r) => [r.row, csvCell(r.text), csvCell(r.uri), csvCell(r.label),
                             r.confidence].join(",")))
    .join("\n");
}

function renderBulkTable(rows, texts, out) {
  const shown = rows.slice(0, QUERY_TABLE_LIMIT);
  const refused = new Set(rows.filter((r) => !r.uri).map((r) => r.row)).size;
  out.innerHTML = `<div class="card">
    <h3>${plural(texts.length, "text")} · ${plural(rows.filter((r) => r.uri).length, "label")}
        assigned${refused ? ` · <strong>${plural(refused, "text")} without a label</strong>` : ""}</h3>
    <p class="muted">A text the model asserts nothing for is listed with an empty label —
       "which ones did it refuse" is part of the answer.${
      rows.length > shown.length
        ? ` Showing the first ${shown.length} of ${rows.length} lines; the download has all.`
        : ""}</p>
    <button type="button" class="small" id="bulk-download">Download as CSV</button>
    <div class="table-wrap"><table>
      <thead><tr><th class="num">Row</th><th>Text</th><th>Label</th>
        <th class="num">Confidence</th></tr></thead>
      <tbody>${shown.map((r) => `<tr${r.uri ? "" : ' class="stale"'}>
        <td class="num">${r.row}</td><td>${esc(r.text)}</td>
        <td>${r.uri ? esc(r.label) : '<span class="muted">no label</span>'}</td>
        <td class="num">${r.uri ? r.confidence.toFixed(3) : "–"}</td></tr>`).join("")}</tbody>
    </table></div></div>`;
  out.querySelector("#bulk-download").addEventListener("click", () => {
    Api.saveBlob(new Blob([bulkCsv(rows)], { type: "text/csv" }), "predictions.csv");
  });
  return `Done: ${plural(texts.length, "text")}, ${plural(refused, "text")} without a label.`;
}

async function runManyTexts(model, out) {
  const texts = $("#query-lines").value.split("\n").map((t) => t.trim()).filter(Boolean);
  if (!texts.length) {
    out.innerHTML = `<p class="error" role="alert">Enter at least one text.</p>`;
    return "";
  }
  if (texts.length > QUERY_MAX_LINES) {
    out.innerHTML = `<p class="error" role="alert">${texts.length} lines is more than this
      view holds. Put them in a CSV and use the file mode — it streams instead of
      collecting every answer in the page.</p>`;
    return "";
  }
  const settings = querySettings();
  const rows = [];
  for (let start = 0; start < texts.length; start += QUERY_BATCH) {
    const slice = texts.slice(start, start + QUERY_BATCH);
    $("#query-status").textContent =
      `Classifying ${Math.min(start + slice.length, texts.length)} of ${texts.length} …`;
    const answer = await Api.post("/predict", { ...settings, model_name: model, texts: slice });
    answer.results.forEach((result, index) => {
      const row = start + index;
      if (!result.predictions.length) rows.push({ row, text: slice[index], uri: "", label: "", confidence: 0 });
      result.predictions.forEach((p) => rows.push({ row, text: slice[index], ...p }));
    });
  }
  return renderBulkTable(rows, texts, out);
}

/* ---------- a CSV file ---------- */

function csvSummary(text, filename) {
  const lines = text.split("\n").slice(1).filter(Boolean);
  // Only the row number is read out of the raw line, and that field is always a bare
  // integer — quoting can affect the label and nothing before it. The file itself is
  // what the user works with; this is the receipt.
  const covered = new Set(lines.map((line) => line.slice(0, line.indexOf(","))));
  const refused = lines.filter((line) => /^\d+,,,,\r?$/.test(line)).length;
  return `<div class="card">
    <h3>${esc(filename)} downloaded</h3>
    <p>${plural(covered.size, "input row")} · ${plural(lines.length - refused, "label")} assigned${
      refused ? ` · <strong>${plural(refused, "row")} without a label</strong>` : ""}.</p>
    <p class="muted">Each line carries the number of the input row it came from, so the
       answers join back onto your own file. Rows the model asserted nothing for are in
       there too, with the prediction fields empty.</p></div>`;
}

async function runCsvFile(model, out) {
  const input = $("#query-file");
  if (!input.files.length) {
    out.innerHTML = `<p class="error" role="alert">Choose a CSV file first.</p>`;
    return "";
  }
  const form = new FormData();
  form.append("file", input.files[0]);
  form.append("model_name", model);
  form.append("separator", $("#query-separator").value || ";");
  const topk = $("#query-topk").value;
  if (topk !== "") form.append("top_k", topk);

  const name = input.files[0].name.replace(/\.csv$/i, "") + "-predictions.csv";
  $("#query-status").textContent = "Classifying the file — the answer downloads when it is done …";
  const blob = await Api.downloadForm("/predict/csv", form, name);
  const text = await blob.text();
  out.innerHTML = csvSummary(text, name);
  return `Done: ${name} downloaded.`;
}

/* ---------- submit ---------- */

async function onQuery(ev) {
  ev.preventDefault();
  const btn = $("#query-btn"), out = $("#query-results"), mode = queryMode();
  const models = selectedModels();
  if (!models.length) {
    out.innerHTML = `<p class="error" role="alert">Select at least one model.</p>`;
    return;
  }
  if (mode !== "one" && models.length > 1) {
    out.innerHTML = `<p class="error" role="alert">Bulk classification runs one model at a
      time — select a single model.</p>`;
    return;
  }
  btn.disabled = true;
  $("#query-status").textContent = "Classifying …";
  try {
    let done = "";
    if (mode === "one") await runSingle(models, out);
    else if (mode === "many") done = await runManyTexts(models[0], out);
    else done = await runCsvFile(models[0], out);
    // The status region is the page's only live region, so it carries the outcome —
    // clearing it unconditionally would leave a screen reader with no completion at all.
    $("#query-status").textContent = done;
  } catch (err) {
    out.innerHTML = queryErrorHtml(err);
    $("#query-status").textContent = "";
  } finally { btn.disabled = false; }
}
