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
// Spelled out rather than built from the mode name: a key that only exists as a
// template is invisible to the test that proves every string is translated.
const QUERY_SUBMIT_KEYS = {
  one: "query.submit.one", many: "query.submit.many", csv: "query.submit.csv",
};

function onQueryModeChange() {
  const mode = queryMode();
  $("#query-input-one").hidden = mode !== "one";
  $("#query-input-many").hidden = mode !== "many";
  $("#query-input-csv").hidden = mode !== "csv";
  // The two reliability signals are per-prediction extras of the JSON endpoints; the
  // CSV answer has no column for them, and offering a control that does nothing is a
  // small lie the rest of this UI does not tell.
  $("#query-signals").hidden = mode === "csv";
  const button = $("#query-btn");
  // The key travels with the element, so switching language re-reads the label of the
  // mode that is actually selected rather than resetting it to the first one.
  button.dataset.i18n = QUERY_SUBMIT_KEYS[mode];
  button.textContent = t(button.dataset.i18n);
  $("#query-results").innerHTML = "";
}

async function loadQueryTab() {
  const sel = $("#query-models");
  try {
    const names = await Api.get("/models");
    sel.innerHTML = names.map((n) => `<option>${esc(n)}</option>`).join("");
    if (!names.length) $("#query-results").innerHTML = `<p class="muted">${t("query.noModels")}</p>`;
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
  const hint = err.status === 422 ? ` ${t("query.staleBundleHint")}` : "";
  // role="alert": the results region is not a live region — announcing a whole table
  // would be noise — so a failure has to carry its own announcement.
  return `<p class="error" role="alert">${esc(err.message)}${hint}</p>`;
}

/* ---------- one text ---------- */

function predictionRow(p) {
  return `
      <div class="pred${p.above_threshold === false ? " below-t" : ""}">
        <span class="name">${esc(p.label)}</span>
        <span class="bar"><span data-width="${Math.round(p.confidence * 100)}"></span></span>
        <span class="val">${fmtFixed(p.confidence, 3)}${p.baseline_diff !== undefined
          ? ` <span class="muted">${t("query.diffTag")} ${p.baseline_diff >= 0 ? "+" : ""}${fmtFixed(p.baseline_diff, 3)}</span>` : ""}${
          // != null covers both: absent when not asked for, null when the bundle
          // carries no score for this label (older or partially scored models).
          p.label_f1 != null ? ` <span class="muted">${t("query.f1Tag")} ${fmtFixed(p.label_f1, 3)}</span>` : ""}${
          p.above_threshold === false ? ` <span class="muted">· ${t("query.belowThreshold")}</span>` : ""}</span>
      </div>`;
}

/* "Nothing above the threshold" is not "no idea": a model that declines still has a
   ranking, and seeing it is the difference between a decision and a dead end. */
function nearestHtml(preds) {
  if (!preds || !preds.length) return "";
  return `<p class="muted">${t("query.nearestBelow")}</p>${preds.map(predictionRow).join("")}`;
}

function renderPredictions(byModel, nearest = {}) {
  return Object.entries(byModel).map(([name, preds]) => `
    <div class="card">
      <div class="detail-head"><h3>${esc(name)}</h3>
        <button type="button" class="small ghost" data-explain="${esc(name)}">${t("query.explainButton")}</button>
        <button type="button" class="small ghost" data-correct="${esc(name)}">${t("query.correctButton")}</button></div>
      ${preds.length ? preds.map(predictionRow).join("")
      : `<p class="muted">${t("query.noLabelAboveThreshold")}</p>${nearestHtml(nearest[name])}`}</div>`).join("");
}

async function predictOneText(models, body) {
  if (models.length === 1) {
    const r = await Api.post("/predict", { ...body, model_name: models[0] });
    return { [models[0]]: r.results[0].predictions };
  }
  const r = await Api.post("/predict/multi", { ...body, model_names: models });
  return r.results[0].predictions_by_model;
}

async function runSingle(models, out) {
  const settings = querySettings();
  const body = { ...settings, texts: [$("#query-text").value] };
  const byModel = await predictOneText(models, body);
  // Ask the models that returned nothing what they almost said. Only when the caller
  // did not pin top_k (then every entry is already shown), and a failure here must not
  // cost the answer that did arrive.
  const silent = Object.entries(byModel).filter(([, preds]) => !preds.length).map(([n]) => n);
  let nearest = {};
  if (silent.length && settings.top_k === undefined) {
    try { nearest = await predictOneText(silent, { ...body, top_k: 3 }); }
    catch { nearest = {}; }
  }
  const text = $("#query-text").value;
  out.innerHTML = renderPredictions(byModel, nearest);
  applyBarWidths(out);
  // Only here: both act on ONE text — the endpoint explains one, and a correction
  // records one. The bulk modes have nothing to bind.
  bindExplainButtons(out, text);
  bindCorrectionButtons(out, byModel);
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
  // Composed from pluralised parts rather than one sentence carrying three counts:
  // "1 Text" and "2 Texte" differ, so each part has to pick its own form.
  const counted = [t("query.bulk.texts", { count: texts.length }),
                   t("query.bulk.labelsAssigned", { count: rows.filter((r) => r.uri).length })];
  if (refused) counted.push(`<strong>${t("query.bulk.withoutLabel", { count: refused })}</strong>`);
  out.innerHTML = `<div class="card">
    <h3>${counted.join(" · ")}</h3>
    <p class="muted">${t("query.bulk.note")}${
      rows.length > shown.length
        ? ` ${t("query.bulk.truncated", { shown: shown.length, total: rows.length })}`
        : ""}</p>
    <button type="button" class="small" id="bulk-download">${t("query.bulk.download")}</button>
    <div class="table-wrap"><table>
      <thead><tr><th class="num">${t("query.table.row")}</th><th>${t("query.table.text")}</th>
        <th>${t("query.table.label")}</th>
        <th class="num">${t("query.table.confidence")}</th></tr></thead>
      <tbody>${shown.map((r) => `<tr${r.uri ? "" : ' class="stale"'}>
        <td class="num">${r.row}</td><td>${esc(r.text)}</td>
        <td>${r.uri ? esc(r.label) : `<span class="muted">${t("query.table.noLabel")}</span>`}</td>
        <td class="num">${r.uri ? fmtFixed(r.confidence, 3) : "–"}</td></tr>`).join("")}</tbody>
    </table></div></div>`;
  out.querySelector("#bulk-download").addEventListener("click", () => {
    Api.saveBlob(new Blob([bulkCsv(rows)], { type: "text/csv" }), "predictions.csv");
  });
  return t("query.bulk.done", {
    texts: t("query.bulk.texts", { count: texts.length }),
    refused: t("query.bulk.withoutLabel", { count: refused }),
  });
}

async function runManyTexts(model, out) {
  const texts = $("#query-lines").value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!texts.length) {
    out.innerHTML = `<p class="error" role="alert">${t("query.error.noTexts")}</p>`;
    return "";
  }
  if (texts.length > QUERY_MAX_LINES) {
    out.innerHTML =
      `<p class="error" role="alert">${t("query.error.tooManyLines", { count: texts.length })}</p>`;
    return "";
  }
  const settings = querySettings();
  const rows = [];
  for (let start = 0; start < texts.length; start += QUERY_BATCH) {
    const slice = texts.slice(start, start + QUERY_BATCH);
    $("#query-status").textContent = t("query.progress", {
      done: Math.min(start + slice.length, texts.length), total: texts.length,
    });
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
  const counted = [t("query.csv.inputRows", { count: covered.size }),
                   t("query.bulk.labelsAssigned", { count: lines.length - refused })];
  if (refused) counted.push(`<strong>${t("query.csv.rowsWithoutLabel", { count: refused })}</strong>`);
  return `<div class="card">
    <h3>${t("query.csv.heading", { name: esc(filename) })}</h3>
    <p>${counted.join(" · ")}.</p>
    <p class="muted">${t("query.csv.note")}</p></div>`;
}

async function runCsvFile(model, out) {
  const input = $("#query-file");
  if (!input.files.length) {
    out.innerHTML = `<p class="error" role="alert">${t("common.error.noCsvFile")}</p>`;
    return "";
  }
  const form = new FormData();
  form.append("file", input.files[0]);
  form.append("model_name", model);
  form.append("separator", $("#query-separator").value || ";");
  const topk = $("#query-topk").value;
  if (topk !== "") form.append("top_k", topk);

  const name = input.files[0].name.replace(/\.csv$/i, "") + "-predictions.csv";
  $("#query-status").textContent = t("query.csv.progress");
  const blob = await Api.downloadForm("/predict/csv", form, name);
  const text = await blob.text();
  out.innerHTML = csvSummary(text, name);
  return t("query.csv.done", { name });   // textContent: the caller does not re-escape
}

/* ---------- submit ---------- */

async function onQuery(ev) {
  ev.preventDefault();
  const btn = $("#query-btn"), out = $("#query-results"), mode = queryMode();
  const models = selectedModels();
  if (!models.length) {
    out.innerHTML = `<p class="error" role="alert">${t("query.error.noModel")}</p>`;
    return;
  }
  if (mode !== "one" && models.length > 1) {
    out.innerHTML = `<p class="error" role="alert">${t("query.error.oneModelOnly")}</p>`;
    return;
  }
  btn.disabled = true;
  $("#query-status").textContent = t("query.progressStart");
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
