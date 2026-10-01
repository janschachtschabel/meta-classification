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
  // Same reason: /predict/csv streams its answer straight through, so this view never
  // holds the rows that metadata would be attached to.
  $("#query-metadata-option").hidden = mode === "csv";
  const button = $("#query-btn");
  // The key travels with the element, so switching language re-reads the label of the
  // mode that is actually selected rather than resetting it to the first one.
  button.dataset.i18n = QUERY_SUBMIT_KEYS[mode];
  button.textContent = t(button.dataset.i18n);
  $("#query-results").innerHTML = "";
}

/* Checkboxes, not `<select multiple size="4">`: picking a second model there needs
   Ctrl-click, which is undiscoverable with a mouse and, in several browsers, not available
   from the keyboard at all — so the app's one multi-value control was the one control a
   keyboard user could not fully operate. Carried over from the September audit as F4. A
   checkbox group needs no instructions and is a native part of the form. */
async function loadQueryTab() {
  const box = $("#query-models");
  try {
    const names = await Api.get("/models");
    // The first is pre-checked, as the <select>'s first option was: the common case is one
    // model, and an empty picker would make the primary screen look broken.
    box.innerHTML = names.map((n, at) => `
      <label class="check"><input type="checkbox" name="query-model" value="${esc(n)}"${
        at === 0 ? " checked" : ""}> <span>${esc(n)}</span></label>`).join("");
    if (!names.length) $("#query-results").innerHTML = `<p class="muted">${t("query.noModels")}</p>`;
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
  return [...document.querySelectorAll('input[name="query-model"]:checked')].map((b) => b.value);
}

const wantsMetadata = () => $("#query-metadata").checked;

/* What a submit is asking for: a classification, descriptive metadata, or both.

   A value rather than two early returns inside the submit handler, because the
   model-free case had no way in: /metadata reads no model — as the option's own hint
   says — so asking for it alone is legitimate, and it used to be refused. The csv
   exception is not arbitrary either. /predict/csv streams its answer straight through,
   so the option is hidden in that mode, and a box ticked before the switch stays
   ticked — honouring it there would start a CSV run with no model to run it. */
function queryPlan(models, metadata, mode) {
  const described = metadata && mode !== "csv";
  if (!models.length && !described) {
    // Not one message for both: in csv mode the metadata option is hidden, so naming
    // it would send the reader looking for a control that is not on screen.
    return { error: mode === "csv" ? "query.error.noModel" : "query.error.nothingRequested" };
  }
  return {
    classify: models.length > 0,
    metadata: described,
    // Carried here because #query-status is the page's ONLY live region: announcing a
    // classification while nothing is classified is what a screen reader would hear.
    progress: models.length ? "query.progressStart" : "query.progressDescribing",
  };
}

/* Descriptive metadata for the same texts, from /metadata — a separate endpoint because it
   reads no model: the proposals come out of the text itself, so they are the same whichever
   model classified it, and asking for them per model would repeat identical work. */
async function fetchMetadata(texts) {
  const answer = await Api.post("/metadata", { texts });
  return answer.results;
}

// /metadata takes 100 texts where /predict takes 1000, because generating costs ~20 ms a text
// against well under a millisecond for a classification. One prediction batch is therefore
// several metadata calls; the rows come back in order, so they concatenate.
const METADATA_BATCH = 100;

async function metadataForBatch(texts) {
  const described = [];
  for (let start = 0; start < texts.length; start += METADATA_BATCH) {
    const answers = await fetchMetadata(texts.slice(start, start + METADATA_BATCH));
    // Only the three fields: `text` is the endpoint's truncated echo, and spreading it over
    // a row would replace the full text the table and the download show.
    answers.forEach(({ title, description, keywords }) =>
      described.push({ title, description, keywords }));
  }
  return described;
}

/* ---------- descriptive metadata ---------- */

function metadataField(key, value) {
  // An empty field is the endpoint's honest answer for a thin text, and it has to read as
  // one: a blank line here looks like the card failed to render.
  const body = value
    ? `<p>${esc(value)}</p>`
    : `<p class="muted">${t("query.metadata.none")}</p>`;
  return `<h4>${t(key)}</h4>${body}`;
}

function metadataCard(proposal) {
  const keywords = proposal.keywords.length
    ? `<p class="keywordlist">${proposal.keywords
        .map((k) => `<span class="pill">${esc(k)}</span>`).join("")}</p>`
    : `<p class="muted">${t("query.metadata.none")}</p>`;
  return `
    <div class="card">
      <h3>${t("query.metadata.heading")}</h3>
      <p class="muted">${t("query.metadata.note")}</p>
      ${metadataField("query.metadata.title", proposal.title)}
      ${metadataField("query.metadata.description", proposal.description)}
      <h4>${t("query.metadata.keywords")}</h4>${keywords}
    </div>`;
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
      : `<p class="muted">${t("query.noLabelAboveThreshold")}</p>${
        // hasOwn, not nearest[name]: a model MAY be called "constructor" or "toString"
        // (safe_name rejects path characters, not property names), and inherited
        // members are not predictions — rendering one throws and the thrown card
        // replaces the whole answer, including the models that did return.
        nearestHtml(Object.hasOwn(nearest, name) ? nearest[name] : null)}`}</div>`).join("");
}

async function predictOneText(models, body) {
  if (models.length === 1) {
    const r = await Api.post("/predict", { ...body, model_name: models[0] });
    return { [models[0]]: r.results[0].predictions };
  }
  const r = await Api.post("/predict/multi", { ...body, model_names: models });
  return r.results[0].predictions_by_model;
}

async function runSingle(models, plan, out) {
  const settings = querySettings();
  // Read ONCE: every part of the answer is about this text. Reading the field again after
  // the awaits gave a user who typed on the metadata of one text beside the classification
  // of another, and a "Why?" for a text the model never saw (audit 2026-09-30, U06).
  const text = $("#query-text").value;
  const body = { ...settings, texts: [text] };
  // Nothing asked to classify: the metadata is the whole answer, and /predict/multi
  // with an empty list would be a request for nothing.
  const byModel = plan.classify ? await predictOneText(models, body) : {};
  // Ask the models that returned nothing what they almost said. Only when the caller
  // did not pin top_k (then every entry is already shown), and a failure here must not
  // cost the answer that did arrive.
  const silent = Object.entries(byModel).filter(([, preds]) => !preds.length).map(([n]) => n);
  let nearest = {};
  if (silent.length && settings.top_k === undefined) {
    try { nearest = await predictOneText(silent, { ...body, top_k: 3 }); }
    // Swallowed on purpose — the answer that DID arrive must still render — but not
    // silently: without this line a rate limit, a server error and a bug in here all
    // look like "the model had nothing close".
    catch (err) { nearest = {}; console.warn("near-miss follow-up failed", err); }
  }
  // Requested before the answer is written so a failure here is a failure of the whole
  // submit: the user ticked a box, and silently leaving the card out would look like the
  // text simply yielded nothing.
  const described = plan.metadata ? (await fetchMetadata([text]))[0] : null;
  out.innerHTML = renderPredictions(byModel, nearest)
    + (described ? metadataCard(described) : "");
  applyBarWidths(out);
  // Only here: both act on ONE text — the endpoint explains one, and a correction
  // records one. The bulk modes have nothing to bind.
  bindExplainButtons(out, text);
  bindCorrectionButtons(out, byModel);
}

/* ---------- many texts ---------- */

/* A cell whose text begins with =, +, - or @ is a live FORMULA when the download is
   opened in Excel, Calc or Sheets — quoting does not help, because the parser consumes the
   quotes and evaluates what is inside. Labels here are WLO URIs and text a caller supplied,
   so this is untrusted content on its way into a spreadsheet. A leading tab is the
   conventional neutraliser: it keeps the value readable, keeps it a string, and costs
   nothing in the readers that never evaluated formulas anyway. */
const FORMULA_LEAD = /^[=+\-@\t\r]/;
const csvCell = (value) => {
  const text = String(value);
  return `"${(FORMULA_LEAD.test(text) ? `\t${text}` : text).replace(/"/g, '""')}"`;
};

/* One source for "these rows carry metadata": the table and the download have to agree, and
   two copies of the rule are two chances to drift into a table with columns the file lacks.
   Keyed on `title` being present at all rather than truthy — the endpoint answers a thin text
   with an empty title, and that is a result to show, not a reason to drop the columns. */
const rowsAreDescribed = (rows) => rows.some((r) => r.title !== undefined);

/* The metadata columns are APPENDED, and only when they were asked for: a consumer already
   parsing this download by position keeps working either way. Metadata belongs to the input
   row, so it repeats across that row's labels — normal for a flat export, and what lets the
   file be joined back onto the caller's own rows. */
function bulkCsv(rows, classified = true) {
  const described = rowsAreDescribed(rows);
  // The columns follow the table's. A file that keeps an empty label and a
  // confidence of 0 for a run with no model states a measurement that was never
  // taken — and 0 reads as "the model was unsure" to whatever opens the file next.
  const head = "row,text" + (classified ? ",uri,label,confidence" : "")
    + (described ? ",title,description,keywords" : "");
  return [head]
    .concat(rows.map((r) => {
      const cells = [r.row, csvCell(r.text)];
      if (classified) cells.push(csvCell(r.uri), csvCell(r.label), r.confidence);
      if (described) {
        cells.push(csvCell(r.title ?? ""), csvCell(r.description ?? ""),
                   // "; " as the API's own label lists are separated, so one cell stays one cell.
                   csvCell((r.keywords ?? []).join("; ")));
      }
      return cells.join(",");
    }))
    .join("\n");
}

function renderBulkTable(rows, texts, out, classified = true) {
  const shown = rows.slice(0, QUERY_TABLE_LIMIT);
  // Nothing classified means nothing was refused: counting every row as "without a
  // label" would report a question nobody asked as that many failures.
  const refused = classified
    ? new Set(rows.filter((r) => !r.uri).map((r) => r.row)).size : 0;
  // The title and the keywords are short enough to read in a cell; a 500-character
  // description is not, so it stays in the download and the note says so.
  const described = rowsAreDescribed(rows);
  // Composed from pluralised parts rather than one sentence carrying three counts:
  // "1 Text" and "2 Texte" differ, so each part has to pick its own form.
  const counted = [t("query.bulk.texts", { count: texts.length })];
  if (classified) {
    counted.push(t("query.bulk.labelsAssigned", { count: rows.filter((r) => r.uri).length }));
    if (refused) counted.push(`<strong>${t("query.bulk.withoutLabel", { count: refused })}</strong>`);
  }
  out.innerHTML = `<div class="card">
    <h3>${counted.join(" · ")}</h3>
    <p class="muted">${t("query.bulk.note")}${
      rows.length > shown.length
        ? ` ${t("query.bulk.truncated", { shown: shown.length, total: rows.length })}`
        : ""}${described ? ` ${t("query.bulk.descriptionInDownload")}` : ""}</p>
    <button type="button" class="small" id="bulk-download">${t("query.bulk.download")}</button>
    <div class="table-wrap" tabindex="0"><table>
      <thead><tr><th class="num">${t("query.table.row")}</th><th>${t("query.table.text")}</th>${classified
        ? `<th>${t("query.table.label")}</th>
        <th class="num">${t("query.table.confidence")}</th>` : ""}${described
          ? `<th>${t("query.table.title")}</th><th>${t("query.table.keywords")}</th>` : ""}</tr></thead>
      <tbody>${shown.map((r) => `<tr${classified && !r.uri ? ' class="stale"' : ""}>
        <td class="num">${r.row}</td><td>${esc(r.text)}</td>${classified
          ? `<td>${r.uri ? esc(r.label) : `<span class="muted">${t("query.table.noLabel")}</span>`}</td>
        <td class="num">${r.uri ? fmtFixed(r.confidence, 3) : "–"}</td>` : ""}${described
          ? `<td>${esc(r.title ?? "")}</td><td>${esc((r.keywords ?? []).join(", "))}</td>` : ""
        }</tr>`).join("")}</tbody>
    </table></div></div>`;
  out.querySelector("#bulk-download").addEventListener("click", () => {
    Api.saveBlob(new Blob([bulkCsv(rows, classified)], { type: "text/csv" }), "predictions.csv");
  });
  const texted = t("query.bulk.texts", { count: texts.length });
  return classified
    ? t("query.bulk.done", { texts: texted,
                             refused: t("query.bulk.withoutLabel", { count: refused }) })
    : t("query.bulk.doneDescribed", { texts: texted });
}

async function runManyTexts(models, plan, out) {
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
    const answer = plan.classify
      ? await Api.post("/predict", { ...settings, model_name: models[0], texts: slice })
      // No model: one empty prediction per text, so the rows below still carry the
      // text and whatever the metadata call returned for it.
      : { results: slice.map(() => ({ predictions: [] })) };
    // Per batch, and only on request: /metadata caps a batch at 100 against /predict's 1000,
    // so a full slice is sent in several calls.
    const described = plan.metadata ? await metadataForBatch(slice) : null;
    answer.results.forEach((result, index) => {
      const row = start + index;
      const extra = described ? described[index] : {};
      const base = { row, text: slice[index], ...extra };
      if (!result.predictions.length) rows.push({ ...base, uri: "", label: "", confidence: 0 });
      result.predictions.forEach((p) => rows.push({ ...base, ...p }));
    });
  }
  return renderBulkTable(rows, texts, out, plan.classify);
}

/* ---------- a CSV file ---------- */

const answerLines = (text) => text.split("\n").slice(1).filter(Boolean);

/* The input rows an answer covers. Every input row gets at least one line -- a refused
   one with empty fields -- so the distinct row numbers ARE the rows that made it into the
   file. Only the row number is read out of the raw line, and that field is always a bare
   integer: quoting can affect the label and nothing before it. */
const rowsCovered = (lines) => new Set(lines.map((line) => line.slice(0, line.indexOf(",")))).size;

/* `expected` is the server's X-Input-Rows, null where it sent none. A stream cut short ends
   like a finished one, so fewer rows than that is said where the success would have been
   (audit 2026-09-30, U02: "Fertig … 500 Eingabezeilen" for a 700-row file). */
function csvSummary(text, filename, expected = null) {
  const lines = answerLines(text);
  // The file itself is what the user works with; this is the receipt.
  const covered = rowsCovered(lines);
  const refused = lines.filter((line) => /^\d+,,,,\r?$/.test(line)).length;
  const counted = [t("query.csv.inputRows", { count: covered }),
                   t("query.bulk.labelsAssigned", { count: lines.length - refused })];
  if (refused) counted.push(`<strong>${t("query.csv.rowsWithoutLabel", { count: refused })}</strong>`);
  const cutShort = expected !== null && covered < expected
    ? `<p class="error" role="alert">${t("query.csv.incomplete", { covered, expected })}</p>` : "";
  return `<div class="card">
    <h3>${t("query.csv.heading", { name: esc(filename) })}</h3>${cutShort}
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
  const { blob, headers } = await Api.downloadForm("/predict/csv", form, name);
  const text = await blob.text();
  const sent = Number.parseInt(headers.get("X-Input-Rows") ?? "", 10);
  const expected = Number.isInteger(sent) ? sent : null;
  out.innerHTML = csvSummary(text, name, expected);
  const complete = expected === null || rowsCovered(answerLines(text)) >= expected;
  // textContent: the caller does not re-escape
  return t(complete ? "query.csv.done" : "query.csv.cutShort", { name });
}

/* ---------- submit ---------- */

async function onQuery(ev) {
  ev.preventDefault();
  const btn = $("#query-btn"), out = $("#query-results"), mode = queryMode();
  const models = selectedModels();
  const plan = queryPlan(models, wantsMetadata(), mode);
  if (plan.error) {
    out.innerHTML = `<p class="error" role="alert">${t(plan.error)}</p>`;
    return;
  }
  if (mode !== "one" && models.length > 1) {
    out.innerHTML = `<p class="error" role="alert">${t("query.error.oneModelOnly")}</p>`;
    return;
  }
  btn.disabled = true;
  $("#query-status").textContent = t(plan.progress);
  try {
    let done = "";
    if (mode === "one") await runSingle(models, plan, out);
    else if (mode === "many") done = await runManyTexts(models, plan, out);
    else done = await runCsvFile(models[0], out);
    // The status region is the page's only live region, so it carries the outcome —
    // clearing it unconditionally would leave a screen reader with no completion at all.
    $("#query-status").textContent = done;
  } catch (err) {
    out.innerHTML = queryErrorHtml(err);
    $("#query-status").textContent = "";
  } finally { btn.disabled = false; }
}
