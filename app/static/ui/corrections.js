/* The Corrections tab: what editors corrected in the query tab, seen and put to use.

   Corrections were recorded (feedback.js) and nothing in the UI read them back: no view, no
   export, no way to train on them without the API. Here they are listed newest first, per
   model; exported as the training CSV or as a complete backup; a backup is imported back; and
   the training CSV is saved as a dataset the Training tab can use (GET/POST /feedback...).
   All of it is admin: the list shows every text an editor ever pasted. */
"use strict";

const CORRECTIONS_PAGE = 50;
let correctionsShown = 0;

function correctionLabels(labels) {
  return labels.length ? labels.map(esc).join("<br>") : `<span class="muted">${t("corrections.noneApply")}</span>`;
}

function correctionRow(item) {
  const more = item.text_truncated ? ` <span class="muted">${t("corrections.truncated")}</span>` : "";
  return `<tr>
    <td>${esc(fmtDate(item.recorded_at))}</td>
    <td>${esc(item.model_name || "–")}</td>
    <td>${esc(item.text)}${more}</td>
    <td>${correctionLabels(item.predicted || [])}</td>
    <td>${correctionLabels(item.corrected || [])}</td>
  </tr>`;
}

function correctionsTable(rows) {
  return `<div class="table-wrap" tabindex="0"><table>
    <thead><tr><th>${t("corrections.col.when")}</th><th>${t("corrections.col.model")}</th>
      <th>${t("corrections.col.text")}</th><th>${t("corrections.col.predicted")}</th>
      <th>${t("corrections.col.corrected")}</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

/* The model filter keeps its choice when the list reloads: the options come from the answer,
   so a correction for a new model appears there without a page reload. */
function fillCorrectionModels(models) {
  const select = $("#corrections-model");
  const chosen = select.value;
  select.innerHTML = `<option value="">${t("corrections.filter.all")}</option>`
    + Object.entries(models).map(([name, count]) =>
      `<option value="${esc(name)}">${esc(name)} (${count})</option>`).join("");
  select.value = chosen;
}

/* `more` appends the next page; otherwise the list starts over (tab opened, filter changed,
   something imported). */
async function loadCorrections(more = false) {
  const list = $("#corrections-list"), moreBtn = $("#corrections-more");
  const model = $("#corrections-model").value;
  const offset = more ? correctionsShown : 0;
  const filter = model ? `&model=${encodeURIComponent(model)}` : "";
  let answer;
  try {
    answer = await Api.get(`/feedback?limit=${CORRECTIONS_PAGE}&offset=${offset}${filter}`);
  } catch (err) {
    moreBtn.hidden = true;
    // A readonly key may record a correction but not read the collection: say that, not 403.
    const message = err.status === 403 ? t("corrections.adminOnly") : esc(err.message);
    list.innerHTML = `<p class="error" role="alert">${message}</p>`;
    return;
  }
  fillCorrectionModels(answer.models);
  correctionsShown = offset + answer.items.length;
  $("#corrections-summary").textContent = answer.total
    ? t("corrections.summary", { shown: correctionsShown, total: answer.total }) : "";
  moreBtn.hidden = correctionsShown >= answer.total;
  if (!answer.total) {
    list.innerHTML = `<p class="muted">${t("corrections.empty")}</p>`;
    return;
  }
  const rows = answer.items.map(correctionRow).join("");
  const body = more && list.querySelector("tbody");
  if (body) body.insertAdjacentHTML("beforeend", rows);
  else list.innerHTML = correctionsTable(rows);
}

async function onCorrectionsDataset(ev) {
  ev.preventDefault();
  const errEl = $("#corrections-dataset-error"), input = $("#corrections-dataset-name");
  errEl.hidden = true;
  const name = input.value.trim();
  if (!name) { showError(errEl, { message: t("corrections.dataset.nameMissing") }); return; }
  const idle = busy($("#corrections-dataset-btn"));
  try {
    const saved = await Api.post("/feedback/dataset", { dataset_name: name });
    toast(t("corrections.dataset.saved", { name: saved.dataset_name, rows: saved.rows }));
    input.value = "";
  } catch (err) { showError(errEl, err); }
  finally { idle(); }
}

async function onCorrectionsImport(ev) {
  ev.preventDefault();
  const errEl = $("#corrections-import-error"), fileInput = $("#corrections-import-file");
  errEl.hidden = true;
  if (!fileInput.files || !fileInput.files.length) {
    showError(errEl, { message: t("corrections.import.fileMissing") });
    return;
  }
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  const idle = busy($("#corrections-import-btn"));
  try {
    const result = await Api.postForm("/feedback/import", form);
    toast(t("corrections.import.done", result));
    fileInput.value = "";
    await loadCorrections();
  } catch (err) { showError(errEl, err); }
  finally { idle(); }
}

async function downloadCorrections(button, path, filename) {
  const idle = busy(button);
  try { await Api.download(path, filename); }
  catch (err) { toastError(err); }
  finally { idle(); }
}

/* Called once from boot(). */
function bindCorrections() {
  $("#corrections-model").addEventListener("change", () => loadCorrections());
  $("#corrections-more").addEventListener("click", () => loadCorrections(true));
  $("#corrections-dataset-form").addEventListener("submit", onCorrectionsDataset);
  $("#corrections-import-form").addEventListener("submit", onCorrectionsImport);
  $("#corrections-export-csv").addEventListener("click", (ev) =>
    downloadCorrections(ev.currentTarget, "/feedback/export", "corrections-training.csv"));
  $("#corrections-export-backup").addEventListener("click", (ev) =>
    downloadCorrections(ev.currentTarget, "/feedback/export?format=jsonl", "corrections-backup.jsonl"));
}
