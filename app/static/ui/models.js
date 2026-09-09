/* The Models tab: list, documentation, import, export, share, delete. */
"use strict";

/* ---------- models ---------- */

/* metrics.json content is bundle-controlled (imports accept arbitrary JSON
   there) — coerce before formatting so a crafted value can neither inject
   markup (esc at the call site) nor crash the table render (toFixed on a
   non-number throws). */
function fmtScore(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "–";
}

/* Bundles older than format 2 keep their TF-IDF vocabulary inside the skops
   container and no longer load — /predict answers 422. They are still listed
   (they exist, can be downloaded and repaired elsewhere), but the table has to
   say so instead of showing their metrics as if they were servable. */
const CURRENT_BUNDLE_FORMAT = 2;
const isServable = (m) => (m.format_version ?? 1) >= CURRENT_BUNDLE_FORMAT;

/* Documentation a model carries when it is handed to someone else. Rendered from
   bundle-controlled data, so every value goes through esc() — an imported archive
   can put arbitrary text in metrics.json. */
const INFO_FIELDS = [
  ["author", "Author / contact", "text", "e.g. Redaktion WLO <redaktion@example.org>"],
  ["description", "Purpose and limits", "textarea", "What is this model for — and what is it NOT for?"],
  ["data_source", "Where the data came from", "text", "e.g. WLO prod export 2026-07-26, school disciplines"],
  ["license", "License / terms", "text", "e.g. CC BY-SA 4.0"],
];

async function editModelInfo(name) {
  const box = document.querySelector("#models-share");
  let model;
  try { model = await Api.get(`/models/${encodeURIComponent(name)}`); }
  catch (err) { toast(err.message); return; }
  const info = (model.metadata && model.metadata.info) || {};
  const vocab = model.label_vocabulary;
  box.innerHTML = `
    <div class="card">
      <strong>Documentation for "${esc(name)}"</strong>
      <p class="muted">Travels inside the exported bundle. Everything else in the model's
         metadata is measured; these are your statements. Editable any time — no retrain.</p>
      <dl class="train-status"><dt>Label vocabulary</dt><dd>${vocab
        ? esc(vocab) + ` <span class="muted">(derived from the labels, not editable)</span>`
        : `<span class="muted">none — the labels share no namespace</span>`}</dd></dl>
      <form id="info-form">
        ${INFO_FIELDS.map(([key, label, kind, hint]) => `
          <label for="info-${key}">${esc(label)}</label>
          ${kind === "textarea"
            ? `<textarea id="info-${key}" name="${key}" rows="3" maxlength="4000"
                 placeholder="${esc(hint)}">${esc(info[key] || "")}</textarea>`
            : `<input id="info-${key}" name="${key}" type="text" maxlength="${key === "data_source" ? 1000 : 200}"
                 placeholder="${esc(hint)}" value="${esc(info[key] || "")}">`}`).join("")}
        <p id="info-error" class="error" role="alert" hidden></p>
        <button type="submit" id="info-save">Save</button>
        <button type="button" class="ghost small" data-close>Close</button>
      </form>
    </div>`;
  box.querySelector("[data-close]").addEventListener("click", () => { box.innerHTML = ""; });
  box.querySelector("#info-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const errEl = box.querySelector("#info-error");
    errEl.hidden = true;
    // Empty means "not stated": send only what was filled in, so the stored block
    // never claims a field the user left blank.
    const body = {};
    for (const [key] of INFO_FIELDS) {
      const value = box.querySelector(`#info-${key}`).value.trim();
      if (value) body[key] = value;
    }
    try {
      await Api.put(`/models/${encodeURIComponent(name)}/info`, body);
      toast("Documentation saved.");
      box.innerHTML = "";
    } catch (err) { showError(errEl, err); }
  });
  box.querySelector("#info-author").focus();
}

async function loadModels() {
  const el = $("#models-list");
  el.innerHTML = `<p class="muted">Loading …</p>`;
  try {
    const names = await Api.get("/models");
    if (!names.length) { el.innerHTML = `<p class="muted">No models yet — start a training run first.</p>`; return; }
    const infos = await Promise.all(names.map((n) => Api.get(`/models/${encodeURIComponent(n)}`)));
    el.innerHTML = `<div class="card table-wrap"><table>
      <thead><tr><th>Name</th><th>Task</th><th class="num">Labels</th><th class="num">F1 macro</th>
      <th class="num">F1 micro</th><th>Evaluation</th><th>Actions</th></tr></thead><tbody>` +
      infos.map((m) => `<tr${isServable(m) ? "" : ' class="stale"'}>
        <td><button type="button" class="linkish" data-detail="${esc(m.name)}">${esc(m.name)}</button>${isServable(m) ? "" :
          ` <span class="badge-stale">needs retraining</span>`}</td><td>${esc(m.task_type)}</td>
        <td class="num">${esc(m.metadata.n_labels ?? "–")}</td>
        <td class="num">${esc(fmtScore(m.metadata.metrics && m.metadata.metrics.f1_macro))}</td>
        <td class="num">${esc(fmtScore(m.metadata.metrics && m.metadata.metrics.f1_micro))}</td>
        <td>${isServable(m) ? esc(m.metadata.evaluation || "–")
          : `Old bundle format ${esc(m.format_version ?? 1)} — cannot be loaded for classification.`}</td>
        <td class="actions">
          <button class="small" data-export="${esc(m.name)}">Download</button>
          <button class="small" data-info="${esc(m.name)}">Info</button>
          <button class="small" data-share="${esc(m.name)}">Share link</button>
          <button class="small danger" data-delete="${esc(m.name)}">Delete</button>
        </td></tr>`).join("") + `</tbody></table></div>`;
    el.querySelectorAll("[data-export]").forEach((b) => b.addEventListener("click", () =>
      Api.download(`/models/${encodeURIComponent(b.dataset.export)}/export`, `${b.dataset.export}.zip`)
        .catch((err) => toast(err.message))));
    el.querySelectorAll("[data-detail]").forEach((b) => b.addEventListener("click", () =>
      showModelDetail(b.dataset.detail)));
    el.querySelectorAll("[data-info]").forEach((b) => b.addEventListener("click", () =>
      editModelInfo(b.dataset.info)));
    el.querySelectorAll("[data-share]").forEach((b) => b.addEventListener("click", () =>
      shareResource("models", b.dataset.share, "#models-share")));
    el.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Delete model "${b.dataset.delete}"? This cannot be undone.`)) return;
      try { await Api.del(`/models/${encodeURIComponent(b.dataset.delete)}`); toast("Model deleted."); loadModels(); }
      catch (err) { toast(err.message); }
    }));
  } catch (err) { el.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
  // In `finally`, not after the table: a link outlives the model it points at, so the
  // two paths that skip the table — no models left, and the list request failing — are
  // exactly the ones where a stale overview keeps offering links that are still live.
  finally { renderShareLinks("models", "#models-links"); }
}

async function onImportModel(ev) {
  ev.preventDefault();
  const errEl = $("#import-error"), btn = $("#import-btn"), fileInput = $("#import-file");
  errEl.hidden = true;
  if (!fileInput.files.length) { showError(errEl, { message: "Choose a model ZIP first." }); return; }
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  const newName = $("#import-name").value.trim();
  if (newName) form.append("new_name", newName);
  btn.disabled = true;
  try {
    await Api.postForm("/models/import", form);
    toast("Model imported.");
    fileInput.value = "";
    $("#import-name").value = "";
    loadModels();
  } catch (err) { showError(errEl, err); }
  finally { btn.disabled = false; }
}
