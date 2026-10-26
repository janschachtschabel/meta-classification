/* The Models tab: list, documentation, rename, import, export, share, delete. */
"use strict";

/* ---------- models ---------- */

/* metrics.json content is bundle-controlled (imports accept arbitrary JSON
   there) — coerce before formatting so a crafted value can neither inject
   markup (esc at the call site) nor crash the table render (toFixed on a
   non-number throws). */
function fmtScore(value) {
  return fmtFixed(value, 3);
}

/* Bundles older than format 2 keep their TF-IDF vocabulary inside the skops
   container and no longer load — /predict answers 422. They are still listed
   (they exist, can be downloaded and repaired elsewhere), but the table has to
   say so instead of showing their metrics as if they were servable. */
const CURRENT_BUNDLE_FORMAT = 2;
const isServable = (m) => (m.format_version ?? 1) >= CURRENT_BUNDLE_FORMAT;

/* The display name a model's documentation gives it, or "" -- bundle data, so only a string
   counts. Shown beside the technical name wherever a model is picked (here and in query.js). */
function modelDisplayName(model) {
  const info = (model && model.metadata && model.metadata.info) || {};
  return typeof info.display_name === "string" ? info.display_name.trim() : "";
}

/* Documentation a model carries when it is handed to someone else. Rendered from
   bundle-controlled data, so every value goes through esc() — an imported archive
   can put arbitrary text in metrics.json. */
const INFO_FIELDS = [
  ["display_name", "models.info.displayName", "models.info.displayName.hint", "text"],
  ["author", "models.info.author", "models.info.author.hint", "text"],
  ["description", "models.info.description", "models.info.description.hint", "textarea"],
  ["data_source", "models.info.dataSource", "models.info.dataSource.hint", "text"],
  ["license", "models.info.license", "models.info.license.hint", "text"],
];

async function editModelInfo(name) {
  const box = document.querySelector("#models-share");
  let model;
  try { model = await Api.get(`/models/${encodeURIComponent(name)}`); }
  catch (err) { toastError(err); return; }
  const info = (model.metadata && model.metadata.info) || {};
  const vocab = model.label_vocabulary;
  box.innerHTML = `
    <div class="card">
      <strong>${t("models.info.heading", { name: esc(name) })}</strong>
      <p class="muted">${t("models.info.note")}</p>
      <dl class="train-status"><dt>${t("models.info.vocabulary")}</dt><dd>${vocab
        ? esc(vocab) + ` <span class="muted">${t("models.info.vocabularyDerived")}</span>`
        : `<span class="muted">${t("models.info.vocabularyNone")}</span>`}</dd></dl>
      <form id="info-form">
        ${INFO_FIELDS.map(([key, labelKey, hintKey, kind]) => `
          <label for="info-${key}">${t(labelKey)}</label>
          ${kind === "textarea"
            ? `<textarea id="info-${key}" name="${key}" rows="3" maxlength="4000"
                 placeholder="${esc(t(hintKey))}">${esc(info[key] || "")}</textarea>`
            : `<input id="info-${key}" name="${key}" type="text" maxlength="${{ data_source: 1000, display_name: 100 }[key] || 200}"
                 placeholder="${esc(t(hintKey))}" value="${esc(info[key] || "")}">`}`).join("")}
        <p id="info-error" class="error" role="alert" hidden></p>
        <button type="submit" id="info-save">${t("common.save")}</button>
        <button type="button" class="ghost small" data-close>${t("common.close")}</button>
      </form>
    </div>`;
  const close = closer(() => { box.innerHTML = ""; });
  box.querySelector("[data-close]").addEventListener("click", close);
  box.querySelector("#info-form").addEventListener("submit", (ev) => submitInfo(ev, name, box, close));
  box.querySelector("#info-display_name").focus();
}

async function submitInfo(ev, name, box, close = () => { box.innerHTML = ""; }) {
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
    toast(t("models.info.saved"));
    close();   // the Save button is inside what this empties; `close` knows where to go
    await loadModels();   // the table shows the display name the documentation holds
  } catch (err) { showError(errEl, err); }
}

/* Correct a model's name. The server revokes the old name's share links -- a link resolves
   by name and would otherwise serve whatever model is given it next -- so the form says how
   many before anyone confirms. Counting them needs the admin key; without it the warning
   stays, only without a number. */
async function renameModel(name) {
  const box = document.querySelector("#models-share");
  let links = null;
  try {
    links = (await Api.get("/share")).filter((l) => l.kind === "model" && l.name === name).length;
  } catch (err) {
    links = null;  // reported as "cannot be counted" below, not hidden
  }
  const warning = links === null ? `<strong>${t("models.rename.linksUnknown")}</strong>`
    : links ? `<strong>${t("models.rename.linksWarning", { count: links })}</strong>`
    : `<span class="muted">${t("models.rename.noLinks")}</span>`;
  box.innerHTML = `
    <div class="card">
      <strong>${t("models.rename.heading", { name: esc(name) })}</strong>
      <p class="muted">${t("models.rename.note")}</p>
      <form id="rename-form" novalidate>
        <label for="rename-new">${t("models.rename.label")}</label>
        <input id="rename-new" type="text" maxlength="200" required value="${esc(name)}"
               aria-describedby="rename-warning">
        <p id="rename-warning">${warning}</p>
        <p id="rename-error" class="error" role="alert" hidden></p>
        <button type="submit" id="rename-save">${t("models.rename.submit")}</button>
        <button type="button" class="ghost small" data-close>${t("common.close")}</button>
      </form>
    </div>`;
  const close = closer(() => { box.innerHTML = ""; });
  box.querySelector("[data-close]").addEventListener("click", close);
  box.querySelector("#rename-form").addEventListener("submit", (ev) => submitRename(ev, name, box, close));
  box.querySelector("#rename-new").focus();
}

async function submitRename(ev, name, box, close = () => { box.innerHTML = ""; }) {
  ev.preventDefault();
  const errEl = box.querySelector("#rename-error");
  errEl.hidden = true;
  const next = box.querySelector("#rename-new").value.trim();
  if (!next) { showError(errEl, { message: t("models.rename.empty") }); return; }
  if (next === name) { showError(errEl, { message: t("models.rename.unchanged") }); return; }
  try {
    const done = await Api.post(`/models/${encodeURIComponent(name)}/rename`, { new_name: next });
    toast(t("models.rename.done", { name: done.name, count: done.revoked_links }));
    close();
    await loadModels();
  } catch (err) { showError(errEl, err); }
}

async function loadModels() {
  const el = $("#models-list");
  el.innerHTML = `<p class="muted">${t("common.loading")}</p>`;
  try {
    const names = await Api.get("/models");
    if (!names.length) { el.innerHTML = `<p class="muted">${t("models.empty")}</p>`; return; }
    const infos = await Promise.all(names.map((n) => Api.get(`/models/${encodeURIComponent(n)}`)));
    el.innerHTML = `<div class="card table-wrap" tabindex="0"><table>
      <thead><tr><th>${t("models.table.name")}</th><th>${t("models.table.task")}</th>
      <th class="num">${t("models.table.labels")}</th><th class="num">${t("models.table.f1Macro")}</th>
      <th class="num">${t("models.table.f1Micro")}</th><th>${t("models.table.evaluation")}</th>
      <th>${t("common.actions")}</th></tr></thead><tbody>` +
      infos.map((m) => `<tr${isServable(m) ? "" : ' class="stale"'}>
        <td><button type="button" class="linkish" data-detail="${esc(m.name)}">${esc(modelDisplayName(m) || m.name)}</button>${
          modelDisplayName(m) ? ` <span class="muted">${esc(m.name)}</span>` : ""}${isServable(m) ? "" :
          ` <span class="badge-stale">${t("models.needsRetraining")}</span>`}</td><td>${esc(m.task_type)}</td>
        <td class="num">${esc(m.metadata.n_labels ?? "–")}</td>
        <td class="num">${esc(fmtScore(m.metadata.metrics && m.metadata.metrics.f1_macro))}</td>
        <td class="num">${esc(fmtScore(m.metadata.metrics && m.metadata.metrics.f1_micro))}</td>
        <td>${isServable(m) ? esc(m.metadata.evaluation || "–")
          : t("models.oldFormat", { version: esc(String(m.format_version ?? 1)) })}</td>
        <td class="actions">
          <button class="small" data-export="${esc(m.name)}">${t("common.download")}</button>
          <button class="small" data-info="${esc(m.name)}">${t("models.action.info")}</button>
          <button class="small" data-rename="${esc(m.name)}">${t("models.action.rename")}</button>
          <button class="small" data-share="${esc(m.name)}">${t("common.shareLink")}</button>
          <button class="small danger" data-delete="${esc(m.name)}">${t("common.delete")}</button>
        </td></tr>`).join("") + `</tbody></table></div>`;
    el.querySelectorAll("[data-export]").forEach((b) => b.addEventListener("click", () =>
      Api.download(`/models/${encodeURIComponent(b.dataset.export)}/export`, `${b.dataset.export}.zip`)
        .catch((err) => toastError(err))));
    el.querySelectorAll("[data-detail]").forEach((b) => b.addEventListener("click", () =>
      showModelDetail(b.dataset.detail)));
    el.querySelectorAll("[data-info]").forEach((b) => b.addEventListener("click", () =>
      editModelInfo(b.dataset.info)));
    el.querySelectorAll("[data-rename]").forEach((b) => b.addEventListener("click", () =>
      renameModel(b.dataset.rename)));
    el.querySelectorAll("[data-share]").forEach((b) => b.addEventListener("click", () =>
      shareResource("models", b.dataset.share, "#models-share")));
    el.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(t("models.deleteConfirm", { name: b.dataset.delete }))) return;
      try { await Api.del(`/models/${encodeURIComponent(b.dataset.delete)}`); toast(t("models.deleted")); loadModels(); }
      catch (err) { toastError(err); }
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
  if (!fileInput.files.length) { showError(errEl, { message: t("models.import.noFile") }); return; }
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  const newName = $("#import-name").value.trim();
  if (newName) form.append("new_name", newName);
  const idle = busy(btn);
  try {
    await Api.postForm("/models/import", form);
    toast(t("models.imported"));
    fileInput.value = "";
    $("#import-name").value = "";
    loadModels();
  } catch (err) { showError(errEl, err); }
  finally { idle(); }
}
