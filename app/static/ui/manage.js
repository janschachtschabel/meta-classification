/* Management views: models (list/export/share/import/delete) and datasets
   (list/upload/download/share/delete). Split from app.js by responsibility. */
"use strict";

/* Create an expiring share link for a model or dataset and show it (with copy). */
async function shareResource(kind, name, boxSel) {
  const box = document.querySelector(boxSel);
  try {
    const r = await Api.post(`/${kind}/${encodeURIComponent(name)}/export`,
                             { generate_share_url: true, expires_hours: 24 });
    const url = location.origin + r.share_url;
    box.innerHTML = `
      <div class="card share-box">
        <strong>Share link for "${esc(name)}"</strong>
        <p class="muted">No API key needed to download; expires ${esc(new Date(r.expires_at).toLocaleString())}.</p>
        <div class="share-row">
          <input type="text" readonly value="${esc(url)}" aria-label="Share URL">
          <button type="button" class="small" data-copy="${esc(url)}">Copy</button>
          <button type="button" class="small ghost" data-close>Close</button>
        </div>
      </div>`;
    // No inline handlers: the UI's CSP has no 'unsafe-inline' for scripts.
    box.querySelector("input[readonly]").addEventListener("focus", (ev) => ev.target.select());
    box.querySelector("[data-copy]").addEventListener("click", async (ev) => {
      await navigator.clipboard.writeText(ev.target.dataset.copy);
      toast("Link copied.");
    });
    box.querySelector("[data-close]").addEventListener("click", () => { box.innerHTML = ""; });
  } catch (err) { toast(err.message); }
}

/* ---------- models ---------- */

/* metrics.json content is bundle-controlled (imports accept arbitrary JSON
   there) — coerce before formatting so a crafted value can neither inject
   markup (esc at the call site) nor crash the table render (toFixed on a
   non-number throws). */
function fmtScore(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "–";
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
      infos.map((m) => `<tr>
        <td>${esc(m.name)}</td><td>${esc(m.task_type)}</td>
        <td class="num">${esc(m.metadata.n_labels ?? "–")}</td>
        <td class="num">${esc(fmtScore(m.metadata.metrics && m.metadata.metrics.f1_macro))}</td>
        <td class="num">${esc(fmtScore(m.metadata.metrics && m.metadata.metrics.f1_micro))}</td>
        <td>${esc(m.metadata.evaluation || "–")}</td>
        <td class="actions">
          <button class="small" data-export="${esc(m.name)}">Download</button>
          <button class="small" data-share="${esc(m.name)}">Share link</button>
          <button class="small danger" data-delete="${esc(m.name)}">Delete</button>
        </td></tr>`).join("") + `</tbody></table></div>`;
    el.querySelectorAll("[data-export]").forEach((b) => b.addEventListener("click", () =>
      Api.download(`/models/${encodeURIComponent(b.dataset.export)}/export`, `${b.dataset.export}.zip`)
        .catch((err) => toast(err.message))));
    el.querySelectorAll("[data-share]").forEach((b) => b.addEventListener("click", () =>
      shareResource("models", b.dataset.share, "#models-share")));
    el.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Delete model "${b.dataset.delete}"? This cannot be undone.`)) return;
      try { await Api.del(`/models/${encodeURIComponent(b.dataset.delete)}`); toast("Model deleted."); loadModels(); }
      catch (err) { toast(err.message); }
    }));
  } catch (err) { el.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
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

/* ---------- datasets ---------- */

async function loadDatasets() {
  const el = $("#datasets-list");
  el.innerHTML = `<p class="muted">Loading …</p>`;
  try {
    const list = await Api.get("/datasets");
    if (!list.length) { el.innerHTML = `<p class="muted">No datasets yet — upload a CSV above.</p>`; return; }
    el.innerHTML = `<div class="card table-wrap"><table>
      <thead><tr><th>Name</th><th class="num">Rows</th><th class="num">Size</th><th>Actions</th></tr></thead><tbody>` +
      list.map((d) => `<tr><td>${esc(d.name)}</td><td class="num">${d.rows}</td>
        <td class="num">${esc(d.size_human)}</td>
        <td class="actions">
          <button class="small" data-dl="${esc(d.name)}">Download</button>
          <button class="small" data-share="${esc(d.name)}">Share link</button>
          <button class="small danger" data-delds="${esc(d.name)}">Delete</button>
        </td></tr>`).join("") +
      `</tbody></table></div>`;
    el.querySelectorAll("[data-dl]").forEach((b) => b.addEventListener("click", () =>
      Api.download(`/datasets/${encodeURIComponent(b.dataset.dl)}/export`, b.dataset.dl)
        .catch((err) => toast(err.message))));
    el.querySelectorAll("[data-share]").forEach((b) => b.addEventListener("click", () =>
      shareResource("datasets", b.dataset.share, "#datasets-share")));
    el.querySelectorAll("[data-delds]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Delete dataset "${b.dataset.delds}"? This cannot be undone.`)) return;
      try { await Api.del(`/datasets/${encodeURIComponent(b.dataset.delds)}`); toast("Dataset deleted."); loadDatasets(); }
      catch (err) { toast(err.message); }
    }));
  } catch (err) { el.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
}

async function onUpload(ev) {
  ev.preventDefault();
  const errEl = $("#upload-error"), btn = $("#upload-btn"), fileInput = $("#upload-file");
  errEl.hidden = true;
  if (!fileInput.files.length) { showError(errEl, { message: "Choose a CSV file first." }); return; }
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  btn.disabled = true;
  try {
    await Api.postForm("/datasets/import", form);
    toast("Dataset uploaded.");
    fileInput.value = "";
    loadDatasets();
  } catch (err) { showError(errEl, err); }
  finally { btn.disabled = false; }
}
