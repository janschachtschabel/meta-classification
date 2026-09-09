/* The Datasets tab: list, upload, download, share, delete. */
"use strict";

/* ---------- datasets ---------- */

async function loadDatasets() {
  const el = $("#datasets-list");
  el.innerHTML = `<p class="muted">Loading …</p>`;
  try {
    const list = await Api.get("/datasets");
    if (!list.length) { el.innerHTML = `<p class="muted">No datasets yet — upload a CSV above.</p>`; return; }
    el.innerHTML = `<div class="card table-wrap"><table>
      <thead><tr><th>Name</th><th class="num">Rows</th><th class="num">Size</th><th>Actions</th></tr></thead><tbody>` +
      list.map((d) => `<tr>
        <td><button type="button" class="linkish" data-dsdetail="${esc(d.name)}">${esc(d.name)}</button></td>
        <td class="num">${d.rows}</td>
        <td class="num">${esc(d.size_human)}</td>
        <td class="actions">
          <button class="small" data-dl="${esc(d.name)}">Download</button>
          <button class="small" data-share="${esc(d.name)}">Share link</button>
          <button class="small danger" data-delds="${esc(d.name)}">Delete</button>
        </td></tr>`).join("") +
      `</tbody></table></div>`;
    el.querySelectorAll("[data-dsdetail]").forEach((b) => b.addEventListener("click", () =>
      showDatasetDetail(b.dataset.dsdetail)));
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
  finally { renderShareLinks("datasets", "#datasets-links"); }  // same reason as loadModels
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
