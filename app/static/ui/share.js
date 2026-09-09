/* Expiring share links, for models and datasets alike: create one, show it with a
   copy button, and list what is still outstanding so it can be revoked.

   Its own file because BOTH management tabs use it — folding it into models.js
   would make the datasets view depend on the models view for no reason. */
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
    renderShareLinks(kind, `#${kind}-links`);   // the new link joins the overview
  } catch (err) { toast(err.message); }
}

/* Active share links for one resource kind. A link is a bearer capability valid for
   up to a week: whoever holds the URL downloads without a key. Handing one out was
   always possible; seeing what is outstanding, and taking it back, was not. */
const fmtWhen = (iso) => (iso ? new Date(iso).toLocaleString() : "–");
// The UI addresses resources in the plural (routes, container ids); the store records
// the singular kind it was created with. Map explicitly rather than slicing an "s".
const SHARE_KIND = { models: "model", datasets: "dataset" };

async function renderShareLinks(kind, boxSel) {
  const box = document.querySelector(boxSel);
  let links;
  try { links = (await Api.get("/share")).filter((l) => l.kind === SHARE_KIND[kind]); }
  catch { box.innerHTML = ""; return; }   // readonly key: the listing is admin-only
  if (!links.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="card table-wrap">
    <h3>Active share links <span class="muted">(${links.length})</span></h3>
    <p class="muted">Anyone with the link can download without an API key until it expires.
       Revoking stops further downloads; it cannot recall what was already fetched.</p>
    <table><thead><tr><th>${kind === "models" ? "Model" : "Dataset"}</th>
      <th>Created</th><th>Expires</th><th>Actions</th></tr></thead><tbody>` +
    links.map((l) => `<tr>
      <td>${esc(l.name)}</td><td>${esc(fmtWhen(l.created_at))}</td>
      <td>${esc(fmtWhen(l.expires_at))}</td>
      <td class="actions">
        <button class="small" data-copylink="${esc(l.share_id)}">Copy link</button>
        <button class="small danger" data-revoke="${esc(l.share_id)}">Revoke</button>
      </td></tr>`).join("") + `</tbody></table></div>`;
  box.querySelectorAll("[data-copylink]").forEach((b) => b.addEventListener("click", async () => {
    await navigator.clipboard.writeText(`${location.origin}/share/${b.dataset.copylink}`);
    toast("Link copied.");
  }));
  box.querySelectorAll("[data-revoke]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Revoke this share link? Anyone still holding it loses access.")) return;
    try {
      await Api.del(`/share/${encodeURIComponent(b.dataset.revoke)}`);
      toast("Share link revoked.");
      renderShareLinks(kind, boxSel);
    } catch (err) { toast(err.message); }
  }));
}
