/* Expiring share links, for models and datasets alike: create one, show it with a
   copy button, and list what is still outstanding so it can be revoked.

   Its own file because BOTH management tabs use it — folding it into models.js
   would make the datasets view depend on the models view for no reason. */
"use strict";

/* What is being shared right now. A double click posted twice and created two links -- two
   bearer capabilities for one intent (audit 2026-09-30, U08). Keyed on the resource rather
   than on a button, because three places offer the same action. */
const sharing = new Set();

/* Create an expiring share link for a model or dataset and show it (with copy). */
async function shareResource(kind, name, boxSel) {
  const key = `${kind}/${name}`;
  if (sharing.has(key)) return;
  sharing.add(key);
  const box = document.querySelector(boxSel);
  try {
    const r = await Api.post(`/${kind}/${encodeURIComponent(name)}/export`,
                             { generate_share_url: true, expires_hours: 24 });
    const url = location.origin + r.share_url;
    box.innerHTML = `
      <div class="card share-box">
        <strong>${t("share.headingFor", { name: esc(name) })}</strong>
        <p class="muted">${t("share.expires", { when: esc(fmtWhen(r.expires_at)) })}</p>
        <div class="share-row">
          <input type="text" readonly value="${esc(url)}" aria-label="${esc(t("share.urlLabel"))}">
          <button type="button" class="small" data-copy="${esc(url)}">${t("common.copy")}</button>
          <button type="button" class="small ghost" data-close>${t("common.close")}</button>
        </div>
      </div>`;
    // No inline handlers: the UI's CSP has no 'unsafe-inline' for scripts.
    box.querySelector("input[readonly]").addEventListener("focus", (ev) => ev.target.select());
    box.querySelector("[data-copy]").addEventListener("click",
      (ev) => copyText(ev.target.dataset.copy, t("share.copied")));
    box.querySelector("[data-close]").addEventListener("click",
                                                       closer(() => { box.innerHTML = ""; }));
    renderShareLinks(kind, `#${kind}-links`);   // the new link joins the overview
  } catch (err) { toastError(err); } finally { sharing.delete(key); }
}

/* Active share links for one resource kind. A link is a bearer capability valid for
   up to a week: whoever holds the URL downloads without a key. Handing one out was
   always possible; seeing what is outstanding, and taking it back, was not. */
const fmtWhen = (iso) => (iso ? new Date(iso).toLocaleString(I18n.locale()) : "–");
// The UI addresses resources in the plural (routes, container ids); the store records
// the singular kind it was created with. Map explicitly rather than slicing an "s".
const SHARE_KIND = { models: "model", datasets: "dataset" };

async function renderShareLinks(kind, boxSel) {
  const box = document.querySelector(boxSel);
  let links;
  try { links = (await Api.get("/share")).filter((l) => l.kind === SHARE_KIND[kind]); }
  catch (err) {
    // A 403 is the expected answer for a readonly key — the listing is admin-only, and
    // showing an error for a permission the user is not meant to have is noise. Anything
    // else is a real failure, and blanking the box for it made outstanding links disappear
    // from the overview while staying live and downloadable for up to a week.
    box.innerHTML = "";
    if (err.status !== 403) box.innerHTML = `<p class="error">${esc(t("share.listFailed"))}</p>`;
    return;
  }
  if (!links.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="card table-wrap" tabindex="0">
    <h3>${t("share.activeHeading")} <span class="muted">(${links.length})</span></h3>
    <p class="muted">${t("share.activeNote")}</p>
    <table><thead><tr><th>${kind === "models" ? t("share.table.model") : t("share.table.dataset")}</th>
      <th>${t("share.table.created")}</th><th>${t("share.table.expires")}</th>
      <th>${t("common.actions")}</th></tr></thead><tbody>` +
    links.map((l) => `<tr>
      <td>${esc(l.name)}</td><td>${esc(fmtWhen(l.created_at))}</td>
      <td>${esc(fmtWhen(l.expires_at))}</td>
      <td class="actions">
        <button class="small" data-copylink="${esc(l.share_id)}">${t("share.copyLink")}</button>
        <button class="small danger" data-revoke="${esc(l.share_id)}">${t("share.revoke")}</button>
      </td></tr>`).join("") + `</tbody></table></div>`;
  box.querySelectorAll("[data-copylink]").forEach((b) => b.addEventListener("click",
    () => copyText(`${location.origin}/share/${b.dataset.copylink}`, t("share.copied"))));
  box.querySelectorAll("[data-revoke]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm(t("share.revokeConfirm"))) return;
    try {
      await Api.del(`/share/${encodeURIComponent(b.dataset.revoke)}`);
      toast(t("share.revoked"));
      renderShareLinks(kind, boxSel);
    } catch (err) { toastError(err); }
  }));
}
