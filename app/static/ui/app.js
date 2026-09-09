/* Admin UI views. Vanilla JS on purpose: no build step, no dependencies.
   Every user-facing string comes from i18n.js — see there for how German and
   English are resolved and what `t()` does and does not escape. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* Bars carry their fill in `data-width` and get it applied here, because the UI's
   CSP (`default-src 'self'`, no 'unsafe-inline') blocks a style ATTRIBUTE: a
   `style="width:4%"` injected via innerHTML is dropped, and the bar then inherits
   its container's full width — a 0.004 prediction looked exactly like a 1.000 one.
   The CSSOM is not governed by style-src, so setting the property works. Call this
   after every innerHTML render that contains a bar. */
function applyBarWidths(root) {
  root.querySelectorAll("[data-width]").forEach((el) => {
    const pct = Math.max(0, Math.min(100, Number(el.dataset.width) || 0));
    el.style.width = `${pct}%`;
  });
}

let toastTimer;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

function showError(el, err) {
  el.textContent = err.message || String(err);
  el.hidden = false;
}

/* ---------- login / shell ---------- */

async function boot() {
  window.addEventListener("apiv3-unauthorized", showLogin);
  $("#login-form").addEventListener("submit", onLogin);
  $("#logout-btn").addEventListener("click", () => { Api.clearKey(); showLogin(); });
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  document.querySelector('[role="tablist"]').addEventListener("keydown", onTablistKeydown);
  $("#query-form").addEventListener("submit", onQuery);
  document.querySelectorAll('input[name="query-mode"]').forEach(
    (radio) => radio.addEventListener("change", onQueryModeChange));
  $("#train-chip").addEventListener("click", () => switchTab("training"));
  $("#train-form").addEventListener("submit", onTrainStart);
  $("#train-dataset").addEventListener("change", loadDatasetColumns);
  $("#train-preflight").addEventListener("click", runPreflight);
  $("#train-name").addEventListener("input", renderNamePreview);
  $("#upload-form").addEventListener("submit", onUpload);
  $("#import-form").addEventListener("submit", onImportModel);

  if (Api.getKey()) {
    try { await Api.get("/models"); showApp(false); return; } catch { Api.clearKey(); }
  }
  // Mirror the server's auth setting (like /docs): with APIV3_AUTH_ENABLED=false
  // everything works without a key, so the UI must not force a sign-in either.
  try { await Api.get("/models"); showApp(true); return; } catch { /* auth is on */ }
  showLogin();
}

function showLogin() {
  stopStatusPolling();
  $("#view-app").hidden = true;
  $("#view-login").hidden = false;
  $("#api-key").focus();
}

async function onLogin(ev) {
  ev.preventDefault();
  const btn = $("#login-btn"), errEl = $("#login-error");
  errEl.hidden = true;
  btn.disabled = true;
  Api.setKey($("#api-key").value.trim());
  try {
    await Api.get("/models");            // any valid key answers 200 here
    $("#api-key").value = "";
    showApp(false);
  } catch (err) {
    Api.clearKey();
    showError(errEl, err.status === 401 ? { message: t("login.rejected") } : err);
  } finally { btn.disabled = false; }
}

function showApp(keyless) {
  $("#view-login").hidden = true;
  $("#view-app").hidden = false;
  $("#logout-btn").hidden = keyless;       // nothing to sign out of without a key
  $("#auth-off-badge").hidden = !keyless;
  switchTab("query");
  startStatusPolling();
}

const loaders = { query: loadQueryTab, training: loadTrainingTab, models: loadModels, datasets: loadDatasets };

function switchTab(name, { focus = false } = {}) {
  document.querySelectorAll(".tab").forEach((b) => {
    const selected = b.dataset.tab === name;
    b.setAttribute("aria-selected", selected ? "true" : "false");
    b.tabIndex = selected ? 0 : -1;  // roving tabindex: only the active tab is in the Tab order
    if (selected && focus) b.focus();
  });
  document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  loaders[name]();
}

// WAI-ARIA tabs keyboard model: Left/Right cycle, Home/End jump. Order comes
// from the DOM so it can never drift from the markup.
function onTablistKeydown(ev) {
  const moves = { ArrowRight: 1, ArrowLeft: -1, Home: "first", End: "last" };
  if (!(ev.key in moves)) return;
  ev.preventDefault();
  const tabs = [...document.querySelectorAll(".tab")];
  const cur = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
  const move = moves[ev.key];
  const next = move === "first" ? 0
    : move === "last" ? tabs.length - 1
    : (cur + move + tabs.length) % tabs.length;
  switchTab(tabs[next].dataset.tab, { focus: true });
}

// Last: every tab's loader has to be defined before the shell asks for one.
boot();
