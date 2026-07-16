/* Admin UI views. Vanilla JS on purpose: no build step, no dependencies.
   simplify: single-locale admin tool — user-facing strings are inline English
   (matching the API's own texts); upgrade path is extracting them to a map. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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
  $("#train-chip").addEventListener("click", () => switchTab("training"));
  $("#train-form").addEventListener("submit", onTrainStart);
  $("#train-dataset").addEventListener("change", loadDatasetColumns);
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
    showError(errEl, err.status === 401 ? { message: "This key was not accepted." } : err);
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

/* ---------- query ---------- */

async function loadQueryTab() {
  const sel = $("#query-models");
  try {
    const names = await Api.get("/models");
    sel.innerHTML = names.map((n) => `<option>${esc(n)}</option>`).join("");
    if (!names.length) $("#query-results").innerHTML =
      `<p class="muted">No models yet — train one on the Training tab first.</p>`;
    else sel.options[0].selected = true;
  } catch (err) { $("#query-results").innerHTML = `<p class="error">${esc(err.message)}</p>`; }
}

async function onQuery(ev) {
  ev.preventDefault();
  const btn = $("#query-btn"), out = $("#query-results");
  const models = [...$("#query-models").selectedOptions].map((o) => o.value);
  if (!models.length) { out.innerHTML = `<p class="error">Select at least one model.</p>`; return; }
  const body = {
    texts: [$("#query-text").value],
    include_baseline_diff: $("#query-diff").checked,
  };
  const topk = $("#query-topk").value;
  if (topk !== "") body.top_k = Number(topk);
  btn.disabled = true;
  $("#query-status").textContent = "Classifying …";
  try {
    let byModel;
    if (models.length === 1) {
      const r = await Api.post("/predict", { ...body, model_name: models[0] });
      byModel = { [models[0]]: r.results[0].predictions };
    } else {
      const r = await Api.post("/predict/multi", { ...body, model_names: models });
      byModel = r.results[0].predictions_by_model;
    }
    out.innerHTML = Object.entries(byModel).map(([name, preds]) => `
      <div class="card"><h3>${esc(name)}</h3>${preds.length ? preds.map((p) => `
        <div class="pred${p.above_threshold === false ? " below-t" : ""}">
          <span class="name">${esc(p.label)}</span>
          <span class="bar"><span style="width:${Math.round(p.confidence * 100)}%"></span></span>
          <span class="val">${p.confidence.toFixed(3)}${p.baseline_diff !== undefined
            ? ` <span class="muted">diff ${p.baseline_diff >= 0 ? "+" : ""}${p.baseline_diff.toFixed(3)}</span>` : ""}${
            p.above_threshold === false ? ` <span class="muted">· below threshold</span>` : ""}</span>
        </div>`).join("")
        : `<p class="muted">No label above the model's threshold.</p>`}</div>`).join("");
  } catch (err) { out.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
  finally { btn.disabled = false; $("#query-status").textContent = ""; }
}

/* ---------- training ---------- */

let pollTimer = null;

async function pollTick() {
  try {
    const s = await Api.get("/train/status");
    renderTrainStatus(s);
    await advanceQueue(s.status);  // start the next queued training when idle
  } catch { /* transient poll failure: keep the last rendered state */ }
}

// Registered ONCE for the page lifetime (guarded by pollTimer): browsers
// throttle timers in hidden tabs, so refresh immediately on return. Attaching it
// inside startStatusPolling() leaked a listener on every logout->login cycle.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && pollTimer) pollTick();
});

function startStatusPolling() {
  if (pollTimer) return;
  pollTick();
  pollTimer = setInterval(pollTick, 2500);
}
function stopStatusPolling() { clearInterval(pollTimer); pollTimer = null; }

function renderTrainStatus(s) {
  renderTrainChip(s);
  const el = $("#train-status");
  const rows = [["Status", s.status], ["Phase", s.phase || "–"], ["Detail", s.phase_detail || "–"],
                ["Model", s.model_name || "–"], ["Elapsed", s.elapsed_seconds != null ? `${s.elapsed_seconds}s` : "–"],
                ["ETA", s.eta_seconds != null ? `~${Math.round(s.eta_seconds)}s` : "–"]];
  let html = `
    <div class="progress" role="progressbar" aria-valuenow="${s.progress}" aria-valuemin="0"
         aria-valuemax="100" aria-label="Training progress"><span style="width:${s.progress}%"></span></div>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
  // elapsed keeps growing even when the thread is dead — only a stale heartbeat
  // (no progress signal from the training thread) reveals a silent stall.
  if (s.status === "running" && s.seconds_since_heartbeat > 120)
    html += `<p class="warn">No progress signal for ${Math.round(s.seconds_since_heartbeat)}s —
      the training thread may be stalled (large save steps can crawl under memory pressure).</p>`;
  if (s.status === "running")
    html += `<button class="small danger" id="train-stop">Stop training</button>`;
  if (s.status === "error") html += `<p class="error">${esc(s.message || "Training failed.")}</p>`;
  if (s.status === "completed" && s.results)
    html += `<p class="ok">Done: ${esc(s.results.model_name)} — F1 macro ${s.results.metrics.f1_macro.toFixed(3)},
             micro ${s.results.metrics.f1_micro.toFixed(3)} (${s.results.n_labels} labels)</p>`;
  el.innerHTML = html;
  const stop = $("#train-stop");
  if (stop) stop.addEventListener("click", async () => {
    trainQueue = [];  // stopping also cancels everything still queued
    renderQueueLine();
    try { await Api.post("/train/stop"); toast("Stop requested — queue cleared."); }
    catch (err) { toast(err.message); }
  });
}

/* Compact status in the topbar so a running training stays visible on EVERY tab. */
function renderTrainChip(s) {
  const chip = $("#train-chip");
  chip.classList.remove("done", "failed");
  if (s.status === "running") {
    const eta = s.eta_seconds != null ? ` · ~${Math.round(s.eta_seconds)}s left` : "";
    chip.innerHTML = `<span class="mini-bar"><span style="width:${s.progress}%"></span></span>
      ${esc(s.model_name || "training")} ${s.progress}%${esc(eta)}`;
    chip.hidden = false;
  } else if (s.status === "completed" && s.model_name) {
    chip.classList.add("done");
    chip.textContent = `✓ ${s.model_name} done`;
    chip.hidden = false;
  } else if (s.status === "error") {
    chip.classList.add("failed");
    chip.textContent = `✗ training failed`;
    chip.hidden = false;
  } else {
    chip.hidden = true; // idle/stopped: no noise in the topbar
  }
}

async function loadTrainingTab() {
  const dsSel = $("#train-dataset");
  try {
    const [datasets, profiles] = await Promise.all([Api.get("/datasets"), Api.get("/train/profiles")]);
    dsSel.innerHTML = `<option value="">— choose —</option>` +
      datasets.map((d) => `<option>${esc(d.name)}</option>`).join("");
    $("#train-profile").innerHTML = profiles.profiles.map((p) =>
      `<option value="${esc(p.name)}" ${p.name === profiles.default_profile ? "selected" : ""}>${esc(p.name)} — ${esc(p.description)}</option>`).join("");
  } catch (err) { showError($("#train-error"), err); }
}

const textColPicker = createPillPicker({
  pills: document.querySelector("#textcol-pills"),
  input: document.querySelector("#textcol-input"),
  datalist: document.querySelector("#textcol-options"),
  emptyHint: "Select a dataset first.",
});
const labelPicker = createPillPicker({
  pills: document.querySelector("#labelcol-pills"),
  input: document.querySelector("#labelcol-input"),
  datalist: document.querySelector("#labelcol-options"),
  emptyHint: "Select a dataset first.",
  onChange: () => renderNamePreview(),
});

async function loadDatasetColumns() {
  const name = $("#train-dataset").value;
  if (!name) { textColPicker.setOptions([]); labelPicker.setOptions([]); return; }
  try {
    const info = await Api.get(`/datasets/${encodeURIComponent(name)}`);
    textColPicker.setOptions(info.columns);
    labelPicker.setOptions(info.columns);
  } catch (err) { showError($("#train-error"), err); }
}

/* One model per selected label field. A single field keeps the typed name; for
   several, the name becomes the base and each model gets a field suffix. */
function plannedModels() {
  const base = $("#train-name").value.trim();
  const fields = labelPicker.values();
  if (!base || !fields.length) return [];
  if (fields.length === 1) return [{ name: base, label_column: fields[0] }];
  const suffix = (f) => f.split(":").pop().replace(/[^A-Za-z0-9_-]+/g, "_");
  return fields.map((f) => ({ name: `${base}_${suffix(f)}`, label_column: f }));
}

function renderNamePreview() {
  const el = $("#train-names-preview");
  const plan = plannedModels();
  if (plan.length <= 1) { el.textContent = ""; return; }
  el.textContent = `Will create ${plan.length} models, trained one after another `
    + `(keep this tab open): ${plan.map((p) => p.name).join(", ")}`;
}

/* Client-side queue: the server trains ONE model at a time (single worker), so
   several selected label fields are trained sequentially from this tab. */
let trainQueue = [];

function renderQueueLine() {
  const el = $("#train-queue");
  el.hidden = !trainQueue.length;
  el.textContent = trainQueue.length
    ? `Queued next (keep this tab open): ${trainQueue.map((b) => b.model_name).join(", ")}`
    : "";
}

async function advanceQueue(status) {
  if (!trainQueue.length) return;
  if (!["completed", "error", "stopped"].includes(status)) return;
  const body = trainQueue.shift();
  renderQueueLine();
  try {
    await Api.post("/train", body);
    toast(`Training "${body.model_name}" started (${trainQueue.length} more queued).`);
  } catch (err) {
    toast(`Queue stopped: ${err.message}`);
    trainQueue = [];
    renderQueueLine();
  }
}

async function onTrainStart(ev) {
  ev.preventDefault();
  const errEl = $("#train-error"), btn = $("#train-btn");
  errEl.hidden = true;
  const textCols = textColPicker.values();
  if (!textCols.length) { showError(errEl, { message: "Pick at least one text column." }); return; }
  const plan = plannedModels();
  if (!plan.length) { showError(errEl, { message: "Enter a model name and pick at least one label field." }); return; }
  const shared = {
    dataset_name: $("#train-dataset").value,
    text_columns: textCols,
    optimize_parameters: $("#train-profile").value,
    label_filter: $("#train-filter").value.trim() || null,
  };
  if ($("#train-cv").value !== "") shared.cv_folds = Number($("#train-cv").value);
  const bodies = plan.map((p) => ({ ...shared, model_name: p.name, label_column: p.label_column }));
  btn.disabled = true;
  try {
    await Api.post("/train", bodies[0]);
    trainQueue = bodies.slice(1);
    renderQueueLine();
    toast(trainQueue.length
      ? `Training "${bodies[0].model_name}" started — ${trainQueue.length} more queued.`
      : `Training "${bodies[0].model_name}" started.`);
  } catch (err) { showError(errEl, err); }
  finally { btn.disabled = false; }
}

/* Models & datasets views live in manage.js (loaded before this file). */

boot();
