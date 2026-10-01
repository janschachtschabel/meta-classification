/* Configuring and starting a training run: the form, the column pickers, the planned
   model names, and sending one run per label field. The queue those runs wait in lives on
   the server (train-status.js shows it); what a run would cost is preflight.js. */
"use strict";

async function loadTrainingTab() {
  const dsSel = $("#train-dataset"), profileSel = $("#train-profile");
  // Both lists are rebuilt on every visit to the tab, which reset the dataset and the profile
  // whenever someone looked at another tab (audit 2026-09-30, U08). A choice stays while it
  // still exists.
  const dataset = dsSel.value, profile = profileSel.value;
  try {
    const [datasets, profiles] = await Promise.all([Api.get("/datasets"), Api.get("/train/profiles")]);
    dsSel.innerHTML = `<option value="">${esc(t("train.dataset.choose"))}</option>` +
      datasets.map((d) => `<option value="${esc(d.name)}">${esc(d.name)}</option>`).join("");
    // The profile descriptions are server configuration (config.yaml), not UI text:
    // they are shown as the deployment wrote them rather than translated here.
    profileSel.innerHTML = profiles.profiles.map((p) =>
      `<option value="${esc(p.name)}" ${p.name === profiles.default_profile ? "selected" : ""}>${esc(p.name)} — ${esc(p.description)}</option>`).join("");
    if (profiles.profiles.some((p) => p.name === profile)) profileSel.value = profile;
    if (datasets.some((d) => d.name === dataset)) dsSel.value = dataset;
    // Deleted meanwhile: the pickers must not go on offering the columns of a dataset the
    // list no longer shows as chosen.
    else if (dataset) loadDatasetColumns();
    // Pre-fill the field weights with the server's configured default instead of a
    // hard-coded guess, so the form shows what a request would actually do.
    defaultColWeights = profiles.default_text_column_weights || {};
  } catch (err) { showError($("#train-error"), err); }
}

/* Field weights: one multiplier input per SELECTED text column, rebuilt whenever the
   pill selection changes. Values already typed survive the rebuild — removing one
   column must not silently reset the others. A column the user has not touched shows
   the server's configured default (title/keywords at 2x), so the form and the API
   agree on what happens. */
let defaultColWeights = {};

/* Every weight on the form, the 1s included: a 1 is a choice too. The rebuild used to read
   only the values above 1 and filled the gap with the server's default, so a deliberate 1 on
   "title" (default 2) jumped back to 2 the moment another column was picked, and the run
   trained with 2 (audit 2026-09-30, U04). */
function typedColumnWeights() {
  const out = {};
  document.querySelectorAll("#textcol-weights-fields [data-weight]").forEach((el) => {
    const n = Number(el.value);
    if (Number.isFinite(n) && n >= 1) out[el.dataset.weight] = n;
  });
  return out;
}

/* What a request sends. A dict sent at all is taken as it is, so a column left out of it
   weighs 1 -- which is how a 1 is said. */
function textColumnWeights() {
  return Object.fromEntries(Object.entries(typedColumnWeights()).filter(([, n]) => n > 1));
}

function renderTextColWeights(cols) {
  const box = $("#textcol-weights");
  const fields = $("#textcol-weights-fields");
  const previous = typedColumnWeights();
  box.hidden = !cols.length;
  fields.innerHTML = cols.map((c, i) => `
    <label for="weight-${i}">
      <span class="col-name">${esc(c)}</span>
      <input id="weight-${i}" type="number" min="1" max="10" step="1"
             value="${previous[c] ?? defaultColWeights[c] ?? 1}" data-weight="${esc(c)}"
             aria-describedby="textcol-weights-help">
    </label>`).join("");
}

const textColPicker = createPillPicker({
  pills: document.querySelector("#textcol-pills"),
  input: document.querySelector("#textcol-input"),
  datalist: document.querySelector("#textcol-options"),
  emptyHintKey: "train.pills.emptyHint",
  onChange: (cols) => renderTextColWeights(cols),
});
const labelPicker = createPillPicker({
  pills: document.querySelector("#labelcol-pills"),
  input: document.querySelector("#labelcol-input"),
  datalist: document.querySelector("#labelcol-options"),
  emptyHintKey: "train.pills.emptyHint",
  onChange: () => renderNamePreview(),
});

/* `change` fires per Arrow keypress on the <select>, and the route behind this reads the
   whole CSV: unguarded, holding Down walked the list and issued one full read per step, and
   a slow earlier answer could land after a newer one — leaving the pickers offering columns
   the selected dataset does not have. `isCurrent()` is how this function declines to apply
   a stale answer; `boot()` wraps it in `latestOnly` (app.js) where the listener is bound.

   Declared as a function rather than `const … = latestOnly(…)` for a load-order reason:
   this module runs before app.js, so calling one of its helpers out here is a
   ReferenceError that aborts the whole file — the same trap escape.js explains. */
async function loadDatasetColumns(isCurrent = () => true) {
  const name = $("#train-dataset").value;
  if (!name) {
    textColPicker.setOptions([]); labelPicker.setOptions([]); syncSyntheticChoice([]);
    return;
  }
  try {
    const info = await Api.get(`/datasets/${encodeURIComponent(name)}`);
    if (!isCurrent()) return;
    textColPicker.setOptions(info.columns);
    labelPicker.setOptions(info.columns);
    syncSyntheticChoice(info.columns);
  } catch (err) { if (isCurrent()) showError($("#train-error"), err); }
}

/* data-prep marks the rows an LLM wrote or touched. Only a dataset carrying such a
   column gets the choice — one without trains exactly as before, so the control stays
   out of the way. "Leave out" needs generated rows to leave out; without the
   generated_for column it is disabled rather than offered as a silent no-op. */
// The same names as app/provenance.py MARK_COLUMNS -- data-prep's contract.
const MARK_COLUMNS = ["generated_for", "example_for", "enriched_fields"];

function syncSyntheticChoice(columns) {
  const select = $("#train-synthetic-rows");
  const exclude = select.querySelector('option[value="exclude"]');
  $("#train-synthetic").hidden = !MARK_COLUMNS.some((c) => columns.includes(c));
  exclude.disabled = !columns.includes("generated_for");
  if (exclude.disabled) select.value = "train";
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
  el.textContent = t("train.namePreview",
                     { count: plan.length, names: plan.map((p) => p.name).join(", ") });
}

/* /train is rate-limited (5 a minute by default) below the number of label fields a batch may
   hold: seven fields ended in a 429 at the sixth, a retry hit the limit again, and later
   "already exists" for the runs the first attempt had queued (audit 2026-09-30, U05). A 429
   now waits the window the server names and sends the same run again -- a bounded number of
   times, because another client on the same address can keep the window full. */
const TRAIN_LIMIT_WAITS = 3;

async function submitRun(body, run, total) {
  for (let waits = 0; ; waits += 1) {
    try {
      return await Api.post("/train", body);
    } catch (err) {
      if (err.status !== 429 || waits === TRAIN_LIMIT_WAITS) throw err;
      const seconds = err.retryAfter ?? 60;
      toast(t("train.waitingForLimit", { run, total, wait: t("common.seconds", { count: seconds }) }));
      await new Promise((resolve) => setTimeout(resolve, seconds * 1000));
    }
  }
}

async function onTrainStart(ev) {
  ev.preventDefault();
  const errEl = $("#train-error"), btn = $("#train-btn");
  errEl.hidden = true;
  // Checked first and by itself: with no dataset the column pickers are empty too, so the
  // "no text column" message below would fire and name the wrong thing to fix. The form
  // carries `novalidate`, so this is the only check there is.
  if (!$("#train-dataset").value) {
    showError(errEl, { message: t("train.error.noDataset") });
    return;
  }
  const textCols = textColPicker.values();
  if (!textCols.length) { showError(errEl, { message: t("common.error.noTextColumn") }); return; }
  const plan = plannedModels();
  if (!plan.length) { showError(errEl, { message: t("train.error.noPlan") }); return; }
  const shared = {
    dataset_name: $("#train-dataset").value,
    text_columns: textCols,
    optimize_parameters: $("#train-profile").value,
    label_filter: $("#train-filter").value.trim() || null,
  };
  if ($("#train-cv").value !== "") shared.cv_folds = Number($("#train-cv").value);
  // Hidden = the dataset carries no marks, and a choice left over from another dataset
  // must not travel with this one.
  if (!$("#train-synthetic").hidden) {
    shared.synthetic_rows = $("#train-synthetic-rows").value;
    shared.thin_label_threshold = $("#train-thin-threshold").value;
  }
  // Empty field = omit, so the request default (20) applies rather than a silent 0.
  const minSamples = $("#train-minsamples").value.trim();
  if (minSamples !== "") shared.min_samples_per_label = Number(minSamples);
  // ALWAYS send it once columns are picked: omitting the field means "apply the
  // config default", which would silently override a user who set every field to 1.
  // The form is what the user sees, so the form has to be authoritative.
  if (textCols.length) shared.text_column_weights = textColumnWeights();
  // Blank = omit, so the profile's cap applies rather than a coerced 0.
  for (const [id, field] of [["#train-maxword", "max_word_features"],
                             ["#train-maxchar", "max_char_features"]]) {
    const raw = $(id).value.trim();
    if (raw !== "") shared[field] = Number(raw);
  }
  const bodies = plan.map((p) => ({ ...shared, model_name: p.name, label_column: p.label_column }));
  btn.disabled = true;
  try {
    // Every run is submitted right away and the SERVER holds the order. This page used
    // to keep the rest in an array and post them as the status changed, which meant a
    // closed tab lost them; sending them now is what makes the tab disposable.
    // Sequentially, because a position is only meaningful against a known queue.
    const accepted = [];
    for (const [at, body] of bodies.entries()) {
      accepted.push(await submitRun(body, at + 1, bodies.length));
    }
    const queued = accepted.filter((a) => a.status === "queued").length;
    toast(queued
      ? t("train.startedQueued", { name: bodies[0].model_name, count: queued })
      : t("train.started", { name: bodies[0].model_name }));
  } catch (err) {
    // Some may already be queued: say so rather than implying nothing happened.
    showError(errEl, { message: t("train.partialFailure", { message: err.message }) });
  } finally { btn.disabled = false; }
}
