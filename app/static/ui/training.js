/* Configuring and starting a training run: the form, the column pickers, the planned
   model names, and the browser-side queue that runs several label fields one after
   another.

   The queue lives here rather than with the status view because it is "what I asked
   for", not "what is happening" — and it is the part a server-side queue would replace. */
"use strict";

async function loadTrainingTab() {
  const dsSel = $("#train-dataset");
  try {
    const [datasets, profiles] = await Promise.all([Api.get("/datasets"), Api.get("/train/profiles")]);
    dsSel.innerHTML = `<option value="">— choose —</option>` +
      datasets.map((d) => `<option>${esc(d.name)}</option>`).join("");
    $("#train-profile").innerHTML = profiles.profiles.map((p) =>
      `<option value="${esc(p.name)}" ${p.name === profiles.default_profile ? "selected" : ""}>${esc(p.name)} — ${esc(p.description)}</option>`).join("");
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

function textColumnWeights() {
  const out = {};
  document.querySelectorAll("#textcol-weights-fields [data-weight]").forEach((el) => {
    const n = Number(el.value);
    if (Number.isFinite(n) && n > 1) out[el.dataset.weight] = n;  // 1 = default, omit
  });
  return out;
}

function renderTextColWeights(cols) {
  const box = $("#textcol-weights");
  const fields = $("#textcol-weights-fields");
  const previous = textColumnWeights();
  box.hidden = !cols.length;
  fields.innerHTML = cols.map((c, i) => `
    <label for="weight-${i}">
      <span class="col-name">${esc(c)}</span>
      <input id="weight-${i}" type="number" min="1" max="10" step="1"
             value="${previous[c] || defaultColWeights[c] || 1}" data-weight="${esc(c)}"
             aria-describedby="textcol-weights-help">
    </label>`).join("");
}

const textColPicker = createPillPicker({
  pills: document.querySelector("#textcol-pills"),
  input: document.querySelector("#textcol-input"),
  datalist: document.querySelector("#textcol-options"),
  emptyHint: "Select a dataset first.",
  onChange: (cols) => renderTextColWeights(cols),
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
    + `on the server: ${plan.map((p) => p.name).join(", ")}`;
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
    for (const body of bodies) {
      const answer = await Api.post("/train", body);
      accepted.push(answer);
    }
    const queued = accepted.filter((a) => a.status === "queued").length;
    toast(queued
      ? `Training "${bodies[0].model_name}" started — ${queued} more queued on the server.`
      : `Training "${bodies[0].model_name}" started.`);
  } catch (err) {
    // Some may already be queued: say so rather than implying nothing happened.
    showError(errEl, { message: `${err.message} Runs accepted before this one are queued.` });
  } finally { btn.disabled = false; }
}


/* ---------- pre-flight ---------- */

/* On demand, not on every change of the label field: this parses the whole CSV, and a
   300 k export costs half a minute each time. The button says what it costs. */
async function runPreflight() {
  const box = $("#train-preflight-out"), button = $("#train-preflight");
  const dataset = $("#train-dataset").value;
  const textColumns = textColPicker.values();
  const labelFields = labelPicker.values();
  if (!dataset || !textColumns.length || !labelFields.length) {
    box.innerHTML = `<p class="error" role="alert">Pick a dataset, at least one text column
      and a label field first.</p>`;
    return;
  }
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Reading every row …";
  try {
    const body = await Api.post("/datasets/analyze", {
      dataset_name: dataset, text_columns: textColumns, label_column: labelFields[0],
      // The training form offers no separator field, so a run uses the request
      // default; the pre-flight has to read the file the same way or its numbers
      // describe a different parse than the one that will happen.
      label_filter: $("#train-filter").value.trim() || null,
    });
    box.innerHTML = preflightSummary(body, labelFields) + analysisHtml(body);
    box.querySelector("[data-use-threshold]")?.addEventListener("click", (ev) => {
      $("#train-minsamples").value = ev.target.dataset.useThreshold;
      ev.target.closest("p").textContent =
        `Threshold set to ${ev.target.dataset.useThreshold}.`;
    });
  } catch (err) {
    box.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function preflightSummary(body, labelFields) {
  const profile = $("#train-profile").value;
  const minutes = (body.estimated_minutes || {})[profile];
  const current = Number($("#train-minsamples").value);
  const kept = body.label_threshold_analysis[`labels_with_${current}+_samples`];
  const recommended = body.recommended_min_samples_per_label;
  const cost = minutes == null ? "no estimate for this profile"
    : minutes < 1 ? "under a minute"
    : minutes < 90 ? `about ${minutes.toFixed(0)} minutes`
    : `about ${(minutes / 60).toFixed(1)} hours`;
  return `<p><strong>${fmtInt(body.total_samples)} rows, ${body.unique_labels} labels.</strong>
      On <code>${esc(profile)}</code> that is ${esc(cost)}${labelFields.length > 1
        ? ` — per model, and you planned ${labelFields.length}` : ""}.</p>
    <p>Your threshold of ${current} keeps ${kept === undefined
      ? "an unknown number of"
      : `<strong>${kept}</strong> of ${body.unique_labels}`} labels.${current === recommended ? ""
      : ` The size heuristic would pick ${recommended}.
          <button type="button" class="small" data-use-threshold="${recommended}">Use ${recommended}</button>`}</p>
    <p class="muted">Checked against <code>${esc(labelFields[0])}</code>${labelFields.length > 1
      ? " — the first of your label fields; the others may differ" : ""}.</p>`;
}
