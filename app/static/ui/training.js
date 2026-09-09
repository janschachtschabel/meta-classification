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
    dsSel.innerHTML = `<option value="">${esc(t("train.dataset.choose"))}</option>` +
      datasets.map((d) => `<option>${esc(d.name)}</option>`).join("");
    // The profile descriptions are server configuration (config.yaml), not UI text:
    // they are shown as the deployment wrote them rather than translated here.
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
  el.textContent = t("train.namePreview",
                     { count: plan.length, names: plan.map((p) => p.name).join(", ") });
}

async function onTrainStart(ev) {
  ev.preventDefault();
  const errEl = $("#train-error"), btn = $("#train-btn");
  errEl.hidden = true;
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
      ? t("train.startedQueued", { name: bodies[0].model_name, count: queued })
      : t("train.started", { name: bodies[0].model_name }));
  } catch (err) {
    // Some may already be queued: say so rather than implying nothing happened.
    showError(errEl, { message: t("train.partialFailure", { message: err.message }) });
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
    box.innerHTML = `<p class="error" role="alert">${t("train.error.preflightInputs")}</p>`;
    return;
  }
  button.disabled = true;
  button.textContent = t("common.readingEveryRow");
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
        t("train.preflight.thresholdSet", { value: Number(ev.target.dataset.useThreshold) });
    });
  } catch (err) {
    box.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = t("train.preflight.button");   // see explain.js: not a copy
  }
}

function preflightSummary(body, labelFields) {
  const profile = $("#train-profile").value;
  const minutes = (body.estimated_minutes || {})[profile];
  const current = Number($("#train-minsamples").value);
  const kept = body.label_threshold_analysis[`labels_with_${current}+_samples`];
  const recommended = body.recommended_min_samples_per_label;
  // One sentence per case rather than one sentence with a swapped fragment: the
  // hedge belongs to a number, and "that is about no estimate for this profile"
  // was what folding it into the wrapper produced.
  const cost = !Number.isFinite(minutes)
    ? t("train.preflight.noEstimateFor", { profile: esc(profile) })
    : minutes < 1
      ? t("train.preflight.underAMinuteOn", { profile: esc(profile) })
      : t("train.preflight.onProfile", { profile: esc(profile), cost: costLabel(minutes) });
  return `<p><strong>${t("train.preflight.size", {
      rows: body.total_samples, labels: body.unique_labels })}</strong>
      ${cost}${labelFields.length > 1
        ? t("train.preflight.perModel", { count: labelFields.length }) : ""}.</p>
    <p>${kept === undefined
      ? t("train.preflight.keepsUnknown", { threshold: current })
      : t("train.preflight.keeps", { threshold: current, kept, total: body.unique_labels })}${
      current === recommended ? ""
      : ` ${t("train.preflight.heuristicPicks", { recommended })}
          <button type="button" class="small" data-use-threshold="${recommended}">${
            t("train.preflight.useValue", { value: recommended })}</button>`}</p>
    <p class="muted">${t("train.preflight.checkedAgainst", { field: esc(labelFields[0]) })}${
      labelFields.length > 1 ? t("train.preflight.firstFieldOnly") : ""}.</p>`;
}
