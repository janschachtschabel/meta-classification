/* The pre-flight: what a run on the chosen dataset and columns would keep and what it
   would cost, asked for before starting one. Split from training.js, which configures and
   starts runs: this half changes when the analysis reports something new, that one when a
   run's levers do. */
"use strict";

/* On demand, not on every change of the label field: this parses the whole CSV, and a
   300 k export costs half a minute each time. The button says what it costs. */
async function runPreflight() {
  const box = $("#train-preflight-out"), button = $("#train-preflight");
  const dataset = $("#train-dataset").value;
  const textColumns = textColPicker.values();
  const labelFields = labelPicker.values();
  if (!dataset || !textColumns.length || !labelFields.length) {
    // role="alert" announces it; the summary element stays out of the way so the two
    // do not say the same thing twice.
    box.innerHTML = `<p class="error" role="alert">${t("train.error.preflightInputs")}</p>`;
    $("#train-preflight-announce").textContent = "";
    return;
  }
  const idle = busy(button);
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
    // The first paragraph is the headline — rows, labels and what the run will cost.
    // Read out of the rendered block rather than built a second time, so what is
    // announced is what is shown. The tables below it are for reading, not hearing.
    $("#train-preflight-announce").textContent =
      (box.querySelector("p")?.textContent || "").replace(/\s+/g, " ").trim();
    box.querySelector("[data-use-threshold]")?.addEventListener("click", (ev) => {
      $("#train-minsamples").value = ev.target.dataset.useThreshold;
      ev.target.closest("p").textContent =
        t("train.preflight.thresholdSet", { value: Number(ev.target.dataset.useThreshold) });
    });
  } catch (err) {
    box.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`;
    $("#train-preflight-announce").textContent = "";
  } finally {
    idle();
    button.textContent = t("train.preflight.button");   // see explain.js: not a copy
  }
}

/* German writes no second period after an abbreviation that ends a sentence, and the
   duration labels are abbreviations ("12 Min.", "1,5 Std.") — appending one printed
   "12 Min..". The labels keep their period: the cost table shows them on their own. */
const endSentence = (text) => (text.endsWith(".") ? text : `${text}.`);

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
  // Said only when the memory budget holds the run back: that is what makes a long
  // estimate long, and a full thread count is not news.
  const planned = (body.planned_head_fit_threads || {})[profile];
  const held = Number.isFinite(minutes) && planned < body.threads_requested
    ? t("train.preflight.threadsLimited", { used: planned, requested: body.threads_requested })
    : "";
  const perModel = labelFields.length > 1
    ? t("train.preflight.perModel", { count: labelFields.length }) : "";
  return `<p><strong>${t("train.preflight.size", {
      rows: body.total_samples, labels: body.unique_labels })}</strong>
      ${endSentence(`${cost}${held}${perModel}`)}</p>
    <p>${kept === undefined
      ? t("train.preflight.keepsUnknown", { threshold: current })
      : t("train.preflight.keeps", { threshold: current, kept, total: body.unique_labels })}${
      current === recommended ? ""
      : ` ${t("train.preflight.heuristicPicks", { recommended })}
          <button type="button" class="small" data-use-threshold="${recommended}">${
            t("train.preflight.useValue", { value: recommended })}</button>`}</p>
    <p class="muted">${endSentence(
      t("train.preflight.checkedAgainst", { field: esc(labelFields[0]) })
      + (labelFields.length > 1 ? t("train.preflight.firstFieldOnly") : ""))}</p>`;
}
