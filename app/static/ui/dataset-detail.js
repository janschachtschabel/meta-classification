/* What is in this CSV, and what would training on it cost — the panel behind a dataset
   name in the Datasets tab.

   Same shape as the model detail: a native <dialog>, so the focus trap, Esc and the
   return of focus are the platform's. Its own file for the same reason too — listing
   datasets and inspecting one change for different reasons.

   Analysis parses the whole CSV, which is a 195 MB job on a real export, so it happens
   on demand rather than on open, and once per click. */
"use strict";

const analyzeState = { columns: [], name: "" };

function columnPickers(columns) {
  const options = (selected) => columns.map((c) =>
    `<option${selected === c ? " selected" : ""}>${esc(c)}</option>`).join("");
  return `<div class="row">
    <div>
      <label for="ds-text-cols">${t("common.textColumnsMulti")}</label>
      <select id="ds-text-cols" multiple size="5">${options(null)}</select>
    </div>
    <div>
      <label for="ds-label-col">${t("common.labelColumn")}</label>
      <select id="ds-label-col">${options(null)}</select>
    </div>
  </div>
  <button type="button" class="small" id="ds-analyze">${t("datasetDetail.analyze")}</button>
  <p class="muted">${t("datasetDetail.analyzeNote")}</p>`;
}

function thresholdRows(analysis, recommended) {
  return Object.entries(analysis)
    .map(([key, kept]) => ({ n: Number(key.match(/\d+/)[0]), kept }))
    .sort((a, b) => a.n - b.n)
    .map(({ n, kept }) => `<tr${n === recommended ? ' class="recommended"' : ""}>
      <td class="num">${n}</td><td class="num">${kept}</td>
      <td>${n === recommended ? t("datasetDetail.recommended") : ""}</td></tr>`)
    .join("");
}

/* "Coffee break or afternoon", in the reader's units — the same three bands the
   pre-flight uses, so the two views never disagree about what a run costs. */
function costLabel(minutes) {
  if (minutes < 1) return t("common.underAMinute");
  if (minutes < 90) return t("common.minutes", { count: Number(minutes.toFixed(0)) });
  return t("common.hours", { count: Number((minutes / 60).toFixed(1)) });
}

function costRows(body) {
  const threads = body.planned_head_fit_threads || {};
  return Object.entries(body.estimated_minutes || {}).map(([profile, minutes]) => `<tr>
    <td>${esc(profile)}</td>
    <td class="num">${costLabel(minutes)}</td>
    <td class="num">${threadsCell(threads[profile], body.threads_requested)}</td></tr>`).join("");
}

/* The head-fit threads an estimate assumed. Fewer than the CPU budget allows is a run
   the memory budget will slow down — the reason a long estimate is long. */
function threadsCell(used, requested) {
  if (used == null || requested == null) return "–";
  return used < requested
    ? t("datasetDetail.threadsLimited", { used, requested })
    : t("common.threadsOf", { used, requested });
}

function analysisHtml(body) {
  const recommended = body.recommended_min_samples_per_label;
  const rare = Object.keys(body.rare_labels_under_10 || {}).length;
  return `<div class="analysis">
    <h3>${t("datasetDetail.summary", { rows: body.total_samples, labels: body.unique_labels })}</h3>
    ${rare ? `<p class="error" role="alert">${t("datasetDetail.rareWarning", { count: rare })}</p>` : ""}
    <h4>${t("datasetDetail.thresholdHeading")}</h4>
    <div class="table-wrap"><table>
      <thead><tr><th class="num">${t("datasetDetail.table.minExamples")}</th>
        <th class="num">${t("datasetDetail.table.labelsKept")}</th><th></th></tr></thead>
      <tbody>${thresholdRows(body.label_threshold_analysis, recommended)}</tbody>
    </table></div>
    <p class="muted">${t("datasetDetail.thresholdNote")}</p>
    <h4>${t("datasetDetail.costHeading")}</h4>
    <div class="table-wrap"><table>
      <thead><tr><th>${t("datasetDetail.table.profile")}</th>
        <th class="num">${t("datasetDetail.table.estimate")}</th>
        <th class="num">${t("datasetDetail.table.threads")}</th></tr></thead>
      <tbody>${costRows(body)}</tbody>
    </table></div>
    <p class="muted">${t("datasetDetail.costNote")}</p>
  </div>`;
}

async function runAnalysis(frame) {
  const textColumns = [...frame.querySelectorAll("#ds-text-cols option")]
    .filter((o) => o.selected).map((o) => o.value);
  const labelColumn = frame.querySelector("#ds-label-col").value;
  const button = frame.querySelector("#ds-analyze");
  frame.querySelector(".analysis")?.remove();
  if (!textColumns.length) {
    frame.insertAdjacentHTML("beforeend",
      `<div class="analysis"><p class="error" role="alert">${t("common.error.noTextColumn")}</p></div>`);
    return;
  }
  button.disabled = true;
  button.textContent = t("common.readingEveryRow");
  try {
    const body = await Api.post("/datasets/analyze", {
      dataset_name: analyzeState.name, text_columns: textColumns, label_column: labelColumn,
    });
    frame.insertAdjacentHTML("beforeend", analysisHtml(body));
  } catch (err) {
    frame.insertAdjacentHTML("beforeend",
      `<div class="analysis"><p class="error" role="alert">${esc(err.message)}</p></div>`);
  } finally {
    button.disabled = false;
    button.textContent = t("datasetDetail.analyze");   // see explain.js: not a copy
  }
}

async function showDatasetDetail(name) {
  const dialog = $("#dataset-detail");
  dialog.innerHTML = `<div class="detail" tabindex="-1"><p class="muted">${t("common.loading")}</p></div>`;
  dialog.showModal();
  const frame = dialog.querySelector(".detail");
  frame.focus();

  let info;
  try {
    info = await Api.get(`/datasets/${encodeURIComponent(name)}`);
  } catch (err) {
    frame.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>
      <button type="button" class="ghost" data-close>${t("common.close")}</button>`;
    frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
    return;
  }
  analyzeState.columns = info.columns || [];
  analyzeState.name = name;

  const sample = info.sample || [];
  frame.innerHTML = `
    <div class="detail-head">
      <h2 id="dataset-detail-title">${esc(name)}</h2>
      <button type="button" class="ghost small" data-close
              aria-label="${esc(t("common.closeDetails"))}">${t("common.close")}</button>
    </div>
    <p class="muted">${t("datasetDetail.subtitle", {
      columns: analyzeState.columns.length, rows: sample.length })}</p>
    <div class="table-wrap"><table>
      <thead><tr>${analyzeState.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${sample.map((row) => `<tr>${analyzeState.columns.map(
        (c) => `<td>${esc(String(row[c] ?? "")).slice(0, 120)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>

    <h3>${t("datasetDetail.beforeTraining")}</h3>
    ${columnPickers(analyzeState.columns)}`;
  frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
  frame.querySelector("#ds-analyze").addEventListener("click", () => runAnalysis(frame));
  frame.focus();
}

document.addEventListener("DOMContentLoaded", () => {
  const dialog = $("#dataset-detail");
  dialog.addEventListener("click", (ev) => { if (ev.target === dialog) dialog.close(); });
});
