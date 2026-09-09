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
      <label for="ds-text-cols">Text columns (Ctrl-click for several)</label>
      <select id="ds-text-cols" multiple size="5">${options(null)}</select>
    </div>
    <div>
      <label for="ds-label-col">Label column</label>
      <select id="ds-label-col">${options(null)}</select>
    </div>
  </div>
  <button type="button" class="small" id="ds-analyze">Analyze</button>
  <p class="muted">Reads every row, so this takes a moment on a large export.</p>`;
}

function thresholdRows(analysis, recommended) {
  return Object.entries(analysis)
    .map(([key, kept]) => ({ n: Number(key.match(/\d+/)[0]), kept }))
    .sort((a, b) => a.n - b.n)
    .map(({ n, kept }) => `<tr${n === recommended ? ' class="recommended"' : ""}>
      <td class="num">${n}</td><td class="num">${kept}</td>
      <td>${n === recommended ? "recommended for this size" : ""}</td></tr>`)
    .join("");
}

function costRows(estimate) {
  return Object.entries(estimate).map(([profile, minutes]) => `<tr>
    <td>${esc(profile)}</td>
    <td class="num">${minutes < 1 ? "under a minute"
      : minutes < 90 ? `${minutes.toFixed(0)} min`
      : `${(minutes / 60).toFixed(1)} h`}</td></tr>`).join("");
}

function analysisHtml(body) {
  const recommended = body.recommended_min_samples_per_label;
  const rare = Object.keys(body.rare_labels_under_10 || {}).length;
  return `<div class="analysis">
    <h3>${fmtInt(body.total_samples)} rows with labels · ${body.unique_labels} labels</h3>
    ${rare ? `<p class="error" role="alert">${rare} of them have fewer than 10 examples.
       A label the model barely saw will score badly however good the run is.</p>` : ""}
    <h4>How many labels survive a threshold</h4>
    <div class="table-wrap"><table>
      <thead><tr><th class="num">min. examples</th><th class="num">labels kept</th><th></th></tr></thead>
      <tbody>${thresholdRows(body.label_threshold_analysis, recommended)}</tbody>
    </table></div>
    <p class="muted">Dropping a label is a decision, not a detail: everything below the
       value you pick is removed from training and can never be predicted. The marked row
       is what the size heuristic would choose — it is a starting point, not an answer.</p>
    <h4>What a run would cost</h4>
    <div class="table-wrap"><table>
      <thead><tr><th>Profile</th><th class="num">estimate</th></tr></thead>
      <tbody>${costRows(body.estimated_minutes || {})}</tbody>
    </table></div>
    <p class="muted"><strong>Estimates.</strong> Scaled from one measured run (156 373 rows
       on <code>auto</code> in 40 min) and linear in the row count, which is what the row
       scaling benchmark found. It cannot see your label count or your machine, so treat
       it as "coffee break or afternoon", not as a schedule.</p>
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
      `<div class="analysis"><p class="error" role="alert">Pick at least one text column.</p></div>`);
    return;
  }
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Reading every row …";
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
    button.textContent = original;
  }
}

async function showDatasetDetail(name) {
  const dialog = $("#dataset-detail");
  dialog.innerHTML = `<div class="detail" tabindex="-1"><p class="muted">Loading …</p></div>`;
  dialog.showModal();
  const frame = dialog.querySelector(".detail");
  frame.focus();

  let info;
  try {
    info = await Api.get(`/datasets/${encodeURIComponent(name)}`);
  } catch (err) {
    frame.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>
      <button type="button" class="ghost" data-close>Close</button>`;
    frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
    return;
  }
  analyzeState.columns = info.columns || [];
  analyzeState.name = name;

  const sample = info.sample || [];
  frame.innerHTML = `
    <div class="detail-head">
      <h2 id="dataset-detail-title">${esc(name)}</h2>
      <button type="button" class="ghost small" data-close aria-label="Close details">Close</button>
    </div>
    <p class="muted">${analyzeState.columns.length} columns · first ${sample.length} rows shown</p>
    <div class="table-wrap"><table>
      <thead><tr>${analyzeState.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${sample.map((row) => `<tr>${analyzeState.columns.map(
        (c) => `<td>${esc(String(row[c] ?? "")).slice(0, 120)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>

    <h3>Before training on it</h3>
    ${columnPickers(analyzeState.columns)}`;
  frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
  frame.querySelector("#ds-analyze").addEventListener("click", () => runAnalysis(frame));
  frame.focus();
}

document.addEventListener("DOMContentLoaded", () => {
  const dialog = $("#dataset-detail");
  dialog.addEventListener("click", (ev) => { if (ev.target === dialog) dialog.close(); });
});
