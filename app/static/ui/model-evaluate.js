/* Starting an evaluation from the model panel, and reading the ones already recorded.

   Its own file: showing what a model IS and asking it to be scored against a dataset
   are different jobs, and the second one is a form with its own state.

   The numbers it renders need their caveats next to them — a macro F1 averaged over a
   label space the dataset barely touches is a true number that reads as a false one. */
"use strict";

function evaluationRows(runs) {
  return runs.map((run) => {
    const m = run.metrics;
    return `<tr>
      <td>${esc(run.dataset)}</td>
      <td class="num">${fmtInt(run.n_rows)}</td>
      <td class="num">${run.labels_covered == null ? "–" : fmtInt(run.labels_covered)}</td>
      <td class="num">${m ? fmtScore(m.f1_macro) : "–"}</td>
      <td class="num">${m ? fmtScore(m.f1_micro) : "–"}</td>
      <td>${esc(fmtDate(run.evaluated_at))}</td>
    </tr>`;
  }).join("");
}

function evaluationsSection(model) {
  const runs = (model.metadata && model.metadata.evaluations) || [];
  if (!runs.length) {
    return `<p class="muted">Not scored against any dataset yet. Comparing two models is
      only a statement if both were measured on the same rows — that is what this is for.</p>`;
  }
  const uncovered = runs.filter((r) => r.rows_without_a_known_label > 0);
  return `<div class="table-wrap"><table>
      <thead><tr><th>Dataset</th><th class="num">Rows</th><th class="num">Labels hit</th>
        <th class="num">F1 macro</th><th class="num">F1 micro</th><th>When</th></tr></thead>
      <tbody>${evaluationRows(runs)}</tbody>
    </table></div>
    <p class="muted"><strong>Labels hit</strong> is how many of this model's labels the
       dataset actually exercises. <strong>F1 macro averages over all of them</strong>, so a
       dataset touching a handful of a large label space drags it down for reasons that have
       nothing to do with quality — read it against the same dataset on another model, never
       across datasets.${uncovered.length ? ` ${plural(uncovered.length, "run")} had rows whose
       labels this model never learned; those rows were excluded, not counted as failures.`
    : ""}</p>`;
}

/* ---------- starting one ---------- */

async function openEvaluateForm(name, frame) {
  frame.querySelector(".evaluate-form")?.remove();
  let datasets;
  try {
    datasets = await Api.get("/datasets");
  } catch (err) {
    frame.insertAdjacentHTML("beforeend",
      `<div class="evaluate-form"><p class="error" role="alert">${esc(err.message)}</p></div>`);
    return;
  }
  if (!datasets.length) {
    frame.insertAdjacentHTML("beforeend", `<div class="evaluate-form"><p class="muted">
      No datasets to score against — upload one on the Datasets tab first.</p></div>`);
    return;
  }
  frame.insertAdjacentHTML("beforeend", `<div class="evaluate-form">
    <label for="ev-dataset">Score against</label>
    <select id="ev-dataset">${datasets.map((d) =>
      `<option>${esc(d.name)}</option>`).join("")}</select>
    <div class="row">
      <div><label for="ev-text-cols">Text columns (Ctrl-click for several)</label>
        <select id="ev-text-cols" multiple size="5"></select></div>
      <div><label for="ev-label-col">Label column</label>
        <select id="ev-label-col"></select></div>
    </div>
    <button type="button" class="small" id="ev-start">Start evaluation</button>
    <p class="muted">Runs on the server as a background job, behind any training already
       going. Assemble the text the way the model was trained — the panel above lists its
       columns and their weights.</p>
    <p id="ev-message" role="alert"></p>
  </div>`);
  const box = frame.querySelector(".evaluate-form");
  const loadColumns = async () => {
    const info = await Api.get(`/datasets/${encodeURIComponent(box.querySelector("#ev-dataset").value)}`);
    const options = (info.columns || []).map((c) => `<option>${esc(c)}</option>`).join("");
    box.querySelector("#ev-text-cols").innerHTML = options;
    box.querySelector("#ev-label-col").innerHTML = options;
  };
  box.querySelector("#ev-dataset").addEventListener("change", loadColumns);
  await loadColumns();
  box.querySelector("#ev-start").addEventListener("click", () => startEvaluation(name, box));
}

async function startEvaluation(name, box) {
  const textColumns = [...box.querySelectorAll("#ev-text-cols option")]
    .filter((o) => o.selected).map((o) => o.value);
  const message = box.querySelector("#ev-message");
  if (!textColumns.length) {
    message.className = "error";
    message.textContent = "Pick at least one text column.";
    return;
  }
  const button = box.querySelector("#ev-start");
  button.disabled = true;
  try {
    const answer = await Api.post(`/models/${encodeURIComponent(name)}/evaluate`, {
      dataset_name: box.querySelector("#ev-dataset").value,
      text_columns: textColumns,
      label_column: box.querySelector("#ev-label-col").value,
    });
    message.className = "ok";
    message.textContent = answer.status === "queued"
      ? `Queued on the server, ${plural(answer.queue_position, "run")} ahead of it. Watch it on the Training tab.`
      : "Started. Watch it on the Training tab; the result appears here when it is done.";
  } catch (err) {
    message.className = "error";
    message.textContent = err.message;
  } finally { button.disabled = false; }
}
