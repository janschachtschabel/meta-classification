/* What the finished runs left behind — trainings and evaluations alike: how each ended, its
   headline scores, how long it took, and for a failed run the reason, which no bundle keeps
   (GET /train/history). The API recorded all of it and the UI never read it, so two models
   on one dataset could only be compared by opening each (audit 2026-09-30, improvement 8).

   Its own file: what HAPPENED changes for different reasons than what is happening
   (train-status.js) or what is being asked for (training.js). */
"use strict";

const HISTORY_SHOWN = 20;
// Spelled out rather than built from the value: a key that exists only as a template is
// invisible to the test that proves every string is translated.
const KIND_KEYS = { training: "trainHistory.kind.training", evaluation: "trainHistory.kind.evaluation" };

function historyRow(run) {
  const reason = run.status === "error" && run.error
    ? `<br><span class="muted">${esc(String(run.error).slice(0, 200))}</span>` : "";
  const dataset = (run.request && run.request.dataset_name) || "–";
  return `<tr>
    <td>${esc(run.model_name || "–")}</td>
    <td>${t(KIND_KEYS[run.kind] || KIND_KEYS.training)}</td>
    <td>${esc(dataset)}</td>
    <td>${esc(stateLabel(run.status))}${reason}</td>
    <td class="num">${fmtScore(run.f1_macro)}</td>
    <td class="num">${fmtScore(run.f1_micro)}</td>
    <td class="num">${run.duration_seconds != null ? esc(costLabel(run.duration_seconds / 60)) : "–"}</td>
    <td>${esc(fmtDate(run.finished_at))}</td>
  </tr>`;
}

/* Read when the Training tab opens and whenever a run ends (train-status.js). */
async function loadTrainHistory() {
  const box = $("#train-history");
  let runs;
  try {
    runs = await Api.get(`/train/history?limit=${HISTORY_SHOWN}`);
  } catch (err) {
    box.innerHTML = `<p class="error" role="alert">${esc(err.message)}</p>`;
    return;
  }
  if (!runs.length) {
    box.innerHTML = `<p class="muted">${t("trainHistory.empty")}</p>`;
    return;
  }
  box.innerHTML = `<div class="table-wrap" tabindex="0"><table>
    <thead><tr><th>${t("trainHistory.col.model")}</th><th>${t("trainHistory.col.kind")}</th>
      <th>${t("trainHistory.col.dataset")}</th><th>${t("trainHistory.col.outcome")}</th>
      <th class="num">${t("trainHistory.col.f1Macro")}</th><th class="num">${t("trainHistory.col.f1Micro")}</th>
      <th class="num">${t("trainHistory.col.duration")}</th><th>${t("trainHistory.col.finished")}</th></tr></thead>
    <tbody>${runs.map(historyRow).join("")}</tbody></table></div>`;
}
