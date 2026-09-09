/* Everything one model records about itself: how it was trained, how well it scores,
   and where it is weak — the per-label table the list view has no room for.

   A native <dialog> rather than a hand-built panel: showModal() traps focus, Esc
   closes, and focus returns to the row button afterwards. Writing that by hand is
   where keyboard support usually goes wrong.

   Everything rendered here comes out of a bundle, and a bundle can be imported —
   so every value goes through esc(), and numbers through the formatters below
   rather than straight into toFixed(). */
"use strict";

const fmtInt = (v) => (Number.isFinite(v) ? v.toLocaleString() : "–");
const fmtMinutes = (v) => (Number.isFinite(v) ? `${(v / 60).toFixed(1)} min` : "–");
const fmtDate = (v) => { const d = new Date(v); return isNaN(d) ? "–" : d.toLocaleString(); };

/* The searched grid is small (3–5 values). When the winner sits at either end, the
   optimum may well lie outside it — worth saying, because the fix is a wider grid,
   not more folds. */
function regularization(meta) {
  const grid = Array.isArray(meta.c_grid) ? meta.c_grid.filter(Number.isFinite) : [];
  const best = meta.best_C;
  if (!Number.isFinite(best)) return "–";
  if (!grid.length) return String(best);
  const atEdge = best <= Math.min(...grid) || best >= Math.max(...grid);
  return `${best} (searched ${grid.join(", ")})` +
    (atEdge ? " — the winner sits at the edge of the grid, so a better value may lie outside it" : "");
}

function textColumns(meta) {
  const columns = Array.isArray(meta.text_columns) ? meta.text_columns : [];
  const weights = (meta.text_column_weights && typeof meta.text_column_weights === "object")
    ? meta.text_column_weights : {};
  if (!columns.length) return "–";
  return columns.map((c) => (weights[c] > 1 ? `${c} ×${weights[c]}` : c)).join(", ");
}

/* Label + how to read one value out of the model, in the order they are shown. An
   entry returning "" is dropped, so optional fields do not leave empty rows. */
const TRAINING_ROWS = [
  ["Dataset", (m) => m.metadata.dataset],
  ["Label column", (m) => m.metadata.label_column],
  ["Text columns", (m) => textColumns(m.metadata)],
  ["Label filter", (m) => m.metadata.label_filter],
  ["Rows used", (m) => fmtInt(m.metadata.n_samples)],
  ["Min. rows per label", (m) => m.metadata.min_samples_per_label],
  ["Profile", (m) => m.metadata.profile],
  ["Regularization C", (m) => regularization(m.metadata)],
  ["Features", (m) => {
    const tfidf = m.metadata.tfidf || {};
    return Number.isFinite(tfidf.n_features)
      ? `${fmtInt(tfidf.n_features)} (${tfidf.use_char ? "word + character" : "word only"}) n-grams`
      : "";
  }],
  ["Evaluation", (m) => m.metadata.evaluation],
  ["Training time", (m) => fmtMinutes(m.metadata.training_time_seconds)],
  ["Created", (m) => fmtDate(m.metadata.created_at)],
];

const metricsOf = (m) => (m.metadata && m.metadata.metrics) || {};

const QUALITY_ROWS = [
  ["F1 macro", (m) => fmtScore(metricsOf(m).f1_macro)],
  ["F1 micro", (m) => fmtScore(metricsOf(m).f1_micro)],
  ["Decision rule", (m) => metricsOf(m).decision_rule],
  ["Labels asserted per row", (m) => {
    const metrics = metricsOf(m);
    return Number.isFinite(metrics.predicted_labels_per_row)
      ? `${metrics.predicted_labels_per_row} (the data carries ${metrics.true_labels_per_row ?? "–"})`
      : "";
  }],
];

function definitionList(model, rows) {
  const cells = rows
    .map(([label, read]) => [label, read(model)])
    .filter(([, value]) => value !== undefined && value !== null && value !== "");
  return `<dl>${cells.map(([label, value]) =>
    `<dt>${esc(label)}</dt><dd>${esc(String(value))}</dd>`).join("")}</dl>`;
}

/* ---------- the per-label table ---------- */

/* Nulls last in BOTH directions: a label nobody scored is unknown, not weak, and
   reversing the sort must not promote it to the top. */
function compareBy(key, ascending) {
  return (a, b) => {
    const x = a[key], y = b[key];
    if (x === null || x === undefined) return (y === null || y === undefined) ? 0 : 1;
    if (y === null || y === undefined) return -1;
    const order = typeof x === "string" ? String(x).localeCompare(String(y)) : x - y;
    return ascending ? order : -order;
  };
}

const LABEL_COLUMNS = [
  ["label", "Label", ""],
  ["f1", "F1", "num"],
  ["support", "Rows", "num"],
  ["threshold", "Threshold", "num"],
];

function renderLabelTable(box, labels, sort) {
  const sorted = [...labels].sort(compareBy(sort.key, sort.ascending));
  box.innerHTML = `<div class="table-wrap"><table>
    <thead><tr>${LABEL_COLUMNS.map(([key, title, cls]) => `
      <th class="${cls}" aria-sort="${sort.key === key ? (sort.ascending ? "ascending" : "descending") : "none"}">
        <button type="button" class="th-sort" data-sort="${key}">${esc(title)}</button>
      </th>`).join("")}</tr></thead>
    <tbody>${sorted.map((row) => `<tr>
      <td>${esc(row.label)}</td>
      <td class="num">${esc(fmtScore(row.f1))}</td>
      <td class="num">${esc(row.support === null || row.support === undefined ? "–" : fmtInt(row.support))}</td>
      <td class="num">${esc(row.threshold === null || row.threshold === undefined ? "argmax" : fmtScore(row.threshold))}</td>
    </tr>`).join("")}</tbody></table></div>`;
  box.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
    const key = button.dataset.sort;
    // Re-clicking the active column reverses it; a new column starts ascending,
    // which for F1 means weakest first — the order this table exists to show.
    renderLabelTable(box, labels, { key, ascending: key === sort.key ? !sort.ascending : true });
    box.querySelector(`[data-sort="${key}"]`).focus();
  }));
}

/* ---------- the panel ---------- */

function curlFor(name) {
  return `curl -X POST ${location.origin}/predict -H "X-API-Key: $KEY" ` +
    `-H "Content-Type: application/json" ` +
    `-d '{"texts": ["..."], "model_name": "${name}"}'`;
}

async function showModelDetail(name) {
  const dialog = $("#model-detail");
  dialog.innerHTML = `<div class="detail" tabindex="-1"><p class="muted">Loading …</p></div>`;
  dialog.showModal();
  const frame = dialog.querySelector(".detail");
  frame.focus();

  let model, labels;
  try {
    [model, labels] = await Promise.all([
      Api.get(`/models/${encodeURIComponent(name)}`),
      Api.get(`/models/${encodeURIComponent(name)}/labels`),
    ]);
  } catch (err) {
    frame.innerHTML = `<p class="error">${esc(err.message)}</p>
      <button type="button" class="ghost" data-close>Close</button>`;
    frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
    return;
  }

  const vocabulary = model.label_vocabulary;
  frame.innerHTML = `
    <div class="detail-head">
      <h2 id="model-detail-title">${esc(name)}</h2>
      <button type="button" class="ghost small" data-close aria-label="Close details">Close</button>
    </div>
    <p class="muted">${esc(model.task_type)} · ${esc(String((model.classes || []).length))} labels${
      vocabulary ? ` · from <code>${esc(vocabulary)}</code>` : ""}</p>
    ${isServable(model) ? "" : `<p class="error">Bundle format ${esc(String(model.format_version ?? 1))}:
      this model cannot be loaded for classification any more. Download it, or retrain it.</p>`}

    <h3>How it was trained</h3>
    <div class="train-status">${definitionList(model, TRAINING_ROWS)}</div>

    <h3>How well it works</h3>
    <div class="train-status">${definitionList(model, QUALITY_ROWS)}</div>

    <h3>Scored against a dataset</h3>
    <div id="detail-evaluations"></div>

    <h3>Per label</h3>
    <p class="muted">Weakest first. A high confidence on a weak label is worth less than the
      same number on a strong one. "Rows" says why a score is low — too few examples is a
      different problem from a hard distinction; it is blank for models trained before the
      count was recorded. "argmax" means this model picks its single best label and reads
      no threshold.</p>
    <div id="detail-labels"></div>

    <div class="detail-actions">
      <button type="button" class="small" data-act="download">Download bundle</button>
      <button type="button" class="small" data-act="curl">Copy curl</button>
      <button type="button" class="small" data-act="evaluate">Evaluate on…</button>
      <button type="button" class="small" data-act="share">Share link</button>
      <button type="button" class="small" data-act="info">Documentation</button>
      <button type="button" class="small danger" data-act="delete">Delete</button>
    </div>`;

  frame.querySelector("#detail-evaluations").innerHTML = evaluationsSection(model);
  renderLabelTable(frame.querySelector("#detail-labels"), labels, { key: "f1", ascending: true });

  // The actions that render into the page behind the modal close it first —
  // otherwise their output would be hidden under the dialog's top layer.
  const actions = {
    download: () => Api.download(`/models/${encodeURIComponent(name)}/export`, `${name}.zip`)
      .catch((err) => toast(err.message)),
    curl: async () => { await navigator.clipboard.writeText(curlFor(name)); toast("curl command copied."); },
    // Stays inside the dialog: the form belongs to THIS model, and closing the panel
    // to fill it in would lose the numbers it is meant to be compared against.
    evaluate: () => openEvaluateForm(name, frame),
    share: () => { dialog.close(); shareResource("models", name, "#models-share"); },
    info: () => { dialog.close(); editModelInfo(name); },
    delete: async () => {
      if (!confirm(`Delete model "${name}"? This cannot be undone.`)) return;
      try {
        await Api.del(`/models/${encodeURIComponent(name)}`);
        dialog.close();
        toast("Model deleted.");
        loadModels();
      } catch (err) { toast(err.message); }
    },
  };
  frame.querySelectorAll("[data-act]").forEach((button) =>
    button.addEventListener("click", () => actions[button.dataset.act]()));
  frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
  frame.focus();
}

/* A click on the backdrop reports the dialog itself as the target; anything inside
   reports that element. Closing on the backdrop is what people expect from a modal. */
document.addEventListener("DOMContentLoaded", () => {
  const dialog = $("#model-detail");
  dialog.addEventListener("click", (ev) => { if (ev.target === dialog) dialog.close(); });
});
