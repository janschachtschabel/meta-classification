/* Everything one model records about itself: how it was trained, how well it scores,
   and where it is weak — the per-label table the list view has no room for.

   A native <dialog> rather than a hand-built panel: showModal() traps focus, Esc
   closes, and focus returns to the row button afterwards. Writing that by hand is
   where keyboard support usually goes wrong.

   Everything rendered here comes out of a bundle, and a bundle can be imported —
   so every value goes through esc(), and numbers through the formatters below
   rather than straight into toFixed(). */
"use strict";

// Locale-aware on purpose: 156373 reads as 156.373 to a German reader and 156,373
// to an English one, and the UI language — not the browser's — is what they picked.
const fmtInt = (v) => (Number.isFinite(v) ? v.toLocaleString(I18n.locale()) : "–");
// Rounded before formatting, not after: Intl would otherwise render 40.1666 in full,
// and one decimal is all this number ever meant.
const fmtMinutes = (v) =>
  (Number.isFinite(v) ? t("common.minutes", { count: Number((v / 60).toFixed(1)) }) : "–");
const fmtDate = (v) => {
  const d = new Date(v);
  return isNaN(d) ? "–" : d.toLocaleString(I18n.locale());
};

/* The searched grid is small (3–5 values). When the winner sits at either end, the
   optimum may well lie outside it — worth saying, because the fix is a wider grid,
   not more folds. */
function regularization(meta) {
  const grid = Array.isArray(meta.c_grid) ? meta.c_grid.filter(Number.isFinite) : [];
  const best = meta.best_C;
  if (!Number.isFinite(best)) return "–";
  if (!grid.length) return String(best);
  const atEdge = best <= Math.min(...grid) || best >= Math.max(...grid);
  // As a string, so Intl does not format `best` while the grid beside it stays
  // raw — that printed "0,5 (durchsucht: 0.5, 1.5)" and truncated a small C to 0.
  return t("modelDetail.cSearched", { best: String(best), grid: grid.join(", ") }) +
    (atEdge ? t("modelDetail.cAtEdge") : "");
}

function textColumns(meta) {
  const columns = Array.isArray(meta.text_columns) ? meta.text_columns : [];
  const weights = (meta.text_column_weights && typeof meta.text_column_weights === "object")
    ? meta.text_column_weights : {};
  if (!columns.length) return "–";
  return columns.map((c) => (weights[c] > 1 ? `${c} ×${weights[c]}` : c)).join(", ");
}

/* What an LLM wrote or touched (the bundle's `synthetic_data` block, absent for a
   dataset without data-prep's marks). A bundle can be imported, so every number is
   checked before it is used and formatted before it is shown. */
const synthOf = (meta) =>
  (meta.synthetic_data && typeof meta.synthetic_data === "object" ? meta.synthetic_data : null);
const countOf = (value) => (Number.isFinite(value) ? value : 0);

/* What was left out (exclude), then what the marked rows that stayed did: trained and
   never validated -- or, after a fallback, validated too. Checked in that order, so the
   line never claims "never validated" beside the warning that says otherwise. */
function aiData(meta) {
  const s = synthOf(meta);
  if (!s) return "";
  const trained = countOf(s.train_only_rows);
  const parts = [];
  if (s.mode === "exclude") {
    parts.push(t("modelDetail.aiData.excluded", { count: countOf(s.excluded_generated_rows) }));
  }
  if (s.validated_on === "all_rows") {
    parts.push(t("modelDetail.aiData.validated", { count: trained }));
  } else if (s.mode === "exclude") {
    if (trained) parts.push(t("modelDetail.aiData.trainOnly", { count: trained }));
  } else {
    parts.push(t("modelDetail.aiData.train", {
      count: trained, generated: fmtInt(countOf(s.generated_rows)),
      examples: fmtInt(countOf(s.example_rows)), enriched: fmtInt(countOf(s.enriched_rows)),
    }));
  }
  return parts.join(" · ");
}

function scoredOn(meta) {
  const s = synthOf(meta);
  if (!s || !Number.isFinite(s.scored_rows)) return "";
  return s.validated_on === "all_rows"
    ? t("modelDetail.scoredOn.all", { count: s.scored_rows })
    : t("modelDetail.scoredOn.real", { count: s.scored_rows });
}

/* The two notices that must not hide in a table row: numbers that include AI-marked
   rows, and labels no real row could score. Text with an icon, not colour alone. */
function aiWarning(meta) {
  const s = synthOf(meta);
  return s && s.validated_on === "all_rows"
    ? `<p class="error">${t("modelDetail.aiData.fallback")}</p>` : "";
}

/* Thin labels reached the training minimum only through AI-marked rows. Cut at their
   own threshold, that threshold rests on a handful of real rows -- a warning, with
   what to do about it; cut at the global one, a note on what was decided. Silent for
   binary/multiclass, where serving picks the best label and reads no threshold. The
   bundle lists URIs; the names come from the label table this panel already loaded. */
function thinNote(model, labels) {
  if (["binary", "multiclass"].includes(model.task_type)) return "";
  const meta = model.metadata || {};
  const s = synthOf(meta);
  const thin = s && Array.isArray(s.thin_labels) ? s.thin_labels : [];
  if (!thin.length) return "";
  const nameOf = new Map((labels || []).map((row) => [row.uri, row.label]));
  const shown = thin.slice(0, 5).map((uri) => esc(String(nameOf.get(uri) || uri))).join(", ");
  const listed = shown + (thin.length > 5
    ? t("modelDetail.thin.more", { count: thin.length - 5 }) : "");
  return s.thin_label_threshold === "global"
    ? `<p class="muted">${t("modelDetail.thin.global", { count: thin.length, labels: listed })}</p>`
    : `<p class="error">${t("modelDetail.thin.own", { count: thin.length, labels: listed })}</p>`;
}

function notValidatedNote(meta) {
  const s = synthOf(meta);
  const labels = s && Array.isArray(s.labels_not_validated) ? s.labels_not_validated : [];
  return labels.length
    ? `<p class="muted">${t("modelDetail.aiData.notValidated", { count: labels.length })}</p>` : "";
}

/* Label key + how to read one value out of the model, in the order they are shown.
   An entry returning "" is dropped, so optional fields do not leave empty rows. */
const TRAINING_ROWS = [
  ["modelDetail.row.dataset", (m) => m.metadata.dataset],
  ["modelDetail.row.labelColumn", (m) => m.metadata.label_column],
  ["modelDetail.row.textColumns", (m) => textColumns(m.metadata)],
  ["modelDetail.row.labelFilter", (m) => m.metadata.label_filter],
  ["modelDetail.row.rowsUsed", (m) => fmtInt(m.metadata.n_samples)],
  ["modelDetail.row.aiData", (m) => aiData(m.metadata)],
  ["modelDetail.row.minRowsPerLabel", (m) => m.metadata.min_samples_per_label],
  ["modelDetail.row.profile", (m) => m.metadata.profile],
  ["modelDetail.row.regularization", (m) => regularization(m.metadata)],
  ["modelDetail.row.features", (m) => {
    const tfidf = m.metadata.tfidf || {};
    return Number.isFinite(tfidf.n_features)
      ? t("modelDetail.features", {
          count: tfidf.n_features,
          kind: tfidf.use_char ? t("modelDetail.features.wordChar") : t("modelDetail.features.word"),
        })
      : "";
  }],
  ["modelDetail.row.evaluation", (m) => m.metadata.evaluation],
  ["modelDetail.row.trainingTime", (m) => fmtMinutes(m.metadata.training_time_seconds)],
  ["modelDetail.row.created", (m) => fmtDate(m.metadata.created_at)],
];

const metricsOf = (m) => (m.metadata && m.metadata.metrics) || {};

const QUALITY_ROWS = [
  ["modelDetail.row.scoredOn", (m) => scoredOn(m.metadata)],
  ["modelDetail.row.f1Macro", (m) => fmtScore(metricsOf(m).f1_macro)],
  ["modelDetail.row.f1Micro", (m) => fmtScore(metricsOf(m).f1_micro)],
  ["modelDetail.row.decisionRule", (m) => metricsOf(m).decision_rule],
  ["modelDetail.row.labelsPerRow", (m) => {
    const metrics = metricsOf(m);
    return Number.isFinite(metrics.predicted_labels_per_row)
      ? t("modelDetail.labelsPerRow", {
          predicted: metrics.predicted_labels_per_row,
          actual: metrics.true_labels_per_row ?? "–",
        })
      : "";
  }],
];

function definitionList(model, rows) {
  const cells = rows
    .map(([labelKey, read]) => [t(labelKey), read(model)])
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
    const order = typeof x === "string"
      ? String(x).localeCompare(String(y), I18n.locale()) : x - y;
    return ascending ? order : -order;
  };
}

const LABEL_COLUMNS = [
  ["label", "modelDetail.labels.label", ""],
  ["f1", "modelDetail.labels.f1", "num"],
  ["support", "modelDetail.labels.rows", "num"],
  ["threshold", "modelDetail.labels.threshold", "num"],
];

function renderLabelTable(box, labels, sort) {
  const sorted = [...labels].sort(compareBy(sort.key, sort.ascending));
  box.innerHTML = `<div class="table-wrap"><table>
    <thead><tr>${LABEL_COLUMNS.map(([key, titleKey, cls]) => `
      <th class="${cls}" aria-sort="${sort.key === key ? (sort.ascending ? "ascending" : "descending") : "none"}">
        <button type="button" class="th-sort" data-sort="${key}">${esc(t(titleKey))}</button>
      </th>`).join("")}</tr></thead>
    <tbody>${sorted.map((row) => `<tr>
      <td>${esc(row.label)}</td>
      <td class="num">${esc(fmtScore(row.f1))}</td>
      <td class="num">${esc(row.support === null || row.support === undefined ? "–" : fmtInt(row.support))}</td>
      <td class="num">${esc(row.threshold === null || row.threshold === undefined
        ? t("modelDetail.argmax") : fmtScore(row.threshold))}</td>
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
  dialog.innerHTML = `<div class="detail" tabindex="-1"><p class="muted">${t("common.loading")}</p></div>`;
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
      <button type="button" class="ghost" data-close>${t("common.close")}</button>`;
    frame.querySelector("[data-close]").addEventListener("click", () => dialog.close());
    return;
  }

  const vocabulary = model.label_vocabulary;
  frame.innerHTML = `
    <div class="detail-head">
      <h2 id="model-detail-title">${esc(name)}</h2>
      <button type="button" class="ghost small" data-close
              aria-label="${esc(t("common.closeDetails"))}">${t("common.close")}</button>
    </div>
    <p class="muted">${esc(model.task_type)} · ${t("modelDetail.labelCount", {
      count: (model.classes || []).length })}${
      vocabulary ? t("modelDetail.fromVocabulary", { vocabulary: esc(vocabulary) }) : ""}</p>
    ${isServable(model) ? "" : `<p class="error">${t("modelDetail.staleBundle", {
      version: esc(String(model.format_version ?? 1)) })}</p>`}

    <h3>${t("modelDetail.howTrained")}</h3>
    <div class="train-status">${definitionList(model, TRAINING_ROWS)}</div>

    <h3>${t("modelDetail.howWell")}</h3>
    ${aiWarning(model.metadata)}
    <div class="train-status">${definitionList(model, QUALITY_ROWS)}</div>

    <h3>${t("modelDetail.scoredAgainst")}</h3>
    <div id="detail-evaluations"></div>

    <h3>${t("modelDetail.perLabel")}</h3>
    <p class="muted">${t("modelDetail.perLabelNote")}</p>
    ${notValidatedNote(model.metadata)}
    ${thinNote(model, labels)}
    <div id="detail-labels"></div>

    <div class="detail-actions">
      <button type="button" class="small" data-act="download">${t("modelDetail.action.download")}</button>
      <button type="button" class="small" data-act="curl">${t("modelDetail.action.curl")}</button>
      <button type="button" class="small" data-act="evaluate">${t("modelDetail.action.evaluate")}</button>
      <button type="button" class="small" data-act="share">${t("common.shareLink")}</button>
      <button type="button" class="small" data-act="info">${t("modelDetail.action.documentation")}</button>
      <button type="button" class="small danger" data-act="delete">${t("common.delete")}</button>
    </div>`;

  frame.querySelector("#detail-evaluations").innerHTML = evaluationsSection(model);
  renderLabelTable(frame.querySelector("#detail-labels"), labels, { key: "f1", ascending: true });

  // The actions that render into the page behind the modal close it first —
  // otherwise their output would be hidden under the dialog's top layer.
  const actions = {
    download: () => Api.download(`/models/${encodeURIComponent(name)}/export`, `${name}.zip`)
      .catch((err) => toast(err.message)),
    curl: async () => { await navigator.clipboard.writeText(curlFor(name)); toast(t("modelDetail.curlCopied")); },
    // Stays inside the dialog: the form belongs to THIS model, and closing the panel
    // to fill it in would lose the numbers it is meant to be compared against.
    evaluate: () => openEvaluateForm(name, frame),
    share: () => { dialog.close(); shareResource("models", name, "#models-share"); },
    info: () => { dialog.close(); editModelInfo(name); },
    delete: async () => {
      if (!confirm(t("models.deleteConfirm", { name }))) return;
      try {
        await Api.del(`/models/${encodeURIComponent(name)}`);
        dialog.close();
        toast(t("models.deleted"));
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
