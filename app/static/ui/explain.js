/* Why did the model say that? — the panel behind the "Why?" button on a single-text
   result. Its own file: running a query and interrogating an answer are different jobs,
   and this one changes when the explanation changes, not when the query form does.

   One call explains one model's answer for the whole text, which is what
   /predict/explain does: it attributes the top predicted labels to the words whose
   removal moves the confidence most (leave-one-out). */
"use strict";

// The endpoint's own default is 5. Eight fits one wrapped row and still shows the
// tail where the impacts turn negative, which is where the interesting part is.
const EXPLAIN_WORDS = 8;

function impactChip(entry, scale) {
  const against = entry.impact < 0;
  const width = scale > 0 ? Math.round((Math.abs(entry.impact) / scale) * 100) : 0;
  const sign = against ? "−" : "+";
  return `<span class="chip-word${against ? " against" : ""}">
    <span class="w">${esc(entry.word)}</span>
    <span class="chip-bar"><span data-width="${width}"></span></span>
    <span class="val">${sign}${fmtFixed(Math.abs(entry.impact), 4)}</span></span>`;
}

function wordSection(uri, entry) {
  const words = entry.top_words || [];
  if (!words.length) return "";
  // Per label, not across labels: on a confident answer every impact is tiny (0.002
  // against a 0.9997 confidence), while a contested label swings by half a point. A
  // shared scale would flatten the first case to nothing. The bars say "which word
  // mattered most for THIS label"; the numbers say by how much.
  const scale = Math.max(...words.map((w) => Math.abs(w.impact)));
  return `<h4>${esc(entry.label)}</h4>
    <div class="chips">${words.map((w) => impactChip(w, scale)).join("")}</div>`;
}

function allScoresTable(scores) {
  const rows = Object.entries(scores)
    .map(([uri, s]) => ({ uri, ...s }))
    .sort((a, b) => b.confidence - a.confidence);
  return `<details class="help">
    <summary>${t("explain.allLabels", { count: rows.length })}</summary>
    <div class="table-wrap"><table>
      <thead><tr><th>${t("explain.table.label")}</th><th class="num">${t("explain.table.confidence")}</th>
        <th class="num">${t("explain.table.diff")}</th>
        <th class="num">${t("explain.table.f1")}</th></tr></thead>
      <tbody>${rows.map((r) => `<tr>
        <td>${esc(r.label)}</td>
        <td class="num">${fmtFixed(r.confidence, 3)}</td>
        <td class="num">${r.baseline_diff >= 0 ? "+" : "−"}${fmtFixed(Math.abs(r.baseline_diff), 3)}</td>
        <td class="num">${r.label_f1 == null ? "–" : fmtFixed(r.label_f1, 3)}</td>
      </tr>`).join("")}</tbody></table></div>
    <p class="muted">${t("explain.tableNote")}</p>
  </details>`;
}

function explanationHtml(body) {
  const sections = Object.entries(body.word_importance || {})
    .map(([uri, entry]) => wordSection(uri, entry)).join("");
  const intro = `<p class="muted">${sections ? t("explain.intro") : t("explain.noAttribution")}</p>`;
  return `<div class="explain">${intro}${sections}${allScoresTable(body.all_scores || {})}</div>`;
}

async function explainAnswer(modelName, text, card, button) {
  button.disabled = true;
  button.textContent = t("explain.button.busy");
  card.querySelector(".explain")?.remove();
  try {
    const body = await Api.post("/predict/explain",
                                { text, model_name: modelName, top_n_words: EXPLAIN_WORDS });
    card.insertAdjacentHTML("beforeend", explanationHtml(body));
    applyBarWidths(card);
    $("#query-status").textContent = t("explain.ready", { name: modelName });
  } catch (err) {
    // Inside the card, so one model failing leaves the other answers standing.
    card.insertAdjacentHTML("beforeend",
      `<div class="explain"><p class="error" role="alert">${esc(err.message)}</p></div>`);
  } finally {
    button.disabled = false;
    // From the key, not from a copy taken before the request: the language may
    // have been switched while it was in flight.
    button.textContent = t("query.explainButton");
  }
}

/* Wire the "Why?" buttons of a freshly rendered single-text result. */
function bindExplainButtons(root, text) {
  root.querySelectorAll("[data-explain]").forEach((button) =>
    button.addEventListener("click", () =>
      explainAnswer(button.dataset.explain, text, button.closest(".card"), button)));
}
