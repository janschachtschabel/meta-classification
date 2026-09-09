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
    <span class="val">${sign}${Math.abs(entry.impact).toFixed(4)}</span></span>`;
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
    <summary>All ${rows.length} labels, strongest first</summary>
    <div class="table-wrap"><table>
      <thead><tr><th>Label</th><th class="num">Confidence</th><th class="num">diff</th>
        <th class="num">F1</th></tr></thead>
      <tbody>${rows.map((r) => `<tr>
        <td>${esc(r.label)}</td>
        <td class="num">${r.confidence.toFixed(3)}</td>
        <td class="num">${r.baseline_diff >= 0 ? "+" : "−"}${Math.abs(r.baseline_diff).toFixed(3)}</td>
        <td class="num">${r.label_f1 == null ? "–" : r.label_f1.toFixed(3)}</td>
      </tr>`).join("")}</tbody></table></div>
    <p class="muted"><strong>diff</strong> is the confidence minus what this model answers
       for an empty text — how much comes from your text rather than from the label simply
       being common. <strong>F1</strong> is how well the model does on that label at all.</p>
  </details>`;
}

function explanationHtml(body) {
  const sections = Object.entries(body.word_importance || {})
    .map(([uri, entry]) => wordSection(uri, entry)).join("");
  const intro = sections
    ? `<p class="muted">How far the confidence falls when a word is removed. On a confident
        answer the numbers are tiny — a model at 0.999 barely moves for any single word —
        so the bars are scaled per label and it is the <em>order</em> that carries the
        meaning. A <strong>−</strong> marks a word that argues <em>against</em> the label:
        removing it would raise the confidence. Each occurrence of a word is scored on its
        own.</p>`
    : `<p class="muted">No per-word attribution for this text: leaving a word out needs at
        least two words, or the model asserted no label to attribute.</p>`;
  return `<div class="explain">${intro}${sections}${allScoresTable(body.all_scores || {})}</div>`;
}

async function explainAnswer(modelName, text, card, button) {
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Explaining …";
  card.querySelector(".explain")?.remove();
  try {
    const body = await Api.post("/predict/explain",
                                { text, model_name: modelName, top_n_words: EXPLAIN_WORDS });
    card.insertAdjacentHTML("beforeend", explanationHtml(body));
    applyBarWidths(card);
    $("#query-status").textContent = `Explanation for ${modelName} ready.`;
  } catch (err) {
    // Inside the card, so one model failing leaves the other answers standing.
    card.insertAdjacentHTML("beforeend",
      `<div class="explain"><p class="error" role="alert">${esc(err.message)}</p></div>`);
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

/* Wire the "Why?" buttons of a freshly rendered single-text result. */
function bindExplainButtons(root, text) {
  root.querySelectorAll("[data-explain]").forEach((button) =>
    button.addEventListener("click", () =>
      explainAnswer(button.dataset.explain, text, button.closest(".card"), button)));
}
