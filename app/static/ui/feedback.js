/* "That answer is wrong, here is the right one" — the half of the loop a person does.

   Its own file: reading an answer and correcting it are different jobs, and this one
   carries a form and a request of its own.

   The label picker is a plain <select multiple> with the URIs as values and the display
   names as text. A pill picker would read nicer but works on strings, which would mean
   mapping a display name back to a URI — and two labels can share a name. The native
   control carries both without a lookup that can be wrong. */
"use strict";

async function openCorrection(modelName, predicted, card, button) {
  card.querySelector(".correction")?.remove();
  button.disabled = true;
  try {
    const labels = await Api.get(`/models/${encodeURIComponent(modelName)}/labels`);
    card.insertAdjacentHTML("beforeend", correctionHtml(modelName, predicted, labels));
  } catch (err) {
    card.insertAdjacentHTML("beforeend",
      `<div class="correction"><p class="error" role="alert">${esc(err.message)}</p></div>`);
    return;
  } finally { button.disabled = false; }

  const box = card.querySelector(".correction");
  box.querySelector("[data-send]").addEventListener("click", () =>
    sendCorrection(modelName, predicted, box));
  box.querySelector("[data-cancel]").addEventListener("click", () => box.remove());
  box.querySelector("select").focus();
}

function correctionHtml(modelName, predicted, labels) {
  const sorted = [...labels].sort((a, b) => a.label.localeCompare(b.label));
  const said = predicted.length
    ? predicted.map((p) => esc(p.label)).join(", ")
    : "nothing";
  return `<div class="correction">
    <p>The model said <strong>${said}</strong>. What should it have said?</p>
    <label for="fb-labels">Correct labels for this text</label>
    <select id="fb-labels" multiple size="8">${sorted.map((l) =>
      `<option value="${esc(l.uri)}">${esc(l.label)}</option>`).join("")}</select>
    <p class="muted">Ctrl-click for several. <strong>Select nothing</strong> to say that
       none of them apply — that is recorded too, and left out of the training export,
       because a row without a label is not something a run can learn from.</p>
    <button type="button" class="small" data-send>Save correction</button>
    <button type="button" class="small ghost" data-cancel>Cancel</button>
    <p class="fb-message" role="alert"></p>
  </div>`;
}

async function sendCorrection(modelName, predicted, box) {
  const corrected = [...box.querySelectorAll("#fb-labels option")]
    .filter((o) => o.selected).map((o) => o.value);
  const message = box.querySelector(".fb-message");
  const send = box.querySelector("[data-send]");
  send.disabled = true;
  try {
    const answer = await Api.post("/feedback", {
      text: $("#query-text").value,
      model_name: modelName,
      predicted: predicted.map((p) => p.uri),
      corrected,
      source: "ui",
    });
    message.className = "fb-message ok";
    message.textContent = corrected.length
      ? `Saved. ${plural(answer.collected, "correction")} collected so far.`
      : `Saved as "none of these apply". ${plural(answer.collected, "correction")} collected so far.`;
    box.querySelector("select").disabled = true;
    send.textContent = "Saved";
  } catch (err) {
    message.className = "fb-message error";
    message.textContent = err.message;
    send.disabled = false;
  }
}

/* Wire the "Correct" buttons of a freshly rendered single-text result. */
function bindCorrectionButtons(root, byModel) {
  root.querySelectorAll("[data-correct]").forEach((button) => {
    const name = button.dataset.correct;
    button.addEventListener("click", () =>
      openCorrection(name, byModel[name] || [], button.closest(".card"), button));
  });
}
