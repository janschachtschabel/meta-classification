/* "That answer is wrong, here is the right one" — the half of the loop a person does.

   Its own file: reading an answer and correcting it are different jobs, and this one
   carries a form and a request of its own.

   The label picker is a plain <select multiple> with the URIs as values and the display
   names as text. A pill picker would read nicer but works on strings, which would mean
   mapping a display name back to a URI — and two labels can share a name. The native
   control carries both without a lookup that can be wrong. */
"use strict";

/* A correction is certain to be about the classified text only while the field still holds
   it: the labels on screen belong to that text, and a user who edited it may mean the new
   one. Reading the field at save time recorded the new text beside the old prediction, a row
   the next training reads as a true pair (audit 2026-09-30, U03). Neither guess is saved. */
const textChangedSince = (classified) => $("#query-text").value !== classified;

async function openCorrection(modelName, predicted, text, card, button) {
  card.querySelector(".correction")?.remove();
  if (textChangedSince(text)) {
    card.insertAdjacentHTML("beforeend",
      `<div class="correction"><p class="error" role="alert">${t("feedback.textChanged")}</p></div>`);
    return;
  }
  const idle = busy(button);
  try {
    const labels = await Api.get(`/models/${encodeURIComponent(modelName)}/labels`);
    card.insertAdjacentHTML("beforeend", correctionHtml(modelName, predicted, labels));
  } catch (err) {
    card.insertAdjacentHTML("beforeend",
      `<div class="correction"><p class="error" role="alert">${esc(err.message)}</p></div>`);
    return;
  } finally { idle(); }

  const box = card.querySelector(".correction");
  box.querySelector("[data-send]").addEventListener("click", () =>
    sendCorrection(modelName, predicted, box, text));
  box.querySelector("[data-cancel]").addEventListener("click", closer(() => box.remove()));
  box.querySelector("select").focus();
}

function correctionHtml(modelName, predicted, labels) {
  const sorted = [...labels].sort((a, b) => a.label.localeCompare(b.label, I18n.locale()));
  const said = predicted.length
    ? predicted.map((p) => esc(p.label)).join(", ")
    : t("feedback.nothing");
  return `<div class="correction">
    <p>${t("feedback.modelSaid", { labels: said })}</p>
    <label for="fb-labels">${t("feedback.labelsLabel")}</label>
    <select id="fb-labels" multiple size="8">${sorted.map((l) =>
      `<option value="${esc(l.uri)}">${esc(l.label)}</option>`).join("")}</select>
    <p class="muted">${t("feedback.note")}</p>
    <button type="button" class="small" data-send>${t("feedback.save")}</button>
    <button type="button" class="small ghost" data-cancel>${t("common.cancel")}</button>
    <p class="fb-message" role="alert"></p>
  </div>`;
}

async function sendCorrection(modelName, predicted, box, text) {
  const corrected = [...box.querySelectorAll("#fb-labels option")]
    .filter((o) => o.selected).map((o) => o.value);
  const message = box.querySelector(".fb-message");
  if (textChangedSince(text)) {   // edited while the form was open
    message.className = "fb-message error";
    message.textContent = t("feedback.textChanged");
    return;
  }
  const send = box.querySelector("[data-send]");
  const idle = busy(send);
  try {
    const answer = await Api.post("/feedback", {
      text,
      model_name: modelName,
      predicted: predicted.map((p) => p.uri),
      corrected,
      source: "ui",
    });
    const collected = t("feedback.corrections", { count: answer.collected });
    message.className = "fb-message ok";
    message.textContent = corrected.length
      ? t("feedback.saved", { collected })
      : t("feedback.savedNone", { collected });
    box.querySelector("select").disabled = true;
    send.textContent = t("feedback.savedButton");
    // Save stays disabled for good, so the focus would end on <body> (U09): what is left to
    // do with the form is to close it.
    const close = box.querySelector("[data-cancel]");
    close.textContent = t("common.close");
    close.focus();
  } catch (err) {
    message.className = "fb-message error";
    message.textContent = err.message;
    idle();
  }
}

/* Wire the "Correct" buttons of a freshly rendered single-text result; `text` is the one
   that was classified. */
function bindCorrectionButtons(root, byModel, text) {
  root.querySelectorAll("[data-correct]").forEach((button) => {
    const name = button.dataset.correct;
    button.addEventListener("click", () =>
      openCorrection(name, byModel[name] || [], text, button.closest(".card"), button));
  });
}
