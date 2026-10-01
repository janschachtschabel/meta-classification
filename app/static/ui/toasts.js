/* Telling the user what happened: confirmations and failures, and where they are shown --
   the page's two stacks, or the open dialog's own pair. Split from app.js, the shell: these
   change when what a message is or where it goes changes, that one when the views do. */
"use strict";

/* Two regions, not one, and the difference is not cosmetic.

   `role="status"` is POLITE: a screen reader finishes what it is saying first and may drop
   the message entirely. That is right for "model deleted" and wrong for the failure that
   explains why nothing happened — and seven call sites were routing errors through it.

   A message is appended rather than assigned, because assigning erased an unread message
   when a second one arrived (two failed deletes in a row showed one). Every message carries
   a dismiss button, and an error is not scheduled to disappear at all: a 4-second auto-hide
   on text the user has to act on is a limit on reading it (SC 2.2.1), and a long German
   error does not fit in four seconds. Confirmations still fade, but hovering or focusing
   the stack holds them. */
const TOAST_HIDE_MS = 6000;

function pushToast(regionSel, msg, hideAfter) {
  const region = toastRegion(regionSel);
  const item = document.createElement("div");
  item.className = "toast-item";
  item.innerHTML = `<span>${esc(msg)}</span>` +
    `<button type="button" class="toast-x" data-toast-dismiss ` +
    `aria-label="${esc(t("common.dismiss"))}">&times;</button>`;
  item.querySelector("[data-toast-dismiss]").addEventListener("click", () => item.remove());
  region.appendChild(item);
  if (hideAfter) {
    // Cleared on hover/focus so the stack can be read at the reader's pace, and re-armed
    // on leave — `:hover` alone would only stop the CSS, not the timer.
    let timer = setTimeout(() => item.remove(), hideAfter);
    const hold = () => clearTimeout(timer);
    const resume = () => { timer = setTimeout(() => item.remove(), hideAfter); };
    item.addEventListener("mouseenter", hold);
    item.addEventListener("mouseleave", resume);
    item.addEventListener("focusin", hold);
    item.addEventListener("focusout", resume);
  }
  return item;
}

/* An open modal dialog makes everything outside it inert -- out of the accessibility tree --
   and paints it under the backdrop, both stacks above included: "copied", or why a delete
   failed, was neither heard nor properly seen while a panel was open (audit 2026-09-30, U09).
   So a dialog opens through `openModal`, which gives it a pair of its own, inserted empty
   BEFORE it opens -- a live region has to exist before its message does to be announced --
   and a message goes to the open dialog's pair when there is one. */
function toastRegion(regionSel) {
  const open = document.querySelector("dialog[open]");
  return (open && open.querySelector(`[data-toast="${regionSel.slice(1)}"]`)) || $(regionSel);
}

function openModal(dialog) {
  if (!dialog.querySelector("[data-toast]")) {
    dialog.insertAdjacentHTML("beforeend",
      '<div class="toast-stack" data-toast="toast" role="status" aria-live="polite"></div>' +
      '<div class="toast-stack toast-alert" data-toast="toast-alert" role="alert" ' +
      'aria-live="assertive"></div>');
  }
  dialog.showModal();
}

/* A confirmation: something asked for happened. Polite, and fades. */
function toast(msg) { return pushToast("#toast", msg, TOAST_HIDE_MS); }

/* A failure: assertive, and stays until dismissed. */
function toastError(err) {
  return pushToast("#toast-alert", (err && err.message) || String(err), 0);
}
